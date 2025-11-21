# c_realtime_inference_lstm.py
import os, json, argparse, time
from collections import deque
import numpy as np
import cv2, torch, mediapipe as mp
import torch.nn as nn

# ============== 모델 (train_lstm.py와 동일) ==============
class AttPool(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w = nn.Linear(d, 1)
    def forward(self, h):                      # h:(B,T,D)
        a = self.w(h).squeeze(-1)             # (B,T)
        w = torch.softmax(a, dim=1).unsqueeze(-1)
        return (h*w).sum(1)                    # (B,D)

class LSTMAnom(nn.Module):
    def __init__(self, feat_dim, hidden=128, layers=2, num_out=1, bidir=True):
        super().__init__()
        self.pre = nn.Sequential(nn.Conv1d(feat_dim, feat_dim, 3, padding=1), nn.ReLU())
        self.lstm = nn.LSTM(feat_dim, hidden, num_layers=layers, batch_first=True,
                            bidirectional=bidir, dropout=0.1)
        d = hidden*(2 if bidir else 1)
        self.pool = AttPool(d)
        self.head = nn.Linear(d, num_out)
    def forward(self, x):                      # x:(B,T,F)
        z = self.pre(x.transpose(1,2)).transpose(1,2)
        h,_ = self.lstm(z)
        g = self.pool(h)
        return self.head(g)                    # 로짓 반환(== 학습과 동일)

# ============== 피처 (make_dataset_all.py와 동일) ==============
def _ffill_bfill(arr):
    T,D = arr.shape
    out = arr.copy()
    last = np.zeros(D,np.float32); has = np.zeros(D,bool)
    for t in range(T):
        nz=~np.isnan(out[t]); last[nz]=out[t,nz]; has|=nz
        miss=np.isnan(out[t])&has; out[t,miss]=last[miss]
    last[:]=0; has[:]=False
    for t in range(T-1,-1,-1):
        nz=~np.isnan(out[t]); last[nz]=out[t,nz]; has|=nz
        miss=np.isnan(out[t])&has; out[t,miss]=last[miss]
    return np.nan_to_num(out,nan=0.0,posinf=0.0,neginf=0.0)

def features_from_buf(buf):                    # buf: list of (33,4)
    k = np.stack(buf,0)                        # (T,33,4)
    T = k.shape[0]
    xy  = k[:,:,:2].reshape(T,-1)              # (T,66)
    vis = k[:,:,3:4].reshape(T,-1)             # (T,33)
    xy  = _ffill_bfill(xy).reshape(T,33,2)
    vis = _ffill_bfill(vis).reshape(T,33,1)

    hip = np.mean(xy[:,[23,24],:],axis=1)
    sh  = np.mean(xy[:,[11,12],:],axis=1)
    sc  = np.linalg.norm(sh-hip,axis=1,keepdims=True); sc[sc<1e-3]=1.0

    xy_n = (xy-hip[:,None,:])/sc[:,None,:]
    vel  = np.diff(xy_n,axis=0,prepend=xy_n[:1])

    def ang(a,b,c):
        v1=a-b; v2=c-b
        n1=np.linalg.norm(v1,axis=-1); n2=np.linalg.norm(v2,axis=-1)
        n1[n1==0]=1e-6; n2[n2==0]=1e-6
        cos=(v1*v2).sum(-1)/(n1*n2)
        return np.arccos(np.clip(cos,-1,1))
    def pick(i): return xy_n[:,i,:]
    angs=np.stack([
        ang(pick(11),pick(13),pick(15)),
        ang(pick(12),pick(14),pick(16)),
        ang(pick(23),pick(25),pick(27)),
        ang(pick(24),pick(26),pick(28)),
    ],axis=1)                                  # (T,4)

    feat=np.concatenate([xy_n.reshape(T,-1),   # 66
                         vel.reshape(T,-1),    # 66
                         angs,                 # 4
                         vis.reshape(T,-1)],1) # 33
    feat=feat.astype(np.float32)               # (T,169)
    return np.clip(feat,-10,10)

def draw_prob_bars(img, events, probs, x=20, y=20, w=220, h=18, gap=6):
    for i,(ev,p) in enumerate(zip(events,probs)):
        y0=y+i*(h+gap)
        cv2.rectangle(img,(x,y0),(x+w,y0+h),(60,60,60),1)
        ww=int(w*float(p))
        cv2.rectangle(img,(x,y0),(x+ww,y0+h),(0,0,255) if p>=0.5 else (0,180,0),-1)
        cv2.putText(img,f"{ev[:14]} {p:.2f}",(x+w+10,y0+h-4),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1,cv2.LINE_AA)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--ckpt",required=True)          # train_lstm.py로 저장한 .pt
    ap.add_argument("--src", required=True)          # "0"=웹캠, 아니면 비디오 경로
    ap.add_argument("--meta",default=None)           # ckpt에 meta 없을 때만
    args=ap.parse_args()

    # ckpt & meta
    ck=torch.load(args.ckpt, map_location="cpu")
    meta=ck.get("meta",{})
    if (not meta) and args.meta and os.path.exists(args.meta):
        meta=json.load(open(args.meta,encoding="utf-8"))

    events    = meta.get("events")
    feat_dim  = ck.get("feat_dim")
    num_out   = ck.get("num_out", len(events) if events else None)
    win       = meta.get("win",16)
    stride    = meta.get("stride",4)
    resize_w  = meta.get("resize_w",640)
    norm_mean = np.array(meta["norm_mean"],np.float32)
    norm_std  = np.array(meta["norm_std"], np.float32)
    model_complexity = meta.get("model_complexity",0)

    assert events is not None and num_out==len(events), "meta.events / num_out 불일치"
    assert feat_dim==169, "feat_dim!=169이면 실시간 피처와 학습 피처가 다릅니다."

    # model
    model=LSTMAnom(feat_dim=feat_dim, num_out=num_out)
    model.load_state_dict(ck["model"]); model.eval()

    # mediapipe
    mp_pose = mp.solutions.pose.Pose(
        static_image_mode=False, model_complexity=model_complexity,
        enable_segmentation=False, min_detection_confidence=0.5, min_tracking_confidence=0.5
    )

    # video
    src = 0 if args.src=="0" else args.src
    cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap.release(); cap=cv2.VideoCapture(src, cv2.CAP_ANY)
        if not cap.isOpened():
            print(f"[ERR] cannot open {args.src}"); return

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT)>0 else -1
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    print(f"[INFO] fps={fps:.2f} total={total}")

    cv2.namedWindow("predict", cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)

    buf=deque(maxlen=win)
    idx=0
    while True:
        ok,bgr=cap.read()
        if not ok: break
        idx+=1

        # resize (학습과 동일 스케일 기준)
        if resize_w:
            h,w=bgr.shape[:2]; s=resize_w/float(w)
            bgr=cv2.resize(bgr,(resize_w,int(h*s)))

        # mediapipe pose → (33,4)
        rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
        res=mp_pose.process(rgb)
        if res and res.pose_landmarks:
            lm=res.pose_landmarks.landmark
            kpt=np.array([[p.x,p.y,p.z,p.visibility] for p in lm],np.float32)
        else:
            kpt=np.full((33,4), np.nan, np.float32)
        buf.append(kpt)

        text="warming..."
        if len(buf)==win and (idx%stride==0):
            feat = features_from_buf(list(buf))                  # (T,169)
            # 학습 통계 정규화(메타와 반드시 일치)
            feat = np.clip((feat - norm_mean) / (norm_std + 1e-6), -6, 6)
            x = torch.from_numpy(feat).unsqueeze(0).float()      # (1,T,F)

            with torch.no_grad():
                logits = model(x)                                 # (1,C) logits
                probs  = torch.sigmoid(logits).cpu().numpy()[0]   # (C,) probs

            # 전체 확률 출력 (한 줄)
            line=",".join(f"{p:.3f}" for p in probs)
            print(f"[{idx:05d}] probs=[{line}]")

            # 화면에도 막대 표시 + 전체 텍스트
            draw_prob_bars(bgr, events, probs, x=20, y=20, w=260, h=18, gap=6)
            text = " | ".join(f"{ev}:{probs[i]:.2f}" for i,ev in enumerate(events))

        # footer
        cv2.putText(bgr, text, (20, bgr.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0,0,255), 2, cv2.LINE_AA)

        cv2.imshow("predict", bgr)
        if cv2.waitKey(max(1,int(1000/fps))) & 0xFF in (27, ord('q')): break

    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL","2")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES","-1")
    main()
