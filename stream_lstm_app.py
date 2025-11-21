import os, time, json, threading
from collections import deque
import numpy as np
import cv2, torch, mediapipe as mp
import torch.nn as nn

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from dotenv import load_dotenv
from pathlib import Path

env_path = Path(__file__).parent / '.env'
load_dotenv(env_path)          # .env에서 로드

RTSP_URL = os.getenv('RTSP_URL')
# 노트북 카메라 사용시
if RTSP_URL == '0':
    RTSP_URL == 0

# ================== 사용자 설정 ==================
CKPT_PATH = "lstm_multilabel_05.pt"        # 학습 시 저장한 ckpt 파일(.pt)
META_PATH = None                   # ckpt에 meta가 없을 때만 사용(선택)
RESIZE_W  = None                   # 대역폭 낮추려면 예: 640 (None이면 원본)
JPEG_QUALITY = 80                  # 1~100 (클수록 화질↑, 용량↑)
STRIDE = None                      # meta에 없으면 여기 값 사용(예: 4)
OVERLAY_STATS = False               # 오버레이에 통계 표시 여부
# =================================================

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# ---------- 모델 정의 (사용자 코드 그대로) ----------
class AttPool(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w = nn.Linear(d, 1)
    def forward(self, h):                      # h:(B,T,D)
        a = self.w(h).squeeze(-1)             # (B,T)
        w = torch.softmax(a, dim=1).unsqueeze(-1)
        return (h*w).sum(1)                   # (B,D)

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
        return self.head(g)                    # (B,C) 로짓

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
"""
def draw_prob_bars(img, events, probs, x=20, y=20, w=260, h=18, gap=6):
    for i,(ev,p) in enumerate(zip(events,probs)):
        y0=y+i*(h+gap)
        cv2.rectangle(img,(x,y0),(x+w,y0+h),(60,60,60),1)
        ww=int(w*float(p))
        cv2.rectangle(img,(x,y0),(x+ww,y0+h),(0,0,255) if p>=0.5 else (0,180,0),-1)
        cv2.putText(img,f"{ev[:14]} {p:.2f}",(x+w+10,y0+h-4),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1,cv2.LINE_AA)

"""

# ---------- 전역 상태 ----------
lock = threading.Lock()
stop_event = threading.Event()
latest_frame_bgr = None        # 오버레이까지 포함된 최신 프레임(BGR)
last_probs = None              # 최근 확률 벡터(np.array)
events = []                    # 메타의 이벤트 라벨
win = 16
stride = 4
norm_mean = None
norm_std  = None
resize_w  = None
model = None
mp_pose = None

def load_model_and_meta():
    global model, events, win, stride, norm_mean, norm_std, resize_w
    ck = torch.load(CKPT_PATH, map_location="cpu")
    meta = ck.get("meta", {})
    if (not meta) and META_PATH and os.path.exists(META_PATH):
        meta = json.load(open(META_PATH, encoding="utf-8"))

    # 필수 메타
    events = meta.get("events")
    feat_dim = ck.get("feat_dim")
    num_out  = ck.get("num_out", len(events) if events else None)
    assert events is not None and num_out == len(events), "meta.events / num_out 불일치"
    assert feat_dim == 169, "feat_dim!=169이면 실시간 피처와 학습 피처가 다릅니다."

    # 하이퍼파라미터/전처리
    win      = meta.get("win", win)
    stride_v = meta.get("stride", STRIDE if STRIDE is not None else stride)
    resize_w = meta.get("resize_w", RESIZE_W if RESIZE_W is not None else None)
    mean     = np.array(meta["norm_mean"], np.float32)
    std      = np.array(meta["norm_std"],  np.float32)

    # 전역 반영
    globals()["stride"] = stride_v
    globals()["norm_mean"] = mean
    globals()["norm_std"]  = std
    globals()["resize_w"]  = resize_w

    # 모델 복원
    m = LSTMAnom(feat_dim=feat_dim, num_out=num_out)
    m.load_state_dict(ck["model"]); m.eval()
    globals()["model"] = m

def make_pose():
    m = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity = (0 if not os.environ.get("POSE_COMPLEX") else int(os.environ["POSE_COMPLEX"])),
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    return m

def capture_and_infer_loop():
    global latest_frame_bgr, last_probs, mp_pose

    # 안정화 옵션: RTSP over TCP
    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

    # MediaPipe 초기화
    mp_pose = make_pose()

    # 프레임 버퍼
    buf = deque(maxlen=win)

    # 카메라 열기
    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap.release(); cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_ANY)
    if not cap.isOpened():
        print("[ERR] RTSP 열기 실패"); return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    tick = time.time()

    while not stop_event.is_set():
        ok, bgr = cap.read()
        if not ok:
            time.sleep(0.1)
            continue

        # 리사이즈(선택)
        if resize_w:
            h,w = bgr.shape[:2]; s = resize_w/float(w)
            bgr = cv2.resize(bgr, (resize_w, int(h*s)))

        # Pose → (33,4)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        res = mp_pose.process(rgb)

        if res and res.pose_landmarks:
            lm = res.pose_landmarks.landmark
            # === 포즈 좌표 오버레이 ===
            draw_pose(bgr, lm, visibility_thr=0.5, r=3, th=2)

            kpt = np.array([[p.x, p.y, p.z, p.visibility] for p in lm], np.float32)
        else:
            kpt = np.full((33,4), np.nan, np.float32)

        if res and res.pose_landmarks:
            lm = res.pose_landmarks.landmark
            kpt = np.array([[p.x,p.y,p.z,p.visibility] for p in lm], np.float32)
        else:
            kpt = np.full((33,4), np.nan, np.float32)
        buf.append(kpt)

        text = "warming..."
        probs = None

        # stride마다 추론
        if len(buf) == win and (time.time() - tick) >= max(1.0/fps, 0.01):
            tick = time.time()
            feat = features_from_buf(list(buf))                         # (T,169)
            feat = np.clip((feat - norm_mean) / (norm_std + 1e-6), -6, 6)
            x = torch.from_numpy(feat).unsqueeze(0).float()             # (1,T,F)
            with torch.no_grad():
                logits = model(x)                                       # (1,C)
                probs  = torch.sigmoid(logits).cpu().numpy()[0]         # (C,)
            text = " | ".join(f"{ev}:{probs[i]:.2f}" for i,ev in enumerate(events))
            

         # === 오버레이 옵션화 ===
        if probs is not None:
            if OVERLAY_STATS:
                draw_prob_bars(bgr, events, probs, x=20, y=20, w=260, h=18, gap=6)
            with lock:
                last_probs = probs.copy()

        # 하단 텍스트도 옵션화
        if OVERLAY_STATS and probs is not None:
            text = " | ".join(f"{ev}:{probs[i]:.2f}" for i,ev in enumerate(events))
        cv2.putText(bgr, ("" if OVERLAY_STATS else ""), (20, bgr.shape[0]-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2, cv2.LINE_AA)
        # =======

    
        # 최신 프레임 저장
        with lock:
            latest_frame_bgr = bgr

    cap.release()

def mjpeg_generator():
    boundary = b"--frame"
    while not stop_event.is_set():
        with lock:
            frame = None if latest_frame_bgr is None else latest_frame_bgr.copy()
        if frame is None:
            time.sleep(0.03)
            continue
        ok, jpeg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        if not ok:
            continue
        data = jpeg.tobytes()
        yield b"%b\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % (boundary, len(data))
        yield data + b"\r\n"
        time.sleep(0.01)  # 전송 간격(대역폭/CPU 조절)

# mediapipe 포즈 그리기 함수
def draw_pose(img, landmarks, visibility_thr=0.5, r=3, th=2):
    h, w = img.shape[:2]
    # 좌표 변환 함수
    def to_px(lm):
        x = int(np.clip(lm.x * w, 0, w-1))
        y = int(np.clip(lm.y * h, 0, h-1))
        return x, y

    # 연결(스켈레톤) 그리기: MediaPipe의 기본 연결 사용
    CONNS = mp.solutions.pose.POSE_CONNECTIONS
    # 선
    for a, b in CONNS:
        la, lb = landmarks[a], landmarks[b]
        if (la.visibility >= visibility_thr) and (lb.visibility >= visibility_thr):
            xa, ya = to_px(la); xb, yb = to_px(lb)
            cv2.line(img, (xa, ya), (xb, yb), (0, 255, 255), th, cv2.LINE_AA)
    # 점
    for lm in landmarks:
        if lm.visibility >= visibility_thr:
            x, y = to_px(lm)
            cv2.circle(img, (x, y), r, (255, 200, 0), -1, cv2.LINE_AA)

# ---------- FastAPI ----------
@app.on_event("startup")
def on_start():
    # 환경 세팅 (원하면 CPU 강제)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL","2")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES","-1")
    load_model_and_meta()
    t = threading.Thread(target=capture_and_infer_loop, daemon=True)
    t.start()

@app.on_event("shutdown")
def on_shutdown():
    stop_event.set()

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/stream")
def stream():
    return StreamingResponse(mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/probs")
def probs():
    with lock:
        p = None if last_probs is None else [float(x) for x in last_probs]
        ev = list(events) if events else []
    return JSONResponse({"events": ev, "probs": p, "ok": p is not None})
