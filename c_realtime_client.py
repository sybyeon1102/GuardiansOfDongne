# c_realtime_client.py
import os, json, argparse, time, math, re
from collections import deque
import numpy as np
import cv2, torch, mediapipe as mp
import torch.nn as nn

# ---------------- Model ----------------
class AttPool(nn.Module):
    def __init__(self, d): super().__init__(); self.w = nn.Linear(d, 1)
    def forward(self, h):
        a = self.w(h).squeeze(-1); w = torch.softmax(a, dim=1).unsqueeze(-1)
        return (h * w).sum(1)

class LSTMAnom(nn.Module):
    def __init__(self, feat_dim, hidden=128, layers=2, num_out=1, bidir=True):
        super().__init__()
        self.pre  = nn.Sequential(nn.Conv1d(feat_dim, feat_dim, 3, padding=1), nn.ReLU())
        self.lstm = nn.LSTM(feat_dim, hidden, num_layers=layers, batch_first=True,
                            bidirectional=bidir, dropout=0.1)
        d = hidden * (2 if bidir else 1)
        self.pool = AttPool(d)
        self.head = nn.Linear(d, num_out)
    def forward(self, x):
        z = self.pre(x.transpose(1,2)).transpose(1,2)
        h,_ = self.lstm(z); g = self.pool(h)
        return self.head(g)

# ---------------- Features ----------------
def _ffill_bfill(arr):
    T,D = arr.shape; out = arr.copy()
    last = np.zeros(D, np.float32); has = np.zeros(D, bool)
    for t in range(T):
        nz = ~np.isnan(out[t]); last[nz] = out[t, nz]; has |= nz
        miss = np.isnan(out[t]) & has; out[t, miss] = last[miss]
    last[:] = 0; has[:] = False
    for t in range(T-1, -1, -1):
        nz = ~np.isnan(out[t]); last[nz] = out[t, nz]; has |= nz
        miss = np.isnan(out[t]) & has; out[t, miss] = last[miss]
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

def features_from_buf(buf):
    k = np.stack(buf,0)              # (T,33,4)
    T = k.shape[0]
    xy  = k[:,:,:2].reshape(T,-1)    # (T,66)
    vis = k[:,:,3:4].reshape(T,-1)   # (T,33)
    xy  = _ffill_bfill(xy).reshape(T,33,2)
    vis = _ffill_bfill(vis).reshape(T,33,1)

    hip = np.mean(xy[:,[23,24],:], axis=1)
    sh  = np.mean(xy[:,[11,12],:], axis=1)
    sc  = np.linalg.norm(sh-hip, axis=1, keepdims=True); sc[sc<1e-3] = 1.0

    xy_n = (xy-hip[:,None,:]) / sc[:,None,:]
    vel  = np.diff(xy_n, axis=0, prepend=xy_n[:1])

    def ang(a,b,c):
        v1=a-b; v2=c-b
        n1=np.linalg.norm(v1,axis=-1); n2=np.linalg.norm(v2,axis=-1)
        n1[n1==0]=1e-6; n2[n2==0]=1e-6
        cos=(v1*v2).sum(-1)/(n1*n2)
        return np.arccos(np.clip(cos, -1, 1))
    def pick(i): return xy_n[:,i,:]
    angs = np.stack([
        ang(pick(11),pick(13),pick(15)),
        ang(pick(12),pick(14),pick(16)),
        ang(pick(23),pick(25),pick(27)),
        ang(pick(24),pick(26),pick(28)),
    ], axis=1)

    feat = np.concatenate([xy_n.reshape(T,-1), vel.reshape(T,-1), angs, vis.reshape(T,-1)],1).astype(np.float32)
    return np.clip(feat, -10, 10)    # (T,169)

# ---------------- HTTP ----------------
def post_json(url, payload, timeout=2.0):
    headers = {"Content-Type":"application/json"}
    try:
        import requests
        r = requests.post(url, data=json.dumps(payload), headers=headers, timeout=timeout)
        return r.status_code
    except Exception:
        from urllib import request
        req = request.Request(url, data=json.dumps(payload).encode(), headers=headers, method="POST")
        try:
            with request.urlopen(req, timeout=timeout) as resp:
                return resp.getcode()
        except Exception:
            return -1

# ---------------- Utils ----------------
def looks_live_src(src_str: str) -> bool:
    if src_str == "0": return True
    if re.match(r"^(rtsp|rtsps|rtmp|http|https|rtp)://", src_str, re.I): return True
    return False

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--src",  required=True)
    ap.add_argument("--meta", default=None)
    ap.add_argument("--api_url", required=True)

    # 안정화 파라미터
    ap.add_argument("--ema_alpha", type=float, default=0.3)
    ap.add_argument("--start_thr", type=float, default=0.80)
    ap.add_argument("--end_thr",   type=float, default=0.55)
    ap.add_argument("--vote_win",  type=int,   default=5)
    ap.add_argument("--min_start_votes", type=int, default=4)
    ap.add_argument("--min_end_votes",   type=int, default=4)
    ap.add_argument("--min_event_sec",   type=float, default=2.0)
    ap.add_argument("--cooldown_sec",    type=float, default=3.0)

    # 전송/로그 정책
    ap.add_argument("--event_only", action="store_true",
                    help="확정된 START/END만 서버 전송 (HEARTBEAT/기타 전송 차단)")
    ap.add_argument("--heartbeat_sec", type=float, default=3.0,
                    help="HEARTBEAT 주기(초). 0이면 전송 안 함")
    ap.add_argument("--log_prefix", default="LOG", help="매 추론 로그 prefix")
    ap.add_argument("--print_probs", action="store_true",
                    help="매 추론 로그에 모든 클래스 확률 출력")

    # 실시간 강제(벽시계 기준 + 파일 스로틀)
    ap.add_argument("--throttle", action="store_true",
                    help="파일 입력을 FPS에 맞춰 실시간 속도로 처리")
    ap.add_argument("--live_gap_timeout", type=float, default=5.0,
                    help="라이브에서 프레임 끊김 간주(초). 끊기면 HB만 유지")
    ap.add_argument("--eof_force_end", action="store_true",
                    help="EOF에서 duration이 짧아도 END 강제(테스트용)")
    args = ap.parse_args()

    # ckpt/meta
    ck = torch.load(args.ckpt, map_location="cpu")
    meta = ck.get("meta", {})
    if (not meta) and args.meta and os.path.exists(args.meta):
        meta = json.load(open(args.meta, encoding="utf-8"))
    events = meta.get("events")
    feat_dim = ck.get("feat_dim")
    num_out  = ck.get("num_out", len(events) if events else None)
    win      = meta.get("win", 16)
    stride   = meta.get("stride", 4)
    resize_w = meta.get("resize_w", 640)
    norm_mean = np.array(meta["norm_mean"], np.float32)
    norm_std  = np.array(meta["norm_std"],  np.float32)
    mc = int(meta.get("model_complexity", 0))

    assert events is not None and num_out == len(events), "meta.events / num_out 불일치"
    assert feat_dim == 169, "feat_dim!=169 (실시간 피처와 학습 피처가 다름)"

    # model
    model = LSTMAnom(feat_dim=feat_dim, num_out=num_out)
    model.load_state_dict(ck["model"]); model.eval()

    # mediapipe (키워드 인자)
    mp_pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=mc,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # open source
    src = 0 if args.src == "0" else args.src

    cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap.release(); cap = cv2.VideoCapture(src, cv2.CAP_ANY)
        if not cap.isOpened():
            print(f"[ERR] cannot open {args.src}"); return

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT)>0 else -1
    fps   = cap.get(cv2.CAP_PROP_FPS) or 0.0
    is_file = (not looks_live_src(str(src))) and (total > 0)
    fps_eff = fps if fps and fps > 0 else 25.0
    target_dt = 1.0 / fps_eff if fps_eff > 0 else 0.0
    print(f"[INFO] fps={fps:.2f} total={total} is_file={is_file}")

    # 벽시계(단조증가) 기준의 '실시간'
    mono = time.monotonic
    t0 = mono()
    last_frame_mono = mono()
    last_hb_sec = 0.0
    last_end_sec = -1e9

    # state
    buf = deque(maxlen=win)
    ema = None
    votes = deque(maxlen=args.vote_win)
    current_event = None   # {"name","start_sec","start_frame"}
    idx = 0

    # helper: 참고용 video_ts
    def get_video_ts():
        pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        if pos_ms and pos_ms > 0:
            return pos_ms / 1000.0
        return (idx / fps_eff) if is_file else None

    # ---- 전송 함수 (event_only면 START/END만 송신) ----
    def send(kind, payload):
        if args.event_only and kind not in ("START","END"):
            return
        wall_ts = time.time()
        pl = {"type": kind, "ts": wall_ts, **payload}
        vt = get_video_ts()
        if vt is not None:
            pl["video_ts"] = float(vt)
        code = post_json(args.api_url, pl)
        log_ts = vt if vt is not None else (mono()-t0)
        print(f"[{kind}] {payload} real_ts={log_ts:.3f} -> HTTP {code}")

    # ---- 로그 함수 (매 추론마다 항상 출력) ----
    def log_step(tag, info: dict, probs=None):
        base = f"[{args.log_prefix}] {tag} " + " ".join(f"{k}={v}" for k,v in info.items())
        if args.print_probs and probs is not None:
            pstr = ",".join(f"{events[i]}:{probs[i]:.3f}" for i in range(len(events)))
            print(f"{base} | probs=[{pstr}]")
        else:
            print(base)

    while True:
        loop_start = time.perf_counter()
        ok, bgr = cap.read()
        now = mono()

        if not ok:
            break
        idx += 1
        last_frame_mono = now
        real_ts = now - t0  # 실시간(벽시계) 초

        if resize_w:
            h,w = bgr.shape[:2]; s = resize_w/float(w)
            bgr = cv2.resize(bgr, (resize_w, int(h*s)))

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        res = mp_pose.process(rgb)
        if res and res.pose_landmarks:
            lm = res.pose_landmarks.landmark
            kpt = np.array([[p.x, p.y, p.z, p.visibility] for p in lm], np.float32)
        else:
            kpt = np.full((33,4), np.nan, np.float32)
        buf.append(kpt)

        # 추론은 stride마다, 전송은 전이때만, 로그는 매 추론마다
        do_infer = (len(buf)==win) and (idx % stride == 0)
        if do_infer:
            feat = features_from_buf(list(buf))
            feat = np.clip((feat - norm_mean) / (norm_std + 1e-6), -6, 6)
            x = torch.from_numpy(feat).unsqueeze(0).float()
            with torch.no_grad():
                probs = torch.sigmoid(model(x)).cpu().numpy()[0]

            # EMA
            if ema is None: ema = probs.copy()
            else: ema = args.ema_alpha*probs + (1.0-args.ema_alpha)*ema

            # top & votes
            top_i = int(np.argmax(ema))
            top_ev, top_p = events[top_i], float(ema[top_i])
            votes.append(top_ev if top_p >= args.start_thr else None)

            # ★ 매 추론 로그(항상 출력) ★
            log_step(
                tag="infer",
                info={
                    "t": f"{real_ts:.3f}",
                    "idx": idx,
                    "top": top_ev,
                    "p": f"{top_p:.3f}",
                    "v_yes": votes.count(top_ev),
                    "v_none": votes.count(None),
                    "state": ("ACTIVE" if current_event else "IDLE")
                },
                probs=ema  # --print_probs 켜면 전체 확률도 추가 출력
            )

            # START 전이 (확정 시 1회 전송)
            if current_event is None:
                gap_ok = (real_ts - last_end_sec) >= args.cooldown_sec
                if gap_ok and (votes.count(top_ev) >= args.min_start_votes):
                    current_event = {"name": top_ev, "start_sec": real_ts, "start_frame": idx}
                    send("START", {"event": top_ev, "prob": round(top_p,4), "frame_idx": idx})
                    votes.clear()

            # END 전이 (확정 시 1회 전송)
            else:
                end_votes = votes.count(None)
                long_enough = (real_ts - current_event["start_sec"]) >= args.min_event_sec
                end_ok = (top_p <= args.end_thr) and (end_votes >= args.min_end_votes) and long_enough
                if end_ok:
                    dur = real_ts - current_event["start_sec"]
                    send("END", {"event": current_event["name"],
                                 "duration_sec": round(dur,3),
                                 "frame_idx": idx})
                    last_end_sec = real_ts
                    current_event = None
                    votes.clear()

        # HEARTBEAT (옵션, event_only면 자동 차단)
        if (not args.event_only) and args.heartbeat_sec > 0 and (real_ts - last_hb_sec) >= args.heartbeat_sec:
            last_hb_sec = real_ts
            hb = {"state": ("ACTIVE" if current_event else "IDLE"), "frame_idx": idx}
            if ema is not None:
                i = int(np.argmax(ema)); hb["top_event"]=events[i]; hb["top_prob"]=round(float(ema[i]),4)
            send("HEARTBEAT", hb)

        # 라이브 끊김 감시(참고용): HB만 유지
        if (not is_file) and (time.monotonic() - last_frame_mono) > args.live_gap_timeout:
            pass

        # 파일 입력은 FPS에 맞춰 실시간 스로틀
        if is_file and args.throttle and target_dt > 0:
            spent = time.perf_counter() - loop_start
            to_sleep = target_dt - spent
            if to_sleep > 0:
                time.sleep(to_sleep)

    # EOF 보정: 진행 중 이벤트가 있으면 종료 처리(테스트용 강제 옵션 포함)
    if current_event is not None:
        real_ts = time.monotonic() - t0
        dur = real_ts - current_event["start_sec"]
        if dur >= args.min_event_sec or args.eof_force_end:
            send("END", {"event": current_event["name"],
                         "duration_sec": round(max(0.0,dur),3),
                         "frame_idx": idx})

    cap.release()
    print("[INFO] finished.")

if __name__ == "__main__":
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL","2")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES","-1")
    main()