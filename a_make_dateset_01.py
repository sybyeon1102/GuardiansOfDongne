# a_make_dateset_01.py
# changed from: a_make_dataset_00.py
# purpose: 구매행동/정식개방데이터 전처리
# key changes:
# EVENTS values updated
# parse_cvat_intervals function modified to handle "moving" label differently
# subfolder name change from VS_... to VL_...
# overlap parameter added to command-line arguments and metadata

# command:
# python a_make_dataset_01.py 
# --video_root "경로예시:./Training/01.원천데이터/" 
# --xml_root "경로예시:./Training/02.라벨링데이터/" 
# --out_dir 폴더이름
# --overlap 0.1
# 나머지 옵션은 default 값 사용


import os, json, argparse, xml.etree.ElementTree as ET
import numpy as np, cv2
from collections import defaultdict
import mediapipe as mp
import time

EVENTS = ["moving","select","test","buying","return","compare"]


def parse_cvat_intervals(xml_path):

    import xml.etree.ElementTree as ET
    from collections import defaultdict

    def norm(s: str) -> str:
        return s.strip().lower().replace(" ", "_")

    # ----- 0) 메타에서 전체 프레임 범위 -----
    root = ET.parse(xml_path).getroot()
    task = root.find("meta/task")
    s0 = int(task.findtext("start_frame", "0") or "0") if task is not None else 0
    e0 = int(task.findtext("stop_frame", "0")  or "0") if task is not None else 0

    # ----- 1) <track>에서 라벨/프레임 수집 -----
    start_frames = defaultdict(list)   # base → [start frames]
    end_frames   = defaultdict(list)   # base → [end frames]
    solid_runs   = defaultdict(list)   # base → [(s,e)]  outside=0 연속 구간(포함 구간)

    for tr in root.findall("track"):
        raw = tr.get("label", "") or ""
        lab = norm(raw)
        base = lab.replace("_start", "").replace("_end", "")

        # box만 사용 (points 등은 제외)
        on = sorted([
            int(x.get("frame"))
            for x in tr.findall("box")
            if x.get("outside", "0") == "0"
        ])
        if not on:
            continue

        if lab.endswith("_start"):
            # 일반적으로 단발(start) 1 프레임 (여러 프레임이어도 첫 프레임 사용)
            start_frames[base].append(on[0])
        elif lab.endswith("_end"):
            # 일반적으로 단발(end) 1 프레임 (여러 프레임이어도 마지막 프레임 사용)
            end_frames[base].append(on[-1])
        else:
            # 일반 트랙(연속 구간) → (s,e) 구간으로 분할
            run_s = on[0]
            prev = on[0]
            for f in on[1:]:
                if f == prev + 1:
                    prev = f
                else:
                    solid_runs[base].append((run_s, prev))
                    run_s = f
                    prev = f
            solid_runs[base].append((run_s, prev))

    # ----- 2) <tag>에도 *_start/*_end가 있으면 함께 페어링에 활용 -----
    tags = root.findall("tag")
    if tags:
        per = defaultdict(list)  # base → [(lab, frame)]
        for tg in tags:
            lab = norm(tg.get("label", "") or "")
            f   = int(tg.get("frame", "0") or "0")
            base = lab.replace("_start", "").replace("_end", "")
            per[base].append((lab, f))

        for base, items in per.items():
            st = sorted([f for (lab, f) in items if lab.endswith("_start")])
            ed = sorted([f for (lab, f) in items if lab.endswith("_end")])

            # st 개수가 더 많으면 남는 start는 e0까지 확장
            # (아래 매칭 루프에서 처리되므로 여기서는 리스트만 채워줌)
            for i in range(min(len(st), len(ed))):
                start_frames[base].append(st[i])
                end_frames[base].append(ed[i])
            if len(st) > len(ed):
                for a in st[len(ed):]:
                    start_frames[base].append(a)
                    end_frames[base].append(e0)

    # ----- 3) start/end 페어로 구간 만들기 + 일반 트랙 구간 합치기 -----
    tmp = defaultdict(list)  # base → [(s,e)]

    # (a) 먼저 모든 base의 일반 트랙 구간을 넣어둠
    for base, ivs in solid_runs.items():
        tmp[base].extend(ivs)

    # (b) *_start/_end 페어링 (start 이후의 첫 end를 매칭)
    bases = set(list(start_frames.keys()) + list(end_frames.keys()))
    for base in bases:
        st = sorted(start_frames.get(base, []))
        ed = sorted(end_frames.get(base, []))
        i = j = 0
        while i < len(st):
            a = st[i]
            while j < len(ed) and ed[j] < a:
                j += 1
            b = ed[j] if j < len(ed) else e0
            if b < a:
                b = a
            tmp[base].append((a, b))
            i += 1
            if j < len(ed):
                j += 1

    # ----- 4) 이벤트별로 병합/필터링 -----
    intervals = {ev: [] for ev in EVENTS}

    for base, ivs in tmp.items():
        if base not in intervals:
            continue

        # moving 특수 규칙: *_start/_end를 무시하고 "연속 구간(solid_runs)"만 사용
        if base == "moving":
            ivs = solid_runs.get("moving", [])
        if not ivs:
            continue

        # 구간 정렬 및 병합(인접/겹침 병합: a ≤ last_e+1 이면 합침)
        ivs = sorted(ivs)
        merged = []
        for a, b in ivs:
            if not merged or a > merged[-1][1] + 1:
                merged.append([a, b])
            else:
                merged[-1][1] = max(merged[-1][1], b)

        intervals[base] = [(int(a), int(b)) for a, b in merged]

    return s0, e0, intervals
    

def extract_pose(video_path, resize_w=640, model_complexity=0):
    mp_pose = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=model_complexity,
                                     enable_segmentation=False, min_detection_confidence=0.5,
                                     min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[WARN] cannot open video: {video_path}")
        return np.zeros((0,33,4), np.float32)
    seq=[]; n=0
    while True:
        ok,bgr=cap.read()
        if not ok: break
        n+=1
        if resize_w:
            scale = resize_w / bgr.shape[1]
            bgr = cv2.resize(bgr,(resize_w,int(bgr.shape[0]*scale)))
        rgb = cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
        res = mp_pose.process(rgb)
        if res and res.pose_landmarks:
            lm = res.pose_landmarks.landmark
            seq.append([[p.x,p.y,p.z,p.visibility] for p in lm])
        else:
            seq.append([[np.nan]*4]*33)
        if n%100==0: print(f"   [pose] {n} frames")
    cap.release(); mp_pose.close()
    return np.array(seq, np.float32)

def ffill_bfill(arr):
    T,D = arr.shape
    out = arr.copy()
    last = np.zeros(D,np.float32); has = np.zeros(D,bool)
    for t in range(T):
        nz = ~np.isnan(out[t]); last[nz]=out[t,nz]; has|=nz
        miss=np.isnan(out[t])&has; out[t,miss]=last[miss]
    last[:]=0; has[:]=False
    for t in range(T-1,-1,-1):
        nz=~np.isnan(out[t]); last[nz]=out[t,nz]; has|=nz
        miss=np.isnan(out[t])&has; out[t,miss]=last[miss]
    return np.nan_to_num(out,nan=0.0,posinf=0.0,neginf=0.0)

def build_features(kpts):
    if kpts.size==0: return np.zeros((0,169),np.float32)
    T=kpts.shape[0]
    xy=kpts[:,:,:2].reshape(T,-1)
    vis=kpts[:,:,3:4].reshape(T,-1)
    xy=ffill_bfill(xy).reshape(T,33,2)
    vis=ffill_bfill(vis).reshape(T,33,1)
    hip=np.mean(xy[:,[23,24],:],axis=1)
    sh=np.mean(xy[:,[11,12],:],axis=1)
    sc=np.linalg.norm(sh-hip,axis=1,keepdims=True); sc[sc<1e-3]=1.0
    xy_n=(xy-hip[:,None,:])/sc[:,None,:]
    vel=np.diff(xy_n,axis=0,prepend=xy_n[:1])
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
    ],axis=1)
    feat=np.concatenate([xy_n.reshape(T,-1), vel.reshape(T,-1), angs, vis.reshape(T,-1)],axis=1)
    return np.clip(feat.astype(np.float32), -10, 10)

def slice_windows(feat,start_f,stop_f,intervals,win=16,stride=4,overlap=0.25):
    C=len(EVENTS)
    if feat.shape[0]==0:
        return np.zeros((0,win,feat.shape[1]),np.float32), np.zeros((0,C),np.float32)
    def ov(a0,a1,b0,b1): return max(0,min(a1,b1)-max(a0,b0)+1)
    Xs,Ys=[],[]
    for s in range(start_f,stop_f-win+2,stride):
        e=s+win-1
        y=np.zeros(C,np.float32)
        for i,ev in enumerate(EVENTS):
            for (a,b) in intervals.get(ev, []):
                if ov(s,e,a,b)/win>=overlap:
                    y[i]=1.0; break
        Xs.append(feat[s:e+1]); Ys.append(y)
    return np.stack(Xs,0), np.stack(Ys,0)

def main():
    start_time = time.time()

    ap=argparse.ArgumentParser()
    ap.add_argument("--video_root", required=True)
    ap.add_argument("--xml_root", required=True)
    ap.add_argument("--out_dir", default="ds_lstm_all")
    ap.add_argument("--win", type=int, default=16)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--overlap", type=float, default=0.25)
    ap.add_argument("--resize_w", type=int, default=640)
    ap.add_argument("--model_complexity", type=int, default=0)
    args=ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    allX, allY = [], []

    for sub in os.listdir(args.video_root):
        vdir=os.path.join(args.video_root,sub)
        if not os.path.isdir(vdir): continue

        v_list = os.listdir(vdir)
        # 카테고리별 갯수 조정시
        #v_list = v_list[:1]
        
        for f in v_list:
            if not f.lower().endswith(".mp4"): continue
            vpath=os.path.join(vdir,f)

            # sub내용 변경 : VS_... => VL_...
            sub_xml = sub.replace("VS", "VL")
            xml_path=os.path.join(args.xml_root,sub_xml,f.replace(".mp4",".xml"))

            if not os.path.exists(xml_path):
                print(f"[SKIP] {f} → xml 없음"); continue
            print(f"[PROC] {sub} :: {f}")
            s,e,intervals=parse_cvat_intervals(xml_path)
            kpts=extract_pose(vpath, args.resize_w, args.model_complexity)
            feat=build_features(kpts)
            X,Y=slice_windows(feat,s,e,intervals,args.win,args.stride,args.overlap)
            if len(X)==0: continue
            allX.append(X); allY.append(Y)

    X=np.concatenate(allX,0); Y=np.concatenate(allY,0)
    print(f"[INFO] 통합 데이터: X={X.shape}, Y={Y.shape}")

    X=np.nan_to_num(X,nan=0.0,posinf=0.0,neginf=0.0)
    ds_mean=X.mean(axis=(0,1),keepdims=True)
    ds_std=X.std(axis=(0,1),keepdims=True)+1e-6
    Xn=np.clip((X-ds_mean)/ds_std, -6,6)

    np.save(os.path.join(args.out_dir,"X.npy"), Xn.astype(np.float32))
    np.save(os.path.join(args.out_dir,"Y.npy"), Y.astype(np.float32))
    meta={
        "events":EVENTS,
        "win":args.win,
        "stride":args.stride,
        "overlap":args.overlap,
        "norm_mean":ds_mean.squeeze().tolist(),
        "norm_std":ds_std.squeeze().tolist(),
        "resize_w":args.resize_w,
        "model_complexity":args.model_complexity,
    }
    json.dump(meta, open(os.path.join(args.out_dir,"meta.json"),"w"), ensure_ascii=False, indent=2)
    print(f"[OK] X:{Xn.shape} Y:{Y.shape} saved to {args.out_dir}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"[TIME] Elapsed time: {elapsed_time:.2f} seconds")

if __name__=="__main__":
    main()
