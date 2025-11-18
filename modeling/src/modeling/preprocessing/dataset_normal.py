"""
정상(구매/일상) 행동 전처리를 담당하는 모듈입니다.

This module builds the normal (purchase/regular) behavior dataset
using the common preprocessing pipeline defined in dataset_common.
"""

from pathlib import Path
from collections import defaultdict

import xml.etree.ElementTree as ET

from project_core import Result
from .dataset_common import (
    DatasetArrays,
    DatasetConfig,
    Intervals,
    build_dataset_multilabel,
    run_dataset_cli,
)


EVENTS_NORMAL: list[str] = [
    "moving",
    "select",
    "test",
    "buying",
    "return",
    "compare",
]
"""
정상(구매/일상) 행동용 이벤트 이름 목록입니다.

List of normal (purchase/regular) behavior event labels.
"""


def parse_intervals_normal(xml_path: Path) -> tuple[int, int, Intervals]:
    """
    CVAT XML에서 정상(구매/일상) 행동 이벤트 구간을 파싱합니다.

    Parse normal behavior event intervals from a CVAT XML file.

    - moving 라벨은 *_start/_end 구간을 무시하고,
      연속 구간(solid_runs)만 사용합니다.
    """
    def norm(s: str) -> str:
        return s.strip().lower().replace(" ", "_")

    root = ET.parse(xml_path).getroot()
    task = root.find("meta/task")
    s0 = int(task.findtext("start_frame", "0") or "0") if task is not None else 0
    e0 = int(task.findtext("stop_frame", "0") or "0") if task is not None else 0

    start_frames: dict[str, list[int]] = defaultdict(list)
    end_frames: dict[str, list[int]] = defaultdict(list)
    solid_runs: dict[str, list[tuple[int, int]]] = defaultdict(list)

    # ----- 1) <track>에서 라벨/프레임 수집 -----
    for tr in root.findall("track"):
        raw = tr.get("label", "") or ""
        lab = norm(raw)
        base = lab.replace("_start", "").replace("_end", "")

        on = sorted(
            int(x.get("frame"))
            for x in tr.findall("box")
            if x.get("outside", "0") == "0"
        )
        if not on:
            continue

        if lab.endswith("_start"):
            start_frames[base].append(on[0])
        elif lab.endswith("_end"):
            end_frames[base].append(on[-1])
        else:
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

    # ----- 2) <tag> 기반 *_start/_end (원 코드에서 moving을 포함) -----
    tags = root.findall("tag")
    if tags:
        per: dict[str, list[tuple[str, int]]] = defaultdict(list)
        for tg in tags:
            lab = norm(tg.get("label", "") or "")
            f = int(tg.get("frame", "0") or "0")
            base = lab.replace("_start", "").replace("_end", "")
            per[base].append((lab, f))

        for base, items in per.items():
            st = sorted(f for (lab, f) in items if lab.endswith("_start"))
            ed = sorted(f for (lab, f) in items if lab.endswith("_end"))

            for i in range(min(len(st), len(ed))):
                start_frames[base].append(st[i])
                end_frames[base].append(ed[i])
            if len(st) > len(ed):
                for a in st[len(ed) :]:
                    start_frames[base].append(a)
                    end_frames[base].append(e0)

    # ----- 3) start/end 페어 + 일반 트랙 구간 합치기 -----
    tmp: dict[str, list[tuple[int, int]]] = defaultdict(list)

    for base, ivs in solid_runs.items():
        tmp[base].extend(ivs)

    bases = set(list(start_frames.keys()) + list(end_frames.keys()))
    for base in bases:
        st = sorted(start_frames.get(base, []))
        ed = sorted(end_frames.get(base, []))
        i = 0
        j = 0
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

    # ----- 4) 이벤트별 병합/필터링 (moving 특수 규칙) -----
    intervals: Intervals = {ev: [] for ev in EVENTS_NORMAL}

    for base, ivs in tmp.items():
        if base not in intervals:
            continue

        # moving 은 *_start/_end를 무시하고 solid_runs 기반으로만 구간 사용
        if base == "moving":
            ivs = solid_runs.get("moving", [])

        if not ivs:
            continue

        ivs = sorted(ivs)
        merged: list[list[int]] = []
        for a, b in ivs:
            if not merged or a > merged[-1][1] + 1:
                merged.append([a, b])
            else:
                merged[-1][1] = max(merged[-1][1], b)

        intervals[base] = [(int(a), int(b)) for a, b in merged]

    return s0, e0, intervals


def build_dataset_normal(config: DatasetConfig) -> Result[DatasetArrays, str]:
    """
    정상(구매/일상) 행동용 LSTM 멀티라벨 데이터셋을 구축합니다.

    Build a multilabel LSTM dataset for normal (purchase/regular) behavior.
    """
    local_conf = DatasetConfig(
        video_root=config.video_root,
        xml_root=config.xml_root,
        out_dir=config.out_dir,
        events=EVENTS_NORMAL,
        window_size=config.window_size,
        stride=config.stride,
        overlap=config.overlap,
        resize_w=config.resize_w,
        model_complexity=config.model_complexity,
    )
    return build_dataset_multilabel(local_conf, parse_intervals=parse_intervals_normal)


def main() -> None:
    run_dataset_cli(
        description="Build normal (purchase/regular) LSTM dataset from videos and CVAT XML.",
        default_events=EVENTS_NORMAL,
        build_fn=build_dataset_normal,
        dataset_name="normal",
    )


if __name__ == "__main__":
    main()
