"""
이상 행동(정식 개방 데이터) 전처리를 담당하는 모듈입니다.

This module builds the anomaly behavior dataset using the common
preprocessing pipeline defined in dataset_common.
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


EVENTS_ANOMALY: list[str] = [
    "abandon",
    "broken",
    "fall",
    "fight",
    "fire",
    "smoke",
    "theft",
    "weak_pedestrian",
]
"""
이상 행동(이벤트) 이름 목록입니다.

List of anomaly event labels used in the anomaly dataset.
"""


def parse_intervals_anomaly(xml_path: Path) -> tuple[int, int, Intervals]:
    """
    CVAT XML에서 이상 행동 이벤트 구간을 파싱합니다.

    Parse anomaly event intervals from a CVAT XML file.

    반환:
    - s0: 전체 관심 구간 시작 프레임
    - e0: 전체 관심 구간 끝 프레임
    - intervals: 이벤트별 (start, end) 구간 딕셔너리
    """
    def norm(s: str) -> str:
        return s.strip().lower().replace(" ", "_")

    root = ET.parse(xml_path).getroot()
    task = root.find("meta/task")
    s0 = int(task.findtext("start_frame", "0") or "0") if task is not None else 0
    e0 = int(task.findtext("stop_frame", "0") or "0") if task is not None else 0

    # 1) <track>에서 라벨/프레임 수집
    start_frames: dict[str, list[int]] = defaultdict(list)
    end_frames: dict[str, list[int]] = defaultdict(list)
    solid_runs: dict[str, list[tuple[int, int]]] = defaultdict(list)

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
            # 일반 트랙(연속 구간)을 (s,e) 구간으로 분할
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

    # 2) 태그 기반 *_start/_end (필요시 추가 확장 가능)
    # 원본 코드에서는 tag 사용이 없으므로, 여기서는 생략 가능.
    # 필요하면 dataset_normal 스타일로 확장하면 된다.

    # 3) start/end 페어로 구간 만들기 + 일반 트랙 구간 합치기
    tmp: dict[str, list[tuple[int, int]]] = defaultdict(list)

    # (a) 일반 트랙 구간
    for base, ivs in solid_runs.items():
        tmp[base].extend(ivs)

    # (b) *_start/_end 페어링
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

    # 4) 이벤트별 병합/필터링
    intervals: Intervals = {ev: [] for ev in EVENTS_ANOMALY}

    for base, ivs in tmp.items():
        if base not in intervals:
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


def build_dataset_anomaly(config: DatasetConfig) -> Result[DatasetArrays, str]:
    """
    이상 행동용 LSTM 멀티라벨 데이터셋을 구축합니다.

    Build a multilabel LSTM dataset for anomaly events.
    """
    local_conf = DatasetConfig(
        video_root=config.video_root,
        xml_root=config.xml_root,
        out_dir=config.out_dir,
        events=EVENTS_ANOMALY,
        window_size=config.window_size,
        stride=config.stride,
        overlap=config.overlap,
        resize_w=config.resize_w,
        model_complexity=config.model_complexity,
    )
    return build_dataset_multilabel(local_conf, parse_intervals=parse_intervals_anomaly)


def main() -> None:
    run_dataset_cli(
        description="Build anomaly LSTM dataset from videos and CVAT XML.",
        default_events=EVENTS_ANOMALY,
        build_fn=build_dataset_anomaly,
        dataset_name="anomaly",
    )


if __name__ == "__main__":
    main()

