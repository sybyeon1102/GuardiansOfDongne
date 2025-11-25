// src/App.tsx
// ----------------------------------------------------------
// [기능 요약]
// - /behavior/latest_all 를 2초마다 호출해 모든 카메라 추론값을 한 번에 가져옴
// - camera_id → BehaviorResult 맵으로 저장
// - 각 카메라마다 임계값( WARNING_THRESHOLD ) 초과 여부 계산
//   · warningCameraIds: 임계값 넘은 카메라 목록
//   · selectedHasWarning: 현재 선택된 카메라가 임계값 넘는지 여부
// - 선택된 카메라 기준으로 RightPanel 상태/Anomaly 값 반영
// - VideoGrid / MainVideoPlayer / RightPanel에 경고 상태를 전달해
//   테두리 깜빡이게 만듦
// - ping/pong 개념으로 lastPongAt 기반 연결/지연 상태 관리
// ----------------------------------------------------------

import { useEffect, useMemo, useState } from "react";
import { Header } from "./components/Header";
import { Sidebar } from "./components/Sidebar";
import { VideoGrid } from "./components/VideoGrid";
import { MainVideoPlayer } from "./components/MainVideoPlayer";
import { RightPanel } from "./components/RightPanel";

// ----------------------------------------------------------
// 타입 정의
// ----------------------------------------------------------

type VideoFeed = {
  id: number;
  name: string;
  cameraId: string;
};

export type Probabilities = {
  fall: number;
  abandon: number;
  broken: number;
  fight: number;
  fire: number;
  smoke: number;
  theft: number;
  weak_pedestrian: number;
};

export type EventLogEntry = {
  id: number;
  timestamp: string;
  cameraId: string | null;
  sourceId: string | null;
  topLabel: string;
  topProb: number;
};

type BehaviorResult = {
  agent_code: string;
  camera_id: string;
  source_id: string | null;
  window_index: number;
  window_start_ts: number | null;
  window_end_ts: number | null;
  is_anomaly: boolean;
  det_prob: number;
  top_label: string | null;
  top_prob: number;
  prob: Record<string, number>;
};

type ConnectionStatus = "ok" | "disconnected";

// 8개 라벨 공통 임계값
const WARNING_THRESHOLD = 0.4;

// ----------------------------------------------------------
// 서버 환경 설정 (.env 없으면 터널 주소 기본값 사용)
// ----------------------------------------------------------

const INFERENCE_BASE_URL =
  import.meta.env.VITE_INFERENCE_BASE_URL ??
  "https://cypxkxhgcjqyccmk.tunnel.elice.io";

const AGENT_CODE = import.meta.env.VITE_AGENT_CODE ?? "agent01";

// ----------------------------------------------------------
// 메인 App
// ----------------------------------------------------------

export default function App() {
  // -----------------------------------------
  // 카메라 목록
  // -----------------------------------------
  const [videoFeeds, setVideoFeeds] = useState<VideoFeed[]>([]);
  const [mainFeedId, setMainFeedId] = useState<number | null>(null);
  const [draggedFeedId, setDraggedFeedId] = useState<number | null>(null);

  // -----------------------------------------
  // 카메라별 추론 결과 (camera_id → BehaviorResult)
  // -----------------------------------------
  const [behaviorByCamera, setBehaviorByCamera] = useState<
    Record<string, BehaviorResult>
  >({});

  // -----------------------------------------
  // 임계값 넘은 카메라들 (VideoGrid 테두리 깜빡임용)
  // -----------------------------------------
  const [warningCameraIds, setWarningCameraIds] = useState<string[]>([]);

  // 현재 선택된 카메라가 임계값을 넘는지 여부 (MainVideo / RightPanel 테두리용)
  const [selectedHasWarning, setSelectedHasWarning] = useState<boolean>(false);

  // -----------------------------------------
  // 서버 연결/지연 상태
  // -----------------------------------------
  const [connectionStatus, setConnectionStatus] =
    useState<ConnectionStatus>("ok");
  const [isDataStale, setIsDataStale] = useState<boolean>(true);
  const [lastPongAt, setLastPongAt] = useState<number | null>(null);

  // -----------------------------------------
  // RightPanel 표시용 상태
  // -----------------------------------------
  const [probabilities, setProbabilities] = useState<Probabilities>({
    fall: 0,
    abandon: 0,
    broken: 0,
    fight: 0,
    fire: 0,
    smoke: 0,
    theft: 0,
    weak_pedestrian: 0,
  });

  const [topLabel, setTopLabel] = useState<string | null>(null);
  const [topProb, setTopProb] = useState<number | null>(null);
  const [lastCameraId, setLastCameraId] = useState<string | null>(null);
  const [lastSourceId, setLastSourceId] = useState<string | null>(null);
  const [eventLogs, setEventLogs] = useState<EventLogEntry[]>([]);

  // ----------------------------------------------------------
  // (1) /behavior/latest_all 폴링 = ping
  //  - 이 API가 "전체 카메라의 최신 추론값"을 한 번에 반환
  //  - 응답 성공 시 lastPongAt 갱신
  //  - camera_id → BehaviorResult 매핑
  //  - warningCameraIds 계산
  // ----------------------------------------------------------
  useEffect(() => {
    const baseUrl = INFERENCE_BASE_URL.replace(/\/+$/, "");

    const fetchLatestAll = async () => {
      try {
        const res = await fetch(
          `${baseUrl}/behavior/latest_all?agent_code=${encodeURIComponent(
            AGENT_CODE
          )}`,
          { cache: "no-store" }
        );

        if (!res.ok) throw new Error(`HTTP ${res.status}`);

        const data: BehaviorResult[] = await res.json();

        // pong 시각 갱신
        setLastPongAt(Date.now());

        // camera_id → BehaviorResult 맵 생성
        const byCam: Record<string, BehaviorResult> = {};
        for (const item of data) {
          if (item.camera_id) {
            byCam[item.camera_id] = item;
          }
        }
        setBehaviorByCamera(byCam);

        // --- (1) 카메라 목록 자동 생성 (최초 1회) ---
        setVideoFeeds((prev) => {
          if (prev.length > 0) return prev;

          const feeds: VideoFeed[] = Object.keys(byCam).map(
            (cameraId, idx) => ({
              id: idx,
              name: `Camera ${String(idx + 1).padStart(2, "0")}`,
              cameraId,
            })
          );

          if (feeds.length > 0 && mainFeedId == null) {
            setMainFeedId(feeds[0].id);
          }

          return feeds;
        });

        // --- (2) 각 카메라의 warning 여부 계산 (임계값 초과) ---
        const newWarningIds: string[] = [];
        for (const [cameraId, item] of Object.entries(byCam)) {
          const prob = item.prob ?? {};
          const anyOverThreshold = (
            Object.keys(prob) as (keyof Probabilities)[]
          ).some((key) => {
            const v = prob[key];
            return typeof v === "number" && v >= WARNING_THRESHOLD;
          });
          if (anyOverThreshold) {
            newWarningIds.push(cameraId);
          }
        }
        setWarningCameraIds(newWarningIds);

        // --- (3) 가장 anomaly가 높은 카메라를 이벤트 로그에 기록 (선택) ---
        let mostCritical: BehaviorResult | null = null;
        for (const item of Object.values(byCam)) {
          if (!item.is_anomaly) continue;
          if (!mostCritical || item.top_prob > mostCritical.top_prob) {
            mostCritical = item;
          }
        }

        if (mostCritical) {
          const now = new Date();
          const ts = now.toLocaleTimeString("ko-KR", { hour12: false });
          setEventLogs((prev) => {
            const newEntry: EventLogEntry = {
              id: prev.length + 1,
              timestamp: ts,
              cameraId: mostCritical.camera_id,
              sourceId: mostCritical.source_id,
              topLabel: mostCritical.top_label ?? "unknown",
              topProb: mostCritical.top_prob,
            };
            return [newEntry, ...prev].slice(0, 50);
          });
        }
      } catch (error) {
        console.error("Failed to fetch latest behavior:", error);
        // 실패 시 lastPongAt 갱신이 멈추고 아래 ping/pong 감시 로직이 상태를 바꿔줌
      }
    };

    fetchLatestAll();
    const interval = setInterval(fetchLatestAll, 2000);
    return () => clearInterval(interval);
  }, [mainFeedId]);

  // ----------------------------------------------------------
  // (2) ping/pong 감시
  //  - lastPongAt 기준으로 지연/오프라인 판단
  //  - 3초 초과 → isDataStale = true
  //  - 10초 초과 → connectionStatus = "disconnected"
  // ----------------------------------------------------------
  useEffect(() => {
    const timer = setInterval(() => {
      if (lastPongAt == null) {
        setIsDataStale(true);
        setConnectionStatus("disconnected");
        return;
      }
      const diff = Date.now() - lastPongAt;
      setIsDataStale(diff > 3000);

      if (diff > 10000) {
        setConnectionStatus("disconnected");
      } else {
        setConnectionStatus("ok");
      }
    }, 1000);

    return () => clearInterval(timer);
  }, [lastPongAt]);

  // ----------------------------------------------------------
  // (3) 선택된(메인) 카메라의 확률/상태를 RightPanel에 반영
  //  - probabilities, topLabel, topProb, lastCameraId, lastSourceId
  //  - selectedHasWarning: 선택된 카메라가 임계값 넘는지 여부
  // ----------------------------------------------------------
  useEffect(() => {
    if (mainFeedId == null || videoFeeds.length === 0) return;

    const feed = videoFeeds.find((f) => f.id === mainFeedId);
    if (!feed) return;

    const behavior = behaviorByCamera[feed.cameraId];
    if (!behavior) return;

    const prob = behavior.prob ?? {};

    const safe = (key: keyof Probabilities): number => {
      const v = prob[key];
      return typeof v === "number" ? v : 0;
    };

    const newProbs: Probabilities = {
      fall: safe("fall"),
      abandon: safe("abandon"),
      broken: safe("broken"),
      fight: safe("fight"),
      fire: safe("fire"),
      smoke: safe("smoke"),
      theft: safe("theft"),
      weak_pedestrian: safe("weak_pedestrian"),
    };

    setProbabilities(newProbs);
    setTopLabel(behavior.top_label);
    setTopProb(behavior.top_prob);
    setLastCameraId(behavior.camera_id);
    setLastSourceId(behavior.source_id ?? null);

    // 선택된 카메라가 임계값을 넘는지 계산
    const anyWarning = (Object.values(newProbs) as number[]).some(
      (v) => v >= WARNING_THRESHOLD
    );
    setSelectedHasWarning(anyWarning);
  }, [behaviorByCamera, mainFeedId, videoFeeds]);

  // ----------------------------------------------------------
  // (4) 메인에 표시할 Feed
  // ----------------------------------------------------------
  const mainFeed = useMemo(() => {
    if (mainFeedId == null) return videoFeeds[0] ?? null;
    return videoFeeds.find((f) => f.id === mainFeedId) ?? videoFeeds[0] ?? null;
  }, [mainFeedId, videoFeeds]);

  const selectedCameraId = mainFeed?.cameraId ?? null;

  // ----------------------------------------------------------
  // (5) Drag & Drop 로직
  // ----------------------------------------------------------
  const handleDragStart = (id: number) => setDraggedFeedId(id);

  const handleDropOnThumbnail = (targetId: number) => {
    if (draggedFeedId == null || draggedFeedId === targetId) return;

    if (draggedFeedId === mainFeedId) {
      setMainFeedId(targetId);
    } else if (targetId === mainFeedId) {
      setMainFeedId(draggedFeedId);
    } else {
      setVideoFeeds((prev) => {
        const arr = [...prev];
        const fromIndex = arr.findIndex((f) => f.id === draggedFeedId);
        const toIndex = arr.findIndex((f) => f.id === targetId);
        if (fromIndex === -1 || toIndex === -1) return prev;
        [arr[fromIndex], arr[toIndex]] = [arr[toIndex], arr[fromIndex]];
        return arr;
      });
    }

    setDraggedFeedId(null);
  };

  const handleDropOnMain = () => {
    if (draggedFeedId == null) return;
    setMainFeedId(draggedFeedId);
    setDraggedFeedId(null);
  };

  // ----------------------------------------------------------
  // 렌더링
  // ----------------------------------------------------------
  return (
    <div className="flex h-screen bg-gray-100">
      <Sidebar />

      <div className="flex-1 flex flex-col">
        <Header />

        <main className="flex-1 overflow-auto p-4">
          <div className="flex gap-4 h-full w-full items-start">
            {/* LEFT SIDE : VideoGrid + MainVideoPlayer */}
            <div className="flex-[2.5] flex flex-col gap-4">
              <VideoGrid
                videos={videoFeeds}
                mainFeedId={mainFeedId}
                warningCameraIds={warningCameraIds}
                onSelectVideo={setMainFeedId}
                onDragStart={handleDragStart}
                onDropOnThumbnail={handleDropOnThumbnail}
                isDataStale={isDataStale}
              />

              {mainFeed && (
                <MainVideoPlayer
                  feedId={mainFeed.id}
                  cameraId={mainFeed.cameraId}
                  isWarning={selectedHasWarning}
                  isMainSelected={true}
                  onDragStart={handleDragStart}
                  onDropOnMain={handleDropOnMain}
                  isDataStale={isDataStale}
                />
              )}
            </div>

            {/* RIGHT SIDE : 상태 패널 */}
            <div className="flex-[1.2] flex-shrink-0">
              <RightPanel
                selectedCameraName={mainFeed?.name ?? null}
                selectedCameraId={selectedCameraId}
                probabilities={probabilities}
                connectionStatus={connectionStatus}
                isDataStale={isDataStale}
                topLabel={topLabel}
                topProb={topProb}
                lastCameraId={lastCameraId}
                lastSourceId={lastSourceId}
                eventLogs={eventLogs}
                selectedHasWarning={selectedHasWarning}
              />
            </div>
          </div>
        </main>
      </div>
    </div>
  );
}
