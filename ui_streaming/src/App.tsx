// src/App.tsx
// ----------------------------------------------------------------------
// 새 구조 (2번):
// - 8001(Agent) 에서 카메라 목록 호출 → 즉시 영상 표시
// - 8000(Server) 은 behavior 값만 가져옴 (없으면 값 0%)
// - /behavior/latest_all 값은 카메라 목록과 독립 동작
// ----------------------------------------------------------------------

import { useEffect, useMemo, useState } from "react";
import { Header } from "./components/Header";
import { Sidebar } from "./components/Sidebar";
import { VideoGrid } from "./components/VideoGrid";
import { MainVideoPlayer } from "./components/MainVideoPlayer";
import { RightPanel } from "./components/RightPanel";

// ----------------------------------------------------------------------
// 타입들
// ----------------------------------------------------------------------

type CameraFeed = {
  cameraId: string;
  name: string;
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

const WARNING_THRESHOLD = 0.4;

// ----------------------------------------------------------------------
// 환경 변수
// ----------------------------------------------------------------------

const INFERENCE_BASE_URL =
  import.meta.env.VITE_INFERENCE_BASE_URL ?? "http://localhost:8000";
const AGENT_CODE =
  import.meta.env.VITE_AGENT_CODE ?? "agent-main-building-01";

// ----------------------------------------------------------------------
// 메인 App
// ----------------------------------------------------------------------

export default function App() {
  // 카메라 목록
  const [videoFeeds, setVideoFeeds] = useState<CameraFeed[]>([]);
  const [mainCameraId, setMainCameraId] = useState<string | null>(null);
  const [draggedCameraId, setDraggedCameraId] = useState<string | null>(null);

  // Behavior 값 저장
  const [behaviorByCamera, setBehaviorByCamera] = useState<
    Record<string, BehaviorResult>
  >({});

  const [warningCameraIds, setWarningCameraIds] = useState<string[]>([]);
  const [selectedHasWarning, setSelectedHasWarning] = useState(false);

  // 서버 연결/지연 상태
  const [connectionStatus, setConnectionStatus] =
    useState<ConnectionStatus>("ok");
  const [isDataStale, setIsDataStale] = useState(true);
  const [lastPongAt, setLastPongAt] = useState<number | null>(null);

  // RightPanel 값
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

  // ----------------------------------------------------------------------
  // (1) 8001 에이전트에서 카메라 목록 가져오기
  // ----------------------------------------------------------------------

  useEffect(() => {
    const fetchCameras = async () => {
      try {
        const res = await fetch("http://localhost:8001/cameras");
        if (!res.ok) throw new Error("Fail");

        const cams = await res.json();
        const feeds: CameraFeed[] = cams.map((cam: any) => ({
          cameraId: cam.id,
          name: cam.display_name ?? cam.id,
        }));

        setVideoFeeds(feeds);

        // 기본 메인 카메라 선택
        if (feeds.length > 0 && !mainCameraId) {
          setMainCameraId(feeds[0].cameraId);
        }
      } catch (err) {
        console.error("Failed to load cameras:", err);
      }
    };

    fetchCameras();
    const id = setInterval(fetchCameras, 5000);
    return () => clearInterval(id);
  }, []);

  // ----------------------------------------------------------------------
  // (2) 8000 에서 Behavior 값 가져오기
  // ----------------------------------------------------------------------

  useEffect(() => {
    const url = `${INFERENCE_BASE_URL}/behavior/latest_all?agent_code=${encodeURIComponent(
      AGENT_CODE
    )}`;

    const fetchBehavior = async () => {
      try {
        const res = await fetch(url, { cache: "no-store" });
        if (!res.ok) throw new Error("HTTP error");

        const data: BehaviorResult[] = await res.json();

        setLastPongAt(Date.now());

        const byCam: Record<string, BehaviorResult> = {};
        for (const item of data) byCam[item.camera_id] = item;
        setBehaviorByCamera(byCam);

        // 워닝 카메라 계산
        const wids: string[] = [];
        for (const br of Object.values(byCam)) {
          const anyOver = Object.values(br.prob).some(
            (v) => v >= WARNING_THRESHOLD
          );
          if (anyOver) wids.push(br.camera_id);
        }
        setWarningCameraIds(wids);

        // 가장 위험한 이벤트 로그
        let most: BehaviorResult | null = null;
        for (const br of Object.values(byCam)) {
          if (!br.is_anomaly) continue;
          if (!most || br.top_prob > most.top_prob) most = br;
        }
        if (most) {
          const now = new Date();
          const ts = now.toLocaleTimeString("ko-KR", { hour12: false });
          setEventLogs((prev) => [
            {
              id: prev.length + 1,
              timestamp: ts,
              cameraId: most.camera_id,
              sourceId: most.source_id,
              topLabel: most.top_label ?? "unknown",
              topProb: most.top_prob,
            },
            ...prev,
          ]);
        }
      } catch (err) {
        console.warn("Behavior fetch failed (server might be off)");
      }
    };

    fetchBehavior();
    const id = setInterval(fetchBehavior, 2000);
    return () => clearInterval(id);
  }, []);

  // ----------------------------------------------------------------------
  // ping/pong → 연결 상태 표시
  // ----------------------------------------------------------------------

  useEffect(() => {
    const timer = setInterval(() => {
      if (!lastPongAt) {
        setIsDataStale(true);
        setConnectionStatus("disconnected");
        return;
      }
      const diff = Date.now() - lastPongAt;
      setIsDataStale(diff > 3000);
      setConnectionStatus(diff > 10000 ? "disconnected" : "ok");
    }, 1000);
    return () => clearInterval(timer);
  }, [lastPongAt]);

  // ----------------------------------------------------------------------
  // (3) RightPanel에 선택된 카메라 값 반영
  // ----------------------------------------------------------------------

  useEffect(() => {
    if (!mainCameraId) return;

    const br = behaviorByCamera[mainCameraId];
    if (!br) {
      // behavior 없음 → 값 0 유지
      setProbabilities({
        fall: 0,
        abandon: 0,
        broken: 0,
        fight: 0,
        fire: 0,
        smoke: 0,
        theft: 0,
        weak_pedestrian: 0,
      });
      setTopLabel(null);
      setTopProb(null);
      setLastCameraId(mainCameraId);
      setLastSourceId(null);
      setSelectedHasWarning(false);
      return;
    }

    const safe = (k: keyof Probabilities) => br.prob[k] ?? 0;

    const probs: Probabilities = {
      fall: safe("fall"),
      abandon: safe("abandon"),
      broken: safe("broken"),
      fight: safe("fight"),
      fire: safe("fire"),
      smoke: safe("smoke"),
      theft: safe("theft"),
      weak_pedestrian: safe("weak_pedestrian"),
    };

    setProbabilities(probs);
    setTopLabel(br.top_label);
    setTopProb(br.top_prob);
    setLastCameraId(br.camera_id);
    setLastSourceId(br.source_id);

    setSelectedHasWarning(
      Object.values(probs).some((v) => v >= WARNING_THRESHOLD)
    );
  }, [mainCameraId, behaviorByCamera]);

  // ----------------------------------------------------------------------
  // (4) 메인 선택 카메라 찾기
  // ----------------------------------------------------------------------

  const mainCamera = useMemo(() => {
    if (!mainCameraId) return null;
    return (
      videoFeeds.find((c) => c.cameraId === mainCameraId) ??
      videoFeeds[0] ??
      null
    );
  }, [mainCameraId, videoFeeds]);

  const selectedCameraId = mainCamera?.cameraId ?? null;

  // ----------------------------------------------------------------------
  // DnD
  // ----------------------------------------------------------------------

  const handleDragStart = (camId: string) => setDraggedCameraId(camId);

  const handleDropOnThumbnail = (targetId: string) => {
    if (!draggedCameraId || draggedCameraId === targetId) return;

    if (draggedCameraId === mainCameraId) {
      setMainCameraId(targetId);
    } else if (targetId === mainCameraId) {
      setMainCameraId(draggedCameraId);
    } else {
      setVideoFeeds((prev) => {
        const arr = [...prev];
        const a = arr.findIndex((c) => c.cameraId === draggedCameraId);
        const b = arr.findIndex((c) => c.cameraId === targetId);
        if (a >= 0 && b >= 0) [arr[a], arr[b]] = [arr[b], arr[a]];
        return arr;
      });
    }

    setDraggedCameraId(null);
  };

  const handleDropOnMain = () => {
    if (draggedCameraId) {
      setMainCameraId(draggedCameraId);
      setDraggedCameraId(null);
    }
  };

  // ----------------------------------------------------------------------
  // Render
  // ----------------------------------------------------------------------

  return (
    <div className="flex h-screen bg-gray-100">
      <Sidebar />
      <div className="flex-1 flex flex-col">
        <Header />

        <main className="flex-1 overflow-auto p-4">
          <div className="flex gap-4 h-full">
            {/* LEFT */}
            <div className="flex-[2.5] flex flex-col gap-4">
              <VideoGrid
                cameras={videoFeeds}
                mainCameraId={mainCameraId}
                warningCameraIds={warningCameraIds}
                onSelectCamera={setMainCameraId}
                onDragStart={handleDragStart}
                onDropOnThumbnail={handleDropOnThumbnail}
                isDataStale={isDataStale}
              />

              {mainCamera && (
                <MainVideoPlayer
                  cameraId={mainCamera.cameraId}
                  isWarning={selectedHasWarning}
                  isMainSelected={true}
                  onDragStart={handleDragStart}
                  onDropOnMain={handleDropOnMain}
                  isDataStale={isDataStale}
                />
              )}
            </div>

            {/* RIGHT */}
            <div className="flex-[1.2]">
              <RightPanel
                selectedCameraName={mainCamera?.name ?? null}
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
