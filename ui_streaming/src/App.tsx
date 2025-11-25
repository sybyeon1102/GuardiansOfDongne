// src/App.tsx
// ----------------------------------------------------------------------
// 최종 완성본 (트래킹 + behavior + 메인비디오 + 그리드 + RightPanel 연동)
// ----------------------------------------------------------------------

import { useEffect, useMemo, useState } from "react";
import { Header } from "./components/Header";
import { Sidebar } from "./components/Sidebar";
import { VideoGrid } from "./components/VideoGrid";
import { MainVideoPlayer } from "./components/MainVideoPlayer";
import { RightPanel } from "./components/RightPanel";

// -------------------------------------------------------------
// 타입 정의
// -------------------------------------------------------------

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

// -------------------------------------------------------------
// Tracking 타입
// -------------------------------------------------------------

export interface TrackingBBox {
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface TrackingObject {
  global_id: string;
  local_track_id: number;
  label: string;
  confidence: number;
  bbox: TrackingBBox;
}

export interface TrackingSnapshot {
  camera_id: string;
  timestamp: number;
  frame_index: number | null;
  objects: TrackingObject[];
}

// -------------------------------------------------------------

const WARNING_THRESHOLD = 0.4;

// -------------------------------------------------------------
// 환경 변수
// -------------------------------------------------------------

const INFERENCE_BASE_URL =
  import.meta.env.VITE_INFERENCE_BASE_URL ?? "http://localhost:8000";
const AGENT_CODE =
  import.meta.env.VITE_AGENT_CODE ?? "agent-main-building-01";
const AGENT_BASE_URL =
  import.meta.env.VITE_AGENT_BASE_URL ?? "http://localhost:8001";

// -------------------------------------------------------------
// 메인 App
// -------------------------------------------------------------

export default function App() {
  // 카메라 목록
  const [videoFeeds, setVideoFeeds] = useState<CameraFeed[]>([]);
  const [mainCameraId, setMainCameraId] = useState<string | null>(null);
  const [draggedCameraId, setDraggedCameraId] = useState<string | null>(null);

  // Behavior 저장
  const [behaviorByCamera, setBehaviorByCamera] = useState<
    Record<string, BehaviorResult>
  >({});

  // Tracking 저장
  const [trackingByCamera, setTrackingByCamera] = useState<
    Record<string, TrackingSnapshot>
  >({});

  const [warningCameraIds, setWarningCameraIds] = useState<string[]>([]);
  const [selectedHasWarning, setSelectedHasWarning] = useState(false);

  // 연결 상태 관리
  const [lastPongAt, setLastPongAt] = useState<number | null>(null);
  const [connectionStatus, setConnectionStatus] =
    useState<ConnectionStatus>("ok");
  const [isDataStale, setIsDataStale] = useState(true);

  // RightPanel 값들
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
  // (1) Agent(8001) 에서 카메라 목록 가져오기
  // ----------------------------------------------------------------------

  useEffect(() => {
    const fetchCameras = async () => {
      try {
        const res = await fetch(`${AGENT_BASE_URL}/cameras`);
        if (!res.ok) throw new Error();

        const cams = await res.json();
        const feeds: CameraFeed[] = cams.map((cam: any) => ({
          cameraId: cam.id,
          name: cam.display_name ?? cam.id,
        }));

        setVideoFeeds(feeds);

        // 초기 메인 카메라 설정
        if (feeds.length > 0 && !mainCameraId) {
          setMainCameraId(feeds[0].cameraId);
        }
      } catch {
        console.warn("[Agent] failed to load camera list");
      }
    };

    fetchCameras();
    const id = setInterval(fetchCameras, 5000);
    return () => clearInterval(id);
  }, [mainCameraId]);

  // ----------------------------------------------------------------------
  // (2) 8000 behavior/latest_all 가져오기
  // ----------------------------------------------------------------------

  useEffect(() => {
    const url = `${INFERENCE_BASE_URL}/behavior/latest_all?agent_code=${AGENT_CODE}`;

    const fetchBehavior = async () => {
      try {
        const res = await fetch(url, { cache: "no-store" });
        if (!res.ok) throw new Error();
        const data: BehaviorResult[] = await res.json();

        setLastPongAt(Date.now());

        const byCam: Record<string, BehaviorResult> = {};
        data.forEach((item) => (byCam[item.camera_id] = item));
        setBehaviorByCamera(byCam);

        // 워닝 카메라
        const warns: string[] = [];
        for (const br of Object.values(byCam)) {
          if (Object.values(br.prob).some((v) => v >= WARNING_THRESHOLD)) {
            warns.push(br.camera_id);
          }
        }
        setWarningCameraIds(warns);

        // 로그 추가
        let most: BehaviorResult | null = null;
        for (const br of Object.values(byCam)) {
          if (!br.is_anomaly) continue;
          if (!most || br.top_prob > most.top_prob) most = br;
        }
        if (most) {
          const ts = new Date().toLocaleTimeString("ko-KR", {
            hour12: false,
          });
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
      } catch {
        console.warn("[Server] behavior fetch failed");
      }
    };

    fetchBehavior();
    const id = setInterval(fetchBehavior, 2000);
    return () => clearInterval(id);
  }, []);

  // ----------------------------------------------------------------------
  // (3) Agent(8001) tracking/latest_all 가져오기
  // ----------------------------------------------------------------------

  useEffect(() => {
    const fetchTracking = async () => {
      try {
        const res = await fetch(`${AGENT_BASE_URL}/tracking/latest_all`);
        if (!res.ok) return;

        const data: TrackingSnapshot[] = await res.json();
        const byCam: Record<string, TrackingSnapshot> = {};
        data.forEach((snap) => (byCam[snap.camera_id] = snap));

        setTrackingByCamera(byCam);
      } catch {
        console.warn("[Agent] tracking fetch failed");
      }
    };

    fetchTracking();
    const id = setInterval(fetchTracking, 500);
    return () => clearInterval(id);
  }, []);

  // ----------------------------------------------------------------------
  // ping/pong 상태 업데이트
  // ----------------------------------------------------------------------

  useEffect(() => {
    const timer = setInterval(() => {
      if (!lastPongAt) {
        setConnectionStatus("disconnected");
        setIsDataStale(true);
        return;
      }
      const diff = Date.now() - lastPongAt;
      setIsDataStale(diff > 3000);
      setConnectionStatus(diff > 10000 ? "disconnected" : "ok");
    }, 1000);
    return () => clearInterval(timer);
  }, [lastPongAt]);

  // ----------------------------------------------------------------------
  // (4) RightPanel = behavior → 메인 카메라 값 반영
  // ----------------------------------------------------------------------

  useEffect(() => {
    if (!mainCameraId) return;

    const br = behaviorByCamera[mainCameraId];
    if (!br) {
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
  // 메인 카메라 & 트래킹 선택
  // ----------------------------------------------------------------------

  const mainCamera = useMemo(() => {
    if (!mainCameraId) return null;
    return (
      videoFeeds.find((v) => v.cameraId === mainCameraId) ??
      videoFeeds[0] ??
      null
    );
  }, [mainCameraId, videoFeeds]);

  const selectedCameraId = mainCamera?.cameraId ?? null;

  const mainTracking =
    mainCameraId && trackingByCamera[mainCameraId]
      ? trackingByCamera[mainCameraId]
      : null;

  // ----------------------------------------------------------------------
  // 드래그 앤 드롭
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
        if (a >= 0 && b >= 0) {
          [arr[a], arr[b]] = [arr[b], arr[a]];
        }
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
  // 렌더링
  // ----------------------------------------------------------------------

  return (
    <div className="flex h-screen bg-gray-100">
      <Sidebar />
      <div className="flex-1 flex flex-col">
        <Header />

        <main className="flex-1 overflow-auto p-4">
          <div className="flex gap-4 h-full">
             {/* LEFT: 남은 공간 전체 사용 */}
          <div className="flex-1 flex flex-col gap-4 min-w-0">
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
                mjpegUrl={`${AGENT_BASE_URL}/streams/${mainCamera.cameraId}.mjpeg`}
                tracking={mainTracking}
              />
            )}
          </div>

          {/* RIGHT: 고정 크기 패널 */}
          <div className="w-[30vw] flex-shrink-0">
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
