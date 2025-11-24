import { useState, useEffect } from "react";
import { Header } from "./components/Header";
import { Sidebar } from "./components/Sidebar";
import { VideoGrid } from "./components/VideoGrid";
import { MainVideoPlayer } from "./components/MainVideoPlayer";
import { VideoControls } from "./components/VideoControls";
import { RightPanel } from "./components/RightPanel";

type VideoFeed = {
  id: number;
  name: string;
  videoUrl: string;
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

export default function App() {
  // warning 임계값
  const WARNING_THRESHOLD = 0.4;

  // ---------- CAMERA FEEDS ----------
  const [videoFeeds] = useState<VideoFeed[]>([
    {
      id: 0,
      name: "Camera 01",
      cameraId: "cam01",
    },
    {
      id: 1,
      name: "Camera 02",
      cameraId: "cam02",
    },
    {
      id: 2,
      name: "Camera 03",
      cameraId: "cam03",
    },
  ]);

  const [mainFeedId, setMainFeedId] = useState<number>(0);
  const [draggedFeedId, setDraggedFeedId] = useState<number | null>(null);

  // 현재 warning 상태인 camera_id (예: "cam01")
  const [activeWarningCamera, setActiveWarningCamera] = useState<string | null>(
    null
  );

  // ---------- VIDEO PLAYER CONTROL ----------
  const [isPlaying, setIsPlaying] = useState<boolean>(true);
  const [isRecording, setIsRecording] = useState<boolean>(false);
  const [currentTime, setCurrentTime] = useState<number>(0);
  const [duration, setDuration] = useState<number>(0);

  // ---------- RIGHT PANEL STATES ----------
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

  const [connectionStatus] = useState<"ok" | "disconnected">("ok");

  const [topLabel, setTopLabel] = useState<string | null>(null);
  const [topProb, setTopProb] = useState<number | null>(null);

  const [lastCameraId, setLastCameraId] = useState<string | null>("cam01");
  const [lastSourceId, setLastSourceId] = useState<string | null>(null);

  const [eventLogs, setEventLogs] = useState<EventLogEntry[]>([]);

  // ----------------------------------------------------------------------------------
  // 테스트용 랜덤 확률 (나중에 백엔드로 대체)
  // ----------------------------------------------------------------------------------
  useEffect(() => {
    const interval = setInterval(() => {
      const p: Probabilities = {
        fall: Math.random() * 0.3,
        abandon: Math.random() * 0.2,
        broken: Math.random() * 0.1,
        fight: Math.random() * 0.25,
        fire: Math.random() * 0.5,
        smoke: Math.random() * 0.2,
        theft: Math.random() * 0.3,
        weak_pedestrian: Math.random() * 0.4,
      };

      setProbabilities(p);

      const entries = Object.entries(p);
      const [maxLabel, maxProb] = entries.reduce((a, b) =>
        a[1] > b[1] ? a : b
      );

      setTopLabel(maxLabel);
      setTopProb(maxProb);

      // 테스트용 camera_id 고정
      const fakeCameraId = "cam01";
      setLastCameraId(fakeCameraId);
      setLastSourceId("video_2025-11-21_01.mp4");

      // warning 임계치 적용
      if (maxProb >= WARNING_THRESHOLD) {
        setActiveWarningCamera(fakeCameraId);
      } else {
        setActiveWarningCamera(null);
      }

      // 이벤트 로그 예시 (원하면 활성화)
      // const now = new Date();
      // const ts = now.toLocaleTimeString("ko-KR", { hour12: false });
      // setEventLogs((prev) => {
      //   const newEntry: EventLogEntry = {
      //     id: prev.length + 1,
      //     timestamp: ts,
      //     cameraId: fakeCameraId,
      //     sourceId: "video_2025-11-21_01.mp4",
      //     topLabel: maxLabel,
      //     topProb: maxProb,
      //   };
      //   return [newEntry, ...prev].slice(0, 50);
      // });
    }, 2000);

    return () => clearInterval(interval);
  }, []);

  // ================================================================================

  // ---------- DRAG & DROP ----------
  const handleDragStart = (id: number) => setDraggedFeedId(id);

  const handleDropOnThumbnail = (targetId: number) => {
    if (draggedFeedId == null || draggedFeedId === targetId) return;

    if (draggedFeedId === mainFeedId) {
      setMainFeedId(targetId);
    } else if (targetId === mainFeedId) {
      setMainFeedId(draggedFeedId);
    }

    setDraggedFeedId(null);
  };

  const handleDropOnMain = () => {
    if (draggedFeedId == null) return;
    setMainFeedId(draggedFeedId);
    setDraggedFeedId(null);
  };

  const mainFeed = videoFeeds.find((v) => v.id === mainFeedId)!;

  // activeWarningCamera → warningFeed 찾기
  const warningFeed =
    activeWarningCamera != null
      ? videoFeeds.find(
          (v) => v.name.replace("Camera ", "cam") === activeWarningCamera
        ) ?? null
      : null;

  // mainFeedId → camera_id 형태로 변환 (cam01, cam02, cam03)
  const mainCameraId = `cam0${mainFeedId + 1}`;

  return (
    <div className="flex h-screen bg-gray-100">
      <Sidebar />

      <div className="flex-1 flex flex-col">
        <Header />

        <main className="flex-1 overflow-auto p-4">
          <div className="flex gap-4 h-full items-start">
            {/* LEFT SIDE */}
            <div className="flex-[2.5] flex flex-col gap-4">
              <VideoGrid
                videos={videoFeeds}
                mainFeedId={mainFeedId}
                activeWarningCamera={activeWarningCamera}
                onSelectVideo={setMainFeedId}
                onDragStart={handleDragStart}
                onDropOnThumbnail={handleDropOnThumbnail}
              />

              <MainVideoPlayer
                feedId={mainFeed.id}
                cameraId={mainFeed.cameraId}
                isPlaying={isPlaying}
                currentTime={currentTime}
                isWarning={activeWarningCamera === mainFeed.cameraId}
                isMainSelected={true}
                onTimeUpdate={setCurrentTime}
                onDurationChange={setDuration}
                onDragStart={handleDragStart}
                onDropOnMain={handleDropOnMain}
              />

              <VideoControls
                isPlaying={isPlaying}
                isRecording={isRecording}
                currentTime={currentTime}
                duration={duration}
                onPlayPause={() => setIsPlaying((v) => !v)}
                onStop={() => {
                  setIsPlaying(false);
                  setCurrentTime(0);
                }}
                onSeek={setCurrentTime}
                onRecordToggle={() => setIsRecording((v) => !v)}
                onDownload={() => alert("Download 준비중")}
              />
            </div>

            {/* RIGHT SIDE */}
            <div className="flex-[1.2] flex-shrink-0">
              <RightPanel
                warningVideoUrl={warningFeed ? warningFeed.videoUrl : ""}
                activeWarningCamera={activeWarningCamera}
                probabilities={probabilities}
                connectionStatus={connectionStatus}
                topLabel={topLabel}
                topProb={topProb}
                lastCameraId={lastCameraId}
                lastSourceId={lastSourceId}
                eventLogs={eventLogs}
                warningFeedName={warningFeed ? warningFeed.name : null}
              />
            </div>
          </div>
        </main>
      </div>
    </div>
  );
}
