// src/App.tsx

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
  // ---------- CAMERA FEEDS ----------
  const [videoFeeds] = useState<VideoFeed[]>([
    {
      id: 0,
      name: "Camera 01",
      videoUrl: "/videos/Camera_01_fight.mp4",
    },
    {
      id: 1,
      name: "Camera 02",
      videoUrl: "/videos/Camera_02_theaf.mp4",
    },
    {
      id: 2,
      name: "Camera 03",
      videoUrl: "/videos/Camera_03_smoke.mp4",
    },
  ]);

  const [mainFeedId, setMainFeedId] = useState<number>(0);
  const [warningFeedId, setWarningFeedId] = useState<number | null>(2);
  const [draggedFeedId, setDraggedFeedId] = useState<number | null>(null);

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
  // ---------- TEST RANDOM PROBABILITIES (현재 테스트용, 나중에 백엔드로 대체 가능)
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

      if (maxProb > 0.6) {
        setWarningFeedId(2);
      }
    }, 2000);

    return () => clearInterval(interval);
  }, []);



  // ================================================================================
  // 실제 백엔드 연결 시 여기에서 데이터를 받으면 됨 (WebSocket 또는 HTTP Polling)
  // ================================================================================

  // --------------------------
  // ① WebSocket 연결 예시
  // --------------------------
  /*
  useEffect(() => {
    const ws = new WebSocket("ws://localhost:8000/ws");

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);

      // 확률 업데이트
      setProbabilities(data.prob);

      setTopLabel(data.top_label);
      setTopProb(data.top_prob);

      // 카메라 / 소스 ID 업데이트
      setLastCameraId(data.camera_id ?? null);
      setLastSourceId(data.source_id ?? null);

      // Warning video 표시
      if (data.is_anomaly) {
        const idx = videoFeeds.findIndex((v) =>
          v.name.includes(data.camera_id)
        );
        if (idx >= 0) setWarningFeedId(idx);
      }

      // 이벤트 로그 기록
      setEventLogs((prev) => {
        const now = new Date();
        const ts = now.toLocaleTimeString("ko-KR", { hour12: false });

        const newEntry = {
          id: prev.length + 1,
          timestamp: ts,
          cameraId: data.camera_id,
          sourceId: data.source_id,
          topLabel: data.top_label,
          topProb: data.top_prob,
        };

        return [newEntry, ...prev].slice(0, 50);
      });
    };

    return () => ws.close();
  }, []);
  */


  // --------------------------
  // ② HTTP Polling 방식 예시
  // --------------------------
  /*
  useEffect(() => {
    const interval = setInterval(async () => {
      const res = await fetch("http://localhost:8000/predict");
      const data = await res.json();

      setProbabilities(data.prob);
      setTopLabel(data.top_label);
      setTopProb(data.top_prob);
      setLastCameraId(data.camera_id);
      setLastSourceId(data.source_id);

      if (data.is_anomaly) {
        const idx = videoFeeds.findIndex((v) =>
          v.name.includes(data.camera_id)
        );
        if (idx >= 0) setWarningFeedId(idx);
      }
    }, 500);

    return () => clearInterval(interval);
  }, []);
  */

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
  const warningFeed =
    warningFeedId != null ? videoFeeds.find((v) => v.id === warningFeedId) : null;

  return (
    <div className="flex h-screen bg-gray-100">
      <Sidebar />

      <div className="flex-1 flex flex-col">
        <Header />

        <main className="flex-1 overflow-auto p-4">
          {/* ⭐ VideoGrid와 RightPanel 상단 위치 일치 */}
          <div className="flex gap-4 h-full items-start">

            {/* LEFT SIDE (2.5 ratio) */}
            <div className="flex-[2.5] flex flex-col gap-4">

              <VideoGrid
                videos={videoFeeds}
                mainFeedId={mainFeedId}
                warningFeedId={warningFeedId}
                onSelectVideo={setMainFeedId}
                onDragStart={handleDragStart}
                onDropOnThumbnail={handleDropOnThumbnail}
              />

              <MainVideoPlayer
                feedId={mainFeed.id}
                videoUrl={mainFeed.videoUrl}
                isPlaying={isPlaying}
                currentTime={currentTime}
                isWarning={warningFeedId === mainFeedId}
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

            {/* RIGHT SIDE (1.2 ratio) */}
            <div className="flex-[1.2] flex-shrink-0">
              <RightPanel
                warningVideoUrl={warningFeed ? warningFeed.videoUrl : ""}
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
