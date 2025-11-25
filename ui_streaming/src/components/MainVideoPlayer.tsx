// src/components/MainVideoPlayer.tsx
// ----------------------------------------------------------
// [기능 요약]
// - Tracking ON: /output/{cameraId}.jpg (추론 결과 이미지)
// - Tracking OFF: Agent(8001)의 MJPEG 스트림 /streams/{cameraId}.mjpeg
// - MJPEG 자동 재연결 Hook (useMjpegStream)
// - FPS 실시간 표시 (/meta/{cameraId})
// - 경고 상태 → 노란 테두리
// ----------------------------------------------------------

import { useEffect, useState } from "react";
import { useMjpegStream } from "../hooks/useMjpegStream";

type MainVideoPlayerProps = {
  cameraId: string | null;
  isWarning: boolean;
  isMainSelected: boolean;
  onDragStart: (cameraId: string) => void;
  onDropOnMain: () => void;
  isDataStale: boolean;
};

export function MainVideoPlayer({
  cameraId,
  isWarning,
  isMainSelected,
  onDragStart,
  onDropOnMain,
  isDataStale,
}: MainVideoPlayerProps) {
  const [trackingEnabled, setTrackingEnabled] = useState(false);
  const [fps, setFps] = useState<number | null>(null);

  // 추론 서버: 8000
  const trackingImageUrl = cameraId
    ? `http://localhost:8000/output/${cameraId}.jpg`
    : null;

  // 🔥 변경 완료 — Agent의 MJPEG 스트림은 8001
  const originalMjpegUrl = cameraId
    ? `http://localhost:8001/streams/${cameraId}.mjpeg`
    : null;

  const { imgRef, handleError } = useMjpegStream(
    originalMjpegUrl ?? "",
    800
  );

  // // FPS 가져오기 (8001)
  // useEffect(() => {
  //   if (!cameraId) return;

  //   const fetchFPS = async () => {
  //     try {
  //       const res = await fetch(`http://localhost:8001/meta/${cameraId}`);
  //       if (!res.ok) return;
  //       const data = await res.json();
  //       if (typeof data.fps === "number") setFps(data.fps);
  //     } catch {
  //       // ignore
  //     }
  //   };

  //   fetchFPS();
  //   const id = setInterval(fetchFPS, 1000);
  //   return () => clearInterval(id);
  // }, [cameraId]);

  const borderClass = isWarning
    ? "warning-border"
    : isMainSelected
    ? "border-indigo-500"
    : "border-gray-300";

  return (
    <section className="bg-white rounded-xl p-4 shadow-sm">
      <div
        draggable={!!cameraId}
        className={`relative rounded-xl overflow-hidden border-4 ${borderClass}`}
        onDragStart={() => cameraId && onDragStart(cameraId)}
        onDrop={(e) => {
          e.preventDefault();
          onDropOnMain();
        }}
        onDragOver={(e) => e.preventDefault()}
      >
        {/* CASE 1: cameraId 없음 */}
        {!cameraId && (
          <div className="w-full aspect-video bg-black flex items-center justify-center">
            <span className="text-gray-400 text-sm">No Signal</span>
          </div>
        )}

        {/* CASE 2: Tracking ON → 추론 이미지 */}
        {cameraId && trackingEnabled && trackingImageUrl && (
          <div className="w-full aspect-video bg-black">
            <img
              src={`${trackingImageUrl}?t=${Date.now()}`}
              className="w-full h-full object-contain"
            />
          </div>
        )}

        {/* CASE 3: Tracking OFF → 원본 MJPEG */}
        {cameraId && !trackingEnabled && originalMjpegUrl && (
          <div className="w-full aspect-video bg-black">
            <img
              ref={imgRef}
              onError={handleError}
              className="w-full h-full object-contain"
            />
          </div>
        )}

        {/* 카메라 라벨 */}
        {cameraId && (
          <div className="absolute top-2 left-2 px-2 py-1 text-xs bg-black/70 text-white rounded">
            {cameraId}
          </div>
        )}

        {/* FPS 표시 */}
        {cameraId && (
          <div className="absolute top-2 right-2 px-2 py-1 text-xs bg-black/70 text-white font-mono rounded">
            FPS: {fps ? fps.toFixed(1) : "--.-"}
          </div>
        )}

        {/* 데이터 지연 표시 */}
        {isDataStale && (
          <div className="absolute bottom-2 left-2 px-2 py-1 text-xs bg-yellow-500 text-black rounded">
            Data Delay
          </div>
        )}

        {/* Tracking 토글 버튼 */}
        {cameraId && (
          <button
            onClick={() => setTrackingEnabled((v) => !v)}
            className="absolute bottom-2 right-2 px-3 py-1 bg-white/80 border rounded text-xs"
          >
            {trackingEnabled ? "Tracking ON" : "Tracking OFF"}
          </button>
        )}
      </div>
    </section>
  );
}
