// src/components/MainVideoPlayer.tsx
// ----------------------------------------------------------
// [기능 요약 - 전체 반영됨]
// - Tracking ON: /output/{cameraId}.jpg (추론 결과 이미지)
// - Tracking OFF: /stream/{cameraId} MJPEG 스트림
// - MJPEG 자동 재연결 Hook 적용
// - FPS 실시간 표시 (/meta)
// - 경고 상태 → 노란 테두리 (warning-border)
// - 메인 출력 선택 영상 → 인디고 테두리
// - 드래그 앤 드롭 지원
// - cameraId가 null 또는 빈 입력일 경우 → "No Signal" 빈 화면 표시 (신규 추가)
// ----------------------------------------------------------

import { useEffect, useState } from "react";
import { useMjpegStream } from "../hooks/useMjpegStream";

type MainVideoPlayerProps = {
  feedId: number;
  cameraId: string | null;      // null 대응 필요
  isWarning: boolean;
  isMainSelected: boolean;
  onDragStart: (id: number) => void;
  onDropOnMain: () => void;
};

// 메인 플레이어
export function MainVideoPlayer({
  feedId,
  cameraId,
  isWarning,
  isMainSelected,
  onDragStart,
  onDropOnMain,
}: MainVideoPlayerProps) {

  const [trackingEnabled, setTrackingEnabled] = useState(true);
  const [fps, setFps] = useState<number | null>(null);

  // cameraId가 null이면 스트림 URL 없음
  const trackingImageUrl = cameraId
    ? `http://localhost:8001/output/${cameraId}.jpg`
    : null;

  const originalMjpegUrl = cameraId
    ? `http://localhost:8001/stream/${cameraId}`
    : null;

  const { imgRef, handleError } = useMjpegStream(
    originalMjpegUrl ?? "",
    800
  );

  // FPS fetch
  useEffect(() => {
    if (!cameraId) return;

    const fetchFPS = async () => {
      try {
        const res = await fetch(`http://localhost:8001/meta/${cameraId}`);
        if (!res.ok) return;
        const data = await res.json();
        if (typeof data.fps === "number") setFps(data.fps);
      } catch {
        // ignore
      }
    };

    fetchFPS();
    const id = setInterval(fetchFPS, 1000);
    return () => clearInterval(id);

  }, [cameraId]);

  const borderClass = isWarning
    ? "warning-border"
    : isMainSelected
    ? "border-indigo-500"
    : "border-gray-300";

  return (
    <section className="bg-white rounded-xl p-4 shadow-sm">
      <div
        draggable
        className={`relative rounded-xl overflow-hidden border-4 ${borderClass}`}
        onDragStart={() => onDragStart(feedId)}
        onDrop={(e) => {
          e.preventDefault();
          onDropOnMain();
        }}
        onDragOver={(e) => e.preventDefault()}
      >

        {/* -------------------------------------------------
            CASE 1: cameraId 없음 → 빈 화면 출력
           ------------------------------------------------- */}
        {!cameraId && (
          <div className="w-full aspect-video bg-black flex items-center justify-center">
            <span className="text-gray-400 text-sm">No Signal</span>
          </div>
        )}

        {/* -------------------------------------------------
            CASE 2: Tracking ON → 추론 이미지 출력
           ------------------------------------------------- */}
        {cameraId && trackingEnabled && trackingImageUrl && (
          <div className="w-full aspect-video bg-black">
            <img
              src={`${trackingImageUrl}?t=${Date.now()}`}
              className="w-full h-full object-contain"
            />
          </div>
        )}

        {/* -------------------------------------------------
            CASE 3: Tracking OFF → 원본 MJPEG 스트림
           ------------------------------------------------- */}
        {cameraId && !trackingEnabled && originalMjpegUrl && (
          <div className="w-full aspect-video bg-black">
            <img
              ref={imgRef}
              onError={handleError}
              className="w-full h-full object-contain"
            />
          </div>
        )}

        {/* ----------- 카메라 라벨 ----------- */}
        {cameraId && (
          <div className="absolute top-2 left-2 px-2 py-1 text-xs bg-black/70 text-white rounded">
            {cameraId}
          </div>
        )}

        {/* ----------- FPS 표시 ----------- */}
        {cameraId && (
          <div className="absolute top-2 right-2 px-2 py-1 text-xs bg-black/70 text-white font-mono rounded">
            FPS: {fps ? fps.toFixed(1) : "--.-"}
          </div>
        )}

        {/* ----------- Tracking 버튼 ----------- */}
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
