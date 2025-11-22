// src/components/MainVideoPlayer.tsx

import { useEffect, useRef } from "react";
import { StreamVideo } from "./StreamVideo";

export function MainVideoPlayer({
  feedId,
  videoUrl,
  isPlaying,
  currentTime,
  isWarning,
  isMainSelected,
  onTimeUpdate,
  onDurationChange,
  onDragStart,
  onDropOnMain,
}) {
  const videoRef = useRef<HTMLVideoElement>(null);

  // 외부 재생/일시정지 제어 및 currentTime 동기화는
  // 직접 videoRef를 써야 하므로 별도 useEffect 유지

  // 재생/일시정지
  useEffect(() => {
    const v = videoRef.current;
    if (!v) return;
    if (isPlaying) {
      v.play().catch(() => {});
    } else {
      v.pause();
    }
  }, [isPlaying]);

  // currentTime 동기화
  useEffect(() => {
    const v = videoRef.current;
    if (!v) return;
    if (Math.abs(v.currentTime - currentTime) > 0.3) {
      v.currentTime = currentTime;
    }
  }, [currentTime]);

  const handleMeta = () => {
    const v = videoRef.current;
    if (!v) return;
    onDurationChange(v.duration || 0);
  };

  const handleTimeUpdate = () => {
    const v = videoRef.current;
    if (!v) return;
    onTimeUpdate(v.currentTime);
  };

  const borderClass = isWarning
    ? "warning-border"
    : isMainSelected
    ? "border-indigo-500"
    : "border-gray-300";

  return (
    <section
      className="bg-white rounded-xl p-4 shadow-sm"
      onDrop={(e) => {
        e.preventDefault();
        onDropOnMain();
      }}
      onDragOver={(e) => e.preventDefault()}
    >
      <div
        className={`relative rounded-xl overflow-hidden border-4 ${borderClass}`}
        draggable
        onDragStart={() => onDragStart(feedId)}
      >
        <div className="aspect-[4/3] w-full bg-black">
          <StreamVideo
            src={videoUrl}
            className="w-full h-full object-cover"
            autoPlay={isPlaying}
            loop
            muted
            onLoadedMetadata={handleMeta}
            onTimeUpdate={handleTimeUpdate}
            // ref를 직접 전달해야 currentTime 제어가 가능하므로,
            // StreamVideo 내부에서 videoRef를 forwardRef로 바꾸는 버전이 필요하면
            // 그때 리팩터링 가능. (지금은 autoPlay 위주로 유지)
          />
        </div>
      </div>
    </section>
  );
}
