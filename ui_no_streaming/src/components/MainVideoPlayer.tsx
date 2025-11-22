import { useEffect, useRef } from "react";

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

  // 영상 URL 변경 → 새로 로드
  useEffect(() => {
    const v = videoRef.current;
    if (!v) return;

    v.pause();
    v.src = videoUrl;
    v.load();
    v.currentTime = 0;
    onTimeUpdate(0);

    if (isPlaying) v.play().catch(() => {});
  }, [videoUrl]);

  // 재생/일시정지
  useEffect(() => {
    const v = videoRef.current;
    if (!v) return;
    isPlaying ? v.play().catch(() => {}) : v.pause();
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
    ? "warning-border" // VideoGrid와 동일한 경고 테두리 (깜빡임)
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
          <video
            ref={videoRef}
            className="w-full h-full object-cover"
            muted
            loop
            onLoadedMetadata={handleMeta}
            onTimeUpdate={handleTimeUpdate}
          />
        </div>
      </div>
    </section>
  );
}
