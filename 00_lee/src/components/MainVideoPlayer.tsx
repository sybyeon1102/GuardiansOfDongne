import { useEffect, useRef } from "react";

export function MainVideoPlayer({
  feedId,
  videoUrl,
  isPlaying,
  currentTime,
  isWarning,
  onTimeUpdate,
  onDurationChange,
  onDragStart,
  onDropOnMain,
}) {
  const videoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    if (isPlaying) video.play();
    else video.pause();

    video.currentTime = currentTime;
  }, [isPlaying, currentTime, videoUrl]);

  const handleTimeUpdate = () => {
    const video = videoRef.current;
    if (!video) return;
    onTimeUpdate(video.currentTime);
  };

  const handleLoadedMetadata = () => {
    const video = videoRef.current;
    if (!video) return;
    onDurationChange(video.duration || 0);
  };

  return (
    <section
      className={`relative bg-white rounded-xl shadow-sm overflow-hidden border-4 ${
        isWarning ? "warning-border" : "border-gray-300"
      }`}
      onDrop={(e) => {
        e.preventDefault();
        onDropOnMain();
      }}
      onDragOver={(e) => e.preventDefault()}
    >
      <div className="aspect-video w-full bg-black">
        <video
          ref={videoRef}
          src={videoUrl}
          className="w-full h-full object-contain"
          muted
          loop
          draggable
          onDragStart={() => onDragStart(feedId)}
          onTimeUpdate={handleTimeUpdate}
          onLoadedMetadata={handleLoadedMetadata}
        />
      </div>
    </section>
  );
}
