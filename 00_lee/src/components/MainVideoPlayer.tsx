import { useEffect, useRef } from "react";
import Hls from "hls.js";

export function MainVideoPlayer({
  feedId,
  cameraId,
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
  const hlsRef = useRef<Hls | null>(null);

  // 프론트에서 받아오는 HLS URL
  // const hlsUrl = `https://pmhmdhwetxzhngkw.tunnel.elice.io/hls/${cameraId}/index.m3u8`;
  const hlsUrl = `http://localhost:8001/hls/${cameraId}/index.m3u8`;

  // 카메라 아이디가 바뀔 때마다 새로운 HLS 스트림 로딩
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    // 기존 인스턴스 제거
    if (hlsRef.current) {
      hlsRef.current.destroy();
      hlsRef.current = null;
    }

    // Hls.js 지원 브라우저
    if (Hls.isSupported()) {
      const hls = new Hls({ enableWorker: true, lowLatencyMode: true });
      hlsRef.current = hls;

      hls.loadSource(hlsUrl);
      hls.attachMedia(video);

      hls.on(Hls.Events.MANIFEST_PARSED, () => {
        if (isPlaying) video.play().catch(() => {});
      });

      hls.on(Hls.Events.ERROR, (event, data) => {
        console.warn("[HLS error]", event, data);
      });
    } else {
      // Safari 등의 네이티브 HLS 지원 브라우저
      video.src = hlsUrl;
      video.addEventListener("loadedmetadata", () => {
        if (isPlaying) video.play().catch(() => {});
      });
    }
  }, [cameraId]);

  // 재생/일시정지 제어
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    if (isPlaying) video.play().catch(() => {});
    else video.pause();
  }, [isPlaying]);

  const handleTimeUpdate = () => {
    const video = videoRef.current;
    if (!video) return;
    onTimeUpdate(video.currentTime);
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
          <video
            ref={videoRef}
            className="w-full h-full object-cover"
            muted
            onTimeUpdate={handleTimeUpdate}
            playsInline
          />
        </div>
      </div>
    </section>
  );
}
