// src/components/VideoGrid.tsx

import Hls from "hls.js";
import { useEffect, useRef } from "react";

export function VideoGrid({
  videos,
  mainFeedId,
  activeWarningCamera,
  onSelectVideo,
  onDragStart,
  onDropOnThumbnail,
}) {
  return (
    <section className="bg-white rounded-xl p-4 shadow-sm">
      <h2 className="text-sm text-gray-700 mb-2">Live Cameras</h2>

      <div className="grid grid-cols-3 gap-4">
        {videos.map((video) => {
          const cameraId = video.cameraId;
          const isWarning = cameraId === activeWarningCamera;
          const isMain = video.id === mainFeedId;

          return (
            <HlsThumbnail
              key={video.id}
              video={video}
              cameraId={cameraId}
              isWarning={isWarning}
              isMain={isMain}
              onSelectVideo={onSelectVideo}
              onDragStart={onDragStart}
              onDropOnThumbnail={onDropOnThumbnail}
            />
          );
        })}
      </div>
    </section>
  );
}

function HlsThumbnail({
  video,
  cameraId,
  isWarning,
  isMain,
  onSelectVideo,
  onDragStart,
  onDropOnThumbnail,
}) {
  const ref = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    const videoEl = ref.current;
    if (!videoEl) return;

    // 🔥 수정된 부분: url → hlsUrl
    // const hlsUrl = `https://pmhmdhwetxzhngkw.tunnel.elice.io/hls/${cameraId}/index.m3u8`;
    const hlsUrl = `http://localhost:8001/hls/${cameraId}/index.m3u8`;

    let hls: Hls | null = null;

    if (Hls.isSupported()) {
      hls = new Hls({ enableWorker: true });

      // 🔥 여기 수정됨
      hls.loadSource(hlsUrl);
      hls.attachMedia(videoEl);
    } else {
      // 🔥 Safari 네이티브 HLS 지원용
      videoEl.src = hlsUrl;
    }

    return () => {
      if (hls) hls.destroy();
    };
  }, [cameraId]);

  return (
    <div
      draggable
      onClick={() => onSelectVideo(video.id)}
      onDragStart={() => onDragStart(video.id)}
      onDrop={(e) => {
        e.preventDefault();
        onDropOnThumbnail(video.id);
      }}
      onDragOver={(e) => e.preventDefault()}
      className={`relative rounded-xl overflow-hidden cursor-pointer border-4 ${
        isWarning
          ? "warning-border"
          : isMain
          ? "border-indigo-500"
          : "border-gray-300"
      }`}
    >
      <div className="aspect-video w-full bg-black">
        <video ref={ref} className="w-full h-full object-cover" muted autoPlay playsInline />
      </div>

      <div className="absolute top-2 left-2 px-2 py-1 text-[10px] bg-black/70 text-white rounded">
        {video.name}
      </div>

      {isWarning && (
        <div className="absolute top-2 right-2 px-2 py-1 text-[10px] rounded-full bg-yellow-400 text-gray-900 font-semibold">
          WARNING
        </div>
      )}
    </div>
  );
}
