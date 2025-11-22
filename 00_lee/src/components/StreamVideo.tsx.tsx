// src/components/StreamVideo.tsx

import { useEffect, useRef } from "react";
import Hls from "hls.js";

type StreamVideoProps = {
  src: string;
  autoPlay?: boolean;
  loop?: boolean;
  muted?: boolean;
  className?: string;
  onLoadedMetadata?: () => void;
  onTimeUpdate?: () => void;
};

export function StreamVideo({
  src,
  autoPlay = true,
  loop = true,
  muted = true,
  className,
  onLoadedMetadata,
  onTimeUpdate,
}: StreamVideoProps) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const hlsRef = useRef<Hls | null>(null);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    // 이전 HLS 인스턴스 정리
    if (hlsRef.current) {
      hlsRef.current.destroy();
      hlsRef.current = null;
    }

    const isHls = src.endsWith(".m3u8");

    if (isHls && Hls.isSupported()) {
      const hls = new Hls();
      hlsRef.current = hls;
      hls.loadSource(src);
      hls.attachMedia(video);

      hls.on(Hls.Events.MANIFEST_PARSED, () => {
        if (autoPlay) {
          video.play().catch(() => {});
        }
      });
    } else {
      // mp4 같은 일반 src
      video.src = src;
      video.load();
      if (autoPlay) {
        video.play().catch(() => {});
      }
    }

    return () => {
      if (hlsRef.current) {
        hlsRef.current.destroy();
        hlsRef.current = null;
      }
    };
  }, [src, autoPlay]);

  return (
    <video
      ref={videoRef}
      className={className}
      muted={muted}
      loop={loop}
      onLoadedMetadata={onLoadedMetadata}
      onTimeUpdate={onTimeUpdate}
    />
  );
}
