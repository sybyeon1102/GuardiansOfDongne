import { useEffect, useRef } from "react";
import Hls from "hls.js";

// 외부에서 받아오는 Props 정의
// src: HLS 스트림 URL
// autoPlay: 자동 재생 여부 (기본 true)
// muted: 음소거 여부 (기본 true — 자동재생하려면 필요)
type Props = {
  src: string;
  autoPlay?: boolean;
  muted?: boolean;
};

// 재사용 가능한 HLS 영상 플레이어 컴포넌트
export default function HlsPlayer({
  src,
  autoPlay = true,
  muted = true,
}: Props) {
  
  // video 태그를 참조하기 위한 Ref
  const videoRef = useRef<HTMLVideoElement | null>(null);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    // 1) 브라우저가 HLS.js를 지원하면(Hls.js 필요)
    if (Hls.isSupported()) {
      // HLS 인스턴스 생성
      const hls = new Hls({
        enableWorker: true, // 백그라운드 작업 허용
        lowLatencyMode: true, // 지연 시간 줄이는 옵션
      });

      // 스트림 URL 로드
      hls.loadSource(src);

      // video 요소에 붙이기
      hls.attachMedia(video);

      // 컴포넌트 unmount 시 HLS 인스턴스 제거
      return () => {
        hls.destroy();
      };
    }

    // 2) Safari처럼 HLS를 네이티브 지원하면 video.src로 직접 재생
    else if (video.canPlayType("application/vnd.apple.mpegurl")) {
      video.src = src;
    }
  }, [src]); // src 변경될 때마다 재초기화

  return (
    <video
      ref={videoRef} // 실제 video DOM 연결
      style={{ width: "100%", height: "100%", objectFit: "cover" }}
      autoPlay={autoPlay} // 자동 재생
      muted={muted}       // 음소거 (자동재생 필수)
      playsInline         // iOS 전체화면 방지
      controls={false}    // 기본 컨트롤 숨김 (원하면 true 가능)
    />
  );
}

