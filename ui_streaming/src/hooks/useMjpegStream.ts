// src/hooks/useMjpegStream.ts
// ----------------------------------------------------------
// [기능 요약]
// - MJPEG 스트림을 <img>로 출력할 때
//   네트워크 오류 발생 시 자동 재연결
// - 캐시 방지를 위해 ?t=timestamp 쿼리 추가
// ----------------------------------------------------------

import { useCallback, useEffect, useRef, useState } from "react";

export function useMjpegStream(baseUrl: string, reconnectInterval = 500) {
  const imgRef = useRef<HTMLImageElement | null>(null);
  const [retryCount, setRetryCount] = useState(0);

  const makeUrl = useCallback(() => {
    return `${baseUrl}?t=${Date.now()}`;
  }, [baseUrl]);

  const connect = useCallback(() => {
    if (!imgRef.current) return;
    imgRef.current.src = makeUrl();
  }, [makeUrl]);

  const handleError = useCallback(() => {
    setTimeout(() => {
      setRetryCount((n) => n + 1);
    }, reconnectInterval);
  }, [reconnectInterval]);

  useEffect(() => {
    connect();
  }, [retryCount, connect]);

  return { imgRef, handleError, retryCount };
}
