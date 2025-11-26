import React, { useEffect, useRef } from "react";

/* =============================================================
   Tracking 타입 정의
============================================================= */
export interface TrackingBBox {
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface TrackingObject {
  global_id: string;
  local_track_id: number;
  label: string;
  confidence: number;
  bbox: TrackingBBox;
}

export interface TrackingSnapshot {
  camera_id: string;
  timestamp: number;
  frame_index: number | null;
  objects: TrackingObject[];
}

interface MainVideoPlayerProps {
  cameraId: string;
  mjpegUrl: string;
  tracking: TrackingSnapshot | null;
}

/* =============================================================
   컴포넌트 본체
============================================================= */
function MainVideoPlayerBase({ cameraId, mjpegUrl, tracking }: MainVideoPlayerProps) {
  const imgRef = useRef<HTMLImageElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const resizeCanvas = () => {
    const img = imgRef.current;
    const canvas = canvasRef.current;
    if (!img || !canvas) return;
    canvas.width = img.clientWidth;
    canvas.height = img.clientHeight;
  };

  useEffect(() => {
    resizeCanvas();
    window.addEventListener("resize", resizeCanvas);
    return () => window.removeEventListener("resize", resizeCanvas);
  }, []);

  const computeContainOffset = (
    naturalW: number,
    naturalH: number,
    renderW: number,
    renderH: number
  ) => {
    const videoRatio = naturalW / naturalH;
    const renderRatio = renderW / renderH;

    let drawW = renderW;
    let drawH = renderH;
    let offsetX = 0;
    let offsetY = 0;

    if (videoRatio > renderRatio) {
      drawH = renderW / videoRatio;
      offsetY = (renderH - drawH) / 2;
    } else {
      drawW = renderH * videoRatio;
      offsetX = (renderW - drawW) / 2;
    }

    return { offsetX, offsetY, drawW, drawH };
  };

  const drawTracking = () => {
    const img = imgRef.current;
    const canvas = canvasRef.current;
    if (!img || !canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!tracking || tracking.objects.length === 0) return;

    const naturalW = img.naturalWidth;
    const naturalH = img.naturalHeight;

    if (!naturalW || !naturalH) return;

    const vw = img.clientWidth;
    const vh = img.clientHeight;

    const { offsetX, offsetY, drawW, drawH } = computeContainOffset(
      naturalW,
      naturalH,
      vw,
      vh
    );

    tracking.objects.forEach((obj) => {
      const bx = offsetX + obj.bbox.x * drawW;
      const by = offsetY + obj.bbox.y * drawH;
      const bw = obj.bbox.w * drawW;
      const bh = obj.bbox.h * drawH;

      ctx.lineWidth = 2;
      ctx.strokeStyle = "#00FF00";
      ctx.strokeRect(bx, by, bw, bh);

      ctx.fillStyle = "rgba(0,0,0,0.6)";
      ctx.fillRect(bx, by - 18, 160, 18);

      ctx.fillStyle = "#fff";
      ctx.font = "12px Arial";
      ctx.fillText(
        `${obj.label} #${obj.local_track_id} (${obj.confidence.toFixed(2)})`,
        bx + 4,
        by - 5
      );
    });
  };

  useEffect(() => {
    drawTracking();                       //drawTracking 트래킹 온/오프 여기서
  }, [tracking]);

  const handleImgLoad = () => {
    resizeCanvas();
    drawTracking();                      //drawTracking 트래킹 온/오프 여기서
  };

  return (
    <div className="relative w-full aspect-video bg-black rounded-xl overflow-hidden">
      <img
        ref={imgRef}
        src={mjpegUrl}
        alt={cameraId}
        className="object-contain w-full h-full select-none"
        onLoad={handleImgLoad}
      />
      <canvas ref={canvasRef} className="absolute inset-0 pointer-events-none" />
    </div>
  );
}

/* ⭐⭐⭐ MainVideoPlayer 메모이제이션 적용 ⭐⭐⭐ */
export const MainVideoPlayer = React.memo(MainVideoPlayerBase);
