// src/components/MainVideoPlayer.tsx
// ----------------------------------------------------------
// display_name + FPS 묶어서 정확한 위치에 배치
// 다른 기능 절대 변경 없음
// ----------------------------------------------------------

import { useEffect, useRef, useState } from "react";
import { TrackingSnapshot } from "../App";

type MainVideoPlayerProps = {
  cameraId: string;
  cameraName: string;
  topLabel: string | null;
  isAnomaly: boolean;
  mjpegUrl: string;
  tracking: TrackingSnapshot | null;
};

export function MainVideoPlayer({
  cameraId,
  cameraName,
  topLabel,
  isAnomaly,
  mjpegUrl,
  tracking,
}: MainVideoPlayerProps) {
  const imgRef = useRef<HTMLImageElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  // ----------------------------------------------------------
  // FPS 계산 (tracking.timestamp 기반)
  // ----------------------------------------------------------
  const prevTs = useRef<number | null>(null);
  const [fps, setFps] = useState<number | null>(null);

  useEffect(() => {
    if (!tracking) {
      prevTs.current = null;
      setFps(null);
      return;
    }

    const curr = tracking.timestamp;
    const prev = prevTs.current;

    if (typeof curr === "number" && typeof prev === "number") {
      const delta = curr - prev;
      if (delta > 0 && Number.isFinite(delta)) {
        const f = 1 / delta;
        if (Number.isFinite(f)) setFps(f);
      }
    }

    prevTs.current = curr;
  }, [tracking?.timestamp]);

  // ----------------------------------------------------------
  // Tracking BBoxes
  // ----------------------------------------------------------
  useEffect(() => {
    if (!imgRef.current || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    canvas.width = imgRef.current.clientWidth;
    canvas.height = imgRef.current.clientHeight;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!tracking) return;

    tracking.objects.forEach((obj) => {
      const { x, y, w, h } = obj.bbox;

      const px = x * canvas.width;
      const py = y * canvas.height;
      const pw = w * canvas.width;
      const ph = h * canvas.height;

      ctx.strokeStyle = isAnomaly ? "#FF0000" : "#00FF00";
      ctx.lineWidth = 2;
      ctx.strokeRect(px, py, pw, ph);

      const labelText =
        topLabel && typeof topLabel === "string" ? topLabel : "";

      ctx.fillStyle = "rgba(0,0,0,0.5)";
      ctx.fillRect(px, py - 14, pw, 14);

      ctx.fillStyle = "#FFF";
      ctx.font = "12px sans-serif";
      ctx.fillText(labelText, px + 2, py - 3);
    });
  }, [tracking, topLabel, isAnomaly]);

  return (
    <div className="relative w-full h-full rounded-lg overflow-hidden bg-black">

      {/* ----------------------------------------------------------
         display_name + FPS 묶어서 한 그룹 absolute fixed
         구조만 묶음, 기능은 변화 없음
      ---------------------------------------------------------- */}
      <div className="absolute top-2 left-2 flex flex-col gap-1 z-20">

        {/* display_name */}
        <div
          className="
            bg-black/60 text-white
            text-lg font-bold
            px-3 py-1.5
            rounded
          "
          style={{
            textShadow:
              "-1px 0 0 black, 1px 0 0 black, 0 -1px 0 black, 0 1px 0 black",
          }}
        >
          {cameraName}
        </div>

        {/* FPS */}
        {fps !== null && (
          <div
            className="
              bg-black/60 text-white
              text-base font-bold
              px-3 py-1.5
              rounded
            "
            style={{
              textShadow:
                "-1px 0 0 black, 1px 0 0 black, 0 -1px 0 black, 0 1px 0 black",
            }}
          >
            FPS: {fps.toFixed(1)}
          </div>
        )}

      </div>

      {/* MJPEG */}
      <img
        ref={imgRef}
        src={mjpegUrl}
        className="w-full h-full object-contain"
        draggable={false}
      />

      {/* Tracking Canvas */}
      <canvas
        ref={canvasRef}
        className="absolute inset-0 pointer-events-none"
      />
    </div>
  );
}
