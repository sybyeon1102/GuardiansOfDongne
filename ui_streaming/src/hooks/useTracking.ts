// src/hooks/useTracking.ts
// ----------------------------------------------------------------------
// Tracking 데이터는 Agent(8001)가 아니라
// Inference Server(8000)의 /tracking/latest/{cameraId} 에서 조회한다.
// ----------------------------------------------------------------------

import { useEffect, useState } from "react";

const INFERENCE_BASE_URL =
  import.meta.env.VITE_INFERENCE_BASE_URL ?? "http://localhost:8000";
const AGENT_CODE =
  import.meta.env.VITE_AGENT_CODE ?? "agent-main-building-02";

export type TrackingBox = {
  id: number;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
};

export function useTracking(cameraId: string | null) {
  const [boxes, setBoxes] = useState<TrackingBox[]>([]);

  useEffect(() => {
    if (!cameraId) return;

    const fetchTracking = async () => {
      try {
        const url = `${INFERENCE_BASE_URL}/tracking/latest/${cameraId}?agent_code=${AGENT_CODE}`;
        const res = await fetch(url, { cache: "no-store" });
        if (!res.ok) return;

        const data = await res.json();
        if (!data || !data.objects) {
          setBoxes([]);
          return;
        }

        const mapped = data.objects.map((obj: any) => ({
          id: obj.local_track_id,
          x1: obj.bbox.x,
          y1: obj.bbox.y,
          x2: obj.bbox.x + obj.bbox.w,
          y2: obj.bbox.y + obj.bbox.h,
        }));

        setBoxes(mapped);
      } catch {}
    };

    fetchTracking();
    const id = setInterval(fetchTracking, 150);
    return () => clearInterval(id);
  }, [cameraId]);

  return { boxes };
}
