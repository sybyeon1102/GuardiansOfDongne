// src/components/RightPanel.tsx
// ----------------------------------------------------------
// 완전 최종 버전
// - Selected Camera
// - Status (OK/Disconnected + 시간 pill, 디자인 동일)
// - Anomaly Classification (제목과 Top Anomaly 간격 넓힘)
// - Event Log 제거됨
// - 우측 상태 뱃지 제거됨
// - warning-border 적용 가능
// ----------------------------------------------------------

import { useEffect, useState } from "react";
import type { Probabilities } from "../App";

type ConnectionStatus = "ok" | "disconnected";

type RightPanelProps = {
  selectedCameraName: string | null;
  selectedCameraId: string | null;
  probabilities: Probabilities;
  connectionStatus: ConnectionStatus;
  isDataStale: boolean;
  topLabel: string | null;
  topProb: number | null;
  lastCameraId: string | null;
  lastSourceId: string | null;
  selectedHasWarning: boolean;
};

const probabilityLabels = [
  { key: "fall", label: "Fall" },
  { key: "abandon", label: "Abandon" },
  { key: "broken", label: "Broken" },
  { key: "fight", label: "Fight" },
  { key: "fire", label: "Fire" },
  { key: "smoke", label: "Smoke" },
  { key: "theft", label: "Theft" },
  { key: "weak_pedestrian", label: "Weak pedestrian" },
] as const;

export function RightPanel({
  selectedCameraName,
  selectedCameraId,
  probabilities,
  connectionStatus,
  isDataStale,
  topLabel,
  topProb,
  lastCameraId,
  lastSourceId,
  selectedHasWarning,
}: RightPanelProps) {
  // -------------------------
  // 실시간 시계
  // -------------------------
  const [now, setNow] = useState(new Date());
  useEffect(() => {
    const timer = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  const timeString = now.toLocaleTimeString("ko-KR", { hour12: false });

  // -------------------------
  // 상태 pill
  // -------------------------
  const isConnected = connectionStatus === "ok";
  const statusText = isConnected ? "OK" : "Disconnected";

  const statusClass =
    isConnected ? "bg-green-500 text-white" : "bg-red-500 text-white";

  const cameraName = selectedCameraName ?? "-";
  const source = lastSourceId ?? lastCameraId ?? selectedCameraId ?? "-";

  // 경고 테두리 여부
  const panelClass = selectedHasWarning ? "warning-border" : "";

  return (
    <aside className={`w-full ${panelClass}`}>

      {/* ---------------------------------- */}
      {/*  Selected Camera Section           */}
      {/* ---------------------------------- */}
      <div className="bg-white rounded-xl shadow-sm border p-4 mb-4">
        <h3 className="text-sm font-semibold text-gray-700 mb-2">
          Selected Camera
        </h3>

        <div className="text-lg font-bold text-gray-900 mb-4">
          {cameraName}
        </div>

        <div className="text-xs text-gray-600 mt-1">
          Source: <span className="font-mono">{source}</span>
        </div>
      </div>

      {/* ---------------------------------- */}
      {/* Status Section                     */}
      {/* ---------------------------------- */}
      <div className="bg-white rounded-xl p-4 shadow-sm mb-4">
        <h2 className="mb-3 text-gray-800 text-sm font-semibold">Status</h2>

        <div className="flex items-center gap-3">

          {/* 연결 pill */}
          <button
            className={`px-4 py-2 rounded-lg text-xs font-semibold ${statusClass}`}
          >
            {statusText}
          </button>

          {/* 시간 pill */}
          <div className="px-4 py-2 rounded-lg bg-gray-800 text-white text-xs font-mono">
            {timeString}
          </div>
        </div>
      </div>

      {/* ---------------------------------- */}
      {/*  Anomaly Classification            */}
      {/* ---------------------------------- */}
      <div className="bg-white rounded-xl shadow-sm border p-4">

        {/* ⬇ 제목과 Top Anomaly 간격 넓힘 (mb-4로 변경됨) */}
        <h3 className="text-sm font-semibold text-gray-800 mb-4">
          Anomaly Classification
        </h3>

        {/* Top Anomaly */}
        <div className="border border-yellow-300 bg-yellow-50 rounded-lg px-3 py-2 mb-4">
          <div className="text-[11px] text-yellow-800 font-semibold mb-1">
            Top Anomaly
          </div>

          <div className="flex items-center justify-between text-[14px] text-yellow-900 font-semibold">
            <span>{topLabel ? topLabel.toUpperCase() : "-"}</span>
            <span>
              {topProb != null ? `${(topProb * 100).toFixed(1)}%` : "--.-%"}
            </span>
          </div>
        </div>

        {/* 전체 라벨 리스트 */}
        <div className="space-y-2">
          {probabilityLabels.map((item) => (
            <div
              key={item.key}
              className="flex items-center justify-between px-3 py-2 rounded-lg text-xs bg-gray-50 border"
            >
              <span>{item.label}</span>
              <span className="font-mono">
                {(probabilities[item.key] * 100).toFixed(1)}%
              </span>
            </div>
          ))}
        </div>
      </div>

    </aside>
  );
}
