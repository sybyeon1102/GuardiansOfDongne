// src/components/RightPanel.tsx
// ----------------------------------------------------------
// Status UI 완전 통일본
// - Anomaly/Normal, Connected, Time 버튼 모두 동일한 스타일
// - baseStatusBtn 공통 클래스 사용
// ----------------------------------------------------------

import React from "react";
import { Probabilities, EventLogEntry } from "../App";

type RightPanelProps = {
  selectedCameraName: string | null;
  selectedCameraId: string | null;
  probabilities: Probabilities;
  connectionStatus: "ok" | "disconnected";
  isDataStale: boolean;
  topLabel: string | null;
  topProb: number | null;
  eventLogs: EventLogEntry[];
  selectedHasWarning: boolean;
  isAnomaly: boolean;
  currentTime: string;
};

export function RightPanel({
  selectedCameraName,
  selectedCameraId,
  probabilities,
  connectionStatus,
  isDataStale,
  topLabel,
  topProb,
  eventLogs,
  isAnomaly,
  currentTime,
}: RightPanelProps) {

  // ----------------------------------------------------------
  // 공통 버튼 스타일 (세 버튼 모두 동일)
  // ----------------------------------------------------------
  const baseStatusBtn = `
    inline-flex items-center justify-center
    px-8 py-2
    min-w-[150px]
    rounded-full
    text-sm font-semibold
    shadow
    font-mono
    select-none
  `;

  return (
    <div className="flex flex-col gap-4 h-full p-3">

      {/* Camera Info -------------------------------------------------- */}
      <section className="bg-white p-4 rounded-xl shadow-sm">
        <h2 className="text-lg font-semibold text-gray-800">
          {selectedCameraName ?? "Camera"}
        </h2>
        <p className="text-sm text-gray-500 mt-1">
          ID: {selectedCameraId ?? "-"}
        </p>
      </section>

      {/* Status -------------------------------------------------------- */}
      <section className="bg-white p-4 rounded-xl shadow-sm">
        <h3 className="text-sm font-semibold text-gray-600 mb-4">Status</h3>

        {/* Row 1: Anomaly + Connected */}
        <div className="flex items-center gap-4 mb-4">

          {/* Normal / Anomaly */}
          <span
            className={`
              ${baseStatusBtn}
              ${isAnomaly ? "bg-yellow-400 text-black" : "bg-green-500 text-white"}
            `}
          >
            {isAnomaly ? "Anomaly" : "Normal"}
          </span>

          {/* Connected */}
          <span
            className={`
              ${baseStatusBtn}
              ${connectionStatus === "ok" ? "bg-green-500 text-white" : "bg-red-500 text-white"}
            `}
          >
            {connectionStatus === "ok" ? "Connected" : "Disconnected"}
          </span>

        </div>

        {/* Row 2: Time */}
        <div>
          <span
            style={{ backgroundColor: "black" }}
            className={`
              ${baseStatusBtn}
              !bg-black text-white
            `}
          >
            {currentTime}
          </span>
        </div>

        {isDataStale && (
          <div className="text-xs text-gray-500 mt-2">
            Data may be delayed
          </div>
        )}
      </section>

      {/* Top Label ---------------------------------------------------- */}
      <section className="bg-white p-4 rounded-xl shadow-sm">
        <h3 className="text-sm font-semibold text-gray-600">Top Label</h3>
        <div className="text-xl font-bold mt-2">{topLabel ?? "—"}</div>
        <div className="text-gray-600 text-sm">
          {topProb != null ? (topProb * 100).toFixed(1) + "%" : "--"}
        </div>
      </section>

      {/* Probabilities ------------------------------------------------ */}
      <section className="bg-white p-4 rounded-xl shadow-sm">
        <h3 className="text-sm font-semibold text-gray-600 mb-2">Probabilities</h3>
        <div className="grid grid-cols-2 gap-y-1 text-sm">
          {Object.entries(probabilities).map(([key, value]) => (
            <div key={key} className="flex justify-between">
              <span className="capitalize">{key.replace("_", " ")}</span>
              <span className="font-mono">{(value * 100).toFixed(1)}%</span>
            </div>
          ))}
        </div>
      </section>

      {/* Event Log ---------------------------------------------------- */}
      <section className="bg-white p-4 rounded-xl shadow-sm flex-1 overflow-auto">
        <h3 className="text-sm font-semibold text-gray-600 mb-2">Event Log</h3>

        <div className="space-y-2">
          {eventLogs.map((log) => (
            <div key={log.id} className="border rounded p-2 text-xs bg-gray-50">
              <div className="font-bold">{log.timestamp}</div>
              <div>Camera: {log.cameraId ?? "-"}</div>
              <div>
                Label: {log.topLabel} ({(log.topProb * 100).toFixed(1)}%)
              </div>
            </div>
          ))}
        </div>
      </section>

    </div>
  );
}
