// src/components/RightPanel.tsx
// ------------------------------------------------------
// RightPanel - isAnomaly 정식 반영 포함 버전
// ------------------------------------------------------

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
  lastCameraId: string | null;
  lastSourceId: string | null;
  eventLogs: EventLogEntry[];
  selectedHasWarning: boolean;

  // 추가된 부분
  isAnomaly: boolean;
};

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
  eventLogs,
  selectedHasWarning,
  isAnomaly,
}: RightPanelProps) {
  return (
    <div className="flex flex-col gap-4 h-full p-3">

      {/* ------------------ Camera Info ------------------ */}
      <div className="bg-white p-4 rounded-xl shadow-sm">
        <h2 className="text-lg font-semibold text-gray-800">
          {selectedCameraName ?? "Camera"}
        </h2>
        <p className="text-sm text-gray-500 mt-1">
          ID: {selectedCameraId ?? "-"}
        </p>
      </div>

      {/* ------------------ Status ------------------ */}
      <div className="bg-white p-4 rounded-xl shadow-sm">
        <h3 className="text-sm font-semibold text-gray-600">Status</h3>
        <div className="mt-2 text-base font-semibold">
          {isAnomaly ? "Anomaly" : "Normal"}
        </div>
      </div>

      {/* ------------------ Top Label ------------------ */}
      <div className="bg-white p-4 rounded-xl shadow-sm">
        <h3 className="text-sm font-semibold text-gray-600">Top Label</h3>
        <div className="text-xl font-bold mt-2">{topLabel ?? "—"}</div>
        <div className="text-gray-600 text-sm">
          {topProb != null ? (topProb * 100).toFixed(1) + "%" : "--"}
        </div>
      </div>

      {/* ------------------ Probabilities ------------------ */}
      <div className="bg-white p-4 rounded-xl shadow-sm">
        <h3 className="text-sm font-semibold text-gray-600 mb-2">
          Probabilities
        </h3>
        <div className="grid grid-cols-2 gap-y-1 text-sm">
          {Object.entries(probabilities).map(([key, value]) => (
            <div key={key} className="flex justify-between">
              <span className="capitalize">{key.replace("_", " ")}</span>
              <span className="font-mono">
                {(value * 100).toFixed(1)}%
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* ------------------ Server Connection ------------------ */}
      <div className="bg-white p-4 rounded-xl shadow-sm">
        <h3 className="text-sm font-semibold text-gray-600">
          Server Connection
        </h3>

        <div className="flex items-center gap-2 mt-2">
          <div
            className={`w-3 h-3 rounded-full ${
              connectionStatus === "ok" ? "bg-green-500" : "bg-red-500"
            }`}
          ></div>
          <span className="text-sm">
            {connectionStatus === "ok" ? "Connected" : "Disconnected"}
          </span>
        </div>
      </div>

      {/* ------------------ Latest Info ------------------ */}
      <div className="bg-white p-4 rounded-xl shadow-sm">
        <h3 className="text-sm font-semibold text-gray-600 mb-2">
          Latest Info
        </h3>
        <p className="text-sm">Camera: {lastCameraId ?? "-"}</p>
        <p className="text-sm">Source: {lastSourceId ?? "-"}</p>
        <p className="text-sm">
          Warning: {selectedHasWarning ? "Yes" : "No"}
        </p>
      </div>

      {/* ------------------ Event Log ------------------ */}
      <div className="bg-white p-4 rounded-xl shadow-sm flex-1 overflow-auto">
        <h3 className="text-sm font-semibold text-gray-600 mb-2">
          Event Log
        </h3>
        <div className="space-y-2">
          {eventLogs.map((log) => (
            <div
              key={log.id}
              className="border rounded p-2 text-xs bg-gray-50"
            >
              <div className="font-bold">{log.timestamp}</div>
              <div>Camera: {log.cameraId ?? "-"}</div>
              <div>
                Label: {log.topLabel} (
                {(log.topProb * 100).toFixed(1)}%)
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
