// src/components/RightPanel.tsx
// ------------------------------------------------------
// 서버 추론값 + 상태 패널 완전 구현본
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
}: RightPanelProps) {
  return (
    <div className="bg-white rounded-xl p-6 shadow-sm h-full flex flex-col">

      {/* ------------------ 카메라 정보 ------------------ */}
      <div className="mb-4">
        <h2 className="text-lg font-semibold">
          {selectedCameraName ?? "Camera"}
        </h2>
        <p className="text-sm text-gray-500">
          ID: {selectedCameraId ?? "-"}
        </p>
      </div>

      {/* ------------------ 라벨 ------------------ */}
      <div className="mb-4">
        <h3 className="text-sm font-semibold text-gray-600">Top Label</h3>
        <div className="text-xl font-bold">
          {topLabel ?? "—"}
        </div>
        <div className="text-gray-600 text-sm">
          {topProb != null ? (topProb * 100).toFixed(1) + "%" : "--"}
        </div>
      </div>

      {/* ------------------ 8개 확률 ------------------ */}
      <div className="mb-4">
        <h3 className="text-sm font-semibold text-gray-600">Probabilities</h3>
        <div className="grid grid-cols-2 gap-y-1 text-sm mt-1">
          {Object.entries(probabilities).map(([key, value]) => (
            <div key={key} className="flex justify-between">
              <span className="capitalize">{key.replace("_", " ")}</span>
              <span className="font-mono">{(value * 100).toFixed(1)}%</span>
            </div>
          ))}
        </div>
      </div>

      {/* ------------------ 서버 상태 ------------------ */}
      <div className="mb-4">
        <h3 className="text-sm font-semibold text-gray-600">Server Status</h3>

        <div className="flex items-center gap-2 mt-1">
          <div
            className={`w-3 h-3 rounded-full ${
              connectionStatus === "ok" ? "bg-green-500" : "bg-red-500"
            }`}
          ></div>
          <span className="text-sm">
            {connectionStatus === "ok" ? "Connected" : "Disconnected"}
          </span>
        </div>

        {isDataStale && (
          <div className="text-xs text-yellow-600 mt-1">
            Data is delayed...
          </div>
        )}
      </div>

      {/* ------------------ 최근 inference 정보 ------------------ */}
      <div className="mb-4 text-sm">
        <h3 className="text-sm font-semibold text-gray-600">Latest Info</h3>
        <p>Camera: {lastCameraId ?? "-"}</p>
        <p>Source: {lastSourceId ?? "-"}</p>
        <p>Warning: {selectedHasWarning ? "⚠️ Yes" : "No"}</p>
      </div>

      {/* ------------------ 이벤트 로그 ------------------ */}
      <div className="flex-1 overflow-auto border-t pt-3">
        <h3 className="text-sm font-semibold text-gray-600 mb-2">Event Log</h3>
        <div className="space-y-2">
          {eventLogs.map((log) => (
            <div
              key={log.id}
              className="border rounded p-2 text-xs bg-gray-50"
            >
              <div className="font-bold">{log.timestamp}</div>
              <div>Camera: {log.cameraId ?? "-"}</div>
              <div>
                Label: {log.topLabel} ({(log.topProb * 100).toFixed(1)}%)
              </div>
            </div>
          ))}
        </div>
      </div>

    </div>
  );
}
