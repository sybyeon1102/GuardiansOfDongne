// src/components/RightPanelFixed.tsx
import React from "react";
import { Probabilities, EventLogEntry } from "../App";

type Props = {
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

export function RightPanelFixed({
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
}: Props) {
  return (
    <aside
      className="
        fixed top-[64px]   /* header height */
        right-0
        w-[28vw]
        h-[calc(100vh-64px)]
        bg-gray-50
        overflow-y-auto
        border-l border-gray-200
        p-5
        flex flex-col gap-6
      "
    >

      {/* Camera Info */}
      <div className="bg-white p-5 rounded-2xl shadow-sm border border-gray-100">
        <h2 className="text-xl font-semibold">{selectedCameraName ?? "Camera"}</h2>
        <p className="text-sm text-gray-500 mt-1">ID: {selectedCameraId ?? "-"}</p>
      </div>

      {/* Status */}
      <div className="bg-white p-5 rounded-2xl shadow-sm border border-gray-100">
        <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider">Status</h3>
        <div className="mt-2 text-lg font-bold">
          {selectedHasWarning ? "⚠️ Warning" : "Normal"}
        </div>
      </div>

      {/* Top Label */}
      <div className="bg-white p-5 rounded-2xl shadow-sm border border-gray-100">
        <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider">Top Label</h3>
        <div className="mt-2 text-2xl font-bold">{topLabel ?? "—"}</div>
        <div className="text-gray-600 text-sm mt-1">
          {topProb != null ? (topProb * 100).toFixed(1) + "%" : "--"}
        </div>
      </div>

      {/* Probabilities */}
      <div className="bg-white p-5 rounded-2xl shadow-sm border border-gray-100">
        <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-3">
          Probabilities
        </h3>
        <div className="grid grid-cols-2 gap-3">
          {Object.entries(probabilities).map(([k, v]) => (
            <div key={k} className="bg-gray-50 p-3 rounded-lg border border-gray-100 flex justify-between">
              <span className="capitalize text-gray-700">{k.replace("_", " ")}</span>
              <span className="font-mono font-semibold">
                {(v * 100).toFixed(1)}%
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Server Connection */}
      <div className="bg-white p-5 rounded-2xl shadow-sm border border-gray-100">
        <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider">
          Server Connection
        </h3>
        <div className="flex items-center gap-2 mt-3">
          <div className={`w-3 h-3 rounded-full ${
            connectionStatus === "ok" ? "bg-green-500" : "bg-red-500"
          }`} />
          <span className="text-sm text-gray-800">
            {connectionStatus === "ok" ? "Connected" : "Disconnected"}
          </span>
        </div>
      </div>

      {/* Latest Info */}
      <div className="bg-white p-5 rounded-2xl shadow-sm border border-gray-100">
        <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">
          Latest Info
        </h3>
        <p className="text-sm">Camera: {lastCameraId ?? "-"}</p>
        <p className="text-sm mt-1">Source: {lastSourceId ?? "-"}</p>
        <p className="text-sm mt-1">
          Warning: {selectedHasWarning ? "Yes" : "No"}
        </p>
      </div>

      {/* Event Log */}
      <div className="bg-white p-5 rounded-2xl shadow-sm border border-gray-100 flex-1">
        <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-3">
          Event Log
        </h3>

        <div className="space-y-3 max-h-[400px] overflow-y-auto pr-1">
          {eventLogs.map((log) => (
            <div key={log.id} className="p-3 bg-gray-50 border border-gray-200 rounded-lg">
              <div className="text-xs font-bold">{log.timestamp}</div>
              <div className="text-xs text-gray-600 mt-1">Camera: {log.cameraId}</div>
              <div className="text-xs text-gray-600 mt-1">
                Label: {log.topLabel} ({(log.topProb * 100).toFixed(1)}%)
              </div>
            </div>
          ))}
        </div>
      </div>

    </aside>
  );
}
