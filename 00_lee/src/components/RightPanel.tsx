import { useEffect, useState } from "react";

type Probabilities = {
  fall: number;
  abandon: number;
  broken: number;
  fight: number;
  fire: number;
  smoke: number;
  theft: number;
  weak_pedestrian: number;
};

type ConnectionStatus = "ok" | "disconnected";

type EventLogEntry = {
  id: number;
  timestamp: string;
  cameraId: string | null;
  sourceId: string | null;
  topLabel: string;
  topProb: number;
};

type RightPanelProps = {
  warningVideoUrl: string;
  probabilities: Probabilities;
  connectionStatus: ConnectionStatus;
  topLabel: string | null;
  topProb: number | null;
  lastCameraId: string | null;
  lastSourceId: string | null;
  eventLogs: EventLogEntry[];
  warningFeedName?: string | null;
};

const probabilityLabels: { key: keyof Probabilities; label: string }[] = [
  { key: "fall", label: "Fall" },
  { key: "abandon", label: "Abandon" },
  { key: "broken", label: "Broken" },
  { key: "fight", label: "Fight" },
  { key: "fire", label: "Fire" },
  { key: "smoke", label: "Smoke" },
  { key: "theft", label: "Theft" },
  { key: "weak_pedestrian", label: "Weak pedestrian" },
];

export function RightPanel({
  warningVideoUrl,
  probabilities,
  connectionStatus,
  topLabel,
  topProb,
  lastCameraId,
  lastSourceId,
  eventLogs,
  warningFeedName,
}: RightPanelProps) {
  const [currentTime, setCurrentTime] = useState("");

  // 시간 업데이트
  useEffect(() => {
    const updateTime = () => {
      const now = new Date();
      setCurrentTime(
        now
          .toLocaleTimeString("ko-KR", {
            hour12: false,
            hour: "2-digit",
            minute: "2-digit",
            second: "2-digit",
          })
          .replaceAll(".", "")
      );
    };
    updateTime();
    const timer = setInterval(updateTime, 1000);
    return () => clearInterval(timer);
  }, []);

  // 가장 높은 확률 라벨 찾기
  const maxLabelKey = (
    Object.keys(probabilities) as (keyof Probabilities)[]
  ).reduce((best, key) =>
    probabilities[key] > probabilities[best] ? key : best
  );

  const displayCameraName =
    warningFeedName ||
    (lastCameraId
      ? `Camera ${lastCameraId}`
      : lastSourceId
      ? `${lastSourceId}`
      : "-");

  return (
    <aside className="w-80 border-l border-gray-200 bg-gray-50 flex flex-col gap-4 p-4">

      {/* Warning Video Section (border-4, VideoGrid 스타일과 통일) */}
      <div className="bg-white rounded-xl overflow-hidden shadow-sm border-4 border-yellow-400">
        <div className="px-4 py-2 border-b border-yellow-300 bg-yellow-50">
          <div className="flex flex-col text-xs">
            <span className="font-semibold text-yellow-800">Warning</span>
            <span className="text-[11px] text-yellow-800">
              Camera: {displayCameraName}
            </span>
          </div>
        </div>

        <video
          src={warningVideoUrl}
          className="w-full h-40 object-cover bg-black"
          autoPlay
          muted
          loop
        />
      </div>

      {/* Anomaly Classification */}
      <div className="bg-white rounded-xl p-4 shadow-sm">
        <h2 className="mb-2 text-gray-800 text-sm font-semibold">
          Anomaly Classification
        </h2>

        {/* Top Anomaly (테두리 더 굵게 + 패딩 통일감) */}
        <div className="mb-3 rounded-lg bg-yellow-50 border-4 border-yellow-400 px-4 py-3 text-xs">
          <div className="text-[11px] text-yellow-800 font-semibold mb-2">
            Top Anomaly
          </div>

          <div className="flex items-center justify-between text-[14px] text-yellow-900 font-semibold">
            <span>{topLabel ? topLabel.toUpperCase() : "-"}</span>

            <span className="font-mono">
              {topProb != null ? `${(topProb * 100).toFixed(1)}%` : "--.-%"}
            </span>
          </div>
        </div>

        {/* 전체 확률 리스트 */}
        <div className="space-y-2">
          {probabilityLabels.map((item) => {
            const value = probabilities[item.key];
            const isMax = item.key === maxLabelKey;

            return (
              <div
                key={item.key}
                className={`flex items-center justify-between px-3 py-2 rounded-lg text-xs ${
                  isMax
                    ? "bg-yellow-100 border border-yellow-400"
                    : "bg-gray-50 border border-transparent"
                }`}
              >
                <span className="text-gray-700">{item.label}</span>
                <span className="font-mono text-gray-900">
                  {(value * 100).toFixed(1)}%
                </span>
              </div>
            );
          })}
        </div>
      </div>

      {/* Status Section */}
      <div className="bg-white rounded-xl p-4 shadow-sm">
        <h2 className="mb-3 text-gray-800 text-sm font-semibold">Status</h2>

        <div className="flex items-center gap-3">
          <button
            className={`px-4 py-2 rounded-lg text-xs font-semibold ${
              connectionStatus === "ok"
                ? "bg-green-500 text-white"
                : "bg-red-500 text-white"
            }`}
          >
            {connectionStatus === "ok" ? "OK" : "Disconnected"}
          </button>

          <div className="px-4 py-2 rounded-lg bg-gray-800 text-white text-xs font-mono">
            {currentTime}
          </div>
        </div>
      </div>

      {/* Event Logs */}
      <div className="bg-white rounded-xl p-4 shadow-sm flex-1 flex flex-col min-h-[180px]">
        <h2 className="mb-3 text-gray-800 text-sm font-semibold">Event Logs</h2>

        <div className="flex-1 overflow-y-auto pr-1 space-y-2 text-[11px]">
          {eventLogs.length === 0 ? (
            <div className="text-gray-400 text-xs">
              아직 기록된 이벤트가 없습니다.
            </div>
          ) : (
            eventLogs.map((log) => {
              const idLabel = log.cameraId || log.sourceId || "Unknown";
              const modeLabel = log.cameraId ? "Camera" : "Source";

              return (
                <div
                  key={log.id}
                  className="border border-gray-200 rounded-lg px-3 py-2 bg-gray-50"
                >
                  <div className="flex items-center justify-between mb-1">
                    <span className="font-mono text-[10px] text-gray-500">
                      {log.timestamp}
                    </span>
                    <span className="text-[10px] text-gray-600">
                      {modeLabel}: {idLabel}
                    </span>
                  </div>

                  <div className="flex items-center justify-between">
                    <span className="font-semibold text-gray-800">
                      {log.topLabel.toUpperCase()}
                    </span>
                    <span className="font-mono text-gray-900">
                      {(log.topProb * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>
              );
            })
          )}
        </div>
      </div>
    </aside>
  );
}
