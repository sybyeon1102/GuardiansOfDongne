// src/components/VideoControls.tsx

import { Play, Pause, Square, Download, Circle } from "lucide-react";
import type React from "react";

type VideoControlsProps = {
  isPlaying: boolean;
  isRecording: boolean;
  currentTime: number;
  duration: number;
  onPlayPause: () => void;
  onStop: () => void;
  onSeek: (time: number) => void;
  onRecordToggle: () => void;
  onDownload: () => void;
};

export function VideoControls({
  isPlaying,
  isRecording,
  currentTime,
  duration,
  onPlayPause,
  onStop,
  onSeek,
  onRecordToggle,
  onDownload,
}: VideoControlsProps) {
  const progress = duration > 0 ? (currentTime / duration) * 100 : 0;

  const handleProgressClick = (e: React.MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const ratio = x / rect.width;
    const newTime = duration * ratio;
    onSeek(newTime);
  };

  const formatTime = (sec: number) => {
    if (!isFinite(sec)) return "00:00";
    const s = Math.floor(sec);
    const m = Math.floor(s / 60);
    const r = s % 60;
    return `${String(m).padStart(2, "0")}:${String(r).padStart(2, "0")}`;
  };

  return (
    <section className="bg-white rounded-2xl px-4 py-3 flex items-center justify-between shadow-sm">
      {/* 왼쪽: 재생/정지 */}
      <div className="flex items-center gap-2">
        <button
          className="p-2 rounded-full bg-gray-200 hover:bg-gray-300 transition-colors"
          onClick={onPlayPause}
        >
          {isPlaying ? (
            <Pause className="w-5 h-5 text-gray-800" />
          ) : (
            <Play className="w-5 h-5 text-gray-800" />
          )}
        </button>
        <button
          className="p-2 rounded-full bg-gray-200 hover:bg-gray-300 transition-colors"
          onClick={onStop}
        >
          <Square className="w-5 h-5 text-gray-800" />
        </button>
      </div>

      {/* 가운데: 타임라인 */}
      <div className="flex-1 flex items-center gap-3 px-4">
        <span className="text-xs font-mono text-gray-500 w-12 text-right">
          {formatTime(currentTime)}
        </span>
        <div
          className="flex-1 h-2 rounded-full bg-gray-200 overflow-hidden cursor-pointer"
          onClick={handleProgressClick}
        >
          <div
            className="h-full bg-indigo-500"
            style={{ width: `${progress}%` }}
          />
        </div>
        <span className="text-xs font-mono text-gray-500 w-12">
          {formatTime(duration)}
        </span>
      </div>

      {/* 오른쪽: 녹화 + 다운로드 */}
      <div className="flex items-center gap-2">
        <button
          className={`p-2 rounded-full border transition-colors flex items-center justify-center ${
            isRecording
              ? "bg-red-500 border-red-500"
              : "bg-gray-200 border-gray-300 hover:bg-gray-300"
          }`}
          onClick={onRecordToggle}
        >
          <Circle
            className={`w-6 h-6 ${
              isRecording ? "text-white fill-white" : "text-gray-700"
            }`}
          />
        </button>

        <button
          className="p-2 hover:bg-gray-300 rounded-full transition-colors"
          onClick={onDownload}
        >
          <Download className="w-5 h-5 text-gray-700" />
        </button>
      </div>
    </section>
  );
}
