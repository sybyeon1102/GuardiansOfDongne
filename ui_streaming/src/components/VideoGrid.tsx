// src/components/VideoGrid.tsx
// ----------------------------------------------------------
// 기능 요약
// ----------------------------------------------------------
// - 카메라 썸네일 3열 그리드
// - 개수가 많아지면 세로 스크롤 생성
// - 썸네일은 MJPEG 실시간 스트림
// - 드래그 앤 드롭 / 경고 표시 / 메인 표시 유지
// ----------------------------------------------------------

type CameraFeed = {
  cameraId: string;
  name: string;
};

type VideoGridProps = {
  cameras: CameraFeed[];
  mainCameraId: string | null;
  warningCameraIds: string[];
  onSelectCamera: (cameraId: string) => void;
  onDragStart: (cameraId: string) => void;
  onDropOnThumbnail: (targetCameraId: string) => void;
};

const AGENT_BASE_URL =
  import.meta.env.VITE_AGENT_BASE_URL ?? "http://localhost:8001";

export function VideoGrid({
  cameras,
  mainCameraId,
  warningCameraIds,
  onSelectCamera,
  onDragStart,
  onDropOnThumbnail,
}: VideoGridProps) {
  if (cameras.length === 0) {
    return (
      <section className="bg-white rounded-xl p-4 shadow-sm">
        <div className="text-sm text-gray-500">
          연결된 카메라가 없습니다.
        </div>
      </section>
    );
  }

  return (
    <section
      className="
        bg-white rounded-xl p-4 shadow-sm
        max-h-[320px]       /* 세로 높이 제한 */
        overflow-y-auto     /* 카메라가 많아지면 스크롤 */
      "
    >
      <h2 className="text-sm font-semibold mb-2">Cameras</h2>

      {/* 3열 고정 grid */}
      <div className="grid grid-cols-3 gap-3">
        {cameras.map((cam) => {
          const isMain = cam.cameraId === mainCameraId;
          const isWarning = warningCameraIds.includes(cam.cameraId);

          const borderClass = isWarning
            ? "warning-border"
            : isMain
            ? "border-indigo-500"
            : "border-gray-300";

          const mjpegUrl = `${AGENT_BASE_URL}/streams/${cam.cameraId}.mjpeg`;

          return (
            <div
              key={cam.cameraId}
              draggable
              onDragStart={() => onDragStart(cam.cameraId)}
              onDrop={(e) => {
                e.preventDefault();
                onDropOnThumbnail(cam.cameraId);
              }}
              onDragOver={(e) => e.preventDefault()}
              onClick={() => onSelectCamera(cam.cameraId)}
              className={`relative cursor-pointer rounded-lg overflow-hidden border-4 ${borderClass} bg-black aspect-video`}
            >
              <img
                src={`${mjpegUrl}?t=${Date.now()}`}
                className="w-full h-full object-cover"
              />

              {isWarning && (
                <div className="absolute top-1 left-1 px-1.5 py-0.5 bg-yellow-400 text-[10px] font-semibold rounded">
                  WARNING
                </div>
              )}

              {isMain && (
                <div className="absolute bottom-1 left-1 px-1.5 py-0.5 bg-indigo-500 text-[10px] text-white rounded">
                  MAIN
                </div>
              )}
            </div>
          );
        })}
      </div>
    </section>
  );
}
