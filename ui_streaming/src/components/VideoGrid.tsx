// src/components/VideoGrid.tsx
// ----------------------------------------------------------
// cameraId 기반 썸네일 UI
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
  isDataStale: boolean;
};

export function VideoGrid({
  cameras,
  mainCameraId,
  warningCameraIds,
  onSelectCamera,
  onDragStart,
  onDropOnThumbnail,
  isDataStale,
}: VideoGridProps) {
  if (cameras.length === 0) {
    return (
      <section className="bg-white rounded-xl p-4 shadow-sm">
        <div className="text-sm text-gray-500">
          연결된 카메라가 없습니다. 에이전트와 서버 상태를 확인해주세요.
        </div>
      </section>
    );
  }

  return (
    <section className="bg-white rounded-xl p-4 shadow-sm">
      <h2 className="text-sm font-semibold mb-2">Cameras</h2>
      <div className="grid grid-cols-3 gap-3">
        {cameras.map((cam) => {
          const isMain = cam.cameraId === mainCameraId;
          const isWarning = warningCameraIds.includes(cam.cameraId);

          const borderClass = isWarning
            ? "warning-border"
            : isMain
            ? "border-indigo-500"
            : "border-gray-300";

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
              className={`relative cursor-pointer rounded-lg overflow-hidden border-4 ${borderClass} bg-black aspect-video flex items-center justify-center`}
            >
              <span className="text-xs text-gray-200">{cam.name}</span>

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

              {isDataStale && (
                <div className="absolute bottom-1 right-1 px-1.5 py-0.5 bg-yellow-500 text-[10px] text-black rounded">
                  Delay
                </div>
              )}
            </div>
          );
        })}
      </div>
    </section>
  );
}
