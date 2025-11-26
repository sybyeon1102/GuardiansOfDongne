// src/components/VideoGrid.tsx
// ----------------------------------------------------------
// 완성본 - 렌더링 안정화, 중복 클릭 방지, Date.now() 제거
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
        max-h-[320px]
        overflow-y-auto
      "
    >
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
              onClick={() => {
                // ← 중복 클릭으로 mainCameraId 흔들리는 문제 완전 차단
                if (mainCameraId !== cam.cameraId) {
                  onSelectCamera(cam.cameraId);
                }
              }}
              className={`relative cursor-pointer rounded-lg overflow-hidden border-4 ${borderClass} bg-black aspect-video`}
            >
              {/* ← MJPEG는 thumbnail에서 refresh 불필요 → Date.now() 제거 */}
              <img src={mjpegUrl} className="w-full h-full object-cover" />

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
