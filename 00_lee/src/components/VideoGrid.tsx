import { Calendar } from "lucide-react";

export function VideoGrid({
  videos,
  mainFeedId,
  warningFeedId,
  onSelectVideo,
  onDragStart,
  onDropOnThumbnail,
}) {
  return (
    <section className="bg-white rounded-xl p-4 shadow-sm">
      <h2 className="text-sm text-gray-700 mb-2 flex items-center gap-2">
        <Calendar className="w-4 h-4" /> Live Cameras
      </h2>

      <div className="grid grid-cols-3 gap-4">
        {videos.map((video) => {
          const isWarning = video.id === warningFeedId;
          const isMain = video.id === mainFeedId;

          return (
            <div
              key={video.id}
              draggable
              onClick={() => onSelectVideo(video.id)}
              onDragStart={() => onDragStart(video.id)}
              onDrop={(e) => {
                e.preventDefault();
                onDropOnThumbnail(video.id);
              }}
              onDragOver={(e) => e.preventDefault()}
              className={`relative rounded-xl overflow-hidden cursor-pointer border-4 ${
                isWarning
                  ? "warning-border"
                  : isMain
                  ? "border-indigo-500"
                  : "border-gray-300"
              }`}
            >
              <div className="aspect-video w-full bg-black">
                <video
                  src={video.videoUrl}
                  className="w-full h-full object-cover"
                  autoPlay
                  loop
                  muted
                />
              </div>

              <div className="absolute top-2 left-2 px-2 py-1 text-[10px] bg-black/70 text-white rounded">
                {video.name}
              </div>

              {isWarning && (
                <div className="absolute top-2 right-2 px-2 py-1 text-[10px] rounded-full bg-yellow-400 text-gray-900 font-semibold">
                  WARNING
                </div>
              )}
            </div>
          );
        })}
      </div>
    </section>
  );
}
