// src/components/VideoGrid.tsx
// ----------------------------------------------------------
// Final Version
// - MJPEG 스트림
// - 기본 3칸 빈 화면
// - 카메라 입력 앞에서부터 채움
// - 3개 넘어가면 뒤로 계속 추가
// - 4개 이상일 때 캐러셀 활성화
// - 사이즈/레이아웃은 예전 VideoGrid와 동일: aspect-video + grid-cols-3
// ----------------------------------------------------------

import { useRef, useState } from "react";
import { useMjpegStream } from "../hooks/useMjpegStream";

type VideoFeed = {
  id: number;
  name: string;
  cameraId: string;
  videoUrl?: string;
  isPlaceholder?: boolean;
};

export function VideoGrid({
  videos,
  mainFeedId,
  activeWarningCamera,
  onSelectVideo,
  onDragStart,
  onDropOnThumbnail,
}) {
  // ------------------------------------------------------
  // 1) 기본 빈칸 3개 (카메라 박스와 동일한 비율/형태)
  // ------------------------------------------------------
  const baseEmptySlots: VideoFeed[] = [
    { id: -1, name: "", cameraId: "empty-1", isPlaceholder: true },
    { id: -2, name: "", cameraId: "empty-2", isPlaceholder: true },
    { id: -3, name: "", cameraId: "empty-3", isPlaceholder: true },
  ];

  // ------------------------------------------------------
  // 2) 앞에서부터 채우기
  // ------------------------------------------------------
  const filledSlots: VideoFeed[] = baseEmptySlots.map((slot, idx) =>
    videos[idx] ? videos[idx] : slot
  );

  // ------------------------------------------------------
  // 3) 4개 이상이면 뒤에 추가
  // ------------------------------------------------------
  const extraSlots = videos.slice(3);
  const finalItems = [...filledSlots, ...extraSlots];

  // ------------------------------------------------------
  // 4) 캐러셀 (아이템 수 ≥ 4)
  // ------------------------------------------------------
  const isCarouselActive = finalItems.length > 3;

  const [index, setIndex] = useState(0);
  const maxIndex = Math.max(0, finalItems.length - 3);

  const movePrev = () => setIndex((i) => Math.max(0, i - 1));
  const moveNext = () => setIndex((i) => Math.min(maxIndex, i + 1));

  // 드래그 캐러셀
  const dragStartX = useRef<number | null>(null);
  const onDragStartSlide = (x: number) => {
    if (!isCarouselActive) return;
    dragStartX.current = x;
  };
  const onDragMoveSlide = (x: number) => {
    if (!isCarouselActive || dragStartX.current === null) return;
    const diff = dragStartX.current - x;
    if (Math.abs(diff) > 50) {
      diff > 0 ? moveNext() : movePrev();
      dragStartX.current = null;
    }
  };

  return (
    <section className="bg-white rounded-xl p-4 shadow-sm">

      {/* HEADER */}
      <div className="flex justify-between items-center mb-3">
        <h2 className="text-sm text-gray-700 font-semibold flex items-center gap-2">
          Live Cameras
        </h2>

        {isCarouselActive && (
          <div className="flex gap-2">
            <button
              onClick={movePrev}
              disabled={index === 0}
              className={`px-2 py-1 text-xs rounded border ${
                index === 0 ? "opacity-30 cursor-default" : "hover:bg-gray-100"
              }`}
            >
              ◀
            </button>

            <button
              onClick={moveNext}
              disabled={index === maxIndex}
              className={`px-2 py-1 text-xs rounded border ${
                index === maxIndex
                  ? "opacity-30 cursor-default"
                  : "hover:bg-gray-100"
              }`}
            >
              ▶
            </button>
          </div>
        )}
      </div>

      {/* BODY */}
      <div
        className="overflow-hidden select-none"
        onMouseDown={(e) => onDragStartSlide(e.clientX)}
        onMouseMove={(e) => onDragMoveSlide(e.clientX)}
        onTouchStart={(e) => onDragStartSlide(e.touches[0].clientX)}
        onTouchMove={(e) => onDragMoveSlide(e.touches[0].clientX)}
      >
        <div
          className="flex transition-transform duration-300 ease-out gap-4"
          style={{
            transform: isCarouselActive
              ? `translateX(-${index * (100 / 3 + 1.3)}%)`
              : "translateX(0%)",
            width: `${finalItems.length * (100 / 3 + 1.3)}%`,
          }}
        >
          {finalItems.map((item) => {
            const isPlaceholder = item.isPlaceholder === true;
            const isWarning =
              !isPlaceholder && item.cameraId === activeWarningCamera;
            const isMain = item.id === mainFeedId;

            const streamUrl = `http://localhost:8001/stream/${item.cameraId}`;
            const { imgRef, handleError } = useMjpegStream(streamUrl, 800);

            return (
              <div
                key={item.id}
                draggable={!isPlaceholder}
                onClick={() => !isPlaceholder && onSelectVideo(item.id)}
                onDragStart={() => !isPlaceholder && onDragStart(item.id)}
                onDrop={(e) => {
                  e.preventDefault();
                  !isPlaceholder && onDropOnThumbnail(item.id);
                }}
                onDragOver={(e) => e.preventDefault()}
                className={`
                  relative rounded-xl overflow-hidden border-4 flex-shrink-0 cursor-pointer
                  ${
                    isPlaceholder
                      ? "border-gray-300 bg-black"
                      : isWarning
                      ? "warning-border"
                      : isMain
                      ? "border-indigo-500"
                      : "border-gray-300"
                  }
                `}
                style={{
                  width: "33%",
                }}
              >
                {/* 16:9 비율 적용 */}
                <div className="aspect-video w-full bg-black">
                  {isPlaceholder ? (
                    <div className="w-full h-full"></div>
                  ) : (
                    <img
                      ref={imgRef}
                      onError={handleError}
                      alt={item.name}
                      className="w-full h-full object-cover"
                    />
                  )}
                </div>

                {/* 라벨 */}
                {!isPlaceholder && (
                  <div className="absolute top-2 left-2 px-2 py-1 text-[10px] bg-black/70 text-white rounded">
                    {item.name}
                  </div>
                )}

                {/* Warning 표시 */}
                {!isPlaceholder && isWarning && (
                  <div className="absolute top-2 right-2 px-2 py-1 text-[10px] bg-yellow-400 text-black rounded font-bold">
                    WARNING
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>
    </section>
  );
}
