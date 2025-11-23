require("dotenv").config();

const path = require("path");
const fs = require("fs");
const { spawn } = require("child_process");
const express = require("express");

const app = express();
const PORT = process.env.PORT || 4000;

// HLS 출력 디렉토리
const HLS_ROOT = path.join(__dirname, "hls");
if (!fs.existsSync(HLS_ROOT)) {
  fs.mkdirSync(HLS_ROOT, { recursive: true });
}

// 카메라 목록 (id와 RTSP 주소 매핑)
const STREAMS = [
  { id: "cam01", rtspUrl: process.env.RTSP_CAM01 },
  { id: "cam02", rtspUrl: process.env.RTSP_CAM02 },
  { id: "cam03", rtspUrl: process.env.RTSP_CAM03 },
].filter((s) => !!s.rtspUrl);

if (STREAMS.length === 0) {
  console.error("RTSP_CAMxx 환경변수가 없습니다. .env를 확인하세요.");
  process.exit(1);
}

function startFfmpegStream(stream) {
  const outDir = path.join(HLS_ROOT, stream.id);
  if (!fs.existsSync(outDir)) {
    fs.mkdirSync(outDir, { recursive: true });
  }

  const outPath = path.join(outDir, "index.m3u8");

  // FFmpeg 명령어: RTSP → HLS (저지연용으로 짧은 세그먼트)
  const args = [
    "-rtsp_transport",
    "tcp",
    "-i",
    stream.rtspUrl,
    "-an", // 오디오 제거 (필요하면 삭제)
    "-c:v",
    "copy", // 인코딩 부담 줄이기 위해 copy (코덱 호환 안 되면 h264 등으로 변경)
    "-f",
    "hls",
    "-hls_time",
    "1",
    "-hls_list_size",
    "5",
    "-hls_flags",
    "delete_segments+append_list",
    outPath,
  ];

  console.log(`[FFmpeg] start stream ${stream.id}`);
  const ff = spawn("ffmpeg", args, { stdio: ["ignore", "pipe", "pipe"] });

  ff.stdout.on("data", (data) => {
    // 디버깅용 출력이 필요하면 주석 해제
    // console.log(`[ffmpeg ${stream.id} stdout]: ${data}`);
  });
  ff.stderr.on("data", (data) => {
    // FFmpeg 로그 (경고/에러 포함)
    console.log(`[ffmpeg ${stream.id}] ${data}`);
  });

  ff.on("close", (code) => {
    console.log(`[FFmpeg] stream ${stream.id} exited with code ${code}. 재시작 시도...`);
    setTimeout(() => startFfmpegStream(stream), 3000);
  });
}

// 모든 카메라에 대해 FFmpeg 실행
STREAMS.forEach((s) => startFfmpegStream(s));

// 정적 HLS 파일 서빙
app.use("/hls", express.static(HLS_ROOT));

app.get("/", (req, res) => {
  res.type("text").send("RTSP → HLS server running.\nHLS root: /hls/<camId>/index.m3u8");
});

app.listen(PORT, () => {
  console.log(`RTSP → HLS server listening on http://localhost:${PORT}`);
  console.log("Available streams:");
  STREAMS.forEach((s) => {
    console.log(`  ${s.id}: http://localhost:${PORT}/hls/${s.id}/index.m3u8`);
  });
});
