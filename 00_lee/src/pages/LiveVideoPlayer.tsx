// HLS 스트림을 재생하는 공용 Player 컴포넌트 가져오기
import HlsPlayer from "../components/HlsPlayer";

// 테스트용 Dashboard 페이지
export default function Dashboard() {
  return (
    // 화면을 좌우 50:50으로 나누고 spacing을 줌
    <div style={{ display: "flex", gap: "12px" }}>
      
      {/* 왼쪽: Camera 01 */}
      <div style={{ width: "50%" }}>
        <h2>Camera 01</h2>

        {/* HlsPlayer에 cam01의 스트리밍 URL 전달 */}
        <HlsPlayer src="https://pmhmdhwetxzhngkw.tunnel.elice.io/hls/cam01/index.m3u8" />

      </div>

      {/* 오른쪽: Camera 02 */}
      <div style={{ width: "50%" }}>
        <h2>Camera 02</h2>

        {/* HlsPlayer에 cam02의 스트리밍 URL 전달 */}
        <HlsPlayer src="https://pmhmdhwetxzhngkw.tunnel.elice.io/hls/cam02/index.m3u8" />
      </div>

    </div>
  );
}
