import { useEffect, useMemo, useRef, useState } from 'react';
import { apiOrigin, staticMediaUrl } from './staticMedia.js';
import { resolveRequestTask } from './taskMode.js';

// ── Cấu hình gốc ────────────────────────────────────────────────────

/** Địa chỉ gốc của backend API (FastAPI đang chạy trên port 8000). */
const API_ORIGIN = apiOrigin(import.meta.env.VITE_API_URL);
const API_BASE = `${API_ORIGIN}/api`;
const STATIC_MEDIA_BASE_URL = import.meta.env.VITE_STATIC_MEDIA_URL
  || (import.meta.env.PROD ? API_ORIGIN : '');

/**
 * Ba chế độ submit bài toán:
 * - KIS   : Known-Item Search — submit 1 frame duy nhất
 * - VQA   : Visual Question Answering — submit frame + câu trả lời văn bản tự do
 * - TRAKE : Temporal Retrieval — submit nhiều frame_id cùng lúc
 */
const TASK_MODES = ['KIS', 'VQA', 'TRAKE'];

/** Các modality mặc định khi mở app: tìm kiếm bằng text và object. */
const defaultModalities = ['text', 'objects'];

// ── CSV loader (Frames.csv từ /public) ──────────────────────────────

/**
 * Cache in-memory cho dữ liệu Frames.csv, tránh đọc lại nhiều lần.
 * Cấu trúc: { video_id: [{ frame_id: number, shot_id: string }, ...] }
 */
let framesDbCache = null;
const keyframeMapCache = new Map();

/**
 * Đọc file Frames.csv từ thư mục /public và parse thành object tra cứu.
 * Lần đầu gọi sẽ fetch từ server; các lần sau trả về từ cache.
 * @returns {{ [video_id: string]: { frame_id: number, shot_id: string }[] }}
 */
async function loadFramesDb() {
  if (framesDbCache) return framesDbCache; // dùng cache nếu đã có
  try {
    const res = await fetch('/Frames.csv');
    const text = await res.text();
    const lines = text.trim().split(/\r?\n/);
    const db = {}; // { video_id: [{frame_id, shot_id}, ...] }
    // Bỏ qua dòng header (i=0), đọc từng dòng dữ liệu
    for (let i = 1; i < lines.length; i++) {
      const [video_id, frame_id, shot_id] = lines[i].split(',');
      if (!video_id) continue;
      if (!db[video_id]) db[video_id] = [];
      db[video_id].push({ frame_id: parseInt(frame_id, 10), shot_id: shot_id?.trim() });
    }
    framesDbCache = db;
    return db;
  } catch {
    return {}; // trả về rỗng nếu có lỗi fetch
  }
}

async function loadKeyframeMap(videoId) {
  if (keyframeMapCache.has(videoId)) return keyframeMapCache.get(videoId);

  try {
    const response = await fetch(staticMediaUrl(`/map-keyframes/${videoId}.csv`, STATIC_MEDIA_BASE_URL));
    if (!response.ok) throw new Error('Keyframe map is unavailable');

    const [header, ...rows] = (await response.text()).trim().split(/\r?\n/);
    const columns = header.split(',');
    const keyframeIndex = columns.indexOf('n');
    const frameIndex = columns.indexOf('frame_idx');
    const mapping = rows
      .map((row) => row.split(','))
      .map((fields) => ({
        keyframeNumber: Number.parseInt(fields[keyframeIndex], 10),
        frameId: Number.parseInt(fields[frameIndex], 10),
      }))
      .filter((entry) => Number.isFinite(entry.keyframeNumber) && Number.isFinite(entry.frameId));

    keyframeMapCache.set(videoId, mapping);
    return mapping;
  } catch {
    keyframeMapCache.set(videoId, []);
    return [];
  }
}

function nearestKeyframeNumber(mapping, frameId) {
  const target = Number(frameId);
  if (!Number.isFinite(target) || !mapping.length) return null;

  return mapping.reduce(
    (closest, candidate) => (
      Math.abs(candidate.frameId - target) < Math.abs(closest.frameId - target)
        ? candidate
        : closest
    ),
  ).keyframeNumber;
}

/**
 * Tạo một hàng query phụ (extra query) mới với ID duy nhất.
 * Được dùng trong danh sách textarea "Additional query".
 */
function createExtraQueryRow(text = '') {
  return {
    id: `${Date.now()}-${Math.random().toString(36).slice(2)}`,
    text,
  };
}

/**
 * Tạo một hàng object query mới (tên vật thể + số lượng cần tìm).
 * Được dùng trong bảng Object Queries trong sidebar.
 */
function createObjectQueryRow(query = '', quantity = '1') {
  return {
    id: `${Date.now()}-${Math.random().toString(36).slice(2)}`,
    query,
    quantity,
  };
}

/**
 * Chuẩn hóa giá trị số lượng vật thể: phải là số nguyên dương.
 * Nếu không hợp lệ, trả về 1.
 */
function normalizeQuantity(value) {
  const parsed = Number.parseInt(value, 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : 1;
}

/**
 * Đọc một File object và chuyển thành chuỗi base64 Data URL.
 * Dùng để gửi ảnh cue (Visual Query) lên backend.
 */
function readFileAsDataUrl(file) {
  return new Promise((resolve, reject) => {
    if (!file) {
      resolve(null);
      return;
    }

    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(file);
  });
}

/**
 * Lấy giá trị đầu tiên không rỗng từ object `item` theo danh sách `keys` ưu tiên.
 * Hỗ trợ cả trường hợp value là mảng (lấy phần tử đầu tiên).
 * Dùng để đọc field có tên khác nhau (vd: frame_id vs frameId).
 */
function firstValue(item, keys) {
  for (const key of keys) {
    const value = item?.[key];
    if (Array.isArray(value) && value.length) return value[0];
    if (value !== undefined && value !== null && value !== '') return value;
  }
  return null;
}

/**
 * Tính đường dẫn ảnh thumbnail cho một keyframe.
 * Ưu tiên dùng trường frames_path/thumbnail từ API nếu có;
 * nếu có keyframe number đã map, ghép URL asset chính thức `.jpg`.
 */
function keyframeImagePath(keyframe) {
  const explicitPath = firstValue(keyframe, ['frames_path', 'frame_path', 'thumbnail']);
  if (explicitPath) return staticMediaUrl(explicitPath, STATIC_MEDIA_BASE_URL);

  const videoId = firstValue(keyframe, ['video_id', 'videoId']);
  const frameId = firstValue(keyframe, ['frame_id', 'frameId']);
  const keyframeNumber = firstValue(keyframe, ['keyframe_number', 'keyframeNumber']);
  if (!videoId || frameId === null) return '';

  if (keyframeNumber !== null) {
    return staticMediaUrl(
      `/keyframes/${videoId}/${String(keyframeNumber).padStart(3, '0')}.jpg`,
      STATIC_MEDIA_BASE_URL,
    );
  }

  return staticMediaUrl(`/keyframes/${videoId}/${parseInt(frameId, 10)}.png`, STATIC_MEDIA_BASE_URL);
}

/**
 * Rút gọn thông tin một keyframe thành object context nhỏ gọn
 * để gửi kèm theo search request (dùng cho temporal/staged search).
 * Trả về null nếu không tính được đường dẫn ảnh.
 */
function keyframeContext(keyframe) {
  const path = keyframeImagePath(keyframe);
  if (!path) return null;

  const score = Number(keyframe?.score);
  return {
    path,
    video_id: firstValue(keyframe, ['video_id', 'videoId']),
    frame_id: firstValue(keyframe, ['frame_id', 'frameId']),
    shot_id: firstValue(keyframe, ['shot_id', 'shotId']),
    score: Number.isFinite(score) ? score : null,
  };
}

/**
 * Chuyển toàn bộ mảng kết quả search thành mảng context object.
 * Dùng để truyền ngữ cảnh kết quả hiện tại cho lần search tiếp theo.
 */
function keyframeContexts(results) {
  if (!Array.isArray(results)) return [];
  return results
    .map((item) => keyframeContext(item))
    .filter(Boolean);
}

/**
 * Chuẩn hóa dữ liệu một frame thành payload chuẩn cho KIS submit.
 * Hỗ trợ cả camelCase lẫn snake_case từ backend.
 */
function normalizeSubmitPayload(frameLike, task = 'KIS') {
  return {
    id: firstValue(frameLike, ['id']),
    frame_id: firstValue(frameLike, ['frame_id', 'frameId']),
    shot_id: firstValue(frameLike, ['shot_id', 'shotId']),
    video_id: firstValue(frameLike, ['video_id', 'videoId']),
    task: String(task || 'KIS').toUpperCase(),
  };
}

/**
 * Tạo khóa duy nhất cho một frame trong tập TRAKE selection.
 * Định dạng: "video_id:frame_id" — dùng làm key trong object `trakedFrames`.
 */
function trakedFrameKey(frameLike) {
  const videoId = stableSubmitIdentity(firstValue(frameLike, ['video_id', 'videoId']), 'video');
  const frameId = stableSubmitIdentity(firstValue(frameLike, ['frame_id', 'frameId']), 'frame');
  return `${videoId}:${frameId}`;
}

/**
 * Chuẩn hóa một giá trị thành chuỗi ổn định để dùng làm khóa.
 * Số nguyên được parse tránh "007" ≠ "7"; chuỗi rỗng trả về fallback.
 */
function stableSubmitIdentity(value, fallback = '') {
  if (value === undefined || value === null || value === '') return fallback;

  const text = String(value).trim();
  if (/^\d+$/.test(text)) {
    return String(Number.parseInt(text, 10)); // "007" → "7"
  }

  return text;
}

/**
 * Tạo khóa duy nhất cho một submit payload để theo dõi trạng thái (loading/success/error).
 * Ưu tiên: frame_id > shot_id > id. Dùng làm key trong `submitStatusByFrame`.
 */
function submitKeyForPayload(payload) {
  const videoKey = stableSubmitIdentity(payload.video_id, 'video');

  if (payload.frame_id !== undefined && payload.frame_id !== null && payload.frame_id !== '') {
    return `${videoKey}:frame:${stableSubmitIdentity(payload.frame_id)}`;
  }

  if (payload.shot_id !== undefined && payload.shot_id !== null && payload.shot_id !== '') {
    return `${videoKey}:shot:${stableSubmitIdentity(payload.shot_id)}`;
  }

  if (payload.id !== undefined && payload.id !== null && payload.id !== '') {
    return `id:${stableSubmitIdentity(payload.id)}`;
  }

  return `${videoKey}:missing`;
}

/**
 * Shortcut: tạo submit key trực tiếp từ một frame object (không cần payload trung gian).
 */
function submitKeyForFrame(frameLike) {
  return submitKeyForPayload(normalizeSubmitPayload(frameLike));
}

/**
 * Tạo React key ổn định cho mỗi card trong result grid.
 * Kết hợp video_id + frame_id + shot_id + id + index để đảm bảo unique.
 */
function resultCardKey(keyframe, index) {
  const videoId = stableSubmitIdentity(firstValue(keyframe, ['video_id', 'videoId']), 'video');
  const frameId = stableSubmitIdentity(firstValue(keyframe, ['frame_id', 'frameId']), 'frame');
  const shotId = stableSubmitIdentity(firstValue(keyframe, ['shot_id', 'shotId']), 'shot');
  const id = stableSubmitIdentity(firstValue(keyframe, ['id']), 'result');

  return `${videoId}:${frameId}:${shotId}:${id}:${index}`;
}

/**
 * Trả về label hiển thị trên nút Submit tương ứng với trạng thái hiện tại.
 * idle → 'Submit' | loading → 'Submitting...' | success → 'Submitted' | error → 'Retry'
 */
function submitButtonLabel(status) {
  if (status === 'loading') return 'Submitting...';
  if (status === 'success') return 'Submitted';
  if (status === 'error') return 'Retry';
  return 'Submit';
}

/**
 * Ghép class name cho nút Submit kèm modifier 'ok' hoặc 'err' theo trạng thái.
 * @param {string} baseClass - class CSS cơ bản của nút
 */
function submitButtonClassName(baseClass, status) {
  return [
    baseClass,
    status === 'success' ? 'ok' : '',
    status === 'error' ? 'err' : '',
  ].filter(Boolean).join(' ');
}

/**
 * Kiểm tra payload có đủ định danh để submit hay không.
 * Cần ít nhất một trong: id, frame_id, shot_id, video_id.
 */
function hasSubmitIdentity(payload) {
  return [payload.id, payload.frame_id, payload.shot_id, payload.video_id].some(
    (value) => value !== undefined && value !== null && value !== '',
  );
}

/* ── Accordion Section Component ─────────────────────────────── */

/**
 * Component Accordion có thể đóng/mở.
 * Dùng trong sidebar để gộp các nhóm tùy chọn (Query, Visual Cues).
 * Props:
 *   - title: tiêu đề hiển thị trên thanh trigger
 *   - defaultOpen: mở sẵn khi render lần đầu (mặc định false)
 *   - children: nội dung bên trong accordion
 */
function Accordion({ title, defaultOpen = false, children }) {
  const [open, setOpen] = useState(defaultOpen);

  return (
    <div className="accordion">
      {/* Nút trigger — click để toggle đóng/mở */}
      <button
        type="button"
        className="accordion-trigger"
        onClick={() => setOpen(!open)}
      >
        <span>{title}</span>
        <span className={`arrow ${open ? 'open' : ''}`}>▼</span>
      </button>
      {/* Nội dung chỉ render khi đang mở */}
      {open && <div className="accordion-content">{children}</div>}
    </div>
  );
}

/* ── Lightbox Component ──────────────────────────────────────── */

/**
 * Component Lightbox — hiển thị ảnh phóng to toàn màn hình.
 * Hỗ trợ zoom (scroll chuột / phím +/-), kéo ảnh khi đang zoom (drag).
 * Phím tắt: ESC đóng, +/- zoom, 0 reset về 100%.
 * Props:
 *   - src: đường dẫn ảnh
 *   - alt: mô tả ảnh (accessibility)
 *   - onClose: callback khi đóng lightbox
 */
function Lightbox({ src, alt, onClose }) {
  const [scale, setScale] = useState(1);            // tỉ lệ zoom hiện tại (1 = 100%)
  const [offset, setOffset] = useState({ x: 0, y: 0 }); // vị trí pan (kéo ảnh)
  const [dragging, setDragging] = useState(false);  // đang kéo ảnh hay không
  const dragStart = useRef(null); // điểm bắt đầu kéo (để tính delta)

  useEffect(() => {
    function onKey(e) {
      if (e.key === 'Escape') onClose();
      if (e.key === '+' || e.key === '=') setScale((s) => Math.min(s + 0.25, 5));
      if (e.key === '-') setScale((s) => Math.max(s - 0.25, 0.5));
      if (e.key === '0') { setScale(1); setOffset({ x: 0, y: 0 }); }
    }
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);

  function handleWheel(e) {
    e.preventDefault();
    const delta = e.deltaY > 0 ? -0.1 : 0.1;
    setScale((s) => Math.min(Math.max(s + delta, 0.3), 6));
  }

  function handleMouseDown(e) {
    if (scale <= 1) return;
    e.preventDefault();
    setDragging(true);
    dragStart.current = { x: e.clientX - offset.x, y: e.clientY - offset.y };
  }

  function handleMouseMove(e) {
    if (!dragging || !dragStart.current) return;
    setOffset({
      x: e.clientX - dragStart.current.x,
      y: e.clientY - dragStart.current.y,
    });
  }

  function handleMouseUp() {
    setDragging(false);
    dragStart.current = null;
  }

  /** Zoom in thêm 50%, tối đa 600% */
  function zoomIn() { setScale((s) => Math.min(s + 0.5, 6)); }
  /** Zoom out 50%, tối thiểu 30% */
  function zoomOut() { setScale((s) => Math.max(s - 0.5, 0.3)); }
  /** Đặt lại về 100% và reset vị trí pan */
  function zoomReset() { setScale(1); setOffset({ x: 0, y: 0 }); }

  return (
    <div className="lightbox-overlay" onClick={onClose}>
      <div className="lightbox-controls" onClick={(e) => e.stopPropagation()}>
        <button className="lb-btn" onClick={zoomOut} title="Zoom out (-)">−</button>
        <span className="lb-scale">{Math.round(scale * 100)}%</span>
        <button className="lb-btn" onClick={zoomIn} title="Zoom in (+)">+</button>
        <button className="lb-btn" onClick={zoomReset} title="Reset (0)">⟳</button>
        <button className="lb-btn lb-close-btn" onClick={onClose} title="Close (Esc)">✕</button>
      </div>

      <div
        className="lightbox-viewport"
        onWheel={handleWheel}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        onClick={(e) => e.stopPropagation()}
        style={{ cursor: dragging ? 'grabbing' : scale > 1 ? 'grab' : 'zoom-out' }}
      >
        <img
          src={src}
          alt={alt}
          className="lightbox-img"
          style={{
            transform: `translate(${offset.x}px, ${offset.y}px) scale(${scale})`,
            transition: dragging ? 'none' : 'transform 0.15s ease',
          }}
          draggable={false}
        />
      </div>

      <p className="lightbox-hint">Scroll to zoom · Drag to pan · ESC to close</p>
    </div>
  );
}

/* ── Frame Slideshow Component ───────────────────────────────── */

/**
 * Component FrameSlideshow — hiển thị trình duyệt frame theo video.
 * Bao gồm: ảnh chính (main viewer), filmstrip thumbnail phía dưới,
 * điều hướng bằng nút ‹ › hoặc phím ← →, và khu vực submit tùy theo taskMode.
 *
 * Props:
 *   - videoId: ID video đang xem
 *   - currentFrameId: frame được chọn từ kết quả search (highlight trong strip)
 *   - taskMode: 'KIS' | 'VQA' | 'TRAKE' — ảnh hưởng đến khu vực submit
 *   - onSubmit: callback submit KIS (nhận frameLike)
 *   - onSubmitVqa: callback submit VQA (nhận frameLike, dùng answer từ vqaAnswers)
 *   - getSubmitStatus: lấy trạng thái submit của một frame (idle/loading/success/error)
 *   - getVqaAnswer: lấy câu trả lời VQA đang nhập của một frame
 *   - setVqaAnswer: cập nhật câu trả lời VQA cho một frame
 *   - isTrakeSelected: kiểm tra frame có đang được chọn trong TRAKE selection không
 *   - toggleTrakeFrame: thêm/xóa frame khỏi TRAKE selection
 */
function FrameSlideshow({
  videoId,
  currentFrameId,
  taskMode = 'KIS',
  onSubmit,
  onSubmitVqa,
  getSubmitStatus,
  getVqaAnswer,
  setVqaAnswer,
  isTrakeSelected,
  toggleTrakeFrame,
}) {
  const [frames, setFrames] = useState([]);      // danh sách tất cả frame của video này
  const [activeIdx, setActiveIdx] = useState(0); // index frame đang xem trong strip
  const [loading, setLoading] = useState(true);  // đang tải dữ liệu frame từ CSV
  const [lightboxSrc, setLightboxSrc] = useState(null); // src ảnh khi mở lightbox từ slideshow
  const [vqaInput, setVqaInput] = useState('');  // VQA answer local state — tránh controlled-input lag
  const [copied, setCopied] = useState(false);
  const stripRef = useRef(null); // ref tới filmstrip để auto-scroll tới active thumb

  // Khi videoId hoặc currentFrameId thay đổi: tải lại danh sách frame và nhảy tới frame được chọn
  useEffect(() => {
    setLoading(true);
    Promise.all([loadFramesDb(), loadKeyframeMap(videoId)]).then(([db, keyframeMap]) => {
      const videoFrames = (db[videoId] || []).map((frame) => ({
        ...frame,
        video_id: videoId,
        keyframe_number: nearestKeyframeNumber(keyframeMap, frame.frame_id),
      }));
      setFrames(videoFrames);
      const targetId = parseInt(currentFrameId, 10);
      const idx = videoFrames.findIndex((f) => f.frame_id === targetId);
      setActiveIdx(idx >= 0 ? idx : 0); // nếu không tìm thấy, về frame đầu
      setLoading(false);
    });
  }, [videoId, currentFrameId]);

  // Khi activeIdx thay đổi: auto-scroll filmstrip để thumbnail active luôn hiển thị
  useEffect(() => {
    if (!stripRef.current) return;
    const active = stripRef.current.querySelector('.strip-thumb.active');
    if (active) {
      active.scrollIntoView({ behavior: 'smooth', block: 'nearest', inline: 'center' });
    }
  }, [activeIdx]);

  // Reset VQA input khi chuyển sang frame khác
  useEffect(() => {
    setVqaInput('');
    setCopied(false);
  }, [activeIdx]);

  // Lắng nghe phím ← → để điều hướng frame không cần click chuột
  useEffect(() => {
    function onKey(e) {
      if (e.key === 'ArrowLeft') setActiveIdx((i) => Math.max(i - 1, 0));
      if (e.key === 'ArrowRight') setActiveIdx((i) => Math.min(i + 1, frames.length - 1));
    }
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [frames.length]);

  if (loading) {
    return (
      <div className="slideshow-loading">
        <div className="ss-spinner" />
        <span>Loading frames…</span>
      </div>
    );
  }

  if (!frames.length) {
    return <div className="slideshow-empty">No frame data found for this video.</div>;
  }

  const activeFrame = frames[activeIdx]; // frame đang xem trong main viewer
  const imgSrc = activeFrame ? keyframeImagePath(activeFrame) : '';
  // Frame object chuẩn hóa để dùng trong hàm submit (có video_id)
  const activeSubmitFrame = activeFrame
    ? { ...activeFrame, video_id: videoId, videoId }
    : null;
  // Trạng thái submit hiện tại của frame đang xem (idle/loading/success/error)
  const activeSubmitStatus = activeSubmitFrame && getSubmitStatus
    ? getSubmitStatus(activeSubmitFrame)
    : 'idle';

  return (
    <div className="slideshow-wrapper">
      <div className="ss-info-bar">
        <span className="ss-label">🎞 {videoId}</span>
        <span className="ss-counter">{activeIdx + 1} / {frames.length} frames</span>
        {/* Task mode badge */}
        <span className={`ss-mode-badge ss-mode-${taskMode.toLowerCase()}`}>{taskMode}</span>
      </div>

      <div className="ss-viewer">
        <button
          className="ss-nav ss-nav-prev"
          onClick={() => setActiveIdx((i) => Math.max(i - 1, 0))}
          disabled={activeIdx === 0}
          title="Previous frame (←)"
        >
          ‹
        </button>

        <div
          className="ss-main-img"
          onClick={() => setLightboxSrc(imgSrc)}
          title="Click to zoom"
        >
          <img src={imgSrc} alt={`Frame ${activeFrame?.frame_id}`} />
          <div className="ss-zoom-hint">🔍 Zoom</div>
          <div className="ss-frame-badge">
            Frame {activeFrame?.frame_id}
            {activeFrame?.frame_id === parseInt(currentFrameId, 10) && (
              <span className="ss-current-tag"> · ★ Selected</span>
            )}
          </div>
          {/* TRAKE: toggle button overlay on main image */}
          {taskMode === 'TRAKE' && activeSubmitFrame && (
            <button
              type="button"
              className={`ss-trake-overlay-btn${isTrakeSelected?.(activeSubmitFrame) ? ' checked' : ''}`}
              onClick={(e) => {
                e.stopPropagation();
                toggleTrakeFrame?.(activeSubmitFrame);
              }}
              title={isTrakeSelected?.(activeSubmitFrame) ? 'Remove from TRAKE' : 'Add to TRAKE'}
            >
              {isTrakeSelected?.(activeSubmitFrame) ? '✓ Selected' : '+ Add to TRAKE'}
            </button>
          )}
        </div>

        <button
          className="ss-nav ss-nav-next"
          onClick={() => setActiveIdx((i) => Math.min(i + 1, frames.length - 1))}
          disabled={activeIdx === frames.length - 1}
          title="Next frame (→)"
        >
          ›
        </button>
      </div>

      <div className="ss-strip" ref={stripRef}>
        {frames.map((f, idx) => {
          const thumbSrc = keyframeImagePath(f);
          const isCurrent = f.frame_id === parseInt(currentFrameId, 10);
          const isActive = idx === activeIdx;
          const frameLikeForStrip = { ...f, video_id: videoId, videoId };
          const isStripTrakeSelected = taskMode === 'TRAKE' && isTrakeSelected?.(frameLikeForStrip);
          return (
            <button
              key={f.frame_id}
              type="button"
              className={[
                'strip-thumb',
                isActive ? 'active' : '',
                isCurrent ? 'current' : '',
                isStripTrakeSelected ? 'trake-checked' : '',
              ].filter(Boolean).join(' ')}
              onClick={() => {
                setActiveIdx(idx);
                // TRAKE mode: click strip thumb also toggles selection
                if (taskMode === 'TRAKE') {
                  toggleTrakeFrame?.(frameLikeForStrip);
                }
              }}
              title={`Frame ${f.frame_id}${isCurrent ? ' (selected result)' : ''}${isStripTrakeSelected ? ' ✓ TRAKE' : ''}`}
            >
              <img src={thumbSrc} alt={`Frame ${f.frame_id}`} />
              {isCurrent && <span className="strip-dot" />}
              {isStripTrakeSelected && <span className="strip-trake-dot">✓</span>}
            </button>
          );
        })}
      </div>

      {/* ── Submit area — adapts per taskMode ── */}
      {activeSubmitFrame && (
        <div className="ss-submit-area">
          {/* KIS: single frame submit */}
          {taskMode === 'KIS' && (
            <>
              <button
                type="button"
                className={submitButtonClassName('ss-submit-big', activeSubmitStatus)}
                disabled={activeSubmitStatus === 'loading'}
                onClick={() => onSubmit?.(activeSubmitFrame)}
              >
                {submitButtonLabel(activeSubmitStatus)}
              </button>
              <span className="ss-submit-hint">
                Video {videoId} / Frame {activeFrame.frame_id}
              </span>
              {activeSubmitStatus === 'success' && (
                <div className="ss-submit-confirmation">
                  <span>Saved locally — submit this answer in the BTC portal.</span>
                  <button
                    type="button"
                    onClick={() => {
                      if (!navigator.clipboard) {
                        setCopied(false);
                        return;
                      }

                      navigator.clipboard.writeText(`${videoId} / Frame ${activeFrame.frame_id}`)
                        .then(() => setCopied(true))
                        .catch(() => setCopied(false));
                    }}
                  >
                    {copied ? 'Copied' : 'Copy answer'}
                  </button>
                </div>
              )}
            </>
          )}

          {/* VQA: answer field + submit */}
          {taskMode === 'VQA' && (
            <div className="ss-vqa-area">
              <span className="ss-submit-hint">Video {videoId} / Frame {activeFrame.frame_id}</span>
              <div className="ss-vqa-row">
                <input
                  type="text"
                  className="vqa-input ss-vqa-input"
                  placeholder="Enter answer…"
                  value={vqaInput}
                  onChange={(e) => setVqaInput(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && vqaInput.trim()) {
                      e.preventDefault();
                      // Truyền answer trực tiếp để tránh React batch state issue
                      onSubmitVqa?.(activeSubmitFrame, vqaInput);
                    }
                  }}
                  aria-label="VQA answer"
                />
                <button
                  type="button"
                  className={submitButtonClassName('ss-submit-big', activeSubmitStatus)}
                  disabled={activeSubmitStatus === 'loading' || !vqaInput.trim()}
                  onClick={() => {
                    onSubmitVqa?.(activeSubmitFrame, vqaInput);
                  }}
                >
                  {submitButtonLabel(activeSubmitStatus)}
                </button>
              </div>
            </div>
          )}

          {/* TRAKE: show current selection count + toggle button */}
          {taskMode === 'TRAKE' && (
            <div className="ss-trake-hint">
              <span>🎯 TRAKE mode — click thumbnails in the strip or the <strong>+ Add to TRAKE</strong> button on the image to select frames. Use the panel below to submit.</span>
            </div>
          )}
        </div>
      )}

      {lightboxSrc && (
        <Lightbox
          src={lightboxSrc}
          alt={`Frame ${activeFrame?.frame_id}`}
          onClose={() => setLightboxSrc(null)}
        />
      )}
    </div>
  );
}

/* ── Main App ────────────────────────────────────────────────── */

/**
 * Component gốc của toàn bộ ứng dụng BoldSearcher.
 * Quản lý mọi state và orchestrate giữa các component con.
 */
export default function App() {
  // ── State: Query / Search inputs ────────────────────────────────────
  const [query, setQuery] = useState('');             // query chính (textarea đầu tiên)
  const [extraQueries, setExtraQueries] = useState([]); // các query phụ thêm vào
  const [modalities, setModalities] = useState(defaultModalities); // ['text', 'objects', 'image']
  const [objectPanelOpen, setObjectPanelOpen] = useState(true); // bảng object query có mở không
  const [objectQueries, setObjectQueries] = useState(() => [createObjectQueryRow()]); // danh sách object query
  const [minConfidence, setMinConfidence] = useState(0.1); // ngưỡng confidence tối thiểu cho object detection
  const [imageCue, setImageCue] = useState(null); // ảnh visual cue đã upload (null = chưa có)

  // ── State: Search results ────────────────────────────────────────────
  const [results, setResults] = useState(null);       // null = chưa search, [] = search xong không có kết quả
  const [selectedKeyframe, setSelectedKeyframe] = useState(null); // frame đang mở trong detail panel
  const [usedQueries, setUsedQueries] = useState([]); // queries đã dùng cho lần search trước (dùng để detect thay đổi)
  const [isLoading, setIsLoading] = useState(false);  // đang chờ API search trả về
  const [error, setError] = useState('');             // thông báo lỗi (API không reach được)

  // ── State: UI layout ─────────────────────────────────────────────────
  const [sidebarOpen, setSidebarOpen] = useState(true); // sidebar bên trái có mở không
  const [theme, setTheme] = useState('dark');          // 'dark' hoặc 'light'

  // ── State: Task mode (chế độ bài toán) ──────────────────────────────
  /** Chế độ submit hiện tại: 'KIS' | 'VQA' | 'TRAKE' */
  const [taskMode, setTaskMode] = useState('KIS');

  // ── State: VQA ───────────────────────────────────────────────────────
  /**
   * Map lưu câu trả lời VQA cho từng frame.
   * Key: submitKey (vd: "L01_V001:frame:5"), Value: chuỗi answer
   */
  const [vqaAnswers, setVqaAnswers] = useState({});

  // ── State: TRAKE ─────────────────────────────────────────────────────
  /**
   * Map lưu các frame đã được chọn để submit TRAKE.
   * Key: trakedFrameKey (vd: "L01_V001:5"), Value: frameLike object
   */
  const [trakedFrames, setTrakedFrames] = useState({});
  /** Trạng thái submit TRAKE chung: 'idle' | 'loading' | 'success' | 'error' */
  const [trakeSubmitStatus, setTrakeSubmitStatus] = useState('idle');

  // ── State: UX ────────────────────────────────────────────────────────
  const [isDragging, setIsDragging] = useState(false); // đang kéo file ảnh vào dropzone
  /** Frame đang mở trong lightbox (quick zoom từ card), null = đóng */
  const [lightboxFrame, setLightboxFrame] = useState(null);
  /**
   * Map trạng thái submit cho từng frame (KIS và VQA).
   * Key: submitKey, Value: 'idle' | 'loading' | 'success' | 'error'
   */
  const [submitStatusByFrame, setSubmitStatusByFrame] = useState({});

  // ── Derived state (tính toán từ state hiện tại) ─────────────────────

  /** Mảng tất cả query text (gộp query chính + extra queries), đã trim và lọc rỗng */
  const queryList = useMemo(
    () =>
      [query, ...extraQueries.map((item) => item.text)]
        .map((item) => item.trim())
        .filter(Boolean),
    [query, extraQueries],
  );

  /** Danh sách object query hợp lệ (đã có tên vật thể) để gửi lên API */
  const objectPayload = useMemo(
    () =>
      objectQueries
        .map((item) => ({
          query: item.query.trim(),
          count: normalizeQuantity(item.quantity),
        }))
        .filter((item) => item.query),
    [objectQueries],
  );

  /** Danh sách frame hiển thị trong grid ([] nếu chưa search) */
  const displayedKeyframes = results ?? [];
  /** true nếu người dùng đã nhập ít nhất một query/image để có thể search */
  const hasRetrievalInput = queryList.length > 0 || objectPayload.length > 0 || Boolean(imageCue);
  /** frame_id của keyframe đang được mở trong detail panel */
  const selectedFrameId = selectedKeyframe
    ? firstValue(selectedKeyframe, ['frame_id', 'frameId'])
    : null;
  /** video_id của keyframe đang được mở trong detail panel */
  const selectedVideoId = selectedKeyframe
    ? firstValue(selectedKeyframe, ['video_id', 'videoId'])
    : null;

  // ── Side effects ─────────────────────────────────────────────────────

  // ESC: đóng lightbox trước, nếu không có lightbox thì đóng detail panel
  useEffect(() => {
    function handleKeyDown(e) {
      if (e.key === 'Escape') {
        if (lightboxFrame) { setLightboxFrame(null); return; }
        setSelectedKeyframe(null);
      }
    }
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [lightboxFrame]);

  // Cập nhật data-theme trên <html> khi người dùng đổi theme → CSS variables áp dụng ngay
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
  }, [theme]);

  // Khi người dùng xóa hết input search → reset kết quả và lỗi
  useEffect(() => {
    if (!hasRetrievalInput) {
      setError('');
      resetResultContext();
    }
  }, [hasRetrievalInput]);

  // Khi queries thay đổi không còn khớp với queries đã dùng → reset kết quả cũ
  // (tránh hiện kết quả cũ khi query mới chưa search)
  useEffect(() => {
    if (!usedQueries.length) return;
    const usedPrefixStillMatches = usedQueries.every(
      (item, index) => queryList[index] === item,
    );
    if (!usedPrefixStillMatches) {
      resetResultContext();
    }
  }, [queryList, usedQueries]);

  /**
   * Xác định loại task cho search request:
   * - KIS/VQA/TRAKE: giữ đúng mode người dùng đang chọn cho text search
   * - submit ảnh vẫn dùng mode hiện tại; visual search tự gửi VKIS tại request body
   */
  function getRequestTask() {
    return resolveRequestTask(taskMode);
  }

  /**
   * Lấy trạng thái submit (idle/loading/success/error) của một frame cụ thể.
   * Dùng để hiển thị đúng label và màu sắc nút Submit.
   */
  function getSubmitStatus(frameLike) {
    return submitStatusByFrame[submitKeyForFrame(frameLike)] || 'idle';
  }

  // ── Task mode change ────────────────────────────────────────
  /**
   * Đổi chế độ bài toán (KIS/VQA/TRAKE).
   * Reset toàn bộ trạng thái submit cũ để tránh lẫn lộn giữa các chế độ.
   */
  function handleTaskModeChange(mode) {
    setTaskMode(mode);
    setVqaAnswers({});         // xóa tất cả câu trả lời VQA
    setTrakedFrames({});       // xóa tất cả frame đã chọn TRAKE
    setTrakeSubmitStatus('idle');
    setSubmitStatusByFrame({}); // xóa trạng thái submit của KIS/VQA
  }

  // ── VQA helpers ─────────────────────────────────────────────

  /** Lấy câu trả lời VQA đang nhập cho một frame cụ thể. */
  function getVqaAnswer(frameLike) {
    return vqaAnswers[submitKeyForFrame(frameLike)] || '';
  }

  /** Cập nhật câu trả lời VQA cho một frame (khi người dùng gõ vào input). */
  function setVqaAnswer(frameLike, value) {
    const key = submitKeyForFrame(frameLike);
    setVqaAnswers((curr) => ({ ...curr, [key]: value }));
  }

  /**
   * Submit VQA: gửi video_id + frame_id + answer lên POST /api/search/submit/vqa.
   * Nhận answer trực tiếp qua tham số để tránh vấn đề React batch state update.
   */
  async function submitVQA(frameLike, answerDirect) {
    const videoId = firstValue(frameLike, ['video_id', 'videoId']);
    const frameId = firstValue(frameLike, ['frame_id', 'frameId']);
    // Dùng answerDirect nếu được truyền, fallback về state lookup
    const answer = answerDirect ?? getVqaAnswer(frameLike);
    const submitKey = submitKeyForFrame(frameLike);

    if (!videoId || frameId === null || frameId === undefined || !answer) return;
    if (submitStatusByFrame[submitKey] === 'loading') return; // chống double-submit

    const payload = { video_id: videoId, frame_id: frameId, answer, task: 'VQA' };
    console.log('%c[VQA Submit] Payload:', 'color:#a78bfa;font-weight:bold', payload);

    setSubmitStatusByFrame((curr) => ({ ...curr, [submitKey]: 'loading' }));
    try {
      const response = await fetch(`${API_BASE}/search/submit/vqa`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (!response.ok) throw new Error('VQA Submit API returned an error');
      await response.json();
      // Clear VQA answer and status after successful submit
      setVqaAnswers({});
      setSubmitStatusByFrame({});
    } catch {
      setSubmitStatusByFrame((curr) => ({ ...curr, [submitKey]: 'error' }));
    }
  }

  // ── TRAKE helpers ────────────────────────────────────────────

  /** Kiểm tra một frame có đang nằm trong tập TRAKE selection không. */
  function isTrakeSelected(frameLike) {
    return Boolean(trakedFrames[trakedFrameKey(frameLike)]);
  }

  /**
   * Thêm hoặc xóa một frame khỏi tập TRAKE selection (toggle).
   * Nếu đã có → xóa; nếu chưa có → thêm vào.
   */
  function toggleTrakeFrame(frameLike) {
    const key = trakedFrameKey(frameLike);
    setTrakedFrames((curr) => {
      const next = { ...curr };
      if (next[key]) {
        delete next[key]; // bỏ chọn
      } else {
        next[key] = frameLike; // chọn thêm
      }
      return next;
    });
  }

  /**
   * Submit TRAKE: gom tất cả frame đã chọn, nhóm theo video_id,
   * rồi gửi mỗi nhóm lên POST /api/search/submit/trake.
   * Mỗi request có body: { video_id, frame_ids: [...], task: 'TRAKE' }
   * Log toàn bộ payload ra console trước khi gọi API.
   */
  async function submitTrake() {
    const frames = Object.values(trakedFrames);
    if (!frames.length || trakeSubmitStatus === 'loading') return;

    // Nhóm frame_id theo video_id, mỗi frame chỉ xuất hiện 1 lần (đã đảm bảo bởi trakedFrameKey)
    const byVideo = {};
    for (const f of frames) {
      const vid = firstValue(f, ['video_id', 'videoId']) || '';
      if (!byVideo[vid]) byVideo[vid] = [];
      byVideo[vid].push(parseInt(firstValue(f, ['frame_id', 'frameId']), 10));
    }
    // Sắp xếp frame_ids theo thứ tự tăng dần trong mỗi video
    for (const vid of Object.keys(byVideo)) {
      byVideo[vid].sort((a, b) => a - b);
    }

    const payloads = Object.entries(byVideo).map(([video_id, frame_ids]) => ({
      video_id,
      frame_ids,
      task: 'TRAKE',
    }));
    console.log('%c[TRAKE Submit] Payloads:', 'color:#fbbf24;font-weight:bold', payloads);

    setTrakeSubmitStatus('loading');
    try {
      // Gửi song song tất cả payload (1 payload / video)
      await Promise.all(
        payloads.map((payload) =>
          fetch(`${API_BASE}/search/submit/trake`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
          }).then((r) => { if (!r.ok) throw new Error('TRAKE API error'); })
        )
      );
      // Clear TRAKE selection and status after successful submit
      setTrakedFrames({});
      setTrakeSubmitStatus('idle');
      setSubmitStatusByFrame({});
    } catch {
      setTrakeSubmitStatus('error');
    }
  }

  /**
   * Tạo label cho nút Submit All của TRAKE panel:
   * Hiển thị số frame đã chọn và trạng thái submit.
   */
  function trakeSubmitLabel() {
    if (trakeSubmitStatus === 'loading') return 'Submitting…';
    if (trakeSubmitStatus === 'success') return '✓ Submitted';
    if (trakeSubmitStatus === 'error') return 'Retry';
    const n = Object.keys(trakedFrames).length;
    return `Submit ${n} frame${n !== 1 ? 's' : ''}`;
  }

  /**
   * Submit KIS: gửi video_id + frame_id lên POST /api/search/submit/kis.
   * Không submit nếu payload thiếu định danh hoặc đang loading.
   * Log payload ra console trước khi gọi API.
   */
  async function submitFrame(frameLike) {
    const payload = normalizeSubmitPayload(frameLike, getRequestTask());
    const submitKey = submitKeyForPayload(payload);

    if (!hasSubmitIdentity(payload) || submitStatusByFrame[submitKey] === 'loading') return;

    console.log('%c[KIS Submit] Payload:', 'color:#5cf5d0;font-weight:bold', payload);

    setSubmitStatusByFrame((current) => ({
      ...current,
      [submitKey]: 'loading',
    }));

    try {
      const response = await fetch(`${API_BASE}/search/submit/kis`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        throw new Error('Submit API returned an error');
      }

      await response.json();
      setSubmitStatusByFrame((current) => ({
        ...current,
        [submitKey]: 'success',
      }));

    } catch {
      setSubmitStatusByFrame((current) => ({
        ...current,
        [submitKey]: 'error',
      }));
    }
  }

  /**
   * Reset toàn bộ context kết quả tìm kiếm:
   * xóa kết quả, frame đang chọn, trạng thái submit, VQA answers, TRAKE selection.
   * Gọi khi: query thay đổi, xóa input, hoặc đổi task mode.
   */
  function resetResultContext() {
    console.log('Resetting result context');
    setResults(null);
    setSelectedKeyframe(null);
    setSubmitStatusByFrame({});
    setVqaAnswers({});
    setTrakedFrames({});
    setTrakeSubmitStatus('idle');
    setUsedQueries((current) => (current.length ? [] : current));
  }

  /**
   * Xóa toàn bộ input tìm kiếm (query, extra queries, objects, image cue)
   * và reset kết quả. Đồng thời giải phóng bộ nhớ cho ảnh preview đã upload.
   */
  function clearRetrievalInputs() {
    setQuery('');
    setExtraQueries([]);
    setObjectQueries([createObjectQueryRow()]);
    setObjectPanelOpen(true);
    if (imageCue?.preview) {
      URL.revokeObjectURL(imageCue.preview); // giải phóng Object URL blob
    }
    setImageCue(null);
    setModalities(defaultModalities);
    setError('');
    resetResultContext();
  }

  /**
   * Hàm tìm kiếm chính — gửi request lên backend và cập nhật kết quả.
   *
   * Hai chế độ:
   * 1. Visual query (có imageCue): POST /search/visual_query với ảnh base64
   * 2. Text/object query: POST /search/query với text + object queries
   *
   * Staged search: nếu đã có kết quả trước đó, truyền thêm frames_context
   * để backend thu hẹp không gian tìm kiếm (temporal refinement).
   */
  async function runSearch(event) {
    event?.preventDefault();
    setError('');

    if (!hasRetrievalInput) {
      resetResultContext();
      return;
    }

    setIsLoading(true);

    try {
      const hasImageReference = Boolean(imageCue);
      // Chọn endpoint phù hợp với loại query
      const endpoint = hasImageReference ? '/search/visual_query' : '/search/query';
      // Lấy context frames hiện tại để dùng cho staged/temporal search
      const currentFrameContext = keyframeContexts(displayedKeyframes);
      const framesPath = currentFrameContext.length
        ? currentFrameContext.map((item) => item.path)
        : null;
      // Chuẩn bị payload ảnh nếu có image cue
      const imageCuePayload = imageCue
        ? {
          name: imageCue.name,
          size: imageCue.size,
          type: imageCue.type,
          dataUrl: await readFileAsDataUrl(imageCue.file), // chuyển File → base64
        }
        : null;
      // Xây dựng request body tùy theo chế độ
      const requestBody = hasImageReference
        ? {
          task: resolveRequestTask(taskMode, true),
          minConfidence,
          topK: 100,
          imageCue: imageCuePayload,
        }
        : {
          query: query.trim(),
          queries: queryList,           // tất cả query text
          task: getRequestTask(),
          topK: 100,
          modalities,
          objects: objectPayload.map((item) => item.query),
          objectQueries: objectPayload, // object + số lượng
          minConfidence,
          frames_path: framesPath,      // null nếu lần đầu search
          used_queries: framesPath ? usedQueries : [],      // queries đã dùng ở lần trước
          frames_context: framesPath ? currentFrameContext : [], // context frames lần trước
        };

      const response = await fetch(`${API_BASE}${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestBody),
      });

      if (!response.ok) {
        throw new Error('Search API returned an error');
      }

      const data = await response.json();
      setResults(data.results || []);
      setSelectedKeyframe(null);
      setSubmitStatusByFrame({});
      // Lưu lại queries vừa dùng để detect thay đổi cho lần sau
      setUsedQueries(hasImageReference ? [] : queryList);
    } catch {
      setError('Could not reach the API. Start the backend on port 8000.');
      setResults([]);
      setSelectedKeyframe(null);
      setSubmitStatusByFrame({});
      setUsedQueries([]);
    } finally {
      setIsLoading(false);
    }
  }

  /** Toggle một giá trị vào/ra khỏi một mảng (dùng cho modalities). */
  function toggleValue(value, selected, setter) {
    setter(
      selected.includes(value)
        ? selected.filter((item) => item !== value)
        : [...selected, value],
    );
  }

  // ── Extra query CRUD ─────────────────────────────────────────
  /** Thêm một textarea query phụ mới vào danh sách. */
  function addExtraQuery() {
    setExtraQueries((current) => [...current, createExtraQueryRow()]);
  }

  /** Cập nhật nội dung text của một extra query theo id. */
  function updateExtraQuery(id, value) {
    setExtraQueries((current) =>
      current.map((item) => (item.id === id ? { ...item, text: value } : item)),
    );
  }

  /** Xóa một extra query khỏi danh sách theo id. */
  function removeExtraQuery(id) {
    setExtraQueries((current) => current.filter((item) => item.id !== id));
  }

  // ── Object query CRUD ────────────────────────────────────────
  /** Thêm một hàng object query mới và mở panel Objects nếu đang đóng. */
  function addObjectQuery() {
    setObjectPanelOpen(true);
    setObjectQueries((current) => [...current, createObjectQueryRow()]);
  }

  /** Cập nhật một field (query hoặc quantity) của object query theo id. */
  function updateObjectQuery(id, field, value) {
    setObjectQueries((current) =>
      current.map((item) =>
        item.id === id ? { ...item, [field]: value } : item,
      ),
    );
  }

  /** Xóa một object query; nếu xóa hết thì tự tạo lại 1 hàng rỗng. */
  function removeObjectQuery(id) {
    setObjectQueries((current) => {
      const next = current.filter((item) => item.id !== id);
      return next.length ? next : [createObjectQueryRow()];
    });
  }

  /**
   * Toggle panel Objects và đảm bảo modality 'objects' luôn được bật
   * khi panel đang mở.
   */
  function handleObjectPanelToggle() {
    setObjectPanelOpen((open) => !open);
    setModalities((current) =>
      current.includes('objects') ? current : [...current, 'objects'],
    );
  }

  /**
   * Toggle một modality (text/objects/image).
   * Đặc biệt: khi toggle 'objects' cũng đóng/mở panel object queries.
   */
  function handleModalityToggle(value) {
    const isActive = modalities.includes(value);
    if (value === 'objects') {
      setObjectPanelOpen(!isActive);
    }
    toggleValue(value, modalities, setModalities);
  }

  /**
   * Xử lý file ảnh được chọn làm visual cue:
   * tạo Object URL để preview và thêm modality 'image'.
   */
  function processImageFile(file) {
    if (!file) return;
    if (imageCue?.preview) {
      URL.revokeObjectURL(imageCue.preview); // giải phóng URL blob cũ
    }
    setImageCue({
      file,
      name: file.name,
      size: file.size,
      type: file.type,
      preview: URL.createObjectURL(file), // tạo URL blob để hiện preview
    });
    setModalities((current) =>
      current.includes('image') ? current : [...current, 'image'],
    );
  }

  /** Xóa ảnh visual cue và bỏ modality 'image'. */
  function clearImageCue() {
    if (imageCue?.preview) {
      URL.revokeObjectURL(imageCue.preview);
    }
    setImageCue(null);
    setModalities((current) => current.filter((item) => item !== 'image'));
  }

  /** Handler cho input[type=file] — lấy file đầu tiên được chọn. */
  function handleImageCue(event) {
    processImageFile(event.target.files?.[0]);
  }

  // ── Drag & Drop handlers cho dropzone ──────────────────────
  function handleDragOver(e) {
    e.preventDefault();
    setIsDragging(true); // hiển thị visual feedback khi kéo file vào
  }

  function handleDragLeave(e) {
    e.preventDefault();
    setIsDragging(false);
  }

  /** Xử lý thả file vào dropzone — chỉ chấp nhận file ảnh. */
  function handleDrop(e) {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file && file.type.startsWith('image/')) {
      processImageFile(file);
    }
  }

  /** Mở detail panel cho một keyframe khi click vào card. */
  function handleSelectKeyframe(keyframe) {
    setSelectedKeyframe(keyframe);
  }

  /**
   * Xử lý keyboard trên card (accessibility):
   * Enter hoặc Space mở detail panel (chỉ khi focus đúng vào card).
   */
  function handleCardKeyDown(event, keyframe) {
    if (event.target !== event.currentTarget) return; // bỏ qua nếu focus vào button con

    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      handleSelectKeyframe(keyframe);
    }
  }

  return (
    <div className="app-shell">
      {/* ── Sidebar ──────────────────────────────────────────── */}
      <aside className={`sidebar ${sidebarOpen ? '' : 'collapsed'}`}>
        <div className="sidebar-header">
          <h1>BoldSearcher</h1>
          <p className="subtitle">A Multi-modal Video Retrieval System</p>
        </div>

        <form className="sidebar-body" onSubmit={runSearch}>
          {/* Section 1: Query */}
          <Accordion title="Query" defaultOpen>
            <div className="query-stack">
              <div className="query-wrapper">
                <textarea
                  id="query"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                      e.preventDefault();
                      runSearch();
                    }
                  }}
                  placeholder="Describe the target keyframe..."
                />
                <div className="query-actions">
                  {query && (
                    <button type="button" className="btn-icon" onClick={() => setQuery('')} title="Clear">✕</button>
                  )}
                </div>
              </div>

              {extraQueries.map((item, index) => (
                <div className="query-wrapper" key={item.id}>
                  <textarea
                    value={item.text}
                    onChange={(e) => updateExtraQuery(item.id, e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' && !e.shiftKey) {
                        e.preventDefault();
                        runSearch();
                      }
                    }}
                    placeholder={`Additional query ${index + 1}`}
                  />
                  <div className="query-actions">
                    <button
                      type="button"
                      className="btn-icon"
                      onClick={() => removeExtraQuery(item.id)}
                      title="Remove query"
                    >
                      ✕
                    </button>
                  </div>
                </div>
              ))}

              <button type="button" className="btn-secondary" onClick={addExtraQuery}>
                + Add query
              </button>
            </div>
          </Accordion>

          {/* Section 2: Visual Cues */}
          <Accordion title="Visual Cues" defaultOpen>
            <div className="filter-group">
              <label className="field-label">Image Reference</label>
              <div
                className={`dropzone ${isDragging ? 'dragging' : ''}`}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
              >
                <input
                  id="image-cue"
                  type="file"
                  accept="image/*"
                  onChange={handleImageCue}
                  className="hidden-file-input"
                />
                {!imageCue ? (
                  <label htmlFor="image-cue" className="dropzone-label">
                    <span>Drag &amp; drop image here</span>
                    <small>or click to browse</small>
                  </label>
                ) : (
                  <div className="image-cue-preview">
                    <img src={imageCue.preview} alt="Uploaded visual cue" />
                    <div className="image-info">
                      <span>{imageCue.name}</span>
                      <button type="button" className="btn-remove-img" onClick={clearImageCue}>✕</button>
                    </div>
                  </div>
                )}
              </div>
            </div>

            <div className="filter-group" style={{ marginTop: 16 }}>
              <button
                type="button"
                className={objectPanelOpen ? 'panel-trigger active' : 'panel-trigger'}
                onClick={handleObjectPanelToggle}
              >
                <span>
                  <strong>Objects</strong>
                  <small>
                    {objectPayload.length
                      ? `${objectPayload.length} object quer${objectPayload.length === 1 ? 'y' : 'ies'}`
                      : 'No object queries'}
                  </small>
                </span>
                <span className={`arrow ${objectPanelOpen ? 'open' : ''}`}>▼</span>
              </button>

              {objectPanelOpen && (
                <div className="object-panel">
                  <div className="object-table">
                    <div className="object-table-head">
                      <span>Object query</span>
                      <span>Quantity</span>
                      <span />
                    </div>

                    {objectQueries.map((item, index) => (
                      <div className="object-table-row" key={item.id}>
                        <input
                          type="text"
                          value={item.query}
                          onChange={(e) =>
                            updateObjectQuery(item.id, 'query', e.target.value)
                          }
                          placeholder={index === 0 ? 'e.g. car' : 'Object query'}
                        />
                        <input
                          type="number"
                          min="1"
                          step="1"
                          inputMode="numeric"
                          value={item.quantity}
                          onChange={(e) =>
                            updateObjectQuery(item.id, 'quantity', e.target.value)
                          }
                          aria-label="Quantity"
                        />
                        <button
                          type="button"
                          className="btn-row-remove"
                          onClick={() => removeObjectQuery(item.id)}
                          title="Remove object query"
                        >
                          ✕
                        </button>
                      </div>
                    ))}
                  </div>

                  <button type="button" className="btn-secondary" onClick={addObjectQuery}>
                    + Add object
                  </button>
                </div>
              )}
            </div>
          </Accordion>
        </form>

        <div className="sidebar-footer">
          <button
            className="btn-primary"
            type="button"
            disabled={isLoading}
            onClick={runSearch}
          >
            {isLoading ? 'Searching…' : 'Search keyframes'}
          </button>
        </div>
      </aside>

      {/* ── Main Content ─────────────────────────────────────── */}
      <div className="main-content">
        {/* Top bar */}
        <header className="topbar">
          <div className="topbar-left">
            <button
              className="btn-toggle"
              type="button"
              onClick={() => setSidebarOpen(!sidebarOpen)}
              aria-label="Toggle filters"
              title="Toggle filters"
            >
              ☰
            </button>
            {!sidebarOpen && (
              <span className="topbar-title">BoldSearcher</span>
            )}
          </div>

          <div className="topbar-right">
            {/* Task Mode Selector */}
            <div className="task-mode-selector" role="group" aria-label="Task mode">
              {TASK_MODES.map((mode) => (
                <button
                  key={mode}
                  type="button"
                  className={`task-mode-btn${taskMode === mode ? ' active' : ''}`}
                  onClick={() => handleTaskModeChange(mode)}
                  title={`Switch to ${mode} mode`}
                >
                  {mode}
                </button>
              ))}
            </div>

            <button
              type="button"
              className={`theme-switch ${theme}`}
              onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
              title="Toggle theme"
            >
              <span className="theme-switch-thumb">
                {theme === 'dark' ? '🌙' : '☀️'}
              </span>
            </button>
            <span className="result-count">
              {displayedKeyframes.length} result{displayedKeyframes.length !== 1 ? 's' : ''}
            </span>
          </div>
        </header>

        {/* Results */}
        <div className="results-wrapper">
          {error && <div className="notice error">{error}</div>}

          {!isLoading && displayedKeyframes.length === 0 && !error && (
            <div className="empty-results">
              <div className="empty-icon">🔍</div>
              <h3>No results found</h3>
              <p>
                Adjust your query or filters in the sidebar and run a search.
              </p>
            </div>
          )}

          <div className="result-grid" aria-live="polite">
            {isLoading
              ? Array.from({ length: 8 }).map((_, index) => (
                <div className="shot-card skeleton" key={index} />
              ))
              : displayedKeyframes.map((keyframe, index) => {
                const imageSrc = keyframe.thumbnail || keyframeImagePath(keyframe);
                const frameId = firstValue(keyframe, ['frame_id', 'frameId']);
                const videoId = firstValue(keyframe, ['video_id', 'videoId']);
                const submitStatus = getSubmitStatus(keyframe);
                const trakeSelected = isTrakeSelected(keyframe);
                const vqaAnswer = getVqaAnswer(keyframe);

                return (
                  <div
                    key={resultCardKey(keyframe, index)}
                    role="button"
                    tabIndex={0}
                    className={[
                      'shot-card',
                      // So sánh bằng submitKey (video_id + frame_id) thay vì .id có thể undefined
                      selectedKeyframe && submitKeyForFrame(selectedKeyframe) === submitKeyForFrame(keyframe) ? 'active' : '',
                      taskMode === 'TRAKE' && trakeSelected ? 'trake-selected' : '',
                    ].filter(Boolean).join(' ')}
                    onClick={() => handleSelectKeyframe(keyframe)}
                    onKeyDown={(event) => handleCardKeyDown(event, keyframe)}
                    aria-label={`Open details for frame ${frameId}`}
                  >

                    <div className="shot-card-img-wrap">
                      <img src={imageSrc} alt={keyframe.title || `Frame ${frameId}`} />
                      <button
                        type="button"
                        className="card-zoom-btn"
                        title="Zoom image"
                        aria-label={`Zoom frame ${frameId}`}
                        onClick={(e) => {
                          e.stopPropagation();
                          setLightboxFrame({
                            src: imageSrc,
                            alt: keyframe.title || `Frame ${frameId}`,
                          });
                        }}
                      >
                        🔍
                      </button>
                    </div>

                    <div className="shot-meta">
                      <span>Video: {videoId}</span>
                      <strong><i>Frame:</i> {frameId}</strong>

                      {/* KIS: normal submit button */}
                      {taskMode === 'KIS' && (
                        <button
                          type="button"
                          className={submitButtonClassName('card-submit-btn', submitStatus)}
                          disabled={submitStatus === 'loading'}
                          onClick={(e) => { e.stopPropagation(); submitFrame(keyframe); }}
                        >
                          {submitButtonLabel(submitStatus)}
                        </button>
                      )}

                      {/* VQA: show a hint to open detail panel for answering */}
                      {taskMode === 'VQA' && (
                        <div className="vqa-answer-field" onClick={(e) => e.stopPropagation()}>
                          <span className="vqa-card-hint">Click card to answer</span>
                        </div>
                      )}

                      {/* TRAKE: no button on card — select inside detail panel */}
                    </div>

                    <div className="score-pill" data-tooltip={keyframe.reasons?.join('\n') || ''}>
                      {Math.round(keyframe.score * 100)}%
                    </div>
                  </div>
                );
              })}
          </div>
        </div>
      </div>

      {/* ── Lightbox (card quick zoom) ────────────────────────── */}
      {lightboxFrame && (
        <Lightbox
          src={lightboxFrame.src}
          alt={lightboxFrame.alt}
          onClose={() => setLightboxFrame(null)}
        />
      )}

      {/* ── Detail Modal (Centered Popup) ─────────────────────── */}
      {selectedKeyframe && (
        <div
          className="detail-overlay"
          onClick={() => setSelectedKeyframe(null)}
        >
          <div
            className="detail-modal"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="detail-header">
              <h3>Keyframe Details — {selectedVideoId} / Frame {selectedFrameId}</h3>
              <button
                className="btn-close"
                type="button"
                onClick={() => setSelectedKeyframe(null)}
              >
                ✕
              </button>
            </div>

            <div className="detail-scroll">
              {/* Main preview — click to zoom */}
              <div
                className="detail-preview detail-preview-clickable"
                onClick={() => {
                  const src = selectedKeyframe.thumbnail || keyframeImagePath(selectedKeyframe);
                  setLightboxFrame({ src, alt: selectedKeyframe.title || `Frame ${selectedFrameId}` });
                }}
                title="Click to zoom"
              >
                <img
                  src={selectedKeyframe.thumbnail || keyframeImagePath(selectedKeyframe)}
                  alt={selectedKeyframe.title}
                />
                <div className="play-overlay">🔍 Click to zoom</div>
              </div>

              <div className="detail-body">
                <p className="eyebrow">{selectedVideoId}</p>
                <h2>{`Frame ${selectedFrameId}`}</h2>
                {selectedKeyframe.description && <p>{selectedKeyframe.description}</p>}

                {/* Slideshow — always visible */}
                <FrameSlideshow
                  videoId={selectedVideoId}
                  currentFrameId={selectedFrameId}
                  taskMode={taskMode}
                  onSubmit={submitFrame}
                  onSubmitVqa={submitVQA}
                  getSubmitStatus={getSubmitStatus}
                  getVqaAnswer={getVqaAnswer}
                  setVqaAnswer={setVqaAnswer}
                  isTrakeSelected={isTrakeSelected}
                  toggleTrakeFrame={toggleTrakeFrame}
                />

                {/* TRAKE: multi-frame selection panel */}
                {taskMode === 'TRAKE' && (
                  <div className="trake-panel">
                    <div className="trake-panel-header">
                      <span className="trake-panel-title">🎯 TRAKE Selection</span>
                      <span className="trake-panel-count">
                        {Object.keys(trakedFrames).length} frame{Object.keys(trakedFrames).length !== 1 ? 's' : ''} selected
                      </span>
                    </div>

                    {Object.keys(trakedFrames).length === 0 ? (
                      <p className="trake-panel-empty">Click cards in the grid or use the slideshow to select frames.</p>
                    ) : (
                      <div className="trake-frame-list">
                        {Object.values(trakedFrames)
                          // Sắp xếp theo frame_id tăng dần
                          .sort((a, b) => {
                            const fa = parseInt(firstValue(a, ['frame_id', 'frameId']), 10) || 0;
                            const fb = parseInt(firstValue(b, ['frame_id', 'frameId']), 10) || 0;
                            return fa - fb;
                          })
                          .map((f) => {
                            const vid = firstValue(f, ['video_id', 'videoId']);
                            const fid = parseInt(firstValue(f, ['frame_id', 'frameId']), 10);
                            return (
                              <div key={trakedFrameKey(f)} className="trake-frame-chip">
                                <span className="trake-chip-vid">{vid}</span>
                                <span className="trake-chip-fid">#{fid}</span>
                                <button
                                  type="button"
                                  className="trake-chip-remove"
                                  onClick={() => toggleTrakeFrame(f)}
                                  title="Remove"
                                >✕</button>
                              </div>
                            );
                          })}
                      </div>
                    )}

                    <div className="trake-panel-actions">
                      {Object.keys(trakedFrames).length > 0 && (
                        <button
                          type="button"
                          className="btn-secondary trake-clear-btn"
                          onClick={() => { setTrakedFrames({}); setTrakeSubmitStatus('idle'); }}
                        >
                          Clear all
                        </button>
                      )}
                      <button
                        type="button"
                        className={[
                          'btn-primary trake-submit-all-btn',
                          trakeSubmitStatus === 'success' ? 'ok' : '',
                          trakeSubmitStatus === 'error' ? 'err' : '',
                        ].filter(Boolean).join(' ')}
                        disabled={Object.keys(trakedFrames).length === 0 || trakeSubmitStatus === 'loading'}
                        onClick={submitTrake}
                      >
                        {trakeSubmitLabel()}
                      </button>
                    </div>
                  </div>
                )}

                {selectedKeyframe.start != null && (
                  <div className="timebar">
                    <span>{selectedKeyframe.start}</span>
                    <div>
                      <i
                        style={{
                          width: `${Math.min(selectedKeyframe.duration * 7, 100)}%`,
                        }}
                      />
                    </div>
                    <span>{selectedKeyframe.end}</span>
                  </div>
                )}

                <div className="metadata-list">
                  {selectedKeyframe.objects?.length > 0 && (
                    <div>
                      <span>Objects</span>
                      <strong>{selectedKeyframe.objects.join(', ')}</strong>
                    </div>
                  )}
                  {selectedKeyframe.transcript && (
                    <div>
                      <span>Transcript</span>
                      <strong>{selectedKeyframe.transcript}</strong>
                    </div>
                  )}
                </div>

                {selectedKeyframe.reasons?.length > 0 && (
                  <div className="reason-list">
                    {selectedKeyframe.reasons.map((reason) => (
                      <span key={reason}>{reason}</span>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
