import { useEffect, useMemo, useRef, useState } from 'react';

const API_BASE = 'http://0.0.0.0:8000/api';

const defaultModalities = ['text', 'objects'];

// ── CSV loader (Frames.csv from /public) ─────────────────────────────

let framesDbCache = null;

async function loadFramesDb() {
  if (framesDbCache) return framesDbCache;
  try {
    const res = await fetch('/Frames.csv');
    const text = await res.text();
    const lines = text.trim().split(/\r?\n/);
    const db = {}; // { video_id: [{frame_id, shot_id}, ...] }
    for (let i = 1; i < lines.length; i++) {
      const [video_id, frame_id, shot_id] = lines[i].split(',');
      if (!video_id) continue;
      if (!db[video_id]) db[video_id] = [];
      db[video_id].push({ frame_id: parseInt(frame_id, 10), shot_id: shot_id?.trim() });
    }
    framesDbCache = db;
    return db;
  } catch {
    return {};
  }
}

function createExtraQueryRow(text = '') {
  return {
    id: `${Date.now()}-${Math.random().toString(36).slice(2)}`,
    text,
  };
}

function createObjectQueryRow(query = '', quantity = '1') {
  return {
    id: `${Date.now()}-${Math.random().toString(36).slice(2)}`,
    query,
    quantity,
  };
}

function normalizeQuantity(value) {
  const parsed = Number.parseInt(value, 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : 1;
}

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

function firstValue(item, keys) {
  for (const key of keys) {
    const value = item?.[key];
    if (Array.isArray(value) && value.length) return value[0];
    if (value !== undefined && value !== null && value !== '') return value;
  }
  return null;
}

function keyframeImagePath(keyframe) {
  const explicitPath = firstValue(keyframe, ['frames_path', 'frame_path', 'thumbnail']);
  if (explicitPath) return String(explicitPath);

  const videoId = firstValue(keyframe, ['video_id', 'videoId']);
  const frameId = firstValue(keyframe, ['frame_id', 'frameId']);
  if (!videoId || frameId === null) return '';

  const frameFile = String(frameId).padStart(3, '0');
  return `/keyframes/${videoId}/${frameFile}.png`;
}

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

function keyframeContexts(results) {
  if (!Array.isArray(results)) return [];
  return results
    .map((item) => keyframeContext(item))
    .filter(Boolean);
}

/* ── Accordion Section Component ─────────────────────────────── */

function Accordion({ title, defaultOpen = false, children }) {
  const [open, setOpen] = useState(defaultOpen);

  return (
    <div className="accordion">
      <button
        type="button"
        className="accordion-trigger"
        onClick={() => setOpen(!open)}
      >
        <span>{title}</span>
        <span className={`arrow ${open ? 'open' : ''}`}>▼</span>
      </button>
      {open && <div className="accordion-content">{children}</div>}
    </div>
  );
}

/* ── Lightbox Component ──────────────────────────────────────── */

function Lightbox({ src, alt, onClose }) {
  const [scale, setScale] = useState(1);
  const [offset, setOffset] = useState({ x: 0, y: 0 });
  const [dragging, setDragging] = useState(false);
  const dragStart = useRef(null);

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

  function zoomIn() { setScale((s) => Math.min(s + 0.5, 6)); }
  function zoomOut() { setScale((s) => Math.max(s - 0.5, 0.3)); }
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

function FrameSlideshow({ videoId, currentFrameId, onClose }) {
  const [frames, setFrames] = useState([]);
  const [activeIdx, setActiveIdx] = useState(0);
  const [loading, setLoading] = useState(true);
  const [lightboxSrc, setLightboxSrc] = useState(null);
  const stripRef = useRef(null);

  useEffect(() => {
    setLoading(true);
    loadFramesDb().then((db) => {
      const videoFrames = db[videoId] || [];
      setFrames(videoFrames);
      const targetId = parseInt(currentFrameId, 10);
      const idx = videoFrames.findIndex((f) => f.frame_id === targetId);
      setActiveIdx(idx >= 0 ? idx : 0);
      setLoading(false);
    });
  }, [videoId, currentFrameId]);

  useEffect(() => {
    if (!stripRef.current) return;
    const active = stripRef.current.querySelector('.strip-thumb.active');
    if (active) {
      active.scrollIntoView({ behavior: 'smooth', block: 'nearest', inline: 'center' });
    }
  }, [activeIdx]);

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

  const activeFrame = frames[activeIdx];
  const imgSrc = activeFrame
    ? `/keyframes/${videoId}/${String(activeFrame.frame_id).padStart(3, '0')}.png`
    : '';

  return (
    <div className="slideshow-wrapper">
      <div className="ss-info-bar">
        <span className="ss-label">🎞 {videoId}</span>
        <span className="ss-counter">{activeIdx + 1} / {frames.length} frames</span>
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

        <div className="ss-main-img" onClick={() => setLightboxSrc(imgSrc)} title="Click to zoom">
          <img src={imgSrc} alt={`Frame ${activeFrame?.frame_id}`} />
          <div className="ss-zoom-hint">🔍 Zoom</div>
          <div className="ss-frame-badge">
            Frame {activeFrame?.frame_id}
            {activeFrame?.frame_id === parseInt(currentFrameId, 10) && (
              <span className="ss-current-tag"> · ★ Selected</span>
            )}
          </div>
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
          const thumbSrc = `/keyframes/${videoId}/${String(f.frame_id).padStart(3, '0')}.png`;
          const isCurrent = f.frame_id === parseInt(currentFrameId, 10);
          const isActive = idx === activeIdx;
          return (
            <button
              key={f.frame_id}
              type="button"
              className={`strip-thumb${isActive ? ' active' : ''}${isCurrent ? ' current' : ''}`}
              onClick={() => setActiveIdx(idx)}
              title={`Frame ${f.frame_id}${isCurrent ? ' (selected result)' : ''}`}
            >
              <img src={thumbSrc} alt={`Frame ${f.frame_id}`} />
              {isCurrent && <span className="strip-dot" />}
            </button>
          );
        })}
      </div>

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

export default function App() {
  const [query, setQuery] = useState('');
  const [extraQueries, setExtraQueries] = useState([]);
  const [modalities, setModalities] = useState(defaultModalities);
  const [objectPanelOpen, setObjectPanelOpen] = useState(true);
  const [objectQueries, setObjectQueries] = useState(() => [createObjectQueryRow()]);
  const [minConfidence, setMinConfidence] = useState(0.1);
  const [imageCue, setImageCue] = useState(null);
  const [results, setResults] = useState(null);
  const [selectedKeyframe, setSelectedKeyframe] = useState(null);
  const [usedQueries, setUsedQueries] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [theme, setTheme] = useState('dark');

  // UX States
  const [isDragging, setIsDragging] = useState(false);

  // Lightbox for quick card-level zoom
  const [lightboxFrame, setLightboxFrame] = useState(null);

  const queryList = useMemo(
    () =>
      [query, ...extraQueries.map((item) => item.text)]
        .map((item) => item.trim())
        .filter(Boolean),
    [query, extraQueries],
  );

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

  const displayedKeyframes = results ?? [];
  const hasRetrievalInput = queryList.length > 0 || objectPayload.length > 0 || Boolean(imageCue);

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

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
  }, [theme]);

  useEffect(() => {
    if (!hasRetrievalInput) {
      setError('');
      resetResultContext();
    }
  }, [hasRetrievalInput]);

  useEffect(() => {
    if (!usedQueries.length) return;
    const usedPrefixStillMatches = usedQueries.every(
      (item, index) => queryList[index] === item,
    );
    if (!usedPrefixStillMatches) {
      resetResultContext();
    }
  }, [queryList, usedQueries]);

  function getRequestTask() {
    return imageCue || modalities.includes('image') ? 'VKIS' : 'KIS';
  }

  function resetResultContext() {
    console.log('Resetting result context');
    setResults(null);
    setSelectedKeyframe(null);
    setUsedQueries((current) => (current.length ? [] : current));
  }

  function clearRetrievalInputs() {
    setQuery('');
    setExtraQueries([]);
    setObjectQueries([createObjectQueryRow()]);
    setObjectPanelOpen(true);
    if (imageCue?.preview) {
      URL.revokeObjectURL(imageCue.preview);
    }
    setImageCue(null);
    setModalities(defaultModalities);
    setError('');
    resetResultContext();
  }

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
      const endpoint = hasImageReference ? '/search/visual_query' : '/search/query';
      const currentFrameContext = keyframeContexts(displayedKeyframes);
      const framesPath = currentFrameContext.length
        ? currentFrameContext.map((item) => item.path)
        : null;
      const imageCuePayload = imageCue
        ? {
          name: imageCue.name,
          size: imageCue.size,
          type: imageCue.type,
          dataUrl: await readFileAsDataUrl(imageCue.file),
        }
        : null;
      const requestBody = hasImageReference
        ? {
          task: 'VKIS',
          minConfidence,
          imageCue: imageCuePayload,
        }
        : {
          query: query.trim(),
          queries: queryList,
          task: getRequestTask(),
          modalities,
          objects: objectPayload.map((item) => item.query),
          objectQueries: objectPayload,
          minConfidence,
          frames_path: framesPath,
          used_queries: framesPath ? usedQueries : [],
          frames_context: framesPath ? currentFrameContext : [],
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
      setUsedQueries(hasImageReference ? [] : queryList);
    } catch {
      setError('Could not reach the API. Start the backend on port 8000.');
      setResults([]);
      setSelectedKeyframe(null);
      setUsedQueries([]);
    } finally {
      setIsLoading(false);
    }
  }

  function toggleValue(value, selected, setter) {
    setter(
      selected.includes(value)
        ? selected.filter((item) => item !== value)
        : [...selected, value],
    );
  }

  function addExtraQuery() {
    setExtraQueries((current) => [...current, createExtraQueryRow()]);
  }

  function updateExtraQuery(id, value) {
    setExtraQueries((current) =>
      current.map((item) => (item.id === id ? { ...item, text: value } : item)),
    );
  }

  function removeExtraQuery(id) {
    setExtraQueries((current) => current.filter((item) => item.id !== id));
  }

  function addObjectQuery() {
    setObjectPanelOpen(true);
    setObjectQueries((current) => [...current, createObjectQueryRow()]);
  }

  function updateObjectQuery(id, field, value) {
    setObjectQueries((current) =>
      current.map((item) =>
        item.id === id ? { ...item, [field]: value } : item,
      ),
    );
  }

  function removeObjectQuery(id) {
    setObjectQueries((current) => {
      const next = current.filter((item) => item.id !== id);
      return next.length ? next : [createObjectQueryRow()];
    });
  }

  function handleObjectPanelToggle() {
    setObjectPanelOpen((open) => !open);
    setModalities((current) =>
      current.includes('objects') ? current : [...current, 'objects'],
    );
  }

  function handleModalityToggle(value) {
    const isActive = modalities.includes(value);
    if (value === 'objects') {
      setObjectPanelOpen(!isActive);
    }
    toggleValue(value, modalities, setModalities);
  }

  function processImageFile(file) {
    if (!file) return;
    if (imageCue?.preview) {
      URL.revokeObjectURL(imageCue.preview);
    }
    setImageCue({
      file,
      name: file.name,
      size: file.size,
      type: file.type,
      preview: URL.createObjectURL(file),
    });
    setModalities((current) =>
      current.includes('image') ? current : [...current, 'image'],
    );
  }

  function clearImageCue() {
    if (imageCue?.preview) {
      URL.revokeObjectURL(imageCue.preview);
    }
    setImageCue(null);
    setModalities((current) => current.filter((item) => item !== 'image'));
  }

  function handleImageCue(event) {
    processImageFile(event.target.files?.[0]);
  }

  function handleDragOver(e) {
    e.preventDefault();
    setIsDragging(true);
  }

  function handleDragLeave(e) {
    e.preventDefault();
    setIsDragging(false);
  }

  function handleDrop(e) {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file && file.type.startsWith('image/')) {
      processImageFile(file);
    }
  }

  function handleSelectKeyframe(keyframe) {
    setSelectedKeyframe(keyframe);
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
              : displayedKeyframes.map((keyframe) => (
                <button
                  key={keyframe.id}
                  type="button"
                  className={
                    selectedKeyframe?.id === keyframe.id
                      ? 'shot-card active'
                      : 'shot-card'
                  }
                  onClick={() => handleSelectKeyframe(keyframe)}
                >
                  <div className="shot-card-img-wrap">
                    <img src={keyframe.thumbnail || keyframeImagePath(keyframe)} alt={keyframe.title} />
                    <div
                      className="card-zoom-btn"
                      role="button"
                      tabIndex={0}
                      title="Zoom image"
                      onClick={(e) => {
                        e.stopPropagation();
                        setLightboxFrame({
                          src: keyframe.thumbnail || keyframeImagePath(keyframe),
                          alt: keyframe.title || `Frame ${keyframe.frame_id}`,
                        });
                      }}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter') {
                          e.stopPropagation();
                          setLightboxFrame({
                            src: keyframe.thumbnail || keyframeImagePath(keyframe),
                            alt: keyframe.title || `Frame ${keyframe.frame_id}`,
                          });
                        }
                      }}
                    >
                      🔍
                    </div>
                  </div>
                  <div className="shot-meta">
                    <span>Video: {keyframe.video_id}</span>
                    <strong><i>Frame:</i> {keyframe.frame_id}</strong>
                  </div>
                  <div className="score-pill" data-tooltip={keyframe.reasons?.join('\n') || ''}>
                    {Math.round(keyframe.score * 100)}%
                  </div>
                </button>
              ))}
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

      {/* ── Detail Panel (Slide-out) ─────────────────────────── */}
      {selectedKeyframe && (
        <>
          <div
            className="detail-overlay"
            onClick={() => setSelectedKeyframe(null)}
          />
          <aside className="detail-panel">
            <div className="detail-header">
              <h3>Keyframe Details</h3>
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
                  setLightboxFrame({ src, alt: selectedKeyframe.title || `Frame ${selectedKeyframe.frame_id}` });
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
                <p className="eyebrow">{selectedKeyframe.video_id || selectedKeyframe.videoId}</p>
                <h2>{`Frame ${selectedKeyframe.frame_id}`}</h2>
                {selectedKeyframe.description && <p>{selectedKeyframe.description}</p>}

                {/* Slideshow — always visible */}
                <FrameSlideshow
                  videoId={selectedKeyframe.video_id || selectedKeyframe.videoId}
                  currentFrameId={selectedKeyframe.frame_id || selectedKeyframe.frameId}
                  onClose={() => setSelectedKeyframe(null)}
                />

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
          </aside>
        </>
      )}
    </div>
  );
}
