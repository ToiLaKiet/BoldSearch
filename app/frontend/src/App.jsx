import { useEffect, useMemo, useState } from 'react';

const API_BASE = import.meta.env.VITE_API_BASE || '/api';

const objectOptions = [
  'person',
  'bicycle',
  'boat',
  'goalkeeper',
  'microscope',
  'train',
  'bowl',
  'microphone',
];

const colorOptions = [
  { label: 'yellow', value: '#ffd84d' },
  { label: 'red', value: '#df3b3b' },
  { label: 'green', value: '#4f9f62' },
  { label: 'blue', value: '#3b82f6' },
  { label: 'white', value: '#f8fafc' },
  { label: 'black', value: '#111827' },
  { label: 'turquoise', value: '#2dd4bf' },
  { label: 'violet', value: '#8b5cf6' },
];

const defaultModalities = ['text', 'objects', 'temporal'];

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

/* ── Main App ────────────────────────────────────────────────── */

export default function App() {
  const [tasks, setTasks] = useState([]);
  const [activeTask, setActiveTask] = useState('KIS');
  const [query, setQuery] = useState('yellow raincoat bicycle at a city crossing');
  const [temporal, setTemporal] = useState({ prefix: 'after', text: 'rain starts' });
  const [modalities, setModalities] = useState(defaultModalities);
  const [selectedObjects, setSelectedObjects] = useState(['bicycle', 'person']);
  const [selectedColors, setSelectedColors] = useState(['yellow']);
  const [minConfidence, setMinConfidence] = useState(0.1);
  const [imageCue, setImageCue] = useState(null);
  const [results, setResults] = useState([]);
  const [selectedShot, setSelectedShot] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [submission, setSubmission] = useState(null);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [theme, setTheme] = useState('dark');
  
  // UX States
  const [isDragging, setIsDragging] = useState(false);
  const [objectSearch, setObjectSearch] = useState('');
  const [colorSearch, setColorSearch] = useState('');

  // Derived options for tags
  const filteredObjects = useMemo(() => 
    objectOptions.filter(o => o.includes(objectSearch.toLowerCase()) && !selectedObjects.includes(o)),
    [objectSearch, selectedObjects]
  );
  
  const filteredColors = useMemo(() => 
    colorOptions.filter(c => c.label.includes(colorSearch.toLowerCase()) && !selectedColors.includes(c.label)),
    [colorSearch, selectedColors]
  );

  const popularObjects = ['person', 'bicycle', 'car', 'tree'].filter(o => objectOptions.includes(o));

  useEffect(() => {
    function handleKeyDown(e) {
      if (e.key === 'Escape') setSelectedShot(null);
    }
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
  }, [theme]);

  useEffect(() => {
    async function loadTasks() {
      try {
        const response = await fetch(`${API_BASE}/search/tasks`);
        const data = await response.json();
        setTasks(data.tasks || []);
      } catch {
        setTasks([
          { id: 'KIS', name: 'Known Item Search' },
          { id: 'VKIS', name: 'Visual Known Item Search' },
        ]);
      }
    }

    loadTasks();
  }, []);

  useEffect(() => {
    runSearch();
  }, []);

  const activeTaskMeta = useMemo(
    () => tasks.find((task) => task.id === activeTask),
    [tasks, activeTask],
  );

  async function runSearch(event) {
    event?.preventDefault();
    setIsLoading(true);
    setError('');
    setSubmission(null);

    try {
      const response = await fetch(`${API_BASE}/search/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query,
          task: activeTask,
          modalities,
          objects: selectedObjects,
          colors: selectedColors,
          temporal: temporal.text.trim() ? `${temporal.prefix} ${temporal.text.trim()}`.trim() : '',
          minConfidence,
          imageCue: imageCue
            ? { name: imageCue.name, size: imageCue.size, type: imageCue.type }
            : null,
        }),
      });

      if (!response.ok) {
        throw new Error('Search API returned an error');
      }

      const data = await response.json();
      setResults(data.results || []);
    } catch {
      setError('Could not reach the API. Start the backend on port 5001.');
      setResults([]);
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

  function handleTaskChange(taskId) {
    setActiveTask(taskId);
    if (taskId === 'VKIS') {
      setModalities(['image', 'objects', 'temporal']);
      setSelectedColors((current) => (current.length ? current : ['red']));
    } else {
      setModalities(['text', 'objects', 'temporal']);
    }
  }

  function processImageFile(file) {
    if (!file) return;
    setImageCue({
      file,
      name: file.name,
      size: file.size,
      type: file.type,
      preview: URL.createObjectURL(file),
    });
    if (!modalities.includes('image')) {
      setModalities([...modalities, 'image']);
    }
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

  async function submitShot() {
    if (!selectedShot) return;

    try {
      const response = await fetch(`${API_BASE}/search/submit`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ shotId: selectedShot.id, task: activeTask }),
      });
      const data = await response.json();
      setSubmission(data);
    } catch {
      setSubmission({ status: 'error', message: 'Submission API unavailable.' });
    }
  }

  const activeBadges = [
    ...modalities.slice(0, 3),
    ...selectedObjects.slice(0, 2),
    ...selectedColors.slice(0, 1),
  ];

  return (
    <div className="app-shell">
      {/* ── Sidebar ──────────────────────────────────────────── */}
      <aside className={`sidebar ${sidebarOpen ? '' : 'collapsed'}`}>
        <div className="sidebar-header">
          <h1>BoldSearcher</h1>
          <p className="subtitle">Interactive shot retrieval</p>
        </div>

        <form className="sidebar-body" onSubmit={runSearch}>
          {/* Section 1: Query & Task */}
          <Accordion title="Query & Task" defaultOpen>
            <div style={{ marginBottom: 12 }}>
              <strong style={{ fontSize: '0.88rem' }}>
                {activeTaskMeta?.name || activeTask}
              </strong>
              <p
                style={{
                  fontSize: '0.78rem',
                  color: 'var(--text-tertiary)',
                  marginTop: 4,
                  lineHeight: 1.4,
                }}
              >
                {activeTaskMeta?.description ||
                  'Choose a retrieval task before searching.'}
              </p>
            </div>
            <div className="query-wrapper" style={{ position: 'relative' }}>
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
                placeholder="Describe the target shot..."
              />
              <div className="query-actions">
                {query && (
                  <button type="button" className="btn-icon" onClick={() => setQuery('')} title="Clear">✕</button>
                )}
                <button type="button" className="btn-icon" onClick={() => alert('ASR module is ready for integration')} title="Voice Search">🎤</button>
              </div>
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
                    <span>Drag & drop image here</span>
                    <small>or click to browse</small>
                  </label>
                ) : (
                  <div className="image-cue-preview">
                    <img src={imageCue.preview} alt="Uploaded visual cue" />
                    <div className="image-info">
                      <span>{imageCue.name}</span>
                      <button type="button" className="btn-remove-img" onClick={() => setImageCue(null)}>✕</button>
                    </div>
                  </div>
                )}
              </div>
            </div>

            <div className="filter-group" style={{ marginTop: 16 }}>
              <label className="field-label">Objects</label>
              <div className="searchable-tags">
                <div className="tags-input-container">
                  {selectedObjects.map((obj) => (
                    <span key={obj} className="tag-chip">
                      {obj}
                      <button type="button" onClick={() => toggleValue(obj, selectedObjects, setSelectedObjects)}>✕</button>
                    </span>
                  ))}
                  <input
                    type="text"
                    value={objectSearch}
                    onChange={(e) => setObjectSearch(e.target.value)}
                    placeholder="Search objects..."
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' && objectSearch && filteredObjects.length > 0) {
                        e.preventDefault();
                        toggleValue(filteredObjects[0], selectedObjects, setSelectedObjects);
                        setObjectSearch('');
                      }
                    }}
                  />
                </div>
                {objectSearch && filteredObjects.length > 0 && (
                  <div className="dropdown-menu">
                    {filteredObjects.slice(0, 5).map(obj => (
                      <button key={obj} type="button" onClick={() => { toggleValue(obj, selectedObjects, setSelectedObjects); setObjectSearch(''); }}>
                        {obj}
                      </button>
                    ))}
                  </div>
                )}
              </div>
              <div className="popular-tags">
                <span className="muted-text">Popular:</span>
                {popularObjects.map(obj => (
                  <button key={obj} type="button" className="quick-chip" onClick={() => !selectedObjects.includes(obj) && toggleValue(obj, selectedObjects, setSelectedObjects)}>
                    {obj}
                  </button>
                ))}
              </div>
            </div>

            <div className="filter-group" style={{ marginTop: 16 }}>
              <label className="field-label">Colors</label>
              <div className="swatch-row">
                {colorOptions.map((color) => (
                  <button
                    key={color.label}
                    type="button"
                    className={
                      selectedColors.includes(color.label)
                        ? 'swatch active'
                        : 'swatch'
                    }
                    style={{ '--swatch': color.value }}
                    onClick={() =>
                      toggleValue(color.label, selectedColors, setSelectedColors)
                    }
                    aria-label={color.label}
                    title={color.label}
                  />
                ))}
              </div>
            </div>
          </Accordion>

          {/* Section 3: Advanced Filters */}
          <Accordion title="Advanced Filters">
            <div className="filter-group">
              <label className="field-label">Temporal Cue</label>
              <div style={{ display: 'flex', gap: '8px' }}>
                <select
                  value={temporal.prefix}
                  onChange={(e) => setTemporal({ ...temporal, prefix: e.target.value })}
                  style={{ width: '100px' }}
                >
                  <option value="before">Before</option>
                  <option value="after">After</option>
                  <option value="during">During</option>
                  <option value="then">Then</option>
                  <option value="">Custom...</option>
                </select>
                <input
                  id="temporal"
                  type="text"
                  value={temporal.text}
                  onChange={(e) => setTemporal({ ...temporal, text: e.target.value })}
                  placeholder="e.g. rain starts"
                  style={{ flex: 1 }}
                />
              </div>
            </div>

            <div className="filter-group" style={{ marginTop: 16 }}>
              <label className="field-label">Search Modalities</label>
              <div className="chip-row">
                {['text', 'image', 'objects', 'temporal'].map((value) => (
                  <button
                    key={value}
                    type="button"
                    className={modalities.includes(value) ? 'chip active' : 'chip'}
                    onClick={() => toggleValue(value, modalities, setModalities)}
                  >
                    {value}
                  </button>
                ))}
              </div>
            </div>

            <div className="filter-group" style={{ marginTop: 16 }}>
              <label className="field-label">
                Min Score: {minConfidence.toFixed(2)}
              </label>
              <input
                id="confidence"
                type="range"
                min="0"
                max="0.9"
                step="0.05"
                value={minConfidence}
                onChange={(e) => setMinConfidence(Number(e.target.value))}
              />
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
            {isLoading ? 'Searching…' : 'Search shots'}
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

          <div className="topbar-center">
            {['KIS', 'VKIS'].map((taskId) => (
              <button
                key={taskId}
                type="button"
                className={
                  activeTask === taskId ? 'task-tab active' : 'task-tab'
                }
                onClick={() => handleTaskChange(taskId)}
              >
                {taskId}
              </button>
            ))}
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
              {results.length} result{results.length !== 1 ? 's' : ''}
            </span>
            <div className="active-filters">
              {activeBadges.map((item) => (
                <span key={item} className="filter-badge">
                  {item}
                </span>
              ))}
            </div>
          </div>
        </header>

        {/* Results */}
        <div className="results-wrapper">
          {error && <div className="notice error">{error}</div>}

          {!isLoading && results.length === 0 && !error && (
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
              : results.map((shot) => (
                  <button
                    key={shot.id}
                    type="button"
                    className={
                      selectedShot?.id === shot.id
                        ? 'shot-card active'
                        : 'shot-card'
                    }
                    onClick={() => setSelectedShot(shot)}
                  >
                    <img src={shot.thumbnail} alt={shot.title} />
                    <div className="shot-meta">
                      <span>{shot.videoId}</span>
                      <strong>{shot.title}</strong>
                      <small>
                        {shot.start} – {shot.end}
                      </small>
                    </div>
                    <div className="score-pill" data-tooltip={shot.reasons?.join('\n') || ''}>
                      {Math.round(shot.score * 100)}%
                    </div>
                  </button>
                ))}
          </div>
        </div>
      </div>

      {/* ── Detail Panel (Slide-out) ─────────────────────────── */}
      {selectedShot && (
        <>
          <div
            className="detail-overlay"
            onClick={() => setSelectedShot(null)}
          />
          <aside className="detail-panel">
            <div className="detail-header">
              <h3>Shot Details</h3>
              <button
                className="btn-close"
                type="button"
                onClick={() => setSelectedShot(null)}
              >
                ✕
              </button>
            </div>

            <div className="detail-scroll">
              <div className="detail-preview">
                <img
                  src={selectedShot.thumbnail}
                  alt={selectedShot.title}
                />
                <div className="play-overlay">▶ Preview</div>
              </div>

              <div className="detail-body">
                <p className="eyebrow">{selectedShot.videoId}</p>
                <h2>{selectedShot.title}</h2>
                <p>{selectedShot.description}</p>

                <div className="timebar">
                  <span>{selectedShot.start}</span>
                  <div>
                    <i
                      style={{
                        width: `${Math.min(selectedShot.duration * 7, 100)}%`,
                      }}
                    />
                  </div>
                  <span>{selectedShot.end}</span>
                </div>

                <div className="metadata-list">
                  <div>
                    <span>Objects</span>
                    <strong>{selectedShot.objects.join(', ')}</strong>
                  </div>
                  <div>
                    <span>Colors</span>
                    <strong>{selectedShot.colors.join(', ')}</strong>
                  </div>
                  <div>
                    <span>Transcript</span>
                    <strong>{selectedShot.transcript}</strong>
                  </div>
                </div>

                <div className="reason-list">
                  {selectedShot.reasons.map((reason) => (
                    <span key={reason}>{reason}</span>
                  ))}
                </div>

                {submission && (
                  <div
                    className={`notice ${
                      submission.status === 'accepted' ? 'success' : 'error'
                    }`}
                  >
                    {submission.status === 'accepted'
                      ? `Accepted: ${submission.submission.videoId} at ${submission.submission.timestamp}`
                      : submission.message}
                  </div>
                )}
              </div>
            </div>

            <div className="detail-footer">
              <button
                className="btn-primary"
                type="button"
                onClick={submitShot}
              >
                Submit this shot
              </button>
            </div>
          </aside>
        </>
      )}
    </div>
  );
}
