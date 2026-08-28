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

export default function App() {
  const [tasks, setTasks] = useState([]);
  const [activeTask, setActiveTask] = useState('KIS');
  const [query, setQuery] = useState('yellow raincoat bicycle at a city crossing');
  const [temporal, setTemporal] = useState('after rain starts');
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

  useEffect(() => {
    async function loadTasks() {
      try {
        const response = await fetch(`${API_BASE}/tasks`);
        const data = await response.json();
        setTasks(data.tasks || []);
      } catch {
        setTasks([
          { id: 'KIS', name: 'Known Item Search' },
          { id: 'TRAKE', name: 'Temporal Action Knowledge Extraction'},
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
      const response = await fetch(`${API_BASE}/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query,
          task: activeTask,
          modalities,
          objects: selectedObjects,
          colors: selectedColors,
          temporal,
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
      setSelectedShot(data.results?.[0] || null);
    } catch (searchError) {
      setError('Could not reach the Flask API. Start the backend on port 5001.');
      setResults([]);
      setSelectedShot(null);
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

  function handleImageCue(event) {
    const file = event.target.files?.[0];
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

  async function submitShot() {
    if (!selectedShot) return;

    try {
      const response = await fetch(`${API_BASE}/submit`, {
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

  return (
    <div className="app-shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">Interactive shot retrieval</p>
          <h1>BoldSearcher</h1>
        </div>
        <div className="task-tabs" aria-label="Task type">
          {['KIS', 'VKIS'].map((taskId) => (
            <button
              key={taskId}
              className={activeTask === taskId ? 'active' : ''}
              type="button"
              onClick={() => handleTaskChange(taskId)}
            >
              {taskId}
            </button>
          ))}
        </div>
      </header>

      <main className="workspace">
        <aside className="query-panel">
          <form onSubmit={runSearch}>
            <section className="panel-section">
              <div className="section-heading">
                <span>Task</span>
                <strong>{activeTaskMeta?.name || activeTask}</strong>
              </div>
              <p className="muted">
                {activeTaskMeta?.description ||
                  'Choose a retrieval task before searching the shot index.'}
              </p>
            </section>

            <section className="panel-section">
              <label htmlFor="query">Query</label>
              <textarea
                id="query"
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                placeholder="Describe the target shot..."
              />
            </section>

            <section className="panel-section">
              <label>Modalities</label>
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
            </section>

            <section className="panel-section">
              <label htmlFor="image-cue">Image cue</label>
              <input id="image-cue" type="file" accept="image/*" onChange={handleImageCue} />
              {imageCue && (
                <div className="image-cue">
                  <img src={imageCue.preview} alt="Uploaded visual cue" />
                  <span>{imageCue.name}</span>
                </div>
              )}
            </section>

            <section className="panel-section">
              <label>Objects</label>
              <div className="chip-row">
                {objectOptions.map((value) => (
                  <button
                    key={value}
                    type="button"
                    className={selectedObjects.includes(value) ? 'chip active' : 'chip'}
                    onClick={() => toggleValue(value, selectedObjects, setSelectedObjects)}
                  >
                    {value}
                  </button>
                ))}
              </div>
            </section>

            <section className="panel-section">
              <label>Colors</label>
              <div className="swatch-row">
                {colorOptions.map((color) => (
                  <button
                    key={color.label}
                    type="button"
                    className={selectedColors.includes(color.label) ? 'swatch active' : 'swatch'}
                    style={{ '--swatch': color.value }}
                    onClick={() => toggleValue(color.label, selectedColors, setSelectedColors)}
                    aria-label={color.label}
                    title={color.label}
                  />
                ))}
              </div>
            </section>

            <section className="panel-section">
              <label htmlFor="temporal">Temporal cue</label>
              <input
                id="temporal"
                value={temporal}
                onChange={(event) => setTemporal(event.target.value)}
                placeholder="first this, then that"
              />
            </section>

            <section className="panel-section">
              <label htmlFor="confidence">Minimum score: {minConfidence.toFixed(2)}</label>
              <input
                id="confidence"
                type="range"
                min="0"
                max="0.9"
                step="0.05"
                value={minConfidence}
                onChange={(event) => setMinConfidence(Number(event.target.value))}
              />
            </section>

            <button className="primary-action" type="submit" disabled={isLoading}>
              {isLoading ? 'Searching...' : 'Search shots'}
            </button>
          </form>
        </aside>

        <section className="results-area">
          <div className="results-toolbar">
            <div>
              <p className="eyebrow">{activeTask} results</p>
              <h2>{results.length} candidate shots</h2>
            </div>
            <div className="toolbar-tags">
              {[...modalities, ...selectedObjects.slice(0, 2), ...selectedColors].map((item) => (
                <span key={item}>{item}</span>
              ))}
            </div>
          </div>

          {error && <div className="notice error">{error}</div>}

          <div className="result-grid" aria-live="polite">
            {isLoading
              ? Array.from({ length: 6 }).map((_, index) => (
                  <div className="shot-card skeleton" key={index} />
                ))
              : results.map((shot) => (
                  <button
                    key={shot.id}
                    type="button"
                    className={selectedShot?.id === shot.id ? 'shot-card active' : 'shot-card'}
                    onClick={() => setSelectedShot(shot)}
                  >
                    <img
                      loading="lazy"
                      decoding="async"
                      src={shot.thumbnail}
                      alt={shot.title}
                    />
                    <div className="shot-meta">
                      <span>{shot.videoId}</span>
                      <strong>{shot.title}</strong>
                      <small>
                        {shot.start} - {shot.end}
                      </small>
                    </div>
                    <div className="score-pill">{Math.round(shot.score * 100)}%</div>
                  </button>
                ))}
          </div>
        </section>

        <aside className="detail-panel">
          {selectedShot ? (
            <>
              <div className="detail-preview">
                <img
                  loading="lazy"
                  decoding="async"
                  src={selectedShot.thumbnail}
                  alt={selectedShot.title}
                />
                <div className="play-overlay">Preview</div>
              </div>
              <div className="detail-body">
                <p className="eyebrow">{selectedShot.videoId}</p>
                <h2>{selectedShot.title}</h2>
                <p>{selectedShot.description}</p>

                <div className="timebar">
                  <span>{selectedShot.start}</span>
                  <div>
                    <i style={{ width: `${Math.min(selectedShot.duration * 7, 100)}%` }} />
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

                <button className="primary-action" type="button" onClick={submitShot}>
                  Submit selected shot
                </button>

                {submission && (
                  <div className={`notice ${submission.status === 'accepted' ? 'success' : 'error'}`}>
                    {submission.status === 'accepted'
                      ? `Accepted: ${submission.submission.videoId} at ${submission.submission.timestamp}`
                      : submission.message}
                  </div>
                )}
              </div>
            </>
          ) : (
            <div className="empty-state">
              <h2>No shot selected</h2>
              <p>Run a search and choose a candidate shot to inspect metadata and submit.</p>
            </div>
          )}
        </aside>
      </main>
    </div>
  );
}
