### [2026-08-21 18:13 UTC+07:00] — [Frontend build] Full static build exceeds local disk capacity

**Status:** Open

**Blocked by:** Vite copies `public/keyframes` into `dist/` during `npm run build`; retaining both the 29 GB source assets and build output exhausts the current disk.

**Impact:** The development server and browser verification work, but a complete local production build cannot finish until disk is freed or keyframes are served outside Vite's build output.

**Next action:** Free enough local storage for a second keyframe copy, or configure deployment/static serving so the downloaded keyframes are not duplicated during the frontend build.

### [2026-08-21 13:32 UTC+07:00] — [Search thumbnails] Some AIC2026 keyframe groups remain uninstalled

**Status:** Resolved

**Blocked by:** The initial local install included only L21, L22, L24, and the L26 partition containing `L26_V145`; the `football` browser query also returned `L23_V024`.

**Impact:** That L23 card had no local image while the other results loaded correctly.

**Next action:** Resolved by downloading and extracting all L21-L30 archives. Asset coverage is 873/873 mapped video directories and the browser `football` query renders 50/50 images without breakage.

### [2026-08-21 13:18 UTC+07:00] — [Search thumbnails] Running backend has stale configuration

**Status:** Resolved

**Blocked by:** The current Uvicorn process started without reload before thumbnail mapping configuration was added.

**Impact:** Browser verification returns 50 results but all 50 image elements load the old `/keyframes/<video_id>/<frame_id>.png` fallback and fail.

**Next action:** Resolved by restarting the backend after applying the mapping configuration. Browser verification of `cat` returned 50 loaded images and 0 broken images.

### [2026-08-21 12:32 UTC+07:00] — [Search query] Milvus output schema mismatch

**Status:** Resolved

**Blocked by:** The configured `MILVUS_OUTPUT_FIELDS` requests `asr_text`, but the active `BoldSearch` Milvus collection does not contain that field.

**Impact:** `POST /api/search/query` returns HTTP 500; the frontend can load, but no text-query result is available.

**Next action:** Verified resolved after the active backend configuration was updated: `POST /api/search/query` returns results again.
