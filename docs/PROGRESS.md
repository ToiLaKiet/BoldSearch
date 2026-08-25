### [2026-08-21 18:38 UTC+07:00] — [Docs] Correct BTC package submission protocol

**Done:**

- Verified the official portal requires a ZIP containing a top-level `submission/` directory and per-query CSV files; it does not accept individual BoldSearch answers directly.
- Corrected the submission guide with CSV schemas for KIS, Q&A, and TRAKE plus the likely causes of the portal's HTTP 400 rejection.

**Changed files:**

- `docs/SUBMISSION_GUIDE.md` — official CSV/ZIP package format and validation checklist.

**Flow explained:**

BoldSearch candidate -> per-query UTF-8 CSV -> `submission/` directory -> ZIP upload to BTC portal.

### [2026-08-21 18:18 UTC+07:00] — [Docs] Add competition submission guide

**Done:**

- Added a Vietnamese operational guide for all 24 supplied questions, grouped into KIS, Q&A, and TRAKE workflows.
- Documented the local-only submit boundary and the required manual transfer of `VIDEO / Frame` to the BTC portal.

**Changed files:**

- `docs/SUBMISSION_GUIDE.md` — query strategy, answer capture, submission checklist, and common recovery steps.

**Flow explained:**

Question mode -> retrieval and frame verification -> local answer confirmation/copy -> BTC portal submission.

### [2026-08-21 18:13 UTC+07:00] — [Frontend] Repair slideshow keyframes and clarify local submission

**Done:**

- Made the slideshow main image and filmstrip resolve each source frame through the official per-video keyframe map, matching the search-result cards.
- Preserved the successful KIS submit state and show the selected `video_id`/`frame_id` as a local answer that can be copied for the BTC portal.
- Browser-verified mapped modal images and a successful submit/copy flow.

**Changed files:**

- `app/frontend/src/App.jsx` — keyframe map lookup, mapped slideshow image paths, and persistent KIS submit confirmation.
- `app/frontend/src/styles.css` — styles the local submission confirmation and copy action.

**Flow explained:**

`Frames.csv frame_id` -> `map-keyframes/<video_id>.csv` nearest `frame_idx` -> `/keyframes/<video_id>/<n>.jpg`; KIS submit -> local accepted response -> copyable BTC answer identifier.

### [2026-08-21 13:16 UTC+07:00] — [Search] Serve official keyframe thumbnails

**Done:**

- Downloaded and extracted every official AIC2026 keyframe archive (L21-L30, including all five L26 partitions) plus all frame mapping CSVs into the Vite public directory.
- Added nearest-frame mapping with three-digit keyframe filenames so text-query results return the matching `/keyframes/...jpg` URL.
- Verified asset coverage: all 873 mapped videos have a non-empty keyframe directory (177,321 JPG files).
- Verified in the browser: the `football` query returns 50 results with 50 loaded images and 0 broken images.

**Changed files:**

- `app/backend/app_config.py` — configured thumbnail URL and official mapping location defaults.
- `app/backend/search/service.py` — resolve a returned source frame to its nearest official keyframe image.
- `app/backend/tests/test_thumbnail_mapping.py` — covers L22 nearest-keyframe resolution.
- `app/backend/.env` — enables the thumbnail URL and mapping path locally.

**Flow explained:**

`frame_id` from Milvus -> `map-keyframes/<video_id>.csv` nearest `frame_idx` -> zero-padded `/keyframes/<video_id>/<n>.jpg` served by Vite.
