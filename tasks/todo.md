# MP4-first BoldSearch — checklist triển khai

> Đây là task list sau khi chốt các câu hỏi trong `tasks/plan.md`; các mục đánh
> dấu phản ánh những slice implementation đã hoàn tất.

## Phase 0 — Contract và baseline

- [ ] Pin commit/revision/config vào corpus manifest.
- [ ] Chạy golden video và lưu frame IDs, schema, vector invariants, ảnh mẫu,
  latency/byte baseline.
- [x] Chốt profile `legacy-compatible` riêng, không đổi default profile.

**Acceptance:** rerun cùng input/config có cùng contract; sai khác fail rõ.

## Phase 1 — MP4 discovery và packaging

- [ ] Pin source `aic_video_pipeline_v1` trong BoldSearch clone.
- [x] Thêm profile L21/L23 và dry-run path validation.
- [x] Kiểm tra resume và không duplicate video ID.

**Acceptance:** chỉ MP4 hợp lệ trực tiếp trong `Videos_Lxx_*/video` được chạy.

## Phase 2 — Publisher artifact

- [x] Validate `Frame.json`, PNG, NPY trước publish.
- [x] Tạo Frames.csv và thumbnail WebP versioned.
- [x] Đảm bảo atomic active-manifest và recovery sau lỗi.

**Acceptance:** row/ảnh/vector counts khớp; không có release partial.

## Phase 3 — Milvus và backend search

- [x] Chốt/tự bootstrap projection schema versioned và primary key idempotent.
- [x] Batch upsert visual vectors và validation acknowledgement.
- [x] Thêm query modality contract config-driven, không fake caption vector.

**Acceptance:** query trả hit từ video MP4 mới và hit nào cũng có ảnh 200.

## Checkpoint — Data path

- [ ] Golden/output tests pass.
- [x] Schema validation pass; retry ledger có thể resume.
- [ ] Search-to-image E2E pass local.

## Phase 4 — UI, gateway và Cloudflare

- [x] Same-origin gateway `/api`, `/Frames.csv`, `/keyframes`.
- [x] WebP thumbnails + lazy/async decoding cho grid/detail.
- [ ] One gateway health lifecycle và tunnel smoke tests trên Kaggle.
- [ ] Restrict CORS, rotate secrets, add public request limits.

**Acceptance:** public URL hiển thị UI, search và ảnh đúng; không tải hàng loạt
PNG full-resolution.

## Phase 5 — Performance/rollout

- [ ] Thu baseline và before/after metrics cho mỗi tối ưu.
- [ ] Đạt budgets trong `tasks/plan.md` hoặc điều chỉnh theo số đo.
- [x] Active manifest rollback; canary runbook vẫn cần số đo thật.

**Acceptance:** performance report, rollback rehearsal, không còn secret trong
source/history hiện hành.
