# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

- **Platform**: Windows 11 + PowerShell (use PowerShell syntax — `$null`, `$env:VAR`, backtick line-continuation).
- **Python**: lives in `venv\` (CUDA 11.8 build of torch + paddlepaddle-gpu, both already GPU-enabled).
- **Crucially**: for one-shot tool calls, invoke `venv\Scripts\python.exe` directly. `python` on PATH is the system Python which lacks Flask/ultralytics/paddleocr. `start.bat` activates the venv for interactive shells; `tools/sim_laptop_*.bat` auto-activate it themselves.
- **Model filenames contain spaces** (e.g. `plate_best (1).pt`). Always quote paths. `models/` now holds only the two weights the code actually loads: `models/yolov8n.pt` (vehicles) + `models/plate_best (1).pt` (plates) — the defaults in `src/main.py` (`--vehicle-model` / `--plate-model`) and `evaluate.py`. Older, unreferenced plate weights (`license_plate_detector.pt`, `indo_plate.pt`, `best (8).pt`) were moved to `unused/models/` (Juli 2026); pass one via `--plate-model` if you ever want to compare. `unused/` at the repo root is the general parking lot for unreferenced files (also holds `vid1.mp4` and June-2026 diagnostic artifacts) — this project is NOT a git repo, so park things there instead of hard-deleting.

## Tech stack (what the main program actually imports)

The main program (`src/`) directly uses, per stage:

- **Detection**: `ultralytics` (YOLOv8) over `torch` (cu118) — vehicles + plates.
- **OCR**: `paddleocr` / `paddlepaddle-gpu` — plate text.
- **Tracking**: `scipy.optimize.linear_sum_assignment` (Hungarian) in `src/vehicle_tracker.py` — the one sanctioned `scipy` import under `src/` (it degrades to a greedy fallback over the same cost matrix if scipy is missing, but scipy is pinned in `requirements.txt`).
- **Image processing**: `cv2` (OpenCV) + `numpy` ONLY for the pipeline. Camera I/O, crops, HSV fuel check, deskew/CLAHE/unsharp, JPEG encode — all OpenCV+NumPy. `scikit-image` appears in `requirements.txt` but is a transitive dep of PaddleOCR/ultralytics, NOT imported under `src/`. `Pillow` is transitive for the pipeline too, but the desktop GUI does import it (`PIL.Image` / `PIL.ImageTk` in `src/gui.py`) to blit frames into Tkinter — that is the ONLY sanctioned `PIL` use. For pipeline image work the canonical tools remain cv2 + numpy.
- **Web / IO**: `Flask` (MJPEG preview server), `requests` (HTTP POST to web monitoring).
- **Desktop GUI**: `tkinter` / `tkinter.ttk` + `PIL.ImageTk` — `src/gui.py` only. The CLI (`src/main.py`) uses `cv2.imshow` and never imports tkinter.

## Pipeline architecture (frame → web push)

The per-frame processing is a strict 5-stage pipeline. It lives in `DetectionPipeline` (`src/pipeline.py`), and **both entry points reuse it** — `src/main.py` (CLI) and `src/gui.py` (Tkinter desktop). Don't duplicate detection logic into either front-end; extend `DetectionPipeline`.

```
frame ─► VehicleDetector ─► FuelClassifier ─► PlateReader (OCR) ─► VehicleTracker ─► HttpUploader
        (2 YOLO models)    (HSV blue strip)   (PaddleOCR)         (1-shot dedup)    (POST)
```

Two side sinks tap the loop without affecting the push path: `preview_server` (MJPEG live stream, on by default) gets each annotated frame, and `DetectionLogger` (`src/logger.py`, opt-in via `--log-jsonl`) writes one JSONL row per detection per frame. Each pipeline stage is its own module under `src/` (`detector.py`, `fuel_classifier.py`, `plate_reader.py`, `vehicle_tracker.py`, `uploader.py`) and is assembled in `src/pipeline.py` (`DetectionPipeline`, called by both `main.py` and `gui.py`) — the stage classes are NOT all defined in `detector.py`.

### Pipeline reuse & the GUI front-end

`DetectionPipeline` (`src/pipeline.py`) is the single source of per-frame logic; the two front-ends are thin shells around it. Invariants to preserve:

- **`process_frame(frame, frame_idx, log_frame=True, draw_overlay=True) -> (annotated, detections)`** is the one entry point. It runs detect → `tracker.update` (which auto-queues pushes via the `on_push` callback) → optional JSONL log → draw bbox + count. **FPS is NOT drawn here** — the caller draws it, deliberately, so CLI timing stays byte-identical to the pre-refactor loop. `draw_overlay=False` hides boxes/labels but detection, tracking, push, and logging still run (the GUI "Tampilkan Hasil Deteksi" toggle).
- **The pipeline owns no camera.** `cap` open/read/release, pause/step/loop stay in the caller — moving them into the pipeline would break CLI stepping. `process_frame` takes one frame.
- **Push is gated at the pipeline, not the tracker.** `push_enabled` (bool) and `connect_web(url)` let the GUI attach an uploader and flip sending at runtime; the tracker still writes local captures regardless (parity with the old CLI). Don't reintroduce push gating inside `VehicleTracker`.
- **`default_config()` in `src/main.py`** (= `build_parser().parse_args([])`) is how the GUI gets CLI defaults without duplicating the flag list. Add new pipeline knobs to `build_parser()` so both front-ends inherit them.
- **`src/gui.py` threading model** (Tkinter is NOT thread-safe): a background worker thread reads frames and calls `process_frame`, storing the annotated frame in `self._latest` under a lock and touching NO widgets; the main thread's `_refresh()` (every ~30ms via `root.after`) reads `_latest`, converts BGR→RGB→`PIL.ImageTk`, and blits to a `Label`. Keep all widget access on the main thread. `_imgtk` MUST stay referenced or it gets GC'd and the preview goes blank.

Key cross-file invariants — read these before changing any stage:

1. **Two detection modes, dual-model is the default** (`src/detector.py`). `VehicleDetector` supports both, selected at construction:
   - **Dual-model (default)** — `_detect_dual()`. Two `YOLO(...)` instances: `yolov8n.pt` for vehicles (COCO classes 2/3/5/7 translated via `COCO_VEHICLE_MAP` — bus and truck both merge to `mobil`) and the plate model (default `plate_best (1).pt`) for plates. `predict()` runs **twice** per frame.
   - **Unified (opt-in via `--model <path>`)** — `_detect_unified()`. One 3-class model (`mobil`/`motor`/`plat`) running **once** per frame; class names are mapped to local names (anything starting with `plat` → `plat`). This path is fully implemented and first-class — it's no longer hypothetical. It is **off by default** (`--model` default is `''`); passing a unified weight flips `self.unified = True`.

   Either way `detect()` returns one unified list and downstream stages don't care which mode produced it. Keep dual-model as the default and don't delete the dual path; switching the default to unified needs explicit instruction (and a shipped 3-class weight).

2. **`VIRTUAL_CLASS_ID`** in `detector.py` (`mobil=0, motor=1, plat=2`) is the contract downstream consumers (`utils.draw_detections`, `vehicle_tracker`, `evaluate.py`) rely on. Don't change the integer mapping.

3. **Plate detection feeds two stages** (orchestrated in `src/main.py`):
   - `FuelClassifier.classify()` (`src/fuel_classifier.py`; HSV blue ratio on bottom 40% of plate crop) — sets `det['fuel_type']` = `'bensin' | 'listrik'`. This is what determines `is_electric` in the web payload. Without plate detection, fuel can't be inferred.
   - `PlateReader.read()` (`src/plate_reader.py`; PaddleOCR) — sets `det['text']`. Cached per-plate via IoU match across frames, re-run only every `ocr_interval` (default 10) frames to keep FPS up.

4. **OCR preprocessing pipeline** (`src/plate_reader.py`) is non-trivial: deskew (minAreaRect on Otsu mask) → CLAHE → upscale to ≥200px → unsharp mask, **then** crop bottom 22% of plate (`crop_bottom_ratio`) before sending to PaddleOCR. The crop step drops the Indonesian masa-berlaku-pajak strip ("07-23") that would otherwise concat into the plate text and break the format regex. Post-OCR, multi-line output is clustered by y-center and the cluster with highest `avg_height × total_length` score wins (plate text > tax date).

5. **`VehicleTracker`** (`src/vehicle_tracker.py`) is the anti-spam layer. Track lifecycle: spawn on new vehicle bbox → match incoming frames → push **exactly once** when `frames_seen >= min_frames_stable` AND `min_vehicle_height` reached AND fuel determined → mark `pushed=True` → cleanup after `track_timeout` frames of no-sighting. The "best frame" is kept for the push image, resized to `max_image_dim=800` and JPEG-encoded at quality 85.

   Three things here exist to stop one vehicle's data landing on another vehicle's row (an ID switch means the **photo** on the web is the wrong car — the failure is silent and unrecoverable):
   - **Matching is a global assignment, not greedy IoU** (`_match` → `scipy.optimize.linear_sum_assignment`) over a combined cost of IoU + normalized centre distance + size/aspect similarity (`match_weights`, default 0.6/0.25/0.15), with cross-class matches forbidden outright. Greedy per-detection IoU swapped IDs whenever two vehicles overlapped. Track bboxes are motion-predicted first (`_predict_bbox`, EMA velocity over the last two sightings) so a briefly occluded vehicle re-matches — but that IoU-0 rescue only applies while the track has been lost ≤ `--track-predict-frames` (10). Beyond that a match needs real IoU: pushed tracks linger `3 × track_timeout` frames as dedup memory, and without the cap a stale one vacuums up a *new* vehicle in the same lane position and PATCHes its plate onto the old row.
   - **Plate→vehicle association is exclusive** (`_associate_plates`): one plate belongs to one vehicle. If a plate centroid falls inside several vehicle bboxes, the smallest wins only when it's clearly nested (`PLATE_NEST_RATIO`, e.g. a motorbike in front of a car); two similarly-sized overlapping cars are treated as **ambiguous and the plate is associated with neither** for that frame. Guessing writes the wrong plate and fuel onto a track; waiting costs a few frames.
   - **The push photo comes from a clean frame** (`_maybe_best_frame`): frames where this vehicle's bbox overlaps another vehicle's by IoU > `--track-max-overlap` (0.5) are not eligible. A clean frame always beats a dirty one; among clean frames the largest bbox (closest) wins. Dirty frames are used only if the track never had a clean one.

6. **`detected_at`** in the push payload comes from track `created_ts` (first-seen wall clock), NOT from push time. Push can lag by seconds due to upload queue; web team explicitly wants the detection moment. Format: ISO 8601 with offset TZ via `datetime.fromtimestamp(ts).astimezone().isoformat(timespec='seconds')` — naive datetimes are a contract violation.

## Web monitoring contract (v1)

POST to `/api/detections` (multipart/form-data) accepts ONLY these fields:

| Field | Values |
|---|---|
| `plate_number` | string, empty allowed (`""`) |
| `vehicle_type` | `"mobil"` \| `"motor"` \| `"unknown"` |
| `is_electric` | `"bensin"` \| `"listrik"` \| `"unknown"` |
| `confidence_score` | float 0.0–1.0 (vehicle YOLO score, not fuel score) |
| `detected_at` | ISO 8601 with TZ offset |
| `photo` | JPG file (server caps at 5 MB) |

**Values are Bahasa Indonesia, NOT English.** A prior iteration translated to `car`/`motorcycle`/`gasoline`/`electric` — that mapping was deliberately removed. Don't re-add it. Server coerces unknown enum values to `"unknown"` with a warning rather than rejecting.

Fields explicitly NOT sent (server has no column, will be silently dropped if added): `camera_id`, `client_event_id`, `location`, `lane`. Add them only when the web team signals their schema is ready.

Internal payload fields prefixed `_` (e.g. `_track_id`, `_image_file`, `_pushed_at`) are sidecar-only — `HttpUploader._strip_internal()` filters them before POST.

`HttpUploader` (`src/uploader.py`) sends from a background worker thread and supports two modes via `--push-mode`: `multipart` (default, includes the `photo` file) and `json` (same fields minus the photo, sent as a JSON body). The v1 field contract above is the multipart shape; `json` mode is a photo-less alternative for connectivity/smoke tests, not a different schema.

**Plate correction** (Juli 2026): whenever a settled plate reading (`plate_settle_frames`, same confirmation gate as push) **differs** from what was last sent to the web — late read (`""` → text) or corrected read (`"D 1 S"` → `"D 15 NW"` as the vehicle gets closer) — the tracker sends a follow-up `PATCH <push_url>/<id>/plate` (JSON: `plate_number`, plus `is_electric` if it was `unknown` at push time). The `<id>` comes from the 201 response and flows uploader → `pipeline._on_registered` → `tracker.set_server_id`; the update flows `tracker.on_update` → `pipeline._update_callback` → `uploader.push_update` (same queue/worker/X-API-Key as POSTs). Server-side rules: X-API-Key only (no session login), never touches the admin-correction audit fields, and skips records already corrected by an admin (200 SKIPPED). Only template-valid plate text ever enters the tracker (`PLATE_TEMPLATE_RE` filter at ingestion), so pushes and PATCHes can't carry garbage OCR reads.

Corrections do **not** run off the live track — they go through `VehicleTracker._pending`, a queue keyed by track id that outlives the track (three separate delivery bugs came from coupling them):
- **`pushed_plate` moves only on PATCH success.** `uploader.push_update(server_id, fields, on_result)` calls `on_result(ok)` from its worker thread; the tracker marks the plate delivered in `_on_update_result` and retries with exponential backoff (`--push-update-retries` 5, `--push-update-backoff` 2.0s, cap 60s) otherwise. Setting `pushed_plate` *before* the PATCH — the old code — made every failed PATCH a permanently lost plate. A 404 counts as terminal (record gone; retrying is pointless), not as a failure to retry.
- **A pending correction survives track death and a missing `server_id`.** `set_server_id` fills in pending entries too, so a plate that settles before the 201 comes back (or after the vehicle already left the frame) still gets PATCHed. Entries give up after `plate_update_ttl` (120s) without a `server_id`.
- **When a track dies, the settle gate is bypassed** in favour of the *most recent* reading seen at least `--push-plate-final-votes` (2) frames — not the most frequent one. OCR corrects in one direction (the vehicle gets closer, the read gets better), so majority-vote reliably picks the early *wrong* read. Without this, a plate only legible for a handful of frames at closest approach never cleared `plate_settle_frames` and was dropped entirely.

`tracker.flush()` (called by `pipeline.release()`) drains that queue for up to `update_wait` seconds before the uploader closes, so shutdown doesn't strand corrections. `tools/mock_server.py` implements the PATCH route — without it every local PATCH 404s and corrections look broken when they aren't.

**Correction guards** (Juli 2026, from a full vid1.mp4 A/B run — reliable delivery exposed that *what* got delivered was often wrong). All in `vehicle_tracker.py`; a correction candidate must pass every one:
- **No degradation** (`_correction_allowed`): a correction may never *reduce* the digit count of what's on the row. Real plates never lose digits as a vehicle approaches; a shrinking reading is OCR decaying as the vehicle recedes — and a decayed-but-consistent reading passes the settle gate, so settle alone can't catch this.
- **No contamination** (`_queue_plate_update`): if the candidate text is already registered to a *different* server row in `_recent_plates`, skip the PATCH — one plate can't own two rows. This is symptom control for plate-association bleed in dense traffic, not a root fix.
- **No flip-flop**: per-track `sent_plates` history — a text that was ever delivered for a row is never re-sent (OCR oscillating `L/F/E 1601 CF` used to PATCH one row 5×). Same-digit-count ("lateral") corrections also need **2× settle** while the track lives, and *more votes than the delivered text* on the expire path (`_final_correction_ok`).
- **Variant-tolerant dedup** (`_plates_similar` / `_find_recent_plate`): cross-track dedup matches OCR variants — same series + one number a suffix of the other (`S 805 TBZ` ≈ `B 5805 TBZ`), or exactly one *letter* differing. One *digit* differing is deliberately NOT a match (`D 1234 AB` vs `D 1235 AB` can be two real vehicles; a wrong merge loses a detection, a duplicate row is merely ugly). On adopt, `pushed_plate` is set to the **row's** text, so a richer own-reading naturally becomes an upgrade correction.
- **Stale-track plate veto** (`_pair_cost`): a pushed track lost longer than `track_timeout` (alive only as dedup memory) may not match a detection carrying a valid plate that isn't a variant of its own — in stop-and-go queues the next car advances into exactly the old car's position (high IoU) and used to get vacuumed into the dead track, which then PATCHed the old row with the new car's plate.
- **Plate association zone** (`_associate_plates`): a plate centroid in the top 35% of a vehicle bbox is never associated (plates live on the lower body; a top-zone hit is the vehicle behind), and candidates containing the *entire* plate bbox beat centroid-only candidates.

**Cross-track dedup** (Juli 2026): flickering detections used to fragment one vehicle into multiple tracks → duplicate rows on the web. Two complementary guards in `vehicle_tracker.py`, deliberately on different time-scales: (1) **IoU re-match** — pushed tracks are retained `3 × track_timeout` *frames* (they're light, `best_frame` freed) so a briefly-lost vehicle (sub-second flicker) re-matches its own track and PATCHes its old row instead of POSTing a new one; (2) **plate memory** — `_recent_plates` keyed by plate text, pruned by **wall-clock seconds** (`--push-dedup-seconds`, default 60, ≤0=off). This is the long, FPS-independent guard: a track whose plate was already pushed within the window skips the POST and adopts the old record's `server_id` (kept in sync by `set_server_id`). It's time-based on purpose — a frame-based window (the original bug) silently shrinks in wall-time as FPS rises, so a vehicle re-detected ~15 s later slipped through and made a duplicate row. Residual edge: a second track that force-pushes (plate-wait timeout) *before* its plate settles can still create a row that only later reveals a duplicate plate — rare, and only when OCR lags the push.

## Running

```powershell
# Desktop GUI (Tkinter): Start/Stop, source picker, web-monitoring connect+push
# toggle, live-to-browser MJPEG, and an overlay on/off toggle. Wraps the same
# DetectionPipeline as the CLI. (Use venv python; system python lacks Pillow/tk deps.)
venv\Scripts\python.exe src\gui.py

# Webcam, default (OCR ON, no web push, MJPEG preview ON at :5001)
python src\main.py

# Video file with anti-spam push to web monitoring
python src\main.py --source video.mp4 --push --push-url http://<ip>:5000/api/detections

# OCR off (smoke-test connectivity without PaddleOCR overhead)
python src\main.py --source video.mp4 --no-ocr

# Headless: no OpenCV window, no preview server, just JSONL log + push
python src\main.py --source video.mp4 --no-window --no-preview --log-jsonl output\run.jsonl --push --push-url http://<ip>:5000/api/detections

# Photo-less JSON push (connectivity test)
python src\main.py --source video.mp4 --push --push-mode json --push-url http://<ip>:5000/api/detections

# Unified single-model (one 3-class weight, 1 predict/frame instead of dual-model's 2)
python src\main.py --source video.mp4 --model "models\unified_3class.pt"

# Simulate 2-laptop setup on one machine (open 2 terminals)
tools\sim_laptop_b.bat          # mock web monitoring on 0.0.0.0:5000
tools\sim_laptop_a.bat video.mp4

# Mock server standalone, with simulated LAN latency / packet loss
python tools\mock_server.py --port 5000 --latency-ms 30 --drop-rate 0.1

# Print this machine's LAN IP (for pointing laptop A at laptop B)
python tools\get_lan_ip.py
```

Detection defaults worth knowing: `--imgsz 960` applies to the **plate** model and unified mode (lowered from 1280 for FPS; use 640 for max FPS, 1280 for distant plates); the **vehicle** model runs at `--vehicle-imgsz 640` (vehicles are large objects, yolov8n's native res — 0 = follow `--imgsz`; don't lower the plate imgsz instead, plates are the small objects). FP16 is auto-ON on CUDA (`--no-half` to force FP32; always FP32 on CPU). `--conf 0.4`, vehicle model `models/yolov8n.pt`, plate model `models/plate_best (1).pt`, `--model ''` (empty = dual-model; pass a 3-class weight to run unified single-model). Optional `--plate-conf` overrides plate threshold separately from `--conf`. **OCR is async** (Juli 2026): `detector._run_ocr` only schedules jobs; `plate_reader.read()` runs solely on the `ocr-worker` daemon thread (crop is copied per job, results land in `_plate_cache` a few frames late — the tracker settle-gate absorbs that). Don't call PaddleOCR from the detect loop again, and don't call `read()` from more than one thread. Push tuning lives behind `--push-*` flags (`--push-stable`, `--push-timeout-frames`, `--push-min-height`, `--push-max-dim`, `--push-jpeg-quality`, `--push-fuel-optional`, `--push-fuel-wait`); run `python src\main.py -h` for the full list.

**Live preview** (`src/preview_server.py`): a Flask MJPEG stream of the annotated frames, started on a daemon thread so it never blocks detection. Default `--preview-host 0.0.0.0 --preview-port 5001 --preview-quality 75`; disable with `--no-preview`. **JSONL logging** (`src/logger.py`, `DetectionLogger`): opt in with `--log-jsonl <path>`, one tail-able JSON row per detection per frame (`ts`, `frame`, `class`, `conf`, `bbox`, `fuel_type`, `blue_ratio`, `text`).

Interactive controls (during webcam/video playback, requires the OpenCV window — i.e. not `--no-window`): `Q` quit, `Space` pause, `S` screenshot to `output/`, `L` toggle loop, `←`/`→` step frames while paused.

## Evaluation

`evaluate.py` runs the full pipeline against `eval_set/` with ground-truth CSV and prints per-stage metrics (plate precision/recall/F1, fuel balanced accuracy, OCR exact match, end-to-end accuracy). Exit code 2 if end-to-end < 0.90 — usable as a CI/release gate.

```powershell
python evaluate.py --eval-set eval_set/ --gt eval_set/ground_truth.csv --out eval_set/results.csv
```

Ground truth CSV header (see `eval_set/README.md` for full spec):
```
filename,vehicle_type,plate_text,fuel_type,plate_x1,plate_y1,plate_x2,plate_y2
```

`eval_set/` itself is just a template — populate it (target ≥200 images across day/night/rain/angle conditions) before relying on the numbers.

## Training a new plate detector

If `evaluate.py` shows plate detection is the bottleneck, fine-tune via `train_plate.py`. Dataset layout (see `dataset/README.md`):

```
dataset/
├── data.yaml                    # nc: 1, names: {0: license_plate}
├── images/{train,val,test}/
└── labels/{train,val,test}/
```

YOLO format labels (`<class> <cx> <cy> <w> <h>`, normalized 0-1). After training, copy `runs/detect/*/weights/best.pt` to `models/license_plate_vN.pt` and pass via `--plate-model`.

Augmentation is tuned for plates: `flipud=0.0` (plates are never upside-down), small rotation only (`degrees=5.0`).

## Things to NOT do

- Don't translate vehicle/fuel enum values back to English in `vehicle_tracker.py` or `uploader.py`. The contract is Indonesian.
- Don't delete the dual-model path or make unified the default without explicit instruction. Both modes now ship in `detector.py` (`_detect_dual` / `_detect_unified`); dual-model is canonical and the default (`--model ''`). Unified is opt-in via `--model <3-class-weight>` — keep both working.
- Don't add fields to the multipart POST that aren't in the v1 contract — `camera_id`, `client_event_id`, etc. will be silently dropped by the server and create false confidence that they "work."
- Don't remove `crop_bottom_ratio` from `PlateReader` — the masa-berlaku-pajak strip on Indonesian plates will corrupt OCR output (`"B 2647 TZO"` becomes `"B647TZO722"` without it).
- Don't replace `time.localtime()`-style naive timestamps with TZ-aware ones in random places; the contract specifically wants `detected_at` to carry offset TZ (`+07:00`), while internal logging uses local time without offset.
- Don't run `python` (system) for any project script; always `venv\Scripts\python.exe` or activate via `start.bat`.
- Don't duplicate per-frame detection logic into `main.py` or `gui.py` — extend `DetectionPipeline` (`src/pipeline.py`) so both front-ends share it. Don't move `cap` ownership or FPS drawing into the pipeline (breaks CLI stepping and timing parity).
- Don't touch Tkinter widgets from the GUI worker thread, and don't drop the `_imgtk` reference in `gui.py` — either one blanks or crashes the preview.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
