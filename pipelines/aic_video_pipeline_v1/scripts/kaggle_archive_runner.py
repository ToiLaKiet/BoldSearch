from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import queue
import shutil
import subprocess
import sys
import tarfile
import tempfile
import traceback
import zipfile
from pathlib import Path, PurePosixPath


DEFAULT_ARCHIVES = [
    "Videos_L26_a.zip",
    "Videos_L26_b.zip",
    "Videos_L26_c.zip",
    "Videos_L26_d.zip",
    "Videos_L26_e.zip",
]


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    os.replace(temporary, path)


def load_state(path: Path) -> dict:
    if not path.is_file():
        return {"schema_version": "1.0", "archives": {}}
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != "1.0" or not isinstance(value.get("archives"), dict):
        raise ValueError(f"invalid runner state: {path}")
    return value


def save_state(work_path: Path, result_root: Path, value: dict) -> None:
    atomic_json(work_path, value)
    atomic_json(result_root / "runner_state.json", value)


def merge_state(target: dict, source: dict) -> None:
    for archive_name, source_archive in source.get("archives", {}).items():
        target_archive = target["archives"].setdefault(
            archive_name, {"completed_members": [], "done": False}
        )
        merged = list(dict.fromkeys(
            list(target_archive.get("completed_members", []))
            + list(source_archive.get("completed_members", []))
        ))
        target_archive["completed_members"] = merged
        target_archive["done"] = bool(
            target_archive.get("done") or source_archive.get("done")
        )


def member_video_id(member_name: str) -> str:
    return Path(PurePosixPath(member_name).name).stem


def reconcile_completed_members(state: dict, result_roots: list[Path]) -> None:
    """Trust only complete outputs, never runner_state or a .tar.tmp alone.

    A Kaggle output can be captured while a session is stopping.  Its state file
    may be newer than its artifacts, so every member is revalidated against the
    available result roots before an archive is skipped.  Kaggle may expose a
    saved TAR as an extracted ``result_root/video_id`` directory, which is also
    accepted after its metadata, frame, vector, and checkpoint are verified.
    """
    for archive in state.get("archives", {}).values():
        verified: list[str] = []
        for member_name in archive.get("completed_members", []):
            video_id = member_video_id(member_name)
            if any(result_exists(root, video_id) for root in result_roots):
                verified.append(member_name)
        archive["completed_members"] = list(dict.fromkeys(verified))
        archive["done"] = False


def free_bytes(path: Path) -> int:
    path.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(path).free


def human_size(value: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    size = float(value)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024
    return str(value)


def valid_result_tar(path: Path, video_id: str) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        with tarfile.open(path, "r") as archive:
            members = {member.name: member for member in archive.getmembers()}
        required = (
            f"metadata/{video_id}/Shot.json",
            f"metadata/{video_id}/Frame.json",
            f"checkpoints/{video_id}.json",
        )
        if not all(
            name in members and members[name].isfile() and members[name].size > 0
            for name in required
        ):
            return False
        return (
            any(member.isfile() and member.size > 0
                and member.name.startswith(f"frames/{video_id}/")
                and member.name.endswith(".png") for member in members.values())
            and any(member.isfile() and member.size > 0
                    and member.name.startswith(f"vectors/{video_id}/")
                    and member.name.endswith(".npy") for member in members.values())
        )
    except (OSError, tarfile.TarError):
        return False


def valid_result_directory(path: Path, video_id: str) -> bool:
    """Validate one TAR payload after Kaggle has extracted it to a directory."""
    if not path.is_dir() or path.name != video_id:
        return False

    # Normal Kaggle extraction preserves the paths stored by package_video().
    # The second layout also accepts tools that strip the repeated video_id
    # directory while unpacking the archive.
    layouts = (
        (
            path / "metadata" / video_id,
            path / "frames" / video_id,
            path / "vectors" / video_id,
        ),
        (path / "metadata", path / "frames", path / "vectors"),
    )
    checkpoint = path / "checkpoints" / f"{video_id}.json"
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        return False

    for metadata, frames, vectors in layouts:
        if not (
            (metadata / "Shot.json").is_file()
            and (metadata / "Frame.json").is_file()
            and frames.is_dir()
            and vectors.is_dir()
        ):
            continue
        if not any(item.is_file() for item in frames.glob("*.png")):
            continue
        if not any(item.is_file() for item in vectors.glob("*.npy")):
            continue
        return True
    return False


def result_exists(result_root: Path, video_id: str) -> bool:
    """Return true for either a final TAR or Kaggle's extracted TAR folder."""
    return (
        valid_result_tar(result_root / f"{video_id}.tar", video_id)
        or valid_result_directory(result_root / video_id, video_id)
    )


def make_metadata_portable(data_root: Path, video_id: str) -> None:
    """Replace ephemeral runtime paths with paths relative to the result TAR."""
    frame_path = data_root / "metadata" / video_id / "Frame.json"
    checkpoint_path = data_root / "checkpoints" / f"{video_id}.json"

    frame_document = json.loads(frame_path.read_text(encoding="utf-8"))
    if (frame_document.get("video_id") != video_id
            or not isinstance(frame_document.get("frames"), list)):
        raise ValueError(f"invalid Frame.json for packaging: {video_id}")

    source_path = frame_document.get("source_video_path")
    if source_path:
        frame_document["source_video_path"] = Path(str(source_path)).name
    for frame in frame_document["frames"]:
        if not isinstance(frame, dict):
            raise ValueError(f"invalid frame record for packaging: {video_id}")
        for key, directory, suffix in (
            ("frame_path", "frames", ".png"),
            ("vector_path", "vectors", ".npy"),
        ):
            value = frame.get(key)
            if not value:
                continue
            name = Path(str(value)).name
            if Path(name).suffix != suffix:
                raise ValueError(f"invalid {key} for packaging: {value}")
            frame[key] = f"{directory}/{video_id}/{name}"
    atomic_json(frame_path, frame_document)

    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    source = checkpoint.get("source")
    if isinstance(source, dict) and source.get("path"):
        source["path"] = Path(str(source["path"])).name
    atomic_json(checkpoint_path, checkpoint)


def package_video(data_root: Path, result_root: Path, video_id: str) -> Path:
    result_root.mkdir(parents=True, exist_ok=True)
    output = result_root / f"{video_id}.tar"
    temporary = output.with_suffix(".tar.tmp")
    temporary.unlink(missing_ok=True)
    sources = [
        (data_root / "metadata" / video_id, Path("metadata") / video_id),
        (data_root / "frames" / video_id, Path("frames") / video_id),
        (data_root / "vectors" / video_id, Path("vectors") / video_id),
        (data_root / "checkpoints" / f"{video_id}.json",
         Path("checkpoints") / f"{video_id}.json"),
    ]
    for source, _arcname in sources:
        if not source.exists():
            raise FileNotFoundError(f"cannot package missing pipeline output: {source}")
    make_metadata_portable(data_root, video_id)
    with tarfile.open(temporary, "w") as archive:
        for source, arcname in sources:
            archive.add(source, arcname=str(arcname), recursive=True)
    os.replace(temporary, output)
    if not valid_result_tar(output, video_id):
        raise ValueError(f"invalid result archive: {output}")
    return output


def remove_materialized_video(data_root: Path, video_id: str) -> None:
    for path in (data_root / "metadata" / video_id,
                 data_root / "frames" / video_id,
                 data_root / "vectors" / video_id):
        if path.exists():
            shutil.rmtree(path)
    (data_root / "checkpoints" / f"{video_id}.json").unlink(missing_ok=True)


def extract_member(archive: zipfile.ZipFile, member: zipfile.ZipInfo,
                   video_path: Path) -> None:
    video_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{video_path.name}.", suffix=".part", dir=video_path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with archive.open(member, "r") as source, temporary.open("wb") as target:
            shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
        if temporary.stat().st_size != member.file_size:
            raise ValueError(f"truncated extracted video: {member.filename}")
        os.replace(temporary, video_path)
    finally:
        temporary.unlink(missing_ok=True)


def gpu_worker_main(worker_id: int, settings: dict[str, str], task_queue,
                    result_queue) -> None:
    """Keep one FP32 FG-CLIP2 instance resident on one physical GPU."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(worker_id)
    os.environ["MPLCONFIGDIR"] = str(
        Path(settings["work_root"]) / "matplotlib" / f"gpu_{worker_id}"
    )
    source_root = Path(settings["pipeline_root"]) / "src"
    sys.path.insert(0, str(source_root))
    try:
        from aic_video_pipeline_v1.orchestrator import VideoPipelineV1

        pipeline = VideoPipelineV1.from_yaml(
            Path(settings["config"]),
            data_root=Path(settings["data_root"]),
            model_path=Path(settings["model_path"]),
            autoshot_root=Path(settings["autoshot_root"]),
            autoshot_checkpoint=Path(settings["autoshot_checkpoint"]),
        )
        print(f"[GPU worker {worker_id}] ready", flush=True)
        while True:
            task = task_queue.get()
            if task is None:
                return
            video_path = Path(task["video_path"])
            video_id = str(task["video_id"])
            try:
                print(f"[GPU {worker_id}] start {video_id}", flush=True)
                result = pipeline.run_streaming(video_path, video_id)
                result_queue.put({"worker_id": worker_id, "ok": True,
                                  "video_id": video_id, "result": result})
            except BaseException as error:
                result_queue.put({
                    "worker_id": worker_id,
                    "ok": False,
                    "video_id": video_id,
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(),
                })
    except BaseException as error:
        result_queue.put({
            "worker_id": worker_id,
            "ok": False,
            "video_id": None,
            "error": f"worker initialization failed: {type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        })


def stop_workers(processes: list[mp.Process], task_queues: list) -> None:
    for task_queue in task_queues:
        try:
            task_queue.put_nowait(None)
        except queue.Full:
            pass
    for process in processes:
        process.join(timeout=5)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)


def wait_for_worker_result(result_queue, processes: list[mp.Process]) -> dict:
    while True:
        try:
            return result_queue.get(timeout=5)
        except queue.Empty:
            dead = [process.name for process in processes
                    if not process.is_alive()]
            if dead:
                raise RuntimeError(f"GPU worker exited unexpectedly: {dead}")


def previous_result_exists(previous_roots: list[Path], video_id: str) -> bool:
    return any(result_exists(root, video_id) for root in previous_roots)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Disk-bounded Kaggle runner with persistent one-process-per-GPU workers"
    )
    parser.add_argument("--pipeline-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--autoshot-root", type=Path, required=True)
    parser.add_argument("--autoshot-checkpoint", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--previous-result-root", type=Path, action="append", default=[])
    parser.add_argument("--base-url", default="https://aic-data.ledo.io.vn")
    parser.add_argument("--archives", nargs="*", default=DEFAULT_ARCHIVES)
    parser.add_argument("--gpu-workers", type=int, default=1,
                        help="persistent GPU workers (Kaggle dual T4: 2)")
    args = parser.parse_args()

    args.pipeline_root = args.pipeline_root.expanduser().resolve()
    args.config = args.config.expanduser().resolve()
    args.model_path = args.model_path.expanduser().resolve()
    args.autoshot_root = args.autoshot_root.expanduser().resolve()
    args.autoshot_checkpoint = args.autoshot_checkpoint.expanduser().resolve()
    args.work_root = args.work_root.expanduser().resolve()
    args.result_root = args.result_root.expanduser().resolve()
    args.previous_result_root = [path.expanduser().resolve()
                                 for path in args.previous_result_root]
    if not 1 <= args.gpu_workers <= 2:
        raise ValueError("gpu-workers must be 1 or 2")

    for required in (args.pipeline_root / "src", args.config, args.model_path,
                     args.autoshot_root, args.autoshot_checkpoint):
        if not required.exists():
            raise FileNotFoundError(required)

    download_root = args.work_root / "downloads"
    video_root = args.work_root / "video"
    data_root = args.work_root / "pipeline_data"
    state_path = args.work_root / "runner_state.json"
    for directory in (download_root, video_root, data_root, args.result_root):
        directory.mkdir(parents=True, exist_ok=True)

    state = load_state(state_path)
    for previous_root in args.previous_result_root:
        previous_state = previous_root / "runner_state.json"
        if previous_state.is_file():
            merge_state(state, load_state(previous_state))
    reconcile_completed_members(
        state, [args.result_root, *args.previous_result_root]
    )
    save_state(state_path, args.result_root, state)

    context = mp.get_context("spawn")
    task_queues = [context.Queue(maxsize=1) for _ in range(args.gpu_workers)]
    result_queue = context.Queue()
    settings = {
        "pipeline_root": str(args.pipeline_root),
        "config": str(args.config),
        "model_path": str(args.model_path),
        "autoshot_root": str(args.autoshot_root),
        "autoshot_checkpoint": str(args.autoshot_checkpoint),
        "work_root": str(args.work_root),
        "data_root": str(data_root),
    }
    processes = [
        context.Process(
            target=gpu_worker_main,
            args=(worker_id, settings, task_queues[worker_id], result_queue),
            name=f"gpu-worker-{worker_id}",
        )
        for worker_id in range(args.gpu_workers)
    ]
    for process in processes:
        process.start()

    try:
        for archive_name in args.archives:
            expected_level = archive_name.removeprefix("Videos_").split("_", 1)[0]
            archive_state = state["archives"].setdefault(
                archive_name, {"completed_members": [], "done": False}
            )
            if archive_state.get("done"):
                print(f"[SKIP archive] {archive_name}", flush=True)
                continue

            archive_path = download_root / archive_name
            url = f"{args.base_url.rstrip('/')}/{archive_name}"
            save_state(state_path, args.result_root, state)
            print(f"[DOWNLOAD] {archive_name}", flush=True)
            subprocess.run(["wget", "-c", "--progress=dot:giga", url,
                            "-O", str(archive_path)], check=True)

            with zipfile.ZipFile(archive_path) as archive:
                members = [item for item in archive.infolist()
                           if not item.is_dir() and item.filename.lower().endswith(".mp4")]
                basenames = [PurePosixPath(item.filename).name for item in members]
                if not members:
                    raise ValueError(f"no MP4 in {archive_name}")
                if len(set(basenames)) != len(basenames):
                    raise ValueError(f"duplicate MP4 basenames in {archive_name}")
                invalid_levels = [
                    basename for basename in basenames
                    if not Path(basename).stem.startswith(f"{expected_level}_V")
                ]
                if invalid_levels:
                    raise ValueError(
                        f"{archive_name} contains video from another level: "
                        f"{invalid_levels[:5]}"
                    )

                completed = set(archive_state.get("completed_members", []))
                jobs: list[dict] = []
                for number, (member, basename) in enumerate(zip(members, basenames), 1):
                    video_id = Path(basename).stem
                    result_tar = args.result_root / f"{video_id}.tar"
                    if (member.filename in completed
                            or valid_result_tar(result_tar, video_id)
                            or previous_result_exists(args.previous_result_root, video_id)):
                        if member.filename not in completed:
                            archive_state["completed_members"].append(member.filename)
                            completed.add(member.filename)
                            save_state(state_path, args.result_root, state)
                        print(f"[SKIP video] {basename}", flush=True)
                        continue
                    jobs.append({"number": number, "member": member,
                                 "basename": basename, "video_id": video_id})

                available = list(range(args.gpu_workers))
                active: dict[int, dict] = {}
                next_job = 0
                while next_job < len(jobs) or active:
                    while available and next_job < len(jobs):
                        worker_id = available.pop(0)
                        job = jobs[next_job]
                        next_job += 1
                        video_path = video_root / job["basename"]
                        member = job["member"]
                        if (not video_path.is_file()
                                or video_path.stat().st_size != member.file_size):
                            video_path.unlink(missing_ok=True)
                            print(
                                f"[EXTRACT {job['number']}/{len(members)}] "
                                f"{job['basename']}", flush=True
                            )
                            extract_member(archive, member, video_path)
                        job["video_path"] = video_path
                        active[worker_id] = job
                        task_queues[worker_id].put({
                            "video_path": str(video_path),
                            "video_id": job["video_id"],
                        })
                        print(
                            f"[DISPATCH GPU {worker_id}] {job['basename']}", flush=True
                        )

                    if not active:
                        continue
                    message = wait_for_worker_result(result_queue, processes)
                    worker_id = int(message["worker_id"])
                    job = active.pop(worker_id, None)
                    if job is None:
                        raise RuntimeError(f"unexpected result from GPU worker {worker_id}")
                    if not message.get("ok"):
                        raise RuntimeError(
                            f"GPU {worker_id} failed for {job['basename']}: "
                            f"{message.get('error')}\n{message.get('traceback', '')}"
                        )

                    video_id = job["video_id"]
                    print(f"[PACKAGE] {video_id}.tar", flush=True)
                    package_video(data_root, args.result_root, video_id)
                    remove_materialized_video(data_root, video_id)
                    Path(job["video_path"]).unlink(missing_ok=True)
                    archive_state["completed_members"].append(job["member"].filename)
                    completed.add(job["member"].filename)
                    save_state(state_path, args.result_root, state)
                    print(
                        f"[DONE GPU {worker_id}] {job['basename']}; "
                        f"free={human_size(free_bytes(args.work_root))}", flush=True
                    )
                    available.append(worker_id)

            archive_state["done"] = True
            save_state(state_path, args.result_root, state)
            archive_path.unlink(missing_ok=True)
            print(f"[DONE archive] {archive_name}; ZIP deleted", flush=True)

        save_state(state_path, args.result_root, state)
        print(f"All requested archives completed. Results: {args.result_root}", flush=True)
        return 0
    finally:
        stop_workers(processes, task_queues)


if __name__ == "__main__":
    raise SystemExit(main())
