from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app_config import AppConfig

FrameObjectIndex = Dict[Tuple[str, str, str], Dict[str, Any]]

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
_REQUIRED_COLUMNS = (
    "video_id",
    "frame_id",
    "object",
    "quantity",
    "bbox_x",
    "bbox_y",
    "bbox_w",
    "bbox_h",
)


def load_object_index(config: AppConfig) -> Optional[FrameObjectIndex]:
    csv_path = config.OBJECTS_CSV_PATH.strip()
    if not csv_path:
        return None
    path = _resolve_csv_path(csv_path)
    if not path.exists():
        raise RuntimeError(f"OBJECTS_CSV_PATH does not exist: {path}")
    if not path.is_file():
        raise RuntimeError(f"OBJECTS_CSV_PATH is not a file: {path}")

    return read_object_csv(path)


def read_object_csv(path: Path) -> FrameObjectIndex:
    docs_by_key: FrameObjectIndex = {}

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        _validate_columns(reader.fieldnames or [])

        for row_number, row in enumerate(reader, start=2):
            normalized = _normalized_row(row)
            video_id = normalized["video_id"]
            frame_id = normalized["frame_id"]
            label = _normalize_label(normalized["object"])
            quantity = 1

            if not video_id or not frame_id or not label or quantity <= 0:
                continue

            key = ("video_frame", video_id, frame_id)
            doc = docs_by_key.setdefault(
                key,
                {
                    "video_id": video_id,
                    "frame_id": frame_id,
                    "objects": {},
                    "detections": [],
                    "source": str(path),
                    "rowCount": 0,
                    "lastRow": row_number,
                },
            )

            doc["objects"][label] = int(doc["objects"].get(label, 0)) + quantity
            doc["detections"].append(
                {
                    "object": label,
                    "quantity": quantity,
                    "bbox": {
                        "x": _number_or_none(normalized["bbox_x"]),
                        "y": _number_or_none(normalized["bbox_y"]),
                        "w": _number_or_none(normalized["bbox_w"]),
                        "h": _number_or_none(normalized["bbox_h"]),
                    },
                }
            )
            doc["rowCount"] += 1
            doc["lastRow"] = row_number

    return docs_by_key


def object_doc_for_row(
    row: Dict[str, Any],
    object_index: Optional[FrameObjectIndex],
) -> Dict[str, Any]:
    if not object_index:
        return {}
    video_id = _clean_value(row.get("video_id"))
    frame_id = _clean_value(row.get("frame_id"))
    if not video_id or not frame_id:
        return {}

    return object_index.get(("video_frame", video_id, frame_id), {})


def _resolve_csv_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path

    backend_relative = (_BACKEND_ROOT / path).resolve()
    if backend_relative.exists():
        return backend_relative

    cwd_relative = (Path.cwd() / path).resolve()
    if cwd_relative.exists():
        return cwd_relative

    return backend_relative


def _validate_columns(fieldnames: List[str]) -> None:
    columns = {name.strip() for name in fieldnames if name}
    missing = [name for name in _REQUIRED_COLUMNS if name not in columns]
    if missing:
        raise ValueError(f"objects.csv is missing required columns: {', '.join(missing)}")


def _normalized_row(row: Dict[str, Any]) -> Dict[str, str]:
    stripped_row = {str(key).strip(): value for key, value in row.items() if key}
    return {key: _clean_value(stripped_row.get(key)) for key in _REQUIRED_COLUMNS}


def _clean_value(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _normalize_label(value: Any) -> str:
    return str(value).strip().lower()


def _quantity(value: Any) -> int:
    try:
        return max(int(float(str(value).strip())), 0)
    except (TypeError, ValueError):
        return 0


def _number_or_none(value: Any) -> Optional[int | float]:
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None

    return int(number) if number.is_integer() else number
