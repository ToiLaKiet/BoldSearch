from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request


DATA_PATH = Path(__file__).resolve().parent / "data" / "shots.json"


def create_app() -> Flask:
    app = Flask(__name__)

    @app.after_request
    def add_cors_headers(response):
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
        return response

    @app.route("/api/health")
    def health():
        return jsonify({"status": "ok", "system": "BoldSearcher"})

    @app.route("/api/tasks")
    def tasks():
        return jsonify(
            {
                "system": "BoldSearcher",
                "tasks": [
                    {
                        "id": "KIS",
                        "name": "Known Item Search",
                        "description": "Find one known target shot from a textual description.",
                        "recommendedSignals": ["caption", "transcript", "objects", "time"],
                    },
                    {
                        "id": "VKIS",
                        "name": "Visual Known Item Search",
                        "description": "Find one known target shot from visual cues such as objects, colors, layout, or an image reference.",
                        "recommendedSignals": ["image", "objects", "colors", "temporal"],
                    },
                ],
            }
        )

    @app.route("/api/shots")
    def shots():
        return jsonify({"shots": load_shots()})

    @app.route("/api/search", methods=["POST", "OPTIONS"])
    def search():
        if request.method == "OPTIONS":
            return ("", 204)

        payload = request.get_json(silent=True) or {}
        query = str(payload.get("query", "")).strip()
        task = str(payload.get("task", "KIS")).upper()
        modalities = payload.get("modalities") or []
        objects = payload.get("objects") or []
        colors = payload.get("colors") or []
        temporal = payload.get("temporal") or ""
        min_confidence = float(payload.get("minConfidence") or 0)

        results = []
        for shot in load_shots():
            score, reasons = score_shot(
                shot=shot,
                query=query,
                task=task,
                modalities=modalities,
                objects=objects,
                colors=colors,
                temporal=temporal,
            )
            if score >= min_confidence:
                enriched = dict(shot)
                enriched["score"] = round(score, 3)
                enriched["reasons"] = reasons
                results.append(enriched)

        results.sort(key=lambda item: item["score"], reverse=True)
        return jsonify(
            {
                "system": "BoldSearcher",
                "task": task if task in {"KIS", "VKIS"} else "KIS",
                "query": query,
                "count": len(results),
                "results": results,
            }
        )

    @app.route("/api/submit", methods=["POST", "OPTIONS"])
    def submit():
        if request.method == "OPTIONS":
            return ("", 204)

        payload = request.get_json(silent=True) or {}
        shot_id = payload.get("shotId")
        shot = next((item for item in load_shots() if item["id"] == shot_id), None)
        if not shot:
            return jsonify({"status": "error", "message": "Shot not found"}), 404

        return jsonify(
            {
                "status": "accepted",
                "system": "BoldSearcher",
                "submission": {
                    "shotId": shot["id"],
                    "videoId": shot["videoId"],
                    "timestamp": shot["start"],
                },
            }
        )

    return app


def load_shots() -> list[dict[str, Any]]:
    with DATA_PATH.open(encoding="utf-8") as file:
        return json.load(file)["shots"]


def score_shot(
    shot: dict[str, Any],
    query: str,
    task: str,
    modalities: list[str],
    objects: list[str],
    colors: list[str],
    temporal: str,
) -> tuple[float, list[str]]:
    reasons: list[str] = []
    searchable = " ".join(
        [
            shot.get("title", ""),
            shot.get("description", ""),
            shot.get("transcript", ""),
            " ".join(shot.get("tags", [])),
            " ".join(shot.get("objects", [])),
            " ".join(shot.get("colors", [])),
            shot.get("location", ""),
        ]
    ).lower()

    query_tokens = tokenize(query)
    matches = [token for token in query_tokens if token in searchable]
    text_score = len(matches) / max(len(query_tokens), 1)
    if matches:
        reasons.append(f"Text match: {', '.join(matches[:4])}")

    requested_objects = normalize_list(objects)
    object_matches = [
        value for value in requested_objects if value in normalize_list(shot.get("objects", []))
    ]
    object_score = len(object_matches) / max(len(requested_objects), 1)
    if object_matches:
        reasons.append(f"Objects: {', '.join(object_matches)}")

    requested_colors = normalize_list(colors)
    color_matches = [
        value for value in requested_colors if value in normalize_list(shot.get("colors", []))
    ]
    color_score = len(color_matches) / max(len(requested_colors), 1)
    if color_matches:
        reasons.append(f"Colors: {', '.join(color_matches)}")

    temporal_score = 0.0
    if temporal:
        temporal_tokens = tokenize(temporal)
        temporal_matches = [token for token in temporal_tokens if token in searchable]
        temporal_score = len(temporal_matches) / max(len(temporal_tokens), 1)
        if temporal_matches:
            reasons.append(f"Temporal cue: {', '.join(temporal_matches[:3])}")

    modality_bonus = 0.04 * len(set(modalities).intersection({"text", "image", "temporal", "objects"}))
    confidence = float(shot.get("confidence", 0.6))

    if task == "VKIS":
        raw_score = (
            text_score * 0.32
            + object_score * 0.24
            + color_score * 0.20
            + temporal_score * 0.12
            + confidence * 0.12
            + modality_bonus
        )
    else:
        raw_score = (
            text_score * 0.52
            + object_score * 0.14
            + color_score * 0.08
            + temporal_score * 0.10
            + confidence * 0.16
            + modality_bonus
        )

    if not reasons:
        reasons.append("Ranked by baseline confidence")

    return min(math.ceil(raw_score * 1000) / 1000, 1.0), reasons


def tokenize(value: str) -> list[str]:
    return [token for token in re.split(r"[^a-zA-Z0-9]+", value.lower()) if len(token) > 2]


def normalize_list(values: list[str]) -> list[str]:
    return [str(value).strip().lower() for value in values if str(value).strip()]


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
