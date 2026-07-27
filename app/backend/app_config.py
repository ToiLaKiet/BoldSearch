from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_FILE = Path(__file__).resolve().with_name(".env")


class AppConfig(BaseSettings):
    """Settings read from the environment or a local `.env`."""

    # ── API ──────────────────────────────────────────────────────
    SYSTEM_NAME: str = "BoldSearch"
    API_PREFIX: str = "/api"

    # ── Server ───────────────────────────────────────────────────
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # ── Zilliz / Milvus ──────────────────────────────────────────
    ZILLIZ_URI: str = "https://in03-a9de8a8c8070a7b.serverless.aws-eu-central-1.cloud.zilliz.com"
    ZILLIZ_TOKEN: str = "61e50779a14d6c243da6455b9c3888fa1002f1700ec85c0ac361ba9398eda5ed05170d447f499cee4cf066180ddc8a6283b84095"
    MILVUS_COLLECTION: str = "BoldSearcher_v4"
    MILVUS_VECTOR_FIELD: str = "embedding"
    MILVUS_OUTPUT_FIELDS: str = (
        "frame_id,shot_id,video_id,asr_text,ocr_text,distance,thumbnail"
    )
    MILVUS_TEXT_SEARCH_PARAMS: str = '{"metric_type":"BM25"}'
    MILVUS_VECTOR_SEARCH_PARAMS: str = '{"metric_type":"IP","params":{"nprobe":10}}'
    MILVUS_RANKER_WEIGHTS: str = "0.6,0.2,0.2"
    SEARCH_TOP_K: int = 3000

    # ── Embedding encoder ─────────────────────────────────────────
    LOAD_FG_CLIP_ON_STARTUP: bool = True
    FG_CLIP_DEVICE: str = "mps"
    HF_TOKEN: str = "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"

    # ── Object detection metadata ───────────────────────────────
    OBJECTS_CSV_PATH: str = "detections.csv"

    # ── Response presentation ────────────────────────────────────
    FRAME_IMAGE_URL_TEMPLATE: str = ""
    INCLUDE_EMBEDDING_IN_RESPONSE: bool = False

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )


app_config = AppConfig()
