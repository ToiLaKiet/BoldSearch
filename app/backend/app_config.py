from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_FILE = Path(__file__).resolve().with_name(".env")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"


class DeployKaggle(BaseModel):
    """Kaggle deployment flow settings (deploy/kaggle wrapper + CI).

    Values flow: repo defaults -> CI injects REPO_URL from the GitHub context
    into the private env dataset -> the wrapper notebook copies that dataset
    to `backend/.env` -> pydantic validates everything here. Deploys always
    track `main`.
    """

    REPO_URL: str = "https://github.com/ToiLaKiet/BoldSearch.git"
    SMOKE_QUERY: str = "person"
    FRONTEND_PORT: int = 5173
    GH_PAT: str = ""  # fine-grained contents:read; used by the wrapper to clone


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
    MILVUS_COLLECTION: str = "BoldSearch"
    MILVUS_VECTOR_FIELD: str = "embedding"
    MILVUS_OUTPUT_FIELDS: str = (
        "frame_id,shot_id,video_id,asr_text,ocr_text,distance,thumbnail"
    )
    MILVUS_TEXT_SEARCH_PARAMS: str = '{"metric_type":"BM25"}'
    MILVUS_VECTOR_SEARCH_PARAMS: str = '{"metric_type":"IP","params":{"nprobe":10}}'
    MILVUS_RANKER_WEIGHTS: str = "0,1"
    SEARCH_TOP_K: int = 3000

    # ── Embedding encoder ─────────────────────────────────────────
    LOAD_FG_CLIP_ON_STARTUP: bool = True
    FG_CLIP_DEVICE: str = "mps"
    HF_TOKEN: str = "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"
    LOAD_BEIT3_ON_STARTUP: bool = True
    BEIT3_DEVICE: str = "mps"

    # ── Object detection metadata ───────────────────────────────
    OBJECTS_CSV_PATH: str = "detections.csv"

    # ── Local data & evaluation artifacts ────────────────────────
    DATA_DIR: Path = DEFAULT_DATA_DIR
    KEYFRAMES_DIR: Path = DEFAULT_DATA_DIR / "keyframes"
    KEYFRAME_MAP_DIR: Path = DEFAULT_DATA_DIR / "map-keyframes"
    EVALUATION_ARTIFACT_DIR: Path = DEFAULT_DATA_DIR / "evaluation-artifacts"

    # ── Evaluation run labels (fill via .env for this machine) ──
    EVALUATION_CORPUS_VERSION: str = ""
    EVALUATION_PREPROCESSING_VERSION: str = ""
    EVALUATION_QUERY_STRATEGY: str = ""

    # ── Response presentation ────────────────────────────────────
    FRAME_IMAGE_URL_TEMPLATE: str = ""
    INCLUDE_EMBEDDING_IN_RESPONSE: bool = False

    # ── Kaggle deployment (deploy/kaggle) ─────────────────────────
    DEPLOY_KAGGLE: DeployKaggle = DeployKaggle()

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
        env_nested_delimiter="__",
    )

    @field_validator(
        "DATA_DIR",
        "KEYFRAMES_DIR",
        "KEYFRAME_MAP_DIR",
        "EVALUATION_ARTIFACT_DIR",
        mode="after",
    )
    @classmethod
    def _resolve_local_path(cls, value: Path) -> Path:
        path = value.expanduser()
        return path.resolve() if path.is_absolute() else (ENV_FILE.parent / path).resolve()

    @model_validator(mode="after")
    def _re_root_data_children(self) -> "AppConfig":
        if "DATA_DIR" not in self.model_fields_set:
            return self

        child_paths = {
            "KEYFRAMES_DIR": "keyframes",
            "KEYFRAME_MAP_DIR": "map-keyframes",
            "EVALUATION_ARTIFACT_DIR": "evaluation-artifacts",
        }
        for field_name, child_path in child_paths.items():
            if field_name not in self.model_fields_set:
                setattr(self, field_name, self.DATA_DIR / child_path)
        return self


app_config = AppConfig()
