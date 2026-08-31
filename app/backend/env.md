# Server
HOST=0.0.0.0
PORT=8000
API_PREFIX=/api
SYSTEM_NAME=BoldSearcher

# Zilliz Cloud / Milvus
ZILLIZ_URI=https://in03-a9de8a8c8070a7b.serverless.aws-eu-central-1.cloud.zilliz.com
ZILLIZ_TOKEN=61e50779a14d6c243da6455b9c3888fa1002f1700ec85c0ac361ba9398eda5ed05170d447f499cee4cf066180ddc8a6283b84095
MILVUS_COLLECTION=BoldSearch
MILVUS_TEXT_FIELD=sparse_vector
MILVUS_VECTOR_FIELD=embedding
MILVUS_OUTPUT_FIELDS=frame_id,shot_id,video_id
MILVUS_TEXT_SEARCH_PARAMS={"metric_type":"BM25"}
MILVUS_VECTOR_SEARCH_PARAMS={"metric_type":"IP","params":{"nprobe":10}}
MILVUS_RANKER_WEIGHTS=1,0
SEARCH_TOP_K=50

# FG-CLIP encoder
LOAD_FG_CLIP_ON_STARTUP=true
FG_CLIP_DEVICE=mps

# Object detection CSV
# Relative paths are resolved from app/backend first; absolute paths are also supported.
OBJECTS_CSV_PATH=detections.csv

# Response presentation
FRAME_IMAGE_URL_TEMPLATE=
INCLUDE_EMBEDDING_IN_RESPONSE=false
