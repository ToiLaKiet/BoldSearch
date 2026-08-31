from functools import lru_cache
from typing import Optional

from encoders.fg_clip import FGClipEncoder
from encoders.beit3 import BEiT3Encoder


@lru_cache(maxsize=1)
def load_beit3_encoder(device: Optional[str] = None, hf_token: Optional[str] = None) -> BEiT3Encoder:
    return BEiT3Encoder(device=device)


@lru_cache(maxsize=1)
def load_fg_clip_encoder(device: Optional[str] = None, hf_token: Optional[str] = None) -> FGClipEncoder:
    return FGClipEncoder(device=device)
