from functools import lru_cache
from typing import Optional
from encoders.fg_clip import FGClipEncoder

@lru_cache(maxsize=1)
def load_fg_clip_encoder(device: Optional[str] = None, hf_token: Optional[str] = None) -> FGClipEncoder:
    return FGClipEncoder(device=device)
