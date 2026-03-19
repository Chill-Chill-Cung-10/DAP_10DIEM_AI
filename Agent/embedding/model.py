"""BGE-M3 embedding model with LRU cache."""

import hashlib
import logging
from collections import OrderedDict

from FlagEmbedding import BGEM3FlagModel

from Agent.config import EMBEDDING_CACHE_MAX

logger = logging.getLogger(__name__)
# usefp16: Dùng độ chính xác 16 bit
_model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True) 
_cache: OrderedDict[str, list] = OrderedDict()


def get_embedding(text: str) -> list:
    """Return dense embedding vector with LRU-(Least Recently Used) eviction."""
    # Tạo key để tra cứu trong cache
    key = hashlib.md5(text.encode()).hexdigest()

    if key in _cache:
        _cache.move_to_end(key)
        return _cache[key]

    vec = _model.encode([text])["dense_vecs"][0].tolist()
    _cache[key] = vec

    if len(_cache) > EMBEDDING_CACHE_MAX:
        _cache.popitem(last=False)

    return vec