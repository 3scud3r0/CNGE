import torch
from torch.nn import functional as F


class FoundationEmbeddingBridge:
    def __init__(self, model_identifier="BAAI/bge-m3", device="cpu", *, backend=None):
        if backend is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "install cnge-engine[semantic] to use foundation embeddings"
                ) from exc
            backend = SentenceTransformer(model_identifier, device=device)
        self.model, self.device = backend, device

    def embed_text(self, texts):
        if not texts or not all(isinstance(t, str) and t.strip() for t in texts):
            raise ValueError("texts must be nonempty strings")
        with torch.no_grad():
            result = self.model.encode(texts, convert_to_tensor=True, normalize_embeddings=True)
        return torch.as_tensor(result, device=self.device).detach()

    @staticmethod
    def similarity(v1, v2):
        return (F.normalize(v1, dim=-1) * F.normalize(v2, dim=-1)).sum(-1)
