import torch
from torch import nn
from torch.nn import functional as F


class ManifoldOptimalTransportBridge(nn.Module):
    """Contrastive projection, not a diffeomorphism or a GW solver."""

    def __init__(self, input_signature_dim, embedding_dim):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_signature_dim, 128), nn.SiLU(), nn.Linear(128, embedding_dim)
        )

    def project_geometry_to_semantic(self, signature):
        return F.normalize(self.projection(signature), dim=-1)

    def alignment_loss(self, signatures, embeddings, temperature=0.07):
        if temperature <= 0 or len(signatures) != len(embeddings) or len(signatures) < 2:
            raise ValueError("need paired batches >=2 and positive temperature")
        logits = (
            self.project_geometry_to_semantic(signatures)
            @ F.normalize(embeddings.detach(), dim=-1).T
            / temperature
        )
        labels = torch.arange(len(signatures), device=signatures.device)
        return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
