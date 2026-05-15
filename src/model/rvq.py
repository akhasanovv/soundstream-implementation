import torch
import torch.nn.functional as F
from torch import nn


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, decay=0.99, epsilon=1e-5):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.decay = decay
        self.epsilon = epsilon

        embed = torch.randn(num_embeddings, embedding_dim)
        self.register_buffer("embed", embed)

        self.register_buffer("cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, x):  # (batch_size, emb_dim, time)
        x_flat = x.permute(0, 2, 1).reshape(
            -1, self.embedding_dim
        )  # (batch_size*time, emb_dim)

        distances = (
            torch.sum(x_flat**2, dim=1, keepdim=True)
            + torch.sum(self.embed**2, dim=1)
            - 2 * torch.matmul(x_flat, self.embed.t())
        )  # ||a-b||_2^2 -> a in x, b in embeddings
        indices = torch.argmin(distances, dim=1)

        z_q = F.embedding(indices, self.embed)
        z_q = z_q.view(x.permute(0, 2, 1).shape)
        z_q = z_q.permute(0, 2, 1).contiguous()

        with torch.no_grad():
            if self.training:
                x_flat_detached = x_flat.detach()
                one_hot = F.one_hot(indices, num_classes=self.num_embeddings).type_as(
                    x_flat_detached
                )

                batch_cluster_counts = one_hot.sum(dim=0)
                batch_embedding_sums = one_hot.T @ x_flat_detached

                self.cluster_size.mul_(self.decay)
                self.cluster_size.add_(batch_cluster_counts, alpha=1 - self.decay)

                self.embed_avg.mul_(self.decay)
                self.embed_avg.add_(batch_embedding_sums, alpha=1 - self.decay)

                total_count = self.cluster_size.sum()

                smoothed_cluster_size = (
                    (self.cluster_size + self.epsilon)
                    / (total_count + self.num_embeddings * self.epsilon)
                    * total_count
                )
                self.embed.copy_(self.embed_avg / smoothed_cluster_size.unsqueeze(1))

        loss = F.mse_loss(z_q.detach(), x)
        z_q = x + (z_q - x).detach()

        return z_q, loss, indices


class ResidualVectorQuantizer(nn.Module):
    """
    RVQ model from [soundstream paper](https://arxiv.org/abs/2107.03312)
    """

    def __init__(self, num_quantizers, num_embeddings, embedding_dim):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                VectorQuantizer(num_embeddings, embedding_dim)
                for _ in range(num_quantizers)
            ]
        )

    def forward(self, x):
        out = torch.zeros_like(x)
        residual = x
        all_indices = []

        for layer in self.layers:
            z_q, _, indices = layer(residual)
            residual = residual - z_q.detach()
            out = out + z_q
            all_indices.append(indices)

        commit_loss = F.mse_loss(out.detach(), x)

        return out, commit_loss, all_indices
