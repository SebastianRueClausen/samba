from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor
from mamba_ssm import Mamba


@dataclass
class VocabPart:
    size: int
    loss_weight: float


@dataclass
class SambaConfig:
    vocab_parts: list[VocabPart]
    dropout_rate: float
    embedded_size: int
    batch_size: int
    head_count: int
    layer_count: int
    d_state: int
    d_conv: int
    device: str

    def state_dict(self) -> dict:
        return vars(self)

    def load_state_dict(self, dict: dict):
        for k, v in dict.items():
            setattr(self, k, v)


class FeedForward(nn.Module):
    def __init__(self, dropout_rate: float, embedded_size: int):
        super().__init__()

        hidden_size = 4 * embedded_size

        self.net = nn.Sequential(
            nn.Linear(embedded_size, hidden_size, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_size, embedded_size, bias=False),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x):
        return self.net(x)


class MambaBlock(nn.Module):
    def __init__(self, config: SambaConfig) -> None:
        super().__init__()

        self.mamba = Mamba(
            d_model=config.embedded_size,
            d_state=config.d_state,
            d_conv=config.d_conv,
            expand=2,
            use_fast_path=True,
            device=config.device,
        )

        self.feed_forward = FeedForward(0.0, config.embedded_size)
        self.layer_norm1 = nn.LayerNorm(config.embedded_size)
        self.layer_norm2 = nn.LayerNorm(config.embedded_size)

    def forward(self, x: Tensor):
        x = x + self.mamba(self.layer_norm1(x))
        x = x + self.feed_forward(self.layer_norm2(x))
        return x


class Samba(nn.Module):
    def __init__(self, config: SambaConfig):
        super().__init__()

        self.token_embeddings = nn.ModuleList([
            nn.Embedding(part.size, config.embedded_size) for part in config.vocab_parts
        ])

        self.blocks = nn.Sequential(
            *[MambaBlock(config) for _ in range(config.layer_count)])

        self.linear_norm = nn.LayerNorm(config.embedded_size)
        self.linear = nn.ModuleList([
            nn.Linear(config.embedded_size, part.size) for part in config.vocab_parts
        ])

        self.config = config
        self.to(config.device)

        self.init_weights()

    def init_weights_apply(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def init_weights(self):
        self.apply(self.init_weights_apply)

    def forward(self, x: Tensor, y: Tensor | None = None) -> tuple[list[Tensor], Tensor | None]:
        batch_size, time_step, _ = x.shape

        embedded = 0
        for i, embedder in enumerate(self.token_embeddings):
            embedded = embedded + embedder(x[:, :, i])

        norm = self.linear_norm(self.blocks(embedded))
        logits_parts = [linear(norm) for linear in self.linear]

        if y is None:
            loss = None
        else:
            loss = 0

            for part_index, (logits, part) in enumerate(
                zip(logits_parts, self.config.vocab_parts)
            ):
                batch_size, time_step, vocab_size = logits.shape

                targets = y[:, :, part_index]
                logits = logits.view(batch_size * time_step, vocab_size)
                targets = targets.reshape(batch_size * time_step)

                loss = loss + part.loss_weight * \
                    nn.functional.cross_entropy(logits, targets)

        return logits_parts, loss
    
    def inference_step(self, hidden_state: Tensor, layer: int):
        block = self.blocks[layer].forward()

    @torch.no_grad()
    def generate(self, context: Tensor, new_token_count: int, top_k: int = 5) -> Tensor:
        self.eval()

        for _ in range(new_token_count):
            logits_parts, _ = self(context, None)

            token_indices = []

            for logits in logits_parts:
                logits = logits[:, -1, :]
                logits[logits < torch.topk(logits, top_k)[
                    0][..., -1, None]] = -float('Inf')
                probs = nn.functional.softmax(logits, dim=-1)
                token_indices.append(torch.multinomial(probs, num_samples=1))

            tokens = torch.tensor(token_indices).unsqueeze(
                0).unsqueeze(0).to(context.device)
            context = torch.cat((context, tokens), dim=1)

        return context
