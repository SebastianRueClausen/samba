from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.backends.mps
from torch import optim, Tensor
from torch.utils.data import Dataset

import vocab
import dataset


@dataclass
class ModelConfig:
    vocab_size: int
    dropout_chance: float
    context_size: int
    embedded_size: int
    batch_size: int
    head_count: int
    layer_count: int
    device: str


def select_device() -> str:
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    return device


class Head(nn.Module):
    """
        A single attention head, a direct implementation of Scaled dot-product
        attention from (Ashish Vaswani et. al. 2017).
    """

    def __init__(self, config: ModelConfig, head_size: int):
        super().__init__()

        self.key = nn.Linear(config.embedded_size, head_size, bias=False)
        self.query = nn.Linear(config.embedded_size, head_size, bias=False)
        self.value = nn.Linear(config.embedded_size, head_size, bias=False)

        self.dropout = nn.Dropout(config.dropout_chance)
        self.dropout_chance = config.dropout_chance

        self.use_flash_attention = hasattr(nn.functional, 'scaled_dot_product_attention')

        if not self.use_flash_attention:
            self.register_buffer('tril', torch.tril(torch.ones(config.context_size, config.context_size)))

    def forward_flash(self, x: Tensor) -> Tensor:
        return nn.functional.scaled_dot_product_attention(
            query=self.query(x),
            key=self.key(x),
            value=self.value(x),
            attn_mask=None,
            dropout_p=self.dropout_chance if self.training else 0.0,
            is_causal=True,
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.use_flash_attention:
            return self.forward_flash(x)

        batch_size, time_step, _ = x.shape

        # Query represents something the head is looking for in tokens, with
        # respect to other tokens. Each token is looking for something, a query,
        # which key represents the answers to for each token.
        key, query = self.key(x), self.query(x)

        # Multiply queries and keys to get the attention matrix, which can the
        # thought of as how relevant each token is to each other token, with
        # respect to what the head is looking for.
        attention = query @ key.transpose(-2, -1)

        # Scale the attention matrix to avoid exploding attention scores.
        attention *= key.shape[-1] ** -0.5

        # Mask to ignore attention score to tokens ahead of the time step.
        attention = attention.masked_fill(
            self.tril[:time_step, :time_step] == 0, float('-inf'),
        )

        # Normalize with softmax.
        attention = nn.functional.softmax(attention, dim=-1)

        attention = self.dropout(attention)

        # Multiply the value vector with the attention matrix. The purpose of
        # the value vector is to map the sentence to the features the head is
        # looking for.
        return attention @ self.value(x)


class MultiHead(nn.Module):
    """ A collection of attention heads. """

    def __init__(self, config: ModelConfig, head_size: int):
        super().__init__()

        self.projection = nn.Linear(head_size * config.head_count, config.embedded_size)
        self.dropout = nn.Dropout(config.dropout_chance)
        self.heads = nn.ModuleList([Head(config, head_size) for _ in range(config.head_count)])

    def forward(self, x: Tensor) -> Tensor:
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.projection(out))
        return out


class FeedForward(nn.Module):
    def __init__(self, dropout_chance: float, embedded_size: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(embedded_size, 4 * embedded_size, bias=False),
            nn.ReLU(),
            nn.Linear(4 * embedded_size, embedded_size, bias=False),
            nn.Dropout(dropout_chance),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        head_size = config.embedded_size // config.head_count

        self.self_attention = MultiHead(config, head_size)
        self.feed_forward = FeedForward(config.dropout_chance, config.embedded_size)
        self.layer_norm1 = nn.LayerNorm(config.embedded_size)
        self.layer_norm2 = nn.LayerNorm(config.embedded_size)

    def forward(self, x: Tensor):
        x = x + self.self_attention(self.layer_norm1(x))
        x = x + self.feed_forward(self.layer_norm2(x))
        return x


class PianoGpt(nn.Module):
    """ An almost exact implementation of a transformer from (Ashish Vaswani et. al. 2017). """

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.token_embedding = nn.Embedding(config.vocab_size, config.embedded_size)
        self.position_embedding = nn.Embedding(config.context_size, config.embedded_size)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.layer_count)])

        self.linear_norm = nn.LayerNorm(config.embedded_size)
        self.linear = nn.Linear(config.embedded_size, config.vocab_size)

        self.config = config
        self.to(config.device)

        self.init_weights()

    def init_weights_apply(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def init_weights(self):
        self.apply(self.init_weights_apply)

    def forward(self, x: Tensor, y: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        batch_size, time_step = x.shape

        embedded_tokens = self.token_embedding(x)
        embedded_positions = self.position_embedding(torch.arange(time_step, device=self.config.device))
        embedded = embedded_tokens + embedded_positions

        logits = self.linear(self.linear_norm(self.blocks(embedded)))

        if y is None:
            loss = None
        else:
            batch_size, time_step, context_size = logits.shape

            logits = logits.view(batch_size * time_step, context_size)
            targets = y.view(batch_size * time_step)

            loss = nn.functional.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, context: Tensor, new_token_count: int) -> Tensor:
        for _ in range(new_token_count):
            input = context[:, -self.config.context_size:]
            logits, loss = self(input, None)
            logits = logits[:, -1, :]
            probs = nn.functional.softmax(logits, dim=-1)
            next = torch.multinomial(probs, num_samples=1)
            context = torch.cat((context, next), dim=1)
        return context


class Trainer:
    def __init__(self, model: PianoGpt, data_folder: str):
        self.model = model
        self.config = model.config

        self.optimizer = optim.AdamW(model.parameters(), lr=4e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=5)

        train_data_loader = dataset.data_loader(
            folder=data_folder,
            split='train',
            context_size=self.config.context_size,
            batch_size=self.config.batch_size,
        )

        validation_data_loader = dataset.data_loader(
            folder=data_folder,
            split='validation',
            context_size=self.config.context_size,
            batch_size=self.config.batch_size,
        )

        self.train_data_iter = iter(train_data_loader)
        self.validation_data_iter = iter(validation_data_loader)

    def train_step(self):
        self.model.train()

        x, y = next(self.train_data_iter)
        x, y = x.to(self.config.device), y.to(self.config.device)

        logits, loss = model(x, y)
        self.optimizer.zero_grad(set_to_none=True)

        loss.backward()

        self.optimizer.step()
        self.scheduler.step(loss)

    @torch.no_grad()
    def validate_step(self):
        self.model.eval()

        iterations = 50
        split_loss = {}

        for split, data_iter in [
            ('train', self.train_data_iter),
            ('validate', self.validation_data_iter),
        ]:
            losses = torch.zeros(iterations)

            for k in range(iterations):
                x, y = next(data_iter)
                x, y = x.to(self.config.device), y.to(self.config.device)

                logits, loss = self.model(x, y)
                losses[k] = loss.item()

                split_loss[split] = losses.mean()

        print(f"loss: {split_loss}")

    def train(self):
        iteration = 0

        while True:
            if iteration % 100 == 0 and iteration != 0:
                self.validate_step()
            self.train_step()

            print(f'itertaion: {iteration}')
            iteration += 1


model = PianoGpt(ModelConfig(
    vocab_size = vocab.VOCAB_SIZE,
    dropout_chance = 0.1,
    context_size = 512 // 2,
    embedded_size = 768 // 2,
    head_count = 4,
    layer_count = 4,
    batch_size = 64,
    device = select_device(),
))

trainer = Trainer(model, data_folder='maestro-v3.0.0')
trainer.train()
