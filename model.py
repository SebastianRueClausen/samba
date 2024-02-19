from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.backends.mps
from torch import optim, Tensor
import tqdm

#from mamba_ssm import Mamba

import simple_vocab
import dataset

@dataclass
class ModelConfig:
    vocab_sizes: list[int]
    dropout_rate: float
    context_size: int
    embedded_size: int
    batch_size: int
    head_count: int
    layer_count: int
    device: str

    def state_dict(self) -> dict:
        return vars(self)

    def load_state_dict(self, dict: dict):
        for k,v in dict.items():
            setattr(self, k, v)


def select_device() -> str:
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    return device


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

"""
class MambaBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        self.mamba = Mamba(
            d_model=config.embedded_size,
            d_state=16,
            d_conv=4,
            expand=2,
            use_fast_path=True,
            device=config.device,
        )

        self.feed_forward = FeedForward(0.0, config.embedded_size)
        self.layer_norm1 = nn.LayerNorm(config.embedded_size)
        self.layer_norm2 = nn.LayerNorm(config.embedded_size)


    def forward(self, x):
        print(x.shape)
        x = x + self.mamba(self.layer_norm1(x))
        x = x + self.feed_forward(self.layer_norm2(x))
        return x
"""

class Head(nn.Module):
    def __init__(self, config: ModelConfig, head_size: int):
        super().__init__()

        self.key = nn.Linear(config.embedded_size, head_size, bias=False)
        self.query = nn.Linear(config.embedded_size, head_size, bias=False)
        self.value = nn.Linear(config.embedded_size, head_size, bias=False)

        self.dropout = nn.Dropout(config.dropout_rate)
        self.dropout_rate = config.dropout_rate

        self.use_flash_attention = hasattr(nn.functional, 'scaled_dot_product_attention')

        if not self.use_flash_attention:
            self.register_buffer('tril', torch.tril(torch.ones(config.context_size, config.context_size)))

    def forward_flash(self, x: Tensor) -> Tensor:
        return nn.functional.scaled_dot_product_attention(
            query=self.query(x),
            key=self.key(x),
            value=self.value(x),
            attn_mask=None,
            dropout_p=self.dropout_rate if self.training else 0.0,
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
        self.dropout = nn.Dropout(config.dropout_rate)
        self.heads = nn.ModuleList([Head(config, head_size) for _ in range(config.head_count)])

    def forward(self, x: Tensor) -> Tensor:
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.projection(out))
        return out



class Block(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        head_size = config.embedded_size // config.head_count

        self.self_attention = MultiHead(config, head_size)
        self.feed_forward = FeedForward(config.dropout_rate, config.embedded_size)
        self.layer_norm1 = nn.LayerNorm(config.embedded_size)
        self.layer_norm2 = nn.LayerNorm(config.embedded_size)

    def forward(self, x: Tensor):
        x = x + self.self_attention(self.layer_norm1(x))
        x = x + self.feed_forward(self.layer_norm2(x))
        return x


class Samba(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.position_embedding = nn.Embedding(config.context_size, config.embedded_size)
        self.token_embeddings = nn.ModuleList([
            nn.Embedding(size, config.embedded_size) for size in config.vocab_sizes
        ])

        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.layer_count)])

        self.linear_norm = nn.LayerNorm(config.embedded_size)
        self.linear = nn.ModuleList([
            nn.Linear(config.embedded_size, size) for size in config.vocab_sizes
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

        embedded = self.position_embedding(torch.arange(time_step, device=self.config.device))
        for i, embedder in enumerate(self.token_embeddings):
            embedded = embedded + embedder(x[:, :, i])

        norm = self.linear_norm(self.blocks(embedded))
        logits = [linear(norm) for linear in self.linear]

        if y is None:
            loss = None
        else:
            loss = 0

            for vocab_index, logit in enumerate(logits):
                batch_size, time_step, vocab_size = logit.shape

                targets = y[:, :, vocab_index]
                logit = logit.view(batch_size * time_step, vocab_size)
                targets = targets.reshape(batch_size * time_step)

                loss = loss + nn.functional.cross_entropy(logit, targets)

        return logits, loss

    @torch.no_grad()
    def generate(self, context: Tensor, new_token_count: int) -> Tensor:
        self.eval()

        for _ in range(new_token_count):
            input = context[:, -self.config.context_size:]

            logits, _ = self(input, None)

            token_indices = []

            for logit in logits:
                logit = logit[:, -1, :]
                logit[logit < torch.topk(logit, 5)[0][..., -1, None]] = -float('Inf')
                probs = nn.functional.softmax(logit, dim=-1)
                token_indices.append(torch.multinomial(probs, num_samples=1))

            tokens = torch.tensor(token_indices).unsqueeze(0).unsqueeze(0).to(context.device)
            context = torch.cat((context, tokens), dim=1)

        return context


class Trainer:
    def __init__(
        self,
        model: Samba,
        data_folder: str,
        checkpoint_path: str = "checkpoint.pth",
        checkpoint_freq: int = 500,
        iteration_count: int = 16000,
    ):
        self.model = model
        self.config = model.config

        self.iteration = 0
        self.iteration_count = iteration_count

        self.max_learning_rate = 1e-4
        self.min_learning_rate = 1e-8
        self.weight_decay = 1e-1
        self.grad_clip = 1.0

        self.optimizer = optim.AdamW(model.parameters(), lr=self.max_learning_rate, weight_decay=self.weight_decay)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=iteration_count,
            eta_min=self.min_learning_rate,
        );

        self.scaler = torch.cuda.amp.GradScaler()

        self.grad_accum_steps = 8

        self.checkpoint_path = checkpoint_path
        self.checkpoint_freq = checkpoint_freq

        self.autocast = torch.amp.autocast(device_type=model.config.device, dtype=torch.float16)

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

        with self.autocast:
            _, loss = model(x, y)
            loss /= self.grad_accum_steps

        self.scaler.scale(loss).backward()

        if self.iteration % self.grad_accum_steps == 0:
            if self.grad_clip != 0.0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.scaler.step(self.optimizer)
            self.scheduler.step()
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)


    @torch.no_grad()
    def validate_step(self):
        self.model.eval()

        iterations = 100
        split_loss = {}

        for split, data_iter in [
            ('train', self.train_data_iter),
            ('validate', self.validation_data_iter),
        ]:
            losses = torch.zeros(iterations)

            for k in range(iterations):
                x, y = next(data_iter)
                x, y = x.to(self.config.device), y.to(self.config.device)

                with self.autocast:
                    _, loss = self.model(x, y)

                losses[k] = loss.item()

            split_loss[split] = losses.mean()

        print(f"loss: {split_loss}")

    def save_checkpoint(self):
        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "config": self.model.config.state_dict(),
            "iteration": self.iteration,
        }

        torch.save(checkpoint, self.checkpoint_path)

    def load_checkpoint(self):
        try:
            checkpoint = torch.load(self.checkpoint_path)
        except Exception as ex:
            print(f"failed to load checkpoint: {ex}")
            return

        self.model.load_state_dict(checkpoint["model"])
        self.model.config.load_state_dict(checkpoint["config"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.iteration = checkpoint["iteration"]

    def train(self):
        bar = tqdm.tqdm(total=self.iteration_count)
        bar.update(self.iteration)

        while self.iteration_count > self.iteration:
            if self.iteration % self.checkpoint_freq == 0:
                self.validate_step()

                if self.iteration != 0:
                    self.save_checkpoint()

            self.train_step()
            self.iteration += 1
            bar.update(1)

        self.save_checkpoint()
        bar.close()


if __name__ == "__main__":
    torch.manual_seed(1337)

    train = True

    model = Samba(ModelConfig(
        vocab_sizes = [
            simple_vocab.PITCH_COUNT,
            simple_vocab.VELOCITY_COUNT,
            simple_vocab.ADVANCE_COUNT,
            simple_vocab.DURATION_COUNT,
        ],
        dropout_rate = 0.1,
        context_size = 512 // 2,
        embedded_size = 768 // 2,
        head_count = 6,
        layer_count = 6,
        batch_size = 32,
        device = select_device(),
    ))

    model = torch.compile(model)
    trainer = Trainer(model, data_folder='maestro-v3.0.0')
    #trainer.load_checkpoint()

    if train:
        trainer.train()

        for index in range(8):
            context = torch.zeros((1, 1), dtype=torch.long, device=model.config.device)
            tokens = model.generate(context, new_token_count=1024*2)
            tokens = tokens.squeeze().cpu().numpy()
            simple_vocab.tokens_to_midi(f"samples/sample_{index}.midi", tokens)
    else:
        for index in range(8):
            context = torch.zeros((1, 1), dtype=torch.long, device=model.config.device)
            tokens = model.generate(context, new_token_count=1024 * 4)
            tokens = tokens.squeeze().cpu().numpy()

            simple_vocab.tokens_to_midi(f"samples/sample_{index + 24}.midi", tokens)
            simple_vocab.tokens_to_midi("out.midi", tokens)

