from dataclasses import dataclass
from model import Samba
import dataset
from torch import optim
import torch
import tqdm

@dataclass
class TrainerConfig:
    iteration_count: int
    checkpoint_freq: int
    max_learning_rate: float
    min_learning_rate: float
    weight_decay: float
    grad_clip: float
    grad_accum_steps: int
    checkpoint_path: str
    context_size: int
    data_folder: str
    

class Trainer:
    def __init__(self, model: Samba, config: TrainerConfig):
        self.model = model
        self.model_config = model.config
        self.config = config

        self.iteration = 0

        self.optimizer = optim.AdamW(
            model.parameters(), lr=config.max_learning_rate, weight_decay=config.weight_decay)

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.iteration_count,
            eta_min=self.config.min_learning_rate,
        )

        self.scaler = torch.cuda.amp.GradScaler()

        self.autocast = torch.amp.autocast(
            device_type=model.config.device, dtype=torch.float16)

        train_data_loader = dataset.data_loader(
            folder=config.data_folder,
            split='train',
            context_size=config.context_size,
            batch_size=self.model_config.batch_size,
        )

        validation_data_loader = dataset.data_loader(
            folder=config.data_folder,
            split='validation',
            context_size=config.context_size,
            batch_size=self.model_config.batch_size,
        )

        self.train_data_iter = iter(train_data_loader)
        self.validation_data_iter = iter(validation_data_loader)

        self.best_validation_loss = float("inf")

    def train_step(self):
        self.model.train()

        x, y = next(self.train_data_iter)
        x, y = x.to(self.model_config.device), y.to(self.model_config.device)

        with self.autocast:
            _, loss = self.model(x, y)
            loss /= self.config.grad_accum_steps

        self.scaler.scale(loss).backward()

        if self.iteration % self.config.grad_accum_steps == 0:
            if self.config.grad_clip != 0.0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.grad_clip)

            self.scaler.step(self.optimizer)
            self.scheduler.step()
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

    @torch.no_grad()
    def validate_step(self) -> float:
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
                x, y = x.to(self.model_config.device), y.to(self.model_config.device)

                with self.autocast:
                    _, loss = self.model(x, y)

                losses[k] = loss.item()

            split_loss[split] = losses.mean().item()

        print(f"loss: {split_loss}")

        return split_loss['validate']

    def save_checkpoint(self):
        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "config": self.model.config.state_dict(),
            "iteration": self.iteration,
        }

        torch.save(checkpoint, self.config.checkpoint_path)

    def load_checkpoint(self, path: str | None = None, load_training_state=False):
        path = self.config.checkpoint_path if path is None else path

        try:
            checkpoint = torch.load(path)
        except Exception as ex:
            print(f"failed to load checkpoint at {path}: {ex}")
            return

        self.model.load_state_dict(checkpoint["model"])
        self.model.config.load_state_dict(checkpoint["config"])

        if load_training_state:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.scheduler.load_state_dict(checkpoint["scheduler"])
            self.iteration = checkpoint["iteration"]

    def train(self):
        bar = tqdm.tqdm(total=self.config.iteration_count)
        bar.update(self.iteration)

        while self.config.iteration_count > self.iteration:
            if self.iteration % self.config.checkpoint_freq == 0:
                loss = self.validate_step()

                if self.iteration != 0 and loss < self.best_validation_loss:
                    self.best_validation_loss = loss
                    self.save_checkpoint()

            self.train_step()
            self.iteration += 1
            bar.update(1)

        self.save_checkpoint()
        bar.close()
