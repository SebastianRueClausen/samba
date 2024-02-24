import glob
import os
import random

import torch.utils.data

import vocab


class Dataset(torch.utils.data.IterableDataset):
    def __init__(self, folder: str, split: str, context_size: int):
        super().__init__()
        self.split = split
        self.context_size = context_size
        self.folder = folder

    def __iter__(self):
        search_space = os.path.join(self.folder, '**')

        files = sorted(glob.glob(os.path.join(search_space, '*.midi'), recursive=True)
            + glob.glob(os.path.join(search_space, '*.mid'), recursive=True))

        split_offset = len(files) * (0.9 if self.split == 'train' else 0.1)
        split_files = files[:int(split_offset)]

        rng = random.Random(1337)

        while True:
            rng.shuffle(split_files)

            for file in split_files:
                transpose = random.randint(-3, 3)
                tokens = vocab.midi_to_tokens(file, transpose)

                if len(tokens) < self.context_size:
                    continue

                batch_count = len(tokens) // self.context_size

                batch_indices = list(range(batch_count))
                rng.shuffle(batch_indices)

                for index in batch_indices:
                    last = len(tokens) - index * self.context_size
                    first = last - self.context_size

                    if first < 0:
                        continue

                    chunk = tokens[first:last]
                    x = chunk[:-1]
                    y = chunk[1:]

                    yield x, y


def data_loader(
    folder: str,
    split: str,
    context_size: int,
    batch_size: int,
) -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(
        dataset=Dataset(folder, split, context_size),
        batch_size=batch_size,
        pin_memory=True,
    )