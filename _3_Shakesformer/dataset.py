import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
import hashlib
import tiktoken

def get_loaders(world_size, rank, batch_size, split_ratio):
    full_dataset = Dataset_Class()
    train_size = int(split_ratio * len(full_dataset))
    validation_size = len(full_dataset) - train_size
    train_dataset, validation_dataset = random_split(full_dataset, [train_size, validation_size])

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)

    validation_sampler = DistributedSampler(validation_dataset, num_replicas=world_size, rank=rank)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, sampler=validation_sampler)

    return train_loader, validation_loader

class Dataset_Class(Dataset):
    def __init__(self, text_file_path = "../input.txt", block_size = 32):
        self.block_size = block_size
        self.data = self.load_and_encode(text_file_path)

    def load_and_encode(self, file_path):
        # TikToken encoding setup
        blobpath = "https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/vocab.bpe"
        cache_key = hashlib.sha1(blobpath.encode()).hexdigest()
        tiktoken_cache_dir = "../tiktoken_cache"
        os.environ["TIKTOKEN_CACHE_DIR"] = tiktoken_cache_dir
        assert os.path.exists(os.path.join(tiktoken_cache_dir, cache_key))
        enc = tiktoken.get_encoding("gpt2")

        # Load and encode the text
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        encoded_text = enc.encode(text)
        return torch.tensor(encoded_text, dtype=torch.long)

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.block_size]
        y = self.data[idx + 1:idx + self.block_size + 1]
        return x, y
