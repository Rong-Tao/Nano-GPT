import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.elastic.multiprocessing.errors import record
import argparse
import hashlib
import os
import tiktoken

##### Initialize Distributed Training #####
def init_distributed_mode():
    dist.init_process_group(backend="nccl")
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    return global_rank, world_size

rank, world_size = init_distributed_mode()
##### Arguments #####
if rank == 0 :
    parser = argparse.ArgumentParser(description='Pass log directories to main script.') 
    parser.add_argument('--output_log', type=str, help='Path to the output log directory.')
    args = parser.parse_args()
##### Directories #####
result_dir = args.output_log if rank == 0 else None
#result_dir = '/public/home/wenyong36/project/yubian/CV/TR/__cahce/nanogpt/_2_Distributed_Shakespeare/temp' if rank == 0 else None
tensor_bd_dir = os.path.join(result_dir, 'tensorboard') if rank == 0 else None
state_dict_dir = os.path.join(result_dir, 'state_dict') if rank == 0 else None
if rank == 0:
    os.makedirs(result_dir, exist_ok=True) 
    os.makedirs(tensor_bd_dir, exist_ok=True)
    os.makedirs(state_dict_dir, exist_ok=True)

##### Tiktoken Caching #####
blobpath = "https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/vocab.bpe"
cache_key = hashlib.sha1(blobpath.encode()).hexdigest()
tiktoken_cache_dir = "../tiktoken_cache"
os.environ["TIKTOKEN_CACHE_DIR"] = tiktoken_cache_dir
assert os.path.exists(os.path.join(tiktoken_cache_dir, cache_key))
enc = tiktoken.get_encoding("gpt2")

##### Parameters #####
with open('../input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

encode = enc.encode
def decode(input_data):
    if isinstance(input_data, torch.Tensor):
        input_data = input_data.tolist()
    return enc.decode(input_data)

##### Dataset #####
data = torch.tensor(encode(text), dtype=torch.long)

class SequenceDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data.cuda()
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.block_size]
        y = self.data[idx+1:idx+self.block_size+1]
        return x, y

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]
block_size = 8
batch_size = 64
BT = block_size * batch_size

##### Model #####
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, vocab_size//10)
        self.token_embedding_table = nn.Sequential(
            nn.Linear(vocab_size//10, vocab_size//10),
            nn.ReLU(),
            nn.Linear(vocab_size//10, vocab_size)
        )

    def forward(self, idx):
        embedded = self.embedding(idx).float()
        logits = self.token_embedding_table(embedded)
        B, T, C = logits.shape
        logits = logits.view(B * T, C)
        return logits

    def generate(self, idx, max_new_tokens):
        old = idx
        for _ in range(max_new_tokens):
            logits = self(old)
            probabilities = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probabilities, num_samples=1)
            old = idx_next
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# Training
def train(rank, world_size):
    model = BigramLanguageModel(enc.n_vocab).cuda()
    model = DDP(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    train_dataset = SequenceDataset(train_data, block_size)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    best_loss = float('inf')

    criterion = torch.nn.CrossEntropyLoss()  # Make sure to define your loss function

    # Initialize SummaryWriter for rank 0
    if rank == 0:
        writer = SummaryWriter(log_dir=tensor_bd_dir)

    for epoch in range(10):
        train_loss = 0.0
        for batch_idx, (xb, yb) in enumerate(train_loader):
            logits = model(xb)
            loss = criterion(logits, yb.view(-1))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if rank == 0:
                # Log batch training loss
                writer.add_scalar('Batch Training Loss', loss.item(), epoch * len(train_loader) + batch_idx)

                if batch_idx % 100 == 0:
                    model.eval()
                    with torch.no_grad():
                        generated_sequence = model.module.generate(idx=torch.randint(0, 10000, (1, 1), dtype=torch.long).cuda(), max_new_tokens=30)[0]
                        generated_text = decode(generated_sequence.tolist())
                        writer.add_text('Generated Text', generated_text, epoch * len(train_loader) + batch_idx)
                    model.train()

        if rank == 0:
            # Log epoch training loss
            writer.add_scalar('Epoch Training Loss', train_loss / len(train_loader), epoch)

            # Save model if current loss is the best so far
            if train_loss < best_loss:
                best_loss = train_loss
                model_save_path = os.path.join(state_dict_dir, f"bigram_language_model_epoch_{epoch}.pth")
                torch.save(model.state_dict(), model_save_path)

    # Close the SummaryWriter after training is complete
    if rank == 0:
        writer.close()

@record
def main():
    train(rank, world_size)

if __name__ == "__main__":
    main()
