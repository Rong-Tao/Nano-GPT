import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
import argparse
import hashlib
import os
import time
################################################################
#                       Arguments                              #
################################################################
parser = argparse.ArgumentParser(description='Pass log directories to main script.')
parser.add_argument('--dmesg_log', type=str, help='Path to the dmesg log directory.')
parser.add_argument('--debug_log', type=str, help='Path to the debug log directory.')
parser.add_argument('--output_log', type=str, help='Path to the output log directory.')
args = parser.parse_args()
result_dir = args.output_log
################################################################
#                       Tiktoken Caching                       #
################################################################
blobpath = "https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/vocab.bpe"
cache_key = hashlib.sha1(blobpath.encode()).hexdigest()
tiktoken_cache_dir = "./tiktoken_cache"
os.environ["TIKTOKEN_CACHE_DIR"] = tiktoken_cache_dir
assert os.path.exists(os.path.join(tiktoken_cache_dir, cache_key))

enc = tiktoken.get_encoding("gpt2")
################################################################
#                       Parameters                             #
################################################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

encode = enc.encode
def decode(input_data):
    # Check if input is a PyTorch tensor
    if isinstance(input_data, torch.Tensor):
        input_data = input_data.tolist()
        print(input_data)
    return enc.decode(input_data)
################################################################
#                       Dataset                                #
################################################################
data = torch.tensor(encode(text), dtype=torch.long)

class SequenceDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data.to(device)
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
# Create datasets

train_dataset = SequenceDataset(train_data, block_size)
val_dataset = SequenceDataset(val_data, block_size)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
################################################################
#                       Model                                  #
################################################################
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
        embedded = self.embedding(idx).float()  # Embed and convert to float
        logits = self.token_embedding_table(embedded)
        B, T, C = logits.shape
        logits = logits.view(B * T, C)
        return logits

    def generate(self, idx, max_new_tokens):
        old = idx
        for _ in range(max_new_tokens):
            logits = self(old)
            probabilities = F.softmax(logits, dim=-1)  # apply softmax here for sampling
            idx_next = torch.multinomial(probabilities, num_samples=1)  # (B, 1)
            old = idx_next
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx

model = BigramLanguageModel(enc.n_vocab)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()
model = model.to(device)
################################################################
#                       Training                               #
################################################################
tensor_bd_dir = os.path.join(result_dir, 'tensorboard')
state_dict_dir = os.path.join(result_dir, 'state_dict')
os.makedirs(result_dir, exist_ok=True) 
os.makedirs(tensor_bd_dir, exist_ok=True)
os.makedirs(state_dict_dir, exist_ok=True)

writer = SummaryWriter(log_dir=tensor_bd_dir)

best_loss = float('inf')

Start_time = time.time()
for epoch in range(10):
    train_loss = 0.0
    for batch_idx, (xb, yb) in enumerate(train_loader):
        #xb, yb = xb.to(device), yb.to(device)  # Move data to GPU

        # Forward pass
        logits = model(xb)
        loss = criterion(logits, yb.view(BT))

        # Backward pass and optimization
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        # Log loss for each batch
        writer.add_scalar('Batch Training Loss', loss.item(), epoch * len(train_loader) + batch_idx)
        cumulative_time = time.time() - Start_time
        writer.add_scalar('Batch Training Loss vs. Time', loss.item(), cumulative_time)
        if batch_idx % 500 == 0:
            model.eval()
            with torch.no_grad():  # No gradients required for generation
                generated_sequence = model.generate(idx = torch.randint(0, 10000, (1, 1), dtype=torch.long, device=device), max_new_tokens=30)[0]
                generated_text = decode(generated_sequence.tolist())
            writer.add_text('Generated Text', generated_text, epoch * len(train_loader) + batch_idx)
            model.train() 
        # Save model if current loss is the best so far
        if loss.item() < best_loss:
            best_loss = loss.item()
            model_save_path = os.path.join(state_dict_dir, f"bigram_language_model_epoch_{epoch}_batch_{batch_idx}.pth")
            torch.save(model.state_dict(), model_save_path)

    # Log epoch-level training loss
    writer.add_scalar('Epoch Training Loss', train_loss / len(train_loader), epoch)
    
writer.close()
print("Training complete and model saved.")