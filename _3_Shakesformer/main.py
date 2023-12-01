import torch
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.elastic.multiprocessing.errors import record
import os
import tiktoken, hashlib

## Import our own scrips ##

from model import Model_Class
from dataset import get_loaders
from util import arg, get_optimizer,batch_logger, epoch_logger_saver
from util import criterion, BATCH_SIZE, EPOCH_NUM, TRAIN_VAL_RATIO

## Initialize Distributed Training #####
def init_distributed_mode():
    dist.init_process_group(backend="nccl")
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    return global_rank, world_size

rank, world_size = init_distributed_mode()

if rank == 0:
    result_dir, state_dict_dir, tensor_bd_dir = arg()
    blobpath = "https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/vocab.bpe"
    cache_key = hashlib.sha1(blobpath.encode()).hexdigest()
    tiktoken_cache_dir = "../tiktoken_cache"
    os.environ["TIKTOKEN_CACHE_DIR"] = tiktoken_cache_dir
    assert os.path.exists(os.path.join(tiktoken_cache_dir, cache_key))
    enc = tiktoken.get_encoding("gpt2")


# Training
def train(rank, world_size):
    model = Model_Class.cuda()
    model = DDP(model)
    optimizer, scheduler = get_optimizer(model)
    train_loader, validation_loader = get_loaders(world_size, rank, BATCH_SIZE, TRAIN_VAL_RATIO)

    best_loss = float('inf')

    # Initialize SummaryWriter for rank 0
    if rank == 0:
        writer = SummaryWriter(log_dir=tensor_bd_dir)

    for epoch in range(EPOCH_NUM):
        model.train()
        train_loss = 0.0
        for batch_idx, (img, gt) in enumerate(train_loader):
            img, gt = img.cuda(), gt.cuda()
            out = model(img)
            loss = criterion(out, gt)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            if rank == 0:
                batch_logger(writer, batch_idx, epoch * len(train_loader) + batch_idx, loss.item())
                if batch_idx % 100 == 0:
                    initial_tokens = torch.tensor([[...]], dtype=torch.long).cuda()  # Start token(s)
                    generated_sequence = model.module.generate(initial_tokens, max_length=30)
                    generated_text = enc.decode(generated_sequence[0].cpu().numpy())
                    writer.add_text('Poem', generated_text, epoch * len(train_loader) + batch_idx)

        model.eval()
        validation_loss = 0.0
        with torch.no_grad():
            for val_batch_idx, (val_img, val_gt) in enumerate(validation_loader):
                val_img, val_gt = val_img.cuda(), val_gt.cuda()
                val_out = model(val_img)
                val_loss = criterion(val_out, val_gt)
                validation_loss += val_loss.item()

        validation_loss /= len(validation_loader)
        scheduler.step(validation_loss)
        
        if rank == 0:
            best_loss = epoch_logger_saver(model, writer, epoch, train_loss/len(train_loader), validation_loss, best_loss, state_dict_dir)

    if rank == 0:
        writer.close()

@record
def main():
    train(rank, world_size)

if __name__ == "__main__":
    main()
