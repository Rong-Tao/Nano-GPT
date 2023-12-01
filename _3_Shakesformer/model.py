import torch
import torch.nn as nn
import torch.nn.functional as F

class Model_Class(torch.nn.Module):
    def __init__(self, 
                 n_vocab = 50257, 
                 n_embed = 768, 
                 n_head = 8, 
                 max_seq_len = 32):
        super(Model_Class, self).__init__()
        self.embedding = nn.Embedding(n_vocab, n_embed)
        self.transformer = nn.Transformer(
            d_model=n_embed,
            nhead=n_head,
            num_encoder_layers=6,
            num_decoder_layers=6,
            dim_feedforward=2048,
            dropout=0.1
        )
        self.output_layer = nn.Linear(n_embed, n_vocab)

        # Initialize learnable positional embeddings
        self.positional_embeddings = nn.Parameter(torch.rand(max_seq_len, n_embed))

    def forward(self, src, tgt):
        # Apply embeddings
        src_emb = self.embedding(src)
        tgt_emb = self.embedding(tgt)

        # Add positional encodings
        src_emb += self.positional_embeddings[:src_emb.size(1), :]
        tgt_emb += self.positional_embeddings[:tgt_emb.size(1), :]

        # Pass through the Transformer
        transformer_output = self.transformer(src_emb, tgt_emb)

        # Apply the output linear layer
        return self.output_layer(transformer_output)
    
    def generate(self, initial_tokens, max_length):
        self.eval()  # Ensure the model is in evaluation mode
        generated = initial_tokens
        for _ in range(max_length - len(initial_tokens)):
            with torch.no_grad():
                logits = self.forward(generated)  # Assuming forward returns logits
                next_token_logits = logits[-1, :]
                next_token = torch.argmax(next_token_logits).unsqueeze(0)
                generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
            if next_token.item() == ...:  # End token ID (if applicable)
                break
        self.train()
        return generated