import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 16
block_size = 256
max_iters = 10000
eval_interval = 1000
learning_rate = 3e-4
device = torch.device("cpu")
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
eval_iters = 50
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

torch.manual_seed(1337)

# Global variables for data and encoders (will be initialized in main)
text = None
chars = None
vocab_size = None
stoi = None
itos = None
encode = None
decode = None
train_data = None
val_data = None

def get_batch(split):
    """Generate a batch of data"""
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def save_model(model, optimizer, filepath, iteration=0):
    """Save model and optimizer state"""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'stoi': stoi,
        'itos': itos,
        'vocab_size': vocab_size,
        'iteration': iteration,
    }
    torch.save(checkpoint, filepath)
    print(f"Model saved to {filepath}")

def load_model(filepath, device):
    """Load model and optimizer state"""
    checkpoint = torch.load(filepath, map_location=device)
    return checkpoint

class Head(nn.Module):
    """One head of self-attention"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """A simple linear layer followed by a non-linearity"""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """Transformer block: communication followed by computation"""

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

def train(checkpoint_path=None):
    """Train the model, optionally resuming from a checkpoint"""
    print("Using:", device)
    
    start_iter = 0
    checkpoint_file = Path(checkpoint_path) if checkpoint_path else Path('model_checkpoint.pt')
    
    # Load checkpoint if provided and exists
    if checkpoint_path and checkpoint_file.exists():
        print(f"Loading checkpoint from {checkpoint_file}...")
        checkpoint = load_model(checkpoint_file, device)
        
        # Restore vocab mappings
        global stoi, itos, vocab_size
        stoi = checkpoint['stoi']
        itos = checkpoint['itos']
        vocab_size = checkpoint['vocab_size']
        
        global encode, decode
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda l: ''.join([itos[i] for i in l])
        
        # Create model and load weights
        model = BigramLanguageModel()
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        
        # Restore optimizer state
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Restore iteration number
        start_iter = checkpoint.get('iteration', 0) + 1
        print(f"Resuming training from iteration {start_iter}")
    else:
        if checkpoint_path:
            print(f"Warning: Checkpoint {checkpoint_path} not found. Starting fresh training.")
        
        model = BigramLanguageModel()
        model = model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    print(f"{sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")

    for iter in range(start_iter, max_iters):
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss(model)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            
            model.eval()
            with torch.no_grad():
                context = torch.zeros((1, 1), dtype=torch.long, device=device)
                sample = decode(model.generate(context, max_new_tokens=200)[0].tolist())
            model.train()
            print(f"\nSample generation:\n{sample}\n")
            
            save_model(model, optimizer, checkpoint_file, iter)

        xb, yb = get_batch('train')
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    save_model(model, optimizer, checkpoint_file, max_iters - 1)
    print("Training complete!")

def generate(checkpoint_path='model_checkpoint.pt', max_new_tokens=2000, output_file='generated_text.txt'):
    """Generate text from a trained model"""
    print("Using:", device)
    
    if not Path(checkpoint_path).exists():
        print(f"Error: Checkpoint file {checkpoint_path} not found. Please train the model first.")
        return
    
    print(f"Loading model from {checkpoint_path}...")
    checkpoint = load_model(checkpoint_path, device)
    
    global stoi, itos, vocab_size
    stoi = checkpoint['stoi']
    itos = checkpoint['itos']
    vocab_size = checkpoint['vocab_size']
    
    global encode, decode
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
    
    model = BigramLanguageModel()
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded: {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")
    
    print(f"Generating {max_new_tokens} tokens...")
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated = model.generate(context, max_new_tokens=max_new_tokens)
    generated_text = decode(generated[0].tolist())
    
    output_path = Path(output_file)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(generated_text)
    
    print(f"Generated text saved to {output_path}")
    print(f"\nFirst 500 characters:\n{generated_text[:500]}...")

def main():
    parser = argparse.ArgumentParser(description='Character-level GPT - Train or Generate')
    parser.add_argument('mode', choices=['train', 'generate'], 
                       help='Mode: train the model or generate text')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to model checkpoint (for resuming training or generating)')
    parser.add_argument('--tokens', type=int, default=2000,
                       help='Number of tokens to generate (for generate mode)')
    parser.add_argument('--output', type=str, default='generated_text.txt',
                       help='Output file for generated text (for generate mode)')
    
    args = parser.parse_args()
    
    # Initialize data (needed for both train and generate)
    global text, chars, vocab_size, stoi, itos, encode, decode, train_data, val_data
    
    data_path = Path('./mobydick.txt')
    if not data_path.exists():
        print(f"Error: Data file {data_path} not found!")
        return
    
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Create vocabulary from unique characters
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
    
    # Train and validation splits
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9*len(data))
    train_data = data[:n]
    val_data = data[n:]
    
    if args.mode == 'train':
        train(args.checkpoint)
    elif args.mode == 'generate':
        checkpoint = args.checkpoint if args.checkpoint else 'model_checkpoint.pt'
        generate(checkpoint, args.tokens, args.output)

if __name__ == '__main__':
    main()
