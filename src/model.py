"""
Neural network architecture for MathGPT
Contains the Transformer-based language model components
"""
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import random


class SinusoidalPositionalEncoding(nn.Module):
    """
    Fixed sinusoidal positional encoding from 'Attention is All You Need'.
    Good for understanding sequential/relative positions.
    """
    def __init__(self, n_embd, block_size):
        super().__init__()
        
        # Create sinusoidal position encodings
        position = torch.arange(block_size).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_embd, 2) * (-math.log(10000.0) / n_embd))
        
        pe = torch.zeros(block_size, n_embd)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not a parameter, but part of state_dict)
        self.register_buffer('pe', pe)
        
    def forward(self, idx):
        """
        Args:
            idx: Tensor of shape (batch, seq_len) with token IDs
        Returns:
            Positional encodings of shape (batch, seq_len, n_embd)
        """
        B, T = idx.shape
        return self.pe[:T, :].unsqueeze(0).expand(B, -1, -1)


class LearnedPositionalEncoding(nn.Module):
    """
    Standard learned positional embeddings.
    Good for task-specific patterns.
    """
    def __init__(self, n_embd, block_size):
        super().__init__()
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        
    def forward(self, idx):
        """
        Args:
            idx: Tensor of shape (batch, seq_len) with token IDs
        Returns:
            Positional encodings of shape (batch, seq_len, n_embd)
        """
        B, T = idx.shape
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  # (T,)
        pos_emb = self.position_embedding_table(pos)  # (T, n_embd)
        return pos_emb.unsqueeze(0).expand(B, -1, -1)  # (B, T, n_embd)


class AbacusEmbedding(nn.Module):
    """
    Abacus Embeddings - learned embeddings reused for each digit position.
    Numbers should be reversed for this to work correctly.
    From: Transformers Can Do Arithmetic with the Right Embeddings, McLeish et al. (2024)
    """
    def __init__(self, n_embd, block_size, digit_tokens, max_k=99):
        super().__init__()
        # Embedding table must be large enough to accommodate max_k shift
        # Use max(1024, block_size + max_k) to ensure sufficient capacity
        max_pos = max(1024, block_size + max_k + 10)  # +10 for safety margin
        self.embedding = nn.Embedding(max_pos, n_embd)
        self.register_buffer("digits", torch.tensor(digit_tokens), persistent=False)
        self.max_k = max_k
        
    def helper(self, mask, device):
        """
        Converts a binary mask of digit locations into spans of consecutive digits.
        Assigns position indices (1, 2, 3...) to each digit in a number.
        """
        mask_shape = mask.shape
        
        # Create a shifted version of the mask to detect changes from 0 to 1
        shifted_mask = torch.cat([torch.zeros((mask_shape[0], 1), device=device, dtype=mask.dtype), mask[:, :-1]], dim=1)
        starts = (shifted_mask != mask) & mask
        
        # Generate IDs for each segment of 1s, processing row-wise
        segment_ids = torch.cumsum(starts, dim=1)
        
        # Generate an index array row-wise
        index = torch.arange(mask.size(1)).repeat(mask.size(0), 1).to(device)
        
        # Reset index at the start of each segment
        reset_index = torch.zeros_like(mask).long()
        second_term = index * starts.long()
        reset_index = reset_index.scatter_add(1, segment_ids, second_term)
        
        # Calculate positions in segment (1-indexed for each digit position)
        positions = index - reset_index.gather(1, segment_ids) + 1
        
        # Ensure only values within 1-segments are non-zero
        result = positions * mask
        
        return result
        
    def forward(self, input_ids):
        """
        Args:
            input_ids: Tensor of shape (batch, seq_len) with token IDs
        Returns:
            Positional embeddings of shape (batch, seq_len, n_embd)
        """
        # Create mask for digit tokens (tokens 0-9)
        mask = torch.isin(input_ids, self.digits)
        output = self.helper(mask, input_ids.device)
        
        # During training, randomly shift position indices by k (data augmentation)
        if self.training:
            k = random.randint(0, self.max_k)
            output[output > 0] += k  # positions become k+1, k+2, k+3...
        
        return self.embedding(output)


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, n_embd, head_size, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out


class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, n_embd, num_heads, head_size, block_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embd, head_size, block_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.LeakyReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_embd, n_head, head_size, block_size, dropout)
        self.ffwd = FeedFoward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, dropout, device, stoi):
        super().__init__()
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.device = device
        self._use_answer_masking = True  # Can disable for faster training
        
        # Store token mappings for MSE loss calculation
        self.stoi = stoi
        self.itos = {v: k for k, v in stoi.items()}
        
        # Token embeddings
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        
        # Get digit token IDs for abacus embeddings
        digit_tokens = [stoi[str(i)] for i in range(10)]
        
        # Use only Abacus embeddings (best for arithmetic)
        self.abacus_encoder = AbacusEmbedding(n_embd, block_size, digit_tokens, max_k=99)
        
        # Transformer blocks
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # Weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # Validate input indices
        if idx.max().item() >= self.vocab_size or idx.min().item() < 0:
            raise ValueError(f"Input token indices out of bounds: min={idx.min().item()}, max={idx.max().item()}, vocab_size={self.vocab_size}")
        if T > self.block_size:
            raise ValueError(f"Sequence length {T} > block_size {self.block_size}")

        # Token embeddings
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        
        # Abacus positional encoding (optimized for arithmetic)
        pos_emb = self.abacus_encoder(idx)  # (B, T, C)
        
        x = tok_emb + pos_emb # (B,T,C)
        
        # Transformer processing
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)
        
        # ANSWER MASKING: Restrict output vocabulary after '=' (OPTIMIZED - vectorized)
        # Only allow digits (0-9), decimal point (.), and newline in answers
        if targets is not None and hasattr(self, '_use_answer_masking') and self._use_answer_masking:
            # Find '=' positions vectorized (avoid .cpu().tolist())
            eq_token = self.stoi['=']
            eq_mask = (targets == eq_token)  # (B, T) boolean mask
            
            # For each sequence, find position after '=' and mask those positions
            # Use cumsum to get positions after '='
            answer_positions = torch.cumsum(eq_mask, dim=1) > 0  # (B, T) - True after '='
            
            # Shift by 1 to start masking AFTER the '=' token
            answer_positions = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=self.device), 
                                         answer_positions[:, :-1]], dim=1)
            
            if answer_positions.any():
                # Create invalid token mask once if not cached
                if not hasattr(self, '_invalid_answer_mask'):
                    valid_answer_tokens = [self.stoi[str(i)] for i in range(10)]  # 0-9
                    valid_answer_tokens.extend([self.stoi['.'], self.stoi['\n'], self.stoi['-']])  # Add minus for negatives
                    invalid_mask = torch.ones(self.vocab_size, dtype=torch.bool, device=self.device)
                    invalid_mask[valid_answer_tokens] = False
                    self.register_buffer('_invalid_answer_mask', invalid_mask)
                
                # Apply mask using broadcasting: (B, T, 1) * (vocab_size,) -> (B, T, vocab_size)
                answer_mask_3d = answer_positions.unsqueeze(-1)  # (B, T, 1)
                invalid_mask_3d = self._invalid_answer_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, vocab_size)
                
                # Where we have answer positions AND invalid tokens, set to -1e10
                mask_to_apply = answer_mask_3d & invalid_mask_3d  # (B, T, vocab_size)
                logits = torch.where(mask_to_apply, torch.tensor(-1e10, device=self.device), logits)

        if targets is None:
            loss = None
        else:
            # Validate targets
            if targets.max().item() >= self.vocab_size or targets.min().item() < 0:
                raise ValueError(f"Target token indices out of bounds: min={targets.min().item()}, max={targets.max().item()}")
            
            B, T, C = logits.shape
            
            # Standard cross-entropy loss
            ce_loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T), reduction='none').view(B, T)
            
            # Weight answer tokens more heavily (OPTIMIZED - vectorized)
            weights = torch.ones(B, T, device=self.device)
            
            # Find '=' positions and weight everything after them
            eq_token = self.stoi['=']
            eq_mask = (targets == eq_token)  # (B, T)
            answer_positions = torch.cumsum(eq_mask, dim=1) > 0  # Everything after first '='
            
            # Shift to start weighting AFTER '=' token
            answer_positions = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=self.device),
                                         answer_positions[:, :-1]], dim=1)
            
            weights[answer_positions] = 3.0
            
            # Weighted cross-entropy
            loss = (ce_loss * weights).mean()

        return logits, loss

    def generate_math_answer(self, expression, encode, decode, itos, max_new_tokens=50):
        """Generate answer for a math expression until newline"""
        expression = expression + "="
        context = torch.tensor(encode(expression), dtype=torch.long, device=self.device).unsqueeze(0)  # (1, T)
        generated = context
        
        # Create answer masking for generation (only allow digits, '.', '-', '\n' after '=')
        # This prevents generating invalid tokens like +, *, /, = in the answer
        if not hasattr(self, '_generation_invalid_mask'):
            valid_answer_tokens = [self.stoi[str(i)] for i in range(10)]  # 0-9
            valid_answer_tokens.extend([self.stoi['.'], self.stoi['\n'], self.stoi['-']])
            invalid_mask = torch.ones(self.vocab_size, dtype=torch.bool, device=self.device)
            invalid_mask[valid_answer_tokens] = False
            self.register_buffer('_generation_invalid_mask', invalid_mask)
        
        for _ in range(max_new_tokens):
            logits, _ = self(generated)
            logits = logits[:, -1, :]  # (1, vocab_size)
            
            # Apply answer masking: we're always after '=' since we added it
            # Block operators (+, -, *, /, =) from being generated in answer
            logits = torch.where(self._generation_invalid_mask, torch.tensor(-1e10, device=self.device), logits)
            
            probs = F.softmax(logits, dim=-1)  # (1, vocab_size)
            next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)
            generated = torch.cat((generated, next_token), dim=1)  # (1, T+1)
            if itos[next_token.item()] == '\n':
                break
        answer = decode(generated[0].tolist())
        return answer
