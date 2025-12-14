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
        
        # Store token mappings for MSE loss calculation
        self.stoi = stoi
        self.itos = {v: k for k, v in stoi.items()}
        
        # Token embeddings
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        
        # Get digit token IDs for abacus embeddings
        digit_tokens = [stoi[str(i)] for i in range(10)]
        
        # THREE positional encoding types
        self.learned_encoder = LearnedPositionalEncoding(n_embd, block_size)
        self.sinusoidal_encoder = SinusoidalPositionalEncoding(n_embd, block_size)
        self.abacus_encoder = AbacusEmbedding(n_embd, block_size, digit_tokens, max_k=99)
        
        # Learnable weights for combining the three encodings
        # Initialize to equal weights (1/3 each after softmax)
        self.weight_learned = nn.Parameter(torch.ones(1))
        self.weight_sinusoidal = nn.Parameter(torch.ones(1))
        self.weight_abacus = nn.Parameter(torch.ones(1))
        
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
        
        # Get all three positional encodings
        learned_pos = self.learned_encoder(idx)      # (B, T, C)
        sinusoidal_pos = self.sinusoidal_encoder(idx)  # (B, T, C)
        abacus_pos = self.abacus_encoder(idx)        # (B, T, C)
        
        # Compute softmax weights (ensures they sum to 1)
        weights = F.softmax(torch.stack([
            self.weight_learned, 
            self.weight_sinusoidal, 
            self.weight_abacus
        ]), dim=0)
        
        # Weighted combination of all three positional encodings
        pos_emb = (weights[0] * learned_pos + 
                   weights[1] * sinusoidal_pos + 
                   weights[2] * abacus_pos)
        
        x = tok_emb + pos_emb # (B,T,C)
        
        # Transformer processing
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)
        
        # ANSWER MASKING: Restrict output vocabulary after '='
        # Only allow digits (0-9), decimal point (.), and newline in answers
        if targets is not None:
            valid_answer_tokens = [self.stoi[str(i)] for i in range(10)]  # 0-9
            valid_answer_tokens.extend([self.stoi['.'], self.stoi['\n']])
            
            # Create mask for invalid tokens
            invalid_mask = torch.ones(self.vocab_size, dtype=torch.bool, device=self.device)
            invalid_mask[valid_answer_tokens] = False
            
            # Apply mask to answer portion (after '=')
            for b in range(B):
                target_seq = targets[b].cpu().tolist()
                if self.stoi['='] in target_seq:
                    eq_pos = target_seq.index(self.stoi['='])
                    # Zero out invalid tokens in answer portion
                    for t in range(eq_pos + 1, T):
                        logits[b, t, invalid_mask] = -1e10  # Very negative = ~0 probability

        if targets is None:
            loss = None
        else:
            # Validate targets
            if targets.max().item() >= self.vocab_size or targets.min().item() < 0:
                raise ValueError(f"Target token indices out of bounds: min={targets.min().item()}, max={targets.max().item()}")
            
            B, T, C = logits.shape
            
            # HYBRID LOSS: Full-number differentiable MSE + Cross-Entropy
            # Computes soft numerical value for entire answer, not digit-by-digit
            
            B, T, C = logits.shape
            
            # Standard cross-entropy component (for general token learning)
            ce_loss = F.cross_entropy(
                logits.view(B*T, C),
                targets.view(B*T),
                reduction='none'
            ).view(B, T)
            
            # INVALID TOKEN PENALTY: Heavily penalize non-numeric tokens in answers
            valid_answer_tokens = set([self.stoi[str(i)] for i in range(10)] + 
                                     [self.stoi['.'], self.stoi['\n']])
            invalid_token_penalty = torch.zeros(B, T, device=self.device)
            
            # Digit token setup for soft numerical computation
            digit_tokens = [self.stoi[str(i)] for i in range(10)]
            digit_values = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 
                                       dtype=torch.float32, device=self.device)
            
            weights = torch.ones(B, T, device=self.device)
            numerical_mse_loss = torch.tensor(0.0, device=self.device)
            valid_samples = 0
            
            for b in range(B):
                target_seq = targets[b].cpu().tolist()
                
                # Find '=' position
                if self.stoi['='] not in target_seq:
                    continue
                    
                eq_pos = target_seq.index(self.stoi['='])
                
                # Check for invalid tokens in answer portion
                for t in range(eq_pos + 1, T):
                    token = target_seq[t]
                    if token not in valid_answer_tokens:
                        # HUGE penalty for invalid tokens like +, -, *, /
                        invalid_token_penalty[b, t] = 100.0
                
                # Find decimal point position (if exists)
                decimal_pos = None
                for t in range(eq_pos + 1, T):
                    if target_seq[t] in self.itos and self.itos[target_seq[t]] == '.':
                        decimal_pos = t
                        break
                
                # Build SOFT predicted number and TRUE target number
                soft_pred_value = torch.tensor(0.0, device=self.device)
                true_value = 0.0
                
                # Parse digits to build full number
                digits_before_decimal = []
                digits_after_decimal = []
                
                for t in range(eq_pos + 1, T):
                    token = target_seq[t]
                    if token == self.stoi.get('\n', -1):
                        break
                    
                    if token in self.itos and self.itos[token] in '0123456789':
                        # Get soft predicted digit value (differentiable!)
                        probs = F.softmax(logits[b, t], dim=-1)
                        digit_probs = probs[digit_tokens]
                        soft_digit = (digit_probs * digit_values).sum()
                        
                        # Get true digit value
                        true_digit = float(self.itos[token])
                        
                        if decimal_pos is None or t < decimal_pos:
                            # Before decimal point
                            digits_before_decimal.append((soft_digit, true_digit))
                        else:
                            # After decimal point
                            digits_after_decimal.append((soft_digit, true_digit))
                
                # Compute full number value with place values
                # Before decimal: reverse order (rightmost = ones, next = tens, etc.)
                for i, (soft_d, true_d) in enumerate(reversed(digits_before_decimal)):
                    place_value = 10 ** i  # 1, 10, 100, ...
                    soft_pred_value = soft_pred_value + soft_d * place_value
                    true_value += true_d * place_value
                
                # After decimal: forward order (first = tenths, second = hundredths, etc.)
                for i, (soft_d, true_d) in enumerate(digits_after_decimal):
                    place_value = 10 ** (-(i + 1))  # 0.1, 0.01, 0.001, ...
                    soft_pred_value = soft_pred_value + soft_d * place_value
                    true_value += true_d * place_value
                
                # MSE on full numerical value (e.g., 9.0 vs 7.999 = ~1.0 difference)
                if len(digits_before_decimal) > 0 or len(digits_after_decimal) > 0:
                    numerical_mse_loss = numerical_mse_loss + (soft_pred_value - true_value) ** 2
                    valid_samples += 1
                
                # Also apply position weights for CE
                for t in range(eq_pos + 1, T):
                    token = target_seq[t]
                    if token in self.itos:
                        if self.itos[token] in '0123456789':
                            if decimal_pos is None or t < decimal_pos:
                                weights[b, t] = 10.0  # Before decimal
                            else:
                                weights[b, t] = 3.0   # After decimal
                        elif self.itos[token] == '.':
                            weights[b, t] = 5.0
            
            # Average MSE over valid samples
            if valid_samples > 0:
                numerical_mse_loss = numerical_mse_loss / valid_samples
            
            # Combine: weighted CE + numerical MSE + invalid token penalty
            weighted_ce = (ce_loss * weights).mean()
            invalid_penalty = invalid_token_penalty.mean()
            total_loss = weighted_ce + numerical_mse_loss * 50.0 + invalid_penalty
            
            loss = total_loss

        return logits, loss

    def generate_math_answer(self, expression, encode, decode, itos, max_new_tokens=50):
        """Generate answer for a math expression until newline"""
        expression = expression + "="
        context = torch.tensor(encode(expression), dtype=torch.long, device=self.device).unsqueeze(0)  # (1, T)
        generated = context
        for _ in range(max_new_tokens):
            logits, _ = self(generated)
            logits = logits[:, -1, :]  # (1, vocab_size)
            probs = F.softmax(logits, dim=-1)  # (1, vocab_size)
            next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)
            generated = torch.cat((generated, next_token), dim=1)  # (1, T+1)
            if itos[next_token.item()] == '\n':
                break
        answer = decode(generated[0].tolist())
        return answer
