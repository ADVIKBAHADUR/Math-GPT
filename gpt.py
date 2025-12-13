import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
import os
import time
from datetime import datetime

# hyperparameters - tuned for math expressions
batch_size = 32 # smaller batch for math dataset
block_size = 28 # shorter context for math expressions  
max_iters = 10000 # fewer iterations for smaller dataset
eval_interval = 50
learning_rate = 5e-4
device = 'xpu' if torch.xpu.is_available() else 'cpu'
eval_iters = 100
n_embd = 96 # smaller embedding for simpler task
n_head = 10 # fewer attention heads
n_layer = 6 # fewer layers
dropout = 0.1 # less dropout
# ------------

torch.manual_seed(1337)

# TensorBoard setup
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_dir = f'tensorboard_logs/mathgpt_all_{timestamp}'
writer = SummaryWriter(log_dir)

# Load math dataset
def load_math_dataset():
    """Load math expressions from dataset file"""
    # TODO: Implement loading from math_dataset.txt
    # Format: "expression=answer\n"
    with open('math_dataset.txt', 'r') as f:
        text = f.read()
    return text
def prepare_math_data():
    """Prepare math dataset for training"""
    # TODO: Load dataset, create character vocabulary from math expressions
    # Should include digits, operators (+,-,*,/), parentheses, equals, newline
    # So all numbers from 0-9, +, -, *, /, (, ), =, \n
    chars = sorted(['0','1','2','3','4','5','6','7','8','9','+','-','*','/','(',')','=','\n', '.'])
    return chars
# Load and prepare math dataset
text = load_math_dataset()
chars = prepare_math_data()
# here are all the unique characters that might occur in this text
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
def create_train_val_split(text, split=0.9):
    """Create training and validation splits from math dataset"""
    # TODO: Implement data splitting
    # Split by expressions, not just characters
    # to avoid partial expressions in different splits
    expressions = text.strip().split('\n')
    n = int(len(expressions) * split)
    train_expressions = expressions[:n]
    val_expressions = expressions[n:]
    train_data = torch.tensor(encode('\n'.join(train_expressions)), dtype=torch.long)
    val_data = torch.tensor(encode('\n'.join(val_expressions)), dtype=torch.long)
    return train_data, val_data

# Create splits from math data
train_data, val_data = create_train_val_split(text)

# Precompute expressions for faster batch generation
train_str = decode(train_data.tolist())
val_str = decode(val_data.tolist())
train_expressions = [expr for expr in train_str.strip().split('\n') if expr.strip()]
val_expressions = [expr for expr in val_str.strip().split('\n') if expr.strip()]

print(f"Training expressions: {len(train_expressions)}")
print(f"Validation expressions: {len(val_expressions)}")

# data loading - optimized expression boundary aware
def get_batch(split):
    """Generate batches that respect expression boundaries (optimized)"""
    expressions = train_expressions if split == 'train' else val_expressions
    
    batch_x = []
    batch_y = []
    
    # Pre-select random expressions for the entire batch
    selected_indices = torch.randint(0, len(expressions), (batch_size,))
    
    for i in range(batch_size):
        expr_idx = selected_indices[i].item()
        expression = expressions[expr_idx] + '\n'  # Add back the newline
        
        # Encode the expression
        encoded_expr = encode(expression)
        
        # Ensure minimum length for training
        if len(encoded_expr) < 2:
            # Fallback to a longer expression
            expr_idx = torch.randint(0, len(expressions), (1,)).item()
            expression = expressions[expr_idx] + '\n'
            encoded_expr = encode(expression)
        
        # Handle sequence length
        if len(encoded_expr) > block_size:
            # Expression is longer than block_size - this should be rare
            # Take the first block_size tokens (will cut off the answer)
            print(f"Warning: Expression too long ({len(encoded_expr)} > {block_size}): {expression[:50]}...")
            x = encoded_expr[:block_size]
            y = encoded_expr[1:block_size+1]
        else:
            # Normal case: expression fits within block_size
            # Pad with zeros to reach block_size
            x = encoded_expr + [0] * (block_size - len(encoded_expr))
            y = encoded_expr[1:] + [0] * (block_size - len(encoded_expr) + 1)
            x = x[:block_size]
            y = y[:block_size]
        
        batch_x.append(x)
        batch_y.append(y)
    
    x = torch.tensor(batch_x, dtype=torch.long, device=device)
    y = torch.tensor(batch_y, dtype=torch.long, device=device)
    return x, y

@torch.no_grad()
def estimate_loss():
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

@torch.no_grad()
def evaluate_math_accuracy():
    """Evaluate model accuracy on math expressions"""
    model.eval()
    test_expressions = ["1+2", "3*4", "5/6", "7-8", "9+1", "12-3", "1-57", "23/45", "67*89", "100+200"]
    correct = 0
    total = len(test_expressions)
    
    results = {}
    for expr in test_expressions:
        try:
            generated = model.generate_math_answer(expr)
            # Extract answer after '='
            if '=' in generated:
                generated_answer = generated.split('=')[1].strip().replace('\n', '')
                correct_answer = str(round(float(eval(expr)), 4))
                is_correct = generated_answer == correct_answer
                if is_correct:
                    correct += 1
                results[expr] = {
                    'generated': generated_answer,
                    'correct': correct_answer,
                    'is_correct': is_correct
                }
        except:
            results[expr] = {'generated': 'ERROR', 'correct': str(eval(expr)), 'is_correct': False}
    
    accuracy = correct / total
    model.train()
    return accuracy, results

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
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

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

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
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
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

class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # better init, not covered in the original GPT video, but important, will cover in followup video
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

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate_math_answer(self, expression, max_new_tokens=50):
        """Generate answer for a math expression until newline"""
        # TODO: Implement math-specific generation
        # 1. Take expression (e.g., "2+3*4")
        # 2. Add "=" to make "2+3*4="
        # 3. Generate tokens until newline character
        # 4. Return generated answer
        expression = expression + "="
        context = torch.tensor(encode(expression), dtype=torch.long, device=device).unsqueeze(0)  # (1, T)
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
    
    def evaluate_math_answer(self, expression, generated_answer):
        """Evaluate if the generated answer is correct"""
        # TODO: Implement evaluation logic
        # 1. Use eval() to compute correct answer
        # 2. Parse generated answer (handle potential formatting issues)
        # 3. Compare and return True/False
        answer = expression.split('=')[1].strip()
        eval_answer = float(eval(expression.split('=')[0]))
        if str(answer) == str(eval_answer):
            return True
        else:
            return False

model = GPTLanguageModel()
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Log hyperparameters
writer.add_hparams({
    'batch_size': batch_size,
    'block_size': block_size,
    'learning_rate': learning_rate,
    'n_embd': n_embd,
    'n_head': n_head,
    'n_layer': n_layer,
    'dropout': dropout,
    'vocab_size': vocab_size
}, {})

print(f"Training MathGPT - ALL Operators")
print(f"Logging to: {log_dir}")
print(f"Vocabulary: {chars}")
print(f"Dataset size: {len(train_data) + len(val_data)} tokens")

# Model saving setup
model_save_dir = f'models/mathgpt_all_{timestamp}'
os.makedirs(model_save_dir, exist_ok=True)
best_val_loss = float('inf')
best_accuracy = 0.0

start_time = time.time()

for iter in range(max_iters):
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        
        # Math accuracy evaluation
        math_accuracy, math_results = evaluate_math_accuracy()
        
        # Log overlayed metrics to TensorBoard (same graph)
        writer.add_scalars('Loss_Comparison', {
            'Train': losses['train'],
            'Validation': losses['val']
        }, iter)
        
        writer.add_scalar('Accuracy/Math_Expressions', math_accuracy, iter)
        writer.add_scalar('Learning/Learning_Rate', learning_rate, iter)
        
        # Log sample predictions as text
        sample_results_text = "\n".join([f"{expr}: {res['generated']} (correct: {res['correct']})" 
                                       for expr, res in list(math_results.items())[:5]])
        writer.add_text('Sample_Predictions', sample_results_text, iter)
        
        elapsed_time = time.time() - start_time
        writer.add_scalar('Training/Time_Elapsed', elapsed_time, iter)
        
        # Save best model based on validation loss
        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': losses['val'],
                'train_loss': losses['train'],
                'math_accuracy': math_accuracy,
                'epoch': iter,
                'vocab_size': vocab_size,
                'chars': chars,
                'stoi': stoi,
                'itos': itos
            }, f'{model_save_dir}/best_model.pt')
            print(f"  -> New best model saved (val_loss: {best_val_loss:.4f})")
        
        # Save best model based on accuracy
        if math_accuracy > best_accuracy:
            best_accuracy = math_accuracy
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': losses['val'],
                'train_loss': losses['train'],
                'math_accuracy': math_accuracy,
                'epoch': iter,
                'vocab_size': vocab_size,
                'chars': chars,
                'stoi': stoi,
                'itos': itos
            }, f'{model_save_dir}/best_accuracy_model.pt')
            print(f"  -> New best accuracy model saved (accuracy: {best_accuracy:.3f})")
        
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, math accuracy {math_accuracy:.3f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    
    # Log training loss every 50 steps
    if iter % 50 == 0:
        writer.add_scalar('Loss/Training_Step', loss.item(), iter)
    
    # Log gradient norms and training metrics overlayed
    if iter % 100 == 0:
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        
        # Log step-wise training loss for comparison with validation
        writer.add_scalars('Loss_Detailed', {
            'Training_Step': loss.item(),
            'Current_Validation': losses['val'] if iter % eval_interval == 0 else None
        }, iter)
        
        writer.add_scalar('Gradients/Total_Norm', total_norm, iter)
    
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Save final model
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'final_iter': max_iters,
    'vocab_size': vocab_size,
    'chars': chars,
    'stoi': stoi,
    'itos': itos
}, f'{model_save_dir}/final_model.pt')

print(f"\nModels saved to: {model_save_dir}")
print(f"- best_model.pt (lowest validation loss: {best_val_loss:.4f})")
print(f"- best_accuracy_model.pt (highest accuracy: {best_accuracy:.3f})")
print(f"- final_model.pt (final training state)")

# Test the trained model
def test_math_expressions():
    """Test the model on sample math expressions"""
    # Focus on addition only for this run
    test_expressions = ["1+1", "2-3", "5*7", "10/7", "2-4", "6/9", "10*10"]
    correct = 0
    total = len(test_expressions)
    
    results_for_tensorboard = {}
    
    print("\n=== Final Math Expression Test ===")
    for expr in test_expressions:
        try:
            generated = model.generate_math_answer(expr)
            if '=' in generated:
                generated_answer = generated.split('=')[1].strip().replace('\n', '')
                correct_answer = str(round(float(eval(expr)), 4))
                is_correct = generated_answer == correct_answer
                if is_correct:
                    correct += 1
                print(f"✓ {expr} = {generated_answer} (expected: {correct_answer})")
                results_for_tensorboard[expr] = is_correct
            else:
                print(f"✗ {expr} = MALFORMED ({generated.strip()})")
                results_for_tensorboard[expr] = False
        except Exception as e:
            print(f"✗ {expr} = ERROR: {str(e)}")
            results_for_tensorboard[expr] = False
    
    final_accuracy = correct / total
    print(f"\nFinal Accuracy: {correct}/{total} = {final_accuracy:.3f}")
    
    # Log final results to TensorBoard with overlayed metrics
    writer.add_scalars('Final_Results', {
        'Test_Accuracy': final_accuracy,
        'Training_Accuracy': math_accuracy if 'math_accuracy' in locals() else 0.0
    }, max_iters)
    
    # Create confusion matrix-style data overlayed
    correct_count = sum(results_for_tensorboard.values())
    writer.add_scalars('Prediction_Breakdown', {
        'Correct_Predictions': correct_count,
        'Incorrect_Predictions': total - correct_count
    }, max_iters)
    
    return final_accuracy

def interactive_math_mode():
    """Interactive mode for user input"""
    # TODO: Implement interactive mode
    # Allow user to input expressions and get AI-generated answers
    while True:
        user_input = input("Enter a math expression (or 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break
        generated = model.generate_math_answer(user_input)
        print(f"AI-generated answer: {generated.strip()}")

# After training, test the model
print("\nTesting MathGPT...")
final_accuracy = test_math_expressions()

# Log model architecture to TensorBoard (fixed tracing issue)
try:
    # Use inference mode (targets=None) to avoid loss computation during tracing
    dummy_input = torch.zeros(1, block_size, dtype=torch.long).to(device)
    model.eval()
    with torch.no_grad():
        # Trace with targets=None so only logits are returned
        writer.add_graph(model, (dummy_input, None))
    model.train()
    print("Model graph successfully logged to TensorBoard")
except Exception as e:
    print(f"Could not log model graph: {e}")

# Final summary
total_time = time.time() - start_time
writer.add_text('Training_Summary', f"""
# MathGPT Training Summary - All Operators

## Configuration
- Model Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M
- Training Time: {total_time/60:.1f} minutes
- Final Accuracy: {final_accuracy:.3f}
- Best Validation Loss: {best_val_loss:.4f}
- Best Accuracy: {best_accuracy:.3f}
- Dataset: All operators expressions

## Model Files
- Best Model: {model_save_dir}/best_model.pt
- Best Accuracy: {model_save_dir}/best_accuracy_model.pt
- Final Model: {model_save_dir}/final_model.pt

## Hyperparameters
- Batch Size: {batch_size}
- Block Size: {block_size}
- Learning Rate: {learning_rate}
- Embedding Dimension: {n_embd}
- Attention Heads: {n_head}
- Transformer Layers: {n_layer}
- Dropout: {dropout}
""", max_iters)

print(f"\nTraining completed in {total_time/60:.1f} minutes")
print(f"TensorBoard logs saved to: {log_dir}")
print(f"To view logs, run: tensorboard --logdir={log_dir}")

# Close TensorBoard writer
writer.close()

print("\nStarting interactive mode...")
interactive_math_mode()
