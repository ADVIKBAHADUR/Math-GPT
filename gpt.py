import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
import os
import time
from datetime import datetime
import csv
import json
import random

# hyperparameters - tuned for math expressions
batch_size = 32 # smaller batch for math dataset
block_size = 28 # shorter context for math expressions  
max_iters = 100000 # fewer iterations for smaller dataset
eval_interval = 50
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
eval_iters = 100
n_embd = 102 # smaller embedding for simpler task
n_head = 8 # fewer attention heads
n_layer = 6 # fewer layers
dropout = 0.2 # less dropout
# ------------

torch.manual_seed(1337)
random.seed(1337)

def generate_test_expressions(num_digits=1, count=10):
    """Generate test expressions including mixed operations"""
    if num_digits == 1:
        min_val, max_val = 0, 9
    else:
        min_val, max_val = 10**(num_digits-1), 10**num_digits - 1
    
    test_set = {
        'addition': [],
        'subtraction': [],
        'multiplication': [],
        'division': [],
        'mixed_add_sub': [],        # Mixed addition/subtraction
        'mixed_add_sub_mult': [],   # Addition, subtraction, and multiplication
        'mixed_all': []             # All operations mixed
    }
    
    # Generate pure addition tests
    for _ in range(count):
        a = random.randint(min_val, max_val)
        b = random.randint(min_val, max_val)
        test_set['addition'].append(f"{a}+{b}")
    
    # Generate pure subtraction tests (ensure positive results)
    for _ in range(count):
        a = random.randint(min_val, max_val)
        b = random.randint(min_val, a)  # b <= a for positive result
        test_set['subtraction'].append(f"{a}-{b}")
    
    # Generate pure multiplication tests
    for _ in range(count):
        a = random.randint(min_val, max_val)
        b = random.randint(min_val, max_val)
        test_set['multiplication'].append(f"{a}*{b}")
    
    # Generate pure division tests (ensure clean division)
    for _ in range(count):
        b = random.randint(max(1, min_val), max_val)  # divisor != 0
        quotient = random.randint(1, 10 if num_digits == 1 else 5)
        a = b * quotient  # ensures clean division
        test_set['division'].append(f"{a}/{b}")
    
    # Generate mixed addition/subtraction tests
    for _ in range(count):
        a = random.randint(min_val, max_val)
        b = random.randint(min_val, max_val)
        c = random.randint(0, min(a + b, max_val))  # Ensure positive result
        test_set['mixed_add_sub'].append(f"{a}+{b}-{c}")
    
    # Generate mixed add/sub/mult tests
    for _ in range(count):
        a = random.randint(min_val, max_val)
        b = random.randint(min_val, max_val)
        c = random.randint(min_val, max_val)
        ops = random.choice([
            f"{a}+{b}*{c}",  # Tests order of operations
            f"{a}*{b}+{c}",
            f"{a}*{b}-{c}"
        ])
        test_set['mixed_add_sub_mult'].append(ops)
    
    # Generate all operations mixed
    for _ in range(count):
        a = random.randint(max(1, min_val), max_val)
        b = random.randint(max(1, min_val), max_val)
        c = random.randint(max(1, min_val), max_val)
        ops = random.choice([
            f"{a*b}/{b}+{c}",  # Ensures clean division
            f"{a}+{b*c}/{b}",
            f"{a*c}/{c}*{b}"
        ])
        test_set['mixed_all'].append(ops)
    
    return test_set

def generate_random_expression(num_digits=1, max_terms=3, max_value=10, operators=['+']):
    """Generate a random math expression on-the-fly"""
    num_terms = random.randint(2, max_terms)
    
    # Generate first number
    if num_digits == 1:
        expression = str(random.randint(0, max_value - 1))
    else:
        expression = str(random.randint(10**(num_digits-1), 10**num_digits - 1))
    
    # Add remaining terms
    for _ in range(num_terms - 1):
        op = random.choice(operators)
        if num_digits == 1:
            num = random.randint(0, max_value - 1)
        else:
            num = random.randint(10**(num_digits-1), 10**num_digits - 1)
        
        # Avoid division by zero
        if op == '/' and num == 0:
            num = random.randint(1, max_value - 1)
        
        expression += op + str(num)
    
    # Calculate answer
    try:
        answer = eval(expression)
        # Always format as float, round to 4 decimals, then strip unnecessary trailing zeros
        answer_float = round(float(answer), 4)
        # Format: remove trailing zeros after decimal point, but keep at least one decimal place if it's a whole number
        if answer_float == int(answer_float):
            answer_str = str(int(answer_float))  # "4" not "4.0"
        else:
            answer_str = f"{answer_float:.4f}".rstrip('0').rstrip('.')  # "3.5" not "3.5000"
    except:
        # If evaluation fails, generate a simpler expression
        return generate_random_expression(num_digits, 2, max_value, ['+'])
    
    return f"{expression}={answer_str}"

def create_curriculum_stages(num_digits_list=[1, 2]):
    """Create curriculum stages for different digit complexities"""
    stages = []
    
    for num_digits in num_digits_list:
        digit_name = 'single' if num_digits == 1 else f'{num_digits}digit'
        max_val = 10 ** num_digits
        
        # Stage 1: Addition only
        stages.append({
            'name': f'{digit_name}_addition',
            'operators': ['+'],
            'num_digits': num_digits,
            'max_value': max_val
        })
        
        # Stage 2: Addition + Subtraction
        stages.append({
            'name': f'{digit_name}_add_sub',
            'operators': ['+', '-'],
            'num_digits': num_digits,
            'max_value': max_val
        })
        
        # Stage 3: Addition + Subtraction + Multiplication
        stages.append({
            'name': f'{digit_name}_add_sub_mult',
            'operators': ['+', '-', '*'],
            'num_digits': num_digits,
            'max_value': max_val
        })
        
        # Stage 4: All operations
        stages.append({
            'name': f'{digit_name}_all_ops',
            'operators': ['+', '-', '*', '/'],
            'num_digits': num_digits,
            'max_value': max_val
        })
    
    return stages

# Create curriculum stages (1-digit and 2-digit)
curriculum_stages = create_curriculum_stages(num_digits_list=[1, 2])

# Generate test expressions for both 1-digit and 2-digit
test_expressions_1digit = generate_test_expressions(num_digits=1, count=10)
test_expressions_2digit = generate_test_expressions(num_digits=2, count=10)

# Curriculum Learning Configuration
curriculum_config = {
    'num_digits': curriculum_stages[0]['num_digits'],
    'max_terms': 3,
    'max_value': curriculum_stages[0]['max_value'],
    'operators': curriculum_stages[0]['operators'],
    'accuracy_threshold': 1.0,
}

current_stage = 0

print(f"\n{'='*70}")
print(f"CURRICULUM LEARNING SETUP")
print(f"{'='*70}")
print(f"Total stages: {len(curriculum_stages)}")
for i, stage in enumerate(curriculum_stages):
    print(f"  Stage {i}: {stage['name']} - {stage['operators']} ({stage['num_digits']}-digit)")
print(f"{'='*70}\n")

# Validation set management
val_expressions = []
val_set_size = 1000

def update_validation_set():
    """Generate fixed validation set for current curriculum stage"""
    global val_expressions
    val_expressions = []
    print(f"Generating fixed validation set ({val_set_size} expressions)...")
    for _ in range(val_set_size):
        expr = generate_random_expression(
            num_digits=curriculum_config['num_digits'],
            max_terms=curriculum_config['max_terms'],
            max_value=curriculum_config['max_value'],
            operators=curriculum_config['operators']
        )
        val_expressions.append(expr)
    print(f"Validation set ready for stage: {curriculum_stages[current_stage]['name']}\n")

# Initialize validation set for first stage
update_validation_set()

# TensorBoard setup
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
run_name = f'curriculum_8stages_lr{learning_rate}_emb{n_embd}_h{n_head}_l{n_layer}_{timestamp}'
log_dir = f'runs/tensorboard/{run_name}'
writer = SummaryWriter(log_dir)

def prepare_math_data():
    """Prepare vocabulary for math expressions"""
    # Define all required characters for math operations
    required_chars = set('0123456789+-*/=.\n')
    chars = sorted(list(required_chars))
    
    print(f"Vocabulary (size {len(chars)}): {chars}")
    return chars

chars = prepare_math_data()
# here are all the unique characters that might occur in this text
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

def safe_encode(s):
    """Safe encoding that validates characters and filters unknown ones"""
    result = []
    for c in s:
        if c in stoi:
            result.append(stoi[c])
        else:
            pass
            # print(f"WARNING: Unknown character '{c}' (ord: {ord(c)}) - using <PAD>")
            # result.append(stoi['<PAD>'])  # fallback to PAD token
    return result

def safe_decode(l):
    """Safe decoding that validates indices"""
    result = []
    for i in l:
        if i in itos:
            result.append(itos[i])
        else:
            print(f"WARNING: Invalid token index {i} - using space")
            result.append(' ')
    return ''.join(result)

encode = safe_encode
decode = safe_decode

print(f"Curriculum Learning Mode: Starting with {curriculum_stages[current_stage]['name']}")
print(f"Operators: {curriculum_stages[current_stage]['operators']}")

# data loading - train from random generation, val from fixed set
def get_batch(split):
    """Generate batches - train uses random expressions, val uses fixed set"""
    batch_x = []
    batch_y = []
    
    for i in range(batch_size):
        if split == 'train':
            # Training: generate random expressions on-the-fly
            expression_str = generate_random_expression(
                num_digits=curriculum_config['num_digits'],
                max_terms=curriculum_config['max_terms'],
                max_value=curriculum_config['max_value'],
                operators=curriculum_config['operators']
            )
        else:  # split == 'val'
            # Validation: sample from fixed validation set
            expr_idx = random.randint(0, len(val_expressions) - 1)
            expression_str = val_expressions[expr_idx]
        
        expression = expression_str + '\n'
        
        # Encode the expression
        encoded_expr = encode(expression)
        
        # Validate encoded tokens
        for token_idx in encoded_expr:
            if token_idx >= vocab_size or token_idx < 0:
                print(f"ERROR: Token index {token_idx} is out of bounds [0, {vocab_size-1}]")
                print(f"Expression: {expression}")
                print(f"Encoded: {encoded_expr}")
                raise ValueError(f"Invalid token index {token_idx}")
        
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
    
    # Final validation before creating tensors
    for batch_idx, (x_seq, y_seq) in enumerate(zip(batch_x, batch_y)):
        for seq_idx, token_idx in enumerate(x_seq + y_seq):
            if token_idx >= vocab_size or token_idx < 0:
                print(f"ERROR: In batch {batch_idx}, sequence position {seq_idx}, token index {token_idx} is out of bounds")
                raise ValueError(f"Invalid token index {token_idx} in batch")
    
    x = torch.tensor(batch_x, dtype=torch.long, device=device)
    y = torch.tensor(batch_y, dtype=torch.long, device=device)
    
    # Validate tensor dimensions
    if x.max().item() >= vocab_size:
        print(f"ERROR: x tensor contains index {x.max().item()} >= vocab_size {vocab_size}")
        raise ValueError("Invalid tensor indices")
    if y.max().item() >= vocab_size:
        print(f"ERROR: y tensor contains index {y.max().item()} >= vocab_size {vocab_size}")
        raise ValueError("Invalid tensor indices")
    
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
def normalize_answer(ans_str):
    """Normalize answer string for consistent comparison"""
    try:
        # Convert to float, round to 4 decimals
        val = round(float(ans_str.strip()), 4)
        # If it's a whole number, return as int string
        if val == int(val):
            return str(int(val))
        # Otherwise, format with up to 4 decimals, stripping trailing zeros
        return f"{val:.4f}".rstrip('0').rstrip('.')
    except:
        return ans_str.strip()

@torch.no_grad()
def evaluate_math_accuracy():
    """Evaluate model accuracy on math expressions with per-operation breakdown"""
    model.eval()
    
    # Select test set based on current stage's digit complexity
    current_num_digits = curriculum_stages[current_stage]['num_digits']
    if current_num_digits == 1:
        test_expressions = test_expressions_1digit
    else:
        test_expressions = test_expressions_2digit
    
    # Determine which test categories to use based on current stage operators
    stage_ops = set(curriculum_stages[current_stage]['operators'])
    
    # Map stage operators to relevant test categories
    test_categories = []
    if stage_ops == {'+'}:
        test_categories = ['addition']
    elif stage_ops == {'+', '-'}:
        test_categories = ['addition', 'subtraction', 'mixed_add_sub']
    elif stage_ops == {'+', '-', '*'}:
        test_categories = ['addition', 'subtraction', 'multiplication', 'mixed_add_sub', 'mixed_add_sub_mult']
    elif stage_ops == {'+', '-', '*', '/'}:
        test_categories = ['addition', 'subtraction', 'multiplication', 'division', 'mixed_add_sub', 'mixed_add_sub_mult', 'mixed_all']
    
    operation_results = {}
    operation_stats = {}
    all_correct = 0
    all_total = 0
    
    for category in test_categories:
        expressions = test_expressions.get(category, [])
        if not expressions:
            continue
            
        correct = 0
        total = len(expressions)
        operation_results[category] = []
        
        for expr in expressions:
            try:
                generated = model.generate_math_answer(expr)
                # Extract answer after '='
                if '=' in generated:
                    generated_answer = generated.split('=')[1].strip().replace('\n', '')
                    
                    # Calculate correct answer
                    correct_val = eval(expr)
                    correct_answer_raw = str(round(float(correct_val), 4))
                    
                    # Normalize both for comparison
                    generated_normalized = normalize_answer(generated_answer)
                    correct_normalized = normalize_answer(correct_answer_raw)
                    
                    is_correct = generated_normalized == correct_normalized
                    
                    if is_correct:
                        correct += 1
                        all_correct += 1
                    operation_results[category].append({
                        'expression': expr,
                        'generated': generated_answer,
                        'generated_normalized': generated_normalized,
                        'correct': correct_normalized,
                        'is_correct': is_correct
                    })
                else:
                    correct_val = eval(expr)
                    correct_normalized = normalize_answer(str(round(float(correct_val), 4)))
                    operation_results[category].append({
                        'expression': expr,
                        'generated': 'MALFORMED',
                        'generated_normalized': 'MALFORMED',
                        'correct': correct_normalized,
                        'is_correct': False
                    })
            except Exception as e:
                correct_val = eval(expr)
                correct_normalized = normalize_answer(str(round(float(correct_val), 4)))
                operation_results[category].append({
                    'expression': expr,
                    'generated': 'ERROR',
                    'generated_normalized': 'ERROR',
                    'correct': correct_normalized,
                    'is_correct': False,
                    'error': str(e)
                })
        
        all_total += total
        accuracy = correct / total if total > 0 else 0
        error_rate = (total - correct) / total if total > 0 else 0
        operation_stats[category] = {
            'accuracy': accuracy,
            'error_rate': error_rate,
            'correct': correct,
            'total': total
        }
    
    overall_accuracy = all_correct / all_total if all_total > 0 else 0
    
    model.train()
    return overall_accuracy, operation_stats, operation_results

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
            nn.LeakyReLU(),
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

        # Validate input indices
        if idx.max().item() >= vocab_size or idx.min().item() < 0:
            raise ValueError(f"Input token indices out of bounds: min={idx.min().item()}, max={idx.max().item()}, vocab_size={vocab_size}")
        if T > block_size:
            raise ValueError(f"Sequence length {T} > block_size {block_size}")

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
            # Validate targets
            if targets.max().item() >= vocab_size or targets.min().item() < 0:
                raise ValueError(f"Target token indices out of bounds: min={targets.min().item()}, max={targets.max().item()}")
            
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

# Model saving setup
model_save_dir = f'runs/models/{run_name}'
os.makedirs(model_save_dir, exist_ok=True)
results_dir = 'experiment_results'
os.makedirs(results_dir, exist_ok=True)
best_val_loss = float('inf')
best_accuracy = 0.0

# Initialize CSV logging for curriculum learning experiments
csv_file = f'{results_dir}/curriculum_learning_results.csv'
csv_exists = os.path.exists(csv_file)

# Save run configuration
config_file = f'{model_save_dir}/run_config.json'
with open(config_file, 'w') as f:
    json.dump({
        'run_name': run_name,
        'timestamp': timestamp,
        'curriculum_config': curriculum_config,
        'curriculum_stages': curriculum_stages,
        'hyperparameters': {
            'batch_size': batch_size,
            'block_size': block_size,
            'learning_rate': learning_rate,
            'n_embd': n_embd,
            'n_head': n_head,
            'n_layer': n_layer,
            'dropout': dropout,
            'max_iters': max_iters,
            'eval_interval': eval_interval
        },
        'vocab_size': vocab_size,
        'vocabulary': chars
    }, f, indent=2)

print(f"\n{'='*70}")
print(f"Training MathGPT - Curriculum Learning")
print(f"Run Name: {run_name}")
print(f"{'='*70}")
print(f"TensorBoard: {log_dir}")
print(f"Model Dir: {model_save_dir}")
print(f"CSV Log: {csv_file}")
print(f"Config: {config_file}")
print(f"Vocabulary: {chars}")
print(f"Starting Stage: {curriculum_stages[current_stage]['name']}")
print(f"Starting Operators: {curriculum_config['operators']}")
print(f"{'='*70}\n")

def log_experiment_to_csv(metrics_dict):
    """Log experimental results to CSV for easy comparison"""
    with open(csv_file, 'a', newline='') as f:
        writer_csv = csv.DictWriter(f, fieldnames=metrics_dict.keys())
        if not csv_exists or f.tell() == 0:
            writer_csv.writeheader()
        writer_csv.writerow(metrics_dict)

start_time = time.time()

for iter in range(max_iters):
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        
        # Math accuracy evaluation with per-operation breakdown
        math_accuracy, operation_stats, operation_results = evaluate_math_accuracy()
        
        # Log overlayed metrics to TensorBoard (same graph)
        writer.add_scalars('Loss_Comparison', {
            'Train': losses['train'],
            'Validation': losses['val']
        }, iter)
        
        writer.add_scalar('Accuracy/Overall', math_accuracy, iter)
        writer.add_scalar('Learning/Learning_Rate', learning_rate, iter)
        
        # Log per-operation accuracy (only for categories that were tested)
        accuracy_dict = {cat.replace('_', ' ').title(): stats['accuracy'] 
                        for cat, stats in operation_stats.items()}
        if accuracy_dict:
            writer.add_scalars('Accuracy/By_Category', accuracy_dict, iter)
        
        # Log per-operation error rates
        error_dict = {cat.replace('_', ' ').title(): stats['error_rate'] 
                     for cat, stats in operation_stats.items()}
        if error_dict:
            writer.add_scalars('Error_Rate/By_Category', error_dict, iter)
        
        # Log sample predictions as text for each category
        for category, results in operation_results.items():
            sample_text = "\n".join([f"{r['expression']}: {r['generated']} (correct: {r['correct']}, ✓={r['is_correct']})" 
                                    for r in results[:3]])
            writer.add_text(f'Sample_Predictions/{category.replace("_", " ").title()}', sample_text, iter)
        
        elapsed_time = time.time() - start_time
        writer.add_scalar('Training/Time_Elapsed', elapsed_time, iter)
        
        # Save best model based on validation loss
        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            torch.save({
                'run_name': run_name,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': losses['val'],
                'train_loss': losses['train'],
                'math_accuracy': math_accuracy,
                'operation_stats': operation_stats,
                'epoch': iter,
                'curriculum_stage': current_stage,
                'curriculum_stage_name': curriculum_stages[current_stage]['name'],
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
                'run_name': run_name,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': losses['val'],
                'train_loss': losses['train'],
                'math_accuracy': math_accuracy,
                'operation_stats': operation_stats,
                'epoch': iter,
                'curriculum_stage': current_stage,
                'curriculum_stage_name': curriculum_stages[current_stage]['name'],
                'vocab_size': vocab_size,
                'chars': chars,
                'stoi': stoi,
                'itos': itos
            }, f'{model_save_dir}/best_accuracy_model.pt')
            print(f"  -> New best accuracy model saved (accuracy: {best_accuracy:.3f})")
        
        # Curriculum progression logic
        stage_info = curriculum_stages[current_stage]
        stage_name = stage_info['name']
        stage_ops = stage_info['operators']
        
        # Update curriculum config to match current stage
        curriculum_config['num_digits'] = stage_info['num_digits']
        curriculum_config['max_value'] = stage_info['max_value']
        curriculum_config['operators'] = stage_ops
        
        # Check accuracy for current curriculum operations only (must be 100%)
        # Sum up all test categories that are relevant for this stage
        current_ops_correct = 0
        current_ops_total = 0
        
        for category, stats in operation_stats.items():
            current_ops_correct += stats['correct']
            current_ops_total += stats['total']
        
        current_ops_accuracy = current_ops_correct / current_ops_total if current_ops_total > 0 else 0
        
        # Display current stage requirements
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, overall acc {math_accuracy:.3f}")
        print(f"  🎯 Stage {current_stage}/{len(curriculum_stages)-1} '{stage_name}': {current_ops_correct}/{current_ops_total} correct ({current_ops_accuracy:.1%}) - Need 100%")
        
        # Progress to next stage if 100% accuracy achieved
        if current_ops_accuracy >= curriculum_config['accuracy_threshold'] and current_stage < len(curriculum_stages) - 1:
            print(f"\n{'='*60}")
            print(f"✓ CURRICULUM PROGRESSION: Stage {current_stage} '{stage_name}' mastered!")
            print(f"   Accuracy: {current_ops_correct}/{current_ops_total} = {current_ops_accuracy:.1%}")
            
            # Save checkpoint BEFORE progressing
            completed_stage_checkpoint = f'{model_save_dir}/stage_{current_stage}_{stage_name}_completed.pt'
            torch.save({
                'run_name': run_name,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'curriculum_stage': current_stage,
                'curriculum_stage_name': stage_name,
                'curriculum_config': curriculum_config.copy(),
                'operation_stats': operation_stats,
                'epoch': iter,
                'math_accuracy': math_accuracy,
                'stage_accuracy': current_ops_accuracy,
                'val_loss': losses['val'],
                'train_loss': losses['train'],
                'vocab_size': vocab_size,
                'chars': chars,
                'stoi': stoi,
                'itos': itos,
                'hyperparameters': {
                    'batch_size': batch_size,
                    'block_size': block_size,
                    'learning_rate': learning_rate,
                    'n_embd': n_embd,
                    'n_head': n_head,
                    'n_layer': n_layer,
                    'dropout': dropout
                }
            }, completed_stage_checkpoint)
            print(f"   Checkpoint saved: {completed_stage_checkpoint}")
            
            # Progress to next stage
            current_stage += 1
            next_stage_info = curriculum_stages[current_stage]
            curriculum_config['operators'] = next_stage_info['operators']
            curriculum_config['num_digits'] = next_stage_info['num_digits']
            curriculum_config['max_value'] = next_stage_info['max_value']
            print(f"   Moving to Stage {current_stage}: {next_stage_info['name']}")
            print(f"   New operators: {curriculum_config['operators']}")
            print(f"   Digit complexity: {curriculum_config['num_digits']}-digit")
            
            # Update validation set for new curriculum stage
            update_validation_set()
            print(f"{'='*60}\n")
            
            # Log curriculum progression to TensorBoard
            writer.add_text('Curriculum_Progression', f"""Stage {current_stage}: {next_stage_info['name']}
Operators: {curriculum_config['operators']}
Digits: {curriculum_config['num_digits']}
Previous stage accuracy: {current_ops_accuracy:.3f} (100%)
Iteration: {iter}""", iter)
            writer.add_scalar('Curriculum/Stage', current_stage, iter)
            writer.add_scalar('Curriculum/NumDigits', curriculum_config['num_digits'], iter)
            
            # After progression, check if we need to re-evaluate immediately
            # (don't wait for next eval_interval if we just entered final stage)
            continue  # Skip to next iteration to immediately evaluate new stage
        
        # Check if all stages completed - EARLY STOPPING
        if current_stage == len(curriculum_stages) - 1 and current_ops_accuracy >= curriculum_config['accuracy_threshold']:
            print(f"\n{'='*60}")
            print(f"🎉 ALL CURRICULUM STAGES MASTERED!")
            print(f"   Final stage {current_stage} '{stage_name}' accuracy: {current_ops_correct}/{current_ops_total} = {current_ops_accuracy:.1%}")
            print(f"   Stopping training early at iteration {iter}/{max_iters}")
            print(f"{'='*60}\n")
            
            # Save completion checkpoint
            completion_checkpoint = f'{model_save_dir}/curriculum_completed_early_iter{iter}.pt'
            torch.save({
                'run_name': run_name,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'curriculum_completed': True,
                'completion_iter': iter,
                'all_stages_mastered': True,
                'final_stage': current_stage,
                'final_stage_name': stage_name,
                'curriculum_config': curriculum_config.copy(),
                'operation_stats': operation_stats,
                'final_accuracy': current_ops_accuracy,
                'overall_accuracy': math_accuracy,
                'val_loss': losses['val'],
                'train_loss': losses['train'],
                'vocab_size': vocab_size,
                'chars': chars,
                'stoi': stoi,
                'itos': itos,
                'hyperparameters': {
                    'batch_size': batch_size,
                    'block_size': block_size,
                    'learning_rate': learning_rate,
                    'n_embd': n_embd,
                    'n_head': n_head,
                    'n_layer': n_layer,
                    'dropout': dropout
                },
                'training_time_min': (time.time() - start_time) / 60
            }, completion_checkpoint)
            print(f"Completion checkpoint saved: {completion_checkpoint}")
            
            # Log completion to TensorBoard
            writer.add_text('Training_Status', f"""ALL CURRICULUM STAGES COMPLETED!
Total iterations: {iter}
Time saved: {max_iters - iter} iterations not needed""", iter)
            
            # Break out of training loop
            break

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
    'run_name': run_name,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'final_iter': max_iters,
    'final_curriculum_stage': current_stage,
    'final_stage_name': curriculum_stages[current_stage]['name'],
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
    """Test the model on sample math expressions with comprehensive per-operation analysis"""
    print("\n=== Final Math Expression Test ===")
    
    # Use the comprehensive evaluation
    final_accuracy, operation_stats, operation_results = evaluate_math_accuracy()
    
    # Print detailed results by operation
    print("\n" + "="*60)
    print("PERFORMANCE BY OPERATION")
    print("="*60)
    
    for category, stats in operation_stats.items():
        print(f"\n{category.replace('_', ' ').upper()}:")
        print(f"  Accuracy: {stats['accuracy']:.3f} ({stats['correct']}/{stats['total']})")
        print(f"  Error Rate: {stats['error_rate']:.3f}")
        print(f"  Sample Results:")
        for result in operation_results[category][:3]:
            status = "✓" if result['is_correct'] else "✗"
            print(f"    {status} {result['expression']} = {result['generated']} (expected: {result['correct']})")
    
    print(f"\n{'='*60}")
    print(f"OVERALL ACCURACY: {final_accuracy:.3f}")
    print(f"{'='*60}\n")
    
    # Log final results to TensorBoard
    final_results_dict = {cat.replace('_', ' ').title(): stats['accuracy'] 
                         for cat, stats in operation_stats.items()}
    final_results_dict['Overall'] = final_accuracy
    writer.add_scalars('Final_Results/By_Category', final_results_dict, max_iters)
    
    # Save detailed results to JSON
    results_file = f'{model_save_dir}/detailed_results.json'
    with open(results_file, 'w') as f:
        json.dump({
            'overall_accuracy': final_accuracy,
            'operation_stats': operation_stats,
            'operation_results': operation_results,
            'hyperparameters': {
                'batch_size': batch_size,
                'block_size': block_size,
                'learning_rate': learning_rate,
                'n_embd': n_embd,
                'n_head': n_head,
                'n_layer': n_layer,
                'dropout': dropout,
                'vocab_size': vocab_size
            }
        }, f, indent=2)
    print(f"Detailed results saved to: {results_file}")
    
    return final_accuracy, operation_stats

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
final_accuracy, final_operation_stats = test_math_expressions()

# Log experiment results to CSV for curriculum learning comparison
experiment_metrics = {
    'run_name': run_name,
    'timestamp': timestamp,
    'curriculum_learning': True,
    'num_digits': curriculum_config['num_digits'],
    'max_terms': curriculum_config['max_terms'],
    'accuracy_threshold': curriculum_config['accuracy_threshold'],
    'final_stage': curriculum_stages[current_stage]['name'],
    'final_stage_num': current_stage,
    'vocab_size': vocab_size,
    'n_embd': n_embd,
    'n_head': n_head,
    'n_layer': n_layer,
    'block_size': block_size,
    'learning_rate': learning_rate,
    'batch_size': batch_size,
    'dropout': dropout,
    'max_iters': max_iters,
    'best_val_loss': best_val_loss,
    'best_train_loss': losses['train'] if 'losses' in locals() else 0,
    'overall_accuracy': final_accuracy,
}

# Add per-category accuracy and error rates dynamically
for category, stats in final_operation_stats.items():
    safe_category = category.replace('_', ' ')
    experiment_metrics[f'{safe_category}_accuracy'] = stats['accuracy']
    experiment_metrics[f'{safe_category}_error_rate'] = stats['error_rate']

experiment_metrics['model_parameters_M'] = sum(p.numel() for p in model.parameters())/1e6
experiment_metrics['training_time_min'] = (time.time() - start_time) / 60
experiment_metrics['model_dir'] = model_save_dir
experiment_metrics['tensorboard_dir'] = log_dir

log_experiment_to_csv(experiment_metrics)
print(f"\nExperiment logged to: {csv_file}")

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
- Overall Accuracy: {final_accuracy:.3f}
- Best Validation Loss: {best_val_loss:.4f}
- Best Training Accuracy: {best_accuracy:.3f}
- Dataset: All operators expressions

## Performance by Category
""" + "\n".join([f"- {cat.replace('_', ' ').title()}: {stats['accuracy']:.3f} (Error: {stats['error_rate']:.3f})" 
                  for cat, stats in final_operation_stats.items()]) + f"""

## Model Files
- Best Model: {model_save_dir}/best_model.pt
- Best Accuracy: {model_save_dir}/best_accuracy_model.pt
- Final Model: {model_save_dir}/final_model.pt
- Detailed Results: {model_save_dir}/detailed_results.json
- CSV Log: {csv_file}

## Hyperparameters (Architecture)
- Vocabulary Size: {vocab_size}
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
