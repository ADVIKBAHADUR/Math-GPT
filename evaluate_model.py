import torch
import torch.nn as nn
from torch.nn import functional as F
import os
import glob
from datetime import datetime

# Model architecture classes (same as training script)
class Head(nn.Module):
    """ one head of self-attention """
    def __init__(self, head_size, n_embd, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """
    def __init__(self, num_heads, head_size, n_embd, dropout, block_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embd, block_size, dropout) for _ in range(num_heads)])
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
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """
    def __init__(self, n_embd, n_head, dropout, block_size):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, dropout, block_size)
        self.ffwd = FeedFoward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embd, block_size, n_head, n_layer, dropout, device):
        super().__init__()
        self.block_size = block_size
        self.device = device
        
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, dropout, block_size) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

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
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device))
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

    def generate_math_answer(self, expression, encode, decode, itos, max_new_tokens=50):
        """Generate answer for a math expression until newline"""
        expression = expression + "="
        context = torch.tensor(encode(expression), dtype=torch.long, device=self.device).unsqueeze(0)
        generated = context
        
        for _ in range(max_new_tokens):
            # Crop to block_size if needed
            idx_cond = generated[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat((generated, next_token), dim=1)
            if itos[next_token.item()] == '\\n':
                break
        
        answer = decode(generated[0].tolist())
        return answer

def load_model(model_path):
    """Load a saved model and return model, vocab info"""
    print(f"Loading model from: {model_path}")
    
    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extract model parameters
    vocab_size = checkpoint['vocab_size']
    chars = checkpoint['chars']
    stoi = checkpoint['stoi']
    itos = checkpoint['itos']
    
    # Create encode/decode functions
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
    
    # Infer model hyperparameters from saved state_dict
    state_dict = checkpoint['model_state_dict']
    
    # Get embedding dimension from token embedding table
    n_embd = state_dict['token_embedding_table.weight'].shape[1]
    
    # Get block size from position embedding table
    block_size = state_dict['position_embedding_table.weight'].shape[0]
    
    # Get number of heads by counting attention head modules
    n_head = len([k for k in state_dict.keys() if 'blocks.0.sa.heads.' in k and '.key.weight' in k])
    
    # Get number of layers by counting blocks
    n_layer = len([k for k in state_dict.keys() if k.startswith('blocks.') and k.endswith('.ln1.weight')])
    
    # Default values for hyperparameters not saved
    dropout = 0.1  # This won't affect inference
    
    print(f"Inferred hyperparameters:")
    print(f"- n_embd: {n_embd}")
    print(f"- block_size: {block_size}")
    print(f"- n_head: {n_head}")
    print(f"- n_layer: {n_layer}")
    print(f"- dropout: {dropout}")
    
    # Create model with inferred hyperparameters
    model = GPTLanguageModel(vocab_size, n_embd, block_size, n_head, n_layer, dropout, device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Print model info
    print(f"Model loaded successfully!")
    print(f"- Vocabulary size: {vocab_size}")
    print(f"- Characters: {chars}")
    print(f"- Device: {device}")
    print(f"- Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    if 'math_accuracy' in checkpoint:
        print(f"- Training accuracy: {checkpoint['math_accuracy']:.3f}")
    if 'val_loss' in checkpoint:
        print(f"- Validation loss: {checkpoint['val_loss']:.4f}")
    
    return model, encode, decode, itos, device

def evaluate_expressions(model, encode, decode, itos, expressions):
    """Evaluate model on a list of expressions"""
    print(f"\\n=== Evaluating {len(expressions)} expressions ===")
    
    correct = 0
    results = []
    
    for expr in expressions:
        try:
            # Generate answer
            generated = model.generate_math_answer(expr, encode, decode, itos)
            
            # Extract generated answer
            if '=' in generated:
                generated_answer = generated.split('=')[1].strip().replace('\\n', '')
                correct_answer = str(float(eval(expr)))
                is_correct = generated_answer == correct_answer
                
                if is_correct:
                    correct += 1
                    print(f"✓ {expr} = {generated_answer}")
                else:
                    print(f"✗ {expr} = {generated_answer} (expected: {correct_answer})")
                
                results.append({
                    'expression': expr,
                    'generated': generated_answer,
                    'correct': correct_answer,
                    'is_correct': is_correct
                })
            else:
                print(f"✗ {expr} = MALFORMED: {generated.strip()}")
                results.append({
                    'expression': expr,
                    'generated': 'MALFORMED',
                    'correct': str(float(eval(expr))),
                    'is_correct': False
                })
                
        except Exception as e:
            print(f"✗ {expr} = ERROR: {str(e)}")
            results.append({
                'expression': expr,
                'generated': 'ERROR',
                'correct': str(float(eval(expr))),
                'is_correct': False
            })
    
    accuracy = correct / len(expressions)
    print(f"\\nAccuracy: {correct}/{len(expressions)} = {accuracy:.3f}")
    
    return accuracy, results

def interactive_mode(model, encode, decode, itos):
    """Interactive mode for user input"""
    print("\\n=== Interactive Mode ===")
    print("Enter math expressions to test the model (or 'exit' to quit)")
    
    while True:
        user_input = input("\\nExpression: ").strip()
        if user_input.lower() in ['exit', 'quit', 'q']:
            break
        
        if user_input:
            try:
                generated = model.generate_math_answer(user_input, encode, decode, itos)
                print(f"AI Answer: {generated.strip()}")
                
                # Show correct answer for comparison
                try:
                    correct_answer = eval(user_input)
                    print(f"Correct Answer: {user_input}={correct_answer}")
                except:
                    print("Could not compute correct answer")
                    
            except Exception as e:
                print(f"Error: {e}")

def main():
    print("=== MathGPT Model Evaluator ===")
    
    # Find available models
    model_dirs = glob.glob("models/mathgpt_*")
    if not model_dirs:
        print("No models found in 'models/' directory!")
        return
    
    print("\\nAvailable models:")
    for i, model_dir in enumerate(model_dirs):
        print(f"{i+1}. {model_dir}")
        # List model files in each directory
        model_files = []
        if os.path.exists(os.path.join(model_dir, 'best_model.pt')):
            model_files.append('best_model.pt')
        if os.path.exists(os.path.join(model_dir, 'best_accuracy_model.pt')):
            model_files.append('best_accuracy_model.pt')
        if os.path.exists(os.path.join(model_dir, 'final_model.pt')):
            model_files.append('final_model.pt')
        print(f"   Files: {', '.join(model_files)}")
    
    # Get user choice
    try:
        choice = int(input("\\nSelect model directory (number): ")) - 1
        selected_dir = model_dirs[choice]
    except (ValueError, IndexError):
        print("Invalid choice!")
        return
    
    # Choose specific model file
    available_files = []
    for filename in ['best_model.pt', 'best_accuracy_model.pt', 'final_model.pt']:
        filepath = os.path.join(selected_dir, filename)
        if os.path.exists(filepath):
            available_files.append((filename, filepath))
    
    if not available_files:
        print("No model files found in selected directory!")
        return
    
    print(f"\\nAvailable model files in {selected_dir}:")
    for i, (filename, _) in enumerate(available_files):
        print(f"{i+1}. {filename}")
    
    try:
        file_choice = int(input("Select model file (number): ")) - 1
        selected_file = available_files[file_choice][1]
    except (ValueError, IndexError):
        print("Invalid choice!")
        return
    
    # Load model
    model, encode, decode, itos, device = load_model(selected_file)
    
    # Test expressions
    test_sets = {
        "Basic Addition": ["1+1", "2+3", "5+7", "12+34"],
        "Subtraction": ["5-2", "10-7", "15-8", "100-25"],
        "Multiplication": ["2*3", "4*5", "7*8", "12*3"],
        "Division": ["6/2", "8/4", "15/3", "20/5"],
        "Mixed Operations": ["2+3*4", "10-2*3", "15/3+2", "8*2-5"],
        "Complex": ["(2+3)*4", "10/(2+3)", "2*3+4*5", "100-50/2"]
    }
    
    print("\\n=== Evaluation Options ===")
    print("1. Test all expression sets")
    print("2. Test specific expression set")
    print("3. Interactive mode")
    print("4. Custom expression list")
    
    try:
        eval_choice = int(input("Select option (number): "))
    except ValueError:
        eval_choice = 1
    
    if eval_choice == 1:
        # Test all sets
        overall_correct = 0
        overall_total = 0
        
        for set_name, expressions in test_sets.items():
            print(f"\\n--- Testing {set_name} ---")
            accuracy, results = evaluate_expressions(model, encode, decode, itos, expressions)
            overall_correct += sum(1 for r in results if r['is_correct'])
            overall_total += len(results)
        
        print(f"\\n=== OVERALL RESULTS ===")
        print(f"Total Accuracy: {overall_correct}/{overall_total} = {overall_correct/overall_total:.3f}")
        
    elif eval_choice == 2:
        # Test specific set
        print("\\nAvailable test sets:")
        set_names = list(test_sets.keys())
        for i, name in enumerate(set_names):
            print(f"{i+1}. {name}")
        
        try:
            set_choice = int(input("Select test set (number): ")) - 1
            selected_set = set_names[set_choice]
            expressions = test_sets[selected_set]
            evaluate_expressions(model, encode, decode, itos, expressions)
        except (ValueError, IndexError):
            print("Invalid choice!")
            
    elif eval_choice == 3:
        # Interactive mode
        interactive_mode(model, encode, decode, itos)
        
    elif eval_choice == 4:
        # Custom expressions
        print("Enter expressions separated by commas:")
        custom_input = input("Expressions: ")
        expressions = [expr.strip() for expr in custom_input.split(',') if expr.strip()]
        if expressions:
            evaluate_expressions(model, encode, decode, itos, expressions)
        else:
            print("No valid expressions provided!")

if __name__ == "__main__":
    main()