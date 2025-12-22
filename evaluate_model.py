import torch
import torch.nn as nn
from torch.nn import functional as F
import os
import glob
import sys
from datetime import datetime

# Import model from src directory
sys.path.insert(0, 'src')
from model import GPTLanguageModel
from data import convert_division_to_multiplication

# Legacy Head class for compatibility (not used with new models)
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


def generate_math_answer(model, expression, encode, decode, itos, device, max_new_tokens=50):
    """Generate answer for a math expression until newline"""
    expression = expression + "="
    context = torch.tensor(encode(expression), dtype=torch.long, device=device).unsqueeze(0)
    generated = context
    
    for _ in range(max_new_tokens):
        # Crop to block_size if needed
        idx_cond = generated[:, -model.block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated = torch.cat((generated, next_token), dim=1)
        if itos[next_token.item()] == '\n':
            break
    
    answer = decode(generated[0].tolist())
    return answer


def load_model(model_path):
    """Load a saved model and return model, vocab info"""
    print(f"Loading model from: {model_path}")
    
    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
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
    
    # Get block size - try multiple sources (different model versions)
    if 'block_size' in checkpoint:
        # Directly saved in checkpoint (preferred)
        block_size = checkpoint['block_size']
    elif 'position_embedding_table.weight' in state_dict:
        # Old architecture with learned positional embeddings
        block_size = state_dict['position_embedding_table.weight'].shape[0]
    else:
        # New architecture with Abacus embeddings - infer from tril buffer size
        # Find any tril buffer in the state dict
        tril_key = next((k for k in state_dict.keys() if 'tril' in k), None)
        if tril_key:
            block_size = state_dict[tril_key].shape[0]
            print(f"  ℹ️  Inferred block_size from {tril_key}: {block_size}")
        else:
            # Last resort - default to 128
            block_size = 128
            print(f"  ⚠️  Could not infer block_size, using default: {block_size}")
    
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
    
    # Create model with inferred hyperparameters (new architecture requires stoi)
    model = GPTLanguageModel(vocab_size, n_embd, n_head, n_layer, block_size, dropout, device, stoi)
    
    # Load state dict with strict=False to ignore cached attributes like _invalid_answer_mask
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    if unexpected_keys:
        # Filter out expected cached attributes
        unexpected_keys = [k for k in unexpected_keys if not k.startswith('_')]
        if unexpected_keys:
            print(f"  ⚠️  Unexpected keys in checkpoint: {unexpected_keys}")
    
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

def evaluate_expressions(model, encode, decode, itos, device, expressions, use_reciprocal=False, verbose=True):
    """Evaluate model on a list of expressions"""
    if verbose:
        print(f"\n=== Evaluating {len(expressions)} expressions ===")
        if use_reciprocal:
            print("(Division will be converted to multiplication with reciprocals)")
    
    correct = 0
    results = []
    
    for expr in expressions:
        try:
            # Convert division if needed
            test_expr = convert_division_to_multiplication(expr) if (use_reciprocal and '/' in expr) else expr
            
            # Generate answer
            generated = generate_math_answer(model, test_expr, encode, decode, itos, device)
            
            # Extract generated answer
            if '=' in generated:
                generated_answer = generated.split('=')[1].strip().replace('\n', '')
                # Compute expected answer from the converted expression (for reciprocals)
                expected_value = float(eval(test_expr))
                correct_answer = f"{expected_value:.4f}"
                
                # Normalize both answers for comparison (use 4 decimals like training)
                try:
                    gen_val = float(generated_answer)
                    is_correct = abs(gen_val - expected_value) < 0.01
                except ValueError:
                    is_correct = False
                
                if is_correct:
                    correct += 1
                    if verbose:
                        print(f"✓ {expr} = {generated_answer}")
                else:
                    if verbose:
                        print(f"✗ {expr} = {generated_answer} (expected: {correct_answer})")
                
                results.append({
                    'expression': expr,
                    'generated': generated_answer,
                    'correct': correct_answer,
                    'is_correct': is_correct
                })
            else:
                expected_value = float(eval(test_expr))
                if verbose:
                    print(f"✗ {expr} = MALFORMED: {generated.strip()}")
                results.append({
                    'expression': expr,
                    'generated': 'MALFORMED',
                    'correct': f"{expected_value:.4f}",
                    'is_correct': False
                })
                
        except Exception as e:
            try:
                expected_value = float(eval(test_expr))
                correct_val = f"{expected_value:.4f}"
            except:
                correct_val = 'N/A'
            if verbose:
                print(f"✗ {expr} = ERROR: {str(e)}")
            results.append({
                'expression': expr,
                'generated': 'ERROR',
                'correct': correct_val,
                'is_correct': False
            })
    
    accuracy = correct / len(expressions) if len(expressions) > 0 else 0.0
    if verbose:
        print(f"\nAccuracy: {correct}/{len(expressions)} = {accuracy:.3f}")
    
    return accuracy, results

def generate_comprehensive_test_set(num_digits=1, samples_per_category=100):
    """Generate comprehensive test sets for each category"""
    import random
    
    test_sets = {}
    max_val = 9 if num_digits == 1 else 99
    min_val = 1 if num_digits == 1 else 10
    
    # 1. Addition only
    addition_exprs = []
    for _ in range(samples_per_category):
        a = random.randint(min_val, max_val)
        b = random.randint(min_val, max_val)
        addition_exprs.append(f"{a}+{b}")
    test_sets['addition'] = addition_exprs
    
    # 2. Subtraction only
    subtraction_exprs = []
    for _ in range(samples_per_category):
        a = random.randint(min_val, max_val)
        b = random.randint(min_val, min(a, max_val))  # Ensure positive result
        subtraction_exprs.append(f"{a}-{b}")
    test_sets['subtraction'] = subtraction_exprs
    
    # 3. Multiplication only
    multiplication_exprs = []
    for _ in range(samples_per_category):
        a = random.randint(min_val, max_val)
        b = random.randint(min_val, max_val)
        multiplication_exprs.append(f"{a}*{b}")
    test_sets['multiplication'] = multiplication_exprs
    
    # 4. Division only - generate valid division expressions
    division_exprs = []
    for _ in range(samples_per_category):
        # Generate divisor first
        b = random.randint(max(2, min_val), max_val)
        # Generate dividend as multiple of divisor (for cleaner division)
        multiplier = random.randint(1, max(1, max_val // b))
        a = b * multiplier
        division_exprs.append(f"{a}/{b}")
    test_sets['division'] = division_exprs
    
    # 5. Addition + Subtraction mixed
    add_sub_exprs = []
    for _ in range(samples_per_category):
        a = random.randint(min_val, max_val)
        b = random.randint(min_val, max_val)
        c = random.randint(min_val, max_val)
        op1 = random.choice(['+', '-'])
        op2 = random.choice(['+', '-'])
        add_sub_exprs.append(f"{a}{op1}{b}{op2}{c}")
    test_sets['add_sub_mixed'] = add_sub_exprs
    
    # 6. All operators
    all_ops_exprs = []
    for _ in range(samples_per_category):
        a = random.randint(min_val, max_val)
        b = random.randint(min_val, max_val)
        c = random.randint(max(2, min_val), max_val)  # Avoid division by zero and 1
        op1 = random.choice(['+', '-', '*', '/'])
        op2 = random.choice(['+', '-', '*', '/'])
        all_ops_exprs.append(f"{a}{op1}{b}{op2}{c}")
    test_sets['all_operators'] = all_ops_exprs
    
    # 7. Decimal cases (division that results in decimals)
    decimal_exprs = []
    attempts = 0
    while len(decimal_exprs) < samples_per_category and attempts < samples_per_category * 3:
        # Generate divisions that produce decimal results
        divisors = [3, 6, 7, 9, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
        b = random.choice(divisors)
        a = random.randint(min_val, max_val)
        if a % b != 0:  # Ensure it's not clean division
            decimal_exprs.append(f"{a}/{b}")
        attempts += 1
    
    # Fill remaining with random divisions if needed
    while len(decimal_exprs) < samples_per_category:
        b = random.randint(max(2, min_val), max_val)
        a = random.randint(min_val, max_val)
        if a % b != 0:
            decimal_exprs.append(f"{a}/{b}")
    
    test_sets['decimal'] = decimal_exprs
    
    return test_sets


def run_comprehensive_evaluation(model, encode, decode, itos, device, use_reciprocal=False):
    """Run comprehensive evaluation and generate LaTeX table"""
    print("\n" + "="*80)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("="*80)
    
    # Generate test sets for 1-digit and 2-digit
    print("\nGenerating test sets...")
    test_sets_1digit = generate_comprehensive_test_set(num_digits=1, samples_per_category=100)
    test_sets_2digit = generate_comprehensive_test_set(num_digits=2, samples_per_category=100)
    
    # Verify test sets were generated correctly
    print("\nTest set sizes (1-digit):")
    for key, exprs in test_sets_1digit.items():
        print(f"  {key}: {len(exprs)} expressions")
    print("\nTest set sizes (2-digit):")
    for key, exprs in test_sets_2digit.items():
        print(f"  {key}: {len(exprs)} expressions")
    
    # Show sample expressions
    print("\nSample 2-digit expressions:")
    for key in ['addition', 'multiplication', 'division']:
        if key in test_sets_2digit:
            print(f"  {key}: {test_sets_2digit[key][:3]}")
    
    results_table = {}
    
    categories = [
        ('addition', 'Addition only'),
        ('subtraction', 'Subtraction only'),
        ('multiplication', 'Multiplication only'),
        ('division', 'Division only'),
        ('add_sub_mixed', 'Addition + Subtraction'),
        ('all_operators', 'All operators'),
        ('decimal', 'Decimal cases')
    ]
    
    # Evaluate each category
    for key, label in categories:
        print(f"\n{'='*80}")
        print(f"Evaluating: {label}")
        print(f"{'='*80}")
        
        # 1-digit evaluation
        print(f"\n--- 1-digit expressions ---")
        exprs_1d = test_sets_1digit[key]
        print(f"Testing {len(exprs_1d)} expressions. First 3: {exprs_1d[:3]}")
        acc_1d, results_1d = evaluate_expressions(model, encode, decode, itos, device, exprs_1d, use_reciprocal, verbose=False)
        print(f"1-digit accuracy: {acc_1d*100:.1f}% ({int(acc_1d*len(exprs_1d))}/{len(exprs_1d)})")
        
        # Show first few results for debugging
        if acc_1d < 0.5:  # If accuracy is low, show examples
            print("  Sample results:")
            for i, (expr, result) in enumerate(zip(exprs_1d[:5], results_1d[:5])):
                status = "✓" if result['is_correct'] else "✗"
                print(f"    {status} {expr} -> {result['generated']} (expected: {result['correct']})")
        
        # 2-digit evaluation
        print(f"\n--- 2-digit expressions ---")
        exprs_2d = test_sets_2digit[key]
        print(f"Testing {len(exprs_2d)} expressions. First 3: {exprs_2d[:3]}")
        acc_2d, results_2d = evaluate_expressions(model, encode, decode, itos, device, exprs_2d, use_reciprocal, verbose=False)
        print(f"2-digit accuracy: {acc_2d*100:.1f}% ({int(acc_2d*len(exprs_2d))}/{len(exprs_2d)})")
        
        # Show first few results for debugging - especially important for 2-digit
        if acc_2d < 0.5:  # If accuracy is low, show examples
            print("  Sample results:")
            for i, (expr, result) in enumerate(zip(exprs_2d[:10], results_2d[:10])):
                status = "✓" if result['is_correct'] else "✗"
                print(f"    {status} {expr} -> {result['generated']} (expected: {result['correct']})")
        
        results_table[label] = {
            '1digit': acc_1d * 100,
            '2digit': acc_2d * 100
        }
    
    # Generate LaTeX table
    print("\n" + "="*80)
    print("RESULTS TABLE (LaTeX format)")
    print("="*80)
    
    latex_table = r"""
\begin{table}[H]
\centering
\renewcommand{\arraystretch}{1.2}
\begin{tabular}{>{\raggedright}p{5cm} c c}
\toprule
\textbf{Category} & \textbf{1-digit Accuracy (\%)} & \textbf{2-digit Accuracy (\%)} \\
\midrule
"""
    
    for key, label in categories:
        acc_1d = results_table[label]['1digit']
        acc_2d = results_table[label]['2digit']
        latex_table += f"{label} & {acc_1d:.1f} & {acc_2d:.1f} \\\\\n"
    
    latex_table += r"""\bottomrule
\end{tabular}
\caption{Testbench categories and evaluation metrics for MathGPT. Each category tested with 100 samples.}
\label{tab:testbench}
\end{table}
"""
    
    print(latex_table)
    
    # Also print summary table in plain text
    print("\n" + "="*80)
    print("SUMMARY TABLE (Plain Text)")
    print("="*80)
    print(f"{'Category':<30} {'1-digit Acc (%)':<20} {'2-digit Acc (%)':<20}")
    print("-" * 70)
    for key, label in categories:
        acc_1d = results_table[label]['1digit']
        acc_2d = results_table[label]['2digit']
        print(f"{label:<30} {acc_1d:>18.1f} {acc_2d:>18.1f}")
    print("="*80)
    
    # Calculate and print overall statistics
    avg_1digit = sum(results_table[label]['1digit'] for _, label in categories) / len(categories)
    avg_2digit = sum(results_table[label]['2digit'] for _, label in categories) / len(categories)
    print(f"\nOverall Average Accuracy:")
    print(f"  1-digit: {avg_1digit:.1f}%")
    print(f"  2-digit: {avg_2digit:.1f}%")
    print(f"  Combined: {(avg_1digit + avg_2digit) / 2:.1f}%")
    
    return results_table, latex_table


def interactive_mode(model, encode, decode, itos, device, use_reciprocal=False):
    """Interactive mode for user input"""
    print(f"\n{'='*70}")
    print("=== Interactive Mode ===")
    print("Enter math expressions to test the model (or 'exit' to quit)")
    if use_reciprocal:
        print("⚠️  Note: Division is converted to multiplication with reciprocals")
    print(f"{'='*70}\n")
    
    while True:
        user_input = input("Enter a math expression (or 'exit' to quit): ").strip()
        if user_input.lower() in ['exit', 'quit', 'q']:
            break
        
        if user_input:
            try:
                # Convert division if using reciprocals
                test_expr = user_input
                if use_reciprocal and '/' in user_input:
                    test_expr = convert_division_to_multiplication(user_input)
                    print(f"  [Converted to: {test_expr}]")
                
                generated = generate_math_answer(model, test_expr, encode, decode, itos, device)
                print(f"AI-generated answer: {generated.strip()}")
                
                # Show expected answer for comparison
                try:
                    expected = eval(test_expr)
                    print(f"Expected: {user_input}={expected:.4f}\n")
                except:
                    print()
                    
            except Exception as e:
                print(f"Error: {e}")

def main():
    print("=== MathGPT Model Evaluator ===")
    
    # Find available models in multiple directories
    model_dirs = []
    for search_path in [
        "models/mathgpt_*",
        "models/reciprocal_*",
        "runs/models/*"
    ]:
        model_dirs.extend(glob.glob(search_path))
    
    # Remove duplicates and sort
    model_dirs = sorted(list(set(model_dirs)))
    
    if not model_dirs:
        print("No models found! Searched in:")
        print("  - models/mathgpt_*")
        print("  - models/reciprocal_*")
        print("  - runs/models/*")
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
    
    # Choose specific model file - scan for ALL .pt files
    available_files = []
    for filepath in sorted(glob.glob(os.path.join(selected_dir, '*.pt'))):
        filename = os.path.basename(filepath)
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
    
    # Check if model uses reciprocal conversion (for division stages)
    use_reciprocal = input("\nDoes this model use reciprocal conversion for division? (y/n, default=y): ").strip().lower()
    use_reciprocal = use_reciprocal != 'n'  # Default to yes
    
    # Test expressions
    test_sets = {
        "Basic Addition": ["1+1", "2+3", "5+7", "12+34"],
        "Subtraction": ["5-2", "10-7", "15-8", "100-25"],
        "Multiplication": ["2*3", "4*5", "7*8", "12*3"],
        "Division": ["6/2", "8/4", "15/3", "20/5"],
        "Mixed Operations": ["2+3*4", "10-2*3", "15/3+2", "8*2-5"],
        "Complex": ["(2+3)*4", "10/(2+3)", "2*3+4*5", "100-50/2"]
    }
    
    print("\n=== Evaluation Options ===")
    print("1. Comprehensive Evaluation (Generate LaTeX table)")
    print("2. Test all expression sets (basic)")
    print("3. Test specific expression set")
    print("4. Interactive mode")
    print("5. Custom expression list")
    
    try:
        eval_choice = int(input("Select option (number): "))
    except ValueError:
        eval_choice = 1
    
    if eval_choice == 1:
        # Comprehensive evaluation with LaTeX table
        results_table, latex_table = run_comprehensive_evaluation(model, encode, decode, itos, device, use_reciprocal)
        
        # Save LaTeX table to file
        output_file = os.path.join(os.path.dirname(selected_file), 'evaluation_results.tex')
        with open(output_file, 'w') as f:
            f.write(latex_table)
        print(f"\n✓ LaTeX table saved to: {output_file}")
        
    elif eval_choice == 2:
        # Test all sets
        overall_correct = 0
        overall_total = 0
        
        for set_name, expressions in test_sets.items():
            print(f"\n--- Testing {set_name} ---")
            accuracy, results = evaluate_expressions(model, encode, decode, itos, device, expressions, use_reciprocal)
            overall_correct += sum(1 for r in results if r['is_correct'])
            overall_total += len(results)
        
        print(f"\n=== OVERALL RESULTS ===")
        print(f"Total Accuracy: {overall_correct}/{overall_total} = {overall_correct/overall_total:.3f}")
        
    elif eval_choice == 3:
        # Test specific set
        print("\nAvailable test sets:")
        set_names = list(test_sets.keys())
        for i, name in enumerate(set_names):
            print(f"{i+1}. {name}")
        
        try:
            set_choice = int(input("Select test set (number): ")) - 1
            selected_set = set_names[set_choice]
            expressions = test_sets[selected_set]
            evaluate_expressions(model, encode, decode, itos, device, expressions, use_reciprocal)
        except (ValueError, IndexError):
            print("Invalid choice!")
            
    elif eval_choice == 4:
        # Interactive mode
        interactive_mode(model, encode, decode, itos, device, use_reciprocal)
        
    elif eval_choice == 5:
        # Custom expressions
        print("Enter expressions separated by commas:")
        custom_input = input("Expressions: ")
        expressions = [expr.strip() for expr in custom_input.split(',') if expr.strip()]
        if expressions:
            evaluate_expressions(model, encode, decode, itos, device, expressions, use_reciprocal)
        else:
            print("No valid expressions provided!")

if __name__ == "__main__":
    main()