"""
Training script for division-only MathGPT module
Trains a model specifically on division operations
"""
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import os
import time
from datetime import datetime
import json
import random

from model import GPTLanguageModel
from data import prepare_math_data, safe_encode, safe_decode


# hyperparameters - tuned for division learning
batch_size = 32
block_size = 28
max_iters = 50000
eval_interval = 50
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
eval_iters = 100
n_embd = 96
n_head = 8
n_layer = 6
dropout = 0.2

torch.manual_seed(1337)
random.seed(1337)

# Prepare vocabulary - EXCLUDE '=' from generation (we add it ourselves)
# Model should only generate: digits 0-9, /, ., and \n
generation_chars = sorted(list(set('0123456789/+-*.\n')))
full_chars = sorted(list(set('0123456789/+-*=.\n')))  # For encoding full expressions

print(f"Generation vocabulary (what model can output): {generation_chars}")
print(f"Full vocabulary (for encoding): {full_chars}")

# Create mappings using full vocabulary for encoding
vocab_size = len(full_chars)
stoi = {ch: i for i, ch in enumerate(full_chars)}
itos = {i: ch for i, ch in enumerate(full_chars)}

encode = lambda s: safe_encode(s, stoi)
decode = lambda l: safe_decode(l, itos)

# Map for generation only (exclude '=')
generation_vocab_size = len(generation_chars)
generation_stoi = {ch: i for i, ch in enumerate(generation_chars)}
generation_itos = {i: ch for i, ch in enumerate(generation_chars)}


def generate_division_expression(num_digits=2, max_terms=4, ensure_clean_division=True):
    """Generate a random division expression with 2-4 terms"""
    num_terms = random.randint(2, max_terms)
    
    if num_digits == 1:
        min_val, max_val = 1, 9  # Avoid division by zero
    else:
        min_val, max_val = 10**(num_digits-1), 10**num_digits - 1
    
    # Start with a dividend
    if ensure_clean_division:
        # Build expression that ensures clean division
        # Start with a result and multiply by divisors
        result = random.randint(min_val, max_val)
        expression = str(result)
        
        for i in range(num_terms - 1):
            divisor = random.randint(max(2, min_val), min(max_val, 20))  # Keep divisors reasonable
            expression += f"/{divisor}"
            result = result / divisor
        
        # Rebuild expression to ensure it evaluates correctly
        # Let's use a different approach: build from divisors
        divisor1 = random.randint(max(2, min_val), max_val)
        quotient = random.randint(1, max_val // divisor1 if divisor1 > 0 else 10)
        expression = f"{divisor1 * quotient}/{divisor1}"
        
        # Add more division terms
        for i in range(num_terms - 2):
            divisor = random.randint(2, 10)  # Keep subsequent divisors small
            expression += f"/{divisor}"
    else:
        # Generate without ensuring clean division
        expression = str(random.randint(min_val, max_val))
        for i in range(num_terms - 1):
            divisor = random.randint(max(2, min_val), max_val)
            expression += f"/{divisor}"
    
    # Calculate answer - ALWAYS format as decimal with up to 4 digits
    try:
        answer = eval(expression)
        answer_float = round(float(answer), 4)
        
        # Always format as float with appropriate precision
        # If it's a whole number, show .0
        if answer_float == int(answer_float) and answer_float < 1000:
            answer_str = f"{int(answer_float)}.0"
        else:
            # Format with up to 4 decimal places, removing trailing zeros after 1 decimal
            answer_str = f"{answer_float:.4f}".rstrip('0')
            # Ensure at least one decimal place
            if '.' not in answer_str:
                answer_str += '.0'
            elif answer_str.endswith('.'):
                answer_str += '0'
    except:
        # Fallback to simpler expression
        return generate_division_expression(num_digits, 2, ensure_clean_division)
    
    return f"{expression}={answer_str}"


def create_division_validation_set(count=30, num_digits=2, max_terms=4):
    """Create a fixed validation set of division expressions"""
    validation_set = []
    print(f"\n{'='*70}")
    print(f"GENERATING {count} DIVISION VALIDATION EXPRESSIONS")
    print(f"{'='*70}\n")
    
    for i in range(count):
        expr = generate_division_expression(num_digits, max_terms, ensure_clean_division=False)
        validation_set.append(expr)
    
    # Print ALL validation expressions before training
    print("Complete validation set (what the model will be tested on):\n")
    for i, expr in enumerate(validation_set, 1):
        print(f"  {i:2d}. {expr}")
    
    print(f"\n{'='*70}")
    print(f"Validation set ready: {len(validation_set)} expressions")
    print(f"{'='*70}\n")
    return validation_set


def get_batch_division(split, batch_size, block_size, val_expressions, encode, vocab_size, device, num_digits=2, max_terms=4):
    """Generate batches - train uses random division expressions, val uses fixed set"""
    batch_x = []
    batch_y = []
    
    for i in range(batch_size):
        if split == 'train':
            # Generate random division expression on-the-fly
            expression_str = generate_division_expression(num_digits, max_terms, ensure_clean_division=True)
        else:  # split == 'val'
            # Sample from fixed validation set
            expr_idx = random.randint(0, len(val_expressions) - 1)
            expression_str = val_expressions[expr_idx]
        
        expression = expression_str + '\n'
        encoded_expr = encode(expression)
        
        # Validate encoded tokens
        for token_idx in encoded_expr:
            if token_idx >= vocab_size or token_idx < 0:
                print(f"ERROR: Token index {token_idx} is out of bounds [0, {vocab_size-1}]")
                raise ValueError(f"Invalid token index {token_idx}")
        
        # Ensure minimum length
        if len(encoded_expr) < 2:
            expr_idx = random.randint(0, len(val_expressions) - 1)
            expression = val_expressions[expr_idx] + '\n'
            encoded_expr = encode(expression)
        
        # Handle sequence length
        if len(encoded_expr) > block_size:
            print(f"Warning: Expression too long ({len(encoded_expr)} > {block_size}): {expression[:50]}...")
            x = encoded_expr[:block_size]
            y = encoded_expr[1:block_size+1]
        else:
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
def normalize_answer(ans_str):
    """Normalize answer string for consistent comparison"""
    try:
        # Remove any division symbols or equals that might have leaked through
        ans_clean = ans_str.strip().replace('/', '').replace('=', '')
        val = round(float(ans_clean), 4)
        
        # Format consistently with the training data
        if val == int(val) and val < 1000:
            return f"{int(val)}.0"
        else:
            result = f"{val:.4f}".rstrip('0')
            if '.' not in result:
                result += '.0'
            elif result.endswith('.'):
                result += '0'
            return result
    except:
        return ans_str.strip()


@torch.no_grad()
def estimate_loss(model, val_expressions, num_digits=2, max_terms=4):
    """Estimate loss on train and validation sets"""
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch_division(split, batch_size, block_size, val_expressions, 
                                     encode, vocab_size, device, num_digits, max_terms)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


@torch.no_grad()
def evaluate_division_accuracy(model, val_expressions):
    """Evaluate model accuracy on division validation set"""
    model.eval()
    
    correct = 0
    total = len(val_expressions)
    results = []
    
    for expr_str in val_expressions:
        # Extract the expression without the answer
        expr = expr_str.split('=')[0]
        
        try:
            # Generate answer using model
            generated = model.generate_math_answer(expr, encode, decode, itos)
            
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
                
                results.append({
                    'expression': expr,
                    'generated': generated_answer,
                    'generated_normalized': generated_normalized,
                    'correct': correct_normalized,
                    'is_correct': is_correct
                })
            else:
                correct_val = eval(expr)
                correct_normalized = normalize_answer(str(round(float(correct_val), 4)))
                results.append({
                    'expression': expr,
                    'generated': 'MALFORMED',
                    'generated_normalized': 'MALFORMED',
                    'correct': correct_normalized,
                    'is_correct': False
                })
        except Exception as e:
            correct_val = eval(expr)
            correct_normalized = normalize_answer(str(round(float(correct_val), 4)))
            results.append({
                'expression': expr,
                'generated': 'ERROR',
                'generated_normalized': 'ERROR',
                'correct': correct_normalized,
                'is_correct': False,
                'error': str(e)
            })
    
    accuracy = correct / total if total > 0 else 0
    model.train()
    return accuracy, correct, total, results


def main():
    # Create validation set
    val_expressions = create_division_validation_set(count=30, num_digits=2, max_terms=4)
    
    # TensorBoard setup
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f'division_only_lr{learning_rate}_emb{n_embd}_h{n_head}_l{n_layer}_{timestamp}'
    log_dir = f'runs/tensorboard/{run_name}'
    writer = SummaryWriter(log_dir)
    
    # Initialize model
    model = GPTLanguageModel(vocab_size, n_embd, n_head, n_layer, block_size, dropout, device, stoi)
    model = model.to(device)
    print(f"Model initialized: {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters\n")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Model saving setup
    model_save_dir = f'runs/models/{run_name}'
    os.makedirs(model_save_dir, exist_ok=True)
    
    # Save configuration
    config_file = f'{model_save_dir}/division_config.json'
    with open(config_file, 'w') as f:
        json.dump({
            'run_name': run_name,
            'timestamp': timestamp,
            'task': 'division_only',
            'validation_set_size': len(val_expressions),
            'target_accuracy': 0.90,
            'num_digits': 2,
            'max_terms': 4,
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
            'vocabulary': full_chars,
            'generation_vocabulary': generation_chars
        }, f, indent=2)
    
    print(f"{'='*70}")
    print(f"Training Division-Only MathGPT")
    print(f"Run Name: {run_name}")
    print(f"{'='*70}")
    print(f"TensorBoard: {log_dir}")
    print(f"Model Dir: {model_save_dir}")
    print(f"Config: {config_file}")
    print(f"Target Accuracy: 90%")
    print(f"Validation Set: {len(val_expressions)} expressions")
    print(f"{'='*70}\n")
    
    best_val_loss = float('inf')
    best_accuracy = 0.0
    target_accuracy = 0.90
    start_time = time.time()
    
    for iter in range(max_iters):
        # Evaluate loss and accuracy
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss(model, val_expressions, num_digits=2, max_terms=4)
            accuracy, correct, total, results = evaluate_division_accuracy(model, val_expressions)
            
            # Log to TensorBoard
            writer.add_scalars('Loss', {
                'Train': losses['train'],
                'Validation': losses['val']
            }, iter)
            
            writer.add_scalar('Accuracy/Validation', accuracy, iter)
            writer.add_scalar('Accuracy/Correct_Count', correct, iter)
            
            elapsed_time = time.time() - start_time
            writer.add_scalar('Training/Time_Elapsed', elapsed_time, iter)
            
            # Log sample predictions
            sample_text = "\n".join([
                f"{'✓' if r['is_correct'] else '✗'} {r['expression']} = {r['generated']} (correct: {r['correct']})"
                for r in results[:5]
            ])
            writer.add_text('Sample_Predictions', sample_text, iter)
            
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, "
                  f"accuracy {accuracy:.3f} ({correct}/{total}) - Target: {target_accuracy:.1%}")
            
            # Save best models
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                torch.save({
                    'run_name': run_name,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': losses['val'],
                    'train_loss': losses['train'],
                    'accuracy': accuracy,
                    'epoch': iter,
                    'vocab_size': vocab_size,
                    'full_chars': full_chars,
                    'generation_chars': generation_chars,
                    'stoi': stoi,
                    'itos': itos
                }, f'{model_save_dir}/best_loss_model.pt')
                print(f"  ✓ Best loss model saved (val_loss: {best_val_loss:.4f})")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save({
                    'run_name': run_name,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': losses['val'],
                    'train_loss': losses['train'],
                    'accuracy': accuracy,
                    'correct': correct,
                    'total': total,
                    'epoch': iter,
                    'vocab_size': vocab_size,
                    'full_chars': full_chars,
                    'generation_chars': generation_chars,
                    'stoi': stoi,
                    'itos': itos
                }, f'{model_save_dir}/best_accuracy_model.pt')
                print(f"  ✓ Best accuracy model saved (accuracy: {best_accuracy:.3f})")
            
            # Check if target accuracy reached
            if accuracy >= target_accuracy:
                print(f"\n{'='*60}")
                print(f"🎉 TARGET ACCURACY REACHED!")
                print(f"   Accuracy: {correct}/{total} = {accuracy:.1%} >= {target_accuracy:.1%}")
                print(f"   Stopping training early at iteration {iter}/{max_iters}")
                print(f"{'='*60}\n")
                
                # Save final model
                torch.save({
                    'run_name': run_name,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'target_reached': True,
                    'completion_iter': iter,
                    'final_accuracy': accuracy,
                    'correct': correct,
                    'total': total,
                    'val_loss': losses['val'],
                    'train_loss': losses['train'],
                    'vocab_size': vocab_size,
                    'full_chars': full_chars,
                    'generation_chars': generation_chars,
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
                }, f'{model_save_dir}/division_final_model.pt')
                print(f"Final model saved: {model_save_dir}/division_final_model.pt")
                
                # Save detailed results
                results_file = f'{model_save_dir}/division_results.json'
                with open(results_file, 'w') as f:
                    json.dump({
                        'accuracy': accuracy,
                        'correct': correct,
                        'total': total,
                        'target_accuracy': target_accuracy,
                        'iterations': iter,
                        'training_time_min': (time.time() - start_time) / 60,
                        'results': results
                    }, f, indent=2)
                print(f"Detailed results saved: {results_file}")
                
                writer.add_text('Training_Status', f"""TARGET ACCURACY REACHED!
Final accuracy: {accuracy:.3f} ({correct}/{total})
Total iterations: {iter}
Training time: {(time.time() - start_time) / 60:.1f} minutes""", iter)
                
                break
        
        # Training step
        xb, yb = get_batch_division('train', batch_size, block_size, val_expressions,
                                    encode, vocab_size, device, num_digits=2, max_terms=4)
        logits, loss = model(xb, yb)
        
        if iter % 50 == 0:
            writer.add_scalar('Loss/Training_Step', loss.item(), iter)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    
    # Final summary
    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"Training completed in {total_time/60:.1f} minutes")
    print(f"Best accuracy achieved: {best_accuracy:.3f}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"{'='*70}\n")
    
    print(f"Model files saved to: {model_save_dir}")
    print(f"- best_loss_model.pt (lowest validation loss)")
    print(f"- best_accuracy_model.pt (highest accuracy)")
    print(f"- division_final_model.pt (final state)")
    print(f"\nTo view training logs: tensorboard --logdir={log_dir}")
    
    writer.close()


if __name__ == '__main__':
    main()
