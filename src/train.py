"""
Training script for MathGPT with curriculum learning
"""
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

from model import GPTLanguageModel
from data import (
    prepare_math_data, safe_encode, safe_decode, generate_test_expressions,
    create_curriculum_stages, get_batch, update_validation_set
)


# hyperparameters - tuned for math expressions
batch_size = 64
block_size = 28
max_iters = 100000
eval_interval = 50
learning_rate = 2e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
eval_iters = 100
n_embd = 120
n_head = 10
n_layer = 6
dropout = 0.2

torch.manual_seed(1337)
random.seed(1337)

# Prepare vocabulary
chars = prepare_math_data()
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

encode = lambda s: safe_encode(s, stoi)
decode = lambda l: safe_decode(l, itos)

# Generate test expressions
test_expressions_1digit = generate_test_expressions(num_digits=1, count=10)
test_expressions_2digit = generate_test_expressions(num_digits=2, count=10)

# Create curriculum stages
curriculum_stages = create_curriculum_stages(num_digits_list=[1, 2])

# Curriculum Learning Configuration - use stage-specific settings
curriculum_config = {
    'num_digits': curriculum_stages[0]['num_digits'],
    'max_terms': curriculum_stages[0].get('max_terms', 3),
    'max_value': curriculum_stages[0]['max_value'],
    'operators': curriculum_stages[0]['operators'],
    'accuracy_threshold': curriculum_stages[0].get('accuracy_threshold', 0.95),
}

current_stage = 0

print(f"\n{'='*70}")
print(f"DIVISION-FIRST CURRICULUM LEARNING SETUP")
print(f"{'='*70}")
print(f"Strategy: Master hardest operation first, then add easier ones")
print(f"Total stages: {len(curriculum_stages)}\n")
for i, stage in enumerate(curriculum_stages):
    desc = stage.get('description', '')
    threshold = stage.get('accuracy_threshold', 0.95)
    print(f"  Stage {i+1:2d}: {stage['name']:30s}")
    print(f"            Operators: {stage['operators']}, Target: {threshold*100:.0f}% - {desc}")
print(f"{'='*70}\n")

# Validation set management
val_expressions = []
val_set_size = 1000
val_expressions = update_validation_set(curriculum_config, val_set_size)
print(f"Validation set ready for stage: {curriculum_stages[current_stage]['name']}\n")

# TensorBoard setup
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
run_name = f'division_first_curriculum_lr{learning_rate}_emb{n_embd}_h{n_head}_l{n_layer}_{timestamp}'
log_dir = f'runs/tensorboard/{run_name}'
writer = SummaryWriter(log_dir)


@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, batch_size, block_size, curriculum_config, 
                           val_expressions, encode, vocab_size, device)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


@torch.no_grad()
def normalize_answer(ans_str):
    """Normalize answer string for consistent comparison - 1 decimal place precision"""
    try:
        # Clean the string
        ans_clean = ans_str.strip().replace('\n', '')
        val = round(float(ans_clean), 1)
        
        # ALWAYS format with 1 decimal place
        return f"{val:.1f}"
    except:
        return ans_str.strip()


@torch.no_grad()
def evaluate_math_accuracy(model, current_stage):
    """Evaluate model accuracy on math expressions with per-operation breakdown"""
    model.eval()
    
    current_num_digits = curriculum_stages[current_stage]['num_digits']
    stage_config = curriculum_stages[current_stage]
    
    # Generate test expressions based on current stage configuration
    if current_num_digits == 1:
        test_expressions = test_expressions_1digit
    else:
        test_expressions = test_expressions_2digit
    
    stage_ops = set(curriculum_stages[current_stage]['operators'])
    
    # Map stage operators to test categories
    test_categories = []
    if stage_ops == {'/'}:
        test_categories = ['division']
    elif stage_ops == {'*'}:
        test_categories = ['multiplication']
    elif stage_ops == {'+'}:
        test_categories = ['addition']
    elif stage_ops == {'-'}:
        test_categories = ['subtraction']
    elif stage_ops == {'*', '/'}:
        test_categories = ['multiplication', 'division']
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
                generated = model.generate_math_answer(expr, encode, decode, itos)
                if '=' in generated:
                    generated_answer = generated.split('=')[1].strip().replace('\n', '')
                    correct_val = eval(expr)
                    correct_answer_raw = f"{float(correct_val):.1f}"
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
                    correct_normalized = normalize_answer(f"{float(correct_val):.1f}")
                    operation_results[category].append({
                        'expression': expr,
                        'generated': 'MALFORMED',
                        'generated_normalized': 'MALFORMED',
                        'correct': correct_normalized,
                        'is_correct': False
                    })
            except Exception as e:
                correct_val = eval(expr)
                correct_normalized = normalize_answer(f"{float(correct_val):.1f}")
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


def log_experiment_to_csv(metrics_dict, csv_file):
    """Log experimental results to CSV for easy comparison"""
    csv_exists = os.path.exists(csv_file)
    with open(csv_file, 'a', newline='') as f:
        writer_csv = csv.DictWriter(f, fieldnames=metrics_dict.keys())
        if not csv_exists or f.tell() == 0:
            writer_csv.writeheader()
        writer_csv.writerow(metrics_dict)


def test_math_expressions(model, model_save_dir):
    """Test the model on sample math expressions with comprehensive per-operation analysis"""
    print("\n=== Final Math Expression Test ===")
    
    final_accuracy, operation_stats, operation_results = evaluate_math_accuracy(model, current_stage)
    
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
    
    final_results_dict = {cat.replace('_', ' ').title(): stats['accuracy'] 
                         for cat, stats in operation_stats.items()}
    final_results_dict['Overall'] = final_accuracy
    writer.add_scalars('Final_Results/By_Category', final_results_dict, max_iters)
    
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


def interactive_math_mode(model):
    """Interactive mode for user input"""
    while True:
        user_input = input("Enter a math expression (or 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break
        generated = model.generate_math_answer(user_input, encode, decode, itos)
        print(f"AI-generated answer: {generated.strip()}")


def main():
    global current_stage, val_expressions, curriculum_config
    
    # Initialize model
    model = GPTLanguageModel(vocab_size, n_embd, n_head, n_layer, block_size, dropout, device, stoi)
    model = model.to(device)
    print(f"{sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")
    
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
    
    csv_file = f'{results_dir}/curriculum_learning_results.csv'
    
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
    print(f"Training MathGPT - Division-First Curriculum")
    print(f"Run Name: {run_name}")
    print(f"{'='*70}")
    print(f"TensorBoard: {log_dir}")
    print(f"Model Dir: {model_save_dir}")
    print(f"CSV Log: {csv_file}")
    print(f"Config: {config_file}")
    print(f"Vocabulary: {chars}")
    print(f"Starting Stage: {curriculum_stages[current_stage]['name']}")
    print(f"Starting Operators: {curriculum_config['operators']} (DIVISION FIRST!)")
    print(f"Target Accuracy: {curriculum_config['accuracy_threshold']*100:.0f}%")
    print(f"{'='*70}\n")
    
    start_time = time.time()
    losses = {}
    
    for iter in range(max_iters):
        # Evaluate loss on train and val sets
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss(model)
            math_accuracy, operation_stats, operation_results = evaluate_math_accuracy(model, current_stage)
            
            # Log positional encoding weights (learned combination)
            with torch.no_grad():
                weights = F.softmax(torch.stack([
                    model.weight_learned, 
                    model.weight_sinusoidal, 
                    model.weight_abacus
                ]), dim=0)
                writer.add_scalars('Positional_Encoding_Weights', {
                    'Learned': weights[0].item(),
                    'Sinusoidal': weights[1].item(),
                    'Abacus': weights[2].item()
                }, iter)
            
            # Log metrics to TensorBoard
            writer.add_scalars('Loss_Comparison', {
                'Train': losses['train'],
                'Validation': losses['val']
            }, iter)
            
            writer.add_scalar('Accuracy/Overall', math_accuracy, iter)
            writer.add_scalar('Learning/Learning_Rate', learning_rate, iter)
            
            accuracy_dict = {cat.replace('_', ' ').title(): stats['accuracy'] 
                            for cat, stats in operation_stats.items()}
            if accuracy_dict:
                writer.add_scalars('Accuracy/By_Category', accuracy_dict, iter)
            
            error_dict = {cat.replace('_', ' ').title(): stats['error_rate'] 
                         for cat, stats in operation_stats.items()}
            if error_dict:
                writer.add_scalars('Error_Rate/By_Category', error_dict, iter)
            
            for category, results in operation_results.items():
                sample_text = "\n".join([f"{r['expression']}: {r['generated']} (correct: {r['correct']}, ✓={r['is_correct']})" 
                                        for r in results[:3]])
                writer.add_text(f'Sample_Predictions/{category.replace("_", " ").title()}', sample_text, iter)
            
            elapsed_time = time.time() - start_time
            writer.add_scalar('Training/Time_Elapsed', elapsed_time, iter)
            
            # Save best models
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
            
            curriculum_config['num_digits'] = stage_info['num_digits']
            curriculum_config['max_value'] = stage_info['max_value']
            curriculum_config['operators'] = stage_ops
            curriculum_config['max_terms'] = stage_info.get('max_terms', 3)
            curriculum_config['accuracy_threshold'] = stage_info.get('accuracy_threshold', 0.95)
            
            current_ops_correct = 0
            current_ops_total = 0
            
            for category, stats in operation_stats.items():
                current_ops_correct += stats['correct']
                current_ops_total += stats['total']
            
            current_ops_accuracy = current_ops_correct / current_ops_total if current_ops_total > 0 else 0
            
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, overall acc {math_accuracy:.3f}")
            target_pct = curriculum_config['accuracy_threshold'] * 100
            print(f"  🎯 Stage {current_stage+1}/{len(curriculum_stages)} '{stage_name}': {current_ops_correct}/{current_ops_total} correct ({current_ops_accuracy:.1%}) - Need {target_pct:.0f}%")
            
            # Show sample predictions every 500 steps for debugging
            if iter % 500 == 0 and iter > 0:
                # Show positional encoding weights
                with torch.no_grad():
                    weights = F.softmax(torch.stack([
                        model.weight_learned, 
                        model.weight_sinusoidal, 
                        model.weight_abacus
                    ]), dim=0)
                    print(f"\n  ⚖️  Positional Encoding Weights:")
                    print(f"      Learned:     {weights[0].item():.3f} ({weights[0].item()*100:.1f}%)")
                    print(f"      Sinusoidal:  {weights[1].item():.3f} ({weights[1].item()*100:.1f}%)")
                    print(f"      Abacus:      {weights[2].item():.3f} ({weights[2].item()*100:.1f}%)")
                
                print(f"\n  📝 Sample predictions for Stage '{stage_name}':")
                for category, results in operation_results.items():
                    if results:
                        print(f"    {category.upper()}:")
                        # Show first 5 examples
                        for r in results[:5]:
                            status = "✓" if r['is_correct'] else "✗"
                            gen = r.get('generated', 'N/A')
                            print(f"      {status} {r['expression']:15s} = {gen:10s} → {r['generated_normalized']:8s} (expected: {r['correct']})")
                print()
            
            # Progress to next stage if 100% accuracy achieved
            if current_ops_accuracy >= curriculum_config['accuracy_threshold'] and current_stage < len(curriculum_stages) - 1:
                print(f"\n{'='*60}")
                print(f"✓ CURRICULUM PROGRESSION: Stage {current_stage} '{stage_name}' mastered!")
                print(f"   Accuracy: {current_ops_correct}/{current_ops_total} = {current_ops_accuracy:.1%}")
                
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
                
                current_stage += 1
                next_stage_info = curriculum_stages[current_stage]
                curriculum_config['operators'] = next_stage_info['operators']
                curriculum_config['num_digits'] = next_stage_info['num_digits']
                curriculum_config['max_value'] = next_stage_info['max_value']
                curriculum_config['max_terms'] = next_stage_info.get('max_terms', 3)
                curriculum_config['accuracy_threshold'] = next_stage_info.get('accuracy_threshold', 0.95)
                print(f"   Moving to Stage {current_stage+1}: {next_stage_info['name']}")
                print(f"   New operators: {curriculum_config['operators']}")
                print(f"   Digit complexity: {curriculum_config['num_digits']}-digit")
                print(f"   New target: {curriculum_config['accuracy_threshold']*100:.0f}%")
                
                val_expressions = update_validation_set(curriculum_config, val_set_size)
                print(f"{'='*60}\n")
                
                writer.add_text('Curriculum_Progression', f"""Stage {current_stage}: {next_stage_info['name']}
Operators: {curriculum_config['operators']}
Digits: {curriculum_config['num_digits']}
Previous stage accuracy: {current_ops_accuracy:.3f} (100%)
Iteration: {iter}""", iter)
                writer.add_scalar('Curriculum/Stage', current_stage, iter)
                writer.add_scalar('Curriculum/NumDigits', curriculum_config['num_digits'], iter)
                continue
            
            # Check if all stages completed - EARLY STOPPING
            if current_stage == len(curriculum_stages) - 1 and current_ops_accuracy >= curriculum_config['accuracy_threshold']:
                print(f"\n{'='*60}")
                print(f"🎉 ALL CURRICULUM STAGES MASTERED!")
                print(f"   Final stage {current_stage} '{stage_name}' accuracy: {current_ops_correct}/{current_ops_total} = {current_ops_accuracy:.1%}")
                print(f"   Stopping training early at iteration {iter}/{max_iters}")
                print(f"{'='*60}\n")
                
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
                
                writer.add_text('Training_Status', f"""ALL CURRICULUM STAGES COMPLETED!
Total iterations: {iter}
Time saved: {max_iters - iter} iterations not needed""", iter)
                break
        
        # Training step
        xb, yb = get_batch('train', batch_size, block_size, curriculum_config, 
                          val_expressions, encode, vocab_size, device)
        logits, loss = model(xb, yb)
        
        if iter % 50 == 0:
            writer.add_scalar('Loss/Training_Step', loss.item(), iter)
        
        if iter % 100 == 0:
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            
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
    print("\nTesting MathGPT...")
    final_accuracy, final_operation_stats = test_math_expressions(model, model_save_dir)
    
    # Log experiment results to CSV
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
        'best_train_loss': losses['train'] if losses else 0,
        'overall_accuracy': final_accuracy,
    }
    
    for category, stats in final_operation_stats.items():
        safe_category = category.replace('_', ' ')
        experiment_metrics[f'{safe_category}_accuracy'] = stats['accuracy']
        experiment_metrics[f'{safe_category}_error_rate'] = stats['error_rate']
    
    experiment_metrics['model_parameters_M'] = sum(p.numel() for p in model.parameters())/1e6
    experiment_metrics['training_time_min'] = (time.time() - start_time) / 60
    experiment_metrics['model_dir'] = model_save_dir
    experiment_metrics['tensorboard_dir'] = log_dir
    
    log_experiment_to_csv(experiment_metrics, csv_file)
    print(f"\nExperiment logged to: {csv_file}")
    
    # Log model architecture to TensorBoard
    try:
        dummy_input = torch.zeros(1, block_size, dtype=torch.long).to(device)
        model.eval()
        with torch.no_grad():
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

## Performance by Category
""" + "\n".join([f"- {cat.replace('_', ' ').title()}: {stats['accuracy']:.3f} (Error: {stats['error_rate']:.3f})" 
                  for cat, stats in final_operation_stats.items()]) + f"""

## Model Files
- Best Model: {model_save_dir}/best_model.pt
- Best Accuracy: {model_save_dir}/best_accuracy_model.pt
- Final Model: {model_save_dir}/final_model.pt

## Hyperparameters
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
    
    writer.close()
    
    print("\nStarting interactive mode...")
    interactive_math_mode(model)


if __name__ == '__main__':
    main()
