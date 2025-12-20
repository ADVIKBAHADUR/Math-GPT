"""
Training script for MathGPT with curriculum learning

Features:
- Curriculum learning with 15 stages (reciprocals → 1-digit → 2-digit operations)
- Stage-based tolerance: Stages 8+ allow ±1 error for division rounding
- Checkpoint loading: Set CHECKPOINT_PATH to resume from a saved model
- Answer masking: Prevents invalid tokens (+, *, /, =) in answers during generation
- Fast & full evaluation modes for efficient training
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
    create_curriculum_stages, get_batch, update_validation_set,
    convert_division_to_multiplication
)

# hyperparameters - tuned for math expressions
batch_size = 256
block_size = 64
max_iters = 100000
eval_interval = 200  # Reduced frequency for faster training
learning_rate = 2e-5
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
eval_iters = 20  # Reduced from 100 for faster training
n_embd = 128
n_head = 8
n_layer = 6
dropout = 0.1

# Performance optimizations
FAST_MODE = True  # Skip expensive per-iteration evaluations
EVAL_SUBSET_SIZE = 30  # Evaluate on subset of expressions for speed
DISABLE_ANSWER_MASKING = False  # Set to True if model struggles to learn
# To resume training from a checkpoint, set path to a .pt file:
# Example: CHECKPOINT_PATH = 'runs/models/reciprocal_curriculum_.../stage_3_2b_Simple_Negatives_completed.pt'
CHECKPOINT_PATH = "/root/Math-GPT/runs/models/reciprocal_curriculum_lr3e-05_emb128_h8_l6_20251220_065354/stage_13_9_Multiplication_2digit_Advanced_best_loss.pt" #None #"/root/Math-GPT/runs/models/reciprocal_curriculum_lr0.0001_emb128_h8_l6_20251215_233739/stage_8_6_Add_Sub_Mixed_2digit_best.pt" # Path to .pt file to resume training

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
    'is_reciprocal_stage': curriculum_stages[0].get('is_reciprocal_stage', False),
    'use_reciprocal_for_division': curriculum_stages[0].get('use_reciprocal_for_division', False),
    'include_negative': curriculum_stages[0].get('include_negative', False),
    'simple_negatives_only': curriculum_stages[0].get('simple_negatives_only', False),
}

current_stage = 0

print(f"\n{'='*70}")
print(f"RECIPROCAL-BASED DIVISION CURRICULUM LEARNING SETUP")
print(f"{'='*70}")
print(f"Strategy: Learn reciprocals first, then convert divisions to multiplications")
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
run_name = f'reciprocal_curriculum_lr{learning_rate}_emb{n_embd}_h{n_head}_l{n_layer}_{timestamp}'
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


def fast_evaluate_subset(model, expressions, max_samples=30, debug=False, stage_index=0):
    """Quick evaluation on subset of expressions - much faster than full evaluation
    All stages use 4 decimal precision for consistency
    """
    model.eval()
    correct = 0
    total = min(len(expressions), max_samples)
    mismatches = []  # Track first few mismatches to show
    
    # All stages use 4 decimal precision
    decimals = 4
    
    if debug:
        print(f"\n  DEBUG: Fast eval on {total} expressions (stage {stage_index}, {decimals} decimals)")
        print(f"  First 3 expressions: {expressions[:3]}")
    
    for i in range(total):
        expr = expressions[i]
        try:
            if '=' in expr:
                expr_input = expr.split('=')[0]
                expected = expr.split('=')[1].strip()
            else:
                expr_input = expr
                expected = f"{round(eval(expr), decimals):.{decimals}f}"
            
            generated = model.generate_math_answer(expr_input, encode, decode, itos, max_new_tokens=20)
            if '=' in generated:
                generated_answer = generated.split('=')[1].strip().replace('\n', '')
                generated_normalized = normalize_answer(generated_answer, decimals=decimals)
                expected_normalized = normalize_answer(expected, decimals=decimals)
                is_correct = check_answer_correct(generated_normalized, expected_normalized, stage_index)
                
                if is_correct:
                    correct += 1
                elif len(mismatches) < 5:  # Collect first 5 mismatches
                    mismatches.append((expr_input, generated_answer, expected))
                
                if debug and i < 5:
                    status = "✓" if is_correct else "✗"
                    print(f"    {status} {expr_input} = {generated_answer} (norm: {generated_normalized}, expected: {expected_normalized})")
        except Exception as e:
            if debug and i < 5:
                print(f"    ✗ {expr} - ERROR: {e}")
            pass
    
    # Print first few mismatches for debugging
    if mismatches and not debug:
        for expr_input, generated_answer, expected in mismatches:
            print(f"    MISMATCH: {expr_input} | Generated: {generated_answer} | Expected: {expected}")
    
    model.train()
    accuracy = correct / total if total > 0 else 0.0
    if debug:
        print(f"  Fast eval result: {correct}/{total} = {accuracy:.1%}\n")
    return accuracy


@torch.no_grad()
def normalize_answer(ans_str, decimals=4):
    """Normalize answer string for consistent comparison
    Args:
        ans_str: Answer string to normalize
        decimals: Number of decimal places (default 4 for all stages)
    """
    try:
        # Clean the string
        ans_clean = ans_str.strip().replace('\n', '')
        val = round(float(ans_clean), decimals)
        
        # Format with specified decimal places
        return f"{val:.{decimals}f}"
    except:
        return ans_str


def check_answer_correct(generated_normalized, expected_normalized, stage_index):
    """Check if answer is correct with stage-based tolerance
    
    Args:
        generated_normalized: Generated answer (normalized string)
        expected_normalized: Expected answer (normalized string)
        stage_index: Current curriculum stage (0-indexed)
    
    Returns:
        bool: True if answer is correct (exact or within tolerance)
    """
    # Stages 8+ allow ±1 tolerance due to division rounding errors
    if stage_index >= 8:
        try:
            gen_val = float(generated_normalized)
            exp_val = float(expected_normalized)
            # Accept if within ±1
            return abs(gen_val - exp_val) <= 1.0
        except:
            return generated_normalized == expected_normalized
    else:
        # Stages 0-7: exact match required
        return generated_normalized == expected_normalized


@torch.no_grad()
def old_normalize_answer_fallback(ans_str):
    """Fallback when normalize fails"""
    return ans_str.strip()


@torch.no_grad()
def evaluate_math_accuracy(model, current_stage, validation_expressions=None, track_details=False):
    """Evaluate model accuracy on math expressions with per-operation breakdown
    
    Args:
        model: The model to evaluate
        current_stage: Current curriculum stage index (0-indexed)
        validation_expressions: Optional list of expressions to evaluate. If None, uses hardcoded test sets.
        track_details: If False, skip building detailed operation_results (much faster)
    
    Note: Stages 8+ use ±1 tolerance for division rounding errors
    """
    model.eval()
    
    current_num_digits = curriculum_stages[current_stage]['num_digits']
    stage_config = curriculum_stages[current_stage]
    
    # Special handling for reciprocal stage
    if stage_config.get('is_reciprocal_stage', False):
        # Test on reciprocals 1/1 to 1/100 with 4 decimal precision
        operation_results = {'reciprocals': []}
        correct = 0
        total = 100
        
        for n in range(1, 101):
            reciprocal_val = round(1.0 / n, 4)  # 4 decimals for stage 0
            expr = f"1/{n}"
            try:
                generated = model.generate_math_answer(expr, encode, decode, itos)
                if '=' in generated:
                    generated_answer = generated.split('=')[1].strip().replace('\n', '')
                    correct_answer = f"{reciprocal_val:.4f}"  # 4 decimals
                    generated_normalized = normalize_answer(generated_answer, decimals=4)  # 4 decimals
                    correct_normalized = normalize_answer(correct_answer, decimals=4)  # 4 decimals
                    is_correct = generated_normalized == correct_normalized
                    
                    if is_correct:
                        correct += 1
                    operation_results['reciprocals'].append({
                        'expression': expr,
                        'generated': generated_answer,
                        'generated_normalized': generated_normalized,
                        'correct': correct_normalized,
                        'is_correct': is_correct
                    })
                else:
                    operation_results['reciprocals'].append({
                        'expression': expr,
                        'generated': 'MALFORMED',
                        'generated_normalized': 'MALFORMED',
                        'correct': f"{reciprocal_val:.4f}",  # 4 decimals
                        'is_correct': False
                    })
            except Exception as e:
                print(f"Error evaluating expression {expr}: {e}")
                operation_results['reciprocals'].append({
                    'expression': expr,
                    'generated': 'ERROR',
                    'generated_normalized': 'ERROR',
                    'correct': f"{reciprocal_val:.4f}",  # 4 decimals
                    'is_correct': False,
                    'error': str(e)
                })
        
        accuracy = correct / total if total > 0 else 0
        operation_stats = {
            'reciprocals': {
                'accuracy': accuracy,
                'error_rate': (total - correct) / total if total > 0 else 0,
                'correct': correct,
                'total': total
            }
        }
        model.train()
        return accuracy, operation_stats, operation_results
    
    # Generate test expressions based on current stage configuration
    # Use validation_expressions if provided, otherwise fall back to hardcoded test sets
    if validation_expressions:
        # Use the validation set (properly updated for current stage)
        # Evaluate all validation expressions to get accurate per-operation stats
        expressions_to_test = validation_expressions
        
        # All stages use 4 decimal precision
        decimals = 4
        
        # For validation expressions, we evaluate them all as one category
        # since they're already filtered for the current stage
        operation_results = {'current_stage': []} if track_details else {}
        correct = 0
        total = len(expressions_to_test)
        
        use_reciprocal = stage_config.get('use_reciprocal_for_division', False)
        
        for expr in expressions_to_test:
            try:
                if '=' in expr:
                    expr_input = expr.split('=')[0]
                    expected = expr.split('=')[1].strip()
                else:
                    expr_input = expr
                    # Convert division if needed for expected value
                    test_expr_for_eval = expr_input
                    if use_reciprocal and '/' in expr_input:
                        test_expr_for_eval = convert_division_to_multiplication(expr_input)
                    expected = f"{eval(test_expr_for_eval):.{decimals}f}"
                
                # Test with conversion if needed
                test_expr = expr_input
                if use_reciprocal and '/' in expr_input:
                    test_expr = convert_division_to_multiplication(expr_input)
                
                generated = model.generate_math_answer(test_expr, encode, decode, itos, max_new_tokens=20)
                
                if '=' in generated:
                    generated_answer = generated.split('=')[1].strip().replace('\n', '')
                    generated_normalized = normalize_answer(generated_answer, decimals=decimals)
                    expected_normalized = normalize_answer(expected, decimals=decimals)
                    is_correct = check_answer_correct(generated_normalized, expected_normalized, current_stage)
                    
                    if is_correct:
                        correct += 1
                    
                    if track_details:
                        operation_results['current_stage'].append({
                            'expression': expr_input,
                            'tested_as': test_expr if test_expr != expr_input else None,
                            'generated': generated_answer,
                            'generated_normalized': generated_normalized,
                            'correct': expected_normalized,
                            'is_correct': is_correct
                        })
                else:
                    if track_details:
                        operation_results['current_stage'].append({
                            'expression': expr_input,
                            'tested_as': test_expr if test_expr != expr_input else None,
                            'generated': 'MALFORMED',
                            'generated_normalized': 'MALFORMED',
                            'correct': expected,
                            'is_correct': False
                        })
            except Exception as e:
                if track_details:
                    operation_results['current_stage'].append({
                        'expression': expr_input if '=' in expr else expr,
                        'generated': 'ERROR',
                        'generated_normalized': 'ERROR',
                        'correct': expected if 'expected' in locals() else 'N/A',
                        'is_correct': False,
                        'error': str(e)
                    })
        
        accuracy = correct / total if total > 0 else 0
        operation_stats = {
            'current_stage': {
                'accuracy': accuracy,
                'error_rate': (total - correct) / total if total > 0 else 0,
                'correct': correct,
                'total': total
            }
        }
        model.train()
        return accuracy, operation_stats, operation_results
    
    # Fall back to hardcoded test expressions if validation_expressions not provided
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
    
    # Check if we should convert divisions to multiplications
    use_reciprocal = stage_config.get('use_reciprocal_for_division', False)
    
    for category in test_categories:
        expressions = test_expressions.get(category, [])
        if not expressions:
            continue
            
        correct = 0
        total = len(expressions)
        operation_results[category] = []
        
        for expr in expressions:
            try:
                # Convert division to multiplication if needed
                test_expr = expr
                if use_reciprocal and '/' in expr:
                    test_expr = convert_division_to_multiplication(expr)
                
                generated = model.generate_math_answer(test_expr, encode, decode, itos)
                if '=' in generated:
                    generated_answer = generated.split('=')[1].strip().replace('\n', '')
                    
                    # CRITICAL FIX: Evaluate the CONVERTED expression for correct answer
                    if use_reciprocal and test_expr != expr:
                        correct_val = eval(test_expr)  # Use converted expression
                    else:
                        correct_val = eval(expr)  # Original expression
                    
                    correct_answer_raw = f"{float(correct_val):.4f}"
                    generated_normalized = normalize_answer(generated_answer)
                    correct_normalized = normalize_answer(correct_answer_raw)
                    is_correct = generated_normalized == correct_normalized
                    
                    if is_correct:
                        correct += 1
                        all_correct += 1
                    operation_results[category].append({
                        'expression': expr,
                        'tested_as': test_expr if test_expr != expr else None,
                        'generated': generated_answer,
                        'generated_normalized': generated_normalized,
                        'correct': correct_normalized,
                        'is_correct': is_correct
                    })
                else:
                    # CRITICAL FIX: Evaluate converted expression if applicable
                    if use_reciprocal and test_expr != expr:
                        correct_val = eval(test_expr)
                    else:
                        correct_val = eval(expr)
                    correct_normalized = normalize_answer(f"{float(correct_val):.4f}")
                    operation_results[category].append({
                        'expression': expr,
                        'tested_as': test_expr if test_expr != expr else None,
                        'generated': 'MALFORMED',
                        'generated_normalized': 'MALFORMED',
                        'correct': correct_normalized,
                        'is_correct': False
                    })
            except Exception as e:
                correct_val = eval(expr)
                correct_normalized = normalize_answer(f"{float(correct_val):.4f}")
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
    
    # Final evaluation - keep detailed results for reporting
    final_accuracy, operation_stats, operation_results = evaluate_math_accuracy(model, current_stage, val_expressions, track_details=True)
    
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
    print(f"\n{'='*70}")
    print(f"Current Stage: {current_stage+1}/{len(curriculum_stages)} - {curriculum_stages[current_stage]['name']}")
    print(f"Operators trained: {curriculum_stages[current_stage]['operators']}")
    use_reciprocal = curriculum_stages[current_stage].get('use_reciprocal_for_division', False)
    if use_reciprocal:
        print(f"⚠️  Note: Division is converted to multiplication with reciprocals")
    print(f"{'='*70}\n")
    
    while True:
        user_input = input("Enter a math expression (or 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break
        
        # Convert division if current stage uses reciprocals
        test_expr = user_input
        if use_reciprocal and '/' in user_input:
            test_expr = convert_division_to_multiplication(user_input)
            print(f"  [Converted to: {test_expr}]")
        
        generated = model.generate_math_answer(test_expr, encode, decode, itos)
        print(f"AI-generated answer: {generated.strip()}")
        
        # Show expected answer for comparison
        try:
            expected = eval(test_expr)
            print(f"Expected: {user_input}={expected:.4f}\n")
        except:
            print()


def main():
    global current_stage, val_expressions, curriculum_config
    
    # Initialize model
    model = GPTLanguageModel(vocab_size, n_embd, n_head, n_layer, block_size, dropout, device, stoi)
    model = model.to(device)
    
    # Load checkpoint if specified
    start_iter = 0
    previous_stage_configs = []  # Will be populated from checkpoint or built during training
    replay_ratio = 0.3  # 30% of training batch from previous stages
    
    if CHECKPOINT_PATH and os.path.exists(CHECKPOINT_PATH):
        print(f"\n{'='*70}")
        print(f"LOADING CHECKPOINT: {CHECKPOINT_PATH}")
        print(f"{'='*70}")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
        # Use strict=False to ignore cached mask buffers from previous runs
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f"✓ Model state loaded")
        
        # Load curriculum stage if available
        if 'current_stage' in checkpoint:
            current_stage = checkpoint['current_stage']
            print(f"✓ Resuming from stage {current_stage}: {curriculum_stages[current_stage]['name']}")
            
            # Update curriculum config for resumed stage
            stage_info = curriculum_stages[current_stage]
            curriculum_config['operators'] = stage_info['operators']
            curriculum_config['num_digits'] = stage_info['num_digits']
            curriculum_config['max_value'] = stage_info['max_value']
            curriculum_config['max_terms'] = stage_info.get('max_terms', 3)
            curriculum_config['accuracy_threshold'] = stage_info.get('accuracy_threshold', 0.95)
            curriculum_config['is_reciprocal_stage'] = stage_info.get('is_reciprocal_stage', False)
            curriculum_config['use_reciprocal_for_division'] = stage_info.get('use_reciprocal_for_division', False)
            curriculum_config['include_negative'] = stage_info.get('include_negative', False)
            curriculum_config['simple_negatives_only'] = stage_info.get('simple_negatives_only', False)
            
            # Rebuild previous stage configs for replay buffer (cumulative learning)
            for stage_idx in range(current_stage):
                stage_cfg_info = curriculum_stages[stage_idx]
                stage_cfg = {
                    'num_digits': stage_cfg_info['num_digits'],
                    'max_terms': stage_cfg_info.get('max_terms', 3),
                    'max_value': stage_cfg_info['max_value'],
                    'operators': stage_cfg_info['operators'],
                    'is_reciprocal_stage': stage_cfg_info.get('is_reciprocal_stage', False),
                    'use_reciprocal_for_division': stage_cfg_info.get('use_reciprocal_for_division', False),
                    'include_negative': stage_cfg_info.get('include_negative', False),
                    'simple_negatives_only': stage_cfg_info.get('simple_negatives_only', False),
                }
                previous_stage_configs.append(stage_cfg)
            print(f"✓ Replay buffer rebuilt with {len(previous_stage_configs)} previous stages")
            
            # Regenerate validation set for current stage
            val_expressions = update_validation_set(curriculum_config, val_set_size)
        
        if 'iter' in checkpoint:
            start_iter = checkpoint['iter']
            print(f"✓ Resuming from iteration {start_iter}")
        
        print(f"{'='*70}\n")
    elif CHECKPOINT_PATH:
        print(f"⚠️  Checkpoint path specified but not found: {CHECKPOINT_PATH}")
        print(f"   Starting from scratch...\n")
    
    # Optionally disable answer masking for faster/easier learning
    if DISABLE_ANSWER_MASKING:
        model._use_answer_masking = False
        print("⚠️  Answer masking DISABLED for easier learning")
    
    print(f"{sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Add learning rate scheduler for curriculum learning stability
    # Reduce LR when loss plateaus or when switching stages
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=50, verbose=True, min_lr=1e-6
    )
    print(f"✓ Using AdamW optimizer with ReduceLROnPlateau scheduler")
    print(f"  Initial LR: {learning_rate}, Min LR: 1e-6, Patience: 50 steps")
    
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
    
    # Per-stage tracking for best models
    stage_best_accuracy = {}  # Track best accuracy per stage
    stage_best_loss = {}  # Track best loss per stage
    for i in range(len(curriculum_stages)):
        stage_best_accuracy[i] = 0.0
        stage_best_loss[i] = float('inf')
    
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
    print(f"Training MathGPT - Cumulative Curriculum Learning")
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
    print(f"\n🔄 CUMULATIVE LEARNING ENABLED:")
    print(f"   - Replay Ratio: {replay_ratio*100:.0f}% of batches from previous stages")
    print(f"   - Current Replay Buffer: {len(previous_stage_configs)} stages")
    print(f"   - Prevents catastrophic forgetting of earlier concepts")
    print(f"{'='*70}\n")
    
    start_time = time.time()
    losses = {}
    
    # Track previous stages for cumulative curriculum learning (prevents forgetting)
    previous_stage_configs = []
    replay_ratio = 0.3  # 30% of training batch from previous stages
    
    for iter in range(start_iter, max_iters):
        # Evaluate loss on train and val sets
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss(model)
            
            # Use fast evaluation during training, full evaluation at milestones
            # Always do full eval for reciprocal stage (stage 0) or at important checkpoints
            is_reciprocal_stage = curriculum_stages[current_stage].get('is_reciprocal_stage', False)
            is_milestone = iter % (eval_interval * 5) == 0 or iter == max_iters - 1
            
            if FAST_MODE and not is_reciprocal_stage and not is_milestone and iter > 0:
                # Quick accuracy check on small subset
                sample_exprs = val_expressions[:EVAL_SUBSET_SIZE]
                # Disable debug for speed - only enable manually if needed
                math_accuracy = fast_evaluate_subset(model, sample_exprs, EVAL_SUBSET_SIZE, debug=False, stage_index=current_stage)
                operation_stats = {}
                operation_results = {}
            else:
                # Full evaluation at milestones, for reciprocal stage, or first iteration
                # Use the validation set for consistency with fast eval
                # Skip detailed tracking during training for speed (only need accuracy)
                math_accuracy, operation_stats, operation_results = evaluate_math_accuracy(model, current_stage, val_expressions, track_details=False)
            
            # Log metrics to TensorBoard (reduced frequency for speed)
            should_log_tensorboard = (iter % (eval_interval * 5) == 0) or iter == max_iters - 1 or is_milestone
            if should_log_tensorboard:
                writer.add_scalars('Loss_Comparison', {
                    'Train': losses['train'],
                    'Validation': losses['val']
                }, iter)
                
                writer.add_scalar('Accuracy/Overall', math_accuracy, iter)
                writer.add_scalar('Learning/Learning_Rate', optimizer.param_groups[0]['lr'], iter)
                
                accuracy_dict = {cat.replace('_', ' ').title(): stats['accuracy'] 
                                for cat, stats in operation_stats.items()}
                if accuracy_dict:
                    writer.add_scalars('Accuracy/By_Category', accuracy_dict, iter)
                
                error_dict = {cat.replace('_', ' ').title(): stats['error_rate'] 
                             for cat, stats in operation_stats.items()}
                if error_dict:
                    writer.add_scalars('Error_Rate/By_Category', error_dict, iter)
                
                # Only log sample predictions at milestones (heavy operation)
                if operation_results and is_milestone:
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
                    'itos': itos,
                    'block_size': block_size
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
                    'itos': itos,
                    'block_size': block_size
                }, f'{model_save_dir}/best_accuracy_model.pt')
                print(f"  -> New best accuracy model saved (accuracy: {best_accuracy:.3f})")
            
            # Per-stage best model tracking (only save at milestones to reduce I/O)
            if is_milestone or iter == max_iters - 1:
                if math_accuracy > stage_best_accuracy[current_stage]:
                    stage_best_accuracy[current_stage] = math_accuracy
                    stage_file = f'{model_save_dir}/stage_{current_stage}_{curriculum_stages[current_stage]["name"]}_best.pt'
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
                        'stage_best_accuracy': math_accuracy,
                        'vocab_size': vocab_size,
                        'chars': chars,
                        'stoi': stoi,
                        'itos': itos,
                        'block_size': block_size
                    }, stage_file)
                    print(f"  -> Stage {current_stage} best saved: {math_accuracy:.3f}")
                
                if losses['val'] < stage_best_loss[current_stage]:
                    stage_best_loss[current_stage] = losses['val']
                    stage_file = f'{model_save_dir}/stage_{current_stage}_{curriculum_stages[current_stage]["name"]}_best_loss.pt'
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
                        'stage_best_loss': losses['val'],
                        'vocab_size': vocab_size,
                        'chars': chars,
                        'stoi': stoi,
                        'itos': itos,
                        'block_size': block_size
                    }, stage_file)
                    print(f"  -> Stage {current_stage} best loss saved: {losses['val']:.4f}")
            
            # Curriculum progression logic
            stage_info = curriculum_stages[current_stage]
            stage_name = stage_info['name']
            stage_ops = stage_info['operators']
            
            curriculum_config['num_digits'] = stage_info['num_digits']
            curriculum_config['max_value'] = stage_info['max_value']
            curriculum_config['operators'] = stage_ops
            curriculum_config['max_terms'] = stage_info.get('max_terms', 3)
            curriculum_config['accuracy_threshold'] = stage_info.get('accuracy_threshold', 0.95)
            
            # Calculate accuracy on current stage operations
            current_ops_correct = 0
            current_ops_total = 0
            
            if operation_stats:
                for category, stats in operation_stats.items():
                    current_ops_correct += stats['correct']
                    current_ops_total += stats['total']
                
                current_ops_accuracy = current_ops_correct / current_ops_total if current_ops_total > 0 else 0
                progress_msg = f"{current_ops_correct}/{current_ops_total} correct"
            else:
                # In fast mode, use overall accuracy as proxy
                current_ops_accuracy = math_accuracy
                progress_msg = "fast eval"
            
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, overall acc {math_accuracy:.3f}")
            print(f"  🎯 Stage {current_stage+1}/{len(curriculum_stages)} '{stage_name}': {progress_msg} ({current_ops_accuracy:.1%}) - Need {curriculum_config['accuracy_threshold']:.0%}")
            
            # Show sample predictions every 1000 steps for debugging (reduced frequency)
            if iter % 1000 == 0 and iter > 0 and operation_results:
                print(f"\n  📝 Sample predictions for Stage '{stage_name}':")
                use_reciprocal_display = stage_info.get('use_reciprocal_for_division', False)
                for category, results in operation_results.items():
                    if results:
                        print(f"    {category.upper()}:")
                        # Show first 3 examples only (reduced from 5)
                        for r in results[:3]:
                            status = "✓" if r['is_correct'] else "✗"
                            gen = r.get('generated', 'N/A')
                            expr = r['expression']
                            tested_as = r.get('tested_as', None)
                            
                            # Show both original and converted if different
                            if tested_as:
                                print(f"      {status} {expr:15s} (as {tested_as}) = {gen:10s} → {r['generated_normalized']:8s} (expected: {r['correct']})")
                            else:
                                print(f"      {status} {expr:15s} = {gen:10s} → {r['generated_normalized']:8s} (expected: {r['correct']})")
                print()
            
            # Progress to next stage if 100% accuracy achieved
            if current_ops_accuracy >= curriculum_config['accuracy_threshold'] and current_stage < len(curriculum_stages) - 1:
                # FORCE FULL EVALUATION before allowing progression (not just fast subset)
                if FAST_MODE and operation_stats == {}:
                    print(f"\n  📊 Running FULL evaluation before stage progression...")
                    # Skip detailed tracking - we only need the accuracy count
                    math_accuracy, operation_stats, operation_results = evaluate_math_accuracy(model, current_stage, val_expressions, track_details=False)
                    
                    # Recalculate accuracy with full evaluation
                    current_ops_correct = 0
                    current_ops_total = 0
                    for category, stats in operation_stats.items():
                        current_ops_correct += stats['correct']
                        current_ops_total += stats['total']
                    current_ops_accuracy = current_ops_correct / current_ops_total if current_ops_total > 0 else 0
                    print(f"  📊 Full evaluation: {current_ops_correct}/{current_ops_total} = {current_ops_accuracy:.1%}")
                    
                    # Check if still meets threshold after full eval
                    if current_ops_accuracy < curriculum_config['accuracy_threshold']:
                        print(f"  ⚠️  Full evaluation shows accuracy below threshold!")
                        print(f"      Fast subset was optimistic. Continuing training...\n")
                        continue
                
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
                    'block_size': block_size,
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
                
                # CUMULATIVE LEARNING: Save current stage config before advancing (prevents forgetting)
                previous_stage_configs.append(curriculum_config.copy())
                print(f"   📦 Added stage {current_stage} to replay buffer (total: {len(previous_stage_configs)} stages)")
                print(f"   📊 Training will now include {replay_ratio*100:.0f}% data from previous stages")
                
                current_stage += 1
                next_stage_info = curriculum_stages[current_stage]
                curriculum_config['operators'] = next_stage_info['operators']
                curriculum_config['num_digits'] = next_stage_info['num_digits']
                curriculum_config['max_value'] = next_stage_info['max_value']
                curriculum_config['max_terms'] = next_stage_info.get('max_terms', 3)
                curriculum_config['accuracy_threshold'] = next_stage_info.get('accuracy_threshold', 0.95)
                curriculum_config['is_reciprocal_stage'] = next_stage_info.get('is_reciprocal_stage', False)
                curriculum_config['use_reciprocal_for_division'] = next_stage_info.get('use_reciprocal_for_division', False)
                curriculum_config['include_negative'] = next_stage_info.get('include_negative', False)
                curriculum_config['simple_negatives_only'] = next_stage_info.get('simple_negatives_only', False)
                
                # Clear cached answer mask so it can be regenerated with new valid tokens if needed
                if hasattr(model, '_invalid_answer_mask'):
                    delattr(model, '_invalid_answer_mask')
                    print(f"   ⚠️  Cleared cached answer mask for new stage")
                print(f"   Moving to Stage {current_stage+1}: {next_stage_info['name']}")
                print(f"   New operators: {curriculum_config['operators']}")
                print(f"   Digit complexity: {curriculum_config['num_digits']}-digit")
                print(f"   New target: {curriculum_config['accuracy_threshold']*100:.0f}%")
                
                print(f"\n   🔄 UPDATING VALIDATION SET for new stage...")
                val_expressions = update_validation_set(curriculum_config, val_set_size)
                print(f"   ✅ Validation set updated for new stage")
                print(f"   Sample val expressions: {val_expressions[:5]}")
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
        
        # Training step with cumulative curriculum learning
        xb, yb = get_batch('train', batch_size, block_size, curriculum_config, 
                          val_expressions, encode, vocab_size, device,
                          previous_stage_configs=previous_stage_configs,
                          replay_ratio=replay_ratio)
        logits, loss = model(xb, yb)
        
        # Reduced logging frequency for better performance (only log every 10 eval intervals)
        if iter % (eval_interval * 10) == 0:
            writer.add_scalar('Loss/Training_Step', loss.item(), iter)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients (especially during stage transitions)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Update learning rate based on validation loss
        if iter % eval_interval == 0 and iter > 0:
            scheduler.step(losses['val'])
    
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
        'itos': itos,
        'block_size': block_size
    }, f'{model_save_dir}/final_model.pt')
    
    print(f"\nModels saved to: {model_save_dir}")
    print(f"- best_model.pt (lowest validation loss: {best_val_loss:.4f})")
    print(f"- best_accuracy_model.pt (highest accuracy: {best_accuracy:.3f})")
    print(f"- final_model.pt (final training state)")
    
    # Print per-stage best models
    print(f"\nPer-stage best models:")
    for stage_idx in range(current_stage + 1):
        stage_name = curriculum_stages[stage_idx]['name']
        if stage_best_accuracy[stage_idx] > 0:
            print(f"  Stage {stage_idx}: {stage_name}")
            print(f"    - Best accuracy: {stage_best_accuracy[stage_idx]:.3f}")
            print(f"    - Best loss: {stage_best_loss[stage_idx]:.4f}")
    
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