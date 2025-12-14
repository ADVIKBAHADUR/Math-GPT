"""
Data generation and curriculum learning utilities for MathGPT
Handles expression generation, test sets, and batch creation
"""
import torch
import random


def prepare_math_data():
    """Prepare vocabulary for math expressions"""
    required_chars = set('0123456789+-*/=.\n')
    chars = sorted(list(required_chars))
    print(f"Vocabulary (size {len(chars)}): {chars}")
    return chars


def safe_encode(s, stoi):
    """Safe encoding that validates characters and filters unknown ones"""
    result = []
    for c in s:
        if c in stoi:
            result.append(stoi[c])
    return result


def safe_decode(l, itos):
    """Safe decoding that validates indices"""
    result = []
    for i in l:
        if i in itos:
            result.append(itos[i])
        else:
            print(f"WARNING: Invalid token index {i} - using space")
            result.append(' ')
    return ''.join(result)


def generate_random_expression(num_digits=1, max_terms=3, max_value=10, operators=['+']):
    """Generate a random math expression on-the-fly with proper decimal handling for division"""
    num_terms = random.randint(2, max_terms)
    
    # Determine min value based on digits
    min_val = 1 if num_digits == 1 else 10
    actual_max = max_value if max_value > 1 else (9 if num_digits == 1 else 99)
    
    # Generate first number
    if num_digits == 1:
        expression = str(random.randint(min_val, actual_max))
    else:
        expression = str(random.randint(10**(num_digits-1), actual_max))
    
    # Add remaining terms
    for _ in range(num_terms - 1):
        op = random.choice(operators)
        
        # For division, avoid zero and very small divisors
        if op == '/':
            if num_digits == 1:
                num = random.randint(2, actual_max)
            else:
                num = random.randint(max(2, 10**(num_digits-1)), actual_max)
        else:
            if num_digits == 1:
                num = random.randint(0, actual_max)
            else:
                num = random.randint(10**(num_digits-1), actual_max)
        
        expression += op + str(num)
    
    # Calculate answer with 1 decimal place precision
    try:
        answer = eval(expression)
        answer_float = round(float(answer), 1)
        
        # ALWAYS format with 1 decimal place
        answer_str = f"{answer_float:.1f}"
                
    except (ZeroDivisionError, ValueError, SyntaxError):
        # If evaluation fails, generate a simpler expression
        return generate_random_expression(num_digits, 2, max_value, ['+'])
    
    return f"{expression}={answer_str}"


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
        'mixed_add_sub': [],
        'mixed_add_sub_mult': [],
        'mixed_all': []
    }
    
    # Generate pure addition tests
    for _ in range(count):
        a = random.randint(min_val, max_val)
        b = random.randint(min_val, max_val)
        test_set['addition'].append(f"{a}+{b}")
    
    # Generate pure subtraction tests (ensure positive results)
    for _ in range(count):
        a = random.randint(min_val, max_val)
        b = random.randint(min_val, a)
        test_set['subtraction'].append(f"{a}-{b}")
    
    # Generate pure multiplication tests
    for _ in range(count):
        a = random.randint(min_val, max_val)
        b = random.randint(min_val, max_val)
        test_set['multiplication'].append(f"{a}*{b}")
    
    # Generate pure division tests (ensure clean division)
    for _ in range(count):
        b = random.randint(max(1, min_val), max_val)
        quotient = random.randint(1, 10 if num_digits == 1 else 5)
        a = b * quotient
        test_set['division'].append(f"{a}/{b}")
    
    # Generate mixed addition/subtraction tests
    for _ in range(count):
        a = random.randint(min_val, max_val)
        b = random.randint(min_val, max_val)
        c = random.randint(0, min(a + b, max_val))
        test_set['mixed_add_sub'].append(f"{a}+{b}-{c}")
    
    # Generate mixed add/sub/mult tests
    for _ in range(count):
        a = random.randint(min_val, max_val)
        b = random.randint(min_val, max_val)
        c = random.randint(min_val, max_val)
        ops = random.choice([
            f"{a}+{b}*{c}",
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
            f"{a*b}/{b}+{c}",
            f"{a}+{b*c}/{b}",
            f"{a*c}/{c}*{b}"
        ])
        test_set['mixed_all'].append(ops)
    
    return test_set


def create_curriculum_stages(num_digits_list=[1, 2]):
    """
    Create curriculum stages - DIVISION-FIRST strategy with EASY warm-up
    Start with easy divisions, then regular divisions, then add other operations
    """
    stages = []
    
    # PHASE 1: Single-digit operations (hardest to easiest)
    stages.append({
        'name': '1_Division_1digit',
        'operators': ['/'],
        'num_digits': 1,
        'max_value': 9,
        'max_terms': 2,
        'accuracy_threshold': 0.90,
        'description': 'Master division with decimals (e.g., 8/3=2.6667)'
    })
    
    stages.append({
        'name': '2_Multiplication_1digit',
        'operators': ['*'],
        'num_digits': 1,
        'max_value': 9,
        'max_terms': 2,
        'accuracy_threshold': 0.95,
        'description': 'Master multiplication (e.g., 7*8=56)'
    })
    
    stages.append({
        'name': '3_Addition_1digit',
        'operators': ['+'],
        'num_digits': 1,
        'max_value': 9,
        'max_terms': 3,
        'accuracy_threshold': 0.95,
        'description': 'Master addition (e.g., 5+3+2=10)'
    })
    
    stages.append({
        'name': '4_Subtraction_1digit',
        'operators': ['-'],
        'num_digits': 1,
        'max_value': 9,
        'max_terms': 3,
        'accuracy_threshold': 0.95,
        'description': 'Master subtraction (e.g., 9-4-2=3)'
    })
    
    stages.append({
        'name': '5_Mult_Div_Mixed_1digit',
        'operators': ['*', '/'],
        'num_digits': 1,
        'max_value': 9,
        'max_terms': 2,
        'accuracy_threshold': 0.88,
        'description': 'Combine multiplication and division'
    })
    
    stages.append({
        'name': '6_Add_Sub_Mixed_1digit',
        'operators': ['+', '-'],
        'num_digits': 1,
        'max_value': 9,
        'max_terms': 3,
        'accuracy_threshold': 0.92,
        'description': 'Combine addition and subtraction'
    })
    
    stages.append({
        'name': '7_All_Ops_1digit',
        'operators': ['+', '-', '*', '/'],
        'num_digits': 1,
        'max_value': 9,
        'max_terms': 3,
        'accuracy_threshold': 0.85,
        'description': 'All operations mixed (1-digit)'
    })
    
    # PHASE 2: Two-digit operations (same progression)
    stages.append({
        'name': '8_Division_2digit',
        'operators': ['/'],
        'num_digits': 2,
        'max_value': 99,
        'max_terms': 2,
        'accuracy_threshold': 0.85,
        'description': 'Two-digit division (e.g., 84/7=12.0)'
    })
    
    stages.append({
        'name': '9_Multiplication_2digit',
        'operators': ['*'],
        'num_digits': 2,
        'max_value': 99,
        'max_terms': 2,
        'accuracy_threshold': 0.90,
        'description': 'Two-digit multiplication (e.g., 12*8=96)'
    })
    
    stages.append({
        'name': '10_Addition_2digit',
        'operators': ['+'],
        'num_digits': 2,
        'max_value': 99,
        'max_terms': 3,
        'accuracy_threshold': 0.92,
        'description': 'Two-digit addition (e.g., 45+38+12=95)'
    })
    
    stages.append({
        'name': '11_Subtraction_2digit',
        'operators': ['-'],
        'num_digits': 2,
        'max_value': 99,
        'max_terms': 3,
        'accuracy_threshold': 0.90,
        'description': 'Two-digit subtraction (e.g., 78-23-15=40)'
    })
    
    stages.append({
        'name': '12_All_Ops_2digit',
        'operators': ['+', '-', '*', '/'],
        'num_digits': 2,
        'max_value': 99,
        'max_terms': 3,
        'accuracy_threshold': 0.80,
        'description': 'All operations mixed (2-digit)'
    })
    
    return stages


def get_batch(split, batch_size, block_size, curriculum_config, val_expressions, 
              encode, vocab_size, device):
    """Generate batches - train uses random expressions, val uses fixed set"""
    batch_x = []
    batch_y = []
    
    for i in range(batch_size):
        if split == 'train':
            expression_str = generate_random_expression(
                num_digits=curriculum_config['num_digits'],
                max_terms=curriculum_config['max_terms'],
                max_value=curriculum_config['max_value'],
                operators=curriculum_config['operators']
            )
        else:  # split == 'val'
            expr_idx = random.randint(0, len(val_expressions) - 1)
            expression_str = val_expressions[expr_idx]
        
        expression = expression_str + '\n'
        encoded_expr = encode(expression)
        
        # Validate encoded tokens
        for token_idx in encoded_expr:
            if token_idx >= vocab_size or token_idx < 0:
                print(f"ERROR: Token index {token_idx} is out of bounds [0, {vocab_size-1}]")
                raise ValueError(f"Invalid token index {token_idx}")
        
        # Ensure minimum length for training
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


def update_validation_set(curriculum_config, val_set_size):
    """Generate fixed validation set for current curriculum stage"""
    val_expressions = []
    print(f"\n{'='*70}")
    print(f"GENERATING VALIDATION SET")
    print(f"{'='*70}")
    print(f"Operators: {curriculum_config['operators']}, Digits: {curriculum_config['num_digits']}")
    print(f"Max Terms: {curriculum_config['max_terms']}, Max Value: {curriculum_config['max_value']}")
    
    # Check if we should use easy division
    use_easy = curriculum_config.get('use_easy_division', False)
    if use_easy:
        print(f"Mode: EASY DIVISION (clean decimals only)")
    
    print(f"\nFirst 10 validation expressions:")
    
    for i in range(val_set_size):
        if use_easy:
            expr = generate_easy_division_expression(
                num_digits=curriculum_config['num_digits']
            )
        else:
            expr = generate_random_expression(
                num_digits=curriculum_config['num_digits'],
                max_terms=curriculum_config['max_terms'],
                max_value=curriculum_config['max_value'],
                operators=curriculum_config['operators']
            )
        val_expressions.append(expr)
        
        # Print first 10 for verification
        if i < 10:
            print(f"  {i+1:2d}. {expr}")
    
    print(f"\n✅ Validation set ready: {len(val_expressions)} expressions")
    print(f"{'='*70}\n")
    return val_expressions
