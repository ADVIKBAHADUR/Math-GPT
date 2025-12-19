"""
Data generation and curriculum learning utilities for MathGPT
Handles expression generation, test sets, and batch creation
"""
import torch
import random


# Reciprocal lookup table (1/1 to 1/100 with 4 decimal places for precision)
# Reciprocal table with 4 decimal precision for stage 0 training
# Example: 1/93 = 0.0108, 1/50 = 0.0200, 1/4 = 0.2500
RECIPROCAL_TABLE = {i: round(1.0 / i, 4) for i in range(1, 101)}


def generate_reciprocal_expression(n, include_negative=False):
    """Generate a reciprocal expression: 1/n=result (4 decimal places)
    If include_negative, sometimes generate -1/n for negative practice
    """
    if n < 1 or n > 100:
        n = random.randint(1, 100)
    
    # Always use 4 decimals for reciprocals (stage 0)
    decimals = 4
    
    # 20% chance of negative reciprocal if enabled
    if include_negative and random.random() < 0.2:
        result = -RECIPROCAL_TABLE[n]
        return f"-1/{n}={result:.{decimals}f}"
    else:
        result = RECIPROCAL_TABLE[n]
        return f"1/{n}={result:.{decimals}f}"


def generate_reciprocal_dataset(count=100, include_negative=False):
    """Generate dataset of reciprocals from 1/1 to 1/100
    If include_negative, adds negative reciprocals for early negative exposure
    """
    expressions = []
    # Cover all reciprocals 1-100
    for i in range(1, 101):
        expressions.append(generate_reciprocal_expression(i, include_negative))
    # Add extra samples for commonly needed reciprocals
    common_divisors = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    for _ in range(count - 100):
        n = random.choice(common_divisors)
        expressions.append(generate_reciprocal_expression(n, include_negative))
    return expressions


def convert_division_to_multiplication(expression):
    """
    Convert division expression to multiplication with reciprocal FRACTION.
    Example: '8/4' becomes '8*(1/4)' - the model must recall that 1/4 = 0.25
    This forces the model to use learned reciprocals from Stage 0.
    """
    # Handle simple division (e.g., '8/4')
    if '/' in expression and expression.count('/') == 1:
        parts = expression.split('/')
        if len(parts) == 2:
            try:
                numerator = parts[0].strip()
                denominator = int(parts[1].strip())
                if 1 <= denominator <= 100:
                    # Return as fraction, not decimal!
                    return f"{numerator}*(1/{denominator})"
            except:
                pass
    return expression


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


def generate_random_expression(num_digits=1, max_terms=3, max_value=10, operators=['+'], use_reciprocal_for_division=False, include_negative=False, simple_negatives_only=False):
    """Generate a random math expression on-the-fly with proper decimal handling
    All expressions use 4 decimal places for consistency
    If include_negative, 20% of expressions will start with a negative number
    If simple_negatives_only, ALWAYS generates -X*Y where X,Y are 1-10 (ignores other settings)
    """
    # Simple negatives mode: -1*1 through -10*10
    if simple_negatives_only:
        first_num = random.randint(1, 10)
        second_num = random.randint(1, 10)
        expression = f"-{first_num}*{second_num}"
        answer = -first_num * second_num
        answer_str = f"{float(answer):.4f}"
        return f"{expression}={answer_str}"
    
    num_terms = random.randint(2, max_terms)
    
    # 20% chance to start with negative number if enabled
    use_negative_start = include_negative and random.random() < 0.2
    
    # Determine min value based on digits
    min_val = 1 if num_digits == 1 else 10
    actual_max = max_value if max_value > 1 else (9 if num_digits == 1 else 99)
    
    # Generate first number (possibly negative)
    if num_digits == 1:
        first_num = random.randint(min_val, actual_max)
    else:
        first_num = random.randint(10**(num_digits-1), actual_max)
    
    if use_negative_start:
        expression = f"-{first_num}"
    else:
        expression = str(first_num)
    
    # Add remaining terms
    for _ in range(num_terms - 1):
        op = random.choice(operators)
        
        # For division, avoid zero and very small divisors
        if op == '/':
            if num_digits == 1:
                num = random.randint(2, min(actual_max, 100))  # Cap at 100 for reciprocal table
            else:
                num = random.randint(max(2, 10**(num_digits-1)), min(actual_max, 100))
        else:
            if num_digits == 1:
                num = random.randint(0, actual_max)
            else:
                num = random.randint(10**(num_digits-1), actual_max)
        
        expression += op + str(num)
    
    # Convert division to multiplication with reciprocal if requested
    if use_reciprocal_for_division and '/' in expression:
        expression = convert_division_to_multiplication(expression)
    
    # Calculate answer with 4 decimal place precision (consistent across all stages)
    try:
        answer = eval(expression)
        answer_float = round(float(answer), 4)
        
        # Format with 4 decimal places
        answer_str = f"{answer_float:.4f}"
                
    except (ZeroDivisionError, ValueError, SyntaxError):
        # If evaluation fails, generate a simpler expression
        return generate_random_expression(num_digits, 2, max_value, ['+'], use_reciprocal_for_division)
    
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
    Create curriculum stages - RECIPROCAL-BASED DIVISION strategy
    Stage 0: Learn all reciprocals (1/1 to 1/100)
    Later stages: Convert divisions to multiplications with reciprocals
    """
    stages = []
    
    # STAGE 0: Learn reciprocals first (Foundation) - includes some negatives
    stages.append({
        'name': '0_Reciprocals',
        'operators': ['reciprocal'],  # Special marker
        'num_digits': 1,
        'max_value': 100,
        'max_terms': 2,
        'accuracy_threshold': 0.95,
        'description': 'Learn reciprocals including negative ones: 1/n, -1/n (4 decimals)',
        'is_reciprocal_stage': True,
        'include_negative': True  # 20% negative reciprocals
    })
    
    # PHASE 1: Single-digit operations (now using reciprocals for division)
    stages.append({
        'name': '1_Division_1digit',
        'operators': ['/'],
        'num_digits': 1,
        'max_value': 9,
        'max_terms': 2,
        'accuracy_threshold': 0.99,
        'description': 'Division as multiplication with reciprocals (e.g., 8/4 → 8*0.2)',
        'use_reciprocal_for_division': True
    })
    
    stages.append({
        'name': '2_Multiplication_1digit - positive only',
        'operators': ['*'],
        'num_digits': 1,
        'max_value': 9,
        'max_terms': 2,
        'accuracy_threshold': 0.95,
        'description': 'Master multiplication positive only (e.g., 7*8=56.0000)',
        'include_negative': False  # No negatives yet
    })
    
    ## Intermediate stage to only do simple negatives before mixing
    stages.append({
        'name': '2b_Simple_Negatives',
        'operators': ['*'],
        'num_digits': 1,
        'max_value': 10,
        'max_terms': 2,
        'accuracy_threshold': 0.95,
        'description': 'Simple negative multiplication: -1*1 through -10*10',
        'simple_negatives_only': True  # Special mode for simple negatives
    })
    
    stages.append({
        'name': '2_Multiplication_1digit - mixed negatives',
        'operators': ['*'],
        'num_digits': 1,
        'max_value': 9,
        'max_terms': 2,
        'accuracy_threshold': 0.95,
        'description': 'Master multiplication including negatives (e.g., -7*8=-56)',
        'include_negative': True  # 20% start with negative
    })
    
    stages.append({
        'name': '3_Addition_1digit',
        'operators': ['+'],
        'num_digits': 1,
        'max_value': 9,
        'max_terms': 3,
        'accuracy_threshold': 0.95,
        'description': 'Master addition including negatives (e.g., -5+3+2=0)',
        'include_negative': True  # 20% start with negative
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
    
    # Intermediate stages to gradually introduce 2-digit operations
    stages.append({
        'name': '5a_Multiplication_2digit_Simple',
        'operators': ['*'],
        'num_digits': 2,
        'max_value': 99,
        'max_terms': 2,
        'accuracy_threshold': 0.90,
        'description': 'Master 2-digit multiplication first (e.g., 12*8=96.0000)'
    })
    
    stages.append({
        'name': '5b_Division_2digit_Simple',
        'operators': ['/'],
        'num_digits': 2,
        'max_value': 99,
        'max_terms': 2,
        'accuracy_threshold': 0.70,
        'description': 'Master 2-digit division using reciprocals (e.g., 48/6 → 48*0.1667)',
        'use_reciprocal_for_division': True
    })
    
    stages.append({
        'name': '5c_Mult_Div_Mixed_2digit',
        'operators': ['*', '/'],
        'num_digits': 2,
        'max_value': 99,
        'max_terms': 2,
        'accuracy_threshold': 0.60,
        'description': 'Combine multiplication and division with 2-digit numbers (convert / to * with reciprocals)',
        'use_reciprocal_for_division': True
    })
    
    stages.append({
        'name': '6_Add_Sub_Mixed_2digit',
        'operators': ['+', '-'],
        'num_digits': 2,
        'max_value': 99,
        'max_terms': 3,
        'accuracy_threshold': 0.95,
        'description': 'Combine addition and subtraction with 2-digit numbers'
    })
    
    stages.append({
        'name': '7_All_Ops_2digit',
        'operators': ['+', '-', '*', '/'],
        'num_digits': 2,
        'max_value': 99,
        'max_terms': 3,
        'accuracy_threshold': 0.5,
        'description': 'All operations mixed with 2-digit numbers - rigorous training (CRITICAL STAGE)',
        'use_reciprocal_for_division': True
    })
    
    # PHASE 2: Advanced 2-digit operations with more terms
    stages.append({
        'name': '8_Division_2digit_Advanced',
        'operators': ['/'],
        'num_digits': 2,
        'max_value': 99,
        'max_terms': 3,
        'accuracy_threshold': 0.4,
        'description': 'Advanced 2-digit division with multiple terms (e.g., 84/7 → 84*0.14)',
        'use_reciprocal_for_division': True
    })
    
    stages.append({
        'name': '9_Multiplication_2digit_Advanced',
        'operators': ['*'],
        'num_digits': 2,
        'max_value': 99,
        'max_terms': 3,
        'accuracy_threshold': 0.8,
        'description': 'Advanced 2-digit multiplication with multiple terms (e.g., 12*8*2)'
    })
    
    stages.append({
        'name': '10_Addition_2digit_Advanced',
        'operators': ['+'],
        'num_digits': 2,
        'max_value': 99,
        'max_terms': 4,
        'accuracy_threshold': 0.9,
        'description': 'Advanced 2-digit addition with more terms (e.g., 45+38+12+5)'
    })
    
    stages.append({
        'name': '11_Subtraction_2digit_Advanced',
        'operators': ['-'],
        'num_digits': 2,
        'max_value': 99,
        'max_terms': 4,
        'accuracy_threshold': 0.9,
        'description': 'Advanced 2-digit subtraction with more terms (e.g., 78-23-15-5)'
    })
    
    stages.append({
        'name': '12_All_Ops_2digit_Master',
        'operators': ['+', '-', '*', '/'],
        'num_digits': 2,
        'max_value': 99,
        'max_terms': 4,
        'accuracy_threshold': 0.5,
        'description': 'Master all operations with 2-digit numbers and multiple terms',
        'use_reciprocal_for_division': True  # CRITICAL: Convert division to multiplication
    })
    
    return stages


def generate_expression_from_stage(stage_config):
    """Generate a single expression based on stage configuration"""
    if stage_config.get('is_reciprocal_stage', False):
        # Generate reciprocal expressions
        n = random.randint(1, 100)
        include_neg = stage_config.get('include_negative', False)
        return generate_reciprocal_expression(n, include_neg)
    else:
        use_reciprocal = stage_config.get('use_reciprocal_for_division', False)
        include_neg = stage_config.get('include_negative', False)
        simple_neg = stage_config.get('simple_negatives_only', False)
        return generate_random_expression(
            num_digits=stage_config['num_digits'],
            max_terms=stage_config['max_terms'],
            max_value=stage_config['max_value'],
            operators=stage_config['operators'],
            use_reciprocal_for_division=use_reciprocal,
            include_negative=include_neg,
            simple_negatives_only=simple_neg
        )


def get_batch(split, batch_size, block_size, curriculum_config, val_expressions, 
              encode, vocab_size, device, previous_stage_configs=None, replay_ratio=0.3):
    """Generate batches - train uses random expressions (mixing current + previous stages), val uses fixed set
    
    Args:
        split: 'train' or 'val'
        batch_size: Number of expressions in batch
        block_size: Maximum sequence length
        curriculum_config: Current stage configuration
        val_expressions: Validation set expressions
        encode: Encoding function
        vocab_size: Size of vocabulary
        device: Device to create tensors on
        previous_stage_configs: List of previous stage configs for replay (prevents forgetting)
        replay_ratio: Fraction of batch to fill with previous stage examples (default 0.3 = 30%)
    """
    batch_x = []
    batch_y = []
    
    for i in range(batch_size):
        if split == 'train':
            # CUMULATIVE CURRICULUM LEARNING: Mix current stage with previous stages
            # This prevents catastrophic forgetting
            if previous_stage_configs and len(previous_stage_configs) > 0 and random.random() < replay_ratio:
                # Sample from a previous stage (uniform random selection)
                stage_config = random.choice(previous_stage_configs)
                expression_str = generate_expression_from_stage(stage_config)
            else:
                # Sample from current stage
                expression_str = generate_expression_from_stage(curriculum_config)
        else:  # split == 'val'
            expr_idx = random.randint(0, len(val_expressions) - 1)
            expression_str = val_expressions[expr_idx]
        
        expression = expression_str + '\n'
        encoded_expr = encode(expression)
        
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
    
    # Create tensors directly (removed expensive per-token validation)
    x = torch.tensor(batch_x, dtype=torch.long, device=device)
    y = torch.tensor(batch_y, dtype=torch.long, device=device)
    
    return x, y


def update_validation_set(curriculum_config, val_set_size):
    """Generate fixed validation set for current curriculum stage"""
    val_expressions = []
    print(f"\n{'='*70}")
    print(f"GENERATING VALIDATION SET")
    print(f"{'='*70}")
    print(f"Operators: {curriculum_config['operators']}, Digits: {curriculum_config['num_digits']}")
    print(f"Max Terms: {curriculum_config['max_terms']}, Max Value: {curriculum_config['max_value']}")
    
    # Check if this is reciprocal stage
    is_reciprocal = curriculum_config.get('is_reciprocal_stage', False)
    use_reciprocal = curriculum_config.get('use_reciprocal_for_division', False)
    include_negative = curriculum_config.get('include_negative', False)
    simple_negatives = curriculum_config.get('simple_negatives_only', False)
    operators = curriculum_config['operators']
    
    if is_reciprocal:
        neg_status = "with negatives" if include_negative else "positive only"
        print(f"Mode: RECIPROCAL LEARNING (1/1 to 1/100, {neg_status})")
    elif use_reciprocal:
        print(f"Mode: DIVISION AS MULTIPLICATION (using reciprocals)")
    elif simple_negatives:
        print(f"Mode: SIMPLE NEGATIVES ONLY (-1*1 through -10*10)")
    
    if include_negative and not is_reciprocal and not simple_negatives:
        print(f"Mode: INCLUDING NEGATIVE NUMBERS (20% chance)")
    
    # For mixed operator stages, ensure balanced distribution
    if len(operators) > 1 and not is_reciprocal:
        print(f"Mode: MIXED OPERATORS - Ensuring balanced distribution")
        exprs_per_op = val_set_size // len(operators)
        remainder = val_set_size % len(operators)
        print(f"  Target: {exprs_per_op} expressions per operator ({operators})")
    
    print(f"\nFirst 10 validation expressions:")
    
    if is_reciprocal:
        # Generate reciprocals normally
        for i in range(val_set_size):
            n = random.randint(1, 100)
            expr = generate_reciprocal_expression(n, include_negative)
            val_expressions.append(expr)
            if i < 10:
                print(f"  {i+1:2d}. {expr}")
    
    elif len(operators) > 1:
        # Mixed operators - ensure balanced distribution
        exprs_per_op = val_set_size // len(operators)
        remainder = val_set_size % len(operators)
        
        # Track original operators before conversion
        op_tracking = {op: 0 for op in operators}
        
        for op_idx, op in enumerate(operators):
            count = exprs_per_op + (1 if op_idx < remainder else 0)
            for i in range(count):
                expr = generate_random_expression(
                    num_digits=curriculum_config['num_digits'],
                    max_terms=curriculum_config['max_terms'],
                    max_value=curriculum_config['max_value'],
                    operators=[op],  # Force this specific operator
                    use_reciprocal_for_division=use_reciprocal,
                    include_negative=include_negative,
                    simple_negatives_only=simple_negatives
                )
                val_expressions.append(expr)
                op_tracking[op] += 1
                if len(val_expressions) <= 10:
                    print(f"  {len(val_expressions):2d}. {expr}")
    
    else:
        # Single operator - generate normally
        for i in range(val_set_size):
            expr = generate_random_expression(
                num_digits=curriculum_config['num_digits'],
                max_terms=curriculum_config['max_terms'],
                max_value=curriculum_config['max_value'],
                operators=curriculum_config['operators'],
                use_reciprocal_for_division=use_reciprocal,
                include_negative=include_negative,
                simple_negatives_only=simple_negatives
            )
            val_expressions.append(expr)
            if i < 10:
                print(f"  {i+1:2d}. {expr}")
    
    # Shuffle to mix operators
    random.shuffle(val_expressions)
    
    # Display operator distribution for mixed operator stages
    if not is_reciprocal and len(operators) > 1 and 'op_tracking' in locals():
        print(f"\\nOperator distribution in validation set (original operators):")
        for op in operators:
            count = op_tracking[op]
            conversion_note = " (converted to * with reciprocals)" if op == '/' and use_reciprocal else ""
            print(f"  {op}: {count} expressions ({count/len(val_expressions)*100:.1f}%){conversion_note}")
    
    print(f"\n✅ Validation set ready: {len(val_expressions)} expressions")
    print(f"{'='*70}\n")
    return val_expressions
