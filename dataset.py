import random

def generate_math_dataset(num_datapoints, max_terms, max_number):
    """
    Generate arithmetic expressions with answers.
    
    Parameters:
    - num_datapoints: Total number of expressions to generate
    - max_terms: Maximum number of terms/numbers in expression
    - max_number: Maximum value for numbers (e.g., 99 for two-digit)
    """
    operators = ['-']
    dataset = []
    failed = 0
    for i in range(num_datapoints):
        # Random number of terms (at least 2)
        num_terms = random.randint(2, max_terms)
        
        expression = ""
        open_brackets = 0
        
        for j in range(num_terms):
            # Randomly decide to open bracket before this term
            if random.random() < 0.0 and open_brackets == 0 and j > 0 and j < num_terms - 1:
                expression += "("
                open_brackets += 1
            
            # Add a number
            num = random.randint(1, max_number)
            expression += str(num)
            
            # If not the last term, add an operator
            if j < num_terms - 1:
                # Add operator
                op = random.choice(operators)
                expression += op
                
                # Randomly close bracket after operator (if one is open)
                if open_brackets > 0 and random.random() < 0.5 and j >= 1:
                    expression += ")"
                    open_brackets -= 1
        
        # Close any remaining open brackets
        while open_brackets > 0:
            expression += ")"
            open_brackets -= 1
        
        # Evaluate the expression
        try:
            answer = eval(expression)
            # Skip if division resulted in non-integer or very large number
            answer = round(float(answer), 4)
            dataset.append(f"{expression}={answer}")
        except:
            print(f"Failed to evaluate: {expression}")
            failed += 1
    
    print(f"\nGenerated {len(dataset)} valid expressions")
    print(f"Failed: {failed} expressions")
    
    return dataset

# Generate dataset
data = generate_math_dataset(
    num_datapoints=1000000,
    max_terms=4,
    max_number=100
)

# Save to file
with open('math_dataset.txt', 'w') as f:
    for line in data:
        f.write(line + '\n')

print("Dataset saved to math_dataset.txt")
print(f"Sample expressions:\n")
for i in range(5):
    print(data[i])
