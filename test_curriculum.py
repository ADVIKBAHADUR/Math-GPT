"""
Test the new division-first curriculum setup
"""
import sys
sys.path.insert(0, '/root/Math-GPT/src')

from data import create_curriculum_stages, generate_random_expression

# Create the new division-first curriculum
stages = create_curriculum_stages(num_digits_list=[1, 2])

print(f"\n{'='*80}")
print(f"DIVISION-FIRST CURRICULUM - 12 STAGES")
print(f"{'='*80}\n")

for i, stage in enumerate(stages, 1):
    print(f"Stage {i:2d}: {stage['name']}")
    print(f"          Operators: {stage['operators']}")
    print(f"          Num Digits: {stage['num_digits']}, Max Value: {stage['max_value']}")
    print(f"          Max Terms: {stage.get('max_terms', 3)}, Target: {stage.get('accuracy_threshold', 0.95)*100:.0f}%")
    print(f"          {stage.get('description', '')}")
    
    # Generate 3 sample expressions
    print(f"          Sample expressions:")
    for j in range(3):
        expr = generate_random_expression(
            num_digits=stage['num_digits'],
            max_terms=stage.get('max_terms', 3),
            max_value=stage['max_value'],
            operators=stage['operators']
        )
        print(f"            {j+1}. {expr}")
    print()

print(f"{'='*80}")
print(f"\n✅ Curriculum ready! Key features:")
print(f"  • Starts with DIVISION (hardest operation)")
print(f"  • Division outputs decimals (e.g., 8/3=2.6667)")
print(f"  • Progressively adds multiplication, addition, subtraction")
print(f"  • Combines operations gradually")
print(f"  • Scales from 1-digit to 2-digit")
print(f"  • 12 total stages with different accuracy thresholds")
print(f"\n🚀 Ready to train: python src/train.py\n")
