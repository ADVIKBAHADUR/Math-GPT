#!/usr/bin/env python3
"""
Test script to demonstrate the reciprocal-based division approach
"""
import sys
sys.path.insert(0, 'src')

from data import (
    RECIPROCAL_TABLE, 
    generate_reciprocal_expression,
    convert_division_to_multiplication,
    create_curriculum_stages
)

print("="*70)
print("RECIPROCAL-BASED DIVISION APPROACH DEMONSTRATION")
print("="*70)

# 1. Show reciprocal table (first 20 entries)
print("\n1. RECIPROCAL TABLE (1/1 to 1/20):")
print("-" * 70)
for i in range(1, 21):
    reciprocal = RECIPROCAL_TABLE[i]
    print(f"  1/{i:2d} = {reciprocal:.1f}", end="")
    if i % 5 == 0:
        print()

# 2. Show reciprocal expression generation
print("\n\n2. RECIPROCAL EXPRESSION GENERATION:")
print("-" * 70)
for i in [1, 2, 3, 4, 5, 7, 9, 10, 25, 50, 100]:
    expr = generate_reciprocal_expression(i)
    print(f"  {expr}")

# 3. Show division to multiplication conversion
print("\n\n3. DIVISION TO MULTIPLICATION CONVERSION:")
print("-" * 70)
test_divisions = [
    "8/4",
    "6/3", 
    "9/2",
    "10/5",
    "12/4",
    "15/3",
    "20/10",
    "7/2",
    "8/5",
    "9/4"
]

for div_expr in test_divisions:
    mult_expr = convert_division_to_multiplication(div_expr)
    div_result = eval(div_expr)
    mult_result = eval(mult_expr) if mult_expr != div_expr else div_result
    print(f"  {div_expr:8s} → {mult_expr:10s} | Original: {div_result:.1f}, Converted: {mult_result:.1f}")

# 4. Show curriculum stages
print("\n\n4. CURRICULUM STAGES:")
print("-" * 70)
stages = create_curriculum_stages()
for i, stage in enumerate(stages):
    print(f"\nStage {i}: {stage['name']}")
    print(f"  Operators: {stage['operators']}")
    print(f"  Description: {stage['description']}")
    print(f"  Accuracy Threshold: {stage['accuracy_threshold']*100:.0f}%")
    if stage.get('is_reciprocal_stage'):
        print(f"  → RECIPROCAL LEARNING STAGE")
    if stage.get('use_reciprocal_for_division'):
        print(f"  → Uses reciprocal conversion for divisions")

print("\n" + "="*70)
print("KEY INSIGHT:")
print("="*70)
print("By learning reciprocals first (1/1 to 1/100), the model can convert")
print("all division problems to multiplication with decimals:")
print("  - Stage 0: Learn 1/2 = 0.5, 1/3 = 0.3, 1/4 = 0.2, etc.")
print("  - Later: 8/4 becomes 8*0.2 (since 1/4 ≈ 0.2 at 1 decimal place)")
print("  - Model uses existing multiplication knowledge + reciprocal lookup")
print("="*70)
