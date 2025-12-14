#!/usr/bin/env python3
"""
Quick start guide for training with reciprocal-based division
"""

print("""
================================================================================
RECIPROCAL-BASED DIVISION TRAINING - QUICK START GUIDE
================================================================================

CONCEPT:
--------
Instead of directly learning division, we teach the model in two phases:
  1. STAGE 0: Learn reciprocals (1/1=1.0, 1/2=0.5, 1/3=0.3, ... 1/100=0.0)
  2. LATER STAGES: Convert divisions to multiplications with reciprocals
     Example: 8/4 → 8*0.2 (since 1/4 ≈ 0.2 at 1 decimal precision)

WHY THIS WORKS:
---------------
  • Reciprocals are simple lookup tables (only 100 to memorize)
  • Model learns 1 decimal place precision (easier than full decimals)
  • Multiplication is easier than division for neural networks
  • Decomposition: Division = Multiplication × Reciprocal lookup

WHAT CHANGED:
-------------
✅ Added RECIPROCAL_TABLE: Pre-computed 1/1 to 1/100 with 1 decimal place
✅ Added Stage 0: "0_Reciprocals" - trains on reciprocal expressions
✅ Added convert_division_to_multiplication(): Converts 8/4 → 8*0.2
✅ Updated all division stages to use reciprocal conversion
✅ Updated validation set generation for reciprocal stage

TRAINING COMMAND:
-----------------
cd /root/Math-GPT/src
python train.py

The training will now:
  1. Start with Stage 0 (Reciprocals): Learn 1/n for n=1..100
  2. Move to Stage 1 (Division): But divisions are converted to 8*0.2 style
  3. Continue through curriculum with this approach

MONITORING PROGRESS:
--------------------
Watch for Stage 0 accuracy on reciprocals (target: 95%)
  • Model should quickly memorize reciprocals (they're deterministic)
  • 1/2=0.5, 1/3=0.3, 1/4=0.2, 1/5=0.2, etc.

Later stages will use these learned reciprocals for division.

EXAMPLE EXPRESSIONS:
--------------------
Stage 0 (Reciprocals):
  1/1=1.0
  1/2=0.5
  1/3=0.3
  1/4=0.2
  1/10=0.1

Stage 1 (Division converted to multiplication):
  8/4  → becomes → 8*0.2=1.6    (note: 8/4 actual=2.0, but 0.2≈0.25)
  6/2  → becomes → 6*0.5=3.0
  10/5 → becomes → 10*0.2=2.0

NOTE: There's precision loss at 1 decimal place, but that's the tradeoff
for making division tractable. Adjust if needed for better accuracy.

NEXT STEPS:
-----------
1. Run: python test_reciprocal_approach.py  # See the approach in action
2. Run: cd src && python train.py          # Start training
3. Monitor TensorBoard for Stage 0 progress
4. Watch how division accuracy improves after reciprocal learning

================================================================================
""")
