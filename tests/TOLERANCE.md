# LAPACK Test Tolerance Methodology

This document explains how LAPACK tests verify correctness and what tolerances are considered acceptable.

## Overview

LAPACK uses **normalized residuals** to verify correctness. A residual measures how well a computed result satisfies the original problem. Normalization makes the residual dimensionless and comparable across different problem sizes and matrix norms.

## Threshold Value

The default threshold is **30.0** (from `dtest.in`). A test passes if:
```
residual < THRESH (30.0)
```

This threshold accounts for:
- Accumulation of rounding errors across many operations
- Variation across different matrix types and sizes
- Safety margin for edge cases

## Verification Routines

### dget01: LU Factorization Verification

**Purpose**: Verify that L*U ≈ P*A (factorization is accurate)

**Formula**:
```
RESID = ||L*U - A|| / (N * ||A|| * EPS)
```

Where:
- `||·||` is the 1-norm (maximum column sum)
- `N` is the matrix dimension
- `EPS` is machine epsilon (~1.1e-16 for double precision)

**Interpretation**:
- RESID < 1: Error is smaller than N * ||A|| * EPS (excellent)
- RESID < 30: Error is within 30× the theoretical minimum (acceptable)
- RESID ≥ 30: Potential bug or numerical issue

### dget02: Solution Residual Verification

**Purpose**: Verify that the computed solution X satisfies A*X ≈ B

**Formula**:
```
RESID = ||B - A*X|| / (||A|| * ||X|| * EPS)
```

**Interpretation**:
- Measures backward error normalized by problem scale
- Should be O(1) for a stable algorithm
- Values < 30 indicate acceptable accuracy

### dget03: Matrix Inverse Verification

**Purpose**: Verify that A * A^(-1) ≈ I

**Formula**:
```
RESID = ||I - AINV*A|| / (N * ||A|| * ||AINV|| * EPS)
```

**Also computes**:
```
RCOND = (1/||A||) / ||AINV||
```

**Interpretation**:
- Small RESID means accurate inverse computation
- RCOND gives true reciprocal condition number

### dget04: Solution Accuracy vs Known Solution

**Purpose**: Compare computed solution to known exact solution

**Formula**:
```
RESID = (||X - XACT||_∞ * RCOND) / (||XACT||_∞ * EPS)
```

**Interpretation**:
- Scales error by condition number (ill-conditioned problems can have larger errors)
- Final division by EPS normalizes to unit roundoff

### dget06: Condition Number Comparison

**Purpose**: Compare estimated RCOND to true RCONDC

**Formula**:
```
RAT = max(RCOND, RCONDC) / min(RCOND, RCONDC) - (1 - EPS)
```

**Interpretation**:
- RAT ≈ 0 means perfect estimate
- RAT < 30 means estimate within factor of 30 (acceptable for condition estimation)

### dget07: Error Bound Verification

**Purpose**: Verify that computed error bounds (FERR, BERR) are valid

**Formula** (for forward error):
```
RESLTS[0] = ||X - XACT||_∞ / (||X||_∞ * FERR)
```

**Interpretation**:
- RESLTS[0] < 1 means FERR is a valid upper bound on the error
- RESLTS[0] ≥ 1 means FERR underestimates the error (bug!)

## Matrix Types for Testing (DGE)

LAPACK tests 11 matrix types for general matrices:

| Type | Description | Condition Number |
|------|-------------|------------------|
| 1 | Diagonal | 2 |
| 2 | Upper triangular | 2 |
| 3 | Lower triangular | 2 |
| 4 | Random, full | 2 |
| 5 | First column zero | 2 |
| 6 | Last column zero | 2 |
| 7 | Middle columns zero | 2 |
| 8 | Random, ill-conditioned | sqrt(0.1/EPS) ≈ 3e7 |
| 9 | Random, very ill-conditioned | 0.1/EPS ≈ 9e15 |
| 10 | Random, scaled near underflow | 2 |
| 11 | Random, scaled near overflow | 2 |

The "BADC" condition numbers are:
```c
BADC1 = sqrt(0.1 / EPS)  ≈ 3e7   (moderately ill-conditioned)
BADC2 = 0.1 / EPS        ≈ 9e15  (severely ill-conditioned)
```

## Why Threshold = 30?

The threshold of 30 is empirical, chosen to:

1. **Allow for algorithm variation**: Different blocking strategies, pivoting choices
2. **Handle near-singular matrices**: Types 8-9 have extreme condition numbers
3. **Provide margin for implementation differences**: Compiler optimizations, BLAS implementations
4. **Not be too loose**: Catches genuine bugs while avoiding false positives

## Practical Guidelines

For our implementation:

1. **Residuals < 1.0**: Excellent accuracy, meeting theoretical bounds
2. **Residuals 1-10**: Good accuracy, typical for well-conditioned problems
3. **Residuals 10-30**: Acceptable, often seen for ill-conditioned problems
4. **Residuals > 30**: Investigate - may indicate bug or extreme conditions

## Testing Strategy

1. **Well-conditioned matrices (cond ≈ 2)**: Expect residuals < 5
2. **Moderately ill-conditioned (cond ≈ 3e7)**: Allow residuals up to 20
3. **Severely ill-conditioned (cond ≈ 9e15)**: May see residuals up to 30
4. **Random matrices**: Should pass with residuals < 10 typically
