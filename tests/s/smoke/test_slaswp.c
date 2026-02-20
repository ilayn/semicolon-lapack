/**
 * @file test_slaswp.c
 * @brief Test suite for slaswp (row interchanges for pivoting).
 *
 * Tests the row interchange routine which applies pivot permutations
 * to a matrix, typically used after LU factorization.
 *
 * Verification:
 * - Direct comparison of permuted matrix with expected result
 * - Test forward and backward permutation directions
 * - Test various matrix sizes and increment values
 */

#include "test_harness.h"

/* Test threshold - see LAPACK dtest.in */
#define THRESH 20.0f

/* Routine under test */
extern void slaswp(const int n, f32 * const restrict A, const int lda,
                   const int k1, const int k2,
                   const int * const restrict ipiv, const int incx);

/**
 * Check if two matrices are equal (within tolerance)
 */
static int matrices_equal(int m, int n, const f32 *A, int lda,
                          const f32 *B, int ldb, f32 tol)
{
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            if (fabsf(A[i + j * lda] - B[i + j * ldb]) > tol) {
                return 0;
            }
        }
    }
    return 1;
}

/**
 * Test slaswp with simple swap
 */
static void test_simple_swap(void **state)
{
    (void)state;

    /* 3x3 matrix, swap row 0 with row 2 */
    f32 A[9] = {
        1, 2, 3,   /* Column 0 */
        4, 5, 6,   /* Column 1 */
        7, 8, 9    /* Column 2 */
    };
    int lda = 3;
    int n = 3;

    /* Pivot: row 0 swaps with row 2 */
    int ipiv[1] = {2};

    /* Expected result after swap */
    f32 expected[9] = {
        3, 2, 1,   /* Row 0 and row 2 swapped in column 0 */
        6, 5, 4,   /* Row 0 and row 2 swapped in column 1 */
        9, 8, 7    /* Row 0 and row 2 swapped in column 2 */
    };

    slaswp(n, A, lda, 0, 0, ipiv, 1);

    assert_true(matrices_equal(3, 3, A, lda, expected, 3, 1e-14));
}

/**
 * Test slaswp with multiple sequential swaps
 */
static void test_multiple_swaps(void **state)
{
    (void)state;

    /* 4x3 matrix */
    f32 A[12] = {
        1, 2, 3, 4,   /* Column 0 */
        5, 6, 7, 8,   /* Column 1 */
        9, 10, 11, 12 /* Column 2 */
    };
    int lda = 4;
    int n = 3;

    /* Pivots: row 0<->1, row 1<->3, row 2<->2 (no swap) */
    int ipiv[3] = {1, 3, 2};

    /* Apply swaps from k1=0 to k2=2 */
    slaswp(n, A, lda, 0, 2, ipiv, 1);

    /* Expected: After swap 0<->1: [2,1,3,4], [6,5,7,8], [10,9,11,12]
     *           After swap 1<->3: [2,4,3,1], [6,8,7,5], [10,12,11,9]
     *           After swap 2<->2: no change */
    f32 expected[12] = {
        2, 4, 3, 1,
        6, 8, 7, 5,
        10, 12, 11, 9
    };

    assert_true(matrices_equal(4, 3, A, lda, expected, 4, 1e-14));
}

/**
 * Test slaswp with negative increment (reverse order)
 */
static void test_reverse_order(void **state)
{
    (void)state;

    /* 4x3 matrix */
    f32 A[12] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12
    };
    int lda = 4;
    int n = 3;

    /* Same pivots but applied in reverse order */
    int ipiv[3] = {1, 3, 2};

    /* Apply swaps from k2=2 to k1=0 (reverse) */
    slaswp(n, A, lda, 0, 2, ipiv, -1);

    /* With incx=-1: ix0 = k1 + (k1-k2)*incx = 0 + (0-2)*(-1) = 2
     * Loop i=2 down to 0:
     *   i=2, ix=2: swap row 2 <-> ipiv[2]=2 (no change)
     *   i=1, ix=1: swap row 1 <-> ipiv[1]=3: [1,4,3,2]
     *   i=0, ix=0: swap row 0 <-> ipiv[0]=1: [4,1,3,2] */
    f32 expected[12] = {
        4, 1, 3, 2,
        8, 5, 7, 6,
        12, 9, 11, 10
    };

    assert_true(matrices_equal(4, 3, A, lda, expected, 4, 1e-14));
}

/**
 * Test slaswp identity permutation (no swaps)
 */
static void test_identity_permutation(void **state)
{
    (void)state;

    /* 3x3 matrix */
    f32 A[9] = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    };
    f32 A_orig[9];
    memcpy(A_orig, A, 9 * sizeof(f32));
    int lda = 3;
    int n = 3;

    /* Identity permutation: each row swaps with itself */
    int ipiv[3] = {0, 1, 2};

    slaswp(n, A, lda, 0, 2, ipiv, 1);

    assert_true(matrices_equal(3, 3, A, lda, A_orig, 3, 1e-14));
}

/**
 * Test slaswp with incx=0 (should do nothing)
 */
static void test_incx_zero(void **state)
{
    (void)state;

    f32 A[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    f32 A_orig[9];
    memcpy(A_orig, A, 9 * sizeof(f32));
    int ipiv[3] = {2, 0, 1};

    slaswp(3, A, 3, 0, 2, ipiv, 0);

    assert_true(matrices_equal(3, 3, A, 3, A_orig, 3, 1e-14));
}

/**
 * Test slaswp with partial range (k1 > 0)
 */
static void test_partial_range(void **state)
{
    (void)state;

    /* 4x4 matrix */
    f32 A[16] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
    };
    int lda = 4;
    int n = 4;

    /* Only swap rows 1 and 2, leaving rows 0 and 3 alone.
     * slaswp accesses ipiv[ix] where ix starts at k1=1, so we need
     * ipiv[1] and ipiv[2] to hold the swap targets. */
    int ipiv[3] = {-1, 3, 1};  /* ipiv[1]=3: row 1<->3, ipiv[2]=1: row 2<->1 */

    slaswp(n, A, lda, 1, 2, ipiv, 1);

    /* After row 1<->3: [1,4,3,2] per column
     * After row 2<->1 (current row 1 holds original row 3):
     *   swap row 2 with row 1: [1,3,4,2] per column */

    /* Row 0 should be unchanged */
    assert_true(A[0] == 1.0f && A[0 + lda] == 5.0f &&
                A[0 + 2*lda] == 9.0f && A[0 + 3*lda] == 13.0f);

    /* Row 3 should be unchanged */
    assert_true(A[3] == 2.0f && A[3 + lda] == 6.0f &&
                A[3 + 2*lda] == 10.0f && A[3 + 3*lda] == 14.0f);

    /* Full expected result: rows 1,2 contain swapped values */
    f32 expected[16] = {
        1, 3, 4, 2,
        5, 7, 8, 6,
        9, 11, 12, 10,
        13, 15, 16, 14
    };
    assert_true(matrices_equal(4, 4, A, lda, expected, 4, 1e-14));
}

/**
 * Test slaswp with large matrix (tests blocked implementation)
 */
static void test_large_matrix(void **state)
{
    (void)state;

    int m = 100;
    int n = 64;  /* Force multiple blocks of 32 */
    int lda = m;

    f32 *A = malloc(lda * n * sizeof(f32));
    f32 *A_copy = malloc(lda * n * sizeof(f32));
    int *ipiv = malloc(m * sizeof(int));

    assert_non_null(A);
    assert_non_null(A_copy);
    assert_non_null(ipiv);

    /* Initialize matrix with distinct values */
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            A[i + j * lda] = (f32)(i * n + j + 1);
        }
    }
    memcpy(A_copy, A, lda * n * sizeof(f32));

    /* Create a cyclic permutation pattern */
    for (int i = 0; i < m; i++) {
        ipiv[i] = (i + 1) % m;  /* Each row swaps with next row */
    }

    /* Apply permutation */
    slaswp(n, A, lda, 0, m - 1, ipiv, 1);

    /* Apply reverse permutation */
    slaswp(n, A, lda, 0, m - 1, ipiv, -1);

    assert_true(matrices_equal(m, n, A, lda, A_copy, lda, 1e-14));

    free(A);
    free(A_copy);
    free(ipiv);
}

/**
 * Test slaswp consistency with LU factorization pivots
 */
static void test_lu_pivot_consistency(void **state)
{
    (void)state;

    /* After LU factorization, we typically have ipiv where ipiv[i] >= i
     * This test simulates applying LU pivots to a right-hand side */

    int n = 4;
    f32 B[4] = {1.0f, 2.0f, 3.0f, 4.0f};  /* RHS vector as 4x1 matrix */

    /* Typical LU pivot pattern: each pivot >= current row */
    int ipiv[4] = {2, 1, 3, 3};  /* row 0<->2, row 1<->1, row 2<->3, row 3<->3 */

    /* Apply forward permutation */
    slaswp(1, B, n, 0, n - 1, ipiv, 1);

    /* After swap 0<->2: [3,2,1,4]
     * After swap 1<->1: [3,2,1,4] (no change)
     * After swap 2<->3: [3,2,4,1]
     * After swap 3<->3: [3,2,4,1] (no change) */
    f32 expected[4] = {3.0f, 2.0f, 4.0f, 1.0f};

    for (int i = 0; i < n; i++) {
        assert_true(fabsf(B[i] - expected[i]) <= 1e-14);
    }
}

/**
 * Test slaswp with wide matrix (many columns)
 */
static void test_wide_matrix(void **state)
{
    (void)state;

    int m = 4;
    int n = 100;  /* More than 3 blocks of 32 */
    int lda = m;

    f32 *A = malloc(lda * n * sizeof(f32));
    f32 *expected = malloc(lda * n * sizeof(f32));
    int ipiv[4] = {3, 2, 3, 3};

    assert_non_null(A);
    assert_non_null(expected);

    /* Initialize */
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            A[i + j * lda] = (f32)(i + 1);  /* Each column is [1,2,3,4] */
        }
    }

    /* Compute expected result manually */
    /* ipiv = [3,2,3,3]: 0<->3, 1<->2, 2<->3, 3<->3 */
    /* After 0<->3: [4,2,3,1]
     * After 1<->2: [4,3,2,1]
     * After 2<->3: [4,3,1,2]
     * After 3<->3: [4,3,1,2] */
    for (int j = 0; j < n; j++) {
        expected[0 + j * lda] = 4.0f;
        expected[1 + j * lda] = 3.0f;
        expected[2 + j * lda] = 1.0f;
        expected[3 + j * lda] = 2.0f;
    }

    slaswp(n, A, lda, 0, 3, ipiv, 1);

    assert_true(matrices_equal(m, n, A, lda, expected, lda, 1e-14));

    free(A);
    free(expected);
}

int main(void)
{
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_simple_swap),
        cmocka_unit_test(test_multiple_swaps),
        cmocka_unit_test(test_reverse_order),
        cmocka_unit_test(test_identity_permutation),
        cmocka_unit_test(test_incx_zero),
        cmocka_unit_test(test_partial_range),
        cmocka_unit_test(test_large_matrix),
        cmocka_unit_test(test_lu_pivot_consistency),
        cmocka_unit_test(test_wide_matrix),
    };

    return cmocka_run_group_tests_name("dlaswp", tests, NULL, NULL);
}
