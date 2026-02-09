/**
 * @file test_dlatmt_indexing.c
 * @brief Test dlatmt array indexing for various packing modes.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "test_harness.h"
#include "verify.h"
#include "test_rng.h"

/* Test that dlatmt generates correct diagonal matrices */
static void test_diagonal_matrix(void** state)
{
    (void)state;

    int n = 5;
    int lda = n;
    double* A = calloc(lda * n, sizeof(double));
    double* d = malloc(n * sizeof(double));
    double* work = malloc(3 * n * sizeof(double));
    int info;

    uint64_t rng_state[4];
    rng_seed(rng_state, 42ULL);

    /* Set known diagonal values */
    for (int i = 0; i < n; i++) {
        d[i] = (double)(i + 1);
    }

    /* Generate diagonal matrix (kl=ku=0) */
    dlatmt(n, n, "U", "N", d, 0, 1.0, 1.0, n, 0, 0, "N", A, lda, work, &info, rng_state);

    assert_info_success(info);

    /* Check diagonal elements */
    for (int i = 0; i < n; i++) {
        assert_double_equal(A[i + i * lda], d[i], 1e-14);
    }

    /* Check off-diagonal elements are zero */
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) {
            if (i != j) {
                assert_double_equal(A[i + j * lda], 0.0, 1e-14);
            }
        }
    }

    free(A);
    free(d);
    free(work);
}

/* Test non-symmetric banded matrix with Givens rotations */
static void test_nonsym_banded_givens(void** state)
{
    (void)state;

    int m = 10, n = 10;
    int kl = 2, ku = 2;
    int lda = m;
    double* A = calloc(lda * n, sizeof(double));
    double* d = malloc(n * sizeof(double));
    double* work = malloc(3 * n * sizeof(double));
    int info;

    uint64_t rng_state[4];
    rng_seed(rng_state, 43ULL);

    /* Generate matrix with mode=3 (geometric distribution of singular values) */
    dlatmt(m, n, "U", "N", d, 3, 10.0, 1.0, n, kl, ku, "N", A, lda, work, &info, rng_state);

    assert_info_success(info);

    /* Check that elements outside bandwidth are zero */
    int errors = 0;
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            int lower_dist = i - j;  /* distance below diagonal */
            int upper_dist = j - i;  /* distance above diagonal */

            if (lower_dist > kl || upper_dist > ku) {
                if (fabs(A[i + j * lda]) > 1e-14) {
                    errors++;
                }
            }
        }
    }
    assert_int_equal(errors, 0);

    free(A);
    free(d);
    free(work);
}

/* Test symmetric matrix generation */
static void test_symmetric_givens(void** state)
{
    (void)state;

    int n = 8;
    int k = 2;  /* bandwidth */
    int lda = n;
    double* A = calloc(lda * n, sizeof(double));
    double* d = malloc(n * sizeof(double));
    double* work = malloc(3 * n * sizeof(double));
    int info;

    uint64_t rng_state[4];
    rng_seed(rng_state, 44ULL);

    /* Generate symmetric matrix with small bandwidth (Givens path) */
    dlatmt(n, n, "U", "S", d, 3, 10.0, 1.0, n, k, k, "N", A, lda, work, &info, rng_state);

    assert_info_success(info);

    /* Check symmetry */
    double max_diff = 0.0;
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) {
            double diff = fabs(A[i + j * lda] - A[j + i * lda]);
            if (diff > max_diff) max_diff = diff;
        }
    }
    assert_true(max_diff < 1e-10);

    /* Check bandwidth */
    int errors = 0;
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) {
            if (abs(i - j) > k) {
                if (fabs(A[i + j * lda]) > 1e-14) {
                    errors++;
                }
            }
        }
    }
    assert_int_equal(errors, 0);

    free(A);
    free(d);
    free(work);
}

/* Test band storage format 'B' (lower triangle in band) */
static void test_band_storage_B(void** state)
{
    (void)state;

    int n = 6;
    int k = 2;  /* bandwidth */
    int lda = k + 1;  /* band storage: kl + 1 rows */
    double* A = calloc(lda * n, sizeof(double));
    double* d = malloc(n * sizeof(double));
    double* work = malloc(3 * n * sizeof(double));
    int info;

    uint64_t rng_state[4];
    rng_seed(rng_state, 45ULL);

    /* Set known eigenvalues */
    for (int i = 0; i < n; i++) {
        d[i] = (double)(n - i);
    }

    /* Generate symmetric matrix in band storage */
    dlatmt(n, n, "U", "S", d, 0, 1.0, 1.0, n, k, k, "B", A, lda, work, &info, rng_state);

    assert_info_success(info);

    /* In band storage 'B', diagonal is in first row */
    /* A(i,j) stored at A(i-j+1, j) for j <= i <= min(n, j+kl) */
    /* Check diagonal exists */
    int has_nonzero_diag = 0;
    for (int j = 0; j < n; j++) {
        if (fabs(A[0 + j * lda]) > 1e-14) {
            has_nonzero_diag = 1;
            break;
        }
    }
    assert_true(has_nonzero_diag);

    free(A);
    free(d);
    free(work);
}

/* Test band storage format 'Q' (upper triangle in band) */
static void test_band_storage_Q(void** state)
{
    (void)state;

    int n = 6;
    int k = 2;  /* bandwidth */
    int lda = k + 1;  /* band storage: ku + 1 rows */
    double* A = calloc(lda * n, sizeof(double));
    double* d = malloc(n * sizeof(double));
    double* work = malloc(3 * n * sizeof(double));
    int info;

    uint64_t rng_state[4];
    rng_seed(rng_state, 46ULL);

    /* Set known eigenvalues */
    for (int i = 0; i < n; i++) {
        d[i] = (double)(n - i);
    }

    /* Generate symmetric matrix in band storage */
    dlatmt(n, n, "U", "S", d, 0, 1.0, 1.0, n, k, k, "Q", A, lda, work, &info, rng_state);

    assert_info_success(info);

    /* In band storage 'Q', diagonal is in last row (row ku) */
    /* A(i,j) stored at A(ku+i-j+1, j) for max(1,j-ku) <= i <= j */
    /* Check diagonal exists (should be in row k = ku) */
    int has_nonzero_diag = 0;
    for (int j = 0; j < n; j++) {
        if (fabs(A[k + j * lda]) > 1e-14) {
            has_nonzero_diag = 1;
            break;
        }
    }
    assert_true(has_nonzero_diag);

    free(A);
    free(d);
    free(work);
}

/* Test full band storage format 'Z' */
static void test_band_storage_Z(void** state)
{
    (void)state;

    int n = 6;
    int kl = 2, ku = 1;
    int lda = kl + ku + 1;
    double* A = calloc(lda * n, sizeof(double));
    double* d = malloc(n * sizeof(double));
    double* work = malloc(3 * n * sizeof(double));
    int info;

    uint64_t rng_state[4];
    rng_seed(rng_state, 47ULL);

    /* Generate non-symmetric banded matrix in full band storage */
    dlatmt(n, n, "U", "N", d, 3, 10.0, 1.0, n, kl, ku, "Z", A, lda, work, &info, rng_state);

    assert_info_success(info);

    /* In 'Z' format, diagonal is at row ku (0-indexed: row ku) */
    /* A(i,j) stored at A(ku+i-j, j) */
    /* Check diagonal exists */
    int has_nonzero_diag = 0;
    for (int j = 0; j < n; j++) {
        if (fabs(A[ku + j * lda]) > 1e-14) {
            has_nonzero_diag = 1;
            break;
        }
    }
    assert_true(has_nonzero_diag);

    free(A);
    free(d);
    free(work);
}

/* Test Householder path (large bandwidth) */
static void test_householder_path(void** state)
{
    (void)state;

    int n = 10;
    int kl = 8, ku = 8;  /* Large bandwidth triggers Householder */
    int lda = n;
    double* A = calloc(lda * n, sizeof(double));
    double* d = malloc(n * sizeof(double));
    double* work = malloc(3 * n * sizeof(double));
    int info;

    uint64_t rng_state[4];
    rng_seed(rng_state, 48ULL);

    /* Generate matrix via Householder (non-symmetric) */
    dlatmt(n, n, "U", "N", d, 3, 10.0, 1.0, n, kl, ku, "N", A, lda, work, &info, rng_state);

    assert_info_success(info);

    /* Check matrix is not all zeros */
    double sum = 0.0;
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) {
            sum += fabs(A[i + j * lda]);
        }
    }
    assert_true(sum > 0.0);

    free(A);
    free(d);
    free(work);
}

/* Test packed storage 'C' (upper columnwise) */
static void test_pack_C(void** state)
{
    (void)state;

    int n = 5;
    int lda = n * (n + 1) / 2;  /* packed storage size */
    double* A = calloc(lda, sizeof(double));
    double* d = malloc(n * sizeof(double));
    double* work = malloc(3 * n * sizeof(double));
    int info;

    uint64_t rng_state[4];
    rng_seed(rng_state, 49ULL);

    /* Generate symmetric matrix in packed storage */
    dlatmt(n, n, "U", "S", d, 3, 10.0, 1.0, n, n-1, n-1, "C", A, lda, work, &info, rng_state);

    assert_info_success(info);

    /* Check not all zeros */
    double sum = 0.0;
    for (int i = 0; i < n * (n + 1) / 2; i++) {
        sum += fabs(A[i]);
    }
    assert_true(sum > 0.0);

    free(A);
    free(d);
    free(work);
}

/* Test error handling */
static void test_error_handling(void** state)
{
    (void)state;

    int n = 5;
    int lda = n;
    double* A = calloc(lda * n, sizeof(double));
    double* d = malloc(n * sizeof(double));
    double* work = malloc(3 * n * sizeof(double));
    int info;
    uint64_t rng_state[4];
    rng_seed(rng_state, 50ULL);

    /* Test invalid DIST */
    dlatmt(n, n, "X", "N", d, 0, 1.0, 1.0, n, 0, 0, "N", A, lda, work, &info, rng_state);
    assert_int_equal(info, -3);

    /* Test invalid SYM */
    dlatmt(n, n, "U", "X", d, 0, 1.0, 1.0, n, 0, 0, "N", A, lda, work, &info, rng_state);
    assert_int_equal(info, -5);

    /* Test invalid MODE */
    dlatmt(n, n, "U", "N", d, 10, 1.0, 1.0, n, 0, 0, "N", A, lda, work, &info, rng_state);
    assert_int_equal(info, -7);

    /* Test invalid COND */
    dlatmt(n, n, "U", "N", d, 3, 0.5, 1.0, n, 0, 0, "N", A, lda, work, &info, rng_state);
    assert_int_equal(info, -8);

    /* Test negative KL */
    dlatmt(n, n, "U", "N", d, 0, 1.0, 1.0, n, -1, 0, "N", A, lda, work, &info, rng_state);
    assert_int_equal(info, -10);

    /* Test KL != KU for symmetric */
    dlatmt(n, n, "U", "S", d, 0, 1.0, 1.0, n, 2, 3, "N", A, lda, work, &info, rng_state);
    assert_int_equal(info, -11);

    free(A);
    free(d);
    free(work);
}

int main(void)
{
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_diagonal_matrix),
        cmocka_unit_test(test_nonsym_banded_givens),
        cmocka_unit_test(test_symmetric_givens),
        cmocka_unit_test(test_band_storage_B),
        cmocka_unit_test(test_band_storage_Q),
        cmocka_unit_test(test_band_storage_Z),
        cmocka_unit_test(test_householder_path),
        cmocka_unit_test(test_pack_C),
        cmocka_unit_test(test_error_handling),
    };

    return cmocka_run_group_tests_name("dlatmt_indexing", tests, NULL, NULL);
}
