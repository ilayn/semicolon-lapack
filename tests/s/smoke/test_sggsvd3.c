/**
 * @file test_sggsvd3.c
 * @brief CMocka test suite for sggsvd3 (Generalized Singular Value Decomposition).
 *
 * Tests the GSVD routine sggsvd3 using LAPACK's verification methodology
 * with normalized residuals via sgsvts3.
 *
 * Verification (6 tests from sgsvts3):
 *   1. ||U'*A*Q - D1*R|| / (max(M,N)*||A||*ULP)
 *   2. ||V'*B*Q - D2*R|| / (max(P,N)*||B||*ULP)
 *   3. ||I - U'*U|| / (M*ULP)
 *   4. ||I - V'*V|| / (P*ULP)
 *   5. ||I - Q'*Q|| / (N*ULP)
 *   6. Check ALPHA is in decreasing order
 *
 * Test dimensions from LAPACK gsv.in:
 *   M   P   N
 *   0   4   3
 *   5   0  10
 *   9  12  15
 *  10  14  12
 *  20  10   8
 *  12  10  20
 *  12  20   8
 *  40  15  20
 */

#include "test_harness.h"
#include "test_rng.h"
#include <cblas.h>
#include <math.h>
#include <string.h>

/* Test threshold */
#define THRESH 30.0f

/* Routine under test */
extern void sggsvd3(const char* jobu, const char* jobv, const char* jobq,
                    const int m, const int n, const int p,
                    int* k, int* l,
                    f32* A, const int lda,
                    f32* B, const int ldb,
                    f32* alpha, f32* beta,
                    f32* U, const int ldu,
                    f32* V, const int ldv,
                    f32* Q, const int ldq,
                    f32* work, const int lwork,
                    int* iwork, int* info);

/* Verification routine */
extern void sgsvts3(const int m, const int p, const int n,
                    const f32* A, f32* AF, const int lda,
                    const f32* B, f32* BF, const int ldb,
                    f32* U, const int ldu,
                    f32* V, const int ldv,
                    f32* Q, const int ldq,
                    f32* alpha, f32* beta,
                    f32* R, const int ldr,
                    int* iwork,
                    f32* work, const int lwork,
                    f32* rwork,
                    f32* result);

/* Utilities */
extern f32 slamch(const char* cmach);
extern void slacpy(const char* uplo, const int m, const int n,
                   const f32* A, const int lda, f32* B, const int ldb);

/*
 * Test fixture: holds all allocated memory for a single test case.
 */
typedef struct {
    int m, p, n;
    f32* A;
    f32* AF;
    f32* B;
    f32* BF;
    f32* U;
    f32* V;
    f32* Q;
    f32* alpha;
    f32* beta;
    f32* R;
    f32* work;
    f32* rwork;
    int* iwork;
    int lwork;
    uint64_t seed;
    uint64_t rng_state[4];
} dggsvd3_fixture_t;

static uint64_t g_seed = 2024;

/**
 * Setup fixture: allocate memory for given dimensions.
 */
static int dggsvd3_setup(void** state, int m, int p, int n)
{
    dggsvd3_fixture_t* fix = malloc(sizeof(dggsvd3_fixture_t));
    assert_non_null(fix);

    fix->m = m;
    fix->p = p;
    fix->n = n;
    fix->seed = g_seed++;

    int maxmpn = m;
    if (p > maxmpn) maxmpn = p;
    if (n > maxmpn) maxmpn = n;

    int lda = (m > 1) ? m : 1;
    int ldb = (p > 1) ? p : 1;
    int ldu = (m > 1) ? m : 1;
    int ldv = (p > 1) ? p : 1;
    int ldq = (n > 1) ? n : 1;
    int ldr = (n > 1) ? n : 1;

    f32 work_query;
    int k_dummy, l_dummy, info;
    sggsvd3("U", "V", "Q", m, n, p, &k_dummy, &l_dummy,
            NULL, lda, NULL, ldb, NULL, NULL,
            NULL, ldu, NULL, ldv, NULL, ldq,
            &work_query, -1, NULL, &info);
    fix->lwork = (int)work_query;
    if (fix->lwork < maxmpn * maxmpn) fix->lwork = maxmpn * maxmpn;
    if (fix->lwork < 1) fix->lwork = 1;

    fix->A = malloc(lda * n * sizeof(f32));
    fix->AF = malloc(lda * n * sizeof(f32));
    fix->B = malloc(ldb * n * sizeof(f32));
    fix->BF = malloc(ldb * n * sizeof(f32));
    fix->U = malloc(ldu * m * sizeof(f32));
    fix->V = malloc(ldv * p * sizeof(f32));
    fix->Q = malloc(ldq * n * sizeof(f32));
    fix->alpha = malloc(n * sizeof(f32));
    fix->beta = malloc(n * sizeof(f32));
    fix->R = malloc(ldr * n * sizeof(f32));
    fix->work = malloc(fix->lwork * sizeof(f32));
    fix->rwork = malloc(maxmpn * sizeof(f32));
    fix->iwork = malloc(n * sizeof(int));

    assert_non_null(fix->A);
    assert_non_null(fix->AF);
    assert_non_null(fix->B);
    assert_non_null(fix->BF);
    assert_non_null(fix->U);
    assert_non_null(fix->V);
    assert_non_null(fix->Q);
    assert_non_null(fix->alpha);
    assert_non_null(fix->beta);
    assert_non_null(fix->R);
    assert_non_null(fix->work);
    assert_non_null(fix->rwork);
    assert_non_null(fix->iwork);

    *state = fix;
    return 0;
}

static int dggsvd3_teardown(void** state)
{
    dggsvd3_fixture_t* fix = *state;
    if (fix) {
        free(fix->A);
        free(fix->AF);
        free(fix->B);
        free(fix->BF);
        free(fix->U);
        free(fix->V);
        free(fix->Q);
        free(fix->alpha);
        free(fix->beta);
        free(fix->R);
        free(fix->work);
        free(fix->rwork);
        free(fix->iwork);
        free(fix);
    }
    return 0;
}

/* Dimension-specific setup functions from LAPACK gsv.in */
static int setup_0_4_3(void** state) { return dggsvd3_setup(state, 0, 4, 3); }
static int setup_5_0_10(void** state) { return dggsvd3_setup(state, 5, 0, 10); }
static int setup_9_12_15(void** state) { return dggsvd3_setup(state, 9, 12, 15); }
static int setup_10_14_12(void** state) { return dggsvd3_setup(state, 10, 14, 12); }
static int setup_20_10_8(void** state) { return dggsvd3_setup(state, 20, 10, 8); }
static int setup_12_10_20(void** state) { return dggsvd3_setup(state, 12, 10, 20); }
static int setup_12_20_8(void** state) { return dggsvd3_setup(state, 12, 20, 8); }
static int setup_40_15_20(void** state) { return dggsvd3_setup(state, 40, 15, 20); }

/**
 * Helper: generate random m x n matrix
 */
static void generate_random_matrix(f32* A, int m, int n, int lda,
                                   uint64_t state[static 4])
{
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            A[i + j * lda] = rng_uniform_symmetric_f32(state);
        }
    }
}

/**
 * Test workspace query (lwork = -1)
 */
static void test_workspace_query(void** state)
{
    dggsvd3_fixture_t* fix = *state;
    int m = fix->m, p = fix->p, n = fix->n;
    f32 work_query;
    int k, l, info;
    int lda = (m > 1) ? m : 1;
    int ldb = (p > 1) ? p : 1;
    int ldu = (m > 1) ? m : 1;
    int ldv = (p > 1) ? p : 1;
    int ldq = (n > 1) ? n : 1;

    sggsvd3("U", "V", "Q", m, n, p, &k, &l,
            NULL, lda, NULL, ldb, NULL, NULL,
            NULL, ldu, NULL, ldv, NULL, ldq,
            &work_query, -1, NULL, &info);
    assert_info_success(info);
    assert_true(work_query >= 1.0f);
}

/**
 * Test random well-conditioned matrices
 */
static void test_random_wellcond(void** state)
{
    dggsvd3_fixture_t* fix = *state;
    int m = fix->m, p = fix->p, n = fix->n;

    if (m == 0 && p == 0 && n == 0) {
        skip();
        return;
    }

    rng_seed(fix->rng_state, fix->seed);

    int lda = (m > 1) ? m : 1;
    int ldb = (p > 1) ? p : 1;
    int ldu = (m > 1) ? m : 1;
    int ldv = (p > 1) ? p : 1;
    int ldq = (n > 1) ? n : 1;
    int ldr = (n > 1) ? n : 1;

    generate_random_matrix(fix->A, m, n, lda, fix->rng_state);
    generate_random_matrix(fix->B, p, n, ldb, fix->rng_state);

    memset(fix->R, 0, ldr * n * sizeof(f32));

    f32 result[6];
    sgsvts3(m, p, n, fix->A, fix->AF, lda, fix->B, fix->BF, ldb,
            fix->U, ldu, fix->V, ldv, fix->Q, ldq,
            fix->alpha, fix->beta, fix->R, ldr,
            fix->iwork, fix->work, fix->lwork, fix->rwork, result);

    assert_residual_below(result[0], THRESH);
    assert_residual_below(result[1], THRESH);
    if (m > 0) assert_residual_below(result[2], THRESH);
    if (p > 0) assert_residual_below(result[3], THRESH);
    if (n > 0) assert_residual_below(result[4], THRESH);
    assert_residual_below(result[5], THRESH);
}

/**
 * Test with diagonal matrices
 */
static void test_diagonal_matrices(void** state)
{
    dggsvd3_fixture_t* fix = *state;
    int m = fix->m, p = fix->p, n = fix->n;

    if (m == 0 || p == 0 || n == 0) {
        skip();
        return;
    }

    rng_seed(fix->rng_state, fix->seed + 100);

    int lda = (m > 1) ? m : 1;
    int ldb = (p > 1) ? p : 1;
    int ldu = (m > 1) ? m : 1;
    int ldv = (p > 1) ? p : 1;
    int ldq = (n > 1) ? n : 1;
    int ldr = (n > 1) ? n : 1;

    memset(fix->A, 0, lda * n * sizeof(f32));
    memset(fix->B, 0, ldb * n * sizeof(f32));

    int minmn = (m < n) ? m : n;
    int minpn = (p < n) ? p : n;

    for (int i = 0; i < minmn; i++) {
        fix->A[i + i * lda] = rng_uniform_f32(fix->rng_state) + 0.1f;
    }
    for (int i = 0; i < minpn; i++) {
        fix->B[i + i * ldb] = rng_uniform_f32(fix->rng_state) + 0.1f;
    }

    memset(fix->R, 0, ldr * n * sizeof(f32));

    f32 result[6];
    sgsvts3(m, p, n, fix->A, fix->AF, lda, fix->B, fix->BF, ldb,
            fix->U, ldu, fix->V, ldv, fix->Q, ldq,
            fix->alpha, fix->beta, fix->R, ldr,
            fix->iwork, fix->work, fix->lwork, fix->rwork, result);

    assert_residual_below(result[0], THRESH);
    assert_residual_below(result[1], THRESH);
    assert_residual_below(result[2], THRESH);
    assert_residual_below(result[3], THRESH);
    assert_residual_below(result[4], THRESH);
    assert_residual_below(result[5], THRESH);
}

/**
 * Test with triangular matrices
 */
static void test_triangular_matrices(void** state)
{
    dggsvd3_fixture_t* fix = *state;
    int m = fix->m, p = fix->p, n = fix->n;

    if (m == 0 || p == 0 || n == 0) {
        skip();
        return;
    }

    rng_seed(fix->rng_state, fix->seed + 200);

    int lda = (m > 1) ? m : 1;
    int ldb = (p > 1) ? p : 1;
    int ldu = (m > 1) ? m : 1;
    int ldv = (p > 1) ? p : 1;
    int ldq = (n > 1) ? n : 1;
    int ldr = (n > 1) ? n : 1;

    memset(fix->A, 0, lda * n * sizeof(f32));
    memset(fix->B, 0, ldb * n * sizeof(f32));

    for (int j = 0; j < n; j++) {
        for (int i = 0; i <= j && i < m; i++) {
            fix->A[i + j * lda] = rng_uniform_symmetric_f32(fix->rng_state);
        }
        for (int i = 0; i <= j && i < p; i++) {
            fix->B[i + j * ldb] = rng_uniform_symmetric_f32(fix->rng_state);
        }
    }

    memset(fix->R, 0, ldr * n * sizeof(f32));

    f32 result[6];
    sgsvts3(m, p, n, fix->A, fix->AF, lda, fix->B, fix->BF, ldb,
            fix->U, ldu, fix->V, ldv, fix->Q, ldq,
            fix->alpha, fix->beta, fix->R, ldr,
            fix->iwork, fix->work, fix->lwork, fix->rwork, result);

    assert_residual_below(result[0], THRESH);
    assert_residual_below(result[1], THRESH);
    assert_residual_below(result[2], THRESH);
    assert_residual_below(result[3], THRESH);
    assert_residual_below(result[4], THRESH);
    assert_residual_below(result[5], THRESH);
}

/**
 * Test zero matrix for A
 */
static void test_zero_A(void** state)
{
    dggsvd3_fixture_t* fix = *state;
    int m = fix->m, p = fix->p, n = fix->n;

    if (p == 0 || n == 0) {
        skip();
        return;
    }

    rng_seed(fix->rng_state, fix->seed + 300);

    int lda = (m > 1) ? m : 1;
    int ldb = (p > 1) ? p : 1;
    int ldu = (m > 1) ? m : 1;
    int ldv = (p > 1) ? p : 1;
    int ldq = (n > 1) ? n : 1;
    int ldr = (n > 1) ? n : 1;

    memset(fix->A, 0, lda * n * sizeof(f32));
    generate_random_matrix(fix->B, p, n, ldb, fix->rng_state);

    memset(fix->R, 0, ldr * n * sizeof(f32));

    f32 result[6];
    sgsvts3(m, p, n, fix->A, fix->AF, lda, fix->B, fix->BF, ldb,
            fix->U, ldu, fix->V, ldv, fix->Q, ldq,
            fix->alpha, fix->beta, fix->R, ldr,
            fix->iwork, fix->work, fix->lwork, fix->rwork, result);

    assert_residual_below(result[0], THRESH);
    assert_residual_below(result[1], THRESH);
    if (m > 0) assert_residual_below(result[2], THRESH);
    assert_residual_below(result[3], THRESH);
    assert_residual_below(result[4], THRESH);
    assert_residual_below(result[5], THRESH);
}

/**
 * Test zero matrix for B
 */
static void test_zero_B(void** state)
{
    dggsvd3_fixture_t* fix = *state;
    int m = fix->m, p = fix->p, n = fix->n;

    if (m == 0 || n == 0) {
        skip();
        return;
    }

    rng_seed(fix->rng_state, fix->seed + 400);

    int lda = (m > 1) ? m : 1;
    int ldb = (p > 1) ? p : 1;
    int ldu = (m > 1) ? m : 1;
    int ldv = (p > 1) ? p : 1;
    int ldq = (n > 1) ? n : 1;
    int ldr = (n > 1) ? n : 1;

    generate_random_matrix(fix->A, m, n, lda, fix->rng_state);
    memset(fix->B, 0, ldb * n * sizeof(f32));

    memset(fix->R, 0, ldr * n * sizeof(f32));

    f32 result[6];
    sgsvts3(m, p, n, fix->A, fix->AF, lda, fix->B, fix->BF, ldb,
            fix->U, ldu, fix->V, ldv, fix->Q, ldq,
            fix->alpha, fix->beta, fix->R, ldr,
            fix->iwork, fix->work, fix->lwork, fix->rwork, result);

    assert_residual_below(result[0], THRESH);
    assert_residual_below(result[1], THRESH);
    assert_residual_below(result[2], THRESH);
    if (p > 0) assert_residual_below(result[3], THRESH);
    assert_residual_below(result[4], THRESH);
    assert_residual_below(result[5], THRESH);
}

/* ============== Main ============== */

int main(void)
{
    const struct CMUnitTest tests[] = {
        /* Dimension set 1: M=0, P=4, N=3 */
        cmocka_unit_test_setup_teardown(test_workspace_query, setup_0_4_3, dggsvd3_teardown),
        cmocka_unit_test_setup_teardown(test_random_wellcond, setup_0_4_3, dggsvd3_teardown),

        /* Dimension set 2: M=5, P=0, N=10 */
        cmocka_unit_test_setup_teardown(test_workspace_query, setup_5_0_10, dggsvd3_teardown),
        cmocka_unit_test_setup_teardown(test_random_wellcond, setup_5_0_10, dggsvd3_teardown),

        /* Dimension set 3: M=9, P=12, N=15 */
        cmocka_unit_test_setup_teardown(test_workspace_query, setup_9_12_15, dggsvd3_teardown),
        cmocka_unit_test_setup_teardown(test_random_wellcond, setup_9_12_15, dggsvd3_teardown),
        cmocka_unit_test_setup_teardown(test_diagonal_matrices, setup_9_12_15, dggsvd3_teardown),
        cmocka_unit_test_setup_teardown(test_triangular_matrices, setup_9_12_15, dggsvd3_teardown),
        cmocka_unit_test_setup_teardown(test_zero_A, setup_9_12_15, dggsvd3_teardown),
        cmocka_unit_test_setup_teardown(test_zero_B, setup_9_12_15, dggsvd3_teardown),

        /* Dimension set 4: M=10, P=14, N=12 */
        cmocka_unit_test_setup_teardown(test_workspace_query, setup_10_14_12, dggsvd3_teardown),
        cmocka_unit_test_setup_teardown(test_random_wellcond, setup_10_14_12, dggsvd3_teardown),
        cmocka_unit_test_setup_teardown(test_diagonal_matrices, setup_10_14_12, dggsvd3_teardown),
        cmocka_unit_test_setup_teardown(test_triangular_matrices, setup_10_14_12, dggsvd3_teardown),

        /* Dimension set 5: M=20, P=10, N=8 (tall A, short n) */
        cmocka_unit_test_setup_teardown(test_workspace_query, setup_20_10_8, dggsvd3_teardown),
        cmocka_unit_test_setup_teardown(test_random_wellcond, setup_20_10_8, dggsvd3_teardown),
        cmocka_unit_test_setup_teardown(test_diagonal_matrices, setup_20_10_8, dggsvd3_teardown),

        /* Dimension set 6: M=12, P=10, N=20 (wide matrices) */
        cmocka_unit_test_setup_teardown(test_workspace_query, setup_12_10_20, dggsvd3_teardown),
        cmocka_unit_test_setup_teardown(test_random_wellcond, setup_12_10_20, dggsvd3_teardown),
        cmocka_unit_test_setup_teardown(test_diagonal_matrices, setup_12_10_20, dggsvd3_teardown),

        /* Dimension set 7: M=12, P=20, N=8 */
        cmocka_unit_test_setup_teardown(test_workspace_query, setup_12_20_8, dggsvd3_teardown),
        cmocka_unit_test_setup_teardown(test_random_wellcond, setup_12_20_8, dggsvd3_teardown),
        cmocka_unit_test_setup_teardown(test_diagonal_matrices, setup_12_20_8, dggsvd3_teardown),

        /* Dimension set 8: M=40, P=15, N=20 (largest) */
        cmocka_unit_test_setup_teardown(test_workspace_query, setup_40_15_20, dggsvd3_teardown),
        cmocka_unit_test_setup_teardown(test_random_wellcond, setup_40_15_20, dggsvd3_teardown),
        cmocka_unit_test_setup_teardown(test_diagonal_matrices, setup_40_15_20, dggsvd3_teardown),
        cmocka_unit_test_setup_teardown(test_triangular_matrices, setup_40_15_20, dggsvd3_teardown),
        cmocka_unit_test_setup_teardown(test_zero_A, setup_40_15_20, dggsvd3_teardown),
        cmocka_unit_test_setup_teardown(test_zero_B, setup_40_15_20, dggsvd3_teardown),
    };

    return cmocka_run_group_tests_name("dggsvd3", tests, NULL, NULL);
}
