/**
 * @file test_sgeqp3.c
 * @brief CMocka test suite for sgeqp3 (QR with column pivoting).
 *
 * Tests the QR factorization with column pivoting routine sgeqp3.
 *
 * Verification:
 *   1. ||A*P - Q*R|| / (M * ||A|| * eps)  (factorization quality)
 *   2. ||I - Q'*Q|| / (M * eps)            (orthogonality of Q)
 *   3. R has non-increasing diagonal magnitudes (pivot quality)
 */

#include "test_harness.h"
#include "verify.h"
#include "test_rng.h"

/* Test threshold - see LAPACK dtest.in */
#define THRESH 20.0f
#include "semicolon_cblas.h"

/* Routine under test */
/* Auxiliary routines for verification */
typedef struct {
    INT m, n;
    INT lda;
    f32 *A;      /* original matrix */
    f32 *AF;     /* factored output */
    f32 *Q;      /* orthogonal factor */
    f32 *R;      /* upper triangular factor */
    f32 *AP;     /* A * P (permuted original) */
    f32 *tau;
    f32 *work;
    f32 *d;
    f32 *genwork;
    INT* jpvt;
    INT lwork;
    uint64_t seed;
} qp_fixture_t;

static uint64_t g_seed = 9001;

static int qp_setup(void **state, INT m, INT n)
{
    qp_fixture_t *fix = malloc(sizeof(qp_fixture_t));
    assert_non_null(fix);

    fix->m = m;
    fix->n = n;
    INT maxmn = m > n ? m : n;
    INT minmn = m < n ? m : n;
    fix->lda = m;
    fix->seed = g_seed++;

    fix->lwork = maxmn * 64 + 3 * n;  /* generous workspace */

    fix->A = calloc(fix->lda * n, sizeof(f32));
    fix->AF = calloc(fix->lda * n, sizeof(f32));
    fix->Q = calloc(m * minmn, sizeof(f32));
    fix->R = calloc(minmn * n, sizeof(f32));
    fix->AP = calloc(fix->lda * n, sizeof(f32));
    fix->tau = calloc(minmn, sizeof(f32));
    fix->work = calloc(fix->lwork, sizeof(f32));
    fix->d = calloc(minmn, sizeof(f32));
    fix->genwork = calloc(3 * maxmn, sizeof(f32));
    fix->jpvt = calloc(n, sizeof(INT));

    assert_non_null(fix->A);
    assert_non_null(fix->AF);
    assert_non_null(fix->Q);
    assert_non_null(fix->R);
    assert_non_null(fix->AP);
    assert_non_null(fix->tau);
    assert_non_null(fix->work);
    assert_non_null(fix->d);
    assert_non_null(fix->genwork);
    assert_non_null(fix->jpvt);

    *state = fix;
    return 0;
}

static int qp_teardown(void **state)
{
    qp_fixture_t *fix = *state;
    if (fix) {
        free(fix->A);
        free(fix->AF);
        free(fix->Q);
        free(fix->R);
        free(fix->AP);
        free(fix->tau);
        free(fix->work);
        free(fix->d);
        free(fix->genwork);
        free(fix->jpvt);
        free(fix);
    }
    return 0;
}

/* Size-specific setups */
static int setup_5x5(void **state) { return qp_setup(state, 5, 5); }
static int setup_10x10(void **state) { return qp_setup(state, 10, 10); }
static int setup_20x20(void **state) { return qp_setup(state, 20, 20); }
static int setup_10x5(void **state) { return qp_setup(state, 10, 5); }
static int setup_20x10(void **state) { return qp_setup(state, 20, 10); }
static int setup_5x10(void **state) { return qp_setup(state, 5, 10); }
static int setup_10x20(void **state) { return qp_setup(state, 10, 20); }

/**
 * Verify QR with column pivoting:
 * 1. Form Q from the Householder vectors
 * 2. Extract R (upper triangular part)
 * 3. Compute A*P (permute columns of original A)
 * 4. Check ||A*P - Q*R|| / (M * ||A|| * eps)
 * 5. Check ||I - Q'*Q|| / (M * eps)
 * 6. Check |R(i,i)| >= |R(i+1,i+1)| (non-increasing diagonal)
 */
static void run_qp3_test(qp_fixture_t *fix, INT imat)
{
    INT info;
    INT m = fix->m, n = fix->n;
    INT lda = fix->lda;
    INT minmn = m < n ? m : n;
    f32 eps = slamch("E");
    char type, dist;
    INT kl, ku, mode;
    f32 anorm_param, cndnum;

    /* Generate matrix A */
    slatb4("SGE", imat, m, n, &type, &kl, &ku, &anorm_param, &mode, &cndnum, &dist);
    uint64_t rng_state[4];
    rng_seed(rng_state, fix->seed);
    slatms(m, n, &dist, &type, fix->d, mode, cndnum, anorm_param,
           kl, ku, "N", fix->A, lda, fix->genwork, &info, rng_state);
    assert_int_equal(info, 0);

    /* Copy A to AF */
    for (INT j = 0; j < n; j++)
        cblas_scopy(m, &fix->A[j * lda], 1, &fix->AF[j * lda], 1);

    /* Initialize jpvt to 0 (all free) */
    for (INT i = 0; i < n; i++)
        fix->jpvt[i] = 0;

    /* Call sgeqp3 */
    sgeqp3(m, n, fix->AF, lda, fix->jpvt, fix->tau,
            fix->work, fix->lwork, &info);
    assert_info_success(info);

    /* Form Q (m x minmn) from Householder vectors in AF */
    for (INT j = 0; j < minmn; j++)
        cblas_scopy(m, &fix->AF[j * lda], 1, &fix->Q[j * m], 1);
    sorgqr(m, minmn, minmn, fix->Q, m, fix->tau,
            fix->work, fix->lwork, &info);
    assert_info_success(info);

    /* Extract R (minmn x n) from upper triangular part of AF */
    memset(fix->R, 0, minmn * n * sizeof(f32));
    for (INT j = 0; j < n; j++) {
        INT imax = j < minmn ? j + 1 : minmn;
        for (INT i = 0; i < imax; i++) {
            fix->R[i + j * minmn] = fix->AF[i + j * lda];
        }
    }

    /* Compute A*P: permute columns of A according to jpvt (0-based) */
    for (INT j = 0; j < n; j++) {
        INT srcj = fix->jpvt[j];
        cblas_scopy(m, &fix->A[srcj * lda], 1, &fix->AP[j * lda], 1);
    }

    /* Compute AP - Q*R (store result in AP) */
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                m, n, minmn, -1.0f, fix->Q, m,
                fix->R, minmn, 1.0f, fix->AP, lda);

    /* Residual 1: ||A*P - Q*R|| / (M * ||A|| * eps) */
    f32 anorm = slange("1", m, n, fix->A, lda, NULL);
    f32 res_norm = slange("1", m, n, fix->AP, lda, NULL);
    f32 resid1;
    if (anorm == 0.0f) {
        resid1 = res_norm > 0.0f ? 1.0f / eps : 0.0f;
    } else {
        resid1 = res_norm / ((f32)m * anorm * eps);
    }
    assert_residual_ok(resid1);

    /* Residual 2: ||I - Q'*Q|| / (minmn * eps)
     * Compute Q'*Q - I */
    f32 *QtQ = fix->AP;  /* reuse AP as workspace (minmn x minmn, fits in m*n) */
    cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                minmn, minmn, m, 1.0f, fix->Q, m,
                fix->Q, m, 0.0f, QtQ, minmn);
    /* Subtract I */
    for (INT i = 0; i < minmn; i++)
        QtQ[i + i * minmn] -= 1.0f;

    f32 orth_norm = slange("1", minmn, minmn, QtQ, minmn, NULL);
    f32 resid2 = orth_norm / ((f32)minmn * eps);
    assert_residual_ok(resid2);

    /* Check 3: |R(i,i)| >= |R(i+1,i+1)| for i = 0, ..., minmn-2 */
    for (INT i = 0; i < minmn - 1; i++) {
        f32 ri = fabsf(fix->R[i + i * minmn]);
        f32 ri1 = fabsf(fix->R[(i + 1) + (i + 1) * minmn]);
        /* Allow small tolerance for floating point */
        assert_true(ri + eps * anorm >= ri1);
    }
}

/* Well-conditioned matrices */
static void test_wellcond(void **state)
{
    qp_fixture_t *fix = *state;
    for (INT imat = 1; imat <= 4; imat++) {
        fix->seed = g_seed++;
        run_qp3_test(fix, imat);
    }
}

/* Ill-conditioned matrices */
static void test_illcond(void **state)
{
    qp_fixture_t *fix = *state;
    for (INT imat = 8; imat <= 9; imat++) {
        fix->seed = g_seed++;
        run_qp3_test(fix, imat);
    }
}

/* Scaled matrices */
static void test_scaled(void **state)
{
    qp_fixture_t *fix = *state;
    for (INT imat = 10; imat <= 11; imat++) {
        fix->seed = g_seed++;
        run_qp3_test(fix, imat);
    }
}

/* Workspace query */
static void test_workspace_query(void **state)
{
    qp_fixture_t *fix = *state;
    f32 wkopt;
    INT info;
    INT* jpvt = fix->jpvt;
    for (INT i = 0; i < fix->n; i++) jpvt[i] = 0;

    sgeqp3(fix->m, fix->n, fix->AF, fix->lda, jpvt, fix->tau,
            &wkopt, -1, &info);
    assert_info_success(info);
    assert_true(wkopt >= 1.0f);
}

int main(void)
{
    const struct CMUnitTest tests[] = {
        /* Square */
        cmocka_unit_test_setup_teardown(test_wellcond, setup_5x5, qp_teardown),
        cmocka_unit_test_setup_teardown(test_wellcond, setup_10x10, qp_teardown),
        cmocka_unit_test_setup_teardown(test_wellcond, setup_20x20, qp_teardown),
        cmocka_unit_test_setup_teardown(test_illcond, setup_10x10, qp_teardown),
        cmocka_unit_test_setup_teardown(test_illcond, setup_20x20, qp_teardown),
        cmocka_unit_test_setup_teardown(test_scaled, setup_10x10, qp_teardown),
        cmocka_unit_test_setup_teardown(test_scaled, setup_20x20, qp_teardown),
        /* Tall (m > n) */
        cmocka_unit_test_setup_teardown(test_wellcond, setup_10x5, qp_teardown),
        cmocka_unit_test_setup_teardown(test_wellcond, setup_20x10, qp_teardown),
        cmocka_unit_test_setup_teardown(test_illcond, setup_20x10, qp_teardown),
        /* Wide (m < n) */
        cmocka_unit_test_setup_teardown(test_wellcond, setup_5x10, qp_teardown),
        cmocka_unit_test_setup_teardown(test_wellcond, setup_10x20, qp_teardown),
        cmocka_unit_test_setup_teardown(test_illcond, setup_10x20, qp_teardown),
        /* Workspace query */
        cmocka_unit_test_setup_teardown(test_workspace_query, setup_10x10, qp_teardown),
    };
    return cmocka_run_group_tests_name("dgeqp3", tests, NULL, NULL);
}
