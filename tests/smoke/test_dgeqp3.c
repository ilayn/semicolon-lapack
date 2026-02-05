/**
 * @file test_dgeqp3.c
 * @brief CMocka test suite for dgeqp3 (QR with column pivoting).
 *
 * Tests the QR factorization with column pivoting routine dgeqp3.
 *
 * Verification:
 *   1. ||A*P - Q*R|| / (M * ||A|| * eps)  (factorization quality)
 *   2. ||I - Q'*Q|| / (M * eps)            (orthogonality of Q)
 *   3. R has non-increasing diagonal magnitudes (pivot quality)
 */

#include "test_harness.h"

/* Test threshold - see LAPACK dtest.in */
#define THRESH 20.0
#include <cblas.h>

/* Routine under test */
extern void dgeqp3(const int m, const int n,
                   double * restrict A, const int lda,
                   int * restrict jpvt,
                   double * restrict tau,
                   double * restrict work, const int lwork,
                   int *info);

/* Auxiliary routines for verification */
extern void dorgqr(const int m, const int n, const int k,
                   double * restrict A, const int lda,
                   const double * restrict tau,
                   double * restrict work, const int lwork,
                   int *info);
extern double dlange(const char *norm, const int m, const int n,
                     const double * restrict A, const int lda,
                     double *work);
extern double dlamch(const char *cmach);

/* Matrix generation */
extern void dlatb4(const char *path, const int imat, const int m, const int n,
                   char *type, int *kl, int *ku, double *anorm, int *mode,
                   double *cndnum, char *dist);
extern void dlatms(const int m, const int n, const char *dist,
                   uint64_t seed, const char *sym, double *d,
                   const int mode, const double cond, const double dmax,
                   const int kl, const int ku, const char *pack,
                   double *A, const int lda, double *work, int *info);

typedef struct {
    int m, n;
    int lda;
    double *A;      /* original matrix */
    double *AF;     /* factored output */
    double *Q;      /* orthogonal factor */
    double *R;      /* upper triangular factor */
    double *AP;     /* A * P (permuted original) */
    double *tau;
    double *work;
    double *d;
    double *genwork;
    int *jpvt;
    int lwork;
    uint64_t seed;
} qp_fixture_t;

static uint64_t g_seed = 9001;

static int qp_setup(void **state, int m, int n)
{
    qp_fixture_t *fix = malloc(sizeof(qp_fixture_t));
    assert_non_null(fix);

    fix->m = m;
    fix->n = n;
    int maxmn = m > n ? m : n;
    int minmn = m < n ? m : n;
    fix->lda = m;
    fix->seed = g_seed++;

    fix->lwork = maxmn * 64 + 3 * n;  /* generous workspace */

    fix->A = calloc(fix->lda * n, sizeof(double));
    fix->AF = calloc(fix->lda * n, sizeof(double));
    fix->Q = calloc(m * minmn, sizeof(double));
    fix->R = calloc(minmn * n, sizeof(double));
    fix->AP = calloc(fix->lda * n, sizeof(double));
    fix->tau = calloc(minmn, sizeof(double));
    fix->work = calloc(fix->lwork, sizeof(double));
    fix->d = calloc(minmn, sizeof(double));
    fix->genwork = calloc(3 * maxmn, sizeof(double));
    fix->jpvt = calloc(n, sizeof(int));

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
static void run_qp3_test(qp_fixture_t *fix, int imat)
{
    int info;
    int m = fix->m, n = fix->n;
    int lda = fix->lda;
    int minmn = m < n ? m : n;
    double eps = dlamch("E");
    char type, dist;
    int kl, ku, mode;
    double anorm_param, cndnum;

    /* Generate matrix A */
    dlatb4("DGE", imat, m, n, &type, &kl, &ku, &anorm_param, &mode, &cndnum, &dist);
    dlatms(m, n, &dist, fix->seed, &type, fix->d, mode, cndnum, anorm_param,
           kl, ku, "N", fix->A, lda, fix->genwork, &info);
    assert_int_equal(info, 0);

    /* Copy A to AF */
    for (int j = 0; j < n; j++)
        cblas_dcopy(m, &fix->A[j * lda], 1, &fix->AF[j * lda], 1);

    /* Initialize jpvt to 0 (all free) */
    for (int i = 0; i < n; i++)
        fix->jpvt[i] = 0;

    /* Call dgeqp3 */
    dgeqp3(m, n, fix->AF, lda, fix->jpvt, fix->tau,
            fix->work, fix->lwork, &info);
    assert_info_success(info);

    /* Form Q (m x minmn) from Householder vectors in AF */
    for (int j = 0; j < minmn; j++)
        cblas_dcopy(m, &fix->AF[j * lda], 1, &fix->Q[j * m], 1);
    dorgqr(m, minmn, minmn, fix->Q, m, fix->tau,
            fix->work, fix->lwork, &info);
    assert_info_success(info);

    /* Extract R (minmn x n) from upper triangular part of AF */
    memset(fix->R, 0, minmn * n * sizeof(double));
    for (int j = 0; j < n; j++) {
        int imax = j < minmn ? j + 1 : minmn;
        for (int i = 0; i < imax; i++) {
            fix->R[i + j * minmn] = fix->AF[i + j * lda];
        }
    }

    /* Compute A*P: permute columns of A according to jpvt (0-based) */
    for (int j = 0; j < n; j++) {
        int srcj = fix->jpvt[j];
        cblas_dcopy(m, &fix->A[srcj * lda], 1, &fix->AP[j * lda], 1);
    }

    /* Compute AP - Q*R (store result in AP) */
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                m, n, minmn, -1.0, fix->Q, m,
                fix->R, minmn, 1.0, fix->AP, lda);

    /* Residual 1: ||A*P - Q*R|| / (M * ||A|| * eps) */
    double anorm = dlange("1", m, n, fix->A, lda, NULL);
    double res_norm = dlange("1", m, n, fix->AP, lda, NULL);
    double resid1;
    if (anorm == 0.0) {
        resid1 = res_norm > 0.0 ? 1.0 / eps : 0.0;
    } else {
        resid1 = res_norm / ((double)m * anorm * eps);
    }
    assert_residual_ok(resid1);

    /* Residual 2: ||I - Q'*Q|| / (minmn * eps)
     * Compute Q'*Q - I */
    double *QtQ = fix->AP;  /* reuse AP as workspace (minmn x minmn, fits in m*n) */
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                minmn, minmn, m, 1.0, fix->Q, m,
                fix->Q, m, 0.0, QtQ, minmn);
    /* Subtract I */
    for (int i = 0; i < minmn; i++)
        QtQ[i + i * minmn] -= 1.0;

    double orth_norm = dlange("1", minmn, minmn, QtQ, minmn, NULL);
    double resid2 = orth_norm / ((double)minmn * eps);
    assert_residual_ok(resid2);

    /* Check 3: |R(i,i)| >= |R(i+1,i+1)| for i = 0, ..., minmn-2 */
    for (int i = 0; i < minmn - 1; i++) {
        double ri = fabs(fix->R[i + i * minmn]);
        double ri1 = fabs(fix->R[(i + 1) + (i + 1) * minmn]);
        /* Allow small tolerance for floating point */
        assert_true(ri + eps * anorm >= ri1);
    }
}

/* Well-conditioned matrices */
static void test_wellcond(void **state)
{
    qp_fixture_t *fix = *state;
    for (int imat = 1; imat <= 4; imat++) {
        fix->seed = g_seed++;
        run_qp3_test(fix, imat);
    }
}

/* Ill-conditioned matrices */
static void test_illcond(void **state)
{
    qp_fixture_t *fix = *state;
    for (int imat = 8; imat <= 9; imat++) {
        fix->seed = g_seed++;
        run_qp3_test(fix, imat);
    }
}

/* Scaled matrices */
static void test_scaled(void **state)
{
    qp_fixture_t *fix = *state;
    for (int imat = 10; imat <= 11; imat++) {
        fix->seed = g_seed++;
        run_qp3_test(fix, imat);
    }
}

/* Workspace query */
static void test_workspace_query(void **state)
{
    qp_fixture_t *fix = *state;
    double wkopt;
    int info;
    int *jpvt = fix->jpvt;
    for (int i = 0; i < fix->n; i++) jpvt[i] = 0;

    dgeqp3(fix->m, fix->n, fix->AF, fix->lda, jpvt, fix->tau,
            &wkopt, -1, &info);
    assert_info_success(info);
    assert_true(wkopt >= 1.0);
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
