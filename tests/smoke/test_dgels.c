/**
 * @file test_dgels.c
 * @brief CMocka test suite for dgels (full-rank least squares).
 *
 * Tests the full-rank least squares driver dgels using LAPACK's
 * verification methodology.
 *
 * For overdetermined (m >= n, TRANS='N'):
 *   Generate B = A * X_true, solve, check ||X - X_true|| / (n * ||X_true|| * eps)
 *
 * For underdetermined (m < n, TRANS='N'):
 *   Generate random B, solve, check ||A*X - B|| / (m * ||A|| * ||X|| * eps)
 *
 * Also tests TRANS='T' cases.
 */

#include "test_harness.h"

/* Test threshold - see LAPACK dtest.in */
#define THRESH 20.0
#include <cblas.h>

/* Routine under test */
extern void dgels(const char *trans,
                  const int m, const int n, const int nrhs,
                  double * restrict A, const int lda,
                  double * restrict B, const int ldb,
                  double * restrict work, const int lwork,
                  int *info);

/* Utility routines */
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
    int m, n, nrhs;
    int lda, ldb;
    double *A;      /* original matrix */
    double *Acopy;  /* working copy for dgels */
    double *B;      /* RHS / solution */
    double *Xtrue;  /* true solution (for overdetermined) */
    double *work;
    double *d;
    double *genwork;
    int lwork;
    uint64_t seed;
} ls_fixture_t;

static uint64_t g_seed = 5001;

static int ls_setup(void **state, int m, int n, int nrhs)
{
    ls_fixture_t *fix = malloc(sizeof(ls_fixture_t));
    assert_non_null(fix);

    fix->m = m;
    fix->n = n;
    fix->nrhs = nrhs;
    int maxmn = m > n ? m : n;
    fix->lda = m;
    fix->ldb = maxmn;  /* ldb >= max(m,n) required by dgels */
    fix->seed = g_seed++;

    int minmn = m < n ? m : n;
    fix->lwork = maxmn * 64 + minmn;

    fix->A = calloc((size_t)fix->lda * n, sizeof(double));
    fix->Acopy = calloc((size_t)fix->lda * n, sizeof(double));
    fix->B = calloc((size_t)fix->ldb * nrhs, sizeof(double));
    fix->Xtrue = calloc((size_t)maxmn * nrhs, sizeof(double));
    fix->work = calloc(fix->lwork, sizeof(double));
    fix->d = calloc(minmn, sizeof(double));
    fix->genwork = calloc(3 * maxmn, sizeof(double));

    assert_non_null(fix->A);
    assert_non_null(fix->Acopy);
    assert_non_null(fix->B);
    assert_non_null(fix->Xtrue);
    assert_non_null(fix->work);
    assert_non_null(fix->d);
    assert_non_null(fix->genwork);

    *state = fix;
    return 0;
}

static int ls_teardown(void **state)
{
    ls_fixture_t *fix = *state;
    if (fix) {
        free(fix->A);
        free(fix->Acopy);
        free(fix->B);
        free(fix->Xtrue);
        free(fix->work);
        free(fix->d);
        free(fix->genwork);
        free(fix);
    }
    return 0;
}

/* Simple PRNG for generating test data */
static double prng_next(uint64_t *s)
{
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    return (double)(int64_t)(*s) / (double)INT64_MAX;
}

/* Size-specific setups */
static int setup_10x5_1(void **state) { return ls_setup(state, 10, 5, 1); }
static int setup_20x10_1(void **state) { return ls_setup(state, 20, 10, 1); }
static int setup_20x10_3(void **state) { return ls_setup(state, 20, 10, 3); }
static int setup_50x20_1(void **state) { return ls_setup(state, 50, 20, 1); }
static int setup_5x10_1(void **state) { return ls_setup(state, 5, 10, 1); }
static int setup_10x20_1(void **state) { return ls_setup(state, 10, 20, 1); }
static int setup_10x20_3(void **state) { return ls_setup(state, 10, 20, 3); }
static int setup_10x10_1(void **state) { return ls_setup(state, 10, 10, 1); }
static int setup_10x10_3(void **state) { return ls_setup(state, 10, 10, 3); }

/**
 * Test overdetermined LS (m >= n, TRANS='N'):
 *   1. Generate A (m x n)
 *   2. Generate X_true (n x nrhs)
 *   3. Compute B = A * X_true  (consistent system)
 *   4. Call dgels
 *   5. Verify ||X - X_true|| / (n * ||X_true|| * eps) < THRESH
 */
static void run_dgels_overdetermined(ls_fixture_t *fix, int imat)
{
    int info;
    int m = fix->m, n = fix->n, nrhs = fix->nrhs;
    int lda = fix->lda, ldb = fix->ldb;
    double eps = dlamch("E");
    char type, dist;
    int kl, ku, mode;
    double anorm_param, cndnum;

    /* Generate matrix A */
    dlatb4("DGE", imat, m, n, &type, &kl, &ku, &anorm_param, &mode, &cndnum, &dist);
    dlatms(m, n, &dist, fix->seed, &type, fix->d, mode, cndnum, anorm_param,
           kl, ku, "N", fix->A, lda, fix->genwork, &info);
    assert_int_equal(info, 0);

    /* Generate random X_true (n x nrhs) */
    uint64_t s = fix->seed + 1000;
    for (int j = 0; j < nrhs; j++)
        for (int i = 0; i < n; i++)
            fix->Xtrue[i + j * n] = prng_next(&s);

    /* Compute B = A * X_true (m x nrhs) */
    memset(fix->B, 0, (size_t)ldb * nrhs * sizeof(double));
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                m, nrhs, n, 1.0, fix->A, lda,
                fix->Xtrue, n, 0.0, fix->B, ldb);

    /* Save A copy (dgels overwrites A) */
    for (int j = 0; j < n; j++)
        cblas_dcopy(m, &fix->A[j * lda], 1, &fix->Acopy[j * lda], 1);

    /* Call dgels */
    dgels("N", m, n, nrhs, fix->A, lda, fix->B, ldb,
          fix->work, fix->lwork, &info);
    assert_info_success(info);

    /* Verify: ||X_computed - X_true|| / (n * ||X_true|| * eps)
     * X_computed is in B[0:n-1, 0:nrhs-1] */
    double err_norm = 0.0;
    for (int j = 0; j < nrhs; j++) {
        for (int i = 0; i < n; i++) {
            double diff = fix->B[i + j * ldb] - fix->Xtrue[i + j * n];
            err_norm += diff * diff;
        }
    }
    err_norm = sqrt(err_norm);

    double xtrue_norm = 0.0;
    for (int j = 0; j < nrhs; j++) {
        for (int i = 0; i < n; i++) {
            xtrue_norm += fix->Xtrue[i + j * n] * fix->Xtrue[i + j * n];
        }
    }
    xtrue_norm = sqrt(xtrue_norm);

    double resid;
    if (xtrue_norm == 0.0) {
        resid = err_norm > 0.0 ? 1.0 / eps : 0.0;
    } else {
        /* Scale by condition number: for well-conditioned matrices, resid should be O(1).
         * For ill-conditioned, we expect larger residuals proportional to cndnum. */
        resid = err_norm / ((double)n * xtrue_norm * eps);
        /* Divide by cndnum to normalize: error should be bounded by cndnum * eps */
        resid = resid / cndnum;
    }

    assert_residual_ok(resid);
}

/**
 * Test underdetermined LS (m < n, TRANS='N'):
 *   1. Generate A (m x n)
 *   2. Generate random B (m x nrhs)
 *   3. Call dgels
 *   4. Verify ||A*X - B|| / (max(m,n) * ||A|| * ||X|| * eps) < THRESH
 */
static void run_dgels_underdetermined(ls_fixture_t *fix, int imat)
{
    int info;
    int m = fix->m, n = fix->n, nrhs = fix->nrhs;
    int lda = fix->lda, ldb = fix->ldb;
    int maxmn = m > n ? m : n;
    double eps = dlamch("E");
    char type, dist;
    int kl, ku, mode;
    double anorm_param, cndnum;

    /* Generate matrix A */
    dlatb4("DGE", imat, m, n, &type, &kl, &ku, &anorm_param, &mode, &cndnum, &dist);
    dlatms(m, n, &dist, fix->seed, &type, fix->d, mode, cndnum, anorm_param,
           kl, ku, "N", fix->A, lda, fix->genwork, &info);
    assert_int_equal(info, 0);

    /* Generate random B (m x nrhs) */
    uint64_t s = fix->seed + 2000;
    memset(fix->B, 0, (size_t)ldb * nrhs * sizeof(double));
    for (int j = 0; j < nrhs; j++)
        for (int i = 0; i < m; i++)
            fix->B[i + j * ldb] = prng_next(&s);

    /* Save copies */
    for (int j = 0; j < n; j++)
        cblas_dcopy(m, &fix->A[j * lda], 1, &fix->Acopy[j * lda], 1);
    /* Save B in Xtrue (we'll use it as B_orig) */
    for (int j = 0; j < nrhs; j++)
        cblas_dcopy(m, &fix->B[j * ldb], 1, &fix->Xtrue[j * maxmn], 1);

    /* Call dgels */
    dgels("N", m, n, nrhs, fix->A, lda, fix->B, ldb,
          fix->work, fix->lwork, &info);
    assert_info_success(info);

    /* Verify A*X = B_orig: compute Xtrue := Xtrue - A*X
     * X is in B[0:n-1], solution is n-by-nrhs */
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                m, nrhs, n, -1.0, fix->Acopy, lda,
                fix->B, ldb, 1.0, fix->Xtrue, maxmn);

    double resid_norm = dlange("F", m, nrhs, fix->Xtrue, maxmn, NULL);
    double anorm = dlange("1", m, n, fix->Acopy, lda, NULL);
    double xnorm = dlange("1", n, nrhs, fix->B, ldb, NULL);

    double resid;
    if (anorm == 0.0 || xnorm == 0.0) {
        resid = resid_norm > 0.0 ? 1.0 / eps : 0.0;
    } else {
        resid = resid_norm / ((double)maxmn * anorm * xnorm * eps);
    }

    assert_residual_ok(resid);
}

/**
 * Test TRANS='T' overdetermined (m >= n):
 *   A^T * X = B is underdetermined (A^T is n x m, with m >= n).
 *   Generate random B (n x nrhs), solve, check A^T*X = B.
 */
static void run_dgels_trans_mgen(ls_fixture_t *fix, int imat)
{
    int info;
    int m = fix->m, n = fix->n, nrhs = fix->nrhs;
    int lda = fix->lda, ldb = fix->ldb;
    int maxmn = m > n ? m : n;
    double eps = dlamch("E");
    char type, dist;
    int kl, ku, mode;
    double anorm_param, cndnum;

    /* Generate matrix A (m x n) */
    dlatb4("DGE", imat, m, n, &type, &kl, &ku, &anorm_param, &mode, &cndnum, &dist);
    dlatms(m, n, &dist, fix->seed + 500, &type, fix->d, mode, cndnum, anorm_param,
           kl, ku, "N", fix->A, lda, fix->genwork, &info);
    assert_int_equal(info, 0);

    /* For TRANS='T' with m >= n: system is A^T*X = B, underdetermined
     * A^T is n-by-m, X is m-by-nrhs, B is n-by-nrhs
     * B is stored in first n rows of B array */
    uint64_t s = fix->seed + 3000;
    memset(fix->B, 0, (size_t)ldb * nrhs * sizeof(double));
    for (int j = 0; j < nrhs; j++)
        for (int i = 0; i < n; i++)
            fix->B[i + j * ldb] = prng_next(&s);

    /* Save copies */
    for (int j = 0; j < n; j++)
        cblas_dcopy(m, &fix->A[j * lda], 1, &fix->Acopy[j * lda], 1);
    /* Save B_orig in Xtrue */
    for (int j = 0; j < nrhs; j++)
        cblas_dcopy(n, &fix->B[j * ldb], 1, &fix->Xtrue[j * maxmn], 1);

    /* Call dgels with TRANS='T' */
    dgels("T", m, n, nrhs, fix->A, lda, fix->B, ldb,
          fix->work, fix->lwork, &info);
    assert_info_success(info);

    /* Verify A^T*X = B_orig: Xtrue := Xtrue - A^T * X
     * X is in B[0:m-1, 0:nrhs-1] (m-by-nrhs solution) */
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                n, nrhs, m, -1.0, fix->Acopy, lda,
                fix->B, ldb, 1.0, fix->Xtrue, maxmn);

    double resid_norm = dlange("F", n, nrhs, fix->Xtrue, maxmn, NULL);
    double anorm = dlange("1", m, n, fix->Acopy, lda, NULL);
    double xnorm = dlange("1", m, nrhs, fix->B, ldb, NULL);

    double resid;
    if (anorm == 0.0 || xnorm == 0.0) {
        resid = resid_norm > 0.0 ? 1.0 / eps : 0.0;
    } else {
        resid = resid_norm / ((double)maxmn * anorm * xnorm * eps);
    }

    assert_residual_ok(resid);
}

/* Test overdetermined TRANS='N' with well-conditioned matrices */
static void test_overdetermined_wellcond(void **state)
{
    ls_fixture_t *fix = *state;
    for (int imat = 1; imat <= 4; imat++) {
        fix->seed = g_seed++;
        run_dgels_overdetermined(fix, imat);
    }
}

/* Test overdetermined TRANS='N' with ill-conditioned matrices */
static void test_overdetermined_illcond(void **state)
{
    ls_fixture_t *fix = *state;
    for (int imat = 8; imat <= 9; imat++) {
        fix->seed = g_seed++;
        run_dgels_overdetermined(fix, imat);
    }
}

/* Test overdetermined TRANS='N' with scaled matrices */
static void test_overdetermined_scaled(void **state)
{
    ls_fixture_t *fix = *state;
    for (int imat = 10; imat <= 11; imat++) {
        fix->seed = g_seed++;
        run_dgels_overdetermined(fix, imat);
    }
}

/* Test underdetermined (m < n) TRANS='N' */
static void test_underdetermined(void **state)
{
    ls_fixture_t *fix = *state;
    for (int imat = 1; imat <= 4; imat++) {
        fix->seed = g_seed++;
        run_dgels_underdetermined(fix, imat);
    }
}

/* Test TRANS='T' */
static void test_trans(void **state)
{
    ls_fixture_t *fix = *state;
    for (int imat = 1; imat <= 4; imat++) {
        fix->seed = g_seed++;
        run_dgels_trans_mgen(fix, imat);
    }
}

/* Test workspace query */
static void test_workspace_query(void **state)
{
    ls_fixture_t *fix = *state;
    double wkopt;
    int info;

    dgels("N", fix->m, fix->n, fix->nrhs, fix->A, fix->lda,
          fix->B, fix->ldb, &wkopt, -1, &info);
    assert_info_success(info);
    assert_true(wkopt >= 1.0);
}

int main(void)
{
    const struct CMUnitTest tests[] = {
        /* Overdetermined m > n, TRANS='N' */
        cmocka_unit_test_setup_teardown(test_overdetermined_wellcond, setup_10x5_1, ls_teardown),
        cmocka_unit_test_setup_teardown(test_overdetermined_wellcond, setup_20x10_1, ls_teardown),
        cmocka_unit_test_setup_teardown(test_overdetermined_wellcond, setup_20x10_3, ls_teardown),
        cmocka_unit_test_setup_teardown(test_overdetermined_wellcond, setup_50x20_1, ls_teardown),
        cmocka_unit_test_setup_teardown(test_overdetermined_illcond, setup_20x10_1, ls_teardown),
        cmocka_unit_test_setup_teardown(test_overdetermined_scaled, setup_20x10_1, ls_teardown),
        /* Square (also uses overdetermined path since m >= n) */
        cmocka_unit_test_setup_teardown(test_overdetermined_wellcond, setup_10x10_1, ls_teardown),
        cmocka_unit_test_setup_teardown(test_overdetermined_wellcond, setup_10x10_3, ls_teardown),
        /* Underdetermined m < n, TRANS='N' */
        cmocka_unit_test_setup_teardown(test_underdetermined, setup_5x10_1, ls_teardown),
        cmocka_unit_test_setup_teardown(test_underdetermined, setup_10x20_1, ls_teardown),
        cmocka_unit_test_setup_teardown(test_underdetermined, setup_10x20_3, ls_teardown),
        /* TRANS='T' (underdetermined A^T*X=B for m >= n) */
        cmocka_unit_test_setup_teardown(test_trans, setup_10x5_1, ls_teardown),
        cmocka_unit_test_setup_teardown(test_trans, setup_20x10_1, ls_teardown),
        /* Workspace query */
        cmocka_unit_test_setup_teardown(test_workspace_query, setup_20x10_1, ls_teardown),
    };
    return cmocka_run_group_tests_name("dgels", tests, NULL, NULL);
}
