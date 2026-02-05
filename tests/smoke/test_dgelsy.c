/**
 * @file test_dgelsy.c
 * @brief CMocka test suite for dgelsy (rank-deficient least squares).
 *
 * Tests the rank-deficient least squares driver dgelsy using LAPACK's
 * verification methodology with consistent systems (B = A * X_true).
 *
 * Verification:
 *   residual = ||B - A*X|| / (max(m,n) * ||A|| * ||X|| * eps)
 *   Also verifies that rank determination is correct.
 */

#include "test_harness.h"

/* Test threshold - see LAPACK dtest.in */
#define THRESH 20.0
#include <cblas.h>

/* Routine under test */
extern void dgelsy(const int m, const int n, const int nrhs,
                   double * restrict A, const int lda,
                   double * restrict B, const int ldb,
                   int * restrict jpvt, const double rcond,
                   int *rank,
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
    double *Acopy;  /* working copy for dgelsy */
    double *B;      /* RHS (= A * X_true) */
    double *Bcopy;  /* working copy for residual computation */
    double *X_true; /* true solution */
    double *work;
    double *d;
    double *genwork;
    int *jpvt;
    int lwork;
    uint64_t seed;
} lsy_fixture_t;

static uint64_t g_seed = 7001;

static int lsy_setup(void **state, int m, int n, int nrhs)
{
    lsy_fixture_t *fix = malloc(sizeof(lsy_fixture_t));
    assert_non_null(fix);

    fix->m = m;
    fix->n = n;
    fix->nrhs = nrhs;
    int maxmn = m > n ? m : n;
    int minmn = m < n ? m : n;
    fix->lda = maxmn;
    fix->ldb = maxmn;
    fix->seed = g_seed++;

    fix->lwork = maxmn * 64 + 3 * minmn + maxmn;  /* generous workspace */

    fix->A = calloc(fix->lda * n, sizeof(double));
    fix->Acopy = calloc(fix->lda * n, sizeof(double));
    fix->B = calloc(fix->ldb * nrhs, sizeof(double));
    fix->Bcopy = calloc(fix->ldb * nrhs, sizeof(double));
    fix->X_true = calloc(n * nrhs, sizeof(double));
    fix->work = calloc(fix->lwork, sizeof(double));
    fix->d = calloc(maxmn, sizeof(double));
    fix->genwork = calloc(3 * maxmn, sizeof(double));
    fix->jpvt = calloc(n, sizeof(int));

    assert_non_null(fix->A);
    assert_non_null(fix->Acopy);
    assert_non_null(fix->B);
    assert_non_null(fix->Bcopy);
    assert_non_null(fix->X_true);
    assert_non_null(fix->work);
    assert_non_null(fix->d);
    assert_non_null(fix->genwork);
    assert_non_null(fix->jpvt);

    *state = fix;
    return 0;
}

static int lsy_teardown(void **state)
{
    lsy_fixture_t *fix = *state;
    if (fix) {
        free(fix->A);
        free(fix->Acopy);
        free(fix->B);
        free(fix->Bcopy);
        free(fix->X_true);
        free(fix->work);
        free(fix->d);
        free(fix->genwork);
        free(fix->jpvt);
        free(fix);
    }
    return 0;
}

/* Size-specific setups */
static int setup_10x5_1(void **state) { return lsy_setup(state, 10, 5, 1); }
static int setup_20x10_1(void **state) { return lsy_setup(state, 20, 10, 1); }
static int setup_20x10_3(void **state) { return lsy_setup(state, 20, 10, 3); }
static int setup_5x10_1(void **state) { return lsy_setup(state, 5, 10, 1); }
static int setup_10x20_1(void **state) { return lsy_setup(state, 10, 20, 1); }
static int setup_10x10_1(void **state) { return lsy_setup(state, 10, 10, 1); }
static int setup_10x10_3(void **state) { return lsy_setup(state, 10, 10, 3); }

/**
 * Run dgelsy on a full-rank matrix with consistent RHS and verify solution.
 * B = A * X_true ensures B is in the column space of A.
 */
static void run_dgelsy_fullrank(lsy_fixture_t *fix, int imat)
{
    int info, rank;
    int m = fix->m, n = fix->n, nrhs = fix->nrhs;
    int lda = fix->lda, ldb = fix->ldb;
    int maxmn = m > n ? m : n;
    int minmn = m < n ? m : n;
    double eps = dlamch("E");
    char type, dist;
    int kl, ku, mode;
    double anorm_param, cndnum;

    /* Generate full-rank matrix A */
    dlatb4("DGE", imat, m, n, &type, &kl, &ku, &anorm_param, &mode, &cndnum, &dist);
    dlatms(m, n, &dist, fix->seed, &type, fix->d, mode, cndnum, anorm_param,
           kl, ku, "N", fix->A, lda, fix->genwork, &info);
    assert_int_equal(info, 0);

    /* Generate random X_true (n x nrhs) */
    uint64_t s = fix->seed + 3000;
    for (int j = 0; j < nrhs; j++) {
        for (int i = 0; i < n; i++) {
            s ^= s << 13; s ^= s >> 7; s ^= s << 17;
            fix->X_true[i + j * n] = (double)(int64_t)s / (double)INT64_MAX;
        }
    }

    /* Compute B = A * X_true (m x nrhs) - consistent system */
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                m, nrhs, n, 1.0, fix->A, lda,
                fix->X_true, n, 0.0, fix->B, ldb);

    /* Save copies */
    for (int j = 0; j < n; j++)
        cblas_dcopy(m, &fix->A[j * lda], 1, &fix->Acopy[j * lda], 1);
    for (int j = 0; j < nrhs; j++)
        cblas_dcopy(maxmn, &fix->B[j * ldb], 1, &fix->Bcopy[j * ldb], 1);

    /* Initialize jpvt to 0 (all free columns) */
    for (int i = 0; i < n; i++)
        fix->jpvt[i] = 0;

    /* RCOND for rank determination - should detect full rank */
    double rcond = eps * (double)maxmn;

    /* Call dgelsy */
    dgelsy(m, n, nrhs, fix->A, lda, fix->B, ldb,
           fix->jpvt, rcond, &rank, fix->work, fix->lwork, &info);
    assert_info_success(info);

    /* For full-rank matrices, rank should equal min(m,n) */
    assert_int_equal(rank, minmn);

    /* Compute residual: Bcopy := Bcopy - Acopy * X
     * X is in B[0:n-1, 0:nrhs-1] */
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                m, nrhs, n, -1.0, fix->Acopy, lda,
                fix->B, ldb, 1.0, fix->Bcopy, ldb);

    double resid_norm = dlange("1", m, nrhs, fix->Bcopy, ldb, NULL);
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
 * Run dgelsy on a rank-deficient matrix with consistent RHS and verify.
 * B = A * X_true ensures B is in the column space of A.
 */
static void run_dgelsy_rankdef(lsy_fixture_t *fix, int def_rank)
{
    int info, rank;
    int m = fix->m, n = fix->n, nrhs = fix->nrhs;
    int lda = fix->lda, ldb = fix->ldb;
    int maxmn = m > n ? m : n;
    int minmn = m < n ? m : n;
    double eps = dlamch("E");

    /* Generate rank-deficient matrix using dlatms with mode=0 and explicit
     * singular values. Set only def_rank singular values to be nonzero. */
    for (int i = 0; i < minmn; i++) {
        if (i < def_rank) {
            fix->d[i] = 1.0 / (1.0 + (double)i);  /* decreasing from 1 to 1/(def_rank) */
        } else {
            fix->d[i] = 0.0;  /* zero singular values -> rank deficiency */
        }
    }

    /* Generate A = U * D * V^T with specified singular values.
     * mode=0 means use d[] directly as singular values. */
    int info_gen;
    dlatms(m, n, "U", fix->seed, "N", fix->d, 0, 1.0, 1.0,
           m, n, "N", fix->A, lda, fix->genwork, &info_gen);
    assert_int_equal(info_gen, 0);

    /* Generate random X_true (n x nrhs) */
    uint64_t s = fix->seed + 4000;
    for (int j = 0; j < nrhs; j++) {
        for (int i = 0; i < n; i++) {
            s ^= s << 13; s ^= s >> 7; s ^= s << 17;
            fix->X_true[i + j * n] = (double)(int64_t)s / (double)INT64_MAX;
        }
    }

    /* Compute B = A * X_true (m x nrhs) - consistent system.
     * Since A is rank-deficient, B is in the column space of A. */
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                m, nrhs, n, 1.0, fix->A, lda,
                fix->X_true, n, 0.0, fix->B, ldb);

    /* Save copies */
    for (int j = 0; j < n; j++)
        cblas_dcopy(m, &fix->A[j * lda], 1, &fix->Acopy[j * lda], 1);
    for (int j = 0; j < nrhs; j++)
        cblas_dcopy(maxmn, &fix->B[j * ldb], 1, &fix->Bcopy[j * ldb], 1);

    /* Initialize jpvt to 0 (all free columns) */
    for (int i = 0; i < n; i++)
        fix->jpvt[i] = 0;

    /* RCOND to detect the rank deficiency:
     * smallest nonzero sv is 1/def_rank, so condition = def_rank.
     * Use rcond small enough to detect exact zeros but large enough
     * to be numerically stable. */
    double rcond = 0.5 / (double)(def_rank + 1);

    /* Call dgelsy */
    dgelsy(m, n, nrhs, fix->A, lda, fix->B, ldb,
           fix->jpvt, rcond, &rank, fix->work, fix->lwork, &info);
    assert_info_success(info);

    /* Rank should be approximately def_rank.
     * Allow +-1 tolerance since incremental condition estimation
     * is approximate, but it should not exceed min(m,n). */
    assert_true(rank <= minmn);
    assert_true(rank >= 1);
    assert_true(rank <= def_rank + 1);
    assert_true(rank >= def_rank - 1 || def_rank <= 1);

    /* Compute residual: Bcopy := Bcopy - Acopy * X_computed
     * X is in B[0:n-1, 0:nrhs-1] */
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                m, nrhs, n, -1.0, fix->Acopy, lda,
                fix->B, ldb, 1.0, fix->Bcopy, ldb);

    double resid_norm = dlange("1", m, nrhs, fix->Bcopy, ldb, NULL);
    double anorm = dlange("1", m, n, fix->Acopy, lda, NULL);
    double xnorm = dlange("1", n, nrhs, fix->B, ldb, NULL);

    /* Since B is in the column space of A, the residual should be small
     * even for rank-deficient matrices. Use the standard LAPACK formula. */
    double resid;
    if (anorm == 0.0 || xnorm == 0.0) {
        resid = resid_norm > 0.0 ? 1.0 / eps : 0.0;
    } else {
        resid = resid_norm / ((double)maxmn * anorm * xnorm * eps);
    }

    assert_residual_ok(resid);
}

/* Full-rank tests with well-conditioned matrices */
static void test_fullrank_wellcond(void **state)
{
    lsy_fixture_t *fix = *state;
    for (int imat = 1; imat <= 4; imat++) {
        fix->seed = g_seed++;
        run_dgelsy_fullrank(fix, imat);
    }
}

/* Full-rank tests with ill-conditioned matrices */
static void test_fullrank_illcond(void **state)
{
    lsy_fixture_t *fix = *state;
    for (int imat = 8; imat <= 9; imat++) {
        fix->seed = g_seed++;
        run_dgelsy_fullrank(fix, imat);
    }
}

/* Full-rank tests with scaled matrices */
static void test_fullrank_scaled(void **state)
{
    lsy_fixture_t *fix = *state;
    for (int imat = 10; imat <= 11; imat++) {
        fix->seed = g_seed++;
        run_dgelsy_fullrank(fix, imat);
    }
}

/* Rank-deficient tests: rank = 3/4 * min(m,n) */
static void test_rankdef_half(void **state)
{
    lsy_fixture_t *fix = *state;
    int minmn = fix->m < fix->n ? fix->m : fix->n;
    int def_rank = (3 * minmn) / 4;  /* matches LAPACK RKSEL=2 */
    if (def_rank < 1) def_rank = 1;
    fix->seed = g_seed++;
    run_dgelsy_rankdef(fix, def_rank);
}

/* Rank-deficient tests: rank = 1 */
static void test_rankdef_one(void **state)
{
    lsy_fixture_t *fix = *state;
    fix->seed = g_seed++;
    run_dgelsy_rankdef(fix, 1);
}

/* Workspace query test */
static void test_workspace_query(void **state)
{
    lsy_fixture_t *fix = *state;
    double wkopt;
    int info, rank;

    dgelsy(fix->m, fix->n, fix->nrhs, fix->A, fix->lda,
           fix->B, fix->ldb, fix->jpvt, 1e-10, &rank,
           &wkopt, -1, &info);
    assert_info_success(info);
    assert_true(wkopt >= 1.0);
}

int main(void)
{
    const struct CMUnitTest tests[] = {
        /* Full-rank overdetermined */
        cmocka_unit_test_setup_teardown(test_fullrank_wellcond, setup_10x5_1, lsy_teardown),
        cmocka_unit_test_setup_teardown(test_fullrank_wellcond, setup_20x10_1, lsy_teardown),
        cmocka_unit_test_setup_teardown(test_fullrank_wellcond, setup_20x10_3, lsy_teardown),
        cmocka_unit_test_setup_teardown(test_fullrank_illcond, setup_20x10_1, lsy_teardown),
        cmocka_unit_test_setup_teardown(test_fullrank_scaled, setup_20x10_1, lsy_teardown),
        /* Full-rank underdetermined */
        cmocka_unit_test_setup_teardown(test_fullrank_wellcond, setup_5x10_1, lsy_teardown),
        cmocka_unit_test_setup_teardown(test_fullrank_wellcond, setup_10x20_1, lsy_teardown),
        /* Full-rank square */
        cmocka_unit_test_setup_teardown(test_fullrank_wellcond, setup_10x10_1, lsy_teardown),
        cmocka_unit_test_setup_teardown(test_fullrank_wellcond, setup_10x10_3, lsy_teardown),
        /* Rank-deficient */
        cmocka_unit_test_setup_teardown(test_rankdef_half, setup_10x5_1, lsy_teardown),
        cmocka_unit_test_setup_teardown(test_rankdef_half, setup_20x10_1, lsy_teardown),
        cmocka_unit_test_setup_teardown(test_rankdef_half, setup_10x10_1, lsy_teardown),
        cmocka_unit_test_setup_teardown(test_rankdef_one, setup_10x5_1, lsy_teardown),
        cmocka_unit_test_setup_teardown(test_rankdef_one, setup_20x10_1, lsy_teardown),
        /* Workspace query */
        cmocka_unit_test_setup_teardown(test_workspace_query, setup_20x10_1, lsy_teardown),
    };
    return cmocka_run_group_tests_name("dgelsy", tests, NULL, NULL);
}
