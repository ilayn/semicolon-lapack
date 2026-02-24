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
#include "verify.h"
#include "test_rng.h"

/* Test threshold - see LAPACK dtest.in */
#define THRESH 20.0
#include "semicolon_cblas.h"

/* Routine under test */
/* Utility routines */
typedef struct {
    INT m, n, nrhs;
    INT lda, ldb;
    f64 *A;      /* original matrix */
    f64 *Acopy;  /* working copy for dgelsy */
    f64 *B;      /* RHS (= A * X_true) */
    f64 *Bcopy;  /* working copy for residual computation */
    f64 *X_true; /* true solution */
    f64 *work;
    f64 *d;
    f64 *genwork;
    INT* jpvt;
    INT lwork;
    uint64_t seed;
} lsy_fixture_t;

static uint64_t g_seed = 7001;

static int lsy_setup(void **state, INT m, INT n, INT nrhs)
{
    lsy_fixture_t *fix = malloc(sizeof(lsy_fixture_t));
    assert_non_null(fix);

    fix->m = m;
    fix->n = n;
    fix->nrhs = nrhs;
    INT maxmn = m > n ? m : n;
    INT minmn = m < n ? m : n;
    fix->lda = maxmn;
    fix->ldb = maxmn;
    fix->seed = g_seed++;

    fix->lwork = maxmn * 64 + 3 * minmn + maxmn;  /* generous workspace */

    fix->A = calloc(fix->lda * n, sizeof(f64));
    fix->Acopy = calloc(fix->lda * n, sizeof(f64));
    fix->B = calloc(fix->ldb * nrhs, sizeof(f64));
    fix->Bcopy = calloc(fix->ldb * nrhs, sizeof(f64));
    fix->X_true = calloc(n * nrhs, sizeof(f64));
    fix->work = calloc(fix->lwork, sizeof(f64));
    fix->d = calloc(maxmn, sizeof(f64));
    fix->genwork = calloc(3 * maxmn, sizeof(f64));
    fix->jpvt = calloc(n, sizeof(INT));

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
static void run_dgelsy_fullrank(lsy_fixture_t *fix, INT imat)
{
    INT info, rank;
    INT m = fix->m, n = fix->n, nrhs = fix->nrhs;
    INT lda = fix->lda, ldb = fix->ldb;
    INT maxmn = m > n ? m : n;
    INT minmn = m < n ? m : n;
    f64 eps = dlamch("E");
    char type, dist;
    INT kl, ku, mode;
    f64 anorm_param, cndnum;

    /* Generate full-rank matrix A */
    dlatb4("DGE", imat, m, n, &type, &kl, &ku, &anorm_param, &mode, &cndnum, &dist);
    uint64_t rng_state[4];
    rng_seed(rng_state, fix->seed);
    dlatms(m, n, &dist, &type, fix->d, mode, cndnum, anorm_param,
           kl, ku, "N", fix->A, lda, fix->genwork, &info, rng_state);
    assert_int_equal(info, 0);

    /* Generate random X_true (n x nrhs) */
    uint64_t s = fix->seed + 3000;
    for (INT j = 0; j < nrhs; j++) {
        for (INT i = 0; i < n; i++) {
            s ^= s << 13; s ^= s >> 7; s ^= s << 17;
            fix->X_true[i + j * n] = (f64)(int64_t)s / (f64)INT64_MAX;
        }
    }

    /* Compute B = A * X_true (m x nrhs) - consistent system */
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                m, nrhs, n, 1.0, fix->A, lda,
                fix->X_true, n, 0.0, fix->B, ldb);

    /* Save copies */
    for (INT j = 0; j < n; j++)
        cblas_dcopy(m, &fix->A[j * lda], 1, &fix->Acopy[j * lda], 1);
    for (INT j = 0; j < nrhs; j++)
        cblas_dcopy(maxmn, &fix->B[j * ldb], 1, &fix->Bcopy[j * ldb], 1);

    /* Initialize jpvt to 0 (all free columns) */
    for (INT i = 0; i < n; i++)
        fix->jpvt[i] = 0;

    /* RCOND for rank determination - should detect full rank */
    f64 rcond = eps * (f64)maxmn;

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

    f64 resid_norm = dlange("1", m, nrhs, fix->Bcopy, ldb, NULL);
    f64 anorm = dlange("1", m, n, fix->Acopy, lda, NULL);
    f64 xnorm = dlange("1", n, nrhs, fix->B, ldb, NULL);

    f64 resid;
    if (anorm == 0.0 || xnorm == 0.0) {
        resid = resid_norm > 0.0 ? 1.0 / eps : 0.0;
    } else {
        resid = resid_norm / ((f64)maxmn * anorm * xnorm * eps);
    }

    assert_residual_ok(resid);
}

/**
 * Run dgelsy on a rank-deficient matrix with consistent RHS and verify.
 * B = A * X_true ensures B is in the column space of A.
 */
static void run_dgelsy_rankdef(lsy_fixture_t *fix, INT def_rank)
{
    INT info, rank;
    INT m = fix->m, n = fix->n, nrhs = fix->nrhs;
    INT lda = fix->lda, ldb = fix->ldb;
    INT maxmn = m > n ? m : n;
    INT minmn = m < n ? m : n;
    f64 eps = dlamch("E");

    /* Generate rank-deficient matrix using dlatms with mode=0 and explicit
     * singular values. Set only def_rank singular values to be nonzero. */
    for (INT i = 0; i < minmn; i++) {
        if (i < def_rank) {
            fix->d[i] = 1.0 / (1.0 + (f64)i);  /* decreasing from 1 to 1/(def_rank) */
        } else {
            fix->d[i] = 0.0;  /* zero singular values -> rank deficiency */
        }
    }

    /* Generate A = U * D * V^T with specified singular values.
     * mode=0 means use d[] directly as singular values. */
    INT info_gen;
    uint64_t rng_state[4];
    rng_seed(rng_state, fix->seed);
    dlatms(m, n, "U", "N", fix->d, 0, 1.0, 1.0,
           m, n, "N", fix->A, lda, fix->genwork, &info_gen, rng_state);
    assert_int_equal(info_gen, 0);

    /* Generate random X_true (n x nrhs) */
    uint64_t s = fix->seed + 4000;
    for (INT j = 0; j < nrhs; j++) {
        for (INT i = 0; i < n; i++) {
            s ^= s << 13; s ^= s >> 7; s ^= s << 17;
            fix->X_true[i + j * n] = (f64)(int64_t)s / (f64)INT64_MAX;
        }
    }

    /* Compute B = A * X_true (m x nrhs) - consistent system.
     * Since A is rank-deficient, B is in the column space of A. */
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                m, nrhs, n, 1.0, fix->A, lda,
                fix->X_true, n, 0.0, fix->B, ldb);

    /* Save copies */
    for (INT j = 0; j < n; j++)
        cblas_dcopy(m, &fix->A[j * lda], 1, &fix->Acopy[j * lda], 1);
    for (INT j = 0; j < nrhs; j++)
        cblas_dcopy(maxmn, &fix->B[j * ldb], 1, &fix->Bcopy[j * ldb], 1);

    /* Initialize jpvt to 0 (all free columns) */
    for (INT i = 0; i < n; i++)
        fix->jpvt[i] = 0;

    /* RCOND to detect the rank deficiency:
     * smallest nonzero sv is 1/def_rank, so condition = def_rank.
     * Use rcond small enough to detect exact zeros but large enough
     * to be numerically stable. */
    f64 rcond = 0.5 / (f64)(def_rank + 1);

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

    f64 resid_norm = dlange("1", m, nrhs, fix->Bcopy, ldb, NULL);
    f64 anorm = dlange("1", m, n, fix->Acopy, lda, NULL);
    f64 xnorm = dlange("1", n, nrhs, fix->B, ldb, NULL);

    /* Since B is in the column space of A, the residual should be small
     * even for rank-deficient matrices. Use the standard LAPACK formula. */
    f64 resid;
    if (anorm == 0.0 || xnorm == 0.0) {
        resid = resid_norm > 0.0 ? 1.0 / eps : 0.0;
    } else {
        resid = resid_norm / ((f64)maxmn * anorm * xnorm * eps);
    }

    assert_residual_ok(resid);
}

/* Full-rank tests with well-conditioned matrices */
static void test_fullrank_wellcond(void **state)
{
    lsy_fixture_t *fix = *state;
    for (INT imat = 1; imat <= 4; imat++) {
        fix->seed = g_seed++;
        run_dgelsy_fullrank(fix, imat);
    }
}

/* Full-rank tests with ill-conditioned matrices */
static void test_fullrank_illcond(void **state)
{
    lsy_fixture_t *fix = *state;
    for (INT imat = 8; imat <= 9; imat++) {
        fix->seed = g_seed++;
        run_dgelsy_fullrank(fix, imat);
    }
}

/* Full-rank tests with scaled matrices */
static void test_fullrank_scaled(void **state)
{
    lsy_fixture_t *fix = *state;
    for (INT imat = 10; imat <= 11; imat++) {
        fix->seed = g_seed++;
        run_dgelsy_fullrank(fix, imat);
    }
}

/* Rank-deficient tests: rank = 3/4 * min(m,n) */
static void test_rankdef_half(void **state)
{
    lsy_fixture_t *fix = *state;
    INT minmn = fix->m < fix->n ? fix->m : fix->n;
    INT def_rank = (3 * minmn) / 4;  /* matches LAPACK RKSEL=2 */
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
    f64 wkopt;
    INT info, rank;

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
