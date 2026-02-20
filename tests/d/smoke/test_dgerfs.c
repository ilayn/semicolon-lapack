/**
 * @file test_dgerfs.c
 * @brief CMocka test suite for dgerfs (iterative refinement for general linear systems).
 *
 * Tests the iterative refinement routine which improves the computed solution
 * to a system of linear equations and provides error bounds.
 *
 * Verification:
 * - dget02: ||B - A*X|| / (||A|| * ||X|| * eps) for solution quality
 * - dget07: Tests error bounds (FERR) and backward error (BERR)
 */

#include "test_harness.h"
#include "verify.h"
#include "test_rng.h"

/* Test threshold - see LAPACK dtest.in */
#define THRESH 20.0
#include <cblas.h>

/* Routines under test */
extern void dgetrf(const int m, const int n, f64 * const restrict A,
                   const int lda, int * const restrict ipiv, int *info);
extern void dgetrs(const char *trans, const int n, const int nrhs,
                   const f64 * const restrict A, const int lda,
                   const int * const restrict ipiv, f64 * const restrict B,
                   const int ldb, int *info);
extern void dgerfs(const char *trans, const int n, const int nrhs,
                   const f64 * const restrict A, const int lda,
                   const f64 * const restrict AF, const int ldaf,
                   const int * const restrict ipiv,
                   const f64 * const restrict B, const int ldb,
                   f64 * const restrict X, const int ldx,
                   f64 * const restrict ferr, f64 * const restrict berr,
                   f64 * const restrict work, int * const restrict iwork,
                   int *info);

/* Utilities */
extern f64 dlamch(const char *cmach);

/*
 * Test fixture: holds all allocated memory for a single test case.
 * CMocka passes this between setup -> test -> teardown.
 */
typedef struct {
    int n, nrhs;
    int lda, ldb;
    f64 *A;       /* Original matrix */
    f64 *AF;      /* Factored matrix */
    f64 *B;       /* Right-hand side */
    f64 *B_orig;  /* Original B for verification */
    f64 *X;       /* Computed solution */
    f64 *XACT;    /* Known exact solution */
    f64 *d;       /* Singular values for dlatms */
    f64 *work;    /* Workspace for dgerfs / dlatms */
    f64 *rwork;   /* Workspace for dget02 */
    f64 *ferr;    /* Forward error bounds */
    f64 *berr;    /* Backward error bounds */
    f64 *reslts;  /* Results from dget07 */
    int *ipiv;       /* Pivot indices */
    int *iwork;      /* Integer workspace for dgerfs */
    uint64_t seed;   /* RNG seed */
} dgerfs_fixture_t;

/* Global seed for test sequence reproducibility */
static uint64_t g_seed = 1729;

/**
 * Setup fixture: allocate memory for given dimensions.
 * Called before each test function.
 */
static int dgerfs_setup(void **state, int n, int nrhs)
{
    dgerfs_fixture_t *fix = malloc(sizeof(dgerfs_fixture_t));
    assert_non_null(fix);

    fix->n = n;
    fix->nrhs = nrhs;
    fix->lda = n;
    fix->ldb = n;
    fix->seed = g_seed++;

    fix->A = malloc(fix->lda * n * sizeof(f64));
    fix->AF = malloc(fix->lda * n * sizeof(f64));
    fix->B = malloc(fix->ldb * nrhs * sizeof(f64));
    fix->B_orig = malloc(fix->ldb * nrhs * sizeof(f64));
    fix->X = malloc(fix->ldb * nrhs * sizeof(f64));
    fix->XACT = malloc(fix->ldb * nrhs * sizeof(f64));
    fix->d = malloc(n * sizeof(f64));
    fix->work = malloc(3 * n * sizeof(f64));
    fix->rwork = malloc(n * sizeof(f64));
    fix->ferr = malloc(nrhs * sizeof(f64));
    fix->berr = malloc(nrhs * sizeof(f64));
    fix->reslts = malloc(2 * sizeof(f64));
    fix->ipiv = malloc(n * sizeof(int));
    fix->iwork = malloc(n * sizeof(int));

    assert_non_null(fix->A);
    assert_non_null(fix->AF);
    assert_non_null(fix->B);
    assert_non_null(fix->B_orig);
    assert_non_null(fix->X);
    assert_non_null(fix->XACT);
    assert_non_null(fix->d);
    assert_non_null(fix->work);
    assert_non_null(fix->rwork);
    assert_non_null(fix->ferr);
    assert_non_null(fix->berr);
    assert_non_null(fix->reslts);
    assert_non_null(fix->ipiv);
    assert_non_null(fix->iwork);

    *state = fix;
    return 0;
}

/**
 * Teardown fixture: free all allocated memory.
 * Called after each test function.
 */
static int dgerfs_teardown(void **state)
{
    dgerfs_fixture_t *fix = *state;
    if (fix) {
        free(fix->A);
        free(fix->AF);
        free(fix->B);
        free(fix->B_orig);
        free(fix->X);
        free(fix->XACT);
        free(fix->d);
        free(fix->work);
        free(fix->rwork);
        free(fix->ferr);
        free(fix->berr);
        free(fix->reslts);
        free(fix->ipiv);
        free(fix->iwork);
        free(fix);
    }
    return 0;
}

/* Size-specific setup functions (nrhs=1) */
static int setup_2(void **state) { return dgerfs_setup(state, 2, 1); }
static int setup_3(void **state) { return dgerfs_setup(state, 3, 1); }
static int setup_5(void **state) { return dgerfs_setup(state, 5, 1); }
static int setup_10(void **state) { return dgerfs_setup(state, 10, 1); }
static int setup_20(void **state) { return dgerfs_setup(state, 20, 1); }

/**
 * Core test logic: generate matrix, factorize, solve, refine, verify.
 *
 * Populates fix->reslts with dget07 output and returns the dget02 residual.
 */
static f64 run_dgerfs_test(dgerfs_fixture_t *fix, int imat, const char* trans)
{
    char type, dist;
    int kl, ku, mode;
    f64 anorm, cndnum;
    int info;

    /* Get matrix parameters */
    dlatb4("DGE", imat, fix->n, fix->n, &type, &kl, &ku, &anorm, &mode, &cndnum, &dist);

    /* Generate test matrix A */
    uint64_t rng_state[4];
    rng_seed(rng_state, fix->seed);
    dlatms(fix->n, fix->n, &dist, &type, fix->d, mode, cndnum, anorm,
           kl, ku, "N", fix->A, fix->lda, fix->work, &info, rng_state);
    assert_int_equal(info, 0);

    /* Generate known exact solution XACT */
    for (int j = 0; j < fix->nrhs; j++) {
        for (int i = 0; i < fix->n; i++) {
            fix->XACT[i + j * fix->ldb] = 1.0 + (f64)i / fix->n + (f64)j / fix->nrhs;
        }
    }

    /* Compute B = op(A) * XACT */
    CBLAS_TRANSPOSE cblas_trans = (trans[0] == 'N') ? CblasNoTrans : CblasTrans;
    cblas_dgemm(CblasColMajor, cblas_trans, CblasNoTrans,
                fix->n, fix->nrhs, fix->n, 1.0, fix->A, fix->lda,
                fix->XACT, fix->ldb, 0.0, fix->B, fix->ldb);
    memcpy(fix->B_orig, fix->B, fix->ldb * fix->nrhs * sizeof(f64));

    /* Factor A = P*L*U */
    memcpy(fix->AF, fix->A, fix->lda * fix->n * sizeof(f64));
    dgetrf(fix->n, fix->n, fix->AF, fix->lda, fix->ipiv, &info);
    assert_info_success(info);

    /* Solve op(A) * X = B using factored AF */
    memcpy(fix->X, fix->B, fix->ldb * fix->nrhs * sizeof(f64));
    dgetrs(trans, fix->n, fix->nrhs, fix->AF, fix->lda, fix->ipiv,
           fix->X, fix->ldb, &info);
    assert_info_success(info);

    /* Apply iterative refinement */
    dgerfs(trans, fix->n, fix->nrhs, fix->A, fix->lda, fix->AF, fix->lda,
           fix->ipiv, fix->B, fix->ldb, fix->X, fix->ldb,
           fix->ferr, fix->berr, fix->work, fix->iwork, &info);
    assert_info_success(info);

    /* Compute solution residual using dget02 */
    f64 *B_copy = malloc(fix->ldb * fix->nrhs * sizeof(f64));
    assert_non_null(B_copy);
    memcpy(B_copy, fix->B_orig, fix->ldb * fix->nrhs * sizeof(f64));

    f64 resid;
    dget02(trans, fix->n, fix->n, fix->nrhs, fix->A, fix->lda,
           fix->X, fix->ldb, B_copy, fix->ldb, fix->rwork, &resid);
    free(B_copy);

    /* Compute error bounds using dget07 */
    int chkferr = (imat <= 4);
    dget07(trans, fix->n, fix->nrhs, fix->A, fix->lda, fix->B_orig, fix->ldb,
           fix->X, fix->ldb, fix->XACT, fix->ldb,
           fix->ferr, chkferr, fix->berr, fix->reslts);

    return resid;
}

/*
 * Test well-conditioned matrices (type 4) with no transpose.
 */
static void test_dgerfs_wellcond_notrans(void **state)
{
    dgerfs_fixture_t *fix = *state;
    fix->seed = g_seed++;

    f64 resid = run_dgerfs_test(fix, 4, "N");
    assert_residual_ok(resid);

    /* FERR check for well-conditioned */
    assert_residual_ok(fix->reslts[0]);
    /* BERR check */
    assert_residual_ok(fix->reslts[1]);
}

/*
 * Test well-conditioned matrices (type 4) with transpose.
 */
static void test_dgerfs_wellcond_trans(void **state)
{
    dgerfs_fixture_t *fix = *state;
    fix->seed = g_seed++;

    f64 resid = run_dgerfs_test(fix, 4, "T");
    assert_residual_ok(resid);

    /* FERR check for well-conditioned */
    assert_residual_ok(fix->reslts[0]);
    /* BERR check */
    assert_residual_ok(fix->reslts[1]);
}

/*
 * Test ill-conditioned matrices (type 8).
 * Only run for n >= 5.
 * Only check BERR (not FERR) since chkferr = false for imat > 4.
 */
static void test_dgerfs_illcond(void **state)
{
    dgerfs_fixture_t *fix = *state;

    if (fix->n < 5) {
        skip();
    }

    /* No transpose */
    fix->seed = g_seed++;
    f64 resid = run_dgerfs_test(fix, 8, "N");
    assert_residual_ok(resid);
    assert_residual_ok(fix->reslts[1]);

    /* Transpose */
    fix->seed = g_seed++;
    resid = run_dgerfs_test(fix, 8, "T");
    assert_residual_ok(resid);
    assert_residual_ok(fix->reslts[1]);
}

/*
 * Macro to generate test entries for a given size.
 */
#define DGERFS_TESTS(setup_fn) \
    cmocka_unit_test_setup_teardown(test_dgerfs_wellcond_notrans, setup_fn, dgerfs_teardown), \
    cmocka_unit_test_setup_teardown(test_dgerfs_wellcond_trans, setup_fn, dgerfs_teardown), \
    cmocka_unit_test_setup_teardown(test_dgerfs_illcond, setup_fn, dgerfs_teardown)

/* ===== Sanity Check Tests (standalone, no fixture) ===== */

/**
 * Sanity check: simple 3x3 system with known solution x = [1, 1, 1]'.
 */
static void test_simple(void **state)
{
    (void)state;

    int n = 3;
    int nrhs = 1;

    /* System: A * x = b where solution is x = [1, 1, 1]' */
    f64 A[9] = {2, 4, 8, 1, 3, 7, 1, 3, 9};  /* Column-major */
    f64 AF[9];
    memcpy(AF, A, 9 * sizeof(f64));

    f64 B[3] = {4, 10, 24};
    f64 B_orig[3];
    memcpy(B_orig, B, 3 * sizeof(f64));

    f64 X[3];
    f64 ferr[1], berr[1];
    f64 work[9];
    int iwork[3];
    int ipiv[3];
    int info;

    /* Factor */
    dgetrf(n, n, AF, n, ipiv, &info);
    assert_info_success(info);

    /* Initial solve */
    memcpy(X, B, 3 * sizeof(f64));
    dgetrs("N", n, nrhs, AF, n, ipiv, X, n, &info);
    assert_info_success(info);

    /* Refine */
    dgerfs("N", n, nrhs, A, n, AF, n, ipiv, B_orig, n, X, n,
           ferr, berr, work, iwork, &info);
    assert_info_success(info);

    f64 tol = 1e-10;
    assert_true(fabs(X[0] - 1.0) < tol);
    assert_true(fabs(X[1] - 1.0) < tol);
    assert_true(fabs(X[2] - 1.0) < tol);
}

/**
 * Sanity check: 3x3 system with transpose solve A'*x = b.
 */
static void test_transpose(void **state)
{
    (void)state;

    int n = 3;
    int nrhs = 1;

    /* System: A' * x = b */
    f64 A[9] = {2, 4, 8, 1, 3, 7, 1, 3, 9};  /* Column-major */
    f64 AF[9];
    memcpy(AF, A, 9 * sizeof(f64));

    /* Known solution x = [1, 1, 1]' */
    f64 xact[3] = {1.0, 1.0, 1.0};

    /* Compute B = A' * xact */
    f64 B[3];
    cblas_dgemv(CblasColMajor, CblasTrans, n, n, 1.0, A, n, xact, 1, 0.0, B, 1);
    f64 B_orig[3];
    memcpy(B_orig, B, 3 * sizeof(f64));

    f64 X[3];
    f64 ferr[1], berr[1];
    f64 work[9];
    int iwork[3];
    int ipiv[3];
    int info;

    /* Factor */
    dgetrf(n, n, AF, n, ipiv, &info);
    assert_info_success(info);

    /* Initial solve with transpose */
    memcpy(X, B, 3 * sizeof(f64));
    dgetrs("T", n, nrhs, AF, n, ipiv, X, n, &info);
    assert_info_success(info);

    /* Refine with transpose */
    dgerfs("T", n, nrhs, A, n, AF, n, ipiv, B_orig, n, X, n,
           ferr, berr, work, iwork, &info);
    assert_info_success(info);

    f64 tol = 1e-10;
    assert_true(fabs(X[0] - 1.0) < tol);
    assert_true(fabs(X[1] - 1.0) < tol);
    assert_true(fabs(X[2] - 1.0) < tol);
}

/**
 * Sanity check: 4x3 system with multiple right-hand sides.
 */
static void test_multiple_rhs(void **state)
{
    (void)state;

    int n = 4;
    int nrhs = 3;

    f64 *A = calloc(n * n, sizeof(f64));
    f64 *AF = malloc(n * n * sizeof(f64));
    f64 *B = malloc(n * nrhs * sizeof(f64));
    f64 *B_orig = malloc(n * nrhs * sizeof(f64));
    f64 *X = malloc(n * nrhs * sizeof(f64));
    f64 *ferr = malloc(nrhs * sizeof(f64));
    f64 *berr = malloc(nrhs * sizeof(f64));
    f64 *work = malloc(3 * n * sizeof(f64));
    int *iwork = malloc(n * sizeof(int));
    int *ipiv = malloc(n * sizeof(int));
    int info;

    assert_non_null(A);
    assert_non_null(AF);
    assert_non_null(B);
    assert_non_null(B_orig);
    assert_non_null(X);

    /* Well-conditioned diagonal dominant matrix */
    for (int i = 0; i < n; i++) {
        A[i + i * n] = (f64)(n + 1);
        for (int j = 0; j < n; j++) {
            if (i != j) {
                A[i + j * n] = 1.0;
            }
        }
    }
    memcpy(AF, A, n * n * sizeof(f64));

    /* Multiple RHS with known solutions */
    for (int j = 0; j < nrhs; j++) {
        for (int i = 0; i < n; i++) {
            B[i + j * n] = 0.0;
            for (int k = 0; k < n; k++) {
                f64 xk = (f64)(k + 1) + (f64)j;
                B[i + j * n] += A[i + k * n] * xk;
            }
        }
    }
    memcpy(B_orig, B, n * nrhs * sizeof(f64));

    /* Factor */
    dgetrf(n, n, AF, n, ipiv, &info);
    assert_info_success(info);

    /* Solve */
    memcpy(X, B, n * nrhs * sizeof(f64));
    dgetrs("N", n, nrhs, AF, n, ipiv, X, n, &info);
    assert_info_success(info);

    /* Refine */
    dgerfs("N", n, nrhs, A, n, AF, n, ipiv, B_orig, n, X, n,
           ferr, berr, work, iwork, &info);
    assert_info_success(info);

    f64 tol = 1e-10;
    for (int j = 0; j < nrhs; j++) {
        for (int i = 0; i < n; i++) {
            f64 expected = (f64)(i + 1) + (f64)j;
            assert_true(fabs(X[i + j * n] - expected) < tol);
        }
    }

    free(A);
    free(AF);
    free(B);
    free(B_orig);
    free(X);
    free(ferr);
    free(berr);
    free(work);
    free(iwork);
    free(ipiv);
}

int main(void)
{
    const struct CMUnitTest tests[] = {
        /* Sanity checks (standalone, no fixture) */
        cmocka_unit_test(test_simple),
        cmocka_unit_test(test_transpose),
        cmocka_unit_test(test_multiple_rhs),

        /* Comprehensive tests with fixture (nrhs=1) */
        DGERFS_TESTS(setup_2),
        DGERFS_TESTS(setup_3),
        DGERFS_TESTS(setup_5),
        DGERFS_TESTS(setup_10),
        DGERFS_TESTS(setup_20),
    };

    return cmocka_run_group_tests_name("dgerfs", tests, NULL, NULL);
}
