/**
 * @file test_dgtrfs.c
 * @brief CMocka test suite for dgtrfs (iterative refinement for tridiagonal systems).
 *
 * Tests the iterative refinement routine dgtrfs which improves the computed
 * solution and provides error bounds.
 *
 * Verification: dgtt05 tests the error bounds from iterative refinement.
 *   reslts[0]: forward error bound check
 *   reslts[1]: backward error bound check
 */

#include "test_harness.h"

/* Test threshold - see LAPACK dtest.in */
#define THRESH 20.0
#include "testutils/test_rng.h"
#include <stdbool.h>

/* Routines under test */
extern void dgttrf(const int n, double * const restrict DL,
                   double * const restrict D, double * const restrict DU,
                   double * const restrict DU2, int * const restrict ipiv,
                   int *info);
extern void dgttrs(const char *trans, const int n, const int nrhs,
                   const double * const restrict DL,
                   const double * const restrict D,
                   const double * const restrict DU,
                   const double * const restrict DU2,
                   const int * const restrict ipiv,
                   double * const restrict B, const int ldb, int *info);
extern void dgtrfs(const char *trans, const int n, const int nrhs,
                   const double * const restrict DL,
                   const double * const restrict D,
                   const double * const restrict DU,
                   const double * const restrict DLF,
                   const double * const restrict DF,
                   const double * const restrict DUF,
                   const double * const restrict DU2,
                   const int * const restrict ipiv,
                   const double * const restrict B, const int ldb,
                   double * const restrict X, const int ldx,
                   double * const restrict ferr, double * const restrict berr,
                   double * const restrict work, int * const restrict iwork,
                   int *info);

/* Verification routine */
extern void dgtt05(const char *trans, const int n, const int nrhs,
                   const double * const restrict DL,
                   const double * const restrict D,
                   const double * const restrict DU,
                   const double * const restrict B, const int ldb,
                   const double * const restrict X, const int ldx,
                   const double * const restrict XACT, const int ldxact,
                   const double * const restrict ferr,
                   const double * const restrict berr,
                   double * const restrict reslts);

/* Matrix generation */
extern void dlatb4(const char *path, const int imat, const int m, const int n,
                   char *type, int *kl, int *ku, double *anorm, int *mode,
                   double *cndnum, char *dist);

/* Utilities */
extern double dlamch(const char *cmach);
extern void dlagtm(const char *trans, const int n, const int nrhs,
                   const double alpha,
                   const double * const restrict DL,
                   const double * const restrict D,
                   const double * const restrict DU,
                   const double * const restrict X, const int ldx,
                   const double beta,
                   double * const restrict B, const int ldb);

/*
 * Test fixture: holds all allocated memory for a single test case.
 */
typedef struct {
    int n, nrhs;
    int ldb;
    double *DL;      /* Original sub-diagonal */
    double *D;       /* Original diagonal */
    double *DU;      /* Original super-diagonal */
    double *DLF;     /* Factored sub-diagonal */
    double *DF;      /* Factored diagonal */
    double *DUF;     /* Factored super-diagonal */
    double *DU2;     /* Second super-diagonal from factorization */
    int *ipiv;       /* Pivot indices */
    double *XACT;    /* Exact solution */
    double *X;       /* Computed/refined solution */
    double *B;       /* Right-hand side */
    double *ferr;    /* Forward error estimates */
    double *berr;    /* Backward error estimates */
    double *work;    /* Workspace for dgtrfs */
    int *iwork;      /* Integer workspace for dgtrfs */
    uint64_t seed;   /* RNG seed */
} dgtrfs_fixture_t;

/* Global seed for test sequence reproducibility */
static uint64_t g_seed = 2718;

/**
 * Generate a diagonally dominant tridiagonal matrix for testing.
 */
static void generate_gt_matrix(int n, int imat, double *DL, double *D, double *DU,
                                uint64_t *seed)
{
    char type, dist;
    int kl, ku, mode;
    double anorm, cndnum;
    int i;

    if (n <= 0) return;

    dlatb4("DGT", imat, n, n, &type, &kl, &ku, &anorm, &mode, &cndnum, &dist);

    rng_seed(*seed);

    /* Generate diagonally dominant matrix for stability */
    for (i = 0; i < n; i++) {
        D[i] = 4.0 + rng_uniform();
    }
    for (i = 0; i < n - 1; i++) {
        DL[i] = rng_uniform() - 0.5;
        DU[i] = rng_uniform() - 0.5;
    }

    /* Scale if needed */
    if (anorm != 1.0) {
        for (i = 0; i < n; i++) {
            D[i] *= anorm;
        }
        for (i = 0; i < n - 1; i++) {
            DL[i] *= anorm;
            DU[i] *= anorm;
        }
    }

    (*seed)++;
}

/**
 * Setup fixture: allocate memory for given dimensions.
 */
static int dgtrfs_setup(void **state, int n, int nrhs)
{
    dgtrfs_fixture_t *fix = malloc(sizeof(dgtrfs_fixture_t));
    assert_non_null(fix);

    int m = (n > 1) ? n - 1 : 0;
    int ldb = (n > 1) ? n : 1;

    fix->n = n;
    fix->nrhs = nrhs;
    fix->ldb = ldb;
    fix->seed = g_seed++;

    fix->DL = malloc((m > 0 ? m : 1) * sizeof(double));
    fix->D = malloc(n * sizeof(double));
    fix->DU = malloc((m > 0 ? m : 1) * sizeof(double));
    fix->DLF = malloc((m > 0 ? m : 1) * sizeof(double));
    fix->DF = malloc(n * sizeof(double));
    fix->DUF = malloc((m > 0 ? m : 1) * sizeof(double));
    fix->DU2 = malloc((n > 2 ? n - 2 : 1) * sizeof(double));
    fix->ipiv = malloc(n * sizeof(int));
    fix->XACT = malloc(ldb * nrhs * sizeof(double));
    fix->X = malloc(ldb * nrhs * sizeof(double));
    fix->B = malloc(ldb * nrhs * sizeof(double));
    fix->ferr = malloc(nrhs * sizeof(double));
    fix->berr = malloc(nrhs * sizeof(double));
    fix->work = malloc(3 * n * sizeof(double));
    fix->iwork = malloc(n * sizeof(int));

    assert_non_null(fix->DL);
    assert_non_null(fix->D);
    assert_non_null(fix->DU);
    assert_non_null(fix->DLF);
    assert_non_null(fix->DF);
    assert_non_null(fix->DUF);
    assert_non_null(fix->DU2);
    assert_non_null(fix->ipiv);
    assert_non_null(fix->XACT);
    assert_non_null(fix->X);
    assert_non_null(fix->B);
    assert_non_null(fix->ferr);
    assert_non_null(fix->berr);
    assert_non_null(fix->work);
    assert_non_null(fix->iwork);

    *state = fix;
    return 0;
}

/**
 * Teardown fixture: free all allocated memory.
 */
static int dgtrfs_teardown(void **state)
{
    dgtrfs_fixture_t *fix = *state;
    if (fix) {
        free(fix->DL);
        free(fix->D);
        free(fix->DU);
        free(fix->DLF);
        free(fix->DF);
        free(fix->DUF);
        free(fix->DU2);
        free(fix->ipiv);
        free(fix->XACT);
        free(fix->X);
        free(fix->B);
        free(fix->ferr);
        free(fix->berr);
        free(fix->work);
        free(fix->iwork);
        free(fix);
    }
    return 0;
}

/* Size-specific setup wrappers: sizes {1,2,3,5,10,50} x nrhs {1,2,15} */
static int setup_n1_nrhs1(void **state) { return dgtrfs_setup(state, 1, 1); }
static int setup_n2_nrhs1(void **state) { return dgtrfs_setup(state, 2, 1); }
static int setup_n3_nrhs1(void **state) { return dgtrfs_setup(state, 3, 1); }
static int setup_n5_nrhs1(void **state) { return dgtrfs_setup(state, 5, 1); }
static int setup_n10_nrhs1(void **state) { return dgtrfs_setup(state, 10, 1); }
static int setup_n50_nrhs1(void **state) { return dgtrfs_setup(state, 50, 1); }
static int setup_n1_nrhs2(void **state) { return dgtrfs_setup(state, 1, 2); }
static int setup_n2_nrhs2(void **state) { return dgtrfs_setup(state, 2, 2); }
static int setup_n3_nrhs2(void **state) { return dgtrfs_setup(state, 3, 2); }
static int setup_n5_nrhs2(void **state) { return dgtrfs_setup(state, 5, 2); }
static int setup_n10_nrhs2(void **state) { return dgtrfs_setup(state, 10, 2); }
static int setup_n50_nrhs2(void **state) { return dgtrfs_setup(state, 50, 2); }
static int setup_n1_nrhs15(void **state) { return dgtrfs_setup(state, 1, 15); }
static int setup_n2_nrhs15(void **state) { return dgtrfs_setup(state, 2, 15); }
static int setup_n3_nrhs15(void **state) { return dgtrfs_setup(state, 3, 15); }
static int setup_n5_nrhs15(void **state) { return dgtrfs_setup(state, 5, 15); }
static int setup_n10_nrhs15(void **state) { return dgtrfs_setup(state, 10, 15); }
static int setup_n50_nrhs15(void **state) { return dgtrfs_setup(state, 50, 15); }

/**
 * Core test logic: generate matrix, factorize, solve, refine, verify.
 * Writes reslts[0] (forward error) and reslts[1] (backward error).
 * Returns true on success, false if factorization was singular.
 */
static bool run_dgtrfs_test(dgtrfs_fixture_t *fix, int imat, const char* trans,
                            double *reslts)
{
    int info;
    int n = fix->n;
    int nrhs = fix->nrhs;
    int m = (n > 1) ? n - 1 : 0;
    int ldb = fix->ldb;
    int ldx = ldb;
    int i, j;

    /* Generate test matrix */
    generate_gt_matrix(n, imat, fix->DL, fix->D, fix->DU, &fix->seed);

    /* Copy to factored arrays */
    memcpy(fix->DLF, fix->DL, (m > 0 ? m : 1) * sizeof(double));
    memcpy(fix->DF, fix->D, n * sizeof(double));
    memcpy(fix->DUF, fix->DU, (m > 0 ? m : 1) * sizeof(double));

    /* Factor */
    dgttrf(n, fix->DLF, fix->DF, fix->DUF, fix->DU2, fix->ipiv, &info);
    if (info != 0) {
        return false;
    }

    /* Generate random exact solution XACT */
    rng_seed(fix->seed);
    for (j = 0; j < nrhs; j++) {
        for (i = 0; i < n; i++) {
            fix->XACT[i + j * ldb] = rng_uniform_symmetric();
        }
    }
    fix->seed++;

    /* Compute B = op(A) * XACT */
    for (j = 0; j < nrhs; j++) {
        for (i = 0; i < n; i++) {
            fix->B[i + j * ldb] = 0.0;
        }
    }
    dlagtm(trans, n, nrhs, 1.0, fix->DL, fix->D, fix->DU, fix->XACT, ldb,
           0.0, fix->B, ldb);

    /* Solve op(A)*X = B to get initial solution */
    memcpy(fix->X, fix->B, ldx * nrhs * sizeof(double));
    dgttrs(trans, n, nrhs, fix->DLF, fix->DF, fix->DUF, fix->DU2, fix->ipiv,
            fix->X, ldx, &info);
    assert_info_success(info);

    /* Apply iterative refinement */
    dgtrfs(trans, n, nrhs, fix->DL, fix->D, fix->DU, fix->DLF, fix->DF, fix->DUF,
            fix->DU2, fix->ipiv, fix->B, ldb, fix->X, ldx, fix->ferr, fix->berr,
            fix->work, fix->iwork, &info);
    assert_info_success(info);

    /* Verify error bounds using dgtt05 */
    dgtt05(trans, n, nrhs, fix->DL, fix->D, fix->DU, fix->B, ldb, fix->X, ldx,
           fix->XACT, ldb, fix->ferr, fix->berr, reslts);

    return true;
}

/*
 * Test well-conditioned types (1-6) with no transpose.
 */
static void test_dgtrfs_notrans(void **state)
{
    dgtrfs_fixture_t *fix = *state;
    double reslts[2];

    for (int imat = 1; imat <= 6; imat++) {
        fix->seed = g_seed++;
        bool ok = run_dgtrfs_test(fix, imat, "N", reslts);
        if (!ok) continue; /* singular - skip */
        assert_residual_ok(reslts[0]); /* forward error */
        assert_residual_ok(reslts[1]); /* backward error */
    }
}

/*
 * Test well-conditioned types (1-6) with transpose.
 */
static void test_dgtrfs_trans(void **state)
{
    dgtrfs_fixture_t *fix = *state;
    double reslts[2];

    for (int imat = 1; imat <= 6; imat++) {
        fix->seed = g_seed++;
        bool ok = run_dgtrfs_test(fix, imat, "T", reslts);
        if (!ok) continue; /* singular - skip */
        assert_residual_ok(reslts[0]); /* forward error */
        assert_residual_ok(reslts[1]); /* backward error */
    }
}

/*
 * Macro to generate test entries for a given setup.
 */
#define DGTRFS_TESTS(setup_fn) \
    cmocka_unit_test_setup_teardown(test_dgtrfs_notrans, setup_fn, dgtrfs_teardown), \
    cmocka_unit_test_setup_teardown(test_dgtrfs_trans, setup_fn, dgtrfs_teardown)

int main(void)
{
    const struct CMUnitTest tests[] = {
        /* nrhs = 1 */
        DGTRFS_TESTS(setup_n1_nrhs1),
        DGTRFS_TESTS(setup_n2_nrhs1),
        DGTRFS_TESTS(setup_n3_nrhs1),
        DGTRFS_TESTS(setup_n5_nrhs1),
        DGTRFS_TESTS(setup_n10_nrhs1),
        DGTRFS_TESTS(setup_n50_nrhs1),

        /* nrhs = 2 */
        DGTRFS_TESTS(setup_n1_nrhs2),
        DGTRFS_TESTS(setup_n2_nrhs2),
        DGTRFS_TESTS(setup_n3_nrhs2),
        DGTRFS_TESTS(setup_n5_nrhs2),
        DGTRFS_TESTS(setup_n10_nrhs2),
        DGTRFS_TESTS(setup_n50_nrhs2),

        /* nrhs = 15 */
        DGTRFS_TESTS(setup_n1_nrhs15),
        DGTRFS_TESTS(setup_n2_nrhs15),
        DGTRFS_TESTS(setup_n3_nrhs15),
        DGTRFS_TESTS(setup_n5_nrhs15),
        DGTRFS_TESTS(setup_n10_nrhs15),
        DGTRFS_TESTS(setup_n50_nrhs15),
    };

    return cmocka_run_group_tests_name("dgtrfs", tests, NULL, NULL);
}
