/**
 * @file test_dgtsvx.c
 * @brief CMocka test suite for dgtsvx (expert tridiagonal system solver).
 *
 * Tests the expert driver dgtsvx which computes the solution to a
 * tridiagonal system with condition estimation and error bounds.
 *
 * Verification:
 * - dgtt02: ||B - op(A)*X|| / (||op(A)|| * ||X|| * eps) for solution
 * - dget06: ratio of estimated to actual condition number
 * - dgtt05: forward and backward error bound verification
 */

#include "test_harness.h"

/* Test threshold - see LAPACK dtest.in */
#define THRESH 20.0
#include "test_rng.h"
#include "verify.h"

/* Routine under test */
extern void dgtsvx(const char *fact, const char *trans, const int n, const int nrhs,
                   const f64 * const restrict DL,
                   const f64 * const restrict D,
                   const f64 * const restrict DU,
                   f64 * const restrict DLF,
                   f64 * const restrict DF,
                   f64 * const restrict DUF,
                   f64 * const restrict DU2,
                   int * const restrict ipiv,
                   const f64 * const restrict B, const int ldb,
                   f64 * const restrict X, const int ldx,
                   f64 *rcond,
                   f64 * const restrict ferr, f64 * const restrict berr,
                   f64 * const restrict work, int * const restrict iwork,
                   int *info);

/* For factored form testing */
extern void dgttrf(const int n, f64 * const restrict DL,
                   f64 * const restrict D, f64 * const restrict DU,
                   f64 * const restrict DU2, int * const restrict ipiv,
                   int *info);

/* Utilities */
extern f64 dlamch(const char *cmach);
extern f64 dlangt(const char *norm, const int n,
                     const f64 * const restrict DL,
                     const f64 * const restrict D,
                     const f64 * const restrict DU);
extern void dlagtm(const char *trans, const int n, const int nrhs,
                   const f64 alpha,
                   const f64 * const restrict DL,
                   const f64 * const restrict D,
                   const f64 * const restrict DU,
                   const f64 * const restrict X, const int ldx,
                   const f64 beta,
                   f64 * const restrict B, const int ldb);
extern void dgttrs(const char *trans, const int n, const int nrhs,
                   const f64 * const restrict DL,
                   const f64 * const restrict D,
                   const f64 * const restrict DU,
                   const f64 * const restrict DU2,
                   const int * const restrict ipiv,
                   f64 * const restrict B, const int ldb, int *info);

/*
 * Test fixture: holds all allocated memory for a single test case.
 */
typedef struct {
    int n, nrhs;
    int ldb;
    f64 *DL;      /* Original sub-diagonal */
    f64 *D;       /* Original diagonal */
    f64 *DU;      /* Original super-diagonal */
    f64 *DLF;     /* Factored sub-diagonal */
    f64 *DF;      /* Factored diagonal */
    f64 *DUF;     /* Factored super-diagonal */
    f64 *DU2;     /* Second super-diagonal from factorization */
    int *ipiv;       /* Pivot indices */
    f64 *XACT;    /* Exact solution */
    f64 *X;       /* Computed solution */
    f64 *B;       /* Right-hand side */
    f64 *B_copy;  /* Copy of RHS for verification */
    f64 *ferr;    /* Forward error estimates */
    f64 *berr;    /* Backward error estimates */
    f64 *work;    /* Workspace */
    int *iwork;      /* Integer workspace */
    f64 *AINV;    /* Workspace for explicit inverse computation */
    uint64_t seed;   /* RNG seed */
    uint64_t rng_state[4]; /* RNG state */
} dgtsvx_fixture_t;

/* Global seed for test sequence reproducibility */
static uint64_t g_seed = 1618;

/**
 * Generate a diagonally dominant tridiagonal matrix for testing.
 */
static void generate_gt_matrix(int n, int imat, f64 *DL, f64 *D, f64 *DU,
                                uint64_t state[static 4])
{
    char type, dist;
    int kl, ku, mode;
    f64 anorm, cndnum;
    int i;

    if (n <= 0) return;

    dlatb4("DGT", imat, n, n, &type, &kl, &ku, &anorm, &mode, &cndnum, &dist);

    /* Generate diagonally dominant matrix for stability */
    for (i = 0; i < n; i++) {
        D[i] = 4.0 + rng_uniform(state);
    }
    for (i = 0; i < n - 1; i++) {
        DL[i] = rng_uniform(state) - 0.5;
        DU[i] = rng_uniform(state) - 0.5;
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
}

/**
 * Compute the actual reciprocal condition number by explicit inversion.
 */
static f64 compute_true_rcond(int n, char norm_char,
                                 const f64 *DLF, const f64 *DF,
                                 const f64 *DUF, const f64 *DU2,
                                 const int *ipiv, f64 anorm,
                                 f64 *AINV)
{
    int i, j, info;
    f64 ainvnm = 0.0;
    int ldb = (n > 1) ? n : 1;

    /* Compute inverse by solving A * X = I */
    for (j = 0; j < n; j++) {
        for (i = 0; i < n; i++) {
            AINV[i + j * n] = (i == j) ? 1.0 : 0.0;
        }
        dgttrs("N", n, 1, DLF, DF, DUF, DU2, ipiv, AINV + j * n, ldb, &info);
    }

    /* Compute norm of inverse */
    if (norm_char == '1' || norm_char == 'O' || norm_char == 'o') {
        for (j = 0; j < n; j++) {
            f64 colsum = 0.0;
            for (i = 0; i < n; i++) {
                colsum += fabs(AINV[i + j * n]);
            }
            if (colsum > ainvnm) ainvnm = colsum;
        }
    } else {
        for (i = 0; i < n; i++) {
            f64 rowsum = 0.0;
            for (j = 0; j < n; j++) {
                rowsum += fabs(AINV[i + j * n]);
            }
            if (rowsum > ainvnm) ainvnm = rowsum;
        }
    }

    if (anorm > 0.0 && ainvnm > 0.0) {
        return (1.0 / anorm) / ainvnm;
    }
    return 0.0;
}

/**
 * Setup fixture: allocate memory for given dimensions.
 */
static int dgtsvx_setup(void **state, int n, int nrhs)
{
    dgtsvx_fixture_t *fix = malloc(sizeof(dgtsvx_fixture_t));
    assert_non_null(fix);

    int m = (n > 1) ? n - 1 : 0;
    int ldb = (n > 1) ? n : 1;

    fix->n = n;
    fix->nrhs = nrhs;
    fix->ldb = ldb;
    fix->seed = g_seed++;
    rng_seed(fix->rng_state, fix->seed);

    fix->DL = malloc((m > 0 ? m : 1) * sizeof(f64));
    fix->D = malloc(n * sizeof(f64));
    fix->DU = malloc((m > 0 ? m : 1) * sizeof(f64));
    fix->DLF = malloc((m > 0 ? m : 1) * sizeof(f64));
    fix->DF = malloc(n * sizeof(f64));
    fix->DUF = malloc((m > 0 ? m : 1) * sizeof(f64));
    fix->DU2 = malloc((n > 2 ? n - 2 : 1) * sizeof(f64));
    fix->ipiv = malloc(n * sizeof(int));
    fix->XACT = malloc(ldb * nrhs * sizeof(f64));
    fix->X = malloc(ldb * nrhs * sizeof(f64));
    fix->B = malloc(ldb * nrhs * sizeof(f64));
    fix->B_copy = malloc(ldb * nrhs * sizeof(f64));
    fix->ferr = malloc(nrhs * sizeof(f64));
    fix->berr = malloc(nrhs * sizeof(f64));
    fix->work = malloc(3 * n * sizeof(f64));
    fix->iwork = malloc(n * sizeof(int));
    fix->AINV = malloc(n * n * sizeof(f64));

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
    assert_non_null(fix->B_copy);
    assert_non_null(fix->ferr);
    assert_non_null(fix->berr);
    assert_non_null(fix->work);
    assert_non_null(fix->iwork);
    assert_non_null(fix->AINV);

    *state = fix;
    return 0;
}

/**
 * Teardown fixture: free all allocated memory.
 */
static int dgtsvx_teardown(void **state)
{
    dgtsvx_fixture_t *fix = *state;
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
        free(fix->B_copy);
        free(fix->ferr);
        free(fix->berr);
        free(fix->work);
        free(fix->iwork);
        free(fix->AINV);
        free(fix);
    }
    return 0;
}

/* Size-specific setup wrappers: sizes {1,2,3,5,10,50} x nrhs {1,2,15} */
static int setup_n1_nrhs1(void **state) { return dgtsvx_setup(state, 1, 1); }
static int setup_n2_nrhs1(void **state) { return dgtsvx_setup(state, 2, 1); }
static int setup_n3_nrhs1(void **state) { return dgtsvx_setup(state, 3, 1); }
static int setup_n5_nrhs1(void **state) { return dgtsvx_setup(state, 5, 1); }
static int setup_n10_nrhs1(void **state) { return dgtsvx_setup(state, 10, 1); }
static int setup_n50_nrhs1(void **state) { return dgtsvx_setup(state, 50, 1); }
static int setup_n1_nrhs2(void **state) { return dgtsvx_setup(state, 1, 2); }
static int setup_n2_nrhs2(void **state) { return dgtsvx_setup(state, 2, 2); }
static int setup_n3_nrhs2(void **state) { return dgtsvx_setup(state, 3, 2); }
static int setup_n5_nrhs2(void **state) { return dgtsvx_setup(state, 5, 2); }
static int setup_n10_nrhs2(void **state) { return dgtsvx_setup(state, 10, 2); }
static int setup_n50_nrhs2(void **state) { return dgtsvx_setup(state, 50, 2); }
static int setup_n1_nrhs15(void **state) { return dgtsvx_setup(state, 1, 15); }
static int setup_n2_nrhs15(void **state) { return dgtsvx_setup(state, 2, 15); }
static int setup_n3_nrhs15(void **state) { return dgtsvx_setup(state, 3, 15); }
static int setup_n5_nrhs15(void **state) { return dgtsvx_setup(state, 5, 15); }
static int setup_n10_nrhs15(void **state) { return dgtsvx_setup(state, 10, 15); }
static int setup_n50_nrhs15(void **state) { return dgtsvx_setup(state, 50, 15); }

/**
 * Result structure for dgtsvx verification (4 residuals).
 */
typedef struct {
    f64 solve_resid;   /* dgtt02: solution residual */
    f64 rcond_ratio;   /* dget06: condition number ratio */
    f64 ferr_resid;    /* dgtt05[0]: forward error bound */
    f64 berr_resid;    /* dgtt05[1]: backward error bound */
    int singular;         /* 1 if matrix was singular */
} dgtsvx_result_t;

/**
 * Core test logic: generate matrix, call dgtsvx, verify all results.
 */
static dgtsvx_result_t run_dgtsvx_test(dgtsvx_fixture_t *fix, int imat,
                                        const char* trans, const char* fact)
{
    dgtsvx_result_t result = {0.0, 0.0, 0.0, 0.0, 0};
    int info;
    int n = fix->n;
    int nrhs = fix->nrhs;
    int m = (n > 1) ? n - 1 : 0;
    int ldb = fix->ldb;
    int ldx = ldb;
    int i, j;
    f64 rcond;

    /* Generate test matrix */
    generate_gt_matrix(n, imat, fix->DL, fix->D, fix->DU, fix->rng_state);

    /* If fact = 'F', pre-compute the factorization */
    if (fact[0] == 'F') {
        memcpy(fix->DLF, fix->DL, (m > 0 ? m : 1) * sizeof(f64));
        memcpy(fix->DF, fix->D, n * sizeof(f64));
        memcpy(fix->DUF, fix->DU, (m > 0 ? m : 1) * sizeof(f64));
        dgttrf(n, fix->DLF, fix->DF, fix->DUF, fix->DU2, fix->ipiv, &info);
        if (info != 0) {
            result.singular = 1;
            return result;
        }
    }

    /* Generate random exact solution XACT */
    for (j = 0; j < nrhs; j++) {
        for (i = 0; i < n; i++) {
            fix->XACT[i + j * ldb] = rng_uniform_symmetric(fix->rng_state);
        }
    }

    /* Compute B = op(A) * XACT */
    for (j = 0; j < nrhs; j++) {
        for (i = 0; i < n; i++) {
            fix->B[i + j * ldb] = 0.0;
        }
    }
    dlagtm(trans, n, nrhs, 1.0, fix->DL, fix->D, fix->DU, fix->XACT, ldb,
           0.0, fix->B, ldb);

    /* Save B for verification */
    memcpy(fix->B_copy, fix->B, ldb * nrhs * sizeof(f64));

    /* Call dgtsvx */
    dgtsvx(fact, trans, n, nrhs, fix->DL, fix->D, fix->DU, fix->DLF, fix->DF,
           fix->DUF, fix->DU2, fix->ipiv, fix->B, ldb, fix->X, ldx, &rcond,
           fix->ferr, fix->berr, fix->work, fix->iwork, &info);

    /* info < 0 means illegal argument */
    assert_true(info >= 0);

    if (info > 0 && info <= n) {
        /* Singular matrix */
        result.singular = 1;
        return result;
    }

    /* Test 1: Solution residual ||B - op(A)*X|| / (||op(A)|| * ||X|| * eps) */
    dgtt02(trans, n, nrhs, fix->DL, fix->D, fix->DU, fix->X, ldx,
           fix->B_copy, ldb, &result.solve_resid);

    /* Test 2: Condition number estimate */
    int notran = (trans[0] == 'N' || trans[0] == 'n');
    f64 anorm = notran ? dlangt("1", n, fix->DL, fix->D, fix->DU)
                          : dlangt("I", n, fix->DL, fix->D, fix->DU);
    f64 rcondc = compute_true_rcond(n, notran ? '1' : 'I', fix->DLF, fix->DF,
                                        fix->DUF, fix->DU2, fix->ipiv, anorm,
                                        fix->AINV);

    if (rcondc > 0.0 && rcond > 0.0) {
        result.rcond_ratio = dget06(rcond, rcondc);
    }

    /* Test 3-4: Error bounds */
    f64 reslts[2];
    dgtt05(trans, n, nrhs, fix->DL, fix->D, fix->DU, fix->B_copy, ldb,
           fix->X, ldx, fix->XACT, ldb, fix->ferr, fix->berr, reslts);
    result.ferr_resid = reslts[0];
    result.berr_resid = reslts[1];

    return result;
}

/*
 * Test with fact='N' (dgtsvx does its own factorization), no transpose.
 * Types 1-6.
 */
static void test_dgtsvx_factN_notrans(void **state)
{
    dgtsvx_fixture_t *fix = *state;

    for (int imat = 1; imat <= 6; imat++) {
        fix->seed = g_seed++;
        rng_seed(fix->rng_state, fix->seed);
        dgtsvx_result_t r = run_dgtsvx_test(fix, imat, "N", "N");
        if (r.singular) continue;
        assert_residual_ok(r.solve_resid);
        if (r.rcond_ratio > 0.0) assert_residual_ok(r.rcond_ratio);
        assert_residual_ok(r.ferr_resid);
        assert_residual_ok(r.berr_resid);
    }
}

/*
 * Test with fact='N', transpose.
 * Types 1-6.
 */
static void test_dgtsvx_factN_trans(void **state)
{
    dgtsvx_fixture_t *fix = *state;

    for (int imat = 1; imat <= 6; imat++) {
        fix->seed = g_seed++;
        rng_seed(fix->rng_state, fix->seed);
        dgtsvx_result_t r = run_dgtsvx_test(fix, imat, "T", "N");
        if (r.singular) continue;
        assert_residual_ok(r.solve_resid);
        if (r.rcond_ratio > 0.0) assert_residual_ok(r.rcond_ratio);
        assert_residual_ok(r.ferr_resid);
        assert_residual_ok(r.berr_resid);
    }
}

/*
 * Test with fact='F' (pre-factored), no transpose.
 * Types 1-6.
 */
static void test_dgtsvx_factF_notrans(void **state)
{
    dgtsvx_fixture_t *fix = *state;

    for (int imat = 1; imat <= 6; imat++) {
        fix->seed = g_seed++;
        rng_seed(fix->rng_state, fix->seed);
        dgtsvx_result_t r = run_dgtsvx_test(fix, imat, "N", "F");
        if (r.singular) continue;
        assert_residual_ok(r.solve_resid);
        if (r.rcond_ratio > 0.0) assert_residual_ok(r.rcond_ratio);
        assert_residual_ok(r.ferr_resid);
        assert_residual_ok(r.berr_resid);
    }
}

/*
 * Test with fact='F' (pre-factored), transpose.
 * Types 1-6.
 */
static void test_dgtsvx_factF_trans(void **state)
{
    dgtsvx_fixture_t *fix = *state;

    for (int imat = 1; imat <= 6; imat++) {
        fix->seed = g_seed++;
        rng_seed(fix->rng_state, fix->seed);
        dgtsvx_result_t r = run_dgtsvx_test(fix, imat, "T", "F");
        if (r.singular) continue;
        assert_residual_ok(r.solve_resid);
        if (r.rcond_ratio > 0.0) assert_residual_ok(r.rcond_ratio);
        assert_residual_ok(r.ferr_resid);
        assert_residual_ok(r.berr_resid);
    }
}

/*
 * Macro to generate test entries for a given setup.
 */
#define DGTSVX_TESTS(setup_fn) \
    cmocka_unit_test_setup_teardown(test_dgtsvx_factN_notrans, setup_fn, dgtsvx_teardown), \
    cmocka_unit_test_setup_teardown(test_dgtsvx_factN_trans, setup_fn, dgtsvx_teardown), \
    cmocka_unit_test_setup_teardown(test_dgtsvx_factF_notrans, setup_fn, dgtsvx_teardown), \
    cmocka_unit_test_setup_teardown(test_dgtsvx_factF_trans, setup_fn, dgtsvx_teardown)

int main(void)
{
    const struct CMUnitTest tests[] = {
        /* nrhs = 1 */
        DGTSVX_TESTS(setup_n1_nrhs1),
        DGTSVX_TESTS(setup_n2_nrhs1),
        DGTSVX_TESTS(setup_n3_nrhs1),
        DGTSVX_TESTS(setup_n5_nrhs1),
        DGTSVX_TESTS(setup_n10_nrhs1),
        DGTSVX_TESTS(setup_n50_nrhs1),

        /* nrhs = 2 */
        DGTSVX_TESTS(setup_n1_nrhs2),
        DGTSVX_TESTS(setup_n2_nrhs2),
        DGTSVX_TESTS(setup_n3_nrhs2),
        DGTSVX_TESTS(setup_n5_nrhs2),
        DGTSVX_TESTS(setup_n10_nrhs2),
        DGTSVX_TESTS(setup_n50_nrhs2),

        /* nrhs = 15 */
        DGTSVX_TESTS(setup_n1_nrhs15),
        DGTSVX_TESTS(setup_n2_nrhs15),
        DGTSVX_TESTS(setup_n3_nrhs15),
        DGTSVX_TESTS(setup_n5_nrhs15),
        DGTSVX_TESTS(setup_n10_nrhs15),
        DGTSVX_TESTS(setup_n50_nrhs15),
    };

    return cmocka_run_group_tests_name("dgtsvx", tests, NULL, NULL);
}
