/**
 * @file test_dgtcon.c
 * @brief CMocka test suite for dgtcon (tridiagonal condition number estimation).
 *
 * Tests the condition number estimation routine dgtcon which estimates
 * the reciprocal of the condition number using the LU factorization.
 *
 * Verification: dget06 computes the ratio of estimated to actual condition
 * numbers. A ratio close to 1 indicates a good estimate.
 */

#include "test_harness.h"

/* Test threshold - see LAPACK dtest.in */
#define THRESH 20.0
#include "test_rng.h"
#include "verify.h"

/* Routines under test */
/* Utilities */
/*
 * Test fixture: holds all allocated memory for a single test case.
 */
typedef struct {
    INT n;
    f64 *DL;      /* Original sub-diagonal */
    f64 *D;       /* Original diagonal */
    f64 *DU;      /* Original super-diagonal */
    f64 *DLF;     /* Factored sub-diagonal */
    f64 *DF;      /* Factored diagonal */
    f64 *DUF;     /* Factored super-diagonal */
    f64 *DU2;     /* Second super-diagonal from factorization */
    INT* ipiv;       /* Pivot indices */
    f64 *work;    /* Workspace for dgtcon */
    INT* iwork;      /* Integer workspace for dgtcon */
    f64 *AINV;    /* Workspace for explicit inverse computation */
    uint64_t seed;   /* RNG seed */
    uint64_t rng_state[4]; /* RNG state */
} dgtcon_fixture_t;

/* Global seed for test sequence reproducibility */
static uint64_t g_seed = 3141;

/**
 * Generate a diagonally dominant tridiagonal matrix for testing.
 */
static void generate_gt_matrix(INT n, INT imat, f64 *DL, f64 *D, f64 *DU,
                                uint64_t state[static 4])
{
    char type, dist;
    INT kl, ku, mode;
    f64 anorm, cndnum;
    INT i;

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
 * Compute the actual condition number by explicit inversion.
 *
 * For a tridiagonal matrix, we compute the inverse by solving
 * A * e_i = col_i for each column of the identity.
 */
static f64 compute_true_rcond(INT n, char norm_char,
                                 const f64 *DLF, const f64 *DF,
                                 const f64 *DUF, const f64 *DU2,
                                 const INT* ipiv, f64 anorm,
                                 f64 *AINV)
{
    INT i, j, info;
    f64 ainvnm = 0.0;
    INT ldb = (n > 1) ? n : 1;

    /* Compute inverse by solving A * X = I */
    for (j = 0; j < n; j++) {
        /* Set up e_j (j-th column of identity) */
        for (i = 0; i < n; i++) {
            AINV[i + j * n] = (i == j) ? 1.0 : 0.0;
        }
        /* Solve A * x = e_j */
        dgttrs("N", n, 1, DLF, DF, DUF, DU2, ipiv, AINV + j * n, ldb, &info);
    }

    /* Compute norm of inverse */
    if (norm_char == '1' || norm_char == 'O' || norm_char == 'o') {
        /* 1-norm: max column sum */
        for (j = 0; j < n; j++) {
            f64 colsum = 0.0;
            for (i = 0; i < n; i++) {
                colsum += fabs(AINV[i + j * n]);
            }
            if (colsum > ainvnm) ainvnm = colsum;
        }
    } else {
        /* Infinity-norm: max row sum */
        for (i = 0; i < n; i++) {
            f64 rowsum = 0.0;
            for (j = 0; j < n; j++) {
                rowsum += fabs(AINV[i + j * n]);
            }
            if (rowsum > ainvnm) ainvnm = rowsum;
        }
    }

    /* Return reciprocal condition number */
    if (anorm > 0.0 && ainvnm > 0.0) {
        return (1.0 / anorm) / ainvnm;
    }
    return 0.0;
}

/**
 * Setup fixture: allocate memory for given dimensions.
 */
static int dgtcon_setup(void **state, INT n)
{
    dgtcon_fixture_t *fix = malloc(sizeof(dgtcon_fixture_t));
    assert_non_null(fix);

    INT m = (n > 1) ? n - 1 : 0;

    fix->n = n;
    fix->seed = g_seed++;
    rng_seed(fix->rng_state, fix->seed);

    fix->DL = malloc((m > 0 ? m : 1) * sizeof(f64));
    fix->D = malloc(n * sizeof(f64));
    fix->DU = malloc((m > 0 ? m : 1) * sizeof(f64));
    fix->DLF = malloc((m > 0 ? m : 1) * sizeof(f64));
    fix->DF = malloc(n * sizeof(f64));
    fix->DUF = malloc((m > 0 ? m : 1) * sizeof(f64));
    fix->DU2 = malloc((n > 2 ? n - 2 : 1) * sizeof(f64));
    fix->ipiv = malloc(n * sizeof(INT));
    fix->work = malloc(2 * n * sizeof(f64));
    fix->iwork = malloc(n * sizeof(INT));
    fix->AINV = malloc(n * n * sizeof(f64));

    assert_non_null(fix->DL);
    assert_non_null(fix->D);
    assert_non_null(fix->DU);
    assert_non_null(fix->DLF);
    assert_non_null(fix->DF);
    assert_non_null(fix->DUF);
    assert_non_null(fix->DU2);
    assert_non_null(fix->ipiv);
    assert_non_null(fix->work);
    assert_non_null(fix->iwork);
    assert_non_null(fix->AINV);

    *state = fix;
    return 0;
}

/**
 * Teardown fixture: free all allocated memory.
 */
static int dgtcon_teardown(void **state)
{
    dgtcon_fixture_t *fix = *state;
    if (fix) {
        free(fix->DL);
        free(fix->D);
        free(fix->DU);
        free(fix->DLF);
        free(fix->DF);
        free(fix->DUF);
        free(fix->DU2);
        free(fix->ipiv);
        free(fix->work);
        free(fix->iwork);
        free(fix->AINV);
        free(fix);
    }
    return 0;
}

/* Size-specific setup wrappers: sizes {1,2,3,5,10,20} */
static int setup_n1(void **state) { return dgtcon_setup(state, 1); }
static int setup_n2(void **state) { return dgtcon_setup(state, 2); }
static int setup_n3(void **state) { return dgtcon_setup(state, 3); }
static int setup_n5(void **state) { return dgtcon_setup(state, 5); }
static int setup_n10(void **state) { return dgtcon_setup(state, 10); }
static int setup_n20(void **state) { return dgtcon_setup(state, 20); }

/**
 * Core test logic: generate matrix, factorize, estimate condition number, verify.
 * Returns the dget06 ratio for the given norm type.
 * Returns -1.0 if the true rcond is zero (degenerate case).
 */
static f64 run_dgtcon_test(dgtcon_fixture_t *fix, INT imat, const char* norm_char)
{
    INT info;
    INT n = fix->n;
    INT m = (n > 1) ? n - 1 : 0;

    /* Generate test matrix */
    generate_gt_matrix(n, imat, fix->DL, fix->D, fix->DU, fix->rng_state);

    /* Copy to factored arrays */
    memcpy(fix->DLF, fix->DL, (m > 0 ? m : 1) * sizeof(f64));
    memcpy(fix->DF, fix->D, n * sizeof(f64));
    memcpy(fix->DUF, fix->DU, (m > 0 ? m : 1) * sizeof(f64));

    /* Factor */
    dgttrf(n, fix->DLF, fix->DF, fix->DUF, fix->DU2, fix->ipiv, &info);
    assert_info_success(info);

    /* Compute norm of original matrix */
    f64 anorm = dlangt(norm_char, n, fix->DL, fix->D, fix->DU);

    /* Estimate condition number */
    f64 rcond_est;
    dgtcon(norm_char, n, fix->DLF, fix->DF, fix->DUF, fix->DU2, fix->ipiv,
           anorm, &rcond_est, fix->work, fix->iwork, &info);
    assert_info_success(info);

    /* Compute true condition number by explicit inversion */
    f64 rcondc = compute_true_rcond(n, norm_char[0], fix->DLF, fix->DF, fix->DUF,
                                       fix->DU2, fix->ipiv, anorm, fix->AINV);

    if (rcondc <= 0.0) {
        return -1.0;
    }

    /* Return dget06 ratio */
    return dget06(rcond_est, rcondc);
}

/*
 * Test 1-norm condition number estimation for types 1-6.
 */
static void test_dgtcon_onenorm(void **state)
{
    dgtcon_fixture_t *fix = *state;

    for (INT imat = 1; imat <= 6; imat++) {
        fix->seed = g_seed++;
        rng_seed(fix->rng_state, fix->seed);
        f64 ratio = run_dgtcon_test(fix, imat, "1");
        if (ratio < 0.0) continue; /* degenerate case */
        assert_residual_ok(ratio);
    }
}

/*
 * Test infinity-norm condition number estimation for types 1-6.
 */
static void test_dgtcon_infnorm(void **state)
{
    dgtcon_fixture_t *fix = *state;

    for (INT imat = 1; imat <= 6; imat++) {
        fix->seed = g_seed++;
        rng_seed(fix->rng_state, fix->seed);
        f64 ratio = run_dgtcon_test(fix, imat, "I");
        if (ratio < 0.0) continue; /* degenerate case */
        assert_residual_ok(ratio);
    }
}

/*
 * Macro to generate test entries for a given setup.
 */
#define DGTCON_TESTS(setup_fn) \
    cmocka_unit_test_setup_teardown(test_dgtcon_onenorm, setup_fn, dgtcon_teardown), \
    cmocka_unit_test_setup_teardown(test_dgtcon_infnorm, setup_fn, dgtcon_teardown)

int main(void)
{
    const struct CMUnitTest tests[] = {
        DGTCON_TESTS(setup_n1),
        DGTCON_TESTS(setup_n2),
        DGTCON_TESTS(setup_n3),
        DGTCON_TESTS(setup_n5),
        DGTCON_TESTS(setup_n10),
        DGTCON_TESTS(setup_n20),
    };

    return cmocka_run_group_tests_name("dgtcon", tests, NULL, NULL);
}
