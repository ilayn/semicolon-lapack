/**
 * @file test_dsytrf.c
 * @brief CMocka test suite for dsytrf and dsytf2 (symmetric indefinite factorization).
 *
 * Tests the blocked (dsytrf) and unblocked (dsytf2) symmetric indefinite
 * factorization routines using LAPACK's verification methodology with
 * normalized residuals.
 *
 * Verification: dsyt01 computes ||L*D*L' - A|| / (N * ||A|| * eps)
 *
 * Matrix types tested (from dlatb4 with "DSY" path, types 1-10):
 *   1. Diagonal
 *   2. Random, well-conditioned (cond=2)
 *   3. First row/column zero
 *   4. Last row/column zero
 *   5. Middle row/column zero
 *   6. Last n/2 rows/columns zero
 *   7. Random, moderate condition
 *   8. Random, ill-conditioned (cond ~ 3e7)
 *   9. Random, very ill-conditioned (cond ~ 9e15)
 *   10. Scaled near underflow
 *
 * Both UPLO='U' and UPLO='L' are tested for each matrix type and size.
 *
 * Parameters from dtest.in:
 *   N values: 0, 1, 2, 3, 5, 10, 50
 *   Matrix types: 1-10
 *   THRESH: 30.0
 *
 * This provides extra coverage for dsytf2 (unblocked) which LAPACK only
 * error-checks in derrsy.f but does not functionally test in dchksy.f.
 */

#include "test_harness.h"
#include "verify.h"
#include "test_rng.h"
/* Test parameters from dtest.in */
static const INT NVAL[] = {0, 1, 2, 3, 5, 10, 50};
static const char UPLOS[] = {'U', 'L'};

#define NN      (sizeof(NVAL) / sizeof(NVAL[0]))
#define NUPLO   (sizeof(UPLOS) / sizeof(UPLOS[0]))
#define NTYPES  10
#define THRESH  30.0
#define NMAX    50  /* Maximum matrix dimension */

/* Routines under test */

/**
 * Test parameters for a single test case.
 */
typedef struct {
    INT n;
    INT imat;       /* Matrix type 1-10 */
    INT iuplo;      /* 0='U', 1='L' */
    INT routine;    /* 0=dsytf2 (unblocked), 1=dsytrf (blocked) */
    char name[64];
} dsytrf_params_t;

/**
 * Workspace for test execution - shared across all tests via group setup.
 */
typedef struct {
    f64* A;      /* Original matrix (NMAX x NMAX) */
    f64* AFAC;   /* Factored matrix (NMAX x NMAX) */
    f64* C;      /* Workspace for dsyt01 (NMAX x NMAX) */
    f64* WORK;   /* General workspace */
    f64* RWORK;  /* Real workspace for dsyt01 */
    f64* D;      /* Singular values for dlatms */
    INT* IPIV;      /* Pivot indices */
} dsytrf_workspace_t;

static dsytrf_workspace_t* g_workspace = NULL;

/**
 * Group setup - allocate workspace once for all tests.
 */
static int group_setup(void** state)
{
    (void)state;
    g_workspace = malloc(sizeof(dsytrf_workspace_t));
    if (!g_workspace) return -1;

    INT lwork = NMAX * 64;  /* Generous workspace for dsytrf */

    g_workspace->A = malloc(NMAX * NMAX * sizeof(f64));
    g_workspace->AFAC = malloc(NMAX * NMAX * sizeof(f64));
    g_workspace->C = malloc(NMAX * NMAX * sizeof(f64));
    g_workspace->WORK = malloc(lwork * sizeof(f64));
    g_workspace->RWORK = malloc(NMAX * sizeof(f64));
    g_workspace->D = malloc(NMAX * sizeof(f64));
    g_workspace->IPIV = malloc(NMAX * sizeof(INT));

    if (!g_workspace->A || !g_workspace->AFAC || !g_workspace->C ||
        !g_workspace->WORK || !g_workspace->RWORK || !g_workspace->D ||
        !g_workspace->IPIV) {
        return -1;
    }

    return 0;
}

/**
 * Group teardown - free workspace.
 */
static int group_teardown(void** state)
{
    (void)state;
    if (g_workspace) {
        free(g_workspace->A);
        free(g_workspace->AFAC);
        free(g_workspace->C);
        free(g_workspace->WORK);
        free(g_workspace->RWORK);
        free(g_workspace->D);
        free(g_workspace->IPIV);
        free(g_workspace);
        g_workspace = NULL;
    }
    return 0;
}

/**
 * Run the factorization test for a single (n, uplo, imat, routine) combination.
 */
static void run_dsytrf_single(INT n, INT iuplo, INT imat, INT routine)
{
    const f64 ZERO = 0.0;
    dsytrf_workspace_t* ws = g_workspace;

    char type, dist;
    char uplo = UPLOS[iuplo];
    char uplo_str[2] = {uplo, '\0'};
    INT kl, ku, mode;
    f64 anorm, cndnum;
    INT info, izero;
    INT lda = (n > 1) ? n : 1;
    INT lwork = NMAX * 64;
    f64 resid;
    char ctx[128];

    /* Get matrix parameters for this type */
    dlatb4("DSY", imat, n, n, &type, &kl, &ku, &anorm, &mode, &cndnum, &dist);

    /* Generate symmetric test matrix */
    uint64_t rng_state[4];
    rng_seed(rng_state, 1988198919901991ULL + (uint64_t)(n * 1000 + iuplo * 100 + imat * 10 + routine));
    dlatms(n, n, &dist, &type, ws->D, mode, cndnum, anorm,
           kl, ku, uplo_str, ws->A, lda, ws->WORK, &info, rng_state);
    assert_int_equal(info, 0);

    /* For types 3-6, zero one or more rows and columns.
     * izero is 0-based index of the row/column to zero. */
    INT zerot = (imat >= 3 && imat <= 6);
    if (zerot) {
        if (imat == 3) {
            izero = 0;  /* First row/column */
        } else if (imat == 4) {
            izero = n - 1;  /* Last row/column */
        } else {
            izero = n / 2;  /* Middle row/column */
        }

        if (imat < 6) {
            /* Zero row and column izero */
            if (iuplo == 0) {
                /* Upper: zero column izero (rows 0 to izero-1) and
                 * row izero (columns izero to n-1) */
                INT ioff = izero * lda;
                for (INT i = 0; i < izero; i++) {
                    ws->A[ioff + i] = ZERO;
                }
                ioff += izero;
                for (INT i = izero; i < n; i++) {
                    ws->A[ioff] = ZERO;
                    ioff += lda;
                }
            } else {
                /* Lower: zero row izero (columns 0 to izero-1) and
                 * column izero (rows izero to n-1) */
                INT ioff = izero;
                for (INT i = 0; i < izero; i++) {
                    ws->A[ioff] = ZERO;
                    ioff += lda;
                }
                ioff = izero * lda + izero;
                for (INT i = izero; i < n; i++) {
                    ws->A[ioff + i - izero] = ZERO;
                }
            }
        } else {
            /* Type 6: zero first izero+1 rows and columns (upper) or last (lower) */
            if (iuplo == 0) {
                INT ioff = 0;
                for (INT j = 0; j <= izero; j++) {
                    for (INT i = 0; i <= j; i++) {
                        ws->A[ioff + i] = ZERO;
                    }
                    ioff += lda;
                }
            } else {
                INT ioff = izero * lda + izero;
                for (INT j = izero; j < n; j++) {
                    for (INT i = j; i < n; i++) {
                        ws->A[ioff + i - j] = ZERO;
                    }
                    ioff += lda;
                }
            }
        }
    }

    /* Copy A to AFAC for factorization */
    memcpy(ws->AFAC, ws->A, lda * n * sizeof(f64));

    /* Set test context for error messages */
    snprintf(ctx, sizeof(ctx), "n=%d uplo=%c imat=%d routine=%s",
             n, uplo, imat, routine == 0 ? "dsytf2" : "dsytrf");
    set_test_context(ctx);

    /* Factorize */
    if (routine == 0) {
        dsytf2(uplo_str, n, ws->AFAC, lda, ws->IPIV, &info);
    } else {
        dsytrf(uplo_str, n, ws->AFAC, lda, ws->IPIV, ws->WORK, lwork, &info);
    }

    /* For singular types (3-6), INFO may be 0 or > 0 depending on pivoting */
    if (zerot) {
        assert_true(info >= 0);
    } else {
        assert_info_success(info);
    }

    /* Verify factorization using dsyt01 (even for singular matrices) */
    dsyt01(uplo_str, n, ws->A, lda, ws->AFAC, lda, ws->IPIV,
           ws->C, lda, ws->RWORK, &resid);
    assert_residual_ok(resid);

    clear_test_context();
}

/**
 * Test function called by CMocka for each parameterized test case.
 */
static void test_dsytrf_case(void** state)
{
    dsytrf_params_t* p = (dsytrf_params_t*)*state;
    run_dsytrf_single(p->n, p->iuplo, p->imat, p->routine);
}

/* Maximum number of test cases:
 * 7 N values × 10 types × 2 UPLO × 2 routines = 280 max
 * But we skip some type/size combinations, so actual count is less. */
#define MAX_TESTS (NN * NTYPES * NUPLO * 2)

static dsytrf_params_t g_params[MAX_TESTS];
static struct CMUnitTest g_tests[MAX_TESTS];
static INT g_num_tests = 0;

/**
 * Build the test array with all parameter combinations.
 */
static void build_test_array(void)
{
    g_num_tests = 0;

    for (INT in = 0; in < (INT)NN; in++) {
        INT n = NVAL[in];

        INT nimat = NTYPES;
        if (n <= 0) {
            nimat = 1;
        }

        for (INT imat = 1; imat <= nimat; imat++) {
            /* Skip types 3, 4, 5, or 6 if matrix size is too small */
            INT zerot = (imat >= 3 && imat <= 6);
            if (zerot && n < imat - 2) {
                continue;
            }

            for (INT iuplo = 0; iuplo < (INT)NUPLO; iuplo++) {
                /* Loop over routines: 0=dsytf2 (unblocked), 1=dsytrf (blocked) */
                for (INT routine = 0; routine <= 1; routine++) {
                    /* Store parameters */
                    dsytrf_params_t* p = &g_params[g_num_tests];
                    p->n = n;
                    p->imat = imat;
                    p->iuplo = iuplo;
                    p->routine = routine;
                    snprintf(p->name, sizeof(p->name), "%s_n%d_%c_type%d",
                             routine == 0 ? "dsytf2" : "dsytrf",
                             n, UPLOS[iuplo], imat);

                    /* Create CMocka test entry */
                    g_tests[g_num_tests].name = p->name;
                    g_tests[g_num_tests].test_func = test_dsytrf_case;
                    g_tests[g_num_tests].setup_func = NULL;
                    g_tests[g_num_tests].teardown_func = NULL;
                    g_tests[g_num_tests].initial_state = p;

                    g_num_tests++;
                }
            }
        }
    }
}

int main(void)
{
    /* Build all test cases */
    build_test_array();

    /* Run all tests with shared workspace.
     * We use _cmocka_run_group_tests directly because the test array
     * is built dynamically and the standard macro uses sizeof() which
     * only works for compile-time array sizes. */
    return _cmocka_run_group_tests("dsytrf", g_tests, g_num_tests,
                                   group_setup, group_teardown);
}
