/**
 * @file test_zdrvsx.c
 * @brief Non-symmetric eigenvalue Schur form expert driver test - port of
 *        LAPACK TESTING/EIG/zdrvsx.f
 *
 * Tests the nonsymmetric eigenvalue problem expert driver ZGEESX.
 *
 * Each (n, jtype, iwk) combination is registered as a separate CMocka test
 * for random matrices (tests 1-15). An additional 20 precomputed matrices
 * test condition number accuracy (tests 16-17).
 *
 * Test ratios (17 total):
 *   (1)  0 if T is in Schur form, 1/ulp otherwise (no sorting)
 *   (2)  | A - VS T VS' | / ( n |A| ulp ) (no sorting)
 *   (3)  | I - VS VS' | / ( n ulp ) (no sorting)
 *   (4)  0 if W are eigenvalues of T (no sorting)
 *   (5)  0 if T(with VS) = T(without VS) (no sorting)
 *   (6)  0 if eigenvalues(with VS) = eigenvalues(without VS) (no sorting)
 *   (7)  0 if T is in Schur form (with sorting)
 *   (8)  | A - VS T VS' | / ( n |A| ulp ) (with sorting)
 *   (9)  | I - VS VS' | / ( n ulp ) (with sorting)
 *  (10)  0 if W are eigenvalues of T (with sorting)
 *  (11)  0 if T(with VS) = T(without VS) (with sorting)
 *  (12)  0 if eigenvalues(with VS) = eigenvalues(without VS) (with sorting)
 *  (13)  0 if sorting successful
 *  (14)  0 if RCONDE same no matter what else computed
 *  (15)  0 if RCONDV same no matter what else computed
 *  (16)  |RCONDE - RCONDE(precomputed)| / cond(RCONDE)
 *  (17)  |RCONDV - RCONDV(precomputed)| / cond(RCONDV)
 *
 * Matrix types (21 total):
 *   Types 1-3:   Zero, Identity, Jordan block
 *   Types 4-8:   Diagonal with scaled eigenvalues (via ZLATMS)
 *   Types 9-12:  Dense with controlled eigenvalues (via ZLATME, CONDS=1)
 *   Types 13-18: Dense with ill-conditioned eigenvectors (via ZLATME, CONDS=sqrt(ulp))
 *   Types 19-21: General matrices with random eigenvalues (via ZLATMR)
 */

#include "test_harness.h"
#include "verify.h"
#include "test_rng.h"
#include "zsx_testdata.h"
#include <math.h>
#include <string.h>
#include <complex.h>

/* Test threshold from zed.in */
#define THRESH 20.0

/* Maximum matrix type to test */
#define MAXTYP 21

/* Test dimensions from zed.in */
static const INT NVAL[] = {0, 1, 2, 3, 5, 10, 20};
#define NNVAL (sizeof(NVAL) / sizeof(NVAL[0]))

/* Test parameters for a single test case */
typedef struct {
    INT n;
    INT jtype;    /* Matrix type (1-21 for random, 22 for precomputed) */
    INT iwk;      /* Workspace variant (1=minimal, 2=generous) */
    INT precomp_idx; /* Index into ZSX_PRECOMPUTED (-1 for random) */
    char name[96];
} zdrvsx_params_t;

/* Workspace structure for all tests */
typedef struct {
    INT nmax;

    /* Matrices (all nmax x nmax) */
    c128* A;      /* Original matrix */
    c128* H;      /* Copy modified by ZGEESX */
    c128* HT;     /* Copy for comparison */
    c128* VS;     /* Schur vectors */
    c128* VS1;    /* Schur vectors (backup) */

    /* Eigenvalues */
    c128* W;      /* Eigenvalues */
    c128* WT;     /* Eigenvalues (temp) */
    c128* WTMP;   /* Eigenvalues (comparison) */

    /* Work arrays */
    c128* work;
    f64* rwork;
    INT* bwork;
    INT lwork;

    /* Test results */
    f64 result[17];

    /* RNG state */
    uint64_t rng_state[4];
} zdrvsx_workspace_t;

/* Global workspace pointer */
static zdrvsx_workspace_t* g_ws = NULL;

/* Matrix type parameters (from zdrvsx.f DATA statements) */
static const INT KTYPE[MAXTYP]  = {1, 2, 3, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 9, 9, 9};
static const INT KMAGN[MAXTYP]  = {1, 1, 1, 1, 1, 1, 2, 3, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 1, 2, 3};
static const INT KMODE[MAXTYP]  = {0, 0, 0, 4, 3, 1, 4, 4, 4, 3, 1, 5, 4, 3, 1, 5, 5, 5, 4, 3, 1};
static const INT KCONDS[MAXTYP] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 0, 0, 0};

/**
 * Group setup: allocate shared workspace.
 */
static int group_setup(void** state)
{
    (void)state;

    g_ws = malloc(sizeof(zdrvsx_workspace_t));
    if (!g_ws) return -1;

    /* 8 is the largest dimension in precomputed input */
    g_ws->nmax = 8;
    for (size_t i = 0; i < NNVAL; i++) {
        if (NVAL[i] > g_ws->nmax) g_ws->nmax = NVAL[i];
    }

    INT nmax = g_ws->nmax;
    INT n2 = nmax * nmax;

    /* Allocate matrices */
    g_ws->A   = malloc(n2 * sizeof(c128));
    g_ws->H   = malloc(n2 * sizeof(c128));
    g_ws->HT  = malloc(n2 * sizeof(c128));
    g_ws->VS  = malloc(n2 * sizeof(c128));
    g_ws->VS1 = malloc(n2 * sizeof(c128));
    g_ws->W    = malloc(nmax * sizeof(c128));
    g_ws->WT   = malloc(nmax * sizeof(c128));
    g_ws->WTMP = malloc(nmax * sizeof(c128));

    /* Workspace: max(2*N, N*(N+1)/2) complex, but at least 2*N^2 for zget24 */
    g_ws->lwork = 2 * n2;
    if (g_ws->lwork < 3 * nmax) g_ws->lwork = 3 * nmax;
    if (g_ws->lwork < 1) g_ws->lwork = 1;
    g_ws->work  = malloc(g_ws->lwork * sizeof(c128));

    /* RWORK dimension: N */
    g_ws->rwork = malloc(nmax * sizeof(f64));

    /* BWORK dimension: N */
    g_ws->bwork = malloc(nmax * sizeof(INT));

    if (!g_ws->A || !g_ws->H || !g_ws->HT || !g_ws->VS || !g_ws->VS1 ||
        !g_ws->W || !g_ws->WT || !g_ws->WTMP ||
        !g_ws->work || !g_ws->rwork || !g_ws->bwork) {
        return -1;
    }

    rng_seed(g_ws->rng_state, 0xDEADBEEFULL);
    return 0;
}

/**
 * Group teardown: free shared workspace.
 */
static int group_teardown(void** state)
{
    (void)state;

    if (g_ws) {
        free(g_ws->A);
        free(g_ws->H);
        free(g_ws->HT);
        free(g_ws->VS);
        free(g_ws->VS1);
        free(g_ws->W);
        free(g_ws->WT);
        free(g_ws->WTMP);
        free(g_ws->work);
        free(g_ws->rwork);
        free(g_ws->bwork);
        free(g_ws);
        g_ws = NULL;
    }
    return 0;
}

/**
 * Generate test matrix according to jtype.
 *
 * Based on zdrvsx.f lines 630-765. Identical to zdrves.f matrix generation.
 */
static INT generate_matrix(INT n, INT jtype, c128* A, INT lda,
                           c128* work, f64* rwork, INT* iwork,
                           uint64_t state[static 4])
{
    INT itype = KTYPE[jtype - 1];
    INT imode = KMODE[jtype - 1];
    f64 anorm, cond, conds;
    INT iinfo = 0;

    f64 ulp = dlamch("P");
    f64 unfl = dlamch("S");
    f64 ovfl = 1.0 / unfl;
    f64 ulpinv = 1.0 / ulp;
    f64 rtulp = sqrt(ulp);
    f64 rtulpi = 1.0 / rtulp;

    c128 czero = CMPLX(0.0, 0.0);
    c128 cone = CMPLX(1.0, 0.0);

    switch (KMAGN[jtype - 1]) {
        case 1: anorm = 1.0; break;
        case 2: anorm = ovfl * ulp; break;
        case 3: anorm = unfl * ulpinv; break;
        default: anorm = 1.0;
    }

    zlaset("F", n, n, czero, czero, A, lda);
    cond = ulpinv;

    if (itype == 1) {
        iinfo = 0;

    } else if (itype == 2) {
        for (INT j = 0; j < n; j++) {
            A[j + j * lda] = CMPLX(anorm, 0.0);
        }

    } else if (itype == 3) {
        for (INT j = 0; j < n; j++) {
            A[j + j * lda] = CMPLX(anorm, 0.0);
            if (j > 0) {
                A[j + (j - 1) * lda] = cone;
            }
        }

    } else if (itype == 4) {
        zlatms(n, n, "S", "H", rwork, imode, cond, anorm,
               0, 0, "N", A, lda, work + n, &iinfo, state);

    } else if (itype == 5) {
        zlatms(n, n, "S", "H", rwork, imode, cond, anorm,
               n, n, "N", A, lda, work + n, &iinfo, state);

    } else if (itype == 6) {
        if (KCONDS[jtype - 1] == 1) {
            conds = 1.0;
        } else if (KCONDS[jtype - 1] == 2) {
            conds = rtulpi;
        } else {
            conds = 0.0;
        }

        zlatme(n, "D", work, imode, cond, cone,
               "T", "T", "T", rwork, 4, conds,
               n, n, anorm, A, lda, work + 2 * n, &iinfo, state);

    } else if (itype == 7) {
        INT idumma[1] = {0};

        zlatmr(n, n, "D", "N", work, 6, 1.0, cone,
               "T", "N", work + n, 1, 1.0,
               work + 2 * n, 1, 1.0, "N", idumma, 0, 0,
               0.0, anorm, "NO", A, lda, iwork, &iinfo, state);

    } else if (itype == 8) {
        INT idumma[1] = {0};

        zlatmr(n, n, "D", "H", work, 6, 1.0, cone,
               "T", "N", work + n, 1, 1.0,
               work + 2 * n, 1, 1.0, "N", idumma, n, n,
               0.0, anorm, "NO", A, lda, iwork, &iinfo, state);

    } else if (itype == 9) {
        INT idumma[1] = {0};

        zlatmr(n, n, "D", "N", work, 6, 1.0, cone,
               "T", "N", work + n, 1, 1.0,
               work + 2 * n, 1, 1.0, "N", idumma, n, n,
               0.0, anorm, "NO", A, lda, iwork, &iinfo, state);

        if (n >= 4) {
            zlaset("F", 2, n, czero, czero, A, lda);
            zlaset("F", n - 3, 1, czero, czero, A + 2, lda);
            zlaset("F", n - 3, 2, czero, czero, A + 2 + (n - 2) * lda, lda);
            zlaset("F", 1, n, czero, czero, A + n - 1, lda);
        }

    } else if (itype == 10) {
        INT idumma[1] = {0};

        zlatmr(n, n, "D", "N", work, 6, 1.0, cone,
               "T", "N", work + n, 1, 1.0,
               work + 2 * n, 1, 1.0, "N", idumma, n, 0,
               0.0, anorm, "NO", A, lda, iwork, &iinfo, state);

    } else {
        iinfo = 1;
    }

    return iinfo;
}

/**
 * Run tests for a single random (n, jtype, iwk) combination.
 *
 * Based on zdrvsx.f lines 761-826.
 */
static void run_zdrvsx_random(zdrvsx_params_t* params)
{
    INT n = params->n;
    INT jtype = params->jtype;
    INT iwk = params->iwk;

    zdrvsx_workspace_t* ws = g_ws;
    INT lda = ws->nmax;
    INT ldvs = ws->nmax;

    c128* A = ws->A;
    c128* H = ws->H;
    c128* HT = ws->HT;
    c128* VS = ws->VS;
    c128* VS1 = ws->VS1;
    c128* W = ws->W;
    c128* WT = ws->WT;
    c128* WTMP = ws->WTMP;
    c128* work = ws->work;
    f64* rwork = ws->rwork;
    INT* bwork = ws->bwork;
    f64* result = ws->result;

    f64 ulpinv = 1.0 / dlamch("P");

    for (INT j = 0; j < 17; j++) {
        result[j] = -1.0;
    }

    if (n == 0) {
        return;
    }

    /* Generate matrix */
    INT iwork_dummy[1] = {0};
    INT iinfo = generate_matrix(n, jtype, A, lda, work, rwork,
                                iwork_dummy, ws->rng_state);
    if (iinfo != 0) {
        result[0] = ulpinv;
        print_message("Matrix generation failed for jtype=%d, n=%d, iinfo=%d\n",
                      jtype, n, iinfo);
        assert_info_success(iinfo);
        return;
    }

    /* Determine workspace size (zdrvsx.f lines 762-767) */
    INT nnwork;
    if (iwk == 1) {
        nnwork = 2 * n;
    } else {
        nnwork = 2 * n;
        INT nhalf = n * (n + 1) / 2;
        if (nhalf > nnwork) nnwork = nhalf;
    }
    if (nnwork < 1) nnwork = 1;

    /* Call zget24 with COMP=0 (zdrvsx.f lines 772-775) */
    INT info = 0;
    INT any_fail = 0;

    zget24(0, jtype, THRESH, n, A, lda, H, HT,
           W, WT, WTMP,
           VS, ldvs, VS1,
           0.0, 0.0,
           0, NULL, 0,
           result, work, nnwork, rwork, bwork, &info);

    /* Check for RESULT(j) > THRESH (zdrvsx.f lines 796-820) */
    for (INT j = 0; j < 15; j++) {
        if (result[j] >= 0.0 && result[j] >= THRESH) {
            print_message("N=%d, IWK=%d, type %d, test(%d)=%g\n",
                          n, iwk, jtype, j + 1, result[j]);
            any_fail = 1;
        }
    }

    assert_int_equal(any_fail, 0);
}

/**
 * Run tests for a single precomputed matrix.
 *
 * Based on zdrvsx.f lines 834-883.
 */
static void run_zdrvsx_precomp(zdrvsx_params_t* params)
{
    INT idx = params->precomp_idx;

    zdrvsx_workspace_t* ws = g_ws;
    INT lda = ws->nmax;
    INT ldvs = ws->nmax;

    c128* A = ws->A;
    c128* H = ws->H;
    c128* HT = ws->HT;
    c128* VS = ws->VS;
    c128* VS1 = ws->VS1;
    c128* W = ws->W;
    c128* WT = ws->WT;
    c128* WTMP = ws->WTMP;
    c128* work = ws->work;
    f64* rwork = ws->rwork;
    INT* bwork = ws->bwork;
    f64* result = ws->result;

    const zsx_precomputed_t* pc = &ZSX_PRECOMPUTED[idx];
    INT n = pc->n;

    for (INT j = 0; j < 17; j++) {
        result[j] = -1.0;
    }

    /* Copy precomputed matrix into workspace A with proper leading dimension */
    c128 czero = CMPLX(0.0, 0.0);
    zlaset("F", lda, n, czero, czero, A, lda);
    for (INT col = 0; col < n; col++) {
        for (INT row = 0; row < n; row++) {
            A[row + col * lda] = pc->A[row + col * n];
        }
    }

    INT info = 0;

    /* Call zget24 with COMP=1 (zdrvsx.f lines 831-834) */
    zget24(1, 22, THRESH, n, A, lda, H, HT,
           W, WT, WTMP,
           VS, ldvs, VS1,
           pc->rcdein, pc->rcdvin,
           pc->nslct, pc->islct, pc->isrt,
           result, work, ws->lwork, rwork, bwork, &info);

    /* Check for RESULT(j) > THRESH (zdrvsx.f lines 855-879) */
    INT any_fail = 0;
    for (INT j = 0; j < 17; j++) {
        if (result[j] >= 0.0 && result[j] >= THRESH) {
            print_message("N=%d, input example=%d, test(%d)=%g\n",
                          n, idx + 1, j + 1, result[j]);
            any_fail = 1;
        }
    }

    assert_int_equal(any_fail, 0);
}

/**
 * Test function wrappers.
 */
static void test_zdrvsx_random_case(void** state)
{
    zdrvsx_params_t* params = *state;
    run_zdrvsx_random(params);
}

static void test_zdrvsx_precomp_case(void** state)
{
    zdrvsx_params_t* params = *state;
    run_zdrvsx_precomp(params);
}

/*
 * Generate all parameter combinations.
 * Random: NNVAL * MAXTYP * 2 = 7 * 21 * 2 = 294 tests
 * Precomputed: ZSX_NUM_PRECOMPUTED = 20 tests
 * Total: 314 tests
 */

#define MAX_RANDOM_TESTS (NNVAL * MAXTYP * 2)
#define MAX_TESTS (MAX_RANDOM_TESTS + ZSX_NUM_PRECOMPUTED)

static zdrvsx_params_t g_params[MAX_TESTS];
static struct CMUnitTest g_tests[MAX_TESTS];
static INT g_num_tests = 0;

/**
 * Build the test array with all parameter combinations.
 */
static void build_test_array(void)
{
    g_num_tests = 0;

    /* Random tests: n x jtype x iwk */
    for (size_t in = 0; in < NNVAL; in++) {
        INT n = NVAL[in];

        for (INT jtype = 1; jtype <= MAXTYP; jtype++) {
            for (INT iwk = 1; iwk <= 2; iwk++) {
                zdrvsx_params_t* p = &g_params[g_num_tests];
                p->n = n;
                p->jtype = jtype;
                p->iwk = iwk;
                p->precomp_idx = -1;
                snprintf(p->name, sizeof(p->name),
                         "zdrvsx_n%d_type%d_wk%d", n, jtype, iwk);

                g_tests[g_num_tests].name = p->name;
                g_tests[g_num_tests].test_func = test_zdrvsx_random_case;
                g_tests[g_num_tests].setup_func = NULL;
                g_tests[g_num_tests].teardown_func = NULL;
                g_tests[g_num_tests].initial_state = p;

                g_num_tests++;
            }
        }
    }

    /* Precomputed tests */
    for (INT idx = 0; idx < ZSX_NUM_PRECOMPUTED; idx++) {
        zdrvsx_params_t* p = &g_params[g_num_tests];
        p->n = ZSX_PRECOMPUTED[idx].n;
        p->jtype = 22;
        p->iwk = 2;
        p->precomp_idx = idx;
        snprintf(p->name, sizeof(p->name),
                 "zdrvsx_precomp_%d_n%d", idx + 1, p->n);

        g_tests[g_num_tests].name = p->name;
        g_tests[g_num_tests].test_func = test_zdrvsx_precomp_case;
        g_tests[g_num_tests].setup_func = NULL;
        g_tests[g_num_tests].teardown_func = NULL;
        g_tests[g_num_tests].initial_state = p;

        g_num_tests++;
    }
}

/* ===== Main ===== */

int main(void)
{
    build_test_array();

    return _cmocka_run_group_tests("zdrvsx", g_tests, g_num_tests,
                                   group_setup, group_teardown);
}
