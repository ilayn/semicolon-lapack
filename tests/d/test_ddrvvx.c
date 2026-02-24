/**
 * @file test_ddrvvx.c
 * @brief Non-symmetric eigenvalue expert driver test - port of LAPACK TESTING/EIG/ddrvvx.f
 *
 * Tests the nonsymmetric eigenvalue problem expert driver DGEEVX.
 *
 * Each (n, jtype, iwk) combination is registered as a separate CMocka test
 * for random matrices (tests 1-9). An additional 39 precomputed matrices
 * test condition number accuracy (tests 10-11).
 *
 * Test ratios (11 total):
 *   (1)  | A * VR - VR * W | / ( n |A| ulp )
 *   (2)  | A^T * VL - VL * W^H | / ( n |A| ulp )
 *   (3)  | |VR(i)| - 1 | / ulp and whether largest component real
 *   (4)  | |VL(i)| - 1 | / ulp and whether largest component real
 *   (5)  W(full) = W(partial)
 *   (6)  VR(full) = VR(partial)
 *   (7)  VL(full) = VL(partial)
 *   (8)  0 if SCALE, ILO, IHI, ABNRM (full) = (partial), 1/ulp otherwise
 *   (9)  RCONDV(full) = RCONDV(partial)
 *  (10)  |RCONDV - RCDVIN| / cond(RCONDV)
 *  (11)  |RCONDE - RCDEIN| / cond(RCONDE)
 *
 * Matrix types (21 total):
 *   Types 1-3:   Zero, Identity, Jordan block
 *   Types 4-8:   Diagonal with scaled eigenvalues (via DLATMS)
 *   Types 9-12:  Dense with controlled eigenvalues (via DLATME, CONDS=1)
 *   Types 13-18: Dense with ill-conditioned eigenvectors (via DLATME, CONDS=sqrt(ulp))
 *   Types 19-21: General matrices with random eigenvalues (via DLATMR)
 */

#include "test_harness.h"
#include "verify.h"
#include "test_rng.h"
#include "dvx_testdata.h"
#include <math.h>
#include <string.h>

/* Test threshold from ded.in */
#define THRESH 20.0

/* Maximum matrix type to test */
#define MAXTYP 21

/* Test dimensions from ded.in */
static const INT NVAL[] = {0, 1, 2, 3, 5, 10, 20};
#define NNVAL (sizeof(NVAL) / sizeof(NVAL[0]))

/* Balancing options */
static const char* BAL[] = {"N", "P", "S", "B"};

/* External function declarations */
/* Test parameters for a single test case */
typedef struct {
    INT n;
    INT jtype;    /* Matrix type (1-21 for random, 22 for precomputed) */
    INT iwk;      /* Workspace variant (1=minimal, 2=medium, 3=generous) */
    INT precomp_idx; /* Index into DVX_PRECOMPUTED (-1 for random) */
    char name[96];
} ddrvvx_params_t;

/* Workspace structure for all tests */
typedef struct {
    INT nmax;

    /* Matrices (all nmax x nmax) */
    f64* A;      /* Original matrix */
    f64* H;      /* Copy modified by DGEEVX */
    f64* VL;     /* Left eigenvectors (full) */
    f64* VR;     /* Right eigenvectors (full) */
    f64* LRE;    /* Left/right eigenvectors (partial) */

    /* Eigenvalues */
    f64* WR;     /* Real parts (full) */
    f64* WI;     /* Imaginary parts (full) */
    f64* WR1;    /* Real parts (partial) */
    f64* WI1;    /* Imaginary parts (partial) */

    /* Condition numbers */
    f64* rcondv;  /* Reciprocal eigenvector condition numbers */
    f64* rcndv1;  /* Partial eigenvector condition numbers */
    f64* rcdvin;  /* Precomputed eigenvector condition numbers */
    f64* rconde;  /* Reciprocal eigenvalue condition numbers */
    f64* rcnde1;  /* Partial eigenvalue condition numbers */
    f64* rcdein;  /* Precomputed eigenvalue condition numbers */

    /* Balancing */
    f64* scale;
    f64* scale1;

    /* Work arrays */
    f64* work;
    INT* iwork;
    INT lwork;

    /* Test results */
    f64 result[11];

    /* RNG state */
    uint64_t rng_state[4];
} ddrvvx_workspace_t;

/* Global workspace pointer */
static ddrvvx_workspace_t* g_ws = NULL;

/* Matrix type parameters (from ddrvvx.f DATA statements)
 * KTYPE: 1, 2, 3, 5*4, 4*6, 6*6, 3*9
 * KMAGN: 3*1, 1, 1, 1, 2, 3, 4*1, 1, 1, 1, 1, 2, 3, 1, 2, 3
 * KMODE: 3*0, 4, 3, 1, 4, 4, 4, 3, 1, 5, 4, 3, 1, 5, 5, 5, 4, 3, 1
 * KCONDS: 3*0, 5*0, 4*1, 6*2, 3*0
 */
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

    g_ws = malloc(sizeof(ddrvvx_workspace_t));
    if (!g_ws) return -1;

    /* Find maximum N; 12 is the largest dimension in precomputed input */
    g_ws->nmax = 12;
    for (size_t i = 0; i < NNVAL; i++) {
        if (NVAL[i] > g_ws->nmax) g_ws->nmax = NVAL[i];
    }

    INT nmax = g_ws->nmax;
    INT n2 = nmax * nmax;

    /* Allocate matrices */
    g_ws->A   = malloc(n2 * sizeof(f64));
    g_ws->H   = malloc(n2 * sizeof(f64));
    g_ws->VL  = malloc(n2 * sizeof(f64));
    g_ws->VR  = malloc(n2 * sizeof(f64));
    g_ws->LRE = malloc(n2 * sizeof(f64));
    g_ws->WR  = malloc(nmax * sizeof(f64));
    g_ws->WI  = malloc(nmax * sizeof(f64));
    g_ws->WR1 = malloc(nmax * sizeof(f64));
    g_ws->WI1 = malloc(nmax * sizeof(f64));

    /* Condition numbers */
    g_ws->rcondv = malloc(nmax * sizeof(f64));
    g_ws->rcndv1 = malloc(nmax * sizeof(f64));
    g_ws->rcdvin = malloc(nmax * sizeof(f64));
    g_ws->rconde = malloc(nmax * sizeof(f64));
    g_ws->rcnde1 = malloc(nmax * sizeof(f64));
    g_ws->rcdein = malloc(nmax * sizeof(f64));

    /* Balancing */
    g_ws->scale  = malloc(nmax * sizeof(f64));
    g_ws->scale1 = malloc(nmax * sizeof(f64));

    /* Workspace: 6*N + 2*N^2 (ddrvvx.f line 461) */
    g_ws->lwork = 6 * nmax + 2 * n2;
    if (g_ws->lwork < 360) g_ws->lwork = 360;
    g_ws->work  = malloc(g_ws->lwork * sizeof(f64));

    /* IWORK dimension: 2*max(NN,12) (ddrvvx.f line 467) */
    g_ws->iwork = malloc(2 * nmax * sizeof(INT));

    if (!g_ws->A || !g_ws->H || !g_ws->VL || !g_ws->VR || !g_ws->LRE ||
        !g_ws->WR || !g_ws->WI || !g_ws->WR1 || !g_ws->WI1 ||
        !g_ws->rcondv || !g_ws->rcndv1 || !g_ws->rcdvin ||
        !g_ws->rconde || !g_ws->rcnde1 || !g_ws->rcdein ||
        !g_ws->scale || !g_ws->scale1 ||
        !g_ws->work || !g_ws->iwork) {
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
        free(g_ws->VL);
        free(g_ws->VR);
        free(g_ws->LRE);
        free(g_ws->WR);
        free(g_ws->WI);
        free(g_ws->WR1);
        free(g_ws->WI1);
        free(g_ws->rcondv);
        free(g_ws->rcndv1);
        free(g_ws->rcdvin);
        free(g_ws->rconde);
        free(g_ws->rcnde1);
        free(g_ws->rcdein);
        free(g_ws->scale);
        free(g_ws->scale1);
        free(g_ws->work);
        free(g_ws->iwork);
        free(g_ws);
        g_ws = NULL;
    }
    return 0;
}

/**
 * Generate test matrix according to jtype.
 *
 * Based on ddrvvx.f lines 694-826.
 */
static INT generate_matrix(INT n, INT jtype, f64* A, INT lda,
                           f64* work, INT* iwork, uint64_t state[static 4])
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

    /* Compute norm based on KMAGN */
    switch (KMAGN[jtype - 1]) {
        case 1: anorm = 1.0; break;
        case 2: anorm = ovfl * ulp; break;
        case 3: anorm = unfl * ulpinv; break;
        default: anorm = 1.0;
    }

    dlaset("F", lda, n, 0.0, 0.0, A, lda);
    cond = ulpinv;

    if (itype == 1) {
        /* Zero */
        iinfo = 0;

    } else if (itype == 2) {
        /* Identity */
        for (INT jcol = 0; jcol < n; jcol++) {
            A[jcol + jcol * lda] = anorm;
        }

    } else if (itype == 3) {
        /* Jordan Block */
        for (INT jcol = 0; jcol < n; jcol++) {
            A[jcol + jcol * lda] = anorm;
            if (jcol > 0) {
                A[jcol + (jcol - 1) * lda] = 1.0;
            }
        }

    } else if (itype == 4) {
        /* Diagonal Matrix, [Eigen]values Specified */
        dlatms(n, n, "S", "S", work, imode, cond, anorm,
               0, 0, "N", A, lda, work + n, &iinfo, state);

    } else if (itype == 5) {
        /* Symmetric, eigenvalues specified */
        dlatms(n, n, "S", "S", work, imode, cond, anorm,
               n, n, "N", A, lda, work + n, &iinfo, state);

    } else if (itype == 6) {
        /* General, eigenvalues specified */
        if (KCONDS[jtype - 1] == 1) {
            conds = 1.0;
        } else if (KCONDS[jtype - 1] == 2) {
            conds = rtulpi;
        } else {
            conds = 0.0;
        }

        char ei[2] = " ";
        dlatme(n, "S", work, imode, cond, 1.0,
               ei, "T", "T", "T", work + n, 4, conds,
               n, n, anorm, A, lda, work + 2 * n, &iinfo, state);

    } else if (itype == 7) {
        /* Diagonal, random eigenvalues */
        INT idumma[1] = {1};
        dlatmr(n, n, "S", "S", work, 6, 1.0, 1.0, "T", "N",
               work + n, 1, 1.0, work + 2 * n, 1, 1.0,
               "N", idumma, 0, 0, 0.0, anorm, "NO",
               A, lda, iwork, &iinfo, state);

    } else if (itype == 8) {
        /* Symmetric, random eigenvalues */
        INT idumma[1] = {1};
        dlatmr(n, n, "S", "S", work, 6, 1.0, 1.0, "T", "N",
               work + n, 1, 1.0, work + 2 * n, 1, 1.0,
               "N", idumma, n, n, 0.0, anorm, "NO",
               A, lda, iwork, &iinfo, state);

    } else if (itype == 9) {
        /* General, random eigenvalues */
        INT idumma[1] = {1};
        dlatmr(n, n, "S", "N", work, 6, 1.0, 1.0, "T", "N",
               work + n, 1, 1.0, work + 2 * n, 1, 1.0,
               "N", idumma, n, n, 0.0, anorm, "NO",
               A, lda, iwork, &iinfo, state);

        if (n >= 4) {
            dlaset("F", 2, n, 0.0, 0.0, A, lda);
            dlaset("F", n - 3, 1, 0.0, 0.0, A + 2, lda);
            dlaset("F", n - 3, 2, 0.0, 0.0, A + 2 + (n - 2) * lda, lda);
            dlaset("F", 1, n, 0.0, 0.0, A + n - 1, lda);
        }

    } else if (itype == 10) {
        /* Triangular, random eigenvalues */
        INT idumma[1] = {1};
        dlatmr(n, n, "S", "N", work, 6, 1.0, 1.0, "T", "N",
               work + n, 1, 1.0, work + 2 * n, 1, 1.0,
               "N", idumma, n, 0, 0.0, anorm, "NO",
               A, lda, iwork, &iinfo, state);

    } else {
        iinfo = 1;
    }

    return iinfo;
}

/**
 * Run tests for a single random (n, jtype, iwk) combination.
 *
 * Based on ddrvvx.f lines 839-897.
 */
static void run_ddrvvx_random(ddrvvx_params_t* params)
{
    INT n = params->n;
    INT jtype = params->jtype;
    INT iwk = params->iwk;

    ddrvvx_workspace_t* ws = g_ws;
    INT lda = ws->nmax;
    INT ldvl = ws->nmax;
    INT ldvr = ws->nmax;
    INT ldlre = ws->nmax;

    f64* A = ws->A;
    f64* H = ws->H;
    f64* VL = ws->VL;
    f64* VR = ws->VR;
    f64* LRE = ws->LRE;
    f64* WR = ws->WR;
    f64* WI = ws->WI;
    f64* WR1 = ws->WR1;
    f64* WI1 = ws->WI1;
    f64* rcondv = ws->rcondv;
    f64* rcndv1 = ws->rcndv1;
    f64* rcdvin = ws->rcdvin;
    f64* rconde = ws->rconde;
    f64* rcnde1 = ws->rcnde1;
    f64* rcdein = ws->rcdein;
    f64* scale_ = ws->scale;
    f64* scale1 = ws->scale1;
    f64* work = ws->work;
    INT* iwork = ws->iwork;
    f64* result = ws->result;

    f64 ulpinv = 1.0 / dlamch("P");

    for (INT j = 0; j < 11; j++) {
        result[j] = -1.0;
    }

    if (n == 0) {
        return;
    }

    /* Generate matrix */
    INT iinfo = generate_matrix(n, jtype, A, lda, work, iwork, ws->rng_state);
    if (iinfo != 0) {
        result[0] = ulpinv;
        print_message("Matrix generation failed for jtype=%d, n=%d, iinfo=%d\n",
                      jtype, n, iinfo);
        assert_info_success(iinfo);
        return;
    }

    /* Determine workspace size (ddrvvx.f lines 839-847) */
    INT nnwork;
    if (iwk == 1) {
        nnwork = 3 * n;
    } else if (iwk == 2) {
        nnwork = 6 * n + n * n;
    } else {
        nnwork = 6 * n + 2 * n * n;
    }
    if (nnwork < 1) nnwork = 1;

    /* Test for all balancing options (ddrvvx.f lines 851-895) */
    INT info = 0;
    INT any_fail = 0;

    for (INT ibal = 0; ibal < 4; ibal++) {
        dget23(0, BAL[ibal], jtype, THRESH, n,
               A, lda, H, WR, WI, WR1, WI1,
               VL, ldvl, VR, ldvr, LRE, ldlre,
               rcondv, rcndv1, rcdvin, rconde, rcnde1, rcdein,
               scale_, scale1, result, work, nnwork, iwork, &info);

        /* Check for RESULT(j) > THRESH (ddrvvx.f lines 867-893) */
        for (INT j = 0; j < 9; j++) {
            if (result[j] >= 0.0 && result[j] >= THRESH) {
                print_message("BALANC='%s', N=%d, IWK=%d, type %d, test(%d)=%g\n",
                              BAL[ibal], n, iwk, jtype, j + 1, result[j]);
                any_fail = 1;
            }
        }
    }

    assert_int_equal(any_fail, 0);
}

/**
 * Run tests for a single precomputed matrix.
 *
 * Based on ddrvvx.f lines 907-959.
 */
static void run_ddrvvx_precomp(ddrvvx_params_t* params)
{
    INT idx = params->precomp_idx;

    ddrvvx_workspace_t* ws = g_ws;
    INT lda = ws->nmax;
    INT ldvl = ws->nmax;
    INT ldvr = ws->nmax;
    INT ldlre = ws->nmax;

    f64* A = ws->A;
    f64* H = ws->H;
    f64* VL = ws->VL;
    f64* VR = ws->VR;
    f64* LRE = ws->LRE;
    f64* WR = ws->WR;
    f64* WI = ws->WI;
    f64* WR1 = ws->WR1;
    f64* WI1 = ws->WI1;
    f64* rcondv = ws->rcondv;
    f64* rcndv1 = ws->rcndv1;
    f64* rcdvin = ws->rcdvin;
    f64* rconde = ws->rconde;
    f64* rcnde1 = ws->rcnde1;
    f64* rcdein = ws->rcdein;
    f64* scale_ = ws->scale;
    f64* scale1 = ws->scale1;
    f64* work = ws->work;
    INT* iwork = ws->iwork;
    f64* result = ws->result;

    const dvx_precomputed_t* pc = &DVX_PRECOMPUTED[idx];
    INT n = pc->n;

    for (INT j = 0; j < 11; j++) {
        result[j] = -1.0;
    }

    /* Copy precomputed matrix into workspace A with proper leading dimension */
    dlaset("F", lda, n, 0.0, 0.0, A, lda);
    for (INT col = 0; col < n; col++) {
        for (INT row = 0; row < n; row++) {
            A[row + col * lda] = pc->A[row + col * n];
        }
    }

    /* Copy precomputed eigenvalues and condition numbers */
    for (INT i = 0; i < n; i++) {
        WR1[i] = pc->wr[i];
        WI1[i] = pc->wi[i];
        rcdein[i] = pc->rcdein[i];
        rcdvin[i] = pc->rcdvin[i];
    }

    INT info = 0;
    INT nnwork = 6 * n + 2 * n * n;
    if (nnwork < 1) nnwork = 1;

    /* Call dget23 with COMP=1 (ddrvvx.f lines 923-927) */
    dget23(1, "N", 22, THRESH, n,
           A, lda, H, WR, WI, WR1, WI1,
           VL, ldvl, VR, ldvr, LRE, ldlre,
           rcondv, rcndv1, rcdvin, rconde, rcnde1, rcdein,
           scale_, scale1, result, work, nnwork, iwork, &info);

    /* Check for RESULT(j) > THRESH (ddrvvx.f lines 931-958) */
    INT any_fail = 0;
    for (INT j = 0; j < 11; j++) {
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
static void test_ddrvvx_random_case(void** state)
{
    ddrvvx_params_t* params = *state;
    run_ddrvvx_random(params);
}

static void test_ddrvvx_precomp_case(void** state)
{
    ddrvvx_params_t* params = *state;
    run_ddrvvx_precomp(params);
}

/*
 * Generate all parameter combinations.
 * Random: NNVAL * MAXTYP * 3 = 7 * 21 * 3 = 441 tests
 * Precomputed: DVX_NUM_PRECOMPUTED = 39 tests
 * Total: 480 tests
 */

#define MAX_RANDOM_TESTS (NNVAL * MAXTYP * 3)
#define MAX_TESTS (MAX_RANDOM_TESTS + DVX_NUM_PRECOMPUTED)

static ddrvvx_params_t g_params[MAX_TESTS];
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
            for (INT iwk = 1; iwk <= 3; iwk++) {
                ddrvvx_params_t* p = &g_params[g_num_tests];
                p->n = n;
                p->jtype = jtype;
                p->iwk = iwk;
                p->precomp_idx = -1;
                snprintf(p->name, sizeof(p->name),
                         "ddrvvx_n%d_type%d_wk%d", n, jtype, iwk);

                g_tests[g_num_tests].name = p->name;
                g_tests[g_num_tests].test_func = test_ddrvvx_random_case;
                g_tests[g_num_tests].setup_func = NULL;
                g_tests[g_num_tests].teardown_func = NULL;
                g_tests[g_num_tests].initial_state = p;

                g_num_tests++;
            }
        }
    }

    /* Precomputed tests */
    for (INT idx = 0; idx < DVX_NUM_PRECOMPUTED; idx++) {
        ddrvvx_params_t* p = &g_params[g_num_tests];
        p->n = DVX_PRECOMPUTED[idx].n;
        p->jtype = 22;
        p->iwk = 3;
        p->precomp_idx = idx;
        snprintf(p->name, sizeof(p->name),
                 "ddrvvx_precomp_%d_n%d", idx + 1, p->n);

        g_tests[g_num_tests].name = p->name;
        g_tests[g_num_tests].test_func = test_ddrvvx_precomp_case;
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

    return _cmocka_run_group_tests("ddrvvx", g_tests, g_num_tests,
                                   group_setup, group_teardown);
}
