/**
 * @file test_sdrves.c
 * @brief Schur decomposition test driver - port of LAPACK TESTING/EIG/ddrves.f
 *
 * Tests the nonsymmetric eigenvalue (Schur form) problem driver SGEES.
 *
 * Each (n, jtype, iwk, isort) combination is registered as a separate CMocka test,
 * providing pytest-like parameterized test behavior with clear failure isolation.
 *
 * Test ratios (13 total):
 *   (1)  0 if T is in Schur form, 1/ulp otherwise (no sorting)
 *   (2)  | A - VS T VS' | / ( n |A| ulp )  (no sorting)
 *   (3)  | I - VS VS' | / ( n ulp )  (no sorting)
 *   (4)  0 if WR+sqrt(-1)*WI are eigenvalues of T (no sorting)
 *   (5)  0 if T(with VS) = T(without VS) (no sorting)
 *   (6)  0 if eigenvalues(with VS) = eigenvalues(without VS) (no sorting)
 *   (7-12) Same as (1-6) but with eigenvalue sorting
 *   (13) 0 if SDIM matches count of selected eigenvalues
 *
 * Matrix types (21 total):
 *   Types 1-3:   Zero, Identity, Jordan block
 *   Types 4-8:   Diagonal with scaled eigenvalues (via SLATMS)
 *   Types 9-12:  Dense with controlled eigenvalues (via SLATME, CONDS=1)
 *   Types 13-18: Dense with ill-conditioned eigenvectors (via SLATME, CONDS=sqrt(ulp))
 *   Types 19-21: General matrices with random eigenvalues (via SLATMR)
 */

#include "test_harness.h"
#include "verify.h"
#include "test_rng.h"
#include <math.h>
#include <string.h>

/* Test threshold from LAPACK ddrves.f */
#define THRESH 30.0f

/* Maximum matrix type to test */
#define MAXTYP 21

/* Test dimensions from nep.in */
static const INT NVAL[] = {0, 1, 2, 3, 5, 10, 20};
#define NNVAL (sizeof(NVAL) / sizeof(NVAL[0]))

/* Selection function type for sgees */
typedef INT (*sselect2_t)(const f32* wr, const f32* wi);

/* Common block for eigenvalue selection (mirrors SSLCT in ddrves.f) */
static INT g_selopt;       /* Selection option */
static INT g_seldim;       /* Dimension of selected eigenvalues */
static INT g_selval[20];   /* Selection indicators */
static f32 g_selwr[20]; /* Real parts of selected eigenvalues */
static f32 g_selwi[20]; /* Imaginary parts of selected eigenvalues */

/**
 * Selection function for sgees.
 * Selects eigenvalues based on global selection criteria.
 */
static INT dslect(const f32* wr, const f32* wi)
{
    INT j;
    f32 eps = slamch("P");

    if (g_selopt == 0) {
        /* Select all eigenvalues */
        return 1;
    }

    for (j = 0; j < g_seldim; j++) {
        f32 rmin = fabsf(*wr - g_selwr[j]);
        if (fabsf(*wi - g_selwi[j]) < rmin) {
            rmin = fabsf(*wi - g_selwi[j]);
        }
        if (rmin < eps) {
            return g_selval[j];
        }
    }
    return 0;
}

/* Test parameters for a single test case */
typedef struct {
    INT n;
    INT jtype;    /* Matrix type (1-21) */
    INT iwk;      /* Workspace variant (1=minimal, 2=generous) */
    INT isort;    /* Sort eigenvalues (0=no, 1=yes) */
    char name[96];
} ddrves_params_t;

/* Workspace structure for all tests */
typedef struct {
    INT nmax;

    /* Matrices (all nmax x nmax) */
    f32* A;      /* Working matrix */
    f32* H;      /* Schur form (with VS) */
    f32* HT;     /* Schur form (without VS) */
    f32* VS;     /* Schur vectors */

    /* Eigenvalues */
    f32* WR;     /* Real parts (with VS) */
    f32* WI;     /* Imaginary parts (with VS) */
    f32* WRT;    /* Real parts (without VS) */
    f32* WIT;    /* Imaginary parts (without VS) */

    /* Work arrays */
    f32* work;
    INT* iwork;
    INT* bwork;
    INT lwork;

    /* Test results */
    f32 result[13];

    /* RNG state */
    uint64_t rng_state[4];
} ddrves_workspace_t;

/* Global workspace pointer */
static ddrves_workspace_t* g_ws = NULL;

/* Matrix type parameters (from ddrves.f DATA statements)
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

    g_ws = malloc(sizeof(ddrves_workspace_t));
    if (!g_ws) return -1;

    /* Find maximum N */
    g_ws->nmax = 0;
    for (size_t i = 0; i < NNVAL; i++) {
        if (NVAL[i] > g_ws->nmax) g_ws->nmax = NVAL[i];
    }
    if (g_ws->nmax < 1) g_ws->nmax = 1;

    INT nmax = g_ws->nmax;
    INT n2 = nmax * nmax;

    /* Allocate matrices */
    g_ws->A   = malloc(n2 * sizeof(f32));
    g_ws->H   = malloc(n2 * sizeof(f32));
    g_ws->HT  = malloc(n2 * sizeof(f32));
    g_ws->VS  = malloc(n2 * sizeof(f32));
    g_ws->WR  = malloc(nmax * sizeof(f32));
    g_ws->WI  = malloc(nmax * sizeof(f32));
    g_ws->WRT = malloc(nmax * sizeof(f32));
    g_ws->WIT = malloc(nmax * sizeof(f32));

    /* Workspace: 5*N + 2*N^2 for generous allocation */
    g_ws->lwork = 5 * nmax + 2 * n2;
    g_ws->work  = malloc(g_ws->lwork * sizeof(f32));
    g_ws->iwork = malloc(nmax * sizeof(INT));
    g_ws->bwork = malloc(nmax * sizeof(INT));

    if (!g_ws->A || !g_ws->H || !g_ws->HT || !g_ws->VS ||
        !g_ws->WR || !g_ws->WI || !g_ws->WRT || !g_ws->WIT ||
        !g_ws->work || !g_ws->iwork || !g_ws->bwork) {
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
        free(g_ws->WR);
        free(g_ws->WI);
        free(g_ws->WRT);
        free(g_ws->WIT);
        free(g_ws->work);
        free(g_ws->iwork);
        free(g_ws->bwork);
        free(g_ws);
        g_ws = NULL;
    }
    return 0;
}

/**
 * Generate test matrix according to jtype.
 *
 * Based on ddrves.f lines 583-700.
 */
static INT generate_matrix(INT n, INT jtype, f32* A, INT lda,
                           f32* work, INT* iwork, uint64_t state[static 4])
{
    INT itype = KTYPE[jtype - 1];
    INT imode = KMODE[jtype - 1];
    f32 anorm, cond, conds;
    INT iinfo = 0;

    f32 ulp = slamch("P");
    f32 unfl = slamch("S");
    f32 ovfl = 1.0f / unfl;
    f32 ulpinv = 1.0f / ulp;
    f32 rtulp = sqrtf(ulp);
    f32 rtulpi = 1.0f / rtulp;

    /* Compute norm based on KMAGN */
    switch (KMAGN[jtype - 1]) {
        case 1: anorm = 1.0f; break;
        case 2: anorm = ovfl * ulp; break;
        case 3: anorm = unfl * ulpinv; break;
        default: anorm = 1.0f;
    }

    /* Initialize A to zero */
    slaset("F", n, n, 0.0f, 0.0f, A, lda);
    cond = ulpinv;

    if (itype == 1) {
        /* Zero matrix */
        iinfo = 0;

    } else if (itype == 2) {
        /* Identity matrix */
        for (INT j = 0; j < n; j++) {
            A[j + j * lda] = anorm;
        }

    } else if (itype == 3) {
        /* Jordan block */
        for (INT j = 0; j < n; j++) {
            A[j + j * lda] = anorm;
            if (j > 0) {
                A[j + (j - 1) * lda] = 1.0f;
            }
        }

    } else if (itype == 4) {
        /* Diagonal matrix, eigenvalues specified via SLATMS */
        slatms(n, n, "S", "S", work, imode, cond, anorm,
               0, 0, "N", A, lda, work + n, &iinfo, state);

    } else if (itype == 5) {
        /* Symmetric, eigenvalues specified via SLATMS */
        slatms(n, n, "S", "S", work, imode, cond, anorm,
               n, n, "N", A, lda, work + n, &iinfo, state);

    } else if (itype == 6) {
        /* General, eigenvalues specified via SLATME */

        /* Determine CONDS based on KCONDS */
        if (KCONDS[jtype - 1] == 1) {
            conds = 1.0f;
        } else if (KCONDS[jtype - 1] == 2) {
            conds = rtulpi;
        } else {
            conds = 0.0f;
        }

        char ei[2] = " ";  /* No complex conjugate pairs specified */
        slatme(n, "S", work, imode, cond, 1.0f,
               ei, "T", "T", "T", work + n, 4, conds,
               n, n, anorm, A, lda, work + 2 * n, &iinfo, state);

    } else if (itype == 9) {
        /* General, random eigenvalues via SLATMR */
        INT idumma[1] = {1};  /* Dummy pivot array */

        slatmr(n, n, "S", "N", work, 6, 1.0f, 1.0f, "T", "N",
               work + n, 1, 1.0f, work + 2 * n, 1, 1.0f,
               "N", idumma, n, n, 0.0f, anorm, "N",
               A, lda, iwork, &iinfo, state);

        /* For n >= 4, zero out some regions to create sparse structure */
        if (n >= 4) {
            /* Zero first 2 rows */
            slaset("F", 2, n, 0.0f, 0.0f, A, lda);
            /* Zero first column except first 2 rows and last row */
            slaset("F", n - 3, 1, 0.0f, 0.0f, A + 2, lda);
            /* Zero last 2 columns except first 2 rows and last row */
            slaset("F", n - 3, 2, 0.0f, 0.0f, A + 2 + (n - 2) * lda, lda);
            /* Zero last row */
            slaset("F", 1, n, 0.0f, 0.0f, A + n - 1, lda);
        }

    } else {
        iinfo = 1;
    }

    return iinfo;
}

/**
 * Run tests for a single (n, jtype, iwk, isort) combination.
 *
 * Based on ddrves.f lines 705-930.
 */
static void run_ddrves_single(ddrves_params_t* params)
{
    INT n = params->n;
    INT jtype = params->jtype;
    INT iwk = params->iwk;
    INT isort = params->isort;

    ddrves_workspace_t* ws = g_ws;
    INT lda = ws->nmax;
    INT ldvs = ws->nmax;

    f32* A = ws->A;
    f32* H = ws->H;
    f32* HT = ws->HT;
    f32* VS = ws->VS;
    f32* WR = ws->WR;
    f32* WI = ws->WI;
    f32* WRT = ws->WRT;
    f32* WIT = ws->WIT;
    f32* work = ws->work;
    INT* bwork = ws->bwork;

    f32 ulp = slamch("P");
    f32 unfl = slamch("S");
    f32 ulpinv = 1.0f / ulp;

    INT rsub = (isort == 0) ? 0 : 6;
    const char* sort = (isort == 0) ? "N" : "S";

    /* Initialize results to -1 (not computed) */
    for (INT j = 0; j < 13; j++) {
        ws->result[j] = -1.0f;
    }

    /* Skip N=0 cases */
    if (n == 0) {
        return;
    }

    /* Generate matrix */
    INT iinfo = generate_matrix(n, jtype, A, lda, work, ws->iwork, ws->rng_state);
    if (iinfo != 0) {
        /* Matrix generation failed */
        ws->result[rsub] = ulpinv;
        print_message("Matrix generation failed for jtype=%d, n=%d, iinfo=%d\n",
                      jtype, n, iinfo);
        assert_info_success(iinfo);
        return;
    }

    /* Determine workspace size */
    INT nnwork;
    if (iwk == 1) {
        nnwork = 3 * n;
    } else {
        nnwork = 5 * n + 2 * n * n;
    }
    if (nnwork < 1) nnwork = 1;

    /* Copy A to H for Schur decomposition with VS */
    slacpy("F", n, n, A, lda, H, lda);

    /* Setup selection */
    g_selopt = 1;

    /* Compute Schur form with vectors */
    INT sdim;
    sgees("V", sort, dslect, n, H, lda, &sdim, WR, WI, VS, ldvs,
          work, nnwork, bwork, &iinfo);

    if (iinfo != 0 && iinfo != n + 2) {
        ws->result[rsub] = ulpinv;
        if (iinfo > 0 && iinfo <= n) {
            /* QR algorithm failed */
            print_message("SGEES failed with info=%d (QR failed)\n", iinfo);
        }
        return;
    }

    /* Test 1 (or 7): Verify quasi-triangular structure */
    ws->result[rsub] = 0.0f;

    /* Check that elements below subdiagonal are zero */
    for (INT j = 0; j < n - 2; j++) {
        for (INT i = j + 2; i < n; i++) {
            if (H[i + j * lda] != 0.0f) {
                ws->result[rsub] = ulpinv;
            }
        }
    }

    /* Check that we don't have two consecutive 2x2 blocks */
    for (INT i = 0; i < n - 2; i++) {
        if (H[i + 1 + i * lda] != 0.0f && H[i + 2 + (i + 1) * lda] != 0.0f) {
            ws->result[rsub] = ulpinv;
        }
    }

    /* Check 2x2 blocks have proper structure */
    for (INT i = 0; i < n - 1; i++) {
        if (H[i + 1 + i * lda] != 0.0f) {
            /* 2x2 block: check structure */
            if (H[i + i * lda] != H[i + 1 + (i + 1) * lda] ||
                H[i + (i + 1) * lda] == 0.0f ||
                (H[i + 1 + i * lda] >= 0.0f) == (H[i + (i + 1) * lda] >= 0.0f)) {
                ws->result[rsub] = ulpinv;
            }
        }
    }

    assert_residual_ok(ws->result[rsub]);

    /* Tests 2-3 (or 8-9): | A - VS*T*VS' | and | I - VS*VS' | */
    INT lwork_hst = 2 * n * n > 1 ? 2 * n * n : 1;
    f32 res[2];
    shst01(n, 1, n, A, lda, H, lda, VS, ldvs, work, lwork_hst, res);
    ws->result[rsub + 1] = res[0];
    ws->result[rsub + 2] = res[1];

    assert_residual_ok(ws->result[rsub + 1]);
    assert_residual_ok(ws->result[rsub + 2]);

    /* Test 4 (or 10): Check eigenvalues match diagonal of T */
    ws->result[rsub + 3] = 0.0f;
    for (INT i = 0; i < n; i++) {
        if (H[i + i * lda] != WR[i]) {
            ws->result[rsub + 3] = ulpinv;
        }
    }

    if (n > 1) {
        if (H[1 + 0 * lda] == 0.0f && WI[0] != 0.0f) {
            ws->result[rsub + 3] = ulpinv;
        }
        if (H[n - 1 + (n - 2) * lda] == 0.0f && WI[n - 1] != 0.0f) {
            ws->result[rsub + 3] = ulpinv;
        }
    }

    for (INT i = 0; i < n - 1; i++) {
        if (H[i + 1 + i * lda] != 0.0f) {
            f32 tmp = sqrtf(fabsf(H[i + 1 + i * lda])) * sqrtf(fabsf(H[i + (i + 1) * lda]));
            f32 wival = fabsf(WI[i] - tmp) / (ulp * tmp > unfl ? ulp * tmp : unfl);
            f32 wival1 = fabsf(WI[i + 1] + tmp) / (ulp * tmp > unfl ? ulp * tmp : unfl);
            if (wival > ws->result[rsub + 3]) ws->result[rsub + 3] = wival;
            if (wival1 > ws->result[rsub + 3]) ws->result[rsub + 3] = wival1;
        } else if (i > 0) {
            if (H[i + (i - 1) * lda] == 0.0f && WI[i] != 0.0f) {
                ws->result[rsub + 3] = ulpinv;
            }
        }
    }

    assert_residual_ok(ws->result[rsub + 3]);

    /* Tests 5-6 (or 11-12): Compare with and without VS */
    /* Copy A to HT for Schur decomposition without VS */
    slacpy("F", n, n, A, lda, HT, lda);

    sgees("N", sort, dslect, n, HT, lda, &sdim, WRT, WIT, VS, ldvs,
          work, nnwork, bwork, &iinfo);

    if (iinfo != 0 && iinfo != n + 2) {
        ws->result[rsub + 4] = ulpinv;
    } else {
        /* Test 5: Compare T matrices */
        ws->result[rsub + 4] = 0.0f;
        for (INT j = 0; j < n; j++) {
            for (INT i = 0; i < n; i++) {
                if (H[i + j * lda] != HT[i + j * lda]) {
                    ws->result[rsub + 4] = ulpinv;
                }
            }
        }

        /* Test 6: Compare eigenvalues */
        ws->result[rsub + 5] = 0.0f;
        for (INT i = 0; i < n; i++) {
            if (WR[i] != WRT[i] || WI[i] != WIT[i]) {
                ws->result[rsub + 5] = ulpinv;
            }
        }
    }

    assert_residual_ok(ws->result[rsub + 4]);
    assert_residual_ok(ws->result[rsub + 5]);

    /* Test 13 (only for sorted case): SDIM validation */
    if (isort == 1) {
        /* Count expected selected eigenvalues */
        INT knteig = 0;
        for (INT i = 0; i < n; i++) {
            if (dslect(&WR[i], &WI[i])) {
                knteig++;
            }
        }
        ws->result[12] = (sdim == knteig) ? 0.0f : ulpinv;
        assert_residual_ok(ws->result[12]);
    }
}

/**
 * Test function wrapper.
 */
static void test_ddrves_case(void** state)
{
    ddrves_params_t* params = *state;
    run_ddrves_single(params);
}

/*
 * Generate all parameter combinations.
 * Total: NNVAL * MAXTYP * 2 * 2 = 7 * 18 * 2 * 2 = 504 tests
 */

/* Maximum number of test cases */
#define MAX_TESTS (NNVAL * MAXTYP * 2 * 2)

static ddrves_params_t g_params[MAX_TESTS];
static struct CMUnitTest g_tests[MAX_TESTS];
static INT g_num_tests = 0;

/**
 * Build the test array with all parameter combinations.
 */
static void build_test_array(void)
{
    g_num_tests = 0;

    for (size_t in = 0; in < NNVAL; in++) {
        INT n = NVAL[in];

        for (INT jtype = 1; jtype <= MAXTYP; jtype++) {
            for (INT iwk = 1; iwk <= 2; iwk++) {
                for (INT isort = 0; isort <= 1; isort++) {
                    ddrves_params_t* p = &g_params[g_num_tests];
                    p->n = n;
                    p->jtype = jtype;
                    p->iwk = iwk;
                    p->isort = isort;
                    snprintf(p->name, sizeof(p->name),
                             "ddrves_n%d_type%d_wk%d_sort%d", n, jtype, iwk, isort);

                    g_tests[g_num_tests].name = p->name;
                    g_tests[g_num_tests].test_func = test_ddrves_case;
                    g_tests[g_num_tests].setup_func = NULL;
                    g_tests[g_num_tests].teardown_func = NULL;
                    g_tests[g_num_tests].initial_state = p;

                    g_num_tests++;
                }
            }
        }
    }
}

/* ===== Main ===== */

int main(void)
{
    /* Build all test cases */
    build_test_array();

    /* Run all tests with shared workspace */
    return _cmocka_run_group_tests("ddrves", g_tests, g_num_tests,
                                   group_setup, group_teardown);
}
