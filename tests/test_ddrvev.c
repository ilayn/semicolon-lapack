/**
 * @file test_ddrvev.c
 * @brief Non-symmetric eigenvalue test driver - port of LAPACK TESTING/EIG/ddrvev.f
 *
 * Tests the nonsymmetric eigenvalue problem driver DGEEV.
 *
 * Each (n, jtype, iwk) combination is registered as a separate CMocka test,
 * providing pytest-like parameterized test behavior with clear failure isolation.
 *
 * Test ratios (7 total):
 *   (1)  | A * VR - VR * W | / ( n |A| ulp )
 *   (2)  | A^T * VL - VL * W^H | / ( n |A| ulp )
 *   (3)  | |VR(i)| - 1 | / ulp and whether largest component real
 *   (4)  | |VL(i)| - 1 | / ulp and whether largest component real
 *   (5)  W(full) = W(partial)  (eigenvalues same whether VR/VL computed or not)
 *   (6)  VR(full) = VR(partial)
 *   (7)  VL(full) = VL(partial)
 *
 * Matrix types (21 total):
 *   Types 1-3:   Zero, Identity, Jordan block
 *   Types 4-8:   Diagonal with scaled eigenvalues (via DLATMS)
 *   Types 9-12:  Dense with controlled eigenvalues (via DLATME, CONDS=1)
 *   Types 13-18: Dense with ill-conditioned eigenvectors (via DLATME, CONDS=sqrt(ulp))
 *   Types 19-21: General matrices with random eigenvalues (via DLATMR)
 */

#include "test_harness.h"
#include "testutils/verify.h"
#include "testutils/test_rng.h"
#include <cblas.h>
#include <math.h>
#include <string.h>

/* Test threshold from LAPACK ddrvev.f */
#define THRESH 30.0

/* Maximum matrix type to test */
#define MAXTYP 21

/* Test dimensions from nep.in */
static const int NVAL[] = {0, 1, 2, 3, 5, 10, 20};
#define NNVAL (sizeof(NVAL) / sizeof(NVAL[0]))

/* External function declarations */
extern void dgeev(const char* jobvl, const char* jobvr, const int n,
                  f64* A, const int lda, f64* wr, f64* wi,
                  f64* VL, const int ldvl, f64* VR, const int ldvr,
                  f64* work, const int lwork, int* info);

extern f64 dlamch(const char* cmach);
extern f64 dlange(const char* norm, const int m, const int n,
                     const f64* A, const int lda, f64* work);
extern void dlacpy(const char* uplo, const int m, const int n,
                   const f64* A, const int lda, f64* B, const int ldb);
extern void dlaset(const char* uplo, const int m, const int n,
                   const f64 alpha, const f64 beta, f64* A, const int lda);

/* Test parameters for a single test case */
typedef struct {
    int n;
    int jtype;    /* Matrix type (1-21) */
    int iwk;      /* Workspace variant (1=minimal, 2=generous) */
    char name[96];
} ddrvev_params_t;

/* Workspace structure for all tests */
typedef struct {
    int nmax;

    /* Matrices (all nmax x nmax) */
    f64* A;      /* Original matrix */
    f64* H;      /* Copy modified by DGEEV */
    f64* VL;     /* Left eigenvectors (full) */
    f64* VR;     /* Right eigenvectors (full) */
    f64* LRE;    /* Left/right eigenvectors (partial) */

    /* Eigenvalues */
    f64* WR;     /* Real parts (full) */
    f64* WI;     /* Imaginary parts (full) */
    f64* WR1;    /* Real parts (partial) */
    f64* WI1;    /* Imaginary parts (partial) */

    /* Work arrays */
    f64* work;
    int* iwork;
    int lwork;

    /* Test results */
    f64 result[7];

    /* RNG state */
    uint64_t rng_state[4];
} ddrvev_workspace_t;

/* Global workspace pointer */
static ddrvev_workspace_t* g_ws = NULL;

/* Matrix type parameters (from ddrvev.f DATA statements)
 * KTYPE: 1, 2, 3, 5*4, 4*6, 6*6, 3*9
 * KMAGN: 3*1, 1, 1, 1, 2, 3, 4*1, 1, 1, 1, 1, 2, 3, 1, 2, 3
 * KMODE: 3*0, 4, 3, 1, 4, 4, 4, 3, 1, 5, 4, 3, 1, 5, 5, 5, 4, 3, 1
 * KCONDS: 3*0, 5*0, 4*1, 6*2, 3*0
 */
static const int KTYPE[MAXTYP]  = {1, 2, 3, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 9, 9, 9};
static const int KMAGN[MAXTYP]  = {1, 1, 1, 1, 1, 1, 2, 3, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 1, 2, 3};
static const int KMODE[MAXTYP]  = {0, 0, 0, 4, 3, 1, 4, 4, 4, 3, 1, 5, 4, 3, 1, 5, 5, 5, 4, 3, 1};
static const int KCONDS[MAXTYP] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 0, 0, 0};

/**
 * Group setup: allocate shared workspace.
 */
static int group_setup(void** state)
{
    (void)state;

    g_ws = malloc(sizeof(ddrvev_workspace_t));
    if (!g_ws) return -1;

    /* Find maximum N */
    g_ws->nmax = 0;
    for (size_t i = 0; i < NNVAL; i++) {
        if (NVAL[i] > g_ws->nmax) g_ws->nmax = NVAL[i];
    }
    if (g_ws->nmax < 1) g_ws->nmax = 1;

    int nmax = g_ws->nmax;
    int n2 = nmax * nmax;

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

    /* Workspace: 5*N + 2*N^2 for generous allocation */
    g_ws->lwork = 5 * nmax + 2 * n2;
    g_ws->work  = malloc(g_ws->lwork * sizeof(f64));
    g_ws->iwork = malloc(nmax * sizeof(int));

    if (!g_ws->A || !g_ws->H || !g_ws->VL || !g_ws->VR || !g_ws->LRE ||
        !g_ws->WR || !g_ws->WI || !g_ws->WR1 || !g_ws->WI1 ||
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
 * Based on ddrvev.f lines 574-707.
 */
static int generate_matrix(int n, int jtype, f64* A, int lda,
                           f64* work, int* iwork, uint64_t state[static 4])
{
    int itype = KTYPE[jtype - 1];
    int imode = KMODE[jtype - 1];
    f64 anorm, cond, conds;
    int iinfo = 0;

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

    /* Initialize A to zero */
    dlaset("F", n, n, 0.0, 0.0, A, lda);
    cond = ulpinv;

    if (itype == 1) {
        /* Zero matrix */
        iinfo = 0;

    } else if (itype == 2) {
        /* Identity matrix */
        for (int j = 0; j < n; j++) {
            A[j + j * lda] = anorm;
        }

    } else if (itype == 3) {
        /* Jordan block */
        for (int j = 0; j < n; j++) {
            A[j + j * lda] = anorm;
            if (j > 0) {
                A[j + (j - 1) * lda] = 1.0;
            }
        }

    } else if (itype == 4) {
        /* Diagonal matrix, eigenvalues specified via DLATMS */
        dlatms(n, n, "S", "S", work, imode, cond, anorm,
               0, 0, "N", A, lda, work + n, &iinfo, state);

    } else if (itype == 5) {
        /* Symmetric, eigenvalues specified via DLATMS */
        dlatms(n, n, "S", "S", work, imode, cond, anorm,
               n, n, "N", A, lda, work + n, &iinfo, state);

    } else if (itype == 6) {
        /* General, eigenvalues specified via DLATME */

        /* Determine CONDS based on KCONDS */
        if (KCONDS[jtype - 1] == 1) {
            conds = 1.0;
        } else if (KCONDS[jtype - 1] == 2) {
            conds = rtulpi;
        } else {
            conds = 0.0;
        }

        char ei[2] = " ";  /* No complex conjugate pairs specified */
        dlatme(n, "S", work, imode, cond, 1.0,
               ei, "T", "T", "T", work + n, 4, conds,
               n, n, anorm, A, lda, work + 2 * n, &iinfo, state);

    } else if (itype == 9) {
        /* General, random eigenvalues via DLATMR */
        int idumma[1] = {1};  /* Dummy pivot array */

        dlatmr(n, n, "S", "N", work, 6, 1.0, 1.0, "T", "N",
               work + n, 1, 1.0, work + 2 * n, 1, 1.0,
               "N", idumma, n, n, 0.0, anorm, "N",
               A, lda, iwork, &iinfo, state);

        /* For n >= 4, zero out some regions to create sparse structure */
        if (n >= 4) {
            /* Zero first 2 rows */
            dlaset("F", 2, n, 0.0, 0.0, A, lda);
            /* Zero first column except first 2 rows and last row */
            dlaset("F", n - 3, 1, 0.0, 0.0, A + 2, lda);
            /* Zero last 2 columns except first 2 rows and last row */
            dlaset("F", n - 3, 2, 0.0, 0.0, A + 2 + (n - 2) * lda, lda);
            /* Zero last row */
            dlaset("F", 1, n, 0.0, 0.0, A + n - 1, lda);
        }

    } else {
        iinfo = 1;
    }

    return iinfo;
}

/**
 * Compute 2-norm using LAPACK's dlapy2 pattern.
 */
static f64 dlapy2(f64 x, f64 y)
{
    f64 xabs = fabs(x);
    f64 yabs = fabs(y);
    f64 w = (xabs > yabs) ? xabs : yabs;
    f64 z = (xabs < yabs) ? xabs : yabs;

    if (z == 0.0) {
        return w;
    }
    f64 temp = z / w;
    return w * sqrt(1.0 + temp * temp);
}

/**
 * Run tests for a single (n, jtype, iwk) combination.
 *
 * Based on ddrvev.f lines 720-930.
 */
static void run_ddrvev_single(ddrvev_params_t* params)
{
    int n = params->n;
    int jtype = params->jtype;
    int iwk = params->iwk;

    ddrvev_workspace_t* ws = g_ws;
    int lda = ws->nmax;
    int ldvl = ws->nmax;
    int ldvr = ws->nmax;
    int ldlre = ws->nmax;

    f64* A = ws->A;
    f64* H = ws->H;
    f64* VL = ws->VL;
    f64* VR = ws->VR;
    f64* LRE = ws->LRE;
    f64* WR = ws->WR;
    f64* WI = ws->WI;
    f64* WR1 = ws->WR1;
    f64* WI1 = ws->WI1;
    f64* work = ws->work;

    f64 ulp = dlamch("P");
    f64 ulpinv = 1.0 / ulp;

    /* Initialize results to -1 (not computed) */
    for (int j = 0; j < 7; j++) {
        ws->result[j] = -1.0;
    }

    /* Skip N=0 cases */
    if (n == 0) {
        return;
    }

    /* Generate matrix */
    int iinfo = generate_matrix(n, jtype, A, lda, work, ws->iwork, ws->rng_state);
    if (iinfo != 0) {
        /* Matrix generation failed */
        ws->result[0] = ulpinv;
        print_message("Matrix generation failed for jtype=%d, n=%d, iinfo=%d\n",
                      jtype, n, iinfo);
        assert_info_success(iinfo);
        return;
    }

    /* Determine workspace size */
    int nnwork;
    if (iwk == 1) {
        nnwork = 4 * n;
    } else {
        nnwork = 5 * n + 2 * n * n;
    }
    if (nnwork < 1) nnwork = 1;

    /* ========== Test DGEEV with both VL and VR ========== */
    dlacpy("F", n, n, A, lda, H, lda);

    dgeev("V", "V", n, H, lda, WR, WI, VL, ldvl, VR, ldvr,
          work, nnwork, &iinfo);

    if (iinfo != 0) {
        ws->result[0] = ulpinv;
        print_message("DGEEV(V,V) failed with info=%d\n", iinfo);
        assert_info_success(iinfo);
        return;
    }

    /* Test 1: | A * VR - VR * W | / ( n |A| ulp ) */
    f64 res[2];
    dget22("N", "N", "N", n, A, lda, VR, ldvr, WR, WI, work, res);
    ws->result[0] = res[0];

    /* Test 2: | A^T * VL - VL * W^H | / ( n |A| ulp ) */
    dget22("T", "N", "T", n, A, lda, VL, ldvl, WR, WI, work, res);
    ws->result[1] = res[0];

    /* Test 3: | |VR(i)| - 1 | / ulp and largest component real */
    ws->result[2] = 0.0;
    for (int j = 0; j < n; j++) {
        f64 tnrm = 1.0;
        if (WI[j] == 0.0) {
            /* Real eigenvalue - single column */
            tnrm = cblas_dnrm2(n, VR + j * ldvr, 1);
        } else if (WI[j] > 0.0) {
            /* Complex conjugate pair - two columns */
            f64 nr1 = cblas_dnrm2(n, VR + j * ldvr, 1);
            f64 nr2 = cblas_dnrm2(n, VR + (j + 1) * ldvr, 1);
            tnrm = dlapy2(nr1, nr2);
        }
        f64 diff = fabs(tnrm - 1.0) / ulp;
        if (diff < ulpinv) {
            if (diff > ws->result[2]) ws->result[2] = diff;
        } else {
            ws->result[2] = ulpinv;
        }

        /* Check that largest component is real for complex pairs */
        if (WI[j] > 0.0) {
            f64 vmx = 0.0;
            f64 vrmx = 0.0;
            for (int jj = 0; jj < n; jj++) {
                f64 vtst = dlapy2(VR[jj + j * ldvr], VR[jj + (j + 1) * ldvr]);
                if (vtst > vmx) vmx = vtst;
                if (VR[jj + (j + 1) * ldvr] == 0.0 && fabs(VR[jj + j * ldvr]) > vrmx) {
                    vrmx = fabs(VR[jj + j * ldvr]);
                }
            }
            if (vrmx / vmx < 1.0 - 2.0 * ulp) {
                ws->result[2] = ulpinv;
            }
        }
    }

    /* Test 4: | |VL(i)| - 1 | / ulp and largest component real */
    ws->result[3] = 0.0;
    for (int j = 0; j < n; j++) {
        f64 tnrm = 1.0;
        if (WI[j] == 0.0) {
            tnrm = cblas_dnrm2(n, VL + j * ldvl, 1);
        } else if (WI[j] > 0.0) {
            f64 nr1 = cblas_dnrm2(n, VL + j * ldvl, 1);
            f64 nr2 = cblas_dnrm2(n, VL + (j + 1) * ldvl, 1);
            tnrm = dlapy2(nr1, nr2);
        }
        f64 diff = fabs(tnrm - 1.0) / ulp;
        if (diff < ulpinv) {
            if (diff > ws->result[3]) ws->result[3] = diff;
        } else {
            ws->result[3] = ulpinv;
        }

        if (WI[j] > 0.0) {
            f64 vmx = 0.0;
            f64 vrmx = 0.0;
            for (int jj = 0; jj < n; jj++) {
                f64 vtst = dlapy2(VL[jj + j * ldvl], VL[jj + (j + 1) * ldvl]);
                if (vtst > vmx) vmx = vtst;
                if (VL[jj + (j + 1) * ldvl] == 0.0 && fabs(VL[jj + j * ldvl]) > vrmx) {
                    vrmx = fabs(VL[jj + j * ldvl]);
                }
            }
            if (vrmx / vmx < 1.0 - 2.0 * ulp) {
                ws->result[3] = ulpinv;
            }
        }
    }

    assert_residual_ok(ws->result[0]);
    assert_residual_ok(ws->result[1]);
    assert_residual_ok(ws->result[2]);
    assert_residual_ok(ws->result[3]);

    /* ========== Test DGEEV with eigenvalues only ========== */
    dlacpy("F", n, n, A, lda, H, lda);

    f64 dum[1];
    dgeev("N", "N", n, H, lda, WR1, WI1, dum, 1, dum, 1,
          work, nnwork, &iinfo);

    if (iinfo != 0) {
        ws->result[4] = ulpinv;
        print_message("DGEEV(N,N) failed with info=%d\n", iinfo);
    } else {
        /* Test 5: W(full) = W(partial) */
        ws->result[4] = 0.0;
        for (int j = 0; j < n; j++) {
            if (WR[j] != WR1[j] || WI[j] != WI1[j]) {
                ws->result[4] = ulpinv;
            }
        }
    }

    /* ========== Test DGEEV with right eigenvectors only ========== */
    dlacpy("F", n, n, A, lda, H, lda);

    dgeev("N", "V", n, H, lda, WR1, WI1, dum, 1, LRE, ldlre,
          work, nnwork, &iinfo);

    if (iinfo != 0) {
        ws->result[5] = ulpinv;
        print_message("DGEEV(N,V) failed with info=%d\n", iinfo);
    } else {
        /* Test 5 again */
        for (int j = 0; j < n; j++) {
            if (WR[j] != WR1[j] || WI[j] != WI1[j]) {
                ws->result[4] = ulpinv;
            }
        }

        /* Test 6: VR(full) = VR(partial) */
        ws->result[5] = 0.0;
        for (int j = 0; j < n; j++) {
            for (int jj = 0; jj < n; jj++) {
                if (VR[jj + j * ldvr] != LRE[jj + j * ldlre]) {
                    ws->result[5] = ulpinv;
                }
            }
        }
    }

    /* ========== Test DGEEV with left eigenvectors only ========== */
    dlacpy("F", n, n, A, lda, H, lda);

    dgeev("V", "N", n, H, lda, WR1, WI1, LRE, ldlre, dum, 1,
          work, nnwork, &iinfo);

    if (iinfo != 0) {
        ws->result[6] = ulpinv;
        print_message("DGEEV(V,N) failed with info=%d\n", iinfo);
    } else {
        /* Test 5 again */
        for (int j = 0; j < n; j++) {
            if (WR[j] != WR1[j] || WI[j] != WI1[j]) {
                ws->result[4] = ulpinv;
            }
        }

        /* Test 7: VL(full) = VL(partial) */
        ws->result[6] = 0.0;
        for (int j = 0; j < n; j++) {
            for (int jj = 0; jj < n; jj++) {
                if (VL[jj + j * ldvl] != LRE[jj + j * ldlre]) {
                    ws->result[6] = ulpinv;
                }
            }
        }
    }

    assert_residual_ok(ws->result[4]);
    assert_residual_ok(ws->result[5]);
    assert_residual_ok(ws->result[6]);
}

/**
 * Test function wrapper.
 */
static void test_ddrvev_case(void** state)
{
    ddrvev_params_t* params = *state;
    run_ddrvev_single(params);
}

/*
 * Generate all parameter combinations.
 * Total: NNVAL * MAXTYP * 2 = 7 * 21 * 2 = 294 tests
 */

/* Maximum number of test cases */
#define MAX_TESTS (NNVAL * MAXTYP * 2)

static ddrvev_params_t g_params[MAX_TESTS];
static struct CMUnitTest g_tests[MAX_TESTS];
static int g_num_tests = 0;

/**
 * Build the test array with all parameter combinations.
 */
static void build_test_array(void)
{
    g_num_tests = 0;

    for (size_t in = 0; in < NNVAL; in++) {
        int n = NVAL[in];

        for (int jtype = 1; jtype <= MAXTYP; jtype++) {
            for (int iwk = 1; iwk <= 2; iwk++) {
                ddrvev_params_t* p = &g_params[g_num_tests];
                p->n = n;
                p->jtype = jtype;
                p->iwk = iwk;
                snprintf(p->name, sizeof(p->name),
                         "ddrvev_n%d_type%d_wk%d", n, jtype, iwk);

                g_tests[g_num_tests].name = p->name;
                g_tests[g_num_tests].test_func = test_ddrvev_case;
                g_tests[g_num_tests].setup_func = NULL;
                g_tests[g_num_tests].teardown_func = NULL;
                g_tests[g_num_tests].initial_state = p;

                g_num_tests++;
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
    return _cmocka_run_group_tests("ddrvev", g_tests, g_num_tests,
                                   group_setup, group_teardown);
}
