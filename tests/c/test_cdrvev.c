/**
 * @file test_cdrvev.c
 * @brief Non-symmetric eigenvalue test driver - port of LAPACK TESTING/EIG/zdrvev.f
 *
 * Tests the nonsymmetric eigenvalue problem driver CGEEV.
 *
 * Each (n, jtype, iwk) combination is registered as a separate CMocka test,
 * providing pytest-like parameterized test behavior with clear failure isolation.
 *
 * Test ratios (7 total):
 *   (1)  | A * VR - VR * W | / ( n |A| ulp )
 *   (2)  | A' * VL - VL * W' | / ( n |A| ulp )
 *   (3)  | |VR(i)| - 1 | / ulp and whether largest component real
 *   (4)  | |VL(i)| - 1 | / ulp and whether largest component real
 *   (5)  W(full) = W(partial)  (eigenvalues same whether VR/VL computed or not)
 *   (6)  VR(full) = VR(partial)
 *   (7)  VL(full) = VL(partial)
 *
 * Matrix types (21 total):
 *   Types 1-3:   Zero, Identity, Jordan block
 *   Types 4-8:   Diagonal with scaled eigenvalues (via CLATMS)
 *   Types 9-12:  Dense with controlled eigenvalues (via CLATME, CONDS=1)
 *   Types 13-18: Dense with ill-conditioned eigenvectors (via CLATME, CONDS=sqrt(ulp))
 *   Types 19-21: General matrices with random eigenvalues (via CLATMR)
 */

#include "test_harness.h"
#include "verify.h"
#include "test_rng.h"
#include "semicolon_cblas.h"
#include <math.h>
#include <string.h>
#include <complex.h>

#define THRESH 30.0f
#define MAXTYP 21

static const INT NVAL[] = {0, 1, 2, 3, 5, 10, 20};
#define NNVAL (sizeof(NVAL) / sizeof(NVAL[0]))

typedef struct {
    INT n;
    INT jtype;
    INT iwk;
    char name[96];
} zdrvev_params_t;

typedef struct {
    INT nmax;

    c64* A;
    c64* H;
    c64* VL;
    c64* VR;
    c64* LRE;

    c64* W;
    c64* W1;

    c64* work;
    f32* rwork;
    INT* iwork;
    INT lwork;

    f32 result[7];

    uint64_t rng_state[4];
} zdrvev_workspace_t;

static zdrvev_workspace_t* g_ws = NULL;

static const INT KTYPE[MAXTYP]  = {1, 2, 3, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 9, 9, 9};
static const INT KMAGN[MAXTYP]  = {1, 1, 1, 1, 1, 1, 2, 3, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 1, 2, 3};
static const INT KMODE[MAXTYP]  = {0, 0, 0, 4, 3, 1, 4, 4, 4, 3, 1, 5, 4, 3, 1, 5, 5, 5, 4, 3, 1};
static const INT KCONDS[MAXTYP] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 0, 0, 0};

static int group_setup(void** state)
{
    (void)state;

    g_ws = malloc(sizeof(zdrvev_workspace_t));
    if (!g_ws) return -1;

    g_ws->nmax = 0;
    for (size_t i = 0; i < NNVAL; i++) {
        if (NVAL[i] > g_ws->nmax) g_ws->nmax = NVAL[i];
    }
    if (g_ws->nmax < 1) g_ws->nmax = 1;

    INT nmax = g_ws->nmax;
    INT n2 = nmax * nmax;

    g_ws->A   = malloc(n2 * sizeof(c64));
    g_ws->H   = malloc(n2 * sizeof(c64));
    g_ws->VL  = malloc(n2 * sizeof(c64));
    g_ws->VR  = malloc(n2 * sizeof(c64));
    g_ws->LRE = malloc(n2 * sizeof(c64));
    g_ws->W   = malloc(nmax * sizeof(c64));
    g_ws->W1  = malloc(nmax * sizeof(c64));

    g_ws->lwork = 5 * nmax + 2 * n2;
    g_ws->work  = malloc(g_ws->lwork * sizeof(c64));
    g_ws->rwork = malloc(2 * nmax * sizeof(f32));
    g_ws->iwork = malloc(nmax * sizeof(INT));

    if (!g_ws->A || !g_ws->H || !g_ws->VL || !g_ws->VR || !g_ws->LRE ||
        !g_ws->W || !g_ws->W1 ||
        !g_ws->work || !g_ws->rwork || !g_ws->iwork) {
        return -1;
    }

    rng_seed(g_ws->rng_state, 0xDEADBEEFULL);
    return 0;
}

static int group_teardown(void** state)
{
    (void)state;

    if (g_ws) {
        free(g_ws->A);
        free(g_ws->H);
        free(g_ws->VL);
        free(g_ws->VR);
        free(g_ws->LRE);
        free(g_ws->W);
        free(g_ws->W1);
        free(g_ws->work);
        free(g_ws->rwork);
        free(g_ws->iwork);
        free(g_ws);
        g_ws = NULL;
    }
    return 0;
}

/**
 * Generate test matrix according to jtype.
 *
 * Based on zdrvev.f lines 574-707.
 */
static INT generate_matrix(INT n, INT jtype, c64* A, INT lda,
                           c64* work, f32* rwork, INT* iwork,
                           uint64_t state[static 4])
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

    c64 czero = CMPLXF(0.0f, 0.0f);
    c64 cone = CMPLXF(1.0f, 0.0f);

    switch (KMAGN[jtype - 1]) {
        case 1: anorm = 1.0f; break;
        case 2: anorm = ovfl * ulp; break;
        case 3: anorm = unfl * ulpinv; break;
        default: anorm = 1.0f;
    }

    claset("F", n, n, czero, czero, A, lda);
    cond = ulpinv;

    if (itype == 1) {
        iinfo = 0;

    } else if (itype == 2) {
        for (INT j = 0; j < n; j++) {
            A[j + j * lda] = CMPLXF(anorm, 0.0f);
        }

    } else if (itype == 3) {
        for (INT j = 0; j < n; j++) {
            A[j + j * lda] = CMPLXF(anorm, 0.0f);
            if (j > 0) {
                A[j + (j - 1) * lda] = cone;
            }
        }

    } else if (itype == 4) {
        clatms(n, n, "S", "H", rwork, imode, cond, anorm,
               0, 0, "N", A, lda, work + n, &iinfo, state);

    } else if (itype == 5) {
        clatms(n, n, "S", "H", rwork, imode, cond, anorm,
               n, n, "N", A, lda, work + n, &iinfo, state);

    } else if (itype == 6) {
        if (KCONDS[jtype - 1] == 1) {
            conds = 1.0f;
        } else if (KCONDS[jtype - 1] == 2) {
            conds = rtulpi;
        } else {
            conds = 0.0f;
        }

        clatme(n, "D", work, imode, cond, cone,
               "T", "T", "T", rwork, 4, conds,
               n, n, anorm, A, lda, work + 2 * n, &iinfo, state);

    } else if (itype == 7) {
        INT idumma[1] = {0};

        clatmr(n, n, "D", "N", work, 6, 1.0f, cone,
               "T", "N", work + n, 1, 1.0f,
               work + 2 * n, 1, 1.0f, "N", idumma, 0, 0,
               0.0f, anorm, "NO", A, lda, iwork, &iinfo, state);

    } else if (itype == 8) {
        INT idumma[1] = {0};

        clatmr(n, n, "D", "H", work, 6, 1.0f, cone,
               "T", "N", work + n, 1, 1.0f,
               work + 2 * n, 1, 1.0f, "N", idumma, n, n,
               0.0f, anorm, "NO", A, lda, iwork, &iinfo, state);

    } else if (itype == 9) {
        INT idumma[1] = {0};

        clatmr(n, n, "D", "N", work, 6, 1.0f, cone,
               "T", "N", work + n, 1, 1.0f,
               work + 2 * n, 1, 1.0f, "N", idumma, n, n,
               0.0f, anorm, "NO", A, lda, iwork, &iinfo, state);

        if (n >= 4) {
            c64 czero_ = CMPLXF(0.0f, 0.0f);
            claset("F", 2, n, czero_, czero_, A, lda);
            claset("F", n - 3, 1, czero_, czero_, A + 2, lda);
            claset("F", n - 3, 2, czero_, czero_, A + 2 + (n - 2) * lda, lda);
            claset("F", 1, n, czero_, czero_, A + n - 1, lda);
        }

    } else if (itype == 10) {
        INT idumma[1] = {0};

        clatmr(n, n, "D", "N", work, 6, 1.0f, cone,
               "T", "N", work + n, 1, 1.0f,
               work + 2 * n, 1, 1.0f, "N", idumma, n, 0,
               0.0f, anorm, "NO", A, lda, iwork, &iinfo, state);

    } else {
        iinfo = 1;
    }

    return iinfo;
}


/**
 * Run tests for a single (n, jtype, iwk) combination.
 *
 * Based on zdrvev.f lines 720-870.
 */
static void run_zdrvev_single(zdrvev_params_t* params)
{
    INT n = params->n;
    INT jtype = params->jtype;
    INT iwk = params->iwk;

    zdrvev_workspace_t* ws = g_ws;
    INT lda = ws->nmax;
    INT ldvl = ws->nmax;
    INT ldvr = ws->nmax;
    INT ldlre = ws->nmax;

    c64* A = ws->A;
    c64* H = ws->H;
    c64* VL = ws->VL;
    c64* VR = ws->VR;
    c64* LRE = ws->LRE;
    c64* W = ws->W;
    c64* W1 = ws->W1;
    c64* work = ws->work;
    f32* rwork = ws->rwork;

    f32 ulp = slamch("P");
    f32 ulpinv = 1.0f / ulp;

    for (INT j = 0; j < 7; j++) {
        ws->result[j] = -1.0f;
    }

    if (n == 0) {
        return;
    }

    INT iinfo = generate_matrix(n, jtype, A, lda, work, rwork,
                                ws->iwork, ws->rng_state);
    if (iinfo != 0) {
        ws->result[0] = ulpinv;
        print_message("Matrix generation failed for jtype=%d, n=%d, iinfo=%d\n",
                      jtype, n, iinfo);
        assert_info_success(iinfo);
        return;
    }

    INT nnwork;
    if (iwk == 1) {
        nnwork = 2 * n;
    } else {
        nnwork = 5 * n + 2 * n * n;
    }
    if (nnwork < 1) nnwork = 1;

    /* ========== Test CGEEV with both VL and VR ========== */
    clacpy("F", n, n, A, lda, H, lda);

    cgeev("V", "V", n, H, lda, W, VL, ldvl, VR, ldvr,
          work, nnwork, rwork, &iinfo);

    if (iinfo != 0) {
        ws->result[0] = ulpinv;
        print_message("CGEEV(V,V) failed with info=%d\n", iinfo);
        assert_info_success(iinfo);
        return;
    }

    /* Test 1: | A * VR - VR * W | / ( n |A| ulp ) */
    f32 res[2];
    cget22("N", "N", "N", n, A, lda, VR, ldvr, W, work, rwork, res);
    ws->result[0] = res[0];

    /* Test 2: | A' * VL - VL * W' | / ( n |A| ulp ) */
    cget22("C", "N", "C", n, A, lda, VL, ldvl, W, work, rwork, res);
    ws->result[1] = res[0];

    /* Test 3: | |VR(i)| - 1 | / ulp and largest component real */
    ws->result[2] = 0.0f;
    for (INT j = 0; j < n; j++) {
        f32 tnrm = cblas_scnrm2(n, VR + j * ldvr, 1);
        f32 diff = fabsf(tnrm - 1.0f) / ulp;
        if (diff < ulpinv) {
            if (diff > ws->result[2]) ws->result[2] = diff;
        } else {
            ws->result[2] = ulpinv;
        }

        f32 vmx = 0.0f;
        f32 vrmx = 0.0f;
        for (INT jj = 0; jj < n; jj++) {
            f32 vtst = cabsf(VR[jj + j * ldvr]);
            if (vtst > vmx) vmx = vtst;
            if (cimagf(VR[jj + j * ldvr]) == 0.0f &&
                fabsf(crealf(VR[jj + j * ldvr])) > vrmx) {
                vrmx = fabsf(crealf(VR[jj + j * ldvr]));
            }
        }
        if (vrmx / vmx < 1.0f - 2.0f * ulp) {
            ws->result[2] = ulpinv;
        }
    }

    /* Test 4: | |VL(i)| - 1 | / ulp and largest component real */
    ws->result[3] = 0.0f;
    for (INT j = 0; j < n; j++) {
        f32 tnrm = cblas_scnrm2(n, VL + j * ldvl, 1);
        f32 diff = fabsf(tnrm - 1.0f) / ulp;
        if (diff < ulpinv) {
            if (diff > ws->result[3]) ws->result[3] = diff;
        } else {
            ws->result[3] = ulpinv;
        }

        f32 vmx = 0.0f;
        f32 vrmx = 0.0f;
        for (INT jj = 0; jj < n; jj++) {
            f32 vtst = cabsf(VL[jj + j * ldvl]);
            if (vtst > vmx) vmx = vtst;
            if (cimagf(VL[jj + j * ldvl]) == 0.0f &&
                fabsf(crealf(VL[jj + j * ldvl])) > vrmx) {
                vrmx = fabsf(crealf(VL[jj + j * ldvl]));
            }
        }
        if (vrmx / vmx < 1.0f - 2.0f * ulp) {
            ws->result[3] = ulpinv;
        }
    }

    assert_residual_ok(ws->result[0]);
    assert_residual_ok(ws->result[1]);
    assert_residual_ok(ws->result[2]);
    assert_residual_ok(ws->result[3]);

    /* ========== Test CGEEV with eigenvalues only ========== */
    clacpy("F", n, n, A, lda, H, lda);

    c64 dum[1];
    cgeev("N", "N", n, H, lda, W1, dum, 1, dum, 1,
          work, nnwork, rwork, &iinfo);

    if (iinfo != 0) {
        ws->result[4] = ulpinv;
        print_message("CGEEV(N,N) failed with info=%d\n", iinfo);
    } else {
        /* Test 5: W(full) = W(partial) */
        ws->result[4] = 0.0f;
        for (INT j = 0; j < n; j++) {
            if (W[j] != W1[j]) {
                ws->result[4] = ulpinv;
            }
        }
    }

    /* ========== Test CGEEV with right eigenvectors only ========== */
    clacpy("F", n, n, A, lda, H, lda);

    cgeev("N", "V", n, H, lda, W1, dum, 1, LRE, ldlre,
          work, nnwork, rwork, &iinfo);

    if (iinfo != 0) {
        ws->result[5] = ulpinv;
        print_message("CGEEV(N,V) failed with info=%d\n", iinfo);
    } else {
        /* Test 5 again */
        for (INT j = 0; j < n; j++) {
            if (W[j] != W1[j]) {
                ws->result[4] = ulpinv;
            }
        }

        /* Test 6: VR(full) = VR(partial) */
        ws->result[5] = 0.0f;
        for (INT j = 0; j < n; j++) {
            for (INT jj = 0; jj < n; jj++) {
                if (VR[jj + j * ldvr] != LRE[jj + j * ldlre]) {
                    ws->result[5] = ulpinv;
                }
            }
        }
    }

    /* ========== Test CGEEV with left eigenvectors only ========== */
    clacpy("F", n, n, A, lda, H, lda);

    cgeev("V", "N", n, H, lda, W1, LRE, ldlre, dum, 1,
          work, nnwork, rwork, &iinfo);

    if (iinfo != 0) {
        ws->result[6] = ulpinv;
        print_message("CGEEV(V,N) failed with info=%d\n", iinfo);
    } else {
        /* Test 5 again */
        for (INT j = 0; j < n; j++) {
            if (W[j] != W1[j]) {
                ws->result[4] = ulpinv;
            }
        }

        /* Test 7: VL(full) = VL(partial) */
        ws->result[6] = 0.0f;
        for (INT j = 0; j < n; j++) {
            for (INT jj = 0; jj < n; jj++) {
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

static void test_zdrvev_case(void** state)
{
    zdrvev_params_t* params = *state;
    run_zdrvev_single(params);
}

#define MAX_TESTS (NNVAL * MAXTYP * 2)

static zdrvev_params_t g_params[MAX_TESTS];
static struct CMUnitTest g_tests[MAX_TESTS];
static INT g_num_tests = 0;

static void build_test_array(void)
{
    g_num_tests = 0;

    for (size_t in = 0; in < NNVAL; in++) {
        INT n = NVAL[in];

        for (INT jtype = 1; jtype <= MAXTYP; jtype++) {
            for (INT iwk = 1; iwk <= 2; iwk++) {
                zdrvev_params_t* p = &g_params[g_num_tests];
                p->n = n;
                p->jtype = jtype;
                p->iwk = iwk;
                snprintf(p->name, sizeof(p->name),
                         "zdrvev_n%d_type%d_wk%d", n, jtype, iwk);

                g_tests[g_num_tests].name = p->name;
                g_tests[g_num_tests].test_func = test_zdrvev_case;
                g_tests[g_num_tests].setup_func = NULL;
                g_tests[g_num_tests].teardown_func = NULL;
                g_tests[g_num_tests].initial_state = p;

                g_num_tests++;
            }
        }
    }
}

int main(void)
{
    build_test_array();

    return _cmocka_run_group_tests("zdrvev", g_tests, g_num_tests,
                                   group_setup, group_teardown);
}
