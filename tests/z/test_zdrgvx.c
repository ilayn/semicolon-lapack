/**
 * @file test_zdrgvx.c
 * @brief Generalized eigenvalue expert driver test - port of LAPACK TESTING/EIG/zdrgvx.f
 *
 * Tests the nonsymmetric generalized eigenvalue expert driver ZGGEVX.
 *
 * Each (iptype, iwa) combination is a separate CMocka test; within each test,
 * all 125 weight combinations (iwb x iwx x iwy) are exercised.
 * Precomputed read-in tests are separate CMocka tests.
 *
 * Test ratios (4 total):
 *                           H
 *   (1)  max | ( b A - a B)  l | / const.
 *   (2)  max | ( b A - a B ) r | / const.
 *   (3)  max ( Sest/Stru, Stru/Sest ) over all eigenvalues
 *   (4)  max ( DIFest/DIFtru, DIFtru/DIFest ) over 1st and last eigenvectors
 */

#include "test_harness.h"
#include "verify.h"
#include <math.h>
#include <complex.h>

#include "zgvx_testdata.h"

#define THRESH  10.0
#define NMAX    5

/* ===================================================================== */
/* Test parameters and workspace                                         */
/* ===================================================================== */

typedef struct {
    INT iptype;
    INT iwa;
    char name[64];
} zdrgvx_params_t;

typedef struct {
    c128* A;
    c128* B;
    c128* AI;
    c128* BI;
    c128* VL;
    c128* VR;
    c128* alpha;
    c128* beta;
    f64*  lscale;
    f64*  rscale;
    f64*  s;
    f64*  dtru;
    f64*  dif;
    f64*  diftru;
    c128* work;
    f64*  rwork;
    INT*  iwork;
    INT*  bwork;
    INT   lwork;
} zdrgvx_workspace_t;

static zdrgvx_workspace_t* g_ws = NULL;

/* Weight values: {0.1, 0.5, 1.0, 2.0, 10.0} */
static const c128 WEIGHT[5] = {
    CMPLX(0.1, 0.0),
    CMPLX(0.5, 0.0),
    CMPLX(1.0, 0.0),
    CMPLX(2.0, 0.0),
    CMPLX(10.0, 0.0)
};

/* ===================================================================== */
/* Group setup / teardown                                                */
/* ===================================================================== */

static int group_setup(void** state)
{
    (void)state;

    g_ws = malloc(sizeof(zdrgvx_workspace_t));
    if (!g_ws) return -1;

    INT n2 = NMAX * NMAX;

    g_ws->A      = malloc(n2 * sizeof(c128));
    g_ws->B      = malloc(n2 * sizeof(c128));
    g_ws->AI     = malloc(n2 * sizeof(c128));
    g_ws->BI     = malloc(n2 * sizeof(c128));
    g_ws->VL     = malloc(n2 * sizeof(c128));
    g_ws->VR     = malloc(n2 * sizeof(c128));
    g_ws->alpha  = malloc(NMAX * sizeof(c128));
    g_ws->beta   = malloc(NMAX * sizeof(c128));
    g_ws->lscale = malloc(NMAX * sizeof(f64));
    g_ws->rscale = malloc(NMAX * sizeof(f64));
    g_ws->s      = malloc(NMAX * sizeof(f64));
    g_ws->dtru   = malloc(NMAX * sizeof(f64));
    g_ws->dif    = malloc(NMAX * sizeof(f64));
    g_ws->diftru = malloc(NMAX * sizeof(f64));

    g_ws->lwork = 2 * NMAX * (NMAX + 1);
    g_ws->work  = malloc(g_ws->lwork * sizeof(c128));

    INT rwork_size = 6 * NMAX;
    g_ws->rwork = malloc(rwork_size * sizeof(f64));

    g_ws->iwork = malloc((NMAX + 2) * sizeof(INT));
    g_ws->bwork = malloc(NMAX * sizeof(INT));

    if (!g_ws->A || !g_ws->B || !g_ws->AI || !g_ws->BI ||
        !g_ws->VL || !g_ws->VR || !g_ws->alpha || !g_ws->beta ||
        !g_ws->lscale || !g_ws->rscale || !g_ws->s || !g_ws->dtru ||
        !g_ws->dif || !g_ws->diftru || !g_ws->work || !g_ws->rwork ||
        !g_ws->iwork || !g_ws->bwork) {
        return -1;
    }

    return 0;
}

static int group_teardown(void** state)
{
    (void)state;

    if (g_ws) {
        free(g_ws->A);
        free(g_ws->B);
        free(g_ws->AI);
        free(g_ws->BI);
        free(g_ws->VL);
        free(g_ws->VR);
        free(g_ws->alpha);
        free(g_ws->beta);
        free(g_ws->lscale);
        free(g_ws->rscale);
        free(g_ws->s);
        free(g_ws->dtru);
        free(g_ws->dif);
        free(g_ws->diftru);
        free(g_ws->work);
        free(g_ws->rwork);
        free(g_ws->iwork);
        free(g_ws->bwork);
        free(g_ws);
        g_ws = NULL;
    }
    return 0;
}

/* ===================================================================== */
/* Built-in test function                                                */
/* Port of zdrgvx.f lines 410-554                                        */
/* Each CMocka test runs 125 (iwb x iwx x iwy) combinations.            */
/* ===================================================================== */

static void test_builtin(void** state)
{
    zdrgvx_params_t* p = *(zdrgvx_params_t**)state;
    zdrgvx_workspace_t* ws = g_ws;

    INT iptype = p->iptype;
    INT iwa    = p->iwa;
    INT n      = 5;
    INT lda    = n;

    f64 ulp    = dlamch("P");
    f64 ulpinv = 1.0 / ulp;
    f64 thrsh2 = 10.0 * THRESH;

    for (INT iwb = 0; iwb < 5; iwb++) {
        for (INT iwx = 0; iwx < 5; iwx++) {
            for (INT iwy = 0; iwy < 5; iwy++) {

                zlatm6(iptype, n, ws->A, lda, ws->B,
                       ws->VR, lda, ws->VL, lda,
                       WEIGHT[iwa], WEIGHT[iwb],
                       WEIGHT[iwx], WEIGHT[iwy],
                       ws->dtru, ws->diftru);

                zlacpy("Full", n, n, ws->A, lda, ws->AI, lda);
                zlacpy("Full", n, n, ws->B, lda, ws->BI, lda);

                INT ilo = 0, ihi = 0;
                f64 anorm = 0.0, bnorm = 0.0;
                INT linfo = 0;

                zggevx("N", "V", "V", "B", n,
                       ws->AI, lda, ws->BI, lda,
                       ws->alpha, ws->beta,
                       ws->VL, lda, ws->VR, lda,
                       &ilo, &ihi,
                       ws->lscale, ws->rscale,
                       &anorm, &bnorm,
                       ws->s, ws->dif,
                       ws->work, ws->lwork,
                       ws->rwork, ws->iwork,
                       ws->bwork, &linfo);

                if (linfo != 0) {
                    fprintf(stderr, "ZGGEVX returned INFO=%d, type=%d, "
                                  "iwa=%d, iwb=%d, iwx=%d, iwy=%d\n",
                                  linfo, iptype, iwa, iwb, iwx, iwy);
                    continue;
                }

                zlacpy("Full", n, n, ws->AI, lda, ws->work, n);
                zlacpy("Full", n, n, ws->BI, lda,
                       ws->work + n * n, n);
                f64 abnrm = zlange("Fro", n, 2 * n,
                                    ws->work, n, ws->rwork);

                f64 result[4] = {0.0, 0.0, 0.0, 0.0};
                f64 res52[2];

                res52[0] = 0.0;
                res52[1] = 0.0;
                zget52(1, n, ws->A, lda, ws->B, lda,
                       ws->VL, lda, ws->alpha, ws->beta,
                       ws->work, ws->rwork, res52);
                result[0] = res52[0];

                res52[0] = 0.0;
                res52[1] = 0.0;
                zget52(0, n, ws->A, lda, ws->B, lda,
                       ws->VR, lda, ws->alpha, ws->beta,
                       ws->work, ws->rwork, res52);
                result[1] = res52[0];

                result[2] = 0.0;
                for (INT i = 0; i < n; i++) {
                    if (ws->s[i] == 0.0) {
                        if (ws->dtru[i] > abnrm * ulp)
                            result[2] = ulpinv;
                    } else if (ws->dtru[i] == 0.0) {
                        if (ws->s[i] > abnrm * ulp)
                            result[2] = ulpinv;
                    } else {
                        f64 ratio = fmax(fabs(ws->dtru[i] / ws->s[i]),
                                         fabs(ws->s[i] / ws->dtru[i]));
                        if (ratio > result[2])
                            result[2] = ratio;
                    }
                }

                result[3] = 0.0;
                if (ws->dif[0] == 0.0) {
                    if (ws->diftru[0] > abnrm * ulp)
                        result[3] = ulpinv;
                } else if (ws->diftru[0] == 0.0) {
                    if (ws->dif[0] > abnrm * ulp)
                        result[3] = ulpinv;
                } else if (ws->dif[n - 1] == 0.0) {
                    if (ws->diftru[n - 1] > abnrm * ulp)
                        result[3] = ulpinv;
                } else if (ws->diftru[n - 1] == 0.0) {
                    if (ws->dif[n - 1] > abnrm * ulp)
                        result[3] = ulpinv;
                } else {
                    f64 ratio1 = fmax(fabs(ws->diftru[0] / ws->dif[0]),
                                      fabs(ws->dif[0] / ws->diftru[0]));
                    f64 ratio2 = fmax(
                        fabs(ws->diftru[n - 1] / ws->dif[n - 1]),
                        fabs(ws->dif[n - 1] / ws->diftru[n - 1]));
                    result[3] = fmax(ratio1, ratio2);
                }

                for (INT j = 0; j < 4; j++) {
                    f64 thr = (j >= 3) ? thrsh2 : THRESH;
                    if (result[j] >= thr) {
                        fprintf(stderr, "  ZDRGVX builtin: type=%d, iwa=%d, "
                                      "iwb=%d, iwx=%d, iwy=%d, "
                                      "test %d = %.6e\n",
                                      iptype, iwa, iwb, iwx, iwy,
                                      j + 1, result[j]);
                    }
                    assert_residual_below(result[j], thr);
                }

            }
        }
    }
}

/* ===================================================================== */
/* Precomputed (read-in) test function                                   */
/* Port of zdrgvx.f lines 558-694                                        */
/* ===================================================================== */

static void test_precomputed(void** state)
{
    INT* idx_ptr = *(INT**)state;
    INT idx = *idx_ptr;
    zdrgvx_workspace_t* ws = g_ws;

    const zgvx_precomputed_t* tc = &ZGVX_PRECOMPUTED[idx];
    INT n   = tc->n;
    INT lda = NMAX;

    f64 ulp    = dlamch("P");
    f64 ulpinv = 1.0 / ulp;
    f64 thrsh2 = 10.0 * THRESH;

    for (INT j = 0; j < n; j++)
        for (INT i = 0; i < n; i++)
            ws->AI[i + j * lda] = tc->A[i + j * n];
    for (INT j = 0; j < n; j++)
        for (INT i = 0; i < n; i++)
            ws->BI[i + j * lda] = tc->B[i + j * n];
    for (INT i = 0; i < n; i++) {
        ws->dtru[i] = tc->dtru[i];
        ws->diftru[i] = tc->diftru[i];
    }

    zlacpy("Full", n, n, ws->AI, lda, ws->A, lda);
    zlacpy("Full", n, n, ws->BI, lda, ws->B, lda);

    INT ilo = 0, ihi = 0;
    f64 anorm = 0.0, bnorm = 0.0;
    INT linfo = 0;

    zggevx("N", "V", "V", "B", n,
           ws->AI, lda, ws->BI, lda,
           ws->alpha, ws->beta,
           ws->VL, lda, ws->VR, lda,
           &ilo, &ihi,
           ws->lscale, ws->rscale,
           &anorm, &bnorm,
           ws->s, ws->dif,
           ws->work, ws->lwork,
           ws->rwork, ws->iwork,
           ws->bwork, &linfo);

    if (linfo != 0) {
        fprintf(stderr, "ZGGEVX returned INFO=%d for precomputed #%d\n",
                       linfo, idx);
        assert_info_success(linfo);
        return;
    }

    zlacpy("Full", n, n, ws->AI, lda, ws->work, n);
    zlacpy("Full", n, n, ws->BI, lda,
           ws->work + n * n, n);
    f64 abnrm = zlange("Fro", n, 2 * n,
                        ws->work, n, ws->rwork);

    f64 result[4] = {0.0, 0.0, 0.0, 0.0};
    f64 res52[2];

    res52[0] = 0.0;
    res52[1] = 0.0;
    zget52(1, n, ws->A, lda, ws->B, lda,
           ws->VL, lda, ws->alpha, ws->beta,
           ws->work, ws->rwork, res52);
    result[0] = res52[0];

    res52[0] = 0.0;
    res52[1] = 0.0;
    zget52(0, n, ws->A, lda, ws->B, lda,
           ws->VR, lda, ws->alpha, ws->beta,
           ws->work, ws->rwork, res52);
    result[1] = res52[0];

    result[2] = 0.0;
    for (INT i = 0; i < n; i++) {
        if (ws->s[i] == 0.0) {
            if (ws->dtru[i] > abnrm * ulp)
                result[2] = ulpinv;
        } else if (ws->dtru[i] == 0.0) {
            if (ws->s[i] > abnrm * ulp)
                result[2] = ulpinv;
        } else {
            f64 ratio = fmax(fabs(ws->dtru[i] / ws->s[i]),
                             fabs(ws->s[i] / ws->dtru[i]));
            if (ratio > result[2])
                result[2] = ratio;
        }
    }

    result[3] = 0.0;
    if (ws->dif[0] == 0.0) {
        if (ws->diftru[0] > abnrm * ulp)
            result[3] = ulpinv;
    } else if (ws->diftru[0] == 0.0) {
        if (ws->dif[0] > abnrm * ulp)
            result[3] = ulpinv;
    } else if (ws->dif[n - 1] == 0.0) {
        if (ws->diftru[n - 1] > abnrm * ulp)
            result[3] = ulpinv;
    } else if (ws->diftru[n - 1] == 0.0) {
        if (ws->dif[n - 1] > abnrm * ulp)
            result[3] = ulpinv;
    } else {
        f64 ratio1 = fmax(fabs(ws->diftru[0] / ws->dif[0]),
                          fabs(ws->dif[0] / ws->diftru[0]));
        f64 ratio2 = fmax(
            fabs(ws->diftru[n - 1] / ws->dif[n - 1]),
            fabs(ws->dif[n - 1] / ws->diftru[n - 1]));
        result[3] = fmax(ratio1, ratio2);
    }

    for (INT j = 0; j < 4; j++) {
        if (result[j] >= thrsh2) {
            fprintf(stderr, "  ZDRGVX precomputed #%d: test %d = %.6e\n",
                          idx, j + 1, result[j]);
        }
        assert_residual_below(result[j], thrsh2);
    }
}

/* ===================================================================== */
/* Test array construction and main                                      */
/* ===================================================================== */

static zdrgvx_params_t g_builtin_params[2 * 5];
static INT g_precomputed_idx[ZGVX_NUM_PRECOMPUTED];

int main(void)
{
    INT nbuiltin = 0;

    for (INT iptype = 1; iptype <= 2; iptype++) {
        for (INT iwa = 0; iwa < 5; iwa++) {
            zdrgvx_params_t* p = &g_builtin_params[nbuiltin];
            p->iptype = iptype;
            p->iwa    = iwa;
            snprintf(p->name, sizeof(p->name),
                     "zdrgvx_builtin_type%d_iwa%d", iptype, iwa);
            nbuiltin++;
        }
    }

    for (INT i = 0; i < ZGVX_NUM_PRECOMPUTED; i++)
        g_precomputed_idx[i] = i;

    INT total = nbuiltin + ZGVX_NUM_PRECOMPUTED;

    struct CMUnitTest* tests = malloc(total * sizeof(struct CMUnitTest));
    if (!tests) return 1;

    for (INT i = 0; i < nbuiltin; i++) {
        tests[i].name = g_builtin_params[i].name;
        tests[i].test_func = test_builtin;
        tests[i].setup_func = NULL;
        tests[i].teardown_func = NULL;
        tests[i].initial_state = &g_builtin_params[i];
    }

    for (INT i = 0; i < ZGVX_NUM_PRECOMPUTED; i++) {
        static char precomp_names[ZGVX_NUM_PRECOMPUTED][64];
        snprintf(precomp_names[i], sizeof(precomp_names[i]),
                 "zdrgvx_precomputed_%d", i);
        tests[nbuiltin + i].name = precomp_names[i];
        tests[nbuiltin + i].test_func = test_precomputed;
        tests[nbuiltin + i].setup_func = NULL;
        tests[nbuiltin + i].teardown_func = NULL;
        tests[nbuiltin + i].initial_state = &g_precomputed_idx[i];
    }

    int ret = _cmocka_run_group_tests("zdrgvx", tests, total,
                                       group_setup, group_teardown);
    free(tests);
    return ret;
}
