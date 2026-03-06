/**
 * @file test_cdrgsx.c
 * @brief Generalized Schur form expert driver test - port of LAPACK TESTING/EIG/zdrgsx.f
 *
 * Tests the nonsymmetric generalized eigenvalue (Schur form) expert driver CGGESX.
 *
 * Each (ifunc, prtype, m, n) combination is registered as a separate CMocka test
 * for built-in matrices. Precomputed read-in tests are separate CMocka tests.
 *
 * Test ratios (10 total):
 *                             H
 *   (1)  | A - Q S Z  | / ( |A| n ulp )
 *                             H
 *   (2)  | B - Q T Z  | / ( |B| n ulp )
 *                H
 *   (3)  | I - QQ  | / ( n ulp )
 *                H
 *   (4)  | I - ZZ  | / ( n ulp )
 *   (5)  A is in Schur form S
 *   (6)  difference between (alpha,beta) and diagonals of (S,T)
 *   (7)  if sorting worked and SDIM is the correct number of selected eigenvalues
 *   (8)  DIF estimate accuracy (vs exact from SVD of Kronecker matrix)
 *   (9)  if reordering failed (INFO=N+3), check DIF=PL=0
 *   (10) PL estimate accuracy (read-in tests only)
 */

#include "test_harness.h"
#include "verify.h"
#include <math.h>
#include <string.h>
#include <complex.h>

#include "cgsx_testdata.h"

#define THRESH  10.0f
#define NSIZE   2
#define NCMAX   3

/* ===================================================================== */
/* Test parameters and workspace                                         */
/* ===================================================================== */

typedef struct {
    INT ifunc;
    INT prtype;
    INT m;
    INT n;
    char name[128];
} zdrgsx_params_t;

typedef struct {
    INT nmax;
    c64* A;      /* original A */
    c64* B;      /* original B */
    c64* AI;     /* working copy, modified by cggesx */
    c64* BI;     /* working copy */
    c64* Q;      /* left Schur vectors */
    c64* Z;      /* right Schur vectors */
    c64* alpha;
    c64* beta;
    c64* C;      /* Kronecker matrix for DIF computation */
    f32*  S;      /* singular values of C */
    c64* work;
    f32*  rwork;
    INT*  iwork;
    INT*  bwork;
    INT   lwork;
    INT   liwork;
    INT   ldc;
    f32   result[10];
} zdrgsx_workspace_t;

static zdrgsx_workspace_t* g_ws = NULL;

/* ===================================================================== */
/* ABS1 statement function: |Re(x)| + |Im(x)|                           */
/* ===================================================================== */

static inline f32 abs1(c64 x)
{
    return fabsf(crealf(x)) + fabsf(cimagf(x));
}

/* ===================================================================== */
/* Group setup / teardown                                                */
/* ===================================================================== */

static int group_setup(void** state)
{
    (void)state;

    g_ws = malloc(sizeof(zdrgsx_workspace_t));
    if (!g_ws) return -1;

    INT nmax = NSIZE;
    if (nmax < 4) nmax = 4;
    g_ws->nmax = nmax;

    INT n2 = nmax * nmax;

    g_ws->A     = malloc(n2 * sizeof(c64));
    g_ws->B     = malloc(n2 * sizeof(c64));
    g_ws->AI    = malloc(n2 * sizeof(c64));
    g_ws->BI    = malloc(n2 * sizeof(c64));
    g_ws->Q     = malloc(n2 * sizeof(c64));
    g_ws->Z     = malloc(n2 * sizeof(c64));
    g_ws->alpha = malloc(nmax * sizeof(c64));
    g_ws->beta  = malloc(nmax * sizeof(c64));

    g_ws->ldc = nmax * nmax / 2;
    if (g_ws->ldc < 1) g_ws->ldc = 1;
    g_ws->C   = malloc(g_ws->ldc * g_ws->ldc * sizeof(c64));
    g_ws->S   = malloc(g_ws->ldc * sizeof(f32));

    g_ws->lwork = 3 * n2;
    if (g_ws->lwork < 1) g_ws->lwork = 1;
    g_ws->work  = malloc(g_ws->lwork * sizeof(c64));

    INT rwork_size = 5 * n2 / 2;
    if (rwork_size < 8 * nmax) rwork_size = 8 * nmax;
    if (rwork_size < 1) rwork_size = 1;
    g_ws->rwork = malloc(rwork_size * sizeof(f32));

    g_ws->liwork = nmax + 2;
    g_ws->iwork = malloc(g_ws->liwork * sizeof(INT));
    g_ws->bwork = malloc(nmax * sizeof(INT));

    if (!g_ws->A || !g_ws->B || !g_ws->AI || !g_ws->BI ||
        !g_ws->Q || !g_ws->Z || !g_ws->alpha || !g_ws->beta ||
        !g_ws->C || !g_ws->S || !g_ws->work || !g_ws->rwork ||
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
        free(g_ws->Q);
        free(g_ws->Z);
        free(g_ws->alpha);
        free(g_ws->beta);
        free(g_ws->C);
        free(g_ws->S);
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
/* Port of zdrgsx.f lines 502-706                                        */
/* ===================================================================== */

static void test_builtin(void** state)
{
    zdrgsx_params_t* p = *(zdrgsx_params_t**)state;
    zdrgsx_workspace_t* ws = g_ws;

    INT ifunc  = p->ifunc;
    INT prtype = p->prtype;
    INT m      = p->m;
    INT n      = p->n;
    INT mplusn = m + n;
    INT lda    = ws->nmax;

    f32 ulp    = slamch("P");
    f32 ulpinv = 1.0f / ulp;
    f32 smlnum = slamch("S") / ulp;
    f32 thrsh2 = 10.0f * THRESH;

    static f32 weight = 0.0f;
    if (weight == 0.0f) weight = sqrtf(ulp);
    weight = 1.0f / weight;

    for (INT i = 0; i < 10; i++)
        ws->result[i] = 0.0f;

    clctsx_reset(m, n, mplusn);

    const c64 czero = CMPLXF(0.0f, 0.0f);

    claset("Full", mplusn, mplusn, czero, czero, ws->AI, lda);
    claset("Full", mplusn, mplusn, czero, czero, ws->BI, lda);

    INT qba = 3;
    INT qbb = 4;
    clatm5(prtype, m, n,
           ws->AI, lda,
           ws->AI + m + m * lda, lda,
           ws->AI + m * lda, lda,
           ws->BI, lda,
           ws->BI + m + m * lda, lda,
           ws->BI + m * lda, lda,
           ws->Q, lda,
           ws->Z, lda,
           weight, qba, qbb);

    const char* sense;
    if (ifunc == 0)      sense = "N";
    else if (ifunc == 1) sense = "E";
    else if (ifunc == 2) sense = "V";
    else                 sense = "B";

    clacpy("Full", mplusn, mplusn, ws->AI, lda, ws->A, lda);
    clacpy("Full", mplusn, mplusn, ws->BI, lda, ws->B, lda);

    INT mm = 0;
    f32 pl[2] = {0.0f, 0.0f};
    f32 difest[2] = {0.0f, 0.0f};
    INT linfo = 0;

    cggesx("V", "V", "S", clctsx, sense, mplusn,
           ws->AI, lda, ws->BI, lda, &mm,
           ws->alpha, ws->beta,
           ws->Q, lda, ws->Z, lda,
           pl, difest,
           ws->work, ws->lwork, ws->rwork,
           ws->iwork, ws->liwork, ws->bwork, &linfo);

    f32 abnrm = 0.0f;
    f32 temp1 = 0.0f;
    f32 diftru_val = 0.0f;
    INT mn2 = 0;

    if (linfo != 0 && linfo != mplusn + 2) {
        ws->result[0] = ulpinv;
        fprintf(stderr, "CGGESX returned INFO=%d, MPLUSN=%d, PRTYPE=%d\n",
                       linfo, mplusn, prtype);
        goto check_results;
    }

    clacpy("Full", mplusn, mplusn, ws->AI, mplusn, ws->work, mplusn);
    clacpy("Full", mplusn, mplusn, ws->BI, mplusn,
           ws->work + mplusn * mplusn, mplusn);
    abnrm = clange("Fro", mplusn, 2 * mplusn, ws->work, mplusn, ws->rwork);

    cget51(1, mplusn, ws->A, lda, ws->AI, lda, ws->Q, lda, ws->Z, lda,
           ws->work, ws->rwork, &ws->result[0]);
    cget51(1, mplusn, ws->B, lda, ws->BI, lda, ws->Q, lda, ws->Z, lda,
           ws->work, ws->rwork, &ws->result[1]);
    cget51(3, mplusn, ws->B, lda, ws->BI, lda, ws->Q, lda, ws->Q, lda,
           ws->work, ws->rwork, &ws->result[2]);
    cget51(3, mplusn, ws->B, lda, ws->BI, lda, ws->Z, lda, ws->Z, lda,
           ws->work, ws->rwork, &ws->result[3]);

    ws->result[4] = 0.0f;
    ws->result[5] = 0.0f;

    for (INT j = 0; j < mplusn; j++) {
        f32 temp2 = (abs1(ws->alpha[j] - ws->AI[j + j * lda]) /
                     fmaxf(smlnum, fmaxf(abs1(ws->alpha[j]),
                                       abs1(ws->AI[j + j * lda]))) +
                     abs1(ws->beta[j] - ws->BI[j + j * lda]) /
                     fmaxf(smlnum, fmaxf(abs1(ws->beta[j]),
                                       abs1(ws->BI[j + j * lda])))) / ulp;

        if (j < mplusn - 1) {
            if (ws->AI[(j + 1) + j * lda] != 0.0f)
                ws->result[4] = ulpinv;
        }
        if (j > 0) {
            if (ws->AI[j + (j - 1) * lda] != 0.0f)
                ws->result[4] = ulpinv;
        }
        if (temp2 > temp1) temp1 = temp2;
    }
    ws->result[5] = temp1;

    ws->result[6] = 0.0f;
    if (linfo == mplusn + 3) {
        ws->result[6] = ulpinv;
    } else if (mm != n) {
        ws->result[6] = ulpinv;
    }

    ws->result[7] = 0.0f;
    mn2 = mm * (mplusn - mm) * 2;
    if (ifunc >= 2 && mn2 <= NCMAX * NCMAX) {
        clakf2(mm, mplusn - mm,
               ws->AI, lda, ws->AI + mm + mm * lda,
               ws->BI, ws->BI + mm + mm * lda,
               ws->C, ws->ldc);

        INT svd_info = 0;
        cgesvd("N", "N", mn2, mn2, ws->C, ws->ldc, ws->S,
               ws->work, 1, ws->work + 1, 1,
               ws->work + 2, ws->lwork - 2,
               ws->rwork, &svd_info);
        diftru_val = ws->S[mn2 - 1];

        if (difest[1] == 0.0f) {
            if (diftru_val > abnrm * ulp)
                ws->result[7] = ulpinv;
        } else if (diftru_val == 0.0f) {
            if (difest[1] > abnrm * ulp)
                ws->result[7] = ulpinv;
        } else if ((diftru_val > thrsh2 * difest[1]) ||
                   (diftru_val * thrsh2 < difest[1])) {
            ws->result[7] = fmaxf(diftru_val / difest[1],
                                 difest[1] / diftru_val);
        }
    }

    ws->result[8] = 0.0f;
    if (linfo == (mplusn + 2)) {
        if (diftru_val > abnrm * ulp)
            ws->result[8] = ulpinv;
        if ((ifunc > 1) && (difest[1] != 0.0f))
            ws->result[8] = ulpinv;
        if ((ifunc == 1) && (pl[0] != 0.0f))
            ws->result[8] = ulpinv;
    }

check_results:
    for (INT j = 0; j < 9; j++) {
        if (ws->result[j] >= THRESH) {
            fprintf(stderr, "  ZDRGSX builtin: MPLUSN=%d, PRTYPE=%d, IFUNC=%d, M=%d, "
                          "test %d = %.6e\n",
                          mplusn, prtype, ifunc, m, j + 1, (double)ws->result[j]);
        }
        assert_residual_below(ws->result[j], THRESH);
    }
}

/* ===================================================================== */
/* Precomputed (read-in) test function                                   */
/* Port of zdrgsx.f lines 717-888                                        */
/* ===================================================================== */

static void test_precomputed(void** state)
{
    INT* idx_ptr = *(INT**)state;
    INT idx = *idx_ptr;
    zdrgsx_workspace_t* ws = g_ws;

    const zgsx_precomputed_t* tc = &ZGSX_PRECOMPUTED[idx];
    INT mplusn = tc->mplusn;
    INT n = tc->n;
    INT m = mplusn - n;
    INT lda = ws->nmax;
    f32 pltru = tc->pltru;
    f32 diftru_val = tc->diftru;

    f32 ulp    = slamch("P");
    f32 ulpinv = 1.0f / ulp;
    f32 smlnum = slamch("S") / ulp;
    f32 thrsh2 = 10.0f * THRESH;

    for (INT i = 0; i < 10; i++)
        ws->result[i] = 0.0f;

    for (INT j = 0; j < mplusn; j++)
        for (INT i = 0; i < mplusn; i++)
            ws->AI[i + j * lda] = tc->A[i + j * mplusn];
    for (INT j = 0; j < mplusn; j++)
        for (INT i = 0; i < mplusn; i++)
            ws->BI[i + j * lda] = tc->B[i + j * mplusn];

    clctsx_reset(m, n, mplusn);

    clacpy("Full", mplusn, mplusn, ws->AI, lda, ws->A, lda);
    clacpy("Full", mplusn, mplusn, ws->BI, lda, ws->B, lda);

    INT mm = 0;
    f32 pl[2] = {0.0f, 0.0f};
    f32 difest[2] = {0.0f, 0.0f};
    INT linfo = 0;

    cggesx("V", "V", "S", clctsx, "B", mplusn,
           ws->AI, lda, ws->BI, lda, &mm,
           ws->alpha, ws->beta,
           ws->Q, lda, ws->Z, lda,
           pl, difest,
           ws->work, ws->lwork, ws->rwork,
           ws->iwork, ws->liwork, ws->bwork, &linfo);

    f32 abnrm = 0.0f;
    f32 temp1 = 0.0f;

    if (linfo != 0 && linfo != mplusn + 2) {
        ws->result[0] = ulpinv;
        fprintf(stderr, "CGGESX returned INFO=%d for precomputed #%d\n", linfo, idx);
        goto check_results;
    }

    clacpy("Full", mplusn, mplusn, ws->AI, mplusn, ws->work, mplusn);
    clacpy("Full", mplusn, mplusn, ws->BI, mplusn,
           ws->work + mplusn * mplusn, mplusn);
    abnrm = clange("Fro", mplusn, 2 * mplusn, ws->work, mplusn, ws->rwork);

    cget51(1, mplusn, ws->A, lda, ws->AI, lda, ws->Q, lda, ws->Z, lda,
           ws->work, ws->rwork, &ws->result[0]);
    cget51(1, mplusn, ws->B, lda, ws->BI, lda, ws->Q, lda, ws->Z, lda,
           ws->work, ws->rwork, &ws->result[1]);
    cget51(3, mplusn, ws->B, lda, ws->BI, lda, ws->Q, lda, ws->Q, lda,
           ws->work, ws->rwork, &ws->result[2]);
    cget51(3, mplusn, ws->B, lda, ws->BI, lda, ws->Z, lda, ws->Z, lda,
           ws->work, ws->rwork, &ws->result[3]);

    ws->result[4] = 0.0f;
    ws->result[5] = 0.0f;

    for (INT j = 0; j < mplusn; j++) {
        f32 temp2 = (abs1(ws->alpha[j] - ws->AI[j + j * lda]) /
                     fmaxf(smlnum, fmaxf(abs1(ws->alpha[j]),
                                       abs1(ws->AI[j + j * lda]))) +
                     abs1(ws->beta[j] - ws->BI[j + j * lda]) /
                     fmaxf(smlnum, fmaxf(abs1(ws->beta[j]),
                                       abs1(ws->BI[j + j * lda])))) / ulp;

        if (j < mplusn - 1) {
            if (ws->AI[(j + 1) + j * lda] != 0.0f)
                ws->result[4] = ulpinv;
        }
        if (j > 0) {
            if (ws->AI[j + (j - 1) * lda] != 0.0f)
                ws->result[4] = ulpinv;
        }
        if (temp2 > temp1) temp1 = temp2;
    }
    ws->result[5] = temp1;

    ws->result[6] = 0.0f;
    if (linfo == mplusn + 3)
        ws->result[6] = ulpinv;

    ws->result[7] = 0.0f;
    if (difest[1] == 0.0f) {
        if (diftru_val > abnrm * ulp)
            ws->result[7] = ulpinv;
    } else if (diftru_val == 0.0f) {
        if (difest[1] > abnrm * ulp)
            ws->result[7] = ulpinv;
    } else if ((diftru_val > thrsh2 * difest[1]) ||
               (diftru_val * thrsh2 < difest[1])) {
        ws->result[7] = fmaxf(diftru_val / difest[1],
                             difest[1] / diftru_val);
    }

    ws->result[8] = 0.0f;
    if (linfo == (mplusn + 2)) {
        if (diftru_val > abnrm * ulp)
            ws->result[8] = ulpinv;
        if (difest[1] != 0.0f)
            ws->result[8] = ulpinv;
        if (pl[0] != 0.0f)
            ws->result[8] = ulpinv;
    }

    ws->result[9] = 0.0f;
    if (pl[0] == 0.0f) {
        if (pltru > abnrm * ulp)
            ws->result[9] = ulpinv;
    } else if (pltru == 0.0f) {
        if (pl[0] > abnrm * ulp)
            ws->result[9] = ulpinv;
    } else if ((pltru > THRESH * pl[0]) ||
               (pltru * THRESH < pl[0])) {
        ws->result[9] = ulpinv;
    }

check_results:
    for (INT j = 0; j < 10; j++) {
        if (ws->result[j] >= THRESH) {
            fprintf(stderr, "  ZDRGSX precomputed #%d: test %d = %.6e\n",
                          idx, j + 1, (double)ws->result[j]);
        }
        assert_residual_below(ws->result[j], THRESH);
    }
}

/* ===================================================================== */
/* Test array construction and main                                      */
/* ===================================================================== */

#define MAX_TESTS (4 * 5 * (NSIZE) * (NSIZE) + ZGSX_NUM_PRECOMPUTED)

static zdrgsx_params_t g_builtin_params[4 * 5 * NSIZE * NSIZE];
static INT g_precomputed_idx[ZGSX_NUM_PRECOMPUTED];

int main(void)
{
    INT nbuiltin = 0;

    for (INT ifunc = 0; ifunc <= 3; ifunc++) {
        for (INT prtype = 1; prtype <= 5; prtype++) {
            for (INT m = 1; m <= NSIZE - 1; m++) {
                for (INT n = 1; n <= NSIZE - m; n++) {
                    zdrgsx_params_t* p = &g_builtin_params[nbuiltin];
                    p->ifunc  = ifunc;
                    p->prtype = prtype;
                    p->m      = m;
                    p->n      = n;
                    snprintf(p->name, sizeof(p->name),
                             "zdrgsx_builtin_sense%d_type%d_m%d_n%d",
                             ifunc, prtype, m, n);
                    nbuiltin++;
                }
            }
        }
    }

    for (INT i = 0; i < ZGSX_NUM_PRECOMPUTED; i++)
        g_precomputed_idx[i] = i;

    INT total = nbuiltin + ZGSX_NUM_PRECOMPUTED;

    struct CMUnitTest* tests = malloc(total * sizeof(struct CMUnitTest));
    if (!tests) return 1;

    for (INT i = 0; i < nbuiltin; i++) {
        tests[i].name = g_builtin_params[i].name;
        tests[i].test_func = test_builtin;
        tests[i].setup_func = NULL;
        tests[i].teardown_func = NULL;
        tests[i].initial_state = &g_builtin_params[i];
    }

    for (INT i = 0; i < ZGSX_NUM_PRECOMPUTED; i++) {
        static char precomp_names[ZGSX_NUM_PRECOMPUTED][64];
        snprintf(precomp_names[i], sizeof(precomp_names[i]),
                 "zdrgsx_precomputed_%d", i);
        tests[nbuiltin + i].name = precomp_names[i];
        tests[nbuiltin + i].test_func = test_precomputed;
        tests[nbuiltin + i].setup_func = NULL;
        tests[nbuiltin + i].teardown_func = NULL;
        tests[nbuiltin + i].initial_state = &g_precomputed_idx[i];
    }

    int ret = _cmocka_run_group_tests("zdrgsx", tests, total,
                                       group_setup, group_teardown);
    free(tests);
    return ret;
}
