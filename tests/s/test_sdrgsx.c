/**
 * @file test_sdrgsx.c
 * @brief Generalized nonsymmetric Schur form expert driver test - port of
 *        LAPACK TESTING/EIG/ddrgsx.f
 *
 * Tests the nonsymmetric generalized eigenvalue (Schur form) problem expert
 * driver SGGESX.
 *
 * SGGESX factors A and B as Q S Z' and Q T Z', where ' means
 * transpose, T is upper triangular, S is in generalized Schur form
 * (block upper triangular, with 1x1 and 2x2 blocks on the diagonal,
 * the 2x2 blocks corresponding to complex conjugate pairs of
 * generalized eigenvalues), and Q and Z are orthogonal. It also
 * computes the generalized eigenvalues (alpha(j),beta(j)), optionally
 * reorders eigenvalues, and computes reciprocal condition numbers.
 *
 * Test ratios (9 total):
 *   (1) | A - Q S Z' | / ( |A| n ulp )
 *   (2) | B - Q T Z' | / ( |B| n ulp )
 *   (3) | I - QQ' | / ( n ulp )
 *   (4) | I - ZZ' | / ( n ulp )
 *   (5) if A is in Schur form (quasi-triangular)
 *   (6) eigenvalue accuracy (diagonal comparison / sget53)
 *   (7) if sorting worked (SDIM == expected)
 *   (8) DIF accuracy vs exact (via slakf2 + sgesvd SVD)
 *   (9) reordering failure checks (INFO = MPLUSN+2)
 *
 * Matrix types: 5 types from SLATM5
 * SENSE options: 'N','E','V','B' (4 variants)
 * Block sizes: m = 1..NSIZE-1, n = 1..NSIZE-m
 * Total for NSIZE=5: 4 x 5 x 10 = 200 test cases.
 */

#include "test_harness.h"
#include "verify.h"
#include <cblas.h>
#include <math.h>
#include <string.h>

#define THRESH 10.0f
#define NSIZE 2
#define NMAX 4

/* Shared state for dlctsx callback (replaces Fortran COMMON /MN/) */
static int g_sel_m, g_sel_n, g_sel_mplusn, g_sel_k;
static int g_sel_fs;

static int dlctsx(const f32* ar, const f32* ai, const f32* beta)
{
    (void)ar; (void)ai; (void)beta;
    int res;
    g_sel_k++;
    if (g_sel_fs) {
        res = (g_sel_k > g_sel_m) ? 1 : 0;
        if (g_sel_k == g_sel_mplusn) {
            g_sel_fs = 0;
            g_sel_k = 0;
        }
    } else {
        res = (g_sel_k <= g_sel_n) ? 1 : 0;
        if (g_sel_k == g_sel_mplusn) {
            g_sel_fs = 1;
            g_sel_k = 0;
        }
    }
    return res;
}

extern void sggesx(const char* jobvsl, const char* jobvsr, const char* sort,
                   int (*selctg)(const f32*, const f32*, const f32*),
                   const char* sense, const int n,
                   f32* A, const int lda, f32* B, const int ldb,
                   int* sdim, f32* alphar, f32* alphai, f32* beta,
                   f32* VSL, const int ldvsl, f32* VSR, const int ldvsr,
                   f32* rconde, f32* rcondv,
                   f32* work, const int lwork,
                   int* iwork, const int liwork, int* bwork, int* info);
extern void sgesvd(const char* jobu, const char* jobvt,
                   const int m, const int n, f32* A, const int lda,
                   f32* S, f32* U, const int ldu, f32* VT, const int ldvt,
                   f32* work, const int lwork, int* info);
extern f32 slamch(const char* cmach);
extern f32 slange(const char* norm, const int m, const int n,
                  const f32* A, const int lda, f32* work);
extern void slacpy(const char* uplo, const int m, const int n,
                   const f32* A, const int lda, f32* B, const int ldb);
extern void slaset(const char* uplo, const int m, const int n,
                   const f32 alpha, const f32 beta_val, f32* A, const int lda);

typedef struct {
    int ifunc;
    int prtype;
    int m;
    int n;
    char name[96];
} ddrgsx_params_t;

typedef struct {
    f32* A;
    f32* B;
    f32* AI;
    f32* BI;
    f32* Q;
    f32* Z;
    f32* alphar;
    f32* alphai;
    f32* beta;
    f32* C;       /* Kronecker matrix for DIF */
    f32* S;       /* Singular values */
    f32* work;
    int* iwork;
    int* bwork;
    int lwork;
    int ldc;
} ddrgsx_workspace_t;

static ddrgsx_workspace_t* g_ws = NULL;

static int group_setup(void** state)
{
    (void)state;

    g_ws = calloc(1, sizeof(ddrgsx_workspace_t));
    if (!g_ws) return -1;

    /* Allocate for NMAX (accommodates read-in 4x4 matrices) */
    const int ns = NMAX;
    const int n2 = ns * ns;

    g_ws->A      = malloc(n2 * sizeof(f32));
    g_ws->B      = malloc(n2 * sizeof(f32));
    g_ws->AI     = malloc(n2 * sizeof(f32));
    g_ws->BI     = malloc(n2 * sizeof(f32));
    g_ws->Q      = malloc(n2 * sizeof(f32));
    g_ws->Z      = malloc(n2 * sizeof(f32));
    g_ws->alphar = malloc(ns * sizeof(f32));
    g_ws->alphai = malloc(ns * sizeof(f32));
    g_ws->beta   = malloc(ns * sizeof(f32));

    /* Kronecker matrix for DIF computation: ldc = ns*ns/2 */
    g_ws->ldc = ns * ns / 2;
    if (g_ws->ldc < 1) g_ws->ldc = 1;
    g_ws->C = malloc(g_ws->ldc * g_ws->ldc * sizeof(f32));
    g_ws->S = malloc(g_ws->ldc * sizeof(f32));

    /* Workspace: max(10*(ns+1), 5*ns*ns/2) + extra for sgesvd */
    int minwrk = 10 * (ns + 1);
    int bdspac = 5 * n2 / 2;
    if (bdspac > minwrk) minwrk = bdspac;
    g_ws->lwork = minwrk + n2;  /* extra margin */
    g_ws->work  = malloc(g_ws->lwork * sizeof(f32));
    g_ws->iwork = malloc((ns + 6) * sizeof(int));
    g_ws->bwork = malloc(ns * sizeof(int));

    if (!g_ws->A || !g_ws->B || !g_ws->AI || !g_ws->BI ||
        !g_ws->Q || !g_ws->Z || !g_ws->alphar || !g_ws->alphai ||
        !g_ws->beta || !g_ws->C || !g_ws->S ||
        !g_ws->work || !g_ws->iwork || !g_ws->bwork) {
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
        free(g_ws->alphar);
        free(g_ws->alphai);
        free(g_ws->beta);
        free(g_ws->C);
        free(g_ws->S);
        free(g_ws->work);
        free(g_ws->iwork);
        free(g_ws->bwork);
        free(g_ws);
        g_ws = NULL;
    }
    return 0;
}

static void test_ddrgsx(void** state)
{
    ddrgsx_params_t* params = (ddrgsx_params_t*)(*state);

    const int ifunc = params->ifunc;
    const int prtype = params->prtype;
    const int m = params->m;
    const int n = params->n;
    const int mplusn = m + n;
    const int lda = NMAX;

    const f32 ulp = slamch("P");
    const f32 ulpinv = 1.0f / ulp;
    const f32 smlnum = slamch("S") / ulp;
    const f32 thrsh2 = 10.0f * THRESH;

    f32 result[10];
    for (int i = 0; i < 10; i++) result[i] = 0.0f;

    /* Weight oscillates: sqrt(ulp), 1/sqrt(ulp), sqrt(ulp), ...
       Fortran initializes WEIGHT = SQRT(ULP), then flips at start of each
       iteration. Compute deterministically from the flat iteration index. */
    int inner_count = 0;
    for (int mm = 1; mm <= NSIZE - 1; mm++)
        for (int nn = 1; nn <= NSIZE - mm; nn++)
            inner_count++;
    int inner_idx = 0;
    for (int mm = 1; mm < m; mm++)
        inner_idx += NSIZE - mm;
    inner_idx += n - 1;
    int iter_idx = ifunc * 5 * inner_count + (prtype - 1) * inner_count + inner_idx;
    f32 weight = ((iter_idx % 2) == 0) ? 1.0f / sqrtf(ulp) : sqrtf(ulp);

    /* Reset selection callback state */
    g_sel_fs = 1;
    g_sel_k = 0;
    g_sel_m = m;
    g_sel_n = n;
    g_sel_mplusn = mplusn;

    /* Generate test matrices */
    slaset("Full", mplusn, mplusn, 0.0f, 0.0f, g_ws->AI, lda);
    slaset("Full", mplusn, mplusn, 0.0f, 0.0f, g_ws->BI, lda);

    slatm5(prtype, m, n,
           g_ws->AI, lda,
           &g_ws->AI[m + m * lda], lda,
           &g_ws->AI[m * lda], lda,
           g_ws->BI, lda,
           &g_ws->BI[m + m * lda], lda,
           &g_ws->BI[m * lda], lda,
           g_ws->Q, lda, g_ws->Z, lda,
           weight, 3, 4);

    const char* sense;
    if (ifunc == 0) sense = "N";
    else if (ifunc == 1) sense = "E";
    else if (ifunc == 2) sense = "V";
    else sense = "B";

    slacpy("Full", mplusn, mplusn, g_ws->AI, lda, g_ws->A, lda);
    slacpy("Full", mplusn, mplusn, g_ws->BI, lda, g_ws->B, lda);

    int mm, linfo;
    f32 pl[2], difest[2];

    /* Reset selection callback again for the actual call */
    g_sel_fs = 1;
    g_sel_k = 0;
    g_sel_m = m;
    g_sel_n = n;
    g_sel_mplusn = mplusn;

    sggesx("V", "V", "S", dlctsx, sense, mplusn,
           g_ws->AI, lda, g_ws->BI, lda, &mm,
           g_ws->alphar, g_ws->alphai, g_ws->beta,
           g_ws->Q, lda, g_ws->Z, lda,
           pl, difest,
           g_ws->work, g_ws->lwork,
           g_ws->iwork, NMAX + 6, g_ws->bwork, &linfo);

    if (linfo != 0 && linfo != mplusn + 2) {
        print_message("SGGESX returned INFO=%d for sense=%s type=%d m=%d n=%d\n",
                      linfo, sense, prtype, m, n);
        result[0] = ulpinv;
        assert_residual_ok(result[0]);
        return;
    }

    /* Compute norm(A, B) */
    slacpy("Full", mplusn, mplusn, g_ws->AI, lda, g_ws->work, mplusn);
    slacpy("Full", mplusn, mplusn, g_ws->BI, lda,
           &g_ws->work[mplusn * mplusn], mplusn);
    f32 abnrm = slange("Fro", mplusn, 2 * mplusn, g_ws->work, mplusn,
                        g_ws->work);

    /* Tests (1) to (4) via sget51 */
    sget51(1, mplusn, g_ws->A, lda, g_ws->AI, lda, g_ws->Q, lda,
           g_ws->Z, lda, g_ws->work, &result[0]);
    sget51(1, mplusn, g_ws->B, lda, g_ws->BI, lda, g_ws->Q, lda,
           g_ws->Z, lda, g_ws->work, &result[1]);
    sget51(3, mplusn, g_ws->B, lda, g_ws->BI, lda, g_ws->Q, lda,
           g_ws->Q, lda, g_ws->work, &result[2]);
    sget51(3, mplusn, g_ws->B, lda, g_ws->BI, lda, g_ws->Z, lda,
           g_ws->Z, lda, g_ws->work, &result[3]);

    /* Tests (5) and (6): Schur form structure and eigenvalue accuracy */
    f32 temp1 = 0.0f;
    result[4] = 0.0f;
    result[5] = 0.0f;

    for (int j = 0; j < mplusn; j++) {
        int ilabad = 0;
        f32 temp2;
        if (g_ws->alphai[j] == 0.0f) {
            temp2 = (fabsf(g_ws->alphar[j] - g_ws->AI[j + j * lda]) /
                     fmaxf(smlnum, fmaxf(fabsf(g_ws->alphar[j]),
                          fabsf(g_ws->AI[j + j * lda]))) +
                     fabsf(g_ws->beta[j] - g_ws->BI[j + j * lda]) /
                     fmaxf(smlnum, fmaxf(fabsf(g_ws->beta[j]),
                          fabsf(g_ws->BI[j + j * lda])))) / ulp;
            if (j < mplusn - 1) {
                if (g_ws->AI[(j + 1) + j * lda] != 0.0f) {
                    ilabad = 1;
                    result[4] = ulpinv;
                }
            }
            if (j > 0) {
                if (g_ws->AI[j + (j - 1) * lda] != 0.0f) {
                    ilabad = 1;
                    result[4] = ulpinv;
                }
            }
        } else {
            int i1;
            if (g_ws->alphai[j] > 0.0f) {
                i1 = j;
            } else {
                i1 = j - 1;
            }
            if (i1 < 0 || i1 >= mplusn) {
                ilabad = 1;
            } else if (i1 < mplusn - 2) {
                if (g_ws->AI[(i1 + 2) + (i1 + 1) * lda] != 0.0f) {
                    ilabad = 1;
                    result[4] = ulpinv;
                }
            } else if (i1 > 0) {
                if (g_ws->AI[i1 + (i1 - 1) * lda] != 0.0f) {
                    ilabad = 1;
                    result[4] = ulpinv;
                }
            }
            if (!ilabad) {
                int iinfo;
                sget53(&g_ws->AI[i1 + i1 * lda], lda,
                       &g_ws->BI[i1 + i1 * lda], lda,
                       g_ws->beta[j], g_ws->alphar[j], g_ws->alphai[j],
                       &temp2, &iinfo);
            } else {
                temp2 = ulpinv;
            }
        }
        temp1 = fmaxf(temp1, temp2);
    }
    result[5] = temp1;

    /* Test (7): if sorting worked */
    result[6] = 0.0f;
    if (linfo == mplusn + 3) {
        result[6] = ulpinv;
    } else if (mm != n) {
        result[6] = ulpinv;
    }

    /* Test (8): DIF accuracy vs exact */
    result[7] = 0.0f;
    int mn2 = mm * (mplusn - mm) * 2;
    int ncmax = NMAX * NMAX;
    if (ifunc >= 2 && mn2 <= ncmax) {
        slakf2(mm, mplusn - mm,
               g_ws->AI, lda,
               &g_ws->AI[mm + mm * lda],
               g_ws->BI,
               &g_ws->BI[mm + mm * lda],
               g_ws->C, g_ws->ldc);

        int svd_info;
        sgesvd("N", "N", mn2, mn2, g_ws->C, g_ws->ldc, g_ws->S,
               g_ws->work, 1, &g_ws->work[1], 1,
               &g_ws->work[2], g_ws->lwork - 2, &svd_info);
        f32 diftru = g_ws->S[mn2 - 1];

        if (difest[1] == 0.0f) {
            if (diftru > abnrm * ulp)
                result[7] = ulpinv;
        } else if (diftru == 0.0f) {
            if (difest[1] > abnrm * ulp)
                result[7] = ulpinv;
        } else if ((diftru > thrsh2 * difest[1]) ||
                   (diftru * thrsh2 < difest[1])) {
            result[7] = fmaxf(diftru / difest[1], difest[1] / diftru);
        }
    }

    /* Test (9): reordering failure */
    result[8] = 0.0f;
    if (linfo == (mplusn + 2)) {
        f32 diftru_local = 0.0f;
        if (ifunc >= 2 && mn2 <= ncmax) {
            /* diftru was computed above */
            diftru_local = g_ws->S[mn2 - 1];
        }
        if (diftru_local > abnrm * ulp)
            result[8] = ulpinv;
        if ((ifunc > 1) && (difest[1] != 0.0f))
            result[8] = ulpinv;
        if ((ifunc == 1) && (pl[0] != 0.0f))
            result[8] = ulpinv;
    }

    /* Check results against thresholds */
    int any_fail = 0;
    for (int j = 0; j < 9; j++) {
        if (result[j] >= THRESH) {
            print_message("sense=%s type=%d m=%d n=%d test(%d)=%g\n",
                          sense, prtype, m, n, j + 1, (double)result[j]);
            any_fail = 1;
        }
    }
    assert_int_equal(any_fail, 0);
}

/*
 * Section 2: Read-in test data from dgd.in (LAPACK TESTING/dgd.in lines 23-51).
 * Two precomputed 4x4 matrix pairs with known condition numbers for the
 * eigenvalue cluster and deflating subspace, used to check accuracy of
 * condition estimation in SGGESX.
 */

#define NREADIN 2

typedef struct {
    int mplusn;
    int n;
    int idx;
    char name[64];
} ddrgsx_readin_params_t;

/* Column-major storage */
static const f32 readin_AI[NREADIN][16] = {
    {  8.0f,  0.0f,  0.0f,  0.0f,
       4.0f,  7.0f,  0.0f,  0.0f,
     -13.0f,-24.0f,  3.0f,  0.0f,
       4.0f, -3.0f, -5.0f, 16.0f },
    {  1.0f,  0.0f,  0.0f,  0.0f,
       2.0f,  5.0f,  0.0f,  0.0f,
       3.0f,  6.0f,  8.0f,  0.0f,
       4.0f,  7.0f,  9.0f, 10.0f }
};

static const f32 readin_BI[NREADIN][16] = {
    {  9.0f,  0.0f,  0.0f,  0.0f,
      -1.0f,  4.0f,  0.0f,  0.0f,
       1.0f, 16.0f,-11.0f,  0.0f,
      -6.0f,-24.0f,  6.0f,  4.0f },
    { -1.0f,  0.0f,  0.0f,  0.0f,
      -1.0f, -1.0f,  0.0f,  0.0f,
      -1.0f, -1.0f,  1.0f,  0.0f,
      -1.0f, -1.0f, -1.0f,  1.0f }
};

static const f32 readin_pltru[NREADIN] = { 2.5901e-01f, 9.8173e-01f };
static const f32 readin_diftru[NREADIN] = { 1.7592e+00f, 6.3649e-01f };

static void test_ddrgsx_readin(void** state)
{
    ddrgsx_readin_params_t* params = (ddrgsx_readin_params_t*)(*state);

    const int mplusn = params->mplusn;
    const int n = params->n;
    const int m = mplusn - n;
    const int lda = NMAX;
    const int ci = params->idx;

    const f32 ulp = slamch("P");
    const f32 ulpinv = 1.0f / ulp;
    const f32 smlnum = slamch("S") / ulp;
    const f32 thrsh2 = 10.0f * THRESH;

    f32 result[10];
    for (int i = 0; i < 10; i++) result[i] = 0.0f;

    /* Load precomputed matrices (lda may differ from mplusn) */
    for (int j = 0; j < mplusn; j++)
        for (int i = 0; i < mplusn; i++) {
            g_ws->AI[i + j * lda] = readin_AI[ci][i + j * mplusn];
            g_ws->BI[i + j * lda] = readin_BI[ci][i + j * mplusn];
        }

    /* Reset selection callback */
    g_sel_fs = 1;
    g_sel_k = 0;
    g_sel_m = m;
    g_sel_n = n;
    g_sel_mplusn = mplusn;

    slacpy("Full", mplusn, mplusn, g_ws->AI, lda, g_ws->A, lda);
    slacpy("Full", mplusn, mplusn, g_ws->BI, lda, g_ws->B, lda);

    int mm, linfo;
    f32 pl[2], difest[2];

    /* Reset selection callback again */
    g_sel_fs = 1;
    g_sel_k = 0;
    g_sel_m = m;
    g_sel_n = n;
    g_sel_mplusn = mplusn;

    sggesx("V", "V", "S", dlctsx, "B", mplusn,
           g_ws->AI, lda, g_ws->BI, lda, &mm,
           g_ws->alphar, g_ws->alphai, g_ws->beta,
           g_ws->Q, lda, g_ws->Z, lda,
           pl, difest,
           g_ws->work, g_ws->lwork,
           g_ws->iwork, NMAX + 6, g_ws->bwork, &linfo);

    if (linfo != 0 && linfo != mplusn + 2) {
        print_message("SGGESX returned INFO=%d for read-in example #%d\n",
                      linfo, ci + 1);
        result[0] = ulpinv;
        assert_residual_below(result[0], thrsh2);
        return;
    }

    /* Compute norm(A, B) */
    slacpy("Full", mplusn, mplusn, g_ws->AI, lda, g_ws->work, mplusn);
    slacpy("Full", mplusn, mplusn, g_ws->BI, lda,
           &g_ws->work[mplusn * mplusn], mplusn);
    f32 abnrm = slange("Fro", mplusn, 2 * mplusn, g_ws->work, mplusn,
                        g_ws->work);

    /* Tests (1) to (4) via sget51 */
    sget51(1, mplusn, g_ws->A, lda, g_ws->AI, lda, g_ws->Q, lda,
           g_ws->Z, lda, g_ws->work, &result[0]);
    sget51(1, mplusn, g_ws->B, lda, g_ws->BI, lda, g_ws->Q, lda,
           g_ws->Z, lda, g_ws->work, &result[1]);
    sget51(3, mplusn, g_ws->B, lda, g_ws->BI, lda, g_ws->Q, lda,
           g_ws->Q, lda, g_ws->work, &result[2]);
    sget51(3, mplusn, g_ws->B, lda, g_ws->BI, lda, g_ws->Z, lda,
           g_ws->Z, lda, g_ws->work, &result[3]);

    /* Tests (5) and (6): Schur form structure and eigenvalue accuracy */
    f32 temp1 = 0.0f;
    result[4] = 0.0f;
    result[5] = 0.0f;

    for (int j = 0; j < mplusn; j++) {
        int ilabad = 0;
        f32 temp2;
        if (g_ws->alphai[j] == 0.0f) {
            temp2 = (fabsf(g_ws->alphar[j] - g_ws->AI[j + j * lda]) /
                     fmaxf(smlnum, fmaxf(fabsf(g_ws->alphar[j]),
                          fabsf(g_ws->AI[j + j * lda]))) +
                     fabsf(g_ws->beta[j] - g_ws->BI[j + j * lda]) /
                     fmaxf(smlnum, fmaxf(fabsf(g_ws->beta[j]),
                          fabsf(g_ws->BI[j + j * lda])))) / ulp;
            if (j < mplusn - 1) {
                if (g_ws->AI[(j + 1) + j * lda] != 0.0f) {
                    ilabad = 1;
                    result[4] = ulpinv;
                }
            }
            if (j > 0) {
                if (g_ws->AI[j + (j - 1) * lda] != 0.0f) {
                    ilabad = 1;
                    result[4] = ulpinv;
                }
            }
        } else {
            int i1;
            if (g_ws->alphai[j] > 0.0f) {
                i1 = j;
            } else {
                i1 = j - 1;
            }
            if (i1 < 0 || i1 >= mplusn) {
                ilabad = 1;
            } else if (i1 < mplusn - 2) {
                if (g_ws->AI[(i1 + 2) + (i1 + 1) * lda] != 0.0f) {
                    ilabad = 1;
                    result[4] = ulpinv;
                }
            } else if (i1 > 0) {
                if (g_ws->AI[i1 + (i1 - 1) * lda] != 0.0f) {
                    ilabad = 1;
                    result[4] = ulpinv;
                }
            }
            if (!ilabad) {
                int iinfo;
                sget53(&g_ws->AI[i1 + i1 * lda], lda,
                       &g_ws->BI[i1 + i1 * lda], lda,
                       g_ws->beta[j], g_ws->alphar[j], g_ws->alphai[j],
                       &temp2, &iinfo);
                if (iinfo >= 3) {
                    print_message("SGET53 returned INFO=%d for eigenvalue %d "
                                  "(read-in #%d)\n", iinfo, j + 1, ci + 1);
                }
            } else {
                temp2 = ulpinv;
            }
        }
        temp1 = fmaxf(temp1, temp2);
    }
    result[5] = temp1;

    /* Test (7): if sorting worked */
    result[6] = 0.0f;
    if (linfo == mplusn + 3)
        result[6] = ulpinv;

    /* Test (8): DIF accuracy vs exact (compare difest[1] with provided diftru) */
    f32 diftru = readin_diftru[ci];
    result[7] = 0.0f;
    if (difest[1] == 0.0f) {
        if (diftru > abnrm * ulp)
            result[7] = ulpinv;
    } else if (diftru == 0.0f) {
        if (difest[1] > abnrm * ulp)
            result[7] = ulpinv;
    } else if ((diftru > thrsh2 * difest[1]) ||
               (diftru * thrsh2 < difest[1])) {
        result[7] = fmaxf(diftru / difest[1], difest[1] / diftru);
    }

    /* Test (9): reordering failure */
    result[8] = 0.0f;
    if (linfo == (mplusn + 2)) {
        if (diftru > abnrm * ulp)
            result[8] = ulpinv;
        if (difest[1] != 0.0f)
            result[8] = ulpinv;
        if (pl[0] != 0.0f)
            result[8] = ulpinv;
    }

    /* Test (10): PL accuracy vs exact */
    f32 pltru = readin_pltru[ci];
    result[9] = 0.0f;
    if (pl[0] == 0.0f) {
        if (pltru > abnrm * ulp)
            result[9] = ulpinv;
    } else if (pltru == 0.0f) {
        if (pl[0] > abnrm * ulp)
            result[9] = ulpinv;
    } else if ((pltru > THRESH * pl[0]) ||
               (pltru * THRESH < pl[0])) {
        result[9] = ulpinv;
    }

    /* Check all 10 results against thresholds */
    int any_fail = 0;
    for (int j = 0; j < 10; j++) {
        if (result[j] >= THRESH) {
            print_message("read-in #%d test(%d)=%g >= %.1f\n",
                          ci + 1, j + 1, (double)result[j], (double)THRESH);
            any_fail = 1;
        }
    }
    assert_int_equal(any_fail, 0);
}

int main(void)
{
    /* Count built-in test cases */
    int builtin_count = 0;
    for (int mm = 1; mm <= NSIZE - 1; mm++)
        for (int nn = 1; nn <= NSIZE - mm; nn++)
            builtin_count++;
    builtin_count *= 4 * 5;  /* ifunc x prtype */

    static ddrgsx_params_t all_params[4 * 5 * NSIZE * NSIZE];
    static ddrgsx_readin_params_t readin_params[NREADIN];
    static struct CMUnitTest all_tests[4 * 5 * NSIZE * NSIZE + NREADIN];
    int idx = 0;

    static const char* sense_names[] = {"N", "E", "V", "B"};

    /* Section 1: Built-in tests via slatm5 */
    for (int ifunc = 0; ifunc <= 3; ifunc++) {
        for (int prtype = 1; prtype <= 5; prtype++) {
            for (int m = 1; m <= NSIZE - 1; m++) {
                for (int n = 1; n <= NSIZE - m; n++) {
                    ddrgsx_params_t* p = &all_params[idx];
                    p->ifunc = ifunc;
                    p->prtype = prtype;
                    p->m = m;
                    p->n = n;
                    snprintf(p->name, sizeof(p->name),
                             "sense=%s_type=%d_m=%d_n=%d",
                             sense_names[ifunc], prtype, m, n);

                    all_tests[idx].name = p->name;
                    all_tests[idx].test_func = test_ddrgsx;
                    all_tests[idx].setup_func = NULL;
                    all_tests[idx].teardown_func = NULL;
                    all_tests[idx].initial_state = p;
                    idx++;
                }
            }
        }
    }

    /* Section 2: Read-in precomputed test matrices (2 cases from dgd.in) */
    for (int ci = 0; ci < NREADIN; ci++) {
        ddrgsx_readin_params_t* rp = &readin_params[ci];
        rp->mplusn = 4;
        rp->n = 2;
        rp->idx = ci;
        snprintf(rp->name, sizeof(rp->name), "readin_%d", ci + 1);

        all_tests[idx].name = rp->name;
        all_tests[idx].test_func = test_ddrgsx_readin;
        all_tests[idx].setup_func = NULL;
        all_tests[idx].teardown_func = NULL;
        all_tests[idx].initial_state = rp;
        idx++;
    }

    return cmocka_run_group_tests_name("ddrgsx", all_tests, group_setup,
                                       group_teardown);
}
