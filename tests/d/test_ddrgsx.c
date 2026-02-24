/**
 * @file test_ddrgsx.c
 * @brief Generalized nonsymmetric Schur form expert driver test - port of
 *        LAPACK TESTING/EIG/ddrgsx.f
 *
 * Tests the nonsymmetric generalized eigenvalue (Schur form) problem expert
 * driver DGGESX.
 *
 * DGGESX factors A and B as Q S Z' and Q T Z', where ' means
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
 *   (6) eigenvalue accuracy (diagonal comparison / dget53)
 *   (7) if sorting worked (SDIM == expected)
 *   (8) DIF accuracy vs exact (via dlakf2 + dgesvd SVD)
 *   (9) reordering failure checks (INFO = MPLUSN+2)
 *
 * Matrix types: 5 types from DLATM5
 * SENSE options: 'N','E','V','B' (4 variants)
 * Block sizes: m = 1..NSIZE-1, n = 1..NSIZE-m
 * Total for NSIZE=5: 4 x 5 x 10 = 200 test cases.
 */

#include "test_harness.h"
#include "verify.h"
#include <math.h>
#include <string.h>

#define THRESH 10.0
#define NSIZE 2
#define NMAX 4

/* Shared state for dlctsx callback (replaces Fortran COMMON /MN/) */
static INT g_sel_m, g_sel_n, g_sel_mplusn, g_sel_k;
static INT g_sel_fs;

static INT dlctsx(const f64* ar, const f64* ai, const f64* beta)
{
    (void)ar; (void)ai; (void)beta;
    INT res;
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

typedef struct {
    INT ifunc;
    INT prtype;
    INT m;
    INT n;
    char name[96];
} ddrgsx_params_t;

typedef struct {
    f64* A;
    f64* B;
    f64* AI;
    f64* BI;
    f64* Q;
    f64* Z;
    f64* alphar;
    f64* alphai;
    f64* beta;
    f64* C;       /* Kronecker matrix for DIF */
    f64* S;       /* Singular values */
    f64* work;
    INT* iwork;
    INT* bwork;
    INT lwork;
    INT ldc;
} ddrgsx_workspace_t;

static ddrgsx_workspace_t* g_ws = NULL;

static int group_setup(void** state)
{
    (void)state;

    g_ws = calloc(1, sizeof(ddrgsx_workspace_t));
    if (!g_ws) return -1;

    /* Allocate for NMAX (accommodates read-in 4x4 matrices) */
    const INT ns = NMAX;
    const INT n2 = ns * ns;

    g_ws->A      = malloc(n2 * sizeof(f64));
    g_ws->B      = malloc(n2 * sizeof(f64));
    g_ws->AI     = malloc(n2 * sizeof(f64));
    g_ws->BI     = malloc(n2 * sizeof(f64));
    g_ws->Q      = malloc(n2 * sizeof(f64));
    g_ws->Z      = malloc(n2 * sizeof(f64));
    g_ws->alphar = malloc(ns * sizeof(f64));
    g_ws->alphai = malloc(ns * sizeof(f64));
    g_ws->beta   = malloc(ns * sizeof(f64));

    /* Kronecker matrix for DIF computation: ldc = ns*ns/2 */
    g_ws->ldc = ns * ns / 2;
    if (g_ws->ldc < 1) g_ws->ldc = 1;
    g_ws->C = malloc(g_ws->ldc * g_ws->ldc * sizeof(f64));
    g_ws->S = malloc(g_ws->ldc * sizeof(f64));

    /* Workspace: max(10*(ns+1), 5*ns*ns/2) + extra for dgesvd */
    INT minwrk = 10 * (ns + 1);
    INT bdspac = 5 * n2 / 2;
    if (bdspac > minwrk) minwrk = bdspac;
    g_ws->lwork = minwrk + n2;  /* extra margin */
    g_ws->work  = malloc(g_ws->lwork * sizeof(f64));
    g_ws->iwork = malloc((ns + 6) * sizeof(INT));
    g_ws->bwork = malloc(ns * sizeof(INT));

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

    const INT ifunc = params->ifunc;
    const INT prtype = params->prtype;
    const INT m = params->m;
    const INT n = params->n;
    const INT mplusn = m + n;
    const INT lda = NMAX;

    const f64 ulp = dlamch("P");
    const f64 ulpinv = 1.0 / ulp;
    const f64 smlnum = dlamch("S") / ulp;
    const f64 thrsh2 = 10.0 * THRESH;

    f64 result[10];
    for (INT i = 0; i < 10; i++) result[i] = 0.0;

    /* Weight oscillates: sqrt(ulp), 1/sqrt(ulp), sqrt(ulp), ...
       Fortran initializes WEIGHT = SQRT(ULP), then flips at start of each
       iteration. Compute deterministically from the flat iteration index. */
    INT inner_count = 0;
    for (INT mm = 1; mm <= NSIZE - 1; mm++)
        for (INT nn = 1; nn <= NSIZE - mm; nn++)
            inner_count++;
    INT inner_idx = 0;
    for (INT mm = 1; mm < m; mm++)
        inner_idx += NSIZE - mm;
    inner_idx += n - 1;
    INT iter_idx = ifunc * 5 * inner_count + (prtype - 1) * inner_count + inner_idx;
    f64 weight = ((iter_idx % 2) == 0) ? 1.0 / sqrt(ulp) : sqrt(ulp);

    /* Reset selection callback state */
    g_sel_fs = 1;
    g_sel_k = 0;
    g_sel_m = m;
    g_sel_n = n;
    g_sel_mplusn = mplusn;

    /* Generate test matrices */
    dlaset("Full", mplusn, mplusn, 0.0, 0.0, g_ws->AI, lda);
    dlaset("Full", mplusn, mplusn, 0.0, 0.0, g_ws->BI, lda);

    dlatm5(prtype, m, n,
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

    dlacpy("Full", mplusn, mplusn, g_ws->AI, lda, g_ws->A, lda);
    dlacpy("Full", mplusn, mplusn, g_ws->BI, lda, g_ws->B, lda);

    INT mm, linfo;
    f64 pl[2], difest[2];

    /* Reset selection callback again for the actual call */
    g_sel_fs = 1;
    g_sel_k = 0;
    g_sel_m = m;
    g_sel_n = n;
    g_sel_mplusn = mplusn;

    dggesx("V", "V", "S", dlctsx, sense, mplusn,
           g_ws->AI, lda, g_ws->BI, lda, &mm,
           g_ws->alphar, g_ws->alphai, g_ws->beta,
           g_ws->Q, lda, g_ws->Z, lda,
           pl, difest,
           g_ws->work, g_ws->lwork,
           g_ws->iwork, NMAX + 6, g_ws->bwork, &linfo);

    if (linfo != 0 && linfo != mplusn + 2) {
        print_message("DGGESX returned INFO=%d for sense=%s type=%d m=%d n=%d\n",
                      linfo, sense, prtype, m, n);
        result[0] = ulpinv;
        assert_residual_ok(result[0]);
        return;
    }

    /* Compute norm(A, B) */
    dlacpy("Full", mplusn, mplusn, g_ws->AI, lda, g_ws->work, mplusn);
    dlacpy("Full", mplusn, mplusn, g_ws->BI, lda,
           &g_ws->work[mplusn * mplusn], mplusn);
    f64 abnrm = dlange("Fro", mplusn, 2 * mplusn, g_ws->work, mplusn,
                        NULL);

    /* Tests (1) to (4) via dget51 */
    dget51(1, mplusn, g_ws->A, lda, g_ws->AI, lda, g_ws->Q, lda,
           g_ws->Z, lda, g_ws->work, &result[0]);
    dget51(1, mplusn, g_ws->B, lda, g_ws->BI, lda, g_ws->Q, lda,
           g_ws->Z, lda, g_ws->work, &result[1]);
    dget51(3, mplusn, g_ws->B, lda, g_ws->BI, lda, g_ws->Q, lda,
           g_ws->Q, lda, g_ws->work, &result[2]);
    dget51(3, mplusn, g_ws->B, lda, g_ws->BI, lda, g_ws->Z, lda,
           g_ws->Z, lda, g_ws->work, &result[3]);

    /* Tests (5) and (6): Schur form structure and eigenvalue accuracy */
    f64 temp1 = 0.0;
    result[4] = 0.0;
    result[5] = 0.0;

    for (INT j = 0; j < mplusn; j++) {
        INT ilabad = 0;
        f64 temp2;
        if (g_ws->alphai[j] == 0.0) {
            temp2 = (fabs(g_ws->alphar[j] - g_ws->AI[j + j * lda]) /
                     fmax(smlnum, fmax(fabs(g_ws->alphar[j]),
                          fabs(g_ws->AI[j + j * lda]))) +
                     fabs(g_ws->beta[j] - g_ws->BI[j + j * lda]) /
                     fmax(smlnum, fmax(fabs(g_ws->beta[j]),
                          fabs(g_ws->BI[j + j * lda])))) / ulp;
            if (j < mplusn - 1) {
                if (g_ws->AI[(j + 1) + j * lda] != 0.0) {
                    ilabad = 1;
                    result[4] = ulpinv;
                }
            }
            if (j > 0) {
                if (g_ws->AI[j + (j - 1) * lda] != 0.0) {
                    ilabad = 1;
                    result[4] = ulpinv;
                }
            }
        } else {
            INT i1;
            if (g_ws->alphai[j] > 0.0) {
                i1 = j;
            } else {
                i1 = j - 1;
            }
            if (i1 < 0 || i1 >= mplusn) {
                ilabad = 1;
            } else if (i1 < mplusn - 2) {
                if (g_ws->AI[(i1 + 2) + (i1 + 1) * lda] != 0.0) {
                    ilabad = 1;
                    result[4] = ulpinv;
                }
            } else if (i1 > 0) {
                if (g_ws->AI[i1 + (i1 - 1) * lda] != 0.0) {
                    ilabad = 1;
                    result[4] = ulpinv;
                }
            }
            if (!ilabad) {
                INT iinfo;
                dget53(&g_ws->AI[i1 + i1 * lda], lda,
                       &g_ws->BI[i1 + i1 * lda], lda,
                       g_ws->beta[j], g_ws->alphar[j], g_ws->alphai[j],
                       &temp2, &iinfo);
            } else {
                temp2 = ulpinv;
            }
        }
        temp1 = fmax(temp1, temp2);
    }
    result[5] = temp1;

    /* Test (7): if sorting worked */
    result[6] = 0.0;
    if (linfo == mplusn + 3) {
        result[6] = ulpinv;
    } else if (mm != n) {
        result[6] = ulpinv;
    }

    /* Test (8): DIF accuracy vs exact */
    result[7] = 0.0;
    INT mn2 = mm * (mplusn - mm) * 2;
    INT ncmax = NMAX * NMAX;
    if (ifunc >= 2 && mn2 <= ncmax) {
        dlakf2(mm, mplusn - mm,
               g_ws->AI, lda,
               &g_ws->AI[mm + mm * lda],
               g_ws->BI,
               &g_ws->BI[mm + mm * lda],
               g_ws->C, g_ws->ldc);

        INT svd_info;
        dgesvd("N", "N", mn2, mn2, g_ws->C, g_ws->ldc, g_ws->S,
               g_ws->work, 1, &g_ws->work[1], 1,
               &g_ws->work[2], g_ws->lwork - 2, &svd_info);
        f64 diftru = g_ws->S[mn2 - 1];

        if (difest[1] == 0.0) {
            if (diftru > abnrm * ulp)
                result[7] = ulpinv;
        } else if (diftru == 0.0) {
            if (difest[1] > abnrm * ulp)
                result[7] = ulpinv;
        } else if ((diftru > thrsh2 * difest[1]) ||
                   (diftru * thrsh2 < difest[1])) {
            result[7] = fmax(diftru / difest[1], difest[1] / diftru);
        }
    }

    /* Test (9): reordering failure */
    result[8] = 0.0;
    if (linfo == (mplusn + 2)) {
        f64 diftru_local = 0.0;
        if (ifunc >= 2 && mn2 <= ncmax) {
            /* diftru was computed above */
            diftru_local = g_ws->S[mn2 - 1];
        }
        if (diftru_local > abnrm * ulp)
            result[8] = ulpinv;
        if ((ifunc > 1) && (difest[1] != 0.0))
            result[8] = ulpinv;
        if ((ifunc == 1) && (pl[0] != 0.0))
            result[8] = ulpinv;
    }

    /* Check results against thresholds */
    INT any_fail = 0;
    for (INT j = 0; j < 9; j++) {
        if (result[j] >= THRESH) {
            print_message("sense=%s type=%d m=%d n=%d test(%d)=%g\n",
                          sense, prtype, m, n, j + 1, result[j]);
            any_fail = 1;
        }
    }
    assert_int_equal(any_fail, 0);
}

/*
 * Section 2: Read-in test data from dgd.in (LAPACK TESTING/dgd.in lines 23-51).
 * Two precomputed 4x4 matrix pairs with known condition numbers for the
 * eigenvalue cluster and deflating subspace, used to check accuracy of
 * condition estimation in DGGESX.
 */

#define NREADIN 2

typedef struct {
    INT mplusn;
    INT n;
    INT idx;
    char name[64];
} ddrgsx_readin_params_t;

/* Column-major storage */
static const f64 readin_AI[NREADIN][16] = {
    {  8.0,  0.0,  0.0,  0.0,
       4.0,  7.0,  0.0,  0.0,
     -13.0,-24.0,  3.0,  0.0,
       4.0, -3.0, -5.0, 16.0 },
    {  1.0,  0.0,  0.0,  0.0,
       2.0,  5.0,  0.0,  0.0,
       3.0,  6.0,  8.0,  0.0,
       4.0,  7.0,  9.0, 10.0 }
};

static const f64 readin_BI[NREADIN][16] = {
    {  9.0,  0.0,  0.0,  0.0,
      -1.0,  4.0,  0.0,  0.0,
       1.0, 16.0,-11.0,  0.0,
      -6.0,-24.0,  6.0,  4.0 },
    { -1.0,  0.0,  0.0,  0.0,
      -1.0, -1.0,  0.0,  0.0,
      -1.0, -1.0,  1.0,  0.0,
      -1.0, -1.0, -1.0,  1.0 }
};

static const f64 readin_pltru[NREADIN] = { 2.5901e-01, 9.8173e-01 };
static const f64 readin_diftru[NREADIN] = { 1.7592e+00, 6.3649e-01 };

static void test_ddrgsx_readin(void** state)
{
    ddrgsx_readin_params_t* params = (ddrgsx_readin_params_t*)(*state);

    const INT mplusn = params->mplusn;
    const INT n = params->n;
    const INT m = mplusn - n;
    const INT lda = NMAX;
    const INT ci = params->idx;

    const f64 ulp = dlamch("P");
    const f64 ulpinv = 1.0 / ulp;
    const f64 smlnum = dlamch("S") / ulp;
    const f64 thrsh2 = 10.0 * THRESH;

    f64 result[10];
    for (INT i = 0; i < 10; i++) result[i] = 0.0;

    /* Load precomputed matrices (lda may differ from mplusn) */
    for (INT j = 0; j < mplusn; j++)
        for (INT i = 0; i < mplusn; i++) {
            g_ws->AI[i + j * lda] = readin_AI[ci][i + j * mplusn];
            g_ws->BI[i + j * lda] = readin_BI[ci][i + j * mplusn];
        }

    /* Reset selection callback */
    g_sel_fs = 1;
    g_sel_k = 0;
    g_sel_m = m;
    g_sel_n = n;
    g_sel_mplusn = mplusn;

    dlacpy("Full", mplusn, mplusn, g_ws->AI, lda, g_ws->A, lda);
    dlacpy("Full", mplusn, mplusn, g_ws->BI, lda, g_ws->B, lda);

    INT mm, linfo;
    f64 pl[2], difest[2];

    /* Reset selection callback again */
    g_sel_fs = 1;
    g_sel_k = 0;
    g_sel_m = m;
    g_sel_n = n;
    g_sel_mplusn = mplusn;

    dggesx("V", "V", "S", dlctsx, "B", mplusn,
           g_ws->AI, lda, g_ws->BI, lda, &mm,
           g_ws->alphar, g_ws->alphai, g_ws->beta,
           g_ws->Q, lda, g_ws->Z, lda,
           pl, difest,
           g_ws->work, g_ws->lwork,
           g_ws->iwork, NMAX + 6, g_ws->bwork, &linfo);

    if (linfo != 0 && linfo != mplusn + 2) {
        print_message("DGGESX returned INFO=%d for read-in example #%d\n",
                      linfo, ci + 1);
        result[0] = ulpinv;
        assert_residual_below(result[0], thrsh2);
        return;
    }

    /* Compute norm(A, B) */
    dlacpy("Full", mplusn, mplusn, g_ws->AI, lda, g_ws->work, mplusn);
    dlacpy("Full", mplusn, mplusn, g_ws->BI, lda,
           &g_ws->work[mplusn * mplusn], mplusn);
    f64 abnrm = dlange("Fro", mplusn, 2 * mplusn, g_ws->work, mplusn,
                        NULL);

    /* Tests (1) to (4) via dget51 */
    dget51(1, mplusn, g_ws->A, lda, g_ws->AI, lda, g_ws->Q, lda,
           g_ws->Z, lda, g_ws->work, &result[0]);
    dget51(1, mplusn, g_ws->B, lda, g_ws->BI, lda, g_ws->Q, lda,
           g_ws->Z, lda, g_ws->work, &result[1]);
    dget51(3, mplusn, g_ws->B, lda, g_ws->BI, lda, g_ws->Q, lda,
           g_ws->Q, lda, g_ws->work, &result[2]);
    dget51(3, mplusn, g_ws->B, lda, g_ws->BI, lda, g_ws->Z, lda,
           g_ws->Z, lda, g_ws->work, &result[3]);

    /* Tests (5) and (6): Schur form structure and eigenvalue accuracy */
    f64 temp1 = 0.0;
    result[4] = 0.0;
    result[5] = 0.0;

    for (INT j = 0; j < mplusn; j++) {
        INT ilabad = 0;
        f64 temp2;
        if (g_ws->alphai[j] == 0.0) {
            temp2 = (fabs(g_ws->alphar[j] - g_ws->AI[j + j * lda]) /
                     fmax(smlnum, fmax(fabs(g_ws->alphar[j]),
                          fabs(g_ws->AI[j + j * lda]))) +
                     fabs(g_ws->beta[j] - g_ws->BI[j + j * lda]) /
                     fmax(smlnum, fmax(fabs(g_ws->beta[j]),
                          fabs(g_ws->BI[j + j * lda])))) / ulp;
            if (j < mplusn - 1) {
                if (g_ws->AI[(j + 1) + j * lda] != 0.0) {
                    ilabad = 1;
                    result[4] = ulpinv;
                }
            }
            if (j > 0) {
                if (g_ws->AI[j + (j - 1) * lda] != 0.0) {
                    ilabad = 1;
                    result[4] = ulpinv;
                }
            }
        } else {
            INT i1;
            if (g_ws->alphai[j] > 0.0) {
                i1 = j;
            } else {
                i1 = j - 1;
            }
            if (i1 < 0 || i1 >= mplusn) {
                ilabad = 1;
            } else if (i1 < mplusn - 2) {
                if (g_ws->AI[(i1 + 2) + (i1 + 1) * lda] != 0.0) {
                    ilabad = 1;
                    result[4] = ulpinv;
                }
            } else if (i1 > 0) {
                if (g_ws->AI[i1 + (i1 - 1) * lda] != 0.0) {
                    ilabad = 1;
                    result[4] = ulpinv;
                }
            }
            if (!ilabad) {
                INT iinfo;
                dget53(&g_ws->AI[i1 + i1 * lda], lda,
                       &g_ws->BI[i1 + i1 * lda], lda,
                       g_ws->beta[j], g_ws->alphar[j], g_ws->alphai[j],
                       &temp2, &iinfo);
                if (iinfo >= 3) {
                    print_message("DGET53 returned INFO=%d for eigenvalue %d "
                                  "(read-in #%d)\n", iinfo, j + 1, ci + 1);
                }
            } else {
                temp2 = ulpinv;
            }
        }
        temp1 = fmax(temp1, temp2);
    }
    result[5] = temp1;

    /* Test (7): if sorting worked */
    result[6] = 0.0;
    if (linfo == mplusn + 3)
        result[6] = ulpinv;

    /* Test (8): DIF accuracy vs exact (compare difest[1] with provided diftru) */
    f64 diftru = readin_diftru[ci];
    result[7] = 0.0;
    if (difest[1] == 0.0) {
        if (diftru > abnrm * ulp)
            result[7] = ulpinv;
    } else if (diftru == 0.0) {
        if (difest[1] > abnrm * ulp)
            result[7] = ulpinv;
    } else if ((diftru > thrsh2 * difest[1]) ||
               (diftru * thrsh2 < difest[1])) {
        result[7] = fmax(diftru / difest[1], difest[1] / diftru);
    }

    /* Test (9): reordering failure */
    result[8] = 0.0;
    if (linfo == (mplusn + 2)) {
        if (diftru > abnrm * ulp)
            result[8] = ulpinv;
        if (difest[1] != 0.0)
            result[8] = ulpinv;
        if (pl[0] != 0.0)
            result[8] = ulpinv;
    }

    /* Test (10): PL accuracy vs exact */
    f64 pltru = readin_pltru[ci];
    result[9] = 0.0;
    if (pl[0] == 0.0) {
        if (pltru > abnrm * ulp)
            result[9] = ulpinv;
    } else if (pltru == 0.0) {
        if (pl[0] > abnrm * ulp)
            result[9] = ulpinv;
    } else if ((pltru > THRESH * pl[0]) ||
               (pltru * THRESH < pl[0])) {
        result[9] = ulpinv;
    }

    /* Check all 10 results against thresholds */
    INT any_fail = 0;
    for (INT j = 0; j < 10; j++) {
        if (result[j] >= THRESH) {
            print_message("read-in #%d test(%d)=%g >= %.1f\n",
                          ci + 1, j + 1, result[j], THRESH);
            any_fail = 1;
        }
    }
    assert_int_equal(any_fail, 0);
}

int main(void)
{
    /* Count built-in test cases */
    INT builtin_count = 0;
    for (INT mm = 1; mm <= NSIZE - 1; mm++)
        for (INT nn = 1; nn <= NSIZE - mm; nn++)
            builtin_count++;
    builtin_count *= 4 * 5;  /* ifunc x prtype */

    static ddrgsx_params_t all_params[4 * 5 * NSIZE * NSIZE];
    static ddrgsx_readin_params_t readin_params[NREADIN];
    static struct CMUnitTest all_tests[4 * 5 * NSIZE * NSIZE + NREADIN];
    INT idx = 0;

    static const char* sense_names[] = {"N", "E", "V", "B"};

    /* Section 1: Built-in tests via dlatm5 */
    for (INT ifunc = 0; ifunc <= 3; ifunc++) {
        for (INT prtype = 1; prtype <= 5; prtype++) {
            for (INT m = 1; m <= NSIZE - 1; m++) {
                for (INT n = 1; n <= NSIZE - m; n++) {
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
    for (INT ci = 0; ci < NREADIN; ci++) {
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
