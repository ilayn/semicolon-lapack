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
#include "testutils/verify.h"
#include <cblas.h>
#include <math.h>
#include <string.h>

#define THRESH 10.0
#define NSIZE 2

/* Shared state for dlctsx callback (replaces Fortran COMMON /MN/) */
static int g_sel_m, g_sel_n, g_sel_mplusn, g_sel_k;
static int g_sel_fs;

static int dlctsx(const f64* ar, const f64* ai, const f64* beta)
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

extern void dggesx(const char* jobvsl, const char* jobvsr, const char* sort,
                   int (*selctg)(const f64*, const f64*, const f64*),
                   const char* sense, const int n,
                   f64* A, const int lda, f64* B, const int ldb,
                   int* sdim, f64* alphar, f64* alphai, f64* beta,
                   f64* VSL, const int ldvsl, f64* VSR, const int ldvsr,
                   f64* rconde, f64* rcondv,
                   f64* work, const int lwork,
                   int* iwork, const int liwork, int* bwork, int* info);
extern void dgesvd(const char* jobu, const char* jobvt,
                   const int m, const int n, f64* A, const int lda,
                   f64* S, f64* U, const int ldu, f64* VT, const int ldvt,
                   f64* work, const int lwork, int* info);
extern f64 dlamch(const char* cmach);
extern f64 dlange(const char* norm, const int m, const int n,
                  const f64* A, const int lda, f64* work);
extern void dlacpy(const char* uplo, const int m, const int n,
                   const f64* A, const int lda, f64* B, const int ldb);
extern void dlaset(const char* uplo, const int m, const int n,
                   const f64 alpha, const f64 beta_val, f64* A, const int lda);

typedef struct {
    int ifunc;
    int prtype;
    int m;
    int n;
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

    const int ns = NSIZE;
    const int n2 = ns * ns;

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
    int minwrk = 10 * (ns + 1);
    int bdspac = 5 * n2 / 2;
    if (bdspac > minwrk) minwrk = bdspac;
    g_ws->lwork = minwrk + n2;  /* extra margin */
    g_ws->work  = malloc(g_ws->lwork * sizeof(f64));
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
    const int lda = NSIZE;

    const f64 ulp = dlamch("P");
    const f64 ulpinv = 1.0 / ulp;
    const f64 smlnum = dlamch("S") / ulp;
    const f64 thrsh2 = 10.0 * THRESH;

    f64 result[10];
    for (int i = 0; i < 10; i++) result[i] = 0.0;

    /* Weight oscillates: sqrt(ulp), 1/sqrt(ulp), sqrt(ulp), ... */
    /* In Fortran: WEIGHT = SQRT(ULP), then 1/WEIGHT each iteration.
       We compute it deterministically from the parameter indices. */
    static f64 weight_val = 0.0;
    if (weight_val == 0.0) weight_val = sqrt(ulp);
    f64 weight = 1.0 / weight_val;
    weight_val = weight;

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

    int mm, linfo;
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
           g_ws->iwork, NSIZE + 6, g_ws->bwork, &linfo);

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
                        g_ws->work);

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

    for (int j = 0; j < mplusn; j++) {
        int ilabad = 0;
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
            int i1;
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
                int iinfo;
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
    int mn2 = mm * (mplusn - mm) * 2;
    int ncmax = NSIZE * NSIZE;
    if (ifunc >= 2 && mn2 <= ncmax) {
        dlakf2(mm, mplusn - mm,
               g_ws->AI, lda,
               &g_ws->AI[mm + mm * lda],
               g_ws->BI,
               &g_ws->BI[mm + mm * lda],
               g_ws->C, g_ws->ldc);

        int svd_info;
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
    int any_fail = 0;
    for (int j = 0; j < 9; j++) {
        if (result[j] >= THRESH) {
            print_message("sense=%s type=%d m=%d n=%d test(%d)=%g\n",
                          sense, prtype, m, n, j + 1, result[j]);
            any_fail = 1;
        }
    }
    assert_int_equal(any_fail, 0);
}

int main(void)
{
    static ddrgsx_params_t all_params[4 * 5 * NSIZE * NSIZE];
    static struct CMUnitTest all_tests[4 * 5 * NSIZE * NSIZE];
    int idx = 0;

    static const char* sense_names[] = {"N", "E", "V", "B"};

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

    return cmocka_run_group_tests_name("ddrgsx", all_tests, group_setup,
                                       group_teardown);
}
