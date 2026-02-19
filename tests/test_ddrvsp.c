/**
 * @file test_ddrvsp.c
 * @brief DDRVSP tests the driver routines DSPSV and DSPSVX.
 *
 * Port of LAPACK TESTING/LIN/ddrvsp.f to C with CMocka parameterization.
 */

#include "test_harness.h"
#include "test_rng.h"
#include <string.h>
#include <stdio.h>
#include <cblas.h>
#include <math.h>

/* Test parameters - matching LAPACK dchkaa.f defaults */
static const int NVAL[] = {0, 1, 2, 3, 5, 10, 50};
#define NN      (sizeof(NVAL) / sizeof(NVAL[0]))
#define NTYPES  10
#define NTESTS  6
#define NFACT   2
#define THRESH  30.0
#define NMAX    50
#define NRHS    2
#define NPP_MAX (NMAX * (NMAX + 1) / 2)

/* Routines under test */
extern void dspsv(const char* uplo, const int n, const int nrhs,
                  f64* AP, int* ipiv, f64* B, const int ldb, int* info);
extern void dspsvx(const char* fact, const char* uplo, const int n, const int nrhs,
                   const f64* AP, f64* AFP, int* ipiv,
                   const f64* B, const int ldb, f64* X, const int ldx,
                   f64* rcond, f64* ferr, f64* berr,
                   f64* work, int* iwork, int* info);

/* Supporting routines */
extern void dsptrf(const char* uplo, const int n, f64* AP, int* ipiv, int* info);
extern void dsptri(const char* uplo, const int n, f64* AP, const int* ipiv,
                   f64* work, int* info);

/* Verification routines */
extern void dspt01(const char* uplo, const int n, const f64* A,
                   const f64* AFAC, const int* ipiv, f64* C, const int ldc,
                   f64* rwork, f64* resid);
extern void dppt02(const char* uplo, const int n, const int nrhs,
                   const f64* A, const f64* X, const int ldx,
                   f64* B, const int ldb, f64* rwork, f64* resid);
extern void dppt05(const char* uplo, const int n, const int nrhs,
                   const f64* AP, const f64* B, const int ldb,
                   const f64* X, const int ldx, const f64* XACT, const int ldxact,
                   const f64* FERR, const f64* BERR, f64* reslts);
extern void dget04(const int n, const int nrhs, const f64* X, const int ldx,
                   const f64* XACT, const int ldxact, const f64 rcond,
                   f64* resid);
extern f64 dget06(const f64 rcond, const f64 rcondc);

/* Matrix generation */
extern void dlatb4(const char* path, const int imat, const int m, const int n,
                   char* type, int* kl, int* ku, f64* anorm, int* mode,
                   f64* cndnum, char* dist);
extern void dlatms(const int m, const int n, const char* dist,
                   const char* sym, f64* d,
                   const int mode, const f64 cond, const f64 dmax,
                   const int kl, const int ku, const char* pack,
                   f64* A, const int lda, f64* work, int* info,
                   uint64_t state[static 4]);
extern void dlarhs(const char* path, const char* xtype, const char* uplo,
                   const char* trans, const int m, const int n,
                   const int kl, const int ku, const int nrhs,
                   const f64* A, const int lda, f64* XACT, const int ldxact,
                   f64* B, const int ldb, int* info, uint64_t state[static 4]);

/* Utilities */
extern void dlacpy(const char* uplo, const int m, const int n,
                   const f64* A, const int lda, f64* B, const int ldb);
extern void dlaset(const char* uplo, const int m, const int n,
                   const f64 alpha, const f64 beta,
                   f64* A, const int lda);
extern f64 dlansp(const char* norm, const char* uplo, const int n,
                  const f64* AP, f64* work);

typedef struct {
    int n;
    int imat;
    int iuplo;      /* 0='U', 1='L' */
    int ifact;      /* 0='F', 1='N' */
    char name[64];
} ddrvsp_params_t;

typedef struct {
    f64* A;
    f64* AFAC;
    f64* AINV;
    f64* B;
    f64* X;
    f64* XACT;
    f64* WORK;
    f64* RWORK;
    int* IWORK;
    int lwork;
} ddrvsp_workspace_t;

static ddrvsp_workspace_t* g_workspace = NULL;

static int group_setup(void** state)
{
    (void)state;
    g_workspace = malloc(sizeof(ddrvsp_workspace_t));
    if (!g_workspace) return -1;

    int nmax = NMAX;
    int lwork = 3 * nmax;
    if (lwork < nmax * NRHS) lwork = nmax * NRHS;

    g_workspace->lwork = lwork;
    g_workspace->A = calloc(nmax * nmax, sizeof(f64));
    g_workspace->AFAC = calloc(NPP_MAX, sizeof(f64));
    g_workspace->AINV = calloc(nmax * nmax, sizeof(f64));
    g_workspace->B = calloc(nmax * NRHS, sizeof(f64));
    g_workspace->X = calloc(nmax * NRHS, sizeof(f64));
    g_workspace->XACT = calloc(nmax * NRHS, sizeof(f64));
    g_workspace->WORK = calloc(lwork, sizeof(f64));
    g_workspace->RWORK = calloc(nmax + 2 * NRHS, sizeof(f64));
    g_workspace->IWORK = calloc(2 * nmax, sizeof(int));

    if (!g_workspace->A || !g_workspace->AFAC || !g_workspace->AINV ||
        !g_workspace->B || !g_workspace->X || !g_workspace->XACT ||
        !g_workspace->WORK || !g_workspace->RWORK || !g_workspace->IWORK) {
        return -1;
    }
    return 0;
}

static int group_teardown(void** state)
{
    (void)state;
    if (g_workspace) {
        free(g_workspace->A);
        free(g_workspace->AFAC);
        free(g_workspace->AINV);
        free(g_workspace->B);
        free(g_workspace->X);
        free(g_workspace->XACT);
        free(g_workspace->WORK);
        free(g_workspace->RWORK);
        free(g_workspace->IWORK);
        free(g_workspace);
        g_workspace = NULL;
    }
    return 0;
}

static void run_ddrvsp_single(int n, int imat, int iuplo, int ifact)
{
    static const char* UPLOS[] = {"U", "L"};
    static const char* FACTS[] = {"F", "N"};

    ddrvsp_workspace_t* ws = g_workspace;
    const char* uplo = UPLOS[iuplo];
    const char* fact = FACTS[ifact];

    int lda = (n > 1) ? n : 1;
    int npp = n * (n + 1) / 2;
    f64 result[NTESTS];
    for (int k = 0; k < NTESTS; k++) result[k] = 0.0;

    int zerot = (imat >= 3 && imat <= 6);
    int izero = 0;

    /* Packed format: 'C' for upper (Column-major packed), 'R' for lower (Row-major packed) */
    const char* packit = (iuplo == 0) ? "C" : "R";

    /* Set up parameters with DLATB4 */
    char type, dist;
    int kl, ku, mode;
    f64 anorm, cndnum;
    dlatb4("DSP", imat, n, n, &type, &kl, &ku, &anorm, &mode, &cndnum, &dist);

    /* Generate test matrix with DLATMS */
    uint64_t rng_state[4];
    rng_seed(rng_state, 1988 + n * 1000 + imat * 100 + iuplo * 10);
    int info;
    dlatms(n, n, &dist, &type, ws->RWORK, mode, cndnum,
           anorm, kl, ku, packit, ws->A, lda, ws->WORK, &info, rng_state);
    if (info != 0) {
        fail_msg("DLATMS info=%d", info);
        return;
    }

    /* For types 3-6, zero one or more rows and columns of the packed matrix */
    if (zerot) {
        if (imat == 3) {
            izero = 1;
        } else if (imat == 4) {
            izero = n;
        } else {
            izero = n / 2 + 1;
        }

        if (imat < 6) {
            /* Set row and column IZERO to zero (packed indexing).
             * Note: izero is 1-based here, matching the Fortran logic.
             * Array accesses use -1 for 0-based C arrays. */
            if (iuplo == 0) {
                /* Upper packed */
                int ioff = (izero - 1) * izero / 2;
                for (int i = 1; i <= izero - 1; i++) {
                    ws->A[ioff + i - 1] = 0.0;
                }
                ioff = ioff + izero - 1;
                for (int i = izero; i <= n; i++) {
                    ws->A[ioff] = 0.0;
                    ioff = ioff + i;
                }
            } else {
                /* Lower packed */
                int ioff = izero;
                for (int i = 1; i <= izero - 1; i++) {
                    ws->A[ioff - 1] = 0.0;
                    ioff = ioff + n - i;
                }
                ioff = ioff - izero;
                for (int i = izero; i <= n; i++) {
                    ws->A[ioff + i - 1] = 0.0;
                }
            }
        } else {
            /* IMAT = 6: zero first/last IZERO rows and columns */
            int ioff = 0;
            if (iuplo == 0) {
                for (int j = 1; j <= n; j++) {
                    int i2 = (j < izero) ? j : izero;
                    for (int i = 1; i <= i2; i++) {
                        ws->A[ioff + i - 1] = 0.0;
                    }
                    ioff = ioff + j;
                }
            } else {
                for (int j = 1; j <= n; j++) {
                    int i1 = (j > izero) ? j : izero;
                    for (int i = i1; i <= n; i++) {
                        ws->A[ioff + i - 1] = 0.0;
                    }
                    ioff = ioff + n - j;
                }
            }
        }
    }

    /* Skip FACT='F' for singular matrices */
    if (zerot && ifact == 0) {
        return;
    }

    /*
     * Compute condition number RCONDC.
     * In LAPACK's nested loops, FACT='F' computes this and FACT='N' reuses it.
     * Since CMocka tests are independent, we always compute it.
     */
    f64 rcondc = 0.0;

    if (zerot) {
        rcondc = 0.0;
    } else if (n == 0) {
        rcondc = 1.0 / cndnum;
    } else {
        f64 anrm = dlansp("1", uplo, n, ws->A, ws->RWORK);

        cblas_dcopy(npp, ws->A, 1, ws->AFAC, 1);
        dsptrf(uplo, n, ws->AFAC, ws->IWORK, &info);

        cblas_dcopy(npp, ws->AFAC, 1, ws->AINV, 1);
        dsptri(uplo, n, ws->AINV, ws->IWORK, ws->WORK, &info);
        f64 ainvnm = dlansp("1", uplo, n, ws->AINV, ws->RWORK);

        if (anrm <= 0.0 || ainvnm <= 0.0) {
            rcondc = 1.0;
        } else {
            rcondc = (1.0 / anrm) / ainvnm;
        }
    }

    /* Form exact solution and set right hand side */
    char xtype = 'N';
    dlarhs("DSP", &xtype, uplo, " ", n, n, kl, ku,
           NRHS, ws->A, lda, ws->XACT, lda, ws->B, lda, &info, rng_state);
    xtype = 'C';

    /* --- Test DSPSV --- */
    if (ifact == 1) {
        cblas_dcopy(npp, ws->A, 1, ws->AFAC, 1);
        dlacpy("Full", n, NRHS, ws->B, lda, ws->X, lda);

        dspsv(uplo, n, NRHS, ws->AFAC, ws->IWORK, ws->X, lda, &info);

        /* Adjust expected value of INFO to account for pivoting */
        int k = 0;
        if (izero > 0) {
            k = izero - 1;
            for (;;) {
                if (ws->IWORK[k] < 0) {
                    if (ws->IWORK[k] != -(k + 1)) {
                        k = -(ws->IWORK[k] + 1);
                        continue;
                    }
                    break;
                } else if (ws->IWORK[k] != k) {
                    k = ws->IWORK[k];
                    continue;
                }
                break;
            }
            k = k + 1;
        }

        /* Check error code from DSPSV */
        if (info != k) {
            fail_msg("DSPSV UPLO=%s info=%d expected=%d (n=%d imat=%d)",
                     uplo, info, k, n, imat);
        } else if (info == 0) {
            /* TEST 1: Reconstruct matrix from factors */
            dspt01(uplo, n, ws->A, ws->AFAC, ws->IWORK,
                   ws->AINV, lda, ws->RWORK, &result[0]);

            /* TEST 2: Compute residual of the computed solution */
            dlacpy("Full", n, NRHS, ws->B, lda, ws->WORK, lda);
            dppt02(uplo, n, NRHS, ws->A, ws->X, lda, ws->WORK,
                   lda, ws->RWORK, &result[1]);

            /* TEST 3: Check solution from generated exact solution */
            dget04(n, NRHS, ws->X, lda, ws->XACT, lda, rcondc, &result[2]);
            int nt = 3;

            for (int i = 0; i < nt; i++) {
                if (result[i] >= THRESH) {
                    fail_msg("DSPSV UPLO=%s test %d failed: result=%e >= thresh=%e",
                             uplo, i + 1, result[i], THRESH);
                }
            }
        }
    }

    /* --- Test DSPSVX --- */
    if (ifact == 1 && npp > 0) {
        dlaset("Full", npp, 1, 0.0, 0.0, ws->AFAC, npp);
    }
    dlaset("Full", n, NRHS, 0.0, 0.0, ws->X, lda);

    f64 rcond;
    dspsvx(fact, uplo, n, NRHS, ws->A, ws->AFAC, ws->IWORK,
           ws->B, lda, ws->X, lda, &rcond,
           ws->RWORK, &ws->RWORK[NRHS], ws->WORK,
           &ws->IWORK[n], &info);

    /* Adjust expected value of INFO to account for pivoting */
    int k = 0;
    if (izero > 0) {
        k = izero - 1;
        for (;;) {
            if (ws->IWORK[k] < 0) {
                if (ws->IWORK[k] != -(k + 1)) {
                    k = -(ws->IWORK[k] + 1);
                    continue;
                }
                break;
            } else if (ws->IWORK[k] != k) {
                k = ws->IWORK[k];
                continue;
            }
            break;
        }
        k = k + 1;
    }

    /* Check error code from DSPSVX */
    if (info != k) {
        fail_msg("DSPSVX FACT=%s UPLO=%s info=%d expected=%d (n=%d imat=%d)",
                 fact, uplo, info, k, n, imat);
        return;
    }

    int k1;
    if (info == 0) {
        if (ifact >= 1) {
            /* TEST 1: Reconstruct matrix from factors */
            dspt01(uplo, n, ws->A, ws->AFAC, ws->IWORK,
                   ws->AINV, lda, &ws->RWORK[2 * NRHS], &result[0]);
            k1 = 1;
        } else {
            k1 = 2;
        }

        /* TEST 2: Compute residual of the computed solution */
        dlacpy("Full", n, NRHS, ws->B, lda, ws->WORK, lda);
        dppt02(uplo, n, NRHS, ws->A, ws->X, lda, ws->WORK,
               lda, &ws->RWORK[2 * NRHS], &result[1]);

        /* TEST 3: Check solution from generated exact solution */
        dget04(n, NRHS, ws->X, lda, ws->XACT, lda, rcondc, &result[2]);

        /* TEST 4-5: Check error bounds from iterative refinement */
        dppt05(uplo, n, NRHS, ws->A, ws->B, lda,
               ws->X, lda, ws->XACT, lda,
               ws->RWORK, &ws->RWORK[NRHS], &result[3]);
    } else {
        k1 = 6;
    }

    /* TEST 6: Compare RCOND from DSPSVX with computed value */
    result[5] = dget06(rcond, rcondc);

    /* Check results */
    for (int i = k1 - 1; i < NTESTS; i++) {
        if (result[i] >= THRESH) {
            fail_msg("DSPSVX FACT=%s UPLO=%s test %d: result=%e >= thresh=%e",
                     fact, uplo, i + 1, result[i], THRESH);
        }
    }
}

static void test_ddrvsp_case(void** state)
{
    ddrvsp_params_t* p = *state;
    run_ddrvsp_single(p->n, p->imat, p->iuplo, p->ifact);
}

#define MAX_TESTS 3000

static ddrvsp_params_t g_params[MAX_TESTS];
static struct CMUnitTest g_tests[MAX_TESTS];
static int g_num_tests = 0;

static void build_test_array(void)
{
    static const char* UPLOS[] = {"U", "L"};
    static const char* FACTS_STR[] = {"F", "N"};

    g_num_tests = 0;

    for (int in = 0; in < (int)NN; in++) {
        int n = NVAL[in];
        int nimat = (n <= 0) ? 1 : NTYPES;

        for (int imat = 1; imat <= nimat; imat++) {
            int zerot = (imat >= 3 && imat <= 6);
            if (zerot && n < imat - 2) continue;

            for (int iuplo = 0; iuplo < 2; iuplo++) {
                for (int ifact = 0; ifact < NFACT; ifact++) {
                    if (zerot && ifact == 0) continue;

                    ddrvsp_params_t* p = &g_params[g_num_tests];
                    p->n = n;
                    p->imat = imat;
                    p->iuplo = iuplo;
                    p->ifact = ifact;
                    snprintf(p->name, sizeof(p->name),
                             "n%d_t%d_%s_%s",
                             n, imat, UPLOS[iuplo], FACTS_STR[ifact]);

                    g_tests[g_num_tests].name = p->name;
                    g_tests[g_num_tests].test_func = test_ddrvsp_case;
                    g_tests[g_num_tests].setup_func = NULL;
                    g_tests[g_num_tests].teardown_func = NULL;
                    g_tests[g_num_tests].initial_state = p;

                    g_num_tests++;
                }
            }
        }
    }
}

int main(void)
{
    build_test_array();
    return _cmocka_run_group_tests("ddrvsp", g_tests, g_num_tests,
                                   group_setup, group_teardown);
}
