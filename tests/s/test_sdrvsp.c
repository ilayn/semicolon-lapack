/**
 * @file test_sdrvsp.c
 * @brief DDRVSP tests the driver routines SSPSV and SSPSVX.
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
#define THRESH  30.0f
#define NMAX    50
#define NRHS    2
#define NPP_MAX (NMAX * (NMAX + 1) / 2)

/* Routines under test */
extern void sspsv(const char* uplo, const int n, const int nrhs,
                  f32* AP, int* ipiv, f32* B, const int ldb, int* info);
extern void sspsvx(const char* fact, const char* uplo, const int n, const int nrhs,
                   const f32* AP, f32* AFP, int* ipiv,
                   const f32* B, const int ldb, f32* X, const int ldx,
                   f32* rcond, f32* ferr, f32* berr,
                   f32* work, int* iwork, int* info);

/* Supporting routines */
extern void ssptrf(const char* uplo, const int n, f32* AP, int* ipiv, int* info);
extern void ssptri(const char* uplo, const int n, f32* AP, const int* ipiv,
                   f32* work, int* info);

/* Verification routines */
extern void sspt01(const char* uplo, const int n, const f32* A,
                   const f32* AFAC, const int* ipiv, f32* C, const int ldc,
                   f32* rwork, f32* resid);
extern void sppt02(const char* uplo, const int n, const int nrhs,
                   const f32* A, const f32* X, const int ldx,
                   f32* B, const int ldb, f32* rwork, f32* resid);
extern void sppt05(const char* uplo, const int n, const int nrhs,
                   const f32* AP, const f32* B, const int ldb,
                   const f32* X, const int ldx, const f32* XACT, const int ldxact,
                   const f32* FERR, const f32* BERR, f32* reslts);
extern void sget04(const int n, const int nrhs, const f32* X, const int ldx,
                   const f32* XACT, const int ldxact, const f32 rcond,
                   f32* resid);
extern f32 sget06(const f32 rcond, const f32 rcondc);

/* Matrix generation */
extern void slatb4(const char* path, const int imat, const int m, const int n,
                   char* type, int* kl, int* ku, f32* anorm, int* mode,
                   f32* cndnum, char* dist);
extern void slatms(const int m, const int n, const char* dist,
                   const char* sym, f32* d,
                   const int mode, const f32 cond, const f32 dmax,
                   const int kl, const int ku, const char* pack,
                   f32* A, const int lda, f32* work, int* info,
                   uint64_t state[static 4]);
extern void slarhs(const char* path, const char* xtype, const char* uplo,
                   const char* trans, const int m, const int n,
                   const int kl, const int ku, const int nrhs,
                   const f32* A, const int lda, f32* XACT, const int ldxact,
                   f32* B, const int ldb, int* info, uint64_t state[static 4]);

/* Utilities */
extern void slacpy(const char* uplo, const int m, const int n,
                   const f32* A, const int lda, f32* B, const int ldb);
extern void slaset(const char* uplo, const int m, const int n,
                   const f32 alpha, const f32 beta,
                   f32* A, const int lda);
extern f32 slansp(const char* norm, const char* uplo, const int n,
                  const f32* AP, f32* work);

typedef struct {
    int n;
    int imat;
    int iuplo;      /* 0='U', 1='L' */
    int ifact;      /* 0='F', 1='N' */
    char name[64];
} ddrvsp_params_t;

typedef struct {
    f32* A;
    f32* AFAC;
    f32* AINV;
    f32* B;
    f32* X;
    f32* XACT;
    f32* WORK;
    f32* RWORK;
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
    g_workspace->A = calloc(nmax * nmax, sizeof(f32));
    g_workspace->AFAC = calloc(NPP_MAX, sizeof(f32));
    g_workspace->AINV = calloc(nmax * nmax, sizeof(f32));
    g_workspace->B = calloc(nmax * NRHS, sizeof(f32));
    g_workspace->X = calloc(nmax * NRHS, sizeof(f32));
    g_workspace->XACT = calloc(nmax * NRHS, sizeof(f32));
    g_workspace->WORK = calloc(lwork, sizeof(f32));
    g_workspace->RWORK = calloc(nmax + 2 * NRHS, sizeof(f32));
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
    f32 result[NTESTS];
    for (int k = 0; k < NTESTS; k++) result[k] = 0.0f;

    int zerot = (imat >= 3 && imat <= 6);
    int izero = 0;

    /* Packed format: 'C' for upper (Column-major packed), 'R' for lower (Row-major packed) */
    const char* packit = (iuplo == 0) ? "C" : "R";

    /* Set up parameters with SLATB4 */
    char type, dist;
    int kl, ku, mode;
    f32 anorm, cndnum;
    slatb4("SSP", imat, n, n, &type, &kl, &ku, &anorm, &mode, &cndnum, &dist);

    /* Generate test matrix with SLATMS */
    uint64_t rng_state[4];
    rng_seed(rng_state, 1988 + n * 1000 + imat * 100 + iuplo * 10);
    int info;
    slatms(n, n, &dist, &type, ws->RWORK, mode, cndnum,
           anorm, kl, ku, packit, ws->A, lda, ws->WORK, &info, rng_state);
    if (info != 0) {
        fail_msg("SLATMS info=%d", info);
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
                    ws->A[ioff + i - 1] = 0.0f;
                }
                ioff = ioff + izero - 1;
                for (int i = izero; i <= n; i++) {
                    ws->A[ioff] = 0.0f;
                    ioff = ioff + i;
                }
            } else {
                /* Lower packed */
                int ioff = izero;
                for (int i = 1; i <= izero - 1; i++) {
                    ws->A[ioff - 1] = 0.0f;
                    ioff = ioff + n - i;
                }
                ioff = ioff - izero;
                for (int i = izero; i <= n; i++) {
                    ws->A[ioff + i - 1] = 0.0f;
                }
            }
        } else {
            /* IMAT = 6: zero first/last IZERO rows and columns */
            int ioff = 0;
            if (iuplo == 0) {
                for (int j = 1; j <= n; j++) {
                    int i2 = (j < izero) ? j : izero;
                    for (int i = 1; i <= i2; i++) {
                        ws->A[ioff + i - 1] = 0.0f;
                    }
                    ioff = ioff + j;
                }
            } else {
                for (int j = 1; j <= n; j++) {
                    int i1 = (j > izero) ? j : izero;
                    for (int i = i1; i <= n; i++) {
                        ws->A[ioff + i - 1] = 0.0f;
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
    f32 rcondc = 0.0f;

    if (zerot) {
        rcondc = 0.0f;
    } else if (n == 0) {
        rcondc = 1.0f / cndnum;
    } else {
        f32 anrm = slansp("1", uplo, n, ws->A, ws->RWORK);

        cblas_scopy(npp, ws->A, 1, ws->AFAC, 1);
        ssptrf(uplo, n, ws->AFAC, ws->IWORK, &info);

        cblas_scopy(npp, ws->AFAC, 1, ws->AINV, 1);
        ssptri(uplo, n, ws->AINV, ws->IWORK, ws->WORK, &info);
        f32 ainvnm = slansp("1", uplo, n, ws->AINV, ws->RWORK);

        if (anrm <= 0.0f || ainvnm <= 0.0f) {
            rcondc = 1.0f;
        } else {
            rcondc = (1.0f / anrm) / ainvnm;
        }
    }

    /* Form exact solution and set right hand side */
    char xtype = 'N';
    slarhs("SSP", &xtype, uplo, " ", n, n, kl, ku,
           NRHS, ws->A, lda, ws->XACT, lda, ws->B, lda, &info, rng_state);
    xtype = 'C';

    /* --- Test SSPSV --- */
    if (ifact == 1) {
        cblas_scopy(npp, ws->A, 1, ws->AFAC, 1);
        slacpy("Full", n, NRHS, ws->B, lda, ws->X, lda);

        sspsv(uplo, n, NRHS, ws->AFAC, ws->IWORK, ws->X, lda, &info);

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

        /* Check error code from SSPSV */
        if (info != k) {
            fail_msg("SSPSV UPLO=%s info=%d expected=%d (n=%d imat=%d)",
                     uplo, info, k, n, imat);
        } else if (info == 0) {
            /* TEST 1: Reconstruct matrix from factors */
            sspt01(uplo, n, ws->A, ws->AFAC, ws->IWORK,
                   ws->AINV, lda, ws->RWORK, &result[0]);

            /* TEST 2: Compute residual of the computed solution */
            slacpy("Full", n, NRHS, ws->B, lda, ws->WORK, lda);
            sppt02(uplo, n, NRHS, ws->A, ws->X, lda, ws->WORK,
                   lda, ws->RWORK, &result[1]);

            /* TEST 3: Check solution from generated exact solution */
            sget04(n, NRHS, ws->X, lda, ws->XACT, lda, rcondc, &result[2]);
            int nt = 3;

            for (int i = 0; i < nt; i++) {
                if (result[i] >= THRESH) {
                    fail_msg("SSPSV UPLO=%s test %d failed: result=%e >= thresh=%e",
                             uplo, i + 1, result[i], THRESH);
                }
            }
        }
    }

    /* --- Test SSPSVX --- */
    if (ifact == 1 && npp > 0) {
        slaset("Full", npp, 1, 0.0f, 0.0f, ws->AFAC, npp);
    }
    slaset("Full", n, NRHS, 0.0f, 0.0f, ws->X, lda);

    f32 rcond;
    sspsvx(fact, uplo, n, NRHS, ws->A, ws->AFAC, ws->IWORK,
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

    /* Check error code from SSPSVX */
    if (info != k) {
        fail_msg("SSPSVX FACT=%s UPLO=%s info=%d expected=%d (n=%d imat=%d)",
                 fact, uplo, info, k, n, imat);
        return;
    }

    int k1;
    if (info == 0) {
        if (ifact >= 1) {
            /* TEST 1: Reconstruct matrix from factors */
            sspt01(uplo, n, ws->A, ws->AFAC, ws->IWORK,
                   ws->AINV, lda, &ws->RWORK[2 * NRHS], &result[0]);
            k1 = 1;
        } else {
            k1 = 2;
        }

        /* TEST 2: Compute residual of the computed solution */
        slacpy("Full", n, NRHS, ws->B, lda, ws->WORK, lda);
        sppt02(uplo, n, NRHS, ws->A, ws->X, lda, ws->WORK,
               lda, &ws->RWORK[2 * NRHS], &result[1]);

        /* TEST 3: Check solution from generated exact solution */
        sget04(n, NRHS, ws->X, lda, ws->XACT, lda, rcondc, &result[2]);

        /* TEST 4-5: Check error bounds from iterative refinement */
        sppt05(uplo, n, NRHS, ws->A, ws->B, lda,
               ws->X, lda, ws->XACT, lda,
               ws->RWORK, &ws->RWORK[NRHS], &result[3]);
    } else {
        k1 = 6;
    }

    /* TEST 6: Compare RCOND from SSPSVX with computed value */
    result[5] = sget06(rcond, rcondc);

    /* Check results */
    for (int i = k1 - 1; i < NTESTS; i++) {
        if (result[i] >= THRESH) {
            fail_msg("SSPSVX FACT=%s UPLO=%s test %d: result=%e >= thresh=%e",
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
