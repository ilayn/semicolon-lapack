/**
 * @file test_ddrvsy_rk.c
 * @brief DDRVSY_RK tests the driver routine DSYSV_RK.
 *
 * Port of LAPACK TESTING/LIN/ddrvsy_rk.f to C with CMocka parameterization.
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
#define NTESTS  3
#define THRESH  30.0
#define NMAX    50
#define NRHS    2

/* Routine under test */
extern void dsysv_rk(const char* uplo, const int n, const int nrhs,
                     f64* A, const int lda, f64* E, int* ipiv,
                     f64* B, const int ldb, f64* work, const int lwork,
                     int* info);

/* Supporting routines */
extern void dsytrf_rk(const char* uplo, const int n, f64* A, const int lda,
                      f64* E, int* ipiv, f64* work, const int lwork, int* info);
extern void dsytri_3(const char* uplo, const int n, f64* A, const int lda,
                     const f64* E, const int* ipiv, f64* work, const int lwork,
                     int* info);

/* Verification routines */
extern void dsyt01_3(const char* uplo, const int n,
                     const f64* A, const int lda,
                     f64* AFAC, const int ldafac, f64* E,
                     int* ipiv,
                     f64* C, const int ldc, f64* rwork, f64* resid);
extern void dpot02(const char* uplo, const int n, const int nrhs,
                   const f64* A, const int lda, const f64* X, const int ldx,
                   f64* B, const int ldb, f64* rwork, f64* resid);
extern void dget04(const int n, const int nrhs, const f64* X, const int ldx,
                   const f64* XACT, const int ldxact, const f64 rcond,
                   f64* resid);

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
extern f64 dlansy(const char* norm, const char* uplo, const int n,
                  const f64* A, const int lda, f64* work);
typedef struct {
    int n;
    int imat;
    int iuplo;      /* 0='U', 1='L' */
    char name[64];
} ddrvsy_rk_params_t;

typedef struct {
    f64* A;
    f64* AFAC;
    f64* E;
    f64* AINV;
    f64* B;
    f64* X;
    f64* XACT;
    f64* WORK;
    f64* RWORK;
    int* IWORK;
    int lwork;
} ddrvsy_rk_workspace_t;

static ddrvsy_rk_workspace_t* g_workspace = NULL;

static int group_setup(void** state)
{
    (void)state;
    g_workspace = malloc(sizeof(ddrvsy_rk_workspace_t));
    if (!g_workspace) return -1;

    int nmax = NMAX;
    int nb = 1;

    xlaenv(1, nb);
    xlaenv(2, 2);

    int lwork = 2 * nmax;
    if (lwork < nmax * NRHS) lwork = nmax * NRHS;
    /* Workspace for dsytri_3 */
    int lwork_tri = (nmax + nb + 1) * (nb + 3);
    if (lwork < lwork_tri) lwork = lwork_tri;

    g_workspace->lwork = lwork;
    g_workspace->A = calloc(nmax * nmax, sizeof(f64));
    g_workspace->AFAC = calloc(nmax * nmax, sizeof(f64));
    g_workspace->E = calloc(nmax, sizeof(f64));
    g_workspace->AINV = calloc(nmax * nmax, sizeof(f64));
    g_workspace->B = calloc(nmax * NRHS, sizeof(f64));
    g_workspace->X = calloc(nmax * NRHS, sizeof(f64));
    g_workspace->XACT = calloc(nmax * NRHS, sizeof(f64));
    g_workspace->WORK = calloc(lwork, sizeof(f64));
    g_workspace->RWORK = calloc(nmax + 2 * NRHS, sizeof(f64));
    g_workspace->IWORK = calloc(2 * nmax, sizeof(int));

    if (!g_workspace->A || !g_workspace->AFAC || !g_workspace->E ||
        !g_workspace->AINV ||
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
        free(g_workspace->E);
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

static void run_ddrvsy_rk_single(int n, int imat, int iuplo)
{
    static const char* UPLOS[] = {"U", "L"};

    ddrvsy_rk_workspace_t* ws = g_workspace;
    const char* uplo = UPLOS[iuplo];

    int lda = (n > 1) ? n : 1;
    int nb = 1;
    f64 result[NTESTS];
    for (int k = 0; k < NTESTS; k++) result[k] = 0.0;

    int zerot = (imat >= 3 && imat <= 6);
    int izero = 0;

    /* Set up parameters with DLATB4 */
    char type, dist;
    int kl, ku, mode;
    f64 anorm, cndnum;
    dlatb4("DSY", imat, n, n, &type, &kl, &ku, &anorm, &mode, &cndnum, &dist);

    /* Generate test matrix with DLATMS */
    uint64_t rng_state[4];
    rng_seed(rng_state, 1988 + n * 1000 + imat * 100 + iuplo * 10);
    int info;
    dlatms(n, n, &dist, &type, ws->RWORK, mode, cndnum,
           anorm, kl, ku, uplo, ws->A, lda, ws->WORK, &info, rng_state);
    if (info != 0) {
        fail_msg("DLATMS info=%d", info);
        return;
    }

    /* For types 3-6, zero one or more rows and columns */
    if (zerot) {
        if (imat == 3) {
            izero = 1;
        } else if (imat == 4) {
            izero = n;
        } else {
            izero = n / 2 + 1;
        }

        if (imat < 6) {
            if (iuplo == 0) {
                int ioff = (izero - 1) * lda;
                for (int i = 0; i < izero - 1; i++) {
                    ws->A[ioff + i] = 0.0;
                }
                ioff = ioff + izero - 1;
                for (int i = izero - 1; i < n; i++) {
                    ws->A[ioff] = 0.0;
                    ioff = ioff + lda;
                }
            } else {
                int ioff = izero - 1;
                for (int i = 0; i < izero - 1; i++) {
                    ws->A[ioff] = 0.0;
                    ioff = ioff + lda;
                }
                ioff = ioff - (izero - 1);
                for (int i = izero - 1; i < n; i++) {
                    ws->A[ioff + i] = 0.0;
                }
            }
        } else {
            int ioff = 0;
            if (iuplo == 0) {
                for (int j = 0; j < n; j++) {
                    int i2 = (j + 1 < izero) ? j + 1 : izero;
                    for (int i = 0; i < i2; i++) {
                        ws->A[ioff + i] = 0.0;
                    }
                    ioff = ioff + lda;
                }
            } else {
                for (int j = 0; j < n; j++) {
                    int i1 = (j + 1 > izero) ? j : izero - 1;
                    for (int i = i1; i < n; i++) {
                        ws->A[ioff + i] = 0.0;
                    }
                    ioff = ioff + lda;
                }
            }
        }
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
        f64 anrm = dlansy("1", uplo, n, ws->A, lda, ws->RWORK);

        dlacpy(uplo, n, n, ws->A, lda, ws->AFAC, lda);
        dsytrf_rk(uplo, n, ws->AFAC, lda, ws->E, ws->IWORK, ws->WORK,
                  ws->lwork, &info);

        dlacpy(uplo, n, n, ws->AFAC, lda, ws->AINV, lda);
        int lwork_tri = (n + nb + 1) * (nb + 3);
        dsytri_3(uplo, n, ws->AINV, lda, ws->E, ws->IWORK, ws->WORK,
                 lwork_tri, &info);
        f64 ainvnm = dlansy("1", uplo, n, ws->AINV, lda, ws->RWORK);

        if (anrm <= 0.0 || ainvnm <= 0.0) {
            rcondc = 1.0;
        } else {
            rcondc = (1.0 / anrm) / ainvnm;
        }
    }

    /* Form exact solution and set right hand side */
    char xtype = 'N';
    dlarhs("DSY", &xtype, uplo, " ", n, n, kl, ku,
           NRHS, ws->A, lda, ws->XACT, lda, ws->B, lda, &info, rng_state);

    /* --- Test DSYSV_RK --- */
    dlacpy(uplo, n, n, ws->A, lda, ws->AFAC, lda);
    dlacpy("Full", n, NRHS, ws->B, lda, ws->X, lda);

    int lwork = 2 * n;
    if (lwork < n * NRHS) lwork = n * NRHS;
    if (lwork < 1) lwork = 1;
    if (lwork > ws->lwork) lwork = ws->lwork;

    dsysv_rk(uplo, n, NRHS, ws->AFAC, lda, ws->E, ws->IWORK,
             ws->X, lda, ws->WORK, lwork, &info);

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

    /* Check error code from DSYSV_RK */
    if (info != k) {
        fail_msg("DSYSV_RK UPLO=%s info=%d expected=%d (n=%d imat=%d)",
                 uplo, info, k, n, imat);
        return;
    } else if (info != 0) {
        return;
    }

    /* TEST 1: Reconstruct matrix from factors */
    dsyt01_3(uplo, n, ws->A, lda, ws->AFAC, lda, ws->E, ws->IWORK,
             ws->AINV, lda, ws->RWORK, &result[0]);

    /* TEST 2: Compute residual of the computed solution */
    dlacpy("Full", n, NRHS, ws->B, lda, ws->WORK, lda);
    dpot02(uplo, n, NRHS, ws->A, lda, ws->X, lda, ws->WORK,
           lda, ws->RWORK, &result[1]);

    /* TEST 3: Check solution from generated exact solution */
    dget04(n, NRHS, ws->X, lda, ws->XACT, lda, rcondc, &result[2]);
    int nt = 3;

    for (int i = 0; i < nt; i++) {
        if (result[i] >= THRESH) {
            fail_msg("DSYSV_RK UPLO=%s test %d failed: result=%e >= thresh=%e",
                     uplo, i + 1, result[i], THRESH);
        }
    }
}

static void test_ddrvsy_rk_case(void** state)
{
    ddrvsy_rk_params_t* p = *state;
    run_ddrvsy_rk_single(p->n, p->imat, p->iuplo);
}

#define MAX_TESTS 1500

static ddrvsy_rk_params_t g_params[MAX_TESTS];
static struct CMUnitTest g_tests[MAX_TESTS];
static int g_num_tests = 0;

static void build_test_array(void)
{
    static const char* UPLOS[] = {"U", "L"};

    g_num_tests = 0;

    for (int in = 0; in < (int)NN; in++) {
        int n = NVAL[in];
        int nimat = (n <= 0) ? 1 : NTYPES;

        for (int imat = 1; imat <= nimat; imat++) {
            int zerot = (imat >= 3 && imat <= 6);
            if (zerot && n < imat - 2) continue;

            for (int iuplo = 0; iuplo < 2; iuplo++) {
                ddrvsy_rk_params_t* p = &g_params[g_num_tests];
                p->n = n;
                p->imat = imat;
                p->iuplo = iuplo;
                snprintf(p->name, sizeof(p->name),
                         "n%d_t%d_%s",
                         n, imat, UPLOS[iuplo]);

                g_tests[g_num_tests].name = p->name;
                g_tests[g_num_tests].test_func = test_ddrvsy_rk_case;
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
    return _cmocka_run_group_tests("ddrvsy_rk", g_tests, g_num_tests,
                                   group_setup, group_teardown);
}
