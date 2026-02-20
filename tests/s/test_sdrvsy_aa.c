/**
 * @file test_sdrvsy_aa.c
 * @brief DDRVSY_AA tests the driver routine SSYSV_AA.
 *
 * Port of LAPACK TESTING/LIN/ddrvsy_aa.f to C with CMocka parameterization.
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
#define THRESH  30.0f
#define NMAX    50
#define NRHS    2

/* Routine under test */
extern void ssysv_aa(const char* uplo, const int n, const int nrhs,
                     f32* A, const int lda, int* ipiv,
                     f32* B, const int ldb, f32* work, const int lwork,
                     int* info);

/* Verification routines */
extern void ssyt01_aa(const char* uplo, const int n,
                      const f32* A, const int lda,
                      const f32* AFAC, const int ldafac,
                      const int* ipiv,
                      f32* C, const int ldc, f32* rwork, f32* resid);
extern void spot02(const char* uplo, const int n, const int nrhs,
                   const f32* A, const int lda, const f32* X, const int ldx,
                   f32* B, const int ldb, f32* rwork, f32* resid);

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
typedef struct {
    int n;
    int imat;
    int iuplo;      /* 0='U', 1='L' */
    char name[64];
} ddrvsy_aa_params_t;

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
} ddrvsy_aa_workspace_t;

static ddrvsy_aa_workspace_t* g_workspace = NULL;

static int group_setup(void** state)
{
    (void)state;
    g_workspace = malloc(sizeof(ddrvsy_aa_workspace_t));
    if (!g_workspace) return -1;

    int nmax = NMAX;
    int nb = 1;

    xlaenv(1, nb);
    xlaenv(2, 2);

    /* LWORK = MAX(3*N-2, N*(1+NB), 1) -- for largest N=NMAX */
    int lwork = 3 * nmax - 2;
    if (lwork < nmax * (1 + nb)) lwork = nmax * (1 + nb);
    if (lwork < 1) lwork = 1;
    /* Also need workspace for slatms and spot02 */
    if (lwork < 2 * nmax) lwork = 2 * nmax;
    if (lwork < nmax * NRHS) lwork = nmax * NRHS;

    g_workspace->lwork = lwork;
    g_workspace->A = calloc(nmax * nmax, sizeof(f32));
    g_workspace->AFAC = calloc(nmax * nmax, sizeof(f32));
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

static void run_ddrvsy_aa_single(int n, int imat, int iuplo)
{
    static const char* UPLOS[] = {"U", "L"};

    ddrvsy_aa_workspace_t* ws = g_workspace;
    const char* uplo = UPLOS[iuplo];

    int lda = (n > 1) ? n : 1;
    f32 result[NTESTS];
    for (int k = 0; k < NTESTS; k++) result[k] = 0.0f;

    int nb = 1;
    int zerot = (imat >= 3 && imat <= 6);
    int izero = 0;

    /* Set up parameters with SLATB4 */
    char type, dist;
    int kl, ku, mode;
    f32 anorm, cndnum;
    slatb4("SSY", imat, n, n, &type, &kl, &ku, &anorm, &mode, &cndnum, &dist);

    /* Generate test matrix with SLATMS */
    uint64_t rng_state[4];
    rng_seed(rng_state, 1988 + n * 1000 + imat * 100 + iuplo * 10);
    int info;
    slatms(n, n, &dist, &type, ws->RWORK, mode, cndnum,
           anorm, kl, ku, uplo, ws->A, lda, ws->WORK, &info, rng_state);
    if (info != 0) {
        fail_msg("SLATMS info=%d", info);
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
                    ws->A[ioff + i] = 0.0f;
                }
                ioff = ioff + izero - 1;
                for (int i = izero - 1; i < n; i++) {
                    ws->A[ioff] = 0.0f;
                    ioff = ioff + lda;
                }
            } else {
                int ioff = izero - 1;
                for (int i = 0; i < izero - 1; i++) {
                    ws->A[ioff] = 0.0f;
                    ioff = ioff + lda;
                }
                ioff = ioff - (izero - 1);
                for (int i = izero - 1; i < n; i++) {
                    ws->A[ioff + i] = 0.0f;
                }
            }
        } else {
            int ioff = 0;
            if (iuplo == 0) {
                for (int j = 0; j < n; j++) {
                    int i2 = (j + 1 < izero) ? j + 1 : izero;
                    for (int i = 0; i < i2; i++) {
                        ws->A[ioff + i] = 0.0f;
                    }
                    ioff = ioff + lda;
                }
                izero = 1;
            } else {
                for (int j = 0; j < n; j++) {
                    int i1 = (j + 1 > izero) ? j : izero - 1;
                    for (int i = i1; i < n; i++) {
                        ws->A[ioff + i] = 0.0f;
                    }
                    ioff = ioff + lda;
                }
            }
        }
    }

    /* Form exact solution and set right hand side */
    char xtype = 'N';
    slarhs("SSY", &xtype, uplo, " ", n, n, kl, ku,
           NRHS, ws->A, lda, ws->XACT, lda, ws->B, lda, &info, rng_state);

    /* --- Test SSYSV_AA --- */
    slacpy(uplo, n, n, ws->A, lda, ws->AFAC, lda);
    slacpy("Full", n, NRHS, ws->B, lda, ws->X, lda);

    int lwork = 3 * n - 2;
    if (lwork < n * (1 + nb)) lwork = n * (1 + nb);
    if (lwork < 1) lwork = 1;
    if (lwork > ws->lwork) lwork = ws->lwork;

    ssysv_aa(uplo, n, NRHS, ws->AFAC, lda, ws->IWORK,
             ws->X, lda, ws->WORK, lwork, &info);

    /* Adjust expected value of INFO to account for pivoting */
    int k = 0;
    if (izero > 0) {
        int j = 0;
        k = izero - 1;
        for (;;) {
            if (j == k) {
                k = ws->IWORK[j];
            } else if (ws->IWORK[j] == k) {
                k = j;
            }
            if (j < k) {
                j = j + 1;
            } else {
                break;
            }
        }
        k = k + 1;
    }

    /* Check error code from SSYSV_AA */
    if (info != k) {
        fail_msg("SSYSV_AA UPLO=%s info=%d expected=%d (n=%d imat=%d)",
                 uplo, info, k, n, imat);
        return;
    } else if (info != 0) {
        return;
    }

    /* TEST 1: Reconstruct matrix from factors */
    ssyt01_aa(uplo, n, ws->A, lda, ws->AFAC, lda, ws->IWORK,
              ws->AINV, lda, ws->RWORK, &result[0]);

    /* TEST 2: Compute residual of the computed solution */
    slacpy("Full", n, NRHS, ws->B, lda, ws->WORK, lda);
    spot02(uplo, n, NRHS, ws->A, lda, ws->X, lda, ws->WORK,
           lda, ws->RWORK, &result[1]);
    int nt = 2;

    for (int i = 0; i < nt; i++) {
        if (result[i] >= THRESH) {
            fail_msg("SSYSV_AA UPLO=%s test %d failed: result=%e >= thresh=%e",
                     uplo, i + 1, (double)result[i], (double)THRESH);
        }
    }
}

static void test_ddrvsy_aa_case(void** state)
{
    ddrvsy_aa_params_t* p = *state;
    run_ddrvsy_aa_single(p->n, p->imat, p->iuplo);
}

#define MAX_TESTS 1500

static ddrvsy_aa_params_t g_params[MAX_TESTS];
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
                ddrvsy_aa_params_t* p = &g_params[g_num_tests];
                p->n = n;
                p->imat = imat;
                p->iuplo = iuplo;
                snprintf(p->name, sizeof(p->name),
                         "n%d_t%d_%s",
                         n, imat, UPLOS[iuplo]);

                g_tests[g_num_tests].name = p->name;
                g_tests[g_num_tests].test_func = test_ddrvsy_aa_case;
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
    return _cmocka_run_group_tests("ddrvsy_aa", g_tests, g_num_tests,
                                   group_setup, group_teardown);
}
