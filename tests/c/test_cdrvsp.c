/**
 * @file test_cdrvsp.c
 * @brief ZDRVSP tests the driver routines CSPSV and CSPSVX.
 *
 * Port of LAPACK TESTING/LIN/zdrvsp.f to C with CMocka parameterization.
 */

#include "test_harness.h"
#include "verify.h"
#include "test_rng.h"
#include <string.h>
#include <stdio.h>
#include <math.h>

/* Test parameters - matching LAPACK zchkaa.f defaults */
static const INT NVAL[] = {0, 1, 2, 3, 5, 10, 50};
#define NN      (sizeof(NVAL) / sizeof(NVAL[0]))
#define NTYPES  10
#define NTESTS  6
#define NFACT   2
#define THRESH  30.0f
#define NMAX    50
#define NRHS    2
#define NPP_MAX (NMAX * (NMAX + 1) / 2)

typedef struct {
    INT n;
    INT imat;
    INT iuplo;      /* 0='U', 1='L' */
    INT ifact;      /* 0='F', 1='N' */
    char name[64];
} zdrvsp_params_t;

typedef struct {
    c64* A;
    c64* AFAC;
    c64* AINV;
    c64* B;
    c64* X;
    c64* XACT;
    c64* WORK;
    f32* RWORK;
    INT* IWORK;
    INT lwork;
} zdrvsp_workspace_t;

static zdrvsp_workspace_t* g_workspace = NULL;

static int group_setup(void** state)
{
    (void)state;
    g_workspace = malloc(sizeof(zdrvsp_workspace_t));
    if (!g_workspace) return -1;

    INT nmax = NMAX;
    INT lwork = 3 * nmax;
    if (lwork < nmax * NRHS) lwork = nmax * NRHS;

    g_workspace->lwork = lwork;
    g_workspace->A = calloc(nmax * nmax, sizeof(c64));
    g_workspace->AFAC = calloc(NPP_MAX, sizeof(c64));
    g_workspace->AINV = calloc(nmax * nmax, sizeof(c64));
    g_workspace->B = calloc(nmax * NRHS, sizeof(c64));
    g_workspace->X = calloc(nmax * NRHS, sizeof(c64));
    g_workspace->XACT = calloc(nmax * NRHS, sizeof(c64));
    g_workspace->WORK = calloc(lwork, sizeof(c64));
    g_workspace->RWORK = calloc(nmax + 2 * NRHS, sizeof(f32));
    g_workspace->IWORK = calloc(2 * nmax, sizeof(INT));

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

static void run_zdrvsp_single(INT n, INT imat, INT iuplo, INT ifact)
{
    static const char* UPLOS[] = {"U", "L"};
    static const char* FACTS[] = {"F", "N"};

    zdrvsp_workspace_t* ws = g_workspace;
    const char* uplo = UPLOS[iuplo];
    const char* fact = FACTS[ifact];

    INT lda = (n > 1) ? n : 1;
    INT npp = n * (n + 1) / 2;
    f32 result[NTESTS];
    for (INT k = 0; k < NTESTS; k++) result[k] = 0.0f;

    INT zerot = (imat >= 3 && imat <= 6);
    INT izero = 0;

    const char* packit = (iuplo == 0) ? "C" : "R";

    /* Set up parameters with CLATB4 */
    char type, dist;
    INT kl, ku, mode;
    f32 anorm, cndnum;
    clatb4("CSP", imat, n, n, &type, &kl, &ku, &anorm, &mode, &cndnum, &dist);

    /* Generate test matrix with CLATMS */
    uint64_t rng_state[4];
    rng_seed(rng_state, 1988 + n * 1000 + imat * 100 + iuplo * 10);
    INT info;
    clatms(n, n, &dist, &type, ws->RWORK, mode, cndnum,
           anorm, kl, ku, packit, ws->A, lda, ws->WORK, &info, rng_state);
    if (info != 0) {
        fail_msg("CLATMS info=%d", info);
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
            if (iuplo == 0) {
                INT ioff = (izero - 1) * izero / 2;
                for (INT i = 1; i <= izero - 1; i++) {
                    ws->A[ioff + i - 1] = 0.0f;
                }
                ioff = ioff + izero - 1;
                for (INT i = izero; i <= n; i++) {
                    ws->A[ioff] = 0.0f;
                    ioff = ioff + i;
                }
            } else {
                INT ioff = izero;
                for (INT i = 1; i <= izero - 1; i++) {
                    ws->A[ioff - 1] = 0.0f;
                    ioff = ioff + n - i;
                }
                ioff = ioff - izero;
                for (INT i = izero; i <= n; i++) {
                    ws->A[ioff + i - 1] = 0.0f;
                }
            }
        } else {
            INT ioff = 0;
            if (iuplo == 0) {
                for (INT j = 1; j <= n; j++) {
                    INT i2 = (j < izero) ? j : izero;
                    for (INT i = 1; i <= i2; i++) {
                        ws->A[ioff + i - 1] = 0.0f;
                    }
                    ioff = ioff + j;
                }
            } else {
                for (INT j = 1; j <= n; j++) {
                    INT i1 = (j > izero) ? j : izero;
                    for (INT i = i1; i <= n; i++) {
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
     * Since CMocka tests are independent, we always compute it.
     */
    f32 rcondc = 0.0f;

    if (zerot) {
        rcondc = 0.0f;
    } else if (n == 0) {
        rcondc = 1.0f / cndnum;
    } else {
        f32 anrm = clansp("1", uplo, n, ws->A, ws->RWORK);

        memcpy(ws->AFAC, ws->A, npp * sizeof(c64));
        csptrf(uplo, n, ws->AFAC, ws->IWORK, &info);

        memcpy(ws->AINV, ws->AFAC, npp * sizeof(c64));
        csptri(uplo, n, ws->AINV, ws->IWORK, ws->WORK, &info);
        f32 ainvnm = clansp("1", uplo, n, ws->AINV, ws->RWORK);

        if (anrm <= 0.0f || ainvnm <= 0.0f) {
            rcondc = 1.0f;
        } else {
            rcondc = (1.0f / anrm) / ainvnm;
        }
    }

    /* Form exact solution and set right hand side */
    char xtype = 'N';
    clarhs("CSP", &xtype, uplo, " ", n, n, kl, ku,
           NRHS, ws->A, lda, ws->XACT, lda, ws->B, lda, &info, rng_state);
    xtype = 'C';

    /* --- Test CSPSV --- */
    if (ifact == 1) {
        memcpy(ws->AFAC, ws->A, npp * sizeof(c64));
        clacpy("Full", n, NRHS, ws->B, lda, ws->X, lda);

        cspsv(uplo, n, NRHS, ws->AFAC, ws->IWORK, ws->X, lda, &info);

        /* Adjust expected value of INFO to account for pivoting */
        INT k = 0;
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

        /* Check error code from CSPSV */
        if (info != k) {
            fail_msg("CSPSV UPLO=%s info=%d expected=%d (n=%d imat=%d)",
                     uplo, info, k, n, imat);
        } else if (info == 0) {
            /* TEST 1: Reconstruct matrix from factors */
            cspt01(uplo, n, ws->A, ws->AFAC, ws->IWORK,
                   ws->AINV, lda, ws->RWORK, &result[0]);

            /* TEST 2: Compute residual of the computed solution */
            clacpy("Full", n, NRHS, ws->B, lda, ws->WORK, lda);
            cspt02(uplo, n, NRHS, ws->A, ws->X, lda, ws->WORK,
                   lda, ws->RWORK, &result[1]);

            /* TEST 3: Check solution from generated exact solution */
            cget04(n, NRHS, ws->X, lda, ws->XACT, lda, rcondc, &result[2]);
            INT nt = 3;

            for (INT i = 0; i < nt; i++) {
                if (result[i] >= THRESH) {
                    fail_msg("CSPSV UPLO=%s test %d failed: result=%e >= thresh=%e",
                             uplo, i + 1, (double)result[i], (double)THRESH);
                }
            }
        }
    }

    /* --- Test CSPSVX --- */
    if (ifact == 1 && npp > 0) {
        claset("Full", npp, 1, CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
               ws->AFAC, npp);
    }
    claset("Full", n, NRHS, CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
           ws->X, lda);

    f32 rcond;
    cspsvx(fact, uplo, n, NRHS, ws->A, ws->AFAC, ws->IWORK,
           ws->B, lda, ws->X, lda, &rcond,
           ws->RWORK, &ws->RWORK[NRHS], ws->WORK,
           &ws->RWORK[2 * NRHS], &info);

    /* Adjust expected value of INFO to account for pivoting */
    INT k = 0;
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

    /* Check error code from CSPSVX */
    if (info != k) {
        fail_msg("CSPSVX FACT=%s UPLO=%s info=%d expected=%d (n=%d imat=%d)",
                 fact, uplo, info, k, n, imat);
        return;
    }

    INT k1;
    if (info == 0) {
        if (ifact >= 1) {
            /* TEST 1: Reconstruct matrix from factors */
            cspt01(uplo, n, ws->A, ws->AFAC, ws->IWORK,
                   ws->AINV, lda, &ws->RWORK[2 * NRHS], &result[0]);
            k1 = 1;
        } else {
            k1 = 2;
        }

        /* TEST 2: Compute residual of the computed solution */
        clacpy("Full", n, NRHS, ws->B, lda, ws->WORK, lda);
        cspt02(uplo, n, NRHS, ws->A, ws->X, lda, ws->WORK,
               lda, &ws->RWORK[2 * NRHS], &result[1]);

        /* TEST 3: Check solution from generated exact solution */
        cget04(n, NRHS, ws->X, lda, ws->XACT, lda, rcondc, &result[2]);

        /* TEST 4-5: Check error bounds from iterative refinement */
        cppt05(uplo, n, NRHS, ws->A, ws->B, lda,
               ws->X, lda, ws->XACT, lda,
               ws->RWORK, &ws->RWORK[NRHS], &result[3]);
    } else {
        k1 = 6;
    }

    /* TEST 6: Compare RCOND from CSPSVX with computed value */
    result[5] = sget06(rcond, rcondc);

    /* Check results */
    for (INT i = k1 - 1; i < NTESTS; i++) {
        if (result[i] >= THRESH) {
            fail_msg("CSPSVX FACT=%s UPLO=%s test %d: result=%e >= thresh=%e",
                     fact, uplo, i + 1, (double)result[i], (double)THRESH);
        }
    }
}

static void test_zdrvsp_case(void** state)
{
    zdrvsp_params_t* p = *state;
    run_zdrvsp_single(p->n, p->imat, p->iuplo, p->ifact);
}

#define MAX_TESTS 3000

static zdrvsp_params_t g_params[MAX_TESTS];
static struct CMUnitTest g_tests[MAX_TESTS];
static INT g_num_tests = 0;

static void build_test_array(void)
{
    static const char* UPLOS[] = {"U", "L"};
    static const char* FACTS_STR[] = {"F", "N"};

    g_num_tests = 0;

    for (INT in = 0; in < (INT)NN; in++) {
        INT n = NVAL[in];
        INT nimat = (n <= 0) ? 1 : NTYPES;

        for (INT imat = 1; imat <= nimat; imat++) {
            INT zerot = (imat >= 3 && imat <= 6);
            if (zerot && n < imat - 2) continue;

            for (INT iuplo = 0; iuplo < 2; iuplo++) {
                for (INT ifact = 0; ifact < NFACT; ifact++) {
                    if (zerot && ifact == 0) continue;

                    zdrvsp_params_t* p = &g_params[g_num_tests];
                    p->n = n;
                    p->imat = imat;
                    p->iuplo = iuplo;
                    p->ifact = ifact;
                    snprintf(p->name, sizeof(p->name),
                             "n%d_t%d_%s_%s",
                             n, imat, UPLOS[iuplo], FACTS_STR[ifact]);

                    g_tests[g_num_tests].name = p->name;
                    g_tests[g_num_tests].test_func = test_zdrvsp_case;
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
    (void)_cmocka_run_group_tests("zdrvsp", g_tests, g_num_tests,
                                   group_setup, group_teardown);
    return 0;
}
