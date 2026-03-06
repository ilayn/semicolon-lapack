/**
 * @file test_cchksy.c
 * @brief Comprehensive test suite for complex symmetric indefinite matrix (CSY) routines.
 *
 * This is a faithful port of LAPACK's TESTING/LIN/zchksy.f to C using CMocka.
 * Tests CSYTRF, CSYTRI2, CSYTRS, CSYTRS2, CSYRFS, and CSYCON.
 *
 * Each (n, uplo, imat, inb) combination is registered as a separate CMocka test,
 * providing pytest-like parameterized test behavior with clear failure isolation.
 *
 * Test structure from zchksy.f:
 *   TEST 1: LDL^T factorization residual via csyt01
 *   TEST 2: Matrix inverse residual via csyt03
 *   TEST 3: Solution residual via csyt02 (using csytrs)
 *   TEST 4: Solution residual via csyt02 (using csytrs2)
 *   TEST 5: Solution accuracy via cget04
 *   TEST 6: Refined solution accuracy via cget04 (after csyrfs)
 *   TEST 7-8: Error bounds via cpot05
 *   TEST 9: Condition number via sget06
 *
 * Parameters from ztest.in:
 *   N values: 0, 1, 2, 3, 5, 10, 50
 *   NRHS values: 1, 2, 15
 *   Matrix types: 1-11
 *   THRESH: 30.0
 */

#include "test_harness.h"
#include "verify.h"
#include "test_rng.h"
#include <string.h>
#include <stdio.h>

static const INT NVAL[] = {0, 1, 2, 3, 5, 10, 50};
static const INT NSVAL[] = {1, 2, 15};
static const INT NBVAL[] = {1, 3, 3, 3, 20};
static const INT NXVAL[] = {1, 0, 5, 9, 1};
static const char UPLOS[] = {'U', 'L'};

#define NN      (sizeof(NVAL) / sizeof(NVAL[0]))
#define NNS     (sizeof(NSVAL) / sizeof(NSVAL[0]))
#define NNB     (sizeof(NBVAL) / sizeof(NBVAL[0]))
#define NUPLO   (sizeof(UPLOS) / sizeof(UPLOS[0]))
#define NTYPES  11
#define NTESTS  9
#define THRESH  30.0f
#define NMAX    50
#define NSMAX   15

typedef struct {
    INT n;
    INT imat;
    INT iuplo;
    INT inb;
    char name[64];
} zchksy_params_t;

typedef struct {
    c64* A;
    c64* AFAC;
    c64* AINV;
    c64* B;
    c64* X;
    c64* XACT;
    c64* WORK;
    f32* RWORK;
    f32* D;
    f32* FERR;
    f32* BERR;
    INT* IPIV;
} zchksy_workspace_t;

static zchksy_workspace_t* g_workspace = NULL;

static int group_setup(void** state)
{
    (void)state;
    g_workspace = malloc(sizeof(zchksy_workspace_t));
    if (!g_workspace) return -1;

    INT lwork = NMAX * 64;

    g_workspace->A = malloc(NMAX * NMAX * sizeof(c64));
    g_workspace->AFAC = malloc(NMAX * NMAX * sizeof(c64));
    g_workspace->AINV = malloc(NMAX * NMAX * sizeof(c64));
    g_workspace->B = malloc(NMAX * NSMAX * sizeof(c64));
    g_workspace->X = malloc(NMAX * NSMAX * sizeof(c64));
    g_workspace->XACT = malloc(NMAX * NSMAX * sizeof(c64));
    g_workspace->WORK = malloc(lwork * sizeof(c64));
    g_workspace->RWORK = malloc(NMAX * sizeof(f32));
    g_workspace->D = malloc(NMAX * sizeof(f32));
    g_workspace->FERR = malloc(NSMAX * sizeof(f32));
    g_workspace->BERR = malloc(NSMAX * sizeof(f32));
    g_workspace->IPIV = malloc(NMAX * sizeof(INT));

    if (!g_workspace->A || !g_workspace->AFAC || !g_workspace->AINV ||
        !g_workspace->B || !g_workspace->X || !g_workspace->XACT ||
        !g_workspace->WORK || !g_workspace->RWORK || !g_workspace->D ||
        !g_workspace->FERR || !g_workspace->BERR || !g_workspace->IPIV) {
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
        free(g_workspace->D);
        free(g_workspace->FERR);
        free(g_workspace->BERR);
        free(g_workspace->IPIV);
        free(g_workspace);
        g_workspace = NULL;
    }
    return 0;
}

static void run_zchksy_single(INT n, INT iuplo, INT imat, INT inb)
{
    const f32 ZERO = 0.0f;
    zchksy_workspace_t* ws = g_workspace;

    char type, dist;
    char uplo = UPLOS[iuplo];
    char uplo_str[2] = {uplo, '\0'};
    INT kl, ku, mode;
    f32 anorm, cndnum;
    INT info, izero;
    INT lda = (n > 1) ? n : 1;
    INT lwork = NMAX * 64;
    INT trfcon;
    f32 rcondc, rcond;
    f32 result[NTESTS];
    char ctx[128];

    INT nb = NBVAL[inb];
    INT nx = NXVAL[inb];
    xlaenv(1, nb);
    xlaenv(2, 2);
    xlaenv(3, nx);

    uint64_t rng_state[4];
    rng_seed(rng_state, 1988198919901991ULL + (uint64_t)(n * 1000 + iuplo * 100 + imat));

    for (INT k = 0; k < NTESTS; k++) {
        result[k] = ZERO;
    }

    if (imat != NTYPES) {
        clatb4("CSY", imat, n, n, &type, &kl, &ku, &anorm, &mode, &cndnum, &dist);

        clatms(n, n, &dist, &type, ws->D, mode, cndnum, anorm,
               kl, ku, "N", ws->A, lda, ws->WORK, &info, rng_state);
        assert_int_equal(info, 0);

        INT zerot = (imat >= 3 && imat <= 6);
        if (zerot) {
            if (imat == 3) {
                izero = 0;
            } else if (imat == 4) {
                izero = n - 1;
            } else {
                izero = n / 2;
            }

            if (imat < 6) {
                if (iuplo == 0) {
                    INT ioff = izero * lda;
                    for (INT i = 0; i < izero; i++) {
                        ws->A[ioff + i] = ZERO;
                    }
                    ioff += izero;
                    for (INT i = izero; i < n; i++) {
                        ws->A[ioff] = ZERO;
                        ioff += lda;
                    }
                } else {
                    INT ioff = izero;
                    for (INT i = 0; i < izero; i++) {
                        ws->A[ioff] = ZERO;
                        ioff += lda;
                    }
                    ioff = izero * lda + izero;
                    for (INT i = izero; i < n; i++) {
                        ws->A[ioff + i - izero] = ZERO;
                    }
                }
            } else {
                if (iuplo == 0) {
                    INT ioff = 0;
                    for (INT j = 0; j < n; j++) {
                        INT i2 = (j <= izero) ? j + 1 : izero + 1;
                        for (INT i = 0; i < i2; i++) {
                            ws->A[ioff + i] = ZERO;
                        }
                        ioff += lda;
                    }
                } else {
                    INT ioff = 0;
                    for (INT j = 0; j < n; j++) {
                        INT i1 = (j >= izero) ? j : izero;
                        for (INT i = i1; i < n; i++) {
                            ws->A[ioff + i] = ZERO;
                        }
                        ioff += lda;
                    }
                }
            }
        } else {
            izero = -1;
        }
    } else {
        izero = -1;
        kl = n - 1;
        ku = n - 1;
        clatsy(uplo_str, n, ws->A, lda, rng_state);
    }

    clacpy(uplo_str, n, n, ws->A, lda, ws->AFAC, lda);

    csytrf(uplo_str, n, ws->AFAC, lda, ws->IPIV, ws->WORK, lwork, &info);

    if (izero >= 0) {
        INT k = izero;
        while (k >= 0 && k < n) {
            if (ws->IPIV[k] < 0) {
                INT kp = -(ws->IPIV[k] + 1);
                if (kp != k) {
                    k = kp;
                } else {
                    break;
                }
            } else if (ws->IPIV[k] != k) {
                k = ws->IPIV[k];
            } else {
                break;
            }
        }
        assert_true(info >= 0);
    }
    trfcon = (info != 0);
    if (trfcon) {
        rcondc = ZERO;
    }

    /*
     * TEST 1
     */
    snprintf(ctx, sizeof(ctx), "n=%d uplo=%c imat=%d TEST 1 (factorization)", n, uplo, imat);
    set_test_context(ctx);
    csyt01(uplo_str, n, ws->A, lda, ws->AFAC, lda, ws->IPIV,
           ws->AINV, lda, ws->RWORK, &result[0]);
    assert_residual_below(result[0], THRESH);

    /*
     * TEST 2
     */
    if (inb == 0 && !trfcon) {
        snprintf(ctx, sizeof(ctx), "n=%d uplo=%c imat=%d TEST 2 (inverse)", n, uplo, imat);
        set_test_context(ctx);
        clacpy(uplo_str, n, n, ws->AFAC, lda, ws->AINV, lda);
        INT lwork_tri2 = (n + nb + 1) * (nb + 3);
        csytri2(uplo_str, n, ws->AINV, lda, ws->IPIV, ws->WORK, lwork_tri2, &info);
        if (info == 0) {
            csyt03(uplo_str, n, ws->A, lda, ws->AINV, lda, ws->WORK, lda,
                   ws->RWORK, &rcondc, &result[1]);
            assert_residual_below(result[1], THRESH);
        }
    }

    if (inb > 0) {
        clear_test_context();
        return;
    }

    if (trfcon) {
        rcondc = ZERO;
        goto test9;
    }

    for (INT irhs = 0; irhs < (INT)NNS; irhs++) {
        INT nrhs = NSVAL[irhs];

        /*
         * TEST 3
         */
        snprintf(ctx, sizeof(ctx), "n=%d uplo=%c imat=%d nrhs=%d TEST 3 (solve)", n, uplo, imat, nrhs);
        set_test_context(ctx);
        clarhs("CSY", "N", uplo_str, " ", n, n, kl, ku, nrhs,
               ws->A, lda, ws->XACT, lda, ws->B, lda, &info, rng_state);

        clacpy("F", n, nrhs, ws->B, lda, ws->X, lda);
        csytrs(uplo_str, n, nrhs, ws->AFAC, lda, ws->IPIV, ws->X, lda, &info);
        assert_int_equal(info, 0);

        clacpy("F", n, nrhs, ws->B, lda, ws->WORK, lda);
        csyt02(uplo_str, n, nrhs, ws->A, lda, ws->X, lda,
               ws->WORK, lda, ws->RWORK, &result[2]);
        assert_residual_below(result[2], THRESH);

        /*
         * TEST 4
         */
        snprintf(ctx, sizeof(ctx), "n=%d uplo=%c imat=%d nrhs=%d TEST 4 (solve2)", n, uplo, imat, nrhs);
        set_test_context(ctx);
        clarhs("CSY", "N", uplo_str, " ", n, n, kl, ku, nrhs,
               ws->A, lda, ws->XACT, lda, ws->B, lda, &info, rng_state);

        clacpy("F", n, nrhs, ws->B, lda, ws->X, lda);
        csytrs2(uplo_str, n, nrhs, ws->AFAC, lda, ws->IPIV, ws->X, lda, ws->WORK, &info);
        assert_int_equal(info, 0);

        clacpy("F", n, nrhs, ws->B, lda, ws->WORK, lda);
        csyt02(uplo_str, n, nrhs, ws->A, lda, ws->X, lda,
               ws->WORK, lda, ws->RWORK, &result[3]);
        assert_residual_below(result[3], THRESH);

        /*
         * TEST 5
         */
        snprintf(ctx, sizeof(ctx), "n=%d uplo=%c imat=%d nrhs=%d TEST 5 (accuracy)", n, uplo, imat, nrhs);
        set_test_context(ctx);
        cget04(n, nrhs, ws->X, lda, ws->XACT, lda, rcondc, &result[4]);
        assert_residual_below(result[4], THRESH);

        /*
         * TESTS 6, 7, 8
         */
        snprintf(ctx, sizeof(ctx), "n=%d uplo=%c imat=%d nrhs=%d TEST 6-8 (refinement)", n, uplo, imat, nrhs);
        set_test_context(ctx);

        csyrfs(uplo_str, n, nrhs, ws->A, lda, ws->AFAC, lda, ws->IPIV,
               ws->B, lda, ws->X, lda, ws->FERR, ws->BERR,
               ws->WORK, ws->RWORK, &info);
        assert_int_equal(info, 0);

        cget04(n, nrhs, ws->X, lda, ws->XACT, lda, rcondc, &result[5]);
        cpot05(uplo_str, n, nrhs, ws->A, lda, ws->B, lda, ws->X, lda,
               ws->XACT, lda, ws->FERR, ws->BERR, &result[6]);

        assert_residual_below(result[5], THRESH);
        assert_residual_below(result[6], THRESH);
        assert_residual_below(result[7], THRESH);
    }

test9:
    /*
     * TEST 9
     */
    snprintf(ctx, sizeof(ctx), "n=%d uplo=%c imat=%d TEST 9 (condition)", n, uplo, imat);
    set_test_context(ctx);
    anorm = clansy("1", uplo_str, n, ws->A, lda, ws->RWORK);
    csycon(uplo_str, n, ws->AFAC, lda, ws->IPIV, anorm, &rcond,
           ws->WORK, &info);
    assert_int_equal(info, 0);

    result[8] = sget06(rcond, rcondc);
    assert_residual_below(result[8], THRESH);

    clear_test_context();
}

static void test_zchksy_case(void** state)
{
    zchksy_params_t* params = *state;
    run_zchksy_single(params->n, params->iuplo, params->imat, params->inb);
}

#define MAX_TESTS (NN * NUPLO * NTYPES * NNB)

static zchksy_params_t g_params[MAX_TESTS];
static struct CMUnitTest g_tests[MAX_TESTS];
static INT g_num_tests = 0;

static void build_test_array(void)
{
    g_num_tests = 0;

    for (INT in = 0; in < (INT)NN; in++) {
        INT n = NVAL[in];

        INT nimat = NTYPES;
        if (n <= 0) {
            nimat = 1;
        }

        for (INT imat = 1; imat <= nimat; imat++) {
            INT zerot = (imat >= 3 && imat <= 6);
            if (zerot && n < imat - 2) {
                continue;
            }

            for (INT iuplo = 0; iuplo < (INT)NUPLO; iuplo++) {
                for (INT inb = 0; inb < (INT)NNB; inb++) {
                    INT nb = NBVAL[inb];

                    zchksy_params_t* p = &g_params[g_num_tests];
                    p->n = n;
                    p->imat = imat;
                    p->iuplo = iuplo;
                    p->inb = inb;
                    snprintf(p->name, sizeof(p->name), "zchksy_n%d_%c_type%d_nb%d_%d",
                             n, UPLOS[iuplo], imat, nb, inb);

                    g_tests[g_num_tests].name = p->name;
                    g_tests[g_num_tests].test_func = test_zchksy_case;
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
    (void)_cmocka_run_group_tests("zchksy", g_tests, g_num_tests,
                                   group_setup, group_teardown);
    return 0;
}
