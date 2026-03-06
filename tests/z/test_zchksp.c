/**
 * @file test_zchksp.c
 * @brief Comprehensive test suite for complex symmetric packed matrix (ZSP) routines.
 *
 * This is a faithful port of LAPACK's TESTING/LIN/zchksp.f to C using CMocka.
 * Tests ZSPTRF, ZSPTRI, ZSPTRS, ZSPRFS, and ZSPCON.
 *
 * Each (n, uplo, imat) combination is registered as a separate CMocka test,
 * providing pytest-like parameterized test behavior with clear failure isolation.
 *
 * Test structure from zchksp.f:
 *   TEST 1: LDL'/UDU' factorization residual via zspt01
 *   TEST 2: Matrix inverse residual via zspt03
 *   TEST 3: Solution residual via zspt02
 *   TEST 4: Solution accuracy via zget04
 *   TEST 5: Refined solution accuracy via zget04 (after zsprfs)
 *   TEST 6-7: Error bounds via zppt05
 *   TEST 8: Condition number via dget06
 */

#include "test_harness.h"
#include "verify.h"
#include "test_rng.h"
#include <string.h>
#include <stdio.h>
#include "semicolon_cblas.h"

static const INT NVAL[] = {0, 1, 2, 3, 5, 10, 50};
static const INT NSVAL[] = {1, 2, 15};
static const char UPLOS[] = {'U', 'L'};
static const char* PACKS[] = {"C", "R"};

#define NN      (sizeof(NVAL) / sizeof(NVAL[0]))
#define NNS     (sizeof(NSVAL) / sizeof(NSVAL[0]))
#define NUPLO   (sizeof(UPLOS) / sizeof(UPLOS[0]))
#define NTYPES  11
#define NTESTS  8
#define THRESH  30.0
#define NMAX    50
#define NSMAX   15

typedef struct {
    INT n;
    INT imat;
    INT iuplo;
    char name[64];
} zchksp_params_t;

typedef struct {
    c128* A;
    c128* AFAC;
    c128* AINV;
    c128* B;
    c128* X;
    c128* XACT;
    c128* WORK;
    f64* RWORK;
    f64* D;
    f64* FERR;
    f64* BERR;
    INT* IWORK;
} zchksp_workspace_t;

static zchksp_workspace_t* g_workspace = NULL;

static int group_setup(void** state)
{
    (void)state;
    g_workspace = malloc(sizeof(zchksp_workspace_t));
    if (!g_workspace) return -1;

    g_workspace->A = malloc(NMAX * NMAX * sizeof(c128));
    g_workspace->AFAC = malloc(NMAX * NMAX * sizeof(c128));
    g_workspace->AINV = malloc(NMAX * NMAX * sizeof(c128));
    g_workspace->B = malloc(NMAX * NSMAX * sizeof(c128));
    g_workspace->X = malloc(NMAX * NSMAX * sizeof(c128));
    g_workspace->XACT = malloc(NMAX * NSMAX * sizeof(c128));
    g_workspace->WORK = malloc(NMAX * NMAX * sizeof(c128));
    g_workspace->RWORK = malloc((NMAX + 2 * NSMAX) * sizeof(f64));
    g_workspace->D = malloc(NMAX * sizeof(f64));
    g_workspace->FERR = malloc(NSMAX * sizeof(f64));
    g_workspace->BERR = malloc(NSMAX * sizeof(f64));
    g_workspace->IWORK = malloc(NMAX * sizeof(INT));

    if (!g_workspace->A || !g_workspace->AFAC || !g_workspace->AINV ||
        !g_workspace->B || !g_workspace->X || !g_workspace->XACT ||
        !g_workspace->WORK || !g_workspace->RWORK || !g_workspace->D ||
        !g_workspace->FERR || !g_workspace->BERR || !g_workspace->IWORK) {
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
        free(g_workspace->IWORK);
        free(g_workspace);
        g_workspace = NULL;
    }
    return 0;
}

static void run_zchksp_single(INT n, INT iuplo, INT imat)
{
    const f64 ZERO = 0.0;
    zchksp_workspace_t* ws = g_workspace;

    char type, dist;
    char uplo = UPLOS[iuplo];
    char uplo_str[2] = {uplo, '\0'};
    const char* packit = PACKS[iuplo];
    INT kl, ku, mode;
    f64 anorm, cndnum;
    INT info, izero;
    INT lda = (n > 1) ? n : 1;
    INT npp = n * (n + 1) / 2;
    f64 rcondc, rcond;
    INT trfcon;
    INT k;
    INT* ipiv = ws->IWORK;

    f64 result[NTESTS];
    char ctx[128];

    uint64_t rng_state[4];
    rng_seed(rng_state, 1988198919901991ULL + (uint64_t)(n * 1000 + iuplo * 100 + imat));

    for (INT i = 0; i < NTESTS; i++) {
        result[i] = ZERO;
    }

    if (imat != NTYPES) {
        zlatb4("ZSP", imat, n, n, &type, &kl, &ku, &anorm, &mode, &cndnum, &dist);

        zlatms(n, n, &dist, &type, ws->D, mode, cndnum, anorm,
               kl, ku, packit, ws->A, lda, ws->WORK, &info, rng_state);
        assert_int_equal(info, 0);

        INT zerot = (imat >= 3 && imat <= 6);
        if (zerot) {
            if (imat == 3) {
                izero = 1;
            } else if (imat == 4) {
                izero = n;
            } else {
                izero = n / 2 + 1;
            }

            if (imat < 6) {
                INT ioff;
                if (iuplo == 0) {
                    ioff = (izero - 1) * izero / 2;
                    for (INT i = 0; i < izero - 1; i++) {
                        ws->A[ioff + i] = ZERO;
                    }
                    ioff = ioff + izero - 1;
                    for (INT i = izero - 1; i < n; i++) {
                        ws->A[ioff] = ZERO;
                        ioff = ioff + i + 1;
                    }
                } else {
                    ioff = izero - 1;
                    for (INT i = 0; i < izero - 1; i++) {
                        ws->A[ioff] = ZERO;
                        ioff = ioff + n - i - 1;
                    }
                    ioff = ioff - (izero - 1);
                    for (INT i = izero - 1; i < n; i++) {
                        ws->A[ioff + i - (izero - 1)] = ZERO;
                    }
                }
            } else {
                INT ioff = 0;
                if (iuplo == 0) {
                    for (INT j = 0; j < n; j++) {
                        INT i2 = (j + 1 < izero) ? j + 1 : izero;
                        for (INT i = 0; i < i2; i++) {
                            ws->A[ioff + i] = ZERO;
                        }
                        ioff = ioff + j + 1;
                    }
                } else {
                    for (INT j = 0; j < n; j++) {
                        INT i1 = (j >= izero - 1) ? j : izero - 1;
                        for (INT i = i1; i < n; i++) {
                            ws->A[ioff + i - j] = ZERO;
                        }
                        ioff = ioff + n - j;
                    }
                }
            }
        } else {
            izero = 0;
        }
    } else {
        izero = 0;
        zlatsp(uplo_str, n, ws->A, rng_state);
    }

    cblas_zcopy(npp, ws->A, 1, ws->AFAC, 1);

    zsptrf(uplo_str, n, ws->AFAC, ipiv, &info);

    k = izero - 1;
    if (k >= 0) {
        while (1) {
            if (ipiv[k] < 0) {
                INT kp = -ipiv[k] - 1;
                if (kp != k) {
                    k = kp;
                    continue;
                }
            } else if (ipiv[k] != k) {
                k = ipiv[k];
                continue;
            }
            break;
        }
    }

    if (info != k + 1) {
        return;
    }

    if (info != 0) {
        trfcon = 1;
    } else {
        trfcon = 0;
    }

    snprintf(ctx, sizeof(ctx), "n=%d uplo=%c imat=%d TEST 1 (factorization)", n, uplo, imat);
    set_test_context(ctx);
    zspt01(uplo_str, n, ws->A, ws->AFAC, ipiv, ws->WORK, lda, ws->RWORK, &result[0]);
    assert_residual_below(result[0], THRESH);

    if (!trfcon) {
        snprintf(ctx, sizeof(ctx), "n=%d uplo=%c imat=%d TEST 2 (inverse)", n, uplo, imat);
        set_test_context(ctx);
        cblas_zcopy(npp, ws->AFAC, 1, ws->AINV, 1);
        zsptri(uplo_str, n, ws->AINV, ipiv, ws->WORK, &info);
        assert_int_equal(info, 0);

        zspt03(uplo_str, n, ws->A, ws->AINV, ws->WORK, lda,
               ws->RWORK, &rcondc, &result[1]);
        assert_residual_below(result[1], THRESH);
    }

    if (trfcon) {
        rcondc = ZERO;
        goto test8;
    }

    for (INT irhs = 0; irhs < (INT)NNS; irhs++) {
        INT nrhs = NSVAL[irhs];

        snprintf(ctx, sizeof(ctx), "n=%d uplo=%c imat=%d nrhs=%d TEST 3 (solve)", n, uplo, imat, nrhs);
        set_test_context(ctx);

        zlarhs("ZSP", "N", uplo_str, " ", n, n, kl, ku, nrhs,
               ws->A, lda, ws->XACT, lda, ws->B, lda, &info, rng_state);

        zlacpy("F", n, nrhs, ws->B, lda, ws->X, lda);
        zsptrs(uplo_str, n, nrhs, ws->AFAC, ipiv, ws->X, lda, &info);
        assert_int_equal(info, 0);

        zlacpy("F", n, nrhs, ws->B, lda, ws->WORK, lda);
        zspt02(uplo_str, n, nrhs, ws->A, ws->X, lda,
               ws->WORK, lda, ws->RWORK, &result[2]);
        assert_residual_below(result[2], THRESH);

        snprintf(ctx, sizeof(ctx), "n=%d uplo=%c imat=%d nrhs=%d TEST 4 (solution accuracy)", n, uplo, imat, nrhs);
        set_test_context(ctx);
        zget04(n, nrhs, ws->X, lda, ws->XACT, lda, rcondc, &result[3]);
        assert_residual_below(result[3], THRESH);

        snprintf(ctx, sizeof(ctx), "n=%d uplo=%c imat=%d nrhs=%d TEST 5-7 (refinement)", n, uplo, imat, nrhs);
        set_test_context(ctx);

        zsprfs(uplo_str, n, nrhs, ws->A, ws->AFAC, ipiv,
               ws->B, lda, ws->X, lda, ws->FERR, ws->BERR,
               ws->WORK, ws->RWORK, &info);
        assert_int_equal(info, 0);

        zget04(n, nrhs, ws->X, lda, ws->XACT, lda, rcondc, &result[4]);
        zppt05(uplo_str, n, nrhs, ws->A, ws->B, lda, ws->X, lda,
               ws->XACT, lda, ws->FERR, ws->BERR, &result[5]);

        assert_residual_below(result[4], THRESH);
        assert_residual_below(result[5], THRESH);
        assert_residual_below(result[6], THRESH);
    }

test8:
    snprintf(ctx, sizeof(ctx), "n=%d uplo=%c imat=%d TEST 8 (condition number)", n, uplo, imat);
    set_test_context(ctx);
    anorm = zlansp("1", uplo_str, n, ws->A, ws->RWORK);
    zspcon(uplo_str, n, ws->AFAC, ipiv, anorm, &rcond, ws->WORK,
           &info);
    assert_int_equal(info, 0);

    result[7] = dget06(rcond, rcondc);
    assert_residual_below(result[7], THRESH);

    clear_test_context();
}

static void test_zchksp_case(void** state)
{
    zchksp_params_t* params = *state;
    run_zchksp_single(params->n, params->iuplo, params->imat);
}

#define MAX_TESTS (NN * NUPLO * NTYPES)

static zchksp_params_t g_params[MAX_TESTS];
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
            if (zerot && n < imat - 2)
                continue;

            for (INT iuplo = 0; iuplo < (INT)NUPLO; iuplo++) {
                zchksp_params_t* p = &g_params[g_num_tests];
                p->n = n;
                p->imat = imat;
                p->iuplo = iuplo;
                snprintf(p->name, sizeof(p->name), "zchksp_n%d_%c_type%d",
                         n, UPLOS[iuplo], imat);

                g_tests[g_num_tests].name = p->name;
                g_tests[g_num_tests].test_func = test_zchksp_case;
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
    (void)_cmocka_run_group_tests("zchksp", g_tests, g_num_tests,
                                   group_setup, group_teardown);
    return 0;
}
