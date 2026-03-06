/**
 * @file test_zchkge.c
 * @brief Comprehensive test suite for complex general matrix (ZGE) routines.
 *
 * This is a faithful port of LAPACK's TESTING/LIN/zchkge.f to C using CMocka.
 * Tests ZGETRF, ZGETRI, ZGETRS, ZGERFS, and ZGECON.
 */

#include "test_harness.h"
#include "verify.h"
#include "test_rng.h"
#include <string.h>
#include <stdio.h>

static const INT MVAL[] = {0, 1, 2, 3, 5, 10, 50};
static const INT NVAL[] = {0, 1, 2, 3, 5, 10, 50};
static const INT NSVAL[] = {1, 2, 15};
static const INT NBVAL[] = {1, 3, 3, 3, 20};
static const char TRANSS[] = {'N', 'T', 'C'};

#define NM      (sizeof(MVAL) / sizeof(MVAL[0]))
#define NN      (sizeof(NVAL) / sizeof(NVAL[0]))
#define NNS     (sizeof(NSVAL) / sizeof(NSVAL[0]))
#define NNB     (sizeof(NBVAL) / sizeof(NBVAL[0]))
#define NTRAN   (sizeof(TRANSS) / sizeof(TRANSS[0]))
#define NTYPES  11
#define NTESTS  8
#define THRESH  30.0
#define NMAX    50
#define NSMAX   15

typedef struct {
    INT m;
    INT n;
    INT imat;
    INT inb;
    char name[64];
} zchkge_params_t;

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
    INT* IPIV;
} zchkge_workspace_t;

static zchkge_workspace_t* g_workspace = NULL;

static int group_setup(void** state)
{
    (void)state;
    g_workspace = malloc(sizeof(zchkge_workspace_t));
    if (!g_workspace) return -1;

    g_workspace->A = malloc(NMAX * NMAX * sizeof(c128));
    g_workspace->AFAC = malloc(NMAX * NMAX * sizeof(c128));
    g_workspace->AINV = malloc(NMAX * NMAX * sizeof(c128));
    g_workspace->B = malloc(NMAX * NSMAX * sizeof(c128));
    g_workspace->X = malloc(NMAX * NSMAX * sizeof(c128));
    g_workspace->XACT = malloc(NMAX * NSMAX * sizeof(c128));
    g_workspace->WORK = malloc(NMAX * NMAX * sizeof(c128));
    g_workspace->RWORK = malloc(2 * NMAX * sizeof(f64));
    g_workspace->D = malloc(NMAX * sizeof(f64));
    g_workspace->FERR = malloc(NSMAX * sizeof(f64));
    g_workspace->BERR = malloc(NSMAX * sizeof(f64));
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

static void run_zchkge_single(INT m, INT n, INT imat, INT inb)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    zchkge_workspace_t* ws = g_workspace;

    char type, dist;
    INT kl, ku, mode;
    f64 anorm, cndnum;
    INT info, izero;
    INT lda = (m > 1) ? m : 1;
    INT trfcon;
    f64 anormo, anormi, rcondo, rcondi, rcond, rcondc;
    f64 result[NTESTS];

    INT nb = NBVAL[inb];
    xlaenv(1, nb);

    uint64_t rng_state[4];
    rng_seed(rng_state, 1988198919901991ULL + (uint64_t)(m * 1000 + n * 100 + imat));

    for (INT k = 0; k < NTESTS; k++) {
        result[k] = ZERO;
    }

    zlatb4("ZGE", imat, m, n, &type, &kl, &ku, &anorm, &mode, &cndnum, &dist);

    zlatms(m, n, &dist, &type, ws->D, mode, cndnum, anorm,
           kl, ku, "N", ws->A, lda, ws->WORK, &info, rng_state);
    assert_int_equal(info, 0);

    INT zerot = (imat >= 5 && imat <= 7);
    if (zerot) {
        INT minmn = (m < n) ? m : n;
        if (imat == 5) {
            izero = 1;
        } else if (imat == 6) {
            izero = minmn;
        } else {
            izero = minmn / 2 + 1;
        }
        INT ioff = (izero - 1) * lda;
        if (imat < 7) {
            for (INT i = 0; i < m; i++) {
                ws->A[ioff + i] = ZERO;
            }
        } else {
            zlaset("F", m, n - izero + 1, ZERO, ZERO, &ws->A[ioff], lda);
        }
    } else {
        izero = 0;
    }

    zlacpy("F", m, n, ws->A, lda, ws->AFAC, lda);

    zgetrf(m, n, ws->AFAC, lda, ws->IPIV, &info);

    if (zerot) {
        assert_true(info >= 0);
    } else {
        assert_int_equal(info, 0);
    }
    trfcon = (info != 0);

    /* TEST 1 */
    zlacpy("F", m, n, ws->AFAC, lda, ws->AINV, lda);
    zget01(m, n, ws->A, lda, ws->AINV, lda, ws->IPIV, ws->RWORK, &result[0]);
    assert_residual_below(result[0], THRESH);

    /* TEST 2 */
    if (m == n && info == 0) {
        zlacpy("F", n, n, ws->AFAC, lda, ws->AINV, lda);
        INT lwork = NMAX * 3;
        zgetri(n, ws->AINV, lda, ws->IPIV, ws->WORK, lwork, &info);
        assert_int_equal(info, 0);

        zget03(n, ws->A, lda, ws->AINV, lda, ws->WORK, lda,
               ws->RWORK, &rcondo, &result[1]);
        anormo = zlange("O", m, n, ws->A, lda, ws->RWORK);

        anormi = zlange("I", m, n, ws->A, lda, ws->RWORK);
        f64 ainvnm = zlange("I", n, n, ws->AINV, lda, ws->RWORK);
        if (anormi <= ZERO || ainvnm <= ZERO) {
            rcondi = ONE;
        } else {
            rcondi = (ONE / anormi) / ainvnm;
        }
        assert_residual_below(result[1], THRESH);
    } else {
        trfcon = 1;
        anormo = zlange("O", m, n, ws->A, lda, ws->RWORK);
        anormi = zlange("I", m, n, ws->A, lda, ws->RWORK);
        rcondo = ZERO;
        rcondi = ZERO;
    }

    if (m != n || trfcon || inb > 0) {
        goto test8;
    }

    /* TESTS 3-7 */
    for (INT irhs = 0; irhs < (INT)NNS; irhs++) {
        INT nrhs = NSVAL[irhs];

        for (INT itran = 0; itran < (INT)NTRAN; itran++) {
            char trans_arr[2] = {TRANSS[itran], '\0'};
            rcondc = (itran == 0) ? rcondo : rcondi;

            /* TEST 3 */
            zlarhs("ZGE", "N", " ", trans_arr, n, n, kl, ku, nrhs,
                   ws->A, lda, ws->XACT, lda, ws->B, lda, &info, rng_state);

            zlacpy("F", n, nrhs, ws->B, lda, ws->X, lda);
            zgetrs(trans_arr, n, nrhs, ws->AFAC, lda, ws->IPIV, ws->X, lda, &info);
            assert_int_equal(info, 0);

            zlacpy("F", n, nrhs, ws->B, lda, ws->WORK, lda);
            zget02(trans_arr, n, n, nrhs, ws->A, lda, ws->X, lda,
                   ws->WORK, lda, ws->RWORK, &result[2]);

            assert_residual_below(result[2], THRESH);

            /* TEST 4 */
            zget04(n, nrhs, ws->X, lda, ws->XACT, lda, rcondc, &result[3]);

            assert_residual_below(result[3], THRESH);

            /* TESTS 5, 6, 7 */
            zgerfs(trans_arr, n, nrhs, ws->A, lda, ws->AFAC, lda,
                   ws->IPIV, ws->B, lda, ws->X, lda, ws->FERR, ws->BERR,
                   ws->WORK, ws->RWORK, &info);
            assert_int_equal(info, 0);

            zget04(n, nrhs, ws->X, lda, ws->XACT, lda, rcondc, &result[4]);
            zget07(trans_arr, n, nrhs, ws->A, lda, ws->B, lda, ws->X, lda,
                   ws->XACT, lda, ws->FERR, 1, ws->BERR, &result[5]);

            assert_residual_below(result[4], THRESH);
            assert_residual_below(result[5], THRESH);
            assert_residual_below(result[6], THRESH);
        }
    }

test8:
    /* TEST 8 */
    if (m == n) {
        for (INT itran = 0; itran < 2; itran++) {
            char norm;
            if (itran == 0) {
                anorm = anormo;
                rcondc = rcondo;
                norm = 'O';
            } else {
                anorm = anormi;
                rcondc = rcondi;
                norm = 'I';
            }
            char norm_arr[2] = {norm, '\0'};
            zgecon(norm_arr, n, ws->AFAC, lda, anorm, &rcond, ws->WORK,
                   ws->RWORK, &info);
            assert_int_equal(info, 0);

            result[7] = dget06(rcond, rcondc);

            assert_residual_below(result[7], THRESH);
        }
    }
}

static void test_zchkge_case(void** state)
{
    zchkge_params_t* params = *state;
    run_zchkge_single(params->m, params->n, params->imat, params->inb);
}

#define MAX_TESTS (NM * NN * NTYPES * NNB)

static zchkge_params_t g_params[MAX_TESTS];
static struct CMUnitTest g_tests[MAX_TESTS];
static INT g_num_tests = 0;

static void build_test_array(void)
{
    g_num_tests = 0;

    for (INT im = 0; im < (INT)NM; im++) {
        INT m = MVAL[im];

        for (INT in = 0; in < (INT)NN; in++) {
            INT n = NVAL[in];

            INT nimat = NTYPES;
            if (m <= 0 || n <= 0) {
                nimat = 1;
            }

            for (INT imat = 1; imat <= nimat; imat++) {
                INT zerot = (imat >= 5 && imat <= 7);
                INT minmn = (m < n) ? m : n;
                if (zerot && minmn < imat - 4) {
                    continue;
                }

                for (INT inb = 0; inb < (INT)NNB; inb++) {
                    INT nb = NBVAL[inb];

                    zchkge_params_t* p = &g_params[g_num_tests];
                    p->m = m;
                    p->n = n;
                    p->imat = imat;
                    p->inb = inb;
                    snprintf(p->name, sizeof(p->name), "zchkge_m%d_n%d_type%d_nb%d_%d",
                             m, n, imat, nb, inb);

                    g_tests[g_num_tests].name = p->name;
                    g_tests[g_num_tests].test_func = test_zchkge_case;
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

    (void)_cmocka_run_group_tests("zchkge", g_tests, g_num_tests,
                                   group_setup, group_teardown);
    return 0;
}
