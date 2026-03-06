/**
 * @file test_zdrvpo.c
 * @brief ZDRVPO tests the driver routines ZPOSV and ZPOSVX.
 *
 * Port of LAPACK TESTING/LIN/zdrvpo.f to C with CMocka parameterization.
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
#define NTYPES  9
#define NTESTS  6
#define THRESH  30.0
#define NMAX    50
#define NRHS    2

typedef struct {
    INT n;
    INT imat;
    INT iuplo;      /* 0='U', 1='L' */
    INT ifact;      /* 0='F', 1='N', 2='E' */
    INT iequed;     /* 0='N', 1='Y' */
    char name[64];
} zdrvpo_params_t;

typedef struct {
    c128* A;
    c128* AFAC;
    c128* ASAV;
    c128* B;
    c128* BSAV;
    c128* X;
    c128* XACT;
    f64* S;
    c128* WORK;
    f64* RWORK;
} zdrvpo_workspace_t;

static zdrvpo_workspace_t* g_workspace = NULL;

static int group_setup(void** state)
{
    (void)state;
    g_workspace = malloc(sizeof(zdrvpo_workspace_t));
    if (!g_workspace) return -1;

    INT nmax = NMAX;
    INT lwork = nmax * (nmax > 3 ? nmax : 3);
    if (lwork < nmax * NRHS) lwork = nmax * NRHS;

    g_workspace->A = calloc(nmax * nmax, sizeof(c128));
    g_workspace->AFAC = calloc(nmax * nmax, sizeof(c128));
    g_workspace->ASAV = calloc(nmax * nmax, sizeof(c128));
    g_workspace->B = calloc(nmax * NRHS, sizeof(c128));
    g_workspace->BSAV = calloc(nmax * NRHS, sizeof(c128));
    g_workspace->X = calloc(nmax * NRHS, sizeof(c128));
    g_workspace->XACT = calloc(nmax * NRHS, sizeof(c128));
    g_workspace->S = calloc(nmax, sizeof(f64));
    g_workspace->WORK = calloc(lwork, sizeof(c128));
    g_workspace->RWORK = calloc(nmax + 2 * NRHS, sizeof(f64));

    if (!g_workspace->A || !g_workspace->AFAC || !g_workspace->ASAV ||
        !g_workspace->B || !g_workspace->BSAV || !g_workspace->X ||
        !g_workspace->XACT || !g_workspace->S || !g_workspace->WORK ||
        !g_workspace->RWORK) {
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
        free(g_workspace->ASAV);
        free(g_workspace->B);
        free(g_workspace->BSAV);
        free(g_workspace->X);
        free(g_workspace->XACT);
        free(g_workspace->S);
        free(g_workspace->WORK);
        free(g_workspace->RWORK);
        free(g_workspace);
        g_workspace = NULL;
    }
    return 0;
}

static void run_zdrvpo_single(INT n, INT imat, INT iuplo, INT ifact, INT iequed)
{
    static const char* UPLOS[] = {"U", "L"};
    static const char* FACTS[] = {"F", "N", "E"};
    static const char* EQUEDS[] = {"N", "Y"};

    zdrvpo_workspace_t* ws = g_workspace;
    const char* uplo = UPLOS[iuplo];
    const char* fact = FACTS[ifact];
    char equed = EQUEDS[iequed][0];
    const c128 CZERO = CMPLX(0.0, 0.0);

    INT prefac = (fact[0] == 'F');
    INT nofact = (fact[0] == 'N');
    INT equil = (fact[0] == 'E');

    INT lda = (n > 1) ? n : 1;
    f64 result[NTESTS];
    for (INT k = 0; k < NTESTS; k++) result[k] = 0.0;

    INT zerot = (imat >= 3 && imat <= 5);
    INT izero = 0;

    char type, dist;
    INT kl, ku, mode;
    f64 anorm, cndnum;
    zlatb4("ZPO", imat, n, n, &type, &kl, &ku, &anorm, &mode, &cndnum, &dist);

    uint64_t rng_state[4];
    rng_seed(rng_state, 1988 + n * 1000 + imat * 100 + iuplo * 10);
    INT info;
    zlatms(n, n, &dist, &type, ws->RWORK, mode, cndnum,
           anorm, kl, ku, uplo, ws->A, lda, ws->WORK, &info, rng_state);
    if (info != 0) {
        fail_msg("ZLATMS info=%d", info);
        return;
    }

    /* For types 3-5, zero one row and column */
    if (zerot) {
        if (imat == 3) {
            izero = 1;
        } else if (imat == 4) {
            izero = n;
        } else {
            izero = n / 2 + 1;
        }
        INT ioff = (izero - 1) * lda;

        if (iuplo == 0) {
            for (INT i = 0; i < izero - 1; i++) {
                ws->A[ioff + i] = CZERO;
            }
            ioff = ioff + izero - 1;
            for (INT i = izero - 1; i < n; i++) {
                ws->A[ioff] = CZERO;
                ioff = ioff + lda;
            }
        } else {
            ioff = izero - 1;
            for (INT i = 0; i < izero - 1; i++) {
                ws->A[ioff] = CZERO;
                ioff = ioff + lda;
            }
            ioff = ioff - (izero - 1);
            for (INT i = izero - 1; i < n; i++) {
                ws->A[ioff + i] = CZERO;
            }
        }
    }

    /* Set the imaginary part of the diagonals */
    zlaipd(n, ws->A, lda + 1, 0);

    /* Save a copy of the matrix A in ASAV */
    zlacpy(uplo, n, n, ws->A, lda, ws->ASAV, lda);

    /* Skip FACT='F' for singular matrices */
    if (zerot && prefac) {
        return;
    }

    f64 rcondc = 0.0;
    f64 roldc = 0.0;
    f64 scond = 0.0, amax = 0.0;

    if (zerot) {
        rcondc = 0.0;
    } else if (n == 0) {
        rcondc = 1.0 / cndnum;
    } else {
        zlacpy(uplo, n, n, ws->ASAV, lda, ws->AFAC, lda);

        if (equil || iequed > 0) {
            zpoequ(n, ws->AFAC, lda, ws->S, &scond, &amax, &info);
            if (info == 0 && n > 0) {
                if (iequed > 0) {
                    scond = 0.0;
                }
                zlaqhe(uplo, n, ws->AFAC, lda, ws->S, scond, amax, &equed);
            }
        }

        if (equil) {
            roldc = rcondc;
        }

        f64 anrm = zlanhe("1", uplo, n, ws->AFAC, lda, ws->RWORK);

        zpotrf(uplo, n, ws->AFAC, lda, &info);

        zlacpy(uplo, n, n, ws->AFAC, lda, ws->A, lda);
        zpotri(uplo, n, ws->A, lda, &info);

        f64 ainvnm = zlanhe("1", uplo, n, ws->A, lda, ws->RWORK);
        if (anrm <= 0.0 || ainvnm <= 0.0) {
            rcondc = 1.0;
        } else {
            rcondc = (1.0 / anrm) / ainvnm;
        }
    }
    if (!equil) {
        roldc = rcondc;
    }

    /* Restore the matrix A */
    zlacpy(uplo, n, n, ws->ASAV, lda, ws->A, lda);

    /* Form exact solution and set right hand side */
    rng_seed(rng_state, 1988 + n * 1000 + imat * 100 + iuplo * 10);
    char xtype = 'N';
    zlarhs("ZPO", &xtype, uplo, " ", n, n, kl, ku,
           NRHS, ws->A, lda, ws->XACT, lda, ws->B, lda, &info, rng_state);
    xtype = 'C';
    zlacpy("Full", n, NRHS, ws->B, lda, ws->BSAV, lda);

    /* --- Test ZPOSV --- */
    if (nofact) {
        zlacpy(uplo, n, n, ws->A, lda, ws->AFAC, lda);
        zlacpy("Full", n, NRHS, ws->B, lda, ws->X, lda);

        zposv(uplo, n, NRHS, ws->AFAC, lda, ws->X, lda, &info);

        if (info != izero) {
            fail_msg("ZPOSV info=%d expected=%d", info, izero);
            return;
        } else if (info != 0) {
            return;
        }

        zpot01(uplo, n, ws->A, lda, ws->AFAC, lda, ws->RWORK, &result[0]);

        zlacpy("Full", n, NRHS, ws->B, lda, ws->WORK, lda);
        zpot02(uplo, n, NRHS, ws->A, lda, ws->X, lda,
               ws->WORK, lda, ws->RWORK, &result[1]);

        zget04(n, NRHS, ws->X, lda, ws->XACT, lda, rcondc, &result[2]);
        INT nt = 3;

        for (INT k = 0; k < nt; k++) {
            if (result[k] >= THRESH) {
                fail_msg("ZPOSV UPLO=%s test %d failed: result=%e >= thresh=%e",
                         uplo, k + 1, result[k], THRESH);
            }
        }
    }

    /* --- Test ZPOSVX --- */
    if (!prefac) {
        zlaset(uplo, n, n, CZERO, CZERO, ws->AFAC, lda);
    }
    zlaset("Full", n, NRHS, CZERO, CZERO, ws->X, lda);

    if (iequed > 0 && n > 0) {
        zlaqhe(uplo, n, ws->A, lda, ws->S, scond, amax, &equed);
    }

    char equed_inout = equed;
    f64 rcond;
    zposvx(fact, uplo, n, NRHS, ws->A, lda, ws->AFAC, lda,
           &equed_inout, ws->S, ws->B, lda, ws->X, lda, &rcond,
           ws->RWORK, &ws->RWORK[NRHS], ws->WORK,
           &ws->RWORK[2 * NRHS], &info);

    if (info != izero) {
        fail_msg("ZPOSVX info=%d expected=%d", info, izero);
        return;
    }

    INT k1;
    if (info == 0) {
        if (!prefac) {
            zpot01(uplo, n, ws->A, lda, ws->AFAC, lda,
                   &ws->RWORK[2 * NRHS], &result[0]);
            k1 = 1;
        } else {
            k1 = 2;
        }

        zlacpy("Full", n, NRHS, ws->BSAV, lda, ws->WORK, lda);
        zpot02(uplo, n, NRHS, ws->ASAV, lda, ws->X, lda,
               ws->WORK, lda, &ws->RWORK[2 * NRHS], &result[1]);

        if (nofact || (prefac && (equed_inout == 'N' || equed_inout == 'n'))) {
            zget04(n, NRHS, ws->X, lda, ws->XACT, lda, rcondc, &result[2]);
        } else {
            zget04(n, NRHS, ws->X, lda, ws->XACT, lda, roldc, &result[2]);
        }

        zpot05(uplo, n, NRHS, ws->ASAV, lda, ws->B, lda,
               ws->X, lda, ws->XACT, lda,
               ws->RWORK, &ws->RWORK[NRHS], &result[3]);
    } else {
        k1 = 6;
    }

    result[5] = dget06(rcond, rcondc);

    if (info == 0) {
        for (INT k = k1 - 1; k < NTESTS; k++) {
            if (result[k] >= THRESH) {
                fail_msg("ZPOSVX FACT=%s UPLO=%s EQUED=%c test %d: result=%e >= thresh=%e",
                         fact, uplo, equed_inout, k + 1, result[k], THRESH);
            }
        }
    } else {
        if (!prefac && result[0] >= THRESH) {
            fail_msg("ZPOSVX FACT=%s UPLO=%s EQUED=%c test 1: result=%e >= thresh=%e",
                     fact, uplo, equed_inout, result[0], THRESH);
        }
        if (result[5] >= THRESH) {
            fail_msg("ZPOSVX FACT=%s UPLO=%s EQUED=%c test 6: result=%e >= thresh=%e",
                     fact, uplo, equed_inout, result[5], THRESH);
        }
    }
}

static void test_zdrvpo_case(void** state)
{
    zdrvpo_params_t* p = *state;
    run_zdrvpo_single(p->n, p->imat, p->iuplo, p->ifact, p->iequed);
}

#define MAX_TESTS 3000

static zdrvpo_params_t g_params[MAX_TESTS];
static struct CMUnitTest g_tests[MAX_TESTS];
static INT g_num_tests = 0;

static void build_test_array(void)
{
    static const char* UPLOS[] = {"U", "L"};
    static const char* FACTS[] = {"F", "N", "E"};
    static const char* EQUEDS[] = {"N", "Y"};

    g_num_tests = 0;

    for (INT in = 0; in < (INT)NN; in++) {
        INT n = NVAL[in];
        INT nimat = (n <= 0) ? 1 : NTYPES;

        for (INT imat = 1; imat <= nimat; imat++) {
            INT zerot = (imat >= 3 && imat <= 5);
            if (zerot && n < imat - 2) continue;

            for (INT iuplo = 0; iuplo < 2; iuplo++) {
                for (INT iequed = 0; iequed < 2; iequed++) {
                    INT nfact = (iequed == 0) ? 3 : 1;

                    for (INT ifact = 0; ifact < nfact; ifact++) {
                        if (zerot && ifact == 0) continue;

                        zdrvpo_params_t* p = &g_params[g_num_tests];
                        p->n = n;
                        p->imat = imat;
                        p->iuplo = iuplo;
                        p->ifact = ifact;
                        p->iequed = iequed;
                        snprintf(p->name, sizeof(p->name),
                                 "n%d_t%d_%s_%s_%s",
                                 n, imat, UPLOS[iuplo], FACTS[ifact], EQUEDS[iequed]);

                        g_tests[g_num_tests].name = p->name;
                        g_tests[g_num_tests].test_func = test_zdrvpo_case;
                        g_tests[g_num_tests].setup_func = NULL;
                        g_tests[g_num_tests].teardown_func = NULL;
                        g_tests[g_num_tests].initial_state = p;

                        g_num_tests++;
                    }
                }
            }
        }
    }
}

int main(void)
{
    build_test_array();
    (void)_cmocka_run_group_tests("zdrvpo", g_tests, g_num_tests,
                                   group_setup, group_teardown);
    return 0;
}
