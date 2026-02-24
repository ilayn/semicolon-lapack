/**
 * @file test_sdrvpo.c
 * @brief DDRVPO tests the driver routines SPOSV and SPOSVX.
 *
 * Port of LAPACK TESTING/LIN/ddrvpo.f to C with CMocka parameterization.
 */

#include "test_harness.h"
#include "verify.h"
#include "test_rng.h"
#include <string.h>
#include <stdio.h>
#include <math.h>

/* Test parameters - matching LAPACK dchkaa.f defaults */
static const INT NVAL[] = {0, 1, 2, 3, 5, 10, 50};
#define NN      (sizeof(NVAL) / sizeof(NVAL[0]))
#define NTYPES  9
#define NTESTS  6
#define THRESH  30.0f
#define NMAX    50
#define NRHS    2

/* Routines under test */
/* Supporting routines */
/* Verification routines */
/* Matrix generation */
/* Utilities */
typedef struct {
    INT n;
    INT imat;
    INT iuplo;      /* 0='U', 1='L' */
    INT ifact;      /* 0='F', 1='N', 2='E' */
    INT iequed;     /* 0='N', 1='Y' */
    char name[64];
} ddrvpo_params_t;

typedef struct {
    f32* A;
    f32* AFAC;
    f32* ASAV;
    f32* B;
    f32* BSAV;
    f32* X;
    f32* XACT;
    f32* S;
    f32* WORK;
    f32* RWORK;
    INT* IWORK;
    INT lwork;
} ddrvpo_workspace_t;

static ddrvpo_workspace_t* g_workspace = NULL;

static int group_setup(void** state)
{
    (void)state;
    g_workspace = malloc(sizeof(ddrvpo_workspace_t));
    if (!g_workspace) return -1;

    INT nmax = NMAX;
    INT lwork = nmax * (nmax > 3 ? nmax : 3);
    if (lwork < nmax * NRHS) lwork = nmax * NRHS;

    g_workspace->lwork = lwork;
    g_workspace->A = calloc(nmax * nmax, sizeof(f32));
    g_workspace->AFAC = calloc(nmax * nmax, sizeof(f32));
    g_workspace->ASAV = calloc(nmax * nmax, sizeof(f32));
    g_workspace->B = calloc(nmax * NRHS, sizeof(f32));
    g_workspace->BSAV = calloc(nmax * NRHS, sizeof(f32));
    g_workspace->X = calloc(nmax * NRHS, sizeof(f32));
    g_workspace->XACT = calloc(nmax * NRHS, sizeof(f32));
    g_workspace->S = calloc(nmax, sizeof(f32));
    g_workspace->WORK = calloc(lwork, sizeof(f32));
    g_workspace->RWORK = calloc(nmax + 2 * NRHS, sizeof(f32));
    g_workspace->IWORK = calloc(nmax, sizeof(INT));

    if (!g_workspace->A || !g_workspace->AFAC || !g_workspace->ASAV ||
        !g_workspace->B || !g_workspace->BSAV || !g_workspace->X ||
        !g_workspace->XACT || !g_workspace->S || !g_workspace->WORK ||
        !g_workspace->RWORK || !g_workspace->IWORK) {
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
        free(g_workspace->IWORK);
        free(g_workspace);
        g_workspace = NULL;
    }
    return 0;
}

static void run_ddrvpo_single(INT n, INT imat, INT iuplo, INT ifact, INT iequed)
{
    static const char* UPLOS[] = {"U", "L"};
    static const char* FACTS[] = {"F", "N", "E"};
    static const char* EQUEDS[] = {"N", "Y"};

    ddrvpo_workspace_t* ws = g_workspace;
    const char* uplo = UPLOS[iuplo];
    const char* fact = FACTS[ifact];
    char equed = EQUEDS[iequed][0];

    INT prefac = (fact[0] == 'F');
    INT nofact = (fact[0] == 'N');
    INT equil = (fact[0] == 'E');

    INT lda = (n > 1) ? n : 1;
    f32 result[NTESTS];
    for (INT k = 0; k < NTESTS; k++) result[k] = 0.0f;

    INT zerot = (imat >= 3 && imat <= 5);
    INT izero = 0;

    /* Set up parameters with SLATB4 */
    char type, dist;
    INT kl, ku, mode;
    f32 anorm, cndnum;
    slatb4("SPO", imat, n, n, &type, &kl, &ku, &anorm, &mode, &cndnum, &dist);

    /* Generate test matrix with SLATMS */
    uint64_t rng_state[4];
    rng_seed(rng_state, 1988 + n * 1000 + imat * 100 + iuplo * 10);
    INT info;
    slatms(n, n, &dist, &type, ws->RWORK, mode, cndnum,
           anorm, kl, ku, uplo, ws->A, lda, ws->WORK, &info, rng_state);
    if (info != 0) {
        fail_msg("SLATMS info=%d", info);
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
            /* UPLO = 'U': zero column IZERO */
            for (INT i = 0; i < izero - 1; i++) {
                ws->A[ioff + i] = 0.0f;
            }
            ioff = ioff + izero - 1;
            for (INT i = izero - 1; i < n; i++) {
                ws->A[ioff] = 0.0f;
                ioff = ioff + lda;
            }
        } else {
            /* UPLO = 'L': zero row IZERO */
            ioff = izero - 1;
            for (INT i = 0; i < izero - 1; i++) {
                ws->A[ioff] = 0.0f;
                ioff = ioff + lda;
            }
            ioff = ioff - (izero - 1);
            for (INT i = izero - 1; i < n; i++) {
                ws->A[ioff + i] = 0.0f;
            }
        }
    }

    /* Save a copy of the matrix A in ASAV */
    slacpy(uplo, n, n, ws->A, lda, ws->ASAV, lda);

    /* Skip FACT='F' for singular matrices */
    if (zerot && prefac) {
        return;
    }

    /*
     * Compute condition number RCONDC for non-singular matrices.
     *
     * In LAPACK's nested loop structure, FACT='N' reuses RCONDC from the
     * previous FACT='F' iteration. Since CMocka parameterized tests are
     * independent, we must always compute RCONDC for non-singular matrices.
     */
    f32 rcondc = 0.0f;
    f32 roldc = 0.0f;
    f32 scond = 0.0f, amax = 0.0f;

    if (zerot) {
        rcondc = 0.0f;
    } else if (n == 0) {
        /* For n=0, use RCONDC = 1/CNDNUM from slatb4 */
        rcondc = 1.0f / cndnum;
    } else {
        slacpy(uplo, n, n, ws->ASAV, lda, ws->AFAC, lda);

        if (equil || iequed > 0) {
            spoequ(n, ws->AFAC, lda, ws->S, &scond, &amax, &info);
            if (info == 0 && n > 0) {
                if (iequed > 0) {
                    scond = 0.0f;
                }
                slaqsy(uplo, n, ws->AFAC, lda, ws->S, scond, amax, &equed);
            }
        }

        if (equil) {
            roldc = rcondc;
        }

        f32 anrm = slansy("1", uplo, n, ws->AFAC, lda, ws->RWORK);

        spotrf(uplo, n, ws->AFAC, lda, &info);

        slacpy(uplo, n, n, ws->AFAC, lda, ws->A, lda);
        spotri(uplo, n, ws->A, lda, &info);

        f32 ainvnm = slansy("1", uplo, n, ws->A, lda, ws->RWORK);
        if (anrm <= 0.0f || ainvnm <= 0.0f) {
            rcondc = 1.0f;
        } else {
            rcondc = (1.0f / anrm) / ainvnm;
        }
    }
    /* For FACT='F' and FACT='N', ROLDC equals RCONDC.
     * For FACT='E', ROLDC was saved before equilibration (line 270). */
    if (!equil) {
        roldc = rcondc;
    }

    /* Restore the matrix A */
    slacpy(uplo, n, n, ws->ASAV, lda, ws->A, lda);

    /* Form exact solution and set right hand side */
    rng_seed(rng_state, 1988 + n * 1000 + imat * 100 + iuplo * 10);
    char xtype = 'N';
    slarhs("SPO", &xtype, uplo, " ", n, n, kl, ku,
           NRHS, ws->A, lda, ws->XACT, lda, ws->B, lda, &info, rng_state);
    xtype = 'C';
    slacpy("Full", n, NRHS, ws->B, lda, ws->BSAV, lda);

    /* --- Test SPOSV --- */
    if (nofact) {
        slacpy(uplo, n, n, ws->A, lda, ws->AFAC, lda);
        slacpy("Full", n, NRHS, ws->B, lda, ws->X, lda);

        sposv(uplo, n, NRHS, ws->AFAC, lda, ws->X, lda, &info);

        if (info != izero) {
            fail_msg("SPOSV info=%d expected=%d", info, izero);
            return;
        } else if (info != 0) {
            return;
        }

        /* TEST 1: Reconstruct matrix from factors */
        spot01(uplo, n, ws->A, lda, ws->AFAC, lda, ws->RWORK, &result[0]);

        /* TEST 2: Compute residual of computed solution */
        slacpy("Full", n, NRHS, ws->B, lda, ws->WORK, lda);
        spot02(uplo, n, NRHS, ws->A, lda, ws->X, lda,
               ws->WORK, lda, ws->RWORK, &result[1]);

        /* TEST 3: Check solution from generated exact solution */
        sget04(n, NRHS, ws->X, lda, ws->XACT, lda, rcondc, &result[2]);
        INT nt = 3;

        for (INT k = 0; k < nt; k++) {
            if (result[k] >= THRESH) {
                fail_msg("SPOSV UPLO=%s test %d failed: result=%e >= thresh=%e",
                         uplo, k + 1, (double)result[k], (double)THRESH);
            }
        }
    }

    /* --- Test SPOSVX --- */
    if (!prefac) {
        slaset(uplo, n, n, 0.0f, 0.0f, ws->AFAC, lda);
    }
    slaset("Full", n, NRHS, 0.0f, 0.0f, ws->X, lda);

    if (iequed > 0 && n > 0) {
        slaqsy(uplo, n, ws->A, lda, ws->S, scond, amax, &equed);
    }

    char equed_inout = equed;
    f32 rcond;
    sposvx(fact, uplo, n, NRHS, ws->A, lda, ws->AFAC, lda,
           &equed_inout, ws->S, ws->B, lda, ws->X, lda, &rcond,
           ws->RWORK, &ws->RWORK[NRHS], ws->WORK, ws->IWORK, &info);

    if (info != izero) {
        fail_msg("SPOSVX info=%d expected=%d", info, izero);
        return;
    }

    INT k1;
    if (info == 0) {
        if (!prefac) {
            /* TEST 1: Reconstruct matrix from factors */
            spot01(uplo, n, ws->A, lda, ws->AFAC, lda,
                   &ws->RWORK[2 * NRHS], &result[0]);
            k1 = 1;
        } else {
            k1 = 2;
        }

        /* TEST 2: Compute residual of computed solution */
        slacpy("Full", n, NRHS, ws->BSAV, lda, ws->WORK, lda);
        spot02(uplo, n, NRHS, ws->ASAV, lda, ws->X, lda,
               ws->WORK, lda, &ws->RWORK[2 * NRHS], &result[1]);

        /* TEST 3: Check solution from generated exact solution */
        if (nofact || (prefac && (equed_inout == 'N' || equed_inout == 'n'))) {
            sget04(n, NRHS, ws->X, lda, ws->XACT, lda, rcondc, &result[2]);
        } else {
            sget04(n, NRHS, ws->X, lda, ws->XACT, lda, roldc, &result[2]);
        }

        /* TEST 4-5: Check error bounds from iterative refinement */
        spot05(uplo, n, NRHS, ws->ASAV, lda, ws->B, lda,
               ws->X, lda, ws->XACT, lda,
               ws->RWORK, &ws->RWORK[NRHS], &result[3]);
    } else {
        k1 = 6;
    }

    /* TEST 6: Compare RCOND from SPOSVX with computed value */
    result[5] = sget06(rcond, rcondc);

    /* Check results */
    if (info == 0) {
        for (INT k = k1 - 1; k < NTESTS; k++) {
            if (result[k] >= THRESH) {
                fail_msg("SPOSVX FACT=%s UPLO=%s EQUED=%c test %d: result=%e >= thresh=%e",
                         fact, uplo, equed_inout, k + 1, (double)result[k], (double)THRESH);
            }
        }
    } else {
        /* TRFCON case: only check tests 1, 6 */
        if (!prefac && result[0] >= THRESH) {
            fail_msg("SPOSVX FACT=%s UPLO=%s EQUED=%c test 1: result=%e >= thresh=%e",
                     fact, uplo, equed_inout, (double)result[0], (double)THRESH);
        }
        if (result[5] >= THRESH) {
            fail_msg("SPOSVX FACT=%s UPLO=%s EQUED=%c test 6: result=%e >= thresh=%e",
                     fact, uplo, equed_inout, (double)result[5], (double)THRESH);
        }
    }
}

static void test_ddrvpo_case(void** state)
{
    ddrvpo_params_t* p = *state;
    run_ddrvpo_single(p->n, p->imat, p->iuplo, p->ifact, p->iequed);
}

#define MAX_TESTS 3000

static ddrvpo_params_t g_params[MAX_TESTS];
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

                        ddrvpo_params_t* p = &g_params[g_num_tests];
                        p->n = n;
                        p->imat = imat;
                        p->iuplo = iuplo;
                        p->ifact = ifact;
                        p->iequed = iequed;
                        snprintf(p->name, sizeof(p->name),
                                 "n%d_t%d_%s_%s_%s",
                                 n, imat, UPLOS[iuplo], FACTS[ifact], EQUEDS[iequed]);

                        g_tests[g_num_tests].name = p->name;
                        g_tests[g_num_tests].test_func = test_ddrvpo_case;
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
    return _cmocka_run_group_tests("ddrvpo", g_tests, g_num_tests,
                                   group_setup, group_teardown);
}
