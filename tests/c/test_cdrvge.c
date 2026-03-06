/**
 * @file test_cdrvge.c
 * @brief ZDRVGE tests the driver routines CGESV and CGESVX.
 *
 * Port of LAPACK TESTING/LIN/zdrvge.f to C with CMocka parameterization.
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
#define NTYPES  11
#define NTESTS  7
#define NTRAN   3
#define THRESH  30.0f
#define NMAX    50
#define NRHS    2

typedef struct {
    INT n;
    INT imat;
    INT ifact;      /* 0='F', 1='N', 2='E' */
    INT itran;      /* 0='N', 1='T', 2='C' */
    INT iequed;     /* 0='N', 1='R', 2='C', 3='B' */
    char name[64];
} zdrvge_params_t;

typedef struct {
    c64* A;
    c64* AFAC;
    c64* ASAV;
    c64* B;
    c64* BSAV;
    c64* X;
    c64* XACT;
    f32* S;
    c64* WORK;
    f32* RWORK;
    INT* IWORK;
    INT lwork;
} zdrvge_workspace_t;

static zdrvge_workspace_t* g_workspace = NULL;

static int group_setup(void** state)
{
    (void)state;
    g_workspace = malloc(sizeof(zdrvge_workspace_t));
    if (!g_workspace) return -1;

    INT nmax = NMAX;
    INT lwork = nmax * (nmax > 3 ? nmax : 3);
    if (lwork < nmax * NRHS) lwork = nmax * NRHS;

    g_workspace->lwork = lwork;
    g_workspace->A = calloc(nmax * nmax, sizeof(c64));
    g_workspace->AFAC = calloc(nmax * nmax, sizeof(c64));
    g_workspace->ASAV = calloc(nmax * nmax, sizeof(c64));
    g_workspace->B = calloc(nmax * NRHS, sizeof(c64));
    g_workspace->BSAV = calloc(nmax * NRHS, sizeof(c64));
    g_workspace->X = calloc(nmax * NRHS, sizeof(c64));
    g_workspace->XACT = calloc(nmax * NRHS, sizeof(c64));
    g_workspace->S = calloc(2 * nmax, sizeof(f32));
    g_workspace->WORK = calloc(lwork, sizeof(c64));
    g_workspace->RWORK = calloc(2 * NRHS + 2 * nmax, sizeof(f32));
    g_workspace->IWORK = calloc(2 * nmax, sizeof(INT));

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

static f32 compute_rcond(zdrvge_workspace_t* ws, INT n, INT lda, const char* norm)
{
    if (n == 0) return 1.0f;

    INT info;
    f32 anrm = clange(norm, n, n, ws->AFAC, lda, ws->RWORK);

    cgetrf(n, n, ws->AFAC, lda, ws->IWORK, &info);
    if (info != 0) return 0.0f;

    clacpy("Full", n, n, ws->AFAC, lda, ws->A, lda);
    INT lwork_getri = NMAX * 3;
    cgetri(n, ws->A, lda, ws->IWORK, ws->WORK, lwork_getri, &info);
    if (info != 0) return 0.0f;

    f32 ainvnm = clange(norm, n, n, ws->A, lda, ws->RWORK);
    if (anrm <= 0.0f || ainvnm <= 0.0f) return 1.0f;

    return (1.0f / anrm) / ainvnm;
}

static void run_zdrvge_single(INT n, INT imat, INT ifact, INT itran, INT iequed)
{
    static const char* FACTS[] = {"F", "N", "E"};
    static const char* TRANSS[] = {"N", "T", "C"};
    static const char* EQUEDS[] = {"N", "R", "C", "B"};

    zdrvge_workspace_t* ws = g_workspace;
    const char* fact = FACTS[ifact];
    const char* trans = TRANSS[itran];
    char equed = EQUEDS[iequed][0];

    INT prefac = (fact[0] == 'F');
    INT nofact = (fact[0] == 'N');
    INT equil = (fact[0] == 'E');

    INT lda = (n > 1) ? n : 1;
    f32 result[NTESTS];
    for (INT k = 0; k < NTESTS; k++) result[k] = 0.0f;

    char type, dist;
    INT kl, ku, mode;
    f32 anorm, cndnum;
    clatb4("CGE", imat, n, n, &type, &kl, &ku, &anorm, &mode, &cndnum, &dist);

    uint64_t rng_state[4];
    rng_seed(rng_state, 1988 + n * 1000 + imat * 100);
    INT info;
    clatms(n, n, &dist, &type, ws->RWORK, mode, cndnum,
           anorm, kl, ku, "N", ws->A, lda, ws->WORK, &info, rng_state);
    if (info != 0) {
        fail_msg("CLATMS info=%d", info);
        return;
    }

    INT izero = 0;
    INT zerot = (imat >= 5 && imat <= 7);
    if (zerot) {
        if (imat == 5) izero = 1;
        else if (imat == 6) izero = n;
        else izero = n / 2 + 1;

        INT ioff = (izero - 1) * lda;
        if (imat < 7) {
            for (INT i = 0; i < n; i++) ws->A[ioff + i] = CMPLXF(0.0f, 0.0f);
        } else {
            claset("Full", n, n - izero + 1, CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
                   &ws->A[ioff], lda);
        }
    }

    clacpy("Full", n, n, ws->A, lda, ws->ASAV, lda);

    if (zerot && prefac) {
        return;
    }

    f32 rcondo = 0.0f, rcondi = 0.0f;
    f32 roldo = 0.0f, roldi = 0.0f;
    f32 rowcnd = 0.0f, colcnd = 0.0f, amax = 0.0f;

    if (zerot) {
        rcondo = 0.0f;
        rcondi = 0.0f;
    } else if (n > 0) {
        clacpy("Full", n, n, ws->ASAV, lda, ws->AFAC, lda);
        roldo = compute_rcond(ws, n, lda, "1");

        clacpy("Full", n, n, ws->ASAV, lda, ws->AFAC, lda);
        roldi = compute_rcond(ws, n, lda, "I");

        if (equil || iequed > 0) {
            clacpy("Full", n, n, ws->ASAV, lda, ws->AFAC, lda);

            cgeequ(n, n, ws->AFAC, lda, ws->S, &ws->S[n],
                   &rowcnd, &colcnd, &amax, &info);

            if (info == 0 && n > 0) {
                if (equed == 'R' || equed == 'r') {
                    rowcnd = 0.0f; colcnd = 1.0f;
                } else if (equed == 'C' || equed == 'c') {
                    rowcnd = 1.0f; colcnd = 0.0f;
                } else if (equed == 'B' || equed == 'b') {
                    rowcnd = 0.0f; colcnd = 0.0f;
                }

                char equed_out;
                claqge(n, n, ws->AFAC, lda, ws->S, &ws->S[n],
                       rowcnd, colcnd, amax, &equed_out);

                equed = equed_out;
            }

            rcondo = compute_rcond(ws, n, lda, "1");

            clacpy("Full", n, n, ws->ASAV, lda, ws->AFAC, lda);
            if (info == 0 && n > 0) {
                char equed_out;
                claqge(n, n, ws->AFAC, lda, ws->S, &ws->S[n],
                       rowcnd, colcnd, amax, &equed_out);
            }
            rcondi = compute_rcond(ws, n, lda, "I");
        } else {
            rcondo = roldo;
            rcondi = roldi;
        }
    }

    f32 rcondc, roldc;
    if (n == 0) {
        rcondc = 1.0f / cndnum;
        roldc = rcondc;
    } else {
        rcondc = (itran == 0) ? rcondo : rcondi;
        roldc = (itran == 0) ? roldo : roldi;
    }

    clacpy("Full", n, n, ws->ASAV, lda, ws->A, lda);

    rng_seed(rng_state, 1988 + n * 1000 + imat * 100 + itran);
    char xtype = 'N';
    clarhs("CGE", &xtype, "Full", trans, n, n, kl, ku,
           NRHS, ws->A, lda, ws->XACT, lda, ws->B, lda, &info, rng_state);
    clacpy("Full", n, NRHS, ws->B, lda, ws->BSAV, lda);

    /* --- Test CGESV --- */
    if (nofact && itran == 0) {
        clacpy("Full", n, n, ws->A, lda, ws->AFAC, lda);
        clacpy("Full", n, NRHS, ws->B, lda, ws->X, lda);

        cgesv(n, NRHS, ws->AFAC, lda, ws->IWORK, ws->X, lda, &info);

        if (info != izero) {
            fail_msg("CGESV info=%d expected=%d", info, izero);
        }

        cget01(n, n, ws->A, lda, ws->AFAC, lda, ws->IWORK,
               ws->RWORK, &result[0]);

        INT nt = 1;
        if (izero == 0) {
            clacpy("Full", n, NRHS, ws->B, lda, ws->WORK, lda);
            cget02("N", n, n, NRHS, ws->A, lda, ws->X, lda,
                   ws->WORK, lda, ws->RWORK, &result[1]);

            cget04(n, NRHS, ws->X, lda, ws->XACT, lda, rcondc, &result[2]);
            nt = 3;
        }

        for (INT k = 0; k < nt; k++) {
            if (result[k] >= THRESH) {
                fail_msg("CGESV test %d failed: result=%e >= thresh=%e",
                         k + 1, (double)result[k], (double)THRESH);
            }
        }
    }

    /* --- Test CGESVX --- */

    if (!prefac) {
        claset("Full", n, n, CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), ws->AFAC, lda);
    }
    claset("Full", n, NRHS, CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), ws->X, lda);

    if (iequed > 0 && n > 0) {
        char equed_out;
        claqge(n, n, ws->A, lda, ws->S, &ws->S[n],
               rowcnd, colcnd, amax, &equed_out);
        equed = equed_out;
    }

    char equed_inout = equed;
    f32 rcond;
    cgesvx(fact, trans, n, NRHS, ws->A, lda, ws->AFAC, lda,
           ws->IWORK, &equed_inout, ws->S, &ws->S[n],
           ws->B, lda, ws->X, lda, &rcond,
           ws->RWORK, &ws->RWORK[NRHS], ws->WORK,
           &ws->RWORK[2 * NRHS], &info);

    if (info != izero) {
        if (!(zerot && info > 0 && info <= n)) {
            fail_msg("CGESVX info=%d expected=%d", info, izero);
        }
    }

    /* TEST 7: Compare RPVGRW */
    f32 rpvgrw;
    if (info != 0 && info <= n) {
        rpvgrw = clantr("M", "U", "N", info, info, ws->AFAC, lda, ws->RWORK);
        if (rpvgrw == 0.0f) {
            rpvgrw = 1.0f;
        } else {
            rpvgrw = clange("M", n, info, ws->A, lda, ws->RWORK) / rpvgrw;
        }
    } else {
        rpvgrw = clantr("M", "U", "N", n, n, ws->AFAC, lda, ws->RWORK);
        if (rpvgrw == 0.0f) {
            rpvgrw = 1.0f;
        } else {
            rpvgrw = clange("M", n, n, ws->A, lda, ws->RWORK) / rpvgrw;
        }
    }
    f32 rpvgrw_svx = ws->RWORK[2 * NRHS];
    result[6] = fabsf(rpvgrw - rpvgrw_svx) / fmaxf(rpvgrw_svx, rpvgrw) / slamch("E");

    INT k1;
    if (!prefac) {
        cget01(n, n, ws->A, lda, ws->AFAC, lda, ws->IWORK,
               &ws->RWORK[2 * NRHS], &result[0]);
        k1 = 0;
    } else {
        k1 = 1;
    }

    INT trfcon;
    if (info == 0) {
        trfcon = 0;

        clacpy("Full", n, NRHS, ws->BSAV, lda, ws->WORK, lda);
        cget02(trans, n, n, NRHS, ws->ASAV, lda, ws->X, lda,
               ws->WORK, lda, &ws->RWORK[2 * NRHS], &result[1]);

        f32 rcond_for_get04;
        if (nofact || (prefac && (equed == 'N' || equed == 'n'))) {
            rcond_for_get04 = rcondc;
        } else {
            rcond_for_get04 = roldc;
        }
        cget04(n, NRHS, ws->X, lda, ws->XACT, lda, rcond_for_get04, &result[2]);

        cget07(trans, n, NRHS, ws->ASAV, lda, ws->B, lda,
               ws->X, lda, ws->XACT, lda,
               ws->RWORK, 1, &ws->RWORK[NRHS], &result[3]);
    } else {
        trfcon = 1;
    }

    result[5] = sget06(rcond, rcondc);

    if (!trfcon) {
        for (INT k = k1; k < NTESTS; k++) {
            if (result[k] >= THRESH) {
                fail_msg("CGESVX FACT=%s TRANS=%s EQUED=%c test %d: result=%e >= thresh=%e",
                         fact, trans, equed, k + 1, (double)result[k], (double)THRESH);
            }
        }
    } else {
        if (!prefac && result[0] >= THRESH) {
            fail_msg("CGESVX FACT=%s TRANS=%s EQUED=%c test 1: result=%e >= thresh=%e",
                     fact, trans, equed, (double)result[0], (double)THRESH);
        }
        if (result[5] >= THRESH) {
            fail_msg("CGESVX FACT=%s TRANS=%s EQUED=%c test 6: result=%e >= thresh=%e",
                     fact, trans, equed, (double)result[5], (double)THRESH);
        }
        if (result[6] >= THRESH) {
            fail_msg("CGESVX FACT=%s TRANS=%s EQUED=%c test 7: result=%e >= thresh=%e",
                     fact, trans, equed, (double)result[6], (double)THRESH);
        }
    }
}

static void test_zdrvge_case(void** state)
{
    zdrvge_params_t* p = *state;
    run_zdrvge_single(p->n, p->imat, p->ifact, p->itran, p->iequed);
}

#define MAX_TESTS 3000

static zdrvge_params_t g_params[MAX_TESTS];
static struct CMUnitTest g_tests[MAX_TESTS];
static INT g_num_tests = 0;

static void build_test_array(void)
{
    static const char* FACTS[] = {"F", "N", "E"};
    static const char* TRANSS[] = {"N", "T", "C"};
    static const char* EQUEDS[] = {"N", "R", "C", "B"};

    g_num_tests = 0;

    for (INT in = 0; in < (INT)NN; in++) {
        INT n = NVAL[in];
        INT nimat = (n <= 0) ? 1 : NTYPES;

        for (INT imat = 1; imat <= nimat; imat++) {
            INT zerot = (imat >= 5 && imat <= 7);
            if (zerot && n < imat - 4) continue;

            for (INT iequed = 0; iequed < 4; iequed++) {
                INT nfact = (iequed == 0) ? 3 : 1;

                for (INT ifact = 0; ifact < nfact; ifact++) {
                    if (zerot && ifact == 0) continue;

                    for (INT itran = 0; itran < NTRAN; itran++) {
                        zdrvge_params_t* p = &g_params[g_num_tests];
                        p->n = n;
                        p->imat = imat;
                        p->ifact = ifact;
                        p->itran = itran;
                        p->iequed = iequed;
                        snprintf(p->name, sizeof(p->name),
                                 "n%d_t%d_%s_%s_%s",
                                 n, imat, FACTS[ifact], TRANSS[itran], EQUEDS[iequed]);

                        g_tests[g_num_tests].name = p->name;
                        g_tests[g_num_tests].test_func = test_zdrvge_case;
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
    return _cmocka_run_group_tests("zdrvge", g_tests, g_num_tests,
                                   group_setup, group_teardown);
}
