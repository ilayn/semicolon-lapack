/**
 * @file test_zdrvgb.c
 * @brief ZDRVGB tests the driver routines ZGBSV and ZGBSVX.
 *
 * Port of LAPACK TESTING/LIN/zdrvgb.f to C with CMocka parameterization.
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
#define NTYPES  8
#define NTESTS  7
#define NTRAN   3
#define THRESH  30.0
#define NMAX    50
#define NRHS    2
#define NBW     4

/* Maximum workspace dimensions */
#define LA      ((2*NMAX - 1) * NMAX)
#define LAFB    ((3*NMAX - 2) * NMAX)

typedef struct {
    INT n;
    INT kl;
    INT ku;
    INT imat;
    INT ifact;     /* 0='F', 1='N', 2='E' */
    INT itran;     /* 0='N', 1='T', 2='C' */
    INT iequed;    /* 0='N', 1='R', 2='C', 3='B' */
    char name[96];
} zdrvgb_params_t;

typedef struct {
    c128* A;
    c128* AFB;
    c128* ASAV;
    c128* B;
    c128* BSAV;
    c128* X;
    c128* XACT;
    f64* S;
    c128* WORK;
    f64* RWORK;
    INT* IWORK;
} zdrvgb_workspace_t;

static zdrvgb_workspace_t* g_workspace = NULL;

static int group_setup(void** state)
{
    (void)state;
    g_workspace = malloc(sizeof(zdrvgb_workspace_t));
    if (!g_workspace) return -1;

    INT lwork = NMAX * NMAX;
    if (lwork < 3 * NMAX) lwork = 3 * NMAX;
    if (lwork < NMAX * NRHS) lwork = NMAX * NRHS;

    g_workspace->A     = calloc(LA, sizeof(c128));
    g_workspace->AFB   = calloc(LAFB, sizeof(c128));
    g_workspace->ASAV  = calloc(LA, sizeof(c128));
    g_workspace->B     = calloc(NMAX * NRHS, sizeof(c128));
    g_workspace->BSAV  = calloc(NMAX * NRHS, sizeof(c128));
    g_workspace->X     = calloc(NMAX * NRHS, sizeof(c128));
    g_workspace->XACT  = calloc(NMAX * NRHS, sizeof(c128));
    g_workspace->S     = calloc(2 * NMAX, sizeof(f64));
    g_workspace->WORK  = calloc(lwork, sizeof(c128));
    g_workspace->RWORK = calloc(2 * NRHS + 2 * NMAX, sizeof(f64));
    g_workspace->IWORK = calloc(2 * NMAX, sizeof(INT));

    if (!g_workspace->A || !g_workspace->AFB || !g_workspace->ASAV ||
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
        free(g_workspace->AFB);
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

static void copy_band_to_factor(const c128* A, INT lda,
                                 c128* AFB, INT ldafb,
                                 INT kl, INT ku, INT n)
{
    for (INT j = 0; j < n; j++) {
        for (INT i = 0; i < kl + ku + 1; i++) {
            AFB[(kl + i) + j * ldafb] = A[i + j * lda];
        }
    }
}

static void run_zdrvgb_single(INT n, INT kl, INT ku, INT imat,
                               INT ifact, INT itran, INT iequed)
{
    static const char* FACTS[]  = {"F", "N", "E"};
    static const char* TRANSS[] = {"N", "T", "C"};
    static const char* EQUEDS[] = {"N", "R", "C", "B"};

    zdrvgb_workspace_t* ws = g_workspace;
    const char* fact  = FACTS[ifact];
    const char* trans = TRANSS[itran];
    char equed = EQUEDS[iequed][0];

    INT prefac = (fact[0] == 'F');
    INT nofact = (fact[0] == 'N');
    INT equil  = (fact[0] == 'E');

    INT lda   = kl + ku + 1;
    INT ldafb = 2 * kl + ku + 1;
    INT ldb   = (n > 1) ? n : 1;

    f64 result[NTESTS];
    for (INT k = 0; k < NTESTS; k++) result[k] = 0.0;

    char type, dist;
    INT kl_out = kl, ku_out = ku, mode;
    f64 anorm, cndnum;
    zlatb4("ZGB", imat, n, n, &type, &kl_out, &ku_out, &anorm, &mode,
           &cndnum, &dist);

    uint64_t rng_state[4];
    rng_seed(rng_state, 1988 + n * 1000 + kl * 100 + ku * 10 + imat);
    INT info;
    zlatms(n, n, &dist, &type, ws->RWORK, mode, cndnum,
           anorm, kl, ku, "Z", ws->A, lda, ws->WORK, &info, rng_state);
    if (info != 0) {
        fail_msg("ZLATMS info=%d (n=%d kl=%d ku=%d imat=%d)", info, n, kl, ku, imat);
        return;
    }

    INT izero = 0;
    INT zerot = (imat >= 2 && imat <= 4);
    if (zerot) {
        if (imat == 2)
            izero = 1;
        else if (imat == 3)
            izero = n;
        else
            izero = n / 2 + 1;

        if (imat < 4) {
            INT ioff = (izero - 1) * lda;
            INT i1_f = ku + 2 - izero;
            INT i2_f = ku + 1 + (n - izero);
            INT i1 = (i1_f > 1) ? i1_f - 1 : 0;
            INT i2 = (i2_f < kl + ku + 1) ? i2_f : kl + ku + 1;
            for (INT i = i1; i < i2; i++)
                ws->A[ioff + i] = CMPLX(0.0, 0.0);
        } else {
            INT ioff = (izero - 1) * lda;
            for (INT j = izero; j <= n; j++) {
                INT i1_f = ku + 2 - j;
                INT i2_f = ku + 1 + (n - j);
                INT i1 = (i1_f > 1) ? i1_f - 1 : 0;
                INT i2 = (i2_f < kl + ku + 1) ? i2_f : kl + ku + 1;
                for (INT i = i1; i < i2; i++)
                    ws->A[ioff + i] = CMPLX(0.0, 0.0);
                ioff += lda;
            }
        }
    }

    zlacpy("Full", kl + ku + 1, n, ws->A, lda, ws->ASAV, lda);

    if (zerot && prefac) {
        return;
    }

    f64 rcondo = 0.0, rcondi = 0.0;
    f64 roldo = 0.0, roldi = 0.0;
    f64 rowcnd = 0.0, colcnd = 0.0, amax = 0.0;

    if (zerot) {
        rcondo = 0.0;
        rcondi = 0.0;
    } else {
        copy_band_to_factor(ws->ASAV, lda, ws->AFB, ldafb, kl, ku, n);

        f64 anormo = zlangb("1", n, kl, ku, &ws->AFB[kl], ldafb, ws->RWORK);
        f64 anormi = zlangb("I", n, kl, ku, &ws->AFB[kl], ldafb, ws->RWORK);

        zgbtrf(n, n, kl, ku, ws->AFB, ldafb, ws->IWORK, &info);

        zlaset("Full", n, n, CMPLX(0.0, 0.0), CMPLX(1.0, 0.0), ws->WORK, ldb);
        zgbtrs("N", n, kl, ku, n, ws->AFB, ldafb,
               ws->IWORK, ws->WORK, ldb, &info);

        f64 ainvnm = zlange("1", n, n, ws->WORK, ldb, ws->RWORK);
        if (anormo <= 0.0 || ainvnm <= 0.0)
            rcondo = 1.0;
        else
            rcondo = (1.0 / anormo) / ainvnm;

        ainvnm = zlange("I", n, n, ws->WORK, ldb, ws->RWORK);
        if (anormi <= 0.0 || ainvnm <= 0.0)
            rcondi = 1.0;
        else
            rcondi = (1.0 / anormi) / ainvnm;

        roldo = rcondo;
        roldi = rcondi;

        if (equil || iequed > 0) {
            copy_band_to_factor(ws->ASAV, lda, ws->AFB, ldafb, kl, ku, n);

            zgbequ(n, n, kl, ku, &ws->AFB[kl], ldafb,
                   ws->S, &ws->S[n], &rowcnd, &colcnd, &amax, &info);
            if (info == 0 && n > 0) {
                if (equed == 'R' || equed == 'r') {
                    rowcnd = 0.0; colcnd = 1.0;
                } else if (equed == 'C' || equed == 'c') {
                    rowcnd = 1.0; colcnd = 0.0;
                } else if (equed == 'B' || equed == 'b') {
                    rowcnd = 0.0; colcnd = 0.0;
                }

                zlaqgb(n, n, kl, ku, &ws->AFB[kl], ldafb,
                       ws->S, &ws->S[n], rowcnd, colcnd, amax, &equed);
            }

            anormo = zlangb("1", n, kl, ku, &ws->AFB[kl], ldafb, ws->RWORK);
            anormi = zlangb("I", n, kl, ku, &ws->AFB[kl], ldafb, ws->RWORK);

            zgbtrf(n, n, kl, ku, ws->AFB, ldafb, ws->IWORK, &info);

            zlaset("Full", n, n, CMPLX(0.0, 0.0), CMPLX(1.0, 0.0), ws->WORK, ldb);
            zgbtrs("N", n, kl, ku, n, ws->AFB, ldafb,
                   ws->IWORK, ws->WORK, ldb, &info);

            ainvnm = zlange("1", n, n, ws->WORK, ldb, ws->RWORK);
            if (anormo <= 0.0 || ainvnm <= 0.0)
                rcondo = 1.0;
            else
                rcondo = (1.0 / anormo) / ainvnm;

            ainvnm = zlange("I", n, n, ws->WORK, ldb, ws->RWORK);
            if (anormi <= 0.0 || ainvnm <= 0.0)
                rcondi = 1.0;
            else
                rcondi = (1.0 / anormi) / ainvnm;
        }
    }

    f64 rcondc = (itran == 0) ? rcondo : rcondi;

    zlacpy("Full", kl + ku + 1, n, ws->ASAV, lda, ws->A, lda);

    rng_seed(rng_state, 1988 + n * 1000 + kl * 100 + ku * 10 + imat + itran);
    char xtype = 'N';
    zlarhs("ZGB", &xtype, "Full", trans, n, n, kl, ku,
           NRHS, ws->A, lda, ws->XACT, ldb, ws->B, ldb, &info, rng_state);
    zlacpy("Full", n, NRHS, ws->B, ldb, ws->BSAV, ldb);

    /* --- Test ZGBSV --- */
    if (nofact && itran == 0) {
        copy_band_to_factor(ws->A, lda, ws->AFB, ldafb, kl, ku, n);
        zlacpy("Full", n, NRHS, ws->B, ldb, ws->X, ldb);

        zgbsv(n, kl, ku, NRHS, ws->AFB, ldafb, ws->IWORK, ws->X, ldb, &info);

        if (info != izero) {
            fail_msg("ZGBSV info=%d expected=%d (n=%d kl=%d ku=%d imat=%d)",
                     info, izero, n, kl, ku, imat);
        }

        zgbt01(n, n, kl, ku, ws->A, lda, ws->AFB, ldafb,
               ws->IWORK, ws->WORK, &result[0]);

        INT nt = 1;
        if (izero == 0) {
            zlacpy("Full", n, NRHS, ws->B, ldb, ws->WORK, ldb);
            zgbt02("N", n, n, kl, ku, NRHS, ws->A, lda,
                   ws->X, ldb, ws->WORK, ldb, ws->RWORK, &result[1]);

            zget04(n, NRHS, ws->X, ldb, ws->XACT, ldb, rcondc, &result[2]);
            nt = 3;
        }

        for (INT k = 0; k < nt; k++) {
            if (result[k] >= THRESH) {
                fail_msg("ZGBSV n=%d kl=%d ku=%d type %d test %d: result=%e >= thresh=%e",
                         n, kl, ku, imat, k + 1, result[k], THRESH);
            }
        }
    }

    /* --- Test ZGBSVX --- */

    if (!prefac)
        zlaset("Full", 2 * kl + ku + 1, n, CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
               ws->AFB, ldafb);
    zlaset("Full", n, NRHS, CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), ws->X, ldb);

    if (iequed > 0 && n > 0) {
        zlaqgb(n, n, kl, ku, ws->A, lda, ws->S, &ws->S[n],
               rowcnd, colcnd, amax, &equed);
    }

    char equed_inout = equed;
    f64 rcond;
    zgbsvx(fact, trans, n, kl, ku, NRHS, ws->A, lda, ws->AFB, ldafb,
           ws->IWORK, &equed_inout, ws->S, &ws->S[n],
           ws->B, ldb, ws->X, ldb, &rcond,
           ws->RWORK, &ws->RWORK[NRHS], ws->WORK, &ws->RWORK[2 * NRHS], &info);

    if (info != izero) {
        if (!(zerot && info > 0 && info <= n)) {
            fail_msg("ZGBSVX info=%d expected=%d (n=%d kl=%d ku=%d imat=%d fact=%s trans=%s)",
                     info, izero, n, kl, ku, imat, fact, trans);
        }
    }

    f64 rpvgrw_svx = ws->RWORK[2 * NRHS];
    f64 rpvgrw;

    if (info != 0 && info <= n) {
        f64 anrmpv = 0.0;
        for (INT j = 0; j < info; j++) {
            INT i1_f = ku + 2 - (j + 1);
            INT i2_f = n + ku + 1 - (j + 1);
            INT i1 = (i1_f > 1) ? i1_f - 1 : 0;
            INT i2 = (i2_f < kl + ku + 1) ? i2_f : kl + ku + 1;
            for (INT i = i1; i < i2; i++) {
                f64 val = cabs(ws->A[i + j * lda]);
                if (val > anrmpv) anrmpv = val;
            }
        }
        INT kband = (info - 1 < kl + ku) ? info - 1 : kl + ku;
        INT afb_off_f = kl + ku + 2 - info;
        INT afb_off = (afb_off_f > 1) ? afb_off_f - 1 : 0;
        rpvgrw = zlantb("M", "U", "N", info, kband,
                        &ws->AFB[afb_off], ldafb, ws->RWORK);
        if (rpvgrw == 0.0)
            rpvgrw = 1.0;
        else
            rpvgrw = anrmpv / rpvgrw;
    } else {
        rpvgrw = zlantb("M", "U", "N", n, kl + ku,
                        ws->AFB, ldafb, ws->RWORK);
        if (rpvgrw == 0.0) {
            rpvgrw = 1.0;
        } else {
            rpvgrw = zlangb("M", n, kl, ku, ws->A, lda, ws->RWORK) / rpvgrw;
        }
    }
    f64 denom = fmax(rpvgrw_svx, rpvgrw);
    if (denom > 0.0)
        result[6] = fabs(rpvgrw - rpvgrw_svx) / denom / dlamch("E");
    else
        result[6] = 0.0;

    INT k1;
    if (!prefac) {
        zgbt01(n, n, kl, ku, ws->A, lda, ws->AFB, ldafb,
               ws->IWORK, ws->WORK, &result[0]);
        k1 = 0;
    } else {
        k1 = 1;
    }

    INT trfcon;
    if (info == 0) {
        trfcon = 0;

        zlacpy("Full", n, NRHS, ws->BSAV, ldb, ws->WORK, ldb);
        zgbt02(trans, n, n, kl, ku, NRHS, ws->ASAV, lda,
               ws->X, ldb, ws->WORK, ldb, &ws->RWORK[2 * NRHS],
               &result[1]);

        f64 rcond_for_get04;
        if (nofact || (prefac && (equed_inout == 'N' || equed_inout == 'n'))) {
            rcond_for_get04 = rcondc;
        } else {
            f64 roldc = (itran == 0) ? roldo : roldi;
            rcond_for_get04 = roldc;
        }
        zget04(n, NRHS, ws->X, ldb, ws->XACT, ldb,
               rcond_for_get04, &result[2]);

        zgbt05(trans, n, kl, ku, NRHS, ws->ASAV, lda,
               ws->B, ldb, ws->X, ldb, ws->XACT, ldb,
               ws->RWORK, &ws->RWORK[NRHS], &result[3]);
    } else {
        trfcon = 1;
    }

    result[5] = dget06(rcond, rcondc);

    if (!trfcon) {
        for (INT k = k1; k < NTESTS; k++) {
            if (result[k] >= THRESH) {
                if (prefac) {
                    fail_msg("ZGBSVX FACT=%s TRANS=%s n=%d kl=%d ku=%d EQUED=%c "
                             "type %d test %d: result=%e >= thresh=%e",
                             fact, trans, n, kl, ku, equed_inout,
                             imat, k + 1, result[k], THRESH);
                } else {
                    fail_msg("ZGBSVX FACT=%s TRANS=%s n=%d kl=%d ku=%d "
                             "type %d test %d: result=%e >= thresh=%e",
                             fact, trans, n, kl, ku,
                             imat, k + 1, result[k], THRESH);
                }
            }
        }
    } else {
        if (!prefac && result[0] >= THRESH) {
            fail_msg("ZGBSVX FACT=%s TRANS=%s n=%d kl=%d ku=%d "
                     "type %d test 1: result=%e >= thresh=%e",
                     fact, trans, n, kl, ku, imat, result[0], THRESH);
        }
        if (result[5] >= THRESH) {
            fail_msg("ZGBSVX FACT=%s TRANS=%s n=%d kl=%d ku=%d "
                     "type %d test 6: result=%e >= thresh=%e",
                     fact, trans, n, kl, ku, imat, result[5], THRESH);
        }
        if (result[6] >= THRESH) {
            fail_msg("ZGBSVX FACT=%s TRANS=%s n=%d kl=%d ku=%d "
                     "type %d test 7: result=%e >= thresh=%e",
                     fact, trans, n, kl, ku, imat, result[6], THRESH);
        }
    }
}

static void test_zdrvgb_case(void** state)
{
    zdrvgb_params_t* p = *state;
    run_zdrvgb_single(p->n, p->kl, p->ku, p->imat,
                      p->ifact, p->itran, p->iequed);
}

static void get_klku_values(INT n, INT vals[NBW])
{
    vals[0] = 0;
    vals[1] = (n > 0) ? n - 1 : 0;
    vals[2] = (3 * n - 1) / 4;
    vals[3] = (n + 1) / 4;
}

static INT build_test_array(zdrvgb_params_t* params, struct CMUnitTest* tests)
{
    static const char* FACTS[]  = {"F", "N", "E"};
    static const char* TRANSS[] = {"N", "T", "C"};
    static const char* EQUEDS[] = {"N", "R", "C", "B"};

    INT idx = 0;

    for (INT in = 0; in < (INT)NN; in++) {
        INT n = NVAL[in];
        INT nkl = (n > 4) ? 4 : ((n > 0) ? n : 1);
        INT nku = nkl;
        INT nimat = (n <= 0) ? 1 : NTYPES;

        INT klval[NBW], kuval[NBW];
        get_klku_values(n, klval);
        get_klku_values(n, kuval);

        for (INT ikl = 0; ikl < nkl; ikl++) {
            INT kl = klval[ikl];
            for (INT iku = 0; iku < nku; iku++) {
                INT ku = kuval[iku];
                INT lda   = kl + ku + 1;
                INT ldafb = 2 * kl + ku + 1;

                if (lda * n > LA || ldafb * n > LAFB)
                    continue;

                for (INT imat = 1; imat <= nimat; imat++) {
                    INT zerot = (imat >= 2 && imat <= 4);
                    if (zerot && n < imat - 1)
                        continue;

                    for (INT iequed = 0; iequed < 4; iequed++) {
                        INT nfact = (iequed == 0) ? 3 : 1;

                        for (INT ifact = 0; ifact < nfact; ifact++) {
                            if (zerot && ifact == 0)
                                continue;

                            for (INT itran = 0; itran < NTRAN; itran++) {
                                if (params && tests) {
                                    zdrvgb_params_t* p = &params[idx];
                                    p->n      = n;
                                    p->kl     = kl;
                                    p->ku     = ku;
                                    p->imat   = imat;
                                    p->ifact  = ifact;
                                    p->itran  = itran;
                                    p->iequed = iequed;
                                    snprintf(p->name, sizeof(p->name),
                                             "n%d_kl%d_ku%d_t%d_%s_%s_%s",
                                             n, kl, ku, imat,
                                             FACTS[ifact], TRANSS[itran],
                                             EQUEDS[iequed]);

                                    tests[idx].name = p->name;
                                    tests[idx].test_func = test_zdrvgb_case;
                                    tests[idx].setup_func = NULL;
                                    tests[idx].teardown_func = NULL;
                                    tests[idx].initial_state = p;
                                }
                                idx++;
                            }
                        }
                    }
                }
            }
        }
    }
    return idx;
}

int main(void)
{
    INT count = build_test_array(NULL, NULL);
    if (count == 0) return 0;

    zdrvgb_params_t* params = malloc(count * sizeof(*params));
    struct CMUnitTest* tests = malloc(count * sizeof(*tests));
    if (!params || !tests) {
        free(params);
        free(tests);
        return 1;
    }

    build_test_array(params, tests);
    INT result = _cmocka_run_group_tests("zdrvgb", tests, count,
                                          group_setup, group_teardown);
    free(tests);
    free(params);
    return result;
}
