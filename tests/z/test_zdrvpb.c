/**
 * @file test_zdrvpb.c
 * @brief Port of LAPACK zdrvpb.f — tests ZPBSV and ZPBSVX.
 */

#include "test_harness.h"
#include "verify.h"
#include "test_rng.h"
#include <string.h>
#include <stdio.h>
#include <math.h>

#define NMAX    50
#define NRHS    2
#define NTYPES  8
#define NTESTS  6
#define THRESH  30.0
#define NBW     4

/* Maximum LDAB = max KD + 1 = NMAX + (NMAX+1)/4 + 1 = 63 */
#define LDAB_MAX  (NMAX + (NMAX + 1) / 4 + 1)

typedef struct {
    INT n;
    INT kd;
    INT imat;
    INT iuplo;
    INT ifact;
    INT iequed;
    char name[96];
} zdrvpb_params_t;

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
} zdrvpb_workspace_t;

static zdrvpb_workspace_t* g_workspace = NULL;

static int group_setup(void** state) {
    (void)state;
    zdrvpb_workspace_t* ws = calloc(1, sizeof(*ws));
    if (!ws) return -1;

    INT band_sz = LDAB_MAX * NMAX;
    INT lwork = NMAX * NMAX;
    if (lwork < NMAX * 3) lwork = NMAX * 3;
    if (lwork < NMAX * NRHS) lwork = NMAX * NRHS;

    ws->A     = calloc(band_sz + LDAB_MAX, sizeof(c128));
    ws->AFAC  = calloc(band_sz, sizeof(c128));
    ws->ASAV  = calloc(band_sz, sizeof(c128));
    ws->B     = calloc(NMAX * NRHS, sizeof(c128));
    ws->BSAV  = calloc(NMAX * NRHS, sizeof(c128));
    ws->X     = calloc(NMAX * NRHS, sizeof(c128));
    ws->XACT  = calloc(NMAX * NRHS, sizeof(c128));
    ws->S     = calloc(NMAX, sizeof(f64));
    ws->WORK  = calloc(lwork, sizeof(c128));
    ws->RWORK = calloc(NMAX + 2 * NRHS, sizeof(f64));

    if (!ws->A || !ws->AFAC || !ws->ASAV || !ws->B || !ws->BSAV ||
        !ws->X || !ws->XACT || !ws->S || !ws->WORK || !ws->RWORK) {
        return -1;
    }

    g_workspace = ws;
    return 0;
}

static int group_teardown(void** state) {
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

static void get_kd_values(INT n, INT kdval[NBW]) {
    kdval[0] = 0;
    kdval[1] = n + (n + 1) / 4;
    kdval[2] = (3 * n - 1) / 4;
    kdval[3] = (n + 1) / 4;
}


static void run_zdrvpb_single(INT n, INT kd, INT imat, INT iuplo,
                               INT ifact, INT iequed)
{
    zdrvpb_workspace_t* ws = g_workspace;
    static const char* UPLOS[]  = {"U", "L"};
    static const char* FACTS[]  = {"F", "N", "E"};
    static const char* EQUEDS[] = {"N", "Y"};
    const c128 CZERO = CMPLX(0.0, 0.0);
    const c128 CONE  = CMPLX(1.0, 0.0);

    const char* uplo = UPLOS[iuplo];
    const char* fact = FACTS[ifact];
    INT prefac = (fact[0] == 'F' || fact[0] == 'f');
    INT nofact = (fact[0] == 'N' || fact[0] == 'n');
    INT equil  = (fact[0] == 'E' || fact[0] == 'e');

    INT ldab = kd + 1;
    INT lda = (n > 1) ? n : 1;
    INT info;

    INT zerot = (imat >= 2 && imat <= 4);
    INT izero = 0;

    f64 result[NTESTS];
    for (INT k = 0; k < NTESTS; k++) result[k] = 0.0;

    char type, dist;
    INT kl, ku, mode;
    f64 anorm, cndnum;
    zlatb4("ZPB", imat, n, n, &type, &kl, &ku, &anorm, &mode, &cndnum, &dist);

    uint64_t rng_state[4];
    rng_seed(rng_state, 1988 + (uint64_t)n * 1000 + (uint64_t)kd * 100
             + (uint64_t)iuplo * 50 + (uint64_t)imat);

    INT koff;
    char packit;
    if (iuplo == 0) {
        packit = 'Q';
        koff = (kd + 1 - n > 0) ? kd + 1 - n : 0;
    } else {
        packit = 'B';
        koff = 0;
    }

    zlatms(n, n, &dist, &type, ws->RWORK, mode, cndnum, anorm,
           kd, kd, &packit, &ws->A[koff], ldab, ws->WORK, &info, rng_state);
    if (info != 0) {
        char msg[128];
        snprintf(msg, sizeof(msg), "ZLATMS info=%d n=%d kd=%d imat=%d uplo=%s",
                 info, n, kd, imat, uplo);
        fail_msg("%s", msg);
        return;
    }

    /* Zero row/column for singular matrix types 2-4 */
    if (zerot) {
        if (imat == 2) izero = 1;
        else if (imat == 3) izero = n;
        else izero = n / 2 + 1;

        INT i1 = (izero - kd > 1) ? izero - kd : 1;
        INT i2 = (izero + kd < n) ? izero + kd : n;

        if (iuplo == 0) {
            INT ioff = (izero - 1) * ldab + kd;
            for (INT i = i1; i <= izero; i++)
                ws->A[ioff - izero + i] = CZERO;
            for (INT i = izero + 1; i <= i2; i++)
                ws->A[(i - 1) * ldab + kd + izero - i] = CZERO;
        } else {
            for (INT i = i1; i < izero; i++)
                ws->A[(i - 1) * ldab + izero - i] = CZERO;
            for (INT i = izero; i <= i2; i++)
                ws->A[(izero - 1) * ldab + i - izero] = CZERO;
        }
    }

    /* Set the imaginary part of the diagonals */
    if (iuplo == 0) {
        zlaipd(n, &ws->A[kd], ldab, 0);
    } else {
        zlaipd(n, ws->A, ldab, 0);
    }

    /* Save A */
    zlacpy("Full", kd + 1, n, ws->A, ldab, ws->ASAV, ldab);

    /* Skip FACT='F' for singular matrices */
    if (zerot && prefac) return;

    f64 rcondc = 0.0, roldc = 0.0;
    f64 scond = 0.0, amax = 0.0;
    char equed = EQUEDS[iequed][0];

    if (zerot) {
        rcondc = 0.0;
    } else {
        zlacpy("Full", kd + 1, n, ws->ASAV, ldab, ws->AFAC, ldab);

        f64 anrm = zlanhb("1", uplo, n, kd, ws->AFAC, ldab, ws->RWORK);
        zpbtrf(uplo, n, kd, ws->AFAC, ldab, &info);

        zlaset("Full", n, n, CZERO, CONE, ws->WORK, lda);
        zpbtrs(uplo, n, kd, n, ws->AFAC, ldab, ws->WORK, lda, &info);

        f64 ainvnm = zlange("1", n, n, ws->WORK, lda, ws->RWORK);
        if (anrm <= 0.0 || ainvnm <= 0.0)
            roldc = 1.0;
        else
            roldc = (1.0 / anrm) / ainvnm;

        if (equil || iequed > 0) {
            zlacpy("Full", kd + 1, n, ws->ASAV, ldab, ws->AFAC, ldab);
            zpbequ(uplo, n, kd, ws->AFAC, ldab, ws->S, &scond, &amax, &info);
            if (info == 0 && n > 0) {
                if (iequed > 0) scond = 0.0;
                zlaqhb(uplo, n, kd, ws->AFAC, ldab, ws->S, scond, amax, &equed);
            }

            anrm = zlanhb("1", uplo, n, kd, ws->AFAC, ldab, ws->RWORK);
            zpbtrf(uplo, n, kd, ws->AFAC, ldab, &info);

            zlaset("Full", n, n, CZERO, CONE, ws->WORK, lda);
            zpbtrs(uplo, n, kd, n, ws->AFAC, ldab, ws->WORK, lda, &info);

            ainvnm = zlange("1", n, n, ws->WORK, lda, ws->RWORK);
            if (anrm <= 0.0 || ainvnm <= 0.0)
                rcondc = 1.0;
            else
                rcondc = (1.0 / anrm) / ainvnm;
        } else {
            rcondc = roldc;
        }
    }

    /* Restore A */
    zlacpy("Full", kd + 1, n, ws->ASAV, ldab, ws->A, ldab);

    /* Generate exact solution and RHS */
    char xtype = 'N';
    zlarhs("ZPB", &xtype, uplo, " ", n, n, kd, kd, NRHS,
           ws->A, ldab, ws->XACT, lda, ws->B, lda, &info, rng_state);
    zlacpy("Full", n, NRHS, ws->B, lda, ws->BSAV, lda);

    /* ====== ZPBSV test ====== */
    if (nofact) {
        zlacpy("Full", kd + 1, n, ws->A, ldab, ws->AFAC, ldab);
        zlacpy("Full", n, NRHS, ws->B, lda, ws->X, lda);

        zpbsv(uplo, n, kd, NRHS, ws->AFAC, ldab, ws->X, lda, &info);

        if (info != izero) {
            char msg[128];
            snprintf(msg, sizeof(msg),
                     "ZPBSV info=%d expected=%d uplo=%s n=%d kd=%d imat=%d",
                     info, izero, uplo, n, kd, imat);
            fail_msg("%s", msg);
            return;
        }

        if (info == 0) {
            zpbt01(uplo, n, kd, ws->A, ldab, ws->AFAC, ldab,
                   ws->RWORK, &result[0]);

            zlacpy("Full", n, NRHS, ws->B, lda, ws->WORK, lda);
            zpbt02(uplo, n, kd, NRHS, ws->A, ldab, ws->X, lda,
                   ws->WORK, lda, ws->RWORK, &result[1]);

            zget04(n, NRHS, ws->X, lda, ws->XACT, lda, rcondc, &result[2]);

            for (INT k = 0; k < 3; k++) {
                if (result[k] >= THRESH) {
                    char msg[256];
                    snprintf(msg, sizeof(msg),
                             "ZPBSV uplo=%s n=%d kd=%d imat=%d test(%d)=%.5g",
                             uplo, n, kd, imat, k + 1, result[k]);
                    fail_msg("%s", msg);
                }
            }
        }
    }

    /* ====== ZPBSVX test ====== */
    zlacpy("Full", kd + 1, n, ws->ASAV, ldab, ws->A, ldab);

    if (!prefac)
        zlaset("Full", kd + 1, n, CZERO, CZERO, ws->AFAC, ldab);
    zlaset("Full", n, NRHS, CZERO, CZERO, ws->X, lda);

    equed = EQUEDS[iequed][0];

    if (iequed > 0 && n > 0) {
        zlaqhb(uplo, n, kd, ws->A, ldab, ws->S, scond, amax, &equed);
    }

    /* Restore B from BSAV */
    zlacpy("Full", n, NRHS, ws->BSAV, lda, ws->B, lda);

    f64 rcond;
    zpbsvx(fact, uplo, n, kd, NRHS, ws->A, ldab, ws->AFAC, ldab,
           &equed, ws->S, ws->B, lda, ws->X, lda, &rcond,
           ws->RWORK, &ws->RWORK[NRHS], ws->WORK,
           &ws->RWORK[2 * NRHS], &info);

    if (info != izero) {
        char msg[128];
        snprintf(msg, sizeof(msg),
                 "ZPBSVX info=%d expected=%d fact=%s uplo=%s n=%d kd=%d imat=%d equed=%c",
                 info, izero, fact, uplo, n, kd, imat, equed);
        fail_msg("%s", msg);
        return;
    }

    INT k1;
    if (info == 0) {
        if (!prefac) {
            zpbt01(uplo, n, kd, ws->A, ldab, ws->AFAC, ldab,
                   &ws->RWORK[2 * NRHS], &result[0]);
            k1 = 0;
        } else {
            k1 = 1;
        }

        zlacpy("Full", n, NRHS, ws->BSAV, lda, ws->WORK, lda);
        zpbt02(uplo, n, kd, NRHS, ws->ASAV, ldab, ws->X, lda,
               ws->WORK, lda, &ws->RWORK[2 * NRHS], &result[1]);

        if (nofact || (prefac && (equed == 'N' || equed == 'n'))) {
            zget04(n, NRHS, ws->X, lda, ws->XACT, lda, rcondc, &result[2]);
        } else {
            zget04(n, NRHS, ws->X, lda, ws->XACT, lda, roldc, &result[2]);
        }

        zpbt05(uplo, n, kd, NRHS, ws->ASAV, ldab, ws->B, lda,
               ws->X, lda, ws->XACT, lda,
               ws->RWORK, &ws->RWORK[NRHS], &result[3]);
    } else {
        k1 = 5;
    }

    result[5] = dget06(rcond, rcondc);

    for (INT k = k1; k < NTESTS; k++) {
        if (result[k] >= THRESH) {
            char msg[256];
            if (prefac) {
                snprintf(msg, sizeof(msg),
                         "ZPBSVX fact=%s uplo=%s n=%d kd=%d equed=%c imat=%d test(%d)=%.5g",
                         fact, uplo, n, kd, equed, imat, k + 1, result[k]);
            } else {
                snprintf(msg, sizeof(msg),
                         "ZPBSVX fact=%s uplo=%s n=%d kd=%d imat=%d test(%d)=%.5g",
                         fact, uplo, n, kd, imat, k + 1, result[k]);
            }
            fail_msg("%s", msg);
        }
    }
}


static void test_zdrvpb_case(void** state) {
    zdrvpb_params_t* p = (zdrvpb_params_t*)*state;
    run_zdrvpb_single(p->n, p->kd, p->imat, p->iuplo, p->ifact, p->iequed);
}


static INT build_test_array(zdrvpb_params_t* params, struct CMUnitTest* tests)
{
    static const INT NVAL[] = {0, 1, 2, 3, 5, 10, 50};
    static const INT NN = sizeof(NVAL) / sizeof(NVAL[0]);
    static const char* FACTS[]  = {"F", "N", "E"};
    static const char* UPLOS[]  = {"U", "L"};
    static const char* EQUEDS[] = {"N", "Y"};

    INT idx = 0;

    for (INT in = 0; in < NN; in++) {
        INT n = NVAL[in];
        INT nkd = (n > 4) ? 4 : ((n > 0) ? n : 1);
        INT nimat = (n <= 0) ? 1 : NTYPES;

        INT kdval[NBW];
        get_kd_values(n, kdval);

        for (INT ikd = 0; ikd < nkd; ikd++) {
            INT kd = kdval[ikd];
            INT ldab = kd + 1;

            if ((long)ldab * n > (long)LDAB_MAX * NMAX) continue;

            for (INT iuplo = 0; iuplo < 2; iuplo++) {
                for (INT imat = 1; imat <= nimat; imat++) {
                    INT zerot = (imat >= 2 && imat <= 4);
                    if (zerot && n < imat - 1) continue;

                    for (INT iequed = 0; iequed < 2; iequed++) {
                        INT nfact = (iequed == 0) ? 3 : 1;

                        for (INT ifact = 0; ifact < nfact; ifact++) {
                            if (zerot && ifact == 0) continue;

                            if (params && tests) {
                                params[idx].n      = n;
                                params[idx].kd     = kd;
                                params[idx].imat   = imat;
                                params[idx].iuplo  = iuplo;
                                params[idx].ifact  = ifact;
                                params[idx].iequed = iequed;
                                snprintf(params[idx].name, sizeof(params[idx].name),
                                         "n=%d kd=%d uplo=%s imat=%d fact=%s eq=%s",
                                         n, kd, UPLOS[iuplo], imat,
                                         FACTS[ifact], EQUEDS[iequed]);

                                tests[idx].name = params[idx].name;
                                tests[idx].test_func = test_zdrvpb_case;
                                tests[idx].setup_func = NULL;
                                tests[idx].teardown_func = NULL;
                                tests[idx].initial_state = &params[idx];
                            }
                            idx++;
                        }
                    }
                }
            }
        }
    }

    return idx;
}


int main(void) {
    INT count = build_test_array(NULL, NULL);
    zdrvpb_params_t* params = malloc(count * sizeof(*params));
    struct CMUnitTest* tests = malloc(count * sizeof(*tests));
    build_test_array(params, tests);
    INT result = _cmocka_run_group_tests("zdrvpb", tests, count,
                                          group_setup, group_teardown);
    free(tests);
    free(params);
    return result;
}
