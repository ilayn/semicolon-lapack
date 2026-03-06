/**
 * @file test_cdrvpb.c
 * @brief Port of LAPACK zdrvpb.f — tests CPBSV and CPBSVX.
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
#define THRESH  30.0f
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

    ws->A     = calloc(band_sz + LDAB_MAX, sizeof(c64));
    ws->AFAC  = calloc(band_sz, sizeof(c64));
    ws->ASAV  = calloc(band_sz, sizeof(c64));
    ws->B     = calloc(NMAX * NRHS, sizeof(c64));
    ws->BSAV  = calloc(NMAX * NRHS, sizeof(c64));
    ws->X     = calloc(NMAX * NRHS, sizeof(c64));
    ws->XACT  = calloc(NMAX * NRHS, sizeof(c64));
    ws->S     = calloc(NMAX, sizeof(f32));
    ws->WORK  = calloc(lwork, sizeof(c64));
    ws->RWORK = calloc(NMAX + 2 * NRHS, sizeof(f32));

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
    const c64 CZERO = CMPLXF(0.0f, 0.0f);
    const c64 CONE  = CMPLXF(1.0f, 0.0f);

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

    f32 result[NTESTS];
    for (INT k = 0; k < NTESTS; k++) result[k] = 0.0f;

    char type, dist;
    INT kl, ku, mode;
    f32 anorm, cndnum;
    clatb4("CPB", imat, n, n, &type, &kl, &ku, &anorm, &mode, &cndnum, &dist);

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

    clatms(n, n, &dist, &type, ws->RWORK, mode, cndnum, anorm,
           kd, kd, &packit, &ws->A[koff], ldab, ws->WORK, &info, rng_state);
    if (info != 0) {
        char msg[128];
        snprintf(msg, sizeof(msg), "CLATMS info=%d n=%d kd=%d imat=%d uplo=%s",
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
        claipd(n, &ws->A[kd], ldab, 0);
    } else {
        claipd(n, ws->A, ldab, 0);
    }

    /* Save A */
    clacpy("Full", kd + 1, n, ws->A, ldab, ws->ASAV, ldab);

    /* Skip FACT='F' for singular matrices */
    if (zerot && prefac) return;

    f32 rcondc = 0.0f, roldc = 0.0f;
    f32 scond = 0.0f, amax = 0.0f;
    char equed = EQUEDS[iequed][0];

    if (zerot) {
        rcondc = 0.0f;
    } else {
        clacpy("Full", kd + 1, n, ws->ASAV, ldab, ws->AFAC, ldab);

        f32 anrm = clanhb("1", uplo, n, kd, ws->AFAC, ldab, ws->RWORK);
        cpbtrf(uplo, n, kd, ws->AFAC, ldab, &info);

        claset("Full", n, n, CZERO, CONE, ws->WORK, lda);
        cpbtrs(uplo, n, kd, n, ws->AFAC, ldab, ws->WORK, lda, &info);

        f32 ainvnm = clange("1", n, n, ws->WORK, lda, ws->RWORK);
        if (anrm <= 0.0f || ainvnm <= 0.0f)
            roldc = 1.0f;
        else
            roldc = (1.0f / anrm) / ainvnm;

        if (equil || iequed > 0) {
            clacpy("Full", kd + 1, n, ws->ASAV, ldab, ws->AFAC, ldab);
            cpbequ(uplo, n, kd, ws->AFAC, ldab, ws->S, &scond, &amax, &info);
            if (info == 0 && n > 0) {
                if (iequed > 0) scond = 0.0f;
                claqhb(uplo, n, kd, ws->AFAC, ldab, ws->S, scond, amax, &equed);
            }

            anrm = clanhb("1", uplo, n, kd, ws->AFAC, ldab, ws->RWORK);
            cpbtrf(uplo, n, kd, ws->AFAC, ldab, &info);

            claset("Full", n, n, CZERO, CONE, ws->WORK, lda);
            cpbtrs(uplo, n, kd, n, ws->AFAC, ldab, ws->WORK, lda, &info);

            ainvnm = clange("1", n, n, ws->WORK, lda, ws->RWORK);
            if (anrm <= 0.0f || ainvnm <= 0.0f)
                rcondc = 1.0f;
            else
                rcondc = (1.0f / anrm) / ainvnm;
        } else {
            rcondc = roldc;
        }
    }

    /* Restore A */
    clacpy("Full", kd + 1, n, ws->ASAV, ldab, ws->A, ldab);

    /* Generate exact solution and RHS */
    char xtype = 'N';
    clarhs("CPB", &xtype, uplo, " ", n, n, kd, kd, NRHS,
           ws->A, ldab, ws->XACT, lda, ws->B, lda, &info, rng_state);
    clacpy("Full", n, NRHS, ws->B, lda, ws->BSAV, lda);

    /* ====== CPBSV test ====== */
    if (nofact) {
        clacpy("Full", kd + 1, n, ws->A, ldab, ws->AFAC, ldab);
        clacpy("Full", n, NRHS, ws->B, lda, ws->X, lda);

        cpbsv(uplo, n, kd, NRHS, ws->AFAC, ldab, ws->X, lda, &info);

        if (info != izero) {
            char msg[128];
            snprintf(msg, sizeof(msg),
                     "CPBSV info=%d expected=%d uplo=%s n=%d kd=%d imat=%d",
                     info, izero, uplo, n, kd, imat);
            fail_msg("%s", msg);
            return;
        }

        if (info == 0) {
            cpbt01(uplo, n, kd, ws->A, ldab, ws->AFAC, ldab,
                   ws->RWORK, &result[0]);

            clacpy("Full", n, NRHS, ws->B, lda, ws->WORK, lda);
            cpbt02(uplo, n, kd, NRHS, ws->A, ldab, ws->X, lda,
                   ws->WORK, lda, ws->RWORK, &result[1]);

            cget04(n, NRHS, ws->X, lda, ws->XACT, lda, rcondc, &result[2]);

            for (INT k = 0; k < 3; k++) {
                if (result[k] >= THRESH) {
                    char msg[256];
                    snprintf(msg, sizeof(msg),
                             "CPBSV uplo=%s n=%d kd=%d imat=%d test(%d)=%.5g",
                             uplo, n, kd, imat, k + 1, (double)result[k]);
                    fail_msg("%s", msg);
                }
            }
        }
    }

    /* ====== CPBSVX test ====== */
    clacpy("Full", kd + 1, n, ws->ASAV, ldab, ws->A, ldab);

    if (!prefac)
        claset("Full", kd + 1, n, CZERO, CZERO, ws->AFAC, ldab);
    claset("Full", n, NRHS, CZERO, CZERO, ws->X, lda);

    equed = EQUEDS[iequed][0];

    if (iequed > 0 && n > 0) {
        claqhb(uplo, n, kd, ws->A, ldab, ws->S, scond, amax, &equed);
    }

    /* Restore B from BSAV */
    clacpy("Full", n, NRHS, ws->BSAV, lda, ws->B, lda);

    f32 rcond;
    cpbsvx(fact, uplo, n, kd, NRHS, ws->A, ldab, ws->AFAC, ldab,
           &equed, ws->S, ws->B, lda, ws->X, lda, &rcond,
           ws->RWORK, &ws->RWORK[NRHS], ws->WORK,
           &ws->RWORK[2 * NRHS], &info);

    if (info != izero) {
        char msg[128];
        snprintf(msg, sizeof(msg),
                 "CPBSVX info=%d expected=%d fact=%s uplo=%s n=%d kd=%d imat=%d equed=%c",
                 info, izero, fact, uplo, n, kd, imat, equed);
        fail_msg("%s", msg);
        return;
    }

    INT k1;
    if (info == 0) {
        if (!prefac) {
            cpbt01(uplo, n, kd, ws->A, ldab, ws->AFAC, ldab,
                   &ws->RWORK[2 * NRHS], &result[0]);
            k1 = 0;
        } else {
            k1 = 1;
        }

        clacpy("Full", n, NRHS, ws->BSAV, lda, ws->WORK, lda);
        cpbt02(uplo, n, kd, NRHS, ws->ASAV, ldab, ws->X, lda,
               ws->WORK, lda, &ws->RWORK[2 * NRHS], &result[1]);

        if (nofact || (prefac && (equed == 'N' || equed == 'n'))) {
            cget04(n, NRHS, ws->X, lda, ws->XACT, lda, rcondc, &result[2]);
        } else {
            cget04(n, NRHS, ws->X, lda, ws->XACT, lda, roldc, &result[2]);
        }

        cpbt05(uplo, n, kd, NRHS, ws->ASAV, ldab, ws->B, lda,
               ws->X, lda, ws->XACT, lda,
               ws->RWORK, &ws->RWORK[NRHS], &result[3]);
    } else {
        k1 = 5;
    }

    result[5] = sget06(rcond, rcondc);

    for (INT k = k1; k < NTESTS; k++) {
        if (result[k] >= THRESH) {
            char msg[256];
            if (prefac) {
                snprintf(msg, sizeof(msg),
                         "CPBSVX fact=%s uplo=%s n=%d kd=%d equed=%c imat=%d test(%d)=%.5g",
                         fact, uplo, n, kd, equed, imat, k + 1, (double)result[k]);
            } else {
                snprintf(msg, sizeof(msg),
                         "CPBSVX fact=%s uplo=%s n=%d kd=%d imat=%d test(%d)=%.5g",
                         fact, uplo, n, kd, imat, k + 1, (double)result[k]);
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
