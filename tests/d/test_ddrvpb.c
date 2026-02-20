/**
 * @file test_ddrvpb.c
 * @brief Port of LAPACK ddrvpb.f — tests DPBSV and DPBSVX.
 */

#include "test_harness.h"
#include "test_rng.h"
#include <string.h>
#include <stdio.h>
#include <cblas.h>
#include <math.h>

#define NMAX    50
#define NRHS    2
#define NTYPES  8
#define NTESTS  6
#define THRESH  30.0
#define NBW     4

/* Maximum LDAB = max KD + 1 = NMAX + (NMAX+1)/4 + 1 = 63 */
#define LDAB_MAX  (NMAX + (NMAX + 1) / 4 + 1)

/* Routines under test */
extern void dpbsv(const char* uplo, const int n, const int kd, const int nrhs,
                  f64* AB, const int ldab, f64* B, const int ldb, int* info);
extern void dpbsvx(const char* fact, const char* uplo, const int n, const int kd,
                   const int nrhs, f64* AB, const int ldab,
                   f64* AFB, const int ldafb, char* equed, f64* S,
                   f64* B, const int ldb, f64* X, const int ldx,
                   f64* rcond, f64* ferr, f64* berr,
                   f64* work, int* iwork, int* info);

/* Supporting routines */
extern void dpbtrf(const char* uplo, const int n, const int kd,
                   f64* AB, const int ldab, int* info);
extern void dpbtrs(const char* uplo, const int n, const int kd, const int nrhs,
                   const f64* AB, const int ldab, f64* B, const int ldb,
                   int* info);
extern void dpbequ(const char* uplo, const int n, const int kd,
                   const f64* AB, const int ldab, f64* S,
                   f64* scond, f64* amax, int* info);
extern void dlaqsb(const char* uplo, const int n, const int kd,
                   f64* AB, const int ldab, const f64* S,
                   const f64 scond, const f64 amax, char* equed);
extern f64 dlansb(const char* norm, const char* uplo, const int n,
                     const int k, const f64* AB, const int ldab,
                     f64* work);
extern f64 dlange(const char* norm, const int m, const int n,
                     const f64* A, const int lda, f64* work);
extern void dlacpy(const char* uplo, const int m, const int n,
                   const f64* A, const int lda, f64* B, const int ldb);
extern void dlaset(const char* uplo, const int m, const int n,
                   const f64 alpha, const f64 beta,
                   f64* A, const int lda);
extern f64 dlamch(const char* cmach);

extern void dlatb4(const char* path, const int imat, const int m, const int n,
                   char* type, int* kl, int* ku, f64* anorm, int* mode,
                   f64* cndnum, char* dist);
extern void dlatms(const int m, const int n, const char* dist,
                   const char* sym, f64* d, const int mode, const f64 cond,
                   const f64 dmax, const int kl, const int ku,
                   const char* pack, f64* A, const int lda,
                   f64* work, int* info, uint64_t state[static 4]);
extern void dlarhs(const char* path, const char* xtype, const char* uplo,
                   const char* trans, const int m, const int n,
                   const int kl, const int ku, const int nrhs,
                   const f64* A, const int lda, f64* X, const int ldx,
                   f64* B, const int ldb, int* info,
                   uint64_t state[static 4]);

/* Verification routines */
extern void dpbt01(const char* uplo, const int n, const int kd,
                   const f64* A, const int lda, f64* AFAC,
                   const int ldafac, f64* rwork, f64* resid);
extern void dpbt02(const char* uplo, const int n, const int kd, const int nrhs,
                   const f64* A, const int lda, const f64* X,
                   const int ldx, f64* B, const int ldb,
                   f64* rwork, f64* resid);
extern void dpbt05(const char* uplo, const int n, const int kd, const int nrhs,
                   const f64* AB, const int ldab,
                   const f64* B, const int ldb,
                   const f64* X, const int ldx,
                   const f64* XACT, const int ldxact,
                   const f64* ferr, const f64* berr,
                   f64* reslts);
extern void dget04(const int n, const int nrhs, const f64* X, const int ldx,
                   const f64* XACT, const int ldxact, const f64 rcond,
                   f64* resid);
extern f64 dget06(const f64 rcond, const f64 rcondc);


typedef struct {
    int n;
    int kd;
    int imat;
    int iuplo;
    int ifact;
    int iequed;
    char name[96];
} ddrvpb_params_t;

typedef struct {
    f64* A;
    f64* AFAC;
    f64* ASAV;
    f64* B;
    f64* BSAV;
    f64* X;
    f64* XACT;
    f64* S;
    f64* WORK;
    f64* RWORK;
    int* IWORK;
} ddrvpb_workspace_t;

static ddrvpb_workspace_t* g_workspace = NULL;

static int group_setup(void** state) {
    (void)state;
    ddrvpb_workspace_t* ws = calloc(1, sizeof(*ws));
    if (!ws) return -1;

    int band_sz = LDAB_MAX * NMAX;
    int lwork = NMAX * NMAX;
    if (lwork < NMAX * 3) lwork = NMAX * 3;
    if (lwork < NMAX * NRHS) lwork = NMAX * NRHS;

    /* A needs extra space for KOFF offset in upper triangular packing */
    ws->A     = calloc(band_sz + LDAB_MAX, sizeof(f64));
    ws->AFAC  = calloc(band_sz, sizeof(f64));
    ws->ASAV  = calloc(band_sz, sizeof(f64));
    ws->B     = calloc(NMAX * NRHS, sizeof(f64));
    ws->BSAV  = calloc(NMAX * NRHS, sizeof(f64));
    ws->X     = calloc(NMAX * NRHS, sizeof(f64));
    ws->XACT  = calloc(NMAX * NRHS, sizeof(f64));
    ws->S     = calloc(NMAX, sizeof(f64));
    ws->WORK  = calloc(lwork, sizeof(f64));
    ws->RWORK = calloc(NMAX + 2 * NRHS, sizeof(f64));
    ws->IWORK = calloc(NMAX, sizeof(int));

    if (!ws->A || !ws->AFAC || !ws->ASAV || !ws->B || !ws->BSAV ||
        !ws->X || !ws->XACT || !ws->S || !ws->WORK || !ws->RWORK ||
        !ws->IWORK) {
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
        free(g_workspace->IWORK);
        free(g_workspace);
        g_workspace = NULL;
    }
    return 0;
}

static void get_kd_values(int n, int kdval[NBW]) {
    kdval[0] = 0;
    kdval[1] = n + (n + 1) / 4;
    kdval[2] = (3 * n - 1) / 4;
    kdval[3] = (n + 1) / 4;
}


static void run_ddrvpb_single(int n, int kd, int imat, int iuplo,
                               int ifact, int iequed)
{
    ddrvpb_workspace_t* ws = g_workspace;
    static const char* UPLOS[]  = {"U", "L"};
    static const char* FACTS[]  = {"F", "N", "E"};
    static const char* EQUEDS[] = {"N", "Y"};

    const char* uplo = UPLOS[iuplo];
    const char* fact = FACTS[ifact];
    int prefac = (fact[0] == 'F' || fact[0] == 'f');
    int nofact = (fact[0] == 'N' || fact[0] == 'n');
    int equil  = (fact[0] == 'E' || fact[0] == 'e');

    int ldab = kd + 1;
    int lda = (n > 1) ? n : 1;
    int info;

    int zerot = (imat >= 2 && imat <= 4);
    int izero = 0;

    f64 result[NTESTS];
    for (int k = 0; k < NTESTS; k++) result[k] = 0.0;

    /* Get matrix parameters from dlatb4 */
    char type, dist;
    int kl, ku, mode;
    f64 anorm, cndnum;
    dlatb4("DPB", imat, n, n, &type, &kl, &ku, &anorm, &mode, &cndnum, &dist);

    /* Seed RNG deterministically */
    uint64_t rng_state[4];
    rng_seed(rng_state, 1988 + (uint64_t)n * 1000 + (uint64_t)kd * 100
             + (uint64_t)iuplo * 50 + (uint64_t)imat);

    /* KOFF for upper triangular band packing */
    int koff;
    char packit;
    if (iuplo == 0) {
        packit = 'Q';
        koff = (kd + 1 - n > 0) ? kd + 1 - n : 0;
    } else {
        packit = 'B';
        koff = 0;
    }

    /* Generate test matrix */
    dlatms(n, n, &dist, &type, ws->RWORK, mode, cndnum, anorm,
           kd, kd, &packit, &ws->A[koff], ldab, ws->WORK, &info, rng_state);
    if (info != 0) {
        char msg[128];
        snprintf(msg, sizeof(msg), "DLATMS info=%d n=%d kd=%d imat=%d uplo=%s",
                 info, n, kd, imat, uplo);
        fail_msg("%s", msg);
        return;
    }

    /* Zero row/column for singular matrix types 2-4 */
    if (zerot) {
        if (imat == 2) izero = 1;
        else if (imat == 3) izero = n;
        else izero = n / 2 + 1;

        /* izero is 1-based */
        int i1 = (izero - kd > 1) ? izero - kd : 1;
        int i2 = (izero + kd < n) ? izero + kd : n;

        if (iuplo == 0) {
            /* Upper: zero column izero, then row izero */
            int ioff = (izero - 1) * ldab + kd;
            /* Column: entries from row i1 to izero (column izero in upper band) */
            for (int i = i1; i <= izero; i++)
                ws->A[ioff - izero + i] = 0.0;
            /* Row: entries from col izero+1 to i2 */
            for (int i = izero + 1; i <= i2; i++)
                ws->A[(i - 1) * ldab + kd + izero - i] = 0.0;
        } else {
            /* Lower: zero row izero (entries from col i1 to izero) */
            for (int i = i1; i < izero; i++)
                ws->A[(i - 1) * ldab + izero - i] = 0.0;
            /* Zero column izero (entries from row izero to i2) */
            for (int i = izero; i <= i2; i++)
                ws->A[(izero - 1) * ldab + i - izero] = 0.0;
        }
    }

    /* Save A */
    dlacpy("Full", kd + 1, n, ws->A, ldab, ws->ASAV, ldab);

    /* Skip FACT='F' for singular matrices */
    if (zerot && prefac) return;

    /*
     * Compute condition number independently per test case.
     */
    f64 rcondc = 0.0, roldc = 0.0;
    f64 scond = 0.0, amax = 0.0;
    char equed = EQUEDS[iequed][0];

    if (zerot) {
        rcondc = 0.0;
    } else {
        /* First compute condition of non-equilibrated matrix */
        dlacpy("Full", kd + 1, n, ws->ASAV, ldab, ws->AFAC, ldab);

        f64 anrm = dlansb("1", uplo, n, kd, ws->AFAC, ldab, ws->RWORK);
        dpbtrf(uplo, n, kd, ws->AFAC, ldab, &info);

        dlaset("Full", n, n, 0.0, 1.0, ws->WORK, lda);
        dpbtrs(uplo, n, kd, n, ws->AFAC, ldab, ws->WORK, lda, &info);

        f64 ainvnm = dlange("1", n, n, ws->WORK, lda, ws->RWORK);
        if (anrm <= 0.0 || ainvnm <= 0.0)
            roldc = 1.0;
        else
            roldc = (1.0 / anrm) / ainvnm;

        /* Now compute condition of equilibrated matrix if needed */
        if (equil || iequed > 0) {
            dlacpy("Full", kd + 1, n, ws->ASAV, ldab, ws->AFAC, ldab);
            dpbequ(uplo, n, kd, ws->AFAC, ldab, ws->S, &scond, &amax, &info);
            if (info == 0 && n > 0) {
                if (iequed > 0) scond = 0.0;
                dlaqsb(uplo, n, kd, ws->AFAC, ldab, ws->S, scond, amax, &equed);
            }

            anrm = dlansb("1", uplo, n, kd, ws->AFAC, ldab, ws->RWORK);
            dpbtrf(uplo, n, kd, ws->AFAC, ldab, &info);

            dlaset("Full", n, n, 0.0, 1.0, ws->WORK, lda);
            dpbtrs(uplo, n, kd, n, ws->AFAC, ldab, ws->WORK, lda, &info);

            ainvnm = dlange("1", n, n, ws->WORK, lda, ws->RWORK);
            if (anrm <= 0.0 || ainvnm <= 0.0)
                rcondc = 1.0;
            else
                rcondc = (1.0 / anrm) / ainvnm;
        } else {
            rcondc = roldc;
        }
    }

    /* Restore A */
    dlacpy("Full", kd + 1, n, ws->ASAV, ldab, ws->A, ldab);

    /* Generate exact solution and RHS */
    char xtype = 'N';
    dlarhs("DPB", &xtype, uplo, " ", n, n, kd, kd, NRHS,
           ws->A, ldab, ws->XACT, lda, ws->B, lda, &info, rng_state);
    dlacpy("Full", n, NRHS, ws->B, lda, ws->BSAV, lda);

    /* ====== DPBSV test ====== */
    if (nofact) {
        dlacpy("Full", kd + 1, n, ws->A, ldab, ws->AFAC, ldab);
        dlacpy("Full", n, NRHS, ws->B, lda, ws->X, lda);

        dpbsv(uplo, n, kd, NRHS, ws->AFAC, ldab, ws->X, lda, &info);

        if (info != izero) {
            char msg[128];
            snprintf(msg, sizeof(msg),
                     "DPBSV info=%d expected=%d uplo=%s n=%d kd=%d imat=%d",
                     info, izero, uplo, n, kd, imat);
            fail_msg("%s", msg);
            return;
        }

        if (info == 0) {
            dpbt01(uplo, n, kd, ws->A, ldab, ws->AFAC, ldab,
                   ws->RWORK, &result[0]);

            dlacpy("Full", n, NRHS, ws->B, lda, ws->WORK, lda);
            dpbt02(uplo, n, kd, NRHS, ws->A, ldab, ws->X, lda,
                   ws->WORK, lda, ws->RWORK, &result[1]);

            dget04(n, NRHS, ws->X, lda, ws->XACT, lda, rcondc, &result[2]);

            for (int k = 0; k < 3; k++) {
                if (result[k] >= THRESH) {
                    char msg[256];
                    snprintf(msg, sizeof(msg),
                             "DPBSV uplo=%s n=%d kd=%d imat=%d test(%d)=%.5g",
                             uplo, n, kd, imat, k + 1, result[k]);
                    fail_msg("%s", msg);
                }
            }
        }
    }

    /* ====== DPBSVX test ====== */
    /* Restore A from ASAV */
    dlacpy("Full", kd + 1, n, ws->ASAV, ldab, ws->A, ldab);

    if (!prefac)
        dlaset("Full", kd + 1, n, 0.0, 0.0, ws->AFAC, ldab);
    dlaset("Full", n, NRHS, 0.0, 0.0, ws->X, lda);

    equed = EQUEDS[iequed][0];

    if (iequed > 0 && n > 0) {
        dlaqsb(uplo, n, kd, ws->A, ldab, ws->S, scond, amax, &equed);
    }

    /* Restore B from BSAV */
    dlacpy("Full", n, NRHS, ws->BSAV, lda, ws->B, lda);

    f64 rcond;
    dpbsvx(fact, uplo, n, kd, NRHS, ws->A, ldab, ws->AFAC, ldab,
           &equed, ws->S, ws->B, lda, ws->X, lda, &rcond,
           ws->RWORK, &ws->RWORK[NRHS], ws->WORK, ws->IWORK, &info);

    if (info != izero) {
        char msg[128];
        snprintf(msg, sizeof(msg),
                 "DPBSVX info=%d expected=%d fact=%s uplo=%s n=%d kd=%d imat=%d equed=%c",
                 info, izero, fact, uplo, n, kd, imat, equed);
        fail_msg("%s", msg);
        return;
    }

    int k1;
    if (info == 0) {
        if (!prefac) {
            dpbt01(uplo, n, kd, ws->A, ldab, ws->AFAC, ldab,
                   &ws->RWORK[2 * NRHS], &result[0]);
            k1 = 0;
        } else {
            k1 = 1;
        }

        dlacpy("Full", n, NRHS, ws->BSAV, lda, ws->WORK, lda);
        dpbt02(uplo, n, kd, NRHS, ws->ASAV, ldab, ws->X, lda,
               ws->WORK, lda, &ws->RWORK[2 * NRHS], &result[1]);

        if (nofact || (prefac && (equed == 'N' || equed == 'n'))) {
            dget04(n, NRHS, ws->X, lda, ws->XACT, lda, rcondc, &result[2]);
        } else {
            dget04(n, NRHS, ws->X, lda, ws->XACT, lda, roldc, &result[2]);
        }

        dpbt05(uplo, n, kd, NRHS, ws->ASAV, ldab, ws->B, lda,
               ws->X, lda, ws->XACT, lda,
               ws->RWORK, &ws->RWORK[NRHS], &result[3]);
    } else {
        k1 = 5;
    }

    result[5] = dget06(rcond, rcondc);

    for (int k = k1; k < NTESTS; k++) {
        if (result[k] >= THRESH) {
            char msg[256];
            if (prefac) {
                snprintf(msg, sizeof(msg),
                         "DPBSVX fact=%s uplo=%s n=%d kd=%d equed=%c imat=%d test(%d)=%.5g",
                         fact, uplo, n, kd, equed, imat, k + 1, result[k]);
            } else {
                snprintf(msg, sizeof(msg),
                         "DPBSVX fact=%s uplo=%s n=%d kd=%d imat=%d test(%d)=%.5g",
                         fact, uplo, n, kd, imat, k + 1, result[k]);
            }
            fail_msg("%s", msg);
        }
    }
}


static void test_ddrvpb_case(void** state) {
    ddrvpb_params_t* p = (ddrvpb_params_t*)*state;
    run_ddrvpb_single(p->n, p->kd, p->imat, p->iuplo, p->ifact, p->iequed);
}


static int build_test_array(ddrvpb_params_t* params, struct CMUnitTest* tests)
{
    static const int NVAL[] = {0, 1, 2, 3, 5, 10, 50};
    static const int NN = sizeof(NVAL) / sizeof(NVAL[0]);
    static const char* FACTS[]  = {"F", "N", "E"};
    static const char* UPLOS[]  = {"U", "L"};
    static const char* EQUEDS[] = {"N", "Y"};

    int idx = 0;

    for (int in = 0; in < NN; in++) {
        int n = NVAL[in];
        int nkd = (n > 4) ? 4 : ((n > 0) ? n : 1);
        int nimat = (n <= 0) ? 1 : NTYPES;

        int kdval[NBW];
        get_kd_values(n, kdval);

        for (int ikd = 0; ikd < nkd; ikd++) {
            int kd = kdval[ikd];
            int ldab = kd + 1;

            /* Check workspace bounds */
            if ((long)ldab * n > (long)LDAB_MAX * NMAX) continue;

            for (int iuplo = 0; iuplo < 2; iuplo++) {
                for (int imat = 1; imat <= nimat; imat++) {
                    int zerot = (imat >= 2 && imat <= 4);
                    if (zerot && n < imat - 1) continue;

                    for (int iequed = 0; iequed < 2; iequed++) {
                        int nfact = (iequed == 0) ? 3 : 1;

                        for (int ifact = 0; ifact < nfact; ifact++) {
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
                                tests[idx].test_func = test_ddrvpb_case;
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
    int count = build_test_array(NULL, NULL);
    ddrvpb_params_t* params = malloc(count * sizeof(*params));
    struct CMUnitTest* tests = malloc(count * sizeof(*tests));
    build_test_array(params, tests);
    int result = _cmocka_run_group_tests("ddrvpb", tests, count,
                                          group_setup, group_teardown);
    free(tests);
    free(params);
    return result;
}
