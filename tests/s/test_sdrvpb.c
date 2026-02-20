/**
 * @file test_sdrvpb.c
 * @brief Port of LAPACK ddrvpb.f — tests SPBSV and SPBSVX.
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
#define THRESH  30.0f
#define NBW     4

/* Maximum LDAB = max KD + 1 = NMAX + (NMAX+1)/4 + 1 = 63 */
#define LDAB_MAX  (NMAX + (NMAX + 1) / 4 + 1)

/* Routines under test */
extern void spbsv(const char* uplo, const int n, const int kd, const int nrhs,
                  f32* AB, const int ldab, f32* B, const int ldb, int* info);
extern void spbsvx(const char* fact, const char* uplo, const int n, const int kd,
                   const int nrhs, f32* AB, const int ldab,
                   f32* AFB, const int ldafb, char* equed, f32* S,
                   f32* B, const int ldb, f32* X, const int ldx,
                   f32* rcond, f32* ferr, f32* berr,
                   f32* work, int* iwork, int* info);

/* Supporting routines */
extern void spbtrf(const char* uplo, const int n, const int kd,
                   f32* AB, const int ldab, int* info);
extern void spbtrs(const char* uplo, const int n, const int kd, const int nrhs,
                   const f32* AB, const int ldab, f32* B, const int ldb,
                   int* info);
extern void spbequ(const char* uplo, const int n, const int kd,
                   const f32* AB, const int ldab, f32* S,
                   f32* scond, f32* amax, int* info);
extern void slaqsb(const char* uplo, const int n, const int kd,
                   f32* AB, const int ldab, const f32* S,
                   const f32 scond, const f32 amax, char* equed);
extern f32 slansb(const char* norm, const char* uplo, const int n,
                     const int k, const f32* AB, const int ldab,
                     f32* work);
extern f32 slange(const char* norm, const int m, const int n,
                     const f32* A, const int lda, f32* work);
extern void slacpy(const char* uplo, const int m, const int n,
                   const f32* A, const int lda, f32* B, const int ldb);
extern void slaset(const char* uplo, const int m, const int n,
                   const f32 alpha, const f32 beta,
                   f32* A, const int lda);
extern f32 slamch(const char* cmach);

extern void slatb4(const char* path, const int imat, const int m, const int n,
                   char* type, int* kl, int* ku, f32* anorm, int* mode,
                   f32* cndnum, char* dist);
extern void slatms(const int m, const int n, const char* dist,
                   const char* sym, f32* d, const int mode, const f32 cond,
                   const f32 dmax, const int kl, const int ku,
                   const char* pack, f32* A, const int lda,
                   f32* work, int* info, uint64_t state[static 4]);
extern void slarhs(const char* path, const char* xtype, const char* uplo,
                   const char* trans, const int m, const int n,
                   const int kl, const int ku, const int nrhs,
                   const f32* A, const int lda, f32* X, const int ldx,
                   f32* B, const int ldb, int* info,
                   uint64_t state[static 4]);

/* Verification routines */
extern void spbt01(const char* uplo, const int n, const int kd,
                   const f32* A, const int lda, f32* AFAC,
                   const int ldafac, f32* rwork, f32* resid);
extern void spbt02(const char* uplo, const int n, const int kd, const int nrhs,
                   const f32* A, const int lda, const f32* X,
                   const int ldx, f32* B, const int ldb,
                   f32* rwork, f32* resid);
extern void spbt05(const char* uplo, const int n, const int kd, const int nrhs,
                   const f32* AB, const int ldab,
                   const f32* B, const int ldb,
                   const f32* X, const int ldx,
                   const f32* XACT, const int ldxact,
                   const f32* ferr, const f32* berr,
                   f32* reslts);
extern void sget04(const int n, const int nrhs, const f32* X, const int ldx,
                   const f32* XACT, const int ldxact, const f32 rcond,
                   f32* resid);
extern f32 sget06(const f32 rcond, const f32 rcondc);


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
    ws->A     = calloc(band_sz + LDAB_MAX, sizeof(f32));
    ws->AFAC  = calloc(band_sz, sizeof(f32));
    ws->ASAV  = calloc(band_sz, sizeof(f32));
    ws->B     = calloc(NMAX * NRHS, sizeof(f32));
    ws->BSAV  = calloc(NMAX * NRHS, sizeof(f32));
    ws->X     = calloc(NMAX * NRHS, sizeof(f32));
    ws->XACT  = calloc(NMAX * NRHS, sizeof(f32));
    ws->S     = calloc(NMAX, sizeof(f32));
    ws->WORK  = calloc(lwork, sizeof(f32));
    ws->RWORK = calloc(NMAX + 2 * NRHS, sizeof(f32));
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

    f32 result[NTESTS];
    for (int k = 0; k < NTESTS; k++) result[k] = 0.0f;

    /* Get matrix parameters from slatb4 */
    char type, dist;
    int kl, ku, mode;
    f32 anorm, cndnum;
    slatb4("SPB", imat, n, n, &type, &kl, &ku, &anorm, &mode, &cndnum, &dist);

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
    slatms(n, n, &dist, &type, ws->RWORK, mode, cndnum, anorm,
           kd, kd, &packit, &ws->A[koff], ldab, ws->WORK, &info, rng_state);
    if (info != 0) {
        char msg[128];
        snprintf(msg, sizeof(msg), "SLATMS info=%d n=%d kd=%d imat=%d uplo=%s",
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
                ws->A[ioff - izero + i] = 0.0f;
            /* Row: entries from col izero+1 to i2 */
            for (int i = izero + 1; i <= i2; i++)
                ws->A[(i - 1) * ldab + kd + izero - i] = 0.0f;
        } else {
            /* Lower: zero row izero (entries from col i1 to izero) */
            for (int i = i1; i < izero; i++)
                ws->A[(i - 1) * ldab + izero - i] = 0.0f;
            /* Zero column izero (entries from row izero to i2) */
            for (int i = izero; i <= i2; i++)
                ws->A[(izero - 1) * ldab + i - izero] = 0.0f;
        }
    }

    /* Save A */
    slacpy("Full", kd + 1, n, ws->A, ldab, ws->ASAV, ldab);

    /* Skip FACT='F' for singular matrices */
    if (zerot && prefac) return;

    /*
     * Compute condition number independently per test case.
     */
    f32 rcondc = 0.0f, roldc = 0.0f;
    f32 scond = 0.0f, amax = 0.0f;
    char equed = EQUEDS[iequed][0];

    if (zerot) {
        rcondc = 0.0f;
    } else {
        /* First compute condition of non-equilibrated matrix */
        slacpy("Full", kd + 1, n, ws->ASAV, ldab, ws->AFAC, ldab);

        f32 anrm = slansb("1", uplo, n, kd, ws->AFAC, ldab, ws->RWORK);
        spbtrf(uplo, n, kd, ws->AFAC, ldab, &info);

        slaset("Full", n, n, 0.0f, 1.0f, ws->WORK, lda);
        spbtrs(uplo, n, kd, n, ws->AFAC, ldab, ws->WORK, lda, &info);

        f32 ainvnm = slange("1", n, n, ws->WORK, lda, ws->RWORK);
        if (anrm <= 0.0f || ainvnm <= 0.0f)
            roldc = 1.0f;
        else
            roldc = (1.0f / anrm) / ainvnm;

        /* Now compute condition of equilibrated matrix if needed */
        if (equil || iequed > 0) {
            slacpy("Full", kd + 1, n, ws->ASAV, ldab, ws->AFAC, ldab);
            spbequ(uplo, n, kd, ws->AFAC, ldab, ws->S, &scond, &amax, &info);
            if (info == 0 && n > 0) {
                if (iequed > 0) scond = 0.0f;
                slaqsb(uplo, n, kd, ws->AFAC, ldab, ws->S, scond, amax, &equed);
            }

            anrm = slansb("1", uplo, n, kd, ws->AFAC, ldab, ws->RWORK);
            spbtrf(uplo, n, kd, ws->AFAC, ldab, &info);

            slaset("Full", n, n, 0.0f, 1.0f, ws->WORK, lda);
            spbtrs(uplo, n, kd, n, ws->AFAC, ldab, ws->WORK, lda, &info);

            ainvnm = slange("1", n, n, ws->WORK, lda, ws->RWORK);
            if (anrm <= 0.0f || ainvnm <= 0.0f)
                rcondc = 1.0f;
            else
                rcondc = (1.0f / anrm) / ainvnm;
        } else {
            rcondc = roldc;
        }
    }

    /* Restore A */
    slacpy("Full", kd + 1, n, ws->ASAV, ldab, ws->A, ldab);

    /* Generate exact solution and RHS */
    char xtype = 'N';
    slarhs("SPB", &xtype, uplo, " ", n, n, kd, kd, NRHS,
           ws->A, ldab, ws->XACT, lda, ws->B, lda, &info, rng_state);
    slacpy("Full", n, NRHS, ws->B, lda, ws->BSAV, lda);

    /* ====== SPBSV test ====== */
    if (nofact) {
        slacpy("Full", kd + 1, n, ws->A, ldab, ws->AFAC, ldab);
        slacpy("Full", n, NRHS, ws->B, lda, ws->X, lda);

        spbsv(uplo, n, kd, NRHS, ws->AFAC, ldab, ws->X, lda, &info);

        if (info != izero) {
            char msg[128];
            snprintf(msg, sizeof(msg),
                     "SPBSV info=%d expected=%d uplo=%s n=%d kd=%d imat=%d",
                     info, izero, uplo, n, kd, imat);
            fail_msg("%s", msg);
            return;
        }

        if (info == 0) {
            spbt01(uplo, n, kd, ws->A, ldab, ws->AFAC, ldab,
                   ws->RWORK, &result[0]);

            slacpy("Full", n, NRHS, ws->B, lda, ws->WORK, lda);
            spbt02(uplo, n, kd, NRHS, ws->A, ldab, ws->X, lda,
                   ws->WORK, lda, ws->RWORK, &result[1]);

            sget04(n, NRHS, ws->X, lda, ws->XACT, lda, rcondc, &result[2]);

            for (int k = 0; k < 3; k++) {
                if (result[k] >= THRESH) {
                    char msg[256];
                    snprintf(msg, sizeof(msg),
                             "SPBSV uplo=%s n=%d kd=%d imat=%d test(%d)=%.5g",
                             uplo, n, kd, imat, k + 1, (double)result[k]);
                    fail_msg("%s", msg);
                }
            }
        }
    }

    /* ====== SPBSVX test ====== */
    /* Restore A from ASAV */
    slacpy("Full", kd + 1, n, ws->ASAV, ldab, ws->A, ldab);

    if (!prefac)
        slaset("Full", kd + 1, n, 0.0f, 0.0f, ws->AFAC, ldab);
    slaset("Full", n, NRHS, 0.0f, 0.0f, ws->X, lda);

    equed = EQUEDS[iequed][0];

    if (iequed > 0 && n > 0) {
        slaqsb(uplo, n, kd, ws->A, ldab, ws->S, scond, amax, &equed);
    }

    /* Restore B from BSAV */
    slacpy("Full", n, NRHS, ws->BSAV, lda, ws->B, lda);

    f32 rcond;
    spbsvx(fact, uplo, n, kd, NRHS, ws->A, ldab, ws->AFAC, ldab,
           &equed, ws->S, ws->B, lda, ws->X, lda, &rcond,
           ws->RWORK, &ws->RWORK[NRHS], ws->WORK, ws->IWORK, &info);

    if (info != izero) {
        char msg[128];
        snprintf(msg, sizeof(msg),
                 "SPBSVX info=%d expected=%d fact=%s uplo=%s n=%d kd=%d imat=%d equed=%c",
                 info, izero, fact, uplo, n, kd, imat, equed);
        fail_msg("%s", msg);
        return;
    }

    int k1;
    if (info == 0) {
        if (!prefac) {
            spbt01(uplo, n, kd, ws->A, ldab, ws->AFAC, ldab,
                   &ws->RWORK[2 * NRHS], &result[0]);
            k1 = 0;
        } else {
            k1 = 1;
        }

        slacpy("Full", n, NRHS, ws->BSAV, lda, ws->WORK, lda);
        spbt02(uplo, n, kd, NRHS, ws->ASAV, ldab, ws->X, lda,
               ws->WORK, lda, &ws->RWORK[2 * NRHS], &result[1]);

        if (nofact || (prefac && (equed == 'N' || equed == 'n'))) {
            sget04(n, NRHS, ws->X, lda, ws->XACT, lda, rcondc, &result[2]);
        } else {
            sget04(n, NRHS, ws->X, lda, ws->XACT, lda, roldc, &result[2]);
        }

        spbt05(uplo, n, kd, NRHS, ws->ASAV, ldab, ws->B, lda,
               ws->X, lda, ws->XACT, lda,
               ws->RWORK, &ws->RWORK[NRHS], &result[3]);
    } else {
        k1 = 5;
    }

    result[5] = sget06(rcond, rcondc);

    for (int k = k1; k < NTESTS; k++) {
        if (result[k] >= THRESH) {
            char msg[256];
            if (prefac) {
                snprintf(msg, sizeof(msg),
                         "SPBSVX fact=%s uplo=%s n=%d kd=%d equed=%c imat=%d test(%d)=%.5g",
                         fact, uplo, n, kd, equed, imat, k + 1, (double)result[k]);
            } else {
                snprintf(msg, sizeof(msg),
                         "SPBSVX fact=%s uplo=%s n=%d kd=%d imat=%d test(%d)=%.5g",
                         fact, uplo, n, kd, imat, k + 1, (double)result[k]);
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
