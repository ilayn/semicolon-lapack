/**
 * @file test_sdrvge.c
 * @brief DDRVGE tests the driver routines SGESV and SGESVX.
 *
 * Port of LAPACK TESTING/LIN/ddrvge.f to C with CMocka parameterization.
 */

#include "test_harness.h"
#include "test_rng.h"
#include <string.h>
#include <stdio.h>
#include <cblas.h>
#include <math.h>

/* Test parameters - matching LAPACK dchkaa.f defaults */
static const int NVAL[] = {0, 1, 2, 3, 5, 10, 50};
#define NN      (sizeof(NVAL) / sizeof(NVAL[0]))
#define NTYPES  11
#define NTESTS  7
#define NTRAN   3
#define THRESH  30.0f
#define NMAX    50
#define NRHS    2

/* Routines under test */
extern void sgesv(const int n, const int nrhs, f32* A, const int lda,
                  int* ipiv, f32* B, const int ldb, int* info);
extern void sgesvx(const char* fact, const char* trans, const int n, const int nrhs,
                   f32* A, const int lda, f32* AF, const int ldaf,
                   int* ipiv, char* equed, f32* R, f32* C,
                   f32* B, const int ldb, f32* X, const int ldx,
                   f32* rcond, f32* ferr, f32* berr,
                   f32* work, int* iwork, int* info);

/* Supporting routines */
extern void sgetrf(const int m, const int n, f32* A, const int lda,
                   int* ipiv, int* info);
extern void sgetri(const int n, f32* A, const int lda, const int* ipiv,
                   f32* work, const int lwork, int* info);
extern void sgeequ(const int m, const int n, const f32* A, const int lda,
                   f32* R, f32* C, f32* rowcnd, f32* colcnd,
                   f32* amax, int* info);
extern void slaqge(const int m, const int n, f32* A, const int lda,
                   const f32* R, const f32* C, const f32 rowcnd,
                   const f32 colcnd, const f32 amax, char* equed);

/* Verification routines */
extern void sget01(const int m, const int n, const f32* A, const int lda,
                   const f32* AFAC, const int ldafac, const int* ipiv,
                   f32* rwork, f32* resid);
extern void sget02(const char* trans, const int m, const int n, const int nrhs,
                   const f32* A, const int lda, const f32* X, const int ldx,
                   f32* B, const int ldb, f32* rwork, f32* resid);
extern void sget04(const int n, const int nrhs, const f32* X, const int ldx,
                   const f32* XACT, const int ldxact, const f32 rcond,
                   f32* resid);
extern f32 sget06(const f32 rcond, const f32 rcondc);
extern void sget07(const char* trans, const int n, const int nrhs,
                   const f32* A, const int lda, const f32* B, const int ldb,
                   const f32* X, const int ldx, const f32* XACT, const int ldxact,
                   const f32* ferr, const int chkferr, const f32* berr,
                   f32* reslts);

/* Matrix generation */
extern void slatb4(const char* path, const int imat, const int m, const int n,
                   char* type, int* kl, int* ku, f32* anorm, int* mode,
                   f32* cndnum, char* dist);
extern void slatms(const int m, const int n, const char* dist,
                   const char* sym, f32* d,
                   const int mode, const f32 cond, const f32 dmax,
                   const int kl, const int ku, const char* pack,
                   f32* A, const int lda, f32* work, int* info,
                   uint64_t state[static 4]);
extern void slarhs(const char* path, const char* xtype, const char* uplo,
                   const char* trans, const int m, const int n,
                   const int kl, const int ku, const int nrhs,
                   const f32* A, const int lda, const f32* XACT, const int ldxact,
                   f32* B, const int ldb, int* info, uint64_t state[static 4]);

/* Utilities */
extern void slacpy(const char* uplo, const int m, const int n,
                   const f32* A, const int lda, f32* B, const int ldb);
extern void slaset(const char* uplo, const int m, const int n,
                   const f32 alpha, const f32 beta,
                   f32* A, const int lda);
extern f32 slange(const char* norm, const int m, const int n,
                     const f32* A, const int lda, f32* work);
extern f32 slantr(const char* norm, const char* uplo, const char* diag,
                     const int m, const int n, const f32* A, const int lda,
                     f32* work);
extern f32 slamch(const char* cmach);

typedef struct {
    int n;
    int imat;
    int ifact;      /* 0='F', 1='N', 2='E' */
    int itran;      /* 0='N', 1='T', 2='C' */
    int iequed;     /* 0='N', 1='R', 2='C', 3='B' */
    char name[64];
} ddrvge_params_t;

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
    int lwork;
} ddrvge_workspace_t;

static ddrvge_workspace_t* g_workspace = NULL;

static int group_setup(void** state)
{
    (void)state;
    g_workspace = malloc(sizeof(ddrvge_workspace_t));
    if (!g_workspace) return -1;

    int nmax = NMAX;
    int lwork = nmax * (nmax > 3 ? nmax : 3);
    if (lwork < nmax * NRHS) lwork = nmax * NRHS;

    g_workspace->lwork = lwork;
    g_workspace->A = calloc(nmax * nmax, sizeof(f32));
    g_workspace->AFAC = calloc(nmax * nmax, sizeof(f32));
    g_workspace->ASAV = calloc(nmax * nmax, sizeof(f32));
    g_workspace->B = calloc(nmax * NRHS, sizeof(f32));
    g_workspace->BSAV = calloc(nmax * NRHS, sizeof(f32));
    g_workspace->X = calloc(nmax * NRHS, sizeof(f32));
    g_workspace->XACT = calloc(nmax * NRHS, sizeof(f32));
    g_workspace->S = calloc(2 * nmax, sizeof(f32));
    g_workspace->WORK = calloc(lwork, sizeof(f32));
    g_workspace->RWORK = calloc(2 * NRHS + nmax, sizeof(f32));
    g_workspace->IWORK = calloc(2 * nmax, sizeof(int));

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

/**
 * Compute condition number of matrix A by explicit inversion.
 * Returns rcond for the given norm ('1' or 'I').
 */
static f32 compute_rcond(ddrvge_workspace_t* ws, int n, int lda, const char* norm)
{
    if (n == 0) return 1.0f;

    int info;
    f32 anrm = slange(norm, n, n, ws->AFAC, lda, ws->RWORK);

    sgetrf(n, n, ws->AFAC, lda, ws->IWORK, &info);
    if (info != 0) return 0.0f;

    slacpy("Full", n, n, ws->AFAC, lda, ws->A, lda);
    int lwork_getri = NMAX * 3;
    sgetri(n, ws->A, lda, ws->IWORK, ws->WORK, lwork_getri, &info);
    if (info != 0) return 0.0f;

    f32 ainvnm = slange(norm, n, n, ws->A, lda, ws->RWORK);
    if (anrm <= 0.0f || ainvnm <= 0.0f) return 1.0f;

    return (1.0f / anrm) / ainvnm;
}

static void run_ddrvge_single(int n, int imat, int ifact, int itran, int iequed)
{
    static const char* FACTS[] = {"F", "N", "E"};
    static const char* TRANSS[] = {"N", "T", "C"};
    static const char* EQUEDS[] = {"N", "R", "C", "B"};

    ddrvge_workspace_t* ws = g_workspace;
    const char* fact = FACTS[ifact];
    const char* trans = TRANSS[itran];
    char equed = EQUEDS[iequed][0];

    int prefac = (fact[0] == 'F');
    int nofact = (fact[0] == 'N');
    int equil = (fact[0] == 'E');

    int lda = (n > 1) ? n : 1;
    f32 result[NTESTS];
    for (int k = 0; k < NTESTS; k++) result[k] = 0.0f;

    /* Set up parameters with SLATB4 */
    char type, dist;
    int kl, ku, mode;
    f32 anorm, cndnum;
    slatb4("SGE", imat, n, n, &type, &kl, &ku, &anorm, &mode, &cndnum, &dist);

    /* Generate test matrix with SLATMS */
    uint64_t rng_state[4];
    rng_seed(rng_state, 1988 + n * 1000 + imat * 100);
    int info;
    slatms(n, n, &dist, &type, ws->RWORK, mode, cndnum,
           anorm, kl, ku, "N", ws->A, lda, ws->WORK, &info, rng_state);
    if (info != 0) {
        fail_msg("SLATMS info=%d", info);
        return;
    }

    /* For types 5-7, zero one or more columns */
    int izero = 0;
    int zerot = (imat >= 5 && imat <= 7);
    if (zerot) {
        if (imat == 5) izero = 1;
        else if (imat == 6) izero = n;
        else izero = n / 2 + 1;

        int ioff = (izero - 1) * lda;
        if (imat < 7) {
            for (int i = 0; i < n; i++) ws->A[ioff + i] = 0.0f;
        } else {
            slaset("Full", n, n - izero + 1, 0.0f, 0.0f, &ws->A[ioff], lda);
        }
    }

    /* Save a copy of the matrix A in ASAV */
    slacpy("Full", n, n, ws->A, lda, ws->ASAV, lda);

    /* Skip FACT='F' for singular matrices (lines 351-353) */
    if (zerot && prefac) {
        return;
    }

    /*
     * Compute condition numbers for verification.
     *
     * LAPACK computes:
     * - RCONDO/RCONDI: condition of (possibly equilibrated) matrix for SGET06
     * - ROLDO/ROLDI: condition of non-equilibrated matrix for SGET04 when EQUIL
     *
     * For parameterized tests, we compute both as needed.
     */
    f32 rcondo = 0.0f, rcondi = 0.0f;
    f32 roldo = 0.0f, roldi = 0.0f;
    f32 rowcnd = 0.0f, colcnd = 0.0f, amax = 0.0f;

    if (zerot) {
        rcondo = 0.0f;
        rcondi = 0.0f;
    } else if (n > 0) {
        /* First compute condition of non-equilibrated matrix (for ROLDO/ROLDI) */
        slacpy("Full", n, n, ws->ASAV, lda, ws->AFAC, lda);
        roldo = compute_rcond(ws, n, lda, "1");

        slacpy("Full", n, n, ws->ASAV, lda, ws->AFAC, lda);
        roldi = compute_rcond(ws, n, lda, "I");

        /* Now compute condition of equilibrated matrix if needed */
        if (equil || iequed > 0) {
            slacpy("Full", n, n, ws->ASAV, lda, ws->AFAC, lda);

            sgeequ(n, n, ws->AFAC, lda, ws->S, &ws->S[n],
                   &rowcnd, &colcnd, &amax, &info);

            if (info == 0 && n > 0) {
                /* Force equilibration type based on EQUED */
                if (equed == 'R' || equed == 'r') {
                    rowcnd = 0.0f; colcnd = 1.0f;
                } else if (equed == 'C' || equed == 'c') {
                    rowcnd = 1.0f; colcnd = 0.0f;
                } else if (equed == 'B' || equed == 'b') {
                    rowcnd = 0.0f; colcnd = 0.0f;
                }

                char equed_out;
                slaqge(n, n, ws->AFAC, lda, ws->S, &ws->S[n],
                       rowcnd, colcnd, amax, &equed_out);

                /* Update equed to what was actually done */
                equed = equed_out;
            }

            rcondo = compute_rcond(ws, n, lda, "1");

            /* Recompute for infinity norm */
            slacpy("Full", n, n, ws->ASAV, lda, ws->AFAC, lda);
            if (info == 0 && n > 0) {
                char equed_out;
                slaqge(n, n, ws->AFAC, lda, ws->S, &ws->S[n],
                       rowcnd, colcnd, amax, &equed_out);
            }
            rcondi = compute_rcond(ws, n, lda, "I");
        } else {
            /* No equilibration - use the non-equilibrated condition */
            rcondo = roldo;
            rcondi = roldi;
        }
    }

    /* Select condition number based on TRANS (lines 440-444) */
    f32 rcondc, roldc;
    if (n == 0) {
        /* For n=0, use RCONDC = 1/CNDNUM from slatb4 (line 294) */
        rcondc = 1.0f / cndnum;
        roldc = rcondc;
    } else {
        rcondc = (itran == 0) ? rcondo : rcondi;
        roldc = (itran == 0) ? roldo : roldi;
    }

    /* Restore the matrix A (line 448) */
    slacpy("Full", n, n, ws->ASAV, lda, ws->A, lda);

    /* Form exact solution and set right hand side (lines 452-457) */
    rng_seed(rng_state, 1988 + n * 1000 + imat * 100 + itran);
    char xtype = 'N';
    slarhs("SGE", &xtype, "Full", trans, n, n, kl, ku,
           NRHS, ws->A, lda, ws->XACT, lda, ws->B, lda, &info, rng_state);
    slacpy("Full", n, NRHS, ws->B, lda, ws->BSAV, lda);

    /* --- Test SGESV (lines 459-516) --- */
    if (nofact && itran == 0) {
        slacpy("Full", n, n, ws->A, lda, ws->AFAC, lda);
        slacpy("Full", n, NRHS, ws->B, lda, ws->X, lda);

        sgesv(n, NRHS, ws->AFAC, lda, ws->IWORK, ws->X, lda, &info);

        if (info != izero) {
            fail_msg("SGESV info=%d expected=%d", info, izero);
        }

        /* TEST 1: Reconstruct matrix from factors */
        sget01(n, n, ws->A, lda, ws->AFAC, lda, ws->IWORK,
               ws->RWORK, &result[0]);

        int nt = 1;
        if (izero == 0) {
            /* TEST 2: Compute residual of computed solution */
            slacpy("Full", n, NRHS, ws->B, lda, ws->WORK, lda);
            sget02("N", n, n, NRHS, ws->A, lda, ws->X, lda,
                   ws->WORK, lda, ws->RWORK, &result[1]);

            /* TEST 3: Check solution from generated exact solution */
            sget04(n, NRHS, ws->X, lda, ws->XACT, lda, rcondc, &result[2]);
            nt = 3;
        }

        for (int k = 0; k < nt; k++) {
            if (result[k] >= THRESH) {
                fail_msg("SGESV test %d failed: result=%e >= thresh=%e",
                         k + 1, (double)result[k], (double)THRESH);
            }
        }
    }

    /* --- Test SGESVX (lines 518-692) --- */

    /* Zero AFAC if not prefactored (lines 520-522) */
    if (!prefac) {
        slaset("Full", n, n, 0.0f, 0.0f, ws->AFAC, lda);
    }
    slaset("Full", n, NRHS, 0.0f, 0.0f, ws->X, lda);

    /* Equilibrate matrix if FACT='F' and EQUED != 'N' (lines 524-531) */
    if (iequed > 0 && n > 0) {
        char equed_out;
        slaqge(n, n, ws->A, lda, ws->S, &ws->S[n],
               rowcnd, colcnd, amax, &equed_out);
        equed = equed_out;
    }

    /* Call SGESVX (lines 537-541) */
    char equed_inout = equed;
    f32 rcond;
    sgesvx(fact, trans, n, NRHS, ws->A, lda, ws->AFAC, lda,
           ws->IWORK, &equed_inout, ws->S, &ws->S[n],
           ws->B, lda, ws->X, lda, &rcond,
           ws->RWORK, &ws->RWORK[NRHS], ws->WORK,
           &ws->IWORK[n], &info);

    /* Check error code (lines 545-548) */
    if (info != izero) {
        if (!(zerot && info > 0 && info <= n)) {
            fail_msg("SGESVX info=%d expected=%d", info, izero);
        }
    }

    /* TEST 7: Compare RPVGRW (lines 553-574) */
    f32 rpvgrw;
    if (info != 0 && info <= n) {
        rpvgrw = slantr("M", "U", "N", info, info, ws->AFAC, lda, ws->WORK);
        if (rpvgrw == 0.0f) {
            rpvgrw = 1.0f;
        } else {
            rpvgrw = slange("M", n, info, ws->A, lda, ws->WORK) / rpvgrw;
        }
    } else {
        rpvgrw = slantr("M", "U", "N", n, n, ws->AFAC, lda, ws->WORK);
        if (rpvgrw == 0.0f) {
            rpvgrw = 1.0f;
        } else {
            rpvgrw = slange("M", n, n, ws->A, lda, ws->WORK) / rpvgrw;
        }
    }
    f32 work0 = ws->WORK[0];
    result[6] = fabsf(rpvgrw - work0) / fmaxf(work0, rpvgrw) / slamch("E");

    /* TEST 1: Reconstruct matrix from factors (lines 576-586)
     * K1 determines which tests to check: 0 if test 1 ran, 1 if skipped (0-indexed) */
    int k1;
    if (!prefac) {
        sget01(n, n, ws->A, lda, ws->AFAC, lda, ws->IWORK,
               &ws->RWORK[2 * NRHS], &result[0]);
        k1 = 0;
    } else {
        k1 = 1;
    }

    int trfcon;
    if (info == 0) {
        trfcon = 0;

        /* TEST 2: Compute residual of computed solution (lines 593-597) */
        slacpy("Full", n, NRHS, ws->BSAV, lda, ws->WORK, lda);
        sget02(trans, n, n, NRHS, ws->ASAV, lda, ws->X, lda,
               ws->WORK, lda, &ws->RWORK[2 * NRHS], &result[1]);

        /* TEST 3: Check solution from generated exact solution (lines 601-613)
         *
         * Use RCONDC for:  NOFACT, or PREFAC with EQUED='N'
         * Use ROLDC for:   EQUIL, or PREFAC with EQUED!='N'
         */
        f32 rcond_for_get04;
        if (nofact || (prefac && (equed == 'N' || equed == 'n'))) {
            rcond_for_get04 = rcondc;
        } else {
            rcond_for_get04 = roldc;
        }
        sget04(n, NRHS, ws->X, lda, ws->XACT, lda, rcond_for_get04, &result[2]);

        /* TEST 4-5: Check error bounds (lines 618-620) */
        sget07(trans, n, NRHS, ws->ASAV, lda, ws->B, lda,
               ws->X, lda, ws->XACT, lda,
               ws->RWORK, 1, &ws->RWORK[NRHS], &result[3]);
    } else {
        trfcon = 1;
    }

    /* TEST 6: Compare RCOND from SGESVX with computed value (line 628) */
    result[5] = sget06(rcond, rcondc);

    /* Check results (lines 633-692) */
    if (!trfcon) {
        for (int k = k1; k < NTESTS; k++) {
            if (result[k] >= THRESH) {
                fail_msg("SGESVX FACT=%s TRANS=%s EQUED=%c test %d: result=%e >= thresh=%e",
                         fact, trans, equed, k + 1, (double)result[k], (double)THRESH);
            }
        }
    } else {
        /* TRFCON case: only check tests 1, 6, 7 */
        if (!prefac && result[0] >= THRESH) {
            fail_msg("SGESVX FACT=%s TRANS=%s EQUED=%c test 1: result=%e >= thresh=%e",
                     fact, trans, equed, (double)result[0], (double)THRESH);
        }
        if (result[5] >= THRESH) {
            fail_msg("SGESVX FACT=%s TRANS=%s EQUED=%c test 6: result=%e >= thresh=%e",
                     fact, trans, equed, (double)result[5], (double)THRESH);
        }
        if (result[6] >= THRESH) {
            fail_msg("SGESVX FACT=%s TRANS=%s EQUED=%c test 7: result=%e >= thresh=%e",
                     fact, trans, equed, (double)result[6], (double)THRESH);
        }
    }
}

static void test_ddrvge_case(void** state)
{
    ddrvge_params_t* p = *state;
    run_ddrvge_single(p->n, p->imat, p->ifact, p->itran, p->iequed);
}

#define MAX_TESTS 3000

static ddrvge_params_t g_params[MAX_TESTS];
static struct CMUnitTest g_tests[MAX_TESTS];
static int g_num_tests = 0;

static void build_test_array(void)
{
    static const char* FACTS[] = {"F", "N", "E"};
    static const char* TRANSS[] = {"N", "T", "C"};
    static const char* EQUEDS[] = {"N", "R", "C", "B"};

    g_num_tests = 0;

    for (int in = 0; in < (int)NN; in++) {
        int n = NVAL[in];
        int nimat = (n <= 0) ? 1 : NTYPES;

        for (int imat = 1; imat <= nimat; imat++) {
            int zerot = (imat >= 5 && imat <= 7);
            if (zerot && n < imat - 4) continue;

            for (int iequed = 0; iequed < 4; iequed++) {
                int nfact = (iequed == 0) ? 3 : 1;

                for (int ifact = 0; ifact < nfact; ifact++) {
                    if (zerot && ifact == 0) continue;

                    for (int itran = 0; itran < NTRAN; itran++) {
                        ddrvge_params_t* p = &g_params[g_num_tests];
                        p->n = n;
                        p->imat = imat;
                        p->ifact = ifact;
                        p->itran = itran;
                        p->iequed = iequed;
                        snprintf(p->name, sizeof(p->name),
                                 "n%d_t%d_%s_%s_%s",
                                 n, imat, FACTS[ifact], TRANSS[itran], EQUEDS[iequed]);

                        g_tests[g_num_tests].name = p->name;
                        g_tests[g_num_tests].test_func = test_ddrvge_case;
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
    return _cmocka_run_group_tests("ddrvge", g_tests, g_num_tests,
                                   group_setup, group_teardown);
}
