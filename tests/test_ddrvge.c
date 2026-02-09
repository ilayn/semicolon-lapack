/**
 * @file test_ddrvge.c
 * @brief DDRVGE tests the driver routines DGESV and DGESVX.
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
#define THRESH  30.0
#define NMAX    50
#define NRHS    2

/* Routines under test */
extern void dgesv(const int n, const int nrhs, double* A, const int lda,
                  int* ipiv, double* B, const int ldb, int* info);
extern void dgesvx(const char* fact, const char* trans, const int n, const int nrhs,
                   double* A, const int lda, double* AF, const int ldaf,
                   int* ipiv, char* equed, double* R, double* C,
                   double* B, const int ldb, double* X, const int ldx,
                   double* rcond, double* ferr, double* berr,
                   double* work, int* iwork, int* info);

/* Supporting routines */
extern void dgetrf(const int m, const int n, double* A, const int lda,
                   int* ipiv, int* info);
extern void dgetri(const int n, double* A, const int lda, const int* ipiv,
                   double* work, const int lwork, int* info);
extern void dgeequ(const int m, const int n, const double* A, const int lda,
                   double* R, double* C, double* rowcnd, double* colcnd,
                   double* amax, int* info);
extern void dlaqge(const int m, const int n, double* A, const int lda,
                   const double* R, const double* C, const double rowcnd,
                   const double colcnd, const double amax, char* equed);

/* Verification routines */
extern void dget01(const int m, const int n, const double* A, const int lda,
                   const double* AFAC, const int ldafac, const int* ipiv,
                   double* rwork, double* resid);
extern void dget02(const char* trans, const int m, const int n, const int nrhs,
                   const double* A, const int lda, const double* X, const int ldx,
                   double* B, const int ldb, double* rwork, double* resid);
extern void dget04(const int n, const int nrhs, const double* X, const int ldx,
                   const double* XACT, const int ldxact, const double rcond,
                   double* resid);
extern double dget06(const double rcond, const double rcondc);
extern void dget07(const char* trans, const int n, const int nrhs,
                   const double* A, const int lda, const double* B, const int ldb,
                   const double* X, const int ldx, const double* XACT, const int ldxact,
                   const double* ferr, const int chkferr, const double* berr,
                   double* reslts);

/* Matrix generation */
extern void dlatb4(const char* path, const int imat, const int m, const int n,
                   char* type, int* kl, int* ku, double* anorm, int* mode,
                   double* cndnum, char* dist);
extern void dlatms(const int m, const int n, const char* dist,
                   const char* sym, double* d,
                   const int mode, const double cond, const double dmax,
                   const int kl, const int ku, const char* pack,
                   double* A, const int lda, double* work, int* info,
                   uint64_t state[static 4]);
extern void dlarhs(const char* path, const char* xtype, const char* uplo,
                   const char* trans, const int m, const int n,
                   const int kl, const int ku, const int nrhs,
                   const double* A, const int lda, const double* XACT, const int ldxact,
                   double* B, const int ldb, int* info, uint64_t state[static 4]);

/* Utilities */
extern void dlacpy(const char* uplo, const int m, const int n,
                   const double* A, const int lda, double* B, const int ldb);
extern void dlaset(const char* uplo, const int m, const int n,
                   const double alpha, const double beta,
                   double* A, const int lda);
extern double dlange(const char* norm, const int m, const int n,
                     const double* A, const int lda, double* work);
extern double dlantr(const char* norm, const char* uplo, const char* diag,
                     const int m, const int n, const double* A, const int lda,
                     double* work);
extern double dlamch(const char* cmach);

typedef struct {
    int n;
    int imat;
    int ifact;      /* 0='F', 1='N', 2='E' */
    int itran;      /* 0='N', 1='T', 2='C' */
    int iequed;     /* 0='N', 1='R', 2='C', 3='B' */
    char name[64];
} ddrvge_params_t;

typedef struct {
    double* A;
    double* AFAC;
    double* ASAV;
    double* B;
    double* BSAV;
    double* X;
    double* XACT;
    double* S;
    double* WORK;
    double* RWORK;
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
    g_workspace->A = calloc(nmax * nmax, sizeof(double));
    g_workspace->AFAC = calloc(nmax * nmax, sizeof(double));
    g_workspace->ASAV = calloc(nmax * nmax, sizeof(double));
    g_workspace->B = calloc(nmax * NRHS, sizeof(double));
    g_workspace->BSAV = calloc(nmax * NRHS, sizeof(double));
    g_workspace->X = calloc(nmax * NRHS, sizeof(double));
    g_workspace->XACT = calloc(nmax * NRHS, sizeof(double));
    g_workspace->S = calloc(2 * nmax, sizeof(double));
    g_workspace->WORK = calloc(lwork, sizeof(double));
    g_workspace->RWORK = calloc(2 * NRHS + nmax, sizeof(double));
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
static double compute_rcond(ddrvge_workspace_t* ws, int n, int lda, const char* norm)
{
    if (n == 0) return 1.0;

    int info;
    double anrm = dlange(norm, n, n, ws->AFAC, lda, ws->RWORK);

    dgetrf(n, n, ws->AFAC, lda, ws->IWORK, &info);
    if (info != 0) return 0.0;

    dlacpy("Full", n, n, ws->AFAC, lda, ws->A, lda);
    int lwork_getri = NMAX * 3;
    dgetri(n, ws->A, lda, ws->IWORK, ws->WORK, lwork_getri, &info);
    if (info != 0) return 0.0;

    double ainvnm = dlange(norm, n, n, ws->A, lda, ws->RWORK);
    if (anrm <= 0.0 || ainvnm <= 0.0) return 1.0;

    return (1.0 / anrm) / ainvnm;
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
    double result[NTESTS];
    for (int k = 0; k < NTESTS; k++) result[k] = 0.0;

    /* Set up parameters with DLATB4 */
    char type, dist;
    int kl, ku, mode;
    double anorm, cndnum;
    dlatb4("DGE", imat, n, n, &type, &kl, &ku, &anorm, &mode, &cndnum, &dist);

    /* Generate test matrix with DLATMS */
    uint64_t rng_state[4];
    rng_seed(rng_state, 1988 + n * 1000 + imat * 100);
    int info;
    dlatms(n, n, &dist, &type, ws->RWORK, mode, cndnum,
           anorm, kl, ku, "N", ws->A, lda, ws->WORK, &info, rng_state);
    if (info != 0) {
        fail_msg("DLATMS info=%d", info);
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
            for (int i = 0; i < n; i++) ws->A[ioff + i] = 0.0;
        } else {
            dlaset("Full", n, n - izero + 1, 0.0, 0.0, &ws->A[ioff], lda);
        }
    }

    /* Save a copy of the matrix A in ASAV */
    dlacpy("Full", n, n, ws->A, lda, ws->ASAV, lda);

    /* Skip FACT='F' for singular matrices (lines 351-353) */
    if (zerot && prefac) {
        return;
    }

    /*
     * Compute condition numbers for verification.
     *
     * LAPACK computes:
     * - RCONDO/RCONDI: condition of (possibly equilibrated) matrix for DGET06
     * - ROLDO/ROLDI: condition of non-equilibrated matrix for DGET04 when EQUIL
     *
     * For parameterized tests, we compute both as needed.
     */
    double rcondo = 0.0, rcondi = 0.0;
    double roldo = 0.0, roldi = 0.0;
    double rowcnd = 0.0, colcnd = 0.0, amax = 0.0;

    if (zerot) {
        rcondo = 0.0;
        rcondi = 0.0;
    } else if (n > 0) {
        /* First compute condition of non-equilibrated matrix (for ROLDO/ROLDI) */
        dlacpy("Full", n, n, ws->ASAV, lda, ws->AFAC, lda);
        roldo = compute_rcond(ws, n, lda, "1");

        dlacpy("Full", n, n, ws->ASAV, lda, ws->AFAC, lda);
        roldi = compute_rcond(ws, n, lda, "I");

        /* Now compute condition of equilibrated matrix if needed */
        if (equil || iequed > 0) {
            dlacpy("Full", n, n, ws->ASAV, lda, ws->AFAC, lda);

            dgeequ(n, n, ws->AFAC, lda, ws->S, &ws->S[n],
                   &rowcnd, &colcnd, &amax, &info);

            if (info == 0 && n > 0) {
                /* Force equilibration type based on EQUED */
                if (equed == 'R' || equed == 'r') {
                    rowcnd = 0.0; colcnd = 1.0;
                } else if (equed == 'C' || equed == 'c') {
                    rowcnd = 1.0; colcnd = 0.0;
                } else if (equed == 'B' || equed == 'b') {
                    rowcnd = 0.0; colcnd = 0.0;
                }

                char equed_out;
                dlaqge(n, n, ws->AFAC, lda, ws->S, &ws->S[n],
                       rowcnd, colcnd, amax, &equed_out);

                /* Update equed to what was actually done */
                equed = equed_out;
            }

            rcondo = compute_rcond(ws, n, lda, "1");

            /* Recompute for infinity norm */
            dlacpy("Full", n, n, ws->ASAV, lda, ws->AFAC, lda);
            if (info == 0 && n > 0) {
                char equed_out;
                dlaqge(n, n, ws->AFAC, lda, ws->S, &ws->S[n],
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
    double rcondc, roldc;
    if (n == 0) {
        /* For n=0, use RCONDC = 1/CNDNUM from dlatb4 (line 294) */
        rcondc = 1.0 / cndnum;
        roldc = rcondc;
    } else {
        rcondc = (itran == 0) ? rcondo : rcondi;
        roldc = (itran == 0) ? roldo : roldi;
    }

    /* Restore the matrix A (line 448) */
    dlacpy("Full", n, n, ws->ASAV, lda, ws->A, lda);

    /* Form exact solution and set right hand side (lines 452-457) */
    rng_seed(rng_state, 1988 + n * 1000 + imat * 100 + itran);
    char xtype = 'N';
    dlarhs("DGE", &xtype, "Full", trans, n, n, kl, ku,
           NRHS, ws->A, lda, ws->XACT, lda, ws->B, lda, &info, rng_state);
    dlacpy("Full", n, NRHS, ws->B, lda, ws->BSAV, lda);

    /* --- Test DGESV (lines 459-516) --- */
    if (nofact && itran == 0) {
        dlacpy("Full", n, n, ws->A, lda, ws->AFAC, lda);
        dlacpy("Full", n, NRHS, ws->B, lda, ws->X, lda);

        dgesv(n, NRHS, ws->AFAC, lda, ws->IWORK, ws->X, lda, &info);

        if (info != izero) {
            fail_msg("DGESV info=%d expected=%d", info, izero);
        }

        /* TEST 1: Reconstruct matrix from factors */
        dget01(n, n, ws->A, lda, ws->AFAC, lda, ws->IWORK,
               ws->RWORK, &result[0]);

        int nt = 1;
        if (izero == 0) {
            /* TEST 2: Compute residual of computed solution */
            dlacpy("Full", n, NRHS, ws->B, lda, ws->WORK, lda);
            dget02("N", n, n, NRHS, ws->A, lda, ws->X, lda,
                   ws->WORK, lda, ws->RWORK, &result[1]);

            /* TEST 3: Check solution from generated exact solution */
            dget04(n, NRHS, ws->X, lda, ws->XACT, lda, rcondc, &result[2]);
            nt = 3;
        }

        for (int k = 0; k < nt; k++) {
            if (result[k] >= THRESH) {
                fail_msg("DGESV test %d failed: result=%e >= thresh=%e",
                         k + 1, result[k], THRESH);
            }
        }
    }

    /* --- Test DGESVX (lines 518-692) --- */

    /* Zero AFAC if not prefactored (lines 520-522) */
    if (!prefac) {
        dlaset("Full", n, n, 0.0, 0.0, ws->AFAC, lda);
    }
    dlaset("Full", n, NRHS, 0.0, 0.0, ws->X, lda);

    /* Equilibrate matrix if FACT='F' and EQUED != 'N' (lines 524-531) */
    if (iequed > 0 && n > 0) {
        char equed_out;
        dlaqge(n, n, ws->A, lda, ws->S, &ws->S[n],
               rowcnd, colcnd, amax, &equed_out);
        equed = equed_out;
    }

    /* Call DGESVX (lines 537-541) */
    char equed_inout = equed;
    double rcond;
    dgesvx(fact, trans, n, NRHS, ws->A, lda, ws->AFAC, lda,
           ws->IWORK, &equed_inout, ws->S, &ws->S[n],
           ws->B, lda, ws->X, lda, &rcond,
           ws->RWORK, &ws->RWORK[NRHS], ws->WORK,
           &ws->IWORK[n], &info);

    /* Check error code (lines 545-548) */
    if (info != izero) {
        if (!(zerot && info > 0 && info <= n)) {
            fail_msg("DGESVX info=%d expected=%d", info, izero);
        }
    }

    /* TEST 7: Compare RPVGRW (lines 553-574) */
    double rpvgrw;
    if (info != 0 && info <= n) {
        rpvgrw = dlantr("M", "U", "N", info, info, ws->AFAC, lda, ws->WORK);
        if (rpvgrw == 0.0) {
            rpvgrw = 1.0;
        } else {
            rpvgrw = dlange("M", n, info, ws->A, lda, ws->WORK) / rpvgrw;
        }
    } else {
        rpvgrw = dlantr("M", "U", "N", n, n, ws->AFAC, lda, ws->WORK);
        if (rpvgrw == 0.0) {
            rpvgrw = 1.0;
        } else {
            rpvgrw = dlange("M", n, n, ws->A, lda, ws->WORK) / rpvgrw;
        }
    }
    double work0 = ws->WORK[0];
    result[6] = fabs(rpvgrw - work0) / fmax(work0, rpvgrw) / dlamch("E");

    /* TEST 1: Reconstruct matrix from factors (lines 576-586)
     * K1 determines which tests to check: 0 if test 1 ran, 1 if skipped (0-indexed) */
    int k1;
    if (!prefac) {
        dget01(n, n, ws->A, lda, ws->AFAC, lda, ws->IWORK,
               &ws->RWORK[2 * NRHS], &result[0]);
        k1 = 0;
    } else {
        k1 = 1;
    }

    int trfcon;
    if (info == 0) {
        trfcon = 0;

        /* TEST 2: Compute residual of computed solution (lines 593-597) */
        dlacpy("Full", n, NRHS, ws->BSAV, lda, ws->WORK, lda);
        dget02(trans, n, n, NRHS, ws->ASAV, lda, ws->X, lda,
               ws->WORK, lda, &ws->RWORK[2 * NRHS], &result[1]);

        /* TEST 3: Check solution from generated exact solution (lines 601-613)
         *
         * Use RCONDC for:  NOFACT, or PREFAC with EQUED='N'
         * Use ROLDC for:   EQUIL, or PREFAC with EQUED!='N'
         */
        double rcond_for_get04;
        if (nofact || (prefac && (equed == 'N' || equed == 'n'))) {
            rcond_for_get04 = rcondc;
        } else {
            rcond_for_get04 = roldc;
        }
        dget04(n, NRHS, ws->X, lda, ws->XACT, lda, rcond_for_get04, &result[2]);

        /* TEST 4-5: Check error bounds (lines 618-620) */
        dget07(trans, n, NRHS, ws->ASAV, lda, ws->B, lda,
               ws->X, lda, ws->XACT, lda,
               ws->RWORK, 1, &ws->RWORK[NRHS], &result[3]);
    } else {
        trfcon = 1;
    }

    /* TEST 6: Compare RCOND from DGESVX with computed value (line 628) */
    result[5] = dget06(rcond, rcondc);

    /* Check results (lines 633-692) */
    if (!trfcon) {
        for (int k = k1; k < NTESTS; k++) {
            if (result[k] >= THRESH) {
                fail_msg("DGESVX FACT=%s TRANS=%s EQUED=%c test %d: result=%e >= thresh=%e",
                         fact, trans, equed, k + 1, result[k], THRESH);
            }
        }
    } else {
        /* TRFCON case: only check tests 1, 6, 7 */
        if (!prefac && result[0] >= THRESH) {
            fail_msg("DGESVX FACT=%s TRANS=%s EQUED=%c test 1: result=%e >= thresh=%e",
                     fact, trans, equed, result[0], THRESH);
        }
        if (result[5] >= THRESH) {
            fail_msg("DGESVX FACT=%s TRANS=%s EQUED=%c test 6: result=%e >= thresh=%e",
                     fact, trans, equed, result[5], THRESH);
        }
        if (result[6] >= THRESH) {
            fail_msg("DGESVX FACT=%s TRANS=%s EQUED=%c test 7: result=%e >= thresh=%e",
                     fact, trans, equed, result[6], THRESH);
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
