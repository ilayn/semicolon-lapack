/**
 * @file test_ddrvpo.c
 * @brief DDRVPO tests the driver routines DPOSV and DPOSVX.
 *
 * Port of LAPACK TESTING/LIN/ddrvpo.f to C with CMocka parameterization.
 */

#include "test_harness.h"
#include <string.h>
#include <stdio.h>
#include <cblas.h>
#include <math.h>

/* Test parameters - matching LAPACK dchkaa.f defaults */
static const int NVAL[] = {0, 1, 2, 3, 5, 10, 50};
#define NN      (sizeof(NVAL) / sizeof(NVAL[0]))
#define NTYPES  9
#define NTESTS  6
#define THRESH  30.0
#define NMAX    50
#define NRHS    2

/* Routines under test */
extern void dposv(const char* uplo, const int n, const int nrhs,
                  double* A, const int lda, double* B, const int ldb, int* info);
extern void dposvx(const char* fact, const char* uplo, const int n, const int nrhs,
                   double* A, const int lda, double* AF, const int ldaf,
                   char* equed, double* S, double* B, const int ldb,
                   double* X, const int ldx, double* rcond,
                   double* ferr, double* berr, double* work, int* iwork, int* info);

/* Supporting routines */
extern void dpotrf(const char* uplo, const int n, double* A, const int lda, int* info);
extern void dpotri(const char* uplo, const int n, double* A, const int lda, int* info);
extern void dpoequ(const int n, const double* A, const int lda,
                   double* S, double* scond, double* amax, int* info);
extern void dlaqsy(const char* uplo, const int n, double* A, const int lda,
                   const double* S, const double scond, const double amax, char* equed);

/* Verification routines */
extern void dpot01(const char* uplo, const int n, const double* A, const int lda,
                   double* AFAC, const int ldafac, double* rwork, double* resid);
extern void dpot02(const char* uplo, const int n, const int nrhs,
                   const double* A, const int lda, const double* X, const int ldx,
                   double* B, const int ldb, double* rwork, double* resid);
extern void dpot05(const char* uplo, const int n, const int nrhs,
                   const double* A, const int lda, const double* B, const int ldb,
                   const double* X, const int ldx, const double* XACT, const int ldxact,
                   const double* ferr, const double* berr, double* reslts);
extern void dget04(const int n, const int nrhs, const double* X, const int ldx,
                   const double* XACT, const int ldxact, const double rcond,
                   double* resid);
extern double dget06(const double rcond, const double rcondc);

/* Matrix generation */
extern void dlatb4(const char* path, const int imat, const int m, const int n,
                   char* type, int* kl, int* ku, double* anorm, int* mode,
                   double* cndnum, char* dist);
extern void dlatms(const int m, const int n, const char* dist,
                   uint64_t seed, const char* sym, double* d,
                   const int mode, const double cond, const double dmax,
                   const int kl, const int ku, const char* pack,
                   double* A, const int lda, double* work, int* info);
extern void dlarhs(const char* path, const char* xtype, const char* uplo,
                   const char* trans, const int m, const int n,
                   const int kl, const int ku, const int nrhs,
                   const double* A, const int lda, double* XACT, const int ldxact,
                   double* B, const int ldb, uint64_t seed, int* info);

/* Utilities */
extern void dlacpy(const char* uplo, const int m, const int n,
                   const double* A, const int lda, double* B, const int ldb);
extern void dlaset(const char* uplo, const int m, const int n,
                   const double alpha, const double beta,
                   double* A, const int lda);
extern double dlansy(const char* norm, const char* uplo, const int n,
                     const double* A, const int lda, double* work);
extern double dlamch(const char* cmach);

typedef struct {
    int n;
    int imat;
    int iuplo;      /* 0='U', 1='L' */
    int ifact;      /* 0='F', 1='N', 2='E' */
    int iequed;     /* 0='N', 1='Y' */
    char name[64];
} ddrvpo_params_t;

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
} ddrvpo_workspace_t;

static ddrvpo_workspace_t* g_workspace = NULL;

static int group_setup(void** state)
{
    (void)state;
    g_workspace = malloc(sizeof(ddrvpo_workspace_t));
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
    g_workspace->S = calloc(nmax, sizeof(double));
    g_workspace->WORK = calloc(lwork, sizeof(double));
    g_workspace->RWORK = calloc(nmax + 2 * NRHS, sizeof(double));
    g_workspace->IWORK = calloc(nmax, sizeof(int));

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

static void run_ddrvpo_single(int n, int imat, int iuplo, int ifact, int iequed)
{
    static const char* UPLOS[] = {"U", "L"};
    static const char* FACTS[] = {"F", "N", "E"};
    static const char* EQUEDS[] = {"N", "Y"};

    ddrvpo_workspace_t* ws = g_workspace;
    const char* uplo = UPLOS[iuplo];
    const char* fact = FACTS[ifact];
    char equed = EQUEDS[iequed][0];

    int prefac = (fact[0] == 'F');
    int nofact = (fact[0] == 'N');
    int equil = (fact[0] == 'E');

    int lda = (n > 1) ? n : 1;
    double result[NTESTS];
    for (int k = 0; k < NTESTS; k++) result[k] = 0.0;

    int zerot = (imat >= 3 && imat <= 5);
    int izero = 0;

    /* Set up parameters with DLATB4 */
    char type, dist;
    int kl, ku, mode;
    double anorm, cndnum;
    dlatb4("DPO", imat, n, n, &type, &kl, &ku, &anorm, &mode, &cndnum, &dist);

    /* Generate test matrix with DLATMS */
    uint64_t seed = 1988 + n * 1000 + imat * 100 + iuplo * 10;
    int info;
    dlatms(n, n, &dist, seed, &type, ws->RWORK, mode, cndnum,
           anorm, kl, ku, uplo, ws->A, lda, ws->WORK, &info);
    if (info != 0) {
        fail_msg("DLATMS info=%d", info);
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
        int ioff = (izero - 1) * lda;

        if (iuplo == 0) {
            /* UPLO = 'U': zero column IZERO */
            for (int i = 0; i < izero - 1; i++) {
                ws->A[ioff + i] = 0.0;
            }
            ioff = ioff + izero - 1;
            for (int i = izero - 1; i < n; i++) {
                ws->A[ioff] = 0.0;
                ioff = ioff + lda;
            }
        } else {
            /* UPLO = 'L': zero row IZERO */
            ioff = izero - 1;
            for (int i = 0; i < izero - 1; i++) {
                ws->A[ioff] = 0.0;
                ioff = ioff + lda;
            }
            ioff = ioff - (izero - 1);
            for (int i = izero - 1; i < n; i++) {
                ws->A[ioff + i] = 0.0;
            }
        }
    }

    /* Save a copy of the matrix A in ASAV */
    dlacpy(uplo, n, n, ws->A, lda, ws->ASAV, lda);

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
    double rcondc = 0.0;
    double roldc = 0.0;
    double scond = 0.0, amax = 0.0;

    if (zerot) {
        rcondc = 0.0;
    } else if (n == 0) {
        /* For n=0, use RCONDC = 1/CNDNUM from dlatb4 */
        rcondc = 1.0 / cndnum;
    } else {
        dlacpy(uplo, n, n, ws->ASAV, lda, ws->AFAC, lda);

        if (equil || iequed > 0) {
            dpoequ(n, ws->AFAC, lda, ws->S, &scond, &amax, &info);
            if (info == 0 && n > 0) {
                if (iequed > 0) {
                    scond = 0.0;
                }
                dlaqsy(uplo, n, ws->AFAC, lda, ws->S, scond, amax, &equed);
            }
        }

        if (equil) {
            roldc = rcondc;
        }

        double anrm = dlansy("1", uplo, n, ws->AFAC, lda, ws->RWORK);

        dpotrf(uplo, n, ws->AFAC, lda, &info);

        dlacpy(uplo, n, n, ws->AFAC, lda, ws->A, lda);
        dpotri(uplo, n, ws->A, lda, &info);

        double ainvnm = dlansy("1", uplo, n, ws->A, lda, ws->RWORK);
        if (anrm <= 0.0 || ainvnm <= 0.0) {
            rcondc = 1.0;
        } else {
            rcondc = (1.0 / anrm) / ainvnm;
        }
    }
    /* For FACT='F' and FACT='N', ROLDC equals RCONDC.
     * For FACT='E', ROLDC was saved before equilibration (line 270). */
    if (!equil) {
        roldc = rcondc;
    }

    /* Restore the matrix A */
    dlacpy(uplo, n, n, ws->ASAV, lda, ws->A, lda);

    /* Form exact solution and set right hand side */
    seed = 1988 + n * 1000 + imat * 100 + iuplo * 10;
    char xtype = 'N';
    dlarhs("DPO", &xtype, uplo, " ", n, n, kl, ku,
           NRHS, ws->A, lda, ws->XACT, lda, ws->B, lda, seed, &info);
    xtype = 'C';
    dlacpy("Full", n, NRHS, ws->B, lda, ws->BSAV, lda);

    /* --- Test DPOSV --- */
    if (nofact) {
        dlacpy(uplo, n, n, ws->A, lda, ws->AFAC, lda);
        dlacpy("Full", n, NRHS, ws->B, lda, ws->X, lda);

        dposv(uplo, n, NRHS, ws->AFAC, lda, ws->X, lda, &info);

        if (info != izero) {
            fail_msg("DPOSV info=%d expected=%d", info, izero);
            return;
        } else if (info != 0) {
            return;
        }

        /* TEST 1: Reconstruct matrix from factors */
        dpot01(uplo, n, ws->A, lda, ws->AFAC, lda, ws->RWORK, &result[0]);

        /* TEST 2: Compute residual of computed solution */
        dlacpy("Full", n, NRHS, ws->B, lda, ws->WORK, lda);
        dpot02(uplo, n, NRHS, ws->A, lda, ws->X, lda,
               ws->WORK, lda, ws->RWORK, &result[1]);

        /* TEST 3: Check solution from generated exact solution */
        dget04(n, NRHS, ws->X, lda, ws->XACT, lda, rcondc, &result[2]);
        int nt = 3;

        for (int k = 0; k < nt; k++) {
            if (result[k] >= THRESH) {
                fail_msg("DPOSV UPLO=%s test %d failed: result=%e >= thresh=%e",
                         uplo, k + 1, result[k], THRESH);
            }
        }
    }

    /* --- Test DPOSVX --- */
    if (!prefac) {
        dlaset(uplo, n, n, 0.0, 0.0, ws->AFAC, lda);
    }
    dlaset("Full", n, NRHS, 0.0, 0.0, ws->X, lda);

    if (iequed > 0 && n > 0) {
        dlaqsy(uplo, n, ws->A, lda, ws->S, scond, amax, &equed);
    }

    char equed_inout = equed;
    double rcond;
    dposvx(fact, uplo, n, NRHS, ws->A, lda, ws->AFAC, lda,
           &equed_inout, ws->S, ws->B, lda, ws->X, lda, &rcond,
           ws->RWORK, &ws->RWORK[NRHS], ws->WORK, ws->IWORK, &info);

    if (info != izero) {
        fail_msg("DPOSVX info=%d expected=%d", info, izero);
        return;
    }

    int k1;
    if (info == 0) {
        if (!prefac) {
            /* TEST 1: Reconstruct matrix from factors */
            dpot01(uplo, n, ws->A, lda, ws->AFAC, lda,
                   &ws->RWORK[2 * NRHS], &result[0]);
            k1 = 1;
        } else {
            k1 = 2;
        }

        /* TEST 2: Compute residual of computed solution */
        dlacpy("Full", n, NRHS, ws->BSAV, lda, ws->WORK, lda);
        dpot02(uplo, n, NRHS, ws->ASAV, lda, ws->X, lda,
               ws->WORK, lda, &ws->RWORK[2 * NRHS], &result[1]);

        /* TEST 3: Check solution from generated exact solution */
        if (nofact || (prefac && (equed_inout == 'N' || equed_inout == 'n'))) {
            dget04(n, NRHS, ws->X, lda, ws->XACT, lda, rcondc, &result[2]);
        } else {
            dget04(n, NRHS, ws->X, lda, ws->XACT, lda, roldc, &result[2]);
        }

        /* TEST 4-5: Check error bounds from iterative refinement */
        dpot05(uplo, n, NRHS, ws->ASAV, lda, ws->B, lda,
               ws->X, lda, ws->XACT, lda,
               ws->RWORK, &ws->RWORK[NRHS], &result[3]);
    } else {
        k1 = 6;
    }

    /* TEST 6: Compare RCOND from DPOSVX with computed value */
    result[5] = dget06(rcond, rcondc);

    /* Check results */
    if (info == 0) {
        for (int k = k1 - 1; k < NTESTS; k++) {
            if (result[k] >= THRESH) {
                fail_msg("DPOSVX FACT=%s UPLO=%s EQUED=%c test %d: result=%e >= thresh=%e",
                         fact, uplo, equed_inout, k + 1, result[k], THRESH);
            }
        }
    } else {
        /* TRFCON case: only check tests 1, 6 */
        if (!prefac && result[0] >= THRESH) {
            fail_msg("DPOSVX FACT=%s UPLO=%s EQUED=%c test 1: result=%e >= thresh=%e",
                     fact, uplo, equed_inout, result[0], THRESH);
        }
        if (result[5] >= THRESH) {
            fail_msg("DPOSVX FACT=%s UPLO=%s EQUED=%c test 6: result=%e >= thresh=%e",
                     fact, uplo, equed_inout, result[5], THRESH);
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
static int g_num_tests = 0;

static void build_test_array(void)
{
    static const char* UPLOS[] = {"U", "L"};
    static const char* FACTS[] = {"F", "N", "E"};
    static const char* EQUEDS[] = {"N", "Y"};

    g_num_tests = 0;

    for (int in = 0; in < (int)NN; in++) {
        int n = NVAL[in];
        int nimat = (n <= 0) ? 1 : NTYPES;

        for (int imat = 1; imat <= nimat; imat++) {
            int zerot = (imat >= 3 && imat <= 5);
            if (zerot && n < imat - 2) continue;

            for (int iuplo = 0; iuplo < 2; iuplo++) {
                for (int iequed = 0; iequed < 2; iequed++) {
                    int nfact = (iequed == 0) ? 3 : 1;

                    for (int ifact = 0; ifact < nfact; ifact++) {
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
