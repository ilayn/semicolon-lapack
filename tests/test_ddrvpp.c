/**
 * @file test_ddrvpp.c
 * @brief DDRVPP tests the driver routines DPPSV and DPPSVX.
 *
 * Port of LAPACK TESTING/LIN/ddrvpp.f to C with CMocka parameterization.
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
#define NTYPES  9
#define NTESTS  6
#define THRESH  30.0
#define NMAX    50
#define NRHS    2
#define NPP_MAX (NMAX * (NMAX + 1) / 2)

/* Routines under test */
extern void dppsv(const char* uplo, const int n, const int nrhs,
                  f64* AP, f64* B, const int ldb, int* info);
extern void dppsvx(const char* fact, const char* uplo, const int n, const int nrhs,
                   f64* AP, f64* AFP, char* equed, f64* S,
                   f64* B, const int ldb, f64* X, const int ldx,
                   f64* rcond, f64* ferr, f64* berr,
                   f64* work, int* iwork, int* info);

/* Supporting routines */
extern void dpptrf(const char* uplo, const int n, f64* AP, int* info);
extern void dpptri(const char* uplo, const int n, f64* AP, int* info);
extern void dppequ(const char* uplo, const int n, const f64* AP,
                   f64* S, f64* scond, f64* amax, int* info);
extern void dlaqsp(const char* uplo, const int n, f64* AP,
                   const f64* S, const f64 scond, const f64 amax, char* equed);

/* Verification routines */
extern void dppt01(const char* uplo, const int n, const f64* A,
                   f64* AFAC, f64* rwork, f64* resid);
extern void dppt02(const char* uplo, const int n, const int nrhs,
                   const f64* A, const f64* X, const int ldx,
                   f64* B, const int ldb, f64* rwork, f64* resid);
extern void dppt05(const char* uplo, const int n, const int nrhs,
                   const f64* AP, const f64* B, const int ldb,
                   const f64* X, const int ldx, const f64* XACT, const int ldxact,
                   const f64* FERR, const f64* BERR, f64* reslts);
extern void dget04(const int n, const int nrhs, const f64* X, const int ldx,
                   const f64* XACT, const int ldxact, const f64 rcond,
                   f64* resid);
extern f64 dget06(const f64 rcond, const f64 rcondc);

/* Matrix generation */
extern void dlatb4(const char* path, const int imat, const int m, const int n,
                   char* type, int* kl, int* ku, f64* anorm, int* mode,
                   f64* cndnum, char* dist);
extern void dlatms(const int m, const int n, const char* dist,
                   const char* sym, f64* d,
                   const int mode, const f64 cond, const f64 dmax,
                   const int kl, const int ku, const char* pack,
                   f64* A, const int lda, f64* work, int* info,
                   uint64_t state[static 4]);
extern void dlarhs(const char* path, const char* xtype, const char* uplo,
                   const char* trans, const int m, const int n,
                   const int kl, const int ku, const int nrhs,
                   const f64* A, const int lda, f64* XACT, const int ldxact,
                   f64* B, const int ldb, int* info, uint64_t state[static 4]);

/* Utilities */
extern void dlacpy(const char* uplo, const int m, const int n,
                   const f64* A, const int lda, f64* B, const int ldb);
extern void dlaset(const char* uplo, const int m, const int n,
                   const f64 alpha, const f64 beta,
                   f64* A, const int lda);
extern f64 dlansp(const char* norm, const char* uplo, const int n,
                     const f64* AP, f64* work);
extern f64 dlamch(const char* cmach);

typedef struct {
    int n;
    int imat;
    int iuplo;      /* 0='U', 1='L' */
    int ifact;      /* 0='F', 1='N', 2='E' */
    int iequed;     /* 0='N', 1='Y' */
    char name[64];
} ddrvpp_params_t;

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
    int lwork;
} ddrvpp_workspace_t;

static ddrvpp_workspace_t* g_workspace = NULL;

static int group_setup(void** state)
{
    (void)state;
    g_workspace = malloc(sizeof(ddrvpp_workspace_t));
    if (!g_workspace) return -1;

    int nmax = NMAX;
    int lwork = nmax * (nmax > 3 ? nmax : 3);
    if (lwork < nmax * NRHS) lwork = nmax * NRHS;

    g_workspace->lwork = lwork;
    g_workspace->A = calloc(nmax * nmax, sizeof(f64));
    g_workspace->AFAC = calloc(NPP_MAX, sizeof(f64));
    g_workspace->ASAV = calloc(NPP_MAX, sizeof(f64));
    g_workspace->B = calloc(nmax * NRHS, sizeof(f64));
    g_workspace->BSAV = calloc(nmax * NRHS, sizeof(f64));
    g_workspace->X = calloc(nmax * NRHS, sizeof(f64));
    g_workspace->XACT = calloc(nmax * NRHS, sizeof(f64));
    g_workspace->S = calloc(nmax, sizeof(f64));
    g_workspace->WORK = calloc(lwork, sizeof(f64));
    g_workspace->RWORK = calloc(nmax + 2 * NRHS, sizeof(f64));
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

static void run_ddrvpp_single(int n, int imat, int iuplo, int ifact, int iequed)
{
    static const char* UPLOS[] = {"U", "L"};
    static const char* FACTS[] = {"F", "N", "E"};
    static const char* EQUEDS[] = {"N", "Y"};

    ddrvpp_workspace_t* ws = g_workspace;
    const char* uplo = UPLOS[iuplo];
    const char* fact = FACTS[ifact];
    char equed = EQUEDS[iequed][0];

    int prefac = (fact[0] == 'F');
    int nofact = (fact[0] == 'N');
    int equil = (fact[0] == 'E');

    int lda = (n > 1) ? n : 1;
    int npp = n * (n + 1) / 2;
    f64 result[NTESTS];
    for (int k = 0; k < NTESTS; k++) result[k] = 0.0;

    int zerot = (imat >= 3 && imat <= 5);
    int izero = 0;

    /* PACK parameter: 'C' for upper, 'R' for lower */
    char packit = (iuplo == 0) ? 'C' : 'R';

    /* Set up parameters with DLATB4 */
    char type, dist;
    int kl, ku, mode;
    f64 anorm, cndnum;
    dlatb4("DPP", imat, n, n, &type, &kl, &ku, &anorm, &mode, &cndnum, &dist);

    f64 rcondc = 1.0 / cndnum;

    /* Generate test matrix with DLATMS */
    uint64_t rng_state[4];
    rng_seed(rng_state, 1988 + n * 1000 + imat * 100 + iuplo * 10);
    int info;
    dlatms(n, n, &dist, &type, ws->RWORK, mode, cndnum,
           anorm, kl, ku, &packit, ws->A, lda, ws->WORK, &info, rng_state);
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

        if (iuplo == 0) {
            /* Upper packed: zero column IZERO (rows 1..IZERO-1)
             * then row IZERO (columns IZERO..N) */
            int ioff = (izero - 1) * izero / 2;
            for (int i = 1; i <= izero - 1; i++)
                ws->A[ioff + i - 1] = 0.0;
            ioff += izero;
            for (int i = izero; i <= n; i++) {
                ws->A[ioff - 1] = 0.0;
                ioff += i;
            }
        } else {
            /* Lower packed: zero row IZERO (columns 1..IZERO-1)
             * then column IZERO (rows IZERO..N) */
            int ioff = izero;
            for (int i = 1; i <= izero - 1; i++) {
                ws->A[ioff - 1] = 0.0;
                ioff += n - i;
            }
            ioff -= izero;
            for (int i = izero; i <= n; i++)
                ws->A[ioff + i - 1] = 0.0;
        }
    } else {
        izero = 0;
    }

    /* Save a copy of the matrix A in ASAV */
    cblas_dcopy(npp, ws->A, 1, ws->ASAV, 1);

    /* Skip FACT='F' for singular matrices */
    if (zerot && prefac) {
        return;
    }

    f64 roldc = 0.0;
    f64 scond = 0.0, amax = 0.0;

    if (zerot) {
        rcondc = 0.0;
    } else if (n == 0) {
        rcondc = 1.0 / cndnum;
    } else {
        cblas_dcopy(npp, ws->ASAV, 1, ws->AFAC, 1);

        if (equil || iequed > 0) {
            dppequ(uplo, n, ws->AFAC, ws->S, &scond, &amax, &info);
            if (info == 0 && n > 0) {
                if (iequed > 0) {
                    scond = 0.0;
                }
                dlaqsp(uplo, n, ws->AFAC, ws->S, scond, amax, &equed);
            }
        }

        if (equil) {
            roldc = rcondc;
        }

        f64 anrm = dlansp("1", uplo, n, ws->AFAC, ws->RWORK);

        dpptrf(uplo, n, ws->AFAC, &info);

        cblas_dcopy(npp, ws->AFAC, 1, ws->A, 1);
        dpptri(uplo, n, ws->A, &info);

        f64 ainvnm = dlansp("1", uplo, n, ws->A, ws->RWORK);
        if (anrm <= 0.0 || ainvnm <= 0.0) {
            rcondc = 1.0;
        } else {
            rcondc = (1.0 / anrm) / ainvnm;
        }
    }
    if (!equil) {
        roldc = rcondc;
    }

    /* Restore the matrix A */
    cblas_dcopy(npp, ws->ASAV, 1, ws->A, 1);

    /* Form exact solution and set right hand side */
    rng_seed(rng_state, 1988 + n * 1000 + imat * 100 + iuplo * 10);
    char xtype = 'N';
    dlarhs("DPP", &xtype, uplo, " ", n, n, kl, ku,
           NRHS, ws->A, lda, ws->XACT, lda, ws->B, lda, &info, rng_state);
    xtype = 'C';
    dlacpy("Full", n, NRHS, ws->B, lda, ws->BSAV, lda);

    /* --- Test DPPSV --- */
    if (nofact) {
        cblas_dcopy(npp, ws->A, 1, ws->AFAC, 1);
        dlacpy("Full", n, NRHS, ws->B, lda, ws->X, lda);

        dppsv(uplo, n, NRHS, ws->AFAC, ws->X, lda, &info);

        if (info != izero) {
            fail_msg("DPPSV info=%d expected=%d", info, izero);
            return;
        } else if (info != 0) {
            return;
        }

        /* TEST 1: Reconstruct matrix from factors */
        dppt01(uplo, n, ws->A, ws->AFAC, ws->RWORK, &result[0]);

        /* TEST 2: Compute residual of computed solution */
        dlacpy("Full", n, NRHS, ws->B, lda, ws->WORK, lda);
        dppt02(uplo, n, NRHS, ws->A, ws->X, lda,
               ws->WORK, lda, ws->RWORK, &result[1]);

        /* TEST 3: Check solution from generated exact solution */
        dget04(n, NRHS, ws->X, lda, ws->XACT, lda, rcondc, &result[2]);
        int nt = 3;

        for (int k = 0; k < nt; k++) {
            if (result[k] >= THRESH) {
                fail_msg("DPPSV UPLO=%s test %d failed: result=%e >= thresh=%e",
                         uplo, k + 1, result[k], THRESH);
            }
        }
    }

    /* --- Test DPPSVX --- */
    if (!prefac && npp > 0) {
        dlaset("Full", npp, 1, 0.0, 0.0, ws->AFAC, npp);
    }
    dlaset("Full", n, NRHS, 0.0, 0.0, ws->X, lda);

    char equed_inout = equed;
    if (iequed > 0 && n > 0) {
        dlaqsp(uplo, n, ws->A, ws->S, scond, amax, &equed_inout);
    }

    /* Restore B */
    dlacpy("Full", n, NRHS, ws->BSAV, lda, ws->B, lda);

    f64 rcond;
    dppsvx(fact, uplo, n, NRHS, ws->A, ws->AFAC, &equed_inout, ws->S,
           ws->B, lda, ws->X, lda, &rcond,
           ws->RWORK, &ws->RWORK[NRHS], ws->WORK, ws->IWORK, &info);

    if (info != izero) {
        fail_msg("DPPSVX info=%d expected=%d", info, izero);
        return;
    }

    int k1;
    if (info == 0) {
        if (!prefac) {
            /* TEST 1: Reconstruct matrix from factors */
            dppt01(uplo, n, ws->A, ws->AFAC,
                   &ws->RWORK[2 * NRHS], &result[0]);
            k1 = 1;
        } else {
            k1 = 2;
        }

        /* TEST 2: Compute residual of computed solution */
        dlacpy("Full", n, NRHS, ws->BSAV, lda, ws->WORK, lda);
        dppt02(uplo, n, NRHS, ws->ASAV, ws->X, lda,
               ws->WORK, lda, &ws->RWORK[2 * NRHS], &result[1]);

        /* TEST 3: Check solution from generated exact solution */
        if (nofact || (prefac && (equed_inout == 'N' || equed_inout == 'n'))) {
            dget04(n, NRHS, ws->X, lda, ws->XACT, lda, rcondc, &result[2]);
        } else {
            dget04(n, NRHS, ws->X, lda, ws->XACT, lda, roldc, &result[2]);
        }

        /* TEST 4-5: Check error bounds from iterative refinement */
        dppt05(uplo, n, NRHS, ws->ASAV, ws->B, lda,
               ws->X, lda, ws->XACT, lda,
               ws->RWORK, &ws->RWORK[NRHS], &result[3]);
    } else {
        k1 = 6;
    }

    /* TEST 6: Compare RCOND from DPPSVX with computed value */
    result[5] = dget06(rcond, rcondc);

    /* Check results */
    if (info == 0) {
        for (int k = k1 - 1; k < NTESTS; k++) {
            if (result[k] >= THRESH) {
                fail_msg("DPPSVX FACT=%s UPLO=%s EQUED=%c test %d: result=%e >= thresh=%e",
                         fact, uplo, equed_inout, k + 1, result[k], THRESH);
            }
        }
    } else {
        if (!prefac && result[0] >= THRESH) {
            fail_msg("DPPSVX FACT=%s UPLO=%s EQUED=%c test 1: result=%e >= thresh=%e",
                     fact, uplo, equed_inout, result[0], THRESH);
        }
        if (result[5] >= THRESH) {
            fail_msg("DPPSVX FACT=%s UPLO=%s EQUED=%c test 6: result=%e >= thresh=%e",
                     fact, uplo, equed_inout, result[5], THRESH);
        }
    }
}

static void test_ddrvpp_case(void** state)
{
    ddrvpp_params_t* p = *state;
    run_ddrvpp_single(p->n, p->imat, p->iuplo, p->ifact, p->iequed);
}

#define MAX_TESTS 3000

static ddrvpp_params_t g_params[MAX_TESTS];
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

                        ddrvpp_params_t* p = &g_params[g_num_tests];
                        p->n = n;
                        p->imat = imat;
                        p->iuplo = iuplo;
                        p->ifact = ifact;
                        p->iequed = iequed;
                        snprintf(p->name, sizeof(p->name),
                                 "n%d_t%d_%s_%s_%s",
                                 n, imat, UPLOS[iuplo], FACTS[ifact], EQUEDS[iequed]);

                        g_tests[g_num_tests].name = p->name;
                        g_tests[g_num_tests].test_func = test_ddrvpp_case;
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
    return _cmocka_run_group_tests("ddrvpp", g_tests, g_num_tests,
                                   group_setup, group_teardown);
}
