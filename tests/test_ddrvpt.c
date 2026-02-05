/**
 * @file test_ddrvpt.c
 * @brief DDRVPT tests the driver routines DPTSV and DPTSVX.
 *
 * Port of LAPACK TESTING/LIN/ddrvpt.f to C with CMocka parameterization.
 */

#include "test_harness.h"
#include <string.h>
#include <stdio.h>
#include <cblas.h>
#include <math.h>

/* Test parameters - matching LAPACK dchkaa.f defaults */
static const int NVAL[] = {0, 1, 2, 3, 5, 10, 50};
#define NN      (sizeof(NVAL) / sizeof(NVAL[0]))
#define NTYPES  12
#define NTESTS  6
#define THRESH  30.0
#define NMAX    50
#define NRHS    2

/* Routines under test */
extern void dptsv(const int n, const int nrhs, double* D, double* E,
                  double* B, const int ldb, int* info);
extern void dptsvx(const char* fact, const int n, const int nrhs,
                   const double* D, const double* E,
                   double* DF, double* EF,
                   const double* B, const int ldb,
                   double* X, const int ldx, double* rcond,
                   double* ferr, double* berr, double* work, int* info);

/* Supporting routines */
extern void dpttrf(const int n, double* D, double* E, int* info);
extern void dpttrs(const int n, const int nrhs,
                   const double* D, const double* E,
                   double* B, const int ldb, int* info);
extern void dlaptm(const int n, const int nrhs,
                   const double alpha, const double* D, const double* E,
                   const double* X, const int ldx,
                   const double beta, double* B, const int ldb);

/* Verification routines */
extern void dptt01(const int n, const double* D, const double* E,
                   const double* DF, const double* EF,
                   double* work, double* resid);
extern void dptt02(const int n, const int nrhs,
                   const double* D, const double* E,
                   const double* X, const int ldx,
                   double* B, const int ldb, double* resid);
extern void dptt05(const int n, const int nrhs,
                   const double* D, const double* E,
                   const double* B, const int ldb,
                   const double* X, const int ldx,
                   const double* XACT, const int ldxact,
                   const double* ferr, const double* berr, double* reslts);
extern void dget04(const int n, const int nrhs, const double* X, const int ldx,
                   const double* XACT, const int ldxact, const double rcond, double* resid);
extern double dget06(const double rcond, const double rcondc);
extern double dlanst(const char* norm, const int n, const double* D, const double* E);

/* Matrix generation */
extern void dlatb4(const char* path, const int imat, const int m, const int n,
                   char* type, int* kl, int* ku, double* anorm, int* mode,
                   double* cndnum, char* dist);
extern void dlatms(const int m, const int n, const char* dist,
                   uint64_t seed, const char* sym, double* d,
                   const int mode, const double cond, const double dmax,
                   const int kl, const int ku, const char* pack,
                   double* A, const int lda, double* work, int* info);

/* Utilities */
extern void dlacpy(const char* uplo, const int m, const int n,
                   const double* A, const int lda, double* B, const int ldb);
extern void dlaset(const char* uplo, const int m, const int n,
                   const double alpha, const double beta, double* A, const int lda);
extern double dlamch(const char* cmach);

#include "testutils/test_rng.h"

typedef struct {
    int n;
    int imat;
    int ifact;      /* 0='F', 1='N' */
    char name[64];
} ddrvpt_params_t;

typedef struct {
    double* D;      /* Diagonal (N) */
    double* E;      /* Off-diagonal (N-1) */
    double* DF;     /* Factored diagonal (N) */
    double* EF;     /* Factored off-diagonal (N-1) */
    double* B;
    double* X;
    double* XACT;
    double* WORK;
    double* RWORK;
    double* A;      /* Band storage for dlatms (2 x N) */
} ddrvpt_workspace_t;

static ddrvpt_workspace_t* g_workspace = NULL;

static int group_setup(void** state)
{
    (void)state;
    g_workspace = malloc(sizeof(ddrvpt_workspace_t));
    if (!g_workspace) return -1;

    int nmax = NMAX;
    /* WORK needs to be large enough for dlatms which uses n*n for full matrix
     * plus additional workspace for dlagsy (roughly 2*n more) */
    int lwork = nmax * nmax + 4 * nmax;
    if (lwork < nmax * NRHS) lwork = nmax * NRHS;

    g_workspace->D = calloc(2 * nmax, sizeof(double));
    g_workspace->E = calloc(2 * nmax, sizeof(double));
    g_workspace->DF = g_workspace->D + nmax;  /* Use second half */
    g_workspace->EF = g_workspace->E + nmax;  /* Use second half */
    g_workspace->B = calloc(nmax * NRHS, sizeof(double));
    g_workspace->X = calloc(nmax * NRHS, sizeof(double));
    g_workspace->XACT = calloc(nmax * NRHS, sizeof(double));
    g_workspace->WORK = calloc(lwork, sizeof(double));
    g_workspace->RWORK = calloc(nmax > 2 * NRHS ? nmax : 2 * NRHS, sizeof(double));
    g_workspace->A = calloc(2 * nmax, sizeof(double));

    if (!g_workspace->D || !g_workspace->E || !g_workspace->B ||
        !g_workspace->X || !g_workspace->XACT || !g_workspace->WORK ||
        !g_workspace->RWORK || !g_workspace->A) {
        return -1;
    }
    return 0;
}

static int group_teardown(void** state)
{
    (void)state;
    if (g_workspace) {
        free(g_workspace->D);
        free(g_workspace->E);
        free(g_workspace->B);
        free(g_workspace->X);
        free(g_workspace->XACT);
        free(g_workspace->WORK);
        free(g_workspace->RWORK);
        free(g_workspace->A);
        free(g_workspace);
        g_workspace = NULL;
    }
    return 0;
}

static void run_ddrvpt_single(int n, int imat, int ifact)
{
    static const char* FACTS[] = {"F", "N"};

    ddrvpt_workspace_t* ws = g_workspace;
    const char* fact = FACTS[ifact];

    int lda = (n > 1) ? n : 1;
    double result[NTESTS];
    for (int k = 0; k < NTESTS; k++) result[k] = 0.0;

    /* Clear workspace to prevent stale data from previous tests */
    memset(ws->D, 0, 2 * NMAX * sizeof(double));
    memset(ws->E, 0, 2 * NMAX * sizeof(double));
    memset(ws->A, 0, 2 * NMAX * sizeof(double));

    double* D = ws->D;
    double* E = ws->E;
    double* DF = ws->DF;
    double* EF = ws->EF;

    int zerot = (imat >= 8 && imat <= 10);
    int izero = 0;

    /* Set up parameters with DLATB4 */
    char type, dist;
    int kl, ku, mode;
    double anorm, cndnum;
    dlatb4("DPT", imat, n, n, &type, &kl, &ku, &anorm, &mode, &cndnum, &dist);

    /* Generate test matrix.
     * For types 1-6, each type has a unique seed.
     * For types 7-12, they share the same base seed because types 8-10
     * are singular versions of the type 7 matrix (with zeroed rows/columns),
     * and types 11-12 are scaled versions of type 7. */
    uint64_t seed;
    if (imat <= 6) {
        seed = 1988 + n * 1000 + imat * 100;
    } else {
        seed = 1988 + n * 1000 + 7 * 100;  /* Types 7-12 share type 7's seed */
    }
    int info;

    (void)type; (void)dist; (void)mode; (void)kl; (void)ku;

    if (imat <= 6) {
        /* Types 1-6: generate a symmetric tridiagonal matrix of known condition
         * number in lower triangular band storage. */
        izero = 0;

        char symtype[2] = {type, '\0'};
        char disttype[2] = {dist, '\0'};

        dlatms(n, n, disttype, seed, symtype, ws->RWORK, mode, cndnum,
               anorm, kl, ku, "B", ws->A, 2, ws->WORK, &info);

        if (info != 0) {
            fail_msg("DLATMS info=%d for imat=%d", info, imat);
            return;
        }

        /* Copy the matrix from band storage to D and E.
         * Band storage format for pack='B' with lda=2:
         * A[0] = D[0], A[1] = E[0]
         * A[2] = D[1], A[3] = E[1]
         * ... */
        int ia = 0;
        for (int i = 0; i < n - 1; i++) {
            D[i] = ws->A[ia];
            E[i] = ws->A[ia + 1];
            ia += 2;
        }
        if (n > 0) {
            D[n - 1] = ws->A[ia];
        }
    } else {
        /* Types 7-12: generate a diagonally dominant matrix with
         * unknown condition number in the vectors D and E. */
        rng_seed(seed);

        if (!zerot) {
            /* Let D and E have values from [-1,1] */
            for (int i = 0; i < n; i++) {
                D[i] = 2.0 * rng_uniform() - 1.0;
            }
            for (int i = 0; i < n - 1; i++) {
                E[i] = 2.0 * rng_uniform() - 1.0;
            }

            /* Make the tridiagonal matrix diagonally dominant */
            if (n == 1) {
                D[0] = fabs(D[0]);
            } else {
                D[0] = fabs(D[0]) + fabs(E[0]);
                D[n - 1] = fabs(D[n - 1]) + fabs(E[n - 2]);
                for (int i = 1; i < n - 1; i++) {
                    D[i] = fabs(D[i]) + fabs(E[i]) + fabs(E[i - 1]);
                }
            }

            /* Scale D and E so the maximum element is ANORM */
            int ix = cblas_idamax(n, D, 1);
            double dmax = D[ix];
            cblas_dscal(n, anorm / dmax, D, 1);
            if (n > 1) {
                cblas_dscal(n - 1, anorm / dmax, E, 1);
            }
        }

        /* Zero out elements for singular matrix types 8-10 */
        if (imat == 8) {
            izero = 1;
            D[0] = 0.0;
            if (n > 1) E[0] = 0.0;
        } else if (imat == 9) {
            izero = n;
            if (n > 1) E[n - 2] = 0.0;
            D[n - 1] = 0.0;
        } else if (imat == 10) {
            izero = (n + 1) / 2;
            if (izero > 1) {
                E[izero - 2] = 0.0;
            }
            if (izero < n) {
                E[izero - 1] = 0.0;
            }
            D[izero - 1] = 0.0;
        } else {
            izero = 0;
        }
    }

    /* Skip FACT='F' for singular matrices */
    if (zerot && ifact == 0) {
        return;
    }

    /* Generate NRHS random solution vectors */
    rng_seed(seed + 42);
    for (int j = 0; j < NRHS; j++) {
        for (int i = 0; i < n; i++) {
            ws->XACT[j * lda + i] = 2.0 * rng_uniform() - 1.0;
        }
    }

    /* Set the right hand side using DLAPTM: B = A * XACT */
    dlaptm(n, NRHS, 1.0, D, E, ws->XACT, lda, 0.0, ws->B, lda);

    /*
     * Compute condition number RCONDC for FACT='F'.
     * For FACT='N', reuse the value computed for FACT='F'.
     * Since tests are independent, we always compute it.
     */
    double rcondc = 0.0;

    if (zerot) {
        rcondc = 0.0;
    } else if (n == 0) {
        rcondc = 1.0 / cndnum;
    } else {
        /* Compute the 1-norm of A */
        double anorm_1 = dlanst("1", n, D, E);

        /* Copy D and E to DF and EF, then factor */
        cblas_dcopy(n, D, 1, DF, 1);
        if (n > 1) cblas_dcopy(n - 1, E, 1, EF, 1);

        dpttrf(n, DF, EF, &info);

        /* Use DPTTRS to solve for one column at a time of inv(A),
         * computing the maximum column sum as we go */
        double ainvnm = 0.0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) ws->X[j] = 0.0;
            ws->X[i] = 1.0;
            dpttrs(n, 1, DF, EF, ws->X, lda, &info);
            double colsum = cblas_dasum(n, ws->X, 1);
            if (colsum > ainvnm) ainvnm = colsum;
        }

        /* Compute the 1-norm condition number of A */
        if (anorm_1 <= 0.0 || ainvnm <= 0.0) {
            rcondc = 1.0;
        } else {
            rcondc = (1.0 / anorm_1) / ainvnm;
        }
    }

    /* --- Test DPTSV --- */
    if (ifact == 1) {
        cblas_dcopy(n, D, 1, DF, 1);
        if (n > 1) cblas_dcopy(n - 1, E, 1, EF, 1);
        dlacpy("Full", n, NRHS, ws->B, lda, ws->X, lda);

        dptsv(n, NRHS, DF, EF, ws->X, lda, &info);

        if (zerot) {
            if (info <= 0) {
                fail_msg("DPTSV: expected INFO > 0 for singular matrix, got %d", info);
                return;
            }
        } else if (info != izero) {
            fail_msg("DPTSV info=%d expected=%d", info, izero);
            return;
        }

        int nt = 0;
        if (izero == 0 && info == 0) {
            /* TEST 1: Check factorization norm(L*D*L' - A) / (n * norm(A) * eps) */
            dptt01(n, D, E, DF, EF, ws->WORK, &result[0]);

            /* TEST 2: Check residual of computed solution */
            dlacpy("Full", n, NRHS, ws->B, lda, ws->WORK, lda);
            dptt02(n, NRHS, D, E, ws->X, lda, ws->WORK, lda, &result[1]);

            /* TEST 3: Check solution from generated exact solution */
            dget04(n, NRHS, ws->X, lda, ws->XACT, lda, rcondc, &result[2]);
            nt = 3;
        }

        for (int k = 0; k < nt; k++) {
            if (result[k] >= THRESH) {
                fail_msg("DPTSV test %d failed: result=%e >= thresh=%e",
                         k + 1, result[k], THRESH);
            }
        }
    }

    /* --- Test DPTSVX --- */
    if (ifact == 0) {
        /* FACT='F': Provide the factorization in DF and EF.
         * Copy D/E to DF/EF and factor with DPTTRF. */
        cblas_dcopy(n, D, 1, DF, 1);
        if (n > 1) cblas_dcopy(n - 1, E, 1, EF, 1);
        dpttrf(n, DF, EF, &info);
        /* Ignore info here - we handle singular matrices via izero */
    } else {
        /* FACT='N': Initialize DF and EF to zero */
        for (int i = 0; i < n; i++) DF[i] = 0.0;
        for (int i = 0; i < n - 1; i++) EF[i] = 0.0;
    }
    dlaset("Full", n, NRHS, 0.0, 0.0, ws->X, lda);

    double rcond;
    dptsvx(fact, n, NRHS, D, E, DF, EF, ws->B, lda, ws->X, lda, &rcond,
           ws->RWORK, &ws->RWORK[NRHS], ws->WORK, &info);

    if (zerot) {
        if (info <= 0) {
            fail_msg("DPTSVX: expected INFO > 0 for singular matrix, got %d", info);
            return;
        }
    } else if (info != izero) {
        fail_msg("DPTSVX info=%d expected=%d", info, izero);
        return;
    }

    int k1;
    if (izero == 0) {
        if (ifact >= 1) {
            /* TEST 1: Check factorization */
            dptt01(n, D, E, DF, EF, ws->WORK, &result[0]);
            k1 = 1;
        } else {
            k1 = 2;
        }

        /* TEST 2: Check residual of computed solution */
        dlacpy("Full", n, NRHS, ws->B, lda, ws->WORK, lda);
        dptt02(n, NRHS, D, E, ws->X, lda, ws->WORK, lda, &result[1]);

        /* TEST 3: Check solution from generated exact solution */
        dget04(n, NRHS, ws->X, lda, ws->XACT, lda, rcondc, &result[2]);

        /* TEST 4-5: Check error bounds from iterative refinement */
        dptt05(n, NRHS, D, E, ws->B, lda, ws->X, lda, ws->XACT, lda,
               ws->RWORK, &ws->RWORK[NRHS], &result[3]);
    } else {
        k1 = 6;
    }

    /* TEST 6: Compare RCOND from DPTSVX with computed value */
    result[5] = dget06(rcond, rcondc);

    /* Check results */
    for (int k = k1 - 1; k < 5; k++) {
        if (result[k] >= THRESH) {
            fail_msg("DPTSVX FACT=%s test %d: result=%e >= thresh=%e",
                     fact, k + 1, result[k], THRESH);
        }
    }
    if (result[5] >= THRESH) {
        fail_msg("DPTSVX FACT=%s test 6: result=%e >= thresh=%e",
                 fact, result[5], THRESH);
    }
}

static void test_ddrvpt_case(void** state)
{
    ddrvpt_params_t* p = *state;
    run_ddrvpt_single(p->n, p->imat, p->ifact);
}

#define MAX_TESTS 1000

static ddrvpt_params_t g_params[MAX_TESTS];
static struct CMUnitTest g_tests[MAX_TESTS];
static int g_num_tests = 0;

static void build_test_array(void)
{
    static const char* FACTS[] = {"F", "N"};

    g_num_tests = 0;

    for (int in = 0; in < (int)NN; in++) {
        int n = NVAL[in];
        int nimat = (n <= 0) ? 1 : NTYPES;

        for (int imat = 1; imat <= nimat; imat++) {
            int zerot = (imat >= 8 && imat <= 10);

            for (int ifact = 0; ifact < 2; ifact++) {
                if (zerot && ifact == 0) continue;

                ddrvpt_params_t* p = &g_params[g_num_tests];
                p->n = n;
                p->imat = imat;
                p->ifact = ifact;
                snprintf(p->name, sizeof(p->name),
                         "n%d_t%d_%s",
                         n, imat, FACTS[ifact]);

                g_tests[g_num_tests].name = p->name;
                g_tests[g_num_tests].test_func = test_ddrvpt_case;
                g_tests[g_num_tests].setup_func = NULL;
                g_tests[g_num_tests].teardown_func = NULL;
                g_tests[g_num_tests].initial_state = p;

                g_num_tests++;
            }
        }
    }
}

int main(void)
{
    build_test_array();
    return _cmocka_run_group_tests("ddrvpt", g_tests, g_num_tests,
                                   group_setup, group_teardown);
}
