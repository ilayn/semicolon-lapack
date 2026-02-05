/**
 * @file test_ddrvgt.c
 * @brief DDRVGT tests the driver routines DGTSV and DGTSVX.
 *
 * Port of LAPACK TESTING/LIN/ddrvgt.f to C with CMocka parameterization.
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
extern void dgtsv(const int n, const int nrhs, double* DL, double* D, double* DU,
                  double* B, const int ldb, int* info);
extern void dgtsvx(const char* fact, const char* trans, const int n, const int nrhs,
                   const double* DL, const double* D, const double* DU,
                   double* DLF, double* DF, double* DUF, double* DU2,
                   int* ipiv, const double* B, const int ldb,
                   double* X, const int ldx, double* rcond,
                   double* ferr, double* berr, double* work, int* iwork, int* info);

/* Supporting routines */
extern void dgttrf(const int n, double* DL, double* D, double* DU, double* DU2,
                   int* ipiv, int* info);
extern void dgttrs(const char* trans, const int n, const int nrhs,
                   const double* DL, const double* D, const double* DU,
                   const double* DU2, const int* ipiv, double* B, const int ldb, int* info);
extern void dlagtm(const char* trans, const int n, const int nrhs,
                   const double alpha, const double* DL, const double* D, const double* DU,
                   const double* X, const int ldx, const double beta, double* B, const int ldb);

/* Verification routines */
extern void dgtt01(const int n, const double* DL, const double* D, const double* DU,
                   const double* DLF, const double* DF, const double* DUF, const double* DU2,
                   const int* ipiv, double* work, const int ldwork, double* resid);
extern void dgtt02(const char* trans, const int n, const int nrhs,
                   const double* DL, const double* D, const double* DU,
                   const double* X, const int ldx, double* B, const int ldb, double* resid);
extern void dgtt05(const char* trans, const int n, const int nrhs,
                   const double* DL, const double* D, const double* DU,
                   const double* B, const int ldb, const double* X, const int ldx,
                   const double* XACT, const int ldxact,
                   const double* ferr, const double* berr, double* reslts);
extern void dget04(const int n, const int nrhs, const double* X, const int ldx,
                   const double* XACT, const int ldxact, const double rcond, double* resid);
extern double dget06(const double rcond, const double rcondc);
extern double dlangt(const char* norm, const int n, const double* DL, const double* D, const double* DU);

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
    int itran;      /* 0='N', 1='T', 2='C' */
    char name[64];
} ddrvgt_params_t;

typedef struct {
    double* A;      /* Tridiagonal storage: DL, D, DU = 3*N */
    double* AF;     /* Factored: DLF, DF, DUF, DU2 = 4*N */
    double* B;
    double* X;
    double* XACT;
    double* WORK;
    double* RWORK;
    int* IWORK;
} ddrvgt_workspace_t;

static ddrvgt_workspace_t* g_workspace = NULL;

static int group_setup(void** state)
{
    (void)state;
    g_workspace = malloc(sizeof(ddrvgt_workspace_t));
    if (!g_workspace) return -1;

    int nmax = NMAX;
    /* WORK needs to be large enough for dlatms which uses n*n for full matrix
     * plus additional workspace for dlagge/dlagsy (roughly 2*n more) */
    int lwork = nmax * nmax + 4 * nmax;
    if (lwork < nmax * NRHS) lwork = nmax * NRHS;

    g_workspace->A = calloc(3 * nmax, sizeof(double));
    g_workspace->AF = calloc(4 * nmax, sizeof(double));
    g_workspace->B = calloc(nmax * NRHS, sizeof(double));
    g_workspace->X = calloc(nmax * NRHS, sizeof(double));
    g_workspace->XACT = calloc(nmax * NRHS, sizeof(double));
    g_workspace->WORK = calloc(lwork, sizeof(double));
    g_workspace->RWORK = calloc(nmax > 2 * NRHS ? nmax : 2 * NRHS, sizeof(double));
    g_workspace->IWORK = calloc(2 * nmax, sizeof(int));

    if (!g_workspace->A || !g_workspace->AF || !g_workspace->B ||
        !g_workspace->X || !g_workspace->XACT || !g_workspace->WORK ||
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
        free(g_workspace->AF);
        free(g_workspace->B);
        free(g_workspace->X);
        free(g_workspace->XACT);
        free(g_workspace->WORK);
        free(g_workspace->RWORK);
        free(g_workspace->IWORK);
        free(g_workspace);
        g_workspace = NULL;
    }
    return 0;
}

static void run_ddrvgt_single(int n, int imat, int ifact, int itran)
{
    static const char* FACTS[] = {"F", "N"};
    static const char* TRANSS[] = {"N", "T", "C"};

    ddrvgt_workspace_t* ws = g_workspace;
    const char* fact = FACTS[ifact];
    const char* trans = TRANSS[itran];

    int m = (n > 1) ? n - 1 : 0;
    int lda = (n > 1) ? n : 1;
    double result[NTESTS];
    for (int k = 0; k < NTESTS; k++) result[k] = 0.0;

    /* Pointers to tridiagonal components in A:
     * A[0..m-1] = DL (subdiagonal)
     * A[m..m+n-1] = D (diagonal)
     * A[m+n..m+2n-2] = DU (superdiagonal) */
    double* DL = ws->A;
    double* D = ws->A + m;
    double* DU = ws->A + m + n;

    /* Pointers to factored components in AF */
    double* DLF = ws->AF;
    double* DF = ws->AF + m;
    double* DUF = ws->AF + m + n;
    double* DU2 = ws->AF + m + 2 * n;

    int zerot = (imat >= 8 && imat <= 10);
    int izero = 0;

    /* Set up parameters with DLATB4 */
    char type, dist;
    int kl, ku, mode;
    double anorm, cndnum;
    dlatb4("DGT", imat, n, n, &type, &kl, &ku, &anorm, &mode, &cndnum, &dist);

    /* Generate test matrix.
     * For types 1-6, each type has a unique seed.
     * For types 7-12, they share the same base seed because types 8-10
     * are singular versions of the type 7 matrix (with zeroed columns),
     * and types 11-12 are scaled versions of type 7. */
    uint64_t seed;
    if (imat <= 6) {
        seed = 1988 + n * 1000 + imat * 100;
    } else {
        seed = 1988 + n * 1000 + 7 * 100;  /* Types 7-12 share type 7's seed */
    }
    int info;

    if (imat <= 6) {
        /* Types 1-6: generate matrices of known condition number.
         * Generate in band storage: kl+ku+1 rows */
        double* AB = ws->AF;  /* Use AF as band storage workspace */
        int lda_band = kl + ku + 1;

        dlatms(n, n, &dist, seed, &type, ws->RWORK, mode, cndnum,
               anorm, kl, ku, "Z", AB, lda_band, ws->WORK, &info);
        if (info != 0) {
            fail_msg("DLATMS info=%d", info);
            return;
        }
        izero = 0;

        /* Extract tridiagonal from band storage.
         * LAPACK band storage: AB(ku+1+i-j, j) = A(i,j)  (1-indexed)
         * In C (0-indexed): AB[ku + i - j + j*lda] = A[i,j]
         * For diagonal (i=j): AB[ku + j*lda] = A[j,j]
         * For superdiag (i=j-1): AB[ku-1 + j*lda] = A[j-1,j], valid for j=1..n-1
         * For subdiag (i=j+1): AB[ku+1 + j*lda] = A[j+1,j], valid for j=0..n-2
         */
        for (int i = 0; i < n; i++) {
            D[i] = AB[ku + lda_band * i];
        }
        /* For diagonal-only matrices (kl=ku=0), DL and DU are zero */
        if (kl == 0) {
            for (int i = 0; i < m; i++) DL[i] = 0.0;
        } else {
            for (int i = 0; i < m; i++) {
                DL[i] = AB[(ku + 1) + lda_band * i];
            }
        }
        if (ku == 0) {
            for (int i = 0; i < m; i++) DU[i] = 0.0;
        } else {
            for (int i = 0; i < m; i++) {
                DU[i] = AB[(ku - 1) + lda_band * (i + 1)];
            }
        }
    } else {
        /* Types 7-12: generate tridiagonal matrices with unknown condition */
        rng_seed(seed);

        if (!zerot) {
            /* Generate matrix with elements from [-1,1] */
            for (int i = 0; i < n + 2 * m; i++) {
                ws->A[i] = 2.0 * rng_uniform() - 1.0;
            }
            if (anorm != 1.0) {
                cblas_dscal(n + 2 * m, anorm, ws->A, 1);
            }
        }

        /* Zero out elements for singular matrix types 8-10.
         * LAPACK uses flat array A(1:3N-2) = [DL(1:M), D(1:N), DU(1:M)]
         * imat=8: Zero first column → A(N) = 0 (D[0]) and A(1) = 0 (DL[0])
         * imat=9: Zero last column → A(3N-2) = 0 (DU[M-1]) and A(2N-1) = 0 (D[N-1])
         * imat=10: Zero middle columns → Zero elements from IZERO to N
         */
        if (imat == 8) {
            izero = 1;
            D[0] = 0.0;
            if (n > 1) DL[0] = 0.0;
        } else if (imat == 9) {
            izero = n;
            if (n > 1) DU[n - 2] = 0.0;
            D[n - 1] = 0.0;
        } else if (imat == 10) {
            izero = (n + 1) / 2;
            /* Zero from column izero to n (1-indexed), i.e., indices izero-1 to n-1 (0-indexed).
             * For N=10, IZERO=5: zero columns 5-10 (1-indexed) = indices 4-9 (0-indexed). */
            for (int i = izero - 1; i < n; i++) {
                if (i < n - 1) DU[i] = 0.0;
                D[i] = 0.0;
                if (i > 0) DL[i - 1] = 0.0;
            }
        } else {
            izero = 0;
        }
    }

    /* Skip FACT='F' for singular matrices */
    if (zerot && ifact == 0) {
        return;
    }

    /*
     * Compute condition numbers RCONDO and RCONDI.
     * For FACT='N', reuse values from FACT='F' iteration.
     * Since tests are independent, we always compute them.
     */
    double rcondo = 0.0, rcondi = 0.0;

    if (zerot) {
        rcondo = 0.0;
        rcondi = 0.0;
    } else if (n == 0) {
        rcondo = 1.0 / cndnum;
        rcondi = 1.0 / cndnum;
    } else {
        /* Copy tridiagonal to AF and factor */
        cblas_dcopy(n + 2 * m, ws->A, 1, ws->AF, 1);

        double anormo = dlangt("1", n, DL, D, DU);
        double anormi = dlangt("I", n, DL, D, DU);

        dgttrf(n, DLF, DF, DUF, DU2, ws->IWORK, &info);

        /* Compute inverse norm using DGTTRS */
        double ainvnm = 0.0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) ws->X[j] = 0.0;
            ws->X[i] = 1.0;
            dgttrs("N", n, 1, DLF, DF, DUF, DU2, ws->IWORK, ws->X, lda, &info);
            double colsum = cblas_dasum(n, ws->X, 1);
            if (colsum > ainvnm) ainvnm = colsum;
        }

        if (anormo <= 0.0 || ainvnm <= 0.0) {
            rcondo = 1.0;
        } else {
            rcondo = (1.0 / anormo) / ainvnm;
        }

        /* Compute infinity-norm condition number */
        ainvnm = 0.0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) ws->X[j] = 0.0;
            ws->X[i] = 1.0;
            dgttrs("T", n, 1, DLF, DF, DUF, DU2, ws->IWORK, ws->X, lda, &info);
            double colsum = cblas_dasum(n, ws->X, 1);
            if (colsum > ainvnm) ainvnm = colsum;
        }

        if (anormi <= 0.0 || ainvnm <= 0.0) {
            rcondi = 1.0;
        } else {
            rcondi = (1.0 / anormi) / ainvnm;
        }
    }

    double rcondc = (itran == 0) ? rcondo : rcondi;

    /* Generate NRHS random solution vectors */
    rng_seed(seed + (uint64_t)itran);
    for (int j = 0; j < NRHS; j++) {
        for (int i = 0; i < n; i++) {
            ws->XACT[j * lda + i] = 2.0 * rng_uniform() - 1.0;
        }
    }

    /* Set the right hand side using DLAGTM */
    dlagtm(trans, n, NRHS, 1.0, DL, D, DU, ws->XACT, lda, 0.0, ws->B, lda);

    /* --- Test DGTSV --- */
    if (ifact == 1 && itran == 0) {
        cblas_dcopy(n + 2 * m, ws->A, 1, ws->AF, 1);
        dlacpy("Full", n, NRHS, ws->B, lda, ws->X, lda);

        dgtsv(n, NRHS, DLF, DF, DUF, ws->X, lda, &info);

        if (zerot) {
            if (info <= 0) {
                fail_msg("DGTSV: expected INFO > 0 for singular matrix, got %d", info);
                return;
            }
        } else if (info != izero) {
            fail_msg("DGTSV info=%d expected=%d", info, izero);
            return;
        }

        int nt = 1;
        if (izero == 0 && info == 0) {
            /* TEST 2: Check residual of computed solution */
            dlacpy("Full", n, NRHS, ws->B, lda, ws->WORK, lda);
            dgtt02(trans, n, NRHS, DL, D, DU, ws->X, lda, ws->WORK, lda, &result[1]);

            /* TEST 3: Check solution from generated exact solution */
            dget04(n, NRHS, ws->X, lda, ws->XACT, lda, rcondc, &result[2]);
            nt = 3;
        }

        for (int k = 1; k < nt; k++) {
            if (result[k] >= THRESH) {
                fail_msg("DGTSV test %d failed: result=%e >= thresh=%e",
                         k + 1, result[k], THRESH);
            }
        }
    }

    /* --- Test DGTSVX --- */
    if (ifact > 0) {
        /* Initialize AF to zero */
        for (int i = 0; i < 3 * n - 2; i++) ws->AF[i] = 0.0;
    }
    dlaset("Full", n, NRHS, 0.0, 0.0, ws->X, lda);

    double rcond;
    dgtsvx(fact, trans, n, NRHS, DL, D, DU, DLF, DF, DUF, DU2,
           ws->IWORK, ws->B, lda, ws->X, lda, &rcond,
           ws->RWORK, &ws->RWORK[NRHS], ws->WORK, &ws->IWORK[n], &info);

    if (zerot) {
        if (info <= 0) {
            fail_msg("DGTSVX: expected INFO > 0 for singular matrix, got %d", info);
            return;
        }
    } else if (info != izero) {
        fail_msg("DGTSVX info=%d expected=%d", info, izero);
        return;
    }

    int k1;
    int nt = 5;
    if (ifact >= 1) {
        /* TEST 1: Reconstruct matrix from factors */
        dgtt01(n, DL, D, DU, DLF, DF, DUF, DU2, ws->IWORK,
               ws->WORK, lda, &result[0]);
        k1 = 1;
    } else {
        k1 = 2;
    }

    if (info == 0) {
        /* TEST 2: Check residual of computed solution */
        dlacpy("Full", n, NRHS, ws->B, lda, ws->WORK, lda);
        dgtt02(trans, n, NRHS, DL, D, DU, ws->X, lda, ws->WORK, lda, &result[1]);

        /* TEST 3: Check solution from generated exact solution */
        dget04(n, NRHS, ws->X, lda, ws->XACT, lda, rcondc, &result[2]);

        /* TEST 4-5: Check error bounds from iterative refinement */
        dgtt05(trans, n, NRHS, DL, D, DU, ws->B, lda, ws->X, lda, ws->XACT, lda,
               ws->RWORK, &ws->RWORK[NRHS], &result[3]);
        nt = 5;
    }

    /* TEST 6: Compare RCOND from DGTSVX with computed value */
    result[5] = dget06(rcond, rcondc);

    /* Check results */
    for (int k = k1 - 1; k < nt; k++) {
        if (result[k] >= THRESH) {
            fail_msg("DGTSVX FACT=%s TRANS=%s test %d: result=%e >= thresh=%e",
                     fact, trans, k + 1, result[k], THRESH);
        }
    }
    if (result[5] >= THRESH) {
        fail_msg("DGTSVX FACT=%s TRANS=%s test 6: result=%e >= thresh=%e",
                 fact, trans, result[5], THRESH);
    }
}

static void test_ddrvgt_case(void** state)
{
    ddrvgt_params_t* p = *state;
    run_ddrvgt_single(p->n, p->imat, p->ifact, p->itran);
}

#define MAX_TESTS 3000

static ddrvgt_params_t g_params[MAX_TESTS];
static struct CMUnitTest g_tests[MAX_TESTS];
static int g_num_tests = 0;

static void build_test_array(void)
{
    static const char* FACTS[] = {"F", "N"};
    static const char* TRANSS[] = {"N", "T", "C"};

    g_num_tests = 0;

    for (int in = 0; in < (int)NN; in++) {
        int n = NVAL[in];
        int nimat = (n <= 0) ? 1 : NTYPES;

        for (int imat = 1; imat <= nimat; imat++) {
            int zerot = (imat >= 8 && imat <= 10);

            for (int ifact = 0; ifact < 2; ifact++) {
                if (zerot && ifact == 0) continue;

                for (int itran = 0; itran < 3; itran++) {
                    ddrvgt_params_t* p = &g_params[g_num_tests];
                    p->n = n;
                    p->imat = imat;
                    p->ifact = ifact;
                    p->itran = itran;
                    snprintf(p->name, sizeof(p->name),
                             "n%d_t%d_%s_%s",
                             n, imat, FACTS[ifact], TRANSS[itran]);

                    g_tests[g_num_tests].name = p->name;
                    g_tests[g_num_tests].test_func = test_ddrvgt_case;
                    g_tests[g_num_tests].setup_func = NULL;
                    g_tests[g_num_tests].teardown_func = NULL;
                    g_tests[g_num_tests].initial_state = p;

                    g_num_tests++;
                }
            }
        }
    }
}

int main(void)
{
    build_test_array();
    return _cmocka_run_group_tests("ddrvgt", g_tests, g_num_tests,
                                   group_setup, group_teardown);
}
