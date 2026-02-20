/**
 * @file test_sdrvpt.c
 * @brief DDRVPT tests the driver routines SPTSV and SPTSVX.
 *
 * Port of LAPACK TESTING/LIN/ddrvpt.f to C with CMocka parameterization.
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
#define NTYPES  12
#define NTESTS  6
#define THRESH  30.0f
#define NMAX    50
#define NRHS    2

/* Routines under test */
extern void sptsv(const int n, const int nrhs, f32* D, f32* E,
                  f32* B, const int ldb, int* info);
extern void sptsvx(const char* fact, const int n, const int nrhs,
                   const f32* D, const f32* E,
                   f32* DF, f32* EF,
                   const f32* B, const int ldb,
                   f32* X, const int ldx, f32* rcond,
                   f32* ferr, f32* berr, f32* work, int* info);

/* Supporting routines */
extern void spttrf(const int n, f32* D, f32* E, int* info);
extern void spttrs(const int n, const int nrhs,
                   const f32* D, const f32* E,
                   f32* B, const int ldb, int* info);
extern void slaptm(const int n, const int nrhs,
                   const f32 alpha, const f32* D, const f32* E,
                   const f32* X, const int ldx,
                   const f32 beta, f32* B, const int ldb);

/* Verification routines */
extern void sptt01(const int n, const f32* D, const f32* E,
                   const f32* DF, const f32* EF,
                   f32* work, f32* resid);
extern void sptt02(const int n, const int nrhs,
                   const f32* D, const f32* E,
                   const f32* X, const int ldx,
                   f32* B, const int ldb, f32* resid);
extern void sptt05(const int n, const int nrhs,
                   const f32* D, const f32* E,
                   const f32* B, const int ldb,
                   const f32* X, const int ldx,
                   const f32* XACT, const int ldxact,
                   const f32* ferr, const f32* berr, f32* reslts);
extern void sget04(const int n, const int nrhs, const f32* X, const int ldx,
                   const f32* XACT, const int ldxact, const f32 rcond, f32* resid);
extern f32 sget06(const f32 rcond, const f32 rcondc);
extern f32 slanst(const char* norm, const int n, const f32* D, const f32* E);

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

/* Utilities */
extern void slacpy(const char* uplo, const int m, const int n,
                   const f32* A, const int lda, f32* B, const int ldb);
extern void slaset(const char* uplo, const int m, const int n,
                   const f32 alpha, const f32 beta, f32* A, const int lda);
extern f32 slamch(const char* cmach);

typedef struct {
    int n;
    int imat;
    int ifact;      /* 0='F', 1='N' */
    char name[64];
} ddrvpt_params_t;

typedef struct {
    f32* D;      /* Diagonal (N) */
    f32* E;      /* Off-diagonal (N-1) */
    f32* DF;     /* Factored diagonal (N) */
    f32* EF;     /* Factored off-diagonal (N-1) */
    f32* B;
    f32* X;
    f32* XACT;
    f32* WORK;
    f32* RWORK;
    f32* A;      /* Band storage for slatms (2 x N) */
} ddrvpt_workspace_t;

static ddrvpt_workspace_t* g_workspace = NULL;

static int group_setup(void** state)
{
    (void)state;
    g_workspace = malloc(sizeof(ddrvpt_workspace_t));
    if (!g_workspace) return -1;

    int nmax = NMAX;
    /* WORK needs to be large enough for slatms which uses n*n for full matrix
     * plus additional workspace for slagsy (roughly 2*n more) */
    int lwork = nmax * nmax + 4 * nmax;
    if (lwork < nmax * NRHS) lwork = nmax * NRHS;

    g_workspace->D = calloc(2 * nmax, sizeof(f32));
    g_workspace->E = calloc(2 * nmax, sizeof(f32));
    g_workspace->DF = g_workspace->D + nmax;  /* Use second half */
    g_workspace->EF = g_workspace->E + nmax;  /* Use second half */
    g_workspace->B = calloc(nmax * NRHS, sizeof(f32));
    g_workspace->X = calloc(nmax * NRHS, sizeof(f32));
    g_workspace->XACT = calloc(nmax * NRHS, sizeof(f32));
    g_workspace->WORK = calloc(lwork, sizeof(f32));
    g_workspace->RWORK = calloc(nmax > 2 * NRHS ? nmax : 2 * NRHS, sizeof(f32));
    g_workspace->A = calloc(2 * nmax, sizeof(f32));

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
    f32 result[NTESTS];
    for (int k = 0; k < NTESTS; k++) result[k] = 0.0f;

    /* Clear workspace to prevent stale data from previous tests */
    memset(ws->D, 0, 2 * NMAX * sizeof(f32));
    memset(ws->E, 0, 2 * NMAX * sizeof(f32));
    memset(ws->A, 0, 2 * NMAX * sizeof(f32));

    f32* D = ws->D;
    f32* E = ws->E;
    f32* DF = ws->DF;
    f32* EF = ws->EF;

    int zerot = (imat >= 8 && imat <= 10);
    int izero = 0;

    /* Set up parameters with SLATB4 */
    char type, dist;
    int kl, ku, mode;
    f32 anorm, cndnum;
    slatb4("SPT", imat, n, n, &type, &kl, &ku, &anorm, &mode, &cndnum, &dist);

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
    uint64_t rng_state[4];
    rng_seed(rng_state, seed);
    int info;

    (void)type; (void)dist; (void)mode; (void)kl; (void)ku;

    if (imat <= 6) {
        /* Types 1-6: generate a symmetric tridiagonal matrix of known condition
         * number in lower triangular band storage. */
        izero = 0;

        char symtype[2] = {type, '\0'};
        char disttype[2] = {dist, '\0'};

        slatms(n, n, disttype, symtype, ws->RWORK, mode, cndnum,
               anorm, kl, ku, "B", ws->A, 2, ws->WORK, &info, rng_state);

        if (info != 0) {
            fail_msg("SLATMS info=%d for imat=%d", info, imat);
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

        if (!zerot) {
            /* Let D and E have values from [-1,1] */
            for (int i = 0; i < n; i++) {
                D[i] = 2.0f * rng_uniform_f32(rng_state) - 1.0f;
            }
            for (int i = 0; i < n - 1; i++) {
                E[i] = 2.0f * rng_uniform_f32(rng_state) - 1.0f;
            }

            /* Make the tridiagonal matrix diagonally dominant */
            if (n == 1) {
                D[0] = fabsf(D[0]);
            } else {
                D[0] = fabsf(D[0]) + fabsf(E[0]);
                D[n - 1] = fabsf(D[n - 1]) + fabsf(E[n - 2]);
                for (int i = 1; i < n - 1; i++) {
                    D[i] = fabsf(D[i]) + fabsf(E[i]) + fabsf(E[i - 1]);
                }
            }

            /* Scale D and E so the maximum element is ANORM */
            int ix = cblas_isamax(n, D, 1);
            f32 dmax = D[ix];
            cblas_sscal(n, anorm / dmax, D, 1);
            if (n > 1) {
                cblas_sscal(n - 1, anorm / dmax, E, 1);
            }
        }

        /* Zero out elements for singular matrix types 8-10 */
        if (imat == 8) {
            izero = 1;
            D[0] = 0.0f;
            if (n > 1) E[0] = 0.0f;
        } else if (imat == 9) {
            izero = n;
            if (n > 1) E[n - 2] = 0.0f;
            D[n - 1] = 0.0f;
        } else if (imat == 10) {
            izero = (n + 1) / 2;
            if (izero > 1) {
                E[izero - 2] = 0.0f;
            }
            if (izero < n) {
                E[izero - 1] = 0.0f;
            }
            D[izero - 1] = 0.0f;
        } else {
            izero = 0;
        }
    }

    /* Skip FACT='F' for singular matrices */
    if (zerot && ifact == 0) {
        return;
    }

    /* Generate NRHS random solution vectors */
    rng_seed(rng_state, seed + 42);
    for (int j = 0; j < NRHS; j++) {
        for (int i = 0; i < n; i++) {
            ws->XACT[j * lda + i] = 2.0f * rng_uniform_f32(rng_state) - 1.0f;
        }
    }

    /* Set the right hand side using SLAPTM: B = A * XACT */
    slaptm(n, NRHS, 1.0f, D, E, ws->XACT, lda, 0.0f, ws->B, lda);

    /*
     * Compute condition number RCONDC for FACT='F'.
     * For FACT='N', reuse the value computed for FACT='F'.
     * Since tests are independent, we always compute it.
     */
    f32 rcondc = 0.0f;

    if (zerot) {
        rcondc = 0.0f;
    } else if (n == 0) {
        rcondc = 1.0f / cndnum;
    } else {
        /* Compute the 1-norm of A */
        f32 anorm_1 = slanst("1", n, D, E);

        /* Copy D and E to DF and EF, then factor */
        cblas_scopy(n, D, 1, DF, 1);
        if (n > 1) cblas_scopy(n - 1, E, 1, EF, 1);

        spttrf(n, DF, EF, &info);

        /* Use SPTTRS to solve for one column at a time of inv(A),
         * computing the maximum column sum as we go */
        f32 ainvnm = 0.0f;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) ws->X[j] = 0.0f;
            ws->X[i] = 1.0f;
            spttrs(n, 1, DF, EF, ws->X, lda, &info);
            f32 colsum = cblas_sasum(n, ws->X, 1);
            if (colsum > ainvnm) ainvnm = colsum;
        }

        /* Compute the 1-norm condition number of A */
        if (anorm_1 <= 0.0f || ainvnm <= 0.0f) {
            rcondc = 1.0f;
        } else {
            rcondc = (1.0f / anorm_1) / ainvnm;
        }
    }

    /* --- Test SPTSV --- */
    if (ifact == 1) {
        cblas_scopy(n, D, 1, DF, 1);
        if (n > 1) cblas_scopy(n - 1, E, 1, EF, 1);
        slacpy("Full", n, NRHS, ws->B, lda, ws->X, lda);

        sptsv(n, NRHS, DF, EF, ws->X, lda, &info);

        if (zerot) {
            if (info <= 0) {
                fail_msg("SPTSV: expected INFO > 0 for singular matrix, got %d", info);
                return;
            }
        } else if (info != izero) {
            fail_msg("SPTSV info=%d expected=%d", info, izero);
            return;
        }

        int nt = 0;
        if (izero == 0 && info == 0) {
            /* TEST 1: Check factorization norm(L*D*L' - A) / (n * norm(A) * eps) */
            sptt01(n, D, E, DF, EF, ws->WORK, &result[0]);

            /* TEST 2: Check residual of computed solution */
            slacpy("Full", n, NRHS, ws->B, lda, ws->WORK, lda);
            sptt02(n, NRHS, D, E, ws->X, lda, ws->WORK, lda, &result[1]);

            /* TEST 3: Check solution from generated exact solution */
            sget04(n, NRHS, ws->X, lda, ws->XACT, lda, rcondc, &result[2]);
            nt = 3;
        }

        for (int k = 0; k < nt; k++) {
            if (result[k] >= THRESH) {
                fail_msg("SPTSV test %d failed: result=%e >= thresh=%e",
                         k + 1, result[k], THRESH);
            }
        }
    }

    /* --- Test SPTSVX --- */
    if (ifact == 0) {
        /* FACT='F': Provide the factorization in DF and EF.
         * Copy D/E to DF/EF and factor with SPTTRF. */
        cblas_scopy(n, D, 1, DF, 1);
        if (n > 1) cblas_scopy(n - 1, E, 1, EF, 1);
        spttrf(n, DF, EF, &info);
        /* Ignore info here - we handle singular matrices via izero */
    } else {
        /* FACT='N': Initialize DF and EF to zero */
        for (int i = 0; i < n; i++) DF[i] = 0.0f;
        for (int i = 0; i < n - 1; i++) EF[i] = 0.0f;
    }
    slaset("Full", n, NRHS, 0.0f, 0.0f, ws->X, lda);

    f32 rcond;
    sptsvx(fact, n, NRHS, D, E, DF, EF, ws->B, lda, ws->X, lda, &rcond,
           ws->RWORK, &ws->RWORK[NRHS], ws->WORK, &info);

    if (zerot) {
        if (info <= 0) {
            fail_msg("SPTSVX: expected INFO > 0 for singular matrix, got %d", info);
            return;
        }
    } else if (info != izero) {
        fail_msg("SPTSVX info=%d expected=%d", info, izero);
        return;
    }

    int k1;
    if (izero == 0) {
        if (ifact >= 1) {
            /* TEST 1: Check factorization */
            sptt01(n, D, E, DF, EF, ws->WORK, &result[0]);
            k1 = 1;
        } else {
            k1 = 2;
        }

        /* TEST 2: Check residual of computed solution */
        slacpy("Full", n, NRHS, ws->B, lda, ws->WORK, lda);
        sptt02(n, NRHS, D, E, ws->X, lda, ws->WORK, lda, &result[1]);

        /* TEST 3: Check solution from generated exact solution */
        sget04(n, NRHS, ws->X, lda, ws->XACT, lda, rcondc, &result[2]);

        /* TEST 4-5: Check error bounds from iterative refinement */
        sptt05(n, NRHS, D, E, ws->B, lda, ws->X, lda, ws->XACT, lda,
               ws->RWORK, &ws->RWORK[NRHS], &result[3]);
    } else {
        k1 = 6;
    }

    /* TEST 6: Compare RCOND from SPTSVX with computed value */
    result[5] = sget06(rcond, rcondc);

    /* Check results */
    for (int k = k1 - 1; k < 5; k++) {
        if (result[k] >= THRESH) {
            fail_msg("SPTSVX FACT=%s test %d: result=%e >= thresh=%e",
                     fact, k + 1, result[k], THRESH);
        }
    }
    if (result[5] >= THRESH) {
        fail_msg("SPTSVX FACT=%s test 6: result=%e >= thresh=%e",
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
