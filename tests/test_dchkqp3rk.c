/**
 * @file test_dchkqp3rk.c
 * @brief Comprehensive test suite for DGEQP3RK (QR with column pivoting, rank-revealing).
 *
 * This is a faithful port of LAPACK's TESTING/LIN/dchkqp3rk.f to C using CMocka.
 * Tests DGEQP3RK for various matrix types and block sizes.
 *
 * Test structure from dchkqp3rk.f:
 *   TEST 1: norm(svd(R) - svd(A)) / (||svd(A)|| * eps * max(M,N)) via dqrt12
 *           (only when KFACT == MINMN)
 *   TEST 2: norm(A*P - Q*R) / (||A|| * eps * max(M,N)) via dqpt01
 *   TEST 3: norm(Q'*Q - I) / (eps * M) via dqrt11
 *   TEST 4: Verify R diagonal is non-increasing in absolute value
 *   TEST 5: norm(Q**T * B - Q**T * B) / (M * eps)
 *
 * Parameters from dtest.in:
 *   M values: 0, 1, 2, 3, 5, 10, 50
 *   N values: 0, 1, 2, 3, 5, 10, 50
 *   NRHS values: 1, 2, 4
 *   NB values: 1, 3, 3, 3, 20
 *   Matrix types: 1-19
 *   THRESH: 30.0
 */

#include "test_harness.h"
#include "test_rng.h"
#include <cblas.h>

/* Test parameters from dtest.in */
static const int MVAL[] = {0, 1, 2, 3, 5, 10, 50};
static const int NVAL[] = {0, 1, 2, 3, 5, 10, 50};
static const int NSVAL[] = {1, 2, 4};
static const int NBVAL[] = {1, 3, 3, 3, 20};
static const int NXVAL[] = {1, 0, 5, 9, 1};

#define NM      (sizeof(MVAL) / sizeof(MVAL[0]))
#define NN      (sizeof(NVAL) / sizeof(NVAL[0]))
#define NNS     (sizeof(NSVAL) / sizeof(NSVAL[0]))
#define NNB     (sizeof(NBVAL) / sizeof(NBVAL[0]))
#define NTYPES  19
#define NTESTS  5
#define THRESH  30.0
#define NMAX    50
#define NSMAX   4
#define BIGNUM  1.0e+38

/* Routine under test */
extern void dgeqp3rk(const int m, const int n, const int nrhs, const int kmax,
                     f64 abstol, f64 reltol, f64* A, const int lda,
                     int* K, f64* maxc2nrmk, f64* relmaxc2nrmk,
                     int* jpiv, f64* tau, f64* work, const int lwork,
                     int* iwork, int* info);

/* Verification routines */
extern f64 dqrt11(const int m, const int k, const f64* A, const int lda,
                     const f64* tau, f64* work, const int lwork);
extern f64 dqrt12(const int m, const int n, const f64* A, const int lda,
                     const f64* S, f64* work, const int lwork);
extern f64 dqpt01(const int m, const int n, const int k,
                     const f64* A, const f64* AF, const int lda,
                     const f64* tau, const int* jpvt,
                     f64* work, const int lwork);

/* Matrix generation */
extern void dlatb4(const char* path, const int imat, const int m, const int n,
                   char* type, int* kl, int* ku, f64* anorm, int* mode,
                   f64* cndnum, char* dist);
extern void dlatms(const int m, const int n, const char* dist,
                   const char* sym, f64* d, const int mode, const f64 cond,
                   const f64 dmax, const int kl, const int ku, const char* pack,
                   f64* A, const int lda, f64* work, int* info,
                   uint64_t state[static 4]);
extern void dlaord(const char* job, const int n, f64* X, const int incx);

/* Utilities */
extern void dlacpy(const char* uplo, const int m, const int n,
                   const f64* A, const int lda, f64* B, const int ldb);
extern void dlaset(const char* uplo, const int m, const int n,
                   const f64 alpha, const f64 beta,
                   f64* A, const int lda);
extern f64 dlamch(const char* cmach);
extern f64 dlange(const char* norm, const int m, const int n,
                     const f64* A, const int lda, f64* work);
extern void dormqr(const char* side, const char* trans,
                   const int m, const int n, const int k,
                   const f64* A, const int lda, const f64* tau,
                   f64* C, const int ldc, f64* work, const int lwork,
                   int* info);

/**
 * Test parameters for a single test case.
 */
typedef struct {
    int m;
    int n;
    int nrhs;
    int imat;
    int inb;
    int kmax;
    char name[128];
} dchkqp3rk_params_t;

/**
 * Workspace for test execution - shared across all tests via group setup.
 */
typedef struct {
    f64* A;       /* Working matrix (NMAX * (NMAX + NSMAX)) - holds A and appended B */
    f64* COPYA;   /* Copy of original A (NMAX * NMAX) */
    f64* B;       /* RHS matrix (NMAX * NSMAX) */
    f64* COPYB;   /* Copy of original B (NMAX * NSMAX) */
    f64* S;       /* Singular values (NMAX) */
    f64* TAU;     /* Scalar factors of elementary reflectors (NMAX) */
    f64* WORK;    /* General workspace */
    int* IWORK;      /* Integer workspace for pivot indicators + pivot array + internal */
    int lwork;
} dchkqp3rk_workspace_t;

static dchkqp3rk_workspace_t* g_workspace = NULL;
static uint64_t g_seed = 0;

/**
 * Group setup - allocate workspace once for all tests.
 */
static int group_setup(void** state)
{
    (void)state;
    g_workspace = malloc(sizeof(dchkqp3rk_workspace_t));
    if (!g_workspace) return -1;

    /* Workspace size from dchkqp3rk.f:
     * LWORK = MAX(1, M*MAX(M,N) + 4*MINMN + MAX(M,N), M*N + 2*MINMN + 4*N)
     * We use a generous upper bound for NMAX.
     */
    int lwork = NMAX * NMAX * 3 + 10 * NMAX;
    g_workspace->lwork = lwork;

    /* A holds M-by-N matrix plus appended M-by-NRHS for RHS */
    g_workspace->A = malloc(NMAX * (NMAX + NSMAX) * sizeof(f64));
    g_workspace->COPYA = malloc(NMAX * NMAX * sizeof(f64));
    g_workspace->B = malloc(NMAX * NSMAX * sizeof(f64));
    g_workspace->COPYB = malloc(NMAX * NSMAX * sizeof(f64));
    g_workspace->S = malloc(NMAX * sizeof(f64));
    g_workspace->TAU = malloc(NMAX * sizeof(f64));
    g_workspace->WORK = malloc(lwork * sizeof(f64));
    /* IWORK: N for zero indicators + N for pivot array + N for internal work */
    g_workspace->IWORK = malloc(3 * NMAX * sizeof(int));

    if (!g_workspace->A || !g_workspace->COPYA || !g_workspace->B ||
        !g_workspace->COPYB || !g_workspace->S || !g_workspace->TAU ||
        !g_workspace->WORK || !g_workspace->IWORK) {
        return -1;
    }

    /* Initialize seed from LAPACK's ISEEDY = {1988, 1989, 1990, 1991} */
    g_seed = 1988198919901991ULL;

    return 0;
}

/**
 * Group teardown - free workspace.
 */
static int group_teardown(void** state)
{
    (void)state;
    if (g_workspace) {
        free(g_workspace->A);
        free(g_workspace->COPYA);
        free(g_workspace->B);
        free(g_workspace->COPYB);
        free(g_workspace->S);
        free(g_workspace->TAU);
        free(g_workspace->WORK);
        free(g_workspace->IWORK);
        free(g_workspace);
        g_workspace = NULL;
    }
    return 0;
}

/**
 * Generate matrix for IMAT type.
 *
 * Matrix types 1-19 from dchkqp3rk.f:
 *   1: Zero matrix
 *   2: Diagonal, well-conditioned
 *   3: Upper triangular, well-conditioned
 *   4: Lower triangular, well-conditioned
 *   5: First column zero
 *   6: Last MINMN column zero
 *   7: Last N column zero
 *   8: Middle column in MINMN zero
 *   9: First half of MINMN columns zero
 *  10: Last columns zero starting from MINMN/2+1
 *  11: Middle half of MINMN columns zero
 *  12: Odd columns zero
 *  13: Even columns zero
 *  14: Well-conditioned random
 *  15: Ill-conditioned (sqrt(0.1/eps))
 *  16: Very ill-conditioned (0.1/eps)
 *  17: One small singular value
 *  18: Scaled near underflow
 *  19: Scaled near overflow
 *
 * @return 0 on success, -1 if this IMAT should be skipped for this size
 */
static int generate_matrix(int m, int n, int imat, f64* COPYA, int lda,
                           f64* S, f64* WORK, uint64_t rng_state[static 4])
{
    const f64 ZERO = 0.0;
    int minmn = (m < n) ? m : n;
    int info;
    char type, dist;
    int kl, ku, mode;
    f64 anorm, cndnum;

    if (imat == 1) {
        /* Matrix 1: Zero matrix */
        dlaset("F", m, n, ZERO, ZERO, COPYA, lda);
        for (int i = 0; i < minmn; i++) {
            S[i] = ZERO;
        }
        return 0;
    }

    if ((imat >= 2 && imat <= 4) || (imat >= 14 && imat <= 19)) {
        /* Matrices 2-4, 14-19: Generate via dlatb4 + dlatms */
        dlatb4("DQK", imat, m, n, &type, &kl, &ku, &anorm, &mode, &cndnum, &dist);

        char dist_str[2] = {dist, '\0'};
        char type_str[2] = {type, '\0'};
        dlatms(m, n, dist_str,
               type_str, S, mode, cndnum, anorm,
               kl, ku, "N", COPYA, lda, WORK, &info, rng_state);

        if (info != 0) {
            return -1;
        }

        /* Sort singular values in decreasing order */
        dlaord("D", minmn, S, 1);
        return 0;
    }

    if (minmn >= 2 && imat >= 5 && imat <= 13) {
        /* Matrices 5-13: Contain zero columns, only for MINMN >= 2 */
        int jb_zero = 0;
        int nb_zero = 0;
        int nb_gen = 0;
        int j_inc = 1;
        int j_first_nz = 1;

        if (imat == 5) {
            /* First column is zero */
            jb_zero = 1;
            nb_zero = 1;
            nb_gen = n - nb_zero;
        } else if (imat == 6) {
            /* Last column MINMN is zero */
            jb_zero = minmn;
            nb_zero = 1;
            nb_gen = n - nb_zero;
        } else if (imat == 7) {
            /* Last column N is zero */
            jb_zero = n;
            nb_zero = 1;
            nb_gen = n - nb_zero;
        } else if (imat == 8) {
            /* Middle column in MINMN is zero */
            jb_zero = minmn / 2 + 1;
            nb_zero = 1;
            nb_gen = n - nb_zero;
        } else if (imat == 9) {
            /* First half of MINMN columns is zero */
            jb_zero = 1;
            nb_zero = minmn / 2;
            nb_gen = n - nb_zero;
        } else if (imat == 10) {
            /* Last columns are zero, starting from MINMN/2+1 */
            jb_zero = minmn / 2 + 1;
            nb_zero = n - jb_zero + 1;
            nb_gen = n - nb_zero;
        } else if (imat == 11) {
            /* Half of the columns in the middle of MINMN are zero */
            jb_zero = minmn / 2 - (minmn / 2) / 2 + 1;
            nb_zero = minmn / 2;
            nb_gen = n - nb_zero;
        } else if (imat == 12) {
            /* Odd-numbered columns are zero */
            nb_gen = n / 2;
            nb_zero = n - nb_gen;
            j_inc = 2;
            j_first_nz = 2;
        } else if (imat == 13) {
            /* Even-numbered columns are zero */
            nb_zero = n / 2;
            nb_gen = n - nb_zero;
            j_inc = 2;
            j_first_nz = 1;
        }

        if (nb_gen <= 0) {
            return -1;  /* Skip this configuration */
        }

        /* Set the first NB_ZERO columns to zero */
        dlaset("F", m, nb_zero, ZERO, ZERO, COPYA, lda);

        /* Generate an M-by-NB_GEN matrix in COPYA starting at column NB_ZERO */
        dlatb4("DQK", imat, m, nb_gen, &type, &kl, &ku, &anorm, &mode, &cndnum, &dist);

        char dist_str[2] = {dist, '\0'};
        char type_str[2] = {type, '\0'};
        int ind_offset_gen = nb_zero * lda;
        dlatms(m, nb_gen, dist_str,
               type_str, S, mode, cndnum, anorm,
               kl, ku, "N", &COPYA[ind_offset_gen], lda, WORK, &info, rng_state);

        if (info != 0) {
            return -1;
        }

        /* Sort singular values in decreasing order for the generated part */
        int minmnb_gen = (m < nb_gen) ? m : nb_gen;
        if (minmnb_gen > 0) {
            dlaord("D", minmnb_gen, S, 1);
        }

        /* Swap the generated columns into correct positions */
        if (imat == 6 || imat == 7 || imat == 8 || imat == 10 || imat == 11) {
            /* Move columns from right block into left positions */
            for (int j = 0; j < jb_zero - 1; j++) {
                cblas_dswap(m, &COPYA[(nb_zero + j) * lda], 1,
                            &COPYA[j * lda], 1);
            }
        } else if (imat == 12 || imat == 13) {
            /* Swap generated columns into even/odd positions */
            for (int j = 0; j < nb_gen; j++) {
                int ind_out = (nb_zero + j) * lda;
                int ind_in = (j_inc * j + (j_first_nz - 1)) * lda;
                cblas_dswap(m, &COPYA[ind_out], 1, &COPYA[ind_in], 1);
            }
        }

        /* Add trailing zeros to singular values */
        for (int i = minmnb_gen; i < minmn; i++) {
            S[i] = ZERO;
        }

        return 0;
    }

    /* MINMN < 2 and IMAT in 5-13: skip */
    return -1;
}

/**
 * Run the full dchkqp3rk test battery for a single configuration.
 */
static void run_dchkqp3rk_single(int m, int n, int nrhs, int imat, int inb, int kmax)
{
    const f64 ZERO = 0.0;
    const f64 NEGONE = -1.0;
    dchkqp3rk_workspace_t* ws = g_workspace;

    int info;
    int lda = (m > 1) ? m : 1;
    int minmn = (m < n) ? m : n;
    f64 eps = dlamch("E");
    f64 result[NTESTS];
    char ctx[256];

    /* Set block size and crossover point */
    int nb = NBVAL[inb];
    int nx = NXVAL[inb];
    xlaenv(1, nb);
    xlaenv(3, nx);

    /* Initialize results */
    for (int k = 0; k < NTESTS; k++) {
        result[k] = ZERO;
    }

    /* Seed based on (m, n, nrhs, imat) for reproducibility */
    uint64_t rng_state[4];
    rng_seed(rng_state, g_seed + (uint64_t)(m * 10000 + n * 1000 + nrhs * 100 + imat));

    /* Generate RHS matrix B (COPYB) with dlatb4/dlatms for IMAT=14 parameters */
    {
        char type, dist;
        int kl, ku, mode;
        f64 anorm, cndnum;
        dlatb4("DQK", 14, m, nrhs, &type, &kl, &ku, &anorm, &mode, &cndnum, &dist);
        char dist_str[2] = {dist, '\0'};
        char type_str[2] = {type, '\0'};
        dlatms(m, nrhs, dist_str,
               type_str, ws->S, mode, cndnum, anorm,
               kl, ku, "N", ws->COPYB, lda, ws->WORK, &info, rng_state);
        if (info != 0) {
            snprintf(ctx, sizeof(ctx), "m=%d n=%d nrhs=%d type=%d: COPYB generation failed (info=%d)",
                     m, n, nrhs, imat, info);
            set_test_context(ctx);
            assert_info_success(info);
            return;
        }
    }

    /* Generate test matrix COPYA */
    if (generate_matrix(m, n, imat, ws->COPYA, lda, ws->S, ws->WORK, rng_state) != 0) {
        /* Skip this configuration */
        return;
    }

    /* Initialize pivot indicators (IWORK[0:n-1]) to zero */
    for (int i = 0; i < n; i++) {
        ws->IWORK[i] = 0;
    }

    /* Get working copies:
     * - COPYA -> A[0:m*n-1]
     * - COPYB -> A[lda*n : lda*n + m*nrhs - 1] (appended)
     * - COPYB -> B
     * - IWORK[0:n-1] -> IWORK[n:2n-1] (pivot array)
     */
    dlacpy("A", m, n, ws->COPYA, lda, ws->A, lda);
    dlacpy("A", m, nrhs, ws->COPYB, lda, &ws->A[lda * n], lda);
    dlacpy("A", m, nrhs, ws->COPYB, lda, ws->B, lda);
    for (int i = 0; i < n; i++) {
        ws->IWORK[n + i] = ws->IWORK[i];
    }

    /* Workspace size for DGEQP3RK */
    int lw = (2 * n + nb * (n + nrhs + 1) > 1) ? (2 * n + nb * (n + nrhs + 1)) : 1;
    if (3 * n + nrhs - 1 > lw) {
        lw = 3 * n + nrhs - 1;
    }

    /* Tolerance values: -1.0 means auto-compute */
    f64 abstol = -1.0;
    f64 reltol = -1.0;

    /* Call DGEQP3RK */
    int kfact;
    f64 maxc2nrmk, relmaxc2nrmk;
    dgeqp3rk(m, n, nrhs, kmax, abstol, reltol, ws->A, lda,
             &kfact, &maxc2nrmk, &relmaxc2nrmk, &ws->IWORK[n], ws->TAU,
             ws->WORK, lw, &ws->IWORK[2 * n], &info);

    if (info < 0) {
        snprintf(ctx, sizeof(ctx), "m=%d n=%d nrhs=%d kmax=%d type=%d nb=%d: DGEQP3RK failed (info=%d)",
                 m, n, nrhs, kmax, imat, nb, info);
        set_test_context(ctx);
        assert_info_success(info);
        return;
    }

    /* TEST 1: norm(svd(R) - svd(A)) / (||svd(A)|| * eps * max(M,N)) via dqrt12
     *         Only when KFACT == MINMN (full rank factorization) */
    if (kfact == minmn) {
        result[0] = dqrt12(m, n, ws->A, lda, ws->S, ws->WORK, ws->lwork);
        snprintf(ctx, sizeof(ctx), "m=%d n=%d nrhs=%d kmax=%d type=%d nb=%d TEST 1 (SVD comparison)",
                 m, n, nrhs, kmax, imat, nb);
        set_test_context(ctx);
        assert_residual_below(result[0], THRESH);
    }

    /* TEST 2: norm(A*P - Q*R) / (||A|| * eps * max(M,N)) via dqpt01 */
    result[1] = dqpt01(m, n, kfact, ws->COPYA, ws->A, lda, ws->TAU,
                       &ws->IWORK[n], ws->WORK, ws->lwork);
    snprintf(ctx, sizeof(ctx), "m=%d n=%d nrhs=%d kmax=%d type=%d nb=%d TEST 2 (factorization)",
             m, n, nrhs, kmax, imat, nb);
    set_test_context(ctx);
    assert_residual_below(result[1], THRESH);

    /* TEST 3: norm(Q'*Q - I) / (eps * M) via dqrt11 */
    result[2] = dqrt11(m, kfact, ws->A, lda, ws->TAU, ws->WORK, ws->lwork);
    snprintf(ctx, sizeof(ctx), "m=%d n=%d nrhs=%d kmax=%d type=%d nb=%d TEST 3 (orthogonality)",
             m, n, nrhs, kmax, imat, nb);
    set_test_context(ctx);
    assert_residual_below(result[2], THRESH);

    /* TEST 4: Verify R diagonal is non-increasing in absolute value
     *         Only when min(KFACT, MINMN) >= 2 */
    int ktest4 = (kfact < minmn) ? kfact : minmn;
    if (ktest4 >= 2) {
        result[3] = ZERO;
        f64 r11 = ws->A[0];  /* R(0,0) */
        if (fabs(r11) > ZERO) {
            for (int j = 0; j < kfact - 1; j++) {
                /* R(j,j) is at A[j + j*lda], R(j+1,j+1) is at A[(j+1) + (j+1)*lda] */
                f64 rjj = fabs(ws->A[j + j * lda]);
                f64 rjj1 = fabs(ws->A[(j + 1) + (j + 1) * lda]);
                f64 dtemp = (rjj - rjj1) / fabs(r11);
                if (dtemp < ZERO) {
                    result[3] = BIGNUM;
                    break;
                }
            }
        }
        snprintf(ctx, sizeof(ctx), "m=%d n=%d nrhs=%d kmax=%d type=%d nb=%d TEST 4 (R diagonal non-increasing)",
                 m, n, nrhs, kmax, imat, nb);
        set_test_context(ctx);
        assert_residual_below(result[3], THRESH);
    }

    /* TEST 5: Verify Q**T applied to RHS correctly
     *         Only when MINMN > 0 */
    if (minmn > 0) {
        /* B already contains COPYB; A[lda*n:] contains Q**T * COPYB from dgeqp3rk */
        /* Apply Q**T to B using dormqr */
        int lwork_mqr = (nrhs > 1) ? nrhs : 1;
        dormqr("L", "T", m, nrhs, kfact, ws->A, lda, ws->TAU,
               ws->B, lda, ws->WORK, lwork_mqr, &info);

        if (info == 0) {
            /* Compute B := B - A[lda*n:] (where A[lda*n:] is Q**T * COPYB from dgeqp3rk) */
            for (int i = 0; i < nrhs; i++) {
                cblas_daxpy(m, NEGONE, &ws->A[lda * n + i * lda], 1,
                            &ws->B[i * lda], 1);
            }

            /* result[4] = ||B|| / (M * eps) */
            f64 rdummy[1];
            f64 bnorm = dlange("1", m, nrhs, ws->B, lda, rdummy);
            result[4] = fabs(bnorm / ((f64)m * eps));

            snprintf(ctx, sizeof(ctx), "m=%d n=%d nrhs=%d kmax=%d type=%d nb=%d TEST 5 (Q**T * B)",
                     m, n, nrhs, kmax, imat, nb);
            set_test_context(ctx);
            assert_residual_below(result[4], THRESH);
        }
    }

    clear_test_context();
}

/**
 * CMocka test function - dispatches to run_dchkqp3rk_single based on prestate.
 */
static void test_dchkqp3rk_case(void** state)
{
    dchkqp3rk_params_t* params = *state;
    run_dchkqp3rk_single(params->m, params->n, params->nrhs, params->imat,
                         params->inb, params->kmax);
}

/*
 * Generate all parameter combinations.
 *
 * To keep test count manageable, we use representative KMAX values:
 *   - kmax = 0
 *   - kmax = minmn/2 (if minmn >= 2)
 *   - kmax = minmn
 *   - kmax = minmn + 1 (tests KMAX > N handling)
 *
 * Total test count will be approximately:
 * NM * NN * NNS * NTYPES * NNB * 4 = 7 * 7 * 3 * 19 * 5 * 4 = 55860 tests
 *
 * We further reduce by only testing a subset of matrix types in the full matrix.
 * For quick sanity, we test all types but only for select (m,n) combinations.
 */

#define MAX_TESTS 100000

static dchkqp3rk_params_t g_params[MAX_TESTS];
static struct CMUnitTest g_tests[MAX_TESTS];
static int g_num_tests = 0;

/**
 * Build the test array with parameter combinations.
 */
static void build_test_array(void)
{
    g_num_tests = 0;

    for (int im = 0; im < (int)NM; im++) {
        int m = MVAL[im];

        for (int in = 0; in < (int)NN; in++) {
            int n = NVAL[in];
            int minmn = (m < n) ? m : n;

            for (int ins = 0; ins < (int)NNS; ins++) {
                int nrhs = NSVAL[ins];

                for (int imat = 1; imat <= NTYPES; imat++) {
                    /* Skip imat 5-13 for minmn < 2 */
                    if (minmn < 2 && imat >= 5 && imat <= 13) {
                        continue;
                    }

                    for (int inb = 0; inb < (int)NNB; inb++) {
                        int nb = NBVAL[inb];

                        /* Representative KMAX values */
                        int kmax_vals[4];
                        int num_kmax = 0;

                        kmax_vals[num_kmax++] = 0;
                        if (minmn >= 2) {
                            kmax_vals[num_kmax++] = minmn / 2;
                        }
                        kmax_vals[num_kmax++] = minmn;
                        kmax_vals[num_kmax++] = minmn + 1;

                        for (int ik = 0; ik < num_kmax; ik++) {
                            int kmax = kmax_vals[ik];

                            if (g_num_tests >= MAX_TESTS) {
                                return;
                            }

                            dchkqp3rk_params_t* p = &g_params[g_num_tests];
                            p->m = m;
                            p->n = n;
                            p->nrhs = nrhs;
                            p->imat = imat;
                            p->inb = inb;
                            p->kmax = kmax;
                            snprintf(p->name, sizeof(p->name),
                                     "dchkqp3rk_m%d_n%d_nrhs%d_type%d_nb%d_kmax%d",
                                     m, n, nrhs, imat, nb, kmax);

                            g_tests[g_num_tests].name = p->name;
                            g_tests[g_num_tests].test_func = test_dchkqp3rk_case;
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
}

int main(void)
{
    build_test_array();
    return _cmocka_run_group_tests("dchkqp3rk", g_tests, g_num_tests,
                                   group_setup, group_teardown);
}
