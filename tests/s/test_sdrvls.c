/**
 * @file test_sdrvls.c
 * @brief Comprehensive test suite for least squares driver routines.
 *
 * This is a faithful port of LAPACK's TESTING/LIN/ddrvls.f to C using CMocka.
 * Tests SGELS, SGELST, SGETSLS, SGELSY, SGELSS, and SGELSD.
 *
 * Each (m, n, nrhs, type, nb) combination is registered as a separate CMocka
 * test, providing pytest-like parameterized test behavior with clear failure
 * isolation.
 *
 * Test structure from ddrvls.f:
 *   - Tests 1-2: SGELS (full-rank, trans='N' and 'T')
 *   - Tests 3-4: SGELST (full-rank, trans='N' and 'T')
 *   - Tests 5-6: SGETSLS (full-rank, TSQR-based, trans='N' and 'T')
 *   - Tests 7-10: SGELSY (rank-revealing QR with pivoting)
 *   - Tests 11-14: SGELSS (SVD-based)
 *   - Tests 15-18: SGELSD (divide-conquer SVD)
 *
 * Matrix types (6 types):
 *   1. Full rank, normally scaled
 *   2. Full rank, scaled up
 *   3. Full rank, scaled down
 *   4. Rank-deficient, normally scaled
 *   5. Rank-deficient, scaled up
 *   6. Rank-deficient, scaled down
 */

#include "test_harness.h"
#include "test_rng.h"
#include <string.h>
#include <stdio.h>
#include <cblas.h>
#include <math.h>

/* Test parameters from dtest.in - full LAPACK test coverage */
static const int MVAL[] = {0, 1, 2, 3, 5, 10, 50};
static const int NVAL[] = {0, 1, 2, 3, 5, 10, 50};
static const int NSVAL[] = {1, 2, 15};
/* Block size parameters from dtest.in (NB, NX pairs) */
static const int NBVAL[] = {1, 3, 3, 3, 20};
static const int NXVAL[] = {1, 0, 5, 9, 1};

#define NM      (sizeof(MVAL) / sizeof(MVAL[0]))
#define NN      (sizeof(NVAL) / sizeof(NVAL[0]))
#define NNS     (sizeof(NSVAL) / sizeof(NSVAL[0]))
#define NNB     (sizeof(NBVAL) / sizeof(NBVAL[0]))
#define NTYPES  6       /* Number of matrix types */
#define NTESTS  18      /* Number of tests per matrix */
#define THRESH  30.0f
#define MMAX    50      /* Maximum M dimension */
#define NMAX    50      /* Maximum N dimension */
#define NSMAX   15      /* Maximum NRHS */
#define SMLSIZ  25      /* Size threshold for divide-conquer */

/* Routines under test */
extern void sgels(const char* trans, const int m, const int n, const int nrhs,
                  f32* A, const int lda, f32* B, const int ldb,
                  f32* work, const int lwork, int* info);
extern void sgelst(const char* trans, const int m, const int n, const int nrhs,
                   f32* A, const int lda, f32* B, const int ldb,
                   f32* work, const int lwork, int* info);
extern void sgelsy(const int m, const int n, const int nrhs,
                   f32* A, const int lda, f32* B, const int ldb,
                   int* jpvt, const f32 rcond, int* rank,
                   f32* work, const int lwork, int* info);
extern void sgelss(const int m, const int n, const int nrhs,
                   f32* A, const int lda, f32* B, const int ldb,
                   f32* S, const f32 rcond, int* rank,
                   f32* work, const int lwork, int* info);
extern void sgelsd(const int m, const int n, const int nrhs,
                   f32* A, const int lda, f32* B, const int ldb,
                   f32* S, const f32 rcond, int* rank,
                   f32* work, const int lwork, int* iwork, int* info);
extern void sgetsls(const char* trans, const int m, const int n, const int nrhs,
                    f32* A, const int lda, f32* B, const int ldb,
                    f32* work, const int lwork, int* info);

/* Verification routines */
extern void sqrt13(const int scale, const int m, const int n,
                   f32* A, const int lda, f32* norma,
                   uint64_t state[static 4]);
extern void sqrt16(const char* trans, const int m, const int n, const int nrhs,
                   const f32* A, const int lda,
                   const f32* X, const int ldx,
                   f32* B, const int ldb,
                   f32* rwork, f32* resid);
extern f32 sqrt17(const char* trans, const int iresid,
                     const int m, const int n, const int nrhs,
                     const f32* A, const int lda,
                     const f32* X, const int ldx,
                     const f32* B, const int ldb,
                     f32* C,
                     f32* work, const int lwork);
extern f32 sqrt14(const char* trans, const int m, const int n, const int nrhs,
                     const f32* A, const int lda, const f32* X, const int ldx,
                     f32* work, const int lwork);
extern f32 sqrt12(const int m, const int n, const f32* A, const int lda,
                     const f32* S, f32* work, const int lwork);
extern void sqrt15(const int scale, const int rksel,
                   const int m, const int n, const int nrhs,
                   f32* A, const int lda, f32* B, const int ldb,
                   f32* S, int* rank, f32* norma, f32* normb,
                   f32* work, const int lwork,
                   uint64_t state[static 4]);

/* Utilities */
extern void slacpy(const char* uplo, const int m, const int n,
                   const f32* A, const int lda, f32* B, const int ldb);
extern void slarnv(const int idist, uint64_t* iseed, const int n, f32* x);
extern f32 slamch(const char* cmach);

/**
 * Test parameters for a single test case.
 */
typedef struct {
    int m;
    int n;
    int nrhs;
    int irank;   /* 1=full-rank, 2=rank-deficient */
    int iscale;  /* 1=normal, 2=scaled up, 3=scaled down */
    int inb;     /* Index into NBVAL[] */
    char name[80];
} ddrvls_params_t;

/**
 * Workspace for test execution - shared across all tests via group setup.
 */
typedef struct {
    f32* A;       /* Matrix (MMAX x NMAX) */
    f32* COPYA;   /* Copy of A (MMAX x NMAX) */
    f32* B;       /* RHS matrix (max(MMAX,NMAX) x NSMAX) */
    f32* COPYB;   /* Copy of B */
    f32* C;       /* Workspace for residual computation */
    f32* S;       /* Singular values (min(MMAX,NMAX)) */
    f32* COPYS;   /* Copy of S for rank-deficient tests */
    f32* WORK;    /* General workspace */
    int* IWORK;      /* Integer workspace */
    int* JPVT;       /* Pivot array for SGELSY */
    int lwork;       /* Workspace size */
    int liwork;      /* Integer workspace size */
    int lwlsy;       /* Workspace size for SGELSY */
} ddrvls_workspace_t;

static ddrvls_workspace_t* g_workspace = NULL;

/**
 * Group setup - allocate workspace once for all tests.
 * Workspace computation follows ddrvls.f lines 299-400.
 */
static int group_setup(void** state)
{
    (void)state;
    g_workspace = malloc(sizeof(ddrvls_workspace_t));
    if (!g_workspace) return -1;

    int mmax = MMAX;
    int nmax = NMAX;
    int mnmax = (mmax > nmax) ? mmax : nmax;
    int minmn = (mmax < nmax) ? mmax : nmax;

    /* Compute workspace sizes (ddrvls.f lines 327-330) */
    int lwork = 1;
    int temp;

    /* (M+N)*NRHS */
    temp = (mmax + nmax) * NSMAX;
    if (temp > lwork) lwork = temp;

    /* (N+NRHS)*(M+2) */
    temp = (nmax + NSMAX) * (mmax + 2);
    if (temp > lwork) lwork = temp;

    /* (M+NRHS)*(N+2) */
    temp = (mmax + NSMAX) * (nmax + 2);
    if (temp > lwork) lwork = temp;

    /* MAX(M+MNMIN, NRHS*MNMIN, 2*N+M) */
    int t1 = mmax + minmn;
    int t2 = NSMAX * minmn;
    int t3 = 2 * nmax + mmax;
    temp = t1;
    if (t2 > temp) temp = t2;
    if (t3 > temp) temp = t3;
    if (temp > lwork) lwork = temp;

    /* MAX(M*N + 4*MNMIN + MAX(M,N), M*N + 2*MNMIN + 4*N) */
    t1 = mmax * nmax + 4 * minmn + mnmax;
    t2 = mmax * nmax + 2 * minmn + 4 * nmax;
    temp = (t1 > t2) ? t1 : t2;
    if (temp > lwork) lwork = temp;

    /* LIWORK for SGELSY and SGELSD (ddrvls.f line 331, 385) */
    int liwork = 1;
    int nlvl = (int)(logf((f32)minmn / (f32)(SMLSIZ + 1)) / logf(2.0f)) + 1;
    if (nlvl < 1) nlvl = 1;
    int liwork_gelsd = 3 * minmn * nlvl + 11 * minmn;
    if (liwork_gelsd > liwork) liwork = liwork_gelsd;
    if (nmax > liwork) liwork = nmax;

    /* Add workspace for routines */
    int nb = 20;
    int lwork_gels = minmn + (minmn > NSMAX ? minmn : NSMAX);
    if (lwork_gels > lwork) lwork = lwork_gels;

    int lwork_gelst = minmn + (minmn > NSMAX ? minmn : NSMAX);
    if (lwork_gelst > lwork) lwork = lwork_gelst;

    int lwork_getsls = mmax * nmax + 4 * minmn;
    if (lwork_getsls > lwork) lwork = lwork_getsls;

    int lwork_gelsy = mmax * nmax + (nmax + 1) * (NSMAX + 1) + nb * (nmax + 1) + 2 * nmax;
    if (lwork_gelsy > lwork) lwork = lwork_gelsy;

    int t = 2 * minmn;
    if (mnmax > t) t = mnmax;
    if (NSMAX > t) t = NSMAX;
    int lwork_gelss = 3 * minmn + t;
    if (lwork_gelss > lwork) lwork = lwork_gelss;

    int lwork_gelsd = 12 * minmn + 2 * minmn * SMLSIZ + 8 * minmn * nlvl +
                      minmn * NSMAX + (SMLSIZ + 1) * (SMLSIZ + 1);
    if (lwork_gelsd > lwork) lwork = lwork_gelsd;

    int lwork_extra = 3 * minmn * minmn + mmax * mmax + nmax * NSMAX;
    lwork += lwork_extra;

    g_workspace->lwork = lwork;
    g_workspace->lwlsy = lwork;
    g_workspace->liwork = liwork;

    g_workspace->A = calloc(mmax * nmax, sizeof(f32));
    g_workspace->COPYA = calloc(mmax * nmax, sizeof(f32));
    g_workspace->B = calloc(mnmax * NSMAX, sizeof(f32));
    g_workspace->COPYB = calloc(mnmax * NSMAX, sizeof(f32));
    g_workspace->C = calloc(mnmax * NSMAX, sizeof(f32));
    g_workspace->S = calloc(minmn, sizeof(f32));
    g_workspace->COPYS = calloc(minmn, sizeof(f32));
    g_workspace->WORK = calloc(lwork, sizeof(f32));
    g_workspace->IWORK = calloc(liwork, sizeof(int));
    g_workspace->JPVT = calloc(nmax, sizeof(int));

    if (!g_workspace->A || !g_workspace->COPYA ||
        !g_workspace->B || !g_workspace->COPYB || !g_workspace->C ||
        !g_workspace->S || !g_workspace->COPYS ||
        !g_workspace->WORK || !g_workspace->IWORK || !g_workspace->JPVT) {
        return -1;
    }

    /* Set constant xlaenv values */
    xlaenv(2, 2);
    xlaenv(9, SMLSIZ);

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
        free(g_workspace->C);
        free(g_workspace->S);
        free(g_workspace->COPYS);
        free(g_workspace->WORK);
        free(g_workspace->IWORK);
        free(g_workspace->JPVT);
        free(g_workspace);
        g_workspace = NULL;
    }
    return 0;
}

/**
 * Run full-rank tests: SGELS (tests 1-2), SGELST (tests 3-4), SGETSLS (tests 5-6)
 */
static void run_fullrank_tests(int m, int n, int nrhs, int iscale, int nb, int nx)
{
    ddrvls_workspace_t* ws = g_workspace;
    f32 result[6];
    f32 norma;
    int info, lda, ldb;
    const f32 ONE = 1.0f;
    const f32 ZERO = 0.0f;
    char ctx[128];
    uint64_t rng_state[4];

    if (m == 0 || n == 0) return;

    lda = (m > 1) ? m : 1;
    ldb = (m > n) ? m : n;
    if (ldb < 1) ldb = 1;

    xlaenv(1, nb);
    xlaenv(3, nx);

    rng_seed(rng_state, 1988 + m * 100 + n * 10 + iscale + nb);
    sqrt13(iscale, m, n, ws->COPYA, lda, &norma, rng_state);

    /* SGELS tests 1-2 */
    for (int itran = 1; itran <= 2; itran++) {
        const char* trans = (itran == 1) ? "N" : "T";
        int nrows = (itran == 1) ? m : n;
        int ncols = (itran == 1) ? n : m;
        int ldwork = (ncols > 1) ? ncols : 1;

        if (ncols > 0) {
            rng_seed(rng_state, 1989 + m * 100 + n * 10 + iscale + itran + nb);
            for (int j = 0; j < nrhs; j++) {
                for (int i = 0; i < ncols; i++) {
                    ws->WORK[i + j * ldwork] = ((f32)(i + j * ncols + 1)) / (ncols * nrhs + 1);
                }
            }
            cblas_sscal(ncols * nrhs, ONE / (f32)ncols, ws->WORK, 1);
        }
        cblas_sgemm(CblasColMajor,
                    (itran == 1) ? CblasNoTrans : CblasTrans,
                    CblasNoTrans,
                    nrows, nrhs, ncols, ONE, ws->COPYA, lda,
                    ws->WORK, ldwork, ZERO, ws->B, ldb);
        slacpy("F", nrows, nrhs, ws->B, ldb, ws->COPYB, ldb);

        if (m > 0 && n > 0) {
            slacpy("F", m, n, ws->COPYA, lda, ws->A, lda);
            slacpy("F", nrows, nrhs, ws->COPYB, ldb, ws->B, ldb);
        }
        sgels(trans, m, n, nrhs, ws->A, lda, ws->B, ldb, ws->WORK, ws->lwork, &info);

        if (nrows > 0 && nrhs > 0) {
            slacpy("F", nrows, nrhs, ws->COPYB, ldb, ws->C, ldb);
        }
        sqrt16(trans, m, n, nrhs, ws->COPYA, lda, ws->B, ldb, ws->C, ldb,
               ws->WORK, &result[itran - 1]);

        snprintf(ctx, sizeof(ctx), "SGELS trans=%s m=%d n=%d nrhs=%d nb=%d scale=%d",
                 trans, m, n, nrhs, nb, iscale);
        set_test_context(ctx);
        assert_residual_below(result[itran - 1], THRESH);
    }

    /* SGELST tests 3-4 */
    rng_seed(rng_state, 2988 + m * 100 + n * 10 + iscale + nb);
    sqrt13(iscale, m, n, ws->COPYA, lda, &norma, rng_state);

    for (int itran = 1; itran <= 2; itran++) {
        const char* trans = (itran == 1) ? "N" : "T";
        int nrows = (itran == 1) ? m : n;
        int ncols = (itran == 1) ? n : m;
        int ldwork = (ncols > 1) ? ncols : 1;

        if (ncols > 0) {
            rng_seed(rng_state, 2989 + m * 100 + n * 10 + iscale + itran + nb);
            for (int j = 0; j < nrhs; j++) {
                for (int i = 0; i < ncols; i++) {
                    ws->WORK[i + j * ldwork] = ((f32)(i + j * ncols + 1)) / (ncols * nrhs + 1);
                }
            }
            cblas_sscal(ncols * nrhs, ONE / (f32)ncols, ws->WORK, 1);
        }
        cblas_sgemm(CblasColMajor,
                    (itran == 1) ? CblasNoTrans : CblasTrans,
                    CblasNoTrans,
                    nrows, nrhs, ncols, ONE, ws->COPYA, lda,
                    ws->WORK, ldwork, ZERO, ws->B, ldb);
        slacpy("F", nrows, nrhs, ws->B, ldb, ws->COPYB, ldb);

        if (m > 0 && n > 0) {
            slacpy("F", m, n, ws->COPYA, lda, ws->A, lda);
            slacpy("F", nrows, nrhs, ws->COPYB, ldb, ws->B, ldb);
        }
        sgelst(trans, m, n, nrhs, ws->A, lda, ws->B, ldb, ws->WORK, ws->lwork, &info);

        if (nrows > 0 && nrhs > 0) {
            slacpy("F", nrows, nrhs, ws->COPYB, ldb, ws->C, ldb);
        }
        sqrt16(trans, m, n, nrhs, ws->COPYA, lda, ws->B, ldb, ws->C, ldb,
               ws->WORK, &result[2 + itran - 1]);

        snprintf(ctx, sizeof(ctx), "SGELST trans=%s m=%d n=%d nrhs=%d nb=%d scale=%d",
                 trans, m, n, nrhs, nb, iscale);
        set_test_context(ctx);
        assert_residual_below(result[2 + itran - 1], THRESH);
    }

    /* SGETSLS tests 5-6 */
    rng_seed(rng_state, 3988 + m * 100 + n * 10 + iscale + nb);
    sqrt13(iscale, m, n, ws->COPYA, lda, &norma, rng_state);

    for (int itran = 1; itran <= 2; itran++) {
        const char* trans = (itran == 1) ? "N" : "T";
        int nrows = (itran == 1) ? m : n;
        int ncols = (itran == 1) ? n : m;
        int ldwork = (ncols > 1) ? ncols : 1;

        if (ncols > 0) {
            rng_seed(rng_state, 3989 + m * 100 + n * 10 + iscale + itran + nb);
            for (int j = 0; j < nrhs; j++) {
                for (int i = 0; i < ncols; i++) {
                    ws->WORK[i + j * ldwork] = ((f32)(i + j * ncols + 1)) / (ncols * nrhs + 1);
                }
            }
            cblas_sscal(ncols * nrhs, ONE / (f32)ncols, ws->WORK, 1);
        }
        cblas_sgemm(CblasColMajor,
                    (itran == 1) ? CblasNoTrans : CblasTrans,
                    CblasNoTrans,
                    nrows, nrhs, ncols, ONE, ws->COPYA, lda,
                    ws->WORK, ldwork, ZERO, ws->B, ldb);
        slacpy("F", nrows, nrhs, ws->B, ldb, ws->COPYB, ldb);

        if (m > 0 && n > 0) {
            slacpy("F", m, n, ws->COPYA, lda, ws->A, lda);
            slacpy("F", nrows, nrhs, ws->COPYB, ldb, ws->B, ldb);
        }
        sgetsls(trans, m, n, nrhs, ws->A, lda, ws->B, ldb, ws->WORK, ws->lwork, &info);

        if (nrows > 0 && nrhs > 0) {
            slacpy("F", nrows, nrhs, ws->COPYB, ldb, ws->C, ldb);
        }
        sqrt16(trans, m, n, nrhs, ws->COPYA, lda, ws->B, ldb, ws->C, ldb,
               ws->WORK, &result[4 + itran - 1]);

        snprintf(ctx, sizeof(ctx), "SGETSLS trans=%s m=%d n=%d nrhs=%d nb=%d scale=%d",
                 trans, m, n, nrhs, nb, iscale);
        set_test_context(ctx);
        assert_residual_below(result[4 + itran - 1], THRESH);
    }

    clear_test_context();
}

/**
 * Run rank-deficient tests: SGELSY (tests 7-10), SGELSS (tests 11-14), SGELSD (tests 15-18)
 */
static void run_rankdef_tests(int m, int n, int nrhs, int iscale, int irank, int nb, int nx)
{
    ddrvls_workspace_t* ws = g_workspace;
    f32 result[NTESTS];
    f32 norma, normb, rcond;
    int info, lda, ldb, crank, rank;
    f32 eps;
    int mnmin = (m < n) ? m : n;
    const f32 ONE = 1.0f;
    const f32 ZERO = 0.0f;
    char ctx[128];
    uint64_t rng_state[4];

    if (m == 0 || n == 0) return;

    lda = (m > 1) ? m : 1;
    ldb = (m > n) ? m : n;
    if (ldb < 1) ldb = 1;
    int ldwork = (m > 1) ? m : 1;

    eps = slamch("E");
    rcond = sqrtf(eps) - (sqrtf(eps) - eps) / 2.0f;

    for (int k = 0; k < NTESTS; k++) {
        result[k] = ZERO;
    }

    xlaenv(1, nb);
    xlaenv(3, nx);

    rng_seed(rng_state, 4988 + m * 100 + n * 10 + iscale + irank * 1000 + nb);
    sqrt15(iscale, irank, m, n, nrhs, ws->COPYA, lda, ws->COPYB, ldb,
           ws->COPYS, &rank, &norma, &normb, ws->WORK, ws->lwork, rng_state);

    int itype = (irank - 1) * 3 + iscale;

    /* SGELSY (Tests 7-10) */
    for (int j = 0; j < n; j++) {
        ws->JPVT[j] = 0;
    }

    slacpy("F", m, n, ws->COPYA, lda, ws->A, lda);
    slacpy("F", m, nrhs, ws->COPYB, ldb, ws->B, ldb);

    sgelsy(m, n, nrhs, ws->A, lda, ws->B, ldb, ws->JPVT, rcond, &crank,
           ws->WORK, ws->lwlsy, &info);

    result[6] = sqrt12(crank, crank, ws->A, lda, ws->COPYS, ws->WORK, ws->lwork);

    slacpy("F", m, nrhs, ws->COPYB, ldb, ws->WORK, ldwork);
    sqrt16("N", m, n, nrhs, ws->COPYA, lda, ws->B, ldb, ws->WORK, ldwork,
           &ws->WORK[m * nrhs], &result[7]);

    if (m > crank) {
        result[8] = sqrt17("N", 1, m, n, nrhs, ws->COPYA, lda, ws->B, ldb,
                           ws->COPYB, ldb, ws->C, ws->WORK, ws->lwork);
    }

    if (n > crank) {
        result[9] = sqrt14("N", m, n, nrhs, ws->COPYA, lda, ws->B, ldb,
                           ws->WORK, ws->lwork);
    }

    snprintf(ctx, sizeof(ctx), "SGELSY m=%d n=%d nrhs=%d type=%d nb=%d", m, n, nrhs, itype, nb);
    set_test_context(ctx);
    assert_residual_below(result[6], THRESH);
    assert_residual_below(result[7], THRESH);
    if (m > crank) assert_residual_below(result[8], THRESH);
    if (n > crank) assert_residual_below(result[9], THRESH);

    /* SGELSS (Tests 11-14) */
    slacpy("F", m, n, ws->COPYA, lda, ws->A, lda);
    slacpy("F", m, nrhs, ws->COPYB, ldb, ws->B, ldb);

    sgelss(m, n, nrhs, ws->A, lda, ws->B, ldb, ws->S, rcond, &crank,
           ws->WORK, ws->lwork, &info);

    if (rank > 0) {
        cblas_saxpy(mnmin, -ONE, ws->COPYS, 1, ws->S, 1);
        result[10] = cblas_sasum(mnmin, ws->S, 1) /
                     cblas_sasum(mnmin, ws->COPYS, 1) /
                     (eps * (f32)mnmin);
    }

    slacpy("F", m, nrhs, ws->COPYB, ldb, ws->WORK, ldwork);
    sqrt16("N", m, n, nrhs, ws->COPYA, lda, ws->B, ldb, ws->WORK, ldwork,
           &ws->WORK[m * nrhs], &result[11]);

    if (m > crank) {
        result[12] = sqrt17("N", 1, m, n, nrhs, ws->COPYA, lda, ws->B, ldb,
                           ws->COPYB, ldb, ws->C, ws->WORK, ws->lwork);
    }

    if (n > crank) {
        result[13] = sqrt14("N", m, n, nrhs, ws->COPYA, lda, ws->B, ldb,
                           ws->WORK, ws->lwork);
    }

    snprintf(ctx, sizeof(ctx), "SGELSS m=%d n=%d nrhs=%d type=%d nb=%d", m, n, nrhs, itype, nb);
    set_test_context(ctx);
    assert_residual_below(result[10], THRESH);
    assert_residual_below(result[11], THRESH);
    if (m > crank) assert_residual_below(result[12], THRESH);
    if (n > crank) assert_residual_below(result[13], THRESH);

    /* SGELSD (Tests 15-18) */
    for (int j = 0; j < n; j++) {
        ws->IWORK[j] = 0;
    }

    slacpy("F", m, n, ws->COPYA, lda, ws->A, lda);
    slacpy("F", m, nrhs, ws->COPYB, ldb, ws->B, ldb);

    sgelsd(m, n, nrhs, ws->A, lda, ws->B, ldb, ws->S, rcond, &crank,
           ws->WORK, ws->lwork, ws->IWORK, &info);

    /* Regenerate COPYS since SGELSS modified S */
    rng_seed(rng_state, 4988 + m * 100 + n * 10 + iscale + irank * 1000 + nb);
    sqrt15(iscale, irank, m, n, nrhs, ws->COPYA, lda, ws->COPYB, ldb,
           ws->COPYS, &rank, &norma, &normb, ws->WORK, ws->lwork, rng_state);

    if (rank > 0) {
        cblas_saxpy(mnmin, -ONE, ws->COPYS, 1, ws->S, 1);
        result[14] = cblas_sasum(mnmin, ws->S, 1) /
                     cblas_sasum(mnmin, ws->COPYS, 1) /
                     (eps * (f32)mnmin);
    }

    slacpy("F", m, nrhs, ws->COPYB, ldb, ws->WORK, ldwork);
    sqrt16("N", m, n, nrhs, ws->COPYA, lda, ws->B, ldb, ws->WORK, ldwork,
           &ws->WORK[m * nrhs], &result[15]);

    if (m > crank) {
        result[16] = sqrt17("N", 1, m, n, nrhs, ws->COPYA, lda, ws->B, ldb,
                           ws->COPYB, ldb, ws->C, ws->WORK, ws->lwork);
    }

    if (n > crank) {
        result[17] = sqrt14("N", m, n, nrhs, ws->COPYA, lda, ws->B, ldb,
                           ws->WORK, ws->lwork);
    }

    snprintf(ctx, sizeof(ctx), "SGELSD m=%d n=%d nrhs=%d type=%d nb=%d", m, n, nrhs, itype, nb);
    set_test_context(ctx);
    assert_residual_below(result[14], THRESH);
    assert_residual_below(result[15], THRESH);
    if (m > crank) assert_residual_below(result[16], THRESH);
    if (n > crank) assert_residual_below(result[17], THRESH);

    clear_test_context();
}

/**
 * Run all tests for a single parameter combination.
 */
static void run_ddrvls_single(ddrvls_params_t* p)
{
    int nb = NBVAL[p->inb];
    int nx = NXVAL[p->inb];

    if (p->irank == 1) {
        /* Full-rank tests: SGELS, SGELST, SGETSLS */
        run_fullrank_tests(p->m, p->n, p->nrhs, p->iscale, nb, nx);
    }

    /* Rank-deficient tests: SGELSY, SGELSS, SGELSD (run for all irank values) */
    run_rankdef_tests(p->m, p->n, p->nrhs, p->iscale, p->irank, nb, nx);
}

/**
 * CMocka test function - dispatches to run_ddrvls_single based on prestate.
 */
static void test_ddrvls_case(void** state)
{
    ddrvls_params_t* params = *state;
    run_ddrvls_single(params);
}

/*
 * Generate all parameter combinations.
 * Total: NM * NN * NNS * 2(irank) * 3(iscale) * NNB
 *      = 7 * 7 * 3 * 2 * 3 * 5 = 4410 tests
 */

/* Maximum number of test cases */
#define MAX_TESTS (NM * NN * NNS * 2 * 3 * NNB)

static ddrvls_params_t g_params[MAX_TESTS];
static struct CMUnitTest g_tests[MAX_TESTS];
static int g_num_tests = 0;

/**
 * Build the test array with all parameter combinations.
 */
static void build_test_array(void)
{
    g_num_tests = 0;

    for (int im = 0; im < (int)NM; im++) {
        int m = MVAL[im];

        for (int in = 0; in < (int)NN; in++) {
            int n = NVAL[in];

            for (int ins = 0; ins < (int)NNS; ins++) {
                int nrhs = NSVAL[ins];

                for (int irank = 1; irank <= 2; irank++) {

                    for (int iscale = 1; iscale <= 3; iscale++) {
                        int itype = (irank - 1) * 3 + iscale;

                        for (int inb = 0; inb < (int)NNB; inb++) {
                            int nb = NBVAL[inb];

                            /* Store parameters */
                            ddrvls_params_t* p = &g_params[g_num_tests];
                            p->m = m;
                            p->n = n;
                            p->nrhs = nrhs;
                            p->irank = irank;
                            p->iscale = iscale;
                            p->inb = inb;
                            snprintf(p->name, sizeof(p->name),
                                     "ddrvls_m%d_n%d_nrhs%d_type%d_nb%d_%d",
                                     m, n, nrhs, itype, nb, inb);

                            /* Create CMocka test entry */
                            g_tests[g_num_tests].name = p->name;
                            g_tests[g_num_tests].test_func = test_ddrvls_case;
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
    /* Build all test cases */
    build_test_array();

    /* Run all tests with shared workspace.
     * We use _cmocka_run_group_tests directly because the test array
     * is built dynamically and the standard macro uses sizeof() which
     * only works for compile-time array sizes. */
    return _cmocka_run_group_tests("ddrvls", g_tests, g_num_tests,
                                   group_setup, group_teardown);
}
