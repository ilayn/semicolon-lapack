/**
 * @file test_ddrvbd.c
 * @brief Comprehensive SVD test driver - port of LAPACK TESTING/EIG/ddrvbd.f
 *
 * Tests all 6 SVD drivers:
 * - DGESVD:  QR iteration SVD
 * - DGESDD:  Divide-and-conquer SVD
 * - DGESVDX: SVD with range selection (bisection)
 * - DGESVDQ: QR-preconditioned SVD
 * - DGESVJ:  Jacobi SVD
 * - DGEJSV:  Preconditioned Jacobi SVD (high accuracy)
 *
 * Each (m, n, type) combination is registered as a separate CMocka test,
 * providing pytest-like parameterized test behavior with clear failure isolation.
 *
 * Test ratios (39 total):
 *   DGESVD  (1-7):   Full and partial SVD
 *   DGESDD  (8-14):  Full and partial SVD
 *   DGESVJ  (15-18): Full SVD (M >= N only)
 *   DGEJSV  (19-22): Full SVD (M >= N only)
 *   DGESVDX (23-35): Full and range-selected SVD
 *   DGESVDQ (36-39): Full SVD (M >= N only)
 *
 * Matrix types (5 total):
 *   1. Zero matrix
 *   2. Identity matrix
 *   3. Random with evenly spaced singular values in [ULP, 1]
 *   4. Same as 3, scaled near underflow
 *   5. Same as 3, scaled near overflow
 */

#include "test_harness.h"
#include "testutils/verify.h"
#include <cblas.h>

/* Test threshold from LAPACK svd.in (line 10) */
#define THRESH 50.0

/* Number of matrix types */
#define NTYPES 5

/* Number of test results */
#define NRESULTS 39

/* Job option strings for partial SVD tests (from ddrvbd.f lines 439-441) */
static const char* CJOB = "NOSA";   /* 'N', 'O', 'S', 'A' for DGESVD/DGESDD */
static const char* CJOBV = "NV";    /* 'N', 'V' for DGESVDX */

/* Test dimension pairs from LAPACK's svd.in
 * These are specific (M,N) pairs, not all combinations.
 * LAPACK tests: 0 0 0 1 1 1 2 2 3 3 3 10 10 16 16 30 30 40 40 (M values)
 *               0 1 3 0 1 2 0 1 0 1 3 10 16 10 16 30 40 30 40 (N values)
 */
typedef struct { int m; int n; } dim_pair_t;
static const dim_pair_t DIM_PAIRS[] = {
    {0, 0}, {0, 1}, {0, 3},     /* M=0 cases */
    {1, 0}, {1, 1}, {1, 2},     /* M=1 cases */
    {2, 0}, {2, 1},             /* M=2 cases */
    {3, 0}, {3, 1}, {3, 3},     /* M=3 cases */
    {10, 10}, {10, 16},         /* M=10 cases */
    {16, 10}, {16, 16},         /* M=16 cases */
    {30, 30}, {30, 40},         /* M=30 cases */
    {40, 30}, {40, 40}          /* M=40 cases */
};
#define NDIM_PAIRS (sizeof(DIM_PAIRS) / sizeof(DIM_PAIRS[0]))

/* SVD driver declarations */
extern void dgesvd(const char* jobu, const char* jobvt, const int m, const int n,
                   double* A, const int lda, double* S,
                   double* U, const int ldu, double* VT, const int ldvt,
                   double* work, const int lwork, int* info);

extern void dgesdd(const char* jobz, const int m, const int n,
                   double* A, const int lda, double* S,
                   double* U, const int ldu, double* VT, const int ldvt,
                   double* work, const int lwork, int* iwork, int* info);

extern void dgesvdx(const char* jobu, const char* jobvt, const char* range,
                    const int m, const int n, double* A, const int lda,
                    const double vl, const double vu, const int il, const int iu,
                    int* ns, double* S, double* U, const int ldu,
                    double* VT, const int ldvt, double* work, const int lwork,
                    int* iwork, int* info);

extern void dgesvdq(const char* joba, const char* jobp, const char* jobr,
                    const char* jobu, const char* jobv,
                    const int m, const int n, double* A, const int lda,
                    double* S, double* U, const int ldu, double* V, const int ldv,
                    int* numrank, int* iwork, const int liwork,
                    double* work, const int lwork, double* rwork, const int lrwork,
                    int* info);

extern void dgesvj(const char* joba, const char* jobu, const char* jobv,
                   const int m, const int n, double* A, const int lda,
                   double* SVA, const int mv, double* V, const int ldv,
                   double* work, const int lwork, int* info);

extern void dgejsv(const char* joba, const char* jobu, const char* jobv,
                   const char* jobr, const char* jobt, const char* jobp,
                   const int m, const int n, double* A, const int lda,
                   double* SVA, double* U, const int ldu, double* V, const int ldv,
                   double* work, const int lwork, int* iwork, int* info);

/* Utility routines */
extern double dlamch(const char* cmach);
extern double dlange(const char* norm, const int m, const int n,
                     const double* A, const int lda, double* work);
extern void dlacpy(const char* uplo, const int m, const int n,
                   const double* A, const int lda, double* B, const int ldb);
extern void dlaset(const char* uplo, const int m, const int n,
                   const double alpha, const double beta, double* A, const int lda);
extern void dlascl(const char* type, const int kl, const int ku,
                   const double cfrom, const double cto, const int m, const int n,
                   double* A, const int lda, int* info);

/* Matrix generation and verification routines from verify.h */

/**
 * Test parameters for a single test case.
 */
typedef struct {
    int m;
    int n;
    int itype;
    char name[64];
} ddrvbd_params_t;

/*
 * Workspace structure for all SVD tests.
 * Allocated once at maximum dimensions, reused for all size combinations.
 */
typedef struct {
    int mmax, nmax, mnmax;

    /* Matrices */
    double* A;      /* m x n working copy */
    double* ASAV;   /* m x n saved copy */
    double* U;      /* m x m left singular vectors */
    double* USAV;   /* m x m saved U */
    double* VT;     /* n x n right singular vectors */
    double* VTSAV;  /* n x n saved VT */

    /* Singular values */
    double* S;      /* min(m,n) computed singular values */
    double* SSAV;   /* min(m,n) saved singular values */

    /* Work arrays */
    double* work;   /* general workspace */
    double* rwork;  /* real workspace for dgesvdq */
    int* iwork;     /* integer workspace */
    int lwork;      /* workspace size */
    int liwork;     /* integer workspace size */
    int lrwork;     /* real workspace size */

    /* Test results */
    double result[NRESULTS];

    /* RNG seed */
    uint64_t seed;
} ddrvbd_workspace_t;

/* Global workspace pointer */
static ddrvbd_workspace_t* g_ws = NULL;

/* ===== RNG Implementation (xoshiro256+) ===== */

static uint64_t rng_state[4];

static void rng_seed(uint64_t s)
{
    for (int i = 0; i < 4; i++) {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        rng_state[i] = s;
    }
}

static double rng_uniform(void)
{
    uint64_t s = rng_state[1] * 5;
    uint64_t r = ((s << 7) | (s >> 57)) * 9;
    uint64_t t = rng_state[1] << 17;
    rng_state[2] ^= rng_state[0];
    rng_state[3] ^= rng_state[1];
    rng_state[1] ^= rng_state[2];
    rng_state[0] ^= rng_state[3];
    rng_state[2] ^= t;
    rng_state[3] = (rng_state[3] << 45) | (rng_state[3] >> 19);
    return (double)(r >> 11) * 0x1.0p-53;
}

/* ===== Helper Functions ===== */

/**
 * Compute maximum workspace needed for all SVD routines.
 */
static int compute_lwork(int mmax, int nmax)
{
    int mnmax = (mmax < nmax) ? mmax : nmax;
    int mxmax = (mmax > nmax) ? mmax : nmax;

    /* From ddrvbd.f:
     * LWORK = MAX(3*MN + MX, 5*MN - 4) + 2*MN^2
     * Plus extra for DGESVDQ
     */
    int lwork1 = 3 * mnmax + mxmax;
    int lwork2 = 5 * mnmax - 4;
    int lwork = (lwork1 > lwork2) ? lwork1 : lwork2;
    lwork += 2 * mnmax * mnmax;

    /* Extra for DGESVDQ */
    int lwork_q = 5 * mnmax * mnmax + 9 * mnmax + mxmax;
    if (lwork_q > lwork) lwork = lwork_q;

    /* Extra for DGEJSV */
    int lwork_j = 6 * mnmax * mnmax + 10 * mnmax + mxmax;
    if (lwork_j > lwork) lwork = lwork_j;

    /* Extra safety margin */
    lwork = lwork + 100;

    return lwork;
}

/**
 * Check that singular values are non-negative and decreasing.
 * Returns 0.0 if valid, 1/ULP if invalid.
 */
static double check_sv_order(const double* S, int n, double ulpinv)
{
    for (int i = 0; i < n; i++) {
        if (S[i] < 0.0) return ulpinv;
    }
    for (int i = 0; i < n - 1; i++) {
        if (S[i] < S[i + 1]) return ulpinv;
    }
    return 0.0;
}

/**
 * Compute ||A - U*diag(S)*VT|| / (||A|| * max(m,n) * ulp).
 *
 * For full SVD: U is m x minmn, S is minmn, VT is minmn x n.
 * Computes: temp = U * diag(S) * VT, then resid = ||A - temp||.
 */
static double svd_residual(int m, int n, int ku,
                           const double* A, int lda,
                           const double* U, int ldu,
                           const double* S,
                           const double* VT, int ldvt,
                           double* work)
{
    /* ku = number of columns in U (economy SVD), also rows in VT
     * For full SVD, ku = min(m,n)
     */
    double ulp = dlamch("P");
    int maxmn = (m > n) ? m : n;

    /* Compute ||A|| */
    double anorm = dlange("F", m, n, A, lda, NULL);
    if (anorm == 0.0) anorm = 1.0;

    /* Compute U * diag(S) in work[0 : m*ku-1] */
    /* Scale columns of U by S */
    double* US = work;
    for (int j = 0; j < ku; j++) {
        for (int i = 0; i < m; i++) {
            US[i + j * m] = U[i + j * ldu] * S[j];
        }
    }

    /* Compute (U*S) * VT in work[m*ku : m*ku + m*n - 1] */
    double* USVT = &work[m * ku];
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                m, n, ku, 1.0, US, m, VT, ldvt, 0.0, USVT, m);

    /* Compute A - U*S*VT */
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            USVT[i + j * m] = A[i + j * lda] - USVT[i + j * m];
        }
    }

    /* Compute ||A - U*S*VT|| */
    double resid = dlange("F", m, n, USVT, m, NULL);

    return resid / (anorm * maxmn * ulp);
}

/**
 * Generate test matrix of specified type.
 *
 * Types (from ddrvbd.f):
 *   1: Zero matrix
 *   2: Identity matrix (diagonal of 1s)
 *   3: Random U*D*V with evenly spaced D in [ULP, 1]
 *   4: Same as 3, scaled near underflow
 *   5: Same as 3, scaled near overflow
 */
static void generate_test_matrix(int itype, int m, int n, double* A, int lda,
                                 double* S, double* work, uint64_t seed, int* info)
{
    double ulp = dlamch("P");
    double unfl = dlamch("S");
    double ovfl = 1.0 / unfl;
    int mnmin = (m < n) ? m : n;

    *info = 0;

    if (m == 0 || n == 0) {
        return;
    }

    if (itype == 1) {
        /* Type 1: Zero matrix */
        dlaset("F", m, n, 0.0, 0.0, A, lda);
        for (int i = 0; i < mnmin; i++) S[i] = 0.0;
    }
    else if (itype == 2) {
        /* Type 2: Identity matrix */
        dlaset("F", m, n, 0.0, 1.0, A, lda);
        for (int i = 0; i < mnmin; i++) S[i] = 1.0;
    }
    else if (itype >= 3 && itype <= 5) {
        /* Types 3-5: Random matrix with controlled singular values */

        /* Generate singular values: evenly spaced from 1 to 1/cond */
        /* mode=4 in dlatms gives evenly spaced SV from 1 to 1/cond */
        /* LAPACK uses cond = mnmin (see ddrvbd.f line 553) */
        double cond = (double)mnmin;
        double anorm = 1.0;

        if (itype == 4) {
            /* Scale near underflow */
            anorm = unfl / ulp;
        }
        else if (itype == 5) {
            /* Scale near overflow */
            anorm = ovfl * ulp;
        }

        /* Use dlatms to generate random matrix with specified singular values */
        dlatms(m, n, "U", seed, "N", S, 4, cond, anorm,
               m - 1, n - 1, "N", A, lda, work, info);

        if (*info != 0) {
            return;
        }

        /* Sort singular values in decreasing order for verification */
        /* dlatms with mode=4 gives SV from 1 to 1/cond */
        /* We need to know what values to expect */
        for (int i = 0; i < mnmin; i++) {
            double t = (double)i / (double)(mnmin > 1 ? mnmin - 1 : 1);
            S[i] = anorm * (1.0 - t * (1.0 - 1.0 / cond));
        }
    }
    else {
        *info = -1;  /* Invalid type */
    }
}

/* ===== DGESVD Tests (1-7) ===== */

/**
 * Test DGESVD with full and partial SVD.
 *
 * Tests 1-4: Full SVD (jobu='A', jobvt='A')
 *   1: |A - U*S*VT| / (|A| max(M,N) ulp)
 *   2: |I - U'*U| / (M ulp)
 *   3: |I - VT*VT'| / (N ulp)
 *   4: S non-negative decreasing
 *
 * Tests 5-7: Partial SVD comparison (max over all JOBU/JOBVT combinations)
 *   5: |U - Upartial| / (M ulp)
 *   6: |VT - VTpartial| / (N ulp)
 *   7: |S - Spartial| / (MNMIN ulp |S|)
 *
 * The partial SVD loop tests all 14 combinations of JOBU/JOBVT in {'N','O','S','A'},
 * skipping (JOBU='A',JOBVT='A') which is tested separately and (JOBU='O',JOBVT='O')
 * which is invalid (both can't overwrite A).
 *
 * @param lswork  Working workspace size (varies in IWS loop)
 */
static void test_dgesvd(int m, int n, const double* ASAV, int lda,
                        int lswork, ddrvbd_workspace_t* ws)
{
    int mnmin = (m < n) ? m : n;
    double ulp = dlamch("P");
    double ulpinv = 1.0 / ulp;
    double unfl = dlamch("S");
    int info;

    /* Initialize results to 0 */
    for (int i = 0; i < 7; i++) ws->result[i] = 0.0;

    if (m == 0 || n == 0) return;

    /* === Full SVD (jobu='A', jobvt='A') === */

    dlacpy("F", m, n, ASAV, lda, ws->A, lda);
    dgesvd("A", "A", m, n, ws->A, lda, ws->SSAV,
           ws->USAV, m, ws->VTSAV, n, ws->work, lswork, &info);

    if (info != 0) {
        ws->result[0] = ulpinv;
        return;
    }

    /* Test 1: Reconstruction */
    ws->result[0] = svd_residual(m, n, mnmin, ASAV, lda, ws->USAV, m,
                                 ws->SSAV, ws->VTSAV, n, ws->work);

    /* Test 2: U orthogonality */
    dort01("C", m, m, ws->USAV, m, ws->work, ws->lwork, &ws->result[1]);

    /* Test 3: VT orthogonality */
    dort01("R", n, n, ws->VTSAV, n, ws->work, ws->lwork, &ws->result[2]);

    /* Test 4: S ordering */
    ws->result[3] = check_sv_order(ws->SSAV, mnmin, ulpinv);

    /* === Partial SVD: test all 14 JOBU/JOBVT combinations (lines 622-686 in ddrvbd.f) === */
    ws->result[4] = 0.0;  /* max |U - Upartial| */
    ws->result[5] = 0.0;  /* max |VT - VTpartial| */
    ws->result[6] = 0.0;  /* max |S - Spartial| */

    for (int iju = 0; iju <= 3; iju++) {
        for (int ijvt = 0; ijvt <= 3; ijvt++) {
            /* Skip ('A','A') - tested above; skip ('O','O') - invalid */
            if ((iju == 3 && ijvt == 3) || (iju == 1 && ijvt == 1)) continue;

            char jobu[2] = {CJOB[iju], '\0'};
            char jobvt[2] = {CJOB[ijvt], '\0'};

            dlacpy("F", m, n, ASAV, lda, ws->A, lda);

            dgesvd(jobu, jobvt, m, n, ws->A, lda, ws->S,
                   ws->U, m, ws->VT, n, ws->work, lswork, &info);

            if (info != 0) continue;

            /* Compare U */
            double dif = 0.0;
            if (m > 0 && n > 0) {
                if (iju == 1) {
                    /* JOBU='O': U stored in A */
                    dort03("C", m, mnmin, m, mnmin, ws->USAV, m, ws->A, lda,
                           ws->work, ws->lwork, &dif, &info);
                } else if (iju == 2) {
                    /* JOBU='S': economy U in ws->U */
                    dort03("C", m, mnmin, m, mnmin, ws->USAV, m, ws->U, m,
                           ws->work, ws->lwork, &dif, &info);
                } else if (iju == 3) {
                    /* JOBU='A': full U in ws->U */
                    dort03("C", m, m, m, mnmin, ws->USAV, m, ws->U, m,
                           ws->work, ws->lwork, &dif, &info);
                }
            }
            if (dif > ws->result[4]) ws->result[4] = dif;

            /* Compare VT */
            dif = 0.0;
            if (m > 0 && n > 0) {
                if (ijvt == 1) {
                    /* JOBVT='O': VT stored in A */
                    dort03("R", n, mnmin, n, mnmin, ws->VTSAV, n, ws->A, lda,
                           ws->work, ws->lwork, &dif, &info);
                } else if (ijvt == 2) {
                    /* JOBVT='S': economy VT in ws->VT */
                    dort03("R", n, mnmin, n, mnmin, ws->VTSAV, n, ws->VT, n,
                           ws->work, ws->lwork, &dif, &info);
                } else if (ijvt == 3) {
                    /* JOBVT='A': full VT in ws->VT */
                    dort03("R", n, n, n, mnmin, ws->VTSAV, n, ws->VT, n,
                           ws->work, ws->lwork, &dif, &info);
                }
            }
            if (dif > ws->result[5]) ws->result[5] = dif;

            /* Compare S */
            dif = 0.0;
            double div = (mnmin * ulp * ws->S[0] > unfl) ? mnmin * ulp * ws->S[0] : unfl;
            for (int i = 0; i < mnmin - 1; i++) {
                if (ws->SSAV[i] < ws->SSAV[i + 1]) dif = ulpinv;
                if (ws->SSAV[i] < 0.0) dif = ulpinv;
                double d = fabs(ws->SSAV[i] - ws->S[i]) / div;
                if (d > dif) dif = d;
            }
            if (dif > ws->result[6]) ws->result[6] = dif;
        }
    }
}

/* ===== DGESDD Tests (8-14) ===== */

/**
 * Test DGESDD with full and partial SVD.
 * Same test structure as DGESVD, results stored in indices 7-13.
 *
 * Tests 8-11: Full SVD (jobz='A')
 * Tests 12-14: Partial SVD (max over jobz='N','O','S')
 *
 * @param lswork  Working workspace size (varies in IWS loop)
 */
static void test_dgesdd(int m, int n, const double* ASAV, int lda,
                        int lswork, ddrvbd_workspace_t* ws)
{
    int mnmin = (m < n) ? m : n;
    double ulp = dlamch("P");
    double ulpinv = 1.0 / ulp;
    double unfl = dlamch("S");
    int info;

    /* Initialize results to 0 */
    for (int i = 7; i < 14; i++) ws->result[i] = 0.0;

    if (m == 0 || n == 0) return;

    /* === Full SVD (jobz='A') === */
    dlacpy("F", m, n, ASAV, lda, ws->A, lda);

    dgesdd("A", m, n, ws->A, lda, ws->SSAV,
           ws->USAV, m, ws->VTSAV, n, ws->work, lswork, ws->iwork, &info);

    if (info != 0) {
        ws->result[7] = ulpinv;
        return;
    }

    /* Test 8: Reconstruction */
    ws->result[7] = svd_residual(m, n, mnmin, ASAV, lda, ws->USAV, m,
                                 ws->SSAV, ws->VTSAV, n, ws->work);

    /* Test 9: U orthogonality */
    dort01("C", m, m, ws->USAV, m, ws->work, ws->lwork, &ws->result[8]);

    /* Test 10: VT orthogonality */
    dort01("R", n, n, ws->VTSAV, n, ws->work, ws->lwork, &ws->result[9]);

    /* Test 11: S ordering */
    ws->result[10] = check_sv_order(ws->SSAV, mnmin, ulpinv);

    /* === Partial SVD: test jobz='N','O','S' (lines 735-797 in ddrvbd.f) === */
    ws->result[11] = 0.0;  /* max |U - Upartial| */
    ws->result[12] = 0.0;  /* max |VT - VTpartial| */
    ws->result[13] = 0.0;  /* max |S - Spartial| */

    for (int ijq = 0; ijq <= 2; ijq++) {
        char jobq[2] = {CJOB[ijq], '\0'};  /* 'N', 'O', 'S' */

        dlacpy("F", m, n, ASAV, lda, ws->A, lda);

        dgesdd(jobq, m, n, ws->A, lda, ws->S,
               ws->U, m, ws->VT, mnmin, ws->work, lswork, ws->iwork, &info);

        if (info != 0) continue;

        /* Compare U */
        double dif = 0.0;
        if (m > 0 && n > 0) {
            if (ijq == 1) {
                /* JOBZ='O': U or VT in A depending on M >= N */
                if (m >= n) {
                    dort03("C", m, mnmin, m, mnmin, ws->USAV, m, ws->A, lda,
                           ws->work, ws->lwork, &dif, &info);
                } else {
                    dort03("C", m, mnmin, m, mnmin, ws->USAV, m, ws->U, m,
                           ws->work, ws->lwork, &dif, &info);
                }
            } else if (ijq == 2) {
                /* JOBZ='S': economy U in ws->U */
                dort03("C", m, mnmin, m, mnmin, ws->USAV, m, ws->U, m,
                       ws->work, ws->lwork, &dif, &info);
            }
        }
        if (dif > ws->result[11]) ws->result[11] = dif;

        /* Compare VT */
        dif = 0.0;
        if (m > 0 && n > 0) {
            if (ijq == 1) {
                /* JOBZ='O': VT or U in A depending on M >= N */
                if (m >= n) {
                    dort03("R", n, mnmin, n, mnmin, ws->VTSAV, n, ws->VT, mnmin,
                           ws->work, ws->lwork, &dif, &info);
                } else {
                    dort03("R", n, mnmin, n, mnmin, ws->VTSAV, n, ws->A, lda,
                           ws->work, ws->lwork, &dif, &info);
                }
            } else if (ijq == 2) {
                /* JOBZ='S': economy VT in ws->VT */
                dort03("R", n, mnmin, n, mnmin, ws->VTSAV, n, ws->VT, mnmin,
                       ws->work, ws->lwork, &dif, &info);
            }
        }
        if (dif > ws->result[12]) ws->result[12] = dif;

        /* Compare S */
        dif = 0.0;
        double div = (mnmin * ulp * ws->S[0] > unfl) ? mnmin * ulp * ws->S[0] : unfl;
        for (int i = 0; i < mnmin - 1; i++) {
            if (ws->SSAV[i] < ws->SSAV[i + 1]) dif = ulpinv;
            if (ws->SSAV[i] < 0.0) dif = ulpinv;
            double d = fabs(ws->SSAV[i] - ws->S[i]) / div;
            if (d > dif) dif = d;
        }
        if (dif > ws->result[13]) ws->result[13] = dif;
    }
}

/* ===== DGESVJ Tests (15-18) ===== */

/**
 * Test DGESVJ (Jacobi SVD).
 * Only works for M >= N.
 * Returns V (not VT), so we need to transpose.
 *
 * @param lswork  Working workspace size (varies in IWS loop)
 */
static void test_dgesvj(int m, int n, const double* ASAV, int lda,
                        int lswork, ddrvbd_workspace_t* ws)
{
    double ulp = dlamch("P");
    double ulpinv = 1.0 / ulp;
    int info;

    /* Initialize results to 0 */
    for (int i = 14; i < 18; i++) ws->result[i] = 0.0;

    /* DGESVJ only works for M >= N */
    if (m < n || m == 0 || n == 0) return;

    int mnmin = n;  /* Since m >= n, mnmin = n */

    dlacpy("F", m, n, ASAV, lda, ws->USAV, m);

    dgesvj("G", "U", "V", m, n, ws->USAV, m, ws->SSAV, 0, ws->A, n,
           ws->work, lswork, &info);

    if (info != 0) {
        ws->result[14] = ulpinv;
        return;
    }

    /* Transpose V to VTSAV: ws->A currently holds V (n x n), need V^T in VTSAV */
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            ws->VTSAV[i + j * n] = ws->A[j + i * n];
        }
    }

    /* Test 15: Reconstruction using economy SVD (M×N U, N×N VT) */
    ws->result[14] = svd_residual(m, n, mnmin, ASAV, lda, ws->USAV, m,
                                  ws->SSAV, ws->VTSAV, n, ws->work);

    /* Test 16: U orthogonality */
    {
        double sfmin = dlamch("S");
        int n2 = 0;
        for (int i = 0; i < mnmin; i++) {
            if (ws->SSAV[i] > sfmin) n2++;
        }
        int ncols_check = (n2 > 0) ? n2 : mnmin;
        dort01("C", m, ncols_check, ws->USAV, m, ws->work, ws->lwork, &ws->result[15]);
    }

    /* Test 17: VT orthogonality - N×N */
    dort01("R", n, n, ws->VTSAV, n, ws->work, ws->lwork, &ws->result[16]);

    /* Test 18: S ordering */
    ws->result[17] = check_sv_order(ws->SSAV, mnmin, ulpinv);
}

/* ===== DGEJSV Tests (19-22) ===== */

/**
 * Test DGEJSV (Preconditioned Jacobi SVD).
 * Only works for M >= N.
 * Returns V (not VT), so we need to transpose.
 */
static void test_dgejsv(int m, int n, const double* ASAV, int lda,
                        int lswork, ddrvbd_workspace_t* ws)
{
    double ulp = dlamch("P");
    double ulpinv = 1.0 / ulp;
    int info;

    /* Initialize results to 0 */
    for (int i = 18; i < 22; i++) ws->result[i] = 0.0;

    /* DGEJSV only works for M >= N */
    if (m < n || m == 0 || n == 0) return;

    int mnmin = n;  /* Since m >= n, mnmin = n */

    dlacpy("F", m, n, ASAV, lda, ws->VTSAV, m);

    dgejsv("G", "U", "V", "R", "N", "N", m, n, ws->VTSAV, m,
           ws->SSAV, ws->USAV, m, ws->A, n, ws->work, lswork, ws->iwork, &info);

    if (info != 0) {
        ws->result[18] = ulpinv;
        return;
    }

    /* DGEJSV returns V in A, transpose to VTSAV */
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            ws->VTSAV[i + j * n] = ws->A[j + i * n];
        }
    }

    /* Test 19: Reconstruction using economy SVD */
    ws->result[18] = svd_residual(m, n, mnmin, ASAV, lda, ws->USAV, m,
                                  ws->SSAV, ws->VTSAV, n, ws->work);

    /* Test 20: U orthogonality */
    {
        double sfmin = dlamch("S");
        int n2 = 0;
        for (int i = 0; i < mnmin; i++) {
            if (ws->SSAV[i] > sfmin) n2++;
        }
        int ncols_check = (n2 > 0) ? n2 : mnmin;
        dort01("C", m, ncols_check, ws->USAV, m, ws->work, ws->lwork, &ws->result[19]);
    }

    /* Test 21: VT orthogonality - N×N */
    dort01("R", n, n, ws->VTSAV, n, ws->work, ws->lwork, &ws->result[20]);

    /* Test 22: S ordering */
    ws->result[21] = check_sv_order(ws->SSAV, mnmin, ulpinv);
}

/* ===== DGESVDX Tests (23-35) ===== */

/**
 * Test DGESVDX (SVD with range selection).
 *
 * Tests 23-26: RANGE='A' (all singular values), JOBU='V', JOBVT='V'
 * Tests 27-29: Partial SVD comparison (max over JOBU/JOBVT in {'N','V'})
 * Tests 30-32: RANGE='I' (index range)
 * Tests 33-35: RANGE='V' (value range)
 */
static void test_dgesvdx(int m, int n, const double* ASAV, int lda,
                         ddrvbd_workspace_t* ws)
{
    int mnmin = (m < n) ? m : n;
    double ulp = dlamch("P");
    double ulpinv = 1.0 / ulp;
    double unfl = dlamch("S");
    int info;
    int ns;
    int lwork = ws->lwork;

    /* Initialize results to 0 */
    for (int i = 22; i < 35; i++) ws->result[i] = 0.0;

    if (m == 0 || n == 0) return;

    /* === RANGE='A' with vectors === */
    dlacpy("F", m, n, ASAV, lda, ws->A, lda);

    dgesvdx("V", "V", "A", m, n, ws->A, lda,
            0.0, 0.0, 0, 0, &ns,
            ws->SSAV, ws->USAV, m, ws->VTSAV, mnmin,
            ws->work, lwork, ws->iwork, &info);

    if (info != 0) {
        ws->result[22] = ulpinv;
        return;
    }

    /* Test 23: Reconstruction */
    ws->result[22] = svd_residual(m, n, mnmin, ASAV, lda, ws->USAV, m,
                                  ws->SSAV, ws->VTSAV, mnmin, ws->work);

    /* Test 24: U orthogonality - U is M x MNMIN */
    dort01("C", m, mnmin, ws->USAV, m, ws->work, ws->lwork, &ws->result[23]);

    /* Test 25: VT orthogonality - VT is MNMIN x N */
    dort01("R", mnmin, n, ws->VTSAV, mnmin, ws->work, ws->lwork, &ws->result[24]);

    /* Test 26: S ordering */
    ws->result[25] = check_sv_order(ws->SSAV, mnmin, ulpinv);

    /* === Partial SVD: test JOBU/JOBVT in {'N','V'} (lines 1017-1067 in ddrvbd.f) === */
    ws->result[26] = 0.0;  /* max |U - Upartial| */
    ws->result[27] = 0.0;  /* max |VT - VTpartial| */
    ws->result[28] = 0.0;  /* max |S - Spartial| */

    for (int iju = 0; iju <= 1; iju++) {
        for (int ijvt = 0; ijvt <= 1; ijvt++) {
            /* Skip ('N','N') and ('V','V') - tested above */
            if ((iju == 0 && ijvt == 0) || (iju == 1 && ijvt == 1)) continue;

            char jobu[2] = {CJOBV[iju], '\0'};
            char jobvt[2] = {CJOBV[ijvt], '\0'};

            dlacpy("F", m, n, ASAV, lda, ws->A, lda);

            dgesvdx(jobu, jobvt, "A", m, n, ws->A, lda,
                    0.0, 0.0, 0, 0, &ns,
                    ws->S, ws->U, m, ws->VT, mnmin,
                    ws->work, lwork, ws->iwork, &info);

            if (info != 0) continue;

            /* Compare U */
            double dif = 0.0;
            if (m > 0 && n > 0 && iju == 1) {
                dort03("C", m, mnmin, m, mnmin, ws->USAV, m, ws->U, m,
                       ws->work, ws->lwork, &dif, &info);
            }
            if (dif > ws->result[26]) ws->result[26] = dif;

            /* Compare VT */
            dif = 0.0;
            if (m > 0 && n > 0 && ijvt == 1) {
                dort03("R", n, mnmin, n, mnmin, ws->VTSAV, mnmin, ws->VT, mnmin,
                       ws->work, ws->lwork, &dif, &info);
            }
            if (dif > ws->result[27]) ws->result[27] = dif;

            /* Compare S */
            dif = 0.0;
            double div = (mnmin * ulp * ws->S[0] > unfl) ? mnmin * ulp * ws->S[0] : unfl;
            for (int i = 0; i < mnmin - 1; i++) {
                if (ws->SSAV[i] < ws->SSAV[i + 1]) dif = ulpinv;
                if (ws->SSAV[i] < 0.0) dif = ulpinv;
                double d = fabs(ws->SSAV[i] - ws->S[i]) / div;
                if (d > dif) dif = d;
            }
            if (dif > ws->result[28]) ws->result[28] = dif;
        }
    }

    /* === RANGE='I' (index range) === */
    if (mnmin > 1) {
        int il = 1 + (int)((mnmin - 1) * rng_uniform());
        int iu = 1 + (int)((mnmin - 1) * rng_uniform());
        if (iu < il) {
            int tmp = il;
            il = iu;
            iu = tmp;
        }
        int nsi = iu - il + 1;

        dlacpy("F", m, n, ASAV, lda, ws->A, lda);

        dgesvdx("V", "V", "I", m, n, ws->A, lda,
                0.0, 0.0, il, iu, &ns,
                ws->S, ws->U, m, ws->VT, nsi,
                ws->work, lwork, ws->iwork, &info);

        if (info == 0 && ns > 0) {
            /* Test 30: Partial SVD reconstruction using dbdt05 */
            dbdt05(m, n, ASAV, lda, ws->S, ns, ws->U, m, ws->VT, nsi,
                   ws->work, &ws->result[29]);

            /* Test 31: U orthogonality */
            dort01("C", m, ns, ws->U, m, ws->work, ws->lwork, &ws->result[30]);

            /* Test 32: VT orthogonality */
            dort01("R", ns, n, ws->VT, nsi, ws->work, ws->lwork, &ws->result[31]);
        }
    }

    /* === RANGE='V' (value range) === */
    if (mnmin > 1 && ws->SSAV[0] > ws->SSAV[mnmin - 1]) {
        double vl = ws->SSAV[mnmin - 1] - ulp * ws->SSAV[0];
        double vu = ws->SSAV[0] + ulp * ws->SSAV[0];

        int mid = mnmin / 2;
        if (mid > 0 && mid < mnmin - 1) {
            vl = ws->SSAV[mid + 1] - ulp * ws->SSAV[0];
            vu = ws->SSAV[mid - 1] + ulp * ws->SSAV[0];
        }

        if (vl < 0.0) vl = 0.0;

        dlacpy("F", m, n, ASAV, lda, ws->A, lda);

        dgesvdx("V", "V", "V", m, n, ws->A, lda,
                vl, vu, 0, 0, &ns,
                ws->S, ws->U, m, ws->VT, mnmin,
                ws->work, lwork, ws->iwork, &info);

        if (info == 0 && ns > 0) {
            /* Test 33: Partial SVD reconstruction using dbdt05 */
            dbdt05(m, n, ASAV, lda, ws->S, ns, ws->U, m, ws->VT, mnmin,
                   ws->work, &ws->result[32]);

            /* Test 34: U orthogonality */
            dort01("C", m, ns, ws->U, m, ws->work, ws->lwork, &ws->result[33]);

            /* Test 35: VT orthogonality */
            dort01("R", ns, n, ws->VT, mnmin, ws->work, ws->lwork, &ws->result[34]);
        }
    }
}

/* ===== DGESVDQ Tests (36-39) ===== */

/**
 * Test DGESVDQ (QR-preconditioned SVD).
 * Only works for M >= N.
 * Returns V (not VT).
 *
 * @param lswork  Working workspace size (varies in IWS loop)
 */
static void test_dgesvdq(int m, int n, const double* ASAV, int lda,
                         int lswork, ddrvbd_workspace_t* ws)
{
    (void)lswork;  /* DGESVDQ uses ws->lwork directly for now */
    double ulp = dlamch("P");
    double ulpinv = 1.0 / ulp;
    int info;
    int numrank;

    /* Initialize results to 0 */
    for (int i = 35; i < 39; i++) ws->result[i] = 0.0;

    /* DGESVDQ only works for M >= N */
    if (m < n || m == 0 || n == 0) return;

    int mnmin = n;

    dlacpy("F", m, n, ASAV, lda, ws->A, lda);

    /* DGESVDQ: joba='A', jobp='N', jobr='N', jobu='A', jobv='A' */
    dgesvdq("A", "N", "N", "A", "A", m, n, ws->A, lda,
            ws->S, ws->U, m, ws->VT, n, &numrank,
            ws->iwork, ws->liwork, ws->work, ws->lwork,
            ws->rwork, ws->lrwork, &info);

    if (info != 0) {
        ws->result[35] = ulpinv;
        return;
    }

    /* Test 36: Reconstruction */
    ws->result[35] = svd_residual(m, n, mnmin, ASAV, lda, ws->U, m, ws->S, ws->VT, n, ws->work);

    /* Test 37: U orthogonality */
    dort01("C", m, mnmin, ws->U, m, ws->work, ws->lwork, &ws->result[36]);

    /* Test 38: VT orthogonality */
    dort01("R", mnmin, n, ws->VT, n, ws->work, ws->lwork, &ws->result[37]);

    /* Test 39: S ordering */
    ws->result[38] = check_sv_order(ws->S, mnmin, ulpinv);
}

/* ===== CMocka Setup/Teardown ===== */

static int group_setup(void** state)
{
    (void)state;

    /* Find maximum dimensions from the dimension pairs */
    int mmax = 0, nmax = 0;
    for (size_t i = 0; i < NDIM_PAIRS; i++) {
        if (DIM_PAIRS[i].m > mmax) mmax = DIM_PAIRS[i].m;
        if (DIM_PAIRS[i].n > nmax) nmax = DIM_PAIRS[i].n;
    }

    /* Ensure minimum size */
    if (mmax < 1) mmax = 1;
    if (nmax < 1) nmax = 1;

    int mnmax = (mmax > nmax) ? mmax : nmax;

    /* Allocate workspace structure */
    g_ws = malloc(sizeof(ddrvbd_workspace_t));
    if (!g_ws) return -1;

    g_ws->mmax = mmax;
    g_ws->nmax = nmax;
    g_ws->mnmax = mnmax;
    g_ws->lwork = compute_lwork(mmax, nmax);
    g_ws->liwork = 12 * mnmax;
    g_ws->lrwork = 2 * mnmax;
    g_ws->seed = 2024;

    /* Allocate arrays */
    g_ws->A = malloc(mmax * nmax * sizeof(double));
    g_ws->ASAV = malloc(mmax * nmax * sizeof(double));
    g_ws->U = malloc(mmax * mmax * sizeof(double));
    g_ws->USAV = malloc(mmax * mmax * sizeof(double));
    g_ws->VT = malloc(nmax * nmax * sizeof(double));
    g_ws->VTSAV = malloc(nmax * nmax * sizeof(double));
    g_ws->S = malloc(2 * mnmax * sizeof(double));
    g_ws->SSAV = malloc(2 * mnmax * sizeof(double));
    g_ws->work = malloc(g_ws->lwork * sizeof(double));
    g_ws->rwork = malloc(g_ws->lrwork * sizeof(double));
    g_ws->iwork = malloc(g_ws->liwork * sizeof(int));

    if (!g_ws->A || !g_ws->ASAV || !g_ws->U || !g_ws->USAV ||
        !g_ws->VT || !g_ws->VTSAV || !g_ws->S || !g_ws->SSAV ||
        !g_ws->work || !g_ws->rwork || !g_ws->iwork) {
        return -1;
    }

    return 0;
}

static int group_teardown(void** state)
{
    (void)state;

    if (g_ws) {
        free(g_ws->A);
        free(g_ws->ASAV);
        free(g_ws->U);
        free(g_ws->USAV);
        free(g_ws->VT);
        free(g_ws->VTSAV);
        free(g_ws->S);
        free(g_ws->SSAV);
        free(g_ws->work);
        free(g_ws->rwork);
        free(g_ws->iwork);
        free(g_ws);
        g_ws = NULL;
    }

    return 0;
}

/* ===== Parameterized Test Function ===== */

/**
 * Run all SVD tests for a single (m, n, itype) combination.
 * This includes the IWS loop over 4 workspace sizes.
 */
static void run_ddrvbd_single(ddrvbd_params_t* p)
{
    ddrvbd_workspace_t* ws = g_ws;
    char context[256];
    int m = p->m;
    int n = p->n;
    int itype = p->itype;

    /* Skip degenerate cases */
    if (m == 0 || n == 0) return;

    /* Seed based on parameters for reproducibility */
    uint64_t seed = 2024 + m * 1000 + n * 100 + itype;
    rng_seed(seed);

    int info;

    /* Generate test matrix */
    generate_test_matrix(itype, m, n, ws->ASAV, m,
                         ws->SSAV, ws->work, seed, &info);

    if (info != 0) {
        snprintf(context, sizeof(context),
                 "Matrix generation failed: m=%d n=%d type=%d info=%d",
                 m, n, itype, info);
        print_message("SKIP: %s\n", context);
        return;
    }

    /* IWS loop: test with 4 different workspace sizes (lines 568-785 in ddrvbd.f)
     * IWS=1: minimal workspace
     * IWS=2,3: intermediate workspace sizes
     * IWS=4: full workspace */
    for (int iws = 1; iws <= 4; iws++) {
        int mnmin = (m < n) ? m : n;
        int mxmax = (m > n) ? m : n;

        /* Compute LSWORK for DGESVD (lines 576-581 in ddrvbd.f) */
        int iwtmp_gesvd = 3 * mnmin + mxmax;
        int tmp = 5 * mnmin;
        if (tmp > iwtmp_gesvd) iwtmp_gesvd = tmp;

        int lswork_gesvd = iwtmp_gesvd + (iws - 1) * (ws->lwork - iwtmp_gesvd) / 3;
        if (lswork_gesvd > ws->lwork) lswork_gesvd = ws->lwork;
        if (lswork_gesvd < 1) lswork_gesvd = 1;
        if (iws == 4) lswork_gesvd = ws->lwork;

        /* Compute LSWORK for DGESDD (lines 690-695 in ddrvbd.f) */
        int iwtmp_gesdd = 5 * mnmin * mnmin + 9 * mnmin + mxmax;
        int lswork_gesdd = iwtmp_gesdd + (iws - 1) * (ws->lwork - iwtmp_gesdd) / 3;
        if (lswork_gesdd > ws->lwork) lswork_gesdd = ws->lwork;
        if (lswork_gesdd < 1) lswork_gesdd = 1;
        if (iws == 4) lswork_gesdd = ws->lwork;

        /* Initialize results to -1 */
        for (int i = 0; i < NRESULTS; i++) ws->result[i] = -1.0;

        /* DGESVD (tests 1-7) */
        test_dgesvd(m, n, ws->ASAV, m, lswork_gesvd, ws);
        for (int i = 0; i < 7; i++) {
            if (ws->result[i] < 0.0) continue;  /* Test not run */
            if (ws->result[i] >= THRESH) {
                snprintf(context, sizeof(context),
                         "DGESVD m=%d n=%d type=%d iws=%d test=%d",
                         m, n, itype, iws, i + 1);
                set_test_context(context);
                assert_residual_below(ws->result[i], THRESH);
            }
        }

        /* DGESDD (tests 8-14) */
        test_dgesdd(m, n, ws->ASAV, m, lswork_gesdd, ws);
        for (int i = 7; i < 14; i++) {
            if (ws->result[i] < 0.0) continue;
            if (ws->result[i] >= THRESH) {
                snprintf(context, sizeof(context),
                         "DGESDD m=%d n=%d type=%d iws=%d test=%d",
                         m, n, itype, iws, i + 1);
                set_test_context(context);
                assert_residual_below(ws->result[i], THRESH);
            }
        }

        /* DGESVJ (tests 15-18, M >= N only) */
        if (m >= n) {
            test_dgesvj(m, n, ws->ASAV, m, lswork_gesdd, ws);
            for (int i = 14; i < 18; i++) {
                if (ws->result[i] < 0.0) continue;
                if (ws->result[i] >= THRESH) {
                    snprintf(context, sizeof(context),
                             "DGESVJ m=%d n=%d type=%d iws=%d test=%d",
                             m, n, itype, iws, i + 1);
                    set_test_context(context);
                    assert_residual_below(ws->result[i], THRESH);
                }
            }
        }

        /* DGEJSV (tests 19-22, M >= N only) */
        if (m >= n) {
            test_dgejsv(m, n, ws->ASAV, m, lswork_gesdd, ws);
            for (int i = 18; i < 22; i++) {
                if (ws->result[i] < 0.0) continue;
                if (ws->result[i] >= THRESH) {
                    snprintf(context, sizeof(context),
                             "DGEJSV m=%d n=%d type=%d iws=%d test=%d",
                             m, n, itype, iws, i + 1);
                    set_test_context(context);
                    assert_residual_below(ws->result[i], THRESH);
                }
            }
        }

        /* DGESVDX (tests 23-35) */
        test_dgesvdx(m, n, ws->ASAV, m, ws);
        for (int i = 22; i < 35; i++) {
            if (ws->result[i] < 0.0) continue;
            if (ws->result[i] >= THRESH) {
                snprintf(context, sizeof(context),
                         "DGESVDX m=%d n=%d type=%d iws=%d test=%d",
                         m, n, itype, iws, i + 1);
                set_test_context(context);
                assert_residual_below(ws->result[i], THRESH);
            }
        }

        /* DGESVDQ (tests 36-39, M >= N only) */
        if (m >= n) {
            test_dgesvdq(m, n, ws->ASAV, m, lswork_gesdd, ws);
            for (int i = 35; i < 39; i++) {
                if (ws->result[i] < 0.0) continue;
                if (ws->result[i] >= THRESH) {
                    snprintf(context, sizeof(context),
                             "DGESVDQ m=%d n=%d type=%d iws=%d test=%d",
                             m, n, itype, iws, i + 1);
                    set_test_context(context);
                    assert_residual_below(ws->result[i], THRESH);
                }
            }
        }
    }  /* End IWS loop */

    clear_test_context();
}

/**
 * CMocka test function - dispatches to run_ddrvbd_single based on prestate.
 */
static void test_ddrvbd_case(void** state)
{
    ddrvbd_params_t* params = *state;
    run_ddrvbd_single(params);
}

/*
 * Generate all parameter combinations.
 * Total: NDIM_PAIRS * NTYPES tests
 */

/* Maximum number of test cases */
#define MAX_TESTS (NDIM_PAIRS * NTYPES)

static ddrvbd_params_t g_params[MAX_TESTS];
static struct CMUnitTest g_tests[MAX_TESTS];
static int g_num_tests = 0;

/**
 * Build the test array with all parameter combinations.
 */
static void build_test_array(void)
{
    g_num_tests = 0;

    for (size_t ipair = 0; ipair < NDIM_PAIRS; ipair++) {
        int m = DIM_PAIRS[ipair].m;
        int n = DIM_PAIRS[ipair].n;

        for (int itype = 1; itype <= NTYPES; itype++) {
            /* Store parameters */
            ddrvbd_params_t* p = &g_params[g_num_tests];
            p->m = m;
            p->n = n;
            p->itype = itype;
            snprintf(p->name, sizeof(p->name),
                     "ddrvbd_m%d_n%d_type%d", m, n, itype);

            /* Create CMocka test entry */
            g_tests[g_num_tests].name = p->name;
            g_tests[g_num_tests].test_func = test_ddrvbd_case;
            g_tests[g_num_tests].setup_func = NULL;
            g_tests[g_num_tests].teardown_func = NULL;
            g_tests[g_num_tests].initial_state = p;

            g_num_tests++;
        }
    }
}

/* ===== Main ===== */

int main(void)
{
    /* Build all test cases */
    build_test_array();

    /* Run all tests with shared workspace */
    return _cmocka_run_group_tests("ddrvbd", g_tests, g_num_tests,
                                   group_setup, group_teardown);
}
