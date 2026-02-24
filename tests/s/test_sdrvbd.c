/**
 * @file test_sdrvbd.c
 * @brief Comprehensive SVD test driver - port of LAPACK TESTING/EIG/ddrvbd.f
 *
 * Tests all 6 SVD drivers:
 * - SGESVD:  QR iteration SVD
 * - SGESDD:  Divide-and-conquer SVD
 * - SGESVDX: SVD with range selection (bisection)
 * - SGESVDQ: QR-preconditioned SVD
 * - SGESVJ:  Jacobi SVD
 * - SGEJSV:  Preconditioned Jacobi SVD (high accuracy)
 *
 * Each (m, n, type) combination is registered as a separate CMocka test,
 * providing pytest-like parameterized test behavior with clear failure isolation.
 *
 * Test ratios (39 total):
 *   SGESVD  (1-7):   Full and partial SVD
 *   SGESDD  (8-14):  Full and partial SVD
 *   SGESVJ  (15-18): Full SVD (M >= N only)
 *   SGEJSV  (19-22): Full SVD (M >= N only)
 *   SGESVDX (23-35): Full and range-selected SVD
 *   SGESVDQ (36-39): Full SVD (M >= N only)
 *
 * Matrix types (5 total):
 *   1. Zero matrix
 *   2. Identity matrix
 *   3. Random with evenly spaced singular values in [ULP, 1]
 *   4. Same as 3, scaled near underflow
 *   5. Same as 3, scaled near overflow
 */

#include "test_harness.h"
#include "test_rng.h"
#include "verify.h"

/* Test threshold from LAPACK svd.in (line 10) */
#define THRESH 50.0f

/* Number of matrix types */
#define NTYPES 5

/* Number of test results */
#define NRESULTS 39

/* Job option strings for partial SVD tests (from ddrvbd.f lines 439-441) */
static const char* CJOB = "NOSA";   /* 'N', 'O', 'S', 'A' for SGESVD/SGESDD */
static const char* CJOBV = "NV";    /* 'N', 'V' for SGESVDX */

/* Test dimension pairs from LAPACK's svd.in
 * These are specific (M,N) pairs, not all combinations.
 * LAPACK tests: 0 0 0 1 1 1 2 2 3 3 3 10 10 16 16 30 30 40 40 (M values)
 *               0 1 3 0 1 2 0 1 0 1 3 10 16 10 16 30 40 30 40 (N values)
 */
typedef struct { INT m; INT n; } dim_pair_t;
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
/* Utility routines */
/* Matrix generation and verification routines from verify.h */

/**
 * Test parameters for a single test case.
 */
typedef struct {
    INT m;
    INT n;
    INT itype;
    char name[64];
} ddrvbd_params_t;

/*
 * Workspace structure for all SVD tests.
 * Allocated once at maximum dimensions, reused for all size combinations.
 */
typedef struct {
    INT mmax, nmax, mnmax;

    /* Matrices */
    f32* A;      /* m x n working copy */
    f32* ASAV;   /* m x n saved copy */
    f32* U;      /* m x m left singular vectors */
    f32* USAV;   /* m x m saved U */
    f32* VT;     /* n x n right singular vectors */
    f32* VTSAV;  /* n x n saved VT */

    /* Singular values */
    f32* S;      /* min(m,n) computed singular values */
    f32* SSAV;   /* min(m,n) saved singular values */

    /* Work arrays */
    f32* work;   /* general workspace */
    f32* rwork;  /* real workspace for sgesvdq */
    INT* iwork;     /* integer workspace */
    INT lwork;      /* workspace size */
    INT liwork;     /* integer workspace size */
    INT lrwork;     /* real workspace size */

    /* Test results */
    f32 result[NRESULTS];

    /* RNG state */
    uint64_t rng_state[4];
} ddrvbd_workspace_t;

/* Global workspace pointer */
static ddrvbd_workspace_t* g_ws = NULL;

/* ===== Helper Functions ===== */

/**
 * Compute maximum workspace needed for all SVD routines.
 */
static INT compute_lwork(INT mmax, INT nmax)
{
    INT mnmax = (mmax < nmax) ? mmax : nmax;
    INT mxmax = (mmax > nmax) ? mmax : nmax;

    /* From ddrvbd.f:
     * LWORK = MAX(3*MN + MX, 5*MN - 4) + 2*MN^2
     * Plus extra for SGESVDQ
     */
    INT lwork1 = 3 * mnmax + mxmax;
    INT lwork2 = 5 * mnmax - 4;
    INT lwork = (lwork1 > lwork2) ? lwork1 : lwork2;
    lwork += 2 * mnmax * mnmax;

    /* Extra for SGESVDQ */
    INT lwork_q = 5 * mnmax * mnmax + 9 * mnmax + mxmax;
    if (lwork_q > lwork) lwork = lwork_q;

    /* Extra for SGEJSV */
    INT lwork_j = 6 * mnmax * mnmax + 10 * mnmax + mxmax;
    if (lwork_j > lwork) lwork = lwork_j;

    /* Extra safety margin */
    lwork = lwork + 100;

    return lwork;
}

/**
 * Check that singular values are non-negative and decreasing.
 * Returns 0.0 if valid, 1/ULP if invalid.
 */
static f32 check_sv_order(const f32* S, INT n, f32 ulpinv)
{
    for (INT i = 0; i < n; i++) {
        if (S[i] < 0.0f) return ulpinv;
    }
    for (INT i = 0; i < n - 1; i++) {
        if (S[i] < S[i + 1]) return ulpinv;
    }
    return 0.0f;
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
static void generate_test_matrix(INT itype, INT m, INT n, f32* A, INT lda,
                                 f32* S, f32* work, INT* info,
                                 uint64_t state[static 4])
{
    f32 ulp = slamch("P");
    f32 unfl = slamch("S");
    f32 ovfl = 1.0f / unfl;
    INT mnmin = (m < n) ? m : n;

    *info = 0;

    if (m == 0 || n == 0) {
        return;
    }

    if (itype == 1) {
        /* Type 1: Zero matrix */
        slaset("F", m, n, 0.0f, 0.0f, A, lda);
        for (INT i = 0; i < mnmin; i++) S[i] = 0.0f;
    }
    else if (itype == 2) {
        /* Type 2: Identity matrix */
        slaset("F", m, n, 0.0f, 1.0f, A, lda);
        for (INT i = 0; i < mnmin; i++) S[i] = 1.0f;
    }
    else if (itype >= 3 && itype <= 5) {
        /* Types 3-5: Random matrix with controlled singular values */

        /* Generate singular values: evenly spaced from 1 to 1/cond */
        /* mode=4 in slatms gives evenly spaced SV from 1 to 1/cond */
        /* LAPACK uses cond = mnmin (see ddrvbd.f line 553) */
        f32 cond = (f32)mnmin;
        f32 anorm = 1.0f;

        if (itype == 4) {
            /* Scale near underflow */
            anorm = unfl / ulp;
        }
        else if (itype == 5) {
            /* Scale near overflow */
            anorm = ovfl * ulp;
        }

        /* Use slatms to generate random matrix with specified singular values */
        slatms(m, n, "U", "N", S, 4, cond, anorm,
               m - 1, n - 1, "N", A, lda, work, info, state);

        if (*info != 0) {
            return;
        }

        /* Sort singular values in decreasing order for verification */
        /* slatms with mode=4 gives SV from 1 to 1/cond */
        /* We need to know what values to expect */
        for (INT i = 0; i < mnmin; i++) {
            f32 t = (f32)i / (f32)(mnmin > 1 ? mnmin - 1 : 1);
            S[i] = anorm * (1.0f - t * (1.0f - 1.0f / cond));
        }
    }
    else {
        *info = -1;  /* Invalid type */
    }
}

/* ===== SGESVD Tests (1-7) ===== */

/**
 * Test SGESVD with full and partial SVD.
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
 * @param m       Number of rows of A
 * @param n       Number of columns of A
 * @param ASAV    Saved copy of original matrix A
 * @param lda     Leading dimension of A
 * @param lswork  Working workspace size (varies in IWS loop)
 * @param ws      Workspace structure
 */
static void test_dgesvd(INT m, INT n, const f32* ASAV, INT lda,
                        INT lswork, ddrvbd_workspace_t* ws)
{
    INT mnmin = (m < n) ? m : n;
    f32 ulp = slamch("P");
    f32 ulpinv = 1.0f / ulp;
    f32 unfl = slamch("S");
    INT info;

    /* Initialize results to 0 */
    for (INT i = 0; i < 7; i++) ws->result[i] = 0.0f;

    if (m == 0 || n == 0) return;

    /* === Full SVD (jobu='A', jobvt='A') === */

    slacpy("F", m, n, ASAV, lda, ws->A, lda);
    sgesvd("A", "A", m, n, ws->A, lda, ws->SSAV,
           ws->USAV, m, ws->VTSAV, n, ws->work, lswork, &info);

    if (info != 0) {
        ws->result[0] = ulpinv;
        return;
    }

    /* Test 1: Reconstruction |A - U*S*VT| / (n * |A| * eps) */
    sbdt01(m, n, 0, ASAV, lda, ws->USAV, m, ws->SSAV, NULL,
           ws->VTSAV, n, ws->work, &ws->result[0]);

    /* Test 2: U orthogonality */
    sort01("C", m, m, ws->USAV, m, ws->work, ws->lwork, &ws->result[1]);

    /* Test 3: VT orthogonality */
    sort01("R", n, n, ws->VTSAV, n, ws->work, ws->lwork, &ws->result[2]);

    /* Test 4: S ordering */
    ws->result[3] = check_sv_order(ws->SSAV, mnmin, ulpinv);

    /* === Partial SVD: test all 14 JOBU/JOBVT combinations (lines 622-686 in ddrvbd.f) === */
    ws->result[4] = 0.0f;  /* max |U - Upartial| */
    ws->result[5] = 0.0f;  /* max |VT - VTpartial| */
    ws->result[6] = 0.0f;  /* max |S - Spartial| */

    for (INT iju = 0; iju <= 3; iju++) {
        for (INT ijvt = 0; ijvt <= 3; ijvt++) {
            /* Skip ('A','A') - tested above; skip ('O','O') - invalid */
            if ((iju == 3 && ijvt == 3) || (iju == 1 && ijvt == 1)) continue;

            char jobu[2] = {CJOB[iju], '\0'};
            char jobvt[2] = {CJOB[ijvt], '\0'};

            slacpy("F", m, n, ASAV, lda, ws->A, lda);

            sgesvd(jobu, jobvt, m, n, ws->A, lda, ws->S,
                   ws->U, m, ws->VT, n, ws->work, lswork, &info);

            if (info != 0) continue;

            /* Compare U */
            f32 dif = 0.0f;
            if (m > 0 && n > 0) {
                if (iju == 1) {
                    /* JOBU='O': U stored in A */
                    sort03("C", m, mnmin, m, mnmin, ws->USAV, m, ws->A, lda,
                           ws->work, ws->lwork, &dif, &info);
                } else if (iju == 2) {
                    /* JOBU='S': economy U in ws->U */
                    sort03("C", m, mnmin, m, mnmin, ws->USAV, m, ws->U, m,
                           ws->work, ws->lwork, &dif, &info);
                } else if (iju == 3) {
                    /* JOBU='A': full U in ws->U */
                    sort03("C", m, m, m, mnmin, ws->USAV, m, ws->U, m,
                           ws->work, ws->lwork, &dif, &info);
                }
            }
            if (dif > ws->result[4]) ws->result[4] = dif;

            /* Compare VT */
            dif = 0.0f;
            if (m > 0 && n > 0) {
                if (ijvt == 1) {
                    /* JOBVT='O': VT stored in A */
                    sort03("R", n, mnmin, n, mnmin, ws->VTSAV, n, ws->A, lda,
                           ws->work, ws->lwork, &dif, &info);
                } else if (ijvt == 2) {
                    /* JOBVT='S': economy VT in ws->VT */
                    sort03("R", n, mnmin, n, mnmin, ws->VTSAV, n, ws->VT, n,
                           ws->work, ws->lwork, &dif, &info);
                } else if (ijvt == 3) {
                    /* JOBVT='A': full VT in ws->VT */
                    sort03("R", n, n, n, mnmin, ws->VTSAV, n, ws->VT, n,
                           ws->work, ws->lwork, &dif, &info);
                }
            }
            if (dif > ws->result[5]) ws->result[5] = dif;

            /* Compare S */
            dif = 0.0f;
            f32 div = (mnmin * ulp * ws->S[0] > unfl) ? mnmin * ulp * ws->S[0] : unfl;
            for (INT i = 0; i < mnmin - 1; i++) {
                if (ws->SSAV[i] < ws->SSAV[i + 1]) dif = ulpinv;
                if (ws->SSAV[i] < 0.0f) dif = ulpinv;
                f32 d = fabsf(ws->SSAV[i] - ws->S[i]) / div;
                if (d > dif) dif = d;
            }
            if (dif > ws->result[6]) ws->result[6] = dif;
        }
    }
}

/* ===== SGESDD Tests (8-14) ===== */

/**
 * Test SGESDD with full and partial SVD.
 * Same test structure as SGESVD, results stored in indices 7-13.
 *
 * Tests 8-11: Full SVD (jobz='A')
 * Tests 12-14: Partial SVD (max over jobz='N','O','S')
 *
 * @param m       Number of rows of A
 * @param n       Number of columns of A
 * @param ASAV    Saved copy of original matrix A
 * @param lda     Leading dimension of A
 * @param lswork  Working workspace size (varies in IWS loop)
 * @param ws      Workspace structure
 */
static void test_dgesdd(INT m, INT n, const f32* ASAV, INT lda,
                        INT lswork, ddrvbd_workspace_t* ws)
{
    INT mnmin = (m < n) ? m : n;
    f32 ulp = slamch("P");
    f32 ulpinv = 1.0f / ulp;
    f32 unfl = slamch("S");
    INT info;

    /* Initialize results to 0 */
    for (INT i = 7; i < 14; i++) ws->result[i] = 0.0f;

    if (m == 0 || n == 0) return;

    /* === Full SVD (jobz='A') === */
    slacpy("F", m, n, ASAV, lda, ws->A, lda);

    sgesdd("A", m, n, ws->A, lda, ws->SSAV,
           ws->USAV, m, ws->VTSAV, n, ws->work, lswork, ws->iwork, &info);

    if (info != 0) {
        ws->result[7] = ulpinv;
        return;
    }

    /* Test 8: Reconstruction |A - U*S*VT| / (n * |A| * eps) */
    sbdt01(m, n, 0, ASAV, lda, ws->USAV, m, ws->SSAV, NULL,
           ws->VTSAV, n, ws->work, &ws->result[7]);

    /* Test 9: U orthogonality */
    sort01("C", m, m, ws->USAV, m, ws->work, ws->lwork, &ws->result[8]);

    /* Test 10: VT orthogonality */
    sort01("R", n, n, ws->VTSAV, n, ws->work, ws->lwork, &ws->result[9]);

    /* Test 11: S ordering */
    ws->result[10] = check_sv_order(ws->SSAV, mnmin, ulpinv);

    /* === Partial SVD: test jobz='N','O','S' (lines 735-797 in ddrvbd.f) === */
    ws->result[11] = 0.0f;  /* max |U - Upartial| */
    ws->result[12] = 0.0f;  /* max |VT - VTpartial| */
    ws->result[13] = 0.0f;  /* max |S - Spartial| */

    for (INT ijq = 0; ijq <= 2; ijq++) {
        char jobq[2] = {CJOB[ijq], '\0'};  /* 'N', 'O', 'S' */

        slacpy("F", m, n, ASAV, lda, ws->A, lda);

        sgesdd(jobq, m, n, ws->A, lda, ws->S,
               ws->U, m, ws->VT, mnmin, ws->work, lswork, ws->iwork, &info);

        if (info != 0) continue;

        /* Compare U */
        f32 dif = 0.0f;
        if (m > 0 && n > 0) {
            if (ijq == 1) {
                /* JOBZ='O': U or VT in A depending on M >= N */
                if (m >= n) {
                    sort03("C", m, mnmin, m, mnmin, ws->USAV, m, ws->A, lda,
                           ws->work, ws->lwork, &dif, &info);
                } else {
                    sort03("C", m, mnmin, m, mnmin, ws->USAV, m, ws->U, m,
                           ws->work, ws->lwork, &dif, &info);
                }
            } else if (ijq == 2) {
                /* JOBZ='S': economy U in ws->U */
                sort03("C", m, mnmin, m, mnmin, ws->USAV, m, ws->U, m,
                       ws->work, ws->lwork, &dif, &info);
            }
        }
        if (dif > ws->result[11]) ws->result[11] = dif;

        /* Compare VT */
        dif = 0.0f;
        if (m > 0 && n > 0) {
            if (ijq == 1) {
                /* JOBZ='O': VT or U in A depending on M >= N */
                if (m >= n) {
                    sort03("R", n, mnmin, n, mnmin, ws->VTSAV, n, ws->VT, mnmin,
                           ws->work, ws->lwork, &dif, &info);
                } else {
                    sort03("R", n, mnmin, n, mnmin, ws->VTSAV, n, ws->A, lda,
                           ws->work, ws->lwork, &dif, &info);
                }
            } else if (ijq == 2) {
                /* JOBZ='S': economy VT in ws->VT */
                sort03("R", n, mnmin, n, mnmin, ws->VTSAV, n, ws->VT, mnmin,
                       ws->work, ws->lwork, &dif, &info);
            }
        }
        if (dif > ws->result[12]) ws->result[12] = dif;

        /* Compare S */
        dif = 0.0f;
        f32 div = (mnmin * ulp * ws->S[0] > unfl) ? mnmin * ulp * ws->S[0] : unfl;
        for (INT i = 0; i < mnmin - 1; i++) {
            if (ws->SSAV[i] < ws->SSAV[i + 1]) dif = ulpinv;
            if (ws->SSAV[i] < 0.0f) dif = ulpinv;
            f32 d = fabsf(ws->SSAV[i] - ws->S[i]) / div;
            if (d > dif) dif = d;
        }
        if (dif > ws->result[13]) ws->result[13] = dif;
    }
}

/* ===== SGESVJ Tests (15-18) ===== */

/**
 * Test SGESVJ (Jacobi SVD).
 * Only works for M >= N.
 * Returns V (not VT), so we need to transpose.
 *
 * @param m       Number of rows of A
 * @param n       Number of columns of A
 * @param ASAV    Saved copy of original matrix A
 * @param lda     Leading dimension of A
 * @param lswork  Working workspace size (varies in IWS loop)
 * @param ws      Workspace structure
 */
static void test_dgesvj(INT m, INT n, const f32* ASAV, INT lda,
                        ddrvbd_workspace_t* ws)
{
    f32 ulpinv = 1.0f / slamch("P");
    INT info;

    /* Initialize results to 0 */
    for (INT i = 14; i < 18; i++) ws->result[i] = 0.0f;

    /* SGESVJ only works for M >= N */
    if (m < n || m == 0 || n == 0) return;

    INT mnmin = n;  /* Since m >= n, mnmin = n */

    slacpy("F", m, n, ASAV, lda, ws->USAV, m);

    sgesvj("G", "U", "V", m, n, ws->USAV, m, ws->SSAV, 0, ws->A, n,
           ws->work, ws->lwork, &info);

    if (info != 0) {
        ws->result[14] = ulpinv;
        return;
    }

    /* Transpose V to VTSAV: ws->A currently holds V (n x n), need V^T in VTSAV */
    for (INT i = 0; i < n; i++) {
        for (INT j = 0; j < n; j++) {
            ws->VTSAV[i + j * n] = ws->A[j + i * n];
        }
    }

    /* Test 15: Reconstruction |A - U*S*VT| / (n * |A| * eps) */
    sbdt01(m, n, 0, ASAV, lda, ws->USAV, m, ws->SSAV, NULL,
           ws->VTSAV, n, ws->work, &ws->result[14]);

    /* Test 16: U orthogonality */
    if (m != 0 && n != 0) {
        sort01("C", m, m, ws->USAV, m, ws->work, ws->lwork, &ws->result[15]);
        sort01("R", n, n, ws->VTSAV, n, ws->work, ws->lwork, &ws->result[16]);
    }

    /* Test 18: S ordering */
    ws->result[17] = check_sv_order(ws->SSAV, mnmin, ulpinv);
}

/* ===== SGEJSV Tests (19-22) ===== */

/**
 * Test SGEJSV (Preconditioned Jacobi SVD).
 * Only works for M >= N.
 * Returns V (not VT), so we need to transpose.
 */
static void test_dgejsv(INT m, INT n, const f32* ASAV, INT lda,
                        ddrvbd_workspace_t* ws)
{
    f32 ulpinv = 1.0f / slamch("P");
    INT info;

    /* Initialize results to 0 */
    for (INT i = 18; i < 22; i++) ws->result[i] = 0.0f;

    /* SGEJSV only works for M >= N */
    if (m < n || m == 0 || n == 0) return;

    INT mnmin = n;  /* Since m >= n, mnmin = n */

    slacpy("F", m, n, ASAV, lda, ws->VTSAV, m);

    sgejsv("G", "U", "V", "R", "N", "N", m, n, ws->VTSAV, m,
           ws->SSAV, ws->USAV, m, ws->A, n, ws->work, ws->lwork, ws->iwork, &info);

    if (info != 0) {
        ws->result[18] = ulpinv;
        return;
    }

    /* SGEJSV returns V in A, transpose to VTSAV */
    for (INT i = 0; i < n; i++) {
        for (INT j = 0; j < n; j++) {
            ws->VTSAV[i + j * n] = ws->A[j + i * n];
        }
    }

    /* Test 19: Reconstruction |A - U*S*VT| / (n * |A| * eps) */
    sbdt01(m, n, 0, ASAV, lda, ws->USAV, m, ws->SSAV, NULL,
           ws->VTSAV, n, ws->work, &ws->result[18]);

    /* Test 20: U orthogonality, Test 21: VT orthogonality */
    if (m != 0 && n != 0) {
        sort01("C", m, m, ws->USAV, m, ws->work, ws->lwork, &ws->result[19]);
        sort01("R", n, n, ws->VTSAV, n, ws->work, ws->lwork, &ws->result[20]);
    }

    /* Test 22: S ordering */
    ws->result[21] = check_sv_order(ws->SSAV, mnmin, ulpinv);
}

/* ===== SGESVDX Tests (23-35) ===== */

/**
 * Test SGESVDX (SVD with range selection).
 *
 * Tests 23-26: RANGE='A' (all singular values), JOBU='V', JOBVT='V'
 * Tests 27-29: Partial SVD comparison (max over JOBU/JOBVT in {'N','V'})
 * Tests 30-32: RANGE='I' (index range)
 * Tests 33-35: RANGE='V' (value range)
 */
static void test_dgesvdx(INT m, INT n, const f32* ASAV, INT lda,
                         ddrvbd_workspace_t* ws)
{
    INT mnmin = (m < n) ? m : n;
    f32 ulp = slamch("P");
    f32 ulpinv = 1.0f / ulp;
    f32 unfl = slamch("S");
    INT info;
    INT ns;
    INT lwork = ws->lwork;

    /* Initialize results to 0 */
    for (INT i = 22; i < 35; i++) ws->result[i] = 0.0f;

    if (m == 0 || n == 0) return;

    /* === RANGE='A' with vectors === */
    slacpy("F", m, n, ASAV, lda, ws->A, lda);

    sgesvdx("V", "V", "A", m, n, ws->A, lda,
            0.0f, 0.0f, 0, 0, &ns,
            ws->SSAV, ws->USAV, m, ws->VTSAV, mnmin,
            ws->work, lwork, ws->iwork, &info);

    if (info != 0) {
        ws->result[22] = ulpinv;
        return;
    }

    /* Test 23: Reconstruction |A - U*S*VT| / (n * |A| * eps) */
    sbdt01(m, n, 0, ASAV, lda, ws->USAV, m, ws->SSAV, NULL,
           ws->VTSAV, mnmin, ws->work, &ws->result[22]);

    /* Test 24: U orthogonality - U is M x MNMIN */
    sort01("C", m, mnmin, ws->USAV, m, ws->work, ws->lwork, &ws->result[23]);

    /* Test 25: VT orthogonality - VT is MNMIN x N */
    sort01("R", mnmin, n, ws->VTSAV, mnmin, ws->work, ws->lwork, &ws->result[24]);

    /* Test 26: S ordering */
    ws->result[25] = check_sv_order(ws->SSAV, mnmin, ulpinv);

    /* === Partial SVD: test JOBU/JOBVT in {'N','V'} (lines 1017-1067 in ddrvbd.f) === */
    ws->result[26] = 0.0f;  /* max |U - Upartial| */
    ws->result[27] = 0.0f;  /* max |VT - VTpartial| */
    ws->result[28] = 0.0f;  /* max |S - Spartial| */

    for (INT iju = 0; iju <= 1; iju++) {
        for (INT ijvt = 0; ijvt <= 1; ijvt++) {
            /* Skip ('N','N') and ('V','V') - tested above */
            if ((iju == 0 && ijvt == 0) || (iju == 1 && ijvt == 1)) continue;

            char jobu[2] = {CJOBV[iju], '\0'};
            char jobvt[2] = {CJOBV[ijvt], '\0'};

            slacpy("F", m, n, ASAV, lda, ws->A, lda);

            sgesvdx(jobu, jobvt, "A", m, n, ws->A, lda,
                    0.0f, 0.0f, 0, 0, &ns,
                    ws->S, ws->U, m, ws->VT, mnmin,
                    ws->work, lwork, ws->iwork, &info);

            if (info != 0) continue;

            /* Compare U */
            f32 dif = 0.0f;
            if (m > 0 && n > 0 && iju == 1) {
                sort03("C", m, mnmin, m, mnmin, ws->USAV, m, ws->U, m,
                       ws->work, ws->lwork, &dif, &info);
            }
            if (dif > ws->result[26]) ws->result[26] = dif;

            /* Compare VT */
            dif = 0.0f;
            if (m > 0 && n > 0 && ijvt == 1) {
                sort03("R", n, mnmin, n, mnmin, ws->VTSAV, mnmin, ws->VT, mnmin,
                       ws->work, ws->lwork, &dif, &info);
            }
            if (dif > ws->result[27]) ws->result[27] = dif;

            /* Compare S */
            dif = 0.0f;
            f32 div = (mnmin * ulp * ws->S[0] > unfl) ? mnmin * ulp * ws->S[0] : unfl;
            for (INT i = 0; i < mnmin - 1; i++) {
                if (ws->SSAV[i] < ws->SSAV[i + 1]) dif = ulpinv;
                if (ws->SSAV[i] < 0.0f) dif = ulpinv;
                f32 d = fabsf(ws->SSAV[i] - ws->S[i]) / div;
                if (d > dif) dif = d;
            }
            if (dif > ws->result[28]) ws->result[28] = dif;
        }
    }

    /* === RANGE='I' (index range) === */
    if (mnmin > 1) {
        INT il = 1 + (INT)((mnmin - 1) * rng_uniform_f32(ws->rng_state));
        INT iu = 1 + (INT)((mnmin - 1) * rng_uniform_f32(ws->rng_state));
        if (iu < il) {
            INT tmp = il;
            il = iu;
            iu = tmp;
        }
        INT nsi = iu - il + 1;

        slacpy("F", m, n, ASAV, lda, ws->A, lda);

        sgesvdx("V", "V", "I", m, n, ws->A, lda,
                0.0f, 0.0f, il, iu, &ns,
                ws->S, ws->U, m, ws->VT, nsi,
                ws->work, lwork, ws->iwork, &info);

        if (info == 0 && ns > 0) {
            /* Test 30: Partial SVD reconstruction using sbdt05 */
            sbdt05(m, n, ASAV, lda, ws->S, ns, ws->U, m, ws->VT, nsi,
                   ws->work, &ws->result[29]);

            /* Test 31: U orthogonality */
            sort01("C", m, ns, ws->U, m, ws->work, ws->lwork, &ws->result[30]);

            /* Test 32: VT orthogonality */
            sort01("R", ns, n, ws->VT, nsi, ws->work, ws->lwork, &ws->result[31]);
        }
    }

    /* === RANGE='V' (value range) === */
    if (mnmin > 1 && ws->SSAV[0] > ws->SSAV[mnmin - 1]) {
        f32 vl = ws->SSAV[mnmin - 1] - ulp * ws->SSAV[0];
        f32 vu = ws->SSAV[0] + ulp * ws->SSAV[0];

        INT mid = mnmin / 2;
        if (mid > 0 && mid < mnmin - 1) {
            vl = ws->SSAV[mid + 1] - ulp * ws->SSAV[0];
            vu = ws->SSAV[mid - 1] + ulp * ws->SSAV[0];
        }

        if (vl < 0.0f) vl = 0.0f;

        slacpy("F", m, n, ASAV, lda, ws->A, lda);

        sgesvdx("V", "V", "V", m, n, ws->A, lda,
                vl, vu, 0, 0, &ns,
                ws->S, ws->U, m, ws->VT, mnmin,
                ws->work, lwork, ws->iwork, &info);

        if (info == 0 && ns > 0) {
            /* Test 33: Partial SVD reconstruction using sbdt05 */
            sbdt05(m, n, ASAV, lda, ws->S, ns, ws->U, m, ws->VT, mnmin,
                   ws->work, &ws->result[32]);

            /* Test 34: U orthogonality */
            sort01("C", m, ns, ws->U, m, ws->work, ws->lwork, &ws->result[33]);

            /* Test 35: VT orthogonality */
            sort01("R", ns, n, ws->VT, mnmin, ws->work, ws->lwork, &ws->result[34]);
        }
    }
}

/* ===== SGESVDQ Tests (36-39) ===== */

/**
 * Test SGESVDQ (QR-preconditioned SVD).
 * Only works for M >= N.
 * Returns V (not VT).
 *
 * @param m       Number of rows of A
 * @param n       Number of columns of A
 * @param ASAV    Saved copy of original matrix A
 * @param lda     Leading dimension of A
 * @param lswork  Working workspace size (varies in IWS loop)
 * @param ws      Workspace structure
 */
static void test_dgesvdq(INT m, INT n, const f32* ASAV, INT lda,
                         INT lswork, ddrvbd_workspace_t* ws)
{
    (void)lswork;  /* SGESVDQ uses ws->lwork directly for now */
    f32 ulp = slamch("P");
    f32 ulpinv = 1.0f / ulp;
    INT info;
    INT numrank;

    /* Initialize results to 0 */
    for (INT i = 35; i < 39; i++) ws->result[i] = 0.0f;

    /* SGESVDQ only works for M >= N */
    if (m < n || m == 0 || n == 0) return;

    INT mnmin = n;

    slacpy("F", m, n, ASAV, lda, ws->A, lda);

    /* SGESVDQ: joba='A', jobp='N', jobr='N', jobu='A', jobv='A' */
    sgesvdq("A", "N", "N", "A", "A", m, n, ws->A, lda,
            ws->S, ws->U, m, ws->VT, n, &numrank,
            ws->iwork, ws->liwork, ws->work, ws->lwork,
            ws->rwork, ws->lrwork, &info);

    if (info != 0) {
        ws->result[35] = ulpinv;
        return;
    }

    /* Test 36: Reconstruction |A - U*S*VT| / (n * |A| * eps) */
    sbdt01(m, n, 0, ASAV, lda, ws->U, m, ws->S, NULL,
           ws->VT, n, ws->work, &ws->result[35]);

    /* Test 37: U orthogonality, Test 38: VT orthogonality */
    if (m != 0 && n != 0) {
        sort01("C", m, m, ws->U, m, ws->work, ws->lwork, &ws->result[36]);
        sort01("R", n, n, ws->VT, n, ws->work, ws->lwork, &ws->result[37]);
    }

    /* Test 39: S ordering */
    ws->result[38] = check_sv_order(ws->S, mnmin, ulpinv);
}

/* ===== CMocka Setup/Teardown ===== */

static int group_setup(void** state)
{
    (void)state;

    /* Find maximum dimensions from the dimension pairs */
    INT mmax = 0, nmax = 0;
    for (size_t i = 0; i < NDIM_PAIRS; i++) {
        if (DIM_PAIRS[i].m > mmax) mmax = DIM_PAIRS[i].m;
        if (DIM_PAIRS[i].n > nmax) nmax = DIM_PAIRS[i].n;
    }

    /* Ensure minimum size */
    if (mmax < 1) mmax = 1;
    if (nmax < 1) nmax = 1;

    INT mnmax = (mmax > nmax) ? mmax : nmax;

    /* Allocate workspace structure */
    g_ws = malloc(sizeof(ddrvbd_workspace_t));
    if (!g_ws) return -1;

    g_ws->mmax = mmax;
    g_ws->nmax = nmax;
    g_ws->mnmax = mnmax;
    g_ws->lwork = compute_lwork(mmax, nmax);
    g_ws->liwork = 12 * mnmax;
    g_ws->lrwork = 2 * mnmax;
    rng_seed(g_ws->rng_state, 2024);

    /* Allocate arrays */
    g_ws->A = malloc(mmax * nmax * sizeof(f32));
    g_ws->ASAV = malloc(mmax * nmax * sizeof(f32));
    g_ws->U = malloc(mmax * mmax * sizeof(f32));
    g_ws->USAV = malloc(mmax * mmax * sizeof(f32));
    g_ws->VT = malloc(nmax * nmax * sizeof(f32));
    g_ws->VTSAV = malloc(nmax * nmax * sizeof(f32));
    g_ws->S = malloc(2 * mnmax * sizeof(f32));
    g_ws->SSAV = malloc(2 * mnmax * sizeof(f32));
    g_ws->work = malloc(g_ws->lwork * sizeof(f32));
    g_ws->rwork = malloc(g_ws->lrwork * sizeof(f32));
    g_ws->iwork = malloc(g_ws->liwork * sizeof(INT));

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
    INT m = p->m;
    INT n = p->n;
    INT itype = p->itype;

    /* Skip degenerate cases */
    if (m == 0 || n == 0) return;

    /* Seed based on parameters for reproducibility */
    uint64_t seed = 2024 + m * 1000 + n * 100 + itype;
    rng_seed(ws->rng_state, seed);

    INT info;

    /* Generate test matrix */
    generate_test_matrix(itype, m, n, ws->ASAV, m,
                         ws->SSAV, ws->work, &info, ws->rng_state);

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
    for (INT iws = 1; iws <= 4; iws++) {
        INT mnmin = (m < n) ? m : n;
        INT mxmax = (m > n) ? m : n;

        /* Compute LSWORK for SGESVD (lines 576-581 in ddrvbd.f) */
        INT iwtmp_gesvd = 3 * mnmin + mxmax;
        INT tmp = 5 * mnmin;
        if (tmp > iwtmp_gesvd) iwtmp_gesvd = tmp;

        INT lswork_gesvd = iwtmp_gesvd + (iws - 1) * (ws->lwork - iwtmp_gesvd) / 3;
        if (lswork_gesvd > ws->lwork) lswork_gesvd = ws->lwork;
        if (lswork_gesvd < 1) lswork_gesvd = 1;
        if (iws == 4) lswork_gesvd = ws->lwork;

        /* Compute LSWORK for SGESDD (lines 690-695 in ddrvbd.f) */
        INT iwtmp_gesdd = 5 * mnmin * mnmin + 9 * mnmin + mxmax;
        INT lswork_gesdd = iwtmp_gesdd + (iws - 1) * (ws->lwork - iwtmp_gesdd) / 3;
        if (lswork_gesdd > ws->lwork) lswork_gesdd = ws->lwork;
        if (lswork_gesdd < 1) lswork_gesdd = 1;
        if (iws == 4) lswork_gesdd = ws->lwork;

        /* Initialize results to -1 */
        for (INT i = 0; i < NRESULTS; i++) ws->result[i] = -1.0f;

        /* SGESVD (tests 1-7) */
        test_dgesvd(m, n, ws->ASAV, m, lswork_gesvd, ws);
        for (INT i = 0; i < 7; i++) {
            if (ws->result[i] < 0.0f) continue;  /* Test not run */
            if (ws->result[i] >= THRESH) {
                snprintf(context, sizeof(context),
                         "SGESVD m=%d n=%d type=%d iws=%d test=%d",
                         m, n, itype, iws, i + 1);
                set_test_context(context);
                assert_residual_below(ws->result[i], THRESH);
            }
        }

        /* SGESDD (tests 8-14) */
        test_dgesdd(m, n, ws->ASAV, m, lswork_gesdd, ws);
        for (INT i = 7; i < 14; i++) {
            if (ws->result[i] < 0.0f) continue;
            if (ws->result[i] >= THRESH) {
                snprintf(context, sizeof(context),
                         "SGESDD m=%d n=%d type=%d iws=%d test=%d",
                         m, n, itype, iws, i + 1);
                set_test_context(context);
                assert_residual_below(ws->result[i], THRESH);
            }
        }

        /* SGESVJ (tests 15-18, M >= N only) */
        if (m >= n) {
            test_dgesvj(m, n, ws->ASAV, m, ws);
            for (INT i = 14; i < 18; i++) {
                if (ws->result[i] < 0.0f) continue;
                if (ws->result[i] >= THRESH) {
                    snprintf(context, sizeof(context),
                             "SGESVJ m=%d n=%d type=%d iws=%d test=%d",
                             m, n, itype, iws, i + 1);
                    set_test_context(context);
                    assert_residual_below(ws->result[i], THRESH);
                }
            }
        }

        /* SGEJSV (tests 19-22, M >= N only) */
        if (m >= n) {
            test_dgejsv(m, n, ws->ASAV, m, ws);
            for (INT i = 18; i < 22; i++) {
                if (ws->result[i] < 0.0f) continue;
                if (ws->result[i] >= THRESH) {
                    snprintf(context, sizeof(context),
                             "SGEJSV m=%d n=%d type=%d iws=%d test=%d",
                             m, n, itype, iws, i + 1);
                    set_test_context(context);
                    assert_residual_below(ws->result[i], THRESH);
                }
            }
        }

        /* SGESVDX (tests 23-35) */
        test_dgesvdx(m, n, ws->ASAV, m, ws);
        for (INT i = 22; i < 35; i++) {
            if (ws->result[i] < 0.0f) continue;
            if (ws->result[i] >= THRESH) {
                snprintf(context, sizeof(context),
                         "SGESVDX m=%d n=%d type=%d iws=%d test=%d",
                         m, n, itype, iws, i + 1);
                set_test_context(context);
                assert_residual_below(ws->result[i], THRESH);
            }
        }

        /* SGESVDQ (tests 36-39, M >= N only) */
        if (m >= n) {
            test_dgesvdq(m, n, ws->ASAV, m, lswork_gesdd, ws);
            for (INT i = 35; i < 39; i++) {
                if (ws->result[i] < 0.0f) continue;
                if (ws->result[i] >= THRESH) {
                    snprintf(context, sizeof(context),
                             "SGESVDQ m=%d n=%d type=%d iws=%d test=%d",
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
static INT g_num_tests = 0;

/**
 * Build the test array with all parameter combinations.
 */
static void build_test_array(void)
{
    g_num_tests = 0;

    for (size_t ipair = 0; ipair < NDIM_PAIRS; ipair++) {
        INT m = DIM_PAIRS[ipair].m;
        INT n = DIM_PAIRS[ipair].n;

        for (INT itype = 1; itype <= NTYPES; itype++) {
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
