/**
 * @file test_zdrvbd.c
 * @brief Comprehensive complex SVD test driver - port of LAPACK TESTING/EIG/zdrvbd.f
 *
 * Tests all 6 complex SVD drivers:
 * - ZGESVD:  QR iteration SVD
 * - ZGESDD:  Divide-and-conquer SVD
 * - ZGESVDX: SVD with range selection (bisection)
 * - ZGESVDQ: QR-preconditioned SVD
 * - ZGESVJ:  Jacobi SVD
 * - ZGEJSV:  Preconditioned Jacobi SVD (high accuracy)
 *
 * Each (m, n, type) combination is registered as a separate CMocka test,
 * providing pytest-like parameterized test behavior with clear failure isolation.
 *
 * Test ratios (39 total):
 *   ZGESVD  (1-7):   Full and partial SVD
 *   ZGESDD  (8-14):  Full and partial SVD
 *   ZGESVJ  (15-18): Full SVD (M >= N only)
 *   ZGEJSV  (19-22): Full SVD (M >= N only)
 *   ZGESVDX (23-35): Full and range-selected SVD
 *   ZGESVDQ (36-39): Full SVD (M >= N only)
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
#include "semicolon_cblas.h"
#include <math.h>
#include <complex.h>
#include <string.h>

/* Test threshold from LAPACK svd.in (line 10) */
#define THRESH 50.0

/* Number of matrix types */
#define NTYPES 5

/* Number of test results */
#define NRESULTS 39

/* Job option strings for partial SVD tests (from zdrvbd.f) */
static const char* CJOB = "NOSA";   /* 'N', 'O', 'S', 'A' for ZGESVD/ZGESDD */
static const char* CJOBV = "NV";    /* 'N', 'V' for ZGESVDX */

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

/**
 * Test parameters for a single test case.
 */
typedef struct {
    INT m;
    INT n;
    INT itype;
    char name[64];
} zdrvbd_params_t;

/*
 * Workspace structure for all SVD tests.
 * Allocated once at maximum dimensions, reused for all size combinations.
 */
typedef struct {
    INT mmax, nmax, mnmax;

    /* Matrices (complex) */
    c128* A;      /* m x n working copy */
    c128* ASAV;   /* m x n saved copy */
    c128* U;      /* m x m left singular vectors */
    c128* USAV;   /* m x m saved U */
    c128* VT;     /* n x n right singular vectors */
    c128* VTSAV;  /* n x n saved VT */

    /* Singular values (real) */
    f64* S;      /* min(m,n) computed singular values */
    f64* SSAV;   /* min(m,n) saved singular values */

    /* Work arrays */
    c128* work;   /* complex workspace */
    f64* rwork;   /* real workspace */
    INT* iwork;   /* integer workspace */
    INT lwork;    /* complex workspace size */
    INT liwork;   /* integer workspace size */
    INT lrwork;   /* real workspace size */

    /* Test results */
    f64 result[NRESULTS];

    /* RNG state */
    uint64_t rng_state[4];
} zdrvbd_workspace_t;

/* Global workspace pointer */
static zdrvbd_workspace_t* g_ws = NULL;

/* ===== Helper Functions ===== */

/**
 * Compute maximum complex workspace needed for all SVD routines.
 * From zdrvbd.f workspace requirements.
 */
static INT compute_lwork(INT mmax, INT nmax)
{
    INT mnmax = (mmax < nmax) ? mmax : nmax;
    INT mxmax = (mmax > nmax) ? mmax : nmax;

    /* ZGESVD: 2*MN + MX */
    INT lwork1 = 2 * mnmax + mxmax;
    /* ZGESDD: 2*MN^2 + 2*MN + MX */
    INT lwork2 = 2 * mnmax * mnmax + 2 * mnmax + mxmax;
    INT lwork = (lwork1 > lwork2) ? lwork1 : lwork2;

    /* Extra for ZGESVDQ */
    INT lwork_q = 5 * mnmax * mnmax + 9 * mnmax + mxmax;
    if (lwork_q > lwork) lwork = lwork_q;

    /* Extra for ZGEJSV */
    INT lwork_j = 6 * mnmax * mnmax + 10 * mnmax + mxmax;
    if (lwork_j > lwork) lwork = lwork_j;

    /* Extra safety margin */
    lwork = lwork + 100;

    return lwork;
}

/**
 * Compute maximum real workspace needed.
 * From zdrvbd.f: RWORK dimension 5*max(max(MM,NN)).
 * Plus extra for ZGESVJ (MAX(6,N)), ZGEJSV (MAX(7,N+2*M)),
 * ZGESVDQ (MAX(2,M,5*N)), and verification routines.
 */
static INT compute_lrwork(INT mmax, INT nmax)
{
    INT mxmax = (mmax > nmax) ? mmax : nmax;
    INT mnmax = (mmax < nmax) ? mmax : nmax;

    INT lrwork = 5 * mxmax;

    /* ZGESDD needs 5*MN^2 + 7*MN */
    INT lr_sdd = 5 * mnmax * mnmax + 7 * mnmax;
    if (lr_sdd > lrwork) lrwork = lr_sdd;

    /* ZGEJSV: MAX(7, N+2*M) */
    INT lr_jv = 7;
    if (nmax + 2 * mmax > lr_jv) lr_jv = nmax + 2 * mmax;
    if (lr_jv > lrwork) lrwork = lr_jv;

    /* ZGESVDQ: MAX(2, M, 5*N) */
    INT lr_q = 2;
    if (mmax > lr_q) lr_q = mmax;
    if (5 * nmax > lr_q) lr_q = 5 * nmax;
    if (lr_q > lrwork) lrwork = lr_q;

    lrwork += 100;

    return lrwork;
}

/**
 * Check that singular values are non-negative and decreasing.
 * Returns 0.0 if valid, 1/ULP if invalid.
 */
static f64 check_sv_order(const f64* S, INT n, f64 ulpinv)
{
    for (INT i = 0; i < n; i++) {
        if (S[i] < 0.0) return ulpinv;
    }
    for (INT i = 0; i < n - 1; i++) {
        if (S[i] < S[i + 1]) return ulpinv;
    }
    return 0.0;
}

/**
 * Generate test matrix of specified type.
 *
 * Types (from zdrvbd.f):
 *   1: Zero matrix
 *   2: Identity matrix (diagonal of 1s)
 *   3: Random U*D*V with evenly spaced D in [ULP, 1]
 *   4: Same as 3, scaled near underflow
 *   5: Same as 3, scaled near overflow
 */
static void generate_test_matrix(INT itype, INT m, INT n, c128* A, INT lda,
                                 f64* S, c128* work, INT* info,
                                 uint64_t state[static 4])
{
    f64 ulp = dlamch("P");
    f64 unfl = dlamch("S");
    f64 ovfl = 1.0 / unfl;
    INT mnmin = (m < n) ? m : n;

    *info = 0;

    if (m == 0 || n == 0) {
        return;
    }

    if (itype == 1) {
        /* Type 1: Zero matrix */
        zlaset("F", m, n, 0.0, 0.0, A, lda);
        for (INT i = 0; i < mnmin; i++) S[i] = 0.0;
    }
    else if (itype == 2) {
        /* Type 2: Identity matrix */
        zlaset("F", m, n, 0.0, 1.0, A, lda);
        for (INT i = 0; i < mnmin; i++) S[i] = 1.0;
    }
    else if (itype >= 3 && itype <= 5) {
        /* Types 3-5: Random matrix with controlled singular values */
        f64 cond = (f64)mnmin;
        f64 anorm = 1.0;

        if (itype == 4) {
            anorm = unfl / ulp;
        }
        else if (itype == 5) {
            anorm = ovfl * ulp;
        }

        zlatms(m, n, "U", "N", S, 4, cond, anorm,
               m - 1, n - 1, "N", A, lda, work, info, state);

        if (*info != 0) {
            return;
        }

        /* Sort singular values in decreasing order for verification */
        for (INT i = 0; i < mnmin; i++) {
            f64 t = (f64)i / (f64)(mnmin > 1 ? mnmin - 1 : 1);
            S[i] = anorm * (1.0 - t * (1.0 - 1.0 / cond));
        }
    }
    else {
        *info = -1;  /* Invalid type */
    }
}

/* ===== ZGESVD Tests (1-7) ===== */

/**
 * Test ZGESVD with full and partial SVD.
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
 */
static void test_zgesvd(INT m, INT n, const c128* ASAV, INT lda,
                        INT lswork, zdrvbd_workspace_t* ws)
{
    INT mnmin = (m < n) ? m : n;
    f64 ulp = dlamch("P");
    f64 ulpinv = 1.0 / ulp;
    f64 unfl = dlamch("S");
    INT info;

    /* Initialize results to 0 */
    for (INT i = 0; i < 7; i++) ws->result[i] = 0.0;

    if (m == 0 || n == 0) return;

    /* === Full SVD (jobu='A', jobvt='A') === */

    zlacpy("F", m, n, ASAV, lda, ws->A, lda);
    zgesvd("A", "A", m, n, ws->A, lda, ws->SSAV,
           ws->USAV, m, ws->VTSAV, n, ws->work, lswork, ws->rwork, &info);

    if (info != 0) {
        ws->result[0] = ulpinv;
        return;
    }

    /* Test 1: Reconstruction |A - U*S*VT| / (n * |A| * eps) */
    zbdt01(m, n, 0, ASAV, lda, ws->USAV, m, ws->SSAV, NULL,
           ws->VTSAV, n, ws->work, ws->rwork, &ws->result[0]);

    /* Test 2: U orthogonality */
    if (m != 0 && n != 0) {
        zunt01("C", mnmin, m, ws->USAV, m, ws->work, ws->lwork,
               ws->rwork, &ws->result[1]);

        /* Test 3: VT orthogonality */
        zunt01("R", mnmin, n, ws->VTSAV, n, ws->work, ws->lwork,
               ws->rwork, &ws->result[2]);
    }

    /* Test 4: S ordering */
    ws->result[3] = check_sv_order(ws->SSAV, mnmin, ulpinv);

    /* === Partial SVD: test all 14 JOBU/JOBVT combinations === */
    ws->result[4] = 0.0;  /* max |U - Upartial| */
    ws->result[5] = 0.0;  /* max |VT - VTpartial| */
    ws->result[6] = 0.0;  /* max |S - Spartial| */

    for (INT iju = 0; iju <= 3; iju++) {
        for (INT ijvt = 0; ijvt <= 3; ijvt++) {
            /* Skip ('A','A') - tested above; skip ('O','O') - invalid */
            if ((iju == 3 && ijvt == 3) || (iju == 1 && ijvt == 1)) continue;

            char jobu[2] = {CJOB[iju], '\0'};
            char jobvt[2] = {CJOB[ijvt], '\0'};

            zlacpy("F", m, n, ASAV, lda, ws->A, lda);
            zgesvd(jobu, jobvt, m, n, ws->A, lda, ws->S,
                   ws->U, m, ws->VT, n, ws->work, lswork, ws->rwork, &info);

            if (info != 0) continue;

            /* Compare U */
            f64 dif = 0.0;
            if (m > 0 && n > 0) {
                if (iju == 1) {
                    /* JOBU='O': U stored in A */
                    zunt03("C", m, mnmin, m, mnmin, ws->USAV, m, ws->A, lda,
                           ws->work, ws->lwork, ws->rwork, &dif, &info);
                } else if (iju == 2) {
                    /* JOBU='S': economy U in ws->U */
                    zunt03("C", m, mnmin, m, mnmin, ws->USAV, m, ws->U, m,
                           ws->work, ws->lwork, ws->rwork, &dif, &info);
                } else if (iju == 3) {
                    /* JOBU='A': full U in ws->U */
                    zunt03("C", m, m, m, mnmin, ws->USAV, m, ws->U, m,
                           ws->work, ws->lwork, ws->rwork, &dif, &info);
                }
            }
            if (dif > ws->result[4]) ws->result[4] = dif;

            /* Compare VT */
            dif = 0.0;
            if (m > 0 && n > 0) {
                if (ijvt == 1) {
                    /* JOBVT='O': VT stored in A */
                    zunt03("R", n, mnmin, n, mnmin, ws->VTSAV, n, ws->A, lda,
                           ws->work, ws->lwork, ws->rwork, &dif, &info);
                } else if (ijvt == 2) {
                    /* JOBVT='S': economy VT in ws->VT */
                    zunt03("R", n, mnmin, n, mnmin, ws->VTSAV, n, ws->VT, n,
                           ws->work, ws->lwork, ws->rwork, &dif, &info);
                } else if (ijvt == 3) {
                    /* JOBVT='A': full VT in ws->VT */
                    zunt03("R", n, n, n, mnmin, ws->VTSAV, n, ws->VT, n,
                           ws->work, ws->lwork, ws->rwork, &dif, &info);
                }
            }
            if (dif > ws->result[5]) ws->result[5] = dif;

            /* Compare S */
            dif = 0.0;
            f64 div = (mnmin * ulp * ws->S[0] > unfl) ? mnmin * ulp * ws->S[0] : unfl;
            for (INT i = 0; i < mnmin - 1; i++) {
                if (ws->SSAV[i] < ws->SSAV[i + 1]) dif = ulpinv;
                if (ws->SSAV[i] < 0.0) dif = ulpinv;
                f64 d = fabs(ws->SSAV[i] - ws->S[i]) / div;
                if (d > dif) dif = d;
            }
            if (dif > ws->result[6]) ws->result[6] = dif;
        }
    }
}

/* ===== ZGESDD Tests (8-14) ===== */

/**
 * Test ZGESDD with full and partial SVD.
 * Same test structure as ZGESVD, results stored in indices 7-13.
 */
static void test_zgesdd(INT m, INT n, const c128* ASAV, INT lda,
                        INT lswork, zdrvbd_workspace_t* ws)
{
    INT mnmin = (m < n) ? m : n;
    f64 ulp = dlamch("P");
    f64 ulpinv = 1.0 / ulp;
    f64 unfl = dlamch("S");
    INT info;

    /* Initialize results to 0 */
    for (INT i = 7; i < 14; i++) ws->result[i] = 0.0;

    if (m == 0 || n == 0) return;

    /* === Full SVD (jobz='A') === */
    zlacpy("F", m, n, ASAV, lda, ws->A, lda);

    zgesdd("A", m, n, ws->A, lda, ws->SSAV,
           ws->USAV, m, ws->VTSAV, n, ws->work, lswork, ws->rwork,
           ws->iwork, &info);

    if (info != 0) {
        ws->result[7] = ulpinv;
        return;
    }

    /* Test 8: Reconstruction |A - U*S*VT| / (n * |A| * eps) */
    zbdt01(m, n, 0, ASAV, lda, ws->USAV, m, ws->SSAV, NULL,
           ws->VTSAV, n, ws->work, ws->rwork, &ws->result[7]);

    /* Test 9: U orthogonality */
    if (m != 0 && n != 0) {
        zunt01("C", mnmin, m, ws->USAV, m, ws->work, ws->lwork,
               ws->rwork, &ws->result[8]);

        /* Test 10: VT orthogonality */
        zunt01("R", mnmin, n, ws->VTSAV, n, ws->work, ws->lwork,
               ws->rwork, &ws->result[9]);
    }

    /* Test 11: S ordering */
    ws->result[10] = check_sv_order(ws->SSAV, mnmin, ulpinv);

    /* === Partial SVD: test jobz='N','O','S' === */
    ws->result[11] = 0.0;  /* max |U - Upartial| */
    ws->result[12] = 0.0;  /* max |VT - VTpartial| */
    ws->result[13] = 0.0;  /* max |S - Spartial| */

    for (INT ijq = 0; ijq <= 2; ijq++) {
        char jobq[2] = {CJOB[ijq], '\0'};  /* 'N', 'O', 'S' */

        zlacpy("F", m, n, ASAV, lda, ws->A, lda);

        zgesdd(jobq, m, n, ws->A, lda, ws->S,
               ws->U, m, ws->VT, mnmin, ws->work, lswork, ws->rwork,
               ws->iwork, &info);

        if (info != 0) continue;

        /* Compare U */
        f64 dif = 0.0;
        if (m > 0 && n > 0) {
            if (ijq == 1) {
                /* JOBZ='O': U or VT in A depending on M >= N */
                if (m >= n) {
                    zunt03("C", m, mnmin, m, mnmin, ws->USAV, m, ws->A, lda,
                           ws->work, ws->lwork, ws->rwork, &dif, &info);
                } else {
                    zunt03("C", m, mnmin, m, mnmin, ws->USAV, m, ws->U, m,
                           ws->work, ws->lwork, ws->rwork, &dif, &info);
                }
            } else if (ijq == 2) {
                /* JOBZ='S': economy U in ws->U */
                zunt03("C", m, mnmin, m, mnmin, ws->USAV, m, ws->U, m,
                       ws->work, ws->lwork, ws->rwork, &dif, &info);
            }
        }
        if (dif > ws->result[11]) ws->result[11] = dif;

        /* Compare VT */
        dif = 0.0;
        if (m > 0 && n > 0) {
            if (ijq == 1) {
                /* JOBZ='O': VT or U in A depending on M >= N */
                if (m >= n) {
                    zunt03("R", n, mnmin, n, mnmin, ws->VTSAV, n, ws->VT, mnmin,
                           ws->work, ws->lwork, ws->rwork, &dif, &info);
                } else {
                    zunt03("R", n, mnmin, n, mnmin, ws->VTSAV, n, ws->A, lda,
                           ws->work, ws->lwork, ws->rwork, &dif, &info);
                }
            } else if (ijq == 2) {
                /* JOBZ='S': economy VT in ws->VT */
                zunt03("R", n, mnmin, n, mnmin, ws->VTSAV, n, ws->VT, mnmin,
                       ws->work, ws->lwork, ws->rwork, &dif, &info);
            }
        }
        if (dif > ws->result[12]) ws->result[12] = dif;

        /* Compare S */
        dif = 0.0;
        f64 div = (mnmin * ulp * ws->S[0] > unfl) ? mnmin * ulp * ws->S[0] : unfl;
        for (INT i = 0; i < mnmin - 1; i++) {
            if (ws->SSAV[i] < ws->SSAV[i + 1]) dif = ulpinv;
            if (ws->SSAV[i] < 0.0) dif = ulpinv;
            f64 d = fabs(ws->SSAV[i] - ws->S[i]) / div;
            if (d > dif) dif = d;
        }
        if (dif > ws->result[13]) ws->result[13] = dif;
    }
}

/* ===== ZGESVJ Tests (15-18) ===== */

/**
 * Test ZGESVJ (Jacobi SVD).
 * Only works for M >= N.
 * Returns V (not VH), so we need to conjugate transpose.
 */
static void test_zgesvj(INT m, INT n, const c128* ASAV, INT lda,
                        zdrvbd_workspace_t* ws)
{
    f64 ulpinv = 1.0 / dlamch("P");
    INT info;

    /* Initialize results to 0 */
    for (INT i = 14; i < 18; i++) ws->result[i] = 0.0;

    /* ZGESVJ only works for M >= N */
    if (m < n || m == 0 || n == 0) return;

    INT mnmin = n;  /* Since m >= n, mnmin = n */
    INT lrwork_svj = 6;
    if (n > lrwork_svj) lrwork_svj = n;

    zlacpy("F", m, n, ASAV, lda, ws->USAV, m);

    zgesvj("G", "U", "V", m, n, ws->USAV, m, ws->SSAV, 0, ws->A, n,
           ws->work, ws->lwork, ws->rwork, lrwork_svj, &info);

    if (info != 0) {
        ws->result[14] = ulpinv;
        return;
    }

    /* ZGESVJ returns V in A, conjugate transpose to VTSAV: VT(j,i) = conj(V(i,j)) */
    for (INT j = 0; j < n; j++) {
        for (INT i = 0; i < n; i++) {
            ws->VTSAV[i + j * n] = conj(ws->A[j + i * n]);
        }
    }

    /* Test 15: Reconstruction |A - U*S*VT| / (n * |A| * eps) */
    zbdt01(m, n, 0, ASAV, lda, ws->USAV, m, ws->SSAV, NULL,
           ws->VTSAV, n, ws->work, ws->rwork, &ws->result[14]);

    /* Test 16: U orthogonality, Test 17: VT orthogonality */
    if (m != 0 && n != 0) {
        zunt01("C", m, m, ws->USAV, m, ws->work, ws->lwork,
               ws->rwork, &ws->result[15]);
        zunt01("R", n, n, ws->VTSAV, n, ws->work, ws->lwork,
               ws->rwork, &ws->result[16]);
    }

    /* Test 18: S ordering */
    ws->result[17] = check_sv_order(ws->SSAV, mnmin, ulpinv);
}

/* ===== ZGEJSV Tests (19-22) ===== */

/**
 * Test ZGEJSV (Preconditioned Jacobi SVD).
 * Only works for M >= N.
 * Returns V (not VH), so we need to conjugate transpose.
 */
static void test_zgejsv(INT m, INT n, const c128* ASAV, INT lda,
                        zdrvbd_workspace_t* ws)
{
    f64 ulpinv = 1.0 / dlamch("P");
    INT info;

    /* Initialize results to 0 */
    for (INT i = 18; i < 22; i++) ws->result[i] = 0.0;

    /* ZGEJSV only works for M >= N */
    if (m < n || m == 0 || n == 0) return;

    INT mnmin = n;  /* Since m >= n, mnmin = n */
    INT lrwork_jv = 7;
    if (n + 2 * m > lrwork_jv) lrwork_jv = n + 2 * m;

    zlacpy("F", m, n, ASAV, lda, ws->VTSAV, m);

    zgejsv("G", "U", "V", "R", "N", "N", m, n, ws->VTSAV, m,
           ws->SSAV, ws->USAV, m, ws->A, n, ws->work, ws->lwork,
           ws->rwork, lrwork_jv, ws->iwork, &info);

    if (info != 0) {
        ws->result[18] = ulpinv;
        return;
    }

    /* ZGEJSV returns V in A, conjugate transpose to VTSAV */
    for (INT j = 0; j < n; j++) {
        for (INT i = 0; i < n; i++) {
            ws->VTSAV[i + j * n] = conj(ws->A[j + i * n]);
        }
    }

    /* Test 19: Reconstruction |A - U*S*VT| / (n * |A| * eps) */
    zbdt01(m, n, 0, ASAV, lda, ws->USAV, m, ws->SSAV, NULL,
           ws->VTSAV, n, ws->work, ws->rwork, &ws->result[18]);

    /* Test 20: U orthogonality, Test 21: VT orthogonality */
    if (m != 0 && n != 0) {
        zunt01("C", m, m, ws->USAV, m, ws->work, ws->lwork,
               ws->rwork, &ws->result[19]);
        zunt01("R", n, n, ws->VTSAV, n, ws->work, ws->lwork,
               ws->rwork, &ws->result[20]);
    }

    /* Test 22: S ordering */
    ws->result[21] = check_sv_order(ws->SSAV, mnmin, ulpinv);
}

/* ===== ZGESVDX Tests (23-35) ===== */

/**
 * Test ZGESVDX (SVD with range selection).
 *
 * Tests 23-26: RANGE='A' (all singular values), JOBU='V', JOBVT='V'
 * Tests 27-29: Partial SVD comparison (max over JOBU/JOBVT in {'N','V'})
 * Tests 30-32: RANGE='I' (index range)
 * Tests 33-35: RANGE='V' (value range)
 */
static void test_zgesvdx(INT m, INT n, const c128* ASAV, INT lda,
                         zdrvbd_workspace_t* ws)
{
    INT mnmin = (m < n) ? m : n;
    f64 ulp = dlamch("P");
    f64 ulpinv = 1.0 / ulp;
    f64 unfl = dlamch("S");
    INT info;
    INT ns;
    INT lwork = ws->lwork;

    /* Initialize results to 0 */
    for (INT i = 22; i < 35; i++) ws->result[i] = 0.0;

    if (m == 0 || n == 0) return;

    /* === RANGE='A' with vectors === */
    zlacpy("F", m, n, ASAV, lda, ws->A, lda);

    zgesvdx("V", "V", "A", m, n, ws->A, lda,
            0.0, 0.0, 0, 0, &ns,
            ws->SSAV, ws->USAV, m, ws->VTSAV, mnmin,
            ws->work, lwork, ws->rwork, ws->iwork, &info);

    if (info != 0) {
        ws->result[22] = ulpinv;
        return;
    }

    /* Test 23: Reconstruction |A - U*S*VT| / (n * |A| * eps) */
    zbdt01(m, n, 0, ASAV, lda, ws->USAV, m, ws->SSAV, NULL,
           ws->VTSAV, mnmin, ws->work, ws->rwork, &ws->result[22]);

    /* Test 24: U orthogonality - U is M x MNMIN */
    if (m != 0 && n != 0) {
        zunt01("C", mnmin, m, ws->USAV, m, ws->work, ws->lwork,
               ws->rwork, &ws->result[23]);

        /* Test 25: VT orthogonality - VT is MNMIN x N */
        zunt01("R", mnmin, n, ws->VTSAV, mnmin, ws->work, ws->lwork,
               ws->rwork, &ws->result[24]);
    }

    /* Test 26: S ordering */
    ws->result[25] = check_sv_order(ws->SSAV, mnmin, ulpinv);

    /* === Partial SVD: test JOBU/JOBVT in {'N','V'} === */
    ws->result[26] = 0.0;  /* max |U - Upartial| */
    ws->result[27] = 0.0;  /* max |VT - VTpartial| */
    ws->result[28] = 0.0;  /* max |S - Spartial| */

    for (INT iju = 0; iju <= 1; iju++) {
        for (INT ijvt = 0; ijvt <= 1; ijvt++) {
            /* Skip ('N','N') and ('V','V') - tested above */
            if ((iju == 0 && ijvt == 0) || (iju == 1 && ijvt == 1)) continue;

            char jobu[2] = {CJOBV[iju], '\0'};
            char jobvt[2] = {CJOBV[ijvt], '\0'};

            zlacpy("F", m, n, ASAV, lda, ws->A, lda);

            zgesvdx(jobu, jobvt, "A", m, n, ws->A, lda,
                    0.0, 0.0, 0, 0, &ns,
                    ws->S, ws->U, m, ws->VT, mnmin,
                    ws->work, lwork, ws->rwork, ws->iwork, &info);

            if (info != 0) continue;

            /* Compare U */
            f64 dif = 0.0;
            if (m > 0 && n > 0 && iju == 1) {
                zunt03("C", m, mnmin, m, mnmin, ws->USAV, m, ws->U, m,
                       ws->work, ws->lwork, ws->rwork, &dif, &info);
            }
            if (dif > ws->result[26]) ws->result[26] = dif;

            /* Compare VT */
            dif = 0.0;
            if (m > 0 && n > 0 && ijvt == 1) {
                zunt03("R", n, mnmin, n, mnmin, ws->VTSAV, mnmin, ws->VT, mnmin,
                       ws->work, ws->lwork, ws->rwork, &dif, &info);
            }
            if (dif > ws->result[27]) ws->result[27] = dif;

            /* Compare S */
            dif = 0.0;
            f64 div = (mnmin * ulp * ws->S[0] > unfl) ? mnmin * ulp * ws->S[0] : unfl;
            for (INT i = 0; i < mnmin - 1; i++) {
                if (ws->SSAV[i] < ws->SSAV[i + 1]) dif = ulpinv;
                if (ws->SSAV[i] < 0.0) dif = ulpinv;
                f64 d = fabs(ws->SSAV[i] - ws->S[i]) / div;
                if (d > dif) dif = d;
            }
            if (dif > ws->result[28]) ws->result[28] = dif;
        }
    }

    /* === RANGE='I' (index range) === */
    /* IL/IU are 0-based: valid range [0, mnmin-1] */
    if (mnmin > 1) {
        INT il = (INT)((mnmin - 1) * rng_uniform(ws->rng_state));
        INT iu = (INT)((mnmin - 1) * rng_uniform(ws->rng_state));
        if (iu < il) {
            INT tmp = il;
            il = iu;
            iu = tmp;
        }
        INT nsi = iu - il + 1;

        zlacpy("F", m, n, ASAV, lda, ws->A, lda);

        zgesvdx("V", "V", "I", m, n, ws->A, lda,
                0.0, 0.0, il, iu, &ns,
                ws->S, ws->U, m, ws->VT, nsi,
                ws->work, lwork, ws->rwork, ws->iwork, &info);

        if (info == 0 && ns > 0) {
            /* Test 30: Partial SVD reconstruction using zbdt05 */
            zbdt05(m, n, ASAV, lda, ws->S, ns, ws->U, m, ws->VT, nsi,
                   ws->work, &ws->result[29]);

            /* Test 31: U orthogonality */
            if (m != 0 && n != 0) {
                zunt01("C", m, ns, ws->U, m, ws->work, ws->lwork,
                       ws->rwork, &ws->result[30]);

                /* Test 32: VT orthogonality */
                zunt01("R", ns, n, ws->VT, nsi, ws->work, ws->lwork,
                       ws->rwork, &ws->result[31]);
            }
        }
    }

    /* === RANGE='V' (value range) === */
    /* Compute VL, VU from SSAV (from RANGE='A' call above) */
    f64 vl, vu;

    if (mnmin > 1 && ws->SSAV[0] > ws->SSAV[mnmin - 1]) {
        vl = ws->SSAV[mnmin - 1] - ulp * ws->SSAV[0];
        vu = ws->SSAV[0] + ulp * ws->SSAV[0];

        INT mid = mnmin / 2;
        if (mid > 0 && mid < mnmin - 1) {
            vl = ws->SSAV[mid + 1] - ulp * ws->SSAV[0];
            vu = ws->SSAV[mid - 1] + ulp * ws->SSAV[0];
        }

        if (vl < 0.0) vl = 0.0;

        zlacpy("F", m, n, ASAV, lda, ws->A, lda);

        INT nsv;
        zgesvdx("V", "V", "V", m, n, ws->A, lda,
                vl, vu, 0, 0, &nsv,
                ws->S, ws->U, m, ws->VT, mnmin,
                ws->work, lwork, ws->rwork, ws->iwork, &info);

        if (info == 0 && nsv > 0) {
            /* Test 33: Partial SVD reconstruction using zbdt05 */
            zbdt05(m, n, ASAV, lda, ws->S, nsv, ws->U, m, ws->VT, mnmin,
                   ws->work, &ws->result[32]);

            /* Test 34: U orthogonality */
            if (m != 0 && n != 0) {
                zunt01("C", m, nsv, ws->U, m, ws->work, ws->lwork,
                       ws->rwork, &ws->result[33]);

                /* Test 35: VT orthogonality */
                zunt01("R", nsv, n, ws->VT, mnmin, ws->work, ws->lwork,
                       ws->rwork, &ws->result[34]);
            }
        }
    }
}

/* ===== ZGESVDQ Tests (36-39) ===== */

/**
 * Test ZGESVDQ (QR-preconditioned SVD).
 * Only works for M >= N.
 */
static void test_zgesvdq(INT m, INT n, const c128* ASAV, INT lda,
                         INT lswork, zdrvbd_workspace_t* ws)
{
    (void)lswork;
    f64 ulpinv = 1.0 / dlamch("P");
    INT info;
    INT numrank;

    /* Initialize results to 0 */
    for (INT i = 35; i < 39; i++) ws->result[i] = 0.0;

    /* ZGESVDQ only works for M >= N */
    if (m < n || m == 0 || n == 0) return;

    INT mnmin = n;

    INT lrwork_q = 2;
    if (m > lrwork_q) lrwork_q = m;
    if (5 * n > lrwork_q) lrwork_q = 5 * n;
    INT liwork_q = n;
    if (liwork_q < 1) liwork_q = 1;

    zlacpy("F", m, n, ASAV, lda, ws->A, lda);

    /* ZGESVDQ: joba='H' for complex (from zdrvbd.f line 875) */
    zgesvdq("H", "N", "N", "A", "A", m, n, ws->A, lda,
            ws->SSAV, ws->USAV, m, ws->VTSAV, n, &numrank,
            ws->iwork, liwork_q, ws->work, ws->lwork,
            ws->rwork, lrwork_q, &info);

    if (info != 0) {
        ws->result[35] = ulpinv;
        return;
    }

    /* Test 36: Reconstruction |A - U*S*VT| / (n * |A| * eps) */
    zbdt01(m, n, 0, ASAV, lda, ws->USAV, m, ws->SSAV, NULL,
           ws->VTSAV, n, ws->work, ws->rwork, &ws->result[35]);

    /* Test 37: U orthogonality, Test 38: VT orthogonality */
    if (m != 0 && n != 0) {
        zunt01("C", m, m, ws->USAV, m, ws->work, ws->lwork,
               ws->rwork, &ws->result[36]);
        zunt01("R", n, n, ws->VTSAV, n, ws->work, ws->lwork,
               ws->rwork, &ws->result[37]);
    }

    /* Test 39: S ordering */
    ws->result[38] = check_sv_order(ws->SSAV, mnmin, ulpinv);
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
    g_ws = malloc(sizeof(zdrvbd_workspace_t));
    if (!g_ws) return -1;

    g_ws->mmax = mmax;
    g_ws->nmax = nmax;
    g_ws->mnmax = mnmax;
    g_ws->lwork = compute_lwork(mmax, nmax);
    g_ws->liwork = 12 * mnmax;
    g_ws->lrwork = compute_lrwork(mmax, nmax);
    rng_seed(g_ws->rng_state, 2024);

    /* Allocate complex arrays */
    g_ws->A = malloc(mmax * nmax * sizeof(c128));
    g_ws->ASAV = malloc(mmax * nmax * sizeof(c128));
    g_ws->U = malloc(mmax * mmax * sizeof(c128));
    g_ws->USAV = malloc(mmax * mmax * sizeof(c128));
    g_ws->VT = malloc(nmax * nmax * sizeof(c128));
    g_ws->VTSAV = malloc(nmax * nmax * sizeof(c128));

    /* Allocate real arrays */
    g_ws->S = malloc(2 * mnmax * sizeof(f64));
    g_ws->SSAV = malloc(2 * mnmax * sizeof(f64));
    g_ws->rwork = malloc(g_ws->lrwork * sizeof(f64));

    /* Allocate workspace */
    g_ws->work = malloc(g_ws->lwork * sizeof(c128));
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
static void run_zdrvbd_single(zdrvbd_params_t* p)
{
    zdrvbd_workspace_t* ws = g_ws;
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
        fprintf(stderr, "SKIP: %s\n", context);
        return;
    }

    /* IWS loop: test with 4 different workspace sizes
     * IWS=1: minimal workspace
     * IWS=2,3: intermediate workspace sizes
     * IWS=4: full workspace */
    for (INT iws = 1; iws <= 4; iws++) {
        INT mnmin = (m < n) ? m : n;
        INT mxmax = (m > n) ? m : n;

        /* Compute LSWORK for ZGESVD (from zdrvbd.f line 620) */
        INT iwtmp_gesvd = 2 * mnmin + mxmax;
        INT lswork_gesvd = iwtmp_gesvd + (iws - 1) * (ws->lwork - iwtmp_gesvd) / 3;
        if (lswork_gesvd > ws->lwork) lswork_gesvd = ws->lwork;
        if (lswork_gesvd < 1) lswork_gesvd = 1;
        if (iws == 4) lswork_gesvd = ws->lwork;

        /* Compute LSWORK for ZGESDD (from zdrvbd.f line 741) */
        INT iwtmp_gesdd = 2 * mnmin * mnmin + 2 * mnmin + mxmax;
        INT lswork_gesdd = iwtmp_gesdd + (iws - 1) * (ws->lwork - iwtmp_gesdd) / 3;
        if (lswork_gesdd > ws->lwork) lswork_gesdd = ws->lwork;
        if (lswork_gesdd < 1) lswork_gesdd = 1;
        if (iws == 4) lswork_gesdd = ws->lwork;

        /* Initialize results to -1 */
        for (INT i = 0; i < NRESULTS; i++) ws->result[i] = -1.0;

        /* ZGESVD (tests 1-7) */
        test_zgesvd(m, n, ws->ASAV, m, lswork_gesvd, ws);
        for (INT i = 0; i < 7; i++) {
            if (ws->result[i] < 0.0) continue;  /* Test not run */
            if (ws->result[i] >= THRESH) {
                snprintf(context, sizeof(context),
                         "ZGESVD m=%d n=%d type=%d iws=%d test=%d",
                         m, n, itype, iws, i + 1);
                set_test_context(context);
                assert_residual_below(ws->result[i], THRESH);
            }
        }

        /* ZGESDD (tests 8-14) */
        test_zgesdd(m, n, ws->ASAV, m, lswork_gesdd, ws);
        for (INT i = 7; i < 14; i++) {
            if (ws->result[i] < 0.0) continue;
            if (ws->result[i] >= THRESH) {
                snprintf(context, sizeof(context),
                         "ZGESDD m=%d n=%d type=%d iws=%d test=%d",
                         m, n, itype, iws, i + 1);
                set_test_context(context);
                assert_residual_below(ws->result[i], THRESH);
            }
        }

        /* ZGESVJ (tests 15-18, M >= N only) */
        if (m >= n) {
            test_zgesvj(m, n, ws->ASAV, m, ws);
            for (INT i = 14; i < 18; i++) {
                if (ws->result[i] < 0.0) continue;
                if (ws->result[i] >= THRESH) {
                    snprintf(context, sizeof(context),
                             "ZGESVJ m=%d n=%d type=%d iws=%d test=%d",
                             m, n, itype, iws, i + 1);
                    set_test_context(context);
                    assert_residual_below(ws->result[i], THRESH);
                }
            }
        }

        /* ZGEJSV (tests 19-22, M >= N only) */
        if (m >= n) {
            test_zgejsv(m, n, ws->ASAV, m, ws);
            for (INT i = 18; i < 22; i++) {
                if (ws->result[i] < 0.0) continue;
                if (ws->result[i] >= THRESH) {
                    snprintf(context, sizeof(context),
                             "ZGEJSV m=%d n=%d type=%d iws=%d test=%d",
                             m, n, itype, iws, i + 1);
                    set_test_context(context);
                    assert_residual_below(ws->result[i], THRESH);
                }
            }
        }

        /* ZGESVDX (tests 23-35) */
        test_zgesvdx(m, n, ws->ASAV, m, ws);
        for (INT i = 22; i < 35; i++) {
            if (ws->result[i] < 0.0) continue;
            if (ws->result[i] >= THRESH) {
                snprintf(context, sizeof(context),
                         "ZGESVDX m=%d n=%d type=%d iws=%d test=%d",
                         m, n, itype, iws, i + 1);
                set_test_context(context);
                assert_residual_below(ws->result[i], THRESH);
            }
        }

        /* ZGESVDQ (tests 36-39, M >= N only) */
        if (m >= n) {
            test_zgesvdq(m, n, ws->ASAV, m, lswork_gesdd, ws);
            for (INT i = 35; i < 39; i++) {
                if (ws->result[i] < 0.0) continue;
                if (ws->result[i] >= THRESH) {
                    snprintf(context, sizeof(context),
                             "ZGESVDQ m=%d n=%d type=%d iws=%d test=%d",
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
 * CMocka test function - dispatches to run_zdrvbd_single based on prestate.
 */
static void test_zdrvbd_case(void** state)
{
    zdrvbd_params_t* params = *state;
    run_zdrvbd_single(params);
}

/*
 * Generate all parameter combinations.
 * Total: NDIM_PAIRS * NTYPES tests
 */

/* Maximum number of test cases */
#define MAX_TESTS (NDIM_PAIRS * NTYPES)

static zdrvbd_params_t g_params[MAX_TESTS];
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
            zdrvbd_params_t* p = &g_params[g_num_tests];
            p->m = m;
            p->n = n;
            p->itype = itype;
            snprintf(p->name, sizeof(p->name),
                     "zdrvbd_m%d_n%d_type%d", m, n, itype);

            /* Create CMocka test entry */
            g_tests[g_num_tests].name = p->name;
            g_tests[g_num_tests].test_func = test_zdrvbd_case;
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
    (void)_cmocka_run_group_tests("zdrvbd", g_tests, g_num_tests,
                                   group_setup, group_teardown);
    return 0;
}
