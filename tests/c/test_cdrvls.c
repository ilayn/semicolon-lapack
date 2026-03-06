/**
 * @file test_cdrvls.c
 * @brief ZDRVLS tests the least squares driver routines CGELS, CGELST,
 *        CGETSLS, CGELSS, CGELSY and CGELSD.
 *
 * Port of LAPACK TESTING/LIN/zdrvls.f to C with CMocka parameterization.
 */

#include "test_harness.h"
#include "verify.h"
#include "test_rng.h"
#include "semicolon_cblas.h"
#include <string.h>
#include <stdio.h>
#include <math.h>

/* Test parameters from ztest.in */
static const INT MVAL[] = {0, 1, 2, 3, 5, 10, 50};
static const INT NVAL[] = {0, 1, 2, 3, 5, 10, 50};
static const INT NSVAL[] = {1, 2, 15};

#define NM      (sizeof(MVAL) / sizeof(MVAL[0]))
#define NN      (sizeof(NVAL) / sizeof(NVAL[0]))
#define NNS     (sizeof(NSVAL) / sizeof(NSVAL[0]))
#define NTYPES  6
#define NTESTS  18
#define THRESH  30.0f
#define MMAX    50
#define NMAX    50
#define NSMAX   15
#define SMLSIZ  25

typedef struct {
    INT m, n, nrhs;
    INT irank;
    INT iscale;
    char name[80];
} zdrvls_params_t;

typedef struct {
    c64* A;
    c64* COPYA;
    c64* B;
    c64* COPYB;
    c64* C;
    f32*  S;
    f32*  COPYS;
    c64* WORK;
    f32*  WORK2;
    f32*  RWORK;
    INT*  IWORK;
    INT   lwork;
    INT   lrwork;
} zdrvls_workspace_t;

static zdrvls_workspace_t* g_workspace = NULL;

static INT imax2(INT a, INT b) { return (a > b) ? a : b; }
static INT imin2(INT a, INT b) { return (a < b) ? a : b; }
static INT imax3(INT a, INT b, INT c) { return imax2(a, imax2(b, c)); }

static int group_setup(void** state)
{
    (void)state;
    g_workspace = malloc(sizeof(zdrvls_workspace_t));
    if (!g_workspace) return -1;

    INT m = MMAX;
    INT n = NMAX;
    INT nrhs = NSMAX;
    INT mnmin = imax2(imin2(m, n), 1);
    INT ldb = imax3(1, m, n);

    /*
     * Compute workspace needed for routines
     * CQRT14, CQRT17 (two side cases), CQRT15 and CQRT12
     */
    INT lwork = imax2(1,
                imax2((m + n) * nrhs,
                imax2((n + nrhs) * (m + 2),
                imax2((m + nrhs) * (n + 2),
                imax2(imax3(m + mnmin, nrhs * mnmin, 2 * n + m),
                      imax2(m * n + 4 * mnmin + imax2(m, n),
                            m * n + 2 * mnmin + 4 * n))))));
    INT lrwork = 1;
    INT liwork = 1;

    /*
     * Iterate through all test cases and compute necessary workspace
     * sizes for CGELS, CGELST, CGETSLS, CGELSY, CGELSS and CGELSD
     * routines.
     */
    for (INT im = 0; im < (INT)NM; im++) {
        m = MVAL[im];
        INT lda = imax2(1, m);
        for (INT in = 0; in < (INT)NN; in++) {
            n = NVAL[in];
            mnmin = imax2(imin2(m, n), 1);
            ldb = imax3(1, m, n);
            for (INT ins = 0; ins < (INT)NNS; ins++) {
                nrhs = NSVAL[ins];
                for (INT irank = 1; irank <= 2; irank++) {
                    for (INT iscale = 1; iscale <= 3; iscale++) {
                        if (irank == 1) {
                            for (INT itran = 0; itran < 2; itran++) {
                                const char* trans = (itran == 0) ? "N" : "C";
                                c64 wq;
                                INT info;

                                /* Workspace needed for CGELS */
                                cgels(trans, m, n, nrhs, NULL, lda,
                                      NULL, ldb, &wq, -1, &info);
                                INT lwork_zgels = (INT)crealf(wq);

                                /* Workspace needed for CGELST */
                                cgelst(trans, m, n, nrhs, NULL, lda,
                                       NULL, ldb, &wq, -1, &info);
                                INT lwork_zgelst = (INT)crealf(wq);

                                /* Workspace needed for CGETSLS */
                                cgetsls(trans, m, n, nrhs, NULL, lda,
                                        NULL, ldb, &wq, -1, &info);
                                INT lwork_zgetsls = (INT)crealf(wq);

                                lwork = imax2(lwork,
                                        imax3(lwork_zgels, lwork_zgelst,
                                              lwork_zgetsls));
                            }
                        }
                        c64 wq;
                        f32 rwq;
                        INT iwq;
                        INT info;
                        INT crank;
                        f32 rcond = sqrtf(FLT_EPSILON) -
                                    (sqrtf(FLT_EPSILON) - FLT_EPSILON) / 2.0f;

                        /* Workspace needed for CGELSY */
                        cgelsy(m, n, nrhs, NULL, lda, NULL, ldb, &iwq,
                               rcond, &crank, &wq, -1, &rwq, &info);
                        INT lwork_zgelsy = (INT)crealf(wq);
                        INT lrwork_zgelsy = 2 * n;

                        /* Workspace needed for CGELSS */
                        cgelss(m, n, nrhs, NULL, lda, NULL, ldb, NULL,
                               rcond, &crank, &wq, -1, &rwq, &info);
                        INT lwork_zgelss = (INT)crealf(wq);
                        INT lrwork_zgelss = 5 * mnmin;

                        /* Workspace needed for CGELSD */
                        cgelsd(m, n, nrhs, NULL, lda, NULL, ldb, NULL,
                               rcond, &crank, &wq, -1, &rwq, &iwq,
                               &info);
                        INT lwork_zgelsd = (INT)crealf(wq);
                        INT lrwork_zgelsd = (INT)rwq;

                        liwork = imax3(liwork, n, iwq);
                        lrwork = imax2(lrwork,
                                 imax3(lrwork_zgelsy, lrwork_zgelss,
                                       lrwork_zgelsd));
                        lwork = imax2(lwork,
                                imax3(lwork_zgelsy, lwork_zgelss,
                                      lwork_zgelsd));
                    }
                }
            }
        }
    }

    g_workspace->lwork = lwork;
    g_workspace->lrwork = lrwork;
    g_workspace->A = calloc((size_t)MMAX * NMAX, sizeof(c64));
    g_workspace->COPYA = calloc((size_t)MMAX * NMAX, sizeof(c64));
    g_workspace->B = calloc((size_t)imax2(MMAX, NMAX) * NSMAX, sizeof(c64));
    g_workspace->COPYB = calloc((size_t)imax2(MMAX, NMAX) * NSMAX, sizeof(c64));
    g_workspace->C = calloc((size_t)imax2(MMAX, NMAX) * NSMAX, sizeof(c64));
    g_workspace->S = calloc(imin2(MMAX, NMAX), sizeof(f32));
    g_workspace->COPYS = calloc(imin2(MMAX, NMAX), sizeof(f32));
    g_workspace->WORK = calloc(lwork, sizeof(c64));
    g_workspace->WORK2 = calloc(2 * lwork, sizeof(f32));
    g_workspace->RWORK = calloc(lrwork, sizeof(f32));
    g_workspace->IWORK = calloc(liwork, sizeof(INT));

    if (!g_workspace->A || !g_workspace->COPYA ||
        !g_workspace->B || !g_workspace->COPYB || !g_workspace->C ||
        !g_workspace->S || !g_workspace->COPYS ||
        !g_workspace->WORK || !g_workspace->WORK2 ||
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
        free(g_workspace->COPYA);
        free(g_workspace->B);
        free(g_workspace->COPYB);
        free(g_workspace->C);
        free(g_workspace->S);
        free(g_workspace->COPYS);
        free(g_workspace->WORK);
        free(g_workspace->WORK2);
        free(g_workspace->RWORK);
        free(g_workspace->IWORK);
        free(g_workspace);
        g_workspace = NULL;
    }
    return 0;
}

static void run_zdrvls_single(INT m, INT n, INT nrhs, INT irank, INT iscale)
{
    zdrvls_workspace_t* ws = g_workspace;
    const INT lwork = ws->lwork;

    f32 eps = FLT_EPSILON;
    f32 rcond = sqrtf(eps) - (sqrtf(eps) - eps) / 2.0f;
    INT itype = (irank - 1) * 3 + iscale;

    INT lda = imax2(1, m);
    INT ldb = imax3(1, m, n);
    INT mnmin = imax2(imin2(m, n), 1);

    f32 result[NTESTS];
    for (INT k = 0; k < NTESTS; k++) result[k] = 0.0f;

    uint64_t rng_state[4];
    rng_seed(rng_state, 1988 + m * 10000 + n * 1000 + nrhs * 100 +
             irank * 10 + iscale);

    f32 norma, normb;
    INT info;
    INT rank;

    /* =====================================================
     *       Begin test CGELS
     * ===================================================== */
    if (irank == 1) {

        /* Generate a matrix of scaling type ISCALE */
        cqrt13(iscale, m, n, ws->COPYA, lda, &norma, rng_state);

        for (INT itran = 0; itran < 2; itran++) {
            const char* trans = (itran == 0) ? "N" : "C";
            INT nrows = (itran == 0) ? m : n;
            INT ncols = (itran == 0) ? n : m;
            INT ldwork = imax2(1, ncols);

            /* Set up a consistent rhs */
            if (ncols > 0) {
                clarnv_rng(2, ncols * nrhs, ws->WORK, rng_state);
                cblas_csscal(ncols * nrhs, 1.0f / (f32)ncols, ws->WORK, 1);
            }
            c64 cone = CMPLXF(1.0f, 0.0f);
            c64 czero = CMPLXF(0.0f, 0.0f);
            cblas_cgemm(CblasColMajor,
                        (itran == 0) ? CblasNoTrans : CblasConjTrans,
                        CblasNoTrans,
                        nrows, nrhs, ncols,
                        &cone, ws->COPYA, lda,
                        ws->WORK, ldwork,
                        &czero, ws->B, ldb);
            clacpy("Full", nrows, nrhs, ws->B, ldb, ws->COPYB, ldb);

            /* Solve LS or overdetermined system */
            if (m > 0 && n > 0) {
                clacpy("Full", m, n, ws->COPYA, lda, ws->A, lda);
                clacpy("Full", nrows, nrhs, ws->COPYB, ldb, ws->B, ldb);
            }
            cgels(trans, m, n, nrhs, ws->A, lda, ws->B, ldb,
                  ws->WORK, lwork, &info);

            if (info != 0) {
                fail_msg("CGELS info=%lld trans=%s m=%lld n=%lld nrhs=%lld type%lld",
                         (long long)info, trans, (long long)m, (long long)n,
                         (long long)nrhs, (long long)itype);
                return;
            }

            /* Test 1: residual norm(B - A*X) / ... */
            if (nrows > 0 && nrhs > 0)
                clacpy("Full", nrows, nrhs, ws->COPYB, ldb, ws->C, ldb);
            cqrt16(trans, m, n, nrhs, ws->COPYA, lda, ws->B, ldb,
                   ws->C, ldb, ws->RWORK, &result[itran * 2 + 0]);

            /* Test 2: check correctness of results */
            if ((itran == 0 && m >= n) ||
                (itran == 1 && m < n)) {
                /* Solving LS system */
                result[itran * 2 + 1] = cqrt17(trans, 1, m, n, nrhs,
                                               ws->COPYA, lda, ws->B, ldb,
                                               ws->COPYB, ldb, ws->C,
                                               ws->WORK, lwork);
            } else {
                /* Solving overdetermined system */
                result[itran * 2 + 1] = cqrt14(trans, m, n, nrhs,
                                               ws->COPYA, lda, ws->B, ldb,
                                               ws->WORK, lwork);
            }

            for (INT k = itran * 2; k < itran * 2 + 2; k++) {
                if (result[k] >= THRESH) {
                    fail_msg("CGELS trans=%s m=%lld n=%lld nrhs=%lld type%lld "
                             "test(%lld)=%.6e",
                             trans, (long long)m, (long long)n,
                             (long long)nrhs, (long long)itype,
                             (long long)(k + 1), (double)result[k]);
                }
            }
        }
    }
    /* =====================================================
     *       End test CGELS
     * ===================================================== */

    /* =====================================================
     *       Begin test CGELST
     * ===================================================== */
    if (irank == 1) {

        /* Generate a matrix of scaling type ISCALE */
        cqrt13(iscale, m, n, ws->COPYA, lda, &norma, rng_state);

        for (INT itran = 0; itran < 2; itran++) {
            const char* trans = (itran == 0) ? "N" : "C";
            INT nrows = (itran == 0) ? m : n;
            INT ncols = (itran == 0) ? n : m;
            INT ldwork = imax2(1, ncols);

            /* Set up a consistent rhs */
            if (ncols > 0) {
                clarnv_rng(2, ncols * nrhs, ws->WORK, rng_state);
                cblas_csscal(ncols * nrhs, 1.0f / (f32)ncols, ws->WORK, 1);
            }
            c64 cone = CMPLXF(1.0f, 0.0f);
            c64 czero = CMPLXF(0.0f, 0.0f);
            cblas_cgemm(CblasColMajor,
                        (itran == 0) ? CblasNoTrans : CblasConjTrans,
                        CblasNoTrans,
                        nrows, nrhs, ncols,
                        &cone, ws->COPYA, lda,
                        ws->WORK, ldwork,
                        &czero, ws->B, ldb);
            clacpy("Full", nrows, nrhs, ws->B, ldb, ws->COPYB, ldb);

            /* Solve LS or overdetermined system */
            if (m > 0 && n > 0) {
                clacpy("Full", m, n, ws->COPYA, lda, ws->A, lda);
                clacpy("Full", nrows, nrhs, ws->COPYB, ldb, ws->B, ldb);
            }
            cgelst(trans, m, n, nrhs, ws->A, lda, ws->B, ldb,
                   ws->WORK, lwork, &info);

            if (info != 0) {
                fail_msg("CGELST info=%lld trans=%s m=%lld n=%lld nrhs=%lld type%lld",
                         (long long)info, trans, (long long)m, (long long)n,
                         (long long)nrhs, (long long)itype);
                return;
            }

            /* Test 3: residual */
            if (nrows > 0 && nrhs > 0)
                clacpy("Full", nrows, nrhs, ws->COPYB, ldb, ws->C, ldb);
            cqrt16(trans, m, n, nrhs, ws->COPYA, lda, ws->B, ldb,
                   ws->C, ldb, ws->RWORK, &result[2 + itran * 2 + 0]);

            /* Test 4 */
            if ((itran == 0 && m >= n) ||
                (itran == 1 && m < n)) {
                result[2 + itran * 2 + 1] = cqrt17(trans, 1, m, n, nrhs,
                                                    ws->COPYA, lda,
                                                    ws->B, ldb,
                                                    ws->COPYB, ldb, ws->C,
                                                    ws->WORK, lwork);
            } else {
                result[2 + itran * 2 + 1] = cqrt14(trans, m, n, nrhs,
                                                    ws->COPYA, lda,
                                                    ws->B, ldb,
                                                    ws->WORK, lwork);
            }

            for (INT k = 2 + itran * 2; k < 2 + itran * 2 + 2; k++) {
                if (result[k] >= THRESH) {
                    fail_msg("CGELST trans=%s m=%lld n=%lld nrhs=%lld type%lld "
                             "test(%lld)=%.6e",
                             trans, (long long)m, (long long)n,
                             (long long)nrhs, (long long)itype,
                             (long long)(k + 1), (double)result[k]);
                }
            }
        }
    }
    /* =====================================================
     *       End test CGELST
     * ===================================================== */

    /* =====================================================
     *       Begin test CGETSLS
     * ===================================================== */
    if (irank == 1) {

        /* Generate a matrix of scaling type ISCALE */
        cqrt13(iscale, m, n, ws->COPYA, lda, &norma, rng_state);

        for (INT itran = 0; itran < 2; itran++) {
            const char* trans = (itran == 0) ? "N" : "C";
            INT nrows = (itran == 0) ? m : n;
            INT ncols = (itran == 0) ? n : m;
            INT ldwork = imax2(1, ncols);

            /* Set up a consistent rhs */
            if (ncols > 0) {
                clarnv_rng(2, ncols * nrhs, ws->WORK, rng_state);
                c64 scale = CMPLXF(1.0f / (f32)ncols, 0.0f);
                cblas_cscal(ncols * nrhs, &scale, ws->WORK, 1);
            }
            c64 cone = CMPLXF(1.0f, 0.0f);
            c64 czero = CMPLXF(0.0f, 0.0f);
            cblas_cgemm(CblasColMajor,
                        (itran == 0) ? CblasNoTrans : CblasConjTrans,
                        CblasNoTrans,
                        nrows, nrhs, ncols,
                        &cone, ws->COPYA, lda,
                        ws->WORK, ldwork,
                        &czero, ws->B, ldb);
            clacpy("Full", nrows, nrhs, ws->B, ldb, ws->COPYB, ldb);

            /* Solve LS or overdetermined system */
            if (m > 0 && n > 0) {
                clacpy("Full", m, n, ws->COPYA, lda, ws->A, lda);
                clacpy("Full", nrows, nrhs, ws->COPYB, ldb, ws->B, ldb);
            }
            cgetsls(trans, m, n, nrhs, ws->A, lda, ws->B, ldb,
                    ws->WORK, lwork, &info);

            if (info != 0) {
                fail_msg("CGETSLS info=%lld trans=%s m=%lld n=%lld nrhs=%lld type%lld",
                         (long long)info, trans, (long long)m, (long long)n,
                         (long long)nrhs, (long long)itype);
                return;
            }

            /* Test 5: residual — uses WORK2 for rwork param */
            if (nrows > 0 && nrhs > 0)
                clacpy("Full", nrows, nrhs, ws->COPYB, ldb, ws->C, ldb);
            cqrt16(trans, m, n, nrhs, ws->COPYA, lda, ws->B, ldb,
                   ws->C, ldb, ws->WORK2, &result[4 + itran * 2 + 0]);

            /* Test 6 */
            if ((itran == 0 && m >= n) ||
                (itran == 1 && m < n)) {
                result[4 + itran * 2 + 1] = cqrt17(trans, 1, m, n, nrhs,
                                                    ws->COPYA, lda,
                                                    ws->B, ldb,
                                                    ws->COPYB, ldb, ws->C,
                                                    ws->WORK, lwork);
            } else {
                result[4 + itran * 2 + 1] = cqrt14(trans, m, n, nrhs,
                                                    ws->COPYA, lda,
                                                    ws->B, ldb,
                                                    ws->WORK, lwork);
            }

            for (INT k = 4 + itran * 2; k < 4 + itran * 2 + 2; k++) {
                if (result[k] >= THRESH) {
                    fail_msg("CGETSLS trans=%s m=%lld n=%lld nrhs=%lld type%lld "
                             "test(%lld)=%.6e",
                             trans, (long long)m, (long long)n,
                             (long long)nrhs, (long long)itype,
                             (long long)(k + 1), (double)result[k]);
                }
            }
        }
    }
    /* =====================================================
     *       End test CGETSLS
     * ===================================================== */

    /*
     * Generate a matrix of scaling type ISCALE and rank type IRANK.
     */
    cqrt15(iscale, irank, m, n, nrhs, ws->COPYA, lda,
           ws->COPYB, ldb, ws->COPYS, &rank, &norma, &normb,
           ws->WORK, lwork, rng_state);

    INT ldwork = imax2(1, m);

    /* =====================================================
     *       Test CGELSY
     * ===================================================== */

    /* CGELSY: Compute the minimum-norm solution X to
     * min( norm( A * X - B ) ) using the rank-revealing
     * orthogonal factorization. */

    clacpy("Full", m, n, ws->COPYA, lda, ws->A, lda);
    clacpy("Full", m, nrhs, ws->COPYB, ldb, ws->B, ldb);

    /* Initialize vector IWORK */
    for (INT j = 0; j < n; j++)
        ws->IWORK[j] = 0;

    INT crank;
    cgelsy(m, n, nrhs, ws->A, lda, ws->B, ldb, ws->IWORK,
           rcond, &crank, ws->WORK, lwork, ws->RWORK, &info);

    if (info != 0) {
        fail_msg("CGELSY info=%lld m=%lld n=%lld nrhs=%lld type%lld",
                 (long long)info, (long long)m, (long long)n,
                 (long long)nrhs, (long long)itype);
        return;
    }

    /* Test 7: Compute relative error in svd */
    result[6] = cqrt12(crank, crank, ws->A, lda, ws->COPYS,
                        ws->WORK, lwork, ws->RWORK);

    /* Test 8: Compute error in solution */
    clacpy("Full", m, nrhs, ws->COPYB, ldb, ws->WORK, ldwork);
    cqrt16("N", m, n, nrhs, ws->COPYA, lda, ws->B, ldb,
           ws->WORK, ldwork, ws->RWORK, &result[7]);

    /* Test 9: Check norm of r'*A */
    result[8] = 0.0f;
    if (m > crank)
        result[8] = cqrt17("N", 1, m, n, nrhs, ws->COPYA, lda,
                           ws->B, ldb, ws->COPYB, ldb, ws->C,
                           ws->WORK, lwork);

    /* Test 10: Check if x is in the rowspace of A */
    result[9] = 0.0f;
    if (n > crank)
        result[9] = cqrt14("N", m, n, nrhs, ws->COPYA, lda,
                           ws->B, ldb, ws->WORK, lwork);

    /* =====================================================
     *       Test CGELSS
     * ===================================================== */

    /* CGELSS: Compute the minimum-norm solution X to
     * min( norm( A * X - B ) ) using the SVD. */

    clacpy("Full", m, n, ws->COPYA, lda, ws->A, lda);
    clacpy("Full", m, nrhs, ws->COPYB, ldb, ws->B, ldb);

    cgelss(m, n, nrhs, ws->A, lda, ws->B, ldb, ws->S,
           rcond, &crank, ws->WORK, lwork, ws->RWORK, &info);

    if (info != 0) {
        fail_msg("CGELSS info=%lld m=%lld n=%lld nrhs=%lld type%lld",
                 (long long)info, (long long)m, (long long)n,
                 (long long)nrhs, (long long)itype);
        return;
    }

    /* Test 11: Compute relative error in svd */
    if (rank > 0) {
        cblas_saxpy(mnmin, -1.0f, ws->COPYS, 1, ws->S, 1);
        result[10] = cblas_sasum(mnmin, ws->S, 1) /
                     cblas_sasum(mnmin, ws->COPYS, 1) /
                     (eps * (f32)mnmin);
    } else {
        result[10] = 0.0f;
    }

    /* Test 12: Compute error in solution */
    clacpy("Full", m, nrhs, ws->COPYB, ldb, ws->WORK, ldwork);
    cqrt16("N", m, n, nrhs, ws->COPYA, lda, ws->B, ldb,
           ws->WORK, ldwork, ws->RWORK, &result[11]);

    /* Test 13: Check norm of r'*A */
    result[12] = 0.0f;
    if (m > crank)
        result[12] = cqrt17("N", 1, m, n, nrhs, ws->COPYA, lda,
                            ws->B, ldb, ws->COPYB, ldb, ws->C,
                            ws->WORK, lwork);

    /* Test 14: Check if x is in the rowspace of A */
    result[13] = 0.0f;
    if (n > crank)
        result[13] = cqrt14("N", m, n, nrhs, ws->COPYA, lda,
                            ws->B, ldb, ws->WORK, lwork);

    /* =====================================================
     *       Test CGELSD
     * ===================================================== */

    /* CGELSD: Compute the minimum-norm solution X to
     * min( norm( A * X - B ) ) using a divide and conquer SVD. */

    clacpy("Full", m, n, ws->COPYA, lda, ws->A, lda);
    clacpy("Full", m, nrhs, ws->COPYB, ldb, ws->B, ldb);

    cgelsd(m, n, nrhs, ws->A, lda, ws->B, ldb, ws->S,
           rcond, &crank, ws->WORK, lwork, ws->RWORK,
           ws->IWORK, &info);

    if (info != 0) {
        fail_msg("CGELSD info=%lld m=%lld n=%lld nrhs=%lld type%lld",
                 (long long)info, (long long)m, (long long)n,
                 (long long)nrhs, (long long)itype);
        return;
    }

    /* Test 15: Compute relative error in svd */
    if (rank > 0) {
        cblas_saxpy(mnmin, -1.0f, ws->COPYS, 1, ws->S, 1);
        result[14] = cblas_sasum(mnmin, ws->S, 1) /
                     cblas_sasum(mnmin, ws->COPYS, 1) /
                     (eps * (f32)mnmin);
    } else {
        result[14] = 0.0f;
    }

    /* Test 16: Compute error in solution */
    clacpy("Full", m, nrhs, ws->COPYB, ldb, ws->WORK, ldwork);
    cqrt16("N", m, n, nrhs, ws->COPYA, lda, ws->B, ldb,
           ws->WORK, ldwork, ws->RWORK, &result[15]);

    /* Test 17: Check norm of r'*A */
    result[16] = 0.0f;
    if (m > crank)
        result[16] = cqrt17("N", 1, m, n, nrhs, ws->COPYA, lda,
                            ws->B, ldb, ws->COPYB, ldb, ws->C,
                            ws->WORK, lwork);

    /* Test 18: Check if x is in the rowspace of A */
    result[17] = 0.0f;
    if (n > crank)
        result[17] = cqrt14("N", m, n, nrhs, ws->COPYA, lda,
                            ws->B, ldb, ws->WORK, lwork);

    /* Print information about the tests that did not pass the threshold */
    for (INT k = 6; k < 18; k++) {
        if (result[k] >= THRESH) {
            fail_msg("m=%lld n=%lld nrhs=%lld type%lld test(%lld)=%.6e",
                     (long long)m, (long long)n, (long long)nrhs,
                     (long long)itype, (long long)(k + 1), (double)result[k]);
        }
    }
}

static void test_zdrvls_case(void** state)
{
    zdrvls_params_t* p = *state;
    run_zdrvls_single(p->m, p->n, p->nrhs, p->irank, p->iscale);
}

#define MAX_TESTS 1000

static zdrvls_params_t g_params[MAX_TESTS];
static struct CMUnitTest g_tests[MAX_TESTS];
static INT g_num_tests = 0;

static void build_test_array(void)
{
    g_num_tests = 0;

    for (INT im = 0; im < (INT)NM; im++) {
        INT m = MVAL[im];
        for (INT in = 0; in < (INT)NN; in++) {
            INT n = NVAL[in];
            for (INT ins = 0; ins < (INT)NNS; ins++) {
                INT nrhs = NSVAL[ins];
                for (INT irank = 1; irank <= 2; irank++) {
                    for (INT iscale = 1; iscale <= 3; iscale++) {
                        if (g_num_tests >= MAX_TESTS) break;

                        zdrvls_params_t* p = &g_params[g_num_tests];
                        p->m = m;
                        p->n = n;
                        p->nrhs = nrhs;
                        p->irank = irank;
                        p->iscale = iscale;
                        snprintf(p->name, sizeof(p->name),
                                 "m%lld_n%lld_nrhs%lld_rank%lld_scale%lld",
                                 (long long)m, (long long)n, (long long)nrhs,
                                 (long long)irank, (long long)iscale);

                        g_tests[g_num_tests].name = p->name;
                        g_tests[g_num_tests].test_func = test_zdrvls_case;
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
    (void)_cmocka_run_group_tests("zdrvls", g_tests, g_num_tests,
                                   group_setup, group_teardown);
    return 0;
}
