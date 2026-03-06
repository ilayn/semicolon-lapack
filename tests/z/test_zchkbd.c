/**
 * @file test_zchkbd.c
 * @brief Bidiagonal SVD comprehensive test driver - port of LAPACK TESTING/EIG/zchkbd.f
 *
 * Tests the singular value decomposition (SVD) routines:
 * - ZGEBRD: reduces a complex general M-by-N matrix to bidiagonal form
 * - ZUNGBR: generates unitary matrices Q and P' from ZGEBRD
 * - ZBDSQR: SVD of bidiagonal matrix via QR iteration
 *
 * For each pair of matrix dimensions (M,N) and each selected matrix
 * type, an M by N matrix A and an M by NRHS matrix X are generated.
 * The problem dimensions are as follows
 *    A:          M x N
 *    Q:          M x min(M,N) (but M x M if NRHS > 0)
 *    P:          min(M,N) x N
 *    B:          min(M,N) x min(M,N)
 *    U, V:       min(M,N) x min(M,N)
 *    S1, S2      diagonal, order min(M,N)
 *    X:          M x NRHS
 *
 * Test ratios (14 total):
 *
 *   ZGEBRD/ZUNGBR (1-3):
 *     (1) | A - Q B PT | / ( |A| max(M,N) ulp )
 *     (2) | I - Q' Q | / ( M ulp )
 *     (3) | I - PT PT' | / ( N ulp )
 *
 *   ZBDSQR on bidiagonal (4-10):
 *     (4) | B - U S1 VT | / ( |B| min(M,N) ulp )
 *     (5) | Y - U Z | / ( |Y| max(min(M,N),k) ulp )
 *     (6) | I - U' U | / ( min(M,N) ulp )
 *     (7) | I - VT VT' | / ( min(M,N) ulp )
 *     (8) S1 non-negative decreasing
 *     (9) | S1 - S2 | / ( |S1| ulp )
 *    (10) Sturm sequence test
 *
 *   ZBDSQR on full A (11-14):
 *    (11) | A - (QU) S (VT PT) | / ( |A| max(M,N) ulp )
 *    (12) | X - (QU) Z | / ( |X| max(M,k) ulp )
 *    (13) | I - (QU)'(QU) | / ( M ulp )
 *    (14) | I - (VT PT)(PT'VT') | / ( N ulp )
 *
 * Matrix types: 16 types (MAXTYP = 16)
 */

#include "test_harness.h"
#include "verify.h"
#include "test_rng.h"
#include "semicolon_cblas.h"
#include <math.h>
#include <string.h>

#define THRESH 50.0
#define MAXTYP 16
#define NTESTS 14
#define NRHS   2

/* M,N pairs from svd.in */
static const INT MVAL[] = {0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 3, 10, 10, 16, 16, 30, 30, 40, 40};
static const INT NVAL[] = {0, 1, 3, 0, 1, 2, 0, 1, 0, 1, 3, 10, 16, 10, 16, 30, 40, 30, 40};
#define NSIZES ((int)(sizeof(MVAL) / sizeof(MVAL[0])))

/* ===================================================================== */
/* DATA arrays from zchkbd.f lines 484-487                               */
/* ===================================================================== */

/*                    1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 */
static const INT ktype[MAXTYP] = {
    1, 2, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 9, 9, 9, 10
};
static const INT kmagn[MAXTYP] = {
    1, 1, 1, 1, 1, 2, 3, 1, 1, 1, 2, 3, 1, 2, 3, 0
};
static const INT kmode[MAXTYP] = {
    0, 0, 4, 3, 1, 4, 4, 4, 3, 1, 4, 4, 0, 0, 0, 0
};

/* ===================================================================== */
/* Test parameters and workspace                                         */
/* ===================================================================== */

typedef struct {
    INT jsize;     /* index into MVAL[]/NVAL[] */
    INT jtype;     /* matrix type 1..16 */
    char name[128];
} zchkbd_params_t;

typedef struct {
    INT mmax;
    INT nmax;
    INT mnmax;
    INT lda;
    INT ldx;
    INT ldq;
    INT ldpt;
    c128* A;
    f64* BD;
    f64* BE;
    f64* S1;
    f64* S2;
    c128* X;
    c128* Y;
    c128* Z;
    c128* Q;
    c128* PT;
    c128* U;
    c128* VT;
    c128* work;
    INT lwork;
    f64* rwork;
    uint64_t rng_state[4];
} zchkbd_workspace_t;

static zchkbd_workspace_t* g_ws = NULL;

/* ===================================================================== */
/* Group setup / teardown                                                */
/* ===================================================================== */

static int group_setup(void** state)
{
    (void)state;

    g_ws = calloc(1, sizeof(zchkbd_workspace_t));
    if (!g_ws) return -1;

    g_ws->mmax = 1;
    g_ws->nmax = 1;
    g_ws->mnmax = 1;
    for (INT i = 0; i < NSIZES; i++) {
        if (MVAL[i] > g_ws->mmax) g_ws->mmax = MVAL[i];
        if (NVAL[i] > g_ws->nmax) g_ws->nmax = NVAL[i];
        INT mn = (MVAL[i] < NVAL[i]) ? MVAL[i] : NVAL[i];
        if (mn > g_ws->mnmax) g_ws->mnmax = mn;
    }

    INT mmax  = g_ws->mmax;
    INT nmax  = g_ws->nmax;
    INT mnmax = g_ws->mnmax;

    g_ws->lda  = (mmax > 1) ? mmax : 1;
    g_ws->ldx  = (mmax > 1) ? mmax : 1;
    g_ws->ldq  = (mmax > 1) ? mmax : 1;
    g_ws->ldpt = (mnmax > 1) ? mnmax : 1;

    INT lda  = g_ws->lda;
    INT ldx  = g_ws->ldx;
    INT ldq  = g_ws->ldq;
    INT ldpt = g_ws->ldpt;

    g_ws->A   = malloc((size_t)lda * nmax * sizeof(c128));
    g_ws->BD  = malloc((size_t)mnmax * sizeof(f64));
    g_ws->BE  = malloc((size_t)mnmax * sizeof(f64));
    g_ws->S1  = malloc((size_t)mnmax * sizeof(f64));
    g_ws->S2  = malloc((size_t)mnmax * sizeof(f64));
    g_ws->X   = malloc((size_t)ldx * NRHS * sizeof(c128));
    g_ws->Y   = malloc((size_t)ldx * NRHS * sizeof(c128));
    g_ws->Z   = malloc((size_t)ldx * NRHS * sizeof(c128));
    g_ws->Q   = malloc((size_t)ldq * mmax * sizeof(c128));
    g_ws->PT  = malloc((size_t)ldpt * nmax * sizeof(c128));
    g_ws->U   = malloc((size_t)ldpt * mnmax * sizeof(c128));
    g_ws->VT  = malloc((size_t)ldpt * mnmax * sizeof(c128));

    /* Workspace: from zchkbd.f lines 509-511
     * LWORK >= max(3*(M+N), M*(M + max(M,N,NRHS) + 1) + N*min(N,M)) */
    INT minwrk = 3 * (mmax + nmax);
    {
        INT mx = mmax;
        if (nmax > mx) mx = nmax;
        if (NRHS > mx) mx = NRHS;
        INT alt = mmax * (mmax + mx + 1) + nmax * mnmax;
        if (alt > minwrk) minwrk = alt;
    }

    g_ws->lwork = minwrk;
    g_ws->work  = malloc((size_t)g_ws->lwork * sizeof(c128));

    /* RWORK: 5*max(min(M,N)) for ZBDSQR (zchkbd.f line 341) */
    g_ws->rwork = malloc((size_t)(5 * mnmax) * sizeof(f64));

    if (!g_ws->A || !g_ws->BD || !g_ws->BE || !g_ws->S1 || !g_ws->S2 ||
        !g_ws->X || !g_ws->Y || !g_ws->Z || !g_ws->Q || !g_ws->PT ||
        !g_ws->U || !g_ws->VT || !g_ws->work || !g_ws->rwork) {
        return -1;
    }

    rng_seed(g_ws->rng_state, 0x2CBD50D1ULL);
    return 0;
}

static int group_teardown(void** state)
{
    (void)state;

    if (g_ws) {
        free(g_ws->A);
        free(g_ws->BD);
        free(g_ws->BE);
        free(g_ws->S1);
        free(g_ws->S2);
        free(g_ws->X);
        free(g_ws->Y);
        free(g_ws->Z);
        free(g_ws->Q);
        free(g_ws->PT);
        free(g_ws->U);
        free(g_ws->VT);
        free(g_ws->work);
        free(g_ws->rwork);
        free(g_ws);
        g_ws = NULL;
    }
    return 0;
}

/* ===================================================================== */
/* Helpers                                                               */
/* ===================================================================== */

/**
 * Check that singular values are non-negative and in non-increasing order.
 * Returns 0.0 if valid, 1/ULP if invalid.
 */
static f64 check_sv_order(const f64* S, INT n, f64 ulpinv)
{
    f64 res = 0.0;
    for (INT i = 0; i < n - 1; i++) {
        if (S[i] < S[i + 1])
            res = ulpinv;
        if (S[i] < 0.0)
            res = ulpinv;
    }
    if (n >= 1) {
        if (S[n - 1] < 0.0)
            res = ulpinv;
    }
    return res;
}

/* ===================================================================== */
/* Single test case                                                      */
/* ===================================================================== */

static void run_zchkbd_single(zchkbd_params_t* params)
{
    const INT m = MVAL[params->jsize];
    const INT n = NVAL[params->jsize];
    const INT jtype = params->jtype;
    const INT jt = jtype - 1;

    const INT mnmin = (m < n) ? m : n;
    INT mxn = m;
    if (n > mxn) mxn = n;
    if (mxn < 1) mxn = 1;
    const f64 amninv = 1.0 / (f64)mxn;

    const INT lda   = g_ws->lda;
    const INT ldx   = g_ws->ldx;
    const INT ldq   = g_ws->ldq;
    const INT ldpt  = g_ws->ldpt;
    const INT lwork = g_ws->lwork;

    c128* A     = g_ws->A;
    f64*  BD    = g_ws->BD;
    f64*  BE    = g_ws->BE;
    f64*  S1    = g_ws->S1;
    f64*  S2    = g_ws->S2;
    c128* X     = g_ws->X;
    c128* Y     = g_ws->Y;
    c128* Z     = g_ws->Z;
    c128* Q     = g_ws->Q;
    c128* PT    = g_ws->PT;
    c128* U     = g_ws->U;
    c128* VT    = g_ws->VT;
    c128* work  = g_ws->work;
    f64*  rwork = g_ws->rwork;
    uint64_t* rng = g_ws->rng_state;

    const f64 unfl    = dlamch("S");
    const f64 ovfl    = dlamch("O");
    const f64 ulp     = dlamch("P");
    const f64 ulpinv  = 1.0 / ulp;
    const f64 rtunfl  = sqrt(unfl);
    const f64 rtovfl  = sqrt(ovfl);
    INT log2ui        = (INT)(log(ulpinv) / log(2.0));

    const c128 cone  = CMPLX(1.0, 0.0);
    const c128 czero = CMPLX(0.0, 0.0);

    f64 result[NTESTS];
    for (INT i = 0; i < NTESTS; i++)
        result[i] = -1.0;

    INT iinfo = 0;
    INT bidiag = 0;
    char ctx[256];
    const char* uplo_str;

    /* ----------------------------------------------------------- */
    /* Compute "A"                                                 */
    /* zchkbd.f lines 605-715                                      */
    /* ----------------------------------------------------------- */

    INT itype = ktype[jt];
    INT imode = kmode[jt];

    f64 anorm;
    switch (kmagn[jt]) {
        case 2:  anorm = (rtovfl * ulp) * amninv; break;
        case 3:  anorm = rtunfl * (f64)(m > n ? m : n) * ulpinv; break;
        default: anorm = 1.0; break;
    }
    f64 cond = ulpinv;

    zlaset("Full", lda, n, czero, czero, A, lda);
    iinfo = 0;
    bidiag = 0;

    if (itype == 1) {
        iinfo = 0;

    } else if (itype == 2) {
        for (INT jcol = 0; jcol < mnmin; jcol++)
            A[jcol + jcol * lda] = CMPLX(anorm, 0.0);

    } else if (itype == 4) {
        zlatms(mnmin, mnmin, "S", "N", rwork, imode, cond, anorm,
               0, 0, "N", A, lda, work, &iinfo, rng);

    } else if (itype == 5) {
        zlatms(mnmin, mnmin, "S", "S", rwork, imode, cond, anorm,
               m, n, "N", A, lda, work, &iinfo, rng);

    } else if (itype == 6) {
        zlatms(m, n, "S", "N", rwork, imode, cond, anorm,
               m, n, "N", A, lda, work, &iinfo, rng);

    } else if (itype == 7) {
        INT idumma[1] = {0};
        zlatmr(mnmin, mnmin, "S", "N", work, 6, 1.0, cone,
               "T", "N", work + mnmin, 1, 1.0,
               work + 2 * mnmin, 1, 1.0, "N", idumma, 0, 0,
               0.0, anorm, "NO", A, lda, idumma, &iinfo, rng);

    } else if (itype == 8) {
        INT idumma[1] = {0};
        zlatmr(mnmin, mnmin, "S", "S", work, 6, 1.0, cone,
               "T", "N", work + mnmin, 1, 1.0,
               work + m + mnmin, 1, 1.0, "N", idumma, m, n,
               0.0, anorm, "NO", A, lda, idumma, &iinfo, rng);

    } else if (itype == 9) {
        INT idumma[1] = {0};
        zlatmr(m, n, "S", "N", work, 6, 1.0, cone,
               "T", "N", work + mnmin, 1, 1.0,
               work + m + mnmin, 1, 1.0, "N", idumma, m, n,
               0.0, anorm, "NO", A, lda, idumma, &iinfo, rng);

    } else if (itype == 10) {
        f64 temp1 = -2.0 * log(ulp);
        for (INT j = 0; j < mnmin; j++) {
            BD[j] = exp(temp1 * rng_uniform_symmetric(rng));
            if (j < mnmin - 1)
                BE[j] = exp(temp1 * rng_uniform_symmetric(rng));
        }

        iinfo = 0;
        bidiag = 1;
        if (m >= n)
            uplo_str = "U";
        else
            uplo_str = "L";
    } else {
        iinfo = 1;
    }

    if (iinfo == 0) {
        if (bidiag) {
            INT idumma[1] = {0};
            zlatmr(mnmin, NRHS, "S", "N", work, 6,
                   1.0, cone, "T", "N", work + mnmin, 1, 1.0,
                   work + 2 * mnmin, 1, 1.0, "N",
                   idumma, mnmin, NRHS, 0.0, 1.0, "NO", Y,
                   ldx, idumma, &iinfo, rng);
        } else {
            INT idumma[1] = {0};
            zlatmr(m, NRHS, "S", "N", work, 6, 1.0,
                   cone, "T", "N", work + m, 1, 1.0,
                   work + 2 * m, 1, 1.0, "N",
                   idumma, m, NRHS, 0.0, 1.0, "NO", X,
                   ldx, idumma, &iinfo, rng);
        }
    }

    if (iinfo != 0) {
        snprintf(ctx, sizeof(ctx),
                 "zchkbd m=%d n=%d type=%d: Generator info=%d",
                 m, n, jtype, iinfo);
        set_test_context(ctx);
        assert_info_success(iinfo);
        clear_test_context();
        return;
    }

    /* ----------------------------------------------------------- */
    /* Call ZGEBRD and ZUNGBR to compute B, Q, and P, do tests.    */
    /* zchkbd.f lines 749-820                                      */
    /* ----------------------------------------------------------- */

    INT mq = 0;

    if (!bidiag) {

        zlacpy(" ", m, n, A, lda, Q, ldq);
        zgebrd(m, n, Q, ldq, BD, BE, work, work + mnmin,
               work + 2 * mnmin, lwork - 2 * mnmin, &iinfo);

        if (iinfo != 0) {
            snprintf(ctx, sizeof(ctx),
                     "ZGEBRD info=%d m=%d n=%d type=%d", iinfo, m, n, jtype);
            set_test_context(ctx);
            assert_info_success(iinfo);
            clear_test_context();
            return;
        }

        zlacpy(" ", m, n, Q, ldq, PT, ldpt);
        if (m >= n)
            uplo_str = "U";
        else
            uplo_str = "L";

        mq = m;
        if (NRHS <= 0)
            mq = mnmin;
        zungbr("Q", m, mq, n, Q, ldq, work,
               work + 2 * mnmin, lwork - 2 * mnmin, &iinfo);

        if (iinfo != 0) {
            snprintf(ctx, sizeof(ctx),
                     "ZUNGBR(Q) info=%d m=%d n=%d type=%d", iinfo, m, n, jtype);
            set_test_context(ctx);
            assert_info_success(iinfo);
            clear_test_context();
            return;
        }

        zungbr("P", mnmin, n, m, PT, ldpt, work + mnmin,
               work + 2 * mnmin, lwork - 2 * mnmin, &iinfo);

        if (iinfo != 0) {
            snprintf(ctx, sizeof(ctx),
                     "ZUNGBR(P) info=%d m=%d n=%d type=%d", iinfo, m, n, jtype);
            set_test_context(ctx);
            assert_info_success(iinfo);
            clear_test_context();
            return;
        }

        /* Y := Q' * X  (conjugate transpose for complex) */
        cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans,
                    m, NRHS, m, &cone, Q, ldq, X, ldx, &czero, Y, ldx);

        /* Test 1:  Check the decomposition A := Q * B * PT */
        zbdt01(m, n, 1, A, lda, Q, ldq, BD, BE, PT, ldpt,
               work, rwork, &result[0]);
        /* Test 2:  Check the orthogonality of Q */
        zunt01("Columns", m, mq, Q, ldq, work, lwork, rwork, &result[1]);
        /* Test 3:  Check the orthogonality of PT */
        zunt01("Rows", mnmin, n, PT, ldpt, work, lwork, rwork, &result[2]);
    }

    /* ----------------------------------------------------------- */
    /* Use ZBDSQR to form the SVD of the bidiagonal matrix B:      */
    /* B := U * S1 * VT, and compute Z = U' * Y.                  */
    /* zchkbd.f lines 826-930                                      */
    /* ----------------------------------------------------------- */

    cblas_dcopy(mnmin, BD, 1, S1, 1);
    if (mnmin > 0)
        cblas_dcopy(mnmin - 1, BE, 1, rwork, 1);
    zlacpy(" ", m, NRHS, Y, ldx, Z, ldx);
    zlaset("Full", mnmin, mnmin, czero, cone, U, ldpt);
    zlaset("Full", mnmin, mnmin, czero, cone, VT, ldpt);

    zbdsqr(uplo_str, mnmin, mnmin, mnmin, NRHS, S1, rwork, VT,
           ldpt, U, ldpt, Z, ldx, rwork + mnmin, &iinfo);

    if (iinfo != 0) {
        snprintf(ctx, sizeof(ctx),
                 "ZBDSQR(vects) info=%d m=%d n=%d type=%d", iinfo, m, n, jtype);
        set_test_context(ctx);
        if (iinfo < 0) {
            assert_info_success(iinfo);
            clear_test_context();
            return;
        } else {
            result[3] = ulpinv;
            goto results;
        }
    }

    /* Use ZBDSQR to compute only the singular values */
    cblas_dcopy(mnmin, BD, 1, S2, 1);
    if (mnmin > 0)
        cblas_dcopy(mnmin - 1, BE, 1, rwork, 1);

    zbdsqr(uplo_str, mnmin, 0, 0, 0, S2, rwork, VT, ldpt, U,
           ldpt, Z, ldx, rwork + mnmin, &iinfo);

    if (iinfo != 0) {
        snprintf(ctx, sizeof(ctx),
                 "ZBDSQR(values) info=%d m=%d n=%d type=%d", iinfo, m, n, jtype);
        set_test_context(ctx);
        if (iinfo < 0) {
            assert_info_success(iinfo);
            clear_test_context();
            return;
        } else {
            result[8] = ulpinv;
            goto results;
        }
    }

    /* Test 4:  Check the decomposition B := U * S1 * VT */
    zbdt03(uplo_str, mnmin, 1, BD, BE, U, ldpt, S1, VT, ldpt,
           work, &result[3]);
    /* Test 5:  Check the computation Z := U' * Y */
    zbdt02(mnmin, NRHS, Y, ldx, Z, ldx, U, ldpt, work, rwork, &result[4]);
    /* Test 6:  Check the orthogonality of U */
    zunt01("Columns", mnmin, mnmin, U, ldpt, work, lwork, rwork, &result[5]);
    /* Test 7:  Check the orthogonality of VT */
    zunt01("Rows", mnmin, mnmin, VT, ldpt, work, lwork, rwork, &result[6]);

    /* Test 8:  Check that the singular values are sorted in
     *          non-increasing order and are non-negative */
    result[7] = check_sv_order(S1, mnmin, ulpinv);

    /* Test 9:  Compare ZBDSQR with and without singular vectors */
    {
        f64 temp2 = 0.0;
        for (INT j = 0; j < mnmin; j++) {
            f64 d1 = sqrt(unfl) * ((S1[0] > 1.0) ? S1[0] : 1.0);
            f64 d2 = ulp * ((fabs(S1[j]) > fabs(S2[j])) ? fabs(S1[j]) : fabs(S2[j]));
            f64 denom = (d1 > d2) ? d1 : d2;
            f64 temp1 = fabs(S1[j] - S2[j]) / denom;
            if (temp1 > temp2) temp2 = temp1;
        }
        result[8] = temp2;
    }

    /* Test 10: Sturm sequence test of singular values */
    {
        f64 temp1 = THRESH * (0.5 - ulp);
        for (INT j = 0; j <= log2ui; j++) {
            dsvdch(mnmin, BD, BE, S1, temp1, &iinfo);
            if (iinfo == 0)
                break;
            temp1 = temp1 * 2.0;
        }
        result[9] = temp1;
    }

    /* ----------------------------------------------------------- */
    /* Use ZBDSQR to form the decomposition A := (QU) S (VT PT)   */
    /* zchkbd.f lines 932-957                                      */
    /* ----------------------------------------------------------- */

    if (!bidiag) {
        cblas_dcopy(mnmin, BD, 1, S2, 1);
        if (mnmin > 0)
            cblas_dcopy(mnmin - 1, BE, 1, rwork, 1);

        zbdsqr(uplo_str, mnmin, n, m, NRHS, S2, rwork, PT, ldpt,
               Q, ldq, Y, ldx, rwork + mnmin, &iinfo);

        /* Test 11:  Check the decomposition A := Q*U * S2 * VT*PT */
        f64 dumma[1] = {0.0};
        zbdt01(m, n, 0, A, lda, Q, ldq, S2, dumma, PT,
               ldpt, work, rwork, &result[10]);
        /* Test 12:  Check the computation Z := U' * Q' * X */
        zbdt02(m, NRHS, X, ldx, Y, ldx, Q, ldq, work, rwork, &result[11]);
        /* Test 13:  Check the orthogonality of Q*U */
        zunt01("Columns", m, mq, Q, ldq, work, lwork, rwork, &result[12]);
        /* Test 14:  Check the orthogonality of VT*PT */
        zunt01("Rows", mnmin, n, PT, ldpt, work, lwork, rwork, &result[13]);
    }

results:
    /* End of Loop -- Check for RESULT(j) >= THRESH */
    for (INT j = 0; j < NTESTS; j++) {
        if (result[j] >= THRESH) {
            snprintf(ctx, sizeof(ctx),
                     "zchkbd M=%d N=%d type=%d test(%d)=%.4g",
                     m, n, jtype, j + 1, result[j]);
            set_test_context(ctx);
            assert_residual_below(result[j], THRESH);
            clear_test_context();
        }
    }
}

/* ===================================================================== */
/* CMocka test wrapper                                                   */
/* ===================================================================== */

static void test_zchkbd(void** state)
{
    (void)state;
    zchkbd_params_t* params = (zchkbd_params_t*)(*state);
    run_zchkbd_single(params);
}

/* ===================================================================== */
/* Test table generation                                                 */
/* ===================================================================== */

#define MAX_TESTS (NSIZES * MAXTYP)

static zchkbd_params_t g_params[MAX_TESTS];
static struct CMUnitTest g_tests[MAX_TESTS];
static INT g_ntests = 0;

static void generate_test_table(void)
{
    g_ntests = 0;
    for (INT js = 0; js < NSIZES; js++) {
        for (INT jt = 1; jt <= MAXTYP; jt++) {
            zchkbd_params_t* p = &g_params[g_ntests];
            p->jsize = js;
            p->jtype = jt;
            snprintf(p->name, sizeof(p->name),
                     "zchkbd M=%d N=%d type=%d",
                     MVAL[js], NVAL[js], jt);

            g_tests[g_ntests].name = p->name;
            g_tests[g_ntests].test_func = test_zchkbd;
            g_tests[g_ntests].initial_state = p;
            g_tests[g_ntests].setup_func = NULL;
            g_tests[g_ntests].teardown_func = NULL;
            g_ntests++;
        }
    }
}

int main(void)
{
    generate_test_table();
    (void)cmocka_run_group_tests_name("zchkbd", g_tests, group_setup,
                                       group_teardown);
    return 0;
}
