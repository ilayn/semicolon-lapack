/**
 * @file test_dchkbd.c
 * @brief Bidiagonal SVD comprehensive test driver - port of LAPACK TESTING/EIG/dchkbd.f
 *
 * Tests the singular value decomposition (SVD) routines:
 * - DGEBRD: reduces a general M-by-N matrix to bidiagonal form
 * - DORGBR: generates orthogonal matrices Q and P' from DGEBRD
 * - DBDSQR: SVD of bidiagonal matrix via QR iteration
 * - DBDSDC: SVD of bidiagonal matrix via divide-and-conquer
 * - DBDSVDX: SVD of bidiagonal matrix via bisection and inverse iteration
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
 * Test ratios (34 total):
 *
 *   DGEBRD/DORGBR (1-3):
 *     (1) | A - Q B PT | / ( |A| max(M,N) ulp )
 *     (2) | I - Q' Q | / ( M ulp )
 *     (3) | I - PT PT' | / ( N ulp )
 *
 *   DBDSQR on bidiagonal (4-10):
 *     (4) | B - U S1 VT | / ( |B| min(M,N) ulp )
 *     (5) | Y - U Z | / ( |Y| max(min(M,N),k) ulp )
 *     (6) | I - U' U | / ( min(M,N) ulp )
 *     (7) | I - VT VT' | / ( min(M,N) ulp )
 *     (8) S1 non-negative decreasing
 *     (9) | S1 - S2 | / ( |S1| ulp )
 *    (10) Sturm sequence test
 *
 *   DBDSQR on full A (11-14):
 *    (11) | A - (QU) S (VT PT) | / ( |A| max(M,N) ulp )
 *    (12) | X - (QU) Z | / ( |X| max(M,k) ulp )
 *    (13) | I - (QU)'(QU) | / ( M ulp )
 *    (14) | I - (VT PT)(PT'VT') | / ( N ulp )
 *
 *   DBDSDC (15-19):
 *    (15) | B - U S1 VT | / ( |B| min(M,N) ulp )
 *    (16) | I - U' U | / ( min(M,N) ulp )
 *    (17) | I - VT VT' | / ( min(M,N) ulp )
 *    (18) S1 non-negative decreasing
 *    (19) | S1 - S2 | / ( |S1| ulp )
 *
 *   DBDSVDX RANGE='A' (20-24):
 *    (20) | B - U S1 VT | / ( |B| min(M,N) ulp )
 *    (21) | I - U' U | / ( min(M,N) ulp )
 *    (22) | I - VT VT' | / ( min(M,N) ulp )
 *    (23) S1 non-negative decreasing
 *    (24) | S1 - S2 | / ( |S1| ulp )
 *
 *   DBDSVDX RANGE='I' (25-29):
 *    (25) | S1 - U' B VT' | / ( |S| n ulp )
 *    (26) | I - U' U | / ( min(M,N) ulp )
 *    (27) | I - VT VT' | / ( min(M,N) ulp )
 *    (28) S1 non-negative decreasing
 *    (29) | S1 - S2 | / ( |S1| ulp )
 *
 *   DBDSVDX RANGE='V' (30-34):
 *    (30) | S1 - U' B VT' | / ( |S1| n ulp )
 *    (31) | I - U' U | / ( min(M,N) ulp )
 *    (32) | I - VT VT' | / ( min(M,N) ulp )
 *    (33) S1 non-negative decreasing
 *    (34) | S1 - S2 | / ( |S1| ulp )
 *
 * Matrix types: 16 types (MAXTYP = 16)
 */

#include "test_harness.h"
#include "verify.h"
#include "test_rng.h"
#include <cblas.h>
#include <math.h>
#include <string.h>

#define THRESH 50.0
#define MAXTYP 16
#define NTESTS 34
#define NRHS   2

/* M,N pairs from svd.in */
static const int MVAL[] = {0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 3, 10, 10, 16, 16, 30, 30, 40, 40};
static const int NVAL[] = {0, 1, 3, 0, 1, 2, 0, 1, 0, 1, 3, 10, 16, 10, 16, 30, 40, 30, 40};
#define NSIZES ((int)(sizeof(MVAL) / sizeof(MVAL[0])))

/* External function declarations */
extern f64  dlamch(const char* cmach);
extern void dgebrd(const int m, const int n, f64* restrict A, const int lda,
                   f64* restrict D, f64* restrict E,
                   f64* restrict tauq, f64* restrict taup,
                   f64* restrict work, const int lwork, int* info);
extern void dorgbr(const char* vect, const int m, const int n, const int k,
                   f64* restrict A, const int lda, const f64* restrict tau,
                   f64* restrict work, const int lwork, int* info);
extern void dbdsqr(const char* uplo, const int n, const int ncvt,
                   const int nru, const int ncc,
                   f64* restrict D, f64* restrict E,
                   f64* restrict VT, const int ldvt,
                   f64* restrict U, const int ldu,
                   f64* restrict C, const int ldc,
                   f64* restrict work, int* info);
extern void dbdsdc(const char* uplo, const char* compq, const int n,
                   f64* restrict D, f64* restrict E,
                   f64* restrict U, const int ldu,
                   f64* restrict VT, const int ldvt,
                   f64* restrict Q, int* restrict IQ,
                   f64* restrict work, int* restrict iwork, int* info);
extern void dbdsvdx(const char* uplo, const char* jobz, const char* range,
                    const int n, f64* restrict D, f64* restrict E,
                    const f64 vl, const f64 vu, const int il, const int iu,
                    int* ns, f64* restrict S, f64* restrict Z, const int ldz,
                    f64* restrict work, int* restrict iwork, int* info);
extern void dlacpy(const char* uplo, const int m, const int n,
                   const f64* A, const int lda, f64* B, const int ldb);
extern void dlaset(const char* uplo, const int m, const int n,
                   const f64 alpha, const f64 beta, f64* A, const int lda);
extern void dlascl(const char* type, const int kl, const int ku,
                   const f64 cfrom, const f64 cto, const int m, const int n,
                   f64* A, const int lda, int* info);

/* ===================================================================== */
/* DATA arrays from dchkbd.f lines 564-567                               */
/* ===================================================================== */

/*                    1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 */
static const int ktype[MAXTYP] = {
    1, 2, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 9, 9, 9, 10
};
static const int kmagn[MAXTYP] = {
    1, 1, 1, 1, 1, 2, 3, 1, 1, 1, 2, 3, 1, 2, 3, 0
};
static const int kmode[MAXTYP] = {
    0, 0, 4, 3, 1, 4, 4, 4, 3, 1, 4, 4, 0, 0, 0, 0
};

/* ===================================================================== */
/* Test parameters and workspace                                         */
/* ===================================================================== */

typedef struct {
    int jsize;     /* index into MVAL[]/NVAL[] */
    int jtype;     /* matrix type 1..16 */
    char name[128];
} dchkbd_params_t;

typedef struct {
    int mmax;
    int nmax;
    int mnmax;
    int lda;
    int ldx;
    int ldq;
    int ldpt;
    f64* A;
    f64* BD;
    f64* BE;
    f64* S1;
    f64* S2;
    f64* X;
    f64* Y;
    f64* Z;
    f64* Q;
    f64* PT;
    f64* U;
    f64* VT;
    f64* work;
    int lwork;
    int* iwork;
    uint64_t rng_state[4];
} dchkbd_workspace_t;

static dchkbd_workspace_t* g_ws = NULL;

/* ===================================================================== */
/* Group setup / teardown                                                */
/* ===================================================================== */

static int group_setup(void** state)
{
    (void)state;

    g_ws = calloc(1, sizeof(dchkbd_workspace_t));
    if (!g_ws) return -1;

    g_ws->mmax = 1;
    g_ws->nmax = 1;
    g_ws->mnmax = 1;
    for (int i = 0; i < NSIZES; i++) {
        if (MVAL[i] > g_ws->mmax) g_ws->mmax = MVAL[i];
        if (NVAL[i] > g_ws->nmax) g_ws->nmax = NVAL[i];
        int mn = (MVAL[i] < NVAL[i]) ? MVAL[i] : NVAL[i];
        if (mn > g_ws->mnmax) g_ws->mnmax = mn;
    }

    int mmax  = g_ws->mmax;
    int nmax  = g_ws->nmax;
    int mnmax = g_ws->mnmax;

    g_ws->lda  = (mmax > 1) ? mmax : 1;
    g_ws->ldx  = (mmax > 1) ? mmax : 1;
    g_ws->ldq  = (mmax > 1) ? mmax : 1;
    g_ws->ldpt = (mnmax > 1) ? mnmax : 1;

    int lda  = g_ws->lda;
    int ldx  = g_ws->ldx;
    int ldq  = g_ws->ldq;
    int ldpt = g_ws->ldpt;

    g_ws->A   = malloc((size_t)lda * nmax * sizeof(f64));
    g_ws->BD  = malloc((size_t)mnmax * sizeof(f64));
    g_ws->BE  = malloc((size_t)mnmax * sizeof(f64));
    /* S1/S2 need 2*mnmax: dbdsvdx's internal TGK matrix is 2n-dimensional,
     * and dstevx can write up to n+1 eigenvalues before the MIN(NSL,NRU)
     * clamp in dbdsvdx. Reference LAPACK masks this via D(NMAX,12). */
    g_ws->S1  = malloc((size_t)(2 * mnmax) * sizeof(f64));
    g_ws->S2  = malloc((size_t)(2 * mnmax) * sizeof(f64));
    g_ws->X   = malloc((size_t)ldx * NRHS * sizeof(f64));
    g_ws->Y   = malloc((size_t)ldx * NRHS * sizeof(f64));
    g_ws->Z   = malloc((size_t)ldx * NRHS * sizeof(f64));
    g_ws->Q   = malloc((size_t)ldq * mmax * sizeof(f64));
    g_ws->PT  = malloc((size_t)ldpt * nmax * sizeof(f64));
    g_ws->U   = malloc((size_t)ldpt * mnmax * sizeof(f64));
    g_ws->VT  = malloc((size_t)ldpt * mnmax * sizeof(f64));

    /* Workspace: from dchkbd.f lines 589-591, plus extra for DBDSVDX */
    int minwrk = 3 * (mmax + nmax);
    {
        /* Pick the formula from dchkbd.f exactly:
         * M*(M + MAX(M,N,NRHS) + 1) + N*MIN(N,M) */
        int mx = mmax;
        if (nmax > mx) mx = nmax;
        if (NRHS > mx) mx = NRHS;
        int alt = mmax * (mmax + mx + 1) + nmax * mnmax;
        if (alt > minwrk) minwrk = alt;
    }
    /* Extra for DBDSVDX workspace: iwbz + 2*mnmax*(mnmax+1) + 14*mnmax */
    int extra = 3 * mnmax + 2 * mnmax * (mnmax + 1) + 14 * mnmax;
    if (extra > 0) minwrk += extra;

    g_ws->lwork = minwrk;
    g_ws->work  = malloc((size_t)g_ws->lwork * sizeof(f64));
    g_ws->iwork = malloc((size_t)(12 * mnmax) * sizeof(int));

    if (!g_ws->A || !g_ws->BD || !g_ws->BE || !g_ws->S1 || !g_ws->S2 ||
        !g_ws->X || !g_ws->Y || !g_ws->Z || !g_ws->Q || !g_ws->PT ||
        !g_ws->U || !g_ws->VT || !g_ws->work || !g_ws->iwork) {
        return -1;
    }

    rng_seed(g_ws->rng_state, 0xDCBD50D1ULL);
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
        free(g_ws->iwork);
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
static f64 check_sv_order(const f64* S, int n, f64 ulpinv)
{
    f64 res = 0.0;
    for (int i = 0; i < n - 1; i++) {
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

/**
 * Compare two sets of singular values.
 * Returns max_j |S1(j) - S2(j)| / max(sqrt(UNFL)*max(S1(1),1), ULP*max(|S1(1)|,|S2(1)|))
 */
static f64 compare_sv(const f64* S1, const f64* S2, int n, f64 ulp, f64 unfl)
{
    f64 temp2 = 0.0;
    for (int j = 0; j < n; j++) {
        f64 denom1 = sqrt(unfl);
        f64 s1max = (S1[0] > 1.0) ? S1[0] : 1.0;
        denom1 *= s1max;
        f64 denom2 = ulp;
        f64 abs1 = fabs(S1[0]);
        f64 abs2 = fabs(S2[0]);
        denom2 *= (abs1 > abs2) ? abs1 : abs2;
        f64 denom = (denom1 > denom2) ? denom1 : denom2;
        f64 temp1 = fabs(S1[j] - S2[j]) / denom;
        if (temp1 > temp2) temp2 = temp1;
    }
    return temp2;
}

/* ===================================================================== */
/* Single test case                                                      */
/* ===================================================================== */

static void run_dchkbd_single(dchkbd_params_t* params)
{
    const int m = MVAL[params->jsize];
    const int n = NVAL[params->jsize];
    const int jtype = params->jtype;
    const int jt = jtype - 1;

    const int mnmin = (m < n) ? m : n;
    int mxn = m;
    if (n > mxn) mxn = n;
    if (mxn < 1) mxn = 1;
    const f64 amninv = 1.0 / (f64)mxn;

    const int lda   = g_ws->lda;
    const int ldx   = g_ws->ldx;
    const int ldq   = g_ws->ldq;
    const int ldpt  = g_ws->ldpt;
    const int lwork = g_ws->lwork;

    f64* A     = g_ws->A;
    f64* BD    = g_ws->BD;
    f64* BE    = g_ws->BE;
    f64* S1    = g_ws->S1;
    f64* S2    = g_ws->S2;
    f64* X     = g_ws->X;
    f64* Y     = g_ws->Y;
    f64* Z     = g_ws->Z;
    f64* Q     = g_ws->Q;
    f64* PT    = g_ws->PT;
    f64* U     = g_ws->U;
    f64* VT    = g_ws->VT;
    f64* work  = g_ws->work;
    int* iwork = g_ws->iwork;
    uint64_t* rng = g_ws->rng_state;

    const f64 unfl    = dlamch("S");
    const f64 ovfl    = dlamch("O");
    const f64 ulp     = dlamch("P");
    const f64 ulpinv  = 1.0 / ulp;
    const f64 rtunfl  = sqrt(unfl);
    const f64 rtovfl  = sqrt(ovfl);
    int log2ui        = (int)(log(ulpinv) / log(2.0));

    f64 result[NTESTS];
    for (int i = 0; i < NTESTS; i++)
        result[i] = -1.0;

    int iinfo = 0;
    int bidiag = 0;
    int ns1 = 0, ns2 = 0;
    f64 anorm_sv = 0.0;
    char ctx[256];
    const char* uplo_str;

    /* ----------------------------------------------------------- */
    /* Compute "A"                                                 */
    /* dchkbd.f lines 686-797                                      */
    /* ----------------------------------------------------------- */

    int itype = ktype[jt];
    int imode = kmode[jt];

    f64 anorm;
    switch (kmagn[jt]) {
        case 2:  anorm = (rtovfl * ulp) * amninv; break;
        case 3:  anorm = rtunfl * (f64)(m > n ? m : n) * ulpinv; break;
        default: anorm = 1.0; break;
    }
    f64 cond = ulpinv;

    dlaset("Full", lda, n, 0.0, 0.0, A, lda);
    iinfo = 0;
    bidiag = 0;

    if (itype == 1) {
        /* Zero matrix */
        iinfo = 0;

    } else if (itype == 2) {
        /* Identity */
        for (int jcol = 0; jcol < mnmin; jcol++)
            A[jcol + jcol * lda] = anorm;

    } else if (itype == 4) {
        /* Diagonal Matrix, [Eigen]values Specified */
        dlatms(mnmin, mnmin, "S", "N", work, imode, cond, anorm,
               0, 0, "N", A, lda, work + mnmin, &iinfo, rng);

    } else if (itype == 5) {
        /* Symmetric, eigenvalues specified */
        dlatms(mnmin, mnmin, "S", "S", work, imode, cond, anorm,
               m, n, "N", A, lda, work + mnmin, &iinfo, rng);

    } else if (itype == 6) {
        /* Nonsymmetric, singular values specified */
        dlatms(m, n, "S", "N", work, imode, cond, anorm,
               m, n, "N", A, lda, work + mnmin, &iinfo, rng);

    } else if (itype == 7) {
        /* Diagonal, random entries */
        int idumma[1] = {0};
        dlatmr(mnmin, mnmin, "S", "N", work, 6, 1.0, 1.0,
               "T", "N", work + mnmin, 1, 1.0,
               work + 2 * mnmin, 1, 1.0, "N", idumma, 0, 0,
               0.0, anorm, "NO", A, lda, iwork, &iinfo, rng);

    } else if (itype == 8) {
        /* Symmetric, random entries */
        int idumma[1] = {0};
        dlatmr(mnmin, mnmin, "S", "S", work, 6, 1.0, 1.0,
               "T", "N", work + mnmin, 1, 1.0,
               work + m + mnmin, 1, 1.0, "N", idumma, m, n,
               0.0, anorm, "NO", A, lda, iwork, &iinfo, rng);

    } else if (itype == 9) {
        /* Nonsymmetric, random entries */
        int idumma[1] = {0};
        dlatmr(m, n, "S", "N", work, 6, 1.0, 1.0,
               "T", "N", work + mnmin, 1, 1.0,
               work + m + mnmin, 1, 1.0, "N", idumma, m, n,
               0.0, anorm, "NO", A, lda, iwork, &iinfo, rng);

    } else if (itype == 10) {
        /* Bidiagonal, random entries */
        f64 temp1 = -2.0 * log(ulp);
        for (int j = 0; j < mnmin; j++) {
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
        /* Generate Right-Hand Side */
        if (bidiag) {
            int idumma[1] = {0};
            dlatmr(mnmin, NRHS, "S", "N", work, 6,
                   1.0, 1.0, "T", "N", work + mnmin, 1, 1.0,
                   work + 2 * mnmin, 1, 1.0, "N",
                   idumma, mnmin, NRHS, 0.0, 1.0, "NO", Y,
                   ldx, iwork, &iinfo, rng);
        } else {
            int idumma[1] = {0};
            dlatmr(m, NRHS, "S", "N", work, 6, 1.0,
                   1.0, "T", "N", work + m, 1, 1.0,
                   work + 2 * m, 1, 1.0, "N",
                   idumma, m, NRHS, 0.0, 1.0, "NO", X,
                   ldx, iwork, &iinfo, rng);
        }
    }

    if (iinfo != 0) {
        snprintf(ctx, sizeof(ctx),
                 "dchkbd m=%d n=%d type=%d: Generator info=%d",
                 m, n, jtype, iinfo);
        set_test_context(ctx);
        assert_info_success(iinfo);
        clear_test_context();
        return;
    }

    /* ----------------------------------------------------------- */
    /* Call DGEBRD and DORGBR to compute B, Q, and P, do tests.  */
    /* dchkbd.f lines 831-902                                     */
    /* ----------------------------------------------------------- */

    int mq = 0;

    if (!bidiag) {

        /* B := Q' * A * P */
        dlacpy(" ", m, n, A, lda, Q, ldq);
        dgebrd(m, n, Q, ldq, BD, BE, work, work + mnmin,
               work + 2 * mnmin, lwork - 2 * mnmin, &iinfo);

        if (iinfo != 0) {
            snprintf(ctx, sizeof(ctx),
                     "DGEBRD info=%d m=%d n=%d type=%d", iinfo, m, n, jtype);
            set_test_context(ctx);
            assert_info_success(iinfo);
            clear_test_context();
            return;
        }

        dlacpy(" ", m, n, Q, ldq, PT, ldpt);
        if (m >= n)
            uplo_str = "U";
        else
            uplo_str = "L";

        /* Generate Q */
        mq = m;
        if (NRHS <= 0)
            mq = mnmin;
        dorgbr("Q", m, mq, n, Q, ldq, work,
               work + 2 * mnmin, lwork - 2 * mnmin, &iinfo);

        if (iinfo != 0) {
            snprintf(ctx, sizeof(ctx),
                     "DORGBR(Q) info=%d m=%d n=%d type=%d", iinfo, m, n, jtype);
            set_test_context(ctx);
            assert_info_success(iinfo);
            clear_test_context();
            return;
        }

        /* Generate P' */
        dorgbr("P", mnmin, n, m, PT, ldpt, work + mnmin,
               work + 2 * mnmin, lwork - 2 * mnmin, &iinfo);

        if (iinfo != 0) {
            snprintf(ctx, sizeof(ctx),
                     "DORGBR(P) info=%d m=%d n=%d type=%d", iinfo, m, n, jtype);
            set_test_context(ctx);
            assert_info_success(iinfo);
            clear_test_context();
            return;
        }

        /* Y := Q' * X */
        cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                    m, NRHS, m, 1.0, Q, ldq, X, ldx, 0.0, Y, ldx);

        /* Test 1:  Check the decomposition A := Q * B * PT */
        dbdt01(m, n, 1, A, lda, Q, ldq, BD, BE, PT, ldpt,
               work, &result[0]);
        /* Test 2:  Check the orthogonality of Q */
        dort01("Columns", m, mq, Q, ldq, work, lwork, &result[1]);
        /* Test 3:  Check the orthogonality of PT */
        dort01("Rows", mnmin, n, PT, ldpt, work, lwork, &result[2]);
    }

    /* ----------------------------------------------------------- */
    /* Use DBDSQR to form the SVD of the bidiagonal matrix B:     */
    /* B := U * S1 * VT, and compute Z = U' * Y.                 */
    /* dchkbd.f lines 904-1010                                     */
    /* ----------------------------------------------------------- */

    cblas_dcopy(mnmin, BD, 1, S1, 1);
    if (mnmin > 0)
        cblas_dcopy(mnmin - 1, BE, 1, work, 1);
    dlacpy(" ", m, NRHS, Y, ldx, Z, ldx);
    dlaset("Full", mnmin, mnmin, 0.0, 1.0, U, ldpt);
    dlaset("Full", mnmin, mnmin, 0.0, 1.0, VT, ldpt);

    dbdsqr(uplo_str, mnmin, mnmin, mnmin, NRHS, S1, work, VT,
           ldpt, U, ldpt, Z, ldx, work + mnmin, &iinfo);

    if (iinfo != 0) {
        snprintf(ctx, sizeof(ctx),
                 "DBDSQR(vects) info=%d m=%d n=%d type=%d", iinfo, m, n, jtype);
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

    /* Use DBDSQR to compute only the singular values */
    cblas_dcopy(mnmin, BD, 1, S2, 1);
    if (mnmin > 0)
        cblas_dcopy(mnmin - 1, BE, 1, work, 1);

    dbdsqr(uplo_str, mnmin, 0, 0, 0, S2, work, VT, ldpt, U,
           ldpt, Z, ldx, work + mnmin, &iinfo);

    if (iinfo != 0) {
        snprintf(ctx, sizeof(ctx),
                 "DBDSQR(values) info=%d m=%d n=%d type=%d", iinfo, m, n, jtype);
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
    dbdt03(uplo_str, mnmin, 1, BD, BE, U, ldpt, S1, VT, ldpt,
           work, &result[3]);
    /* Test 5:  Check the computation Z := U' * Y */
    dbdt02(mnmin, NRHS, Y, ldx, Z, ldx, U, ldpt, work, &result[4]);
    /* Test 6:  Check the orthogonality of U */
    dort01("Columns", mnmin, mnmin, U, ldpt, work, lwork, &result[5]);
    /* Test 7:  Check the orthogonality of VT */
    dort01("Rows", mnmin, mnmin, VT, ldpt, work, lwork, &result[6]);

    /* Test 8:  Check that the singular values are sorted in
     *          non-increasing order and are non-negative */
    result[7] = check_sv_order(S1, mnmin, ulpinv);

    /* Test 9:  Compare DBDSQR with and without singular vectors */
    {
        f64 temp2 = 0.0;
        for (int j = 0; j < mnmin; j++) {
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
        for (int j = 0; j <= log2ui; j++) {
            dsvdch(mnmin, BD, BE, S1, temp1, &iinfo);
            if (iinfo == 0)
                break;
            temp1 = temp1 * 2.0;
        }
        result[9] = temp1;
    }

    /* ----------------------------------------------------------- */
    /* Use DBDSQR to form the decomposition A := (QU) S (VT PT)  */
    /* dchkbd.f lines 1015-1036                                    */
    /* ----------------------------------------------------------- */

    if (!bidiag) {
        cblas_dcopy(mnmin, BD, 1, S2, 1);
        if (mnmin > 0)
            cblas_dcopy(mnmin - 1, BE, 1, work, 1);

        dbdsqr(uplo_str, mnmin, n, m, NRHS, S2, work, PT, ldpt,
               Q, ldq, Y, ldx, work + mnmin, &iinfo);

        /* Test 11:  Check the decomposition A := Q*U * S2 * VT*PT */
        f64 dumma[1] = {0.0};
        dbdt01(m, n, 0, A, lda, Q, ldq, S2, dumma, PT,
               ldpt, work, &result[10]);
        /* Test 12:  Check the computation Z := U' * Q' * X */
        dbdt02(m, NRHS, X, ldx, Y, ldx, Q, ldq, work, &result[11]);
        /* Test 13:  Check the orthogonality of Q*U */
        dort01("Columns", m, mq, Q, ldq, work, lwork, &result[12]);
        /* Test 14:  Check the orthogonality of VT*PT */
        dort01("Rows", mnmin, n, PT, ldpt, work, lwork, &result[13]);
    }

    /* ----------------------------------------------------------- */
    /* Use DBDSDC to form the SVD of the bidiagonal matrix B:     */
    /* B := U * S1 * VT                                           */
    /* dchkbd.f lines 1038-1125                                    */
    /* ----------------------------------------------------------- */

    {
        cblas_dcopy(mnmin, BD, 1, S1, 1);
        if (mnmin > 0)
            cblas_dcopy(mnmin - 1, BE, 1, work, 1);
        dlaset("Full", mnmin, mnmin, 0.0, 1.0, U, ldpt);
        dlaset("Full", mnmin, mnmin, 0.0, 1.0, VT, ldpt);

        f64 dum[1] = {0.0};
        int idum[1] = {0};
        dbdsdc(uplo_str, "I", mnmin, S1, work, U, ldpt, VT, ldpt,
               dum, idum, work + mnmin, iwork, &iinfo);

        if (iinfo != 0) {
            snprintf(ctx, sizeof(ctx),
                     "DBDSDC(vects) info=%d m=%d n=%d type=%d", iinfo, m, n, jtype);
            set_test_context(ctx);
            if (iinfo < 0) {
                assert_info_success(iinfo);
                clear_test_context();
                return;
            } else {
                result[14] = ulpinv;
                goto results;
            }
        }

        cblas_dcopy(mnmin, BD, 1, S2, 1);
        if (mnmin > 0)
            cblas_dcopy(mnmin - 1, BE, 1, work, 1);

        f64 dum2[1] = {0.0};
        f64 dum3[1] = {0.0};
        int idum2[1] = {0};
        dbdsdc(uplo_str, "N", mnmin, S2, work, dum2, 1, dum3, 1,
               dum, idum2, work + mnmin, iwork, &iinfo);

        if (iinfo != 0) {
            snprintf(ctx, sizeof(ctx),
                     "DBDSDC(values) info=%d m=%d n=%d type=%d", iinfo, m, n, jtype);
            set_test_context(ctx);
            if (iinfo < 0) {
                assert_info_success(iinfo);
                clear_test_context();
                return;
            } else {
                result[17] = ulpinv;
                goto results;
            }
        }

        /* Test 15:  Check the decomposition B := U * S1 * VT */
        dbdt03(uplo_str, mnmin, 1, BD, BE, U, ldpt, S1, VT, ldpt,
               work, &result[14]);
        /* Test 16:  Check the orthogonality of U */
        dort01("Columns", mnmin, mnmin, U, ldpt, work, lwork, &result[15]);
        /* Test 17:  Check the orthogonality of VT */
        dort01("Rows", mnmin, mnmin, VT, ldpt, work, lwork, &result[16]);

        /* Test 18:  Check that the singular values are sorted in
         *           non-increasing order and are non-negative */
        result[17] = check_sv_order(S1, mnmin, ulpinv);

        /* Test 19:  Compare DBDSDC with and without singular vectors */
        result[18] = compare_sv(S1, S2, mnmin, ulp, unfl);
    }

    /* ----------------------------------------------------------- */
    /* Use DBDSVDX to compute the SVD of the bidiagonal matrix B: */
    /* B := U * S1 * VT                                           */
    /* dchkbd.f lines 1127-1251                                    */
    /* ----------------------------------------------------------- */

    if (jtype == 10 || jtype == 16) {
        for (int j = 19; j < NTESTS; j++)
            result[j] = 0.0;
        goto results;
    }

    {
        int iwbs   = 0;
        int iwbd   = iwbs + mnmin;
        int iwbe   = iwbd + mnmin;
        int iwbz   = iwbe + mnmin;
        int iwwork = iwbz + 2 * mnmin * (mnmin + 1);
        int mnmin2 = (mnmin * 2 > 1) ? mnmin * 2 : 1;

        cblas_dcopy(mnmin, BD, 1, work + iwbd, 1);
        if (mnmin > 0)
            cblas_dcopy(mnmin - 1, BE, 1, work + iwbe, 1);

        dbdsvdx(uplo_str, "V", "A", mnmin, work + iwbd,
                work + iwbe, 0.0, 0.0, 0, 0, &ns1, S1,
                work + iwbz, mnmin2, work + iwwork,
                iwork, &iinfo);

        if (iinfo != 0) {
            snprintf(ctx, sizeof(ctx),
                     "DBDSVDX(vects,A) info=%d m=%d n=%d type=%d",
                     iinfo, m, n, jtype);
            set_test_context(ctx);
            if (iinfo < 0) {
                assert_info_success(iinfo);
                clear_test_context();
                return;
            } else {
                result[19] = ulpinv;
                goto results;
            }
        }

        {
            int j = iwbz;
            for (int i = 0; i < ns1; i++) {
                cblas_dcopy(mnmin, work + j, 1, U + i * ldpt, 1);
                j += mnmin;
                cblas_dcopy(mnmin, work + j, 1, VT + i, ldpt);
                j += mnmin;
            }
        }

        if (jtype == 9) {
            result[23] = 0.0;
            goto results;
        }

        cblas_dcopy(mnmin, BD, 1, work + iwbd, 1);
        if (mnmin > 0)
            cblas_dcopy(mnmin - 1, BE, 1, work + iwbe, 1);

        dbdsvdx(uplo_str, "N", "A", mnmin, work + iwbd,
                work + iwbe, 0.0, 0.0, 0, 0, &ns2, S2,
                work + iwbz, mnmin2, work + iwwork,
                iwork, &iinfo);

        if (iinfo != 0) {
            snprintf(ctx, sizeof(ctx),
                     "DBDSVDX(values,A) info=%d m=%d n=%d type=%d",
                     iinfo, m, n, jtype);
            set_test_context(ctx);
            if (iinfo < 0) {
                assert_info_success(iinfo);
                clear_test_context();
                return;
            } else {
                result[23] = ulpinv;
                goto results;
            }
        }

        /* Save S1 for tests 30-34 */
        cblas_dcopy(mnmin, S1, 1, work + iwbs, 1);

        /* Test 20:  Check the decomposition B := U * S1 * VT */
        dbdt03(uplo_str, mnmin, 1, BD, BE, U, ldpt, S1, VT, ldpt,
               work + iwbs + mnmin, &result[19]);
        /* Test 21:  Check the orthogonality of U */
        dort01("Columns", mnmin, mnmin, U, ldpt,
               work + iwbs + mnmin, lwork - mnmin, &result[20]);
        /* Test 22:  Check the orthogonality of VT */
        dort01("Rows", mnmin, mnmin, VT, ldpt,
               work + iwbs + mnmin, lwork - mnmin, &result[21]);

        /* Test 23:  Check that the singular values are sorted in
         *           non-increasing order and are non-negative */
        result[22] = check_sv_order(S1, mnmin, ulpinv);

        /* Test 24:  Compare DBDSVDX with and without singular vectors */
        result[23] = compare_sv(S1, S2, mnmin, ulp, unfl);
        anorm_sv = (mnmin > 0) ? S1[0] : 0.0;

    /* ----------------------------------------------------------- */
    /* Use DBDSVDX with RANGE='I': choose random values for IL    */
    /* and IU, and ask for the IL-th through IU-th singular values */
    /* and corresponding vectors.                                  */
    /* dchkbd.f lines 1253-1366                                    */
    /* ----------------------------------------------------------- */

    {
        uint64_t iseed2[4];
        memcpy(iseed2, rng, 4 * sizeof(uint64_t));

        int il, iu;
        if (mnmin <= 1) {
            il = 0;
            iu = mnmin - 1;
        } else {
            il = (int)((mnmin - 1) * rng_uniform(iseed2));
            iu = (int)((mnmin - 1) * rng_uniform(iseed2));
            if (iu < il) {
                int itemp = iu;
                iu = il;
                il = itemp;
            }
        }

        cblas_dcopy(mnmin, BD, 1, work + iwbd, 1);
        if (mnmin > 0)
            cblas_dcopy(mnmin - 1, BE, 1, work + iwbe, 1);

        dbdsvdx(uplo_str, "V", "I", mnmin, work + iwbd,
                work + iwbe, 0.0, 0.0, il, iu, &ns1, S1,
                work + iwbz, mnmin2, work + iwwork,
                iwork, &iinfo);

        if (iinfo != 0) {
            snprintf(ctx, sizeof(ctx),
                     "DBDSVDX(vects,I) info=%d m=%d n=%d type=%d",
                     iinfo, m, n, jtype);
            set_test_context(ctx);
            if (iinfo < 0) {
                assert_info_success(iinfo);
                clear_test_context();
                return;
            } else {
                result[24] = ulpinv;
                goto results;
            }
        }

        {
            int j = iwbz;
            for (int i = 0; i < ns1; i++) {
                cblas_dcopy(mnmin, work + j, 1, U + i * ldpt, 1);
                j += mnmin;
                cblas_dcopy(mnmin, work + j, 1, VT + i, ldpt);
                j += mnmin;
            }
        }

        cblas_dcopy(mnmin, BD, 1, work + iwbd, 1);
        if (mnmin > 0)
            cblas_dcopy(mnmin - 1, BE, 1, work + iwbe, 1);

        dbdsvdx(uplo_str, "N", "I", mnmin, work + iwbd,
                work + iwbe, 0.0, 0.0, il, iu, &ns2, S2,
                work + iwbz, mnmin2, work + iwwork,
                iwork, &iinfo);

        if (iinfo != 0) {
            snprintf(ctx, sizeof(ctx),
                     "DBDSVDX(values,I) info=%d m=%d n=%d type=%d",
                     iinfo, m, n, jtype);
            set_test_context(ctx);
            if (iinfo < 0) {
                assert_info_success(iinfo);
                clear_test_context();
                return;
            } else {
                result[28] = ulpinv;
                goto results;
            }
        }

        /* Test 25:  Check S1 - U' * B * VT' */
        dbdt04(uplo_str, mnmin, BD, BE, S1, ns1, U,
               ldpt, VT, ldpt, work + iwbs + mnmin,
               &result[24]);
        /* Test 26:  Check the orthogonality of U */
        dort01("Columns", mnmin, ns1, U, ldpt,
               work + iwbs + mnmin, lwork - mnmin, &result[25]);
        /* Test 27:  Check the orthogonality of VT */
        dort01("Rows", ns1, mnmin, VT, ldpt,
               work + iwbs + mnmin, lwork - mnmin, &result[26]);

        /* Test 28:  Check that the singular values are sorted in
         *           non-increasing order and are non-negative */
        result[27] = check_sv_order(S1, ns1, ulpinv);

        /* Test 29:  Compare DBDSVDX with and without singular vectors */
        result[28] = compare_sv(S1, S2, ns1, ulp, unfl);

        /* ----------------------------------------------------------- */
        /* Use DBDSVDX with RANGE='V': determine the values VL and VU */
        /* of the IL-th and IU-th singular values and ask for all     */
        /* singular values in this range.                              */
        /* dchkbd.f lines 1368-1490                                    */
        /* ----------------------------------------------------------- */

        cblas_dcopy(mnmin, work + iwbs, 1, S1, 1);

        f64 vl, vu;
        if (mnmin > 0) {
            if (il != 0) {
                f64 d1 = 0.5 * fabs(S1[il] - S1[il - 1]);
                f64 d2 = ulp * anorm_sv;
                f64 d3 = 2.0 * rtunfl;
                f64 mx = d1;
                if (d2 > mx) mx = d2;
                if (d3 > mx) mx = d3;
                vu = S1[il] + mx;
            } else {
                f64 d1 = 0.5 * fabs(S1[mnmin - 1] - S1[0]);
                f64 d2 = ulp * anorm_sv;
                f64 d3 = 2.0 * rtunfl;
                f64 mx = d1;
                if (d2 > mx) mx = d2;
                if (d3 > mx) mx = d3;
                vu = S1[0] + mx;
            }
            if (iu != ns1 - 1) {
                f64 d1 = ulp * anorm_sv;
                f64 d2 = 2.0 * rtunfl;
                f64 d3 = 0.5 * fabs(S1[iu + 1] - S1[iu]);
                f64 mx = d1;
                if (d2 > mx) mx = d2;
                if (d3 > mx) mx = d3;
                vl = S1[iu] - mx;
            } else {
                f64 d1 = ulp * anorm_sv;
                f64 d2 = 2.0 * rtunfl;
                f64 d3 = 0.5 * fabs(S1[mnmin - 1] - S1[0]);
                f64 mx = d1;
                if (d2 > mx) mx = d2;
                if (d3 > mx) mx = d3;
                vl = S1[ns1 - 1] - mx;
            }
            if (vl < 0.0) vl = 0.0;
            if (vu < 0.0) vu = 0.0;
            if (vl >= vu) vu = (vu * 2.0 > vu + vl + 0.5) ? vu * 2.0 : vu + vl + 0.5;
        } else {
            vl = 0.0;
            vu = 1.0;
        }

        cblas_dcopy(mnmin, BD, 1, work + iwbd, 1);
        if (mnmin > 0)
            cblas_dcopy(mnmin - 1, BE, 1, work + iwbe, 1);

        dbdsvdx(uplo_str, "V", "V", mnmin, work + iwbd,
                work + iwbe, vl, vu, 0, 0, &ns1, S1,
                work + iwbz, mnmin2, work + iwwork,
                iwork, &iinfo);

        if (iinfo != 0) {
            snprintf(ctx, sizeof(ctx),
                     "DBDSVDX(vects,V) info=%d m=%d n=%d type=%d",
                     iinfo, m, n, jtype);
            set_test_context(ctx);
            if (iinfo < 0) {
                assert_info_success(iinfo);
                clear_test_context();
                return;
            } else {
                result[29] = ulpinv;
                goto results;
            }
        }

        {
            int j = iwbz;
            for (int i = 0; i < ns1; i++) {
                cblas_dcopy(mnmin, work + j, 1, U + i * ldpt, 1);
                j += mnmin;
                cblas_dcopy(mnmin, work + j, 1, VT + i, ldpt);
                j += mnmin;
            }
        }

        cblas_dcopy(mnmin, BD, 1, work + iwbd, 1);
        if (mnmin > 0)
            cblas_dcopy(mnmin - 1, BE, 1, work + iwbe, 1);

        dbdsvdx(uplo_str, "N", "V", mnmin, work + iwbd,
                work + iwbe, vl, vu, 0, 0, &ns2, S2,
                work + iwbz, mnmin2, work + iwwork,
                iwork, &iinfo);

        if (iinfo != 0) {
            snprintf(ctx, sizeof(ctx),
                     "DBDSVDX(values,V) info=%d m=%d n=%d type=%d",
                     iinfo, m, n, jtype);
            set_test_context(ctx);
            if (iinfo < 0) {
                assert_info_success(iinfo);
                clear_test_context();
                return;
            } else {
                result[33] = ulpinv;
                goto results;
            }
        }

        /* Test 30:  Check S1 - U' * B * VT' */
        dbdt04(uplo_str, mnmin, BD, BE, S1, ns1, U,
               ldpt, VT, ldpt, work + iwbs + mnmin,
               &result[29]);
        /* Test 31:  Check the orthogonality of U */
        dort01("Columns", mnmin, ns1, U, ldpt,
               work + iwbs + mnmin, lwork - mnmin, &result[30]);
        /* Test 32:  Check the orthogonality of VT */
        dort01("Rows", ns1, mnmin, VT, ldpt,
               work + iwbs + mnmin, lwork - mnmin, &result[31]);

        /* Test 33:  Check that the singular values are sorted in
         *           non-increasing order and are non-negative
         *           NOTE: Fortran dchkbd.f lines 1472-1480 write RESULT(28)
         *           instead of RESULT(33). We fix this bug here. */
        result[32] = check_sv_order(S1, ns1, ulpinv);

        /* Test 34:  Compare DBDSVDX with and without singular vectors */
        result[33] = compare_sv(S1, S2, ns1, ulp, unfl);
    }
    }

results:
    /* End of Loop -- Check for RESULT(j) >= THRESH */
    for (int j = 0; j < NTESTS; j++) {
        if (result[j] >= THRESH) {
            snprintf(ctx, sizeof(ctx),
                     "dchkbd M=%d N=%d type=%d test(%d)=%.4g",
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

static void test_dchkbd(void** state)
{
    (void)state;
    dchkbd_params_t* params = (dchkbd_params_t*)(*state);
    run_dchkbd_single(params);
}

/* ===================================================================== */
/* Test table generation                                                 */
/* ===================================================================== */

#define MAX_TESTS (NSIZES * MAXTYP)

static dchkbd_params_t g_params[MAX_TESTS];
static struct CMUnitTest g_tests[MAX_TESTS];
static int g_ntests = 0;

static void generate_test_table(void)
{
    g_ntests = 0;
    for (int js = 0; js < NSIZES; js++) {
        for (int jt = 1; jt <= MAXTYP; jt++) {
            dchkbd_params_t* p = &g_params[g_ntests];
            p->jsize = js;
            p->jtype = jt;
            snprintf(p->name, sizeof(p->name),
                     "dchkbd M=%d N=%d type=%d",
                     MVAL[js], NVAL[js], jt);

            g_tests[g_ntests].name = p->name;
            g_tests[g_ntests].test_func = test_dchkbd;
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
    return cmocka_run_group_tests_name("dchkbd", g_tests, group_setup,
                                       group_teardown);
}
