/**
 * @file test_cchkhs.c
 * @brief Hessenberg Schur test driver - port of LAPACK TESTING/EIG/zchkhs.f
 *
 * Tests the nonsymmetric eigenvalue problem routines (complex).
 *
 *         CGEHRD factors A as  U H U* , where * means conjugate transpose,
 *         H is hessenberg, and U is a unitary matrix.
 *
 *         CUNGHR generates the unitary matrix U.
 *
 *         CUNMHR multiplies a matrix by the unitary matrix U.
 *
 *         CHSEQR factors H as  Z T Z* , where Z is unitary and
 *         T is upper triangular (Schur form), and the eigenvalue vector W.
 *
 *         CTREVC computes the left and right eigenvector matrices
 *         L and R for T.
 *
 *         CHSEIN computes the left and right eigenvector matrices
 *         Y and X for H, using inverse iteration.
 *
 * Test ratios (16 total):
 *                    *
 *   (1)  | A - U H U  | / ( |A| n ulp )
 *                *
 *   (2)  | I - UU  | / ( n ulp )
 *                    *
 *   (3)  | H - Z T Z  | / ( |H| n ulp )
 *                *
 *   (4)  | I - ZZ  | / ( n ulp )
 *                      *
 *   (5)  | A - UZ T (UZ)  | / ( |A| n ulp )
 *                  *
 *   (6)  | I - (UZ)(UZ)  | / ( n ulp )
 *
 *   (7)  | T2 - T1 | / ( |T| n ulp )
 *
 *   (8)  | W2 - W1 | / ( max(|W1|,|W2|) ulp )
 *
 *   (9)  | TR - RW | / ( |T| |R| ulp )
 *
 *   (10) | L*T - W*L | / ( |T| |L| ulp )
 *
 *   (11) | HX - XW | / ( |H| |X| ulp )     (from inverse iteration)
 *
 *   (12) | Y*H - W*Y | / ( |H| |Y| ulp )   (from inverse iteration)
 *
 *   (13) | AX - XW | / ( |A| |X| ulp )      (CUNMHR on inverse iteration)
 *
 *   (14) | Y*A - W*Y | / ( |A| |Y| ulp )    (CUNMHR on inverse iteration)
 *
 *   (15) | AR - RW | / ( |A| |R| ulp )      (CTREVC3 backtransform)
 *
 *   (16) | L*A - W*L | / ( |A| |L| ulp )   (CTREVC3 backtransform)
 *
 * Matrix types: 21 types (MAXTYP = 21)
 */

#include "test_harness.h"
#include "verify.h"
#include "test_rng.h"
#include "semicolon_cblas.h"
#include <math.h>
#include <string.h>

#define THRESH 30.0f
#define MAXTYP 21
#define NTEST  16

static const INT NVAL[] = {0, 1, 2, 3, 5, 10, 20, 50};
#define NNVAL (sizeof(NVAL) / sizeof(NVAL[0]))

/* ===================================================================== */
/* DATA arrays from zchkhs.f lines 481-486                               */
/* ===================================================================== */

/*                  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 */
static const INT ktype[MAXTYP] = {
    1, 2, 3, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 9, 9, 9
};

static const INT kmagn[MAXTYP] = {
    1, 1, 1, 1, 1, 1, 2, 3, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 1, 2, 3
};

static const INT kmode[MAXTYP] = {
    0, 0, 0, 4, 3, 1, 4, 4, 4, 3, 1, 5, 4, 3, 1, 5, 5, 5, 4, 3, 1
};

static const INT kconds[MAXTYP] = {
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 0, 0, 0
};

/* ===================================================================== */
/* Test parameters and workspace                                         */
/* ===================================================================== */

typedef struct {
    INT jsize;    /* index into NVAL[] */
    INT jtype;    /* matrix type 1..21 */
    char name[96];
} zchkhs_params_t;

typedef struct {
    INT nmax;
    c64* A;
    c64* H;
    c64* T1;
    c64* T2;
    c64* U;
    c64* Z;
    c64* UZ;
    c64* UU;
    c64* evectl;
    c64* evectr;
    c64* evecty;
    c64* evectx;
    c64* W1;
    c64* W3;
    c64* tau;
    c64* work;
    INT nwork;
    f32* rwork;
    INT* iwork;
    INT* selectarr;
    uint64_t rng_state[4];
} zchkhs_workspace_t;

static zchkhs_workspace_t* g_ws = NULL;

/* ===================================================================== */
/* Group setup / teardown                                                */
/* ===================================================================== */

static int group_setup(void** state)
{
    (void)state;

    g_ws = calloc(1, sizeof(zchkhs_workspace_t));
    if (!g_ws) return -1;

    g_ws->nmax = 0;
    for (size_t i = 0; i < NNVAL; i++) {
        if (NVAL[i] > g_ws->nmax) g_ws->nmax = NVAL[i];
    }
    if (g_ws->nmax < 1) g_ws->nmax = 1;

    INT nmax = g_ws->nmax;
    INT n2 = nmax * nmax;

    g_ws->A      = malloc(n2 * sizeof(c64));
    g_ws->H      = malloc(n2 * sizeof(c64));
    g_ws->T1     = malloc(n2 * sizeof(c64));
    g_ws->T2     = malloc(n2 * sizeof(c64));
    g_ws->U      = malloc(n2 * sizeof(c64));
    g_ws->Z      = malloc(n2 * sizeof(c64));
    g_ws->UZ     = malloc(n2 * sizeof(c64));
    g_ws->UU     = malloc(n2 * sizeof(c64));
    g_ws->evectl = malloc(n2 * sizeof(c64));
    g_ws->evectr = malloc(n2 * sizeof(c64));
    g_ws->evecty = malloc(n2 * sizeof(c64));
    g_ws->evectx = malloc(n2 * sizeof(c64));

    g_ws->W1  = malloc(nmax * sizeof(c64));
    g_ws->W3  = malloc(nmax * sizeof(c64));
    g_ws->tau  = malloc(nmax * sizeof(c64));

    g_ws->nwork = 4 * n2 + 2;
    if (g_ws->nwork < 1) g_ws->nwork = 1;
    g_ws->work = malloc(g_ws->nwork * sizeof(c64));

    g_ws->rwork     = malloc(nmax * sizeof(f32));
    g_ws->iwork     = malloc(2 * nmax * sizeof(INT));
    g_ws->selectarr = malloc(nmax * sizeof(INT));

    if (!g_ws->A || !g_ws->H || !g_ws->T1 || !g_ws->T2 ||
        !g_ws->U || !g_ws->Z || !g_ws->UZ || !g_ws->UU ||
        !g_ws->evectl || !g_ws->evectr || !g_ws->evecty || !g_ws->evectx ||
        !g_ws->W1 || !g_ws->W3 || !g_ws->tau ||
        !g_ws->work || !g_ws->rwork || !g_ws->iwork || !g_ws->selectarr) {
        return -1;
    }

    rng_seed(g_ws->rng_state, 0x7CE4B501ULL);
    return 0;
}

static int group_teardown(void** state)
{
    (void)state;

    if (g_ws) {
        free(g_ws->A);
        free(g_ws->H);
        free(g_ws->T1);
        free(g_ws->T2);
        free(g_ws->U);
        free(g_ws->Z);
        free(g_ws->UZ);
        free(g_ws->UU);
        free(g_ws->evectl);
        free(g_ws->evectr);
        free(g_ws->evecty);
        free(g_ws->evectx);
        free(g_ws->W1);
        free(g_ws->W3);
        free(g_ws->tau);
        free(g_ws->work);
        free(g_ws->rwork);
        free(g_ws->iwork);
        free(g_ws->selectarr);
        free(g_ws);
        g_ws = NULL;
    }
    return 0;
}

/* ===================================================================== */
/* Single test case                                                      */
/* ===================================================================== */

static void run_zchkhs_single(zchkhs_params_t* params)
{
    const INT n = NVAL[params->jsize];
    const INT jtype = params->jtype;
    const INT jt = jtype - 1;

    if (n == 0) return;

    const INT lda = g_ws->nmax;
    const INT ldu = g_ws->nmax;
    const INT nwork = g_ws->nwork;

    c64* A      = g_ws->A;
    c64* H      = g_ws->H;
    c64* T1     = g_ws->T1;
    c64* T2     = g_ws->T2;
    c64* U      = g_ws->U;
    c64* Z      = g_ws->Z;
    c64* UZ     = g_ws->UZ;
    c64* UU     = g_ws->UU;
    c64* evectl = g_ws->evectl;
    c64* evectr = g_ws->evectr;
    c64* evecty = g_ws->evecty;
    c64* evectx = g_ws->evectx;
    c64* W1     = g_ws->W1;
    c64* W3     = g_ws->W3;
    c64* tau    = g_ws->tau;
    c64* work   = g_ws->work;
    f32*  rwork  = g_ws->rwork;
    INT*  iwork  = g_ws->iwork;
    INT*  selectarr = g_ws->selectarr;

    uint64_t* rng = g_ws->rng_state;

    const f32 unfl = slamch("S");
    const f32 ovfl = slamch("O");
    const f32 ulp = slamch("P");
    const f32 ulpinv = 1.0f / ulp;
    const f32 rtunfl = sqrtf(unfl);
    const f32 rtovfl = sqrtf(ovfl);
    const f32 rtulp = sqrtf(ulp);
    const f32 rtulpi = 1.0f / rtulp;

    const INT n1 = (n > 1) ? n : 1;
    const f32 aninv = 1.0f / (f32)n1;

    const c64 czero = CMPLXF(0.0f, 0.0f);
    const c64 cone  = CMPLXF(1.0f, 0.0f);

    f32 result[NTEST];
    for (INT i = 0; i < NTEST; i++)
        result[i] = 0.0f;

    char ctx[256];
    INT iinfo;

    /* ================================================================ */
    /* Generate test matrix A                                           */
    /* ================================================================ */

    if (jtype <= MAXTYP) {
        INT itype = ktype[jt];
        INT imode = kmode[jt];

        f32 anorm;
        switch (kmagn[jt]) {
            case 2:  anorm = (rtovfl * ulp) * aninv; break;
            case 3:  anorm = rtunfl * n * ulpinv;    break;
            default: anorm = 1.0f;                    break;
        }

        claset("Full", lda, n, czero, czero, A, lda);
        iinfo = 0;
        f32 cond = ulpinv;

        if (itype == 1) {
            /* Zero */
            iinfo = 0;

        } else if (itype == 2) {
            /* Identity */
            for (INT jcol = 0; jcol < n; jcol++)
                A[jcol + jcol * lda] = CMPLXF(anorm, 0.0f);

        } else if (itype == 3) {
            /* Jordan Block */
            for (INT jcol = 0; jcol < n; jcol++) {
                A[jcol + jcol * lda] = CMPLXF(anorm, 0.0f);
                if (jcol > 0)
                    A[jcol + (jcol - 1) * lda] = cone;
            }

        } else if (itype == 4) {
            /* Diagonal Matrix, eigenvalues specified */
            clatmr(n, n, "D", "N", work, imode, cond, cone,
                   "T", "N", work + n, 1, 1.0f,
                   work + 2 * n, 1, 1.0f, "N", iwork, 0, 0,
                   0.0f, anorm, "NO", A, lda, iwork, &iinfo, rng);

        } else if (itype == 5) {
            /* Hermitian, eigenvalues specified */
            clatms(n, n, "D", "H", rwork, imode, cond, anorm,
                   n, n, "N", A, lda, work, &iinfo, rng);

        } else if (itype == 6) {
            /* General, eigenvalues specified */
            f32 conds;
            if (kconds[jt] == 1)
                conds = 1.0f;
            else if (kconds[jt] == 2)
                conds = rtulpi;
            else
                conds = 0.0f;

            clatme(n, "D", work, imode, cond, cone,
                   "T", "T", "T", rwork, 4,
                   conds, n, n, anorm, A, lda, work + n,
                   &iinfo, rng);

        } else if (itype == 7) {
            /* Diagonal, random eigenvalues */
            INT idumma[1] = {0};
            clatmr(n, n, "D", "N", work, 6, 1.0f, cone,
                   "T", "N", work + n, 1, 1.0f,
                   work + 2 * n, 1, 1.0f, "N", idumma, 0, 0,
                   0.0f, anorm, "NO", A, lda, iwork, &iinfo, rng);

        } else if (itype == 8) {
            /* Hermitian, random eigenvalues */
            INT idumma[1] = {0};
            clatmr(n, n, "D", "H", work, 6, 1.0f, cone,
                   "T", "N", work + n, 1, 1.0f,
                   work + 2 * n, 1, 1.0f, "N", idumma, n, n,
                   0.0f, anorm, "NO", A, lda, iwork, &iinfo, rng);

        } else if (itype == 9) {
            /* General, random eigenvalues */
            INT idumma[1] = {0};
            clatmr(n, n, "D", "N", work, 6, 1.0f, cone,
                   "T", "N", work + n, 1, 1.0f,
                   work + 2 * n, 1, 1.0f, "N", idumma, n, n,
                   0.0f, anorm, "NO", A, lda, iwork, &iinfo, rng);

        } else if (itype == 10) {
            /* Triangular, random eigenvalues */
            INT idumma[1] = {0};
            clatmr(n, n, "D", "N", work, 6, 1.0f, cone,
                   "T", "N", work + n, 1, 1.0f,
                   work + 2 * n, 1, 1.0f, "N", idumma, n, 0,
                   0.0f, anorm, "NO", A, lda, iwork, &iinfo, rng);

        } else {
            iinfo = 1;
        }

        if (iinfo != 0) {
            snprintf(ctx, sizeof(ctx),
                     "zchkhs n=%d type=%d: Generator info=%d", n, jtype, iinfo);
            set_test_context(ctx);
            assert_info_success(iinfo);
            return;
        }
    }

    /* ================================================================ */
    /* Tests 1-2: CGEHRD + CUNGHR                                       */
    /* ================================================================ */

    clacpy(" ", n, n, A, lda, H, lda);

    cgehrd(n, 0, n - 1, H, lda, work, work + n, nwork - n, &iinfo);
    if (iinfo != 0) {
        result[0] = ulpinv;
        snprintf(ctx, sizeof(ctx),
                 "zchkhs n=%d type=%d: CGEHRD info=%d", n, jtype, iinfo);
        set_test_context(ctx);
        assert_info_success(iinfo);
        return;
    }

    /* Extract reflectors and zero sub-Hessenberg part */
    for (INT j = 0; j < n - 1; j++) {
        UU[j + 1 + j * ldu] = czero;
        for (INT i = j + 2; i < n; i++) {
            U[i + j * ldu] = H[i + j * lda];
            UU[i + j * ldu] = H[i + j * lda];
            H[i + j * lda] = czero;
        }
    }
    cblas_ccopy(n - 1, work, 1, tau, 1);

    cunghr(n, 0, n - 1, U, ldu, work, work + n, nwork - n, &iinfo);

    /* Tests 1-2 */
    chst01(n, 0, n - 1, A, lda, H, lda, U, ldu, work, nwork, rwork, result);

    /* ================================================================ */
    /* Tests 3-8: CHSEQR eigenvalues and Schur form                     */
    /* ================================================================ */

    /* Eigenvalues only (W3) */
    clacpy(" ", n, n, H, lda, T2, lda);
    result[2] = ulpinv;

    chseqr("E", "N", n, 0, n - 1, T2, lda, W3,
           UZ, ldu, work, nwork, &iinfo);
    if (iinfo != 0) {
        if (iinfo <= n + 2) {
            snprintf(ctx, sizeof(ctx),
                     "zchkhs n=%d type=%d: CHSEQR(E) info=%d", n, jtype, iinfo);
            set_test_context(ctx);
            assert_info_success(iinfo);
            return;
        }
    }

    /* Eigenvalues (W1) and Full Schur Form (T2) */
    clacpy(" ", n, n, H, lda, T2, lda);

    chseqr("S", "N", n, 0, n - 1, T2, lda, W1,
           UZ, ldu, work, nwork, &iinfo);
    if (iinfo != 0 && iinfo <= n + 2) {
        snprintf(ctx, sizeof(ctx),
                 "zchkhs n=%d type=%d: CHSEQR(S) info=%d", n, jtype, iinfo);
        set_test_context(ctx);
        assert_info_success(iinfo);
        return;
    }

    /* Eigenvalues (W1), Schur Form (T1), and Schur vectors (UZ) */
    clacpy(" ", n, n, H, lda, T1, lda);
    clacpy(" ", n, n, U, ldu, UZ, ldu);

    chseqr("S", "V", n, 0, n - 1, T1, lda, W1,
           UZ, ldu, work, nwork, &iinfo);
    if (iinfo != 0 && iinfo <= n + 2) {
        snprintf(ctx, sizeof(ctx),
                 "zchkhs n=%d type=%d: CHSEQR(V) info=%d", n, jtype, iinfo);
        set_test_context(ctx);
        assert_info_success(iinfo);
        return;
    }

    /* Compute Z = U* * UZ (conjugate transpose of U times UZ) */
    {
        const c64 alpha = cone;
        const c64 beta  = czero;
        cblas_cgemm(CblasColMajor, CblasConjTrans, CblasNoTrans,
                    n, n, n, &alpha, U, ldu, UZ, ldu, &beta, Z, ldu);
    }

    /* Tests 3-4: | H - Z T Z* | / ( |H| n ulp ) and | I - Z Z* | / ( n ulp ) */
    chst01(n, 0, n - 1, H, lda, T1, lda, Z, ldu, work, nwork, rwork, &result[2]);

    /* Tests 5-6: | A - UZ T (UZ)* | / ( |A| n ulp ) and | I - UZ (UZ)* | / ( n ulp ) */
    chst01(n, 0, n - 1, A, lda, T1, lda, UZ, ldu, work, nwork, rwork, &result[4]);

    /* Test 7: | T2 - T1 | / ( |T| n ulp ) */
    cget10(n, n, T2, lda, T1, lda, work, rwork, &result[6]);

    /* Test 8: | W3 - W1 | / ( max(|W1|,|W3|) ulp ) */
    {
        f32 temp1 = 0.0f;
        f32 temp2 = 0.0f;
        for (INT j = 0; j < n; j++) {
            f32 w1mag = cabsf(W1[j]);
            f32 w3mag = cabsf(W3[j]);
            f32 wmax = (w1mag > w3mag) ? w1mag : w3mag;
            if (wmax > temp1) temp1 = wmax;

            f32 wdiff = cabsf(W1[j] - W3[j]);
            if (wdiff > temp2) temp2 = wdiff;
        }
        f32 denom = (temp1 > temp2) ? temp1 : temp2;
        denom = ulp * denom;
        if (denom < unfl) denom = unfl;
        result[7] = temp2 / denom;
    }

    /* ================================================================ */
    /* Tests 9-10: CTREVC eigenvectors of T                             */
    /* ================================================================ */

    result[8] = ulpinv;

    /* Select every other eigenvector */
    for (INT j = 0; j < n; j++)
        selectarr[j] = 0;
    for (INT j = 0; j < n; j += 2)
        selectarr[j] = 1;

    /* Right eigenvectors: "All" */
    c64 cdumma[4];
    f32 dumma[4];
    INT in;
    ctrevc("R", "A", selectarr, n, T1, lda, cdumma, ldu,
           evectr, ldu, n, &in, work, rwork, &iinfo);
    if (iinfo != 0) {
        snprintf(ctx, sizeof(ctx),
                 "zchkhs n=%d type=%d: CTREVC(R,A) info=%d", n, jtype, iinfo);
        set_test_context(ctx);
        assert_info_success(iinfo);
        return;
    }

    /* Test 9: | TR - RW | / ( |T| |R| ulp ) */
    cget22("N", "N", "N", n, T1, lda, evectr, ldu, W1, work, rwork, dumma);
    result[8] = dumma[0];

    /* Compute selected right eigenvectors and confirm they agree */
    ctrevc("R", "S", selectarr, n, T1, lda, cdumma, ldu,
           evectl, ldu, n, &in, work, rwork, &iinfo);
    if (iinfo != 0) {
        snprintf(ctx, sizeof(ctx),
                 "zchkhs n=%d type=%d: CTREVC(R,S) info=%d", n, jtype, iinfo);
        set_test_context(ctx);
        assert_info_success(iinfo);
        return;
    }

    {
        INT k = 0;
        for (INT j = 0; j < n; j++) {
            if (selectarr[j]) {
                for (INT jj = 0; jj < n; jj++) {
                    if (evectr[jj + j * ldu] != evectl[jj + k * ldu])
                        break;
                }
                k++;
            }
        }
    }

    /* Left eigenvectors: "All" */
    result[9] = ulpinv;
    ctrevc("L", "A", selectarr, n, T1, lda, evectl, ldu,
           cdumma, ldu, n, &in, work, rwork, &iinfo);
    if (iinfo != 0) {
        snprintf(ctx, sizeof(ctx),
                 "zchkhs n=%d type=%d: CTREVC(L,A) info=%d", n, jtype, iinfo);
        set_test_context(ctx);
        assert_info_success(iinfo);
        return;
    }

    /* Test 10: | L*T - W*L | / ( |T| |L| ulp ) */
    cget22("C", "N", "C", n, T1, lda, evectl, ldu, W1, work, rwork, &dumma[2]);
    result[9] = dumma[2];

    /* Compute selected left eigenvectors and confirm they agree */
    ctrevc("L", "S", selectarr, n, T1, lda, evectr, ldu,
           cdumma, ldu, n, &in, work, rwork, &iinfo);
    if (iinfo != 0) {
        snprintf(ctx, sizeof(ctx),
                 "zchkhs n=%d type=%d: CTREVC(L,S) info=%d", n, jtype, iinfo);
        set_test_context(ctx);
        assert_info_success(iinfo);
        return;
    }

    {
        INT k = 0;
        for (INT j = 0; j < n; j++) {
            if (selectarr[j]) {
                for (INT jj = 0; jj < n; jj++) {
                    if (evectl[jj + j * ldu] != evectr[jj + k * ldu])
                        break;
                }
                k++;
            }
        }
    }

    /* ================================================================ */
    /* Tests 11-12: CHSEIN eigenvectors of H via inverse iteration      */
    /* ================================================================ */

    result[10] = ulpinv;
    for (INT j = 0; j < n; j++)
        selectarr[j] = 1;

    chsein("R", "Q", "N", selectarr, n, H, lda,
           W3, cdumma, ldu, evectx, ldu, n1, &in,
           work, rwork, iwork, iwork + lda, &iinfo);
    if (iinfo != 0) {
        snprintf(ctx, sizeof(ctx),
                 "zchkhs n=%d type=%d: CHSEIN(R) info=%d", n, jtype, iinfo);
        set_test_context(ctx);
        if (iinfo < 0) {
            assert_info_success(iinfo);
            return;
        }
    } else {
        /* Test 11: | HX - XW | / ( |H| |X| ulp ) */
        cget22("N", "N", "N", n, H, lda, evectx, ldu, W3,
               work, rwork, dumma);
        if (dumma[0] < ulpinv)
            result[10] = dumma[0] * aninv;
    }

    /* Left eigenvectors of H via inverse iteration */
    result[11] = ulpinv;
    for (INT j = 0; j < n; j++)
        selectarr[j] = 1;

    chsein("L", "Q", "N", selectarr, n, H, lda,
           W3, evecty, ldu, cdumma, ldu, n1, &in,
           work, rwork, iwork, iwork + lda, &iinfo);
    if (iinfo != 0) {
        snprintf(ctx, sizeof(ctx),
                 "zchkhs n=%d type=%d: CHSEIN(L) info=%d", n, jtype, iinfo);
        set_test_context(ctx);
        if (iinfo < 0) {
            assert_info_success(iinfo);
            return;
        }
    } else {
        /* Test 12: | Y*H - W*Y | / ( |H| |Y| ulp ) */
        cget22("C", "N", "C", n, H, lda, evecty, ldu, W3,
               work, rwork, &dumma[2]);
        if (dumma[2] < ulpinv)
            result[11] = dumma[2] * aninv;
    }

    /* ================================================================ */
    /* Tests 13-14: CUNMHR back-transform CHSEIN eigenvectors           */
    /* ================================================================ */

    result[12] = ulpinv;

    cunmhr("L", "N", n, n, 0, n - 1, UU, ldu, tau, evectx, ldu,
           work, nwork, &iinfo);
    if (iinfo != 0) {
        snprintf(ctx, sizeof(ctx),
                 "zchkhs n=%d type=%d: CUNMHR(R) info=%d", n, jtype, iinfo);
        set_test_context(ctx);
        if (iinfo < 0) {
            assert_info_success(iinfo);
            return;
        }
    } else {
        /* Test 13: | AX - XW | / ( |A| |X| ulp ) */
        cget22("N", "N", "N", n, A, lda, evectx, ldu, W3,
               work, rwork, dumma);
        if (dumma[0] < ulpinv)
            result[12] = dumma[0] * aninv;
    }

    result[13] = ulpinv;

    cunmhr("L", "N", n, n, 0, n - 1, UU, ldu, tau, evecty, ldu,
           work, nwork, &iinfo);
    if (iinfo != 0) {
        snprintf(ctx, sizeof(ctx),
                 "zchkhs n=%d type=%d: CUNMHR(L) info=%d", n, jtype, iinfo);
        set_test_context(ctx);
        if (iinfo < 0) {
            assert_info_success(iinfo);
            return;
        }
    } else {
        /* Test 14: | Y*A - W*Y | / ( |A| |Y| ulp ) */
        cget22("C", "N", "C", n, A, lda, evecty, ldu, W3,
               work, rwork, &dumma[2]);
        if (dumma[2] < ulpinv)
            result[13] = dumma[2] * aninv;
    }

    /* ================================================================ */
    /* Tests 15-16: CTREVC3 eigenvectors of A via back-transform        */
    /* ================================================================ */

    result[14] = ulpinv;

    clacpy(" ", n, n, UZ, ldu, evectr, ldu);

    ctrevc3("R", "B", selectarr, n, T1, lda, cdumma, ldu,
            evectr, ldu, n, &in, work, nwork, rwork, n, &iinfo);
    if (iinfo != 0) {
        snprintf(ctx, sizeof(ctx),
                 "zchkhs n=%d type=%d: CTREVC3(R,B) info=%d", n, jtype, iinfo);
        set_test_context(ctx);
        assert_info_success(iinfo);
        return;
    }

    /* Test 15: | AR - RW | / ( |A| |R| ulp ) */
    cget22("N", "N", "N", n, A, lda, evectr, ldu, W1, work, rwork, dumma);
    result[14] = dumma[0];

    /* Left eigenvectors via back-transform */
    result[15] = ulpinv;

    clacpy(" ", n, n, UZ, ldu, evectl, ldu);

    ctrevc3("L", "B", selectarr, n, T1, lda, evectl, ldu,
            cdumma, ldu, n, &in, work, nwork, rwork, n, &iinfo);
    if (iinfo != 0) {
        snprintf(ctx, sizeof(ctx),
                 "zchkhs n=%d type=%d: CTREVC3(L,B) info=%d", n, jtype, iinfo);
        set_test_context(ctx);
        assert_info_success(iinfo);
        return;
    }

    /* Test 16: | L*A - W*L | / ( |A| |L| ulp ) */
    cget22("C", "N", "C", n, A, lda, evectl, ldu, W1, work, rwork, &dumma[2]);
    result[15] = dumma[2];

    /* ================================================================ */
    /* Check results against threshold                                  */
    /* ================================================================ */

    for (INT jr = 0; jr < NTEST; jr++) {
        snprintf(ctx, sizeof(ctx), "zchkhs n=%d type=%d TEST %d",
                 n, jtype, jr + 1);
        set_test_context(ctx);
        assert_residual_ok(result[jr]);
    }
    clear_test_context();
}

/* ===================================================================== */
/* CMocka test wrapper                                                   */
/* ===================================================================== */

static void test_zchkhs_case(void** state)
{
    zchkhs_params_t* params = *state;
    run_zchkhs_single(params);
}

/* ===================================================================== */
/* Build test array and main                                             */
/* ===================================================================== */

#define MAX_TESTS (NNVAL * MAXTYP)

static zchkhs_params_t g_params[MAX_TESTS];
static struct CMUnitTest g_tests[MAX_TESTS];
static INT g_num_tests = 0;

static void build_test_array(void)
{
    g_num_tests = 0;

    for (size_t in = 0; in < NNVAL; in++) {
        INT n = NVAL[in];
        for (INT jtype = 1; jtype <= MAXTYP; jtype++) {
            zchkhs_params_t* p = &g_params[g_num_tests];
            p->jsize = (INT)in;
            p->jtype = jtype;
            snprintf(p->name, sizeof(p->name),
                     "zchkhs_n%d_type%d", n, jtype);

            g_tests[g_num_tests].name = p->name;
            g_tests[g_num_tests].test_func = test_zchkhs_case;
            g_tests[g_num_tests].setup_func = NULL;
            g_tests[g_num_tests].teardown_func = NULL;
            g_tests[g_num_tests].initial_state = p;

            g_num_tests++;
        }
    }
}

int main(void)
{
    build_test_array();
    return _cmocka_run_group_tests("zchkhs", g_tests, g_num_tests,
                                    group_setup, group_teardown);
}
