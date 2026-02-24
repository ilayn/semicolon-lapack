/**
 * @file test_dchkhs.c
 * @brief Hessenberg Schur test driver - port of LAPACK TESTING/EIG/dchkhs.f
 *
 * Tests the nonsymmetric eigenvalue problem routines.
 *
 *         DGEHRD factors A as  U H U' , where ' means transpose,
 *         H is hessenberg, and U is an orthogonal matrix.
 *
 *         DORGHR generates the orthogonal matrix U.
 *
 *         DORMHR multiplies a matrix by the orthogonal matrix U.
 *
 *         DHSEQR factors H as  Z T Z' , where Z is orthogonal and
 *         T is "quasi-triangular", and the eigenvalue vector W.
 *
 *         DTREVC computes the left and right eigenvector matrices
 *         L and R for T.
 *
 *         DHSEIN computes the left and right eigenvector matrices
 *         Y and X for H, using inverse iteration.
 *
 * Test ratios (16 total):
 *                    T
 *   (1)  | A - U H U  | / ( |A| n ulp )
 *                T
 *   (2)  | I - UU  | / ( n ulp )
 *                    T
 *   (3)  | H - Z T Z  | / ( |H| n ulp )
 *                T
 *   (4)  | I - ZZ  | / ( n ulp )
 *                      T
 *   (5)  | A - UZ T (UZ)  | / ( |A| n ulp )
 *                  T
 *   (6)  | I - (UZ)(UZ)  | / ( n ulp )
 *
 *   (7)  | T2 - T1 | / ( |T| n ulp )
 *
 *   (8)  | W2 - W1 | / ( max(|W1|,|W2|) ulp )
 *
 *   (9)  | TR - RW | / ( |T| |R| ulp )
 *
 *   (10) | L'T - W'L | / ( |T| |L| ulp )
 *
 *   (11) | HX - XW | / ( |H| |X| ulp )     (from inverse iteration)
 *
 *   (12) | Y'H - W'Y | / ( |H| |Y| ulp )   (from inverse iteration)
 *
 *   (13) | AX - XW | / ( |A| |X| ulp )      (DORMHR on inverse iteration)
 *
 *   (14) | Y'A - W'Y | / ( |A| |Y| ulp )    (DORMHR on inverse iteration)
 *
 *   (15) | AR - RW | / ( |A| |R| ulp )      (DTREVC3 backtransform)
 *
 *   (16) | L'A - W'L | / ( |A| |L| ulp )   (DTREVC3 backtransform)
 *
 * Matrix types: 21 types (MAXTYP = 21)
 */

#include "test_harness.h"
#include "verify.h"
#include "test_rng.h"
#include "semicolon_cblas.h"
#include <math.h>
#include <string.h>

#define THRESH 30.0
#define MAXTYP 21
#define NTEST  16

static const INT NVAL[] = {0, 1, 2, 3, 5, 10, 20, 50};
#define NNVAL (sizeof(NVAL) / sizeof(NVAL[0]))

/* External function declarations */
/* ===================================================================== */
/* DATA arrays from dchkhs.f lines 481-486                               */
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
} dchkhs_params_t;

typedef struct {
    INT nmax;
    f64* A;
    f64* H;
    f64* T1;
    f64* T2;
    f64* U;
    f64* Z;
    f64* UZ;
    f64* UU;
    f64* evectl;
    f64* evectr;
    f64* evecty;
    f64* evectx;
    f64* wr1;
    f64* wi1;
    f64* wr2;
    f64* wi2;
    f64* wr3;
    f64* wi3;
    f64* tau;
    f64* work;
    INT nwork;
    INT* iwork;
    INT* selectarr;
    uint64_t rng_state[4];
} dchkhs_workspace_t;

static dchkhs_workspace_t* g_ws = NULL;

/* ===================================================================== */
/* Group setup / teardown                                                */
/* ===================================================================== */

static int group_setup(void** state)
{
    (void)state;

    g_ws = calloc(1, sizeof(dchkhs_workspace_t));
    if (!g_ws) return -1;

    g_ws->nmax = 0;
    for (size_t i = 0; i < NNVAL; i++) {
        if (NVAL[i] > g_ws->nmax) g_ws->nmax = NVAL[i];
    }
    if (g_ws->nmax < 1) g_ws->nmax = 1;

    INT nmax = g_ws->nmax;
    INT n2 = nmax * nmax;

    g_ws->A      = malloc(n2 * sizeof(f64));
    g_ws->H      = malloc(n2 * sizeof(f64));
    g_ws->T1     = malloc(n2 * sizeof(f64));
    g_ws->T2     = malloc(n2 * sizeof(f64));
    g_ws->U      = malloc(n2 * sizeof(f64));
    g_ws->Z      = malloc(n2 * sizeof(f64));
    g_ws->UZ     = malloc(n2 * sizeof(f64));
    g_ws->UU     = malloc(n2 * sizeof(f64));
    g_ws->evectl = malloc(n2 * sizeof(f64));
    g_ws->evectr = malloc(n2 * sizeof(f64));
    g_ws->evecty = malloc(n2 * sizeof(f64));
    g_ws->evectx = malloc(n2 * sizeof(f64));

    g_ws->wr1 = malloc(nmax * sizeof(f64));
    g_ws->wi1 = malloc(nmax * sizeof(f64));
    g_ws->wr2 = malloc(nmax * sizeof(f64));
    g_ws->wi2 = malloc(nmax * sizeof(f64));
    g_ws->wr3 = malloc(nmax * sizeof(f64));
    g_ws->wi3 = malloc(nmax * sizeof(f64));
    g_ws->tau = malloc(nmax * sizeof(f64));

    g_ws->nwork = 4 * n2 + 2;
    if (g_ws->nwork < 1) g_ws->nwork = 1;
    g_ws->work = malloc(g_ws->nwork * sizeof(f64));

    g_ws->iwork     = malloc(2 * nmax * sizeof(INT));
    g_ws->selectarr = malloc(nmax * sizeof(INT));

    if (!g_ws->A || !g_ws->H || !g_ws->T1 || !g_ws->T2 ||
        !g_ws->U || !g_ws->Z || !g_ws->UZ || !g_ws->UU ||
        !g_ws->evectl || !g_ws->evectr || !g_ws->evecty || !g_ws->evectx ||
        !g_ws->wr1 || !g_ws->wi1 || !g_ws->wr2 || !g_ws->wi2 ||
        !g_ws->wr3 || !g_ws->wi3 || !g_ws->tau ||
        !g_ws->work || !g_ws->iwork || !g_ws->selectarr) {
        return -1;
    }

    rng_seed(g_ws->rng_state, 0xDCE4B501ULL);
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
        free(g_ws->wr1);
        free(g_ws->wi1);
        free(g_ws->wr2);
        free(g_ws->wi2);
        free(g_ws->wr3);
        free(g_ws->wi3);
        free(g_ws->tau);
        free(g_ws->work);
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

static void run_dchkhs_single(dchkhs_params_t* params)
{
    const INT n = NVAL[params->jsize];
    const INT jtype = params->jtype;
    const INT jt = jtype - 1;

    if (n == 0) return;

    const INT lda = g_ws->nmax;
    const INT ldu = g_ws->nmax;
    const INT nwork = g_ws->nwork;

    f64* A      = g_ws->A;
    f64* H      = g_ws->H;
    f64* T1     = g_ws->T1;
    f64* T2     = g_ws->T2;
    f64* U      = g_ws->U;
    f64* Z      = g_ws->Z;
    f64* UZ     = g_ws->UZ;
    f64* UU     = g_ws->UU;
    f64* evectl = g_ws->evectl;
    f64* evectr = g_ws->evectr;
    f64* evecty = g_ws->evecty;
    f64* evectx = g_ws->evectx;
    f64* wr1    = g_ws->wr1;
    f64* wi1    = g_ws->wi1;
    f64* wr2    = g_ws->wr2;
    f64* wi2    = g_ws->wi2;
    f64* wr3    = g_ws->wr3;
    f64* wi3    = g_ws->wi3;
    f64* tau    = g_ws->tau;
    f64* work   = g_ws->work;
    INT* iwork  = g_ws->iwork;
    INT* selectarr = g_ws->selectarr;

    uint64_t* rng = g_ws->rng_state;

    const f64 unfl = dlamch("S");
    const f64 ovfl = dlamch("O");
    const f64 ulp = dlamch("P");
    const f64 ulpinv = 1.0 / ulp;
    const f64 rtunfl = sqrt(unfl);
    const f64 rtovfl = sqrt(ovfl);
    const f64 rtulp = sqrt(ulp);
    const f64 rtulpi = 1.0 / rtulp;

    const INT n1 = (n > 1) ? n : 1;
    const f64 aninv = 1.0 / (f64)n1;

    f64 result[NTEST];
    for (INT i = 0; i < NTEST; i++)
        result[i] = 0.0;

    char ctx[256];
    INT iinfo;

    /* ================================================================ */
    /* Generate test matrix A                                           */
    /* ================================================================ */

    if (jtype <= MAXTYP) {
        INT itype = ktype[jt];
        INT imode = kmode[jt];

        f64 anorm;
        switch (kmagn[jt]) {
            case 2:  anorm = (rtovfl * ulp) * aninv; break;
            case 3:  anorm = rtunfl * n * ulpinv;    break;
            default: anorm = 1.0;                    break;
        }

        dlaset("Full", lda, n, 0.0, 0.0, A, lda);
        iinfo = 0;
        f64 cond = ulpinv;

        if (itype == 1) {
            /* Zero */
            iinfo = 0;

        } else if (itype == 2) {
            /* Identity */
            for (INT jcol = 0; jcol < n; jcol++)
                A[jcol + jcol * lda] = anorm;

        } else if (itype == 3) {
            /* Jordan Block */
            for (INT jcol = 0; jcol < n; jcol++) {
                A[jcol + jcol * lda] = anorm;
                if (jcol > 0)
                    A[jcol + (jcol - 1) * lda] = 1.0;
            }

        } else if (itype == 4) {
            /* Diagonal Matrix, eigenvalues specified */
            dlatms(n, n, "S", "S", work, imode, cond, anorm,
                   0, 0, "N", A, lda, work + n, &iinfo, rng);

        } else if (itype == 5) {
            /* Symmetric, eigenvalues specified */
            dlatms(n, n, "S", "S", work, imode, cond, anorm,
                   n, n, "N", A, lda, work + n, &iinfo, rng);

        } else if (itype == 6) {
            /* General, eigenvalues specified */
            f64 conds;
            if (kconds[jt] == 1)
                conds = 1.0;
            else if (kconds[jt] == 2)
                conds = rtulpi;
            else
                conds = 0.0;

            dlatme(n, "S", work, imode, cond, 1.0,
                   " ", "T", "T", "T", work + n, 4,
                   conds, n, n, anorm, A, lda, work + 2 * n,
                   &iinfo, rng);

        } else if (itype == 7) {
            /* Diagonal, random eigenvalues */
            INT idumma[1] = {0};
            dlatmr(n, n, "S", "S", work, 6, 1.0, 1.0,
                   "T", "N", work + n, 1, 1.0,
                   work + 2 * n, 1, 1.0, "N", idumma, 0, 0,
                   0.0, anorm, "NO", A, lda, iwork, &iinfo, rng);

        } else if (itype == 8) {
            /* Symmetric, random eigenvalues */
            INT idumma[1] = {0};
            dlatmr(n, n, "S", "S", work, 6, 1.0, 1.0,
                   "T", "N", work + n, 1, 1.0,
                   work + 2 * n, 1, 1.0, "N", idumma, n, n,
                   0.0, anorm, "NO", A, lda, iwork, &iinfo, rng);

        } else if (itype == 9) {
            /* General, random eigenvalues */
            INT idumma[1] = {0};
            dlatmr(n, n, "S", "N", work, 6, 1.0, 1.0,
                   "T", "N", work + n, 1, 1.0,
                   work + 2 * n, 1, 1.0, "N", idumma, n, n,
                   0.0, anorm, "NO", A, lda, iwork, &iinfo, rng);

        } else if (itype == 10) {
            /* Triangular, random eigenvalues */
            INT idumma[1] = {0};
            dlatmr(n, n, "S", "N", work, 6, 1.0, 1.0,
                   "T", "N", work + n, 1, 1.0,
                   work + 2 * n, 1, 1.0, "N", idumma, n, 0,
                   0.0, anorm, "NO", A, lda, iwork, &iinfo, rng);

        } else {
            iinfo = 1;
        }

        if (iinfo != 0) {
            snprintf(ctx, sizeof(ctx),
                     "dchkhs n=%d type=%d: Generator info=%d", n, jtype, iinfo);
            set_test_context(ctx);
            assert_info_success(iinfo);
            return;
        }
    }
    /* else: jtype > MAXTYP => skip matrix generation, use A as-is (all zeros from previous) */

    /* ================================================================ */
    /* Tests 1-2: DGEHRD + DORGHR                                       */
    /* ================================================================ */

    dlacpy(" ", n, n, A, lda, H, lda);

    dgehrd(n, 0, n - 1, H, lda, work, work + n, nwork - n, &iinfo);
    if (iinfo != 0) {
        result[0] = ulpinv;
        snprintf(ctx, sizeof(ctx),
                 "dchkhs n=%d type=%d: DGEHRD info=%d", n, jtype, iinfo);
        set_test_context(ctx);
        assert_info_success(iinfo);
        return;
    }

    /* Extract reflectors and zero sub-Hessenberg part */
    for (INT j = 0; j < n - 1; j++) {
        UU[j + 1 + j * ldu] = 0.0;
        for (INT i = j + 2; i < n; i++) {
            U[i + j * ldu] = H[i + j * lda];
            UU[i + j * ldu] = H[i + j * lda];
            H[i + j * lda] = 0.0;
        }
    }
    cblas_dcopy(n - 1, work, 1, tau, 1);

    dorghr(n, 0, n - 1, U, ldu, work, work + n, nwork - n, &iinfo);

    /* Tests 1-2 */
    dhst01(n, 0, n - 1, A, lda, H, lda, U, ldu, work, nwork, result);

    /* ================================================================ */
    /* Tests 3-8: DHSEQR eigenvalues and Schur form                     */
    /* ================================================================ */

    /* Eigenvalues only (WR3, WI3) */
    dlacpy(" ", n, n, H, lda, T2, lda);
    result[2] = ulpinv;

    dhseqr("E", "N", n, 0, n - 1, T2, lda, wr3, wi3,
           UZ, ldu, work, nwork, &iinfo);
    if (iinfo != 0) {
        if (iinfo <= n + 2) {
            snprintf(ctx, sizeof(ctx),
                     "dchkhs n=%d type=%d: DHSEQR(E) info=%d", n, jtype, iinfo);
            set_test_context(ctx);
            assert_info_success(iinfo);
            return;
        }
    }

    /* Eigenvalues (WR2, WI2) and Full Schur Form (T2) */
    dlacpy(" ", n, n, H, lda, T2, lda);

    dhseqr("S", "N", n, 0, n - 1, T2, lda, wr2, wi2,
           UZ, ldu, work, nwork, &iinfo);
    if (iinfo != 0 && iinfo <= n + 2) {
        snprintf(ctx, sizeof(ctx),
                 "dchkhs n=%d type=%d: DHSEQR(S) info=%d", n, jtype, iinfo);
        set_test_context(ctx);
        assert_info_success(iinfo);
        return;
    }

    /* Eigenvalues (WR1, WI1), Schur Form (T1), and Schur vectors (UZ) */
    dlacpy(" ", n, n, H, lda, T1, lda);
    dlacpy(" ", n, n, U, ldu, UZ, ldu);

    dhseqr("S", "V", n, 0, n - 1, T1, lda, wr1, wi1,
           UZ, ldu, work, nwork, &iinfo);
    if (iinfo != 0 && iinfo <= n + 2) {
        snprintf(ctx, sizeof(ctx),
                 "dchkhs n=%d type=%d: DHSEQR(V) info=%d", n, jtype, iinfo);
        set_test_context(ctx);
        assert_info_success(iinfo);
        return;
    }

    /* Compute Z = U' * UZ */
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                n, n, n, 1.0, U, ldu, UZ, ldu, 0.0, Z, ldu);

    /* Tests 3-4: | H - Z T Z' | / ( |H| n ulp ) and | I - Z Z' | / ( n ulp ) */
    dhst01(n, 0, n - 1, H, lda, T1, lda, Z, ldu, work, nwork, &result[2]);

    /* Tests 5-6: | A - UZ T (UZ)' | / ( |A| n ulp ) and | I - UZ (UZ)' | / ( n ulp ) */
    dhst01(n, 0, n - 1, A, lda, T1, lda, UZ, ldu, work, nwork, &result[4]);

    /* Test 7: | T2 - T1 | / ( |T| n ulp ) */
    dget10(n, n, T2, lda, T1, lda, work, &result[6]);

    /* Test 8: | W2 - W1 | / ( max(|W1|,|W2|) ulp ) */
    {
        f64 temp1 = 0.0;
        f64 temp2 = 0.0;
        for (INT j = 0; j < n; j++) {
            f64 w1mag = fabs(wr1[j]) + fabs(wi1[j]);
            f64 w2mag = fabs(wr2[j]) + fabs(wi2[j]);
            f64 wmax = (w1mag > w2mag) ? w1mag : w2mag;
            if (wmax > temp1) temp1 = wmax;

            f64 wdiff = fabs(wr1[j] - wr2[j]) + fabs(wi1[j] - wi2[j]);
            if (wdiff > temp2) temp2 = wdiff;
        }
        f64 denom = (temp1 > temp2) ? temp1 : temp2;
        denom = ulp * denom;
        if (denom < unfl) denom = unfl;
        result[7] = temp2 / denom;
    }

    /* ================================================================ */
    /* Tests 9-10: DTREVC eigenvectors of T                             */
    /* ================================================================ */

    result[8] = ulpinv;

    /* Select last max(N/4,1) real, max(N/4,1) complex eigenvectors */
    INT nselc = 0;
    INT nselr = 0;
    INT nsel_max = (n / 4 > 1) ? n / 4 : 1;
    {
        INT j = n - 1;
        while (j >= 0) {
            if (wi1[j] == 0.0) {
                if (nselr < nsel_max) {
                    nselr++;
                    selectarr[j] = 1;
                } else {
                    selectarr[j] = 0;
                }
                j--;
            } else {
                if (nselc < nsel_max) {
                    nselc++;
                    selectarr[j] = 1;
                    selectarr[j - 1] = 0;
                } else {
                    selectarr[j] = 0;
                    selectarr[j - 1] = 0;
                }
                j -= 2;
            }
        }
    }

    /* Right eigenvectors: "All" */
    f64 dumma[6];
    INT in;
    dtrevc("R", "A", selectarr, n, T1, lda, dumma, ldu,
           evectr, ldu, n, &in, work, &iinfo);
    if (iinfo != 0) {
        snprintf(ctx, sizeof(ctx),
                 "dchkhs n=%d type=%d: DTREVC(R,A) info=%d", n, jtype, iinfo);
        set_test_context(ctx);
        assert_info_success(iinfo);
        return;
    }

    /* Test 9: | TR - RW | / ( |T| |R| ulp ) */
    dget22("N", "N", "N", n, T1, lda, evectr, ldu, wr1, wi1, work, dumma);
    result[8] = dumma[0];

    /* Compute selected right eigenvectors and confirm they agree */
    dtrevc("R", "S", selectarr, n, T1, lda, dumma, ldu,
           evectl, ldu, n, &in, work, &iinfo);
    if (iinfo != 0) {
        snprintf(ctx, sizeof(ctx),
                 "dchkhs n=%d type=%d: DTREVC(R,S) info=%d", n, jtype, iinfo);
        set_test_context(ctx);
        assert_info_success(iinfo);
        return;
    }

    {
        INT k = 0;
        INT match = 1;
        for (INT j = 0; j < n && match; j++) {
            if (selectarr[j] && wi1[j] == 0.0) {
                for (INT jj = 0; jj < n; jj++) {
                    if (evectr[jj + j * ldu] != evectl[jj + k * ldu]) {
                        match = 0;
                        break;
                    }
                }
                k++;
            } else if (selectarr[j] && wi1[j] != 0.0) {
                for (INT jj = 0; jj < n; jj++) {
                    if (evectr[jj + j * ldu] != evectl[jj + k * ldu] ||
                        evectr[jj + (j + 1) * ldu] != evectl[jj + (k + 1) * ldu]) {
                        match = 0;
                        break;
                    }
                }
                k += 2;
            }
        }
        /* match == 0 would be logged in Fortran; here we just proceed */
    }

    /* Left eigenvectors: "All" */
    result[9] = ulpinv;
    dtrevc("L", "A", selectarr, n, T1, lda, evectl, ldu,
           dumma, ldu, n, &in, work, &iinfo);
    if (iinfo != 0) {
        snprintf(ctx, sizeof(ctx),
                 "dchkhs n=%d type=%d: DTREVC(L,A) info=%d", n, jtype, iinfo);
        set_test_context(ctx);
        assert_info_success(iinfo);
        return;
    }

    /* Test 10: | L'T - W'L | / ( |T| |L| ulp ) */
    dget22("T", "N", "C", n, T1, lda, evectl, ldu, wr1, wi1, work, &dumma[2]);
    result[9] = dumma[2];

    /* Compute selected left eigenvectors and confirm they agree */
    dtrevc("L", "S", selectarr, n, T1, lda, evectr, ldu,
           dumma, ldu, n, &in, work, &iinfo);
    if (iinfo != 0) {
        snprintf(ctx, sizeof(ctx),
                 "dchkhs n=%d type=%d: DTREVC(L,S) info=%d", n, jtype, iinfo);
        set_test_context(ctx);
        assert_info_success(iinfo);
        return;
    }

    {
        INT k = 0;
        INT match = 1;
        for (INT j = 0; j < n && match; j++) {
            if (selectarr[j] && wi1[j] == 0.0) {
                for (INT jj = 0; jj < n; jj++) {
                    if (evectl[jj + j * ldu] != evectr[jj + k * ldu]) {
                        match = 0;
                        break;
                    }
                }
                k++;
            } else if (selectarr[j] && wi1[j] != 0.0) {
                for (INT jj = 0; jj < n; jj++) {
                    if (evectl[jj + j * ldu] != evectr[jj + k * ldu] ||
                        evectl[jj + (j + 1) * ldu] != evectr[jj + (k + 1) * ldu]) {
                        match = 0;
                        break;
                    }
                }
                k += 2;
            }
        }
    }

    /* ================================================================ */
    /* Tests 11-12: DHSEIN eigenvectors of H via inverse iteration      */
    /* ================================================================ */

    result[10] = ulpinv;
    for (INT j = 0; j < n; j++)
        selectarr[j] = 1;

    dhsein("R", "Q", "N", selectarr, n, H, lda,
           wr3, wi3, dumma, ldu, evectx, ldu, n1, &in,
           work, iwork, iwork + n, &iinfo);
    if (iinfo != 0) {
        snprintf(ctx, sizeof(ctx),
                 "dchkhs n=%d type=%d: DHSEIN(R) info=%d", n, jtype, iinfo);
        set_test_context(ctx);
        if (iinfo < 0) {
            assert_info_success(iinfo);
            return;
        }
    } else {
        /* Test 11: | HX - XW | / ( |H| |X| ulp ) */
        dget22("N", "N", "N", n, H, lda, evectx, ldu, wr3, wi3,
               work, dumma);
        if (dumma[0] < ulpinv)
            result[10] = dumma[0] * aninv;
    }

    /* Left eigenvectors of H via inverse iteration */
    result[11] = ulpinv;
    for (INT j = 0; j < n; j++)
        selectarr[j] = 1;

    dhsein("L", "Q", "N", selectarr, n, H, lda,
           wr3, wi3, evecty, ldu, dumma, ldu, n1, &in,
           work, iwork, iwork + n, &iinfo);
    if (iinfo != 0) {
        snprintf(ctx, sizeof(ctx),
                 "dchkhs n=%d type=%d: DHSEIN(L) info=%d", n, jtype, iinfo);
        set_test_context(ctx);
        if (iinfo < 0) {
            assert_info_success(iinfo);
            return;
        }
    } else {
        /* Test 12: | Y'H - W'Y | / ( |H| |Y| ulp ) */
        dget22("C", "N", "C", n, H, lda, evecty, ldu, wr3, wi3,
               work, &dumma[2]);
        if (dumma[2] < ulpinv)
            result[11] = dumma[2] * aninv;
    }

    /* ================================================================ */
    /* Tests 13-14: DORMHR back-transform DHSEIN eigenvectors           */
    /* ================================================================ */

    result[12] = ulpinv;

    dormhr("L", "N", n, n, 0, n - 1, UU, ldu, tau, evectx, ldu,
           work, nwork, &iinfo);
    if (iinfo != 0) {
        snprintf(ctx, sizeof(ctx),
                 "dchkhs n=%d type=%d: DORMHR(R) info=%d", n, jtype, iinfo);
        set_test_context(ctx);
        if (iinfo < 0) {
            assert_info_success(iinfo);
            return;
        }
    } else {
        /* Test 13: | AX - XW | / ( |A| |X| ulp ) */
        dget22("N", "N", "N", n, A, lda, evectx, ldu, wr3, wi3,
               work, dumma);
        if (dumma[0] < ulpinv)
            result[12] = dumma[0] * aninv;
    }

    result[13] = ulpinv;

    dormhr("L", "N", n, n, 0, n - 1, UU, ldu, tau, evecty, ldu,
           work, nwork, &iinfo);
    if (iinfo != 0) {
        snprintf(ctx, sizeof(ctx),
                 "dchkhs n=%d type=%d: DORMHR(L) info=%d", n, jtype, iinfo);
        set_test_context(ctx);
        if (iinfo < 0) {
            assert_info_success(iinfo);
            return;
        }
    } else {
        /* Test 14: | Y'A - W'Y | / ( |A| |Y| ulp ) */
        dget22("C", "N", "C", n, A, lda, evecty, ldu, wr3, wi3,
               work, &dumma[2]);
        if (dumma[2] < ulpinv)
            result[13] = dumma[2] * aninv;
    }

    /* ================================================================ */
    /* Tests 15-16: DTREVC3 eigenvectors of A via back-transform        */
    /* ================================================================ */

    result[14] = ulpinv;

    dlacpy(" ", n, n, UZ, ldu, evectr, ldu);

    dtrevc3("R", "B", selectarr, n, T1, lda, dumma, ldu,
            evectr, ldu, n, &in, work, nwork, &iinfo);
    if (iinfo != 0) {
        snprintf(ctx, sizeof(ctx),
                 "dchkhs n=%d type=%d: DTREVC3(R,B) info=%d", n, jtype, iinfo);
        set_test_context(ctx);
        assert_info_success(iinfo);
        return;
    }

    /* Test 15: | AR - RW | / ( |A| |R| ulp ) */
    dget22("N", "N", "N", n, A, lda, evectr, ldu, wr1, wi1, work, dumma);
    result[14] = dumma[0];

    /* Left eigenvectors via back-transform */
    result[15] = ulpinv;

    dlacpy(" ", n, n, UZ, ldu, evectl, ldu);

    dtrevc3("L", "B", selectarr, n, T1, lda, evectl, ldu,
            dumma, ldu, n, &in, work, nwork, &iinfo);
    if (iinfo != 0) {
        snprintf(ctx, sizeof(ctx),
                 "dchkhs n=%d type=%d: DTREVC3(L,B) info=%d", n, jtype, iinfo);
        set_test_context(ctx);
        assert_info_success(iinfo);
        return;
    }

    /* Test 16: | L'A - W'L | / ( |A| |L| ulp ) */
    dget22("T", "N", "C", n, A, lda, evectl, ldu, wr1, wi1, work, &dumma[2]);
    result[15] = dumma[2];

    /* ================================================================ */
    /* Check results against threshold                                  */
    /* ================================================================ */

    for (INT jr = 0; jr < NTEST; jr++) {
        snprintf(ctx, sizeof(ctx), "dchkhs n=%d type=%d TEST %d",
                 n, jtype, jr + 1);
        set_test_context(ctx);
        assert_residual_ok(result[jr]);
    }
    clear_test_context();
}

/* ===================================================================== */
/* CMocka test wrapper                                                   */
/* ===================================================================== */

static void test_dchkhs_case(void** state)
{
    dchkhs_params_t* params = *state;
    run_dchkhs_single(params);
}

/* ===================================================================== */
/* Build test array and main                                             */
/* ===================================================================== */

#define MAX_TESTS (NNVAL * MAXTYP)

static dchkhs_params_t g_params[MAX_TESTS];
static struct CMUnitTest g_tests[MAX_TESTS];
static INT g_num_tests = 0;

static void build_test_array(void)
{
    g_num_tests = 0;

    for (size_t in = 0; in < NNVAL; in++) {
        INT n = NVAL[in];
        for (INT jtype = 1; jtype <= MAXTYP; jtype++) {
            dchkhs_params_t* p = &g_params[g_num_tests];
            p->jsize = (INT)in;
            p->jtype = jtype;
            snprintf(p->name, sizeof(p->name),
                     "dchkhs_n%d_type%d", n, jtype);

            g_tests[g_num_tests].name = p->name;
            g_tests[g_num_tests].test_func = test_dchkhs_case;
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
    return _cmocka_run_group_tests("dchkhs", g_tests, g_num_tests,
                                    group_setup, group_teardown);
}
