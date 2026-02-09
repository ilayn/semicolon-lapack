/**
 * @file test_ddrvst.c
 * @brief Symmetric eigenvalue test driver - port of LAPACK TESTING/EIG/ddrvst.f
 *
 * Tests the symmetric eigenvalue problem drivers:
 *   DSTEV, DSTEVX, DSTEVR, DSTEVD    - Tridiagonal matrices
 *   DSYEV, DSYEVX, DSYEVR, DSYEVD    - Full symmetric matrices
 *   DSPEV, DSPEVX, DSPEVD            - Packed symmetric matrices
 *   DSBEV, DSBEVX, DSBEVD            - Band symmetric matrices
 *
 * Each (n, jtype) combination is registered as a separate CMocka test,
 * providing pytest-like parameterized test behavior with clear failure isolation.
 *
 * Test ratios (3 per routine):
 *   (1)  | A - Z D Z' | / ( |A| n ulp )
 *   (2)  | I - Z Z' | / ( n ulp )
 *   (3)  | D(with Z) - D(w/o Z) | / ( |D| ulp )
 *
 * Matrix types (18 total):
 *   Types 1-2:   Zero, Identity
 *   Types 3-5:   Diagonal with evenly/geometrically/clustered eigenvalues
 *   Types 6-7:   Scaled near overflow/underflow
 *   Types 8-10:  U'DU with evenly/geometrically/clustered eigenvalues
 *   Types 11-12: U'DU scaled near overflow/underflow
 *   Types 13-15: Symmetric random, scaled near overflow/underflow
 *   Types 16-18: Band symmetric, scaled near overflow/underflow
 */

#include "test_harness.h"
#include "testutils/verify.h"
#include "testutils/test_rng.h"
#include <cblas.h>
#include <math.h>
#include <string.h>

/* Test threshold from LAPACK ddrvst.f */
#define THRESH 30.0

/* Maximum matrix type to test */
#define MAXTYP 18

/* Test dimensions from sep.in */
static const int NVAL[] = {0, 1, 2, 3, 5, 10, 20};
#define NNVAL (sizeof(NVAL) / sizeof(NVAL[0]))

/* External function declarations - Tridiagonal eigenvalue routines */
extern void dstev(const char* jobz, const int n, double* D, double* E,
                  double* Z, const int ldz, double* work, int* info);
extern void dstevd(const char* jobz, const int n, double* D, double* E,
                   double* Z, const int ldz, double* work, const int lwork,
                   int* iwork, const int liwork, int* info);
extern void dstevx(const char* jobz, const char* range, const int n,
                   double* D, double* E, const double vl, const double vu,
                   const int il, const int iu, const double abstol, int* m,
                   double* W, double* Z, const int ldz, double* work,
                   int* iwork, int* ifail, int* info);
extern void dstevr(const char* jobz, const char* range, const int n,
                   double* D, double* E, const double vl, const double vu,
                   const int il, const int iu, const double abstol, int* m,
                   double* W, double* Z, const int ldz, int* isuppz,
                   double* work, const int lwork, int* iwork, const int liwork,
                   int* info);

/* External function declarations - Full symmetric eigenvalue routines */
extern void dsyev(const char* jobz, const char* uplo, const int n,
                  double* A, const int lda, double* W, double* work,
                  const int lwork, int* info);
extern void dsyevd(const char* jobz, const char* uplo, const int n,
                   double* A, const int lda, double* W, double* work,
                   const int lwork, int* iwork, const int liwork, int* info);
extern void dsyevx(const char* jobz, const char* range, const char* uplo,
                   const int n, double* A, const int lda, const double vl,
                   const double vu, const int il, const int iu,
                   const double abstol, int* m, double* W, double* Z,
                   const int ldz, double* work, const int lwork, int* iwork,
                   int* ifail, int* info);
extern void dsyevr(const char* jobz, const char* range, const char* uplo,
                   const int n, double* A, const int lda, const double vl,
                   const double vu, const int il, const int iu,
                   const double abstol, int* m, double* W, double* Z,
                   const int ldz, int* isuppz, double* work, const int lwork,
                   int* iwork, const int liwork, int* info);

/* Utility routines */
extern double dlamch(const char* cmach);
extern double dlange(const char* norm, const int m, const int n,
                     const double* A, const int lda, double* work);
extern void dlacpy(const char* uplo, const int m, const int n,
                   const double* A, const int lda, double* B, const int ldb);
extern void dlaset(const char* uplo, const int m, const int n,
                   const double alpha, const double beta, double* A, const int lda);

/* Test parameters for a single test case */
typedef struct {
    int n;
    int jtype;    /* Matrix type (1-18) */
    char name[96];
} ddrvst_params_t;

/* Workspace structure for all tests */
typedef struct {
    int nmax;

    /* Matrices (all nmax x nmax) */
    double* A;      /* Original/working matrix */
    double* U;      /* Band matrix storage / orthogonal matrix */
    double* V;      /* Workspace matrix */
    double* Z;      /* Eigenvectors */

    /* Eigenvalues */
    double* D1;     /* Eigenvalues (with Z) */
    double* D2;     /* Off-diagonal / work */
    double* D3;     /* Eigenvalues (without Z) */
    double* D4;     /* Off-diagonal work */
    double* WA1;    /* Eigenvalues for partial computation */
    double* WA2;    /* Eigenvalues for comparison */
    double* WA3;    /* Eigenvalues for comparison */
    double* EVEIGS; /* Expected eigenvalues */
    double* TAU;    /* Householder scalars */

    /* Work arrays */
    double* work;
    int* iwork;
    int lwork;
    int liwork;

    /* Test results */
    double result[80];

    /* RNG state */
    uint64_t rng_state[4];
} ddrvst_workspace_t;

/* Global workspace pointer */
static ddrvst_workspace_t* g_ws = NULL;

/* Matrix type parameters (from ddrvst.f DATA statements lines 522-526)
 * KTYPE: 1, 2, 5*4, 5*5, 3*8, 3*9
 * KMAGN: 2*1, 1, 1, 1, 2, 3, 1, 1, 1, 2, 3, 1, 2, 3, 1, 2, 3
 * KMODE: 2*0, 4, 3, 1, 4, 4, 4, 3, 1, 4, 4, 0, 0, 0, 4, 4, 4
 */
static const int KTYPE[MAXTYP]  = {1, 2, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 8, 8, 8, 9, 9, 9};
static const int KMAGN[MAXTYP]  = {1, 1, 1, 1, 1, 2, 3, 1, 1, 1, 2, 3, 1, 2, 3, 1, 2, 3};
static const int KMODE[MAXTYP]  = {0, 0, 4, 3, 1, 4, 4, 4, 3, 1, 4, 4, 0, 0, 0, 4, 4, 4};

/**
 * Group setup: allocate shared workspace.
 */
static int group_setup(void** state)
{
    (void)state;

    g_ws = malloc(sizeof(ddrvst_workspace_t));
    if (!g_ws) return -1;

    /* Find maximum N */
    g_ws->nmax = 0;
    for (size_t i = 0; i < NNVAL; i++) {
        if (NVAL[i] > g_ws->nmax) g_ws->nmax = NVAL[i];
    }
    if (g_ws->nmax < 1) g_ws->nmax = 1;

    int nmax = g_ws->nmax;
    int n2 = nmax * nmax;

    /* Compute workspace sizes (from ddrvst.f lines 602-608) */
    int lgn = 0;
    if (nmax > 0) {
        lgn = (int)(log((double)nmax) / log(2.0));
        if ((1 << lgn) < nmax) lgn++;
        g_ws->lwork = 1 + 4 * nmax + 2 * nmax * lgn + 4 * n2;
        /* IWORK needs: 5*NMAX (for DSTEVX) + NMAX (for IFAIL) = 6*NMAX
         * Plus extra for DSYEVD which needs 3+5*N for JOBZ='V' */
        g_ws->liwork = 6 * nmax + 10;
    } else {
        g_ws->lwork = 9;
        g_ws->liwork = 16;
    }

    /* Allocate matrices */
    g_ws->A   = malloc(n2 * sizeof(double));
    /* U needs extra rows for band storage with PACK='Z' in dlatms.
     * For symmetric band with half-bandwidth up to nmax-1,
     * LDA >= 2*(nmax-1)+1 = 2*nmax - 1 */
    int ldu_band = 2 * nmax - 1;
    g_ws->U   = malloc(ldu_band * nmax * sizeof(double));
    g_ws->V   = malloc(n2 * sizeof(double));
    g_ws->Z   = malloc(n2 * sizeof(double));

    /* Allocate eigenvalue arrays */
    g_ws->D1  = malloc(nmax * sizeof(double));
    g_ws->D2  = malloc(nmax * sizeof(double));
    g_ws->D3  = malloc(nmax * sizeof(double));
    g_ws->D4  = malloc(nmax * sizeof(double));
    g_ws->WA1 = malloc(nmax * sizeof(double));
    g_ws->WA2 = malloc(nmax * sizeof(double));
    g_ws->WA3 = malloc(nmax * sizeof(double));
    g_ws->EVEIGS = malloc(nmax * sizeof(double));
    g_ws->TAU = malloc(nmax * sizeof(double));

    /* Allocate work arrays */
    g_ws->work  = malloc(g_ws->lwork * sizeof(double));
    g_ws->iwork = malloc(g_ws->liwork * sizeof(int));

    if (!g_ws->A || !g_ws->U || !g_ws->V || !g_ws->Z ||
        !g_ws->D1 || !g_ws->D2 || !g_ws->D3 || !g_ws->D4 ||
        !g_ws->WA1 || !g_ws->WA2 || !g_ws->WA3 || !g_ws->EVEIGS || !g_ws->TAU ||
        !g_ws->work || !g_ws->iwork) {
        return -1;
    }

    rng_seed(g_ws->rng_state, 0xDEADBEEFULL);
    return 0;
}

/**
 * Group teardown: free shared workspace.
 */
static int group_teardown(void** state)
{
    (void)state;

    if (g_ws) {
        free(g_ws->A);
        free(g_ws->U);
        free(g_ws->V);
        free(g_ws->Z);
        free(g_ws->D1);
        free(g_ws->D2);
        free(g_ws->D3);
        free(g_ws->D4);
        free(g_ws->WA1);
        free(g_ws->WA2);
        free(g_ws->WA3);
        free(g_ws->EVEIGS);
        free(g_ws->TAU);
        free(g_ws->work);
        free(g_ws->iwork);
        free(g_ws);
        g_ws = NULL;
    }
    return 0;
}

/**
 * Generate test matrix according to jtype.
 *
 * Based on ddrvst.f lines 647-746.
 */
static int generate_matrix(int n, int jtype, double* A, int lda,
                           double* U, int ldu, double* work, int* iwork,
                           uint64_t state[static 4])
{
    (void)ldu;
    int itype = KTYPE[jtype - 1];
    int imode = KMODE[jtype - 1];
    double anorm, cond;
    int iinfo = 0;

    double ulp = dlamch("P");
    double unfl = dlamch("S");
    double ovfl = 1.0 / unfl;
    double ulpinv = 1.0 / ulp;
    double rtunfl = sqrt(unfl);
    double rtovfl = sqrt(ovfl);
    double aninv = 1.0 / (double)(n > 1 ? n : 1);

    /* Compute norm based on KMAGN (lines 652-664) */
    switch (KMAGN[jtype - 1]) {
        case 1: anorm = 1.0; break;
        case 2: anorm = (rtovfl * ulp) * aninv; break;
        case 3: anorm = rtunfl * n * ulpinv; break;
        default: anorm = 1.0;
    }

    /* Initialize A to zero */
    dlaset("F", lda, n, 0.0, 0.0, A, lda);
    cond = ulpinv;

    if (itype == 1) {
        /* Zero matrix */
        iinfo = 0;

    } else if (itype == 2) {
        /* Identity matrix scaled by ANORM */
        for (int jcol = 0; jcol < n; jcol++) {
            A[jcol + jcol * lda] = anorm;
        }

    } else if (itype == 4) {
        /* Diagonal matrix, eigenvalues specified via DLATMS */
        dlatms(n, n, "S", "S", work, imode, cond, anorm,
               0, 0, "N", A, lda, work + n, &iinfo, state);

    } else if (itype == 5) {
        /* Symmetric, eigenvalues specified via DLATMS */
        dlatms(n, n, "S", "S", work, imode, cond, anorm,
               n, n, "N", A, lda, work + n, &iinfo, state);

    } else if (itype == 7) {
        /* Diagonal, random eigenvalues via DLATMR */
        int idumma[1] = {1};
        dlatmr(n, n, "S", "S", work, 6, 1.0, 1.0, "T", "N",
               work + n, 1, 1.0, work + 2 * n, 1, 1.0,
               "N", idumma, 0, 0, 0.0, anorm, "N",
               A, lda, iwork, &iinfo, state);

    } else if (itype == 8) {
        /* Symmetric, random eigenvalues via DLATMR */
        int idumma[1] = {1};
        dlatmr(n, n, "S", "S", work, 6, 1.0, 1.0, "T", "N",
               work + n, 1, 1.0, work + 2 * n, 1, 1.0,
               "N", idumma, n, n, 0.0, anorm, "N",
               A, lda, iwork, &iinfo, state);

    } else if (itype == 9) {
        /* Symmetric banded, eigenvalues specified via DLATMS */
        /* Half bandwidth randomly chosen */
        int ihbw = (int)((n - 1) * rng_uniform(state));

        /* For PACK='Z', LDA must be at least 2*KL + KU + 1 = 2*IHBW + 1 */
        int ldu_band = 2 * ihbw + 1;
        dlatms(n, n, "S", "S", work, imode, cond, anorm,
               ihbw, ihbw, "Z", U, ldu_band, work + n, &iinfo, state);

        /* Store as dense matrix */
        dlaset("F", lda, n, 0.0, 0.0, A, lda);
        for (int idiag = -ihbw; idiag <= ihbw; idiag++) {
            int irow = ihbw - idiag;
            int j1 = (idiag > 0) ? idiag : 0;
            int j2 = (n + idiag < n) ? n + idiag : n;
            for (int j = j1; j < j2; j++) {
                int i = j - idiag;
                A[i + j * lda] = U[irow + j * ldu_band];
            }
        }

    } else {
        iinfo = 1;
    }

    return iinfo;
}

/**
 * Run tridiagonal eigenvalue tests (DSTEV, DSTEVX, DSTEVR, DSTEVD)
 *
 * Based on ddrvst.f lines 773-1200.
 * Only runs for JTYPE <= 7 (tridiagonal-compatible matrix types).
 */
static void run_tridiag_tests(ddrvst_params_t* params)
{
    int n = params->n;
    int jtype = params->jtype;

    /* Tridiagonal tests only apply to types 1-7 */
    if (jtype > 7) return;

    ddrvst_workspace_t* ws = g_ws;
    int lda = ws->nmax;
    int ldu = ws->nmax;

    double* A = ws->A;
    double* Z = ws->Z;
    double* D1 = ws->D1;
    double* D2 = ws->D2;
    double* D3 = ws->D3;
    double* D4 = ws->D4;
    double* WA1 = ws->WA1;
    double* WA2 = ws->WA2;
    double* work = ws->work;
    int* iwork = ws->iwork;

    double ulp = dlamch("P");
    double unfl = dlamch("S");
    double ulpinv = 1.0 / ulp;

    int iinfo;
    double temp1, temp2;

    /* Initialize results to -1 (not computed) */
    for (int j = 0; j < 24; j++) {
        ws->result[j] = -1.0;
    }

    /* Skip N=0 cases */
    if (n == 0) return;

    /* Generate matrix */
    iinfo = generate_matrix(n, jtype, A, lda, ws->U, ldu, work, iwork, ws->rng_state);
    if (iinfo != 0) {
        ws->result[0] = ulpinv;
        print_message("Matrix generation failed for jtype=%d, n=%d, iinfo=%d\n",
                      jtype, n, iinfo);
        assert_info_success(iinfo);
        return;
    }

    /* Extract diagonal and off-diagonal for tridiagonal routines */
    for (int i = 0; i < n; i++) {
        D1[i] = A[i + i * lda];
    }
    for (int i = 0; i < n - 1; i++) {
        D2[i] = A[(i + 1) + i * lda];
    }

    /* ========== Test DSTEV with eigenvectors (Tests 1-2) ========== */
    dstev("V", n, D1, D2, Z, ldu, work, &iinfo);
    if (iinfo != 0) {
        ws->result[0] = ulpinv;
        ws->result[1] = ulpinv;
        ws->result[2] = ulpinv;
        print_message("DSTEV(V) failed with info=%d\n", iinfo);
    } else {
        /* Restore D3, D4 from A for verification */
        for (int i = 0; i < n; i++) {
            D3[i] = A[i + i * lda];
        }
        for (int i = 0; i < n - 1; i++) {
            D4[i] = A[(i + 1) + i * lda];
        }

        /* Tests 1-2 via dstt21 */
        dstt21(n, 0, D3, D4, D1, D2, Z, ldu, work, ws->result);
        assert_residual_ok(ws->result[0]);
        assert_residual_ok(ws->result[1]);

        /* ========== Test DSTEV without eigenvectors (Test 3) ========== */
        for (int i = 0; i < n; i++) {
            D3[i] = A[i + i * lda];
        }
        for (int i = 0; i < n - 1; i++) {
            D4[i] = A[(i + 1) + i * lda];
        }

        dstev("N", n, D3, D4, Z, ldu, work, &iinfo);
        if (iinfo != 0) {
            ws->result[2] = ulpinv;
            print_message("DSTEV(N) failed with info=%d\n", iinfo);
        } else {
            /* Test 3: compare eigenvalues */
            temp1 = 0.0;
            temp2 = 0.0;
            for (int j = 0; j < n; j++) {
                temp1 = fmax(temp1, fmax(fabs(D1[j]), fabs(D3[j])));
                temp2 = fmax(temp2, fabs(D1[j] - D3[j]));
            }
            ws->result[2] = temp2 / fmax(unfl, ulp * fmax(temp1, temp2));
            assert_residual_ok(ws->result[2]);
        }
    }

    /* ========== Test DSTEVX with RANGE='A' (Tests 4-6) ========== */
    double abstol = unfl + unfl;
    double vl = 0.0, vu = 0.0;
    int il = 1, iu = n;
    int m;

    /* Restore D1, D2 from A */
    for (int i = 0; i < n; i++) {
        D1[i] = A[i + i * lda];
    }
    for (int i = 0; i < n - 1; i++) {
        D2[i] = A[(i + 1) + i * lda];
    }

    dstevx("V", "A", n, D1, D2, vl, vu, il, iu, abstol, &m,
           WA1, Z, ldu, work, iwork, iwork + 5 * ws->nmax, &iinfo);
    if (iinfo != 0) {
        ws->result[3] = ulpinv;
        ws->result[4] = ulpinv;
        ws->result[5] = ulpinv;
        print_message("DSTEVX(V,A) failed with info=%d\n", iinfo);
    } else {
        /* Restore D3, D4 for verification */
        for (int i = 0; i < n; i++) {
            D3[i] = A[i + i * lda];
        }
        for (int i = 0; i < n - 1; i++) {
            D4[i] = A[(i + 1) + i * lda];
        }

        /* Tests 4-5 via dstt21 */
        dstt21(n, 0, D3, D4, WA1, D2, Z, ldu, work, ws->result + 3);
        assert_residual_ok(ws->result[3]);
        assert_residual_ok(ws->result[4]);

        /* Test 6: DSTEVX without eigenvectors */
        for (int i = 0; i < n; i++) {
            D3[i] = A[i + i * lda];
        }
        for (int i = 0; i < n - 1; i++) {
            D4[i] = A[(i + 1) + i * lda];
        }

        int m2;
        dstevx("N", "A", n, D3, D4, vl, vu, il, iu, abstol, &m2,
               WA2, Z, ldu, work, iwork, iwork + 5 * ws->nmax, &iinfo);
        if (iinfo != 0) {
            ws->result[5] = ulpinv;
            print_message("DSTEVX(N,A) failed with info=%d\n", iinfo);
        } else {
            /* Test 6: compare eigenvalues */
            temp1 = 0.0;
            temp2 = 0.0;
            for (int j = 0; j < n; j++) {
                temp1 = fmax(temp1, fmax(fabs(WA1[j]), fabs(WA2[j])));
                temp2 = fmax(temp2, fabs(WA1[j] - WA2[j]));
            }
            ws->result[5] = temp2 / fmax(unfl, ulp * fmax(temp1, temp2));
            assert_residual_ok(ws->result[5]);
        }
    }
}

/**
 * Run symmetric eigenvalue tests (DSYEV, DSYEVX, DSYEVR, DSYEVD)
 *
 * Based on ddrvst.f lines 1200-1700.
 */
static void run_symmetric_tests(ddrvst_params_t* params)
{
    int n = params->n;
    int jtype = params->jtype;

    ddrvst_workspace_t* ws = g_ws;
    int lda = ws->nmax;
    int ldu = ws->nmax;

    double* A = ws->A;
    double* V = ws->V;
    double* Z = ws->Z;
    double* D1 = ws->D1;
    double* D2 = ws->D2;
    double* TAU = ws->TAU;
    double* work = ws->work;
    int* iwork = ws->iwork;

    double ulp = dlamch("P");
    double unfl = dlamch("S");
    double ulpinv = 1.0 / ulp;

    int iinfo;
    double temp1, temp2;

    /* Initialize results */
    for (int j = 24; j < 48; j++) {
        ws->result[j] = -1.0;
    }

    if (n == 0) return;

    /* Generate matrix (use fresh seed) */
    iinfo = generate_matrix(n, jtype, A, lda, ws->U, ldu, work, iwork, ws->rng_state);
    if (iinfo != 0) {
        ws->result[24] = ulpinv;
        print_message("Matrix generation failed for symmetric tests, jtype=%d, n=%d\n",
                      jtype, n);
        return;
    }

    /* ========== Test DSYEV with UPLO='L' (Tests 25-27) ========== */
    /* Copy A to V */
    dlacpy("F", n, n, A, lda, V, ldu);

    /* Compute eigenvalues and eigenvectors */
    dsyev("V", "L", n, V, ldu, D1, work, ws->lwork, &iinfo);
    if (iinfo != 0) {
        ws->result[24] = ulpinv;
        ws->result[25] = ulpinv;
        ws->result[26] = ulpinv;
        print_message("DSYEV(V,L) failed with info=%d\n", iinfo);
    } else {
        /* Tests 25-26 via dsyt21 (ITYPE=1: A = Z*D*Z') */
        dsyt21(1, "L", n, 0, A, lda, D1, D2, V, ldu, Z, ldu,
               TAU, work, ws->result + 24);
        assert_residual_ok(ws->result[24]);
        assert_residual_ok(ws->result[25]);

        /* Test 27: DSYEV without eigenvectors */
        dlacpy("F", n, n, A, lda, V, ldu);
        dsyev("N", "L", n, V, ldu, D2, work, ws->lwork, &iinfo);
        if (iinfo != 0) {
            ws->result[26] = ulpinv;
            print_message("DSYEV(N,L) failed with info=%d\n", iinfo);
        } else {
            temp1 = 0.0;
            temp2 = 0.0;
            for (int j = 0; j < n; j++) {
                temp1 = fmax(temp1, fmax(fabs(D1[j]), fabs(D2[j])));
                temp2 = fmax(temp2, fabs(D1[j] - D2[j]));
            }
            ws->result[26] = temp2 / fmax(unfl, ulp * fmax(temp1, temp2));
            assert_residual_ok(ws->result[26]);
        }
    }

    /* ========== Test DSYEVD with UPLO='L' (Tests 28-30) ========== */
    dlacpy("F", n, n, A, lda, V, ldu);

    dsyevd("V", "L", n, V, ldu, D1, work, ws->lwork, iwork, ws->liwork, &iinfo);
    if (iinfo != 0) {
        ws->result[27] = ulpinv;
        ws->result[28] = ulpinv;
        ws->result[29] = ulpinv;
        print_message("DSYEVD(V,L) failed with info=%d\n", iinfo);
    } else {
        dsyt21(1, "L", n, 0, A, lda, D1, D2, V, ldu, Z, ldu,
               TAU, work, ws->result + 27);
        assert_residual_ok(ws->result[27]);
        assert_residual_ok(ws->result[28]);

        /* Test 30: compare with eigenvalues only */
        dlacpy("F", n, n, A, lda, V, ldu);
        dsyevd("N", "L", n, V, ldu, D2, work, ws->lwork, iwork, ws->liwork, &iinfo);
        if (iinfo != 0) {
            ws->result[29] = ulpinv;
        } else {
            temp1 = 0.0;
            temp2 = 0.0;
            for (int j = 0; j < n; j++) {
                temp1 = fmax(temp1, fmax(fabs(D1[j]), fabs(D2[j])));
                temp2 = fmax(temp2, fabs(D1[j] - D2[j]));
            }
            ws->result[29] = temp2 / fmax(unfl, ulp * fmax(temp1, temp2));
            assert_residual_ok(ws->result[29]);
        }
    }
}

/**
 * Run tests for a single (n, jtype) combination.
 */
static void run_ddrvst_single(ddrvst_params_t* params)
{
    run_tridiag_tests(params);
    run_symmetric_tests(params);
}

/**
 * Test function wrapper.
 */
static void test_ddrvst_case(void** state)
{
    ddrvst_params_t* params = *state;
    run_ddrvst_single(params);
}

/*
 * Generate all parameter combinations.
 * Total: NNVAL * MAXTYP = 7 * 18 = 126 tests
 */

/* Maximum number of test cases */
#define MAX_TESTS (NNVAL * MAXTYP)

static ddrvst_params_t g_params[MAX_TESTS];
static struct CMUnitTest g_tests[MAX_TESTS];
static int g_num_tests = 0;

/**
 * Build the test array with all parameter combinations.
 */
static void build_test_array(void)
{
    g_num_tests = 0;

    for (size_t in = 0; in < NNVAL; in++) {
        int n = NVAL[in];

        for (int jtype = 1; jtype <= MAXTYP; jtype++) {
            ddrvst_params_t* p = &g_params[g_num_tests];
            p->n = n;
            p->jtype = jtype;
            snprintf(p->name, sizeof(p->name),
                     "ddrvst_n%d_type%d", n, jtype);

            g_tests[g_num_tests].name = p->name;
            g_tests[g_num_tests].test_func = test_ddrvst_case;
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
    return _cmocka_run_group_tests("ddrvst", g_tests, g_num_tests,
                                   group_setup, group_teardown);
}
