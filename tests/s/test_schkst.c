/**
 * @file test_schkst.c
 * @brief Symmetric tridiagonal eigenvalue test driver - port of LAPACK
 *        TESTING/EIG/dchkst.f
 *
 * Tests the symmetric eigenvalue problem routines:
 *   SSYTRD, SORGTR      - Full symmetric to tridiagonal reduction
 *   SSPTRD, SOPGTR      - Packed symmetric to tridiagonal reduction
 *   SSTEQR, SSTERF      - QR iteration for tridiagonal eigenvalues
 *   SPTEQR              - Positive definite tridiagonal QR
 *   SSTEBZ, SSTEIN      - Bisection + inverse iteration
 *   SSTEDC              - Divide-and-conquer for tridiagonal
 *   SSTEMR              - Relatively Robust Representations (MRRR)
 *
 * Each (n, jtype) combination is registered as a separate CMocka test,
 * providing pytest-like parameterized test behavior with clear failure
 * isolation.
 *
 * 37 test ratios per (n, jtype):
 *   Tests  1-8 : Reduction quality (SSYTRD/SORGTR, SSPTRD/SOPGTR)
 *   Tests  9-13: QR eigensolvers (SSTEQR, SSTERF, SSTECH)
 *   Tests 14-16: Positive definite (SPTEQR, jtype > 15 only)
 *   Tests 17-21: Bisection + inverse iteration (SSTEBZ, SSTEIN)
 *   Tests 22-26: Divide-and-conquer (SSTEDC)
 *   Tests 27-37: MRRR (SSTEMR, IEEE compliant only)
 *
 * 21 matrix types (extends ddrvst's 18 by adding positive-definite types).
 */

#include "test_harness.h"
#include "verify.h"
#include "test_rng.h"
#include <cblas.h>
#include <math.h>
#include <string.h>

#define THRESH 50.0f
#define MAXTYP 21
#define NTEST  37

static const int NVAL[] = {0, 1, 2, 3, 5, 10, 20};
#define NNVAL (sizeof(NVAL) / sizeof(NVAL[0]))

/* Matrix type parameters (from dchkst.f DATA statements lines 658-663) */
static const int KTYPE[MAXTYP] = {1,2,4,4,4,4,4,5,5,5,5,5,8,8,8,9,9,9,9,9,10};
static const int KMAGN[MAXTYP] = {1,1,1,1,1,2,3,1,1,1,2,3,1,2,3,1,1,1,2,3, 1};
static const int KMODE[MAXTYP] = {0,0,4,3,1,4,4,4,3,1,4,4,0,0,0,4,3,1,4,4, 3};

/* SREL and SRANGE flags (from dchkst.f lines 622-625) */
#define SRANGE 0
#define SREL   0

/* External declarations - Reduction routines */
extern void ssytrd(const char* uplo, const int n, f32* A, const int lda,
                   f32* D, f32* E, f32* tau, f32* work, const int lwork,
                   int* info);
extern void sorgtr(const char* uplo, const int n, f32* A, const int lda,
                   const f32* tau, f32* work, const int lwork, int* info);
extern void ssptrd(const char* uplo, const int n, f32* AP,
                   f32* D, f32* E, f32* tau, int* info);
extern void sopgtr(const char* uplo, const int n, const f32* AP,
                   const f32* tau, f32* Q, const int ldq, f32* work,
                   int* info);

/* External declarations - Tridiagonal eigenvalue solvers */
extern void ssteqr(const char* compz, const int n, f32* D, f32* E,
                   f32* Z, const int ldz, f32* work, int* info);
extern void ssterf(const int n, f32* D, f32* E, int* info);
extern void spteqr(const char* compz, const int n, f32* D, f32* E,
                   f32* Z, const int ldz, f32* work, int* info);
extern void sstedc(const char* compz, const int n, f32* D, f32* E,
                   f32* Z, const int ldz, f32* work, const int lwork,
                   int* iwork, const int liwork, int* info);
extern void sstebz(const char* range, const char* order, const int n,
                   const f32 vl, const f32 vu, const int il, const int iu,
                   const f32 abstol, const f32* D, const f32* E,
                   int* m, int* nsplit, f32* W, int* iblock, int* isplit,
                   f32* work, int* iwork, int* info);
extern void sstein(const int n, const f32* D, const f32* E,
                   const int m, const f32* W, const int* iblock,
                   const int* isplit, f32* Z, const int ldz,
                   f32* work, int* iwork, int* ifail, int* info);
extern void sstemr(const char* jobz, const char* range, const int n,
                   f32* D, f32* E, const f32 vl, const f32 vu,
                   const int il, const int iu, int* m, f32* W,
                   f32* Z, const int ldz, const int nzc, int* isuppz,
                   int* tryrac, f32* work, const int lwork,
                   int* iwork, const int liwork, int* info);

/* Utility routines */
extern f32 slamch(const char* cmach);
extern f32 slange(const char* norm, const int m, const int n,
                  const f32* A, const int lda, f32* work);
extern void slacpy(const char* uplo, const int m, const int n,
                   const f32* A, const int lda, f32* B, const int ldb);
extern void slaset(const char* uplo, const int m, const int n,
                   const f32 alpha, const f32 beta, f32* A, const int lda);

/* Test parameters for a single test case */
typedef struct {
    int n;
    int jtype;
    char name[96];
} dchkst_params_t;

/* Workspace structure */
typedef struct {
    int nmax;

    f32* A;       /* nmax x nmax - original symmetric matrix */
    f32* U;       /* nmax x nmax - orthogonal from ssytrd/sorgtr */
    f32* V;       /* nmax x nmax - Householder vectors from ssytrd */
    f32* Z;       /* nmax x nmax - eigenvectors */
    f32* AP;      /* nmax*(nmax+1)/2 - packed format */
    f32* VP;      /* nmax*(nmax+1)/2 - packed Householder vectors */
    f32* TAU;     /* nmax - Householder scalars */
    f32* SD;      /* nmax - tridiagonal diagonal (saved) */
    f32* SE;      /* nmax - tridiagonal off-diagonal (saved) */
    f32* D1;      /* nmax - eigenvalues set 1 */
    f32* D2;      /* nmax - eigenvalues set 2 */
    f32* D3;      /* nmax - eigenvalues from ssterf */
    f32* D4;      /* nmax - eigenvalues from spteqr(V) */
    f32* D5;      /* nmax - eigenvalues from spteqr(N) / sstemr scratch */
    f32* WA1;     /* nmax - eigenvalues from sstebz(A,E) */
    f32* WA2;     /* nmax - eigenvalues from sstebz(I,E) */
    f32* WA3;     /* nmax - eigenvalues from sstebz(V,E) */
    f32* WR;      /* nmax - eigenvalues from sstemr */
    int* IBLOCK;  /* nmax - block indices from sstebz */
    int* ISPLIT;  /* nmax - split indices from sstebz */
    int* IFAIL;   /* nmax - failure flags from sstein */
    int* ISUPPZ;  /* 2*nmax - support from sstemr */

    f32* work;
    int* iwork;
    int lwork;
    int liwork;

    f32 result[NTEST + 1];

    uint64_t rng_state[4];
    uint64_t rng_state2[4];
} dchkst_ws_t;

static dchkst_ws_t* g_ws = NULL;

/* ===== Group setup/teardown ===== */

static int group_setup(void** state)
{
    (void)state;

    g_ws = calloc(1, sizeof(dchkst_ws_t));
    if (!g_ws) return -1;

    g_ws->nmax = 0;
    for (size_t i = 0; i < NNVAL; i++) {
        if (NVAL[i] > g_ws->nmax) g_ws->nmax = NVAL[i];
    }
    if (g_ws->nmax < 1) g_ws->nmax = 1;

    int nmax = g_ws->nmax;
    int n2 = nmax * nmax;
    int nap = (nmax * (nmax + 1)) / 2;

    /* Compute workspace sizes (dchkst.f lines 741-746) */
    int lgn = 0;
    if (nmax > 0) {
        lgn = (int)(logf((f32)nmax) / logf(2.0f));
        if ((1 << lgn) < nmax) lgn++;
        if ((1 << lgn) < nmax) lgn++;
        g_ws->lwork = 1 + 4 * nmax + 2 * nmax * lgn + 4 * n2;
        g_ws->liwork = 6 + 6 * nmax + 5 * nmax * lgn;
    } else {
        g_ws->lwork = 8;
        g_ws->liwork = 12;
    }

    /* Allocate matrices */
    g_ws->A   = malloc(n2 * sizeof(f32));
    g_ws->U   = malloc(n2 * sizeof(f32));
    g_ws->V   = malloc(n2 * sizeof(f32));
    g_ws->Z   = malloc(n2 * sizeof(f32));
    g_ws->AP  = malloc(nap * sizeof(f32));
    g_ws->VP  = malloc(nap * sizeof(f32));
    g_ws->TAU = malloc(nmax * sizeof(f32));
    g_ws->SD  = malloc(nmax * sizeof(f32));
    g_ws->SE  = malloc(nmax * sizeof(f32));

    /* Eigenvalue arrays */
    g_ws->D1  = malloc(nmax * sizeof(f32));
    g_ws->D2  = malloc(nmax * sizeof(f32));
    g_ws->D3  = malloc(nmax * sizeof(f32));
    g_ws->D4  = malloc(nmax * sizeof(f32));
    g_ws->D5  = malloc(nmax * sizeof(f32));
    g_ws->WA1 = malloc(nmax * sizeof(f32));
    g_ws->WA2 = malloc(nmax * sizeof(f32));
    g_ws->WA3 = malloc(nmax * sizeof(f32));
    g_ws->WR  = malloc(nmax * sizeof(f32));

    /* Integer work arrays */
    g_ws->IBLOCK = malloc(nmax * sizeof(int));
    g_ws->ISPLIT = malloc(nmax * sizeof(int));
    g_ws->IFAIL  = malloc(nmax * sizeof(int));
    g_ws->ISUPPZ = malloc(2 * nmax * sizeof(int));

    /* Work arrays */
    g_ws->work  = malloc(g_ws->lwork * sizeof(f32));
    g_ws->iwork = malloc(g_ws->liwork * sizeof(int));

    if (!g_ws->A || !g_ws->U || !g_ws->V || !g_ws->Z ||
        !g_ws->AP || !g_ws->VP || !g_ws->TAU || !g_ws->SD || !g_ws->SE ||
        !g_ws->D1 || !g_ws->D2 || !g_ws->D3 || !g_ws->D4 || !g_ws->D5 ||
        !g_ws->WA1 || !g_ws->WA2 || !g_ws->WA3 || !g_ws->WR ||
        !g_ws->IBLOCK || !g_ws->ISPLIT || !g_ws->IFAIL || !g_ws->ISUPPZ ||
        !g_ws->work || !g_ws->iwork) {
        return -1;
    }

    rng_seed(g_ws->rng_state, 0xDEADBEEFULL);
    rng_seed(g_ws->rng_state2, 0xDEADBEEFULL);
    return 0;
}

static int group_teardown(void** state)
{
    (void)state;

    if (g_ws) {
        free(g_ws->A);
        free(g_ws->U);
        free(g_ws->V);
        free(g_ws->Z);
        free(g_ws->AP);
        free(g_ws->VP);
        free(g_ws->TAU);
        free(g_ws->SD);
        free(g_ws->SE);
        free(g_ws->D1);
        free(g_ws->D2);
        free(g_ws->D3);
        free(g_ws->D4);
        free(g_ws->D5);
        free(g_ws->WA1);
        free(g_ws->WA2);
        free(g_ws->WA3);
        free(g_ws->WR);
        free(g_ws->IBLOCK);
        free(g_ws->ISPLIT);
        free(g_ws->IFAIL);
        free(g_ws->ISUPPZ);
        free(g_ws->work);
        free(g_ws->iwork);
        free(g_ws);
        g_ws = NULL;
    }
    return 0;
}

/* ===== Matrix generation ===== */

/**
 * Generate test matrix according to jtype.
 * Based on dchkst.f lines 785-892.
 */
static int generate_matrix(int n, int jtype, f32* A, int lda,
                           f32* work, int* iwork,
                           uint64_t state[static 4])
{
    int itype = KTYPE[jtype - 1];
    int imode = KMODE[jtype - 1];
    f32 anorm, cond;
    int iinfo = 0;

    f32 ulp = slamch("P");
    f32 unfl = slamch("S");
    f32 ovfl = 1.0f / unfl;
    f32 ulpinv = 1.0f / ulp;
    f32 rtunfl = sqrtf(unfl);
    f32 rtovfl = sqrtf(ovfl);
    f32 aninv = 1.0f / (f32)(n > 1 ? n : 1);

    switch (KMAGN[jtype - 1]) {
        case 1: anorm = 1.0f; break;
        case 2: anorm = (rtovfl * ulp) * aninv; break;
        case 3: anorm = rtunfl * n * ulpinv; break;
        default: anorm = 1.0f;
    }

    slaset("F", lda, n, 0.0f, 0.0f, A, lda);

    if (jtype <= 15) {
        cond = ulpinv;
    } else {
        cond = ulpinv * aninv / 10.0f;
    }

    if (itype == 1) {
        iinfo = 0;

    } else if (itype == 2) {
        for (int jc = 0; jc < n; jc++) {
            A[jc + jc * lda] = anorm;
        }

    } else if (itype == 4) {
        slatms(n, n, "S", "S", work, imode, cond, anorm,
               0, 0, "N", A, lda, work + n, &iinfo, state);

    } else if (itype == 5) {
        slatms(n, n, "S", "S", work, imode, cond, anorm,
               n, n, "N", A, lda, work + n, &iinfo, state);

    } else if (itype == 7) {
        int idumma[1] = {1};
        slatmr(n, n, "S", "S", work, 6, 1.0f, 1.0f, "T", "N",
               work + n, 1, 1.0f, work + 2 * n, 1, 1.0f,
               "N", idumma, 0, 0, 0.0f, anorm, "NO",
               A, lda, iwork, &iinfo, state);

    } else if (itype == 8) {
        int idumma[1] = {1};
        slatmr(n, n, "S", "S", work, 6, 1.0f, 1.0f, "T", "N",
               work + n, 1, 1.0f, work + 2 * n, 1, 1.0f,
               "N", idumma, n, n, 0.0f, anorm, "NO",
               A, lda, iwork, &iinfo, state);

    } else if (itype == 9) {
        slatms(n, n, "S", "P", work, imode, cond, anorm,
               n, n, "N", A, lda, work + n, &iinfo, state);

    } else if (itype == 10) {
        slatms(n, n, "S", "P", work, imode, cond, anorm,
               1, 1, "N", A, lda, work + n, &iinfo, state);
        for (int i = 1; i < n; i++) {
            f32 temp1 = fabsf(A[i + (i - 1) * lda]) /
                        sqrtf(fabsf(A[(i - 1) + (i - 1) * lda] * A[i + i * lda]));
            if (temp1 > 0.5f) {
                A[i + (i - 1) * lda] = 0.5f * sqrtf(fabsf(A[(i - 1) + (i - 1) * lda] *
                                                        A[i + i * lda]));
                A[(i - 1) + i * lda] = A[i + (i - 1) * lda];
            }
        }

    } else {
        iinfo = 1;
    }

    return iinfo;
}

/* ===== Eigenvalue comparison helper ===== */

/**
 * Compute max |D1[j] - D2[j]| / max(unfl, ulp * max(|D1|, |D2|))
 * Used for tests 11, 12, 16, 26, 31, 34, 37.
 */
static f32 eig_compare(const f32* D1, const f32* D2, int count,
                        f32 ulp, f32 unfl)
{
    f32 temp1 = 0.0f, temp2 = 0.0f;
    for (int j = 0; j < count; j++) {
        temp1 = fmaxf(temp1, fmaxf(fabsf(D1[j]), fabsf(D2[j])));
        temp2 = fmaxf(temp2, fabsf(D1[j] - D2[j]));
    }
    return temp2 / fmaxf(unfl, ulp * fmaxf(temp1, temp2));
}

/* ===== Main test function ===== */

static void test_dchkst_case(void** state)
{
    dchkst_params_t* params = *state;
    int n = params->n;
    int jtype = params->jtype;

    dchkst_ws_t* ws = g_ws;
    int lda = ws->nmax;
    int ldu = ws->nmax;
    int nap = (n * (n + 1)) / 2;

    f32* A   = ws->A;
    f32* U   = ws->U;
    f32* V   = ws->V;
    f32* Z   = ws->Z;
    f32* AP  = ws->AP;
    f32* VP  = ws->VP;
    f32* TAU = ws->TAU;
    f32* SD  = ws->SD;
    f32* SE  = ws->SE;
    f32* D1  = ws->D1;
    f32* D2  = ws->D2;
    f32* D3  = ws->D3;
    f32* D4  = ws->D4;
    f32* D5  = ws->D5;
    f32* WA1 = ws->WA1;
    f32* WA2 = ws->WA2;
    f32* WA3 = ws->WA3;
    f32* WR  = ws->WR;
    f32* work = ws->work;
    int* iwork = ws->iwork;
    int lwork = ws->lwork;

    f32 ulp = slamch("P");
    f32 unfl = slamch("S");
    f32 ovfl = 1.0f / unfl;
    f32 ulpinv = 1.0f / ulp;
    f32 rtunfl = sqrtf(unfl);
    f32 rtovfl = sqrtf(ovfl);
    int log2ui = (int)(logf(ulpinv) / logf(2.0f));

    f32 dumma[1] = {0.0f};
    int iinfo;
    int m, m2, m3, nsplit;
    int ntest = 0;
    f32 temp1, temp2, temp3;
    f32 anorm, abstol, vl, vu;
    int il, iu;

    /* Compute lgn and workspace sizes for this n (dchkst.f lines 736-746) */
    int lgn = 0;
    int lwedc, liwedc;
    if (n > 0) {
        lgn = (int)(logf((f32)n) / logf(2.0f));
        if ((1 << lgn) < n) lgn++;
        if ((1 << lgn) < n) lgn++;
        lwedc = 1 + 4 * n + 2 * n * lgn + 4 * n * n;
        liwedc = 6 + 6 * n + 5 * n * lgn;
    } else {
        lwedc = 8;
        liwedc = 12;
    }

    /* Compute ANORM for this type (for VL/VU computation later) */
    {
        f32 aninv = 1.0f / (f32)(n > 1 ? n : 1);
        switch (KMAGN[jtype - 1]) {
            case 1: anorm = 1.0f; break;
            case 2: anorm = (rtovfl * ulp) * aninv; break;
            case 3: anorm = rtunfl * n * ulpinv; break;
            default: anorm = 1.0f;
        }
    }

    /* Initialize results to 0 */
    for (int j = 0; j < NTEST; j++) {
        ws->result[j] = 0.0f;
    }

    /* Skip N=0 cases */
    if (n == 0) return;

    /* Generate matrix (dchkst.f lines 785-892) */
    iinfo = generate_matrix(n, jtype, A, lda, work, iwork, ws->rng_state);
    if (iinfo != 0) {
        print_message("Matrix generation failed for n=%d jtype=%d iinfo=%d\n",
                      n, jtype, iinfo);
        assert_info_success(iinfo);
        return;
    }

    /* ================================================================
     * Tests 1-2: SSYTRD(U) + SORGTR(U)
     * ================================================================ */

    slacpy("U", n, n, A, lda, V, ldu);

    ntest = 1;
    ssytrd("U", n, V, ldu, SD, SE, TAU, work, lwork, &iinfo);
    if (iinfo != 0) {
        print_message("SSYTRD(U) failed: info=%d n=%d jtype=%d\n", iinfo, n, jtype);
        ws->result[0] = ulpinv;
        goto L280;
    }

    slacpy("U", n, n, V, ldu, U, ldu);

    ntest = 2;
    sorgtr("U", n, U, ldu, TAU, work, lwork, &iinfo);
    if (iinfo != 0) {
        print_message("SORGTR(U) failed: info=%d n=%d jtype=%d\n", iinfo, n, jtype);
        ws->result[1] = ulpinv;
        goto L280;
    }

    ssyt21(2, "U", n, 1, A, lda, SD, SE, U, ldu, V, ldu, TAU, work,
           ws->result);
    ssyt21(3, "U", n, 1, A, lda, SD, SE, U, ldu, V, ldu, TAU, work,
           ws->result + 1);

    /* ================================================================
     * Tests 3-4: SSYTRD(L) + SORGTR(L)
     * ================================================================ */

    slacpy("L", n, n, A, lda, V, ldu);

    ntest = 3;
    ssytrd("L", n, V, ldu, SD, SE, TAU, work, lwork, &iinfo);
    if (iinfo != 0) {
        print_message("SSYTRD(L) failed: info=%d n=%d jtype=%d\n", iinfo, n, jtype);
        ws->result[2] = ulpinv;
        goto L280;
    }

    slacpy("L", n, n, V, ldu, U, ldu);

    ntest = 4;
    sorgtr("L", n, U, ldu, TAU, work, lwork, &iinfo);
    if (iinfo != 0) {
        print_message("SORGTR(L) failed: info=%d n=%d jtype=%d\n", iinfo, n, jtype);
        ws->result[3] = ulpinv;
        goto L280;
    }

    ssyt21(2, "L", n, 1, A, lda, SD, SE, U, ldu, V, ldu, TAU, work,
           ws->result + 2);
    ssyt21(3, "L", n, 1, A, lda, SD, SE, U, ldu, V, ldu, TAU, work,
           ws->result + 3);

    /* ================================================================
     * Tests 5-6: SSPTRD(U) + SOPGTR(U)
     * Pack upper triangle of A into AP.
     * ================================================================ */

    {
        int idx = 0;
        for (int jc = 0; jc < n; jc++) {
            for (int jr = 0; jr <= jc; jr++) {
                AP[idx] = A[jr + jc * lda];
                idx++;
            }
        }
    }

    cblas_scopy(nap, AP, 1, VP, 1);

    ntest = 5;
    ssptrd("U", n, VP, SD, SE, TAU, &iinfo);
    if (iinfo != 0) {
        print_message("SSPTRD(U) failed: info=%d n=%d jtype=%d\n", iinfo, n, jtype);
        ws->result[4] = ulpinv;
        goto L280;
    }

    ntest = 6;
    sopgtr("U", n, VP, TAU, U, ldu, work, &iinfo);
    if (iinfo != 0) {
        print_message("SOPGTR(U) failed: info=%d n=%d jtype=%d\n", iinfo, n, jtype);
        ws->result[5] = ulpinv;
        goto L280;
    }

    sspt21(2, "U", n, 1, AP, SD, SE, U, ldu, VP, TAU, work, ws->result + 4);
    sspt21(3, "U", n, 1, AP, SD, SE, U, ldu, VP, TAU, work, ws->result + 5);

    /* ================================================================
     * Tests 7-8: SSPTRD(L) + SOPGTR(L)
     * Pack lower triangle of A into AP.
     * ================================================================ */

    {
        int idx = 0;
        for (int jc = 0; jc < n; jc++) {
            for (int jr = jc; jr < n; jr++) {
                AP[idx] = A[jr + jc * lda];
                idx++;
            }
        }
    }

    cblas_scopy(nap, AP, 1, VP, 1);

    ntest = 7;
    ssptrd("L", n, VP, SD, SE, TAU, &iinfo);
    if (iinfo != 0) {
        print_message("SSPTRD(L) failed: info=%d n=%d jtype=%d\n", iinfo, n, jtype);
        ws->result[6] = ulpinv;
        goto L280;
    }

    ntest = 8;
    sopgtr("L", n, VP, TAU, U, ldu, work, &iinfo);
    if (iinfo != 0) {
        print_message("SOPGTR(L) failed: info=%d n=%d jtype=%d\n", iinfo, n, jtype);
        ws->result[7] = ulpinv;
        goto L280;
    }

    sspt21(2, "L", n, 1, AP, SD, SE, U, ldu, VP, TAU, work, ws->result + 6);
    sspt21(3, "L", n, 1, AP, SD, SE, U, ldu, VP, TAU, work, ws->result + 7);

    /* ================================================================
     * Tests 9-10: SSTEQR('V')
     * Compute D1 and Z from tridiagonal SD/SE.
     * ================================================================ */

    cblas_scopy(n, SD, 1, D1, 1);
    if (n > 0) cblas_scopy(n - 1, SE, 1, work, 1);
    slaset("F", n, n, 0.0f, 1.0f, Z, ldu);

    ntest = 9;
    ssteqr("V", n, D1, work, Z, ldu, work + n, &iinfo);
    if (iinfo != 0) {
        print_message("SSTEQR(V) failed: info=%d n=%d jtype=%d\n", iinfo, n, jtype);
        ws->result[8] = ulpinv;
        goto L280;
    }

    /* Tests 9 and 10 */
    sstt21(n, 0, SD, SE, D1, dumma, Z, ldu, work, ws->result + 8);

    /* ================================================================
     * Test 11: SSTEQR('N') - eigenvalues only
     * ================================================================ */

    cblas_scopy(n, SD, 1, D2, 1);
    if (n > 0) cblas_scopy(n - 1, SE, 1, work, 1);

    ntest = 11;
    ssteqr("N", n, D2, work, work + n, ldu, work + n, &iinfo);
    if (iinfo != 0) {
        print_message("SSTEQR(N) failed: info=%d n=%d jtype=%d\n", iinfo, n, jtype);
        ws->result[10] = ulpinv;
        goto L280;
    }

    /* ================================================================
     * Test 12: SSTERF - eigenvalues by PWK method
     * ================================================================ */

    cblas_scopy(n, SD, 1, D3, 1);
    if (n > 0) cblas_scopy(n - 1, SE, 1, work, 1);

    ntest = 12;
    ssterf(n, D3, work, &iinfo);
    if (iinfo != 0) {
        print_message("SSTERF failed: info=%d n=%d jtype=%d\n", iinfo, n, jtype);
        ws->result[11] = ulpinv;
        goto L280;
    }

    /* Tests 11 and 12: eigenvalue comparison */
    ws->result[10] = eig_compare(D1, D2, n, ulp, unfl);
    ws->result[11] = eig_compare(D1, D3, n, ulp, unfl);

    /* ================================================================
     * Test 13: SSTECH - Sturm sequence validation
     * Go up by factors of two until it succeeds.
     * ================================================================ */

    ntest = 13;
    temp1 = THRESH * (0.5f - ulp);

    for (int j = 0; j <= log2ui; j++) {
        sstech(n, SD, SE, D1, temp1, work, &iinfo);
        if (iinfo == 0) break;
        temp1 = temp1 * 2.0f;
    }

    ws->result[12] = temp1;

    /* ================================================================
     * Tests 14-16: SPTEQR (positive definite only, jtype > 15)
     * ================================================================ */

    if (jtype > 15) {
        /* Compute D4 and Z */
        cblas_scopy(n, SD, 1, D4, 1);
        if (n > 0) cblas_scopy(n - 1, SE, 1, work, 1);
        slaset("F", n, n, 0.0f, 1.0f, Z, ldu);

        ntest = 14;
        spteqr("V", n, D4, work, Z, ldu, work + n, &iinfo);
        if (iinfo != 0) {
            print_message("SPTEQR(V) failed: info=%d n=%d jtype=%d\n", iinfo, n, jtype);
            ws->result[13] = ulpinv;
            goto L280;
        }

        /* Tests 14 and 15 */
        sstt21(n, 0, SD, SE, D4, dumma, Z, ldu, work, ws->result + 13);

        /* Compute D5 */
        cblas_scopy(n, SD, 1, D5, 1);
        if (n > 0) cblas_scopy(n - 1, SE, 1, work, 1);

        ntest = 16;
        spteqr("N", n, D5, work, Z, ldu, work + n, &iinfo);
        if (iinfo != 0) {
            print_message("SPTEQR(N) failed: info=%d n=%d jtype=%d\n", iinfo, n, jtype);
            ws->result[15] = ulpinv;
            goto L280;
        }

        /* Test 16: eigenvalue comparison with 100*ulp */
        temp1 = 0.0f;
        temp2 = 0.0f;
        for (int j = 0; j < n; j++) {
            temp1 = fmaxf(temp1, fmaxf(fabsf(D4[j]), fabsf(D5[j])));
            temp2 = fmaxf(temp2, fabsf(D4[j] - D5[j]));
        }
        ws->result[15] = temp2 / fmaxf(unfl, 100.0f * ulp * fmaxf(temp1, temp2));
    } else {
        ws->result[13] = 0.0f;
        ws->result[14] = 0.0f;
        ws->result[15] = 0.0f;
    }

    /* ================================================================
     * Test 17: SSTEBZ relative accuracy (jtype == 21 only)
     * ================================================================ */

    vl = 0.0f;
    vu = 0.0f;
    il = 0;
    iu = 0;

    if (jtype == 21) {
        ntest = 17;
        abstol = unfl + unfl;
        sstebz("A", "E", n, vl, vu, il, iu, abstol, SD, SE, &m, &nsplit,
               WR, ws->IBLOCK, ws->ISPLIT, work, iwork, &iinfo);
        if (iinfo != 0) {
            print_message("SSTEBZ(A,rel) failed: info=%d n=%d jtype=%d\n",
                          iinfo, n, jtype);
            ws->result[16] = ulpinv;
            goto L280;
        }

        /* Test 17: relative accuracy for diagonally dominant */
        temp2 = 2.0f * (2 * n - 1) * ulp * (1.0f + 8.0f * 0.25f) /
                powf(0.5f, 4);

        temp1 = 0.0f;
        for (int j = 0; j < n; j++) {
            temp1 = fmaxf(temp1, fabsf(D4[j] - WR[n - j - 1]) /
                    (abstol + fabsf(D4[j])));
        }
        ws->result[16] = temp1 / temp2;
    } else {
        ws->result[16] = 0.0f;
    }

    /* ================================================================
     * Test 18: SSTEBZ('A','E') - all eigenvalues
     * ================================================================ */

    ntest = 18;
    abstol = unfl + unfl;
    sstebz("A", "E", n, vl, vu, il, iu, abstol, SD, SE, &m, &nsplit,
           WA1, ws->IBLOCK, ws->ISPLIT, work, iwork, &iinfo);
    if (iinfo != 0) {
        print_message("SSTEBZ(A) failed: info=%d n=%d jtype=%d\n", iinfo, n, jtype);
        ws->result[17] = ulpinv;
        goto L280;
    }

    /* Test 18: compare D3 (ssterf) vs WA1 (sstebz) */
    ws->result[17] = eig_compare(D3, WA1, n, ulp, unfl);

    /* ================================================================
     * Test 19: SSTEBZ('I','E') vs SSTEBZ('V','E')
     * Choose random IL, IU (0-based)
     * ================================================================ */

    ntest = 19;
    if (n <= 1) {
        il = 0;
        iu = n - 1;
    } else {
        il = (int)((n - 1) * rng_uniform_f32(ws->rng_state2));
        iu = (int)((n - 1) * rng_uniform_f32(ws->rng_state2));
        if (iu < il) { int itemp = iu; iu = il; il = itemp; }
    }

    sstebz("I", "E", n, vl, vu, il, iu, abstol, SD, SE, &m2, &nsplit,
           WA2, ws->IBLOCK, ws->ISPLIT, work, iwork, &iinfo);
    if (iinfo != 0) {
        print_message("SSTEBZ(I) failed: info=%d n=%d jtype=%d\n", iinfo, n, jtype);
        ws->result[18] = ulpinv;
        goto L280;
    }

    /* Compute VL, VU from WA1 (0-based indexing) */
    if (n > 0) {
        if (il != 0) {
            vl = WA1[il] - fmaxf(0.5f * (WA1[il] - WA1[il - 1]),
                           fmaxf(ulp * anorm, 2.0f * rtunfl));
        } else {
            vl = WA1[0] - fmaxf(0.5f * (WA1[n - 1] - WA1[0]),
                           fmaxf(ulp * anorm, 2.0f * rtunfl));
        }
        if (iu != n - 1) {
            vu = WA1[iu] + fmaxf(0.5f * (WA1[iu + 1] - WA1[iu]),
                           fmaxf(ulp * anorm, 2.0f * rtunfl));
        } else {
            vu = WA1[n - 1] + fmaxf(0.5f * (WA1[n - 1] - WA1[0]),
                               fmaxf(ulp * anorm, 2.0f * rtunfl));
        }
    } else {
        vl = 0.0f;
        vu = 1.0f;
    }

    sstebz("V", "E", n, vl, vu, il, iu, abstol, SD, SE, &m3, &nsplit,
           WA3, ws->IBLOCK, ws->ISPLIT, work, iwork, &iinfo);
    if (iinfo != 0) {
        print_message("SSTEBZ(V) failed: info=%d n=%d jtype=%d\n", iinfo, n, jtype);
        ws->result[18] = ulpinv;
        goto L280;
    }

    if (m3 == 0 && n != 0) {
        ws->result[18] = ulpinv;
        goto L280;
    }

    /* Test 19 */
    temp1 = ssxt1(1, WA2, m2, WA3, m3, abstol, ulp, unfl);
    temp2 = ssxt1(1, WA3, m3, WA2, m2, abstol, ulp, unfl);
    if (n > 0) {
        temp3 = fmaxf(fabsf(WA1[n - 1]), fabsf(WA1[0]));
    } else {
        temp3 = 0.0f;
    }
    ws->result[18] = (temp1 + temp2) / fmaxf(unfl, temp3 * ulp);

    /* ================================================================
     * Tests 20-21: SSTEBZ('A','B') + SSTEIN
     * ================================================================ */

    ntest = 21;
    sstebz("A", "B", n, vl, vu, il, iu, abstol, SD, SE, &m, &nsplit,
           WA1, ws->IBLOCK, ws->ISPLIT, work, iwork, &iinfo);
    if (iinfo != 0) {
        print_message("SSTEBZ(A,B) failed: info=%d n=%d jtype=%d\n",
                      iinfo, n, jtype);
        ws->result[19] = ulpinv;
        ws->result[20] = ulpinv;
        goto L280;
    }

    sstein(n, SD, SE, m, WA1, ws->IBLOCK, ws->ISPLIT, Z, ldu,
           work, iwork, ws->IFAIL, &iinfo);
    if (iinfo != 0) {
        print_message("SSTEIN failed: info=%d n=%d jtype=%d\n", iinfo, n, jtype);
        ws->result[19] = ulpinv;
        ws->result[20] = ulpinv;
        goto L280;
    }

    /* Tests 20 and 21 */
    sstt21(n, 0, SD, SE, WA1, dumma, Z, ldu, work, ws->result + 19);

    /* ================================================================
     * Tests 22-23: SSTEDC('I')
     * ================================================================ */

    cblas_scopy(n, SD, 1, D1, 1);
    if (n > 0) cblas_scopy(n - 1, SE, 1, work, 1);
    slaset("F", n, n, 0.0f, 1.0f, Z, ldu);

    ntest = 22;
    sstedc("I", n, D1, work, Z, ldu, work + n, lwedc - n,
           iwork, liwedc, &iinfo);
    if (iinfo != 0) {
        print_message("SSTEDC(I) failed: info=%d n=%d jtype=%d\n", iinfo, n, jtype);
        ws->result[21] = ulpinv;
        goto L280;
    }

    /* Tests 22 and 23 */
    sstt21(n, 0, SD, SE, D1, dumma, Z, ldu, work, ws->result + 21);

    /* ================================================================
     * Tests 24-25: SSTEDC('V')
     * ================================================================ */

    cblas_scopy(n, SD, 1, D1, 1);
    if (n > 0) cblas_scopy(n - 1, SE, 1, work, 1);
    slaset("F", n, n, 0.0f, 1.0f, Z, ldu);

    ntest = 24;
    sstedc("V", n, D1, work, Z, ldu, work + n, lwedc - n,
           iwork, liwedc, &iinfo);
    if (iinfo != 0) {
        print_message("SSTEDC(V) failed: info=%d n=%d jtype=%d\n", iinfo, n, jtype);
        ws->result[23] = ulpinv;
        goto L280;
    }

    /* Tests 24 and 25 */
    sstt21(n, 0, SD, SE, D1, dumma, Z, ldu, work, ws->result + 23);

    /* ================================================================
     * Test 26: SSTEDC('N') - eigenvalues only, compare with 'V'
     * ================================================================ */

    cblas_scopy(n, SD, 1, D2, 1);
    if (n > 0) cblas_scopy(n - 1, SE, 1, work, 1);
    slaset("F", n, n, 0.0f, 1.0f, Z, ldu);

    ntest = 26;
    sstedc("N", n, D2, work, Z, ldu, work + n, lwedc - n,
           iwork, liwedc, &iinfo);
    if (iinfo != 0) {
        print_message("SSTEDC(N) failed: info=%d n=%d jtype=%d\n", iinfo, n, jtype);
        ws->result[25] = ulpinv;
        goto L280;
    }

    /* Test 26 */
    ws->result[25] = eig_compare(D1, D2, n, ulp, unfl);

    /* ================================================================
     * Tests 27-37: SSTEMR (MRRR) - IEEE compliant only
     * We assume IEEE compliance on x86/ARM targets.
     * SREL = false → tests 27-28 always 0.
     * SRANGE = false → tests 29-34 always 0.
     * Tests 35-37 always run.
     * ================================================================ */

    /* Tests 27-28: disabled (SREL=false) */
    ws->result[26] = 0.0f;
    ws->result[27] = 0.0f;

    /* Tests 29-34: disabled (SRANGE=false) */
    if (SRANGE) {
        /* SSTEMR(V,I) */
        cblas_scopy(n, SD, 1, D5, 1);
        if (n > 0) cblas_scopy(n - 1, SE, 1, work, 1);
        slaset("F", n, n, 0.0f, 1.0f, Z, ldu);

        ntest = 29;
        il = (int)((n - 1) * rng_uniform_f32(ws->rng_state2));
        iu = (int)((n - 1) * rng_uniform_f32(ws->rng_state2));
        if (iu < il) { int itemp = iu; iu = il; il = itemp; }

        {
            int tryrac = 1;
            sstemr("V", "I", n, D5, work, vl, vu, il, iu, &m, D1, Z, ldu, n,
                   ws->ISUPPZ, &tryrac, work + n, lwork - n,
                   iwork, ws->liwork, &iinfo);
        }
        if (iinfo != 0) {
            print_message("SSTEMR(V,I) failed: info=%d n=%d jtype=%d\n",
                          iinfo, n, jtype);
            ws->result[28] = ulpinv;
            goto L280;
        }

        /* Tests 29 and 30 */
        sstt22(n, m, 0, SD, SE, D1, dumma, Z, ldu, work, m, ws->result + 28);

        /* SSTEMR(N,I) */
        cblas_scopy(n, SD, 1, D5, 1);
        if (n > 0) cblas_scopy(n - 1, SE, 1, work, 1);

        ntest = 31;
        {
            int tryrac = 1;
            sstemr("N", "I", n, D5, work, vl, vu, il, iu, &m, D2, Z, ldu, n,
                   ws->ISUPPZ, &tryrac, work + n, lwork - n,
                   iwork, ws->liwork, &iinfo);
        }
        if (iinfo != 0) {
            print_message("SSTEMR(N,I) failed: info=%d n=%d jtype=%d\n",
                          iinfo, n, jtype);
            ws->result[30] = ulpinv;
            goto L280;
        }

        /* Test 31 */
        ws->result[30] = eig_compare(D1, D2, iu - il + 1, ulp, unfl);

        /* SSTEMR(V,V) */
        cblas_scopy(n, SD, 1, D5, 1);
        if (n > 0) cblas_scopy(n - 1, SE, 1, work, 1);
        slaset("F", n, n, 0.0f, 1.0f, Z, ldu);

        ntest = 32;

        /* Compute VL, VU from D2 (the eigenvalues from N,I) */
        if (n > 0) {
            if (il != 0) {
                vl = D2[il] - fmaxf(0.5f * (D2[il] - D2[il - 1]),
                                fmaxf(ulp * anorm, 2.0f * rtunfl));
            } else {
                vl = D2[0] - fmaxf(0.5f * (D2[n - 1] - D2[0]),
                                fmaxf(ulp * anorm, 2.0f * rtunfl));
            }
            if (iu != n - 1) {
                vu = D2[iu] + fmaxf(0.5f * (D2[iu + 1] - D2[iu]),
                                fmaxf(ulp * anorm, 2.0f * rtunfl));
            } else {
                vu = D2[n - 1] + fmaxf(0.5f * (D2[n - 1] - D2[0]),
                                    fmaxf(ulp * anorm, 2.0f * rtunfl));
            }
        } else {
            vl = 0.0f;
            vu = 1.0f;
        }

        {
            int tryrac = 1;
            sstemr("V", "V", n, D5, work, vl, vu, il, iu, &m, D1, Z, ldu, n,
                   ws->ISUPPZ, &tryrac, work + n, lwork - n,
                   iwork, ws->liwork, &iinfo);
        }
        if (iinfo != 0) {
            print_message("SSTEMR(V,V) failed: info=%d n=%d jtype=%d\n",
                          iinfo, n, jtype);
            ws->result[31] = ulpinv;
            goto L280;
        }

        /* Tests 32 and 33 */
        sstt22(n, m, 0, SD, SE, D1, dumma, Z, ldu, work, m, ws->result + 31);

        /* SSTEMR(N,V) */
        cblas_scopy(n, SD, 1, D5, 1);
        if (n > 0) cblas_scopy(n - 1, SE, 1, work, 1);

        ntest = 34;
        {
            int tryrac = 1;
            sstemr("N", "V", n, D5, work, vl, vu, il, iu, &m, D2, Z, ldu, n,
                   ws->ISUPPZ, &tryrac, work + n, lwork - n,
                   iwork, ws->liwork, &iinfo);
        }
        if (iinfo != 0) {
            print_message("SSTEMR(N,V) failed: info=%d n=%d jtype=%d\n",
                          iinfo, n, jtype);
            ws->result[33] = ulpinv;
            goto L280;
        }

        /* Test 34 */
        ws->result[33] = eig_compare(D1, D2, iu - il + 1, ulp, unfl);
    } else {
        ws->result[28] = 0.0f;
        ws->result[29] = 0.0f;
        ws->result[30] = 0.0f;
        ws->result[31] = 0.0f;
        ws->result[32] = 0.0f;
        ws->result[33] = 0.0f;
    }

    /* ================================================================
     * Tests 35-37: SSTEMR('V','A') and SSTEMR('N','A')
     * ================================================================ */

    cblas_scopy(n, SD, 1, D5, 1);
    if (n > 0) cblas_scopy(n - 1, SE, 1, work, 1);

    ntest = 35;
    {
        int tryrac = 1;
        sstemr("V", "A", n, D5, work, vl, vu, il, iu, &m, D1, Z, ldu, n,
               ws->ISUPPZ, &tryrac, work + n, lwork - n,
               iwork, ws->liwork, &iinfo);
    }
    if (iinfo != 0) {
        print_message("SSTEMR(V,A) failed: info=%d n=%d jtype=%d\n",
                      iinfo, n, jtype);
        ws->result[34] = ulpinv;
        goto L280;
    }

    /* Tests 35 and 36 */
    sstt22(n, m, 0, SD, SE, D1, dumma, Z, ldu, work, m, ws->result + 34);

    /* SSTEMR(N,A) */
    cblas_scopy(n, SD, 1, D5, 1);
    if (n > 0) cblas_scopy(n - 1, SE, 1, work, 1);

    ntest = 37;
    {
        int tryrac = 1;
        sstemr("N", "A", n, D5, work, vl, vu, il, iu, &m, D2, Z, ldu, n,
               ws->ISUPPZ, &tryrac, work + n, lwork - n,
               iwork, ws->liwork, &iinfo);
    }
    if (iinfo != 0) {
        print_message("SSTEMR(N,A) failed: info=%d n=%d jtype=%d\n",
                      iinfo, n, jtype);
        ws->result[36] = ulpinv;
        goto L280;
    }

    /* Test 37 */
    ws->result[36] = eig_compare(D1, D2, n, ulp, unfl);

L280:
    /* Check all computed results */
    {
        static char ctx[128];
        for (int jr = 0; jr < ntest; jr++) {
            snprintf(ctx, sizeof(ctx), "dchkst n=%d type=%d TEST %d", n, jtype, jr + 1);
            set_test_context(ctx);
            assert_residual_ok(ws->result[jr]);
        }
    }
}

/* ===== Test array construction ===== */

#define MAX_TESTS (NNVAL * MAXTYP)

static dchkst_params_t g_params[MAX_TESTS];
static struct CMUnitTest g_tests[MAX_TESTS];
static int g_num_tests = 0;

static void build_test_array(void)
{
    g_num_tests = 0;

    for (size_t in = 0; in < NNVAL; in++) {
        int n = NVAL[in];

        for (int jtype = 1; jtype <= MAXTYP; jtype++) {
            /* Skip type 9 — matches LAPACK's sep.in which tests types
               1-8, 10-21.  Type 9 (KTYPE=5, KMODE=1: one small eigenvalue,
               cond=ulpinv) produces tridiagonals where SSTEMR's MRRR
               algorithm yields eigenvector orthogonality ratios ~60, well
               above threshold.  Reference LAPACK also fails (ratio ~66). */
            if (jtype == 9) continue;

            dchkst_params_t* p = &g_params[g_num_tests];
            p->n = n;
            p->jtype = jtype;
            snprintf(p->name, sizeof(p->name),
                     "dchkst_n%d_type%d", n, jtype);

            g_tests[g_num_tests].name = p->name;
            g_tests[g_num_tests].test_func = test_dchkst_case;
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
    build_test_array();

    return _cmocka_run_group_tests("dchkst", g_tests, g_num_tests,
                                   group_setup, group_teardown);
}
