/**
 * @file test_cdrvbd.c
 * @brief CDRVBD checks the singular value decomposition (SVD) drivers
 *        CGESVD, CGESDD, CGESVJ, CGEJSV, CGESVDX, and CGESVDQ - port of
 *        LAPACK TESTING/EIG/cdrvbd.f
 *
 * CGESVD and CGESDD factors A = U diag(S) VT, where U and VT are
 * unitary and diag(S) is diagonal with the entries of the array S on
 * its diagonal. The entries of S are the singular values, nonnegative
 * and stored in decreasing order.  U and VT can be optionally not
 * computed, overwritten on A, or computed partially.
 *
 * A is M by N. Let MNMIN = min( M, N ). S has dimension MNMIN.
 * U can be M by M or M by MNMIN. VT can be N by N or MNMIN by N.
 *
 * When CDRVBD is called, a number of matrix "sizes" (M's and N's)
 * and a number of matrix "types" are specified.  For each size (M,N)
 * and each type of matrix, and for the minimal workspace as well as
 * workspace adequate to permit blocking, an  M x N  matrix "A" will be
 * generated and used to test the SVD routines.  For each matrix, A will
 * be factored as A = U diag(S) VT and the following 12 tests computed:
 *
 * Test for CGESVD:
 *
 * (1)   | A - U diag(S) VT | / ( |A| max(M,N) ulp )
 *
 * (2)   | I - U'U | / ( M ulp )
 *
 * (3)   | I - VT VT' | / ( N ulp )
 *
 * (4)   S contains MNMIN nonnegative values in decreasing order.
 *       (Return 0 if true, 1/ULP if false.)
 *
 * (5)   | U - Upartial | / ( M ulp ) where Upartial is a partially
 *       computed U.
 *
 * (6)   | VT - VTpartial | / ( N ulp ) where VTpartial is a partially
 *       computed VT.
 *
 * (7)   | S - Spartial | / ( MNMIN ulp |S| ) where Spartial is the
 *       vector of singular values from the partial SVD
 *
 * Test for CGESDD:
 *
 * (8)   | A - U diag(S) VT | / ( |A| max(M,N) ulp )
 *
 * (9)   | I - U'U | / ( M ulp )
 *
 * (10)  | I - VT VT' | / ( N ulp )
 *
 * (11)  S contains MNMIN nonnegative values in decreasing order.
 *       (Return 0 if true, 1/ULP if false.)
 *
 * (12)  | U - Upartial | / ( M ulp ) where Upartial is a partially
 *       computed U.
 *
 * (13)  | VT - VTpartial | / ( N ulp ) where VTpartial is a partially
 *       computed VT.
 *
 * (14)  | S - Spartial | / ( MNMIN ulp |S| ) where Spartial is the
 *       vector of singular values from the partial SVD
 *
 * Test for CGESVDQ:
 *
 * (36)  | A - U diag(S) VT | / ( |A| max(M,N) ulp )
 *
 * (37)  | I - U'U | / ( M ulp )
 *
 * (38)  | I - VT VT' | / ( N ulp )
 *
 * (39)  S contains MNMIN nonnegative values in decreasing order.
 *       (Return 0 if true, 1/ULP if false.)
 *
 * Test for CGESVJ:
 *
 * (15)  | A - U diag(S) VT | / ( |A| max(M,N) ulp )
 *
 * (16)  | I - U'U | / ( M ulp )
 *
 * (17)  | I - VT VT' | / ( N ulp )
 *
 * (18)  S contains MNMIN nonnegative values in decreasing order.
 *       (Return 0 if true, 1/ULP if false.)
 *
 * Test for CGEJSV:
 *
 * (19)  | A - U diag(S) VT | / ( |A| max(M,N) ulp )
 *
 * (20)  | I - U'U | / ( M ulp )
 *
 * (21)  | I - VT VT' | / ( N ulp )
 *
 * (22)  S contains MNMIN nonnegative values in decreasing order.
 *        (Return 0 if true, 1/ULP if false.)
 *
 * Test for CGESVDX( 'V', 'V', 'A' )/CGESVDX( 'N', 'N', 'A' )
 *
 * (23)  | A - U diag(S) VT | / ( |A| max(M,N) ulp )
 *
 * (24)  | I - U'U | / ( M ulp )
 *
 * (25)  | I - VT VT' | / ( N ulp )
 *
 * (26)  S contains MNMIN nonnegative values in decreasing order.
 *       (Return 0 if true, 1/ULP if false.)
 *
 * (27)  | U - Upartial | / ( M ulp ) where Upartial is a partially
 *       computed U.
 *
 * (28)  | VT - VTpartial | / ( N ulp ) where VTpartial is a partially
 *       computed VT.
 *
 * (29)  | S - Spartial | / ( MNMIN ulp |S| ) where Spartial is the
 *       vector of singular values from the partial SVD
 *
 * Test for CGESVDX( 'V', 'V', 'I' )
 *
 * (30)  | U' A VT''' - diag(S) | / ( |A| max(M,N) ulp )
 *
 * (31)  | I - U'U | / ( M ulp )
 *
 * (32)  | I - VT VT' | / ( N ulp )
 *
 * Test for CGESVDX( 'V', 'V', 'V' )
 *
 * (33)   | U' A VT''' - diag(S) | / ( |A| max(M,N) ulp )
 *
 * (34)   | I - U'U | / ( M ulp )
 *
 * (35)   | I - VT VT' | / ( N ulp )
 *
 * Currently, the list of possible types is:
 *
 * (1)  The zero matrix.
 * (2)  The identity matrix.
 * (3)  A matrix of the form  U D V, where U and V are unitary and
 *      D has evenly spaced entries 1, ..., ULP with random signs
 *      on the diagonal.
 * (4)  Same as (3), but multiplied by the underflow-threshold / ULP.
 * (5)  Same as (3), but multiplied by the overflow-threshold * ULP.
 */

#include "test_harness.h"
#include "test_rng.h"
#include "verify.h"
#include <math.h>
#include <complex.h>
#include <string.h>

/* THRESH from svd.in */
#define THRESH 50.0f

#define MAXTYP 5

/* NMAX, LWORK = NMAX*(5*NMAX+20) and LIWORK = NMAX*(NMAX+20) of CCHKEE,
 * which passes LDA = LDU = LDVT = NMAX, WORK(LWORK), RWORK(LWORK) and
 * IWORK(LIWORK) to CDRVBD */
#define CHKEE_NMAX 132

/* M,N pairs from svd.in */
static const INT MVAL[] = {0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 3, 10, 10, 16, 16, 30, 30, 40, 40};
static const INT NVAL[] = {0, 1, 3, 0, 1, 2, 0, 1, 0, 1, 3, 10, 16, 10, 16, 30, 40, 30, 40};
#define NSIZES ((INT)(sizeof(MVAL) / sizeof(MVAL[0])))

/* NBVAL, NBMIN, NXVAL from svd.in */
static const INT NBVAL[] = {1, 3, 3, 3, 20};
static const INT NBMIN[] = {2, 2, 2, 2, 2};
static const INT NXVAL[] = {1, 0, 5, 9, 1};
#define NPARMS (sizeof(NBVAL) / sizeof(NBVAL[0]))

static const char CJOB[4] = {'N', 'O', 'S', 'A'};
static const char CJOBR[3] = {'A', 'V', 'I'};
static const char CJOBV[2] = {'N', 'V'};

typedef struct {
    INT jsize;
    INT jtype;
    INT iparm;
    char name[64];
} cdrvbd_params_t;

typedef struct {
    INT lda;
    INT ldu;
    INT ldvt;

    c64* A;
    c64* U;
    c64* VT;
    c64* ASAV;
    c64* USAV;
    c64* VTSAV;
    f32* S;
    f32* SSAV;
    f32* E;

    c64* work;
    INT lwork;
    f32* rwork;
    INT* iwork;

    f32 anorm;
    uint64_t rng_state[4];
    INT ntestf;
    INT ntestf_iparm;
} cdrvbd_workspace_t;

static cdrvbd_workspace_t* g_ws = NULL;

static inline f32 fort_max(const f32 a, const f32 b)
{
    return (a > b) ? a : b;
}

static int group_setup(void** state)
{
    (void)state;

    g_ws = calloc(1, sizeof(cdrvbd_workspace_t));
    if (!g_ws) return -1;

    g_ws->lda = CHKEE_NMAX;
    g_ws->ldu = CHKEE_NMAX;
    g_ws->ldvt = CHKEE_NMAX;
    g_ws->lwork = CHKEE_NMAX * (5 * CHKEE_NMAX + 20);
    INT liwork = CHKEE_NMAX * (CHKEE_NMAX + 20);

    /* Check for errors */
    INT info = 0;
    INT badmm = 0;
    INT badnn = 0;
    INT mmax = 1;
    INT nmax = 1;
    INT minwrk = 1;
    for (INT j = 0; j < NSIZES; j++) {
        if (MVAL[j] > mmax) mmax = MVAL[j];
        if (MVAL[j] < 0) badmm = 1;
        if (NVAL[j] > nmax) nmax = NVAL[j];
        if (NVAL[j] < 0) badnn = 1;
        INT mnj = (MVAL[j] < NVAL[j]) ? MVAL[j] : NVAL[j];
        INT mxj = (MVAL[j] > NVAL[j]) ? MVAL[j] : NVAL[j];
        INT w = 3 * mnj + mxj * mxj;
        if (5 * mnj > w) w = 5 * mnj;
        if (3 * mxj > w) w = 3 * mxj;
        if (w > minwrk) minwrk = w;
    }

    if (badmm) {
        info = -2;
    } else if (badnn) {
        info = -3;
    } else if (g_ws->lda < mmax) {
        info = -10;
    } else if (g_ws->ldu < mmax) {
        info = -12;
    } else if (g_ws->ldvt < nmax) {
        info = -14;
    } else if (minwrk > g_ws->lwork) {
        info = -21;
    }

    if (info != 0) {
        fprintf(stderr, " CDRVBD: parameter number %d had an illegal value\n",
                (int)(-info));
        free(g_ws);
        g_ws = NULL;
        return -1;
    }

    INT nn = CHKEE_NMAX * CHKEE_NMAX;
    g_ws->A     = malloc((size_t)nn * sizeof(c64));
    g_ws->U     = malloc((size_t)nn * sizeof(c64));
    g_ws->VT    = malloc((size_t)nn * sizeof(c64));
    g_ws->ASAV  = malloc((size_t)nn * sizeof(c64));
    g_ws->USAV  = malloc((size_t)nn * sizeof(c64));
    g_ws->VTSAV = malloc((size_t)nn * sizeof(c64));
    g_ws->S     = calloc(CHKEE_NMAX, sizeof(f32));
    g_ws->SSAV  = calloc(CHKEE_NMAX, sizeof(f32));
    g_ws->E     = calloc(CHKEE_NMAX, sizeof(f32));
    g_ws->work  = malloc((size_t)g_ws->lwork * sizeof(c64));
    g_ws->rwork = malloc((size_t)g_ws->lwork * sizeof(f32));
    g_ws->iwork = malloc((size_t)liwork * sizeof(INT));

    if (!g_ws->A || !g_ws->U || !g_ws->VT || !g_ws->ASAV || !g_ws->USAV ||
        !g_ws->VTSAV || !g_ws->S || !g_ws->SSAV || !g_ws->E ||
        !g_ws->work || !g_ws->rwork || !g_ws->iwork) {
        return -1;
    }

    g_ws->anorm = 0.0f;
    g_ws->ntestf = 0;
    g_ws->ntestf_iparm = -1;
    rng_seed(g_ws->rng_state, 2024);
    xlaenv(9, 25);
    xlaenv(1, 1);
    return 0;
}

static int group_teardown(void** state)
{
    (void)state;

    if (g_ws) {
        free(g_ws->A);
        free(g_ws->U);
        free(g_ws->VT);
        free(g_ws->ASAV);
        free(g_ws->USAV);
        free(g_ws->VTSAV);
        free(g_ws->S);
        free(g_ws->SSAV);
        free(g_ws->E);
        free(g_ws->work);
        free(g_ws->rwork);
        free(g_ws->iwork);
        free(g_ws);
        g_ws = NULL;
    }
    return 0;
}

static void print_header(void)
{
    fprintf(stderr,
        " SVD -- Complex Singular Value Decomposition Driver \n"
        " Matrix types (see CDRVBD for details):\n"
        "\n"
        " 1 = Zero matrix\n"
        " 2 = Identity matrix\n"
        " 3 = Evenly spaced singular values near 1\n"
        " 4 = Evenly spaced singular values near underflow\n"
        " 5 = Evenly spaced singular values near overflow\n"
        "\n"
        " Tests performed: ( A is dense, U and V are unitary,\n"
        "                    S is an array, and Upartial, VTpartial, and\n"
        "                    Spartial are partially computed U, VT and S),\n"
        "\n");
    fprintf(stderr,
        " Tests performed with Test Threshold = %8.2f\n"
        " CGESVD: \n"
        " 1 = | A - U diag(S) VT | / ( |A| max(M,N) ulp ) \n"
        " 2 = | I - U**T U | / ( M ulp ) \n"
        " 3 = | I - VT VT**T | / ( N ulp ) \n"
        " 4 = 0 if S contains min(M,N) nonnegative values in"
        " decreasing order, else 1/ulp\n"
        " 5 = | U - Upartial | / ( M ulp )\n"
        " 6 = | VT - VTpartial | / ( N ulp )\n"
        " 7 = | S - Spartial | / ( min(M,N) ulp |S| )\n"
        " CGESDD: \n"
        " 8 = | A - U diag(S) VT | / ( |A| max(M,N) ulp ) \n"
        " 9 = | I - U**T U | / ( M ulp ) \n"
        "10 = | I - VT VT**T | / ( N ulp ) \n"
        "11 = 0 if S contains min(M,N) nonnegative values in"
        " decreasing order, else 1/ulp\n"
        "12 = | U - Upartial | / ( M ulp )\n"
        "13 = | VT - VTpartial | / ( N ulp )\n"
        "14 = | S - Spartial | / ( min(M,N) ulp |S| )\n"
        " CGESVJ: \n"
        "\n"
        "15 = | A - U diag(S) VT | / ( |A| max(M,N) ulp ) \n"
        "16 = | I - U**T U | / ( M ulp ) \n"
        "17 = | I - VT VT**T | / ( N ulp ) \n"
        "18 = 0 if S contains min(M,N) nonnegative values in"
        " decreasing order, else 1/ulp\n"
        " CGESJV: \n"
        "\n"
        "19 = | A - U diag(S) VT | / ( |A| max(M,N) ulp )\n"
        "20 = | I - U**T U | / ( M ulp ) \n"
        "21 = | I - VT VT**T | / ( N ulp ) \n"
        "22 = 0 if S contains min(M,N) nonnegative values in"
        " decreasing order, else 1/ulp\n"
        " CGESVDX(V,V,A): \n"
        "23 = | A - U diag(S) VT | / ( |A| max(M,N) ulp ) \n"
        "24 = | I - U**T U | / ( M ulp ) \n"
        "25 = | I - VT VT**T | / ( N ulp ) \n"
        "26 = 0 if S contains min(M,N) nonnegative values in"
        " decreasing order, else 1/ulp\n"
        "27 = | U - Upartial | / ( M ulp )\n"
        "28 = | VT - VTpartial | / ( N ulp )\n"
        "29 = | S - Spartial | / ( min(M,N) ulp |S| )\n"
        " CGESVDX(V,V,I): \n"
        "30 = | U**T A VT**T - diag(S) | / ( |A| max(M,N) ulp )\n"
        "31 = | I - U**T U | / ( M ulp ) \n"
        "32 = | I - VT VT**T | / ( N ulp ) \n"
        " CGESVDX(V,V,V) \n"
        "33 = | U**T A VT**T - diag(S) | / ( |A| max(M,N) ulp )\n"
        "34 = | I - U**T U | / ( M ulp ) \n"
        "35 = | I - VT VT**T | / ( N ulp ) "
        " CGESVDQ(H,N,N,A,A\n"
        "36 = | A - U diag(S) VT | / ( |A| max(M,N) ulp ) \n"
        "37 = | I - U**T U | / ( M ulp ) \n"
        "38 = | I - VT VT**T | / ( N ulp ) \n"
        "39 = 0 if S contains min(M,N) nonnegative values in"
        " decreasing order, else 1/ulp\n"
        "\n"
        "\n",
        (double)THRESH);
}

static void info_fail(const char* name, const INT iinfo, const INT m,
                      const INT n, const INT jtype, const INT lswork,
                      const uint64_t* ioldsd)
{
    fprintf(stderr, " CDRVBD: %s returned INFO=%6d.\n"
                    "         M=%6d, N=%6d, JTYPE=%6d, LSWORK=%6d\n"
                    "         ISEED=(%llu,%llu,%llu,%llu)\n",
            name, (int)iinfo, (int)m, (int)n, (int)jtype, (int)lswork,
            (unsigned long long)ioldsd[0], (unsigned long long)ioldsd[1],
            (unsigned long long)ioldsd[2], (unsigned long long)ioldsd[3]);
    fail_msg("CDRVBD: %s returned INFO=%d M=%d N=%d JTYPE=%d LSWORK=%d",
             name, (int)iinfo, (int)m, (int)n, (int)jtype, (int)lswork);
}

static void run_cdrvbd_single(const cdrvbd_params_t* p)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;
    const f32 TWO = 2.0f;
    const f32 HALF = 0.5f;
    const c64 CZERO = CMPLXF(0.0f, 0.0f);
    const c64 CONE = CMPLXF(1.0f, 0.0f);

    cdrvbd_workspace_t* ws = g_ws;
    const INT lda = ws->lda;
    const INT ldu = ws->ldu;
    const INT ldvt = ws->ldvt;
    const INT lwork = ws->lwork;
    c64* A = ws->A;
    c64* U = ws->U;
    c64* VT = ws->VT;
    c64* ASAV = ws->ASAV;
    c64* USAV = ws->USAV;
    c64* VTSAV = ws->VTSAV;
    f32* S = ws->S;
    f32* SSAV = ws->SSAV;
    f32* E = ws->E;
    c64* work = ws->work;
    f32* rwork = ws->rwork;
    INT* iwork = ws->iwork;
    uint64_t* iseed = ws->rng_state;

    xlaenv(1, NBVAL[p->iparm]);
    xlaenv(2, NBMIN[p->iparm]);
    xlaenv(3, NXVAL[p->iparm]);

    const INT m = MVAL[p->jsize];
    const INT n = NVAL[p->jsize];
    const INT jtype = p->jtype;

    char jobq[2] = {0, 0};
    char jobu[2] = {0, 0};
    char jobvt[2] = {0, 0};
    char range[2] = {0, 0};
    INT i, iinfo, iju, ijq, ijvt, itemp, iwspc, iwtmp, j, lswork;
    INT nfail, ns, nsi, nsv, ntest;
    INT lrwork, liwork, numrank;
    INT il = 0, iu = 0;
    f32 dif, div;
    f32 vl = ZERO, vu = ZERO;
    f32 result[39];
    uint64_t iseed2[4];
    uint64_t ioldsd[4];
    INT nerrs = 0;
    INT ntestt = 0;

    /* More Important constants */
    const f32 unfl = slamch("S");
    const f32 ovfl = ONE / unfl;
    const f32 ulp = slamch("E");
    const f32 ulpinv = ONE / ulp;
    const f32 rtunfl = sqrtf(unfl);

    const INT mnmin = (m < n) ? m : n;
    const INT mxmn = (m > n) ? m : n;

    if (p->iparm != ws->ntestf_iparm) {
        ws->ntestf_iparm = p->iparm;
        ws->ntestf = 0;
    }

    ntest = 0;

    for (j = 0; j < 4; j++) {
        ioldsd[j] = iseed[j];
    }

    /* Compute "A" */

    if (jtype == 1) {

        /* Zero matrix */

        claset("Full", m, n, CZERO, CZERO, A, lda);
        for (i = 0; i < mnmin; i++) {
            S[i] = ZERO;
        }

    } else if (jtype == 2) {

        /* Identity matrix */

        claset("Full", m, n, CZERO, CONE, A, lda);
        for (i = 0; i < mnmin; i++) {
            S[i] = ONE;
        }

    } else {

        /* (Scaled) random matrix */

        if (jtype == 3)
            ws->anorm = ONE;
        if (jtype == 4)
            ws->anorm = unfl / ulp;
        if (jtype == 5)
            ws->anorm = ovfl * ulp;
        clatms(m, n, "U", "N", S, 4, (f32)mnmin, ws->anorm, m - 1, n - 1,
               "N", A, lda, work, &iinfo, iseed);
        if (iinfo != 0) {
            fprintf(stderr, " CDRVBD: Generator returned INFO=%6d.\n"
                            "         M=%6d, N=%6d, JTYPE=%6d, "
                            "ISEED=(%llu,%llu,%llu,%llu)\n",
                    (int)iinfo, (int)m, (int)n, (int)jtype,
                    (unsigned long long)ioldsd[0], (unsigned long long)ioldsd[1],
                    (unsigned long long)ioldsd[2], (unsigned long long)ioldsd[3]);
            fail_msg("CDRVBD: Generator returned INFO=%d M=%d N=%d JTYPE=%d",
                     (int)iinfo, (int)m, (int)n, (int)jtype);
            return;
        }
    }
    const f32 anorm = ws->anorm;

    clacpy("F", m, n, A, lda, ASAV, lda);

    /* Do for minimal and adequate (for blocking) workspace */

    for (iwspc = 1; iwspc <= 4; iwspc++) {

        /* Test for CGESVD */

        iwtmp = 2 * mnmin + mxmn;
        lswork = iwtmp + (iwspc - 1) * (lwork - iwtmp) / 3;
        if (lswork > lwork) lswork = lwork;
        if (lswork < 1) lswork = 1;
        if (iwspc == 4)
            lswork = lwork;

        for (j = 0; j < 35; j++) {
            result[j] = -ONE;
        }

        /* Factorize A */

        if (iwspc > 1)
            clacpy("F", m, n, ASAV, lda, A, lda);
        cgesvd("A", "A", m, n, A, lda, SSAV, USAV, ldu,
               VTSAV, ldvt, work, lswork, rwork, &iinfo);
        if (iinfo != 0) {
            info_fail("GESVD", iinfo, m, n, jtype, lswork, ioldsd);
            return;
        }

        /* Do tests 1--4 */

        cbdt01(m, n, 0, ASAV, lda, USAV, ldu, SSAV, E,
               VTSAV, ldvt, work, rwork, &result[0]);
        if (m != 0 && n != 0) {
            cunt01("Columns", mnmin, m, USAV, ldu, work,
                   lwork, rwork, &result[1]);
            cunt01("Rows", mnmin, n, VTSAV, ldvt, work,
                   lwork, rwork, &result[2]);
        }
        result[3] = 0;
        for (i = 0; i < mnmin - 1; i++) {
            if (SSAV[i] < SSAV[i + 1])
                result[3] = ulpinv;
            if (SSAV[i] < ZERO)
                result[3] = ulpinv;
        }
        if (mnmin >= 1) {
            if (SSAV[mnmin - 1] < ZERO)
                result[3] = ulpinv;
        }

        /* Do partial SVDs, comparing to SSAV, USAV, and VTSAV */

        result[4] = ZERO;
        result[5] = ZERO;
        result[6] = ZERO;
        for (iju = 0; iju <= 3; iju++) {
            for (ijvt = 0; ijvt <= 3; ijvt++) {
                if ((iju == 3 && ijvt == 3) ||
                    (iju == 1 && ijvt == 1)) continue;
                jobu[0] = CJOB[iju];
                jobvt[0] = CJOB[ijvt];
                clacpy("F", m, n, ASAV, lda, A, lda);
                cgesvd(jobu, jobvt, m, n, A, lda, S, U, ldu,
                       VT, ldvt, work, lswork, rwork, &iinfo);

                /* Compare U */

                dif = ZERO;
                if (m > 0 && n > 0) {
                    if (iju == 1) {
                        cunt03("C", m, mnmin, m, mnmin, USAV,
                               ldu, A, lda, work, lwork, rwork,
                               &dif, &iinfo);
                    } else if (iju == 2) {
                        cunt03("C", m, mnmin, m, mnmin, USAV,
                               ldu, U, ldu, work, lwork, rwork,
                               &dif, &iinfo);
                    } else if (iju == 3) {
                        cunt03("C", m, m, m, mnmin, USAV, ldu,
                               U, ldu, work, lwork, rwork, &dif,
                               &iinfo);
                    }
                }
                result[4] = fort_max(result[4], dif);

                /* Compare VT */

                dif = ZERO;
                if (m > 0 && n > 0) {
                    if (ijvt == 1) {
                        cunt03("R", n, mnmin, n, mnmin, VTSAV,
                               ldvt, A, lda, work, lwork,
                               rwork, &dif, &iinfo);
                    } else if (ijvt == 2) {
                        cunt03("R", n, mnmin, n, mnmin, VTSAV,
                               ldvt, VT, ldvt, work, lwork,
                               rwork, &dif, &iinfo);
                    } else if (ijvt == 3) {
                        cunt03("R", n, n, n, mnmin, VTSAV,
                               ldvt, VT, ldvt, work, lwork,
                               rwork, &dif, &iinfo);
                    }
                }
                result[5] = fort_max(result[5], dif);

                /* Compare S */

                dif = ZERO;
                div = fort_max((f32)mnmin * ulp * S[0], slamch("S"));
                for (i = 0; i < mnmin - 1; i++) {
                    if (SSAV[i] < SSAV[i + 1])
                        dif = ulpinv;
                    if (SSAV[i] < ZERO)
                        dif = ulpinv;
                    dif = fort_max(dif, fabsf(SSAV[i] - S[i]) / div);
                }
                result[6] = fort_max(result[6], dif);
            }
        }

        /* Test for CGESDD */

        iwtmp = 2 * mnmin * mnmin + 2 * mnmin + mxmn;
        lswork = iwtmp + (iwspc - 1) * (lwork - iwtmp) / 3;
        if (lswork > lwork) lswork = lwork;
        if (lswork < 1) lswork = 1;
        if (iwspc == 4)
            lswork = lwork;

        /* Factorize A */

        clacpy("F", m, n, ASAV, lda, A, lda);
        cgesdd("A", m, n, A, lda, SSAV, USAV, ldu, VTSAV,
               ldvt, work, lswork, rwork, iwork, &iinfo);
        if (iinfo != 0) {
            info_fail("GESDD", iinfo, m, n, jtype, lswork, ioldsd);
            return;
        }

        /* Do tests 1--4 */

        cbdt01(m, n, 0, ASAV, lda, USAV, ldu, SSAV, E,
               VTSAV, ldvt, work, rwork, &result[7]);
        if (m != 0 && n != 0) {
            cunt01("Columns", mnmin, m, USAV, ldu, work,
                   lwork, rwork, &result[8]);
            cunt01("Rows", mnmin, n, VTSAV, ldvt, work,
                   lwork, rwork, &result[9]);
        }
        result[10] = 0;
        for (i = 0; i < mnmin - 1; i++) {
            if (SSAV[i] < SSAV[i + 1])
                result[10] = ulpinv;
            if (SSAV[i] < ZERO)
                result[10] = ulpinv;
        }
        if (mnmin >= 1) {
            if (SSAV[mnmin - 1] < ZERO)
                result[10] = ulpinv;
        }

        /* Do partial SVDs, comparing to SSAV, USAV, and VTSAV */

        result[11] = ZERO;
        result[12] = ZERO;
        result[13] = ZERO;
        for (ijq = 0; ijq <= 2; ijq++) {
            jobq[0] = CJOB[ijq];
            clacpy("F", m, n, ASAV, lda, A, lda);
            cgesdd(jobq, m, n, A, lda, S, U, ldu, VT, ldvt,
                   work, lswork, rwork, iwork, &iinfo);

            /* Compare U */

            dif = ZERO;
            if (m > 0 && n > 0) {
                if (ijq == 1) {
                    if (m >= n) {
                        cunt03("C", m, mnmin, m, mnmin, USAV,
                               ldu, A, lda, work, lwork, rwork,
                               &dif, &iinfo);
                    } else {
                        cunt03("C", m, mnmin, m, mnmin, USAV,
                               ldu, U, ldu, work, lwork, rwork,
                               &dif, &iinfo);
                    }
                } else if (ijq == 2) {
                    cunt03("C", m, mnmin, m, mnmin, USAV, ldu,
                           U, ldu, work, lwork, rwork, &dif,
                           &iinfo);
                }
            }
            result[11] = fort_max(result[11], dif);

            /* Compare VT */

            dif = ZERO;
            if (m > 0 && n > 0) {
                if (ijq == 1) {
                    if (m >= n) {
                        cunt03("R", n, mnmin, n, mnmin, VTSAV,
                               ldvt, VT, ldvt, work, lwork,
                               rwork, &dif, &iinfo);
                    } else {
                        cunt03("R", n, mnmin, n, mnmin, VTSAV,
                               ldvt, A, lda, work, lwork,
                               rwork, &dif, &iinfo);
                    }
                } else if (ijq == 2) {
                    cunt03("R", n, mnmin, n, mnmin, VTSAV,
                           ldvt, VT, ldvt, work, lwork, rwork,
                           &dif, &iinfo);
                }
            }
            result[12] = fort_max(result[12], dif);

            /* Compare S */

            dif = ZERO;
            div = fort_max((f32)mnmin * ulp * S[0], slamch("S"));
            for (i = 0; i < mnmin - 1; i++) {
                if (SSAV[i] < SSAV[i + 1])
                    dif = ulpinv;
                if (SSAV[i] < ZERO)
                    dif = ulpinv;
                dif = fort_max(dif, fabsf(SSAV[i] - S[i]) / div);
            }
            result[13] = fort_max(result[13], dif);
        }

        /* Test CGESVDQ
           Note: CGESVDQ only works for M >= N */

        result[35] = ZERO;
        result[36] = ZERO;
        result[37] = ZERO;
        result[38] = ZERO;

        if (m >= n) {
            iwtmp = 2 * mnmin * mnmin + 2 * mnmin + mxmn;
            lswork = iwtmp + (iwspc - 1) * (lwork - iwtmp) / 3;
            if (lswork > lwork) lswork = lwork;
            if (lswork < 1) lswork = 1;
            if (iwspc == 4)
                lswork = lwork;

            clacpy("F", m, n, ASAV, lda, A, lda);

            lrwork = 2;
            if (m > lrwork) lrwork = m;
            if (5 * n > lrwork) lrwork = 5 * n;
            liwork = (n > 1) ? n : 1;
            cgesvdq("H", "N", "N", "A", "A",
                    m, n, A, lda, SSAV, USAV, ldu,
                    VTSAV, ldvt, &numrank, iwork, liwork,
                    work, lwork, rwork, lrwork, &iinfo);

            if (iinfo != 0) {
                info_fail("CGESVDQ", iinfo, m, n, jtype, lswork, ioldsd);
                return;
            }

            /* Do tests 36--39 */

            cbdt01(m, n, 0, ASAV, lda, USAV, ldu, SSAV, E,
                   VTSAV, ldvt, work, rwork, &result[35]);
            if (m != 0 && n != 0) {
                cunt01("Columns", m, m, USAV, ldu, work,
                       lwork, rwork, &result[36]);
                cunt01("Rows", n, n, VTSAV, ldvt, work,
                       lwork, rwork, &result[37]);
            }
            result[38] = ZERO;
            for (i = 0; i < mnmin - 1; i++) {
                if (SSAV[i] < SSAV[i + 1])
                    result[38] = ulpinv;
                if (SSAV[i] < ZERO)
                    result[38] = ulpinv;
            }
            if (mnmin >= 1) {
                if (SSAV[mnmin - 1] < ZERO)
                    result[38] = ulpinv;
            }
        }

        /* Test CGESVJ
           Note: CGESVJ only works for M >= N */

        result[14] = ZERO;
        result[15] = ZERO;
        result[16] = ZERO;
        result[17] = ZERO;

        if (m >= n) {
            iwtmp = 2 * mnmin * mnmin + 2 * mnmin + mxmn;
            lswork = iwtmp + (iwspc - 1) * (lwork - iwtmp) / 3;
            if (lswork > lwork) lswork = lwork;
            if (lswork < 1) lswork = 1;
            lrwork = (n > 6) ? n : 6;
            if (iwspc == 4)
                lswork = lwork;

            clacpy("F", m, n, ASAV, lda, USAV, lda);
            cgesvj("G", "U", "V", m, n, USAV, lda, SSAV,
                   0, A, ldvt, work, lwork, rwork,
                   lrwork, &iinfo);

            /* CGESVJ returns V not VH */

            for (j = 0; j < n; j++) {
                for (i = 0; i < n; i++) {
                    VTSAV[j + i * ldvt] = conjf(A[i + j * lda]);
                }
            }

            if (iinfo != 0) {
                info_fail("GESVJ", iinfo, m, n, jtype, lswork, ioldsd);
                return;
            }

            /* Do tests 15--18 */

            cbdt01(m, n, 0, ASAV, lda, USAV, ldu, SSAV, E,
                   VTSAV, ldvt, work, rwork, &result[14]);
            if (m != 0 && n != 0) {
                cunt01("Columns", m, m, USAV, ldu, work,
                       lwork, rwork, &result[15]);
                cunt01("Rows", n, n, VTSAV, ldvt, work,
                       lwork, rwork, &result[16]);
            }
            result[17] = ZERO;
            for (i = 0; i < mnmin - 1; i++) {
                if (SSAV[i] < SSAV[i + 1])
                    result[17] = ulpinv;
                if (SSAV[i] < ZERO)
                    result[17] = ulpinv;
            }
            if (mnmin >= 1) {
                if (SSAV[mnmin - 1] < ZERO)
                    result[17] = ulpinv;
            }
        }

        /* Test CGEJSV
           Note: CGEJSV only works for M >= N */

        result[18] = ZERO;
        result[19] = ZERO;
        result[20] = ZERO;
        result[21] = ZERO;
        if (m >= n) {
            iwtmp = 2 * mnmin * mnmin + 2 * mnmin + mxmn;
            lswork = iwtmp + (iwspc - 1) * (lwork - iwtmp) / 3;
            if (lswork > lwork) lswork = lwork;
            if (lswork < 1) lswork = 1;
            if (iwspc == 4)
                lswork = lwork;
            lrwork = (n + 2 * m > 7) ? n + 2 * m : 7;

            clacpy("F", m, n, ASAV, lda, VTSAV, lda);
            cgejsv("G", "U", "V", "R", "N", "N",
                   m, n, VTSAV, lda, SSAV, USAV, ldu, A, ldvt,
                   work, lwork, rwork,
                   lrwork, iwork, &iinfo);

            /* CGEJSV returns V not VH */

            for (j = 0; j < n; j++) {
                for (i = 0; i < n; i++) {
                    VTSAV[j + i * ldvt] = conjf(A[i + j * lda]);
                }
            }

            if (iinfo != 0) {
                info_fail("GEJSV", iinfo, m, n, jtype, lswork, ioldsd);
                return;
            }

            /* Do tests 19--22 */

            cbdt01(m, n, 0, ASAV, lda, USAV, ldu, SSAV, E,
                   VTSAV, ldvt, work, rwork, &result[18]);
            if (m != 0 && n != 0) {
                cunt01("Columns", m, m, USAV, ldu, work,
                       lwork, rwork, &result[19]);
                cunt01("Rows", n, n, VTSAV, ldvt, work,
                       lwork, rwork, &result[20]);
            }
            result[21] = ZERO;
            for (i = 0; i < mnmin - 1; i++) {
                if (SSAV[i] < SSAV[i + 1])
                    result[21] = ulpinv;
                if (SSAV[i] < ZERO)
                    result[21] = ulpinv;
            }
            if (mnmin >= 1) {
                if (SSAV[mnmin - 1] < ZERO)
                    result[21] = ulpinv;
            }
        }

        /* Test CGESVDX */

        /* Factorize A */

        clacpy("F", m, n, ASAV, lda, A, lda);
        cgesvdx("V", "V", "A", m, n, A, lda,
                vl, vu, il, iu, &ns, SSAV, USAV, ldu,
                VTSAV, ldvt, work, lwork, rwork,
                iwork, &iinfo);
        if (iinfo != 0) {
            info_fail("GESVDX", iinfo, m, n, jtype, lswork, ioldsd);
            return;
        }

        /* Do tests 1--4 */

        result[22] = ZERO;
        result[23] = ZERO;
        result[24] = ZERO;
        cbdt01(m, n, 0, ASAV, lda, USAV, ldu, SSAV, E,
               VTSAV, ldvt, work, rwork, &result[22]);
        if (m != 0 && n != 0) {
            cunt01("Columns", mnmin, m, USAV, ldu, work,
                   lwork, rwork, &result[23]);
            cunt01("Rows", mnmin, n, VTSAV, ldvt, work,
                   lwork, rwork, &result[24]);
        }
        result[25] = ZERO;
        for (i = 0; i < mnmin - 1; i++) {
            if (SSAV[i] < SSAV[i + 1])
                result[25] = ulpinv;
            if (SSAV[i] < ZERO)
                result[25] = ulpinv;
        }
        if (mnmin >= 1) {
            if (SSAV[mnmin - 1] < ZERO)
                result[25] = ulpinv;
        }

        /* Do partial SVDs, comparing to SSAV, USAV, and VTSAV */

        result[26] = ZERO;
        result[27] = ZERO;
        result[28] = ZERO;
        for (iju = 0; iju <= 1; iju++) {
            for (ijvt = 0; ijvt <= 1; ijvt++) {
                if ((iju == 0 && ijvt == 0) ||
                    (iju == 1 && ijvt == 1)) continue;
                jobu[0] = CJOBV[iju];
                jobvt[0] = CJOBV[ijvt];
                range[0] = CJOBR[0];
                clacpy("F", m, n, ASAV, lda, A, lda);
                cgesvdx(jobu, jobvt, "A", m, n, A, lda,
                        vl, vu, il, iu, &ns, SSAV, U, ldu,
                        VT, ldvt, work, lwork, rwork,
                        iwork, &iinfo);

                /* Compare U */

                dif = ZERO;
                if (m > 0 && n > 0) {
                    if (iju == 1) {
                        cunt03("C", m, mnmin, m, mnmin, USAV,
                               ldu, U, ldu, work, lwork, rwork,
                               &dif, &iinfo);
                    }
                }
                result[26] = fort_max(result[26], dif);

                /* Compare VT */

                dif = ZERO;
                if (m > 0 && n > 0) {
                    if (ijvt == 1) {
                        cunt03("R", n, mnmin, n, mnmin, VTSAV,
                               ldvt, VT, ldvt, work, lwork,
                               rwork, &dif, &iinfo);
                    }
                }
                result[27] = fort_max(result[27], dif);

                /* Compare S */

                dif = ZERO;
                div = fort_max((f32)mnmin * ulp * S[0], slamch("S"));
                for (i = 0; i < mnmin - 1; i++) {
                    if (SSAV[i] < SSAV[i + 1])
                        dif = ulpinv;
                    if (SSAV[i] < ZERO)
                        dif = ulpinv;
                    dif = fort_max(dif, fabsf(SSAV[i] - S[i]) / div);
                }
                result[28] = fort_max(result[28], dif);
            }
        }
        (void)range;

        /* Do tests 8--10 */

        for (i = 0; i < 4; i++) {
            iseed2[i] = iseed[i];
        }
        if (mnmin <= 1) {
            il = 0;
            iu = ((mnmin > 1) ? mnmin : 1) - 1;
        } else {
            il = (INT)(mnmin * rng_uniform_f32(iseed2));
            iu = (INT)(mnmin * rng_uniform_f32(iseed2));
            if (iu < il) {
                itemp = iu;
                iu = il;
                il = itemp;
            }
        }
        clacpy("F", m, n, ASAV, lda, A, lda);
        cgesvdx("V", "V", "I", m, n, A, lda,
                vl, vu, il, iu, &nsi, S, U, ldu,
                VT, ldvt, work, lwork, rwork,
                iwork, &iinfo);
        if (iinfo != 0) {
            info_fail("GESVDX", iinfo, m, n, jtype, lswork, ioldsd);
            return;
        }

        result[29] = ZERO;
        result[30] = ZERO;
        result[31] = ZERO;
        cbdt05(m, n, ASAV, lda, S, nsi, U, ldu,
               VT, ldvt, work, &result[29]);
        if (m != 0 && n != 0) {
            cunt01("Columns", m, nsi, U, ldu, work,
                   lwork, rwork, &result[30]);
            cunt01("Rows", nsi, n, VT, ldvt, work,
                   lwork, rwork, &result[31]);
        }

        /* Do tests 11--13 */

        if (mnmin > 0 && nsi > 1) {
            if (il != 0) {
                vu = SSAV[il] +
                     fort_max(fort_max(HALF * fabsf(SSAV[il] - SSAV[il - 1]),
                                       ulp * anorm), TWO * rtunfl);
            } else {
                vu = SSAV[0] +
                     fort_max(fort_max(HALF * fabsf(SSAV[ns - 1] - SSAV[0]),
                                       ulp * anorm), TWO * rtunfl);
            }
            if (iu != ns - 1) {
                vl = SSAV[iu] - fort_max(fort_max(ulp * anorm, TWO * rtunfl),
                                         HALF * fabsf(SSAV[iu + 1] - SSAV[iu]));
            } else {
                vl = SSAV[ns - 1] - fort_max(fort_max(ulp * anorm, TWO * rtunfl),
                                             HALF * fabsf(SSAV[ns - 1] - SSAV[0]));
            }
            vl = fort_max(vl, ZERO);
            vu = fort_max(vu, ZERO);
            if (vl >= vu) vu = fort_max(vu * 2, vu + vl + HALF);
        } else {
            vl = ZERO;
            vu = ONE;
        }
        clacpy("F", m, n, ASAV, lda, A, lda);
        cgesvdx("V", "V", "V", m, n, A, lda,
                vl, vu, il, iu, &nsv, S, U, ldu,
                VT, ldvt, work, lwork, rwork,
                iwork, &iinfo);
        if (iinfo != 0) {
            info_fail("GESVDX", iinfo, m, n, jtype, lswork, ioldsd);
            return;
        }

        result[32] = ZERO;
        result[33] = ZERO;
        result[34] = ZERO;
        cbdt05(m, n, ASAV, lda, S, nsv, U, ldu,
               VT, ldvt, work, &result[32]);
        if (m != 0 && n != 0) {
            cunt01("Columns", m, nsv, U, ldu, work,
                   lwork, rwork, &result[33]);
            cunt01("Rows", nsv, n, VT, ldvt, work,
                   lwork, rwork, &result[34]);
        }

        /* End of Loop -- Check for RESULT(j) > THRESH */

        ntest = 0;
        nfail = 0;
        for (j = 0; j < 39; j++) {
            if (result[j] >= ZERO)
                ntest = ntest + 1;
            if (result[j] >= THRESH)
                nfail = nfail + 1;
        }

        if (nfail > 0)
            ws->ntestf = ws->ntestf + 1;
        if (ws->ntestf == 1) {
            print_header();
            ws->ntestf = 2;
        }

        for (j = 0; j < 39; j++) {
            if (result[j] >= THRESH) {
                fprintf(stderr, " M=%5d, N=%5d, type %1d, IWS=%1d, "
                                "seed=%llu,%llu,%llu,%llu, test(%2d)=%11.4g\n",
                        (int)m, (int)n, (int)jtype, (int)iwspc,
                        (unsigned long long)ioldsd[0], (unsigned long long)ioldsd[1],
                        (unsigned long long)ioldsd[2], (unsigned long long)ioldsd[3],
                        (int)(j + 1), (double)result[j]);
            }
        }

        nerrs = nerrs + nfail;
        ntestt = ntestt + ntest;

    }

    if (nerrs > 0) {
        fail_msg("CDRVBD M=%d N=%d type=%d: %d of %d test ratios >= THRESH",
                 (int)m, (int)n, (int)jtype, (int)nerrs, (int)ntestt);
    }
}

static void test_cerred(void** state)
{
    (void)state;
    INT ok, nt;
    cerred("CBD", &ok, &nt);
    assert_true(ok);
}

static void test_cdrvbd_case(void** state)
{
    const cdrvbd_params_t* params = *state;
    run_cdrvbd_single(params);
}

#define MAX_TESTS (1 + NPARMS * NSIZES * MAXTYP)

static cdrvbd_params_t g_params[MAX_TESTS];
static struct CMUnitTest g_tests[MAX_TESTS];
static INT g_num_tests = 0;

static void build_test_array(void)
{
    g_num_tests = 0;

    g_tests[g_num_tests].name = "cerred";
    g_tests[g_num_tests].test_func = test_cerred;
    g_tests[g_num_tests].initial_state = NULL;
    g_tests[g_num_tests].setup_func = NULL;
    g_tests[g_num_tests].teardown_func = NULL;
    g_num_tests++;

    for (size_t ip = 0; ip < NPARMS; ip++) {
        for (INT jsize = 0; jsize < NSIZES; jsize++) {
            for (INT jtype = 1; jtype <= MAXTYP; jtype++) {
                cdrvbd_params_t* p = &g_params[g_num_tests];
                p->jsize = jsize;
                p->jtype = jtype;
                p->iparm = (INT)ip;
                snprintf(p->name, sizeof(p->name),
                         "cdrvbd_p%d_m%d_n%d_type%d", (int)ip,
                         (int)MVAL[jsize], (int)NVAL[jsize], (int)jtype);

                g_tests[g_num_tests].name = p->name;
                g_tests[g_num_tests].test_func = test_cdrvbd_case;
                g_tests[g_num_tests].setup_func = NULL;
                g_tests[g_num_tests].teardown_func = NULL;
                g_tests[g_num_tests].initial_state = p;

                g_num_tests++;
            }
        }
    }
}

int main(void)
{
    build_test_array();

    (void)_cmocka_run_group_tests("cdrvbd", g_tests, g_num_tests,
                                   group_setup, group_teardown);
    return 0;
}
