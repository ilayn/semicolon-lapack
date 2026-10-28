/**
 * @file test_sdrvbd.c
 * @brief SDRVBD checks the singular value decomposition (SVD) drivers
 *        SGESVD, SGESDD, SGESVDQ, SGESVJ, SGEJSV, and SGESVDX.
 *
 * Port of LAPACK TESTING/EIG/sdrvbd.f, with the parameters SCHKEE passes
 * to it for the SVD path and the values of TESTING/svd.in.
 *
 * Both SGESVD and SGESDD factor A = U diag(S) VT, where U and VT are
 * orthogonal and diag(S) is diagonal with the entries of the array S
 * on its diagonal. The entries of S are the singular values,
 * nonnegative and stored in decreasing order.  U and VT can be
 * optionally not computed, overwritten on A, or computed partially.
 *
 * A is M by N. Let MNMIN = min( M, N ). S has dimension MNMIN.
 * U can be M by M or M by MNMIN. VT can be N by N or MNMIN by N.
 *
 * When SDRVBD is called, a number of matrix "sizes" (M's and N's)
 * and a number of matrix "types" are specified.  For each size (M,N)
 * and each type of matrix, and for the minimal workspace as well as
 * workspace adequate to permit blocking, an  M x N  matrix "A" will be
 * generated and used to test the SVD routines.  For each matrix, A will
 * be factored as A = U diag(S) VT and the following 12 tests computed:
 *
 * Test for SGESVD:
 *
 * (1)    | A - U diag(S) VT | / ( |A| max(M,N) ulp )
 * (2)    | I - U'U | / ( M ulp )
 * (3)    | I - VT VT' | / ( N ulp )
 * (4)    S contains MNMIN nonnegative values in decreasing order.
 *        (Return 0 if true, 1/ULP if false.)
 * (5)    | U - Upartial | / ( M ulp ) where Upartial is a partially
 *        computed U.
 * (6)    | VT - VTpartial | / ( N ulp ) where VTpartial is a partially
 *        computed VT.
 * (7)    | S - Spartial | / ( MNMIN ulp |S| ) where Spartial is the
 *        vector of singular values from the partial SVD
 *
 * Test for SGESDD:
 *
 * (8)    | A - U diag(S) VT | / ( |A| max(M,N) ulp )
 * (9)    | I - U'U | / ( M ulp )
 * (10)   | I - VT VT' | / ( N ulp )
 * (11)   S contains MNMIN nonnegative values in decreasing order.
 *        (Return 0 if true, 1/ULP if false.)
 * (12)   | U - Upartial | / ( M ulp ) where Upartial is a partially
 *        computed U.
 * (13)   | VT - VTpartial | / ( N ulp ) where VTpartial is a partially
 *        computed VT.
 * (14)   | S - Spartial | / ( MNMIN ulp |S| ) where Spartial is the
 *        vector of singular values from the partial SVD
 *
 * Test for SGESVDQ:
 *
 * (36)   | A - U diag(S) VT | / ( |A| max(M,N) ulp )
 * (37)   | I - U'U | / ( M ulp )
 * (38)   | I - VT VT' | / ( N ulp )
 * (39)   S contains MNMIN nonnegative values in decreasing order.
 *        (Return 0 if true, 1/ULP if false.)
 *
 * Test for SGESVJ:
 *
 * (15)   | A - U diag(S) VT | / ( |A| max(M,N) ulp )
 * (16)   | I - U'U | / ( M ulp )
 * (17)   | I - VT VT' | / ( N ulp )
 * (18)   S contains MNMIN nonnegative values in decreasing order.
 *        (Return 0 if true, 1/ULP if false.)
 *
 * Test for SGEJSV:
 *
 * (19)   | A - U diag(S) VT | / ( |A| max(M,N) ulp )
 * (20)   | I - U'U | / ( M ulp )
 * (21)   | I - VT VT' | / ( N ulp )
 * (22)   S contains MNMIN nonnegative values in decreasing order.
 *        (Return 0 if true, 1/ULP if false.)
 *
 * Test for SGESVDX( 'V', 'V', 'A' )/SGESVDX( 'N', 'N', 'A' )
 *
 * (23)   | A - U diag(S) VT | / ( |A| max(M,N) ulp )
 * (24)   | I - U'U | / ( M ulp )
 * (25)   | I - VT VT' | / ( N ulp )
 * (26)   S contains MNMIN nonnegative values in decreasing order.
 *        (Return 0 if true, 1/ULP if false.)
 * (27)   | U - Upartial | / ( M ulp ) where Upartial is a partially
 *        computed U.
 * (28)   | VT - VTpartial | / ( N ulp ) where VTpartial is a partially
 *        computed VT.
 * (29)   | S - Spartial | / ( MNMIN ulp |S| ) where Spartial is the
 *        vector of singular values from the partial SVD
 *
 * Test for SGESVDX( 'V', 'V', 'I' )
 *
 * (30)   | U' A VT''' - diag(S) | / ( |A| max(M,N) ulp )
 * (31)   | I - U'U | / ( M ulp )
 * (32)   | I - VT VT' | / ( N ulp )
 *
 * Test for SGESVDX( 'V', 'V', 'V' )
 *
 * (33)   | U' A VT''' - diag(S) | / ( |A| max(M,N) ulp )
 * (34)   | I - U'U | / ( M ulp )
 * (35)   | I - VT VT' | / ( N ulp )
 *
 * Currently, the list of possible types is:
 *
 * (1)  The zero matrix.
 * (2)  The identity matrix.
 * (3)  A matrix of the form  U D V, where U and V are orthogonal and
 *      D has evenly spaced entries 1, ..., ULP with random signs
 *      on the diagonal.
 * (4)  Same as (3), but multiplied by the underflow-threshold / ULP.
 * (5)  Same as (3), but multiplied by the overflow-threshold * ULP.
 */

#include "test_harness.h"
#include "test_rng.h"
#include "verify.h"

/* svd.in */
#define THRESH 50.0f

/* sdrvbd.f */
#define MAXTYP 5

/* schkee.F: MAXTYP = 16, NTYPES = MIN( MAXTYP, NTYPES ) with svd.in 'SVD 16' */
#define NTYPES 16

/* sdrvbd.f: MTYPES = MIN( MAXTYP, NTYPES ) since NSIZES != 1 */
#define MTYPES ((MAXTYP < NTYPES) ? MAXTYP : NTYPES)

/* schkee.F: NMAX, LWORK = NMAX*(5*NMAX+5)+1, LIWORK = NMAX*(5*NMAX+20) */
#define NMAX   132
#define LWORK  (NMAX * (5 * NMAX + 5) + 1)
#define LIWORK (NMAX * (5 * NMAX + 20))

/* svd.in: values of M and N */
static const INT MVAL[] = {0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 3, 10, 10, 16, 16, 30, 30, 40, 40};
static const INT NVAL[] = {0, 1, 3, 0, 1, 2, 0, 1, 0, 1, 3, 10, 16, 10, 16, 30, 40, 30, 40};
#define NSIZES ((INT)(sizeof(MVAL) / sizeof(MVAL[0])))

/* svd.in: NB, NBMIN, NX */
static const INT NBVAL[] = {1, 3, 3, 3, 20};
static const INT NBMIN[] = {2, 2, 2, 2, 2};
static const INT NXVAL[] = {1, 0, 5, 9, 1};
#define NPARMS ((INT)(sizeof(NBVAL) / sizeof(NBVAL[0])))

static const char CJOB[4]  = {'N', 'O', 'S', 'A'};
static const char CJOBR[3] = {'A', 'V', 'I'};
static const char CJOBV[2] = {'N', 'V'};

typedef struct {
    INT jsize;
    INT jtype;
    INT iparm;
    char name[64];
} sdrvbd_params_t;

typedef struct {
    f32* A;
    f32* U;
    f32* VT;
    f32* ASAV;
    f32* USAV;
    f32* VTSAV;
    f32* S;
    f32* SSAV;
    f32* E;
    f32* work;
    INT* iwork;
    uint64_t iseed[4];
    f32 anorm;
    INT nfail;
    INT nfail_iparm;
} sdrvbd_workspace_t;

static sdrvbd_workspace_t* g_ws = NULL;

static int group_setup(void** state)
{
    (void)state;

    /* Check for errors */
    INT lda = NMAX;
    INT ldu = NMAX;
    INT ldvt = NMAX;
    INT info = 0;
    INT badmm = 0;
    INT badnn = 0;
    INT mmax = 1;
    INT nmax = 1;
    INT minwrk = 1;
    for (INT j = 0; j < NSIZES; j++) {
        mmax = (mmax > MVAL[j]) ? mmax : MVAL[j];
        if (MVAL[j] < 0)
            badmm = 1;
        nmax = (nmax > NVAL[j]) ? nmax : NVAL[j];
        if (NVAL[j] < 0)
            badnn = 1;
        INT mn = (MVAL[j] < NVAL[j]) ? MVAL[j] : NVAL[j];
        INT mx = (MVAL[j] > NVAL[j]) ? MVAL[j] : NVAL[j];
        INT t1 = 3 * mn + mx;
        INT t2 = 5 * ((MVAL[j] < NVAL[j] - 4) ? MVAL[j] : NVAL[j] - 4);
        INT t = ((t1 > t2) ? t1 : t2) + 2 * mn * mn;
        minwrk = (minwrk > t) ? minwrk : t;
    }

    /* Check for errors */
    if (NSIZES < 0) {
        info = -1;
    } else if (badmm) {
        info = -2;
    } else if (badnn) {
        info = -3;
    } else if (NTYPES < 0) {
        info = -4;
    } else if (lda < ((1 > mmax) ? 1 : mmax)) {
        info = -10;
    } else if (ldu < ((1 > mmax) ? 1 : mmax)) {
        info = -12;
    } else if (ldvt < ((1 > nmax) ? 1 : nmax)) {
        info = -14;
    } else if (minwrk > LWORK) {
        info = -21;
    }

    if (info != 0) {
        fprintf(stderr, "SDRVBD: parameter number %d had an illegal value\n",
                (int)(-info));
        return -1;
    }

    g_ws = calloc(1, sizeof(sdrvbd_workspace_t));
    if (!g_ws) return -1;

    g_ws->A     = calloc((size_t)NMAX * NMAX, sizeof(f32));
    g_ws->U     = calloc((size_t)NMAX * NMAX, sizeof(f32));
    g_ws->VT    = calloc((size_t)NMAX * NMAX, sizeof(f32));
    g_ws->ASAV  = calloc((size_t)NMAX * NMAX, sizeof(f32));
    g_ws->USAV  = calloc((size_t)NMAX * NMAX, sizeof(f32));
    g_ws->VTSAV = calloc((size_t)NMAX * NMAX, sizeof(f32));
    g_ws->S     = calloc(NMAX, sizeof(f32));
    g_ws->SSAV  = calloc(NMAX, sizeof(f32));
    g_ws->E     = calloc(NMAX, sizeof(f32));
    g_ws->work  = calloc(LWORK, sizeof(f32));
    g_ws->iwork = calloc(LIWORK, sizeof(INT));

    if (!g_ws->A || !g_ws->U || !g_ws->VT || !g_ws->ASAV || !g_ws->USAV ||
        !g_ws->VTSAV || !g_ws->S || !g_ws->SSAV || !g_ws->E ||
        !g_ws->work || !g_ws->iwork) {
        return -1;
    }

    g_ws->anorm = 0.0f;
    g_ws->nfail = 0;
    g_ws->nfail_iparm = -1;
    rng_seed(g_ws->iseed, 2024);

    xlaenv(1, 1);
    xlaenv(9, 25);

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
        free(g_ws->iwork);
        free(g_ws);
        g_ws = NULL;
    }
    return 0;
}

static void print_header(void)
{
    fprintf(stderr,
        " SVD -- Real Singular Value Decomposition Driver \n"
        " Matrix types (see SDRVBD for details):\n"
        "\n"
        " 1 = Zero matrix\n"
        " 2 = Identity matrix\n"
        " 3 = Evenly spaced singular values near 1\n"
        " 4 = Evenly spaced singular values near underflow\n"
        " 5 = Evenly spaced singular values near overflow\n"
        "\n"
        " Tests performed: ( A is dense, U and V are orthogonal,\n"
        "                    S is an array, and Upartial, VTpartial, and\n"
        "                    Spartial are partially computed U, VT and S),\n"
        "\n");
    fprintf(stderr,
        " 1 = | A - U diag(S) VT | / ( |A| max(M,N) ulp ) \n"
        " 2 = | I - U**T U | / ( M ulp ) \n"
        " 3 = | I - VT VT**T | / ( N ulp ) \n"
        " 4 = 0 if S contains min(M,N) nonnegative values in"
        " decreasing order, else 1/ulp\n"
        " 5 = | U - Upartial | / ( M ulp )\n"
        " 6 = | VT - VTpartial | / ( N ulp )\n"
        " 7 = | S - Spartial | / ( min(M,N) ulp |S| )\n"
        " 8 = | A - U diag(S) VT | / ( |A| max(M,N) ulp ) \n"
        " 9 = | I - U**T U | / ( M ulp ) \n"
        "10 = | I - VT VT**T | / ( N ulp ) \n"
        "11 = 0 if S contains min(M,N) nonnegative values in"
        " decreasing order, else 1/ulp\n"
        "12 = | U - Upartial | / ( M ulp )\n"
        "13 = | VT - VTpartial | / ( N ulp )\n"
        "14 = | S - Spartial | / ( min(M,N) ulp |S| )\n"
        "15 = | A - U diag(S) VT | / ( |A| max(M,N) ulp ) \n"
        "16 = | I - U**T U | / ( M ulp ) \n"
        "17 = | I - VT VT**T | / ( N ulp ) \n"
        "18 = 0 if S contains min(M,N) nonnegative values in"
        " decreasing order, else 1/ulp\n"
        "19 = | U - Upartial | / ( M ulp )\n"
        "20 = | VT - VTpartial | / ( N ulp )\n"
        "21 = | S - Spartial | / ( min(M,N) ulp |S| )\n"
        "22 = 0 if S contains min(M,N) nonnegative values in"
        " decreasing order, else 1/ulp"
        " SGESVDX(V,V,A) \n"
        "23 = | A - U diag(S) VT | / ( |A| max(M,N) ulp ),\n"
        "24 = | I - U**T U | / ( M ulp ) \n"
        "25 = | I - VT VT**T | / ( N ulp ) \n"
        "26 = 0 if S contains min(M,N) nonnegative values in"
        " decreasing order, else 1/ulp\n"
        "27 = | U - Upartial | / ( M ulp )\n"
        "28 = | VT - VTpartial | / ( N ulp )\n"
        "29 = | S - Spartial | / ( min(M,N) ulp |S| )\n"
        "30 = | U**T A VT**T - diag(S) | / ( |A| max(M,N) ulp ),"
        " SGESVDX(V,V,I) \n"
        "31 = | I - U**T U | / ( M ulp ) \n"
        "32 = | I - VT VT**T | / ( N ulp ) \n"
        "33 = | U**T A VT**T - diag(S) | / ( |A| max(M,N) ulp ),"
        " SGESVDX(V,V,V) \n"
        "34 = | I - U**T U | / ( M ulp ) \n"
        "35 = | I - VT VT**T | / ( N ulp ) "
        " SGESVDQ(H,N,N,A,A\n"
        "36 = | A - U diag(S) VT | / ( |A| max(M,N) ulp ) \n"
        "37 = | I - U**T U | / ( M ulp ) \n"
        "38 = | I - VT VT**T | / ( N ulp ) \n"
        "39 = 0 if S contains min(M,N) nonnegative values in"
        " decreasing order, else 1/ulp\n"
        "\n"
        "\n");
}

static void run_sdrvbd_single(const sdrvbd_params_t* params)
{
    const INT lda = NMAX;
    const INT ldu = NMAX;
    const INT ldvt = NMAX;
    const INT lwork = LWORK;

    f32* A     = g_ws->A;
    f32* U     = g_ws->U;
    f32* VT    = g_ws->VT;
    f32* ASAV  = g_ws->ASAV;
    f32* USAV  = g_ws->USAV;
    f32* VTSAV = g_ws->VTSAV;
    f32* S     = g_ws->S;
    f32* SSAV  = g_ws->SSAV;
    f32* E     = g_ws->E;
    f32* work  = g_ws->work;
    INT* iwork = g_ws->iwork;
    uint64_t* iseed = g_ws->iseed;

    char jobq[2] = {0, 0};
    char jobu[2] = {0, 0};
    char jobvt[2] = {0, 0};
    char range[2] = {0, 0};
    INT info = 0;
    INT iinfo = 0;
    INT il = 0, iu = 0, ns = 0, nsi = 0, nsv = 0;
    INT iwtmp, lswork, itemp;
    f32 dif, div, vl = 0.0f, vu = 0.0f, t;
    INT liwork, lrwork, numrank;
    f32 rwork[2];
    uint64_t iseed2[4];
    uint64_t ioldsd[4];
    f32 result[39];
    char ctx[256];

    xlaenv(1, NBVAL[params->iparm]);
    xlaenv(2, NBMIN[params->iparm]);
    xlaenv(3, NXVAL[params->iparm]);

    /* Initialize constants */
    const f32 unfl = slamch("Safe minimum");
    const f32 ovfl = 1.0f / unfl;
    const f32 ulp = slamch("Precision");
    const f32 rtunfl = sqrtf(unfl);
    const f32 ulpinv = 1.0f / ulp;

    /* Loop over sizes, types */
    const INT m = MVAL[params->jsize];
    const INT n = NVAL[params->jsize];
    const INT mnmin = (m < n) ? m : n;
    const INT jtype = params->jtype;

    if (params->iparm != g_ws->nfail_iparm) {
        g_ws->nfail_iparm = params->iparm;
        g_ws->nfail = 0;
    }

    for (INT j = 0; j < 4; j++) {
        ioldsd[j] = iseed[j];
    }

    /* Compute "A" */
    if (jtype == 1) {

        /* Zero matrix */
        slaset("Full", m, n, 0.0f, 0.0f, A, lda);

    } else if (jtype == 2) {

        /* Identity matrix */
        slaset("Full", m, n, 0.0f, 1.0f, A, lda);

    } else {

        /* (Scaled) random matrix */
        if (jtype == 3)
            g_ws->anorm = 1.0f;
        if (jtype == 4)
            g_ws->anorm = unfl / ulp;
        if (jtype == 5)
            g_ws->anorm = ovfl * ulp;
        slatms(m, n, "U", "N", S, 4, (f32)mnmin, g_ws->anorm, m - 1, n - 1,
               "N", A, lda, work, &iinfo, iseed);
        if (iinfo != 0) {
            snprintf(ctx, sizeof(ctx),
                     "SDRVBD: Generator returned INFO=%d M=%d N=%d JTYPE=%d"
                     " ISEED=(%llu,%llu,%llu,%llu)",
                     (int)iinfo, (int)m, (int)n, (int)jtype,
                     (unsigned long long)ioldsd[0], (unsigned long long)ioldsd[1],
                     (unsigned long long)ioldsd[2], (unsigned long long)ioldsd[3]);
            set_test_context(ctx);
            assert_info_success(iinfo);
            clear_test_context();
            return;
        }
    }
    const f32 anorm = g_ws->anorm;

    slacpy("F", m, n, A, lda, ASAV, lda);

    /* Do for minimal and adequate (for blocking) workspace */
    for (INT iws = 1; iws <= 4; iws++) {

        for (INT j = 0; j < 32; j++) {
            result[j] = -1.0f;
        }

        /* Test SGESVD: Factorize A */
        iwtmp = (3 * mnmin + ((m > n) ? m : n) > 5 * mnmin)
                    ? 3 * mnmin + ((m > n) ? m : n) : 5 * mnmin;
        lswork = iwtmp + (iws - 1) * (lwork - iwtmp) / 3;
        lswork = (lswork < lwork) ? lswork : lwork;
        lswork = (lswork > 1) ? lswork : 1;
        if (iws == 4)
            lswork = lwork;

        if (iws > 1)
            slacpy("F", m, n, ASAV, lda, A, lda);
        sgesvd("A", "A", m, n, A, lda, SSAV, USAV, ldu,
               VTSAV, ldvt, work, lswork, &iinfo);
        if (iinfo != 0) {
            snprintf(ctx, sizeof(ctx),
                     "SDRVBD: GESVD returned INFO=%d M=%d N=%d JTYPE=%d LSWORK=%d"
                     " ISEED=(%llu,%llu,%llu,%llu)",
                     (int)iinfo, (int)m, (int)n, (int)jtype, (int)lswork,
                     (unsigned long long)ioldsd[0], (unsigned long long)ioldsd[1],
                     (unsigned long long)ioldsd[2], (unsigned long long)ioldsd[3]);
            set_test_context(ctx);
            assert_info_success(iinfo);
            clear_test_context();
            return;
        }

        /* Do tests 1--4 */
        sbdt01(m, n, 0, ASAV, lda, USAV, ldu, SSAV, E,
               VTSAV, ldvt, work, &result[0]);
        if (m != 0 && n != 0) {
            sort01("Columns", m, m, USAV, ldu, work, lwork, &result[1]);
            sort01("Rows", n, n, VTSAV, ldvt, work, lwork, &result[2]);
        }
        result[3] = 0.0f;
        for (INT i = 0; i < mnmin - 1; i++) {
            if (SSAV[i] < SSAV[i + 1])
                result[3] = ulpinv;
            if (SSAV[i] < 0.0f)
                result[3] = ulpinv;
        }
        if (mnmin >= 1) {
            if (SSAV[mnmin - 1] < 0.0f)
                result[3] = ulpinv;
        }

        /* Do partial SVDs, comparing to SSAV, USAV, and VTSAV */
        result[4] = 0.0f;
        result[5] = 0.0f;
        result[6] = 0.0f;
        for (INT iju = 0; iju <= 3; iju++) {
            for (INT ijvt = 0; ijvt <= 3; ijvt++) {
                if ((iju == 3 && ijvt == 3) ||
                    (iju == 1 && ijvt == 1))
                    continue;
                jobu[0] = CJOB[iju];
                jobvt[0] = CJOB[ijvt];
                slacpy("F", m, n, ASAV, lda, A, lda);
                sgesvd(jobu, jobvt, m, n, A, lda, S, U, ldu,
                       VT, ldvt, work, lswork, &iinfo);

                /* Compare U */
                dif = 0.0f;
                if (m > 0 && n > 0) {
                    if (iju == 1) {
                        sort03("C", m, mnmin, m, mnmin, USAV,
                               ldu, A, lda, work, lwork, &dif,
                               &iinfo);
                    } else if (iju == 2) {
                        sort03("C", m, mnmin, m, mnmin, USAV,
                               ldu, U, ldu, work, lwork, &dif,
                               &iinfo);
                    } else if (iju == 3) {
                        sort03("C", m, m, m, mnmin, USAV, ldu,
                               U, ldu, work, lwork, &dif,
                               &iinfo);
                    }
                }
                result[4] = (result[4] > dif) ? result[4] : dif;

                /* Compare VT */
                dif = 0.0f;
                if (m > 0 && n > 0) {
                    if (ijvt == 1) {
                        sort03("R", n, mnmin, n, mnmin, VTSAV,
                               ldvt, A, lda, work, lwork, &dif,
                               &iinfo);
                    } else if (ijvt == 2) {
                        sort03("R", n, mnmin, n, mnmin, VTSAV,
                               ldvt, VT, ldvt, work, lwork,
                               &dif, &iinfo);
                    } else if (ijvt == 3) {
                        sort03("R", n, n, n, mnmin, VTSAV,
                               ldvt, VT, ldvt, work, lwork,
                               &dif, &iinfo);
                    }
                }
                result[5] = (result[5] > dif) ? result[5] : dif;

                /* Compare S */
                dif = 0.0f;
                div = ((f32)mnmin * ulp * S[0] > unfl) ? (f32)mnmin * ulp * S[0] : unfl;
                for (INT i = 0; i < mnmin - 1; i++) {
                    if (SSAV[i] < SSAV[i + 1])
                        dif = ulpinv;
                    if (SSAV[i] < 0.0f)
                        dif = ulpinv;
                    t = fabsf(SSAV[i] - S[i]) / div;
                    dif = (dif > t) ? dif : t;
                }
                result[6] = (result[6] > dif) ? result[6] : dif;
            }
        }

        /* Test SGESDD: Factorize A */
        iwtmp = 5 * mnmin * mnmin + 9 * mnmin + ((m > n) ? m : n);
        lswork = iwtmp + (iws - 1) * (lwork - iwtmp) / 3;
        lswork = (lswork < lwork) ? lswork : lwork;
        lswork = (lswork > 1) ? lswork : 1;
        if (iws == 4)
            lswork = lwork;

        slacpy("F", m, n, ASAV, lda, A, lda);
        sgesdd("A", m, n, A, lda, SSAV, USAV, ldu, VTSAV,
               ldvt, work, lswork, iwork, &iinfo);
        if (iinfo != 0) {
            snprintf(ctx, sizeof(ctx),
                     "SDRVBD: GESDD returned INFO=%d M=%d N=%d JTYPE=%d LSWORK=%d"
                     " ISEED=(%llu,%llu,%llu,%llu)",
                     (int)iinfo, (int)m, (int)n, (int)jtype, (int)lswork,
                     (unsigned long long)ioldsd[0], (unsigned long long)ioldsd[1],
                     (unsigned long long)ioldsd[2], (unsigned long long)ioldsd[3]);
            set_test_context(ctx);
            assert_info_success(iinfo);
            clear_test_context();
            return;
        }

        /* Do tests 8--11 */
        sbdt01(m, n, 0, ASAV, lda, USAV, ldu, SSAV, E,
               VTSAV, ldvt, work, &result[7]);
        if (m != 0 && n != 0) {
            sort01("Columns", m, m, USAV, ldu, work, lwork, &result[8]);
            sort01("Rows", n, n, VTSAV, ldvt, work, lwork, &result[9]);
        }
        result[10] = 0.0f;
        for (INT i = 0; i < mnmin - 1; i++) {
            if (SSAV[i] < SSAV[i + 1])
                result[10] = ulpinv;
            if (SSAV[i] < 0.0f)
                result[10] = ulpinv;
        }
        if (mnmin >= 1) {
            if (SSAV[mnmin - 1] < 0.0f)
                result[10] = ulpinv;
        }

        /* Do partial SVDs, comparing to SSAV, USAV, and VTSAV */
        result[11] = 0.0f;
        result[12] = 0.0f;
        result[13] = 0.0f;
        for (INT ijq = 0; ijq <= 2; ijq++) {
            jobq[0] = CJOB[ijq];
            slacpy("F", m, n, ASAV, lda, A, lda);
            sgesdd(jobq, m, n, A, lda, S, U, ldu, VT, ldvt,
                   work, lswork, iwork, &iinfo);

            /* Compare U */
            dif = 0.0f;
            if (m > 0 && n > 0) {
                if (ijq == 1) {
                    if (m >= n) {
                        sort03("C", m, mnmin, m, mnmin, USAV,
                               ldu, A, lda, work, lwork, &dif,
                               &info);
                    } else {
                        sort03("C", m, mnmin, m, mnmin, USAV,
                               ldu, U, ldu, work, lwork, &dif,
                               &info);
                    }
                } else if (ijq == 2) {
                    sort03("C", m, mnmin, m, mnmin, USAV, ldu,
                           U, ldu, work, lwork, &dif, &info);
                }
            }
            result[11] = (result[11] > dif) ? result[11] : dif;

            /* Compare VT */
            dif = 0.0f;
            if (m > 0 && n > 0) {
                if (ijq == 1) {
                    if (m >= n) {
                        sort03("R", n, mnmin, n, mnmin, VTSAV,
                               ldvt, VT, ldvt, work, lwork,
                               &dif, &info);
                    } else {
                        sort03("R", n, mnmin, n, mnmin, VTSAV,
                               ldvt, A, lda, work, lwork, &dif,
                               &info);
                    }
                } else if (ijq == 2) {
                    sort03("R", n, mnmin, n, mnmin, VTSAV,
                           ldvt, VT, ldvt, work, lwork, &dif,
                           &info);
                }
            }
            result[12] = (result[12] > dif) ? result[12] : dif;

            /* Compare S */
            dif = 0.0f;
            div = ((f32)mnmin * ulp * S[0] > unfl) ? (f32)mnmin * ulp * S[0] : unfl;
            for (INT i = 0; i < mnmin - 1; i++) {
                if (SSAV[i] < SSAV[i + 1])
                    dif = ulpinv;
                if (SSAV[i] < 0.0f)
                    dif = ulpinv;
                t = fabsf(SSAV[i] - S[i]) / div;
                dif = (dif > t) ? dif : t;
            }
            result[13] = (result[13] > dif) ? result[13] : dif;
        }

        /* Test SGESVDQ
         * Note: SGESVDQ only works for M >= N */
        result[35] = 0.0f;
        result[36] = 0.0f;
        result[37] = 0.0f;
        result[38] = 0.0f;

        if (m >= n) {
            iwtmp = 5 * mnmin * mnmin + 9 * mnmin + ((m > n) ? m : n);
            lswork = iwtmp + (iws - 1) * (lwork - iwtmp) / 3;
            lswork = (lswork < lwork) ? lswork : lwork;
            lswork = (lswork > 1) ? lswork : 1;
            if (iws == 4)
                lswork = lwork;

            slacpy("F", m, n, ASAV, lda, A, lda);

            lrwork = 2;
            liwork = (n > 1) ? n : 1;
            sgesvdq("H", "N", "N", "A", "A",
                    m, n, A, lda, SSAV, USAV, ldu,
                    VTSAV, ldvt, &numrank, iwork, liwork,
                    work, lwork, rwork, lrwork, &iinfo);

            if (iinfo != 0) {
                snprintf(ctx, sizeof(ctx),
                         "SDRVBD: SGESVDQ returned INFO=%d M=%d N=%d JTYPE=%d LSWORK=%d"
                         " ISEED=(%llu,%llu,%llu,%llu)",
                         (int)iinfo, (int)m, (int)n, (int)jtype, (int)lswork,
                         (unsigned long long)ioldsd[0], (unsigned long long)ioldsd[1],
                         (unsigned long long)ioldsd[2], (unsigned long long)ioldsd[3]);
                set_test_context(ctx);
                assert_info_success(iinfo);
                clear_test_context();
                return;
            }

            /* Do tests 36--39 */
            sbdt01(m, n, 0, ASAV, lda, USAV, ldu, SSAV, E,
                   VTSAV, ldvt, work, &result[35]);
            if (m != 0 && n != 0) {
                sort01("Columns", m, m, USAV, ldu, work,
                       lwork, &result[36]);
                sort01("Rows", n, n, VTSAV, ldvt, work,
                       lwork, &result[37]);
            }
            result[38] = 0.0f;
            for (INT i = 0; i < mnmin - 1; i++) {
                if (SSAV[i] < SSAV[i + 1])
                    result[38] = ulpinv;
                if (SSAV[i] < 0.0f)
                    result[38] = ulpinv;
            }
            if (mnmin >= 1) {
                if (SSAV[mnmin - 1] < 0.0f)
                    result[38] = ulpinv;
            }
        }

        /* Test SGESVJ
         * Note: SGESVJ only works for M >= N */
        result[14] = 0.0f;
        result[15] = 0.0f;
        result[16] = 0.0f;
        result[17] = 0.0f;

        if (m >= n) {
            iwtmp = 5 * mnmin * mnmin + 9 * mnmin + ((m > n) ? m : n);
            lswork = iwtmp + (iws - 1) * (lwork - iwtmp) / 3;
            lswork = (lswork < lwork) ? lswork : lwork;
            lswork = (lswork > 1) ? lswork : 1;
            if (iws == 4)
                lswork = lwork;

            slacpy("F", m, n, ASAV, lda, USAV, lda);
            sgesvj("G", "U", "V", m, n, USAV, lda, SSAV,
                   0, A, ldvt, work, lwork, &info);

            /* SGESVJ returns V not VT */
            for (INT j = 0; j < n; j++) {
                for (INT i = 0; i < n; i++) {
                    VTSAV[j + i * ldvt] = A[i + j * lda];
                }
            }

            if (iinfo != 0) {
                snprintf(ctx, sizeof(ctx),
                         "SDRVBD: GESVJ returned INFO=%d M=%d N=%d JTYPE=%d LSWORK=%d"
                         " ISEED=(%llu,%llu,%llu,%llu)",
                         (int)iinfo, (int)m, (int)n, (int)jtype, (int)lswork,
                         (unsigned long long)ioldsd[0], (unsigned long long)ioldsd[1],
                         (unsigned long long)ioldsd[2], (unsigned long long)ioldsd[3]);
                set_test_context(ctx);
                assert_info_success(iinfo);
                clear_test_context();
                return;
            }

            /* Do tests 15--18 */
            sbdt01(m, n, 0, ASAV, lda, USAV, ldu, SSAV, E,
                   VTSAV, ldvt, work, &result[14]);
            if (m != 0 && n != 0) {
                sort01("Columns", m, m, USAV, ldu, work,
                       lwork, &result[15]);
                sort01("Rows", n, n, VTSAV, ldvt, work,
                       lwork, &result[16]);
            }
            result[17] = 0.0f;
            for (INT i = 0; i < mnmin - 1; i++) {
                if (SSAV[i] < SSAV[i + 1])
                    result[17] = ulpinv;
                if (SSAV[i] < 0.0f)
                    result[17] = ulpinv;
            }
            if (mnmin >= 1) {
                if (SSAV[mnmin - 1] < 0.0f)
                    result[17] = ulpinv;
            }
        }

        /* Test SGEJSV
         * Note: SGEJSV only works for M >= N */
        result[18] = 0.0f;
        result[19] = 0.0f;
        result[20] = 0.0f;
        result[21] = 0.0f;
        if (m >= n) {
            iwtmp = 5 * mnmin * mnmin + 9 * mnmin + ((m > n) ? m : n);
            lswork = iwtmp + (iws - 1) * (lwork - iwtmp) / 3;
            lswork = (lswork < lwork) ? lswork : lwork;
            lswork = (lswork > 1) ? lswork : 1;
            if (iws == 4)
                lswork = lwork;

            slacpy("F", m, n, ASAV, lda, VTSAV, lda);
            sgejsv("G", "U", "V", "R", "N", "N",
                   m, n, VTSAV, lda, SSAV, USAV, ldu, A, ldvt,
                   work, lwork, iwork, &info);

            /* SGEJSV returns V not VT */
            for (INT j = 0; j < n; j++) {
                for (INT i = 0; i < n; i++) {
                    VTSAV[j + i * ldvt] = A[i + j * lda];
                }
            }

            if (iinfo != 0) {
                snprintf(ctx, sizeof(ctx),
                         "SDRVBD: GEJSV returned INFO=%d M=%d N=%d JTYPE=%d LSWORK=%d"
                         " ISEED=(%llu,%llu,%llu,%llu)",
                         (int)iinfo, (int)m, (int)n, (int)jtype, (int)lswork,
                         (unsigned long long)ioldsd[0], (unsigned long long)ioldsd[1],
                         (unsigned long long)ioldsd[2], (unsigned long long)ioldsd[3]);
                set_test_context(ctx);
                assert_info_success(iinfo);
                clear_test_context();
                return;
            }

            /* Do tests 19--22 */
            sbdt01(m, n, 0, ASAV, lda, USAV, ldu, SSAV, E,
                   VTSAV, ldvt, work, &result[18]);
            if (m != 0 && n != 0) {
                sort01("Columns", m, m, USAV, ldu, work,
                       lwork, &result[19]);
                sort01("Rows", n, n, VTSAV, ldvt, work,
                       lwork, &result[20]);
            }
            result[21] = 0.0f;
            for (INT i = 0; i < mnmin - 1; i++) {
                if (SSAV[i] < SSAV[i + 1])
                    result[21] = ulpinv;
                if (SSAV[i] < 0.0f)
                    result[21] = ulpinv;
            }
            if (mnmin >= 1) {
                if (SSAV[mnmin - 1] < 0.0f)
                    result[21] = ulpinv;
            }
        }

        /* Test SGESVDX */
        slacpy("F", m, n, ASAV, lda, A, lda);
        sgesvdx("V", "V", "A", m, n, A, lda,
                vl, vu, il, iu, &ns, SSAV, USAV, ldu,
                VTSAV, ldvt, work, lwork, iwork,
                &iinfo);
        if (iinfo != 0) {
            snprintf(ctx, sizeof(ctx),
                     "SDRVBD: GESVDX returned INFO=%d M=%d N=%d JTYPE=%d LSWORK=%d"
                     " ISEED=(%llu,%llu,%llu,%llu)",
                     (int)iinfo, (int)m, (int)n, (int)jtype, (int)lswork,
                     (unsigned long long)ioldsd[0], (unsigned long long)ioldsd[1],
                     (unsigned long long)ioldsd[2], (unsigned long long)ioldsd[3]);
            set_test_context(ctx);
            assert_info_success(iinfo);
            clear_test_context();
            return;
        }

        /* Do tests 23--29 */
        result[22] = 0.0f;
        result[23] = 0.0f;
        result[24] = 0.0f;
        sbdt01(m, n, 0, ASAV, lda, USAV, ldu, SSAV, E,
               VTSAV, ldvt, work, &result[22]);
        if (m != 0 && n != 0) {
            sort01("Columns", m, m, USAV, ldu, work, lwork,
                   &result[23]);
            sort01("Rows", n, n, VTSAV, ldvt, work, lwork,
                   &result[24]);
        }
        result[25] = 0.0f;
        for (INT i = 0; i < mnmin - 1; i++) {
            if (SSAV[i] < SSAV[i + 1])
                result[25] = ulpinv;
            if (SSAV[i] < 0.0f)
                result[25] = ulpinv;
        }
        if (mnmin >= 1) {
            if (SSAV[mnmin - 1] < 0.0f)
                result[25] = ulpinv;
        }

        /* Do partial SVDs, comparing to SSAV, USAV, and VTSAV */
        result[26] = 0.0f;
        result[27] = 0.0f;
        result[28] = 0.0f;
        for (INT iju = 0; iju <= 1; iju++) {
            for (INT ijvt = 0; ijvt <= 1; ijvt++) {
                if ((iju == 0 && ijvt == 0) ||
                    (iju == 1 && ijvt == 1))
                    continue;
                jobu[0] = CJOBV[iju];
                jobvt[0] = CJOBV[ijvt];
                range[0] = CJOBR[0];
                slacpy("F", m, n, ASAV, lda, A, lda);
                sgesvdx(jobu, jobvt, range, m, n, A, lda,
                        vl, vu, il, iu, &ns, S, U, ldu,
                        VT, ldvt, work, lwork, iwork,
                        &iinfo);

                /* Compare U */
                dif = 0.0f;
                if (m > 0 && n > 0) {
                    if (iju == 1) {
                        sort03("C", m, mnmin, m, mnmin, USAV,
                               ldu, U, ldu, work, lwork, &dif,
                               &iinfo);
                    }
                }
                result[26] = (result[26] > dif) ? result[26] : dif;

                /* Compare VT */
                dif = 0.0f;
                if (m > 0 && n > 0) {
                    if (ijvt == 1) {
                        sort03("R", n, mnmin, n, mnmin, VTSAV,
                               ldvt, VT, ldvt, work, lwork,
                               &dif, &iinfo);
                    }
                }
                result[27] = (result[27] > dif) ? result[27] : dif;

                /* Compare S */
                dif = 0.0f;
                div = ((f32)mnmin * ulp * S[0] > unfl) ? (f32)mnmin * ulp * S[0] : unfl;
                for (INT i = 0; i < mnmin - 1; i++) {
                    if (SSAV[i] < SSAV[i + 1])
                        dif = ulpinv;
                    if (SSAV[i] < 0.0f)
                        dif = ulpinv;
                    t = fabsf(SSAV[i] - S[i]) / div;
                    dif = (dif > t) ? dif : t;
                }
                result[28] = (result[28] > dif) ? result[28] : dif;
            }
        }

        /* Do tests 30--32: SGESVDX( 'V', 'V', 'I' ) */
        for (INT i = 0; i < 4; i++) {
            iseed2[i] = iseed[i];
        }
        if (mnmin <= 1) {
            il = 0;
            iu = ((1 > mnmin) ? 1 : mnmin) - 1;
        } else {
            il = (INT)((f32)mnmin * rng_uniform_f32(iseed2));
            iu = (INT)((f32)mnmin * rng_uniform_f32(iseed2));
            if (iu < il) {
                itemp = iu;
                iu = il;
                il = itemp;
            }
        }
        slacpy("F", m, n, ASAV, lda, A, lda);
        sgesvdx("V", "V", "I", m, n, A, lda,
                vl, vu, il, iu, &nsi, S, U, ldu,
                VT, ldvt, work, lwork, iwork,
                &iinfo);
        if (iinfo != 0) {
            snprintf(ctx, sizeof(ctx),
                     "SDRVBD: GESVDX returned INFO=%d M=%d N=%d JTYPE=%d LSWORK=%d"
                     " ISEED=(%llu,%llu,%llu,%llu)",
                     (int)iinfo, (int)m, (int)n, (int)jtype, (int)lswork,
                     (unsigned long long)ioldsd[0], (unsigned long long)ioldsd[1],
                     (unsigned long long)ioldsd[2], (unsigned long long)ioldsd[3]);
            set_test_context(ctx);
            assert_info_success(iinfo);
            clear_test_context();
            return;
        }

        result[29] = 0.0f;
        result[30] = 0.0f;
        result[31] = 0.0f;
        sbdt05(m, n, ASAV, lda, S, nsi, U, ldu,
               VT, ldvt, work, &result[29]);
        sort01("Columns", m, nsi, U, ldu, work, lwork,
               &result[30]);
        sort01("Rows", nsi, n, VT, ldvt, work, lwork,
               &result[31]);

        /* Do tests 33--35: SGESVDX( 'V', 'V', 'V' ) */
        if (mnmin > 0 && nsi > 1) {
            if (il != 0) {
                t = 0.5f * fabsf(SSAV[il] - SSAV[il - 1]);
                t = (t > ulp * anorm) ? t : ulp * anorm;
                t = (t > 2.0f * rtunfl) ? t : 2.0f * rtunfl;
                vu = SSAV[il] + t;
            } else {
                t = 0.5f * fabsf(SSAV[ns - 1] - SSAV[0]);
                t = (t > ulp * anorm) ? t : ulp * anorm;
                t = (t > 2.0f * rtunfl) ? t : 2.0f * rtunfl;
                vu = SSAV[0] + t;
            }
            if (iu != ns - 1) {
                t = (ulp * anorm > 2.0f * rtunfl) ? ulp * anorm : 2.0f * rtunfl;
                t = (t > 0.5f * fabsf(SSAV[iu + 1] - SSAV[iu]))
                        ? t : 0.5f * fabsf(SSAV[iu + 1] - SSAV[iu]);
                vl = SSAV[iu] - t;
            } else {
                t = (ulp * anorm > 2.0f * rtunfl) ? ulp * anorm : 2.0f * rtunfl;
                t = (t > 0.5f * fabsf(SSAV[ns - 1] - SSAV[0]))
                        ? t : 0.5f * fabsf(SSAV[ns - 1] - SSAV[0]);
                vl = SSAV[ns - 1] - t;
            }
            vl = (vl > 0.0f) ? vl : 0.0f;
            vu = (vu > 0.0f) ? vu : 0.0f;
            if (vl >= vu)
                vu = (vu * 2 > vu + vl + 0.5f) ? vu * 2 : vu + vl + 0.5f;
        } else {
            vl = 0.0f;
            vu = 1.0f;
        }
        slacpy("F", m, n, ASAV, lda, A, lda);
        sgesvdx("V", "V", "V", m, n, A, lda,
                vl, vu, il, iu, &nsv, S, U, ldu,
                VT, ldvt, work, lwork, iwork,
                &iinfo);
        if (iinfo != 0) {
            snprintf(ctx, sizeof(ctx),
                     "SDRVBD: GESVDX returned INFO=%d M=%d N=%d JTYPE=%d LSWORK=%d"
                     " ISEED=(%llu,%llu,%llu,%llu)",
                     (int)iinfo, (int)m, (int)n, (int)jtype, (int)lswork,
                     (unsigned long long)ioldsd[0], (unsigned long long)ioldsd[1],
                     (unsigned long long)ioldsd[2], (unsigned long long)ioldsd[3]);
            set_test_context(ctx);
            assert_info_success(iinfo);
            clear_test_context();
            return;
        }

        result[32] = 0.0f;
        result[33] = 0.0f;
        result[34] = 0.0f;
        sbdt05(m, n, ASAV, lda, S, nsv, U, ldu,
               VT, ldvt, work, &result[32]);
        sort01("Columns", m, nsv, U, ldu, work, lwork,
               &result[33]);
        sort01("Rows", nsv, n, VT, ldvt, work, lwork,
               &result[34]);

        /* End of Loop -- Check for RESULT(j) > THRESH */
        for (INT j = 0; j < 39; j++) {
            if (result[j] >= THRESH) {
                if (g_ws->nfail == 0) {
                    print_header();
                }
                g_ws->nfail = g_ws->nfail + 1;
                snprintf(ctx, sizeof(ctx),
                         "sdrvbd M=%d N=%d type=%d IWS=%d"
                         " seed=%llu,%llu,%llu,%llu test(%d)=%.4g",
                         (int)m, (int)n, (int)jtype, (int)iws,
                         (unsigned long long)ioldsd[0], (unsigned long long)ioldsd[1],
                         (unsigned long long)ioldsd[2], (unsigned long long)ioldsd[3],
                         (int)(j + 1), (double)result[j]);
                set_test_context(ctx);
                assert_residual_below(result[j], THRESH);
                clear_test_context();
            }
        }
    }
}

static void test_serred(void** state)
{
    (void)state;
    INT ok, nt;
    serred("SBD", &ok, &nt);
    assert_true(ok);
}

static void test_sdrvbd_case(void** state)
{
    const sdrvbd_params_t* params = *state;
    run_sdrvbd_single(params);
}

#define MAX_TESTS (1 + NPARMS * NSIZES * MTYPES)

static sdrvbd_params_t g_params[MAX_TESTS];
static struct CMUnitTest g_tests[MAX_TESTS];
static INT g_num_tests = 0;

static void build_test_array(void)
{
    g_num_tests = 0;

    g_tests[g_num_tests].name = "serred";
    g_tests[g_num_tests].test_func = test_serred;
    g_tests[g_num_tests].initial_state = NULL;
    g_tests[g_num_tests].setup_func = NULL;
    g_tests[g_num_tests].teardown_func = NULL;
    g_num_tests++;

    for (INT ip = 0; ip < NPARMS; ip++) {
        for (INT js = 0; js < NSIZES; js++) {
            for (INT jt = 1; jt <= MTYPES; jt++) {
                sdrvbd_params_t* p = &g_params[g_num_tests];
                p->jsize = js;
                p->jtype = jt;
                p->iparm = ip;
                snprintf(p->name, sizeof(p->name),
                         "sdrvbd_p%d_m%d_n%d_type%d", (int)ip,
                         (int)MVAL[js], (int)NVAL[js], (int)jt);

                g_tests[g_num_tests].name = p->name;
                g_tests[g_num_tests].test_func = test_sdrvbd_case;
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

    (void)_cmocka_run_group_tests("sdrvbd", g_tests, g_num_tests,
                                   group_setup, group_teardown);
    return 0;
}
