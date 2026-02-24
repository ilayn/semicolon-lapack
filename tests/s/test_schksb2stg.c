/**
 * @file test_schksb2stg.c
 * @brief Symmetric band tridiagonal reduction test driver (2-stage) - port of
 *        LAPACK TESTING/EIG/dchksb2stg.f
 *
 * Tests the reduction of a symmetric band matrix to tridiagonal form,
 * comparing the 1-stage (SSBTRD) and 2-stage (SSYTRD_SB2ST) routines.
 *
 * SSBTRD factors a symmetric band matrix A as  U S U' , where ' means
 * transpose, S is symmetric tridiagonal, and U is orthogonal.
 * SSBTRD can use either just the lower or just the upper triangle
 * of A; DCHKSB2STG checks both cases.
 *
 * SSYTRD_SB2ST factors a symmetric band matrix A as  U S U' ,
 * where ' means transpose, S is symmetric tridiagonal, and U is
 * orthogonal. SSYTRD_SB2ST can use either just the lower or just
 * the upper triangle of A; DCHKSB2STG checks both cases.
 *
 * SSTEQR factors S as  Z D1 Z'.
 * D1 is the matrix of eigenvalues computed when Z is not computed
 * and from the S resulting of SSBTRD "U" (used as reference for SSYTRD_SB2ST)
 * D2 is the matrix of eigenvalues computed when Z is not computed
 * and from the S resulting of SSYTRD_SB2ST "U".
 * D3 is the matrix of eigenvalues computed when Z is not computed
 * and from the S resulting of SSYTRD_SB2ST "L".
 *
 * Test ratios (6 total):
 *
 *   (1)  | A - U S U' | / ( |A| n ulp )  UPLO='U'
 *
 *   (2)  | I - U U' | / ( n ulp )         UPLO='U'
 *
 *   (3)  | A - U S U' | / ( |A| n ulp )  UPLO='L'
 *
 *   (4)  | I - U U' | / ( n ulp )         UPLO='L'
 *
 *   (5)  | D1 - D2 | / ( |D1| ulp )      D1 from SSBTRD(U), D2 from
 *                                         SSYTRD_SB2ST(U)
 *
 *   (6)  | D1 - D3 | / ( |D1| ulp )      D1 from SSBTRD(U), D3 from
 *                                         SSYTRD_SB2ST(L)
 *
 * Matrix types: 15 types (MAXTYP = 15)
 */

#include "test_harness.h"
#include "verify.h"
#include "test_rng.h"
#include <math.h>
#include <string.h>
#include "semicolon_cblas.h"

#define THRESH 20.0f
#define MAXTYP 15
#define NTEST  6

/* Matrix sizes */
static const INT NVAL[] = {0, 1, 2, 3, 5, 10, 20};
#define NSIZES (sizeof(NVAL) / sizeof(NVAL[0]))

/* Bandwidth values */
static const INT KK[] = {0, 1, 2, 3, 10};
#define NWDTHS (sizeof(KK) / sizeof(KK[0]))

/* External function declarations */
/* ===================================================================== */
/* DATA arrays from dchksb2stg.f lines 386-390                           */
/* ===================================================================== */

/*                    1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 */
static const INT ktype[MAXTYP] = {
    1, 2, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 8, 8, 8
};
static const INT kmagn[MAXTYP] = {
    1, 1, 1, 1, 1, 2, 3, 1, 1, 1, 2, 3, 1, 2, 3
};
static const INT kmode[MAXTYP] = {
    0, 0, 4, 3, 1, 4, 4, 4, 3, 1, 4, 4, 0, 0, 0
};

/* ===================================================================== */
/* Test parameters and workspace                                         */
/* ===================================================================== */

typedef struct {
    INT jsize;     /* index into NVAL[] */
    INT jwidth;    /* index into KK[] */
    INT jtype;     /* matrix type 1..15 */
    char name[128];
} dchksb2stg_params_t;

typedef struct {
    INT nmax;
    INT kmax;
    INT lda;
    f32* A;
    f32* SD;
    f32* SE;
    f32* D1;
    f32* D2;
    f32* D3;
    f32* U;
    f32* work;
    INT lwork;
    uint64_t rng_state[4];
} dchksb2stg_workspace_t;

static dchksb2stg_workspace_t* g_ws = NULL;

/* ===================================================================== */
/* Group setup / teardown                                                */
/* ===================================================================== */

static int group_setup(void** state)
{
    (void)state;

    g_ws = calloc(1, sizeof(dchksb2stg_workspace_t));
    if (!g_ws) return -1;

    g_ws->nmax = 1;
    for (size_t i = 0; i < NSIZES; i++) {
        if (NVAL[i] > g_ws->nmax) g_ws->nmax = NVAL[i];
    }
    g_ws->kmax = 0;
    for (size_t i = 0; i < NWDTHS; i++) {
        if (KK[i] > g_ws->kmax) g_ws->kmax = KK[i];
    }
    INT nmax_m1 = g_ws->nmax - 1;
    if (nmax_m1 < g_ws->kmax) g_ws->kmax = nmax_m1;
    if (g_ws->kmax < 0) g_ws->kmax = 0;

    INT nmax = g_ws->nmax;
    INT kmax = g_ws->kmax;

    g_ws->lda = kmax + 1;
    if (g_ws->lda < 2) g_ws->lda = 2;
    INT lda = g_ws->lda;

    /* Extra kmax room because ITYPE=4,7 pass &A[k] to slatms/slatmr,
       which internally zeroes lda*n elements from that offset */
    g_ws->A    = malloc((size_t)(lda * nmax + kmax) * sizeof(f32));
    g_ws->SD   = malloc((size_t)nmax * sizeof(f32));
    g_ws->SE   = malloc((size_t)nmax * sizeof(f32));
    g_ws->D1   = malloc((size_t)nmax * sizeof(f32));
    g_ws->D2   = malloc((size_t)nmax * sizeof(f32));
    g_ws->D3   = malloc((size_t)nmax * sizeof(f32));
    g_ws->U    = malloc((size_t)nmax * nmax * sizeof(f32));

    INT maxln = lda > nmax ? lda : nmax;
    g_ws->lwork = (maxln + 1) * nmax;
    /* ssbt21 needs n*n + n workspace */
    if (g_ws->lwork < nmax * nmax + nmax) g_ws->lwork = nmax * nmax + nmax;
    /* ssytrd_sb2st needs LH = max(1, 4*n) for hous, plus lwmin for work.
       lwmin = (2*ib+1)*n + ib*nthreads where ib=16, nthreads=1 for SB2ST.
       Ensure total lwork >= LH + lwmin. */
    {
        INT lh_need = 4 * nmax;
        if (lh_need < 1) lh_need = 1;
        INT ib_est = 16;
        INT lw_need = (2 * ib_est + 1) * nmax + ib_est;
        INT total_sb2st = lh_need + lw_need;
        if (g_ws->lwork < total_sb2st) g_ws->lwork = total_sb2st;
    }
    g_ws->work = malloc((size_t)g_ws->lwork * sizeof(f32));

    if (!g_ws->A || !g_ws->SD || !g_ws->SE || !g_ws->D1 ||
        !g_ws->D2 || !g_ws->D3 || !g_ws->U || !g_ws->work) {
        return -1;
    }

    rng_seed(g_ws->rng_state, 0xD5B15ACEULL);
    return 0;
}

static int group_teardown(void** state)
{
    (void)state;

    if (g_ws) {
        free(g_ws->A);
        free(g_ws->SD);
        free(g_ws->SE);
        free(g_ws->D1);
        free(g_ws->D2);
        free(g_ws->D3);
        free(g_ws->U);
        free(g_ws->work);
        free(g_ws);
        g_ws = NULL;
    }
    return 0;
}

/* ===================================================================== */
/* Single test case                                                      */
/* ===================================================================== */

static void run_dchksb2stg_single(dchksb2stg_params_t* params)
{
    const INT n = NVAL[params->jsize];
    INT k = KK[params->jwidth];
    const INT jtype = params->jtype;
    const INT jt = jtype - 1;

    if (k > n)
        return;

    /* Clamp k */
    if (k > n - 1) k = n - 1;
    if (k < 0) k = 0;

    const INT lda   = g_ws->lda;
    const INT ldu   = (g_ws->nmax > 1) ? g_ws->nmax : 1;

    f32* A     = g_ws->A;
    f32* SD    = g_ws->SD;
    f32* SE    = g_ws->SE;
    f32* D1    = g_ws->D1;
    f32* D2    = g_ws->D2;
    f32* D3    = g_ws->D3;
    f32* U     = g_ws->U;
    f32* work  = g_ws->work;
    INT lwork  = g_ws->lwork;
    uint64_t* rng = g_ws->rng_state;

    const f32 unfl    = slamch("S");
    const f32 ovfl    = 1.0f / unfl;
    const f32 ulp     = slamch("P");
    const f32 ulpinv  = 1.0f / ulp;
    const f32 rtunfl  = sqrtf(unfl);
    const f32 rtovfl  = sqrtf(ovfl);

    f32 result[NTEST];
    for (INT i = 0; i < NTEST; i++)
        result[i] = 0.0f;

    INT iinfo = 0;
    INT ntest = 0;
    INT lh = 0, lw = 0;
    char ctx[256];

    const f32 aninv = 1.0f / (f32)(n > 1 ? n : 1);

    /* ----------------------------------------------------------- */
    /* Compute "A"                                                 */
    /* Store as "Upper"; later, we will copy to other format.      */
    /* ----------------------------------------------------------- */

    if (jtype <= MAXTYP) {
        INT itype = ktype[jt];
        INT imode = kmode[jt];

        f32 anorm;
        switch (kmagn[jt]) {
            case 2:  anorm = (rtovfl * ulp) * aninv; break;
            case 3:  anorm = rtunfl * (f32)n * ulpinv; break;
            default: anorm = 1.0f; break;
        }

        f32 cond;
        if (jtype <= 15) {
            cond = ulpinv;
        } else {
            cond = ulpinv * aninv / 10.0f;
        }

        slaset("Full", lda, n, 0.0f, 0.0f, A, lda);
        iinfo = 0;

        if (itype == 1) {
            /* Zero matrix */
            iinfo = 0;

        } else if (itype == 2) {
            /* Identity */
            for (INT jcol = 0; jcol < n; jcol++)
                A[k + jcol * lda] = anorm;

        } else if (itype == 4) {
            /* Diagonal Matrix, [Eigen]values Specified */
            slatms(n, n, "S", "S", work, imode, cond, anorm,
                   0, 0, "Q", &A[k], lda, work + n, &iinfo, rng);

        } else if (itype == 5) {
            /* Symmetric, eigenvalues specified */
            slatms(n, n, "S", "S", work, imode, cond, anorm,
                   k, k, "Q", A, lda, work + n, &iinfo, rng);

        } else if (itype == 7) {
            /* Diagonal, random eigenvalues */
            INT idumma[1] = {0};
            slatmr(n, n, "S", "S", work, 6, 1.0f, 1.0f,
                   "T", "N", work + n, 1, 1.0f,
                   work + 2 * n, 1, 1.0f, "N", idumma, 0, 0,
                   0.0f, anorm, "Q", &A[k], lda, idumma, &iinfo, rng);

        } else if (itype == 8) {
            /* Symmetric, random eigenvalues */
            INT idumma[1] = {0};
            slatmr(n, n, "S", "S", work, 6, 1.0f, 1.0f,
                   "T", "N", work + n, 1, 1.0f,
                   work + 2 * n, 1, 1.0f, "N", idumma, k, k,
                   0.0f, anorm, "Q", A, lda, idumma, &iinfo, rng);

        } else if (itype == 9) {
            /* Positive definite, eigenvalues specified */
            slatms(n, n, "S", "P", work, imode, cond, anorm,
                   k, k, "Q", A, lda, work + n, &iinfo, rng);

        } else if (itype == 10) {
            /* Positive definite tridiagonal, eigenvalues specified */
            if (n > 1 && k < 1) k = 1;
            slatms(n, n, "S", "P", work, imode, cond, anorm,
                   1, 1, "Q", &A[k - 1], lda, work + n, &iinfo, rng);
            for (INT i = 1; i < n; i++) {
                f32 temp1 = fabsf(A[(k - 1) + i * lda]) /
                            sqrtf(fabsf(A[k + (i - 1) * lda] * A[k + i * lda]));
                if (temp1 > 0.5f) {
                    A[(k - 1) + i * lda] = 0.5f *
                        sqrtf(fabsf(A[k + (i - 1) * lda] * A[k + i * lda]));
                }
            }

        } else {
            iinfo = 1;
        }

        if (iinfo != 0) {
            snprintf(ctx, sizeof(ctx),
                     "dchksb2stg n=%d k=%d type=%d: Generator info=%d",
                     n, k, jtype, iinfo);
            set_test_context(ctx);
            assert_info_success(iinfo);
            clear_test_context();
            return;
        }
    }

    /* ----------------------------------------------------------- */
    /* Call SSBTRD to compute S and U from upper triangle           */
    /* ----------------------------------------------------------- */

    slacpy(" ", k + 1, n, A, lda, work, lda);

    ntest = 1;
    ssbtrd("V", "U", n, k, work, lda, SD, SE, U, ldu,
           work + lda * n, &iinfo);

    if (iinfo != 0) {
        snprintf(ctx, sizeof(ctx),
                 "dchksb2stg n=%d k=%d type=%d: SSBTRD(U) info=%d",
                 n, k, jtype, iinfo);
        set_test_context(ctx);
        if (iinfo < 0) {
            assert_info_success(iinfo);
            clear_test_context();
            return;
        } else {
            result[0] = ulpinv;
            goto check_results;
        }
    }

    /* Do tests 1 and 2 */
    ssbt21("Upper", n, k, 1, A, lda, SD, SE, U, ldu,
           work, &result[0]);

    /* ----------------------------------------------------------- */
    /* Before converting A into lower for SSBTRD, run SSYTRD_SB2ST */
    /* otherwise matrix A will be converted to lower and then need */
    /* to be converted back to upper in order to run the upper case */
    /* of SSYTRD_SB2ST                                              */
    /* ----------------------------------------------------------- */

    /* Compute D1 from the SSBTRD and used as reference for the
       SSYTRD_SB2ST */

    cblas_scopy(n, SD, 1, D1, 1);
    if (n > 0)
        cblas_scopy(n - 1, SE, 1, work, 1);

    ssteqr("N", n, D1, work, NULL, ldu,
           NULL, &iinfo);
    if (iinfo != 0) {
        snprintf(ctx, sizeof(ctx),
                 "dchksb2stg n=%d k=%d type=%d: SSTEQR(N) info=%d",
                 n, k, jtype, iinfo);
        set_test_context(ctx);
        if (iinfo < 0) {
            assert_info_success(iinfo);
            clear_test_context();
            return;
        } else {
            result[4] = ulpinv;
            goto check_results;
        }
    }

    /* SSYTRD_SB2ST Upper case is used to compute D2.
       Note to set SD and SE to zero to be sure not reusing
       the one from above. Compare it with D1 computed
       using the SSBTRD. */

    slaset("Full", n, 1, 0.0f, 0.0f, SD, n > 0 ? n : 1);
    slaset("Full", n, 1, 0.0f, 0.0f, SE, n > 0 ? n : 1);
    slacpy(" ", k + 1, n, A, lda, U, ldu);
    lh = 4 * n;
    if (lh < 1) lh = 1;
    lw = lwork - lh;
    ssytrd_sb2st("N", "N", "U", n, k, U, ldu, SD, SE,
                 work, lh, work + lh, lw, &iinfo);

    /* Compute D2 from the SSYTRD_SB2ST Upper case */

    cblas_scopy(n, SD, 1, D2, 1);
    if (n > 0)
        cblas_scopy(n - 1, SE, 1, work, 1);

    ssteqr("N", n, D2, work, NULL, ldu,
           NULL, &iinfo);
    if (iinfo != 0) {
        snprintf(ctx, sizeof(ctx),
                 "dchksb2stg n=%d k=%d type=%d: SSTEQR(N) D2 info=%d",
                 n, k, jtype, iinfo);
        set_test_context(ctx);
        if (iinfo < 0) {
            assert_info_success(iinfo);
            clear_test_context();
            return;
        } else {
            result[4] = ulpinv;
            goto check_results;
        }
    }

    /* ----------------------------------------------------------- */
    /* Convert A from Upper-Triangle-Only storage to                */
    /* Lower-Triangle-Only storage.                                 */
    /* ----------------------------------------------------------- */

    for (INT jc = 0; jc < n; jc++) {
        INT jrmax = k < n - 1 - jc ? k : n - 1 - jc;
        for (INT jr = 0; jr <= jrmax; jr++)
            A[jr + jc * lda] = A[k - jr + (jc + jr) * lda];
    }
    for (INT jc = n - k; jc < n; jc++) {
        if (jc < 0) continue;
        INT jrmin_lim = k < n - 1 - jc ? k : n - 1 - jc;
        for (INT jr = jrmin_lim + 1; jr <= k; jr++)
            A[jr + jc * lda] = 0.0f;
    }

    /* ----------------------------------------------------------- */
    /* Call SSBTRD to compute S and U from lower triangle           */
    /* ----------------------------------------------------------- */

    slacpy(" ", k + 1, n, A, lda, work, lda);

    ntest = 3;
    ssbtrd("V", "L", n, k, work, lda, SD, SE, U, ldu,
           work + lda * n, &iinfo);

    if (iinfo != 0) {
        snprintf(ctx, sizeof(ctx),
                 "dchksb2stg n=%d k=%d type=%d: SSBTRD(L) info=%d",
                 n, k, jtype, iinfo);
        set_test_context(ctx);
        if (iinfo < 0) {
            assert_info_success(iinfo);
            clear_test_context();
            return;
        } else {
            result[2] = ulpinv;
            goto check_results;
        }
    }
    ntest = 4;

    /* Do tests 3 and 4 */
    ssbt21("Lower", n, k, 1, A, lda, SD, SE, U, ldu,
           work, &result[2]);

    /* ----------------------------------------------------------- */
    /* SSYTRD_SB2ST Lower case is used to compute D3.              */
    /* Note to set SD and SE to zero to be sure not reusing        */
    /* the one from above. Compare it with D1 computed             */
    /* using the SSBTRD.                                           */
    /* ----------------------------------------------------------- */

    slaset("Full", n, 1, 0.0f, 0.0f, SD, n > 0 ? n : 1);
    slaset("Full", n, 1, 0.0f, 0.0f, SE, n > 0 ? n : 1);
    slacpy(" ", k + 1, n, A, lda, U, ldu);
    lh = 4 * n;
    if (lh < 1) lh = 1;
    lw = lwork - lh;
    ssytrd_sb2st("N", "N", "L", n, k, U, ldu, SD, SE,
                 work, lh, work + lh, lw, &iinfo);

    /* Compute D3 from the SSYTRD_SB2ST Lower case */

    cblas_scopy(n, SD, 1, D3, 1);
    if (n > 0)
        cblas_scopy(n - 1, SE, 1, work, 1);

    ssteqr("N", n, D3, work, NULL, ldu,
           NULL, &iinfo);
    if (iinfo != 0) {
        snprintf(ctx, sizeof(ctx),
                 "dchksb2stg n=%d k=%d type=%d: SSTEQR(N) D3 info=%d",
                 n, k, jtype, iinfo);
        set_test_context(ctx);
        if (iinfo < 0) {
            assert_info_success(iinfo);
            clear_test_context();
            return;
        } else {
            result[5] = ulpinv;
            goto check_results;
        }
    }

    /* ----------------------------------------------------------- */
    /* Do Tests 5 and 6: compare eigenvalues from SSBTRD vs        */
    /* SSYTRD_SB2ST                                                */
    /* ----------------------------------------------------------- */

    ntest = 6;
    {
        f32 temp1 = 0.0f, temp2 = 0.0f, temp3 = 0.0f, temp4 = 0.0f;

        for (INT j = 0; j < n; j++) {
            f32 ad1 = fabsf(D1[j]);
            f32 ad2 = fabsf(D2[j]);
            f32 ad3 = fabsf(D3[j]);
            if (ad1 > temp1) temp1 = ad1;
            if (ad2 > temp1) temp1 = ad2;
            f32 diff12 = fabsf(D1[j] - D2[j]);
            if (diff12 > temp2) temp2 = diff12;
            if (ad1 > temp3) temp3 = ad1;
            if (ad3 > temp3) temp3 = ad3;
            f32 diff13 = fabsf(D1[j] - D3[j]);
            if (diff13 > temp4) temp4 = diff13;
        }

        f32 denom5 = unfl > ulp * (temp1 > temp2 ? temp1 : temp2)
                         ? unfl : ulp * (temp1 > temp2 ? temp1 : temp2);
        result[4] = temp2 / denom5;

        f32 denom6 = unfl > ulp * (temp3 > temp4 ? temp3 : temp4)
                         ? unfl : ulp * (temp3 > temp4 ? temp3 : temp4);
        result[5] = temp4 / denom6;
    }

    /* ----------------------------------------------------------- */
    /* End of Loop -- Check for RESULT(j) > THRESH                 */
    /* ----------------------------------------------------------- */

check_results:
    for (INT jr = 0; jr < ntest; jr++) {
        snprintf(ctx, sizeof(ctx), "dchksb2stg n=%d k=%d type=%d TEST %d",
                 n, k, jtype, jr + 1);
        set_test_context(ctx);
        assert_residual_below(result[jr], THRESH);
    }
    clear_test_context();
}

/* ===================================================================== */
/* CMocka wrapper                                                        */
/* ===================================================================== */

static void test_dchksb2stg_case(void** state)
{
    (void)state;
    dchksb2stg_params_t* params = *state;
    run_dchksb2stg_single(params);
}

/* ===================================================================== */
/* Build test array                                                      */
/* ===================================================================== */

#define MAX_TESTS (NSIZES * NWDTHS * MAXTYP)

static dchksb2stg_params_t g_params[MAX_TESTS];
static struct CMUnitTest g_tests[MAX_TESTS];
static INT g_num_tests = 0;

static void build_test_array(void)
{
    g_num_tests = 0;

    for (size_t is = 0; is < NSIZES; is++) {
        INT n = NVAL[is];
        for (size_t iw = 0; iw < NWDTHS; iw++) {
            INT kval = KK[iw];
            if (kval > n)
                continue;
            for (INT jtype = 1; jtype <= MAXTYP; jtype++) {
                dchksb2stg_params_t* p = &g_params[g_num_tests];
                p->jsize = (INT)is;
                p->jwidth = (INT)iw;
                p->jtype = jtype;
                snprintf(p->name, sizeof(p->name),
                         "dchksb2stg_n%d_k%d_type%d", n, kval, jtype);

                g_tests[g_num_tests].name = p->name;
                g_tests[g_num_tests].test_func = test_dchksb2stg_case;
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
    return _cmocka_run_group_tests("dchksb2stg", g_tests, g_num_tests,
                                    group_setup, group_teardown);
}
