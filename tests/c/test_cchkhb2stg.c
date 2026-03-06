/**
 * @file test_cchkhb2stg.c
 * @brief Hermitian band tridiagonal reduction test driver (2-stage) - port of
 *        LAPACK TESTING/EIG/zchkhb2stg.f
 *
 * Tests the reduction of a Hermitian band matrix to tridiagonal form,
 * comparing the 1-stage (CHBTRD) and 2-stage (CHETRD_HB2ST) routines.
 *
 * CHBTRD factors a Hermitian band matrix A as  U S U* , where * means
 * conjugate transpose, S is symmetric tridiagonal, and U is unitary.
 * CHBTRD can use either just the lower or just the upper triangle
 * of A; ZCHKHB2STG checks both cases.
 *
 * CHETRD_HB2ST factors a Hermitian band matrix A as  U S U* ,
 * where * means conjugate transpose, S is symmetric tridiagonal, and U is
 * unitary. CHETRD_HB2ST can use either just the lower or just
 * the upper triangle of A; ZCHKHB2STG checks both cases.
 *
 * CSTEQR factors S as  Z D1 Z'.
 * D1 is the matrix of eigenvalues computed when Z is not computed
 * and from the S resulting of CHBTRD "U" (used as reference for CHETRD_HB2ST)
 * D2 is the matrix of eigenvalues computed when Z is not computed
 * and from the S resulting of CHETRD_HB2ST "U".
 * D3 is the matrix of eigenvalues computed when Z is not computed
 * and from the S resulting of CHETRD_HB2ST "L".
 *
 * Test ratios (6 total):
 *
 *   (1)  | A - U S U* | / ( |A| n ulp )  UPLO='U'
 *
 *   (2)  | I - U U* | / ( n ulp )         UPLO='U'
 *
 *   (3)  | A - U S U* | / ( |A| n ulp )  UPLO='L'
 *
 *   (4)  | I - U U* | / ( n ulp )         UPLO='L'
 *
 *   (5)  | D1 - D2 | / ( |D1| ulp )      D1 from CHBTRD(U), D2 from
 *                                         CHETRD_HB2ST(U)
 *
 *   (6)  | D1 - D3 | / ( |D1| ulp )      D1 from CHBTRD(U), D3 from
 *                                         CHETRD_HB2ST(L)
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

typedef struct {
    INT jsize;
    INT jwidth;
    INT jtype;
    char name[128];
} zchkhb2stg_params_t;

typedef struct {
    INT nmax;
    INT kmax;
    INT lda;
    c64* A;
    f32* SD;
    f32* SE;
    f32* D1;
    f32* D2;
    f32* D3;
    c64* U;
    c64* work;
    f32* rwork;
    INT lwork;
    uint64_t rng_state[4];
} zchkhb2stg_workspace_t;

static zchkhb2stg_workspace_t* g_ws = NULL;

static int group_setup(void** state)
{
    (void)state;

    g_ws = calloc(1, sizeof(zchkhb2stg_workspace_t));
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

    g_ws->A    = malloc((size_t)(lda * nmax + kmax) * sizeof(c64));
    g_ws->SD   = malloc((size_t)nmax * sizeof(f32));
    g_ws->SE   = malloc((size_t)nmax * sizeof(f32));
    g_ws->D1   = malloc((size_t)nmax * sizeof(f32));
    g_ws->D2   = malloc((size_t)nmax * sizeof(f32));
    g_ws->D3   = malloc((size_t)nmax * sizeof(f32));
    g_ws->U    = malloc((size_t)nmax * nmax * sizeof(c64));

    INT maxln = lda > nmax ? lda : nmax;
    g_ws->lwork = (maxln + 1) * nmax;
    if (g_ws->lwork < nmax * nmax + nmax) g_ws->lwork = nmax * nmax + nmax;
    {
        INT lh_need = 4 * nmax;
        if (lh_need < 1) lh_need = 1;
        INT ib_est = 16;
        INT lw_need = (2 * ib_est + 1) * nmax + ib_est;
        INT total_hb2st = lh_need + lw_need;
        if (g_ws->lwork < total_hb2st) g_ws->lwork = total_hb2st;
    }
    g_ws->work  = malloc((size_t)g_ws->lwork * sizeof(c64));
    g_ws->rwork = malloc((size_t)(2 * nmax) * sizeof(f32));

    if (!g_ws->A || !g_ws->SD || !g_ws->SE || !g_ws->D1 ||
        !g_ws->D2 || !g_ws->D3 || !g_ws->U || !g_ws->work ||
        !g_ws->rwork) {
        return -1;
    }

    rng_seed(g_ws->rng_state, 0xE5B2FACEULL);
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
        free(g_ws->rwork);
        free(g_ws);
        g_ws = NULL;
    }
    return 0;
}

static void run_zchkhb2stg_single(zchkhb2stg_params_t* params)
{
    const INT n = NVAL[params->jsize];
    INT k = KK[params->jwidth];
    const INT jtype = params->jtype;
    const INT jt = jtype - 1;

    if (k > n)
        return;

    if (k > n - 1) k = n - 1;
    if (k < 0) k = 0;

    const INT lda   = g_ws->lda;
    const INT ldu   = (g_ws->nmax > 1) ? g_ws->nmax : 1;

    c64* A     = g_ws->A;
    f32*  SD    = g_ws->SD;
    f32*  SE    = g_ws->SE;
    f32*  D1    = g_ws->D1;
    f32*  D2    = g_ws->D2;
    f32*  D3    = g_ws->D3;
    c64* U     = g_ws->U;
    c64* work  = g_ws->work;
    f32*  rwork = g_ws->rwork;
    INT lwork   = g_ws->lwork;
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

        claset("Full", lda, n, CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), A, lda);
        iinfo = 0;

        if (itype == 1) {
            iinfo = 0;

        } else if (itype == 2) {
            for (INT jcol = 0; jcol < n; jcol++)
                A[k + jcol * lda] = CMPLXF(anorm, 0.0f);

        } else if (itype == 4) {
            clatms(n, n, "S", "H", rwork, imode, cond, anorm,
                   0, 0, "Q", &A[k], lda, work, &iinfo, rng);

        } else if (itype == 5) {
            clatms(n, n, "S", "H", rwork, imode, cond, anorm,
                   k, k, "Q", A, lda, work, &iinfo, rng);

        } else if (itype == 7) {
            INT idumma[1] = {0};
            clatmr(n, n, "S", "H", work, 6, 1.0f, CMPLXF(1.0f, 0.0f),
                   "T", "N", work + n, 1, 1.0f,
                   work + 2 * n, 1, 1.0f, "N", idumma, 0, 0,
                   0.0f, anorm, "Q", &A[k], lda, idumma, &iinfo, rng);

        } else if (itype == 8) {
            INT idumma[1] = {0};
            clatmr(n, n, "S", "H", work, 6, 1.0f, CMPLXF(1.0f, 0.0f),
                   "T", "N", work + n, 1, 1.0f,
                   work + 2 * n, 1, 1.0f, "N", idumma, k, k,
                   0.0f, anorm, "Q", A, lda, idumma, &iinfo, rng);

        } else if (itype == 9) {
            clatms(n, n, "S", "P", rwork, imode, cond, anorm,
                   k, k, "Q", A, lda, work + n, &iinfo, rng);

        } else if (itype == 10) {
            if (n > 1 && k < 1) k = 1;
            clatms(n, n, "S", "P", rwork, imode, cond, anorm,
                   1, 1, "Q", &A[k - 1], lda, work, &iinfo, rng);
            for (INT i = 1; i < n; i++) {
                f32 temp1 = cabsf(A[(k - 1) + i * lda]) /
                            sqrtf(cabsf(A[k + (i - 1) * lda] * A[k + i * lda]));
                if (temp1 > 0.5f) {
                    A[(k - 1) + i * lda] = CMPLXF(
                        0.5f * sqrtf(cabsf(A[k + (i - 1) * lda] * A[k + i * lda])),
                        0.0f);
                }
            }

        } else {
            iinfo = 1;
        }

        if (iinfo != 0) {
            snprintf(ctx, sizeof(ctx),
                     "zchkhb2stg n=%d k=%d type=%d: Generator info=%d",
                     n, k, jtype, iinfo);
            set_test_context(ctx);
            assert_info_success(iinfo);
            clear_test_context();
            return;
        }
    }

    clacpy(" ", k + 1, n, A, lda, work, lda);

    ntest = 1;
    chbtrd("V", "U", n, k, work, lda, SD, SE, U, ldu,
           work + lda * n, &iinfo);

    if (iinfo != 0) {
        snprintf(ctx, sizeof(ctx),
                 "zchkhb2stg n=%d k=%d type=%d: CHBTRD(U) info=%d",
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

    chbt21("Upper", n, k, 1, A, lda, SD, SE, U, ldu,
           work, rwork, &result[0]);

    cblas_scopy(n, SD, 1, D1, 1);
    if (n > 0)
        cblas_scopy(n - 1, SE, 1, rwork, 1);

    csteqr("N", n, D1, rwork, NULL, ldu,
           NULL, &iinfo);
    if (iinfo != 0) {
        snprintf(ctx, sizeof(ctx),
                 "zchkhb2stg n=%d k=%d type=%d: CSTEQR(N) info=%d",
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

    slaset("Full", n, 1, 0.0f, 0.0f, SD, n > 0 ? n : 1);
    slaset("Full", n, 1, 0.0f, 0.0f, SE, n > 0 ? n : 1);
    clacpy(" ", k + 1, n, A, lda, U, ldu);
    lh = 4 * n;
    if (lh < 1) lh = 1;
    lw = lwork - lh;
    chetrd_hb2st("N", "N", "U", n, k, U, ldu, SD, SE,
                 work, lh, work + lh, lw, &iinfo);
    cblas_scopy(n, SD, 1, D2, 1);
    if (n > 0)
        cblas_scopy(n - 1, SE, 1, rwork, 1);

    csteqr("N", n, D2, rwork, NULL, ldu,
           NULL, &iinfo);
    if (iinfo != 0) {
        snprintf(ctx, sizeof(ctx),
                 "zchkhb2stg n=%d k=%d type=%d: CSTEQR(N) D2 info=%d",
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

    for (INT jc = 0; jc < n; jc++) {
        INT jrmax = k < n - 1 - jc ? k : n - 1 - jc;
        for (INT jr = 0; jr <= jrmax; jr++)
            A[jr + jc * lda] = conjf(A[k - jr + (jc + jr) * lda]);
    }
    for (INT jc = n - k; jc < n; jc++) {
        if (jc < 0) continue;
        INT jrmin_lim = k < n - 1 - jc ? k : n - 1 - jc;
        for (INT jr = jrmin_lim + 1; jr <= k; jr++)
            A[jr + jc * lda] = CMPLXF(0.0f, 0.0f);
    }

    clacpy(" ", k + 1, n, A, lda, work, lda);

    ntest = 3;
    chbtrd("V", "L", n, k, work, lda, SD, SE, U, ldu,
           work + lda * n, &iinfo);

    if (iinfo != 0) {
        snprintf(ctx, sizeof(ctx),
                 "zchkhb2stg n=%d k=%d type=%d: CHBTRD(L) info=%d",
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

    chbt21("Lower", n, k, 1, A, lda, SD, SE, U, ldu,
           work, rwork, &result[2]);

    slaset("Full", n, 1, 0.0f, 0.0f, SD, n > 0 ? n : 1);
    slaset("Full", n, 1, 0.0f, 0.0f, SE, n > 0 ? n : 1);
    clacpy(" ", k + 1, n, A, lda, U, ldu);
    lh = 4 * n;
    if (lh < 1) lh = 1;
    lw = lwork - lh;
    chetrd_hb2st("N", "N", "L", n, k, U, ldu, SD, SE,
                 work, lh, work + lh, lw, &iinfo);

    cblas_scopy(n, SD, 1, D3, 1);
    if (n > 0)
        cblas_scopy(n - 1, SE, 1, rwork, 1);

    csteqr("N", n, D3, rwork, NULL, ldu,
           NULL, &iinfo);
    if (iinfo != 0) {
        snprintf(ctx, sizeof(ctx),
                 "zchkhb2stg n=%d k=%d type=%d: CSTEQR(N) D3 info=%d",
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

check_results:
    for (INT jr = 0; jr < ntest; jr++) {
        snprintf(ctx, sizeof(ctx), "zchkhb2stg n=%d k=%d type=%d TEST %d",
                 n, k, jtype, jr + 1);
        set_test_context(ctx);
        assert_residual_below(result[jr], THRESH);
    }
    clear_test_context();
}

static void test_zchkhb2stg_case(void** state)
{
    (void)state;
    zchkhb2stg_params_t* params = *state;
    run_zchkhb2stg_single(params);
}

#define MAX_TESTS (NSIZES * NWDTHS * MAXTYP)

static zchkhb2stg_params_t g_params[MAX_TESTS];
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
                zchkhb2stg_params_t* p = &g_params[g_num_tests];
                p->jsize = (INT)is;
                p->jwidth = (INT)iw;
                p->jtype = jtype;
                snprintf(p->name, sizeof(p->name),
                         "zchkhb2stg_n%d_k%d_type%d", n, kval, jtype);

                g_tests[g_num_tests].name = p->name;
                g_tests[g_num_tests].test_func = test_zchkhb2stg_case;
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
    (void)_cmocka_run_group_tests("zchkhb2stg", g_tests, g_num_tests,
                                    group_setup, group_teardown);
    return 0;
}
