/**
 * @file test_zchkhb.c
 * @brief Hermitian band tridiagonal reduction test driver - port of LAPACK
 *        TESTING/EIG/zchkhb.f
 *
 * Tests the reduction of a Hermitian band matrix to tridiagonal form.
 *
 * ZHBTRD factors a Hermitian band matrix A as  U S U* , where * means
 * conjugate transpose, S is symmetric tridiagonal, and U is unitary.
 * ZHBTRD can use either just the lower or just the upper triangle
 * of A; ZCHKHB checks both cases.
 *
 * For each size ("n"), each bandwidth ("k") less than or equal to "n",
 * and each type of matrix, one matrix will be generated and used to test
 * the Hermitian banded reduction routine.
 *
 * Test ratios (4 total):
 *
 *   (1)  | A - U S U* | / ( |A| n ulp )  UPLO='U'
 *
 *   (2)  | I - U U* | / ( n ulp )         UPLO='U'
 *
 *   (3)  | A - U S U* | / ( |A| n ulp )  UPLO='L'
 *
 *   (4)  | I - U U* | / ( n ulp )         UPLO='L'
 *
 * Matrix types: 15 types (MAXTYP = 15)
 */

#include "test_harness.h"
#include "verify.h"
#include "test_rng.h"
#include <math.h>
#include <string.h>

#define THRESH 20.0
#define MAXTYP 15
#define NTEST  4

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
} zchkhb_params_t;

typedef struct {
    INT nmax;
    INT kmax;
    INT lda;
    c128* A;
    f64* SD;
    f64* SE;
    c128* U;
    c128* work;
    f64* rwork;
    INT lwork;
    uint64_t rng_state[4];
} zchkhb_workspace_t;

static zchkhb_workspace_t* g_ws = NULL;

static int group_setup(void** state)
{
    (void)state;

    g_ws = calloc(1, sizeof(zchkhb_workspace_t));
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

    g_ws->A    = malloc((size_t)(lda * nmax + kmax) * sizeof(c128));
    g_ws->SD   = malloc((size_t)nmax * sizeof(f64));
    g_ws->SE   = malloc((size_t)nmax * sizeof(f64));
    g_ws->U    = malloc((size_t)nmax * nmax * sizeof(c128));

    INT maxln = lda > nmax ? lda : nmax;
    g_ws->lwork = (maxln + 1) * nmax;
    if (g_ws->lwork < nmax * nmax + nmax) g_ws->lwork = nmax * nmax + nmax;
    g_ws->work  = malloc((size_t)g_ws->lwork * sizeof(c128));
    g_ws->rwork = malloc((size_t)nmax * sizeof(f64));

    if (!g_ws->A || !g_ws->SD || !g_ws->SE || !g_ws->U ||
        !g_ws->work || !g_ws->rwork) {
        return -1;
    }

    rng_seed(g_ws->rng_state, 0xE5B15ACEULL);
    return 0;
}

static int group_teardown(void** state)
{
    (void)state;

    if (g_ws) {
        free(g_ws->A);
        free(g_ws->SD);
        free(g_ws->SE);
        free(g_ws->U);
        free(g_ws->work);
        free(g_ws->rwork);
        free(g_ws);
        g_ws = NULL;
    }
    return 0;
}

static void run_zchkhb_single(zchkhb_params_t* params)
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

    c128* A     = g_ws->A;
    f64*  SD    = g_ws->SD;
    f64*  SE    = g_ws->SE;
    c128* U     = g_ws->U;
    c128* work  = g_ws->work;
    f64*  rwork = g_ws->rwork;
    uint64_t* rng = g_ws->rng_state;

    const f64 unfl    = dlamch("S");
    const f64 ovfl    = 1.0 / unfl;
    const f64 ulp     = dlamch("P");
    const f64 ulpinv  = 1.0 / ulp;
    const f64 rtunfl  = sqrt(unfl);
    const f64 rtovfl  = sqrt(ovfl);

    f64 result[NTEST];
    for (INT i = 0; i < NTEST; i++)
        result[i] = 0.0;

    INT iinfo = 0;
    INT ntest = 0;
    char ctx[256];

    const f64 aninv = 1.0 / (f64)(n > 1 ? n : 1);

    if (jtype <= MAXTYP) {
        INT itype = ktype[jt];
        INT imode = kmode[jt];

        f64 anorm;
        switch (kmagn[jt]) {
            case 2:  anorm = (rtovfl * ulp) * aninv; break;
            case 3:  anorm = rtunfl * (f64)n * ulpinv; break;
            default: anorm = 1.0; break;
        }

        f64 cond;
        if (jtype <= 15) {
            cond = ulpinv;
        } else {
            cond = ulpinv * aninv / 10.0;
        }

        zlaset("Full", lda, n, CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), A, lda);
        iinfo = 0;

        if (itype == 1) {
            iinfo = 0;

        } else if (itype == 2) {
            for (INT jcol = 0; jcol < n; jcol++)
                A[k + jcol * lda] = CMPLX(anorm, 0.0);

        } else if (itype == 4) {
            zlatms(n, n, "S", "H", rwork, imode, cond, anorm,
                   0, 0, "Q", &A[k], lda, work, &iinfo, rng);

        } else if (itype == 5) {
            zlatms(n, n, "S", "H", rwork, imode, cond, anorm,
                   k, k, "Q", A, lda, work, &iinfo, rng);

        } else if (itype == 7) {
            INT idumma[1] = {0};
            zlatmr(n, n, "S", "H", work, 6, 1.0, CMPLX(1.0, 0.0),
                   "T", "N", work + n, 1, 1.0,
                   work + 2 * n, 1, 1.0, "N", idumma, 0, 0,
                   0.0, anorm, "Q", &A[k], lda, idumma, &iinfo, rng);

        } else if (itype == 8) {
            INT idumma[1] = {0};
            zlatmr(n, n, "S", "H", work, 6, 1.0, CMPLX(1.0, 0.0),
                   "T", "N", work + n, 1, 1.0,
                   work + 2 * n, 1, 1.0, "N", idumma, k, k,
                   0.0, anorm, "Q", A, lda, idumma, &iinfo, rng);

        } else if (itype == 9) {
            zlatms(n, n, "S", "P", rwork, imode, cond, anorm,
                   k, k, "Q", A, lda, work + n, &iinfo, rng);

        } else if (itype == 10) {
            if (n > 1 && k < 1) k = 1;
            zlatms(n, n, "S", "P", rwork, imode, cond, anorm,
                   1, 1, "Q", &A[k - 1], lda, work, &iinfo, rng);
            for (INT i = 1; i < n; i++) {
                f64 temp1 = cabs(A[(k - 1) + i * lda]) /
                            sqrt(cabs(A[k + (i - 1) * lda] * A[k + i * lda]));
                if (temp1 > 0.5) {
                    A[(k - 1) + i * lda] = CMPLX(
                        0.5 * sqrt(cabs(A[k + (i - 1) * lda] * A[k + i * lda])),
                        0.0);
                }
            }

        } else {
            iinfo = 1;
        }

        if (iinfo != 0) {
            snprintf(ctx, sizeof(ctx),
                     "zchkhb n=%d k=%d type=%d: Generator info=%d",
                     n, k, jtype, iinfo);
            set_test_context(ctx);
            assert_info_success(iinfo);
            clear_test_context();
            return;
        }
    }

    zlacpy(" ", k + 1, n, A, lda, work, lda);

    ntest = 1;
    zhbtrd("V", "U", n, k, work, lda, SD, SE, U, ldu,
           work + lda * n, &iinfo);

    if (iinfo != 0) {
        snprintf(ctx, sizeof(ctx),
                 "zchkhb n=%d k=%d type=%d: ZHBTRD(U) info=%d",
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

    zhbt21("Upper", n, k, 1, A, lda, SD, SE, U, ldu,
           work, rwork, &result[0]);

    for (INT jc = 0; jc < n; jc++) {
        INT jrmax = k < n - 1 - jc ? k : n - 1 - jc;
        for (INT jr = 0; jr <= jrmax; jr++)
            A[jr + jc * lda] = conj(A[k - jr + (jc + jr) * lda]);
    }
    for (INT jc = n - k; jc < n; jc++) {
        if (jc < 0) continue;
        INT jrmin_lim = k < n - 1 - jc ? k : n - 1 - jc;
        for (INT jr = jrmin_lim + 1; jr <= k; jr++)
            A[jr + jc * lda] = CMPLX(0.0, 0.0);
    }

    zlacpy(" ", k + 1, n, A, lda, work, lda);

    ntest = 3;
    zhbtrd("V", "L", n, k, work, lda, SD, SE, U, ldu,
           work + lda * n, &iinfo);

    if (iinfo != 0) {
        snprintf(ctx, sizeof(ctx),
                 "zchkhb n=%d k=%d type=%d: ZHBTRD(L) info=%d",
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

    zhbt21("Lower", n, k, 1, A, lda, SD, SE, U, ldu,
           work, rwork, &result[2]);

check_results:
    for (INT jr = 0; jr < ntest; jr++) {
        snprintf(ctx, sizeof(ctx), "zchkhb n=%d k=%d type=%d TEST %d",
                 n, k, jtype, jr + 1);
        set_test_context(ctx);
        assert_residual_below(result[jr], THRESH);
    }
    clear_test_context();
}

static void test_zchkhb_case(void** state)
{
    (void)state;
    zchkhb_params_t* params = *state;
    run_zchkhb_single(params);
}

#define MAX_TESTS (NSIZES * NWDTHS * MAXTYP)

static zchkhb_params_t g_params[MAX_TESTS];
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
                zchkhb_params_t* p = &g_params[g_num_tests];
                p->jsize = (INT)is;
                p->jwidth = (INT)iw;
                p->jtype = jtype;
                snprintf(p->name, sizeof(p->name),
                         "zchkhb_n%d_k%d_type%d", n, kval, jtype);

                g_tests[g_num_tests].name = p->name;
                g_tests[g_num_tests].test_func = test_zchkhb_case;
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
    return _cmocka_run_group_tests("zchkhb", g_tests, g_num_tests,
                                    group_setup, group_teardown);
}
