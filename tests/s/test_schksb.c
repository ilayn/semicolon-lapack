/**
 * @file test_schksb.c
 * @brief Symmetric band tridiagonal reduction test driver - port of LAPACK
 *        TESTING/EIG/dchksb.f
 *
 * Tests the reduction of a symmetric band matrix to tridiagonal form.
 *
 * SSBTRD factors a symmetric band matrix A as  U S U' , where ' means
 * transpose, S is symmetric tridiagonal, and U is orthogonal.
 * SSBTRD can use either just the lower or just the upper triangle
 * of A; DCHKSB checks both cases.
 *
 * For each size ("n"), each bandwidth ("k") less than or equal to "n",
 * and each type of matrix, one matrix will be generated and used to test
 * the symmetric banded reduction routine.
 *
 * Test ratios (4 total):
 *
 *   (1)  | A - U S U' | / ( |A| n ulp )  UPLO='U'
 *
 *   (2)  | I - U U' | / ( n ulp )         UPLO='U'
 *
 *   (3)  | A - U S U' | / ( |A| n ulp )  UPLO='L'
 *
 *   (4)  | I - U U' | / ( n ulp )         UPLO='L'
 *
 * Matrix types: 15 types (MAXTYP = 15)
 */

#include "test_harness.h"
#include "verify.h"
#include "test_rng.h"
#include <math.h>
#include <string.h>

#define THRESH 20.0f
#define MAXTYP 15
#define NTEST  4

/* Matrix sizes */
static const INT NVAL[] = {0, 1, 2, 3, 5, 10, 20};
#define NSIZES (sizeof(NVAL) / sizeof(NVAL[0]))

/* Bandwidth values */
static const INT KK[] = {0, 1, 2, 3, 10};
#define NWDTHS (sizeof(KK) / sizeof(KK[0]))

/* External function declarations */
/* ===================================================================== */
/* DATA arrays from dchksb.f lines 346-350                               */
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
} dchksb_params_t;

typedef struct {
    INT nmax;
    INT kmax;
    INT lda;
    f32* A;
    f32* SD;
    f32* SE;
    f32* U;
    f32* work;
    INT lwork;
    uint64_t rng_state[4];
} dchksb_workspace_t;

static dchksb_workspace_t* g_ws = NULL;

/* ===================================================================== */
/* Group setup / teardown                                                */
/* ===================================================================== */

static int group_setup(void** state)
{
    (void)state;

    g_ws = calloc(1, sizeof(dchksb_workspace_t));
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
    g_ws->U    = malloc((size_t)nmax * nmax * sizeof(f32));

    INT maxln = lda > nmax ? lda : nmax;
    g_ws->lwork = (maxln + 1) * nmax;
    /* ssbt21 needs n*n + n workspace */
    if (g_ws->lwork < nmax * nmax + nmax) g_ws->lwork = nmax * nmax + nmax;
    g_ws->work = malloc((size_t)g_ws->lwork * sizeof(f32));

    if (!g_ws->A || !g_ws->SD || !g_ws->SE || !g_ws->U || !g_ws->work) {
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

static void run_dchksb_single(dchksb_params_t* params)
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
    f32* U     = g_ws->U;
    f32* work  = g_ws->work;
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
                     "dchksb n=%d k=%d type=%d: Generator info=%d",
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
                 "dchksb n=%d k=%d type=%d: SSBTRD(U) info=%d",
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
                 "dchksb n=%d k=%d type=%d: SSBTRD(L) info=%d",
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
    /* End of Loop -- Check for RESULT(j) > THRESH                 */
    /* ----------------------------------------------------------- */

check_results:
    for (INT jr = 0; jr < ntest; jr++) {
        snprintf(ctx, sizeof(ctx), "dchksb n=%d k=%d type=%d TEST %d",
                 n, k, jtype, jr + 1);
        set_test_context(ctx);
        assert_residual_below(result[jr], THRESH);
    }
    clear_test_context();
}

/* ===================================================================== */
/* CMocka wrapper                                                        */
/* ===================================================================== */

static void test_dchksb_case(void** state)
{
    (void)state;
    dchksb_params_t* params = *state;
    run_dchksb_single(params);
}

/* ===================================================================== */
/* Build test array                                                      */
/* ===================================================================== */

#define MAX_TESTS (NSIZES * NWDTHS * MAXTYP)

static dchksb_params_t g_params[MAX_TESTS];
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
                dchksb_params_t* p = &g_params[g_num_tests];
                p->jsize = (INT)is;
                p->jwidth = (INT)iw;
                p->jtype = jtype;
                snprintf(p->name, sizeof(p->name),
                         "dchksb_n%d_k%d_type%d", n, kval, jtype);

                g_tests[g_num_tests].name = p->name;
                g_tests[g_num_tests].test_func = test_dchksb_case;
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
    return _cmocka_run_group_tests("dchksb", g_tests, g_num_tests,
                                    group_setup, group_teardown);
}
