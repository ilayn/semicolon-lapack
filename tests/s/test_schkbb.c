/**
 * @file test_schkbb.c
 * @brief Band bidiagonal reduction test driver - port of LAPACK TESTING/EIG/dchkbb.f
 *
 * Tests the reduction of a general real rectangular band matrix to
 * bidiagonal form.
 *
 * SGBBRD factors a general band matrix A as  Q B P* , where * means
 * transpose, B is upper bidiagonal, and Q and P are orthogonal;
 * SGBBRD can also overwrite a given matrix C with Q* C .
 *
 * For each pair of matrix dimensions (M,N) and each selected matrix
 * type, an M by N matrix A and an M by NRHS matrix C are generated.
 * The problem dimensions are as follows
 *    A:          M x N
 *    Q:          M x M
 *    P:          N x N
 *    B:          min(M,N) x min(M,N)
 *    C:          M x NRHS
 *
 * Test ratios (4 total):
 *
 *   (1)  | A - Q B PT | / ( |A| max(M,N) ulp ), PT = P'
 *
 *   (2)  | I - Q' Q | / ( M ulp )
 *
 *   (3)  | I - PT PT' | / ( N ulp )
 *
 *   (4)  | Y - Q' C | / ( |Y| max(M,NRHS) ulp ), where Y = Q' C.
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
#define NRHS   2

/* M,N pairs from dbb.in */
static const INT MVAL[] = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 10, 10, 16, 16};
static const INT NVAL[] = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 10, 16, 10, 16};
#define NSIZES (sizeof(MVAL) / sizeof(MVAL[0]))

/* Bandwidth values from dbb.in */
static const INT KK[] = {0, 1, 2, 3, 16};
#define NWDTHS (sizeof(KK) / sizeof(KK[0]))

/* External function declarations */
/* ===================================================================== */
/* DATA arrays from dchkbb.f lines 407-410                               */
/* ===================================================================== */

/*                    1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 */
static const INT ktype[MAXTYP] = {
    1, 2, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 9, 9, 9
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
    INT jsize;     /* index into MVAL[]/NVAL[] */
    INT jwidth;    /* index into KK[] */
    INT jtype;     /* matrix type 1..15 */
    char name[128];
} dchkbb_params_t;

typedef struct {
    INT mmax;
    INT nmax;
    INT kmax;
    INT mnmax;
    INT lda;
    INT ldab;
    f32* A;
    f32* AB;
    f32* BD;
    f32* BE;
    f32* Q;
    f32* P;
    f32* C;
    f32* CC;
    f32* work;
    INT lwork;
    INT* iwork;
    uint64_t rng_state[4];
} dchkbb_workspace_t;

static dchkbb_workspace_t* g_ws = NULL;

/* ===================================================================== */
/* Group setup / teardown                                                */
/* ===================================================================== */

static int group_setup(void** state)
{
    (void)state;

    g_ws = calloc(1, sizeof(dchkbb_workspace_t));
    if (!g_ws) return -1;

    g_ws->mmax = 1;
    g_ws->nmax = 1;
    for (size_t i = 0; i < NSIZES; i++) {
        if (MVAL[i] > g_ws->mmax) g_ws->mmax = MVAL[i];
        if (NVAL[i] > g_ws->nmax) g_ws->nmax = NVAL[i];
    }
    g_ws->kmax = 0;
    for (size_t i = 0; i < NWDTHS; i++) {
        if (KK[i] > g_ws->kmax) g_ws->kmax = KK[i];
    }
    g_ws->mnmax = (g_ws->mmax > g_ws->nmax) ? g_ws->mmax : g_ws->nmax;

    INT mmax  = g_ws->mmax;
    INT nmax  = g_ws->nmax;
    INT mnmax = g_ws->mnmax;
    INT kmax  = g_ws->kmax;

    g_ws->lda = (mmax > nmax) ? mmax : nmax;
    if (g_ws->lda < 1) g_ws->lda = 1;
    INT lda = g_ws->lda;

    g_ws->ldab = 2 * kmax + 1;
    if (g_ws->ldab < 2) g_ws->ldab = 2;
    INT ldab = g_ws->ldab;

    g_ws->A    = malloc((size_t)lda * nmax * sizeof(f32));
    g_ws->AB   = malloc((size_t)ldab * nmax * sizeof(f32));
    g_ws->BD   = malloc((size_t)mnmax * sizeof(f32));
    g_ws->BE   = malloc((size_t)mnmax * sizeof(f32));
    g_ws->Q    = malloc((size_t)mmax * mmax * sizeof(f32));
    g_ws->P    = malloc((size_t)nmax * nmax * sizeof(f32));
    g_ws->C    = malloc((size_t)lda * NRHS * sizeof(f32));
    g_ws->CC   = malloc((size_t)lda * NRHS * sizeof(f32));

    /* Workspace: max(lda+1, nmax+1) * nmax, plus room for generators */
    g_ws->lwork = ((lda > nmax ? lda : nmax) + 1) * nmax;
    if (g_ws->lwork < 4 * mnmax) g_ws->lwork = 4 * mnmax;
    g_ws->work  = malloc((size_t)g_ws->lwork * sizeof(f32));
    g_ws->iwork = malloc((size_t)mnmax * sizeof(INT));

    if (!g_ws->A || !g_ws->AB || !g_ws->BD || !g_ws->BE ||
        !g_ws->Q || !g_ws->P || !g_ws->C || !g_ws->CC ||
        !g_ws->work || !g_ws->iwork) {
        return -1;
    }

    rng_seed(g_ws->rng_state, 0xDBB15ACEULL);
    return 0;
}

static int group_teardown(void** state)
{
    (void)state;

    if (g_ws) {
        free(g_ws->A);
        free(g_ws->AB);
        free(g_ws->BD);
        free(g_ws->BE);
        free(g_ws->Q);
        free(g_ws->P);
        free(g_ws->C);
        free(g_ws->CC);
        free(g_ws->work);
        free(g_ws->iwork);
        free(g_ws);
        g_ws = NULL;
    }
    return 0;
}

/* ===================================================================== */
/* Single test case                                                      */
/* ===================================================================== */

static void run_dchkbb_single(dchkbb_params_t* params)
{
    const INT m = MVAL[params->jsize];
    const INT n = NVAL[params->jsize];
    const INT k = KK[params->jwidth];
    const INT jtype = params->jtype;
    const INT jt = jtype - 1;

    /* Skip if k >= m AND k >= n */
    if (k >= m && k >= n)
        return;

    const INT mnmin = (m < n) ? m : n;
    INT kl = (m - 1 < k) ? m - 1 : k;
    if (kl < 0) kl = 0;
    INT ku = (n - 1 < k) ? n - 1 : k;
    if (ku < 0) ku = 0;

    INT mxn = m;
    if (n > mxn) mxn = n;
    if (mxn < 1) mxn = 1;
    const f32 amninv = 1.0f / (f32)mxn;

    const INT lda   = g_ws->lda;
    const INT ldab  = g_ws->ldab;
    const INT ldq   = (g_ws->mmax > 1) ? g_ws->mmax : 1;
    const INT ldp   = (g_ws->nmax > 1) ? g_ws->nmax : 1;
    const INT ldc   = lda;
    const INT lwork = g_ws->lwork;

    f32* A     = g_ws->A;
    f32* AB    = g_ws->AB;
    f32* BD    = g_ws->BD;
    f32* BE    = g_ws->BE;
    f32* Q     = g_ws->Q;
    f32* P     = g_ws->P;
    f32* C     = g_ws->C;
    f32* CC    = g_ws->CC;
    f32* work  = g_ws->work;
    INT* iwork = g_ws->iwork;
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

    /* ----------------------------------------------------------- */
    /* Compute "A"                                                 */
    /* ----------------------------------------------------------- */

    if (jtype <= MAXTYP) {
        INT itype = ktype[jt];
        INT imode = kmode[jt];

        f32 anorm;
        switch (kmagn[jt]) {
            case 2:  anorm = (rtovfl * ulp) * amninv; break;
            case 3:  anorm = rtunfl * (f32)(m > n ? m : n) * ulpinv; break;
            default: anorm = 1.0f; break;
        }
        f32 cond = ulpinv;

        slaset("Full", lda, n, 0.0f, 0.0f, A, lda);
        slaset("Full", ldab, n, 0.0f, 0.0f, AB, ldab);
        iinfo = 0;

        if (itype == 1) {
            /* Zero matrix */
            iinfo = 0;

        } else if (itype == 2) {
            /* Identity */
            for (INT jcol = 0; jcol < mnmin; jcol++)
                A[jcol + jcol * lda] = anorm;

        } else if (itype == 4) {
            /* Diagonal matrix, singular values specified */
            slatms(m, n, "S", "N", work, imode, cond, anorm,
                   0, 0, "N", A, lda, work + m, &iinfo, rng);

        } else if (itype == 6) {
            /* Nonhermitian, singular values specified */
            slatms(m, n, "S", "N", work, imode, cond, anorm,
                   kl, ku, "N", A, lda, work + m, &iinfo, rng);

        } else if (itype == 9) {
            /* Nonhermitian, random entries */
            INT idumma[1] = {0};
            slatmr(m, n, "S", "N", work, 6, 1.0f, 1.0f,
                   "T", "N", work + n, 1, 1.0f,
                   work + 2 * n, 1, 1.0f, "N", idumma, kl, ku,
                   0.0f, anorm, "N", A, lda, iwork, &iinfo, rng);

        } else {
            iinfo = 1;
        }

        if (iinfo != 0) {
            snprintf(ctx, sizeof(ctx),
                     "dchkbb m=%d n=%d k=%d type=%d: Generator info=%d",
                     m, n, k, jtype, iinfo);
            set_test_context(ctx);
            assert_info_success(iinfo);
            clear_test_context();
            return;
        }
    }

    /* Generate Right-Hand Side */
    {
        INT idumma[1] = {0};
        slatmr(m, NRHS, "S", "N", work, 6, 1.0f, 1.0f,
               "T", "N", work + m, 1, 1.0f,
               work + 2 * m, 1, 1.0f, "N", idumma, m, NRHS,
               0.0f, 1.0f, "NO", C, ldc, iwork, &iinfo, rng);
    }

    if (iinfo != 0) {
        snprintf(ctx, sizeof(ctx),
                 "dchkbb m=%d n=%d k=%d type=%d: RHS Generator info=%d",
                 m, n, k, jtype, iinfo);
        set_test_context(ctx);
        assert_info_success(iinfo);
        clear_test_context();
        return;
    }

    /* ----------------------------------------------------------- */
    /* Copy A to band storage                                      */
    /* ----------------------------------------------------------- */

    for (INT j = 0; j < n; j++) {
        INT imin = j - ku;
        if (imin < 0) imin = 0;
        INT imax = j + kl;
        if (imax > m - 1) imax = m - 1;
        for (INT i = imin; i <= imax; i++)
            AB[ku + i - j + j * ldab] = A[i + j * lda];
    }

    /* Copy C */
    slacpy("Full", m, NRHS, C, ldc, CC, ldc);

    /* ----------------------------------------------------------- */
    /* Call SGBBRD                                                 */
    /* ----------------------------------------------------------- */

    sgbbrd("B", m, n, NRHS, kl, ku, AB, ldab, BD, BE,
           Q, ldq, P, ldp, CC, ldc, work, &iinfo);

    if (iinfo != 0) {
        snprintf(ctx, sizeof(ctx),
                 "dchkbb m=%d n=%d k=%d type=%d: SGBBRD info=%d",
                 m, n, k, jtype, iinfo);
        set_test_context(ctx);
        if (iinfo < 0) {
            assert_info_success(iinfo);
            clear_test_context();
            return;
        } else {
            result[0] = ulpinv;
            ntest = 1;
            goto check_results;
        }
    }

    /* ----------------------------------------------------------- */
    /* Test 1:  Check the decomposition A := Q * B * P'            */
    /* Test 2:  Check the orthogonality of Q                       */
    /* Test 3:  Check the orthogonality of P                       */
    /* Test 4:  Check the computation of Q' * C                    */
    /* ----------------------------------------------------------- */

    sbdt01(m, n, -1, A, lda, Q, ldq, BD, BE, P, ldp,
           work, &result[0]);
    sort01("Columns", m, m, Q, ldq, work, lwork, &result[1]);
    sort01("Rows", n, n, P, ldp, work, lwork, &result[2]);
    sbdt02(m, NRHS, C, ldc, CC, ldc, Q, ldq, work, &result[3]);

    ntest = 4;

check_results:
    for (INT jr = 0; jr < ntest; jr++) {
        snprintf(ctx, sizeof(ctx), "dchkbb m=%d n=%d k=%d type=%d TEST %d",
                 m, n, k, jtype, jr + 1);
        set_test_context(ctx);
        assert_residual_below(result[jr], THRESH);
    }
    clear_test_context();
}

/* ===================================================================== */
/* CMocka wrapper                                                        */
/* ===================================================================== */

static void test_dchkbb_case(void** state)
{
    dchkbb_params_t* params = *state;
    run_dchkbb_single(params);
}

/* ===================================================================== */
/* Build test array                                                      */
/* ===================================================================== */

#define MAX_TESTS (NSIZES * NWDTHS * MAXTYP)

static dchkbb_params_t g_params[MAX_TESTS];
static struct CMUnitTest g_tests[MAX_TESTS];
static INT g_num_tests = 0;

static void build_test_array(void)
{
    g_num_tests = 0;

    for (size_t is = 0; is < NSIZES; is++) {
        INT m = MVAL[is];
        INT n = NVAL[is];
        for (size_t iw = 0; iw < NWDTHS; iw++) {
            INT kval = KK[iw];
            if (kval >= m && kval >= n)
                continue;
            for (INT jtype = 1; jtype <= MAXTYP; jtype++) {
                dchkbb_params_t* p = &g_params[g_num_tests];
                p->jsize = (INT)is;
                p->jwidth = (INT)iw;
                p->jtype = jtype;
                snprintf(p->name, sizeof(p->name),
                         "dchkbb_m%d_n%d_k%d_type%d", m, n, kval, jtype);

                g_tests[g_num_tests].name = p->name;
                g_tests[g_num_tests].test_func = test_dchkbb_case;
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
    return _cmocka_run_group_tests("dchkbb", g_tests, g_num_tests,
                                    group_setup, group_teardown);
}
