/**
 * @file test_dchksb.c
 * @brief Symmetric band tridiagonal reduction test driver - port of LAPACK
 *        TESTING/EIG/dchksb.f
 *
 * Tests the reduction of a symmetric band matrix to tridiagonal form.
 *
 * DSBTRD factors a symmetric band matrix A as  U S U' , where ' means
 * transpose, S is symmetric tridiagonal, and U is orthogonal.
 * DSBTRD can use either just the lower or just the upper triangle
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

#define THRESH 20.0
#define MAXTYP 15
#define NTEST  4

/* Matrix sizes */
static const int NVAL[] = {0, 1, 2, 3, 5, 10, 20};
#define NSIZES (sizeof(NVAL) / sizeof(NVAL[0]))

/* Bandwidth values */
static const int KK[] = {0, 1, 2, 3, 10};
#define NWDTHS (sizeof(KK) / sizeof(KK[0]))

/* External function declarations */
extern f64  dlamch(const char* cmach);
extern void dsbtrd(const char* vect, const char* uplo, const int n,
                   const int kd, f64* restrict AB, const int ldab,
                   f64* restrict D, f64* restrict E,
                   f64* restrict Q, const int ldq,
                   f64* restrict work, int* info);
extern void dlacpy(const char* uplo, const int m, const int n,
                   const f64* A, const int lda, f64* B, const int ldb);
extern void dlaset(const char* uplo, const int m, const int n,
                   const f64 alpha, const f64 beta, f64* A, const int lda);

/* ===================================================================== */
/* DATA arrays from dchksb.f lines 346-350                               */
/* ===================================================================== */

/*                    1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 */
static const int ktype[MAXTYP] = {
    1, 2, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 8, 8, 8
};
static const int kmagn[MAXTYP] = {
    1, 1, 1, 1, 1, 2, 3, 1, 1, 1, 2, 3, 1, 2, 3
};
static const int kmode[MAXTYP] = {
    0, 0, 4, 3, 1, 4, 4, 4, 3, 1, 4, 4, 0, 0, 0
};

/* ===================================================================== */
/* Test parameters and workspace                                         */
/* ===================================================================== */

typedef struct {
    int jsize;     /* index into NVAL[] */
    int jwidth;    /* index into KK[] */
    int jtype;     /* matrix type 1..15 */
    char name[128];
} dchksb_params_t;

typedef struct {
    int nmax;
    int kmax;
    int lda;
    f64* A;
    f64* SD;
    f64* SE;
    f64* U;
    f64* work;
    int lwork;
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
    int nmax_m1 = g_ws->nmax - 1;
    if (nmax_m1 < g_ws->kmax) g_ws->kmax = nmax_m1;
    if (g_ws->kmax < 0) g_ws->kmax = 0;

    int nmax = g_ws->nmax;
    int kmax = g_ws->kmax;

    g_ws->lda = kmax + 1;
    if (g_ws->lda < 2) g_ws->lda = 2;
    int lda = g_ws->lda;

    /* Extra kmax room because ITYPE=4,7 pass &A[k] to dlatms/dlatmr,
       which internally zeroes lda*n elements from that offset */
    g_ws->A    = malloc((size_t)(lda * nmax + kmax) * sizeof(f64));
    g_ws->SD   = malloc((size_t)nmax * sizeof(f64));
    g_ws->SE   = malloc((size_t)nmax * sizeof(f64));
    g_ws->U    = malloc((size_t)nmax * nmax * sizeof(f64));

    int maxln = lda > nmax ? lda : nmax;
    g_ws->lwork = (maxln + 1) * nmax;
    /* dsbt21 needs n*n + n workspace */
    if (g_ws->lwork < nmax * nmax + nmax) g_ws->lwork = nmax * nmax + nmax;
    g_ws->work = malloc((size_t)g_ws->lwork * sizeof(f64));

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
    const int n = NVAL[params->jsize];
    int k = KK[params->jwidth];
    const int jtype = params->jtype;
    const int jt = jtype - 1;

    if (k > n)
        return;

    /* Clamp k */
    if (k > n - 1) k = n - 1;
    if (k < 0) k = 0;

    const int lda   = g_ws->lda;
    const int ldu   = (g_ws->nmax > 1) ? g_ws->nmax : 1;

    f64* A     = g_ws->A;
    f64* SD    = g_ws->SD;
    f64* SE    = g_ws->SE;
    f64* U     = g_ws->U;
    f64* work  = g_ws->work;
    uint64_t* rng = g_ws->rng_state;

    const f64 unfl    = dlamch("S");
    const f64 ovfl    = 1.0 / unfl;
    const f64 ulp     = dlamch("P");
    const f64 ulpinv  = 1.0 / ulp;
    const f64 rtunfl  = sqrt(unfl);
    const f64 rtovfl  = sqrt(ovfl);

    f64 result[NTEST];
    for (int i = 0; i < NTEST; i++)
        result[i] = 0.0;

    int iinfo = 0;
    int ntest = 0;
    char ctx[256];

    const f64 aninv = 1.0 / (f64)(n > 1 ? n : 1);

    /* ----------------------------------------------------------- */
    /* Compute "A"                                                 */
    /* Store as "Upper"; later, we will copy to other format.      */
    /* ----------------------------------------------------------- */

    if (jtype <= MAXTYP) {
        int itype = ktype[jt];
        int imode = kmode[jt];

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

        dlaset("Full", lda, n, 0.0, 0.0, A, lda);
        iinfo = 0;

        if (itype == 1) {
            /* Zero matrix */
            iinfo = 0;

        } else if (itype == 2) {
            /* Identity */
            for (int jcol = 0; jcol < n; jcol++)
                A[k + jcol * lda] = anorm;

        } else if (itype == 4) {
            /* Diagonal Matrix, [Eigen]values Specified */
            dlatms(n, n, "S", "S", work, imode, cond, anorm,
                   0, 0, "Q", &A[k], lda, work + n, &iinfo, rng);

        } else if (itype == 5) {
            /* Symmetric, eigenvalues specified */
            dlatms(n, n, "S", "S", work, imode, cond, anorm,
                   k, k, "Q", A, lda, work + n, &iinfo, rng);

        } else if (itype == 7) {
            /* Diagonal, random eigenvalues */
            int idumma[1] = {0};
            dlatmr(n, n, "S", "S", work, 6, 1.0, 1.0,
                   "T", "N", work + n, 1, 1.0,
                   work + 2 * n, 1, 1.0, "N", idumma, 0, 0,
                   0.0, anorm, "Q", &A[k], lda, idumma, &iinfo, rng);

        } else if (itype == 8) {
            /* Symmetric, random eigenvalues */
            int idumma[1] = {0};
            dlatmr(n, n, "S", "S", work, 6, 1.0, 1.0,
                   "T", "N", work + n, 1, 1.0,
                   work + 2 * n, 1, 1.0, "N", idumma, k, k,
                   0.0, anorm, "Q", A, lda, idumma, &iinfo, rng);

        } else if (itype == 9) {
            /* Positive definite, eigenvalues specified */
            dlatms(n, n, "S", "P", work, imode, cond, anorm,
                   k, k, "Q", A, lda, work + n, &iinfo, rng);

        } else if (itype == 10) {
            /* Positive definite tridiagonal, eigenvalues specified */
            if (n > 1 && k < 1) k = 1;
            dlatms(n, n, "S", "P", work, imode, cond, anorm,
                   1, 1, "Q", &A[k - 1], lda, work + n, &iinfo, rng);
            for (int i = 1; i < n; i++) {
                f64 temp1 = fabs(A[(k - 1) + i * lda]) /
                            sqrt(fabs(A[k + (i - 1) * lda] * A[k + i * lda]));
                if (temp1 > 0.5) {
                    A[(k - 1) + i * lda] = 0.5 *
                        sqrt(fabs(A[k + (i - 1) * lda] * A[k + i * lda]));
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
    /* Call DSBTRD to compute S and U from upper triangle           */
    /* ----------------------------------------------------------- */

    dlacpy(" ", k + 1, n, A, lda, work, lda);

    ntest = 1;
    dsbtrd("V", "U", n, k, work, lda, SD, SE, U, ldu,
           work + lda * n, &iinfo);

    if (iinfo != 0) {
        snprintf(ctx, sizeof(ctx),
                 "dchksb n=%d k=%d type=%d: DSBTRD(U) info=%d",
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
    dsbt21("Upper", n, k, 1, A, lda, SD, SE, U, ldu,
           work, &result[0]);

    /* ----------------------------------------------------------- */
    /* Convert A from Upper-Triangle-Only storage to                */
    /* Lower-Triangle-Only storage.                                 */
    /* ----------------------------------------------------------- */

    for (int jc = 0; jc < n; jc++) {
        int jrmax = k < n - 1 - jc ? k : n - 1 - jc;
        for (int jr = 0; jr <= jrmax; jr++)
            A[jr + jc * lda] = A[k - jr + (jc + jr) * lda];
    }
    for (int jc = n - k; jc < n; jc++) {
        if (jc < 0) continue;
        int jrmin_lim = k < n - 1 - jc ? k : n - 1 - jc;
        for (int jr = jrmin_lim + 1; jr <= k; jr++)
            A[jr + jc * lda] = 0.0;
    }

    /* ----------------------------------------------------------- */
    /* Call DSBTRD to compute S and U from lower triangle           */
    /* ----------------------------------------------------------- */

    dlacpy(" ", k + 1, n, A, lda, work, lda);

    ntest = 3;
    dsbtrd("V", "L", n, k, work, lda, SD, SE, U, ldu,
           work + lda * n, &iinfo);

    if (iinfo != 0) {
        snprintf(ctx, sizeof(ctx),
                 "dchksb n=%d k=%d type=%d: DSBTRD(L) info=%d",
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
    dsbt21("Lower", n, k, 1, A, lda, SD, SE, U, ldu,
           work, &result[2]);

    /* ----------------------------------------------------------- */
    /* End of Loop -- Check for RESULT(j) > THRESH                 */
    /* ----------------------------------------------------------- */

check_results:
    for (int jr = 0; jr < ntest; jr++) {
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
static int g_num_tests = 0;

static void build_test_array(void)
{
    g_num_tests = 0;

    for (size_t is = 0; is < NSIZES; is++) {
        int n = NVAL[is];
        for (size_t iw = 0; iw < NWDTHS; iw++) {
            int kval = KK[iw];
            if (kval > n)
                continue;
            for (int jtype = 1; jtype <= MAXTYP; jtype++) {
                dchksb_params_t* p = &g_params[g_num_tests];
                p->jsize = (int)is;
                p->jwidth = (int)iw;
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
