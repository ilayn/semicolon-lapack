/**
 * @file test_zdrges.c
 * @brief Generalized Schur form driver test - port of LAPACK TESTING/EIG/zdrges.f
 *
 * Tests the nonsymmetric generalized eigenvalue (Schur form) driver ZGGES.
 *
 * Each (n, jtype) combination is registered as a separate CMocka test,
 * providing pytest-like parameterized test behavior with clear failure isolation.
 *
 * Test ratios (13 total):
 * Without ordering:
 *                             H
 *   (1)  | A - Q S Z  | / ( |A| n ulp )
 *                             H
 *   (2)  | B - Q T Z  | / ( |B| n ulp )
 *                H
 *   (3)  | I - QQ  | / ( n ulp )
 *                H
 *   (4)  | I - ZZ  | / ( n ulp )
 *   (5)  A is in Schur form S
 *   (6)  difference between (alpha,beta) and diagonals of (S,T)
 *
 * With ordering:
 *                                         H
 *   (7)  | (A,B) - Q (S,T) Z  | / ( |(A,B)| n ulp )
 *                H
 *   (8)  | I - QQ  | / ( n ulp )
 *                H
 *   (9)  | I - ZZ  | / ( n ulp )
 *   (10) A is in Schur form S
 *   (11) difference between (alpha,beta) and diagonals of (S,T)
 *   (12) SDIM is the correct number of selected eigenvalues
 *
 * Matrix types: 26 types from ZLATM4 (see zdrges.f documentation)
 */

#include "test_harness.h"
#include "verify.h"
#include "test_rng.h"
#include <math.h>
#include <string.h>
#include <complex.h>

#define THRESH 30.0
#define MAXTYP 26

static const INT NVAL[] = {0, 1, 2, 3, 5, 10, 20};
#define NNVAL (sizeof(NVAL) / sizeof(NVAL[0]))

/* ===================================================================== */
/* DATA arrays from zdrges.f lines 451-473                               */
/* Note: KAZERO/KBZERO values are 1-based indices into KZ1/KZ2/KADD.    */
/* The -1 adjustment is applied at point of use.                         */
/* KADD non-zero values are 1-based matrix positions; -1 at use site.    */
/* ===================================================================== */

static const INT KCLASS[MAXTYP] = {
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    3
};

static const INT KATYPE[MAXTYP] = {
    0, 1, 0, 1, 2, 3, 4, 1, 4, 4, 1, 1, 4, 4, 4,
    2, 4, 5, 8, 7, 9, 4, 4, 4, 4, 0
};

static const INT KBTYPE[MAXTYP] = {
    0, 0, 1, 1, 2, -3, 1, 4, 1, 1, 4, 4, 1, 1, -4,
    2, -4, 8, 8, 8, 8, 8, 8, 8, 8, 0
};

static const INT KAZERO[MAXTYP] = {
    1, 1, 1, 1, 1, 1, 2, 1, 2, 2, 1, 1, 2, 2, 3,
    1, 3, 5, 5, 5, 5, 3, 3, 3, 3, 1
};

static const INT KBZERO[MAXTYP] = {
    1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 2, 1, 1, 4,
    1, 4, 6, 6, 6, 6, 4, 4, 4, 4, 1
};

static const INT KAMAGN[MAXTYP] = {
    1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 2, 3, 2, 3, 1,
    1, 1, 1, 1, 1, 1, 2, 3, 3, 2, 1
};

static const INT KBMAGN[MAXTYP] = {
    1, 1, 1, 1, 1, 1, 1, 1, 3, 2, 3, 2, 2, 3, 1,
    1, 1, 1, 1, 1, 1, 3, 2, 3, 2, 1
};

static const INT KTRIAN[MAXTYP] = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1
};

static const INT LASIGN[MAXTYP] = {
    0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1,
    0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0
};

static const INT LBSIGN[MAXTYP] = {
    0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1,
    0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0
};

static const INT KZ1[6] = {0, 1, 2, 1, 3, 3};
static const INT KZ2[6] = {0, 0, 1, 2, 1, 1};
static const INT KADD[6] = {0, 0, 0, 0, 3, 2};

/* ===================================================================== */
/* Test parameters and workspace                                         */
/* ===================================================================== */

typedef struct {
    INT n;
    INT jtype;    /* matrix type 1..26 */
    char name[96];
} zdrges_params_t;

typedef struct {
    INT nmax;
    c128* A;
    c128* B;
    c128* S;
    c128* T;
    c128* Q;
    c128* Z;
    c128* alpha;
    c128* beta;
    c128* work;
    f64*  rwork;
    INT*  bwork;
    INT   lwork;
    f64   result[13];
    uint64_t rng_state[4];
} zdrges_workspace_t;

static zdrges_workspace_t* g_ws = NULL;

/* ===================================================================== */
/* Group setup / teardown                                                */
/* ===================================================================== */

static int group_setup(void** state)
{
    (void)state;

    g_ws = malloc(sizeof(zdrges_workspace_t));
    if (!g_ws) return -1;

    g_ws->nmax = 0;
    for (size_t i = 0; i < NNVAL; i++) {
        if (NVAL[i] > g_ws->nmax) g_ws->nmax = NVAL[i];
    }
    if (g_ws->nmax < 1) g_ws->nmax = 1;

    INT nmax = g_ws->nmax;
    INT n2 = nmax * nmax;

    g_ws->A     = malloc(n2 * sizeof(c128));
    g_ws->B     = malloc(n2 * sizeof(c128));
    g_ws->S     = malloc(n2 * sizeof(c128));
    g_ws->T     = malloc(n2 * sizeof(c128));
    g_ws->Q     = malloc(n2 * sizeof(c128));
    g_ws->Z     = malloc(n2 * sizeof(c128));
    g_ws->alpha = malloc(nmax * sizeof(c128));
    g_ws->beta  = malloc(nmax * sizeof(c128));

    g_ws->lwork = 3 * n2;
    if (g_ws->lwork < 1) g_ws->lwork = 1;
    g_ws->work  = malloc(g_ws->lwork * sizeof(c128));
    g_ws->rwork = malloc(8 * nmax * sizeof(f64));
    g_ws->bwork = malloc(nmax * sizeof(INT));

    if (!g_ws->A || !g_ws->B || !g_ws->S || !g_ws->T ||
        !g_ws->Q || !g_ws->Z || !g_ws->alpha || !g_ws->beta ||
        !g_ws->work || !g_ws->rwork || !g_ws->bwork) {
        return -1;
    }

    rng_seed(g_ws->rng_state, 0xDEADBEEFULL);
    return 0;
}

static int group_teardown(void** state)
{
    (void)state;

    if (g_ws) {
        free(g_ws->A);
        free(g_ws->B);
        free(g_ws->S);
        free(g_ws->T);
        free(g_ws->Q);
        free(g_ws->Z);
        free(g_ws->alpha);
        free(g_ws->beta);
        free(g_ws->work);
        free(g_ws->rwork);
        free(g_ws->bwork);
        free(g_ws);
        g_ws = NULL;
    }
    return 0;
}

/* ===================================================================== */
/* ABS1 statement function: |Re(x)| + |Im(x)|                           */
/* ===================================================================== */

static inline f64 abs1(c128 x)
{
    return fabs(creal(x)) + fabs(cimag(x));
}

/* ===================================================================== */
/* Matrix generation - port of zdrges.f lines 583-715                    */
/* ===================================================================== */

static INT generate_matrices(INT n, INT jtype, c128* A, c128* B,
                             INT lda, c128* Q, c128* Z, INT ldq,
                             c128* work, uint64_t state[static 4])
{
    const c128 czero = CMPLX(0.0, 0.0);
    const c128 cone = CMPLX(1.0, 0.0);
    INT jt = jtype - 1;
    INT iinfo = 0;

    f64 ulp = dlamch("P");
    f64 safmin = dlamch("S") / ulp;
    f64 safmax = 1.0 / safmin;
    f64 ulpinv = 1.0 / ulp;
    INT n1 = (n > 1) ? n : 1;

    f64 rmagn[4];
    rmagn[0] = 0.0;
    rmagn[1] = 1.0;
    rmagn[2] = safmax * ulp / (f64)n1;
    rmagn[3] = safmin * ulpinv * (f64)n1;

    if (KCLASS[jt] < 3) {
        /* Generate A (w/o rotation) */
        INT in = n;
        if (abs(KATYPE[jt]) == 3) {
            in = 2 * ((n - 1) / 2) + 1;
            if (in != n)
                zlaset("Full", n, n, czero, czero, A, lda);
        }
        zlatm4(KATYPE[jt], in, KZ1[KAZERO[jt] - 1], KZ2[KAZERO[jt] - 1],
               LASIGN[jt], rmagn[KAMAGN[jt]], ulp,
               rmagn[KTRIAN[jt] * KAMAGN[jt]], 2, A, lda, state);
        INT iadd = KADD[KAZERO[jt] - 1];
        if (iadd > 0 && iadd <= n)
            A[(iadd - 1) + (iadd - 1) * lda] = CMPLX(rmagn[KAMAGN[jt]], 0.0);

        /* Generate B (w/o rotation) */
        in = n;
        if (abs(KBTYPE[jt]) == 3) {
            in = 2 * ((n - 1) / 2) + 1;
            if (in != n)
                zlaset("Full", n, n, czero, czero, B, lda);
        }
        zlatm4(KBTYPE[jt], in, KZ1[KBZERO[jt] - 1], KZ2[KBZERO[jt] - 1],
               LBSIGN[jt], rmagn[KBMAGN[jt]], 1.0,
               rmagn[KTRIAN[jt] * KBMAGN[jt]], 2, B, lda, state);
        INT iadd_b = KADD[KBZERO[jt] - 1];
        if (iadd_b != 0 && iadd_b <= n)
            B[(iadd_b - 1) + (iadd_b - 1) * lda] = CMPLX(rmagn[KBMAGN[jt]], 0.0);

        if (KCLASS[jt] == 2 && n > 0) {
            /* Include rotations:
             * Generate Q, Z as Householder transformations times
             * a diagonal matrix. */
            for (INT jc = 0; jc < n - 1; jc++) {
                for (INT jr = jc; jr < n; jr++) {
                    Q[jr + jc * ldq] = zlarnd_rng(3, state);
                    Z[jr + jc * ldq] = zlarnd_rng(3, state);
                }
                zlarfg(n - jc, &Q[jc + jc * ldq], &Q[jc + 1 + jc * ldq], 1,
                       &work[jc]);
                work[2 * n + jc] = CMPLX(copysign(1.0, creal(Q[jc + jc * ldq])), 0.0);
                Q[jc + jc * ldq] = cone;
                zlarfg(n - jc, &Z[jc + jc * ldq], &Z[jc + 1 + jc * ldq], 1,
                       &work[n + jc]);
                work[3 * n + jc] = CMPLX(copysign(1.0, creal(Z[jc + jc * ldq])), 0.0);
                Z[jc + jc * ldq] = cone;
            }
            c128 ctemp = zlarnd_rng(3, state);
            Q[(n - 1) + (n - 1) * ldq] = cone;
            work[n - 1] = czero;
            work[3 * n - 1] = ctemp / cabs(ctemp);
            ctemp = zlarnd_rng(3, state);
            Z[(n - 1) + (n - 1) * ldq] = cone;
            work[2 * n - 1] = czero;
            work[4 * n - 1] = ctemp / cabs(ctemp);

            /* Apply the diagonal matrices */
            for (INT jc = 0; jc < n; jc++) {
                for (INT jr = 0; jr < n; jr++) {
                    A[jr + jc * lda] = work[2 * n + jr] *
                                       conj(work[3 * n + jc]) *
                                       A[jr + jc * lda];
                    B[jr + jc * lda] = work[2 * n + jr] *
                                       conj(work[3 * n + jc]) *
                                       B[jr + jc * lda];
                }
            }
            zunm2r("L", "N", n, n, n - 1, Q, ldq, work, A, lda,
                   &work[2 * n], &iinfo);
            if (iinfo != 0) return iinfo;
            zunm2r("R", "C", n, n, n - 1, Z, ldq, &work[n], A, lda,
                   &work[2 * n], &iinfo);
            if (iinfo != 0) return iinfo;
            zunm2r("L", "N", n, n, n - 1, Q, ldq, work, B, lda,
                   &work[2 * n], &iinfo);
            if (iinfo != 0) return iinfo;
            zunm2r("R", "C", n, n, n - 1, Z, ldq, &work[n], B, lda,
                   &work[2 * n], &iinfo);
            if (iinfo != 0) return iinfo;
        }
    } else {
        /* Random matrices */
        for (INT jc = 0; jc < n; jc++) {
            for (INT jr = 0; jr < n; jr++) {
                A[jr + jc * lda] = CMPLX(rmagn[KAMAGN[jt]], 0.0) *
                                   zlarnd_rng(4, state);
                B[jr + jc * lda] = CMPLX(rmagn[KBMAGN[jt]], 0.0) *
                                   zlarnd_rng(4, state);
            }
        }
    }

    return 0;
}

/* ===================================================================== */
/* Run tests for a single (n, jtype) combination.                        */
/* Port of zdrges.f lines 726-871.                                       */
/* ===================================================================== */

static void run_zdrges_single(zdrges_params_t* params)
{
    INT n = params->n;
    INT jtype = params->jtype;

    zdrges_workspace_t* ws = g_ws;
    INT lda = ws->nmax;
    INT ldq = ws->nmax;

    c128* A     = ws->A;
    c128* B     = ws->B;
    c128* S     = ws->S;
    c128* T     = ws->T;
    c128* Q     = ws->Q;
    c128* Z     = ws->Z;
    c128* alpha = ws->alpha;
    c128* beta  = ws->beta;
    c128* work  = ws->work;
    f64*  rwork = ws->rwork;
    INT*  bwork = ws->bwork;
    INT   lwork = ws->lwork;

    f64 ulp = dlamch("P");
    f64 safmin = dlamch("S") / ulp;
    f64 ulpinv = 1.0 / ulp;

    for (INT j = 0; j < 13; j++)
        ws->result[j] = -1.0;

    if (n == 0)
        return;

    INT iinfo = generate_matrices(n, jtype, A, B, lda, Q, Z, ldq,
                                  work, ws->rng_state);
    if (iinfo != 0) {
        ws->result[0] = ulpinv;
        fprintf(stderr, "Matrix generation failed for jtype=%d, n=%d, iinfo=%d\n",
                      jtype, n, iinfo);
        assert_info_success(iinfo);
        return;
    }

    for (INT j = 0; j < 13; j++)
        ws->result[j] = -1.0;

    /* Test with and without sorting of eigenvalues */
    for (INT isort = 0; isort <= 1; isort++) {
        const char* sort;
        INT rsub;

        if (isort == 0) {
            sort = "N";
            rsub = 0;
        } else {
            sort = "S";
            rsub = 5;
        }

        /* Copy A -> S and B -> T */
        zlacpy("Full", n, n, A, lda, S, lda);
        zlacpy("Full", n, n, B, lda, T, lda);

        ws->result[rsub + isort] = ulpinv;

        INT sdim;
        zgges("V", "V", sort, zlctes, n, S, lda, T, lda,
              &sdim, alpha, beta, Q, ldq, Z, ldq,
              work, lwork, rwork, bwork, &iinfo);

        if (iinfo != 0 && iinfo != n + 2) {
            ws->result[rsub + isort] = ulpinv;
            fprintf(stderr, "ZGGES failed with info=%d (n=%d, jtype=%d, isort=%d)\n",
                          iinfo, n, jtype, isort);
            continue;
        }

        /* Tests 1-4 (or 7-9 when reordering) */
        if (isort == 0) {
            /* Test 1: | A - Q S Z' | / ( |A| n ulp ) */
            zget51(1, n, A, lda, S, lda, Q, ldq, Z, ldq,
                   work, rwork, &ws->result[0]);
            /* Test 2: | B - Q T Z' | / ( |B| n ulp ) */
            zget51(1, n, B, lda, T, lda, Q, ldq, Z, ldq,
                   work, rwork, &ws->result[1]);
        } else {
            /* Test 7: | (A,B) - Q (S,T) Z' | / ( |(A,B)| n ulp ) */
            zget54(n, A, lda, B, lda, S, lda, T, lda, Q, ldq, Z, ldq,
                   work, &ws->result[1 + rsub]);
        }

        /* Test 3 (or 8): | I - QQ' | / ( n ulp ) */
        zget51(3, n, B, lda, T, lda, Q, ldq, Q, ldq,
               work, rwork, &ws->result[2 + rsub]);
        /* Test 4 (or 9): | I - ZZ' | / ( n ulp ) */
        zget51(3, n, B, lda, T, lda, Z, ldq, Z, ldq,
               work, rwork, &ws->result[3 + rsub]);

        /* Tests 5-6 (or 10-11): check Schur form of S and compare
         * eigenvalues with diagonals of (S, T). */
        f64 temp1 = 0.0;

        for (INT j = 0; j < n; j++) {
            f64 temp2 = (abs1(alpha[j] - S[j + j * lda]) /
                         fmax(safmin, fmax(abs1(alpha[j]), abs1(S[j + j * lda])))
                       + abs1(beta[j] - T[j + j * lda]) /
                         fmax(safmin, fmax(abs1(beta[j]), abs1(T[j + j * lda]))))
                        / ulp;

            if (j < n - 1) {
                if (S[j + 1 + j * lda] != CMPLX(0.0, 0.0)) {
                    ws->result[4 + rsub] = ulpinv;
                }
            }
            if (j > 0) {
                if (S[j + (j - 1) * lda] != CMPLX(0.0, 0.0)) {
                    ws->result[4 + rsub] = ulpinv;
                }
            }
            if (temp2 > temp1)
                temp1 = temp2;
        }
        ws->result[5 + rsub] = temp1;

        if (isort >= 1) {
            /* Test 12: SDIM validation */
            ws->result[11] = 0.0;
            INT knteig = 0;
            for (INT i = 0; i < n; i++) {
                if (zlctes(&alpha[i], &beta[i]))
                    knteig++;
            }
            if (sdim != knteig)
                ws->result[12] = ulpinv;
        }
    }

    /* Assert all results */
    for (INT j = 0; j < 13; j++) {
        if (ws->result[j] >= 0.0) {
            assert_residual_ok(ws->result[j]);
        }
    }
}

/* ===================================================================== */
/* CMocka test wrapper                                                    */
/* ===================================================================== */

static void test_zdrges_case(void** state)
{
    zdrges_params_t* params = *state;
    run_zdrges_single(params);
}

#define MAX_TESTS (NNVAL * MAXTYP)

static zdrges_params_t g_params[MAX_TESTS];
static struct CMUnitTest g_tests[MAX_TESTS];
static INT g_num_tests = 0;

static void build_test_array(void)
{
    g_num_tests = 0;

    for (size_t in = 0; in < NNVAL; in++) {
        INT n = NVAL[in];

        for (INT jtype = 1; jtype <= MAXTYP; jtype++) {
            zdrges_params_t* p = &g_params[g_num_tests];
            p->n = n;
            p->jtype = jtype;
            snprintf(p->name, sizeof(p->name),
                     "zdrges_n%d_type%d", n, jtype);

            g_tests[g_num_tests].name = p->name;
            g_tests[g_num_tests].test_func = test_zdrges_case;
            g_tests[g_num_tests].setup_func = NULL;
            g_tests[g_num_tests].teardown_func = NULL;
            g_tests[g_num_tests].initial_state = p;

            g_num_tests++;
        }
    }
}

int main(void)
{
    build_test_array();

    return _cmocka_run_group_tests("zdrges", g_tests, g_num_tests,
                                   group_setup, group_teardown);
}
