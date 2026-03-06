/**
 * @file test_zdrgev3.c
 * @brief Generalized eigenvalue driver test - port of LAPACK TESTING/EIG/zdrgev3.f
 *
 * Tests the nonsymmetric generalized eigenvalue problem driver ZGGEV3.
 * This is nearly identical to test_zdrgev.c but calls ZGGEV3 (which uses the
 * blocked QZ algorithm ZLAQZ0) instead of ZGGEV, and sets XLAENV parameters
 * before each call.
 *
 * Each (n, jtype) combination is registered as a separate CMocka test,
 * providing pytest-like parameterized test behavior with clear failure isolation.
 *
 * Test ratios (7 total):
 *
 *   (0)  max | VL^H * (beta A - alpha B) | / ( ulp max(|beta A|, |alpha B|) )
 *
 *   (1)  | |VL(i)| - 1 | / ulp and whether largest component real
 *
 *   (2)  max | (beta A - alpha B) * VR | / ( ulp max(|beta A|, |alpha B|) )
 *
 *   (3)  | |VR(i)| - 1 | / ulp and whether largest component real
 *
 *   (4)  W(full) = W(partial)
 *        W(full) denotes the eigenvalues computed when both l and r
 *        are also computed, and W(partial) denotes the eigenvalues
 *        computed when only W, only W and r, or only W and l are
 *        computed.
 *
 *   (5)  VL(full) = VL(partial)
 *        VL(full) denotes the left eigenvectors computed when both l
 *        and r are computed, and VL(partial) denotes the result
 *        when only l is computed.
 *
 *   (6)  VR(full) = VR(partial)
 *        VR(full) denotes the right eigenvectors computed when both l
 *        and r are also computed, and VR(partial) denotes the result
 *        when only r is computed.
 *
 * Matrix types: 26 types from ZLATM4 (see zdrgev3.f documentation)
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
/* DATA arrays from zdrgev3.f lines 461-484                              */
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
} zdrgev3_params_t;

typedef struct {
    INT nmax;
    c128* A;
    c128* B;
    c128* S;
    c128* T;
    c128* Q;
    c128* Z;
    c128* QE;
    c128* alpha;
    c128* beta;
    c128* alpha1;
    c128* beta1;
    c128* work;
    f64*  rwork;
    INT   lwork;
    f64   result[7];
    uint64_t rng_state[4];
} zdrgev3_workspace_t;

static zdrgev3_workspace_t* g_ws = NULL;

/* ===================================================================== */
/* Group setup / teardown                                                */
/* ===================================================================== */

static int group_setup(void** state)
{
    (void)state;

    g_ws = malloc(sizeof(zdrgev3_workspace_t));
    if (!g_ws) return -1;

    g_ws->nmax = 0;
    for (size_t i = 0; i < NNVAL; i++) {
        if (NVAL[i] > g_ws->nmax) g_ws->nmax = NVAL[i];
    }
    if (g_ws->nmax < 1) g_ws->nmax = 1;

    INT nmax = g_ws->nmax;
    INT n2 = nmax * nmax;

    g_ws->A      = malloc(n2 * sizeof(c128));
    g_ws->B      = malloc(n2 * sizeof(c128));
    g_ws->S      = malloc(n2 * sizeof(c128));
    g_ws->T      = malloc(n2 * sizeof(c128));
    g_ws->Q      = malloc(n2 * sizeof(c128));
    g_ws->Z      = malloc(n2 * sizeof(c128));
    g_ws->QE     = malloc(n2 * sizeof(c128));
    g_ws->alpha  = malloc(nmax * sizeof(c128));
    g_ws->beta   = malloc(nmax * sizeof(c128));
    g_ws->alpha1 = malloc(nmax * sizeof(c128));
    g_ws->beta1  = malloc(nmax * sizeof(c128));

    g_ws->lwork = 3 * n2;
    if (g_ws->lwork < 1) g_ws->lwork = 1;
    g_ws->work  = malloc(g_ws->lwork * sizeof(c128));
    g_ws->rwork = malloc(8 * nmax * sizeof(f64));

    if (!g_ws->A || !g_ws->B || !g_ws->S || !g_ws->T ||
        !g_ws->Q || !g_ws->Z || !g_ws->QE ||
        !g_ws->alpha || !g_ws->beta ||
        !g_ws->alpha1 || !g_ws->beta1 ||
        !g_ws->work || !g_ws->rwork) {
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
        free(g_ws->QE);
        free(g_ws->alpha);
        free(g_ws->beta);
        free(g_ws->alpha1);
        free(g_ws->beta1);
        free(g_ws->work);
        free(g_ws->rwork);
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
/* Matrix generation - port of zdrgev3.f lines 583-719                   */
/* Identical to zdrgev.f / zdrges.f matrix generation (same 26 types).   */
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
/* Port of zdrgev3.f lines 732-847.                                      */
/* ===================================================================== */

static void run_zdrgev3_single(zdrgev3_params_t* params)
{
    INT n = params->n;
    INT jtype = params->jtype;

    zdrgev3_workspace_t* ws = g_ws;
    INT lda = ws->nmax;
    INT ldq = ws->nmax;
    INT ldqe = ws->nmax;

    c128* A      = ws->A;
    c128* B      = ws->B;
    c128* S      = ws->S;
    c128* T      = ws->T;
    c128* Q      = ws->Q;
    c128* Z      = ws->Z;
    c128* QE     = ws->QE;
    c128* alpha  = ws->alpha;
    c128* beta   = ws->beta;
    c128* alpha1 = ws->alpha1;
    c128* beta1  = ws->beta1;
    c128* work   = ws->work;
    f64*  rwork  = ws->rwork;
    INT   lwork  = ws->lwork;

    f64 ulpinv = 1.0 / dlamch("P");

    for (INT j = 0; j < 7; j++)
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

    for (INT j = 0; j < 7; j++)
        ws->result[j] = -1.0;

    /* Set the parameters used in ZLAQZ0 */
    xlaenv(12, 10);
    xlaenv(13, 12);
    xlaenv(14, 13);
    xlaenv(15, 2);
    xlaenv(17, 10);

    /* Call ZGGEV3 to compute eigenvalues and eigenvectors. */
    zlacpy(" ", n, n, A, lda, S, lda);
    zlacpy(" ", n, n, B, lda, T, lda);
    zggev3("V", "V", n, S, lda, T, lda, alpha, beta, Q,
           ldq, Z, ldq, work, lwork, rwork, &iinfo);
    if (iinfo != 0 && iinfo != n + 1) {
        ws->result[0] = ulpinv;
        fprintf(stderr, "ZGGEV31 returned info=%d (n=%d, jtype=%d)\n",
                      iinfo, n, jtype);
        goto done;
    }

    /* Tests 0-1: left eigenvector check via ZGET52 */
    zget52(1, n, A, lda, B, lda, Q, ldq, alpha, beta,
           work, rwork, &ws->result[0]);
    if (ws->result[1] > THRESH) {
        fprintf(stderr, "Left eigenvectors from ZGGEV31 incorrectly normalized. "
                      "Bits of error=%g, n=%d, jtype=%d\n",
                      ws->result[1], n, jtype);
    }

    /* Tests 2-3: right eigenvector check via ZGET52 */
    zget52(0, n, A, lda, B, lda, Z, ldq, alpha, beta,
           work, rwork, &ws->result[2]);
    if (ws->result[3] > THRESH) {
        fprintf(stderr, "Right eigenvectors from ZGGEV31 incorrectly normalized. "
                      "Bits of error=%g, n=%d, jtype=%d\n",
                      ws->result[3], n, jtype);
    }

    /* Test 4: eigenvalues-only computation, compare with full. */
    zlacpy(" ", n, n, A, lda, S, lda);
    zlacpy(" ", n, n, B, lda, T, lda);
    zggev3("N", "N", n, S, lda, T, lda, alpha1, beta1, Q,
           ldq, Z, ldq, work, lwork, rwork, &iinfo);
    if (iinfo != 0 && iinfo != n + 1) {
        ws->result[0] = ulpinv;
        fprintf(stderr, "ZGGEV32 returned info=%d (n=%d, jtype=%d)\n",
                      iinfo, n, jtype);
        goto done;
    }

    for (INT j = 0; j < n; j++) {
        if (alpha[j] != alpha1[j] || beta[j] != beta1[j])
            ws->result[4] = ulpinv;
    }

    /* Test 5: left eigenvectors only, compare eigenvalues and eigenvectors. */
    zlacpy(" ", n, n, A, lda, S, lda);
    zlacpy(" ", n, n, B, lda, T, lda);
    zggev3("V", "N", n, S, lda, T, lda, alpha1, beta1, QE,
           ldqe, Z, ldq, work, lwork, rwork, &iinfo);
    if (iinfo != 0 && iinfo != n + 1) {
        ws->result[0] = ulpinv;
        fprintf(stderr, "ZGGEV33 returned info=%d (n=%d, jtype=%d)\n",
                      iinfo, n, jtype);
        goto done;
    }

    for (INT j = 0; j < n; j++) {
        if (alpha[j] != alpha1[j] || beta[j] != beta1[j])
            ws->result[5] = ulpinv;
    }

    for (INT j = 0; j < n; j++) {
        for (INT jc = 0; jc < n; jc++) {
            if (Q[j + jc * ldq] != QE[j + jc * ldqe])
                ws->result[5] = ulpinv;
        }
    }

    /* Test 6: right eigenvectors only, compare eigenvalues and eigenvectors. */
    zlacpy(" ", n, n, A, lda, S, lda);
    zlacpy(" ", n, n, B, lda, T, lda);
    zggev3("N", "V", n, S, lda, T, lda, alpha1, beta1, Q,
           ldq, QE, ldqe, work, lwork, rwork, &iinfo);
    if (iinfo != 0 && iinfo != n + 1) {
        ws->result[0] = ulpinv;
        fprintf(stderr, "ZGGEV34 returned info=%d (n=%d, jtype=%d)\n",
                      iinfo, n, jtype);
        goto done;
    }

    for (INT j = 0; j < n; j++) {
        if (alpha[j] != alpha1[j] || beta[j] != beta1[j])
            ws->result[6] = ulpinv;
    }

    for (INT j = 0; j < n; j++) {
        for (INT jc = 0; jc < n; jc++) {
            if (Z[j + jc * ldq] != QE[j + jc * ldqe])
                ws->result[6] = ulpinv;
        }
    }

done:
    /* Assert all results */
    for (INT j = 0; j < 7; j++) {
        if (ws->result[j] >= 0.0) {
            assert_residual_ok(ws->result[j]);
        }
    }
}

/* ===================================================================== */
/* CMocka test wrapper                                                    */
/* ===================================================================== */

static void test_zdrgev3_case(void** state)
{
    zdrgev3_params_t* params = *state;
    run_zdrgev3_single(params);
}

#define MAX_TESTS (NNVAL * MAXTYP)

static zdrgev3_params_t g_params[MAX_TESTS];
static struct CMUnitTest g_tests[MAX_TESTS];
static INT g_num_tests = 0;

static void build_test_array(void)
{
    g_num_tests = 0;

    for (size_t in = 0; in < NNVAL; in++) {
        INT n = NVAL[in];

        for (INT jtype = 1; jtype <= MAXTYP; jtype++) {
            zdrgev3_params_t* p = &g_params[g_num_tests];
            p->n = n;
            p->jtype = jtype;
            snprintf(p->name, sizeof(p->name),
                     "zdrgev3_n%d_type%d", n, jtype);

            g_tests[g_num_tests].name = p->name;
            g_tests[g_num_tests].test_func = test_zdrgev3_case;
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

    return _cmocka_run_group_tests("zdrgev3", g_tests, g_num_tests,
                                   group_setup, group_teardown);
}
