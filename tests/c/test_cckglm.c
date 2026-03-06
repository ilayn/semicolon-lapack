/**
 * @file test_cckglm.c
 * @brief GLM test driver - port of LAPACK TESTING/EIG/zckglm.f
 *
 * Tests CGGGLM - the generalized linear model solver:
 *   minimize ||y||_2 subject to d = A*x + B*y
 *
 * Each (m, p, n, imat) combination is registered as a separate CMocka test.
 *
 * Matrix types (8 total, from slatb9):
 *   Type 1: A diagonal, B lower triangular
 *   Type 2: A lower triangular, B diagonal
 *   Type 3: A lower triangular, B upper triangular
 *   Types 4-8: A general dense, B general dense
 */

#include "test_harness.h"
#include "verify.h"
#include "test_rng.h"
#include <math.h>
#include <string.h>

/* Test threshold from LAPACK zckglm.f */
#define THRESH 30.0f

/* Number of matrix types */
#define NTYPES 8

/* Dimension triplets from TESTING/glm.in */
static const INT MVAL[] = { 0,  5,  8, 15, 20, 40};
static const INT PVAL[] = { 9,  0, 15, 12, 15, 30};
static const INT NVAL[] = { 5,  5, 10, 25, 30, 40};
#define NN (sizeof(NVAL) / sizeof(NVAL[0]))

/* Maximum dimension */
#define NMAX 40

/* Maximum number of test cases: NN * NTYPES */
#define MAX_TESTS (NN * NTYPES)

typedef struct {
    INT ik;
    INT imat;
    char name[64];
} zckglm_params_t;

typedef struct {
    c64* A;
    c64* AF;
    c64* B;
    c64* BF;
    c64* D;
    c64* DF;
    c64* X;
    c64* U;
    c64* work;
    f32* rwork;
} zckglm_workspace_t;

static zckglm_workspace_t* g_ws = NULL;

static int group_setup(void** state)
{
    (void)state;

    g_ws = malloc(sizeof(zckglm_workspace_t));
    if (!g_ws) return -1;

    INT lda = NMAX;
    INT ldb = NMAX;
    INT lwork = NMAX * NMAX;

    g_ws->A     = malloc(lda * NMAX * sizeof(c64));
    g_ws->AF    = malloc(lda * NMAX * sizeof(c64));
    g_ws->B     = malloc(ldb * NMAX * sizeof(c64));
    g_ws->BF    = malloc(ldb * NMAX * sizeof(c64));
    g_ws->D     = malloc(NMAX * sizeof(c64));
    g_ws->DF    = malloc(NMAX * sizeof(c64));
    g_ws->X     = malloc(NMAX * sizeof(c64));
    g_ws->U     = malloc(NMAX * sizeof(c64));
    g_ws->work  = malloc(lwork * sizeof(c64));
    g_ws->rwork = malloc(NMAX * sizeof(f32));

    if (!g_ws->A || !g_ws->AF || !g_ws->B || !g_ws->BF ||
        !g_ws->D || !g_ws->DF || !g_ws->X || !g_ws->U ||
        !g_ws->work || !g_ws->rwork) {
        return -1;
    }

    return 0;
}

static int group_teardown(void** state)
{
    (void)state;

    if (g_ws) {
        free(g_ws->A);
        free(g_ws->AF);
        free(g_ws->B);
        free(g_ws->BF);
        free(g_ws->D);
        free(g_ws->DF);
        free(g_ws->X);
        free(g_ws->U);
        free(g_ws->work);
        free(g_ws->rwork);
        free(g_ws);
        g_ws = NULL;
    }

    return 0;
}

static void test_zckglm_case(void** state)
{
    zckglm_params_t* params = *state;
    INT ik = params->ik;
    INT imat = params->imat;

    INT m = MVAL[ik];
    INT p = PVAL[ik];
    INT n = NVAL[ik];
    INT lda = NMAX;
    INT ldb = NMAX;
    INT lwork = NMAX * NMAX;

    /* Skip invalid dimension triplets */
    if (m > n || n > m + p) {
        skip();
    }

    char type;
    INT kla, kua, klb, kub;
    f32 anorm, bnorm, cndnma, cndnmb;
    INT modea, modeb;
    char dista, distb;

    slatb9("GLM", imat, m, p, n, &type, &kla, &kua, &klb, &kub,
           &anorm, &bnorm, &modea, &modeb, &cndnma, &cndnmb,
           &dista, &distb);

    uint64_t rng_state[4];
    rng_seed(rng_state, 2024 + (uint64_t)ik * 100 + (uint64_t)imat);

    INT iinfo;
    char dista_str[2] = {dista, '\0'};
    char type_str[2] = {type, '\0'};
    clatms(n, m, dista_str, type_str, g_ws->rwork, modea, cndnma,
           anorm, kla, kua, "N", g_ws->A, lda, g_ws->work, &iinfo,
           rng_state);
    assert_int_equal(iinfo, 0);

    char distb_str[2] = {distb, '\0'};
    clatms(n, p, distb_str, type_str, g_ws->rwork, modeb, cndnmb,
           bnorm, klb, kub, "N", g_ws->B, ldb, g_ws->work, &iinfo,
           rng_state);
    assert_int_equal(iinfo, 0);

    clarnv_rng(2, n, g_ws->D, rng_state);

    f32 resid;
    cglmts(n, m, p, g_ws->A, g_ws->AF, lda, g_ws->B, g_ws->BF, ldb,
           g_ws->D, g_ws->DF, g_ws->X, g_ws->U,
           g_ws->work, lwork, g_ws->rwork, &resid);

    assert_residual_ok(resid);
}

static zckglm_params_t g_params[MAX_TESTS];
static struct CMUnitTest g_tests[MAX_TESTS];
static INT g_num_tests = 0;

static void build_test_array(void)
{
    g_num_tests = 0;

    for (INT ik = 0; ik < (INT)NN; ik++) {
        INT m = MVAL[ik];
        INT p = PVAL[ik];
        INT n = NVAL[ik];

        if (m > n || n > m + p)
            continue;

        for (INT imat = 1; imat <= NTYPES; imat++) {
            zckglm_params_t* par = &g_params[g_num_tests];
            par->ik = ik;
            par->imat = imat;
            snprintf(par->name, sizeof(par->name),
                     "GLM_m%d_p%d_n%d_type%d", m, p, n, imat);

            g_tests[g_num_tests].name = par->name;
            g_tests[g_num_tests].test_func = test_zckglm_case;
            g_tests[g_num_tests].setup_func = NULL;
            g_tests[g_num_tests].teardown_func = NULL;
            g_tests[g_num_tests].initial_state = par;

            g_num_tests++;
        }
    }
}

int main(void)
{
    build_test_array();

    return _cmocka_run_group_tests("zckglm", g_tests, g_num_tests,
                                   group_setup, group_teardown);
}
