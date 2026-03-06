/**
 * @file test_ccklse.c
 * @brief LSE test driver - port of LAPACK TESTING/EIG/zcklse.f
 *
 * Tests CGGLSE - the linear equality constrained least squares solver:
 *   minimize ||A*x - c||_2 subject to B*x = d
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

#define THRESH 20.0f

#define NTYPES 8

/* Dimension triplets from TESTING/lse.in */
static const INT MVAL[] = { 6,  0,  5,  8, 10, 30};
static const INT PVAL[] = { 0,  5,  5,  5,  8, 20};
static const INT NVAL[] = { 5,  5,  6,  8, 12, 40};
#define NN (sizeof(NVAL) / sizeof(NVAL[0]))

#define NMAX 40

#define MAX_TESTS (NN * NTYPES)

typedef struct {
    INT ik;
    INT imat;
    char name[64];
} zcklse_params_t;

typedef struct {
    c64* A;
    c64* AF;
    c64* B;
    c64* BF;
    c64* C;
    c64* CF;
    c64* D;
    c64* DF;
    c64* X;
    c64* work;
    f32* rwork;
} zcklse_workspace_t;

static zcklse_workspace_t* g_ws = NULL;

static int group_setup(void** state)
{
    (void)state;

    g_ws = malloc(sizeof(zcklse_workspace_t));
    if (!g_ws) return -1;

    INT lda = NMAX;
    INT ldb = NMAX;
    INT lwork = NMAX * NMAX;

    g_ws->A     = malloc(lda * NMAX * sizeof(c64));
    g_ws->AF    = malloc(lda * NMAX * sizeof(c64));
    g_ws->B     = malloc(ldb * NMAX * sizeof(c64));
    g_ws->BF    = malloc(ldb * NMAX * sizeof(c64));
    g_ws->C     = malloc(NMAX * sizeof(c64));
    g_ws->CF    = malloc(NMAX * sizeof(c64));
    g_ws->D     = malloc(NMAX * sizeof(c64));
    g_ws->DF    = malloc(NMAX * sizeof(c64));
    g_ws->X     = malloc(NMAX * sizeof(c64));
    g_ws->work  = malloc(lwork * sizeof(c64));
    g_ws->rwork = malloc(NMAX * sizeof(f32));

    if (!g_ws->A || !g_ws->AF || !g_ws->B || !g_ws->BF ||
        !g_ws->C || !g_ws->CF || !g_ws->D || !g_ws->DF ||
        !g_ws->X || !g_ws->work || !g_ws->rwork) {
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
        free(g_ws->C);
        free(g_ws->CF);
        free(g_ws->D);
        free(g_ws->DF);
        free(g_ws->X);
        free(g_ws->work);
        free(g_ws->rwork);
        free(g_ws);
        g_ws = NULL;
    }

    return 0;
}

static void test_zcklse_case(void** state)
{
    zcklse_params_t* params = *state;
    INT ik = params->ik;
    INT imat = params->imat;

    INT m = MVAL[ik];
    INT p = PVAL[ik];
    INT n = NVAL[ik];
    INT lda = NMAX;
    INT ldb = NMAX;
    INT lwork = NMAX * NMAX;

    /* Skip invalid dimension triplets: need P <= N <= M+P */
    if (p > n || n > m + p) {
        skip();
    }

    char type;
    INT kla, kua, klb, kub;
    f32 anorm, bnorm, cndnma, cndnmb;
    INT modea, modeb;
    char dista, distb;

    slatb9("LSE", imat, m, p, n, &type, &kla, &kua, &klb, &kub,
           &anorm, &bnorm, &modea, &modeb, &cndnma, &cndnmb,
           &dista, &distb);

    uint64_t rng_state[4];
    rng_seed(rng_state, 2024 + (uint64_t)ik * 100 + (uint64_t)imat);

    INT iinfo;
    char dista_str[2] = {dista, '\0'};
    char type_str[2] = {type, '\0'};
    clatms(m, n, dista_str, type_str, g_ws->rwork, modea, cndnma,
           anorm, kla, kua, "N", g_ws->A, lda, g_ws->work, &iinfo,
           rng_state);
    assert_int_equal(iinfo, 0);

    char distb_str[2] = {distb, '\0'};
    clatms(p, n, distb_str, type_str, g_ws->rwork, modeb, cndnmb,
           bnorm, klb, kub, "N", g_ws->B, ldb, g_ws->work, &iinfo,
           rng_state);
    assert_int_equal(iinfo, 0);

    clarhs("CGE", "New solution", "Upper", "N", m, n,
           (m - 1 > 0) ? m - 1 : 0, (n - 1 > 0) ? n - 1 : 0, 1,
           g_ws->A, lda, g_ws->X, (n > 1) ? n : 1,
           g_ws->C, (m > 1) ? m : 1, &iinfo, rng_state);

    clarhs("CGE", "Computed", "Upper", "N", p, n,
           (p - 1 > 0) ? p - 1 : 0, (n - 1 > 0) ? n - 1 : 0, 1,
           g_ws->B, ldb, g_ws->X, (n > 1) ? n : 1,
           g_ws->D, (p > 1) ? p : 1, &iinfo, rng_state);

    f32 result[2];
    clsets(m, p, n, g_ws->A, g_ws->AF, lda, g_ws->B, g_ws->BF, ldb,
           g_ws->C, g_ws->CF, g_ws->D, g_ws->DF, g_ws->X,
           g_ws->work, lwork, g_ws->rwork, result);

    assert_residual_below(result[0], THRESH);
    assert_residual_below(result[1], THRESH);
}

static zcklse_params_t g_params[MAX_TESTS];
static struct CMUnitTest g_tests[MAX_TESTS];
static INT g_num_tests = 0;

static void build_test_array(void)
{
    g_num_tests = 0;

    for (INT ik = 0; ik < (INT)NN; ik++) {
        INT m = MVAL[ik];
        INT p = PVAL[ik];
        INT n = NVAL[ik];

        if (p > n || n > m + p)
            continue;

        for (INT imat = 1; imat <= NTYPES; imat++) {
            zcklse_params_t* par = &g_params[g_num_tests];
            par->ik = ik;
            par->imat = imat;
            snprintf(par->name, sizeof(par->name),
                     "LSE_m%d_p%d_n%d_type%d", m, p, n, imat);

            g_tests[g_num_tests].name = par->name;
            g_tests[g_num_tests].test_func = test_zcklse_case;
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

    (void)_cmocka_run_group_tests("zcklse", g_tests, g_num_tests,
                                   group_setup, group_teardown);
    return 0;
}
