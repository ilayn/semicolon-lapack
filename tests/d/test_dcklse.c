/**
 * @file test_dcklse.c
 * @brief LSE test driver - port of LAPACK TESTING/EIG/dcklse.f
 *
 * Tests DGGLSE - the linear equality constrained least squares solver:
 *   minimize ||A*x - c||_2 subject to B*x = d
 *
 * Each (m, p, n, imat) combination is registered as a separate CMocka test.
 *
 * Matrix types (8 total, from dlatb9):
 *   Type 1: A diagonal, B lower triangular
 *   Type 2: A lower triangular, B diagonal
 *   Type 3: A lower triangular, B upper triangular
 *   Types 4-8: A general dense, B general dense
 */

#include "test_harness.h"
#include "verify.h"
#include "test_rng.h"
#include <cblas.h>
#include <math.h>
#include <string.h>

/* Test threshold from LAPACK lse.in */
#define THRESH 20.0

/* Number of matrix types */
#define NTYPES 8

/* Dimension triplets from TESTING/lse.in */
static const int MVAL[] = { 6,  0,  5,  8, 10, 30};
static const int PVAL[] = { 0,  5,  5,  5,  8, 20};
static const int NVAL[] = { 5,  5,  6,  8, 12, 40};
#define NN (sizeof(NVAL) / sizeof(NVAL[0]))

/* Maximum dimension */
#define NMAX 40

/* Maximum number of test cases: NN * NTYPES */
#define MAX_TESTS (NN * NTYPES)

/* External declarations */
extern f64 dlamch(const char* cmach);

/* Test parameters for a single test case */
typedef struct {
    int ik;
    int imat;
    char name[64];
} dcklse_params_t;

/* Global workspace - allocated once, shared across all tests */
typedef struct {
    f64* A;
    f64* AF;
    f64* B;
    f64* BF;
    f64* C;
    f64* CF;
    f64* D;
    f64* DF;
    f64* X;
    f64* work;
    f64* rwork;
} dcklse_workspace_t;

static dcklse_workspace_t* g_ws = NULL;

static int group_setup(void** state)
{
    (void)state;

    g_ws = malloc(sizeof(dcklse_workspace_t));
    if (!g_ws) return -1;

    int lda = NMAX;
    int ldb = NMAX;
    int lwork = NMAX * NMAX;

    g_ws->A     = malloc(lda * NMAX * sizeof(f64));
    g_ws->AF    = malloc(lda * NMAX * sizeof(f64));
    g_ws->B     = malloc(ldb * NMAX * sizeof(f64));
    g_ws->BF    = malloc(ldb * NMAX * sizeof(f64));
    g_ws->C     = malloc(NMAX * sizeof(f64));
    g_ws->CF    = malloc(NMAX * sizeof(f64));
    g_ws->D     = malloc(NMAX * sizeof(f64));
    g_ws->DF    = malloc(NMAX * sizeof(f64));
    g_ws->X     = malloc(NMAX * sizeof(f64));
    g_ws->work  = malloc(lwork * sizeof(f64));
    g_ws->rwork = malloc(NMAX * sizeof(f64));

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

static void test_dcklse_case(void** state)
{
    dcklse_params_t* params = *state;
    int ik = params->ik;
    int imat = params->imat;

    int m = MVAL[ik];
    int p = PVAL[ik];
    int n = NVAL[ik];
    int lda = NMAX;
    int ldb = NMAX;
    int lwork = NMAX * NMAX;

    /* Skip invalid dimension triplets: need P <= N <= M+P */
    if (p > n || n > m + p) {
        skip();
    }

    /* Set up parameters with DLATB9 */
    char type;
    int kla, kua, klb, kub;
    f64 anorm, bnorm, cndnma, cndnmb;
    int modea, modeb;
    char dista, distb;

    dlatb9("LSE", imat, m, p, n, &type, &kla, &kua, &klb, &kub,
           &anorm, &bnorm, &modea, &modeb, &cndnma, &cndnmb,
           &dista, &distb);

    /* Seed RNG for reproducibility */
    uint64_t rng_state[4];
    rng_seed(rng_state, 2024 + (uint64_t)ik * 100 + (uint64_t)imat);

    /* Generate test matrix A (M x N) */
    int iinfo;
    char dista_str[2] = {dista, '\0'};
    char type_str[2] = {type, '\0'};
    dlatms(m, n, dista_str, type_str, g_ws->rwork, modea, cndnma,
           anorm, kla, kua, "N", g_ws->A, lda, g_ws->work, &iinfo,
           rng_state);
    assert_int_equal(iinfo, 0);

    /* Generate test matrix B (P x N) */
    char distb_str[2] = {distb, '\0'};
    dlatms(p, n, distb_str, type_str, g_ws->rwork, modeb, cndnmb,
           bnorm, klb, kub, "N", g_ws->B, ldb, g_ws->work, &iinfo,
           rng_state);
    assert_int_equal(iinfo, 0);

    /* Generate the right-hand sides C and D for the LSE */
    dlarhs("DGE", "New solution", "Upper", "N", m, n,
           (m - 1 > 0) ? m - 1 : 0, (n - 1 > 0) ? n - 1 : 0, 1,
           g_ws->A, lda, g_ws->X, (n > 1) ? n : 1,
           g_ws->C, (m > 1) ? m : 1, &iinfo, rng_state);

    dlarhs("DGE", "Computed", "Upper", "N", p, n,
           (p - 1 > 0) ? p - 1 : 0, (n - 1 > 0) ? n - 1 : 0, 1,
           g_ws->B, ldb, g_ws->X, (n > 1) ? n : 1,
           g_ws->D, (p > 1) ? p : 1, &iinfo, rng_state);

    /* Test DGGLSE */
    f64 result[2];
    dlsets(m, p, n, g_ws->A, g_ws->AF, lda, g_ws->B, g_ws->BF, ldb,
           g_ws->C, g_ws->CF, g_ws->D, g_ws->DF, g_ws->X,
           g_ws->work, lwork, g_ws->rwork, result);

    assert_residual_below(result[0], THRESH);
    assert_residual_below(result[1], THRESH);
}

/* Parameterized test arrays */
static dcklse_params_t g_params[MAX_TESTS];
static struct CMUnitTest g_tests[MAX_TESTS];
static int g_num_tests = 0;

static void build_test_array(void)
{
    g_num_tests = 0;

    for (int ik = 0; ik < (int)NN; ik++) {
        int m = MVAL[ik];
        int p = PVAL[ik];
        int n = NVAL[ik];

        /* Skip invalid dimension triplets */
        if (p > n || n > m + p)
            continue;

        for (int imat = 1; imat <= NTYPES; imat++) {
            dcklse_params_t* par = &g_params[g_num_tests];
            par->ik = ik;
            par->imat = imat;
            snprintf(par->name, sizeof(par->name),
                     "LSE_m%d_p%d_n%d_type%d", m, p, n, imat);

            g_tests[g_num_tests].name = par->name;
            g_tests[g_num_tests].test_func = test_dcklse_case;
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

    return _cmocka_run_group_tests("dcklse", g_tests, g_num_tests,
                                   group_setup, group_teardown);
}
