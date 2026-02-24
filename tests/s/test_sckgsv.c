/**
 * @file test_sckgsv.c
 * @brief GSV test driver - port of LAPACK TESTING/EIG/dckgsv.f
 *
 * Tests SGGSVD3 - the GSVD for M-by-N matrix A and P-by-N matrix B:
 *   U'*A*Q = D1*R and V'*B*Q = D2*R
 *
 * Each (im, imat) combination is registered as a separate CMocka test.
 *
 * Matrix types (8 total, from slatb9):
 *   Type 1: A diagonal, B upper triangular
 *   Type 2: A upper triangular, B upper triangular
 *   Type 3: A lower triangular, B upper triangular
 *   Types 4-8: A general dense, B general dense
 */

#include "test_harness.h"
#include "verify.h"
#include "test_rng.h"
#include <math.h>
#include <string.h>

/* Test threshold from LAPACK gsv.in */
#define THRESH 20.0f

/* Number of matrix types */
#define NTYPES 8

/* Number of test results per call to sgsvts3 */
#define NTESTS 6

/* Dimension triplets from TESTING/gsv.in */
static const INT MVAL[] = { 0,  5,  9, 10, 20, 12, 12, 40 };
static const INT PVAL[] = { 4,  0, 12, 14, 10, 10, 20, 15 };
static const INT NVAL[] = { 3, 10, 15, 12,  8, 20,  8, 20 };
#define NM 8

/* Maximum dimension */
#define NMAX 40

/* Maximum number of test cases: NM * NTYPES */
#define MAX_TESTS (NM * NTYPES)

/* External declarations */
/* Test parameters for a single test case */
typedef struct {
    INT im;
    INT imat;
    char name[80];
} dckgsv_params_t;

/* Global workspace - allocated once, shared across all tests */
typedef struct {
    f32* A;
    f32* AF;
    f32* B;
    f32* BF;
    f32* U;
    f32* V;
    f32* Q;
    f32* R;
    f32* alpha;
    f32* beta;
    f32* work;
    f32* rwork;
    INT* iwork;
} dckgsv_workspace_t;

static dckgsv_workspace_t* g_ws = NULL;

static int group_setup(void** state)
{
    (void)state;

    g_ws = malloc(sizeof(dckgsv_workspace_t));
    if (!g_ws) return -1;

    INT lwork = NMAX * NMAX;

    g_ws->A     = malloc(NMAX * NMAX * sizeof(f32));
    g_ws->AF    = malloc(NMAX * NMAX * sizeof(f32));
    g_ws->B     = malloc(NMAX * NMAX * sizeof(f32));
    g_ws->BF    = malloc(NMAX * NMAX * sizeof(f32));
    g_ws->U     = malloc(NMAX * NMAX * sizeof(f32));
    g_ws->V     = malloc(NMAX * NMAX * sizeof(f32));
    g_ws->Q     = malloc(NMAX * NMAX * sizeof(f32));
    g_ws->R     = malloc(NMAX * NMAX * sizeof(f32));
    g_ws->alpha = malloc(NMAX * sizeof(f32));
    g_ws->beta  = malloc(NMAX * sizeof(f32));
    g_ws->work  = malloc(lwork * sizeof(f32));
    g_ws->rwork = malloc(NMAX * sizeof(f32));
    g_ws->iwork = malloc(NMAX * sizeof(INT));

    if (!g_ws->A || !g_ws->AF || !g_ws->B || !g_ws->BF ||
        !g_ws->U || !g_ws->V || !g_ws->Q || !g_ws->R ||
        !g_ws->alpha || !g_ws->beta ||
        !g_ws->work || !g_ws->rwork || !g_ws->iwork) {
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
        free(g_ws->U);
        free(g_ws->V);
        free(g_ws->Q);
        free(g_ws->R);
        free(g_ws->alpha);
        free(g_ws->beta);
        free(g_ws->work);
        free(g_ws->rwork);
        free(g_ws->iwork);
        free(g_ws);
        g_ws = NULL;
    }

    return 0;
}

static void test_dckgsv_case(void** state)
{
    dckgsv_params_t* params = *state;
    INT m = MVAL[params->im];
    INT p = PVAL[params->im];
    INT n = NVAL[params->im];
    INT imat = params->imat;
    INT lda = NMAX;
    INT ldb = NMAX;
    INT ldu = NMAX;
    INT ldv = NMAX;
    INT ldq = NMAX;
    INT ldr = NMAX;
    INT lwork = NMAX * NMAX;

    char type;
    INT kla, kua, klb, kub;
    f32 anorm, bnorm, cndnma, cndnmb;
    INT modea, modeb;
    char dista, distb;
    INT iinfo;
    f32 result[NTESTS];

    /* Seed RNG for reproducibility */
    uint64_t rng_state[4];
    rng_seed(rng_state, 2024 + (uint64_t)params->im * 100 +
             (uint64_t)imat);

    slatb9("GSV", imat, m, p, n, &type, &kla, &kua, &klb, &kub,
           &anorm, &bnorm, &modea, &modeb, &cndnma, &cndnmb,
           &dista, &distb);

    /* Generate M by N matrix A */
    char dista_str[2] = {dista, '\0'};
    char type_str[2] = {type, '\0'};
    slatms(m, n, dista_str, type_str, g_ws->rwork, modea, cndnma,
           anorm, kla, kua, "N", g_ws->A, lda, g_ws->work, &iinfo,
           rng_state);
    assert_int_equal(iinfo, 0);

    /* Generate P by N matrix B */
    char distb_str[2] = {distb, '\0'};
    slatms(p, n, distb_str, type_str, g_ws->rwork, modeb, cndnmb,
           bnorm, klb, kub, "N", g_ws->B, ldb, g_ws->work, &iinfo,
           rng_state);
    assert_int_equal(iinfo, 0);

    sgsvts3(m, p, n, g_ws->A, g_ws->AF, lda, g_ws->B, g_ws->BF, ldb,
            g_ws->U, ldu, g_ws->V, ldv, g_ws->Q, ldq,
            g_ws->alpha, g_ws->beta, g_ws->R, ldr,
            g_ws->iwork, g_ws->work, lwork, g_ws->rwork, result);

    for (INT i = 0; i < NTESTS; i++) {
        if (result[i] >= THRESH) {
            print_message("  GSV: M=%d P=%d N=%d type=%d test=%d ratio=%g\n",
                          m, p, n, imat, i + 1, (double)result[i]);
        }
        assert_residual_below(result[i], THRESH);
    }
}

/* Parameterized test arrays */
static dckgsv_params_t g_params[MAX_TESTS];
static struct CMUnitTest g_tests[MAX_TESTS];
static INT g_num_tests = 0;

static void build_test_array(void)
{
    g_num_tests = 0;

    for (INT im = 0; im < NM; im++) {
        for (INT imat = 1; imat <= NTYPES; imat++) {
            dckgsv_params_t* par = &g_params[g_num_tests];
            par->im = im;
            par->imat = imat;
            snprintf(par->name, sizeof(par->name),
                     "GSV_m%d_p%d_n%d_type%d",
                     MVAL[im], PVAL[im], NVAL[im], imat);

            g_tests[g_num_tests].name = par->name;
            g_tests[g_num_tests].test_func = test_dckgsv_case;
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

    return _cmocka_run_group_tests("dckgsv", g_tests, g_num_tests,
                                   group_setup, group_teardown);
}
