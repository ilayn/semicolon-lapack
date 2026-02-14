/**
 * @file test_dckgqr.c
 * @brief GQR/GRQ test driver - port of LAPACK TESTING/EIG/dckgqr.f
 *
 * Tests DGGQRF and DGGRQF - the generalized QR and RQ factorizations:
 *   DGGQRF: A = Q*R, B = Q*T*Z  (N-by-M matrix A, N-by-P matrix B)
 *   DGGRQF: A = R*Q, B = Z*T*Q  (M-by-N matrix A, P-by-N matrix B)
 *
 * Each (im, ip, in, imat) combination is registered as a separate CMocka test.
 * For each combination, both DGGRQF (via dgrqts) and DGGQRF (via dgqrts) are tested.
 *
 * Matrix types (8 total, from dlatb9):
 *   Type 1: A diagonal, B upper/lower triangular
 *   Type 2: A upper/lower triangular, B upper triangular/diagonal
 *   Type 3: A lower triangular, B upper triangular
 *   Types 4-8: A general dense, B general dense
 */

#include "test_harness.h"
#include "testutils/verify.h"
#include "testutils/test_rng.h"
#include <cblas.h>
#include <math.h>
#include <string.h>

/* Test threshold from LAPACK gqr.in */
#define THRESH 20.0

/* Number of matrix types */
#define NTYPES 8

/* Number of test results per factorization */
#define NTESTS 4

/* Dimension triplets from TESTING/gqr.in */
static const int MVAL[] = { 0,  3, 10 };
static const int PVAL[] = { 0,  5, 20 };
static const int NVAL[] = { 0,  3, 30 };
#define NM (sizeof(MVAL) / sizeof(MVAL[0]))
#define NP (sizeof(PVAL) / sizeof(PVAL[0]))
#define NN (sizeof(NVAL) / sizeof(NVAL[0]))

/* Maximum dimension */
#define NMAX 30

/* Maximum number of test cases: NM * NP * NN * NTYPES */
#define MAX_TESTS (NM * NP * NN * NTYPES)

/* External declarations */
extern f64 dlamch(const char* cmach);

/* Test parameters for a single test case */
typedef struct {
    int im;
    int ip;
    int in;
    int imat;
    char name[80];
} dckgqr_params_t;

/* Global workspace - allocated once, shared across all tests */
typedef struct {
    f64* A;
    f64* AF;
    f64* AQ;
    f64* AR;
    f64* taua;
    f64* B;
    f64* BF;
    f64* BZ;
    f64* BT;
    f64* BWK;
    f64* taub;
    f64* work;
    f64* rwork;
} dckgqr_workspace_t;

static dckgqr_workspace_t* g_ws = NULL;

static int group_setup(void** state)
{
    (void)state;

    g_ws = malloc(sizeof(dckgqr_workspace_t));
    if (!g_ws) return -1;

    int lda = NMAX;
    int ldb = NMAX;
    int lwork = NMAX * NMAX;

    g_ws->A     = malloc(lda * NMAX * sizeof(f64));
    g_ws->AF    = malloc(lda * NMAX * sizeof(f64));
    g_ws->AQ    = malloc(lda * NMAX * sizeof(f64));
    g_ws->AR    = malloc(lda * NMAX * sizeof(f64));
    g_ws->taua  = malloc(NMAX * sizeof(f64));
    g_ws->B     = malloc(ldb * NMAX * sizeof(f64));
    g_ws->BF    = malloc(ldb * NMAX * sizeof(f64));
    g_ws->BZ    = malloc(ldb * NMAX * sizeof(f64));
    g_ws->BT    = malloc(ldb * NMAX * sizeof(f64));
    g_ws->BWK   = malloc(ldb * NMAX * sizeof(f64));
    g_ws->taub  = malloc(NMAX * sizeof(f64));
    g_ws->work  = malloc(lwork * sizeof(f64));
    g_ws->rwork = malloc(NMAX * sizeof(f64));

    if (!g_ws->A || !g_ws->AF || !g_ws->AQ || !g_ws->AR ||
        !g_ws->taua || !g_ws->B || !g_ws->BF || !g_ws->BZ ||
        !g_ws->BT || !g_ws->BWK || !g_ws->taub ||
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
        free(g_ws->AQ);
        free(g_ws->AR);
        free(g_ws->taua);
        free(g_ws->B);
        free(g_ws->BF);
        free(g_ws->BZ);
        free(g_ws->BT);
        free(g_ws->BWK);
        free(g_ws->taub);
        free(g_ws->work);
        free(g_ws->rwork);
        free(g_ws);
        g_ws = NULL;
    }

    return 0;
}

static void test_dckgqr_case(void** state)
{
    dckgqr_params_t* params = *state;
    int m = MVAL[params->im];
    int p = PVAL[params->ip];
    int n = NVAL[params->in];
    int imat = params->imat;
    int lda = NMAX;
    int ldb = NMAX;
    int lwork = NMAX * NMAX;

    char type;
    int kla, kua, klb, kub;
    f64 anorm, bnorm, cndnma, cndnmb;
    int modea, modeb;
    char dista, distb;
    int iinfo;
    f64 result[NTESTS];

    /* Seed RNG for reproducibility */
    uint64_t rng_state[4];
    rng_seed(rng_state, 2024 + (uint64_t)params->im * 1000 +
             (uint64_t)params->ip * 100 + (uint64_t)params->in * 10 +
             (uint64_t)imat);

    /* ============================================================
     * Test DGGRQF
     * A: M-by-N, B: P-by-N
     * ============================================================ */

    dlatb9("GRQ", imat, m, p, n, &type, &kla, &kua, &klb, &kub,
           &anorm, &bnorm, &modea, &modeb, &cndnma, &cndnmb,
           &dista, &distb);

    /* Generate M by N matrix A */
    char dista_str[2] = {dista, '\0'};
    char type_str[2] = {type, '\0'};
    dlatms(m, n, dista_str, type_str, g_ws->rwork, modea, cndnma,
           anorm, kla, kua, "N", g_ws->A, lda, g_ws->work, &iinfo,
           rng_state);
    assert_int_equal(iinfo, 0);

    /* Generate P by N matrix B */
    char distb_str[2] = {distb, '\0'};
    dlatms(p, n, distb_str, type_str, g_ws->rwork, modeb, cndnmb,
           bnorm, klb, kub, "N", g_ws->B, ldb, g_ws->work, &iinfo,
           rng_state);
    assert_int_equal(iinfo, 0);

    dgrqts(m, p, n, g_ws->A, g_ws->AF, g_ws->AQ, g_ws->AR, lda,
           g_ws->taua, g_ws->B, g_ws->BF, g_ws->BZ, g_ws->BT,
           g_ws->BWK, ldb, g_ws->taub, g_ws->work, lwork,
           g_ws->rwork, result);

    for (int i = 0; i < NTESTS; i++) {
        if (result[i] >= THRESH) {
            print_message("  GRQ: M=%d P=%d N=%d type=%d test=%d ratio=%g\n",
                          m, p, n, imat, i + 1, result[i]);
        }
        assert_residual_below(result[i], THRESH);
    }

    /* ============================================================
     * Test DGGQRF
     * A: N-by-M, B: N-by-P
     * ============================================================ */

    dlatb9("GQR", imat, m, p, n, &type, &kla, &kua, &klb, &kub,
           &anorm, &bnorm, &modea, &modeb, &cndnma, &cndnmb,
           &dista, &distb);

    /* Generate N-by-M matrix A */
    dista_str[0] = dista;
    type_str[0] = type;
    dlatms(n, m, dista_str, type_str, g_ws->rwork, modea, cndnma,
           anorm, kla, kua, "N", g_ws->A, lda, g_ws->work, &iinfo,
           rng_state);
    assert_int_equal(iinfo, 0);

    /* Generate N-by-P matrix B */
    distb_str[0] = distb;
    dlatms(n, p, distb_str, type_str, g_ws->rwork, modea, cndnma,
           bnorm, klb, kub, "N", g_ws->B, ldb, g_ws->work, &iinfo,
           rng_state);
    assert_int_equal(iinfo, 0);

    dgqrts(n, m, p, g_ws->A, g_ws->AF, g_ws->AQ, g_ws->AR, lda,
           g_ws->taua, g_ws->B, g_ws->BF, g_ws->BZ, g_ws->BT,
           g_ws->BWK, ldb, g_ws->taub, g_ws->work, lwork,
           g_ws->rwork, result);

    for (int i = 0; i < NTESTS; i++) {
        if (result[i] >= THRESH) {
            print_message("  GQR: N=%d M=%d P=%d type=%d test=%d ratio=%g\n",
                          n, m, p, imat, i + 1, result[i]);
        }
        assert_residual_below(result[i], THRESH);
    }
}

/* Parameterized test arrays */
static dckgqr_params_t g_params[MAX_TESTS];
static struct CMUnitTest g_tests[MAX_TESTS];
static int g_num_tests = 0;

static void build_test_array(void)
{
    g_num_tests = 0;

    for (int im = 0; im < (int)NM; im++) {
        for (int ip = 0; ip < (int)NP; ip++) {
            for (int in = 0; in < (int)NN; in++) {
                for (int imat = 1; imat <= NTYPES; imat++) {
                    dckgqr_params_t* par = &g_params[g_num_tests];
                    par->im = im;
                    par->ip = ip;
                    par->in = in;
                    par->imat = imat;
                    snprintf(par->name, sizeof(par->name),
                             "GQR_m%d_p%d_n%d_type%d",
                             MVAL[im], PVAL[ip], NVAL[in], imat);

                    g_tests[g_num_tests].name = par->name;
                    g_tests[g_num_tests].test_func = test_dckgqr_case;
                    g_tests[g_num_tests].setup_func = NULL;
                    g_tests[g_num_tests].teardown_func = NULL;
                    g_tests[g_num_tests].initial_state = par;

                    g_num_tests++;
                }
            }
        }
    }
}

int main(void)
{
    build_test_array();

    return _cmocka_run_group_tests("dckgqr", g_tests, g_num_tests,
                                   group_setup, group_teardown);
}
