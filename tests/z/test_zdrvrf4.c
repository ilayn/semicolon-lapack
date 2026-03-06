/**
 * @file test_zdrvrf4.c
 * @brief Test suite for ZHFRK (RFP Hermitian rank-k update).
 *
 * Port of LAPACK's TESTING/LIN/zdrvrf4.f to C using CMocka.
 * Tests ZHFRK by comparing it against cblas_zherk on the same input:
 *   result = ||C_zherk - C_zhfrk||_I / (max(|alpha|*||A||^2 + |beta|*||C||, 1) * max(N,1) * eps)
 *
 * Note: The Fortran source has a typo at line 191 (K = NVAL(IIN) instead of
 * K = NVAL(IIK)). We use the correct loop variable for proper K coverage.
 */

#include "test_harness.h"
#include "verify.h"
#include "test_rng.h"
#include "semicolon_cblas.h"

#define THRESH 5.0
#define NMAX   50

static const INT NVAL[] = {0, 1, 2, 3, 5, 10, 50};
#define NN (sizeof(NVAL) / sizeof(NVAL[0]))

typedef struct {
    INT in;
    INT ik;
    INT iform;
    INT iuplo;
    INT itrans;
    INT ialpha;
    char name[80];
} zdrvrf4_params_t;

static void run_zdrvrf4_single(INT n, INT k, INT iform, INT iuplo,
                                INT itrans, INT ialpha)
{
    INT lda = NMAX;
    INT ldc = NMAX;
    INT info;
    char ctx[128];
    uint64_t rng_state[4];
    rng_seed(rng_state, 1988);

    f64 eps = dlamch("P");

    const char* cform = (iform == 0) ? "N" : "C";
    const char* uplo  = (iuplo == 0) ? "U" : "L";
    const char* trans = (itrans == 0) ? "N" : "C";

    f64 alpha, beta;
    if (ialpha == 1) {
        alpha = 0.0; beta = 0.0;
    } else if (ialpha == 2) {
        alpha = 1.0; beta = 0.0;
    } else if (ialpha == 3) {
        alpha = 0.0; beta = 1.0;
    } else {
        alpha = rng_uniform_symmetric(rng_state);
        beta = rng_uniform_symmetric(rng_state);
    }

    c128 A[NMAX * NMAX];
    c128 C1[NMAX * NMAX];
    c128 C2[NMAX * NMAX];
    c128 CRF[NMAX * (NMAX + 1) / 2];
    f64 D_WORK_ZLANGE[NMAX];

    INT rows_a, cols_a;
    if (itrans == 0) {
        rows_a = n;
        cols_a = k;
    } else {
        rows_a = k;
        cols_a = n;
    }

    for (INT j = 0; j < cols_a; j++)
        for (INT i = 0; i < rows_a; i++)
            A[i + j * lda] = zlarnd_rng(4, rng_state);

    f64 norma = zlange("I", rows_a, cols_a, A, lda, D_WORK_ZLANGE);

    for (INT j = 0; j < n; j++)
        for (INT i = 0; i < n; i++) {
            C1[i + j * ldc] = zlarnd_rng(4, rng_state);
            C2[i + j * ldc] = C1[i + j * ldc];
        }

    f64 normc = zlange("I", n, n, C1, ldc, D_WORK_ZLANGE);

    ztrttf(cform, uplo, n, C1, ldc, CRF, &info);

    CBLAS_UPLO uploC = (iuplo == 0) ? CblasUpper : CblasLower;
    CBLAS_TRANSPOSE transC = (itrans == 0) ? CblasNoTrans : CblasConjTrans;

    cblas_zherk(CblasColMajor, uploC, transC, n, k,
                alpha, A, lda, beta, C1, ldc);

    zhfrk(cform, uplo, trans, n, k, alpha, A, lda, beta, CRF);

    ztfttr(cform, uplo, n, CRF, C2, ldc, &info);

    for (INT j = 0; j < n; j++)
        for (INT i = 0; i < n; i++)
            C1[i + j * ldc] -= C2[i + j * ldc];

    f64 result = zlange("I", n, n, C1, ldc, D_WORK_ZLANGE);

    f64 denom = fabs(alpha) * norma * norma + fabs(beta) * normc;
    if (denom < 1.0) denom = 1.0;
    INT n_max = (n > 1) ? n : 1;
    result = result / denom / n_max / eps;

    snprintf(ctx, sizeof(ctx),
             "n=%d k=%d form=%s uplo=%s trans=%s alpha=%.2f beta=%.2f resid=%.6e",
             n, k, cform, uplo, trans, alpha, beta, result);
    set_test_context(ctx);
    assert_residual_below(result, THRESH);
    clear_test_context();
}

static void test_zdrvrf4_case(void** state)
{
    zdrvrf4_params_t* p = *state;
    run_zdrvrf4_single(NVAL[p->in], NVAL[p->ik], p->iform, p->iuplo,
                        p->itrans, p->ialpha);
}

/* 7x7x2x2x2x4 = 3136 cases */
#define MAX_TESTS (NN * NN * 2 * 2 * 2 * 4)

static zdrvrf4_params_t g_params[MAX_TESTS];
static struct CMUnitTest g_tests[MAX_TESTS];
static INT g_num_tests = 0;

static void build_test_array(void)
{
    g_num_tests = 0;

    for (INT in = 0; in < (INT)NN; in++) {
        for (INT ik = 0; ik < (INT)NN; ik++) {
            for (INT iform = 0; iform < 2; iform++) {
                for (INT iuplo = 0; iuplo < 2; iuplo++) {
                    for (INT itrans = 0; itrans < 2; itrans++) {
                        for (INT ialpha = 1; ialpha <= 4; ialpha++) {
                            zdrvrf4_params_t* p = &g_params[g_num_tests];
                            p->in = in;
                            p->ik = ik;
                            p->iform = iform;
                            p->iuplo = iuplo;
                            p->itrans = itrans;
                            p->ialpha = ialpha;
                            snprintf(p->name, sizeof(p->name),
                                     "zdrvrf4_n%d_k%d_%c_%c_%c_a%d",
                                     NVAL[in], NVAL[ik],
                                     (iform == 0) ? 'N' : 'C',
                                     (iuplo == 0) ? 'U' : 'L',
                                     (itrans == 0) ? 'N' : 'C',
                                     ialpha);

                            g_tests[g_num_tests].name = p->name;
                            g_tests[g_num_tests].test_func = test_zdrvrf4_case;
                            g_tests[g_num_tests].setup_func = NULL;
                            g_tests[g_num_tests].teardown_func = NULL;
                            g_tests[g_num_tests].initial_state = p;

                            g_num_tests++;
                        }
                    }
                }
            }
        }
    }
}

int main(void)
{
    build_test_array();

    if (g_num_tests == 0) {
        printf("No valid test cases generated\n");
        return 0;
    }

    (void)_cmocka_run_group_tests("zdrvrf4", g_tests, g_num_tests, NULL, NULL);
    return 0;
}
