/**
 * @file test_sdrvrf4.c
 * @brief Test suite for SSFRK (RFP symmetric rank-k update).
 *
 * Port of LAPACK's TESTING/LIN/ddrvrf4.f to C using CMocka.
 * Tests SSFRK by comparing it against cblas_dsyrk on the same input:
 *   result = ||C_dsyrk - C_dsfrk||_I / (max(|alpha|*||A|| + |beta|, 1) * max(N,1) * eps)
 *
 * Note: The Fortran source has a typo at line 193 (K = NVAL(IIN) instead of
 * K = NVAL(IIK)). We use the correct loop variable for proper K coverage.
 */

#include "test_harness.h"
#include "test_rng.h"
#include <cblas.h>

#define THRESH 5.0f
#define NMAX   50

static const int NVAL[] = {0, 1, 2, 3, 5, 10, 50};
#define NN (sizeof(NVAL) / sizeof(NVAL[0]))

extern f32 slamch(const char* cmach);
extern f32 slange(const char* norm, const int m, const int n,
                     const f32* A, const int lda, f32* work);
extern void strttf(const char* transr, const char* uplo, const int n,
                   const f32* A, const int lda, f32* ARF, int* info);
extern void stfttr(const char* transr, const char* uplo, const int n,
                   const f32* ARF, f32* A, const int lda, int* info);
extern void ssfrk(const char* transr, const char* uplo, const char* trans,
                  const int n, const int k, const f32 alpha,
                  const f32* A, const int lda, const f32 beta, f32* C);

typedef struct {
    int in;
    int ik;
    int iform;
    int iuplo;
    int itrans;
    int ialpha;
    char name[80];
} ddrvrf4_params_t;

static void run_ddrvrf4_single(int n, int k, int iform, int iuplo,
                                int itrans, int ialpha)
{
    int lda = NMAX;
    int ldc = NMAX;
    int info;
    char ctx[128];
    uint64_t rng_state[4];
    rng_seed(rng_state, 1988);

    f32 eps = slamch("P");

    const char* cform = (iform == 0) ? "N" : "T";
    const char* uplo  = (iuplo == 0) ? "U" : "L";
    const char* trans = (itrans == 0) ? "N" : "T";

    f32 alpha, beta;
    if (ialpha == 1) {
        alpha = 0.0f; beta = 0.0f;
    } else if (ialpha == 2) {
        alpha = 1.0f; beta = 0.0f;
    } else if (ialpha == 3) {
        alpha = 0.0f; beta = 1.0f;
    } else {
        alpha = rng_uniform_symmetric_f32(rng_state);
        beta = rng_uniform_symmetric_f32(rng_state);
    }

    f32 A[NMAX * NMAX];
    f32 C1[NMAX * NMAX];
    f32 C2[NMAX * NMAX];
    f32 CRF[NMAX * (NMAX + 1) / 2];
    f32 D_WORK_DLANGE[NMAX];

    int rows_a, cols_a;
    if (itrans == 0) {
        rows_a = n;
        cols_a = k;
    } else {
        rows_a = k;
        cols_a = n;
    }

    for (int j = 0; j < cols_a; j++)
        for (int i = 0; i < rows_a; i++)
            A[i + j * lda] = rng_uniform_symmetric_f32(rng_state);

    f32 norma = slange("I", rows_a, cols_a, A, lda, D_WORK_DLANGE);

    for (int j = 0; j < n; j++)
        for (int i = 0; i < n; i++) {
            C1[i + j * ldc] = rng_uniform_symmetric_f32(rng_state);
            C2[i + j * ldc] = C1[i + j * ldc];
        }

    strttf(cform, uplo, n, C1, ldc, CRF, &info);

    CBLAS_UPLO uploC = (iuplo == 0) ? CblasUpper : CblasLower;
    CBLAS_TRANSPOSE transC = (itrans == 0) ? CblasNoTrans : CblasTrans;

    cblas_ssyrk(CblasColMajor, uploC, transC, n, k,
                alpha, A, lda, beta, C1, ldc);

    ssfrk(cform, uplo, trans, n, k, alpha, A, lda, beta, CRF);

    stfttr(cform, uplo, n, CRF, C2, ldc, &info);

    for (int j = 0; j < n; j++)
        for (int i = 0; i < n; i++)
            C1[i + j * ldc] -= C2[i + j * ldc];

    f32 result = slange("I", n, n, C1, ldc, D_WORK_DLANGE);

    f32 denom = fabsf(alpha) * norma + fabsf(beta);
    if (denom < 1.0f) denom = 1.0f;
    int n_max = (n > 1) ? n : 1;
    result = result / denom / n_max / eps;

    snprintf(ctx, sizeof(ctx),
             "n=%d k=%d form=%s uplo=%s trans=%s alpha=%.2f beta=%.2f resid=%.6e",
             n, k, cform, uplo, trans, alpha, beta, result);
    set_test_context(ctx);
    assert_residual_below(result, THRESH);
    clear_test_context();
}

static void test_ddrvrf4_case(void** state)
{
    ddrvrf4_params_t* p = *state;
    run_ddrvrf4_single(NVAL[p->in], NVAL[p->ik], p->iform, p->iuplo,
                        p->itrans, p->ialpha);
}

/* 7×7×2×2×2×4 = 3136 cases */
#define MAX_TESTS (NN * NN * 2 * 2 * 2 * 4)

static ddrvrf4_params_t g_params[MAX_TESTS];
static struct CMUnitTest g_tests[MAX_TESTS];
static int g_num_tests = 0;

static void build_test_array(void)
{
    g_num_tests = 0;

    for (int in = 0; in < (int)NN; in++) {
        for (int ik = 0; ik < (int)NN; ik++) {
            for (int iform = 0; iform < 2; iform++) {
                for (int iuplo = 0; iuplo < 2; iuplo++) {
                    for (int itrans = 0; itrans < 2; itrans++) {
                        for (int ialpha = 1; ialpha <= 4; ialpha++) {
                            ddrvrf4_params_t* p = &g_params[g_num_tests];
                            p->in = in;
                            p->ik = ik;
                            p->iform = iform;
                            p->iuplo = iuplo;
                            p->itrans = itrans;
                            p->ialpha = ialpha;
                            snprintf(p->name, sizeof(p->name),
                                     "ddrvrf4_n%d_k%d_%c_%c_%c_a%d",
                                     NVAL[in], NVAL[ik],
                                     (iform == 0) ? 'N' : 'T',
                                     (iuplo == 0) ? 'U' : 'L',
                                     (itrans == 0) ? 'N' : 'T',
                                     ialpha);

                            g_tests[g_num_tests].name = p->name;
                            g_tests[g_num_tests].test_func = test_ddrvrf4_case;
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

    return _cmocka_run_group_tests("ddrvrf4", g_tests, g_num_tests, NULL, NULL);
}
