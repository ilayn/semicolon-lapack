/**
 * @file test_ddrvrf3.c
 * @brief Test suite for DTFSM (RFP triangular solve).
 *
 * Port of LAPACK's TESTING/LIN/ddrvrf3.f to C using CMocka.
 * Tests DTFSM by comparing it against cblas_dtrsm on the same input:
 *   result = ||B_dtfsm - B_dtrsm||_I / (sqrt(eps) * max(M,N))
 *
 * To get a well-conditioned triangular matrix, the R factor of a
 * QR factorization (upper) or L factor of an LQ factorization (lower)
 * of a random matrix is used.
 */

#include "test_harness.h"
#include "test_rng.h"
#include <cblas.h>

#define THRESH 1.0
#define NMAX   50

static const int NVAL[] = {0, 1, 2, 3, 5, 10, 50};
#define NN (sizeof(NVAL) / sizeof(NVAL[0]))

extern f64 dlamch(const char* cmach);
extern f64 dlange(const char* norm, const int m, const int n,
                     const f64* A, const int lda, f64* work);
extern void dtrttf(const char* transr, const char* uplo, const int n,
                   const f64* A, const int lda, f64* ARF, int* info);
extern void dtfsm(const char* transr, const char* side, const char* uplo,
                  const char* trans, const char* diag, const int m, const int n,
                  const f64 alpha, const f64* A, f64* B, const int ldb);
extern void dgeqrf(const int m, const int n, f64* A, const int lda,
                   f64* tau, f64* work, const int lwork, int* info);
extern void dgelqf(const int m, const int n, f64* A, const int lda,
                   f64* tau, f64* work, const int lwork, int* info);

typedef struct {
    int im;
    int in;
    int iform;
    int iuplo;
    int iside;
    int itrans;
    int idiag;
    int ialpha;
    char name[80];
} ddrvrf3_params_t;

static void run_ddrvrf3_single(int m, int n, int iform, int iuplo,
                                int iside, int itrans, int idiag, int ialpha)
{
    int lda = NMAX;
    int info;
    char ctx[128];
    uint64_t rng_state[4];
    rng_seed(rng_state, 1988);

    f64 eps = dlamch("P");

    const char* cform = (iform == 0) ? "N" : "T";
    const char* uplo  = (iuplo == 0) ? "U" : "L";
    const char* side  = (iside == 0) ? "L" : "R";
    const char* trans = (itrans == 0) ? "N" : "T";
    const char* diag  = (idiag == 0) ? "N" : "U";

    f64 alpha;
    if (ialpha == 1)
        alpha = 0.0;
    else if (ialpha == 2)
        alpha = 1.0;
    else
        alpha = rng_uniform_symmetric(rng_state);

    int na = (iside == 0) ? m : n;

    f64 A[NMAX * NMAX];
    f64 ARF[NMAX * (NMAX + 1) / 2];
    f64 B1[NMAX * NMAX];
    f64 B2[NMAX * NMAX];
    f64 TAU[NMAX];
    f64 D_WORK_DGEQRF[NMAX];
    f64 D_WORK_DLANGE[NMAX];

    for (int j = 0; j < na; j++)
        for (int i = 0; i < na; i++)
            A[i + j * lda] = rng_uniform_symmetric(rng_state);

    if (iuplo == 0) {
        dgeqrf(na, na, A, lda, TAU, D_WORK_DGEQRF, lda, &info);

        if (idiag == 1) {
            for (int j = 0; j < na; j++)
                for (int i = 0; i <= j; i++)
                    A[i + j * lda] /= (2.0 * A[j + j * lda]);
        }
    } else {
        dgelqf(na, na, A, lda, TAU, D_WORK_DGEQRF, lda, &info);

        if (idiag == 1) {
            for (int i = 0; i < na; i++)
                for (int j = 0; j <= i; j++)
                    A[i + j * lda] /= (2.0 * A[i + i * lda]);
        }
    }

    dtrttf(cform, uplo, na, A, lda, ARF, &info);

    for (int j = 0; j < n; j++)
        for (int i = 0; i < m; i++) {
            B1[i + j * lda] = rng_uniform_symmetric(rng_state);
            B2[i + j * lda] = B1[i + j * lda];
        }

    CBLAS_SIDE sideC   = (iside == 0) ? CblasLeft : CblasRight;
    CBLAS_UPLO uploC   = (iuplo == 0) ? CblasUpper : CblasLower;
    CBLAS_TRANSPOSE transC = (itrans == 0) ? CblasNoTrans : CblasTrans;
    CBLAS_DIAG diagC   = (idiag == 0) ? CblasNonUnit : CblasUnit;

    cblas_dtrsm(CblasColMajor, sideC, uploC, transC, diagC, m, n,
                alpha, A, lda, B1, lda);

    dtfsm(cform, side, uplo, trans, diag, m, n, alpha, ARF, B2, lda);

    for (int j = 0; j < n; j++)
        for (int i = 0; i < m; i++)
            B1[i + j * lda] = B2[i + j * lda] - B1[i + j * lda];

    f64 result = dlange("I", m, n, B1, lda, D_WORK_DLANGE);

    int mn_max = (m > n) ? m : n;
    if (mn_max < 1) mn_max = 1;
    result = result / sqrt(eps) / mn_max;

    snprintf(ctx, sizeof(ctx),
             "m=%d n=%d form=%s side=%s uplo=%s trans=%s diag=%s alpha=%.2f resid=%.6e",
             m, n, cform, side, uplo, trans, diag, alpha, result);
    set_test_context(ctx);
    assert_residual_below(result, THRESH);
    clear_test_context();
}

static void test_ddrvrf3_case(void** state)
{
    ddrvrf3_params_t* p = *state;
    run_ddrvrf3_single(NVAL[p->im], NVAL[p->in], p->iform, p->iuplo,
                        p->iside, p->itrans, p->idiag, p->ialpha);
}

/* 7×7×2×2×2×2×2×3 = 4704 cases */
#define MAX_TESTS (NN * NN * 2 * 2 * 2 * 2 * 2 * 3)

static ddrvrf3_params_t g_params[MAX_TESTS];
static struct CMUnitTest g_tests[MAX_TESTS];
static int g_num_tests = 0;

static void build_test_array(void)
{
    g_num_tests = 0;
    const char sides[] = {'L', 'R'};
    const char uplos[] = {'U', 'L'};
    const char transs[] = {'N', 'T'};
    const char diags[] = {'N', 'U'};

    for (int im = 0; im < (int)NN; im++) {
        for (int in = 0; in < (int)NN; in++) {
            for (int iform = 0; iform < 2; iform++) {
                for (int iuplo = 0; iuplo < 2; iuplo++) {
                    for (int iside = 0; iside < 2; iside++) {
                        for (int itrans = 0; itrans < 2; itrans++) {
                            for (int idiag = 0; idiag < 2; idiag++) {
                                for (int ialpha = 1; ialpha <= 3; ialpha++) {
                                    ddrvrf3_params_t* p = &g_params[g_num_tests];
                                    p->im = im;
                                    p->in = in;
                                    p->iform = iform;
                                    p->iuplo = iuplo;
                                    p->iside = iside;
                                    p->itrans = itrans;
                                    p->idiag = idiag;
                                    p->ialpha = ialpha;
                                    snprintf(p->name, sizeof(p->name),
                                             "ddrvrf3_m%d_n%d_%c_%c_%c_%c_%c_a%d",
                                             NVAL[im], NVAL[in],
                                             (iform == 0) ? 'N' : 'T',
                                             uplos[iuplo], sides[iside],
                                             transs[itrans], diags[idiag],
                                             ialpha);

                                    g_tests[g_num_tests].name = p->name;
                                    g_tests[g_num_tests].test_func = test_ddrvrf3_case;
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
    }
}

int main(void)
{
    build_test_array();

    if (g_num_tests == 0) {
        printf("No valid test cases generated\n");
        return 0;
    }

    return _cmocka_run_group_tests("ddrvrf3", g_tests, g_num_tests, NULL, NULL);
}
