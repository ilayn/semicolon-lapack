/**
 * @file test_zdrvrf3.c
 * @brief Test suite for ZTFSM (RFP triangular solve).
 *
 * Port of LAPACK's TESTING/LIN/zdrvrf3.f to C using CMocka.
 * Tests ZTFSM by comparing it against cblas_ztrsm on the same input:
 *   result = ||B_ztfsm - B_ztrsm||_I / (sqrt(eps) * max(M,N))
 *
 * To get a well-conditioned triangular matrix, the R factor of a
 * QR factorization (upper) or L factor of an LQ factorization (lower)
 * of a random matrix is used.
 */

#include "test_harness.h"
#include "verify.h"
#include "test_rng.h"
#include "semicolon_cblas.h"

#define THRESH 1.0
#define NMAX   50

static const INT NVAL[] = {0, 1, 2, 3, 5, 10, 50};
#define NN (sizeof(NVAL) / sizeof(NVAL[0]))

typedef struct {
    INT im;
    INT in;
    INT iform;
    INT iuplo;
    INT iside;
    INT itrans;
    INT idiag;
    INT ialpha;
    char name[80];
} zdrvrf3_params_t;

static void run_zdrvrf3_single(INT m, INT n, INT iform, INT iuplo,
                                INT iside, INT itrans, INT idiag, INT ialpha)
{
    INT lda = NMAX;
    INT info;
    char ctx[128];
    uint64_t rng_state[4];
    rng_seed(rng_state, 1988);

    f64 eps = dlamch("P");

    const char* cform = (iform == 0) ? "N" : "C";
    const char* uplo  = (iuplo == 0) ? "U" : "L";
    const char* side  = (iside == 0) ? "L" : "R";
    const char* trans = (itrans == 0) ? "N" : "C";
    const char* diag  = (idiag == 0) ? "N" : "U";

    c128 alpha;
    if (ialpha == 1)
        alpha = CMPLX(0.0, 0.0);
    else if (ialpha == 2)
        alpha = CMPLX(1.0, 0.0);
    else
        alpha = zlarnd_rng(4, rng_state);

    INT na = (iside == 0) ? m : n;

    c128 A[NMAX * NMAX];
    c128 ARF[NMAX * (NMAX + 1) / 2];
    c128 B1[NMAX * NMAX];
    c128 B2[NMAX * NMAX];
    c128 TAU[NMAX];
    c128 Z_WORK_ZGEQRF[NMAX];
    f64 D_WORK_ZLANGE[NMAX];

    for (INT j = 0; j < na; j++)
        for (INT i = 0; i < na; i++)
            A[i + j * lda] = zlarnd_rng(4, rng_state);

    if (iuplo == 0) {
        zgeqrf(na, na, A, lda, TAU, Z_WORK_ZGEQRF, lda, &info);

        if (idiag == 1) {
            for (INT j = 0; j < na; j++)
                for (INT i = 0; i <= j; i++)
                    A[i + j * lda] /= (2.0 * A[j + j * lda]);
        }
    } else {
        zgelqf(na, na, A, lda, TAU, Z_WORK_ZGEQRF, lda, &info);

        if (idiag == 1) {
            for (INT i = 0; i < na; i++)
                for (INT j = 0; j <= i; j++)
                    A[i + j * lda] /= (2.0 * A[i + i * lda]);
        }
    }

    for (INT j = 0; j < na; j++)
        A[j + j * lda] *= zlarnd_rng(5, rng_state);

    ztrttf(cform, uplo, na, A, lda, ARF, &info);

    for (INT j = 0; j < n; j++)
        for (INT i = 0; i < m; i++) {
            B1[i + j * lda] = zlarnd_rng(4, rng_state);
            B2[i + j * lda] = B1[i + j * lda];
        }

    CBLAS_SIDE sideC   = (iside == 0) ? CblasLeft : CblasRight;
    CBLAS_UPLO uploC   = (iuplo == 0) ? CblasUpper : CblasLower;
    CBLAS_TRANSPOSE transC = (itrans == 0) ? CblasNoTrans : CblasConjTrans;
    CBLAS_DIAG diagC   = (idiag == 0) ? CblasNonUnit : CblasUnit;

    cblas_ztrsm(CblasColMajor, sideC, uploC, transC, diagC, m, n,
                &alpha, A, lda, B1, lda);

    ztfsm(cform, side, uplo, trans, diag, m, n, alpha, ARF, B2, lda);

    for (INT j = 0; j < n; j++)
        for (INT i = 0; i < m; i++)
            B1[i + j * lda] = B2[i + j * lda] - B1[i + j * lda];

    f64 result = zlange("I", m, n, B1, lda, D_WORK_ZLANGE);

    INT mn_max = (m > n) ? m : n;
    if (mn_max < 1) mn_max = 1;
    result = result / sqrt(eps) / mn_max;

    snprintf(ctx, sizeof(ctx),
             "m=%d n=%d form=%s side=%s uplo=%s trans=%s diag=%s resid=%.6e",
             m, n, cform, side, uplo, trans, diag, result);
    set_test_context(ctx);
    assert_residual_below(result, THRESH);
    clear_test_context();
}

static void test_zdrvrf3_case(void** state)
{
    zdrvrf3_params_t* p = *state;
    run_zdrvrf3_single(NVAL[p->im], NVAL[p->in], p->iform, p->iuplo,
                        p->iside, p->itrans, p->idiag, p->ialpha);
}

/* 7x7x2x2x2x2x2x3 = 4704 cases */
#define MAX_TESTS (NN * NN * 2 * 2 * 2 * 2 * 2 * 3)

static zdrvrf3_params_t g_params[MAX_TESTS];
static struct CMUnitTest g_tests[MAX_TESTS];
static INT g_num_tests = 0;

static void build_test_array(void)
{
    g_num_tests = 0;

    for (INT im = 0; im < (INT)NN; im++) {
        for (INT in = 0; in < (INT)NN; in++) {
            for (INT iform = 0; iform < 2; iform++) {
                for (INT iuplo = 0; iuplo < 2; iuplo++) {
                    for (INT iside = 0; iside < 2; iside++) {
                        for (INT itrans = 0; itrans < 2; itrans++) {
                            for (INT idiag = 0; idiag < 2; idiag++) {
                                for (INT ialpha = 1; ialpha <= 3; ialpha++) {
                                    zdrvrf3_params_t* p = &g_params[g_num_tests];
                                    p->im = im;
                                    p->in = in;
                                    p->iform = iform;
                                    p->iuplo = iuplo;
                                    p->iside = iside;
                                    p->itrans = itrans;
                                    p->idiag = idiag;
                                    p->ialpha = ialpha;
                                    snprintf(p->name, sizeof(p->name),
                                             "zdrvrf3_m%d_n%d_%c_%c_%c_%c_%c_a%d",
                                             NVAL[im], NVAL[in],
                                             (iform == 0) ? 'N' : 'C',
                                             (iuplo == 0) ? 'U' : 'L',
                                             (iside == 0) ? 'L' : 'R',
                                             (itrans == 0) ? 'N' : 'C',
                                             (idiag == 0) ? 'N' : 'U',
                                             ialpha);

                                    g_tests[g_num_tests].name = p->name;
                                    g_tests[g_num_tests].test_func = test_zdrvrf3_case;
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

    return _cmocka_run_group_tests("zdrvrf3", g_tests, g_num_tests, NULL, NULL);
}
