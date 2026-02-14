/**
 * @file test_ddrvrf1.c
 * @brief Test suite for DLANSF (RFP symmetric matrix norm).
 *
 * Port of LAPACK's TESTING/LIN/ddrvrf1.f to C using CMocka.
 * Tests DLANSF by comparing it against DLANSY on the same matrix:
 *   result = |DLANSY(A) - DLANSF(ARF)| / (DLANSY(A) * eps)
 *
 * For each (N, scaling, UPLO, CFORM) combination, all four norms
 * ('M', '1', 'I', 'F') are checked within a single test case.
 */

#include "test_harness.h"
#include "test_rng.h"

#define THRESH 2.0
#define NMAX   50

static const int NVAL[] = {0, 1, 2, 3, 5, 10, 50};
#define NN (sizeof(NVAL) / sizeof(NVAL[0]))

extern f64 dlamch(const char* cmach);
extern f64 dlansf(const char* norm, const char* transr, const char* uplo,
                     const int n, const f64* A, f64* work);
extern f64 dlansy(const char* norm, const char* uplo, const int n,
                     const f64* A, const int lda, f64* work);
extern void dtrttf(const char* transr, const char* uplo, const int n,
                   const f64* A, const int lda, f64* ARF, int* info);

typedef struct {
    int in;
    int iit;
    int iuplo;
    int iform;
    char name[80];
} ddrvrf1_params_t;

static void run_ddrvrf1_single(int n, int iit, int iuplo, int iform)
{
    if (n == 0) return;

    int lda = NMAX;
    int info;
    char ctx[128];
    uint64_t rng_state[4];
    rng_seed(rng_state, 1988);

    const char* uplo = (iuplo == 0) ? "U" : "L";
    const char* cform = (iform == 0) ? "N" : "T";

    f64 eps = dlamch("P");
    f64 small_val = dlamch("S");
    f64 large_val = 1.0 / small_val;
    small_val = small_val * lda * lda;
    large_val = large_val / lda / lda;

    f64 A[NMAX * NMAX];
    f64 ARF[NMAX * (NMAX + 1) / 2];
    f64 WORK[NMAX];

    for (int j = 0; j < n; j++)
        for (int i = 0; i < n; i++)
            A[i + j * lda] = rng_uniform_symmetric(rng_state);

    if (iit == 2) {
        for (int j = 0; j < n; j++)
            for (int i = 0; i < n; i++)
                A[i + j * lda] *= large_val;
    }

    if (iit == 3) {
        for (int j = 0; j < n; j++)
            for (int i = 0; i < n; i++)
                A[i + j * lda] *= small_val;
    }

    dtrttf(cform, uplo, n, A, lda, ARF, &info);

    if (info != 0) {
        snprintf(ctx, sizeof(ctx), "n=%d iit=%d uplo=%s form=%s DTRTTF info=%d",
                 n, iit, uplo, cform, info);
        set_test_context(ctx);
        assert_info_success(info);
        return;
    }

    const char* norms[] = {"M", "1", "I", "F"};

    for (int inorm = 0; inorm < 4; inorm++) {
        f64 normarf = dlansf(norms[inorm], cform, uplo, n, ARF, WORK);
        f64 norma = dlansy(norms[inorm], uplo, n, A, lda, WORK);

        f64 result;
        if (norma == 0.0) {
            result = normarf / eps;
        } else {
            result = (norma - normarf) / norma / eps;
        }

        snprintf(ctx, sizeof(ctx), "n=%d iit=%d uplo=%s form=%s norm=%s resid=%.6e",
                 n, iit, uplo, cform, norms[inorm], result);
        set_test_context(ctx);
        assert_residual_below(result, THRESH);
    }

    clear_test_context();
}

static void test_ddrvrf1_case(void** state)
{
    ddrvrf1_params_t* p = *state;
    run_ddrvrf1_single(NVAL[p->in], p->iit, p->iuplo, p->iform);
}

/* N=0 exits immediately so skip it: 6 N × 3 scalings × 2 uplo × 2 form = 72 */
#define MAX_TESTS (NN * 3 * 2 * 2)

static ddrvrf1_params_t g_params[MAX_TESTS];
static struct CMUnitTest g_tests[MAX_TESTS];
static int g_num_tests = 0;

static void build_test_array(void)
{
    g_num_tests = 0;

    for (int in = 0; in < (int)NN; in++) {
        if (NVAL[in] == 0) continue;

        for (int iit = 1; iit <= 3; iit++) {
            for (int iuplo = 0; iuplo < 2; iuplo++) {
                for (int iform = 0; iform < 2; iform++) {
                    ddrvrf1_params_t* p = &g_params[g_num_tests];
                    p->in = in;
                    p->iit = iit;
                    p->iuplo = iuplo;
                    p->iform = iform;
                    snprintf(p->name, sizeof(p->name),
                             "ddrvrf1_n%d_scl%d_%c_%c",
                             NVAL[in], iit,
                             (iuplo == 0) ? 'U' : 'L',
                             (iform == 0) ? 'N' : 'T');

                    g_tests[g_num_tests].name = p->name;
                    g_tests[g_num_tests].test_func = test_ddrvrf1_case;
                    g_tests[g_num_tests].setup_func = NULL;
                    g_tests[g_num_tests].teardown_func = NULL;
                    g_tests[g_num_tests].initial_state = p;

                    g_num_tests++;
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

    return _cmocka_run_group_tests("ddrvrf1", g_tests, g_num_tests, NULL, NULL);
}
