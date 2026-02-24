/**
 * @file test_sdrvrf2.c
 * @brief Test suite for RFP conversion routines roundtrip.
 *
 * Port of LAPACK's TESTING/LIN/ddrvrf2.f to C using CMocka.
 * Tests 6 conversion routines via two roundtrip chains:
 *   Chain 1: dense -> RFP -> packed -> dense  (STRTTF, STFTTP, STPTTR)
 *   Chain 2: dense -> packed -> RFP -> dense  (STRTTP, STPTTF, STFTTR)
 *
 * Verification: exact element-by-element equality of the relevant triangle.
 */

#include "test_harness.h"
#include "test_rng.h"

#define NMAX    50
#define NPP_MAX (NMAX * (NMAX + 1) / 2)

static const INT NVAL[] = {0, 1, 2, 3, 5, 10, 50};
#define NN (sizeof(NVAL) / sizeof(NVAL[0]))

typedef struct {
    INT in;
    INT iuplo;
    INT iform;
    char name[80];
} ddrvrf2_params_t;

static void run_ddrvrf2_single(INT n, INT iuplo, INT iform)
{
    INT lda = NMAX;
    INT info;
    char ctx[128];
    uint64_t rng_state[4];
    rng_seed(rng_state, 1988);

    const char* uplo = (iuplo == 0) ? "U" : "L";
    const char* cform = (iform == 0) ? "N" : "T";
    INT lower = (iuplo == 1);

    f32 A[NMAX * NMAX] = {0};
    f32 ASAV[NMAX * NMAX];
    f32 ARF[NPP_MAX];
    f32 AP[NPP_MAX];

    for (INT j = 0; j < n; j++)
        for (INT i = 0; i < n; i++)
            A[i + j * lda] = rng_uniform_symmetric_f32(rng_state);

    /* Chain 1: dense -> RFP -> packed -> dense */
    strttf(cform, uplo, n, A, lda, ARF, &info);
    stfttp(cform, uplo, n, ARF, AP, &info);
    stpttr(uplo, n, AP, ASAV, lda, &info);

    INT ok1 = 1;
    if (lower) {
        for (INT j = 0; j < n; j++)
            for (INT i = j; i < n; i++)
                if (A[i + j * lda] != ASAV[i + j * lda])
                    ok1 = 0;
    } else {
        for (INT j = 0; j < n; j++)
            for (INT i = 0; i <= j; i++)
                if (A[i + j * lda] != ASAV[i + j * lda])
                    ok1 = 0;
    }

    /* Chain 2: dense -> packed -> RFP -> dense */
    strttp(uplo, n, A, lda, AP, &info);
    stpttf(cform, uplo, n, AP, ARF, &info);
    stfttr(cform, uplo, n, ARF, ASAV, lda, &info);

    INT ok2 = 1;
    if (lower) {
        for (INT j = 0; j < n; j++)
            for (INT i = j; i < n; i++)
                if (A[i + j * lda] != ASAV[i + j * lda])
                    ok2 = 0;
    } else {
        for (INT j = 0; j < n; j++)
            for (INT i = 0; i <= j; i++)
                if (A[i + j * lda] != ASAV[i + j * lda])
                    ok2 = 0;
    }

    snprintf(ctx, sizeof(ctx), "n=%d uplo=%s form=%s chain1=%s chain2=%s",
             n, uplo, cform, ok1 ? "ok" : "FAIL", ok2 ? "ok" : "FAIL");
    set_test_context(ctx);
    assert_true(ok1 && ok2);
    clear_test_context();
}

static void test_ddrvrf2_case(void** state)
{
    ddrvrf2_params_t* p = *state;
    run_ddrvrf2_single(NVAL[p->in], p->iuplo, p->iform);
}

/* 7 N × 2 uplo × 2 form = 28 cases */
#define MAX_TESTS (NN * 2 * 2)

static ddrvrf2_params_t g_params[MAX_TESTS];
static struct CMUnitTest g_tests[MAX_TESTS];
static INT g_num_tests = 0;

static void build_test_array(void)
{
    g_num_tests = 0;

    for (INT in = 0; in < (INT)NN; in++) {
        for (INT iuplo = 0; iuplo < 2; iuplo++) {
            for (INT iform = 0; iform < 2; iform++) {
                ddrvrf2_params_t* p = &g_params[g_num_tests];
                p->in = in;
                p->iuplo = iuplo;
                p->iform = iform;
                snprintf(p->name, sizeof(p->name),
                         "ddrvrf2_n%d_%c_%c",
                         NVAL[in],
                         (iuplo == 0) ? 'U' : 'L',
                         (iform == 0) ? 'N' : 'T');

                g_tests[g_num_tests].name = p->name;
                g_tests[g_num_tests].test_func = test_ddrvrf2_case;
                g_tests[g_num_tests].setup_func = NULL;
                g_tests[g_num_tests].teardown_func = NULL;
                g_tests[g_num_tests].initial_state = p;

                g_num_tests++;
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

    return _cmocka_run_group_tests("ddrvrf2", g_tests, g_num_tests, NULL, NULL);
}
