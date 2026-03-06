/**
 * @file test_zdrvrf2.c
 * @brief Test suite for RFP conversion routines roundtrip (complex).
 *
 * Port of LAPACK's TESTING/LIN/zdrvrf2.f to C using CMocka.
 * Tests 6 conversion routines via two roundtrip chains:
 *   Chain 1: dense -> RFP -> packed -> dense  (ZTRTTF, ZTFTTP, ZTPTTR)
 *   Chain 2: dense -> packed -> RFP -> dense  (ZTRTTP, ZTPTTF, ZTFTTR)
 *
 * Verification: exact element-by-element equality of the relevant triangle.
 */

#include "test_harness.h"
#include "verify.h"
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
} zdrvrf2_params_t;

static void run_zdrvrf2_single(INT n, INT iuplo, INT iform)
{
    INT lda = NMAX;
    INT info;
    char ctx[128];
    uint64_t rng_state[4];
    rng_seed(rng_state, 1988);

    const char* uplo = (iuplo == 0) ? "U" : "L";
    const char* cform = (iform == 0) ? "N" : "C";
    INT lower = (iuplo == 1);

    c128 A[NMAX * NMAX];
    memset(A, 0, sizeof(A));
    c128 ASAV[NMAX * NMAX];
    c128 ARF[NPP_MAX];
    c128 AP[NPP_MAX];

    for (INT j = 0; j < n; j++)
        for (INT i = 0; i < n; i++)
            A[i + j * lda] = zlarnd_rng(4, rng_state);

    /* Chain 1: dense -> RFP -> packed -> dense */
    ztrttf(cform, uplo, n, A, lda, ARF, &info);
    ztfttp(cform, uplo, n, ARF, AP, &info);
    ztpttr(uplo, n, AP, ASAV, lda, &info);

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
    ztrttp(uplo, n, A, lda, AP, &info);
    ztpttf(cform, uplo, n, AP, ARF, &info);
    ztfttr(cform, uplo, n, ARF, ASAV, lda, &info);

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

static void test_zdrvrf2_case(void** state)
{
    zdrvrf2_params_t* p = *state;
    run_zdrvrf2_single(NVAL[p->in], p->iuplo, p->iform);
}

/* 7 N x 2 uplo x 2 form = 28 cases */
#define MAX_TESTS (NN * 2 * 2)

static zdrvrf2_params_t g_params[MAX_TESTS];
static struct CMUnitTest g_tests[MAX_TESTS];
static INT g_num_tests = 0;

static void build_test_array(void)
{
    g_num_tests = 0;

    for (INT in = 0; in < (INT)NN; in++) {
        for (INT iuplo = 0; iuplo < 2; iuplo++) {
            for (INT iform = 0; iform < 2; iform++) {
                zdrvrf2_params_t* p = &g_params[g_num_tests];
                p->in = in;
                p->iuplo = iuplo;
                p->iform = iform;
                snprintf(p->name, sizeof(p->name),
                         "zdrvrf2_n%d_%c_%c",
                         NVAL[in],
                         (iuplo == 0) ? 'U' : 'L',
                         (iform == 0) ? 'N' : 'C');

                g_tests[g_num_tests].name = p->name;
                g_tests[g_num_tests].test_func = test_zdrvrf2_case;
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

    (void)_cmocka_run_group_tests("zdrvrf2", g_tests, g_num_tests, NULL, NULL);
    return 0;
}
