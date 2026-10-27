/**
 * @file test_zwq_rwork.c
 * @brief Workspace queries must return exact RWORK sizes for the z
 *        divide-and-conquer eigensolvers.
 *
 * Port of LAPACK TESTING/EIG/test_wq_rwork.f.
 *
 * TEST_WQ_RWORK validates that workspace query calls (LWORK=-1,
 * LRWORK=-1, LIWORK=-1) return exact RWORK sizes for COMPLEX*16
 * routines with O(N^2) LRWMIN formulas.
 *
 * When LRWMIN > 2^24 (approx N > 2896 for formula 1+5N+2N^2),
 * storing the value through a REAL (float32) intermediary loses
 * precision. This test catches that regression by checking
 * INT(RWORK(1)) == expected at N values above the threshold.
 *
 * No large matrices are allocated -- workspace queries return
 * immediately after storing sizes, so the test runs in microseconds.
 */

#include "test_harness.h"
#include "verify.h"

/* Test N values: 1000 (below 2^24 threshold), 3000 and 5000 (above) */
static const INT NVALS[] = {1000, 3000, 5000};
#define NNVALS ((INT)(sizeof(NVALS) / sizeof(NVALS[0])))

enum { ZHEEVD, ZHEGVD, ZHBEVD, ZHPEVD, ZHPGVD, ZHBGVD, ZSTEDC, NROUTINES };
static const char* const NAMES[NROUTINES] = {
    "zheevd", "zhegvd", "zhbevd", "zhpevd", "zhpgvd", "zhbgvd", "zstedc"};

typedef struct {
    INT routine;
    INT n;
    char name[32];
} wq_params_t;

static void test_wq_rwork(void** state)
{
    const wq_params_t* p = *state;
    const INT n = p->n;

    /* Minimal dummy arrays for workspace queries (never accessed
     * by the routines when LWORK=-1). */
    INT iwork[1];
    c128 a[1], b[1], ab[1], bb[1], ap[1], bp[1], z[1], work[1];
    f64 w[1], rwork[1], d[1], e[1];
    INT info = 0;
    INT lrwexp;

    if (p->routine == ZSTEDC) {
        /* Expected: 1 + 4*N + 2*N**2 */
        lrwexp = 1 + 4 * n + 2 * n * n;
    } else {
        /* Expected LRWMIN for JOBZ='V': 1 + 5*N + 2*N**2
         * (common to ZHEEVD, ZHEGVD, ZHBEVD, ZHPEVD, ZHPGVD, ZHBGVD) */
        lrwexp = 1 + 5 * n + 2 * n * n;
    }

    switch (p->routine) {
    case ZHEEVD:
        zheevd("V", "U", n, a, n, w, work, -1, rwork, -1, iwork, -1, &info);
        break;
    case ZHEGVD:
        zhegvd(1, "V", "U", n, a, n, b, n, w, work, -1, rwork, -1, iwork, -1, &info);
        break;
    case ZHBEVD:
        /* KD=0 (diagonal band matrix), LDAB=1, LDZ=N */
        zhbevd("V", "U", n, 0, ab, 1, w, z, n, work, -1, rwork, -1, iwork, -1, &info);
        break;
    case ZHPEVD:
        /* LDZ=N */
        zhpevd("V", "U", n, ap, w, z, n, work, -1, rwork, -1, iwork, -1, &info);
        break;
    case ZHPGVD:
        /* ITYPE=1, LDZ=N */
        zhpgvd(1, "V", "U", n, ap, bp, w, z, n, work, -1, rwork, -1, iwork, -1, &info);
        break;
    case ZHBGVD:
        /* KA=0, KB=0, LDAB=1, LDBB=1, LDZ=N */
        zhbgvd("V", "U", n, 0, 0, ab, 1, bb, 1, w, z, n, work, -1, rwork, -1,
               iwork, -1, &info);
        break;
    default:
        zstedc("I", n, d, e, z, n, work, -1, rwork, -1, iwork, -1, &info);
        break;
    }

    assert_int_equal(info, 0);
    assert_int_equal((INT)rwork[0], lrwexp);
}

static wq_params_t g_params[NROUTINES * NNVALS];
static struct CMUnitTest g_tests[NROUTINES * NNVALS];

int main(void)
{
    INT nt = 0;
    for (INT i = 0; i < NNVALS; i++) {
        for (INT r = 0; r < NROUTINES; r++) {
            wq_params_t* p = &g_params[nt];
            p->routine = r;
            p->n = NVALS[i];
            snprintf(p->name, sizeof(p->name), "%s N=%d", NAMES[r], (int)NVALS[i]);
            g_tests[nt].name = p->name;
            g_tests[nt].test_func = test_wq_rwork;
            g_tests[nt].initial_state = p;
            g_tests[nt].setup_func = NULL;
            g_tests[nt].teardown_func = NULL;
            nt++;
        }
    }
    (void)cmocka_run_group_tests_name("zwq_rwork", g_tests, NULL, NULL);
    return 0;
}
