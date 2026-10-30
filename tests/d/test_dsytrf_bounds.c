/**
 * @file test_dsytrf_bounds.c
 * @brief DSYTRF family regression test for writes outside A's triangle and WORK.
 *
 * DSYTRF, DSYTRF_ROOK, DSYTRF_RK and DSYTRF_AA leave the triangle of A not
 * selected by UPLO unreferenced and need only the queried LWORK entries of
 * WORK. Each case factors a random symmetric matrix large enough for the
 * blocked code with exactly the queried LWORK, and checks that the other
 * triangle and the elements after WORK are left untouched.
 */

#include "test_harness.h"
#include "verify.h"
#include "test_rng.h"
#include <stdlib.h>

#define CANARY -12345.0
#define NGUARD 256

typedef enum { BK, ROOK, RK, AA } variant_t;

static void factor(const variant_t v, const char* uplo, const INT n, f64* A,
                   f64* E, INT* ipiv, f64* work, const INT lwork, INT* info)
{
    switch (v) {
    case BK:
        dsytrf(uplo, n, A, n, ipiv, work, lwork, info);
        break;
    case ROOK:
        dsytrf_rook(uplo, n, A, n, ipiv, work, lwork, info);
        break;
    case RK:
        dsytrf_rk(uplo, n, A, n, E, ipiv, work, lwork, info);
        break;
    case AA:
        dsytrf_aa(uplo, n, A, n, ipiv, work, lwork, info);
        break;
    }
}

static void run_variant(const variant_t v)
{
    const INT n = 150;
    const char* uplos[2] = {"U", "L"};
    INT info, lwork;
    f64 wq;
    uint64_t rng_state[4];

    f64* A = malloc(n * n * sizeof(f64));
    f64* E = malloc(n * sizeof(f64));
    INT* ipiv = malloc(n * sizeof(INT));
    assert_non_null(A); assert_non_null(E); assert_non_null(ipiv);

    rng_seed(rng_state, 0x5A17F0ULL);
    for (INT u = 0; u < 2; u++) {
        const INT upper = (u == 0);
        for (INT j = 0; j < n; j++) {
            for (INT i = 0; i < n; i++) {
                if (upper ? (i <= j) : (i >= j)) {
                    A[i + j * n] = rng_uniform_symmetric(rng_state);
                } else {
                    A[i + j * n] = CANARY;
                }
            }
        }

        factor(v, uplos[u], n, A, E, ipiv, &wq, -1, &info);
        assert_info_success(info);
        lwork = (INT)wq;
        f64* work = malloc((lwork + NGUARD) * sizeof(f64));
        assert_non_null(work);
        for (INT i = lwork; i < lwork + NGUARD; i++) {
            work[i] = CANARY;
        }

        factor(v, uplos[u], n, A, E, ipiv, work, lwork, &info);
        assert_info_success(info);

        INT ntri = 0, nguard = 0;
        for (INT j = 0; j < n; j++) {
            for (INT i = 0; i < n; i++) {
                if (!(upper ? (i <= j) : (i >= j)) && A[i + j * n] != CANARY) {
                    ntri++;
                }
            }
        }
        for (INT i = lwork; i < lwork + NGUARD; i++) {
            if (work[i] != CANARY) {
                nguard++;
            }
        }
        free(work);
        if (ntri != 0 || nguard != 0) {
            fail_msg("UPLO=%s: %d entries of the unreferenced triangle and %d "
                     "entries after WORK(LWORK=%d) were written",
                     uplos[u], (int)ntri, (int)nguard, (int)lwork);
        }
    }

    free(A); free(E); free(ipiv);
}

static void test_sytrf(void** state)
{
    (void)state;
    run_variant(BK);
}

static void test_sytrf_rook(void** state)
{
    (void)state;
    run_variant(ROOK);
}

static void test_sytrf_rk(void** state)
{
    (void)state;
    run_variant(RK);
}

static void test_sytrf_aa(void** state)
{
    (void)state;
    run_variant(AA);
}

int main(void)
{
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_sytrf),
        cmocka_unit_test(test_sytrf_rook),
        cmocka_unit_test(test_sytrf_rk),
        cmocka_unit_test(test_sytrf_aa),
    };
    (void)cmocka_run_group_tests_name("dsytrf_bounds", tests, NULL, NULL);
    return 0;
}
