/**
 * @file test_csytrf_bounds.c
 * @brief CSYTRF and CHETRF family regression test for writes outside A's
 *        triangle and WORK.
 *
 * CSYTRF, CSYTRF_ROOK, CSYTRF_RK, CSYTRF_AA and their CHETRF counterparts
 * leave the triangle of A not selected by UPLO unreferenced and need only
 * the queried LWORK entries of WORK. Each case factors a random symmetric or
 * Hermitian matrix large enough for the blocked code with exactly the
 * queried LWORK, and checks that the other triangle and the elements after
 * WORK are left untouched.
 */

#include "test_harness.h"
#include "verify.h"
#include "test_rng.h"
#include <stdlib.h>

#define CANARY CMPLXF(-12345.0f, 0.0f)
#define NGUARD 256

typedef enum { SY, SY_ROOK, SY_RK, SY_AA, HE, HE_ROOK, HE_RK, HE_AA } variant_t;

static void factor(const variant_t v, const char* uplo, const INT n, c64* A,
                   c64* E, INT* ipiv, c64* work, const INT lwork, INT* info)
{
    switch (v) {
    case SY:
        csytrf(uplo, n, A, n, ipiv, work, lwork, info);
        break;
    case SY_ROOK:
        csytrf_rook(uplo, n, A, n, ipiv, work, lwork, info);
        break;
    case SY_RK:
        csytrf_rk(uplo, n, A, n, E, ipiv, work, lwork, info);
        break;
    case SY_AA:
        csytrf_aa(uplo, n, A, n, ipiv, work, lwork, info);
        break;
    case HE:
        chetrf(uplo, n, A, n, ipiv, work, lwork, info);
        break;
    case HE_ROOK:
        chetrf_rook(uplo, n, A, n, ipiv, work, lwork, info);
        break;
    case HE_RK:
        chetrf_rk(uplo, n, A, n, E, ipiv, work, lwork, info);
        break;
    case HE_AA:
        chetrf_aa(uplo, n, A, n, ipiv, work, lwork, info);
        break;
    }
}

static void run_variant(const variant_t v)
{
    const INT n = 150;
    const INT herm = (v == HE || v == HE_ROOK || v == HE_RK || v == HE_AA);
    const char* uplos[2] = {"U", "L"};
    INT info, lwork;
    c64 wq;
    uint64_t rng_state[4];

    c64* A = malloc(n * n * sizeof(c64));
    c64* E = malloc(n * sizeof(c64));
    INT* ipiv = malloc(n * sizeof(INT));
    assert_non_null(A); assert_non_null(E); assert_non_null(ipiv);

    rng_seed(rng_state, 0x5A17F0ULL);
    for (INT u = 0; u < 2; u++) {
        const INT upper = (u == 0);
        for (INT j = 0; j < n; j++) {
            for (INT i = 0; i < n; i++) {
                if (upper ? (i <= j) : (i >= j)) {
                    f32 re = rng_uniform_symmetric_f32(rng_state);
                    f32 im = rng_uniform_symmetric_f32(rng_state);
                    A[i + j * n] = CMPLXF(re, (herm && i == j) ? 0.0f : im);
                } else {
                    A[i + j * n] = CANARY;
                }
            }
        }

        factor(v, uplos[u], n, A, E, ipiv, &wq, -1, &info);
        assert_info_success(info);
        lwork = (INT)crealf(wq);
        c64* work = malloc((lwork + NGUARD) * sizeof(c64));
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
    run_variant(SY);
}

static void test_sytrf_rook(void** state)
{
    (void)state;
    run_variant(SY_ROOK);
}

static void test_sytrf_rk(void** state)
{
    (void)state;
    run_variant(SY_RK);
}

static void test_sytrf_aa(void** state)
{
    (void)state;
    run_variant(SY_AA);
}

static void test_hetrf(void** state)
{
    (void)state;
    run_variant(HE);
}

static void test_hetrf_rook(void** state)
{
    (void)state;
    run_variant(HE_ROOK);
}

static void test_hetrf_rk(void** state)
{
    (void)state;
    run_variant(HE_RK);
}

static void test_hetrf_aa(void** state)
{
    (void)state;
    run_variant(HE_AA);
}

int main(void)
{
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_sytrf),
        cmocka_unit_test(test_sytrf_rook),
        cmocka_unit_test(test_sytrf_rk),
        cmocka_unit_test(test_sytrf_aa),
        cmocka_unit_test(test_hetrf),
        cmocka_unit_test(test_hetrf_rook),
        cmocka_unit_test(test_hetrf_rk),
        cmocka_unit_test(test_hetrf_aa),
    };
    (void)cmocka_run_group_tests_name("csytrf_bounds", tests, NULL, NULL);
    return 0;
}
