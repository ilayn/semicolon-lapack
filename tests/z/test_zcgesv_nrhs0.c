/**
 * @file test_zcgesv_nrhs0.c
 * @brief ZCGESV and ZCPOSV regression test for NRHS = 0.
 *
 * The reference ZCGESV and ZCPOSV return early only for N = 0. With
 * NRHS = 0 and N > 0 they still factor A: in single precision when A fits,
 * which sets IPIV (ZCGESV) and ITER = 0, and otherwise in double precision
 * with ITER = -2 (A does not fit in single precision) or ITER = -3 (the
 * single precision factorization failed), which overwrites A with its
 * factors and reports a singular or non positive definite A in INFO.
 */

#include "test_harness.h"
#include "test_rng.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define N 5

static void fill_random(c128* A, uint64_t seed)
{
    uint64_t rng_state[4];
    rng_seed(rng_state, seed);
    for (INT j = 0; j < N; j++) {
        for (INT i = 0; i < N; i++) {
            f64 re = rng_uniform_symmetric(rng_state);
            f64 im = rng_uniform_symmetric(rng_state);
            A[i + j * N] = CMPLX(re, im);
        }
    }
}

static void fill_hpd(c128* A, uint64_t seed)
{
    uint64_t rng_state[4];
    rng_seed(rng_state, seed);
    for (INT j = 0; j < N; j++) {
        for (INT i = 0; i < j; i++) {
            f64 re = rng_uniform_symmetric(rng_state);
            f64 im = rng_uniform_symmetric(rng_state);
            A[i + j * N] = CMPLX(re, im);
            A[j + i * N] = CMPLX(re, -im);
        }
        A[j + j * N] = CMPLX((f64)(2 * N), 0.0);
    }
}

static void run_cgesv(c128* A, INT* ipiv, INT* iter, INT* info)
{
    c128 B[N], X[N], work[N];
    c64 swork[N * (N + 1)];
    f64 rwork[N];
    for (INT i = 0; i < N; i++) {
        B[i] = 0.0;
        X[i] = 0.0;
        ipiv[i] = -1;
    }
    *iter = 99;
    *info = 99;
    zcgesv(N, 0, A, N, ipiv, B, N, X, N, work, swork, rwork, iter, info);
}

static void run_cposv(const char* uplo, c128* A, INT* iter, INT* info)
{
    c128 B[N], X[N], work[N];
    c64 swork[N * (N + 1)];
    f64 rwork[N];
    for (INT i = 0; i < N; i++) {
        B[i] = 0.0;
        X[i] = 0.0;
    }
    *iter = 99;
    *info = 99;
    zcposv(uplo, N, 0, A, N, B, N, X, N, work, swork, rwork, iter, info);
}

static void test_zcgesv_single(void** state)
{
    (void)state;
    c128 A[N * N], A0[N * N];
    INT ipiv[N], iter, info;

    fill_random(A, 0x5E5A1ULL);
    memcpy(A0, A, sizeof(A));
    run_cgesv(A, ipiv, &iter, &info);
    assert_info_success(info);
    assert_int_equal(iter, 0);
    for (INT i = 0; i < N; i++) {
        if (ipiv[i] < i || ipiv[i] >= N) {
            fail_msg("IPIV(%d) = %d was not set by the factorization",
                     (int)i, (int)ipiv[i]);
        }
    }
    assert_memory_equal(A, A0, sizeof(A));
}

static void test_zcgesv_singular(void** state)
{
    (void)state;
    c128 A[N * N];
    INT ipiv[N], iter, info;

    fill_random(A, 0x5E5A2ULL);
    for (INT i = 0; i < N; i++) {
        A[i + (N - 1) * N] = 0.0;
    }
    run_cgesv(A, ipiv, &iter, &info);
    assert_int_equal(iter, -3);
    assert_int_equal(info, N);
}

static void test_zcgesv_overflow(void** state)
{
    (void)state;
    c128 A[N * N], LU[N * N];
    INT ipiv[N], ipiv0[N], iter, info, info0;

    fill_random(A, 0x5E5A3ULL);
    A[0] = CMPLX(1.0e300, 0.0);
    memcpy(LU, A, sizeof(A));
    zgetrf(N, N, LU, N, ipiv0, &info0);
    assert_info_success(info0);

    run_cgesv(A, ipiv, &iter, &info);
    assert_int_equal(iter, -2);
    assert_info_success(info);
    assert_memory_equal(ipiv, ipiv0, sizeof(ipiv));
    assert_memory_equal(A, LU, sizeof(A));
}

static void test_zcposv_single(void** state)
{
    (void)state;
    const char* uplos[2] = {"U", "L"};
    for (INT u = 0; u < 2; u++) {
        c128 A[N * N], A0[N * N];
        INT iter, info;
        fill_hpd(A, 0x5E5A4ULL);
        memcpy(A0, A, sizeof(A));
        run_cposv(uplos[u], A, &iter, &info);
        assert_info_success(info);
        assert_int_equal(iter, 0);
        assert_memory_equal(A, A0, sizeof(A));
    }
}

static void test_zcposv_indefinite(void** state)
{
    (void)state;
    const char* uplos[2] = {"U", "L"};
    for (INT u = 0; u < 2; u++) {
        c128 A[N * N];
        INT iter, info;
        for (INT k = 0; k < N * N; k++) {
            A[k] = 0.0;
        }
        for (INT i = 0; i < N; i++) {
            A[i + i * N] = 1.0;
        }
        A[2 + 2 * N] = -1.0;
        run_cposv(uplos[u], A, &iter, &info);
        assert_int_equal(iter, -3);
        assert_int_equal(info, 3);
    }
}

static void test_zcposv_overflow(void** state)
{
    (void)state;
    const char* uplos[2] = {"U", "L"};
    for (INT u = 0; u < 2; u++) {
        c128 A[N * N], C[N * N];
        INT iter, info, info0;
        fill_hpd(A, 0x5E5A5ULL);
        for (INT k = 0; k < N * N; k++) {
            A[k] *= 1.0e300;
        }
        memcpy(C, A, sizeof(A));
        zpotrf(uplos[u], N, C, N, &info0);
        assert_info_success(info0);

        run_cposv(uplos[u], A, &iter, &info);
        assert_int_equal(iter, -2);
        assert_info_success(info);
        assert_memory_equal(A, C, sizeof(A));
    }
}

int main(void)
{
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_zcgesv_single),
        cmocka_unit_test(test_zcgesv_singular),
        cmocka_unit_test(test_zcgesv_overflow),
        cmocka_unit_test(test_zcposv_single),
        cmocka_unit_test(test_zcposv_indefinite),
        cmocka_unit_test(test_zcposv_overflow),
    };
    (void)cmocka_run_group_tests_name("zcgesv_nrhs0", tests, NULL, NULL);
    return 0;
}
