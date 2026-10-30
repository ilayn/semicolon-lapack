/**
 * @file test_dlaqz3_window.c
 * @brief DLAQZ3 regression test for a 1-by-1 deflation window.
 *
 * With IHI = KWTOP the reference DLAQZ3 tries a regular deflation and then
 * continues with the general window code, which runs DLAQZ0 on the window,
 * making B(KWTOP,KWTOP) non-negative, and applies QC and ZC to A, B, Q and
 * Z. Each case calls DLAQZ3 with NW = 1 on a pencil whose last diagonal
 * entry of B is negative, with and without a negligible subdiagonal, and
 * checks BETA(KWTOP) >= 0 and the residuals of A and B against Q and Z.
 */

#include "test_harness.h"
#include "verify.h"
#include "test_rng.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define THRESH 30.0

static void run_window(const f64 sub)
{
    const INT n = 6;
    const INT lwork = 4096;
    INT ns, nd, info;
    f64 QC[1], ZC[1], result[2];
    uint64_t rng_state[4];

    f64* A = calloc(n * n, sizeof(f64));
    f64* B = calloc(n * n, sizeof(f64));
    f64* A0 = malloc(n * n * sizeof(f64));
    f64* B0 = malloc(n * n * sizeof(f64));
    f64* Q = calloc(n * n, sizeof(f64));
    f64* Z = calloc(n * n, sizeof(f64));
    f64* alphar = calloc(n, sizeof(f64));
    f64* alphai = calloc(n, sizeof(f64));
    f64* beta = calloc(n, sizeof(f64));
    f64* work = malloc(lwork * sizeof(f64));
    f64* work51 = malloc(2 * n * n * sizeof(f64));
    assert_non_null(A); assert_non_null(B); assert_non_null(A0);
    assert_non_null(B0); assert_non_null(Q); assert_non_null(Z);
    assert_non_null(alphar); assert_non_null(alphai); assert_non_null(beta);
    assert_non_null(work); assert_non_null(work51);

    rng_seed(rng_state, 0x1A2E3ULL);
    for (INT j = 0; j < n; j++) {
        for (INT i = 0; i <= j + 1 && i < n; i++) {
            A[i + j * n] = rng_uniform_symmetric(rng_state);
        }
        for (INT i = 0; i <= j; i++) {
            B[i + j * n] = rng_uniform_symmetric(rng_state);
        }
        Q[j + j * n] = 1.0;
        Z[j + j * n] = 1.0;
    }
    B[(n - 1) + (n - 1) * n] = -0.5;
    A[(n - 1) + (n - 2) * n] = sub;
    memcpy(A0, A, n * n * sizeof(f64));
    memcpy(B0, B, n * n * sizeof(f64));

    dlaqz3(1, 1, 1, n, 0, n - 1, 1, A, n, B, n, Q, n, Z, n, &ns, &nd,
           alphar, alphai, beta, QC, 1, ZC, 1, work, lwork, 0, &info);
    assert_info_success(info);
    assert_int_equal(ns + nd, 1);

    if (!(beta[n - 1] >= 0.0)) {
        fail_msg("BETA(KWTOP) = %.6e is negative", (double)beta[n - 1]);
    }
    if (!(B[(n - 1) + (n - 1) * n] >= 0.0)) {
        fail_msg("B(KWTOP,KWTOP) = %.6e is negative",
                 (double)B[(n - 1) + (n - 1) * n]);
    }

    dget51(1, n, A0, n, A, n, Q, n, Z, n, work51, &result[0]);
    dget51(1, n, B0, n, B, n, Q, n, Z, n, work51, &result[1]);
    assert_residual_ok(result[0]);
    assert_residual_ok(result[1]);

    free(A); free(B); free(A0); free(B0); free(Q); free(Z);
    free(alphar); free(alphai); free(beta); free(work); free(work51);
}

static void test_undeflated(void** state)
{
    (void)state;
    run_window(0.3);
}

static void test_deflated(void** state)
{
    (void)state;
    run_window(1.0e-30);
}

int main(void)
{
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_undeflated),
        cmocka_unit_test(test_deflated),
    };
    (void)cmocka_run_group_tests_name("dlaqz3_window", tests, NULL, NULL);
    return 0;
}
