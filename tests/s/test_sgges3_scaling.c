/**
 * @file test_sgges3_scaling.c
 * @brief SGGES3 regression tests for SLAQZ0 and the multishift sweep.
 *
 * test_negligible_pivot is a port of LAPACK TESTING/EIG/test_dgges3_scaling.f90.
 * test_multishift runs SGGES3 on a random pencil with N above the IPARMQ
 * crossover NMIN = 75, where SLAQZ0 uses the multishift sweep of SLAQZ4.
 */

#include "test_harness.h"
#include "verify.h"
#include "test_rng.h"
#include <math.h>
#include <stdlib.h>

#define THRESH 30.0f

static INT select_none(const f32* alphar, const f32* alphai, const f32* beta)
{
    (void)alphar;
    (void)alphai;
    (void)beta;
    return 0;
}

static f32 recon_err(const INT n, const f32* X0, const f32* X,
                     const f32* Q, const f32* Z, const f32 xnorm, f32* T)
{
    for (INT j = 0; j < n; j++) {
        for (INT i = 0; i < n; i++) {
            f32 s = 0.0f;
            for (INT l = 0; l < n; l++) {
                s += (X[i + l * n] / xnorm) * Z[j + l * n];
            }
            T[i + j * n] = s;
        }
    }
    f32 err = 0.0f;
    for (INT j = 0; j < n; j++) {
        for (INT i = 0; i < n; i++) {
            f32 s = 0.0f;
            for (INT k = 0; k < n; k++) {
                s += Q[i + k * n] * T[k + j * n];
            }
            f32 r = fabsf(X0[i + j * n] / xnorm - s);
            if (isnan(r) || r > err) {
                err = r;
            }
        }
    }
    return err;
}

/* Check that a negligible nonzero pivot in SLAQZ0 does not leave fill
   below B's diagonal before SGGES3 restores the original scale. */
static void test_negligible_pivot(void** state)
{
    (void)state;
    const INT n = 80;
    const INT lwork = 200000;
    INT info, sdim;

    f32* A = calloc(n * n, sizeof(f32));
    f32* B = calloc(n * n, sizeof(f32));
    f32* A0 = malloc(n * n * sizeof(f32));
    f32* B0 = malloc(n * n * sizeof(f32));
    f32* Q = malloc(n * n * sizeof(f32));
    f32* Z = malloc(n * n * sizeof(f32));
    f32* T = malloc(n * n * sizeof(f32));
    f32* alphar = malloc(n * sizeof(f32));
    f32* alphai = malloc(n * sizeof(f32));
    f32* beta = malloc(n * sizeof(f32));
    f32* work = malloc(lwork * sizeof(f32));
    INT* bwork = malloc(n * sizeof(INT));
    assert_non_null(A); assert_non_null(B); assert_non_null(A0);
    assert_non_null(B0); assert_non_null(Q); assert_non_null(Z);
    assert_non_null(T); assert_non_null(alphar); assert_non_null(alphai);
    assert_non_null(beta); assert_non_null(work); assert_non_null(bwork);

    const f32 scale = 1.0e-30f;
    for (INT i = 0; i < n; i++) {
        A[i + i * n] = 1.0f;
        B[i + i * n] = 1.0f;
    }
    A[0 + 0 * n] = 0.0f; A[0 + 1 * n] = -1.0f; A[0 + 2 * n] = -1.0f;
    A[1 + 0 * n] = 1.0e-10f; A[1 + 1 * n] = 0.0f; A[1 + 2 * n] = -1.0f;
    A[2 + 1 * n] = 1.0f; A[2 + 2 * n] = -1.0f;
    B[0 + 1 * n] = 1.0f; B[0 + 2 * n] = 1.0f; B[1 + 2 * n] = 1.0f;
    B[1 + 1 * n] = 1.0e-8f;
    for (INT i = 0; i < n * n; i++) {
        A[i] *= scale;
        B[i] *= scale;
        A0[i] = A[i];
        B0[i] = B[i];
    }

    sgges3("V", "V", "N", select_none, n, A, n, B, n, &sdim, alphar, alphai,
           beta, Q, n, Z, n, work, lwork, bwork, &info);
    assert_info_success(info);

    f32 anorm = 0.0f, bnorm = 0.0f;
    for (INT i = 0; i < n * n; i++) {
        if (fabsf(A0[i]) > anorm) anorm = fabsf(A0[i]);
        if (fabsf(B0[i]) > bnorm) bnorm = fabsf(B0[i]);
    }
    f32 aerr = recon_err(n, A0, A, Q, Z, anorm, T);
    f32 berr = recon_err(n, B0, B, Q, Z, bnorm, T);
    f32 lower = 0.0f;
    for (INT j = 0; j < n; j++) {
        for (INT i = j + 1; i < n; i++) {
            if (!(fabsf(B[i + j * n]) <= lower)) lower = fabsf(B[i + j * n]);
        }
    }
    f32 tol = 100 * n * slamch("P");

    if (!(aerr <= tol && berr <= tol && lower == 0.0f)) {
        fail_msg("A residual %.3e, B residual %.3e, B strict lower max %.3e (tol %.3e)",
                 (double)aerr, (double)berr, (double)lower, (double)tol);
    }

    free(A); free(B); free(A0); free(B0); free(Q); free(Z); free(T);
    free(alphar); free(alphai); free(beta); free(work); free(bwork);
}

static void test_multishift(void** state)
{
    (void)state;
    const INT n = 100;
    INT info, sdim;
    uint64_t rng_state[4];
    f32 wq, result[2];

    f32* A = malloc(n * n * sizeof(f32));
    f32* B = malloc(n * n * sizeof(f32));
    f32* A0 = malloc(n * n * sizeof(f32));
    f32* B0 = malloc(n * n * sizeof(f32));
    f32* Q = malloc(n * n * sizeof(f32));
    f32* Z = malloc(n * n * sizeof(f32));
    f32* alphar = malloc(n * sizeof(f32));
    f32* alphai = malloc(n * sizeof(f32));
    f32* beta = malloc(n * sizeof(f32));
    f32* work51 = malloc(2 * n * n * sizeof(f32));
    INT* bwork = malloc(n * sizeof(INT));
    assert_non_null(A); assert_non_null(B); assert_non_null(A0);
    assert_non_null(B0); assert_non_null(Q); assert_non_null(Z);
    assert_non_null(alphar); assert_non_null(alphai); assert_non_null(beta);
    assert_non_null(work51); assert_non_null(bwork);

    rng_seed(rng_state, 0x9A2E5ULL);
    for (INT i = 0; i < n * n; i++) {
        A[i] = rng_uniform_symmetric_f32(rng_state);
        B[i] = rng_uniform_symmetric_f32(rng_state);
        A0[i] = A[i];
        B0[i] = B[i];
    }

    sgges3("V", "V", "N", select_none, n, A, n, B, n, &sdim, alphar, alphai,
           beta, Q, n, Z, n, &wq, -1, bwork, &info);
    assert_info_success(info);
    INT lwork = (INT)wq;
    f32* work = malloc(lwork * sizeof(f32));
    assert_non_null(work);

    sgges3("V", "V", "N", select_none, n, A, n, B, n, &sdim, alphar, alphai,
           beta, Q, n, Z, n, work, lwork, bwork, &info);
    assert_info_success(info);

    sget51(1, n, A0, n, A, n, Q, n, Z, n, work51, &result[0]);
    sget51(1, n, B0, n, B, n, Q, n, Z, n, work51, &result[1]);
    assert_residual_ok(result[0]);
    assert_residual_ok(result[1]);

    free(A); free(B); free(A0); free(B0); free(Q); free(Z);
    free(alphar); free(alphai); free(beta); free(work51); free(bwork);
    free(work);
}

int main(void)
{
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_negligible_pivot),
        cmocka_unit_test(test_multishift),
    };
    (void)cmocka_run_group_tests_name("dgges3_scaling", tests, NULL, NULL);
    return 0;
}
