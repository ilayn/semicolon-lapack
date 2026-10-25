/**
 * @file test_dgges3_scaling.c
 * @brief DGGES3 regression tests for DLAQZ0 and the multishift sweep.
 *
 * test_negligible_pivot is a port of LAPACK TESTING/EIG/test_dgges3_scaling.f90.
 * test_multishift runs DGGES3 on a random pencil with N above the IPARMQ
 * crossover NMIN = 75, where DLAQZ0 uses the multishift sweep of DLAQZ4.
 */

#include "test_harness.h"
#include "verify.h"
#include "test_rng.h"
#include <math.h>
#include <stdlib.h>

#define THRESH 30.0

static INT select_none(const f64* alphar, const f64* alphai, const f64* beta)
{
    (void)alphar;
    (void)alphai;
    (void)beta;
    return 0;
}

static f64 recon_err(const INT n, const f64* X0, const f64* X,
                     const f64* Q, const f64* Z, const f64 xnorm, f64* T)
{
    for (INT j = 0; j < n; j++) {
        for (INT i = 0; i < n; i++) {
            f64 s = 0.0;
            for (INT l = 0; l < n; l++) {
                s += (X[i + l * n] / xnorm) * Z[j + l * n];
            }
            T[i + j * n] = s;
        }
    }
    f64 err = 0.0;
    for (INT j = 0; j < n; j++) {
        for (INT i = 0; i < n; i++) {
            f64 s = 0.0;
            for (INT k = 0; k < n; k++) {
                s += Q[i + k * n] * T[k + j * n];
            }
            f64 r = fabs(X0[i + j * n] / xnorm - s);
            if (isnan(r) || r > err) {
                err = r;
            }
        }
    }
    return err;
}

/* Check that a negligible nonzero pivot in DLAQZ0 does not leave fill
   below B's diagonal before DGGES3 restores the original scale. */
static void test_negligible_pivot(void** state)
{
    (void)state;
    const INT n = 80;
    const INT lwork = 200000;
    INT info, sdim;

    f64* A = calloc(n * n, sizeof(f64));
    f64* B = calloc(n * n, sizeof(f64));
    f64* A0 = malloc(n * n * sizeof(f64));
    f64* B0 = malloc(n * n * sizeof(f64));
    f64* Q = malloc(n * n * sizeof(f64));
    f64* Z = malloc(n * n * sizeof(f64));
    f64* T = malloc(n * n * sizeof(f64));
    f64* alphar = malloc(n * sizeof(f64));
    f64* alphai = malloc(n * sizeof(f64));
    f64* beta = malloc(n * sizeof(f64));
    f64* work = malloc(lwork * sizeof(f64));
    INT* bwork = malloc(n * sizeof(INT));
    assert_non_null(A); assert_non_null(B); assert_non_null(A0);
    assert_non_null(B0); assert_non_null(Q); assert_non_null(Z);
    assert_non_null(T); assert_non_null(alphar); assert_non_null(alphai);
    assert_non_null(beta); assert_non_null(work); assert_non_null(bwork);

    const f64 scale = 1.0e-280;
    for (INT i = 0; i < n; i++) {
        A[i + i * n] = 1.0;
        B[i + i * n] = 1.0;
    }
    A[0 + 0 * n] = 0.0; A[0 + 1 * n] = -1.0; A[0 + 2 * n] = -1.0;
    A[1 + 0 * n] = 1.0e-20; A[1 + 1 * n] = 0.0; A[1 + 2 * n] = -1.0;
    A[2 + 1 * n] = 1.0; A[2 + 2 * n] = -1.0;
    B[0 + 1 * n] = 1.0; B[0 + 2 * n] = 1.0; B[1 + 2 * n] = 1.0;
    B[1 + 1 * n] = 1.0e-18;
    for (INT i = 0; i < n * n; i++) {
        A[i] *= scale;
        B[i] *= scale;
        A0[i] = A[i];
        B0[i] = B[i];
    }

    dgges3("V", "V", "N", select_none, n, A, n, B, n, &sdim, alphar, alphai,
           beta, Q, n, Z, n, work, lwork, bwork, &info);
    assert_info_success(info);

    f64 anorm = 0.0, bnorm = 0.0;
    for (INT i = 0; i < n * n; i++) {
        if (fabs(A0[i]) > anorm) anorm = fabs(A0[i]);
        if (fabs(B0[i]) > bnorm) bnorm = fabs(B0[i]);
    }
    f64 aerr = recon_err(n, A0, A, Q, Z, anorm, T);
    f64 berr = recon_err(n, B0, B, Q, Z, bnorm, T);
    f64 lower = 0.0;
    for (INT j = 0; j < n; j++) {
        for (INT i = j + 1; i < n; i++) {
            if (!(fabs(B[i + j * n]) <= lower)) lower = fabs(B[i + j * n]);
        }
    }
    f64 tol = 100 * n * dlamch("P");

    if (!(aerr <= tol && berr <= tol && lower == 0.0)) {
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
    f64 wq, result[2];

    f64* A = malloc(n * n * sizeof(f64));
    f64* B = malloc(n * n * sizeof(f64));
    f64* A0 = malloc(n * n * sizeof(f64));
    f64* B0 = malloc(n * n * sizeof(f64));
    f64* Q = malloc(n * n * sizeof(f64));
    f64* Z = malloc(n * n * sizeof(f64));
    f64* alphar = malloc(n * sizeof(f64));
    f64* alphai = malloc(n * sizeof(f64));
    f64* beta = malloc(n * sizeof(f64));
    f64* work51 = malloc(2 * n * n * sizeof(f64));
    INT* bwork = malloc(n * sizeof(INT));
    assert_non_null(A); assert_non_null(B); assert_non_null(A0);
    assert_non_null(B0); assert_non_null(Q); assert_non_null(Z);
    assert_non_null(alphar); assert_non_null(alphai); assert_non_null(beta);
    assert_non_null(work51); assert_non_null(bwork);

    rng_seed(rng_state, 0x9A2E5ULL);
    for (INT i = 0; i < n * n; i++) {
        A[i] = rng_uniform_symmetric(rng_state);
        B[i] = rng_uniform_symmetric(rng_state);
        A0[i] = A[i];
        B0[i] = B[i];
    }

    dgges3("V", "V", "N", select_none, n, A, n, B, n, &sdim, alphar, alphai,
           beta, Q, n, Z, n, &wq, -1, bwork, &info);
    assert_info_success(info);
    INT lwork = (INT)wq;
    f64* work = malloc(lwork * sizeof(f64));
    assert_non_null(work);

    dgges3("V", "V", "N", select_none, n, A, n, B, n, &sdim, alphar, alphai,
           beta, Q, n, Z, n, work, lwork, bwork, &info);
    assert_info_success(info);

    dget51(1, n, A0, n, A, n, Q, n, Z, n, work51, &result[0]);
    dget51(1, n, B0, n, B, n, Q, n, Z, n, work51, &result[1]);
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
