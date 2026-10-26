/**
 * @file test_cgges3_scaling.c
 * @brief CGGES3 regression tests for CLAQZ0 and the multishift sweep.
 *
 * test_negligible_pivot is a port of LAPACK TESTING/EIG/test_zgges3_scaling.f90.
 * test_multishift runs CGGES3 on a random pencil with N above the IPARMQ
 * crossover NMIN = 75, where CLAQZ0 uses aggressive early deflation (CLAQZ2)
 * and the multishift sweep of CLAQZ3.
 */

#include "test_harness.h"
#include "verify.h"
#include "test_rng.h"
#include <math.h>
#include <stdlib.h>

#define THRESH 30.0f

static INT select_none(const c64* alpha, const c64* beta)
{
    (void)alpha;
    (void)beta;
    return 0;
}

static f32 recon_err(const INT n, const c64* X0, const c64* X,
                     const c64* Q, const c64* Z, const f32 xnorm, c64* T)
{
    for (INT j = 0; j < n; j++) {
        for (INT i = 0; i < n; i++) {
            c64 s = 0.0f;
            for (INT l = 0; l < n; l++) {
                s += (X[i + l * n] / xnorm) * conjf(Z[j + l * n]);
            }
            T[i + j * n] = s;
        }
    }
    f32 err = 0.0f;
    for (INT j = 0; j < n; j++) {
        for (INT i = 0; i < n; i++) {
            c64 s = 0.0f;
            for (INT k = 0; k < n; k++) {
                s += Q[i + k * n] * T[k + j * n];
            }
            f32 r = cabsf(X0[i + j * n] / xnorm - s);
            if (isnan(r) || r > err) {
                err = r;
            }
        }
    }
    return err;
}

/* Check that a negligible nonzero pivot in CLAQZ0 does not leave fill
   below B's diagonal. Such fill is not rescaled by CGGES3's final
   CLASCL('U',...) call and corrupts the decomposition of a small pencil. */
static void test_negligible_pivot(void** state)
{
    (void)state;
    /* Pad the active 3-by-3 block to enable multishift QZ with the default
       IPARMQ crossover NMIN=75, without overriding the tuning routines. */
    const INT n = 80;
    const INT lwork = 200000;
    INT info, sdim;

    c64* A = calloc(n * n, sizeof(c64));
    c64* B = calloc(n * n, sizeof(c64));
    c64* A0 = malloc(n * n * sizeof(c64));
    c64* B0 = malloc(n * n * sizeof(c64));
    c64* Q = malloc(n * n * sizeof(c64));
    c64* Z = malloc(n * n * sizeof(c64));
    c64* T = malloc(n * n * sizeof(c64));
    c64* alpha = malloc(n * sizeof(c64));
    c64* beta = malloc(n * sizeof(c64));
    /* Keep the large workspace off the stack, including recursive builds. */
    c64* work = malloc(lwork * sizeof(c64));
    f32* rwork = malloc(8 * n * sizeof(f32));
    INT* bwork = malloc(n * sizeof(INT));
    assert_non_null(A); assert_non_null(B); assert_non_null(A0);
    assert_non_null(B0); assert_non_null(Q); assert_non_null(Z);
    assert_non_null(T); assert_non_null(alpha); assert_non_null(beta);
    assert_non_null(work); assert_non_null(rwork); assert_non_null(bwork);

    const f32 scale = 1.0e-30f;
    for (INT i = 0; i < n; i++) {
        A[i + i * n] = 1.0f;
        B[i + i * n] = 1.0f;
    }
    /* A = [0 -1 -1; 1e-10 0 -1; 0 1 -1]
       B = [1  1  1; 0 (1+2i)*1e-8 1; 0 0 1] */
    A[0 + 0 * n] = 0.0f; A[0 + 1 * n] = -1.0f; A[0 + 2 * n] = -1.0f;
    A[1 + 0 * n] = 1.0e-10f; A[1 + 1 * n] = 0.0f; A[1 + 2 * n] = -1.0f;
    A[2 + 1 * n] = 1.0f; A[2 + 2 * n] = -1.0f;
    B[0 + 1 * n] = 1.0f; B[0 + 2 * n] = 1.0f; B[1 + 2 * n] = 1.0f;
    B[1 + 1 * n] = CMPLXF(1.0e-8f, 2.0e-8f);
    for (INT i = 0; i < n * n; i++) {
        A[i] *= scale;
        B[i] *= scale;
        A0[i] = A[i];
        B0[i] = B[i];
    }

    cgges3("V", "V", "N", select_none, n, A, n, B, n, &sdim, alpha, beta,
           Q, n, Z, n, work, lwork, rwork, bwork, &info);
    assert_info_success(info);

    /* Normalize before reconstruction to avoid underflow in the checks. */
    f32 anorm = 0.0f, bnorm = 0.0f;
    for (INT i = 0; i < n * n; i++) {
        if (cabsf(A0[i]) > anorm) anorm = cabsf(A0[i]);
        if (cabsf(B0[i]) > bnorm) bnorm = cabsf(B0[i]);
    }
    f32 aerr = recon_err(n, A0, A, Q, Z, anorm, T);
    f32 berr = recon_err(n, B0, B, Q, Z, bnorm, T);
    f32 lower = 0.0f;
    for (INT j = 0; j < n; j++) {
        for (INT i = j + 1; i < n; i++) {
            if (!(cabsf(B[i + j * n]) <= lower)) lower = cabsf(B[i + j * n]);
        }
    }
    f32 tol = 100 * n * slamch("P");

    /* Negated comparisons also reject NaNs. */
    if (!(aerr <= tol && berr <= tol && lower == 0.0f)) {
        fail_msg("A residual %.3e, B residual %.3e, B strict lower max %.3e (tol %.3e)",
                 (double)aerr, (double)berr, (double)lower, (double)tol);
    }

    free(A); free(B); free(A0); free(B0); free(Q); free(Z); free(T);
    free(alpha); free(beta); free(work); free(rwork); free(bwork);
}

static void test_multishift(void** state)
{
    (void)state;
    const INT n = 100;
    INT info, sdim;
    uint64_t rng_state[4];
    c64 wq;
    f32 result[2];

    c64* A = malloc(n * n * sizeof(c64));
    c64* B = malloc(n * n * sizeof(c64));
    c64* A0 = malloc(n * n * sizeof(c64));
    c64* B0 = malloc(n * n * sizeof(c64));
    c64* Q = malloc(n * n * sizeof(c64));
    c64* Z = malloc(n * n * sizeof(c64));
    c64* alpha = malloc(n * sizeof(c64));
    c64* beta = malloc(n * sizeof(c64));
    c64* work51 = malloc(2 * n * n * sizeof(c64));
    f32* rwork = malloc(8 * n * sizeof(f32));
    INT* bwork = malloc(n * sizeof(INT));
    assert_non_null(A); assert_non_null(B); assert_non_null(A0);
    assert_non_null(B0); assert_non_null(Q); assert_non_null(Z);
    assert_non_null(alpha); assert_non_null(beta); assert_non_null(work51);
    assert_non_null(rwork); assert_non_null(bwork);

    rng_seed(rng_state, 0x9A2E5ULL);
    for (INT i = 0; i < n * n; i++) {
        f32 ar = rng_uniform_symmetric_f32(rng_state);
        f32 ai = rng_uniform_symmetric_f32(rng_state);
        f32 br = rng_uniform_symmetric_f32(rng_state);
        f32 bi = rng_uniform_symmetric_f32(rng_state);
        A[i] = CMPLXF(ar, ai);
        B[i] = CMPLXF(br, bi);
        A0[i] = A[i];
        B0[i] = B[i];
    }

    cgges3("V", "V", "N", select_none, n, A, n, B, n, &sdim, alpha, beta,
           Q, n, Z, n, &wq, -1, rwork, bwork, &info);
    assert_info_success(info);
    INT lwork = (INT)crealf(wq);
    c64* work = malloc(lwork * sizeof(c64));
    assert_non_null(work);

    cgges3("V", "V", "N", select_none, n, A, n, B, n, &sdim, alpha, beta,
           Q, n, Z, n, work, lwork, rwork, bwork, &info);
    assert_info_success(info);

    cget51(1, n, A0, n, A, n, Q, n, Z, n, work51, rwork, &result[0]);
    cget51(1, n, B0, n, B, n, Q, n, Z, n, work51, rwork, &result[1]);
    assert_residual_ok(result[0]);
    assert_residual_ok(result[1]);

    free(A); free(B); free(A0); free(B0); free(Q); free(Z);
    free(alpha); free(beta); free(work51); free(rwork); free(bwork);
    free(work);
}

int main(void)
{
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_negligible_pivot),
        cmocka_unit_test(test_multishift),
    };
    (void)cmocka_run_group_tests_name("cgges3_scaling", tests, NULL, NULL);
    return 0;
}
