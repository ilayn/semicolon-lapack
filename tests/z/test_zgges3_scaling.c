/**
 * @file test_zgges3_scaling.c
 * @brief ZGGES3 regression tests for ZLAQZ0 and the multishift sweep.
 *
 * test_negligible_pivot is a port of LAPACK TESTING/EIG/test_zgges3_scaling.f90.
 * test_multishift runs ZGGES3 on a random pencil with N above the IPARMQ
 * crossover NMIN = 75, where ZLAQZ0 uses aggressive early deflation (ZLAQZ2)
 * and the multishift sweep of ZLAQZ3.
 */

#include "test_harness.h"
#include "verify.h"
#include "test_rng.h"
#include <math.h>
#include <stdlib.h>

#define THRESH 30.0

static INT select_none(const c128* alpha, const c128* beta)
{
    (void)alpha;
    (void)beta;
    return 0;
}

static f64 recon_err(const INT n, const c128* X0, const c128* X,
                     const c128* Q, const c128* Z, const f64 xnorm, c128* T)
{
    for (INT j = 0; j < n; j++) {
        for (INT i = 0; i < n; i++) {
            c128 s = 0.0;
            for (INT l = 0; l < n; l++) {
                s += (X[i + l * n] / xnorm) * conj(Z[j + l * n]);
            }
            T[i + j * n] = s;
        }
    }
    f64 err = 0.0;
    for (INT j = 0; j < n; j++) {
        for (INT i = 0; i < n; i++) {
            c128 s = 0.0;
            for (INT k = 0; k < n; k++) {
                s += Q[i + k * n] * T[k + j * n];
            }
            f64 r = cabs(X0[i + j * n] / xnorm - s);
            if (isnan(r) || r > err) {
                err = r;
            }
        }
    }
    return err;
}

/* Check that a negligible nonzero pivot in ZLAQZ0 does not leave fill
   below B's diagonal. Such fill is not rescaled by ZGGES3's final
   ZLASCL('U',...) call and corrupts the decomposition of a small pencil. */
static void test_negligible_pivot(void** state)
{
    (void)state;
    /* Pad the active 3-by-3 block to enable multishift QZ with the default
       IPARMQ crossover NMIN=75, without overriding the tuning routines. */
    const INT n = 80;
    const INT lwork = 200000;
    INT info, sdim;

    c128* A = calloc(n * n, sizeof(c128));
    c128* B = calloc(n * n, sizeof(c128));
    c128* A0 = malloc(n * n * sizeof(c128));
    c128* B0 = malloc(n * n * sizeof(c128));
    c128* Q = malloc(n * n * sizeof(c128));
    c128* Z = malloc(n * n * sizeof(c128));
    c128* T = malloc(n * n * sizeof(c128));
    c128* alpha = malloc(n * sizeof(c128));
    c128* beta = malloc(n * sizeof(c128));
    /* Keep the large workspace off the stack, including recursive builds. */
    c128* work = malloc(lwork * sizeof(c128));
    f64* rwork = malloc(8 * n * sizeof(f64));
    INT* bwork = malloc(n * sizeof(INT));
    assert_non_null(A); assert_non_null(B); assert_non_null(A0);
    assert_non_null(B0); assert_non_null(Q); assert_non_null(Z);
    assert_non_null(T); assert_non_null(alpha); assert_non_null(beta);
    assert_non_null(work); assert_non_null(rwork); assert_non_null(bwork);

    const f64 scale = 1.0e-280;
    for (INT i = 0; i < n; i++) {
        A[i + i * n] = 1.0;
        B[i + i * n] = 1.0;
    }
    /* A = [0 -1 -1; 1e-20 0 -1; 0 1 -1]
       B = [1  1  1; 0 (1+2i)*1e-18 1; 0 0 1] */
    A[0 + 0 * n] = 0.0; A[0 + 1 * n] = -1.0; A[0 + 2 * n] = -1.0;
    A[1 + 0 * n] = 1.0e-20; A[1 + 1 * n] = 0.0; A[1 + 2 * n] = -1.0;
    A[2 + 1 * n] = 1.0; A[2 + 2 * n] = -1.0;
    B[0 + 1 * n] = 1.0; B[0 + 2 * n] = 1.0; B[1 + 2 * n] = 1.0;
    B[1 + 1 * n] = CMPLX(1.0e-18, 2.0e-18);
    for (INT i = 0; i < n * n; i++) {
        A[i] *= scale;
        B[i] *= scale;
        A0[i] = A[i];
        B0[i] = B[i];
    }

    zgges3("V", "V", "N", select_none, n, A, n, B, n, &sdim, alpha, beta,
           Q, n, Z, n, work, lwork, rwork, bwork, &info);
    assert_info_success(info);

    /* Normalize before reconstruction to avoid underflow in the checks. */
    f64 anorm = 0.0, bnorm = 0.0;
    for (INT i = 0; i < n * n; i++) {
        if (cabs(A0[i]) > anorm) anorm = cabs(A0[i]);
        if (cabs(B0[i]) > bnorm) bnorm = cabs(B0[i]);
    }
    f64 aerr = recon_err(n, A0, A, Q, Z, anorm, T);
    f64 berr = recon_err(n, B0, B, Q, Z, bnorm, T);
    f64 lower = 0.0;
    for (INT j = 0; j < n; j++) {
        for (INT i = j + 1; i < n; i++) {
            if (!(cabs(B[i + j * n]) <= lower)) lower = cabs(B[i + j * n]);
        }
    }
    f64 tol = 100 * n * dlamch("P");

    /* Negated comparisons also reject NaNs. */
    if (!(aerr <= tol && berr <= tol && lower == 0.0)) {
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
    c128 wq;
    f64 result[2];

    c128* A = malloc(n * n * sizeof(c128));
    c128* B = malloc(n * n * sizeof(c128));
    c128* A0 = malloc(n * n * sizeof(c128));
    c128* B0 = malloc(n * n * sizeof(c128));
    c128* Q = malloc(n * n * sizeof(c128));
    c128* Z = malloc(n * n * sizeof(c128));
    c128* alpha = malloc(n * sizeof(c128));
    c128* beta = malloc(n * sizeof(c128));
    c128* work51 = malloc(2 * n * n * sizeof(c128));
    f64* rwork = malloc(8 * n * sizeof(f64));
    INT* bwork = malloc(n * sizeof(INT));
    assert_non_null(A); assert_non_null(B); assert_non_null(A0);
    assert_non_null(B0); assert_non_null(Q); assert_non_null(Z);
    assert_non_null(alpha); assert_non_null(beta); assert_non_null(work51);
    assert_non_null(rwork); assert_non_null(bwork);

    rng_seed(rng_state, 0x9A2E5ULL);
    for (INT i = 0; i < n * n; i++) {
        f64 ar = rng_uniform_symmetric(rng_state);
        f64 ai = rng_uniform_symmetric(rng_state);
        f64 br = rng_uniform_symmetric(rng_state);
        f64 bi = rng_uniform_symmetric(rng_state);
        A[i] = CMPLX(ar, ai);
        B[i] = CMPLX(br, bi);
        A0[i] = A[i];
        B0[i] = B[i];
    }

    zgges3("V", "V", "N", select_none, n, A, n, B, n, &sdim, alpha, beta,
           Q, n, Z, n, &wq, -1, rwork, bwork, &info);
    assert_info_success(info);
    INT lwork = (INT)creal(wq);
    c128* work = malloc(lwork * sizeof(c128));
    assert_non_null(work);

    zgges3("V", "V", "N", select_none, n, A, n, B, n, &sdim, alpha, beta,
           Q, n, Z, n, work, lwork, rwork, bwork, &info);
    assert_info_success(info);

    zget51(1, n, A0, n, A, n, Q, n, Z, n, work51, rwork, &result[0]);
    zget51(1, n, B0, n, B, n, Q, n, Z, n, work51, rwork, &result[1]);
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
    (void)cmocka_run_group_tests_name("zgges3_scaling", tests, NULL, NULL);
    return 0;
}
