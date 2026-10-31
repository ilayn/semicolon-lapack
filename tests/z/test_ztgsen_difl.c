/**
 * @file test_ztgsen_difl.c
 * @brief ZTGSEN regression test for the transposed Difl solve of
 *        IJOB = 3 and 5 when N1 > N2.
 *
 * For IJOB = 3 and 5, ZTGSEN estimates DIF(2) = Difl with ZLACN2, solving
 * the transposed generalized Sylvester equation with ZTGSYL('C', ..., N2,
 * N1, A22, A11, ..., D = B22, E = B11, ...) as DTGSEN does. Passing B11 as
 * D and B22 as E makes ZTGSYL read an N1-by-N1 block starting at B(N1+1,
 * N1+1), which runs past the end of B when N1 > N2. Each case selects
 * N1 = 6 of N = 9 eigenvalues of an upper triangular pencil, places B at
 * the start of a padded allocation, and calls ZTGSEN twice, once with zero
 * padding and once with NaN padding. DIF must be positive, finite and the
 * same in both calls.
 */

#include "test_harness.h"
#include "verify.h"
#include "test_rng.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

static void run_tgsen(const INT ijob, const c128 pad, const c128* A0,
                      const c128* B0, const INT* select, const INT n,
                      const INT npad, f64* dif)
{
    INT m, info, lwork, liwork, iwq;
    f64 pl, pr;
    c128 wq;

    c128* A = malloc((size_t)n * n * sizeof(c128));
    c128* B = malloc((size_t)(n * n + npad) * sizeof(c128));
    c128* Q = malloc((size_t)n * n * sizeof(c128));
    c128* Z = malloc((size_t)n * n * sizeof(c128));
    c128* alpha = malloc((size_t)n * sizeof(c128));
    c128* beta = malloc((size_t)n * sizeof(c128));
    assert_non_null(A); assert_non_null(B); assert_non_null(Q);
    assert_non_null(Z); assert_non_null(alpha); assert_non_null(beta);

    memcpy(A, A0, (size_t)n * n * sizeof(c128));
    memcpy(B, B0, (size_t)n * n * sizeof(c128));
    for (INT i = 0; i < npad; i++) {
        B[n * n + i] = pad;
    }
    zlaset("F", n, n, CMPLX(0.0, 0.0), CMPLX(1.0, 0.0), Q, n);
    zlaset("F", n, n, CMPLX(0.0, 0.0), CMPLX(1.0, 0.0), Z, n);

    ztgsen(ijob, 1, 1, select, n, A, n, B, n, alpha, beta, Q, n, Z, n,
           &m, &pl, &pr, dif, &wq, -1, &iwq, -1, &info);
    assert_info_success(info);
    lwork = (INT)creal(wq);
    liwork = iwq;

    c128* work = malloc((size_t)lwork * sizeof(c128));
    INT* iwork = malloc((size_t)liwork * sizeof(INT));
    assert_non_null(work); assert_non_null(iwork);

    ztgsen(ijob, 1, 1, select, n, A, n, B, n, alpha, beta, Q, n, Z, n,
           &m, &pl, &pr, dif, work, lwork, iwork, liwork, &info);
    assert_info_success(info);
    assert_int_equal(m, 6);

    free(A); free(B); free(Q); free(Z); free(alpha); free(beta);
    free(work); free(iwork);
}

static void run_case(const INT ijob)
{
    const INT n = 9;
    const INT npad = 4 * n;
    f64 dif0[2], dif1[2];
    uint64_t rng_state[4];

    c128* A0 = calloc((size_t)n * n, sizeof(c128));
    c128* B0 = calloc((size_t)n * n, sizeof(c128));
    INT* select = calloc((size_t)n, sizeof(INT));
    assert_non_null(A0); assert_non_null(B0); assert_non_null(select);

    rng_seed(rng_state, 0xD1F12ULL);
    for (INT j = 0; j < n; j++) {
        for (INT i = 0; i <= j; i++) {
            f64 ar = rng_uniform_symmetric(rng_state);
            f64 ai = rng_uniform_symmetric(rng_state);
            f64 br = rng_uniform_symmetric(rng_state);
            f64 bi = rng_uniform_symmetric(rng_state);
            A0[i + j * n] = CMPLX(ar, ai);
            B0[i + j * n] = CMPLX(br, bi);
        }
        A0[j + j * n] += (f64)(j + 2);
        B0[j + j * n] += 2.0;
    }
    for (INT k = 0; k < 6; k++) {
        select[k] = 1;
    }

    run_tgsen(ijob, CMPLX(0.0, 0.0), A0, B0, select, n, npad, dif0);
    run_tgsen(ijob, CMPLX(NAN, NAN), A0, B0, select, n, npad, dif1);

    for (INT k = 0; k < 2; k++) {
        if (!(isfinite(dif0[k]) && dif0[k] > 0.0) || dif0[k] != dif1[k]) {
            fail_msg("IJOB = %d: DIF(%d) = %.17g with zero padding after B, %.17g with NaN padding",
                     (int)ijob, (int)(k + 1), (double)dif0[k], (double)dif1[k]);
        }
    }

    free(A0); free(B0); free(select);
}

static void test_ijob3(void** state)
{
    (void)state;
    run_case(3);
}

static void test_ijob5(void** state)
{
    (void)state;
    run_case(5);
}

int main(void)
{
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_ijob3),
        cmocka_unit_test(test_ijob5),
    };
    (void)cmocka_run_group_tests_name("ztgsen_difl", tests, NULL, NULL);
    return 0;
}
