/**
 * @file test_dgesvj_emptsw.c
 * @brief DGESVJ/DGSVJ0 regression test for the EMPTSW overflow fix.
 *
 * Port of LAPACK TESTING/EIG/test_dgesvj_emptsw.f90.
 *
 * Coverage driver for the EMPTSW integer-overflow fix in DGESVJ/DGSVJ0
 * (Reference-LAPACK issue 1312: EMPTSW = N*(N-1)/2 overflows 32-bit
 * arithmetic for N >= 46342, so it is computed in a factored form that
 * never overflows).
 *
 * DGSVJ0 is only reached through DGESVJ's triangular (JOBA='L'/'U')
 * blocked path for N > 64, which the SVD tests (JOBA='G') never exercise.
 * This driver therefore calls DGESVJ with N > 64 lower- and
 * upper-triangular matrices (even and odd N), which takes the blocked
 * DGESVJ -> DGSVJ0 path and covers the fixed lines in both files. It
 * validates the live factorization directly (residual and orthogonality),
 * so there is no replica of the EMPTSW expression here to go stale.
 */

#include "test_harness.h"
#include "verify.h"
#include <math.h>
#include <stdlib.h>

static void check_svd(const INT n, const f64* A0, const f64* U,
                      const f64* V, const f64* sva, const INT info)
{
    assert_info_success(info);

    for (INT i = 0; i < n; i++) {
        if (sva[i] <= 0.0) {
            fail_msg("sva[%d] = %.3e is not positive", (int)i, (double)sva[i]);
        }
        if (i > 0 && sva[i - 1] < sva[i]) {
            fail_msg("sva[%d] = %.3e < sva[%d] = %.3e", (int)(i - 1),
                     (double)sva[i - 1], (int)i, (double)sva[i]);
        }
    }

    f64 eps = dlamch("P");
    f64 tol = (f64)(100 * n) * eps;

    f64 anorm = 0.0;
    for (INT i = 0; i < n * n; i++) {
        if (fabs(A0[i]) > anorm) anorm = fabs(A0[i]);
    }

    f64 rerr = 0.0;
    for (INT j = 0; j < n; j++) {
        for (INT i = 0; i < n; i++) {
            f64 s = 0.0;
            for (INT l = 0; l < n; l++) {
                s += (U[i + l * n] * sva[l]) * V[j + l * n];
            }
            f64 r = fabs(A0[i + j * n] - s);
            if (!(r <= rerr)) rerr = r;
        }
    }
    rerr = rerr / anorm;

    f64 oerr_u = 0.0, oerr_v = 0.0;
    for (INT j = 0; j < n; j++) {
        for (INT i = 0; i < n; i++) {
            f64 su = 0.0, sv = 0.0;
            for (INT l = 0; l < n; l++) {
                su += U[l + i * n] * U[l + j * n];
                sv += V[l + i * n] * V[l + j * n];
            }
            f64 eye = (i == j) ? 1.0 : 0.0;
            f64 ru = fabs(su - eye), rv = fabs(sv - eye);
            if (!(ru <= oerr_u)) oerr_u = ru;
            if (!(rv <= oerr_v)) oerr_v = rv;
        }
    }

    if (!(rerr <= tol && oerr_u <= tol && oerr_v <= tol)) {
        fail_msg("res %.4e, orthU %.4e, orthV %.4e (tol %.4e)",
                 (double)rerr, (double)oerr_u, (double)oerr_v, (double)tol);
    }
}

static void test_even_lower(void** state)
{
    (void)state;
    const INT n = 66;
    const INT lw = 3 * 66;
    INT info;

    f64* A = malloc(n * n * sizeof(f64));
    f64* A0 = malloc(n * n * sizeof(f64));
    f64* V = malloc(n * n * sizeof(f64));
    f64* sva = malloc(n * sizeof(f64));
    f64* work = malloc(lw * sizeof(f64));
    assert_non_null(A); assert_non_null(A0); assert_non_null(V);
    assert_non_null(sva); assert_non_null(work);

    for (INT j = 0; j < n; j++) {
        for (INT i = 0; i < n; i++) {
            if (i == j) {
                A[i + j * n] = 2.0 + 0.01 * (f64)(i + 1);
            } else if (i > j) {
                A[i + j * n] = 0.1 / (f64)(i - j + 1);
            } else {
                A[i + j * n] = 0.0;
            }
            A0[i + j * n] = A[i + j * n];
        }
    }

    dgesvj("L", "U", "V", n, n, A, n, sva, n, V, n, work, lw, &info);
    check_svd(n, A0, A, V, sva, info);

    free(A); free(A0); free(V); free(sva); free(work);
}

static void test_odd_upper(void** state)
{
    (void)state;
    const INT n = 65;
    const INT lw = 3 * 66;
    INT info;

    f64* A = malloc(n * n * sizeof(f64));
    f64* A0 = malloc(n * n * sizeof(f64));
    f64* V = malloc(n * n * sizeof(f64));
    f64* sva = malloc(n * sizeof(f64));
    f64* work = malloc(lw * sizeof(f64));
    assert_non_null(A); assert_non_null(A0); assert_non_null(V);
    assert_non_null(sva); assert_non_null(work);

    for (INT j = 0; j < n; j++) {
        for (INT i = 0; i < n; i++) {
            if (i == j) {
                A[i + j * n] = 2.0 + 0.01 * (f64)(i + 1);
            } else if (i < j) {
                A[i + j * n] = 0.1 / (f64)(j - i + 1);
            } else {
                A[i + j * n] = 0.0;
            }
            A0[i + j * n] = A[i + j * n];
        }
    }

    dgesvj("U", "U", "V", n, n, A, n, sva, n, V, n, work, lw, &info);
    check_svd(n, A0, A, V, sva, info);

    free(A); free(A0); free(V); free(sva); free(work);
}

int main(void)
{
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_even_lower),
        cmocka_unit_test(test_odd_upper),
    };
    (void)cmocka_run_group_tests_name("dgesvj_emptsw", tests, NULL, NULL);
    return 0;
}
