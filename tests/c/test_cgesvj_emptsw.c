/**
 * @file test_cgesvj_emptsw.c
 * @brief CGESVJ/CGSVJ0 regression test for the EMPTSW overflow fix.
 *
 * Port of LAPACK TESTING/EIG/test_zgesvj_emptsw.f90.
 *
 * Coverage driver for the EMPTSW integer-overflow fix in CGESVJ/CGSVJ0
 * (Reference-LAPACK issue 1312: EMPTSW = N*(N-1)/2 overflows 32-bit
 * arithmetic for N >= 46342, so it is computed in a factored form that
 * never overflows).
 *
 * CGSVJ0 is only reached through CGESVJ's triangular (JOBA='L'/'U')
 * blocked path for N > 64, which the SVD tests (JOBA='G') never exercise.
 * This driver therefore calls CGESVJ with N > 64 lower- and
 * upper-triangular matrices (even and odd N), which takes the blocked
 * CGESVJ -> CGSVJ0 path and covers the fixed lines in both files. It
 * validates the live factorization directly (residual and orthogonality),
 * so there is no replica of the EMPTSW expression here to go stale.
 */

#include "test_harness.h"
#include "verify.h"
#include <math.h>
#include <stdlib.h>

static void check_svd(const INT n, const c64* A0, const c64* U,
                      const c64* V, const f32* sva, const INT info)
{
    assert_info_success(info);

    for (INT i = 0; i < n; i++) {
        if (sva[i] <= 0.0f) {
            fail_msg("sva[%d] = %.3e is not positive", (int)i, (double)sva[i]);
        }
        if (i > 0 && sva[i - 1] < sva[i]) {
            fail_msg("sva[%d] = %.3e < sva[%d] = %.3e", (int)(i - 1),
                     (double)sva[i - 1], (int)i, (double)sva[i]);
        }
    }

    f32 eps = slamch("P");
    f32 tol = (f32)(100 * n) * eps;

    f32 anorm = 0.0f;
    for (INT i = 0; i < n * n; i++) {
        if (cabsf(A0[i]) > anorm) anorm = cabsf(A0[i]);
    }

    f32 rerr = 0.0f;
    for (INT j = 0; j < n; j++) {
        for (INT i = 0; i < n; i++) {
            c64 s = CMPLXF(0.0f, 0.0f);
            for (INT l = 0; l < n; l++) {
                s += (U[i + l * n] * sva[l]) * conjf(V[j + l * n]);
            }
            f32 r = cabsf(A0[i + j * n] - s);
            if (!(r <= rerr)) rerr = r;
        }
    }
    rerr = rerr / anorm;

    f32 oerr_u = 0.0f, oerr_v = 0.0f;
    for (INT j = 0; j < n; j++) {
        for (INT i = 0; i < n; i++) {
            c64 su = CMPLXF(0.0f, 0.0f), sv = CMPLXF(0.0f, 0.0f);
            for (INT l = 0; l < n; l++) {
                su += conjf(U[l + i * n]) * U[l + j * n];
                sv += conjf(V[l + i * n]) * V[l + j * n];
            }
            c64 eye = (i == j) ? CMPLXF(1.0f, 0.0f) : CMPLXF(0.0f, 0.0f);
            f32 ru = cabsf(su - eye), rv = cabsf(sv - eye);
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
    const INT lw = 3 * 66, lrw = 2 * 66;
    INT info;

    c64* A = malloc(n * n * sizeof(c64));
    c64* A0 = malloc(n * n * sizeof(c64));
    c64* V = malloc(n * n * sizeof(c64));
    f32* sva = malloc(n * sizeof(f32));
    c64* cwork = malloc(lw * sizeof(c64));
    f32* rwork = malloc(lrw * sizeof(f32));
    assert_non_null(A); assert_non_null(A0); assert_non_null(V);
    assert_non_null(sva); assert_non_null(cwork); assert_non_null(rwork);

    for (INT j = 0; j < n; j++) {
        for (INT i = 0; i < n; i++) {
            if (i == j) {
                A[i + j * n] = CMPLXF(2.0f + 0.01f * (f32)(i + 1),
                                      0.02f * (f32)(i + 1));
            } else if (i > j) {
                A[i + j * n] = CMPLXF(0.1f / (f32)(i - j + 1),
                                      0.05f / (f32)(i - j + 1));
            } else {
                A[i + j * n] = CMPLXF(0.0f, 0.0f);
            }
            A0[i + j * n] = A[i + j * n];
        }
    }

    cgesvj("L", "U", "V", n, n, A, n, sva, n, V, n, cwork, lw, rwork, lrw,
           &info);
    check_svd(n, A0, A, V, sva, info);

    free(A); free(A0); free(V); free(sva); free(cwork); free(rwork);
}

static void test_odd_upper(void** state)
{
    (void)state;
    const INT n = 65;
    const INT lw = 3 * 66, lrw = 2 * 66;
    INT info;

    c64* A = malloc(n * n * sizeof(c64));
    c64* A0 = malloc(n * n * sizeof(c64));
    c64* V = malloc(n * n * sizeof(c64));
    f32* sva = malloc(n * sizeof(f32));
    c64* cwork = malloc(lw * sizeof(c64));
    f32* rwork = malloc(lrw * sizeof(f32));
    assert_non_null(A); assert_non_null(A0); assert_non_null(V);
    assert_non_null(sva); assert_non_null(cwork); assert_non_null(rwork);

    for (INT j = 0; j < n; j++) {
        for (INT i = 0; i < n; i++) {
            if (i == j) {
                A[i + j * n] = CMPLXF(2.0f + 0.01f * (f32)(i + 1),
                                      0.02f * (f32)(i + 1));
            } else if (i < j) {
                A[i + j * n] = CMPLXF(0.1f / (f32)(j - i + 1),
                                      0.05f / (f32)(j - i + 1));
            } else {
                A[i + j * n] = CMPLXF(0.0f, 0.0f);
            }
            A0[i + j * n] = A[i + j * n];
        }
    }

    cgesvj("U", "U", "V", n, n, A, n, sva, n, V, n, cwork, lw, rwork, lrw,
           &info);
    check_svd(n, A0, A, V, sva, info);

    free(A); free(A0); free(V); free(sva); free(cwork); free(rwork);
}

int main(void)
{
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_even_lower),
        cmocka_unit_test(test_odd_upper),
    };
    (void)cmocka_run_group_tests_name("cgesvj_emptsw", tests, NULL, NULL);
    return 0;
}
