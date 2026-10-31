/**
 * @file test_zgsvj0_sweeps.c
 * @brief ZGSVJ0 and ZGSVJ1 regression test for the convergence test after
 *        the second sweep.
 *
 * ZGSVJ0 and ZGSVJ1 stop after sweep i when i > SWBAND+1,
 * MXAAPQ < SQRT(N)*TOL and N*MXAAPQ*MXSINJ < TOL, with SWBAND = 0, so the
 * second sweep is the first one after which they can stop; if no sweep
 * passes the test they return INFO = NSWEEP-1. Each case calls the routine
 * with NSWEEP = 2 on a graded matrix with nonorthogonal columns, for which
 * the second sweep still rotates and then passes the test for TOL from 0.7
 * to 1.4 times the value used here. It checks INFO = 0, that A equals
 * A0*V, and that SVA holds the column norms of A*diag(D).
 */

#include "test_harness.h"
#include "verify.h"
#include "test_rng.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define THRESH 30.0

static void run_case(const INT use_gsvj1, const char* jobv, const f64 delta,
                     const f64 tol, const uint64_t seed)
{
    const INT n = 32;
    const INT m = n + 4;
    const INT lda = m;
    const INT ldv = n;
    const INT n1 = n / 2;
    const INT lwork = m;
    const f64 grade = 4.0;
    const f64 eps = dlamch("E");
    const f64 sfmin = dlamch("S");
    INT info;
    f64 anorm, resid, nrmerr;
    uint64_t rng_state[4];

    c128* A0 = calloc((size_t)lda * n, sizeof(c128));
    c128* A = calloc((size_t)lda * n, sizeof(c128));
    c128* V = calloc((size_t)ldv * n, sizeof(c128));
    c128* D = calloc(n, sizeof(c128));
    f64* SVA = calloc(n, sizeof(f64));
    c128* work = calloc(lwork, sizeof(c128));
    assert_non_null(A0); assert_non_null(A); assert_non_null(V);
    assert_non_null(D); assert_non_null(SVA); assert_non_null(work);

    rng_seed(rng_state, seed);
    for (INT j = 0; j < n; j++) {
        A0[j + j * lda] = CMPLX(1.0, 0.0);
        for (INT i = 0; i < n; i++) {
            if (i == j) continue;
            if (use_gsvj1 && !(i < n1 && j >= n1)) continue;
            f64 re = rng_uniform_symmetric(rng_state);
            f64 im = rng_uniform_symmetric(rng_state);
            A0[i + j * lda] = delta * CMPLX(re, im);
        }
    }
    for (INT j = 0; j < n; j++) {
        const f64 s = pow(grade, -(f64)j);
        for (INT i = 0; i < m; i++) A0[i + j * lda] *= s;
    }
    memcpy(A, A0, (size_t)lda * n * sizeof(c128));
    for (INT j = 0; j < n; j++) {
        f64 s = 0.0;
        for (INT i = 0; i < m; i++) {
            s += creal(A[i + j * lda] * conj(A[i + j * lda]));
        }
        SVA[j] = sqrt(s);
        D[j] = CMPLX(1.0, 0.0);
        V[j + j * ldv] = CMPLX(1.0, 0.0);
    }

    if (use_gsvj1) {
        zgsvj1(jobv, m, n, n1, A, lda, D, SVA, n, V, ldv, eps, sfmin, tol, 2,
               work, lwork, &info);
    } else {
        zgsvj0(jobv, m, n, A, lda, D, SVA, n, V, ldv, eps, sfmin, tol, 2,
               work, lwork, &info);
    }
    assert_info_success(info);

    anorm = 0.0;
    for (INT j = 0; j < n; j++) {
        if (SVA[j] > anorm) anorm = SVA[j];
    }

    nrmerr = 0.0;
    for (INT j = 0; j < n; j++) {
        f64 s = 0.0;
        for (INT i = 0; i < m; i++) {
            c128 t = A[i + j * lda] * D[j];
            s += creal(t * conj(t));
        }
        f64 e = fabs(sqrt(s) - SVA[j]);
        if (e > nrmerr) nrmerr = e;
    }
    nrmerr = nrmerr / ((f64)m * eps * anorm);
    assert_residual_ok(nrmerr);

    if (jobv[0] == 'V') {
        resid = 0.0;
        for (INT j = 0; j < n; j++) {
            for (INT i = 0; i < m; i++) {
                c128 s = CMPLX(0.0, 0.0);
                for (INT k = 0; k < n; k++) s += A0[i + k * lda] * V[k + j * ldv];
                f64 e = cabs(s - A[i + j * lda]);
                if (e > resid) resid = e;
            }
        }
        resid = resid / ((f64)n * eps * anorm);
        assert_residual_ok(resid);
    }

    free(A0); free(A); free(V); free(D); free(SVA); free(work);
}

static void test_zgsvj0(void** state)
{
    (void)state;
    const uint64_t seeds[3] = {0x6A22ULL, 0x6A24ULL, 0x6A2BULL};
    for (INT k = 0; k < 3; k++) {
        run_case(0, "V", 0.3, 2.0e-3, seeds[k]);
        run_case(0, "N", 0.3, 2.0e-3, seeds[k]);
    }
}

static void test_zgsvj1(void** state)
{
    (void)state;
    const uint64_t seeds[3] = {0x6A37ULL, 0x6A44ULL, 0x6A92ULL};
    for (INT k = 0; k < 3; k++) {
        run_case(1, "V", 1.0, 5.0e-3, seeds[k]);
        run_case(1, "N", 1.0, 5.0e-3, seeds[k]);
    }
}

int main(void)
{
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_zgsvj0),
        cmocka_unit_test(test_zgsvj1),
    };
    (void)cmocka_run_group_tests_name("zgsvj0_sweeps", tests, NULL, NULL);
    return 0;
}
