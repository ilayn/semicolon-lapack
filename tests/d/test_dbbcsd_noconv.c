/**
 * @file test_dbbcsd_noconv.c
 * @brief DBBCSD regression test for INFO when the iteration does not
 *        converge.
 *
 * When DBBCSD does not converge, INFO is the number of nonzero entries
 * of PHI, which has dimension Q-1. A NaN in THETA or PHI keeps the
 * bulge chase from converging, so DBBCSD takes this exit. Each case puts
 * the angles at the start of an allocation with one extra element after
 * PHI, set first to 0 and then to 1, and checks that INFO is the same in
 * both calls and lies in 1..Q-1.
 */

#include "test_harness.h"
#include <math.h>
#include <stdlib.h>

static INT run(const INT q, const INT nanphi, const f64 after)
{
    const INT p = q;
    const INT m = 2 * q;
    const INT lwork = 8 * q;
    INT info;

    f64* theta = malloc(q * sizeof(f64));
    f64* phi = malloc(q * sizeof(f64));
    f64* U1 = calloc(p * p, sizeof(f64));
    f64* U2 = calloc((m - p) * (m - p), sizeof(f64));
    f64* V1T = calloc(q * q, sizeof(f64));
    f64* V2T = calloc((m - q) * (m - q), sizeof(f64));
    f64* B = calloc(8 * q, sizeof(f64));
    f64* work = calloc(lwork, sizeof(f64));
    assert_non_null(theta); assert_non_null(phi); assert_non_null(U1);
    assert_non_null(U2); assert_non_null(V1T); assert_non_null(V2T);
    assert_non_null(B); assert_non_null(work);

    for (INT i = 0; i < q; i++) {
        theta[i] = 0.1 + 0.3 * (f64)i;
    }
    for (INT i = 0; i < q - 1; i++) {
        phi[i] = 0.2 + 0.1 * (f64)i;
    }
    if (nanphi) {
        phi[0] = NAN;
    } else {
        theta[0] = NAN;
    }
    phi[q - 1] = after;
    for (INT i = 0; i < p; i++) U1[i + i * p] = 1.0;
    for (INT i = 0; i < m - p; i++) U2[i + i * (m - p)] = 1.0;
    for (INT i = 0; i < q; i++) V1T[i + i * q] = 1.0;
    for (INT i = 0; i < m - q; i++) V2T[i + i * (m - q)] = 1.0;

    dbbcsd("Y", "Y", "Y", "Y", "N", m, p, q, theta, phi,
           U1, p, U2, m - p, V1T, q, V2T, m - q,
           B, B + q, B + 2 * q, B + 3 * q, B + 4 * q, B + 5 * q, B + 6 * q,
           B + 7 * q, work, lwork, &info);

    free(theta); free(phi); free(U1); free(U2); free(V1T); free(V2T);
    free(B); free(work);
    return info;
}

static void test_noconv_info(void** state)
{
    (void)state;
    for (INT nanphi = 0; nanphi < 2; nanphi++) {
        for (INT q = 2; q <= 6; q++) {
            INT info0 = run(q, nanphi, 0.0);
            INT info1 = run(q, nanphi, 1.0);
            if (info0 != info1 || info0 < 1 || info0 > q - 1) {
                fail_msg("Q = %d, NaN in %s: INFO = %d with 0 after PHI(Q-1), %d with 1",
                         (int)q, nanphi ? "PHI" : "THETA", (int)info0, (int)info1);
            }
        }
    }
}

int main(void)
{
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_noconv_info),
    };
    (void)cmocka_run_group_tests_name("dbbcsd_noconv", tests, NULL, NULL);
    return 0;
}
