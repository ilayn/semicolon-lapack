/**
 * @file test_dbbcsd_restart.c
 * @brief DBBCSD regression test for the U2 rotation on a bulge-chase restart.
 *
 * test_restart_u2_rotation runs DBBCSD on angles where RESTART21 and
 * RESTART22 both hold with NU < MU on the last step of the chase, and puts a
 * NaN just past the end of B21E (dimension Q-1). The U2 rotation in that
 * branch is built from B21E and B21D, so the NaN must not reach THETA, U1,
 * U2, V1T or V2T, and INFO must be 0.
 */

#include "test_harness.h"
#include <math.h>
#include <stdlib.h>

static void test_restart_u2_rotation(void** state)
{
    (void)state;
    const INT q = 3;
    const INT p = q;
    const INT m = 2 * q;
    const INT lwork = 8 * q;
    const f64 piover2 = 1.57079632679489661923132169163975144210;
    INT info;

    f64* theta = malloc(q * sizeof(f64));
    f64* phi = malloc((q - 1) * sizeof(f64));
    f64* U1 = calloc(p * p, sizeof(f64));
    f64* U2 = calloc((m - p) * (m - p), sizeof(f64));
    f64* V1T = calloc(q * q, sizeof(f64));
    f64* V2T = calloc((m - q) * (m - q), sizeof(f64));
    f64* B11D = calloc(q, sizeof(f64));
    f64* B11E = calloc(q - 1, sizeof(f64));
    f64* B12D = calloc(q, sizeof(f64));
    f64* B12E = calloc(q - 1, sizeof(f64));
    f64* B21D = calloc(q, sizeof(f64));
    f64* B21E = calloc(q, sizeof(f64));
    f64* B22D = calloc(q, sizeof(f64));
    f64* B22E = calloc(q - 1, sizeof(f64));
    f64* work = malloc(lwork * sizeof(f64));
    assert_non_null(theta); assert_non_null(phi); assert_non_null(U1);
    assert_non_null(U2); assert_non_null(V1T); assert_non_null(V2T);
    assert_non_null(B11D); assert_non_null(B11E); assert_non_null(B12D);
    assert_non_null(B12E); assert_non_null(B21D); assert_non_null(B21E);
    assert_non_null(B22D); assert_non_null(B22E); assert_non_null(work);

    theta[0] = 0.25; theta[1] = 0.0; theta[2] = 0.0;
    phi[0] = piover2 - 1.0e-4; phi[1] = 0.5;
    for (INT i = 0; i < p; i++) U1[i + i * p] = 1.0;
    for (INT i = 0; i < m - p; i++) U2[i + i * (m - p)] = 1.0;
    for (INT i = 0; i < q; i++) V1T[i + i * q] = 1.0;
    for (INT i = 0; i < m - q; i++) V2T[i + i * (m - q)] = 1.0;
    B21E[q - 1] = (f64)NAN;

    dbbcsd("Y", "Y", "Y", "Y", "N", m, p, q, theta, phi,
           U1, p, U2, m - p, V1T, q, V2T, m - q,
           B11D, B11E, B12D, B12E, B21D, B21E, B22D, B22E,
           work, lwork, &info);

    assert_info_success(info);
    for (INT i = 0; i < q; i++) {
        assert_true(isfinite(theta[i]));
    }
    for (INT i = 0; i < p * p; i++) {
        assert_true(isfinite(U1[i]));
    }
    for (INT i = 0; i < (m - p) * (m - p); i++) {
        assert_true(isfinite(U2[i]));
    }
    for (INT i = 0; i < q * q; i++) {
        assert_true(isfinite(V1T[i]));
    }
    for (INT i = 0; i < (m - q) * (m - q); i++) {
        assert_true(isfinite(V2T[i]));
    }

    free(theta); free(phi); free(U1); free(U2); free(V1T); free(V2T);
    free(B11D); free(B11E); free(B12D); free(B12E);
    free(B21D); free(B21E); free(B22D); free(B22E); free(work);
}

int main(void)
{
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_restart_u2_rotation),
    };
    (void)cmocka_run_group_tests_name("dbbcsd_restart", tests, NULL, NULL);
    return 0;
}
