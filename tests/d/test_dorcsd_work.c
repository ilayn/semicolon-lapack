/**
 * @file test_dorcsd_work.c
 * @brief DORCSD and DORCSD2BY1 regression test for WORK(1) on exit.
 *
 * The reference DORCSD and DORCSD2BY1 start the workspace at IPHI = 2, so
 * WORK(1) still holds the optimal LWORK when the routines return. Each case
 * queries LWORK, runs the decomposition of a random orthogonal matrix with
 * that LWORK and checks that WORK(1) on exit equals the queried value.
 */

#include "test_harness.h"
#include "test_rng.h"
#include <stdlib.h>

static f64* random_orthogonal(const INT m, uint64_t seed)
{
    INT info;
    const INT lw = 64 * m;
    uint64_t rng_state[4];
    f64* X = malloc(m * m * sizeof(f64));
    f64* tau = malloc(m * sizeof(f64));
    f64* w = malloc(lw * sizeof(f64));
    assert_non_null(X); assert_non_null(tau); assert_non_null(w);

    rng_seed(rng_state, seed);
    for (INT i = 0; i < m * m; i++) {
        X[i] = rng_uniform_symmetric(rng_state);
    }
    dgeqrf(m, m, X, m, tau, w, lw, &info);
    assert_info_success(info);
    dorgqr(m, m, m, X, m, tau, w, lw, &info);
    assert_info_success(info);
    free(tau);
    free(w);
    return X;
}

static void run_2by1(const INT m, const INT p, const INT q)
{
    INT info;
    f64 wquery;
    f64* X = random_orthogonal(m, 0xC5D0ULL + 100 * m + 10 * p + q);
    f64* theta = malloc(m * sizeof(f64));
    f64* U1 = calloc(m * m, sizeof(f64));
    f64* U2 = calloc(m * m, sizeof(f64));
    f64* V1T = calloc(m * m, sizeof(f64));
    INT* iwork = malloc(m * sizeof(INT));
    assert_non_null(theta); assert_non_null(U1); assert_non_null(U2);
    assert_non_null(V1T); assert_non_null(iwork);

    dorcsd2by1("Y", "Y", "Y", m, p, q, X, m, &X[p], m, theta, U1, m, U2, m,
               V1T, m, &wquery, -1, iwork, &info);
    assert_info_success(info);
    const INT lwork = (INT)wquery;
    f64* work = malloc(lwork * sizeof(f64));
    assert_non_null(work);

    dorcsd2by1("Y", "Y", "Y", m, p, q, X, m, &X[p], m, theta, U1, m, U2, m,
               V1T, m, work, lwork, iwork, &info);
    assert_info_success(info);
    if (work[0] != wquery) {
        fail_msg("DORCSD2BY1 m=%d p=%d q=%d: WORK(1) = %.17g on exit, optimal LWORK = %.17g",
                 (int)m, (int)p, (int)q, (double)work[0], (double)wquery);
    }

    free(X); free(theta); free(U1); free(U2); free(V1T); free(iwork); free(work);
}

static void run_csd(const INT m, const INT p, const INT q)
{
    INT info;
    f64 wquery;
    f64* X = random_orthogonal(m, 0xC5D1ULL + 100 * m + 10 * p + q);
    f64* theta = malloc(m * sizeof(f64));
    f64* U1 = calloc(m * m, sizeof(f64));
    f64* U2 = calloc(m * m, sizeof(f64));
    f64* V1T = calloc(m * m, sizeof(f64));
    f64* V2T = calloc(m * m, sizeof(f64));
    INT* iwork = malloc(m * sizeof(INT));
    assert_non_null(theta); assert_non_null(U1); assert_non_null(U2);
    assert_non_null(V1T); assert_non_null(V2T); assert_non_null(iwork);

    dorcsd("Y", "Y", "Y", "Y", "N", "O", m, p, q, X, m, &X[q * m], m,
           &X[p], m, &X[p + q * m], m, theta, U1, m, U2, m, V1T, m, V2T, m,
           &wquery, -1, iwork, &info);
    assert_info_success(info);
    const INT lwork = (INT)wquery;
    f64* work = malloc(lwork * sizeof(f64));
    assert_non_null(work);

    dorcsd("Y", "Y", "Y", "Y", "N", "O", m, p, q, X, m, &X[q * m], m,
           &X[p], m, &X[p + q * m], m, theta, U1, m, U2, m, V1T, m, V2T, m,
           work, lwork, iwork, &info);
    assert_info_success(info);
    if (work[0] != wquery) {
        fail_msg("DORCSD m=%d p=%d q=%d: WORK(1) = %.17g on exit, optimal LWORK = %.17g",
                 (int)m, (int)p, (int)q, (double)work[0], (double)wquery);
    }

    free(X); free(theta); free(U1); free(U2); free(V1T); free(V2T);
    free(iwork); free(work);
}

static void test_2by1_r_eq_q(void** state)   { (void)state; run_2by1(10, 6, 3); }
static void test_2by1_r_eq_p(void** state)   { (void)state; run_2by1(10, 3, 6); }
static void test_2by1_r_eq_mp(void** state)  { (void)state; run_2by1(10, 7, 5); }
static void test_2by1_r_eq_mq(void** state)  { (void)state; run_2by1(10, 5, 7); }
static void test_csd_square(void** state)    { (void)state; run_csd(10, 5, 5); }
static void test_csd_rect(void** state)      { (void)state; run_csd(10, 6, 3); }

int main(void)
{
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_2by1_r_eq_q),
        cmocka_unit_test(test_2by1_r_eq_p),
        cmocka_unit_test(test_2by1_r_eq_mp),
        cmocka_unit_test(test_2by1_r_eq_mq),
        cmocka_unit_test(test_csd_square),
        cmocka_unit_test(test_csd_rect),
    };
    (void)cmocka_run_group_tests_name("dorcsd_work", tests, NULL, NULL);
    return 0;
}
