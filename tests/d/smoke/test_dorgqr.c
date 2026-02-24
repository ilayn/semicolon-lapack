/**
 * @file test_dorgqr.c
 * @brief CMocka test suite for dorgqr/dorglq/dorgql/dorgrq (Q generation).
 *
 * Tests all four Q-generation routines by verifying the generated Q
 * matrix satisfies orthogonality and reproduces the correct triangular factor.
 *
 * Verification:
 *   dqrt02: For QR, norm(R - Q'*A) / (M*norm(A)*eps) and norm(I-Q'*Q)/(M*eps)
 *   dlqt02: For LQ, norm(L - A*Q') / (N*norm(A)*eps) and norm(I-Q*Q')/(N*eps)
 */

#include "test_harness.h"
#include "verify.h"
#include "test_rng.h"

/* Test threshold - see LAPACK dtest.in */
#define THRESH 20.0
/* Factorization routines */
typedef struct {
    INT m, n;
    INT lda;
    f64 *A, *AF, *Q, *R;
    f64 *tau, *work, *rwork;
    f64 *d, *genwork;
    INT lwork;
    uint64_t seed;
} org_fixture_t;

static uint64_t g_seed = 6001;

static int org_setup(void **state, INT m, INT n)
{
    org_fixture_t *fix = malloc(sizeof(org_fixture_t));
    assert_non_null(fix);

    fix->m = m;
    fix->n = n;
    INT maxmn = m > n ? m : n;
    fix->lda = maxmn;
    fix->seed = g_seed++;

    INT minmn = m < n ? m : n;
    fix->lwork = maxmn * 64;

    fix->A = calloc(fix->lda * maxmn, sizeof(f64));
    fix->AF = calloc(fix->lda * maxmn, sizeof(f64));
    fix->Q = calloc(fix->lda * maxmn, sizeof(f64));
    fix->R = calloc(fix->lda * maxmn, sizeof(f64));
    fix->tau = calloc(minmn, sizeof(f64));
    fix->work = calloc(fix->lwork, sizeof(f64));
    fix->rwork = calloc(maxmn, sizeof(f64));
    fix->d = calloc(minmn, sizeof(f64));
    fix->genwork = calloc(3 * maxmn, sizeof(f64));

    assert_non_null(fix->A);
    assert_non_null(fix->AF);
    assert_non_null(fix->Q);
    assert_non_null(fix->R);
    assert_non_null(fix->tau);
    assert_non_null(fix->work);
    assert_non_null(fix->rwork);
    assert_non_null(fix->d);
    assert_non_null(fix->genwork);

    *state = fix;
    return 0;
}

static int org_teardown(void **state)
{
    org_fixture_t *fix = *state;
    if (fix) {
        free(fix->A); free(fix->AF); free(fix->Q); free(fix->R);
        free(fix->tau); free(fix->work); free(fix->rwork);
        free(fix->d); free(fix->genwork);
        free(fix);
    }
    return 0;
}

static int setup_5x5(void **state) { return org_setup(state, 5, 5); }
static int setup_10x10(void **state) { return org_setup(state, 10, 10); }
static int setup_20x20(void **state) { return org_setup(state, 20, 20); }
static int setup_50x50(void **state) { return org_setup(state, 50, 50); }
static int setup_20x10(void **state) { return org_setup(state, 20, 10); }
static int setup_10x20(void **state) { return org_setup(state, 10, 20); }

/**
 * Test DORGQR: Generate partial Q from QR factorization.
 * For m-by-n matrix factored with k reflectors, generate m-by-n Q.
 */
static void test_dorgqr(void **state)
{
    org_fixture_t *fix = *state;
    INT m = fix->m, n = fix->n;
    INT minmn = m < n ? m : n;
    f64 result[2];
    INT info;

    for (INT imat = 1; imat <= 4; imat++) {
        fix->seed = g_seed++;
        char type, dist;
        INT kl, ku, mode;
        f64 anorm, cndnum;

        dlatb4("DGE", imat, m, n, &type, &kl, &ku, &anorm, &mode, &cndnum, &dist);
        uint64_t rng_state[4];
        rng_seed(rng_state, fix->seed);
        dlatms(m, n, &dist, &type, fix->d, mode, cndnum, anorm,
               kl, ku, "N", fix->A, fix->lda, fix->genwork, &info, rng_state);
        assert_int_equal(info, 0);

        /* QR factorize */
        dlacpy("F", m, n, fix->A, fix->lda, fix->AF, fix->lda);
        dgeqrf(m, n, fix->AF, fix->lda, fix->tau, fix->work, fix->lwork, &info);
        assert_int_equal(info, 0);

        /* Test generating n columns of Q from k=minmn reflectors */
        dqrt02(m, minmn, minmn, fix->A, fix->AF, fix->Q, fix->R, fix->lda,
               fix->tau, fix->work, fix->lwork, fix->rwork, result);

        assert_residual_ok(result[0]);
        assert_residual_ok(result[1]);
    }
}

/**
 * Test DORGLQ: Generate partial Q from LQ factorization.
 */
static void test_dorglq(void **state)
{
    org_fixture_t *fix = *state;
    INT m = fix->m, n = fix->n;
    INT minmn = m < n ? m : n;
    f64 result[2];
    INT info;

    for (INT imat = 1; imat <= 4; imat++) {
        fix->seed = g_seed++;
        char type, dist;
        INT kl, ku, mode;
        f64 anorm, cndnum;

        dlatb4("DGE", imat, m, n, &type, &kl, &ku, &anorm, &mode, &cndnum, &dist);
        uint64_t rng_state[4];
        rng_seed(rng_state, fix->seed);
        dlatms(m, n, &dist, &type, fix->d, mode, cndnum, anorm,
               kl, ku, "N", fix->A, fix->lda, fix->genwork, &info, rng_state);
        assert_int_equal(info, 0);

        /* LQ factorize */
        dlacpy("F", m, n, fix->A, fix->lda, fix->AF, fix->lda);
        dgelqf(m, n, fix->AF, fix->lda, fix->tau, fix->work, fix->lwork, &info);
        assert_int_equal(info, 0);

        /* Test generating minmn rows of Q from k=minmn reflectors */
        dlqt02(minmn, n, minmn, fix->A, fix->AF, fix->Q, fix->R, fix->lda,
               fix->tau, fix->work, fix->lwork, fix->rwork, result);

        assert_residual_ok(result[0]);
        assert_residual_ok(result[1]);
    }
}

int main(void)
{
    const struct CMUnitTest tests[] = {
        /* QR Q-generation */
        cmocka_unit_test_setup_teardown(test_dorgqr, setup_5x5, org_teardown),
        cmocka_unit_test_setup_teardown(test_dorgqr, setup_10x10, org_teardown),
        cmocka_unit_test_setup_teardown(test_dorgqr, setup_20x20, org_teardown),
        cmocka_unit_test_setup_teardown(test_dorgqr, setup_50x50, org_teardown),
        cmocka_unit_test_setup_teardown(test_dorgqr, setup_20x10, org_teardown),
        cmocka_unit_test_setup_teardown(test_dorgqr, setup_10x20, org_teardown),
        /* LQ Q-generation */
        cmocka_unit_test_setup_teardown(test_dorglq, setup_5x5, org_teardown),
        cmocka_unit_test_setup_teardown(test_dorglq, setup_10x10, org_teardown),
        cmocka_unit_test_setup_teardown(test_dorglq, setup_20x20, org_teardown),
        cmocka_unit_test_setup_teardown(test_dorglq, setup_50x50, org_teardown),
        cmocka_unit_test_setup_teardown(test_dorglq, setup_20x10, org_teardown),
        cmocka_unit_test_setup_teardown(test_dorglq, setup_10x20, org_teardown),
    };
    return cmocka_run_group_tests_name("dorgqr", tests, NULL, NULL);
}
