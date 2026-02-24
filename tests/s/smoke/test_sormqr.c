/**
 * @file test_sormqr.c
 * @brief CMocka test suite for sormqr/sormlq (Q application).
 *
 * Tests both SORMQR (QR Q-application) and SORMLQ (LQ Q-application)
 * by comparing implicit Q application via DORM* routines against
 * explicit Q*C or C*Q via DGEMM.
 *
 * Verification:
 *   sqrt03: 4 residuals for SORMQR (Left/Right × NoTrans/Trans)
 *   slqt03: 4 residuals for SORMLQ (Left/Right × NoTrans/Trans)
 */

#include "test_harness.h"
#include "verify.h"
#include "test_rng.h"

/* Test threshold - see LAPACK dtest.in */
#define THRESH 20.0f
/* Factorization routines */
typedef struct {
    INT m, n;    /* dimensions of the matrix to factor */
    INT nrhs;    /* "other" dimension for C in the multiplication test */
    INT lda;
    f32 *A, *AF;
    f32 *C, *CC, *Q;
    f32 *tau, *work, *rwork;
    f32 *d, *genwork;
    INT lwork;
    uint64_t seed;
} orm_fixture_t;

static uint64_t g_seed = 7001;

static int orm_setup(void **state, INT m, INT n, INT nrhs)
{
    orm_fixture_t *fix = malloc(sizeof(orm_fixture_t));
    assert_non_null(fix);

    fix->m = m;
    fix->n = n;
    fix->nrhs = nrhs;
    INT maxmn = m > n ? m : n;
    INT maxall = maxmn > nrhs ? maxmn : nrhs;
    fix->lda = maxall;
    fix->seed = g_seed++;

    INT minmn = m < n ? m : n;
    fix->lwork = maxall * 64;

    fix->A = calloc(fix->lda * maxall, sizeof(f32));
    fix->AF = calloc(fix->lda * maxall, sizeof(f32));
    fix->C = calloc(fix->lda * maxall, sizeof(f32));
    fix->CC = calloc(fix->lda * maxall, sizeof(f32));
    fix->Q = calloc(fix->lda * maxall, sizeof(f32));
    fix->tau = calloc(minmn, sizeof(f32));
    fix->work = calloc(fix->lwork, sizeof(f32));
    fix->rwork = calloc(maxall, sizeof(f32));
    fix->d = calloc(minmn, sizeof(f32));
    fix->genwork = calloc(3 * maxall, sizeof(f32));

    assert_non_null(fix->A);
    assert_non_null(fix->AF);
    assert_non_null(fix->C);
    assert_non_null(fix->CC);
    assert_non_null(fix->Q);
    assert_non_null(fix->tau);
    assert_non_null(fix->work);
    assert_non_null(fix->rwork);
    assert_non_null(fix->d);
    assert_non_null(fix->genwork);

    *state = fix;
    return 0;
}

static int orm_teardown(void **state)
{
    orm_fixture_t *fix = *state;
    if (fix) {
        free(fix->A); free(fix->AF);
        free(fix->C); free(fix->CC); free(fix->Q);
        free(fix->tau); free(fix->work); free(fix->rwork);
        free(fix->d); free(fix->genwork);
        free(fix);
    }
    return 0;
}

/* Setup with m=n (square) and nrhs for the C matrix dimension */
static int setup_5x5(void **state) { return orm_setup(state, 5, 5, 3); }
static int setup_10x10(void **state) { return orm_setup(state, 10, 10, 5); }
static int setup_20x20(void **state) { return orm_setup(state, 20, 20, 7); }
static int setup_50x50(void **state) { return orm_setup(state, 50, 50, 10); }
static int setup_20x10(void **state) { return orm_setup(state, 20, 10, 5); }
static int setup_10x20(void **state) { return orm_setup(state, 10, 20, 5); }

/**
 * Test SORMQR: Apply Q from QR factorization.
 * Q is m-by-m, C is m-by-nrhs (Left) or nrhs-by-m (Right).
 */
static void test_dormqr(void **state)
{
    orm_fixture_t *fix = *state;
    INT m = fix->m, n = fix->n;
    INT minmn = m < n ? m : n;
    f32 result[4];
    INT info;

    for (INT imat = 1; imat <= 4; imat++) {
        fix->seed = g_seed++;
        char type, dist;
        INT kl, ku, mode;
        f32 anorm, cndnum;

        slatb4("SGE", imat, m, n, &type, &kl, &ku, &anorm, &mode, &cndnum, &dist);
        uint64_t rng_state[4];
        rng_seed(rng_state, fix->seed);
        slatms(m, n, &dist, &type, fix->d, mode, cndnum, anorm,
               kl, ku, "N", fix->A, fix->lda, fix->genwork, &info, rng_state);
        assert_int_equal(info, 0);

        /* QR factorize into AF */
        slacpy("F", m, n, fix->A, fix->lda, fix->AF, fix->lda);
        sgeqrf(m, n, fix->AF, fix->lda, fix->tau, fix->work, fix->lwork, &info);
        assert_int_equal(info, 0);

        /* Test all 4 combinations of side/trans.
         * sqrt03(m, n, k, AF, C, CC, Q, lda, tau, work, lwork, rwork, result)
         * Here m = order of Q, n = other dimension of C, k = number of reflectors */
        sqrt03(m, fix->nrhs, minmn, fix->AF, fix->C, fix->CC, fix->Q,
               fix->lda, fix->tau, fix->work, fix->lwork, fix->rwork, result);

        assert_residual_ok(result[0]);
        assert_residual_ok(result[1]);
        assert_residual_ok(result[2]);
        assert_residual_ok(result[3]);
    }
}

/**
 * Test SORMLQ: Apply Q from LQ factorization.
 * Q is n-by-n, C is n-by-nrhs (Left) or nrhs-by-n (Right).
 */
static void test_dormlq(void **state)
{
    orm_fixture_t *fix = *state;
    INT m = fix->m, n = fix->n;
    INT minmn = m < n ? m : n;
    f32 result[4];
    INT info;

    for (INT imat = 1; imat <= 4; imat++) {
        fix->seed = g_seed++;
        char type, dist;
        INT kl, ku, mode;
        f32 anorm, cndnum;

        slatb4("SGE", imat, m, n, &type, &kl, &ku, &anorm, &mode, &cndnum, &dist);
        uint64_t rng_state[4];
        rng_seed(rng_state, fix->seed);
        slatms(m, n, &dist, &type, fix->d, mode, cndnum, anorm,
               kl, ku, "N", fix->A, fix->lda, fix->genwork, &info, rng_state);
        assert_int_equal(info, 0);

        /* LQ factorize into AF */
        slacpy("F", m, n, fix->A, fix->lda, fix->AF, fix->lda);
        sgelqf(m, n, fix->AF, fix->lda, fix->tau, fix->work, fix->lwork, &info);
        assert_int_equal(info, 0);

        /* Test all 4 combinations.
         * slqt03(m, n, k, ...) where m=other dim, n=order of Q, k=reflectors */
        slqt03(fix->nrhs, n, minmn, fix->AF, fix->C, fix->CC, fix->Q,
               fix->lda, fix->tau, fix->work, fix->lwork, fix->rwork, result);

        assert_residual_ok(result[0]);
        assert_residual_ok(result[1]);
        assert_residual_ok(result[2]);
        assert_residual_ok(result[3]);
    }
}

int main(void)
{
    const struct CMUnitTest tests[] = {
        /* SORMQR tests */
        cmocka_unit_test_setup_teardown(test_dormqr, setup_5x5, orm_teardown),
        cmocka_unit_test_setup_teardown(test_dormqr, setup_10x10, orm_teardown),
        cmocka_unit_test_setup_teardown(test_dormqr, setup_20x20, orm_teardown),
        cmocka_unit_test_setup_teardown(test_dormqr, setup_50x50, orm_teardown),
        cmocka_unit_test_setup_teardown(test_dormqr, setup_20x10, orm_teardown),
        cmocka_unit_test_setup_teardown(test_dormqr, setup_10x20, orm_teardown),
        /* SORMLQ tests */
        cmocka_unit_test_setup_teardown(test_dormlq, setup_5x5, orm_teardown),
        cmocka_unit_test_setup_teardown(test_dormlq, setup_10x10, orm_teardown),
        cmocka_unit_test_setup_teardown(test_dormlq, setup_20x20, orm_teardown),
        cmocka_unit_test_setup_teardown(test_dormlq, setup_50x50, orm_teardown),
        cmocka_unit_test_setup_teardown(test_dormlq, setup_20x10, orm_teardown),
        cmocka_unit_test_setup_teardown(test_dormlq, setup_10x20, orm_teardown),
    };
    return cmocka_run_group_tests_name("dormqr", tests, NULL, NULL);
}
