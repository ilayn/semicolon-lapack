/**
 * @file test_sgeqrf.c
 * @brief CMocka test suite for sgeqrf (QR factorization).
 *
 * Tests the blocked QR factorization routine sgeqrf using LAPACK's
 * verification methodology with normalized residuals.
 *
 * Verification:
 *   sqrt01: norm(R - Q'*A) / (M * norm(A) * eps) and
 *           norm(I - Q'*Q) / (M * eps)
 *
 * Matrix types tested (from slatb4):
 *   1. Diagonal
 *   2. Upper triangular
 *   3. Lower triangular
 *   4. Random, well-conditioned (cond=2)
 *   8. Random, ill-conditioned (cond ~ 3e7)
 *   9. Random, very ill-conditioned (cond ~ 9e15)
 *   10. Scaled near underflow
 *   11. Scaled near overflow
 */

#include "test_harness.h"
#include "verify.h"
#include "test_rng.h"

/* Test threshold - see LAPACK dtest.in */
#define THRESH 20.0f
#include <cblas.h>

typedef struct {
    int m, n;
    int lda;
    f32 *A;
    f32 *AF;
    f32 *Q;
    f32 *R;
    f32 *tau;
    f32 *work;
    f32 *rwork;
    f32 *d;
    f32 *genwork;
    int lwork;
    uint64_t seed;
} qr_fixture_t;

static uint64_t g_seed = 2001;

static int qr_setup(void **state, int m, int n)
{
    qr_fixture_t *fix = malloc(sizeof(qr_fixture_t));
    assert_non_null(fix);

    fix->m = m;
    fix->n = n;
    int maxmn = m > n ? m : n;
    fix->lda = maxmn;  /* lda >= max(m,n) for verification routines */
    fix->seed = g_seed++;

    int minmn = m < n ? m : n;
    fix->lwork = maxmn * 64;  /* generous workspace */

    fix->A = calloc(fix->lda * maxmn, sizeof(f32));
    fix->AF = calloc(fix->lda * maxmn, sizeof(f32));
    fix->Q = calloc(fix->lda * maxmn, sizeof(f32));
    fix->R = calloc(fix->lda * maxmn, sizeof(f32));
    fix->tau = calloc(minmn, sizeof(f32));
    fix->work = calloc(fix->lwork, sizeof(f32));
    fix->rwork = calloc(maxmn, sizeof(f32));
    fix->d = calloc(minmn, sizeof(f32));
    fix->genwork = calloc(3 * maxmn, sizeof(f32));

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

static int qr_teardown(void **state)
{
    qr_fixture_t *fix = *state;
    if (fix) {
        free(fix->A);
        free(fix->AF);
        free(fix->Q);
        free(fix->R);
        free(fix->tau);
        free(fix->work);
        free(fix->rwork);
        free(fix->d);
        free(fix->genwork);
        free(fix);
    }
    return 0;
}

/* Size-specific setups */
static int setup_5x5(void **state) { return qr_setup(state, 5, 5); }
static int setup_10x10(void **state) { return qr_setup(state, 10, 10); }
static int setup_20x20(void **state) { return qr_setup(state, 20, 20); }
static int setup_50x50(void **state) { return qr_setup(state, 50, 50); }
static int setup_10x5(void **state) { return qr_setup(state, 10, 5); }
static int setup_20x10(void **state) { return qr_setup(state, 20, 10); }
static int setup_5x10(void **state) { return qr_setup(state, 5, 10); }
static int setup_10x20(void **state) { return qr_setup(state, 10, 20); }

static void run_qrt01(qr_fixture_t *fix, int imat)
{
    char type, dist;
    int kl, ku, mode, info;
    f32 anorm, cndnum;
    f32 result[2];

    slatb4("SGE", imat, fix->m, fix->n, &type, &kl, &ku, &anorm, &mode, &cndnum, &dist);

    uint64_t rng_state[4];
    rng_seed(rng_state, fix->seed);
    slatms(fix->m, fix->n, &dist, &type, fix->d, mode, cndnum, anorm,
           kl, ku, "N", fix->A, fix->lda, fix->genwork, &info, rng_state);
    assert_int_equal(info, 0);

    sqrt01(fix->m, fix->n, fix->A, fix->AF, fix->Q, fix->R, fix->lda,
           fix->tau, fix->work, fix->lwork, fix->rwork, result);

    assert_residual_ok(result[0]);
    assert_residual_ok(result[1]);
}

static void test_wellcond(void **state)
{
    qr_fixture_t *fix = *state;
    for (int imat = 1; imat <= 4; imat++) {
        fix->seed = g_seed++;
        run_qrt01(fix, imat);
    }
}

static void test_illcond(void **state)
{
    qr_fixture_t *fix = *state;
    if (fix->m < 3) { skip_test("requires m >= 3"); return; }
    for (int imat = 8; imat <= 9; imat++) {
        fix->seed = g_seed++;
        run_qrt01(fix, imat);
    }
}

static void test_scaled(void **state)
{
    qr_fixture_t *fix = *state;
    for (int imat = 10; imat <= 11; imat++) {
        fix->seed = g_seed++;
        run_qrt01(fix, imat);
    }
}

int main(void)
{
    const struct CMUnitTest tests[] = {
        /* Square matrices */
        cmocka_unit_test_setup_teardown(test_wellcond, setup_5x5, qr_teardown),
        cmocka_unit_test_setup_teardown(test_wellcond, setup_10x10, qr_teardown),
        cmocka_unit_test_setup_teardown(test_wellcond, setup_20x20, qr_teardown),
        cmocka_unit_test_setup_teardown(test_wellcond, setup_50x50, qr_teardown),
        cmocka_unit_test_setup_teardown(test_illcond, setup_10x10, qr_teardown),
        cmocka_unit_test_setup_teardown(test_illcond, setup_20x20, qr_teardown),
        cmocka_unit_test_setup_teardown(test_scaled, setup_10x10, qr_teardown),
        cmocka_unit_test_setup_teardown(test_scaled, setup_20x20, qr_teardown),
        /* Tall matrices (m > n) */
        cmocka_unit_test_setup_teardown(test_wellcond, setup_10x5, qr_teardown),
        cmocka_unit_test_setup_teardown(test_wellcond, setup_20x10, qr_teardown),
        cmocka_unit_test_setup_teardown(test_illcond, setup_20x10, qr_teardown),
        /* Wide matrices (m < n) */
        cmocka_unit_test_setup_teardown(test_wellcond, setup_5x10, qr_teardown),
        cmocka_unit_test_setup_teardown(test_wellcond, setup_10x20, qr_teardown),
        cmocka_unit_test_setup_teardown(test_illcond, setup_10x20, qr_teardown),
    };
    return cmocka_run_group_tests_name("dgeqrf", tests, NULL, NULL);
}
