/**
 * @file test_spoequ.c
 * @brief CMocka test suite for spoequ (row/column equilibration).
 *
 * Tests the equilibration routine spoequ which computes scaling factors
 * S(i) = 1/sqrt(A(i,i)) for a symmetric positive definite matrix.
 *
 * Verification: Direct checks that S(i) = 1/sqrt(A(i,i)), and that
 * scond and amax are correct.
 */

#include "test_harness.h"
#include "verify.h"
#include "test_rng.h"

/* Test threshold - see LAPACK dtest.in */
#define THRESH 20.0f
/* Routine under test */
/* Utilities */
/*
 * Test fixture
 */
typedef struct {
    INT n;
    INT lda;
    f32* A;
    f32* S;
    f32* d;
    f32* work;
    uint64_t seed;
} dpoequ_fixture_t;

static uint64_t g_seed = 5500;

static int dpoequ_setup(void** state, INT n)
{
    dpoequ_fixture_t* fix = malloc(sizeof(dpoequ_fixture_t));
    assert_non_null(fix);

    fix->n = n;
    fix->lda = n;
    fix->seed = g_seed++;

    fix->A = malloc(fix->lda * n * sizeof(f32));
    fix->S = malloc(n * sizeof(f32));
    fix->d = malloc(n * sizeof(f32));
    fix->work = malloc(3 * n * sizeof(f32));

    assert_non_null(fix->A);
    assert_non_null(fix->S);
    assert_non_null(fix->d);
    assert_non_null(fix->work);

    *state = fix;
    return 0;
}

static int dpoequ_teardown(void** state)
{
    dpoequ_fixture_t* fix = *state;
    if (fix) {
        free(fix->A);
        free(fix->S);
        free(fix->d);
        free(fix->work);
        free(fix);
    }
    return 0;
}

static int setup_5(void** state) { return dpoequ_setup(state, 5); }
static int setup_10(void** state) { return dpoequ_setup(state, 10); }

/**
 * Test that spoequ computes correct scale factors.
 * For a well-conditioned SPD matrix, S(i) = 1/sqrt(A(i,i)).
 */
static void test_dpoequ_wellcond(void** state)
{
    dpoequ_fixture_t* fix = *state;
    f32 eps = slamch("E");

    for (INT imat = 1; imat <= 5; imat++) {
        fix->seed = g_seed++;
        char type, dist;
        INT kl, ku, mode;
        f32 anorm, cndnum;
        INT info;

        slatb4("SPO", imat, fix->n, fix->n, &type, &kl, &ku, &anorm, &mode, &cndnum, &dist);

        char sym_str[2] = {type, '\0'};
        uint64_t rng_state[4];
        rng_seed(rng_state, fix->seed);
        slatms(fix->n, fix->n, &dist, sym_str, fix->d, mode, cndnum, anorm,
               kl, ku, "N", fix->A, fix->lda, fix->work, &info, rng_state);
        assert_int_equal(info, 0);

        f32 scond, amax_out;
        spoequ(fix->n, fix->A, fix->lda, fix->S, &scond, &amax_out, &info);
        assert_info_success(info);

        /* Verify S(i) = 1/sqrt(A(i,i)) */
        for (INT i = 0; i < fix->n; i++) {
            f32 expected = 1.0f / sqrtf(fix->A[i + i * fix->lda]);
            f32 diff = fabsf(fix->S[i] - expected);
            f32 resid = diff / (expected * eps);
            assert_residual_ok(resid);
        }

        /* Verify amax = max(A(i,i)) */
        f32 amax_expected = 0.0f;
        for (INT i = 0; i < fix->n; i++) {
            if (fix->A[i + i * fix->lda] > amax_expected)
                amax_expected = fix->A[i + i * fix->lda];
        }
        f32 amax_diff = fabsf(amax_out - amax_expected);
        assert_true(amax_diff <= eps * amax_expected);

        /* Verify scond is reasonable (should be > 0 for well-conditioned) */
        assert_true(scond > 0.0f);
        assert_true(scond <= 1.0f);
    }
}

/**
 * Test spoequ with a matrix that has a zero diagonal element.
 * spoequ should return info > 0 indicating the first non-positive diagonal.
 */
static void test_dpoequ_zero_diag(void** state)
{
    dpoequ_fixture_t* fix = *state;
    INT info;

    /* Generate a well-conditioned matrix first */
    fix->seed = g_seed++;
    char type, dist;
    INT kl, ku, mode;
    f32 anorm, cndnum;

    slatb4("SPO", 4, fix->n, fix->n, &type, &kl, &ku, &anorm, &mode, &cndnum, &dist);

    char sym_str[2] = {type, '\0'};
    uint64_t rng_state[4];
    rng_seed(rng_state, fix->seed);
    slatms(fix->n, fix->n, &dist, sym_str, fix->d, mode, cndnum, anorm,
           kl, ku, "N", fix->A, fix->lda, fix->work, &info, rng_state);
    assert_int_equal(info, 0);

    /* Set a diagonal element to zero */
    INT bad_idx = fix->n / 2;
    fix->A[bad_idx + bad_idx * fix->lda] = 0.0f;

    f32 scond, amax_out;
    spoequ(fix->n, fix->A, fix->lda, fix->S, &scond, &amax_out, &info);

    /* info should be bad_idx + 1 (1-based index of first non-positive diagonal) */
    assert_int_equal(info, bad_idx + 1);
}

/**
 * Test spoequ with n=1 (trivial case).
 */
static void test_dpoequ_n1(void** state)
{
    (void)state;
    f32 A[1] = {4.0f};
    f32 S[1];
    f32 scond, amax_out;
    INT info;

    spoequ(1, A, 1, S, &scond, &amax_out, &info);
    assert_info_success(info);
    assert_true((double)fabsf(S[0] - 0.5f) < 1e-15);  /* 1/sqrt(4) = 0.5 */
    assert_true((double)fabsf(amax_out - 4.0f) < 1e-15);
    assert_true((double)fabsf(scond - 1.0f) < 1e-15);
}

int main(void)
{
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(test_dpoequ_wellcond, setup_5, dpoequ_teardown),
        cmocka_unit_test_setup_teardown(test_dpoequ_wellcond, setup_10, dpoequ_teardown),
        cmocka_unit_test_setup_teardown(test_dpoequ_zero_diag, setup_5, dpoequ_teardown),
        cmocka_unit_test_setup_teardown(test_dpoequ_zero_diag, setup_10, dpoequ_teardown),
        cmocka_unit_test(test_dpoequ_n1),
    };

    return cmocka_run_group_tests_name("dpoequ", tests, NULL, NULL);
}
