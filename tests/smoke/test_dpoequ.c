/**
 * @file test_dpoequ.c
 * @brief CMocka test suite for dpoequ (row/column equilibration).
 *
 * Tests the equilibration routine dpoequ which computes scaling factors
 * S(i) = 1/sqrt(A(i,i)) for a symmetric positive definite matrix.
 *
 * Verification: Direct checks that S(i) = 1/sqrt(A(i,i)), and that
 * scond and amax are correct.
 */

#include "test_harness.h"

/* Test threshold - see LAPACK dtest.in */
#define THRESH 20.0
#include <cblas.h>

/* Routine under test */
extern void dpoequ(const int n, const double* const restrict A, const int lda,
                   double* const restrict S, double* scond, double* amax,
                   int* info);

/* Matrix generation */
extern void dlatb4(const char* path, const int imat, const int m, const int n,
                   char* type, int* kl, int* ku, double* anorm, int* mode,
                   double* cndnum, char* dist);
extern void dlatms(const int m, const int n, const char* dist,
                   uint64_t seed, const char* sym, double* d,
                   const int mode, const double cond, const double dmax,
                   const int kl, const int ku, const char* pack,
                   double* A, const int lda, double* work, int* info);

/* Utilities */
extern double dlamch(const char* cmach);

/*
 * Test fixture
 */
typedef struct {
    int n;
    int lda;
    double* A;
    double* S;
    double* d;
    double* work;
    uint64_t seed;
} dpoequ_fixture_t;

static uint64_t g_seed = 5500;

static int dpoequ_setup(void** state, int n)
{
    dpoequ_fixture_t* fix = malloc(sizeof(dpoequ_fixture_t));
    assert_non_null(fix);

    fix->n = n;
    fix->lda = n;
    fix->seed = g_seed++;

    fix->A = malloc(fix->lda * n * sizeof(double));
    fix->S = malloc(n * sizeof(double));
    fix->d = malloc(n * sizeof(double));
    fix->work = malloc(3 * n * sizeof(double));

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
 * Test that dpoequ computes correct scale factors.
 * For a well-conditioned SPD matrix, S(i) = 1/sqrt(A(i,i)).
 */
static void test_dpoequ_wellcond(void** state)
{
    dpoequ_fixture_t* fix = *state;
    double eps = dlamch("E");

    for (int imat = 1; imat <= 5; imat++) {
        fix->seed = g_seed++;
        char type, dist;
        int kl, ku, mode;
        double anorm, cndnum;
        int info;

        dlatb4("DPO", imat, fix->n, fix->n, &type, &kl, &ku, &anorm, &mode, &cndnum, &dist);

        char sym_str[2] = {type, '\0'};
        dlatms(fix->n, fix->n, &dist, fix->seed, sym_str, fix->d, mode, cndnum, anorm,
               kl, ku, "N", fix->A, fix->lda, fix->work, &info);
        assert_int_equal(info, 0);

        double scond, amax_out;
        dpoequ(fix->n, fix->A, fix->lda, fix->S, &scond, &amax_out, &info);
        assert_info_success(info);

        /* Verify S(i) = 1/sqrt(A(i,i)) */
        for (int i = 0; i < fix->n; i++) {
            double expected = 1.0 / sqrt(fix->A[i + i * fix->lda]);
            double diff = fabs(fix->S[i] - expected);
            double resid = diff / (expected * eps);
            assert_residual_ok(resid);
        }

        /* Verify amax = max(A(i,i)) */
        double amax_expected = 0.0;
        for (int i = 0; i < fix->n; i++) {
            if (fix->A[i + i * fix->lda] > amax_expected)
                amax_expected = fix->A[i + i * fix->lda];
        }
        double amax_diff = fabs(amax_out - amax_expected);
        assert_true(amax_diff <= eps * amax_expected);

        /* Verify scond is reasonable (should be > 0 for well-conditioned) */
        assert_true(scond > 0.0);
        assert_true(scond <= 1.0);
    }
}

/**
 * Test dpoequ with a matrix that has a zero diagonal element.
 * dpoequ should return info > 0 indicating the first non-positive diagonal.
 */
static void test_dpoequ_zero_diag(void** state)
{
    dpoequ_fixture_t* fix = *state;
    int info;

    /* Generate a well-conditioned matrix first */
    fix->seed = g_seed++;
    char type, dist;
    int kl, ku, mode;
    double anorm, cndnum;

    dlatb4("DPO", 4, fix->n, fix->n, &type, &kl, &ku, &anorm, &mode, &cndnum, &dist);

    char sym_str[2] = {type, '\0'};
    dlatms(fix->n, fix->n, &dist, fix->seed, sym_str, fix->d, mode, cndnum, anorm,
           kl, ku, "N", fix->A, fix->lda, fix->work, &info);
    assert_int_equal(info, 0);

    /* Set a diagonal element to zero */
    int bad_idx = fix->n / 2;
    fix->A[bad_idx + bad_idx * fix->lda] = 0.0;

    double scond, amax_out;
    dpoequ(fix->n, fix->A, fix->lda, fix->S, &scond, &amax_out, &info);

    /* info should be bad_idx + 1 (1-based index of first non-positive diagonal) */
    assert_int_equal(info, bad_idx + 1);
}

/**
 * Test dpoequ with n=1 (trivial case).
 */
static void test_dpoequ_n1(void** state)
{
    (void)state;
    double A[1] = {4.0};
    double S[1];
    double scond, amax_out;
    int info;

    dpoequ(1, A, 1, S, &scond, &amax_out, &info);
    assert_info_success(info);
    assert_true(fabs(S[0] - 0.5) < 1e-15);  /* 1/sqrt(4) = 0.5 */
    assert_true(fabs(amax_out - 4.0) < 1e-15);
    assert_true(fabs(scond - 1.0) < 1e-15);
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
