/**
 * @file test_dgees.c
 * @brief Test Schur decomposition routine dgees.
 *
 * Tests based on LAPACK TESTING/EIG/dchkhs.f and ddrves.f, adapted to CMocka.
 * Verifies:
 *   | A - VS*T*VS' | / ( |A| n ulp )
 *   | I - VS*VS' | / ( n ulp )
 *   Eigenvalue ordering when using SELECT
 */

#include "test_harness.h"

/* Test threshold - matches LAPACK dchkhs.f */
#define THRESH 30.0

#include "testutils/verify.h"
#include "testutils/test_rng.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <cblas.h>

/* Test fixture */
typedef struct {
    int n;
    double* A;       /* Original matrix */
    double* Acopy;   /* Copy for verification */
    double* VS;      /* Schur vectors */
    double* wr;      /* Real eigenvalues */
    double* wi;      /* Imaginary eigenvalues */
    double* work;    /* Workspace */
    int* bwork;      /* Boolean work for SELECT */
    uint64_t seed;
    uint64_t rng_state[4];
} dgees_fixture_t;

/* Forward declarations from semicolon_lapack */
typedef int (*dselect2_t)(const double* wr, const double* wi);

extern void dgees(const char* jobvs, const char* sort, dselect2_t select,
                  const int n, double* A, const int lda, int* sdim,
                  double* wr, double* wi, double* VS, const int ldvs,
                  double* work, const int lwork, int* bwork, int* info);
extern void dlacpy(const char* uplo, const int m, const int n,
                   const double* A, const int lda, double* B, const int ldb);
extern void dlaset(const char* uplo, const int m, const int n,
                   const double alpha, const double beta,
                   double* A, const int lda);
extern double dlamch(const char* cmach);
extern double dlange(const char* norm, const int m, const int n,
                     const double* A, const int lda, double* work);

/* Selection function: select eigenvalues with negative real part */
static int select_negative_real(const double* wr, const double* wi)
{
    (void)wi;  /* unused */
    return (*wr < 0.0) ? 1 : 0;
}


/* Setup function parameterized by N */
static int setup_N(void** state, int n) {
    dgees_fixture_t* fix = malloc(sizeof(dgees_fixture_t));
    if (!fix) return -1;

    fix->n = n;
    fix->seed = 0xCAFEBABEULL;

    /* Allocate matrices */
    fix->A = malloc(n * n * sizeof(double));
    fix->Acopy = malloc(n * n * sizeof(double));
    fix->VS = malloc(n * n * sizeof(double));
    fix->wr = malloc(n * sizeof(double));
    fix->wi = malloc(n * sizeof(double));
    fix->bwork = malloc(n * sizeof(int));

    /* Workspace: generous allocation */
    int lwork = 10 * n * n;
    fix->work = malloc(lwork * sizeof(double));

    if (!fix->A || !fix->Acopy || !fix->VS ||
        !fix->wr || !fix->wi || !fix->bwork || !fix->work) {
        free(fix->A); free(fix->Acopy); free(fix->VS);
        free(fix->wr); free(fix->wi); free(fix->bwork); free(fix->work);
        free(fix);
        return -1;
    }

    rng_seed(fix->rng_state, fix->seed);
    *state = fix;
    return 0;
}

static int teardown(void** state) {
    dgees_fixture_t* fix = *state;
    if (fix) {
        free(fix->A);
        free(fix->Acopy);
        free(fix->VS);
        free(fix->wr);
        free(fix->wi);
        free(fix->bwork);
        free(fix->work);
        free(fix);
    }
    return 0;
}

/* Setup wrappers for different sizes */
static int setup_5(void** state) { return setup_N(state, 5); }
static int setup_10(void** state) { return setup_N(state, 10); }
static int setup_20(void** state) { return setup_N(state, 20); }
static int setup_32(void** state) { return setup_N(state, 32); }

/**
 * Generate random test matrix.
 */
static void generate_random_matrix(int n, double* A, int lda, double anorm,
                                   uint64_t state[static 4])
{
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) {
            A[i + j * lda] = anorm * rng_uniform_symmetric(state);
        }
    }
}

/**
 * Test Schur decomposition with Schur vectors.
 */
static void test_schur_with_vectors(dgees_fixture_t* fix)
{
    int n = fix->n;
    int lda = n;
    int sdim, info;
    double result[2];

    /* Generate random matrix */
    generate_random_matrix(n, fix->A, lda, 1.0, fix->rng_state);

    /* Keep a copy for verification */
    dlacpy(" ", n, n, fix->A, lda, fix->Acopy, lda);

    /* Compute Schur decomposition */
    int lwork = 8 * n * n;
    dgees("V", "N", NULL, n, fix->A, lda, &sdim,
          fix->wr, fix->wi, fix->VS, lda, fix->work, lwork, NULL, &info);

    assert_info_success(info);

    /* Eigenvalues should be finite */
    for (int j = 0; j < n; j++) {
        assert_true(isfinite(fix->wr[j]));
        assert_true(isfinite(fix->wi[j]));
    }

    /* Test: | A - VS*T*VS' | / ( |A| n ulp ) and | I - VS*VS' | / ( n ulp ) */
    /* Here A (in fix->A) now contains T (the Schur form) */
    int lwork_verify = 2 * n * n;
    dhst01(n, 1, n, fix->Acopy, lda, fix->A, lda, fix->VS, lda,
           fix->work, lwork_verify, result);

    assert_residual_ok(result[0]);  /* Schur decomposition accuracy */
    assert_residual_ok(result[1]);  /* VS orthogonality */

    /* Verify that T is quasi-triangular (2x2 blocks on diagonal for complex eigenvalues) */
    double ulp = dlamch("P");
    for (int j = 0; j < n - 1; j++) {
        /* Elements below subdiagonal should be zero */
        for (int i = j + 2; i < n; i++) {
            assert_double_equal(fix->A[i + j * lda], 0.0, ulp * 100);
        }
    }
}

/**
 * Test Schur decomposition with eigenvalue ordering.
 *
 * Note: LAPACK's dgees may return info = n+1 if eigenvalues are too close
 * to reorder, or info = n+2 if roundoff changed values after reordering.
 * These are acceptable outcomes for random matrices per LAPACK test methodology.
 */
static void test_schur_with_ordering(dgees_fixture_t* fix)
{
    int n = fix->n;
    int lda = n;
    int sdim, info;

    /* Generate random matrix */
    generate_random_matrix(n, fix->A, lda, 1.0, fix->rng_state);

    /* Keep a copy */
    dlacpy(" ", n, n, fix->A, lda, fix->Acopy, lda);

    /* Compute Schur decomposition with ordering */
    int lwork = 8 * n * n;
    dgees("V", "S", select_negative_real, n, fix->A, lda, &sdim,
          fix->wr, fix->wi, fix->VS, lda, fix->work, lwork, fix->bwork, &info);

    /* Accept info = 0 (success), info = n+1 (eigenvalues too close to reorder),
     * or info = n+2 (roundoff changed values after reordering).
     * Only info > 0 && info <= n (QR failed) or info < 0 (bad args) are errors. */
    if (info != 0 && info != n + 1 && info != n + 2) {
        printf("info = %d: algorithm failure\n", info);
        assert_info_success(info);
    }

    /* If info = n+1 or n+2, ordering was incomplete but computation succeeded */
    if (info == n + 1 || info == n + 2) {
        /* Skip ordering verification - eigenvalues couldn't be reordered */
        return;
    }

    /* Verify eigenvalue ordering: first sdim eigenvalues should have wr < 0 */
    for (int j = 0; j < sdim; j++) {
        if (fix->wi[j] == 0.0) {
            /* Real eigenvalue */
            assert_true(fix->wr[j] < 0.0);
        } else {
            /* Complex conjugate pair - check the pair */
            if (j + 1 < sdim && fix->wi[j] != 0.0 && fix->wi[j + 1] == -fix->wi[j]) {
                /* Both eigenvalues of the pair should have same real part */
                assert_true(fix->wr[j] < 0.0);
            }
        }
    }

    /* Remaining eigenvalues should not satisfy the selection criterion */
    for (int j = sdim; j < n; j++) {
        if (fix->wi[j] == 0.0) {
            assert_true(fix->wr[j] >= 0.0);
        }
    }
}

/**
 * Test eigenvalue-only computation (no Schur vectors).
 */
static void test_eigenvalues_only(void** state)
{
    dgees_fixture_t* fix = *state;
    int n = fix->n;
    int lda = n;
    int sdim, info;

    /* Generate random matrix */
    generate_random_matrix(n, fix->A, lda, 1.0, fix->rng_state);

    /* Compute eigenvalues only */
    int lwork = 8 * n * n;
    dgees("N", "N", NULL, n, fix->A, lda, &sdim,
          fix->wr, fix->wi, NULL, 1, fix->work, lwork, NULL, &info);

    assert_info_success(info);

    /* Eigenvalues should be finite */
    for (int j = 0; j < n; j++) {
        assert_true(isfinite(fix->wr[j]));
        assert_true(isfinite(fix->wi[j]));
    }
}

/**
 * Test workspace query.
 */
static void test_workspace_query(void** state)
{
    dgees_fixture_t* fix = *state;
    int n = fix->n;
    int sdim, info;
    double work_query;

    /* Query optimal workspace */
    dgees("V", "S", select_negative_real, n, fix->A, n, &sdim,
          fix->wr, fix->wi, fix->VS, n, &work_query, -1, fix->bwork, &info);

    assert_info_success(info);
    assert_true(work_query >= (double)(3 * n));
}

/**
 * Test with symmetric matrix (real eigenvalues, diagonal Schur form).
 */
static void test_symmetric_matrix(void** state)
{
    dgees_fixture_t* fix = *state;
    int n = fix->n;
    int lda = n;
    int sdim, info;
    double result[2];

    /* Generate symmetric random matrix */
    for (int j = 0; j < n; j++) {
        for (int i = j; i < n; i++) {
            double val = rng_uniform_symmetric(fix->rng_state);
            fix->A[i + j * lda] = val;
            fix->A[j + i * lda] = val;
        }
    }

    /* Keep a copy */
    dlacpy(" ", n, n, fix->A, lda, fix->Acopy, lda);

    /* Compute Schur decomposition */
    int lwork = 8 * n * n;
    dgees("V", "N", NULL, n, fix->A, lda, &sdim,
          fix->wr, fix->wi, fix->VS, lda, fix->work, lwork, NULL, &info);

    assert_info_success(info);

    /* For symmetric matrix, all eigenvalues should be real */
    for (int j = 0; j < n; j++) {
        assert_double_equal(fix->wi[j], 0.0, 1e-10);
    }

    /* Schur form should be diagonal (since all eigenvalues are real) */
    double ulp = dlamch("P");
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) {
            if (i != j) {
                assert_double_equal(fix->A[i + j * lda], 0.0, ulp * 1000);
            }
        }
    }

    /* Verify decomposition */
    int lwork_verify = 2 * n * n;
    dhst01(n, 1, n, fix->Acopy, lda, fix->A, lda, fix->VS, lda,
           fix->work, lwork_verify, result);

    assert_residual_ok(result[0]);
    assert_residual_ok(result[1]);
}

/**
 * Test with identity matrix.
 */
static void test_identity_matrix(void** state)
{
    dgees_fixture_t* fix = *state;
    int n = fix->n;
    int lda = n;
    int sdim, info;

    const double ZERO = 0.0;
    const double ONE = 1.0;

    /* Create identity matrix */
    dlaset("F", n, n, ZERO, ONE, fix->A, lda);

    /* Compute Schur decomposition */
    int lwork = 8 * n * n;
    dgees("V", "N", NULL, n, fix->A, lda, &sdim,
          fix->wr, fix->wi, fix->VS, lda, fix->work, lwork, NULL, &info);

    assert_info_success(info);

    /* All eigenvalues should be 1 */
    for (int j = 0; j < n; j++) {
        assert_double_equal(fix->wr[j], ONE, 1e-10);
        assert_double_equal(fix->wi[j], ZERO, 1e-10);
    }

    /* VS should be identity (or permutation of identity) */
    /* At minimum, it should be orthogonal */
    double resid;
    dort01("C", n, n, fix->VS, lda, fix->work, n * n, &resid);
    assert_residual_ok(resid);
}

/* Test wrappers */
static void test_vectors_n5(void** state) { test_schur_with_vectors(*state); }
static void test_vectors_n10(void** state) { test_schur_with_vectors(*state); }
static void test_vectors_n20(void** state) { test_schur_with_vectors(*state); }
static void test_vectors_n32(void** state) { test_schur_with_vectors(*state); }

static void test_ordering_n10(void** state) { test_schur_with_ordering(*state); }
static void test_ordering_n20(void** state) { test_schur_with_ordering(*state); }

int main(void)
{
    const struct CMUnitTest tests_n5[] = {
        cmocka_unit_test_setup_teardown(test_vectors_n5, setup_5, teardown),
        cmocka_unit_test_setup_teardown(test_eigenvalues_only, setup_5, teardown),
        cmocka_unit_test_setup_teardown(test_workspace_query, setup_5, teardown),
        cmocka_unit_test_setup_teardown(test_identity_matrix, setup_5, teardown),
    };

    const struct CMUnitTest tests_n10[] = {
        cmocka_unit_test_setup_teardown(test_vectors_n10, setup_10, teardown),
        cmocka_unit_test_setup_teardown(test_ordering_n10, setup_10, teardown),
        cmocka_unit_test_setup_teardown(test_symmetric_matrix, setup_10, teardown),
    };

    const struct CMUnitTest tests_n20[] = {
        cmocka_unit_test_setup_teardown(test_vectors_n20, setup_20, teardown),
        cmocka_unit_test_setup_teardown(test_ordering_n20, setup_20, teardown),
    };

    const struct CMUnitTest tests_n32[] = {
        cmocka_unit_test_setup_teardown(test_vectors_n32, setup_32, teardown),
    };

    int result = 0;
    result += cmocka_run_group_tests_name("dgees_n5", tests_n5, NULL, NULL);
    result += cmocka_run_group_tests_name("dgees_n10", tests_n10, NULL, NULL);
    result += cmocka_run_group_tests_name("dgees_n20", tests_n20, NULL, NULL);
    result += cmocka_run_group_tests_name("dgees_n32", tests_n32, NULL, NULL);

    return result;
}
