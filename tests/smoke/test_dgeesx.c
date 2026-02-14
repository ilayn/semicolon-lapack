/**
 * @file test_dgeesx.c
 * @brief Test expert Schur decomposition routine dgeesx.
 *
 * Tests the expert driver which provides:
 * - Eigenvalue ordering via SELECT
 * - Reciprocal condition numbers for eigenvalue cluster and invariant subspace
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
    f64* A;        /* Original matrix */
    f64* Acopy;    /* Copy for verification */
    f64* VS;       /* Schur vectors */
    f64* wr;       /* Real eigenvalues */
    f64* wi;       /* Imaginary eigenvalues */
    f64* work;     /* Workspace */
    int* iwork;       /* Integer workspace */
    int* bwork;       /* Boolean work for SELECT */
    uint64_t seed;
    uint64_t rng_state[4];
} dgeesx_fixture_t;

/* Forward declarations from semicolon_lapack */
typedef int (*dselect2_t)(const f64* wr, const f64* wi);

extern void dgeesx(const char* jobvs, const char* sort, dselect2_t select,
                   const char* sense, const int n, f64* A, const int lda,
                   int* sdim, f64* wr, f64* wi, f64* VS, const int ldvs,
                   f64* rconde, f64* rcondv,
                   f64* work, const int lwork,
                   int* iwork, const int liwork, int* bwork, int* info);
extern void dlacpy(const char* uplo, const int m, const int n,
                   const f64* A, const int lda, f64* B, const int ldb);
extern void dlaset(const char* uplo, const int m, const int n,
                   const f64 alpha, const f64 beta,
                   f64* A, const int lda);
extern f64 dlamch(const char* cmach);
extern f64 dlange(const char* norm, const int m, const int n,
                     const f64* A, const int lda, f64* work);

/* Selection function: select eigenvalues with negative real part */
static int select_negative_real(const f64* wr, const f64* wi)
{
    (void)wi;
    return (*wr < 0.0) ? 1 : 0;
}

/* Selection function: select eigenvalues inside unit circle */
static int select_inside_unit_circle(const f64* wr, const f64* wi)
{
    f64 magnitude = sqrt((*wr) * (*wr) + (*wi) * (*wi));
    return (magnitude < 1.0) ? 1 : 0;
}

/* Setup function parameterized by N */
static int setup_N(void** state, int n) {
    dgeesx_fixture_t* fix = malloc(sizeof(dgeesx_fixture_t));
    if (!fix) return -1;

    fix->n = n;
    fix->seed = 0xBADC0FFEULL;

    /* Allocate matrices */
    fix->A = malloc(n * n * sizeof(f64));
    fix->Acopy = malloc(n * n * sizeof(f64));
    fix->VS = malloc(n * n * sizeof(f64));
    fix->wr = malloc(n * sizeof(f64));
    fix->wi = malloc(n * sizeof(f64));
    fix->bwork = malloc(n * sizeof(int));

    /* Integer workspace for condition number computation */
    int liwork = n * n / 4;
    if (liwork < 1) liwork = 1;
    fix->iwork = malloc(liwork * sizeof(int));

    /* Workspace: generous allocation */
    int lwork = 12 * n * n;
    fix->work = malloc(lwork * sizeof(f64));

    if (!fix->A || !fix->Acopy || !fix->VS ||
        !fix->wr || !fix->wi || !fix->bwork ||
        !fix->iwork || !fix->work) {
        free(fix->A); free(fix->Acopy); free(fix->VS);
        free(fix->wr); free(fix->wi); free(fix->bwork);
        free(fix->iwork); free(fix->work);
        free(fix);
        return -1;
    }

    rng_seed(fix->rng_state, fix->seed);
    *state = fix;
    return 0;
}

static int teardown(void** state) {
    dgeesx_fixture_t* fix = *state;
    if (fix) {
        free(fix->A);
        free(fix->Acopy);
        free(fix->VS);
        free(fix->wr);
        free(fix->wi);
        free(fix->bwork);
        free(fix->iwork);
        free(fix->work);
        free(fix);
    }
    return 0;
}

/* Setup wrappers for different sizes */
static int setup_5(void** state) { return setup_N(state, 5); }
static int setup_10(void** state) { return setup_N(state, 10); }
static int setup_20(void** state) { return setup_N(state, 20); }

/**
 * Generate random test matrix.
 */
static void generate_random_matrix(int n, f64* A, int lda, f64 anorm,
                                   uint64_t state[static 4])
{
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) {
            A[i + j * lda] = anorm * rng_uniform_symmetric(state);
        }
    }
}

/**
 * Test basic Schur decomposition without ordering or condition numbers.
 */
static void test_basic(void** state)
{
    dgeesx_fixture_t* fix = *state;
    int n = fix->n;
    int lda = n;
    int sdim, info;
    f64 rconde, rcondv;
    f64 result[2];

    /* Generate random matrix */
    generate_random_matrix(n, fix->A, lda, 1.0, fix->rng_state);

    /* Keep a copy for verification */
    dlacpy(" ", n, n, fix->A, lda, fix->Acopy, lda);

    /* Compute Schur decomposition */
    int lwork = 10 * n * n;
    int liwork = n * n / 4;
    if (liwork < 1) liwork = 1;

    dgeesx("V", "N", NULL, "N", n, fix->A, lda, &sdim,
           fix->wr, fix->wi, fix->VS, lda, &rconde, &rcondv,
           fix->work, lwork, fix->iwork, liwork, NULL, &info);

    assert_info_success(info);

    /* Eigenvalues should be finite */
    for (int j = 0; j < n; j++) {
        assert_true(isfinite(fix->wr[j]));
        assert_true(isfinite(fix->wi[j]));
    }

    /* Verify Schur decomposition */
    int lwork_verify = 2 * n * n;
    dhst01(n, 1, n, fix->Acopy, lda, fix->A, lda, fix->VS, lda,
           fix->work, lwork_verify, result);

    assert_residual_ok(result[0]);
    assert_residual_ok(result[1]);
}

/**
 * Test with eigenvalue ordering.
 */
static void test_with_ordering(void** state)
{
    dgeesx_fixture_t* fix = *state;
    int n = fix->n;
    int lda = n;
    int sdim, info;
    f64 rconde, rcondv;
    f64 result[2];

    /* Generate random matrix */
    generate_random_matrix(n, fix->A, lda, 1.0, fix->rng_state);

    /* Keep a copy */
    dlacpy(" ", n, n, fix->A, lda, fix->Acopy, lda);

    /* Compute Schur decomposition with ordering */
    int lwork = 10 * n * n;
    int liwork = n * n / 4;
    if (liwork < 1) liwork = 1;

    dgeesx("V", "S", select_negative_real, "N", n, fix->A, lda, &sdim,
           fix->wr, fix->wi, fix->VS, lda, &rconde, &rcondv,
           fix->work, lwork, fix->iwork, liwork, fix->bwork, &info);

    /* Accept info = 0, n+1 (eigenvalues too close), or n+2 (roundoff after reordering) */
    if (info != 0 && info != n + 1 && info != n + 2) {
        printf("info = %d: algorithm failure\n", info);
        assert_info_success(info);
    }

    /* If ordering failed due to close eigenvalues, skip ordering verification */
    if (info == n + 1 || info == n + 2) {
        return;
    }

    /* Verify eigenvalue ordering */
    for (int j = 0; j < sdim; j++) {
        if (fix->wi[j] == 0.0) {
            assert_true(fix->wr[j] < 0.0);
        }
    }

    /* Verify Schur decomposition */
    int lwork_verify = 2 * n * n;
    dhst01(n, 1, n, fix->Acopy, lda, fix->A, lda, fix->VS, lda,
           fix->work, lwork_verify, result);

    assert_residual_ok(result[0]);
    assert_residual_ok(result[1]);
}

/**
 * Test with condition number for eigenvalue cluster.
 */
static void test_condition_eigenvalue(void** state)
{
    dgeesx_fixture_t* fix = *state;
    int n = fix->n;
    int lda = n;
    int sdim, info;
    f64 rconde, rcondv;

    /* Generate random matrix */
    generate_random_matrix(n, fix->A, lda, 1.0, fix->rng_state);

    /* Compute with condition number for eigenvalue cluster */
    int lwork = 10 * n * n;
    int liwork = n * n / 4;
    if (liwork < 1) liwork = 1;

    dgeesx("V", "S", select_negative_real, "E", n, fix->A, lda, &sdim,
           fix->wr, fix->wi, fix->VS, lda, &rconde, &rcondv,
           fix->work, lwork, fix->iwork, liwork, fix->bwork, &info);

    /* Accept info = 0, n+1 (eigenvalues too close), or n+2 (roundoff after reordering) */
    if (info != 0 && info != n + 1 && info != n + 2) {
        printf("info = %d: algorithm failure\n", info);
        assert_info_success(info);
    }

    /* If ordering failed, skip condition number verification */
    if (info == n + 1 || info == n + 2) {
        return;
    }

    /* If any eigenvalues were selected, rconde should be valid */
    if (sdim > 0 && sdim < n) {
        assert_true(rconde > 0.0);
        assert_true(rconde <= 1.0);
    }
}

/**
 * Test with condition number for invariant subspace.
 */
static void test_condition_subspace(void** state)
{
    dgeesx_fixture_t* fix = *state;
    int n = fix->n;
    int lda = n;
    int sdim, info;
    f64 rconde, rcondv;

    /* Generate random matrix */
    generate_random_matrix(n, fix->A, lda, 1.0, fix->rng_state);

    /* Compute with condition number for invariant subspace */
    int lwork = 10 * n * n;
    int liwork = n * n / 4;
    if (liwork < 1) liwork = 1;

    dgeesx("V", "S", select_negative_real, "V", n, fix->A, lda, &sdim,
           fix->wr, fix->wi, fix->VS, lda, &rconde, &rcondv,
           fix->work, lwork, fix->iwork, liwork, fix->bwork, &info);

    /* Accept info = 0, n+1 (eigenvalues too close), or n+2 (roundoff after reordering) */
    if (info != 0 && info != n + 1 && info != n + 2) {
        printf("info = %d: algorithm failure\n", info);
        assert_info_success(info);
    }

    /* If ordering failed, skip condition number verification */
    if (info == n + 1 || info == n + 2) {
        return;
    }

    /* If any eigenvalues were selected, rcondv should be positive */
    if (sdim > 0 && sdim < n) {
        assert_true(rcondv > 0.0);
        assert_true(isfinite(rcondv));
    }
}

/**
 * Test with both condition numbers.
 */
static void test_condition_both(void** state)
{
    dgeesx_fixture_t* fix = *state;
    int n = fix->n;
    int lda = n;
    int sdim, info;
    f64 rconde, rcondv;
    f64 result[2];

    /* Generate random matrix */
    generate_random_matrix(n, fix->A, lda, 1.0, fix->rng_state);
    dlacpy(" ", n, n, fix->A, lda, fix->Acopy, lda);

    /* Compute with both condition numbers */
    int lwork = 10 * n * n;
    int liwork = n * n / 4;
    if (liwork < 1) liwork = 1;

    dgeesx("V", "S", select_inside_unit_circle, "B", n, fix->A, lda, &sdim,
           fix->wr, fix->wi, fix->VS, lda, &rconde, &rcondv,
           fix->work, lwork, fix->iwork, liwork, fix->bwork, &info);

    /* Accept info = 0, n+1 (eigenvalues too close), or n+2 (roundoff after reordering) */
    if (info != 0 && info != n + 1 && info != n + 2) {
        printf("info = %d: algorithm failure\n", info);
        assert_info_success(info);
    }

    /* If ordering failed, skip the rest of the test */
    if (info == n + 1 || info == n + 2) {
        return;
    }

    /* Condition numbers should be valid if eigenvalues were selected */
    if (sdim > 0 && sdim < n) {
        assert_true(rconde > 0.0);
        assert_true(rconde <= 1.0);
        assert_true(rcondv > 0.0);
    }

    /* Verify Schur decomposition */
    int lwork_verify = 2 * n * n;
    dhst01(n, 1, n, fix->Acopy, lda, fix->A, lda, fix->VS, lda,
           fix->work, lwork_verify, result);

    assert_residual_ok(result[0]);
    assert_residual_ok(result[1]);
}

/**
 * Test workspace query.
 */
static void test_workspace_query(void** state)
{
    dgeesx_fixture_t* fix = *state;
    int n = fix->n;
    int sdim, info;
    f64 rconde, rcondv;
    f64 work_query;
    int iwork_query;

    /* Query optimal workspace */
    dgeesx("V", "S", select_negative_real, "B", n, fix->A, n, &sdim,
           fix->wr, fix->wi, fix->VS, n, &rconde, &rcondv,
           &work_query, -1, &iwork_query, -1, fix->bwork, &info);

    assert_info_success(info);
    assert_true(work_query >= (f64)(3 * n));
}

/**
 * Test with no Schur vectors.
 */
static void test_no_vectors(void** state)
{
    dgeesx_fixture_t* fix = *state;
    int n = fix->n;
    int lda = n;
    int sdim, info;
    f64 rconde, rcondv;

    /* Generate random matrix */
    generate_random_matrix(n, fix->A, lda, 1.0, fix->rng_state);

    /* Compute without Schur vectors */
    int lwork = 10 * n * n;
    int liwork = n * n / 4;
    if (liwork < 1) liwork = 1;

    dgeesx("N", "N", NULL, "N", n, fix->A, lda, &sdim,
           fix->wr, fix->wi, NULL, 1, &rconde, &rcondv,
           fix->work, lwork, fix->iwork, liwork, NULL, &info);

    assert_info_success(info);

    /* Eigenvalues should be finite */
    for (int j = 0; j < n; j++) {
        assert_true(isfinite(fix->wr[j]));
        assert_true(isfinite(fix->wi[j]));
    }
}

/**
 * Test with symmetric matrix.
 */
static void test_symmetric_matrix(void** state)
{
    dgeesx_fixture_t* fix = *state;
    int n = fix->n;
    int lda = n;
    int sdim, info;
    f64 rconde, rcondv;

    /* Generate symmetric matrix */
    for (int j = 0; j < n; j++) {
        for (int i = j; i < n; i++) {
            f64 val = rng_uniform_symmetric(fix->rng_state);
            fix->A[i + j * lda] = val;
            fix->A[j + i * lda] = val;
        }
    }

    /* Compute Schur decomposition */
    int lwork = 10 * n * n;
    int liwork = n * n / 4;
    if (liwork < 1) liwork = 1;

    dgeesx("V", "N", NULL, "N", n, fix->A, lda, &sdim,
           fix->wr, fix->wi, fix->VS, lda, &rconde, &rcondv,
           fix->work, lwork, fix->iwork, liwork, NULL, &info);

    assert_info_success(info);

    /* All eigenvalues should be real */
    for (int j = 0; j < n; j++) {
        assert_double_equal(fix->wi[j], 0.0, 1e-10);
    }
}

int main(void)
{
    const struct CMUnitTest tests_n5[] = {
        cmocka_unit_test_setup_teardown(test_basic, setup_5, teardown),
        cmocka_unit_test_setup_teardown(test_with_ordering, setup_5, teardown),
        cmocka_unit_test_setup_teardown(test_workspace_query, setup_5, teardown),
        cmocka_unit_test_setup_teardown(test_no_vectors, setup_5, teardown),
    };

    const struct CMUnitTest tests_n10[] = {
        cmocka_unit_test_setup_teardown(test_basic, setup_10, teardown),
        cmocka_unit_test_setup_teardown(test_with_ordering, setup_10, teardown),
        cmocka_unit_test_setup_teardown(test_condition_eigenvalue, setup_10, teardown),
        cmocka_unit_test_setup_teardown(test_condition_subspace, setup_10, teardown),
        cmocka_unit_test_setup_teardown(test_condition_both, setup_10, teardown),
        cmocka_unit_test_setup_teardown(test_symmetric_matrix, setup_10, teardown),
    };

    const struct CMUnitTest tests_n20[] = {
        cmocka_unit_test_setup_teardown(test_basic, setup_20, teardown),
        cmocka_unit_test_setup_teardown(test_condition_both, setup_20, teardown),
    };

    int result = 0;
    result += cmocka_run_group_tests_name("dgeesx_n5", tests_n5, NULL, NULL);
    result += cmocka_run_group_tests_name("dgeesx_n10", tests_n10, NULL, NULL);
    result += cmocka_run_group_tests_name("dgeesx_n20", tests_n20, NULL, NULL);

    return result;
}
