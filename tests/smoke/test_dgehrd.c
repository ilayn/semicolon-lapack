/**
 * @file test_dgehrd.c
 * @brief Test Hessenberg reduction routine dgehrd.
 *
 * Tests based on LAPACK TESTING/EIG/dchkhs.f, adapted to CMocka framework.
 * Verifies:
 *   (1) | A - Q*H*Q' | / ( |A| n ulp )
 *   (2) | I - Q*Q' | / ( n ulp )
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

/* Maximum dimension for static workspace */
#define NMAX 64

/* Test fixture */
typedef struct {
    int n;
    double* A;       /* Original matrix */
    double* H;       /* Hessenberg result */
    double* Q;       /* Orthogonal matrix */
    double* work;    /* Workspace */
    double* tau;     /* Householder reflectors */
    uint64_t seed;
} dgehrd_fixture_t;

/* Forward declarations from semicolon_lapack */
extern void dgehrd(const int n, const int ilo, const int ihi,
                   double* A, const int lda, double* tau,
                   double* work, const int lwork, int* info);
extern void dorghr(const int n, const int ilo, const int ihi,
                   double* A, const int lda, const double* tau,
                   double* work, const int lwork, int* info);
extern void dlacpy(const char* uplo, const int m, const int n,
                   const double* A, const int lda, double* B, const int ldb);
extern void dlaset(const char* uplo, const int m, const int n,
                   const double alpha, const double beta,
                   double* A, const int lda);
extern double dlamch(const char* cmach);
extern double dlange(const char* norm, const int m, const int n,
                     const double* A, const int lda, double* work);

/* Setup function parameterized by N */
static int setup_N(void** state, int n) {
    dgehrd_fixture_t* fix = malloc(sizeof(dgehrd_fixture_t));
    if (!fix) return -1;

    fix->n = n;
    fix->seed = 0x12345678ULL;

    /* Allocate matrices */
    fix->A = malloc(n * n * sizeof(double));
    fix->H = malloc(n * n * sizeof(double));
    fix->Q = malloc(n * n * sizeof(double));
    fix->tau = malloc(n * sizeof(double));

    /* Workspace: need at least 2*n*n for verification + lwork for routines */
    int lwork = 4 * n * n + 2 * n;
    fix->work = malloc(lwork * sizeof(double));

    if (!fix->A || !fix->H || !fix->Q || !fix->tau || !fix->work) {
        free(fix->A); free(fix->H); free(fix->Q);
        free(fix->tau); free(fix->work);
        free(fix);
        return -1;
    }

    rng_seed(fix->seed);
    *state = fix;
    return 0;
}

static int teardown(void** state) {
    dgehrd_fixture_t* fix = *state;
    if (fix) {
        free(fix->A);
        free(fix->H);
        free(fix->Q);
        free(fix->tau);
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
 * Generate test matrix according to LAPACK test methodology.
 *
 * @param itype  Matrix type (1-21 as in dchkhs.f)
 * @param n      Matrix dimension
 * @param A      Output matrix
 * @param lda    Leading dimension
 * @param anorm  Desired matrix norm
 */
static void generate_test_matrix(int itype, int n, double* A, int lda, double anorm)
{
    const double ZERO = 0.0;
    const double ONE = 1.0;

    double ulp = dlamch("P");
    (void)ulp;  /* Used in some matrix types via dlatm4 */

    /* Initialize to zero */
    dlaset("F", n, n, ZERO, ZERO, A, lda);

    switch (itype) {
        case 1:
            /* Zero matrix */
            break;

        case 2:
            /* Identity matrix scaled by anorm */
            for (int j = 0; j < n; j++) {
                A[j + j * lda] = anorm;
            }
            break;

        case 3:
            /* Jordan block */
            for (int j = 0; j < n; j++) {
                A[j + j * lda] = anorm;
                if (j > 0) {
                    A[j + (j - 1) * lda] = ONE;
                }
            }
            break;

        case 4:
        case 5:
        case 6:
            /* Diagonal matrix with specified eigenvalue distribution */
            /* For simplicity, use dlatm4 for eigenvalue test matrices */
            {
                int mode = (itype == 4) ? 4 : (itype == 5) ? 3 : 1;
                dlatm4(mode, n, 0, 0, 1, anorm, ulp, ZERO, 2, A, lda);
            }
            break;

        case 7:
        case 8:
            /* Scaled diagonal matrices */
            {
                double scale = (itype == 7) ? sqrt(dlamch("O")) * ulp / (double)n
                                            : sqrt(dlamch("U")) * (double)n / ulp;
                dlatm4(4, n, 0, 0, 1, scale, ulp, ZERO, 2, A, lda);
            }
            break;

        default:
            /* General random matrix for types 9-21 */
            /* Generate random entries */
            for (int j = 0; j < n; j++) {
                for (int i = 0; i < n; i++) {
                    A[i + j * lda] = anorm * rng_uniform_symmetric();
                }
            }
            break;
    }
}

/**
 * Test Hessenberg reduction for a specific matrix type.
 */
static void test_hessenberg_reduction(dgehrd_fixture_t* fix, int itype)
{
    int n = fix->n;
    int lda = n;
    int info;
    double result[2];

    const double ONE = 1.0;
    double anorm = ONE;

    /* Generate test matrix */
    generate_test_matrix(itype, n, fix->A, lda, anorm);

    /* Copy A to H before reduction */
    dlacpy(" ", n, n, fix->A, lda, fix->H, lda);

    /* Perform Hessenberg reduction: H = Q' * A * Q */
    int ilo = 0;      /* 0-based for semicolon-lapack */
    int ihi = n - 1;
    int lwork = n * n;  /* Generous workspace */

    dgehrd(n, ilo, ihi, fix->H, lda, fix->tau, fix->work, lwork, &info);
    assert_info_success(info);

    /* Extract Q from the Householder reflectors stored in H */
    /* Copy lower triangular part (Householder vectors) to Q */
    dlacpy(" ", n, n, fix->H, lda, fix->Q, lda);

    /* Zero out the lower triangular part of H (it contains Householder vectors) */
    for (int j = 0; j < n - 1; j++) {
        for (int i = j + 2; i < n; i++) {
            fix->H[i + j * lda] = 0.0;
        }
    }

    /* Generate the orthogonal matrix Q using dorghr */
    dorghr(n, ilo, ihi, fix->Q, lda, fix->tau, fix->work, lwork, &info);
    assert_info_success(info);

    /* Test: | A - Q*H*Q' | / ( |A| n ulp ) and | I - Q*Q' | / ( n ulp ) */
    int lwork_verify = 2 * n * n;
    dhst01(n, ilo, ihi, fix->A, lda, fix->H, lda, fix->Q, lda,
           fix->work, lwork_verify, result);

    /* Check that Hessenberg reduction is accurate */
    if (itype != 1) {  /* Skip zero matrix which gives 0/0 */
        assert_residual_ok(result[0]);
    }

    /* Check that Q is orthogonal */
    assert_residual_ok(result[1]);
}

/* Test functions for different matrix types */
static void test_zero_matrix(void** state)
{
    dgehrd_fixture_t* fix = *state;
    test_hessenberg_reduction(fix, 1);  /* Zero matrix */
}

static void test_identity_matrix(void** state)
{
    dgehrd_fixture_t* fix = *state;
    test_hessenberg_reduction(fix, 2);  /* Identity */
}

static void test_jordan_block(void** state)
{
    dgehrd_fixture_t* fix = *state;
    test_hessenberg_reduction(fix, 3);  /* Jordan block */
}

static void test_diagonal_arithmetic(void** state)
{
    dgehrd_fixture_t* fix = *state;
    test_hessenberg_reduction(fix, 4);  /* Arithmetic diagonal */
}

static void test_diagonal_geometric(void** state)
{
    dgehrd_fixture_t* fix = *state;
    test_hessenberg_reduction(fix, 5);  /* Geometric diagonal */
}

static void test_diagonal_clustered(void** state)
{
    dgehrd_fixture_t* fix = *state;
    test_hessenberg_reduction(fix, 6);  /* Clustered diagonal */
}

static void test_random_general(void** state)
{
    dgehrd_fixture_t* fix = *state;
    test_hessenberg_reduction(fix, 19);  /* Random general matrix */
}

/**
 * Test workspace query functionality.
 */
static void test_workspace_query(void** state)
{
    dgehrd_fixture_t* fix = *state;
    int n = fix->n;
    int info;
    double work_query;

    /* Query optimal workspace */
    dgehrd(n, 0, n - 1, fix->H, n, fix->tau, &work_query, -1, &info);
    assert_info_success(info);
    assert_true(work_query >= (double)n);

    /* Also test dorghr workspace query */
    dlacpy(" ", n, n, fix->A, n, fix->H, n);
    dgehrd(n, 0, n - 1, fix->H, n, fix->tau, fix->work, n * n, &info);
    assert_info_success(info);

    dlacpy(" ", n, n, fix->H, n, fix->Q, n);
    dorghr(n, 0, n - 1, fix->Q, n, fix->tau, &work_query, -1, &info);
    assert_info_success(info);
    assert_true(work_query >= (double)n);
}

/**
 * Test with ilo and ihi different from full range.
 * This tests the case where the matrix is already partially reduced
 * (as would come from dgebal with 'P' or 'B' balancing).
 *
 * IMPORTANT: For partial range reduction to satisfy A = Q*H*Q', the
 * original matrix A must already be upper triangular in rows/columns
 * outside [ilo, ihi]. This is the standard use case - dgebal permutes
 * the matrix so that already-isolated eigenvalues are moved to corners.
 *
 * We generate such a matrix and verify the full Hessenberg relationship.
 */
static void test_partial_range(void** state)
{
    dgehrd_fixture_t* fix = *state;
    int n = fix->n;
    if (n < 4) {
        skip_test("n too small for partial range test");
        return;
    }

    int lda = n;
    int info;
    double result[2];

    const double ZERO = 0.0;

    /* Reduce only the middle portion (0-based: from column 1 to n-2) */
    int ilo = 1;
    int ihi = n - 2;

    /* Generate a matrix that is already upper triangular outside [ilo, ihi].
     * This simulates what dgebal produces when eigenvalues are isolated at corners.
     *
     * Structure:
     *   [d  a  a  a  a  a]    Row 0 is upper triangular
     *   [0  x  x  x  x  a]    Rows ilo:ihi are active (x = random)
     *   [0  x  x  x  x  a]
     *   [0  x  x  x  x  a]
     *   [0  x  x  x  x  a]
     *   [0  0  0  0  0  d]    Row n-1 is already diagonal
     */
    dlaset("F", n, n, ZERO, ZERO, fix->A, lda);

    /* Set diagonal for rows 0 and n-1 (isolated eigenvalues) */
    fix->A[0 + 0 * lda] = rng_uniform_symmetric() + 2.0;
    fix->A[(n-1) + (n-1) * lda] = rng_uniform_symmetric() + 2.0;

    /* Upper part of row 0 (can be non-zero) */
    for (int j = 1; j < n; j++) {
        fix->A[0 + j * lda] = rng_uniform_symmetric();
    }

    /* Active submatrix: rows/columns ilo:ihi */
    for (int j = ilo; j <= ihi; j++) {
        for (int i = ilo; i <= ihi; i++) {
            fix->A[i + j * lda] = rng_uniform_symmetric();
        }
    }

    /* Elements connecting active block to last row/column
     * Last column can have entries in rows ilo:ihi */
    for (int i = ilo; i <= ihi; i++) {
        fix->A[i + (n - 1) * lda] = rng_uniform_symmetric();
    }

    /* Copy to H */
    dlacpy(" ", n, n, fix->A, lda, fix->H, lda);

    /* Reduce the active portion */
    dgehrd(n, ilo, ihi, fix->H, lda, fix->tau, fix->work, n * n, &info);
    assert_info_success(info);

    /* Copy H (with Householder vectors) to Q for generating orthogonal matrix */
    dlacpy(" ", n, n, fix->H, lda, fix->Q, lda);

    /* Zero out the lower triangular part of H (Householder vectors storage) */
    for (int j = 0; j < n - 1; j++) {
        for (int i = j + 2; i < n; i++) {
            fix->H[i + j * lda] = 0.0;
        }
    }

    /* Generate Q from Householder vectors using dorghr */
    dorghr(n, ilo, ihi, fix->Q, lda, fix->tau, fix->work, n * n, &info);
    assert_info_success(info);

    /* Verify using dhst01: | A - Q*H*Q' | / (|A| n ulp) and | I - Q*Q' | / (n ulp) */
    int lwork_verify = 2 * n * n;
    dhst01(n, ilo, ihi, fix->A, lda, fix->H, lda, fix->Q, lda,
           fix->work, lwork_verify, result);

    /* Check that partial Hessenberg reduction is accurate */
    assert_residual_ok(result[0]);

    /* Check that Q is orthogonal */
    assert_residual_ok(result[1]);
}

int main(void)
{
    const struct CMUnitTest tests_n5[] = {
        cmocka_unit_test_setup_teardown(test_zero_matrix, setup_5, teardown),
        cmocka_unit_test_setup_teardown(test_identity_matrix, setup_5, teardown),
        cmocka_unit_test_setup_teardown(test_jordan_block, setup_5, teardown),
        cmocka_unit_test_setup_teardown(test_diagonal_arithmetic, setup_5, teardown),
        cmocka_unit_test_setup_teardown(test_random_general, setup_5, teardown),
        cmocka_unit_test_setup_teardown(test_workspace_query, setup_5, teardown),
    };

    const struct CMUnitTest tests_n10[] = {
        cmocka_unit_test_setup_teardown(test_zero_matrix, setup_10, teardown),
        cmocka_unit_test_setup_teardown(test_identity_matrix, setup_10, teardown),
        cmocka_unit_test_setup_teardown(test_jordan_block, setup_10, teardown),
        cmocka_unit_test_setup_teardown(test_diagonal_geometric, setup_10, teardown),
        cmocka_unit_test_setup_teardown(test_random_general, setup_10, teardown),
        cmocka_unit_test_setup_teardown(test_partial_range, setup_10, teardown),
    };

    const struct CMUnitTest tests_n20[] = {
        cmocka_unit_test_setup_teardown(test_identity_matrix, setup_20, teardown),
        cmocka_unit_test_setup_teardown(test_diagonal_clustered, setup_20, teardown),
        cmocka_unit_test_setup_teardown(test_random_general, setup_20, teardown),
    };

    const struct CMUnitTest tests_n32[] = {
        cmocka_unit_test_setup_teardown(test_random_general, setup_32, teardown),
        cmocka_unit_test_setup_teardown(test_partial_range, setup_32, teardown),
    };

    int result = 0;
    result += cmocka_run_group_tests_name("dgehrd_n5", tests_n5, NULL, NULL);
    result += cmocka_run_group_tests_name("dgehrd_n10", tests_n10, NULL, NULL);
    result += cmocka_run_group_tests_name("dgehrd_n20", tests_n20, NULL, NULL);
    result += cmocka_run_group_tests_name("dgehrd_n32", tests_n32, NULL, NULL);

    return result;
}
