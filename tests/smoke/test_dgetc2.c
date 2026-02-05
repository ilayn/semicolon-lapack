/**
 * @file test_dgetc2.c
 * @brief CMocka test suite for dgetc2 (LU factorization with complete pivoting).
 *
 * Tests the LU factorization A = P * L * U * Q where P and Q are
 * permutation matrices.
 *
 * Verification: Compute ||P*L*U*Q - A|| / (n * ||A|| * eps)
 */

#include "test_harness.h"

/* Test threshold - see LAPACK dtest.in */
#define THRESH 20.0
#include <cblas.h>

/* Routine under test */
extern void dgetc2(const int n, double * const restrict A, const int lda,
                   int * const restrict ipiv, int * const restrict jpiv, int *info);

/* Matrix generation */
extern void dlatb4(const char *path, const int imat, const int m, const int n,
                   char *type, int *kl, int *ku, double *anorm, int *mode,
                   double *cndnum, char *dist);
extern void dlatms(const int m, const int n, const char *dist,
                   uint64_t seed, const char *sym, double *d,
                   const int mode, const double cond, const double dmax,
                   const int kl, const int ku, const char *pack,
                   double *A, const int lda, double *work, int *info);

/* Utilities */
extern double dlamch(const char *cmach);
extern double dlange(const char *norm, const int m, const int n,
                     const double * const restrict A, const int lda,
                     double * const restrict work);

/*
 * Test fixture
 */
typedef struct {
    int n;
    int lda;
    double *A;       /* Factored matrix (overwritten by dgetc2) */
    double *A_orig;  /* Original matrix */
    double *d;       /* Singular values for dlatms */
    double *work;    /* Workspace */
    int *ipiv;       /* Row pivot indices */
    int *jpiv;       /* Column pivot indices */
    uint64_t seed;
} dgetc2_fixture_t;

static uint64_t g_seed = 1729;

/**
 * Apply row permutation to matrix.
 */
static void apply_row_perm(int n, double *A, int lda, const int *ipiv, int dir)
{
    if (dir > 0) {
        for (int i = 0; i < n - 1; i++) {
            if (ipiv[i] != i) {
                cblas_dswap(n, &A[i], lda, &A[ipiv[i]], lda);
            }
        }
    } else {
        for (int i = n - 2; i >= 0; i--) {
            if (ipiv[i] != i) {
                cblas_dswap(n, &A[i], lda, &A[ipiv[i]], lda);
            }
        }
    }
}

/**
 * Apply column permutation to matrix.
 */
static void apply_col_perm(int n, double *A, int lda, const int *jpiv, int dir)
{
    if (dir > 0) {
        for (int j = 0; j < n - 1; j++) {
            if (jpiv[j] != j) {
                cblas_dswap(n, &A[j * lda], 1, &A[jpiv[j] * lda], 1);
            }
        }
    } else {
        for (int j = n - 2; j >= 0; j--) {
            if (jpiv[j] != j) {
                cblas_dswap(n, &A[j * lda], 1, &A[jpiv[j] * lda], 1);
            }
        }
    }
}

/**
 * Verify LU factorization with complete pivoting.
 * Computes ||P*L*U*Q - A|| / (n * ||A|| * eps)
 */
static double verify_lu(int n, const double *A_orig, const double *LU,
                        int lda, const int *ipiv, const int *jpiv)
{
    double eps = dlamch("E");
    double *lwork = malloc(n * sizeof(double));
    double *L = calloc(n * n, sizeof(double));
    double *U = calloc(n * n, sizeof(double));
    double *PLU = malloc(n * n * sizeof(double));
    double *PLUQ = malloc(n * n * sizeof(double));

    assert_non_null(lwork);
    assert_non_null(L);
    assert_non_null(U);
    assert_non_null(PLU);
    assert_non_null(PLUQ);

    /* Extract L (unit lower triangular) */
    for (int j = 0; j < n; j++) {
        L[j + j * n] = 1.0;
        for (int i = j + 1; i < n; i++) {
            L[i + j * n] = LU[i + j * lda];
        }
    }

    /* Extract U (upper triangular) */
    for (int j = 0; j < n; j++) {
        for (int i = 0; i <= j; i++) {
            U[i + j * n] = LU[i + j * lda];
        }
    }

    /* Compute L * U */
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                n, n, n, 1.0, L, n, U, n, 0.0, PLU, n);

    /* Apply P^{-1} (reverse row permutation) */
    apply_row_perm(n, PLU, n, ipiv, -1);

    /* Apply Q^{-1} (reverse column permutation) */
    memcpy(PLUQ, PLU, n * n * sizeof(double));
    apply_col_perm(n, PLUQ, n, jpiv, -1);

    /* Compute PLUQ - A */
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) {
            PLUQ[i + j * n] -= A_orig[i + j * lda];
        }
    }

    /* Compute ||PLUQ - A|| / (n * ||A|| * eps) */
    double anorm = dlange("1", n, n, A_orig, lda, lwork);
    double resid_norm = dlange("1", n, n, PLUQ, n, lwork);

    double resid;
    if (anorm <= 0.0) {
        resid = resid_norm / eps;
    } else {
        resid = resid_norm / (n * anorm * eps);
    }

    free(lwork);
    free(L);
    free(U);
    free(PLU);
    free(PLUQ);

    return resid;
}

static int dgetc2_setup(void **state, int n)
{
    dgetc2_fixture_t *fix = malloc(sizeof(dgetc2_fixture_t));
    assert_non_null(fix);

    fix->n = n;
    fix->lda = n;
    fix->seed = g_seed++;

    fix->A = malloc(fix->lda * n * sizeof(double));
    fix->A_orig = malloc(fix->lda * n * sizeof(double));
    fix->d = malloc(n * sizeof(double));
    fix->work = malloc(3 * n * sizeof(double));
    fix->ipiv = malloc(n * sizeof(int));
    fix->jpiv = malloc(n * sizeof(int));

    assert_non_null(fix->A);
    assert_non_null(fix->A_orig);
    assert_non_null(fix->d);
    assert_non_null(fix->work);
    assert_non_null(fix->ipiv);
    assert_non_null(fix->jpiv);

    *state = fix;
    return 0;
}

static int dgetc2_teardown(void **state)
{
    dgetc2_fixture_t *fix = *state;
    if (fix) {
        free(fix->A);
        free(fix->A_orig);
        free(fix->d);
        free(fix->work);
        free(fix->ipiv);
        free(fix->jpiv);
        free(fix);
    }
    return 0;
}

static int setup_2(void **state) { return dgetc2_setup(state, 2); }
static int setup_3(void **state) { return dgetc2_setup(state, 3); }
static int setup_4(void **state) { return dgetc2_setup(state, 4); }
static int setup_5(void **state) { return dgetc2_setup(state, 5); }
static int setup_8(void **state) { return dgetc2_setup(state, 8); }

/**
 * Core test logic: generate matrix, factorize, verify.
 */
static double run_dgetc2_test(dgetc2_fixture_t *fix, int imat)
{
    char type, dist;
    int kl, ku, mode;
    double anorm, cndnum;
    int info;

    dlatb4("DGE", imat, fix->n, fix->n, &type, &kl, &ku, &anorm, &mode, &cndnum, &dist);

    dlatms(fix->n, fix->n, &dist, fix->seed, &type, fix->d, mode, cndnum, anorm,
           kl, ku, "N", fix->A, fix->lda, fix->work, &info);
    assert_int_equal(info, 0);

    memcpy(fix->A_orig, fix->A, fix->lda * fix->n * sizeof(double));

    /* Factor A using complete pivoting */
    dgetc2(fix->n, fix->A, fix->lda, fix->ipiv, fix->jpiv, &info);
    /* info > 0 means matrix was perturbed; not an error for dgetc2 */

    /* Verify factorization */
    return verify_lu(fix->n, fix->A_orig, fix->A, fix->lda, fix->ipiv, fix->jpiv);
}

/*
 * Test well-conditioned matrices (types 1-4).
 */
static void test_dgetc2_wellcond(void **state)
{
    dgetc2_fixture_t *fix = *state;
    double resid;

    for (int imat = 1; imat <= 4; imat++) {
        fix->seed = g_seed++;
        resid = run_dgetc2_test(fix, imat);
        assert_residual_ok(resid);
    }
}

/*
 * Test ill-conditioned matrices (type 8).
 */
static void test_dgetc2_illcond(void **state)
{
    dgetc2_fixture_t *fix = *state;
    fix->seed = g_seed++;
    double resid = run_dgetc2_test(fix, 8);
    assert_residual_ok(resid);
}

/*
 * Sanity check: identity matrix.
 */
static void test_dgetc2_identity(void **state)
{
    (void)state;

    int n = 4;
    double *A = calloc(n * n, sizeof(double));
    double *A_orig = calloc(n * n, sizeof(double));
    int *ipiv = malloc(n * sizeof(int));
    int *jpiv = malloc(n * sizeof(int));
    int info;

    assert_non_null(A);
    assert_non_null(A_orig);
    assert_non_null(ipiv);
    assert_non_null(jpiv);

    for (int i = 0; i < n; i++) {
        A[i + i * n] = 1.0;
        A_orig[i + i * n] = 1.0;
    }

    dgetc2(n, A, n, ipiv, jpiv, &info);

    double resid = verify_lu(n, A_orig, A, n, ipiv, jpiv);
    assert_residual_ok(resid);

    free(A);
    free(A_orig);
    free(ipiv);
    free(jpiv);
}

/*
 * Sanity check: simple 3x3 matrix.
 */
static void test_dgetc2_simple(void **state)
{
    (void)state;

    double A[9] = {2, 4, 8, 1, 3, 7, 1, 3, 9};
    double A_orig[9];
    memcpy(A_orig, A, 9 * sizeof(double));

    int ipiv[3], jpiv[3];
    int info;

    dgetc2(3, A, 3, ipiv, jpiv, &info);

    double resid = verify_lu(3, A_orig, A, 3, ipiv, jpiv);
    assert_residual_ok(resid);
}

#define DGETC2_TESTS(setup_fn) \
    cmocka_unit_test_setup_teardown(test_dgetc2_wellcond, setup_fn, dgetc2_teardown), \
    cmocka_unit_test_setup_teardown(test_dgetc2_illcond, setup_fn, dgetc2_teardown)

int main(void)
{
    const struct CMUnitTest tests[] = {
        /* Sanity checks */
        cmocka_unit_test(test_dgetc2_simple),
        cmocka_unit_test(test_dgetc2_identity),

        /* Comprehensive tests */
        DGETC2_TESTS(setup_2),
        DGETC2_TESTS(setup_3),
        DGETC2_TESTS(setup_4),
        DGETC2_TESTS(setup_5),
        DGETC2_TESTS(setup_8),
    };

    return cmocka_run_group_tests_name("dgetc2", tests, NULL, NULL);
}
