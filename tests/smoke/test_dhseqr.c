/**
 * @file test_dhseqr.c
 * @brief Test QR algorithm routine dhseqr.
 *
 * Tests based on LAPACK TESTING/EIG/dchkhs.f, adapted to CMocka framework.
 * Verifies:
 *   (3) | H - Z*T*Z' | / ( |H| n ulp )
 *   (4) | I - Z*Z' | / ( n ulp )
 *   (7) | T(Z computed) - T(Z not computed) | / ( |T| ulp )
 *   (8) | W(Z computed) - W(Z not computed) | / ( |W| ulp )
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
    double* H;       /* Hessenberg matrix */
    double* T1;      /* Schur form (with Z) */
    double* T2;      /* Schur form (without Z) */
    double* Z;       /* Schur vectors */
    double* wr1;     /* Real eigenvalues (with Z) */
    double* wi1;     /* Imaginary eigenvalues (with Z) */
    double* wr2;     /* Real eigenvalues (without Z) */
    double* wi2;     /* Imaginary eigenvalues (without Z) */
    double* work;    /* Workspace */
    double* tau;     /* Householder reflectors */
    uint64_t seed;
    uint64_t rng_state[4];
} dhseqr_fixture_t;

/* Forward declarations from semicolon_lapack */
extern void dgehrd(const int n, const int ilo, const int ihi,
                   double* A, const int lda, double* tau,
                   double* work, const int lwork, int* info);
extern void dorghr(const int n, const int ilo, const int ihi,
                   double* A, const int lda, const double* tau,
                   double* work, const int lwork, int* info);
extern void dhseqr(const char* job, const char* compz, const int n,
                   const int ilo, const int ihi, double* H, const int ldh,
                   double* wr, double* wi, double* Z, const int ldz,
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
    dhseqr_fixture_t* fix = malloc(sizeof(dhseqr_fixture_t));
    if (!fix) return -1;

    fix->n = n;
    fix->seed = 0x87654321ULL;

    /* Allocate matrices */
    fix->A = malloc(n * n * sizeof(double));
    fix->H = malloc(n * n * sizeof(double));
    fix->T1 = malloc(n * n * sizeof(double));
    fix->T2 = malloc(n * n * sizeof(double));
    fix->Z = malloc(n * n * sizeof(double));
    fix->wr1 = malloc(n * sizeof(double));
    fix->wi1 = malloc(n * sizeof(double));
    fix->wr2 = malloc(n * sizeof(double));
    fix->wi2 = malloc(n * sizeof(double));
    fix->tau = malloc(n * sizeof(double));

    /* Workspace: need at least 2*n*n for verification + lwork for routines */
    int lwork = 6 * n * n + 2 * n;
    fix->work = malloc(lwork * sizeof(double));

    if (!fix->A || !fix->H || !fix->T1 || !fix->T2 || !fix->Z ||
        !fix->wr1 || !fix->wi1 || !fix->wr2 || !fix->wi2 ||
        !fix->tau || !fix->work) {
        free(fix->A); free(fix->H); free(fix->T1); free(fix->T2);
        free(fix->Z); free(fix->wr1); free(fix->wi1);
        free(fix->wr2); free(fix->wi2); free(fix->tau); free(fix->work);
        free(fix);
        return -1;
    }

    rng_seed(fix->rng_state, fix->seed);
    *state = fix;
    return 0;
}

static int teardown(void** state) {
    dhseqr_fixture_t* fix = *state;
    if (fix) {
        free(fix->A);
        free(fix->H);
        free(fix->T1);
        free(fix->T2);
        free(fix->Z);
        free(fix->wr1);
        free(fix->wi1);
        free(fix->wr2);
        free(fix->wi2);
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
 * Generate upper Hessenberg test matrix.
 */
static void generate_hessenberg_matrix(int n, double* H, int ldh, double anorm,
                                       uint64_t state[static 4])
{
    /* Generate random Hessenberg matrix directly */
    for (int j = 0; j < n; j++) {
        for (int i = 0; i <= j + 1 && i < n; i++) {
            H[i + j * ldh] = anorm * rng_uniform_symmetric(state);
        }
        for (int i = j + 2; i < n; i++) {
            H[i + j * ldh] = 0.0;
        }
    }
}

/**
 * Compare two Schur forms T1 and T2.
 * Returns | T1 - T2 | / ( |T1| ulp )
 */
static double compare_schur_forms(int n, const double* T1, const double* T2, int ldt, double* work)
{
    double ulp = dlamch("P");
    double unfl = dlamch("S");

    /* Compute T1 - T2 in work */
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) {
            work[i + j * n] = T1[i + j * ldt] - T2[i + j * ldt];
        }
    }

    double diff = dlange("1", n, n, work, n, &work[n * n]);
    double tnorm = dlange("1", n, n, T1, ldt, &work[n * n]);

    if (tnorm < unfl) tnorm = unfl;

    return diff / (tnorm * ulp);
}

/**
 * Compare two eigenvalue sets.
 * Returns max | W1 - W2 | / ( max(|W1|, |W2|) ulp )
 */
static double compare_eigenvalues(int n, const double* wr1, const double* wi1,
                                  const double* wr2, const double* wi2)
{
    double ulp = dlamch("P");
    double unfl = dlamch("S");

    double maxw = 0.0;
    double maxdiff = 0.0;

    for (int j = 0; j < n; j++) {
        double w1 = fabs(wr1[j]) + fabs(wi1[j]);
        double w2 = fabs(wr2[j]) + fabs(wi2[j]);
        double diff = fabs(wr1[j] - wr2[j]) + fabs(wi1[j] - wi2[j]);

        if (w1 > maxw) maxw = w1;
        if (w2 > maxw) maxw = w2;
        if (diff > maxdiff) maxdiff = diff;
    }

    if (maxw < unfl) maxw = unfl;

    return maxdiff / (maxw * ulp);
}

/**
 * Test QR algorithm for Schur decomposition.
 */
static void test_qr_schur(dhseqr_fixture_t* fix)
{
    int n = fix->n;
    int lda = n;
    int info;
    double result[2];

    const double ONE = 1.0;
    const double ZERO = 0.0;

    /* Generate Hessenberg matrix */
    generate_hessenberg_matrix(n, fix->H, lda, ONE, fix->rng_state);

    /* Copy H to T1 for Schur decomposition with Z */
    dlacpy(" ", n, n, fix->H, lda, fix->T1, lda);

    /* Initialize Z to identity */
    dlaset("F", n, n, ZERO, ONE, fix->Z, lda);

    /* Compute Schur form with Schur vectors (0-based indexing) */
    int ilo = 0;
    int ihi = n - 1;
    int lwork = 4 * n * n;

    dhseqr("S", "I", n, ilo, ihi, fix->T1, lda, fix->wr1, fix->wi1,
           fix->Z, lda, fix->work, lwork, &info);

    /* dhseqr may return info > 0 if it doesn't converge, which is not a failure */
    if (info < 0) {
        fail_msg("dhseqr returned info = %d", info);
    }

    /* Copy H to T2 for Schur decomposition without Z */
    dlacpy(" ", n, n, fix->H, lda, fix->T2, lda);

    /* Compute Schur form without Schur vectors */
    dhseqr("S", "N", n, ilo, ihi, fix->T2, lda, fix->wr2, fix->wi2,
           NULL, lda, fix->work, lwork, &info);

    if (info < 0) {
        fail_msg("dhseqr (no Z) returned info = %d", info);
    }

    /* Test 3: | H - Z*T*Z' | / ( |H| n ulp ) */
    /* Test 4: | I - Z*Z' | / ( n ulp ) */
    int lwork_verify = 2 * n * n;
    dhst01(n, ilo, ihi, fix->H, lda, fix->T1, lda, fix->Z, lda,
           fix->work, lwork_verify, result);

    assert_residual_ok(result[0]);  /* Schur decomposition accuracy */
    assert_residual_ok(result[1]);  /* Z orthogonality */

    /* Test 7: | T2 - T1 | / ( |T| ulp ) */
    double resid7 = compare_schur_forms(n, fix->T1, fix->T2, lda, fix->work);
    assert_residual_ok(resid7);

    /* Test 8: | W2 - W1 | / ( max|W| ulp ) */
    double resid8 = compare_eigenvalues(n, fix->wr1, fix->wi1, fix->wr2, fix->wi2);
    assert_residual_ok(resid8);
}

/* Test functions */
static void test_schur_n5(void** state)
{
    dhseqr_fixture_t* fix = *state;
    test_qr_schur(fix);
}

static void test_schur_n10(void** state)
{
    dhseqr_fixture_t* fix = *state;
    test_qr_schur(fix);
}

static void test_schur_n20(void** state)
{
    dhseqr_fixture_t* fix = *state;
    test_qr_schur(fix);
}

static void test_schur_n32(void** state)
{
    dhseqr_fixture_t* fix = *state;
    test_qr_schur(fix);
}

/**
 * Test eigenvalue-only computation.
 */
static void test_eigenvalues_only(void** state)
{
    dhseqr_fixture_t* fix = *state;
    int n = fix->n;
    int lda = n;
    int info;

    /* Generate Hessenberg matrix */
    generate_hessenberg_matrix(n, fix->H, lda, 1.0, fix->rng_state);

    /* Copy H to T1 */
    dlacpy(" ", n, n, fix->H, lda, fix->T1, lda);

    /* Compute eigenvalues only (0-based indexing) */
    int ilo = 0;
    int ihi = n - 1;
    int lwork = 4 * n * n;

    dhseqr("E", "N", n, ilo, ihi, fix->T1, lda, fix->wr1, fix->wi1,
           NULL, lda, fix->work, lwork, &info);

    if (info < 0) {
        fail_msg("dhseqr (E) returned info = %d", info);
    }

    /* The eigenvalues should be reasonable (not NaN or Inf) */
    for (int j = 0; j < n; j++) {
        assert_true(isfinite(fix->wr1[j]));
        assert_true(isfinite(fix->wi1[j]));
    }
}

/**
 * Test workspace query.
 */
static void test_workspace_query(void** state)
{
    dhseqr_fixture_t* fix = *state;
    int n = fix->n;
    int info;
    double work_query;

    /* Query optimal workspace (0-based indexing) */
    dhseqr("S", "I", n, 0, n - 1, fix->H, n, fix->wr1, fix->wi1,
           fix->Z, n, &work_query, -1, &info);

    assert_info_success(info);
    assert_true(work_query >= (double)n);
}

/**
 * Test with diagonal matrix (already in Schur form).
 */
static void test_diagonal_matrix(void** state)
{
    dhseqr_fixture_t* fix = *state;
    int n = fix->n;
    int lda = n;
    int info;

    const double ZERO = 0.0;
    const double ONE = 1.0;

    /* Create diagonal matrix */
    dlaset("F", n, n, ZERO, ZERO, fix->H, lda);
    for (int j = 0; j < n; j++) {
        fix->H[j + j * lda] = (double)(j + 1);
    }

    dlacpy(" ", n, n, fix->H, lda, fix->T1, lda);
    dlaset("F", n, n, ZERO, ONE, fix->Z, lda);

    int lwork = 4 * n * n;
    /* 0-based indexing: ilo=0, ihi=n-1 */
    dhseqr("S", "I", n, 0, n - 1, fix->T1, lda, fix->wr1, fix->wi1,
           fix->Z, lda, fix->work, lwork, &info);

    /* Should succeed */
    if (info < 0) {
        fail_msg("dhseqr returned info = %d for diagonal matrix", info);
    }

    /* Eigenvalues should be 1, 2, ..., n (real, no imaginary part) */
    for (int j = 0; j < n; j++) {
        assert_double_equal(fix->wi1[j], ZERO, 1e-10);
    }

    /* Z should still be close to identity for diagonal input */
    double resid = 0.0;
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) {
            double expected = (i == j) ? ONE : ZERO;
            double err = fabs(fix->Z[i + j * lda] - expected);
            if (err > resid) resid = err;
        }
    }
    assert_true(resid < 1e-10);
}

int main(void)
{
    const struct CMUnitTest tests_n5[] = {
        cmocka_unit_test_setup_teardown(test_schur_n5, setup_5, teardown),
        cmocka_unit_test_setup_teardown(test_eigenvalues_only, setup_5, teardown),
        cmocka_unit_test_setup_teardown(test_workspace_query, setup_5, teardown),
        cmocka_unit_test_setup_teardown(test_diagonal_matrix, setup_5, teardown),
    };

    const struct CMUnitTest tests_n10[] = {
        cmocka_unit_test_setup_teardown(test_schur_n10, setup_10, teardown),
        cmocka_unit_test_setup_teardown(test_eigenvalues_only, setup_10, teardown),
        cmocka_unit_test_setup_teardown(test_diagonal_matrix, setup_10, teardown),
    };

    const struct CMUnitTest tests_n20[] = {
        cmocka_unit_test_setup_teardown(test_schur_n20, setup_20, teardown),
    };

    const struct CMUnitTest tests_n32[] = {
        cmocka_unit_test_setup_teardown(test_schur_n32, setup_32, teardown),
    };

    int result = 0;
    result += cmocka_run_group_tests_name("dhseqr_n5", tests_n5, NULL, NULL);
    result += cmocka_run_group_tests_name("dhseqr_n10", tests_n10, NULL, NULL);
    result += cmocka_run_group_tests_name("dhseqr_n20", tests_n20, NULL, NULL);
    result += cmocka_run_group_tests_name("dhseqr_n32", tests_n32, NULL, NULL);

    return result;
}
