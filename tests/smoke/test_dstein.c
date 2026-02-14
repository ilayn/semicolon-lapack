/**
 * @file test_dstein.c
 * @brief CMocka test suite for dstein (inverse iteration eigenvectors of
 *        symmetric tridiagonal matrix).
 *
 * Tests the dstein routine which computes eigenvectors of a real symmetric
 * tridiagonal matrix T corresponding to specified eigenvalues, using
 * inverse iteration.
 *
 * Verification: dstt21 computes
 *   result[0] = |T - Z*diag(W)*Z'| / (|T| * n * ulp)
 *   result[1] = |I - Z*Z'| / (n * ulp)
 *
 * Matrix types tested:
 *   1. Identity (D=1, E=0) - trivial eigenvectors
 *   2. 1-2-1 Toeplitz (D=2, E=1) - well-separated eigenvalues
 *   3. Wilkinson (D[i]=|i-n/2|, E[i]=1) - some clustered eigenvalues
 *   4. Random symmetric tridiagonal
 *   5. Graded diagonal (D[i]=2^(-i), E[i]=1) - wide dynamic range
 */

#include "test_harness.h"
#include "test_rng.h"

/* Test threshold - see LAPACK dtest.in */
#define THRESH 20.0
#include <cblas.h>

/* Routine under test */
extern void dstein(const int n, const f64* D, const f64* E,
                   const int m, const f64* W,
                   const int* iblock, const int* isplit,
                   f64* Z, const int ldz,
                   f64* work, int* iwork, int* ifail, int* info);

/* Eigenvalue computation */
extern void dstebz(const char* range, const char* order, const int n,
                   const f64 vl, const f64 vu,
                   const int il, const int iu, const f64 abstol,
                   const f64* D, const f64* E,
                   int* m, int* nsplit, f64* W,
                   int* iblock, int* isplit,
                   f64* work, int* iwork, int* info);

/* Verification routine */
extern void dstt21(const int n, const int kband,
                   const f64* AD, const f64* AE,
                   const f64* SD, const f64* SE,
                   const f64* U, const int ldu,
                   f64* work, f64* result);

/* Utilities */
extern f64 dlamch(const char* cmach);

/* ---------- Test fixture ---------- */

typedef struct {
    int n;
    f64* D;       /* diagonal (n) */
    f64* E;       /* off-diagonal (n-1) */
    f64* W;       /* eigenvalues from dstebz (n) */
    int* iblock;     /* block indices from dstebz (n) */
    int* isplit;     /* split points from dstebz (n) */
    f64* Z;       /* eigenvectors (n x n) */
    f64* work;    /* workspace: max(5*n for dstein, n*(n+1) for dstt21, 4*n for dstebz) */
    int* iwork;      /* int workspace: max(n for dstein, 3*n for dstebz) */
    int* ifail;      /* convergence info (n) */
    f64* result;  /* dstt21 results (2) */
    uint64_t seed;
    uint64_t rng_state[4];
} dstein_fixture_t;

/* Global seed for test sequence reproducibility */
static uint64_t g_seed = 4321;

static int dstein_setup(void** state, int n)
{
    dstein_fixture_t* fix = malloc(sizeof(dstein_fixture_t));
    assert_non_null(fix);

    fix->n = n;
    fix->seed = g_seed++;

    fix->D = malloc(n * sizeof(f64));
    fix->E = malloc((n > 1 ? n - 1 : 1) * sizeof(f64));
    fix->W = malloc(n * sizeof(f64));
    fix->iblock = malloc(n * sizeof(int));
    fix->isplit = malloc(n * sizeof(int));
    fix->Z = calloc(n * n, sizeof(f64));

    /* Workspace: max(5*n, n*(n+1), 4*n) = n*(n+1) for n >= 5 */
    int work_sz = n * (n + 1);
    if (work_sz < 5 * n) work_sz = 5 * n;
    if (work_sz < 4 * n) work_sz = 4 * n;
    fix->work = malloc(work_sz * sizeof(f64));

    /* Integer workspace: max(n, 3*n) = 3*n */
    fix->iwork = malloc(3 * n * sizeof(int));
    fix->ifail = malloc(n * sizeof(int));
    fix->result = malloc(2 * sizeof(f64));

    assert_non_null(fix->D);
    assert_non_null(fix->E);
    assert_non_null(fix->W);
    assert_non_null(fix->iblock);
    assert_non_null(fix->isplit);
    assert_non_null(fix->Z);
    assert_non_null(fix->work);
    assert_non_null(fix->iwork);
    assert_non_null(fix->ifail);
    assert_non_null(fix->result);

    *state = fix;
    return 0;
}

static int dstein_teardown(void** state)
{
    dstein_fixture_t* fix = *state;
    if (fix) {
        free(fix->D);
        free(fix->E);
        free(fix->W);
        free(fix->iblock);
        free(fix->isplit);
        free(fix->Z);
        free(fix->work);
        free(fix->iwork);
        free(fix->ifail);
        free(fix->result);
        free(fix);
    }
    return 0;
}

/* Size-specific setup functions */
static int setup_2(void** state)  { return dstein_setup(state, 2); }
static int setup_5(void** state)  { return dstein_setup(state, 5); }
static int setup_10(void** state) { return dstein_setup(state, 10); }
static int setup_20(void** state) { return dstein_setup(state, 20); }
static int setup_50(void** state) { return dstein_setup(state, 50); }

/* ---------- Matrix generators ---------- */

/**
 * Type 1: Identity tridiagonal (D=1, E=0).
 * All eigenvalues are 1; eigenvectors are trivial (any orthonormal basis).
 */
static void gen_identity(int n, f64* D, f64* E)
{
    for (int i = 0; i < n; i++) D[i] = 1.0;
    for (int i = 0; i < n - 1; i++) E[i] = 0.0;
}

/**
 * Type 2: 1-2-1 Toeplitz (D=2, E=1).
 * Eigenvalues are 2 + 2*cos(k*pi/(n+1)) for k=1..n, well-separated.
 */
static void gen_toeplitz_121(int n, f64* D, f64* E)
{
    for (int i = 0; i < n; i++) D[i] = 2.0;
    for (int i = 0; i < n - 1; i++) E[i] = 1.0;
}

/**
 * Type 3: Wilkinson matrix (D[i]=|i - n/2|, E[i]=1).
 * Has some clustered eigenvalues near zero for even n.
 */
static void gen_wilkinson(int n, f64* D, f64* E)
{
    int half = n / 2;
    for (int i = 0; i < n; i++) {
        D[i] = (f64)(i >= half ? i - half : half - i);
    }
    for (int i = 0; i < n - 1; i++) E[i] = 1.0;
}

/**
 * Type 4: Random symmetric tridiagonal.
 * D[i] uniform in [-1, 1], E[i] uniform in [-1, 1].
 */
static void gen_random(int n, f64* D, f64* E, uint64_t state[static 4])
{
    for (int i = 0; i < n; i++) D[i] = rng_uniform_symmetric(state);
    for (int i = 0; i < n - 1; i++) E[i] = rng_uniform_symmetric(state);
}

/**
 * Type 5: Graded diagonal (D[i]=2^(-i), E[i]=1).
 * Wide dynamic range in diagonal entries.
 */
static void gen_graded(int n, f64* D, f64* E)
{
    f64 scale = 1.0;
    for (int i = 0; i < n; i++) {
        D[i] = scale;
        scale *= 0.5;
    }
    for (int i = 0; i < n - 1; i++) E[i] = 1.0;
}

/* ---------- Core test logic ---------- */

/**
 * Run dstein test for a given matrix type.
 * Computes eigenvalues via dstebz, eigenvectors via dstein,
 * then verifies with dstt21.
 */
static void run_dstein_test(dstein_fixture_t* fix, int imat)
{
    int n = fix->n;
    int info;
    int m, nsplit;
    f64 abstol = 2.0 * dlamch("S");

    /* Generate tridiagonal matrix */
    switch (imat) {
        case 1: gen_identity(n, fix->D, fix->E); break;
        case 2: gen_toeplitz_121(n, fix->D, fix->E); break;
        case 3: gen_wilkinson(n, fix->D, fix->E); break;
        case 4:
            rng_seed(fix->rng_state, fix->seed);
            gen_random(n, fix->D, fix->E, fix->rng_state);
            break;
        case 5: gen_graded(n, fix->D, fix->E); break;
    }

    /* Compute all eigenvalues with dstebz, block-ordered for dstein */
    dstebz("A", "B", n, 0.0, 0.0, 0, 0, abstol,
           fix->D, fix->E, &m, &nsplit, fix->W,
           fix->iblock, fix->isplit, fix->work, fix->iwork, &info);
    assert_info_success(info);
    assert_int_equal(m, n);

    /* Clear Z and ifail */
    memset(fix->Z, 0, n * n * sizeof(f64));
    memset(fix->ifail, 0, n * sizeof(int));

    /* Compute eigenvectors via inverse iteration */
    dstein(n, fix->D, fix->E, m, fix->W,
           fix->iblock, fix->isplit,
           fix->Z, n, fix->work, fix->iwork, fix->ifail, &info);
    assert_info_success(info);

    /* Check all ifail entries are zero (all eigenvectors converged) */
    for (int i = 0; i < m; i++) {
        assert_int_equal(fix->ifail[i], 0);
    }

    /* Verify with dstt21: A = Z * diag(W) * Z', kband=0 (diagonal S) */
    fix->result[0] = 0.0;
    fix->result[1] = 0.0;
    dstt21(n, 0, fix->D, fix->E, fix->W, NULL, fix->Z, n,
           fix->work, fix->result);
    assert_residual_ok(fix->result[0]);
    assert_residual_ok(fix->result[1]);
}

/* ---------- Test functions ---------- */

/**
 * Test all eigenvectors for all matrix types.
 * Uses dstebz RANGE='A' to get all n eigenvalues, then dstein for all.
 */
static void test_all_eigenvectors(void** state)
{
    dstein_fixture_t* fix = *state;

    for (int imat = 1; imat <= 5; imat++) {
        fix->seed = g_seed++;
        run_dstein_test(fix, imat);
    }
}

/**
 * Test partial eigenvectors (subset of eigenvalues).
 * Uses dstebz RANGE='I' to get a subset of eigenvalues,
 * then computes eigenvectors for only those and verifies
 * orthogonality and T*z = lambda*z.
 */
static void test_partial(void** state)
{
    dstein_fixture_t* fix = *state;
    int n = fix->n;
    int info;
    int m, nsplit;
    f64 abstol = 2.0 * dlamch("S");
    f64 ulp = dlamch("P");

    if (n < 4) {
        skip_test("partial test requires n >= 4");
    }

    /* Use 1-2-1 Toeplitz for predictable, well-separated eigenvalues */
    gen_toeplitz_121(n, fix->D, fix->E);

    /* Request eigenvalues il..iu (1-based in Fortran, but our dstebz uses
     * 0-based if the C port follows project convention; however the
     * specification states il, iu as in LAPACK convention) */
    int il = 2;
    int iu = n / 2 + 1;
    int expected_m = iu - il + 1;

    dstebz("I", "B", n, 0.0, 0.0, il, iu, abstol,
           fix->D, fix->E, &m, &nsplit, fix->W,
           fix->iblock, fix->isplit, fix->work, fix->iwork, &info);
    assert_info_success(info);
    assert_int_equal(m, expected_m);

    /* Clear Z and ifail */
    memset(fix->Z, 0, n * n * sizeof(f64));
    memset(fix->ifail, 0, n * sizeof(int));

    /* Compute eigenvectors for the subset */
    dstein(n, fix->D, fix->E, m, fix->W,
           fix->iblock, fix->isplit,
           fix->Z, n, fix->work, fix->iwork, fix->ifail, &info);
    assert_info_success(info);

    for (int i = 0; i < m; i++) {
        assert_int_equal(fix->ifail[i], 0);
    }

    /* Verify T*z_j = lambda_j * z_j for each eigenvector */
    /* Compute residual: || T*z - lambda*z || / (||T|| * ||z|| * ulp) */
    f64* Tz = fix->work;  /* reuse workspace, needs n doubles */
    f64 anorm = fabs(fix->D[0]);
    for (int i = 1; i < n; i++) {
        f64 row_sum = fabs(fix->D[i]) + fabs(fix->E[i - 1]);
        if (i < n - 1) row_sum += fabs(fix->E[i]);
        if (row_sum > anorm) anorm = row_sum;
    }
    /* Also check first row */
    {
        f64 row0 = fabs(fix->D[0]);
        if (n > 1) row0 += fabs(fix->E[0]);
        if (row0 > anorm) anorm = row0;
    }

    for (int j = 0; j < m; j++) {
        f64* zj = fix->Z + j * n;  /* column j of Z (column-major) */

        /* Compute T*zj */
        Tz[0] = fix->D[0] * zj[0];
        if (n > 1) Tz[0] += fix->E[0] * zj[1];
        for (int i = 1; i < n - 1; i++) {
            Tz[i] = fix->E[i - 1] * zj[i - 1] + fix->D[i] * zj[i] + fix->E[i] * zj[i + 1];
        }
        if (n > 1) {
            Tz[n - 1] = fix->E[n - 2] * zj[n - 2] + fix->D[n - 1] * zj[n - 1];
        }

        /* Compute || T*z - lambda*z || */
        f64 resid_norm = 0.0;
        for (int i = 0; i < n; i++) {
            f64 diff = Tz[i] - fix->W[j] * zj[i];
            resid_norm += diff * diff;
        }
        resid_norm = sqrt(resid_norm);

        /* Compute ||z|| (should be ~1 for normalized eigenvectors) */
        f64 znorm = cblas_dnrm2(n, zj, 1);

        /* Normalized residual */
        f64 resid = resid_norm / (anorm * znorm * n * ulp);
        assert_residual_ok(resid);
    }

    /* Verify orthogonality: || Z'*Z - I || / (m * ulp)
     * where Z is n x m (the m computed eigenvectors) */
    f64* ZtZ = fix->work;  /* m*m, fits in n*(n+1) workspace */
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                m, m, n, 1.0, fix->Z, n, fix->Z, n, 0.0, ZtZ, m);

    /* Subtract identity */
    for (int j = 0; j < m; j++) {
        ZtZ[j + j * m] -= 1.0;
    }

    /* 1-norm of (Z'Z - I) */
    f64 orth_norm = 0.0;
    for (int j = 0; j < m; j++) {
        f64 col_sum = 0.0;
        for (int i = 0; i < m; i++) {
            col_sum += fabs(ZtZ[i + j * m]);
        }
        if (col_sum > orth_norm) orth_norm = col_sum;
    }

    f64 orth_resid = orth_norm / (m * ulp);
    assert_residual_ok(orth_resid);
}

/* ---------- Main ---------- */

#define DSTEIN_TESTS(setup_fn) \
    cmocka_unit_test_setup_teardown(test_all_eigenvectors, setup_fn, dstein_teardown), \
    cmocka_unit_test_setup_teardown(test_partial, setup_fn, dstein_teardown)

int main(void)
{
    const struct CMUnitTest tests[] = {
        DSTEIN_TESTS(setup_2),
        DSTEIN_TESTS(setup_5),
        DSTEIN_TESTS(setup_10),
        DSTEIN_TESTS(setup_20),
        DSTEIN_TESTS(setup_50),
    };

    return cmocka_run_group_tests_name("dstein", tests, NULL, NULL);
}
