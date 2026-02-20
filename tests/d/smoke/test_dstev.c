/**
 * @file test_dstev.c
 * @brief CMocka test suite for dstev (symmetric tridiagonal eigenvalue driver).
 *
 * Tests the simple eigenvalue driver dstev which computes all eigenvalues
 * and, optionally, eigenvectors of a real symmetric tridiagonal matrix.
 *
 * Verification:
 *   - dstt21: checks ||U S U' - A|| / (n * ||A|| * eps) and ||U U' - I|| / (n * eps)
 *   - dstech: Sturm sequence verification of eigenvalues
 *
 * Matrix types tested (symmetric tridiagonal):
 *   1. Zero matrix
 *   2. Identity (D=1, E=0)
 *   3. 1-2-1 Toeplitz (D=2, E=1)
 *   4. Wilkinson (D[i]=|i-n/2|, E[i]=1)
 *   5. Random symmetric tridiagonal
 *   6. Graded diagonal (D[i]=2^(-i), E[i]=1)
 */

#include "test_harness.h"
#include "test_rng.h"

/* Test threshold - see LAPACK dtest.in */
#define THRESH 20.0
#include <cblas.h>

/* Routine under test */
extern void dstev(const char* jobz, const int n,
                  f64* const restrict D, f64* const restrict E,
                  f64* const restrict Z, const int ldz,
                  f64* const restrict work, int* info);

/* Verification routines */
extern void dstt21(const int n, const int kband,
                   const f64* AD, const f64* AE,
                   const f64* SD, const f64* SE,
                   const f64* U, const int ldu,
                   f64* work, f64* result);
extern void dstech(const int n, const f64* A, const f64* B,
                   const f64* eig, const f64 tol, f64* work, int* info);

/* Utilities */
extern f64 dlamch(const char* cmach);

/* ---------- Test fixture ---------- */
typedef struct {
    int n;
    f64* AD;     /* original diagonal */
    f64* AE;     /* original off-diagonal */
    f64* D;      /* diagonal (overwritten by dstev) */
    f64* E;      /* off-diagonal (overwritten by dstev) */
    f64* D2;     /* second copy for JOBZ='N' */
    f64* E2;     /* second copy for JOBZ='N' */
    f64* Z;      /* eigenvectors (n x n) */
    f64* work;   /* workspace: max(2*(n-1) for dstev, n*(n+1) for dstt21) */
    f64* result; /* dstt21 results (2) */
    uint64_t seed;
    uint64_t rng_state[4];
} dstev_fixture_t;

/* Global seed for reproducibility */
static uint64_t g_seed = 7331;

static int dstev_setup(void** state, int n)
{
    dstev_fixture_t* fix = malloc(sizeof(dstev_fixture_t));
    assert_non_null(fix);

    fix->n = n;
    fix->seed = g_seed++;

    fix->AD = malloc(n * sizeof(f64));
    fix->AE = malloc((n > 1 ? n - 1 : 1) * sizeof(f64));
    fix->D = malloc(n * sizeof(f64));
    fix->E = malloc((n > 1 ? n - 1 : 1) * sizeof(f64));
    fix->D2 = malloc(n * sizeof(f64));
    fix->E2 = malloc((n > 1 ? n - 1 : 1) * sizeof(f64));
    fix->Z = malloc(n * n * sizeof(f64));

    /* workspace: max(2*(n-1), n*(n+1)) */
    int work_stev = (n > 1) ? 2 * (n - 1) : 1;
    int work_stt21 = n * (n + 1);
    int work_stech = 2 * n;
    int work_sz = work_stev;
    if (work_stt21 > work_sz) work_sz = work_stt21;
    if (work_stech > work_sz) work_sz = work_stech;
    fix->work = malloc(work_sz * sizeof(f64));

    fix->result = malloc(2 * sizeof(f64));

    assert_non_null(fix->AD);
    assert_non_null(fix->AE);
    assert_non_null(fix->D);
    assert_non_null(fix->E);
    assert_non_null(fix->D2);
    assert_non_null(fix->E2);
    assert_non_null(fix->Z);
    assert_non_null(fix->work);
    assert_non_null(fix->result);

    *state = fix;
    return 0;
}

static int dstev_teardown(void** state)
{
    dstev_fixture_t* fix = *state;
    if (fix) {
        free(fix->AD);
        free(fix->AE);
        free(fix->D);
        free(fix->E);
        free(fix->D2);
        free(fix->E2);
        free(fix->Z);
        free(fix->work);
        free(fix->result);
        free(fix);
    }
    return 0;
}

/* Size-specific setup functions */
static int setup_1(void** state) { return dstev_setup(state, 1); }
static int setup_2(void** state) { return dstev_setup(state, 2); }
static int setup_5(void** state) { return dstev_setup(state, 5); }
static int setup_10(void** state) { return dstev_setup(state, 10); }
static int setup_20(void** state) { return dstev_setup(state, 20); }
static int setup_50(void** state) { return dstev_setup(state, 50); }

/* ---------- Matrix generation helpers ---------- */

/* Type 1: Zero matrix */
static void gen_zero(int n, f64* D, f64* E)
{
    for (int i = 0; i < n; i++) D[i] = 0.0;
    for (int i = 0; i < n - 1; i++) E[i] = 0.0;
}

/* Type 2: Identity (D=1, E=0) */
static void gen_identity(int n, f64* D, f64* E)
{
    for (int i = 0; i < n; i++) D[i] = 1.0;
    for (int i = 0; i < n - 1; i++) E[i] = 0.0;
}

/* Type 3: 1-2-1 Toeplitz (D=2, E=1) */
static void gen_toeplitz_121(int n, f64* D, f64* E)
{
    for (int i = 0; i < n; i++) D[i] = 2.0;
    for (int i = 0; i < n - 1; i++) E[i] = 1.0;
}

/* Type 4: Wilkinson (D[i]=|i - n/2|, E[i]=1) */
static void gen_wilkinson(int n, f64* D, f64* E)
{
    for (int i = 0; i < n; i++) {
        D[i] = fabs((f64)i - (f64)(n / 2));
    }
    for (int i = 0; i < n - 1; i++) E[i] = 1.0;
}

/* Type 5: Random symmetric tridiagonal */
static void gen_random(int n, f64* D, f64* E, uint64_t state[static 4])
{
    for (int i = 0; i < n; i++) {
        D[i] = rng_uniform_symmetric(state);
    }
    for (int i = 0; i < n - 1; i++) {
        E[i] = rng_uniform_symmetric(state);
    }
}

/* Type 6: Graded diagonal (D[i]=2^(-i), E[i]=1) */
static void gen_graded(int n, f64* D, f64* E)
{
    for (int i = 0; i < n; i++) {
        D[i] = pow(2.0, -(f64)i);
    }
    for (int i = 0; i < n - 1; i++) E[i] = 1.0;
}

/**
 * Generate test matrix of given type and copy to working arrays.
 */
static void gen_matrix(dstev_fixture_t* fix, int imat)
{
    int n = fix->n;

    switch (imat) {
    case 1: gen_zero(n, fix->AD, fix->AE); break;
    case 2: gen_identity(n, fix->AD, fix->AE); break;
    case 3: gen_toeplitz_121(n, fix->AD, fix->AE); break;
    case 4: gen_wilkinson(n, fix->AD, fix->AE); break;
    case 5:
        rng_seed(fix->rng_state, fix->seed + (uint64_t)imat);
        gen_random(n, fix->AD, fix->AE, fix->rng_state);
        break;
    case 6: gen_graded(n, fix->AD, fix->AE); break;
    }

    /* Copy to working arrays */
    memcpy(fix->D, fix->AD, n * sizeof(f64));
    if (n > 1) {
        memcpy(fix->E, fix->AE, (n - 1) * sizeof(f64));
    }
}

/* ---------- Test functions ---------- */

/**
 * JOBZ='V': compute eigenvalues and eigenvectors, verify with dstt21.
 */
static void test_jobz_V(void** state)
{
    dstev_fixture_t* fix = *state;
    int n = fix->n;
    int info;

    for (int imat = 1; imat <= 6; imat++) {
        gen_matrix(fix, imat);

        dstev("V", n, fix->D, fix->E, fix->Z, n, fix->work, &info);
        assert_info_success(info);

        /* Verify: dstt21 checks U S U' = A and U U' = I */
        fix->result[0] = 0.0;
        fix->result[1] = 0.0;
        dstt21(n, 0, fix->AD, fix->AE, fix->D, NULL, fix->Z, n,
               fix->work, fix->result);

        assert_residual_ok(fix->result[0]);
        assert_residual_ok(fix->result[1]);
    }
}

/**
 * JOBZ='N': compute eigenvalues only, compare with JOBZ='V' eigenvalues.
 */
static void test_jobz_N(void** state)
{
    dstev_fixture_t* fix = *state;
    int n = fix->n;
    int info;
    f64 ulp = dlamch("P");

    for (int imat = 1; imat <= 6; imat++) {
        gen_matrix(fix, imat);

        /* Get reference eigenvalues via JOBZ='V' */
        dstev("V", n, fix->D, fix->E, fix->Z, n, fix->work, &info);
        assert_info_success(info);

        /* Re-generate and compute with JOBZ='N' */
        gen_matrix(fix, imat);
        memcpy(fix->D2, fix->D, n * sizeof(f64));
        if (n > 1) {
            memcpy(fix->E2, fix->E, (n - 1) * sizeof(f64));
        }

        /* First call overwrites D with eigenvalues */
        dstev("V", n, fix->D, fix->E, fix->Z, n, fix->work, &info);
        assert_info_success(info);

        /* Now call JOBZ='N' on the copy */
        dstev("N", n, fix->D2, fix->E2, NULL, 1, fix->work, &info);
        assert_info_success(info);

        /* Compare eigenvalues: max|D[i]-D2[i]| / (n * ulp * max|D|) */
        f64 maxd = 0.0;
        for (int i = 0; i < n; i++) {
            f64 ad = fabs(fix->D[i]);
            if (ad > maxd) maxd = ad;
        }

        f64 maxdiff = 0.0;
        for (int i = 0; i < n; i++) {
            f64 diff = fabs(fix->D[i] - fix->D2[i]);
            if (diff > maxdiff) maxdiff = diff;
        }

        f64 resid;
        if (maxd == 0.0) {
            resid = maxdiff / ulp;
        } else {
            resid = maxdiff / ((f64)n * ulp * maxd);
        }
        assert_residual_ok(resid);
    }
}

/**
 * Sturm sequence verification of eigenvalues via dstech.
 */
static void test_sturm(void** state)
{
    dstev_fixture_t* fix = *state;
    int n = fix->n;
    int info;

    for (int imat = 1; imat <= 6; imat++) {
        /* Skip zero matrix for Sturm test (all zero eigenvalues are trivially correct) */
        if (imat == 1) continue;

        gen_matrix(fix, imat);

        dstev("V", n, fix->D, fix->E, fix->Z, n, fix->work, &info);
        assert_info_success(info);

        /* dstech verifies eigenvalues via Sturm sequences */
        int stech_info;
        dstech(n, fix->AD, fix->AE, fix->D, THRESH, fix->work, &stech_info);
        assert_info_success(stech_info);
    }
}

/* ---------- Main ---------- */

#define DSTEV_TESTS(setup_fn) \
    cmocka_unit_test_setup_teardown(test_jobz_V, setup_fn, dstev_teardown), \
    cmocka_unit_test_setup_teardown(test_jobz_N, setup_fn, dstev_teardown), \
    cmocka_unit_test_setup_teardown(test_sturm, setup_fn, dstev_teardown)

int main(void)
{
    const struct CMUnitTest tests[] = {
        DSTEV_TESTS(setup_1),
        DSTEV_TESTS(setup_2),
        DSTEV_TESTS(setup_5),
        DSTEV_TESTS(setup_10),
        DSTEV_TESTS(setup_20),
        DSTEV_TESTS(setup_50),
    };

    return cmocka_run_group_tests_name("dstev", tests, NULL, NULL);
}
