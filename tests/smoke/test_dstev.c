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

/* Test threshold - see LAPACK dtest.in */
#define THRESH 20.0
#include <cblas.h>

/* Routine under test */
extern void dstev(const char* jobz, const int n,
                  double* const restrict D, double* const restrict E,
                  double* const restrict Z, const int ldz,
                  double* const restrict work, int* info);

/* Verification routines */
extern void dstt21(const int n, const int kband,
                   const double* AD, const double* AE,
                   const double* SD, const double* SE,
                   const double* U, const int ldu,
                   double* work, double* result);
extern void dstech(const int n, const double* A, const double* B,
                   const double* eig, const double tol, double* work, int* info);

/* Utilities */
extern double dlamch(const char* cmach);

/* ---------- xoshiro256+ RNG ---------- */
static uint64_t rng_state[4];

static void rng_seed(uint64_t s)
{
    for (int i = 0; i < 4; i++) {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        rng_state[i] = s;
    }
}

static double rng_uniform(void)
{
    uint64_t s = rng_state[1] * 5;
    uint64_t r = ((s << 7) | (s >> 57)) * 9;
    uint64_t t = rng_state[1] << 17;
    rng_state[2] ^= rng_state[0];
    rng_state[3] ^= rng_state[1];
    rng_state[1] ^= rng_state[2];
    rng_state[0] ^= rng_state[3];
    rng_state[2] ^= t;
    rng_state[3] = (rng_state[3] << 45) | (rng_state[3] >> 19);
    return (double)(r >> 11) * 0x1.0p-53;
}

/* ---------- Test fixture ---------- */
typedef struct {
    int n;
    double* AD;     /* original diagonal */
    double* AE;     /* original off-diagonal */
    double* D;      /* diagonal (overwritten by dstev) */
    double* E;      /* off-diagonal (overwritten by dstev) */
    double* D2;     /* second copy for JOBZ='N' */
    double* E2;     /* second copy for JOBZ='N' */
    double* Z;      /* eigenvectors (n x n) */
    double* work;   /* workspace: max(2*(n-1) for dstev, n*(n+1) for dstt21) */
    double* result; /* dstt21 results (2) */
    uint64_t seed;
} dstev_fixture_t;

/* Global seed for reproducibility */
static uint64_t g_seed = 7331;

static int dstev_setup(void** state, int n)
{
    dstev_fixture_t* fix = malloc(sizeof(dstev_fixture_t));
    assert_non_null(fix);

    fix->n = n;
    fix->seed = g_seed++;

    fix->AD = malloc(n * sizeof(double));
    fix->AE = malloc((n > 1 ? n - 1 : 1) * sizeof(double));
    fix->D = malloc(n * sizeof(double));
    fix->E = malloc((n > 1 ? n - 1 : 1) * sizeof(double));
    fix->D2 = malloc(n * sizeof(double));
    fix->E2 = malloc((n > 1 ? n - 1 : 1) * sizeof(double));
    fix->Z = malloc(n * n * sizeof(double));

    /* workspace: max(2*(n-1), n*(n+1)) */
    int work_stev = (n > 1) ? 2 * (n - 1) : 1;
    int work_stt21 = n * (n + 1);
    int work_stech = 2 * n;
    int work_sz = work_stev;
    if (work_stt21 > work_sz) work_sz = work_stt21;
    if (work_stech > work_sz) work_sz = work_stech;
    fix->work = malloc(work_sz * sizeof(double));

    fix->result = malloc(2 * sizeof(double));

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
static void gen_zero(int n, double* D, double* E)
{
    for (int i = 0; i < n; i++) D[i] = 0.0;
    for (int i = 0; i < n - 1; i++) E[i] = 0.0;
}

/* Type 2: Identity (D=1, E=0) */
static void gen_identity(int n, double* D, double* E)
{
    for (int i = 0; i < n; i++) D[i] = 1.0;
    for (int i = 0; i < n - 1; i++) E[i] = 0.0;
}

/* Type 3: 1-2-1 Toeplitz (D=2, E=1) */
static void gen_toeplitz_121(int n, double* D, double* E)
{
    for (int i = 0; i < n; i++) D[i] = 2.0;
    for (int i = 0; i < n - 1; i++) E[i] = 1.0;
}

/* Type 4: Wilkinson (D[i]=|i - n/2|, E[i]=1) */
static void gen_wilkinson(int n, double* D, double* E)
{
    for (int i = 0; i < n; i++) {
        D[i] = fabs((double)i - (double)(n / 2));
    }
    for (int i = 0; i < n - 1; i++) E[i] = 1.0;
}

/* Type 5: Random symmetric tridiagonal */
static void gen_random(int n, double* D, double* E, uint64_t seed)
{
    rng_seed(seed);
    for (int i = 0; i < n; i++) {
        D[i] = 2.0 * rng_uniform() - 1.0;
    }
    for (int i = 0; i < n - 1; i++) {
        E[i] = 2.0 * rng_uniform() - 1.0;
    }
}

/* Type 6: Graded diagonal (D[i]=2^(-i), E[i]=1) */
static void gen_graded(int n, double* D, double* E)
{
    for (int i = 0; i < n; i++) {
        D[i] = pow(2.0, -(double)i);
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
    case 5: gen_random(n, fix->AD, fix->AE, fix->seed + (uint64_t)imat); break;
    case 6: gen_graded(n, fix->AD, fix->AE); break;
    }

    /* Copy to working arrays */
    memcpy(fix->D, fix->AD, n * sizeof(double));
    if (n > 1) {
        memcpy(fix->E, fix->AE, (n - 1) * sizeof(double));
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
    double ulp = dlamch("P");

    for (int imat = 1; imat <= 6; imat++) {
        gen_matrix(fix, imat);

        /* Get reference eigenvalues via JOBZ='V' */
        dstev("V", n, fix->D, fix->E, fix->Z, n, fix->work, &info);
        assert_info_success(info);

        /* Re-generate and compute with JOBZ='N' */
        gen_matrix(fix, imat);
        memcpy(fix->D2, fix->D, n * sizeof(double));
        if (n > 1) {
            memcpy(fix->E2, fix->E, (n - 1) * sizeof(double));
        }

        /* First call overwrites D with eigenvalues */
        dstev("V", n, fix->D, fix->E, fix->Z, n, fix->work, &info);
        assert_info_success(info);

        /* Now call JOBZ='N' on the copy */
        dstev("N", n, fix->D2, fix->E2, NULL, 1, fix->work, &info);
        assert_info_success(info);

        /* Compare eigenvalues: max|D[i]-D2[i]| / (n * ulp * max|D|) */
        double maxd = 0.0;
        for (int i = 0; i < n; i++) {
            double ad = fabs(fix->D[i]);
            if (ad > maxd) maxd = ad;
        }

        double maxdiff = 0.0;
        for (int i = 0; i < n; i++) {
            double diff = fabs(fix->D[i] - fix->D2[i]);
            if (diff > maxdiff) maxdiff = diff;
        }

        double resid;
        if (maxd == 0.0) {
            resid = maxdiff / ulp;
        } else {
            resid = maxdiff / ((double)n * ulp * maxd);
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
