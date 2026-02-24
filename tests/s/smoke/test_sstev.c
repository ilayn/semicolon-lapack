/**
 * @file test_sstev.c
 * @brief CMocka test suite for sstev (symmetric tridiagonal eigenvalue driver).
 *
 * Tests the simple eigenvalue driver sstev which computes all eigenvalues
 * and, optionally, eigenvectors of a real symmetric tridiagonal matrix.
 *
 * Verification:
 *   - sstt21: checks ||U S U' - A|| / (n * ||A|| * eps) and ||U U' - I|| / (n * eps)
 *   - sstech: Sturm sequence verification of eigenvalues
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
#include "verify.h"
#include "test_rng.h"

/* Test threshold - see LAPACK dtest.in */
#define THRESH 20.0f
/* Routine under test */
/* Verification routines */
/* Utilities */
/* ---------- Test fixture ---------- */
typedef struct {
    INT n;
    f32* AD;     /* original diagonal */
    f32* AE;     /* original off-diagonal */
    f32* D;      /* diagonal (overwritten by sstev) */
    f32* E;      /* off-diagonal (overwritten by sstev) */
    f32* D2;     /* second copy for JOBZ='N' */
    f32* E2;     /* second copy for JOBZ='N' */
    f32* Z;      /* eigenvectors (n x n) */
    f32* work;   /* workspace: max(2*(n-1) for sstev, n*(n+1) for sstt21) */
    f32* result; /* sstt21 results (2) */
    uint64_t seed;
    uint64_t rng_state[4];
} dstev_fixture_t;

/* Global seed for reproducibility */
static uint64_t g_seed = 7331;

static int dstev_setup(void** state, INT n)
{
    dstev_fixture_t* fix = malloc(sizeof(dstev_fixture_t));
    assert_non_null(fix);

    fix->n = n;
    fix->seed = g_seed++;

    fix->AD = malloc(n * sizeof(f32));
    fix->AE = malloc((n > 1 ? n - 1 : 1) * sizeof(f32));
    fix->D = malloc(n * sizeof(f32));
    fix->E = malloc((n > 1 ? n - 1 : 1) * sizeof(f32));
    fix->D2 = malloc(n * sizeof(f32));
    fix->E2 = malloc((n > 1 ? n - 1 : 1) * sizeof(f32));
    fix->Z = malloc(n * n * sizeof(f32));

    /* workspace: max(2*(n-1), n*(n+1)) */
    INT work_stev = (n > 1) ? 2 * (n - 1) : 1;
    INT work_stt21 = n * (n + 1);
    INT work_stech = 2 * n;
    INT work_sz = work_stev;
    if (work_stt21 > work_sz) work_sz = work_stt21;
    if (work_stech > work_sz) work_sz = work_stech;
    fix->work = malloc(work_sz * sizeof(f32));

    fix->result = malloc(2 * sizeof(f32));

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
static void gen_zero(INT n, f32* D, f32* E)
{
    for (INT i = 0; i < n; i++) D[i] = 0.0f;
    for (INT i = 0; i < n - 1; i++) E[i] = 0.0f;
}

/* Type 2: Identity (D=1, E=0) */
static void gen_identity(INT n, f32* D, f32* E)
{
    for (INT i = 0; i < n; i++) D[i] = 1.0f;
    for (INT i = 0; i < n - 1; i++) E[i] = 0.0f;
}

/* Type 3: 1-2-1 Toeplitz (D=2, E=1) */
static void gen_toeplitz_121(INT n, f32* D, f32* E)
{
    for (INT i = 0; i < n; i++) D[i] = 2.0f;
    for (INT i = 0; i < n - 1; i++) E[i] = 1.0f;
}

/* Type 4: Wilkinson (D[i]=|i - n/2|, E[i]=1) */
static void gen_wilkinson(INT n, f32* D, f32* E)
{
    for (INT i = 0; i < n; i++) {
        D[i] = fabsf((f32)i - (f32)(n / 2));
    }
    for (INT i = 0; i < n - 1; i++) E[i] = 1.0f;
}

/* Type 5: Random symmetric tridiagonal */
static void gen_random(INT n, f32* D, f32* E, uint64_t state[static 4])
{
    for (INT i = 0; i < n; i++) {
        D[i] = rng_uniform_symmetric_f32(state);
    }
    for (INT i = 0; i < n - 1; i++) {
        E[i] = rng_uniform_symmetric_f32(state);
    }
}

/* Type 6: Graded diagonal (D[i]=2^(-i), E[i]=1) */
static void gen_graded(INT n, f32* D, f32* E)
{
    for (INT i = 0; i < n; i++) {
        D[i] = powf(2.0f, -(f32)i);
    }
    for (INT i = 0; i < n - 1; i++) E[i] = 1.0f;
}

/**
 * Generate test matrix of given type and copy to working arrays.
 */
static void gen_matrix(dstev_fixture_t* fix, INT imat)
{
    INT n = fix->n;

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
    memcpy(fix->D, fix->AD, n * sizeof(f32));
    if (n > 1) {
        memcpy(fix->E, fix->AE, (n - 1) * sizeof(f32));
    }
}

/* ---------- Test functions ---------- */

/**
 * JOBZ='V': compute eigenvalues and eigenvectors, verify with sstt21.
 */
static void test_jobz_V(void** state)
{
    dstev_fixture_t* fix = *state;
    INT n = fix->n;
    INT info;

    for (INT imat = 1; imat <= 6; imat++) {
        gen_matrix(fix, imat);

        sstev("V", n, fix->D, fix->E, fix->Z, n, fix->work, &info);
        assert_info_success(info);

        /* Verify: sstt21 checks U S U' = A and U U' = I */
        fix->result[0] = 0.0f;
        fix->result[1] = 0.0f;
        sstt21(n, 0, fix->AD, fix->AE, fix->D, NULL, fix->Z, n,
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
    INT n = fix->n;
    INT info;
    f32 ulp = slamch("P");

    for (INT imat = 1; imat <= 6; imat++) {
        gen_matrix(fix, imat);

        /* Get reference eigenvalues via JOBZ='V' */
        sstev("V", n, fix->D, fix->E, fix->Z, n, fix->work, &info);
        assert_info_success(info);

        /* Re-generate and compute with JOBZ='N' */
        gen_matrix(fix, imat);
        memcpy(fix->D2, fix->D, n * sizeof(f32));
        if (n > 1) {
            memcpy(fix->E2, fix->E, (n - 1) * sizeof(f32));
        }

        /* First call overwrites D with eigenvalues */
        sstev("V", n, fix->D, fix->E, fix->Z, n, fix->work, &info);
        assert_info_success(info);

        /* Now call JOBZ='N' on the copy */
        sstev("N", n, fix->D2, fix->E2, NULL, 1, fix->work, &info);
        assert_info_success(info);

        /* Compare eigenvalues: max|D[i]-D2[i]| / (n * ulp * max|D|) */
        f32 maxd = 0.0f;
        for (INT i = 0; i < n; i++) {
            f32 ad = fabsf(fix->D[i]);
            if (ad > maxd) maxd = ad;
        }

        f32 maxdiff = 0.0f;
        for (INT i = 0; i < n; i++) {
            f32 diff = fabsf(fix->D[i] - fix->D2[i]);
            if (diff > maxdiff) maxdiff = diff;
        }

        f32 resid;
        if (maxd == 0.0f) {
            resid = maxdiff / ulp;
        } else {
            resid = maxdiff / ((f32)n * ulp * maxd);
        }
        assert_residual_ok(resid);
    }
}

/**
 * Sturm sequence verification of eigenvalues via sstech.
 */
static void test_sturm(void** state)
{
    dstev_fixture_t* fix = *state;
    INT n = fix->n;
    INT info;

    for (INT imat = 1; imat <= 6; imat++) {
        /* Skip zero matrix for Sturm test (all zero eigenvalues are trivially correct) */
        if (imat == 1) continue;

        gen_matrix(fix, imat);

        sstev("V", n, fix->D, fix->E, fix->Z, n, fix->work, &info);
        assert_info_success(info);

        /* sstech verifies eigenvalues via Sturm sequences */
        INT stech_info;
        sstech(n, fix->AD, fix->AE, fix->D, THRESH, fix->work, &stech_info);
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
