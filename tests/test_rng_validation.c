/**
 * @file test_rng_validation.c
 * @brief Validates correctness of the xoshiro256+ RNG in test_rng.h.
 *
 * Tests cover:
 *   1. Reproducibility (deterministic sequences with hardcoded reference values)
 *   2. Different seeds produce different sequences
 *   3. Uniform(0,1) range and statistics (mean, variance, chi-square)
 *   4. Uniform(-1,1) range and statistics
 *   5. Normal(0,1) statistics (mean, variance, symmetry)
 *   6. rng_fill / rng_dist consistency (batch vs scalar equivalence)
 *   7. SplitMix64 seed avalanche quality
 *   8. Non-degeneracy (no absorbing states, edge-case seeds)
 */

#include "test_harness.h"
#include "test_rng.h"

#define THRESH 30.0  /* unused but required by test_harness.h macros */

/* ========================================================================= */
/* Test 1: Reproducibility — same seed always produces the exact same        */
/*         sequence, verified against hardcoded reference values.            */
/* ========================================================================= */
static void test_reproducibility(void** cmocka_state) {
    (void)cmocka_state;

    uint64_t state[4];
    rng_seed(state, 42);

    /* Verify seed expansion matches reference SplitMix64 output */
    assert_true(state[0] == 0xBDD732262FEB6E95ULL);
    assert_true(state[1] == 0x28EFE333B266F103ULL);
    assert_true(state[2] == 0x47526757130F9F52ULL);
    assert_true(state[3] == 0x581CE1FF0E4AE394ULL);

    /* Verify first 10 xoshiro256+ outputs match reference */
    static const uint64_t expected[10] = {
        0x15F414253E365229ULL,
        0x4F771F08F4211387ULL,
        0x100492BD8828891EULL,
        0x4E743FCE495374AEULL,
        0x0002D0BAE53F7541ULL,
        0x4D95B0309B62834AULL,
        0x166D954E9D491EF0ULL,
        0x3A1EE212EB52573BULL,
        0xDCE029EA733F8136ULL,
        0x85F3F89092A19882ULL,
    };

    for (int i = 0; i < 10; i++) {
        uint64_t val = rng_next(state);
        assert_true(val == expected[i]);
    }

    /* Re-seed with same value: two independent streams must be identical */
    uint64_t s1[4], s2[4];
    rng_seed(s1, 42);
    rng_seed(s2, 42);
    for (int i = 0; i < 1000; i++) {
        assert_true(rng_next(s1) == rng_next(s2));
    }
}

/* ========================================================================= */
/* Test 2: Different seeds produce different sequences.                      */
/* ========================================================================= */
static void test_different_seeds(void** cmocka_state) {
    (void)cmocka_state;

    uint64_t sa[4], sb[4];
    rng_seed(sa, 1);
    rng_seed(sb, 2);

    int differ = 0;
    for (int i = 0; i < 10; i++) {
        if (rng_next(sa) != rng_next(sb)) differ = 1;
    }
    assert_true(differ);
}

/* ========================================================================= */
/* Test 3: Uniform(0,1) range and basic statistics.                          */
/*         Checks: strict (0,1), mean ~0.5, variance ~1/12, chi-square.     */
/* ========================================================================= */
static void test_uniform_stats(void** cmocka_state) {
    (void)cmocka_state;

    const int N = 100000;
    const int K = 100;
    uint64_t state[4];
    rng_seed(state, 12345);

    f64 sum = 0.0, sum2 = 0.0;
    int bins[100];
    memset(bins, 0, sizeof(bins));

    for (int i = 0; i < N; i++) {
        f64 u = rng_uniform(state);
        assert_true(u > 0.0 && u < 1.0);
        sum += u;
        sum2 += u * u;
        int bin = (int)(u * K);
        if (bin >= K) bin = K - 1;
        bins[bin]++;
    }

    f64 mean = sum / N;
    f64 var = sum2 / N - mean * mean;

    /* For N=100000, std(mean) ~ 1/sqrt(12*N) ~ 0.00091 */
    assert_true(fabs(mean - 0.5) < 0.01);
    /* std(var) ~ 1/(12*sqrt(5*N)) ~ 0.00012 */
    assert_true(fabs(var - 1.0 / 12.0) < 0.005);

    /* Chi-square goodness-of-fit: K=100 bins, expected count = N/K = 1000 */
    f64 expected = (f64)N / K;
    f64 chi2 = 0.0;
    for (int b = 0; b < K; b++) {
        f64 diff = bins[b] - expected;
        chi2 += diff * diff / expected;
    }
    /* chi-square with 99 df: P(chi2 > 150) < 0.0003, P(chi2 < 50) < 0.0001 */
    assert_true(chi2 > 50.0 && chi2 < 180.0);
}

/* ========================================================================= */
/* Test 4: Uniform(-1,1) range and statistics.                               */
/*         Checks: strict (-1,1), mean ~0, variance ~1/3.                   */
/* ========================================================================= */
static void test_uniform_symmetric_stats(void** cmocka_state) {
    (void)cmocka_state;

    const int N = 100000;
    uint64_t state[4];
    rng_seed(state, 54321);

    f64 sum = 0.0, sum2 = 0.0;
    for (int i = 0; i < N; i++) {
        f64 u = rng_uniform_symmetric(state);
        assert_true(u > -1.0 && u < 1.0);
        sum += u;
        sum2 += u * u;
    }

    f64 mean = sum / N;
    f64 var = sum2 / N - mean * mean;

    /* For N=100000, std(mean) ~ 1/sqrt(3*N) ~ 0.0018 */
    assert_true(fabs(mean) < 0.02);
    /* Var[U(-1,1)] = 1/3 */
    assert_true(fabs(var - 1.0 / 3.0) < 0.01);
}

/* ========================================================================= */
/* Test 5: Normal(0,1) statistics.                                           */
/*         Checks: mean ~0, variance ~1, range, symmetry.                   */
/* ========================================================================= */
static void test_normal_stats(void** cmocka_state) {
    (void)cmocka_state;

    const int N = 100000;
    uint64_t state[4];
    rng_seed(state, 99999);

    f64 sum = 0.0, sum2 = 0.0;
    int n_pos = 0;

    for (int i = 0; i < N; i++) {
        f64 x = rng_normal(state);
        /* Sanity: values beyond 10 sigma are vanishingly unlikely */
        assert_true(x > -10.0 && x < 10.0);
        sum += x;
        sum2 += x * x;
        if (x > 0.0) n_pos++;
    }

    f64 mean = sum / N;
    f64 var = sum2 / N - mean * mean;

    /* std(mean) ~ 1/sqrt(N) ~ 0.003 */
    assert_true(fabs(mean) < 0.02);
    /* std(var) ~ sqrt(2/N) ~ 0.004 */
    assert_true(fabs(var - 1.0) < 0.05);
    /* Symmetry: |n_pos - N/2| should be small relative to sqrt(N/4) */
    int sym_dev = n_pos - N / 2;
    if (sym_dev < 0) sym_dev = -sym_dev;
    assert_true(sym_dev < 4 * (int)sqrt((f64)N / 4.0));
}

/* ========================================================================= */
/* Test 6: rng_fill / rng_dist consistency.                                  */
/*         Batch fill must produce identical values to scalar loop.          */
/* ========================================================================= */
static void test_fill_consistency(void** cmocka_state) {
    (void)cmocka_state;

    for (int idist = 1; idist <= 3; idist++) {
        uint64_t s1[4], s2[4];
        rng_seed(s1, 777 + (uint64_t)idist);
        rng_seed(s2, 777 + (uint64_t)idist);

        f64 x_fill[50], x_loop[50];
        rng_fill(s1, idist, 50, x_fill);
        for (int i = 0; i < 50; i++) {
            x_loop[i] = rng_dist(s2, idist);
        }

        for (int i = 0; i < 50; i++) {
            assert_true(x_fill[i] == x_loop[i]);  /* exact bitwise equality */
        }

        /* State must also be identical after same number of draws */
        for (int k = 0; k < 4; k++) {
            assert_true(s1[k] == s2[k]);
        }
    }
}

/* ========================================================================= */
/* Test 7: SplitMix64 seed avalanche quality.                                */
/*         Seeds differing by 1 bit must produce very different states.      */
/* ========================================================================= */
static void test_seed_avalanche(void** cmocka_state) {
    (void)cmocka_state;

    for (int bit = 0; bit < 64; bit++) {
        uint64_t s1[4], s2[4];
        uint64_t seed1 = 1ULL << bit;
        uint64_t seed2 = seed1 ^ 1ULL;  /* flip bit 0 */

        /* Skip if seeds are identical (bit=0 case: 1^1=0 vs 1) */
        if (seed1 == seed2) continue;

        rng_seed(s1, seed1);
        rng_seed(s2, seed2);

        /* Hamming distance between 256-bit states */
        int hamming = 0;
        for (int k = 0; k < 4; k++) {
            uint64_t diff = s1[k] ^ s2[k];
            /* Portable popcount */
            while (diff) {
                hamming++;
                diff &= diff - 1;
            }
        }
        /* SplitMix64 has excellent avalanche: expect ~128 bits different */
        assert_true(hamming > 64);
    }
}

/* ========================================================================= */
/* Test 8: Non-degeneracy — no absorbing states for edge-case seeds.         */
/*         All-zeros is the absorbing state for xoshiro256+; verify          */
/*         rng_seed never produces it and iteration doesn't reach it.        */
/* ========================================================================= */
static void test_no_absorbing_state(void** cmocka_state) {
    (void)cmocka_state;

    uint64_t edge_seeds[] = {0, 1, UINT64_MAX, 0xDEADBEEFULL, 42};
    int n_seeds = (int)(sizeof(edge_seeds) / sizeof(edge_seeds[0]));

    for (int i = 0; i < n_seeds; i++) {
        uint64_t state[4];
        rng_seed(state, edge_seeds[i]);

        /* At least one state element must be nonzero after seeding */
        assert_true(state[0] | state[1] | state[2] | state[3]);

        /* After 1000 iterations, state should still be nonzero */
        for (int j = 0; j < 1000; j++) rng_next(state);
        assert_true(state[0] | state[1] | state[2] | state[3]);

        /* And output should be nonzero (at least sometimes) */
        int any_nonzero = 0;
        for (int j = 0; j < 10; j++) {
            if (rng_next(state) != 0) any_nonzero = 1;
        }
        assert_true(any_nonzero);
    }

    /* Verify seed=0 produces a specific known state */
    uint64_t state[4];
    rng_seed(state, 0);
    assert_true(state[0] == 0xE220A8397B1DCDAFULL);
    assert_true(state[1] == 0x6E789E6AA1B965F4ULL);
    assert_true(state[2] == 0x06C45D188009454FULL);
    assert_true(state[3] == 0xF88BB8A8724C81ECULL);
}

/* ========================================================================= */
/* Test 9: Cross-validate with known xoshiro256+ state transitions.          */
/*         Use trivial state {1,2,3,4} and verify exact step-by-step         */
/*         output and state evolution.                                       */
/* ========================================================================= */
static void test_xoshiro_reference(void** cmocka_state) {
    (void)cmocka_state;

    uint64_t s[4] = {1, 2, 3, 4};

    /* First call: result = s[0]+s[3] = 1+4 = 5 */
    uint64_t val1 = rng_next(s);
    assert_true(val1 == 5);

    /* Verify state after first step */
    assert_true(s[0] == 0x0000000000000007ULL);
    assert_true(s[1] == 0x0000000000000000ULL);
    assert_true(s[2] == 0x0000000000040002ULL);
    assert_true(s[3] == 0x0000C00000000000ULL);

    /* Second call */
    uint64_t val2 = rng_next(s);
    assert_true(val2 == 0x0000C00000000007ULL);

    /* Verify state after second step */
    assert_true(s[0] == 0x0000C00000000007ULL);
    assert_true(s[1] == 0x0000000000040005ULL);
    assert_true(s[2] == 0x0000000000040005ULL);
    assert_true(s[3] == 0x0000000018000000ULL);

    /* Third call */
    uint64_t val3 = rng_next(s);
    assert_true(val3 == 0x0000C00018000007ULL);
}

/* ========================================================================= */
/* Main: register all tests                                                  */
/* ========================================================================= */
int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_reproducibility),
        cmocka_unit_test(test_different_seeds),
        cmocka_unit_test(test_uniform_stats),
        cmocka_unit_test(test_uniform_symmetric_stats),
        cmocka_unit_test(test_normal_stats),
        cmocka_unit_test(test_fill_consistency),
        cmocka_unit_test(test_seed_avalanche),
        cmocka_unit_test(test_no_absorbing_state),
        cmocka_unit_test(test_xoshiro_reference),
    };
    return cmocka_run_group_tests_name("rng_validation", tests, NULL, NULL);
}
