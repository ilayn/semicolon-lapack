/**
 * @file test_schkbl.c
 * @brief Tests SGEBAL, a routine for balancing a general real matrix.
 *
 * Port of LAPACK TESTING/EIG/dchkbl.f with embedded test data from
 * TESTING/dbal.in. All test matrices and expected results are hardcoded
 * as static arrays.
 */

#include "test_harness.h"
#include <math.h>
#include <string.h>

#define THRESH 30.0f

typedef float f32;

extern f32 slamch(const char* cmach);
extern f32 slange(const char* norm, const int m, const int n,
                  const f32* A, const int lda, f32* work);
extern void sgebal(const char* job, const int n, f32* A, const int lda,
                   int* ilo, int* ihi, f32* scale, int* info);

#define LDA 20

/* ---------- helpers ---------- */

static void rowmajor_to_colmajor(const f32* rm, f32* cm, int n, int ld)
{
    memset(cm, 0, (size_t)ld * n * sizeof(f32));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            cm[i + j * ld] = rm[i * n + j];
}

/* ---------- test case data from TESTING/dbal.in ---------- */

typedef struct {
    int n;
    int iloin;
    int ihiin;
    const f32* a_rm;
    const f32* ain_rm;
    const f32* scalin;
} dbal_case_t;

/* Case 0: N=5 diagonal */
static const f32 c0_a[] = {
    1, 0, 0, 0, 0,
    0, 2, 0, 0, 0,
    0, 0, 3, 0, 0,
    0, 0, 0, 4, 0,
    0, 0, 0, 0, 5,
};
static const f32 c0_ain[] = {
    1, 0, 0, 0, 0,
    0, 2, 0, 0, 0,
    0, 0, 3, 0, 0,
    0, 0, 0, 4, 0,
    0, 0, 0, 0, 5,
};
static const f32 c0_s[] = {1, 1, 2, 3, 4};

/* Case 1: N=5 lower triangular */
static const f32 c1_a[] = {
    1, 0, 0, 0, 0,
    1, 2, 0, 0, 0,
    1, 2, 3, 0, 0,
    1, 2, 3, 4, 0,
    1, 2, 3, 4, 5,
};
static const f32 c1_ain[] = {
    5, 4, 3, 2, 1,
    0, 4, 3, 2, 1,
    0, 0, 3, 2, 1,
    0, 0, 0, 2, 1,
    0, 0, 0, 0, 1,
};
static const f32 c1_s[] = {1, 1, 2, 1, 0};

/* Case 2: N=5 sub-diagonal */
static const f32 c2_a[] = {
    1, 0, 0, 0, 0,
    1, 1, 0, 0, 0,
    0, 1, 1, 0, 0,
    0, 0, 1, 1, 0,
    0, 0, 0, 1, 1,
};
static const f32 c2_ain[] = {
    1, 1, 0, 0, 0,
    0, 1, 1, 0, 0,
    0, 0, 1, 1, 0,
    0, 0, 0, 1, 1,
    0, 0, 0, 0, 1,
};
static const f32 c2_s[] = {1, 1, 2, 1, 0};

/* Case 3: N=4 */
static const f32 c3_a[] = {
    0, 2, 0.1f, 0,
    2, 0, 0,   0.1f,
    100, 0, 0, 2,
    0, 100, 2, 0,
};
static const f32 c3_ain[] = {
    0,    2.0f,  3.2f,  0,
    2.0f,  0,    0,    3.2f,
    3.125f, 0,   0,    2.0f,
    0,    3.125f, 2.0f, 0,
};
static const f32 c3_s[] = {0.0625f, 0.0625f, 2.0f, 2.0f};

/* Case 4: N=6 */
static const f32 c4_a[] = {
    2,   0,   0,        0,     0,      1024,
    0,   0,   0,        0,     0,      128,
    0,   2,   3000,     0,     0,      2,
    128, 4,   0.004f,    5,     600,    8,
    0,   0,   0,        0,     0.002f,  2,
    8,   8192, 0,       0,     0,      2,
};
static const f32 c4_ain[] = {
    5,   0.004f,  600,   1024,  0.5f,     8,
    0,   3000,   0,     0,     0.25f,    2,
    0,   0,      0.002f, 0,     0,       2,
    0,   0,      0,     2,     0,       128,
    0,   0,      0,     0,     0,       1024,
    0,   0,      0,     64,    1024,    2,
};
static const f32 c4_s[] = {3, 2, 4, 8, 0.125f, 1};

/* Case 5: N=5 */
static const f32 c5_a[] = {
    1,       0,    0,     0,       8,
    0,       2,    8192,  2,       4,
    2.5e-4f,  1.25e-4f, 4, 0,       64,
    0,       2,    1024,  4,       8,
    0,       8192, 0,     0,       8,
};
static const f32 c5_ain[] = {
    1.0f,     0,       0,       0,       0.25f,
    0,       2.0f,     1024.0f,  16.0f,    16.0f,
    0.256f,   0.001f,   4.0f,     0,       2048.0f,
    0,       0.25f,    16.0f,    4.0f,     4.0f,
    0,       2048.0f,  0,       0,       8.0f,
};
static const f32 c5_s[] = {64, 0.5f, 0.0625f, 4, 2};

/* Case 6: N=4 */
static const f32 c6_a[] = {
    1,     1e6,   1e6,   1e6,
    -2e6,  3,     2e-6,  3e-6,
    -3e6,  0,     1e-6,  2,
    1e6,   0,     3e-6,  4e6,
};
static const f32 c6_ain[] = {
    1.0f,    1e6,   2e6,   1e6,
    -2e6,   3.0f,   4e-6,  3e-6,
    -1.5e6f, 0,     1e-6,  1.0f,
    1e6,    0,     6e-6,  4e6,
};
static const f32 c6_s[] = {1, 1, 2, 1};

/* Case 7: N=4 */
static const f32 c7_a[] = {
    1,     1e4,  1e4,   1e4,
    -2e4,  3,    2e-3,  3e-3,
    0,     2,    0,     -3e4,
    0,     0,    1e4,   0,
};
static const f32 c7_ain[] = {
    1.0f,    1e4,   1e4,   5e3,
    -2e4,   3.0f,   2e-3,  1.5e-3f,
    0,      2.0f,   0,     -1.5e4f,
    0,      0,     2e4,   0,
};
static const f32 c7_s[] = {1, 1, 1, 0.5f};

/* Case 8: N=5 */
static const f32 c8_a[] = {
    1,   512,    4096,    32768,     262144,
    8,   0,      0,       0,         0,
    0,   8,      0,       0,         0,
    0,   0,      8,       0,         0,
    0,   0,      0,       8,         0,
};
static const f32 c8_ain[] = {
    1.0f,     32.0f,   32.0f,   32.0f,   32.0f,
    128.0f,   0,      0,      0,      0,
    0,       64.0f,   0,      0,      0,
    0,       0,      64.0f,   0,      0,
    0,       0,      0,      64.0f,   0,
};
static const f32 c8_s[] = {256, 16, 2, 0.25f, 0.03125f};

/* Case 9: N=6 with isolation */
static const f32 c9_a[] = {
    1, 1, 0, 1, 1, 1,
    1, 1, 0, 1, 1, 1,
    1, 1, 1, 1, 1, 1,
    0, 0, 0, 1, 0, 0,
    1, 1, 0, 1, 1, 1,
    1, 1, 0, 1, 1, 1,
};
static const f32 c9_ain[] = {
    1, 1, 1, 1, 1, 1,
    0, 1, 1, 1, 1, 1,
    0, 1, 1, 1, 1, 1,
    0, 1, 1, 1, 1, 1,
    0, 1, 1, 1, 1, 1,
    0, 0, 0, 0, 0, 1,
};
static const f32 c9_s[] = {2, 1, 1, 1, 1, 3};

/* Case 10: N=7 */
static const f32 c10_a[] = {
    6,  0,     0,       0,        0,     1,   0,
    0,  4,     0,       2.5e-4f,   0.0125f, 0.02f, 0.125f,
    1,  128,   64,      0,        0,     -2,   16,
    0,  16384, 0,       1,       -400,   256, -4000,
    -2, -256,  0,       0.0125f,   2,     2,   32,
    0,  0,     0,       0,        0,     0,   0,
    0,  8,     0,       0.004f,    0.125f, -0.2f, 3,
};
static const f32 c10_ain[] = {
    64,   0.25f,   0.5f,     0,      0,      1.0f,    -2.0f,
    0,    4.0f,    2.0f,     4.096f,  1.6f,    0,       10.24f,
    0,    0.5f,    3.0f,     4.096f,  1.0f,    0,      -6.4f,
    0,    1.0f,   -3.90625f, 1.0f,   -3.125f,  0,       8.0f,
    0,   -2.0f,    4.0f,     1.6f,    2.0f,   -8.0f,    8.0f,
    0,    0,      0,       0,      0,      6.0f,     1.0f,
    0,    0,      0,       0,      0,      0,       0,
};
static const f32 c10_s[] = {2, 1.953125e-3f, 0.03125f, 32, 0.25f, 0, 5};

/* Case 11: N=5 */
static const f32 c11_a[] = {
    1000,  2,     3,    4,     5e5,
    9,     0,     2e-4, 1,     3,
    0,    -300,   2,    1,     1,
    9,     0.002f, 1,    1,    -1000,
    6,     200,   1,    600,   3,
};
static const f32 c11_ain[] = {
    1000,     0.03125f,  0.375f,    0.0625f,     3906.25f,
    576,      0,        1.6e-3f,   1.0f,        1.5f,
    0,       -37.5f,     2.0f,      0.125f,      0.0625f,
    576,      2e-3f,     8.0f,      1.0f,       -500,
    768,      400,      16,       1200,        3.0f,
};
static const f32 c11_s[] = {128, 2.0f, 16, 2.0f, 1.0f};

/* Case 12: N=5 extreme magnitudes */
static const f32 c12_a[] = {
    1,     1e15f,  0,      0,      0,
    1e-15f, 1,     1e15f,  0,      0,
    0,     1e-15f, 1,      1e15f,  0,
    0,     0,      1e-15f, 1,      1e15f,
    0,     0,      0,      1e-15f, 1,
};
static const f32 c12_ain[] = {
    1.0f,          7.1054273f,    0,             0,             0,
    0.14073749f,   1.0f,          3.5527136f,    0,             0,
    0,             0.28147498f,   1.0f,          1.7763568f,    0,
    0,             0,             0.56294996f,   1.0f,          0.88817841f,
    0,             0,             0,             1.1258999f,    1.0f,
};
static const f32 c12_s[] = {
    5.0706024e30f,
    3.6028797e16f,
    1.28e2f,
    2.2737368e-13f,
    2.0194839e-28f,
};

static const dbal_case_t cases[] = {
    { 5, 0, 0, c0_a, c0_ain, c0_s },
    { 5, 0, 0, c1_a, c1_ain, c1_s },
    { 5, 0, 0, c2_a, c2_ain, c2_s },
    { 4, 0, 3, c3_a, c3_ain, c3_s },
    { 6, 3, 5, c4_a, c4_ain, c4_s },
    { 5, 0, 4, c5_a, c5_ain, c5_s },
    { 4, 0, 3, c6_a, c6_ain, c6_s },
    { 4, 0, 3, c7_a, c7_ain, c7_s },
    { 5, 0, 4, c8_a, c8_ain, c8_s },
    { 6, 1, 4, c9_a, c9_ain, c9_s },
    { 7, 1, 4, c10_a, c10_ain, c10_s },
    { 5, 0, 4, c11_a, c11_ain, c11_s },
    { 5, 0, 4, c12_a, c12_ain, c12_s },
};

#define NCASES ((int)(sizeof(cases) / sizeof(cases[0])))

/* ---------- test ---------- */

static void test_dgebal(void** state)
{
    (void)state;

    f32 sfmin = slamch("S");

    f32 a[LDA * LDA], ain[LDA * LDA];
    f32 scale[LDA];
    int ilo, ihi, info;
    f32 rmax = 0.0f, vmax;
    int ninfo = 0, knt = 0;
    int lmax_info = 0, lmax_idx = 0, lmax_resid = 0;

    for (int tc = 0; tc < NCASES; tc++) {
        const dbal_case_t* c = &cases[tc];
        int n = c->n;
        int iloin = c->iloin;
        int ihiin = c->ihiin;

        rowmajor_to_colmajor(c->a_rm, a, n, LDA);
        rowmajor_to_colmajor(c->ain_rm, ain, n, LDA);

        knt++;

        sgebal("B", n, a, LDA, &ilo, &ihi, scale, &info);

        if (info != 0) {
            ninfo++;
            lmax_info = knt;
        }

        if (ilo != iloin || ihi != ihiin) {
            ninfo++;
            lmax_idx = knt;
            print_message("Case %d: ilo/ihi mismatch: got (%d,%d) expected (%d,%d)\n",
                          tc, ilo, ihi, iloin, ihiin);
        }

        /* Compare balanced matrix */
        vmax = 0.0f;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                f32 aij = a[i + j * LDA];
                f32 ainij = ain[i + j * LDA];
                f32 temp = fabsf(aij);
                if (fabsf(ainij) > temp) temp = fabsf(ainij);
                if (sfmin > temp) temp = sfmin;
                f32 diff = fabsf(aij - ainij) / temp;
                if (diff > vmax) vmax = diff;
            }
        }

        /* Compare scale factors */
        for (int i = 0; i < n; i++) {
            f32 si = scale[i];
            f32 ei = c->scalin[i];
            f32 temp = fabsf(si);
            if (fabsf(ei) > temp) temp = fabsf(ei);
            if (sfmin > temp) temp = sfmin;
            f32 diff = fabsf(si - ei) / temp;
            if (diff > vmax) vmax = diff;
        }

        if (vmax > rmax) {
            lmax_resid = knt;
            rmax = vmax;
        }
    }

    print_message("SGEBAL: %d cases, max residual = %.3e (case %d)\n",
                  knt, (double)rmax, lmax_resid);
    if (ninfo > 0)
        print_message("  INFO/index errors: %d (info case %d, idx case %d)\n",
                      ninfo, lmax_info, lmax_idx);

    assert_true(ninfo == 0);
    assert_residual_ok(rmax);
}

int main(void)
{
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_dgebal),
    };
    return cmocka_run_group_tests_name("dchkbl", tests, NULL, NULL);
}
