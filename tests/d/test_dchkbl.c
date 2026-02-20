/**
 * @file test_dchkbl.c
 * @brief Tests DGEBAL, a routine for balancing a general real matrix.
 *
 * Port of LAPACK TESTING/EIG/dchkbl.f with embedded test data from
 * TESTING/dbal.in. All test matrices and expected results are hardcoded
 * as static arrays.
 */

#include "test_harness.h"
#include <math.h>
#include <string.h>

#define THRESH 30.0

typedef double f64;

extern f64 dlamch(const char* cmach);
extern f64 dlange(const char* norm, const int m, const int n,
                  const f64* A, const int lda, f64* work);
extern void dgebal(const char* job, const int n, f64* A, const int lda,
                   int* ilo, int* ihi, f64* scale, int* info);

#define LDA 20

/* ---------- helpers ---------- */

static void rowmajor_to_colmajor(const f64* rm, f64* cm, int n, int ld)
{
    memset(cm, 0, (size_t)ld * n * sizeof(f64));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            cm[i + j * ld] = rm[i * n + j];
}

/* ---------- test case data from TESTING/dbal.in ---------- */

typedef struct {
    int n;
    int iloin;
    int ihiin;
    const f64* a_rm;
    const f64* ain_rm;
    const f64* scalin;
} dbal_case_t;

/* Case 0: N=5 diagonal */
static const f64 c0_a[] = {
    1, 0, 0, 0, 0,
    0, 2, 0, 0, 0,
    0, 0, 3, 0, 0,
    0, 0, 0, 4, 0,
    0, 0, 0, 0, 5,
};
static const f64 c0_ain[] = {
    1, 0, 0, 0, 0,
    0, 2, 0, 0, 0,
    0, 0, 3, 0, 0,
    0, 0, 0, 4, 0,
    0, 0, 0, 0, 5,
};
static const f64 c0_s[] = {1, 1, 2, 3, 4};

/* Case 1: N=5 lower triangular */
static const f64 c1_a[] = {
    1, 0, 0, 0, 0,
    1, 2, 0, 0, 0,
    1, 2, 3, 0, 0,
    1, 2, 3, 4, 0,
    1, 2, 3, 4, 5,
};
static const f64 c1_ain[] = {
    5, 4, 3, 2, 1,
    0, 4, 3, 2, 1,
    0, 0, 3, 2, 1,
    0, 0, 0, 2, 1,
    0, 0, 0, 0, 1,
};
static const f64 c1_s[] = {1, 1, 2, 1, 0};

/* Case 2: N=5 sub-diagonal */
static const f64 c2_a[] = {
    1, 0, 0, 0, 0,
    1, 1, 0, 0, 0,
    0, 1, 1, 0, 0,
    0, 0, 1, 1, 0,
    0, 0, 0, 1, 1,
};
static const f64 c2_ain[] = {
    1, 1, 0, 0, 0,
    0, 1, 1, 0, 0,
    0, 0, 1, 1, 0,
    0, 0, 0, 1, 1,
    0, 0, 0, 0, 1,
};
static const f64 c2_s[] = {1, 1, 2, 1, 0};

/* Case 3: N=4 */
static const f64 c3_a[] = {
    0, 2, 0.1, 0,
    2, 0, 0,   0.1,
    100, 0, 0, 2,
    0, 100, 2, 0,
};
static const f64 c3_ain[] = {
    0,    2.0,  3.2,  0,
    2.0,  0,    0,    3.2,
    3.125, 0,   0,    2.0,
    0,    3.125, 2.0, 0,
};
static const f64 c3_s[] = {0.0625, 0.0625, 2.0, 2.0};

/* Case 4: N=6 */
static const f64 c4_a[] = {
    2,   0,   0,        0,     0,      1024,
    0,   0,   0,        0,     0,      128,
    0,   2,   3000,     0,     0,      2,
    128, 4,   0.004,    5,     600,    8,
    0,   0,   0,        0,     0.002,  2,
    8,   8192, 0,       0,     0,      2,
};
static const f64 c4_ain[] = {
    5,   0.004,  600,   1024,  0.5,     8,
    0,   3000,   0,     0,     0.25,    2,
    0,   0,      0.002, 0,     0,       2,
    0,   0,      0,     2,     0,       128,
    0,   0,      0,     0,     0,       1024,
    0,   0,      0,     64,    1024,    2,
};
static const f64 c4_s[] = {3, 2, 4, 8, 0.125, 1};

/* Case 5: N=5 */
static const f64 c5_a[] = {
    1,       0,    0,     0,       8,
    0,       2,    8192,  2,       4,
    2.5e-4,  1.25e-4, 4, 0,       64,
    0,       2,    1024,  4,       8,
    0,       8192, 0,     0,       8,
};
static const f64 c5_ain[] = {
    1.0,     0,       0,       0,       2.0,
    0,       2.0,     1024.0,  16.0,    16.0,
    0.032,   0.001,   4.0,     0,       2048.0,
    0,       0.25,    16.0,    4.0,     4.0,
    0,       2048.0,  0,       0,       8.0,
};
static const f64 c5_s[] = {8, 0.5, 0.0625, 4, 2};

/* Case 6: N=4 */
static const f64 c6_a[] = {
    1,     1e6,   1e6,   1e6,
    -2e6,  3,     2e-6,  3e-6,
    -3e6,  0,     1e-6,  2,
    1e6,   0,     3e-6,  4e6,
};
static const f64 c6_ain[] = {
    1.0,    1e6,   2e6,   1e6,
    -2e6,   3.0,   4e-6,  3e-6,
    -1.5e6, 0,     1e-6,  1.0,
    1e6,    0,     6e-6,  4e6,
};
static const f64 c6_s[] = {1, 1, 2, 1};

/* Case 7: N=4 */
static const f64 c7_a[] = {
    1,     1e4,  1e4,   1e4,
    -2e4,  3,    2e-3,  3e-3,
    0,     2,    0,     -3e4,
    0,     0,    1e4,   0,
};
static const f64 c7_ain[] = {
    1.0,    1e4,   1e4,   5e3,
    -2e4,   3.0,   2e-3,  1.5e-3,
    0,      2.0,   0,     -1.5e4,
    0,      0,     2e4,   0,
};
static const f64 c7_s[] = {1, 1, 1, 0.5};

/* Case 8: N=5 */
static const f64 c8_a[] = {
    1,   512,    4096,    32768,     262144,
    8,   0,      0,       0,         0,
    0,   8,      0,       0,         0,
    0,   0,      8,       0,         0,
    0,   0,      0,       8,         0,
};
static const f64 c8_ain[] = {
    1.0,     32.0,   32.0,   32.0,   32.0,
    128.0,   0,      0,      0,      0,
    0,       64.0,   0,      0,      0,
    0,       0,      64.0,   0,      0,
    0,       0,      0,      64.0,   0,
};
static const f64 c8_s[] = {256, 16, 2, 0.25, 0.03125};

/* Case 9: N=6 with isolation */
static const f64 c9_a[] = {
    1, 1, 0, 1, 1, 1,
    1, 1, 0, 1, 1, 1,
    1, 1, 1, 1, 1, 1,
    0, 0, 0, 1, 0, 0,
    1, 1, 0, 1, 1, 1,
    1, 1, 0, 1, 1, 1,
};
static const f64 c9_ain[] = {
    1, 1, 1, 1, 1, 1,
    0, 1, 1, 1, 1, 1,
    0, 1, 1, 1, 1, 1,
    0, 1, 1, 1, 1, 1,
    0, 1, 1, 1, 1, 1,
    0, 0, 0, 0, 0, 1,
};
static const f64 c9_s[] = {2, 1, 1, 1, 1, 3};

/* Case 10: N=7 */
static const f64 c10_a[] = {
    6,  0,     0,       0,        0,     1,   0,
    0,  4,     0,       2.5e-4,   0.0125, 0.02, 0.125,
    1,  128,   64,      0,        0,     -2,   16,
    0,  16384, 0,       1,       -400,   256, -4000,
    -2, -256,  0,       0.0125,   2,     2,   32,
    0,  0,     0,       0,        0,     0,   0,
    0,  8,     0,       0.004,    0.125, -0.2, 3,
};
static const f64 c10_ain[] = {
    64,   1.0,    0.5,     0,      0,      1.0,    -2.0,
    0,    4.0,    0.5,     1.024,  0.8,    0,       2.56,
    0,    2.0,    3.0,     4.096,  2.0,    0,      -6.4,
    0,    4.0,   -3.90625, 1.0,   -6.25,   0,       8.0,
    0,   -4.0,    2.0,     0.8,    2.0,   -4.0,     4.0,
    0,    0,      0,       0,      0,      6.0,     1.0,
    0,    0,      0,       0,      0,      0,       0,
};
static const f64 c10_s[] = {2, 7.8125e-3, 0.03125, 32, 0.5, 0, 5};

/* Case 11: N=5 */
static const f64 c11_a[] = {
    1000,  2,     3,    4,     5e5,
    9,     0,     2e-4, 1,     3,
    0,    -300,   2,    1,     1,
    9,     0.002, 1,    1,    -1000,
    6,     200,   1,    600,   3,
};
static const f64 c11_ain[] = {
    1000,     0.03125,  0.375,    0.03125,    1953.125,
    576,      0,        1.6e-3,   0.5,        0.75,
    0,       -37.5,     2.0,      0.0625,     0.03125,
    1152,     4e-3,     16,       1.0,       -500,
    1536,     800,      32,       1200,       3.0,
};
static const f64 c11_s[] = {32, 0.5, 4, 0.25, 0.125};

/* Case 12: N=6 extreme magnitudes */
static const f64 c12_a[] = {
    1,      1e120,  0,      0,      0,      0,
    1e-120, 1,      1e120,  0,      0,      0,
    0,      1e-120, 1,      1e120,  0,      0,
    0,      0,      1e-120, 1,      1e120,  0,
    0,      0,      0,      1e-120, 1,      1e120,
    0,      0,      0,      0,      1e-120, 1,
};
static const f64 c12_ain[] = {
    1.0, 6.344854593289123e3,  0, 0, 0, 0,
    1.576080247855779e-4, 1.0, 6.344854593289123e3,  0, 0, 0,
    0, 1.576080247855779e-4, 1.0, 3.172427296644561e3,  0, 0,
    0, 0, 3.152160495711558e-4, 1.0, 1.586213648322281e3,  0,
    0, 0, 0, 6.304320991423117e-4, 1.0, 7.931068241611404e2,
    0, 0, 0, 0, 1.260864198284623e-3, 1.0,
};
static const f64 c12_s[] = {
    2.4948003869184e291,
    1.582914569427869e175,
    1.004336277661869e59,
    3.186183822264905e-58,
    5.053968264940244e-175,
    4.008336720018e-292,
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
    { 6, 0, 5, c12_a, c12_ain, c12_s },
};

#define NCASES ((int)(sizeof(cases) / sizeof(cases[0])))

/* ---------- test ---------- */

static void test_dgebal(void** state)
{
    (void)state;

    f64 sfmin = dlamch("S");

    f64 a[LDA * LDA], ain[LDA * LDA];
    f64 scale[LDA];
    int ilo, ihi, info;
    f64 rmax = 0.0, vmax;
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

        dgebal("B", n, a, LDA, &ilo, &ihi, scale, &info);

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
        vmax = 0.0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                f64 aij = a[i + j * LDA];
                f64 ainij = ain[i + j * LDA];
                f64 temp = fabs(aij);
                if (fabs(ainij) > temp) temp = fabs(ainij);
                if (sfmin > temp) temp = sfmin;
                f64 diff = fabs(aij - ainij) / temp;
                if (diff > vmax) vmax = diff;
            }
        }

        /* Compare scale factors */
        for (int i = 0; i < n; i++) {
            f64 si = scale[i];
            f64 ei = c->scalin[i];
            f64 temp = fabs(si);
            if (fabs(ei) > temp) temp = fabs(ei);
            if (sfmin > temp) temp = sfmin;
            f64 diff = fabs(si - ei) / temp;
            if (diff > vmax) vmax = diff;
        }

        if (vmax > rmax) {
            lmax_resid = knt;
            rmax = vmax;
        }
    }

    print_message("DGEBAL: %d cases, max residual = %.3e (case %d)\n",
                  knt, rmax, lmax_resid);
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
