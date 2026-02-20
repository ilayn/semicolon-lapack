/**
 * @file test_schkgl.c
 * @brief Tests SGGBAL, a routine for balancing a matrix pair (A, B).
 *
 * Port of LAPACK TESTING/EIG/dchkgl.f with embedded test data from
 * TESTING/dgbal.in.
 */

#include "test_harness.h"
#include <math.h>
#include <string.h>

#define THRESH 30.0f

typedef float f32;

extern f32 slamch(const char* cmach);
extern f32 slange(const char* norm, const int m, const int n,
                  const f32* A, const int lda, f32* work);
extern void sggbal(const char* job, const int n, f32* A, const int lda,
                   f32* B, const int ldb, int* ilo, int* ihi,
                   f32* lscale, f32* rscale, f32* work, int* info);

#define LDA 20
#define LWORK (6 * LDA)

/* ---------- helpers ---------- */

static void rowmajor_to_colmajor(const f32* rm, f32* cm, int n, int ld)
{
    memset(cm, 0, (size_t)ld * n * sizeof(f32));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            cm[i + j * ld] = rm[i * n + j];
}

/* ---------- test case data from TESTING/dgbal.in ---------- */

typedef struct {
    int n;
    int iloin;
    int ihiin;
    const f32* a_rm;
    const f32* b_rm;
    const f32* ain_rm;
    const f32* bin_rm;
    const f32* lsclin;
    const f32* rsclin;
} dgbal_case_t;

/* Case 0: N=6 diagonal pair */
static const f32 c0_a[] = {
    1, 0, 0, 0, 0, 0,
    0, 2, 0, 0, 0, 0,
    0, 0, 3, 0, 0, 0,
    0, 0, 0, 4, 0, 0,
    0, 0, 0, 0, 5, 0,
    0, 0, 0, 0, 0, 6,
};
static const f32 c0_b[] = {
    6, 0, 0, 0, 0, 0,
    0, 5, 0, 0, 0, 0,
    0, 0, 4, 0, 0, 0,
    0, 0, 0, 3, 0, 0,
    0, 0, 0, 0, 2, 0,
    0, 0, 0, 0, 0, 1,
};
static const f32 c0_ain[] = {
    1, 0, 0, 0, 0, 0,
    0, 2, 0, 0, 0, 0,
    0, 0, 3, 0, 0, 0,
    0, 0, 0, 4, 0, 0,
    0, 0, 0, 0, 5, 0,
    0, 0, 0, 0, 0, 6,
};
static const f32 c0_bin[] = {
    6, 0, 0, 0, 0, 0,
    0, 5, 0, 0, 0, 0,
    0, 0, 4, 0, 0, 0,
    0, 0, 0, 3, 0, 0,
    0, 0, 0, 0, 2, 0,
    0, 0, 0, 0, 0, 1,
};
static const f32 c0_ls[] = {1, 1, 2, 3, 4, 5};
static const f32 c0_rs[] = {1, 1, 2, 3, 4, 5};

/* Case 1: N=6, sub-diagonal A, identity B */
static const f32 c1_a[] = {
    1, 0, 0, 0, 0, 0,
    1, 1, 0, 0, 0, 0,
    0, 1, 1, 0, 0, 0,
    0, 0, 1, 1, 0, 0,
    0, 0, 0, 1, 1, 0,
    0, 0, 0, 0, 1, 1,
};
static const f32 c1_b[] = {
    1, 0, 0, 0, 0, 0,
    0, 1, 0, 0, 0, 0,
    0, 0, 1, 0, 0, 0,
    0, 0, 0, 1, 0, 0,
    0, 0, 0, 0, 1, 0,
    0, 0, 0, 0, 0, 1,
};
static const f32 c1_ain[] = {
    1, 1, 0, 0, 0, 0,
    0, 1, 1, 0, 0, 0,
    0, 0, 1, 1, 0, 0,
    0, 0, 0, 1, 1, 0,
    0, 0, 0, 0, 1, 1,
    0, 0, 0, 0, 0, 1,
};
static const f32 c1_bin[] = {
    1, 0, 0, 0, 0, 0,
    0, 1, 0, 0, 0, 0,
    0, 0, 1, 0, 0, 0,
    0, 0, 0, 1, 0, 0,
    0, 0, 0, 0, 1, 0,
    0, 0, 0, 0, 0, 1,
};
static const f32 c1_ls[] = {1, 1, 2, 2, 1, 0};
static const f32 c1_rs[] = {1, 1, 2, 2, 1, 0};

/* Case 2: N=6, lower triangular A, similar B */
static const f32 c2_a[] = {
    1, 0, 0, 0, 0, 0,
    1, 2, 0, 0, 0, 0,
    1, 2, 3, 0, 0, 0,
    1, 2, 3, 4, 0, 0,
    1, 2, 3, 4, 5, 0,
    1, 2, 3, 4, 5, 6,
};
static const f32 c2_b[] = {
    1, 0, 0, 0, 0, 0,
    1, 2, 0, 0, 0, 0,
    1, 2, 3, 0, 0, 0,
    1, 2, 3, 4, 0, 0,
    1, 2, 3, 4, 5, 0,
    1, 2, 3, 4, 5, 6,
};
static const f32 c2_ain[] = {
    6, 5, 4, 3, 2, 1,
    0, 5, 4, 3, 2, 1,
    0, 0, 4, 3, 2, 1,
    0, 0, 0, 3, 2, 1,
    0, 0, 0, 0, 2, 1,
    0, 0, 0, 0, 0, 1,
};
static const f32 c2_bin[] = {
    6, 5, 4, 3, 2, 1,
    0, 5, 4, 3, 2, 1,
    0, 0, 4, 3, 2, 1,
    0, 0, 0, 3, 2, 1,
    0, 0, 0, 0, 2, 1,
    0, 0, 0, 0, 0, 1,
};
static const f32 c2_ls[] = {1, 1, 2, 2, 1, 0};
static const f32 c2_rs[] = {1, 1, 2, 2, 1, 0};

/* Case 3: N=5, lower triangular + identity */
static const f32 c3_a[] = {
    1, 0, 0, 0, 0,
    1, 2, 0, 0, 0,
    1, 2, 3, 0, 0,
    1, 2, 3, 4, 0,
    1, 2, 3, 4, 5,
};
static const f32 c3_b[] = {
    1, 0, 0, 0, 0,
    0, 1, 0, 0, 0,
    0, 0, 1, 0, 0,
    0, 0, 0, 1, 0,
    0, 0, 0, 0, 1,
};
static const f32 c3_ain[] = {
    5, 4, 3, 2, 1,
    0, 4, 3, 2, 1,
    0, 0, 3, 2, 1,
    0, 0, 0, 2, 1,
    0, 0, 0, 0, 1,
};
static const f32 c3_bin[] = {
    1, 0, 0, 0, 0,
    0, 1, 0, 0, 0,
    0, 0, 1, 0, 0,
    0, 0, 0, 1, 0,
    0, 0, 0, 0, 1,
};
static const f32 c3_ls[] = {1, 1, 2, 1, 0};
static const f32 c3_rs[] = {1, 1, 2, 1, 0};

/* Case 4: N=6, 1e10 upper triangular pair */
static const f32 c4_a[] = {
    1, 1e10, 1e10, 1e10, 1e10, 1e10,
    1, 1,    1e10, 1e10, 1e10, 1e10,
    1, 1,    1,    1e10, 1e10, 1e10,
    1, 1,    1,    1,    1e10, 1e10,
    1, 1,    1,    1,    1,    1e10,
    1, 1,    1,    1,    1,    1,
};
static const f32 c4_b[] = {
    1, 1e10, 1e10, 1e10, 1e10, 1e10,
    1, 1,    1e10, 1e10, 1e10, 1e10,
    1, 1,    1,    1e10, 1e10, 1e10,
    1, 1,    1,    1,    1e10, 1e10,
    1, 1,    1,    1,    1,    1e10,
    1, 1,    1,    1,    1,    1,
};
static const f32 c4_ain[] = {
    1e-4,   1e4,  1e3,  1e1,  1e-1,   1e-2,
    1e-3,   1e-5, 1e4,  1e2,  1,      1e-1,
    1e-1,   1e-3, 1e-4, 1e4,  1e2,    1e1,
    1e1,    1e-1, 1e-2, 1e-4, 1e4,    1e3,
    1e2,    1,    1e-1, 1e-3, 1e-5,   1e4,
    1e4,    1e2,  1e1,  1e-1, 1e-3,   1e-4,
};
static const f32 c4_bin[] = {
    1e-4,   1e4,  1e3,  1e1,  1e-1,   1e-2,
    1e-3,   1e-5, 1e4,  1e2,  1,      1e-1,
    1e-1,   1e-3, 1e-4, 1e4,  1e2,    1e1,
    1e1,    1e-1, 1e-2, 1e-4, 1e4,    1e3,
    1e2,    1,    1e-1, 1e-3, 1e-5,   1e4,
    1e4,    1e2,  1e1,  1e-1, 1e-3,   1e-4,
};
static const f32 c4_ls[] = {1e-6, 1e-5, 1e-3, 1e-1, 1, 1e2};
static const f32 c4_rs[] = {1e2, 1, 1e-1, 1e-3, 1e-5, 1e-6};

/* Case 5: N=6, structured with 1e6 entries and isolation */
static const f32 c5_a[] = {
    1, 0, 1, 1, 1, 1,
    1, 0, 0, 1, 1, 1,
    1e6, 0, 0, 0, 1e-6, 1e6,
    1, 1, 1, 1, 1, 1,
    1e6, 0, 0, 0, 1e-6, 1e-6,
    1e6, 0, 0, 0, 1e6, 1e6,
};
static const f32 c5_b[] = {
    1, 0, 1, 1, 1, 1,
    1, 0, 0, 1, 1, 1,
    1e6, 0, 0, 0, 1e-6, 1e6,
    1, 1, 1, 1, 1, 1,
    1e6, 0, 0, 0, 1e-6, 1e-6,
    1e6, 0, 0, 0, 1e6, 1e6,
};
static const f32 c5_ain[] = {
    1, 1, 1, 1e-5, 1e3, 1e-1,
    0, 1, 1, 1e-5, 1e3, 1e-1,
    0, 0, 1, 1e-5, 1e3, 1e-1,
    0, 0, 0, 1, 1e-4, 1e4,
    0, 0, 0, 1e4, 1, 1e-4,
    0, 0, 0, 1e-4, 1e4, 1,
};
static const f32 c5_bin[] = {
    1, 1, 1, 1e-5, 1e3, 1e-1,
    0, 1, 1, 1e-5, 1e3, 1e-1,
    0, 0, 1, 1e-5, 1e3, 1e-1,
    0, 0, 0, 1, 1e-4, 1e4,
    0, 0, 0, 1e4, 1, 1e-4,
    0, 0, 0, 1e-4, 1e4, 1,
};
static const f32 c5_ls[] = {3, 3, 3, 1e-1, 1e3, 1e-5};
static const f32 c5_rs[] = {1, 2, 3, 1e-5, 1e3, 1e-1};

/* Case 6: N=7, with isolation at top and bottom */
static const f32 c6_a[] = {
    0, 1, 0, 1, 1, 1, 1,
    0, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1,
    0, 1, 0, 1, 1, 1, 1,
    0, 1, 0, 0, 0, 0, 0,
    0, 1, 0, 0, 0, 0, 1,
    0, 1, 0, 1, 1, 1, 1,
};
static const f32 c6_b[] = {
    0, 1, 0, 1, 1, 1, 1,
    0, 1, 0, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1,
    0, 1, 0, 1, 1, 1, 1,
    0, 1, 0, 0, 0, 0, 0,
    0, 1, 0, 0, 0, 0, 1,
    0, 1, 0, 1, 1, 1, 1,
};
static const f32 c6_ain[] = {
    1, 1, 1, 1, 1, 1, 1,
    0, 1, 1, 1, 1, 1, 1,
    0, 0, 1, 1, 1, 1, 1,
    0, 0, 1, 1, 1, 1, 1,
    0, 0, 1, 1, 1, 1, 1,
    0, 0, 0, 0, 0, 1, 1,
    0, 0, 0, 0, 0, 0, 1,
};
static const f32 c6_bin[] = {
    1, 1, 1, 1, 1, 1, 1,
    0, 0, 1, 1, 1, 1, 1,
    0, 0, 1, 1, 1, 1, 1,
    0, 0, 1, 1, 1, 1, 1,
    0, 0, 1, 1, 1, 1, 1,
    0, 0, 0, 0, 0, 1, 1,
    0, 0, 0, 0, 0, 0, 1,
};
static const f32 c6_ls[] = {2, 1, 1, 1, 1, 5, 4};
static const f32 c6_rs[] = {0, 2, 1, 1, 1, 1, 1};

/* Case 7: N=6, large magnitude differences */
static const f32 c7_a[] = {
    -20,   -1e4,  -2,     -1e6,  -10,   -2e5,
     6e-3,  4,     6e-4,   200,   3e-3,  30,
    -0.2f,  -300,  -0.04f,  -1e4,   0,     3000,
     6e-5,  0.04f,  9e-6,   9,     3e-5,  0.5f,
     0.06f,  50,    8e-3,  -4000,  0.08f,  0,
     0,     1000,  0.7f,   -2e5,   13,   -6e4,
};
static const f32 c7_b[] = {
    -20,   -1e4,   2,     -2e6,   10,   -1e5,
     5e-3,  3,    -2e-4,   400,  -1e-3,  30,
     0,    -100,  -0.08f,   2e4,  -0.4f,   0,
     5e-5,  0.03f,  2e-6,   4,     2e-5,  0.1f,
     0.04f,  30,   -1e-3,   3000, -0.01f,  600,
    -1,     0,     0.4f,   -1e5,   4,     2e4,
};
static const f32 c7_ain[] = {
    -0.2f,  -1,    -0.2f,   -1,    -1,     -2,
     0.6f,   4,     0.6f,    2,     3,      3,
    -0.2f,  -3,    -0.4f,   -1,     0,      3,
     0.6f,   4,     0.9f,    9,     3,      5,
     0.6f,   5,     0.8f,   -4,     8,      0,
     0,     1,     0.7f,   -2,     13,    -6,
};
static const f32 c7_bin[] = {
    -0.2f,  -1,     0.2f,   -2,     1,     -1,
     0.5f,   3,    -0.2f,    4,    -1,      3,
     0,    -1,    -0.8f,    2,    -4,      0,
     0.5f,   3,     0.2f,    4,     2,      1,
     0.4f,   3,    -0.1f,    3,    -1,      6,
    -0.1f,   0,     0.4f,   -1,     4,      2,
};
static const f32 c7_ls[] = {1e-3, 1e1, 1e-1, 1e3, 1, 1e-2};
static const f32 c7_rs[] = {1e1, 1e-1, 1e2, 1e-3, 1e2, 1e-2};

static const dgbal_case_t cases[] = {
    { 6, 0, 0, c0_a, c0_b, c0_ain, c0_bin, c0_ls, c0_rs },
    { 6, 0, 0, c1_a, c1_b, c1_ain, c1_bin, c1_ls, c1_rs },
    { 6, 0, 0, c2_a, c2_b, c2_ain, c2_bin, c2_ls, c2_rs },
    { 5, 0, 0, c3_a, c3_b, c3_ain, c3_bin, c3_ls, c3_rs },
    { 6, 0, 5, c4_a, c4_b, c4_ain, c4_bin, c4_ls, c4_rs },
    { 6, 3, 5, c5_a, c5_b, c5_ain, c5_bin, c5_ls, c5_rs },
    { 7, 2, 4, c6_a, c6_b, c6_ain, c6_bin, c6_ls, c6_rs },
    { 6, 0, 5, c7_a, c7_b, c7_ain, c7_bin, c7_ls, c7_rs },
};

#define NCASES ((int)(sizeof(cases) / sizeof(cases[0])))

/* ---------- test ---------- */

static void test_dggbal(void** state)
{
    (void)state;

    f32 eps = slamch("P");

    f32 a[LDA * LDA], b[LDA * LDA];
    f32 ain[LDA * LDA], bin[LDA * LDA];
    f32 lscale[LDA], rscale[LDA], work[LWORK];
    f32 dummy[1];
    int ilo, ihi, info;
    f32 rmax = 0.0f, vmax;
    int ninfo = 0, knt = 0;
    int lmax_info = 0, lmax_idx = 0, lmax_resid = 0;

    for (int tc = 0; tc < NCASES; tc++) {
        const dgbal_case_t* c = &cases[tc];
        int n = c->n;
        int iloin = c->iloin;
        int ihiin = c->ihiin;

        rowmajor_to_colmajor(c->a_rm, a, n, LDA);
        rowmajor_to_colmajor(c->b_rm, b, n, LDA);
        rowmajor_to_colmajor(c->ain_rm, ain, n, LDA);
        rowmajor_to_colmajor(c->bin_rm, bin, n, LDA);

        f32 anorm = slange("M", n, n, a, LDA, dummy);
        f32 bnorm = slange("M", n, n, b, LDA, dummy);

        knt++;

        sggbal("B", n, a, LDA, b, LDA, &ilo, &ihi, lscale, rscale, work, &info);

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

        /* Compare balanced matrices and scale vectors */
        vmax = 0.0f;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                f32 diff = fabsf(a[i + j * LDA] - ain[i + j * LDA]);
                if (diff > vmax) vmax = diff;
                diff = fabsf(b[i + j * LDA] - bin[i + j * LDA]);
                if (diff > vmax) vmax = diff;
            }
        }

        for (int i = 0; i < n; i++) {
            f32 diff = fabsf(lscale[i] - c->lsclin[i]);
            if (diff > vmax) vmax = diff;
            diff = fabsf(rscale[i] - c->rsclin[i]);
            if (diff > vmax) vmax = diff;
        }

        f32 maxnorm = anorm > bnorm ? anorm : bnorm;
        vmax = vmax / (eps * maxnorm);

        if (vmax > rmax) {
            lmax_resid = knt;
            rmax = vmax;
        }
    }

    print_message("SGGBAL: %d cases, max residual = %.3e (case %d)\n",
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
        cmocka_unit_test(test_dggbal),
    };
    return cmocka_run_group_tests_name("dchkgl", tests, NULL, NULL);
}
