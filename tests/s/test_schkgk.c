/**
 * @file test_schkgk.c
 * @brief Tests SGGBAK, backward balancing of matrix pair eigenvectors.
 *
 * Port of LAPACK TESTING/EIG/dchkgk.f with embedded test data from
 * TESTING/dgbak.in.
 *
 * Verification: after SGGBAL balances (A,B) and SGGBAK transforms the
 * eigenvectors, check that VL'*AF*VR == VLF'*A*VRF (and same for B),
 * where AF/BF are the original matrices and VLF/VRF are pre-balance copies.
 */

#include "test_harness.h"
#include "semicolon_cblas.h"
#include <math.h>
#include <string.h>

#define THRESH 30.0f

typedef float f32;

#define LD 50

/* ---------- helpers ---------- */

static void rowmajor_to_colmajor(const f32* rm, f32* cm,
                                  INT nrow, INT ncol, INT ld)
{
    memset(cm, 0, (size_t)ld * ncol * sizeof(f32));
    for (INT i = 0; i < nrow; i++)
        for (INT j = 0; j < ncol; j++)
            cm[i + j * ld] = rm[i * ncol + j];
}

/* ---------- test case data from TESTING/dgbak.in ---------- */

typedef struct {
    INT n;
    INT m;           /* number of eigenvector columns */
    const f32* a_rm; /* n*n, row-major */
    const f32* b_rm; /* n*n, row-major */
    const f32* vl_rm; /* n*m, row-major */
    const f32* vr_rm; /* n*m, row-major */
} dgbak_case_t;

/* Case 0: N=6, M=3, diagonal pair */
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
static const f32 c0_vl[] = {
    1, 1, 1,   2, 2, 2,   3, 3, 3,
    4, 4, 4,   5, 5, 5,   6, 6, 6,
};
static const f32 c0_vr[] = {
    -1, -1, -1,  -2, -2, -2,  -3, -3, -3,
    -4, -4, -4,  -5, -5, -5,  -6, -6, -6,
};

/* Case 1: N=6, M=3, sub-diagonal + diagonal */
static const f32 c1_a[] = {
    1,   0,   0,   0,   0,   0,
    2,   2.1f, 0,   0,   0,   0,
    0,   3,   3.1f, 0,   0,   0,
    0,   0,   4,   4.1f, 0,   0,
    0,   0,   0,   5,   5.1f, 0,
    0,   0,   0,   0,   6,   6.1f,
};
static const f32 c1_b[] = {
    1, 0, 0, 0, 0, 0,
    0, 2, 0, 0, 0, 0,
    0, 0, 3, 0, 0, 0,
    0, 0, 0, 4, 0, 0,
    0, 0, 0, 0, 5, 0,
    0, 0, 0, 0, 0, 6,
};
static const f32 c1_vl[] = {
    1, 1, 1,   2, 2, 2,   3, 3, 3,
    4, 4, 4,   5, 5, 5,   6, 6, 6,
};
static const f32 c1_vr[] = {
    -1, -1, -1,  -2, -2, -2,  -3, -3, -3,
    -4, -4, -4,  -5, -5, -5,  -6, -6, -6,
};

/* Case 2: N=5, M=5, lower triangular + identity */
static const f32 c2_a[] = {
    1, 0, 0, 0, 0,
    1, 2, 0, 0, 0,
    1, 2, 3, 0, 0,
    1, 2, 3, 4, 0,
    1, 2, 3, 4, 5,
};
static const f32 c2_b[] = {
    1, 0, 0, 0, 0,
    0, 1, 0, 0, 0,
    0, 0, 1, 0, 0,
    0, 0, 0, 1, 0,
    0, 0, 0, 0, 1,
};
static const f32 c2_vl[] = {
    1, 1, 1, 1, 1,
    2, 2, 2, 2, 2,
    3, 3, 3, 3, 3,
    4, 4, 4, 4, 4,
    5, 5, 5, 5, 5,
};
static const f32 c2_vr[] = {
    1, 1, 1, 1, 1,
    2, 2, 2, 2, 2,
    3, 3, 3, 3, 3,
    4, 4, 4, 4, 4,
    5, 5, 5, 5, 5,
};

/* Case 3: N=6, M=5, 1e10 upper triangular */
static const f32 c3_a[] = {
    1, 1e10, 1e10, 1e10, 1e10, 1e10,
    1, 1,    1e10, 1e10, 1e10, 1e10,
    1, 1,    1,    1e10, 1e10, 1e10,
    1, 1,    1,    1,    1e10, 1e10,
    1, 1,    1,    1,    1,    1e10,
    1, 1,    1,    1,    1,    1,
};
static const f32 c3_b[] = {
    1, 1e10, 1e10, 1e10, 1e10, 1e10,
    1, 1,    1e10, 1e10, 1e10, 1e10,
    1, 1,    1,    1e10, 1e10, 1e10,
    1, 1,    1,    1,    1e10, 1e10,
    1, 1,    1,    1,    1,    1e10,
    1, 1,    1,    1,    1,    1,
};
static const f32 c3_vl[] = {
    1,  2, -3, 4,  5,
    8,  9,  0, 9,  2,
    0, -9,  2, 1,  1,
    8,  2,  1, 0,  2,
    0,  3,  2, 1,  1,
    2,  1,  9, 0,  1,
};
static const f32 c3_vr[] = {
     1, -2,  3, 4, 5,
    -8,  9,  0, 9, 2,
     0,  9,  2, 1, 1,
     8,  2,  1, 0, 2,
     0,  3,  2, 1, 1,
     2,  8,  9, 0, 1,
};

/* Case 4: N=6, M=2, structured with 1e6 entries */
static const f32 c4_a[] = {
    1, 0, 1, 1, 1, 1,
    1, 0, 0, 1, 1, 1,
    1e6, 0, 0, 0, 1e-6, 1e6,
    1, 1, 1, 1, 1, 1,
    1e6, 0, 0, 0, 1e-6, 1e-6,
    1e6, 0, 0, 0, 1e6, 1e6,
};
static const f32 c4_b[] = {
    1, 0, 1, 1, 1, 1,
    1, 0, 0, 1, 1, 1,
    1e6, 0, 0, 0, 1e-6, 1e6,
    1, 1, 1, 1, 1, 1,
    1e6, 0, 0, 0, 1e-6, 1e-6,
    1e6, 0, 0, 0, 1e6, 1e6,
};
static const f32 c4_vl[] = {
    1, 1,   2, 2,   3, 3,   4, 4,   5, 5,   6, 6,
};
static const f32 c4_vr[] = {
    1.1f, 1.1f,   2.2f, 2.2f,   3.3f, 3.3f,   4.4f, 4.4f,   5.5f, 5.5f,   6.6f, 6.6f,
};

/* Case 5: N=7, M=3, structured with isolation */
static const f32 c5_a[] = {
    0, 1, 0, 1, 1, 1, 1,
    0, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1,
    0, 1, 0, 1, 1, 1, 1,
    0, 1, 0, 0, 0, 0, 0,
    0, 1, 0, 0, 0, 0, 1,
    0, 1, 0, 1, 1, 1, 1,
};
static const f32 c5_b[] = {
    0, 1, 0, 1, 1, 1, 1,
    0, 1, 0, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1,
    0, 1, 0, 1, 1, 1, 1,
    0, 1, 0, 0, 0, 0, 0,
    0, 1, 0, 0, 0, 0, 1,
    0, 1, 0, 1, 1, 1, 1,
};
static const f32 c5_vl[] = {
    1, 1, 1,   2, 2, 2,   3, 3, 3,
    4, 4, 4,   5, 5, 5,   6, 6, 6,   7, 7, 7,
};
static const f32 c5_vr[] = {
    -1, -1, -1,  -2, -2, -2,  -3, -3, -3,
    -4, -4, -4,  -5, -5, -5,  -6, -6, -6,  -7, -7, -7,
};

/* Case 6: N=7, M=3, large magnitude differences */
static const f32 c6_a[] = {
    0, 1e3,  0,    1e3,  1e3,  1e3,  1e-5,
    0, 1e-5, 1e3,  1e-5, 1e-5, 1e3,  1e3,
    1e3, 1e3, 1e-5, 1e3,  1e3,  1e3,  1e3,
    0, 1e-5, 0,    0.1f,  1e3,  1e-5, 1e3,
    0, 1e3,  0,    0,    0,    0,    0,
    0, 4e-5, 0,    0,    0,    0,    1e-5,
    0, 1e3,  0,    1e3,  1e3,  1e-5, 1e3,
};
static const f32 c6_b[] = {
    0, 1e-2, 0,    1e3,  1e-5, 1e3,  1e3,
    0, 1e3,  1e3,  1e3,  1e3,  0.1f,  1e3,
    1e3, 1e3, 1e3,  1e3,  1e-5, 1e3,  1e3,
    0, 4e-2, 0,    1e3,  1,    1e3,  1e3,
    0, 1,    0,    0,    0,    0,    0,
    0, 1,    0,    0,    0,    0,    1,
    0, 1e-5, 0,    1e3,  1,    1,    1e-5,
};
static const f32 c6_vl[] = {
    1, 1, 1,   2, 2, 2,   3, 3, 3,
    4, 4, 4,   5, 5, 5,   6, 6, 6,   7, 7, 7,
};
static const f32 c6_vr[] = {
    1, 1, 1,   2, 2, 2,   3, 3, 3,
    4, 4, 4,   5, 5, 5,   6, 6, 6,   7, 7, 7,
};

/* Case 7: N=6, M=2, large magnitude with all active */
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
static const f32 c7_vl[] = {
    1, 1,   2, 2,   3, 3,   4, 4,   5, 5,   6, 6,
};
static const f32 c7_vr[] = {
    10, 10,   20, 20,   30, 30,   40, 40,   50, 50,   60, 60,
};

static const dgbak_case_t cases[] = {
    { 6, 3, c0_a, c0_b, c0_vl, c0_vr },
    { 6, 3, c1_a, c1_b, c1_vl, c1_vr },
    { 5, 5, c2_a, c2_b, c2_vl, c2_vr },
    { 6, 5, c3_a, c3_b, c3_vl, c3_vr },
    { 6, 2, c4_a, c4_b, c4_vl, c4_vr },
    { 7, 3, c5_a, c5_b, c5_vl, c5_vr },
    { 7, 3, c6_a, c6_b, c6_vl, c6_vr },
    { 6, 2, c7_a, c7_b, c7_vl, c7_vr },
};

#define NCASES ((int)(sizeof(cases) / sizeof(cases[0])))

/* ---------- test ---------- */

static void test_dggbak(void** state)
{
    (void)state;

    f32 eps = slamch("P");

    f32 a[LD * LD], b[LD * LD], af[LD * LD], bf[LD * LD];
    f32 vl[LD * LD], vr[LD * LD], vlf[LD * LD], vrf[LD * LD];
    f32 e[LD * LD], f[LD * LD], work[LD * LD];
    f32 lscale[LD], rscale[LD], dummy[1];
    INT ilo, ihi, info;
    f32 rmax = 0.0f, vmax;
    INT ninfo = 0, knt = 0;
    INT lmax[4] = {0, 0, 0, 0};

    for (INT tc = 0; tc < NCASES; tc++) {
        const dgbak_case_t* c = &cases[tc];
        INT n = c->n;
        INT m = c->m;

        /* Convert row-major to column-major */
        rowmajor_to_colmajor(c->a_rm, a, n, n, LD);
        rowmajor_to_colmajor(c->b_rm, b, n, n, LD);
        rowmajor_to_colmajor(c->vl_rm, vl, n, m, LD);
        rowmajor_to_colmajor(c->vr_rm, vr, n, m, LD);

        knt++;

        f32 anorm = slange("M", n, n, a, LD, dummy);
        f32 bnorm = slange("M", n, n, b, LD, dummy);

        /* Save originals */
        slacpy("F", n, n, a, LD, af, LD);
        slacpy("F", n, n, b, LD, bf, LD);

        /* Balance the pair */
        sggbal("B", n, a, LD, b, LD, &ilo, &ihi, lscale, rscale, work, &info);
        if (info != 0) {
            ninfo++;
            lmax[0] = knt;
        }

        /* Save pre-balance eigenvectors */
        slacpy("F", n, m, vl, LD, vlf, LD);
        slacpy("F", n, m, vr, LD, vrf, LD);

        /* Back-transform left eigenvectors */
        sggbak("B", "L", n, ilo, ihi, lscale, rscale, m, vl, LD, &info);
        if (info != 0) {
            ninfo++;
            lmax[1] = knt;
        }

        /* Back-transform right eigenvectors */
        sggbak("B", "R", n, ilo, ihi, lscale, rscale, m, vr, LD, &info);
        if (info != 0) {
            ninfo++;
            lmax[2] = knt;
        }

        /*
         * Verify for A:
         *   E = VL' * AF * VR    (using back-transformed vectors and original matrix)
         *   F = VLF' * A * VRF   (using pre-balance vectors and balanced matrix)
         * These should be equal.
         */

        /* E = VL' * AF * VR: first work = AF * VR, then E = VL' * work */
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    n, m, n, 1.0f, af, LD, vr, LD, 0.0f, work, LD);
        cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                    m, m, n, 1.0f, vl, LD, work, LD, 0.0f, e, LD);

        /* F = VLF' * A * VRF: first work = A * VRF, then F = VLF' * work */
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    n, m, n, 1.0f, a, LD, vrf, LD, 0.0f, work, LD);
        cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                    m, m, n, 1.0f, vlf, LD, work, LD, 0.0f, f, LD);

        vmax = 0.0f;
        for (INT j = 0; j < m; j++)
            for (INT i = 0; i < m; i++) {
                f32 diff = fabsf(e[i + j * LD] - f[i + j * LD]);
                if (diff > vmax) vmax = diff;
            }

        f32 maxnorm = anorm > bnorm ? anorm : bnorm;
        vmax = vmax / (eps * maxnorm);
        if (vmax > rmax) {
            lmax[3] = knt;
            rmax = vmax;
        }

        /*
         * Verify for B: same check with BF/B instead of AF/A
         */
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    n, m, n, 1.0f, bf, LD, vr, LD, 0.0f, work, LD);
        cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                    m, m, n, 1.0f, vl, LD, work, LD, 0.0f, e, LD);

        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    n, m, n, 1.0f, b, LD, vrf, LD, 0.0f, work, LD);
        cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                    m, m, n, 1.0f, vlf, LD, work, LD, 0.0f, f, LD);

        vmax = 0.0f;
        for (INT j = 0; j < m; j++)
            for (INT i = 0; i < m; i++) {
                f32 diff = fabsf(e[i + j * LD] - f[i + j * LD]);
                if (diff > vmax) vmax = diff;
            }

        vmax = vmax / (eps * maxnorm);
        if (vmax > rmax) {
            lmax[3] = knt;
            rmax = vmax;
        }
    }

    print_message("SGGBAK: %d cases, max residual = %.3e (case %d)\n",
                  knt, (double)rmax, lmax[3]);
    if (ninfo > 0)
        print_message("  INFO errors: %d (bal=%d, bakL=%d, bakR=%d)\n",
                      ninfo, lmax[0], lmax[1], lmax[2]);

    assert_true(ninfo == 0);
    assert_residual_ok(rmax);
}

int main(void)
{
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_dggbak),
    };
    return cmocka_run_group_tests_name("dchkgk", tests, NULL, NULL);
}
