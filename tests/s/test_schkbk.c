/**
 * @file test_schkbk.c
 * @brief Tests SGEBAK, backward transformation of balanced eigenvectors.
 *
 * Port of LAPACK TESTING/EIG/dchkbk.f with embedded test data from
 * TESTING/dbak.in.
 */

#include "test_harness.h"
#include <math.h>
#include <string.h>

#define THRESH 30.0f

typedef float f32;

#define LDE 20

/* ---------- helpers ---------- */

static void rowmajor_to_colmajor(const f32* rm, f32* cm, INT nrow, INT ncol, INT ld)
{
    memset(cm, 0, (size_t)ld * ncol * sizeof(f32));
    for (INT i = 0; i < nrow; i++)
        for (INT j = 0; j < ncol; j++)
            cm[i + j * ld] = rm[i * ncol + j];
}

/* ---------- test case data from TESTING/dbak.in ---------- */

typedef struct {
    INT n;
    INT ilo;
    INT ihi;
    const f32* scale;
    const f32* e_rm;
    const f32* ein_rm;
} dbak_case_t;

/* Case 0: N=5 */
static const f32 c0_scale[] = {1, 1, 2, 3, 4};
static const f32 c0_e[] = {
    1, 0, 0, 0, 0,
    0, 1, 0, 0, 0,
    0, 0, 1, 0, 0,
    0, 0, 0, 1, 0,
    0, 0, 0, 0, 1,
};
static const f32 c0_ein[] = {
    1, 0, 0, 0, 0,
    0, 1, 0, 0, 0,
    0, 0, 1, 0, 0,
    0, 0, 0, 1, 0,
    0, 0, 0, 0, 1,
};

/* Case 1: N=5 */
static const f32 c1_scale[] = {1, 1, 2, 1, 0};
static const f32 c1_e[] = {
     1,      1,      1,     -0.6667f, -0.04167f,
     0,     -0.25f,  -0.6667f, 1,       0.1667f,
     0,      0,      0.2222f,-1,      -0.5f,
     0,      0,      0,      0.5f,     1,
     0,      0,      0,      0,      -1,
};
static const f32 c1_ein[] = {
     0,      0,      0,      0,      -1,
     0,      0,      0,      0.5f,     1,
     0,      0,      0.2222f,-1,      -0.5f,
     0,     -0.25f,  -0.6667f, 1,       0.1667f,
     1,      1,      1,     -0.6667f, -0.04167f,
};

/* Case 2: N=5 */
static const f32 c2_scale[] = {1, 1, 2, 1, 0};
static const f32 c2_e[] = {
     1,      1,      1,      1,      1,
     0,     -6e-18, -6e-18, -6e-18, -6e-18,
     0,      0,      3.6e-35f, 3.6e-35f, 3.6e-35f,
     0,      0,      0,      0,      0,
     0,      0,      0,      0,      0,
};
static const f32 c2_ein[] = {
     0,      0,      0,      0,      0,
     0,      0,      0,      0,      0,
     0,      0,      3.6e-35f, 3.6e-35f, 3.6e-35f,
     0,     -6e-18, -6e-18, -6e-18, -6e-18,
     1,      1,      1,      1,      1,
};

/* Case 3: N=6 */
static const f32 c3_scale[] = {3, 2, 4, 100, 0.1f, 1};
static const f32 c3_e[] = {
     1,      1.336e-6f, 1,      1,      1,      1,
     0,      1,      0,     -3.001e-11f,-3.252e-5f, 0.01305f,
     0,      0,     -8.33e-3f, 8.929e-10f,-6.712e-5f, 6.687e-5f,
     0,      0,      0,     -4.455e-6f,-3.355e-3f, 3.345e-3f,
     0,      0,      0,      4.455e-7f,-3.356e-2f, 3.344e-2f,
     0,      0,      0,      4.411e-10f, 0.1011f, 0.1008f,
};
static const f32 c3_ein[] = {
     0,      0,      0,     -4.455e-4f,-0.3355f,  0.3345f,
     0,      0,      0,      4.455e-8f,-3.356e-3f, 3.344e-3f,
     0,      1,      0,     -3.001e-11f,-3.252e-5f, 0.01305f,
     1,      1.336e-6f, 1,    1,      1,      1,
     0,      0,     -8.33e-3f, 8.929e-10f,-6.712e-5f, 6.687e-5f,
     0,      0,      0,      4.411e-10f, 0.1011f, 0.1008f,
};

/* Case 4: N=5 */
static const f32 c4_scale[] = {100, 0.1f, 0.01f, 1, 10};
static const f32 c4_e[] = {
     1.366e-4f,-6.829e-5f, 1.252e-4f, 1,       1.95e-15f,
     1,       1,        -2.776e-17f, 3.601e-6f,-6.073e-18f,
     0.2736f,  -0.1363f,   0.2503f, -3.322e-6f,-2e-3,
     6.909e-3f,-3.443e-3f, 6.196e-3f, 0.01666f, 1,
     0.3899f,  -0.2033f,  -0.342f,  -1e-3,    6e-15,
};
static const f32 c4_ein[] = {
     0.01366f, -6.829e-3f, 0.01252f, 100,      1.95e-13f,
     0.1f,     0.1f,      -2.776e-18f, 3.601e-7f,-6.073e-19f,
     2.736e-3f,-1.363e-3f, 2.503e-3f,-3.322e-8f,-2e-5,
     6.909e-3f,-3.443e-3f, 6.196e-3f, 0.01666f, 1,
     3.899f,   -2.033f,   -3.42f,   -0.01f,    6e-14,
};

/* Case 5: N=6 */
static const f32 c5_scale[] = {2, 1, 1, 1, 1, 3};
static const f32 c5_e[] = {
     1,      1,      2.776e-16f,-2.405e-17f, 0,      1,
     0,      0.75f,   1,       0.0852f,    0,     -1.52e-17f,
     0,      0.75f,  -0.8093f,  1,         0,     -1.52e-17f,
     0,      0.75f,  -0.09533f,-0.5426f,    1,     -1.52e-17f,
     0,      0.75f,  -0.09533f,-0.5426f,   -1,     -1.52e-17f,
     0,      0,      0,       0,         0,      4.559e-17f,
};
static const f32 c5_ein[] = {
     0,      0.75f,  -0.8093f,  1,         0,     -1.52e-17f,
     0,      0.75f,   1,       0.0852f,    0,     -1.52e-17f,
     1,      1,      2.776e-16f,-2.405e-17f, 0,    1,
     0,      0,      0,       0,         0,      4.559e-17f,
     0,      0.75f,  -0.09533f,-0.5426f,   -1,     -1.52e-17f,
     0,      0.75f,  -0.09533f,-0.5426f,    1,     -1.52e-17f,
};

/* Case 6: N=7 */
static const f32 c6_scale[] = {2, 1e-3, 0.01f, 10, 0.1f, 0, 5};
static const f32 c6_e[] = {
     1,     -0.01105f, 0.03794f,-0.09378f,-0.03481f, 0.4465f, -0.03602f,
     0,     -0.4556f, -0.4545f,  1,       0.4639f, -0.6512f,  0.4781f,
     0,     -0.2734f, -0.7946f,  0.6303f,  1,      -0.6279f,  1,
     0,      1,      -6.939e-18f, 0.04259f,-0.6495f,-0.5581f, -0.6452f,
     0,     -0.3904f, -0.4029f, -0.1685f, -0.9429f,  1,      -0.9371f,
     0,      0,       0,       0,       0,      -0.2558f,  3.308e-4f,
     0,      0,       0,       0,       0,       0,      -1.985e-3f,
};
static const f32 c6_ein[] = {
     0,      0,       0,       0,       0,      -0.2558f,  3.308e-4f,
     0,     -4.556e-4f,-4.545e-4f, 1e-3,   4.639e-4f,-6.512e-4f, 4.781e-4f,
     1,     -0.01105f, 0.03794f,-0.09378f,-0.03481f, 0.4465f, -0.03602f,
     0,      10,     -6.939e-17f, 0.4259f, -6.495f,  -5.581f,  -6.452f,
     0,     -0.03904f,-0.04029f,-0.01685f,-0.09429f, 0.1f,    -0.09371f,
     0,      0,       0,       0,       0,       0,      -1.985e-3f,
     0,     -2.734e-3f,-7.946e-3f, 6.303e-3f, 0.01f, -6.279e-3f, 0.01f,
};

static const dbak_case_t cases[] = {
    { 5, 0, 0, c0_scale, c0_e, c0_ein },
    { 5, 0, 0, c1_scale, c1_e, c1_ein },
    { 5, 0, 0, c2_scale, c2_e, c2_ein },
    { 6, 3, 5, c3_scale, c3_e, c3_ein },
    { 5, 0, 4, c4_scale, c4_e, c4_ein },
    { 6, 1, 4, c5_scale, c5_e, c5_ein },
    { 7, 1, 4, c6_scale, c6_e, c6_ein },
};

#define NCASES ((int)(sizeof(cases) / sizeof(cases[0])))

/* ---------- test ---------- */

static void test_dgebak(void** state)
{
    (void)state;

    f32 eps = slamch("E");
    f32 safmin = slamch("S");

    f32 e[LDE * LDE], ein[LDE * LDE];
    INT info;
    f32 rmax = 0.0f, vmax;
    INT ninfo = 0, knt = 0;
    INT lmax_info = 0, lmax_resid = 0;

    for (INT tc = 0; tc < NCASES; tc++) {
        const dbak_case_t* c = &cases[tc];
        INT n = c->n;
        INT ilo = c->ilo;
        INT ihi = c->ihi;

        rowmajor_to_colmajor(c->e_rm, e, n, n, LDE);
        rowmajor_to_colmajor(c->ein_rm, ein, n, n, LDE);

        knt++;

        sgebak("B", "R", n, ilo, ihi, c->scale, n, e, LDE, &info);

        if (info != 0) {
            ninfo++;
            lmax_info = knt;
        }

        vmax = 0.0f;
        for (INT i = 0; i < n; i++) {
            for (INT j = 0; j < n; j++) {
                f32 x = fabsf(e[i + j * LDE] - ein[i + j * LDE]) / eps;
                if (fabsf(e[i + j * LDE]) > safmin)
                    x = x / fabsf(e[i + j * LDE]);
                if (x > vmax) vmax = x;
            }
        }

        if (vmax > rmax) {
            lmax_resid = knt;
            rmax = vmax;
        }
    }

    print_message("SGEBAK: %d cases, max residual = %.3e (case %d)\n",
                  knt, (double)rmax, lmax_resid);
    if (ninfo > 0)
        print_message("  INFO errors: %d (case %d)\n", ninfo, lmax_info);

    assert_true(ninfo == 0);
    assert_residual_ok(rmax);
}

int main(void)
{
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_dgebak),
    };
    return cmocka_run_group_tests_name("dchkbk", tests, NULL, NULL);
}
