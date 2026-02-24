/**
 * @file test_dchkbk.c
 * @brief Tests DGEBAK, backward transformation of balanced eigenvectors.
 *
 * Port of LAPACK TESTING/EIG/dchkbk.f with embedded test data from
 * TESTING/dbak.in.
 */

#include "test_harness.h"
#include <math.h>
#include <string.h>

#define THRESH 30.0

typedef double f64;

#define LDE 20

/* ---------- helpers ---------- */

static void rowmajor_to_colmajor(const f64* rm, f64* cm, INT nrow, INT ncol, INT ld)
{
    memset(cm, 0, (size_t)ld * ncol * sizeof(f64));
    for (INT i = 0; i < nrow; i++)
        for (INT j = 0; j < ncol; j++)
            cm[i + j * ld] = rm[i * ncol + j];
}

/* ---------- test case data from TESTING/dbak.in ---------- */

typedef struct {
    INT n;
    INT ilo;
    INT ihi;
    const f64* scale;
    const f64* e_rm;
    const f64* ein_rm;
} dbak_case_t;

/* Case 0: N=5 */
static const f64 c0_scale[] = {1, 1, 2, 3, 4};
static const f64 c0_e[] = {
    1, 0, 0, 0, 0,
    0, 1, 0, 0, 0,
    0, 0, 1, 0, 0,
    0, 0, 0, 1, 0,
    0, 0, 0, 0, 1,
};
static const f64 c0_ein[] = {
    1, 0, 0, 0, 0,
    0, 1, 0, 0, 0,
    0, 0, 1, 0, 0,
    0, 0, 0, 1, 0,
    0, 0, 0, 0, 1,
};

/* Case 1: N=5 */
static const f64 c1_scale[] = {1, 1, 2, 1, 0};
static const f64 c1_e[] = {
     1,      1,      1,     -0.6667, -0.04167,
     0,     -0.25,  -0.6667, 1,       0.1667,
     0,      0,      0.2222,-1,      -0.5,
     0,      0,      0,      0.5,     1,
     0,      0,      0,      0,      -1,
};
static const f64 c1_ein[] = {
     0,      0,      0,      0,      -1,
     0,      0,      0,      0.5,     1,
     0,      0,      0.2222,-1,      -0.5,
     0,     -0.25,  -0.6667, 1,       0.1667,
     1,      1,      1,     -0.6667, -0.04167,
};

/* Case 2: N=5 */
static const f64 c2_scale[] = {1, 1, 2, 1, 0};
static const f64 c2_e[] = {
     1,      1,      1,      1,      1,
     0,     -6e-18, -6e-18, -6e-18, -6e-18,
     0,      0,      3.6e-35, 3.6e-35, 3.6e-35,
     0,      0,      0,      0,      0,
     0,      0,      0,      0,      0,
};
static const f64 c2_ein[] = {
     0,      0,      0,      0,      0,
     0,      0,      0,      0,      0,
     0,      0,      3.6e-35, 3.6e-35, 3.6e-35,
     0,     -6e-18, -6e-18, -6e-18, -6e-18,
     1,      1,      1,      1,      1,
};

/* Case 3: N=6 */
static const f64 c3_scale[] = {3, 2, 4, 100, 0.1, 1};
static const f64 c3_e[] = {
     1,      1.336e-6, 1,      1,      1,      1,
     0,      1,      0,     -3.001e-11,-3.252e-5, 0.01305,
     0,      0,     -8.33e-3, 8.929e-10,-6.712e-5, 6.687e-5,
     0,      0,      0,     -4.455e-6,-3.355e-3, 3.345e-3,
     0,      0,      0,      4.455e-7,-3.356e-2, 3.344e-2,
     0,      0,      0,      4.411e-10, 0.1011, 0.1008,
};
static const f64 c3_ein[] = {
     0,      0,      0,     -4.455e-4,-0.3355,  0.3345,
     0,      0,      0,      4.455e-8,-3.356e-3, 3.344e-3,
     0,      1,      0,     -3.001e-11,-3.252e-5, 0.01305,
     1,      1.336e-6, 1,    1,      1,      1,
     0,      0,     -8.33e-3, 8.929e-10,-6.712e-5, 6.687e-5,
     0,      0,      0,      4.411e-10, 0.1011, 0.1008,
};

/* Case 4: N=5 */
static const f64 c4_scale[] = {100, 0.1, 0.01, 1, 10};
static const f64 c4_e[] = {
     1.366e-4,-6.829e-5, 1.252e-4, 1,       1.95e-15,
     1,       1,        -2.776e-17, 3.601e-6,-6.073e-18,
     0.2736,  -0.1363,   0.2503, -3.322e-6,-2e-3,
     6.909e-3,-3.443e-3, 6.196e-3, 0.01666, 1,
     0.3899,  -0.2033,  -0.342,  -1e-3,    6e-15,
};
static const f64 c4_ein[] = {
     0.01366, -6.829e-3, 0.01252, 100,      1.95e-13,
     0.1,     0.1,      -2.776e-18, 3.601e-7,-6.073e-19,
     2.736e-3,-1.363e-3, 2.503e-3,-3.322e-8,-2e-5,
     6.909e-3,-3.443e-3, 6.196e-3, 0.01666, 1,
     3.899,   -2.033,   -3.42,   -0.01,    6e-14,
};

/* Case 5: N=6 */
static const f64 c5_scale[] = {2, 1, 1, 1, 1, 3};
static const f64 c5_e[] = {
     1,      1,      2.776e-16,-2.405e-17, 0,      1,
     0,      0.75,   1,       0.0852,    0,     -1.52e-17,
     0,      0.75,  -0.8093,  1,         0,     -1.52e-17,
     0,      0.75,  -0.09533,-0.5426,    1,     -1.52e-17,
     0,      0.75,  -0.09533,-0.5426,   -1,     -1.52e-17,
     0,      0,      0,       0,         0,      4.559e-17,
};
static const f64 c5_ein[] = {
     0,      0.75,  -0.8093,  1,         0,     -1.52e-17,
     0,      0.75,   1,       0.0852,    0,     -1.52e-17,
     1,      1,      2.776e-16,-2.405e-17, 0,    1,
     0,      0,      0,       0,         0,      4.559e-17,
     0,      0.75,  -0.09533,-0.5426,   -1,     -1.52e-17,
     0,      0.75,  -0.09533,-0.5426,    1,     -1.52e-17,
};

/* Case 6: N=7 */
static const f64 c6_scale[] = {2, 1e-3, 0.01, 10, 0.1, 0, 5};
static const f64 c6_e[] = {
     1,     -0.01105, 0.03794,-0.09378,-0.03481, 0.4465, -0.03602,
     0,     -0.4556, -0.4545,  1,       0.4639, -0.6512,  0.4781,
     0,     -0.2734, -0.7946,  0.6303,  1,      -0.6279,  1,
     0,      1,      -6.939e-18, 0.04259,-0.6495,-0.5581, -0.6452,
     0,     -0.3904, -0.4029, -0.1685, -0.9429,  1,      -0.9371,
     0,      0,       0,       0,       0,      -0.2558,  3.308e-4,
     0,      0,       0,       0,       0,       0,      -1.985e-3,
};
static const f64 c6_ein[] = {
     0,      0,       0,       0,       0,      -0.2558,  3.308e-4,
     0,     -4.556e-4,-4.545e-4, 1e-3,   4.639e-4,-6.512e-4, 4.781e-4,
     1,     -0.01105, 0.03794,-0.09378,-0.03481, 0.4465, -0.03602,
     0,      10,     -6.939e-17, 0.4259, -6.495,  -5.581,  -6.452,
     0,     -0.03904,-0.04029,-0.01685,-0.09429, 0.1,    -0.09371,
     0,      0,       0,       0,       0,       0,      -1.985e-3,
     0,     -2.734e-3,-7.946e-3, 6.303e-3, 0.01, -6.279e-3, 0.01,
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

    f64 eps = dlamch("E");
    f64 safmin = dlamch("S");

    f64 e[LDE * LDE], ein[LDE * LDE];
    INT info;
    f64 rmax = 0.0, vmax;
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

        dgebak("B", "R", n, ilo, ihi, c->scale, n, e, LDE, &info);

        if (info != 0) {
            ninfo++;
            lmax_info = knt;
        }

        vmax = 0.0;
        for (INT i = 0; i < n; i++) {
            for (INT j = 0; j < n; j++) {
                f64 x = fabs(e[i + j * LDE] - ein[i + j * LDE]) / eps;
                if (fabs(e[i + j * LDE]) > safmin)
                    x = x / fabs(e[i + j * LDE]);
                if (x > vmax) vmax = x;
            }
        }

        if (vmax > rmax) {
            lmax_resid = knt;
            rmax = vmax;
        }
    }

    print_message("DGEBAK: %d cases, max residual = %.3e (case %d)\n",
                  knt, rmax, lmax_resid);
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
