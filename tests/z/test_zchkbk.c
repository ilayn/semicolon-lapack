/**
 * @file test_zchkbk.c
 * @brief Tests ZGEBAK, backward transformation of balanced eigenvectors.
 *
 * Port of LAPACK TESTING/EIG/zchkbk.f with embedded test data from
 * TESTING/zbak.in.
 */

#include "test_harness.h"
#include "verify.h"

#define THRESH 30.0

#define LDE 20

/* ---------- helpers ---------- */

static void rowmajor_to_colmajor(const f64* rm, c128* cm, INT nrow, INT ncol, INT ld)
{
    memset(cm, 0, (size_t)ld * ncol * sizeof(c128));
    for (INT i = 0; i < nrow; i++)
        for (INT j = 0; j < ncol; j++)
            cm[i + j * ld] = CMPLX(rm[i * ncol + j], 0.0);
}

/* ---------- test case data from TESTING/zbak.in ---------- */

typedef struct {
    INT n;
    INT ilo;
    INT ihi;
    const f64* scale;
    const f64* e_rm;
    const f64* ein_rm;
} zbak_case_t;

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
     1,      1,      1,     -0.66667, -0.041667,
     0,     -0.25,  -0.66667, 1,       0.16667,
     0,      0,      0.22222,-1,      -0.5,
     0,      0,      0,      0.5,     1,
     0,      0,      0,      0,      -1,
};
static const f64 c1_ein[] = {
     0,      0,      0,      0,      -1,
     0,      0,      0,      0.5,     1,
     0,      0,      0.22222,-1,      -0.5,
     0,     -0.25,  -0.66667, 1,       0.16667,
     1,      1,      1,     -0.66667, -0.041667,
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
     1,       1.3356e-6, 1,        1,         1,         1,
     0,       1,         0,       -3.0007e-11,-3.2523e-5, 0.01305,
     0,       0,        -8.33e-3,  8.9289e-10,-6.7123e-5, 6.6874e-5,
     0,       0,         0,       -4.4554e-6, -3.355e-3,  3.3448e-3,
     0,       0,         0,        4.4554e-7, -3.3561e-2, 3.3437e-2,
     0,       0,         0,        4.4113e-10, 0.10115,   0.10084,
};
static const f64 c3_ein[] = {
     0,       0,         0,       -4.4554e-4, -0.3355,    0.33448,
     0,       0,         0,        4.4554e-8, -3.3561e-3, 3.3437e-3,
     0,       1,         0,       -3.0007e-11,-3.2523e-5, 0.01305,
     1,       1.3356e-6, 1,        1,         1,         1,
     0,       0,        -8.33e-3,  8.9289e-10,-6.7123e-5, 6.6874e-5,
     0,       0,         0,        4.4113e-10, 0.10115,   0.10084,
};

/* Case 4: N=5 */
static const f64 c4_scale[] = {100, 0.1, 0.01, 1, 10};
static const f64 c4_e[] = {
     1.3663e-4,-6.829e-5,  1.2516e-4, 1,          1.9503e-15,
     1,         1,        -2.7756e-17, 3.6012e-6, -6.0728e-18,
     0.27355,  -0.13627,   0.2503,   -3.3221e-6, -2e-3,
     6.9088e-3,-3.4434e-3, 6.1959e-3, 0.016661,   1,
     0.38988,  -0.20327,  -0.342,    -1e-3,       6.0004e-15,
};
static const f64 c4_ein[] = {
     0.013663, -6.829e-3,  0.012516,  100,         1.9503e-13,
     0.1,       0.1,      -2.7756e-18, 3.6012e-7, -6.0728e-19,
     2.7355e-3,-1.3627e-3, 2.503e-3, -3.3221e-8,  -2e-5,
     6.9088e-3,-3.4434e-3, 6.1959e-3, 0.016661,    1,
     3.8988,   -2.0327,   -3.42,     -0.01,        6.0004e-14,
};

/* Case 5: N=6 */
static const f64 c5_scale[] = {2, 1, 1, 1, 1, 3};
static const f64 c5_e[] = {
     1,      1,       2.7764e-16,-2.4046e-17, 0,       1,
     0,      0.75,    1,          0.085197,   0,      -1.5196e-17,
     0,      0.75,   -0.80934,    1,          0,      -1.5196e-17,
     0,      0.75,   -0.095328,  -0.5426,     1,      -1.5196e-17,
     0,      0.75,   -0.095328,  -0.5426,    -1,      -1.5196e-17,
     0,      0,       0,          0,          0,       4.5588e-17,
};
static const f64 c5_ein[] = {
     0,      0.75,   -0.80934,    1,          0,      -1.5196e-17,
     0,      0.75,    1,          0.085197,   0,      -1.5196e-17,
     1,      1,       2.7764e-16,-2.4046e-17, 0,       1,
     0,      0,       0,          0,          0,       4.5588e-17,
     0,      0.75,   -0.095328,  -0.5426,    -1,      -1.5196e-17,
     0,      0.75,   -0.095328,  -0.5426,     1,      -1.5196e-17,
};

/* Case 6: N=7 */
static const f64 c6_scale[] = {2, 1e-3, 0.01, 10, 0.1, 0, 5};
static const f64 c6_e[] = {
     1,      -0.011048,  0.037942, -0.093781, -0.034815,  0.44651,  -0.036016,
     0,      -0.45564,  -0.45447,   1,         0.46394,  -0.65116,   0.47808,
     0,      -0.27336,  -0.79459,   0.63028,   1,        -0.62791,   1,
     0,       1,        -6.9389e-18, 0.042585, -0.64954,  -0.55814,  -0.64516,
     0,      -0.39041,  -0.40294,  -0.16849,  -0.94294,   1,        -0.93714,
     0,       0,         0,         0,         0,        -0.25581,    3.3085e-4,
     0,       0,         0,         0,         0,         0,         -1.9851e-3,
};
static const f64 c6_ein[] = {
     0,       0,         0,         0,         0,        -0.25581,    3.3085e-4,
     0,      -4.5564e-4,-4.5447e-4, 1e-3,      4.6394e-4,-6.5116e-4,  4.7808e-4,
     1,      -0.011048,  0.037942, -0.093781, -0.034815,  0.44651,   -0.036016,
     0,       10,       -6.9389e-17, 0.42585,  -6.4954,   -5.5814,   -6.4516,
     0,      -0.039041, -0.040294, -0.016849, -0.094294,   0.1,      -0.093714,
     0,       0,         0,         0,         0,          0,         -1.9851e-3,
     0,      -2.7336e-3,-7.9459e-3, 6.3028e-3, 0.01,     -6.2791e-3,  0.01,
};

static const zbak_case_t cases[] = {
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

static void test_zgebak(void** state)
{
    (void)state;

    f64 eps = dlamch("E");
    f64 safmin = dlamch("S");

    c128 e[LDE * LDE], ein[LDE * LDE];
    INT info;
    f64 rmax = 0.0, vmax;
    INT ninfo = 0, knt = 0;
    INT lmax_info = 0, lmax_resid = 0;

    for (INT tc = 0; tc < NCASES; tc++) {
        const zbak_case_t* c = &cases[tc];
        INT n = c->n;
        INT ilo = c->ilo;
        INT ihi = c->ihi;

        rowmajor_to_colmajor(c->e_rm, e, n, n, LDE);
        rowmajor_to_colmajor(c->ein_rm, ein, n, n, LDE);

        knt++;

        zgebak("B", "R", n, ilo, ihi, c->scale, n, e, LDE, &info);

        if (info != 0) {
            ninfo++;
            lmax_info = knt;
        }

        vmax = 0.0;
        for (INT i = 0; i < n; i++) {
            for (INT j = 0; j < n; j++) {
                f64 x = cabs1(e[i + j * LDE] - ein[i + j * LDE]) / eps;
                if (cabs1(e[i + j * LDE]) > safmin)
                    x = x / cabs1(e[i + j * LDE]);
                if (x > vmax) vmax = x;
            }
        }

        if (vmax > rmax) {
            lmax_resid = knt;
            rmax = vmax;
        }
    }

    fprintf(stderr, "ZGEBAK: %d cases, max residual = %.3e (case %d)\n",
                  knt, rmax, lmax_resid);
    if (ninfo > 0)
        fprintf(stderr, "  INFO errors: %d (case %d)\n", ninfo, lmax_info);

    assert_true(ninfo == 0);
    assert_residual_ok(rmax);
}

int main(void)
{
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_zgebak),
    };
    return cmocka_run_group_tests_name("zchkbk", tests, NULL, NULL);
}
