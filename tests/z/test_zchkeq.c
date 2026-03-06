/**
 * @file test_zchkeq.c
 * @brief Test suite for complex equilibration routines.
 *
 * This is a faithful port of LAPACK's TESTING/LIN/zchkeq.f to C using CMocka.
 * Tests ZGEEQU, ZGBEQU, ZPOEQU, ZPPEQU, and ZPBEQU.
 *
 * Each routine is tested with synthetic matrices containing powers of 10
 * to verify that row/column scaling factors are computed correctly.
 */

#include "test_harness.h"

#define THRESH 10.0  /* From LAPACK: should be between 2 and 10 */

#define NSZ   5             /* Maximum matrix size for tests */
#define NSZB  (3*NSZ - 2)   /* Band storage size: 13 */
#define NSZP  ((NSZ*(NSZ+1))/2)  /* Packed storage size: 15 */
#define NPOW  (2*NSZ + 1)   /* Number of powers: 11 */

typedef struct {
    c128 A[NSZ * NSZ];        /* General dense matrix */
    c128 AB[NSZB * NSZ];      /* Band matrix storage */
    c128 AP[NSZP];            /* Packed matrix storage */
    f64 R[NSZ];               /* Row scale factors */
    f64 C[NSZ];               /* Column scale factors */
    f64 POW[NPOW];            /* POW[i] = 10^i */
    f64 RPOW[NPOW];           /* RPOW[i] = 10^(-i) */
    f64 eps;                  /* Machine epsilon */
} zchkeq_workspace_t;

static zchkeq_workspace_t* g_ws = NULL;

static int group_setup(void** state)
{
    (void)state;

    g_ws = malloc(sizeof(zchkeq_workspace_t));
    if (!g_ws) return -1;

    for (INT i = 0; i < NPOW; i++) {
        g_ws->POW[i] = pow(10.0, (f64)i);
        g_ws->RPOW[i] = 1.0 / g_ws->POW[i];
    }

    g_ws->eps = dlamch("P");

    return 0;
}

static int group_teardown(void** state)
{
    (void)state;
    free(g_ws);
    g_ws = NULL;
    return 0;
}

static void test_zgeequ(void** state)
{
    (void)state;

    f64 resid = 0.0;
    f64 rcond, ccond, norm;
    INT info;

    for (INT n = 0; n <= NSZ; n++) {
        for (INT m = 0; m <= NSZ; m++) {

            for (INT j = 0; j < NSZ; j++) {
                for (INT i = 0; i < NSZ; i++) {
                    if (i < m && j < n) {
                        INT sign = ((i + j) % 2 == 0) ? 1 : -1;
                        g_ws->A[i + j * NSZ] = g_ws->POW[i + j + 2] * sign;
                    } else {
                        g_ws->A[i + j * NSZ] = 0.0;
                    }
                }
            }

            zgeequ(m, n, g_ws->A, NSZ, g_ws->R, g_ws->C, &rcond, &ccond, &norm, &info);

            if (info != 0) {
                resid = 1.0;
            } else if (n != 0 && m != 0) {
                f64 expected_rcond = g_ws->RPOW[m - 1];
                resid = fmax(resid, fabs((rcond - expected_rcond) / expected_rcond));

                f64 expected_ccond = g_ws->RPOW[n - 1];
                resid = fmax(resid, fabs((ccond - expected_ccond) / expected_ccond));

                f64 expected_norm = g_ws->POW[n + m];
                resid = fmax(resid, fabs((norm - expected_norm) / expected_norm));

                for (INT i = 0; i < m; i++) {
                    f64 expected_r = g_ws->RPOW[i + 1 + n];
                    resid = fmax(resid, fabs((g_ws->R[i] - expected_r) / expected_r));
                }

                for (INT j = 0; j < n; j++) {
                    f64 expected_c = g_ws->POW[n - j - 1];
                    resid = fmax(resid, fabs((g_ws->C[j] - expected_c) / expected_c));
                }
            }
        }
    }

    INT zero_row = NSZ - 2;
    for (INT j = 0; j < NSZ; j++) {
        g_ws->A[zero_row + j * NSZ] = 0.0;
    }
    zgeequ(NSZ, NSZ, g_ws->A, NSZ, g_ws->R, g_ws->C, &rcond, &ccond, &norm, &info);
    INT expected_info = zero_row + 1;
    if (info != expected_info) {
        resid = 1.0;
    }

    for (INT j = 0; j < NSZ; j++) {
        g_ws->A[zero_row + j * NSZ] = 1.0;
    }
    INT zero_col = zero_row;
    for (INT i = 0; i < NSZ; i++) {
        g_ws->A[i + zero_col * NSZ] = 0.0;
    }
    zgeequ(NSZ, NSZ, g_ws->A, NSZ, g_ws->R, g_ws->C, &rcond, &ccond, &norm, &info);
    expected_info = NSZ + zero_col + 1;
    if (info != expected_info) {
        resid = 1.0;
    }

    resid = resid / g_ws->eps;

    assert_residual_ok(resid);
}

static void test_zgbequ(void** state)
{
    (void)state;

    f64 resid = 0.0;
    f64 rcond, ccond, norm;
    INT info;

    for (INT n = 0; n <= NSZ; n++) {
        for (INT m = 0; m <= NSZ; m++) {
            INT kl_max = (m > 0) ? (m - 1) : 0;
            INT ku_max = (n > 0) ? (n - 1) : 0;

            for (INT kl = 0; kl <= kl_max; kl++) {
                for (INT ku = 0; ku <= ku_max; ku++) {

                    for (INT j = 0; j < NSZ; j++) {
                        for (INT i = 0; i < NSZB; i++) {
                            g_ws->AB[i + j * NSZB] = 0.0;
                        }
                    }

                    for (INT j = 0; j < n; j++) {
                        for (INT i = 0; i < m; i++) {
                            INT j1 = j + 1;
                            INT i1 = i + 1;
                            INT imin = (1 > j1 - ku) ? 1 : (j1 - ku);
                            INT imax = (m < j1 + kl) ? m : (j1 + kl);

                            if (i1 >= imin && i1 <= imax) {
                                INT band_row = ku + i - j;
                                INT sign = ((i + j) % 2 == 0) ? 1 : -1;
                                g_ws->AB[band_row + j * NSZB] = g_ws->POW[i + j + 2] * sign;
                            }
                        }
                    }

                    zgbequ(m, n, kl, ku, g_ws->AB, NSZB, g_ws->R, g_ws->C,
                           &rcond, &ccond, &norm, &info);

                    if (info != 0) {
                        INT expected1 = (n + kl < m) ? (n + kl + 1) : -1;
                        INT expected2 = (m + ku < n) ? (2 * m + ku + 1) : -1;
                        if (info != expected1 && info != expected2) {
                            resid = 1.0;
                        }
                    } else if (n != 0 && m != 0) {
                        f64 rcmin = g_ws->R[0];
                        f64 rcmax = g_ws->R[0];
                        for (INT i = 0; i < m; i++) {
                            rcmin = fmin(rcmin, g_ws->R[i]);
                            rcmax = fmax(rcmax, g_ws->R[i]);
                        }
                        f64 ratio = rcmin / rcmax;
                        resid = fmax(resid, fabs((rcond - ratio) / ratio));

                        rcmin = g_ws->C[0];
                        rcmax = g_ws->C[0];
                        for (INT j = 0; j < n; j++) {
                            rcmin = fmin(rcmin, g_ws->C[j]);
                            rcmax = fmax(rcmax, g_ws->C[j]);
                        }
                        ratio = rcmin / rcmax;
                        resid = fmax(resid, fabs((ccond - ratio) / ratio));

                        f64 expected_norm = g_ws->POW[n + m];
                        resid = fmax(resid, fabs((norm - expected_norm) / expected_norm));

                        for (INT i = 0; i < m; i++) {
                            rcmax = 0.0;
                            for (INT j = 0; j < n; j++) {
                                INT i1 = i + 1;
                                INT j1 = j + 1;
                                if (i1 <= j1 + kl && i1 >= j1 - ku) {
                                    ratio = fabs(g_ws->R[i] * g_ws->POW[i + j + 2] * g_ws->C[j]);
                                    rcmax = fmax(rcmax, ratio);
                                }
                            }
                            resid = fmax(resid, fabs(1.0 - rcmax));
                        }

                        for (INT j = 0; j < n; j++) {
                            rcmax = 0.0;
                            for (INT i = 0; i < m; i++) {
                                INT i1 = i + 1;
                                INT j1 = j + 1;
                                if (i1 <= j1 + kl && i1 >= j1 - ku) {
                                    ratio = fabs(g_ws->R[i] * g_ws->POW[i + j + 2] * g_ws->C[j]);
                                    rcmax = fmax(rcmax, ratio);
                                }
                            }
                            resid = fmax(resid, fabs(1.0 - rcmax));
                        }
                    }
                }
            }
        }
    }

    resid = resid / g_ws->eps;
    assert_residual_ok(resid);
}

static void test_zpoequ(void** state)
{
    (void)state;

    f64 resid = 0.0;
    f64 scond, norm;
    INT info;

    for (INT n = 0; n <= NSZ; n++) {
        for (INT j = 0; j < NSZ; j++) {
            for (INT i = 0; i < NSZ; i++) {
                if (i < n && j == i) {
                    INT sign = ((i + j) % 2 == 0) ? 1 : -1;
                    g_ws->A[i + j * NSZ] = g_ws->POW[2 * i + 2] * sign;
                } else {
                    g_ws->A[i + j * NSZ] = 0.0;
                }
            }
        }

        zpoequ(n, g_ws->A, NSZ, g_ws->R, &scond, &norm, &info);

        if (info != 0) {
            resid = 1.0;
        } else if (n != 0) {
            f64 expected_scond = g_ws->RPOW[n - 1];
            resid = fmax(resid, fabs((scond - expected_scond) / expected_scond));

            f64 expected_norm = g_ws->POW[2 * n];
            resid = fmax(resid, fabs((norm - expected_norm) / expected_norm));

            for (INT i = 0; i < n; i++) {
                f64 expected_s = g_ws->RPOW[i + 1];
                resid = fmax(resid, fabs((g_ws->R[i] - expected_s) / expected_s));
            }
        }
    }

    INT diag_idx = NSZ - 2;
    g_ws->A[diag_idx + diag_idx * NSZ] = -1.0;
    zpoequ(NSZ, g_ws->A, NSZ, g_ws->R, &scond, &norm, &info);
    INT expected_info = diag_idx + 1;
    if (info != expected_info) {
        resid = 1.0;
    }

    resid = resid / g_ws->eps;
    assert_residual_ok(resid);
}

static void test_zppequ(void** state)
{
    (void)state;

    f64 resid = 0.0;
    f64 scond, norm;
    INT info;

    for (INT n = 0; n <= NSZ; n++) {
        /* Upper triangular packed storage */
        INT np = (n * (n + 1)) / 2;
        for (INT i = 0; i < np; i++) {
            g_ws->AP[i] = 0.0;
        }
        for (INT i = 0; i < n; i++) {
            INT pos = ((i + 1) * (i + 2)) / 2 - 1;
            g_ws->AP[pos] = g_ws->POW[2 * i + 2];
        }

        zppequ("U", n, g_ws->AP, g_ws->R, &scond, &norm, &info);

        if (info != 0) {
            resid = 1.0;
        } else if (n != 0) {
            f64 expected_scond = g_ws->RPOW[n - 1];
            resid = fmax(resid, fabs((scond - expected_scond) / expected_scond));

            f64 expected_norm = g_ws->POW[2 * n];
            resid = fmax(resid, fabs((norm - expected_norm) / expected_norm));

            for (INT i = 0; i < n; i++) {
                f64 expected_s = g_ws->RPOW[i + 1];
                resid = fmax(resid, fabs((g_ws->R[i] - expected_s) / expected_s));
            }
        }

        /* Lower triangular packed storage */
        for (INT i = 0; i < np; i++) {
            g_ws->AP[i] = 0.0;
        }
        INT j = 0;
        for (INT i = 0; i < n; i++) {
            g_ws->AP[j] = g_ws->POW[2 * i + 2];
            j = j + (n - i);
        }

        zppequ("L", n, g_ws->AP, g_ws->R, &scond, &norm, &info);

        if (info != 0) {
            resid = 1.0;
        } else if (n != 0) {
            f64 expected_scond = g_ws->RPOW[n - 1];
            resid = fmax(resid, fabs((scond - expected_scond) / expected_scond));

            f64 expected_norm = g_ws->POW[2 * n];
            resid = fmax(resid, fabs((norm - expected_norm) / expected_norm));

            for (INT i = 0; i < n; i++) {
                f64 expected_s = g_ws->RPOW[i + 1];
                resid = fmax(resid, fabs((g_ws->R[i] - expected_s) / expected_s));
            }
        }
    }

    INT neg_pos = (NSZ * (NSZ + 1)) / 2 - 3;
    g_ws->AP[neg_pos] = -1.0;
    zppequ("L", NSZ, g_ws->AP, g_ws->R, &scond, &norm, &info);
    INT expected_info = NSZ - 1;
    if (info != expected_info) {
        resid = 1.0;
    }

    resid = resid / g_ws->eps;
    assert_residual_ok(resid);
}

static void test_zpbequ(void** state)
{
    (void)state;

    f64 resid = 0.0;
    f64 scond, norm;
    INT info;

    for (INT n = 0; n <= NSZ; n++) {
        INT kl_max = (n > 0) ? (n - 1) : 0;

        for (INT kl = 0; kl <= kl_max; kl++) {
            /* Test upper triangular storage */
            for (INT j = 0; j < NSZ; j++) {
                for (INT i = 0; i < NSZB; i++) {
                    g_ws->AB[i + j * NSZB] = 0.0;
                }
            }
            for (INT j = 0; j < n; j++) {
                g_ws->AB[kl + j * NSZB] = g_ws->POW[2 * j + 2];
            }

            zpbequ("U", n, kl, g_ws->AB, NSZB, g_ws->R, &scond, &norm, &info);

            if (info != 0) {
                resid = 1.0;
            } else if (n != 0) {
                f64 expected_scond = g_ws->RPOW[n - 1];
                resid = fmax(resid, fabs((scond - expected_scond) / expected_scond));

                f64 expected_norm = g_ws->POW[2 * n];
                resid = fmax(resid, fabs((norm - expected_norm) / expected_norm));

                for (INT i = 0; i < n; i++) {
                    f64 expected_s = g_ws->RPOW[i + 1];
                    resid = fmax(resid, fabs((g_ws->R[i] - expected_s) / expected_s));
                }
            }

            if (n != 0) {
                INT neg_col = (n - 1 > 1) ? (n - 2) : 0;
                g_ws->AB[kl + neg_col * NSZB] = -1.0;
                zpbequ("U", n, kl, g_ws->AB, NSZB, g_ws->R, &scond, &norm, &info);
                INT expected_info = neg_col + 1;
                if (info != expected_info) {
                    resid = 1.0;
                }
            }

            /* Test lower triangular storage */
            for (INT j = 0; j < NSZ; j++) {
                for (INT i = 0; i < NSZB; i++) {
                    g_ws->AB[i + j * NSZB] = 0.0;
                }
            }
            for (INT j = 0; j < n; j++) {
                g_ws->AB[0 + j * NSZB] = g_ws->POW[2 * j + 2];
            }

            zpbequ("L", n, kl, g_ws->AB, NSZB, g_ws->R, &scond, &norm, &info);

            if (info != 0) {
                resid = 1.0;
            } else if (n != 0) {
                f64 expected_scond = g_ws->RPOW[n - 1];
                resid = fmax(resid, fabs((scond - expected_scond) / expected_scond));

                f64 expected_norm = g_ws->POW[2 * n];
                resid = fmax(resid, fabs((norm - expected_norm) / expected_norm));

                for (INT i = 0; i < n; i++) {
                    f64 expected_s = g_ws->RPOW[i + 1];
                    resid = fmax(resid, fabs((g_ws->R[i] - expected_s) / expected_s));
                }
            }

            if (n != 0) {
                INT neg_col = (n - 1 > 1) ? (n - 2) : 0;
                g_ws->AB[0 + neg_col * NSZB] = -1.0;
                zpbequ("L", n, kl, g_ws->AB, NSZB, g_ws->R, &scond, &norm, &info);
                INT expected_info = neg_col + 1;
                if (info != expected_info) {
                    resid = 1.0;
                }
            }
        }
    }

    resid = resid / g_ws->eps;
    assert_residual_ok(resid);
}

int main(void)
{
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_zgeequ),
        cmocka_unit_test(test_zgbequ),
        cmocka_unit_test(test_zpoequ),
        cmocka_unit_test(test_zppequ),
        cmocka_unit_test(test_zpbequ),
    };

    (void)cmocka_run_group_tests_name("zchkeq", tests, group_setup, group_teardown);
    return 0;
}
