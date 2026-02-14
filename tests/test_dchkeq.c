/**
 * @file test_dchkeq.c
 * @brief Test suite for equilibration routines.
 *
 * This is a faithful port of LAPACK's TESTING/LIN/dchkeq.f to C using CMocka.
 * Tests DGEEQU, DGBEQU, DPOEQU, DPPEQU, and DPBEQU.
 *
 * Each routine is tested with synthetic matrices containing powers of 10
 * to verify that row/column scaling factors are computed correctly.
 */

#include "test_harness.h"

#define THRESH 10.0  /* From LAPACK: should be between 2 and 10 */

/* Parameters from dchkeq.f */
#define NSZ   5             /* Maximum matrix size for tests */
#define NSZB  (3*NSZ - 2)   /* Band storage size: 13 */
#define NSZP  ((NSZ*(NSZ+1))/2)  /* Packed storage size: 15 */
#define NPOW  (2*NSZ + 1)   /* Number of powers: 11 */

/* Equilibration routines under test */
extern void dgeequ(const int m, const int n, const f64* A, const int lda,
                   f64* r, f64* c, f64* rowcnd, f64* colcnd,
                   f64* amax, int* info);

extern void dgbequ(const int m, const int n, const int kl, const int ku,
                   const f64* AB, const int ldab,
                   f64* r, f64* c, f64* rowcnd, f64* colcnd,
                   f64* amax, int* info);

extern void dpoequ(const int n, const f64* A, const int lda,
                   f64* s, f64* scond, f64* amax, int* info);

extern void dppequ(const char* uplo, const int n, const f64* AP,
                   f64* s, f64* scond, f64* amax, int* info);

extern void dpbequ(const char* uplo, const int n, const int kd,
                   const f64* AB, const int ldab,
                   f64* s, f64* scond, f64* amax, int* info);

/* Utility */
extern f64 dlamch(const char* cmach);

/**
 * Test workspace - shared across all tests.
 */
typedef struct {
    f64 A[NSZ * NSZ];        /* General dense matrix */
    f64 AB[NSZB * NSZ];      /* Band matrix storage */
    f64 AP[NSZP];            /* Packed matrix storage */
    f64 R[NSZ];              /* Row scale factors */
    f64 C[NSZ];              /* Column scale factors */
    f64 POW[NPOW];           /* POW[i] = 10^i */
    f64 RPOW[NPOW];          /* RPOW[i] = 10^(-i) */
    f64 eps;                 /* Machine epsilon */
} dchkeq_workspace_t;

static dchkeq_workspace_t* g_ws = NULL;

/**
 * Group setup - allocate workspace and initialize power arrays.
 */
static int group_setup(void** state)
{
    (void)state;

    g_ws = malloc(sizeof(dchkeq_workspace_t));
    if (!g_ws) return -1;

    /* Initialize power arrays: POW[i] = 10^i, RPOW[i] = 10^(-i) */
    for (int i = 0; i < NPOW; i++) {
        g_ws->POW[i] = pow(10.0, (f64)i);
        g_ws->RPOW[i] = 1.0 / g_ws->POW[i];
    }

    g_ws->eps = dlamch("P");

    return 0;
}

/**
 * Group teardown - free workspace.
 */
static int group_teardown(void** state)
{
    (void)state;
    free(g_ws);
    g_ws = NULL;
    return 0;
}

/**
 * Test DGEEQU - Row and column equilibration for general matrices.
 *
 * Tests M x N matrices for M, N in [0, NSZ].
 * Matrix entries: A(i,j) = POW(i+j+1) * (-1)^(i+j)
 * Expected row scaling: R(i) = RPOW(i+N+1)
 * Expected column scaling: C(j) = POW(N-j+1)
 * Expected RCOND = RPOW(M), CCOND = RPOW(N)
 * Expected NORM = POW(N+M+1)
 */
static void test_dgeequ(void** state)
{
    (void)state;

    f64 resid = 0.0;
    f64 rcond, ccond, norm;
    int info;

    /* Test all M x N combinations */
    for (int n = 0; n <= NSZ; n++) {
        for (int m = 0; m <= NSZ; m++) {

            /* Build test matrix: A(i,j) = POW(i+j+1) * (-1)^(i+j) */
            /* Note: 0-based indexing in C vs 1-based in Fortran */
            for (int j = 0; j < NSZ; j++) {
                for (int i = 0; i < NSZ; i++) {
                    if (i < m && j < n) {
                        /* Fortran: A(I,J) = POW(I+J+1) * (-1)**(I+J) */
                        /* i,j are 0-based, so we use (i+1)+(j+1)+1 = i+j+3 for POW index */
                        /* But Fortran POW is 1-indexed: POW(k) = 10^(k-1) */
                        /* So POW(I+J+1) with I,J 1-based = 10^(I+J+1-1) = 10^(I+J) */
                        /* With 0-based: 10^(i+j) = g_ws->POW[i+j] */
                        /* Actually, let's be careful: */
                        /* Fortran: DO I=1,NSZ; IF I<=M AND J<=N THEN A(I,J) = POW(I+J+1) */
                        /* POW(I+J+1) where I,J start at 1, so min is POW(3) = 10^2 */
                        /* Our POW[k] = 10^k, so we need POW[i+j+2] for 0-based i,j */
                        int sign = ((i + j) % 2 == 0) ? 1 : -1;
                        g_ws->A[i + j * NSZ] = g_ws->POW[i + j + 2] * sign;
                    } else {
                        g_ws->A[i + j * NSZ] = 0.0;
                    }
                }
            }

            dgeequ(m, n, g_ws->A, NSZ, g_ws->R, g_ws->C, &rcond, &ccond, &norm, &info);

            if (info != 0) {
                resid = 1.0;
            } else if (n != 0 && m != 0) {
                /* Check RCOND = RPOW(M) */
                /* Fortran: RPOW(M) with 1-based M in [1,NSZ] */
                /* RPOW(k) = 10^(-(k-1)), so RPOW(M) = 10^(-(M-1)) */
                /* Our RPOW[k] = 10^(-k), so RPOW(M) = RPOW[M-1] with 0-based */
                /* But m here is the actual dimension (0 to NSZ), not 1-based */
                /* For m>0, RPOW(m) in Fortran = 10^(-(m-1)) = g_ws->RPOW[m-1] */
                f64 expected_rcond = g_ws->RPOW[m - 1];
                resid = fmax(resid, fabs((rcond - expected_rcond) / expected_rcond));

                /* Check CCOND = RPOW(N) */
                f64 expected_ccond = g_ws->RPOW[n - 1];
                resid = fmax(resid, fabs((ccond - expected_ccond) / expected_ccond));

                /* Check NORM = POW(N+M+1) */
                /* Fortran POW(N+M+1) = 10^(N+M+1-1) = 10^(N+M) = g_ws->POW[n+m] */
                f64 expected_norm = g_ws->POW[n + m];
                resid = fmax(resid, fabs((norm - expected_norm) / expected_norm));

                /* Check row scalings R(I) = RPOW(I+N+1) */
                /* Fortran I=1..M: RPOW(I+N+1) = 10^(-(I+N+1-1)) = 10^(-(I+N)) */
                /* 0-based i=0..m-1: expected = 10^(-(i+1+n)) = g_ws->RPOW[i+1+n] */
                for (int i = 0; i < m; i++) {
                    f64 expected_r = g_ws->RPOW[i + 1 + n];
                    resid = fmax(resid, fabs((g_ws->R[i] - expected_r) / expected_r));
                }

                /* Check column scalings C(J) = POW(N-J+1) */
                /* Fortran J=1..N: POW(N-J+1) = 10^(N-J+1-1) = 10^(N-J) */
                /* 0-based j=0..n-1: expected = 10^(n-(j+1)) = 10^(n-j-1) = g_ws->POW[n-j-1] */
                for (int j = 0; j < n; j++) {
                    f64 expected_c = g_ws->POW[n - j - 1];
                    resid = fmax(resid, fabs((g_ws->C[j] - expected_c) / expected_c));
                }
            }
        }
    }

    /* Test with zero rows */
    /* Set row MAX(NSZ-1,1) to zero (Fortran 1-based), which is row index max(NSZ-2,0) in 0-based */
    int zero_row = NSZ - 2;  /* 0-based index; for NSZ=5 this is row 3 */
    for (int j = 0; j < NSZ; j++) {
        g_ws->A[zero_row + j * NSZ] = 0.0;
    }
    dgeequ(NSZ, NSZ, g_ws->A, NSZ, g_ws->R, g_ws->C, &rcond, &ccond, &norm, &info);
    /* Expected INFO = MAX(NSZ-1,1) in 1-based = zero_row + 1 */
    int expected_info = zero_row + 1;
    if (info != expected_info) {
        resid = 1.0;
    }

    /* Restore row and test with zero column */
    for (int j = 0; j < NSZ; j++) {
        g_ws->A[zero_row + j * NSZ] = 1.0;
    }
    int zero_col = zero_row;  /* Same index for column */
    for (int i = 0; i < NSZ; i++) {
        g_ws->A[i + zero_col * NSZ] = 0.0;
    }
    dgeequ(NSZ, NSZ, g_ws->A, NSZ, g_ws->R, g_ws->C, &rcond, &ccond, &norm, &info);
    /* Expected INFO = NSZ + MAX(NSZ-1,1) in 1-based = NSZ + zero_col + 1 */
    expected_info = NSZ + zero_col + 1;
    if (info != expected_info) {
        resid = 1.0;
    }

    /* Normalize by epsilon */
    resid = resid / g_ws->eps;

    assert_residual_ok(resid);
}

/**
 * Test DGBEQU - Row and column equilibration for general banded matrices.
 */
static void test_dgbequ(void** state)
{
    (void)state;

    f64 resid = 0.0;
    f64 rcond, ccond, norm;
    int info;

    for (int n = 0; n <= NSZ; n++) {
        for (int m = 0; m <= NSZ; m++) {
            int kl_max = (m > 0) ? (m - 1) : 0;
            int ku_max = (n > 0) ? (n - 1) : 0;

            for (int kl = 0; kl <= kl_max; kl++) {
                for (int ku = 0; ku <= ku_max; ku++) {

                    /* Zero the band matrix */
                    for (int j = 0; j < NSZ; j++) {
                        for (int i = 0; i < NSZB; i++) {
                            g_ws->AB[i + j * NSZB] = 0.0;
                        }
                    }

                    /* Fill band matrix: only elements within band */
                    /* Fortran: DO J=1,N; DO I=1,M; IF I<=MIN(M,J+KL) AND I>=MAX(1,J-KU) */
                    /* Band storage: AB(KU+1+I-J, J) */
                    for (int j = 0; j < n; j++) {
                        for (int i = 0; i < m; i++) {
                            int j1 = j + 1;  /* 1-based */
                            int i1 = i + 1;  /* 1-based */
                            int imin = (1 > j1 - ku) ? 1 : (j1 - ku);
                            int imax = (m < j1 + kl) ? m : (j1 + kl);

                            if (i1 >= imin && i1 <= imax) {
                                /* AB(KU+1+I-J, J) in Fortran 1-based */
                                /* 0-based: AB[ku + i - j, j] */
                                int band_row = ku + i - j;
                                int sign = ((i + j) % 2 == 0) ? 1 : -1;
                                g_ws->AB[band_row + j * NSZB] = g_ws->POW[i + j + 2] * sign;
                            }
                        }
                    }

                    dgbequ(m, n, kl, ku, g_ws->AB, NSZB, g_ws->R, g_ws->C,
                           &rcond, &ccond, &norm, &info);

                    if (info != 0) {
                        /* Check if this is an expected zero row/column case */
                        /* Fortran: IF NOT ((N+KL<M AND INFO=N+KL+1) OR (M+KU<N AND INFO=2*M+KU+1)) */
                        int expected1 = (n + kl < m) ? (n + kl + 1) : -1;
                        int expected2 = (m + ku < n) ? (2 * m + ku + 1) : -1;
                        if (info != expected1 && info != expected2) {
                            resid = 1.0;
                        }
                    } else if (n != 0 && m != 0) {
                        /* Check RCOND = min(R)/max(R) */
                        f64 rcmin = g_ws->R[0];
                        f64 rcmax = g_ws->R[0];
                        for (int i = 0; i < m; i++) {
                            rcmin = fmin(rcmin, g_ws->R[i]);
                            rcmax = fmax(rcmax, g_ws->R[i]);
                        }
                        f64 ratio = rcmin / rcmax;
                        resid = fmax(resid, fabs((rcond - ratio) / ratio));

                        /* Check CCOND = min(C)/max(C) */
                        rcmin = g_ws->C[0];
                        rcmax = g_ws->C[0];
                        for (int j = 0; j < n; j++) {
                            rcmin = fmin(rcmin, g_ws->C[j]);
                            rcmax = fmax(rcmax, g_ws->C[j]);
                        }
                        ratio = rcmin / rcmax;
                        resid = fmax(resid, fabs((ccond - ratio) / ratio));

                        /* Check NORM = POW(N+M+1) */
                        f64 expected_norm = g_ws->POW[n + m];
                        resid = fmax(resid, fabs((norm - expected_norm) / expected_norm));

                        /* Check that equilibrated matrix has max element ~1 per row */
                        for (int i = 0; i < m; i++) {
                            rcmax = 0.0;
                            for (int j = 0; j < n; j++) {
                                int i1 = i + 1;
                                int j1 = j + 1;
                                if (i1 <= j1 + kl && i1 >= j1 - ku) {
                                    ratio = fabs(g_ws->R[i] * g_ws->POW[i + j + 2] * g_ws->C[j]);
                                    rcmax = fmax(rcmax, ratio);
                                }
                            }
                            resid = fmax(resid, fabs(1.0 - rcmax));
                        }

                        /* Check that equilibrated matrix has max element ~1 per column */
                        for (int j = 0; j < n; j++) {
                            rcmax = 0.0;
                            for (int i = 0; i < m; i++) {
                                int i1 = i + 1;
                                int j1 = j + 1;
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

/**
 * Test DPOEQU - Equilibration for symmetric positive definite matrices.
 */
static void test_dpoequ(void** state)
{
    (void)state;

    f64 resid = 0.0;
    f64 scond, norm;
    int info;

    for (int n = 0; n <= NSZ; n++) {
        /* Build diagonal test matrix: A(i,i) = POW(2*i+1) */
        /* Fortran I=1..N: A(I,I) = POW(I+J+1) = POW(2*I+1) since I=J */
        /* POW(2*I+1) = 10^(2*I+1-1) = 10^(2*I) */
        /* 0-based: 10^(2*(i+1)) = 10^(2*i+2) = g_ws->POW[2*i+2] */
        for (int j = 0; j < NSZ; j++) {
            for (int i = 0; i < NSZ; i++) {
                if (i < n && j == i) {
                    int sign = ((i + j) % 2 == 0) ? 1 : -1;
                    g_ws->A[i + j * NSZ] = g_ws->POW[2 * i + 2] * sign;
                } else {
                    g_ws->A[i + j * NSZ] = 0.0;
                }
            }
        }

        dpoequ(n, g_ws->A, NSZ, g_ws->R, &scond, &norm, &info);

        if (info != 0) {
            resid = 1.0;
        } else if (n != 0) {
            /* Check SCOND = RPOW(N) = 10^(-(N-1)) = g_ws->RPOW[n-1] */
            f64 expected_scond = g_ws->RPOW[n - 1];
            resid = fmax(resid, fabs((scond - expected_scond) / expected_scond));

            /* Check NORM = POW(2*N+1) = 10^(2*N) = g_ws->POW[2*n] */
            f64 expected_norm = g_ws->POW[2 * n];
            resid = fmax(resid, fabs((norm - expected_norm) / expected_norm));

            /* Check S(I) = RPOW(I+1) = 10^(-I) = g_ws->RPOW[i+1] for 0-based i */
            for (int i = 0; i < n; i++) {
                f64 expected_s = g_ws->RPOW[i + 1];
                resid = fmax(resid, fabs((g_ws->R[i] - expected_s) / expected_s));
            }
        }
    }

    /* Test with negative diagonal element */
    int diag_idx = NSZ - 2;  /* for NSZ=5 this is index 3 */
    g_ws->A[diag_idx + diag_idx * NSZ] = -1.0;
    dpoequ(NSZ, g_ws->A, NSZ, g_ws->R, &scond, &norm, &info);
    int expected_info = diag_idx + 1;
    if (info != expected_info) {
        resid = 1.0;
    }

    resid = resid / g_ws->eps;
    assert_residual_ok(resid);
}

/**
 * Test DPPEQU - Equilibration for symmetric positive definite packed matrices.
 */
static void test_dppequ(void** state)
{
    (void)state;

    f64 resid = 0.0;
    f64 scond, norm;
    int info;

    for (int n = 0; n <= NSZ; n++) {
        /* Upper triangular packed storage */
        int np = (n * (n + 1)) / 2;
        for (int i = 0; i < np; i++) {
            g_ws->AP[i] = 0.0;
        }
        /* Set diagonal: AP((I*(I+1))/2) = POW(2*I+1) for I=1..N (1-based) */
        /* 0-based i=0..n-1: position = ((i+1)*(i+2))/2 - 1 = (i+1)*(i+2)/2 - 1 */
        /* Value = POW(2*(i+1)+1) = 10^(2*i+2) = g_ws->POW[2*i+2] */
        for (int i = 0; i < n; i++) {
            int pos = ((i + 1) * (i + 2)) / 2 - 1;
            g_ws->AP[pos] = g_ws->POW[2 * i + 2];
        }

        dppequ("U", n, g_ws->AP, g_ws->R, &scond, &norm, &info);

        if (info != 0) {
            resid = 1.0;
        } else if (n != 0) {
            f64 expected_scond = g_ws->RPOW[n - 1];
            resid = fmax(resid, fabs((scond - expected_scond) / expected_scond));

            f64 expected_norm = g_ws->POW[2 * n];
            resid = fmax(resid, fabs((norm - expected_norm) / expected_norm));

            for (int i = 0; i < n; i++) {
                f64 expected_s = g_ws->RPOW[i + 1];
                resid = fmax(resid, fabs((g_ws->R[i] - expected_s) / expected_s));
            }
        }

        /* Lower triangular packed storage */
        for (int i = 0; i < np; i++) {
            g_ws->AP[i] = 0.0;
        }
        /* Set diagonal for lower: different positions */
        /* Fortran: J=1; DO I=1,N; AP(J) = POW(2*I+1); J = J + (N-I+1) */
        /* 0-based: j starts at 0, increment by (n - i) for each i */
        int j = 0;
        for (int i = 0; i < n; i++) {
            g_ws->AP[j] = g_ws->POW[2 * i + 2];
            j = j + (n - i);
        }

        dppequ("L", n, g_ws->AP, g_ws->R, &scond, &norm, &info);

        if (info != 0) {
            resid = 1.0;
        } else if (n != 0) {
            f64 expected_scond = g_ws->RPOW[n - 1];
            resid = fmax(resid, fabs((scond - expected_scond) / expected_scond));

            f64 expected_norm = g_ws->POW[2 * n];
            resid = fmax(resid, fabs((norm - expected_norm) / expected_norm));

            for (int i = 0; i < n; i++) {
                f64 expected_s = g_ws->RPOW[i + 1];
                resid = fmax(resid, fabs((g_ws->R[i] - expected_s) / expected_s));
            }
        }
    }

    /* Test with negative diagonal in lower storage */
    /* Position: I = (NSZ*(NSZ+1))/2 - 2 (Fortran 1-based index) */
    /* This is the second-to-last diagonal element */
    int neg_pos = (NSZ * (NSZ + 1)) / 2 - 3;  /* 0-based */
    g_ws->AP[neg_pos] = -1.0;
    dppequ("L", NSZ, g_ws->AP, g_ws->R, &scond, &norm, &info);
    int expected_info = NSZ - 1;  /* for NSZ=5 this is 4 */
    if (info != expected_info) {
        resid = 1.0;
    }

    resid = resid / g_ws->eps;
    assert_residual_ok(resid);
}

/**
 * Test DPBEQU - Equilibration for symmetric positive definite banded matrices.
 */
static void test_dpbequ(void** state)
{
    (void)state;

    f64 resid = 0.0;
    f64 scond, norm;
    int info;

    for (int n = 0; n <= NSZ; n++) {
        int kl_max = (n > 0) ? (n - 1) : 0;

        for (int kl = 0; kl <= kl_max; kl++) {
            /* Test upper triangular storage */
            for (int j = 0; j < NSZ; j++) {
                for (int i = 0; i < NSZB; i++) {
                    g_ws->AB[i + j * NSZB] = 0.0;
                }
            }
            /* Set diagonal: AB(KL+1, J) = POW(2*J+1) for J=1..N */
            /* 0-based: AB[kl, j] = POW[2*(j+1)+1-1] = POW[2*j+2] */
            for (int j = 0; j < n; j++) {
                g_ws->AB[kl + j * NSZB] = g_ws->POW[2 * j + 2];
            }

            dpbequ("U", n, kl, g_ws->AB, NSZB, g_ws->R, &scond, &norm, &info);

            if (info != 0) {
                resid = 1.0;
            } else if (n != 0) {
                f64 expected_scond = g_ws->RPOW[n - 1];
                resid = fmax(resid, fabs((scond - expected_scond) / expected_scond));

                f64 expected_norm = g_ws->POW[2 * n];
                resid = fmax(resid, fabs((norm - expected_norm) / expected_norm));

                for (int i = 0; i < n; i++) {
                    f64 expected_s = g_ws->RPOW[i + 1];
                    resid = fmax(resid, fabs((g_ws->R[i] - expected_s) / expected_s));
                }
            }

            /* Test with negative diagonal (upper) */
            if (n != 0) {
                int neg_col = (n - 1 > 1) ? (n - 2) : 0;
                g_ws->AB[kl + neg_col * NSZB] = -1.0;
                dpbequ("U", n, kl, g_ws->AB, NSZB, g_ws->R, &scond, &norm, &info);
                int expected_info = neg_col + 1;
                if (info != expected_info) {
                    resid = 1.0;
                }
            }

            /* Test lower triangular storage */
            for (int j = 0; j < NSZ; j++) {
                for (int i = 0; i < NSZB; i++) {
                    g_ws->AB[i + j * NSZB] = 0.0;
                }
            }
            /* Set diagonal: AB(1, J) = POW(2*J+1) for J=1..N */
            /* 0-based: AB[0, j] = POW[2*j+2] */
            for (int j = 0; j < n; j++) {
                g_ws->AB[0 + j * NSZB] = g_ws->POW[2 * j + 2];
            }

            dpbequ("L", n, kl, g_ws->AB, NSZB, g_ws->R, &scond, &norm, &info);

            if (info != 0) {
                resid = 1.0;
            } else if (n != 0) {
                f64 expected_scond = g_ws->RPOW[n - 1];
                resid = fmax(resid, fabs((scond - expected_scond) / expected_scond));

                f64 expected_norm = g_ws->POW[2 * n];
                resid = fmax(resid, fabs((norm - expected_norm) / expected_norm));

                for (int i = 0; i < n; i++) {
                    f64 expected_s = g_ws->RPOW[i + 1];
                    resid = fmax(resid, fabs((g_ws->R[i] - expected_s) / expected_s));
                }
            }

            /* Test with negative diagonal (lower) */
            if (n != 0) {
                int neg_col = (n - 1 > 1) ? (n - 2) : 0;
                g_ws->AB[0 + neg_col * NSZB] = -1.0;
                dpbequ("L", n, kl, g_ws->AB, NSZB, g_ws->R, &scond, &norm, &info);
                int expected_info = neg_col + 1;
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
        cmocka_unit_test(test_dgeequ),
        cmocka_unit_test(test_dgbequ),
        cmocka_unit_test(test_dpoequ),
        cmocka_unit_test(test_dppequ),
        cmocka_unit_test(test_dpbequ),
    };

    return cmocka_run_group_tests_name("dchkeq", tests, group_setup, group_teardown);
}
