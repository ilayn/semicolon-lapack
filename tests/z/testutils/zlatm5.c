/**
 * @file zlatm5.c
 * @brief ZLATM5 generates matrices involved in the Generalized Sylvester equation.
 *
 * Faithful port of LAPACK TESTING/MATGEN/zlatm5.f
 */

#include <math.h>
#include <complex.h>
#include "semicolon_cblas.h"
#include "verify.h"

/**
 * ZLATM5 generates matrices involved in the Generalized Sylvester
 * equation:
 *
 *     A * R - L * B = C
 *     D * R - L * E = F
 *
 * @param[in] prtype
 *     "Points" to a certain type of the matrices to generate.
 *
 * @param[in] m
 *     Specifies the order of A and D and the number of rows in
 *     C, F, R and L.
 *
 * @param[in] n
 *     Specifies the order of B and E and the number of columns in
 *     C, F, R and L.
 *
 * @param[out] A      Complex array, dimension (lda, m).
 * @param[in]  lda    The leading dimension of A.
 * @param[out] B      Complex array, dimension (ldb, n).
 * @param[in]  ldb    The leading dimension of B.
 * @param[out] C      Complex array, dimension (ldc, n).
 * @param[in]  ldc    The leading dimension of C.
 * @param[out] D      Complex array, dimension (ldd, m).
 * @param[in]  ldd    The leading dimension of D.
 * @param[out] E      Complex array, dimension (lde, n).
 * @param[in]  lde    The leading dimension of E.
 * @param[out] F      Complex array, dimension (ldf, n).
 * @param[in]  ldf    The leading dimension of F.
 * @param[out] R      Complex array, dimension (ldr, n).
 * @param[in]  ldr    The leading dimension of R.
 * @param[out] L      Complex array, dimension (ldl, n).
 * @param[in]  ldl    The leading dimension of L.
 * @param[in]  alpha  Parameter used in generating PRTYPE = 1 and 5 matrices.
 * @param[in]  qblcka When PRTYPE = 3, distance between 2-by-2 blocks in A.
 * @param[in]  qblckb When PRTYPE = 3, distance between 2-by-2 blocks in B.
 */
void zlatm5(const INT prtype, const INT m, const INT n,
            c128* A, const INT lda,
            c128* B, const INT ldb,
            c128* C, const INT ldc,
            c128* D, const INT ldd,
            c128* E, const INT lde,
            c128* F, const INT ldf,
            c128* R, const INT ldr,
            c128* L, const INT ldl,
            const f64 alpha, INT qblcka, INT qblckb)
{
    const c128 CONE = CMPLX(1.0, 0.0);
    const c128 CZERO = CMPLX(0.0, 0.0);
    const c128 CTWO = CMPLX(2.0, 0.0);
    const c128 CHALF = CMPLX(0.5, 0.0);
    const c128 CTWENTY = CMPLX(20.0, 0.0);
    const c128 CNEG_ONE = CMPLX(-1.0, 0.0);

    INT i, j, k;
    c128 imeps, reeps;

    if (prtype == 1) {
        for (i = 0; i < m; i++) {
            for (j = 0; j < m; j++) {
                if (i == j) {
                    A[i + j * lda] = CONE;
                    D[i + j * ldd] = CONE;
                } else if (i == j - 1) {
                    A[i + j * lda] = CNEG_ONE;
                    D[i + j * ldd] = CZERO;
                } else {
                    A[i + j * lda] = CZERO;
                    D[i + j * ldd] = CZERO;
                }
            }
        }

        for (i = 0; i < n; i++) {
            for (j = 0; j < n; j++) {
                if (i == j) {
                    B[i + j * ldb] = CONE - alpha;
                    E[i + j * lde] = CONE;
                } else if (i == j - 1) {
                    B[i + j * ldb] = CONE;
                    E[i + j * lde] = CZERO;
                } else {
                    B[i + j * ldb] = CZERO;
                    E[i + j * lde] = CZERO;
                }
            }
        }

        for (i = 0; i < m; i++) {
            for (j = 0; j < n; j++) {
                R[i + j * ldr] = (CHALF - sin((f64)((i + 1) / (j + 1)))) * CTWENTY;
                L[i + j * ldl] = R[i + j * ldr];
            }
        }

    } else if (prtype == 2 || prtype == 3) {
        for (i = 0; i < m; i++) {
            for (j = 0; j < m; j++) {
                if (i <= j) {
                    A[i + j * lda] = (CHALF - sin((f64)(i + 1))) * CTWO;
                    D[i + j * ldd] = (CHALF - sin((f64)((i + 1) * (j + 1)))) * CTWO;
                } else {
                    A[i + j * lda] = CZERO;
                    D[i + j * ldd] = CZERO;
                }
            }
        }

        for (i = 0; i < n; i++) {
            for (j = 0; j < n; j++) {
                if (i <= j) {
                    B[i + j * ldb] = (CHALF - sin((f64)((i + 1) + (j + 1)))) * CTWO;
                    E[i + j * lde] = (CHALF - sin((f64)(j + 1))) * CTWO;
                } else {
                    B[i + j * ldb] = CZERO;
                    E[i + j * lde] = CZERO;
                }
            }
        }

        for (i = 0; i < m; i++) {
            for (j = 0; j < n; j++) {
                R[i + j * ldr] = (CHALF - sin((f64)((i + 1) * (j + 1)))) * CTWENTY;
                L[i + j * ldl] = (CHALF - sin((f64)((i + 1) + (j + 1)))) * CTWENTY;
            }
        }

        if (prtype == 3) {
            if (qblcka <= 1) {
                qblcka = 2;
            }
            for (k = 0; k < m - 1; k += qblcka) {
                A[(k + 1) + (k + 1) * lda] = A[k + k * lda];
                A[(k + 1) + k * lda] = -csin(A[k + (k + 1) * lda]);
            }

            if (qblckb <= 1) {
                qblckb = 2;
            }
            for (k = 0; k < n - 1; k += qblckb) {
                B[(k + 1) + (k + 1) * ldb] = B[k + k * ldb];
                B[(k + 1) + k * ldb] = -csin(B[k + (k + 1) * ldb]);
            }
        }

    } else if (prtype == 4) {
        for (i = 0; i < m; i++) {
            for (j = 0; j < m; j++) {
                A[i + j * lda] = (CHALF - sin((f64)((i + 1) * (j + 1)))) * CTWENTY;
                D[i + j * ldd] = (CHALF - sin((f64)((i + 1) + (j + 1)))) * CTWO;
            }
        }

        for (i = 0; i < n; i++) {
            for (j = 0; j < n; j++) {
                B[i + j * ldb] = (CHALF - sin((f64)((i + 1) + (j + 1)))) * CTWENTY;
                E[i + j * lde] = (CHALF - sin((f64)((i + 1) * (j + 1)))) * CTWO;
            }
        }

        for (i = 0; i < m; i++) {
            for (j = 0; j < n; j++) {
                R[i + j * ldr] = (CHALF - sin((f64)((j + 1) / (i + 1)))) * CTWENTY;
                L[i + j * ldl] = (CHALF - sin((f64)((i + 1) * (j + 1)))) * CTWO;
            }
        }

    } else if (prtype >= 5) {
        reeps = CHALF * CTWO * CTWENTY / alpha;
        imeps = (CHALF - CTWO) / alpha;

        for (i = 0; i < m; i++) {
            for (j = 0; j < n; j++) {
                R[i + j * ldr] = (CHALF - sin((f64)((i + 1) * (j + 1)))) * alpha / CTWENTY;
                L[i + j * ldl] = (CHALF - sin((f64)((i + 1) + (j + 1)))) * alpha / CTWENTY;
            }
        }

        /* Initialize A to zero, then set specific elements */
        for (i = 0; i < m; i++) {
            for (j = 0; j < m; j++) {
                A[i + j * lda] = CZERO;
            }
        }

        for (i = 0; i < m; i++) {
            D[i + i * ldd] = CONE;
        }

        for (i = 0; i < m; i++) {
            INT i_f = i + 1;
            if (i_f <= 4) {
                A[i + i * lda] = CONE;
                if (i_f > 2) {
                    A[i + i * lda] = CONE + reeps;
                }
                if ((i_f % 2) != 0 && i_f < m) {
                    A[i + (i + 1) * lda] = imeps;
                } else if (i_f > 1) {
                    A[i + (i - 1) * lda] = -imeps;
                }
            } else if (i_f <= 8) {
                if (i_f <= 6) {
                    A[i + i * lda] = reeps;
                } else {
                    A[i + i * lda] = -reeps;
                }
                if ((i_f % 2) != 0 && i_f < m) {
                    A[i + (i + 1) * lda] = CONE;
                } else if (i_f > 1) {
                    A[i + (i - 1) * lda] = CNEG_ONE;
                }
            } else {
                A[i + i * lda] = CONE;
                if ((i_f % 2) != 0 && i_f < m) {
                    A[i + (i + 1) * lda] = imeps * 2;
                } else if (i_f > 1) {
                    A[i + (i - 1) * lda] = -imeps * 2;
                }
            }
        }

        /* Initialize B and E */
        for (i = 0; i < n; i++) {
            for (j = 0; j < n; j++) {
                B[i + j * ldb] = CZERO;
                E[i + j * lde] = CZERO;
            }
        }

        for (i = 0; i < n; i++) {
            INT i_f = i + 1;
            E[i + i * lde] = CONE;
            if (i_f <= 4) {
                B[i + i * ldb] = CNEG_ONE;
                if (i_f > 2) {
                    B[i + i * ldb] = CONE - reeps;
                }
                if ((i_f % 2) != 0 && i_f < n) {
                    B[i + (i + 1) * ldb] = imeps;
                } else if (i_f > 1) {
                    B[i + (i - 1) * ldb] = -imeps;
                }
            } else if (i_f <= 8) {
                if (i_f <= 6) {
                    B[i + i * ldb] = reeps;
                } else {
                    B[i + i * ldb] = -reeps;
                }
                if ((i_f % 2) != 0 && i_f < n) {
                    B[i + (i + 1) * ldb] = CONE + imeps;
                } else if (i_f > 1) {
                    B[i + (i - 1) * ldb] = CNEG_ONE - imeps;
                }
            } else {
                B[i + i * ldb] = CONE - reeps;
                if ((i_f % 2) != 0 && i_f < n) {
                    B[i + (i + 1) * ldb] = imeps * 2;
                } else if (i_f > 1) {
                    B[i + (i - 1) * ldb] = -imeps * 2;
                }
            }
        }
    }

    /* Compute rhs (C, F)
     * C = A*R - L*B
     * F = D*R - L*E */
    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                m, n, m, &CONE, A, lda, R, ldr, &CZERO, C, ldc);
    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                m, n, n, &CNEG_ONE, L, ldl, B, ldb, &CONE, C, ldc);
    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                m, n, m, &CONE, D, ldd, R, ldr, &CZERO, F, ldf);
    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                m, n, n, &CNEG_ONE, L, ldl, E, lde, &CONE, F, ldf);
}
