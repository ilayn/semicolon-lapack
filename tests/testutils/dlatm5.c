/**
 * @file dlatm5.c
 * @brief DLATM5 generates matrices involved in the Generalized Sylvester equation.
 */

#include <math.h>
#include <cblas.h>
#include "verify.h"

/**
 * DLATM5 generates matrices involved in the Generalized Sylvester
 * equation:
 *
 *     A * R - L * B = C
 *     D * R - L * E = F
 *
 * They also satisfy (the diagonalization condition)
 *
 *  [ I -L ] ( [ A  -C ], [ D -F ] ) [ I  R ] = ( [ A    ], [ D    ] )
 *  [    I ] ( [     B ]  [    E ] ) [    I ]   ( [    B ]  [    E ] )
 *
 * @param[in] prtype
 *     "Points" to a certain type of the matrices to generate.
 *     See further details below.
 *
 * @param[in] m
 *     Specifies the order of A and D and the number of rows in
 *     C, F, R and L.
 *
 * @param[in] n
 *     Specifies the order of B and E and the number of columns in
 *     C, F, R and L.
 *
 * @param[out] A
 *     Double precision array, dimension (lda, m).
 *     On exit A M-by-M is initialized according to PRTYPE.
 *
 * @param[in] lda
 *     The leading dimension of A.
 *
 * @param[out] B
 *     Double precision array, dimension (ldb, n).
 *     On exit B N-by-N is initialized according to PRTYPE.
 *
 * @param[in] ldb
 *     The leading dimension of B.
 *
 * @param[out] C
 *     Double precision array, dimension (ldc, n).
 *     On exit C M-by-N is initialized according to PRTYPE.
 *
 * @param[in] ldc
 *     The leading dimension of C.
 *
 * @param[out] D
 *     Double precision array, dimension (ldd, m).
 *     On exit D M-by-M is initialized according to PRTYPE.
 *
 * @param[in] ldd
 *     The leading dimension of D.
 *
 * @param[out] E
 *     Double precision array, dimension (lde, n).
 *     On exit E N-by-N is initialized according to PRTYPE.
 *
 * @param[in] lde
 *     The leading dimension of E.
 *
 * @param[out] F
 *     Double precision array, dimension (ldf, n).
 *     On exit F M-by-N is initialized according to PRTYPE.
 *
 * @param[in] ldf
 *     The leading dimension of F.
 *
 * @param[out] R
 *     Double precision array, dimension (ldr, n).
 *     On exit R M-by-N is initialized according to PRTYPE.
 *
 * @param[in] ldr
 *     The leading dimension of R.
 *
 * @param[out] L
 *     Double precision array, dimension (ldl, n).
 *     On exit L M-by-N is initialized according to PRTYPE.
 *
 * @param[in] ldl
 *     The leading dimension of L.
 *
 * @param[in] alpha
 *     Parameter used in generating PRTYPE = 1 and 5 matrices.
 *
 * @param[in] qblcka
 *     When PRTYPE = 3, specifies the distance between 2-by-2
 *     blocks on the diagonal in A. Otherwise, QBLCKA is not
 *     referenced. QBLCKA > 1.
 *
 * @param[in] qblckb
 *     When PRTYPE = 3, specifies the distance between 2-by-2
 *     blocks on the diagonal in B. Otherwise, QBLCKB is not
 *     referenced. QBLCKB > 1.
 *
 * Further Details:
 *
 *  PRTYPE = 1: A and B are Jordan blocks, D and E are identity matrices
 *
 *             A : if (i == j) then A(i, j) = 1.0
 *                 if (j == i + 1) then A(i, j) = -1.0
 *                 else A(i, j) = 0.0,            i, j = 0...M-1
 *
 *             B : if (i == j) then B(i, j) = 1.0 - ALPHA
 *                 if (j == i + 1) then B(i, j) = 1.0
 *                 else B(i, j) = 0.0,            i, j = 0...N-1
 *
 *             D : if (i == j) then D(i, j) = 1.0
 *                 else D(i, j) = 0.0,            i, j = 0...M-1
 *
 *             E : if (i == j) then E(i, j) = 1.0
 *                 else E(i, j) = 0.0,            i, j = 0...N-1
 *
 *             L =  R are chosen from [-10...10],
 *                  which specifies the right hand sides (C, F).
 *
 *  PRTYPE = 2 or 3: Triangular and/or quasi- triangular.
 *
 *             A : if (i <= j) then A(i, j) = [-1...1]
 *                 else A(i, j) = 0.0,             i, j = 0...M-1
 *
 *                 if (PRTYPE = 3) then
 *                    A(k + 1, k + 1) = A(k, k)
 *                    A(k + 1, k) = [-1...1]
 *                    sign(A(k, k + 1) = -(sin(A(k + 1, k))
 *                        k = 0, M - 2, QBLCKA
 *
 *             B : if (i <= j) then B(i, j) = [-1...1]
 *                 else B(i, j) = 0.0,            i, j = 0...N-1
 *
 *                 if (PRTYPE = 3) then
 *                    B(k + 1, k + 1) = B(k, k)
 *                    B(k + 1, k) = [-1...1]
 *                    sign(B(k, k + 1) = -(sign(B(k + 1, k))
 *                        k = 0, N - 2, QBLCKB
 *
 *             D : if (i <= j) then D(i, j) = [-1...1].
 *                 else D(i, j) = 0.0,            i, j = 0...M-1
 *
 *
 *             E : if (i <= j) then D(i, j) = [-1...1]
 *                 else E(i, j) = 0.0,            i, j = 0...N-1
 *
 *                 L, R are chosen from [-10...10],
 *                 which specifies the right hand sides (C, F).
 *
 *  PRTYPE = 4 Full
 *             A(i, j) = [-10...10]
 *             D(i, j) = [-1...1]    i,j = 0...M-1
 *             B(i, j) = [-10...10]
 *             E(i, j) = [-1...1]    i,j = 0...N-1
 *             R(i, j) = [-10...10]
 *             L(i, j) = [-1...1]    i = 0..M-1 ,j = 0...N-1
 *
 *             L, R specifies the right hand sides (C, F).
 *
 *  PRTYPE = 5 special case common and/or close eigs.
 */
void dlatm5(const int prtype, const int m, const int n,
            f64* A, const int lda,
            f64* B, const int ldb,
            f64* C, const int ldc,
            f64* D, const int ldd,
            f64* E, const int lde,
            f64* F, const int ldf,
            f64* R, const int ldr,
            f64* L, const int ldl,
            const f64 alpha, int qblcka, int qblckb)
{
    const f64 ONE = 1.0;
    const f64 ZERO = 0.0;
    const f64 TWENTY = 20.0;
    const f64 HALF = 0.5;
    const f64 TWO = 2.0;

    int i, j, k;
    f64 imeps, reeps;

    if (prtype == 1) {
        for (i = 0; i < m; i++) {
            for (j = 0; j < m; j++) {
                if (i == j) {
                    A[i + j * lda] = ONE;
                    D[i + j * ldd] = ONE;
                } else if (i == j - 1) {
                    A[i + j * lda] = -ONE;
                    D[i + j * ldd] = ZERO;
                } else {
                    A[i + j * lda] = ZERO;
                    D[i + j * ldd] = ZERO;
                }
            }
        }

        for (i = 0; i < n; i++) {
            for (j = 0; j < n; j++) {
                if (i == j) {
                    B[i + j * ldb] = ONE - alpha;
                    E[i + j * lde] = ONE;
                } else if (i == j - 1) {
                    B[i + j * ldb] = ONE;
                    E[i + j * lde] = ZERO;
                } else {
                    B[i + j * ldb] = ZERO;
                    E[i + j * lde] = ZERO;
                }
            }
        }

        for (i = 0; i < m; i++) {
            for (j = 0; j < n; j++) {
                /* Fortran: (HALF - SIN(DBLE(I/J))) * TWENTY
                 * with 1-based I,J. In C with 0-based:
                 * use (i+1)/(j+1) to match Fortran integer division */
                R[i + j * ldr] = (HALF - sin((f64)((i + 1) / (j + 1)))) * TWENTY;
                L[i + j * ldl] = R[i + j * ldr];
            }
        }

    } else if (prtype == 2 || prtype == 3) {
        for (i = 0; i < m; i++) {
            for (j = 0; j < m; j++) {
                if (i <= j) {
                    A[i + j * lda] = (HALF - sin((f64)(i + 1))) * TWO;
                    D[i + j * ldd] = (HALF - sin((f64)((i + 1) * (j + 1)))) * TWO;
                } else {
                    A[i + j * lda] = ZERO;
                    D[i + j * ldd] = ZERO;
                }
            }
        }

        for (i = 0; i < n; i++) {
            for (j = 0; j < n; j++) {
                if (i <= j) {
                    B[i + j * ldb] = (HALF - sin((f64)((i + 1) + (j + 1)))) * TWO;
                    E[i + j * lde] = (HALF - sin((f64)(j + 1))) * TWO;
                } else {
                    B[i + j * ldb] = ZERO;
                    E[i + j * lde] = ZERO;
                }
            }
        }

        for (i = 0; i < m; i++) {
            for (j = 0; j < n; j++) {
                R[i + j * ldr] = (HALF - sin((f64)((i + 1) * (j + 1)))) * TWENTY;
                L[i + j * ldl] = (HALF - sin((f64)((i + 1) + (j + 1)))) * TWENTY;
            }
        }

        if (prtype == 3) {
            if (qblcka <= 1) {
                qblcka = 2;
            }
            /* Fortran: DO K = 1, M-1, QBLCKA
             * In C 0-based: k goes 0, qblcka, 2*qblcka, ... while k < m-1 */
            for (k = 0; k < m - 1; k += qblcka) {
                A[(k + 1) + (k + 1) * lda] = A[k + k * lda];
                A[(k + 1) + k * lda] = -sin(A[k + (k + 1) * lda]);
            }

            if (qblckb <= 1) {
                qblckb = 2;
            }
            for (k = 0; k < n - 1; k += qblckb) {
                B[(k + 1) + (k + 1) * ldb] = B[k + k * ldb];
                B[(k + 1) + k * ldb] = -sin(B[k + (k + 1) * ldb]);
            }
        }

    } else if (prtype == 4) {
        for (i = 0; i < m; i++) {
            for (j = 0; j < m; j++) {
                A[i + j * lda] = (HALF - sin((f64)((i + 1) * (j + 1)))) * TWENTY;
                D[i + j * ldd] = (HALF - sin((f64)((i + 1) + (j + 1)))) * TWO;
            }
        }

        for (i = 0; i < n; i++) {
            for (j = 0; j < n; j++) {
                B[i + j * ldb] = (HALF - sin((f64)((i + 1) + (j + 1)))) * TWENTY;
                E[i + j * lde] = (HALF - sin((f64)((i + 1) * (j + 1)))) * TWO;
            }
        }

        for (i = 0; i < m; i++) {
            for (j = 0; j < n; j++) {
                /* Fortran: (HALF - SIN(DBLE(J/I))) * TWENTY
                 * with 1-based indices */
                R[i + j * ldr] = (HALF - sin((f64)((j + 1) / (i + 1)))) * TWENTY;
                L[i + j * ldl] = (HALF - sin((f64)((i + 1) * (j + 1)))) * TWO;
            }
        }

    } else if (prtype >= 5) {
        reeps = HALF * TWO * TWENTY / alpha;
        imeps = (HALF - TWO) / alpha;

        for (i = 0; i < m; i++) {
            for (j = 0; j < n; j++) {
                R[i + j * ldr] = (HALF - sin((f64)((i + 1) * (j + 1)))) * alpha / TWENTY;
                L[i + j * ldl] = (HALF - sin((f64)((i + 1) + (j + 1)))) * alpha / TWENTY;
            }
        }

        for (i = 0; i < m; i++) {
            D[i + i * ldd] = ONE;
        }

        for (i = 0; i < m; i++) {
            /* Clear non-diagonal elements first (not done in Fortran but needed
             * since we don't initialize the full matrix) */
            for (j = 0; j < m; j++) {
                if (i != j && !(i == j - 1) && !(i == j + 1)) {
                    A[i + j * lda] = ZERO;
                }
            }
            if (i == j - 1 || i == j + 1) {
                /* These will be set below */
            }
        }

        /* Initialize A to zero first */
        for (i = 0; i < m; i++) {
            for (j = 0; j < m; j++) {
                A[i + j * lda] = ZERO;
            }
        }

        /* Now set the specific elements */
        for (i = 0; i < m; i++) {
            /* Using 1-based logic from Fortran: i_f = i + 1 */
            int i_f = i + 1;
            if (i_f <= 4) {
                A[i + i * lda] = ONE;
                if (i_f > 2) {
                    A[i + i * lda] = ONE + reeps;
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
                    A[i + (i + 1) * lda] = ONE;
                } else if (i_f > 1) {
                    A[i + (i - 1) * lda] = -ONE;
                }
            } else {
                A[i + i * lda] = ONE;
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
                B[i + j * ldb] = ZERO;
                E[i + j * lde] = ZERO;
            }
        }

        for (i = 0; i < n; i++) {
            int i_f = i + 1;
            E[i + i * lde] = ONE;
            if (i_f <= 4) {
                B[i + i * ldb] = -ONE;
                if (i_f > 2) {
                    B[i + i * ldb] = ONE - reeps;
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
                    B[i + (i + 1) * ldb] = ONE + imeps;
                } else if (i_f > 1) {
                    B[i + (i - 1) * ldb] = -ONE - imeps;
                }
            } else {
                B[i + i * ldb] = ONE - reeps;
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
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                m, n, m, ONE, A, lda, R, ldr, ZERO, C, ldc);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                m, n, n, -ONE, L, ldl, B, ldb, ONE, C, ldc);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                m, n, m, ONE, D, ldd, R, ldr, ZERO, F, ldf);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                m, n, n, -ONE, L, ldl, E, lde, ONE, F, ldf);
}
