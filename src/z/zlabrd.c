/**
 * @file zlabrd.c
 * @brief ZLABRD reduces the first nb rows and columns of a general matrix to a bidiagonal form.
 */

#include "semicolon_lapack_complex_double.h"
#include <complex.h>
#include <cblas.h>

/**
 * ZLABRD reduces the first NB rows and columns of a complex general
 * m by n matrix A to upper or lower real bidiagonal form by a unitary
 * transformation Q**H * A * P, and returns the matrices X and Y which
 * are needed to apply the transformation to the unreduced part of A.
 *
 * If m >= n, A is reduced to upper bidiagonal form; if m < n, to lower
 * bidiagonal form.
 *
 * This is an auxiliary routine called by ZGEBRD.
 *
 * @param[in]     m     The number of rows in the matrix A.
 * @param[in]     n     The number of columns in the matrix A.
 * @param[in]     nb    The number of leading rows and columns of A to be reduced.
 * @param[in,out] A     Complex array, dimension (lda, n).
 *                      On entry, the m by n general matrix to be reduced.
 *                      On exit, the first NB rows and columns of the matrix are
 *                      overwritten; the rest of the array is unchanged.
 * @param[in]     lda   The leading dimension of the array A. lda >= max(1,m).
 * @param[out]    D     Double precision array, dimension (nb).
 *                      The diagonal elements of the first NB rows and columns of
 *                      the reduced matrix. D[i] = A[i,i].
 * @param[out]    E     Double precision array, dimension (nb).
 *                      The off-diagonal elements of the first NB rows and columns of
 *                      the reduced matrix.
 * @param[out]    tauq  Complex array, dimension (nb).
 *                      The scalar factors of the elementary reflectors which
 *                      represent the unitary matrix Q.
 * @param[out]    taup  Complex array, dimension (nb).
 *                      The scalar factors of the elementary reflectors which
 *                      represent the unitary matrix P.
 * @param[out]    X     Complex array, dimension (ldx, nb).
 *                      The m-by-nb matrix X required to update the unreduced part
 *                      of A.
 * @param[in]     ldx   The leading dimension of the array X. ldx >= max(1,m).
 * @param[out]    Y     Complex array, dimension (ldy, nb).
 *                      The n-by-nb matrix Y required to update the unreduced part
 *                      of A.
 * @param[in]     ldy   The leading dimension of the array Y. ldy >= max(1,n).
 */
void zlabrd(const int m, const int n, const int nb,
            double complex* const restrict A, const int lda,
            double* const restrict D, double* const restrict E,
            double complex* const restrict tauq, double complex* const restrict taup,
            double complex* const restrict X, const int ldx,
            double complex* const restrict Y, const int ldy)
{
    const double complex ZERO = CMPLX(0.0, 0.0);
    const double complex ONE  = CMPLX(1.0, 0.0);
    const double complex NEG_ONE = CMPLX(-1.0, 0.0);

    int i;
    double complex alpha;

    /* Quick return if possible */
    if (m <= 0 || n <= 0) {
        return;
    }

    if (m >= n) {
        /* Reduce to upper bidiagonal form */
        for (i = 0; i < nb; i++) {
            /* Update A[i:m-1, i] */
            zlacgv(i, &Y[i], ldy);
            cblas_zgemv(CblasColMajor, CblasNoTrans, m - i, i,
                        &NEG_ONE, &A[i], lda, &Y[i], ldy,
                        &ONE, &A[i + i * lda], 1);
            zlacgv(i, &Y[i], ldy);
            cblas_zgemv(CblasColMajor, CblasNoTrans, m - i, i,
                        &NEG_ONE, &X[i], ldx, &A[i * lda], 1,
                        &ONE, &A[i + i * lda], 1);

            /* Generate reflection Q(i) to annihilate A[i+1:m-1, i] */
            alpha = A[i + i * lda];
            zlarfg(m - i, &alpha, &A[((i + 1) < m ? (i + 1) : (m - 1)) + i * lda], 1,
                   &tauq[i]);
            D[i] = creal(alpha);

            if (i < n - 1) {
                A[i + i * lda] = ONE;

                /* Compute Y[i+1:n-1, i] */
                cblas_zgemv(CblasColMajor, CblasConjTrans, m - i, n - i - 1,
                            &ONE, &A[i + (i + 1) * lda], lda,
                            &A[i + i * lda], 1, &ZERO, &Y[i + 1 + i * ldy], 1);
                cblas_zgemv(CblasColMajor, CblasConjTrans, m - i, i,
                            &ONE, &A[i], lda, &A[i + i * lda], 1,
                            &ZERO, &Y[i * ldy], 1);
                cblas_zgemv(CblasColMajor, CblasNoTrans, n - i - 1, i,
                            &NEG_ONE, &Y[i + 1], ldy, &Y[i * ldy], 1,
                            &ONE, &Y[i + 1 + i * ldy], 1);
                cblas_zgemv(CblasColMajor, CblasConjTrans, m - i, i,
                            &ONE, &X[i], ldx, &A[i + i * lda], 1,
                            &ZERO, &Y[i * ldy], 1);
                cblas_zgemv(CblasColMajor, CblasConjTrans, i, n - i - 1,
                            &NEG_ONE, &A[(i + 1) * lda], lda, &Y[i * ldy], 1,
                            &ONE, &Y[i + 1 + i * ldy], 1);
                cblas_zscal(n - i - 1, &tauq[i], &Y[i + 1 + i * ldy], 1);

                /* Update A[i, i+1:n-1] */
                zlacgv(n - i - 1, &A[i + (i + 1) * lda], lda);
                zlacgv(i + 1, &A[i], lda);
                cblas_zgemv(CblasColMajor, CblasNoTrans, n - i - 1, i + 1,
                            &NEG_ONE, &Y[i + 1], ldy, &A[i], lda,
                            &ONE, &A[i + (i + 1) * lda], lda);
                zlacgv(i + 1, &A[i], lda);
                zlacgv(i, &X[i], ldx);
                cblas_zgemv(CblasColMajor, CblasConjTrans, i, n - i - 1,
                            &NEG_ONE, &A[(i + 1) * lda], lda, &X[i], ldx,
                            &ONE, &A[i + (i + 1) * lda], lda);
                zlacgv(i, &X[i], ldx);

                /* Generate reflection P(i) to annihilate A[i, i+2:n-1] */
                alpha = A[i + (i + 1) * lda];
                zlarfg(n - i - 1, &alpha,
                       &A[i + ((i + 2) < n ? (i + 2) : (n - 1)) * lda], lda,
                       &taup[i]);
                E[i] = creal(alpha);
                A[i + (i + 1) * lda] = ONE;

                /* Compute X[i+1:m-1, i] */
                cblas_zgemv(CblasColMajor, CblasNoTrans, m - i - 1, n - i - 1,
                            &ONE, &A[i + 1 + (i + 1) * lda], lda,
                            &A[i + (i + 1) * lda], lda, &ZERO, &X[i + 1 + i * ldx], 1);
                cblas_zgemv(CblasColMajor, CblasConjTrans, n - i - 1, i + 1,
                            &ONE, &Y[i + 1], ldy, &A[i + (i + 1) * lda], lda,
                            &ZERO, &X[i * ldx], 1);
                cblas_zgemv(CblasColMajor, CblasNoTrans, m - i - 1, i + 1,
                            &NEG_ONE, &A[i + 1], lda, &X[i * ldx], 1,
                            &ONE, &X[i + 1 + i * ldx], 1);
                cblas_zgemv(CblasColMajor, CblasNoTrans, i, n - i - 1,
                            &ONE, &A[(i + 1) * lda], lda,
                            &A[i + (i + 1) * lda], lda, &ZERO, &X[i * ldx], 1);
                cblas_zgemv(CblasColMajor, CblasNoTrans, m - i - 1, i,
                            &NEG_ONE, &X[i + 1], ldx, &X[i * ldx], 1,
                            &ONE, &X[i + 1 + i * ldx], 1);
                cblas_zscal(m - i - 1, &taup[i], &X[i + 1 + i * ldx], 1);
                zlacgv(n - i - 1, &A[i + (i + 1) * lda], lda);
            }
        }
    } else {
        /* Reduce to lower bidiagonal form */
        for (i = 0; i < nb; i++) {
            /* Update A[i, i:n-1] */
            zlacgv(n - i, &A[i + i * lda], lda);
            zlacgv(i, &A[i], lda);
            cblas_zgemv(CblasColMajor, CblasNoTrans, n - i, i,
                        &NEG_ONE, &Y[i], ldy, &A[i], lda,
                        &ONE, &A[i + i * lda], lda);
            zlacgv(i, &A[i], lda);
            zlacgv(i, &X[i], ldx);
            cblas_zgemv(CblasColMajor, CblasConjTrans, i, n - i,
                        &NEG_ONE, &A[i * lda], lda, &X[i], ldx,
                        &ONE, &A[i + i * lda], lda);
            zlacgv(i, &X[i], ldx);

            /* Generate reflection P(i) to annihilate A[i, i+1:n-1] */
            alpha = A[i + i * lda];
            zlarfg(n - i, &alpha, &A[i + ((i + 1) < n ? (i + 1) : (n - 1)) * lda], lda,
                   &taup[i]);
            D[i] = creal(alpha);

            if (i < m - 1) {
                A[i + i * lda] = ONE;

                /* Compute X[i+1:m-1, i] */
                cblas_zgemv(CblasColMajor, CblasNoTrans, m - i - 1, n - i,
                            &ONE, &A[i + 1 + i * lda], lda,
                            &A[i + i * lda], lda, &ZERO, &X[i + 1 + i * ldx], 1);
                cblas_zgemv(CblasColMajor, CblasConjTrans, n - i, i,
                            &ONE, &Y[i], ldy, &A[i + i * lda], lda,
                            &ZERO, &X[i * ldx], 1);
                cblas_zgemv(CblasColMajor, CblasNoTrans, m - i - 1, i,
                            &NEG_ONE, &A[i + 1], lda, &X[i * ldx], 1,
                            &ONE, &X[i + 1 + i * ldx], 1);
                cblas_zgemv(CblasColMajor, CblasNoTrans, i, n - i,
                            &ONE, &A[i * lda], lda, &A[i + i * lda], lda,
                            &ZERO, &X[i * ldx], 1);
                cblas_zgemv(CblasColMajor, CblasNoTrans, m - i - 1, i,
                            &NEG_ONE, &X[i + 1], ldx, &X[i * ldx], 1,
                            &ONE, &X[i + 1 + i * ldx], 1);
                cblas_zscal(m - i - 1, &taup[i], &X[i + 1 + i * ldx], 1);
                zlacgv(n - i, &A[i + i * lda], lda);

                /* Update A[i+1:m-1, i] */
                zlacgv(i, &Y[i], ldy);
                cblas_zgemv(CblasColMajor, CblasNoTrans, m - i - 1, i,
                            &NEG_ONE, &A[i + 1], lda, &Y[i], ldy,
                            &ONE, &A[i + 1 + i * lda], 1);
                zlacgv(i, &Y[i], ldy);
                cblas_zgemv(CblasColMajor, CblasNoTrans, m - i - 1, i + 1,
                            &NEG_ONE, &X[i + 1], ldx, &A[i * lda], 1,
                            &ONE, &A[i + 1 + i * lda], 1);

                /* Generate reflection Q(i) to annihilate A[i+2:m-1, i] */
                alpha = A[i + 1 + i * lda];
                zlarfg(m - i - 1, &alpha,
                       &A[((i + 2) < m ? (i + 2) : (m - 1)) + i * lda], 1,
                       &tauq[i]);
                E[i] = creal(alpha);
                A[i + 1 + i * lda] = ONE;

                /* Compute Y[i+1:n-1, i] */
                cblas_zgemv(CblasColMajor, CblasConjTrans, m - i - 1, n - i - 1,
                            &ONE, &A[i + 1 + (i + 1) * lda], lda,
                            &A[i + 1 + i * lda], 1, &ZERO, &Y[i + 1 + i * ldy], 1);
                cblas_zgemv(CblasColMajor, CblasConjTrans, m - i - 1, i,
                            &ONE, &A[i + 1], lda, &A[i + 1 + i * lda], 1,
                            &ZERO, &Y[i * ldy], 1);
                cblas_zgemv(CblasColMajor, CblasNoTrans, n - i - 1, i,
                            &NEG_ONE, &Y[i + 1], ldy, &Y[i * ldy], 1,
                            &ONE, &Y[i + 1 + i * ldy], 1);
                cblas_zgemv(CblasColMajor, CblasConjTrans, m - i - 1, i + 1,
                            &ONE, &X[i + 1], ldx, &A[i + 1 + i * lda], 1,
                            &ZERO, &Y[i * ldy], 1);
                cblas_zgemv(CblasColMajor, CblasConjTrans, i + 1, n - i - 1,
                            &NEG_ONE, &A[(i + 1) * lda], lda, &Y[i * ldy], 1,
                            &ONE, &Y[i + 1 + i * ldy], 1);
                cblas_zscal(n - i - 1, &tauq[i], &Y[i + 1 + i * ldy], 1);
            } else {
                zlacgv(n - i, &A[i + i * lda], lda);
            }
        }
    }
}
