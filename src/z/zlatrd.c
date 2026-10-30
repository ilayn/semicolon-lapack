/**
 * @file zlatrd.c
 * @brief ZLATRD reduces NB rows and columns of a complex Hermitian matrix to
 *        real tridiagonal form.
 */

#include "semicolon_lapack_complex_double.h"
#include <complex.h>
#include "semicolon_cblas.h"

/**
 * ZLATRD reduces NB rows and columns of a complex Hermitian matrix A to
 * Hermitian tridiagonal form by a unitary similarity
 * transformation Q**H * A * Q, and returns the matrices V and W which are
 * needed to apply the transformation to the unreduced part of A.
 *
 * If UPLO = 'U', ZLATRD reduces the last NB rows and columns of a
 * matrix, of which the upper triangle is supplied;
 * if UPLO = 'L', ZLATRD reduces the first NB rows and columns of a
 * matrix, of which the lower triangle is supplied.
 *
 * This is an auxiliary routine called by ZHETRD.
 *
 * @param[in]     uplo  Specifies whether the upper or lower triangular part
 *                      of the Hermitian matrix A is stored:
 *                      = 'U': Upper triangular
 *                      = 'L': Lower triangular
 * @param[in]     n     The order of the matrix A. n >= 0.
 * @param[in]     nb    The number of rows and columns to be reduced.
 * @param[in,out] A     Complex array, dimension (lda, n).
 *                      On entry, the Hermitian matrix A.
 *                      On exit, if UPLO = 'U', the last NB columns have been
 *                      reduced to tridiagonal form; if UPLO = 'L', the first
 *                      NB columns have been reduced to tridiagonal form.
 * @param[in]     lda   The leading dimension of the array A. lda >= max(1,n).
 * @param[out]    E     Double precision array, dimension (n-1).
 *                      The off-diagonal elements of the tridiagonal matrix.
 * @param[out]    tau   Complex array, dimension (n-1).
 *                      The scalar factors of the elementary reflectors.
 * @param[out]    W     Complex array, dimension (ldw, nb).
 *                      The n-by-nb matrix W.
 * @param[in]     ldw   The leading dimension of the array W. ldw >= max(1,n).
 */
void zlatrd(const char* uplo, const INT n, const INT nb,
            c128* restrict A, const INT lda,
            f64* restrict E, c128* restrict tau,
            c128* restrict W, const INT ldw)
{
    const c128 ZERO = CMPLX(0.0, 0.0);
    const c128 ONE  = CMPLX(1.0, 0.0);
    const c128 HALF = CMPLX(0.5, 0.0);
    const c128 NEG_ONE = CMPLX(-1.0, 0.0);

    INT i, iw;
    c128 alpha;

    /* Quick return if possible */
    if (n <= 0) {
        return;
    }

    if (uplo[0] == 'U' || uplo[0] == 'u') {
        /* Reduce last NB columns of upper triangle. */
        for (i = n - 1; i >= n - nb; i--) {
            iw = i - n + nb;

            if (i < n - 1) {
                /* Update A(0:i, i). */
                A[i + i * lda] = CMPLX(creal(A[i + i * lda]), 0.0);
                zlacgv(n - i - 1, &W[i + (iw + 1) * ldw], ldw);
                cblas_zgemv(CblasColMajor, CblasNoTrans,
                            i + 1, n - i - 1, &NEG_ONE,
                            &A[(i + 1) * lda], lda,
                            &W[i + (iw + 1) * ldw], ldw,
                            &ONE, &A[i * lda], 1);
                zlacgv(n - i - 1, &W[i + (iw + 1) * ldw], ldw);
                zlacgv(n - i - 1, &A[i + (i + 1) * lda], lda);
                cblas_zgemv(CblasColMajor, CblasNoTrans,
                            i + 1, n - i - 1, &NEG_ONE,
                            &W[(iw + 1) * ldw], ldw,
                            &A[i + (i + 1) * lda], lda,
                            &ONE, &A[i * lda], 1);
                zlacgv(n - i - 1, &A[i + (i + 1) * lda], lda);
                A[i + i * lda] = CMPLX(creal(A[i + i * lda]), 0.0);
            }

            if (i > 0) {
                /* Generate elementary reflector H(i) to annihilate A(0:i-2, i). */
                alpha = A[(i - 1) + i * lda];
                zlarfg(i,
                       &alpha,
                       &A[i * lda],
                       1,
                       &tau[i - 1]);

                E[i - 1] = creal(alpha);
                A[(i - 1) + i * lda] = ONE;

                /* Compute W(0:i-1, iw). */
                cblas_zhemv(CblasColMajor, CblasUpper,
                            i, &ONE,
                            A, lda,
                            &A[i * lda], 1,
                            &ZERO, &W[iw * ldw], 1);

                if (i < n - 1) {
                    cblas_zgemv(CblasColMajor, CblasConjTrans,
                                i, n - i - 1, &ONE,
                                &W[(iw + 1) * ldw], ldw,
                                &A[i * lda], 1,
                                &ZERO, &W[(i + 1) + iw * ldw], 1);
                    cblas_zgemv(CblasColMajor, CblasNoTrans,
                                i, n - i - 1, &NEG_ONE,
                                &A[(i + 1) * lda], lda,
                                &W[(i + 1) + iw * ldw], 1,
                                &ONE, &W[iw * ldw], 1);
                    cblas_zgemv(CblasColMajor, CblasConjTrans,
                                i, n - i - 1, &ONE,
                                &A[(i + 1) * lda], lda,
                                &A[i * lda], 1,
                                &ZERO, &W[(i + 1) + iw * ldw], 1);
                    cblas_zgemv(CblasColMajor, CblasNoTrans,
                                i, n - i - 1, &NEG_ONE,
                                &W[(iw + 1) * ldw], ldw,
                                &W[(i + 1) + iw * ldw], 1,
                                &ONE, &W[iw * ldw], 1);
                }

                cblas_zscal(i, &tau[i - 1], &W[iw * ldw], 1);
                c128 dotc;
                cblas_zdotc_sub(i,
                                &W[iw * ldw], 1,
                                &A[i * lda], 1,
                                &dotc);
                alpha = -HALF * tau[i - 1] * dotc;
                cblas_zaxpy(i, &alpha,
                            &A[i * lda], 1,
                            &W[iw * ldw], 1);
            }
        }

    } else {
        /* Reduce first NB columns of lower triangle. */
        for (i = 0; i < nb; i++) {
            /* Update A(i:n-1, i). */
            A[i + i * lda] = CMPLX(creal(A[i + i * lda]), 0.0);
            zlacgv(i, &W[i], ldw);
            cblas_zgemv(CblasColMajor, CblasNoTrans,
                        n - i, i, &NEG_ONE,
                        &A[i], lda,
                        &W[i], ldw,
                        &ONE, &A[i + i * lda], 1);
            zlacgv(i, &W[i], ldw);
            zlacgv(i, &A[i], lda);
            cblas_zgemv(CblasColMajor, CblasNoTrans,
                        n - i, i, &NEG_ONE,
                        &W[i], ldw,
                        &A[i], lda,
                        &ONE, &A[i + i * lda], 1);
            zlacgv(i, &A[i], lda);
            A[i + i * lda] = CMPLX(creal(A[i + i * lda]), 0.0);

            if (i < n - 1) {
                /* Generate elementary reflector H(i) to annihilate A(i+2:n-1, i). */
                alpha = A[(i + 1) + i * lda];
                INT min_row = (i + 2 < n - 1) ? (i + 2) : (n - 1);
                zlarfg(n - i - 1,
                       &alpha,
                       &A[min_row + i * lda],
                       1,
                       &tau[i]);

                E[i] = creal(alpha);
                A[(i + 1) + i * lda] = ONE;

                /* Compute W(i+1:n-1, i). */
                cblas_zhemv(CblasColMajor, CblasLower,
                            n - i - 1, &ONE,
                            &A[(i + 1) + (i + 1) * lda], lda,
                            &A[(i + 1) + i * lda], 1,
                            &ZERO, &W[(i + 1) + i * ldw], 1);
                cblas_zgemv(CblasColMajor, CblasConjTrans,
                            n - i - 1, i, &ONE,
                            &W[i + 1], ldw,
                            &A[(i + 1) + i * lda], 1,
                            &ZERO, &W[i * ldw], 1);
                cblas_zgemv(CblasColMajor, CblasNoTrans,
                            n - i - 1, i, &NEG_ONE,
                            &A[i + 1], lda,
                            &W[i * ldw], 1,
                            &ONE, &W[(i + 1) + i * ldw], 1);
                cblas_zgemv(CblasColMajor, CblasConjTrans,
                            n - i - 1, i, &ONE,
                            &A[i + 1], lda,
                            &A[(i + 1) + i * lda], 1,
                            &ZERO, &W[i * ldw], 1);
                cblas_zgemv(CblasColMajor, CblasNoTrans,
                            n - i - 1, i, &NEG_ONE,
                            &W[i + 1], ldw,
                            &W[i * ldw], 1,
                            &ONE, &W[(i + 1) + i * ldw], 1);

                cblas_zscal(n - i - 1, &tau[i], &W[(i + 1) + i * ldw], 1);
                c128 dotc;
                cblas_zdotc_sub(n - i - 1,
                                &W[(i + 1) + i * ldw], 1,
                                &A[(i + 1) + i * lda], 1,
                                &dotc);
                alpha = -HALF * tau[i] * dotc;
                cblas_zaxpy(n - i - 1, &alpha,
                            &A[(i + 1) + i * lda], 1,
                            &W[(i + 1) + i * ldw], 1);
            }
        }
    }
}
