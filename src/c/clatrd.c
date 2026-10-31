/**
 * @file clatrd.c
 * @brief CLATRD reduces NB rows and columns of a complex Hermitian matrix to
 *        real tridiagonal form.
 */

#include "semicolon_lapack_complex_single.h"
#include <complex.h>
#include "semicolon_cblas.h"

/**
 * CLATRD reduces NB rows and columns of a complex Hermitian matrix A to
 * Hermitian tridiagonal form by a unitary similarity
 * transformation Q**H * A * Q, and returns the matrices V and W which are
 * needed to apply the transformation to the unreduced part of A.
 *
 * If UPLO = 'U', CLATRD reduces the last NB rows and columns of a
 * matrix, of which the upper triangle is supplied;
 * if UPLO = 'L', CLATRD reduces the first NB rows and columns of a
 * matrix, of which the lower triangle is supplied.
 *
 * This is an auxiliary routine called by CHETRD.
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
 * @param[out]    E     Single precision array, dimension (n-1).
 *                      The off-diagonal elements of the tridiagonal matrix.
 * @param[out]    tau   Complex array, dimension (n-1).
 *                      The scalar factors of the elementary reflectors.
 * @param[out]    W     Complex array, dimension (ldw, nb).
 *                      The n-by-nb matrix W.
 * @param[in]     ldw   The leading dimension of the array W. ldw >= max(1,n).
 */
void clatrd(const char* uplo, const INT n, const INT nb,
            c64* restrict A, const INT lda,
            f32* restrict E, c64* restrict tau,
            c64* restrict W, const INT ldw)
{
    const c64 ZERO = CMPLXF(0.0f, 0.0f);
    const c64 ONE  = CMPLXF(1.0f, 0.0f);
    const c64 HALF = CMPLXF(0.5f, 0.0f);
    const c64 NEG_ONE = CMPLXF(-1.0f, 0.0f);

    INT i, iw;
    c64 alpha;

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
                A[i + i * lda] = CMPLXF(crealf(A[i + i * lda]), 0.0f);
                clacgv(n - i - 1, &W[i + (iw + 1) * ldw], ldw);
                cblas_cgemv(CblasColMajor, CblasNoTrans,
                            i + 1, n - i - 1, &NEG_ONE,
                            &A[(i + 1) * lda], lda,
                            &W[i + (iw + 1) * ldw], ldw,
                            &ONE, &A[i * lda], 1);
                clacgv(n - i - 1, &W[i + (iw + 1) * ldw], ldw);
                clacgv(n - i - 1, &A[i + (i + 1) * lda], lda);
                cblas_cgemv(CblasColMajor, CblasNoTrans,
                            i + 1, n - i - 1, &NEG_ONE,
                            &W[(iw + 1) * ldw], ldw,
                            &A[i + (i + 1) * lda], lda,
                            &ONE, &A[i * lda], 1);
                clacgv(n - i - 1, &A[i + (i + 1) * lda], lda);
                A[i + i * lda] = CMPLXF(crealf(A[i + i * lda]), 0.0f);
            }

            if (i > 0) {
                /* Generate elementary reflector H(i) to annihilate A(0:i-2, i). */
                alpha = A[(i - 1) + i * lda];
                clarfg(i,
                       &alpha,
                       &A[i * lda],
                       1,
                       &tau[i - 1]);

                E[i - 1] = crealf(alpha);
                A[(i - 1) + i * lda] = ONE;

                /* Compute W(0:i-1, iw). */
                cblas_chemv(CblasColMajor, CblasUpper,
                            i, &ONE,
                            A, lda,
                            &A[i * lda], 1,
                            &ZERO, &W[iw * ldw], 1);

                if (i < n - 1) {
                    cblas_cgemv(CblasColMajor, CblasConjTrans,
                                i, n - i - 1, &ONE,
                                &W[(iw + 1) * ldw], ldw,
                                &A[i * lda], 1,
                                &ZERO, &W[(i + 1) + iw * ldw], 1);
                    cblas_cgemv(CblasColMajor, CblasNoTrans,
                                i, n - i - 1, &NEG_ONE,
                                &A[(i + 1) * lda], lda,
                                &W[(i + 1) + iw * ldw], 1,
                                &ONE, &W[iw * ldw], 1);
                    cblas_cgemv(CblasColMajor, CblasConjTrans,
                                i, n - i - 1, &ONE,
                                &A[(i + 1) * lda], lda,
                                &A[i * lda], 1,
                                &ZERO, &W[(i + 1) + iw * ldw], 1);
                    cblas_cgemv(CblasColMajor, CblasNoTrans,
                                i, n - i - 1, &NEG_ONE,
                                &W[(iw + 1) * ldw], ldw,
                                &W[(i + 1) + iw * ldw], 1,
                                &ONE, &W[iw * ldw], 1);
                }

                cblas_cscal(i, &tau[i - 1], &W[iw * ldw], 1);
                c64 dotc;
                cblas_cdotc_sub(i,
                                &W[iw * ldw], 1,
                                &A[i * lda], 1,
                                &dotc);
                alpha = -HALF * tau[i - 1] * dotc;
                cblas_caxpy(i, &alpha,
                            &A[i * lda], 1,
                            &W[iw * ldw], 1);
            }
        }

    } else {
        /* Reduce first NB columns of lower triangle. */
        for (i = 0; i < nb; i++) {
            /* Update A(i:n-1, i). */
            A[i + i * lda] = CMPLXF(crealf(A[i + i * lda]), 0.0f);
            clacgv(i, &W[i], ldw);
            cblas_cgemv(CblasColMajor, CblasNoTrans,
                        n - i, i, &NEG_ONE,
                        &A[i], lda,
                        &W[i], ldw,
                        &ONE, &A[i + i * lda], 1);
            clacgv(i, &W[i], ldw);
            clacgv(i, &A[i], lda);
            cblas_cgemv(CblasColMajor, CblasNoTrans,
                        n - i, i, &NEG_ONE,
                        &W[i], ldw,
                        &A[i], lda,
                        &ONE, &A[i + i * lda], 1);
            clacgv(i, &A[i], lda);
            A[i + i * lda] = CMPLXF(crealf(A[i + i * lda]), 0.0f);

            if (i < n - 1) {
                /* Generate elementary reflector H(i) to annihilate A(i+2:n-1, i). */
                alpha = A[(i + 1) + i * lda];
                INT min_row = (i + 2 < n - 1) ? (i + 2) : (n - 1);
                clarfg(n - i - 1,
                       &alpha,
                       &A[min_row + i * lda],
                       1,
                       &tau[i]);

                E[i] = crealf(alpha);
                A[(i + 1) + i * lda] = ONE;

                /* Compute W(i+1:n-1, i). */
                cblas_chemv(CblasColMajor, CblasLower,
                            n - i - 1, &ONE,
                            &A[(i + 1) + (i + 1) * lda], lda,
                            &A[(i + 1) + i * lda], 1,
                            &ZERO, &W[(i + 1) + i * ldw], 1);
                cblas_cgemv(CblasColMajor, CblasConjTrans,
                            n - i - 1, i, &ONE,
                            &W[i + 1], ldw,
                            &A[(i + 1) + i * lda], 1,
                            &ZERO, &W[i * ldw], 1);
                cblas_cgemv(CblasColMajor, CblasNoTrans,
                            n - i - 1, i, &NEG_ONE,
                            &A[i + 1], lda,
                            &W[i * ldw], 1,
                            &ONE, &W[(i + 1) + i * ldw], 1);
                cblas_cgemv(CblasColMajor, CblasConjTrans,
                            n - i - 1, i, &ONE,
                            &A[i + 1], lda,
                            &A[(i + 1) + i * lda], 1,
                            &ZERO, &W[i * ldw], 1);
                cblas_cgemv(CblasColMajor, CblasNoTrans,
                            n - i - 1, i, &NEG_ONE,
                            &W[i + 1], ldw,
                            &W[i * ldw], 1,
                            &ONE, &W[(i + 1) + i * ldw], 1);

                cblas_cscal(n - i - 1, &tau[i], &W[(i + 1) + i * ldw], 1);
                c64 dotc;
                cblas_cdotc_sub(n - i - 1,
                                &W[(i + 1) + i * ldw], 1,
                                &A[(i + 1) + i * lda], 1,
                                &dotc);
                alpha = -HALF * tau[i] * dotc;
                cblas_caxpy(n - i - 1, &alpha,
                            &A[(i + 1) + i * lda], 1,
                            &W[(i + 1) + i * ldw], 1);
            }
        }
    }
}
