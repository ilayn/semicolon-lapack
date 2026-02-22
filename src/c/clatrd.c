/**
 * @file clatrd.c
 * @brief CLATRD reduces NB rows and columns of a complex Hermitian matrix to
 *        real tridiagonal form.
 */

#include "internal_build_defs.h"
#include "semicolon_lapack_complex_single.h"
#include <complex.h>
#include <cblas.h>

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

    INT ii, iw;
    c64 alpha;

    /* Quick return if possible */
    if (n <= 0) {
        return;
    }

    if (uplo[0] == 'U' || uplo[0] == 'u') {
        /* Reduce last NB columns of upper triangle. */
        for (ii = n; ii >= n - nb + 1; ii--) {
            iw = ii - n + nb;

            if (ii < n) {
                /* Update A(1:ii, ii). */
                A[(ii - 1) + (ii - 1) * lda] = CMPLXF(crealf(A[(ii - 1) + (ii - 1) * lda]), 0.0f);
                clacgv(n - ii, &W[(ii - 1) + iw * ldw], ldw);
                cblas_cgemv(CblasColMajor, CblasNoTrans,
                            ii, n - ii, &NEG_ONE,
                            &A[ii * lda], lda,
                            &W[(ii - 1) + iw * ldw], ldw,
                            &ONE, &A[(ii - 1) * lda], 1);
                clacgv(n - ii, &W[(ii - 1) + iw * ldw], ldw);
                clacgv(n - ii, &A[(ii - 1) + ii * lda], lda);
                cblas_cgemv(CblasColMajor, CblasNoTrans,
                            ii, n - ii, &NEG_ONE,
                            &W[iw * ldw], ldw,
                            &A[(ii - 1) + ii * lda], lda,
                            &ONE, &A[(ii - 1) * lda], 1);
                clacgv(n - ii, &A[(ii - 1) + ii * lda], lda);
                A[(ii - 1) + (ii - 1) * lda] = CMPLXF(crealf(A[(ii - 1) + (ii - 1) * lda]), 0.0f);
            }

            if (ii > 1) {
                /* Generate elementary reflector H(ii) to annihilate A(1:ii-2, ii). */
                alpha = A[(ii - 2) + (ii - 1) * lda];
                clarfg(ii - 1,
                       &alpha,
                       &A[(ii - 1) * lda],
                       1,
                       &tau[ii - 2]);

                E[ii - 2] = crealf(alpha);
                A[(ii - 2) + (ii - 1) * lda] = ONE;

                /* Compute W(1:ii-1, iw). */
                cblas_chemv(CblasColMajor, CblasUpper,
                            ii - 1, &ONE,
                            A, lda,
                            &A[(ii - 1) * lda], 1,
                            &ZERO, &W[(iw - 1) * ldw], 1);

                if (ii < n) {
                    cblas_cgemv(CblasColMajor, CblasConjTrans,
                                ii - 1, n - ii, &ONE,
                                &W[iw * ldw], ldw,
                                &A[(ii - 1) * lda], 1,
                                &ZERO, &W[ii + (iw - 1) * ldw], 1);
                    cblas_cgemv(CblasColMajor, CblasNoTrans,
                                ii - 1, n - ii, &NEG_ONE,
                                &A[ii * lda], lda,
                                &W[ii + (iw - 1) * ldw], 1,
                                &ONE, &W[(iw - 1) * ldw], 1);
                    cblas_cgemv(CblasColMajor, CblasConjTrans,
                                ii - 1, n - ii, &ONE,
                                &A[ii * lda], lda,
                                &A[(ii - 1) * lda], 1,
                                &ZERO, &W[ii + (iw - 1) * ldw], 1);
                    cblas_cgemv(CblasColMajor, CblasNoTrans,
                                ii - 1, n - ii, &NEG_ONE,
                                &W[iw * ldw], ldw,
                                &W[ii + (iw - 1) * ldw], 1,
                                &ONE, &W[(iw - 1) * ldw], 1);
                }

                cblas_cscal(ii - 1, &tau[ii - 2], &W[(iw - 1) * ldw], 1);
                c64 dotc;
                cblas_cdotc_sub(ii - 1,
                                &W[(iw - 1) * ldw], 1,
                                &A[(ii - 1) * lda], 1,
                                &dotc);
                alpha = -HALF * tau[ii - 2] * dotc;
                cblas_caxpy(ii - 1, &alpha,
                            &A[(ii - 1) * lda], 1,
                            &W[(iw - 1) * ldw], 1);
            }
        }

    } else {
        /* Reduce first NB columns of lower triangle. */
        for (ii = 1; ii <= nb; ii++) {
            /* Update A(ii:n, ii). */
            A[(ii - 1) + (ii - 1) * lda] = CMPLXF(crealf(A[(ii - 1) + (ii - 1) * lda]), 0.0f);
            clacgv(ii - 1, &W[ii - 1], ldw);
            cblas_cgemv(CblasColMajor, CblasNoTrans,
                        n - ii + 1, ii - 1, &NEG_ONE,
                        &A[ii - 1], lda,
                        &W[ii - 1], ldw,
                        &ONE, &A[(ii - 1) + (ii - 1) * lda], 1);
            clacgv(ii - 1, &W[ii - 1], ldw);
            clacgv(ii - 1, &A[ii - 1], lda);
            cblas_cgemv(CblasColMajor, CblasNoTrans,
                        n - ii + 1, ii - 1, &NEG_ONE,
                        &W[ii - 1], ldw,
                        &A[ii - 1], lda,
                        &ONE, &A[(ii - 1) + (ii - 1) * lda], 1);
            clacgv(ii - 1, &A[ii - 1], lda);
            A[(ii - 1) + (ii - 1) * lda] = CMPLXF(crealf(A[(ii - 1) + (ii - 1) * lda]), 0.0f);

            if (ii < n) {
                /* Generate elementary reflector H(ii) to annihilate A(ii+2:n, ii). */
                alpha = A[ii + (ii - 1) * lda];
                INT min_row = (ii + 1 < n - 1) ? (ii + 1) : (n - 1);
                clarfg(n - ii,
                       &alpha,
                       &A[min_row + (ii - 1) * lda],
                       1,
                       &tau[ii - 1]);

                E[ii - 1] = crealf(alpha);
                A[ii + (ii - 1) * lda] = ONE;

                /* Compute W(ii+1:n, ii). */
                cblas_chemv(CblasColMajor, CblasLower,
                            n - ii, &ONE,
                            &A[ii + ii * lda], lda,
                            &A[ii + (ii - 1) * lda], 1,
                            &ZERO, &W[ii + (ii - 1) * ldw], 1);
                cblas_cgemv(CblasColMajor, CblasConjTrans,
                            n - ii, ii - 1, &ONE,
                            &W[ii], ldw,
                            &A[ii + (ii - 1) * lda], 1,
                            &ZERO, &W[(ii - 1) * ldw], 1);
                cblas_cgemv(CblasColMajor, CblasNoTrans,
                            n - ii, ii - 1, &NEG_ONE,
                            &A[ii], lda,
                            &W[(ii - 1) * ldw], 1,
                            &ONE, &W[ii + (ii - 1) * ldw], 1);
                cblas_cgemv(CblasColMajor, CblasConjTrans,
                            n - ii, ii - 1, &ONE,
                            &A[ii], lda,
                            &A[ii + (ii - 1) * lda], 1,
                            &ZERO, &W[(ii - 1) * ldw], 1);
                cblas_cgemv(CblasColMajor, CblasNoTrans,
                            n - ii, ii - 1, &NEG_ONE,
                            &W[ii], ldw,
                            &W[(ii - 1) * ldw], 1,
                            &ONE, &W[ii + (ii - 1) * ldw], 1);

                cblas_cscal(n - ii, &tau[ii - 1], &W[ii + (ii - 1) * ldw], 1);
                c64 dotc;
                cblas_cdotc_sub(n - ii,
                                &W[ii + (ii - 1) * ldw], 1,
                                &A[ii + (ii - 1) * lda], 1,
                                &dotc);
                alpha = -HALF * tau[ii - 1] * dotc;
                cblas_caxpy(n - ii, &alpha,
                            &A[ii + (ii - 1) * lda], 1,
                            &W[ii + (ii - 1) * ldw], 1);
            }
        }
    }
}
