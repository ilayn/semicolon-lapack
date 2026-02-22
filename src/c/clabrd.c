/**
 * @file clabrd.c
 * @brief CLABRD reduces the first nb rows and columns of a general matrix to a bidiagonal form.
 */

#include "internal_build_defs.h"
#include "semicolon_lapack_complex_single.h"
#include <complex.h>
#include <cblas.h>

/**
 * CLABRD reduces the first NB rows and columns of a complex general
 * m by n matrix A to upper or lower real bidiagonal form by a unitary
 * transformation Q**H * A * P, and returns the matrices X and Y which
 * are needed to apply the transformation to the unreduced part of A.
 *
 * If m >= n, A is reduced to upper bidiagonal form; if m < n, to lower
 * bidiagonal form.
 *
 * This is an auxiliary routine called by CGEBRD.
 *
 * @param[in]     m     The number of rows in the matrix A.
 * @param[in]     n     The number of columns in the matrix A.
 * @param[in]     nb    The number of leading rows and columns of A to be reduced.
 * @param[in,out] A     Complex array, dimension (lda, n).
 *                      On entry, the m by n general matrix to be reduced.
 *                      On exit, the first NB rows and columns of the matrix are
 *                      overwritten; the rest of the array is unchanged.
 * @param[in]     lda   The leading dimension of the array A. lda >= max(1,m).
 * @param[out]    D     Single precision array, dimension (nb).
 *                      The diagonal elements of the first NB rows and columns of
 *                      the reduced matrix. D[i] = A[i,i].
 * @param[out]    E     Single precision array, dimension (nb).
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
void clabrd(const INT m, const INT n, const INT nb,
            c64* restrict A, const INT lda,
            f32* restrict D, f32* restrict E,
            c64* restrict tauq, c64* restrict taup,
            c64* restrict X, const INT ldx,
            c64* restrict Y, const INT ldy)
{
    const c64 ZERO = CMPLXF(0.0f, 0.0f);
    const c64 ONE  = CMPLXF(1.0f, 0.0f);
    const c64 NEG_ONE = CMPLXF(-1.0f, 0.0f);

    INT i;
    c64 alpha;

    /* Quick return if possible */
    if (m <= 0 || n <= 0) {
        return;
    }

    if (m >= n) {
        /* Reduce to upper bidiagonal form */
        for (i = 0; i < nb; i++) {
            /* Update A[i:m-1, i] */
            clacgv(i, &Y[i], ldy);
            cblas_cgemv(CblasColMajor, CblasNoTrans, m - i, i,
                        &NEG_ONE, &A[i], lda, &Y[i], ldy,
                        &ONE, &A[i + i * lda], 1);
            clacgv(i, &Y[i], ldy);
            cblas_cgemv(CblasColMajor, CblasNoTrans, m - i, i,
                        &NEG_ONE, &X[i], ldx, &A[i * lda], 1,
                        &ONE, &A[i + i * lda], 1);

            /* Generate reflection Q(i) to annihilate A[i+1:m-1, i] */
            alpha = A[i + i * lda];
            clarfg(m - i, &alpha, &A[((i + 1) < m ? (i + 1) : (m - 1)) + i * lda], 1,
                   &tauq[i]);
            D[i] = crealf(alpha);

            if (i < n - 1) {
                A[i + i * lda] = ONE;

                /* Compute Y[i+1:n-1, i] */
                cblas_cgemv(CblasColMajor, CblasConjTrans, m - i, n - i - 1,
                            &ONE, &A[i + (i + 1) * lda], lda,
                            &A[i + i * lda], 1, &ZERO, &Y[i + 1 + i * ldy], 1);
                cblas_cgemv(CblasColMajor, CblasConjTrans, m - i, i,
                            &ONE, &A[i], lda, &A[i + i * lda], 1,
                            &ZERO, &Y[i * ldy], 1);
                cblas_cgemv(CblasColMajor, CblasNoTrans, n - i - 1, i,
                            &NEG_ONE, &Y[i + 1], ldy, &Y[i * ldy], 1,
                            &ONE, &Y[i + 1 + i * ldy], 1);
                cblas_cgemv(CblasColMajor, CblasConjTrans, m - i, i,
                            &ONE, &X[i], ldx, &A[i + i * lda], 1,
                            &ZERO, &Y[i * ldy], 1);
                cblas_cgemv(CblasColMajor, CblasConjTrans, i, n - i - 1,
                            &NEG_ONE, &A[(i + 1) * lda], lda, &Y[i * ldy], 1,
                            &ONE, &Y[i + 1 + i * ldy], 1);
                cblas_cscal(n - i - 1, &tauq[i], &Y[i + 1 + i * ldy], 1);

                /* Update A[i, i+1:n-1] */
                clacgv(n - i - 1, &A[i + (i + 1) * lda], lda);
                clacgv(i + 1, &A[i], lda);
                cblas_cgemv(CblasColMajor, CblasNoTrans, n - i - 1, i + 1,
                            &NEG_ONE, &Y[i + 1], ldy, &A[i], lda,
                            &ONE, &A[i + (i + 1) * lda], lda);
                clacgv(i + 1, &A[i], lda);
                clacgv(i, &X[i], ldx);
                cblas_cgemv(CblasColMajor, CblasConjTrans, i, n - i - 1,
                            &NEG_ONE, &A[(i + 1) * lda], lda, &X[i], ldx,
                            &ONE, &A[i + (i + 1) * lda], lda);
                clacgv(i, &X[i], ldx);

                /* Generate reflection P(i) to annihilate A[i, i+2:n-1] */
                alpha = A[i + (i + 1) * lda];
                clarfg(n - i - 1, &alpha,
                       &A[i + ((i + 2) < n ? (i + 2) : (n - 1)) * lda], lda,
                       &taup[i]);
                E[i] = crealf(alpha);
                A[i + (i + 1) * lda] = ONE;

                /* Compute X[i+1:m-1, i] */
                cblas_cgemv(CblasColMajor, CblasNoTrans, m - i - 1, n - i - 1,
                            &ONE, &A[i + 1 + (i + 1) * lda], lda,
                            &A[i + (i + 1) * lda], lda, &ZERO, &X[i + 1 + i * ldx], 1);
                cblas_cgemv(CblasColMajor, CblasConjTrans, n - i - 1, i + 1,
                            &ONE, &Y[i + 1], ldy, &A[i + (i + 1) * lda], lda,
                            &ZERO, &X[i * ldx], 1);
                cblas_cgemv(CblasColMajor, CblasNoTrans, m - i - 1, i + 1,
                            &NEG_ONE, &A[i + 1], lda, &X[i * ldx], 1,
                            &ONE, &X[i + 1 + i * ldx], 1);
                cblas_cgemv(CblasColMajor, CblasNoTrans, i, n - i - 1,
                            &ONE, &A[(i + 1) * lda], lda,
                            &A[i + (i + 1) * lda], lda, &ZERO, &X[i * ldx], 1);
                cblas_cgemv(CblasColMajor, CblasNoTrans, m - i - 1, i,
                            &NEG_ONE, &X[i + 1], ldx, &X[i * ldx], 1,
                            &ONE, &X[i + 1 + i * ldx], 1);
                cblas_cscal(m - i - 1, &taup[i], &X[i + 1 + i * ldx], 1);
                clacgv(n - i - 1, &A[i + (i + 1) * lda], lda);
            }
        }
    } else {
        /* Reduce to lower bidiagonal form */
        for (i = 0; i < nb; i++) {
            /* Update A[i, i:n-1] */
            clacgv(n - i, &A[i + i * lda], lda);
            clacgv(i, &A[i], lda);
            cblas_cgemv(CblasColMajor, CblasNoTrans, n - i, i,
                        &NEG_ONE, &Y[i], ldy, &A[i], lda,
                        &ONE, &A[i + i * lda], lda);
            clacgv(i, &A[i], lda);
            clacgv(i, &X[i], ldx);
            cblas_cgemv(CblasColMajor, CblasConjTrans, i, n - i,
                        &NEG_ONE, &A[i * lda], lda, &X[i], ldx,
                        &ONE, &A[i + i * lda], lda);
            clacgv(i, &X[i], ldx);

            /* Generate reflection P(i) to annihilate A[i, i+1:n-1] */
            alpha = A[i + i * lda];
            clarfg(n - i, &alpha, &A[i + ((i + 1) < n ? (i + 1) : (n - 1)) * lda], lda,
                   &taup[i]);
            D[i] = crealf(alpha);

            if (i < m - 1) {
                A[i + i * lda] = ONE;

                /* Compute X[i+1:m-1, i] */
                cblas_cgemv(CblasColMajor, CblasNoTrans, m - i - 1, n - i,
                            &ONE, &A[i + 1 + i * lda], lda,
                            &A[i + i * lda], lda, &ZERO, &X[i + 1 + i * ldx], 1);
                cblas_cgemv(CblasColMajor, CblasConjTrans, n - i, i,
                            &ONE, &Y[i], ldy, &A[i + i * lda], lda,
                            &ZERO, &X[i * ldx], 1);
                cblas_cgemv(CblasColMajor, CblasNoTrans, m - i - 1, i,
                            &NEG_ONE, &A[i + 1], lda, &X[i * ldx], 1,
                            &ONE, &X[i + 1 + i * ldx], 1);
                cblas_cgemv(CblasColMajor, CblasNoTrans, i, n - i,
                            &ONE, &A[i * lda], lda, &A[i + i * lda], lda,
                            &ZERO, &X[i * ldx], 1);
                cblas_cgemv(CblasColMajor, CblasNoTrans, m - i - 1, i,
                            &NEG_ONE, &X[i + 1], ldx, &X[i * ldx], 1,
                            &ONE, &X[i + 1 + i * ldx], 1);
                cblas_cscal(m - i - 1, &taup[i], &X[i + 1 + i * ldx], 1);
                clacgv(n - i, &A[i + i * lda], lda);

                /* Update A[i+1:m-1, i] */
                clacgv(i, &Y[i], ldy);
                cblas_cgemv(CblasColMajor, CblasNoTrans, m - i - 1, i,
                            &NEG_ONE, &A[i + 1], lda, &Y[i], ldy,
                            &ONE, &A[i + 1 + i * lda], 1);
                clacgv(i, &Y[i], ldy);
                cblas_cgemv(CblasColMajor, CblasNoTrans, m - i - 1, i + 1,
                            &NEG_ONE, &X[i + 1], ldx, &A[i * lda], 1,
                            &ONE, &A[i + 1 + i * lda], 1);

                /* Generate reflection Q(i) to annihilate A[i+2:m-1, i] */
                alpha = A[i + 1 + i * lda];
                clarfg(m - i - 1, &alpha,
                       &A[((i + 2) < m ? (i + 2) : (m - 1)) + i * lda], 1,
                       &tauq[i]);
                E[i] = crealf(alpha);
                A[i + 1 + i * lda] = ONE;

                /* Compute Y[i+1:n-1, i] */
                cblas_cgemv(CblasColMajor, CblasConjTrans, m - i - 1, n - i - 1,
                            &ONE, &A[i + 1 + (i + 1) * lda], lda,
                            &A[i + 1 + i * lda], 1, &ZERO, &Y[i + 1 + i * ldy], 1);
                cblas_cgemv(CblasColMajor, CblasConjTrans, m - i - 1, i,
                            &ONE, &A[i + 1], lda, &A[i + 1 + i * lda], 1,
                            &ZERO, &Y[i * ldy], 1);
                cblas_cgemv(CblasColMajor, CblasNoTrans, n - i - 1, i,
                            &NEG_ONE, &Y[i + 1], ldy, &Y[i * ldy], 1,
                            &ONE, &Y[i + 1 + i * ldy], 1);
                cblas_cgemv(CblasColMajor, CblasConjTrans, m - i - 1, i + 1,
                            &ONE, &X[i + 1], ldx, &A[i + 1 + i * lda], 1,
                            &ZERO, &Y[i * ldy], 1);
                cblas_cgemv(CblasColMajor, CblasConjTrans, i + 1, n - i - 1,
                            &NEG_ONE, &A[(i + 1) * lda], lda, &Y[i * ldy], 1,
                            &ONE, &Y[i + 1 + i * ldy], 1);
                cblas_cscal(n - i - 1, &tauq[i], &Y[i + 1 + i * ldy], 1);
            } else {
                clacgv(n - i, &A[i + i * lda], lda);
            }
        }
    }
}
