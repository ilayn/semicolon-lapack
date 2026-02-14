/**
 * @file zlahef_aa.c
 * @brief ZLAHEF_AA factorizes a panel of a complex hermitian matrix A using Aasen's algorithm.
 */

#include <complex.h>
#include <math.h>
#include <cblas.h>
#include "semicolon_lapack_complex_double.h"

/**
 * ZLAHEF_AA factorizes a panel of a complex hermitian matrix A using
 * the Aasen's algorithm. The panel consists of a set of NB rows of A
 * when UPLO is U, or a set of NB columns when UPLO is L.
 *
 * In order to factorize the panel, the Aasen's algorithm requires the
 * last row, or column, of the previous panel. The first row, or column,
 * of A is set to be the first row, or column, of an identity matrix,
 * which is used to factorize the first panel.
 *
 * The resulting J-th row of U, or J-th column of L, is stored in the
 * (J-1)-th row, or column, of A (without the unit diagonals), while
 * the diagonal and subdiagonal of A are overwritten by those of T.
 *
 * @param[in] uplo
 *          = 'U':  Upper triangle of A is stored;
 *          = 'L':  Lower triangle of A is stored.
 *
 * @param[in] j1
 *          The location of the first row, or column, of the panel
 *          within the submatrix of A, passed to this routine, e.g.,
 *          when called by ZHETRF_AA, for the first panel, J1 is 1,
 *          while for the remaining panels, J1 is 2.
 *
 * @param[in] m
 *          The dimension of the submatrix. M >= 0.
 *
 * @param[in] nb
 *          The dimension of the panel to be factorized.
 *
 * @param[in,out] A
 *          Double complex array, dimension (lda, m) for
 *          the first panel, while dimension (lda, m+1) for the
 *          remaining panels.
 *
 * @param[in] lda
 *          The leading dimension of the array A. lda >= max(1, m).
 *
 * @param[out] ipiv
 *          Integer array, dimension (m).
 *          Details of the row and column interchanges.
 *
 * @param[in,out] H
 *          Double complex workspace, dimension (ldh, nb).
 *
 * @param[in] ldh
 *          The leading dimension of the workspace H. ldh >= max(1, m).
 *
 * @param[out] work
 *          Double complex workspace, dimension (m).
 */
void zlahef_aa(
    const char* uplo,
    const int j1,
    const int m,
    const int nb,
    c128* restrict A,
    const int lda,
    int* restrict ipiv,
    c128* restrict H,
    const int ldh,
    c128* restrict work)
{
    const c128 ONE = CMPLX(1.0, 0.0);
    const c128 NEG_ONE = CMPLX(-1.0, 0.0);

    int j, k, k1, i1, i2, mj;
    c128 piv, alpha;
    int minval;

    j = 0;

    k1 = (2 - j1) + 1 - 1;

    if (uplo[0] == 'U' || uplo[0] == 'u') {

        minval = (m < nb) ? m : nb;
        while (j < minval) {

            k = j1 + j - 1;
            if (j == m - 1) {
                mj = 1;
            } else {
                mj = m - j;
            }

            if (k > 1) {
                zlacgv(j - k1, &A[0 + j * lda], 1);
                cblas_zgemv(CblasColMajor, CblasNoTrans, mj, j - k1,
                            &NEG_ONE, &H[j + k1 * ldh], ldh,
                            &A[0 + j * lda], 1,
                            &ONE, &H[j + j * ldh], 1);
                zlacgv(j - k1, &A[0 + j * lda], 1);
            }

            cblas_zcopy(mj, &H[j + j * ldh], 1, &work[0], 1);

            if (j > k1) {
                alpha = -conj(A[(k - 1) + j * lda]);
                cblas_zaxpy(mj, &alpha, &A[(k - 2) + j * lda], lda, &work[0], 1);
            }

            A[k + j * lda] = CMPLX(creal(work[0]), 0.0);

            if (j < m - 1) {

                if (k > 0) {
                    alpha = -A[k + j * lda];
                    cblas_zaxpy(m - j - 1, &alpha, &A[(k - 1) + (j + 1) * lda], lda,
                                &work[1], 1);
                }

                i2 = cblas_izamax(m - j - 1, &work[1], 1) + 1;
                piv = work[i2];

                if ((i2 != 1) && (piv != CMPLX(0.0, 0.0))) {

                    i1 = 1;
                    work[i2] = work[i1];
                    work[i1] = piv;

                    i1 = i1 + j;
                    i2 = i2 + j;
                    cblas_zswap(i2 - i1 - 1, &A[(j1 + i1 - 1) + (i1 + 1) * lda], lda,
                                &A[(j1 + i1) + i2 * lda], 1);
                    zlacgv(i2 - i1, &A[(j1 + i1 - 1) + (i1 + 1) * lda], lda);
                    zlacgv(i2 - i1 - 1, &A[(j1 + i1) + i2 * lda], 1);

                    if (i2 < m - 1) {
                        cblas_zswap(m - i2 - 1, &A[(j1 + i1 - 1) + (i2 + 1) * lda], lda,
                                    &A[(j1 + i2 - 1) + (i2 + 1) * lda], lda);
                    }

                    piv = A[(i1 + j1 - 1) + i1 * lda];
                    A[(j1 + i1 - 1) + i1 * lda] = A[(j1 + i2 - 1) + i2 * lda];
                    A[(j1 + i2 - 1) + i2 * lda] = piv;

                    cblas_zswap(i1, &H[i1 + 0 * ldh], ldh, &H[i2 + 0 * ldh], ldh);
                    ipiv[i1] = i2;

                    if (i1 > k1) {
                        cblas_zswap(i1 - k1 + 1, &A[0 + i1 * lda], 1,
                                    &A[0 + i2 * lda], 1);
                    }
                } else {
                    ipiv[j + 1] = j + 1;
                }

                A[k + (j + 1) * lda] = work[1];

                if (j < nb - 1) {
                    cblas_zcopy(m - j - 1, &A[(k + 1) + (j + 1) * lda], lda,
                                &H[(j + 1) + (j + 1) * ldh], 1);
                }

                if (j < m - 2) {
                    if (A[k + (j + 1) * lda] != CMPLX(0.0, 0.0)) {
                        alpha = ONE / A[k + (j + 1) * lda];
                        cblas_zcopy(m - j - 2, &work[2], 1, &A[k + (j + 2) * lda], lda);
                        cblas_zscal(m - j - 2, &alpha, &A[k + (j + 2) * lda], lda);
                    } else {
                        zlaset("Full", 1, m - j - 2, CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
                               &A[k + (j + 2) * lda], lda);
                    }
                }
            }
            j = j + 1;
        }

    } else {

        minval = (m < nb) ? m : nb;
        while (j < minval) {

            k = j1 + j - 1;
            if (j == m - 1) {
                mj = 1;
            } else {
                mj = m - j;
            }

            if (k > 1) {
                zlacgv(j - k1, &A[j + 0 * lda], lda);
                cblas_zgemv(CblasColMajor, CblasNoTrans, mj, j - k1,
                            &NEG_ONE, &H[j + k1 * ldh], ldh,
                            &A[j + 0 * lda], lda,
                            &ONE, &H[j + j * ldh], 1);
                zlacgv(j - k1, &A[j + 0 * lda], lda);
            }

            cblas_zcopy(mj, &H[j + j * ldh], 1, &work[0], 1);

            if (j > k1) {
                alpha = -conj(A[j + (k - 1) * lda]);
                cblas_zaxpy(mj, &alpha, &A[j + (k - 2) * lda], 1, &work[0], 1);
            }

            A[j + k * lda] = CMPLX(creal(work[0]), 0.0);

            if (j < m - 1) {

                if (k > 0) {
                    alpha = -A[j + k * lda];
                    cblas_zaxpy(m - j - 1, &alpha, &A[(j + 1) + (k - 1) * lda], 1,
                                &work[1], 1);
                }

                i2 = cblas_izamax(m - j - 1, &work[1], 1) + 1;
                piv = work[i2];

                if ((i2 != 1) && (piv != CMPLX(0.0, 0.0))) {

                    i1 = 1;
                    work[i2] = work[i1];
                    work[i1] = piv;

                    i1 = i1 + j;
                    i2 = i2 + j;
                    cblas_zswap(i2 - i1 - 1, &A[(i1 + 1) + (j1 + i1 - 1) * lda], 1,
                                &A[i2 + (j1 + i1) * lda], lda);
                    zlacgv(i2 - i1, &A[(i1 + 1) + (j1 + i1 - 1) * lda], 1);
                    zlacgv(i2 - i1 - 1, &A[i2 + (j1 + i1) * lda], lda);

                    if (i2 < m - 1) {
                        cblas_zswap(m - i2 - 1, &A[(i2 + 1) + (j1 + i1 - 1) * lda], 1,
                                    &A[(i2 + 1) + (j1 + i2 - 1) * lda], 1);
                    }

                    piv = A[i1 + (j1 + i1 - 1) * lda];
                    A[i1 + (j1 + i1 - 1) * lda] = A[i2 + (j1 + i2 - 1) * lda];
                    A[i2 + (j1 + i2 - 1) * lda] = piv;

                    cblas_zswap(i1, &H[i1 + 0 * ldh], ldh, &H[i2 + 0 * ldh], ldh);
                    ipiv[i1] = i2;

                    if (i1 > k1) {
                        cblas_zswap(i1 - k1 + 1, &A[i1 + 0 * lda], lda,
                                    &A[i2 + 0 * lda], lda);
                    }
                } else {
                    ipiv[j + 1] = j + 1;
                }

                A[(j + 1) + k * lda] = work[1];

                if (j < nb - 1) {
                    cblas_zcopy(m - j - 1, &A[(j + 1) + (k + 1) * lda], 1,
                                &H[(j + 1) + (j + 1) * ldh], 1);
                }

                if (j < m - 2) {
                    if (A[(j + 1) + k * lda] != CMPLX(0.0, 0.0)) {
                        alpha = ONE / A[(j + 1) + k * lda];
                        cblas_zcopy(m - j - 2, &work[2], 1, &A[(j + 2) + k * lda], 1);
                        cblas_zscal(m - j - 2, &alpha, &A[(j + 2) + k * lda], 1);
                    } else {
                        zlaset("Full", m - j - 2, 1, CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
                               &A[(j + 2) + k * lda], lda);
                    }
                }
            }
            j = j + 1;
        }
    }
}
