/**
 * @file slaqp2.c
 * @brief SLAQP2 computes a QR factorization with column pivoting of a matrix
 *        block using an unblocked algorithm.
 */

#include <math.h>
#include <float.h>
#include <cblas.h>
#include "semicolon_lapack_single.h"

/**
 * SLAQP2 computes a QR factorization with column pivoting of
 * the block A(offset:m-1, 0:n-1).
 * The block A(0:offset-1, 0:n-1) is accordingly pivoted, but not factorized.
 *
 * @param[in]     m       The number of rows of the matrix A. m >= 0.
 * @param[in]     n       The number of columns of the matrix A. n >= 0.
 * @param[in]     offset  The number of rows of the matrix A that must be
 *                        pivoted but not factorized. offset >= 0.
 * @param[in,out] A       Double precision array, dimension (lda, n).
 *                        On entry, the m-by-n matrix A.
 *                        On exit, the upper triangle of block
 *                        A(offset:m-1, 0:n-1) is the triangular factor
 *                        obtained; the elements in block A(offset:m-1, 0:n-1)
 *                        below the diagonal, together with the array tau,
 *                        represent the orthogonal matrix Q as a product of
 *                        elementary reflectors. Block A(0:offset-1, 0:n-1) has
 *                        been accordingly pivoted, but not factorized.
 * @param[in]     lda     The leading dimension of the array A. lda >= max(1, m).
 * @param[in,out] jpvt    Integer array, dimension (n).
 *                        On entry, if jpvt[i] != 0, the i-th column of A is
 *                        permuted to the front of A*P (a leading column);
 *                        if jpvt[i] = 0, the i-th column of A is a free column.
 *                        On exit, if jpvt[i] = k, then the i-th column of A*P
 *                        was the k-th column of A.
 * @param[out]    tau     Double precision array, dimension (min(m-offset, n)).
 *                        The scalar factors of the elementary reflectors.
 * @param[in,out] vn1     Double precision array, dimension (n).
 *                        The vector with the partial column norms.
 * @param[in,out] vn2     Double precision array, dimension (n).
 *                        The vector with the exact column norms.
 * @param[out]    work    Double precision array, dimension (n).
 */
void slaqp2(const int m, const int n, const int offset,
            f32 * const restrict A, const int lda,
            int * const restrict jpvt,
            f32 * const restrict tau,
            f32 * const restrict vn1,
            f32 * const restrict vn2,
            f32 * const restrict work)
{
    int mn = (m - offset) < n ? (m - offset) : n;
    f32 tol3z = sqrtf(FLT_EPSILON);

    /* Compute factorization. */
    for (int i = 0; i < mn; i++) {
        int offpi = offset + i;

        /* Determine i-th pivot column and swap if necessary. */
        int pvt = i + cblas_isamax(n - i, &vn1[i], 1);

        if (pvt != i) {
            cblas_sswap(m, &A[0 + pvt * lda], 1, &A[0 + i * lda], 1);
            int itemp = jpvt[pvt];
            jpvt[pvt] = jpvt[i];
            jpvt[i] = itemp;
            vn1[pvt] = vn1[i];
            vn2[pvt] = vn2[i];
        }

        /* Generate elementary reflector H(i). */
        if (offpi < m - 1) {
            slarfg(m - offpi, &A[offpi + i * lda],
                   &A[offpi + 1 + i * lda], 1, &tau[i]);
        } else {
            slarfg(1, &A[m - 1 + i * lda],
                   &A[m - 1 + i * lda], 1, &tau[i]);
        }

        if (i < n - 1) {
            /* Apply H(i)**T to A(offpi:m-1, i+1:n-1) from the left. */
            slarf1f("L", m - offpi, n - 1 - i,
                    &A[offpi + i * lda], 1, tau[i],
                    &A[offpi + (i + 1) * lda], lda, work);
        }

        /* Update partial column norms. */
        for (int j = i + 1; j < n; j++) {
            if (vn1[j] != 0.0f) {
                /*
                 * NOTE: The following lines follow from the analysis in
                 * LAPACK Working Note 176.
                 */
                f32 temp = 1.0f - powf(fabsf(A[offpi + j * lda]) / vn1[j], 2);
                temp = temp > 0.0f ? temp : 0.0f;
                f32 temp2 = temp * powf(vn1[j] / vn2[j], 2);
                if (temp2 <= tol3z) {
                    if (offpi < m - 1) {
                        vn1[j] = cblas_snrm2(m - offpi - 1,
                                             &A[offpi + 1 + j * lda], 1);
                        vn2[j] = vn1[j];
                    } else {
                        vn1[j] = 0.0f;
                        vn2[j] = 0.0f;
                    }
                } else {
                    vn1[j] = vn1[j] * sqrtf(temp);
                }
            }
        }
    }
}
