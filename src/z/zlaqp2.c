/**
 * @file zlaqp2.c
 * @brief ZLAQP2 computes a QR factorization with column pivoting of a matrix
 *        block using an unblocked algorithm.
 */

#include <math.h>
#include <float.h>
#include <complex.h>
#include "semicolon_cblas.h"
#include "semicolon_lapack_complex_double.h"

/**
 * ZLAQP2 computes a QR factorization with column pivoting of
 * the block A(offset:m-1, 0:n-1).
 * The block A(0:offset-1, 0:n-1) is accordingly pivoted, but not factorized.
 *
 * @param[in]     m       The number of rows of the matrix A. m >= 0.
 * @param[in]     n       The number of columns of the matrix A. n >= 0.
 * @param[in]     offset  The number of rows of the matrix A that must be
 *                        pivoted but not factorized. offset >= 0.
 * @param[in,out] A       Double complex array, dimension (lda, n).
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
 * @param[out]    tau     Double complex array, dimension (min(m-offset, n)).
 *                        The scalar factors of the elementary reflectors.
 * @param[in,out] vn1     Double precision array, dimension (n).
 *                        The vector with the partial column norms.
 * @param[in,out] vn2     Double precision array, dimension (n).
 *                        The vector with the exact column norms.
 * @param[out]    work    Double complex array, dimension (n).
 */
void zlaqp2(const INT m, const INT n, const INT offset,
            c128* restrict A, const INT lda,
            INT* restrict jpvt,
            c128* restrict tau,
            f64* restrict vn1,
            f64* restrict vn2,
            c128* restrict work)
{
    INT mn = (m - offset) < n ? (m - offset) : n;
    f64 tol3z = sqrt(DBL_EPSILON);

    /* Compute factorization. */
    for (INT i = 0; i < mn; i++) {
        INT offpi = offset + i;

        /* Determine i-th pivot column and swap if necessary. */
        INT pvt = i + cblas_idamax(n - i, &vn1[i], 1);

        if (pvt != i) {
            cblas_zswap(m, &A[0 + pvt * lda], 1, &A[0 + i * lda], 1);
            INT itemp = jpvt[pvt];
            jpvt[pvt] = jpvt[i];
            jpvt[i] = itemp;
            vn1[pvt] = vn1[i];
            vn2[pvt] = vn2[i];
        }

        /* Generate elementary reflector H(i). */
        if (offpi < m - 1) {
            zlarfg(m - offpi, &A[offpi + i * lda],
                   &A[offpi + 1 + i * lda], 1, &tau[i]);
        } else {
            zlarfg(1, &A[m - 1 + i * lda],
                   &A[m - 1 + i * lda], 1, &tau[i]);
        }

        if (i < n - 1) {
            /* Apply H(i)**H to A(offpi:m-1, i+1:n-1) from the left. */
            zlarf1f("L", m - offpi, n - 1 - i,
                    &A[offpi + i * lda], 1, conj(tau[i]),
                    &A[offpi + (i + 1) * lda], lda, work);
        }

        /* Update partial column norms. */
        for (INT j = i + 1; j < n; j++) {
            if (vn1[j] != 0.0) {
                /*
                 * NOTE: The following lines follow from the analysis in
                 * LAPACK Working Note 176.
                 */
                f64 temp = 1.0 - pow(cabs(A[offpi + j * lda]) / vn1[j], 2);
                temp = temp > 0.0 ? temp : 0.0;
                f64 temp2 = temp * pow(vn1[j] / vn2[j], 2);
                if (temp2 <= tol3z) {
                    if (offpi < m - 1) {
                        vn1[j] = cblas_dznrm2(m - offpi - 1,
                                             &A[offpi + 1 + j * lda], 1);
                        vn2[j] = vn1[j];
                    } else {
                        vn1[j] = 0.0;
                        vn2[j] = 0.0;
                    }
                } else {
                    vn1[j] = vn1[j] * sqrt(temp);
                }
            }
        }
    }
}
