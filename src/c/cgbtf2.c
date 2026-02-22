/**
 * @file cgbtf2.c
 * @brief CGBTF2 computes the LU factorization of a general band matrix
 *        using the unblocked version of the algorithm.
 */

#include <complex.h>
#include "semicolon_cblas.h"
#include "semicolon_lapack_complex_single.h"

/**
 * CGBTF2 computes an LU factorization of a complex m-by-n band matrix A
 * using partial pivoting with row interchanges.
 *
 * This is the unblocked version of the algorithm, calling Level 2 BLAS.
 *
 * The factorization has the form A = P * L * U where P is a permutation
 * matrix, L is lower triangular with unit diagonal elements (lower
 * trapezoidal if m > n), and U is upper triangular (upper trapezoidal
 * if m < n).
 *
 * @param[in]     m     The number of rows of the matrix A. m >= 0.
 * @param[in]     n     The number of columns of the matrix A. n >= 0.
 * @param[in]     kl    The number of subdiagonals within the band of A. kl >= 0.
 * @param[in]     ku    The number of superdiagonals within the band of A. ku >= 0.
 * @param[in,out] AB    Single complex array, dimension (ldab, n).
 *                      On entry, the matrix A in band storage, in rows kl to
 *                      2*kl+ku; rows 0 to kl-1 of the array need not be set.
 *                      The j-th column of A is stored in the j-th column of
 *                      the array AB as follows:
 *                      AB[kl+ku+i-j + j*ldab] = A(i,j) for max(0,j-ku) <= i <= min(m-1,j+kl).
 *
 *                      On exit, details of the factorization: U is stored as an
 *                      upper triangular band matrix with kl+ku superdiagonals in
 *                      rows 0 to kl+ku, and the multipliers used during the
 *                      factorization are stored in rows kl+ku+1 to 2*kl+ku.
 * @param[in]     ldab  The leading dimension of the array AB. ldab >= 2*kl+ku+1.
 * @param[out]    ipiv  Integer array, dimension (min(m,n)).
 *                      The pivot indices; for 0 <= i < min(m,n), row i of the
 *                      matrix was interchanged with row ipiv[i]. 0-based indexing.
 * @param[out]    info
 *                           Exit status:
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 *                         - > 0: if info = i, U(i-1,i-1) is exactly zero. The factorization
 *                           has been completed, but the factor U is exactly
 *                           singular, and division by zero will occur if it is used
 *                           to solve a system of equations.
 * @par Further Details:
 * The band storage scheme is illustrated by the following example, when
 * m = n = 6, kl = 2, ku = 1:
 *
 * On entry (0-based indexing, rows 0 to 5, kl=2, ku=1, kv=kl+ku=3):
 *
 *     Row 0:  *    *    *   (fill-in storage)
 *     Row 1:  *    *   a02  a13  a24  a35   (U superdiagonals)
 *     Row 2:  *   a01  a12  a23  a34  a45   (U superdiagonals)
 *     Row 3: a00  a11  a22  a33  a44  a55   (diagonal)
 *     Row 4: a10  a21  a32  a43  a54   *    (L multipliers after factorization)
 *     Row 5: a20  a31  a42  a53   *    *    (L multipliers after factorization)
 *
 * On exit:
 *     Row 0:  *    *    *   u03  u14  u25   (U fill-in from pivoting)
 *     Row 1:  *    *   u02  u13  u24  u35   (U superdiagonals)
 *     Row 2:  *   u01  u12  u23  u34  u45   (U superdiagonals)
 *     Row 3: u00  u11  u22  u33  u44  u55   (U diagonal)
 *     Row 4: m10  m21  m32  m43  m54   *    (L multipliers)
 *     Row 5: m20  m31  m42  m53   *    *    (L multipliers)
 *
 * Array elements marked * are not used by the routine.
 */
void cgbtf2(
    const INT m,
    const INT n,
    const INT kl,
    const INT ku,
    c64* restrict AB,
    const INT ldab,
    INT* restrict ipiv,
    INT* info)
{
    const c64 ONE = CMPLXF(1.0f, 0.0f);
    const c64 ZERO = CMPLXF(0.0f, 0.0f);

    INT i, j, jp, ju, km, kv;

    /* kv is the number of superdiagonals in the factor U, allowing for fill-in */
    kv = ku + kl;

    /* Test the input parameters */
    *info = 0;
    if (m < 0) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (kl < 0) {
        *info = -3;
    } else if (ku < 0) {
        *info = -4;
    } else if (ldab < kl + kv + 1) {
        *info = -6;
    }
    if (*info != 0) {
        xerbla("CGBTF2", -(*info));
        return;
    }

    /* Quick return if possible */
    if (m == 0 || n == 0) {
        return;
    }

    /* Gaussian elimination with partial pivoting */

    /* Set fill-in elements in columns ku+1 to kv-1 to zero.
       0-based: columns ku+1 to kv-1 (if kv > ku+1)
       LAPACK 1-based: columns KU+2 to KV */
    for (j = ku + 1; j < kv && j < n; j++) {
        /*
         * 0-based: rows from kv-j to kl-1
         * LAPACK 1-based: I = KV-J+2 to KL, maps to I-1 = KV-J+1 to KL-1
         * In 0-based: row indices from (kv - j) to (kl - 1)
         */
        for (i = kv - j; i < kl; i++) {
            AB[i + j * ldab] = ZERO;
        }
    }

    /* ju is the index of the last column affected by the current stage
       of the factorization. 0-based, initialized to 0. */
    ju = 0;

    INT minmn = (m < n) ? m : n;

    for (j = 0; j < minmn; j++) {
        /* Set fill-in elements in column j+kv to zero */
        if (j + kv < n) {
            for (i = 0; i < kl; i++) {
                AB[i + (j + kv) * ldab] = ZERO;
            }
        }

        /* Find pivot and test for singularity. km is the number of
           subdiagonal elements in the current column. */
        km = (kl < m - 1 - j) ? kl : m - 1 - j;
        if (km < 0) km = 0;

        /*
         * Search for pivot in column j, rows kv to kv+km (0-based).
         * cblas_izamax returns 0-based index relative to start.
         * Diagonal is at row kv in band storage.
         */
        jp = cblas_icamax(km + 1, &AB[kv + j * ldab], 1);
        ipiv[j] = jp + j;  /* Convert to global 0-based row index */

        if (AB[kv + jp + j * ldab] != ZERO) {
            /* Update ju: index of last column affected by current stage
               0-based: ju = max(ju, min(j + ku + jp, n - 1)) */
            INT new_ju = j + ku + jp;
            if (new_ju > n - 1) new_ju = n - 1;
            if (new_ju > ju) ju = new_ju;

            /* Apply interchange to columns j to ju.
               Swap row kv+jp with row kv (in band storage).
               Stride is ldab-1 because we're moving along a row in column-major band storage. */
            if (jp != 0) {
                cblas_cswap(ju - j + 1, &AB[kv + jp + j * ldab], ldab - 1,
                            &AB[kv + j * ldab], ldab - 1);
            }

            if (km > 0) {
                /* Compute multipliers.
                   Scale elements in column j, rows kv+1 to kv+km (0-based) */
                const c64 scale = ONE / AB[kv + j * ldab];
                cblas_cscal(km, &scale, &AB[kv + 1 + j * ldab], 1);

                /* Update trailing submatrix within the band.
                   A[kv+1:kv+km, j+1:ju] -= A[kv+1:kv+km, j] * A[kv, j+1:ju]

                   Note: The row vector A[kv, j+1:ju] has stride ldab-1 in band storage.
                   The column vector A[kv+1:kv+km, j] has stride 1.
                   The submatrix A[kv+1:kv+km, j+1:ju] has leading dimension ldab-1. */
                if (ju > j) {
                    const c64 NEG_ONE = CMPLXF(-1.0f, 0.0f);
                    cblas_cgeru(CblasColMajor, km, ju - j, &NEG_ONE,
                               &AB[kv + 1 + j * ldab], 1,           /* column vector */
                               &AB[kv - 1 + (j + 1) * ldab], ldab - 1,  /* row vector at row kv-1 of next col */
                               &AB[kv + (j + 1) * ldab], ldab - 1); /* submatrix */
                }
            }
        } else {
            /* If pivot is zero, set INFO to the index of the pivot
               unless a zero pivot has already been found.
               Report 1-based index for compatibility. */
            if (*info == 0) {
                *info = j + 1;
            }
        }
    }
}
