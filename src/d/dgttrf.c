/**
 * @file dgttrf.c
 * @brief DGTTRF computes an LU factorization of a real tridiagonal matrix.
 */

#include <math.h>
#include "semicolon_lapack_double.h"

/**
 * DGTTRF computes an LU factorization of a real tridiagonal matrix A
 * using elimination with partial pivoting and row interchanges.
 *
 * The factorization has the form
 *    A = L * U
 * where L is a product of permutation and unit lower bidiagonal
 * matrices and U is upper triangular with nonzeros in only the main
 * diagonal and first two superdiagonals.
 *
 * @param[in]     n     The order of the matrix A. n >= 0.
 * @param[in,out] DL    On entry, the (n-1) sub-diagonal elements of A.
 *                      On exit, the (n-1) multipliers that define the matrix L
 *                      from the LU factorization of A.
 *                      Array of dimension (n-1).
 * @param[in,out] D     On entry, the diagonal elements of A.
 *                      On exit, the n diagonal elements of the upper triangular
 *                      matrix U from the LU factorization of A.
 *                      Array of dimension (n).
 * @param[in,out] DU    On entry, the (n-1) super-diagonal elements of A.
 *                      On exit, the (n-1) elements of the first super-diagonal of U.
 *                      Array of dimension (n-1).
 * @param[out]    DU2   On exit, the (n-2) elements of the second super-diagonal of U.
 *                      Array of dimension (n-2).
 * @param[out]    ipiv  The pivot indices; for 0 <= i < n, row i of the matrix was
 *                      interchanged with row ipiv[i]. ipiv[i] will always be either
 *                      i or i+1; ipiv[i] = i indicates a row interchange was not
 *                      required.
 *                      Array of dimension (n).
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -k, the k-th argument had an illegal value
 *                         - > 0: if info = k, U(k-1,k-1) is exactly zero (0-based).
 *                           The factorization has been completed, but the factor U
 *                           is exactly singular, and division by zero will occur
 *                           if it is used to solve a system of equations.
 */
void dgttrf(
    const int n,
    double * const restrict DL,
    double * const restrict D,
    double * const restrict DU,
    double * const restrict DU2,
    int * const restrict ipiv,
    int *info)
{
    const double ZERO = 0.0;
    int i;
    double fact, temp;

    *info = 0;
    if (n < 0) {
        *info = -1;
        xerbla("DGTTRF", -(*info));
        return;
    }

    /* Quick return if possible */
    if (n == 0) {
        return;
    }

    /* Initialize ipiv[i] = i and DU2[i] = 0 */
    for (i = 0; i < n; i++) {
        ipiv[i] = i;
    }
    for (i = 0; i < n - 2; i++) {
        DU2[i] = ZERO;
    }

    /* Main elimination loop for columns 0 to n-3 */
    for (i = 0; i < n - 2; i++) {
        if (fabs(D[i]) >= fabs(DL[i])) {
            /* No row interchange required, eliminate DL[i] */
            if (D[i] != ZERO) {
                fact = DL[i] / D[i];
                DL[i] = fact;
                D[i + 1] = D[i + 1] - fact * DU[i];
            }
        } else {
            /* Interchange rows i and i+1, eliminate DL[i] */
            fact = D[i] / DL[i];
            D[i] = DL[i];
            DL[i] = fact;
            temp = DU[i];
            DU[i] = D[i + 1];
            D[i + 1] = temp - fact * D[i + 1];
            DU2[i] = DU[i + 1];
            DU[i + 1] = -fact * DU[i + 1];
            ipiv[i] = i + 1;
        }
    }

    /* Handle the last row (n-1) if n > 1 */
    if (n > 1) {
        i = n - 2;  /* Last pair of rows: i = n-2, i+1 = n-1 */
        if (fabs(D[i]) >= fabs(DL[i])) {
            if (D[i] != ZERO) {
                fact = DL[i] / D[i];
                DL[i] = fact;
                D[i + 1] = D[i + 1] - fact * DU[i];
            }
        } else {
            fact = D[i] / DL[i];
            D[i] = DL[i];
            DL[i] = fact;
            temp = DU[i];
            DU[i] = D[i + 1];
            D[i + 1] = temp - fact * D[i + 1];
            ipiv[i] = i + 1;
        }
    }

    /* Check for a zero on the diagonal of U */
    for (i = 0; i < n; i++) {
        if (D[i] == ZERO) {
            *info = i + 1;  /* 1-based for compatibility with LAPACK convention */
            break;
        }
    }
}
