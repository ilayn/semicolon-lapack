#include "internal_build_defs.h"
#include "semicolon_lapack_single.h"
/**
 * @file sgbsv.c
 * @brief Solves a general banded system of linear equations A * X = B.
 */

/**
 * SGBSV computes the solution to a real system of linear equations
 *    A * X = B,
 * where A is a band matrix of order N with KL subdiagonals and KU
 * superdiagonals, and X and B are N-by-NRHS matrices.
 *
 * The LU decomposition with partial pivoting and row interchanges is
 * used to factor A as A = L * U, where L is a product of permutation
 * and unit lower triangular matrices with KL subdiagonals, and U is
 * upper triangular with KL+KU superdiagonals. The factored form of A
 * is then used to solve the system of equations A * X = B.
 *
 * @param[in]     n       The number of linear equations, i.e., the order of the
 *                        matrix A (n >= 0).
 * @param[in]     kl      The number of subdiagonals within the band of A (kl >= 0).
 * @param[in]     ku      The number of superdiagonals within the band of A (ku >= 0).
 * @param[in]     nrhs    The number of right hand sides, i.e., the number of
 *                        columns of the matrix B (nrhs >= 0).
 * @param[in,out] AB      On entry, the matrix A in band storage, in rows kl to
 *                        2*kl+ku; rows 0 to kl-1 of the array need not be set.
 *                        The j-th column of A is stored in the j-th column of
 *                        the array AB as follows:
 *                        AB[kl+ku+i-j + j*ldab] = A(i,j) for max(0,j-ku)<=i<=min(n-1,j+kl).
 *                        On exit, details of the factorization: U is stored as an
 *                        upper triangular band matrix with kl+ku superdiagonals in
 *                        rows 0 to kl+ku, and the multipliers used during the
 *                        factorization are stored in rows kl+ku+1 to 2*kl+ku.
 *                        Array of dimension (ldab, n).
 * @param[in]     ldab    The leading dimension of the array AB (ldab >= 2*kl+ku+1).
 * @param[out]    ipiv    The pivot indices that define the permutation matrix P;
 *                        row i of the matrix was interchanged with row ipiv[i].
 *                        Array of dimension n, 0-based.
 * @param[in,out] B       On entry, the N-by-NRHS right hand side matrix B.
 *                        On exit, if info = 0, the N-by-NRHS solution matrix X.
 *                        Array of dimension (ldb, nrhs).
 * @param[in]     ldb     The leading dimension of the array B (ldb >= max(1,n)).
 * @param[out]    info
 *                           Exit status:
 *                           - = 0: successful exit
 *                           - < 0: if info = -i, the i-th argument had an illegal value
 *                           - > 0: if info = i, U(i-1,i-1) is exactly zero. The factorization
 *                           has been completed, but the factor U is exactly
 *                           singular, and the solution has not been computed.
 */
void sgbsv(
    const INT n,
    const INT kl,
    const INT ku,
    const INT nrhs,
    f32* restrict AB,
    const INT ldab,
    INT* restrict ipiv,
    f32* restrict B,
    const INT ldb,
    INT* info)
{
    /* Test the input parameters */
    *info = 0;
    if (n < 0) {
        *info = -1;
    } else if (kl < 0) {
        *info = -2;
    } else if (ku < 0) {
        *info = -3;
    } else if (nrhs < 0) {
        *info = -4;
    } else if (ldab < 2 * kl + ku + 1) {
        *info = -6;
    } else if (ldb < (n > 1 ? n : 1)) {
        *info = -9;
    }

    if (*info != 0) {
        xerbla("SGBSV ", -(*info));
        return;
    }

    /* Compute the LU factorization of the band matrix A */
    sgbtrf(n, n, kl, ku, AB, ldab, ipiv, info);

    if (*info == 0) {
        /* Solve the system A*X = B, overwriting B with X */
        sgbtrs("N", n, kl, ku, nrhs, AB, ldab, ipiv, B, ldb, info);
    }
}
