/**
 * @file dgbtrs.c
 * @brief DGBTRS solves a system of linear equations with a general band
 *        matrix using the LU factorization computed by DGBTRF.
 */

#include "semicolon_cblas.h"
#include "semicolon_lapack_double.h"

/**
 * DGBTRS solves a system of linear equations
 *    A * X = B  or  A**T * X = B
 * with a general band matrix A using the LU factorization computed
 * by DGBTRF.
 *
 * @param[in]     trans   Specifies the form of the system of equations:
 *                        = 'N': A * X = B  (No transpose)
 *                        = 'T': A**T * X = B  (Transpose)
 *                        = 'C': A**T * X = B  (Conjugate transpose = Transpose)
 * @param[in]     n       The order of the matrix A. n >= 0.
 * @param[in]     kl      The number of subdiagonals within the band of A. kl >= 0.
 * @param[in]     ku      The number of superdiagonals within the band of A. ku >= 0.
 * @param[in]     nrhs    The number of right hand sides, i.e., the number of
 *                        columns of the matrix B. nrhs >= 0.
 * @param[in]     AB      Double precision array, dimension (ldab, n).
 *                        Details of the LU factorization of the band matrix A,
 *                        as computed by DGBTRF. U is stored as an upper triangular
 *                        band matrix with kl+ku superdiagonals in rows 0 to kl+ku,
 *                        and the multipliers used during the factorization are
 *                        stored in rows kl+ku+1 to 2*kl+ku.
 * @param[in]     ldab    The leading dimension of the array AB. ldab >= 2*kl+ku+1.
 * @param[in]     ipiv    Integer array, dimension (n).
 *                        The pivot indices; for 0 <= i < n, row i of the matrix
 *                        was interchanged with row ipiv[i]. 0-based indexing.
 * @param[in,out] B       Double precision array, dimension (ldb, nrhs).
 *                        On entry, the right hand side matrix B.
 *                        On exit, the solution matrix X.
 * @param[in]     ldb     The leading dimension of the array B. ldb >= max(1,n).
 * @param[out]    info
 *                           Exit status:
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 */
void dgbtrs(
    const char* trans,
    const INT n,
    const INT kl,
    const INT ku,
    const INT nrhs,
    const f64* restrict AB,
    const INT ldab,
    const INT* restrict ipiv,
    f64* restrict B,
    const INT ldb,
    INT* info)
{
    const f64 ONE = 1.0;
    const f64 NEG_ONE = -1.0;

    INT notran, lnoti;
    INT i, j, kd, l, lm;

    /* Test the input parameters */
    *info = 0;
    notran = (trans[0] == 'N' || trans[0] == 'n');
    if (!notran && trans[0] != 'T' && trans[0] != 't' && trans[0] != 'C' && trans[0] != 'c') {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (kl < 0) {
        *info = -3;
    } else if (ku < 0) {
        *info = -4;
    } else if (nrhs < 0) {
        *info = -5;
    } else if (ldab < 2 * kl + ku + 1) {
        *info = -7;
    } else if (ldb < (n > 1 ? n : 1)) {
        *info = -10;
    }
    if (*info != 0) {
        xerbla("DGBTRS", -(*info));
        return;
    }

    /* Quick return if possible */
    if (n == 0 || nrhs == 0) {
        return;
    }

    /* kd is the row index of the diagonal in band storage (0-based) */
    kd = ku + kl;
    lnoti = (kl > 0);

    if (notran) {
        /* Solve A*X = B.
         *
         * Solve L*X = B, overwriting B with X.
         *
         * L is represented as a product of permutations and unit lower
         * triangular matrices L = P(0) * L(0) * ... * P(n-2) * L(n-2),
         * where each transformation L(i) is a rank-one modification of
         * the identity matrix.
         */
        if (lnoti) {
            for (j = 0; j < n - 1; j++) {
                lm = (kl < n - 1 - j) ? kl : n - 1 - j;
                l = ipiv[j];
                if (l != j) {
                    cblas_dswap(nrhs, &B[l], ldb, &B[j], ldb);
                }
                /* B[j+1:j+lm, :] -= AB[kd+1:kd+lm, j] * B[j, :]
                 * This is a rank-1 update: outer product of column vector and row */
                cblas_dger(CblasColMajor, lm, nrhs, NEG_ONE,
                           &AB[kd + 1 + j * ldab], 1,
                           &B[j], ldb,
                           &B[j + 1], ldb);
            }
        }

        /* Solve U*X = B, overwriting B with X.
         * Process each right-hand side column */
        for (i = 0; i < nrhs; i++) {
            cblas_dtbsv(CblasColMajor, CblasUpper, CblasNoTrans, CblasNonUnit,
                        n, kl + ku, AB, ldab, &B[i * ldb], 1);
        }
    } else {
        /* Solve A**T * X = B.
         *
         * Solve U**T * X = B, overwriting B with X.
         * Process each right-hand side column */
        for (i = 0; i < nrhs; i++) {
            cblas_dtbsv(CblasColMajor, CblasUpper, CblasTrans, CblasNonUnit,
                        n, kl + ku, AB, ldab, &B[i * ldb], 1);
        }

        /* Solve L**T * X = B, overwriting B with X. */
        if (lnoti) {
            for (j = n - 2; j >= 0; j--) {
                lm = (kl < n - 1 - j) ? kl : n - 1 - j;
                /* B[j, :] -= AB[kd+1:kd+lm, j]^T * B[j+1:j+lm, :]
                 * This is a GEMV: row update from multiplying column by submatrix */
                cblas_dgemv(CblasColMajor, CblasTrans, lm, nrhs, NEG_ONE,
                            &B[j + 1], ldb,
                            &AB[kd + 1 + j * ldab], 1,
                            ONE, &B[j], ldb);
                l = ipiv[j];
                if (l != j) {
                    cblas_dswap(nrhs, &B[l], ldb, &B[j], ldb);
                }
            }
        }
    }
}
