/**
 * @file cgbtrs.c
 * @brief CGBTRS solves a system of linear equations with a general band
 *        matrix using the LU factorization computed by CGBTRF.
 */

#include <complex.h>
#include <cblas.h>
#include "semicolon_lapack_complex_single.h"

/**
 * CGBTRS solves a system of linear equations
 *    A * X = B,  A**T * X = B,  or  A**H * X = B
 * with a general band matrix A using the LU factorization computed
 * by CGBTRF.
 *
 * @param[in]     trans   Specifies the form of the system of equations:
 *                        = 'N': A * X = B     (No transpose)
 *                        = 'T': A**T * X = B  (Transpose)
 *                        = 'C': A**H * X = B  (Conjugate transpose)
 * @param[in]     n       The order of the matrix A. n >= 0.
 * @param[in]     kl      The number of subdiagonals within the band of A. kl >= 0.
 * @param[in]     ku      The number of superdiagonals within the band of A. ku >= 0.
 * @param[in]     nrhs    The number of right hand sides, i.e., the number of
 *                        columns of the matrix B. nrhs >= 0.
 * @param[in]     AB      Complex*16 array, dimension (ldab, n).
 *                        Details of the LU factorization of the band matrix A,
 *                        as computed by CGBTRF. U is stored as an upper triangular
 *                        band matrix with kl+ku superdiagonals in rows 0 to kl+ku,
 *                        and the multipliers used during the factorization are
 *                        stored in rows kl+ku+1 to 2*kl+ku.
 * @param[in]     ldab    The leading dimension of the array AB. ldab >= 2*kl+ku+1.
 * @param[in]     ipiv    Integer array, dimension (n).
 *                        The pivot indices; for 0 <= i < n, row i of the matrix
 *                        was interchanged with row ipiv[i]. 0-based indexing.
 * @param[in,out] B       Complex*16 array, dimension (ldb, nrhs).
 *                        On entry, the right hand side matrix B.
 *                        On exit, the solution matrix X.
 * @param[in]     ldb     The leading dimension of the array B. ldb >= max(1,n).
 * @param[out]    info
 *                           Exit status:
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 */
void cgbtrs(
    const char* trans,
    const int n,
    const int kl,
    const int ku,
    const int nrhs,
    const c64* restrict AB,
    const int ldab,
    const int* restrict ipiv,
    c64* restrict B,
    const int ldb,
    int* info)
{
    const c64 ONE = CMPLXF(1.0f, 0.0f);
    const c64 NEG_ONE = CMPLXF(-1.0f, 0.0f);

    int notran, lnoti;
    int i, j, kd, l, lm;

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
        xerbla("CGBTRS", -(*info));
        return;
    }

    if (n == 0 || nrhs == 0) {
        return;
    }

    kd = ku + kl;
    lnoti = (kl > 0);

    if (notran) {
        /*
         * Solve  A*X = B.
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
                    cblas_cswap(nrhs, &B[l], ldb, &B[j], ldb);
                }
                cblas_cgeru(CblasColMajor, lm, nrhs, &NEG_ONE,
                            &AB[kd + 1 + j * ldab], 1,
                            &B[j], ldb,
                            &B[j + 1], ldb);
            }
        }

        for (i = 0; i < nrhs; i++) {
            cblas_ctbsv(CblasColMajor, CblasUpper, CblasNoTrans, CblasNonUnit,
                        n, kl + ku, AB, ldab, &B[i * ldb], 1);
        }

    } else if (trans[0] == 'T' || trans[0] == 't') {
        /*
         * Solve A**T * X = B.
         */
        for (i = 0; i < nrhs; i++) {
            cblas_ctbsv(CblasColMajor, CblasUpper, CblasTrans, CblasNonUnit,
                        n, kl + ku, AB, ldab, &B[i * ldb], 1);
        }

        if (lnoti) {
            for (j = n - 2; j >= 0; j--) {
                lm = (kl < n - 1 - j) ? kl : n - 1 - j;
                cblas_cgemv(CblasColMajor, CblasTrans, lm, nrhs, &NEG_ONE,
                            &B[j + 1], ldb,
                            &AB[kd + 1 + j * ldab], 1,
                            &ONE, &B[j], ldb);
                l = ipiv[j];
                if (l != j) {
                    cblas_cswap(nrhs, &B[l], ldb, &B[j], ldb);
                }
            }
        }

    } else {
        /*
         * Solve A**H * X = B.
         */
        for (i = 0; i < nrhs; i++) {
            cblas_ctbsv(CblasColMajor, CblasUpper, CblasConjTrans, CblasNonUnit,
                        n, kl + ku, AB, ldab, &B[i * ldb], 1);
        }

        if (lnoti) {
            for (j = n - 2; j >= 0; j--) {
                lm = (kl < n - 1 - j) ? kl : n - 1 - j;
                clacgv(nrhs, &B[j], ldb);
                cblas_cgemv(CblasColMajor, CblasConjTrans, lm, nrhs, &NEG_ONE,
                            &B[j + 1], ldb,
                            &AB[kd + 1 + j * ldab], 1,
                            &ONE, &B[j], ldb);
                clacgv(nrhs, &B[j], ldb);
                l = ipiv[j];
                if (l != j) {
                    cblas_cswap(nrhs, &B[l], ldb, &B[j], ldb);
                }
            }
        }
    }
}
