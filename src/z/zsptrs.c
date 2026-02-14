/**
 * @file zsptrs.c
 * @brief ZSPTRS solves a system of linear equations with a symmetric matrix
 *        stored in packed format using the factorization from ZSPTRF.
 */

#include <complex.h>
#include <cblas.h>
#include "semicolon_lapack_complex_double.h"

/**
 * ZSPTRS solves a system of linear equations A*X = B with a complex symmetric
 * matrix A stored in packed format using the factorization A = U*D*U**T or
 * A = L*D*L**T computed by ZSPTRF.
 *
 * @param[in]     uplo   Specifies whether the details of the factorization are
 *                       stored as an upper or lower triangular matrix:
 *                       - = 'U': Upper triangular, form is A = U*D*U**T
 *                       - = 'L': Lower triangular, form is A = L*D*L**T
 * @param[in]     n      The order of the matrix A. n >= 0.
 * @param[in]     nrhs   The number of right hand sides, i.e., the number of
 *                       columns of the matrix B. nrhs >= 0.
 * @param[in]     AP     The block diagonal matrix D and the multipliers used
 *                       to obtain the factor U or L as computed by ZSPTRF,
 *                       stored as a packed triangular matrix of dimension n*(n+1)/2.
 * @param[in]     ipiv   Details of the interchanges and the block structure of D
 *                       as determined by ZSPTRF. Array of dimension n.
 * @param[in,out] B      On entry, the right hand side matrix B of dimension (ldb, nrhs).
 *                       On exit, the solution matrix X.
 * @param[in]     ldb    The leading dimension of the array B. ldb >= max(1,n).
 * @param[out]    info
 *                           Exit status:
 *                           - = 0: successful exit
 *                           - < 0: if info = -i, the i-th argument had an illegal value
 */
void zsptrs(
    const char* uplo,
    const int n,
    const int nrhs,
    const double complex* const restrict AP,
    const int* const restrict ipiv,
    double complex* const restrict B,
    const int ldb,
    int* info)
{
    const double complex ONE = CMPLX(1.0, 0.0);
    const double complex NEG_ONE = CMPLX(-1.0, 0.0);

    int upper;
    int j, k, kc, kp;
    double complex ak, akm1, akm1k, bk, bkm1, denom;

    *info = 0;
    upper = (uplo[0] == 'U' || uplo[0] == 'u');
    if (!upper && !(uplo[0] == 'L' || uplo[0] == 'l')) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (nrhs < 0) {
        *info = -3;
    } else if (ldb < (1 > n ? 1 : n)) {
        *info = -7;
    }
    if (*info != 0) {
        xerbla("ZSPTRS", -(*info));
        return;
    }

    if (n == 0 || nrhs == 0)
        return;

    if (upper) {
        /*
         * Solve A*X = B, where A = U*D*U**T.
         *
         * First solve U*D*X = B, overwriting B with X.
         *
         * K is the main loop index, decreasing from N-1 to 0 in steps of
         * 1 or 2, depending on the size of the diagonal blocks.
         */
        k = n - 1;
        kc = n * (n + 1) / 2;

        while (k >= 0) {
            kc = kc - (k + 1);
            if (ipiv[k] >= 0) {
                /* 1 x 1 diagonal block
                 * Interchange rows K and IPIV(K). */
                kp = ipiv[k];
                if (kp != k)
                    cblas_zswap(nrhs, &B[k], ldb, &B[kp], ldb);

                /* Multiply by inv(U(K)), where U(K) is the transformation
                 * stored in column K of A. */
                if (k > 0)
                    cblas_zgeru(CblasColMajor, k, nrhs, &NEG_ONE, &AP[kc], 1, &B[k], ldb, B, ldb);

                /* Multiply by the inverse of the diagonal block. */
                {
                    double complex tmp = ONE / AP[kc + k];
                    cblas_zscal(nrhs, &tmp, &B[k], ldb);
                }
                k = k - 1;
            } else {
                /* 2 x 2 diagonal block
                 * Interchange rows K-1 and -IPIV(K). */
                kp = -ipiv[k] - 1;
                if (kp != k - 1)
                    cblas_zswap(nrhs, &B[k - 1], ldb, &B[kp], ldb);

                /* Multiply by inv(U(K)), where U(K) is the transformation
                 * stored in columns K-1 and K of A. */
                if (k > 1) {
                    cblas_zgeru(CblasColMajor, k - 1, nrhs, &NEG_ONE, &AP[kc], 1, &B[k], ldb, B, ldb);
                    cblas_zgeru(CblasColMajor, k - 1, nrhs, &NEG_ONE, &AP[kc - k], 1, &B[k - 1], ldb, B, ldb);
                }

                /* Multiply by the inverse of the diagonal block. */
                akm1k = AP[kc + k - 1];
                akm1 = AP[kc - 1] / akm1k;
                ak = AP[kc + k] / akm1k;
                denom = akm1 * ak - ONE;
                for (j = 0; j < nrhs; j++) {
                    bkm1 = B[(k - 1) + j * ldb] / akm1k;
                    bk = B[k + j * ldb] / akm1k;
                    B[(k - 1) + j * ldb] = (ak * bkm1 - bk) / denom;
                    B[k + j * ldb] = (akm1 * bk - bkm1) / denom;
                }
                kc = kc - k;
                k = k - 2;
            }
        }

        /*
         * Next solve U**T*X = B, overwriting B with X.
         *
         * K is the main loop index, increasing from 0 to N-1 in steps of
         * 1 or 2, depending on the size of the diagonal blocks.
         */
        k = 0;
        kc = 0;

        while (k < n) {
            if (ipiv[k] >= 0) {
                /* 1 x 1 diagonal block
                 * Multiply by inv(U**T(K)), where U(K) is the transformation
                 * stored in column K of A. */
                if (k > 0)
                    cblas_zgemv(CblasColMajor, CblasTrans, k, nrhs, &NEG_ONE, B, ldb, &AP[kc], 1, &ONE, &B[k], ldb);

                /* Interchange rows K and IPIV(K). */
                kp = ipiv[k];
                if (kp != k)
                    cblas_zswap(nrhs, &B[k], ldb, &B[kp], ldb);
                kc = kc + k + 1;
                k = k + 1;
            } else {
                /* 2 x 2 diagonal block
                 * Multiply by inv(U**T(K+1)), where U(K+1) is the transformation
                 * stored in columns K and K+1 of A. */
                if (k > 0) {
                    cblas_zgemv(CblasColMajor, CblasTrans, k, nrhs, &NEG_ONE, B, ldb, &AP[kc], 1, &ONE, &B[k], ldb);
                    cblas_zgemv(CblasColMajor, CblasTrans, k, nrhs, &NEG_ONE, B, ldb, &AP[kc + k + 1], 1, &ONE, &B[k + 1], ldb);
                }

                /* Interchange rows K and -IPIV(K). */
                kp = -ipiv[k] - 1;
                if (kp != k)
                    cblas_zswap(nrhs, &B[k], ldb, &B[kp], ldb);
                kc = kc + 2 * (k + 1) + 1;
                k = k + 2;
            }
        }
    } else {
        /*
         * Solve A*X = B, where A = L*D*L**T.
         *
         * First solve L*D*X = B, overwriting B with X.
         *
         * K is the main loop index, increasing from 0 to N-1 in steps of
         * 1 or 2, depending on the size of the diagonal blocks.
         */
        k = 0;
        kc = 0;

        while (k < n) {
            if (ipiv[k] >= 0) {
                /* 1 x 1 diagonal block
                 * Interchange rows K and IPIV(K). */
                kp = ipiv[k];
                if (kp != k)
                    cblas_zswap(nrhs, &B[k], ldb, &B[kp], ldb);

                /* Multiply by inv(L(K)), where L(K) is the transformation
                 * stored in column K of A. */
                if (k < n - 1)
                    cblas_zgeru(CblasColMajor, n - k - 1, nrhs, &NEG_ONE, &AP[kc + 1], 1, &B[k], ldb, &B[k + 1], ldb);

                /* Multiply by the inverse of the diagonal block. */
                {
                    double complex tmp = ONE / AP[kc];
                    cblas_zscal(nrhs, &tmp, &B[k], ldb);
                }
                kc = kc + n - k;
                k = k + 1;
            } else {
                /* 2 x 2 diagonal block
                 * Interchange rows K+1 and -IPIV(K). */
                kp = -ipiv[k] - 1;
                if (kp != k + 1)
                    cblas_zswap(nrhs, &B[k + 1], ldb, &B[kp], ldb);

                /* Multiply by inv(L(K)), where L(K) is the transformation
                 * stored in columns K and K+1 of A. */
                if (k < n - 2) {
                    cblas_zgeru(CblasColMajor, n - k - 2, nrhs, &NEG_ONE, &AP[kc + 2], 1, &B[k], ldb, &B[k + 2], ldb);
                    cblas_zgeru(CblasColMajor, n - k - 2, nrhs, &NEG_ONE, &AP[kc + n - k + 1], 1, &B[k + 1], ldb, &B[k + 2], ldb);
                }

                /* Multiply by the inverse of the diagonal block. */
                akm1k = AP[kc + 1];
                akm1 = AP[kc] / akm1k;
                ak = AP[kc + n - k] / akm1k;
                denom = akm1 * ak - ONE;
                for (j = 0; j < nrhs; j++) {
                    bkm1 = B[k + j * ldb] / akm1k;
                    bk = B[(k + 1) + j * ldb] / akm1k;
                    B[k + j * ldb] = (ak * bkm1 - bk) / denom;
                    B[(k + 1) + j * ldb] = (akm1 * bk - bkm1) / denom;
                }
                kc = kc + 2 * (n - k) - 1;
                k = k + 2;
            }
        }

        /*
         * Next solve L**T*X = B, overwriting B with X.
         *
         * K is the main loop index, decreasing from N-1 to 0 in steps of
         * 1 or 2, depending on the size of the diagonal blocks.
         */
        k = n - 1;
        kc = n * (n + 1) / 2;

        while (k >= 0) {
            kc = kc - (n - k);
            if (ipiv[k] >= 0) {
                /* 1 x 1 diagonal block
                 * Multiply by inv(L**T(K)), where L(K) is the transformation
                 * stored in column K of A. */
                if (k < n - 1)
                    cblas_zgemv(CblasColMajor, CblasTrans, n - k - 1, nrhs, &NEG_ONE, &B[k + 1], ldb, &AP[kc + 1], 1, &ONE, &B[k], ldb);

                /* Interchange rows K and IPIV(K). */
                kp = ipiv[k];
                if (kp != k)
                    cblas_zswap(nrhs, &B[k], ldb, &B[kp], ldb);
                k = k - 1;
            } else {
                /* 2 x 2 diagonal block
                 * Multiply by inv(L**T(K-1)), where L(K-1) is the transformation
                 * stored in columns K-1 and K of A. */
                if (k < n - 1) {
                    cblas_zgemv(CblasColMajor, CblasTrans, n - k - 1, nrhs, &NEG_ONE, &B[k + 1], ldb, &AP[kc + 1], 1, &ONE, &B[k], ldb);
                    cblas_zgemv(CblasColMajor, CblasTrans, n - k - 1, nrhs, &NEG_ONE, &B[k + 1], ldb, &AP[kc - (n - k) + 1], 1, &ONE, &B[k - 1], ldb);
                }

                /* Interchange rows K and -IPIV(K). */
                kp = -ipiv[k] - 1;
                if (kp != k)
                    cblas_zswap(nrhs, &B[k], ldb, &B[kp], ldb);
                kc = kc - (n - k + 1);
                k = k - 2;
            }
        }
    }
}
