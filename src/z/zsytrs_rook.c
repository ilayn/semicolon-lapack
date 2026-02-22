/**
 * @file zsytrs_rook.c
 * @brief ZSYTRS_ROOK solves a system of linear equations A*X = B with a complex symmetric matrix A using the factorization computed by ZSYTRF_ROOK.
 */

#include "semicolon_lapack_complex_double.h"
#include <complex.h>
#include "semicolon_cblas.h"

/**
 * ZSYTRS_ROOK solves a system of linear equations A*X = B with
 * a complex symmetric matrix A using the factorization A = U*D*U**T or
 * A = L*D*L**T computed by ZSYTRF_ROOK.
 *
 * @param[in] uplo
 *          Specifies whether the details of the factorization are stored
 *          as an upper or lower triangular matrix.
 *          = 'U':  Upper triangular, form is A = U*D*U**T;
 *          = 'L':  Lower triangular, form is A = L*D*L**T.
 *
 * @param[in] n
 *          The order of the matrix A. n >= 0.
 *
 * @param[in] nrhs
 *          The number of right hand sides, i.e., the number of columns
 *          of the matrix B. nrhs >= 0.
 *
 * @param[in] A
 *          Double complex array, dimension (lda, n).
 *          The block diagonal matrix D and the multipliers used to
 *          obtain the factor U or L as computed by ZSYTRF_ROOK.
 *
 * @param[in] lda
 *          The leading dimension of the array A. lda >= max(1, n).
 *
 * @param[in] ipiv
 *          Integer array, dimension (n).
 *          Details of the interchanges and the block structure of D
 *          as determined by ZSYTRF_ROOK.
 *
 * @param[in,out] B
 *          Double complex array, dimension (ldb, nrhs).
 *          On entry, the right hand side matrix B.
 *          On exit, the solution matrix X.
 *
 * @param[in] ldb
 *          The leading dimension of the array B. ldb >= max(1, n).
 *
 * @param[out] info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 */
void zsytrs_rook(
    const char* uplo,
    const INT n,
    const INT nrhs,
    const c128* restrict A,
    const INT lda,
    const INT* restrict ipiv,
    c128* restrict B,
    const INT ldb,
    INT* info)
{
    const c128 CONE = CMPLX(1.0, 0.0);
    const c128 NEG_CONE = CMPLX(-1.0, 0.0);

    INT upper;
    INT j, k, kp;
    c128 ak, akm1, akm1k, bk, bkm1, denom;

    *info = 0;
    upper = (uplo[0] == 'U' || uplo[0] == 'u');

    if (!upper && !(uplo[0] == 'L' || uplo[0] == 'l')) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (nrhs < 0) {
        *info = -3;
    } else if (lda < (1 > n ? 1 : n)) {
        *info = -5;
    } else if (ldb < (1 > n ? 1 : n)) {
        *info = -8;
    }
    if (*info != 0) {
        xerbla("ZSYTRS_ROOK", -(*info));
        return;
    }

    if (n == 0 || nrhs == 0) {
        return;
    }

    if (upper) {

        k = n - 1;
        while (k >= 0) {

            if (ipiv[k] >= 0) {

                kp = ipiv[k];
                if (kp != k) {
                    cblas_zswap(nrhs, &B[k], ldb, &B[kp], ldb);
                }

                cblas_zgeru(CblasColMajor, k, nrhs, &NEG_CONE, &A[0 + k * lda], 1,
                            &B[k], ldb, &B[0], ldb);

                c128 s = CONE / A[k + k * lda];
                cblas_zscal(nrhs, &s, &B[k], ldb);
                k = k - 1;

            } else {

                kp = -ipiv[k] - 1;
                if (kp != k) {
                    cblas_zswap(nrhs, &B[k], ldb, &B[kp], ldb);
                }

                kp = -ipiv[k - 1] - 1;
                if (kp != k - 1) {
                    cblas_zswap(nrhs, &B[k - 1], ldb, &B[kp], ldb);
                }

                if (k > 1) {
                    cblas_zgeru(CblasColMajor, k - 1, nrhs, &NEG_CONE, &A[0 + k * lda], 1,
                                &B[k], ldb, &B[0], ldb);
                    cblas_zgeru(CblasColMajor, k - 1, nrhs, &NEG_CONE, &A[0 + (k - 1) * lda], 1,
                                &B[k - 1], ldb, &B[0], ldb);
                }

                akm1k = A[k - 1 + k * lda];
                akm1 = A[k - 1 + (k - 1) * lda] / akm1k;
                ak = A[k + k * lda] / akm1k;
                denom = akm1 * ak - CONE;
                for (j = 0; j < nrhs; j++) {
                    bkm1 = B[k - 1 + j * ldb] / akm1k;
                    bk = B[k + j * ldb] / akm1k;
                    B[k - 1 + j * ldb] = (ak * bkm1 - bk) / denom;
                    B[k + j * ldb] = (akm1 * bk - bkm1) / denom;
                }
                k = k - 2;
            }
        }

        k = 0;
        while (k < n) {

            if (ipiv[k] >= 0) {

                if (k > 0) {
                    cblas_zgemv(CblasColMajor, CblasTrans, k, nrhs, &NEG_CONE, B,
                                ldb, &A[0 + k * lda], 1, &CONE, &B[k], ldb);
                }

                kp = ipiv[k];
                if (kp != k) {
                    cblas_zswap(nrhs, &B[k], ldb, &B[kp], ldb);
                }
                k = k + 1;

            } else {

                if (k > 0) {
                    cblas_zgemv(CblasColMajor, CblasTrans, k, nrhs, &NEG_CONE, B,
                                ldb, &A[0 + k * lda], 1, &CONE, &B[k], ldb);
                    cblas_zgemv(CblasColMajor, CblasTrans, k, nrhs, &NEG_CONE, B,
                                ldb, &A[0 + (k + 1) * lda], 1, &CONE, &B[k + 1], ldb);
                }

                kp = -ipiv[k] - 1;
                if (kp != k) {
                    cblas_zswap(nrhs, &B[k], ldb, &B[kp], ldb);
                }

                kp = -ipiv[k + 1] - 1;
                if (kp != k + 1) {
                    cblas_zswap(nrhs, &B[k + 1], ldb, &B[kp], ldb);
                }

                k = k + 2;
            }
        }

    } else {

        k = 0;
        while (k < n) {

            if (ipiv[k] >= 0) {

                kp = ipiv[k];
                if (kp != k) {
                    cblas_zswap(nrhs, &B[k], ldb, &B[kp], ldb);
                }

                if (k < n - 1) {
                    cblas_zgeru(CblasColMajor, n - k - 1, nrhs, &NEG_CONE, &A[k + 1 + k * lda], 1,
                                &B[k], ldb, &B[k + 1], ldb);
                }

                c128 s = CONE / A[k + k * lda];
                cblas_zscal(nrhs, &s, &B[k], ldb);
                k = k + 1;

            } else {

                kp = -ipiv[k] - 1;
                if (kp != k) {
                    cblas_zswap(nrhs, &B[k], ldb, &B[kp], ldb);
                }

                kp = -ipiv[k + 1] - 1;
                if (kp != k + 1) {
                    cblas_zswap(nrhs, &B[k + 1], ldb, &B[kp], ldb);
                }

                if (k < n - 2) {
                    cblas_zgeru(CblasColMajor, n - k - 2, nrhs, &NEG_CONE, &A[k + 2 + k * lda], 1,
                                &B[k], ldb, &B[k + 2], ldb);
                    cblas_zgeru(CblasColMajor, n - k - 2, nrhs, &NEG_CONE, &A[k + 2 + (k + 1) * lda], 1,
                                &B[k + 1], ldb, &B[k + 2], ldb);
                }

                akm1k = A[k + 1 + k * lda];
                akm1 = A[k + k * lda] / akm1k;
                ak = A[k + 1 + (k + 1) * lda] / akm1k;
                denom = akm1 * ak - CONE;
                for (j = 0; j < nrhs; j++) {
                    bkm1 = B[k + j * ldb] / akm1k;
                    bk = B[k + 1 + j * ldb] / akm1k;
                    B[k + j * ldb] = (ak * bkm1 - bk) / denom;
                    B[k + 1 + j * ldb] = (akm1 * bk - bkm1) / denom;
                }
                k = k + 2;
            }
        }

        k = n - 1;
        while (k >= 0) {

            if (ipiv[k] >= 0) {

                if (k < n - 1) {
                    cblas_zgemv(CblasColMajor, CblasTrans, n - k - 1, nrhs, &NEG_CONE,
                                &B[k + 1], ldb, &A[k + 1 + k * lda], 1, &CONE, &B[k], ldb);
                }

                kp = ipiv[k];
                if (kp != k) {
                    cblas_zswap(nrhs, &B[k], ldb, &B[kp], ldb);
                }
                k = k - 1;

            } else {

                if (k < n - 1) {
                    cblas_zgemv(CblasColMajor, CblasTrans, n - k - 1, nrhs, &NEG_CONE,
                                &B[k + 1], ldb, &A[k + 1 + k * lda], 1, &CONE, &B[k], ldb);
                    cblas_zgemv(CblasColMajor, CblasTrans, n - k - 1, nrhs, &NEG_CONE,
                                &B[k + 1], ldb, &A[k + 1 + (k - 1) * lda], 1, &CONE, &B[k - 1], ldb);
                }

                kp = -ipiv[k] - 1;
                if (kp != k) {
                    cblas_zswap(nrhs, &B[k], ldb, &B[kp], ldb);
                }

                kp = -ipiv[k - 1] - 1;
                if (kp != k - 1) {
                    cblas_zswap(nrhs, &B[k - 1], ldb, &B[kp], ldb);
                }

                k = k - 2;
            }
        }
    }
}
