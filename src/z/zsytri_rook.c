/**
 * @file zsytri_rook.c
 * @brief ZSYTRI_ROOK computes the inverse of a complex symmetric matrix using the factorization computed by ZSYTRF_ROOK.
 */

#include <complex.h>
#include <math.h>
#include <cblas.h>
#include "semicolon_lapack_complex_double.h"

/**
 * ZSYTRI_ROOK computes the inverse of a complex symmetric
 * matrix A using the factorization A = U*D*U**T or A = L*D*L**T
 * computed by ZSYTRF_ROOK.
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
 * @param[in,out] A
 *          Double complex array, dimension (lda, n).
 *          On entry, the block diagonal matrix D and the multipliers
 *          used to obtain the factor U or L as computed by ZSYTRF_ROOK.
 *          On exit, if info = 0, the (symmetric) inverse of the original
 *          matrix.
 *
 * @param[in] lda
 *          The leading dimension of the array A. lda >= max(1, n).
 *
 * @param[in] ipiv
 *          Integer array, dimension (n).
 *          Details of the interchanges and the block structure of D
 *          as determined by ZSYTRF_ROOK.
 *
 * @param[out] work
 *          Double complex array, dimension (n).
 *
 * @param[out] info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 *                         - > 0: if info = i, D(i,i) = 0; the matrix is singular and its
 *                           inverse could not be computed.
 */
void zsytri_rook(
    const char* uplo,
    const int n,
    c128* restrict A,
    const int lda,
    const int* restrict ipiv,
    c128* restrict work,
    int* info)
{
    const c128 ONE = CMPLX(1.0, 0.0);
    const c128 ZERO = CMPLX(0.0, 0.0);
    const c128 NEG_ONE = CMPLX(-1.0, 0.0);

    int upper;
    int k, kp, kstep;
    c128 ak, akkp1, akp1, d, t, temp;

    *info = 0;
    upper = (uplo[0] == 'U' || uplo[0] == 'u');

    if (!upper && !(uplo[0] == 'L' || uplo[0] == 'l')) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (lda < (1 > n ? 1 : n)) {
        *info = -4;
    }
    if (*info != 0) {
        xerbla("ZSYTRI_ROOK", -(*info));
        return;
    }

    if (n == 0) {
        return;
    }

    if (upper) {

        for (*info = n - 1; *info >= 0; (*info)--) {
            if (ipiv[*info] >= 0 && A[*info + (*info) * lda] == ZERO) {
                (*info)++;
                return;
            }
        }

    } else {

        for (*info = 0; *info < n; (*info)++) {
            if (ipiv[*info] >= 0 && A[*info + (*info) * lda] == ZERO) {
                (*info)++;
                return;
            }
        }
    }
    *info = 0;

    if (upper) {

        k = 0;
        while (k < n) {

            if (ipiv[k] >= 0) {

                A[k + k * lda] = ONE / A[k + k * lda];

                if (k > 0) {
                    c128 dotval;
                    cblas_zcopy(k, &A[0 + k * lda], 1, work, 1);
                    zsymv(uplo, k, NEG_ONE, A, lda,
                          work, 1, ZERO, &A[0 + k * lda], 1);
                    cblas_zdotu_sub(k, work, 1, &A[0 + k * lda], 1, &dotval);
                    A[k + k * lda] -= dotval;
                }
                kstep = 1;

            } else {

                t = A[k + (k + 1) * lda];
                ak = A[k + k * lda] / t;
                akp1 = A[k + 1 + (k + 1) * lda] / t;
                akkp1 = A[k + (k + 1) * lda] / t;
                d = t * (ak * akp1 - ONE);
                A[k + k * lda] = akp1 / d;
                A[k + 1 + (k + 1) * lda] = ak / d;
                A[k + (k + 1) * lda] = -akkp1 / d;

                if (k > 0) {
                    c128 dotval;
                    cblas_zcopy(k, &A[0 + k * lda], 1, work, 1);
                    zsymv(uplo, k, NEG_ONE, A, lda,
                          work, 1, ZERO, &A[0 + k * lda], 1);
                    cblas_zdotu_sub(k, work, 1, &A[0 + k * lda], 1, &dotval);
                    A[k + k * lda] -= dotval;
                    cblas_zdotu_sub(k, &A[0 + k * lda], 1, &A[0 + (k + 1) * lda], 1, &dotval);
                    A[k + (k + 1) * lda] -= dotval;
                    cblas_zcopy(k, &A[0 + (k + 1) * lda], 1, work, 1);
                    zsymv(uplo, k, NEG_ONE, A, lda,
                          work, 1, ZERO, &A[0 + (k + 1) * lda], 1);
                    cblas_zdotu_sub(k, work, 1, &A[0 + (k + 1) * lda], 1, &dotval);
                    A[k + 1 + (k + 1) * lda] -= dotval;
                }
                kstep = 2;
            }

            if (kstep == 1) {

                kp = ipiv[k];
                if (kp != k) {
                    if (kp > 0) {
                        cblas_zswap(kp, &A[0 + k * lda], 1, &A[0 + kp * lda], 1);
                    }
                    cblas_zswap(k - kp - 1, &A[kp + 1 + k * lda], 1, &A[kp + (kp + 1) * lda], lda);
                    temp = A[k + k * lda];
                    A[k + k * lda] = A[kp + kp * lda];
                    A[kp + kp * lda] = temp;
                }

            } else {

                kp = -ipiv[k] - 1;
                if (kp != k) {
                    if (kp > 0) {
                        cblas_zswap(kp, &A[0 + k * lda], 1, &A[0 + kp * lda], 1);
                    }
                    cblas_zswap(k - kp - 1, &A[kp + 1 + k * lda], 1, &A[kp + (kp + 1) * lda], lda);

                    temp = A[k + k * lda];
                    A[k + k * lda] = A[kp + kp * lda];
                    A[kp + kp * lda] = temp;
                    temp = A[k + (k + 1) * lda];
                    A[k + (k + 1) * lda] = A[kp + (k + 1) * lda];
                    A[kp + (k + 1) * lda] = temp;
                }

                k = k + 1;
                kp = -ipiv[k] - 1;
                if (kp != k) {
                    if (kp > 0) {
                        cblas_zswap(kp, &A[0 + k * lda], 1, &A[0 + kp * lda], 1);
                    }
                    cblas_zswap(k - kp - 1, &A[kp + 1 + k * lda], 1, &A[kp + (kp + 1) * lda], lda);
                    temp = A[k + k * lda];
                    A[k + k * lda] = A[kp + kp * lda];
                    A[kp + kp * lda] = temp;
                }
            }

            k = k + 1;
        }

    } else {

        k = n - 1;
        while (k >= 0) {

            if (ipiv[k] >= 0) {

                A[k + k * lda] = ONE / A[k + k * lda];

                if (k < n - 1) {
                    c128 dotval;
                    cblas_zcopy(n - k - 1, &A[k + 1 + k * lda], 1, work, 1);
                    zsymv(uplo, n - k - 1, NEG_ONE,
                          &A[k + 1 + (k + 1) * lda], lda, work, 1, ZERO, &A[k + 1 + k * lda], 1);
                    cblas_zdotu_sub(n - k - 1, work, 1, &A[k + 1 + k * lda], 1, &dotval);
                    A[k + k * lda] -= dotval;
                }
                kstep = 1;

            } else {

                t = A[k + (k - 1) * lda];
                ak = A[k - 1 + (k - 1) * lda] / t;
                akp1 = A[k + k * lda] / t;
                akkp1 = A[k + (k - 1) * lda] / t;
                d = t * (ak * akp1 - ONE);
                A[k - 1 + (k - 1) * lda] = akp1 / d;
                A[k + k * lda] = ak / d;
                A[k + (k - 1) * lda] = -akkp1 / d;

                if (k < n - 1) {
                    c128 dotval;
                    cblas_zcopy(n - k - 1, &A[k + 1 + k * lda], 1, work, 1);
                    zsymv(uplo, n - k - 1, NEG_ONE,
                          &A[k + 1 + (k + 1) * lda], lda, work, 1, ZERO, &A[k + 1 + k * lda], 1);
                    cblas_zdotu_sub(n - k - 1, work, 1, &A[k + 1 + k * lda], 1, &dotval);
                    A[k + k * lda] -= dotval;
                    cblas_zdotu_sub(n - k - 1, &A[k + 1 + k * lda], 1, &A[k + 1 + (k - 1) * lda], 1, &dotval);
                    A[k + (k - 1) * lda] -= dotval;
                    cblas_zcopy(n - k - 1, &A[k + 1 + (k - 1) * lda], 1, work, 1);
                    zsymv(uplo, n - k - 1, NEG_ONE,
                          &A[k + 1 + (k + 1) * lda], lda, work, 1, ZERO, &A[k + 1 + (k - 1) * lda], 1);
                    cblas_zdotu_sub(n - k - 1, work, 1, &A[k + 1 + (k - 1) * lda], 1, &dotval);
                    A[k - 1 + (k - 1) * lda] -= dotval;
                }
                kstep = 2;
            }

            if (kstep == 1) {

                kp = ipiv[k];
                if (kp != k) {
                    if (kp < n - 1) {
                        cblas_zswap(n - kp - 1, &A[kp + 1 + k * lda], 1, &A[kp + 1 + kp * lda], 1);
                    }
                    cblas_zswap(kp - k - 1, &A[k + 1 + k * lda], 1, &A[kp + (k + 1) * lda], lda);
                    temp = A[k + k * lda];
                    A[k + k * lda] = A[kp + kp * lda];
                    A[kp + kp * lda] = temp;
                }

            } else {

                kp = -ipiv[k] - 1;
                if (kp != k) {
                    if (kp < n - 1) {
                        cblas_zswap(n - kp - 1, &A[kp + 1 + k * lda], 1, &A[kp + 1 + kp * lda], 1);
                    }
                    cblas_zswap(kp - k - 1, &A[k + 1 + k * lda], 1, &A[kp + (k + 1) * lda], lda);

                    temp = A[k + k * lda];
                    A[k + k * lda] = A[kp + kp * lda];
                    A[kp + kp * lda] = temp;
                    temp = A[k + (k - 1) * lda];
                    A[k + (k - 1) * lda] = A[kp + (k - 1) * lda];
                    A[kp + (k - 1) * lda] = temp;
                }

                k = k - 1;
                kp = -ipiv[k] - 1;
                if (kp != k) {
                    if (kp < n - 1) {
                        cblas_zswap(n - kp - 1, &A[kp + 1 + k * lda], 1, &A[kp + 1 + kp * lda], 1);
                    }
                    cblas_zswap(kp - k - 1, &A[k + 1 + k * lda], 1, &A[kp + (k + 1) * lda], lda);
                    temp = A[k + k * lda];
                    A[k + k * lda] = A[kp + kp * lda];
                    A[kp + kp * lda] = temp;
                }
            }

            k = k - 1;
        }
    }
}
