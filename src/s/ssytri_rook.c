/**
 * @file ssytri_rook.c
 * @brief SSYTRI_ROOK computes the inverse of a real symmetric matrix using the factorization computed by SSYTRF_ROOK.
 */

#include "internal_build_defs.h"
#include "semicolon_lapack_single.h"
#include <cblas.h>
#include <math.h>

/**
 * SSYTRI_ROOK computes the inverse of a real symmetric
 * matrix A using the factorization A = U*D*U**T or A = L*D*L**T
 * computed by SSYTRF_ROOK.
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
 *          Double precision array, dimension (lda, n).
 *          On entry, the block diagonal matrix D and the multipliers
 *          used to obtain the factor U or L as computed by SSYTRF_ROOK.
 *          On exit, if info = 0, the (symmetric) inverse of the original
 *          matrix.
 *
 * @param[in] lda
 *          The leading dimension of the array A. lda >= max(1, n).
 *
 * @param[in] ipiv
 *          Integer array, dimension (n).
 *          Details of the interchanges and the block structure of D
 *          as determined by SSYTRF_ROOK.
 *
 * @param[out] work
 *          Double precision array, dimension (n).
 *
 * @param[out] info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 *                         - > 0: if info = i, D(i,i) = 0; the matrix is singular and its
 *                           inverse could not be computed.
 */
void ssytri_rook(
    const char* uplo,
    const INT n,
    f32* restrict A,
    const INT lda,
    const INT* restrict ipiv,
    f32* restrict work,
    INT* info)
{
    INT upper;
    INT k, kp, kstep;
    f32 ak, akkp1, akp1, d, t, temp;

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
        xerbla("SSYTRI_ROOK", -(*info));
        return;
    }

    if (n == 0) {
        return;
    }

    if (upper) {

        for (*info = n - 1; *info >= 0; (*info)--) {
            if (ipiv[*info] >= 0 && A[*info + (*info) * lda] == 0.0f) {
                (*info)++;
                return;
            }
        }

    } else {

        for (*info = 0; *info < n; (*info)++) {
            if (ipiv[*info] >= 0 && A[*info + (*info) * lda] == 0.0f) {
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

                A[k + k * lda] = 1.0f / A[k + k * lda];

                if (k > 0) {
                    cblas_scopy(k, &A[0 + k * lda], 1, work, 1);
                    cblas_ssymv(CblasColMajor, CblasUpper, k, -1.0f, A, lda,
                                work, 1, 0.0f, &A[0 + k * lda], 1);
                    A[k + k * lda] = A[k + k * lda] - cblas_sdot(k, work, 1, &A[0 + k * lda], 1);
                }
                kstep = 1;

            } else {

                t = fabsf(A[k + (k + 1) * lda]);
                ak = A[k + k * lda] / t;
                akp1 = A[k + 1 + (k + 1) * lda] / t;
                akkp1 = A[k + (k + 1) * lda] / t;
                d = t * (ak * akp1 - 1.0f);
                A[k + k * lda] = akp1 / d;
                A[k + 1 + (k + 1) * lda] = ak / d;
                A[k + (k + 1) * lda] = -akkp1 / d;

                if (k > 0) {
                    cblas_scopy(k, &A[0 + k * lda], 1, work, 1);
                    cblas_ssymv(CblasColMajor, CblasUpper, k, -1.0f, A, lda,
                                work, 1, 0.0f, &A[0 + k * lda], 1);
                    A[k + k * lda] = A[k + k * lda] - cblas_sdot(k, work, 1, &A[0 + k * lda], 1);
                    A[k + (k + 1) * lda] = A[k + (k + 1) * lda] -
                                           cblas_sdot(k, &A[0 + k * lda], 1, &A[0 + (k + 1) * lda], 1);
                    cblas_scopy(k, &A[0 + (k + 1) * lda], 1, work, 1);
                    cblas_ssymv(CblasColMajor, CblasUpper, k, -1.0f, A, lda,
                                work, 1, 0.0f, &A[0 + (k + 1) * lda], 1);
                    A[k + 1 + (k + 1) * lda] = A[k + 1 + (k + 1) * lda] -
                                               cblas_sdot(k, work, 1, &A[0 + (k + 1) * lda], 1);
                }
                kstep = 2;
            }

            if (kstep == 1) {

                kp = ipiv[k];
                if (kp != k) {
                    if (kp > 0) {
                        cblas_sswap(kp, &A[0 + k * lda], 1, &A[0 + kp * lda], 1);
                    }
                    cblas_sswap(k - kp - 1, &A[kp + 1 + k * lda], 1, &A[kp + (kp + 1) * lda], lda);
                    temp = A[k + k * lda];
                    A[k + k * lda] = A[kp + kp * lda];
                    A[kp + kp * lda] = temp;
                }

            } else {

                kp = -ipiv[k] - 1;
                if (kp != k) {
                    if (kp > 0) {
                        cblas_sswap(kp, &A[0 + k * lda], 1, &A[0 + kp * lda], 1);
                    }
                    cblas_sswap(k - kp - 1, &A[kp + 1 + k * lda], 1, &A[kp + (kp + 1) * lda], lda);

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
                        cblas_sswap(kp, &A[0 + k * lda], 1, &A[0 + kp * lda], 1);
                    }
                    cblas_sswap(k - kp - 1, &A[kp + 1 + k * lda], 1, &A[kp + (kp + 1) * lda], lda);
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

                A[k + k * lda] = 1.0f / A[k + k * lda];

                if (k < n - 1) {
                    cblas_scopy(n - k - 1, &A[k + 1 + k * lda], 1, work, 1);
                    cblas_ssymv(CblasColMajor, CblasLower, n - k - 1, -1.0f,
                                &A[k + 1 + (k + 1) * lda], lda, work, 1, 0.0f, &A[k + 1 + k * lda], 1);
                    A[k + k * lda] = A[k + k * lda] - cblas_sdot(n - k - 1, work, 1, &A[k + 1 + k * lda], 1);
                }
                kstep = 1;

            } else {

                t = fabsf(A[k + (k - 1) * lda]);
                ak = A[k - 1 + (k - 1) * lda] / t;
                akp1 = A[k + k * lda] / t;
                akkp1 = A[k + (k - 1) * lda] / t;
                d = t * (ak * akp1 - 1.0f);
                A[k - 1 + (k - 1) * lda] = akp1 / d;
                A[k + k * lda] = ak / d;
                A[k + (k - 1) * lda] = -akkp1 / d;

                if (k < n - 1) {
                    cblas_scopy(n - k - 1, &A[k + 1 + k * lda], 1, work, 1);
                    cblas_ssymv(CblasColMajor, CblasLower, n - k - 1, -1.0f,
                                &A[k + 1 + (k + 1) * lda], lda, work, 1, 0.0f, &A[k + 1 + k * lda], 1);
                    A[k + k * lda] = A[k + k * lda] - cblas_sdot(n - k - 1, work, 1, &A[k + 1 + k * lda], 1);
                    A[k + (k - 1) * lda] = A[k + (k - 1) * lda] -
                                           cblas_sdot(n - k - 1, &A[k + 1 + k * lda], 1, &A[k + 1 + (k - 1) * lda], 1);
                    cblas_scopy(n - k - 1, &A[k + 1 + (k - 1) * lda], 1, work, 1);
                    cblas_ssymv(CblasColMajor, CblasLower, n - k - 1, -1.0f,
                                &A[k + 1 + (k + 1) * lda], lda, work, 1, 0.0f, &A[k + 1 + (k - 1) * lda], 1);
                    A[k - 1 + (k - 1) * lda] = A[k - 1 + (k - 1) * lda] -
                                               cblas_sdot(n - k - 1, work, 1, &A[k + 1 + (k - 1) * lda], 1);
                }
                kstep = 2;
            }

            if (kstep == 1) {

                kp = ipiv[k];
                if (kp != k) {
                    if (kp < n - 1) {
                        cblas_sswap(n - kp - 1, &A[kp + 1 + k * lda], 1, &A[kp + 1 + kp * lda], 1);
                    }
                    cblas_sswap(kp - k - 1, &A[k + 1 + k * lda], 1, &A[kp + (k + 1) * lda], lda);
                    temp = A[k + k * lda];
                    A[k + k * lda] = A[kp + kp * lda];
                    A[kp + kp * lda] = temp;
                }

            } else {

                kp = -ipiv[k] - 1;
                if (kp != k) {
                    if (kp < n - 1) {
                        cblas_sswap(n - kp - 1, &A[kp + 1 + k * lda], 1, &A[kp + 1 + kp * lda], 1);
                    }
                    cblas_sswap(kp - k - 1, &A[k + 1 + k * lda], 1, &A[kp + (k + 1) * lda], lda);

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
                        cblas_sswap(n - kp - 1, &A[kp + 1 + k * lda], 1, &A[kp + 1 + kp * lda], 1);
                    }
                    cblas_sswap(kp - k - 1, &A[k + 1 + k * lda], 1, &A[kp + (k + 1) * lda], lda);
                    temp = A[k + k * lda];
                    A[k + k * lda] = A[kp + kp * lda];
                    A[kp + kp * lda] = temp;
                }
            }

            k = k - 1;
        }
    }
}
