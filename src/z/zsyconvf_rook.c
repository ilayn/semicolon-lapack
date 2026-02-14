/**
 * @file zsyconvf_rook.c
 * @brief ZSYCONVF_ROOK converts between factorization formats used in ZSYTRF_ROOK and ZSYTRF_RK/ZSYTRF_BK.
 */

#include <complex.h>
#include <cblas.h>
#include "semicolon_lapack_complex_double.h"

/**
 * If parameter WAY = 'C':
 * ZSYCONVF_ROOK converts the factorization output format used in
 * ZSYTRF_ROOK provided on entry in parameter A into the factorization
 * output format used in ZSYTRF_RK (or ZSYTRF_BK) that is stored
 * on exit in parameters A and E. IPIV format for ZSYTRF_ROOK and
 * ZSYTRF_RK (or ZSYTRF_BK) is the same and is not converted.
 *
 * If parameter WAY = 'R':
 * ZSYCONVF_ROOK performs the conversion in reverse direction, i.e.
 * converts the factorization output format used in ZSYTRF_RK
 * (or ZSYTRF_BK) provided on entry in parameters A and E into
 * the factorization output format used in ZSYTRF_ROOK that is stored
 * on exit in parameter A. IPIV format for ZSYTRF_ROOK and
 * ZSYTRF_RK (or ZSYTRF_BK) is the same and is not converted.
 *
 * @param[in] uplo
 *          Specifies whether the details of the factorization are
 *          stored as an upper or lower triangular matrix A.
 *          = 'U':  Upper triangular
 *          = 'L':  Lower triangular
 *
 * @param[in] way
 *          = 'C': Convert
 *          = 'R': Revert
 *
 * @param[in] n
 *          The order of the matrix A. n >= 0.
 *
 * @param[in,out] A
 *          Double complex array, dimension (lda, n).
 *
 * @param[in] lda
 *          The leading dimension of the array A. lda >= max(1, n).
 *
 * @param[in,out] E
 *          Double complex array, dimension (n).
 *
 * @param[in] ipiv
 *          Integer array, dimension (n).
 *
 * @param[out] info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 */
void zsyconvf_rook(
    const char* uplo,
    const char* way,
    const int n,
    c128* restrict A,
    const int lda,
    c128* restrict E,
    const int* restrict ipiv,
    int* info)
{
    const c128 ZERO = CMPLX(0.0, 0.0);

    int upper, convert;
    int i, ip, ip2;

    *info = 0;
    upper = (uplo[0] == 'U' || uplo[0] == 'u');
    convert = (way[0] == 'C' || way[0] == 'c');
    if (!upper && !(uplo[0] == 'L' || uplo[0] == 'l')) {
        *info = -1;
    } else if (!convert && !(way[0] == 'R' || way[0] == 'r')) {
        *info = -2;
    } else if (n < 0) {
        *info = -3;
    } else if (lda < (1 > n ? 1 : n)) {
        *info = -5;
    }
    if (*info != 0) {
        xerbla("ZSYCONVF_ROOK", -(*info));
        return;
    }

    if (n == 0) {
        return;
    }

    if (upper) {

        if (convert) {

            i = n - 1;
            E[0] = ZERO;
            while (i > 0) {
                if (ipiv[i] < 0) {
                    E[i] = A[i - 1 + i * lda];
                    E[i - 1] = ZERO;
                    A[i - 1 + i * lda] = ZERO;
                    i = i - 1;
                } else {
                    E[i] = ZERO;
                }
                i = i - 1;
            }

            i = n - 1;
            while (i >= 0) {
                if (ipiv[i] >= 0) {

                    ip = ipiv[i];
                    if (i < n - 1) {
                        if (ip != i) {
                            cblas_zswap(n - i - 1, &A[i + (i + 1) * lda], lda,
                                        &A[ip + (i + 1) * lda], lda);
                        }
                    }

                } else {

                    ip = -ipiv[i] - 1;
                    ip2 = -ipiv[i - 1] - 1;
                    if (i < n - 1) {
                        if (ip != i) {
                            cblas_zswap(n - i - 1, &A[i + (i + 1) * lda], lda,
                                        &A[ip + (i + 1) * lda], lda);
                        }
                        if (ip2 != i - 1) {
                            cblas_zswap(n - i - 1, &A[i - 1 + (i + 1) * lda], lda,
                                        &A[ip2 + (i + 1) * lda], lda);
                        }
                    }
                    i = i - 1;

                }
                i = i - 1;
            }

        } else {

            i = 0;
            while (i < n) {
                if (ipiv[i] >= 0) {

                    ip = ipiv[i];
                    if (i < n - 1) {
                        if (ip != i) {
                            cblas_zswap(n - i - 1, &A[ip + (i + 1) * lda], lda,
                                        &A[i + (i + 1) * lda], lda);
                        }
                    }

                } else {

                    i = i + 1;
                    ip = -ipiv[i] - 1;
                    ip2 = -ipiv[i - 1] - 1;
                    if (i < n - 1) {
                        if (ip2 != i - 1) {
                            cblas_zswap(n - i - 1, &A[ip2 + (i + 1) * lda], lda,
                                        &A[i - 1 + (i + 1) * lda], lda);
                        }
                        if (ip != i) {
                            cblas_zswap(n - i - 1, &A[ip + (i + 1) * lda], lda,
                                        &A[i + (i + 1) * lda], lda);
                        }
                    }

                }
                i = i + 1;
            }

            i = n - 1;
            while (i > 0) {
                if (ipiv[i] < 0) {
                    A[i - 1 + i * lda] = E[i];
                    i = i - 1;
                }
                i = i - 1;
            }

        }

    } else {

        if (convert) {

            i = 0;
            E[n - 1] = ZERO;
            while (i < n) {
                if (i < n - 1 && ipiv[i] < 0) {
                    E[i] = A[i + 1 + i * lda];
                    E[i + 1] = ZERO;
                    A[i + 1 + i * lda] = ZERO;
                    i = i + 1;
                } else {
                    E[i] = ZERO;
                }
                i = i + 1;
            }

            i = 0;
            while (i < n) {
                if (ipiv[i] >= 0) {

                    ip = ipiv[i];
                    if (i > 0) {
                        if (ip != i) {
                            cblas_zswap(i, &A[i + 0 * lda], lda,
                                        &A[ip + 0 * lda], lda);
                        }
                    }

                } else {

                    ip = -ipiv[i] - 1;
                    ip2 = -ipiv[i + 1] - 1;
                    if (i > 0) {
                        if (ip != i) {
                            cblas_zswap(i, &A[i + 0 * lda], lda,
                                        &A[ip + 0 * lda], lda);
                        }
                        if (ip2 != i + 1) {
                            cblas_zswap(i, &A[i + 1 + 0 * lda], lda,
                                        &A[ip2 + 0 * lda], lda);
                        }
                    }
                    i = i + 1;

                }
                i = i + 1;
            }

        } else {

            i = n - 1;
            while (i >= 0) {
                if (ipiv[i] >= 0) {

                    ip = ipiv[i];
                    if (i > 0) {
                        if (ip != i) {
                            cblas_zswap(i, &A[ip + 0 * lda], lda,
                                        &A[i + 0 * lda], lda);
                        }
                    }

                } else {

                    i = i - 1;
                    ip = -ipiv[i] - 1;
                    ip2 = -ipiv[i + 1] - 1;
                    if (i > 0) {
                        if (ip2 != i + 1) {
                            cblas_zswap(i, &A[ip2 + 0 * lda], lda,
                                        &A[i + 1 + 0 * lda], lda);
                        }
                        if (ip != i) {
                            cblas_zswap(i, &A[ip + 0 * lda], lda,
                                        &A[i + 0 * lda], lda);
                        }
                    }

                }
                i = i - 1;
            }

            i = 0;
            while (i < n - 1) {
                if (ipiv[i] < 0) {
                    A[i + 1 + i * lda] = E[i];
                    i = i + 1;
                }
                i = i + 1;
            }

        }

    }
}
