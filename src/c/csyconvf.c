/**
 * @file csyconvf.c
 * @brief CSYCONVF converts between factorization formats used in CSYTRF and CSYTRF_RK/ZSYTRF_BK.
 */

#include "internal_build_defs.h"
#include <complex.h>
#include <cblas.h>
#include "semicolon_lapack_complex_single.h"

/**
 * If parameter WAY = 'C':
 * CSYCONVF converts the factorization output format used in
 * CSYTRF provided on entry in parameter A into the factorization
 * output format used in CSYTRF_RK (or ZSYTRF_BK) that is stored
 * on exit in parameters A and E. It also converts in place details of
 * the interchanges stored in IPIV from the format used in CSYTRF into
 * the format used in CSYTRF_RK (or ZSYTRF_BK).
 *
 * If parameter WAY = 'R':
 * CSYCONVF performs the conversion in reverse direction, i.e.
 * converts the factorization output format used in CSYTRF_RK
 * (or ZSYTRF_BK) provided on entry in parameters A and E into
 * the factorization output format used in CSYTRF that is stored
 * on exit in parameter A. It also converts in place details of
 * the interchanges stored in IPIV from the format used in CSYTRF_RK
 * (or ZSYTRF_BK) into the format used in CSYTRF.
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
 *          Single complex array, dimension (lda, n).
 *
 * @param[in] lda
 *          The leading dimension of the array A. lda >= max(1, n).
 *
 * @param[in,out] E
 *          Single complex array, dimension (n).
 *
 * @param[in,out] ipiv
 *          Integer array, dimension (n).
 *
 * @param[out] info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 */
void csyconvf(
    const char* uplo,
    const char* way,
    const INT n,
    c64* restrict A,
    const INT lda,
    c64* restrict E,
    INT* restrict ipiv,
    INT* info)
{
    const c64 ZERO = CMPLXF(0.0f, 0.0f);

    INT upper, convert;
    INT i, ip;

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
        xerbla("CSYCONVF", -(*info));
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
                            cblas_cswap(n - i - 1, &A[i + (i + 1) * lda], lda,
                                        &A[ip + (i + 1) * lda], lda);
                        }
                    }

                } else {

                    ip = -ipiv[i] - 1;
                    if (i < n - 1) {
                        if (ip != i - 1) {
                            cblas_cswap(n - i - 1, &A[i - 1 + (i + 1) * lda], lda,
                                        &A[ip + (i + 1) * lda], lda);
                        }
                    }

                    ipiv[i] = i + 1;

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
                            cblas_cswap(n - i - 1, &A[ip + (i + 1) * lda], lda,
                                        &A[i + (i + 1) * lda], lda);
                        }
                    }

                } else {

                    i = i + 1;
                    ip = -ipiv[i] - 1;
                    if (i < n - 1) {
                        if (ip != i - 1) {
                            cblas_cswap(n - i - 1, &A[ip + (i + 1) * lda], lda,
                                        &A[i - 1 + (i + 1) * lda], lda);
                        }
                    }

                    ipiv[i] = ipiv[i - 1];

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
                            cblas_cswap(i, &A[i + 0 * lda], lda,
                                        &A[ip + 0 * lda], lda);
                        }
                    }

                } else {

                    ip = -ipiv[i] - 1;
                    if (i > 0) {
                        if (ip != i + 1) {
                            cblas_cswap(i, &A[i + 1 + 0 * lda], lda,
                                        &A[ip + 0 * lda], lda);
                        }
                    }

                    ipiv[i] = i + 1;

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
                            cblas_cswap(i, &A[ip + 0 * lda], lda,
                                        &A[i + 0 * lda], lda);
                        }
                    }

                } else {

                    i = i - 1;
                    ip = -ipiv[i] - 1;
                    if (i > 0) {
                        if (ip != i + 1) {
                            cblas_cswap(i, &A[ip + 0 * lda], lda,
                                        &A[i + 1 + 0 * lda], lda);
                        }
                    }

                    ipiv[i] = ipiv[i + 1];

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
