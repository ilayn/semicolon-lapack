/**
 * @file ssyconvf_rook.c
 * @brief SSYCONVF_ROOK converts between factorization formats used in SSYTRF_ROOK and SSYTRF_RK/SSYTRF_BK.
 */

#include "semicolon_cblas.h"
#include "semicolon_lapack_single.h"

/**
 * If parameter WAY = 'C':
 * SSYCONVF_ROOK converts the factorization output format used in
 * SSYTRF_ROOK provided on entry in parameter A into the factorization
 * output format used in SSYTRF_RK (or SSYTRF_BK) that is stored
 * on exit in parameters A and E. IPIV format for SSYTRF_ROOK and
 * SSYTRF_RK (or SSYTRF_BK) is the same and is not converted.
 *
 * If parameter WAY = 'R':
 * SSYCONVF_ROOK performs the conversion in reverse direction, i.e.
 * converts the factorization output format used in SSYTRF_RK
 * (or SSYTRF_BK) provided on entry in parameters A and E into
 * the factorization output format used in SSYTRF_ROOK that is stored
 * on exit in parameter A. IPIV format for SSYTRF_ROOK and
 * SSYTRF_RK (or SSYTRF_BK) is the same and is not converted.
 *
 * @param[in]     uplo
 *                      - `'U'`: Upper triangular
 *                      - `'L'`: Lower triangular
 * @param[in]     way
 *                      - `'C'`: Convert
 *                      - `'R'`: Revert
 * @param[in]     n     The order of the matrix A. `n>=0`.
 * @param[in,out] A     Array of dimension `(lda,n)`.
 *
 *                      1) If `way='C'`:
 *
 *                      On entry, contains factorization details in format
 *                      used in `ssytrf_rook`:
 *
 *                      - all elements of the symmetric block diagonal
 *                        matrix D on the diagonal of A and on
 *                        superdiagonal (or subdiagonal) of A, and
 *                      - If `uplo='U'`: multipliers used to obtain factor
 *                        U in the superdiagonal part of A. If `uplo='L'`:
 *                        multipliers used to obtain factor L in the
 *                        superdiagonal part of A.
 *
 *                      On exit, contains factorization details in format
 *                      used in `ssytrf_rk` or `ssytrf_bk`:
 *
 *                      - Only diagonal elements of the symmetric block
 *                        diagonal matrix D on the diagonal of A, i.e.
 *                        `D(k,k)=A(k,k)`; (superdiagonal (or subdiagonal)
 *                        elements of D are stored on exit in array `E`),
 *                        and
 *                      - If `uplo='U'`: factor U in the superdiagonal part
 *                        of A. If `uplo='L'`: factor L in the subdiagonal
 *                        part of A.
 *
 *                      2) If `way='R'`:
 *
 *                      On entry, contains factorization details in format
 *                      used in `ssytrf_rk` or `ssytrf_bk`:
 *
 *                      - Only diagonal elements of the symmetric block
 *                        diagonal matrix D on the diagonal of A, i.e.
 *                        `D(k,k)=A(k,k)`; (superdiagonal (or subdiagonal)
 *                        elements of D are stored on exit in array `E`),
 *                        and
 *                      - If `uplo='U'`: factor U in the superdiagonal part
 *                        of A. If `uplo='L'`: factor L in the subdiagonal
 *                        part of A.
 *
 *                      On exit, contains factorization details in format
 *                      used in `ssytrf_rook`:
 *
 *                      - all elements of the symmetric block diagonal
 *                        matrix D on the diagonal of A and on
 *                        superdiagonal (or subdiagonal) of A, and
 *                      - If `uplo='U'`: multipliers used to obtain factor
 *                        U in the superdiagonal part of A. If `uplo='L'`:
 *                        multipliers used to obtain factor L in the
 *                        superdiagonal part of A.
 * @param[in]     lda   The leading dimension of the array A. `lda>=max(1,n)`.
 * @param[in,out] E     Array of dimension `n`.
 *
 *                      1) If `way='C'`:
 *
 *                      On entry, just a workspace.
 *
 *                      On exit, contains the superdiagonal (or
 *                      subdiagonal) elements of the symmetric block
 *                      diagonal matrix D with 1-by-1 or 2-by-2 diagonal
 *                      blocks, where if `uplo='U'`: `E[i]=D(i-1,i)`,
 *                      `i=1:n-1`, `E[0]` is set to 0; if `uplo='L'`:
 *                      `E[i]=D(i+1,i)`, `i=0:n-2`, `E[n-1]` is set to 0.
 *
 *                      2) If `way='R'`:
 *
 *                      On entry, contains the superdiagonal (or
 *                      subdiagonal) elements of the symmetric block
 *                      diagonal matrix D with 1-by-1 or 2-by-2 diagonal
 *                      blocks, where if `uplo='U'`: `E[i]=D(i-1,i)`,
 *                      `i=1:n-1`, `E[0]` not referenced; if `uplo='L'`:
 *                      `E[i]=D(i+1,i)`, `i=0:n-2`, `E[n-1]` not referenced.
 *
 *                      On exit, is not changed.
 * @param[in]     ipiv  Array of dimension `n`.
 *                      On entry, details of the interchanges and the block
 *                      structure of D as determined:
 *
 *                      1) by `ssytrf_rook`, if `way='C'`;
 *                      2) by `ssytrf_rk` (or `ssytrf_bk`), if `way='R'`.
 *
 *                      The ipiv format is the same for all these routines.
 *
 *                      On exit, is not changed.
 * @param[out]    info
 *                         - `info=0`: successful exit
 *                         - `info<0`: if `info=-i`, the i-th argument had an illegal
 *                           value
 */
void ssyconvf_rook(
    const char* uplo,
    const char* way,
    const INT n,
    f32* restrict A,
    const INT lda,
    f32* restrict E,
    const INT* restrict ipiv,
    INT* info)
{
    const f32 ZERO = 0.0f;

    INT upper, convert;
    INT i, ip, ip2;

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
        xerbla("SSYCONVF_ROOK", -(*info));
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
                            cblas_sswap(n - i - 1, &A[i + (i + 1) * lda], lda,
                                        &A[ip + (i + 1) * lda], lda);
                        }
                    }

                } else {

                    ip = -ipiv[i] - 1;
                    ip2 = -ipiv[i - 1] - 1;
                    if (i < n - 1) {
                        if (ip != i) {
                            cblas_sswap(n - i - 1, &A[i + (i + 1) * lda], lda,
                                        &A[ip + (i + 1) * lda], lda);
                        }
                        if (ip2 != i - 1) {
                            cblas_sswap(n - i - 1, &A[i - 1 + (i + 1) * lda], lda,
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
                            cblas_sswap(n - i - 1, &A[ip + (i + 1) * lda], lda,
                                        &A[i + (i + 1) * lda], lda);
                        }
                    }

                } else {

                    i = i + 1;
                    ip = -ipiv[i] - 1;
                    ip2 = -ipiv[i - 1] - 1;
                    if (i < n - 1) {
                        if (ip2 != i - 1) {
                            cblas_sswap(n - i - 1, &A[ip2 + (i + 1) * lda], lda,
                                        &A[i - 1 + (i + 1) * lda], lda);
                        }
                        if (ip != i) {
                            cblas_sswap(n - i - 1, &A[ip + (i + 1) * lda], lda,
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
                            cblas_sswap(i, &A[i + 0 * lda], lda,
                                        &A[ip + 0 * lda], lda);
                        }
                    }

                } else {

                    ip = -ipiv[i] - 1;
                    ip2 = -ipiv[i + 1] - 1;
                    if (i > 0) {
                        if (ip != i) {
                            cblas_sswap(i, &A[i + 0 * lda], lda,
                                        &A[ip + 0 * lda], lda);
                        }
                        if (ip2 != i + 1) {
                            cblas_sswap(i, &A[i + 1 + 0 * lda], lda,
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
                            cblas_sswap(i, &A[ip + 0 * lda], lda,
                                        &A[i + 0 * lda], lda);
                        }
                    }

                } else {

                    i = i - 1;
                    ip = -ipiv[i] - 1;
                    ip2 = -ipiv[i + 1] - 1;
                    if (i > 0) {
                        if (ip2 != i + 1) {
                            cblas_sswap(i, &A[ip2 + 0 * lda], lda,
                                        &A[i + 1 + 0 * lda], lda);
                        }
                        if (ip != i) {
                            cblas_sswap(i, &A[ip + 0 * lda], lda,
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
