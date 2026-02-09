/**
 * @file ssyconv.c
 * @brief SSYCONV converts A given by TRF into L and D and vice-versa.
 */

#include "semicolon_lapack_single.h"

/**
 * SSYCONV convert A given by TRF into L and D and vice-versa.
 * Get Non-diag elements of D (returned in workspace) and
 * apply or reverse permutation done in TRF.
 *
 * @param[in] uplo
 *          Specifies whether the details of the factorization are stored
 *          as an upper or lower triangular matrix.
 *          = 'U':  Upper triangular, form is A = U*D*U**T;
 *          = 'L':  Lower triangular, form is A = L*D*L**T.
 *
 * @param[in] way
 *          = 'C': Convert
 *          = 'R': Revert
 *
 * @param[in] n
 *          The order of the matrix A. n >= 0.
 *
 * @param[in,out] A
 *          Double precision array, dimension (lda, n).
 *          The block diagonal matrix D and the multipliers used to
 *          obtain the factor U or L as computed by SSYTRF.
 *
 * @param[in] lda
 *          The leading dimension of the array A. lda >= max(1, n).
 *
 * @param[in] ipiv
 *          Integer array, dimension (n).
 *          Details of the interchanges and the block structure of D
 *          as determined by SSYTRF.
 *
 * @param[out] E
 *          Double precision array, dimension (n).
 *          E stores the supdiagonal/subdiagonal of the symmetric 1-by-1
 *          or 2-by-2 block diagonal matrix D in LDLT.
 *
 * @param[out] info
 *          = 0: successful exit
 *          < 0: if info = -i, the i-th argument had an illegal value
 */
void ssyconv(
    const char* uplo,
    const char* way,
    const int n,
    float* const restrict A,
    const int lda,
    const int* restrict ipiv,
    float* restrict E,
    int* info)
{
    const float ZERO = 0.0f;

    int upper, convert;
    int i, ip, j;
    float temp;

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
        xerbla("SSYCONV", -(*info));
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
                        for (j = i + 1; j < n; j++) {
                            temp = A[ip + j * lda];
                            A[ip + j * lda] = A[i + j * lda];
                            A[i + j * lda] = temp;
                        }
                    }
                } else {
                    ip = -ipiv[i] - 1;
                    if (i < n - 1) {
                        for (j = i + 1; j < n; j++) {
                            temp = A[ip + j * lda];
                            A[ip + j * lda] = A[i - 1 + j * lda];
                            A[i - 1 + j * lda] = temp;
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
                        for (j = i + 1; j < n; j++) {
                            temp = A[ip + j * lda];
                            A[ip + j * lda] = A[i + j * lda];
                            A[i + j * lda] = temp;
                        }
                    }
                } else {
                    ip = -ipiv[i] - 1;
                    i = i + 1;
                    if (i < n - 1) {
                        for (j = i + 1; j < n; j++) {
                            temp = A[ip + j * lda];
                            A[ip + j * lda] = A[i - 1 + j * lda];
                            A[i - 1 + j * lda] = temp;
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
                        for (j = 0; j < i; j++) {
                            temp = A[ip + j * lda];
                            A[ip + j * lda] = A[i + j * lda];
                            A[i + j * lda] = temp;
                        }
                    }
                } else {
                    ip = -ipiv[i] - 1;
                    if (i > 0) {
                        for (j = 0; j < i; j++) {
                            temp = A[ip + j * lda];
                            A[ip + j * lda] = A[i + 1 + j * lda];
                            A[i + 1 + j * lda] = temp;
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
                        for (j = 0; j < i; j++) {
                            temp = A[i + j * lda];
                            A[i + j * lda] = A[ip + j * lda];
                            A[ip + j * lda] = temp;
                        }
                    }
                } else {
                    ip = -ipiv[i] - 1;
                    i = i - 1;
                    if (i > 0) {
                        for (j = 0; j < i; j++) {
                            temp = A[i + 1 + j * lda];
                            A[i + 1 + j * lda] = A[ip + j * lda];
                            A[ip + j * lda] = temp;
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
