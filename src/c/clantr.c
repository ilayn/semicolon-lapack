/**
 * @file clantr.c
 * @brief CLANTR returns the value of the 1-norm, Frobenius norm,
 *        infinity-norm, or the largest absolute value of any element of
 *        a trapezoidal or triangular matrix.
 */

#include "internal_build_defs.h"
#include <math.h>
#include <complex.h>
#include "semicolon_lapack_complex_single.h"

/**
 * CLANTR returns the value of the one norm, or the Frobenius norm, or
 * the infinity norm, or the element of largest absolute value of a
 * trapezoidal or triangular matrix A.
 *
 * @param[in] norm  Specifies the value to be returned:
 *                  = 'M' or 'm': max(abs(A(i,j)))
 *                  = "1", 'O' or 'o': norm1(A) (max column sum)
 *                  = 'I' or 'i': normI(A) (max row sum)
 *                  = "F", "f", 'E' or 'e': normF(A) (Frobenius norm)
 * @param[in] uplo  Specifies whether the matrix A is upper or lower trapezoidal.
 *                  = 'U': Upper trapezoidal
 *                  = 'L': Lower trapezoidal
 *                  Note that A is triangular instead of trapezoidal if m = n.
 * @param[in] diag  Specifies whether or not the matrix A has unit diagonal.
 *                  = 'N': Non-unit diagonal
 *                  = 'U': Unit diagonal
 * @param[in] m     The number of rows of the matrix A. m >= 0.
 *                  When m = 0, clantr returns zero.
 * @param[in] n     The number of columns of the matrix A. n >= 0.
 *                  When n = 0, clantr returns zero.
 * @param[in] A     The trapezoidal matrix A (A is triangular if m = n).
 *                  Array of dimension (lda, n).
 * @param[in] lda   The leading dimension of the array A. lda >= max(m, 1).
 * @param[out] work Workspace array of dimension max(1, lwork), where
 *                  lwork >= m when norm = 'I'; otherwise, work is not referenced.
 *
 * @return The computed norm value.
 */
f32 clantr(
    const char* norm,
    const char* uplo,
    const char* diag,
    const INT m,
    const INT n,
    const c64* restrict A,
    const INT lda,
    f32* restrict work)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;

    INT i, j;
    f32 scale, sum, value, temp;
    INT udiag;  /* unit diagonal flag */
    INT minmn;

    /* Quick return if possible */
    minmn = (m < n) ? m : n;
    if (minmn == 0) {
        return ZERO;
    }

    udiag = (diag[0] == 'U' || diag[0] == 'u');

    if (norm[0] == 'M' || norm[0] == 'm') {
        /* Find max(abs(A(i,j))) */
        if (udiag) {
            value = ONE;
            if (uplo[0] == 'U' || uplo[0] == 'u') {
                for (j = 0; j < n; j++) {
                    INT imax = (j < m) ? j : m;
                    for (i = 0; i < imax; i++) {
                        temp = cabsf(A[i + j * lda]);
                        if (value < temp || isnan(temp)) {
                            value = temp;
                        }
                    }
                }
            } else {
                for (j = 0; j < n; j++) {
                    for (i = j + 1; i < m; i++) {
                        temp = cabsf(A[i + j * lda]);
                        if (value < temp || isnan(temp)) {
                            value = temp;
                        }
                    }
                }
            }
        } else {
            value = ZERO;
            if (uplo[0] == 'U' || uplo[0] == 'u') {
                for (j = 0; j < n; j++) {
                    INT imax = (j + 1 < m) ? j + 1 : m;
                    for (i = 0; i < imax; i++) {
                        temp = cabsf(A[i + j * lda]);
                        if (value < temp || isnan(temp)) {
                            value = temp;
                        }
                    }
                }
            } else {
                for (j = 0; j < n; j++) {
                    for (i = j; i < m; i++) {
                        temp = cabsf(A[i + j * lda]);
                        if (value < temp || isnan(temp)) {
                            value = temp;
                        }
                    }
                }
            }
        }
    } else if (norm[0] == 'O' || norm[0] == 'o' || norm[0] == '1') {
        /* Find norm1(A) - maximum column sum */
        value = ZERO;
        if (uplo[0] == 'U' || uplo[0] == 'u') {
            for (j = 0; j < n; j++) {
                if (udiag && j < m) {
                    sum = ONE;
                    for (i = 0; i < j; i++) {
                        sum += cabsf(A[i + j * lda]);
                    }
                } else {
                    sum = ZERO;
                    INT imax = (j + 1 < m) ? j + 1 : m;
                    for (i = 0; i < imax; i++) {
                        sum += cabsf(A[i + j * lda]);
                    }
                }
                if (value < sum || isnan(sum)) {
                    value = sum;
                }
            }
        } else {
            for (j = 0; j < n; j++) {
                if (udiag) {
                    sum = ONE;
                    for (i = j + 1; i < m; i++) {
                        sum += cabsf(A[i + j * lda]);
                    }
                } else {
                    sum = ZERO;
                    for (i = j; i < m; i++) {
                        sum += cabsf(A[i + j * lda]);
                    }
                }
                if (value < sum || isnan(sum)) {
                    value = sum;
                }
            }
        }
    } else if (norm[0] == 'I' || norm[0] == 'i') {
        /* Find normI(A) - maximum row sum */
        if (uplo[0] == 'U' || uplo[0] == 'u') {
            if (udiag) {
                for (i = 0; i < m; i++) {
                    work[i] = ONE;
                }
                for (j = 0; j < n; j++) {
                    INT imax = (j < m) ? j : m;
                    for (i = 0; i < imax; i++) {
                        work[i] += cabsf(A[i + j * lda]);
                    }
                }
            } else {
                for (i = 0; i < m; i++) {
                    work[i] = ZERO;
                }
                for (j = 0; j < n; j++) {
                    INT imax = (j + 1 < m) ? j + 1 : m;
                    for (i = 0; i < imax; i++) {
                        work[i] += cabsf(A[i + j * lda]);
                    }
                }
            }
        } else {
            if (udiag) {
                INT minmn_local = (m < n) ? m : n;
                for (i = 0; i < minmn_local; i++) {
                    work[i] = ONE;
                }
                for (i = n; i < m; i++) {
                    work[i] = ZERO;
                }
                for (j = 0; j < n; j++) {
                    for (i = j + 1; i < m; i++) {
                        work[i] += cabsf(A[i + j * lda]);
                    }
                }
            } else {
                for (i = 0; i < m; i++) {
                    work[i] = ZERO;
                }
                for (j = 0; j < n; j++) {
                    for (i = j; i < m; i++) {
                        work[i] += cabsf(A[i + j * lda]);
                    }
                }
            }
        }
        value = ZERO;
        for (i = 0; i < m; i++) {
            temp = work[i];
            if (value < temp || isnan(temp)) {
                value = temp;
            }
        }
    } else if (norm[0] == 'F' || norm[0] == 'f' || norm[0] == 'E' || norm[0] == 'e') {
        /* Find normF(A) - Frobenius norm using classq */
        if (uplo[0] == 'U' || uplo[0] == 'u') {
            if (udiag) {
                scale = ONE;
                sum = (f32)minmn;  /* count of unit diagonal elements */
                for (j = 1; j < n; j++) {
                    INT col_len = (j < m) ? j : m;
                    if (col_len > 0) {
                        classq(col_len, &A[j * lda], 1, &scale, &sum);
                    }
                }
            } else {
                scale = ZERO;
                sum = ONE;
                for (j = 0; j < n; j++) {
                    INT col_len = (j + 1 < m) ? j + 1 : m;
                    if (col_len > 0) {
                        classq(col_len, &A[j * lda], 1, &scale, &sum);
                    }
                }
            }
        } else {
            /* Lower triangular */
            if (udiag) {
                scale = ONE;
                sum = (f32)minmn;  /* count of unit diagonal elements */
                for (j = 0; j < n; j++) {
                    INT col_len = m - j - 1;
                    if (col_len > 0) {
                        INT start_idx = (j + 1 < m) ? j + 1 : m;
                        classq(col_len, &A[start_idx + j * lda], 1, &scale, &sum);
                    }
                }
            } else {
                scale = ZERO;
                sum = ONE;
                for (j = 0; j < n; j++) {
                    INT col_len = m - j;
                    if (col_len > 0) {
                        classq(col_len, &A[j + j * lda], 1, &scale, &sum);
                    }
                }
            }
        }
        value = scale * sqrtf(sum);
    } else {
        value = ZERO;
    }

    return value;
}
