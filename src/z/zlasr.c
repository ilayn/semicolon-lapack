/**
 * @file zlasr.c
 * @brief ZLASR applies a sequence of plane rotations to a general
 *        rectangular matrix.
 */

#include "internal_build_defs.h"
#include "semicolon_lapack_complex_double.h"
#include <complex.h>

/**
 * ZLASR applies a sequence of real plane rotations to a complex matrix A,
 * from either the left or the right.
 *
 * When SIDE = 'L', the transformation takes the form
 *    A := P*A
 * and when SIDE = 'R', the transformation takes the form
 *    A := A*P**T
 *
 * where P is an orthogonal matrix consisting of a sequence of z plane
 * rotations, with z = M when SIDE = 'L' and z = N when SIDE = 'R',
 * and P**T is the transpose of P.
 *
 * @param[in]     side    'L': Left, compute A := P*A
 *                        'R': Right, compute A := A*P**T
 * @param[in]     pivot   'V': Variable pivot, the plane (k,k+1)
 *                        'T': Top pivot, the plane (1,k+1)
 *                        'B': Bottom pivot, the plane (k,z)
 * @param[in]     direct  'F': Forward, P = P(z-1)*...*P(2)*P(1)
 *                        'B': Backward, P = P(1)*P(2)*...*P(z-1)
 * @param[in]     m       The number of rows of the matrix A.
 * @param[in]     n       The number of columns of the matrix A.
 * @param[in]     C_rot   The cosines c(k) of the plane rotations.
 *                        Dimension (m-1) if SIDE='L', (n-1) if SIDE='R'.
 * @param[in]     S_rot   The sines s(k) of the plane rotations.
 *                        Dimension (m-1) if SIDE='L', (n-1) if SIDE='R'.
 * @param[in,out] A       Complex*16 array, dimension (lda, n).
 *                        On exit, A is overwritten by P*A or A*P**T.
 * @param[in]     lda     The leading dimension of A. lda >= max(1,m).
 */
void zlasr(const char* side, const char* pivot, const char* direct,
           const INT m, const INT n,
           const f64* restrict C_rot, const f64* restrict S_rot,
           c128* restrict A, const INT lda)
{
    INT i, j;
    f64 ctemp, stemp;
    c128 temp;

    /* Quick return if possible */
    if (m == 0 || n == 0) return;

    if (side[0] == 'L' || side[0] == 'l') {
        /* Form P * A */
        if (pivot[0] == 'V' || pivot[0] == 'v') {
            if (direct[0] == 'F' || direct[0] == 'f') {
                for (j = 0; j < m - 1; j++) {
                    ctemp = C_rot[j];
                    stemp = S_rot[j];
                    if (ctemp != 1.0 || stemp != 0.0) {
                        for (i = 0; i < n; i++) {
                            temp = A[(j + 1) + i * lda];
                            A[(j + 1) + i * lda] = ctemp * temp - stemp * A[j + i * lda];
                            A[j + i * lda] = stemp * temp + ctemp * A[j + i * lda];
                        }
                    }
                }
            } else {
                /* direct == 'B' */
                for (j = m - 2; j >= 0; j--) {
                    ctemp = C_rot[j];
                    stemp = S_rot[j];
                    if (ctemp != 1.0 || stemp != 0.0) {
                        for (i = 0; i < n; i++) {
                            temp = A[(j + 1) + i * lda];
                            A[(j + 1) + i * lda] = ctemp * temp - stemp * A[j + i * lda];
                            A[j + i * lda] = stemp * temp + ctemp * A[j + i * lda];
                        }
                    }
                }
            }
        } else if (pivot[0] == 'T' || pivot[0] == 't') {
            if (direct[0] == 'F' || direct[0] == 'f') {
                for (j = 1; j < m; j++) {
                    ctemp = C_rot[j - 1];
                    stemp = S_rot[j - 1];
                    if (ctemp != 1.0 || stemp != 0.0) {
                        for (i = 0; i < n; i++) {
                            temp = A[j + i * lda];
                            A[j + i * lda] = ctemp * temp - stemp * A[0 + i * lda];
                            A[0 + i * lda] = stemp * temp + ctemp * A[0 + i * lda];
                        }
                    }
                }
            } else {
                /* direct == 'B' */
                for (j = m - 1; j >= 1; j--) {
                    ctemp = C_rot[j - 1];
                    stemp = S_rot[j - 1];
                    if (ctemp != 1.0 || stemp != 0.0) {
                        for (i = 0; i < n; i++) {
                            temp = A[j + i * lda];
                            A[j + i * lda] = ctemp * temp - stemp * A[0 + i * lda];
                            A[0 + i * lda] = stemp * temp + ctemp * A[0 + i * lda];
                        }
                    }
                }
            }
        } else if (pivot[0] == 'B' || pivot[0] == 'b') {
            if (direct[0] == 'F' || direct[0] == 'f') {
                for (j = 0; j < m - 1; j++) {
                    ctemp = C_rot[j];
                    stemp = S_rot[j];
                    if (ctemp != 1.0 || stemp != 0.0) {
                        for (i = 0; i < n; i++) {
                            temp = A[j + i * lda];
                            A[j + i * lda] = stemp * A[(m - 1) + i * lda] + ctemp * temp;
                            A[(m - 1) + i * lda] = ctemp * A[(m - 1) + i * lda] - stemp * temp;
                        }
                    }
                }
            } else {
                /* direct == 'B' */
                for (j = m - 2; j >= 0; j--) {
                    ctemp = C_rot[j];
                    stemp = S_rot[j];
                    if (ctemp != 1.0 || stemp != 0.0) {
                        for (i = 0; i < n; i++) {
                            temp = A[j + i * lda];
                            A[j + i * lda] = stemp * A[(m - 1) + i * lda] + ctemp * temp;
                            A[(m - 1) + i * lda] = ctemp * A[(m - 1) + i * lda] - stemp * temp;
                        }
                    }
                }
            }
        }
    } else if (side[0] == 'R' || side[0] == 'r') {
        /* Form A * P**T */
        if (pivot[0] == 'V' || pivot[0] == 'v') {
            if (direct[0] == 'F' || direct[0] == 'f') {
                for (j = 0; j < n - 1; j++) {
                    ctemp = C_rot[j];
                    stemp = S_rot[j];
                    if (ctemp != 1.0 || stemp != 0.0) {
                        for (i = 0; i < m; i++) {
                            temp = A[i + (j + 1) * lda];
                            A[i + (j + 1) * lda] = ctemp * temp - stemp * A[i + j * lda];
                            A[i + j * lda] = stemp * temp + ctemp * A[i + j * lda];
                        }
                    }
                }
            } else {
                /* direct == 'B' */
                for (j = n - 2; j >= 0; j--) {
                    ctemp = C_rot[j];
                    stemp = S_rot[j];
                    if (ctemp != 1.0 || stemp != 0.0) {
                        for (i = 0; i < m; i++) {
                            temp = A[i + (j + 1) * lda];
                            A[i + (j + 1) * lda] = ctemp * temp - stemp * A[i + j * lda];
                            A[i + j * lda] = stemp * temp + ctemp * A[i + j * lda];
                        }
                    }
                }
            }
        } else if (pivot[0] == 'T' || pivot[0] == 't') {
            if (direct[0] == 'F' || direct[0] == 'f') {
                for (j = 1; j < n; j++) {
                    ctemp = C_rot[j - 1];
                    stemp = S_rot[j - 1];
                    if (ctemp != 1.0 || stemp != 0.0) {
                        for (i = 0; i < m; i++) {
                            temp = A[i + j * lda];
                            A[i + j * lda] = ctemp * temp - stemp * A[i + 0 * lda];
                            A[i + 0 * lda] = stemp * temp + ctemp * A[i + 0 * lda];
                        }
                    }
                }
            } else {
                /* direct == 'B' */
                for (j = n - 1; j >= 1; j--) {
                    ctemp = C_rot[j - 1];
                    stemp = S_rot[j - 1];
                    if (ctemp != 1.0 || stemp != 0.0) {
                        for (i = 0; i < m; i++) {
                            temp = A[i + j * lda];
                            A[i + j * lda] = ctemp * temp - stemp * A[i + 0 * lda];
                            A[i + 0 * lda] = stemp * temp + ctemp * A[i + 0 * lda];
                        }
                    }
                }
            }
        } else if (pivot[0] == 'B' || pivot[0] == 'b') {
            if (direct[0] == 'F' || direct[0] == 'f') {
                for (j = 0; j < n - 1; j++) {
                    ctemp = C_rot[j];
                    stemp = S_rot[j];
                    if (ctemp != 1.0 || stemp != 0.0) {
                        for (i = 0; i < m; i++) {
                            temp = A[i + j * lda];
                            A[i + j * lda] = stemp * A[i + (n - 1) * lda] + ctemp * temp;
                            A[i + (n - 1) * lda] = ctemp * A[i + (n - 1) * lda] - stemp * temp;
                        }
                    }
                }
            } else {
                /* direct == 'B' */
                for (j = n - 2; j >= 0; j--) {
                    ctemp = C_rot[j];
                    stemp = S_rot[j];
                    if (ctemp != 1.0 || stemp != 0.0) {
                        for (i = 0; i < m; i++) {
                            temp = A[i + j * lda];
                            A[i + j * lda] = stemp * A[i + (n - 1) * lda] + ctemp * temp;
                            A[i + (n - 1) * lda] = ctemp * A[i + (n - 1) * lda] - stemp * temp;
                        }
                    }
                }
            }
        }
    }
}
