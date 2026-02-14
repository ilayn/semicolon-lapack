/**
 * @file zlaqge.c
 * @brief ZLAQGE scales a general rectangular matrix, using row and column
 *        scaling factors computed by zgeequ.
 */

#include <complex.h>
#include <float.h>
#include "semicolon_lapack_complex_double.h"

/**
 * ZLAQGE equilibrates a general M by N matrix A using the row and
 * column scaling factors in the vectors R and C.
 *
 * @param[in]     m       The number of rows of the matrix A. m >= 0.
 * @param[in]     n       The number of columns of the matrix A. n >= 0.
 * @param[in,out] A       On entry, the M by N matrix A.
 *                        On exit, the equilibrated matrix. See equed for the
 *                        form of the equilibrated matrix.
 *                        Array of dimension (lda, n).
 * @param[in]     lda     The leading dimension of the array A. lda >= max(m, 1).
 * @param[in]     R       The row scale factors for A. Array of dimension (m).
 * @param[in]     C       The column scale factors for A. Array of dimension (n).
 * @param[in]     rowcnd  Ratio of the smallest R(i) to the largest R(i).
 * @param[in]     colcnd  Ratio of the smallest C(i) to the largest C(i).
 * @param[in]     amax    Absolute value of largest matrix entry.
 * @param[out]    equed   Specifies the form of equilibration that was done:
 *                        = 'N': No equilibration
 *                        = 'R': Row equilibration, i.e., A has been premultiplied
 *                               by diag(R).
 *                        = 'C': Column equilibration, i.e., A has been postmultiplied
 *                               by diag(C).
 *                        = 'B': Both row and column equilibration, i.e., A has been
 *                               replaced by diag(R) * A * diag(C).
 */
void zlaqge(
    const int m,
    const int n,
    c128* const restrict A,
    const int lda,
    const f64* const restrict R,
    const f64* const restrict C,
    const f64 rowcnd,
    const f64 colcnd,
    const f64 amax,
    char *equed)
{
    const f64 ONE = 1.0;
    const f64 THRESH = 0.1;

    int i, j;
    f64 cj, large, small;

    // Quick return if possible
    if (m <= 0 || n <= 0) {
        *equed = 'N';
        return;
    }

    // Initialize LARGE and SMALL
    // SMALL = safe minimum / machine precision
    // This is the smallest number such that 1/SMALL doesn't overflow
    small = DBL_MIN / DBL_EPSILON;
    large = ONE / small;

    if (rowcnd >= THRESH && amax >= small && amax <= large) {
        // No row scaling
        if (colcnd >= THRESH) {
            // No column scaling
            *equed = 'N';
        } else {
            // Column scaling only
            for (j = 0; j < n; j++) {
                cj = C[j];
                for (i = 0; i < m; i++) {
                    A[i + j * lda] = cj * A[i + j * lda];
                }
            }
            *equed = 'C';
        }
    } else if (colcnd >= THRESH) {
        // Row scaling, no column scaling
        for (j = 0; j < n; j++) {
            for (i = 0; i < m; i++) {
                A[i + j * lda] = R[i] * A[i + j * lda];
            }
        }
        *equed = 'R';
    } else {
        // Row and column scaling
        for (j = 0; j < n; j++) {
            cj = C[j];
            for (i = 0; i < m; i++) {
                A[i + j * lda] = cj * R[i] * A[i + j * lda];
            }
        }
        *equed = 'B';
    }
}
