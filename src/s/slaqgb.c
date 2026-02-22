/**
 * @file slaqgb.c
 * @brief SLAQGB scales a general band matrix, using row and column
 *        scaling factors computed by sgbequ.
 */

#include "internal_build_defs.h"
#include <float.h>
#include "semicolon_lapack_single.h"

/**
 * SLAQGB equilibrates a general M by N band matrix A with KL subdiagonals
 * and KU superdiagonals using the row and column scaling factors in the
 * vectors R and C.
 *
 * @param[in]     m       The number of rows of the matrix A. m >= 0.
 * @param[in]     n       The number of columns of the matrix A. n >= 0.
 * @param[in]     kl      The number of subdiagonals within the band of A. kl >= 0.
 * @param[in]     ku      The number of superdiagonals within the band of A. ku >= 0.
 * @param[in,out] AB      On entry, the matrix A in band storage, in rows 0 to kl+ku.
 *                        The j-th column of A is stored in the j-th column of
 *                        the array AB as follows:
 *                        AB[ku+i-j + j*ldab] = A(i,j) for max(0,j-ku) <= i <= min(m-1,j+kl).
 *                        On exit, the equilibrated matrix in the same storage format.
 *                        Array of dimension (ldab, n).
 * @param[in]     ldab    The leading dimension of the array AB. ldab >= kl+ku+1.
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
void slaqgb(
    const INT m,
    const INT n,
    const INT kl,
    const INT ku,
    f32* restrict AB,
    const INT ldab,
    const f32* restrict R,
    const f32* restrict C,
    const f32 rowcnd,
    const f32 colcnd,
    const f32 amax,
    char* equed)
{
    const f32 ONE = 1.0f;
    const f32 THRESH = 0.1f;

    INT i, j;
    f32 cj, large, small;
    INT i_start, i_end;

    /* Quick return if possible */
    if (m <= 0 || n <= 0) {
        *equed = 'N';
        return;
    }

    /* Initialize LARGE and SMALL
       SMALL = safe minimum / machine precision
       This is the smallest number such that 1/SMALL doesn't overflow */
    small = FLT_MIN / FLT_EPSILON;
    large = ONE / small;

    if (rowcnd >= THRESH && amax >= small && amax <= large) {
        /* No row scaling */
        if (colcnd >= THRESH) {
            /* No column scaling */
            *equed = 'N';
        } else {
            /* Column scaling only */
            for (j = 0; j < n; j++) {
                cj = C[j];
                /*
                 * Row range for column j in 0-based indexing:
                 * i from max(0, j-ku) to min(m-1, j+kl)
                 * Band storage: AB[ku + i - j + j*ldab] = A(i,j)
                 */
                i_start = (j - ku > 0) ? j - ku : 0;
                i_end = (j + kl < m - 1) ? j + kl : m - 1;
                for (i = i_start; i <= i_end; i++) {
                    AB[ku + i - j + j * ldab] = cj * AB[ku + i - j + j * ldab];
                }
            }
            *equed = 'C';
        }
    } else if (colcnd >= THRESH) {
        /* Row scaling, no column scaling */
        for (j = 0; j < n; j++) {
            i_start = (j - ku > 0) ? j - ku : 0;
            i_end = (j + kl < m - 1) ? j + kl : m - 1;
            for (i = i_start; i <= i_end; i++) {
                AB[ku + i - j + j * ldab] = R[i] * AB[ku + i - j + j * ldab];
            }
        }
        *equed = 'R';
    } else {
        /* Row and column scaling */
        for (j = 0; j < n; j++) {
            cj = C[j];
            i_start = (j - ku > 0) ? j - ku : 0;
            i_end = (j + kl < m - 1) ? j + kl : m - 1;
            for (i = i_start; i <= i_end; i++) {
                AB[ku + i - j + j * ldab] = cj * R[i] * AB[ku + i - j + j * ldab];
            }
        }
        *equed = 'B';
    }
}
