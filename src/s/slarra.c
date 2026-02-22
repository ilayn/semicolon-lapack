/**
 * @file slarra.c
 * @brief SLARRA computes the splitting points with the specified threshold.
 */

#include <math.h>
#include "semicolon_lapack_single.h"

/**
 * SLARRA computes the splitting points with threshold SPLTOL.
 * SLARRA sets any "small" off-diagonal elements to zero.
 *
 * @param[in]     n       The order of the matrix. N > 0.
 * @param[in]     D       Double precision array, dimension (N).
 *                        The N diagonal elements of the tridiagonal matrix T.
 * @param[in,out] E       Double precision array, dimension (N).
 *                        On entry, the first (N-1) entries contain the subdiagonal
 *                        elements of the tridiagonal matrix T; E(N) need not be set.
 *                        On exit, the entries E(ISPLIT(I)), 0 <= I < NSPLIT,
 *                        are set to zero, the other entries of E are untouched.
 * @param[in,out] E2      Double precision array, dimension (N).
 *                        On entry, the first (N-1) entries contain the SQUARES of the
 *                        subdiagonal elements of the tridiagonal matrix T.
 *                        On exit, the entries E2(ISPLIT(I)) have been set to zero.
 * @param[in]     spltol  The threshold for splitting. Two criteria can be used:
 *                        SPLTOL<0 : criterion based on absolute off-diagonal value.
 *                        SPLTOL>0 : criterion that preserves relative accuracy.
 * @param[in]     tnrm    The norm of the matrix.
 * @param[out]    nsplit  The number of blocks T splits into. 1 <= NSPLIT <= N.
 * @param[out]    isplit  Integer array, dimension (N).
 *                        The splitting points, at which T breaks up into blocks.
 *                        The first block consists of rows/columns 0 to ISPLIT(0),
 *                        the second of rows/columns ISPLIT(0)+1 through ISPLIT(1),
 *                        etc.
 * @param[out]    info
 *                         - = 0: successful exit.
 */
void slarra(const INT n, const f32* restrict D,
            f32* restrict E, f32* restrict E2,
            const f32 spltol, const f32 tnrm,
            INT* nsplit, INT* restrict isplit, INT* info)
{
    INT i;
    f32 eabs, tmp1;

    *info = 0;
    *nsplit = 1;

    /* Quick return if possible */
    if (n <= 0) {
        return;
    }

    /* Compute splitting points */
    if (spltol < 0.0f) {
        /* Criterion based on absolute off-diagonal value */
        tmp1 = fabsf(spltol) * tnrm;
        for (i = 0; i < n - 1; i++) {
            eabs = fabsf(E[i]);
            if (eabs <= tmp1) {
                E[i] = 0.0f;
                E2[i] = 0.0f;
                isplit[*nsplit - 1] = i;
                *nsplit = *nsplit + 1;
            }
        }
    } else {
        /* Criterion that guarantees relative accuracy */
        for (i = 0; i < n - 1; i++) {
            eabs = fabsf(E[i]);
            if (eabs <= spltol * sqrtf(fabsf(D[i])) * sqrtf(fabsf(D[i + 1]))) {
                E[i] = 0.0f;
                E2[i] = 0.0f;
                isplit[*nsplit - 1] = i;
                *nsplit = *nsplit + 1;
            }
        }
    }
    isplit[*nsplit - 1] = n - 1;
}
