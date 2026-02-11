/**
 * @file slarrc.c
 * @brief SLARRC computes the number of eigenvalues of the symmetric tridiagonal matrix.
 */

#include "semicolon_lapack_single.h"

/**
 * SLARRC finds the number of eigenvalues of the symmetric tridiagonal
 * matrix T that are in the interval (VL,VU] if JOBT = 'T', and of
 * L D L^T if JOBT = 'L'.
 *
 * @param[in]     jobt    CHARACTER*1.
 *                        = 'T': Compute Sturm count for matrix T.
 *                        = 'L': Compute Sturm count for matrix L D L^T.
 * @param[in]     n       The order of the matrix. N > 0.
 * @param[in]     vl      The lower bound for the eigenvalues.
 * @param[in]     vu      The upper bound for the eigenvalues.
 * @param[in]     D       Double precision array, dimension (N).
 *                        JOBT = 'T': The N diagonal elements of the tridiagonal matrix T.
 *                        JOBT = 'L': The N diagonal elements of the diagonal matrix D.
 * @param[in]     E       Double precision array, dimension (N).
 *                        JOBT = 'T': The N-1 offdiagonal elements of the matrix T.
 *                        JOBT = 'L': The N-1 offdiagonal elements of the matrix L.
 * @param[in]     pivmin  The minimum pivot in the Sturm sequence for T.
 * @param[out]    eigcnt  The number of eigenvalues in the interval (VL,VU].
 * @param[out]    lcnt    The left negcount of the interval.
 * @param[out]    rcnt    The right negcount of the interval.
 * @param[out]    info
 *                         - = 0: successful exit.
 */
void slarrc(const char* jobt, const int n, const float vl, const float vu,
            const float* const restrict D, const float* const restrict E,
            const float pivmin, int* eigcnt, int* lcnt, int* rcnt, int* info)
{
    int i;
    int matt;
    float lpivot, rpivot, sl, su, tmp, tmp2;

    (void)pivmin;

    *info = 0;
    *lcnt = 0;
    *rcnt = 0;
    *eigcnt = 0;

    /* Quick return if possible */
    if (n <= 0) {
        return;
    }

    matt = (jobt[0] == 'T' || jobt[0] == 't');

    if (matt) {
        /* Sturm sequence count on T */
        lpivot = D[0] - vl;
        rpivot = D[0] - vu;
        if (lpivot <= 0.0f) {
            *lcnt = *lcnt + 1;
        }
        if (rpivot <= 0.0f) {
            *rcnt = *rcnt + 1;
        }
        for (i = 0; i < n - 1; i++) {
            tmp = E[i] * E[i];
            lpivot = (D[i + 1] - vl) - tmp / lpivot;
            rpivot = (D[i + 1] - vu) - tmp / rpivot;
            if (lpivot <= 0.0f) {
                *lcnt = *lcnt + 1;
            }
            if (rpivot <= 0.0f) {
                *rcnt = *rcnt + 1;
            }
        }
    } else {
        /* Sturm sequence count on L D L^T */
        sl = -vl;
        su = -vu;
        for (i = 0; i < n - 1; i++) {
            lpivot = D[i] + sl;
            rpivot = D[i] + su;
            if (lpivot <= 0.0f) {
                *lcnt = *lcnt + 1;
            }
            if (rpivot <= 0.0f) {
                *rcnt = *rcnt + 1;
            }
            tmp = E[i] * D[i] * E[i];

            tmp2 = tmp / lpivot;
            if (tmp2 == 0.0f) {
                sl = tmp - vl;
            } else {
                sl = sl * tmp2 - vl;
            }

            tmp2 = tmp / rpivot;
            if (tmp2 == 0.0f) {
                su = tmp - vu;
            } else {
                su = su * tmp2 - vu;
            }
        }
        lpivot = D[n - 1] + sl;
        rpivot = D[n - 1] + su;
        if (lpivot <= 0.0f) {
            *lcnt = *lcnt + 1;
        }
        if (rpivot <= 0.0f) {
            *rcnt = *rcnt + 1;
        }
    }
    *eigcnt = *rcnt - *lcnt;
}
