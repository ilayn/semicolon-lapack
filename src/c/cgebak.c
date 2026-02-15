/**
 * @file cgebak.c
 * @brief CGEBAK forms the right or left eigenvectors of a complex general
 *        matrix by backward transformation on the computed eigenvectors of the
 *        balanced matrix output by CGEBAL.
 */

#include "semicolon_lapack_complex_single.h"
#include <complex.h>
#include <cblas.h>

/**
 * CGEBAK forms the right or left eigenvectors of a complex general matrix
 * by backward transformation on the computed eigenvectors of the
 * balanced matrix output by CGEBAL.
 *
 * @param[in]     job    Specifies the type of backward transformation required:
 *                       = 'N': do nothing, return immediately;
 *                       = 'P': do backward transformation for permutation only;
 *                       = 'S': do backward transformation for scaling only;
 *                       = 'B': do backward transformations for both permutation and
 *                              scaling.
 *                       JOB must be the same as the argument JOB supplied to CGEBAL.
 * @param[in]     side   = 'R': V contains right eigenvectors;
 *                       = 'L': V contains left eigenvectors.
 * @param[in]     n      The number of rows of the matrix V. N >= 0.
 * @param[in]     ilo    The integers ILO and IHI determined by CGEBAL.
 *                       0 <= ILO <= IHI <= N-1, if N > 0; ILO=0 and IHI=-1, if N=0.
 *                       Uses 0-based indexing.
 * @param[in]     ihi    See ILO. Uses 0-based indexing.
 * @param[in]     scale  Single precision array, dimension (N).
 *                       Details of the permutation and scaling factors, as returned
 *                       by CGEBAL. Permutation indices are 0-based.
 * @param[in]     m      The number of columns of the matrix V. M >= 0.
 * @param[in,out] V      Complex*16 array, dimension (LDV,M).
 *                       On entry, the matrix of right or left eigenvectors to be
 *                       transformed, as returned by CHSEIN or CTREVC.
 *                       On exit, V is overwritten by the transformed eigenvectors.
 * @param[in]     ldv    The leading dimension of the array V. LDV >= max(1,N).
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if INFO = -i, the i-th argument had an illegal value.
 */
void cgebak(const char* job, const char* side, const int n, const int ilo,
            const int ihi, const f32* scale, const int m, c64* V,
            const int ldv, int* info)
{
    /* Constants */
    const f32 ONE = 1.0f;

    /* Local variables */
    int leftv, rightv;
    int i, ii, k;
    f32 s;

    /* Decode and test the input parameters */
    rightv = (side[0] == 'R' || side[0] == 'r');
    leftv = (side[0] == 'L' || side[0] == 'l');

    *info = 0;
    if (!(job[0] == 'N' || job[0] == 'n') &&
        !(job[0] == 'P' || job[0] == 'p') &&
        !(job[0] == 'S' || job[0] == 's') &&
        !(job[0] == 'B' || job[0] == 'b')) {
        *info = -1;
    } else if (!rightv && !leftv) {
        *info = -2;
    } else if (n < 0) {
        *info = -3;
    } else if (ilo < 0 || (n > 0 && ilo > n - 1)) {
        *info = -4;
    } else if ((n > 0 && ihi < ilo) || ihi > n - 1) {
        *info = -5;
    } else if (m < 0) {
        *info = -7;
    } else if (ldv < (1 > n ? 1 : n)) {
        *info = -9;
    }
    if (*info != 0) {
        xerbla("CGEBAK", -(*info));
        return;
    }

    /* Quick return if possible */
    if (n == 0) return;
    if (m == 0) return;
    if (job[0] == 'N' || job[0] == 'n') return;

    if (ilo == ihi) goto L30;

    /* Backward balance */
    if (job[0] == 'S' || job[0] == 's' || job[0] == 'B' || job[0] == 'b') {
        if (rightv) {
            for (i = ilo; i <= ihi; i++) {
                s = scale[i];
                cblas_csscal(m, s, &V[i + 0 * ldv], ldv);
            }
        }

        if (leftv) {
            for (i = ilo; i <= ihi; i++) {
                s = ONE / scale[i];
                cblas_csscal(m, s, &V[i + 0 * ldv], ldv);
            }
        }
    }

    /* Backward permutation */
    /*
     * For I = ILO-1 step -1 until 0,
     *         IHI+1 step 1 until N-1 do --
     */
L30:
    if (job[0] == 'P' || job[0] == 'p' || job[0] == 'B' || job[0] == 'b') {
        if (rightv) {
            for (ii = 0; ii < n; ii++) {
                i = ii;
                /* Skip indices in range [ilo, ihi] */
                if (i >= ilo && i <= ihi) continue;
                if (i < ilo) {
                    /* i = ilo - 1 - ii */
                    i = ilo - 1 - ii;
                    if (i < 0) continue;
                }
                /* scale stores 0-based indices */
                k = (int)scale[i];
                if (k == i) continue;
                cblas_cswap(m, &V[i + 0 * ldv], ldv, &V[k + 0 * ldv], ldv);
            }
        }

        if (leftv) {
            for (ii = 0; ii < n; ii++) {
                i = ii;
                /* Skip indices in range [ilo, ihi] */
                if (i >= ilo && i <= ihi) continue;
                if (i < ilo) {
                    i = ilo - 1 - ii;
                    if (i < 0) continue;
                }
                /* scale stores 0-based indices */
                k = (int)scale[i];
                if (k == i) continue;
                cblas_cswap(m, &V[i + 0 * ldv], ldv, &V[k + 0 * ldv], ldv);
            }
        }
    }
}
