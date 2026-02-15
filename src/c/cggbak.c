/**
 * @file cggbak.c
 * @brief CGGBAK forms the right or left eigenvectors of a complex generalized eigenvalue problem.
 */

#include <cblas.h>
#include "semicolon_lapack_complex_single.h"

/**
 * CGGBAK forms the right or left eigenvectors of a complex generalized
 * eigenvalue problem A*x = lambda*B*x, by backward transformation on
 * the computed eigenvectors of the balanced pair of matrices output by
 * CGGBAL.
 *
 * @param[in]     job     Specifies the type of backward transformation required:
 *                        = 'N': do nothing, return immediately;
 *                        = 'P': do backward transformation for permutation only;
 *                        = 'S': do backward transformation for scaling only;
 *                        = 'B': do backward transformations for both permutation and scaling.
 *                        JOB must be the same as the argument JOB supplied to CGGBAL.
 * @param[in]     side    = 'R': V contains right eigenvectors;
 *                        = 'L': V contains left eigenvectors.
 * @param[in]     n       The number of rows of the matrix V. n >= 0.
 * @param[in]     ilo     The integers ILO and IHI determined by CGGBAL.
 *                        0 <= ILO <= IHI <= N-1, if N > 0; ILO=0 and IHI=-1, if N=0.
 * @param[in]     ihi     See ILO.
 * @param[in]     lscale  Array of dimension (n). Details of the permutations and/or
 *                        scaling factors applied to the left side of A and B, as
 *                        returned by CGGBAL.
 * @param[in]     rscale  Array of dimension (n). Details of the permutations and/or
 *                        scaling factors applied to the right side of A and B, as
 *                        returned by CGGBAL.
 * @param[in]     m       The number of columns of the matrix V. m >= 0.
 * @param[in,out] V       Array of dimension (ldv, m). On entry, the matrix of right
 *                        or left eigenvectors to be transformed, as returned by CTGEVC.
 *                        On exit, V is overwritten by the transformed eigenvectors.
 * @param[in]     ldv     The leading dimension of V. ldv >= max(1,n).
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value.
 */
void cggbak(
    const char* job,
    const char* side,
    const int n,
    const int ilo,
    const int ihi,
    const f32* restrict lscale,
    const f32* restrict rscale,
    const int m,
    c64* restrict V,
    const int ldv,
    int* info)
{
    int leftv, rightv;
    int i, k;

    /* Test the input parameters */

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
        *info = -8;
    } else if (ldv < (1 > n ? 1 : n)) {
        *info = -10;
    }
    if (*info != 0) {
        xerbla("CGGBAK", -(*info));
        return;
    }

    /* Quick return if possible */

    if (n == 0) return;
    if (m == 0) return;
    if (job[0] == 'N' || job[0] == 'n') return;

    if (ilo != ihi) {

        /* Backward balance */

        if (job[0] == 'S' || job[0] == 's' || job[0] == 'B' || job[0] == 'b') {

            /* Backward transformation on right eigenvectors */

            if (rightv) {
                for (i = ilo; i <= ihi; i++) {
                    cblas_csscal(m, rscale[i], &V[i], ldv);
                }
            }

            /* Backward transformation on left eigenvectors */

            if (leftv) {
                for (i = ilo; i <= ihi; i++) {
                    cblas_csscal(m, lscale[i], &V[i], ldv);
                }
            }
        }
    }

    /* Backward permutation */

    if (job[0] == 'P' || job[0] == 'p' || job[0] == 'B' || job[0] == 'b') {

        /* Backward permutation on right eigenvectors */

        if (rightv) {
            for (i = ilo - 1; i >= 0; i--) {
                k = (int)rscale[i];
                if (k == i) continue;
                cblas_cswap(m, &V[i], ldv, &V[k], ldv);
            }
            for (i = ihi + 1; i < n; i++) {
                k = (int)rscale[i];
                if (k == i) continue;
                cblas_cswap(m, &V[i], ldv, &V[k], ldv);
            }
        }

        /* Backward permutation on left eigenvectors */

        if (leftv) {
            for (i = ilo - 1; i >= 0; i--) {
                k = (int)lscale[i];
                if (k == i) continue;
                cblas_cswap(m, &V[i], ldv, &V[k], ldv);
            }
            for (i = ihi + 1; i < n; i++) {
                k = (int)lscale[i];
                if (k == i) continue;
                cblas_cswap(m, &V[i], ldv, &V[k], ldv);
            }
        }
    }
}
