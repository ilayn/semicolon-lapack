/**
 * @file cggbak.c
 * @brief CGGBAK forms the right or left eigenvectors of a complex generalized eigenvalue problem.
 */

#include <complex.h>
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
 * @param[in]     side    = 'R': V contains right eigenvectors;
 *                        = 'L': V contains left eigenvectors.
 * @param[in]     n       The number of rows of the matrix V. n >= 0.
 * @param[in]     ilo     See ihi.
 * @param[in]     ihi     The integers ILO and IHI determined by CGGBAL.
 * @param[in]     lscale  Array of dimension (n). Details of permutations and/or
 *                        scaling factors applied to left side, as returned by CGGBAL.
 * @param[in]     rscale  Array of dimension (n). Details of permutations and/or
 *                        scaling factors applied to right side, as returned by CGGBAL.
 * @param[in]     m       The number of columns of the matrix V. m >= 0.
 * @param[in,out] V       Array of dimension (ldv, m). On entry, the matrix of
 *                        eigenvectors to be transformed. On exit, V is overwritten
 *                        by the transformed eigenvectors.
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
    } else if (ilo < 1) {
        *info = -4;
    } else if (n == 0 && ihi == 0 && ilo != 1) {
        *info = -4;
    } else if (n > 0 && (ihi < ilo || ihi > (1 > n ? 1 : n))) {
        *info = -5;
    } else if (n == 0 && ilo == 1 && ihi != 0) {
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

    if (n == 0)
        return;
    if (m == 0)
        return;
    if (job[0] == 'N' || job[0] == 'n')
        return;

    if (ilo == ihi)
        goto L30;

    if (job[0] == 'S' || job[0] == 's' || job[0] == 'B' || job[0] == 'b') {
        if (rightv) {
            for (i = ilo - 1; i < ihi; i++) {
                cblas_csscal(m, rscale[i], &V[i], ldv);
            }
        }

        if (leftv) {
            for (i = ilo - 1; i < ihi; i++) {
                cblas_csscal(m, lscale[i], &V[i], ldv);
            }
        }
    }

L30:
    if (job[0] == 'P' || job[0] == 'p' || job[0] == 'B' || job[0] == 'b') {
        if (rightv) {
            if (ilo == 1)
                goto L50;

            for (i = ilo - 2; i >= 0; i--) {
                k = (int)rscale[i];
                if (k == i + 1)
                    continue;
                cblas_cswap(m, &V[i], ldv, &V[k - 1], ldv);
            }

        L50:
            if (ihi == n)
                goto L70;
            for (i = ihi; i < n; i++) {
                k = (int)rscale[i];
                if (k == i + 1)
                    continue;
                cblas_cswap(m, &V[i], ldv, &V[k - 1], ldv);
            }
        }

    L70:
        if (leftv) {
            if (ilo == 1)
                goto L90;

            for (i = ilo - 2; i >= 0; i--) {
                k = (int)lscale[i];
                if (k == i + 1)
                    continue;
                cblas_cswap(m, &V[i], ldv, &V[k - 1], ldv);
            }

        L90:
            if (ihi == n)
                return;
            for (i = ihi; i < n; i++) {
                k = (int)lscale[i];
                if (k == i + 1)
                    continue;
                cblas_cswap(m, &V[i], ldv, &V[k - 1], ldv);
            }
        }
    }
}
