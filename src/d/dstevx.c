/**
 * @file dstevx.c
 * @brief DSTEVX computes selected eigenvalues and, optionally, eigenvectors
 *        of a real symmetric tridiagonal matrix A.
 */

#include <math.h>
#include <cblas.h>
#include "semicolon_lapack_double.h"

/**
 * DSTEVX computes selected eigenvalues and, optionally, eigenvectors
 * of a real symmetric tridiagonal matrix A. Eigenvalues and
 * eigenvectors can be selected by specifying either a range of values
 * or a range of indices for the desired eigenvalues.
 *
 * @param[in]     jobz   = 'N': Compute eigenvalues only;
 *                         = 'V': Compute eigenvalues and eigenvectors.
 * @param[in]     range  = 'A': all eigenvalues will be found.
 *                         = 'V': all eigenvalues in the half-open interval
 *                                (vl, vu] will be found.
 *                         = 'I': the il-th through iu-th eigenvalues will be
 *                                found.
 * @param[in]     n      The order of the matrix. n >= 0.
 * @param[in,out] D      Double precision array, dimension (n).
 *                       On entry, the n diagonal elements of the tridiagonal
 *                       matrix A. On exit, D may be multiplied by a constant
 *                       factor chosen to avoid over/underflow.
 * @param[in,out] E      Double precision array, dimension (max(1,n-1)).
 *                       On entry, the (n-1) subdiagonal elements. On exit, E
 *                       may be multiplied by a constant factor.
 * @param[in]     vl     If range='V', the lower bound of the interval.
 * @param[in]     vu     If range='V', the upper bound of the interval.
 * @param[in]     il     If range='I', the index of the smallest eigenvalue
 *                       to be returned (1-based).
 * @param[in]     iu     If range='I', the index of the largest eigenvalue
 *                       to be returned (1-based).
 * @param[in]     abstol The absolute error tolerance for the eigenvalues.
 * @param[out]    m      The total number of eigenvalues found.
 * @param[out]    W      Double precision array, dimension (n). The first m
 *                       elements contain the selected eigenvalues in ascending
 *                       order.
 * @param[out]    Z      Double precision array, dimension (ldz, max(1,m)).
 *                       If jobz='V', the first m columns contain the
 *                       orthonormal eigenvectors.
 * @param[in]     ldz    The leading dimension of Z. ldz >= 1, and if
 *                       jobz='V', ldz >= max(1,n).
 * @param[out]    work   Double precision array, dimension (5*n).
 * @param[out]    iwork  Integer array, dimension (5*n).
 * @param[out]    ifail  Integer array, dimension (n). If jobz='V', on normal
 *                       exit all elements are zero. If info > 0, contains
 *                       indices of eigenvectors that failed to converge.
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal
 *                           value
 *                         - > 0: if info = i, then i eigenvectors failed to
 *                           converge.
 */
void dstevx(const char* jobz, const char* range, const int n,
            f64* const restrict D, f64* const restrict E,
            const f64 vl, const f64 vu,
            const int il, const int iu, const f64 abstol,
            int* m, f64* const restrict W,
            f64* const restrict Z, const int ldz,
            f64* const restrict work, int* const restrict iwork,
            int* const restrict ifail, int* info)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    /* Test the input parameters. */
    int wantz = (jobz[0] == 'V' || jobz[0] == 'v');
    int alleig = (range[0] == 'A' || range[0] == 'a');
    int valeig = (range[0] == 'V' || range[0] == 'v');
    int indeig = (range[0] == 'I' || range[0] == 'i');

    *info = 0;
    if (!wantz && !(jobz[0] == 'N' || jobz[0] == 'n')) {
        *info = -1;
    } else if (!alleig && !valeig && !indeig) {
        *info = -2;
    } else if (n < 0) {
        *info = -3;
    } else {
        if (valeig) {
            if (n > 0 && vu <= vl)
                *info = -7;
        } else if (indeig) {
            if (il < 1 || il > (1 > n ? 1 : n)) {
                *info = -8;
            } else if (iu < (n < il ? n : il) || iu > n) {
                *info = -9;
            }
        }
    }
    if (*info == 0) {
        if (ldz < 1 || (wantz && ldz < n))
            *info = -14;
    }

    if (*info != 0) {
        xerbla("DSTEVX", -(*info));
        return;
    }

    /* Quick return if possible */
    *m = 0;
    if (n == 0)
        return;

    if (n == 1) {
        if (alleig || indeig) {
            *m = 1;
            W[0] = D[0];
        } else {
            if (vl < D[0] && vu >= D[0]) {
                *m = 1;
                W[0] = D[0];
            }
        }
        if (wantz)
            Z[0] = ONE;
        return;
    }

    /* Get machine constants. */
    f64 safmin = dlamch("S");
    f64 eps = dlamch("P");
    f64 smlnum = safmin / eps;
    f64 bignum = ONE / smlnum;
    f64 rmin = sqrt(smlnum);
    f64 rmax_val = sqrt(bignum);
    f64 rmax2 = ONE / sqrt(sqrt(safmin));
    f64 rmax = rmax_val < rmax2 ? rmax_val : rmax2;

    /* Scale matrix to allowable range, if necessary. */
    int iscale = 0;
    f64 vll = ZERO, vuu = ZERO;
    if (valeig) {
        vll = vl;
        vuu = vu;
    }
    f64 tnrm = dlanst("M", n, D, E);
    f64 sigma = ZERO;
    if (tnrm > ZERO && tnrm < rmin) {
        iscale = 1;
        sigma = rmin / tnrm;
    } else if (tnrm > rmax) {
        iscale = 1;
        sigma = rmax / tnrm;
    }
    if (iscale == 1) {
        cblas_dscal(n, sigma, D, 1);
        cblas_dscal(n - 1, sigma, E, 1);
        if (valeig) {
            vll = vl * sigma;
            vuu = vu * sigma;
        }
    }

    /* If all eigenvalues are desired and ABSTOL is less than or equal to
     * zero, then call DSTERF or DSTEQR. If this fails for some eigenvalue,
     * then try DSTEBZ. */
    int test = 0;
    if (indeig) {
        if (il == 1 && iu == n)
            test = 1;
    }
    if ((alleig || test) && abstol <= ZERO) {
        cblas_dcopy(n, D, 1, W, 1);
        cblas_dcopy(n - 1, E, 1, work, 1);
        int indwrk = n;
        if (!wantz) {
            dsterf(n, W, work, info);
        } else {
            dsteqr("I", n, W, work, Z, ldz, &work[indwrk], info);
            if (*info == 0) {
                for (int i = 0; i < n; i++)
                    ifail[i] = 0;
            }
        }
        if (*info == 0) {
            *m = n;
            goto rescale;
        }
        *info = 0;
    }

    /* Otherwise, call DSTEBZ and, if eigenvectors are desired, DSTEIN. */
    const char* order_str;
    if (wantz) {
        order_str = "B";
    } else {
        order_str = "E";
    }

    int nsplit;
    dstebz(range, order_str, n, vll, vuu, il, iu, abstol, D, E,
           m, &nsplit, W, iwork, &iwork[n], &work[0], &iwork[2 * n], info);

    if (wantz) {
        dstein(n, D, E, *m, W, iwork, &iwork[n],
               Z, ldz, &work[0], &iwork[2 * n], ifail, info);
    }

    /* If matrix was scaled, then rescale eigenvalues appropriately. */
rescale:
    if (iscale == 1) {
        int imax;
        if (*info == 0) {
            imax = *m;
        } else {
            imax = *info - 1;
        }
        cblas_dscal(imax, ONE / sigma, W, 1);
    }

    /* If eigenvalues are not in order, then sort them, along with
     * eigenvectors. */
    if (wantz) {
        for (int j = 0; j < *m - 1; j++) {
            int imin = 0;
            f64 tmp1 = W[j];
            for (int jj = j + 1; jj < *m; jj++) {
                if (W[jj] < tmp1) {
                    imin = jj;
                    tmp1 = W[jj];
                }
            }

            if (imin != 0) {
                W[imin] = W[j];
                W[j] = tmp1;
                cblas_dswap(n, &Z[0 + imin * ldz], 1, &Z[0 + j * ldz], 1);
                if (*info != 0) {
                    int itmp1 = ifail[imin];
                    ifail[imin] = ifail[j];
                    ifail[j] = itmp1;
                }
            }
        }
    }
}
