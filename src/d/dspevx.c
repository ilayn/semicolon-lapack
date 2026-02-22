/**
 * @file dspevx.c
 * @brief DSPEVX computes selected eigenvalues and, optionally, eigenvectors
 *        of a real symmetric matrix A in packed storage.
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "semicolon_lapack_double.h"

/**
 * DSPEVX computes selected eigenvalues and, optionally, eigenvectors
 * of a real symmetric matrix A in packed storage. Eigenvalues/vectors
 * can be selected by specifying either a range of values or a range of
 * indices for the desired eigenvalues.
 *
 * @param[in]     jobz   = 'N': Compute eigenvalues only;
 *                       = 'V': Compute eigenvalues and eigenvectors.
 * @param[in]     range  = 'A': all eigenvalues will be found;
 *                       = 'V': all eigenvalues in the half-open interval (VL,VU]
 *                              will be found;
 *                       = 'I': the IL-th through IU-th eigenvalues will be found.
 * @param[in]     uplo   = 'U': Upper triangle of A is stored;
 *                       = 'L': Lower triangle of A is stored.
 * @param[in]     n      The order of the matrix A. n >= 0.
 * @param[in,out] AP     Double precision array, dimension (n*(n+1)/2).
 *                       On entry, the upper or lower triangle of the symmetric
 *                       matrix A, packed columnwise in a linear array.
 *                       On exit, AP is overwritten by values generated during
 *                       the reduction to tridiagonal form.
 * @param[in]     vl     If range='V', the lower bound of the interval to
 *                       be searched for eigenvalues. vl < vu.
 *                       Not referenced if range = 'A' or 'I'.
 * @param[in]     vu     If range='V', the upper bound of the interval to
 *                       be searched for eigenvalues. vl < vu.
 *                       Not referenced if range = 'A' or 'I'.
 * @param[in]     il     If range='I', the index of the smallest eigenvalue
 *                       to be returned (0-based).
 *                       0 <= il <= iu <= n-1, if n > 0.
 *                       Not referenced if range = 'A' or 'V'.
 * @param[in]     iu     If range='I', the index of the largest eigenvalue
 *                       to be returned (0-based).
 *                       0 <= il <= iu <= n-1, if n > 0.
 *                       Not referenced if range = 'A' or 'V'.
 * @param[in]     abstol The absolute error tolerance for the eigenvalues.
 * @param[out]    m      The total number of eigenvalues found. 0 <= m <= n.
 * @param[out]    W      Double precision array, dimension (n).
 *                       If info = 0, the selected eigenvalues in ascending order.
 * @param[out]    Z      Double precision array, dimension (ldz, max(1,m)).
 *                       If jobz = 'V', contains the orthonormal eigenvectors.
 *                       If jobz = 'N', Z is not referenced.
 * @param[in]     ldz    The leading dimension of the array Z. ldz >= 1, and if
 *                       jobz = 'V', ldz >= max(1,n).
 * @param[out]    work   Double precision array, dimension (8*n).
 * @param[out]    iwork  Integer array, dimension (5*n).
 * @param[out]    ifail  Integer array, dimension (n).
 *                       If jobz = 'V', then if info = 0, the first m elements
 *                       of ifail are zero. If info > 0, then ifail contains
 *                       the indices of the eigenvectors that failed to converge.
 *                       If jobz = 'N', then ifail is not referenced.
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 *                         - > 0: if info = i, then i eigenvectors failed to converge.
 */
void dspevx(const char* jobz, const char* range, const char* uplo,
            const INT n, f64* restrict AP,
            const f64 vl, const f64 vu, const INT il, const INT iu,
            const f64 abstol, INT* m, f64* restrict W,
            f64* restrict Z, const INT ldz,
            f64* restrict work, INT* restrict iwork,
            INT* restrict ifail, INT* info)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    INT alleig, indeig, test, valeig, wantz;
    char order;
    INT i, iinfo, imax, indd, inde, indee, indisp, indiwo, indtau, indwrk;
    INT iscale, itmp1, j, jj, nsplit;
    f64 abstll, anrm, bignum, eps, rmax, rmin, safmin, sigma, smlnum;
    f64 tmp1, vll, vuu;

    *info = 0;
    wantz = (jobz[0] == 'V' || jobz[0] == 'v');
    alleig = (range[0] == 'A' || range[0] == 'a');
    valeig = (range[0] == 'V' || range[0] == 'v');
    indeig = (range[0] == 'I' || range[0] == 'i');

    if (!(wantz || (jobz[0] == 'N' || jobz[0] == 'n'))) {
        *info = -1;
    } else if (!(alleig || valeig || indeig)) {
        *info = -2;
    } else if (!((uplo[0] == 'L' || uplo[0] == 'l') ||
                 (uplo[0] == 'U' || uplo[0] == 'u'))) {
        *info = -3;
    } else if (n < 0) {
        *info = -4;
    } else {
        if (valeig) {
            if (n > 0 && vu <= vl) {
                *info = -7;
            }
        } else if (indeig) {
            if (il < 0 || il > (0 > n - 1 ? 0 : n - 1)) {
                *info = -8;
            } else if (iu < ((n - 1) < il ? (n - 1) : il) || iu > n - 1) {
                *info = -9;
            }
        }
    }
    if (*info == 0) {
        if (ldz < 1 || (wantz && ldz < n)) {
            *info = -14;
        }
    }

    if (*info != 0) {
        xerbla("DSPEVX", -(*info));
        return;
    }

    *m = 0;
    if (n == 0) {
        return;
    }

    if (n == 1) {
        if (alleig || indeig) {
            *m = 1;
            W[0] = AP[0];
        } else {
            if (vl < AP[0] && vu >= AP[0]) {
                *m = 1;
                W[0] = AP[0];
            }
        }
        if (wantz) {
            Z[0] = ONE;
        }
        return;
    }

    safmin = dlamch("Safe minimum");
    eps = dlamch("Precision");
    smlnum = safmin / eps;
    bignum = ONE / smlnum;
    rmin = sqrt(smlnum);
    rmax = sqrt(bignum) < ONE / sqrt(sqrt(safmin)) ?
           sqrt(bignum) : ONE / sqrt(sqrt(safmin));

    iscale = 0;
    abstll = abstol;
    if (valeig) {
        vll = vl;
        vuu = vu;
    } else {
        vll = ZERO;
        vuu = ZERO;
    }
    anrm = dlansp("M", uplo, n, AP, work);
    if (anrm > ZERO && anrm < rmin) {
        iscale = 1;
        sigma = rmin / anrm;
    } else if (anrm > rmax) {
        iscale = 1;
        sigma = rmax / anrm;
    }
    if (iscale == 1) {
        cblas_dscal((n * (n + 1)) / 2, sigma, AP, 1);
        if (abstol > 0) {
            abstll = abstol * sigma;
        }
        if (valeig) {
            vll = vl * sigma;
            vuu = vu * sigma;
        }
    }

    indtau = 0;
    inde = indtau + n;
    indd = inde + n;
    indwrk = indd + n;
    dsptrd(uplo, n, AP, &work[indd], &work[inde], &work[indtau], &iinfo);

    test = 0;
    if (indeig) {
        if (il == 0 && iu == n - 1) {
            test = 1;
        }
    }
    if ((alleig || test) && (abstol <= ZERO)) {
        cblas_dcopy(n, &work[indd], 1, W, 1);
        indee = indwrk + 2 * n;
        if (!wantz) {
            cblas_dcopy(n - 1, &work[inde], 1, &work[indee], 1);
            dsterf(n, W, &work[indee], info);
        } else {
            dopgtr(uplo, n, AP, &work[indtau], Z, ldz, &work[indwrk], &iinfo);
            cblas_dcopy(n - 1, &work[inde], 1, &work[indee], 1);
            dsteqr(jobz, n, W, &work[indee], Z, ldz, &work[indwrk], info);
            if (*info == 0) {
                for (i = 0; i < n; i++) {
                    ifail[i] = 0;
                }
            }
        }
        if (*info == 0) {
            *m = n;
            goto L20;
        }
        *info = 0;
    }

    if (wantz) {
        order = 'B';
    } else {
        order = 'E';
    }
    indisp = n;
    indiwo = indisp + n;
    {
        char order_str[2] = {order, '\0'};
        dstebz(range, order_str, n, vll, vuu, il, iu, abstll,
               &work[indd], &work[inde], m, &nsplit, W,
               &iwork[0], &iwork[indisp], &work[indwrk],
               &iwork[indiwo], info);
    }

    if (wantz) {
        dstein(n, &work[indd], &work[inde], *m, W,
               &iwork[0], &iwork[indisp], Z, ldz,
               &work[indwrk], &iwork[indiwo], ifail, info);

        {
            const char* side = "L";
            const char* trans = "N";
            dopmtr(side, uplo, trans, n, *m, AP, &work[indtau], Z, ldz,
                   &work[indwrk], &iinfo);
        }
    }

L20:
    if (iscale == 1) {
        if (*info == 0) {
            imax = *m;
        } else {
            imax = *info - 1;
        }
        cblas_dscal(imax, ONE / sigma, W, 1);
    }

    if (wantz) {
        for (j = 0; j < *m - 1; j++) {
            i = 0;
            tmp1 = W[j];
            for (jj = j + 1; jj < *m; jj++) {
                if (W[jj] < tmp1) {
                    i = jj;
                    tmp1 = W[jj];
                }
            }

            if (i != 0) {
                itmp1 = iwork[i];
                W[i] = W[j];
                iwork[i] = iwork[j];
                W[j] = tmp1;
                iwork[j] = itmp1;
                cblas_dswap(n, &Z[i * ldz], 1, &Z[j * ldz], 1);
                if (*info != 0) {
                    itmp1 = ifail[i];
                    ifail[i] = ifail[j];
                    ifail[j] = itmp1;
                }
            }
        }
    }
}
