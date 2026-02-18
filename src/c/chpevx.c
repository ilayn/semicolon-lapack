/**
 * @file chpevx.c
 * @brief CHPEVX computes selected eigenvalues and, optionally, eigenvectors
 *        of a complex Hermitian matrix in packed storage.
 */

#include <math.h>
#include <complex.h>
#include <cblas.h>
#include "semicolon_lapack_complex_single.h"

/**
 * CHPEVX computes selected eigenvalues and, optionally, eigenvectors
 * of a complex Hermitian matrix A in packed storage.
 * Eigenvalues/vectors can be selected by specifying either a range of
 * values or a range of indices for the desired eigenvalues.
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
 * @param[in,out] AP     Complex array, dimension (n*(n+1)/2).
 *                       On entry, the upper or lower triangle of the Hermitian
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
 * @param[out]    W      Single precision array, dimension (n).
 *                       If info = 0, the selected eigenvalues in ascending order.
 * @param[out]    Z      Complex array, dimension (ldz, max(1,m)).
 *                       If jobz = 'V', contains the orthonormal eigenvectors.
 *                       If jobz = 'N', Z is not referenced.
 * @param[in]     ldz    The leading dimension of the array Z. ldz >= 1, and if
 *                       jobz = 'V', ldz >= max(1,n).
 * @param[out]    work   Complex array, dimension (2*n).
 * @param[out]    rwork  Single precision array, dimension (7*n).
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
 *                           Their indices are stored in array IFAIL.
 */
void chpevx(const char* jobz, const char* range, const char* uplo,
            const int n, c64* restrict AP,
            const f32 vl, const f32 vu, const int il, const int iu,
            const f32 abstol, int* m, f32* restrict W,
            c64* restrict Z, const int ldz,
            c64* restrict work,
            f32* restrict rwork,
            int* restrict iwork,
            int* restrict ifail,
            int* info)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;
    const c64 CONE = CMPLXF(1.0f, 0.0f);

    int alleig, indeig, test, valeig, wantz;
    int i, iinfo, imax, indd, inde, indee, indisp, indiwk, indrwk;
    int indtau, indwrk, iscale, itmp1, j, jj, nsplit;
    f32 abstll, anrm, bignum, eps, rmax, rmin, safmin, sigma, smlnum;
    f32 tmp1, vll, vuu;

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
        xerbla("CHPEVX", -(*info));
        return;
    }

    *m = 0;
    if (n == 0) {
        return;
    }

    if (n == 1) {
        if (alleig || indeig) {
            *m = 1;
            W[0] = crealf(AP[0]);
        } else {
            if (vl < crealf(AP[0]) && vu >= crealf(AP[0])) {
                *m = 1;
                W[0] = crealf(AP[0]);
            }
        }
        if (wantz) {
            Z[0] = CONE;
        }
        return;
    }

    /* Get machine constants. */

    safmin = slamch("Safe minimum");
    eps = slamch("Precision");
    smlnum = safmin / eps;
    bignum = ONE / smlnum;
    rmin = sqrtf(smlnum);
    rmax = sqrtf(bignum) < ONE / sqrtf(sqrtf(safmin)) ?
           sqrtf(bignum) : ONE / sqrtf(sqrtf(safmin));

    /* Scale matrix to allowable range, if necessary. */

    iscale = 0;
    abstll = abstol;
    if (valeig) {
        vll = vl;
        vuu = vu;
    } else {
        vll = ZERO;
        vuu = ZERO;
    }
    anrm = clanhp("M", uplo, n, AP, rwork);
    if (anrm > ZERO && anrm < rmin) {
        iscale = 1;
        sigma = rmin / anrm;
    } else if (anrm > rmax) {
        iscale = 1;
        sigma = rmax / anrm;
    }
    if (iscale == 1) {
        cblas_csscal((n * (n + 1)) / 2, sigma, AP, 1);
        if (abstol > 0) {
            abstll = abstol * sigma;
        }
        if (valeig) {
            vll = vl * sigma;
            vuu = vu * sigma;
        }
    }

    /* Call CHPTRD to reduce Hermitian packed matrix to tridiagonal form. */

    indd = 0;
    inde = indd + n;
    indrwk = inde + n;
    indtau = 0;
    indwrk = indtau + n;
    chptrd(uplo, n, AP, &rwork[indd], &rwork[inde], &work[indtau], &iinfo);

    /*
     * If all eigenvalues are desired and ABSTOL is less than or equal
     * to zero, then call SSTERF or CUPGTR and CSTEQR. If this fails
     * for some eigenvalue, then try SSTEBZ.
     */

    test = 0;
    if (indeig) {
        if (il == 0 && iu == n - 1) {
            test = 1;
        }
    }
    if ((alleig || test) && (abstol <= ZERO)) {
        cblas_scopy(n, &rwork[indd], 1, W, 1);
        indee = indrwk + 2 * n;
        if (!wantz) {
            cblas_scopy(n - 1, &rwork[inde], 1, &rwork[indee], 1);
            ssterf(n, W, &rwork[indee], info);
        } else {
            cupgtr(uplo, n, AP, &work[indtau], Z, ldz, &work[indwrk], &iinfo);
            cblas_scopy(n - 1, &rwork[inde], 1, &rwork[indee], 1);
            csteqr(jobz, n, W, &rwork[indee], Z, ldz, &rwork[indrwk], info);
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

    /* Otherwise, call SSTEBZ and, if eigenvectors are desired, CSTEIN. */

    const char* order;
    if (wantz) {
        order = "B";
    } else {
        order = "E";
    }
    indisp = n;
    indiwk = indisp + n;
    sstebz(range, order, n, vll, vuu, il, iu, abstll,
           &rwork[indd], &rwork[inde], m, &nsplit, W,
           &iwork[0], &iwork[indisp], &rwork[indrwk],
           &iwork[indiwk], info);

    if (wantz) {
        cstein(n, &rwork[indd], &rwork[inde], *m, W,
               &iwork[0], &iwork[indisp], Z, ldz,
               &rwork[indrwk], &iwork[indiwk], ifail, info);

        /* Apply unitary matrix used in reduction to tridiagonal
         * form to eigenvectors returned by CSTEIN. */

        indwrk = indtau + n;
        cupmtr("L", uplo, "N", n, *m, AP, &work[indtau], Z, ldz,
               &work[indwrk], &iinfo);
    }

    /* If matrix was scaled, then rescale eigenvalues appropriately. */

L20:
    if (iscale == 1) {
        if (*info == 0) {
            imax = *m;
        } else {
            imax = *info - 1;
        }
        cblas_sscal(imax, ONE / sigma, W, 1);
    }

    /* If eigenvalues are not in order, then sort them, along with
     * eigenvectors. */

    if (wantz) {
        for (j = 0; j < *m - 1; j++) {
            i = -1;
            tmp1 = W[j];
            for (jj = j + 1; jj < *m; jj++) {
                if (W[jj] < tmp1) {
                    i = jj;
                    tmp1 = W[jj];
                }
            }

            if (i >= 0) {
                itmp1 = iwork[i];
                W[i] = W[j];
                iwork[i] = iwork[j];
                W[j] = tmp1;
                iwork[j] = itmp1;
                cblas_cswap(n, &Z[i * ldz], 1, &Z[j * ldz], 1);
                if (*info != 0) {
                    itmp1 = ifail[i];
                    ifail[i] = ifail[j];
                    ifail[j] = itmp1;
                }
            }
        }
    }
}
