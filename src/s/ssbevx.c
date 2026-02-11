/**
 * @file ssbevx.c
 * @brief SSBEVX computes selected eigenvalues and eigenvectors of a symmetric band matrix.
 */

#include <math.h>
#include <cblas.h>
#include "semicolon_lapack_single.h"

/**
 * SSBEVX computes selected eigenvalues and, optionally, eigenvectors
 * of a real symmetric band matrix A. Eigenvalues and eigenvectors can
 * be selected by specifying either a range of values or a range of
 * indices for the desired eigenvalues.
 *
 * @param[in]     jobz   = 'N': Compute eigenvalues only
 *                        = 'V': Compute eigenvalues and eigenvectors
 * @param[in]     range  = 'A': all eigenvalues
 *                        = 'V': eigenvalues in (vl,vu]
 *                        = 'I': eigenvalues il through iu
 * @param[in]     uplo   = 'U': Upper triangle of A is stored
 *                        = 'L': Lower triangle of A is stored
 * @param[in]     n      The order of the matrix A. n >= 0.
 * @param[in]     kd     The number of super-diagonals or sub-diagonals. kd >= 0.
 * @param[in,out] AB     The banded matrix A. Array of dimension (ldab, n).
 * @param[in]     ldab   The leading dimension of AB. ldab >= kd+1.
 * @param[out]    Q      If jobz='V', the orthogonal matrix for reduction.
 * @param[in]     ldq    The leading dimension of Q. ldq >= max(1,n) if jobz='V'.
 * @param[in]     vl     Lower bound of interval (if range='V').
 * @param[in]     vu     Upper bound of interval (if range='V').
 * @param[in]     il     Index of smallest eigenvalue (if range='I').
 * @param[in]     iu     Index of largest eigenvalue (if range='I').
 * @param[in]     abstol Absolute error tolerance for eigenvalues.
 * @param[out]    m      The total number of eigenvalues found.
 * @param[out]    W      The selected eigenvalues in ascending order.
 * @param[out]    Z      If jobz='V', the eigenvectors. Array of dimension (ldz, max(1,m)).
 * @param[in]     ldz    The leading dimension of Z. ldz >= 1, and >= n if jobz='V'.
 * @param[out]    work   Workspace array of dimension (7*n).
 * @param[out]    iwork  Integer workspace array of dimension (5*n).
 * @param[out]    ifail  If jobz='V', indices of eigenvectors that failed to converge.
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 *                         - > 0: if info = i, then i eigenvectors failed to converge
 */
void ssbevx(
    const char* jobz,
    const char* range,
    const char* uplo,
    const int n,
    const int kd,
    float* const restrict AB,
    const int ldab,
    float* const restrict Q,
    const int ldq,
    const float vl,
    const float vu,
    const int il,
    const int iu,
    const float abstol,
    int* m,
    float* const restrict W,
    float* const restrict Z,
    const int ldz,
    float* const restrict work,
    int* const restrict iwork,
    int* const restrict ifail,
    int* info)
{
    const float ZERO = 0.0f;
    const float ONE = 1.0f;

    int alleig, indeig, lower, test, valeig, wantz;
    char order;
    int i, iinfo, imax, indd, inde, indee, indibl, indisp, indiwo, indwrk;
    int iscale, itmp1, j, jj, nsplit;
    float abstll, anrm, bignum, eps, rmax, rmin, safmin, sigma, smlnum, tmp1, vll, vuu;

    wantz = (jobz[0] == 'V' || jobz[0] == 'v');
    alleig = (range[0] == 'A' || range[0] == 'a');
    valeig = (range[0] == 'V' || range[0] == 'v');
    indeig = (range[0] == 'I' || range[0] == 'i');
    lower = (uplo[0] == 'L' || uplo[0] == 'l');

    *info = 0;
    if (!wantz && !(jobz[0] == 'N' || jobz[0] == 'n')) {
        *info = -1;
    } else if (!alleig && !valeig && !indeig) {
        *info = -2;
    } else if (!lower && !(uplo[0] == 'U' || uplo[0] == 'u')) {
        *info = -3;
    } else if (n < 0) {
        *info = -4;
    } else if (kd < 0) {
        *info = -5;
    } else if (ldab < kd + 1) {
        *info = -7;
    } else if (wantz && ldq < (1 > n ? 1 : n)) {
        *info = -9;
    } else {
        if (valeig) {
            if (n > 0 && vu <= vl)
                *info = -11;
        } else if (indeig) {
            if (il < 1 || il > (1 > n ? 1 : n)) {
                *info = -12;
            } else if (iu < (n < il ? n : il) || iu > n) {
                *info = -13;
            }
        }
    }
    if (*info == 0) {
        if (ldz < 1 || (wantz && ldz < n))
            *info = -18;
    }

    if (*info != 0) {
        xerbla("SSBEVX", -(*info));
        return;
    }

    *m = 0;
    if (n == 0)
        return;

    if (n == 1) {
        *m = 1;
        if (lower) {
            tmp1 = AB[0 + 0 * ldab];
        } else {
            tmp1 = AB[kd + 0 * ldab];
        }
        if (valeig) {
            if (!(vl < tmp1 && vu >= tmp1))
                *m = 0;
        }
        if (*m == 1) {
            W[0] = tmp1;
            if (wantz)
                Z[0 + 0 * ldz] = ONE;
        }
        return;
    }

    // Get machine constants
    safmin = slamch("S");
    eps = slamch("P");
    smlnum = safmin / eps;
    bignum = ONE / smlnum;
    rmin = sqrtf(smlnum);
    rmax = sqrtf(bignum);
    if (rmax > ONE / sqrtf(sqrtf(safmin)))
        rmax = ONE / sqrtf(sqrtf(safmin));

    // Scale matrix to allowable range, if necessary
    iscale = 0;
    abstll = abstol;
    if (valeig) {
        vll = vl;
        vuu = vu;
    } else {
        vll = ZERO;
        vuu = ZERO;
    }
    anrm = slansb("M", uplo, n, kd, AB, ldab, work);
    if (anrm > ZERO && anrm < rmin) {
        iscale = 1;
        sigma = rmin / anrm;
    } else if (anrm > rmax) {
        iscale = 1;
        sigma = rmax / anrm;
    }
    if (iscale == 1) {
        if (lower) {
            slascl("B", kd, kd, ONE, sigma, n, n, AB, ldab, info);
        } else {
            slascl("Q", kd, kd, ONE, sigma, n, n, AB, ldab, info);
        }
        if (abstol > 0)
            abstll = abstol * sigma;
        if (valeig) {
            vll = vl * sigma;
            vuu = vu * sigma;
        }
    }

    // Call SSBTRD to reduce symmetric band matrix to tridiagonal form
    indd = 0;
    inde = indd + n;
    indwrk = inde + n;
    ssbtrd(jobz, uplo, n, kd, AB, ldab, &work[indd], &work[inde], Q, ldq, &work[indwrk], &iinfo);

    // If all eigenvalues are desired and ABSTOL is less than or equal
    // to zero, then call SSTERF or SSTEQR. If this fails, try SSTEBZ.
    test = 0;
    if (indeig) {
        if (il == 1 && iu == n) {
            test = 1;
        }
    }
    if ((alleig || test) && (abstol <= ZERO)) {
        cblas_scopy(n, &work[indd], 1, W, 1);
        indee = indwrk + 2 * n;
        if (!wantz) {
            cblas_scopy(n - 1, &work[inde], 1, &work[indee], 1);
            ssterf(n, W, &work[indee], info);
        } else {
            slacpy("A", n, n, Q, ldq, Z, ldz);
            cblas_scopy(n - 1, &work[inde], 1, &work[indee], 1);
            ssteqr(jobz, n, W, &work[indee], Z, ldz, &work[indwrk], info);
            if (*info == 0) {
                for (i = 0; i < n; i++) {
                    ifail[i] = 0;
                }
            }
        }
        if (*info == 0) {
            *m = n;
            goto L30;
        }
        *info = 0;
    }

    // Otherwise, call SSTEBZ and, if eigenvectors are desired, SSTEIN.
    if (wantz) {
        order = 'B';
    } else {
        order = 'E';
    }
    indibl = 0;
    indisp = indibl + n;
    indiwo = indisp + n;
    sstebz(range, &order, n, vll, vuu, il, iu, abstll, &work[indd], &work[inde],
           m, &nsplit, W, &iwork[indibl], &iwork[indisp], &work[indwrk],
           &iwork[indiwo], info);

    if (wantz) {
        sstein(n, &work[indd], &work[inde], *m, W, &iwork[indibl], &iwork[indisp],
               Z, ldz, &work[indwrk], &iwork[indiwo], ifail, info);

        // Apply orthogonal matrix used in reduction to tridiagonal
        // form to eigenvectors returned by SSTEIN.
        for (j = 0; j < *m; j++) {
            cblas_scopy(n, &Z[0 + j * ldz], 1, work, 1);
            cblas_sgemv(CblasColMajor, CblasNoTrans, n, n, ONE, Q, ldq,
                        work, 1, ZERO, &Z[0 + j * ldz], 1);
        }
    }

L30:
    // If matrix was scaled, then rescale eigenvalues appropriately.
    if (iscale == 1) {
        if (*info == 0) {
            imax = *m;
        } else {
            imax = *info - 1;
        }
        cblas_sscal(imax, ONE / sigma, W, 1);
    }

    // If eigenvalues are not in order, then sort them, along with eigenvectors.
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
                itmp1 = iwork[indibl + i];
                W[i] = W[j];
                iwork[indibl + i] = iwork[indibl + j];
                W[j] = tmp1;
                iwork[indibl + j] = itmp1;
                cblas_sswap(n, &Z[0 + i * ldz], 1, &Z[0 + j * ldz], 1);
                if (*info != 0) {
                    itmp1 = ifail[i];
                    ifail[i] = ifail[j];
                    ifail[j] = itmp1;
                }
            }
        }
    }
}
