/**
 * @file zheevx.c
 * @brief ZHEEVX computes selected eigenvalues and, optionally, eigenvectors of a
 *        complex Hermitian matrix using bisection and inverse iteration.
 */

#include <math.h>
#include <complex.h>
#include <cblas.h>
#include "semicolon_lapack_complex_double.h"
#include "../include/lapack_tuning.h"

/**
 * ZHEEVX computes selected eigenvalues and, optionally, eigenvectors
 * of a complex Hermitian matrix A. Eigenvalues and eigenvectors can be
 * selected by specifying either a range of values or a range of indices
 * for the desired eigenvalues.
 *
 * @param[in]     jobz    = 'N': Compute eigenvalues only;
 *                          = 'V': Compute eigenvalues and eigenvectors.
 * @param[in]     range   = 'A': all eigenvalues will be found.
 *                          = 'V': all eigenvalues in the half-open interval (VL,VU]
 *                                 will be found.
 *                          = 'I': the IL-th through IU-th eigenvalues will be found.
 * @param[in]     uplo    = 'U': Upper triangle of A is stored;
 *                          = 'L': Lower triangle of A is stored.
 * @param[in]     n       The order of the matrix A. n >= 0.
 * @param[in,out] A       Double complex array, dimension (LDA, N).
 *                        On entry, the Hermitian matrix A. If UPLO = 'U', the
 *                        leading N-by-N upper triangular part of A contains the
 *                        upper triangular part of the matrix A. If UPLO = 'L',
 *                        the leading N-by-N lower triangular part of A contains
 *                        the lower triangular part of the matrix A.
 *                        On exit, the lower triangle (if UPLO='L') or the upper
 *                        triangle (if UPLO='U') of A, including the diagonal, is
 *                        destroyed.
 * @param[in]     lda     The leading dimension of A. lda >= max(1, n).
 * @param[in]     vl      If RANGE='V', the lower bound of the interval to
 *                        be searched for eigenvalues. VL < VU.
 *                        Not referenced if RANGE = 'A' or 'I'.
 * @param[in]     vu      If RANGE='V', the upper bound of the interval to
 *                        be searched for eigenvalues. VL < VU.
 *                        Not referenced if RANGE = 'A' or 'I'.
 * @param[in]     il      If RANGE='I', the index of the smallest eigenvalue
 *                        to be returned (0-based).
 *                        Not referenced if RANGE = 'A' or 'V'.
 * @param[in]     iu      If RANGE='I', the index of the largest eigenvalue
 *                        to be returned (0-based).
 *                        Not referenced if RANGE = 'A' or 'V'.
 * @param[in]     abstol  The absolute error tolerance for the eigenvalues.
 * @param[out]    m       The total number of eigenvalues found. 0 <= M <= N.
 * @param[out]    W       Double precision array, dimension (N). On normal exit,
 *                        the first M elements contain the selected eigenvalues
 *                        in ascending order.
 * @param[out]    Z       Double complex array, dimension (LDZ, max(1,M)).
 *                        If JOBZ = 'V', the first M columns contain the orthonormal
 *                        eigenvectors. If JOBZ = 'N', Z is not referenced.
 * @param[in]     ldz     Leading dimension of Z. LDZ >= 1, and if JOBZ = 'V', LDZ >= N.
 * @param[out]    work    Double complex workspace array, dimension (max(1, lwork)).
 *                        On exit, if info = 0, work[0] returns the optimal lwork.
 * @param[in]     lwork   The dimension of work. LWORK >= 1 when N <= 1, otherwise 2*N.
 *                        For optimal efficiency, LWORK >= (NB+1)*N.
 *                        If lwork = -1, workspace query only.
 * @param[out]    rwork   Double precision workspace array, dimension (7*N).
 * @param[out]    iwork   Integer workspace array, dimension (5*N).
 * @param[out]    ifail   Integer array, dimension (N).
 *                        If JOBZ = 'V', indices of eigenvectors that failed to converge.
 *                        If JOBZ = 'N', IFAIL is not referenced.
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 *                         - > 0: if info = i, then i eigenvectors failed to converge.
 *                           Their indices are stored in array IFAIL.
 */
void zheevx(const char* jobz, const char* range, const char* uplo,
            const int n, c128* restrict A, const int lda,
            const f64 vl, const f64 vu, const int il, const int iu,
            const f64 abstol, int* m,
            f64* restrict W,
            c128* restrict Z, const int ldz,
            c128* restrict work, const int lwork,
            f64* restrict rwork,
            int* restrict iwork,
            int* restrict ifail,
            int* info)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const c128 CONE = CMPLX(1.0, 0.0);

    int alleig, indeig, lower, lquery, test, valeig, wantz;
    int i, iinfo, imax, indd, inde, indee, indibl, indisp, indiwk;
    int indrwk, indtau, indwrk, iscale, itmp1, j, jj;
    int llwork, lwkmin, lwkopt, nb, nsplit;
    f64 abstll, anrm, bignum, eps, rmax, rmin, safmin, sigma, smlnum,
           tmp1, vll = 0.0, vuu = 0.0;

    /* Test the input parameters */
    lower = (uplo[0] == 'L' || uplo[0] == 'l');
    wantz = (jobz[0] == 'V' || jobz[0] == 'v');
    alleig = (range[0] == 'A' || range[0] == 'a');
    valeig = (range[0] == 'V' || range[0] == 'v');
    indeig = (range[0] == 'I' || range[0] == 'i');
    lquery = (lwork == -1);

    *info = 0;
    if (!(wantz || jobz[0] == 'N' || jobz[0] == 'n')) {
        *info = -1;
    } else if (!(alleig || valeig || indeig)) {
        *info = -2;
    } else if (!(lower || uplo[0] == 'U' || uplo[0] == 'u')) {
        *info = -3;
    } else if (n < 0) {
        *info = -4;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -6;
    } else {
        if (valeig) {
            if (n > 0 && vu <= vl) {
                *info = -8;
            }
        } else if (indeig) {
            if (il < 0 || il > (0 > n - 1 ? 0 : n - 1)) {
                *info = -9;
            } else if (iu < ((n - 1) < il ? (n - 1) : il) || iu > n - 1) {
                *info = -10;
            }
        }
    }
    if (*info == 0) {
        if (ldz < 1 || (wantz && ldz < n)) {
            *info = -15;
        }
    }

    if (*info == 0) {
        if (n <= 1) {
            lwkmin = 1;
            lwkopt = 1;
        } else {
            lwkmin = 2 * n;
            nb = lapack_get_nb("HETRD");
            lwkopt = (nb + 1) * n > lwkmin ? (nb + 1) * n : lwkmin;
        }
        work[0] = CMPLX((f64)lwkopt, 0.0);

        if (lwork < lwkmin && !lquery) {
            *info = -17;
        }
    }

    if (*info != 0) {
        xerbla("ZHEEVX", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    /* Quick return if possible */
    *m = 0;
    if (n == 0) {
        return;
    }

    if (n == 1) {
        if (alleig || indeig) {
            *m = 1;
            W[0] = creal(A[0]);
        } else {
            if (vl < creal(A[0]) && vu >= creal(A[0])) {
                *m = 1;
                W[0] = creal(A[0]);
            }
        }
        if (wantz) {
            Z[0] = CONE;
        }
        return;
    }

    /* Get machine constants */
    safmin = dlamch("S");
    eps = dlamch("E");
    smlnum = safmin / eps;
    bignum = ONE / smlnum;
    rmin = sqrt(smlnum);
    rmax = sqrt(bignum);
    if (rmax > ONE / sqrt(sqrt(safmin))) {
        rmax = ONE / sqrt(sqrt(safmin));
    }

    /* Scale matrix to allowable range, if necessary */
    iscale = 0;
    abstll = abstol;
    if (valeig) {
        vll = vl;
        vuu = vu;
    }
    anrm = zlanhe("M", uplo, n, A, lda, rwork);
    if (anrm > ZERO && anrm < rmin) {
        iscale = 1;
        sigma = rmin / anrm;
    } else if (anrm > rmax) {
        iscale = 1;
        sigma = rmax / anrm;
    }
    if (iscale == 1) {
        if (lower) {
            for (j = 0; j < n; j++) {
                cblas_zdscal(n - j, sigma, &A[j + j * lda], 1);
            }
        } else {
            for (j = 0; j < n; j++) {
                cblas_zdscal(j + 1, sigma, &A[0 + j * lda], 1);
            }
        }
        if (abstol > 0) {
            abstll = abstol * sigma;
        }
        if (valeig) {
            vll = vl * sigma;
            vuu = vu * sigma;
        }
    }

    /* Call ZHETRD to reduce Hermitian matrix to tridiagonal form */
    indd = 0;
    inde = indd + n;
    indrwk = inde + n;
    indtau = 0;
    indwrk = indtau + n;
    llwork = lwork - indwrk;

    zhetrd(uplo, n, A, lda, &rwork[indd], &rwork[inde], &work[indtau],
           &work[indwrk], llwork, &iinfo);

    /* If all eigenvalues are desired and ABSTOL is less than or equal to
     * zero, then call DSTERF or ZUNGTR and ZSTEQR. If this fails for
     * some eigenvalue, then try DSTEBZ. */
    test = 0;
    if (indeig) {
        if (il == 0 && iu == n - 1) {
            test = 1;
        }
    }
    if ((alleig || test) && abstol <= ZERO) {
        cblas_dcopy(n, &rwork[indd], 1, W, 1);
        indee = indrwk + 2 * n;
        if (!wantz) {
            cblas_dcopy(n - 1, &rwork[inde], 1, &rwork[indee], 1);
            dsterf(n, W, &rwork[indee], info);
        } else {
            zlacpy("A", n, n, A, lda, Z, ldz);
            zungtr(uplo, n, Z, ldz, &work[indtau],
                   &work[indwrk], llwork, &iinfo);
            cblas_dcopy(n - 1, &rwork[inde], 1, &rwork[indee], 1);
            zsteqr(jobz, n, W, &rwork[indee], Z, ldz,
                   &rwork[indrwk], info);
            if (*info == 0) {
                for (i = 0; i < n; i++) {
                    ifail[i] = 0;
                }
            }
        }
        if (*info == 0) {
            *m = n;
            goto L40;
        }
        *info = 0;
    }

    /* Otherwise, call DSTEBZ and, if eigenvectors are desired, ZSTEIN. */
    const char* order;
    if (wantz) {
        order = "B";
    } else {
        order = "E";
    }
    indibl = 0;
    indisp = indibl + n;
    indiwk = indisp + n;
    dstebz(range, order, n, vll, vuu, il, iu, abstll,
           &rwork[indd], &rwork[inde], m, &nsplit, W,
           &iwork[indibl], &iwork[indisp], &rwork[indrwk],
           &iwork[indiwk], info);

    if (wantz) {
        zstein(n, &rwork[indd], &rwork[inde], *m, W,
               &iwork[indibl], &iwork[indisp], Z, ldz,
               &rwork[indrwk], &iwork[indiwk], ifail, info);

        /* Apply unitary matrix used in reduction to tridiagonal
         * form to eigenvectors returned by ZSTEIN. */
        zunmtr("L", uplo, "N", n, *m, A, lda, &work[indtau],
               Z, ldz, &work[indwrk], llwork, &iinfo);
    }

    /* If matrix was scaled, then rescale eigenvalues appropriately. */
L40:
    if (iscale == 1) {
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
                itmp1 = iwork[indibl + i];
                W[i] = W[j];
                iwork[indibl + i] = iwork[indibl + j];
                W[j] = tmp1;
                iwork[indibl + j] = itmp1;
                cblas_zswap(n, &Z[0 + i * ldz], 1, &Z[0 + j * ldz], 1);
                if (*info != 0) {
                    itmp1 = ifail[i];
                    ifail[i] = ifail[j];
                    ifail[j] = itmp1;
                }
            }
        }
    }

    /* Set WORK(1) to optimal complex workspace size */
    work[0] = CMPLX((f64)lwkopt, 0.0);
}
