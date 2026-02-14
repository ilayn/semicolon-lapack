/**
 * @file zhbevx.c
 * @brief ZHBEVX computes selected eigenvalues and eigenvectors of a complex
 *        Hermitian band matrix.
 */

#include <math.h>
#include <complex.h>
#include <cblas.h>
#include "semicolon_lapack_complex_double.h"

/**
 * ZHBEVX computes selected eigenvalues and, optionally, eigenvectors
 * of a complex Hermitian band matrix A. Eigenvalues and eigenvectors can
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
 * @param[in]     kd     The number of super-/sub-diagonals of A. kd >= 0.
 * @param[in,out] AB     Complex array, dimension (ldab, n).
 *                        On entry, the upper or lower triangle of the Hermitian
 *                        band matrix A. On exit, overwritten by reduction values.
 * @param[in]     ldab   The leading dimension of AB. ldab >= kd+1.
 * @param[out]    Q      Complex array, dimension (ldq, n). If jobz='V', the
 *                        unitary matrix used in the reduction to tridiagonal form.
 * @param[in]     ldq    The leading dimension of Q. ldq >= max(1,n) if jobz='V'.
 * @param[in]     vl     Lower bound of interval (if range='V').
 * @param[in]     vu     Upper bound of interval (if range='V').
 * @param[in]     il     Index of smallest eigenvalue (if range='I').
 * @param[in]     iu     Index of largest eigenvalue (if range='I').
 * @param[in]     abstol Absolute error tolerance for eigenvalues.
 * @param[out]    m      The total number of eigenvalues found.
 * @param[out]    W      The selected eigenvalues in ascending order.
 * @param[out]    Z      Complex array, dimension (ldz, max(1,m)). If jobz='V',
 *                        the eigenvectors.
 * @param[in]     ldz    The leading dimension of Z. ldz >= 1, and >= n if jobz='V'.
 * @param[out]    work   Complex workspace array of dimension (n).
 * @param[out]    rwork  Double precision workspace array of dimension (7*n).
 * @param[out]    iwork  Integer workspace array of dimension (5*n).
 * @param[out]    ifail  If jobz='V', indices of eigenvectors that failed to converge.
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 *                         - > 0: if info = i, then i eigenvectors failed to converge
 */
void zhbevx(
    const char* jobz,
    const char* range,
    const char* uplo,
    const int n,
    const int kd,
    c128* restrict AB,
    const int ldab,
    c128* restrict Q,
    const int ldq,
    const f64 vl,
    const f64 vu,
    const int il,
    const int iu,
    const f64 abstol,
    int* m,
    f64* restrict W,
    c128* restrict Z,
    const int ldz,
    c128* restrict work,
    f64* restrict rwork,
    int* restrict iwork,
    int* restrict ifail,
    int* info)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const c128 CZERO = CMPLX(0.0, 0.0);
    const c128 CONE = CMPLX(1.0, 0.0);

    int alleig, indeig, lower, test, valeig, wantz;
    char order;
    int i, iinfo, imax, indd, inde, indee, indibl, indisp, indiwk, indrwk, indwrk;
    int iscale, itmp1, j, jj, nsplit;
    f64 abstll, anrm, bignum, eps, rmax, rmin, safmin, sigma, smlnum, tmp1, vll, vuu;
    c128 ctmp1;

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
        xerbla("ZHBEVX", -(*info));
        return;
    }

    *m = 0;
    if (n == 0)
        return;

    if (n == 1) {
        *m = 1;
        if (lower) {
            ctmp1 = AB[0];
        } else {
            ctmp1 = AB[kd];
        }
        tmp1 = creal(ctmp1);
        if (valeig) {
            if (!(vl < tmp1 && vu >= tmp1))
                *m = 0;
        }
        if (*m == 1) {
            W[0] = creal(ctmp1);
            if (wantz)
                Z[0] = CONE;
        }
        return;
    }

    safmin = dlamch("S");
    eps = dlamch("P");
    smlnum = safmin / eps;
    bignum = ONE / smlnum;
    rmin = sqrt(smlnum);
    rmax = sqrt(bignum);
    if (rmax > ONE / sqrt(sqrt(safmin)))
        rmax = ONE / sqrt(sqrt(safmin));

    iscale = 0;
    abstll = abstol;
    if (valeig) {
        vll = vl;
        vuu = vu;
    } else {
        vll = ZERO;
        vuu = ZERO;
    }
    anrm = zlanhb("M", uplo, n, kd, AB, ldab, rwork);
    if (anrm > ZERO && anrm < rmin) {
        iscale = 1;
        sigma = rmin / anrm;
    } else if (anrm > rmax) {
        iscale = 1;
        sigma = rmax / anrm;
    }
    if (iscale == 1) {
        if (lower) {
            zlascl("B", kd, kd, ONE, sigma, n, n, AB, ldab, info);
        } else {
            zlascl("Q", kd, kd, ONE, sigma, n, n, AB, ldab, info);
        }
        if (abstol > 0)
            abstll = abstol * sigma;
        if (valeig) {
            vll = vl * sigma;
            vuu = vu * sigma;
        }
    }

    indd = 0;
    inde = indd + n;
    indrwk = inde + n;
    indwrk = 0;
    zhbtrd(jobz, uplo, n, kd, AB, ldab, &rwork[indd],
           &rwork[inde], Q, ldq, &work[indwrk], &iinfo);

    test = 0;
    if (indeig) {
        if (il == 1 && iu == n) {
            test = 1;
        }
    }
    if ((alleig || test) && (abstol <= ZERO)) {
        cblas_dcopy(n, &rwork[indd], 1, W, 1);
        indee = indrwk + 2 * n;
        if (!wantz) {
            cblas_dcopy(n - 1, &rwork[inde], 1, &rwork[indee], 1);
            dsterf(n, W, &rwork[indee], info);
        } else {
            zlacpy("A", n, n, Q, ldq, Z, ldz);
            cblas_dcopy(n - 1, &rwork[inde], 1, &rwork[indee], 1);
            zsteqr(jobz, n, W, &rwork[indee], Z, ldz, &rwork[indrwk], info);
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

    if (wantz) {
        order = 'B';
    } else {
        order = 'E';
    }
    indibl = 0;
    indisp = indibl + n;
    indiwk = indisp + n;
    dstebz(range, &order, n, vll, vuu, il, iu, abstll, &rwork[indd], &rwork[inde],
           m, &nsplit, W, &iwork[indibl], &iwork[indisp], &rwork[indrwk],
           &iwork[indiwk], info);

    if (wantz) {
        zstein(n, &rwork[indd], &rwork[inde], *m, W, &iwork[indibl], &iwork[indisp],
               Z, ldz, &rwork[indrwk], &iwork[indiwk], ifail, info);

        for (j = 0; j < *m; j++) {
            cblas_zcopy(n, &Z[j * ldz], 1, work, 1);
            cblas_zgemv(CblasColMajor, CblasNoTrans, n, n, &CONE, Q, ldq,
                        work, 1, &CZERO, &Z[j * ldz], 1);
        }
    }

L30:
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
                itmp1 = iwork[indibl + i];
                W[i] = W[j];
                iwork[indibl + i] = iwork[indibl + j];
                W[j] = tmp1;
                iwork[indibl + j] = itmp1;
                cblas_zswap(n, &Z[i * ldz], 1, &Z[j * ldz], 1);
                if (*info != 0) {
                    itmp1 = ifail[i];
                    ifail[i] = ifail[j];
                    ifail[j] = itmp1;
                }
            }
        }
    }
}
