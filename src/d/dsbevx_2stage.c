/**
 * @file dsbevx_2stage.c
 * @brief DSBEVX_2STAGE computes selected eigenvalues and optionally eigenvectors
 *        of a real symmetric band matrix using 2-stage reduction to tridiagonal.
 */

#include "semicolon_lapack_double.h"
#include <cblas.h>
#include <math.h>

/**
 * DSBEVX_2STAGE computes selected eigenvalues and, optionally, eigenvectors
 * of a real symmetric band matrix A using the 2stage technique for
 * the reduction to tridiagonal. Eigenvalues and eigenvectors can
 * be selected by specifying either a range of values or a range of
 * indices for the desired eigenvalues.
 *
 * @param[in]     jobz    = 'N': eigenvalues only; = 'V': not available in this release.
 * @param[in]     range   = 'A': all eigenvalues; = 'V': eigenvalues in (vl,vu];
 *                          = 'I': il-th through iu-th eigenvalues.
 * @param[in]     uplo    = 'U': upper triangle stored; = 'L': lower triangle stored
 * @param[in]     n       The order of the matrix A. n >= 0.
 * @param[in]     kd      Number of super/sub-diagonals. kd >= 0.
 * @param[in,out] AB      Band matrix, overwritten on exit.
 * @param[in]     ldab    Leading dimension of AB. ldab >= kd+1.
 * @param[out]    Q       Orthogonal matrix if jobz='V'; not referenced if jobz='N'.
 * @param[in]     ldq     Leading dimension of Q. ldq >= 1, or ldq >= n if jobz='V'.
 * @param[in]     vl      Lower bound if range='V'.
 * @param[in]     vu      Upper bound if range='V'. vl < vu.
 * @param[in]     il      Smallest eigenvalue index if range='I'. 1 <= il <= iu.
 * @param[in]     iu      Largest eigenvalue index if range='I'. il <= iu <= n.
 * @param[in]     abstol  Absolute error tolerance for eigenvalues.
 * @param[out]    m       Number of eigenvalues found.
 * @param[out]    W       Selected eigenvalues in ascending order.
 * @param[out]    Z       Eigenvectors if jobz='V'; not referenced if jobz='N'.
 * @param[in]     ldz     Leading dimension of Z. ldz >= 1, or ldz >= n if jobz='V'.
 * @param[out]    work    Workspace. On exit, work[0] = optimal LWORK.
 * @param[in]     lwork   Length of work. If -1, workspace query.
 * @param[out]    iwork   Integer workspace, dimension (5*n).
 * @param[out]    ifail   Indices of eigenvectors that failed to converge.
 * @param[out]    info
 *                         - = 0: success; < 0: illegal argument; > 0: convergence failure.
 */
void dsbevx_2stage(
    const char* jobz,
    const char* range,
    const char* uplo,
    const int n,
    const int kd,
    f64* restrict AB,
    const int ldab,
    f64* restrict Q,
    const int ldq,
    const f64 vl,
    const f64 vu,
    const int il,
    const int iu,
    const f64 abstol,
    int* m,
    f64* restrict W,
    f64* restrict Z,
    const int ldz,
    f64* restrict work,
    const int lwork,
    int* restrict iwork,
    int* restrict ifail,
    int* info)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    int alleig, indeig, lower, test, valeig, wantz, lquery;
    char order;
    int i, iinfo, imax, indd, inde, indee, indibl;
    int indisp, indiwo, indwrk, iscale, itmp1, j, jj;
    int llwork, lwmin, lhtrd = 0, lwtrd, ib, indhous, nsplit;
    f64 abstll, anrm, bignum, eps, rmax, rmin, safmin;
    f64 sigma, smlnum, tmp1, vll, vuu;

    wantz = (jobz[0] == 'V' || jobz[0] == 'v');
    alleig = (range[0] == 'A' || range[0] == 'a');
    valeig = (range[0] == 'V' || range[0] == 'v');
    indeig = (range[0] == 'I' || range[0] == 'i');
    lower = (uplo[0] == 'L' || uplo[0] == 'l');
    lquery = (lwork == -1);

    *info = 0;
    if (!(jobz[0] == 'N' || jobz[0] == 'n')) {
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
    } else if (wantz && ldq < ((1 > n) ? 1 : n)) {
        *info = -9;
    } else {
        if (valeig) {
            if (n > 0 && vu <= vl) {
                *info = -11;
            }
        } else if (indeig) {
            if (il < 1 || il > ((1 > n) ? 1 : n)) {
                *info = -12;
            } else if (iu < ((n < il) ? n : il) || iu > n) {
                *info = -13;
            }
        }
    }
    if (*info == 0) {
        if (ldz < 1 || (wantz && ldz < n)) {
            *info = -18;
        }
    }

    if (*info == 0) {
        if (n <= 1) {
            lwmin = 1;
            work[0] = (f64)lwmin;
        } else {
            ib = ilaenv2stage(2, "DSYTRD_SB2ST", jobz, n, kd, -1, -1);
            lhtrd = ilaenv2stage(3, "DSYTRD_SB2ST", jobz, n, kd, ib, -1);
            lwtrd = ilaenv2stage(4, "DSYTRD_SB2ST", jobz, n, kd, ib, -1);
            lwmin = 2 * n + lhtrd + lwtrd;
            work[0] = (f64)lwmin;
        }

        if (lwork < lwmin && !lquery) {
            *info = -20;
        }
    }

    if (*info != 0) {
        xerbla("DSBEVX_2STAGE", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    *m = 0;
    if (n == 0) {
        return;
    }

    if (n == 1) {
        *m = 1;
        if (lower) {
            tmp1 = AB[0];
        } else {
            tmp1 = AB[kd];
        }
        if (valeig) {
            if (!(vl < tmp1 && vu >= tmp1)) {
                *m = 0;
            }
        }
        if (*m == 1) {
            W[0] = tmp1;
            if (wantz) {
                Z[0] = ONE;
            }
        }
        return;
    }

    safmin = dlamch("Safe minimum");
    eps = dlamch("Precision");
    smlnum = safmin / eps;
    bignum = ONE / smlnum;
    rmin = sqrt(smlnum);
    rmax = (sqrt(bignum) < ONE / sqrt(sqrt(safmin))) ? sqrt(bignum) : ONE / sqrt(sqrt(safmin));

    iscale = 0;
    abstll = abstol;
    if (valeig) {
        vll = vl;
        vuu = vu;
    } else {
        vll = ZERO;
        vuu = ZERO;
    }
    anrm = dlansb("M", uplo, n, kd, AB, ldab, work);
    if (anrm > ZERO && anrm < rmin) {
        iscale = 1;
        sigma = rmin / anrm;
    } else if (anrm > rmax) {
        iscale = 1;
        sigma = rmax / anrm;
    }
    if (iscale == 1) {
        if (lower) {
            dlascl("B", kd, kd, ONE, sigma, n, n, AB, ldab, info);
        } else {
            dlascl("Q", kd, kd, ONE, sigma, n, n, AB, ldab, info);
        }
        if (abstol > 0) {
            abstll = abstol * sigma;
        }
        if (valeig) {
            vll = vl * sigma;
            vuu = vu * sigma;
        }
    }

    indd = 0;
    inde = indd + n;
    indhous = inde + n;
    indwrk = indhous + lhtrd;
    llwork = lwork - indwrk;

    dsytrd_sb2st("N", jobz, uplo, n, kd, AB, ldab,
                 &work[indd], &work[inde], &work[indhous], lhtrd,
                 &work[indwrk], llwork, &iinfo);

    test = 0;
    if (indeig) {
        if (il == 1 && iu == n) {
            test = 1;
        }
    }
    if ((alleig || test) && abstol <= ZERO) {
        cblas_dcopy(n, &work[indd], 1, W, 1);
        indee = indwrk + 2 * n;
        if (!wantz) {
            cblas_dcopy(n - 1, &work[inde], 1, &work[indee], 1);
            dsterf(n, W, &work[indee], info);
        } else {
            dlacpy("A", n, n, Q, ldq, Z, ldz);
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
    indiwo = indisp + n;
    {
        char range_str[2] = {range[0], '\0'};
        char order_str[2] = {order, '\0'};
        dstebz(range_str, order_str, n, vll, vuu, il, iu, abstll,
               &work[indd], &work[inde], m, &nsplit, W,
               &iwork[indibl], &iwork[indisp], &work[indwrk],
               &iwork[indiwo], info);
    }

    if (wantz) {
        dstein(n, &work[indd], &work[inde], *m, W,
               &iwork[indibl], &iwork[indisp], Z, ldz,
               &work[indwrk], &iwork[indiwo], ifail, info);

        for (j = 0; j < *m; j++) {
            cblas_dcopy(n, &Z[j * ldz], 1, work, 1);
            cblas_dgemv(CblasColMajor, CblasNoTrans, n, n, ONE, Q, ldq,
                        work, 1, ZERO, &Z[j * ldz], 1);
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
                cblas_dswap(n, &Z[i * ldz], 1, &Z[j * ldz], 1);
                if (*info != 0) {
                    itmp1 = ifail[i];
                    ifail[i] = ifail[j];
                    ifail[j] = itmp1;
                }
            }
        }
    }

    work[0] = (f64)lwmin;
}
