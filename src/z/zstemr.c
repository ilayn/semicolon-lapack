/**
 * @file zstemr.c
 * @brief ZSTEMR computes selected eigenvalues and, optionally, eigenvectors
 *        of a real symmetric tridiagonal matrix T using the Relatively Robust
 *        Representations.
 */

#include <math.h>
#include <stdlib.h>
#include <complex.h>
#include <cblas.h>
#include "semicolon_lapack_complex_double.h"

/**
 * ZSTEMR computes selected eigenvalues and, optionally, eigenvectors
 * of a real symmetric tridiagonal matrix T. Any such unreduced matrix has
 * a well defined set of pairwise different real eigenvalues, the corresponding
 * real eigenvectors are pairwise orthogonal.
 *
 * @param[in]     jobz    'N': eigenvalues only; 'V': eigenvalues and eigenvectors.
 * @param[in]     range   'A': all; 'V': in (vl,vu]; 'I': il-th through iu-th.
 * @param[in]     n       The order of the matrix. n >= 0.
 * @param[in,out] D       Double precision array, dimension (n). Diagonal elements.
 * @param[in,out] E       Double precision array, dimension (n). Subdiagonal elements.
 * @param[in]     vl      Lower bound of interval (if range='V').
 * @param[in]     vu      Upper bound of interval (if range='V').
 * @param[in]     il      Index of smallest eigenvalue (0-based, if range='I').
 * @param[in]     iu      Index of largest eigenvalue (0-based, if range='I').
 * @param[out]    m       Total number of eigenvalues found.
 * @param[out]    W       Double precision array, dimension (n). Eigenvalues.
 * @param[out]    Z       Complex array, dimension (ldz, nzc). Eigenvectors.
 * @param[in]     ldz     Leading dimension of Z.
 * @param[in]     nzc     Number of eigenvector columns available. -1 for query.
 * @param[out]    isuppz  Integer array, dimension (2*max(1,m)). Support of eigenvectors.
 * @param[in,out] tryrac  If nonzero, try for relative accuracy.
 * @param[out]    work    Double precision workspace, dimension (lwork).
 * @param[in]     lwork   Dimension of work. -1 for query.
 * @param[out]    iwork   Integer workspace, dimension (liwork).
 * @param[in]     liwork  Dimension of iwork. -1 for query.
 * @param[out]    info
 *                         - = 0: success. < 0: illegal argument. > 0: internal error.
 */
void zstemr(const char* jobz, const char* range, const int n,
            double* D, double* E, const double vl, const double vu,
            const int il, const int iu, int* m, double* W,
            double complex* Z, const int ldz, const int nzc, int* isuppz,
            int* tryrac, double* work, const int lwork,
            int* iwork, const int liwork, int* info)
{
    const double ZERO = 0.0;
    const double ONE = 1.0;
    const double FOUR = 4.0;
    const double MINRGP = 1.0e-3;

    int wantz, alleig, valeig, indeig, lquery, zquery, laeswap;
    int i, ibegin, iend, ifirst, iil, iindbl, iindw;
    int iindwk, iinfo, iinspl, iiu, ilast, in, indd;
    int inde2, inderr, indgp, indgrs, indwrk, itmp, itmp2;
    int j, jblk, jj, liwmin, lwmin, nsplit, nzcmin, offset, wbegin, wend;
    double bignum, cs, eps, pivmin, r1 = 0.0, r2 = 0.0, rmax, rmin;
    double rtol1, rtol2, safmin, scale, smlnum, sn;
    double thresh, tmp, tnrm, wl, wu;

    wantz = (jobz[0] == 'V' || jobz[0] == 'v');
    alleig = (range[0] == 'A' || range[0] == 'a');
    valeig = (range[0] == 'V' || range[0] == 'v');
    indeig = (range[0] == 'I' || range[0] == 'i');

    lquery = (lwork == -1 || liwork == -1);
    zquery = (nzc == -1);
    laeswap = 0;

    /* DSTEMR needs WORK of size 6*N, IWORK of size 3*N.
       In addition, DLARRE needs WORK of size 6*N, IWORK of size 5*N.
       Furthermore, ZLARRV needs WORK of size 12*N, IWORK of size 7*N. */
    if (wantz) {
        lwmin = 18 * n;
        liwmin = 10 * n;
    } else {
        lwmin = 12 * n;
        liwmin = 8 * n;
    }

    wl = ZERO;
    wu = ZERO;
    iil = 0;
    iiu = 0;
    nsplit = 0;

    if (valeig) {
        wl = vl;
        wu = vu;
    } else if (indeig) {
        iil = il;
        iiu = iu;
    }

    *info = 0;
    if (!(wantz || jobz[0] == 'N' || jobz[0] == 'n')) {
        *info = -1;
    } else if (!(alleig || valeig || indeig)) {
        *info = -2;
    } else if (n < 0) {
        *info = -3;
    } else if (valeig && n > 0 && wu <= wl) {
        *info = -7;
    } else if (indeig && (iil < 0 || iil >= n)) {
        *info = -8;
    } else if (indeig && (iiu < iil || iiu >= n)) {
        *info = -9;
    } else if (ldz < 1 || (wantz && ldz < n)) {
        *info = -13;
    } else if (lwork < lwmin && !lquery) {
        *info = -17;
    } else if (liwork < liwmin && !lquery) {
        *info = -19;
    }

    /* Get machine constants. */
    safmin = dlamch("S");
    eps = dlamch("P");
    smlnum = safmin / eps;
    bignum = ONE / smlnum;
    rmin = sqrt(smlnum);
    rmax = fmin(sqrt(bignum), ONE / sqrt(sqrt(safmin)));

    if (*info == 0) {
        work[0] = lwmin;
        iwork[0] = liwmin;

        if (wantz && alleig) {
            nzcmin = n;
        } else if (wantz && valeig) {
            dlarrc("T", n, vl, vu, D, E, safmin, &nzcmin, &itmp, &itmp2, info);
        } else if (wantz && indeig) {
            nzcmin = iiu - iil + 1;
        } else {
            nzcmin = 0;
        }
        if (zquery && *info == 0) {
            Z[0] = (double complex)nzcmin;
        } else if (nzc < nzcmin && !zquery) {
            *info = -14;
        }
    }

    if (*info != 0) {
        xerbla("ZSTEMR", -(*info));
        return;
    } else if (lquery || zquery) {
        return;
    }

    /* Handle N = 0, 1, and 2 cases immediately */
    *m = 0;
    if (n == 0) return;

    if (n == 1) {
        if (alleig || indeig) {
            *m = 1;
            W[0] = D[0];
        } else {
            if (wl < D[0] && wu >= D[0]) {
                *m = 1;
                W[0] = D[0];
            }
        }
        if (wantz && !zquery) {
            Z[0] = ONE;
            isuppz[0] = 0;
            isuppz[1] = 0;
        }
        return;
    }

    if (n == 2) {
        if (!wantz) {
            dlae2(D[0], E[0], D[1], &r1, &r2);
        } else if (wantz && !zquery) {
            dlaev2(D[0], E[0], D[1], &r1, &r2, &cs, &sn);
        }
        /* DLAE2 and DLAEV2 outputs satisfy |R1| >= |R2|. However,
           the following code requires R1 >= R2. Hence, we correct
           the order of R1, R2, CS, SN if R1 < R2 before further processing. */
        if (r1 < r2) {
            E[1] = r1;
            r1 = r2;
            r2 = E[1];
            laeswap = 1;
        }
        if (alleig ||
            (valeig && r2 > wl && r2 <= wu) ||
            (indeig && iil == 0)) {
            (*m)++;
            W[*m - 1] = r2;
            if (wantz && !zquery) {
                if (laeswap) {
                    Z[0 + (*m - 1) * ldz] = cs;
                    Z[1 + (*m - 1) * ldz] = sn;
                } else {
                    Z[0 + (*m - 1) * ldz] = -sn;
                    Z[1 + (*m - 1) * ldz] = cs;
                }
                /* Note: At most one of SN and CS can be zero. */
                if (sn != ZERO) {
                    if (cs != ZERO) {
                        isuppz[2 * (*m) - 2] = 0;
                        isuppz[2 * (*m) - 1] = 1;
                    } else {
                        isuppz[2 * (*m) - 2] = 0;
                        isuppz[2 * (*m) - 1] = 0;
                    }
                } else {
                    isuppz[2 * (*m) - 2] = 1;
                    isuppz[2 * (*m) - 1] = 1;
                }
            }
        }
        if (alleig ||
            (valeig && r1 > wl && r1 <= wu) ||
            (indeig && iiu == 1)) {
            (*m)++;
            W[*m - 1] = r1;
            if (wantz && !zquery) {
                if (laeswap) {
                    Z[0 + (*m - 1) * ldz] = -sn;
                    Z[1 + (*m - 1) * ldz] = cs;
                } else {
                    Z[0 + (*m - 1) * ldz] = cs;
                    Z[1 + (*m - 1) * ldz] = sn;
                }
                if (sn != ZERO) {
                    if (cs != ZERO) {
                        isuppz[2 * (*m) - 2] = 0;
                        isuppz[2 * (*m) - 1] = 1;
                    } else {
                        isuppz[2 * (*m) - 2] = 0;
                        isuppz[2 * (*m) - 1] = 0;
                    }
                } else {
                    isuppz[2 * (*m) - 2] = 1;
                    isuppz[2 * (*m) - 1] = 1;
                }
            }
        }

    } else {

        /* Continue with general N */

        indgrs = 0;
        inderr = 2 * n;
        indgp = 3 * n;
        indd = 4 * n;
        inde2 = 5 * n;
        indwrk = 6 * n;

        iinspl = 0;
        iindbl = n;
        iindw = 2 * n;
        iindwk = 3 * n;

        /* Scale matrix to allowable range, if necessary. */
        scale = ONE;
        tnrm = dlanst("M", n, D, E);
        if (tnrm > ZERO && tnrm < rmin) {
            scale = rmin / tnrm;
        } else if (tnrm > rmax) {
            scale = rmax / tnrm;
        }
        if (scale != ONE) {
            cblas_dscal(n, scale, D, 1);
            cblas_dscal(n - 1, scale, E, 1);
            tnrm *= scale;
            if (valeig) {
                /* If eigenvalues in interval have to be found,
                   scale (WL, WU] accordingly */
                wl *= scale;
                wu *= scale;
            }
        }

        /* Compute the desired eigenvalues of the tridiagonal after splitting
           into smaller subblocks if the corresponding off-diagonal elements
           are small
           THRESH is the splitting parameter for DLARRE
           A negative THRESH forces the old splitting criterion based on the
           size of the off-diagonal. A positive THRESH switches to splitting
           which preserves relative accuracy. */

        if (*tryrac) {
            /* Test whether the matrix warrants the more expensive relative approach. */
            dlarrr(n, D, E, &iinfo);
        } else {
            /* The user does not care about relative accurately eigenvalues */
            iinfo = -1;
        }
        /* Set the splitting criterion */
        if (iinfo == 0) {
            thresh = eps;
        } else {
            thresh = -eps;
            /* relative accuracy is desired but T does not guarantee it */
            *tryrac = 0;
        }

        if (*tryrac) {
            /* Copy original diagonal, needed to guarantee relative accuracy */
            cblas_dcopy(n, D, 1, &work[indd], 1);
        }
        /* Store the squares of the offdiagonal values of T */
        for (j = 0; j < n - 1; j++) {
            work[inde2 + j] = E[j] * E[j];
        }

        /* Set the tolerance parameters for bisection */
        if (!wantz) {
            /* DLARRE computes the eigenvalues to full precision. */
            rtol1 = FOUR * eps;
            rtol2 = FOUR * eps;
        } else {
            /* DLARRE computes the eigenvalues to less than full precision.
               ZLARRV will refine the eigenvalue approximations, and we can
               need less accurate initial bisection in DLARRE. */
            rtol1 = sqrt(eps);
            rtol2 = fmax(sqrt(eps) * 5.0e-3, FOUR * eps);
        }
        dlarre(range, n, &wl, &wu, iil, iiu, D, E,
               &work[inde2], rtol1, rtol2, thresh, &nsplit,
               &iwork[iinspl], m, W, &work[inderr],
               &work[indgp], &iwork[iindbl],
               &iwork[iindw], &work[indgrs], &pivmin,
               &work[indwrk], &iwork[iindwk], &iinfo);
        if (iinfo != 0) {
            *info = 10 + abs(iinfo);
            return;
        }
        /* Note that if RANGE != 'V', DLARRE computes bounds on the desired
           part of the spectrum. All desired eigenvalues are contained in
           (WL,WU] */

        if (wantz) {

            /* Compute the desired eigenvectors corresponding to the computed
               eigenvalues */
            zlarrv(n, wl, wu, D, E,
                   pivmin, &iwork[iinspl], *m,
                   0, *m - 1, MINRGP, rtol1, rtol2,
                   W, &work[inderr], &work[indgp], &iwork[iindbl],
                   &iwork[iindw], &work[indgrs], Z, ldz,
                   isuppz, &work[indwrk], &iwork[iindwk], &iinfo);
            if (iinfo != 0) {
                *info = 20 + abs(iinfo);
                return;
            }
        } else {
            /* DLARRE computes eigenvalues of the (shifted) root representation
               ZLARRV returns the eigenvalues of the unshifted matrix.
               However, if the eigenvectors are not desired by the user, we need
               to apply the corresponding shifts from DLARRE to obtain the
               eigenvalues of the original matrix. */
            for (j = 0; j < *m; j++) {
                itmp = iwork[iindbl + j];
                W[j] += E[iwork[iinspl + itmp]];
            }
        }

        if (*tryrac) {
            /* Refine computed eigenvalues so that they are relatively accurate
               with respect to the original matrix T. */
            ibegin = 0;
            wbegin = 0;
            for (jblk = 0; jblk <= iwork[iindbl + *m - 1]; jblk++) {
                iend = iwork[iinspl + jblk];
                in = iend - ibegin + 1;
                wend = wbegin - 1;
                /* check if any eigenvalues have to be refined in this block */
                while (wend < *m - 1) {
                    if (iwork[iindbl + wend + 1] == jblk) {
                        wend++;
                    } else {
                        break;
                    }
                }
                if (wend < wbegin) {
                    ibegin = iend + 1;
                    continue;
                }

                offset = iwork[iindw + wbegin];
                ifirst = iwork[iindw + wbegin];
                ilast = iwork[iindw + wend];
                rtol2 = FOUR * eps;
                dlarrj(in,
                       &work[indd + ibegin], &work[inde2 + ibegin],
                       ifirst, ilast, rtol2, offset, &W[wbegin],
                       &work[inderr + wbegin],
                       &work[indwrk], &iwork[iindwk], pivmin,
                       tnrm, &iinfo);
                ibegin = iend + 1;
                wbegin = wend + 1;
            }
        }

        /* If matrix was scaled, then rescale eigenvalues appropriately. */
        if (scale != ONE) {
            cblas_dscal(*m, ONE / scale, W, 1);
        }
    }

    /* If eigenvalues are not in increasing order, then sort them,
       possibly along with eigenvectors. */
    if (nsplit > 1 || n == 2) {
        if (!wantz) {
            dlasrt("I", *m, W, &iinfo);
            if (iinfo != 0) {
                *info = 3;
                return;
            }
        } else {
            for (j = 0; j < *m - 1; j++) {
                i = -1;
                tmp = W[j];
                for (jj = j + 1; jj < *m; jj++) {
                    if (W[jj] < tmp) {
                        i = jj;
                        tmp = W[jj];
                    }
                }
                if (i != -1) {
                    W[i] = W[j];
                    W[j] = tmp;
                    if (wantz) {
                        cblas_zswap(n, &Z[i * ldz], 1, &Z[j * ldz], 1);
                        itmp = isuppz[2 * i];
                        isuppz[2 * i] = isuppz[2 * j];
                        isuppz[2 * j] = itmp;
                        itmp = isuppz[2 * i + 1];
                        isuppz[2 * i + 1] = isuppz[2 * j + 1];
                        isuppz[2 * j + 1] = itmp;
                    }
                }
            }
        }
    }

    work[0] = lwmin;
    iwork[0] = liwmin;
}
