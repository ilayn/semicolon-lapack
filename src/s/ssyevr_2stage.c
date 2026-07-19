/**
 * @file ssyevr_2stage.c
 * @brief SSYEVR_2STAGE computes selected eigenvalues and optionally eigenvectors using RRR.
 */

#include "semicolon_lapack_single.h"
#include "semicolon_cblas.h"
#include <math.h>

/**
 * SSYEVR_2STAGE computes selected eigenvalues and, optionally, eigenvectors
 * of a real symmetric matrix `A` using the 2stage technique for
 * the reduction to tridiagonal. Eigenvalues and eigenvectors can be
 * selected by specifying either a range of values or a range of
 * indices for the desired eigenvalues.
 *
 * SSYEVR_2STAGE first reduces the matrix `A` to tridiagonal form T with a call
 * to SSYTRD. Then, whenever possible, SSYEVR_2STAGE calls SSTEMR to compute
 * the eigenspectrum using Relatively Robust Representations. SSTEMR
 * computes eigenvalues by the dqds algorithm, while orthogonal
 * eigenvectors are computed from various "good" L D L^T representations
 * (also known as Relatively Robust Representations). Gram-Schmidt
 * orthogonalization is avoided as far as possible. More specifically,
 * the various steps of the algorithm are as follows.
 *
 * For each unreduced block (submatrix) of T,
 * @rst
 * .. code-block:: text
 *
 *     (a) Compute T - sigma I  = L D L^T, so that L and D
 *         define all the wanted eigenvalues to high relative accuracy.
 *         This means that small relative changes in the entries of D and L
 *         cause only small relative changes in the eigenvalues and
 *         eigenvectors. The standard (unfactored) representation of the
 *         tridiagonal matrix T does not have this property in general.
 *     (b) Compute the eigenvalues to suitable accuracy.
 *         If the eigenvectors are desired, the algorithm attains full
 *         accuracy of the computed eigenvalues only right before
 *         the corresponding vectors have to be computed, see steps c) and d).
 *     (c) For each cluster of close eigenvalues, select a new
 *         shift close to the cluster, find a new factorization, and refine
 *         the shifted eigenvalues to suitable accuracy.
 *     (d) For each eigenvalue with a large enough relative separation compute
 *         the corresponding eigenvector by forming a rank revealing twisted
 *         factorization. Go back to (c) for any clusters that remain.
 * @endrst
 * The desired accuracy of the output can be specified by the input
 * parameter `abstol`.
 *
 * For more details, see SSTEMR's documentation and:
 * @rst
 * Inderjit S. Dhillon and Beresford N. Parlett: "Multiple representations
 * to compute orthogonal eigenvectors of symmetric tridiagonal matrices,"
 * Linear Algebra and its Applications, 387(1), pp. 1-28, August 2004.
 *
 * Inderjit Dhillon and Beresford Parlett: "Orthogonal Eigenvectors and
 * Relative Gaps," SIAM Journal on Matrix Analysis and Applications, Vol. 25,
 * 2004. Also LAPACK Working Note 154.
 *
 * Inderjit Dhillon: "A new O(n^2) algorithm for the symmetric
 * tridiagonal eigenvalue/eigenvector problem",
 * Computer Science Division Technical Report No. UCB/CSD-97-971,
 * UC Berkeley, May 1997.
 * @endrst
 * Note 1 : SSYEVR_2STAGE calls SSTEMR when the full spectrum is requested
 * on machines which conform to the ieee-754 floating point standard.
 * SSYEVR_2STAGE calls SSTEBZ and SSTEIN on non-ieee machines and
 * when partial spectrum requests are made.
 *
 * Normal execution of SSTEMR may create NaNs and infinities and
 * hence may abort due to a floating point exception in environments
 * which do not handle NaNs and infinities in the ieee standard default
 * manner.
 *
 * @param[in]     jobz   `jobz='N'`: Compute eigenvalues only;
 *                       `jobz='V'`: Compute eigenvalues and eigenvectors.
 *                       Not available in this release.
 * @param[in]     range  `range='A'`: all eigenvalues will be found.
 *                       `range='V'`: all eigenvalues in the half-open interval
 *                       (`vl`,`vu`] will be found.
 *                       `range='I'`: the `il`-th through `iu`-th eigenvalues will
 *                       be found.
 *                       For `range='V'` or `range='I'` and `iu-il<n-1`, SSTEBZ and
 *                       SSTEIN are called.
 * @param[in]     uplo   `uplo='U'`: Upper triangle of `A` is stored;
 *                       `uplo='L'`: Lower triangle of `A` is stored.
 * @param[in]     n      The order of the matrix `A`. `n>=0`.
 * @param[in,out] A      Array of dimension (`lda`, `n`). On entry, the symmetric
 *                       matrix `A`. If `uplo='U'`, the leading `n`-by-`n` upper
 *                       triangular part of `A` contains the upper triangular part
 *                       of the matrix `A`. If `uplo='L'`, the leading `n`-by-`n`
 *                       lower triangular part of `A` contains the lower triangular
 *                       part of the matrix `A`.
 *                       On exit, the lower triangle (if `uplo='L'`) or the upper
 *                       triangle (if `uplo='U'`) of `A`, including the diagonal, is
 *                       destroyed.
 * @param[in]     lda    The leading dimension of the array `A`. `lda>=max(1,n)`.
 * @param[in]     vl     If `range='V'`, the lower bound of the interval to
 *                       be searched for eigenvalues. `vl<vu`.
 *                       Not referenced if `range='A'` or `range='I'`.
 * @param[in]     vu     If `range='V'`, the upper bound of the interval to
 *                       be searched for eigenvalues. `vl<vu`.
 *                       Not referenced if `range='A'` or `range='I'`.
 * @param[in]     il     If `range='I'`, the index of the smallest eigenvalue to be
 *                       returned. `0<=il<=iu<=n-1`, if `n>0`; `il=0` and `iu=-1`
 *                       if `n=0`. Not referenced if `range='A'` or `range='V'`.
 * @param[in]     iu     If `range='I'`, the index of the largest eigenvalue to be
 *                       returned. `0<=il<=iu<=n-1`, if `n>0`; `il=0` and `iu=-1`
 *                       if `n=0`. Not referenced if `range='A'` or `range='V'`.
 * @param[in]     abstol The absolute error tolerance for the eigenvalues.
 *                       An approximate eigenvalue is accepted as converged
 *                       when it is determined to lie in an interval [a,b]
 *                       of width less than or equal to
 *                       `abstol + EPS * max(|a|,|b|)`,
 *                       where EPS is the machine precision. If `abstol` is less than
 *                       or equal to zero, then `EPS*|T|` will be used in its place,
 *                       where `|T|` is the 1-norm of the tridiagonal matrix obtained
 *                       by reducing `A` to tridiagonal form.
 *                       See "Computing Small Singular Values of Bidiagonal Matrices
 *                       with Guaranteed High Relative Accuracy," by Demmel and
 *                       Kahan, LAPACK Working Note #3.
 *                       If high relative accuracy is important, set `abstol` to
 *                       `slamch("Safe minimum")`. Doing so will guarantee that
 *                       eigenvalues are computed to high relative accuracy when
 *                       possible in future releases. The current code does not
 *                       make any guarantees about high relative accuracy, but
 *                       future releases will. See J. Barlow and J. Demmel,
 *                       "Computing Accurate Eigensystems of Scaled Diagonally
 *                       Dominant Matrices", LAPACK Working Note #7, for a discussion
 *                       of which matrices define their eigenvalues to high relative
 *                       accuracy.
 * @param[out]    m      The total number of eigenvalues found. `0<=m<=n`.
 *                       If `range='A'`, `m=n`, and if `range='I'`, `m=iu-il+1`.
 * @param[out]    W      Array of dimension (`n`). The first `m` elements contain the
 *                       selected eigenvalues in ascending order.
 * @param[out]    Z      Array of dimension (`ldz`, `max(1,m)`).
 *                       If `jobz='V'`, then if `info=0`, the first `m` columns of `Z`
 *                       contain the orthonormal eigenvectors of the matrix `A`
 *                       corresponding to the selected eigenvalues, with the i-th
 *                       column of `Z` holding the eigenvector associated with `W[i]`.
 *                       If `jobz='N'`, then `Z` is not referenced.
 *                       Note: the user must ensure that at least `max(1,m)` columns are
 *                       supplied in the array `Z`; if `range='V'`, the exact value of `m`
 *                       is not known in advance and an upper bound must be used.
 *                       Supplying `n` columns is always safe.
 * @param[in]     ldz    The leading dimension of the array `Z`. `ldz>=1`, and if
 *                       `jobz='V'`, `ldz>=max(1,n)`.
 * @param[out]    isuppz Integer array of dimension (`2*max(1,m)`).
 *                       The support of the eigenvectors in `Z`, i.e., the indices
 *                       indicating the nonzero elements in `Z`. The i-th eigenvector
 *                       is nonzero only in elements `isuppz[2*i]` through
 *                       `isuppz[2*i+1]`. This is an output of SSTEMR (tridiagonal
 *                       matrix). The support of the eigenvectors of `A` is typically
 *                       `0:n-1` because of the orthogonal transformations applied by
 *                       SORMTR.
 *                       Implemented only for `range='A'` or `range='I'` and `iu-il=n-1`.
 * @param[out]    work   Array of dimension (`max(1,lwork)`). On exit, if `info=0`,
 *                       `work[0]` returns the optimal `lwork`.
 * @param[in]     lwork  The dimension of the array `work`. `lwork>=1`, when `n<=1`;
 *                       otherwise:
 *                       If `jobz='N'` and `n>1`, `lwork` must be queried.
 *                       @rst
 *                       .. code-block:: text
 *
 *                           lwork = MAX(1, 26*n, dimension) where
 *                           dimension = max(stage1,stage2) + (kd+1)*n + 5*n
 *                                     = n*kd + n*max(kd+1,FACTOPTNB)
 *                                       + max(2*kd*kd, kd*NTHREADS)
 *                                       + (kd+1)*n + 5*n
 *                       @endrst
 *                       where `kd` is the blocking size of the reduction,
 *                       FACTOPTNB is the blocking used by the QR or LQ
 *                       algorithm, usually FACTOPTNB=128 is a good choice,
 *                       NTHREADS is the number of threads used when openMP
 *                       compilation is enabled, otherwise =1.
 *                       If `jobz='V'` and `n>1`, `lwork` must be queried. Not yet
 *                       available.
 *                       If `lwork=-1`, then a workspace query is assumed; the routine
 *                       only calculates the optimal size of the `work` array, returns
 *                       this value as the first entry of the `work` array, and no error
 *                       message related to `lwork` is issued.
 * @param[out]    iwork  Integer array of dimension (`max(1,liwork)`). On exit, if
 *                       `info=0`, `iwork[0]` returns the optimal `liwork`.
 * @param[in]     liwork The dimension of the array `iwork`.
 *                       If `n<=1`, `liwork>=1`, else `liwork>=10*n`.
 *                       If `liwork=-1`, then a workspace query is assumed; the
 *                       routine only calculates the optimal size of the `iwork` array,
 *                       returns this value as the first entry of the `iwork` array, and
 *                       no error message related to `liwork` is issued.
 * @param[out]    info
 *                       - `info=0`: successful exit
 *                       - `info<0`: if `info=-i`, the i-th argument had an illegal value
 *                       - `info>0`: Internal error
 *
 * @par Contributors:
 * @rst
 * | Inderjit Dhillon, IBM Almaden, USA
 * | Osni Marques, LBNL/NERSC, USA
 * | Ken Stanley, Computer Science Division, University of California at Berkeley, USA
 * | Jason Riedy, Computer Science Division, University of California at Berkeley, USA
 * @endrst
 *
 * @par Further Details:
 * @rst
 * All details about the 2stage techniques are available in:
 *
 * Azzam Haidar, Hatem Ltaief, and Jack Dongarra.
 * Parallel reduction to condensed forms for symmetric eigenvalue problems
 * using aggregated fine-grained and memory-aware kernels. In Proceedings
 * of 2011 International Conference for High Performance Computing,
 * Networking, Storage and Analysis (SC '11), New York, NY, USA,
 * Article 8 , 11 pages.
 * https://doi.org/10.1145/2063384.2063394
 *
 * A. Haidar, J. Kurzak, P. Luszczek, 2013.
 * An improved parallel singular value algorithm and its implementation
 * for multicore hardware, In Proceedings of 2013 International Conference
 * for High Performance Computing, Networking, Storage and Analysis (SC '13).
 * Denver, Colorado, USA, 2013.
 * Article 90, 12 pages.
 * https://doi.org/10.1145/2503210.2503292
 *
 * A. Haidar, R. Solca, S. Tomov, T. Schulthess and J. Dongarra.
 * A novel hybrid CPU-GPU generalized eigensolver for electronic structure
 * calculations based on fine-grained memory aware tasks.
 * International Journal of High Performance Computing Applications.
 * Volume 28 Issue 2, Pages 196-209, May 2014.
 * https://doi.org/10.1177/1094342013502097
 * @endrst
 */
void ssyevr_2stage(const char* jobz, const char* range, const char* uplo,
                   const INT n, f32* A, const INT lda,
                   const f32 vl, const f32 vu,
                   const INT il, const INT iu,
                   const f32 abstol, INT* m,
                   f32* W, f32* Z, const INT ldz, INT* isuppz,
                   f32* work, const INT lwork,
                   INT* iwork, const INT liwork, INT* info)
{
    const f32 zero = 0.0f;
    const f32 one = 1.0f;
    const f32 two = 2.0f;

    INT alleig, indeig, lower, lquery, valeig, wantz, tryrac;
    char order;
    INT i, ieeeok, iinfo, imax, indd, inddd, inde;
    INT indee, indibl, indifl, indisp, indiwo, indtau;
    INT indwk, indwkn, iscale, j, jj, liwmin;
    INT llwork, llwrkn, lwmin, nsplit;
    INT lhtrd, lwtrd, kd, ib, indhous;
    f32 abstll, anrm, bignum, eps, rmax, rmin, safmin;
    f32 sigma, smlnum, tmp1, vll, vuu;

    /* IEEEOK = 1 means IEEE arithmetic is assumed (NaN/Inf handled properly) */
    ieeeok = 1;

    lower = (uplo[0] == 'L' || uplo[0] == 'l');
    wantz = (jobz[0] == 'V' || jobz[0] == 'v');
    alleig = (range[0] == 'A' || range[0] == 'a');
    valeig = (range[0] == 'V' || range[0] == 'v');
    indeig = (range[0] == 'I' || range[0] == 'i');

    lquery = (lwork == -1) || (liwork == -1);

    kd = ilaenv2stage(1, "SSYTRD_2STAGE", jobz, n, -1, -1, -1);
    ib = ilaenv2stage(2, "SSYTRD_2STAGE", jobz, n, kd, -1, -1);
    lhtrd = ilaenv2stage(3, "SSYTRD_2STAGE", jobz, n, kd, ib, -1);
    lwtrd = ilaenv2stage(4, "SSYTRD_2STAGE", jobz, n, kd, ib, -1);

    if (n <= 1) {
        lwmin = 1;
        liwmin = 1;
    } else {
        lwmin = (26 * n > 5 * n + lhtrd + lwtrd) ? 26 * n : 5 * n + lhtrd + lwtrd;
        liwmin = 10 * n;
    }

    *info = 0;
    if (!(jobz[0] == 'N' || jobz[0] == 'n')) {
        *info = -1;
    } else if (!(alleig || valeig || indeig)) {
        *info = -2;
    } else if (!lower && !(uplo[0] == 'U' || uplo[0] == 'u')) {
        *info = -3;
    } else if (n < 0) {
        *info = -4;
    } else if (lda < ((1 > n) ? 1 : n)) {
        *info = -6;
    } else {
        if (valeig) {
            if (n > 0 && vu <= vl) {
                *info = -8;
            }
        } else if (indeig) {
            if (il < 0 || il > ((0 > n - 1) ? 0 : n - 1)) {
                *info = -9;
            } else if (iu < ((n - 1 < il) ? n - 1 : il) || iu > n - 1) {
                *info = -10;
            }
        }
    }
    if (*info == 0) {
        if (ldz < 1 || (wantz && ldz < n)) {
            *info = -15;
        } else if (lwork < lwmin && !lquery) {
            *info = -18;
        } else if (liwork < liwmin && !lquery) {
            *info = -20;
        }
    }

    if (*info == 0) {
        work[0] = (f32)lwmin;
        iwork[0] = liwmin;
    }

    if (*info != 0) {
        xerbla("SSYEVR_2STAGE", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    *m = 0;
    if (n == 0) {
        work[0] = 1.0f;
        return;
    }

    if (n == 1) {
        work[0] = 1.0f;
        if (alleig || indeig) {
            *m = 1;
            W[0] = A[0];
        } else {
            if (vl < A[0] && vu >= A[0]) {
                *m = 1;
                W[0] = A[0];
            }
        }
        if (wantz) {
            Z[0] = one;
            isuppz[0] = 1;
            isuppz[1] = 1;
        }
        return;
    }

    safmin = slamch("Safe minimum");
    eps = slamch("Precision");
    smlnum = safmin / eps;
    bignum = one / smlnum;
    rmin = sqrtf(smlnum);
    rmax = (sqrtf(bignum) < one / sqrtf(sqrtf(safmin))) ? sqrtf(bignum) : one / sqrtf(sqrtf(safmin));

    iscale = 0;
    abstll = abstol;
    vll = vl;
    vuu = vu;
    anrm = slansy("M", uplo, n, A, lda, work);
    if (anrm > zero && anrm < rmin) {
        iscale = 1;
        sigma = rmin / anrm;
    } else if (anrm > rmax) {
        iscale = 1;
        sigma = rmax / anrm;
    }
    if (iscale == 1) {
        if (lower) {
            for (j = 0; j < n; j++) {
                cblas_sscal(n - j, sigma, &A[j + j * lda], 1);
            }
        } else {
            for (j = 0; j < n; j++) {
                cblas_sscal(j + 1, sigma, &A[0 + j * lda], 1);
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

    indtau = 0;
    indd = indtau + n;
    inde = indd + n;
    inddd = inde + n;
    indee = inddd + n;
    indhous = indee + n;
    indwk = indhous + lhtrd;
    llwork = lwork - indwk;

    indibl = 0;
    indisp = indibl + n;
    indifl = indisp + n;
    indiwo = indifl + n;

    ssytrd_2stage(jobz, uplo, n, A, lda, &work[indd],
                  &work[inde], &work[indtau], &work[indhous],
                  lhtrd, &work[indwk], llwork, &iinfo);

    if ((alleig || (indeig && il == 0 && iu == n - 1)) && ieeeok == 1) {
        if (!wantz) {
            cblas_scopy(n, &work[indd], 1, W, 1);
            cblas_scopy(n - 1, &work[inde], 1, &work[indee], 1);
            ssterf(n, W, &work[indee], info);
        } else {
            cblas_scopy(n - 1, &work[inde], 1, &work[indee], 1);
            cblas_scopy(n, &work[indd], 1, &work[inddd], 1);

            if (abstol <= two * n * eps) {
                tryrac = 1;
            } else {
                tryrac = 0;
            }
            sstemr(jobz, "A", n, &work[inddd], &work[indee],
                   vl, vu, il, iu, m, W, Z, ldz, n, isuppz,
                   &tryrac, &work[indwk], lwork, iwork, liwork, info);

            if (wantz && *info == 0) {
                indwkn = inde;
                llwrkn = lwork - indwkn;
                sormtr("L", uplo, "N", n, *m, A, lda,
                       &work[indtau], Z, ldz, &work[indwkn],
                       llwrkn, &iinfo);
            }
        }

        if (*info == 0) {
            *m = n;
            goto label30;
        }
        *info = 0;
    }

    if (wantz) {
        order = 'B';
    } else {
        order = 'E';
    }

    {
        char order_str[2] = {order, '\0'};
        sstebz(range, order_str, n, vll, vuu, il, iu, abstll,
               &work[indd], &work[inde], m, &nsplit, W,
               &iwork[indibl], &iwork[indisp], &work[indwk],
               &iwork[indiwo], info);
    }

    if (wantz) {
        sstein(n, &work[indd], &work[inde], *m, W,
               &iwork[indibl], &iwork[indisp], Z, ldz,
               &work[indwk], &iwork[indiwo], &iwork[indifl], info);

        indwkn = inde;
        llwrkn = lwork - indwkn;
        sormtr("L", uplo, "N", n, *m, A, lda, &work[indtau],
               Z, ldz, &work[indwkn], llwrkn, &iinfo);
    }

label30:
    if (iscale == 1) {
        if (*info == 0) {
            imax = *m;
        } else {
            imax = *info - 1;
        }
        cblas_sscal(imax, one / sigma, W, 1);
    }

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
                W[i] = W[j];
                W[j] = tmp1;
                cblas_sswap(n, &Z[0 + i * ldz], 1, &Z[0 + j * ldz], 1);
            }
        }
    }

    work[0] = (f32)lwmin;
    iwork[0] = liwmin;
}
