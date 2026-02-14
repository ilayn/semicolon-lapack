/**
 * @file dstegr.c
 * @brief DSTEGR computes selected eigenvalues and, optionally, eigenvectors
 *        of a real symmetric tridiagonal matrix T.
 */

#include "semicolon_lapack_double.h"

/**
 * DSTEGR computes selected eigenvalues and, optionally, eigenvectors
 * of a real symmetric tridiagonal matrix T. Any such unreduced matrix has
 * a well defined set of pairwise different real eigenvalues, the corresponding
 * real eigenvectors are pairwise orthogonal.
 *
 * The spectrum may be computed either completely or partially by specifying
 * either an interval (vl,vu] or a range of indices il:iu for the desired
 * eigenvalues.
 *
 * DSTEGR is a compatibility wrapper around the improved DSTEMR routine.
 * See DSTEMR for further details.
 *
 * One important change is that the ABSTOL parameter no longer provides any
 * benefit and hence is no longer used.
 *
 * Note: DSTEGR and DSTEMR work only on machines which follow
 * IEEE-754 floating-point standard in their handling of infinities and
 * NaNs. Normal execution may create these exceptional values and hence
 * may abort due to a floating point exception in environments which
 * do not conform to the IEEE-754 standard.
 *
 * @param[in]     jobz   Specifies whether to compute eigenvalues only or
 *                       eigenvectors as well.
 *                       = 'N': Compute eigenvalues only
 *                       = 'V': Compute eigenvalues and eigenvectors
 * @param[in]     range  Specifies the range of eigenvalues to compute.
 *                       = 'A': All eigenvalues will be found
 *                       = 'V': All eigenvalues in the half-open interval (vl,vu]
 *                              will be found
 *                       = 'I': The il-th through iu-th eigenvalues will be found
 * @param[in]     n      The order of the matrix. n >= 0.
 * @param[in,out] D      Double precision array, dimension (n).
 *                       On entry, the n diagonal elements of the tridiagonal matrix T.
 *                       On exit, D is overwritten.
 * @param[in,out] E      Double precision array, dimension (n).
 *                       On entry, the (n-1) subdiagonal elements of the tridiagonal
 *                       matrix T in elements 0 to n-2 of E. E[n-1] need not be set
 *                       on input, but is used internally as workspace.
 *                       On exit, E is overwritten.
 * @param[in]     vl     If range = 'V', the lower bound of the interval to be
 *                       searched for eigenvalues. vl < vu.
 *                       Not referenced if range = 'A' or 'I'.
 * @param[in]     vu     If range = 'V', the upper bound of the interval to be
 *                       searched for eigenvalues. vl < vu.
 *                       Not referenced if range = 'A' or 'I'.
 * @param[in]     il     If range = 'I', the index of the smallest eigenvalue
 *                       to be returned (1-based).
 *                       1 <= il <= iu <= n, if n > 0.
 *                       Not referenced if range = 'A' or 'V'.
 * @param[in]     iu     If range = 'I', the index of the largest eigenvalue
 *                       to be returned (1-based).
 *                       1 <= il <= iu <= n, if n > 0.
 *                       Not referenced if range = 'A' or 'V'.
 * @param[in]     abstol Unused. Was the absolute error tolerance in previous versions.
 * @param[out]    m      The total number of eigenvalues found. 0 <= m <= n.
 *                       If range = 'A', m = n, and if range = 'I', m = iu-il+1.
 * @param[out]    W      Double precision array, dimension (n).
 *                       The first m elements contain the selected eigenvalues in
 *                       ascending order.
 * @param[out]    Z      Double precision array, dimension (ldz, max(1,m)).
 *                       If jobz = 'V', and if info = 0, then the first m columns
 *                       of Z contain the orthonormal eigenvectors of the matrix T
 *                       corresponding to the selected eigenvalues, with the i-th
 *                       column of Z holding the eigenvector associated with W[i].
 *                       If jobz = 'N', then Z is not referenced.
 *                       Note: the user must ensure that at least max(1,m) columns
 *                       are supplied in the array Z; if range = 'V', the exact
 *                       value of m is not known in advance and an upper bound
 *                       must be used. Supplying n columns is always safe.
 * @param[in]     ldz    The leading dimension of the array Z. ldz >= 1, and if
 *                       jobz = 'V', then ldz >= max(1,n).
 * @param[out]    isuppz Integer array, dimension (2*max(1,m)).
 *                       The support of the eigenvectors in Z, i.e., the indices
 *                       indicating the nonzero elements in Z. The i-th computed
 *                       eigenvector is nonzero only in elements isuppz[2*i] through
 *                       isuppz[2*i+1] (0-based). This is relevant in the case when
 *                       the matrix is split. isuppz is only accessed when jobz is
 *                       'V' and n > 0.
 * @param[out]    work   Double precision array, dimension (lwork).
 *                       On exit, if info = 0, work[0] returns the optimal
 *                       (and minimal) lwork.
 * @param[in]     lwork  The dimension of the array work. lwork >= max(1,18*n)
 *                       if jobz = 'V', and lwork >= max(1,12*n) if jobz = 'N'.
 *                       If lwork = -1, then a workspace query is assumed; the
 *                       routine only calculates the optimal size of the work
 *                       array, returns this value as the first entry of the
 *                       work array, and no error message related to lwork is
 *                       issued by xerbla.
 * @param[out]    iwork  Integer array, dimension (liwork).
 *                       On exit, if info = 0, iwork[0] returns the optimal liwork.
 * @param[in]     liwork The dimension of the array iwork. liwork >= max(1,10*n)
 *                       if the eigenvectors are desired, and liwork >= max(1,8*n)
 *                       if only the eigenvalues are to be computed.
 *                       If liwork = -1, then a workspace query is assumed; the
 *                       routine only calculates the optimal size of the iwork
 *                       array, returns this value as the first entry of the
 *                       iwork array, and no error message related to liwork is
 *                       issued by xerbla.
 * @param[out]    info
 *                           Exit status:
 *                           - = 0: successful exit
 *                           - < 0: if info = -i, the i-th argument had an illegal value
 *                           - > 0: if info = 1X, internal error in DLARRE,
 *                           if info = 2X, internal error in DLARRV.
 *                           Here, the digit X = ABS(iinfo) < 10, where iinfo is
 *                           the nonzero error code returned by DLARRE or DLARRV.
 */
void dstegr(
    const char* jobz,
    const char* range,
    const int n,
    f64* const restrict D,
    f64* const restrict E,
    const f64 vl,
    const f64 vu,
    const int il,
    const int iu,
    const f64 abstol,
    int* m,
    f64* const restrict W,
    f64* const restrict Z,
    const int ldz,
    int* const restrict isuppz,
    f64* const restrict work,
    const int lwork,
    int* const restrict iwork,
    const int liwork,
    int* info)
{
    int tryrac = 0;  // FALSE: don't try to use the high relative accuracy algorithm

    // DSTEGR is simply a wrapper around DSTEMR
    // Note: abstol is unused in dstemr, tryrac is set to false
    (void)abstol;  // Unused parameter

    *info = 0;

    dstemr(jobz, range, n, D, E, vl, vu, il, iu, m, W, Z, ldz, n, isuppz,
           &tryrac, work, lwork, iwork, liwork, info);
}
