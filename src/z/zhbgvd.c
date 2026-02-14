/**
 * @file zhbgvd.c
 * @brief ZHBGVD computes all eigenvalues and, optionally, eigenvectors of a
 *        complex generalized Hermitian-definite banded eigenproblem.
 */

#include "semicolon_lapack_complex_double.h"
#include <complex.h>
#include <cblas.h>

/**
 * ZHBGVD computes all the eigenvalues, and optionally, the eigenvectors
 * of a complex generalized Hermitian-definite banded eigenproblem, of
 * the form A*x=(lambda)*B*x. Here A and B are assumed to be Hermitian
 * and banded, and B is also positive definite.  If eigenvectors are
 * desired, it uses a divide and conquer algorithm.
 *
 * @param[in]     jobz   = 'N':  Compute eigenvalues only;
 *                         = 'V':  Compute eigenvalues and eigenvectors.
 * @param[in]     uplo   = 'U':  Upper triangles of A and B are stored;
 *                         = 'L':  Lower triangles of A and B are stored.
 * @param[in]     n      The order of the matrices A and B.  n >= 0.
 * @param[in]     ka     The number of superdiagonals of the matrix A if
 *                       uplo = 'U', or the number of subdiagonals if
 *                       uplo = 'L'. ka >= 0.
 * @param[in]     kb     The number of superdiagonals of the matrix B if
 *                       uplo = 'U', or the number of subdiagonals if
 *                       uplo = 'L'. kb >= 0.
 * @param[in,out] AB     Complex array, dimension (ldab, n).
 *                       On entry, the upper or lower triangle of the Hermitian
 *                       band matrix A, stored in the first ka+1 rows.
 *                       On exit, the contents of AB are destroyed.
 * @param[in]     ldab   The leading dimension of AB. ldab >= ka+1.
 * @param[in,out] BB     Complex array, dimension (ldbb, n).
 *                       On entry, the upper or lower triangle of the Hermitian
 *                       band matrix B, stored in the first kb+1 rows.
 *                       On exit, the factor S from the split Cholesky
 *                       factorization B = S**H*S, as returned by ZPBSTF.
 * @param[in]     ldbb   The leading dimension of BB. ldbb >= kb+1.
 * @param[out]    W      Double precision array, dimension (n).
 *                       If info = 0, the eigenvalues in ascending order.
 * @param[out]    Z      Complex array, dimension (ldz, n).
 *                       If jobz = 'V', then if info = 0, Z contains the
 *                       matrix Z of eigenvectors.
 *                       If jobz = 'N', then Z is not referenced.
 * @param[in]     ldz    The leading dimension of Z. ldz >= 1, and if
 *                       jobz = 'V', ldz >= n.
 * @param[out]    work   Complex array, dimension (max(1,lwork)).
 *                       On exit, if info=0, work[0] returns the optimal lwork.
 * @param[in]     lwork  The dimension of the array work.
 *                       If n <= 1,               lwork >= 1.
 *                       If jobz = 'N' and n > 1, lwork >= n.
 *                       If jobz = 'V' and n > 1, lwork >= 2*n**2.
 *                       If lwork = -1, then a workspace query is assumed.
 * @param[out]    rwork  Double precision array, dimension (max(1,lrwork)).
 *                       On exit, if info=0, rwork[0] returns the optimal lrwork.
 * @param[in]     lrwork The dimension of array rwork.
 *                       If n <= 1,               lrwork >= 1.
 *                       If jobz = 'N' and n > 1, lrwork >= n.
 *                       If jobz = 'V' and n > 1, lrwork >= 1 + 5*n + 2*n**2.
 *                       If lrwork = -1, then a workspace query is assumed.
 * @param[out]    iwork  Integer array, dimension (max(1,liwork)).
 *                       On exit, if info=0, iwork[0] returns the optimal liwork.
 * @param[in]     liwork The dimension of array iwork.
 *                       If jobz = 'N' or n <= 1, liwork >= 1.
 *                       If jobz = 'V' and n > 1, liwork >= 3 + 5*n.
 *                       If liwork = -1, then a workspace query is assumed.
 * @param[out]    info   = 0:  successful exit
 *                       < 0:  if info = -i, the i-th argument had an illegal value
 *                       > 0:  if info = i, and i is:
 *                          <= n:  the algorithm failed to converge
 *                          > n:   if info = n + i, for 1 <= i <= n, then ZPBSTF
 *                                 returned info = i: B is not positive definite.
 */
void zhbgvd(
    const char* jobz,
    const char* uplo,
    const int n,
    const int ka,
    const int kb,
    c128* const restrict AB,
    const int ldab,
    c128* const restrict BB,
    const int ldbb,
    f64* const restrict W,
    c128* const restrict Z,
    const int ldz,
    c128* const restrict work,
    const int lwork,
    f64* const restrict rwork,
    const int lrwork,
    int* const restrict iwork,
    const int liwork,
    int* info)
{
    const c128 CONE = CMPLX(1.0, 0.0);
    const c128 CZERO = CMPLX(0.0, 0.0);

    int iinfo;
    int inde, indwk2, indwrk, liwmin, llrwk, llwk2, lrwmin, lwmin;
    char vect;

    int wantz = (jobz[0] == 'V' || jobz[0] == 'v');
    int upper = (uplo[0] == 'U' || uplo[0] == 'u');
    int lquery = (lwork == -1 || lrwork == -1 || liwork == -1);

    *info = 0;
    if (n <= 1) {
        lwmin = 1 + n;
        lrwmin = 1 + n;
        liwmin = 1;
    } else if (wantz) {
        lwmin = 2 * n * n;
        lrwmin = 1 + 5 * n + 2 * n * n;
        liwmin = 3 + 5 * n;
    } else {
        lwmin = n;
        lrwmin = n;
        liwmin = 1;
    }
    if (!(wantz || jobz[0] == 'N' || jobz[0] == 'n')) {
        *info = -1;
    } else if (!(upper || uplo[0] == 'L' || uplo[0] == 'l')) {
        *info = -2;
    } else if (n < 0) {
        *info = -3;
    } else if (ka < 0) {
        *info = -4;
    } else if (kb < 0 || kb > ka) {
        *info = -5;
    } else if (ldab < ka + 1) {
        *info = -7;
    } else if (ldbb < kb + 1) {
        *info = -9;
    } else if (ldz < 1 || (wantz && ldz < n)) {
        *info = -12;
    }

    if (*info == 0) {
        work[0] = CMPLX((f64)lwmin, 0.0);
        rwork[0] = (f64)lrwmin;
        iwork[0] = liwmin;

        if (lwork < lwmin && !lquery) {
            *info = -14;
        } else if (lrwork < lrwmin && !lquery) {
            *info = -16;
        } else if (liwork < liwmin && !lquery) {
            *info = -18;
        }
    }

    if (*info != 0) {
        xerbla("ZHBGVD", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    if (n == 0) {
        return;
    }

    zpbstf(uplo, n, kb, BB, ldbb, info);
    if (*info != 0) {
        *info = n + *info;
        return;
    }

    inde = 0;
    indwrk = inde + n;
    indwk2 = n * n;
    llwk2 = lwork - indwk2;
    llrwk = lrwork - indwrk;
    zhbgst(jobz, uplo, n, ka, kb, AB, ldab, BB, ldbb, Z, ldz,
           work, rwork, &iinfo);

    if (wantz) {
        vect = 'U';
    } else {
        vect = 'N';
    }
    zhbtrd(&vect, uplo, n, ka, AB, ldab, W, &rwork[inde], Z,
           ldz, work, &iinfo);

    if (!wantz) {
        dsterf(n, W, &rwork[inde], info);
    } else {
        zstedc("I", n, W, &rwork[inde], work, n,
               &work[indwk2], llwk2, &rwork[indwrk], llrwk, iwork, liwork,
               info);
        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    n, n, n, &CONE, Z, ldz, work, n, &CZERO,
                    &work[indwk2], n);
        zlacpy("A", n, n, &work[indwk2], n, Z, ldz);
    }

    work[0] = CMPLX((f64)lwmin, 0.0);
    rwork[0] = (f64)lrwmin;
    iwork[0] = liwmin;
}
