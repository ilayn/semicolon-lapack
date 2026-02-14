/**
 * @file zhpgvd.c
 * @brief ZHPGVD computes all the eigenvalues and, optionally, the eigenvectors of a complex generalized Hermitian-definite eigenproblem.
 */

#include "semicolon_lapack_complex_double.h"
#include <complex.h>
#include <cblas.h>
#include <math.h>

/**
 * ZHPGVD computes all the eigenvalues and, optionally, the eigenvectors
 * of a complex generalized Hermitian-definite eigenproblem, of the form
 * A*x=(lambda)*B*x,  A*Bx=(lambda)*x,  or B*A*x=(lambda)*x.  Here A and
 * B are assumed to be Hermitian, stored in packed format, and B is also
 * positive definite.
 * If eigenvectors are desired, it uses a divide and conquer algorithm.
 *
 * @param[in]     itype  Specifies the problem type to be solved:
 *                       = 1:  A*x = (lambda)*B*x
 *                       = 2:  A*B*x = (lambda)*x
 *                       = 3:  B*A*x = (lambda)*x
 * @param[in]     jobz   = 'N':  Compute eigenvalues only;
 *                        = 'V':  Compute eigenvalues and eigenvectors.
 * @param[in]     uplo   = 'U':  Upper triangles of A and B are stored;
 *                        = 'L':  Lower triangles of A and B are stored.
 * @param[in]     n      The order of the matrices A and B.  n >= 0.
 * @param[in,out] AP     Complex array, dimension (n*(n+1)/2).
 *                       On entry, the upper or lower triangle of the Hermitian
 *                       matrix A, packed columnwise in a linear array.
 *                       On exit, the contents of AP are destroyed.
 * @param[in,out] BP     Complex array, dimension (n*(n+1)/2).
 *                       On entry, the upper or lower triangle of the Hermitian
 *                       positive definite matrix B, packed columnwise.
 *                       On exit, the triangular factor U or L from the Cholesky
 *                       factorization B = U**H*U or B = L*L**H.
 * @param[out]    W      Double precision array, dimension (n).
 *                       If info = 0, the eigenvalues in ascending order.
 * @param[out]    Z      Complex array, dimension (ldz, n).
 *                       If jobz = 'V', then if info = 0, Z contains the matrix
 *                       Z of eigenvectors.
 * @param[in]     ldz    The leading dimension of the array Z.  ldz >= 1, and if
 *                       jobz = 'V', ldz >= max(1,n).
 * @param[out]    work   Complex array, dimension (max(1,lwork)).
 *                       On exit, if info = 0, work[0] returns the required lwork.
 * @param[in]     lwork  The dimension of the array work.
 *                       If n <= 1,               lwork >= 1.
 *                       If jobz = 'N' and n > 1, lwork >= n.
 *                       If jobz = 'V' and n > 1, lwork >= 2*n.
 *                       If lwork = -1, then a workspace query is assumed.
 * @param[out]    rwork  Double precision array, dimension (max(1,lrwork)).
 *                       On exit, if info = 0, rwork[0] returns the required lrwork.
 * @param[in]     lrwork The dimension of array rwork.
 *                       If n <= 1,               lrwork >= 1.
 *                       If jobz = 'N' and n > 1, lrwork >= n.
 *                       If jobz = 'V' and n > 1, lrwork >= 1 + 5*n + 2*n**2.
 *                       If lrwork = -1, then a workspace query is assumed.
 * @param[out]    iwork  Integer array, dimension (max(1,liwork)).
 *                       On exit, if info = 0, iwork[0] returns the required liwork.
 * @param[in]     liwork The dimension of array iwork.
 *                       If jobz = 'N' or n <= 1, liwork >= 1.
 *                       If jobz = 'V' and n > 1, liwork >= 3 + 5*n.
 *                       If liwork = -1, then a workspace query is assumed.
 * @param[out]    info   = 0:  successful exit
 *                       < 0:  if info = -i, the i-th argument had an illegal value
 *                       > 0:  ZPPTRF or ZHPEVD returned an error code:
 *                          <= n:  if info = i, ZHPEVD failed to converge;
 *                                 i off-diagonal elements of an intermediate
 *                                 tridiagonal form did not converge to zero;
 *                          > n:   if info = n + i, for 1 <= i <= n, then the leading
 *                                 principal minor of order i of B is not positive.
 */
void zhpgvd(
    const int itype,
    const char* jobz,
    const char* uplo,
    const int n,
    double complex* const restrict AP,
    double complex* const restrict BP,
    double* const restrict W,
    double complex* const restrict Z,
    const int ldz,
    double complex* const restrict work,
    const int lwork,
    double* const restrict rwork,
    const int lrwork,
    int* const restrict iwork,
    const int liwork,
    int* info)
{
    int wantz = (jobz[0] == 'V' || jobz[0] == 'v');
    int upper = (uplo[0] == 'U' || uplo[0] == 'u');
    int lquery = (lwork == -1 || lrwork == -1 || liwork == -1);

    *info = 0;
    if (itype < 1 || itype > 3) {
        *info = -1;
    } else if (!(wantz || jobz[0] == 'N' || jobz[0] == 'n')) {
        *info = -2;
    } else if (!(upper || uplo[0] == 'L' || uplo[0] == 'l')) {
        *info = -3;
    } else if (n < 0) {
        *info = -4;
    } else if (ldz < 1 || (wantz && ldz < n)) {
        *info = -9;
    }

    int lwmin, lrwmin, liwmin;
    if (*info == 0) {
        if (n <= 1) {
            lwmin = 1;
            liwmin = 1;
            lrwmin = 1;
        } else {
            if (wantz) {
                lwmin = 2 * n;
                lrwmin = 1 + 5 * n + 2 * n * n;
                liwmin = 3 + 5 * n;
            } else {
                lwmin = n;
                lrwmin = n;
                liwmin = 1;
            }
        }

        work[0] = CMPLX((double)lwmin, 0.0);
        rwork[0] = (double)lrwmin;
        iwork[0] = liwmin;
        if (lwork < lwmin && !lquery) {
            *info = -11;
        } else if (lrwork < lrwmin && !lquery) {
            *info = -13;
        } else if (liwork < liwmin && !lquery) {
            *info = -15;
        }
    }

    if (*info != 0) {
        xerbla("ZHPGVD", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    /* Quick return if possible */
    if (n == 0)
        return;

    /* Form a Cholesky factorization of B. */
    zpptrf(uplo, n, BP, info);
    if (*info != 0) {
        *info = n + *info;
        return;
    }

    /* Transform problem to standard eigenvalue problem and solve. */
    zhpgst(itype, uplo, n, AP, BP, info);
    zhpevd(jobz, uplo, n, AP, W, Z, ldz, work, lwork, rwork,
           lrwork, iwork, liwork, info);
    lwmin = (int)fmax((double)lwmin, creal(work[0]));
    lrwmin = (int)fmax((double)lrwmin, rwork[0]);
    liwmin = (int)fmax((double)liwmin, (double)iwork[0]);

    if (wantz) {

        /* Backtransform eigenvectors to the original problem. */

        int neig = n;
        if (*info > 0)
            neig = *info - 1;
        if (itype == 1 || itype == 2) {

            /* For A*x=(lambda)*B*x and A*B*x=(lambda)*x;
               backtransform eigenvectors: x = inv(L)**H *y or inv(U)*y */

            char trans;
            if (upper) {
                trans = 'N';
            } else {
                trans = 'C';
            }

            for (int j = 0; j < neig; j++) {
                cblas_ztpsv(CblasColMajor,
                            upper ? CblasUpper : CblasLower,
                            trans == 'N' ? CblasNoTrans : CblasConjTrans,
                            CblasNonUnit, n, BP, &Z[j * ldz], 1);
            }

        } else if (itype == 3) {

            /* For B*A*x=(lambda)*x;
               backtransform eigenvectors: x = L*y or U**H *y */

            char trans;
            if (upper) {
                trans = 'C';
            } else {
                trans = 'N';
            }

            for (int j = 0; j < neig; j++) {
                cblas_ztpmv(CblasColMajor,
                            upper ? CblasUpper : CblasLower,
                            trans == 'C' ? CblasConjTrans : CblasNoTrans,
                            CblasNonUnit, n, BP, &Z[j * ldz], 1);
            }
        }
    }

    work[0] = CMPLX((double)lwmin, 0.0);
    rwork[0] = (double)lrwmin;
    iwork[0] = liwmin;
}
