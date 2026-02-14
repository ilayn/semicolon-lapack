/**
 * @file zhpgv.c
 * @brief ZHPGV computes all the eigenvalues and, optionally, the eigenvectors
 *        of a complex generalized Hermitian-definite eigenproblem.
 */

#include "semicolon_lapack_complex_double.h"
#include <complex.h>
#include <cblas.h>

/**
 * ZHPGV computes all the eigenvalues and, optionally, the eigenvectors
 * of a complex generalized Hermitian-definite eigenproblem, of the form
 * A*x=(lambda)*B*x,  A*Bx=(lambda)*x,  or B*A*x=(lambda)*x.
 * Here A and B are assumed to be Hermitian, stored in packed format,
 * and B is also positive definite.
 *
 * @param[in]     itype  Specifies the problem type to be solved:
 *                       = 1: A*x = (lambda)*B*x
 *                       = 2: A*B*x = (lambda)*x
 *                       = 3: B*A*x = (lambda)*x
 * @param[in]     jobz   = 'N': Compute eigenvalues only;
 *                        = 'V': Compute eigenvalues and eigenvectors.
 * @param[in]     uplo   = 'U': Upper triangles of A and B are stored;
 *                        = 'L': Lower triangles of A and B are stored.
 * @param[in]     n      The order of the matrices A and B. n >= 0.
 * @param[in,out] AP     Complex*16 array, dimension (n*(n+1)/2).
 *                       On entry, the upper or lower triangle of the Hermitian
 *                       matrix A, packed columnwise in a linear array.
 *                       On exit, the contents of AP are destroyed.
 * @param[in,out] BP     Complex*16 array, dimension (n*(n+1)/2).
 *                       On entry, the upper or lower triangle of the Hermitian
 *                       positive definite matrix B, packed columnwise.
 *                       On exit, the triangular factor U or L from the Cholesky
 *                       factorization B = U**H*U or B = L*L**H.
 * @param[out]    W      Double precision array, dimension (n).
 *                       If info = 0, the eigenvalues in ascending order.
 * @param[out]    Z      Complex*16 array, dimension (ldz, n).
 *                       If jobz = 'V', then if info = 0, Z contains the matrix Z of
 *                       eigenvectors. The eigenvectors are normalized as follows:
 *                       if itype = 1 or 2, Z**H*B*Z = I;
 *                       if itype = 3, Z**H*inv(B)*Z = I.
 *                       If jobz = 'N', then Z is not referenced.
 * @param[in]     ldz    The leading dimension of the array Z. ldz >= 1, and if
 *                       jobz = 'V', ldz >= max(1,n).
 * @param[out]    work   Complex*16 array, dimension (max(1, 2*n-1)).
 * @param[out]    rwork  Double precision array, dimension (max(1, 3*n-2)).
 * @param[out]    info   = 0: successful exit
 *                       < 0: if info = -i, the i-th argument had an illegal value
 *                       > 0: ZPPTRF or ZHPEV returned an error code:
 *                          <= n: if info = i, ZHPEV failed to converge;
 *                                i off-diagonal elements of an intermediate
 *                                tridiagonal form did not converge to zero;
 *                          > n:  if info = n + i, for 1 <= i <= n, then the leading
 *                                principal minor of order i of B is not positive.
 *                                The factorization of B could not be completed and
 *                                no eigenvalues or eigenvectors were computed.
 */
void zhpgv(
    const int itype,
    const char* jobz,
    const char* uplo,
    const int n,
    c128* const restrict AP,
    c128* const restrict BP,
    f64* const restrict W,
    c128* const restrict Z,
    const int ldz,
    c128* const restrict work,
    f64* const restrict rwork,
    int* info)
{
    int wantz = (jobz[0] == 'V' || jobz[0] == 'v');
    int upper = (uplo[0] == 'U' || uplo[0] == 'u');

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
    if (*info != 0) {
        xerbla("ZHPGV ", -(*info));
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
    zhpev(jobz, uplo, n, AP, W, Z, ldz, work, rwork, info);

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
}
