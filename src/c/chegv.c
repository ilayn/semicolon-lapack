/**
 * @file chegv.c
 * @brief CHEGV computes all the eigenvalues, and optionally, the eigenvectors
 *        of a complex generalized Hermitian-definite eigenproblem.
 */

#include "semicolon_lapack_complex_single.h"
#include <complex.h>
#include <cblas.h>
#include "lapack_tuning.h"

/**
 * CHEGV computes all the eigenvalues, and optionally, the eigenvectors
 * of a complex generalized Hermitian-definite eigenproblem, of the form
 * A*x=(lambda)*B*x,  A*Bx=(lambda)*x,  or B*A*x=(lambda)*x.
 * Here A and B are assumed to be Hermitian and B is also
 * positive definite.
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
 * @param[in,out] A      Complex*16 array, dimension (lda, n).
 *                       On entry, the Hermitian matrix A.
 *                       On exit, if jobz = 'V', then if info = 0, A contains
 *                       the matrix Z of eigenvectors.
 * @param[in]     lda    The leading dimension of the array A. lda >= max(1,n).
 * @param[in,out] B      Complex*16 array, dimension (ldb, n).
 *                       On entry, the Hermitian positive definite matrix B.
 *                       On exit, if info <= n, the part of B containing the
 *                       matrix is overwritten by the triangular factor U or L
 *                       from the Cholesky factorization B = U**H*U or B = L*L**H.
 * @param[in]     ldb    The leading dimension of the array B. ldb >= max(1,n).
 * @param[out]    W      Single precision array, dimension (n).
 *                       If info = 0, the eigenvalues in ascending order.
 * @param[out]    work   Complex*16 array, dimension (max(1,lwork)).
 *                       On exit, if info = 0, work[0] returns the optimal lwork.
 * @param[in]     lwork  The length of the array work. lwork >= max(1,2*n-1).
 *                       For optimal efficiency, lwork >= (nb+1)*n,
 *                       where nb is the blocksize for CHETRD.
 *                       If lwork = -1, then a workspace query is assumed.
 * @param[out]    rwork  Single precision array, dimension (max(1, 3*n-2)).
 * @param[out]    info   = 0: successful exit
 *                       < 0: if info = -i, the i-th argument had an illegal value
 *                       > 0: CPOTRF or CHEEV returned an error code:
 *                          <= n: if info = i, CHEEV failed to converge;
 *                          > n: if info = n + i, for 1 <= i <= n, then the leading
 *                               principal minor of order i of B is not positive.
 */
void chegv(
    const int itype,
    const char* jobz,
    const char* uplo,
    const int n,
    c64* restrict A,
    const int lda,
    c64* restrict B,
    const int ldb,
    f32* restrict W,
    c64* restrict work,
    const int lwork,
    f32* restrict rwork,
    int* info)
{
    const c64 ONE = CMPLXF(1.0f, 0.0f);

    int wantz = (jobz[0] == 'V' || jobz[0] == 'v');
    int upper = (uplo[0] == 'U' || uplo[0] == 'u');
    int lquery = (lwork == -1);

    *info = 0;
    if (itype < 1 || itype > 3) {
        *info = -1;
    } else if (!wantz && !(jobz[0] == 'N' || jobz[0] == 'n')) {
        *info = -2;
    } else if (!upper && !(uplo[0] == 'L' || uplo[0] == 'l')) {
        *info = -3;
    } else if (n < 0) {
        *info = -4;
    } else if (lda < (1 > n ? 1 : n)) {
        *info = -6;
    } else if (ldb < (1 > n ? 1 : n)) {
        *info = -8;
    }

    int lwkopt = 1;
    if (*info == 0) {
        int nb = lapack_get_nb("HETRD");
        lwkopt = (nb + 1) * n;
        if (lwkopt < 1) {
            lwkopt = 1;
        }
        work[0] = CMPLXF((f32)lwkopt, 0.0f);

        if (lwork < (1 > 2 * n - 1 ? 1 : 2 * n - 1) && !lquery) {
            *info = -11;
        }
    }

    if (*info != 0) {
        xerbla("CHEGV ", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    if (n == 0) {
        return;
    }

    cpotrf(uplo, n, B, ldb, info);
    if (*info != 0) {
        *info = n + *info;
        return;
    }

    chegst(itype, uplo, n, A, lda, B, ldb, info);
    cheev(jobz, uplo, n, A, lda, W, work, lwork, rwork, info);

    if (wantz) {

        int neig = n;
        if (*info > 0) {
            neig = *info - 1;
        }
        if (itype == 1 || itype == 2) {

            char trans;
            if (upper) {
                trans = 'N';
            } else {
                trans = 'C';
            }

            cblas_ctrsm(CblasColMajor, CblasLeft,
                        upper ? CblasUpper : CblasLower,
                        trans == 'N' ? CblasNoTrans : CblasConjTrans,
                        CblasNonUnit, n, neig,
                        &ONE, B, ldb, A, lda);

        } else if (itype == 3) {

            char trans;
            if (upper) {
                trans = 'C';
            } else {
                trans = 'N';
            }

            cblas_ctrmm(CblasColMajor, CblasLeft,
                        upper ? CblasUpper : CblasLower,
                        trans == 'C' ? CblasConjTrans : CblasNoTrans,
                        CblasNonUnit, n, neig,
                        &ONE, B, ldb, A, lda);
        }
    }

    work[0] = CMPLXF((f32)lwkopt, 0.0f);
}
