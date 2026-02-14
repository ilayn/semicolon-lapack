/**
 * @file zhpgvx.c
 * @brief ZHPGVX computes selected eigenvalues and, optionally, eigenvectors
 *        of a complex generalized Hermitian-definite eigenproblem.
 */

#include "semicolon_lapack_complex_double.h"
#include <complex.h>
#include <cblas.h>

/**
 * ZHPGVX computes selected eigenvalues and, optionally, eigenvectors
 * of a complex generalized Hermitian-definite eigenproblem, of the form
 * A*x=(lambda)*B*x,  A*Bx=(lambda)*x,  or B*A*x=(lambda)*x.  Here A and
 * B are assumed to be Hermitian, stored in packed format, and B is also
 * positive definite.  Eigenvalues and eigenvectors can be selected by
 * specifying either a range of values or a range of indices for the
 * desired eigenvalues.
 *
 * @param[in]     itype  Specifies the problem type to be solved:
 *                       = 1:  A*x = (lambda)*B*x
 *                       = 2:  A*B*x = (lambda)*x
 *                       = 3:  B*A*x = (lambda)*x
 * @param[in]     jobz   = 'N':  Compute eigenvalues only;
 *                        = 'V':  Compute eigenvalues and eigenvectors.
 * @param[in]     range  = 'A': all eigenvalues will be found;
 *                        = 'V': all eigenvalues in the half-open interval (VL,VU]
 *                               will be found;
 *                        = 'I': the IL-th through IU-th eigenvalues will be found.
 * @param[in]     uplo   = 'U':  Upper triangles of A and B are stored;
 *                        = 'L':  Lower triangles of A and B are stored.
 * @param[in]     n      The order of the matrices A and B.  n >= 0.
 * @param[in,out] AP     Complex array, dimension (n*(n+1)/2).
 *                       On entry, the upper or lower triangle of the Hermitian
 *                       matrix A, packed columnwise in a linear array.
 *                       On exit, the contents of AP are destroyed.
 * @param[in,out] BP     Complex array, dimension (n*(n+1)/2).
 *                       On entry, the upper or lower triangle of the Hermitian
 *                       matrix B, packed columnwise in a linear array.
 *                       On exit, the triangular factor U or L from the Cholesky
 *                       factorization B = U**H*U or B = L*L**H.
 * @param[in]     vl     If RANGE='V', the lower bound of the interval to
 *                       be searched for eigenvalues. vl < vu.
 *                       Not referenced if RANGE = 'A' or 'I'.
 * @param[in]     vu     If RANGE='V', the upper bound of the interval to
 *                       be searched for eigenvalues. vl < vu.
 *                       Not referenced if RANGE = 'A' or 'I'.
 * @param[in]     il     If RANGE='I', the index of the smallest eigenvalue
 *                       to be returned. 1 <= il <= iu <= n, if n > 0.
 *                       Not referenced if RANGE = 'A' or 'V'.
 * @param[in]     iu     If RANGE='I', the index of the largest eigenvalue
 *                       to be returned. 1 <= il <= iu <= n, if n > 0.
 *                       Not referenced if RANGE = 'A' or 'V'.
 * @param[in]     abstol The absolute error tolerance for the eigenvalues.
 * @param[out]    m      The total number of eigenvalues found. 0 <= m <= n.
 * @param[out]    W      Double precision array, dimension (n).
 *                       On normal exit, the first m elements contain the selected
 *                       eigenvalues in ascending order.
 * @param[out]    Z      Complex array, dimension (ldz, n).
 *                       If JOBZ = 'V', the first m columns of Z contain the
 *                       orthonormal eigenvectors.
 * @param[in]     ldz    The leading dimension of the array Z. ldz >= 1, and if
 *                       JOBZ = 'V', ldz >= max(1,n).
 * @param[out]    work   Complex array, dimension (2*n).
 * @param[out]    rwork  Double precision array, dimension (7*n).
 * @param[out]    iwork  Integer array, dimension (5*n).
 * @param[out]    ifail  Integer array, dimension (n).
 *                       If JOBZ = 'V', then if info = 0, the first m elements of
 *                       IFAIL are zero. If info > 0, then IFAIL contains the
 *                       indices of the eigenvectors that failed to converge.
 * @param[out]    info   = 0:  successful exit
 *                       < 0:  if info = -i, the i-th argument had an illegal value
 *                       > 0:  ZPPTRF or ZHPEVX returned an error code.
 */
void zhpgvx(
    const int itype,
    const char* jobz,
    const char* range,
    const char* uplo,
    const int n,
    double complex* const restrict AP,
    double complex* const restrict BP,
    const double vl,
    const double vu,
    const int il,
    const int iu,
    const double abstol,
    int* m,
    double* const restrict W,
    double complex* const restrict Z,
    const int ldz,
    double complex* const restrict work,
    double* const restrict rwork,
    int* const restrict iwork,
    int* const restrict ifail,
    int* info)
{
    int wantz = (jobz[0] == 'V' || jobz[0] == 'v');
    int upper = (uplo[0] == 'U' || uplo[0] == 'u');
    int alleig = (range[0] == 'A' || range[0] == 'a');
    int valeig = (range[0] == 'V' || range[0] == 'v');
    int indeig = (range[0] == 'I' || range[0] == 'i');

    *info = 0;
    if (itype < 1 || itype > 3) {
        *info = -1;
    } else if (!(wantz || (jobz[0] == 'N' || jobz[0] == 'n'))) {
        *info = -2;
    } else if (!(alleig || valeig || indeig)) {
        *info = -3;
    } else if (!(upper || (uplo[0] == 'L' || uplo[0] == 'l'))) {
        *info = -4;
    } else if (n < 0) {
        *info = -5;
    } else {
        if (valeig) {
            if (n > 0 && vu <= vl) {
                *info = -9;
            }
        } else if (indeig) {
            if (il < 1) {
                *info = -10;
            } else if (iu < (n < il ? n : il) || iu > n) {
                *info = -11;
            }
        }
    }
    if (*info == 0) {
        if (ldz < 1 || (wantz && ldz < n)) {
            *info = -16;
        }
    }

    if (*info != 0) {
        xerbla("ZHPGVX", -(*info));
        return;
    }

    if (n == 0) {
        return;
    }

    zpptrf(uplo, n, BP, info);
    if (*info != 0) {
        *info = n + *info;
        return;
    }

    zhpgst(itype, uplo, n, AP, BP, info);
    zhpevx(jobz, range, uplo, n, AP, vl, vu, il, iu, abstol,
           m, W, Z, ldz, work, rwork, iwork, ifail, info);

    if (wantz) {
        if (*info > 0) {
            *m = *info - 1;
        }
        if (itype == 1 || itype == 2) {
            char trans;
            if (upper) {
                trans = 'N';
            } else {
                trans = 'C';
            }

            CBLAS_TRANSPOSE cblas_trans = (trans == 'N') ?
                CblasNoTrans : CblasConjTrans;

            for (int j = 0; j < *m; j++) {
                cblas_ztpsv(CblasColMajor,
                            upper ? CblasUpper : CblasLower,
                            cblas_trans, CblasNonUnit,
                            n, BP, &Z[j * ldz], 1);
            }
        } else if (itype == 3) {
            char trans;
            if (upper) {
                trans = 'C';
            } else {
                trans = 'N';
            }

            CBLAS_TRANSPOSE cblas_trans = (trans == 'N') ?
                CblasNoTrans : CblasConjTrans;

            for (int j = 0; j < *m; j++) {
                cblas_ztpmv(CblasColMajor,
                            upper ? CblasUpper : CblasLower,
                            cblas_trans, CblasNonUnit,
                            n, BP, &Z[j * ldz], 1);
            }
        }
    }
}
