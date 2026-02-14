/**
 * @file sspgv.c
 * @brief SSPGV computes all eigenvalues and optionally eigenvectors of a
 *        real generalized symmetric-definite eigenproblem (packed storage).
 */

#include "semicolon_lapack_single.h"
#include <cblas.h>

/**
 * SSPGV computes all the eigenvalues, and optionally, the eigenvectors
 * of a real generalized symmetric-definite eigenproblem, of the form
 * A*x=(lambda)*B*x, A*B*x=(lambda)*x, or B*A*x=(lambda)*x.
 * Here A and B are assumed to be symmetric, stored in packed format,
 * and B is also positive definite.
 *
 * @param[in]     itype  = 1: A*x = lambda*B*x; = 2: A*B*x = lambda*x; = 3: B*A*x = lambda*x
 * @param[in]     jobz   = 'N': eigenvalues only; = 'V': eigenvalues and eigenvectors
 * @param[in]     uplo   = 'U': upper triangles stored; = 'L': lower triangles stored
 * @param[in]     n      The order of the matrices A and B. n >= 0.
 * @param[in,out] AP     Packed symmetric matrix A. On exit, destroyed.
 * @param[in,out] BP     Packed symmetric positive definite B. On exit, Cholesky factor.
 * @param[out]    W      Eigenvalues in ascending order.
 * @param[out]    Z      Eigenvectors if jobz='V'; not referenced if jobz='N'.
 * @param[in]     ldz    Leading dimension of Z. ldz >= 1, or ldz >= n if jobz='V'.
 * @param[out]    work   Workspace array, dimension (3*n).
 * @param[out]    info
 *                         - = 0: success; < 0: illegal argument; > 0: SPPTRF/SSPEV error.
 */
void sspgv(
    const int itype,
    const char* jobz,
    const char* uplo,
    const int n,
    f32* restrict AP,
    f32* restrict BP,
    f32* restrict W,
    f32* restrict Z,
    const int ldz,
    f32* restrict work,
    int* info)
{
    int wantz, upper;
    int j, neig;

    wantz = (jobz[0] == 'V' || jobz[0] == 'v');
    upper = (uplo[0] == 'U' || uplo[0] == 'u');

    *info = 0;
    if (itype < 1 || itype > 3) {
        *info = -1;
    } else if (!wantz && !(jobz[0] == 'N' || jobz[0] == 'n')) {
        *info = -2;
    } else if (!upper && !(uplo[0] == 'L' || uplo[0] == 'l')) {
        *info = -3;
    } else if (n < 0) {
        *info = -4;
    } else if (ldz < 1 || (wantz && ldz < n)) {
        *info = -9;
    }

    if (*info != 0) {
        xerbla("SSPGV ", -(*info));
        return;
    }

    if (n == 0) {
        return;
    }

    spptrf(uplo, n, BP, info);
    if (*info != 0) {
        *info = n + *info;
        return;
    }

    sspgst(itype, uplo, n, AP, BP, info);
    sspev(jobz, uplo, n, AP, W, Z, ldz, work, info);

    if (wantz) {
        neig = n;
        if (*info > 0) {
            neig = *info - 1;
        }
        if (itype == 1 || itype == 2) {
            if (upper) {
                for (j = 0; j < neig; j++) {
                    cblas_stpsv(CblasColMajor, CblasUpper, CblasNoTrans,
                                CblasNonUnit, n, BP, &Z[j * ldz], 1);
                }
            } else {
                for (j = 0; j < neig; j++) {
                    cblas_stpsv(CblasColMajor, CblasLower, CblasTrans,
                                CblasNonUnit, n, BP, &Z[j * ldz], 1);
                }
            }
        } else if (itype == 3) {
            if (upper) {
                for (j = 0; j < neig; j++) {
                    cblas_stpmv(CblasColMajor, CblasUpper, CblasTrans,
                                CblasNonUnit, n, BP, &Z[j * ldz], 1);
                }
            } else {
                for (j = 0; j < neig; j++) {
                    cblas_stpmv(CblasColMajor, CblasLower, CblasNoTrans,
                                CblasNonUnit, n, BP, &Z[j * ldz], 1);
                }
            }
        }
    }
}
