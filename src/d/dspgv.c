/**
 * @file dspgv.c
 * @brief DSPGV computes all eigenvalues and optionally eigenvectors of a
 *        real generalized symmetric-definite eigenproblem (packed storage).
 */

#include "internal_build_defs.h"
#include "semicolon_lapack_double.h"
#include <cblas.h>

/**
 * DSPGV computes all the eigenvalues, and optionally, the eigenvectors
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
 *                         - = 0: success; < 0: illegal argument; > 0: DPPTRF/DSPEV error.
 */
void dspgv(
    const INT itype,
    const char* jobz,
    const char* uplo,
    const INT n,
    f64* restrict AP,
    f64* restrict BP,
    f64* restrict W,
    f64* restrict Z,
    const INT ldz,
    f64* restrict work,
    INT* info)
{
    INT wantz, upper;
    INT j, neig;

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
        xerbla("DSPGV ", -(*info));
        return;
    }

    if (n == 0) {
        return;
    }

    dpptrf(uplo, n, BP, info);
    if (*info != 0) {
        *info = n + *info;
        return;
    }

    dspgst(itype, uplo, n, AP, BP, info);
    dspev(jobz, uplo, n, AP, W, Z, ldz, work, info);

    if (wantz) {
        neig = n;
        if (*info > 0) {
            neig = *info - 1;
        }
        if (itype == 1 || itype == 2) {
            if (upper) {
                for (j = 0; j < neig; j++) {
                    cblas_dtpsv(CblasColMajor, CblasUpper, CblasNoTrans,
                                CblasNonUnit, n, BP, &Z[j * ldz], 1);
                }
            } else {
                for (j = 0; j < neig; j++) {
                    cblas_dtpsv(CblasColMajor, CblasLower, CblasTrans,
                                CblasNonUnit, n, BP, &Z[j * ldz], 1);
                }
            }
        } else if (itype == 3) {
            if (upper) {
                for (j = 0; j < neig; j++) {
                    cblas_dtpmv(CblasColMajor, CblasUpper, CblasTrans,
                                CblasNonUnit, n, BP, &Z[j * ldz], 1);
                }
            } else {
                for (j = 0; j < neig; j++) {
                    cblas_dtpmv(CblasColMajor, CblasLower, CblasNoTrans,
                                CblasNonUnit, n, BP, &Z[j * ldz], 1);
                }
            }
        }
    }
}
