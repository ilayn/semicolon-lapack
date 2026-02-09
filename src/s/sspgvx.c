/**
 * @file sspgvx.c
 * @brief SSPGVX computes selected eigenvalues and optionally eigenvectors of a
 *        real generalized symmetric-definite eigenproblem (packed storage).
 */

#include "semicolon_lapack_single.h"
#include <cblas.h>

/**
 * SSPGVX computes selected eigenvalues, and optionally, eigenvectors
 * of a real generalized symmetric-definite eigenproblem, of the form
 * A*x=(lambda)*B*x, A*B*x=(lambda)*x, or B*A*x=(lambda)*x. Here A
 * and B are assumed to be symmetric, stored in packed storage, and B
 * is also positive definite. Eigenvalues and eigenvectors can be
 * selected by specifying either a range of values or a range of indices
 * for the desired eigenvalues.
 *
 * @param[in]     itype   = 1: A*x = lambda*B*x; = 2: A*B*x = lambda*x; = 3: B*A*x = lambda*x
 * @param[in]     jobz    = 'N': eigenvalues only; = 'V': eigenvalues and eigenvectors
 * @param[in]     range   = 'A': all eigenvalues; = 'V': eigenvalues in (vl,vu]; = 'I': il-th through iu-th
 * @param[in]     uplo    = 'U': upper triangles stored; = 'L': lower triangles stored
 * @param[in]     n       The order of the matrices A and B. n >= 0.
 * @param[in,out] AP      Packed symmetric matrix A. On exit, destroyed.
 * @param[in,out] BP      Packed symmetric positive definite B. On exit, Cholesky factor.
 * @param[in]     vl      Lower bound if range='V'.
 * @param[in]     vu      Upper bound if range='V'. vl < vu.
 * @param[in]     il      Smallest eigenvalue index if range='I'. 1 <= il <= iu.
 * @param[in]     iu      Largest eigenvalue index if range='I'. il <= iu <= n.
 * @param[in]     abstol  Absolute error tolerance for eigenvalues.
 * @param[out]    m       Number of eigenvalues found.
 * @param[out]    W       The first m elements contain selected eigenvalues in ascending order.
 * @param[out]    Z       Eigenvectors if jobz='V'; not referenced if jobz='N'.
 * @param[in]     ldz     Leading dimension of Z. ldz >= 1, or ldz >= n if jobz='V'.
 * @param[out]    work    Workspace array, dimension (8*n).
 * @param[out]    iwork   Integer workspace, dimension (5*n).
 * @param[out]    ifail   Indices of eigenvectors that failed to converge.
 * @param[out]    info    = 0: success; < 0: illegal argument; > 0: SPPTRF/SSPEVX error.
 */
void sspgvx(
    const int itype,
    const char* jobz,
    const char* range,
    const char* uplo,
    const int n,
    float* restrict AP,
    float* restrict BP,
    const float vl,
    const float vu,
    const int il,
    const int iu,
    const float abstol,
    int* m,
    float* restrict W,
    float* restrict Z,
    const int ldz,
    float* restrict work,
    int* restrict iwork,
    int* restrict ifail,
    int* info)
{
    int upper, wantz, alleig, valeig, indeig;
    int j;

    upper = (uplo[0] == 'U' || uplo[0] == 'u');
    wantz = (jobz[0] == 'V' || jobz[0] == 'v');
    alleig = (range[0] == 'A' || range[0] == 'a');
    valeig = (range[0] == 'V' || range[0] == 'v');
    indeig = (range[0] == 'I' || range[0] == 'i');

    *info = 0;
    if (itype < 1 || itype > 3) {
        *info = -1;
    } else if (!wantz && !(jobz[0] == 'N' || jobz[0] == 'n')) {
        *info = -2;
    } else if (!alleig && !valeig && !indeig) {
        *info = -3;
    } else if (!upper && !(uplo[0] == 'L' || uplo[0] == 'l')) {
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
            } else if (iu < ((n < il) ? n : il) || iu > n) {
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
        xerbla("SSPGVX", -(*info));
        return;
    }

    *m = 0;
    if (n == 0) {
        return;
    }

    spptrf(uplo, n, BP, info);
    if (*info != 0) {
        *info = n + *info;
        return;
    }

    sspgst(itype, uplo, n, AP, BP, info);
    sspevx(jobz, range, uplo, n, AP, vl, vu, il, iu, abstol, m, W, Z, ldz,
           work, iwork, ifail, info);

    if (wantz) {
        if (*info > 0) {
            *m = *info - 1;
        }
        if (itype == 1 || itype == 2) {
            if (upper) {
                for (j = 0; j < *m; j++) {
                    cblas_stpsv(CblasColMajor, CblasUpper, CblasNoTrans,
                                CblasNonUnit, n, BP, &Z[j * ldz], 1);
                }
            } else {
                for (j = 0; j < *m; j++) {
                    cblas_stpsv(CblasColMajor, CblasLower, CblasTrans,
                                CblasNonUnit, n, BP, &Z[j * ldz], 1);
                }
            }
        } else if (itype == 3) {
            if (upper) {
                for (j = 0; j < *m; j++) {
                    cblas_stpmv(CblasColMajor, CblasUpper, CblasTrans,
                                CblasNonUnit, n, BP, &Z[j * ldz], 1);
                }
            } else {
                for (j = 0; j < *m; j++) {
                    cblas_stpmv(CblasColMajor, CblasLower, CblasNoTrans,
                                CblasNonUnit, n, BP, &Z[j * ldz], 1);
                }
            }
        }
    }
}
