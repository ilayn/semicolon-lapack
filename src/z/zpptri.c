/**
 * @file zpptri.c
 * @brief ZPPTRI computes the inverse of a Hermitian positive definite matrix using its Cholesky factorization in packed format.
 */

#include <complex.h>
#include <cblas.h>
#include "semicolon_lapack_complex_double.h"

/**
 * ZPPTRI computes the inverse of a complex Hermitian positive definite
 * matrix A using the Cholesky factorization A = U**H*U or A = L*L**H
 * computed by ZPPTRF.
 *
 * @param[in]     uplo   = 'U': Upper triangular factor is stored in AP;
 *                        = 'L': Lower triangular factor is stored in AP.
 * @param[in]     n      The order of the matrix A. n >= 0.
 * @param[in,out] AP     On entry, the triangular factor U or L from the Cholesky
 *                       factorization A = U**H*U or A = L*L**H, packed columnwise
 *                       as a linear array.
 *                       On exit, the upper or lower triangle of the (Hermitian)
 *                       inverse of A, overwriting the input factor U or L.
 *                       Array of dimension (n*(n+1)/2).
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 *                         - > 0: if info = i, the (i,i) element of the factor U or L is
 *                           zero, and the inverse could not be computed.
 */
void zpptri(
    const char* uplo,
    const int n,
    c128* restrict AP,
    int* info)
{
    // zpptri.f lines 108-109: Parameters
    const f64 ONE = 1.0;

    // zpptri.f lines 128-138: Test the input parameters
    *info = 0;
    int upper = (uplo[0] == 'U' || uplo[0] == 'u');
    if (!upper && !(uplo[0] == 'L' || uplo[0] == 'l')) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    }
    if (*info != 0) {
        xerbla("ZPPTRI", -(*info));
        return;
    }

    // zpptri.f lines 142-143: Quick return if possible
    if (n == 0) {
        return;
    }

    // zpptri.f lines 147-149: Invert the triangular Cholesky factor U or L
    ztptri(uplo, "N", n, AP, info);
    if (*info > 0) {
        return;
    }

    if (upper) {
        // zpptri.f lines 153-165: Compute the product inv(U) * inv(U)**H
        int jj = -1;
        for (int j = 0; j < n; j++) {
            int jc = jj + 1;
            jj = jj + (j + 1);
            if (j > 0) {
                cblas_zhpr(CblasColMajor, CblasUpper, j, ONE, &AP[jc], 1, AP);
            }
            f64 ajj = creal(AP[jj]);
            cblas_zdscal(j + 1, ajj, &AP[jc], 1);
        }
    } else {
        // zpptri.f lines 167-182: Compute the product inv(L)**H * inv(L)
        int jj = 0;
        for (int j = 0; j < n; j++) {
            int jjn = jj + n - j;
            c128 dotc;
            cblas_zdotc_sub(n - j, &AP[jj], 1, &AP[jj], 1, &dotc);
            AP[jj] = creal(dotc);
            if (j < n - 1) {
                cblas_ztpmv(CblasColMajor, CblasLower, CblasConjTrans, CblasNonUnit,
                            n - j - 1, &AP[jjn], &AP[jj + 1], 1);
            }
            jj = jjn;
        }
    }
}
