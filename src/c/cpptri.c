/**
 * @file cpptri.c
 * @brief CPPTRI computes the inverse of a Hermitian positive definite matrix using its Cholesky factorization in packed format.
 */

#include "internal_build_defs.h"
#include <complex.h>
#include <cblas.h>
#include "semicolon_lapack_complex_single.h"

/**
 * CPPTRI computes the inverse of a complex Hermitian positive definite
 * matrix A using the Cholesky factorization A = U**H*U or A = L*L**H
 * computed by CPPTRF.
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
void cpptri(
    const char* uplo,
    const INT n,
    c64* restrict AP,
    INT* info)
{
    // cpptri.f lines 108-109: Parameters
    const f32 ONE = 1.0f;

    // cpptri.f lines 128-138: Test the input parameters
    *info = 0;
    INT upper = (uplo[0] == 'U' || uplo[0] == 'u');
    if (!upper && !(uplo[0] == 'L' || uplo[0] == 'l')) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    }
    if (*info != 0) {
        xerbla("CPPTRI", -(*info));
        return;
    }

    // cpptri.f lines 142-143: Quick return if possible
    if (n == 0) {
        return;
    }

    // cpptri.f lines 147-149: Invert the triangular Cholesky factor U or L
    ctptri(uplo, "N", n, AP, info);
    if (*info > 0) {
        return;
    }

    if (upper) {
        // cpptri.f lines 153-165: Compute the product inv(U) * inv(U)**H
        INT jj = -1;
        for (INT j = 0; j < n; j++) {
            INT jc = jj + 1;
            jj = jj + (j + 1);
            if (j > 0) {
                cblas_chpr(CblasColMajor, CblasUpper, j, ONE, &AP[jc], 1, AP);
            }
            f32 ajj = crealf(AP[jj]);
            cblas_csscal(j + 1, ajj, &AP[jc], 1);
        }
    } else {
        // cpptri.f lines 167-182: Compute the product inv(L)**H * inv(L)
        INT jj = 0;
        for (INT j = 0; j < n; j++) {
            INT jjn = jj + n - j;
            c64 dotc;
            cblas_cdotc_sub(n - j, &AP[jj], 1, &AP[jj], 1, &dotc);
            AP[jj] = crealf(dotc);
            if (j < n - 1) {
                cblas_ctpmv(CblasColMajor, CblasLower, CblasConjTrans, CblasNonUnit,
                            n - j - 1, &AP[jjn], &AP[jj + 1], 1);
            }
            jj = jjn;
        }
    }
}
