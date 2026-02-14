/**
 * @file spptri.c
 * @brief SPPTRI computes the inverse of a symmetric positive definite matrix using its Cholesky factorization in packed format.
 */

#include <cblas.h>
#include "semicolon_lapack_single.h"

/**
 * SPPTRI computes the inverse of a real symmetric positive definite
 * matrix A using the Cholesky factorization A = U**T*U or A = L*L**T
 * computed by SPPTRF.
 *
 * @param[in]     uplo   = 'U': Upper triangular factor is stored in AP;
 *                        = 'L': Lower triangular factor is stored in AP.
 * @param[in]     n      The order of the matrix A. n >= 0.
 * @param[in,out] AP     On entry, the triangular factor U or L from the Cholesky
 *                       factorization A = U**T*U or A = L*L**T, packed columnwise
 *                       as a linear array.
 *                       On exit, the upper or lower triangle of the (symmetric)
 *                       inverse of A, overwriting the input factor U or L.
 *                       Array of dimension (n*(n+1)/2).
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 *                         - > 0: if info = i, the (i,i) element of the factor U or L is
 *                           zero, and the inverse could not be computed.
 */
void spptri(
    const char* uplo,
    const int n,
    f32* restrict AP,
    int* info)
{
    // spptri.f lines 108-109: Parameters
    const f32 ONE = 1.0f;

    // spptri.f lines 128-138: Test the input parameters
    *info = 0;
    int upper = (uplo[0] == 'U' || uplo[0] == 'u');
    if (!upper && !(uplo[0] == 'L' || uplo[0] == 'l')) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    }
    if (*info != 0) {
        xerbla("SPPTRI", -(*info));
        return;
    }

    // spptri.f lines 142-143: Quick return if possible
    if (n == 0) {
        return;
    }

    // spptri.f lines 147-149: Invert the triangular Cholesky factor U or L
    stptri(uplo, "N", n, AP, info);
    if (*info > 0) {
        return;
    }

    if (upper) {
        // spptri.f lines 151-163: Compute the product inv(U) * inv(U)**T
        int jj = -1;  // spptri.f line 155: JJ = 0 (0-based: -1 so first jj+j gives proper position)
        for (int j = 0; j < n; j++) {  // spptri.f line 156: DO 10 J = 1, N
            int jc = jj + 1;  // spptri.f line 157: JC = JJ + 1
            jj = jj + (j + 1);  // spptri.f line 158: JJ = JJ + J
            // spptri.f lines 159-160: IF( J.GT.1 ) CALL DSPR( 'Upper', J-1, ONE, AP( JC ), 1, AP )
            if (j > 0) {
                cblas_sspr(CblasColMajor, CblasUpper, j, ONE, &AP[jc], 1, AP);
            }
            // spptri.f lines 161-162
            f32 ajj = AP[jj];
            cblas_sscal(j + 1, ajj, &AP[jc], 1);
        }
    } else {
        // spptri.f lines 165-178: Compute the product inv(L)**T * inv(L)
        int jj = 0;  // spptri.f line 169: JJ = 1 (0-based: 0)
        for (int j = 0; j < n; j++) {  // spptri.f line 170: DO 20 J = 1, N
            // spptri.f line 171: JJN = JJ + N - J + 1
            // In 1-based with J=1..N: JJN = JJ + N - J + 1
            // In 0-based with j=0..n-1: jjn = jj + n - j
            int jjn = jj + n - j;
            // spptri.f line 172: AP( JJ ) = DDOT( N-J+1, AP( JJ ), 1, AP( JJ ), 1 )
            // In 0-based: length is n - j
            AP[jj] = cblas_sdot(n - j, &AP[jj], 1, &AP[jj], 1);
            // spptri.f lines 173-175
            if (j < n - 1) {
                cblas_stpmv(CblasColMajor, CblasLower, CblasTrans, CblasNonUnit,
                            n - j - 1, &AP[jjn], &AP[jj + 1], 1);
            }
            jj = jjn;  // spptri.f line 176: JJ = JJN
        }
    }
}
