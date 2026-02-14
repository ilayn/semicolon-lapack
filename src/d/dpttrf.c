/**
 * @file dpttrf.c
 * @brief DPTTRF computes the L*D*L**T factorization of a real symmetric
 *        positive definite tridiagonal matrix A.
 */

#include "semicolon_lapack_double.h"

/**
 * DPTTRF computes the L*D*L**T factorization of a real symmetric
 * positive definite tridiagonal matrix A.  The factorization may also
 * be regarded as having the form A = U**T*D*U.
 *
 * @param[in]     n     The order of the matrix A. n >= 0.
 * @param[in,out] D     Double precision array, dimension (n).
 *                      On entry, the n diagonal elements of the tridiagonal matrix A.
 *                      On exit, the n diagonal elements of the diagonal matrix D
 *                      from the L*D*L**T factorization of A.
 * @param[in,out] E     Double precision array, dimension (n-1).
 *                      On entry, the (n-1) subdiagonal elements of the tridiagonal
 *                      matrix A.
 *                      On exit, the (n-1) subdiagonal elements of the unit bidiagonal
 *                      factor L from the L*D*L**T factorization of A.
 *                      E can also be regarded as the superdiagonal of the unit
 *                      bidiagonal factor U from the U**T*D*U factorization of A.
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -k, the k-th argument had an illegal value
 *                         - > 0: if info = k, the leading principal minor of order k
 *                           is not positive; if k < n, the factorization could not
 *                           be completed, while if k = n, the factorization was
 *                           completed, but D(n-1) <= 0 (0-based).
 */
void dpttrf(
    const int n,
    f64* restrict D,
    f64* restrict E,
    int* info)
{
    const f64 ZERO = 0.0;
    int i, i4;
    f64 ei;

    *info = 0;
    if (n < 0) {
        *info = -1;
        xerbla("DPTTRF", -(*info));
        return;
    }

    // Quick return if possible
    if (n == 0)
        return;

    // Compute the L*D*L**T (or U**T*D*U) factorization of A.
    // Unrolled loop: handle remainder first (Fortran: I4 = MOD(N-1, 4))
    i4 = (n - 1) % 4;

    for (i = 0; i < i4; i++) {
        if (D[i] <= ZERO) {
            *info = i + 1;  // 1-based info
            return;
        }
        ei = E[i];
        E[i] = ei / D[i];
        D[i + 1] = D[i + 1] - E[i] * ei;
    }

    // Main loop, unrolled by 4
    for (i = i4; i < n - 4; i += 4) {
        // Drop out of the loop if d(i) <= 0: the matrix is not positive definite.
        if (D[i] <= ZERO) {
            *info = i + 1;
            return;
        }

        // Solve for e(i) and d(i+1).
        ei = E[i];
        E[i] = ei / D[i];
        D[i + 1] = D[i + 1] - E[i] * ei;

        if (D[i + 1] <= ZERO) {
            *info = i + 2;
            return;
        }

        // Solve for e(i+1) and d(i+2).
        ei = E[i + 1];
        E[i + 1] = ei / D[i + 1];
        D[i + 2] = D[i + 2] - E[i + 1] * ei;

        if (D[i + 2] <= ZERO) {
            *info = i + 3;
            return;
        }

        // Solve for e(i+2) and d(i+3).
        ei = E[i + 2];
        E[i + 2] = ei / D[i + 2];
        D[i + 3] = D[i + 3] - E[i + 2] * ei;

        if (D[i + 3] <= ZERO) {
            *info = i + 4;
            return;
        }

        // Solve for e(i+3) and d(i+4).
        ei = E[i + 3];
        E[i + 3] = ei / D[i + 3];
        D[i + 4] = D[i + 4] - E[i + 3] * ei;
    }

    // Check d(n-1) for positive definiteness (0-based: D[n-1])
    if (D[n - 1] <= ZERO) {
        *info = n;
    }
}
