/**
 * @file cpttrf.c
 * @brief CPTTRF computes the L*D*L**H factorization of a complex Hermitian
 *        positive definite tridiagonal matrix A.
 */

#include "internal_build_defs.h"
#include "semicolon_lapack_complex_single.h"
#include <complex.h>

/**
 * CPTTRF computes the L*D*L**H factorization of a complex Hermitian
 * positive definite tridiagonal matrix A.  The factorization may also
 * be regarded as having the form A = U**H *D*U.
 *
 * @param[in]     n     The order of the matrix A. n >= 0.
 * @param[in,out] D     Single precision array, dimension (n).
 *                      On entry, the n diagonal elements of the tridiagonal matrix A.
 *                      On exit, the n diagonal elements of the diagonal matrix D
 *                      from the L*D*L**H factorization of A.
 * @param[in,out] E     Complex*16 array, dimension (n-1).
 *                      On entry, the (n-1) subdiagonal elements of the tridiagonal
 *                      matrix A.
 *                      On exit, the (n-1) subdiagonal elements of the unit bidiagonal
 *                      factor L from the L*D*L**H factorization of A.
 *                      E can also be regarded as the superdiagonal of the unit
 *                      bidiagonal factor U from the U**H *D*U factorization of A.
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -k, the k-th argument had an illegal value
 *                         - > 0: if info = k, the leading principal minor of order k
 *                           is not positive; if k < n, the factorization could not
 *                           be completed, while if k = n, the factorization was
 *                           completed, but D(n-1) <= 0 (0-based).
 */
void cpttrf(
    const INT n,
    f32* restrict D,
    c64* restrict E,
    INT* info)
{
    const f32 ZERO = 0.0f;
    INT i, i4;
    f32 eir, eii, f, g;

    *info = 0;
    if (n < 0) {
        *info = -1;
        xerbla("CPTTRF", -(*info));
        return;
    }

    if (n == 0)
        return;

    i4 = (n - 1) % 4;

    for (i = 0; i < i4; i++) {
        if (D[i] <= ZERO) {
            *info = i + 1;
            return;
        }
        eir = crealf(E[i]);
        eii = cimagf(E[i]);
        f = eir / D[i];
        g = eii / D[i];
        E[i] = CMPLXF(f, g);
        D[i + 1] = D[i + 1] - f * eir - g * eii;
    }

    for (i = i4; i < n - 4; i += 4) {

        if (D[i] <= ZERO) {
            *info = i + 1;
            return;
        }

        eir = crealf(E[i]);
        eii = cimagf(E[i]);
        f = eir / D[i];
        g = eii / D[i];
        E[i] = CMPLXF(f, g);
        D[i + 1] = D[i + 1] - f * eir - g * eii;

        if (D[i + 1] <= ZERO) {
            *info = i + 2;
            return;
        }

        eir = crealf(E[i + 1]);
        eii = cimagf(E[i + 1]);
        f = eir / D[i + 1];
        g = eii / D[i + 1];
        E[i + 1] = CMPLXF(f, g);
        D[i + 2] = D[i + 2] - f * eir - g * eii;

        if (D[i + 2] <= ZERO) {
            *info = i + 3;
            return;
        }

        eir = crealf(E[i + 2]);
        eii = cimagf(E[i + 2]);
        f = eir / D[i + 2];
        g = eii / D[i + 2];
        E[i + 2] = CMPLXF(f, g);
        D[i + 3] = D[i + 3] - f * eir - g * eii;

        if (D[i + 3] <= ZERO) {
            *info = i + 4;
            return;
        }

        eir = crealf(E[i + 3]);
        eii = cimagf(E[i + 3]);
        f = eir / D[i + 3];
        g = eii / D[i + 3];
        E[i + 3] = CMPLXF(f, g);
        D[i + 4] = D[i + 4] - f * eir - g * eii;
    }

    if (D[n - 1] <= ZERO) {
        *info = n;
    }
}
