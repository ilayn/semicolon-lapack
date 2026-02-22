/**
 * @file dsptrd.c
 * @brief DSPTRD reduces a real symmetric matrix stored in packed form to
 *        symmetric tridiagonal form by an orthogonal similarity transformation.
 */

#include "internal_build_defs.h"
#include <cblas.h>
#include "semicolon_lapack_double.h"

/**
 * DSPTRD reduces a real symmetric matrix A stored in packed form to
 * symmetric tridiagonal form T by an orthogonal similarity
 * transformation: Q**T * A * Q = T.
 *
 * @param[in]     uplo   Specifies whether the upper or lower triangular part
 *                       of the symmetric matrix A is stored:
 *                       = 'U': Upper triangle of A is stored;
 *                       = 'L': Lower triangle of A is stored.
 * @param[in]     n      The order of the matrix A. n >= 0.
 * @param[in,out] AP     Double precision array, dimension (n*(n+1)/2).
 *                       On entry, the upper or lower triangle of the symmetric
 *                       matrix A, packed columnwise in a linear array.
 *                       On exit, if uplo = 'U', the diagonal and first
 *                       superdiagonal of A are overwritten by the corresponding
 *                       elements of the tridiagonal matrix T, and the elements
 *                       above the first superdiagonal, with the array tau,
 *                       represent the orthogonal matrix Q as a product of
 *                       elementary reflectors; if uplo = 'L', the diagonal and
 *                       first subdiagonal of A are overwritten by the
 *                       corresponding elements of the tridiagonal matrix T,
 *                       and the elements below the first subdiagonal, with the
 *                       array tau, represent the orthogonal matrix Q as a
 *                       product of elementary reflectors.
 * @param[out]    D      Double precision array, dimension (n).
 *                       The diagonal elements of the tridiagonal matrix T.
 * @param[out]    E      Double precision array, dimension (n-1).
 *                       The off-diagonal elements of the tridiagonal matrix T.
 * @param[out]    tau    Double precision array, dimension (n-1).
 *                       The scalar factors of the elementary reflectors.
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value.
 */
void dsptrd(const char* uplo, const INT n, f64* restrict AP,
            f64* restrict D, f64* restrict E,
            f64* restrict tau, INT* info)
{
    const f64 ONE = 1.0;
    const f64 ZERO = 0.0;
    const f64 HALF = 0.5;

    INT upper;
    INT i, i1, i1i1, ii;
    f64 alpha, taui;

    *info = 0;
    upper = (uplo[0] == 'U' || uplo[0] == 'u');
    if (!upper && !(uplo[0] == 'L' || uplo[0] == 'l')) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    }
    if (*info != 0) {
        xerbla("DSPTRD", -(*info));
        return;
    }

    if (n <= 0) {
        return;
    }

    if (upper) {
        /* Reduce the upper triangle of A.
           i1 is the index in AP of A(0,i+1) in 0-based indexing. */

        i1 = n * (n - 1) / 2;
        for (i = n - 2; i >= 0; i--) {

            /* Generate elementary reflector H(i) = I - tau * v * v**T
               to annihilate A(0:i-1,i+1) */

            dlarfg(i + 1, &AP[i1 + i], &AP[i1], 1, &taui);
            E[i] = AP[i1 + i];

            if (taui != ZERO) {

                /* Apply H(i) from both sides to A(0:i,0:i) */

                AP[i1 + i] = ONE;

                /* Compute  y := tau * A * v  storing y in tau(0:i) */

                cblas_dspmv(CblasColMajor, CblasUpper, i + 1, taui, AP,
                            &AP[i1], 1, ZERO, tau, 1);

                /* Compute  w := y - 1/2 * tau * (y**T *v) * v */

                alpha = -HALF * taui * cblas_ddot(i + 1, tau, 1, &AP[i1], 1);
                cblas_daxpy(i + 1, alpha, &AP[i1], 1, tau, 1);

                /* Apply the transformation as a rank-2 update:
                      A := A - v * w**T - w * v**T */

                cblas_dspr2(CblasColMajor, CblasUpper, i + 1, -ONE,
                            &AP[i1], 1, tau, 1, AP);

                AP[i1 + i] = E[i];
            }
            D[i + 1] = AP[i1 + i + 1];
            tau[i] = taui;
            i1 = i1 - (i + 1);
        }
        D[0] = AP[0];
    } else {
        /* Reduce the lower triangle of A. ii is the index in AP of
           A(i,i) and i1i1 is the index of A(i+1,i+1). */

        ii = 0;
        for (i = 0; i < n - 1; i++) {
            i1i1 = ii + n - i;

            /* Generate elementary reflector H(i) = I - tau * v * v**T
               to annihilate A(i+2:n-1,i) */

            dlarfg(n - i - 1, &AP[ii + 1], &AP[ii + 2], 1, &taui);
            E[i] = AP[ii + 1];

            if (taui != ZERO) {

                /* Apply H(i) from both sides to A(i+1:n-1,i+1:n-1) */

                AP[ii + 1] = ONE;

                /* Compute  y := tau * A * v  storing y in tau(i:n-2) */

                cblas_dspmv(CblasColMajor, CblasLower, n - i - 1, taui,
                            &AP[i1i1], &AP[ii + 1], 1, ZERO, &tau[i], 1);

                /* Compute  w := y - 1/2 * tau * (y**T *v) * v */

                alpha = -HALF * taui * cblas_ddot(n - i - 1, &tau[i], 1,
                                                   &AP[ii + 1], 1);
                cblas_daxpy(n - i - 1, alpha, &AP[ii + 1], 1, &tau[i], 1);

                /* Apply the transformation as a rank-2 update:
                      A := A - v * w**T - w * v**T */

                cblas_dspr2(CblasColMajor, CblasLower, n - i - 1, -ONE,
                            &AP[ii + 1], 1, &tau[i], 1, &AP[i1i1]);

                AP[ii + 1] = E[i];
            }
            D[i] = AP[ii];
            tau[i] = taui;
            ii = i1i1;
        }
        D[n - 1] = AP[ii];
    }
}
