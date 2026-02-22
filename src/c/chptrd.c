/**
 * @file chptrd.c
 * @brief CHPTRD reduces a complex Hermitian matrix stored in packed form to
 *        real symmetric tridiagonal form.
 */

#include "internal_build_defs.h"
#include "semicolon_lapack_complex_single.h"
#include <complex.h>
#include <cblas.h>

/**
 * CHPTRD reduces a complex Hermitian matrix A stored in packed form to
 * real symmetric tridiagonal form T by a unitary similarity
 * transformation: Q**H * A * Q = T.
 *
 * @param[in]     uplo  = 'U':  Upper triangle of A is stored;
 *                        = 'L':  Lower triangle of A is stored.
 * @param[in]     n     The order of the matrix A.  N >= 0.
 * @param[in,out] AP    Complex*16 array, dimension (N*(N+1)/2).
 *                       On entry, the upper or lower triangle of the Hermitian
 *                       matrix A, packed columnwise in a linear array.  The
 *                       j-th column of A is stored in the array AP as follows:
 *                       if UPLO = 'U', AP(i + (j-1)*j/2) = A(i,j) for 0<=i<=j;
 *                       if UPLO = 'L', AP(i + (j-1)*(2*n-j)/2) = A(i,j) for j<=i<=n-1.
 *                       On exit, if UPLO = 'U', the diagonal and first
 *                       superdiagonal of A are overwritten by the corresponding
 *                       elements of the tridiagonal matrix T, and the elements
 *                       above the first superdiagonal, with the array TAU,
 *                       represent the unitary matrix Q as a product of
 *                       elementary reflectors; if UPLO = 'L', the diagonal and
 *                       first subdiagonal of A are overwritten by the
 *                       corresponding elements of the tridiagonal matrix T, and
 *                       the elements below the first subdiagonal, with the
 *                       array TAU, represent the unitary matrix Q as a product
 *                       of elementary reflectors. See Further Details.
 * @param[out]    d     Single precision array, dimension (N).
 *                       The diagonal elements of the tridiagonal matrix T:
 *                       D(i) = A(i,i).
 * @param[out]    e     Single precision array, dimension (N-1).
 *                       The off-diagonal elements of the tridiagonal matrix T:
 *                       E(i) = A(i,i+1) if UPLO = 'U', E(i) = A(i+1,i) if
 *                       UPLO = 'L'.
 * @param[out]    tau   Complex*16 array, dimension (N-1).
 *                       The scalar factors of the elementary reflectors (see
 *                       Further Details).
 * @param[out]    info  = 0:  successful exit
 *                       < 0:  if info = -i, the i-th argument had an illegal
 *                             value.
 *
 *  If UPLO = 'U', the matrix Q is represented as a product of elementary
 *  reflectors
 *
 *     Q = H(n-1) . . . H(2) H(1).
 *
 *  Each H(i) has the form
 *
 *     H(i) = I - tau * v * v**H
 *
 *  where tau is a complex scalar, and v is a complex vector with
 *  v(i+1:n) = 0 and v(i) = 1; v(1:i-1) is stored on exit in AP,
 *  overwriting A(1:i-1,i+1), and tau is stored in TAU(i).
 *
 *  If UPLO = 'L', the matrix Q is represented as a product of elementary
 *  reflectors
 *
 *     Q = H(1) H(2) . . . H(n-1).
 *
 *  Each H(i) has the form
 *
 *     H(i) = I - tau * v * v**H
 *
 *  where tau is a complex scalar, and v is a complex vector with
 *  v(1:i) = 0 and v(i+1) = 1; v(i+2:n) is stored on exit in AP,
 *  overwriting A(i+2:n,i), and tau is stored in TAU(i).
 */
void chptrd(const char* uplo, const INT n, c64* AP,
            f32* d, f32* e, c64* tau, INT* info)
{
    const c64 ONE = CMPLXF(1.0f, 0.0f);
    const c64 ZERO = CMPLXF(0.0f, 0.0f);
    const c64 HALF = CMPLXF(0.5f, 0.0f);

    INT upper;
    INT i, i1, i1i1, ii;
    c64 alpha, taui;

    *info = 0;
    upper = (uplo[0] == 'U' || uplo[0] == 'u');
    if (!upper && !(uplo[0] == 'L' || uplo[0] == 'l')) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    }
    if (*info != 0) {
        xerbla("CHPTRD", -(*info));
        return;
    }

    if (n <= 0)
        return;

    if (upper) {

        /* Reduce the upper triangle of A. */
        /* i1 is the 0-based index in AP of A(0, i+1). */

        i1 = n * (n - 1) / 2;
        AP[i1 + n - 1] = CMPLXF(crealf(AP[i1 + n - 1]), 0.0f);
        for (i = n - 2; i >= 0; i--) {

            /* Generate elementary reflector H(i) = I - tau * v * v**H */
            /* to annihilate A(0:i-1, i+1) */

            alpha = AP[i1 + i];
            clarfg(i + 1, &alpha, &AP[i1], 1, &taui);
            e[i] = crealf(alpha);

            if (crealf(taui) != 0.0f || cimagf(taui) != 0.0f) {

                /* Apply H(i) from both sides to A(0:i, 0:i) */

                AP[i1 + i] = ONE;

                /* Compute  y := tau * A * v  storing y in TAU(0:i) */

                cblas_chpmv(CblasColMajor, CblasUpper, i + 1, &taui, AP,
                            &AP[i1], 1, &ZERO, tau, 1);

                /* Compute  w := y - 1/2 * tau * (y**H * v) * v */

                c64 dot;
                cblas_cdotc_sub(i + 1, tau, 1, &AP[i1], 1, &dot);
                alpha = -HALF * taui * dot;
                cblas_caxpy(i + 1, &alpha, &AP[i1], 1, tau, 1);

                /* Apply the transformation as a rank-2 update: */
                /*    A := A - v * w**H - w * v**H */

                c64 neg_one = -ONE;
                cblas_chpr2(CblasColMajor, CblasUpper, i + 1, &neg_one,
                            &AP[i1], 1, tau, 1, AP);

            }
            AP[i1 + i] = e[i];
            d[i + 1] = crealf(AP[i1 + i + 1]);
            tau[i] = taui;
            i1 = i1 - (i + 1);
        }
        d[0] = crealf(AP[0]);
    } else {

        /* Reduce the lower triangle of A. ii is the 0-based index in AP of */
        /* A(i,i) and i1i1 is the index of A(i+1,i+1). */

        ii = 0;
        AP[0] = CMPLXF(crealf(AP[0]), 0.0f);
        for (i = 0; i < n - 1; i++) {
            i1i1 = ii + n - i;

            /* Generate elementary reflector H(i) = I - tau * v * v**H */
            /* to annihilate A(i+2:n-1, i) */

            alpha = AP[ii + 1];
            clarfg(n - i - 1, &alpha, &AP[ii + 2], 1, &taui);
            e[i] = crealf(alpha);

            if (crealf(taui) != 0.0f || cimagf(taui) != 0.0f) {

                /* Apply H(i) from both sides to A(i+1:n-1, i+1:n-1) */

                AP[ii + 1] = ONE;

                /* Compute  y := tau * A * v  storing y in TAU(i:n-2) */

                cblas_chpmv(CblasColMajor, CblasLower, n - i - 1, &taui,
                            &AP[i1i1], &AP[ii + 1], 1, &ZERO, &tau[i], 1);

                /* Compute  w := y - 1/2 * tau * (y**H * v) * v */

                c64 dot;
                cblas_cdotc_sub(n - i - 1, &tau[i], 1, &AP[ii + 1], 1, &dot);
                alpha = -HALF * taui * dot;
                cblas_caxpy(n - i - 1, &alpha, &AP[ii + 1], 1, &tau[i], 1);

                /* Apply the transformation as a rank-2 update: */
                /*    A := A - v * w**H - w * v**H */

                c64 neg_one = -ONE;
                cblas_chpr2(CblasColMajor, CblasLower, n - i - 1, &neg_one,
                            &AP[ii + 1], 1, &tau[i], 1, &AP[i1i1]);

            }
            AP[ii + 1] = e[i];
            d[i] = crealf(AP[ii]);
            tau[i] = taui;
            ii = i1i1;
        }
        d[n - 1] = crealf(AP[ii]);
    }
}
