/**
 * @file csyequb.c
 * @brief CSYEQUB computes row and column scalings intended to equilibrate
 *        a symmetric matrix and reduce its condition number.
 */

#include <complex.h>
#include <math.h>
#include "semicolon_lapack_complex_single.h"

/**
 * CSYEQUB computes row and column scalings intended to equilibrate a
 * symmetric matrix A (with respect to the Euclidean norm) and reduce
 * its condition number. The scale factors S are computed by the BIN
 * algorithm (see references) so that the scaled matrix B with elements
 * B(i,j) = S(i)*A(i,j)*S(j) has a condition number within a factor N of
 * the smallest possible condition number over all possible diagonal
 * scalings.
 *
 * @param[in]     uplo   Specifies whether the upper or lower triangular part
 *                       of the symmetric matrix A is stored.
 *                       = 'U': Upper triangle of A is stored
 *                       = 'L': Lower triangle of A is stored
 * @param[in]     n      The order of the matrix A. n >= 0.
 * @param[in]     A      Complex*16 array, dimension (lda, n).
 *                       The N-by-N symmetric matrix whose scaling factors
 *                       are to be computed.
 * @param[in]     lda    The leading dimension of the array A. lda >= max(1,n).
 * @param[out]    S      Single precision array, dimension (n).
 *                       If info = 0, S contains the scale factors for A.
 * @param[out]    scond  If info = 0, S contains the ratio of the smallest S(i)
 *                       to the largest S(i). If scond >= 0.1 and amax is neither
 *                       too large nor too small, it is not worth scaling by S.
 * @param[out]    amax   Largest absolute value of any matrix element. If amax is
 *                       very close to overflow or very close to underflow, the
 *                       matrix should be scaled.
 * @param[out]    work   Complex*16 array, dimension (2*n).
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 *                         - > 0: if info = i, the i-th diagonal element is nonpositive.
 *                           References:
 *                           Livne, O.E. and Golub, G.H., "Scaling by Binormalization",
 *                           Numerical Algorithms, vol. 35, no. 1, pp. 97-120, January 2004.
 */
void csyequb(
    const char* uplo,
    const INT n,
    const c64* restrict A,
    const INT lda,
    f32* restrict S,
    f32* scond,
    f32* amax,
    c64* restrict work,
    INT* info)
{
    const f32 ONE = 1.0f;
    const f32 ZERO = 0.0f;
    const INT MAX_ITER = 100;

    *info = 0;
    INT up = (uplo[0] == 'U' || uplo[0] == 'u');

    if (!up && !(uplo[0] == 'L' || uplo[0] == 'l')) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -4;
    }
    if (*info != 0) {
        xerbla("CSYEQUB", -(*info));
        return;
    }

    *amax = ZERO;

    if (n == 0) {
        *scond = ONE;
        return;
    }

    for (INT i = 0; i < n; i++) {
        S[i] = ZERO;
    }

    *amax = ZERO;
    if (up) {
        for (INT j = 0; j < n; j++) {
            for (INT i = 0; i < j; i++) {
                f32 absval = cabs1f(A[i + j * lda]);
                S[i] = (S[i] > absval) ? S[i] : absval;
                S[j] = (S[j] > absval) ? S[j] : absval;
                *amax = (*amax > absval) ? *amax : absval;
            }
            f32 absdiag = cabs1f(A[j + j * lda]);
            S[j] = (S[j] > absdiag) ? S[j] : absdiag;
            *amax = (*amax > absdiag) ? *amax : absdiag;
        }
    } else {
        for (INT j = 0; j < n; j++) {
            f32 absdiag = cabs1f(A[j + j * lda]);
            S[j] = (S[j] > absdiag) ? S[j] : absdiag;
            *amax = (*amax > absdiag) ? *amax : absdiag;
            for (INT i = j + 1; i < n; i++) {
                f32 absval = cabs1f(A[i + j * lda]);
                S[i] = (S[i] > absval) ? S[i] : absval;
                S[j] = (S[j] > absval) ? S[j] : absval;
                *amax = (*amax > absval) ? *amax : absval;
            }
        }
    }
    for (INT j = 0; j < n; j++) {
        S[j] = ONE / S[j];
    }

    f32 tol_val = ONE / sqrtf(2.0f * n);
    f32 avg = ZERO;

    for (INT iter = 0; iter < MAX_ITER; iter++) {
        f32 scale = ZERO;
        f32 sumsq = ZERO;

        for (INT i = 0; i < n; i++) {
            work[i] = CMPLXF(0.0f, 0.0f);
        }
        if (up) {
            for (INT j = 0; j < n; j++) {
                for (INT i = 0; i < j; i++) {
                    f32 absval = cabs1f(A[i + j * lda]);
                    work[i] = work[i] + absval * S[j];
                    work[j] = work[j] + absval * S[i];
                }
                work[j] = work[j] + cabs1f(A[j + j * lda]) * S[j];
            }
        } else {
            for (INT j = 0; j < n; j++) {
                work[j] = work[j] + cabs1f(A[j + j * lda]) * S[j];
                for (INT i = j + 1; i < n; i++) {
                    f32 absval = cabs1f(A[i + j * lda]);
                    work[i] = work[i] + absval * S[j];
                    work[j] = work[j] + absval * S[i];
                }
            }
        }

        avg = ZERO;
        for (INT i = 0; i < n; i++) {
            avg = avg + S[i] * crealf(work[i]);
        }
        avg = avg / n;

        f32 std_dev = ZERO;
        for (INT i = 0; i < n; i++) {
            work[n + i] = S[i] * work[i] - avg;
        }
        classq(n, &work[n], 1, &scale, &sumsq);
        std_dev = scale * sqrtf(sumsq / n);

        if (std_dev < tol_val * avg) {
            break;
        }

        for (INT i = 0; i < n; i++) {
            f32 t = cabs1f(A[i + i * lda]);
            f32 si = S[i];
            f32 c2 = (n - 1) * t;
            f32 c1 = (n - 2) * (crealf(work[i]) - t * si);
            f32 c0 = -(t * si) * si + 2 * crealf(work[i]) * si - n * avg;
            f32 d = c1 * c1 - 4 * c0 * c2;

            if (d <= ZERO) {
                *info = -1;
                return;
            }
            si = -2 * c0 / (c1 + sqrtf(d));

            d = si - S[i];
            f32 u = ZERO;
            if (up) {
                for (INT j = 0; j <= i; j++) {
                    t = cabs1f(A[j + i * lda]);
                    u = u + S[j] * t;
                    work[j] = work[j] + d * t;
                }
                for (INT j = i + 1; j < n; j++) {
                    t = cabs1f(A[i + j * lda]);
                    u = u + S[j] * t;
                    work[j] = work[j] + d * t;
                }
            } else {
                for (INT j = 0; j <= i; j++) {
                    t = cabs1f(A[i + j * lda]);
                    u = u + S[j] * t;
                    work[j] = work[j] + d * t;
                }
                for (INT j = i + 1; j < n; j++) {
                    t = cabs1f(A[j + i * lda]);
                    u = u + S[j] * t;
                    work[j] = work[j] + d * t;
                }
            }

            avg = avg + (u + crealf(work[i])) * d / n;
            S[i] = si;
        }
    }

    f32 smlnum = slamch("SAFEMIN");
    f32 bignum = ONE / smlnum;
    f32 smin = bignum;
    f32 smax = ZERO;
    f32 t = ONE / sqrtf(avg > ZERO ? avg : 1.0f);
    f32 base = slamch("B");
    f32 u = ONE / logf(base);
    for (INT i = 0; i < n; i++) {
        S[i] = powf(base, (INT)(u * logf(S[i] * t)));
        smin = (smin < S[i]) ? smin : S[i];
        smax = (smax > S[i]) ? smax : S[i];
    }
    f32 smin_safe = (smin > smlnum) ? smin : smlnum;
    f32 smax_safe = (smax < bignum) ? smax : bignum;
    *scond = smin_safe / smax_safe;
}
