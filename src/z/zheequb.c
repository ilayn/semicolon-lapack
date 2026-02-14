/**
 * @file zheequb.c
 * @brief ZHEEQUB computes row and column scalings intended to equilibrate
 *        a Hermitian matrix and reduce its condition number.
 */

#include <complex.h>
#include <math.h>
#include "semicolon_lapack_complex_double.h"

/**
 * ZHEEQUB computes row and column scalings intended to equilibrate a
 * Hermitian matrix A (with respect to the Euclidean norm) and reduce
 * its condition number. The scale factors S are computed by the BIN
 * algorithm (see references) so that the scaled matrix B with elements
 * B(i,j) = S(i)*A(i,j)*S(j) has a condition number within a factor N of
 * the smallest possible condition number over all possible diagonal
 * scalings.
 *
 * @param[in]     uplo   Specifies whether the upper or lower triangular part
 *                       of the Hermitian matrix A is stored.
 *                       = 'U': Upper triangle of A is stored
 *                       = 'L': Lower triangle of A is stored
 * @param[in]     n      The order of the matrix A. n >= 0.
 * @param[in]     A      Complex*16 array, dimension (lda, n).
 *                       The N-by-N Hermitian matrix whose scaling factors
 *                       are to be computed.
 * @param[in]     lda    The leading dimension of the array A. lda >= max(1,n).
 * @param[out]    S      Double precision array, dimension (n).
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
 */
void zheequb(
    const char* uplo,
    const int n,
    const c128* const restrict A,
    const int lda,
    f64* const restrict S,
    f64* scond,
    f64* amax,
    c128* const restrict work,
    int* info)
{
    const f64 ONE = 1.0;
    const f64 ZERO = 0.0;
    const int MAX_ITER = 100;

    *info = 0;
    int up = (uplo[0] == 'U' || uplo[0] == 'u');

    if (!up && !(uplo[0] == 'L' || uplo[0] == 'l')) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -4;
    }
    if (*info != 0) {
        xerbla("ZHEEQUB", -(*info));
        return;
    }

    *amax = ZERO;

    if (n == 0) {
        *scond = ONE;
        return;
    }

    for (int i = 0; i < n; i++) {
        S[i] = ZERO;
    }

    *amax = ZERO;
    if (up) {
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < j; i++) {
                f64 absval = cabs1(A[i + j * lda]);
                S[i] = (S[i] > absval) ? S[i] : absval;
                S[j] = (S[j] > absval) ? S[j] : absval;
                *amax = (*amax > absval) ? *amax : absval;
            }
            f64 absdiag = cabs1(A[j + j * lda]);
            S[j] = (S[j] > absdiag) ? S[j] : absdiag;
            *amax = (*amax > absdiag) ? *amax : absdiag;
        }
    } else {
        for (int j = 0; j < n; j++) {
            f64 absdiag = cabs1(A[j + j * lda]);
            S[j] = (S[j] > absdiag) ? S[j] : absdiag;
            *amax = (*amax > absdiag) ? *amax : absdiag;
            for (int i = j + 1; i < n; i++) {
                f64 absval = cabs1(A[i + j * lda]);
                S[i] = (S[i] > absval) ? S[i] : absval;
                S[j] = (S[j] > absval) ? S[j] : absval;
                *amax = (*amax > absval) ? *amax : absval;
            }
        }
    }
    for (int j = 0; j < n; j++) {
        S[j] = ONE / S[j];
    }

    f64 tol_val = ONE / sqrt(2.0 * n);
    f64 avg = ZERO;

    for (int iter = 0; iter < MAX_ITER; iter++) {
        f64 scale = ZERO;
        f64 sumsq = ZERO;

        for (int i = 0; i < n; i++) {
            work[i] = CMPLX(ZERO, 0.0);
        }
        if (up) {
            for (int j = 0; j < n; j++) {
                for (int i = 0; i < j; i++) {
                    f64 absval = cabs1(A[i + j * lda]);
                    work[i] = work[i] + absval * S[j];
                    work[j] = work[j] + absval * S[i];
                }
                work[j] = work[j] + cabs1(A[j + j * lda]) * S[j];
            }
        } else {
            for (int j = 0; j < n; j++) {
                work[j] = work[j] + cabs1(A[j + j * lda]) * S[j];
                for (int i = j + 1; i < n; i++) {
                    f64 absval = cabs1(A[i + j * lda]);
                    work[i] = work[i] + absval * S[j];
                    work[j] = work[j] + absval * S[i];
                }
            }
        }

        avg = ZERO;
        for (int i = 0; i < n; i++) {
            avg = avg + S[i] * creal(work[i]);
        }
        avg = avg / n;

        f64 std_dev = ZERO;
        for (int i = 0; i < n; i++) {
            work[n + i] = CMPLX(S[i] * creal(work[i]) - avg, 0.0);
        }
        zlassq(n, &work[n], 1, &scale, &sumsq);
        std_dev = scale * sqrt(sumsq / n);

        if (std_dev < tol_val * avg) {
            break;
        }

        for (int i = 0; i < n; i++) {
            f64 t = cabs1(A[i + i * lda]);
            f64 si = S[i];
            f64 c2 = (n - 1) * t;
            f64 c1 = (n - 2) * (creal(work[i]) - t * si);
            f64 c0 = -(t * si) * si + 2 * creal(work[i]) * si - n * avg;
            f64 d = c1 * c1 - 4 * c0 * c2;

            if (d <= ZERO) {
                *info = -1;
                return;
            }
            si = -2 * c0 / (c1 + sqrt(d));

            d = si - S[i];
            f64 u = ZERO;
            if (up) {
                for (int j = 0; j <= i; j++) {
                    t = cabs1(A[j + i * lda]);
                    u = u + S[j] * t;
                    work[j] = work[j] + d * t;
                }
                for (int j = i + 1; j < n; j++) {
                    t = cabs1(A[i + j * lda]);
                    u = u + S[j] * t;
                    work[j] = work[j] + d * t;
                }
            } else {
                for (int j = 0; j <= i; j++) {
                    t = cabs1(A[i + j * lda]);
                    u = u + S[j] * t;
                    work[j] = work[j] + d * t;
                }
                for (int j = i + 1; j < n; j++) {
                    t = cabs1(A[j + i * lda]);
                    u = u + S[j] * t;
                    work[j] = work[j] + d * t;
                }
            }

            avg = avg + (u + creal(work[i])) * d / n;
            S[i] = si;
        }
    }

    f64 smlnum = dlamch("SAFEMIN");
    f64 bignum = ONE / smlnum;
    f64 smin = bignum;
    f64 smax = ZERO;
    f64 t = ONE / sqrt(avg > ZERO ? avg : 1.0);
    f64 base = dlamch("B");
    f64 u = ONE / log(base);
    for (int i = 0; i < n; i++) {
        S[i] = pow(base, (int)(u * log(S[i] * t)));
        smin = (smin < S[i]) ? smin : S[i];
        smax = (smax > S[i]) ? smax : S[i];
    }
    f64 smin_safe = (smin > smlnum) ? smin : smlnum;
    f64 smax_safe = (smax < bignum) ? smax : bignum;
    *scond = smin_safe / smax_safe;
}
