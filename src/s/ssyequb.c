/**
 * @file ssyequb.c
 * @brief SSYEQUB computes row and column scalings intended to equilibrate
 *        a symmetric matrix and reduce its condition number.
 */

#include <math.h>
#include "semicolon_lapack_single.h"

/**
 * SSYEQUB computes row and column scalings intended to equilibrate a
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
 * @param[in]     A      Double precision array, dimension (lda, n).
 *                       The N-by-N symmetric matrix whose scaling factors
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
 * @param[out]    work   Double precision array, dimension (2*n).
 * @param[out]    info   = 0: successful exit
 *                       < 0: if info = -i, the i-th argument had an illegal value
 *                       > 0: if info = i, the i-th diagonal element is nonpositive.
 *
 * References:
 *   Livne, O.E. and Golub, G.H., "Scaling by Binormalization",
 *   Numerical Algorithms, vol. 35, no. 1, pp. 97-120, January 2004.
 */
void ssyequb(
    const char* uplo,
    const int n,
    const float* const restrict A,
    const int lda,
    float* const restrict S,
    float* scond,
    float* amax,
    float* const restrict work,
    int* info)
{
    const float ONE = 1.0f;
    const float ZERO = 0.0f;
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
        xerbla("SSYEQUB", -(*info));
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
                float absval = fabsf(A[i + j * lda]);
                S[i] = (S[i] > absval) ? S[i] : absval;
                S[j] = (S[j] > absval) ? S[j] : absval;
                *amax = (*amax > absval) ? *amax : absval;
            }
            float absdiag = fabsf(A[j + j * lda]);
            S[j] = (S[j] > absdiag) ? S[j] : absdiag;
            *amax = (*amax > absdiag) ? *amax : absdiag;
        }
    } else {
        for (int j = 0; j < n; j++) {
            float absdiag = fabsf(A[j + j * lda]);
            S[j] = (S[j] > absdiag) ? S[j] : absdiag;
            *amax = (*amax > absdiag) ? *amax : absdiag;
            for (int i = j + 1; i < n; i++) {
                float absval = fabsf(A[i + j * lda]);
                S[i] = (S[i] > absval) ? S[i] : absval;
                S[j] = (S[j] > absval) ? S[j] : absval;
                *amax = (*amax > absval) ? *amax : absval;
            }
        }
    }
    for (int j = 0; j < n; j++) {
        S[j] = ONE / S[j];
    }

    float tol_val = ONE / sqrtf(2.0f * n);
    float avg = ZERO;

    for (int iter = 0; iter < MAX_ITER; iter++) {
        float scale = ZERO;
        float sumsq = ZERO;

        // beta = |A|s
        for (int i = 0; i < n; i++) {
            work[i] = ZERO;
        }
        if (up) {
            for (int j = 0; j < n; j++) {
                for (int i = 0; i < j; i++) {
                    float absval = fabsf(A[i + j * lda]);
                    work[i] = work[i] + absval * S[j];
                    work[j] = work[j] + absval * S[i];
                }
                work[j] = work[j] + fabsf(A[j + j * lda]) * S[j];
            }
        } else {
            for (int j = 0; j < n; j++) {
                work[j] = work[j] + fabsf(A[j + j * lda]) * S[j];
                for (int i = j + 1; i < n; i++) {
                    float absval = fabsf(A[i + j * lda]);
                    work[i] = work[i] + absval * S[j];
                    work[j] = work[j] + absval * S[i];
                }
            }
        }

        // avg = s^T beta / n
        avg = ZERO;
        for (int i = 0; i < n; i++) {
            avg = avg + S[i] * work[i];
        }
        avg = avg / n;

        float std_dev = ZERO;
        for (int i = 0; i < n; i++) {
            work[n + i] = S[i] * work[i] - avg;
        }
        slassq(n, &work[n], 1, &scale, &sumsq);
        std_dev = scale * sqrtf(sumsq / n);

        if (std_dev < tol_val * avg) {
            break;
        }

        for (int i = 0; i < n; i++) {
            float t = fabsf(A[i + i * lda]);
            float si = S[i];
            float c2 = (n - 1) * t;
            float c1 = (n - 2) * (work[i] - t * si);
            float c0 = -(t * si) * si + 2 * work[i] * si - n * avg;
            float d = c1 * c1 - 4 * c0 * c2;

            if (d <= ZERO) {
                *info = -1;
                return;
            }
            si = -2 * c0 / (c1 + sqrtf(d));

            d = si - S[i];
            float u = ZERO;
            if (up) {
                for (int j = 0; j <= i; j++) {
                    t = fabsf(A[j + i * lda]);
                    u = u + S[j] * t;
                    work[j] = work[j] + d * t;
                }
                for (int j = i + 1; j < n; j++) {
                    t = fabsf(A[i + j * lda]);
                    u = u + S[j] * t;
                    work[j] = work[j] + d * t;
                }
            } else {
                for (int j = 0; j <= i; j++) {
                    t = fabsf(A[i + j * lda]);
                    u = u + S[j] * t;
                    work[j] = work[j] + d * t;
                }
                for (int j = i + 1; j < n; j++) {
                    t = fabsf(A[j + i * lda]);
                    u = u + S[j] * t;
                    work[j] = work[j] + d * t;
                }
            }

            avg = avg + (u + work[i]) * d / n;
            S[i] = si;
        }
    }

    float smlnum = slamch("SAFEMIN");
    float bignum = ONE / smlnum;
    float smin = bignum;
    float smax = ZERO;
    float t = ONE / sqrtf(avg > ZERO ? avg : 1.0f);
    float base = slamch("B");
    float u = ONE / logf(base);
    for (int i = 0; i < n; i++) {
        S[i] = powf(base, (int)(u * logf(S[i] * t)));
        smin = (smin < S[i]) ? smin : S[i];
        smax = (smax > S[i]) ? smax : S[i];
    }
    float smin_safe = (smin > smlnum) ? smin : smlnum;
    float smax_safe = (smax < bignum) ? smax : bignum;
    *scond = smin_safe / smax_safe;
}
