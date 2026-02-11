/**
 * @file ssbgst.c
 * @brief SSBGST reduces a symmetric-definite banded generalized eigenproblem to standard form.
 */

#include <cblas.h>
#include "semicolon_lapack_single.h"

/**
 * SSBGST reduces a real symmetric-definite banded generalized
 * eigenproblem A*x = lambda*B*x to standard form C*y = lambda*y,
 * such that C has the same bandwidth as A.
 *
 * B must have been previously factorized as S**T*S by SPBSTF, using a
 * split Cholesky factorization. A is overwritten by C = X**T*A*X, where
 * X = S**(-1)*Q and Q is an orthogonal matrix chosen to preserve the
 * bandwidth of A.
 *
 * @param[in]     vect   = 'N': do not form the transformation matrix X
 *                        = 'V': form X
 * @param[in]     uplo   = 'U': Upper triangle of A is stored
 *                        = 'L': Lower triangle of A is stored
 * @param[in]     n      The order of the matrices A and B. n >= 0.
 * @param[in]     ka     The number of superdiagonals (if uplo='U') or
 *                       subdiagonals (if uplo='L') of A. ka >= 0.
 * @param[in]     kb     The number of superdiagonals (if uplo='U') or
 *                       subdiagonals (if uplo='L') of B. ka >= kb >= 0.
 * @param[in,out] AB     The banded matrix A. Array of dimension (ldab, n).
 *                       On exit, the transformed matrix C.
 * @param[in]     ldab   The leading dimension of AB. ldab >= ka+1.
 * @param[in]     BB     The banded factor S from the split Cholesky
 *                       factorization of B. Array of dimension (ldbb, n).
 * @param[in]     ldbb   The leading dimension of BB. ldbb >= kb+1.
 * @param[out]    X      If vect='V', the n-by-n transformation matrix X.
 *                       Array of dimension (ldx, n).
 * @param[in]     ldx    The leading dimension of X.
 *                       ldx >= max(1,n) if vect='V'; ldx >= 1 otherwise.
 * @param[out]    work   Workspace array of dimension (2*n).
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 */
void ssbgst(
    const char* vect,
    const char* uplo,
    const int n,
    const int ka,
    const int kb,
    float* const restrict AB,
    const int ldab,
    const float* const restrict BB,
    const int ldbb,
    float* const restrict X,
    const int ldx,
    float* const restrict work,
    int* info)
{
    const float ZERO = 0.0f;
    const float ONE = 1.0f;

    int update, upper, wantx;
    int i, i0, i1, i2, inca, j, j1, j1t, j2, j2t, k, ka1, kbt, l, m, nr, nrt, nx;
    float bii, ra, ra1 = 0.0f, t;

    *info = 0;
    wantx = (vect[0] == 'V' || vect[0] == 'v');
    upper = (uplo[0] == 'U' || uplo[0] == 'u');
    ka1 = ka + 1;

    if (!wantx && !(vect[0] == 'N' || vect[0] == 'n')) {
        *info = -1;
    } else if (!upper && !(uplo[0] == 'L' || uplo[0] == 'l')) {
        *info = -2;
    } else if (n < 0) {
        *info = -3;
    } else if (ka < 0) {
        *info = -4;
    } else if (kb < 0 || kb > ka) {
        *info = -5;
    } else if (ldab < ka + 1) {
        *info = -7;
    } else if (ldbb < kb + 1) {
        *info = -9;
    } else if (ldx < 1 || (wantx && ldx < (1 > n ? 1 : n))) {
        *info = -11;
    }
    if (*info != 0) {
        xerbla("SSBGST", -(*info));
        return;
    }

    if (n == 0)
        return;

    inca = ldab * ka1;

    // Initialize X to the unit matrix, if needed
    if (wantx)
        slaset("Full", n, n, ZERO, ONE, X, ldx);

    // Set M to the splitting point m (same as SPBSTF)
    m = (n + kb) / 2;

    // Phase 1: Process rows n, n-1, ..., m+1 then propagate bulges
    update = 1;
    i = n + 1;

L10:
    if (update) {
        i = i - 1;
        kbt = (kb < i - 1) ? kb : (i - 1);
        i0 = i - 1;
        i1 = (n < i + ka) ? n : (i + ka);
        i2 = i - kbt + ka1;
        if (i < m + 1) {
            update = 0;
            i = i + 1;
            i0 = m;
            if (ka == 0)
                goto L480;
            goto L10;
        }
    } else {
        i = i + ka;
        if (i > n - 1)
            goto L480;
    }

    if (upper) {
        // Transform A, working with the upper triangle
        if (update) {
            // Form inv(S(i))**T * A * inv(S(i))
            bii = BB[kb + i * ldbb];  // BB(KB1, I) in 1-based
            for (j = i; j <= i1; j++) {
                AB[i - j + ka + j * ldab] = AB[i - j + ka + j * ldab] / bii;
            }
            int jmax = (1 > i - ka) ? 1 : (i - ka);
            for (j = jmax; j <= i; j++) {
                AB[j - i + ka + i * ldab] = AB[j - i + ka + i * ldab] / bii;
            }
            for (k = i - kbt; k <= i - 1; k++) {
                for (j = i - kbt; j <= k; j++) {
                    AB[j - k + ka + k * ldab] = AB[j - k + ka + k * ldab]
                        - BB[j - i + kb + i * ldbb] * AB[k - i + ka + i * ldab]
                        - BB[k - i + kb + i * ldbb] * AB[j - i + ka + i * ldab]
                        + AB[ka + i * ldab] * BB[j - i + kb + i * ldbb] * BB[k - i + kb + i * ldbb];
                }
                jmax = (1 > i - ka) ? 1 : (i - ka);
                for (j = jmax; j <= i - kbt - 1; j++) {
                    AB[j - k + ka + k * ldab] = AB[j - k + ka + k * ldab]
                        - BB[k - i + kb + i * ldbb] * AB[j - i + ka + i * ldab];
                }
            }
            for (j = i; j <= i1; j++) {
                int kmax = (j - ka > i - kbt) ? (j - ka) : (i - kbt);
                for (k = kmax; k <= i - 1; k++) {
                    AB[k - j + ka + j * ldab] = AB[k - j + ka + j * ldab]
                        - BB[k - i + kb + i * ldbb] * AB[i - j + ka + j * ldab];
                }
            }

            if (wantx) {
                // post-multiply X by inv(S(i))
                cblas_sscal(n - m, ONE / bii, &X[m + i * ldx], 1);
                if (kbt > 0)
                    cblas_sger(CblasColMajor, n - m, kbt, -ONE, &X[m + i * ldx], 1,
                               &BB[kb - kbt + i * ldbb], 1, &X[m + (i - kbt) * ldx], ldx);
            }

            // store a(i,i1) in RA1 for use in next loop over K
            ra1 = AB[i - i1 + ka + i1 * ldab];
        }

        // Generate and apply vectors of rotations to chase all the
        // existing bulges KA positions down toward the bottom of the band
        for (k = 1; k <= kb - 1; k++) {
            if (update) {
                // Determine the rotations which would annihilate the bulge
                if (i - k + ka < n && i - k > 1) {
                    // generate rotation to annihilate a(i,i-k+ka+1)
                    slartg(AB[k + (i - k + ka) * ldab], ra1,
                           &work[n + i - k + ka - m - 1], &work[i - k + ka - m - 1], &ra);

                    // create nonzero element a(i-k,i-k+ka+1) outside the band
                    t = -BB[kb - k + i * ldbb] * ra1;
                    work[i - k - 1] = work[n + i - k + ka - m - 1] * t
                        - work[i - k + ka - m - 1] * AB[0 + (i - k + ka) * ldab];
                    AB[0 + (i - k + ka) * ldab] = work[i - k + ka - m - 1] * t
                        + work[n + i - k + ka - m - 1] * AB[0 + (i - k + ka) * ldab];
                    ra1 = ra;
                }
            }
            j2 = i - k - 1 + ((1 > k - i0 + 2) ? 1 : (k - i0 + 2)) * ka1;
            nr = (n - j2 + ka) / ka1;
            j1 = j2 + (nr - 1) * ka1;
            if (update) {
                j2t = (j2 > i + 2 * ka - k + 1) ? j2 : (i + 2 * ka - k + 1);
            } else {
                j2t = j2;
            }
            nrt = (n - j2t + ka) / ka1;
            for (j = j2t; j <= j1; j += ka1) {
                // create nonzero element a(j-ka,j+1) outside the band
                work[j - m - 1] = work[j - m - 1] * AB[0 + (j + 1) * ldab];
                AB[0 + (j + 1) * ldab] = work[n + j - m - 1] * AB[0 + (j + 1) * ldab];
            }

            // generate rotations in 1st set to annihilate elements outside the band
            if (nrt > 0)
                slargv(nrt, &AB[0 + j2t * ldab], inca, &work[j2t - m - 1], ka1,
                       &work[n + j2t - m - 1], ka1);
            if (nr > 0) {
                // apply rotations in 1st set from the right
                for (l = 1; l <= ka - 1; l++) {
                    slartv(nr, &AB[ka1 - l - 1 + j2 * ldab], inca,
                           &AB[ka - l - 1 + (j2 + 1) * ldab], inca,
                           &work[n + j2 - m - 1], &work[j2 - m - 1], ka1);
                }

                // apply rotations in 1st set from both sides to diagonal blocks
                slar2v(nr, &AB[ka + j2 * ldab], &AB[ka + (j2 + 1) * ldab],
                       &AB[ka - 1 + (j2 + 1) * ldab], inca,
                       &work[n + j2 - m - 1], &work[j2 - m - 1], ka1);
            }

            // start applying rotations in 1st set from the left
            for (l = ka - 1; l >= kb - k + 1; l--) {
                nrt = (n - j2 + l) / ka1;
                if (nrt > 0)
                    slartv(nrt, &AB[l - 1 + (j2 + ka1 - l) * ldab], inca,
                           &AB[l + (j2 + ka1 - l) * ldab], inca,
                           &work[n + j2 - m - 1], &work[j2 - m - 1], ka1);
            }

            if (wantx) {
                // post-multiply X by product of rotations in 1st set
                for (j = j2; j <= j1; j += ka1) {
                    cblas_srot(n - m, &X[m + j * ldx], 1, &X[m + (j + 1) * ldx], 1,
                               work[n + j - m - 1], work[j - m - 1]);
                }
            }
        }

        if (update) {
            if (i2 <= n && kbt > 0) {
                // create nonzero element a(i-kbt,i-kbt+ka+1) outside the band
                work[i - kbt - 1] = -BB[kb - kbt + i * ldbb] * ra1;
            }
        }

        for (k = kb; k >= 1; k--) {
            if (update) {
                j2 = i - k - 1 + ((2 > k - i0 + 1) ? 2 : (k - i0 + 1)) * ka1;
            } else {
                j2 = i - k - 1 + ((1 > k - i0 + 1) ? 1 : (k - i0 + 1)) * ka1;
            }

            // finish applying rotations in 2nd set from the left
            for (l = kb - k; l >= 1; l--) {
                nrt = (n - j2 + ka + l) / ka1;
                if (nrt > 0)
                    slartv(nrt, &AB[l - 1 + (j2 - l + 1) * ldab], inca,
                           &AB[l + (j2 - l + 1) * ldab], inca,
                           &work[n + j2 - ka - 1], &work[j2 - ka - 1], ka1);
            }
            nr = (n - j2 + ka) / ka1;
            j1 = j2 + (nr - 1) * ka1;
            for (j = j1; j >= j2; j -= ka1) {
                work[j - 1] = work[j - ka - 1];
                work[n + j - 1] = work[n + j - ka - 1];
            }
            for (j = j2; j <= j1; j += ka1) {
                // create nonzero element a(j-ka,j+1) outside the band
                work[j - 1] = work[j - 1] * AB[0 + (j + 1) * ldab];
                AB[0 + (j + 1) * ldab] = work[n + j - 1] * AB[0 + (j + 1) * ldab];
            }
            if (update) {
                if (i - k < n - ka && k <= kbt)
                    work[i - k + ka - 1] = work[i - k - 1];
            }
        }

        for (k = kb; k >= 1; k--) {
            j2 = i - k - 1 + ((1 > k - i0 + 1) ? 1 : (k - i0 + 1)) * ka1;
            nr = (n - j2 + ka) / ka1;
            j1 = j2 + (nr - 1) * ka1;
            if (nr > 0) {
                // generate rotations in 2nd set to annihilate elements outside the band
                slargv(nr, &AB[0 + j2 * ldab], inca, &work[j2 - 1], ka1, &work[n + j2 - 1], ka1);

                // apply rotations in 2nd set from the right
                for (l = 1; l <= ka - 1; l++) {
                    slartv(nr, &AB[ka1 - l - 1 + j2 * ldab], inca,
                           &AB[ka - l - 1 + (j2 + 1) * ldab], inca,
                           &work[n + j2 - 1], &work[j2 - 1], ka1);
                }

                // apply rotations in 2nd set from both sides to diagonal blocks
                slar2v(nr, &AB[ka + j2 * ldab], &AB[ka + (j2 + 1) * ldab],
                       &AB[ka - 1 + (j2 + 1) * ldab], inca,
                       &work[n + j2 - 1], &work[j2 - 1], ka1);
            }

            // start applying rotations in 2nd set from the left
            for (l = ka - 1; l >= kb - k + 1; l--) {
                nrt = (n - j2 + l) / ka1;
                if (nrt > 0)
                    slartv(nrt, &AB[l - 1 + (j2 + ka1 - l) * ldab], inca,
                           &AB[l + (j2 + ka1 - l) * ldab], inca,
                           &work[n + j2 - 1], &work[j2 - 1], ka1);
            }

            if (wantx) {
                // post-multiply X by product of rotations in 2nd set
                for (j = j2; j <= j1; j += ka1) {
                    cblas_srot(n - m, &X[m + j * ldx], 1, &X[m + (j + 1) * ldx], 1,
                               work[n + j - 1], work[j - 1]);
                }
            }
        }

        for (k = 1; k <= kb - 1; k++) {
            j2 = i - k - 1 + ((1 > k - i0 + 2) ? 1 : (k - i0 + 2)) * ka1;

            // finish applying rotations in 1st set from the left
            for (l = kb - k; l >= 1; l--) {
                nrt = (n - j2 + l) / ka1;
                if (nrt > 0)
                    slartv(nrt, &AB[l - 1 + (j2 + ka1 - l) * ldab], inca,
                           &AB[l + (j2 + ka1 - l) * ldab], inca,
                           &work[n + j2 - m - 1], &work[j2 - m - 1], ka1);
            }
        }

        if (kb > 1) {
            for (j = n - 1; j >= i - kb + 2 * ka + 1; j--) {
                work[n + j - m - 1] = work[n + j - ka - m - 1];
                work[j - m - 1] = work[j - ka - m - 1];
            }
        }

    } else {
        // Transform A, working with the lower triangle
        if (update) {
            // Form inv(S(i))**T * A * inv(S(i))
            bii = BB[0 + i * ldbb];  // BB(1, I) in 1-based
            for (j = i; j <= i1; j++) {
                AB[j - i + i * ldab] = AB[j - i + i * ldab] / bii;
            }
            int jmax = (1 > i - ka) ? 1 : (i - ka);
            for (j = jmax; j <= i; j++) {
                AB[i - j + j * ldab] = AB[i - j + j * ldab] / bii;
            }
            for (k = i - kbt; k <= i - 1; k++) {
                for (j = i - kbt; j <= k; j++) {
                    AB[k - j + j * ldab] = AB[k - j + j * ldab]
                        - BB[i - j + j * ldbb] * AB[i - k + k * ldab]
                        - BB[i - k + k * ldbb] * AB[i - j + j * ldab]
                        + AB[0 + i * ldab] * BB[i - j + j * ldbb] * BB[i - k + k * ldbb];
                }
                jmax = (1 > i - ka) ? 1 : (i - ka);
                for (j = jmax; j <= i - kbt - 1; j++) {
                    AB[k - j + j * ldab] = AB[k - j + j * ldab]
                        - BB[i - k + k * ldbb] * AB[i - j + j * ldab];
                }
            }
            for (j = i; j <= i1; j++) {
                int kmax = (j - ka > i - kbt) ? (j - ka) : (i - kbt);
                for (k = kmax; k <= i - 1; k++) {
                    AB[j - k + k * ldab] = AB[j - k + k * ldab]
                        - BB[i - k + k * ldbb] * AB[j - i + i * ldab];
                }
            }

            if (wantx) {
                // post-multiply X by inv(S(i))
                cblas_sscal(n - m, ONE / bii, &X[m + i * ldx], 1);
                if (kbt > 0)
                    cblas_sger(CblasColMajor, n - m, kbt, -ONE, &X[m + i * ldx], 1,
                               &BB[kbt + (i - kbt) * ldbb], ldbb - 1, &X[m + (i - kbt) * ldx], ldx);
            }

            // store a(i1,i) in RA1 for use in next loop over K
            ra1 = AB[i1 - i + i * ldab];
        }

        // Generate and apply vectors of rotations to chase all the
        // existing bulges KA positions down toward the bottom of the band
        for (k = 1; k <= kb - 1; k++) {
            if (update) {
                // Determine the rotations which would annihilate the bulge
                if (i - k + ka < n && i - k > 1) {
                    // generate rotation to annihilate a(i-k+ka+1,i)
                    slartg(AB[ka1 - k - 1 + i * ldab], ra1,
                           &work[n + i - k + ka - m - 1], &work[i - k + ka - m - 1], &ra);

                    // create nonzero element a(i-k+ka+1,i-k) outside the band
                    t = -BB[k + (i - k) * ldbb] * ra1;
                    work[i - k - 1] = work[n + i - k + ka - m - 1] * t
                        - work[i - k + ka - m - 1] * AB[ka + (i - k) * ldab];
                    AB[ka + (i - k) * ldab] = work[i - k + ka - m - 1] * t
                        + work[n + i - k + ka - m - 1] * AB[ka + (i - k) * ldab];
                    ra1 = ra;
                }
            }
            j2 = i - k - 1 + ((1 > k - i0 + 2) ? 1 : (k - i0 + 2)) * ka1;
            nr = (n - j2 + ka) / ka1;
            j1 = j2 + (nr - 1) * ka1;
            if (update) {
                j2t = (j2 > i + 2 * ka - k + 1) ? j2 : (i + 2 * ka - k + 1);
            } else {
                j2t = j2;
            }
            nrt = (n - j2t + ka) / ka1;
            for (j = j2t; j <= j1; j += ka1) {
                // create nonzero element a(j+1,j-ka) outside the band
                work[j - m - 1] = work[j - m - 1] * AB[ka + (j - ka + 1) * ldab];
                AB[ka + (j - ka + 1) * ldab] = work[n + j - m - 1] * AB[ka + (j - ka + 1) * ldab];
            }

            // generate rotations in 1st set to annihilate elements outside the band
            if (nrt > 0)
                slargv(nrt, &AB[ka + (j2t - ka) * ldab], inca, &work[j2t - m - 1], ka1,
                       &work[n + j2t - m - 1], ka1);
            if (nr > 0) {
                // apply rotations in 1st set from the left
                for (l = 1; l <= ka - 1; l++) {
                    slartv(nr, &AB[l + (j2 - l) * ldab], inca,
                           &AB[l + 1 + (j2 - l) * ldab], inca,
                           &work[n + j2 - m - 1], &work[j2 - m - 1], ka1);
                }

                // apply rotations in 1st set from both sides to diagonal blocks
                slar2v(nr, &AB[0 + j2 * ldab], &AB[0 + (j2 + 1) * ldab],
                       &AB[1 + j2 * ldab], inca,
                       &work[n + j2 - m - 1], &work[j2 - m - 1], ka1);
            }

            // start applying rotations in 1st set from the right
            for (l = ka - 1; l >= kb - k + 1; l--) {
                nrt = (n - j2 + l) / ka1;
                if (nrt > 0)
                    slartv(nrt, &AB[ka1 - l + j2 * ldab], inca,
                           &AB[ka - l + (j2 + 1) * ldab], inca,
                           &work[n + j2 - m - 1], &work[j2 - m - 1], ka1);
            }

            if (wantx) {
                // post-multiply X by product of rotations in 1st set
                for (j = j2; j <= j1; j += ka1) {
                    cblas_srot(n - m, &X[m + j * ldx], 1, &X[m + (j + 1) * ldx], 1,
                               work[n + j - m - 1], work[j - m - 1]);
                }
            }
        }

        if (update) {
            if (i2 <= n && kbt > 0) {
                // create nonzero element a(i-kbt+ka+1,i-kbt) outside the band
                work[i - kbt - 1] = -BB[kbt + (i - kbt) * ldbb] * ra1;
            }
        }

        for (k = kb; k >= 1; k--) {
            if (update) {
                j2 = i - k - 1 + ((2 > k - i0 + 1) ? 2 : (k - i0 + 1)) * ka1;
            } else {
                j2 = i - k - 1 + ((1 > k - i0 + 1) ? 1 : (k - i0 + 1)) * ka1;
            }

            // finish applying rotations in 2nd set from the right
            for (l = kb - k; l >= 1; l--) {
                nrt = (n - j2 + ka + l) / ka1;
                if (nrt > 0)
                    slartv(nrt, &AB[ka1 - l + (j2 - ka) * ldab], inca,
                           &AB[ka - l + (j2 - ka + 1) * ldab], inca,
                           &work[n + j2 - ka - 1], &work[j2 - ka - 1], ka1);
            }
            nr = (n - j2 + ka) / ka1;
            j1 = j2 + (nr - 1) * ka1;
            for (j = j1; j >= j2; j -= ka1) {
                work[j - 1] = work[j - ka - 1];
                work[n + j - 1] = work[n + j - ka - 1];
            }
            for (j = j2; j <= j1; j += ka1) {
                // create nonzero element a(j+1,j-ka) outside the band
                work[j - 1] = work[j - 1] * AB[ka + (j - ka + 1) * ldab];
                AB[ka + (j - ka + 1) * ldab] = work[n + j - 1] * AB[ka + (j - ka + 1) * ldab];
            }
            if (update) {
                if (i - k < n - ka && k <= kbt)
                    work[i - k + ka - 1] = work[i - k - 1];
            }
        }

        for (k = kb; k >= 1; k--) {
            j2 = i - k - 1 + ((1 > k - i0 + 1) ? 1 : (k - i0 + 1)) * ka1;
            nr = (n - j2 + ka) / ka1;
            j1 = j2 + (nr - 1) * ka1;
            if (nr > 0) {
                // generate rotations in 2nd set to annihilate elements outside the band
                slargv(nr, &AB[ka + (j2 - ka) * ldab], inca, &work[j2 - 1], ka1, &work[n + j2 - 1], ka1);

                // apply rotations in 2nd set from the left
                for (l = 1; l <= ka - 1; l++) {
                    slartv(nr, &AB[l + (j2 - l) * ldab], inca,
                           &AB[l + 1 + (j2 - l) * ldab], inca,
                           &work[n + j2 - 1], &work[j2 - 1], ka1);
                }

                // apply rotations in 2nd set from both sides to diagonal blocks
                slar2v(nr, &AB[0 + j2 * ldab], &AB[0 + (j2 + 1) * ldab],
                       &AB[1 + j2 * ldab], inca,
                       &work[n + j2 - 1], &work[j2 - 1], ka1);
            }

            // start applying rotations in 2nd set from the right
            for (l = ka - 1; l >= kb - k + 1; l--) {
                nrt = (n - j2 + l) / ka1;
                if (nrt > 0)
                    slartv(nrt, &AB[ka1 - l + j2 * ldab], inca,
                           &AB[ka - l + (j2 + 1) * ldab], inca,
                           &work[n + j2 - 1], &work[j2 - 1], ka1);
            }

            if (wantx) {
                // post-multiply X by product of rotations in 2nd set
                for (j = j2; j <= j1; j += ka1) {
                    cblas_srot(n - m, &X[m + j * ldx], 1, &X[m + (j + 1) * ldx], 1,
                               work[n + j - 1], work[j - 1]);
                }
            }
        }

        for (k = 1; k <= kb - 1; k++) {
            j2 = i - k - 1 + ((1 > k - i0 + 2) ? 1 : (k - i0 + 2)) * ka1;

            // finish applying rotations in 1st set from the right
            for (l = kb - k; l >= 1; l--) {
                nrt = (n - j2 + l) / ka1;
                if (nrt > 0)
                    slartv(nrt, &AB[ka1 - l + j2 * ldab], inca,
                           &AB[ka - l + (j2 + 1) * ldab], inca,
                           &work[n + j2 - m - 1], &work[j2 - m - 1], ka1);
            }
        }

        if (kb > 1) {
            for (j = n - 1; j >= i - kb + 2 * ka + 1; j--) {
                work[n + j - m - 1] = work[n + j - ka - m - 1];
                work[j - m - 1] = work[j - ka - m - 1];
            }
        }
    }

    goto L10;

L480:
    // Phase 2: Process rows 1, 2, ..., m then propagate bulges upward
    update = 1;
    i = 0;

L490:
    if (update) {
        i = i + 1;
        kbt = (kb < m - i) ? kb : (m - i);
        i0 = i + 1;
        i1 = (1 > i - ka) ? 1 : (i - ka);
        i2 = i + kbt - ka1;
        if (i > m) {
            update = 0;
            i = i - 1;
            i0 = m + 1;
            if (ka == 0)
                return;
            goto L490;
        }
    } else {
        i = i - ka;
        if (i < 2)
            return;
    }

    if (i < m - kbt) {
        nx = m;
    } else {
        nx = n;
    }

    if (upper) {
        // Transform A, working with the upper triangle
        if (update) {
            // Form inv(S(i))**T * A * inv(S(i))
            bii = BB[kb + i * ldbb];
            for (j = i1; j <= i; j++) {
                AB[j - i + ka + i * ldab] = AB[j - i + ka + i * ldab] / bii;
            }
            int jmin = (n < i + ka) ? n : (i + ka);
            for (j = i; j <= jmin; j++) {
                AB[i - j + ka + j * ldab] = AB[i - j + ka + j * ldab] / bii;
            }
            for (k = i + 1; k <= i + kbt; k++) {
                for (j = k; j <= i + kbt; j++) {
                    AB[k - j + ka + j * ldab] = AB[k - j + ka + j * ldab]
                        - BB[i - j + kb + j * ldbb] * AB[i - k + ka + k * ldab]
                        - BB[i - k + kb + k * ldbb] * AB[i - j + ka + j * ldab]
                        + AB[ka + i * ldab] * BB[i - j + kb + j * ldbb] * BB[i - k + kb + k * ldbb];
                }
                jmin = (n < i + ka) ? n : (i + ka);
                for (j = i + kbt + 1; j <= jmin; j++) {
                    AB[k - j + ka + j * ldab] = AB[k - j + ka + j * ldab]
                        - BB[i - k + kb + k * ldbb] * AB[i - j + ka + j * ldab];
                }
            }
            for (j = i1; j <= i; j++) {
                int kmin = (j + ka < i + kbt) ? (j + ka) : (i + kbt);
                for (k = i + 1; k <= kmin; k++) {
                    AB[j - k + ka + k * ldab] = AB[j - k + ka + k * ldab]
                        - BB[i - k + kb + k * ldbb] * AB[j - i + ka + i * ldab];
                }
            }

            if (wantx) {
                // post-multiply X by inv(S(i))
                cblas_sscal(nx, ONE / bii, &X[0 + i * ldx], 1);
                if (kbt > 0)
                    cblas_sger(CblasColMajor, nx, kbt, -ONE, &X[0 + i * ldx], 1,
                               &BB[kb - 1 + (i + 1) * ldbb], ldbb - 1, &X[0 + (i + 1) * ldx], ldx);
            }

            // store a(i1,i) in RA1 for use in next loop over K
            ra1 = AB[i1 - i + ka + i * ldab];
        }

        // Generate and apply vectors of rotations to chase all the
        // existing bulges KA positions up toward the top of the band
        for (k = 1; k <= kb - 1; k++) {
            if (update) {
                // Determine the rotations which would annihilate the bulge
                if (i + k - ka1 > 0 && i + k < m) {
                    // generate rotation to annihilate a(i+k-ka-1,i)
                    slartg(AB[k + i * ldab], ra1,
                           &work[n + i + k - ka - 1], &work[i + k - ka - 1], &ra);

                    // create nonzero element a(i+k-ka-1,i+k) outside the band
                    t = -BB[kb - k + (i + k) * ldbb] * ra1;
                    work[m - kb + i + k - 1] = work[n + i + k - ka - 1] * t
                        - work[i + k - ka - 1] * AB[0 + (i + k) * ldab];
                    AB[0 + (i + k) * ldab] = work[i + k - ka - 1] * t
                        + work[n + i + k - ka - 1] * AB[0 + (i + k) * ldab];
                    ra1 = ra;
                }
            }
            j2 = i + k + 1 - ((1 > k + i0 - m + 1) ? 1 : (k + i0 - m + 1)) * ka1;
            nr = (j2 + ka - 1) / ka1;
            j1 = j2 - (nr - 1) * ka1;
            if (update) {
                j2t = (j2 < i - 2 * ka + k - 1) ? j2 : (i - 2 * ka + k - 1);
            } else {
                j2t = j2;
            }
            nrt = (j2t + ka - 1) / ka1;
            for (j = j1; j <= j2t; j += ka1) {
                // create nonzero element a(j-1,j+ka) outside the band
                work[j - 1] = work[j - 1] * AB[0 + (j + ka - 1) * ldab];
                AB[0 + (j + ka - 1) * ldab] = work[n + j - 1] * AB[0 + (j + ka - 1) * ldab];
            }

            // generate rotations in 1st set to annihilate elements outside the band
            if (nrt > 0)
                slargv(nrt, &AB[0 + (j1 + ka) * ldab], inca, &work[j1 - 1], ka1,
                       &work[n + j1 - 1], ka1);
            if (nr > 0) {
                // apply rotations in 1st set from the left
                for (l = 1; l <= ka - 1; l++) {
                    slartv(nr, &AB[ka1 - l - 1 + (j1 + l) * ldab], inca,
                           &AB[ka - l - 1 + (j1 + l) * ldab], inca,
                           &work[n + j1 - 1], &work[j1 - 1], ka1);
                }

                // apply rotations in 1st set from both sides to diagonal blocks
                slar2v(nr, &AB[ka + j1 * ldab], &AB[ka + (j1 - 1) * ldab],
                       &AB[ka - 1 + j1 * ldab], inca,
                       &work[n + j1 - 1], &work[j1 - 1], ka1);
            }

            // start applying rotations in 1st set from the right
            for (l = ka - 1; l >= kb - k + 1; l--) {
                nrt = (j2 + l - 1) / ka1;
                j1t = j2 - (nrt - 1) * ka1;
                if (nrt > 0)
                    slartv(nrt, &AB[l - 1 + j1t * ldab], inca,
                           &AB[l + (j1t - 1) * ldab], inca,
                           &work[n + j1t - 1], &work[j1t - 1], ka1);
            }

            if (wantx) {
                // post-multiply X by product of rotations in 1st set
                for (j = j1; j <= j2; j += ka1) {
                    cblas_srot(nx, &X[0 + j * ldx], 1, &X[0 + (j - 1) * ldx], 1,
                               work[n + j - 1], work[j - 1]);
                }
            }
        }

        if (update) {
            if (i2 > 0 && kbt > 0) {
                // create nonzero element a(i+kbt-ka-1,i+kbt) outside the band
                work[m - kb + i + kbt - 1] = -BB[kb - kbt + (i + kbt) * ldbb] * ra1;
            }
        }

        for (k = kb; k >= 1; k--) {
            if (update) {
                j2 = i + k + 1 - ((2 > k + i0 - m) ? 2 : (k + i0 - m)) * ka1;
            } else {
                j2 = i + k + 1 - ((1 > k + i0 - m) ? 1 : (k + i0 - m)) * ka1;
            }

            // finish applying rotations in 2nd set from the right
            for (l = kb - k; l >= 1; l--) {
                nrt = (j2 + ka + l - 1) / ka1;
                j1t = j2 - (nrt - 1) * ka1;
                if (nrt > 0)
                    slartv(nrt, &AB[l - 1 + (j1t + ka) * ldab], inca,
                           &AB[l + (j1t + ka - 1) * ldab], inca,
                           &work[n + m - kb + j1t + ka - 1], &work[m - kb + j1t + ka - 1], ka1);
            }
            nr = (j2 + ka - 1) / ka1;
            j1 = j2 - (nr - 1) * ka1;
            for (j = j1; j <= j2; j += ka1) {
                work[m - kb + j - 1] = work[m - kb + j + ka - 1];
                work[n + m - kb + j - 1] = work[n + m - kb + j + ka - 1];
            }
            for (j = j1; j <= j2; j += ka1) {
                // create nonzero element a(j-1,j+ka) outside the band
                work[m - kb + j - 1] = work[m - kb + j - 1] * AB[0 + (j + ka - 1) * ldab];
                AB[0 + (j + ka - 1) * ldab] = work[n + m - kb + j - 1] * AB[0 + (j + ka - 1) * ldab];
            }
            if (update) {
                if (i + k > ka1 && k <= kbt)
                    work[m - kb + i + k - ka - 1] = work[m - kb + i + k - 1];
            }
        }

        for (k = kb; k >= 1; k--) {
            j2 = i + k + 1 - ((1 > k + i0 - m) ? 1 : (k + i0 - m)) * ka1;
            nr = (j2 + ka - 1) / ka1;
            j1 = j2 - (nr - 1) * ka1;
            if (nr > 0) {
                // generate rotations in 2nd set to annihilate elements outside the band
                slargv(nr, &AB[0 + (j1 + ka) * ldab], inca, &work[m - kb + j1 - 1], ka1,
                       &work[n + m - kb + j1 - 1], ka1);

                // apply rotations in 2nd set from the left
                for (l = 1; l <= ka - 1; l++) {
                    slartv(nr, &AB[ka1 - l - 1 + (j1 + l) * ldab], inca,
                           &AB[ka - l - 1 + (j1 + l) * ldab], inca,
                           &work[n + m - kb + j1 - 1], &work[m - kb + j1 - 1], ka1);
                }

                // apply rotations in 2nd set from both sides to diagonal blocks
                slar2v(nr, &AB[ka + j1 * ldab], &AB[ka + (j1 - 1) * ldab],
                       &AB[ka - 1 + j1 * ldab], inca,
                       &work[n + m - kb + j1 - 1], &work[m - kb + j1 - 1], ka1);
            }

            // start applying rotations in 2nd set from the right
            for (l = ka - 1; l >= kb - k + 1; l--) {
                nrt = (j2 + l - 1) / ka1;
                j1t = j2 - (nrt - 1) * ka1;
                if (nrt > 0)
                    slartv(nrt, &AB[l - 1 + j1t * ldab], inca,
                           &AB[l + (j1t - 1) * ldab], inca,
                           &work[n + m - kb + j1t - 1], &work[m - kb + j1t - 1], ka1);
            }

            if (wantx) {
                // post-multiply X by product of rotations in 2nd set
                for (j = j1; j <= j2; j += ka1) {
                    cblas_srot(nx, &X[0 + j * ldx], 1, &X[0 + (j - 1) * ldx], 1,
                               work[n + m - kb + j - 1], work[m - kb + j - 1]);
                }
            }
        }

        for (k = 1; k <= kb - 1; k++) {
            j2 = i + k + 1 - ((1 > k + i0 - m + 1) ? 1 : (k + i0 - m + 1)) * ka1;

            // finish applying rotations in 1st set from the right
            for (l = kb - k; l >= 1; l--) {
                nrt = (j2 + l - 1) / ka1;
                j1t = j2 - (nrt - 1) * ka1;
                if (nrt > 0)
                    slartv(nrt, &AB[l - 1 + j1t * ldab], inca,
                           &AB[l + (j1t - 1) * ldab], inca,
                           &work[n + j1t - 1], &work[j1t - 1], ka1);
            }
        }

        if (kb > 1) {
            int jmin = (i + kb < m) ? (i + kb) : m;
            for (j = 2; j <= jmin - 2 * ka - 1; j++) {
                work[n + j - 1] = work[n + j + ka - 1];
                work[j - 1] = work[j + ka - 1];
            }
        }

    } else {
        // Transform A, working with the lower triangle
        if (update) {
            // Form inv(S(i))**T * A * inv(S(i))
            bii = BB[0 + i * ldbb];
            for (j = i1; j <= i; j++) {
                AB[i - j + j * ldab] = AB[i - j + j * ldab] / bii;
            }
            int jmin = (n < i + ka) ? n : (i + ka);
            for (j = i; j <= jmin; j++) {
                AB[j - i + i * ldab] = AB[j - i + i * ldab] / bii;
            }
            for (k = i + 1; k <= i + kbt; k++) {
                for (j = k; j <= i + kbt; j++) {
                    AB[j - k + k * ldab] = AB[j - k + k * ldab]
                        - BB[j - i + i * ldbb] * AB[k - i + i * ldab]
                        - BB[k - i + i * ldbb] * AB[j - i + i * ldab]
                        + AB[0 + i * ldab] * BB[j - i + i * ldbb] * BB[k - i + i * ldbb];
                }
                jmin = (n < i + ka) ? n : (i + ka);
                for (j = i + kbt + 1; j <= jmin; j++) {
                    AB[j - k + k * ldab] = AB[j - k + k * ldab]
                        - BB[k - i + i * ldbb] * AB[j - i + i * ldab];
                }
            }
            for (j = i1; j <= i; j++) {
                int kmin = (j + ka < i + kbt) ? (j + ka) : (i + kbt);
                for (k = i + 1; k <= kmin; k++) {
                    AB[k - j + j * ldab] = AB[k - j + j * ldab]
                        - BB[k - i + i * ldbb] * AB[i - j + j * ldab];
                }
            }

            if (wantx) {
                // post-multiply X by inv(S(i))
                cblas_sscal(nx, ONE / bii, &X[0 + i * ldx], 1);
                if (kbt > 0)
                    cblas_sger(CblasColMajor, nx, kbt, -ONE, &X[0 + i * ldx], 1,
                               &BB[1 + i * ldbb], 1, &X[0 + (i + 1) * ldx], ldx);
            }

            // store a(i,i1) in RA1 for use in next loop over K
            ra1 = AB[i - i1 + i1 * ldab];
        }

        // Generate and apply vectors of rotations to chase all the
        // existing bulges KA positions up toward the top of the band
        for (k = 1; k <= kb - 1; k++) {
            if (update) {
                // Determine the rotations which would annihilate the bulge
                if (i + k - ka1 > 0 && i + k < m) {
                    // generate rotation to annihilate a(i,i+k-ka-1)
                    slartg(AB[ka1 - k - 1 + (i + k - ka) * ldab], ra1,
                           &work[n + i + k - ka - 1], &work[i + k - ka - 1], &ra);

                    // create nonzero element a(i+k,i+k-ka-1) outside the band
                    t = -BB[k + i * ldbb] * ra1;
                    work[m - kb + i + k - 1] = work[n + i + k - ka - 1] * t
                        - work[i + k - ka - 1] * AB[ka + (i + k - ka) * ldab];
                    AB[ka + (i + k - ka) * ldab] = work[i + k - ka - 1] * t
                        + work[n + i + k - ka - 1] * AB[ka + (i + k - ka) * ldab];
                    ra1 = ra;
                }
            }
            j2 = i + k + 1 - ((1 > k + i0 - m + 1) ? 1 : (k + i0 - m + 1)) * ka1;
            nr = (j2 + ka - 1) / ka1;
            j1 = j2 - (nr - 1) * ka1;
            if (update) {
                j2t = (j2 < i - 2 * ka + k - 1) ? j2 : (i - 2 * ka + k - 1);
            } else {
                j2t = j2;
            }
            nrt = (j2t + ka - 1) / ka1;
            for (j = j1; j <= j2t; j += ka1) {
                // create nonzero element a(j+ka,j-1) outside the band
                work[j - 1] = work[j - 1] * AB[ka + (j - 1) * ldab];
                AB[ka + (j - 1) * ldab] = work[n + j - 1] * AB[ka + (j - 1) * ldab];
            }

            // generate rotations in 1st set to annihilate elements outside the band
            if (nrt > 0)
                slargv(nrt, &AB[ka + j1 * ldab], inca, &work[j1 - 1], ka1,
                       &work[n + j1 - 1], ka1);
            if (nr > 0) {
                // apply rotations in 1st set from the right
                for (l = 1; l <= ka - 1; l++) {
                    slartv(nr, &AB[l + j1 * ldab], inca,
                           &AB[l + 1 + (j1 - 1) * ldab], inca,
                           &work[n + j1 - 1], &work[j1 - 1], ka1);
                }

                // apply rotations in 1st set from both sides to diagonal blocks
                slar2v(nr, &AB[0 + j1 * ldab], &AB[0 + (j1 - 1) * ldab],
                       &AB[1 + (j1 - 1) * ldab], inca,
                       &work[n + j1 - 1], &work[j1 - 1], ka1);
            }

            // start applying rotations in 1st set from the left
            for (l = ka - 1; l >= kb - k + 1; l--) {
                nrt = (j2 + l - 1) / ka1;
                j1t = j2 - (nrt - 1) * ka1;
                if (nrt > 0)
                    slartv(nrt, &AB[ka1 - l + (j1t - ka1 + l) * ldab], inca,
                           &AB[ka - l + (j1t - ka1 + l) * ldab], inca,
                           &work[n + j1t - 1], &work[j1t - 1], ka1);
            }

            if (wantx) {
                // post-multiply X by product of rotations in 1st set
                for (j = j1; j <= j2; j += ka1) {
                    cblas_srot(nx, &X[0 + j * ldx], 1, &X[0 + (j - 1) * ldx], 1,
                               work[n + j - 1], work[j - 1]);
                }
            }
        }

        if (update) {
            if (i2 > 0 && kbt > 0) {
                // create nonzero element a(i+kbt,i+kbt-ka-1) outside the band
                work[m - kb + i + kbt - 1] = -BB[kbt + i * ldbb] * ra1;
            }
        }

        for (k = kb; k >= 1; k--) {
            if (update) {
                j2 = i + k + 1 - ((2 > k + i0 - m) ? 2 : (k + i0 - m)) * ka1;
            } else {
                j2 = i + k + 1 - ((1 > k + i0 - m) ? 1 : (k + i0 - m)) * ka1;
            }

            // finish applying rotations in 2nd set from the left
            for (l = kb - k; l >= 1; l--) {
                nrt = (j2 + ka + l - 1) / ka1;
                j1t = j2 - (nrt - 1) * ka1;
                if (nrt > 0)
                    slartv(nrt, &AB[ka1 - l + (j1t + l - 1) * ldab], inca,
                           &AB[ka - l + (j1t + l - 1) * ldab], inca,
                           &work[n + m - kb + j1t + ka - 1], &work[m - kb + j1t + ka - 1], ka1);
            }
            nr = (j2 + ka - 1) / ka1;
            j1 = j2 - (nr - 1) * ka1;
            for (j = j1; j <= j2; j += ka1) {
                work[m - kb + j - 1] = work[m - kb + j + ka - 1];
                work[n + m - kb + j - 1] = work[n + m - kb + j + ka - 1];
            }
            for (j = j1; j <= j2; j += ka1) {
                // create nonzero element a(j+ka,j-1) outside the band
                work[m - kb + j - 1] = work[m - kb + j - 1] * AB[ka + (j - 1) * ldab];
                AB[ka + (j - 1) * ldab] = work[n + m - kb + j - 1] * AB[ka + (j - 1) * ldab];
            }
            if (update) {
                if (i + k > ka1 && k <= kbt)
                    work[m - kb + i + k - ka - 1] = work[m - kb + i + k - 1];
            }
        }

        for (k = kb; k >= 1; k--) {
            j2 = i + k + 1 - ((1 > k + i0 - m) ? 1 : (k + i0 - m)) * ka1;
            nr = (j2 + ka - 1) / ka1;
            j1 = j2 - (nr - 1) * ka1;
            if (nr > 0) {
                // generate rotations in 2nd set to annihilate elements outside the band
                slargv(nr, &AB[ka + j1 * ldab], inca, &work[m - kb + j1 - 1], ka1,
                       &work[n + m - kb + j1 - 1], ka1);

                // apply rotations in 2nd set from the right
                for (l = 1; l <= ka - 1; l++) {
                    slartv(nr, &AB[l + j1 * ldab], inca,
                           &AB[l + 1 + (j1 - 1) * ldab], inca,
                           &work[n + m - kb + j1 - 1], &work[m - kb + j1 - 1], ka1);
                }

                // apply rotations in 2nd set from both sides to diagonal blocks
                slar2v(nr, &AB[0 + j1 * ldab], &AB[0 + (j1 - 1) * ldab],
                       &AB[1 + (j1 - 1) * ldab], inca,
                       &work[n + m - kb + j1 - 1], &work[m - kb + j1 - 1], ka1);
            }

            // start applying rotations in 2nd set from the left
            for (l = ka - 1; l >= kb - k + 1; l--) {
                nrt = (j2 + l - 1) / ka1;
                j1t = j2 - (nrt - 1) * ka1;
                if (nrt > 0)
                    slartv(nrt, &AB[ka1 - l + (j1t - ka1 + l) * ldab], inca,
                           &AB[ka - l + (j1t - ka1 + l) * ldab], inca,
                           &work[n + m - kb + j1t - 1], &work[m - kb + j1t - 1], ka1);
            }

            if (wantx) {
                // post-multiply X by product of rotations in 2nd set
                for (j = j1; j <= j2; j += ka1) {
                    cblas_srot(nx, &X[0 + j * ldx], 1, &X[0 + (j - 1) * ldx], 1,
                               work[n + m - kb + j - 1], work[m - kb + j - 1]);
                }
            }
        }

        for (k = 1; k <= kb - 1; k++) {
            j2 = i + k + 1 - ((1 > k + i0 - m + 1) ? 1 : (k + i0 - m + 1)) * ka1;

            // finish applying rotations in 1st set from the left
            for (l = kb - k; l >= 1; l--) {
                nrt = (j2 + l - 1) / ka1;
                j1t = j2 - (nrt - 1) * ka1;
                if (nrt > 0)
                    slartv(nrt, &AB[ka1 - l + (j1t - ka1 + l) * ldab], inca,
                           &AB[ka - l + (j1t - ka1 + l) * ldab], inca,
                           &work[n + j1t - 1], &work[j1t - 1], ka1);
            }
        }

        if (kb > 1) {
            int jmin = (i + kb < m) ? (i + kb) : m;
            for (j = 2; j <= jmin - 2 * ka - 1; j++) {
                work[n + j - 1] = work[n + j + ka - 1];
                work[j - 1] = work[j + ka - 1];
            }
        }
    }

    goto L490;
}
