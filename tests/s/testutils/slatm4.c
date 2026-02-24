/**
 * @file slatm4.c
 * @brief SLATM4 generates basic square matrices for eigenvalue testing.
 */

#include <math.h>
#include "verify.h"
#include "test_rng.h"

/**
 * SLATM4 generates basic square matrices, which may later be
 * multiplied by others in order to produce test matrices. It is
 * intended mainly to be used to test the generalized eigenvalue
 * routines.
 *
 * It first generates the diagonal and (possibly) subdiagonal,
 * according to the value of ITYPE, NZ1, NZ2, ISIGN, AMAGN, and RCOND.
 * It then fills in the upper triangle with random numbers, if TRIANG is
 * non-zero.
 *
 * @param[in] itype   The "type" of matrix on the diagonal and sub-diagonal.
 *                    If itype < 0, then type abs(itype) is generated and then
 *                    swapped end for end (A(i,j) := A'(n-j,n-i)).
 *
 *                    Special types:
 *                    = 0:  the zero matrix.
 *                    = 1:  the identity.
 *                    = 2:  a transposed Jordan block.
 *                    = 3:  If n is odd, then a k+1 x k+1 transposed Jordan block
 *                          followed by a k x k identity block, where k=(n-1)/2.
 *                          If n is even, then k=(n-2)/2, and a zero diagonal entry
 *                          is tacked onto the end.
 *
 *                    Diagonal types. The diagonal consists of nz1 zeros, then
 *                    k=n-nz1-nz2 nonzeros. The subdiagonal is zero.
 *                    = 4:  1, ..., k
 *                    = 5:  1, RCOND, ..., RCOND
 *                    = 6:  1, ..., 1, RCOND
 *                    = 7:  1, a, a^2, ..., a^(k-1)=RCOND
 *                    = 8:  1, 1-d, 1-2*d, ..., 1-(k-1)*d=RCOND
 *                    = 9:  random numbers chosen from (RCOND,1)
 *                    = 10: random numbers with distribution IDIST
 *
 * @param[in] n       The order of the matrix.
 * @param[in] nz1     If abs(itype) > 3, then the first nz1 diagonal entries will be zero.
 * @param[in] nz2     If abs(itype) > 3, then the last nz2 diagonal entries will be zero.
 * @param[in] isign   = 0: The sign of the diagonal and subdiagonal entries will be unchanged.
 *                    = 1: The diagonal and subdiagonal entries will have their
 *                         sign changed at random.
 *                    = 2: If itype is 2 or 3, then the same as isign=1.
 *                         Otherwise, with probability 0.5, odd-even pairs of
 *                         diagonal entries A(2*j-1,2*j-1), A(2*j,2*j) will be
 *                         converted to a 2x2 block by pre- and post-multiplying
 *                         by distinct random orthogonal rotations.
 * @param[in] amagn   The diagonal and subdiagonal entries will be multiplied by AMAGN.
 * @param[in] rcond   If abs(itype) > 4, then the smallest diagonal entry will be RCOND.
 *                    RCOND must be between 0 and 1.
 * @param[in] triang  The entries above the diagonal will be random numbers with
 *                    magnitude bounded by TRIANG.
 * @param[in] idist   Specifies the type of distribution to be used to generate a
 *                    random matrix: 1=UNIFORM(0,1), 2=UNIFORM(-1,1), 3=NORMAL(0,1)
 * @param[out] A      Array to be computed, dimension (lda, n).
 * @param[in] lda     Leading dimension of A. Must be at least 1 and at least n.
 */
void slatm4(const INT itype, const INT n, const INT nz1, const INT nz2,
            const INT isign, const f32 amagn, const f32 rcond,
            const f32 triang, const INT idist,
            f32* A, const INT lda,
            uint64_t state[static 4])
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;
    const f32 TWO = 2.0f;
    const f32 HALF = 0.5f;

    INT i, ioff, isdb, isde, jc, jd, jr, k, kbeg, kend, klen;
    f32 alpha, cl, cr, safmin, sl, sr, sv1, sv2, temp;
    INT abstype;

    if (n <= 0) return;

    /* Initialize A to zero */
    for (jc = 0; jc < n; jc++) {
        for (jr = 0; jr < n; jr++) {
            A[jr + jc * lda] = ZERO;
        }
    }

    /* Compute diagonal and subdiagonal according to ITYPE, NZ1, NZ2, and RCOND */
    if (itype != 0) {
        abstype = (itype < 0) ? -itype : itype;

        if (abstype >= 4) {
            kbeg = (1 > nz1 + 1) ? 1 : nz1 + 1;
            kbeg = (kbeg < n) ? kbeg : n;
            kend = (n - nz2 > kbeg) ? n - nz2 : kbeg;
            kend = (kend < n) ? kend : n;
            klen = kend - kbeg + 1;
        } else {
            kbeg = 1;
            kend = n;
            klen = n;
        }

        isdb = 1;
        isde = 0;

        switch (abstype) {
            case 1:
                /* Identity */
                for (jd = 0; jd < n; jd++) {
                    A[jd + jd * lda] = ONE;
                }
                break;

            case 2:
                /* Transposed Jordan block */
                for (jd = 0; jd < n - 1; jd++) {
                    A[(jd + 1) + jd * lda] = ONE;
                }
                isdb = 1;
                isde = n - 1;
                break;

            case 3:
                /* Transposed Jordan block, followed by identity */
                k = (n - 1) / 2;
                for (jd = 0; jd < k; jd++) {
                    A[(jd + 1) + jd * lda] = ONE;
                }
                isdb = 1;
                isde = k;
                for (jd = k + 1; jd < 2 * k + 1; jd++) {
                    A[jd + jd * lda] = ONE;
                }
                break;

            case 4:
                /* 1, ..., k */
                for (jd = kbeg - 1; jd < kend; jd++) {
                    A[jd + jd * lda] = (f32)(jd - nz1 + 1);
                }
                break;

            case 5:
                /* One large D value: 1, RCOND, ..., RCOND */
                for (jd = kbeg; jd < kend; jd++) {
                    A[jd + jd * lda] = rcond;
                }
                A[(kbeg - 1) + (kbeg - 1) * lda] = ONE;
                break;

            case 6:
                /* One small D value: 1, ..., 1, RCOND */
                for (jd = kbeg - 1; jd < kend - 1; jd++) {
                    A[jd + jd * lda] = ONE;
                }
                A[(kend - 1) + (kend - 1) * lda] = rcond;
                break;

            case 7:
                /* Exponentially distributed D values */
                A[(kbeg - 1) + (kbeg - 1) * lda] = ONE;
                if (klen > 1) {
                    alpha = powf(rcond, ONE / (f32)(klen - 1));
                    for (i = 1; i < klen; i++) {
                        A[(nz1 + i) + (nz1 + i) * lda] = powf(alpha, (f32)i);
                    }
                }
                break;

            case 8:
                /* Arithmetically distributed D values */
                A[(kbeg - 1) + (kbeg - 1) * lda] = ONE;
                if (klen > 1) {
                    alpha = (ONE - rcond) / (f32)(klen - 1);
                    for (i = 1; i < klen; i++) {
                        A[(nz1 + i) + (nz1 + i) * lda] = (f32)(klen - 1 - i) * alpha + rcond;
                    }
                }
                break;

            case 9:
                /* Randomly distributed D values on (RCOND, 1) */
                alpha = logf(rcond);
                for (jd = kbeg - 1; jd < kend; jd++) {
                    A[jd + jd * lda] = expf(alpha * rng_uniform_f32(state));
                }
                break;

            case 10:
                /* Randomly distributed D values from DIST */
                for (jd = kbeg - 1; jd < kend; jd++) {
                    A[jd + jd * lda] = rng_dist_f32(state, idist);
                }
                break;
        }

        /* Scale by AMAGN */
        for (jd = kbeg - 1; jd < kend; jd++) {
            A[jd + jd * lda] = amagn * A[jd + jd * lda];
        }
        for (jd = isdb - 1; jd < isde; jd++) {
            A[(jd + 1) + jd * lda] = amagn * A[(jd + 1) + jd * lda];
        }

        /* If ISIGN = 1 or 2, assign random signs to diagonal and subdiagonal */
        if (isign > 0) {
            for (jd = kbeg - 1; jd < kend; jd++) {
                if (A[jd + jd * lda] != ZERO) {
                    if (rng_uniform_f32(state) > HALF)
                        A[jd + jd * lda] = -A[jd + jd * lda];
                }
            }
            for (jd = isdb - 1; jd < isde; jd++) {
                if (A[(jd + 1) + jd * lda] != ZERO) {
                    if (rng_uniform_f32(state) > HALF)
                        A[(jd + 1) + jd * lda] = -A[(jd + 1) + jd * lda];
                }
            }
        }

        /* Reverse if ITYPE < 0 */
        if (itype < 0) {
            for (jd = kbeg - 1; jd < (kbeg + kend - 1) / 2; jd++) {
                temp = A[jd + jd * lda];
                A[jd + jd * lda] = A[(kbeg + kend - 2 - jd) + (kbeg + kend - 2 - jd) * lda];
                A[(kbeg + kend - 2 - jd) + (kbeg + kend - 2 - jd) * lda] = temp;
            }
            for (jd = 0; jd < (n - 1) / 2; jd++) {
                temp = A[(jd + 1) + jd * lda];
                A[(jd + 1) + jd * lda] = A[(n - 1 - jd) + (n - 2 - jd) * lda];
                A[(n - 1 - jd) + (n - 2 - jd) * lda] = temp;
            }
        }

        /* If ISIGN = 2, and no subdiagonals already, then apply
           random rotations to make 2x2 blocks. */
        if (isign == 2 && abstype != 2 && abstype != 3) {
            safmin = slamch("S");
            for (jd = kbeg - 1; jd < kend - 1; jd += 2) {
                if (rng_uniform_f32(state) > HALF) {
                    /* Rotation on left */
                    cl = TWO * rng_uniform_f32(state) - ONE;
                    sl = TWO * rng_uniform_f32(state) - ONE;
                    temp = ONE / fmaxf(safmin, sqrtf(cl * cl + sl * sl));
                    cl = cl * temp;
                    sl = sl * temp;

                    /* Rotation on right */
                    cr = TWO * rng_uniform_f32(state) - ONE;
                    sr = TWO * rng_uniform_f32(state) - ONE;
                    temp = ONE / fmaxf(safmin, sqrtf(cr * cr + sr * sr));
                    cr = cr * temp;
                    sr = sr * temp;

                    /* Apply */
                    sv1 = A[jd + jd * lda];
                    sv2 = A[(jd + 1) + (jd + 1) * lda];
                    A[jd + jd * lda] = cl * cr * sv1 + sl * sr * sv2;
                    A[(jd + 1) + jd * lda] = -sl * cr * sv1 + cl * sr * sv2;
                    A[jd + (jd + 1) * lda] = -cl * sr * sv1 + sl * cr * sv2;
                    A[(jd + 1) + (jd + 1) * lda] = sl * sr * sv1 + cl * cr * sv2;
                }
            }
        }
    }

    /* Fill in upper triangle (except for 2x2 blocks) */
    if (triang != ZERO) {
        if (isign != 2 || (itype < 0 ? -itype : itype) == 2 || (itype < 0 ? -itype : itype) == 3) {
            ioff = 1;
        } else {
            ioff = 2;
            for (jr = 0; jr < n - 1; jr++) {
                if (A[(jr + 1) + jr * lda] == ZERO)
                    A[jr + (jr + 1) * lda] = triang * rng_dist_f32(state, idist);
            }
        }

        for (jc = 1; jc < n; jc++) {
            for (jr = 0; jr < jc - ioff + 1; jr++) {
                A[jr + jc * lda] = triang * rng_dist_f32(state, idist);
            }
        }
    }
}
