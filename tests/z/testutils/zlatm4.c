/**
 * @file zlatm4.c
 * @brief ZLATM4 generates basic square matrices for eigenvalue testing.
 */

#include <complex.h>
#include <math.h>
#include "verify.h"
#include "test_rng.h"

/**
 * ZLATM4 generates basic square matrices, which may later be
 * multiplied by others in order to produce test matrices. It is
 * intended mainly to be used to test the generalized eigenvalue
 * routines.
 *
 * It first generates the diagonal and (possibly) subdiagonal,
 * according to the value of ITYPE, NZ1, NZ2, RSIGN, AMAGN, and RCOND.
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
 *                    = 10: random numbers with distribution IDIST (see ZLARND.)
 *
 * @param[in] n       The order of the matrix.
 * @param[in] nz1     If abs(itype) > 3, then the first nz1 diagonal entries will be zero.
 * @param[in] nz2     If abs(itype) > 3, then the last nz2 diagonal entries will be zero.
 * @param[in] rsign   = 1: The diagonal and subdiagonal entries will be multiplied
 *                         by random numbers of magnitude 1.
 *                    = 0: The diagonal and subdiagonal entries will be left as they
 *                         are (usually non-negative real.)
 * @param[in] amagn   The diagonal and subdiagonal entries will be multiplied by AMAGN.
 * @param[in] rcond   If abs(itype) > 4, then the smallest diagonal entry will be RCOND.
 *                    RCOND must be between 0 and 1.
 * @param[in] triang  The entries above the diagonal will be random numbers with
 *                    magnitude bounded by TRIANG.
 * @param[in] idist   Specifies the type of distribution to be used to generate a
 *                    random matrix: 1=UNIFORM(0,1), 2=UNIFORM(-1,1), 3=NORMAL(0,1),
 *                    4=uniform in DISK(0,1).
 * @param[out] A      Array to be computed, dimension (lda, n).
 * @param[in] lda     Leading dimension of A. Must be at least 1 and at least n.
 */
void zlatm4(const INT itype, const INT n, const INT nz1, const INT nz2,
            const INT rsign, const f64 amagn, const f64 rcond,
            const f64 triang, const INT idist,
            c128* A, const INT lda,
            uint64_t state[static 4])
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const c128 CONE = CMPLX(1.0, 0.0);

    INT i, isdb, isde, jc, jd, jr, k, kbeg, kend, klen;
    f64 alpha;
    c128 ctemp;
    INT abstype;

    if (n <= 0) return;

    /* Initialize A to zero */
    for (jc = 0; jc < n; jc++) {
        for (jr = 0; jr < n; jr++) {
            A[jr + jc * lda] = CMPLX(0.0, 0.0);
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
                    A[jd + jd * lda] = CONE;
                }
                break;

            case 2:
                /* Transposed Jordan block */
                for (jd = 0; jd < n - 1; jd++) {
                    A[(jd + 1) + jd * lda] = CONE;
                }
                isdb = 1;
                isde = n - 1;
                break;

            case 3:
                /* Transposed Jordan block, followed by identity */
                k = (n - 1) / 2;
                for (jd = 0; jd < k; jd++) {
                    A[(jd + 1) + jd * lda] = CONE;
                }
                isdb = 1;
                isde = k;
                for (jd = k + 1; jd < 2 * k + 1; jd++) {
                    A[jd + jd * lda] = CONE;
                }
                break;

            case 4:
                /* 1, ..., k */
                for (jd = kbeg - 1; jd < kend; jd++) {
                    A[jd + jd * lda] = CMPLX((f64)(jd - nz1 + 1), 0.0);
                }
                break;

            case 5:
                /* One large D value: 1, RCOND, ..., RCOND */
                for (jd = kbeg; jd < kend; jd++) {
                    A[jd + jd * lda] = CMPLX(rcond, 0.0);
                }
                A[(kbeg - 1) + (kbeg - 1) * lda] = CONE;
                break;

            case 6:
                /* One small D value: 1, ..., 1, RCOND */
                for (jd = kbeg - 1; jd < kend - 1; jd++) {
                    A[jd + jd * lda] = CONE;
                }
                A[(kend - 1) + (kend - 1) * lda] = CMPLX(rcond, 0.0);
                break;

            case 7:
                /* Exponentially distributed D values */
                A[(kbeg - 1) + (kbeg - 1) * lda] = CONE;
                if (klen > 1) {
                    alpha = pow(rcond, ONE / (f64)(klen - 1));
                    for (i = 1; i < klen; i++) {
                        A[(nz1 + i) + (nz1 + i) * lda] = CMPLX(pow(alpha, (f64)i), 0.0);
                    }
                }
                break;

            case 8:
                /* Arithmetically distributed D values */
                A[(kbeg - 1) + (kbeg - 1) * lda] = CONE;
                if (klen > 1) {
                    alpha = (ONE - rcond) / (f64)(klen - 1);
                    for (i = 1; i < klen; i++) {
                        A[(nz1 + i) + (nz1 + i) * lda] = CMPLX((f64)(klen - 1 - i) * alpha + rcond, 0.0);
                    }
                }
                break;

            case 9:
                /* Randomly distributed D values on (RCOND, 1) */
                alpha = log(rcond);
                for (jd = kbeg - 1; jd < kend; jd++) {
                    A[jd + jd * lda] = exp(alpha * rng_uniform(state));
                }
                break;

            case 10:
                /* Randomly distributed D values from DIST */
                for (jd = kbeg - 1; jd < kend; jd++) {
                    A[jd + jd * lda] = zlarnd_rng(idist, state);
                }
                break;
        }

        /* Scale by AMAGN */
        for (jd = kbeg - 1; jd < kend; jd++) {
            A[jd + jd * lda] = amagn * creal(A[jd + jd * lda]);
        }
        for (jd = isdb - 1; jd < isde; jd++) {
            A[(jd + 1) + jd * lda] = amagn * creal(A[(jd + 1) + jd * lda]);
        }

        /* If RSIGN, assign random signs to diagonal and subdiagonal */
        if (rsign) {
            for (jd = kbeg - 1; jd < kend; jd++) {
                if (creal(A[jd + jd * lda]) != ZERO) {
                    ctemp = zlarnd_rng(3, state);
                    ctemp = ctemp / cabs(ctemp);
                    A[jd + jd * lda] = ctemp * creal(A[jd + jd * lda]);
                }
            }
            for (jd = isdb - 1; jd < isde; jd++) {
                if (creal(A[(jd + 1) + jd * lda]) != ZERO) {
                    ctemp = zlarnd_rng(3, state);
                    ctemp = ctemp / cabs(ctemp);
                    A[(jd + 1) + jd * lda] = ctemp * creal(A[(jd + 1) + jd * lda]);
                }
            }
        }

        /* Reverse if ITYPE < 0 */
        if (itype < 0) {
            for (jd = kbeg - 1; jd < (kbeg + kend - 1) / 2; jd++) {
                ctemp = A[jd + jd * lda];
                A[jd + jd * lda] = A[(kbeg + kend - 2 - jd) + (kbeg + kend - 2 - jd) * lda];
                A[(kbeg + kend - 2 - jd) + (kbeg + kend - 2 - jd) * lda] = ctemp;
            }
            for (jd = 0; jd < (n - 1) / 2; jd++) {
                ctemp = A[(jd + 1) + jd * lda];
                A[(jd + 1) + jd * lda] = A[(n - 1 - jd) + (n - 2 - jd) * lda];
                A[(n - 1 - jd) + (n - 2 - jd) * lda] = ctemp;
            }
        }
    }

    /* Fill in upper triangle */
    if (triang != ZERO) {
        for (jc = 1; jc < n; jc++) {
            for (jr = 0; jr < jc; jr++) {
                A[jr + jc * lda] = triang * zlarnd_rng(idist, state);
            }
        }
    }
}
