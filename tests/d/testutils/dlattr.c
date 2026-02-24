/**
 * @file dlattr.c
 * @brief DLATTR generates a triangular test matrix.
 *
 * Port of LAPACK TESTING/LIN/dlattr.f to C.
 *
 * IMAT and UPLO uniquely specify the properties of the test matrix:
 *   IMAT 1-6:  Non-unit triangular generated via DLATMS
 *   IMAT 7:    Unit triangular identity
 *   IMAT 8-10: Non-trivial unit triangular with controlled condition
 *   IMAT 11-19: Pathological test cases for DLATRS
 */

#include <math.h>
#include <stdlib.h>
#include "semicolon_cblas.h"
#include "verify.h"
#include "test_rng.h"

/* External declarations */
/**
 * DLATTR generates a triangular test matrix.
 *
 * @param[in]     imat    An integer key describing which matrix to generate (1-19).
 * @param[in]     uplo    'U' for upper triangular, 'L' for lower triangular.
 * @param[in]     trans   'N' for no transpose, 'T' or 'C' for transpose.
 *                        Used to flip the matrix if transpose will be used.
 * @param[out]    diag    Returns 'N' for non-unit triangular, 'U' for unit triangular.
 * @param[in]     n       The order of the matrix. n >= 0.
 * @param[out]    A       Array (lda, n). The triangular matrix A.
 * @param[in]     lda     The leading dimension of A. lda >= max(1,n).
 * @param[out]    B       Array (n). The right hand side vector (for IMAT > 10).
 * @param[out]    work    Array (3*n). Workspace.
 * @param[out]    info    0 = successful exit, < 0 = illegal argument.
 * @param[in,out] state   RNG state array of 4 uint64_t elements.
 */
void dlattr(const INT imat, const char* uplo, const char* trans, char* diag,
            const INT n, f64* A, const INT lda,
            f64* B, f64* work, INT* info, uint64_t state[static 4])
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const f64 TWO = 2.0;

    f64 unfl, ulp, smlnum, bignum;
    INT upper;
    char type, dist;
    INT kl, ku, mode;
    f64 anorm, cndnum;
    INT i, j, jcount, iy;
    f64 bnorm, bscal, tscal, texp, tleft;
    f64 plus1, plus2, star1, sfac, rexp;
    f64 x, y, z, c, s, ra, rb;

    *info = 0;

    /* Set DIAG based on IMAT */
    if ((imat >= 7 && imat <= 10) || imat == 18) {
        *diag = 'U';
    } else {
        *diag = 'N';
    }

    /* Quick return if n <= 0 */
    if (n <= 0) {
        return;
    }

    /* Machine parameters */
    unfl = dlamch("S");
    ulp = dlamch("E") * dlamch("B");
    smlnum = unfl;
    bignum = (ONE - ulp) / smlnum;

    upper = (uplo[0] == 'U' || uplo[0] == 'u');

    /* Call dlatb4 to set parameters for DLATMS */
    if (upper) {
        dlatb4("DTR", imat, n, n, &type, &kl, &ku, &anorm, &mode, &cndnum, &dist);
    } else {
        dlatb4("DTR", -imat, n, n, &type, &kl, &ku, &anorm, &mode, &cndnum, &dist);
    }

    /* IMAT <= 6: Non-unit triangular matrix */
    if (imat <= 6) {
        char symm[2] = "N";
        symm[0] = type;
        char dstr[2] = "S";
        dstr[0] = dist;
        dlatms(n, n, dstr, symm, B, mode, cndnum, anorm,
               kl, ku, "N", A, lda, work, info, state);
    }
    /* IMAT = 7: Unit triangular identity */
    else if (imat == 7) {
        if (upper) {
            for (j = 0; j < n; j++) {
                for (i = 0; i < j; i++) {
                    A[j * lda + i] = ZERO;
                }
                A[j * lda + j] = (f64)(j + 1);
            }
        } else {
            for (j = 0; j < n; j++) {
                A[j * lda + j] = (f64)(j + 1);
                for (i = j + 1; i < n; i++) {
                    A[j * lda + i] = ZERO;
                }
            }
        }
    }
    /* IMAT 8-10: Non-trivial unit triangular with controlled condition */
    else if (imat <= 10) {
        /* Initialize to identity-like structure with diagonal = j+1 */
        if (upper) {
            for (j = 0; j < n; j++) {
                for (i = 0; i < j; i++) {
                    A[j * lda + i] = ZERO;
                }
                A[j * lda + j] = (f64)(j + 1);
            }
        } else {
            for (j = 0; j < n; j++) {
                A[j * lda + j] = (f64)(j + 1);
                for (i = j + 1; i < n; i++) {
                    A[j * lda + i] = ZERO;
                }
            }
        }

        /* Build the work arrays for superdiagonal structure.
         * The matrix constructed has singular values s, 1, 1, ..., 1, 1/s
         * where s = sqrt(CNDNUM).
         */
        star1 = 0.25;
        sfac = 0.5;
        plus1 = sfac;

        for (j = 0; j < n; j += 2) {
            plus2 = star1 / plus1;
            work[j] = plus1;
            work[n + j] = star1;
            if (j + 1 < n) {
                work[j + 1] = plus2;
                work[n + j + 1] = ZERO;
                plus1 = star1 / plus2;
                /* Generate a random value in (-1, 1) using dlarnv with idist=2 */
                rng_fill(state, 2, 1, &rexp);
                star1 = star1 * pow(sfac, rexp);
                if (rexp < ZERO) {
                    star1 = -pow(sfac, ONE - rexp);
                } else {
                    star1 = pow(sfac, ONE + rexp);
                }
            }
        }

        x = sqrt(cndnum) - ONE / sqrt(cndnum);
        if (n > 2) {
            y = sqrt(TWO / (f64)(n - 2)) * x;
        } else {
            y = ZERO;
        }
        z = x * x;

        if (upper) {
            /* Copy work to superdiagonals:
             * A(2,3), A(3,4), ... gets work[0], work[1], ...
             * A(2,4), A(3,5), ... gets work[n], work[n+1], ...
             * In column-major: A[(j+2)*lda + (j+1)] = work[j] for j=0..n-4
             */
            if (n > 3) {
                for (i = 0; i < n - 3; i++) {
                    A[(i + 2) * lda + (i + 1)] = work[i];
                }
                if (n > 4) {
                    for (i = 0; i < n - 4; i++) {
                        A[(i + 3) * lda + (i + 1)] = work[n + i];
                    }
                }
            }
            /* Set first row and last column to y */
            for (j = 1; j < n - 1; j++) {
                A[j * lda + 0] = y;         /* A(1, j+1) = y in Fortran = A[j*lda + 0] */
                A[(n - 1) * lda + j] = y;   /* A(j+1, n) = y in Fortran */
            }
            A[(n - 1) * lda + 0] = z;       /* A(1, n) = z */
        } else {
            /* Lower triangular */
            if (n > 3) {
                for (i = 0; i < n - 3; i++) {
                    A[(i + 1) * lda + (i + 2)] = work[i];
                }
                if (n > 4) {
                    for (i = 0; i < n - 4; i++) {
                        A[(i + 1) * lda + (i + 3)] = work[n + i];
                    }
                }
            }
            for (j = 1; j < n - 1; j++) {
                A[0 * lda + j] = y;           /* A(j+1, 1) = y */
                A[j * lda + (n - 1)] = y;     /* A(n, j+1) = y */
            }
            A[0 * lda + (n - 1)] = z;         /* A(n, 1) = z */
        }

        /* Fill in zeros using Givens rotations via cblas_drotg and cblas_drot */
        if (upper) {
            for (j = 0; j < n - 1; j++) {
                ra = A[(j + 1) * lda + j];    /* A(j+1, j+2) in Fortran = A[j, j+1] in 0-based */
                rb = TWO;
                cblas_drotg(&ra, &rb, &c, &s);

                /* Multiply by [c s; -s c] on the left for columns j+2:n-1 */
                if (n > j + 2) {
                    cblas_drot(n - j - 2,
                               &A[(j + 2) * lda + j], lda,
                               &A[(j + 2) * lda + (j + 1)], lda, c, s);
                }

                /* Multiply by [-c -s; s -c] on the right for rows 0:j-1 */
                if (j > 0) {
                    cblas_drot(j,
                               &A[(j + 1) * lda + 0], 1,
                               &A[j * lda + 0], 1, -c, -s);
                }

                /* Negate A(j, j+1) */
                A[(j + 1) * lda + j] = -A[(j + 1) * lda + j];
            }
        } else {
            for (j = 0; j < n - 1; j++) {
                ra = A[j * lda + (j + 1)];    /* A(j+2, j+1) in Fortran */
                rb = TWO;
                cblas_drotg(&ra, &rb, &c, &s);

                /* Multiply by [c -s; s c] on the right for rows j+2:n-1 */
                if (n > j + 2) {
                    cblas_drot(n - j - 2,
                               &A[(j + 1) * lda + (j + 2)], 1,
                               &A[j * lda + (j + 2)], 1, c, -s);
                }

                /* Multiply by [-c s; -s -c] on the left for columns 0:j-1 */
                if (j > 0) {
                    cblas_drot(j,
                               &A[0 * lda + j], lda,
                               &A[0 * lda + (j + 1)], lda, -c, s);
                }

                /* Negate A(j+1, j) */
                A[j * lda + (j + 1)] = -A[j * lda + (j + 1)];
            }
        }
    }
    /* IMAT = 11: Well-conditioned with large RHS */
    else if (imat == 11) {
        if (upper) {
            for (j = 0; j < n; j++) {
                rng_fill(state, 2, j + 1, &A[j * lda]);
                A[j * lda + j] = (A[j * lda + j] >= ZERO) ? TWO : -TWO;
            }
        } else {
            for (j = 0; j < n; j++) {
                rng_fill(state, 2, n - j, &A[j * lda + j]);
                A[j * lda + j] = (A[j * lda + j] >= ZERO) ? TWO : -TWO;
            }
        }

        /* Set RHS so largest value is BIGNUM */
        rng_fill(state, 2, n, B);
        iy = cblas_idamax(n, B, 1);
        bnorm = fabs(B[iy]);
        bscal = bignum / fmax(ONE, bnorm);
        cblas_dscal(n, bscal, B, 1);
    }
    /* IMAT = 12: Small diagonal, small off-diagonal (CNORM < 1) */
    else if (imat == 12) {
        rng_fill(state, 2, n, B);
        tscal = ONE / fmax(ONE, (f64)(n - 1));

        if (upper) {
            for (j = 0; j < n; j++) {
                rng_fill(state, 2, j + 1, &A[j * lda]);
                cblas_dscal(j, tscal, &A[j * lda], 1);
                A[j * lda + j] = (A[j * lda + j] >= ZERO) ? ONE : -ONE;
            }
            A[(n - 1) * lda + (n - 1)] = smlnum * A[(n - 1) * lda + (n - 1)];
        } else {
            for (j = 0; j < n; j++) {
                rng_fill(state, 2, n - j, &A[j * lda + j]);
                if (j < n - 1) {
                    cblas_dscal(n - j - 1, tscal, &A[j * lda + j + 1], 1);
                }
                A[j * lda + j] = (A[j * lda + j] >= ZERO) ? ONE : -ONE;
            }
            A[0] = smlnum * A[0];
        }
    }
    /* IMAT = 13: Small diagonal, O(1) off-diagonal (CNORM > 1) */
    else if (imat == 13) {
        rng_fill(state, 2, n, B);

        if (upper) {
            for (j = 0; j < n; j++) {
                rng_fill(state, 2, j + 1, &A[j * lda]);
                A[j * lda + j] = (A[j * lda + j] >= ZERO) ? ONE : -ONE;
            }
            A[(n - 1) * lda + (n - 1)] = smlnum * A[(n - 1) * lda + (n - 1)];
        } else {
            for (j = 0; j < n; j++) {
                rng_fill(state, 2, n - j, &A[j * lda + j]);
                A[j * lda + j] = (A[j * lda + j] >= ZERO) ? ONE : -ONE;
            }
            A[0] = smlnum * A[0];
        }
    }
    /* IMAT = 14: Diagonal with small entries causing underflow */
    else if (imat == 14) {
        if (upper) {
            jcount = 1;
            for (j = n - 1; j >= 0; j--) {
                for (i = 0; i < j; i++) {
                    A[j * lda + i] = ZERO;
                }
                if (jcount <= 2) {
                    A[j * lda + j] = smlnum;
                } else {
                    A[j * lda + j] = ONE;
                }
                jcount++;
                if (jcount > 4) jcount = 1;
            }
        } else {
            jcount = 1;
            for (j = 0; j < n; j++) {
                for (i = j + 1; i < n; i++) {
                    A[j * lda + i] = ZERO;
                }
                if (jcount <= 2) {
                    A[j * lda + j] = smlnum;
                } else {
                    A[j * lda + j] = ONE;
                }
                jcount++;
                if (jcount > 4) jcount = 1;
            }
        }

        /* Set RHS alternately zero and small */
        if (upper) {
            B[0] = ZERO;
            for (i = n - 1; i >= 1; i -= 2) {
                B[i] = ZERO;
                B[i - 1] = smlnum;
            }
        } else {
            B[n - 1] = ZERO;
            for (i = 0; i < n - 1; i += 2) {
                B[i] = ZERO;
                B[i + 1] = smlnum;
            }
        }
    }
    /* IMAT = 15: Bidiagonal with small diagonal causing gradual overflow */
    else if (imat == 15) {
        texp = ONE / fmax(ONE, (f64)(n - 1));
        tscal = pow(smlnum, texp);
        rng_fill(state, 2, n, B);

        if (upper) {
            for (j = 0; j < n; j++) {
                for (i = 0; i < j - 1; i++) {
                    A[j * lda + i] = ZERO;
                }
                if (j > 0) {
                    A[j * lda + (j - 1)] = -ONE;
                }
                A[j * lda + j] = tscal;
            }
            B[n - 1] = ONE;
        } else {
            for (j = 0; j < n; j++) {
                for (i = j + 2; i < n; i++) {
                    A[j * lda + i] = ZERO;
                }
                if (j < n - 1) {
                    A[j * lda + (j + 1)] = -ONE;
                }
                A[j * lda + j] = tscal;
            }
            B[0] = ONE;
        }
    }
    /* IMAT = 16: One zero diagonal element */
    else if (imat == 16) {
        iy = n / 2;

        if (upper) {
            for (j = 0; j < n; j++) {
                rng_fill(state, 2, j + 1, &A[j * lda]);
                if (j != iy) {
                    A[j * lda + j] = (A[j * lda + j] >= ZERO) ? TWO : -TWO;
                } else {
                    A[j * lda + j] = ZERO;
                }
            }
        } else {
            for (j = 0; j < n; j++) {
                rng_fill(state, 2, n - j, &A[j * lda + j]);
                if (j != iy) {
                    A[j * lda + j] = (A[j * lda + j] >= ZERO) ? TWO : -TWO;
                } else {
                    A[j * lda + j] = ZERO;
                }
            }
        }
        rng_fill(state, 2, n, B);
        cblas_dscal(n, TWO, B, 1);
    }
    /* IMAT = 17: Large off-diagonal causing overflow */
    else if (imat == 17) {
        tscal = unfl / ulp;
        tscal = (ONE - ulp) / tscal;

        /* Initialize to zero */
        for (j = 0; j < n; j++) {
            for (i = 0; i < n; i++) {
                A[j * lda + i] = ZERO;
            }
        }

        texp = ONE;
        if (upper) {
            for (j = n - 1; j >= 1; j -= 2) {
                A[j * lda + 0] = -tscal / (f64)(n + 1);
                A[j * lda + j] = ONE;
                B[j] = texp * (ONE - ulp);
                A[(j - 1) * lda + 0] = -(tscal / (f64)(n + 1)) / (f64)(n + 2);
                A[(j - 1) * lda + (j - 1)] = ONE;
                B[j - 1] = texp * (f64)(n * n + n - 1);
                texp = texp * TWO;
            }
            B[0] = ((f64)(n + 1) / (f64)(n + 2)) * tscal;
        } else {
            for (j = 0; j < n - 1; j += 2) {
                A[j * lda + (n - 1)] = -tscal / (f64)(n + 1);
                A[j * lda + j] = ONE;
                B[j] = texp * (ONE - ulp);
                A[(j + 1) * lda + (n - 1)] = -(tscal / (f64)(n + 1)) / (f64)(n + 2);
                A[(j + 1) * lda + (j + 1)] = ONE;
                B[j + 1] = texp * (f64)(n * n + n - 1);
                texp = texp * TWO;
            }
            B[n - 1] = ((f64)(n + 1) / (f64)(n + 2)) * tscal;
        }
    }
    /* IMAT = 18: Unit triangular with large RHS */
    else if (imat == 18) {
        if (upper) {
            for (j = 0; j < n; j++) {
                if (j > 0) {
                    rng_fill(state, 2, j, &A[j * lda]);
                }
                A[j * lda + j] = ZERO;
            }
        } else {
            for (j = 0; j < n; j++) {
                if (j < n - 1) {
                    rng_fill(state, 2, n - j - 1, &A[j * lda + j + 1]);
                }
                A[j * lda + j] = ZERO;
            }
        }

        /* Set RHS so largest value is BIGNUM */
        rng_fill(state, 2, n, B);
        iy = cblas_idamax(n, B, 1);
        bnorm = fabs(B[iy]);
        bscal = bignum / fmax(ONE, bnorm);
        cblas_dscal(n, bscal, B, 1);
    }
    /* IMAT = 19: Large elements causing column norm to exceed BIGNUM */
    else if (imat == 19) {
        tleft = bignum / fmax(ONE, (f64)(n - 1));
        tscal = bignum * ((f64)(n - 1) / fmax(ONE, (f64)n));

        if (upper) {
            for (j = 0; j < n; j++) {
                rng_fill(state, 2, j + 1, &A[j * lda]);
                for (i = 0; i <= j; i++) {
                    f64 sign = (A[j * lda + i] >= ZERO) ? ONE : -ONE;
                    A[j * lda + i] = sign * tleft + tscal * A[j * lda + i];
                }
            }
        } else {
            for (j = 0; j < n; j++) {
                rng_fill(state, 2, n - j, &A[j * lda + j]);
                for (i = j; i < n; i++) {
                    f64 sign = (A[j * lda + i] >= ZERO) ? ONE : -ONE;
                    A[j * lda + i] = sign * tleft + tscal * A[j * lda + i];
                }
            }
        }
        rng_fill(state, 2, n, B);
        cblas_dscal(n, TWO, B, 1);
    }

    /* Flip the matrix if the transpose will be used.
     * Fortran: DSWAP(N-2*J+1, A(J,J), LDA, A(J+1,N-J+1), -1) for upper
     *          DSWAP(N-2*J+1, A(J,J), 1, A(N-J+1,J+1), -LDA) for lower
     * J runs 1..N/2 in Fortran, j runs 0..n/2-1 in C.
     * Length N-2*J+1 at J=k becomes n-2*(k+1)+1 = n-2*k-1 at j=k. */
    if (trans[0] != 'N' && trans[0] != 'n') {
        if (upper) {
            for (j = 0; j < n / 2; j++) {
                INT len = n - 2 * j - 1;
                for (INT k = 0; k < len; k++) {
                    /* Swap A[j, j+k] (column j) with A[j+1+k, n-1-j-k] (row j+1+k, descending col) */
                    f64 temp = A[j * lda + (j + k)];
                    A[j * lda + (j + k)] = A[(n - 2 - j - k) * lda + (j + 1 + k)];
                    A[(n - 2 - j - k) * lda + (j + 1 + k)] = temp;
                }
            }
        } else {
            for (j = 0; j < n / 2; j++) {
                INT len = n - 2 * j - 1;
                for (INT k = 0; k < len; k++) {
                    /* Swap A[j+k, j] (row j) with A[n-2-j-k, j+1+k] (descending row, column j+1+k) */
                    f64 temp = A[(j + k) * lda + j];
                    A[(j + k) * lda + j] = A[(j + 1 + k) * lda + (n - 2 - j - k)];
                    A[(j + 1 + k) * lda + (n - 2 - j - k)] = temp;
                }
            }
        }
    }
}
