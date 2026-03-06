/**
 * @file clattr.c
 * @brief CLATTR generates a triangular test matrix in 2-dimensional storage.
 */

#include <math.h>
#include <stdlib.h>
#include "semicolon_cblas.h"
#include "verify.h"
#include "test_rng.h"

void clattr(const INT imat, const char* uplo, const char* trans, char* diag,
            const INT n, c64* A, const INT lda,
            c64* B, c64* work, f32* rwork, INT* info,
            uint64_t state[static 4])
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;
    const f32 TWO = 2.0f;

    f32 unfl, ulp, smlnum, bignum;
    INT upper;
    char type, dist;
    INT kl, ku, mode;
    f32 anorm, cndnum;
    INT i, j, jcount, iy;
    f32 bnorm, bscal, tscal, texp, tleft;
    c64 plus1, plus2, star1;
    f32 sfac, rexp;
    f32 x, y, z;
    c64 ra, rb, s;
    f32 c;

    *info = 0;

    if ((imat >= 7 && imat <= 10) || imat == 18) {
        *diag = 'U';
    } else {
        *diag = 'N';
    }

    /* Quick return if n <= 0 */
    if (n <= 0)
        return;

    unfl = slamch("S");
    ulp = slamch("E") * slamch("B");
    smlnum = unfl;
    bignum = (ONE - ulp) / smlnum;

    upper = (uplo[0] == 'U' || uplo[0] == 'u');

    /* Call clatb4 to set parameters for CLATMS */
    if (upper) {
        clatb4("CTR", imat, n, n, &type, &kl, &ku, &anorm, &mode, &cndnum, &dist);
    } else {
        clatb4("CTR", -imat, n, n, &type, &kl, &ku, &anorm, &mode, &cndnum, &dist);
    }

    /* IMAT <= 6: Non-unit triangular matrix */
    if (imat <= 6) {
        char symm[2] = "N";
        symm[0] = type;
        char dstr[2] = "S";
        dstr[0] = dist;
        clatms(n, n, dstr, symm, rwork, mode, cndnum, anorm,
               kl, ku, "N", A, lda, work, info, state);
    }
    /* IMAT = 7: Matrix is the identity */
    else if (imat == 7) {
        if (upper) {
            for (j = 0; j < n; j++) {
                for (i = 0; i < j; i++) {
                    A[j * lda + i] = ZERO;
                }
                A[j * lda + j] = (f32)(j + 1);
            }
        } else {
            for (j = 0; j < n; j++) {
                A[j * lda + j] = (f32)(j + 1);
                for (i = j + 1; i < n; i++) {
                    A[j * lda + i] = ZERO;
                }
            }
        }
    }
    /* IMAT 8-10: Non-trivial unit triangular with controlled condition */
    else if (imat <= 10) {
        if (upper) {
            for (j = 0; j < n; j++) {
                for (i = 0; i < j; i++) {
                    A[j * lda + i] = ZERO;
                }
                A[j * lda + j] = (f32)(j + 1);
            }
        } else {
            for (j = 0; j < n; j++) {
                A[j * lda + j] = (f32)(j + 1);
                for (i = j + 1; i < n; i++) {
                    A[j * lda + i] = ZERO;
                }
            }
        }

        /*
         * Since the trace of a unit triangular matrix is 1, the product
         * of its singular values must be 1.  Let s = sqrt(CNDNUM),
         * x = sqrt(s) - 1/sqrt(s), y = sqrt(2/(n-2))*x, and z = x**2.
         * The following triangular matrix has singular values s, 1, 1,
         * ..., 1, 1/s:
         *
         * 1  y  y  y  ...  y  y  z
         *    1  0  0  ...  0  0  y
         *       1  0  ...  0  0  y
         *          .  ...  .  .  .
         *              .   .  .  .
         *                  1  0  y
         *                     1  y
         *                        1
         *
         * To fill in the zeros, we first multiply by a matrix with small
         * condition number of the form
         *
         * 1  0  0  0  0  ...
         *    1  +  *  0  0  ...
         *       1  +  0  0  0
         *          1  +  *  0  0
         *             1  +  0  0
         *                ...
         *                   1  +  0
         *                      1  0
         *                         1
         *
         * Each element marked with a '*' is formed by taking the product
         * of the adjacent elements marked with '+'.  The '*'s can be
         * chosen freely, and the '+'s are chosen so that the inverse of
         * T will have elements of the same magnitude as T.  If the *'s in
         * both T and inv(T) have small magnitude, T is well conditioned.
         * The two offdiagonals of T are stored in WORK.
         *
         * The product of these two matrices has the form
         *
         * 1  y  y  y  y  y  .  y  y  z
         *    1  +  *  0  0  .  0  0  y
         *       1  +  0  0  .  0  0  y
         *          1  +  *  .  .  .  .
         *             1  +  .  .  .  .
         *                .  .  .  .  .
         *                   .  .  .  .
         *                      1  +  y
         *                         1  y
         *                            1
         *
         * Now we multiply by Givens rotations, using the fact that
         *
         *       [  c   s ] [  1   w ] [ -c  -s ] =  [  1  -w ]
         *       [ -s   c ] [  0   1 ] [  s  -c ]    [  0   1 ]
         * and
         *       [ -c  -s ] [  1   0 ] [  c   s ] =  [  1   0 ]
         *       [  s  -c ] [  w   1 ] [ -s   c ]    [ -w   1 ]
         *
         * where c = w / sqrt(w**2+4) and s = 2 / sqrt(w**2+4).
         */
        star1 = 0.25f * clarnd_rng(5, state);
        sfac = 0.5f;
        plus1 = sfac * clarnd_rng(5, state);
        for (j = 0; j < n; j += 2) {
            plus2 = star1 / plus1;
            work[j] = plus1;
            work[n + j] = star1;
            if (j + 1 < n) {
                work[j + 1] = plus2;
                work[n + j + 1] = ZERO;
                plus1 = star1 / plus2;
                rng_fill_f32(state, 2, 1, &rexp);
                if (rexp < ZERO) {
                    star1 = -powf(sfac, ONE - rexp) * clarnd_rng(5, state);
                } else {
                    star1 = powf(sfac, ONE + rexp) * clarnd_rng(5, state);
                }
            }
        }

        x = sqrtf(cndnum) - ONE / sqrtf(cndnum);
        if (n > 2) {
            y = sqrtf(TWO / (f32)(n - 2)) * x;
        } else {
            y = ZERO;
        }
        z = x * x;

        if (upper) {
            /* Copy work to superdiagonals:
             * Fortran: ZCOPY(N-3, WORK, 1, A(2, 3), LDA+1)
             * A(2,3) in 0-based: A[row=1, col=2] = A[2*lda + 1]
             */
            if (n > 3) {
                cblas_ccopy(n - 3, work, 1, &A[2 * lda + 1], lda + 1);
                if (n > 4)
                    cblas_ccopy(n - 4, &work[n], 1, &A[3 * lda + 1], lda + 1);
            }
            for (j = 1; j < n - 1; j++) {
                A[j * lda + 0] = y;
                A[(n - 1) * lda + j] = y;
            }
            A[(n - 1) * lda + 0] = z;
        } else {
            /* Fortran: ZCOPY(N-3, WORK, 1, A(3, 2), LDA+1)
             * A(3,2) in 0-based: A[row=2, col=1] = A[1*lda + 2]
             */
            if (n > 3) {
                cblas_ccopy(n - 3, work, 1, &A[1 * lda + 2], lda + 1);
                if (n > 4)
                    cblas_ccopy(n - 4, &work[n], 1, &A[1 * lda + 3], lda + 1);
            }
            for (j = 1; j < n - 1; j++) {
                A[0 * lda + j] = y;
                A[j * lda + (n - 1)] = y;
            }
            A[0 * lda + (n - 1)] = z;
        }

        /* Fill in zeros using Givens rotations */
        if (upper) {
            for (j = 0; j < n - 1; j++) {
                ra = A[(j + 1) * lda + j];
                rb = CMPLXF(2.0f, 0.0f);
                cblas_crotg(&ra, &rb, &c, &s);

                /* Multiply by [ c  s; -conjg(s)  c] on the left */
                if (n > j + 2) {
                    crot(n - j - 2, &A[(j + 2) * lda + j], lda,
                         &A[(j + 2) * lda + (j + 1)], lda, c, s);
                }

                /* Multiply by [-c -s;  conjg(s) -c] on the right */
                if (j > 0) {
                    c64 neg_s = -s;
                    crot(j, &A[(j + 1) * lda + 0], 1,
                         &A[j * lda + 0], 1, -c, neg_s);
                }

                /* Negate A(j, j+1) */
                A[(j + 1) * lda + j] = -A[(j + 1) * lda + j];
            }
        } else {
            for (j = 0; j < n - 1; j++) {
                ra = A[j * lda + (j + 1)];
                rb = CMPLXF(2.0f, 0.0f);
                cblas_crotg(&ra, &rb, &c, &s);
                s = conjf(s);

                /* Multiply by [ c -s;  conjg(s) c] on the right */
                if (n > j + 2) {
                    c64 neg_s = -s;
                    crot(n - j - 2, &A[(j + 1) * lda + (j + 2)], 1,
                         &A[j * lda + (j + 2)], 1, c, neg_s);
                }

                /* Multiply by [-c  s; -conjg(s) -c] on the left */
                if (j > 0) {
                    crot(j, &A[0 * lda + j], lda,
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
                clarnv_rng(4, j, &A[j * lda], state);
                A[j * lda + j] = clarnd_rng(5, state) * TWO;
            }
        } else {
            for (j = 0; j < n; j++) {
                if (j < n - 1)
                    clarnv_rng(4, n - j - 1, &A[j * lda + j + 1], state);
                A[j * lda + j] = clarnd_rng(5, state) * TWO;
            }
        }

        /* Set the right hand side so that the largest value is BIGNUM */
        clarnv_rng(2, n, B, state);
        iy = cblas_icamax(n, B, 1);
        bnorm = cabsf(B[iy]);
        bscal = bignum / fmaxf(ONE, bnorm);
        cblas_csscal(n, bscal, B, 1);
    }
    /* IMAT = 12: Small diagonal, small off-diagonal (CNORM < 1) */
    else if (imat == 12) {
        clarnv_rng(2, n, B, state);
        tscal = ONE / fmaxf(ONE, (f32)(n - 1));
        if (upper) {
            for (j = 0; j < n; j++) {
                clarnv_rng(4, j, &A[j * lda], state);
                cblas_csscal(j, tscal, &A[j * lda], 1);
                A[j * lda + j] = clarnd_rng(5, state);
            }
            A[(n - 1) * lda + (n - 1)] = smlnum * A[(n - 1) * lda + (n - 1)];
        } else {
            for (j = 0; j < n; j++) {
                if (j < n - 1) {
                    clarnv_rng(4, n - j - 1, &A[j * lda + j + 1], state);
                    cblas_csscal(n - j - 1, tscal, &A[j * lda + j + 1], 1);
                }
                A[j * lda + j] = clarnd_rng(5, state);
            }
            A[0] = smlnum * A[0];
        }
    }
    /* IMAT = 13: Small diagonal, O(1) off-diagonal (CNORM > 1) */
    else if (imat == 13) {
        clarnv_rng(2, n, B, state);
        if (upper) {
            for (j = 0; j < n; j++) {
                clarnv_rng(4, j, &A[j * lda], state);
                A[j * lda + j] = clarnd_rng(5, state);
            }
            A[(n - 1) * lda + (n - 1)] = smlnum * A[(n - 1) * lda + (n - 1)];
        } else {
            for (j = 0; j < n; j++) {
                if (j < n - 1)
                    clarnv_rng(4, n - j - 1, &A[j * lda + j + 1], state);
                A[j * lda + j] = clarnd_rng(5, state);
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
                    A[j * lda + j] = smlnum * clarnd_rng(5, state);
                } else {
                    A[j * lda + j] = clarnd_rng(5, state);
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
                    A[j * lda + j] = smlnum * clarnd_rng(5, state);
                } else {
                    A[j * lda + j] = clarnd_rng(5, state);
                }
                jcount++;
                if (jcount > 4) jcount = 1;
            }
        }

        /* Set the right hand side alternately zero and small */
        if (upper) {
            B[0] = ZERO;
            for (i = n - 1; i >= 1; i -= 2) {
                B[i] = ZERO;
                B[i - 1] = smlnum * clarnd_rng(5, state);
            }
        } else {
            B[n - 1] = ZERO;
            for (i = 0; i < n - 1; i += 2) {
                B[i] = ZERO;
                B[i + 1] = smlnum * clarnd_rng(5, state);
            }
        }
    }
    /* IMAT = 15: Bidiagonal with small diagonal causing gradual overflow */
    else if (imat == 15) {
        texp = ONE / fmaxf(ONE, (f32)(n - 1));
        tscal = powf(smlnum, texp);
        clarnv_rng(4, n, B, state);
        if (upper) {
            for (j = 0; j < n; j++) {
                for (i = 0; i < j - 1; i++) {
                    A[j * lda + i] = ZERO;
                }
                if (j > 0)
                    A[j * lda + (j - 1)] = CMPLXF(-ONE, -ONE);
                A[j * lda + j] = tscal * clarnd_rng(5, state);
            }
            B[n - 1] = CMPLXF(ONE, ONE);
        } else {
            for (j = 0; j < n; j++) {
                for (i = j + 2; i < n; i++) {
                    A[j * lda + i] = ZERO;
                }
                if (j < n - 1)
                    A[j * lda + (j + 1)] = CMPLXF(-ONE, -ONE);
                A[j * lda + j] = tscal * clarnd_rng(5, state);
            }
            B[0] = CMPLXF(ONE, ONE);
        }
    }
    /* IMAT = 16: One zero diagonal element */
    else if (imat == 16) {
        iy = n / 2;

        if (upper) {
            for (j = 0; j < n; j++) {
                clarnv_rng(4, j, &A[j * lda], state);
                if (j != iy) {
                    A[j * lda + j] = clarnd_rng(5, state) * TWO;
                } else {
                    A[j * lda + j] = ZERO;
                }
            }
        } else {
            for (j = 0; j < n; j++) {
                if (j < n - 1)
                    clarnv_rng(4, n - j - 1, &A[j * lda + j + 1], state);
                if (j != iy) {
                    A[j * lda + j] = clarnd_rng(5, state) * TWO;
                } else {
                    A[j * lda + j] = ZERO;
                }
            }
        }
        clarnv_rng(2, n, B, state);
        cblas_csscal(n, TWO, B, 1);
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
                A[j * lda + 0] = -tscal / (f32)(n + 1);
                A[j * lda + j] = ONE;
                B[j] = texp * (ONE - ulp);
                A[(j - 1) * lda + 0] = -(tscal / (f32)(n + 1)) / (f32)(n + 2);
                A[(j - 1) * lda + (j - 1)] = ONE;
                B[j - 1] = texp * (f32)(n * n + n - 1);
                texp = texp * TWO;
            }
            B[0] = ((f32)(n + 1) / (f32)(n + 2)) * tscal;
        } else {
            for (j = 0; j < n - 1; j += 2) {
                A[j * lda + (n - 1)] = -tscal / (f32)(n + 1);
                A[j * lda + j] = ONE;
                B[j] = texp * (ONE - ulp);
                A[(j + 1) * lda + (n - 1)] = -(tscal / (f32)(n + 1)) / (f32)(n + 2);
                A[(j + 1) * lda + (j + 1)] = ONE;
                B[j + 1] = texp * (f32)(n * n + n - 1);
                texp = texp * TWO;
            }
            B[n - 1] = ((f32)(n + 1) / (f32)(n + 2)) * tscal;
        }
    }
    /* IMAT = 18: Unit triangular with large RHS */
    else if (imat == 18) {
        if (upper) {
            for (j = 0; j < n; j++) {
                clarnv_rng(4, j, &A[j * lda], state);
                A[j * lda + j] = ZERO;
            }
        } else {
            for (j = 0; j < n; j++) {
                if (j < n - 1)
                    clarnv_rng(4, n - j - 1, &A[j * lda + j + 1], state);
                A[j * lda + j] = ZERO;
            }
        }

        /* Set the right hand side so that the largest value is BIGNUM */
        clarnv_rng(2, n, B, state);
        iy = cblas_icamax(n, B, 1);
        bnorm = cabsf(B[iy]);
        bscal = bignum / fmaxf(ONE, bnorm);
        cblas_csscal(n, bscal, B, 1);
    }
    /* IMAT = 19: Large elements causing column norm to exceed BIGNUM */
    else if (imat == 19) {
        tleft = bignum / fmaxf(ONE, (f32)(n - 1));
        tscal = bignum * ((f32)(n - 1) / fmaxf(ONE, (f32)n));
        if (upper) {
            for (j = 0; j < n; j++) {
                clarnv_rng(5, j + 1, &A[j * lda], state);
                rng_fill_f32(state, 1, j + 1, rwork);
                for (i = 0; i <= j; i++) {
                    A[j * lda + i] = A[j * lda + i] * (tleft + rwork[i] * tscal);
                }
            }
        } else {
            for (j = 0; j < n; j++) {
                clarnv_rng(5, n - j, &A[j * lda + j], state);
                rng_fill_f32(state, 1, n - j, rwork);
                for (i = j; i < n; i++) {
                    A[j * lda + i] = A[j * lda + i] * (tleft + rwork[i - j] * tscal);
                }
            }
        }
        clarnv_rng(2, n, B, state);
        cblas_csscal(n, TWO, B, 1);
    }

    /* Flip the matrix if the transpose will be used.
     *
     * Upper: swap row j with column n-j-1 (reversed)
     * Lower: swap column j with row n-j-1 (reversed)
     */
    if (trans[0] != 'N' && trans[0] != 'n') {
        if (upper) {
            for (j = 0; j < n / 2; j++) {
                INT len = n - 2 * j - 1;
                for (INT k = 0; k < len; k++) {
                    c64 temp = A[(j + k) * lda + j];
                    A[(j + k) * lda + j] = A[(n - j - 1) * lda + (n - j - 1 - k)];
                    A[(n - j - 1) * lda + (n - j - 1 - k)] = temp;
                }
            }
        } else {
            for (j = 0; j < n / 2; j++) {
                INT len = n - 2 * j - 1;
                for (INT k = 0; k < len; k++) {
                    c64 temp = A[j * lda + (j + k)];
                    A[j * lda + (j + k)] = A[(n - j - 1 - k) * lda + (n - j - 1)];
                    A[(n - j - 1 - k) * lda + (n - j - 1)] = temp;
                }
            }
        }
    }
}
