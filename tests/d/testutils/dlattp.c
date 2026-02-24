/**
 * @file dlattp.c
 * @brief DLATTP generates a triangular test matrix in packed storage.
 *
 * Port of LAPACK TESTING/LIN/dlattp.f to C.
 *
 * IMAT and UPLO uniquely specify the properties of the test matrix:
 *   IMAT 1-6:   Non-unit triangular generated via DLATMS
 *   IMAT 7:     Unit triangular identity
 *   IMAT 8-10:  Non-trivial unit triangular with controlled condition
 *   IMAT 11-19: Pathological test cases for DLATPS
 */

#include <math.h>
#include <stdlib.h>
#include "semicolon_cblas.h"
#include "verify.h"
#include "test_rng.h"

/* External declarations */
/**
 * DLATTP generates a triangular test matrix in packed storage.
 *
 * @param[in]     imat    An integer key describing which matrix to generate (1-19).
 * @param[in]     uplo    'U' for upper triangular, 'L' for lower triangular.
 * @param[in]     trans   'N' for no transpose, 'T' or 'C' for transpose.
 *                        Used to flip the matrix if transpose will be used.
 * @param[out]    diag    Returns 'N' for non-unit triangular, 'U' for unit triangular.
 * @param[in,out] seed    Random number seed.
 * @param[in]     n       The order of the matrix. n >= 0.
 * @param[out]    A       Array (n*(n+1)/2). The triangular matrix A in packed storage.
 *                        If UPLO='U', A(i,j) is stored at A((j-1)*j/2 + i) for i<=j.
 *                        If UPLO='L', A(i,j) is stored at A((j-1)*(2n-j)/2 + i) for i>=j.
 * @param[out]    B       Array (n). The right hand side vector (for IMAT > 10).
 * @param[out]    work    Array (3*n). Workspace.
 * @param[out]    info    0 = successful exit, < 0 = illegal argument.
 */
void dlattp(const INT imat, const char* uplo, const char* trans, char* diag,
            const INT n, f64* A, f64* B, f64* work,
            INT* info, uint64_t state[static 4])
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const f64 TWO = 2.0;

    f64 unfl, ulp, smlnum, bignum;
    INT upper;
    char type, dist, packit;
    INT kl, ku, mode;
    f64 anorm, cndnum;
    INT i, j, jc, jcnext, jcount, jj, jl, jr, jx, iy;
    f64 bnorm, bscal, tscal, texp, tleft;
    f64 plus1, plus2, star1, sfac, rexp;
    f64 x, y, z, c, s, ra, rb, stemp, t;

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
        dlatb4("DTP", imat, n, n, &type, &kl, &ku, &anorm, &mode, &cndnum, &dist);
        packit = 'C';
    } else {
        dlatb4("DTP", -imat, n, n, &type, &kl, &ku, &anorm, &mode, &cndnum, &dist);
        packit = 'R';
    }

    /* IMAT <= 6: Non-unit triangular matrix via DLATMS */
    if (imat <= 6) {
        char symm[2] = "N";
        symm[0] = type;
        char dstr[2] = "S";
        dstr[0] = dist;
        char pack[2] = "C";
        pack[0] = packit;
        dlatms(n, n, dstr, symm, B, mode, cndnum, anorm,
               kl, ku, pack, A, n, work, info, state);
    }
    /* IMAT = 7: Unit triangular identity */
    else if (imat == 7) {
        if (upper) {
            jc = 0;
            for (j = 0; j < n; j++) {
                for (i = 0; i < j; i++) {
                    A[jc + i] = ZERO;
                }
                A[jc + j] = (f64)(j + 1);
                jc += j + 1;
            }
        } else {
            jc = 0;
            for (j = 0; j < n; j++) {
                A[jc] = (f64)(j + 1);
                for (i = j + 1; i < n; i++) {
                    A[jc + i - j] = ZERO;
                }
                jc += n - j;
            }
        }
    }
    /* IMAT 8-10: Non-trivial unit triangular with controlled condition */
    else if (imat <= 10) {
        /* Initialize to identity-like structure with diagonal = j+1 */
        if (upper) {
            jc = 0;
            for (j = 0; j < n; j++) {
                for (i = 0; i < j; i++) {
                    A[jc + i] = ZERO;
                }
                A[jc + j] = (f64)(j + 1);
                jc += j + 1;
            }
        } else {
            jc = 0;
            for (j = 0; j < n; j++) {
                A[jc] = (f64)(j + 1);
                for (i = j + 1; i < n; i++) {
                    A[jc + i - j] = ZERO;
                }
                jc += n - j;
            }
        }

        /* Build work arrays for superdiagonal structure.
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
                rexp = rng_uniform(state) * 2.0 - 1.0;
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
            /* Set the upper triangle of A with a unit triangular matrix
             * of known condition number. */
            jc = 0;
            for (j = 1; j < n; j++) {
                A[jc + 1] = y;
                if (j > 1) {
                    A[jc + j] = work[j - 2];
                }
                if (j > 2) {
                    A[jc + j - 1] = work[n + j - 3];
                }
                jc += j + 1;
            }
            jc -= n;
            A[jc + 1] = z;
            for (j = 1; j < n - 1; j++) {
                A[jc + j + 1] = y;
            }
        } else {
            /* Set the lower triangle of A with a unit triangular matrix
             * of known condition number. */
            for (i = 1; i < n - 1; i++) {
                A[i] = y;
            }
            A[n - 1] = z;
            jc = n;
            for (j = 1; j < n - 1; j++) {
                A[jc + 1] = work[j - 1];
                if (j < n - 2) {
                    A[jc + 2] = work[n + j - 1];
                }
                A[jc + n - j - 1] = y;
                jc += n - j;
            }
        }

        /* Fill in the zeros using Givens rotations */
        if (upper) {
            jc = 0;
            for (j = 0; j < n - 1; j++) {
                jcnext = jc + j + 1;
                ra = A[jcnext + j];
                rb = TWO;
                cblas_drotg(&ra, &rb, &c, &s);

                /* Multiply by [ c  s; -s  c] on the left */
                if (n > j + 2) {
                    jx = jcnext + j + 1;
                    for (i = j + 2; i < n; i++) {
                        stemp = c * A[jx + j] + s * A[jx + j + 1];
                        A[jx + j + 1] = -s * A[jx + j] + c * A[jx + j + 1];
                        A[jx + j] = stemp;
                        jx += i + 1;
                    }
                }

                /* Multiply by [-c -s;  s -c] on the right */
                if (j > 0) {
                    cblas_drot(j, &A[jcnext], 1, &A[jc], 1, -c, -s);
                }

                /* Negate A(j,j+1) */
                A[jcnext + j] = -A[jcnext + j];
                jc = jcnext;
            }
        } else {
            jc = 0;
            for (j = 0; j < n - 1; j++) {
                jcnext = jc + n - j;
                ra = A[jc + 1];
                rb = TWO;
                cblas_drotg(&ra, &rb, &c, &s);

                /* Multiply by [ c -s;  s  c] on the right */
                if (n > j + 2) {
                    cblas_drot(n - j - 2, &A[jcnext + 1], 1, &A[jc + 2], 1, c, -s);
                }

                /* Multiply by [-c  s; -s -c] on the left */
                if (j > 0) {
                    jx = 0;
                    for (i = 0; i < j; i++) {
                        stemp = -c * A[jx + j - i] + s * A[jx + j - i + 1];
                        A[jx + j - i + 1] = -s * A[jx + j - i] - c * A[jx + j - i + 1];
                        A[jx + j - i] = stemp;
                        jx += n - i;
                    }
                }

                /* Negate A(j+1,j) */
                A[jc + 1] = -A[jc + 1];
                jc = jcnext;
            }
        }
    }
    /* IMAT = 11: Generate a triangular matrix with elements between -1 and 1.
     * Give the diagonal norm 2 to make it well-conditioned.
     * Make the right hand side large so that it requires scaling. */
    else if (imat == 11) {
        if (upper) {
            jc = 0;
            for (j = 0; j < n; j++) {
                dlarnv_rng(2, j + 1, &A[jc], state);
                A[jc + j] = (A[jc + j] >= 0) ? TWO : -TWO;
                jc += j + 1;
            }
        } else {
            jc = 0;
            for (j = 0; j < n; j++) {
                dlarnv_rng(2, n - j, &A[jc], state);
                A[jc] = (A[jc] >= 0) ? TWO : -TWO;
                jc += n - j;
            }
        }

        /* Set the right hand side so that the largest value is BIGNUM */
        dlarnv_rng(2, n, B, state);
        iy = cblas_idamax(n, B, 1);
        bnorm = fabs(B[iy]);
        bscal = bignum / fmax(ONE, bnorm);
        cblas_dscal(n, bscal, B, 1);
    }
    /* IMAT = 12: Make the first diagonal element in the solve small to
     * cause immediate overflow when dividing by T(j,j).
     * In type 12, the offdiagonal elements are small (CNORM(j) < 1). */
    else if (imat == 12) {
        dlarnv_rng(2, n, B, state);
        tscal = ONE / fmax(ONE, (f64)(n - 1));
        if (upper) {
            jc = 0;
            for (j = 0; j < n; j++) {
                dlarnv_rng(2, j, &A[jc], state);
                cblas_dscal(j, tscal, &A[jc], 1);
                A[jc + j] = (rng_uniform(state) >= 0.5) ? ONE : -ONE;
                jc += j + 1;
            }
            A[n * (n + 1) / 2 - 1] = smlnum;
        } else {
            jc = 0;
            for (j = 0; j < n; j++) {
                dlarnv_rng(2, n - j - 1, &A[jc + 1], state);
                cblas_dscal(n - j - 1, tscal, &A[jc + 1], 1);
                A[jc] = (rng_uniform(state) >= 0.5) ? ONE : -ONE;
                jc += n - j;
            }
            A[0] = smlnum;
        }
    }
    /* IMAT = 13: Make the first diagonal element in the solve small to
     * cause immediate overflow when dividing by T(j,j).
     * In type 13, the offdiagonal elements are O(1) (CNORM(j) > 1). */
    else if (imat == 13) {
        dlarnv_rng(2, n, B, state);
        if (upper) {
            jc = 0;
            for (j = 0; j < n; j++) {
                dlarnv_rng(2, j, &A[jc], state);
                A[jc + j] = (rng_uniform(state) >= 0.5) ? ONE : -ONE;
                jc += j + 1;
            }
            A[n * (n + 1) / 2 - 1] = smlnum;
        } else {
            jc = 0;
            for (j = 0; j < n; j++) {
                dlarnv_rng(2, n - j - 1, &A[jc + 1], state);
                A[jc] = (rng_uniform(state) >= 0.5) ? ONE : -ONE;
                jc += n - j;
            }
            A[0] = smlnum;
        }
    }
    /* IMAT = 14: T is diagonal with small numbers on the diagonal to
     * make the growth factor underflow, but a small right hand side
     * chosen so that the solution does not overflow. */
    else if (imat == 14) {
        if (upper) {
            jcount = 1;
            jc = (n - 1) * n / 2;
            for (j = n - 1; j >= 0; j--) {
                for (i = 0; i < j; i++) {
                    A[jc + i] = ZERO;
                }
                if (jcount <= 2) {
                    A[jc + j] = smlnum;
                } else {
                    A[jc + j] = ONE;
                }
                jcount++;
                if (jcount > 4) {
                    jcount = 1;
                }
                jc -= j;
            }
        } else {
            jcount = 1;
            jc = 0;
            for (j = 0; j < n; j++) {
                for (i = j + 1; i < n; i++) {
                    A[jc + i - j] = ZERO;
                }
                if (jcount <= 2) {
                    A[jc] = smlnum;
                } else {
                    A[jc] = ONE;
                }
                jcount++;
                if (jcount > 4) {
                    jcount = 1;
                }
                jc += n - j;
            }
        }

        /* Set the right hand side alternately zero and small */
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
    /* IMAT = 15: Make the diagonal elements small to cause gradual
     * overflow when dividing by T(j,j). To control the amount of
     * scaling needed, the matrix is bidiagonal. */
    else if (imat == 15) {
        texp = ONE / fmax(ONE, (f64)(n - 1));
        tscal = pow(smlnum, texp);
        dlarnv_rng(2, n, B, state);
        if (upper) {
            jc = 0;
            for (j = 0; j < n; j++) {
                for (i = 0; i < j - 1; i++) {
                    A[jc + i] = ZERO;
                }
                if (j > 0) {
                    A[jc + j - 1] = -ONE;
                }
                A[jc + j] = tscal;
                jc += j + 1;
            }
            B[n - 1] = ONE;
        } else {
            jc = 0;
            for (j = 0; j < n; j++) {
                for (i = j + 2; i < n; i++) {
                    A[jc + i - j] = ZERO;
                }
                if (j < n - 1) {
                    A[jc + 1] = -ONE;
                }
                A[jc] = tscal;
                jc += n - j;
            }
            B[0] = ONE;
        }
    }
    /* IMAT = 16: One zero diagonal element. */
    else if (imat == 16) {
        iy = n / 2;
        if (upper) {
            jc = 0;
            for (j = 0; j < n; j++) {
                dlarnv_rng(2, j + 1, &A[jc], state);
                if (j != iy) {
                    A[jc + j] = (A[jc + j] >= 0) ? TWO : -TWO;
                } else {
                    A[jc + j] = ZERO;
                }
                jc += j + 1;
            }
        } else {
            jc = 0;
            for (j = 0; j < n; j++) {
                dlarnv_rng(2, n - j, &A[jc], state);
                if (j != iy) {
                    A[jc] = (A[jc] >= 0) ? TWO : -TWO;
                } else {
                    A[jc] = ZERO;
                }
                jc += n - j;
            }
        }
        dlarnv_rng(2, n, B, state);
        cblas_dscal(n, TWO, B, 1);
    }
    /* IMAT = 17: Make the offdiagonal elements large to cause overflow
     * when adding a column of T. In the non-transposed case, the
     * matrix is constructed to cause overflow when adding a column in
     * every other step. */
    else if (imat == 17) {
        tscal = unfl / ulp;
        tscal = (ONE - ulp) / tscal;
        for (j = 0; j < n * (n + 1) / 2; j++) {
            A[j] = ZERO;
        }
        texp = ONE;
        if (upper) {
            jc = (n - 1) * n / 2;
            for (j = n - 1; j >= 1; j -= 2) {
                A[jc] = -tscal / (f64)(n + 1);
                A[jc + j] = ONE;
                B[j] = texp * (ONE - ulp);
                jc -= j;
                A[jc] = -(tscal / (f64)(n + 1)) / (f64)(n + 2);
                A[jc + j - 1] = ONE;
                B[j - 1] = texp * (f64)(n * n + n - 1);
                texp *= TWO;
                jc -= j - 1;
            }
            B[0] = ((f64)(n + 1) / (f64)(n + 2)) * tscal;
        } else {
            jc = 0;
            for (j = 0; j < n - 1; j += 2) {
                A[jc + n - j - 1] = -tscal / (f64)(n + 1);
                A[jc] = ONE;
                B[j] = texp * (ONE - ulp);
                jc += n - j;
                A[jc + n - j - 2] = -(tscal / (f64)(n + 1)) / (f64)(n + 2);
                A[jc] = ONE;
                B[j + 1] = texp * (f64)(n * n + n - 1);
                texp *= TWO;
                jc += n - j - 1;
            }
            B[n - 1] = ((f64)(n + 1) / (f64)(n + 2)) * tscal;
        }
    }
    /* IMAT = 18: Generate a unit triangular matrix with elements
     * between -1 and 1, and make the right hand side large so that it
     * requires scaling. */
    else if (imat == 18) {
        if (upper) {
            jc = 0;
            for (j = 0; j < n; j++) {
                dlarnv_rng(2, j, &A[jc], state);
                A[jc + j] = ZERO;
                jc += j + 1;
            }
        } else {
            jc = 0;
            for (j = 0; j < n; j++) {
                if (j < n - 1) {
                    dlarnv_rng(2, n - j - 1, &A[jc + 1], state);
                }
                A[jc] = ZERO;
                jc += n - j;
            }
        }

        /* Set the right hand side so that the largest value is BIGNUM */
        dlarnv_rng(2, n, B, state);
        iy = cblas_idamax(n, B, 1);
        bnorm = fabs(B[iy]);
        bscal = bignum / fmax(ONE, bnorm);
        cblas_dscal(n, bscal, B, 1);
    }
    /* IMAT = 19: Generate a triangular matrix with elements between
     * BIGNUM/(n-1) and BIGNUM so that at least one of the column
     * norms will exceed BIGNUM. */
    else if (imat == 19) {
        tleft = bignum / fmax(ONE, (f64)(n - 1));
        tscal = bignum * ((f64)(n - 1) / fmax(ONE, (f64)n));
        if (upper) {
            jc = 0;
            for (j = 0; j < n; j++) {
                dlarnv_rng(2, j + 1, &A[jc], state);
                for (i = 0; i <= j; i++) {
                    A[jc + i] = (A[jc + i] >= 0 ? tleft : -tleft) + tscal * A[jc + i];
                }
                jc += j + 1;
            }
        } else {
            jc = 0;
            for (j = 0; j < n; j++) {
                dlarnv_rng(2, n - j, &A[jc], state);
                for (i = j; i < n; i++) {
                    A[jc + i - j] = (A[jc + i - j] >= 0 ? tleft : -tleft) + tscal * A[jc + i - j];
                }
                jc += n - j;
            }
        }
        dlarnv_rng(2, n, B, state);
        cblas_dscal(n, TWO, B, 1);
    }

    /* Flip the matrix across its counter-diagonal if the transpose will
     * be used. */
    if (trans[0] != 'N' && trans[0] != 'n') {
        if (upper) {
            jj = 0;
            jr = n * (n + 1) / 2 - 1;
            for (j = 0; j < n / 2; j++) {
                jl = jj;
                for (i = j; i < n - j - 1; i++) {
                    t = A[jr - i + j];
                    A[jr - i + j] = A[jl];
                    A[jl] = t;
                    jl += i + 1;
                }
                jj += j + 2;
                jr -= n - j;
            }
        } else {
            jl = 0;
            jj = n * (n + 1) / 2 - 1;
            for (j = 0; j < n / 2; j++) {
                jr = jj;
                for (i = j; i < n - j - 1; i++) {
                    t = A[jl + i - j];
                    A[jl + i - j] = A[jr];
                    A[jr] = t;
                    jr -= i + 1;
                }
                jl += n - j;
                jj -= j + 2;
            }
        }
    }
}
