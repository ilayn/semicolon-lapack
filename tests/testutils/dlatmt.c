/**
 * @file dlatmt.c
 * @brief DLATMT generates random matrices with specified singular values.
 *
 * Faithful port of LAPACK TESTING/MATGEN/dlatmt.f
 */

#include <math.h>
#include <cblas.h>
#include "verify.h"
#include "test_rng.h"

/* Forward declarations */
extern void xerbla(const char* srname, const int info);
extern void dlaset(const char* uplo, const int m, const int n,
                   const double alpha, const double beta,
                   double* A, const int lda);
extern void dlartg(const double f, const double g, double* c, double* s, double* r);

/* Constants */
static const double ZERO = 0.0;
static const double ONE = 1.0;
static const double TWOPI = 6.28318530717958647692528676655900576839;

/**
 * DLATMT generates random matrices with specified singular values
 * (or symmetric/hermitian with specified eigenvalues)
 * for testing LAPACK programs.
 *
 * DLATMT operates by applying the following sequence of operations:
 *
 *   Set the diagonal to D, where D may be input or computed according
 *   to MODE, COND, DMAX, and SYM.
 *
 *   Generate a matrix with the appropriate band structure, by one
 *   of two methods:
 *
 *   Method A:
 *       Generate a dense M x N matrix by multiplying D on the left
 *       and the right by random unitary matrices, then reduce the
 *       bandwidth according to KL and KU, using Householder
 *       transformations.
 *
 *   Method B:
 *       Convert the bandwidth-0 (diagonal) matrix to bandwidth-1
 *       using Givens rotations, "chasing" out-of-band elements back,
 *       then convert to bandwidth-2, etc.
 *
 *   Method A is chosen if the bandwidth is a large fraction of the
 *   order of the matrix, and LDA >= M. Method B is chosen if the
 *   bandwidth is small (< 1/2 N for symmetric, < 0.3(N+M) for
 *   non-symmetric), or LDA is less than M but >= the bandwidth.
 *
 *   Pack the matrix if desired.
 *
 * @param[in] m       Number of rows of A.
 * @param[in] n       Number of columns of A.
 * @param[in] dist    Distribution: 'U'=uniform(0,1), 'S'=uniform(-1,1), 'N'=normal(0,1).
 * @param[in] sym     Symmetry: 'N'=nonsymmetric, 'S'/'H'=symmetric, 'P'=positive definite.
 * @param[in,out] d   Array of singular/eigenvalues, dimension min(m,n).
 * @param[in] mode    How to compute D (0-6, negative reverses order).
 * @param[in] cond    Condition number (>= 1).
 * @param[in] dmax    Maximum singular value.
 * @param[in] rank    Rank of matrix for modes 1,2,3.
 * @param[in] kl      Lower bandwidth.
 * @param[in] ku      Upper bandwidth.
 * @param[in] pack    Packing: 'N'=none, 'U'/'L'=triangular, 'C'/'R'=packed,
 *                    'B'/'Q'=band, 'Z'=full band.
 * @param[out] A      Output matrix, dimension (lda, n).
 * @param[in] lda     Leading dimension of A.
 * @param[out] work   Workspace, dimension (3*max(m,n)).
 * @param[out] info   0=success, <0=argument error, >0=other error.
 */
void dlatmt(const int m, const int n, const char* dist,
            const char* sym, double* d, const int mode,
            const double cond, const double dmax, const int rank,
            const int kl, const int ku, const char* pack,
            double* A, const int lda, double* work, int* info,
            uint64_t state[static 4])
{
    /* Local scalars */
    double alpha, angle, c, dummy, extra, s, temp;
    int i, ic, icol = 0, idist, iendch, iinfo, il, ilda;
    int ioffg, ioffst, ipack, ipackg, ir, ir1, ir2;
    int irow = 0, irsign, iskew, isym, isympk, j, jc, jch;
    int jkl, jku, jr, k, llb, minlda, mnmin, mr, nc, uub;
    int givens, ilextr, iltemp, topdwn;

    *info = 0;

    /* Quick return if possible */
    if (m == 0 || n == 0) {
        return;
    }

    /* Decode DIST */
    if (dist[0] == 'U' || dist[0] == 'u') {
        idist = 1;
    } else if (dist[0] == 'S' || dist[0] == 's') {
        idist = 2;
    } else if (dist[0] == 'N' || dist[0] == 'n') {
        idist = 3;
    } else {
        idist = -1;
    }

    /* Decode SYM */
    if (sym[0] == 'N' || sym[0] == 'n') {
        isym = 1;
        irsign = 0;
    } else if (sym[0] == 'P' || sym[0] == 'p') {
        isym = 2;
        irsign = 0;
    } else if (sym[0] == 'S' || sym[0] == 's' ||
               sym[0] == 'H' || sym[0] == 'h') {
        isym = 2;
        irsign = 1;
    } else {
        isym = -1;
        irsign = 0;
    }

    /* Decode PACK */
    isympk = 0;
    if (pack[0] == 'N' || pack[0] == 'n') {
        ipack = 0;
    } else if (pack[0] == 'U' || pack[0] == 'u') {
        ipack = 1;
        isympk = 1;
    } else if (pack[0] == 'L' || pack[0] == 'l') {
        ipack = 2;
        isympk = 1;
    } else if (pack[0] == 'C' || pack[0] == 'c') {
        ipack = 3;
        isympk = 2;
    } else if (pack[0] == 'R' || pack[0] == 'r') {
        ipack = 4;
        isympk = 3;
    } else if (pack[0] == 'B' || pack[0] == 'b') {
        ipack = 5;
        isympk = 3;
    } else if (pack[0] == 'Q' || pack[0] == 'q') {
        ipack = 6;
        isympk = 2;
    } else if (pack[0] == 'Z' || pack[0] == 'z') {
        ipack = 7;
    } else {
        ipack = -1;
    }

    /* Set certain internal parameters */
    mnmin = (m < n) ? m : n;
    llb = (kl < m - 1) ? kl : m - 1;
    uub = (ku < n - 1) ? ku : n - 1;
    mr = (m < n + llb) ? m : n + llb;
    nc = (n < m + uub) ? n : m + uub;

    if (ipack == 5 || ipack == 6) {
        minlda = uub + 1;
    } else if (ipack == 7) {
        minlda = llb + uub + 1;
    } else {
        minlda = m;
    }

    /* Use Givens rotation method if bandwidth small enough,
     * or if LDA is too small to store the matrix unpacked. */
    givens = 0;
    if (isym == 1) {
        if ((double)(llb + uub) < 0.3 * (double)((mr + nc > 1) ? mr + nc : 1)) {
            givens = 1;
        }
    } else {
        if (2 * llb < m) {
            givens = 1;
        }
    }
    if (lda < m && lda >= minlda) {
        givens = 1;
    }

    /* Set INFO if an error */
    if (m < 0) {
        *info = -1;
    } else if (m != n && isym != 1) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (idist == -1) {
        *info = -3;
    } else if (isym == -1) {
        *info = -5;
    } else if (mode < -6 || mode > 6) {
        *info = -7;
    } else if ((mode != 0 && mode != 6 && mode != -6) && cond < ONE) {
        *info = -8;
    } else if (kl < 0) {
        *info = -10;
    } else if (ku < 0 || (isym != 1 && kl != ku)) {
        *info = -11;
    } else if (ipack == -1 || (isympk == 1 && isym == 1) ||
               (isympk == 2 && isym == 1 && kl > 0) ||
               (isympk == 3 && isym == 1 && ku > 0) ||
               (isympk != 0 && m != n)) {
        *info = -12;
    } else if (lda < ((1 > minlda) ? 1 : minlda)) {
        *info = -14;
    }

    if (*info != 0) {
        xerbla("DLATMT", -(*info));
        return;
    }

    /* 2) Set up D if indicated.
     *    Compute D according to COND and MODE */
    dlatm7(mode, cond, irsign, idist, d, mnmin, rank, &iinfo, state);
    if (iinfo != 0) {
        *info = 1;
        return;
    }

    /* Choose Top-Down if D is (apparently) increasing,
     * Bottom-Up if D is (apparently) decreasing. */
    if (fabs(d[0]) <= fabs(d[rank - 1])) {
        topdwn = 1;
    } else {
        topdwn = 0;
    }

    if (mode != 0 && mode != 6 && mode != -6) {
        /* Scale by DMAX */
        temp = fabs(d[0]);
        for (i = 1; i < rank; i++) {
            if (fabs(d[i]) > temp) {
                temp = fabs(d[i]);
            }
        }

        if (temp > ZERO) {
            alpha = dmax / temp;
        } else {
            *info = 2;
            return;
        }

        cblas_dscal(rank, alpha, d, 1);
    }

    /* 3) Generate Banded Matrix using Givens rotations.
     *    Also the special case of UUB=LLB=0
     *
     *    Compute Addressing constants to cover all storage formats.
     *    Whether GE, SY, GB, or SB, upper or lower triangle or both,
     *    the (i,j)-th element is in A[i - ISKEW*j + IOFFST + j*lda]
     *    (using 0-based indexing in C) */

    if (ipack > 4) {
        ilda = lda - 1;
        iskew = 1;
        if (ipack > 5) {
            ioffst = uub + 1;
        } else {
            ioffst = 1;
        }
    } else {
        ilda = lda;
        iskew = 0;
        ioffst = 0;
    }

    /* IPACKG is the format that the matrix is generated in. If this is
     * different from IPACK, then the matrix must be repacked at the end.
     * It also signals how to compute the norm, for scaling. */
    ipackg = 0;
    dlaset("Full", lda, n, ZERO, ZERO, A, lda);

    /* Diagonal Matrix -- We are done, unless it
     * is to be stored SP/PP/TP (PACK='R' or 'C') */
    if (llb == 0 && uub == 0) {
        /* Copy D to diagonal using DCOPY with stride = ilda+1
         * In Fortran: CALL DCOPY(MNMIN, D, 1, A(1-ISKEW+IOFFST, 1), ILDA+1)
         * In C (0-based): starting at A[0 - iskew*0 + ioffst - 1] but we need
         * to be careful about the addressing. For unpacked (iskew=0, ioffst=0):
         * A[0 + 0*lda], A[1 + 1*lda], etc. with stride = lda+1.
         * For band format, the formula is different. */
        int diag_start = (1 - iskew) + ioffst - 1;  /* C 0-based offset */
        if (diag_start < 0) diag_start = 0;
        cblas_dcopy(mnmin, d, 1, &A[diag_start], ilda + 1);
        if (ipack <= 2 || ipack >= 5) {
            ipackg = ipack;
        }

    } else if (givens) {
        /* Check whether to use Givens rotations,
         * Householder transformations, or nothing. */

        if (isym == 1) {
            /* Non-symmetric -- A = U D V */
            if (ipack > 4) {
                ipackg = ipack;
            } else {
                ipackg = 0;
            }

            /* Copy D to diagonal */
            int diag_start = (1 - iskew) + ioffst - 1;
            if (diag_start < 0) diag_start = 0;
            cblas_dcopy(mnmin, d, 1, &A[diag_start], ilda + 1);

            if (topdwn) {
                jkl = 0;
                for (jku = 1; jku <= uub; jku++) {
                    /* Transform from bandwidth JKL, JKU-1 to JKL, JKU
                     * Last row actually rotated is M
                     * Last column actually rotated is MIN(M+JKU, N) */
                    int limit = ((m + jku < n) ? m + jku : n) + jkl - 1;
                    for (jr = 1; jr <= limit; jr++) {
                        extra = ZERO;
                        angle = TWOPI * rng_uniform(state);
                        c = cos(angle);
                        s = sin(angle);
                        icol = (1 > jr - jkl) ? 1 : jr - jkl;
                        if (jr < m) {
                            il = ((n < jr + jku) ? n : jr + jku) + 1 - icol;
                            /* Fortran: A(JR-ISKEW*ICOL+IOFFST, ICOL)
                             * C: A[(jr-1) - iskew*(icol-1) + ioffst - 1 + (icol-1)*lda] */
                            int aoff = (jr - 1) - iskew * (icol - 1) + ioffst - 1 + (icol - 1) * lda;
                            if (ioffst == 0 && iskew == 0) {
                                aoff = (jr - 1) + (icol - 1) * lda;
                            }
                            dlarot(1, jr > jkl ? 1 : 0, 0, il, c, s,
                                   &A[aoff], ilda, &extra, &dummy);
                        }

                        /* Chase "EXTRA" back up */
                        ir = jr;
                        ic = icol;
                        for (jch = jr - jkl; jch >= 1; jch -= jkl + jku) {
                            if (ir < m) {
                                int idx = (ir + 1 - 1) - iskew * (ic + 1 - 1) + ioffst - 1 + (ic + 1 - 1) * lda;
                                if (ioffst == 0 && iskew == 0) {
                                    idx = ir + ic * lda;
                                }
                                dlartg(A[idx], extra, &c, &s, &dummy);
                            }
                            irow = (1 > jch - jku) ? 1 : jch - jku;
                            il = ir + 2 - irow;
                            temp = ZERO;
                            iltemp = jch > jku ? 1 : 0;
                            {
                                int aoff = (irow - 1) - iskew * (ic - 1) + ioffst - 1 + (ic - 1) * lda;
                                if (ioffst == 0 && iskew == 0) {
                                    aoff = (irow - 1) + (ic - 1) * lda;
                                }
                                dlarot(0, iltemp, 1, il, c, -s,
                                       &A[aoff], ilda, &temp, &extra);
                            }
                            if (iltemp) {
                                int idx = (irow + 1 - 1) - iskew * (ic + 1 - 1) + ioffst - 1 + (ic + 1 - 1) * lda;
                                if (ioffst == 0 && iskew == 0) {
                                    idx = irow + ic * lda;
                                }
                                dlartg(A[idx], temp, &c, &s, &dummy);
                                icol = (1 > jch - jku - jkl) ? 1 : jch - jku - jkl;
                                il = ic + 2 - icol;
                                extra = ZERO;
                                {
                                    int aoff = (irow - 1) - iskew * (icol - 1) + ioffst - 1 + (icol - 1) * lda;
                                    if (ioffst == 0 && iskew == 0) {
                                        aoff = (irow - 1) + (icol - 1) * lda;
                                    }
                                    dlarot(1, jch > jku + jkl ? 1 : 0, 1, il, c, -s,
                                           &A[aoff], ilda, &extra, &temp);
                                }
                                ic = icol;
                                ir = irow;
                            }
                        }
                    }
                }

                jku = uub;
                for (jkl = 1; jkl <= llb; jkl++) {
                    /* Transform from bandwidth JKL-1, JKU to JKL, JKU */
                    int limit = ((n + jkl < m) ? n + jkl : m) + jku - 1;
                    for (jc = 1; jc <= limit; jc++) {
                        extra = ZERO;
                        angle = TWOPI * rng_uniform(state);
                        c = cos(angle);
                        s = sin(angle);
                        irow = (1 > jc - jku) ? 1 : jc - jku;
                        if (jc < n) {
                            il = ((m < jc + jkl) ? m : jc + jkl) + 1 - irow;
                            int aoff = (irow - 1) - iskew * (jc - 1) + ioffst - 1 + (jc - 1) * lda;
                            if (ioffst == 0 && iskew == 0) {
                                aoff = (irow - 1) + (jc - 1) * lda;
                            }
                            dlarot(0, jc > jku ? 1 : 0, 0, il, c, s,
                                   &A[aoff], ilda, &extra, &dummy);
                        }

                        /* Chase "EXTRA" back up */
                        ic = jc;
                        ir = irow;
                        for (jch = jc - jku; jch >= 1; jch -= jkl + jku) {
                            if (ic < n) {
                                int idx = (ir + 1 - 1) - iskew * (ic + 1 - 1) + ioffst - 1 + (ic + 1 - 1) * lda;
                                if (ioffst == 0 && iskew == 0) {
                                    idx = ir + ic * lda;
                                }
                                dlartg(A[idx], extra, &c, &s, &dummy);
                            }
                            icol = (1 > jch - jkl) ? 1 : jch - jkl;
                            il = ic + 2 - icol;
                            temp = ZERO;
                            iltemp = jch > jkl ? 1 : 0;
                            {
                                int aoff = (ir - 1) - iskew * (icol - 1) + ioffst - 1 + (icol - 1) * lda;
                                if (ioffst == 0 && iskew == 0) {
                                    aoff = (ir - 1) + (icol - 1) * lda;
                                }
                                dlarot(1, iltemp, 1, il, c, -s,
                                       &A[aoff], ilda, &temp, &extra);
                            }
                            if (iltemp) {
                                int idx = (ir + 1 - 1) - iskew * (icol + 1 - 1) + ioffst - 1 + (icol + 1 - 1) * lda;
                                if (ioffst == 0 && iskew == 0) {
                                    idx = ir + icol * lda;
                                }
                                dlartg(A[idx], temp, &c, &s, &dummy);
                                irow = (1 > jch - jkl - jku) ? 1 : jch - jkl - jku;
                                il = ir + 2 - irow;
                                extra = ZERO;
                                {
                                    int aoff = (irow - 1) - iskew * (icol - 1) + ioffst - 1 + (icol - 1) * lda;
                                    if (ioffst == 0 && iskew == 0) {
                                        aoff = (irow - 1) + (icol - 1) * lda;
                                    }
                                    dlarot(0, jch > jkl + jku ? 1 : 0, 1, il, c, -s,
                                           &A[aoff], ilda, &extra, &temp);
                                }
                                ic = icol;
                                ir = irow;
                            }
                        }
                    }
                }

            } else {
                /* Bottom-Up -- Start at the bottom right. */
                jkl = 0;
                for (jku = 1; jku <= uub; jku++) {
                    /* Transform from bandwidth JKL, JKU-1 to JKL, JKU
                     * First row actually rotated is M
                     * First column actually rotated is MIN(M+JKU, N) */
                    iendch = ((m < n + jkl) ? m : n + jkl) - 1;
                    int start_jc = ((m + jku < n) ? m + jku : n) - 1;
                    for (jc = start_jc; jc >= 1 - jkl; jc--) {
                        extra = ZERO;
                        angle = TWOPI * rng_uniform(state);
                        c = cos(angle);
                        s = sin(angle);
                        irow = (1 > jc - jku + 1) ? 1 : jc - jku + 1;
                        if (jc > 0) {
                            il = ((m < jc + jkl + 1) ? m : jc + jkl + 1) + 1 - irow;
                            int aoff = (irow - 1) - iskew * (jc - 1) + ioffst - 1 + (jc - 1) * lda;
                            if (ioffst == 0 && iskew == 0) {
                                aoff = (irow - 1) + (jc - 1) * lda;
                            }
                            dlarot(0, 0, jc + jkl < m ? 1 : 0, il, c, s,
                                   &A[aoff], ilda, &dummy, &extra);
                        }

                        /* Chase "EXTRA" back down */
                        ic = jc;
                        for (jch = jc + jkl; jch <= iendch; jch += jkl + jku) {
                            ilextr = ic > 0 ? 1 : 0;
                            if (ilextr) {
                                int idx = (jch - 1) - iskew * (ic - 1) + ioffst - 1 + (ic - 1) * lda;
                                if (ioffst == 0 && iskew == 0) {
                                    idx = (jch - 1) + (ic - 1) * lda;
                                }
                                dlartg(A[idx], extra, &c, &s, &dummy);
                            }
                            ic = (1 > ic) ? 1 : ic;
                            icol = ((n - 1 < jch + jku) ? n - 1 : jch + jku);
                            iltemp = jch + jku < n ? 1 : 0;
                            temp = ZERO;
                            {
                                int aoff = (jch - 1) - iskew * (ic - 1) + ioffst - 1 + (ic - 1) * lda;
                                if (ioffst == 0 && iskew == 0) {
                                    aoff = (jch - 1) + (ic - 1) * lda;
                                }
                                dlarot(1, ilextr, iltemp, icol + 2 - ic, c, s,
                                       &A[aoff], ilda, &extra, &temp);
                            }
                            if (iltemp) {
                                int idx = (jch - 1) - iskew * (icol - 1) + ioffst - 1 + (icol - 1) * lda;
                                if (ioffst == 0 && iskew == 0) {
                                    idx = (jch - 1) + (icol - 1) * lda;
                                }
                                dlartg(A[idx], temp, &c, &s, &dummy);
                                il = ((iendch < jch + jkl + jku) ? iendch : jch + jkl + jku) + 2 - jch;
                                extra = ZERO;
                                {
                                    int aoff = (jch - 1) - iskew * (icol - 1) + ioffst - 1 + (icol - 1) * lda;
                                    if (ioffst == 0 && iskew == 0) {
                                        aoff = (jch - 1) + (icol - 1) * lda;
                                    }
                                    dlarot(0, 1, jch + jkl + jku <= iendch ? 1 : 0, il, c, s,
                                           &A[aoff], ilda, &temp, &extra);
                                }
                                ic = icol;
                            }
                        }
                    }
                }

                jku = uub;
                for (jkl = 1; jkl <= llb; jkl++) {
                    /* Transform from bandwidth JKL-1, JKU to JKL, JKU
                     * First row actually rotated is MIN(N+JKL, M)
                     * First column actually rotated is N */
                    iendch = ((n < m + jku) ? n : m + jku) - 1;
                    int start_jr = ((n + jkl < m) ? n + jkl : m) - 1;
                    for (jr = start_jr; jr >= 1 - jku; jr--) {
                        extra = ZERO;
                        angle = TWOPI * rng_uniform(state);
                        c = cos(angle);
                        s = sin(angle);
                        icol = (1 > jr - jkl + 1) ? 1 : jr - jkl + 1;
                        if (jr > 0) {
                            il = ((n < jr + jku + 1) ? n : jr + jku + 1) + 1 - icol;
                            int aoff = (jr - 1) - iskew * (icol - 1) + ioffst - 1 + (icol - 1) * lda;
                            if (ioffst == 0 && iskew == 0) {
                                aoff = (jr - 1) + (icol - 1) * lda;
                            }
                            dlarot(1, 0, jr + jku < n ? 1 : 0, il, c, s,
                                   &A[aoff], ilda, &dummy, &extra);
                        }

                        /* Chase "EXTRA" back down */
                        ir = jr;
                        for (jch = jr + jku; jch <= iendch; jch += jkl + jku) {
                            ilextr = ir > 0 ? 1 : 0;
                            if (ilextr) {
                                int idx = (ir - 1) - iskew * (jch - 1) + ioffst - 1 + (jch - 1) * lda;
                                if (ioffst == 0 && iskew == 0) {
                                    idx = (ir - 1) + (jch - 1) * lda;
                                }
                                dlartg(A[idx], extra, &c, &s, &dummy);
                            }
                            ir = (1 > ir) ? 1 : ir;
                            irow = ((m - 1 < jch + jkl) ? m - 1 : jch + jkl);
                            iltemp = jch + jkl < m ? 1 : 0;
                            temp = ZERO;
                            {
                                int aoff = (ir - 1) - iskew * (jch - 1) + ioffst - 1 + (jch - 1) * lda;
                                if (ioffst == 0 && iskew == 0) {
                                    aoff = (ir - 1) + (jch - 1) * lda;
                                }
                                dlarot(0, ilextr, iltemp, irow + 2 - ir, c, s,
                                       &A[aoff], ilda, &extra, &temp);
                            }
                            if (iltemp) {
                                int idx = (irow - 1) - iskew * (jch - 1) + ioffst - 1 + (jch - 1) * lda;
                                if (ioffst == 0 && iskew == 0) {
                                    idx = (irow - 1) + (jch - 1) * lda;
                                }
                                dlartg(A[idx], temp, &c, &s, &dummy);
                                il = ((iendch < jch + jkl + jku) ? iendch : jch + jkl + jku) + 2 - jch;
                                extra = ZERO;
                                {
                                    int aoff = (irow - 1) - iskew * (jch - 1) + ioffst - 1 + (jch - 1) * lda;
                                    if (ioffst == 0 && iskew == 0) {
                                        aoff = (irow - 1) + (jch - 1) * lda;
                                    }
                                    dlarot(1, 1, jch + jkl + jku <= iendch ? 1 : 0, il, c, s,
                                           &A[aoff], ilda, &temp, &extra);
                                }
                                ir = irow;
                            }
                        }
                    }
                }
            }

        } else {
            /* Symmetric -- A = U D U' */
            ipackg = ipack;
            ioffg = ioffst;

            if (topdwn) {
                /* Top-Down -- Generate Upper triangle only */
                if (ipack >= 5) {
                    ipackg = 6;
                    ioffg = uub + 1;
                } else {
                    ipackg = 1;
                }
                {
                    int diag_start = (1 - iskew) + ioffg - 1;
                    if (diag_start < 0) diag_start = 0;
                    cblas_dcopy(mnmin, d, 1, &A[diag_start], ilda + 1);
                }

                for (k = 1; k <= uub; k++) {
                    for (jc = 1; jc <= n - 1; jc++) {
                        irow = (1 > jc - k) ? 1 : jc - k;
                        il = ((jc + 1 < k + 2) ? jc + 1 : k + 2);
                        extra = ZERO;
                        {
                            int idx = (jc - 1) - iskew * (jc + 1 - 1) + ioffg - 1 + (jc + 1 - 1) * lda;
                            if (ioffg == 0 && iskew == 0) {
                                idx = (jc - 1) + jc * lda;
                            }
                            temp = A[idx];
                        }
                        angle = TWOPI * rng_uniform(state);
                        c = cos(angle);
                        s = sin(angle);
                        {
                            int aoff = (irow - 1) - iskew * (jc - 1) + ioffg - 1 + (jc - 1) * lda;
                            if (ioffg == 0 && iskew == 0) {
                                aoff = (irow - 1) + (jc - 1) * lda;
                            }
                            dlarot(0, jc > k ? 1 : 0, 1, il, c, s,
                                   &A[aoff], ilda, &extra, &temp);
                        }
                        {
                            int aoff = (1 - iskew) * jc + ioffg - 1 + (jc - 1) * lda;
                            if (ioffg == 0 && iskew == 0) {
                                aoff = (jc - 1) + (jc - 1) * lda;
                            }
                            int len = ((k < n - jc) ? k : n - jc) + 1;
                            dlarot(1, 1, 0, len, c, s,
                                   &A[aoff], ilda, &temp, &dummy);
                        }

                        /* Chase EXTRA back up the matrix */
                        icol = jc;
                        for (jch = jc - k; jch >= 1; jch -= k) {
                            {
                                int idx = (jch + 1 - 1) - iskew * (icol + 1 - 1) + ioffg - 1 + (icol + 1 - 1) * lda;
                                if (ioffg == 0 && iskew == 0) {
                                    idx = jch + icol * lda;
                                }
                                dlartg(A[idx], extra, &c, &s, &dummy);
                            }
                            {
                                int idx = (jch - 1) - iskew * (jch + 1 - 1) + ioffg - 1 + (jch + 1 - 1) * lda;
                                if (ioffg == 0 && iskew == 0) {
                                    idx = (jch - 1) + jch * lda;
                                }
                                temp = A[idx];
                            }
                            {
                                int aoff = (1 - iskew) * jch + ioffg - 1 + (jch - 1) * lda;
                                if (ioffg == 0 && iskew == 0) {
                                    aoff = (jch - 1) + (jch - 1) * lda;
                                }
                                dlarot(1, 1, 1, k + 2, c, -s,
                                       &A[aoff], ilda, &temp, &extra);
                            }
                            irow = (1 > jch - k) ? 1 : jch - k;
                            il = ((jch + 1 < k + 2) ? jch + 1 : k + 2);
                            extra = ZERO;
                            {
                                int aoff = (irow - 1) - iskew * (jch - 1) + ioffg - 1 + (jch - 1) * lda;
                                if (ioffg == 0 && iskew == 0) {
                                    aoff = (irow - 1) + (jch - 1) * lda;
                                }
                                dlarot(0, jch > k ? 1 : 0, 1, il, c, -s,
                                       &A[aoff], ilda, &extra, &temp);
                            }
                            icol = jch;
                        }
                    }
                }

                /* If we need lower triangle, copy from upper.
                 * Note that the order of copying is chosen to work for 'q' -> 'b' */
                if (ipack != ipackg && ipack != 3) {
                    for (jc = 1; jc <= n; jc++) {
                        irow = ioffst - iskew * jc;
                        int jr_end = ((n < jc + uub) ? n : jc + uub);
                        for (jr = jc; jr <= jr_end; jr++) {
                            /* A[jr + irow - 1 + (jc-1)*lda] = A[(jc-1) - iskew*(jr-1) + ioffg - 1 + (jr-1)*lda] */
                            int dest = jr + irow - 1 + (jc - 1) * lda;
                            int src = (jc - 1) - iskew * (jr - 1) + ioffg - 1 + (jr - 1) * lda;
                            if (ioffst == 0 && iskew == 0) {
                                dest = (jr - 1) + (jc - 1) * lda;
                            }
                            if (ioffg == 0 && iskew == 0) {
                                src = (jc - 1) + (jr - 1) * lda;
                            }
                            A[dest] = A[src];
                        }
                    }
                    if (ipack == 5) {
                        for (jc = n - uub + 1; jc <= n; jc++) {
                            for (jr = n + 2 - jc; jr <= uub + 1; jr++) {
                                A[(jr - 1) + (jc - 1) * lda] = ZERO;
                            }
                        }
                    }
                    if (ipackg == 6) {
                        ipackg = ipack;
                    } else {
                        ipackg = 0;
                    }
                }

            } else {
                /* Bottom-Up -- Generate Lower triangle only */
                if (ipack >= 5) {
                    ipackg = 5;
                    if (ipack == 6) {
                        ioffg = 1;
                    }
                } else {
                    ipackg = 2;
                }
                {
                    int diag_start = (1 - iskew) + ioffg - 1;
                    if (diag_start < 0) diag_start = 0;
                    cblas_dcopy(mnmin, d, 1, &A[diag_start], ilda + 1);
                }

                for (k = 1; k <= uub; k++) {
                    for (jc = n - 1; jc >= 1; jc--) {
                        il = ((n + 1 - jc < k + 2) ? n + 1 - jc : k + 2);
                        extra = ZERO;
                        {
                            int idx = 1 + (1 - iskew) * jc + ioffg - 1 + (jc - 1) * lda;
                            if (ioffg == 0 && iskew == 0) {
                                idx = jc + (jc - 1) * lda;
                            }
                            temp = A[idx];
                        }
                        angle = TWOPI * rng_uniform(state);
                        c = cos(angle);
                        s = -sin(angle);
                        {
                            int aoff = (1 - iskew) * jc + ioffg - 1 + (jc - 1) * lda;
                            if (ioffg == 0 && iskew == 0) {
                                aoff = (jc - 1) + (jc - 1) * lda;
                            }
                            dlarot(0, 1, n - jc > k ? 1 : 0, il, c, s,
                                   &A[aoff], ilda, &temp, &extra);
                        }
                        icol = (1 > jc - k + 1) ? 1 : jc - k + 1;
                        {
                            int aoff = (jc - 1) - iskew * (icol - 1) + ioffg - 1 + (icol - 1) * lda;
                            if (ioffg == 0 && iskew == 0) {
                                aoff = (jc - 1) + (icol - 1) * lda;
                            }
                            dlarot(1, 0, 1, jc + 2 - icol, c, s,
                                   &A[aoff], ilda, &dummy, &temp);
                        }

                        /* Chase EXTRA back down the matrix */
                        icol = jc;
                        for (jch = jc + k; jch <= n - 1; jch += k) {
                            {
                                int idx = (jch - 1) - iskew * (icol - 1) + ioffg - 1 + (icol - 1) * lda;
                                if (ioffg == 0 && iskew == 0) {
                                    idx = (jch - 1) + (icol - 1) * lda;
                                }
                                dlartg(A[idx], extra, &c, &s, &dummy);
                            }
                            {
                                int idx = 1 + (1 - iskew) * jch + ioffg - 1 + (jch - 1) * lda;
                                if (ioffg == 0 && iskew == 0) {
                                    idx = jch + (jch - 1) * lda;
                                }
                                temp = A[idx];
                            }
                            {
                                int aoff = (jch - 1) - iskew * (icol - 1) + ioffg - 1 + (icol - 1) * lda;
                                if (ioffg == 0 && iskew == 0) {
                                    aoff = (jch - 1) + (icol - 1) * lda;
                                }
                                dlarot(1, 1, 1, k + 2, c, s,
                                       &A[aoff], ilda, &extra, &temp);
                            }
                            il = ((n + 1 - jch < k + 2) ? n + 1 - jch : k + 2);
                            extra = ZERO;
                            {
                                int aoff = (1 - iskew) * jch + ioffg - 1 + (jch - 1) * lda;
                                if (ioffg == 0 && iskew == 0) {
                                    aoff = (jch - 1) + (jch - 1) * lda;
                                }
                                dlarot(0, 1, n - jch > k ? 1 : 0, il, c, s,
                                       &A[aoff], ilda, &temp, &extra);
                            }
                            icol = jch;
                        }
                    }
                }

                /* If we need upper triangle, copy from lower.
                 * Note that the order of copying is chosen to work for 'b' -> 'q' */
                if (ipack != ipackg && ipack != 4) {
                    for (jc = n; jc >= 1; jc--) {
                        irow = ioffst - iskew * jc;
                        int jr_start = ((1 > jc - uub) ? 1 : jc - uub);
                        for (jr = jc; jr >= jr_start; jr--) {
                            int dest = jr + irow - 1 + (jc - 1) * lda;
                            int src = (jc - 1) - iskew * (jr - 1) + ioffg - 1 + (jr - 1) * lda;
                            if (ioffst == 0 && iskew == 0) {
                                dest = (jr - 1) + (jc - 1) * lda;
                            }
                            if (ioffg == 0 && iskew == 0) {
                                src = (jc - 1) + (jr - 1) * lda;
                            }
                            A[dest] = A[src];
                        }
                    }
                    if (ipack == 6) {
                        for (jc = 1; jc <= uub; jc++) {
                            for (jr = 1; jr <= uub + 1 - jc; jr++) {
                                A[(jr - 1) + (jc - 1) * lda] = ZERO;
                            }
                        }
                    }
                    if (ipackg == 5) {
                        ipackg = ipack;
                    } else {
                        ipackg = 0;
                    }
                }
            }
        }

    } else {
        /* 4) Generate Banded Matrix by first
         *    Rotating by random Unitary matrices,
         *    then reducing the bandwidth using Householder
         *    transformations.
         *
         *    Note: we should get here only if LDA >= N */

        if (isym == 1) {
            /* Non-symmetric -- A = U D V */
            dlagge(mr, nc, llb, uub, d, A, lda, work, &iinfo, state);
        } else {
            /* Symmetric -- A = U D U' */
            dlagsy(m, llb, d, A, lda, work, &iinfo, state);
        }
        if (iinfo != 0) {
            *info = 3;
            return;
        }
    }

    /* 5) Pack the matrix */
    if (ipack != ipackg) {
        if (ipack == 1) {
            /* 'U' -- Upper triangular, not packed */
            for (j = 1; j <= m; j++) {
                for (i = j + 1; i <= m; i++) {
                    A[(i - 1) + (j - 1) * lda] = ZERO;
                }
            }

        } else if (ipack == 2) {
            /* 'L' -- Lower triangular, not packed */
            for (j = 2; j <= m; j++) {
                for (i = 1; i <= j - 1; i++) {
                    A[(i - 1) + (j - 1) * lda] = ZERO;
                }
            }

        } else if (ipack == 3) {
            /* 'C' -- Upper triangle packed Columnwise. */
            icol = 1;
            irow = 0;
            for (j = 1; j <= m; j++) {
                for (i = 1; i <= j; i++) {
                    irow = irow + 1;
                    if (irow > lda) {
                        irow = 1;
                        icol = icol + 1;
                    }
                    A[(irow - 1) + (icol - 1) * lda] = A[(i - 1) + (j - 1) * lda];
                }
            }

        } else if (ipack == 4) {
            /* 'R' -- Lower triangle packed Columnwise. */
            icol = 1;
            irow = 0;
            for (j = 1; j <= m; j++) {
                for (i = j; i <= m; i++) {
                    irow = irow + 1;
                    if (irow > lda) {
                        irow = 1;
                        icol = icol + 1;
                    }
                    A[(irow - 1) + (icol - 1) * lda] = A[(i - 1) + (j - 1) * lda];
                }
            }

        } else if (ipack >= 5) {
            /* 'B' -- The lower triangle is packed as a band matrix.
             * 'Q' -- The upper triangle is packed as a band matrix.
             * 'Z' -- The whole matrix is packed as a band matrix. */

            if (ipack == 5) {
                uub = 0;
            }
            if (ipack == 6) {
                llb = 0;
            }

            for (j = 1; j <= uub; j++) {
                int i_start = ((j + llb < m) ? j + llb : m);
                for (i = i_start; i >= 1; i--) {
                    A[(i - j + uub + 1 - 1) + (j - 1) * lda] = A[(i - 1) + (j - 1) * lda];
                }
            }

            for (j = uub + 2; j <= n; j++) {
                int i_end = ((j + llb < m) ? j + llb : m);
                for (i = j - uub; i <= i_end; i++) {
                    A[(i - j + uub + 1 - 1) + (j - 1) * lda] = A[(i - 1) + (j - 1) * lda];
                }
            }
        }

        /* If packed, zero out extraneous elements. */

        /* Symmetric/Triangular Packed --
         * zero out everything after A(IROW,ICOL) */
        if (ipack == 3 || ipack == 4) {
            for (jc = icol; jc <= m; jc++) {
                for (jr = irow + 1; jr <= lda; jr++) {
                    A[(jr - 1) + (jc - 1) * lda] = ZERO;
                }
                irow = 0;
            }

        } else if (ipack >= 5) {
            /* Packed Band --
             *   1st row is now in A(UUB+2-j, j), zero above it
             *   m-th row is now in A(M+UUB-j, j), zero below it
             *   last non-zero diagonal is now in A(UUB+LLB+1, j),
             *      zero below it, too. */
            ir1 = uub + llb + 2;
            ir2 = uub + m + 2;
            for (jc = 1; jc <= n; jc++) {
                for (jr = 1; jr <= uub + 1 - jc; jr++) {
                    A[(jr - 1) + (jc - 1) * lda] = ZERO;
                }
                int jr_start = (1 > ((ir1 < ir2 - jc) ? ir1 : ir2 - jc)) ?
                               1 : ((ir1 < ir2 - jc) ? ir1 : ir2 - jc);
                for (jr = jr_start; jr <= lda; jr++) {
                    A[(jr - 1) + (jc - 1) * lda] = ZERO;
                }
            }
        }
    }
}
