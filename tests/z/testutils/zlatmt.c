/**
 * @file zlatmt.c
 * @brief ZLATMT generates random matrices with specified singular values
 *        (or hermitian with specified eigenvalues) for testing LAPACK programs.
 *
 * Port of LAPACK TESTING/MATGEN/zlatmt.f to C.
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"
#include "test_rng.h"

extern void xerbla(const char* srname, const INT info);

void zlatmt(
    const INT m,
    const INT n,
    const char* dist,
    const char* sym,
    f64* d,
    const INT mode,
    const f64 cond,
    const f64 dmax_,
    const INT rank,
    const INT kl,
    const INT ku,
    const char* pack,
    c128* A,
    const INT lda,
    c128* work,
    INT* info,
    uint64_t state[static 4])
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const c128 CZERO = CMPLX(0.0, 0.0);
    const f64 TWOPI = 6.28318530717958647692528676655900576839;

    INT givens, zsym = 0;
    INT i, ic, icol = 0, idist, iendch, iinfo, il, ilda,
        ioffg, ioffg0, ioffst, ioffst0, ipack, ipackg, ir, ir1, ir2,
        irow = 0, irsign = 0, iskew, isym, isympk, j, jc, jch,
        jkl, jku, jr, k, llb, minlda, mnmin, mr, nc,
        uub;
    INT topdwn, ilextr, iltemp;
    f64 alpha, angle, realc, temp;
    c128 c, ct, ctemp, dummy, extra, s, st;

    *info = 0;

    if (m == 0 || n == 0)
        return;

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
        zsym = 0;
    } else if (sym[0] == 'P' || sym[0] == 'p') {
        isym = 2;
        irsign = 0;
        zsym = 0;
    } else if (sym[0] == 'S' || sym[0] == 's') {
        isym = 2;
        irsign = 0;
        zsym = 1;
    } else if (sym[0] == 'H' || sym[0] == 'h') {
        isym = 2;
        irsign = 1;
        zsym = 0;
    } else {
        isym = -1;
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

    givens = 0;
    if (isym == 1) {
        if ((f64)(llb + uub) < 0.3 * (f64)((1 > mr + nc) ? 1 : mr + nc))
            givens = 1;
    } else {
        if (2 * llb < m)
            givens = 1;
    }
    if (lda < m && lda >= minlda)
        givens = 1;

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
    } else if (mode != 0 && mode != 6 && mode != -6 && cond < ONE) {
        *info = -8;
    } else if (kl < 0) {
        *info = -11;
    } else if (ku < 0 || (isym != 1 && kl != ku)) {
        *info = -12;
    } else if (ipack == -1 || (isympk == 1 && isym == 1) ||
               (isympk == 2 && isym == 1 && kl > 0) ||
               (isympk == 3 && isym == 1 && ku > 0) ||
               (isympk != 0 && m != n)) {
        *info = -13;
    } else if (lda < (1 > minlda ? 1 : minlda)) {
        *info = -15;
    }

    if (*info != 0) {
        xerbla("ZLATMT", -(*info));
        return;
    }

    /* 2) Set up D -- eigenvalues are always real */
    dlatm7(mode, cond, irsign, idist, d, mnmin, rank, &iinfo, state);
    if (iinfo != 0) {
        *info = 1;
        return;
    }

    topdwn = (fabs(d[0]) <= fabs(d[rank - 1]));

    if (mode != 0 && mode != 6 && mode != -6) {
        temp = fabs(d[0]);
        for (i = 1; i < rank; i++) {
            if (fabs(d[i]) > temp) temp = fabs(d[i]);
        }
        if (temp > ZERO) {
            alpha = dmax_ / temp;
        } else {
            *info = 2;
            return;
        }
        cblas_dscal(rank, alpha, d, 1);
    }

    zlaset("F", lda, n, CZERO, CZERO, A, lda);

    /*
     * 3) Generate Banded Matrix using Givens rotations.
     *    Also the special case of UUB=LLB=0
     *
     *    Compute Addressing constants to cover all storage formats.
     *    Whether GE, HE, SY, GB, HB, or SB, upper or lower triangle
     *    or both, the (i,j)-th element is in
     *    A( i - ISKEW*j + IOFFST, j )   [Fortran 1-based]
     *
     *    For 0-based (i0, j0):
     *    flat = i0 - iskew*j0 + ioffst0 + j0*lda
     *    where ioffst0 = ioffst - iskew
     */

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
    ioffst0 = ioffst - iskew;

    ipackg = 0;

/*
 * AIND(i, j): flat index for 0-based logical position (i, j)
 * AINDG(i, j): same but with ioffg0 (for symmetric generation addressing)
 */
#define AIND(i, j)  ((i) - iskew*(j) + ioffst0 + (j)*lda)
#define AINDG(i, j) ((i) - iskew*(j) + ioffg0 + (j)*lda)

    if (llb == 0 && uub == 0) {
        /* Diagonal Matrix: place real D on the diagonal */
        for (j = 0; j < mnmin; j++)
            A[AIND(j, j)] = CMPLX(d[j], 0.0);

        if (ipack <= 2 || ipack >= 5)
            ipackg = ipack;

    } else if (givens) {

        if (isym == 1) {
            /* Non-symmetric -- A = U D V */
            if (ipack > 4) {
                ipackg = ipack;
            } else {
                ipackg = 0;
            }

            for (j = 0; j < mnmin; j++)
                A[AIND(j, j)] = CMPLX(d[j], 0.0);

            if (topdwn) {
                /* Top-Down -- expand upper bandwidth first, then lower */
                jkl = 0;
                for (jku = 1; jku <= uub; jku++) {

                    for (jr = 0; jr < ((m + jku < n) ? m + jku : n) + jkl - 1; jr++) {
                        extra = CZERO;
                        angle = TWOPI * rng_uniform(state);
                        c = cos(angle) * zlarnd_rng(5, state);
                        s = sin(angle) * zlarnd_rng(5, state);
                        icol = (jr - jkl > 0) ? jr - jkl : 0;
                        if (jr < m - 1) {
                            il = ((n - 1 < jr + jku) ? n - 1 : jr + jku) + 1 - icol;
                            zlarot(1, jr >= jkl, 0, il,
                                   c, s, &A[AIND(jr, icol)],
                                   ilda, &extra, &dummy);
                        }

                        ir = jr;
                        ic = icol;
                        for (jch = jr - jkl; jch >= 0; jch -= jkl + jku) {
                            if (ir < m - 1) {
                                zlartg(A[AIND(ir + 1, ic + 1)],
                                       extra, &realc, &s, &dummy);
                                dummy = zlarnd_rng(5, state);
                                c = conj(realc * dummy);
                                s = conj(-s * dummy);
                            }
                            irow = (jch - jku > 0) ? jch - jku : 0;
                            il = ir + 2 - irow;
                            ctemp = CZERO;
                            iltemp = (jch >= jku);
                            zlarot(0, iltemp, 1, il, c,
                                   s, &A[AIND(irow, ic)],
                                   ilda, &ctemp, &extra);
                            if (iltemp) {
                                zlartg(A[AIND(irow + 1, ic + 1)],
                                       ctemp, &realc, &s, &dummy);
                                dummy = zlarnd_rng(5, state);
                                c = conj(realc * dummy);
                                s = conj(-s * dummy);

                                icol = (jch - jku - jkl > 0) ? jch - jku - jkl : 0;
                                il = ic + 2 - icol;
                                extra = CZERO;
                                zlarot(1, jch >= jku + jkl,
                                       1, il, c, s,
                                       &A[AIND(irow, icol)],
                                       ilda, &extra, &ctemp);
                                ic = icol;
                                ir = irow;
                            }
                        }
                    }
                }

                jku = uub;
                for (jkl = 1; jkl <= llb; jkl++) {

                    for (jc = 0; jc < ((n + jkl < m) ? n + jkl : m) + jku - 1; jc++) {
                        extra = CZERO;
                        angle = TWOPI * rng_uniform(state);
                        c = cos(angle) * zlarnd_rng(5, state);
                        s = sin(angle) * zlarnd_rng(5, state);
                        irow = (jc - jku > 0) ? jc - jku : 0;
                        if (jc < n - 1) {
                            il = ((m - 1 < jc + jkl) ? m - 1 : jc + jkl) + 1 - irow;
                            zlarot(0, jc >= jku, 0, il,
                                   c, s, &A[AIND(irow, jc)],
                                   ilda, &extra, &dummy);
                        }

                        ic = jc;
                        ir = irow;
                        for (jch = jc - jku; jch >= 0; jch -= jkl + jku) {
                            if (ic < n - 1) {
                                zlartg(A[AIND(ir + 1, ic + 1)],
                                       extra, &realc, &s, &dummy);
                                dummy = zlarnd_rng(5, state);
                                c = conj(realc * dummy);
                                s = conj(-s * dummy);
                            }
                            icol = (jch - jkl > 0) ? jch - jkl : 0;
                            il = ic + 2 - icol;
                            ctemp = CZERO;
                            iltemp = (jch >= jkl);
                            zlarot(1, iltemp, 1, il, c,
                                   s, &A[AIND(ir, icol)],
                                   ilda, &ctemp, &extra);
                            if (iltemp) {
                                zlartg(A[AIND(ir + 1, icol + 1)],
                                       ctemp, &realc, &s, &dummy);
                                dummy = zlarnd_rng(5, state);
                                c = conj(realc * dummy);
                                s = conj(-s * dummy);
                                irow = (jch - jkl - jku > 0) ? jch - jkl - jku : 0;
                                il = ir + 2 - irow;
                                extra = CZERO;
                                zlarot(0, jch >= jkl + jku,
                                       1, il, c, s,
                                       &A[AIND(irow, icol)],
                                       ilda, &extra, &ctemp);
                                ic = icol;
                                ir = irow;
                            }
                        }
                    }
                }

            } else {
                /* Bottom-Up */
                jkl = 0;
                for (jku = 1; jku <= uub; jku++) {

                    iendch = ((m < n + jkl) ? m : n + jkl) - 1;
                    for (jc = ((m + jku < n) ? m + jku : n) - 2; jc >= -jkl; jc--) {
                        extra = CZERO;
                        angle = TWOPI * rng_uniform(state);
                        c = cos(angle) * zlarnd_rng(5, state);
                        s = sin(angle) * zlarnd_rng(5, state);
                        irow = (jc - jku + 1 > 0) ? jc - jku + 1 : 0;
                        if (jc >= 0) {
                            il = ((m - 1 < jc + jkl + 1) ? m - 1 : jc + jkl + 1) + 1 - irow;
                            zlarot(0, 0, jc + jkl < m - 1,
                                   il, c, s,
                                   &A[AIND(irow, jc)],
                                   ilda, &dummy, &extra);
                        }

                        ic = jc;
                        for (jch = jc + jkl; jch <= iendch; jch += jkl + jku) {
                            ilextr = (ic >= 0);
                            if (ilextr) {
                                zlartg(A[AIND(jch, ic)],
                                       extra, &realc, &s, &dummy);
                                dummy = zlarnd_rng(5, state);
                                c = realc * dummy;
                                s = s * dummy;
                            }
                            ic = (ic > 0) ? ic : 0;
                            icol = (n - 2 < jch + jku) ? n - 2 : jch + jku;
                            iltemp = (jch + jku < n - 1);
                            ctemp = CZERO;
                            zlarot(1, ilextr, iltemp,
                                   icol + 2 - ic, c, s,
                                   &A[AIND(jch, ic)],
                                   ilda, &extra, &ctemp);
                            if (iltemp) {
                                zlartg(A[AIND(jch, icol)],
                                       ctemp, &realc, &s, &dummy);
                                dummy = zlarnd_rng(5, state);
                                c = realc * dummy;
                                s = s * dummy;
                                il = ((iendch < jch + jkl + jku) ? iendch : jch + jkl + jku) + 2 - jch;
                                extra = CZERO;
                                zlarot(0, 1,
                                       jch + jkl + jku <= iendch, il, c, s,
                                       &A[AIND(jch, icol)],
                                       ilda, &ctemp, &extra);
                                ic = icol;
                            }
                        }
                    }
                }

                jku = uub;
                for (jkl = 1; jkl <= llb; jkl++) {

                    iendch = ((n < m + jku) ? n : m + jku) - 1;
                    for (jr = ((n + jkl < m) ? n + jkl : m) - 2; jr >= -jku; jr--) {
                        extra = CZERO;
                        angle = TWOPI * rng_uniform(state);
                        c = cos(angle) * zlarnd_rng(5, state);
                        s = sin(angle) * zlarnd_rng(5, state);
                        icol = (jr - jkl + 1 > 0) ? jr - jkl + 1 : 0;
                        if (jr >= 0) {
                            il = ((n - 1 < jr + jku + 1) ? n - 1 : jr + jku + 1) + 1 - icol;
                            zlarot(1, 0, jr + jku < n - 1,
                                   il, c, s,
                                   &A[AIND(jr, icol)],
                                   ilda, &dummy, &extra);
                        }

                        ir = jr;
                        for (jch = jr + jku; jch <= iendch; jch += jkl + jku) {
                            ilextr = (ir >= 0);
                            if (ilextr) {
                                zlartg(A[AIND(ir, jch)],
                                       extra, &realc, &s, &dummy);
                                dummy = zlarnd_rng(5, state);
                                c = realc * dummy;
                                s = s * dummy;
                            }
                            ir = (ir > 0) ? ir : 0;
                            irow = (m - 2 < jch + jkl) ? m - 2 : jch + jkl;
                            iltemp = (jch + jkl < m - 1);
                            ctemp = CZERO;
                            zlarot(0, ilextr, iltemp,
                                   irow + 2 - ir, c, s,
                                   &A[AIND(ir, jch)],
                                   ilda, &extra, &ctemp);
                            if (iltemp) {
                                zlartg(A[AIND(irow, jch)],
                                       ctemp, &realc, &s, &dummy);
                                dummy = zlarnd_rng(5, state);
                                c = realc * dummy;
                                s = s * dummy;
                                il = ((iendch < jch + jkl + jku) ? iendch : jch + jkl + jku) + 2 - jch;
                                extra = CZERO;
                                zlarot(1, 1,
                                       jch + jkl + jku <= iendch, il, c, s,
                                       &A[AIND(irow, jch)],
                                       ilda, &ctemp, &extra);
                                ir = irow;
                            }
                        }
                    }
                }
            }

        } else {
            /* Symmetric -- A = U D U'  (zsym)
             * Hermitian -- A = U D U*  (!zsym) */

            ipackg = ipack;
            ioffg = ioffst;
            ioffg0 = ioffg - iskew;

            if (topdwn) {
                /* Top-Down -- Generate Upper triangle only */
                if (ipack >= 5) {
                    ipackg = 6;
                    ioffg = uub + 1;
                    ioffg0 = ioffg - iskew;
                } else {
                    ipackg = 1;
                }

                for (j = 0; j < mnmin; j++)
                    A[AINDG(j, j)] = CMPLX(d[j], 0.0);

                for (k = 0; k < uub; k++) {
                    for (jc = 0; jc < n - 1; jc++) {
                        irow = (jc - k - 1 > 0) ? jc - k - 1 : 0;
                        il = (jc + 2 < k + 3) ? jc + 2 : k + 3;
                        extra = CZERO;
                        ctemp = A[AINDG(jc, jc + 1)];
                        angle = TWOPI * rng_uniform(state);
                        c = cos(angle) * zlarnd_rng(5, state);
                        s = sin(angle) * zlarnd_rng(5, state);
                        if (zsym) {
                            ct = c;
                            st = s;
                        } else {
                            ctemp = conj(ctemp);
                            ct = conj(c);
                            st = conj(s);
                        }
                        zlarot(0, jc > k, 1, il, c, s,
                               &A[AINDG(irow, jc)], ilda,
                               &extra, &ctemp);
                        zlarot(1, 1, 0,
                               ((k + 1 < n - 1 - jc) ? k + 1 : n - 1 - jc) + 1, ct, st,
                               &A[AINDG(jc, jc)], ilda,
                               &ctemp, &dummy);

                        icol = jc;
                        for (jch = jc - k - 1; jch >= 0; jch -= k + 1) {
                            zlartg(A[AINDG(jch + 1, icol + 1)],
                                   extra, &realc, &s, &dummy);
                            dummy = zlarnd_rng(5, state);
                            c = conj(realc * dummy);
                            s = conj(-s * dummy);
                            ctemp = A[AINDG(jch, jch + 1)];
                            if (zsym) {
                                ct = c;
                                st = s;
                            } else {
                                ctemp = conj(ctemp);
                                ct = conj(c);
                                st = conj(s);
                            }
                            zlarot(1, 1, 1, k + 3, c,
                                   s,
                                   &A[AINDG(jch, jch)], ilda,
                                   &ctemp, &extra);
                            irow = (jch - k - 1 > 0) ? jch - k - 1 : 0;
                            il = (jch + 2 < k + 3) ? jch + 2 : k + 3;
                            extra = CZERO;
                            zlarot(0, jch > k, 1, il,
                                   ct, st,
                                   &A[AINDG(irow, jch)], ilda,
                                   &extra, &ctemp);
                            icol = jch;
                        }
                    }
                }

                if (ipack != ipackg && ipack != 3) {
                    for (jc = 0; jc < n; jc++) {
                        for (jr = jc; jr <= (n - 1 < jc + uub ? n - 1 : jc + uub); jr++) {
                            if (zsym)
                                A[AIND(jr, jc)] = A[AINDG(jc, jr)];
                            else
                                A[AIND(jr, jc)] = conj(A[AINDG(jc, jr)]);
                        }
                    }
                    if (ipack == 5) {
                        for (jc = n - uub; jc < n; jc++) {
                            for (jr = n - jc; jr <= uub; jr++) {
                                A[jr + jc * lda] = CZERO;
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
                        ioffg0 = ioffg - iskew;
                    }
                } else {
                    ipackg = 2;
                }

                for (j = 0; j < mnmin; j++)
                    A[AINDG(j, j)] = CMPLX(d[j], 0.0);

                for (k = 0; k < uub; k++) {
                    for (jc = n - 2; jc >= 0; jc--) {
                        il = (n - jc < k + 3) ? n - jc : k + 3;
                        extra = CZERO;
                        ctemp = A[AINDG(jc + 1, jc)];
                        angle = TWOPI * rng_uniform(state);
                        c = cos(angle) * zlarnd_rng(5, state);
                        s = sin(angle) * zlarnd_rng(5, state);
                        if (zsym) {
                            ct = c;
                            st = s;
                        } else {
                            ctemp = conj(ctemp);
                            ct = conj(c);
                            st = conj(s);
                        }
                        zlarot(0, 1, n - 1 - jc > k + 1, il, c,
                               s,
                               &A[AINDG(jc, jc)], ilda,
                               &ctemp, &extra);
                        icol = (jc - k > 0) ? jc - k : 0;
                        zlarot(1, 0, 1, jc + 2 - icol,
                               ct, st,
                               &A[AINDG(jc, icol)], ilda,
                               &dummy, &ctemp);

                        icol = jc;
                        for (jch = jc + k + 1; jch <= n - 2; jch += k + 1) {
                            zlartg(A[AINDG(jch, icol)],
                                   extra, &realc, &s, &dummy);
                            dummy = zlarnd_rng(5, state);
                            c = realc * dummy;
                            s = s * dummy;
                            ctemp = A[AINDG(jch + 1, jch)];
                            if (zsym) {
                                ct = c;
                                st = s;
                            } else {
                                ctemp = conj(ctemp);
                                ct = conj(c);
                                st = conj(s);
                            }
                            zlarot(1, 1, 1, k + 3, c,
                                   s,
                                   &A[AINDG(jch, icol)], ilda,
                                   &extra, &ctemp);
                            il = (n - jch < k + 3) ? n - jch : k + 3;
                            extra = CZERO;
                            zlarot(0, 1, n - 1 - jch > k + 1, il,
                                   ct, st,
                                   &A[AINDG(jch, jch)], ilda,
                                   &ctemp, &extra);
                            icol = jch;
                        }
                    }
                }

                if (ipack != ipackg && ipack != 4) {
                    for (jc = n - 1; jc >= 0; jc--) {
                        for (jr = jc; jr >= (jc - uub > 0 ? jc - uub : 0); jr--) {
                            if (zsym)
                                A[AIND(jr, jc)] = A[AINDG(jc, jr)];
                            else
                                A[AIND(jr, jc)] = conj(A[AINDG(jc, jr)]);
                        }
                    }
                    if (ipack == 6) {
                        for (jc = 0; jc < uub; jc++) {
                            for (jr = 0; jr < uub - jc; jr++) {
                                A[jr + jc * lda] = CZERO;
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

            /* Ensure that the diagonal is real if Hermitian */
            if (!zsym) {
                for (jc = 0; jc < n; jc++) {
                    INT idx = ioffst0 + (1 - iskew) * jc + jc * lda;
                    A[idx] = CMPLX(creal(A[idx]), 0.0);
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
            zlagge(mr, nc, llb, uub, d, A, lda, work, &iinfo, state);
        } else {
            if (zsym)
                zlagsy(m, llb, d, A, lda, work, &iinfo, state);
            else
                zlaghe(m, llb, d, A, lda, work, &iinfo, state);
        }
        if (iinfo != 0) {
            *info = 3;
            return;
        }
    }

    /* 5) Pack the matrix */
    if (ipack != ipackg) {
        if (ipack == 1) {
            for (j = 0; j < m; j++) {
                for (i = j + 1; i < m; i++) {
                    A[i + j * lda] = CZERO;
                }
            }

        } else if (ipack == 2) {
            for (j = 1; j < m; j++) {
                for (i = 0; i < j; i++) {
                    A[i + j * lda] = CZERO;
                }
            }

        } else if (ipack == 3) {
            icol = 0;
            irow = -1;
            for (j = 0; j < m; j++) {
                for (i = 0; i <= j; i++) {
                    irow++;
                    if (irow >= lda) {
                        irow = 0;
                        icol++;
                    }
                    A[irow + icol * lda] = A[i + j * lda];
                }
            }

        } else if (ipack == 4) {
            icol = 0;
            irow = -1;
            for (j = 0; j < m; j++) {
                for (i = j; i < m; i++) {
                    irow++;
                    if (irow >= lda) {
                        irow = 0;
                        icol++;
                    }
                    A[irow + icol * lda] = A[i + j * lda];
                }
            }

        } else if (ipack >= 5) {
            if (ipack == 5)
                uub = 0;
            if (ipack == 6)
                llb = 0;

            for (j = 0; j < uub; j++) {
                for (i = ((j + llb < m - 1) ? j + llb : m - 1); i >= 0; i--) {
                    A[(i - j + uub) + j * lda] = A[i + j * lda];
                }
            }

            for (j = uub + 1; j < n; j++) {
                for (i = j - uub; i <= ((j + llb < m - 1) ? j + llb : m - 1); i++) {
                    A[(i - j + uub) + j * lda] = A[i + j * lda];
                }
            }
        }

        if (ipack == 3 || ipack == 4) {
            for (jc = icol; jc < m; jc++) {
                for (jr = irow + 1; jr < lda; jr++) {
                    A[jr + jc * lda] = CZERO;
                }
                irow = -1;
            }

        } else if (ipack >= 5) {
            ir1 = uub + llb + 1;
            ir2 = uub + m + 1;
            for (jc = 0; jc < n; jc++) {
                for (jr = 0; jr < uub - jc; jr++) {
                    A[jr + jc * lda] = CZERO;
                }
                for (jr = ((ir1 < ir2 - jc - 1) ? ir1 : ir2 - jc - 1);
                     jr < lda; jr++) {
                    A[jr + jc * lda] = CZERO;
                }
            }
        }
    }

#undef AIND
#undef AINDG
}
