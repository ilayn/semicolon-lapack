/**
 * @file slatms.c
 * @brief SLATMS generates random matrices with specified singular values
 *        (or symmetric/hermitian with specified eigenvalues) for testing
 *        LAPACK programs.
 *
 * Port of LAPACK TESTING/MATGEN/slatms.f to C.
 */

#include <math.h>
#include <stdlib.h>
#include "semicolon_cblas.h"
#include "verify.h"
#include <string.h>
#include "test_rng.h"


void slatms(
    const INT m,
    const INT n,
    const char* dist,
    const char* sym,
    f32* d,
    const INT mode,
    const f32 cond,
    const f32 dmax,
    const INT kl,
    const INT ku,
    const char* pack,
    f32* A,
    const INT lda,
    f32* work,
    INT* info,
    uint64_t state[static 4])
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;
    const f32 TWOPI = 6.28318530717958647692528676655900576839f;

    INT givens;
    INT i, ic, icol = 0, idist, iendch, iinfo, il, ilda,
        ioffg, ioffst, ipack, ipackg, ir, ir1, ir2,
        irow = 0, irsign = 0, iskew, isym, isympk, j, jc, jch,
        jkl, jku, jr, k, llb, minlda, mnmin, mr, nc,
        uub;
    INT topdwn, ilextr, iltemp;
    f32 alpha, angle, c, dummy, extra, s, temp;

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
    } else if (sym[0] == 'P' || sym[0] == 'p') {
        isym = 2;
        irsign = 0;
    } else if (sym[0] == 'S' || sym[0] == 's' ||
               sym[0] == 'H' || sym[0] == 'h') {
        isym = 2;
        irsign = 1;
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
        if ((f32)(llb + uub) < 0.3f * (f32)((1 > mr + nc) ? 1 : mr + nc))
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
    } else if (lda < (1 > minlda ? 1 : minlda)) {
        *info = -14;
    }

    if (*info != 0) {
        xerbla("SLATMS", -(*info));
        return;
    }

    /* 2) Set up D */
    slatm1(mode, cond, irsign, idist, d, mnmin, &iinfo, state);
    if (iinfo != 0) {
        *info = 1;
        return;
    }

    topdwn = (fabsf(d[0]) <= fabsf(d[mnmin - 1]));

    if (mode != 0 && abs(mode) != 6) {
        temp = fabsf(d[0]);
        for (i = 1; i < mnmin; i++) {
            if (fabsf(d[i]) > temp) temp = fabsf(d[i]);
        }
        if (temp > ZERO) {
            alpha = dmax / temp;
        } else {
            *info = 2;
            return;
        }
        cblas_sscal(mnmin, alpha, d, 1);
    }

    /* 3) Generate Banded Matrix using Givens rotations.
     *    Also the special case of UUB=LLB=0
     *
     *    Compute Addressing constants to cover all storage formats.
     *    Whether GE, SY, GB, or SB, upper or lower triangle or both,
     *    the (i,j)-th element is in A(i - ISKEW*j + IOFFST, j) */

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

    /* ipackg is the format that the matrix is generated in */
    ipackg = 0;
    slaset("F", lda, n, ZERO, ZERO, A, lda);

#define AIND(i1, j1) ((i1) - iskew*(j1) + ioffst - 1 + ((j1)-1)*lda)
#define AINDG(i1, j1) ((i1) - iskew*(j1) + ioffg - 1 + ((j1)-1)*lda)

    if (llb == 0 && uub == 0) {
        /* Diagonal Matrix */
        cblas_scopy(mnmin, d, 1, &A[ioffst - iskew], ilda + 1);
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

            cblas_scopy(mnmin, d, 1, &A[ioffst - iskew], ilda + 1);

            if (topdwn) {
                jkl = 0;
                for (jku = 1; jku <= uub; jku++) {

                    for (jr = 1; jr <= (((m + jku < n) ? m + jku : n) + jkl - 1); jr++) {
                        extra = ZERO;
                        angle = TWOPI * rng_uniform_f32(state);
                        c = cosf(angle);
                        s = sinf(angle);
                        icol = (1 > jr - jkl) ? 1 : jr - jkl;
                        if (jr < m) {
                            il = ((n < jr + jku) ? n : jr + jku) + 1 - icol;
                            slarot(1, jr > jkl, 0, il,
                                   c, s, &A[AIND(jr, icol)],
                                   ilda, &extra, &dummy);
                        }

                        ir = jr;
                        ic = icol;
                        for (jch = jr - jkl; jch >= 1; jch -= jkl + jku) {
                            if (ir < m) {
                                slartg(A[AIND(ir + 1, ic + 1)],
                                       extra, &c, &s, &dummy);
                            }
                            irow = (1 > jch - jku) ? 1 : jch - jku;
                            il = ir + 2 - irow;
                            temp = ZERO;
                            iltemp = (jch > jku);
                            slarot(0, iltemp, 1, il, c,
                                   -s, &A[AIND(irow, ic)],
                                   ilda, &temp, &extra);
                            if (iltemp) {
                                slartg(A[AIND(irow + 1, ic + 1)],
                                       temp, &c, &s, &dummy);
                                icol = (1 > jch - jku - jkl) ? 1 : jch - jku - jkl;
                                il = ic + 2 - icol;
                                extra = ZERO;
                                slarot(1, jch > jku + jkl,
                                       1, il, c, -s,
                                       &A[AIND(irow, icol)],
                                       ilda, &extra, &temp);
                                ic = icol;
                                ir = irow;
                            }
                        }
                    }
                }

                jku = uub;
                for (jkl = 1; jkl <= llb; jkl++) {

                    for (jc = 1; jc <= (((n + jkl < m) ? n + jkl : m) + jku - 1); jc++) {
                        extra = ZERO;
                        angle = TWOPI * rng_uniform_f32(state);
                        c = cosf(angle);
                        s = sinf(angle);
                        irow = (1 > jc - jku) ? 1 : jc - jku;
                        if (jc < n) {
                            il = ((m < jc + jkl) ? m : jc + jkl) + 1 - irow;
                            slarot(0, jc > jku, 0, il,
                                   c, s, &A[AIND(irow, jc)],
                                   ilda, &extra, &dummy);
                        }

                        ic = jc;
                        ir = irow;
                        for (jch = jc - jku; jch >= 1; jch -= jkl + jku) {
                            if (ic < n) {
                                slartg(A[AIND(ir + 1, ic + 1)],
                                       extra, &c, &s, &dummy);
                            }
                            icol = (1 > jch - jkl) ? 1 : jch - jkl;
                            il = ic + 2 - icol;
                            temp = ZERO;
                            iltemp = (jch > jkl);
                            slarot(1, iltemp, 1, il, c,
                                   -s, &A[AIND(ir, icol)],
                                   ilda, &temp, &extra);
                            if (iltemp) {
                                slartg(A[AIND(ir + 1, icol + 1)],
                                       temp, &c, &s, &dummy);
                                irow = (1 > jch - jkl - jku) ? 1 : jch - jkl - jku;
                                il = ir + 2 - irow;
                                extra = ZERO;
                                slarot(0, jch > jkl + jku,
                                       1, il, c, -s,
                                       &A[AIND(irow, icol)],
                                       ilda, &extra, &temp);
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
                    for (jc = ((m + jku < n) ? m + jku : n) - 1; jc >= 1 - jkl; jc--) {
                        extra = ZERO;
                        angle = TWOPI * rng_uniform_f32(state);
                        c = cosf(angle);
                        s = sinf(angle);
                        irow = (1 > jc - jku + 1) ? 1 : jc - jku + 1;
                        if (jc > 0) {
                            il = ((m < jc + jkl + 1) ? m : jc + jkl + 1) + 1 - irow;
                            slarot(0, 0, jc + jkl < m,
                                   il, c, s,
                                   &A[AIND(irow, jc)],
                                   ilda, &dummy, &extra);
                        }

                        ic = jc;
                        for (jch = jc + jkl; jch <= iendch; jch += jkl + jku) {
                            ilextr = (ic > 0);
                            if (ilextr) {
                                slartg(A[AIND(jch, ic)],
                                       extra, &c, &s, &dummy);
                            }
                            ic = (1 > ic) ? 1 : ic;
                            icol = (n - 1 < jch + jku) ? n - 1 : jch + jku;
                            iltemp = (jch + jku < n);
                            temp = ZERO;
                            slarot(1, ilextr, iltemp,
                                   icol + 2 - ic, c, s,
                                   &A[AIND(jch, ic)],
                                   ilda, &extra, &temp);
                            if (iltemp) {
                                slartg(A[AIND(jch, icol)],
                                       temp, &c, &s, &dummy);
                                il = ((iendch < jch + jkl + jku) ? iendch : jch + jkl + jku) + 2 - jch;
                                extra = ZERO;
                                slarot(0, 1,
                                       jch + jkl + jku <= iendch, il, c, s,
                                       &A[AIND(jch, icol)],
                                       ilda, &temp, &extra);
                                ic = icol;
                            }
                        }
                    }
                }

                jku = uub;
                for (jkl = 1; jkl <= llb; jkl++) {

                    iendch = ((n < m + jku) ? n : m + jku) - 1;
                    for (jr = ((n + jkl < m) ? n + jkl : m) - 1; jr >= 1 - jku; jr--) {
                        extra = ZERO;
                        angle = TWOPI * rng_uniform_f32(state);
                        c = cosf(angle);
                        s = sinf(angle);
                        icol = (1 > jr - jkl + 1) ? 1 : jr - jkl + 1;
                        if (jr > 0) {
                            il = ((n < jr + jku + 1) ? n : jr + jku + 1) + 1 - icol;
                            slarot(1, 0, jr + jku < n,
                                   il, c, s,
                                   &A[AIND(jr, icol)],
                                   ilda, &dummy, &extra);
                        }

                        ir = jr;
                        for (jch = jr + jku; jch <= iendch; jch += jkl + jku) {
                            ilextr = (ir > 0);
                            if (ilextr) {
                                slartg(A[AIND(ir, jch)],
                                       extra, &c, &s, &dummy);
                            }
                            ir = (1 > ir) ? 1 : ir;
                            irow = (m - 1 < jch + jkl) ? m - 1 : jch + jkl;
                            iltemp = (jch + jkl < m);
                            temp = ZERO;
                            slarot(0, ilextr, iltemp,
                                   irow + 2 - ir, c, s,
                                   &A[AIND(ir, jch)],
                                   ilda, &extra, &temp);
                            if (iltemp) {
                                slartg(A[AIND(irow, jch)],
                                       temp, &c, &s, &dummy);
                                il = ((iendch < jch + jkl + jku) ? iendch : jch + jkl + jku) + 2 - jch;
                                extra = ZERO;
                                slarot(1, 1,
                                       jch + jkl + jku <= iendch, il, c, s,
                                       &A[AIND(irow, jch)],
                                       ilda, &temp, &extra);
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
                cblas_scopy(mnmin, d, 1, &A[ioffg - iskew], ilda + 1);

                for (k = 1; k <= uub; k++) {
                    for (jc = 1; jc <= n - 1; jc++) {
                        irow = (1 > jc - k) ? 1 : jc - k;
                        il = ((jc + 1 < k + 2) ? jc + 1 : k + 2);
                        extra = ZERO;
                        temp = A[AINDG(jc, jc + 1)];
                        angle = TWOPI * rng_uniform_f32(state);
                        c = cosf(angle);
                        s = sinf(angle);
                        slarot(0, jc > k, 1, il, c, s,
                               &A[AINDG(irow, jc)], ilda,
                               &extra, &temp);
                        slarot(1, 1, 0,
                               ((k < n - jc) ? k : n - jc) + 1, c, s,
                               &A[AINDG(jc, jc)], ilda,
                               &temp, &dummy);

                        icol = jc;
                        for (jch = jc - k; jch >= 1; jch -= k) {
                            slartg(A[AINDG(jch + 1, icol + 1)],
                                   extra, &c, &s, &dummy);
                            temp = A[AINDG(jch, jch + 1)];
                            slarot(1, 1, 1, k + 2, c,
                                   -s,
                                   &A[AINDG(jch, jch)], ilda,
                                   &temp, &extra);
                            irow = (1 > jch - k) ? 1 : jch - k;
                            il = ((jch + 1 < k + 2) ? jch + 1 : k + 2);
                            extra = ZERO;
                            slarot(0, jch > k, 1, il,
                                   c, -s,
                                   &A[AINDG(irow, jch)], ilda,
                                   &extra, &temp);
                            icol = jch;
                        }
                    }
                }

                if (ipack != ipackg && ipack != 3) {
                    for (jc = 1; jc <= n; jc++) {
                        irow = ioffst - iskew * jc;
                        for (jr = jc; jr <= ((n < jc + uub) ? n : jc + uub); jr++) {
                            A[(jr + irow - 1) + (jc - 1) * lda] =
                                A[AINDG(jc, jr)];
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
                    if (ipack == 6)
                        ioffg = 1;
                } else {
                    ipackg = 2;
                }
                cblas_scopy(mnmin, d, 1, &A[ioffg - iskew], ilda + 1);

                for (k = 1; k <= uub; k++) {
                    for (jc = n - 1; jc >= 1; jc--) {
                        il = ((n + 1 - jc < k + 2) ? n + 1 - jc : k + 2);
                        extra = ZERO;
                        temp = A[AINDG(jc + 1, jc)];
                        angle = TWOPI * rng_uniform_f32(state);
                        c = cosf(angle);
                        s = -sinf(angle);
                        slarot(0, 1, n - jc > k, il, c,
                               s,
                               &A[AINDG(jc, jc)], ilda,
                               &temp, &extra);
                        icol = (1 > jc - k + 1) ? 1 : jc - k + 1;
                        slarot(1, 0, 1, jc + 2 - icol,
                               c, s,
                               &A[AINDG(jc, icol)], ilda,
                               &dummy, &temp);

                        icol = jc;
                        for (jch = jc + k; jch <= n - 1; jch += k) {
                            slartg(A[AINDG(jch, icol)],
                                   extra, &c, &s, &dummy);
                            temp = A[AINDG(jch + 1, jch)];
                            slarot(1, 1, 1, k + 2, c,
                                   s,
                                   &A[AINDG(jch, icol)], ilda,
                                   &extra, &temp);
                            il = ((n + 1 - jch < k + 2) ? n + 1 - jch : k + 2);
                            extra = ZERO;
                            slarot(0, 1, n - jch > k, il,
                                   c, s,
                                   &A[AINDG(jch, jch)], ilda,
                                   &temp, &extra);
                            icol = jch;
                        }
                    }
                }

                if (ipack != ipackg && ipack != 4) {
                    for (jc = n; jc >= 1; jc--) {
                        irow = ioffst - iskew * jc;
                        for (jr = jc; jr >= (1 > jc - uub ? 1 : jc - uub); jr--) {
                            A[(jr + irow - 1) + (jc - 1) * lda] =
                                A[AINDG(jc, jr)];
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
            slagge(mr, nc, llb, uub, d, A, lda, work, &iinfo, state);
        } else {
            slagsy(m, llb, d, A, lda, work, &iinfo, state);
        }
        if (iinfo != 0) {
            *info = 3;
            return;
        }
    }

    /* 5) Pack the matrix */
    if (ipack != ipackg) {
        if (ipack == 1) {
            for (j = 1; j <= m; j++) {
                for (i = j + 1; i <= m; i++) {
                    A[(i - 1) + (j - 1) * lda] = ZERO;
                }
            }

        } else if (ipack == 2) {
            for (j = 2; j <= m; j++) {
                for (i = 1; i <= j - 1; i++) {
                    A[(i - 1) + (j - 1) * lda] = ZERO;
                }
            }

        } else if (ipack == 3) {
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
            if (ipack == 5)
                uub = 0;
            if (ipack == 6)
                llb = 0;

            for (j = 1; j <= uub; j++) {
                for (i = ((j + llb < m) ? j + llb : m); i >= 1; i--) {
                    A[(i - j + uub) + (j - 1) * lda] = A[(i - 1) + (j - 1) * lda];
                }
            }

            for (j = uub + 2; j <= n; j++) {
                for (i = j - uub; i <= ((j + llb < m) ? j + llb : m); i++) {
                    A[(i - j + uub) + (j - 1) * lda] = A[(i - 1) + (j - 1) * lda];
                }
            }
        }

        if (ipack == 3 || ipack == 4) {
            for (jc = icol; jc <= m; jc++) {
                for (jr = irow + 1; jr <= lda; jr++) {
                    A[(jr - 1) + (jc - 1) * lda] = ZERO;
                }
                irow = 0;
            }

        } else if (ipack >= 5) {
            ir1 = uub + llb + 2;
            ir2 = uub + m + 2;
            for (jc = 1; jc <= n; jc++) {
                for (jr = 1; jr <= uub + 1 - jc; jr++) {
                    A[(jr - 1) + (jc - 1) * lda] = ZERO;
                }
                for (jr = (1 > ((ir1 < ir2 - jc) ? ir1 : ir2 - jc)) ? 1 : ((ir1 < ir2 - jc) ? ir1 : ir2 - jc);
                     jr <= lda; jr++) {
                    A[(jr - 1) + (jc - 1) * lda] = ZERO;
                }
            }
        }
    }

#undef AIND
#undef AINDG
}
