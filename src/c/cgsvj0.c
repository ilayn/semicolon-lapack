/**
 * @file cgsvj0.c
 * @brief CGSVJ0 is a pre-processor for CGESVJ that applies Jacobi rotations.
 */

#include "semicolon_lapack_complex_single.h"
#include <complex.h>
#include <math.h>
#include "semicolon_cblas.h"

static const f32 ZERO = 0.0f;
static const f32 HALF = 0.5f;
static const f32 ONE = 1.0f;

/**
 * CGSVJ0 is called from CGESVJ as a pre-processor. It applies Jacobi
 * rotations in the same way as CGESVJ does, but it does not check convergence
 * (stopping criterion). Few tuning parameters are available for the implementer.
 *
 * @param[in]     jobv    = 'V': accumulate rotations by postmultiplying N-by-N V.
 *                         = 'A': accumulate rotations by postmultiplying MV-by-N V.
 *                         = 'N': do not accumulate rotations.
 * @param[in]     m       Number of rows of A. m >= 0.
 * @param[in]     n       Number of columns of A. m >= n >= 0.
 * @param[in,out] A       Complex*16 array (lda, n). On entry, M-by-N matrix such that
 *                        A*diag(D) represents the input. On exit, post-multiplied by rotations.
 * @param[in]     lda     Leading dimension of A. lda >= max(1,m).
 * @param[in,out] D       Complex*16 array (n). Scaling factors from complex scaled Jacobi rotations.
 * @param[in,out] SVA     Single precision array (n). Euclidean norms of columns of A*diag(D).
 * @param[in]     mv      If jobv='A', number of rows of V to post-multiply.
 * @param[in,out] V       Complex*16 array (ldv, n). Accumulates rotations if jobv='V' or 'A'.
 * @param[in]     ldv     Leading dimension of V.
 * @param[in]     eps     Machine epsilon (slamch('E')).
 * @param[in]     sfmin   Safe minimum (slamch('S')).
 * @param[in]     tol     Threshold for Jacobi rotations.
 * @param[in]     nsweep  Number of sweeps of Jacobi rotations.
 * @param[out]    work    Complex*16 workspace array of dimension lwork.
 * @param[in]     lwork   Dimension of work. lwork >= m.
 * @param[out]    info
 *                         - = 0: success. < 0: illegal argument.
 */
void cgsvj0(const char* jobv, const INT m, const INT n,
            c64* restrict A, const INT lda,
            c64* restrict D, f32* restrict SVA,
            const INT mv, c64* restrict V, const INT ldv,
            const f32 eps, const f32 sfmin, const f32 tol,
            const INT nsweep, c64* restrict work, const INT lwork,
            INT* info)
{
    INT applv, rsvec, mvl;
    INT i, ibr, igl, ir1, p, q, kbl, nbl;
    INT blskip, rowskip, lkahead, swband;
    INT notrot, pskipped, iswrot, ijblsk, emptsw;
    INT ierr;
    c64 aapq, ompq;
    f32 aapp, aapp0, aapq1, aaqq, apoaq, aqoap;
    f32 big, bigtheta, cs, sn, t, temp1, theta, thsign;
    f32 mxaapq, mxsinj, rootbig, rooteps, rootsfmin, roottol, small;

    /* Test the input parameters */
    applv = (jobv[0] == 'A' || jobv[0] == 'a');
    rsvec = (jobv[0] == 'V' || jobv[0] == 'v');

    if (!rsvec && !applv && !(jobv[0] == 'N' || jobv[0] == 'n')) {
        *info = -1;
    } else if (m < 0) {
        *info = -2;
    } else if (n < 0 || n > m) {
        *info = -3;
    } else if (lda < m) {
        *info = -5;
    } else if ((rsvec || applv) && mv < 0) {
        *info = -8;
    } else if ((rsvec && ldv < n) || (applv && ldv < mv)) {
        *info = -10;
    } else if (tol <= eps) {
        *info = -13;
    } else if (nsweep < 0) {
        *info = -14;
    } else if (lwork < m) {
        *info = -16;
    } else {
        *info = 0;
    }

    if (*info != 0) {
        xerbla("CGSVJ0", -(*info));
        return;
    }

    if (rsvec) {
        mvl = n;
    } else if (applv) {
        mvl = mv;
    } else {
        mvl = 0;
    }
    rsvec = rsvec || applv;

    rooteps = sqrtf(eps);
    rootsfmin = sqrtf(sfmin);
    small = sfmin / eps;
    big = ONE / sfmin;
    rootbig = ONE / rootsfmin;
    bigtheta = ONE / rooteps;
    roottol = sqrtf(tol);

    /* Row-cyclic Jacobi SVD algorithm with column pivoting */
    emptsw = (n / 2) * (n - 1 + n % 2);

    /* Row-cyclic pivot strategy with de Rijk's pivoting */
    kbl = (8 < n) ? 8 : n;
    nbl = n / kbl;
    if (nbl * kbl != n) nbl = nbl + 1;
    blskip = kbl * kbl;
    rowskip = (5 < kbl) ? 5 : kbl;
    lkahead = 1;
    swband = 0;

    /* .. go go go ... */
    for (i = 0; i < nsweep; i++) {
        mxaapq = ZERO;
        mxsinj = ZERO;
        iswrot = 0;
        notrot = 0;
        pskipped = 0;

        /* Each sweep is unrolled using KBL-by-KBL tiles over the pivot pairs
         * 1 <= p < q <= N. This is the first step toward a blocked implementation
         * of the rotations. New implementation, based on block transformations,
         * is under development. */
        for (ibr = 0; ibr < nbl; ibr++) {

            igl = ibr * kbl;

            for (ir1 = 0; ir1 <= (lkahead < nbl - ibr - 1 ? lkahead : nbl - ibr - 1); ir1++) {
                igl = igl + ir1 * kbl;

                for (p = igl; p < ((igl + kbl < n - 1) ? (igl + kbl) : (n - 1)); p++) {
                    /* de Rijk's pivoting */
                    q = cblas_isamax(n - p, &SVA[p], 1) + p;
                    if (p != q) {
                        cblas_cswap(m, &A[p * lda], 1, &A[q * lda], 1);
                        if (rsvec) {
                            cblas_cswap(mvl, &V[p * ldv], 1, &V[q * ldv], 1);
                        }
                        temp1 = SVA[p];
                        SVA[p] = SVA[q];
                        SVA[q] = temp1;
                        c64 dtmp = D[p];
                        D[p] = D[q];
                        D[q] = dtmp;
                    }

                    if (ir1 == 0) {
                        if (SVA[p] < rootbig && SVA[p] > rootsfmin) {
                            SVA[p] = cblas_scnrm2(m, &A[p * lda], 1);
                        } else {
                            temp1 = ZERO;
                            aapp = ONE;
                            classq(m, &A[p * lda], 1, &temp1, &aapp);
                            SVA[p] = temp1 * sqrtf(aapp);
                        }
                        aapp = SVA[p];
                    } else {
                        aapp = SVA[p];
                    }

                    if (aapp > ZERO) {
                        pskipped = 0;

                        for (q = p + 1; q < ((igl + kbl < n) ? igl + kbl : n); q++) {
                            aaqq = SVA[q];

                            if (aaqq > ZERO) {
                                aapp0 = aapp;
                                INT rotok;

                                if (aaqq >= ONE) {
                                    rotok = (small * aapp) <= aaqq;
                                    if (aapp < (big / aaqq)) {
                                        cblas_cdotc_sub(m, &A[p * lda], 1, &A[q * lda], 1, &aapq);
                                        aapq = (aapq / aaqq) / aapp;
                                    } else {
                                        cblas_ccopy(m, &A[p * lda], 1, work, 1);
                                        clascl("G", 0, 0, aapp, ONE, m, 1, work, lda, &ierr);
                                        cblas_cdotc_sub(m, work, 1, &A[q * lda], 1, &aapq);
                                        aapq = aapq / aaqq;
                                    }
                                } else {
                                    rotok = aapp <= (aaqq / small);
                                    if (aapp > (small / aaqq)) {
                                        cblas_cdotc_sub(m, &A[p * lda], 1, &A[q * lda], 1, &aapq);
                                        aapq = (aapq / aapp) / aaqq;
                                    } else {
                                        cblas_ccopy(m, &A[q * lda], 1, work, 1);
                                        clascl("G", 0, 0, aaqq, ONE, m, 1, work, lda, &ierr);
                                        cblas_cdotc_sub(m, &A[p * lda], 1, work, 1, &aapq);
                                        aapq = aapq / aapp;
                                    }
                                }

                                aapq1 = -cabsf(aapq);
                                mxaapq = (mxaapq > -aapq1) ? mxaapq : -aapq1;

                                /* TO rotate or NOT to rotate, THAT is the question ... */
                                if (fabsf(aapq1) > tol) {
                                    ompq = aapq / cabsf(aapq);

                                    if (ir1 == 0) {
                                        notrot = 0;
                                        pskipped = 0;
                                        iswrot = iswrot + 1;
                                    }

                                    if (rotok) {
                                        aqoap = aaqq / aapp;
                                        apoaq = aapp / aaqq;
                                        theta = -HALF * fabsf(aqoap - apoaq) / aapq1;

                                        if (fabsf(theta) > bigtheta) {
                                            t = HALF / theta;
                                            cs = ONE;

                                            crot(m, &A[p * lda], 1, &A[q * lda], 1,
                                                 cs, conjf(ompq) * t);
                                            if (rsvec) {
                                                crot(mvl, &V[p * ldv], 1, &V[q * ldv], 1,
                                                     cs, conjf(ompq) * t);
                                            }

                                            SVA[q] = aaqq * sqrtf((ZERO > ONE + t * apoaq * aapq1)
                                                ? ZERO : ONE + t * apoaq * aapq1);
                                            aapp = aapp * sqrtf((ZERO > ONE - t * aqoap * aapq1)
                                                ? ZERO : ONE - t * aqoap * aapq1);
                                            mxsinj = (mxsinj > fabsf(t)) ? mxsinj : fabsf(t);
                                        } else {
                                            /* choose correct signum for THETA and rotate */
                                            thsign = -copysignf(ONE, aapq1);
                                            t = ONE / (theta + thsign * sqrtf(ONE + theta * theta));
                                            cs = sqrtf(ONE / (ONE + t * t));
                                            sn = t * cs;

                                            mxsinj = (mxsinj > fabsf(sn)) ? mxsinj : fabsf(sn);
                                            SVA[q] = aaqq * sqrtf((ZERO > ONE + t * apoaq * aapq1)
                                                ? ZERO : ONE + t * apoaq * aapq1);
                                            aapp = aapp * sqrtf((ZERO > ONE - t * aqoap * aapq1)
                                                ? ZERO : ONE - t * aqoap * aapq1);

                                            crot(m, &A[p * lda], 1, &A[q * lda], 1,
                                                 cs, conjf(ompq) * sn);
                                            if (rsvec) {
                                                crot(mvl, &V[p * ldv], 1, &V[q * ldv], 1,
                                                     cs, conjf(ompq) * sn);
                                            }
                                        }
                                        D[p] = -D[q] * ompq;

                                    } else {
                                        /* .. have to use modified Gram-Schmidt like transformation */
                                        cblas_ccopy(m, &A[p * lda], 1, work, 1);
                                        clascl("G", 0, 0, aapp, ONE, m, 1, work, lda, &ierr);
                                        clascl("G", 0, 0, aaqq, ONE, m, 1, &A[q * lda], lda, &ierr);
                                        c64 neg_aapq = -aapq;
                                        cblas_caxpy(m, &neg_aapq, work, 1, &A[q * lda], 1);
                                        clascl("G", 0, 0, ONE, aaqq, m, 1, &A[q * lda], lda, &ierr);
                                        SVA[q] = aaqq * sqrtf((ZERO > ONE - aapq1 * aapq1)
                                            ? ZERO : ONE - aapq1 * aapq1);
                                        mxsinj = (mxsinj > sfmin) ? mxsinj : sfmin;
                                    }

                                    /* In the case of cancellation in updating SVA(q), SVA(p)
                                     * recompute SVA(q), SVA(p). */
                                    if ((SVA[q] / aaqq) * (SVA[q] / aaqq) <= rooteps) {
                                        if (aaqq < rootbig && aaqq > rootsfmin) {
                                            SVA[q] = cblas_scnrm2(m, &A[q * lda], 1);
                                        } else {
                                            t = ZERO;
                                            aaqq = ONE;
                                            classq(m, &A[q * lda], 1, &t, &aaqq);
                                            SVA[q] = t * sqrtf(aaqq);
                                        }
                                    }
                                    if ((aapp / aapp0) <= rooteps) {
                                        if (aapp < rootbig && aapp > rootsfmin) {
                                            aapp = cblas_scnrm2(m, &A[p * lda], 1);
                                        } else {
                                            t = ZERO;
                                            aapp = ONE;
                                            classq(m, &A[p * lda], 1, &t, &aapp);
                                            aapp = t * sqrtf(aapp);
                                        }
                                        SVA[p] = aapp;
                                    }
                                } else {
                                    /* A(:,p) and A(:,q) already numerically orthogonal */
                                    if (ir1 == 0) notrot++;
                                    pskipped++;
                                }
                            } else {
                                /* A(:,q) is zero column */
                                if (ir1 == 0) notrot++;
                                pskipped++;
                            }

                            if (i < swband && pskipped > rowskip) {
                                if (ir1 == 0) aapp = -aapp;
                                notrot = 0;
                                goto L2103;
                            }
                        } /* END q-LOOP */
L2103:
                        SVA[p] = aapp;
                    } else {
                        SVA[p] = aapp;
                        if (ir1 == 0 && aapp == ZERO) {
                            notrot += ((igl + kbl < n) ? (igl + kbl) : n) - 1 - p;
                        }
                    }
                } /* end of the p-loop */
            } /* end of ir1-loop */

            /* ... go to the off diagonal blocks */
            igl = ibr * kbl;
            for (INT jbc = ibr + 1; jbc < nbl; jbc++) {
                INT jgl = jbc * kbl;
                ijblsk = 0;

                for (p = igl; p < ((igl + kbl < n) ? igl + kbl : n); p++) {
                    aapp = SVA[p];

                    if (aapp > ZERO) {
                        pskipped = 0;

                        for (q = jgl; q < ((jgl + kbl < n) ? jgl + kbl : n); q++) {
                            aaqq = SVA[q];

                            if (aaqq > ZERO) {
                                aapp0 = aapp;
                                INT rotok;

                                /* Safe Gram matrix computation */
                                if (aaqq >= ONE) {
                                    if (aapp >= aaqq) {
                                        rotok = (small * aapp) <= aaqq;
                                    } else {
                                        rotok = (small * aaqq) <= aapp;
                                    }
                                    if (aapp < (big / aaqq)) {
                                        cblas_cdotc_sub(m, &A[p * lda], 1, &A[q * lda], 1, &aapq);
                                        aapq = (aapq / aaqq) / aapp;
                                    } else {
                                        cblas_ccopy(m, &A[p * lda], 1, work, 1);
                                        clascl("G", 0, 0, aapp, ONE, m, 1, work, lda, &ierr);
                                        cblas_cdotc_sub(m, work, 1, &A[q * lda], 1, &aapq);
                                        aapq = aapq / aaqq;
                                    }
                                } else {
                                    if (aapp >= aaqq) {
                                        rotok = aapp <= (aaqq / small);
                                    } else {
                                        rotok = aaqq <= (aapp / small);
                                    }
                                    if (aapp > (small / aaqq)) {
                                        cblas_cdotc_sub(m, &A[p * lda], 1, &A[q * lda], 1, &aapq);
                                        f32 mx = (aaqq > aapp) ? aaqq : aapp;
                                        f32 mn = (aaqq < aapp) ? aaqq : aapp;
                                        aapq = (aapq / mx) / mn;
                                    } else {
                                        cblas_ccopy(m, &A[q * lda], 1, work, 1);
                                        clascl("G", 0, 0, aaqq, ONE, m, 1, work, lda, &ierr);
                                        cblas_cdotc_sub(m, &A[p * lda], 1, work, 1, &aapq);
                                        aapq = aapq / aapp;
                                    }
                                }

                                aapq1 = -cabsf(aapq);
                                mxaapq = (mxaapq > -aapq1) ? mxaapq : -aapq1;

                                if (fabsf(aapq1) > tol) {
                                    ompq = aapq / cabsf(aapq);
                                    notrot = 0;
                                    pskipped = 0;
                                    iswrot++;

                                    if (rotok) {
                                        aqoap = aaqq / aapp;
                                        apoaq = aapp / aaqq;
                                        theta = -HALF * fabsf(aqoap - apoaq) / aapq1;
                                        if (aaqq > aapp0) theta = -theta;

                                        if (fabsf(theta) > bigtheta) {
                                            t = HALF / theta;
                                            cs = ONE;
                                            crot(m, &A[p * lda], 1, &A[q * lda], 1,
                                                 cs, conjf(ompq) * t);
                                            if (rsvec) {
                                                crot(mvl, &V[p * ldv], 1, &V[q * ldv], 1,
                                                     cs, conjf(ompq) * t);
                                            }
                                            SVA[q] = aaqq * sqrtf((ZERO > ONE + t * apoaq * aapq1)
                                                ? ZERO : ONE + t * apoaq * aapq1);
                                            aapp = aapp * sqrtf((ZERO > ONE - t * aqoap * aapq1)
                                                ? ZERO : ONE - t * aqoap * aapq1);
                                            mxsinj = (mxsinj > fabsf(t)) ? mxsinj : fabsf(t);
                                        } else {
                                            thsign = -copysignf(ONE, aapq1);
                                            if (aaqq > aapp0) thsign = -thsign;
                                            t = ONE / (theta + thsign * sqrtf(ONE + theta * theta));
                                            cs = sqrtf(ONE / (ONE + t * t));
                                            sn = t * cs;
                                            mxsinj = (mxsinj > fabsf(sn)) ? mxsinj : fabsf(sn);
                                            SVA[q] = aaqq * sqrtf((ZERO > ONE + t * apoaq * aapq1)
                                                ? ZERO : ONE + t * apoaq * aapq1);
                                            aapp = aapp * sqrtf((ZERO > ONE - t * aqoap * aapq1)
                                                ? ZERO : ONE - t * aqoap * aapq1);

                                            crot(m, &A[p * lda], 1, &A[q * lda], 1,
                                                 cs, conjf(ompq) * sn);
                                            if (rsvec) {
                                                crot(mvl, &V[p * ldv], 1, &V[q * ldv], 1,
                                                     cs, conjf(ompq) * sn);
                                            }
                                        }
                                        D[p] = -D[q] * ompq;

                                    } else {
                                        /* .. have to use modified Gram-Schmidt like transformation */
                                        if (aapp > aaqq) {
                                            cblas_ccopy(m, &A[p * lda], 1, work, 1);
                                            clascl("G", 0, 0, aapp, ONE, m, 1, work, lda, &ierr);
                                            clascl("G", 0, 0, aaqq, ONE, m, 1, &A[q * lda], lda, &ierr);
                                            c64 neg_aapq = -aapq;
                                            cblas_caxpy(m, &neg_aapq, work, 1, &A[q * lda], 1);
                                            clascl("G", 0, 0, ONE, aaqq, m, 1, &A[q * lda], lda, &ierr);
                                            SVA[q] = aaqq * sqrtf((ZERO > ONE - aapq1 * aapq1)
                                                ? ZERO : ONE - aapq1 * aapq1);
                                            mxsinj = (mxsinj > sfmin) ? mxsinj : sfmin;
                                        } else {
                                            cblas_ccopy(m, &A[q * lda], 1, work, 1);
                                            clascl("G", 0, 0, aaqq, ONE, m, 1, work, lda, &ierr);
                                            clascl("G", 0, 0, aapp, ONE, m, 1, &A[p * lda], lda, &ierr);
                                            c64 neg_conjg_aapq = -conjf(aapq);
                                            cblas_caxpy(m, &neg_conjg_aapq, work, 1, &A[p * lda], 1);
                                            clascl("G", 0, 0, ONE, aapp, m, 1, &A[p * lda], lda, &ierr);
                                            SVA[p] = aapp * sqrtf((ZERO > ONE - aapq1 * aapq1)
                                                ? ZERO : ONE - aapq1 * aapq1);
                                            mxsinj = (mxsinj > sfmin) ? mxsinj : sfmin;
                                        }
                                    }

                                    /* In the case of cancellation in updating SVA(q), SVA(p)
                                     * .. recompute SVA(q), SVA(p) */
                                    if ((SVA[q] / aaqq) * (SVA[q] / aaqq) <= rooteps) {
                                        if (aaqq < rootbig && aaqq > rootsfmin) {
                                            SVA[q] = cblas_scnrm2(m, &A[q * lda], 1);
                                        } else {
                                            t = ZERO;
                                            aaqq = ONE;
                                            classq(m, &A[q * lda], 1, &t, &aaqq);
                                            SVA[q] = t * sqrtf(aaqq);
                                        }
                                    }
                                    if ((aapp / aapp0) * (aapp / aapp0) <= rooteps) {
                                        if (aapp < rootbig && aapp > rootsfmin) {
                                            aapp = cblas_scnrm2(m, &A[p * lda], 1);
                                        } else {
                                            t = ZERO;
                                            aapp = ONE;
                                            classq(m, &A[p * lda], 1, &t, &aapp);
                                            aapp = t * sqrtf(aapp);
                                        }
                                        SVA[p] = aapp;
                                    }
                                } else {
                                    notrot++;
                                    pskipped++;
                                    ijblsk++;
                                }
                            } else {
                                notrot++;
                                pskipped++;
                                ijblsk++;
                            }

                            if (i < swband && ijblsk >= blskip) {
                                SVA[p] = aapp;
                                notrot = 0;
                                goto L2011;
                            }
                            if (i < swband && pskipped > rowskip) {
                                aapp = -aapp;
                                notrot = 0;
                                goto L2203;
                            }
                        } /* end of the q-loop */
L2203:
                        SVA[p] = aapp;
                    } else {
                        if (aapp == ZERO) {
                            notrot += ((jgl + kbl < n) ? jgl + kbl : n) - jgl;
                        }
                        if (aapp < ZERO) notrot = 0;
                    }
                } /* end of the p-loop */
            } /* end of the jbc-loop */

L2011:
            /* bailed out of the jbc-loop */
            for (p = igl; p < ((igl + kbl < n) ? igl + kbl : n); p++) {
                SVA[p] = fabsf(SVA[p]);
            }
        } /* end of the ibr-loop */

        /* .. update SVA(n-1) */
        if (SVA[n - 1] < rootbig && SVA[n - 1] > rootsfmin) {
            SVA[n - 1] = cblas_scnrm2(m, &A[(n - 1) * lda], 1);
        } else {
            t = ZERO;
            aapp = ONE;
            classq(m, &A[(n - 1) * lda], 1, &t, &aapp);
            SVA[n - 1] = t * sqrtf(aapp);
        }

        /* Additional steering devices */
        if (i + 1 < swband && (mxaapq <= roottol || iswrot <= n)) {
            swband = i + 1;
        }

        if (i > swband && mxaapq < sqrtf((f32)n) * tol && (f32)n * mxaapq * mxsinj < tol) {
            goto L1994;
        }

        if (notrot >= emptsw) goto L1994;
    } /* end i=0:NSWEEP-1 loop */

    /* Reaching this point means that the procedure has not converged. */
    *info = nsweep - 1;
    goto L1995;

L1994:
    /* Reaching this point means numerical convergence after the i-th
     * sweep. */
    *info = 0;

L1995:
    /* Sort the vector SVA() of column norms. */
    for (p = 0; p < n - 1; p++) {
        q = cblas_isamax(n - p, &SVA[p], 1) + p;
        if (p != q) {
            temp1 = SVA[p];
            SVA[p] = SVA[q];
            SVA[q] = temp1;
            c64 dtmp = D[p];
            D[p] = D[q];
            D[q] = dtmp;
            cblas_cswap(m, &A[p * lda], 1, &A[q * lda], 1);
            if (rsvec) {
                cblas_cswap(mvl, &V[p * ldv], 1, &V[q * ldv], 1);
            }
        }
    }
}
