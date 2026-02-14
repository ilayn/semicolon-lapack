/**
 * @file zgsvj0.c
 * @brief ZGSVJ0 is a pre-processor for ZGESVJ that applies Jacobi rotations.
 */

#include "semicolon_lapack_complex_double.h"
#include <complex.h>
#include <math.h>
#include <cblas.h>

static const f64 ZERO = 0.0;
static const f64 HALF = 0.5;
static const f64 ONE = 1.0;

/**
 * ZGSVJ0 is called from ZGESVJ as a pre-processor. It applies Jacobi
 * rotations in the same way as ZGESVJ does, but it does not check convergence
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
 * @param[in,out] SVA     Double precision array (n). Euclidean norms of columns of A*diag(D).
 * @param[in]     mv      If jobv='A', number of rows of V to post-multiply.
 * @param[in,out] V       Complex*16 array (ldv, n). Accumulates rotations if jobv='V' or 'A'.
 * @param[in]     ldv     Leading dimension of V.
 * @param[in]     eps     Machine epsilon (dlamch('E')).
 * @param[in]     sfmin   Safe minimum (dlamch('S')).
 * @param[in]     tol     Threshold for Jacobi rotations.
 * @param[in]     nsweep  Number of sweeps of Jacobi rotations.
 * @param[out]    work    Complex*16 workspace array of dimension lwork.
 * @param[in]     lwork   Dimension of work. lwork >= m.
 * @param[out]    info
 *                         - = 0: success. < 0: illegal argument.
 */
void zgsvj0(const char* jobv, const int m, const int n,
            c128* const restrict A, const int lda,
            c128* const restrict D, f64* const restrict SVA,
            const int mv, c128* const restrict V, const int ldv,
            const f64 eps, const f64 sfmin, const f64 tol,
            const int nsweep, c128* const restrict work, const int lwork,
            int* info)
{
    int applv, rsvec, mvl;
    int i, ibr, igl, ir1, p, q, kbl, nbl;
    int blskip, rowskip, lkahead, swband;
    int notrot, pskipped, iswrot, ijblsk, emptsw;
    int ierr;
    c128 aapq, ompq;
    f64 aapp, aapp0, aapq1, aaqq, apoaq, aqoap;
    f64 big, bigtheta, cs, sn, t, temp1, theta, thsign;
    f64 mxaapq, mxsinj, rootbig, rooteps, rootsfmin, roottol, small;

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
        xerbla("ZGSVJ0", -(*info));
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

    rooteps = sqrt(eps);
    rootsfmin = sqrt(sfmin);
    small = sfmin / eps;
    big = ONE / sfmin;
    rootbig = ONE / rootsfmin;
    bigtheta = ONE / rooteps;
    roottol = sqrt(tol);

    /* Row-cyclic Jacobi SVD algorithm with column pivoting */
    emptsw = (n * (n - 1)) / 2;
    notrot = 0;

    /* Tuning parameters */
    swband = 0;
    kbl = (8 < n) ? 8 : n;
    nbl = n / kbl;
    if (nbl * kbl != n) nbl = nbl + 1;
    blskip = kbl * kbl + 1;
    rowskip = (5 < kbl) ? 5 : kbl;
    lkahead = 1;
    swband = 0;
    pskipped = 0;

    /* Main sweep loop */
    for (i = 0; i < nsweep; i++) {
        mxaapq = ZERO;
        mxsinj = ZERO;
        iswrot = 0;
        notrot = 0;
        pskipped = 0;

        /* Block loop */
        for (ibr = 0; ibr < nbl; ibr++) {
            igl = ibr * kbl;

            for (ir1 = 0; ir1 <= (lkahead < nbl - ibr - 1 ? lkahead : nbl - ibr - 1); ir1++) {
                igl = ibr * kbl + ir1 * kbl;

                /* p-loop within diagonal block */
                for (p = igl; p < ((igl + kbl < n - 1) ? (igl + kbl) : (n - 1)); p++) {
                    /* de Rijk's pivoting */
                    q = cblas_idamax(n - p, &SVA[p], 1) + p;
                    if (p != q) {
                        cblas_zswap(m, &A[p * lda], 1, &A[q * lda], 1);
                        if (rsvec) {
                            cblas_zswap(mvl, &V[p * ldv], 1, &V[q * ldv], 1);
                        }
                        temp1 = SVA[p];
                        SVA[p] = SVA[q];
                        SVA[q] = temp1;
                        c128 dtmp = D[p];
                        D[p] = D[q];
                        D[q] = dtmp;
                    }

                    if (ir1 == 0) {
                        if (SVA[p] < rootbig && SVA[p] > rootsfmin) {
                            SVA[p] = cblas_dznrm2(m, &A[p * lda], 1);
                        } else {
                            temp1 = ZERO;
                            aapp = ONE;
                            zlassq(m, &A[p * lda], 1, &temp1, &aapp);
                            SVA[p] = temp1 * sqrt(aapp);
                        }
                        aapp = SVA[p];
                    } else {
                        aapp = SVA[p];
                    }

                    if (aapp > ZERO) {
                        pskipped = 0;

                        /* q-loop within diagonal block */
                        for (q = p + 1; q < ((igl + kbl < n) ? igl + kbl : n); q++) {
                            aaqq = SVA[q];

                            if (aaqq > ZERO) {
                                aapp0 = aapp;
                                int rotok;

                                if (aaqq >= ONE) {
                                    rotok = (small * aapp) <= aaqq;
                                    if (aapp < (big / aaqq)) {
                                        cblas_zdotc_sub(m, &A[p * lda], 1, &A[q * lda], 1, &aapq);
                                        aapq = (aapq / aaqq) / aapp;
                                    } else {
                                        cblas_zcopy(m, &A[p * lda], 1, work, 1);
                                        zlascl("G", 0, 0, aapp, ONE, m, 1, work, lda, &ierr);
                                        cblas_zdotc_sub(m, work, 1, &A[q * lda], 1, &aapq);
                                        aapq = aapq / aaqq;
                                    }
                                } else {
                                    rotok = aapp <= (aaqq / small);
                                    if (aapp > (small / aaqq)) {
                                        cblas_zdotc_sub(m, &A[p * lda], 1, &A[q * lda], 1, &aapq);
                                        aapq = (aapq / aapp) / aaqq;
                                    } else {
                                        cblas_zcopy(m, &A[q * lda], 1, work, 1);
                                        zlascl("G", 0, 0, aaqq, ONE, m, 1, work, lda, &ierr);
                                        cblas_zdotc_sub(m, &A[p * lda], 1, work, 1, &aapq);
                                        aapq = aapq / aapp;
                                    }
                                }

                                aapq1 = -cabs(aapq);
                                mxaapq = (mxaapq > -aapq1) ? mxaapq : -aapq1;

                                /* TO rotate or NOT to rotate */
                                if (fabs(aapq1) > tol) {
                                    ompq = aapq / cabs(aapq);

                                    if (ir1 == 0) {
                                        notrot = 0;
                                        pskipped = 0;
                                        iswrot = iswrot + 1;
                                    }

                                    if (rotok) {
                                        aqoap = aaqq / aapp;
                                        apoaq = aapp / aaqq;
                                        theta = -HALF * fabs(aqoap - apoaq) / aapq1;

                                        if (fabs(theta) > bigtheta) {
                                            t = HALF / theta;
                                            cs = ONE;

                                            zrot(m, &A[p * lda], 1, &A[q * lda], 1,
                                                 cs, conj(ompq) * t);
                                            if (rsvec) {
                                                zrot(mvl, &V[p * ldv], 1, &V[q * ldv], 1,
                                                     cs, conj(ompq) * t);
                                            }

                                            SVA[q] = aaqq * sqrt(fmax(ZERO, ONE + t * apoaq * aapq1));
                                            aapp = aapp * sqrt(fmax(ZERO, ONE - t * aqoap * aapq1));
                                            mxsinj = (mxsinj > fabs(t)) ? mxsinj : fabs(t);
                                        } else {
                                            /* choose correct signum for THETA and rotate */
                                            thsign = -copysign(ONE, aapq1);
                                            t = ONE / (theta + thsign * sqrt(ONE + theta * theta));
                                            cs = sqrt(ONE / (ONE + t * t));
                                            sn = t * cs;

                                            mxsinj = (mxsinj > fabs(sn)) ? mxsinj : fabs(sn);
                                            SVA[q] = aaqq * sqrt(fmax(ZERO, ONE + t * apoaq * aapq1));
                                            aapp = aapp * sqrt(fmax(ZERO, ONE - t * aqoap * aapq1));

                                            zrot(m, &A[p * lda], 1, &A[q * lda], 1,
                                                 cs, conj(ompq) * sn);
                                            if (rsvec) {
                                                zrot(mvl, &V[p * ldv], 1, &V[q * ldv], 1,
                                                     cs, conj(ompq) * sn);
                                            }
                                        }
                                        D[p] = -D[q] * ompq;

                                    } else {
                                        /* Modified Gram-Schmidt transformation */
                                        cblas_zcopy(m, &A[p * lda], 1, work, 1);
                                        zlascl("G", 0, 0, aapp, ONE, m, 1, work, lda, &ierr);
                                        zlascl("G", 0, 0, aaqq, ONE, m, 1, &A[q * lda], lda, &ierr);
                                        c128 neg_aapq = -aapq;
                                        cblas_zaxpy(m, &neg_aapq, work, 1, &A[q * lda], 1);
                                        zlascl("G", 0, 0, ONE, aaqq, m, 1, &A[q * lda], lda, &ierr);
                                        SVA[q] = aaqq * sqrt(fmax(ZERO, ONE - aapq1 * aapq1));
                                        mxsinj = (mxsinj > sfmin) ? mxsinj : sfmin;
                                    }

                                    /* Recompute SVA if needed due to cancellation */
                                    if ((SVA[q] / aaqq) * (SVA[q] / aaqq) <= rooteps) {
                                        if (aaqq < rootbig && aaqq > rootsfmin) {
                                            SVA[q] = cblas_dznrm2(m, &A[q * lda], 1);
                                        } else {
                                            t = ZERO;
                                            aaqq = ONE;
                                            zlassq(m, &A[q * lda], 1, &t, &aaqq);
                                            SVA[q] = t * sqrt(aaqq);
                                        }
                                    }
                                    if ((aapp / aapp0) <= rooteps) {
                                        if (aapp < rootbig && aapp > rootsfmin) {
                                            aapp = cblas_dznrm2(m, &A[p * lda], 1);
                                        } else {
                                            t = ZERO;
                                            aapp = ONE;
                                            zlassq(m, &A[p * lda], 1, &t, &aapp);
                                            aapp = t * sqrt(aapp);
                                        }
                                        SVA[p] = aapp;
                                    }
                                } else {
                                    /* Already numerically orthogonal */
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
                        } /* end q-loop */
L2103:
                        SVA[p] = aapp;
                    } else {
                        SVA[p] = aapp;
                        if (ir1 == 0 && aapp == ZERO) {
                            notrot += ((igl + kbl < n) ? (igl + kbl) : n) - 1 - p;
                        }
                    }
                } /* end p-loop */
            } /* end ir1-loop */

            /* Off-diagonal blocks */
            igl = ibr * kbl;
            for (int jbc = ibr + 1; jbc < nbl; jbc++) {
                int jgl = jbc * kbl;
                ijblsk = 0;

                for (p = igl; p < ((igl + kbl < n) ? igl + kbl : n); p++) {
                    aapp = SVA[p];

                    if (aapp > ZERO) {
                        pskipped = 0;

                        for (q = jgl; q < ((jgl + kbl < n) ? jgl + kbl : n); q++) {
                            aaqq = SVA[q];

                            if (aaqq > ZERO) {
                                aapp0 = aapp;
                                int rotok;

                                /* Safe Gram matrix computation */
                                if (aaqq >= ONE) {
                                    if (aapp >= aaqq) {
                                        rotok = (small * aapp) <= aaqq;
                                    } else {
                                        rotok = (small * aaqq) <= aapp;
                                    }
                                    if (aapp < (big / aaqq)) {
                                        cblas_zdotc_sub(m, &A[p * lda], 1, &A[q * lda], 1, &aapq);
                                        aapq = (aapq / aaqq) / aapp;
                                    } else {
                                        cblas_zcopy(m, &A[p * lda], 1, work, 1);
                                        zlascl("G", 0, 0, aapp, ONE, m, 1, work, lda, &ierr);
                                        cblas_zdotc_sub(m, work, 1, &A[q * lda], 1, &aapq);
                                        aapq = aapq / aaqq;
                                    }
                                } else {
                                    if (aapp >= aaqq) {
                                        rotok = aapp <= (aaqq / small);
                                    } else {
                                        rotok = aaqq <= (aapp / small);
                                    }
                                    if (aapp > (small / aaqq)) {
                                        cblas_zdotc_sub(m, &A[p * lda], 1, &A[q * lda], 1, &aapq);
                                        f64 mx = (aaqq > aapp) ? aaqq : aapp;
                                        f64 mn = (aaqq < aapp) ? aaqq : aapp;
                                        aapq = (aapq / mx) / mn;
                                    } else {
                                        cblas_zcopy(m, &A[q * lda], 1, work, 1);
                                        zlascl("G", 0, 0, aaqq, ONE, m, 1, work, lda, &ierr);
                                        cblas_zdotc_sub(m, &A[p * lda], 1, work, 1, &aapq);
                                        aapq = aapq / aapp;
                                    }
                                }

                                aapq1 = -cabs(aapq);
                                mxaapq = (mxaapq > -aapq1) ? mxaapq : -aapq1;

                                if (fabs(aapq1) > tol) {
                                    ompq = aapq / cabs(aapq);
                                    notrot = 0;
                                    pskipped = 0;
                                    iswrot++;

                                    if (rotok) {
                                        aqoap = aaqq / aapp;
                                        apoaq = aapp / aaqq;
                                        theta = -HALF * fabs(aqoap - apoaq) / aapq1;
                                        if (aaqq > aapp0) theta = -theta;

                                        if (fabs(theta) > bigtheta) {
                                            t = HALF / theta;
                                            cs = ONE;
                                            zrot(m, &A[p * lda], 1, &A[q * lda], 1,
                                                 cs, conj(ompq) * t);
                                            if (rsvec) {
                                                zrot(mvl, &V[p * ldv], 1, &V[q * ldv], 1,
                                                     cs, conj(ompq) * t);
                                            }
                                            SVA[q] = aaqq * sqrt(fmax(ZERO, ONE + t * apoaq * aapq1));
                                            aapp = aapp * sqrt(fmax(ZERO, ONE - t * aqoap * aapq1));
                                            mxsinj = (mxsinj > fabs(t)) ? mxsinj : fabs(t);
                                        } else {
                                            thsign = -copysign(ONE, aapq1);
                                            if (aaqq > aapp0) thsign = -thsign;
                                            t = ONE / (theta + thsign * sqrt(ONE + theta * theta));
                                            cs = sqrt(ONE / (ONE + t * t));
                                            sn = t * cs;
                                            mxsinj = (mxsinj > fabs(sn)) ? mxsinj : fabs(sn);
                                            SVA[q] = aaqq * sqrt(fmax(ZERO, ONE + t * apoaq * aapq1));
                                            aapp = aapp * sqrt(fmax(ZERO, ONE - t * aqoap * aapq1));

                                            zrot(m, &A[p * lda], 1, &A[q * lda], 1,
                                                 cs, conj(ompq) * sn);
                                            if (rsvec) {
                                                zrot(mvl, &V[p * ldv], 1, &V[q * ldv], 1,
                                                     cs, conj(ompq) * sn);
                                            }
                                        }
                                        D[p] = -D[q] * ompq;

                                    } else {
                                        /* Modified Gram-Schmidt */
                                        if (aapp > aaqq) {
                                            cblas_zcopy(m, &A[p * lda], 1, work, 1);
                                            zlascl("G", 0, 0, aapp, ONE, m, 1, work, lda, &ierr);
                                            zlascl("G", 0, 0, aaqq, ONE, m, 1, &A[q * lda], lda, &ierr);
                                            c128 neg_aapq = -aapq;
                                            cblas_zaxpy(m, &neg_aapq, work, 1, &A[q * lda], 1);
                                            zlascl("G", 0, 0, ONE, aaqq, m, 1, &A[q * lda], lda, &ierr);
                                            SVA[q] = aaqq * sqrt(fmax(ZERO, ONE - aapq1 * aapq1));
                                            mxsinj = (mxsinj > sfmin) ? mxsinj : sfmin;
                                        } else {
                                            cblas_zcopy(m, &A[q * lda], 1, work, 1);
                                            zlascl("G", 0, 0, aaqq, ONE, m, 1, work, lda, &ierr);
                                            zlascl("G", 0, 0, aapp, ONE, m, 1, &A[p * lda], lda, &ierr);
                                            c128 neg_conjg_aapq = -conj(aapq);
                                            cblas_zaxpy(m, &neg_conjg_aapq, work, 1, &A[p * lda], 1);
                                            zlascl("G", 0, 0, ONE, aapp, m, 1, &A[p * lda], lda, &ierr);
                                            SVA[p] = aapp * sqrt(fmax(ZERO, ONE - aapq1 * aapq1));
                                            mxsinj = (mxsinj > sfmin) ? mxsinj : sfmin;
                                        }
                                    }

                                    /* Recompute SVA if needed */
                                    if ((SVA[q] / aaqq) * (SVA[q] / aaqq) <= rooteps) {
                                        if (aaqq < rootbig && aaqq > rootsfmin) {
                                            SVA[q] = cblas_dznrm2(m, &A[q * lda], 1);
                                        } else {
                                            t = ZERO;
                                            aaqq = ONE;
                                            zlassq(m, &A[q * lda], 1, &t, &aaqq);
                                            SVA[q] = t * sqrt(aaqq);
                                        }
                                    }
                                    if ((aapp / aapp0) * (aapp / aapp0) <= rooteps) {
                                        if (aapp < rootbig && aapp > rootsfmin) {
                                            aapp = cblas_dznrm2(m, &A[p * lda], 1);
                                        } else {
                                            t = ZERO;
                                            aapp = ONE;
                                            zlassq(m, &A[p * lda], 1, &t, &aapp);
                                            aapp = t * sqrt(aapp);
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
                        } /* end q-loop */
L2203:
                        SVA[p] = aapp;
                    } else {
                        if (aapp == ZERO) {
                            notrot += ((jgl + kbl < n) ? jgl + kbl : n) - jgl;
                        }
                        if (aapp < ZERO) notrot = 0;
                    }
                } /* end p-loop */
            } /* end jbc-loop */

L2011:
            /* Make SVA positive */
            for (p = igl; p < ((igl + kbl < n) ? igl + kbl : n); p++) {
                SVA[p] = fabs(SVA[p]);
            }
        } /* end ibr-loop */

        /* Update SVA(n) */
        if (SVA[n - 1] < rootbig && SVA[n - 1] > rootsfmin) {
            SVA[n - 1] = cblas_dznrm2(m, &A[(n - 1) * lda], 1);
        } else {
            t = ZERO;
            aapp = ONE;
            zlassq(m, &A[(n - 1) * lda], 1, &t, &aapp);
            SVA[n - 1] = t * sqrt(aapp);
        }

        /* Additional steering */
        if (i < swband && (mxaapq <= roottol || iswrot <= n)) {
            swband = i;
        }

        if (i > swband + 1 && mxaapq < sqrt((f64)n) * tol && (f64)n * mxaapq * mxsinj < tol) {
            goto L1994;
        }

        if (notrot >= emptsw) goto L1994;
    } /* end sweep loop */

    /* Procedure completed given number of iterations */
    *info = nsweep - 1;
    goto L1995;

L1994:
    /* Early exit - all pivots below threshold */
    *info = 0;

L1995:
    /* Sort the vector D */
    for (p = 0; p < n - 1; p++) {
        q = cblas_idamax(n - p, &SVA[p], 1) + p;
        if (p != q) {
            temp1 = SVA[p];
            SVA[p] = SVA[q];
            SVA[q] = temp1;
            c128 dtmp = D[p];
            D[p] = D[q];
            D[q] = dtmp;
            cblas_zswap(m, &A[p * lda], 1, &A[q * lda], 1);
            if (rsvec) {
                cblas_zswap(mvl, &V[p * ldv], 1, &V[q * ldv], 1);
            }
        }
    }
}
