/**
 * @file sgsvj0.c
 * @brief SGSVJ0 is a pre-processor for SGESVJ that applies Jacobi rotations.
 */

#include "semicolon_lapack_single.h"
#include <math.h>
#include "semicolon_cblas.h"

static const f32 ZERO = 0.0f;
static const f32 HALF = 0.5f;
static const f32 ONE = 1.0f;

/**
 * SGSVJ0 is called from SGESVJ as a pre-processor. It applies Jacobi
 * rotations in the same way as SGESVJ does, but it does not check convergence
 * (stopping criterion). Few tuning parameters are available for the implementer.
 *
 * @param[in]     jobv    = 'V': accumulate rotations by postmultiplying N-by-N V.
 *                         = 'A': accumulate rotations by postmultiplying MV-by-N V.
 *                         = 'N': do not accumulate rotations.
 * @param[in]     m       Number of rows of A. m >= 0.
 * @param[in]     n       Number of columns of A. m >= n >= 0.
 * @param[in,out] A       Array (lda, n). On entry, M-by-N matrix such that A*diag(D)
 *                        represents the input. On exit, post-multiplied by rotations.
 * @param[in]     lda     Leading dimension of A. lda >= max(1,m).
 * @param[in,out] D       Array (n). Scaling factors from fast scaled Jacobi rotations.
 * @param[in,out] SVA     Array (n). Euclidean norms of columns of A*diag(D).
 * @param[in]     mv      If jobv='A', number of rows of V to post-multiply.
 * @param[in,out] V       Array (ldv, n). Accumulates rotations if jobv='V' or 'A'.
 * @param[in]     ldv     Leading dimension of V.
 * @param[in]     eps     Machine epsilon (slamch('E')).
 * @param[in]     sfmin   Safe minimum (slamch('S')).
 * @param[in]     tol     Threshold for Jacobi rotations.
 * @param[in]     nsweep  Number of sweeps of Jacobi rotations.
 * @param[out]    work    Workspace array of dimension lwork.
 * @param[in]     lwork   Dimension of work. lwork >= m.
 * @param[out]    info
 *                         - = 0: success. < 0: illegal argument.
 */
void sgsvj0(const char* jobv, const INT m, const INT n,
            f32* restrict A, const INT lda,
            f32* restrict D, f32* restrict SVA,
            const INT mv, f32* restrict V, const INT ldv,
            const f32 eps, const f32 sfmin, const f32 tol,
            const INT nsweep, f32* restrict work, const INT lwork,
            INT* info)
{
    INT applv, rsvec, mvl;
    INT i, ibr, igl, ir1, p, q, kbl, nbl;
    INT blskip, rowskip, lkahead, swband;
    INT notrot, pskipped, iswrot, ijblsk, emptsw;
    INT ierr;
    f32 aapp, aapp0, aapq, aaqq, apoaq, aqoap;
    f32 big, bigtheta, cs, sn, t, temp1, theta, thsign;
    f32 mxaapq, mxsinj, rootbig, rooteps, rootsfmin, roottol, small;
    f32 fastr[5];

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
        xerbla("SGSVJ0", -(*info));
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
    emptsw = (n * (n - 1)) / 2;
    fastr[0] = ZERO;

    /* Tuning parameters */
    kbl = (8 < n) ? 8 : n;
    nbl = n / kbl;
    if (nbl * kbl != n) nbl = nbl + 1;
    blskip = kbl * kbl + 1;
    rowskip = (5 < kbl) ? 5 : kbl;
    lkahead = 1;
    swband = 0;

    /* Main sweep loop */
    for (i = 0; i < nsweep; i++) {
        mxaapq = ZERO;
        mxsinj = ZERO;
        iswrot = 0;
        notrot = 0;

        /* Block loop */
        for (ibr = 0; ibr < nbl; ibr++) {

            for (ir1 = 0; ir1 <= (lkahead < nbl - ibr - 1 ? lkahead : nbl - ibr - 1); ir1++) {
                igl = ibr * kbl + ir1 * kbl;

                /* p-loop within diagonal block */
                for (p = igl; p < ((igl + kbl < n - 1) ? (igl + kbl) : (n - 1)); p++) {
                    /* de Rijk's pivoting */
                    q = cblas_isamax(n - p, &SVA[p], 1) + p;
                    if (p != q) {
                        cblas_sswap(m, &A[p * lda], 1, &A[q * lda], 1);
                        if (rsvec) {
                            cblas_sswap(mvl, &V[p * ldv], 1, &V[q * ldv], 1);
                        }
                        temp1 = SVA[p];
                        SVA[p] = SVA[q];
                        SVA[q] = temp1;
                        temp1 = D[p];
                        D[p] = D[q];
                        D[q] = temp1;
                    }

                    if (ir1 == 0) {
                        /* Periodically recompute column norms */
                        if (SVA[p] < rootbig && SVA[p] > rootsfmin) {
                            SVA[p] = cblas_snrm2(m, &A[p * lda], 1) * D[p];
                        } else {
                            temp1 = ZERO;
                            aapp = ONE;
                            slassq(m, &A[p * lda], 1, &temp1, &aapp);
                            SVA[p] = temp1 * sqrtf(aapp) * D[p];
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
                                INT rotok;

                                if (aaqq >= ONE) {
                                    rotok = (small * aapp) <= aaqq;
                                    if (aapp < (big / aaqq)) {
                                        aapq = (cblas_sdot(m, &A[p * lda], 1, &A[q * lda], 1) *
                                                D[p] * D[q] / aaqq) / aapp;
                                    } else {
                                        cblas_scopy(m, &A[p * lda], 1, work, 1);
                                        slascl("G", 0, 0, aapp, D[p], m, 1, work, lda, &ierr);
                                        aapq = cblas_sdot(m, work, 1, &A[q * lda], 1) * D[q] / aaqq;
                                    }
                                } else {
                                    rotok = aapp <= (aaqq / small);
                                    if (aapp > (small / aaqq)) {
                                        aapq = (cblas_sdot(m, &A[p * lda], 1, &A[q * lda], 1) *
                                                D[p] * D[q] / aaqq) / aapp;
                                    } else {
                                        cblas_scopy(m, &A[q * lda], 1, work, 1);
                                        slascl("G", 0, 0, aaqq, D[q], m, 1, work, lda, &ierr);
                                        aapq = cblas_sdot(m, work, 1, &A[p * lda], 1) * D[p] / aapp;
                                    }
                                }

                                mxaapq = (mxaapq > fabsf(aapq)) ? mxaapq : fabsf(aapq);

                                /* TO rotate or NOT to rotate */
                                if (fabsf(aapq) > tol) {
                                    if (ir1 == 0) {
                                        notrot = 0;
                                        pskipped = 0;
                                        iswrot = iswrot + 1;
                                    }

                                    if (rotok) {
                                        aqoap = aaqq / aapp;
                                        apoaq = aapp / aaqq;
                                        theta = -HALF * fabsf(aqoap - apoaq) / aapq;

                                        if (fabsf(theta) > bigtheta) {
                                            t = HALF / theta;
                                            fastr[2] = t * D[p] / D[q];
                                            fastr[3] = -t * D[q] / D[p];
                                            cblas_srotm(m, &A[p * lda], 1, &A[q * lda], 1, fastr);
                                            if (rsvec) {
                                                cblas_srotm(mvl, &V[p * ldv], 1, &V[q * ldv], 1, fastr);
                                            }
                                            SVA[q] = aaqq * sqrtf(fmaxf(ZERO, ONE + t * apoaq * aapq));
                                            aapp = aapp * sqrtf(fmaxf(ZERO, ONE - t * aqoap * aapq));
                                            mxsinj = (mxsinj > fabsf(t)) ? mxsinj : fabsf(t);
                                        } else {
                                            /* choose correct signum for THETA and rotate */
                                            thsign = -copysignf(ONE, aapq);
                                            t = ONE / (theta + thsign * sqrtf(ONE + theta * theta));
                                            cs = sqrtf(ONE / (ONE + t * t));
                                            sn = t * cs;

                                            mxsinj = (mxsinj > fabsf(sn)) ? mxsinj : fabsf(sn);
                                            SVA[q] = aaqq * sqrtf(fmaxf(ZERO, ONE + t * apoaq * aapq));
                                            aapp = aapp * sqrtf(fmaxf(ZERO, ONE - t * aqoap * aapq));

                                            apoaq = D[p] / D[q];
                                            aqoap = D[q] / D[p];

                                            if (D[p] >= ONE) {
                                                if (D[q] >= ONE) {
                                                    fastr[2] = t * apoaq;
                                                    fastr[3] = -t * aqoap;
                                                    D[p] = D[p] * cs;
                                                    D[q] = D[q] * cs;
                                                    cblas_srotm(m, &A[p * lda], 1, &A[q * lda], 1, fastr);
                                                    if (rsvec) {
                                                        cblas_srotm(mvl, &V[p * ldv], 1, &V[q * ldv], 1, fastr);
                                                    }
                                                } else {
                                                    cblas_saxpy(m, -t * aqoap, &A[q * lda], 1, &A[p * lda], 1);
                                                    cblas_saxpy(m, cs * sn * apoaq, &A[p * lda], 1, &A[q * lda], 1);
                                                    D[p] = D[p] * cs;
                                                    D[q] = D[q] / cs;
                                                    if (rsvec) {
                                                        cblas_saxpy(mvl, -t * aqoap, &V[q * ldv], 1, &V[p * ldv], 1);
                                                        cblas_saxpy(mvl, cs * sn * apoaq, &V[p * ldv], 1, &V[q * ldv], 1);
                                                    }
                                                }
                                            } else {
                                                if (D[q] >= ONE) {
                                                    cblas_saxpy(m, t * apoaq, &A[p * lda], 1, &A[q * lda], 1);
                                                    cblas_saxpy(m, -cs * sn * aqoap, &A[q * lda], 1, &A[p * lda], 1);
                                                    D[p] = D[p] / cs;
                                                    D[q] = D[q] * cs;
                                                    if (rsvec) {
                                                        cblas_saxpy(mvl, t * apoaq, &V[p * ldv], 1, &V[q * ldv], 1);
                                                        cblas_saxpy(mvl, -cs * sn * aqoap, &V[q * ldv], 1, &V[p * ldv], 1);
                                                    }
                                                } else {
                                                    if (D[p] >= D[q]) {
                                                        cblas_saxpy(m, -t * aqoap, &A[q * lda], 1, &A[p * lda], 1);
                                                        cblas_saxpy(m, cs * sn * apoaq, &A[p * lda], 1, &A[q * lda], 1);
                                                        D[p] = D[p] * cs;
                                                        D[q] = D[q] / cs;
                                                        if (rsvec) {
                                                            cblas_saxpy(mvl, -t * aqoap, &V[q * ldv], 1, &V[p * ldv], 1);
                                                            cblas_saxpy(mvl, cs * sn * apoaq, &V[p * ldv], 1, &V[q * ldv], 1);
                                                        }
                                                    } else {
                                                        cblas_saxpy(m, t * apoaq, &A[p * lda], 1, &A[q * lda], 1);
                                                        cblas_saxpy(m, -cs * sn * aqoap, &A[q * lda], 1, &A[p * lda], 1);
                                                        D[p] = D[p] / cs;
                                                        D[q] = D[q] * cs;
                                                        if (rsvec) {
                                                            cblas_saxpy(mvl, t * apoaq, &V[p * ldv], 1, &V[q * ldv], 1);
                                                            cblas_saxpy(mvl, -cs * sn * aqoap, &V[q * ldv], 1, &V[p * ldv], 1);
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    } else {
                                        /* Modified Gram-Schmidt transformation */
                                        cblas_scopy(m, &A[p * lda], 1, work, 1);
                                        slascl("G", 0, 0, aapp, ONE, m, 1, work, lda, &ierr);
                                        slascl("G", 0, 0, aaqq, ONE, m, 1, &A[q * lda], lda, &ierr);
                                        temp1 = -aapq * D[p] / D[q];
                                        cblas_saxpy(m, temp1, work, 1, &A[q * lda], 1);
                                        slascl("G", 0, 0, ONE, aaqq, m, 1, &A[q * lda], lda, &ierr);
                                        SVA[q] = aaqq * sqrtf(fmaxf(ZERO, ONE - aapq * aapq));
                                        mxsinj = (mxsinj > sfmin) ? mxsinj : sfmin;
                                    }

                                    /* Recompute SVA if needed due to cancellation */
                                    if ((SVA[q] / aaqq) * (SVA[q] / aaqq) <= rooteps) {
                                        if (aaqq < rootbig && aaqq > rootsfmin) {
                                            SVA[q] = cblas_snrm2(m, &A[q * lda], 1) * D[q];
                                        } else {
                                            t = ZERO;
                                            aaqq = ONE;
                                            slassq(m, &A[q * lda], 1, &t, &aaqq);
                                            SVA[q] = t * sqrtf(aaqq) * D[q];
                                        }
                                    }
                                    if ((aapp / aapp0) <= rooteps) {
                                        if (aapp < rootbig && aapp > rootsfmin) {
                                            aapp = cblas_snrm2(m, &A[p * lda], 1) * D[p];
                                        } else {
                                            t = ZERO;
                                            aapp = ONE;
                                            slassq(m, &A[p * lda], 1, &t, &aapp);
                                            aapp = t * sqrtf(aapp) * D[p];
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
                                        aapq = (cblas_sdot(m, &A[p * lda], 1, &A[q * lda], 1) *
                                                D[p] * D[q] / aaqq) / aapp;
                                    } else {
                                        cblas_scopy(m, &A[p * lda], 1, work, 1);
                                        slascl("G", 0, 0, aapp, D[p], m, 1, work, lda, &ierr);
                                        aapq = cblas_sdot(m, work, 1, &A[q * lda], 1) * D[q] / aaqq;
                                    }
                                } else {
                                    if (aapp >= aaqq) {
                                        rotok = aapp <= (aaqq / small);
                                    } else {
                                        rotok = aaqq <= (aapp / small);
                                    }
                                    if (aapp > (small / aaqq)) {
                                        aapq = (cblas_sdot(m, &A[p * lda], 1, &A[q * lda], 1) *
                                                D[p] * D[q] / aaqq) / aapp;
                                    } else {
                                        cblas_scopy(m, &A[q * lda], 1, work, 1);
                                        slascl("G", 0, 0, aaqq, D[q], m, 1, work, lda, &ierr);
                                        aapq = cblas_sdot(m, work, 1, &A[p * lda], 1) * D[p] / aapp;
                                    }
                                }

                                mxaapq = (mxaapq > fabsf(aapq)) ? mxaapq : fabsf(aapq);

                                if (fabsf(aapq) > tol) {
                                    notrot = 0;
                                    pskipped = 0;
                                    iswrot++;

                                    if (rotok) {
                                        aqoap = aaqq / aapp;
                                        apoaq = aapp / aaqq;
                                        theta = -HALF * fabsf(aqoap - apoaq) / aapq;
                                        if (aaqq > aapp0) theta = -theta;

                                        if (fabsf(theta) > bigtheta) {
                                            t = HALF / theta;
                                            fastr[2] = t * D[p] / D[q];
                                            fastr[3] = -t * D[q] / D[p];
                                            cblas_srotm(m, &A[p * lda], 1, &A[q * lda], 1, fastr);
                                            if (rsvec) {
                                                cblas_srotm(mvl, &V[p * ldv], 1, &V[q * ldv], 1, fastr);
                                            }
                                            SVA[q] = aaqq * sqrtf(fmaxf(ZERO, ONE + t * apoaq * aapq));
                                            aapp = aapp * sqrtf(fmaxf(ZERO, ONE - t * aqoap * aapq));
                                            mxsinj = (mxsinj > fabsf(t)) ? mxsinj : fabsf(t);
                                        } else {
                                            thsign = -copysignf(ONE, aapq);
                                            if (aaqq > aapp0) thsign = -thsign;
                                            t = ONE / (theta + thsign * sqrtf(ONE + theta * theta));
                                            cs = sqrtf(ONE / (ONE + t * t));
                                            sn = t * cs;
                                            mxsinj = (mxsinj > fabsf(sn)) ? mxsinj : fabsf(sn);
                                            SVA[q] = aaqq * sqrtf(fmaxf(ZERO, ONE + t * apoaq * aapq));
                                            aapp = aapp * sqrtf(fmaxf(ZERO, ONE - t * aqoap * aapq));

                                            apoaq = D[p] / D[q];
                                            aqoap = D[q] / D[p];

                                            if (D[p] >= ONE) {
                                                if (D[q] >= ONE) {
                                                    fastr[2] = t * apoaq;
                                                    fastr[3] = -t * aqoap;
                                                    D[p] = D[p] * cs;
                                                    D[q] = D[q] * cs;
                                                    cblas_srotm(m, &A[p * lda], 1, &A[q * lda], 1, fastr);
                                                    if (rsvec) {
                                                        cblas_srotm(mvl, &V[p * ldv], 1, &V[q * ldv], 1, fastr);
                                                    }
                                                } else {
                                                    cblas_saxpy(m, -t * aqoap, &A[q * lda], 1, &A[p * lda], 1);
                                                    cblas_saxpy(m, cs * sn * apoaq, &A[p * lda], 1, &A[q * lda], 1);
                                                    if (rsvec) {
                                                        cblas_saxpy(mvl, -t * aqoap, &V[q * ldv], 1, &V[p * ldv], 1);
                                                        cblas_saxpy(mvl, cs * sn * apoaq, &V[p * ldv], 1, &V[q * ldv], 1);
                                                    }
                                                    D[p] = D[p] * cs;
                                                    D[q] = D[q] / cs;
                                                }
                                            } else {
                                                if (D[q] >= ONE) {
                                                    cblas_saxpy(m, t * apoaq, &A[p * lda], 1, &A[q * lda], 1);
                                                    cblas_saxpy(m, -cs * sn * aqoap, &A[q * lda], 1, &A[p * lda], 1);
                                                    if (rsvec) {
                                                        cblas_saxpy(mvl, t * apoaq, &V[p * ldv], 1, &V[q * ldv], 1);
                                                        cblas_saxpy(mvl, -cs * sn * aqoap, &V[q * ldv], 1, &V[p * ldv], 1);
                                                    }
                                                    D[p] = D[p] / cs;
                                                    D[q] = D[q] * cs;
                                                } else {
                                                    if (D[p] >= D[q]) {
                                                        cblas_saxpy(m, -t * aqoap, &A[q * lda], 1, &A[p * lda], 1);
                                                        cblas_saxpy(m, cs * sn * apoaq, &A[p * lda], 1, &A[q * lda], 1);
                                                        D[p] = D[p] * cs;
                                                        D[q] = D[q] / cs;
                                                        if (rsvec) {
                                                            cblas_saxpy(mvl, -t * aqoap, &V[q * ldv], 1, &V[p * ldv], 1);
                                                            cblas_saxpy(mvl, cs * sn * apoaq, &V[p * ldv], 1, &V[q * ldv], 1);
                                                        }
                                                    } else {
                                                        cblas_saxpy(m, t * apoaq, &A[p * lda], 1, &A[q * lda], 1);
                                                        cblas_saxpy(m, -cs * sn * aqoap, &A[q * lda], 1, &A[p * lda], 1);
                                                        D[p] = D[p] / cs;
                                                        D[q] = D[q] * cs;
                                                        if (rsvec) {
                                                            cblas_saxpy(mvl, t * apoaq, &V[p * ldv], 1, &V[q * ldv], 1);
                                                            cblas_saxpy(mvl, -cs * sn * aqoap, &V[q * ldv], 1, &V[p * ldv], 1);
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    } else {
                                        /* Modified Gram-Schmidt */
                                        if (aapp > aaqq) {
                                            cblas_scopy(m, &A[p * lda], 1, work, 1);
                                            slascl("G", 0, 0, aapp, ONE, m, 1, work, lda, &ierr);
                                            slascl("G", 0, 0, aaqq, ONE, m, 1, &A[q * lda], lda, &ierr);
                                            temp1 = -aapq * D[p] / D[q];
                                            cblas_saxpy(m, temp1, work, 1, &A[q * lda], 1);
                                            slascl("G", 0, 0, ONE, aaqq, m, 1, &A[q * lda], lda, &ierr);
                                            SVA[q] = aaqq * sqrtf(fmaxf(ZERO, ONE - aapq * aapq));
                                            mxsinj = (mxsinj > sfmin) ? mxsinj : sfmin;
                                        } else {
                                            cblas_scopy(m, &A[q * lda], 1, work, 1);
                                            slascl("G", 0, 0, aaqq, ONE, m, 1, work, lda, &ierr);
                                            slascl("G", 0, 0, aapp, ONE, m, 1, &A[p * lda], lda, &ierr);
                                            temp1 = -aapq * D[q] / D[p];
                                            cblas_saxpy(m, temp1, work, 1, &A[p * lda], 1);
                                            slascl("G", 0, 0, ONE, aapp, m, 1, &A[p * lda], lda, &ierr);
                                            SVA[p] = aapp * sqrtf(fmaxf(ZERO, ONE - aapq * aapq));
                                            mxsinj = (mxsinj > sfmin) ? mxsinj : sfmin;
                                        }
                                    }

                                    /* Recompute SVA if needed */
                                    if ((SVA[q] / aaqq) * (SVA[q] / aaqq) <= rooteps) {
                                        if (aaqq < rootbig && aaqq > rootsfmin) {
                                            SVA[q] = cblas_snrm2(m, &A[q * lda], 1) * D[q];
                                        } else {
                                            t = ZERO;
                                            aaqq = ONE;
                                            slassq(m, &A[q * lda], 1, &t, &aaqq);
                                            SVA[q] = t * sqrtf(aaqq) * D[q];
                                        }
                                    }
                                    if ((aapp / aapp0) * (aapp / aapp0) <= rooteps) {
                                        if (aapp < rootbig && aapp > rootsfmin) {
                                            aapp = cblas_snrm2(m, &A[p * lda], 1) * D[p];
                                        } else {
                                            t = ZERO;
                                            aapp = ONE;
                                            slassq(m, &A[p * lda], 1, &t, &aapp);
                                            aapp = t * sqrtf(aapp) * D[p];
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
                SVA[p] = fabsf(SVA[p]);
            }
        } /* end ibr-loop */

        /* Update SVA(n) */
        if (SVA[n - 1] < rootbig && SVA[n - 1] > rootsfmin) {
            SVA[n - 1] = cblas_snrm2(m, &A[(n - 1) * lda], 1) * D[n - 1];
        } else {
            t = ZERO;
            aapp = ONE;
            slassq(m, &A[(n - 1) * lda], 1, &t, &aapp);
            SVA[n - 1] = t * sqrtf(aapp) * D[n - 1];
        }

        /* Additional steering */
        if (i < swband && (mxaapq <= roottol || iswrot <= n)) {
            swband = i;
        }

        if (i > swband + 1 && mxaapq < (f32)n * tol && (f32)n * mxaapq * mxsinj < tol) {
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
        q = cblas_isamax(n - p, &SVA[p], 1) + p;
        if (p != q) {
            temp1 = SVA[p];
            SVA[p] = SVA[q];
            SVA[q] = temp1;
            temp1 = D[p];
            D[p] = D[q];
            D[q] = temp1;
            cblas_sswap(m, &A[p * lda], 1, &A[q * lda], 1);
            if (rsvec) {
                cblas_sswap(mvl, &V[p * ldv], 1, &V[q * ldv], 1);
            }
        }
    }
}
