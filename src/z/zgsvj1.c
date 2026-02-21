/**
 * @file zgsvj1.c
 * @brief ZGSVJ1 is a pre-processor for ZGESVJ that applies Jacobi rotations
 *        targeting specific pivot pairs in off-diagonal blocks.
 */

#include "semicolon_lapack_complex_double.h"
#include <complex.h>
#include <math.h>
#include <cblas.h>

static const f64 ZERO = 0.0;
static const f64 HALF = 0.5;
static const f64 ONE = 1.0;

/**
 * ZGSVJ1 is called from ZGESVJ as a pre-processor. It applies Jacobi
 * rotations in the same way as ZGESVJ does, but it targets only particular
 * pivots and it does not check convergence (stopping criterion).
 *
 * ZGSVJ1 applies few sweeps of Jacobi rotations in the column space of
 * the input M-by-N matrix A. The pivot pairs are taken from the (1,2)
 * off-diagonal block in the corresponding N-by-N Gram matrix A^H * A.
 *
 * @param[in]     jobv    = 'V': accumulate rotations by postmultiplying N-by-N V.
 *                         = 'A': accumulate rotations by postmultiplying MV-by-N V.
 *                         = 'N': do not accumulate rotations.
 * @param[in]     m       Number of rows of A. m >= 0.
 * @param[in]     n       Number of columns of A. m >= n >= 0.
 * @param[in]     n1      First n1 columns are rotated against remaining n-n1.
 * @param[in,out] A       Complex*16 array (lda, n). On entry, M-by-N matrix.
 *                        On exit, post-multiplied by rotations.
 * @param[in]     lda     Leading dimension of A. lda >= max(1,m).
 * @param[in,out] D       Complex*16 array (n). Scaling factors.
 * @param[in,out] SVA     Double precision array (n). Euclidean norms of columns.
 * @param[in]     mv      If jobv='A', number of rows of V to post-multiply.
 * @param[in,out] V       Complex*16 array (ldv, n). Accumulates rotations if requested.
 * @param[in]     ldv     Leading dimension of V.
 * @param[in]     eps     Machine epsilon.
 * @param[in]     sfmin   Safe minimum.
 * @param[in]     tol     Threshold for Jacobi rotations.
 * @param[in]     nsweep  Number of sweeps.
 * @param[out]    work    Complex*16 workspace of dimension lwork.
 * @param[in]     lwork   lwork >= m.
 * @param[out]    info
 *                         - = 0: success. < 0: illegal argument.
 */
void zgsvj1(const char* jobv, const int m, const int n, const int n1,
            c128* restrict A, const int lda,
            c128* restrict D, f64* restrict SVA,
            const int mv, c128* restrict V, const int ldv,
            const f64 eps, const f64 sfmin, const f64 tol,
            const int nsweep, c128* restrict work, const int lwork,
            int* info)
{
    int applv, rsvec, mvl;
    int i, ibr, igl, p, q, kbl, nblr, nblc;
    int blskip, rowskip, swband;
    int notrot, pskipped, iswrot, ijblsk, emptsw;
    int ierr, jbc, jgl;
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
    } else if (n1 < 0) {
        *info = -4;
    } else if (lda < m) {
        *info = -6;
    } else if ((rsvec || applv) && mv < 0) {
        *info = -9;
    } else if ((rsvec && ldv < n) || (applv && ldv < mv)) {
        *info = -11;
    } else if (tol <= eps) {
        *info = -14;
    } else if (nsweep < 0) {
        *info = -15;
    } else if (lwork < m) {
        *info = -17;
    } else {
        *info = 0;
    }

    if (*info != 0) {
        xerbla("ZGSVJ1", -(*info));
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

    emptsw = n1 * (n - n1);

    /* Tuning parameters */
    kbl = (8 < n) ? 8 : n;
    nblr = n1 / kbl;
    if (nblr * kbl != n1) nblr = nblr + 1;

    nblc = (n - n1) / kbl;
    if (nblc * kbl != (n - n1)) nblc = nblc + 1;

    blskip = kbl * kbl + 1;
    rowskip = (5 < kbl) ? 5 : kbl;
    swband = 0;

    /* Main sweep loop */
    for (i = 0; i < nsweep; i++) {
        mxaapq = ZERO;
        mxsinj = ZERO;
        iswrot = 0;
        notrot = 0;

        /* Block loop over row blocks */
        for (ibr = 0; ibr < nblr; ibr++) {
            igl = ibr * kbl;

            /* Loop over column blocks (off-diagonal) */
            for (jbc = 0; jbc < nblc; jbc++) {
                jgl = n1 + jbc * kbl;
                ijblsk = 0;

                /* p-loop within row block */
                for (p = igl; p < ((igl + kbl < n1) ? igl + kbl : n1); p++) {
                    aapp = SVA[p];

                    if (aapp > ZERO) {
                        pskipped = 0;

                        /* q-loop within column block */
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
