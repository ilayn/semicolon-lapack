/**
 * @file sgsvj1.c
 * @brief SGSVJ1 is a pre-processor for SGESVJ that applies Jacobi rotations
 *        targeting specific pivot pairs in off-diagonal blocks.
 */

#include "semicolon_lapack_single.h"
#include <math.h>
#include <cblas.h>

static const float ZERO = 0.0f;
static const float HALF = 0.5f;
static const float ONE = 1.0f;

/**
 * SGSVJ1 is called from SGESVJ as a pre-processor. It applies Jacobi
 * rotations in the same way as SGESVJ does, but it targets only particular
 * pivots and it does not check convergence (stopping criterion).
 *
 * SGSVJ1 applies few sweeps of Jacobi rotations in the column space of
 * the input M-by-N matrix A. The pivot pairs are taken from the (1,2)
 * off-diagonal block in the corresponding N-by-N Gram matrix A^T * A.
 *
 * @param[in]     jobv    = 'V': accumulate rotations by postmultiplying N-by-N V.
 *                         = 'A': accumulate rotations by postmultiplying MV-by-N V.
 *                         = 'N': do not accumulate rotations.
 * @param[in]     m       Number of rows of A. m >= 0.
 * @param[in]     n       Number of columns of A. m >= n >= 0.
 * @param[in]     n1      First n1 columns are rotated against remaining n-n1.
 * @param[in,out] A       Array (lda, n). On entry, M-by-N matrix.
 *                        On exit, post-multiplied by rotations.
 * @param[in]     lda     Leading dimension of A. lda >= max(1,m).
 * @param[in,out] D       Array (n). Scaling factors.
 * @param[in,out] SVA     Array (n). Euclidean norms of columns.
 * @param[in]     mv      If jobv='A', number of rows of V to post-multiply.
 * @param[in,out] V       Array (ldv, n). Accumulates rotations if requested.
 * @param[in]     ldv     Leading dimension of V.
 * @param[in]     eps     Machine epsilon.
 * @param[in]     sfmin   Safe minimum.
 * @param[in]     tol     Threshold for Jacobi rotations.
 * @param[in]     nsweep  Number of sweeps.
 * @param[out]    work    Workspace of dimension lwork.
 * @param[in]     lwork   lwork >= m.
 * @param[out]    info
 *                         - = 0: success. < 0: illegal argument.
 */
void sgsvj1(const char* jobv, const int m, const int n, const int n1,
            float* const restrict A, const int lda,
            float* const restrict D, float* const restrict SVA,
            const int mv, float* const restrict V, const int ldv,
            const float eps, const float sfmin, const float tol,
            const int nsweep, float* const restrict work, const int lwork,
            int* info)
{
    int applv, rsvec, mvl;
    int i, ibr, igl, p, q, kbl, nblr, nblc;
    int blskip, rowskip, swband;
    int notrot, pskipped, iswrot, ijblsk, emptsw;
    int ierr, jbc, jgl;
    float aapp, aapp0, aapq, aaqq, apoaq, aqoap;
    float big, bigtheta, cs, sn, t, temp1, theta, thsign;
    float mxaapq, mxsinj, rootbig, rooteps, rootsfmin, roottol, small;
    float fastr[5];

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
        xerbla("SGSVJ1", -(*info));
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

    emptsw = n1 * (n - n1);
    notrot = 0;
    fastr[0] = ZERO;

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
        pskipped = 0;

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

                                /* TO rotate or NOT to rotate */
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
                                            /* choose correct signum for THETA and rotate */
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

        if (i > swband + 1 && mxaapq < (float)n * tol && (float)n * mxaapq * mxsinj < tol) {
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
