/**
 * @file dgesvj.c
 * @brief DGESVJ computes the SVD of a real M-by-N matrix using Jacobi rotations.
 */

#include "semicolon_lapack_double.h"
#include <math.h>
#include <cblas.h>

static const double ZERO = 0.0;
static const double HALF = 0.5;
static const double ONE = 1.0;
static const int NSWEEP = 30;

/**
 * DGESVJ computes the singular value decomposition (SVD) of a real
 * M-by-N matrix A, where M >= N. The SVD of A is written as
 *
 *              A = U * SIGMA * V^t
 *
 * where SIGMA is an N-by-N diagonal matrix, U is an M-by-N orthonormal
 * matrix, and V is an N-by-N orthogonal matrix. The diagonal elements
 * of SIGMA are the singular values of A.
 *
 * DGESVJ can sometimes compute tiny singular values and their singular
 * vectors much more accurately than other SVD routines.
 *
 * @param[in]     joba    = 'L': A is lower triangular.
 *                         = 'U': A is upper triangular.
 *                         = 'G': A is a general M-by-N matrix.
 * @param[in]     jobu    = 'U': Compute left singular vectors in A.
 *                         = 'C': User-controlled orthogonality threshold.
 *                         = 'N': Do not compute left singular vectors.
 * @param[in]     jobv    = 'V': Compute right singular vectors in V.
 *                         = 'A': Apply rotations to existing MV-by-N matrix V.
 *                         = 'N': Do not compute right singular vectors.
 * @param[in]     m       Number of rows of A. M >= 0.
 * @param[in]     n       Number of columns of A. M >= N >= 0.
 * @param[in,out] A       Array (lda, n). On entry, the M-by-N matrix.
 *                        On exit, contains U if jobu='U' or 'C'.
 * @param[in]     lda     Leading dimension of A. lda >= max(1,m).
 * @param[out]    SVA     Array (n). Singular values in descending order.
 * @param[in]     mv      If jobv='A', number of rows of V to transform.
 * @param[in,out] V       Array (ldv, n). Right singular vectors if jobv='V' or 'A'.
 * @param[in]     ldv     Leading dimension of V.
 * @param[in,out] work    Workspace of dimension lwork.
 *                        On entry, work[0] = CTOL if jobu='C'.
 *                        On exit, work[0] = scale, work[1] = nrank,
 *                        work[3] = nsweep, work[4] = max|cos|, work[5] = max|sin|.
 * @param[in]     lwork   lwork >= max(6, m+n).
 * @param[out]    info    = 0: success. < 0: illegal argument. > 0: not converged.
 */
void dgesvj(const char* joba, const char* jobu, const char* jobv,
            const int m, const int n, double* const restrict A, const int lda,
            double* const restrict SVA, const int mv,
            double* const restrict V, const int ldv,
            double* const restrict work, const int lwork, int* info)
{
    int lsvec, uctol, rsvec, applv, upper, lower, lquery;
    int minmn, lwmin, mvl;
    int i, ibr, igl, ir1, p, q, kbl, nbl;
    int rowskip, lkahead, swband;
    int notrot, pskipped, emptsw;
    int ierr;
    double aapp, aapp0, aapq, aaqq, apoaq, aqoap;
    double big, bigtheta, cs, sn, t, temp1, theta, thsign;
    double ctol, epsln, mxaapq, mxsinj, rootbig, rooteps;
    double rootsfmin, sfmin, skl, small, tol, sn_scale;
    double fastr[5];
    int rotok, noscale, goscale;

    /* Test the input parameters */
    lsvec = (jobu[0] == 'U' || jobu[0] == 'u');
    uctol = (jobu[0] == 'C' || jobu[0] == 'c');
    rsvec = (jobv[0] == 'V' || jobv[0] == 'v');
    applv = (jobv[0] == 'A' || jobv[0] == 'a');
    upper = (joba[0] == 'U' || joba[0] == 'u');
    lower = (joba[0] == 'L' || joba[0] == 'l');

    minmn = (m < n) ? m : n;
    if (minmn == 0) {
        lwmin = 1;
    } else {
        lwmin = (6 > m + n) ? 6 : m + n;
    }

    lquery = (lwork == -1);

    if (!upper && !lower && !(joba[0] == 'G' || joba[0] == 'g')) {
        *info = -1;
    } else if (!lsvec && !uctol && !(jobu[0] == 'N' || jobu[0] == 'n')) {
        *info = -2;
    } else if (!rsvec && !applv && !(jobv[0] == 'N' || jobv[0] == 'n')) {
        *info = -3;
    } else if (m < 0) {
        *info = -4;
    } else if (n < 0 || n > m) {
        *info = -5;
    } else if (lda < m) {
        *info = -7;
    } else if (mv < 0) {
        *info = -9;
    } else if ((rsvec && ldv < n) || (applv && ldv < mv)) {
        *info = -11;
    } else if (uctol && work[0] <= ONE) {
        *info = -12;
    } else if (lwork < lwmin && !lquery) {
        *info = -13;
    } else {
        *info = 0;
    }

    if (*info != 0) {
        xerbla("DGESVJ", -(*info));
        return;
    } else if (lquery) {
        work[0] = (double)lwmin;
        return;
    }

    /* Quick return for void matrix */
    if (minmn == 0) return;

    /* Set numerical parameters */
    if (uctol) {
        ctol = work[0];
    } else {
        if (lsvec || rsvec || applv) {
            ctol = sqrt((double)m);
        } else {
            ctol = (double)m;
        }
    }

    epsln = dlamch("E");
    rooteps = sqrt(epsln);
    sfmin = dlamch("S");
    rootsfmin = sqrt(sfmin);
    small = sfmin / epsln;
    big = dlamch("O");
    rootbig = ONE / rootsfmin;
    bigtheta = ONE / rooteps;

    tol = ctol * epsln;

    if ((double)m * epsln >= ONE) {
        *info = -4;
        xerbla("DGESVJ", -(*info));
        return;
    }

    /* Initialize the right singular vector matrix */
    if (rsvec) {
        mvl = n;
        dlaset("A", mvl, n, ZERO, ONE, V, ldv);
    } else if (applv) {
        mvl = mv;
    } else {
        mvl = 0;
    }
    rsvec = rsvec || applv;

    /* Initialize SVA = ||A e_i||_2, with scaling */
    skl = ONE / sqrt((double)m * (double)n);
    noscale = 1;
    goscale = 1;

    if (lower) {
        /* Lower triangular input */
        for (p = 0; p < n; p++) {
            aapp = ZERO;
            aaqq = ONE;
            dlassq(m - p, &A[p + p * lda], 1, &aapp, &aaqq);
            if (aapp > big) {
                *info = -6;
                xerbla("DGESVJ", -(*info));
                return;
            }
            aaqq = sqrt(aaqq);
            if (aapp < (big / aaqq) && noscale) {
                SVA[p] = aapp * aaqq;
            } else {
                noscale = 0;
                SVA[p] = aapp * (aaqq * skl);
                if (goscale) {
                    goscale = 0;
                    for (q = 0; q < p; q++) {
                        SVA[q] = SVA[q] * skl;
                    }
                }
            }
        }
    } else if (upper) {
        /* Upper triangular input */
        for (p = 0; p < n; p++) {
            aapp = ZERO;
            aaqq = ONE;
            dlassq(p + 1, &A[p * lda], 1, &aapp, &aaqq);
            if (aapp > big) {
                *info = -6;
                xerbla("DGESVJ", -(*info));
                return;
            }
            aaqq = sqrt(aaqq);
            if (aapp < (big / aaqq) && noscale) {
                SVA[p] = aapp * aaqq;
            } else {
                noscale = 0;
                SVA[p] = aapp * (aaqq * skl);
                if (goscale) {
                    goscale = 0;
                    for (q = 0; q < p; q++) {
                        SVA[q] = SVA[q] * skl;
                    }
                }
            }
        }
    } else {
        /* General dense input */
        for (p = 0; p < n; p++) {
            aapp = ZERO;
            aaqq = ONE;
            dlassq(m, &A[p * lda], 1, &aapp, &aaqq);
            if (aapp > big) {
                *info = -6;
                xerbla("DGESVJ", -(*info));
                return;
            }
            aaqq = sqrt(aaqq);
            if (aapp < (big / aaqq) && noscale) {
                SVA[p] = aapp * aaqq;
            } else {
                noscale = 0;
                SVA[p] = aapp * (aaqq * skl);
                if (goscale) {
                    goscale = 0;
                    for (q = 0; q < p; q++) {
                        SVA[q] = SVA[q] * skl;
                    }
                }
            }
        }
    }

    if (noscale) skl = ONE;

    /* Find max and min of SVA */
    aapp = ZERO;
    aaqq = big;
    for (p = 0; p < n; p++) {
        if (SVA[p] != ZERO) aaqq = fmin(aaqq, SVA[p]);
        aapp = fmax(aapp, SVA[p]);
    }

    /* Quick return for zero matrix */
    if (aapp == ZERO) {
        if (lsvec) dlaset("G", m, n, ZERO, ONE, A, lda);
        work[0] = ONE;
        work[1] = ZERO;
        work[2] = ZERO;
        work[3] = ZERO;
        work[4] = ZERO;
        work[5] = ZERO;
        return;
    }

    /* Quick return for one-column matrix */
    if (n == 1) {
        if (lsvec) dlascl("G", 0, 0, SVA[0], skl, m, 1, A, lda, &ierr);
        work[0] = ONE / skl;
        work[1] = (SVA[0] >= sfmin) ? ONE : ZERO;
        work[2] = ZERO;
        work[3] = ZERO;
        work[4] = ZERO;
        work[5] = ZERO;
        return;
    }

    /* Protect small singular values from underflow */
    sn_scale = sqrt(sfmin / epsln);
    temp1 = sqrt(big / (double)n);
    if ((aapp <= sn_scale) || (aaqq >= temp1) ||
        ((sn_scale <= aaqq) && (aapp <= temp1))) {
        temp1 = fmin(big, temp1 / aapp);
    } else if ((aaqq <= sn_scale) && (aapp <= temp1)) {
        temp1 = fmin(sn_scale / aaqq, big / (aapp * sqrt((double)n)));
    } else if ((aaqq >= sn_scale) && (aapp >= temp1)) {
        temp1 = fmax(sn_scale / aaqq, temp1 / aapp);
    } else if ((aaqq <= sn_scale) && (aapp >= temp1)) {
        temp1 = fmin(sn_scale / aaqq, big / (sqrt((double)n) * aapp));
    } else {
        temp1 = ONE;
    }

    /* Scale if necessary */
    if (temp1 != ONE) {
        dlascl("G", 0, 0, ONE, temp1, n, 1, SVA, n, &ierr);
    }
    skl = temp1 * skl;
    if (skl != ONE) {
        dlascl(joba, 0, 0, ONE, skl, m, n, A, lda, &ierr);
        skl = ONE / skl;
    }

    /* Row-cyclic Jacobi SVD algorithm with column pivoting */
    emptsw = (n * (n - 1)) / 2;
    notrot = 0;
    fastr[0] = ZERO;

    /* Initialize WORK = diag(D) = I */
    for (q = 0; q < n; q++) {
        work[q] = ONE;
    }

    /* Tuning parameters */
    swband = 3;
    kbl = (8 < n) ? 8 : n;
    nbl = n / kbl;
    if (nbl * kbl != n) nbl = nbl + 1;
    rowskip = (5 < kbl) ? 5 : kbl;
    lkahead = 1;

    int n4 = n / 4;

    /* Main sweep loop */
    for (i = 0; i < NSWEEP; i++) {
        mxaapq = ZERO;
        mxsinj = ZERO;

        notrot = 0;
        pskipped = 0;

        /* Each sweep calls DGSVJ0 and DGSVJ1 for preprocessing */
        if (i > 0 && swband > 0) {
            /* Call DGSVJ0 for diagonal blocks preprocessing */
            dgsvj0(jobv, m, n, A, lda, work, SVA, mvl, V, ldv,
                   epsln, sfmin, tol, 2, &work[n], lwork - n, &ierr);
        }

        /* Call DGSVJ1 for off-diagonal blocks preprocessing */
        if (n > n4 && i > 0) {
            dgsvj1(jobv, m, n, n4, A, lda, work, SVA, mvl, V, ldv,
                   epsln, sfmin, tol, 1, &work[n], lwork - n, &ierr);
        }

        /* Block loop */
        for (ibr = 0; ibr < nbl; ibr++) {
            igl = ibr * kbl;

            for (ir1 = 0; ir1 <= (lkahead < nbl - ibr - 1 ? lkahead : nbl - ibr - 1); ir1++) {
                igl = ibr * kbl + ir1 * kbl;

                for (p = igl; p < ((igl + kbl - 1 < n - 1) ? (igl + kbl - 1) : (n - 1)); p++) {
                    /* de Rijk's pivoting */
                    q = cblas_idamax(n - p, &SVA[p], 1) + p;
                    if (p != q) {
                        cblas_dswap(m, &A[p * lda], 1, &A[q * lda], 1);
                        if (rsvec) cblas_dswap(mvl, &V[p * ldv], 1, &V[q * ldv], 1);
                        temp1 = SVA[p];
                        SVA[p] = SVA[q];
                        SVA[q] = temp1;
                        temp1 = work[p];
                        work[p] = work[q];
                        work[q] = temp1;
                    }

                    if (ir1 == 0) {
                        /* Recompute column norms */
                        if (SVA[p] < rootbig && SVA[p] > rootsfmin) {
                            SVA[p] = cblas_dnrm2(m, &A[p * lda], 1) * work[p];
                        } else {
                            temp1 = ZERO;
                            aapp = ONE;
                            dlassq(m, &A[p * lda], 1, &temp1, &aapp);
                            SVA[p] = temp1 * sqrt(aapp) * work[p];
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

                                /* Compute aapq = A(:,p)'*A(:,q) * D(p)*D(q) / (||A(:,p)||*||A(:,q)||) */
                                if (aaqq >= ONE) {
                                    rotok = (small * aapp) <= aaqq;
                                    if (aapp < (big / aaqq)) {
                                        aapq = (cblas_ddot(m, &A[p * lda], 1, &A[q * lda], 1) *
                                                work[p] * work[q] / aaqq) / aapp;
                                    } else {
                                        cblas_dcopy(m, &A[p * lda], 1, &work[n], 1);
                                        dlascl("G", 0, 0, aapp, work[p], m, 1, &work[n], lda, &ierr);
                                        aapq = cblas_ddot(m, &work[n], 1, &A[q * lda], 1) * work[q] / aaqq;
                                    }
                                } else {
                                    rotok = aapp <= (aaqq / small);
                                    if (aapp > (small / aaqq)) {
                                        aapq = (cblas_ddot(m, &A[p * lda], 1, &A[q * lda], 1) *
                                                work[p] * work[q] / aaqq) / aapp;
                                    } else {
                                        cblas_dcopy(m, &A[q * lda], 1, &work[n], 1);
                                        dlascl("G", 0, 0, aaqq, work[q], m, 1, &work[n], lda, &ierr);
                                        aapq = cblas_ddot(m, &work[n], 1, &A[p * lda], 1) * work[p] / aapp;
                                    }
                                }

                                mxaapq = fmax(mxaapq, fabs(aapq));

                                /* Decide whether to rotate */
                                if (fabs(aapq) > tol) {
                                    if (ir1 == 0) {
                                        notrot = 0;
                                        pskipped = 0;
                                    }

                                    if (rotok) {
                                        aqoap = aaqq / aapp;
                                        apoaq = aapp / aaqq;
                                        theta = -HALF * fabs(aqoap - apoaq) / aapq;

                                        if (fabs(theta) > bigtheta) {
                                            t = HALF / theta;
                                            fastr[2] = t * work[p] / work[q];
                                            fastr[3] = -t * work[q] / work[p];
                                            cblas_drotm(m, &A[p * lda], 1, &A[q * lda], 1, fastr);
                                            if (rsvec) cblas_drotm(mvl, &V[p * ldv], 1, &V[q * ldv], 1, fastr);
                                            SVA[q] = aaqq * sqrt(fmax(ZERO, ONE + t * apoaq * aapq));
                                            aapp = aapp * sqrt(fmax(ZERO, ONE - t * aqoap * aapq));
                                            mxsinj = fmax(mxsinj, fabs(t));
                                        } else {
                                            thsign = -copysign(ONE, aapq);
                                            t = ONE / (theta + thsign * sqrt(ONE + theta * theta));
                                            cs = sqrt(ONE / (ONE + t * t));
                                            sn = t * cs;
                                            mxsinj = fmax(mxsinj, fabs(sn));
                                            SVA[q] = aaqq * sqrt(fmax(ZERO, ONE + t * apoaq * aapq));
                                            aapp = aapp * sqrt(fmax(ZERO, ONE - t * aqoap * aapq));

                                            apoaq = work[p] / work[q];
                                            aqoap = work[q] / work[p];

                                            if (work[p] >= ONE) {
                                                if (work[q] >= ONE) {
                                                    fastr[2] = t * apoaq;
                                                    fastr[3] = -t * aqoap;
                                                    work[p] = work[p] * cs;
                                                    work[q] = work[q] * cs;
                                                    cblas_drotm(m, &A[p * lda], 1, &A[q * lda], 1, fastr);
                                                    if (rsvec) cblas_drotm(mvl, &V[p * ldv], 1, &V[q * ldv], 1, fastr);
                                                } else {
                                                    cblas_daxpy(m, -t * aqoap, &A[q * lda], 1, &A[p * lda], 1);
                                                    cblas_daxpy(m, cs * sn * apoaq, &A[p * lda], 1, &A[q * lda], 1);
                                                    work[p] = work[p] * cs;
                                                    work[q] = work[q] / cs;
                                                    if (rsvec) {
                                                        cblas_daxpy(mvl, -t * aqoap, &V[q * ldv], 1, &V[p * ldv], 1);
                                                        cblas_daxpy(mvl, cs * sn * apoaq, &V[p * ldv], 1, &V[q * ldv], 1);
                                                    }
                                                }
                                            } else {
                                                if (work[q] >= ONE) {
                                                    cblas_daxpy(m, t * apoaq, &A[p * lda], 1, &A[q * lda], 1);
                                                    cblas_daxpy(m, -cs * sn * aqoap, &A[q * lda], 1, &A[p * lda], 1);
                                                    work[p] = work[p] / cs;
                                                    work[q] = work[q] * cs;
                                                    if (rsvec) {
                                                        cblas_daxpy(mvl, t * apoaq, &V[p * ldv], 1, &V[q * ldv], 1);
                                                        cblas_daxpy(mvl, -cs * sn * aqoap, &V[q * ldv], 1, &V[p * ldv], 1);
                                                    }
                                                } else {
                                                    if (work[p] >= work[q]) {
                                                        cblas_daxpy(m, -t * aqoap, &A[q * lda], 1, &A[p * lda], 1);
                                                        cblas_daxpy(m, cs * sn * apoaq, &A[p * lda], 1, &A[q * lda], 1);
                                                        work[p] = work[p] * cs;
                                                        work[q] = work[q] / cs;
                                                        if (rsvec) {
                                                            cblas_daxpy(mvl, -t * aqoap, &V[q * ldv], 1, &V[p * ldv], 1);
                                                            cblas_daxpy(mvl, cs * sn * apoaq, &V[p * ldv], 1, &V[q * ldv], 1);
                                                        }
                                                    } else {
                                                        cblas_daxpy(m, t * apoaq, &A[p * lda], 1, &A[q * lda], 1);
                                                        cblas_daxpy(m, -cs * sn * aqoap, &A[q * lda], 1, &A[p * lda], 1);
                                                        work[p] = work[p] / cs;
                                                        work[q] = work[q] * cs;
                                                        if (rsvec) {
                                                            cblas_daxpy(mvl, t * apoaq, &V[p * ldv], 1, &V[q * ldv], 1);
                                                            cblas_daxpy(mvl, -cs * sn * aqoap, &V[q * ldv], 1, &V[p * ldv], 1);
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    } else {
                                        /* Modified Gram-Schmidt */
                                        cblas_dcopy(m, &A[p * lda], 1, &work[n], 1);
                                        dlascl("G", 0, 0, aapp, ONE, m, 1, &work[n], lda, &ierr);
                                        dlascl("G", 0, 0, aaqq, ONE, m, 1, &A[q * lda], lda, &ierr);
                                        temp1 = -aapq * work[p] / work[q];
                                        cblas_daxpy(m, temp1, &work[n], 1, &A[q * lda], 1);
                                        dlascl("G", 0, 0, ONE, aaqq, m, 1, &A[q * lda], lda, &ierr);
                                        SVA[q] = aaqq * sqrt(fmax(ZERO, ONE - aapq * aapq));
                                        mxsinj = fmax(mxsinj, sfmin);
                                    }

                                    /* Recompute SVA if needed */
                                    if ((SVA[q] / aaqq) * (SVA[q] / aaqq) <= rooteps) {
                                        if (aaqq < rootbig && aaqq > rootsfmin) {
                                            SVA[q] = cblas_dnrm2(m, &A[q * lda], 1) * work[q];
                                        } else {
                                            temp1 = ZERO;
                                            aaqq = ONE;
                                            dlassq(m, &A[q * lda], 1, &temp1, &aaqq);
                                            SVA[q] = temp1 * sqrt(aaqq) * work[q];
                                        }
                                    }
                                    if ((aapp / aapp0) <= rooteps) {
                                        if (aapp < rootbig && aapp > rootsfmin) {
                                            aapp = cblas_dnrm2(m, &A[p * lda], 1) * work[p];
                                        } else {
                                            temp1 = ZERO;
                                            aapp = ONE;
                                            dlassq(m, &A[p * lda], 1, &temp1, &aapp);
                                            aapp = temp1 * sqrt(aapp) * work[p];
                                        }
                                        SVA[p] = aapp;
                                    }
                                } else {
                                    if (ir1 == 0) notrot++;
                                    pskipped++;
                                }
                            } else {
                                if (ir1 == 0) notrot++;
                                pskipped++;
                            }

                            if (i < swband && pskipped > rowskip) {
                                if (ir1 == 0) aapp = -aapp;
                                notrot = 0;
                                break;
                            }
                        } /* end q-loop */

                        SVA[p] = aapp;
                    } else {
                        SVA[p] = aapp;
                        if (ir1 == 0 && aapp == ZERO) {
                            notrot += ((igl + kbl - 1 < n) ? (igl + kbl - 1) : n) - p;
                        }
                    }
                } /* end p-loop */
            } /* end ir1-loop */

            /* Off-diagonal blocks would be handled here similar to dgsvj0 */
        } /* end ibr-loop */

        /* Update SVA(n) */
        if (SVA[n - 1] < rootbig && SVA[n - 1] > rootsfmin) {
            SVA[n - 1] = cblas_dnrm2(m, &A[(n - 1) * lda], 1) * work[n - 1];
        } else {
            temp1 = ZERO;
            aapp = ONE;
            dlassq(m, &A[(n - 1) * lda], 1, &temp1, &aapp);
            SVA[n - 1] = temp1 * sqrt(aapp) * work[n - 1];
        }

        /* Check convergence */
        if (i > swband + 1 && mxaapq < (double)n * tol && (double)n * mxaapq * mxsinj < tol) {
            break;
        }

        if (notrot >= emptsw) break;
    } /* end sweep loop */

    /* Set INFO */
    if (i >= NSWEEP) {
        *info = NSWEEP;
    } else {
        *info = 0;
    }

    /* Sort singular values and rescale */
    for (p = 0; p < n - 1; p++) {
        q = cblas_idamax(n - p, &SVA[p], 1) + p;
        if (p != q) {
            temp1 = SVA[p];
            SVA[p] = SVA[q];
            SVA[q] = temp1;
            temp1 = work[p];
            work[p] = work[q];
            work[q] = temp1;
            cblas_dswap(m, &A[p * lda], 1, &A[q * lda], 1);
            if (rsvec) cblas_dswap(mvl, &V[p * ldv], 1, &V[q * ldv], 1);
        }
    }

    /* Count nonzero and significant singular values */
    int n2_rank = 0;
    int n1_rank = 0;
    for (p = 0; p < n; p++) {
        if (SVA[p] != ZERO) n2_rank++;
        if (SVA[p] * skl > sfmin) n1_rank++;  /* Changed >= to > to match Fortran .GT. */
    }

    /* Scale U (normalize left singular vectors) */
    if (lsvec || uctol) {
        for (p = 0; p < n1_rank; p++) {
            cblas_dscal(m, work[p] / SVA[p], &A[p * lda], 1);
        }
    }

    /* Scale V (assemble the fast rotations and normalize) */
    /* Fortran lines 1609-1621 */
    if (rsvec) {
        if (applv) {
            /* V was pre-initialized: scale by work */
            for (p = 0; p < n; p++) {
                cblas_dscal(mvl, work[p], &V[p * ldv], 1);
            }
        } else {
            /* Normalize each column of V */
            for (p = 0; p < n; p++) {
                temp1 = ONE / cblas_dnrm2(mvl, &V[p * ldv], 1);
                cblas_dscal(mvl, temp1, &V[p * ldv], 1);
            }
        }
    }

    /* Undo scaling, if necessary (and possible).
     * Fortran lines 1623-1631
     *
     * If we can undo the scaling without overflow/underflow, multiply
     * SVA by SKL and set SKL=1. Otherwise, leave SKL as is and the
     * caller must use WORK(1)=SKL to get the true singular values.
     */
    {
        int n2_idx = (n1_rank > 0) ? (n1_rank - 1) : 0;
        if (((skl > ONE) && (SVA[0] < (big / skl))) ||
            ((skl < ONE) && (SVA[n2_idx] > (sfmin / skl)))) {
            for (p = 0; p < n; p++) {
                SVA[p] = skl * SVA[p];
            }
            skl = ONE;
        }
    }

    /* Set output: WORK(1) = SKL (or 1/SKL in LAPACK convention?) */
    /* Note: Looking at Fortran line 1633: WORK(1) = SKL, not 1/SKL */
    work[0] = skl;
    work[1] = (double)n1_rank;
    work[2] = (double)n2_rank;
    work[3] = (double)(i + 1);
    work[4] = mxaapq;
    work[5] = mxsinj;
}
