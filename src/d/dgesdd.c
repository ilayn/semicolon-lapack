/**
 * @file dgesdd.c
 * @brief DGESDD computes the singular value decomposition (SVD) of a real
 *        M-by-N matrix using a divide-and-conquer algorithm.
 */

#include "semicolon_lapack_double.h"
#include <stdlib.h>
#include <math.h>
#include "semicolon_cblas.h"

static const f64 ZERO = 0.0;
static const f64 ONE = 1.0;

/**
 * DGESDD computes the singular value decomposition (SVD) of a real
 * M-by-N matrix A, optionally computing the left and right singular
 * vectors.  If singular vectors are desired, it uses a
 * divide-and-conquer algorithm.
 *
 * The SVD is written
 *
 *      A = U * SIGMA * transpose(V)
 *
 * where SIGMA is an M-by-N matrix which is zero except for its
 * min(m,n) diagonal elements, U is an M-by-M orthogonal matrix, and
 * V is an N-by-N orthogonal matrix.  The diagonal elements of SIGMA
 * are the singular values of A; they are real and non-negative, and
 * are returned in descending order.  The first min(m,n) columns of
 * U and V are the left and right singular vectors of A.
 *
 * Note that the routine returns VT = V**T, not V.
 *
 * @param[in]     jobz    = 'A': all M columns of U and all N rows of V**T returned.
 *                         = 'S': first min(M,N) columns of U and rows of V**T returned.
 *                         = 'O': If M >= N, first N columns of U overwritten on A and
 *                                all rows of V**T returned in VT; otherwise all columns
 *                                of U returned in U and first M rows of V**T overwritten on A.
 *                         = 'N': no columns of U or rows of V**T computed.
 * @param[in]     m       Number of rows of A. m >= 0.
 * @param[in]     n       Number of columns of A. n >= 0.
 * @param[in,out] A       Array (lda, n). On entry, M-by-N matrix A.
 *                        On exit, contents depend on jobz.
 * @param[in]     lda     Leading dimension of A. lda >= max(1, m).
 * @param[out]    S       Array of dimension min(m,n). Singular values in descending order.
 * @param[out]    U       Array (ldu, ucol). Left singular vectors if requested.
 * @param[in]     ldu     Leading dimension of U.
 * @param[out]    VT      Array (ldvt, n). Right singular vectors transposed if requested.
 * @param[in]     ldvt    Leading dimension of VT.
 * @param[out]    work    Array of dimension max(1, lwork).
 * @param[in]     lwork   Dimension of work. If lwork=-1, workspace query.
 * @param[out]    IWORK   Integer array of dimension 8*min(m,n).
 * @param[out]    info
 *                         - = 0: success. < 0: illegal argument. > 0: DC did not converge.
 */
void dgesdd(const char* jobz, const INT m, const INT n,
            f64* restrict A, const INT lda,
            f64* restrict S,
            f64* restrict U, const INT ldu,
            f64* restrict VT, const INT ldvt,
            f64* restrict work, const INT lwork,
            INT* restrict IWORK, INT* info)
{
    INT lquery, wntqa, wntqas, wntqn, wntqo, wntqs;
    INT bdspac, blk, chunk, i, ie, ierr, il, ir, iscl;
    INT itau, itaup, itauq, iu, ivt, ldwkvt, ldwrkl, ldwrkr, ldwrku;
    INT maxwrk, minmn, minwrk, mnthr, nwork, wrkbl;
    INT lwork_dgebrd_mn, lwork_dgebrd_mm, lwork_dgebrd_nn;
    INT lwork_dgelqf_mn, lwork_dgeqrf_mn;
    INT lwork_dorglq_mn, lwork_dorglq_nn;
    INT lwork_dorgqr_mm, lwork_dorgqr_mn;
    INT lwork_dormbr_prt_mm, lwork_dormbr_qln_mm;
    INT lwork_dormbr_prt_mn, lwork_dormbr_qln_mn;
    INT lwork_dormbr_prt_nn, lwork_dormbr_qln_nn;
    f64 anrm, bignum, eps, smlnum;
    f64 dum[1];   /* For dlange work and dbdsdc Q when not used */
    f64 dum1[1];  /* For workspace query results */
    INT idum[1];     /* For dbdsdc IQ when not used */

    /* Test the input arguments */
    *info = 0;
    minmn = (m < n) ? m : n;
    wntqa = (jobz[0] == 'A' || jobz[0] == 'a');
    wntqs = (jobz[0] == 'S' || jobz[0] == 's');
    wntqas = wntqa || wntqs;
    wntqo = (jobz[0] == 'O' || jobz[0] == 'o');
    wntqn = (jobz[0] == 'N' || jobz[0] == 'n');
    lquery = (lwork == -1);

    if (!(wntqa || wntqs || wntqo || wntqn)) {
        *info = -1;
    } else if (m < 0) {
        *info = -2;
    } else if (n < 0) {
        *info = -3;
    } else if (lda < (m > 1 ? m : 1)) {
        *info = -5;
    } else if (ldu < 1 || (wntqas && ldu < m) || (wntqo && m < n && ldu < m)) {
        *info = -8;
    } else if (ldvt < 1 || (wntqa && ldvt < n) || (wntqs && ldvt < minmn) ||
               (wntqo && m >= n && ldvt < n)) {
        *info = -10;
    }

    /* Compute workspace */
    if (*info == 0) {
        minwrk = 1;
        maxwrk = 1;
        bdspac = 0;
        mnthr = (INT)(minmn * 11.0 / 6.0);

        if (m >= n && minmn > 0) {
            /* Compute space needed for DBDSDC */
            if (wntqn) {
                bdspac = 7 * n;
            } else {
                bdspac = 3 * n * n + 4 * n;
            }

            /* Query workspace sizes - pass NULL for unused arrays, only work returns result */
            dgebrd(m, n, NULL, m, NULL, NULL, NULL, NULL, dum1, -1, &ierr);
            lwork_dgebrd_mn = (INT)dum1[0];

            dgebrd(n, n, NULL, n, NULL, NULL, NULL, NULL, dum1, -1, &ierr);
            lwork_dgebrd_nn = (INT)dum1[0];

            dgeqrf(m, n, NULL, m, NULL, dum1, -1, &ierr);
            lwork_dgeqrf_mn = (INT)dum1[0];

            dorgbr("Q", n, n, n, NULL, n, NULL, dum1, -1, &ierr);

            dorgqr(m, m, n, NULL, m, NULL, dum1, -1, &ierr);
            lwork_dorgqr_mm = (INT)dum1[0];

            dorgqr(m, n, n, NULL, m, NULL, dum1, -1, &ierr);
            lwork_dorgqr_mn = (INT)dum1[0];

            dormbr("P", "R", "T", n, n, n, NULL, n, NULL, NULL, n, dum1, -1, &ierr);
            lwork_dormbr_prt_nn = (INT)dum1[0];

            dormbr("Q", "L", "N", n, n, n, NULL, n, NULL, NULL, n, dum1, -1, &ierr);
            lwork_dormbr_qln_nn = (INT)dum1[0];

            dormbr("Q", "L", "N", m, n, n, NULL, m, NULL, NULL, m, dum1, -1, &ierr);
            lwork_dormbr_qln_mn = (INT)dum1[0];

            dormbr("Q", "L", "N", m, m, n, NULL, m, NULL, NULL, m, dum1, -1, &ierr);
            lwork_dormbr_qln_mm = (INT)dum1[0];

            if (m >= mnthr) {
                if (wntqn) {
                    /* Path 1 (M >> N, JOBZ='N') */
                    wrkbl = n + lwork_dgeqrf_mn;
                    wrkbl = (wrkbl > 3 * n + lwork_dgebrd_nn) ? wrkbl : 3 * n + lwork_dgebrd_nn;
                    maxwrk = (wrkbl > bdspac + n) ? wrkbl : bdspac + n;
                    minwrk = bdspac + n;
                } else if (wntqo) {
                    /* Path 2 (M >> N, JOBZ='O') */
                    wrkbl = n + lwork_dgeqrf_mn;
                    wrkbl = (wrkbl > n + lwork_dorgqr_mn) ? wrkbl : n + lwork_dorgqr_mn;
                    wrkbl = (wrkbl > 3 * n + lwork_dgebrd_nn) ? wrkbl : 3 * n + lwork_dgebrd_nn;
                    wrkbl = (wrkbl > 3 * n + lwork_dormbr_qln_nn) ? wrkbl : 3 * n + lwork_dormbr_qln_nn;
                    wrkbl = (wrkbl > 3 * n + lwork_dormbr_prt_nn) ? wrkbl : 3 * n + lwork_dormbr_prt_nn;
                    wrkbl = (wrkbl > 3 * n + bdspac) ? wrkbl : 3 * n + bdspac;
                    maxwrk = wrkbl + 2 * n * n;
                    minwrk = bdspac + 2 * n * n + 3 * n;
                } else if (wntqs) {
                    /* Path 3 (M >> N, JOBZ='S') */
                    wrkbl = n + lwork_dgeqrf_mn;
                    wrkbl = (wrkbl > n + lwork_dorgqr_mn) ? wrkbl : n + lwork_dorgqr_mn;
                    wrkbl = (wrkbl > 3 * n + lwork_dgebrd_nn) ? wrkbl : 3 * n + lwork_dgebrd_nn;
                    wrkbl = (wrkbl > 3 * n + lwork_dormbr_qln_nn) ? wrkbl : 3 * n + lwork_dormbr_qln_nn;
                    wrkbl = (wrkbl > 3 * n + lwork_dormbr_prt_nn) ? wrkbl : 3 * n + lwork_dormbr_prt_nn;
                    wrkbl = (wrkbl > 3 * n + bdspac) ? wrkbl : 3 * n + bdspac;
                    maxwrk = wrkbl + n * n;
                    minwrk = bdspac + n * n + 3 * n;
                } else if (wntqa) {
                    /* Path 4 (M >> N, JOBZ='A') */
                    wrkbl = n + lwork_dgeqrf_mn;
                    wrkbl = (wrkbl > n + lwork_dorgqr_mm) ? wrkbl : n + lwork_dorgqr_mm;
                    wrkbl = (wrkbl > 3 * n + lwork_dgebrd_nn) ? wrkbl : 3 * n + lwork_dgebrd_nn;
                    wrkbl = (wrkbl > 3 * n + lwork_dormbr_qln_nn) ? wrkbl : 3 * n + lwork_dormbr_qln_nn;
                    wrkbl = (wrkbl > 3 * n + lwork_dormbr_prt_nn) ? wrkbl : 3 * n + lwork_dormbr_prt_nn;
                    wrkbl = (wrkbl > 3 * n + bdspac) ? wrkbl : 3 * n + bdspac;
                    maxwrk = wrkbl + n * n;
                    minwrk = n * n + ((3 * n + bdspac > n + m) ? 3 * n + bdspac : n + m);
                }
            } else {
                /* Path 5 (M >= N, but not much larger) */
                wrkbl = 3 * n + lwork_dgebrd_mn;
                if (wntqn) {
                    maxwrk = (wrkbl > 3 * n + bdspac) ? wrkbl : 3 * n + bdspac;
                    minwrk = 3 * n + ((m > bdspac) ? m : bdspac);
                } else if (wntqo) {
                    wrkbl = (wrkbl > 3 * n + lwork_dormbr_prt_nn) ? wrkbl : 3 * n + lwork_dormbr_prt_nn;
                    wrkbl = (wrkbl > 3 * n + lwork_dormbr_qln_mn) ? wrkbl : 3 * n + lwork_dormbr_qln_mn;
                    wrkbl = (wrkbl > 3 * n + bdspac) ? wrkbl : 3 * n + bdspac;
                    maxwrk = wrkbl + m * n;
                    minwrk = 3 * n + ((m > n * n + bdspac) ? m : n * n + bdspac);
                } else if (wntqs) {
                    wrkbl = (wrkbl > 3 * n + lwork_dormbr_qln_mn) ? wrkbl : 3 * n + lwork_dormbr_qln_mn;
                    wrkbl = (wrkbl > 3 * n + lwork_dormbr_prt_nn) ? wrkbl : 3 * n + lwork_dormbr_prt_nn;
                    maxwrk = (wrkbl > 3 * n + bdspac) ? wrkbl : 3 * n + bdspac;
                    minwrk = 3 * n + ((m > bdspac) ? m : bdspac);
                } else if (wntqa) {
                    wrkbl = (wrkbl > 3 * n + lwork_dormbr_qln_mm) ? wrkbl : 3 * n + lwork_dormbr_qln_mm;
                    wrkbl = (wrkbl > 3 * n + lwork_dormbr_prt_nn) ? wrkbl : 3 * n + lwork_dormbr_prt_nn;
                    maxwrk = (wrkbl > 3 * n + bdspac) ? wrkbl : 3 * n + bdspac;
                    minwrk = 3 * n + ((m > bdspac) ? m : bdspac);
                }
            }
        } else if (minmn > 0) {
            /* M < N */
            /* Compute space needed for DBDSDC */
            if (wntqn) {
                bdspac = 7 * m;
            } else {
                bdspac = 3 * m * m + 4 * m;
            }

            /* Query workspace sizes - pass NULL for unused arrays, only work returns result */
            dgebrd(m, n, NULL, m, NULL, NULL, NULL, NULL, dum1, -1, &ierr);
            lwork_dgebrd_mn = (INT)dum1[0];

            dgebrd(m, m, NULL, m, NULL, NULL, NULL, NULL, dum1, -1, &ierr);
            lwork_dgebrd_mm = (INT)dum1[0];

            dgelqf(m, n, NULL, m, NULL, dum1, -1, &ierr);
            lwork_dgelqf_mn = (INT)dum1[0];

            dorglq(n, n, m, NULL, n, NULL, dum1, -1, &ierr);
            lwork_dorglq_nn = (INT)dum1[0];

            dorglq(m, n, m, NULL, m, NULL, dum1, -1, &ierr);
            lwork_dorglq_mn = (INT)dum1[0];

            dorgbr("P", m, m, m, NULL, n, NULL, dum1, -1, &ierr);

            dormbr("P", "R", "T", m, m, m, NULL, m, NULL, NULL, m, dum1, -1, &ierr);
            lwork_dormbr_prt_mm = (INT)dum1[0];

            dormbr("P", "R", "T", m, n, m, NULL, m, NULL, NULL, m, dum1, -1, &ierr);
            lwork_dormbr_prt_mn = (INT)dum1[0];

            dormbr("P", "R", "T", n, n, m, NULL, n, NULL, NULL, n, dum1, -1, &ierr);
            lwork_dormbr_prt_nn = (INT)dum1[0];

            dormbr("Q", "L", "N", m, m, m, NULL, m, NULL, NULL, m, dum1, -1, &ierr);
            lwork_dormbr_qln_mm = (INT)dum1[0];

            if (n >= mnthr) {
                if (wntqn) {
                    /* Path 1t (N >> M, JOBZ='N') */
                    wrkbl = m + lwork_dgelqf_mn;
                    wrkbl = (wrkbl > 3 * m + lwork_dgebrd_mm) ? wrkbl : 3 * m + lwork_dgebrd_mm;
                    maxwrk = (wrkbl > bdspac + m) ? wrkbl : bdspac + m;
                    minwrk = bdspac + m;
                } else if (wntqo) {
                    /* Path 2t (N >> M, JOBZ='O') */
                    wrkbl = m + lwork_dgelqf_mn;
                    wrkbl = (wrkbl > m + lwork_dorglq_mn) ? wrkbl : m + lwork_dorglq_mn;
                    wrkbl = (wrkbl > 3 * m + lwork_dgebrd_mm) ? wrkbl : 3 * m + lwork_dgebrd_mm;
                    wrkbl = (wrkbl > 3 * m + lwork_dormbr_qln_mm) ? wrkbl : 3 * m + lwork_dormbr_qln_mm;
                    wrkbl = (wrkbl > 3 * m + lwork_dormbr_prt_mm) ? wrkbl : 3 * m + lwork_dormbr_prt_mm;
                    wrkbl = (wrkbl > 3 * m + bdspac) ? wrkbl : 3 * m + bdspac;
                    maxwrk = wrkbl + 2 * m * m;
                    minwrk = bdspac + 2 * m * m + 3 * m;
                } else if (wntqs) {
                    /* Path 3t (N >> M, JOBZ='S') */
                    wrkbl = m + lwork_dgelqf_mn;
                    wrkbl = (wrkbl > m + lwork_dorglq_mn) ? wrkbl : m + lwork_dorglq_mn;
                    wrkbl = (wrkbl > 3 * m + lwork_dgebrd_mm) ? wrkbl : 3 * m + lwork_dgebrd_mm;
                    wrkbl = (wrkbl > 3 * m + lwork_dormbr_qln_mm) ? wrkbl : 3 * m + lwork_dormbr_qln_mm;
                    wrkbl = (wrkbl > 3 * m + lwork_dormbr_prt_mm) ? wrkbl : 3 * m + lwork_dormbr_prt_mm;
                    wrkbl = (wrkbl > 3 * m + bdspac) ? wrkbl : 3 * m + bdspac;
                    maxwrk = wrkbl + m * m;
                    minwrk = bdspac + m * m + 3 * m;
                } else if (wntqa) {
                    /* Path 4t (N >> M, JOBZ='A') */
                    wrkbl = m + lwork_dgelqf_mn;
                    wrkbl = (wrkbl > m + lwork_dorglq_nn) ? wrkbl : m + lwork_dorglq_nn;
                    wrkbl = (wrkbl > 3 * m + lwork_dgebrd_mm) ? wrkbl : 3 * m + lwork_dgebrd_mm;
                    wrkbl = (wrkbl > 3 * m + lwork_dormbr_qln_mm) ? wrkbl : 3 * m + lwork_dormbr_qln_mm;
                    wrkbl = (wrkbl > 3 * m + lwork_dormbr_prt_mm) ? wrkbl : 3 * m + lwork_dormbr_prt_mm;
                    wrkbl = (wrkbl > 3 * m + bdspac) ? wrkbl : 3 * m + bdspac;
                    maxwrk = wrkbl + m * m;
                    minwrk = m * m + ((3 * m + bdspac > m + n) ? 3 * m + bdspac : m + n);
                }
            } else {
                /* Path 5t (N > M, but not much larger) */
                wrkbl = 3 * m + lwork_dgebrd_mn;
                if (wntqn) {
                    maxwrk = (wrkbl > 3 * m + bdspac) ? wrkbl : 3 * m + bdspac;
                    minwrk = 3 * m + ((n > bdspac) ? n : bdspac);
                } else if (wntqo) {
                    wrkbl = (wrkbl > 3 * m + lwork_dormbr_qln_mm) ? wrkbl : 3 * m + lwork_dormbr_qln_mm;
                    wrkbl = (wrkbl > 3 * m + lwork_dormbr_prt_mn) ? wrkbl : 3 * m + lwork_dormbr_prt_mn;
                    wrkbl = (wrkbl > 3 * m + bdspac) ? wrkbl : 3 * m + bdspac;
                    maxwrk = wrkbl + m * n;
                    minwrk = 3 * m + ((n > m * m + bdspac) ? n : m * m + bdspac);
                } else if (wntqs) {
                    wrkbl = (wrkbl > 3 * m + lwork_dormbr_qln_mm) ? wrkbl : 3 * m + lwork_dormbr_qln_mm;
                    wrkbl = (wrkbl > 3 * m + lwork_dormbr_prt_mn) ? wrkbl : 3 * m + lwork_dormbr_prt_mn;
                    maxwrk = (wrkbl > 3 * m + bdspac) ? wrkbl : 3 * m + bdspac;
                    minwrk = 3 * m + ((n > bdspac) ? n : bdspac);
                } else if (wntqa) {
                    wrkbl = (wrkbl > 3 * m + lwork_dormbr_qln_mm) ? wrkbl : 3 * m + lwork_dormbr_qln_mm;
                    wrkbl = (wrkbl > 3 * m + lwork_dormbr_prt_nn) ? wrkbl : 3 * m + lwork_dormbr_prt_nn;
                    maxwrk = (wrkbl > 3 * m + bdspac) ? wrkbl : 3 * m + bdspac;
                    minwrk = 3 * m + ((n > bdspac) ? n : bdspac);
                }
            }
        }

        maxwrk = (maxwrk > minwrk) ? maxwrk : minwrk;
        work[0] = (f64)maxwrk;

        if (lwork < minwrk && !lquery) {
            *info = -12;
        }
    }

    if (*info != 0) {
        xerbla("DGESDD", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    /* Quick return if possible */
    if (m == 0 || n == 0) {
        return;
    }

    /* Get machine constants */
    eps = dlamch("P");
    smlnum = sqrt(dlamch("S")) / eps;
    bignum = ONE / smlnum;

    /* Scale A if max element outside range [SMLNUM, BIGNUM] */
    anrm = dlange("M", m, n, A, lda, dum);
    if (disnan(anrm)) {
        *info = -4;
        return;
    }
    iscl = 0;
    if (anrm > ZERO && anrm < smlnum) {
        iscl = 1;
        dlascl("G", 0, 0, anrm, smlnum, m, n, A, lda, &ierr);
    } else if (anrm > bignum) {
        iscl = 1;
        dlascl("G", 0, 0, anrm, bignum, m, n, A, lda, &ierr);
    }

    if (m >= n) {
        /* A has at least as many rows as columns. If A has sufficiently
         * more rows than columns, first reduce using the QR decomposition */
        if (m >= mnthr) {
            if (wntqn) {
                /* Path 1 (M >> N, JOBZ='N') */
                itau = 0;
                nwork = itau + n;

                dgeqrf(m, n, A, lda, &work[itau], &work[nwork], lwork - nwork, &ierr);
                dlaset("L", n - 1, n - 1, ZERO, ZERO, &A[1], lda);

                ie = 0;
                itauq = ie + n;
                itaup = itauq + n;
                nwork = itaup + n;

                dgebrd(n, n, A, lda, S, &work[ie], &work[itauq], &work[itaup],
                       &work[nwork], lwork - nwork, &ierr);
                nwork = ie + n;

                dbdsdc("U", "N", n, S, &work[ie], NULL, 1, NULL, 1, NULL, NULL,
                       &work[nwork], IWORK, info);
            } else if (wntqo) {
                /* Path 2 (M >> N, JOBZ='O') */
                ir = 0;
                if (lwork >= lda * n + n * n + 3 * n + bdspac) {
                    ldwrkr = lda;
                } else {
                    ldwrkr = (lwork - n * n - 3 * n - bdspac) / n;
                }
                itau = ir + ldwrkr * n;
                nwork = itau + n;

                dgeqrf(m, n, A, lda, &work[itau], &work[nwork], lwork - nwork, &ierr);
                dlacpy("U", n, n, A, lda, &work[ir], ldwrkr);
                dlaset("L", n - 1, n - 1, ZERO, ZERO, &work[ir + 1], ldwrkr);

                dorgqr(m, n, n, A, lda, &work[itau], &work[nwork], lwork - nwork, &ierr);

                ie = itau;
                itauq = ie + n;
                itaup = itauq + n;
                nwork = itaup + n;

                dgebrd(n, n, &work[ir], ldwrkr, S, &work[ie], &work[itauq],
                       &work[itaup], &work[nwork], lwork - nwork, &ierr);

                iu = nwork;
                nwork = iu + n * n;

                dbdsdc("U", "I", n, S, &work[ie], &work[iu], n, VT, ldvt, dum, idum,
                       &work[nwork], IWORK, info);

                dormbr("Q", "L", "N", n, n, n, &work[ir], ldwrkr, &work[itauq],
                       &work[iu], n, &work[nwork], lwork - nwork, &ierr);
                dormbr("P", "R", "T", n, n, n, &work[ir], ldwrkr, &work[itaup],
                       VT, ldvt, &work[nwork], lwork - nwork, &ierr);

                for (i = 0; i < m; i += ldwrkr) {
                    chunk = ((m - i) < ldwrkr) ? (m - i) : ldwrkr;
                    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                chunk, n, n, ONE, &A[i], lda, &work[iu], n,
                                ZERO, &work[ir], ldwrkr);
                    dlacpy("F", chunk, n, &work[ir], ldwrkr, &A[i], lda);
                }
            } else if (wntqs) {
                /* Path 3 (M >> N, JOBZ='S') */
                ir = 0;
                ldwrkr = n;
                itau = ir + ldwrkr * n;
                nwork = itau + n;

                dgeqrf(m, n, A, lda, &work[itau], &work[nwork], lwork - nwork, &ierr);
                dlacpy("U", n, n, A, lda, &work[ir], ldwrkr);
                dlaset("L", n - 1, n - 1, ZERO, ZERO, &work[ir + 1], ldwrkr);

                dorgqr(m, n, n, A, lda, &work[itau], &work[nwork], lwork - nwork, &ierr);

                ie = itau;
                itauq = ie + n;
                itaup = itauq + n;
                nwork = itaup + n;

                dgebrd(n, n, &work[ir], ldwrkr, S, &work[ie], &work[itauq],
                       &work[itaup], &work[nwork], lwork - nwork, &ierr);

                dbdsdc("U", "I", n, S, &work[ie], U, ldu, VT, ldvt, dum, idum,
                       &work[nwork], IWORK, info);

                dormbr("Q", "L", "N", n, n, n, &work[ir], ldwrkr, &work[itauq],
                       U, ldu, &work[nwork], lwork - nwork, &ierr);
                dormbr("P", "R", "T", n, n, n, &work[ir], ldwrkr, &work[itaup],
                       VT, ldvt, &work[nwork], lwork - nwork, &ierr);

                dlacpy("F", n, n, U, ldu, &work[ir], ldwrkr);
                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                            m, n, n, ONE, A, lda, &work[ir], ldwrkr, ZERO, U, ldu);
            } else if (wntqa) {
                /* Path 4 (M >> N, JOBZ='A') */
                iu = 0;
                ldwrku = n;
                itau = iu + ldwrku * n;
                nwork = itau + n;

                dgeqrf(m, n, A, lda, &work[itau], &work[nwork], lwork - nwork, &ierr);
                dlacpy("L", m, n, A, lda, U, ldu);

                dorgqr(m, m, n, U, ldu, &work[itau], &work[nwork], lwork - nwork, &ierr);

                dlaset("L", n - 1, n - 1, ZERO, ZERO, &A[1], lda);

                ie = itau;
                itauq = ie + n;
                itaup = itauq + n;
                nwork = itaup + n;

                dgebrd(n, n, A, lda, S, &work[ie], &work[itauq], &work[itaup],
                       &work[nwork], lwork - nwork, &ierr);

                dbdsdc("U", "I", n, S, &work[ie], &work[iu], n, VT, ldvt, dum, idum,
                       &work[nwork], IWORK, info);

                dormbr("Q", "L", "N", n, n, n, A, lda, &work[itauq], &work[iu],
                       ldwrku, &work[nwork], lwork - nwork, &ierr);
                dormbr("P", "R", "T", n, n, n, A, lda, &work[itaup], VT, ldvt,
                       &work[nwork], lwork - nwork, &ierr);

                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                            m, n, n, ONE, U, ldu, &work[iu], ldwrku, ZERO, A, lda);
                dlacpy("F", m, n, A, lda, U, ldu);
            }
        } else {
            /* Path 5 (M >= N, but not much larger) */
            ie = 0;
            itauq = ie + n;
            itaup = itauq + n;
            nwork = itaup + n;

            dgebrd(m, n, A, lda, S, &work[ie], &work[itauq], &work[itaup],
                   &work[nwork], lwork - nwork, &ierr);

            if (wntqn) {
                dbdsdc("U", "N", n, S, &work[ie], NULL, 1, NULL, 1, NULL, NULL,
                       &work[nwork], IWORK, info);
            } else if (wntqo) {
                iu = nwork;
                if (lwork >= m * n + 3 * n + bdspac) {
                    ldwrku = m;
                    dlaset("F", m, n, ZERO, ZERO, &work[iu], ldwrku);
                    ir = -1;
                } else {
                    ldwrku = n;
                    nwork = iu + ldwrku * n;
                    ir = nwork;
                    ldwrkr = (lwork - n * n - 3 * n) / n;
                }
                nwork = iu + ldwrku * n;

                dbdsdc("U", "I", n, S, &work[ie], &work[iu], ldwrku, VT, ldvt,
                       dum, idum, &work[nwork], IWORK, info);

                dormbr("P", "R", "T", n, n, n, A, lda, &work[itaup], VT, ldvt,
                       &work[nwork], lwork - nwork, &ierr);

                if (lwork >= m * n + 3 * n + bdspac) {
                    dormbr("Q", "L", "N", m, n, n, A, lda, &work[itauq],
                           &work[iu], ldwrku, &work[nwork], lwork - nwork, &ierr);
                    dlacpy("F", m, n, &work[iu], ldwrku, A, lda);
                } else {
                    dorgbr("Q", m, n, n, A, lda, &work[itauq], &work[nwork],
                           lwork - nwork, &ierr);
                    for (i = 0; i < m; i += ldwrkr) {
                        chunk = ((m - i) < ldwrkr) ? (m - i) : ldwrkr;
                        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                    chunk, n, n, ONE, &A[i], lda, &work[iu], ldwrku,
                                    ZERO, &work[ir], ldwrkr);
                        dlacpy("F", chunk, n, &work[ir], ldwrkr, &A[i], lda);
                    }
                }
            } else if (wntqs) {
                dlaset("F", m, n, ZERO, ZERO, U, ldu);
                dbdsdc("U", "I", n, S, &work[ie], U, ldu, VT, ldvt, dum, idum,
                       &work[nwork], IWORK, info);

                dormbr("Q", "L", "N", m, n, n, A, lda, &work[itauq], U, ldu,
                       &work[nwork], lwork - nwork, &ierr);
                dormbr("P", "R", "T", n, n, n, A, lda, &work[itaup], VT, ldvt,
                       &work[nwork], lwork - nwork, &ierr);
            } else if (wntqa) {
                dlaset("F", m, m, ZERO, ZERO, U, ldu);
                dbdsdc("U", "I", n, S, &work[ie], U, ldu, VT, ldvt, dum, idum,
                       &work[nwork], IWORK, info);

                if (m > n) {
                    dlaset("F", m - n, m - n, ZERO, ONE, &U[n + n * ldu], ldu);
                }

                dormbr("Q", "L", "N", m, m, n, A, lda, &work[itauq], U, ldu,
                       &work[nwork], lwork - nwork, &ierr);
                dormbr("P", "R", "T", n, n, m, A, lda, &work[itaup], VT, ldvt,
                       &work[nwork], lwork - nwork, &ierr);
            }
        }
    } else {
        /* A has more columns than rows. If A has sufficiently more columns
         * than rows, first reduce using the LQ decomposition */
        if (n >= mnthr) {
            if (wntqn) {
                /* Path 1t (N >> M, JOBZ='N') */
                itau = 0;
                nwork = itau + m;

                dgelqf(m, n, A, lda, &work[itau], &work[nwork], lwork - nwork, &ierr);
                dlaset("U", m - 1, m - 1, ZERO, ZERO, &A[lda], lda);

                ie = 0;
                itauq = ie + m;
                itaup = itauq + m;
                nwork = itaup + m;

                dgebrd(m, m, A, lda, S, &work[ie], &work[itauq], &work[itaup],
                       &work[nwork], lwork - nwork, &ierr);
                nwork = ie + m;

                dbdsdc("U", "N", m, S, &work[ie], NULL, 1, NULL, 1, NULL, NULL,
                       &work[nwork], IWORK, info);
            } else if (wntqo) {
                /* Path 2t (N >> M, JOBZ='O') */
                ivt = 0;
                il = ivt + m * m;
                if (lwork >= m * n + m * m + 3 * m + bdspac) {
                    ldwrkl = m;
                    chunk = n;
                } else {
                    ldwrkl = m;
                    chunk = (lwork - m * m) / m;
                }
                itau = il + ldwrkl * m;
                nwork = itau + m;

                dgelqf(m, n, A, lda, &work[itau], &work[nwork], lwork - nwork, &ierr);
                dlacpy("L", m, m, A, lda, &work[il], ldwrkl);
                dlaset("U", m - 1, m - 1, ZERO, ZERO, &work[il + ldwrkl], ldwrkl);

                dorglq(m, n, m, A, lda, &work[itau], &work[nwork], lwork - nwork, &ierr);

                ie = itau;
                itauq = ie + m;
                itaup = itauq + m;
                nwork = itaup + m;

                dgebrd(m, m, &work[il], ldwrkl, S, &work[ie], &work[itauq],
                       &work[itaup], &work[nwork], lwork - nwork, &ierr);

                dbdsdc("U", "I", m, S, &work[ie], U, ldu, &work[ivt], m, dum, idum,
                       &work[nwork], IWORK, info);

                dormbr("Q", "L", "N", m, m, m, &work[il], ldwrkl, &work[itauq],
                       U, ldu, &work[nwork], lwork - nwork, &ierr);
                dormbr("P", "R", "T", m, m, m, &work[il], ldwrkl, &work[itaup],
                       &work[ivt], m, &work[nwork], lwork - nwork, &ierr);

                for (i = 0; i < n; i += chunk) {
                    blk = ((n - i) < chunk) ? (n - i) : chunk;
                    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                m, blk, m, ONE, &work[ivt], m, &A[i * lda], lda,
                                ZERO, &work[il], ldwrkl);
                    dlacpy("F", m, blk, &work[il], ldwrkl, &A[i * lda], lda);
                }
            } else if (wntqs) {
                /* Path 3t (N >> M, JOBZ='S') */
                il = 0;
                ldwrkl = m;
                itau = il + ldwrkl * m;
                nwork = itau + m;

                dgelqf(m, n, A, lda, &work[itau], &work[nwork], lwork - nwork, &ierr);
                dlacpy("L", m, m, A, lda, &work[il], ldwrkl);
                dlaset("U", m - 1, m - 1, ZERO, ZERO, &work[il + ldwrkl], ldwrkl);

                dorglq(m, n, m, A, lda, &work[itau], &work[nwork], lwork - nwork, &ierr);

                ie = itau;
                itauq = ie + m;
                itaup = itauq + m;
                nwork = itaup + m;

                dgebrd(m, m, &work[il], ldwrkl, S, &work[ie], &work[itauq],
                       &work[itaup], &work[nwork], lwork - nwork, &ierr);

                dbdsdc("U", "I", m, S, &work[ie], U, ldu, VT, ldvt, dum, idum,
                       &work[nwork], IWORK, info);

                dormbr("Q", "L", "N", m, m, m, &work[il], ldwrkl, &work[itauq],
                       U, ldu, &work[nwork], lwork - nwork, &ierr);
                dormbr("P", "R", "T", m, m, m, &work[il], ldwrkl, &work[itaup],
                       VT, ldvt, &work[nwork], lwork - nwork, &ierr);

                dlacpy("F", m, m, VT, ldvt, &work[il], ldwrkl);
                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                            m, n, m, ONE, &work[il], ldwrkl, A, lda, ZERO, VT, ldvt);
            } else if (wntqa) {
                /* Path 4t (N >> M, JOBZ='A') */
                ivt = 0;
                ldwkvt = m;
                itau = ivt + ldwkvt * m;
                nwork = itau + m;

                dgelqf(m, n, A, lda, &work[itau], &work[nwork], lwork - nwork, &ierr);
                dlacpy("U", m, n, A, lda, VT, ldvt);

                dorglq(n, n, m, VT, ldvt, &work[itau], &work[nwork], lwork - nwork, &ierr);

                dlaset("U", m - 1, m - 1, ZERO, ZERO, &A[lda], lda);

                ie = itau;
                itauq = ie + m;
                itaup = itauq + m;
                nwork = itaup + m;

                dgebrd(m, m, A, lda, S, &work[ie], &work[itauq], &work[itaup],
                       &work[nwork], lwork - nwork, &ierr);

                dbdsdc("U", "I", m, S, &work[ie], U, ldu, &work[ivt], ldwkvt,
                       dum, idum, &work[nwork], IWORK, info);

                dormbr("Q", "L", "N", m, m, m, A, lda, &work[itauq], U, ldu,
                       &work[nwork], lwork - nwork, &ierr);
                dormbr("P", "R", "T", m, m, m, A, lda, &work[itaup], &work[ivt],
                       ldwkvt, &work[nwork], lwork - nwork, &ierr);

                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                            m, n, m, ONE, &work[ivt], ldwkvt, VT, ldvt, ZERO, A, lda);
                dlacpy("F", m, n, A, lda, VT, ldvt);
            }
        } else {
            /* Path 5t (N > M, but not much larger) */
            ie = 0;
            itauq = ie + m;
            itaup = itauq + m;
            nwork = itaup + m;

            dgebrd(m, n, A, lda, S, &work[ie], &work[itauq], &work[itaup],
                   &work[nwork], lwork - nwork, &ierr);

            if (wntqn) {
                dbdsdc("L", "N", m, S, &work[ie], NULL, 1, NULL, 1, NULL, NULL,
                       &work[nwork], IWORK, info);
            } else if (wntqo) {
                ldwkvt = m;
                ivt = nwork;
                if (lwork >= m * n + 3 * m + bdspac) {
                    dlaset("F", m, n, ZERO, ZERO, &work[ivt], ldwkvt);
                    nwork = ivt + ldwkvt * n;
                    il = -1;
                } else {
                    nwork = ivt + ldwkvt * m;
                    il = nwork;
                    chunk = (lwork - m * m - 3 * m) / m;
                }

                dbdsdc("L", "I", m, S, &work[ie], U, ldu, &work[ivt], ldwkvt,
                       dum, idum, &work[nwork], IWORK, info);

                dormbr("Q", "L", "N", m, m, n, A, lda, &work[itauq], U, ldu,
                       &work[nwork], lwork - nwork, &ierr);

                if (lwork >= m * n + 3 * m + bdspac) {
                    dormbr("P", "R", "T", m, n, m, A, lda, &work[itaup],
                           &work[ivt], ldwkvt, &work[nwork], lwork - nwork, &ierr);
                    dlacpy("F", m, n, &work[ivt], ldwkvt, A, lda);
                } else {
                    dorgbr("P", m, n, m, A, lda, &work[itaup], &work[nwork],
                           lwork - nwork, &ierr);
                    for (i = 0; i < n; i += chunk) {
                        blk = ((n - i) < chunk) ? (n - i) : chunk;
                        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                    m, blk, m, ONE, &work[ivt], ldwkvt, &A[i * lda], lda,
                                    ZERO, &work[il], m);
                        dlacpy("F", m, blk, &work[il], m, &A[i * lda], lda);
                    }
                }
            } else if (wntqs) {
                dlaset("F", m, n, ZERO, ZERO, VT, ldvt);
                dbdsdc("L", "I", m, S, &work[ie], U, ldu, VT, ldvt, dum, idum,
                       &work[nwork], IWORK, info);

                dormbr("Q", "L", "N", m, m, n, A, lda, &work[itauq], U, ldu,
                       &work[nwork], lwork - nwork, &ierr);
                dormbr("P", "R", "T", m, n, m, A, lda, &work[itaup], VT, ldvt,
                       &work[nwork], lwork - nwork, &ierr);
            } else if (wntqa) {
                dlaset("F", n, n, ZERO, ZERO, VT, ldvt);
                dbdsdc("L", "I", m, S, &work[ie], U, ldu, VT, ldvt, dum, idum,
                       &work[nwork], IWORK, info);

                if (n > m) {
                    dlaset("F", n - m, n - m, ZERO, ONE, &VT[m + m * ldvt], ldvt);
                }

                dormbr("Q", "L", "N", m, m, n, A, lda, &work[itauq], U, ldu,
                       &work[nwork], lwork - nwork, &ierr);
                dormbr("P", "R", "T", n, n, m, A, lda, &work[itaup], VT, ldvt,
                       &work[nwork], lwork - nwork, &ierr);
            }
        }
    }

    /* Undo scaling if necessary */
    if (iscl == 1) {
        if (anrm > bignum) {
            dlascl("G", 0, 0, bignum, anrm, minmn, 1, S, minmn, &ierr);
        }
        if (anrm < smlnum) {
            dlascl("G", 0, 0, smlnum, anrm, minmn, 1, S, minmn, &ierr);
        }
    }

    /* Return optimal workspace in WORK(1) */
    work[0] = (f64)maxwrk;
}
