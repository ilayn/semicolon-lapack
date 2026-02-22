/**
 * @file dgelss.c
 * @brief DGELSS computes the minimum norm solution to a real linear least
 *        squares problem using the SVD.
 */

#include "internal_build_defs.h"
#include "semicolon_lapack_double.h"
#include "../include/lapack_tuning.h"
#include <math.h>
#include <cblas.h>

/**
 * DGELSS computes the minimum norm solution to a real linear least
 * squares problem:
 *
 *     Minimize 2-norm(| b - A*x |).
 *
 * using the singular value decomposition (SVD) of A. A is an M-by-N
 * matrix which may be rank-deficient.
 *
 * Several right hand side vectors b and solution vectors x can be
 * handled in a single call; they are stored as the columns of the
 * M-by-NRHS right hand side matrix B and the N-by-NRHS solution matrix
 * X.
 *
 * The effective rank of A is determined by treating as zero those
 * singular values which are less than RCOND times the largest singular
 * value.
 *
 * @param[in]     m      The number of rows of the matrix A. m >= 0.
 * @param[in]     n      The number of columns of the matrix A. n >= 0.
 * @param[in]     nrhs   The number of right hand sides. nrhs >= 0.
 * @param[in,out] A      Double precision array, dimension (lda, n).
 *                       On entry, the M-by-N matrix A.
 *                       On exit, the first min(m,n) rows of A are overwritten
 *                       with its right singular vectors, stored rowwise.
 * @param[in]     lda    The leading dimension of A. lda >= max(1, m).
 * @param[in,out] B      Double precision array, dimension (ldb, nrhs).
 *                       On entry, the M-by-NRHS right hand side matrix B.
 *                       On exit, B is overwritten by the N-by-NRHS solution
 *                       matrix X. If m >= n and RANK = n, the residual
 *                       sum-of-squares for the solution in the i-th column is
 *                       given by the sum of squares of elements n+1:m in that
 *                       column.
 * @param[in]     ldb    The leading dimension of B. ldb >= max(1, max(m, n)).
 * @param[out]    S      Double precision array, dimension (min(m, n)).
 *                       The singular values of A in decreasing order.
 *                       The condition number of A in the 2-norm = S(1)/S(min(m,n)).
 * @param[in]     rcond  RCOND is used to determine the effective rank of A.
 *                       Singular values S(i) <= RCOND*S(1) are treated as zero.
 *                       If RCOND < 0, machine precision is used instead.
 * @param[out]    rank   The effective rank of A, i.e., the number of singular
 *                       values which are greater than RCOND*S(1).
 * @param[out]    work   Double precision array, dimension (max(1, lwork)).
 *                       On exit, if info = 0, work[0] returns the optimal lwork.
 * @param[in]     lwork  The dimension of work. lwork >= 1, and also:
 *                       lwork >= 3*min(m,n) + max(2*min(m,n), max(m,n), nrhs).
 *                       For good performance, lwork should generally be larger.
 *                       If lwork = -1, a workspace query is assumed.
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 *                         - > 0: the algorithm for computing the SVD failed to converge;
 *                           if info = i, i off-diagonal elements of an intermediate
 *                           bidiagonal form did not converge to zero.
 */
void dgelss(const INT m, const INT n, const INT nrhs,
            f64* restrict A, const INT lda,
            f64* restrict B, const INT ldb,
            f64* restrict S, const f64 rcond, INT* rank,
            f64* restrict work, const INT lwork,
            INT* info)
{
    INT lquery;
    INT bdspac, bl, chunk, i, iascl, ibscl, ie, il, itau, itaup, itauq;
    INT iwork, ldwork, maxmn, maxwrk, minmn, minwrk, mm, mnthr;
    INT lwork_dgeqrf, lwork_dormqr, lwork_dgebrd, lwork_dormbr, lwork_dorgbr;
    INT lwork_dgelqf, lwork_dormlq;
    f64 anrm, bignum, bnrm, eps, sfmin, smlnum, thr;
    f64 wkopt[1];
    INT iinfo;

    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    /* Test the input arguments */
    *info = 0;
    minmn = m < n ? m : n;
    maxmn = m > n ? m : n;
    lquery = (lwork == -1);

    if (m < 0) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (nrhs < 0) {
        *info = -3;
    } else if (lda < (m > 1 ? m : 1)) {
        *info = -5;
    } else if (ldb < (maxmn > 1 ? maxmn : 1)) {
        *info = -7;
    }

    /* Compute workspace */
    if (*info == 0) {
        minwrk = 1;
        maxwrk = 1;
        if (minmn > 0) {
            mm = m;
            mnthr = lapack_get_mnthr("GELSS", m, n);
            if (m >= n && m >= mnthr) {
                /* Path 1a - overdetermined, with many more rows than columns */
                /* Compute space needed for DGEQRF */
                dgeqrf(m, n, A, lda, NULL, wkopt, -1, &iinfo);
                lwork_dgeqrf = (INT)wkopt[0];
                /* Compute space needed for DORMQR */
                dormqr("L", "T", m, nrhs, n, A, lda, NULL, B, ldb, wkopt, -1, &iinfo);
                lwork_dormqr = (INT)wkopt[0];
                mm = n;
                maxwrk = n + lwork_dgeqrf;
                if (n + lwork_dormqr > maxwrk) maxwrk = n + lwork_dormqr;
            }
            if (m >= n) {
                /* Path 1 - overdetermined or exactly determined */
                /* Compute workspace needed for DBDSQR */
                bdspac = 5 * n > 1 ? 5 * n : 1;
                /* Compute space needed for DGEBRD */
                dgebrd(mm, n, A, lda, S, NULL, NULL, NULL, wkopt, -1, &iinfo);
                lwork_dgebrd = (INT)wkopt[0];
                /* Compute space needed for DORMBR */
                dormbr("Q", "L", "T", mm, nrhs, n, A, lda, NULL, B, ldb, wkopt, -1, &iinfo);
                lwork_dormbr = (INT)wkopt[0];
                /* Compute space needed for DORGBR */
                dorgbr("P", n, n, n, A, lda, NULL, wkopt, -1, &iinfo);
                lwork_dorgbr = (INT)wkopt[0];
                /* Compute total workspace needed */
                if (3 * n + lwork_dgebrd > maxwrk) maxwrk = 3 * n + lwork_dgebrd;
                if (3 * n + lwork_dormbr > maxwrk) maxwrk = 3 * n + lwork_dormbr;
                if (3 * n + lwork_dorgbr > maxwrk) maxwrk = 3 * n + lwork_dorgbr;
                if (bdspac > maxwrk) maxwrk = bdspac;
                if (n * nrhs > maxwrk) maxwrk = n * nrhs;
                minwrk = 3 * n + mm;
                if (3 * n + nrhs > minwrk) minwrk = 3 * n + nrhs;
                if (bdspac > minwrk) minwrk = bdspac;
                if (maxwrk < minwrk) maxwrk = minwrk;
            }
            if (n > m) {
                /* Compute workspace needed for DBDSQR */
                bdspac = 5 * m > 1 ? 5 * m : 1;
                minwrk = 3 * m + nrhs;
                if (3 * m + n > minwrk) minwrk = 3 * m + n;
                if (bdspac > minwrk) minwrk = bdspac;
                if (n >= mnthr) {
                    /* Path 2a - underdetermined, with many more columns than rows */
                    /* Compute space needed for DGELQF */
                    dgelqf(m, n, A, lda, NULL, wkopt, -1, &iinfo);
                    lwork_dgelqf = (INT)wkopt[0];
                    /* Compute space needed for DGEBRD */
                    dgebrd(m, m, A, lda, S, NULL, NULL, NULL, wkopt, -1, &iinfo);
                    lwork_dgebrd = (INT)wkopt[0];
                    /* Compute space needed for DORMBR */
                    dormbr("Q", "L", "T", m, nrhs, n, A, lda, NULL, B, ldb, wkopt, -1, &iinfo);
                    lwork_dormbr = (INT)wkopt[0];
                    /* Compute space needed for DORGBR */
                    dorgbr("P", m, m, m, A, lda, NULL, wkopt, -1, &iinfo);
                    lwork_dorgbr = (INT)wkopt[0];
                    /* Compute space needed for DORMLQ */
                    dormlq("L", "T", n, nrhs, m, A, lda, NULL, B, ldb, wkopt, -1, &iinfo);
                    lwork_dormlq = (INT)wkopt[0];
                    /* Compute total workspace needed */
                    maxwrk = m + lwork_dgelqf;
                    if (m * m + 4 * m + lwork_dgebrd > maxwrk) maxwrk = m * m + 4 * m + lwork_dgebrd;
                    if (m * m + 4 * m + lwork_dormbr > maxwrk) maxwrk = m * m + 4 * m + lwork_dormbr;
                    if (m * m + 4 * m + lwork_dorgbr > maxwrk) maxwrk = m * m + 4 * m + lwork_dorgbr;
                    if (m * m + m + bdspac > maxwrk) maxwrk = m * m + m + bdspac;
                    if (nrhs > 1) {
                        if (m * m + m + m * nrhs > maxwrk) maxwrk = m * m + m + m * nrhs;
                    } else {
                        if (m * m + 2 * m > maxwrk) maxwrk = m * m + 2 * m;
                    }
                    if (m + lwork_dormlq > maxwrk) maxwrk = m + lwork_dormlq;
                } else {
                    /* Path 2 - underdetermined */
                    /* Compute space needed for DGEBRD */
                    dgebrd(m, n, A, lda, S, NULL, NULL, NULL, wkopt, -1, &iinfo);
                    lwork_dgebrd = (INT)wkopt[0];
                    /* Compute space needed for DORMBR */
                    dormbr("Q", "L", "T", m, nrhs, m, A, lda, NULL, B, ldb, wkopt, -1, &iinfo);
                    lwork_dormbr = (INT)wkopt[0];
                    /* Compute space needed for DORGBR */
                    dorgbr("P", m, n, m, A, lda, NULL, wkopt, -1, &iinfo);
                    lwork_dorgbr = (INT)wkopt[0];
                    maxwrk = 3 * m + lwork_dgebrd;
                    if (3 * m + lwork_dormbr > maxwrk) maxwrk = 3 * m + lwork_dormbr;
                    if (3 * m + lwork_dorgbr > maxwrk) maxwrk = 3 * m + lwork_dorgbr;
                    if (bdspac > maxwrk) maxwrk = bdspac;
                    if (n * nrhs > maxwrk) maxwrk = n * nrhs;
                }
            }
            if (maxwrk < minwrk) maxwrk = minwrk;
        }
        work[0] = (f64)maxwrk;

        if (lwork < minwrk && !lquery) {
            *info = -12;
        }
    }

    if (*info != 0) {
        xerbla("DGELSS", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    /* Quick return if possible */
    if (m == 0 || n == 0) {
        *rank = 0;
        return;
    }

    /* Get machine parameters */
    eps = dlamch("P");
    sfmin = dlamch("S");
    smlnum = sfmin / eps;
    bignum = ONE / smlnum;

    /* Scale A if max element outside range [SMLNUM, BIGNUM] */
    anrm = dlange("M", m, n, A, lda, work);
    iascl = 0;
    if (anrm > ZERO && anrm < smlnum) {
        /* Scale matrix norm up to SMLNUM */
        dlascl("G", 0, 0, anrm, smlnum, m, n, A, lda, &iinfo);
        iascl = 1;
    } else if (anrm > bignum) {
        /* Scale matrix norm down to BIGNUM */
        dlascl("G", 0, 0, anrm, bignum, m, n, A, lda, &iinfo);
        iascl = 2;
    } else if (anrm == ZERO) {
        /* Matrix all zero. Return zero solution. */
        dlaset("F", maxmn, nrhs, ZERO, ZERO, B, ldb);
        dlaset("F", minmn, 1, ZERO, ZERO, S, minmn);
        *rank = 0;
        goto cleanup;
    }

    /* Scale B if max element outside range [SMLNUM, BIGNUM] */
    bnrm = dlange("M", m, nrhs, B, ldb, work);
    ibscl = 0;
    if (bnrm > ZERO && bnrm < smlnum) {
        /* Scale matrix norm up to SMLNUM */
        dlascl("G", 0, 0, bnrm, smlnum, m, nrhs, B, ldb, &iinfo);
        ibscl = 1;
    } else if (bnrm > bignum) {
        /* Scale matrix norm down to BIGNUM */
        dlascl("G", 0, 0, bnrm, bignum, m, nrhs, B, ldb, &iinfo);
        ibscl = 2;
    }

    /* Overdetermined case */
    if (m >= n) {
        /* Path 1 - overdetermined or exactly determined */
        mm = m;
        mnthr = lapack_get_mnthr("GELSS", m, n);
        if (m >= mnthr) {
            /* Path 1a - overdetermined, with many more rows than columns */
            mm = n;
            itau = 0;
            iwork = itau + n;

            /* Compute A=Q*R */
            dgeqrf(m, n, A, lda, &work[itau], &work[iwork], lwork - iwork, &iinfo);

            /* Multiply B by transpose(Q) */
            dormqr("L", "T", m, nrhs, n, A, lda, &work[itau], B, ldb,
                   &work[iwork], lwork - iwork, &iinfo);

            /* Zero out below R */
            if (n > 1) {
                dlaset("L", n - 1, n - 1, ZERO, ZERO, &A[1], lda);
            }
        }

        ie = 0;
        itauq = ie + n;
        itaup = itauq + n;
        iwork = itaup + n;

        /* Bidiagonalize R in A */
        dgebrd(mm, n, A, lda, S, &work[ie], &work[itauq], &work[itaup],
               &work[iwork], lwork - iwork, &iinfo);

        /* Multiply B by transpose of left bidiagonalizing vectors of R */
        dormbr("Q", "L", "T", mm, nrhs, n, A, lda, &work[itauq],
               B, ldb, &work[iwork], lwork - iwork, &iinfo);

        /* Generate right bidiagonalizing vectors of R in A */
        dorgbr("P", n, n, n, A, lda, &work[itaup], &work[iwork], lwork - iwork, &iinfo);
        iwork = ie + n;

        /* Perform bidiagonal QR iteration
         *   multiply B by transpose of left singular vectors
         *   compute right singular vectors in A */
        dbdsqr("U", n, n, 0, nrhs, S, &work[ie], A, lda, NULL, 1, B, ldb,
               &work[iwork], info);
        if (*info != 0) {
            goto cleanup;
        }

        /* Multiply B by reciprocals of singular values */
        thr = rcond * S[0];
        if (thr < ZERO) {
            thr = eps * S[0];
        }
        if (thr < sfmin) {
            thr = sfmin;
        }
        *rank = 0;
        for (i = 0; i < n; i++) {
            if (S[i] > thr) {
                drscl(nrhs, S[i], &B[i], ldb);
                (*rank)++;
            } else {
                dlaset("F", 1, nrhs, ZERO, ZERO, &B[i], ldb);
            }
        }

        /* Multiply B by right singular vectors */
        if (lwork >= ldb * nrhs && nrhs > 1) {
            cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                        n, nrhs, n, ONE, A, lda, B, ldb, ZERO, work, ldb);
            dlacpy("G", n, nrhs, work, ldb, B, ldb);
        } else if (nrhs > 1) {
            chunk = lwork / n;
            for (i = 0; i < nrhs; i += chunk) {
                bl = nrhs - i < chunk ? nrhs - i : chunk;
                cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                            n, bl, n, ONE, A, lda, &B[i * ldb], ldb, ZERO, work, n);
                dlacpy("G", n, bl, work, n, &B[i * ldb], ldb);
            }
        } else if (nrhs == 1) {
            cblas_dgemv(CblasColMajor, CblasTrans, n, n, ONE, A, lda, B, 1, ZERO, work, 1);
            cblas_dcopy(n, work, 1, B, 1);
        }

    } else if (n >= lapack_get_mnthr("GELSS", m, n) && lwork >= 4 * m + m * m +
               (m > 2 * m - 4 ? m : 2 * m - 4) +
               (nrhs > n - 3 * m ? nrhs : n - 3 * m)) {
        /* Path 2a - underdetermined, with many more columns than rows
         * and sufficient workspace for an efficient algorithm */
        ldwork = m;
        INT temp1 = m > 2 * m - 4 ? m : 2 * m - 4;
        temp1 = temp1 > nrhs ? temp1 : nrhs;
        temp1 = temp1 > n - 3 * m ? temp1 : n - 3 * m;
        if (lwork >= 4 * m + m * lda + temp1) {
            if (lwork >= m * lda + m + m * nrhs) {
                ldwork = lda;
            }
        }
        itau = 0;
        iwork = m;

        /* Compute A=L*Q */
        dgelqf(m, n, A, lda, &work[itau], &work[iwork], lwork - iwork, &iinfo);
        il = iwork;

        /* Copy L to WORK(IL), zeroing out above it */
        dlacpy("L", m, m, A, lda, &work[il], ldwork);
        dlaset("U", m - 1, m - 1, ZERO, ZERO, &work[il + ldwork], ldwork);
        ie = il + ldwork * m;
        itauq = ie + m;
        itaup = itauq + m;
        iwork = itaup + m;

        /* Bidiagonalize L in WORK(IL) */
        dgebrd(m, m, &work[il], ldwork, S, &work[ie], &work[itauq],
               &work[itaup], &work[iwork], lwork - iwork, &iinfo);

        /* Multiply B by transpose of left bidiagonalizing vectors of L */
        dormbr("Q", "L", "T", m, nrhs, m, &work[il], ldwork, &work[itauq],
               B, ldb, &work[iwork], lwork - iwork, &iinfo);

        /* Generate right bidiagonalizing vectors of R in WORK(IL) */
        dorgbr("P", m, m, m, &work[il], ldwork, &work[itaup],
               &work[iwork], lwork - iwork, &iinfo);
        iwork = ie + m;

        /* Perform bidiagonal QR iteration,
         *   computing right singular vectors of L in WORK(IL) and
         *   multiplying B by transpose of left singular vectors */
        dbdsqr("U", m, m, 0, nrhs, S, &work[ie], &work[il], ldwork,
               A, lda, B, ldb, &work[iwork], info);
        if (*info != 0) {
            goto cleanup;
        }

        /* Multiply B by reciprocals of singular values */
        thr = rcond * S[0];
        if (thr < ZERO) {
            thr = eps * S[0];
        }
        if (thr < sfmin) {
            thr = sfmin;
        }
        *rank = 0;
        for (i = 0; i < m; i++) {
            if (S[i] > thr) {
                drscl(nrhs, S[i], &B[i], ldb);
                (*rank)++;
            } else {
                dlaset("F", 1, nrhs, ZERO, ZERO, &B[i], ldb);
            }
        }
        iwork = ie;

        /* Multiply B by right singular vectors of L in WORK(IL) */
        if (lwork >= ldb * nrhs + iwork && nrhs > 1) {
            cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                        m, nrhs, m, ONE, &work[il], ldwork, B, ldb, ZERO, &work[iwork], ldb);
            dlacpy("G", m, nrhs, &work[iwork], ldb, B, ldb);
        } else if (nrhs > 1) {
            chunk = (lwork - iwork) / m;
            for (i = 0; i < nrhs; i += chunk) {
                bl = nrhs - i < chunk ? nrhs - i : chunk;
                cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                            m, bl, m, ONE, &work[il], ldwork, &B[i * ldb], ldb, ZERO, &work[iwork], m);
                dlacpy("G", m, bl, &work[iwork], m, &B[i * ldb], ldb);
            }
        } else if (nrhs == 1) {
            cblas_dgemv(CblasColMajor, CblasTrans, m, m, ONE, &work[il], ldwork, B, 1, ZERO, &work[iwork], 1);
            cblas_dcopy(m, &work[iwork], 1, B, 1);
        }

        /* Zero out below first M rows of B */
        dlaset("F", n - m, nrhs, ZERO, ZERO, &B[m], ldb);
        iwork = itau + m;

        /* Multiply transpose(Q) by B */
        dormlq("L", "T", n, nrhs, m, A, lda, &work[itau], B, ldb,
               &work[iwork], lwork - iwork, &iinfo);

    } else {
        /* Path 2 - remaining underdetermined cases */
        ie = 0;
        itauq = ie + m;
        itaup = itauq + m;
        iwork = itaup + m;

        /* Bidiagonalize A */
        dgebrd(m, n, A, lda, S, &work[ie], &work[itauq], &work[itaup],
               &work[iwork], lwork - iwork, &iinfo);

        /* Multiply B by transpose of left bidiagonalizing vectors */
        dormbr("Q", "L", "T", m, nrhs, n, A, lda, &work[itauq],
               B, ldb, &work[iwork], lwork - iwork, &iinfo);

        /* Generate right bidiagonalizing vectors in A */
        dorgbr("P", m, n, m, A, lda, &work[itaup], &work[iwork], lwork - iwork, &iinfo);
        iwork = ie + m;

        /* Perform bidiagonal QR iteration,
         *   computing right singular vectors of A in A and
         *   multiplying B by transpose of left singular vectors */
        dbdsqr("L", m, n, 0, nrhs, S, &work[ie], A, lda, NULL, 1, B, ldb,
               &work[iwork], info);
        if (*info != 0) {
            goto cleanup;
        }

        /* Multiply B by reciprocals of singular values */
        thr = rcond * S[0];
        if (thr < ZERO) {
            thr = eps * S[0];
        }
        if (thr < sfmin) {
            thr = sfmin;
        }
        *rank = 0;
        for (i = 0; i < m; i++) {
            if (S[i] > thr) {
                drscl(nrhs, S[i], &B[i], ldb);
                (*rank)++;
            } else {
                dlaset("F", 1, nrhs, ZERO, ZERO, &B[i], ldb);
            }
        }

        /* Multiply B by right singular vectors of A */
        if (lwork >= ldb * nrhs && nrhs > 1) {
            cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                        n, nrhs, m, ONE, A, lda, B, ldb, ZERO, work, ldb);
            dlacpy("F", n, nrhs, work, ldb, B, ldb);
        } else if (nrhs > 1) {
            chunk = lwork / n;
            for (i = 0; i < nrhs; i += chunk) {
                bl = nrhs - i < chunk ? nrhs - i : chunk;
                cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                            n, bl, m, ONE, A, lda, &B[i * ldb], ldb, ZERO, work, n);
                dlacpy("F", n, bl, work, n, &B[i * ldb], ldb);
            }
        } else if (nrhs == 1) {
            cblas_dgemv(CblasColMajor, CblasTrans, m, n, ONE, A, lda, B, 1, ZERO, work, 1);
            cblas_dcopy(n, work, 1, B, 1);
        }
    }

    /* Undo scaling */
    if (iascl == 1) {
        dlascl("G", 0, 0, anrm, smlnum, n, nrhs, B, ldb, &iinfo);
        dlascl("G", 0, 0, smlnum, anrm, minmn, 1, S, minmn, &iinfo);
    } else if (iascl == 2) {
        dlascl("G", 0, 0, anrm, bignum, n, nrhs, B, ldb, &iinfo);
        dlascl("G", 0, 0, bignum, anrm, minmn, 1, S, minmn, &iinfo);
    }
    if (ibscl == 1) {
        dlascl("G", 0, 0, smlnum, bnrm, n, nrhs, B, ldb, &iinfo);
    } else if (ibscl == 2) {
        dlascl("G", 0, 0, bignum, bnrm, n, nrhs, B, ldb, &iinfo);
    }

cleanup:
    work[0] = (f64)maxwrk;
}
