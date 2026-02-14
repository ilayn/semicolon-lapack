/**
 * @file dgelsd.c
 * @brief DGELSD computes the minimum norm solution to a real linear least
 *        squares problem using the SVD with divide and conquer.
 */

#include "semicolon_lapack_double.h"
#include "../include/lapack_tuning.h"
#include <math.h>
#include <cblas.h>

/* SMLSIZ from ilaenv ISPEC=9: maximum size of subproblems at bottom of D&C tree */
#define SMLSIZ 25

/**
 * DGELSD computes the minimum-norm solution to a real linear least
 * squares problem:
 *     minimize 2-norm(| b - A*x |)
 * using the singular value decomposition (SVD) of A. A is an M-by-N
 * matrix which may be rank-deficient.
 *
 * Several right hand side vectors b and solution vectors x can be
 * handled in a single call; they are stored as the columns of the
 * M-by-NRHS right hand side matrix B and the N-by-NRHS solution
 * matrix X.
 *
 * The problem is solved in three steps:
 * (1) Reduce the coefficient matrix A to bidiagonal form with
 *     Householder transformations, reducing the original problem
 *     into a "bidiagonal least squares problem" (BLS)
 * (2) Solve the BLS using a divide and conquer approach.
 * (3) Apply back all the Householder transformations to solve
 *     the original least squares problem.
 *
 * The effective rank of A is determined by treating as zero those
 * singular values which are less than RCOND times the largest singular
 * value.
 *
 * @param[in]     m      The number of rows of A. m >= 0.
 * @param[in]     n      The number of columns of A. n >= 0.
 * @param[in]     nrhs   The number of right hand sides. nrhs >= 0.
 * @param[in,out] A      Double precision array, dimension (lda, n).
 *                       On entry, the M-by-N matrix A.
 *                       On exit, A has been destroyed.
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
 * @param[in]     lwork  The dimension of work. lwork must be at least 1.
 *                       The exact minimum amount of workspace needed depends on M,
 *                       N and NRHS. As long as LWORK is at least
 *                           12*N + 2*N*SMLSIZ + 8*N*NLVL + N*NRHS + (SMLSIZ+1)^2,
 *                       if M is greater than or equal to N or
 *                           12*M + 2*M*SMLSIZ + 8*M*NLVL + M*NRHS + (SMLSIZ+1)^2,
 *                       if M is less than N, the code will execute correctly.
 *                       SMLSIZ is returned by ILAENV and is equal to the maximum
 *                       size of the subproblems at the bottom of the computation
 *                       tree (usually about 25), and
 *                          NLVL = MAX( 0, INT( LOG_2( MIN( M,N )/(SMLSIZ+1) ) ) + 1 )
 *                       For good performance, LWORK should generally be larger.
 *                       If LWORK = -1, a workspace query is assumed.
 * @param[out]    iwork  Integer array, dimension (max(1, liwork)).
 *                       liwork >= max(1, 3 * MINMN * NLVL + 11 * MINMN),
 *                       where MINMN = MIN( M,N ).
 *                       On exit, if INFO = 0, IWORK(1) returns the minimum LIWORK.
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 *                         - > 0: the algorithm for computing the SVD failed to converge;
 *                           if info = i, i off-diagonal elements of an intermediate
 *                           bidiagonal form did not converge to zero.
 */
void dgelsd(const int m, const int n, const int nrhs,
            f64* restrict A, const int lda,
            f64* restrict B, const int ldb,
            f64* restrict S, const f64 rcond, int* rank,
            f64* restrict work, const int lwork,
            int* restrict iwork, int* info)
{
    int lquery;
    int iascl, ibscl, ie, il, itau, itaup, itauq, ldwork, liwork;
    int maxmn, maxwrk, minmn, minwrk, mm, mnthr, nlvl, nwork, smlsiz, wlalsd;
    f64 anrm, bignum, bnrm, eps, sfmin, smlnum;
    int iinfo;
    int nb;

    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const f64 TWO = 2.0;

    /* Test the input arguments */
    *info = 0;
    fflush(stderr);
    minmn = m < n ? m : n;
    maxmn = m > n ? m : n;
    mnthr = lapack_get_mnthr("GELSS", m, n);
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

    smlsiz = SMLSIZ;

    /* Compute workspace */
    minwrk = 1;
    liwork = 1;
    if (minmn > 0) {
        minmn = minmn > 1 ? minmn : 1;
    }
    nlvl = (int)(log((f64)minmn / (f64)(smlsiz + 1)) / log(TWO)) + 1;
    if (nlvl < 0) nlvl = 0;

    if (*info == 0) {
        maxwrk = 1;
        liwork = 3 * minmn * nlvl + 11 * minmn;
        mm = m;
        if (m >= n && m >= mnthr) {
            /* Path 1a - overdetermined, with many more rows than columns */
            mm = n;
            nb = lapack_get_nb("GEQRF");
            maxwrk = n + n * nb;
            nb = lapack_get_nb("ORMQR");
            if (n + nrhs * nb > maxwrk) maxwrk = n + nrhs * nb;
        }
        if (m >= n) {
            /* Path 1 - overdetermined or exactly determined */
            nb = lapack_get_nb("GEBRD");
            if (3 * n + (mm + n) * nb > maxwrk) maxwrk = 3 * n + (mm + n) * nb;
            nb = lapack_get_nb("ORMBR");
            if (3 * n + nrhs * nb > maxwrk) maxwrk = 3 * n + nrhs * nb;
            if (3 * n + (n - 1) * nb > maxwrk) maxwrk = 3 * n + (n - 1) * nb;
            wlalsd = 9 * n + 2 * n * smlsiz + 8 * n * nlvl + n * nrhs + (smlsiz + 1) * (smlsiz + 1);
            if (3 * n + wlalsd > maxwrk) maxwrk = 3 * n + wlalsd;
            minwrk = 3 * n + mm;
            if (3 * n + nrhs > minwrk) minwrk = 3 * n + nrhs;
            if (3 * n + wlalsd > minwrk) minwrk = 3 * n + wlalsd;
        }
        if (n > m) {
            wlalsd = 9 * m + 2 * m * smlsiz + 8 * m * nlvl + m * nrhs + (smlsiz + 1) * (smlsiz + 1);
            if (n >= mnthr) {
                /* Path 2a - underdetermined, with many more columns than rows */
                nb = lapack_get_nb("GELQF");
                maxwrk = m + m * nb;
                nb = lapack_get_nb("GEBRD");
                if (m * m + 4 * m + 2 * m * nb > maxwrk) maxwrk = m * m + 4 * m + 2 * m * nb;
                nb = lapack_get_nb("ORMBR");
                if (m * m + 4 * m + nrhs * nb > maxwrk) maxwrk = m * m + 4 * m + nrhs * nb;
                if (m * m + 4 * m + (m - 1) * nb > maxwrk) maxwrk = m * m + 4 * m + (m - 1) * nb;
                if (nrhs > 1) {
                    if (m * m + m + m * nrhs > maxwrk) maxwrk = m * m + m + m * nrhs;
                } else {
                    if (m * m + 2 * m > maxwrk) maxwrk = m * m + 2 * m;
                }
                nb = lapack_get_nb("ORMLQ");
                if (m + nrhs * nb > maxwrk) maxwrk = m + nrhs * nb;
                if (m * m + 4 * m + wlalsd > maxwrk) maxwrk = m * m + 4 * m + wlalsd;
                /* Ensure the Path 2a case below is triggered */
                int temp = m > 2 * m - 4 ? m : 2 * m - 4;
                temp = temp > nrhs ? temp : nrhs;
                temp = temp > n - 3 * m ? temp : n - 3 * m;
                if (4 * m + m * m + temp > maxwrk) maxwrk = 4 * m + m * m + temp;
            } else {
                /* Path 2 - remaining underdetermined cases */
                nb = lapack_get_nb("GEBRD");
                maxwrk = 3 * m + (n + m) * nb;
                nb = lapack_get_nb("ORMBR");
                if (3 * m + nrhs * nb > maxwrk) maxwrk = 3 * m + nrhs * nb;
                if (3 * m + m * nb > maxwrk) maxwrk = 3 * m + m * nb;
                if (3 * m + wlalsd > maxwrk) maxwrk = 3 * m + wlalsd;
            }
            minwrk = 3 * m + nrhs;
            if (3 * m + m > minwrk) minwrk = 3 * m + m;
            if (3 * m + wlalsd > minwrk) minwrk = 3 * m + wlalsd;
        }
        if (minwrk > maxwrk) minwrk = maxwrk;
        work[0] = (f64)maxwrk;
        iwork[0] = liwork;

        if (lwork < minwrk && !lquery) {
            *info = -12;
        }
    }

    if (*info != 0) {
        xerbla("DGELSD", -(*info));
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

    /* Scale A if max entry outside range [SMLNUM, BIGNUM] */
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
        dlaset("F", minmn, 1, ZERO, ZERO, S, 1);
        *rank = 0;
        goto cleanup;
    }

    /* Scale B if max entry outside range [SMLNUM, BIGNUM] */
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

    /* If M < N make sure certain entries of B are zero */
    if (m < n) {
        dlaset("F", n - m, nrhs, ZERO, ZERO, &B[m], ldb);
    }

    /* Overdetermined case */
    if (m >= n) {
        /* Path 1 - overdetermined or exactly determined */
        mm = m;
        if (m >= mnthr) {
            /* Path 1a - overdetermined, with many more rows than columns */
            mm = n;
            itau = 0;
            nwork = itau + n;

            /* Compute A=Q*R */
            dgeqrf(m, n, A, lda, &work[itau], &work[nwork], lwork - nwork, &iinfo);

            /* Multiply B by transpose(Q) */
            dormqr("L", "T", m, nrhs, n, A, lda, &work[itau], B, ldb,
                   &work[nwork], lwork - nwork, &iinfo);

            /* Zero out below R */
            if (n > 1) {
                dlaset("L", n - 1, n - 1, ZERO, ZERO, &A[1], lda);
            }
        }

        ie = 0;
        itauq = ie + n;
        itaup = itauq + n;
        nwork = itaup + n;

        /* Bidiagonalize R in A */
        dgebrd(mm, n, A, lda, S, &work[ie], &work[itauq], &work[itaup],
               &work[nwork], lwork - nwork, &iinfo);

        /* Multiply B by transpose of left bidiagonalizing vectors of R */
        dormbr("Q", "L", "T", mm, nrhs, n, A, lda, &work[itauq],
               B, ldb, &work[nwork], lwork - nwork, &iinfo);

        /* Solve the bidiagonal least squares problem */
        dlalsd("U", smlsiz, n, nrhs, S, &work[ie], B, ldb, rcond, rank,
               &work[nwork], iwork, info);
        if (*info != 0) {
            goto cleanup;
        }

        /* Multiply B by right bidiagonalizing vectors of R */
        dormbr("P", "L", "N", n, nrhs, n, A, lda, &work[itaup],
               B, ldb, &work[nwork], lwork - nwork, &iinfo);

    } else if (n >= mnthr && lwork >= 4 * m + m * m +
               (m > 2 * m - 4 ? m : 2 * m - 4) +
               (nrhs > n - 3 * m ? nrhs : n - 3 * m)) {
        /* Path 2a - underdetermined, with many more columns than rows
         * and sufficient workspace for an efficient algorithm */
        ldwork = m;
        int temp = m > 2 * m - 4 ? m : 2 * m - 4;
        temp = temp > nrhs ? temp : nrhs;
        temp = temp > n - 3 * m ? temp : n - 3 * m;
        wlalsd = 9 * m + 2 * m * smlsiz + 8 * m * nlvl + m * nrhs + (smlsiz + 1) * (smlsiz + 1);
        if (lwork >= 4 * m + m * lda + temp) {
            if (lwork >= m * lda + m + m * nrhs) {
                if (lwork >= 4 * m + m * lda + wlalsd) {
                    ldwork = lda;
                }
            }
        }
        itau = 0;
        nwork = m;

        /* Compute A=L*Q */
        dgelqf(m, n, A, lda, &work[itau], &work[nwork], lwork - nwork, &iinfo);
        il = nwork;

        /* Copy L to WORK(IL), zeroing out above its diagonal */
        dlacpy("L", m, m, A, lda, &work[il], ldwork);
        dlaset("U", m - 1, m - 1, ZERO, ZERO, &work[il + ldwork], ldwork);
        ie = il + ldwork * m;
        itauq = ie + m;
        itaup = itauq + m;
        nwork = itaup + m;

        /* Bidiagonalize L in WORK(IL) */
        dgebrd(m, m, &work[il], ldwork, S, &work[ie], &work[itauq],
               &work[itaup], &work[nwork], lwork - nwork, &iinfo);

        /* Multiply B by transpose of left bidiagonalizing vectors of L */
        dormbr("Q", "L", "T", m, nrhs, m, &work[il], ldwork, &work[itauq],
               B, ldb, &work[nwork], lwork - nwork, &iinfo);

        /* Solve the bidiagonal least squares problem */
        dlalsd("U", smlsiz, m, nrhs, S, &work[ie], B, ldb, rcond, rank,
               &work[nwork], iwork, info);
        if (*info != 0) {
            goto cleanup;
        }

        /* Multiply B by right bidiagonalizing vectors of L */
        dormbr("P", "L", "N", m, nrhs, m, &work[il], ldwork, &work[itaup],
               B, ldb, &work[nwork], lwork - nwork, &iinfo);

        /* Zero out below first M rows of B */
        dlaset("F", n - m, nrhs, ZERO, ZERO, &B[m], ldb);
        nwork = itau + m;

        /* Multiply transpose(Q) by B */
        dormlq("L", "T", n, nrhs, m, A, lda, &work[itau], B, ldb,
               &work[nwork], lwork - nwork, &iinfo);

    } else {
        /* Path 2 - remaining underdetermined cases */
        ie = 0;
        itauq = ie + m;
        itaup = itauq + m;
        nwork = itaup + m;

        /* Bidiagonalize A */
        dgebrd(m, n, A, lda, S, &work[ie], &work[itauq], &work[itaup],
               &work[nwork], lwork - nwork, &iinfo);

        /* Multiply B by transpose of left bidiagonalizing vectors */
        dormbr("Q", "L", "T", m, nrhs, n, A, lda, &work[itauq],
               B, ldb, &work[nwork], lwork - nwork, &iinfo);

        /* Solve the bidiagonal least squares problem */
        dlalsd("L", smlsiz, m, nrhs, S, &work[ie], B, ldb, rcond, rank,
               &work[nwork], iwork, info);
        if (*info != 0) {
            goto cleanup;
        }

        /* Multiply B by right bidiagonalizing vectors of A */
        dormbr("P", "L", "N", n, nrhs, m, A, lda, &work[itaup],
               B, ldb, &work[nwork], lwork - nwork, &iinfo);
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
    iwork[0] = liwork;
}
