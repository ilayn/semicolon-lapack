/**
 * @file cgelsd.c
 * @brief CGELSD computes the minimum norm solution to a complex linear least
 *        squares problem using the SVD with divide and conquer.
 */

#include "semicolon_lapack_complex_single.h"
#include "../include/lapack_tuning.h"
#include <complex.h>
#include <math.h>
#include <cblas.h>

/* SMLSIZ from ilaenv ISPEC=9: maximum size of subproblems at bottom of D&C tree */
#define SMLSIZ 25

/**
 * CGELSD computes the minimum-norm solution to a complex linear least
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
 * @param[in,out] A      Complex*16 array, dimension (lda, n).
 *                       On entry, the M-by-N matrix A.
 *                       On exit, A has been destroyed.
 * @param[in]     lda    The leading dimension of A. lda >= max(1, m).
 * @param[in,out] B      Complex*16 array, dimension (ldb, nrhs).
 *                       On entry, the M-by-NRHS right hand side matrix B.
 *                       On exit, B is overwritten by the N-by-NRHS solution
 *                       matrix X. If m >= n and RANK = n, the residual
 *                       sum-of-squares for the solution in the i-th column is
 *                       given by the sum of squares of the modulus of elements
 *                       n+1:m in that column.
 * @param[in]     ldb    The leading dimension of B. ldb >= max(1, max(m, n)).
 * @param[out]    S      Single precision array, dimension (min(m, n)).
 *                       The singular values of A in decreasing order.
 *                       The condition number of A in the 2-norm = S(1)/S(min(m,n)).
 * @param[in]     rcond  RCOND is used to determine the effective rank of A.
 *                       Singular values S(i) <= RCOND*S(1) are treated as zero.
 *                       If RCOND < 0, machine precision is used instead.
 * @param[out]    rank   The effective rank of A, i.e., the number of singular
 *                       values which are greater than RCOND*S(1).
 * @param[out]    work   Complex*16 array, dimension (max(1, lwork)).
 *                       On exit, if info = 0, work[0] returns the optimal lwork.
 * @param[in]     lwork  The dimension of work. lwork must be at least 1.
 *                       If LWORK = -1, a workspace query is assumed.
 * @param[out]    rwork  Single precision array, dimension (max(1, lrwork)).
 *                       On exit, if info = 0, rwork[0] returns the minimum lrwork.
 * @param[out]    iwork  Integer array, dimension (max(1, liwork)).
 *                       On exit, if INFO = 0, IWORK(1) returns the minimum LIWORK.
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 *                         - > 0: the algorithm for computing the SVD failed to converge;
 *                           if info = i, i off-diagonal elements of an intermediate
 *                           bidiagonal form did not converge to zero.
 */
void cgelsd(const int m, const int n, const int nrhs,
            c64* restrict A, const int lda,
            c64* restrict B, const int ldb,
            f32* restrict S, const f32 rcond, int* rank,
            c64* restrict work, const int lwork,
            f32* restrict rwork,
            int* restrict iwork, int* info)
{
    int lquery;
    int iascl, ibscl, ie, il, itau, itaup, itauq, ldwork, liwork, lrwork;
    int maxmn, maxwrk, minmn, minwrk, mm, mnthr, nlvl, nrwork, nwork, smlsiz;
    f32 anrm, bignum, bnrm, eps, sfmin, smlnum;
    int iinfo;
    int nb;

    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;
    const f32 TWO = 2.0f;
    static const c64 CZERO = CMPLXF(0.0f, 0.0f);

    *info = 0;
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
    maxwrk = 1;
    liwork = 1;
    lrwork = 1;
    if (minmn > 0) {
        nlvl = (int)(logf((f32)minmn / (f32)(smlsiz + 1)) / logf(TWO)) + 1;
        if (nlvl < 0) nlvl = 0;
        liwork = 3 * minmn * nlvl + 11 * minmn;
        mm = m;
        if (m >= n && m >= mnthr) {
            mm = n;
            nb = lapack_get_nb("GEQRF");
            maxwrk = n * nb;
            nb = lapack_get_nb("ORMQR");
            if (nrhs * nb > maxwrk) maxwrk = nrhs * nb;
        }
        if (m >= n) {
            lrwork = 10 * n + 2 * n * smlsiz + 8 * n * nlvl + 3 * smlsiz * nrhs +
                     ((smlsiz + 1) * (smlsiz + 1) > n * (1 + nrhs) + 2 * nrhs ?
                      (smlsiz + 1) * (smlsiz + 1) : n * (1 + nrhs) + 2 * nrhs);
            nb = lapack_get_nb("GEBRD");
            if (2 * n + (mm + n) * nb > maxwrk) maxwrk = 2 * n + (mm + n) * nb;
            nb = lapack_get_nb("ORMBR");
            if (2 * n + nrhs * nb > maxwrk) maxwrk = 2 * n + nrhs * nb;
            if (2 * n + (n - 1) * nb > maxwrk) maxwrk = 2 * n + (n - 1) * nb;
            if (2 * n + n * nrhs > maxwrk) maxwrk = 2 * n + n * nrhs;
            minwrk = 2 * n + mm;
            if (2 * n + n * nrhs > minwrk) minwrk = 2 * n + n * nrhs;
        }
        if (n > m) {
            lrwork = 10 * m + 2 * m * smlsiz + 8 * m * nlvl + 3 * smlsiz * nrhs +
                     ((smlsiz + 1) * (smlsiz + 1) > n * (1 + nrhs) + 2 * nrhs ?
                      (smlsiz + 1) * (smlsiz + 1) : n * (1 + nrhs) + 2 * nrhs);
            if (n >= mnthr) {
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
                if (m * m + 4 * m + m * nrhs > maxwrk) maxwrk = m * m + 4 * m + m * nrhs;
                /* Ensure the Path 2a case below is triggered */
                int temp = m > 2 * m - 4 ? m : 2 * m - 4;
                temp = temp > nrhs ? temp : nrhs;
                temp = temp > n - 3 * m ? temp : n - 3 * m;
                if (4 * m + m * m + temp > maxwrk) maxwrk = 4 * m + m * m + temp;
            } else {
                nb = lapack_get_nb("GEBRD");
                maxwrk = 2 * m + (n + m) * nb;
                nb = lapack_get_nb("ORMBR");
                if (2 * m + nrhs * nb > maxwrk) maxwrk = 2 * m + nrhs * nb;
                if (2 * m + m * nb > maxwrk) maxwrk = 2 * m + m * nb;
                if (2 * m + m * nrhs > maxwrk) maxwrk = 2 * m + m * nrhs;
            }
            minwrk = 2 * m + n;
            if (2 * m + m * nrhs > minwrk) minwrk = 2 * m + m * nrhs;
        }
    } else {
        nlvl = 0;
    }
    if (minwrk > maxwrk) minwrk = maxwrk;
    work[0] = (c64)maxwrk;
    iwork[0] = liwork;
    rwork[0] = (f32)lrwork;

    if (*info == 0) {
        if (lwork < minwrk && !lquery) {
            *info = -12;
        }
    }

    if (*info != 0) {
        xerbla("CGELSD", -(*info));
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
    eps = slamch("P");
    sfmin = slamch("S");
    smlnum = sfmin / eps;
    bignum = ONE / smlnum;

    /* Scale A if max entry outside range [SMLNUM, BIGNUM] */
    anrm = clange("M", m, n, A, lda, rwork);
    iascl = 0;
    if (anrm > ZERO && anrm < smlnum) {
        clascl("G", 0, 0, anrm, smlnum, m, n, A, lda, &iinfo);
        iascl = 1;
    } else if (anrm > bignum) {
        clascl("G", 0, 0, anrm, bignum, m, n, A, lda, &iinfo);
        iascl = 2;
    } else if (anrm == ZERO) {
        claset("F", maxmn, nrhs, CZERO, CZERO, B, ldb);
        slaset("F", minmn, 1, ZERO, ZERO, S, 1);
        *rank = 0;
        goto cleanup;
    }

    /* Scale B if max entry outside range [SMLNUM, BIGNUM] */
    bnrm = clange("M", m, nrhs, B, ldb, rwork);
    ibscl = 0;
    if (bnrm > ZERO && bnrm < smlnum) {
        clascl("G", 0, 0, bnrm, smlnum, m, nrhs, B, ldb, &iinfo);
        ibscl = 1;
    } else if (bnrm > bignum) {
        clascl("G", 0, 0, bnrm, bignum, m, nrhs, B, ldb, &iinfo);
        ibscl = 2;
    }

    /* If M < N make sure B(M+1:N,:) = 0 */
    if (m < n) {
        claset("F", n - m, nrhs, CZERO, CZERO, &B[m], ldb);
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
            cgeqrf(m, n, A, lda, &work[itau], &work[nwork], lwork - nwork, &iinfo);

            /* Multiply B by transpose(Q) */
            cunmqr("L", "C", m, nrhs, n, A, lda, &work[itau], B, ldb,
                   &work[nwork], lwork - nwork, &iinfo);

            /* Zero out below R */
            if (n > 1) {
                claset("L", n - 1, n - 1, CZERO, CZERO, &A[1], lda);
            }
        }

        itauq = 0;
        itaup = itauq + n;
        nwork = itaup + n;
        ie = 0;
        nrwork = ie + n;

        /* Bidiagonalize R in A */
        cgebrd(mm, n, A, lda, S, &rwork[ie], &work[itauq], &work[itaup],
               &work[nwork], lwork - nwork, &iinfo);

        /* Multiply B by transpose of left bidiagonalizing vectors of R */
        cunmbr("Q", "L", "C", mm, nrhs, n, A, lda, &work[itauq],
               B, ldb, &work[nwork], lwork - nwork, &iinfo);

        /* Solve the bidiagonal least squares problem */
        clalsd("U", smlsiz, n, nrhs, S, &rwork[ie], B, ldb, rcond, rank,
               &work[nwork], &rwork[nrwork], iwork, info);
        if (*info != 0) {
            goto cleanup;
        }

        /* Multiply B by right bidiagonalizing vectors of R */
        cunmbr("P", "L", "N", n, nrhs, n, A, lda, &work[itaup],
               B, ldb, &work[nwork], lwork - nwork, &iinfo);

    } else if (n >= mnthr && lwork >= 4 * m + m * m +
               (m > 2 * m - 4 ? m : 2 * m - 4) +
               (nrhs > n - 3 * m ? nrhs : n - 3 * m)) {
        /* Path 2a - underdetermined, with many more columns than rows
         * and sufficient workspace for an efficient algorithm */
        ldwork = m;
        if (lwork >= 4 * m + m * lda +
            (m > 2 * m - 4 ? m : 2 * m - 4) +
            (nrhs > n - 3 * m ? nrhs : n - 3 * m)) {
            if (lwork >= m * lda + m + m * nrhs) {
                ldwork = lda;
            }
        }
        itau = 0;
        nwork = m;

        /* Compute A=L*Q */
        cgelqf(m, n, A, lda, &work[itau], &work[nwork], lwork - nwork, &iinfo);
        il = nwork;

        /* Copy L to WORK(IL), zeroing out above its diagonal */
        clacpy("L", m, m, A, lda, &work[il], ldwork);
        claset("U", m - 1, m - 1, CZERO, CZERO, &work[il + ldwork], ldwork);
        itauq = il + ldwork * m;
        itaup = itauq + m;
        nwork = itaup + m;
        ie = 0;
        nrwork = ie + m;

        /* Bidiagonalize L in WORK(IL) */
        cgebrd(m, m, &work[il], ldwork, S, &rwork[ie], &work[itauq],
               &work[itaup], &work[nwork], lwork - nwork, &iinfo);

        /* Multiply B by transpose of left bidiagonalizing vectors of L */
        cunmbr("Q", "L", "C", m, nrhs, m, &work[il], ldwork, &work[itauq],
               B, ldb, &work[nwork], lwork - nwork, &iinfo);

        /* Solve the bidiagonal least squares problem */
        clalsd("U", smlsiz, m, nrhs, S, &rwork[ie], B, ldb, rcond, rank,
               &work[nwork], &rwork[nrwork], iwork, info);
        if (*info != 0) {
            goto cleanup;
        }

        /* Multiply B by right bidiagonalizing vectors of L */
        cunmbr("P", "L", "N", m, nrhs, m, &work[il], ldwork, &work[itaup],
               B, ldb, &work[nwork], lwork - nwork, &iinfo);

        /* Zero out below first M rows of B */
        claset("F", n - m, nrhs, CZERO, CZERO, &B[m], ldb);
        nwork = itau + m;

        /* Multiply transpose(Q) by B */
        cunmlq("L", "C", n, nrhs, m, A, lda, &work[itau], B, ldb,
               &work[nwork], lwork - nwork, &iinfo);

    } else {
        /* Path 2 - remaining underdetermined cases */
        itauq = 0;
        itaup = itauq + m;
        nwork = itaup + m;
        ie = 0;
        nrwork = ie + m;

        /* Bidiagonalize A */
        cgebrd(m, n, A, lda, S, &rwork[ie], &work[itauq], &work[itaup],
               &work[nwork], lwork - nwork, &iinfo);

        /* Multiply B by transpose of left bidiagonalizing vectors */
        cunmbr("Q", "L", "C", m, nrhs, n, A, lda, &work[itauq],
               B, ldb, &work[nwork], lwork - nwork, &iinfo);

        /* Solve the bidiagonal least squares problem */
        clalsd("L", smlsiz, m, nrhs, S, &rwork[ie], B, ldb, rcond, rank,
               &work[nwork], &rwork[nrwork], iwork, info);
        if (*info != 0) {
            goto cleanup;
        }

        /* Multiply B by right bidiagonalizing vectors of A */
        cunmbr("P", "L", "N", n, nrhs, m, A, lda, &work[itaup],
               B, ldb, &work[nwork], lwork - nwork, &iinfo);
    }

    /* Undo scaling */
    if (iascl == 1) {
        clascl("G", 0, 0, anrm, smlnum, n, nrhs, B, ldb, &iinfo);
        slascl("G", 0, 0, smlnum, anrm, minmn, 1, S, minmn, &iinfo);
    } else if (iascl == 2) {
        clascl("G", 0, 0, anrm, bignum, n, nrhs, B, ldb, &iinfo);
        slascl("G", 0, 0, bignum, anrm, minmn, 1, S, minmn, &iinfo);
    }
    if (ibscl == 1) {
        clascl("G", 0, 0, smlnum, bnrm, n, nrhs, B, ldb, &iinfo);
    } else if (ibscl == 2) {
        clascl("G", 0, 0, bignum, bnrm, n, nrhs, B, ldb, &iinfo);
    }

cleanup:
    work[0] = (c64)maxwrk;
    iwork[0] = liwork;
    rwork[0] = (f32)lrwork;
}
