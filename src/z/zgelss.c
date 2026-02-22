/**
 * @file zgelss.c
 * @brief ZGELSS computes the minimum norm solution to a complex linear least
 *        squares problem using the SVD.
 */

#include "internal_build_defs.h"
#include "semicolon_lapack_complex_double.h"
#include "semicolon_lapack_double.h"
#include "../include/lapack_tuning.h"
#include <math.h>
#include <complex.h>
#include <cblas.h>

/**
 * ZGELSS computes the minimum norm solution to a complex linear least
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
 * @param[in,out] A      Complex*16 array, dimension (lda, n).
 *                       On entry, the M-by-N matrix A.
 *                       On exit, the first min(m,n) rows of A are overwritten
 *                       with its right singular vectors, stored rowwise.
 * @param[in]     lda    The leading dimension of A. lda >= max(1, m).
 * @param[in,out] B      Complex*16 array, dimension (ldb, nrhs).
 *                       On entry, the M-by-NRHS right hand side matrix B.
 *                       On exit, B is overwritten by the N-by-NRHS solution
 *                       matrix X.
 * @param[in]     ldb    The leading dimension of B. ldb >= max(1, max(m, n)).
 * @param[out]    S      Double precision array, dimension (min(m, n)).
 *                       The singular values of A in decreasing order.
 * @param[in]     rcond  RCOND is used to determine the effective rank of A.
 *                       Singular values S(i) <= RCOND*S(1) are treated as zero.
 *                       If RCOND < 0, machine precision is used instead.
 * @param[out]    rank   The effective rank of A.
 * @param[out]    work   Complex*16 array, dimension (max(1, lwork)).
 *                       On exit, if info = 0, work[0] returns the optimal lwork.
 * @param[in]     lwork  The dimension of work. lwork >= 1, and also:
 *                       lwork >= 2*min(m,n) + max(m,n,nrhs).
 *                       If lwork = -1, a workspace query is assumed.
 * @param[out]    rwork  Double precision array, dimension (5*min(m,n)).
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 *                         - > 0: the algorithm for computing the SVD failed to converge;
 *                           if info = i, i off-diagonal elements of an intermediate
 *                           bidiagonal form did not converge to zero.
 */
void zgelss(const INT m, const INT n, const INT nrhs,
            c128* restrict A, const INT lda,
            c128* restrict B, const INT ldb,
            f64* restrict S, const f64 rcond, INT* rank,
            c128* restrict work, const INT lwork,
            f64* restrict rwork,
            INT* info)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const c128 CZERO = CMPLX(0.0, 0.0);
    const c128 CONE = CMPLX(1.0, 0.0);

    INT lquery;
    INT bl, chunk, i, iascl, ibscl, ie, il, irwork, itau, itaup, itauq;
    INT iwork, ldwork, maxmn, maxwrk, minmn, minwrk, mm, mnthr = 0;
    INT lwork_zgeqrf, lwork_zunmqr, lwork_zgebrd, lwork_zunmbr, lwork_zungbr;
    INT lwork_zgelqf, lwork_zunmlq;
    f64 anrm, bignum, bnrm, eps, sfmin, smlnum, thr;
    c128 wkopt[1];
    INT iinfo;

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
                zgeqrf(m, n, A, lda, NULL, wkopt, -1, &iinfo);
                lwork_zgeqrf = (INT)creal(wkopt[0]);
                zunmqr("L", "C", m, nrhs, n, A, lda, NULL, B, ldb, wkopt, -1, &iinfo);
                lwork_zunmqr = (INT)creal(wkopt[0]);
                mm = n;
                {
                    INT t = n + lwork_zgeqrf;
                    if (t > maxwrk) maxwrk = t;
                }
                {
                    INT t = n + lwork_zunmqr;
                    if (t > maxwrk) maxwrk = t;
                }
            }
            if (m >= n) {
                /* Path 1 - overdetermined or exactly determined */
                zgebrd(mm, n, A, lda, S, NULL, NULL, NULL, wkopt, -1, &iinfo);
                lwork_zgebrd = (INT)creal(wkopt[0]);
                zunmbr("Q", "L", "C", mm, nrhs, n, A, lda, NULL, B, ldb, wkopt, -1, &iinfo);
                lwork_zunmbr = (INT)creal(wkopt[0]);
                zungbr("P", n, n, n, A, lda, NULL, wkopt, -1, &iinfo);
                lwork_zungbr = (INT)creal(wkopt[0]);
                {
                    INT t = 2 * n + lwork_zgebrd;
                    if (t > maxwrk) maxwrk = t;
                }
                {
                    INT t = 2 * n + lwork_zunmbr;
                    if (t > maxwrk) maxwrk = t;
                }
                {
                    INT t = 2 * n + lwork_zungbr;
                    if (t > maxwrk) maxwrk = t;
                }
                if (n * nrhs > maxwrk) maxwrk = n * nrhs;
                minwrk = 2 * n + (nrhs > m ? nrhs : m);
            }
            if (n > m) {
                minwrk = 2 * m + (nrhs > n ? nrhs : n);
                if (n >= mnthr) {
                    /* Path 2a - underdetermined, with many more columns than rows */
                    zgelqf(m, n, A, lda, NULL, wkopt, -1, &iinfo);
                    lwork_zgelqf = (INT)creal(wkopt[0]);
                    zgebrd(m, m, A, lda, S, NULL, NULL, NULL, wkopt, -1, &iinfo);
                    lwork_zgebrd = (INT)creal(wkopt[0]);
                    zunmbr("Q", "L", "C", m, nrhs, n, A, lda, NULL, B, ldb, wkopt, -1, &iinfo);
                    lwork_zunmbr = (INT)creal(wkopt[0]);
                    zungbr("P", m, m, m, A, lda, NULL, wkopt, -1, &iinfo);
                    lwork_zungbr = (INT)creal(wkopt[0]);
                    zunmlq("L", "C", n, nrhs, m, A, lda, NULL, B, ldb, wkopt, -1, &iinfo);
                    lwork_zunmlq = (INT)creal(wkopt[0]);
                    maxwrk = m + lwork_zgelqf;
                    {
                        INT t = 3 * m + m * m + lwork_zgebrd;
                        if (t > maxwrk) maxwrk = t;
                    }
                    {
                        INT t = 3 * m + m * m + lwork_zunmbr;
                        if (t > maxwrk) maxwrk = t;
                    }
                    {
                        INT t = 3 * m + m * m + lwork_zungbr;
                        if (t > maxwrk) maxwrk = t;
                    }
                    if (nrhs > 1) {
                        INT t = m * m + m + m * nrhs;
                        if (t > maxwrk) maxwrk = t;
                    } else {
                        INT t = m * m + 2 * m;
                        if (t > maxwrk) maxwrk = t;
                    }
                    {
                        INT t = m + lwork_zunmlq;
                        if (t > maxwrk) maxwrk = t;
                    }
                } else {
                    /* Path 2 - underdetermined */
                    zgebrd(m, n, A, lda, S, NULL, NULL, NULL, wkopt, -1, &iinfo);
                    lwork_zgebrd = (INT)creal(wkopt[0]);
                    zunmbr("Q", "L", "C", m, nrhs, m, A, lda, NULL, B, ldb, wkopt, -1, &iinfo);
                    lwork_zunmbr = (INT)creal(wkopt[0]);
                    zungbr("P", m, n, m, A, lda, NULL, wkopt, -1, &iinfo);
                    lwork_zungbr = (INT)creal(wkopt[0]);
                    maxwrk = 2 * m + lwork_zgebrd;
                    {
                        INT t = 2 * m + lwork_zunmbr;
                        if (t > maxwrk) maxwrk = t;
                    }
                    {
                        INT t = 2 * m + lwork_zungbr;
                        if (t > maxwrk) maxwrk = t;
                    }
                    if (n * nrhs > maxwrk) maxwrk = n * nrhs;
                }
            }
            if (maxwrk < minwrk) maxwrk = minwrk;
        }
        work[0] = CMPLX((f64)maxwrk, 0.0);

        if (lwork < minwrk && !lquery) {
            *info = -12;
        }
    }

    if (*info != 0) {
        xerbla("ZGELSS", -(*info));
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
    anrm = zlange("M", m, n, A, lda, rwork);
    iascl = 0;
    if (anrm > ZERO && anrm < smlnum) {
        zlascl("G", 0, 0, anrm, smlnum, m, n, A, lda, &iinfo);
        iascl = 1;
    } else if (anrm > bignum) {
        zlascl("G", 0, 0, anrm, bignum, m, n, A, lda, &iinfo);
        iascl = 2;
    } else if (anrm == ZERO) {
        zlaset("F", maxmn, nrhs, CZERO, CZERO, B, ldb);
        dlaset("F", minmn, 1, ZERO, ZERO, S, minmn);
        *rank = 0;
        goto cleanup;
    }

    /* Scale B if max element outside range [SMLNUM, BIGNUM] */
    bnrm = zlange("M", m, nrhs, B, ldb, rwork);
    ibscl = 0;
    if (bnrm > ZERO && bnrm < smlnum) {
        zlascl("G", 0, 0, bnrm, smlnum, m, nrhs, B, ldb, &iinfo);
        ibscl = 1;
    } else if (bnrm > bignum) {
        zlascl("G", 0, 0, bnrm, bignum, m, nrhs, B, ldb, &iinfo);
        ibscl = 2;
    }

    /* Overdetermined case */
    if (m >= n) {
        /* Path 1 - overdetermined or exactly determined */
        mm = m;
        if (m >= mnthr) {
            /* Path 1a - overdetermined, with many more rows than columns */
            mm = n;
            itau = 0;
            iwork = itau + n;

            /* Compute A=Q*R */
            zgeqrf(m, n, A, lda, &work[itau], &work[iwork], lwork - iwork, &iinfo);

            /* Multiply B by transpose(Q) */
            zunmqr("L", "C", m, nrhs, n, A, lda, &work[itau], B, ldb,
                   &work[iwork], lwork - iwork, &iinfo);

            /* Zero out below R */
            if (n > 1) {
                zlaset("L", n - 1, n - 1, CZERO, CZERO, &A[1], lda);
            }
        }

        ie = 0;
        itauq = 0;
        itaup = itauq + n;
        iwork = itaup + n;

        /* Bidiagonalize R in A */
        zgebrd(mm, n, A, lda, S, &rwork[ie], &work[itauq], &work[itaup],
               &work[iwork], lwork - iwork, &iinfo);

        /* Multiply B by transpose of left bidiagonalizing vectors of R */
        zunmbr("Q", "L", "C", mm, nrhs, n, A, lda, &work[itauq],
               B, ldb, &work[iwork], lwork - iwork, &iinfo);

        /* Generate right bidiagonalizing vectors of R in A */
        zungbr("P", n, n, n, A, lda, &work[itaup], &work[iwork], lwork - iwork, &iinfo);
        irwork = ie + n;

        /* Perform bidiagonal QR iteration
         *   multiply B by transpose of left singular vectors
         *   compute right singular vectors in A */
        zbdsqr("U", n, n, 0, nrhs, S, &rwork[ie], A, lda, NULL, 1, B, ldb,
               &rwork[irwork], info);
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
                zdrscl(nrhs, S[i], &B[i], ldb);
                (*rank)++;
            } else {
                zlaset("F", 1, nrhs, CZERO, CZERO, &B[i], ldb);
            }
        }

        /* Multiply B by right singular vectors */
        if (lwork >= ldb * nrhs && nrhs > 1) {
            cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans,
                        n, nrhs, n, &CONE, A, lda, B, ldb, &CZERO, work, ldb);
            zlacpy("G", n, nrhs, work, ldb, B, ldb);
        } else if (nrhs > 1) {
            chunk = lwork / n;
            for (i = 0; i < nrhs; i += chunk) {
                bl = nrhs - i < chunk ? nrhs - i : chunk;
                cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans,
                            n, bl, n, &CONE, A, lda, &B[i * ldb], ldb, &CZERO, work, n);
                zlacpy("G", n, bl, work, n, &B[i * ldb], ldb);
            }
        } else if (nrhs == 1) {
            cblas_zgemv(CblasColMajor, CblasConjTrans, n, n, &CONE, A, lda, B, 1, &CZERO, work, 1);
            cblas_zcopy(n, work, 1, B, 1);
        }

    } else if (n >= mnthr && lwork >= 3 * m + m * m +
               (m > nrhs ? m : nrhs) + (n - 2 * m > 0 ? n - 2 * m : 0)) {
        /* Path 2a - underdetermined, with many more columns than rows
         * and sufficient workspace for an efficient algorithm */
        ldwork = m;
        if (lwork >= 3 * m + m * lda +
            (m > nrhs ? m : nrhs) + (n - 2 * m > 0 ? n - 2 * m : 0))
            ldwork = lda;
        itau = 0;
        iwork = m;

        /* Compute A=L*Q */
        zgelqf(m, n, A, lda, &work[itau], &work[iwork], lwork - iwork, &iinfo);
        il = iwork;

        /* Copy L to WORK(IL), zeroing out above it */
        zlacpy("L", m, m, A, lda, &work[il], ldwork);
        zlaset("U", m - 1, m - 1, CZERO, CZERO, &work[il + ldwork], ldwork);
        ie = 0;
        itauq = il + ldwork * m;
        itaup = itauq + m;
        iwork = itaup + m;

        /* Bidiagonalize L in WORK(IL) */
        zgebrd(m, m, &work[il], ldwork, S, &rwork[ie], &work[itauq],
               &work[itaup], &work[iwork], lwork - iwork, &iinfo);

        /* Multiply B by transpose of left bidiagonalizing vectors of L */
        zunmbr("Q", "L", "C", m, nrhs, m, &work[il], ldwork, &work[itauq],
               B, ldb, &work[iwork], lwork - iwork, &iinfo);

        /* Generate right bidiagonalizing vectors of R in WORK(IL) */
        zungbr("P", m, m, m, &work[il], ldwork, &work[itaup],
               &work[iwork], lwork - iwork, &iinfo);
        irwork = ie + m;

        /* Perform bidiagonal QR iteration,
         *   computing right singular vectors of L in WORK(IL) and
         *   multiplying B by transpose of left singular vectors */
        zbdsqr("U", m, m, 0, nrhs, S, &rwork[ie], &work[il], ldwork,
               A, lda, B, ldb, &rwork[irwork], info);
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
                zdrscl(nrhs, S[i], &B[i], ldb);
                (*rank)++;
            } else {
                zlaset("F", 1, nrhs, CZERO, CZERO, &B[i], ldb);
            }
        }
        iwork = il + m * ldwork;

        /* Multiply B by right singular vectors of L in WORK(IL) */
        if (lwork >= ldb * nrhs + iwork && nrhs > 1) {
            cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans,
                        m, nrhs, m, &CONE, &work[il], ldwork, B, ldb, &CZERO, &work[iwork], ldb);
            zlacpy("G", m, nrhs, &work[iwork], ldb, B, ldb);
        } else if (nrhs > 1) {
            chunk = (lwork - iwork) / m;
            for (i = 0; i < nrhs; i += chunk) {
                bl = nrhs - i < chunk ? nrhs - i : chunk;
                cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans,
                            m, bl, m, &CONE, &work[il], ldwork, &B[i * ldb], ldb, &CZERO, &work[iwork], m);
                zlacpy("G", m, bl, &work[iwork], m, &B[i * ldb], ldb);
            }
        } else if (nrhs == 1) {
            cblas_zgemv(CblasColMajor, CblasConjTrans, m, m, &CONE, &work[il], ldwork, B, 1, &CZERO, &work[iwork], 1);
            cblas_zcopy(m, &work[iwork], 1, B, 1);
        }

        /* Zero out below first M rows of B */
        zlaset("F", n - m, nrhs, CZERO, CZERO, &B[m], ldb);
        iwork = itau + m;

        /* Multiply transpose(Q) by B */
        zunmlq("L", "C", n, nrhs, m, A, lda, &work[itau], B, ldb,
               &work[iwork], lwork - iwork, &iinfo);

    } else {
        /* Path 2 - remaining underdetermined cases */
        ie = 0;
        itauq = 0;
        itaup = itauq + m;
        iwork = itaup + m;

        /* Bidiagonalize A */
        zgebrd(m, n, A, lda, S, &rwork[ie], &work[itauq], &work[itaup],
               &work[iwork], lwork - iwork, &iinfo);

        /* Multiply B by transpose of left bidiagonalizing vectors */
        zunmbr("Q", "L", "C", m, nrhs, n, A, lda, &work[itauq],
               B, ldb, &work[iwork], lwork - iwork, &iinfo);

        /* Generate right bidiagonalizing vectors in A */
        zungbr("P", m, n, m, A, lda, &work[itaup], &work[iwork], lwork - iwork, &iinfo);
        irwork = ie + m;

        /* Perform bidiagonal QR iteration,
         *   computing right singular vectors of A in A and
         *   multiplying B by transpose of left singular vectors */
        zbdsqr("L", m, n, 0, nrhs, S, &rwork[ie], A, lda, NULL, 1, B, ldb,
               &rwork[irwork], info);
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
                zdrscl(nrhs, S[i], &B[i], ldb);
                (*rank)++;
            } else {
                zlaset("F", 1, nrhs, CZERO, CZERO, &B[i], ldb);
            }
        }

        /* Multiply B by right singular vectors of A */
        if (lwork >= ldb * nrhs && nrhs > 1) {
            cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans,
                        n, nrhs, m, &CONE, A, lda, B, ldb, &CZERO, work, ldb);
            zlacpy("G", n, nrhs, work, ldb, B, ldb);
        } else if (nrhs > 1) {
            chunk = lwork / n;
            for (i = 0; i < nrhs; i += chunk) {
                bl = nrhs - i < chunk ? nrhs - i : chunk;
                cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans,
                            n, bl, m, &CONE, A, lda, &B[i * ldb], ldb, &CZERO, work, n);
                zlacpy("F", n, bl, work, n, &B[i * ldb], ldb);
            }
        } else if (nrhs == 1) {
            cblas_zgemv(CblasColMajor, CblasConjTrans, m, n, &CONE, A, lda, B, 1, &CZERO, work, 1);
            cblas_zcopy(n, work, 1, B, 1);
        }
    }

    /* Undo scaling */
    if (iascl == 1) {
        zlascl("G", 0, 0, anrm, smlnum, n, nrhs, B, ldb, &iinfo);
        dlascl("G", 0, 0, smlnum, anrm, minmn, 1, S, minmn, &iinfo);
    } else if (iascl == 2) {
        zlascl("G", 0, 0, anrm, bignum, n, nrhs, B, ldb, &iinfo);
        dlascl("G", 0, 0, bignum, anrm, minmn, 1, S, minmn, &iinfo);
    }
    if (ibscl == 1) {
        zlascl("G", 0, 0, smlnum, bnrm, n, nrhs, B, ldb, &iinfo);
    } else if (ibscl == 2) {
        zlascl("G", 0, 0, bignum, bnrm, n, nrhs, B, ldb, &iinfo);
    }

cleanup:
    work[0] = CMPLX((f64)maxwrk, 0.0);
}
