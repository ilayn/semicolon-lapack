/**
 * @file zgesdd.c
 * @brief ZGESDD computes the singular value decomposition (SVD) of a complex
 *        M-by-N matrix using a divide-and-conquer algorithm.
 */

#include "semicolon_lapack_complex_double.h"
#include <complex.h>
#include <math.h>
#include <cblas.h>

static const f64 ZERO = 0.0;
static const f64 ONE = 1.0;
static const c128 CZERO = CMPLX(0.0, 0.0);
static const c128 CONE = CMPLX(1.0, 0.0);

/**
 * ZGESDD computes the singular value decomposition (SVD) of a complex
 * M-by-N matrix A, optionally computing the left and/or right singular
 * vectors, by using divide-and-conquer method. The SVD is written
 *
 *      A = U * SIGMA * conjugate-transpose(V)
 *
 * where SIGMA is an M-by-N matrix which is zero except for its
 * min(m,n) diagonal elements, U is an M-by-M unitary matrix, and
 * V is an N-by-N unitary matrix.  The diagonal elements of SIGMA
 * are the singular values of A; they are real and non-negative, and
 * are returned in descending order.  The first min(m,n) columns of
 * U and V are the left and right singular vectors of A.
 *
 * Note that the routine returns VT = V**H, not V.
 *
 * @param[in]     jobz    = 'A': all M columns of U and all N rows of V**H returned.
 *                         = 'S': first min(M,N) columns of U and rows of V**H returned.
 *                         = 'O': If M >= N, first N columns of U overwritten on A and
 *                                all rows of V**H returned; otherwise all columns of U
 *                                returned and first M rows of V**H overwritten on A.
 *                         = 'N': no columns of U or rows of V**H computed.
 * @param[in]     m       Number of rows of A. m >= 0.
 * @param[in]     n       Number of columns of A. n >= 0.
 * @param[in,out] A       Complex array (lda, n). On entry, M-by-N matrix A.
 *                        On exit, contents depend on jobz.
 * @param[in]     lda     Leading dimension of A. lda >= max(1, m).
 * @param[out]    S       Real array of dimension min(m,n). Singular values in descending order.
 * @param[out]    U       Complex array (ldu, ucol). Left singular vectors if requested.
 * @param[in]     ldu     Leading dimension of U.
 * @param[out]    VT      Complex array (ldvt, n). Right singular vectors conjugate-transposed.
 * @param[in]     ldvt    Leading dimension of VT.
 * @param[out]    work    Complex array of dimension max(1, lwork).
 * @param[in]     lwork   Dimension of work. If lwork=-1, workspace query.
 * @param[out]    rwork   Real array.
 * @param[out]    iwork   Integer array of dimension 8*min(m,n).
 * @param[out]    info    = 0: success. < 0: illegal argument. > 0: DC did not converge.
 */
void zgesdd(const char* jobz, const int m, const int n,
            c128* const restrict A, const int lda,
            f64* const restrict S,
            c128* const restrict U, const int ldu,
            c128* const restrict VT, const int ldvt,
            c128* const restrict work, const int lwork,
            f64* const restrict rwork, int* const restrict iwork, int* info)
{
    int lquery, wntqa, wntqas, wntqn, wntqo, wntqs;
    int blk, chunk, i, ie, ierr, il, ir, iru, irvt;
    int iscl, itau, itaup, itauq, iu, ivt, ldwkvt;
    int ldwrkl, ldwrkr, ldwrku, maxwrk, minmn, minwrk;
    int mnthr1, mnthr2, nrwork, nwork, wrkbl;
    int lwork_zgebrd_mn, lwork_zgebrd_mm, lwork_zgebrd_nn;
    int lwork_zgelqf_mn, lwork_zgeqrf_mn;
    int lwork_zungbr_p_mn, lwork_zungbr_p_nn;
    int lwork_zungbr_q_mn, lwork_zungbr_q_mm;
    int lwork_zunglq_mn, lwork_zunglq_nn;
    int lwork_zungqr_mm, lwork_zungqr_mn;
    int lwork_zunmbr_prc_mm, lwork_zunmbr_qln_mm;
    int lwork_zunmbr_prc_mn, lwork_zunmbr_qln_mn;
    int lwork_zunmbr_prc_nn, lwork_zunmbr_qln_nn;
    f64 anrm, bignum, eps, smlnum;
    int idum[1];
    f64 dum[1];
    c128 cdum[1];

    *info = 0;
    minmn = (m < n) ? m : n;
    mnthr1 = (int)(minmn * 17.0 / 9.0);
    mnthr2 = (int)(minmn * 5.0 / 3.0);
    wntqa = (jobz[0] == 'A' || jobz[0] == 'a');
    wntqs = (jobz[0] == 'S' || jobz[0] == 's');
    wntqas = wntqa || wntqs;
    wntqo = (jobz[0] == 'O' || jobz[0] == 'o');
    wntqn = (jobz[0] == 'N' || jobz[0] == 'n');
    lquery = (lwork == -1);
    minwrk = 1;
    maxwrk = 1;

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

    if (*info == 0) {
        minwrk = 1;
        maxwrk = 1;
        if (m >= n && minmn > 0) {

            zgebrd(m, n, cdum, m, dum, dum, cdum, cdum, cdum, -1, &ierr);
            lwork_zgebrd_mn = (int)creal(cdum[0]);

            zgebrd(n, n, cdum, n, dum, dum, cdum, cdum, cdum, -1, &ierr);
            lwork_zgebrd_nn = (int)creal(cdum[0]);

            zgeqrf(m, n, cdum, m, cdum, cdum, -1, &ierr);
            lwork_zgeqrf_mn = (int)creal(cdum[0]);

            zungbr("P", n, n, n, cdum, n, cdum, cdum, -1, &ierr);
            lwork_zungbr_p_nn = (int)creal(cdum[0]);

            zungbr("Q", m, m, n, cdum, m, cdum, cdum, -1, &ierr);
            lwork_zungbr_q_mm = (int)creal(cdum[0]);

            zungbr("Q", m, n, n, cdum, m, cdum, cdum, -1, &ierr);
            lwork_zungbr_q_mn = (int)creal(cdum[0]);

            zungqr(m, m, n, cdum, m, cdum, cdum, -1, &ierr);
            lwork_zungqr_mm = (int)creal(cdum[0]);

            zungqr(m, n, n, cdum, m, cdum, cdum, -1, &ierr);
            lwork_zungqr_mn = (int)creal(cdum[0]);

            zunmbr("P", "R", "C", n, n, n, cdum, n, cdum, cdum, n, cdum, -1, &ierr);
            lwork_zunmbr_prc_nn = (int)creal(cdum[0]);

            zunmbr("Q", "L", "N", m, m, n, cdum, m, cdum, cdum, m, cdum, -1, &ierr);
            lwork_zunmbr_qln_mm = (int)creal(cdum[0]);

            zunmbr("Q", "L", "N", m, n, n, cdum, m, cdum, cdum, m, cdum, -1, &ierr);
            lwork_zunmbr_qln_mn = (int)creal(cdum[0]);

            zunmbr("Q", "L", "N", n, n, n, cdum, n, cdum, cdum, n, cdum, -1, &ierr);
            lwork_zunmbr_qln_nn = (int)creal(cdum[0]);

            if (m >= mnthr1) {
                if (wntqn) {

                    /* Path 1 (M >> N, JOBZ='N') */
                    maxwrk = n + lwork_zgeqrf_mn;
                    maxwrk = (maxwrk > 2 * n + lwork_zgebrd_nn) ? maxwrk : 2 * n + lwork_zgebrd_nn;
                    minwrk = 3 * n;
                } else if (wntqo) {

                    /* Path 2 (M >> N, JOBZ='O') */
                    wrkbl = n + lwork_zgeqrf_mn;
                    wrkbl = (wrkbl > n + lwork_zungqr_mn) ? wrkbl : n + lwork_zungqr_mn;
                    wrkbl = (wrkbl > 2 * n + lwork_zgebrd_nn) ? wrkbl : 2 * n + lwork_zgebrd_nn;
                    wrkbl = (wrkbl > 2 * n + lwork_zunmbr_qln_nn) ? wrkbl : 2 * n + lwork_zunmbr_qln_nn;
                    wrkbl = (wrkbl > 2 * n + lwork_zunmbr_prc_nn) ? wrkbl : 2 * n + lwork_zunmbr_prc_nn;
                    maxwrk = m * n + n * n + wrkbl;
                    minwrk = 2 * n * n + 3 * n;
                } else if (wntqs) {

                    /* Path 3 (M >> N, JOBZ='S') */
                    wrkbl = n + lwork_zgeqrf_mn;
                    wrkbl = (wrkbl > n + lwork_zungqr_mn) ? wrkbl : n + lwork_zungqr_mn;
                    wrkbl = (wrkbl > 2 * n + lwork_zgebrd_nn) ? wrkbl : 2 * n + lwork_zgebrd_nn;
                    wrkbl = (wrkbl > 2 * n + lwork_zunmbr_qln_nn) ? wrkbl : 2 * n + lwork_zunmbr_qln_nn;
                    wrkbl = (wrkbl > 2 * n + lwork_zunmbr_prc_nn) ? wrkbl : 2 * n + lwork_zunmbr_prc_nn;
                    maxwrk = n * n + wrkbl;
                    minwrk = n * n + 3 * n;
                } else if (wntqa) {

                    /* Path 4 (M >> N, JOBZ='A') */
                    wrkbl = n + lwork_zgeqrf_mn;
                    wrkbl = (wrkbl > n + lwork_zungqr_mm) ? wrkbl : n + lwork_zungqr_mm;
                    wrkbl = (wrkbl > 2 * n + lwork_zgebrd_nn) ? wrkbl : 2 * n + lwork_zgebrd_nn;
                    wrkbl = (wrkbl > 2 * n + lwork_zunmbr_qln_nn) ? wrkbl : 2 * n + lwork_zunmbr_qln_nn;
                    wrkbl = (wrkbl > 2 * n + lwork_zunmbr_prc_nn) ? wrkbl : 2 * n + lwork_zunmbr_prc_nn;
                    maxwrk = n * n + wrkbl;
                    minwrk = n * n + ((3 * n > n + m) ? 3 * n : n + m);
                }
            } else if (m >= mnthr2) {

                /* Path 5 (M >> N, but not as much as MNTHR1) */
                maxwrk = 2 * n + lwork_zgebrd_mn;
                minwrk = 2 * n + m;
                if (wntqo) {
                    /* Path 5o (M >> N, JOBZ='O') */
                    maxwrk = (maxwrk > 2 * n + lwork_zungbr_p_nn) ? maxwrk : 2 * n + lwork_zungbr_p_nn;
                    maxwrk = (maxwrk > 2 * n + lwork_zungbr_q_mn) ? maxwrk : 2 * n + lwork_zungbr_q_mn;
                    maxwrk = maxwrk + m * n;
                    minwrk = minwrk + n * n;
                } else if (wntqs) {
                    /* Path 5s (M >> N, JOBZ='S') */
                    maxwrk = (maxwrk > 2 * n + lwork_zungbr_p_nn) ? maxwrk : 2 * n + lwork_zungbr_p_nn;
                    maxwrk = (maxwrk > 2 * n + lwork_zungbr_q_mn) ? maxwrk : 2 * n + lwork_zungbr_q_mn;
                } else if (wntqa) {
                    /* Path 5a (M >> N, JOBZ='A') */
                    maxwrk = (maxwrk > 2 * n + lwork_zungbr_p_nn) ? maxwrk : 2 * n + lwork_zungbr_p_nn;
                    maxwrk = (maxwrk > 2 * n + lwork_zungbr_q_mm) ? maxwrk : 2 * n + lwork_zungbr_q_mm;
                }
            } else {

                /* Path 6 (M >= N, but not much larger) */
                maxwrk = 2 * n + lwork_zgebrd_mn;
                minwrk = 2 * n + m;
                if (wntqo) {
                    /* Path 6o (M >= N, JOBZ='O') */
                    maxwrk = (maxwrk > 2 * n + lwork_zunmbr_prc_nn) ? maxwrk : 2 * n + lwork_zunmbr_prc_nn;
                    maxwrk = (maxwrk > 2 * n + lwork_zunmbr_qln_mn) ? maxwrk : 2 * n + lwork_zunmbr_qln_mn;
                    maxwrk = maxwrk + m * n;
                    minwrk = minwrk + n * n;
                } else if (wntqs) {
                    /* Path 6s (M >= N, JOBZ='S') */
                    maxwrk = (maxwrk > 2 * n + lwork_zunmbr_qln_mn) ? maxwrk : 2 * n + lwork_zunmbr_qln_mn;
                    maxwrk = (maxwrk > 2 * n + lwork_zunmbr_prc_nn) ? maxwrk : 2 * n + lwork_zunmbr_prc_nn;
                } else if (wntqa) {
                    /* Path 6a (M >= N, JOBZ='A') */
                    maxwrk = (maxwrk > 2 * n + lwork_zunmbr_qln_mm) ? maxwrk : 2 * n + lwork_zunmbr_qln_mm;
                    maxwrk = (maxwrk > 2 * n + lwork_zunmbr_prc_nn) ? maxwrk : 2 * n + lwork_zunmbr_prc_nn;
                }
            }
        } else if (minmn > 0) {

            zgebrd(m, n, cdum, m, dum, dum, cdum, cdum, cdum, -1, &ierr);
            lwork_zgebrd_mn = (int)creal(cdum[0]);

            zgebrd(m, m, cdum, m, dum, dum, cdum, cdum, cdum, -1, &ierr);
            lwork_zgebrd_mm = (int)creal(cdum[0]);

            zgelqf(m, n, cdum, m, cdum, cdum, -1, &ierr);
            lwork_zgelqf_mn = (int)creal(cdum[0]);

            zungbr("P", m, n, m, cdum, m, cdum, cdum, -1, &ierr);
            lwork_zungbr_p_mn = (int)creal(cdum[0]);

            zungbr("P", n, n, m, cdum, n, cdum, cdum, -1, &ierr);
            lwork_zungbr_p_nn = (int)creal(cdum[0]);

            zungbr("Q", m, m, n, cdum, m, cdum, cdum, -1, &ierr);
            lwork_zungbr_q_mm = (int)creal(cdum[0]);

            zunglq(m, n, m, cdum, m, cdum, cdum, -1, &ierr);
            lwork_zunglq_mn = (int)creal(cdum[0]);

            zunglq(n, n, m, cdum, n, cdum, cdum, -1, &ierr);
            lwork_zunglq_nn = (int)creal(cdum[0]);

            zunmbr("P", "R", "C", m, m, m, cdum, m, cdum, cdum, m, cdum, -1, &ierr);
            lwork_zunmbr_prc_mm = (int)creal(cdum[0]);

            zunmbr("P", "R", "C", m, n, m, cdum, m, cdum, cdum, m, cdum, -1, &ierr);
            lwork_zunmbr_prc_mn = (int)creal(cdum[0]);

            zunmbr("P", "R", "C", n, n, m, cdum, n, cdum, cdum, n, cdum, -1, &ierr);
            lwork_zunmbr_prc_nn = (int)creal(cdum[0]);

            zunmbr("Q", "L", "N", m, m, m, cdum, m, cdum, cdum, m, cdum, -1, &ierr);
            lwork_zunmbr_qln_mm = (int)creal(cdum[0]);

            if (n >= mnthr1) {
                if (wntqn) {

                    /* Path 1t (N >> M, JOBZ='N') */
                    maxwrk = m + lwork_zgelqf_mn;
                    maxwrk = (maxwrk > 2 * m + lwork_zgebrd_mm) ? maxwrk : 2 * m + lwork_zgebrd_mm;
                    minwrk = 3 * m;
                } else if (wntqo) {

                    /* Path 2t (N >> M, JOBZ='O') */
                    wrkbl = m + lwork_zgelqf_mn;
                    wrkbl = (wrkbl > m + lwork_zunglq_mn) ? wrkbl : m + lwork_zunglq_mn;
                    wrkbl = (wrkbl > 2 * m + lwork_zgebrd_mm) ? wrkbl : 2 * m + lwork_zgebrd_mm;
                    wrkbl = (wrkbl > 2 * m + lwork_zunmbr_qln_mm) ? wrkbl : 2 * m + lwork_zunmbr_qln_mm;
                    wrkbl = (wrkbl > 2 * m + lwork_zunmbr_prc_mm) ? wrkbl : 2 * m + lwork_zunmbr_prc_mm;
                    maxwrk = m * n + m * m + wrkbl;
                    minwrk = 2 * m * m + 3 * m;
                } else if (wntqs) {

                    /* Path 3t (N >> M, JOBZ='S') */
                    wrkbl = m + lwork_zgelqf_mn;
                    wrkbl = (wrkbl > m + lwork_zunglq_mn) ? wrkbl : m + lwork_zunglq_mn;
                    wrkbl = (wrkbl > 2 * m + lwork_zgebrd_mm) ? wrkbl : 2 * m + lwork_zgebrd_mm;
                    wrkbl = (wrkbl > 2 * m + lwork_zunmbr_qln_mm) ? wrkbl : 2 * m + lwork_zunmbr_qln_mm;
                    wrkbl = (wrkbl > 2 * m + lwork_zunmbr_prc_mm) ? wrkbl : 2 * m + lwork_zunmbr_prc_mm;
                    maxwrk = m * m + wrkbl;
                    minwrk = m * m + 3 * m;
                } else if (wntqa) {

                    /* Path 4t (N >> M, JOBZ='A') */
                    wrkbl = m + lwork_zgelqf_mn;
                    wrkbl = (wrkbl > m + lwork_zunglq_nn) ? wrkbl : m + lwork_zunglq_nn;
                    wrkbl = (wrkbl > 2 * m + lwork_zgebrd_mm) ? wrkbl : 2 * m + lwork_zgebrd_mm;
                    wrkbl = (wrkbl > 2 * m + lwork_zunmbr_qln_mm) ? wrkbl : 2 * m + lwork_zunmbr_qln_mm;
                    wrkbl = (wrkbl > 2 * m + lwork_zunmbr_prc_mm) ? wrkbl : 2 * m + lwork_zunmbr_prc_mm;
                    maxwrk = m * m + wrkbl;
                    minwrk = m * m + ((3 * m > m + n) ? 3 * m : m + n);
                }
            } else if (n >= mnthr2) {

                /* Path 5t (N >> M, but not as much as MNTHR1) */
                maxwrk = 2 * m + lwork_zgebrd_mn;
                minwrk = 2 * m + n;
                if (wntqo) {
                    /* Path 5to (N >> M, JOBZ='O') */
                    maxwrk = (maxwrk > 2 * m + lwork_zungbr_q_mm) ? maxwrk : 2 * m + lwork_zungbr_q_mm;
                    maxwrk = (maxwrk > 2 * m + lwork_zungbr_p_mn) ? maxwrk : 2 * m + lwork_zungbr_p_mn;
                    maxwrk = maxwrk + m * n;
                    minwrk = minwrk + m * m;
                } else if (wntqs) {
                    /* Path 5ts (N >> M, JOBZ='S') */
                    maxwrk = (maxwrk > 2 * m + lwork_zungbr_q_mm) ? maxwrk : 2 * m + lwork_zungbr_q_mm;
                    maxwrk = (maxwrk > 2 * m + lwork_zungbr_p_mn) ? maxwrk : 2 * m + lwork_zungbr_p_mn;
                } else if (wntqa) {
                    /* Path 5ta (N >> M, JOBZ='A') */
                    maxwrk = (maxwrk > 2 * m + lwork_zungbr_q_mm) ? maxwrk : 2 * m + lwork_zungbr_q_mm;
                    maxwrk = (maxwrk > 2 * m + lwork_zungbr_p_nn) ? maxwrk : 2 * m + lwork_zungbr_p_nn;
                }
            } else {

                /* Path 6t (N > M, but not much larger) */
                maxwrk = 2 * m + lwork_zgebrd_mn;
                minwrk = 2 * m + n;
                if (wntqo) {
                    /* Path 6to (N > M, JOBZ='O') */
                    maxwrk = (maxwrk > 2 * m + lwork_zunmbr_qln_mm) ? maxwrk : 2 * m + lwork_zunmbr_qln_mm;
                    maxwrk = (maxwrk > 2 * m + lwork_zunmbr_prc_mn) ? maxwrk : 2 * m + lwork_zunmbr_prc_mn;
                    maxwrk = maxwrk + m * n;
                    minwrk = minwrk + m * m;
                } else if (wntqs) {
                    /* Path 6ts (N > M, JOBZ='S') */
                    maxwrk = (maxwrk > 2 * m + lwork_zunmbr_qln_mm) ? maxwrk : 2 * m + lwork_zunmbr_qln_mm;
                    maxwrk = (maxwrk > 2 * m + lwork_zunmbr_prc_mn) ? maxwrk : 2 * m + lwork_zunmbr_prc_mn;
                } else if (wntqa) {
                    /* Path 6ta (N > M, JOBZ='A') */
                    maxwrk = (maxwrk > 2 * m + lwork_zunmbr_qln_mm) ? maxwrk : 2 * m + lwork_zunmbr_qln_mm;
                    maxwrk = (maxwrk > 2 * m + lwork_zunmbr_prc_nn) ? maxwrk : 2 * m + lwork_zunmbr_prc_nn;
                }
            }
        }
        maxwrk = (maxwrk > minwrk) ? maxwrk : minwrk;
    }
    if (*info == 0) {
        work[0] = CMPLX((f64)maxwrk, 0.0);
        if (lwork < minwrk && !lquery) {
            *info = -12;
        }
    }

    if (*info != 0) {
        xerbla("ZGESDD", -(*info));
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

    /* Scale A if max element outside range [SMLNUM,BIGNUM] */
    anrm = zlange("M", m, n, A, lda, dum);
    if (isnan(anrm)) {
        *info = -4;
        return;
    }
    iscl = 0;
    if (anrm > ZERO && anrm < smlnum) {
        iscl = 1;
        zlascl("G", 0, 0, anrm, smlnum, m, n, A, lda, &ierr);
    } else if (anrm > bignum) {
        iscl = 1;
        zlascl("G", 0, 0, anrm, bignum, m, n, A, lda, &ierr);
    }

    if (m >= n) {

        if (m >= mnthr1) {

            if (wntqn) {

                /* Path 1 (M >> N, JOBZ='N') */
                itau = 0;
                nwork = itau + n;

                zgeqrf(m, n, A, lda, &work[itau], &work[nwork],
                       lwork - nwork, &ierr);

                zlaset("L", n - 1, n - 1, CZERO, CZERO, &A[1], lda);
                ie = 0;
                itauq = 0;
                itaup = itauq + n;
                nwork = itaup + n;

                zgebrd(n, n, A, lda, S, &rwork[ie], &work[itauq],
                       &work[itaup], &work[nwork], lwork - nwork, &ierr);
                nrwork = ie + n;

                dbdsdc("U", "N", n, S, &rwork[ie], dum, 1, dum, 1,
                       dum, idum, &rwork[nrwork], iwork, info);

            } else if (wntqo) {

                /* Path 2 (M >> N, JOBZ='O') */
                iu = 0;

                ldwrku = n;
                ir = iu + ldwrku * n;
                if (lwork >= m * n + n * n + 3 * n) {
                    ldwrkr = m;
                } else {
                    ldwrkr = (lwork - n * n - 3 * n) / n;
                }
                itau = ir + ldwrkr * n;
                nwork = itau + n;

                zgeqrf(m, n, A, lda, &work[itau], &work[nwork],
                       lwork - nwork, &ierr);

                zlacpy("U", n, n, A, lda, &work[ir], ldwrkr);
                zlaset("L", n - 1, n - 1, CZERO, CZERO, &work[ir + 1],
                       ldwrkr);

                zungqr(m, n, n, A, lda, &work[itau], &work[nwork],
                       lwork - nwork, &ierr);
                ie = 0;
                itauq = itau;
                itaup = itauq + n;
                nwork = itaup + n;

                zgebrd(n, n, &work[ir], ldwrkr, S, &rwork[ie],
                       &work[itauq], &work[itaup], &work[nwork],
                       lwork - nwork, &ierr);

                iru = ie + n;
                irvt = iru + n * n;
                nrwork = irvt + n * n;
                dbdsdc("U", "I", n, S, &rwork[ie], &rwork[iru],
                       n, &rwork[irvt], n, dum, idum,
                       &rwork[nrwork], iwork, info);

                zlacp2("F", n, n, &rwork[iru], n, &work[iu], ldwrku);
                zunmbr("Q", "L", "N", n, n, n, &work[ir], ldwrkr,
                       &work[itauq], &work[iu], ldwrku,
                       &work[nwork], lwork - nwork, &ierr);

                zlacp2("F", n, n, &rwork[irvt], n, VT, ldvt);
                zunmbr("P", "R", "C", n, n, n, &work[ir], ldwrkr,
                       &work[itaup], VT, ldvt, &work[nwork],
                       lwork - nwork, &ierr);

                for (i = 0; i < m; i += ldwrkr) {
                    chunk = ((m - i) < ldwrkr) ? (m - i) : ldwrkr;
                    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                chunk, n, n, &CONE, &A[i], lda, &work[iu], ldwrku,
                                &CZERO, &work[ir], ldwrkr);
                    zlacpy("F", chunk, n, &work[ir], ldwrkr, &A[i], lda);
                }

            } else if (wntqs) {

                /* Path 3 (M >> N, JOBZ='S') */
                ir = 0;

                ldwrkr = n;
                itau = ir + ldwrkr * n;
                nwork = itau + n;

                zgeqrf(m, n, A, lda, &work[itau], &work[nwork],
                       lwork - nwork, &ierr);

                zlacpy("U", n, n, A, lda, &work[ir], ldwrkr);
                zlaset("L", n - 1, n - 1, CZERO, CZERO, &work[ir + 1],
                       ldwrkr);

                zungqr(m, n, n, A, lda, &work[itau], &work[nwork],
                       lwork - nwork, &ierr);
                ie = 0;
                itauq = itau;
                itaup = itauq + n;
                nwork = itaup + n;

                zgebrd(n, n, &work[ir], ldwrkr, S, &rwork[ie],
                       &work[itauq], &work[itaup], &work[nwork],
                       lwork - nwork, &ierr);

                iru = ie + n;
                irvt = iru + n * n;
                nrwork = irvt + n * n;
                dbdsdc("U", "I", n, S, &rwork[ie], &rwork[iru],
                       n, &rwork[irvt], n, dum, idum,
                       &rwork[nrwork], iwork, info);

                zlacp2("F", n, n, &rwork[iru], n, U, ldu);
                zunmbr("Q", "L", "N", n, n, n, &work[ir], ldwrkr,
                       &work[itauq], U, ldu, &work[nwork],
                       lwork - nwork, &ierr);

                zlacp2("F", n, n, &rwork[irvt], n, VT, ldvt);
                zunmbr("P", "R", "C", n, n, n, &work[ir], ldwrkr,
                       &work[itaup], VT, ldvt, &work[nwork],
                       lwork - nwork, &ierr);

                zlacpy("F", n, n, U, ldu, &work[ir], ldwrkr);
                cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                            m, n, n, &CONE, A, lda, &work[ir], ldwrkr,
                            &CZERO, U, ldu);

            } else if (wntqa) {

                /* Path 4 (M >> N, JOBZ='A') */
                iu = 0;

                ldwrku = n;
                itau = iu + ldwrku * n;
                nwork = itau + n;

                zgeqrf(m, n, A, lda, &work[itau], &work[nwork],
                       lwork - nwork, &ierr);
                zlacpy("L", m, n, A, lda, U, ldu);

                zungqr(m, m, n, U, ldu, &work[itau], &work[nwork],
                       lwork - nwork, &ierr);

                zlaset("L", n - 1, n - 1, CZERO, CZERO, &A[1], lda);
                ie = 0;
                itauq = itau;
                itaup = itauq + n;
                nwork = itaup + n;

                zgebrd(n, n, A, lda, S, &rwork[ie], &work[itauq],
                       &work[itaup], &work[nwork], lwork - nwork, &ierr);
                iru = ie + n;
                irvt = iru + n * n;
                nrwork = irvt + n * n;

                dbdsdc("U", "I", n, S, &rwork[ie], &rwork[iru],
                       n, &rwork[irvt], n, dum, idum,
                       &rwork[nrwork], iwork, info);

                zlacp2("F", n, n, &rwork[iru], n, &work[iu], ldwrku);
                zunmbr("Q", "L", "N", n, n, n, A, lda,
                       &work[itauq], &work[iu], ldwrku,
                       &work[nwork], lwork - nwork, &ierr);

                zlacp2("F", n, n, &rwork[irvt], n, VT, ldvt);
                zunmbr("P", "R", "C", n, n, n, A, lda,
                       &work[itaup], VT, ldvt, &work[nwork],
                       lwork - nwork, &ierr);

                cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                            m, n, n, &CONE, U, ldu, &work[iu], ldwrku,
                            &CZERO, A, lda);

                zlacpy("F", m, n, A, lda, U, ldu);

            }

        } else if (m >= mnthr2) {

            /* Path 5 (M >> N, but not as much as MNTHR1) */
            ie = 0;
            nrwork = ie + n;
            itauq = 0;
            itaup = itauq + n;
            nwork = itaup + n;

            zgebrd(m, n, A, lda, S, &rwork[ie], &work[itauq],
                   &work[itaup], &work[nwork], lwork - nwork, &ierr);
            if (wntqn) {

                /* Path 5n (M >> N, JOBZ='N') */
                dbdsdc("U", "N", n, S, &rwork[ie], dum, 1, dum, 1,
                       dum, idum, &rwork[nrwork], iwork, info);
            } else if (wntqo) {
                iu = nwork;
                iru = nrwork;
                irvt = iru + n * n;
                nrwork = irvt + n * n;

                /* Path 5o (M >> N, JOBZ='O') */
                zlacpy("U", n, n, A, lda, VT, ldvt);
                zungbr("P", n, n, n, VT, ldvt, &work[itaup],
                       &work[nwork], lwork - nwork, &ierr);

                zungbr("Q", m, n, n, A, lda, &work[itauq],
                       &work[nwork], lwork - nwork, &ierr);

                if (lwork >= m * n + 3 * n) {
                    ldwrku = m;
                } else {
                    ldwrku = (lwork - 3 * n) / n;
                }
                nwork = iu + ldwrku * n;

                dbdsdc("U", "I", n, S, &rwork[ie], &rwork[iru],
                       n, &rwork[irvt], n, dum, idum,
                       &rwork[nrwork], iwork, info);

                zlarcm(n, n, &rwork[irvt], n, VT, ldvt,
                       &work[iu], ldwrku, &rwork[nrwork]);
                zlacpy("F", n, n, &work[iu], ldwrku, VT, ldvt);

                nrwork = irvt;
                for (i = 0; i < m; i += ldwrku) {
                    chunk = ((m - i) < ldwrku) ? (m - i) : ldwrku;
                    zlacrm(chunk, n, &A[i], lda, &rwork[iru],
                           n, &work[iu], ldwrku, &rwork[nrwork]);
                    zlacpy("F", chunk, n, &work[iu], ldwrku,
                           &A[i], lda);
                }

            } else if (wntqs) {

                /* Path 5s (M >> N, JOBZ='S') */
                zlacpy("U", n, n, A, lda, VT, ldvt);
                zungbr("P", n, n, n, VT, ldvt, &work[itaup],
                       &work[nwork], lwork - nwork, &ierr);

                zlacpy("L", m, n, A, lda, U, ldu);
                zungbr("Q", m, n, n, U, ldu, &work[itauq],
                       &work[nwork], lwork - nwork, &ierr);

                iru = nrwork;
                irvt = iru + n * n;
                nrwork = irvt + n * n;
                dbdsdc("U", "I", n, S, &rwork[ie], &rwork[iru],
                       n, &rwork[irvt], n, dum, idum,
                       &rwork[nrwork], iwork, info);

                zlarcm(n, n, &rwork[irvt], n, VT, ldvt, A, lda,
                       &rwork[nrwork]);
                zlacpy("F", n, n, A, lda, VT, ldvt);

                nrwork = irvt;
                zlacrm(m, n, U, ldu, &rwork[iru], n, A, lda,
                       &rwork[nrwork]);
                zlacpy("F", m, n, A, lda, U, ldu);
            } else {

                /* Path 5a (M >> N, JOBZ='A') */
                zlacpy("U", n, n, A, lda, VT, ldvt);
                zungbr("P", n, n, n, VT, ldvt, &work[itaup],
                       &work[nwork], lwork - nwork, &ierr);

                zlacpy("L", m, n, A, lda, U, ldu);
                zungbr("Q", m, m, n, U, ldu, &work[itauq],
                       &work[nwork], lwork - nwork, &ierr);

                iru = nrwork;
                irvt = iru + n * n;
                nrwork = irvt + n * n;
                dbdsdc("U", "I", n, S, &rwork[ie], &rwork[iru],
                       n, &rwork[irvt], n, dum, idum,
                       &rwork[nrwork], iwork, info);

                zlarcm(n, n, &rwork[irvt], n, VT, ldvt, A, lda,
                       &rwork[nrwork]);
                zlacpy("F", n, n, A, lda, VT, ldvt);

                nrwork = irvt;
                zlacrm(m, n, U, ldu, &rwork[iru], n, A, lda,
                       &rwork[nrwork]);
                zlacpy("F", m, n, A, lda, U, ldu);
            }

        } else {

            /* Path 6 (M >= N, but not much larger) */
            ie = 0;
            nrwork = ie + n;
            itauq = 0;
            itaup = itauq + n;
            nwork = itaup + n;

            zgebrd(m, n, A, lda, S, &rwork[ie], &work[itauq],
                   &work[itaup], &work[nwork], lwork - nwork, &ierr);
            if (wntqn) {

                /* Path 6n (M >= N, JOBZ='N') */
                dbdsdc("U", "N", n, S, &rwork[ie], dum, 1, dum, 1,
                       dum, idum, &rwork[nrwork], iwork, info);
            } else if (wntqo) {
                iu = nwork;
                iru = nrwork;
                irvt = iru + n * n;
                nrwork = irvt + n * n;
                if (lwork >= m * n + 3 * n) {
                    ldwrku = m;
                } else {
                    ldwrku = (lwork - 3 * n) / n;
                }
                nwork = iu + ldwrku * n;

                /* Path 6o (M >= N, JOBZ='O') */
                dbdsdc("U", "I", n, S, &rwork[ie], &rwork[iru],
                       n, &rwork[irvt], n, dum, idum,
                       &rwork[nrwork], iwork, info);

                zlacp2("F", n, n, &rwork[irvt], n, VT, ldvt);
                zunmbr("P", "R", "C", n, n, n, A, lda,
                       &work[itaup], VT, ldvt, &work[nwork],
                       lwork - nwork, &ierr);

                if (lwork >= m * n + 3 * n) {

                    /* Path 6o-fast */
                    zlaset("F", m, n, CZERO, CZERO, &work[iu], ldwrku);
                    zlacp2("F", n, n, &rwork[iru], n, &work[iu], ldwrku);
                    zunmbr("Q", "L", "N", m, n, n, A, lda,
                           &work[itauq], &work[iu], ldwrku,
                           &work[nwork], lwork - nwork, &ierr);
                    zlacpy("F", m, n, &work[iu], ldwrku, A, lda);
                } else {

                    /* Path 6o-slow */
                    zungbr("Q", m, n, n, A, lda, &work[itauq],
                           &work[nwork], lwork - nwork, &ierr);

                    nrwork = irvt;
                    for (i = 0; i < m; i += ldwrku) {
                        chunk = ((m - i) < ldwrku) ? (m - i) : ldwrku;
                        zlacrm(chunk, n, &A[i], lda, &rwork[iru],
                               n, &work[iu], ldwrku, &rwork[nrwork]);
                        zlacpy("F", chunk, n, &work[iu], ldwrku,
                               &A[i], lda);
                    }
                }

            } else if (wntqs) {

                /* Path 6s (M >= N, JOBZ='S') */
                iru = nrwork;
                irvt = iru + n * n;
                nrwork = irvt + n * n;
                dbdsdc("U", "I", n, S, &rwork[ie], &rwork[iru],
                       n, &rwork[irvt], n, dum, idum,
                       &rwork[nrwork], iwork, info);

                zlaset("F", m, n, CZERO, CZERO, U, ldu);
                zlacp2("F", n, n, &rwork[iru], n, U, ldu);
                zunmbr("Q", "L", "N", m, n, n, A, lda,
                       &work[itauq], U, ldu, &work[nwork],
                       lwork - nwork, &ierr);

                zlacp2("F", n, n, &rwork[irvt], n, VT, ldvt);
                zunmbr("P", "R", "C", n, n, n, A, lda,
                       &work[itaup], VT, ldvt, &work[nwork],
                       lwork - nwork, &ierr);
            } else {

                /* Path 6a (M >= N, JOBZ='A') */
                iru = nrwork;
                irvt = iru + n * n;
                nrwork = irvt + n * n;
                dbdsdc("U", "I", n, S, &rwork[ie], &rwork[iru],
                       n, &rwork[irvt], n, dum, idum,
                       &rwork[nrwork], iwork, info);

                zlaset("F", m, m, CZERO, CZERO, U, ldu);
                if (m > n) {
                    zlaset("F", m - n, m - n, CZERO, CONE,
                           &U[n + n * ldu], ldu);
                }

                zlacp2("F", n, n, &rwork[iru], n, U, ldu);
                zunmbr("Q", "L", "N", m, m, n, A, lda,
                       &work[itauq], U, ldu, &work[nwork],
                       lwork - nwork, &ierr);

                zlacp2("F", n, n, &rwork[irvt], n, VT, ldvt);
                zunmbr("P", "R", "C", n, n, n, A, lda,
                       &work[itaup], VT, ldvt, &work[nwork],
                       lwork - nwork, &ierr);
            }

        }

    } else {

        if (n >= mnthr1) {

            if (wntqn) {

                /* Path 1t (N >> M, JOBZ='N') */
                itau = 0;
                nwork = itau + m;

                zgelqf(m, n, A, lda, &work[itau], &work[nwork],
                       lwork - nwork, &ierr);

                zlaset("U", m - 1, m - 1, CZERO, CZERO, &A[lda], lda);
                ie = 0;
                itauq = 0;
                itaup = itauq + m;
                nwork = itaup + m;

                zgebrd(m, m, A, lda, S, &rwork[ie], &work[itauq],
                       &work[itaup], &work[nwork], lwork - nwork, &ierr);
                nrwork = ie + m;

                dbdsdc("U", "N", m, S, &rwork[ie], dum, 1, dum, 1,
                       dum, idum, &rwork[nrwork], iwork, info);

            } else if (wntqo) {

                /* Path 2t (N >> M, JOBZ='O') */
                ivt = 0;
                ldwkvt = m;

                il = ivt + ldwkvt * m;
                if (lwork >= m * n + m * m + 3 * m) {
                    ldwrkl = m;
                    chunk = n;
                } else {
                    ldwrkl = m;
                    chunk = (lwork - m * m - 3 * m) / m;
                }
                itau = il + ldwrkl * chunk;
                nwork = itau + m;

                zgelqf(m, n, A, lda, &work[itau], &work[nwork],
                       lwork - nwork, &ierr);

                zlacpy("L", m, m, A, lda, &work[il], ldwrkl);
                zlaset("U", m - 1, m - 1, CZERO, CZERO,
                       &work[il + ldwrkl], ldwrkl);

                zunglq(m, n, m, A, lda, &work[itau], &work[nwork],
                       lwork - nwork, &ierr);
                ie = 0;
                itauq = itau;
                itaup = itauq + m;
                nwork = itaup + m;

                zgebrd(m, m, &work[il], ldwrkl, S, &rwork[ie],
                       &work[itauq], &work[itaup], &work[nwork],
                       lwork - nwork, &ierr);

                iru = ie + m;
                irvt = iru + m * m;
                nrwork = irvt + m * m;
                dbdsdc("U", "I", m, S, &rwork[ie], &rwork[iru],
                       m, &rwork[irvt], m, dum, idum,
                       &rwork[nrwork], iwork, info);

                zlacp2("F", m, m, &rwork[iru], m, U, ldu);
                zunmbr("Q", "L", "N", m, m, m, &work[il], ldwrkl,
                       &work[itauq], U, ldu, &work[nwork],
                       lwork - nwork, &ierr);

                zlacp2("F", m, m, &rwork[irvt], m, &work[ivt], ldwkvt);
                zunmbr("P", "R", "C", m, m, m, &work[il], ldwrkl,
                       &work[itaup], &work[ivt], ldwkvt,
                       &work[nwork], lwork - nwork, &ierr);

                for (i = 0; i < n; i += chunk) {
                    blk = ((n - i) < chunk) ? (n - i) : chunk;
                    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                m, blk, m, &CONE, &work[ivt], m,
                                &A[i * lda], lda, &CZERO, &work[il], ldwrkl);
                    zlacpy("F", m, blk, &work[il], ldwrkl, &A[i * lda], lda);
                }

            } else if (wntqs) {

                /* Path 3t (N >> M, JOBZ='S') */
                il = 0;

                ldwrkl = m;
                itau = il + ldwrkl * m;
                nwork = itau + m;

                zgelqf(m, n, A, lda, &work[itau], &work[nwork],
                       lwork - nwork, &ierr);

                zlacpy("L", m, m, A, lda, &work[il], ldwrkl);
                zlaset("U", m - 1, m - 1, CZERO, CZERO,
                       &work[il + ldwrkl], ldwrkl);

                zunglq(m, n, m, A, lda, &work[itau], &work[nwork],
                       lwork - nwork, &ierr);
                ie = 0;
                itauq = itau;
                itaup = itauq + m;
                nwork = itaup + m;

                zgebrd(m, m, &work[il], ldwrkl, S, &rwork[ie],
                       &work[itauq], &work[itaup], &work[nwork],
                       lwork - nwork, &ierr);

                iru = ie + m;
                irvt = iru + m * m;
                nrwork = irvt + m * m;
                dbdsdc("U", "I", m, S, &rwork[ie], &rwork[iru],
                       m, &rwork[irvt], m, dum, idum,
                       &rwork[nrwork], iwork, info);

                zlacp2("F", m, m, &rwork[iru], m, U, ldu);
                zunmbr("Q", "L", "N", m, m, m, &work[il], ldwrkl,
                       &work[itauq], U, ldu, &work[nwork],
                       lwork - nwork, &ierr);

                zlacp2("F", m, m, &rwork[irvt], m, VT, ldvt);
                zunmbr("P", "R", "C", m, m, m, &work[il], ldwrkl,
                       &work[itaup], VT, ldvt, &work[nwork],
                       lwork - nwork, &ierr);

                zlacpy("F", m, m, VT, ldvt, &work[il], ldwrkl);
                cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                            m, n, m, &CONE, &work[il], ldwrkl,
                            A, lda, &CZERO, VT, ldvt);

            } else if (wntqa) {

                /* Path 4t (N >> M, JOBZ='A') */
                ivt = 0;

                ldwkvt = m;
                itau = ivt + ldwkvt * m;
                nwork = itau + m;

                zgelqf(m, n, A, lda, &work[itau], &work[nwork],
                       lwork - nwork, &ierr);
                zlacpy("U", m, n, A, lda, VT, ldvt);

                zunglq(n, n, m, VT, ldvt, &work[itau], &work[nwork],
                       lwork - nwork, &ierr);

                zlaset("U", m - 1, m - 1, CZERO, CZERO, &A[lda], lda);
                ie = 0;
                itauq = itau;
                itaup = itauq + m;
                nwork = itaup + m;

                zgebrd(m, m, A, lda, S, &rwork[ie], &work[itauq],
                       &work[itaup], &work[nwork], lwork - nwork, &ierr);

                iru = ie + m;
                irvt = iru + m * m;
                nrwork = irvt + m * m;
                dbdsdc("U", "I", m, S, &rwork[ie], &rwork[iru],
                       m, &rwork[irvt], m, dum, idum,
                       &rwork[nrwork], iwork, info);

                zlacp2("F", m, m, &rwork[iru], m, U, ldu);
                zunmbr("Q", "L", "N", m, m, m, A, lda,
                       &work[itauq], U, ldu, &work[nwork],
                       lwork - nwork, &ierr);

                zlacp2("F", m, m, &rwork[irvt], m, &work[ivt], ldwkvt);
                zunmbr("P", "R", "C", m, m, m, A, lda,
                       &work[itaup], &work[ivt], ldwkvt,
                       &work[nwork], lwork - nwork, &ierr);

                cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                            m, n, m, &CONE, &work[ivt], ldwkvt,
                            VT, ldvt, &CZERO, A, lda);

                zlacpy("F", m, n, A, lda, VT, ldvt);

            }

        } else if (n >= mnthr2) {

            /* Path 5t (N >> M, but not as much as MNTHR1) */
            ie = 0;
            nrwork = ie + m;
            itauq = 0;
            itaup = itauq + m;
            nwork = itaup + m;

            zgebrd(m, n, A, lda, S, &rwork[ie], &work[itauq],
                   &work[itaup], &work[nwork], lwork - nwork, &ierr);

            if (wntqn) {

                /* Path 5tn (N >> M, JOBZ='N') */
                dbdsdc("L", "N", m, S, &rwork[ie], dum, 1, dum, 1,
                       dum, idum, &rwork[nrwork], iwork, info);
            } else if (wntqo) {
                irvt = nrwork;
                iru = irvt + m * m;
                nrwork = iru + m * m;
                ivt = nwork;

                /* Path 5to (N >> M, JOBZ='O') */
                zlacpy("L", m, m, A, lda, U, ldu);
                zungbr("Q", m, m, n, U, ldu, &work[itauq],
                       &work[nwork], lwork - nwork, &ierr);

                zungbr("P", m, n, m, A, lda, &work[itaup],
                       &work[nwork], lwork - nwork, &ierr);

                ldwkvt = m;
                if (lwork >= m * n + 3 * m) {
                    nwork = ivt + ldwkvt * n;
                    chunk = n;
                } else {
                    chunk = (lwork - 3 * m) / m;
                    nwork = ivt + ldwkvt * chunk;
                }

                dbdsdc("L", "I", m, S, &rwork[ie], &rwork[iru],
                       m, &rwork[irvt], m, dum, idum,
                       &rwork[nrwork], iwork, info);

                zlacrm(m, m, U, ldu, &rwork[iru], m,
                       &work[ivt], ldwkvt, &rwork[nrwork]);
                zlacpy("F", m, m, &work[ivt], ldwkvt, U, ldu);

                nrwork = iru;
                for (i = 0; i < n; i += chunk) {
                    blk = ((n - i) < chunk) ? (n - i) : chunk;
                    zlarcm(m, blk, &rwork[irvt], m, &A[i * lda],
                           lda, &work[ivt], ldwkvt, &rwork[nrwork]);
                    zlacpy("F", m, blk, &work[ivt], ldwkvt,
                           &A[i * lda], lda);
                }
            } else if (wntqs) {

                /* Path 5ts (N >> M, JOBZ='S') */
                zlacpy("L", m, m, A, lda, U, ldu);
                zungbr("Q", m, m, n, U, ldu, &work[itauq],
                       &work[nwork], lwork - nwork, &ierr);

                zlacpy("U", m, n, A, lda, VT, ldvt);
                zungbr("P", m, n, m, VT, ldvt, &work[itaup],
                       &work[nwork], lwork - nwork, &ierr);

                irvt = nrwork;
                iru = irvt + m * m;
                nrwork = iru + m * m;
                dbdsdc("L", "I", m, S, &rwork[ie], &rwork[iru],
                       m, &rwork[irvt], m, dum, idum,
                       &rwork[nrwork], iwork, info);

                zlacrm(m, m, U, ldu, &rwork[iru], m, A, lda,
                       &rwork[nrwork]);
                zlacpy("F", m, m, A, lda, U, ldu);

                nrwork = iru;
                zlarcm(m, n, &rwork[irvt], m, VT, ldvt, A, lda,
                       &rwork[nrwork]);
                zlacpy("F", m, n, A, lda, VT, ldvt);
            } else {

                /* Path 5ta (N >> M, JOBZ='A') */
                zlacpy("L", m, m, A, lda, U, ldu);
                zungbr("Q", m, m, n, U, ldu, &work[itauq],
                       &work[nwork], lwork - nwork, &ierr);

                zlacpy("U", m, n, A, lda, VT, ldvt);
                zungbr("P", n, n, m, VT, ldvt, &work[itaup],
                       &work[nwork], lwork - nwork, &ierr);

                irvt = nrwork;
                iru = irvt + m * m;
                nrwork = iru + m * m;
                dbdsdc("L", "I", m, S, &rwork[ie], &rwork[iru],
                       m, &rwork[irvt], m, dum, idum,
                       &rwork[nrwork], iwork, info);

                zlacrm(m, m, U, ldu, &rwork[iru], m, A, lda,
                       &rwork[nrwork]);
                zlacpy("F", m, m, A, lda, U, ldu);

                nrwork = iru;
                zlarcm(m, n, &rwork[irvt], m, VT, ldvt, A, lda,
                       &rwork[nrwork]);
                zlacpy("F", m, n, A, lda, VT, ldvt);
            }

        } else {

            /* Path 6t (N > M, but not much larger) */
            ie = 0;
            nrwork = ie + m;
            itauq = 0;
            itaup = itauq + m;
            nwork = itaup + m;

            zgebrd(m, n, A, lda, S, &rwork[ie], &work[itauq],
                   &work[itaup], &work[nwork], lwork - nwork, &ierr);
            if (wntqn) {

                /* Path 6tn (N > M, JOBZ='N') */
                dbdsdc("L", "N", m, S, &rwork[ie], dum, 1, dum, 1,
                       dum, idum, &rwork[nrwork], iwork, info);
            } else if (wntqo) {
                /* Path 6to (N > M, JOBZ='O') */
                ldwkvt = m;
                ivt = nwork;
                if (lwork >= m * n + 3 * m) {
                    zlaset("F", m, n, CZERO, CZERO, &work[ivt], ldwkvt);
                    nwork = ivt + ldwkvt * n;
                } else {
                    chunk = (lwork - 3 * m) / m;
                    nwork = ivt + ldwkvt * chunk;
                }

                irvt = nrwork;
                iru = irvt + m * m;
                nrwork = iru + m * m;
                dbdsdc("L", "I", m, S, &rwork[ie], &rwork[iru],
                       m, &rwork[irvt], m, dum, idum,
                       &rwork[nrwork], iwork, info);

                zlacp2("F", m, m, &rwork[iru], m, U, ldu);
                zunmbr("Q", "L", "N", m, m, n, A, lda,
                       &work[itauq], U, ldu, &work[nwork],
                       lwork - nwork, &ierr);

                if (lwork >= m * n + 3 * m) {

                    /* Path 6to-fast */
                    zlacp2("F", m, m, &rwork[irvt], m, &work[ivt],
                           ldwkvt);
                    zunmbr("P", "R", "C", m, n, m, A, lda,
                           &work[itaup], &work[ivt], ldwkvt,
                           &work[nwork], lwork - nwork, &ierr);
                    zlacpy("F", m, n, &work[ivt], ldwkvt, A, lda);
                } else {

                    /* Path 6to-slow */
                    zungbr("P", m, n, m, A, lda, &work[itaup],
                           &work[nwork], lwork - nwork, &ierr);

                    nrwork = iru;
                    for (i = 0; i < n; i += chunk) {
                        blk = ((n - i) < chunk) ? (n - i) : chunk;
                        zlarcm(m, blk, &rwork[irvt], m, &A[i * lda],
                               lda, &work[ivt], ldwkvt,
                               &rwork[nrwork]);
                        zlacpy("F", m, blk, &work[ivt], ldwkvt,
                               &A[i * lda], lda);
                    }
                }
            } else if (wntqs) {

                /* Path 6ts (N > M, JOBZ='S') */
                irvt = nrwork;
                iru = irvt + m * m;
                nrwork = iru + m * m;
                dbdsdc("L", "I", m, S, &rwork[ie], &rwork[iru],
                       m, &rwork[irvt], m, dum, idum,
                       &rwork[nrwork], iwork, info);

                zlacp2("F", m, m, &rwork[iru], m, U, ldu);
                zunmbr("Q", "L", "N", m, m, n, A, lda,
                       &work[itauq], U, ldu, &work[nwork],
                       lwork - nwork, &ierr);

                zlaset("F", m, n, CZERO, CZERO, VT, ldvt);
                zlacp2("F", m, m, &rwork[irvt], m, VT, ldvt);
                zunmbr("P", "R", "C", m, n, m, A, lda,
                       &work[itaup], VT, ldvt, &work[nwork],
                       lwork - nwork, &ierr);
            } else {

                /* Path 6ta (N > M, JOBZ='A') */
                irvt = nrwork;
                iru = irvt + m * m;
                nrwork = iru + m * m;

                dbdsdc("L", "I", m, S, &rwork[ie], &rwork[iru],
                       m, &rwork[irvt], m, dum, idum,
                       &rwork[nrwork], iwork, info);

                zlacp2("F", m, m, &rwork[iru], m, U, ldu);
                zunmbr("Q", "L", "N", m, m, n, A, lda,
                       &work[itauq], U, ldu, &work[nwork],
                       lwork - nwork, &ierr);

                zlaset("F", n, n, CZERO, CONE, VT, ldvt);

                zlacp2("F", m, m, &rwork[irvt], m, VT, ldvt);
                zunmbr("P", "R", "C", n, n, m, A, lda,
                       &work[itaup], VT, ldvt, &work[nwork],
                       lwork - nwork, &ierr);
            }

        }

    }

    /* Undo scaling if necessary */
    if (iscl == 1) {
        if (anrm > bignum) {
            dlascl("G", 0, 0, bignum, anrm, minmn, 1, S, minmn, &ierr);
        }
        if (*info != 0 && anrm > bignum) {
            dlascl("G", 0, 0, bignum, anrm, minmn - 1, 1,
                   &rwork[ie], minmn, &ierr);
        }
        if (anrm < smlnum) {
            dlascl("G", 0, 0, smlnum, anrm, minmn, 1, S, minmn, &ierr);
        }
        if (*info != 0 && anrm < smlnum) {
            dlascl("G", 0, 0, smlnum, anrm, minmn - 1, 1,
                   &rwork[ie], minmn, &ierr);
        }
    }

    /* Return optimal workspace in WORK(1) */
    work[0] = CMPLX((f64)maxwrk, 0.0);
}
