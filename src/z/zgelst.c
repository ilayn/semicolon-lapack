/**
 * @file zgelst.c
 * @brief ZGELST solves overdetermined or underdetermined systems using QR or LQ factorization with compact WY representation.
 */

#include <complex.h>
#include "semicolon_cblas.h"
#include "semicolon_lapack_complex_double.h"

/**
 * ZGELST solves overdetermined or underdetermined complex linear systems
 * involving an M-by-N matrix A, or its conjugate-transpose, using a QR or LQ
 * factorization of A with compact WY representation of Q.
 *
 * It is assumed that A has full rank.
 *
 * @param[in] trans
 *          = 'N': the linear system involves A;
 *          = 'C': the linear system involves A**H.
 *
 * @param[in] m
 *          The number of rows of the matrix A. m >= 0.
 *
 * @param[in] n
 *          The number of columns of the matrix A. n >= 0.
 *
 * @param[in] nrhs
 *          The number of right hand sides. nrhs >= 0.
 *
 * @param[in,out] A
 *          Double complex array, dimension (lda, n).
 *          On entry, the M-by-N matrix A.
 *          On exit, the factors from QR or LQ factorization.
 *
 * @param[in] lda
 *          The leading dimension of the array A. lda >= max(1, m).
 *
 * @param[in,out] B
 *          Double complex array, dimension (ldb, nrhs).
 *          On entry, the right hand side matrix B.
 *          On exit, the solution matrix X.
 *
 * @param[in] ldb
 *          The leading dimension of the array B. ldb >= max(1, m, n).
 *
 * @param[out] work
 *          Double complex workspace of size (max(1, lwork)).
 *          On exit, work[0] returns the optimal lwork.
 *
 * @param[in] lwork
 *          The dimension of the array work.
 *          lwork >= max(1, mn + max(mn, nrhs)).
 *          If lwork = -1, then a workspace query is assumed.
 *
 * @param[out] info
 *                         - = 0:  successful exit
 *                         - < 0:  if info = -i, the i-th argument had an illegal value
 *                         - > 0:  if info = i, the i-th diagonal element of the triangular
 *                           factor is zero.
 */
void zgelst(
    const char* trans,
    const INT m,
    const INT n,
    const INT nrhs,
    c128* restrict A,
    const INT lda,
    c128* restrict B,
    const INT ldb,
    c128* restrict work,
    const INT lwork,
    INT* info)
{
    INT lquery, tpsd;
    INT brow, i, iascl, ibscl, j, lwopt, mn, mnnrhs, nb, nbmin, scllen;
    f64 anrm, bignum, bnrm, smlnum;
    INT max_mn, max_ldb;

    *info = 0;
    mn = (m < n) ? m : n;
    lquery = (lwork == -1);
    if (!(trans[0] == 'N' || trans[0] == 'n' || trans[0] == 'C' || trans[0] == 'c')) {
        *info = -1;
    } else if (m < 0) {
        *info = -2;
    } else if (n < 0) {
        *info = -3;
    } else if (nrhs < 0) {
        *info = -4;
    } else if (lda < (1 > m ? 1 : m)) {
        *info = -6;
    } else {
        max_mn = (m > n) ? m : n;
        max_ldb = (1 > max_mn) ? 1 : max_mn;
        if (ldb < max_ldb) {
            *info = -8;
        } else {
            INT min_work = (mn > nrhs) ? mn : nrhs;
            min_work = 1 > (mn + min_work) ? 1 : (mn + min_work);
            if (lwork < min_work && !lquery) {
                *info = -10;
            }
        }
    }

    if (*info == 0 || *info == -10) {

        tpsd = 1;
        if (trans[0] == 'N' || trans[0] == 'n')
            tpsd = 0;

        nb = 32;

        mnnrhs = (mn > nrhs) ? mn : nrhs;
        lwopt = (1 > (mn + mnnrhs) * nb) ? 1 : ((mn + mnnrhs) * nb);
        work[0] = (c128)lwopt;

    }

    if (*info != 0) {
        xerbla("ZGELST ", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    max_mn = (m > n) ? m : n;
    if (mn == 0 || nrhs == 0) {
        const c128 CZERO = CMPLX(0.0, 0.0);
        zlaset("F", max_mn, nrhs, CZERO, CZERO, B, ldb);
        work[0] = (c128)lwopt;
        return;
    }

    if (nb > mn) nb = mn;

    nb = (nb < lwork / (mn + mnnrhs)) ? nb : (lwork / (mn + mnnrhs));

    nbmin = 2;

    if (nb < nbmin) {
        nb = 1;
    }

    smlnum = dlamch("S") / dlamch("P");
    bignum = 1.0 / smlnum;

    anrm = zlange("M", m, n, A, lda, NULL);
    iascl = 0;
    if (anrm > 0.0 && anrm < smlnum) {

        zlascl("G", 0, 0, anrm, smlnum, m, n, A, lda, info);
        iascl = 1;
    } else if (anrm > bignum) {

        zlascl("G", 0, 0, anrm, bignum, m, n, A, lda, info);
        iascl = 2;
    } else if (anrm == 0.0) {

        const c128 CZERO = CMPLX(0.0, 0.0);
        zlaset("F", max_mn, nrhs, CZERO, CZERO, B, ldb);
        work[0] = (c128)lwopt;
        return;
    }

    brow = m;
    if (tpsd)
        brow = n;
    bnrm = zlange("M", brow, nrhs, B, ldb, NULL);
    ibscl = 0;
    if (bnrm > 0.0 && bnrm < smlnum) {

        zlascl("G", 0, 0, bnrm, smlnum, brow, nrhs, B, ldb, info);
        ibscl = 1;
    } else if (bnrm > bignum) {

        zlascl("G", 0, 0, bnrm, bignum, brow, nrhs, B, ldb, info);
        ibscl = 2;
    }

    if (m >= n) {

        zgeqrt(m, n, nb, A, lda, &work[0], nb, &work[mn * nb], info);

        if (!tpsd) {

            zgemqrt("L", "C", m, nrhs, n, nb, A, lda,
                    &work[0], nb, B, ldb, &work[mn * nb], info);

            ztrtrs("U", "N", "N", n, nrhs, A, lda, B, ldb, info);

            if (*info > 0) {
                return;
            }

            scllen = n;

        } else {

            ztrtrs("U", "C", "N", n, nrhs, A, lda, B, ldb, info);

            if (*info > 0) {
                return;
            }

            for (j = 0; j < nrhs; j++) {
                for (i = n; i < m; i++) {
                    B[i + j * ldb] = CMPLX(0.0, 0.0);
                }
            }

            zgemqrt("L", "N", m, nrhs, n, nb, A, lda,
                    &work[0], nb, B, ldb, &work[mn * nb], info);

            scllen = m;

        }

    } else {

        zgelqt(m, n, nb, A, lda, &work[0], nb, &work[mn * nb], info);

        if (!tpsd) {

            ztrtrs("L", "N", "N", m, nrhs, A, lda, B, ldb, info);

            if (*info > 0) {
                return;
            }

            for (j = 0; j < nrhs; j++) {
                for (i = m; i < n; i++) {
                    B[i + j * ldb] = CMPLX(0.0, 0.0);
                }
            }

            zgemlqt("L", "C", n, nrhs, m, nb, A, lda,
                    &work[0], nb, B, ldb, &work[mn * nb], info);

            scllen = n;

        } else {

            zgemlqt("L", "N", n, nrhs, m, nb, A, lda,
                    &work[0], nb, B, ldb, &work[mn * nb], info);

            ztrtrs("L", "C", "N", m, nrhs, A, lda, B, ldb, info);

            if (*info > 0) {
                return;
            }

            scllen = m;

        }

    }

    if (iascl == 1) {
        zlascl("G", 0, 0, anrm, smlnum, scllen, nrhs, B, ldb, info);
    } else if (iascl == 2) {
        zlascl("G", 0, 0, anrm, bignum, scllen, nrhs, B, ldb, info);
    }
    if (ibscl == 1) {
        zlascl("G", 0, 0, smlnum, bnrm, scllen, nrhs, B, ldb, info);
    } else if (ibscl == 2) {
        zlascl("G", 0, 0, bignum, bnrm, scllen, nrhs, B, ldb, info);
    }

    work[0] = (c128)lwopt;
}
