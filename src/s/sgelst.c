/**
 * @file sgelst.c
 * @brief SGELST solves overdetermined or underdetermined systems using QR or LQ factorization with compact WY representation.
 */

#include <cblas.h>
#include "semicolon_lapack_single.h"

/**
 * SGELST solves overdetermined or underdetermined real linear systems
 * involving an M-by-N matrix A, or its transpose, using a QR or LQ
 * factorization of A with compact WY representation of Q.
 *
 * It is assumed that A has full rank.
 *
 * @param[in] trans
 *          = 'N': the linear system involves A;
 *          = 'T': the linear system involves A**T.
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
 *          Double precision array, dimension (lda, n).
 *          On entry, the M-by-N matrix A.
 *          On exit, the factors from QR or LQ factorization.
 *
 * @param[in] lda
 *          The leading dimension of the array A. lda >= max(1, m).
 *
 * @param[in,out] B
 *          Double precision array, dimension (ldb, nrhs).
 *          On entry, the right hand side matrix B.
 *          On exit, the solution matrix X.
 *
 * @param[in] ldb
 *          The leading dimension of the array B. ldb >= max(1, m, n).
 *
 * @param[out] work
 *          Double precision workspace of size (max(1, lwork)).
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
void sgelst(
    const char* trans,
    const int m,
    const int n,
    const int nrhs,
    float* const restrict A,
    const int lda,
    float* const restrict B,
    const int ldb,
    float* restrict work,
    const int lwork,
    int* info)
{
    int lquery, tpsd;
    int brow, i, iascl, ibscl, j, lwopt, mn, mnnrhs, nb, nbmin, scllen;
    float anrm, bignum, bnrm, smlnum;
    int max_mn, max_ldb;

    *info = 0;
    mn = (m < n) ? m : n;
    lquery = (lwork == -1);
    if (!(trans[0] == 'N' || trans[0] == 'n' || trans[0] == 'T' || trans[0] == 't')) {
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
            int min_work = (mn > nrhs) ? mn : nrhs;
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
        work[0] = (float)lwopt;

    }

    if (*info != 0) {
        xerbla("SGELST ", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    max_mn = (m > n) ? m : n;
    if (mn == 0 || nrhs == 0) {
        slaset("F", max_mn, nrhs, 0.0f, 0.0f, B, ldb);
        work[0] = (float)lwopt;
        return;
    }

    if (nb > mn) nb = mn;

    nb = (nb < lwork / (mn + mnnrhs)) ? nb : (lwork / (mn + mnnrhs));

    nbmin = 2;

    if (nb < nbmin) {
        nb = 1;
    }

    smlnum = slamch("S") / slamch("P");
    bignum = 1.0f / smlnum;

    anrm = slange("M", m, n, A, lda, NULL);
    iascl = 0;
    if (anrm > 0.0f && anrm < smlnum) {

        slascl("G", 0, 0, anrm, smlnum, m, n, A, lda, info);
        iascl = 1;
    } else if (anrm > bignum) {

        slascl("G", 0, 0, anrm, bignum, m, n, A, lda, info);
        iascl = 2;
    } else if (anrm == 0.0f) {

        slaset("F", max_mn, nrhs, 0.0f, 0.0f, B, ldb);
        work[0] = (float)lwopt;
        return;
    }

    brow = m;
    if (tpsd)
        brow = n;
    bnrm = slange("M", brow, nrhs, B, ldb, NULL);
    ibscl = 0;
    if (bnrm > 0.0f && bnrm < smlnum) {

        slascl("G", 0, 0, bnrm, smlnum, brow, nrhs, B, ldb, info);
        ibscl = 1;
    } else if (bnrm > bignum) {

        slascl("G", 0, 0, bnrm, bignum, brow, nrhs, B, ldb, info);
        ibscl = 2;
    }

    if (m >= n) {

        sgeqrt(m, n, nb, A, lda, &work[0], nb, &work[mn * nb], info);

        if (!tpsd) {

            sgemqrt("L", "T", m, nrhs, n, nb, A, lda,
                    &work[0], nb, B, ldb, &work[mn * nb], info);

            strtrs("U", "N", "N", n, nrhs, A, lda, B, ldb, info);

            if (*info > 0) {
                return;
            }

            scllen = n;

        } else {

            strtrs("U", "T", "N", n, nrhs, A, lda, B, ldb, info);

            if (*info > 0) {
                return;
            }

            for (j = 0; j < nrhs; j++) {
                for (i = n; i < m; i++) {
                    B[i + j * ldb] = 0.0f;
                }
            }

            sgemqrt("L", "N", m, nrhs, n, nb, A, lda,
                    &work[0], nb, B, ldb, &work[mn * nb], info);

            scllen = m;

        }

    } else {

        sgelqt(m, n, nb, A, lda, &work[0], nb, &work[mn * nb], info);

        if (!tpsd) {

            strtrs("L", "N", "N", m, nrhs, A, lda, B, ldb, info);

            if (*info > 0) {
                return;
            }

            for (j = 0; j < nrhs; j++) {
                for (i = m; i < n; i++) {
                    B[i + j * ldb] = 0.0f;
                }
            }

            sgemlqt("L", "T", n, nrhs, m, nb, A, lda,
                    &work[0], nb, B, ldb, &work[mn * nb], info);

            scllen = n;

        } else {

            sgemlqt("L", "N", n, nrhs, m, nb, A, lda,
                    &work[0], nb, B, ldb, &work[mn * nb], info);

            strtrs("L", "T", "N", m, nrhs, A, lda, B, ldb, info);

            if (*info > 0) {
                return;
            }

            scllen = m;

        }

    }

    if (iascl == 1) {
        slascl("G", 0, 0, anrm, smlnum, scllen, nrhs, B, ldb, info);
    } else if (iascl == 2) {
        slascl("G", 0, 0, anrm, bignum, scllen, nrhs, B, ldb, info);
    }
    if (ibscl == 1) {
        slascl("G", 0, 0, smlnum, bnrm, scllen, nrhs, B, ldb, info);
    } else if (ibscl == 2) {
        slascl("G", 0, 0, bignum, bnrm, scllen, nrhs, B, ldb, info);
    }

    work[0] = (float)lwopt;
}
