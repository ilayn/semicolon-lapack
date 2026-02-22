/**
 * @file cgels.c
 * @brief CGELS solves overdetermined or underdetermined complex linear systems
 *        using QR or LQ factorization.
 */

#include <complex.h>
#include <math.h>
#include "semicolon_cblas.h"
#include "../include/lapack_tuning.h"
#include "semicolon_lapack_complex_single.h"

/**
 * CGELS solves overdetermined or underdetermined complex linear systems
 * involving an M-by-N matrix A, or its conjugate-transpose, using a QR
 * or LQ factorization of A.
 *
 * It is assumed that A has full rank, and only a rudimentary protection
 * against rank-deficient matrices is provided.
 *
 * The following options are provided:
 *
 * 1. If TRANS = 'N' and m >= n: find the least squares solution of
 *    an overdetermined system, minimize || B - A*X ||.
 *
 * 2. If TRANS = 'N' and m < n: find the minimum norm solution of
 *    an underdetermined system A * X = B.
 *
 * 3. If TRANS = 'C' and m >= n: find the minimum norm solution of
 *    an underdetermined system A^H * X = B.
 *
 * 4. If TRANS = 'C' and m < n: find the least squares solution of
 *    an overdetermined system, minimize || B - A^H * X ||.
 *
 * @param[in]     trans  'N': the linear system involves A;
 *                       'C': the linear system involves A^H.
 * @param[in]     m      The number of rows of A. m >= 0.
 * @param[in]     n      The number of columns of A. n >= 0.
 * @param[in]     nrhs   The number of right hand sides. nrhs >= 0.
 * @param[in,out] A      Complex*16 array, dimension (lda, n).
 *                       On entry, the m-by-n matrix A.
 *                       On exit, overwritten by its QR or LQ factorization.
 * @param[in]     lda    Leading dimension of A. lda >= max(1, m).
 * @param[in,out] B      Complex*16 array, dimension (ldb, nrhs).
 *                       On entry, the right hand side matrix B.
 *                       On exit, overwritten by the solution vectors.
 * @param[in]     ldb    Leading dimension of B. ldb >= max(1, m, n).
 * @param[out]    work   Complex*16 array, dimension (max(1, lwork)).
 *                       On exit, work[0] returns the optimal lwork.
 * @param[in]     lwork  Dimension of work. lwork >= max(1, mn + max(mn, nrhs)).
 *                       If lwork == -1, workspace query only.
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 *                         - > 0: if info = i, the i-th diagonal element of the
 *                           triangular factor of A is zero, so A does not have
 *                           full rank.
 */
void cgels(const char* trans,
           const INT m, const INT n, const INT nrhs,
           c64* restrict A, const INT lda,
           c64* restrict B, const INT ldb,
           c64* restrict work, const INT lwork,
           INT* info)
{
    const c64 CZERO = CMPLXF(0.0f, 0.0f);

    INT lquery, tpsd;
    INT brow, iascl, ibscl, mn, nb, scllen, wsize;
    f32 anrm, bignum, bnrm, smlnum;
    f32 rwork[1];
    INT iinfo;

    /* Test the input arguments */
    *info = 0;
    mn = m < n ? m : n;
    lquery = (lwork == -1);

    if (trans[0] != 'N' && trans[0] != 'n' &&
        trans[0] != 'C' && trans[0] != 'c') {
        *info = -1;
    } else if (m < 0) {
        *info = -2;
    } else if (n < 0) {
        *info = -3;
    } else if (nrhs < 0) {
        *info = -4;
    } else if (lda < (m > 1 ? m : 1)) {
        *info = -6;
    } else if (ldb < (m > n ? (m > 1 ? m : 1) : (n > 1 ? n : 1))) {
        *info = -8;
    } else if (lwork < (mn + (mn > nrhs ? mn : nrhs) > 1 ?
               mn + (mn > nrhs ? mn : nrhs) : 1) && !lquery) {
        *info = -10;
    }

    /* Figure out optimal block size */
    if (*info == 0 || *info == -10) {
        tpsd = (trans[0] != 'N' && trans[0] != 'n');

        if (m >= n) {
            nb = lapack_get_nb("GEQRF");
            INT nb2 = lapack_get_nb("ORMQR");
            if (nb2 > nb) nb = nb2;
        } else {
            nb = lapack_get_nb("GELQF");
            INT nb2 = lapack_get_nb("ORMLQ");
            if (nb2 > nb) nb = nb2;
        }

        INT mn_nrhs = mn > nrhs ? mn : nrhs;
        wsize = mn + mn_nrhs * nb;
        if (wsize < 1) wsize = 1;
        work[0] = (c64)wsize;
    }

    if (*info != 0) {
        xerbla("CGELS ", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    /* Quick return if possible */
    if (mn == 0 || nrhs == 0) {
        INT maxmn = m > n ? m : n;
        claset("F", maxmn, nrhs, CZERO, CZERO, B, ldb);
        return;
    }

    /* Get machine parameters */
    smlnum = slamch("S") / slamch("P");
    bignum = 1.0f / smlnum;

    /* Scale A, B if max element outside range [SMLNUM, BIGNUM] */
    anrm = clange("M", m, n, A, lda, rwork);
    iascl = 0;
    if (anrm > 0.0f && anrm < smlnum) {
        /* Scale matrix norm up to SMLNUM */
        clascl("G", 0, 0, anrm, smlnum, m, n, A, lda, &iinfo);
        iascl = 1;
    } else if (anrm > bignum) {
        /* Scale matrix norm down to BIGNUM */
        clascl("G", 0, 0, anrm, bignum, m, n, A, lda, &iinfo);
        iascl = 2;
    } else if (anrm == 0.0f) {
        /* Matrix all zero. Return zero solution. */
        INT maxmn = m > n ? m : n;
        claset("F", maxmn, nrhs, CZERO, CZERO, B, ldb);
        work[0] = (c64)wsize;
        return;
    }

    brow = m;
    if (tpsd)
        brow = n;
    bnrm = clange("M", brow, nrhs, B, ldb, rwork);
    ibscl = 0;
    if (bnrm > 0.0f && bnrm < smlnum) {
        /* Scale matrix norm up to SMLNUM */
        clascl("G", 0, 0, bnrm, smlnum, brow, nrhs, B, ldb, &iinfo);
        ibscl = 1;
    } else if (bnrm > bignum) {
        /* Scale matrix norm down to BIGNUM */
        clascl("G", 0, 0, bnrm, bignum, brow, nrhs, B, ldb, &iinfo);
        ibscl = 2;
    }

    if (m >= n) {
        /* Compute QR factorization of A */
        /* work[0..mn-1] = tau, work[mn..] = sub-workspace */
        cgeqrf(m, n, A, lda, work, &work[mn], lwork - mn, &iinfo);

        if (!tpsd) {
            /* Least-Squares Problem min || A * X - B ||
             *
             * B(0:m-1, 0:nrhs-1) := Q^H * B(0:m-1, 0:nrhs-1) */
            cunmqr("L", "C", m, nrhs, n, A, lda,
                   work, B, ldb, &work[mn], lwork - mn, &iinfo);

            /* B(0:n-1, 0:nrhs-1) := inv(R) * B(0:n-1, 0:nrhs-1) */
            ctrtrs("U", "N", "N", n, nrhs, A, lda, B, ldb, info);

            if (*info > 0) {
                return;
            }

            scllen = n;
        } else {
            /* Underdetermined system of equations A^H * X = B
             *
             * B(0:n-1, 0:nrhs-1) := inv(R^H) * B(0:n-1, 0:nrhs-1) */
            ctrtrs("U", "C", "N", n, nrhs, A, lda, B, ldb, info);

            if (*info > 0) {
                return;
            }

            /* B(n:m-1, 0:nrhs-1) = 0 */
            for (INT j = 0; j < nrhs; j++) {
                for (INT i = n; i < m; i++) {
                    B[i + j * ldb] = CZERO;
                }
            }

            /* B(0:m-1, 0:nrhs-1) := Q * B(0:n-1, 0:nrhs-1) */
            cunmqr("L", "N", m, nrhs, n, A, lda,
                   work, B, ldb, &work[mn], lwork - mn, &iinfo);

            scllen = m;
        }
    } else {
        /* Compute LQ factorization of A */
        /* work[0..mn-1] = tau, work[mn..] = sub-workspace */
        cgelqf(m, n, A, lda, work, &work[mn], lwork - mn, &iinfo);

        if (!tpsd) {
            /* Underdetermined system of equations A * X = B
             *
             * B(0:m-1, 0:nrhs-1) := inv(L) * B(0:m-1, 0:nrhs-1) */
            ctrtrs("L", "N", "N", m, nrhs, A, lda, B, ldb, info);

            if (*info > 0) {
                return;
            }

            /* B(m:n-1, 0:nrhs-1) = 0 */
            for (INT j = 0; j < nrhs; j++) {
                for (INT i = m; i < n; i++) {
                    B[i + j * ldb] = CZERO;
                }
            }

            /* B(0:n-1, 0:nrhs-1) := Q^H * B(0:m-1, 0:nrhs-1) */
            cunmlq("L", "C", n, nrhs, m, A, lda,
                   work, B, ldb, &work[mn], lwork - mn, &iinfo);

            scllen = n;
        } else {
            /* Overdetermined system min || A^H * X - B ||
             *
             * B(0:n-1, 0:nrhs-1) := Q * B(0:n-1, 0:nrhs-1) */
            cunmlq("L", "N", n, nrhs, m, A, lda,
                   work, B, ldb, &work[mn], lwork - mn, &iinfo);

            /* B(0:m-1, 0:nrhs-1) := inv(L^H) * B(0:m-1, 0:nrhs-1) */
            ctrtrs("L", "C", "N", m, nrhs, A, lda, B, ldb, info);

            if (*info > 0) {
                return;
            }

            scllen = m;
        }
    }

    /* Undo scaling */
    if (iascl == 1) {
        clascl("G", 0, 0, anrm, smlnum, scllen, nrhs, B, ldb, &iinfo);
    } else if (iascl == 2) {
        clascl("G", 0, 0, anrm, bignum, scllen, nrhs, B, ldb, &iinfo);
    }
    if (ibscl == 1) {
        clascl("G", 0, 0, smlnum, bnrm, scllen, nrhs, B, ldb, &iinfo);
    } else if (ibscl == 2) {
        clascl("G", 0, 0, bignum, bnrm, scllen, nrhs, B, ldb, &iinfo);
    }

    work[0] = (c64)wsize;
}
