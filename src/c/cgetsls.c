/**
 * @file cgetsls.c
 * @brief CGETSLS solves overdetermined or underdetermined linear systems using tall skinny QR or short wide LQ factorization.
 */

#include <math.h>
#include <complex.h>
#include <float.h>
#include "semicolon_cblas.h"
#include "semicolon_lapack_complex_single.h"

/**
 * CGETSLS solves overdetermined or underdetermined complex linear systems
 * involving an M-by-N matrix A, using a tall skinny QR or short wide LQ
 * factorization of A.
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
 *    an undetermined system A**H * X = B.
 *
 * 4. If TRANS = 'C' and m < n: find the least squares solution of
 *    an overdetermined system, minimize || B - A**H * X ||.
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
 *          Single complex array, dimension (lda, n).
 *          On entry, the M-by-N matrix A.
 *          On exit, A is overwritten by details of its QR or LQ factorization.
 *
 * @param[in] lda
 *          The leading dimension of the array A. lda >= max(1, m).
 *
 * @param[in,out] B
 *          Single complex array, dimension (ldb, nrhs).
 *          On entry, the right hand side matrix B.
 *          On exit, the solution vectors.
 *
 * @param[in] ldb
 *          The leading dimension of the array B. ldb >= max(1, m, n).
 *
 * @param[out] work
 *          Single complex array, dimension (max(1, lwork)).
 *          On exit, if info = 0, work[0] contains the optimal lwork.
 *
 * @param[in] lwork
 *          The dimension of the array work.
 *          If lwork = -1 or -2, a workspace query is assumed.
 *          If lwork = -1, returns optimal workspace size.
 *          If lwork = -2, returns minimal workspace size.
 *
 * @param[out] info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 *                         - > 0: if info = i, the i-th diagonal element of the triangular
 *                           factor of A is exactly zero.
 */
void cgetsls(
    const char* trans,
    const INT m,
    const INT n,
    const INT nrhs,
    c64* restrict A,
    const INT lda,
    c64* restrict B,
    const INT ldb,
    c64* restrict work,
    const INT lwork,
    INT* info)
{
    const c64 czero = CMPLXF(0.0f, 0.0f);
    const f32 zero = 0.0f;
    const f32 one = 1.0f;

    INT i, iascl, ibscl, j, maxmn, brow;
    INT scllen, tszo = 0, tszm = 0, lwo = 0, lwm = 0, lw1, lw2;
    INT wsizeo, wsizem, info2;
    INT lquery, tran;
    f32 anrm, bignum, bnrm, smlnum;
    c64 tq[5], workq[1];

    *info = 0;
    maxmn = (m > n) ? m : n;
    tran = (trans[0] == 'C' || trans[0] == 'c');

    lquery = (lwork == -1 || lwork == -2);

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
        INT ldb_min = 1;
        if (m > ldb_min) ldb_min = m;
        if (n > ldb_min) ldb_min = n;
        if (ldb < ldb_min) {
            *info = -8;
        }
    }

    if (*info == 0) {
        INT minmnrhs = m;
        if (n < minmnrhs) minmnrhs = n;
        if (nrhs < minmnrhs) minmnrhs = nrhs;

        if (minmnrhs == 0) {
            wsizem = 1;
            wsizeo = 1;
        } else if (m >= n) {
            cgeqr(m, n, A, lda, tq, -1, workq, -1, &info2);
            tszo = (INT)crealf(tq[0]);
            lwo = (INT)crealf(workq[0]);
            cgemqr("L", trans, m, nrhs, n, A, lda, tq, tszo, B, ldb, workq, -1, &info2);
            if ((INT)crealf(workq[0]) > lwo) lwo = (INT)crealf(workq[0]);
            cgeqr(m, n, A, lda, tq, -2, workq, -2, &info2);
            tszm = (INT)crealf(tq[0]);
            lwm = (INT)crealf(workq[0]);
            cgemqr("L", trans, m, nrhs, n, A, lda, tq, tszm, B, ldb, workq, -1, &info2);
            if ((INT)crealf(workq[0]) > lwm) lwm = (INT)crealf(workq[0]);
            wsizeo = tszo + lwo;
            wsizem = tszm + lwm;
        } else {
            cgelq(m, n, A, lda, tq, -1, workq, -1, &info2);
            tszo = (INT)crealf(tq[0]);
            lwo = (INT)crealf(workq[0]);
            cgemlq("L", trans, n, nrhs, m, A, lda, tq, tszo, B, ldb, workq, -1, &info2);
            if ((INT)crealf(workq[0]) > lwo) lwo = (INT)crealf(workq[0]);
            cgelq(m, n, A, lda, tq, -2, workq, -2, &info2);
            tszm = (INT)crealf(tq[0]);
            lwm = (INT)crealf(workq[0]);
            cgemlq("L", trans, n, nrhs, m, A, lda, tq, tszm, B, ldb, workq, -1, &info2);
            if ((INT)crealf(workq[0]) > lwm) lwm = (INT)crealf(workq[0]);
            wsizeo = tszo + lwo;
            wsizem = tszm + lwm;
        }

        if (lwork < wsizem && !lquery) {
            *info = -10;
        }

        work[0] = (f32)(wsizeo);
    }

    if (*info != 0) {
        xerbla("CGETSLS", -(*info));
        return;
    }
    if (lquery) {
        if (lwork == -2) work[0] = (f32)(wsizem);
        return;
    }
    if (lwork < wsizeo) {
        lw1 = tszm;
        lw2 = lwm;
    } else {
        lw1 = tszo;
        lw2 = lwo;
    }

    {
        INT minmnrhs = m;
        if (n < minmnrhs) minmnrhs = n;
        if (nrhs < minmnrhs) minmnrhs = nrhs;
        if (minmnrhs == 0) {
            claset("F", maxmn, nrhs, czero, czero, B, ldb);
            return;
        }
    }

    smlnum = slamch("S") / slamch("P");
    bignum = one / smlnum;

    anrm = clange("M", m, n, A, lda, NULL);
    iascl = 0;
    if (anrm > zero && anrm < smlnum) {
        clascl("G", 0, 0, anrm, smlnum, m, n, A, lda, info);
        iascl = 1;
    } else if (anrm > bignum) {
        clascl("G", 0, 0, anrm, bignum, m, n, A, lda, info);
        iascl = 2;
    } else if (anrm == zero) {
        claset("F", maxmn, nrhs, czero, czero, B, ldb);
        work[0] = (f32)(tszo + lwo);
        return;
    }

    brow = m;
    if (tran) brow = n;
    bnrm = clange("M", brow, nrhs, B, ldb, NULL);
    ibscl = 0;
    if (bnrm > zero && bnrm < smlnum) {
        clascl("G", 0, 0, bnrm, smlnum, brow, nrhs, B, ldb, info);
        ibscl = 1;
    } else if (bnrm > bignum) {
        clascl("G", 0, 0, bnrm, bignum, brow, nrhs, B, ldb, info);
        ibscl = 2;
    }

    if (m >= n) {

        cgeqr(m, n, A, lda, &work[lw2], lw1, work, lw2, info);

        if (!tran) {

            cgemqr("L", "C", m, nrhs, n, A, lda, &work[lw2], lw1, B, ldb, work, lw2, info);

            ctrtrs("U", "N", "N", n, nrhs, A, lda, B, ldb, info);
            if (*info > 0) {
                return;
            }
            scllen = n;
        } else {

            ctrtrs("U", "C", "N", n, nrhs, A, lda, B, ldb, info);
            if (*info > 0) {
                return;
            }

            for (j = 0; j < nrhs; j++) {
                for (i = n; i < m; i++) {
                    B[i + j * ldb] = czero;
                }
            }

            cgemqr("L", "N", m, nrhs, n, A, lda, &work[lw2], lw1, B, ldb, work, lw2, info);

            scllen = m;
        }

    } else {

        cgelq(m, n, A, lda, &work[lw2], lw1, work, lw2, info);

        if (!tran) {

            ctrtrs("L", "N", "N", m, nrhs, A, lda, B, ldb, info);
            if (*info > 0) {
                return;
            }

            for (j = 0; j < nrhs; j++) {
                for (i = m; i < n; i++) {
                    B[i + j * ldb] = czero;
                }
            }

            cgemlq("L", "C", n, nrhs, m, A, lda, &work[lw2], lw1, B, ldb, work, lw2, info);

            scllen = n;

        } else {

            cgemlq("L", "N", n, nrhs, m, A, lda, &work[lw2], lw1, B, ldb, work, lw2, info);

            ctrtrs("L", "C", "N", m, nrhs, A, lda, B, ldb, info);
            if (*info > 0) {
                return;
            }

            scllen = m;
        }

    }

    if (iascl == 1) {
        clascl("G", 0, 0, anrm, smlnum, scllen, nrhs, B, ldb, info);
    } else if (iascl == 2) {
        clascl("G", 0, 0, anrm, bignum, scllen, nrhs, B, ldb, info);
    }
    if (ibscl == 1) {
        clascl("G", 0, 0, smlnum, bnrm, scllen, nrhs, B, ldb, info);
    } else if (ibscl == 2) {
        clascl("G", 0, 0, bignum, bnrm, scllen, nrhs, B, ldb, info);
    }

    work[0] = (f32)(tszo + lwo);
}
