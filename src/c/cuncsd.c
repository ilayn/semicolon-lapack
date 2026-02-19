/**
 * @file cuncsd.c
 * @brief CUNCSD computes the CS decomposition of an M-by-M partitioned unitary matrix.
 */

#include <complex.h>
#include <cblas.h>
#include "semicolon_lapack_complex_single.h"

/**
 * CUNCSD computes the CS decomposition of an M-by-M partitioned
 * unitary matrix X:
 *
 *                                 [  I  0  0 |  0  0  0 ]
 *                                 [  0  C  0 |  0 -S  0 ]
 *     [ X11 | X12 ]   [ U1 |    ] [  0  0  0 |  0  0 -I ] [ V1 |    ]**H
 * X = [-----------] = [---------] [---------------------] [---------]   .
 *     [ X21 | X22 ]   [    | U2 ] [  0  0  0 |  I  0  0 ] [    | V2 ]
 *                                 [  0  S  0 |  0  C  0 ]
 *                                 [  0  0  I |  0  0  0 ]
 *
 * X11 is P-by-Q. The unitary matrices U1, U2, V1, and V2 are P-by-P,
 * (M-P)-by-(M-P), Q-by-Q, and (M-Q)-by-(M-Q), respectively. C and S are
 * R-by-R nonnegative diagonal matrices satisfying C^2 + S^2 = I, in
 * which R = MIN(P,M-P,Q,M-Q).
 *
 * @param[in] jobu1
 *          = 'Y': U1 is computed;
 *          otherwise: U1 is not computed.
 *
 * @param[in] jobu2
 *          = 'Y': U2 is computed;
 *          otherwise: U2 is not computed.
 *
 * @param[in] jobv1t
 *          = 'Y': V1T is computed;
 *          otherwise: V1T is not computed.
 *
 * @param[in] jobv2t
 *          = 'Y': V2T is computed;
 *          otherwise: V2T is not computed.
 *
 * @param[in] trans
 *          = 'T': X, U1, U2, V1T, and V2T are stored in row-major order;
 *          otherwise: X, U1, U2, V1T, and V2T are stored in column-major order.
 *
 * @param[in] signs
 *          = 'O': The lower-left block is made nonpositive ("other" convention);
 *          otherwise: The upper-right block is made nonpositive ("default" convention).
 *
 * @param[in] m
 *          The number of rows and columns in X.
 *
 * @param[in] p
 *          The number of rows in X11 and X12. 0 <= p <= m.
 *
 * @param[in] q
 *          The number of columns in X11 and X21. 0 <= q <= m.
 *
 * @param[in,out] X11
 *          Complex*16 array, dimension (ldx11, q).
 *          On entry, part of the unitary matrix whose CSD is desired.
 *
 * @param[in] ldx11
 *          The leading dimension of X11.
 *
 * @param[in,out] X12
 *          Complex*16 array, dimension (ldx12, m-q).
 *          On entry, part of the unitary matrix whose CSD is desired.
 *
 * @param[in] ldx12
 *          The leading dimension of X12.
 *
 * @param[in,out] X21
 *          Complex*16 array, dimension (ldx21, q).
 *          On entry, part of the unitary matrix whose CSD is desired.
 *
 * @param[in] ldx21
 *          The leading dimension of X21.
 *
 * @param[in,out] X22
 *          Complex*16 array, dimension (ldx22, m-q).
 *          On entry, part of the unitary matrix whose CSD is desired.
 *
 * @param[in] ldx22
 *          The leading dimension of X22.
 *
 * @param[out] theta
 *          Single precision array, dimension (r), where r = min(p, m-p, q, m-q).
 *          C = DIAG( COS(THETA(1)), ... , COS(THETA(R)) ) and
 *          S = DIAG( SIN(THETA(1)), ... , SIN(THETA(R)) ).
 *
 * @param[out] U1
 *          If jobu1 = 'Y', U1 contains the P-by-P unitary matrix U1.
 *
 * @param[in] ldu1
 *          The leading dimension of U1.
 *
 * @param[out] U2
 *          If jobu2 = 'Y', U2 contains the (M-P)-by-(M-P) unitary matrix U2.
 *
 * @param[in] ldu2
 *          The leading dimension of U2.
 *
 * @param[out] V1T
 *          If jobv1t = 'Y', V1T contains the Q-by-Q unitary matrix V1**H.
 *
 * @param[in] ldv1t
 *          The leading dimension of V1T.
 *
 * @param[out] V2T
 *          If jobv2t = 'Y', V2T contains the (M-Q)-by-(M-Q) unitary matrix V2**H.
 *
 * @param[in] ldv2t
 *          The leading dimension of V2T.
 *
 * @param[out] work
 *          Complex*16 array, dimension (max(1, lwork)).
 *          On exit, if info = 0, work[0] returns the optimal lwork.
 *
 * @param[in] lwork
 *          The dimension of the array work.
 *          If lwork = -1, a workspace query is assumed.
 *
 * @param[out] rwork
 *          Single precision array, dimension (max(1, lrwork)).
 *          On exit, if info = 0, rwork[0] returns the optimal lrwork.
 *
 * @param[in] lrwork
 *          The dimension of the array rwork.
 *          If lrwork = -1, a workspace query is assumed.
 *
 * @param[out] iwork
 *          Integer array, dimension (m - min(p, m-p, q, m-q)).
 *
 * @param[out] info
 *                         - = 0: successful exit.
 *                         - < 0: if info = -i, the i-th argument had an illegal value.
 *                         - > 0: CBBCSD did not converge.
 */
void cuncsd(
    const char* jobu1,
    const char* jobu2,
    const char* jobv1t,
    const char* jobv2t,
    const char* trans,
    const char* signs,
    const int m,
    const int p,
    const int q,
    c64* X11,
    const int ldx11,
    c64* X12,
    const int ldx12,
    c64* X21,
    const int ldx21,
    c64* X22,
    const int ldx22,
    f32* restrict theta,
    c64* restrict U1,
    const int ldu1,
    c64* restrict U2,
    const int ldu2,
    c64* restrict V1T,
    const int ldv1t,
    c64* restrict V2T,
    const int ldv2t,
    c64* restrict work,
    const int lwork,
    f32* restrict rwork,
    const int lrwork,
    int* restrict iwork,
    int* info)
{
    const c64 one = CMPLXF(1.0f, 0.0f);
    const c64 zero = CMPLXF(0.0f, 0.0f);

    int childinfo, i;
    int ib11d = 0, ib11e = 0, ib12d = 0, ib12e = 0;
    int ib21d = 0, ib21e = 0, ib22d = 0, ib22e = 0, ibbcsd = 0, iorbdb = 0;
    int iorglq = 0, iorgqr = 0, iphi, itaup1 = 0, itaup2 = 0, itauq1 = 0, itauq2 = 0;
    int j;
    int lbbcsdwork = 0, lbbcsdworkmin, lbbcsdworkopt;
    int lorbdbwork = 0, lorbdbworkopt;
    int lorglqwork = 0, lorglqworkmin, lorglqworkopt;
    int lorgqrwork = 0, lorgqrworkmin, lorgqrworkopt;
    int lworkmin, lworkopt;
    int lrworkmin, lrworkopt;
    int colmajor, defaultsigns, lquery, lrquery, wantu1, wantu2, wantv1t, wantv2t;

    *info = 0;
    wantu1 = (jobu1[0] == 'Y' || jobu1[0] == 'y');
    wantu2 = (jobu2[0] == 'Y' || jobu2[0] == 'y');
    wantv1t = (jobv1t[0] == 'Y' || jobv1t[0] == 'y');
    wantv2t = (jobv2t[0] == 'Y' || jobv2t[0] == 'y');
    colmajor = !(trans[0] == 'T' || trans[0] == 't');
    defaultsigns = !(signs[0] == 'O' || signs[0] == 'o');
    lquery = (lwork == -1);
    lrquery = (lrwork == -1);

    if (m < 0) {
        *info = -7;
    } else if (p < 0 || p > m) {
        *info = -8;
    } else if (q < 0 || q > m) {
        *info = -9;
    } else if (colmajor && ldx11 < (1 > p ? 1 : p)) {
        *info = -11;
    } else if (!colmajor && ldx11 < (1 > q ? 1 : q)) {
        *info = -11;
    } else if (colmajor && ldx12 < (1 > p ? 1 : p)) {
        *info = -13;
    } else if (!colmajor && ldx12 < (1 > (m - q) ? 1 : (m - q))) {
        *info = -13;
    } else if (colmajor && ldx21 < (1 > (m - p) ? 1 : (m - p))) {
        *info = -15;
    } else if (!colmajor && ldx21 < (1 > q ? 1 : q)) {
        *info = -15;
    } else if (colmajor && ldx22 < (1 > (m - p) ? 1 : (m - p))) {
        *info = -17;
    } else if (!colmajor && ldx22 < (1 > (m - q) ? 1 : (m - q))) {
        *info = -17;
    } else if (wantu1 && ldu1 < p) {
        *info = -20;
    } else if (wantu2 && ldu2 < m - p) {
        *info = -22;
    } else if (wantv1t && ldv1t < q) {
        *info = -24;
    } else if (wantv2t && ldv2t < m - q) {
        *info = -26;
    }

    /* Work with transpose if convenient */
    int minpmp = (p < m - p) ? p : (m - p);
    int minqmq = (q < m - q) ? q : (m - q);
    if (*info == 0 && minpmp < minqmq) {
        const char* transt = colmajor ? "T" : "N";
        const char* signst = defaultsigns ? "O" : "D";
        cuncsd(jobv1t, jobv2t, jobu1, jobu2, transt, signst, m,
               q, p, X11, ldx11, X21, ldx21, X12, ldx12, X22, ldx22,
               theta, V1T, ldv1t, V2T, ldv2t, U1, ldu1, U2, ldu2,
               work, lwork, rwork, lrwork, iwork, info);
        return;
    }

    /* Work with permutation [ 0 I; I 0 ] * X * [ 0 I; I 0 ] if convenient */
    if (*info == 0 && m - q < q) {
        const char* signst = defaultsigns ? "O" : "D";
        cuncsd(jobu2, jobu1, jobv2t, jobv1t, trans, signst, m,
               m - p, m - q, X22, ldx22, X21, ldx21, X12, ldx12, X11, ldx11,
               theta, U2, ldu2, U1, ldu1, V2T, ldv2t, V1T, ldv1t,
               work, lwork, rwork, lrwork, iwork, info);
        return;
    }

    /* Compute workspace */
    if (*info == 0) {

        /* Real workspace */
        iphi = 2;
        ib11d = iphi + (1 > (q - 1) ? 1 : (q - 1));
        ib11e = ib11d + (1 > q ? 1 : q);
        ib12d = ib11e + (1 > (q - 1) ? 1 : (q - 1));
        ib12e = ib12d + (1 > q ? 1 : q);
        ib21d = ib12e + (1 > (q - 1) ? 1 : (q - 1));
        ib21e = ib21d + (1 > q ? 1 : q);
        ib22d = ib21e + (1 > (q - 1) ? 1 : (q - 1));
        ib22e = ib22d + (1 > q ? 1 : q);
        ibbcsd = ib22e + (1 > (q - 1) ? 1 : (q - 1));

        cbbcsd(jobu1, jobu2, jobv1t, jobv2t, trans, m, p, q,
               NULL, NULL, NULL, ldu1, NULL, ldu2, NULL, ldv1t,
               NULL, ldv2t, NULL, NULL, NULL, NULL, NULL,
               NULL, NULL, NULL, rwork, -1, &childinfo);
        lbbcsdworkopt = (int)rwork[0];
        lbbcsdworkmin = lbbcsdworkopt;
        lrworkopt = ibbcsd + lbbcsdworkopt - 1;
        lrworkmin = ibbcsd + lbbcsdworkmin - 1;
        rwork[0] = (f32)lrworkopt;

        /* Complex workspace */
        itaup1 = 2;
        itaup2 = itaup1 + (1 > p ? 1 : p);
        itauq1 = itaup2 + (1 > (m - p) ? 1 : (m - p));
        itauq2 = itauq1 + (1 > q ? 1 : q);
        iorgqr = itauq2 + (1 > (m - q) ? 1 : (m - q));

        int ld_temp = (1 > (m - q)) ? 1 : (m - q);
        cungqr(m - q, m - q, m - q, NULL, ld_temp, NULL, work, -1, &childinfo);
        lorgqrworkopt = (int)crealf(work[0]);
        lorgqrworkmin = (1 > (m - q)) ? 1 : (m - q);

        iorglq = itauq2 + (1 > (m - q) ? 1 : (m - q));
        cunglq(m - q, m - q, m - q, NULL, ld_temp, NULL, work, -1, &childinfo);
        lorglqworkopt = (int)crealf(work[0]);
        lorglqworkmin = (1 > (m - q)) ? 1 : (m - q);

        iorbdb = itauq2 + (1 > (m - q) ? 1 : (m - q));
        cunbdb(trans, signs, m, p, q, X11, ldx11, X12, ldx12,
               X21, ldx21, X22, ldx22, theta, NULL, NULL, NULL,
               NULL, NULL, work, -1, &childinfo);
        lorbdbworkopt = (int)crealf(work[0]);

        lworkopt = iorgqr + lorgqrworkopt;
        if (iorglq + lorglqworkopt > lworkopt) lworkopt = iorglq + lorglqworkopt;
        if (iorbdb + lorbdbworkopt > lworkopt) lworkopt = iorbdb + lorbdbworkopt;
        lworkopt = lworkopt - 1;

        lworkmin = iorgqr + lorgqrworkmin;
        if (iorglq + lorglqworkmin > lworkmin) lworkmin = iorglq + lorglqworkmin;
        if (iorbdb + lorbdbworkopt > lworkmin) lworkmin = iorbdb + lorbdbworkopt;
        lworkmin = lworkmin - 1;

        work[0] = CMPLXF((f32)(lworkopt > lworkmin ? lworkopt : lworkmin), 0.0f);

        if (lwork < lworkmin && !(lquery || lrquery)) {
            *info = -22;
        } else if (lrwork < lrworkmin && !(lquery || lrquery)) {
            *info = -24;
        } else {
            lorgqrwork = lwork - iorgqr + 1;
            lorglqwork = lwork - iorglq + 1;
            lorbdbwork = lwork - iorbdb + 1;
            lbbcsdwork = lrwork - ibbcsd + 1;
        }
    }

    /* Abort if any illegal arguments */
    if (*info != 0) {
        xerbla("CUNCSD", -(*info));
        return;
    } else if (lquery || lrquery) {
        return;
    }

    /* Transform to bidiagonal block form */
    cunbdb(trans, signs, m, p, q, X11, ldx11, X12, ldx12, X21,
           ldx21, X22, ldx22, theta, &rwork[iphi - 1], &work[itaup1 - 1],
           &work[itaup2 - 1], &work[itauq1 - 1], &work[itauq2 - 1],
           &work[iorbdb - 1], lorbdbwork, &childinfo);

    /* Accumulate Householder reflectors */
    if (colmajor) {
        if (wantu1 && p > 0) {
            clacpy("L", p, q, X11, ldx11, U1, ldu1);
            cungqr(p, p, q, U1, ldu1, &work[itaup1 - 1],
                   &work[iorgqr - 1], lorgqrwork, info);
        }
        if (wantu2 && m - p > 0) {
            clacpy("L", m - p, q, X21, ldx21, U2, ldu2);
            cungqr(m - p, m - p, q, U2, ldu2, &work[itaup2 - 1],
                   &work[iorgqr - 1], lorgqrwork, info);
        }
        if (wantv1t && q > 0) {
            clacpy("U", q - 1, q - 1, &X11[1 * ldx11], ldx11,
                   &V1T[1 + 1 * ldv1t], ldv1t);
            V1T[0] = one;
            for (j = 1; j < q; j++) {
                V1T[j * ldv1t] = zero;
                V1T[j] = zero;
            }
            cunglq(q - 1, q - 1, q - 1, &V1T[1 + 1 * ldv1t], ldv1t,
                   &work[itauq1 - 1], &work[iorglq - 1], lorglqwork, info);
        }
        if (wantv2t && m - q > 0) {
            clacpy("U", p, m - q, X12, ldx12, V2T, ldv2t);
            if (m - p > q) {
                clacpy("U", m - p - q, m - p - q, &X22[q + p * ldx22], ldx22,
                       &V2T[p + p * ldv2t], ldv2t);
            }
            if (m > q) {
                cunglq(m - q, m - q, m - q, V2T, ldv2t, &work[itauq2 - 1],
                       &work[iorglq - 1], lorglqwork, info);
            }
        }
    } else {
        if (wantu1 && p > 0) {
            clacpy("U", q, p, X11, ldx11, U1, ldu1);
            cunglq(p, p, q, U1, ldu1, &work[itaup1 - 1],
                   &work[iorglq - 1], lorglqwork, info);
        }
        if (wantu2 && m - p > 0) {
            clacpy("U", q, m - p, X21, ldx21, U2, ldu2);
            cunglq(m - p, m - p, q, U2, ldu2, &work[itaup2 - 1],
                   &work[iorglq - 1], lorglqwork, info);
        }
        if (wantv1t && q > 0) {
            clacpy("L", q - 1, q - 1, &X11[1], ldx11, &V1T[1 + 1 * ldv1t], ldv1t);
            V1T[0] = one;
            for (j = 1; j < q; j++) {
                V1T[j * ldv1t] = zero;
                V1T[j] = zero;
            }
            cungqr(q - 1, q - 1, q - 1, &V1T[1 + 1 * ldv1t], ldv1t,
                   &work[itauq1 - 1], &work[iorgqr - 1], lorgqrwork, info);
        }
        if (wantv2t && m - q > 0) {
            clacpy("L", m - q, p, X12, ldx12, V2T, ldv2t);
            if (m > p + q) {
                clacpy("L", m - p - q, m - p - q, &X22[p + q * ldx22],
                       ldx22, &V2T[p + p * ldv2t], ldv2t);
            }
            cungqr(m - q, m - q, m - q, V2T, ldv2t, &work[itauq2 - 1],
                   &work[iorgqr - 1], lorgqrwork, info);
        }
    }

    /* Compute the CSD of the matrix in bidiagonal-block form */
    cbbcsd(jobu1, jobu2, jobv1t, jobv2t, trans, m, p, q, theta,
           &rwork[iphi - 1], U1, ldu1, U2, ldu2, V1T, ldv1t, V2T, ldv2t,
           &rwork[ib11d - 1], &rwork[ib11e - 1], &rwork[ib12d - 1],
           &rwork[ib12e - 1], &rwork[ib21d - 1], &rwork[ib21e - 1],
           &rwork[ib22d - 1], &rwork[ib22e - 1], &rwork[ibbcsd - 1],
           lbbcsdwork, info);

    /* Permute rows and columns to place identity submatrices in
       preferred positions */
    if (q > 0 && wantu2) {
        for (i = 0; i < q; i++) {
            iwork[i] = m - p - q + i;
        }
        for (i = q; i < m - p; i++) {
            iwork[i] = i - q;
        }
        if (colmajor) {
            clapmt(0, m - p, m - p, U2, ldu2, iwork);
        } else {
            clapmr(0, m - p, m - p, U2, ldu2, iwork);
        }
    }
    if (m > 0 && wantv2t) {
        for (i = 0; i < p; i++) {
            iwork[i] = m - p - q + i;
        }
        for (i = p; i < m - q; i++) {
            iwork[i] = i - p;
        }
        if (!colmajor) {
            clapmt(0, m - q, m - q, V2T, ldv2t, iwork);
        } else {
            clapmr(0, m - q, m - q, V2T, ldv2t, iwork);
        }
    }
}
