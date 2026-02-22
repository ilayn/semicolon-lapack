/**
 * @file dorcsd2by1.c
 * @brief DORCSD2BY1 computes the CS decomposition of an M-by-Q matrix with orthonormal columns partitioned into a 2-by-1 block structure.
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "semicolon_lapack_double.h"

/**
 * DORCSD2BY1 computes the CS decomposition of an M-by-Q matrix X with
 * orthonormal columns that has been partitioned into a 2-by-1 block
 * structure:
 *
 *                                [  I1 0  0 ]
 *                                [  0  C  0 ]
 *          [ X11 ]   [ U1 |    ] [  0  0  0 ]
 *      X = [-----] = [---------] [----------] V1**T .
 *          [ X21 ]   [    | U2 ] [  0  0  0 ]
 *                                [  0  S  0 ]
 *                                [  0  0  I2]
 *
 * X11 is P-by-Q. The orthogonal matrices U1, U2, and V1 are P-by-P,
 * (M-P)-by-(M-P), and Q-by-Q, respectively. C and S are R-by-R
 * nonnegative diagonal matrices satisfying C^2 + S^2 = I, in which
 * R = MIN(P,M-P,Q,M-Q). I1 is a K1-by-K1 identity matrix and I2 is a
 * K2-by-K2 identity matrix, where K1 = MAX(Q+P-M,0), K2 = MAX(Q-P,0).
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
 * @param[in] m
 *          The number of rows in X.
 *
 * @param[in] p
 *          The number of rows in X11. 0 <= p <= m.
 *
 * @param[in] q
 *          The number of columns in X11 and X21. 0 <= q <= m.
 *
 * @param[in,out] X11
 *          Double precision array, dimension (ldx11, q).
 *          On entry, part of the orthogonal matrix whose CSD is desired.
 *
 * @param[in] ldx11
 *          The leading dimension of X11. ldx11 >= max(1, p).
 *
 * @param[in,out] X21
 *          Double precision array, dimension (ldx21, q).
 *          On entry, part of the orthogonal matrix whose CSD is desired.
 *
 * @param[in] ldx21
 *          The leading dimension of X21. ldx21 >= max(1, m-p).
 *
 * @param[out] theta
 *          Double precision array, dimension (r), where r = min(p, m-p, q, m-q).
 *          C = DIAG( COS(THETA(1)), ... , COS(THETA(R)) ) and
 *          S = DIAG( SIN(THETA(1)), ... , SIN(THETA(R)) ).
 *
 * @param[out] U1
 *          Double precision array, dimension (ldu1, p).
 *          If jobu1 = 'Y', U1 contains the P-by-P orthogonal matrix U1.
 *
 * @param[in] ldu1
 *          The leading dimension of U1. If jobu1 = 'Y', ldu1 >= max(1, p).
 *
 * @param[out] U2
 *          Double precision array, dimension (ldu2, m-p).
 *          If jobu2 = 'Y', U2 contains the (M-P)-by-(M-P) orthogonal matrix U2.
 *
 * @param[in] ldu2
 *          The leading dimension of U2. If jobu2 = 'Y', ldu2 >= max(1, m-p).
 *
 * @param[out] V1T
 *          Double precision array, dimension (ldv1t, q).
 *          If jobv1t = 'Y', V1T contains the Q-by-Q orthogonal matrix V1**T.
 *
 * @param[in] ldv1t
 *          The leading dimension of V1T. If jobv1t = 'Y', ldv1t >= max(1, q).
 *
 * @param[out] work
 *          Double precision array, dimension (max(1, lwork)).
 *          On exit, if info = 0, work[0] returns the optimal lwork.
 *
 * @param[in] lwork
 *          The dimension of the array work.
 *          If lwork = -1, a workspace query is assumed.
 *
 * @param[out] iwork
 *          Integer array, dimension (m - min(p, m-p, q, m-q)).
 *
 * @param[out] info
 *                         - = 0: successful exit.
 *                         - < 0: if info = -i, the i-th argument had an illegal value.
 *                         - > 0: DBBCSD did not converge.
 */
void dorcsd2by1(
    const char* jobu1,
    const char* jobu2,
    const char* jobv1t,
    const INT m,
    const INT p,
    const INT q,
    f64* restrict X11,
    const INT ldx11,
    f64* restrict X21,
    const INT ldx21,
    f64* restrict theta,
    f64* restrict U1,
    const INT ldu1,
    f64* restrict U2,
    const INT ldu2,
    f64* restrict V1T,
    const INT ldv1t,
    f64* restrict work,
    const INT lwork,
    INT* restrict iwork,
    INT* info)
{
    const f64 one = 1.0;
    const f64 zero = 0.0;

    INT childinfo, i, ib11d, ib11e, ib12d, ib12e;
    INT ib21d, ib21e, ib22d, ib22e, ibbcsd, iorbdb;
    INT iorglq, iorgqr, iphi, itaup1, itaup2, itauq1;
    INT j, lbbcsd, lorbdb, lorglq, lorglqmin;
    INT lorglqopt, lorgqr, lorgqrmin, lorgqropt;
    INT lworkmin, lworkopt, r;
    INT lquery, wantu1, wantu2, wantv1t;
    f64 dum1[1];
    f64 dum2[1];

    *info = 0;
    wantu1 = (jobu1[0] == 'Y' || jobu1[0] == 'y');
    wantu2 = (jobu2[0] == 'Y' || jobu2[0] == 'y');
    wantv1t = (jobv1t[0] == 'Y' || jobv1t[0] == 'y');
    lquery = (lwork == -1);

    if (m < 0) {
        *info = -4;
    } else if (p < 0 || p > m) {
        *info = -5;
    } else if (q < 0 || q > m) {
        *info = -6;
    } else if (ldx11 < (1 > p ? 1 : p)) {
        *info = -8;
    } else if (ldx21 < (1 > (m - p) ? 1 : (m - p))) {
        *info = -10;
    } else if (wantu1 && ldu1 < (1 > p ? 1 : p)) {
        *info = -13;
    } else if (wantu2 && ldu2 < (1 > (m - p) ? 1 : (m - p))) {
        *info = -15;
    } else if (wantv1t && ldv1t < (1 > q ? 1 : q)) {
        *info = -17;
    }

    r = p;
    if (m - p < r) r = m - p;
    if (q < r) r = q;
    if (m - q < r) r = m - q;

    if (*info == 0) {
        iphi = 1;
        ib11d = iphi + (1 > (r - 1) ? 1 : (r - 1));
        ib11e = ib11d + (1 > r ? 1 : r);
        ib12d = ib11e + (1 > (r - 1) ? 1 : (r - 1));
        ib12e = ib12d + (1 > r ? 1 : r);
        ib21d = ib12e + (1 > (r - 1) ? 1 : (r - 1));
        ib21e = ib21d + (1 > r ? 1 : r);
        ib22d = ib21e + (1 > (r - 1) ? 1 : (r - 1));
        ib22e = ib22d + (1 > r ? 1 : r);
        ibbcsd = ib22e + (1 > (r - 1) ? 1 : (r - 1));
        itaup1 = iphi + (1 > (r - 1) ? 1 : (r - 1));
        itaup2 = itaup1 + (1 > p ? 1 : p);
        itauq1 = itaup2 + (1 > (m - p) ? 1 : (m - p));
        iorbdb = itauq1 + (1 > q ? 1 : q);
        iorgqr = itauq1 + (1 > q ? 1 : q);
        iorglq = itauq1 + (1 > q ? 1 : q);
        lorgqrmin = 1;
        lorgqropt = 1;
        lorglqmin = 1;
        lorglqopt = 1;

        if (r == q) {
            dorbdb1(m, p, q, X11, ldx11, X21, ldx21, theta,
                    NULL, NULL, NULL, NULL, work, -1, &childinfo);
            lorbdb = (INT)work[0];
            if (wantu1 && p > 0) {
                dorgqr(p, p, q, U1, ldu1, dum1, work, -1, &childinfo);
                lorgqrmin = lorgqrmin > p ? lorgqrmin : p;
                lorgqropt = lorgqropt > (INT)work[0] ? lorgqropt : (INT)work[0];
            }
            if (wantu2 && m - p > 0) {
                dorgqr(m - p, m - p, q, U2, ldu2, dum1, work, -1, &childinfo);
                lorgqrmin = lorgqrmin > (m - p) ? lorgqrmin : (m - p);
                lorgqropt = lorgqropt > (INT)work[0] ? lorgqropt : (INT)work[0];
            }
            if (wantv1t && q > 0) {
                dorglq(q - 1, q - 1, q - 1, V1T, ldv1t, dum1, work, -1, &childinfo);
                lorglqmin = lorglqmin > (q - 1) ? lorglqmin : (q - 1);
                lorglqopt = lorglqopt > (INT)work[0] ? lorglqopt : (INT)work[0];
            }
            dbbcsd(jobu1, jobu2, jobv1t, "N", "N", m, p, q, theta,
                   NULL, U1, ldu1, U2, ldu2, V1T, ldv1t,
                   dum2, 1, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
                   work, -1, &childinfo);
            lbbcsd = (INT)work[0];
        } else if (r == p) {
            dorbdb2(m, p, q, X11, ldx11, X21, ldx21, theta,
                    NULL, NULL, NULL, NULL, work, -1, &childinfo);
            lorbdb = (INT)work[0];
            if (wantu1 && p > 0) {
                dorgqr(p - 1, p - 1, p - 1, &U1[1 + 1 * ldu1], ldu1, dum1, work, -1, &childinfo);
                lorgqrmin = lorgqrmin > (p - 1) ? lorgqrmin : (p - 1);
                lorgqropt = lorgqropt > (INT)work[0] ? lorgqropt : (INT)work[0];
            }
            if (wantu2 && m - p > 0) {
                dorgqr(m - p, m - p, q, U2, ldu2, dum1, work, -1, &childinfo);
                lorgqrmin = lorgqrmin > (m - p) ? lorgqrmin : (m - p);
                lorgqropt = lorgqropt > (INT)work[0] ? lorgqropt : (INT)work[0];
            }
            if (wantv1t && q > 0) {
                dorglq(q, q, r, V1T, ldv1t, dum1, work, -1, &childinfo);
                lorglqmin = lorglqmin > q ? lorglqmin : q;
                lorglqopt = lorglqopt > (INT)work[0] ? lorglqopt : (INT)work[0];
            }
            dbbcsd(jobv1t, "N", jobu1, jobu2, "T", m, q, p, theta,
                   NULL, V1T, ldv1t, dum2, 1, U1, ldu1, U2, ldu2,
                   NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
                   work, -1, &childinfo);
            lbbcsd = (INT)work[0];
        } else if (r == m - p) {
            dorbdb3(m, p, q, X11, ldx11, X21, ldx21, theta,
                    NULL, NULL, NULL, NULL, work, -1, &childinfo);
            lorbdb = (INT)work[0];
            if (wantu1 && p > 0) {
                dorgqr(p, p, q, U1, ldu1, dum1, work, -1, &childinfo);
                lorgqrmin = lorgqrmin > p ? lorgqrmin : p;
                lorgqropt = lorgqropt > (INT)work[0] ? lorgqropt : (INT)work[0];
            }
            if (wantu2 && m - p > 0) {
                dorgqr(m - p - 1, m - p - 1, m - p - 1, &U2[1 + 1 * ldu2], ldu2,
                       dum1, work, -1, &childinfo);
                lorgqrmin = lorgqrmin > (m - p - 1) ? lorgqrmin : (m - p - 1);
                lorgqropt = lorgqropt > (INT)work[0] ? lorgqropt : (INT)work[0];
            }
            if (wantv1t && q > 0) {
                dorglq(q, q, r, V1T, ldv1t, dum1, work, -1, &childinfo);
                lorglqmin = lorglqmin > q ? lorglqmin : q;
                lorglqopt = lorglqopt > (INT)work[0] ? lorglqopt : (INT)work[0];
            }
            dbbcsd("N", jobv1t, jobu2, jobu1, "T", m, m - q, m - p,
                   theta, NULL, dum2, 1, V1T, ldv1t, U2, ldu2, U1, ldu1,
                   NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
                   work, -1, &childinfo);
            lbbcsd = (INT)work[0];
        } else {
            dorbdb4(m, p, q, X11, ldx11, X21, ldx21, theta,
                    NULL, NULL, NULL, NULL, NULL, work, -1, &childinfo);
            lorbdb = m + (INT)work[0];
            if (wantu1 && p > 0) {
                dorgqr(p, p, m - q, U1, ldu1, dum1, work, -1, &childinfo);
                lorgqrmin = lorgqrmin > p ? lorgqrmin : p;
                lorgqropt = lorgqropt > (INT)work[0] ? lorgqropt : (INT)work[0];
            }
            if (wantu2 && m - p > 0) {
                dorgqr(m - p, m - p, m - q, U2, ldu2, dum1, work, -1, &childinfo);
                lorgqrmin = lorgqrmin > (m - p) ? lorgqrmin : (m - p);
                lorgqropt = lorgqropt > (INT)work[0] ? lorgqropt : (INT)work[0];
            }
            if (wantv1t && q > 0) {
                dorglq(q, q, q, V1T, ldv1t, dum1, work, -1, &childinfo);
                lorglqmin = lorglqmin > q ? lorglqmin : q;
                lorglqopt = lorglqopt > (INT)work[0] ? lorglqopt : (INT)work[0];
            }
            dbbcsd(jobu2, jobu1, "N", jobv1t, "N", m, m - p, m - q,
                   theta, NULL, U2, ldu2, U1, ldu1, dum2, 1, V1T, ldv1t,
                   NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
                   work, -1, &childinfo);
            lbbcsd = (INT)work[0];
        }

        lworkmin = iorbdb + lorbdb - 1;
        if (iorgqr + lorgqrmin - 1 > lworkmin) lworkmin = iorgqr + lorgqrmin - 1;
        if (iorglq + lorglqmin - 1 > lworkmin) lworkmin = iorglq + lorglqmin - 1;
        if (ibbcsd + lbbcsd - 1 > lworkmin) lworkmin = ibbcsd + lbbcsd - 1;

        lworkopt = iorbdb + lorbdb - 1;
        if (iorgqr + lorgqropt - 1 > lworkopt) lworkopt = iorgqr + lorgqropt - 1;
        if (iorglq + lorglqopt - 1 > lworkopt) lworkopt = iorglq + lorglqopt - 1;
        if (ibbcsd + lbbcsd - 1 > lworkopt) lworkopt = ibbcsd + lbbcsd - 1;

        work[0] = (f64)lworkopt;
        if (lwork < lworkmin && !lquery) {
            *info = -19;
        }
    }

    if (*info != 0) {
        xerbla("DORCSD2BY1", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    lorgqr = lwork - iorgqr + 1;
    lorglq = lwork - iorglq + 1;

    if (r == q) {

        dorbdb1(m, p, q, X11, ldx11, X21, ldx21, theta,
                &work[iphi - 1], &work[itaup1 - 1], &work[itaup2 - 1],
                &work[itauq1 - 1], &work[iorbdb - 1], lorbdb, &childinfo);

        if (wantu1 && p > 0) {
            dlacpy("L", p, q, X11, ldx11, U1, ldu1);
            dorgqr(p, p, q, U1, ldu1, &work[itaup1 - 1],
                   &work[iorgqr - 1], lorgqr, &childinfo);
        }
        if (wantu2 && m - p > 0) {
            dlacpy("L", m - p, q, X21, ldx21, U2, ldu2);
            dorgqr(m - p, m - p, q, U2, ldu2, &work[itaup2 - 1],
                   &work[iorgqr - 1], lorgqr, &childinfo);
        }
        if (wantv1t && q > 0) {
            V1T[0] = one;
            for (j = 1; j < q; j++) {
                V1T[j * ldv1t] = zero;
                V1T[j] = zero;
            }
            dlacpy("U", q - 1, q - 1, &X21[1 * ldx21], ldx21, &V1T[1 + 1 * ldv1t], ldv1t);
            dorglq(q - 1, q - 1, q - 1, &V1T[1 + 1 * ldv1t], ldv1t,
                   &work[itauq1 - 1], &work[iorglq - 1], lorglq, &childinfo);
        }

        dbbcsd(jobu1, jobu2, jobv1t, "N", "N", m, p, q, theta,
               &work[iphi - 1], U1, ldu1, U2, ldu2, V1T, ldv1t,
               dum2, 1, &work[ib11d - 1], &work[ib11e - 1],
               &work[ib12d - 1], &work[ib12e - 1], &work[ib21d - 1],
               &work[ib21e - 1], &work[ib22d - 1], &work[ib22e - 1],
               &work[ibbcsd - 1], lbbcsd, &childinfo);

        if (q > 0 && wantu2) {
            for (i = 0; i < q; i++) {
                iwork[i] = m - p - q + i;
            }
            for (i = q; i < m - p; i++) {
                iwork[i] = i - q;
            }
            dlapmt(0, m - p, m - p, U2, ldu2, iwork);
        }

    } else if (r == p) {

        dorbdb2(m, p, q, X11, ldx11, X21, ldx21, theta,
                &work[iphi - 1], &work[itaup1 - 1], &work[itaup2 - 1],
                &work[itauq1 - 1], &work[iorbdb - 1], lorbdb, &childinfo);

        if (wantu1 && p > 0) {
            U1[0] = one;
            for (j = 1; j < p; j++) {
                U1[j * ldu1] = zero;
                U1[j] = zero;
            }
            dlacpy("L", p - 1, p - 1, &X11[1], ldx11, &U1[1 + 1 * ldu1], ldu1);
            dorgqr(p - 1, p - 1, p - 1, &U1[1 + 1 * ldu1], ldu1, &work[itaup1 - 1],
                   &work[iorgqr - 1], lorgqr, &childinfo);
        }
        if (wantu2 && m - p > 0) {
            dlacpy("L", m - p, q, X21, ldx21, U2, ldu2);
            dorgqr(m - p, m - p, q, U2, ldu2, &work[itaup2 - 1],
                   &work[iorgqr - 1], lorgqr, &childinfo);
        }
        if (wantv1t && q > 0) {
            dlacpy("U", p, q, X11, ldx11, V1T, ldv1t);
            dorglq(q, q, r, V1T, ldv1t, &work[itauq1 - 1],
                   &work[iorglq - 1], lorglq, &childinfo);
        }

        dbbcsd(jobv1t, "N", jobu1, jobu2, "T", m, q, p, theta,
               &work[iphi - 1], V1T, ldv1t, dum1, 1, U1, ldu1, U2, ldu2,
               &work[ib11d - 1], &work[ib11e - 1], &work[ib12d - 1],
               &work[ib12e - 1], &work[ib21d - 1], &work[ib21e - 1],
               &work[ib22d - 1], &work[ib22e - 1], &work[ibbcsd - 1], lbbcsd,
               &childinfo);

        if (q > 0 && wantu2) {
            for (i = 0; i < q; i++) {
                iwork[i] = m - p - q + i;
            }
            for (i = q; i < m - p; i++) {
                iwork[i] = i - q;
            }
            dlapmt(0, m - p, m - p, U2, ldu2, iwork);
        }

    } else if (r == m - p) {

        dorbdb3(m, p, q, X11, ldx11, X21, ldx21, theta,
                &work[iphi - 1], &work[itaup1 - 1], &work[itaup2 - 1],
                &work[itauq1 - 1], &work[iorbdb - 1], lorbdb, &childinfo);

        if (wantu1 && p > 0) {
            dlacpy("L", p, q, X11, ldx11, U1, ldu1);
            dorgqr(p, p, q, U1, ldu1, &work[itaup1 - 1],
                   &work[iorgqr - 1], lorgqr, &childinfo);
        }
        if (wantu2 && m - p > 0) {
            U2[0] = one;
            for (j = 1; j < m - p; j++) {
                U2[j * ldu2] = zero;
                U2[j] = zero;
            }
            dlacpy("L", m - p - 1, m - p - 1, &X21[1], ldx21, &U2[1 + 1 * ldu2], ldu2);
            dorgqr(m - p - 1, m - p - 1, m - p - 1, &U2[1 + 1 * ldu2], ldu2,
                   &work[itaup2 - 1], &work[iorgqr - 1], lorgqr, &childinfo);
        }
        if (wantv1t && q > 0) {
            dlacpy("U", m - p, q, X21, ldx21, V1T, ldv1t);
            dorglq(q, q, r, V1T, ldv1t, &work[itauq1 - 1],
                   &work[iorglq - 1], lorglq, &childinfo);
        }

        dbbcsd("N", jobv1t, jobu2, jobu1, "T", m, m - q, m - p, theta,
               &work[iphi - 1], dum1, 1, V1T, ldv1t, U2, ldu2, U1, ldu1,
               &work[ib11d - 1], &work[ib11e - 1], &work[ib12d - 1],
               &work[ib12e - 1], &work[ib21d - 1], &work[ib21e - 1],
               &work[ib22d - 1], &work[ib22e - 1], &work[ibbcsd - 1], lbbcsd,
               &childinfo);

        if (q > r) {
            for (i = 0; i < r; i++) {
                iwork[i] = q - r + i;
            }
            for (i = r; i < q; i++) {
                iwork[i] = i - r;
            }
            if (wantu1) {
                dlapmt(0, p, q, U1, ldu1, iwork);
            }
            if (wantv1t) {
                dlapmr(0, q, q, V1T, ldv1t, iwork);
            }
        }

    } else {

        dorbdb4(m, p, q, X11, ldx11, X21, ldx21, theta,
                &work[iphi - 1], &work[itaup1 - 1], &work[itaup2 - 1],
                &work[itauq1 - 1], &work[iorbdb - 1], &work[iorbdb + m - 1],
                lorbdb - m, &childinfo);

        if (wantu2 && m - p > 0) {
            cblas_dcopy(m - p, &work[iorbdb + p - 1], 1, U2, 1);
        }
        if (wantu1 && p > 0) {
            cblas_dcopy(p, &work[iorbdb - 1], 1, U1, 1);
            for (j = 1; j < p; j++) {
                U1[j * ldu1] = zero;
            }
            dlacpy("L", p - 1, m - q - 1, &X11[1], ldx11, &U1[1 + 1 * ldu1], ldu1);
            dorgqr(p, p, m - q, U1, ldu1, &work[itaup1 - 1],
                   &work[iorgqr - 1], lorgqr, &childinfo);
        }
        if (wantu2 && m - p > 0) {
            for (j = 1; j < m - p; j++) {
                U2[j * ldu2] = zero;
            }
            dlacpy("L", m - p - 1, m - q - 1, &X21[1], ldx21, &U2[1 + 1 * ldu2], ldu2);
            dorgqr(m - p, m - p, m - q, U2, ldu2, &work[itaup2 - 1],
                   &work[iorgqr - 1], lorgqr, &childinfo);
        }
        if (wantv1t && q > 0) {
            dlacpy("U", m - q, q, X21, ldx21, V1T, ldv1t);
            dlacpy("U", p - (m - q), q - (m - q), &X11[(m - q) + (m - q) * ldx11], ldx11,
                   &V1T[(m - q) + (m - q) * ldv1t], ldv1t);
            dlacpy("U", -p + q, q - p, &X21[(m - q) + p * ldx21], ldx21,
                   &V1T[p + p * ldv1t], ldv1t);
            dorglq(q, q, q, V1T, ldv1t, &work[itauq1 - 1],
                   &work[iorglq - 1], lorglq, &childinfo);
        }

        dbbcsd(jobu2, jobu1, "N", jobv1t, "N", m, m - p, m - q, theta,
               &work[iphi - 1], U2, ldu2, U1, ldu1, dum1, 1, V1T, ldv1t,
               &work[ib11d - 1], &work[ib11e - 1], &work[ib12d - 1],
               &work[ib12e - 1], &work[ib21d - 1], &work[ib21e - 1],
               &work[ib22d - 1], &work[ib22e - 1], &work[ibbcsd - 1], lbbcsd,
               &childinfo);

        if (p > r) {
            for (i = 0; i < r; i++) {
                iwork[i] = p - r + i;
            }
            for (i = r; i < p; i++) {
                iwork[i] = i - r;
            }
            if (wantu1) {
                dlapmt(0, p, p, U1, ldu1, iwork);
            }
            if (wantv1t) {
                dlapmr(0, p, q, V1T, ldv1t, iwork);
            }
        }
    }
}
