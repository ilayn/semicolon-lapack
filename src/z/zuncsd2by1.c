/**
 * @file zuncsd2by1.c
 * @brief ZUNCSD2BY1 computes the CS decomposition of an M-by-Q matrix with orthonormal columns partitioned into a 2-by-1 block structure.
 */

#include <math.h>
#include <cblas.h>
#include "semicolon_lapack_complex_double.h"

/**
 * ZUNCSD2BY1 computes the CS decomposition of an M-by-Q matrix X with
 * orthonormal columns that has been partitioned into a 2-by-1 block
 * structure:
 *
 *                                [  I1 0  0 ]
 *                                [  0  C  0 ]
 *          [ X11 ]   [ U1 |    ] [  0  0  0 ]
 *      X = [-----] = [---------] [----------] V1**H .
 *          [ X21 ]   [    | U2 ] [  0  0  0 ]
 *                                [  0  S  0 ]
 *                                [  0  0  I2]
 *
 * X11 is P-by-Q. The unitary matrices U1, U2, and V1 are P-by-P,
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
 *          Complex*16 array, dimension (ldx11, q).
 *          On entry, part of the unitary matrix whose CSD is desired.
 *
 * @param[in] ldx11
 *          The leading dimension of X11. ldx11 >= max(1, p).
 *
 * @param[in,out] X21
 *          Complex*16 array, dimension (ldx21, q).
 *          On entry, part of the unitary matrix whose CSD is desired.
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
 *          Complex*16 array, dimension (ldu1, p).
 *          If jobu1 = 'Y', U1 contains the P-by-P unitary matrix U1.
 *
 * @param[in] ldu1
 *          The leading dimension of U1. If jobu1 = 'Y', ldu1 >= max(1, p).
 *
 * @param[out] U2
 *          Complex*16 array, dimension (ldu2, m-p).
 *          If jobu2 = 'Y', U2 contains the (M-P)-by-(M-P) unitary matrix U2.
 *
 * @param[in] ldu2
 *          The leading dimension of U2. If jobu2 = 'Y', ldu2 >= max(1, m-p).
 *
 * @param[out] V1T
 *          Complex*16 array, dimension (ldv1t, q).
 *          If jobv1t = 'Y', V1T contains the Q-by-Q unitary matrix V1**H.
 *
 * @param[in] ldv1t
 *          The leading dimension of V1T. If jobv1t = 'Y', ldv1t >= max(1, q).
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
 *          Double precision array, dimension (max(1, lrwork)).
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
 *                         - > 0: ZBBCSD did not converge.
 */
void zuncsd2by1(
    const char* jobu1,
    const char* jobu2,
    const char* jobv1t,
    const int m,
    const int p,
    const int q,
    c128* restrict X11,
    const int ldx11,
    c128* restrict X21,
    const int ldx21,
    f64* restrict theta,
    c128* restrict U1,
    const int ldu1,
    c128* restrict U2,
    const int ldu2,
    c128* restrict V1T,
    const int ldv1t,
    c128* restrict work,
    const int lwork,
    f64* restrict rwork,
    const int lrwork,
    int* restrict iwork,
    int* info)
{
    const c128 one = CMPLX(1.0, 0.0);
    const c128 zero_c = CMPLX(0.0, 0.0);

    int childinfo, i, ib11d, ib11e, ib12d, ib12e;
    int ib21d, ib21e, ib22d, ib22e, ibbcsd, iorbdb;
    int iorglq, iorgqr, iphi, itaup1, itaup2, itauq1;
    int j, lbbcsd, lorbdb, lorglq, lorglqmin;
    int lorglqopt, lorgqr, lorgqrmin, lorgqropt;
    int lworkmin, lworkopt, lrworkmin, lrworkopt, r;
    int lquery, wantu1, wantu2, wantv1t;
    c128 cdum[1];

    *info = 0;
    wantu1 = (jobu1[0] == 'Y' || jobu1[0] == 'y');
    wantu2 = (jobu2[0] == 'Y' || jobu2[0] == 'y');
    wantv1t = (jobv1t[0] == 'Y' || jobv1t[0] == 'y');
    lquery = (lwork == -1 || lrwork == -1);

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
        iphi = 2;
        ib11d = iphi + (1 > (r - 1) ? 1 : (r - 1));
        ib11e = ib11d + (1 > r ? 1 : r);
        ib12d = ib11e + (1 > (r - 1) ? 1 : (r - 1));
        ib12e = ib12d + (1 > r ? 1 : r);
        ib21d = ib12e + (1 > (r - 1) ? 1 : (r - 1));
        ib21e = ib21d + (1 > r ? 1 : r);
        ib22d = ib21e + (1 > (r - 1) ? 1 : (r - 1));
        ib22e = ib22d + (1 > r ? 1 : r);
        ibbcsd = ib22e + (1 > (r - 1) ? 1 : (r - 1));
        itaup1 = 2;
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
            zunbdb1(m, p, q, X11, ldx11, X21, ldx21, theta,
                    NULL, NULL, NULL, NULL, work, -1, &childinfo);
            lorbdb = (int)creal(work[0]);
            if (wantu1 && p > 0) {
                zungqr(p, p, q, U1, ldu1, cdum, work, -1, &childinfo);
                lorgqrmin = lorgqrmin > p ? lorgqrmin : p;
                lorgqropt = lorgqropt > (int)creal(work[0]) ? lorgqropt : (int)creal(work[0]);
            }
            if (wantu2 && m - p > 0) {
                zungqr(m - p, m - p, q, U2, ldu2, cdum, work, -1, &childinfo);
                lorgqrmin = lorgqrmin > (m - p) ? lorgqrmin : (m - p);
                lorgqropt = lorgqropt > (int)creal(work[0]) ? lorgqropt : (int)creal(work[0]);
            }
            if (wantv1t && q > 0) {
                zunglq(q - 1, q - 1, q - 1, V1T, ldv1t, cdum, work, -1, &childinfo);
                lorglqmin = lorglqmin > (q - 1) ? lorglqmin : (q - 1);
                lorglqopt = lorglqopt > (int)creal(work[0]) ? lorglqopt : (int)creal(work[0]);
            }
            zbbcsd(jobu1, jobu2, jobv1t, "N", "N", m, p, q, theta,
                   NULL, U1, ldu1, U2, ldu2, V1T, ldv1t,
                   cdum, 1, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
                   rwork, -1, &childinfo);
            lbbcsd = (int)rwork[0];
        } else if (r == p) {
            zunbdb2(m, p, q, X11, ldx11, X21, ldx21, theta,
                    NULL, NULL, NULL, NULL, work, -1, &childinfo);
            lorbdb = (int)creal(work[0]);
            if (wantu1 && p > 0) {
                zungqr(p - 1, p - 1, p - 1, &U1[1 + 1 * ldu1], ldu1, cdum, work, -1, &childinfo);
                lorgqrmin = lorgqrmin > (p - 1) ? lorgqrmin : (p - 1);
                lorgqropt = lorgqropt > (int)creal(work[0]) ? lorgqropt : (int)creal(work[0]);
            }
            if (wantu2 && m - p > 0) {
                zungqr(m - p, m - p, q, U2, ldu2, cdum, work, -1, &childinfo);
                lorgqrmin = lorgqrmin > (m - p) ? lorgqrmin : (m - p);
                lorgqropt = lorgqropt > (int)creal(work[0]) ? lorgqropt : (int)creal(work[0]);
            }
            if (wantv1t && q > 0) {
                zunglq(q, q, r, V1T, ldv1t, cdum, work, -1, &childinfo);
                lorglqmin = lorglqmin > q ? lorglqmin : q;
                lorglqopt = lorglqopt > (int)creal(work[0]) ? lorglqopt : (int)creal(work[0]);
            }
            zbbcsd(jobv1t, "N", jobu1, jobu2, "T", m, q, p, theta,
                   NULL, V1T, ldv1t, cdum, 1, U1, ldu1, U2, ldu2,
                   NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
                   rwork, -1, &childinfo);
            lbbcsd = (int)rwork[0];
        } else if (r == m - p) {
            zunbdb3(m, p, q, X11, ldx11, X21, ldx21, theta,
                    NULL, NULL, NULL, NULL, work, -1, &childinfo);
            lorbdb = (int)creal(work[0]);
            if (wantu1 && p > 0) {
                zungqr(p, p, q, U1, ldu1, cdum, work, -1, &childinfo);
                lorgqrmin = lorgqrmin > p ? lorgqrmin : p;
                lorgqropt = lorgqropt > (int)creal(work[0]) ? lorgqropt : (int)creal(work[0]);
            }
            if (wantu2 && m - p > 0) {
                zungqr(m - p - 1, m - p - 1, m - p - 1, &U2[1 + 1 * ldu2], ldu2,
                       cdum, work, -1, &childinfo);
                lorgqrmin = lorgqrmin > (m - p - 1) ? lorgqrmin : (m - p - 1);
                lorgqropt = lorgqropt > (int)creal(work[0]) ? lorgqropt : (int)creal(work[0]);
            }
            if (wantv1t && q > 0) {
                zunglq(q, q, r, V1T, ldv1t, cdum, work, -1, &childinfo);
                lorglqmin = lorglqmin > q ? lorglqmin : q;
                lorglqopt = lorglqopt > (int)creal(work[0]) ? lorglqopt : (int)creal(work[0]);
            }
            zbbcsd("N", jobv1t, jobu2, jobu1, "T", m, m - q, m - p,
                   theta, NULL, cdum, 1, V1T, ldv1t, U2, ldu2, U1, ldu1,
                   NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
                   rwork, -1, &childinfo);
            lbbcsd = (int)rwork[0];
        } else {
            zunbdb4(m, p, q, X11, ldx11, X21, ldx21, theta,
                    NULL, NULL, NULL, NULL, NULL, work, -1, &childinfo);
            lorbdb = m + (int)creal(work[0]);
            if (wantu1 && p > 0) {
                zungqr(p, p, m - q, U1, ldu1, cdum, work, -1, &childinfo);
                lorgqrmin = lorgqrmin > p ? lorgqrmin : p;
                lorgqropt = lorgqropt > (int)creal(work[0]) ? lorgqropt : (int)creal(work[0]);
            }
            if (wantu2 && m - p > 0) {
                zungqr(m - p, m - p, m - q, U2, ldu2, cdum, work, -1, &childinfo);
                lorgqrmin = lorgqrmin > (m - p) ? lorgqrmin : (m - p);
                lorgqropt = lorgqropt > (int)creal(work[0]) ? lorgqropt : (int)creal(work[0]);
            }
            if (wantv1t && q > 0) {
                zunglq(q, q, q, V1T, ldv1t, cdum, work, -1, &childinfo);
                lorglqmin = lorglqmin > q ? lorglqmin : q;
                lorglqopt = lorglqopt > (int)creal(work[0]) ? lorglqopt : (int)creal(work[0]);
            }
            zbbcsd(jobu2, jobu1, "N", jobv1t, "N", m, m - p, m - q,
                   theta, NULL, U2, ldu2, U1, ldu1, cdum, 1, V1T, ldv1t,
                   NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
                   rwork, -1, &childinfo);
            lbbcsd = (int)rwork[0];
        }

        lrworkmin = ibbcsd + lbbcsd - 1;
        lrworkopt = lrworkmin;
        rwork[0] = (f64)lrworkopt;

        lworkmin = iorbdb + lorbdb - 1;
        if (iorgqr + lorgqrmin - 1 > lworkmin) lworkmin = iorgqr + lorgqrmin - 1;
        if (iorglq + lorglqmin - 1 > lworkmin) lworkmin = iorglq + lorglqmin - 1;

        lworkopt = iorbdb + lorbdb - 1;
        if (iorgqr + lorgqropt - 1 > lworkopt) lworkopt = iorgqr + lorgqropt - 1;
        if (iorglq + lorglqopt - 1 > lworkopt) lworkopt = iorglq + lorglqopt - 1;

        work[0] = CMPLX((f64)lworkopt, 0.0);
        if (lwork < lworkmin && !lquery) {
            *info = -19;
        }
        if (lrwork < lrworkmin && !lquery) {
            *info = -21;
        }
    }

    if (*info != 0) {
        xerbla("ZUNCSD2BY1", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    lorgqr = lwork - iorgqr + 1;
    lorglq = lwork - iorglq + 1;

    if (r == q) {

        zunbdb1(m, p, q, X11, ldx11, X21, ldx21, theta,
                &rwork[iphi - 1], &work[itaup1 - 1], &work[itaup2 - 1],
                &work[itauq1 - 1], &work[iorbdb - 1], lorbdb, &childinfo);

        if (wantu1 && p > 0) {
            zlacpy("L", p, q, X11, ldx11, U1, ldu1);
            zungqr(p, p, q, U1, ldu1, &work[itaup1 - 1],
                   &work[iorgqr - 1], lorgqr, &childinfo);
        }
        if (wantu2 && m - p > 0) {
            zlacpy("L", m - p, q, X21, ldx21, U2, ldu2);
            zungqr(m - p, m - p, q, U2, ldu2, &work[itaup2 - 1],
                   &work[iorgqr - 1], lorgqr, &childinfo);
        }
        if (wantv1t && q > 0) {
            V1T[0] = one;
            for (j = 1; j < q; j++) {
                V1T[j * ldv1t] = zero_c;
                V1T[j] = zero_c;
            }
            zlacpy("U", q - 1, q - 1, &X21[1 * ldx21], ldx21, &V1T[1 + 1 * ldv1t], ldv1t);
            zunglq(q - 1, q - 1, q - 1, &V1T[1 + 1 * ldv1t], ldv1t,
                   &work[itauq1 - 1], &work[iorglq - 1], lorglq, &childinfo);
        }

        zbbcsd(jobu1, jobu2, jobv1t, "N", "N", m, p, q, theta,
               &rwork[iphi - 1], U1, ldu1, U2, ldu2, V1T, ldv1t,
               cdum, 1, &rwork[ib11d - 1], &rwork[ib11e - 1],
               &rwork[ib12d - 1], &rwork[ib12e - 1], &rwork[ib21d - 1],
               &rwork[ib21e - 1], &rwork[ib22d - 1], &rwork[ib22e - 1],
               &rwork[ibbcsd - 1], lrwork - ibbcsd + 1, &childinfo);

        if (q > 0 && wantu2) {
            for (i = 0; i < q; i++) {
                iwork[i] = m - p - q + i + 1;
            }
            for (i = q; i < m - p; i++) {
                iwork[i] = i - q + 1;
            }
            zlapmt(0, m - p, m - p, U2, ldu2, iwork);
        }

    } else if (r == p) {

        zunbdb2(m, p, q, X11, ldx11, X21, ldx21, theta,
                &rwork[iphi - 1], &work[itaup1 - 1], &work[itaup2 - 1],
                &work[itauq1 - 1], &work[iorbdb - 1], lorbdb, &childinfo);

        if (wantu1 && p > 0) {
            U1[0] = one;
            for (j = 1; j < p; j++) {
                U1[j * ldu1] = zero_c;
                U1[j] = zero_c;
            }
            zlacpy("L", p - 1, p - 1, &X11[1], ldx11, &U1[1 + 1 * ldu1], ldu1);
            zungqr(p - 1, p - 1, p - 1, &U1[1 + 1 * ldu1], ldu1, &work[itaup1 - 1],
                   &work[iorgqr - 1], lorgqr, &childinfo);
        }
        if (wantu2 && m - p > 0) {
            zlacpy("L", m - p, q, X21, ldx21, U2, ldu2);
            zungqr(m - p, m - p, q, U2, ldu2, &work[itaup2 - 1],
                   &work[iorgqr - 1], lorgqr, &childinfo);
        }
        if (wantv1t && q > 0) {
            zlacpy("U", p, q, X11, ldx11, V1T, ldv1t);
            zunglq(q, q, r, V1T, ldv1t, &work[itauq1 - 1],
                   &work[iorglq - 1], lorglq, &childinfo);
        }

        zbbcsd(jobv1t, "N", jobu1, jobu2, "T", m, q, p, theta,
               &rwork[iphi - 1], V1T, ldv1t, cdum, 1, U1, ldu1, U2, ldu2,
               &rwork[ib11d - 1], &rwork[ib11e - 1], &rwork[ib12d - 1],
               &rwork[ib12e - 1], &rwork[ib21d - 1], &rwork[ib21e - 1],
               &rwork[ib22d - 1], &rwork[ib22e - 1], &rwork[ibbcsd - 1], lbbcsd,
               &childinfo);

        if (q > 0 && wantu2) {
            for (i = 0; i < q; i++) {
                iwork[i] = m - p - q + i + 1;
            }
            for (i = q; i < m - p; i++) {
                iwork[i] = i - q + 1;
            }
            zlapmt(0, m - p, m - p, U2, ldu2, iwork);
        }

    } else if (r == m - p) {

        zunbdb3(m, p, q, X11, ldx11, X21, ldx21, theta,
                &rwork[iphi - 1], &work[itaup1 - 1], &work[itaup2 - 1],
                &work[itauq1 - 1], &work[iorbdb - 1], lorbdb, &childinfo);

        if (wantu1 && p > 0) {
            zlacpy("L", p, q, X11, ldx11, U1, ldu1);
            zungqr(p, p, q, U1, ldu1, &work[itaup1 - 1],
                   &work[iorgqr - 1], lorgqr, &childinfo);
        }
        if (wantu2 && m - p > 0) {
            U2[0] = one;
            for (j = 1; j < m - p; j++) {
                U2[j * ldu2] = zero_c;
                U2[j] = zero_c;
            }
            zlacpy("L", m - p - 1, m - p - 1, &X21[1], ldx21, &U2[1 + 1 * ldu2], ldu2);
            zungqr(m - p - 1, m - p - 1, m - p - 1, &U2[1 + 1 * ldu2], ldu2,
                   &work[itaup2 - 1], &work[iorgqr - 1], lorgqr, &childinfo);
        }
        if (wantv1t && q > 0) {
            zlacpy("U", m - p, q, X21, ldx21, V1T, ldv1t);
            zunglq(q, q, r, V1T, ldv1t, &work[itauq1 - 1],
                   &work[iorglq - 1], lorglq, &childinfo);
        }

        zbbcsd("N", jobv1t, jobu2, jobu1, "T", m, m - q, m - p, theta,
               &rwork[iphi - 1], cdum, 1, V1T, ldv1t, U2, ldu2, U1, ldu1,
               &rwork[ib11d - 1], &rwork[ib11e - 1], &rwork[ib12d - 1],
               &rwork[ib12e - 1], &rwork[ib21d - 1], &rwork[ib21e - 1],
               &rwork[ib22d - 1], &rwork[ib22e - 1], &rwork[ibbcsd - 1], lbbcsd,
               &childinfo);

        if (q > r) {
            for (i = 0; i < r; i++) {
                iwork[i] = q - r + i + 1;
            }
            for (i = r; i < q; i++) {
                iwork[i] = i - r + 1;
            }
            if (wantu1) {
                zlapmt(0, p, q, U1, ldu1, iwork);
            }
            if (wantv1t) {
                zlapmr(0, q, q, V1T, ldv1t, iwork);
            }
        }

    } else {

        zunbdb4(m, p, q, X11, ldx11, X21, ldx21, theta,
                &rwork[iphi - 1], &work[itaup1 - 1], &work[itaup2 - 1],
                &work[itauq1 - 1], &work[iorbdb - 1], &work[iorbdb + m - 1],
                lorbdb - m, &childinfo);

        if (wantu2 && m - p > 0) {
            cblas_zcopy(m - p, &work[iorbdb + p - 1], 1, U2, 1);
        }
        if (wantu1 && p > 0) {
            cblas_zcopy(p, &work[iorbdb - 1], 1, U1, 1);
            for (j = 1; j < p; j++) {
                U1[j * ldu1] = zero_c;
            }
            zlacpy("L", p - 1, m - q - 1, &X11[1], ldx11, &U1[1 + 1 * ldu1], ldu1);
            zungqr(p, p, m - q, U1, ldu1, &work[itaup1 - 1],
                   &work[iorgqr - 1], lorgqr, &childinfo);
        }
        if (wantu2 && m - p > 0) {
            for (j = 1; j < m - p; j++) {
                U2[j * ldu2] = zero_c;
            }
            zlacpy("L", m - p - 1, m - q - 1, &X21[1], ldx21, &U2[1 + 1 * ldu2], ldu2);
            zungqr(m - p, m - p, m - q, U2, ldu2, &work[itaup2 - 1],
                   &work[iorgqr - 1], lorgqr, &childinfo);
        }
        if (wantv1t && q > 0) {
            zlacpy("U", m - q, q, X21, ldx21, V1T, ldv1t);
            zlacpy("U", p - (m - q), q - (m - q), &X11[(m - q) + (m - q) * ldx11], ldx11,
                   &V1T[(m - q) + (m - q) * ldv1t], ldv1t);
            zlacpy("U", -p + q, q - p, &X21[(m - q) + p * ldx21], ldx21,
                   &V1T[p + p * ldv1t], ldv1t);
            zunglq(q, q, q, V1T, ldv1t, &work[itauq1 - 1],
                   &work[iorglq - 1], lorglq, &childinfo);
        }

        zbbcsd(jobu2, jobu1, "N", jobv1t, "N", m, m - p, m - q, theta,
               &rwork[iphi - 1], U2, ldu2, U1, ldu1, cdum, 1, V1T, ldv1t,
               &rwork[ib11d - 1], &rwork[ib11e - 1], &rwork[ib12d - 1],
               &rwork[ib12e - 1], &rwork[ib21d - 1], &rwork[ib21e - 1],
               &rwork[ib22d - 1], &rwork[ib22e - 1], &rwork[ibbcsd - 1], lbbcsd,
               &childinfo);

        if (p > r) {
            for (i = 0; i < r; i++) {
                iwork[i] = p - r + i + 1;
            }
            for (i = r; i < p; i++) {
                iwork[i] = i - r + 1;
            }
            if (wantu1) {
                zlapmt(0, p, p, U1, ldu1, iwork);
            }
            if (wantv1t) {
                zlapmr(0, p, q, V1T, ldv1t, iwork);
            }
        }
    }
}
