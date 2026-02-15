/**
 * @file cgesvdx.c
 * @brief CGESVDX computes the singular value decomposition (SVD) of a complex
 *        M-by-N matrix, optionally computing a subset of singular values/vectors.
 */

#include "semicolon_lapack_complex_single.h"
#include "lapack_tuning.h"
#include <complex.h>
#include <math.h>
#include <cblas.h>

static const f32 ZERO = 0.0f;
static const f32 ONE = 1.0f;
static const c64 CZERO = CMPLXF(0.0f, 0.0f);

/**
 * CGESVDX computes the singular value decomposition (SVD) of a complex
 * M-by-N matrix A, optionally computing the left and/or right singular
 * vectors. The SVD is written
 *
 *      A = U * SIGMA * transpose(V)
 *
 * where SIGMA is an M-by-N matrix which is zero except for its
 * min(m,n) diagonal elements, U is an M-by-M unitary matrix, and
 * V is an N-by-N unitary matrix. The diagonal elements of SIGMA
 * are the singular values of A; they are real and non-negative, and
 * are returned in descending order. The first min(m,n) columns of
 * U and V are the left and right singular vectors of A.
 *
 * CGESVDX uses an eigenvalue problem for obtaining the SVD, which
 * allows for the computation of a subset of singular values and
 * vectors. See SBDSVDX for details.
 *
 * Note that the routine returns V**T, not V.
 *
 * @param[in]     jobu    = 'V': Compute left singular vectors.
 *                         = 'N': Do not compute left singular vectors.
 * @param[in]     jobvt   = 'V': Compute right singular vectors (V**H).
 *                         = 'N': Do not compute right singular vectors.
 * @param[in]     range   = 'A': all singular values will be found.
 *                         = 'V': all singular values in (VL,VU] will be found.
 *                         = 'I': the IL-th through IU-th singular values.
 * @param[in]     m       Number of rows of A. m >= 0.
 * @param[in]     n       Number of columns of A. n >= 0.
 * @param[in,out] A       Complex*16 array (lda, n). On entry, the M-by-N matrix A.
 *                        On exit, contents are destroyed.
 * @param[in]     lda     Leading dimension of A. lda >= max(1,m).
 * @param[in]     vl      If range='V', lower bound of interval (exclusive).
 * @param[in]     vu      If range='V', upper bound of interval. vu > vl.
 * @param[in]     il      If range='I', index of smallest singular value (1-based).
 * @param[in]     iu      If range='I', index of largest singular value (1-based).
 * @param[out]    ns      Number of singular values found.
 * @param[out]    S       Single precision array (min(m,n)). Singular values in descending order.
 * @param[out]    U       Complex*16 array (ldu, *). Left singular vectors if jobu='V'.
 * @param[in]     ldu     Leading dimension of U. ldu >= 1; ldu >= m if jobu='V'.
 * @param[out]    VT      Complex*16 array (ldvt, n). Right singular vectors if jobvt='V'.
 * @param[in]     ldvt    Leading dimension of VT. ldvt >= 1; ldvt >= ns if jobvt='V'.
 * @param[out]    work    Complex*16 workspace array of dimension lwork.
 * @param[in]     lwork   Dimension of work. If lwork = -1, workspace query.
 * @param[out]    rwork   Single precision workspace array.
 * @param[out]    iwork   Integer workspace of dimension 12*min(m,n).
 * @param[out]    info
 *                         - = 0: success.
 *                         - < 0: illegal argument.
 *                         - > 0: i eigenvectors failed to converge in SBDSVDX.
 */
void cgesvdx(const char* jobu, const char* jobvt, const char* range,
             const int m, const int n, c64* restrict A, const int lda,
             const f32 vl, const f32 vu, const int il, const int iu,
             int* ns, f32* restrict S, c64* restrict U,
             const int ldu, c64* restrict VT, const int ldvt,
             c64* restrict work, const int lwork,
             f32* restrict rwork, int* restrict iwork, int* info)
{
    int alls, inds, lquery, vals, wantu, wantvt;
    int i, id, ie, ierr, ilqf, iltgk, iutgk, iqrf, iscl, itau, itaup, itauq;
    int itemp, itempr, itgkz, j, k, maxwrk, minmn, minwrk, mnthr;
    f32 anrm, bignum, eps, smlnum;
    char jobz, rngtgk;
    f32 dum[1];

    /* Test the input parameters */
    *ns = 0;
    *info = 0;
    lquery = (lwork == -1);
    minmn = (m < n) ? m : n;

    wantu = (jobu[0] == 'V' || jobu[0] == 'v');
    wantvt = (jobvt[0] == 'V' || jobvt[0] == 'v');
    if (wantu || wantvt) {
        jobz = 'V';
    } else {
        jobz = 'N';
    }
    alls = (range[0] == 'A' || range[0] == 'a');
    vals = (range[0] == 'V' || range[0] == 'v');
    inds = (range[0] == 'I' || range[0] == 'i');

    *info = 0;
    if (!(jobu[0] == 'V' || jobu[0] == 'v') &&
        !(jobu[0] == 'N' || jobu[0] == 'n')) {
        *info = -1;
    } else if (!(jobvt[0] == 'V' || jobvt[0] == 'v') &&
               !(jobvt[0] == 'N' || jobvt[0] == 'n')) {
        *info = -2;
    } else if (!alls && !vals && !inds) {
        *info = -3;
    } else if (m < 0) {
        *info = -4;
    } else if (n < 0) {
        *info = -5;
    } else if (m > lda) {
        *info = -7;
    } else if (minmn > 0) {
        if (vals) {
            if (vl < ZERO) {
                *info = -8;
            } else if (vu <= vl) {
                *info = -9;
            }
        } else if (inds) {
            if (il < 1 || il > (minmn > 1 ? minmn : 1)) {
                *info = -10;
            } else if (iu < (minmn < il ? minmn : il) || iu > minmn) {
                *info = -11;
            }
        }
        if (*info == 0) {
            if (wantu && ldu < m) {
                *info = -15;
            } else if (wantvt) {
                if (inds) {
                    if (ldvt < iu - il + 1) {
                        *info = -17;
                    }
                } else if (ldvt < minmn) {
                    *info = -17;
                }
            }
        }
    }

    /* Compute workspace */
    if (*info == 0) {
        minwrk = 1;
        maxwrk = 1;
        if (minmn > 0) {
            int nb = lapack_get_nb("dgebrd");
            if (nb < 1) nb = 32;

            if (m >= n) {
                mnthr = (int)(1.6f * n);
                if (m >= mnthr) {
                    /* Path 1 (M much larger than N) */
                    minwrk = n * (n + 5);
                    maxwrk = n + n * nb;
                    maxwrk = (maxwrk > n * n + 2 * n + 2 * n * nb) ?
                             maxwrk : n * n + 2 * n + 2 * n * nb;
                    if (wantu || wantvt) {
                        maxwrk = (maxwrk > n * n + 2 * n + n * nb) ?
                                 maxwrk : n * n + 2 * n + n * nb;
                    }
                } else {
                    /* Path 2 (M at least N, but not much larger) */
                    minwrk = 3 * n + m;
                    maxwrk = 2 * n + (m + n) * nb;
                    if (wantu || wantvt) {
                        maxwrk = (maxwrk > 2 * n + n * nb) ?
                                 maxwrk : 2 * n + n * nb;
                    }
                }
            } else {
                mnthr = (int)(1.6f * m);
                if (n >= mnthr) {
                    /* Path 1t (N much larger than M) */
                    minwrk = m * (m + 5);
                    maxwrk = m + m * nb;
                    maxwrk = (maxwrk > m * m + 2 * m + 2 * m * nb) ?
                             maxwrk : m * m + 2 * m + 2 * m * nb;
                    if (wantu || wantvt) {
                        maxwrk = (maxwrk > m * m + 2 * m + m * nb) ?
                                 maxwrk : m * m + 2 * m + m * nb;
                    }
                } else {
                    /* Path 2t (N greater than M, but not much larger) */
                    minwrk = 3 * m + n;
                    maxwrk = 2 * m + (m + n) * nb;
                    if (wantu || wantvt) {
                        maxwrk = (maxwrk > 2 * m + m * nb) ?
                                 maxwrk : 2 * m + m * nb;
                    }
                }
            }
        }
        maxwrk = (maxwrk > minwrk) ? maxwrk : minwrk;
        work[0] = CMPLXF((f32)maxwrk, 0.0f);

        if (lwork < minwrk && !lquery) {
            *info = -19;
        }
    }

    if (*info != 0) {
        xerbla("CGESVDX", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    /* Quick return if possible */
    if (m == 0 || n == 0) {
        return;
    }

    /* Set singular value indices according to RANGE */
    if (alls) {
        rngtgk = 'I';
        iltgk = 1;
        iutgk = minmn;
    } else if (inds) {
        rngtgk = 'I';
        iltgk = il;
        iutgk = iu;
    } else {
        rngtgk = 'V';
        iltgk = 0;
        iutgk = 0;
    }

    /* Get machine constants */
    eps = slamch("P");
    smlnum = sqrtf(slamch("S")) / eps;
    bignum = ONE / smlnum;

    /* Scale A if max element outside range [SMLNUM, BIGNUM] */
    anrm = clange("M", m, n, A, lda, dum);
    iscl = 0;
    if (anrm > ZERO && anrm < smlnum) {
        iscl = 1;
        clascl("G", 0, 0, anrm, smlnum, m, n, A, lda, &ierr);
    } else if (anrm > bignum) {
        iscl = 1;
        clascl("G", 0, 0, anrm, bignum, m, n, A, lda, &ierr);
    }

    if (m >= n) {
        /* A has at least as many rows as columns */
        mnthr = (int)(1.6f * n);

        if (m >= mnthr) {
            /* Path 1 (M much larger than N):
             * A = Q * R = Q * ( QB * B * PB**H )
             *           = Q * ( QB * ( UB * S * VB**H ) * PB**H )
             * U = Q * QB * UB; V**H = VB**H * PB**H
             *
             * Compute A=Q*R */
            itau = 0;
            itemp = itau + n;
            cgeqrf(m, n, A, lda, &work[itau], &work[itemp], lwork - itemp, &ierr);

            /* Copy R into WORK and bidiagonalize it */
            iqrf = itemp;
            itauq = iqrf + n * n;
            itaup = itauq + n;
            itemp = itaup + n;
            id = 0;
            ie = id + n;
            itgkz = ie + n;
            clacpy("U", n, n, A, lda, &work[iqrf], n);
            claset("L", n - 1, n - 1, CZERO, CZERO, &work[iqrf + 1], n);
            cgebrd(n, n, &work[iqrf], n, &rwork[id], &rwork[ie],
                   &work[itauq], &work[itaup], &work[itemp],
                   lwork - itemp, &ierr);
            itempr = itgkz + n * (2 * n + 1);

            /* Solve eigenvalue problem TGK*Z=Z*S */
            char jobz_str[2] = {jobz, '\0'};
            char rngtgk_str[2] = {rngtgk, '\0'};
            sbdsvdx("U", jobz_str, rngtgk_str, n, &rwork[id], &rwork[ie],
                    vl, vu, iltgk, iutgk, ns, S, &rwork[itgkz], 2 * n,
                    &rwork[itempr], iwork, info);

            /* If needed, compute left singular vectors */
            if (wantu) {
                k = itgkz;
                for (i = 0; i < *ns; i++) {
                    for (j = 0; j < n; j++) {
                        U[j + i * ldu] = CMPLXF(rwork[k], 0.0f);
                        k++;
                    }
                    k += n;
                }
                claset("A", m - n, *ns, CZERO, CZERO, &U[n], ldu);

                /* Call CUNMBR to compute QB*UB */
                cunmbr("Q", "L", "N", n, *ns, n, &work[iqrf], n,
                       &work[itauq], U, ldu, &work[itemp],
                       lwork - itemp, &ierr);

                /* Call CUNMQR to compute Q*(QB*UB) */
                cunmqr("L", "N", m, *ns, n, A, lda, &work[itau],
                       U, ldu, &work[itemp], lwork - itemp, &ierr);
            }

            /* If needed, compute right singular vectors */
            if (wantvt) {
                k = itgkz + n;
                for (i = 0; i < *ns; i++) {
                    for (j = 0; j < n; j++) {
                        VT[i + j * ldvt] = CMPLXF(rwork[k], 0.0f);
                        k++;
                    }
                    k += n;
                }

                /* Call CUNMBR to compute VB**H * PB**H */
                cunmbr("P", "R", "C", *ns, n, n, &work[iqrf], n,
                       &work[itaup], VT, ldvt, &work[itemp],
                       lwork - itemp, &ierr);
            }
        } else {
            /* Path 2 (M at least N, but not much larger)
             * Reduce A to bidiagonal form without QR decomposition */
            itauq = 0;
            itaup = itauq + n;
            itemp = itaup + n;
            id = 0;
            ie = id + n;
            itgkz = ie + n;
            cgebrd(m, n, A, lda, &rwork[id], &rwork[ie],
                   &work[itauq], &work[itaup], &work[itemp],
                   lwork - itemp, &ierr);
            itempr = itgkz + n * (2 * n + 1);

            /* Solve eigenvalue problem TGK*Z=Z*S */
            char jobz_str[2] = {jobz, '\0'};
            char rngtgk_str[2] = {rngtgk, '\0'};
            sbdsvdx("U", jobz_str, rngtgk_str, n, &rwork[id], &rwork[ie],
                    vl, vu, iltgk, iutgk, ns, S, &rwork[itgkz], 2 * n,
                    &rwork[itempr], iwork, info);

            /* If needed, compute left singular vectors */
            if (wantu) {
                k = itgkz;
                for (i = 0; i < *ns; i++) {
                    for (j = 0; j < n; j++) {
                        U[j + i * ldu] = CMPLXF(rwork[k], 0.0f);
                        k++;
                    }
                    k += n;
                }
                claset("A", m - n, *ns, CZERO, CZERO, &U[n], ldu);

                /* Call CUNMBR to compute QB*UB */
                cunmbr("Q", "L", "N", m, *ns, n, A, lda, &work[itauq],
                       U, ldu, &work[itemp], lwork - itemp, &ierr);
            }

            /* If needed, compute right singular vectors */
            if (wantvt) {
                k = itgkz + n;
                for (i = 0; i < *ns; i++) {
                    for (j = 0; j < n; j++) {
                        VT[i + j * ldvt] = CMPLXF(rwork[k], 0.0f);
                        k++;
                    }
                    k += n;
                }

                /* Call CUNMBR to compute VB**H * PB**H */
                cunmbr("P", "R", "C", *ns, n, n, A, lda, &work[itaup],
                       VT, ldvt, &work[itemp], lwork - itemp, &ierr);
            }
        }
    } else {
        /* A has more columns than rows */
        mnthr = (int)(1.6f * m);

        if (n >= mnthr) {
            /* Path 1t (N much larger than M):
             * A = L * Q = ( QB * B * PB**H ) * Q
             *           = ( QB * ( UB * S * VB**H ) * PB**H ) * Q
             * U = QB * UB ; V**H = VB**H * PB**H * Q
             *
             * Compute A=L*Q */
            itau = 0;
            itemp = itau + m;
            cgelqf(m, n, A, lda, &work[itau], &work[itemp], lwork - itemp, &ierr);

            /* Copy L into WORK and bidiagonalize it */
            ilqf = itemp;
            itauq = ilqf + m * m;
            itaup = itauq + m;
            itemp = itaup + m;
            id = 0;
            ie = id + m;
            itgkz = ie + m;
            clacpy("L", m, m, A, lda, &work[ilqf], m);
            claset("U", m - 1, m - 1, CZERO, CZERO, &work[ilqf + m], m);
            cgebrd(m, m, &work[ilqf], m, &rwork[id], &rwork[ie],
                   &work[itauq], &work[itaup], &work[itemp],
                   lwork - itemp, &ierr);
            itempr = itgkz + m * (2 * m + 1);

            /* Solve eigenvalue problem TGK*Z=Z*S */
            char jobz_str[2] = {jobz, '\0'};
            char rngtgk_str[2] = {rngtgk, '\0'};
            sbdsvdx("U", jobz_str, rngtgk_str, m, &rwork[id], &rwork[ie],
                    vl, vu, iltgk, iutgk, ns, S, &rwork[itgkz], 2 * m,
                    &rwork[itempr], iwork, info);

            /* If needed, compute left singular vectors */
            if (wantu) {
                k = itgkz;
                for (i = 0; i < *ns; i++) {
                    for (j = 0; j < m; j++) {
                        U[j + i * ldu] = CMPLXF(rwork[k], 0.0f);
                        k++;
                    }
                    k += m;
                }

                /* Call CUNMBR to compute QB*UB */
                cunmbr("Q", "L", "N", m, *ns, m, &work[ilqf], m,
                       &work[itauq], U, ldu, &work[itemp],
                       lwork - itemp, &ierr);
            }

            /* If needed, compute right singular vectors */
            if (wantvt) {
                k = itgkz + m;
                for (i = 0; i < *ns; i++) {
                    for (j = 0; j < m; j++) {
                        VT[i + j * ldvt] = CMPLXF(rwork[k], 0.0f);
                        k++;
                    }
                    k += m;
                }
                claset("A", *ns, n - m, CZERO, CZERO, &VT[m * ldvt], ldvt);

                /* Call CUNMBR to compute (VB**H)*(PB**H) */
                cunmbr("P", "R", "C", *ns, m, m, &work[ilqf], m,
                       &work[itaup], VT, ldvt, &work[itemp],
                       lwork - itemp, &ierr);

                /* Call CUNMLQ to compute ((VB**H)*(PB**H))*Q */
                cunmlq("R", "N", *ns, n, m, A, lda, &work[itau],
                       VT, ldvt, &work[itemp], lwork - itemp, &ierr);
            }
        } else {
            /* Path 2t (N greater than M, but not much larger)
             * Reduce to bidiagonal form without LQ decomposition */
            itauq = 0;
            itaup = itauq + m;
            itemp = itaup + m;
            id = 0;
            ie = id + m;
            itgkz = ie + m;
            cgebrd(m, n, A, lda, &rwork[id], &rwork[ie],
                   &work[itauq], &work[itaup], &work[itemp],
                   lwork - itemp, &ierr);
            itempr = itgkz + m * (2 * m + 1);

            /* Solve eigenvalue problem TGK*Z=Z*S */
            char jobz_str[2] = {jobz, '\0'};
            char rngtgk_str[2] = {rngtgk, '\0'};
            sbdsvdx("L", jobz_str, rngtgk_str, m, &rwork[id], &rwork[ie],
                    vl, vu, iltgk, iutgk, ns, S, &rwork[itgkz], 2 * m,
                    &rwork[itempr], iwork, info);

            /* If needed, compute left singular vectors */
            if (wantu) {
                k = itgkz;
                for (i = 0; i < *ns; i++) {
                    for (j = 0; j < m; j++) {
                        U[j + i * ldu] = CMPLXF(rwork[k], 0.0f);
                        k++;
                    }
                    k += m;
                }

                /* Call CUNMBR to compute QB*UB */
                cunmbr("Q", "L", "N", m, *ns, n, A, lda, &work[itauq],
                       U, ldu, &work[itemp], lwork - itemp, &ierr);
            }

            /* If needed, compute right singular vectors */
            if (wantvt) {
                k = itgkz + m;
                for (i = 0; i < *ns; i++) {
                    for (j = 0; j < m; j++) {
                        VT[i + j * ldvt] = CMPLXF(rwork[k], 0.0f);
                        k++;
                    }
                    k += m;
                }
                claset("A", *ns, n - m, CZERO, CZERO, &VT[m * ldvt], ldvt);

                /* Call CUNMBR to compute VB**H * PB**H */
                cunmbr("P", "R", "C", *ns, n, m, A, lda, &work[itaup],
                       VT, ldvt, &work[itemp], lwork - itemp, &ierr);
            }
        }
    }

    /* Undo scaling if necessary */
    if (iscl == 1) {
        if (anrm > bignum) {
            slascl("G", 0, 0, bignum, anrm, minmn, 1, S, minmn, &ierr);
        }
        if (anrm < smlnum) {
            slascl("G", 0, 0, smlnum, anrm, minmn, 1, S, minmn, &ierr);
        }
    }

    /* Return optimal workspace */
    work[0] = CMPLXF((f32)maxwrk, 0.0f);
}
