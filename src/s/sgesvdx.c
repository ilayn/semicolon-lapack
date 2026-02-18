/**
 * @file sgesvdx.c
 * @brief SGESVDX computes the singular value decomposition (SVD) of a real
 *        M-by-N matrix, optionally computing a subset of singular values/vectors.
 */

#include "semicolon_lapack_single.h"
#include "lapack_tuning.h"
#include <math.h>
#include <cblas.h>

static const f32 ZERO = 0.0f;
static const f32 ONE = 1.0f;

/**
 * SGESVDX computes the singular value decomposition (SVD) of a real
 * M-by-N matrix A, optionally computing the left and/or right singular
 * vectors. The SVD is written
 *
 *      A = U * SIGMA * transpose(V)
 *
 * where SIGMA is an M-by-N matrix which is zero except for its
 * min(m,n) diagonal elements, U is an M-by-M orthogonal matrix, and
 * V is an N-by-N orthogonal matrix. The diagonal elements of SIGMA
 * are the singular values of A; they are real and non-negative, and
 * are returned in descending order. The first min(m,n) columns of
 * U and V are the left and right singular vectors of A.
 *
 * SGESVDX uses an eigenvalue problem for obtaining the SVD, which
 * allows for the computation of a subset of singular values and
 * vectors. See SBDSVDX for details.
 *
 * Note that the routine returns V**T, not V.
 *
 * @param[in]     jobu    = 'V': Compute left singular vectors.
 *                         = 'N': Do not compute left singular vectors.
 * @param[in]     jobvt   = 'V': Compute right singular vectors (V**T).
 *                         = 'N': Do not compute right singular vectors.
 * @param[in]     range   = 'A': all singular values will be found.
 *                         = 'V': all singular values in (VL,VU] will be found.
 *                         = 'I': the IL-th through IU-th singular values.
 * @param[in]     m       Number of rows of A. m >= 0.
 * @param[in]     n       Number of columns of A. n >= 0.
 * @param[in,out] A       Array (lda, n). On entry, the M-by-N matrix A.
 *                        On exit, contents are destroyed.
 * @param[in]     lda     Leading dimension of A. lda >= max(1,m).
 * @param[in]     vl      If range='V', lower bound of interval (exclusive).
 * @param[in]     vu      If range='V', upper bound of interval. vu > vl.
 * @param[in]     il      If range='I', index of smallest singular value (0-based).
 * @param[in]     iu      If range='I', index of largest singular value (0-based).
 * @param[out]    ns      Number of singular values found.
 * @param[out]    S       Array (min(m,n)). Singular values in descending order.
 * @param[out]    U       Array (ldu, *). Left singular vectors if jobu='V'.
 * @param[in]     ldu     Leading dimension of U. ldu >= 1; ldu >= m if jobu='V'.
 * @param[out]    VT      Array (ldvt, n). Right singular vectors if jobvt='V'.
 * @param[in]     ldvt    Leading dimension of VT. ldvt >= 1; ldvt >= ns if jobvt='V'.
 * @param[out]    work    Workspace array of dimension lwork.
 * @param[in]     lwork   Dimension of work. If lwork = -1, workspace query.
 * @param[out]    iwork   Integer workspace of dimension 12*min(m,n).
 * @param[out]    info
 *                         - = 0: success.
 *                         - < 0: illegal argument.
 *                         - > 0: i eigenvectors failed to converge in SBDSVDX.
 */
void sgesvdx(const char* jobu, const char* jobvt, const char* range,
             const int m, const int n, f32* restrict A, const int lda,
             const f32 vl, const f32 vu, const int il, const int iu,
             int* ns, f32* restrict S, f32* restrict U,
             const int ldu, f32* restrict VT, const int ldvt,
             f32* restrict work, const int lwork,
             int* restrict iwork, int* info)
{
    int alls, inds, lquery, vals, wantu, wantvt;
    int i, id, ie, ierr, ilqf, iltgk, iqrf, iscl, itau, itaup, itauq;
    int itemp, itgkz, iutgk, j, maxwrk, minmn, minwrk, mnthr;
    f32 anrm, bignum, eps, smlnum;
    char jobz, rngtgk;

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
            if (il < 0 || il > (0 > minmn - 1 ? 0 : minmn - 1)) {
                *info = -10;
            } else if (iu < ((minmn - 1) < il ? (minmn - 1) : il) || iu > minmn - 1) {
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
            /* Get block size from tuning parameters */
            int nb = lapack_get_nb("dgebrd");
            if (nb < 1) nb = 32;

            if (m >= n) {
                /* MNTHR from ILAENV(6, 'SGESVD', ...) - typically 1.6 * n */
                mnthr = (int)(1.6f * n);
                if (m >= mnthr) {
                    /* Path 1 (M much larger than N) */
                    maxwrk = n + n * nb;
                    maxwrk = (maxwrk > n * (n + 5) + 2 * n * nb) ?
                             maxwrk : n * (n + 5) + 2 * n * nb;
                    if (wantu) {
                        maxwrk = (maxwrk > n * (3 * n + 6) + n * nb) ?
                                 maxwrk : n * (3 * n + 6) + n * nb;
                    }
                    if (wantvt) {
                        maxwrk = (maxwrk > n * (3 * n + 6) + n * nb) ?
                                 maxwrk : n * (3 * n + 6) + n * nb;
                    }
                    minwrk = n * (3 * n + 20);
                } else {
                    /* Path 2 (M at least N, but not much larger) */
                    maxwrk = 4 * n + (m + n) * nb;
                    if (wantu) {
                        maxwrk = (maxwrk > n * (2 * n + 5) + n * nb) ?
                                 maxwrk : n * (2 * n + 5) + n * nb;
                    }
                    if (wantvt) {
                        maxwrk = (maxwrk > n * (2 * n + 5) + n * nb) ?
                                 maxwrk : n * (2 * n + 5) + n * nb;
                    }
                    minwrk = (n * (2 * n + 19) > 4 * n + m) ?
                             n * (2 * n + 19) : 4 * n + m;
                }
            } else {
                /* MNTHR from ILAENV(6, 'SGESVD', ...) - typically 1.6 * m */
                mnthr = (int)(1.6f * m);
                if (n >= mnthr) {
                    /* Path 1t (N much larger than M) */
                    maxwrk = m + m * nb;
                    maxwrk = (maxwrk > m * (m + 5) + 2 * m * nb) ?
                             maxwrk : m * (m + 5) + 2 * m * nb;
                    if (wantu) {
                        maxwrk = (maxwrk > m * (3 * m + 6) + m * nb) ?
                                 maxwrk : m * (3 * m + 6) + m * nb;
                    }
                    if (wantvt) {
                        maxwrk = (maxwrk > m * (3 * m + 6) + m * nb) ?
                                 maxwrk : m * (3 * m + 6) + m * nb;
                    }
                    minwrk = m * (3 * m + 20);
                } else {
                    /* Path 2t (N at least M, but not much larger) */
                    maxwrk = 4 * m + (m + n) * nb;
                    if (wantu) {
                        maxwrk = (maxwrk > m * (2 * m + 5) + m * nb) ?
                                 maxwrk : m * (2 * m + 5) + m * nb;
                    }
                    if (wantvt) {
                        maxwrk = (maxwrk > m * (2 * m + 5) + m * nb) ?
                                 maxwrk : m * (2 * m + 5) + m * nb;
                    }
                    minwrk = (m * (2 * m + 19) > 4 * m + n) ?
                             m * (2 * m + 19) : 4 * m + n;
                }
            }
        }
        maxwrk = (maxwrk > minwrk) ? maxwrk : minwrk;
        work[0] = (f32)maxwrk;

        if (lwork < minwrk && !lquery) {
            *info = -19;
        }
    }

    if (*info != 0) {
        xerbla("SGESVDX", -(*info));
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
        iltgk = 0;
        iutgk = minmn - 1;
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
    anrm = slange("M", m, n, A, lda, work);
    iscl = 0;
    if (anrm > ZERO && anrm < smlnum) {
        iscl = 1;
        slascl("G", 0, 0, anrm, smlnum, m, n, A, lda, &ierr);
    } else if (anrm > bignum) {
        iscl = 1;
        slascl("G", 0, 0, anrm, bignum, m, n, A, lda, &ierr);
    }

    if (m >= n) {
        /* A has at least as many rows as columns */
        /* MNTHR from ILAENV(6, 'SGESVD', ...) */
        mnthr = (int)(1.6f * n);

        if (m >= mnthr) {
            /* Path 1 (M much larger than N):
             * A = Q * R = Q * ( QB * B * PB**T )
             *           = Q * ( QB * ( UB * S * VB**T ) * PB**T )
             * U = Q * QB * UB; V**T = VB**T * PB**T
             *
             * Compute A=Q*R */
            itau = 0;
            itemp = itau + n;
            sgeqrf(m, n, A, lda, &work[itau], &work[itemp], lwork - itemp, &ierr);

            /* Copy R into WORK and bidiagonalize it */
            iqrf = itemp;
            id = iqrf + n * n;
            ie = id + n;
            itauq = ie + n;
            itaup = itauq + n;
            itemp = itaup + n;
            slacpy("U", n, n, A, lda, &work[iqrf], n);
            slaset("L", n - 1, n - 1, ZERO, ZERO, &work[iqrf + 1], n);
            sgebrd(n, n, &work[iqrf], n, &work[id], &work[ie],
                   &work[itauq], &work[itaup], &work[itemp],
                   lwork - itemp, &ierr);

            /* Solve eigenvalue problem TGK*Z=Z*S */
            itgkz = itemp;
            itemp = itgkz + n * (2 * n + 1);

            char jobz_str[2] = {jobz, '\0'};
            char rngtgk_str[2] = {rngtgk, '\0'};
            sbdsvdx("U", jobz_str, rngtgk_str, n, &work[id], &work[ie],
                    vl, vu, iltgk, iutgk, ns, S, &work[itgkz], 2 * n,
                    &work[itemp], iwork, info);

            /* If needed, compute left singular vectors */
            if (wantu) {
                j = itgkz;
                for (i = 0; i < *ns; i++) {
                    cblas_scopy(n, &work[j], 1, &U[i * ldu], 1);
                    j += 2 * n;
                }
                slaset("A", m - n, *ns, ZERO, ZERO, &U[n], ldu);

                /* Call SORMBR to compute QB*UB */
                sormbr("Q", "L", "N", n, *ns, n, &work[iqrf], n,
                       &work[itauq], U, ldu, &work[itemp],
                       lwork - itemp, &ierr);

                /* Call SORMQR to compute Q*(QB*UB) */
                sormqr("L", "N", m, *ns, n, A, lda, &work[itau],
                       U, ldu, &work[itemp], lwork - itemp, &ierr);
            }

            /* If needed, compute right singular vectors */
            if (wantvt) {
                j = itgkz + n;
                for (i = 0; i < *ns; i++) {
                    cblas_scopy(n, &work[j], 1, &VT[i], ldvt);
                    j += 2 * n;
                }

                /* Call SORMBR to compute VB**T * PB**T */
                sormbr("P", "R", "T", *ns, n, n, &work[iqrf], n,
                       &work[itaup], VT, ldvt, &work[itemp],
                       lwork - itemp, &ierr);
            }
        } else {
            /* Path 2 (M at least N, but not much larger)
             * Reduce A to bidiagonal form without QR decomposition */
            id = 0;
            ie = id + n;
            itauq = ie + n;
            itaup = itauq + n;
            itemp = itaup + n;
            sgebrd(m, n, A, lda, &work[id], &work[ie],
                   &work[itauq], &work[itaup], &work[itemp],
                   lwork - itemp, &ierr);

            /* Solve eigenvalue problem TGK*Z=Z*S */
            itgkz = itemp;
            itemp = itgkz + n * (2 * n + 1);

            char jobz_str[2] = {jobz, '\0'};
            char rngtgk_str[2] = {rngtgk, '\0'};
            sbdsvdx("U", jobz_str, rngtgk_str, n, &work[id], &work[ie],
                    vl, vu, iltgk, iutgk, ns, S, &work[itgkz], 2 * n,
                    &work[itemp], iwork, info);

            /* If needed, compute left singular vectors */
            if (wantu) {
                j = itgkz;
                for (i = 0; i < *ns; i++) {
                    cblas_scopy(n, &work[j], 1, &U[i * ldu], 1);
                    j += 2 * n;
                }
                slaset("A", m - n, *ns, ZERO, ZERO, &U[n], ldu);

                /* Call SORMBR to compute QB*UB */
                sormbr("Q", "L", "N", m, *ns, n, A, lda, &work[itauq],
                       U, ldu, &work[itemp], lwork - itemp, &ierr);
            }

            /* If needed, compute right singular vectors */
            if (wantvt) {
                j = itgkz + n;
                for (i = 0; i < *ns; i++) {
                    cblas_scopy(n, &work[j], 1, &VT[i], ldvt);
                    j += 2 * n;
                }

                /* Call SORMBR to compute VB**T * PB**T */
                sormbr("P", "R", "T", *ns, n, n, A, lda, &work[itaup],
                       VT, ldvt, &work[itemp], lwork - itemp, &ierr);
            }
        }
    } else {
        /* A has more columns than rows */
        /* MNTHR from ILAENV(6, 'SGESVD', ...) */
        mnthr = (int)(1.6f * m);

        if (n >= mnthr) {
            /* Path 1t (N much larger than M):
             * A = L * Q = ( QB * B * PB**T ) * Q
             *           = ( QB * ( UB * S * VB**T ) * PB**T ) * Q
             * U = QB * UB ; V**T = VB**T * PB**T * Q
             *
             * Compute A=L*Q */
            itau = 0;
            itemp = itau + m;
            sgelqf(m, n, A, lda, &work[itau], &work[itemp], lwork - itemp, &ierr);

            /* Copy L into WORK and bidiagonalize it */
            ilqf = itemp;
            id = ilqf + m * m;
            ie = id + m;
            itauq = ie + m;
            itaup = itauq + m;
            itemp = itaup + m;
            slacpy("L", m, m, A, lda, &work[ilqf], m);
            slaset("U", m - 1, m - 1, ZERO, ZERO, &work[ilqf + m], m);
            sgebrd(m, m, &work[ilqf], m, &work[id], &work[ie],
                   &work[itauq], &work[itaup], &work[itemp],
                   lwork - itemp, &ierr);

            /* Solve eigenvalue problem TGK*Z=Z*S */
            itgkz = itemp;
            itemp = itgkz + m * (2 * m + 1);

            char jobz_str[2] = {jobz, '\0'};
            char rngtgk_str[2] = {rngtgk, '\0'};
            sbdsvdx("U", jobz_str, rngtgk_str, m, &work[id], &work[ie],
                    vl, vu, iltgk, iutgk, ns, S, &work[itgkz], 2 * m,
                    &work[itemp], iwork, info);

            /* If needed, compute left singular vectors */
            if (wantu) {
                j = itgkz;
                for (i = 0; i < *ns; i++) {
                    cblas_scopy(m, &work[j], 1, &U[i * ldu], 1);
                    j += 2 * m;
                }

                /* Call SORMBR to compute QB*UB */
                sormbr("Q", "L", "N", m, *ns, m, &work[ilqf], m,
                       &work[itauq], U, ldu, &work[itemp],
                       lwork - itemp, &ierr);
            }

            /* If needed, compute right singular vectors */
            if (wantvt) {
                j = itgkz + m;
                for (i = 0; i < *ns; i++) {
                    cblas_scopy(m, &work[j], 1, &VT[i], ldvt);
                    j += 2 * m;
                }
                slaset("A", *ns, n - m, ZERO, ZERO, &VT[m * ldvt], ldvt);

                /* Call SORMBR to compute (VB**T)*(PB**T) */
                sormbr("P", "R", "T", *ns, m, m, &work[ilqf], m,
                       &work[itaup], VT, ldvt, &work[itemp],
                       lwork - itemp, &ierr);

                /* Call SORMLQ to compute ((VB**T)*(PB**T))*Q */
                sormlq("R", "N", *ns, n, m, A, lda, &work[itau],
                       VT, ldvt, &work[itemp], lwork - itemp, &ierr);
            }
        } else {
            /* Path 2t (N greater than M, but not much larger)
             * Reduce to bidiagonal form without LQ decomposition */
            id = 0;
            ie = id + m;
            itauq = ie + m;
            itaup = itauq + m;
            itemp = itaup + m;
            sgebrd(m, n, A, lda, &work[id], &work[ie],
                   &work[itauq], &work[itaup], &work[itemp],
                   lwork - itemp, &ierr);

            /* Solve eigenvalue problem TGK*Z=Z*S */
            itgkz = itemp;
            itemp = itgkz + m * (2 * m + 1);

            char jobz_str[2] = {jobz, '\0'};
            char rngtgk_str[2] = {rngtgk, '\0'};
            sbdsvdx("L", jobz_str, rngtgk_str, m, &work[id], &work[ie],
                    vl, vu, iltgk, iutgk, ns, S, &work[itgkz], 2 * m,
                    &work[itemp], iwork, info);

            /* If needed, compute left singular vectors */
            if (wantu) {
                j = itgkz;
                for (i = 0; i < *ns; i++) {
                    cblas_scopy(m, &work[j], 1, &U[i * ldu], 1);
                    j += 2 * m;
                }

                /* Call SORMBR to compute QB*UB */
                sormbr("Q", "L", "N", m, *ns, n, A, lda, &work[itauq],
                       U, ldu, &work[itemp], lwork - itemp, &ierr);
            }

            /* If needed, compute right singular vectors */
            if (wantvt) {
                j = itgkz + m;
                for (i = 0; i < *ns; i++) {
                    cblas_scopy(m, &work[j], 1, &VT[i], ldvt);
                    j += 2 * m;
                }
                slaset("A", *ns, n - m, ZERO, ZERO, &VT[m * ldvt], ldvt);

                /* Call SORMBR to compute VB**T * PB**T */
                sormbr("P", "R", "T", *ns, n, m, A, lda, &work[itaup],
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
    work[0] = (f32)maxwrk;
}
