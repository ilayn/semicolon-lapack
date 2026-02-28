/**
 * @file slaqz0.c
 * @brief SLAQZ0 computes the eigenvalues of a matrix pair (H,T) using multishift QZ.
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "semicolon_lapack_single.h"

/**
 * SLAQZ0 computes the eigenvalues of a real matrix pair (H,T),
 * where H is an upper Hessenberg matrix and T is upper triangular,
 * using the double-shift QZ method with aggressive early deflation.
 *
 * @param[in]     wants    'E': Compute eigenvalues only; 'S': Compute Schur form.
 * @param[in]     wantq    'N': Q not computed; 'I': Q initialized; 'V': Q updated.
 * @param[in]     wantz    'N': Z not computed; 'I': Z initialized; 'V': Z updated.
 * @param[in]     n        The order of the matrices A, B, Q, and Z. n >= 0.
 * @param[in]     ilo      Lower bound of active submatrix (0-based).
 * @param[in]     ihi      Upper bound of active submatrix (0-based).
 * @param[in,out] A        Upper Hessenberg matrix A.
 * @param[in]     lda      Leading dimension of A.
 * @param[in,out] B        Upper triangular matrix B.
 * @param[in]     ldb      Leading dimension of B.
 * @param[out]    alphar   Real parts of eigenvalues.
 * @param[out]    alphai   Imaginary parts of eigenvalues.
 * @param[out]    beta     Scale factors for eigenvalues.
 * @param[in,out] Q        Left Schur vectors.
 * @param[in]     ldq      Leading dimension of Q.
 * @param[in,out] Z        Right Schur vectors.
 * @param[in]     ldz      Leading dimension of Z.
 * @param[out]    work     Workspace array.
 * @param[in]     lwork    Dimension of workspace. If lwork = -1, workspace query.
 * @param[in]     rec      Current recursion level. Should be set to 0 on first call.
 * @param[out]    info
 *                         - = 0: successful exit.
 */
void slaqz0(
    const char* wants,
    const char* wantq,
    const char* wantz,
    const INT n,
    const INT ilo,
    const INT ihi,
    f32* restrict A,
    const INT lda,
    f32* restrict B,
    const INT ldb,
    f32* restrict alphar,
    f32* restrict alphai,
    f32* restrict beta,
    f32* restrict Q,
    const INT ldq,
    f32* restrict Z,
    const INT ldz,
    f32* restrict work,
    const INT lwork,
    const INT rec,
    INT* info)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;

    f32 smlnum, ulp, eshift, safmin, c1, s1;
    f32 temp, swap, bnorm, btol;
    INT istart, istop, iiter, maxit, istart2, k, nshifts;
    INT nblock, nw, nmin, nibble, n_undeflated, n_deflated;
    INT ns, sweep_info, shiftpos, lworkreq, k2, istartm;
    INT istopm, iwants, iwantq, iwantz, norm_info, aed_info;
    INT nwr, nbr, nsr, itemp1, itemp2, rcost, i;
    INT ilschur = 0, ilq = 0, ilz = 0;
    char jbcmpz[4];

    /* Decode wantS, wantQ, wantZ */
    if (wants[0] == 'E' || wants[0] == 'e') {
        ilschur = 0;
        iwants = 1;
    } else if (wants[0] == 'S' || wants[0] == 's') {
        ilschur = 1;
        iwants = 2;
    } else {
        iwants = 0;
    }

    if (wantq[0] == 'N' || wantq[0] == 'n') {
        ilq = 0;
        iwantq = 1;
    } else if (wantq[0] == 'V' || wantq[0] == 'v') {
        ilq = 1;
        iwantq = 2;
    } else if (wantq[0] == 'I' || wantq[0] == 'i') {
        ilq = 1;
        iwantq = 3;
    } else {
        iwantq = 0;
    }

    if (wantz[0] == 'N' || wantz[0] == 'n') {
        ilz = 0;
        iwantz = 1;
    } else if (wantz[0] == 'V' || wantz[0] == 'v') {
        ilz = 1;
        iwantz = 2;
    } else if (wantz[0] == 'I' || wantz[0] == 'i') {
        ilz = 1;
        iwantz = 3;
    } else {
        iwantz = 0;
    }

    /* Check Argument Values */
    *info = 0;
    if (iwants == 0) {
        *info = -1;
    } else if (iwantq == 0) {
        *info = -2;
    } else if (iwantz == 0) {
        *info = -3;
    } else if (n < 0) {
        *info = -4;
    } else if (ilo < 0) {
        *info = -5;
    } else if (ihi > n - 1 || ihi < ilo - 1) {
        *info = -6;
    } else if (lda < n) {
        *info = -8;
    } else if (ldb < n) {
        *info = -10;
    } else if (ldq < 1 || (ilq && ldq < n)) {
        *info = -15;
    } else if (ldz < 1 || (ilz && ldz < n)) {
        *info = -17;
    }
    if (*info != 0) {
        xerbla("SLAQZ0", -(*info));
        return;
    }

    /* Quick return if possible */
    if (n <= 0) {
        work[0] = ONE;
        return;
    }

    /* Get the parameters */
    jbcmpz[0] = wants[0];
    jbcmpz[1] = wantq[0];
    jbcmpz[2] = wantz[0];
    jbcmpz[3] = '\0';

    nmin = iparmq(12, "SLAQZ0", jbcmpz, n, ilo + 1, ihi + 1, lwork);
    nwr = iparmq(13, "SLAQZ0", jbcmpz, n, ilo + 1, ihi + 1, lwork);
    nwr = (2 > nwr) ? 2 : nwr;
    {
        INT tmp = ihi - ilo + 1;
        INT tmp2 = (n - 1) / 3;
        tmp = (tmp < tmp2) ? tmp : tmp2;
        nwr = (nwr < tmp) ? nwr : tmp;
    }

    nibble = iparmq(14, "SLAQZ0", jbcmpz, n, ilo + 1, ihi + 1, lwork);

    nsr = iparmq(15, "SLAQZ0", jbcmpz, n, ilo + 1, ihi + 1, lwork);
    {
        INT tmp = (n + 6) / 9;
        INT tmp2 = ihi - ilo;
        nsr = (nsr < tmp) ? nsr : tmp;
        nsr = (nsr < tmp2) ? nsr : tmp2;
    }
    nsr = (2 > nsr - (nsr % 2)) ? 2 : nsr - (nsr % 2);

    rcost = iparmq(17, "SLAQZ0", jbcmpz, n, ilo + 1, ihi + 1, lwork);
    itemp1 = (INT)(nsr / sqrtf(1.0f + 2.0f * nsr / ((f32)rcost / 100.0f * n)));
    itemp1 = ((itemp1 - 1) / 4) * 4 + 4;
    nbr = nsr + itemp1;

    if (n < nmin || rec >= 2) {
        shgeqz(wants, wantq, wantz, n, ilo, ihi, A, lda, B, ldb,
               alphar, alphai, beta, Q, ldq, Z, ldz, work, lwork, info);
        return;
    }

    /* Find out required workspace */

    /* Workspace query to slaqz3 */
    nw = (nwr > nmin) ? nwr : nmin;
    slaqz3(ilschur, ilq, ilz, n, ilo, ihi, nw, A, lda, B, ldb,
           Q, ldq, Z, ldz, &n_undeflated, &n_deflated, alphar, alphai, beta,
           NULL, nw, NULL, nw, work, -1, rec, &aed_info);
    itemp1 = (INT)work[0];

    /* Workspace query to slaqz4 */
    slaqz4(ilschur, ilq, ilz, n, ilo, ihi, nsr, nbr, alphar, alphai, beta,
           A, lda, B, ldb, Q, ldq, Z, ldz, NULL, nbr, NULL, nbr, work, -1,
           &sweep_info);
    itemp2 = (INT)work[0];

    lworkreq = (itemp1 + 2 * nw * nw > itemp2 + 2 * nbr * nbr) ?
               itemp1 + 2 * nw * nw : itemp2 + 2 * nbr * nbr;
    if (lwork == -1) {
        work[0] = (f32)lworkreq;
        return;
    } else if (lwork < lworkreq) {
        *info = -19;
    }
    if (*info != 0) {
        xerbla("SLAQZ0", *info);
        return;
    }

    /* Initialize Q and Z */
    if (iwantq == 3) slaset("F", n, n, ZERO, ONE, Q, ldq);
    if (iwantz == 3) slaset("F", n, n, ZERO, ONE, Z, ldz);

    /* Get machine constants */
    safmin = slamch("S");
    (void)(ONE / safmin);  /* safmax computed in Fortran but unused */
    ulp = slamch("P");
    smlnum = safmin * ((f32)n / ulp);

    bnorm = slanhs("F", ihi - ilo + 1, &B[ilo + ilo * ldb], ldb, work);
    btol = (safmin > ulp * bnorm) ? safmin : ulp * bnorm;

    istart = ilo;
    istop = ihi;
    maxit = 3 * (ihi - ilo + 1);
    eshift = ZERO;
    INT ld = 0;

    for (iiter = 0; iiter < maxit; iiter++) {
        if (iiter >= maxit - 1) {
            *info = istop + 1;
            goto label80;
        }
        if (istart + 1 >= istop) {
            break;
        }

        /* Check deflations at the end */
        {
            f32 tmp = fabsf(A[(istop - 1) + (istop - 1) * lda]) +
                         fabsf(A[(istop - 2) + (istop - 2) * lda]);
            f32 thresh = (smlnum > ulp * tmp) ? smlnum : ulp * tmp;
            if (fabsf(A[(istop - 1) + (istop - 2) * lda]) <= thresh) {
                A[(istop - 1) + (istop - 2) * lda] = ZERO;
                istop = istop - 2;
                ld = 0;
                eshift = ZERO;
            } else {
                tmp = fabsf(A[istop + istop * lda]) + fabsf(A[(istop - 1) + (istop - 1) * lda]);
                thresh = (smlnum > ulp * tmp) ? smlnum : ulp * tmp;
                if (fabsf(A[istop + (istop - 1) * lda]) <= thresh) {
                    A[istop + (istop - 1) * lda] = ZERO;
                    istop = istop - 1;
                    ld = 0;
                    eshift = ZERO;
                }
            }
        }

        /* Check deflations at the start */
        {
            f32 tmp = fabsf(A[(istart + 1) + (istart + 1) * lda]) +
                         fabsf(A[(istart + 2) + (istart + 2) * lda]);
            f32 thresh = (smlnum > ulp * tmp) ? smlnum : ulp * tmp;
            if (fabsf(A[(istart + 2) + (istart + 1) * lda]) <= thresh) {
                A[(istart + 2) + (istart + 1) * lda] = ZERO;
                istart = istart + 2;
                ld = 0;
                eshift = ZERO;
            } else {
                tmp = fabsf(A[istart + istart * lda]) + fabsf(A[(istart + 1) + (istart + 1) * lda]);
                thresh = (smlnum > ulp * tmp) ? smlnum : ulp * tmp;
                if (fabsf(A[(istart + 1) + istart * lda]) <= thresh) {
                    A[(istart + 1) + istart * lda] = ZERO;
                    istart = istart + 1;
                    ld = 0;
                    eshift = ZERO;
                }
            }
        }

        if (istart + 1 >= istop) {
            break;
        }

        /* Check interior deflations */
        istart2 = istart;
        for (k = istop; k >= istart + 1; k--) {
            f32 tmp = fabsf(A[k + k * lda]) + fabsf(A[(k - 1) + (k - 1) * lda]);
            f32 thresh = (smlnum > ulp * tmp) ? smlnum : ulp * tmp;
            if (fabsf(A[k + (k - 1) * lda]) <= thresh) {
                A[k + (k - 1) * lda] = ZERO;
                istart2 = k;
                break;
            }
        }

        /* Get range to apply rotations to */
        if (ilschur) {
            istartm = 0;
            istopm = n - 1;
        } else {
            istartm = istart2;
            istopm = istop;
        }

        /* Check infinite eigenvalues */
        k = istop;
        while (k >= istart2) {
            if (fabsf(B[k + k * ldb]) < btol) {
                /* A diagonal element of B is negligible, move it to the top and deflate it */
                for (k2 = k; k2 >= istart2 + 1; k2--) {
                    slartg(B[(k2 - 1) + k2 * ldb], B[(k2 - 1) + (k2 - 1) * ldb],
                           &c1, &s1, &temp);
                    B[(k2 - 1) + k2 * ldb] = temp;
                    B[(k2 - 1) + (k2 - 1) * ldb] = ZERO;

                    cblas_srot(k2 - 1 - istartm, &B[istartm + k2 * ldb], 1,
                               &B[istartm + (k2 - 1) * ldb], 1, c1, s1);
                    {
                        INT cnt = ((k2 + 1 < istop) ? k2 + 1 : istop) - istartm + 1;
                        cblas_srot(cnt, &A[istartm + k2 * lda], 1,
                                   &A[istartm + (k2 - 1) * lda], 1, c1, s1);
                    }
                    if (ilz) {
                        cblas_srot(n, &Z[0 + k2 * ldz], 1, &Z[0 + (k2 - 1) * ldz], 1, c1, s1);
                    }

                    if (k2 < istop) {
                        slartg(A[k2 + (k2 - 1) * lda], A[(k2 + 1) + (k2 - 1) * lda],
                               &c1, &s1, &temp);
                        A[k2 + (k2 - 1) * lda] = temp;
                        A[(k2 + 1) + (k2 - 1) * lda] = ZERO;

                        cblas_srot(istopm - k2 + 1, &A[k2 + k2 * lda], lda,
                                   &A[(k2 + 1) + k2 * lda], lda, c1, s1);
                        cblas_srot(istopm - k2 + 1, &B[k2 + k2 * ldb], ldb,
                                   &B[(k2 + 1) + k2 * ldb], ldb, c1, s1);
                        if (ilq) {
                            cblas_srot(n, &Q[0 + k2 * ldq], 1, &Q[0 + (k2 + 1) * ldq], 1, c1, s1);
                        }
                    }
                }

                if (istart2 < istop) {
                    slartg(A[istart2 + istart2 * lda], A[(istart2 + 1) + istart2 * lda],
                           &c1, &s1, &temp);
                    A[istart2 + istart2 * lda] = temp;
                    A[(istart2 + 1) + istart2 * lda] = ZERO;

                    cblas_srot(istopm - (istart2 + 1) + 1, &A[istart2 + (istart2 + 1) * lda], lda,
                               &A[(istart2 + 1) + (istart2 + 1) * lda], lda, c1, s1);
                    cblas_srot(istopm - (istart2 + 1) + 1, &B[istart2 + (istart2 + 1) * ldb], ldb,
                               &B[(istart2 + 1) + (istart2 + 1) * ldb], ldb, c1, s1);
                    if (ilq) {
                        cblas_srot(n, &Q[0 + istart2 * ldq], 1, &Q[0 + (istart2 + 1) * ldq], 1, c1, s1);
                    }
                }

                istart2 = istart2 + 1;
            }
            k = k - 1;
        }

        /* istart2 now points to the top of the bottom right unreduced Hessenberg block */
        if (istart2 >= istop) {
            istop = istart2 - 1;
            ld = 0;
            eshift = ZERO;
            continue;
        }

        nw = nwr;
        nshifts = nsr;
        nblock = nbr;

        if (istop - istart2 + 1 < nmin) {
            /* Setting nw to the size of the subblock will make AED deflate all the eigenvalues */
            if (istop - istart + 1 < nmin) {
                nw = istop - istart + 1;
                istart2 = istart;
            } else {
                nw = istop - istart2 + 1;
            }
        }

        /* Time for AED */
        slaqz3(ilschur, ilq, ilz, n, istart2, istop, nw, A, lda, B, ldb,
               Q, ldq, Z, ldz, &n_undeflated, &n_deflated, alphar, alphai, beta,
               work, nw, &work[nw * nw], nw, &work[2 * nw * nw], lwork - 2 * nw * nw,
               rec, &aed_info);

        if (n_deflated > 0) {
            istop = istop - n_deflated;
            ld = 0;
            eshift = ZERO;
        }

        if (100 * n_deflated > nibble * (n_deflated + n_undeflated) ||
            istop - istart2 + 1 < nmin) {
            /* AED has uncovered many eigenvalues. Skip a QZ sweep and run AED again. */
            continue;
        }

        ld = ld + 1;

        ns = nshifts;
        if (ns > istop - istart2) ns = istop - istart2;
        if (ns > n_undeflated) ns = n_undeflated;
        shiftpos = istop - n_undeflated + 1;

        /* Shuffle shifts to put double shifts in front */
        for (i = shiftpos; i <= shiftpos + n_undeflated - 1; i += 2) {
            if (alphai[i] != -alphai[i + 1]) {
                swap = alphar[i];
                alphar[i] = alphar[i + 1];
                alphar[i + 1] = alphar[i + 2];
                alphar[i + 2] = swap;

                swap = alphai[i];
                alphai[i] = alphai[i + 1];
                alphai[i + 1] = alphai[i + 2];
                alphai[i + 2] = swap;

                swap = beta[i];
                beta[i] = beta[i + 1];
                beta[i + 1] = beta[i + 2];
                beta[i + 2] = swap;
            }
        }

        if ((ld % 6) == 0) {
            /* Exceptional shift */
            if (((f32)maxit * safmin) * fabsf(A[istop + (istop - 1) * lda]) <
                fabsf(A[(istop - 1) + (istop - 1) * lda])) {
                eshift = A[istop + (istop - 1) * lda] / B[(istop - 1) + (istop - 1) * ldb];
            } else {
                eshift = eshift + ONE / (safmin * (f32)maxit);
            }
            alphar[shiftpos] = ONE;
            alphar[shiftpos + 1] = ZERO;
            alphai[shiftpos] = ZERO;
            alphai[shiftpos + 1] = ZERO;
            beta[shiftpos] = eshift;
            beta[shiftpos + 1] = eshift;
            ns = 2;
        }

        /* Time for a QZ sweep */
        slaqz4(ilschur, ilq, ilz, n, istart2, istop, ns, nblock,
               &alphar[shiftpos], &alphai[shiftpos], &beta[shiftpos],
               A, lda, B, ldb, Q, ldq, Z, ldz,
               work, nblock, &work[nblock * nblock], nblock,
               &work[2 * nblock * nblock], lwork - 2 * nblock * nblock,
               &sweep_info);
    }

label80:
    /* Call SHGEQZ to normalize the eigenvalue blocks and set the eigenvalues */
    shgeqz(wants, wantq, wantz, n, ilo, ihi, A, lda, B, ldb,
           alphar, alphai, beta, Q, ldq, Z, ldz, work, lwork, &norm_info);

    *info = norm_info;
}
