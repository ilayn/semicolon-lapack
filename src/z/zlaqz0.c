/**
 * @file zlaqz0.c
 * @brief ZLAQZ0 computes the eigenvalues of a complex matrix pair (H,T) using multishift QZ.
 */

#include <math.h>
#include <complex.h>
#include <cblas.h>
#include "semicolon_lapack_complex_double.h"

/**
 * ZLAQZ0 computes the eigenvalues of a complex matrix pair (H,T),
 * where H is an upper Hessenberg matrix and T is upper triangular,
 * using the single-shift QZ method.
 * Matrix pairs of this type are produced by the reduction to
 * generalized upper Hessenberg form of a complex matrix pair (A,B):
 *
 *    A = Q1*H*Z1**H,  B = Q1*T*Z1**H,
 *
 * as computed by ZGGHRD.
 *
 * If JOB='S', then the Hessenberg-triangular pair (H,T) is
 * also reduced to generalized Schur form,
 *
 *    H = Q*S*Z**H,  T = Q*P*Z**H,
 *
 * where Q and Z are unitary matrices, P and S are upper triangular.
 *
 * @param[in]     wants    'E': Compute eigenvalues only; 'S': Compute Schur form.
 * @param[in]     wantq    'N': Q not computed; 'I': Q initialized; 'V': Q updated.
 * @param[in]     wantz    'N': Z not computed; 'I': Z initialized; 'V': Z updated.
 * @param[in]     n        The order of the matrices A, B, Q, and Z. n >= 0.
 * @param[in]     ilo      0-based lower bound of active submatrix.
 * @param[in]     ihi      0-based upper bound of active submatrix.
 * @param[in,out] A        Complex array, dimension (lda, n). Upper Hessenberg matrix.
 * @param[in]     lda      Leading dimension of A.
 * @param[in,out] B        Complex array, dimension (ldb, n). Upper triangular matrix.
 * @param[in]     ldb      Leading dimension of B.
 * @param[out]    alpha    Complex array, dimension (n). Eigenvalue numerators.
 * @param[out]    beta     Complex array, dimension (n). Eigenvalue denominators.
 * @param[in,out] Q        Complex array, dimension (ldq, n). Left Schur vectors.
 * @param[in]     ldq      Leading dimension of Q.
 * @param[in,out] Z        Complex array, dimension (ldz, n). Right Schur vectors.
 * @param[in]     ldz      Leading dimension of Z.
 * @param[out]    work     Complex workspace array, dimension (max(1, lwork)).
 * @param[in]     lwork    Dimension of work. If lwork = -1, workspace query.
 * @param[out]    rwork    Double precision array, dimension (n).
 * @param[in]     rec      Current recursion level. Should be set to 0 on first call.
 * @param[out]    info     = 0: successful exit.
 *                         < 0: if info = -i, the i-th argument had an illegal value.
 *                         = 1,...,N: the QZ iteration did not converge.
 */
void zlaqz0(
    const char* wants,
    const char* wantq,
    const char* wantz,
    const int n,
    const int ilo,
    const int ihi,
    c128* const restrict A,
    const int lda,
    c128* const restrict B,
    const int ldb,
    c128* const restrict alpha,
    c128* const restrict beta,
    c128* const restrict Q,
    const int ldq,
    c128* const restrict Z,
    const int ldz,
    c128* const restrict work,
    const int lwork,
    f64* const restrict rwork,
    const int rec,
    int* info)
{
    const c128 CZERO = CMPLX(0.0, 0.0);
    const c128 CONE = CMPLX(1.0, 0.0);
    const f64 ONE = 1.0;

    f64 smlnum, ulp, safmin, c1;
    f64 bnorm, btol;
    c128 eshift, s1, temp;
    int istart, istop, iiter, maxit, istart2, k, nshifts;
    int nblock, nw, nmin, nibble, n_undeflated, n_deflated;
    int ns, sweep_info, shiftpos, lworkreq, k2, istartm;
    int istopm, iwants, iwantq, iwantz, norm_info, aed_info;
    int nwr, nbr, nsr, itemp1, itemp2, rcost;
    int ilschur = 0, ilq = 0, ilz = 0;
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
        xerbla("ZLAQZ0", -(*info));
        return;
    }

    /* Quick return if possible */
    if (n <= 0) {
        work[0] = CMPLX(1.0, 0.0);
        return;
    }

    /* Get the parameters */
    jbcmpz[0] = wants[0];
    jbcmpz[1] = wantq[0];
    jbcmpz[2] = wantz[0];
    jbcmpz[3] = '\0';

    nmin = iparmq(12, "ZLAQZ0", jbcmpz, n, ilo + 1, ihi + 1, lwork);
    nwr = iparmq(13, "ZLAQZ0", jbcmpz, n, ilo + 1, ihi + 1, lwork);
    nwr = (2 > nwr) ? 2 : nwr;
    {
        int tmp = ihi - ilo + 1;
        int tmp2 = (n - 1) / 3;
        tmp = (tmp < tmp2) ? tmp : tmp2;
        nwr = (nwr < tmp) ? nwr : tmp;
    }

    nibble = iparmq(14, "ZLAQZ0", jbcmpz, n, ilo + 1, ihi + 1, lwork);

    nsr = iparmq(15, "ZLAQZ0", jbcmpz, n, ilo + 1, ihi + 1, lwork);
    {
        int tmp = (n + 6) / 9;
        int tmp2 = ihi - ilo;
        nsr = (nsr < tmp) ? nsr : tmp;
        nsr = (nsr < tmp2) ? nsr : tmp2;
    }
    nsr = (2 > nsr - (nsr % 2)) ? 2 : nsr - (nsr % 2);

    rcost = iparmq(17, "ZLAQZ0", jbcmpz, n, ilo + 1, ihi + 1, lwork);
    itemp1 = (int)(nsr / sqrt(1.0 + 2.0 * nsr / ((f64)rcost / 100.0 * n)));
    itemp1 = ((itemp1 - 1) / 4) * 4 + 4;
    nbr = nsr + itemp1;

    if (n < nmin || rec >= 2) {
        zhgeqz(wants, wantq, wantz, n, ilo, ihi, A, lda, B, ldb,
               alpha, beta, Q, ldq, Z, ldz, work, lwork, rwork, info);
        return;
    }

    /* Find out required workspace */

    /* Workspace query to ZLAQZ2 */
    nw = (nwr > nmin) ? nwr : nmin;
    zlaqz2(ilschur, ilq, ilz, n, ilo, ihi, nw, A, lda, B, ldb,
           Q, ldq, Z, ldz, &n_undeflated, &n_deflated, alpha, beta,
           work, nw, work, nw, work, -1, rwork, rec, &aed_info);
    itemp1 = (int)creal(work[0]);

    /* Workspace query to ZLAQZ3 */
    zlaqz3(ilschur, ilq, ilz, n, ilo, ihi, nsr, nbr, alpha, beta,
           A, lda, B, ldb, Q, ldq, Z, ldz, work, nbr, work, nbr,
           work, -1, &sweep_info);
    itemp2 = (int)creal(work[0]);

    lworkreq = (itemp1 + 2 * nw * nw > itemp2 + 2 * nbr * nbr) ?
               itemp1 + 2 * nw * nw : itemp2 + 2 * nbr * nbr;
    if (lwork == -1) {
        work[0] = CMPLX((f64)lworkreq, 0.0);
        return;
    } else if (lwork < lworkreq) {
        *info = -19;
    }
    if (*info != 0) {
        xerbla("ZLAQZ0", -(*info));
        return;
    }

    /* Initialize Q and Z */
    if (iwantq == 3) zlaset("F", n, n, CZERO, CONE, Q, ldq);
    if (iwantz == 3) zlaset("F", n, n, CZERO, CONE, Z, ldz);

    /* Get machine constants */
    safmin = dlamch("S");
    (void)(ONE / safmin);
    ulp = dlamch("P");
    smlnum = safmin * ((f64)n / ulp);

    bnorm = zlanhs("F", ihi - ilo + 1, &B[ilo + ilo * ldb], ldb, rwork);
    btol = (safmin > ulp * bnorm) ? safmin : ulp * bnorm;

    istart = ilo;
    istop = ihi;
    maxit = 30 * (ihi - ilo + 1);
    eshift = CZERO;
    int ld = 0;

    for (iiter = 0; iiter < maxit; iiter++) {
        if (iiter >= maxit - 1) {
            *info = istop + 1;
            goto label80;
        }
        if (istart + 1 >= istop) {
            istop = istart;
            break;
        }

        /* Check deflations at the end */
        if (cabs(A[istop + (istop - 1) * lda]) <=
            fmax(smlnum, ulp * (cabs(A[istop + istop * lda]) +
                                cabs(A[(istop - 1) + (istop - 1) * lda])))) {
            A[istop + (istop - 1) * lda] = CZERO;
            istop = istop - 1;
            ld = 0;
            eshift = CZERO;
        }

        /* Check deflations at the start */
        if (cabs(A[(istart + 1) + istart * lda]) <=
            fmax(smlnum, ulp * (cabs(A[istart + istart * lda]) +
                                cabs(A[(istart + 1) + (istart + 1) * lda])))) {
            A[(istart + 1) + istart * lda] = CZERO;
            istart = istart + 1;
            ld = 0;
            eshift = CZERO;
        }

        if (istart + 1 >= istop) {
            break;
        }

        /* Check interior deflations */
        istart2 = istart;
        for (k = istop; k >= istart + 1; k--) {
            if (cabs(A[k + (k - 1) * lda]) <=
                fmax(smlnum, ulp * (cabs(A[k + k * lda]) +
                                    cabs(A[(k - 1) + (k - 1) * lda])))) {
                A[k + (k - 1) * lda] = CZERO;
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

        /* Check infinite eigenvalues, this is done without blocking so might
           slow down the method when many infinite eigenvalues are present */
        k = istop;
        while (k >= istart2) {

            if (cabs(B[k + k * ldb]) < btol) {
                /* A diagonal element of B is negligible, move it
                   to the top and deflate it */
                for (k2 = k; k2 >= istart2 + 1; k2--) {
                    zlartg(B[(k2 - 1) + k2 * ldb], B[(k2 - 1) + (k2 - 1) * ldb],
                           &c1, &s1, &temp);
                    B[(k2 - 1) + k2 * ldb] = temp;
                    B[(k2 - 1) + (k2 - 1) * ldb] = CZERO;

                    zrot(k2 - 1 - istartm, &B[istartm + k2 * ldb], 1,
                         &B[istartm + (k2 - 1) * ldb], 1, c1, s1);
                    {
                        int cnt = ((k2 + 1 < istop) ? k2 + 1 : istop) - istartm + 1;
                        zrot(cnt, &A[istartm + k2 * lda], 1,
                             &A[istartm + (k2 - 1) * lda], 1, c1, s1);
                    }
                    if (ilz) {
                        zrot(n, &Z[0 + k2 * ldz], 1, &Z[0 + (k2 - 1) * ldz], 1,
                             c1, s1);
                    }

                    if (k2 < istop) {
                        zlartg(A[k2 + (k2 - 1) * lda], A[(k2 + 1) + (k2 - 1) * lda],
                               &c1, &s1, &temp);
                        A[k2 + (k2 - 1) * lda] = temp;
                        A[(k2 + 1) + (k2 - 1) * lda] = CZERO;

                        zrot(istopm - k2 + 1, &A[k2 + k2 * lda], lda,
                             &A[(k2 + 1) + k2 * lda], lda, c1, s1);
                        zrot(istopm - k2 + 1, &B[k2 + k2 * ldb], ldb,
                             &B[(k2 + 1) + k2 * ldb], ldb, c1, s1);
                        if (ilq) {
                            zrot(n, &Q[0 + k2 * ldq], 1, &Q[0 + (k2 + 1) * ldq], 1,
                                 c1, conj(s1));
                        }
                    }
                }

                if (istart2 < istop) {
                    zlartg(A[istart2 + istart2 * lda], A[(istart2 + 1) + istart2 * lda],
                           &c1, &s1, &temp);
                    A[istart2 + istart2 * lda] = temp;
                    A[(istart2 + 1) + istart2 * lda] = CZERO;

                    zrot(istopm - (istart2 + 1) + 1, &A[istart2 + (istart2 + 1) * lda], lda,
                         &A[(istart2 + 1) + (istart2 + 1) * lda], lda, c1, s1);
                    zrot(istopm - (istart2 + 1) + 1, &B[istart2 + (istart2 + 1) * ldb], ldb,
                         &B[(istart2 + 1) + (istart2 + 1) * ldb], ldb, c1, s1);
                    if (ilq) {
                        zrot(n, &Q[0 + istart2 * ldq], 1, &Q[0 + (istart2 + 1) * ldq], 1,
                             c1, conj(s1));
                    }
                }

                istart2 = istart2 + 1;
            }
            k = k - 1;
        }

        /* istart2 now points to the top of the bottom right
           unreduced Hessenberg block */
        if (istart2 >= istop) {
            istop = istart2 - 1;
            ld = 0;
            eshift = CZERO;
            continue;
        }

        nw = nwr;
        nshifts = nsr;
        nblock = nbr;

        if (istop - istart2 + 1 < nmin) {
            /* Setting nw to the size of the subblock will make AED deflate
               all the eigenvalues. This is slightly more efficient than just
               using qz_small because the off diagonal part gets updated via BLAS. */
            if (istop - istart + 1 < nmin) {
                nw = istop - istart + 1;
                istart2 = istart;
            } else {
                nw = istop - istart2 + 1;
            }
        }

        /* Time for AED */
        zlaqz2(ilschur, ilq, ilz, n, istart2, istop, nw, A, lda, B, ldb,
               Q, ldq, Z, ldz, &n_undeflated, &n_deflated, alpha, beta,
               work, nw, &work[nw * nw], nw, &work[2 * nw * nw],
               lwork - 2 * nw * nw, rwork, rec, &aed_info);

        if (n_deflated > 0) {
            istop = istop - n_deflated;
            ld = 0;
            eshift = CZERO;
        }

        if (100 * n_deflated > nibble * (n_deflated + n_undeflated) ||
            istop - istart2 + 1 < nmin) {
            /* AED has uncovered many eigenvalues. Skip a QZ sweep and run
               AED again. */
            continue;
        }

        ld = ld + 1;

        ns = nshifts;
        if (ns > istop - istart2) ns = istop - istart2;
        if (ns > n_undeflated) ns = n_undeflated;
        shiftpos = istop - n_undeflated + 1;

        if ((ld % 6) == 0) {
            /* Exceptional shift.  Chosen for no particularly good reason. */
            if (((f64)maxit * safmin) * cabs(A[istop + (istop - 1) * lda]) <
                cabs(A[(istop - 1) + (istop - 1) * lda])) {
                eshift = A[istop + (istop - 1) * lda] /
                         B[(istop - 1) + (istop - 1) * ldb];
            } else {
                eshift = eshift + CONE / (safmin * (f64)maxit);
            }
            alpha[shiftpos] = CONE;
            beta[shiftpos] = eshift;
            ns = 1;
        }

        /* Time for a QZ sweep */
        zlaqz3(ilschur, ilq, ilz, n, istart2, istop, ns, nblock,
               &alpha[shiftpos], &beta[shiftpos], A, lda, B, ldb,
               Q, ldq, Z, ldz, work, nblock, &work[nblock * nblock],
               nblock, &work[2 * nblock * nblock],
               lwork - 2 * nblock * nblock, &sweep_info);
    }

    /* Call ZHGEQZ to normalize the eigenvalue blocks and set the eigenvalues
       If all the eigenvalues have been found, ZHGEQZ will not do any iterations
       and only normalize the blocks. In case of a rare convergence failure,
       the single shift might perform better. */
label80:
    zhgeqz(wants, wantq, wantz, n, ilo, ihi, A, lda, B, ldb,
           alpha, beta, Q, ldq, Z, ldz, work, lwork, rwork, &norm_info);

    *info = norm_info;
}
