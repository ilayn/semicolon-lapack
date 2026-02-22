/**
 * @file chgeqz.c
 * @brief CHGEQZ computes eigenvalues of a complex matrix pair (H,T) using the single-shift QZ method.
 */

#include "internal_build_defs.h"
#include <math.h>
#include <complex.h>
#include <cblas.h>
#include "semicolon_lapack_complex_single.h"

/**
 * CHGEQZ computes the eigenvalues of a complex matrix pair (H,T),
 * where H is an upper Hessenberg matrix and T is upper triangular,
 * using the single-shift QZ method.
 * Matrix pairs of this type are produced by the reduction to
 * generalized upper Hessenberg form of a complex matrix pair (A,B):
 *
 *    A = Q1*H*Z1**H,  B = Q1*T*Z1**H,
 *
 * as computed by CGGHRD.
 *
 * If JOB='S', then the Hessenberg-triangular pair (H,T) is
 * also reduced to generalized Schur form,
 *
 *    H = Q*S*Z**H,  T = Q*P*Z**H,
 *
 * where Q and Z are unitary matrices and S and P are upper triangular.
 *
 * @param[in]     job     = 'E': Compute eigenvalues only;
 *                        = 'S': Compute eigenvalues and the Schur form.
 * @param[in]     compq   = 'N': Left Schur vectors (Q) are not computed;
 *                        = 'I': Q is initialized to the unit matrix;
 *                        = 'V': Q must contain a unitary matrix Q1 on entry.
 * @param[in]     compz   = 'N': Right Schur vectors (Z) are not computed;
 *                        = 'I': Z is initialized to the unit matrix;
 *                        = 'V': Z must contain a unitary matrix Z1 on entry.
 * @param[in]     n       The order of the matrices H, T, Q, and Z. n >= 0.
 * @param[in]     ilo     0-based lower bound of active submatrix.
 * @param[in]     ihi     0-based upper bound of active submatrix.
 * @param[in,out] H       Complex array, dimension (ldh, n). Upper Hessenberg matrix.
 * @param[in]     ldh     Leading dimension of H. ldh >= max(1,n).
 * @param[in,out] T       Complex array, dimension (ldt, n). Upper triangular matrix.
 * @param[in]     ldt     Leading dimension of T. ldt >= max(1,n).
 * @param[out]    alpha   Complex array, dimension (n). Eigenvalue numerators.
 * @param[out]    beta    Complex array, dimension (n). Eigenvalue denominators.
 * @param[in,out] Q       Complex array, dimension (ldq, n). Left Schur vectors.
 * @param[in]     ldq     Leading dimension of Q. ldq >= 1; ldq >= n if COMPQ='V' or 'I'.
 * @param[in,out] Z       Complex array, dimension (ldz, n). Right Schur vectors.
 * @param[in]     ldz     Leading dimension of Z. ldz >= 1; ldz >= n if COMPZ='V' or 'I'.
 * @param[out]    work    Complex workspace array, dimension (max(1,lwork)).
 * @param[in]     lwork   Dimension of work. lwork >= max(1,n). If lwork = -1, workspace query.
 * @param[out]    rwork   Single precision array, dimension (n).
 * @param[out]    info    = 0: successful exit.
 *                        < 0: if info = -i, the i-th argument had an illegal value.
 *                        = 1,...,N: the QZ iteration did not converge.
 *                        = N+1,...,2*N: the shift calculation failed.
 */
void chgeqz(
    const char* job,
    const char* compq,
    const char* compz,
    const INT n,
    const INT ilo,
    const INT ihi,
    c64* restrict H,
    const INT ldh,
    c64* restrict T,
    const INT ldt,
    c64* restrict alpha,
    c64* restrict beta,
    c64* restrict Q,
    const INT ldq,
    c64* restrict Z,
    const INT ldz,
    c64* restrict work,
    const INT lwork,
    f32* restrict rwork,
    INT* info)
{
    const c64 CZERO = CMPLXF(0.0f, 0.0f);
    const c64 CONE = CMPLXF(1.0f, 0.0f);
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;
    const f32 HALF = 0.5f;

    INT ilazr2, ilazro, ilq = 0, ilschr = 0, ilz = 0, lquery;
    INT icompq, icompz, ifirst, ifrstm, iiter, ilast;
    INT ilastm, in, ischur, istart, j, jc, jch, jiter;
    INT jr, maxit;
    f32 absb, anorm, ascale, atol, bnorm, bscale, btol;
    f32 c, safmin, temp, temp2, tempr, ulp;
    c64 ad11, ad12, ad21, ad22, abi22, abi12;
    c64 ctemp, ctemp2, ctemp3, eshift, s, shift;
    c64 signbc, u12, x, y;

    /* Decode JOB, COMPQ, COMPZ */

    if (job[0] == 'E' || job[0] == 'e') {
        ilschr = 0;
        ischur = 1;
    } else if (job[0] == 'S' || job[0] == 's') {
        ilschr = 1;
        ischur = 2;
    } else {
        ilschr = 1;
        ischur = 0;
    }

    if (compq[0] == 'N' || compq[0] == 'n') {
        ilq = 0;
        icompq = 1;
    } else if (compq[0] == 'V' || compq[0] == 'v') {
        ilq = 1;
        icompq = 2;
    } else if (compq[0] == 'I' || compq[0] == 'i') {
        ilq = 1;
        icompq = 3;
    } else {
        ilq = 1;
        icompq = 0;
    }

    if (compz[0] == 'N' || compz[0] == 'n') {
        ilz = 0;
        icompz = 1;
    } else if (compz[0] == 'V' || compz[0] == 'v') {
        ilz = 1;
        icompz = 2;
    } else if (compz[0] == 'I' || compz[0] == 'i') {
        ilz = 1;
        icompz = 3;
    } else {
        ilz = 1;
        icompz = 0;
    }

    /* Check Argument Values */
    *info = 0;
    work[0] = CMPLXF((f32)(1 > n ? 1 : n), 0.0f);
    lquery = (lwork == -1);
    if (ischur == 0) {
        *info = -1;
    } else if (icompq == 0) {
        *info = -2;
    } else if (icompz == 0) {
        *info = -3;
    } else if (n < 0) {
        *info = -4;
    } else if (ilo < 0) {
        *info = -5;
    } else if (ihi > n - 1 || ihi < ilo - 1) {
        *info = -6;
    } else if (ldh < n) {
        *info = -8;
    } else if (ldt < n) {
        *info = -10;
    } else if (ldq < 1 || (ilq && ldq < n)) {
        *info = -14;
    } else if (ldz < 1 || (ilz && ldz < n)) {
        *info = -16;
    } else if (lwork < (1 > n ? 1 : n) && !lquery) {
        *info = -18;
    }
    if (*info != 0) {
        xerbla("CHGEQZ", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    /* Quick return if possible */
    if (n <= 0) {
        work[0] = CONE;
        return;
    }

    /* Initialize Q and Z */

    if (icompq == 3)
        claset("F", n, n, CZERO, CONE, Q, ldq);
    if (icompz == 3)
        claset("F", n, n, CZERO, CONE, Z, ldz);

    /* Machine Constants */

    in = ihi + 1 - ilo;
    safmin = slamch("S");
    ulp = slamch("E") * slamch("B");
    anorm = clanhs("F", in, &H[ilo + ilo * ldh], ldh, rwork);
    bnorm = clanhs("F", in, &T[ilo + ilo * ldt], ldt, rwork);
    atol = (safmin > ulp * anorm) ? safmin : ulp * anorm;
    btol = (safmin > ulp * bnorm) ? safmin : ulp * bnorm;
    ascale = ONE / ((safmin > anorm) ? safmin : anorm);
    bscale = ONE / ((safmin > bnorm) ? safmin : bnorm);

    /* Set Eigenvalues IHI+1:N */
    for (j = ihi + 1; j < n; j++) {
        absb = cabsf(T[j + j * ldt]);
        if (absb > safmin) {
            signbc = conjf(T[j + j * ldt] / absb);
            T[j + j * ldt] = CMPLXF(absb, 0.0f);
            if (ilschr) {
                cblas_cscal(j, &signbc, &T[0 + j * ldt], 1);
                cblas_cscal(j + 1, &signbc, &H[0 + j * ldh], 1);
            } else {
                cblas_cscal(1, &signbc, &H[j + j * ldh], 1);
            }
            if (ilz)
                cblas_cscal(n, &signbc, &Z[0 + j * ldz], 1);
        } else {
            T[j + j * ldt] = CZERO;
        }
        alpha[j] = H[j + j * ldh];
        beta[j] = T[j + j * ldt];
    }

    /* If IHI < ILO, skip QZ steps */

    if (ihi < ilo)
        goto L190;

    /* MAIN QZ ITERATION LOOP */

    ilast = ihi;
    if (ilschr) {
        ifrstm = 0;
        ilastm = n - 1;
    } else {
        ifrstm = ilo;
        ilastm = ihi;
    }
    iiter = 0;
    eshift = CZERO;
    maxit = 30 * (ihi - ilo + 1);

    for (jiter = 1; jiter <= maxit; jiter++) {

        /* Check for too many iterations. */

        if (jiter > maxit)
            goto L180;

        /* Split the matrix if possible.
         *
         * Two tests:
         *    1: H(j,j-1)=0  or  j=ILO
         *    2: T(j,j)=0
         *
         * Special case: j=ILAST
         */

        if (ilast == ilo) {
            goto L60;
        } else {
            if (cabs1f(H[ilast + (ilast - 1) * ldh]) <=
                fmaxf(safmin, ulp * (cabs1f(H[ilast + ilast * ldh]) +
                                    cabs1f(H[(ilast - 1) + (ilast - 1) * ldh])))) {
                H[ilast + (ilast - 1) * ldh] = CZERO;
                goto L60;
            }
        }

        if (cabsf(T[ilast + ilast * ldt]) <= btol) {
            T[ilast + ilast * ldt] = CZERO;
            goto L50;
        }

        /* General case: j<ILAST */

        for (j = ilast - 1; j >= ilo; j--) {

            /* Test 1: for H(j,j-1)=0 or j=ILO */

            if (j == ilo) {
                ilazro = 1;
            } else {
                if (cabs1f(H[j + (j - 1) * ldh]) <=
                    fmaxf(safmin, ulp * (cabs1f(H[j + j * ldh]) +
                                        cabs1f(H[(j - 1) + (j - 1) * ldh])))) {
                    H[j + (j - 1) * ldh] = CZERO;
                    ilazro = 1;
                } else {
                    ilazro = 0;
                }
            }

            /* Test 2: for T(j,j)=0 */

            if (cabsf(T[j + j * ldt]) < btol) {
                T[j + j * ldt] = CZERO;

                /* Test 1a: Check for 2 consecutive small subdiagonals in A */

                ilazr2 = 0;
                if (!ilazro) {
                    if (cabs1f(H[j + (j - 1) * ldh]) * (ascale * cabs1f(H[(j + 1) + j * ldh]))
                        <= cabs1f(H[j + j * ldh]) * (ascale * atol))
                        ilazr2 = 1;
                }

                if (ilazro || ilazr2) {
                    for (jch = j; jch < ilast; jch++) {
                        ctemp = H[jch + jch * ldh];
                        clartg(ctemp, H[(jch + 1) + jch * ldh], &c, &s,
                               &H[jch + jch * ldh]);
                        H[(jch + 1) + jch * ldh] = CZERO;
                        crot(ilastm - jch, &H[jch + (jch + 1) * ldh], ldh,
                             &H[(jch + 1) + (jch + 1) * ldh], ldh, c, s);
                        crot(ilastm - jch, &T[jch + (jch + 1) * ldt], ldt,
                             &T[(jch + 1) + (jch + 1) * ldt], ldt, c, s);
                        if (ilq)
                            crot(n, &Q[0 + jch * ldq], 1, &Q[0 + (jch + 1) * ldq], 1,
                                 c, conjf(s));
                        if (ilazr2)
                            H[jch + (jch - 1) * ldh] = H[jch + (jch - 1) * ldh] * c;
                        ilazr2 = 0;
                        if (cabs1f(T[(jch + 1) + (jch + 1) * ldt]) >= btol) {
                            if (jch + 1 >= ilast) {
                                goto L60;
                            } else {
                                ifirst = jch + 1;
                                goto L70;
                            }
                        }
                        T[(jch + 1) + (jch + 1) * ldt] = CZERO;
                    }
                    goto L50;
                } else {

                    /* Only test 2 passed -- chase the zero to T(ILAST,ILAST)
                     * Then process as in the case T(ILAST,ILAST)=0
                     */

                    for (jch = j; jch < ilast; jch++) {
                        ctemp = T[jch + (jch + 1) * ldt];
                        clartg(ctemp, T[(jch + 1) + (jch + 1) * ldt], &c, &s,
                               &T[jch + (jch + 1) * ldt]);
                        T[(jch + 1) + (jch + 1) * ldt] = CZERO;
                        if (jch < ilastm - 1)
                            crot(ilastm - jch - 1, &T[jch + (jch + 2) * ldt], ldt,
                                 &T[(jch + 1) + (jch + 2) * ldt], ldt, c, s);
                        crot(ilastm - jch + 2, &H[jch + (jch - 1) * ldh], ldh,
                             &H[(jch + 1) + (jch - 1) * ldh], ldh, c, s);
                        if (ilq)
                            crot(n, &Q[0 + jch * ldq], 1, &Q[0 + (jch + 1) * ldq], 1,
                                 c, conjf(s));
                        ctemp = H[(jch + 1) + jch * ldh];
                        clartg(ctemp, H[(jch + 1) + (jch - 1) * ldh], &c, &s,
                               &H[(jch + 1) + jch * ldh]);
                        H[(jch + 1) + (jch - 1) * ldh] = CZERO;
                        crot(jch + 1 - ifrstm, &H[ifrstm + jch * ldh], 1,
                             &H[ifrstm + (jch - 1) * ldh], 1, c, s);
                        crot(jch - ifrstm, &T[ifrstm + jch * ldt], 1,
                             &T[ifrstm + (jch - 1) * ldt], 1, c, s);
                        if (ilz)
                            crot(n, &Z[0 + jch * ldz], 1, &Z[0 + (jch - 1) * ldz], 1,
                                 c, s);
                    }
                    goto L50;
                }
            } else if (ilazro) {

                /* Only test 1 passed -- work on J:ILAST */

                ifirst = j;
                goto L70;
            }

            /* Neither test passed -- try next J */

        }

        /* (Drop-through is "impossible") */

        *info = 2 * n + 1;
        goto L210;

        /* T(ILAST,ILAST)=0 -- clear H(ILAST,ILAST-1) to split off a
         * 1x1 block.
         */

L50:
        ctemp = H[ilast + ilast * ldh];
        clartg(ctemp, H[ilast + (ilast - 1) * ldh], &c, &s,
               &H[ilast + ilast * ldh]);
        H[ilast + (ilast - 1) * ldh] = CZERO;
        crot(ilast - ifrstm, &H[ifrstm + ilast * ldh], 1,
             &H[ifrstm + (ilast - 1) * ldh], 1, c, s);
        crot(ilast - ifrstm, &T[ifrstm + ilast * ldt], 1,
             &T[ifrstm + (ilast - 1) * ldt], 1, c, s);
        if (ilz)
            crot(n, &Z[0 + ilast * ldz], 1, &Z[0 + (ilast - 1) * ldz], 1, c, s);

        /* H(ILAST,ILAST-1)=0 -- Standardize B, set ALPHA and BETA */

L60:
        absb = cabsf(T[ilast + ilast * ldt]);
        if (absb > safmin) {
            signbc = conjf(T[ilast + ilast * ldt] / absb);
            T[ilast + ilast * ldt] = CMPLXF(absb, 0.0f);
            if (ilschr) {
                cblas_cscal(ilast - ifrstm, &signbc, &T[ifrstm + ilast * ldt], 1);
                cblas_cscal(ilast + 1 - ifrstm, &signbc, &H[ifrstm + ilast * ldh], 1);
            } else {
                cblas_cscal(1, &signbc, &H[ilast + ilast * ldh], 1);
            }
            if (ilz)
                cblas_cscal(n, &signbc, &Z[0 + ilast * ldz], 1);
        } else {
            T[ilast + ilast * ldt] = CZERO;
        }
        alpha[ilast] = H[ilast + ilast * ldh];
        beta[ilast] = T[ilast + ilast * ldt];

        /* Go to next block -- exit if finished. */

        ilast = ilast - 1;
        if (ilast < ilo)
            goto L190;

        /* Reset counters */

        iiter = 0;
        eshift = CZERO;
        if (!ilschr) {
            ilastm = ilast;
            if (ifrstm > ilast)
                ifrstm = ilo;
        }
        continue;

        /* QZ step
         *
         * This iteration only involves rows/columns IFIRST:ILAST.  We
         * assume IFIRST < ILAST, and that the diagonal of B is non-zero.
         */

L70:
        iiter = iiter + 1;
        if (!ilschr) {
            ifrstm = ifirst;
        }

        /* Compute the Shift.
         *
         * At this point, IFIRST < ILAST, and the diagonal elements of
         * T(IFIRST:ILAST,IFIRST,ILAST) are larger than BTOL (in
         * magnitude)
         */

        if ((iiter / 10) * 10 != iiter) {

            /* The Wilkinson shift (AEP p.512), i.e., the eigenvalue of
             * the bottom-right 2x2 block of A inv(B) which is nearest to
             * the bottom-right element.
             *
             * We factor B as U*D, where U has unit diagonals, and
             * compute (A*inv(D))*inv(U).
             */

            u12 = (bscale * T[(ilast - 1) + ilast * ldt]) /
                  (bscale * T[ilast + ilast * ldt]);
            ad11 = (ascale * H[(ilast - 1) + (ilast - 1) * ldh]) /
                   (bscale * T[(ilast - 1) + (ilast - 1) * ldt]);
            ad21 = (ascale * H[ilast + (ilast - 1) * ldh]) /
                   (bscale * T[(ilast - 1) + (ilast - 1) * ldt]);
            ad12 = (ascale * H[(ilast - 1) + ilast * ldh]) /
                   (bscale * T[ilast + ilast * ldt]);
            ad22 = (ascale * H[ilast + ilast * ldh]) /
                   (bscale * T[ilast + ilast * ldt]);
            abi22 = ad22 - u12 * ad21;
            abi12 = ad12 - u12 * ad11;

            shift = abi22;
            ctemp = csqrtf(abi12) * csqrtf(ad21);
            temp = cabs1f(ctemp);
            if (ctemp != ZERO) {
                x = HALF * (ad11 - shift);
                temp2 = cabs1f(x);
                temp = (temp > cabs1f(x)) ? temp : cabs1f(x);
                y = temp * csqrtf((x / temp) * (x / temp) + (ctemp / temp) * (ctemp / temp));
                if (temp2 > ZERO) {
                    if (crealf(x / temp2) * crealf(y) +
                        cimagf(x / temp2) * cimagf(y) < ZERO)
                        y = -y;
                }
                shift = shift - ctemp * cladiv(ctemp, (x + y));
            }
        } else {

            /* Exceptional shift.  Chosen for no particularly good reason. */

            if ((iiter / 20) * 20 == iiter &&
                bscale * cabs1f(T[ilast + ilast * ldt]) > safmin) {
                eshift = eshift + (ascale * H[ilast + ilast * ldh]) /
                                  (bscale * T[ilast + ilast * ldt]);
            } else {
                eshift = eshift + (ascale * H[ilast + (ilast - 1) * ldh]) /
                                  (bscale * T[(ilast - 1) + (ilast - 1) * ldt]);
            }
            shift = eshift;
        }

        /* Now check for two consecutive small subdiagonals. */

        for (j = ilast - 1; j >= ifirst + 1; j--) {
            istart = j;
            ctemp = ascale * H[j + j * ldh] - shift * (bscale * T[j + j * ldt]);
            temp = cabs1f(ctemp);
            temp2 = ascale * cabs1f(H[(j + 1) + j * ldh]);
            tempr = (temp > temp2) ? temp : temp2;
            if (tempr < ONE && tempr != ZERO) {
                temp = temp / tempr;
                temp2 = temp2 / tempr;
            }
            if (cabs1f(H[j + (j - 1) * ldh]) * temp2 <= temp * atol)
                goto L90;
        }

        istart = ifirst;
        ctemp = ascale * H[ifirst + ifirst * ldh] -
                shift * (bscale * T[ifirst + ifirst * ldt]);
L90:

        /* Do an implicit-shift QZ sweep.
         *
         * Initial Q
         */

        ctemp2 = ascale * H[(istart + 1) + istart * ldh];
        clartg(ctemp, ctemp2, &c, &s, &ctemp3);

        /* Sweep */

        for (j = istart; j < ilast; j++) {
            if (j > istart) {
                ctemp = H[j + (j - 1) * ldh];
                clartg(ctemp, H[(j + 1) + (j - 1) * ldh], &c, &s, &H[j + (j - 1) * ldh]);
                H[(j + 1) + (j - 1) * ldh] = CZERO;
            }

            for (jc = j; jc <= ilastm; jc++) {
                ctemp = c * H[j + jc * ldh] + s * H[(j + 1) + jc * ldh];
                H[(j + 1) + jc * ldh] = -conjf(s) * H[j + jc * ldh] + c * H[(j + 1) + jc * ldh];
                H[j + jc * ldh] = ctemp;
                ctemp2 = c * T[j + jc * ldt] + s * T[(j + 1) + jc * ldt];
                T[(j + 1) + jc * ldt] = -conjf(s) * T[j + jc * ldt] + c * T[(j + 1) + jc * ldt];
                T[j + jc * ldt] = ctemp2;
            }
            if (ilq) {
                for (jr = 0; jr < n; jr++) {
                    ctemp = c * Q[jr + j * ldq] + conjf(s) * Q[jr + (j + 1) * ldq];
                    Q[jr + (j + 1) * ldq] = -s * Q[jr + j * ldq] + c * Q[jr + (j + 1) * ldq];
                    Q[jr + j * ldq] = ctemp;
                }
            }

            ctemp = T[(j + 1) + (j + 1) * ldt];
            clartg(ctemp, T[(j + 1) + j * ldt], &c, &s, &T[(j + 1) + (j + 1) * ldt]);
            T[(j + 1) + j * ldt] = CZERO;

            {
                INT jrmax = (j + 2 < ilast) ? j + 2 : ilast;
                for (jr = ifrstm; jr <= jrmax; jr++) {
                    ctemp = c * H[jr + (j + 1) * ldh] + s * H[jr + j * ldh];
                    H[jr + j * ldh] = -conjf(s) * H[jr + (j + 1) * ldh] + c * H[jr + j * ldh];
                    H[jr + (j + 1) * ldh] = ctemp;
                }
            }
            for (jr = ifrstm; jr <= j; jr++) {
                ctemp = c * T[jr + (j + 1) * ldt] + s * T[jr + j * ldt];
                T[jr + j * ldt] = -conjf(s) * T[jr + (j + 1) * ldt] + c * T[jr + j * ldt];
                T[jr + (j + 1) * ldt] = ctemp;
            }
            if (ilz) {
                for (jr = 0; jr < n; jr++) {
                    ctemp = c * Z[jr + (j + 1) * ldz] + s * Z[jr + j * ldz];
                    Z[jr + j * ldz] = -conjf(s) * Z[jr + (j + 1) * ldz] + c * Z[jr + j * ldz];
                    Z[jr + (j + 1) * ldz] = ctemp;
                }
            }
        }

        continue;

    }

    /* Drop-through = non-convergence */

L180:
    *info = ilast + 1;
    goto L210;

    /* Successful completion of all QZ steps */

L190:

    /* Set Eigenvalues 1:ILO-1 */

    for (j = 0; j < ilo; j++) {
        absb = cabsf(T[j + j * ldt]);
        if (absb > safmin) {
            signbc = conjf(T[j + j * ldt] / absb);
            T[j + j * ldt] = CMPLXF(absb, 0.0f);
            if (ilschr) {
                cblas_cscal(j, &signbc, &T[0 + j * ldt], 1);
                cblas_cscal(j + 1, &signbc, &H[0 + j * ldh], 1);
            } else {
                cblas_cscal(1, &signbc, &H[j + j * ldh], 1);
            }
            if (ilz)
                cblas_cscal(n, &signbc, &Z[0 + j * ldz], 1);
        } else {
            T[j + j * ldt] = CZERO;
        }
        alpha[j] = H[j + j * ldh];
        beta[j] = T[j + j * ldt];
    }

    /* Normal Termination */

    *info = 0;

    /* Exit (other than argument error) -- return optimal workspace size */

L210:
    work[0] = CMPLXF((f32)n, 0.0f);
    return;
}
