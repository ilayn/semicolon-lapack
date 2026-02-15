/**
 * @file shgeqz.c
 * @brief SHGEQZ computes eigenvalues of a real matrix pair (H,T) using the double-shift QZ method.
 */

#include <math.h>
#include <cblas.h>
#include "semicolon_lapack_single.h"

/**
 * SHGEQZ computes the eigenvalues of a real matrix pair (H,T),
 * where H is an upper Hessenberg matrix and T is upper triangular,
 * using the double-shift QZ method.
 * Matrix pairs of this type are produced by the reduction to
 * generalized upper Hessenberg form of a real matrix pair (A,B):
 *
 *    A = Q1*H*Z1**T,  B = Q1*T*Z1**T,
 *
 * as computed by SGGHRD.
 *
 * If JOB='S', then the Hessenberg-triangular pair (H,T) is
 * also reduced to generalized Schur form,
 *
 *    H = Q*S*Z**T,  T = Q*P*Z**T,
 *
 * where Q and Z are orthogonal matrices, P is an upper triangular
 * matrix, and S is a quasi-triangular matrix with 1-by-1 and 2-by-2
 * diagonal blocks.
 *
 * @param[in]     job     = 'E': Compute eigenvalues only;
 *                        = 'S': Compute eigenvalues and the Schur form.
 * @param[in]     compq   = 'N': Left Schur vectors (Q) are not computed;
 *                        = 'I': Q is initialized to the unit matrix;
 *                        = 'V': Q must contain an orthogonal matrix Q1 on entry.
 * @param[in]     compz   = 'N': Right Schur vectors (Z) are not computed;
 *                        = 'I': Z is initialized to the unit matrix;
 *                        = 'V': Z must contain an orthogonal matrix Z1 on entry.
 * @param[in]     n       The order of the matrices H, T, Q, and Z. n >= 0.
 * @param[in]     ilo     See ihi.
 * @param[in]     ihi     ILO and IHI mark the rows and columns of H which are in
 *                        Hessenberg form. 0 <= ILO <= IHI <= N-1, if N > 0.
 * @param[in,out] H       Array of dimension (ldh, n). On entry, the N-by-N upper
 *                        Hessenberg matrix H. On exit, if JOB = 'S', H contains the
 *                        upper quasi-triangular matrix S.
 * @param[in]     ldh     The leading dimension of H. ldh >= max(1,n).
 * @param[in,out] T       Array of dimension (ldt, n). On entry, the N-by-N upper
 *                        triangular matrix T. On exit, if JOB = 'S', T contains the
 *                        upper triangular matrix P.
 * @param[in]     ldt     The leading dimension of T. ldt >= max(1,n).
 * @param[out]    alphar  Array of dimension (n). The real parts of alpha.
 * @param[out]    alphai  Array of dimension (n). The imaginary parts of alpha.
 * @param[out]    beta    Array of dimension (n). The scalars beta.
 * @param[in,out] Q       Array of dimension (ldq, n). The orthogonal matrix Q.
 * @param[in]     ldq     The leading dimension of Q. ldq >= 1; ldq >= n if COMPQ='V' or 'I'.
 * @param[in,out] Z       Array of dimension (ldz, n). The orthogonal matrix Z.
 * @param[in]     ldz     The leading dimension of Z. ldz >= 1; ldz >= n if COMPZ='V' or 'I'.
 * @param[out]    work    Workspace array of dimension (lwork).
 * @param[in]     lwork   The dimension of work. lwork >= max(1,n).
 *                        If lwork = -1, workspace query is performed.
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 *                         - = 1,...,N: the QZ iteration did not converge.
 *                         - = N+1,...,2*N: the shift calculation failed.
 */
void shgeqz(
    const char* job,
    const char* compq,
    const char* compz,
    const int n,
    const int ilo,
    const int ihi,
    f32* restrict H,
    const int ldh,
    f32* restrict T,
    const int ldt,
    f32* restrict alphar,
    f32* restrict alphai,
    f32* restrict beta,
    f32* restrict Q,
    const int ldq,
    f32* restrict Z,
    const int ldz,
    f32* restrict work,
    const int lwork,
    int* info)
{
    const f32 HALF = 0.5f;
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;
    const f32 SAFETY = 100.0f;

    int ilazr2, ilazro, ilpivt, ilq = 0, ilschr = 0, ilz = 0, lquery;
    int icompq, icompz, ifirst, ifrstm, iiter, ilast;
    int ilastm, in, ischur, istart, j, jc, jch, jiter;
    int jr, maxit;
    f32 a11, a12, a1i, a1r, a21, a22, a2i, a2r, ad11;
    f32 ad11l, ad12, ad12l, ad21, ad21l, ad22, ad22l;
    f32 ad32l, an, anorm, ascale, atol, b11, b1a, b1i;
    f32 b1r, b22, b2a, b2i, b2r, bn, bnorm, bscale;
    f32 btol, c, c11i, c11r, c12, c21, c22i, c22r, cl;
    f32 cq, cr, cz, eshift, s, s1, s1inv, s2, safmax;
    f32 safmin, scale, sl, sqi, sqr, sr, szi, szr, t1;
    f32 t2, t3, tau, temp, temp2, tempi, tempr, u1;
    f32 u12, u12l, u2, ulp, vs, w11, w12, w21, w22;
    f32 wabs, wi, wr, wr2;
    f32 v[3];

    /* Decode JOB, COMPQ, COMPZ */

    if (job[0] == 'E' || job[0] == 'e') {
        ilschr = 0;
        ischur = 1;
    } else if (job[0] == 'S' || job[0] == 's') {
        ilschr = 1;
        ischur = 2;
    } else {
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
        icompz = 0;
    }

    /* Check Argument Values */
    *info = 0;
    work[0] = (f32)((1 > n) ? 1 : n);
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
        *info = -15;
    } else if (ldz < 1 || (ilz && ldz < n)) {
        *info = -17;
    } else if (lwork < (1 > n ? 1 : n) && !lquery) {
        *info = -19;
    }
    if (*info != 0) {
        xerbla("SHGEQZ", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    /* Quick return if possible */
    if (n <= 0) {
        work[0] = 1.0f;
        return;
    }

    /* Initialize Q and Z */

    if (icompq == 3)
        slaset("F", n, n, ZERO, ONE, Q, ldq);
    if (icompz == 3)
        slaset("F", n, n, ZERO, ONE, Z, ldz);

    /* Machine Constants */

    in = ihi - ilo + 1;
    safmin = slamch("S");
    safmax = ONE / safmin;
    ulp = slamch("E") * slamch("B");
    anorm = slanhs("F", in, &H[ilo + ilo * ldh], ldh, work);
    bnorm = slanhs("F", in, &T[ilo + ilo * ldt], ldt, work);
    atol = (safmin > ulp * anorm) ? safmin : ulp * anorm;
    btol = (safmin > ulp * bnorm) ? safmin : ulp * bnorm;
    ascale = ONE / ((safmin > anorm) ? safmin : anorm);
    bscale = ONE / ((safmin > bnorm) ? safmin : bnorm);

    /* Set Eigenvalues IHI+1:N-1 */

    for (j = ihi + 1; j < n; j++) {
        if (T[j + j * ldt] < ZERO) {
            if (ilschr) {
                for (jr = 0; jr <= j; jr++) {
                    H[jr + j * ldh] = -H[jr + j * ldh];
                    T[jr + j * ldt] = -T[jr + j * ldt];
                }
            } else {
                H[j + j * ldh] = -H[j + j * ldh];
                T[j + j * ldt] = -T[j + j * ldt];
            }
            if (ilz) {
                for (jr = 0; jr < n; jr++)
                    Z[jr + j * ldz] = -Z[jr + j * ldz];
            }
        }
        alphar[j] = H[j + j * ldh];
        alphai[j] = ZERO;
        beta[j] = T[j + j * ldt];
    }

    /* If IHI < ILO, skip QZ steps */

    if (ihi < ilo)
        goto label_380;

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
    eshift = ZERO;
    maxit = 30 * (ihi - ilo + 1);

    for (jiter = 1; jiter <= maxit; jiter++) {

        /* Split the matrix if possible.
         *
         * Two tests:
         *    1: H(j,j-1)=0  or  j=ILO
         *    2: T(j,j)=0
         */

        int do_split = 0;
        int do_zero_t = 0;
        int do_qz_step = 0;
        ifirst = ilo;

        if (ilast == ilo) {

            /* Special case: j=ILAST */

            do_split = 1;
        } else {
            if (fabsf(H[ilast + (ilast - 1) * ldh]) <= fmaxf(safmin, ulp *
                (fabsf(H[ilast + ilast * ldh]) +
                 fabsf(H[(ilast - 1) + (ilast - 1) * ldh])))) {
                H[ilast + (ilast - 1) * ldh] = ZERO;
                do_split = 1;
            }
        }

        if (!do_split) {
            if (fabsf(T[ilast + ilast * ldt]) <= btol) {
                T[ilast + ilast * ldt] = ZERO;
                do_zero_t = 1;
            }
        }

        /* General case: j<ILAST */

        if (!do_split && !do_zero_t) {
            for (j = ilast - 1; j >= ilo; j--) {

                /* Test 1: for H(j,j-1)=0 or j=ILO */

                if (j == ilo) {
                    ilazro = 1;
                } else {
                    if (fabsf(H[j + (j - 1) * ldh]) <= fmaxf(safmin, ulp *
                        (fabsf(H[j + j * ldh]) +
                         fabsf(H[(j - 1) + (j - 1) * ldh])))) {
                        H[j + (j - 1) * ldh] = ZERO;
                        ilazro = 1;
                    } else {
                        ilazro = 0;
                    }
                }

                /* Test 2: for T(j,j)=0 */

                if (fabsf(T[j + j * ldt]) < btol) {
                    T[j + j * ldt] = ZERO;

                    /* Test 1a: Check for 2 consecutive small subdiagonals in A */

                    ilazr2 = 0;
                    if (!ilazro) {
                        temp = fabsf(H[j + (j - 1) * ldh]);
                        temp2 = fabsf(H[j + j * ldh]);
                        tempr = fmaxf(temp, temp2);
                        if (tempr < ONE && tempr != ZERO) {
                            temp = temp / tempr;
                            temp2 = temp2 / tempr;
                        }
                        if (temp * (ascale * fabsf(H[(j + 1) + j * ldh])) <=
                            temp2 * (ascale * atol))
                            ilazr2 = 1;
                    }

                    if (ilazro || ilazr2) {
                        for (jch = j; jch <= ilast - 1; jch++) {
                            temp = H[jch + jch * ldh];
                            slartg(temp, H[(jch + 1) + jch * ldh], &c, &s,
                                   &H[jch + jch * ldh]);
                            H[(jch + 1) + jch * ldh] = ZERO;
                            cblas_srot(ilastm - jch, &H[jch + (jch + 1) * ldh], ldh,
                                       &H[(jch + 1) + (jch + 1) * ldh], ldh, c, s);
                            cblas_srot(ilastm - jch, &T[jch + (jch + 1) * ldt], ldt,
                                       &T[(jch + 1) + (jch + 1) * ldt], ldt, c, s);
                            if (ilq)
                                cblas_srot(n, &Q[jch * ldq], 1,
                                           &Q[(jch + 1) * ldq], 1, c, s);
                            if (ilazr2)
                                H[jch + (jch - 1) * ldh] = H[jch + (jch - 1) * ldh] * c;
                            ilazr2 = 0;
                            if (fabsf(T[(jch + 1) + (jch + 1) * ldt]) >= btol) {
                                if (jch + 1 >= ilast) {
                                    do_split = 1;
                                } else {
                                    ifirst = jch + 1;
                                    do_qz_step = 1;
                                }
                                break;
                            }
                            T[(jch + 1) + (jch + 1) * ldt] = ZERO;
                        }
                        if (!do_split && !do_qz_step)
                            do_zero_t = 1;
                        break;
                    } else {

                        /* Only test 2 passed -- chase the zero to T(ILAST,ILAST)
                         * Then process as in the case T(ILAST,ILAST)=0 */

                        for (jch = j; jch <= ilast - 1; jch++) {
                            temp = T[jch + (jch + 1) * ldt];
                            slartg(temp, T[(jch + 1) + (jch + 1) * ldt], &c, &s,
                                   &T[jch + (jch + 1) * ldt]);
                            T[(jch + 1) + (jch + 1) * ldt] = ZERO;
                            if (jch < ilastm - 1)
                                cblas_srot(ilastm - jch - 1,
                                           &T[jch + (jch + 2) * ldt], ldt,
                                           &T[(jch + 1) + (jch + 2) * ldt], ldt, c, s);
                            cblas_srot(ilastm - jch + 2,
                                       &H[jch + (jch - 1) * ldh], ldh,
                                       &H[(jch + 1) + (jch - 1) * ldh], ldh, c, s);
                            if (ilq)
                                cblas_srot(n, &Q[jch * ldq], 1,
                                           &Q[(jch + 1) * ldq], 1, c, s);
                            temp = H[(jch + 1) + jch * ldh];
                            slartg(temp, H[(jch + 1) + (jch - 1) * ldh], &c, &s,
                                   &H[(jch + 1) + jch * ldh]);
                            H[(jch + 1) + (jch - 1) * ldh] = ZERO;
                            cblas_srot(jch + 1 - ifrstm,
                                       &H[ifrstm + jch * ldh], 1,
                                       &H[ifrstm + (jch - 1) * ldh], 1, c, s);
                            cblas_srot(jch - ifrstm,
                                       &T[ifrstm + jch * ldt], 1,
                                       &T[ifrstm + (jch - 1) * ldt], 1, c, s);
                            if (ilz)
                                cblas_srot(n, &Z[jch * ldz], 1,
                                           &Z[(jch - 1) * ldz], 1, c, s);
                        }
                        do_zero_t = 1;
                        break;
                    }
                } else if (ilazro) {

                    /* Only test 1 passed -- work on J:ILAST */

                    ifirst = j;
                    do_qz_step = 1;
                    break;
                }

                /* Neither test passed -- try next J */
            }

            /* Drop-through is "impossible" */

            if (!do_split && !do_zero_t && !do_qz_step) {
                *info = n + 1;
                goto label_exit;
            }
        }

        /* T(ILAST,ILAST)=0 -- clear H(ILAST,ILAST-1) to split off a
         * 1x1 block. */

        if (do_zero_t) {
            temp = H[ilast + ilast * ldh];
            slartg(temp, H[ilast + (ilast - 1) * ldh], &c, &s,
                   &H[ilast + ilast * ldh]);
            H[ilast + (ilast - 1) * ldh] = ZERO;
            cblas_srot(ilast - ifrstm, &H[ifrstm + ilast * ldh], 1,
                       &H[ifrstm + (ilast - 1) * ldh], 1, c, s);
            cblas_srot(ilast - ifrstm, &T[ifrstm + ilast * ldt], 1,
                       &T[ifrstm + (ilast - 1) * ldt], 1, c, s);
            if (ilz)
                cblas_srot(n, &Z[ilast * ldz], 1,
                           &Z[(ilast - 1) * ldz], 1, c, s);
            do_split = 1;
        }

        /* H(ILAST,ILAST-1)=0 -- Standardize B, set ALPHAR, ALPHAI,
         *                        and BETA */

        if (do_split) {
            if (T[ilast + ilast * ldt] < ZERO) {
                if (ilschr) {
                    for (j = ifrstm; j <= ilast; j++) {
                        H[j + ilast * ldh] = -H[j + ilast * ldh];
                        T[j + ilast * ldt] = -T[j + ilast * ldt];
                    }
                } else {
                    H[ilast + ilast * ldh] = -H[ilast + ilast * ldh];
                    T[ilast + ilast * ldt] = -T[ilast + ilast * ldt];
                }
                if (ilz) {
                    for (j = 0; j < n; j++)
                        Z[j + ilast * ldz] = -Z[j + ilast * ldz];
                }
            }
            alphar[ilast] = H[ilast + ilast * ldh];
            alphai[ilast] = ZERO;
            beta[ilast] = T[ilast + ilast * ldt];

            /* Go to next block -- exit if finished. */

            ilast = ilast - 1;
            if (ilast < ilo)
                goto label_380;

            /* Reset counters */

            iiter = 0;
            eshift = ZERO;
            if (!ilschr) {
                ilastm = ilast;
                if (ifrstm > ilast)
                    ifrstm = ilo;
            }
            continue;
        }

        /*
         * QZ step
         *
         * This iteration only involves rows/columns IFIRST:ILAST. We
         * assume IFIRST < ILAST, and that the diagonal of B is non-zero.
         */

        iiter = iiter + 1;
        if (!ilschr) {
            ifrstm = ifirst;
        }

        /* Compute single shifts. */

        if ((iiter / 10) * 10 == iiter) {

            /* Exceptional shift.  Chosen for no particularly good reason.
             * (Single shift only.) */

            if (((f32)maxit * safmin) * fabsf(H[ilast + (ilast - 1) * ldh]) <
                fabsf(T[(ilast - 1) + (ilast - 1) * ldt])) {
                eshift = H[ilast + (ilast - 1) * ldh] /
                         T[(ilast - 1) + (ilast - 1) * ldt];
            } else {
                eshift = eshift + ONE / (safmin * (f32)maxit);
            }
            s1 = ONE;
            wr = eshift;

        } else {

            /* Shifts based on the generalized eigenvalues of the
             * bottom-right 2x2 block of A and B. The first eigenvalue
             * returned by SLAG2 is the Wilkinson shift (AEP p.512), */

            slag2(&H[(ilast - 1) + (ilast - 1) * ldh], ldh,
                  &T[(ilast - 1) + (ilast - 1) * ldt], ldt, safmin * SAFETY,
                  &s1, &s2, &wr, &wr2, &wi);

            if (fabsf((wr / s1) * T[ilast + ilast * ldt] - H[ilast + ilast * ldh]) >
                fabsf((wr2 / s2) * T[ilast + ilast * ldt] - H[ilast + ilast * ldh])) {
                temp = wr;
                wr = wr2;
                wr2 = temp;
                temp = s1;
                s1 = s2;
                s2 = temp;
            }
            temp = fmaxf(s1, safmin * fmaxf(ONE, fmaxf(fabsf(wr), fabsf(wi))));
            if (wi != ZERO)
                goto label_200;
        }

        /* Fiddle with shift to avoid overflow */

        temp = fminf(ascale, ONE) * (HALF * safmax);
        if (s1 > temp) {
            scale = temp / s1;
        } else {
            scale = ONE;
        }

        temp = fminf(bscale, ONE) * (HALF * safmax);
        if (fabsf(wr) > temp)
            scale = fminf(scale, temp / fabsf(wr));
        s1 = scale * s1;
        wr = scale * wr;

        /* Now check for two consecutive small subdiagonals. */

        for (j = ilast - 1; j >= ifirst + 1; j--) {
            istart = j;
            temp = fabsf(s1 * H[j + (j - 1) * ldh]);
            temp2 = fabsf(s1 * H[j + j * ldh] - wr * T[j + j * ldt]);
            tempr = fmaxf(temp, temp2);
            if (tempr < ONE && tempr != ZERO) {
                temp = temp / tempr;
                temp2 = temp2 / tempr;
            }
            if (fabsf((ascale * H[(j + 1) + j * ldh]) * temp) <=
                (ascale * atol) * temp2)
                goto label_130;
        }

        istart = ifirst;
label_130:

        /* Do an implicit single-shift QZ sweep.
         *
         * Initial Q */

        temp = s1 * H[istart + istart * ldh] - wr * T[istart + istart * ldt];
        temp2 = s1 * H[(istart + 1) + istart * ldh];
        slartg(temp, temp2, &c, &s, &tempr);

        /* Sweep */

        for (j = istart; j <= ilast - 1; j++) {
            if (j > istart) {
                temp = H[j + (j - 1) * ldh];
                slartg(temp, H[(j + 1) + (j - 1) * ldh], &c, &s,
                       &H[j + (j - 1) * ldh]);
                H[(j + 1) + (j - 1) * ldh] = ZERO;
            }

            for (jc = j; jc <= ilastm; jc++) {
                temp = c * H[j + jc * ldh] + s * H[(j + 1) + jc * ldh];
                H[(j + 1) + jc * ldh] = -s * H[j + jc * ldh] + c * H[(j + 1) + jc * ldh];
                H[j + jc * ldh] = temp;
                temp2 = c * T[j + jc * ldt] + s * T[(j + 1) + jc * ldt];
                T[(j + 1) + jc * ldt] = -s * T[j + jc * ldt] + c * T[(j + 1) + jc * ldt];
                T[j + jc * ldt] = temp2;
            }
            if (ilq) {
                for (jr = 0; jr < n; jr++) {
                    temp = c * Q[jr + j * ldq] + s * Q[jr + (j + 1) * ldq];
                    Q[jr + (j + 1) * ldq] = -s * Q[jr + j * ldq] + c * Q[jr + (j + 1) * ldq];
                    Q[jr + j * ldq] = temp;
                }
            }

            temp = T[(j + 1) + (j + 1) * ldt];
            slartg(temp, T[(j + 1) + j * ldt], &c, &s, &T[(j + 1) + (j + 1) * ldt]);
            T[(j + 1) + j * ldt] = ZERO;

            for (jr = ifrstm; jr <= (j + 2 < ilast ? j + 2 : ilast); jr++) {
                temp = c * H[jr + (j + 1) * ldh] + s * H[jr + j * ldh];
                H[jr + j * ldh] = -s * H[jr + (j + 1) * ldh] + c * H[jr + j * ldh];
                H[jr + (j + 1) * ldh] = temp;
            }
            for (jr = ifrstm; jr <= j; jr++) {
                temp = c * T[jr + (j + 1) * ldt] + s * T[jr + j * ldt];
                T[jr + j * ldt] = -s * T[jr + (j + 1) * ldt] + c * T[jr + j * ldt];
                T[jr + (j + 1) * ldt] = temp;
            }
            if (ilz) {
                for (jr = 0; jr < n; jr++) {
                    temp = c * Z[jr + (j + 1) * ldz] + s * Z[jr + j * ldz];
                    Z[jr + j * ldz] = -s * Z[jr + (j + 1) * ldz] + c * Z[jr + j * ldz];
                    Z[jr + (j + 1) * ldz] = temp;
                }
            }
        }

        goto label_350;

        /* Use Francis double-shift
         *
         * Note: the Francis double-shift should work with real shifts,
         *       but only if the block is at least 3x3.
         *       This code may break if this point is reached with
         *       a 2x2 block with real eigenvalues. */

label_200:
        if (ifirst + 1 == ilast) {

            /* Special case -- 2x2 block with complex eigenvectors
             *
             * Step 1: Standardize, that is, rotate so that
             *
             *                     ( B11  0  )
             *                 B = (         )  with B11 non-negative.
             *                     (  0  B22 )
             */

            slasv2(T[(ilast - 1) + (ilast - 1) * ldt], T[(ilast - 1) + ilast * ldt],
                   T[ilast + ilast * ldt], &b22, &b11, &sr, &cr, &sl, &cl);

            if (b11 < ZERO) {
                cr = -cr;
                sr = -sr;
                b11 = -b11;
                b22 = -b22;
            }

            cblas_srot(ilastm + 1 - ifirst,
                       &H[(ilast - 1) + (ilast - 1) * ldh], ldh,
                       &H[ilast + (ilast - 1) * ldh], ldh, cl, sl);
            cblas_srot(ilast + 1 - ifrstm,
                       &H[ifrstm + (ilast - 1) * ldh], 1,
                       &H[ifrstm + ilast * ldh], 1, cr, sr);

            if (ilast < ilastm)
                cblas_srot(ilastm - ilast,
                           &T[(ilast - 1) + (ilast + 1) * ldt], ldt,
                           &T[ilast + (ilast + 1) * ldt], ldt, cl, sl);
            if (ifrstm < ilast - 1)
                cblas_srot(ifirst - ifrstm,
                           &T[ifrstm + (ilast - 1) * ldt], 1,
                           &T[ifrstm + ilast * ldt], 1, cr, sr);

            if (ilq)
                cblas_srot(n, &Q[(ilast - 1) * ldq], 1,
                           &Q[ilast * ldq], 1, cl, sl);
            if (ilz)
                cblas_srot(n, &Z[(ilast - 1) * ldz], 1,
                           &Z[ilast * ldz], 1, cr, sr);

            T[(ilast - 1) + (ilast - 1) * ldt] = b11;
            T[(ilast - 1) + ilast * ldt] = ZERO;
            T[ilast + (ilast - 1) * ldt] = ZERO;
            T[ilast + ilast * ldt] = b22;

            /* If B22 is negative, negate column ILAST */

            if (b22 < ZERO) {
                for (j = ifrstm; j <= ilast; j++) {
                    H[j + ilast * ldh] = -H[j + ilast * ldh];
                    T[j + ilast * ldt] = -T[j + ilast * ldt];
                }

                if (ilz) {
                    for (j = 0; j < n; j++)
                        Z[j + ilast * ldz] = -Z[j + ilast * ldz];
                }
                b22 = -b22;
            }

            /* Step 2: Compute ALPHAR, ALPHAI, and BETA (see refs.)
             *
             * Recompute shift */

            slag2(&H[(ilast - 1) + (ilast - 1) * ldh], ldh,
                  &T[(ilast - 1) + (ilast - 1) * ldt], ldt, safmin * SAFETY,
                  &s1, &temp, &wr, &temp2, &wi);

            /* If standardization has perturbed the shift onto real line,
             * do another (real single-shift) QR step. */

            if (wi == ZERO)
                goto label_350;
            s1inv = ONE / s1;

            /* Do EISPACK (QZVAL) computation of alpha and beta */

            a11 = H[(ilast - 1) + (ilast - 1) * ldh];
            a21 = H[ilast + (ilast - 1) * ldh];
            a12 = H[(ilast - 1) + ilast * ldh];
            a22 = H[ilast + ilast * ldh];

            /* Compute complex Givens rotation on right
             * (Assume some element of C = (sA - wB) > unfl )
             *                          __
             * (sA - wB) ( CZ   -SZ )
             *           ( SZ    CZ )
             */

            c11r = s1 * a11 - wr * b11;
            c11i = -wi * b11;
            c12 = s1 * a12;
            c21 = s1 * a21;
            c22r = s1 * a22 - wr * b22;
            c22i = -wi * b22;

            if (fabsf(c11r) + fabsf(c11i) + fabsf(c12) > fabsf(c21) + fabsf(c22r) + fabsf(c22i)) {
                t1 = slapy3(c12, c11r, c11i);
                cz = c12 / t1;
                szr = -c11r / t1;
                szi = -c11i / t1;
            } else {
                cz = slapy2(c22r, c22i);
                if (cz <= safmin) {
                    cz = ZERO;
                    szr = ONE;
                    szi = ZERO;
                } else {
                    tempr = c22r / cz;
                    tempi = c22i / cz;
                    t1 = slapy2(cz, c21);
                    cz = cz / t1;
                    szr = -c21 * tempr / t1;
                    szi = c21 * tempi / t1;
                }
            }

            /* Compute Givens rotation on left
             *
             * (  CQ   SQ )
             * (  __      )  A or B
             * ( -SQ   CQ )
             */

            an = fabsf(a11) + fabsf(a12) + fabsf(a21) + fabsf(a22);
            bn = fabsf(b11) + fabsf(b22);
            wabs = fabsf(wr) + fabsf(wi);
            if (s1 * an > wabs * bn) {
                cq = cz * b11;
                sqr = szr * b22;
                sqi = -szi * b22;
            } else {
                a1r = cz * a11 + szr * a12;
                a1i = szi * a12;
                a2r = cz * a21 + szr * a22;
                a2i = szi * a22;
                cq = slapy2(a1r, a1i);
                if (cq <= safmin) {
                    cq = ZERO;
                    sqr = ONE;
                    sqi = ZERO;
                } else {
                    tempr = a1r / cq;
                    tempi = a1i / cq;
                    sqr = tempr * a2r + tempi * a2i;
                    sqi = tempi * a2r - tempr * a2i;
                }
            }
            t1 = slapy3(cq, sqr, sqi);
            cq = cq / t1;
            sqr = sqr / t1;
            sqi = sqi / t1;

            /* Compute diagonal elements of QBZ */

            tempr = sqr * szr - sqi * szi;
            tempi = sqr * szi + sqi * szr;
            b1r = cq * cz * b11 + tempr * b22;
            b1i = tempi * b22;
            b1a = slapy2(b1r, b1i);
            b2r = cq * cz * b22 + tempr * b11;
            b2i = -tempi * b11;
            b2a = slapy2(b2r, b2i);

            /* Normalize so beta > 0, and Im( alpha1 ) > 0 */

            beta[ilast - 1] = b1a;
            beta[ilast] = b2a;
            alphar[ilast - 1] = (wr * b1a) * s1inv;
            alphai[ilast - 1] = (wi * b1a) * s1inv;
            alphar[ilast] = (wr * b2a) * s1inv;
            alphai[ilast] = -(wi * b2a) * s1inv;

            /* Step 3: Go to next block -- exit if finished. */

            ilast = ifirst - 1;
            if (ilast < ilo)
                goto label_380;

            /* Reset counters */

            iiter = 0;
            eshift = ZERO;
            if (!ilschr) {
                ilastm = ilast;
                if (ifrstm > ilast)
                    ifrstm = ilo;
            }
            goto label_350;
        } else {

            /* Usual case: 3x3 or larger block, using Francis implicit
             *             double-shift
             *
             *                                  2
             * Eigenvalue equation is  w  - c w + d = 0,
             *
             *                                       -1 2        -1
             * so compute 1st column of  (A B  )  - c A B   + d
             * using the formula in QZIT (from EISPACK)
             *
             * We assume that the block is at least 3x3
             */

            ad11 = (ascale * H[(ilast - 1) + (ilast - 1) * ldh]) /
                   (bscale * T[(ilast - 1) + (ilast - 1) * ldt]);
            ad21 = (ascale * H[ilast + (ilast - 1) * ldh]) /
                   (bscale * T[(ilast - 1) + (ilast - 1) * ldt]);
            ad12 = (ascale * H[(ilast - 1) + ilast * ldh]) /
                   (bscale * T[ilast + ilast * ldt]);
            ad22 = (ascale * H[ilast + ilast * ldh]) /
                   (bscale * T[ilast + ilast * ldt]);
            u12 = T[(ilast - 1) + ilast * ldt] / T[ilast + ilast * ldt];
            ad11l = (ascale * H[ifirst + ifirst * ldh]) /
                    (bscale * T[ifirst + ifirst * ldt]);
            ad21l = (ascale * H[(ifirst + 1) + ifirst * ldh]) /
                    (bscale * T[ifirst + ifirst * ldt]);
            ad12l = (ascale * H[ifirst + (ifirst + 1) * ldh]) /
                    (bscale * T[(ifirst + 1) + (ifirst + 1) * ldt]);
            ad22l = (ascale * H[(ifirst + 1) + (ifirst + 1) * ldh]) /
                    (bscale * T[(ifirst + 1) + (ifirst + 1) * ldt]);
            ad32l = (ascale * H[(ifirst + 2) + (ifirst + 1) * ldh]) /
                    (bscale * T[(ifirst + 1) + (ifirst + 1) * ldt]);
            u12l = T[ifirst + (ifirst + 1) * ldt] / T[(ifirst + 1) + (ifirst + 1) * ldt];

            v[0] = (ad11 - ad11l) * (ad22 - ad11l) - ad12 * ad21 +
                   ad21 * u12 * ad11l + (ad12l - ad11l * u12l) * ad21l;
            v[1] = ((ad22l - ad11l) - ad21l * u12l - (ad11 - ad11l) -
                   (ad22 - ad11l) + ad21 * u12) * ad21l;
            v[2] = ad32l * ad21l;

            istart = ifirst;

            slarfg(3, &v[0], &v[1], 1, &tau);
            v[0] = ONE;

            /* Sweep */

            for (j = istart; j <= ilast - 2; j++) {

                /* All but last elements: use 3x3 Householder transforms.
                 *
                 * Zero (j-1)st column of A */

                if (j > istart) {
                    v[0] = H[j + (j - 1) * ldh];
                    v[1] = H[(j + 1) + (j - 1) * ldh];
                    v[2] = H[(j + 2) + (j - 1) * ldh];

                    slarfg(3, &H[j + (j - 1) * ldh], &v[1], 1, &tau);
                    v[0] = ONE;
                    H[(j + 1) + (j - 1) * ldh] = ZERO;
                    H[(j + 2) + (j - 1) * ldh] = ZERO;
                }

                t2 = tau * v[1];
                t3 = tau * v[2];
                for (jc = j; jc <= ilastm; jc++) {
                    temp = H[j + jc * ldh] + v[1] * H[(j + 1) + jc * ldh] +
                           v[2] * H[(j + 2) + jc * ldh];
                    H[j + jc * ldh] = H[j + jc * ldh] - temp * tau;
                    H[(j + 1) + jc * ldh] = H[(j + 1) + jc * ldh] - temp * t2;
                    H[(j + 2) + jc * ldh] = H[(j + 2) + jc * ldh] - temp * t3;
                    temp2 = T[j + jc * ldt] + v[1] * T[(j + 1) + jc * ldt] +
                            v[2] * T[(j + 2) + jc * ldt];
                    T[j + jc * ldt] = T[j + jc * ldt] - temp2 * tau;
                    T[(j + 1) + jc * ldt] = T[(j + 1) + jc * ldt] - temp2 * t2;
                    T[(j + 2) + jc * ldt] = T[(j + 2) + jc * ldt] - temp2 * t3;
                }
                if (ilq) {
                    for (jr = 0; jr < n; jr++) {
                        temp = Q[jr + j * ldq] + v[1] * Q[jr + (j + 1) * ldq] +
                               v[2] * Q[jr + (j + 2) * ldq];
                        Q[jr + j * ldq] = Q[jr + j * ldq] - temp * tau;
                        Q[jr + (j + 1) * ldq] = Q[jr + (j + 1) * ldq] - temp * t2;
                        Q[jr + (j + 2) * ldq] = Q[jr + (j + 2) * ldq] - temp * t3;
                    }
                }

                /* Zero j-th column of B (see DLAGBC for details)
                 *
                 * Swap rows to pivot */

                ilpivt = 0;
                temp = fmaxf(fabsf(T[(j + 1) + (j + 1) * ldt]),
                            fabsf(T[(j + 1) + (j + 2) * ldt]));
                temp2 = fmaxf(fabsf(T[(j + 2) + (j + 1) * ldt]),
                             fabsf(T[(j + 2) + (j + 2) * ldt]));
                if (fmaxf(temp, temp2) < safmin) {
                    scale = ZERO;
                    u1 = ONE;
                    u2 = ZERO;
                } else {
                    if (temp >= temp2) {
                        w11 = T[(j + 1) + (j + 1) * ldt];
                        w21 = T[(j + 2) + (j + 1) * ldt];
                        w12 = T[(j + 1) + (j + 2) * ldt];
                        w22 = T[(j + 2) + (j + 2) * ldt];
                        u1 = T[(j + 1) + j * ldt];
                        u2 = T[(j + 2) + j * ldt];
                    } else {
                        w21 = T[(j + 1) + (j + 1) * ldt];
                        w11 = T[(j + 2) + (j + 1) * ldt];
                        w22 = T[(j + 1) + (j + 2) * ldt];
                        w12 = T[(j + 2) + (j + 2) * ldt];
                        u2 = T[(j + 1) + j * ldt];
                        u1 = T[(j + 2) + j * ldt];
                    }

                    /* Swap columns if nec. */

                    if (fabsf(w12) > fabsf(w11)) {
                        ilpivt = 1;
                        temp = w12;
                        temp2 = w22;
                        w12 = w11;
                        w22 = w21;
                        w11 = temp;
                        w21 = temp2;
                    }

                    /* LU-factor */

                    temp = w21 / w11;
                    u2 = u2 - temp * u1;
                    w22 = w22 - temp * w12;
                    w21 = ZERO;

                    /* Compute SCALE */

                    scale = ONE;
                    if (fabsf(w22) < safmin) {
                        scale = ZERO;
                        u2 = ONE;
                        u1 = -w12 / w11;
                    } else {
                        if (fabsf(w22) < fabsf(u2))
                            scale = fabsf(w22 / u2);
                        if (fabsf(w11) < fabsf(u1))
                            scale = fminf(scale, fabsf(w11 / u1));

                        /* Solve */

                        u2 = (scale * u2) / w22;
                        u1 = (scale * u1 - w12 * u2) / w11;
                    }
                }

                if (ilpivt) {
                    temp = u2;
                    u2 = u1;
                    u1 = temp;
                }

                /* Compute Householder Vector */

                t1 = sqrtf(scale * scale + u1 * u1 + u2 * u2);
                tau = ONE + scale / t1;
                vs = -ONE / (scale + t1);
                v[0] = ONE;
                v[1] = vs * u1;
                v[2] = vs * u2;

                /* Apply transformations from the right. */

                t2 = tau * v[1];
                t3 = tau * v[2];
                for (jr = ifrstm; jr <= (j + 3 < ilast ? j + 3 : ilast); jr++) {
                    temp = H[jr + j * ldh] + v[1] * H[jr + (j + 1) * ldh] +
                           v[2] * H[jr + (j + 2) * ldh];
                    H[jr + j * ldh] = H[jr + j * ldh] - temp * tau;
                    H[jr + (j + 1) * ldh] = H[jr + (j + 1) * ldh] - temp * t2;
                    H[jr + (j + 2) * ldh] = H[jr + (j + 2) * ldh] - temp * t3;
                }
                for (jr = ifrstm; jr <= j + 2; jr++) {
                    temp = T[jr + j * ldt] + v[1] * T[jr + (j + 1) * ldt] +
                           v[2] * T[jr + (j + 2) * ldt];
                    T[jr + j * ldt] = T[jr + j * ldt] - temp * tau;
                    T[jr + (j + 1) * ldt] = T[jr + (j + 1) * ldt] - temp * t2;
                    T[jr + (j + 2) * ldt] = T[jr + (j + 2) * ldt] - temp * t3;
                }
                if (ilz) {
                    for (jr = 0; jr < n; jr++) {
                        temp = Z[jr + j * ldz] + v[1] * Z[jr + (j + 1) * ldz] +
                               v[2] * Z[jr + (j + 2) * ldz];
                        Z[jr + j * ldz] = Z[jr + j * ldz] - temp * tau;
                        Z[jr + (j + 1) * ldz] = Z[jr + (j + 1) * ldz] - temp * t2;
                        Z[jr + (j + 2) * ldz] = Z[jr + (j + 2) * ldz] - temp * t3;
                    }
                }
                T[(j + 1) + j * ldt] = ZERO;
                T[(j + 2) + j * ldt] = ZERO;
            }

            /* Last elements: Use Givens rotations
             *
             * Rotations from the left */

            j = ilast - 1;
            temp = H[j + (j - 1) * ldh];
            slartg(temp, H[(j + 1) + (j - 1) * ldh], &c, &s, &H[j + (j - 1) * ldh]);
            H[(j + 1) + (j - 1) * ldh] = ZERO;

            for (jc = j; jc <= ilastm; jc++) {
                temp = c * H[j + jc * ldh] + s * H[(j + 1) + jc * ldh];
                H[(j + 1) + jc * ldh] = -s * H[j + jc * ldh] + c * H[(j + 1) + jc * ldh];
                H[j + jc * ldh] = temp;
                temp2 = c * T[j + jc * ldt] + s * T[(j + 1) + jc * ldt];
                T[(j + 1) + jc * ldt] = -s * T[j + jc * ldt] + c * T[(j + 1) + jc * ldt];
                T[j + jc * ldt] = temp2;
            }
            if (ilq) {
                for (jr = 0; jr < n; jr++) {
                    temp = c * Q[jr + j * ldq] + s * Q[jr + (j + 1) * ldq];
                    Q[jr + (j + 1) * ldq] = -s * Q[jr + j * ldq] + c * Q[jr + (j + 1) * ldq];
                    Q[jr + j * ldq] = temp;
                }
            }

            /* Rotations from the right. */

            temp = T[(j + 1) + (j + 1) * ldt];
            slartg(temp, T[(j + 1) + j * ldt], &c, &s, &T[(j + 1) + (j + 1) * ldt]);
            T[(j + 1) + j * ldt] = ZERO;

            for (jr = ifrstm; jr <= ilast; jr++) {
                temp = c * H[jr + (j + 1) * ldh] + s * H[jr + j * ldh];
                H[jr + j * ldh] = -s * H[jr + (j + 1) * ldh] + c * H[jr + j * ldh];
                H[jr + (j + 1) * ldh] = temp;
            }
            for (jr = ifrstm; jr <= ilast - 1; jr++) {
                temp = c * T[jr + (j + 1) * ldt] + s * T[jr + j * ldt];
                T[jr + j * ldt] = -s * T[jr + (j + 1) * ldt] + c * T[jr + j * ldt];
                T[jr + (j + 1) * ldt] = temp;
            }
            if (ilz) {
                for (jr = 0; jr < n; jr++) {
                    temp = c * Z[jr + (j + 1) * ldz] + s * Z[jr + j * ldz];
                    Z[jr + j * ldz] = -s * Z[jr + (j + 1) * ldz] + c * Z[jr + j * ldz];
                    Z[jr + (j + 1) * ldz] = temp;
                }
            }

            /* End of Double-Shift code */

        }

label_350:
        (void)0;
    }

    /* Drop-through = non-convergence */

    *info = ilast + 1;
    goto label_exit;

    /* Successful completion of all QZ steps */

label_380:

    /* Set Eigenvalues 0:ILO-1 */

    for (j = 0; j < ilo; j++) {
        if (T[j + j * ldt] < ZERO) {
            if (ilschr) {
                for (jr = 0; jr <= j; jr++) {
                    H[jr + j * ldh] = -H[jr + j * ldh];
                    T[jr + j * ldt] = -T[jr + j * ldt];
                }
            } else {
                H[j + j * ldh] = -H[j + j * ldh];
                T[j + j * ldt] = -T[j + j * ldt];
            }
            if (ilz) {
                for (jr = 0; jr < n; jr++)
                    Z[jr + j * ldz] = -Z[jr + j * ldz];
            }
        }
        alphar[j] = H[j + j * ldh];
        alphai[j] = ZERO;
        beta[j] = T[j + j * ldt];
    }

    /* Normal Termination */

    *info = 0;

    /* Exit (other than argument error) -- return optimal workspace size */

label_exit:
    work[0] = (f32)n;
}
