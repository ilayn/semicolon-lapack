/**
 * @file ctgevc.c
 * @brief CTGEVC computes eigenvectors of a pair of complex upper triangular matrices.
 */

#include <complex.h>
#include <math.h>
#include <cblas.h>
#include "semicolon_lapack_complex_single.h"

/**
 * CTGEVC computes some or all of the right and/or left eigenvectors of
 * a pair of complex matrices (S,P), where S and P are upper triangular.
 * Matrix pairs of this type are produced by the generalized Schur
 * factorization of a complex matrix pair (A,B):
 *
 *    A = Q*S*Z**H,  B = Q*P*Z**H
 *
 * as computed by CGGHRD + CHGEQZ.
 *
 * The right eigenvector x and the left eigenvector y of (S,P)
 * corresponding to an eigenvalue w are defined by:
 *
 *    S*x = w*P*x,  (y**H)*S = w*(y**H)*P,
 *
 * where y**H denotes the conjugate transpose of y.
 * The eigenvalues are not input to this routine, but are computed
 * directly from the diagonal elements of S and P.
 *
 * This routine returns the matrices X and/or Y of right and left
 * eigenvectors of (S,P), or the products Z*X and/or Q*Y,
 * where Z and Q are input matrices.
 * If Q and Z are the unitary factors from the generalized Schur
 * factorization of a matrix pair (A,B), then Z*X and Q*Y
 * are the matrices of right and left eigenvectors of (A,B).
 *
 * @param[in]     side    = 'R': compute right eigenvectors only;
 *                         = 'L': compute left eigenvectors only;
 *                         = 'B': compute both right and left eigenvectors.
 * @param[in]     howmny  = 'A': compute all right and/or left eigenvectors;
 *                         = 'B': compute all right and/or left eigenvectors,
 *                                backtransformed by the matrices in VR and/or VL;
 *                         = 'S': compute selected right and/or left eigenvectors,
 *                                specified by the logical array select.
 * @param[in]     select  Integer array, dimension (n).
 *                        If howmny='S', select specifies the eigenvectors to be
 *                        computed. Nonzero means compute the eigenvector.
 *                        Not referenced if howmny = 'A' or 'B'.
 * @param[in]     n       The order of the matrices S and P. n >= 0.
 * @param[in]     S       Complex array, dimension (lds, n). The upper triangular
 *                        matrix S from a generalized Schur factorization.
 * @param[in]     lds     The leading dimension of S. lds >= max(1,n).
 * @param[in]     P       Complex array, dimension (ldp, n). The upper triangular
 *                        matrix P. P must have real diagonal elements.
 * @param[in]     ldp     The leading dimension of P. ldp >= max(1,n).
 * @param[in,out] VL      Complex array, dimension (ldvl, mm). Left eigenvectors.
 * @param[in]     ldvl    The leading dimension of VL. ldvl >= 1, and if
 *                        side = 'L' or 'B', ldvl >= n.
 * @param[in,out] VR      Complex array, dimension (ldvr, mm). Right eigenvectors.
 * @param[in]     ldvr    The leading dimension of VR. ldvr >= 1, and if
 *                        side = 'R' or 'B', ldvr >= n.
 * @param[in]     mm      The number of columns in VL and/or VR. mm >= m.
 * @param[out]    m       The number of columns in VL and/or VR actually used.
 * @param[out]    work    Complex workspace array, dimension (2*n).
 * @param[out]    rwork   Single precision workspace array, dimension (2*n).
 * @param[out]    info
 *                         - = 0: successful exit.
 *                         - < 0: if info = -i, the i-th argument had an illegal value.
 */
void ctgevc(
    const char* side,
    const char* howmny,
    const int* restrict select,
    const int n,
    const c64* restrict S,
    const int lds,
    const c64* restrict P,
    const int ldp,
    c64* restrict VL,
    const int ldvl,
    c64* restrict VR,
    const int ldvr,
    const int mm,
    int* m,
    c64* restrict work,
    f32* restrict rwork,
    int* info)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;
    const c64 CZERO = CMPLXF(0.0f, 0.0f);
    const c64 CONE = CMPLXF(1.0f, 0.0f);

    int compl_, compr, ilall, ilback, ilbbad, ilcomp;
    int ihwmny, iside;
    int i, ibeg, ieig, iend, im, isrc, j, je, jr;
    f32 acoefa, acoeff, anorm, ascale, bcoefa, big, bignum;
    f32 bnorm, bscale, dmin_, safmin, sbeta, scale, small_, temp, ulp, xmax;
    c64 bcoeff, ca, cb, d, salpha, sum_, suma, sumb;
    int lsa, lsb;

    *info = 0;

    if (howmny[0] == 'A' || howmny[0] == 'a') {
        ihwmny = 1;
        ilall = 1;
        ilback = 0;
    } else if (howmny[0] == 'S' || howmny[0] == 's') {
        ihwmny = 2;
        ilall = 0;
        ilback = 0;
    } else if (howmny[0] == 'B' || howmny[0] == 'b') {
        ihwmny = 3;
        ilall = 1;
        ilback = 1;
    } else {
        ihwmny = -1;
        ilall = 0;
        ilback = 0;
    }

    if (side[0] == 'R' || side[0] == 'r') {
        iside = 1;
        compl_ = 0;
        compr = 1;
    } else if (side[0] == 'L' || side[0] == 'l') {
        iside = 2;
        compl_ = 1;
        compr = 0;
    } else if (side[0] == 'B' || side[0] == 'b') {
        iside = 3;
        compl_ = 1;
        compr = 1;
    } else {
        iside = -1;
        compl_ = 0;
        compr = 0;
    }

    if (iside < 0) {
        *info = -1;
    } else if (ihwmny < 0) {
        *info = -2;
    } else if (n < 0) {
        *info = -4;
    } else if (lds < (1 > n ? 1 : n)) {
        *info = -6;
    } else if (ldp < (1 > n ? 1 : n)) {
        *info = -8;
    }
    if (*info != 0) {
        xerbla("CTGEVC", -(*info));
        return;
    }

    /* Count the number of eigenvectors */
    if (!ilall) {
        im = 0;
        for (j = 0; j < n; j++) {
            if (select[j])
                im = im + 1;
        }
    } else {
        im = n;
    }

    /* Check diagonal of B */
    ilbbad = 0;
    for (j = 0; j < n; j++) {
        if (cimagf(P[j + j * ldp]) != ZERO)
            ilbbad = 1;
    }

    if (ilbbad) {
        *info = -7;
    } else if ((compl_ && ldvl < n) || ldvl < 1) {
        *info = -10;
    } else if ((compr && ldvr < n) || ldvr < 1) {
        *info = -12;
    } else if (mm < im) {
        *info = -13;
    }
    if (*info != 0) {
        xerbla("CTGEVC", -(*info));
        return;
    }

    /* Quick return if possible */
    *m = im;
    if (n == 0)
        return;

    /* Machine Constants */
    safmin = slamch("Safe minimum");
    ulp = slamch("Epsilon") * slamch("Base");
    small_ = safmin * n / ulp;
    big = ONE / small_;
    bignum = ONE / (safmin * n);

    /* Compute the 1-norm of each column of the strictly upper triangular
       part of A and B to check for possible overflow in the triangular
       solver. */
    anorm = cabs1f(S[0 + 0 * lds]);
    bnorm = cabs1f(P[0 + 0 * ldp]);
    rwork[0] = ZERO;
    rwork[n] = ZERO;
    for (j = 1; j < n; j++) {
        rwork[j] = ZERO;
        rwork[n + j] = ZERO;
        for (i = 0; i < j; i++) {
            rwork[j] = rwork[j] + cabs1f(S[i + j * lds]);
            rwork[n + j] = rwork[n + j] + cabs1f(P[i + j * ldp]);
        }
        anorm = fmaxf(anorm, rwork[j] + cabs1f(S[j + j * lds]));
        bnorm = fmaxf(bnorm, rwork[n + j] + cabs1f(P[j + j * ldp]));
    }

    ascale = ONE / fmaxf(anorm, safmin);
    bscale = ONE / fmaxf(bnorm, safmin);

    /* Left eigenvectors */
    if (compl_) {
        ieig = 0;

        /* Main loop over eigenvalues */
        for (je = 0; je < n; je++) {
            if (ilall) {
                ilcomp = 1;
            } else {
                ilcomp = select[je];
            }
            if (ilcomp) {
                ieig = ieig + 1;

                if (cabs1f(S[je + je * lds]) <= safmin &&
                    fabsf(crealf(P[je + je * ldp])) <= safmin) {

                    /* Singular matrix pencil -- return unit eigenvector */
                    for (jr = 0; jr < n; jr++)
                        VL[jr + (ieig - 1) * ldvl] = CZERO;
                    VL[(ieig - 1) + (ieig - 1) * ldvl] = CONE;
                    continue;
                }

                /* Non-singular eigenvalue:
                   Compute coefficients  a  and  b  in
                        H
                      y  ( a A - b B ) = 0 */

                temp = ONE / fmaxf(fmaxf(cabs1f(S[je + je * lds]) * ascale,
                             fabsf(crealf(P[je + je * ldp])) * bscale), safmin);
                salpha = (temp * S[je + je * lds]) * ascale;
                sbeta = (temp * crealf(P[je + je * ldp])) * bscale;
                acoeff = sbeta * ascale;
                bcoeff = salpha * bscale;

                /* Scale to avoid underflow */
                lsa = fabsf(sbeta) >= safmin && fabsf(acoeff) < small_;
                lsb = cabs1f(salpha) >= safmin && cabs1f(bcoeff) < small_;

                scale = ONE;
                if (lsa)
                    scale = (small_ / fabsf(sbeta)) * fminf(anorm, big);
                if (lsb)
                    scale = fmaxf(scale, (small_ / cabs1f(salpha)) * fminf(bnorm, big));
                if (lsa || lsb) {
                    scale = fminf(scale, ONE /
                            (safmin * fmaxf(fmaxf(ONE, fabsf(acoeff)),
                            cabs1f(bcoeff))));
                    if (lsa) {
                        acoeff = ascale * (scale * sbeta);
                    } else {
                        acoeff = scale * acoeff;
                    }
                    if (lsb) {
                        bcoeff = bscale * (scale * salpha);
                    } else {
                        bcoeff = scale * bcoeff;
                    }
                }

                acoefa = fabsf(acoeff);
                bcoefa = cabs1f(bcoeff);
                xmax = ONE;
                for (jr = 0; jr < n; jr++)
                    work[jr] = CZERO;
                work[je] = CONE;
                dmin_ = fmaxf(fmaxf(ulp * acoefa * anorm, ulp * bcoefa * bnorm), safmin);

                /*                                    H
                   Triangular solve of  (a A - b B)  y = 0

                                            H
                   (rowwise in  (a A - b B) , or columnwise in a A - b B) */

                for (j = je + 1; j < n; j++) {

                    /* Compute
                          j-1
                    SUM = sum  conjg( a*S(k,j) - b*P(k,j) )*x(k)
                          k=je
                    (Scale if necessary) */

                    temp = ONE / xmax;
                    if (acoefa * rwork[j] + bcoefa * rwork[n + j] > bignum * temp) {
                        for (jr = je; jr < j; jr++)
                            work[jr] = temp * work[jr];
                        xmax = ONE;
                    }
                    suma = CZERO;
                    sumb = CZERO;

                    for (jr = je; jr < j; jr++) {
                        suma = suma + conjf(S[jr + j * lds]) * work[jr];
                        sumb = sumb + conjf(P[jr + j * ldp]) * work[jr];
                    }
                    sum_ = acoeff * suma - conjf(bcoeff) * sumb;

                    /* Form x(j) = - SUM / conjg( a*S(j,j) - b*P(j,j) )

                       with scaling and perturbation of the denominator */

                    d = conjf(acoeff * S[j + j * lds] - bcoeff * P[j + j * ldp]);
                    if (cabs1f(d) <= dmin_)
                        d = CMPLXF(dmin_, 0.0f);

                    if (cabs1f(d) < ONE) {
                        if (cabs1f(sum_) >= bignum * cabs1f(d)) {
                            temp = ONE / cabs1f(sum_);
                            for (jr = je; jr < j; jr++)
                                work[jr] = temp * work[jr];
                            xmax = temp * xmax;
                            sum_ = temp * sum_;
                        }
                    }
                    work[j] = cladiv(-sum_, d);
                    xmax = fmaxf(xmax, cabs1f(work[j]));
                }

                /* Back transform eigenvector if HOWMNY='B'. */
                if (ilback) {
                    cblas_cgemv(CblasColMajor, CblasNoTrans, n, n - je, &CONE,
                                &VL[0 + je * ldvl], ldvl,
                                &work[je], 1, &CZERO, &work[n], 1);
                    isrc = 2;
                    ibeg = 0;
                } else {
                    isrc = 1;
                    ibeg = je;
                }

                /* Copy and scale eigenvector into column of VL */
                xmax = ZERO;
                for (jr = ibeg; jr < n; jr++)
                    xmax = fmaxf(xmax, cabs1f(work[(isrc - 1) * n + jr]));

                if (xmax > safmin) {
                    temp = ONE / xmax;
                    for (jr = ibeg; jr < n; jr++)
                        VL[jr + (ieig - 1) * ldvl] = temp * work[(isrc - 1) * n + jr];
                } else {
                    ibeg = n;
                }

                for (jr = 0; jr < ibeg; jr++)
                    VL[jr + (ieig - 1) * ldvl] = CZERO;

            }
        }
    }

    /* Right eigenvectors */
    if (compr) {
        ieig = im;

        /* Main loop over eigenvalues */
        for (je = n - 1; je >= 0; je--) {
            if (ilall) {
                ilcomp = 1;
            } else {
                ilcomp = select[je];
            }
            if (ilcomp) {
                ieig = ieig - 1;

                if (cabs1f(S[je + je * lds]) <= safmin &&
                    fabsf(crealf(P[je + je * ldp])) <= safmin) {

                    /* Singular matrix pencil -- return unit eigenvector */
                    for (jr = 0; jr < n; jr++)
                        VR[jr + ieig * ldvr] = CZERO;
                    VR[ieig + ieig * ldvr] = CONE;
                    continue;
                }

                /* Non-singular eigenvalue:
                   Compute coefficients  a  and  b  in

                   ( a A - b B ) x  = 0 */

                temp = ONE / fmaxf(fmaxf(cabs1f(S[je + je * lds]) * ascale,
                             fabsf(crealf(P[je + je * ldp])) * bscale), safmin);
                salpha = (temp * S[je + je * lds]) * ascale;
                sbeta = (temp * crealf(P[je + je * ldp])) * bscale;
                acoeff = sbeta * ascale;
                bcoeff = salpha * bscale;

                /* Scale to avoid underflow */
                lsa = fabsf(sbeta) >= safmin && fabsf(acoeff) < small_;
                lsb = cabs1f(salpha) >= safmin && cabs1f(bcoeff) < small_;

                scale = ONE;
                if (lsa)
                    scale = (small_ / fabsf(sbeta)) * fminf(anorm, big);
                if (lsb)
                    scale = fmaxf(scale, (small_ / cabs1f(salpha)) * fminf(bnorm, big));
                if (lsa || lsb) {
                    scale = fminf(scale, ONE /
                            (safmin * fmaxf(fmaxf(ONE, fabsf(acoeff)),
                            cabs1f(bcoeff))));
                    if (lsa) {
                        acoeff = ascale * (scale * sbeta);
                    } else {
                        acoeff = scale * acoeff;
                    }
                    if (lsb) {
                        bcoeff = bscale * (scale * salpha);
                    } else {
                        bcoeff = scale * bcoeff;
                    }
                }

                acoefa = fabsf(acoeff);
                bcoefa = cabs1f(bcoeff);
                for (jr = 0; jr < n; jr++)
                    work[jr] = CZERO;
                work[je] = CONE;
                dmin_ = fmaxf(fmaxf(ulp * acoefa * anorm, ulp * bcoefa * bnorm), safmin);

                /* Triangular solve of  (a A - b B) x = 0  (columnwise)

                   WORK(1:j-1) contains sums w,
                   WORK(j+1:JE) contains x */

                for (jr = 0; jr < je; jr++)
                    work[jr] = acoeff * S[jr + je * lds] - bcoeff * P[jr + je * ldp];
                work[je] = CONE;

                for (j = je - 1; j >= 0; j--) {

                    /* Form x(j) := - w(j) / d
                       with scaling and perturbation of the denominator */

                    d = acoeff * S[j + j * lds] - bcoeff * P[j + j * ldp];
                    if (cabs1f(d) <= dmin_)
                        d = CMPLXF(dmin_, 0.0f);

                    if (cabs1f(d) < ONE) {
                        if (cabs1f(work[j]) >= bignum * cabs1f(d)) {
                            temp = ONE / cabs1f(work[j]);
                            for (jr = 0; jr <= je; jr++)
                                work[jr] = temp * work[jr];
                        }
                    }

                    work[j] = cladiv(-work[j], d);

                    if (j > 0) {

                        /* w = w + x(j)*(a S(*,j) - b P(*,j) ) with scaling */

                        if (cabs1f(work[j]) > ONE) {
                            temp = ONE / cabs1f(work[j]);
                            if (acoefa * rwork[j] + bcoefa * rwork[n + j] >=
                                bignum * temp) {
                                for (jr = 0; jr <= je; jr++)
                                    work[jr] = temp * work[jr];
                            }
                        }

                        ca = acoeff * work[j];
                        cb = bcoeff * work[j];
                        for (jr = 0; jr < j; jr++)
                            work[jr] = work[jr] + ca * S[jr + j * lds] -
                                       cb * P[jr + j * ldp];
                    }
                }

                /* Back transform eigenvector if HOWMNY='B'. */
                if (ilback) {
                    cblas_cgemv(CblasColMajor, CblasNoTrans, n, je + 1, &CONE,
                                VR, ldvr, work, 1, &CZERO, &work[n], 1);
                    isrc = 2;
                    iend = n;
                } else {
                    isrc = 1;
                    iend = je + 1;
                }

                /* Copy and scale eigenvector into column of VR */
                xmax = ZERO;
                for (jr = 0; jr < iend; jr++)
                    xmax = fmaxf(xmax, cabs1f(work[(isrc - 1) * n + jr]));

                if (xmax > safmin) {
                    temp = ONE / xmax;
                    for (jr = 0; jr < iend; jr++)
                        VR[jr + ieig * ldvr] = temp * work[(isrc - 1) * n + jr];
                } else {
                    iend = 0;
                }

                for (jr = iend; jr < n; jr++)
                    VR[jr + ieig * ldvr] = CZERO;

            }
        }
    }
}
