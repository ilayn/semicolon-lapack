/**
 * @file stgevc.c
 * @brief STGEVC computes eigenvectors of a pair of real matrices (S,P) in generalized Schur form.
 */

#include <math.h>
#include <cblas.h>
#include "semicolon_lapack_single.h"

/**
 * STGEVC computes some or all of the right and/or left eigenvectors of
 * a pair of real matrices (S,P), where S is a quasi-triangular matrix
 * and P is upper triangular. Matrix pairs of this type are produced by
 * the generalized Schur factorization of a matrix pair (A,B):
 *
 *    A = Q*S*Z**T,  B = Q*P*Z**T
 *
 * as computed by SGGHRD + SHGEQZ.
 *
 * The right eigenvector x and the left eigenvector y of (S,P)
 * corresponding to an eigenvalue w are defined by:
 *
 *    S*x = w*P*x,  (y**H)*S = w*(y**H)*P,
 *
 * where y**H denotes the conjugate transpose of y.
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
 * @param[in]     n       The order of the matrices S and P. n >= 0.
 * @param[in]     S       Array of dimension (lds, n). The upper quasi-triangular
 *                        matrix S from a generalized Schur factorization.
 * @param[in]     lds     The leading dimension of S. lds >= max(1,n).
 * @param[in]     P       Array of dimension (ldp, n). The upper triangular matrix P.
 * @param[in]     ldp     The leading dimension of P. ldp >= max(1,n).
 * @param[in,out] VL      Array of dimension (ldvl, mm). Left eigenvectors.
 * @param[in]     ldvl    The leading dimension of VL. ldvl >= 1, and if
 *                        side = 'L' or 'B', ldvl >= n.
 * @param[in,out] VR      Array of dimension (ldvr, mm). Right eigenvectors.
 * @param[in]     ldvr    The leading dimension of VR. ldvr >= 1, and if
 *                        side = 'R' or 'B', ldvr >= n.
 * @param[in]     mm      The number of columns in VL and/or VR. mm >= m.
 * @param[out]    m       The number of columns in VL and/or VR actually used.
 * @param[out]    work    Workspace array of dimension (6*n).
 * @param[out]    info
 *                         - = 0: successful exit.
 *                         - < 0: if info = -i, the i-th argument had an illegal value.
 *                         - > 0: the 2-by-2 block (info:info+1) does not have a complex
 *                           eigenvalue.
 */
void stgevc(
    const char* side,
    const char* howmny,
    const int* const restrict select,
    const int n,
    const float* const restrict S,
    const int lds,
    const float* const restrict P,
    const int ldp,
    float* const restrict VL,
    const int ldvl,
    float* const restrict VR,
    const int ldvr,
    const int mm,
    int* m,
    float* const restrict work,
    int* info)
{
    const float ZERO = 0.0f;
    const float ONE = 1.0f;
    const float SAFETY = 100.0f;

    int compl, compr, ilall, ilback, ilcplx;
    int ihwmny, iside;
    int ilabad, ilbbad;
    int i, ibeg, iend, ieig, iinfo, ilcomp, im, j, ja, je, jr, jw, na, nw;
    int il2by2;
    float acoef, acoefa, anorm, ascale, bcoefa, bcoefi, bcoefr;
    float big, bignum, bnorm, bscale;
    float cim2a, cim2b, cimaga, cimagb, cre2a, cre2b, creala, crealb;
    float dmin, safmin, salfar, sbeta, scale, small;
    float temp, temp2, temp2i, temp2r, ulp, xmax, xscale;
    float bdiag[2], sum[2][2], sums[2][2], sump[2][2];

    *info = 0;

    /* Decode and test input parameters */
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
        ilall = 1;
        ilback = 0;
    }

    if (side[0] == 'R' || side[0] == 'r') {
        iside = 1;
        compl = 0;
        compr = 1;
    } else if (side[0] == 'L' || side[0] == 'l') {
        iside = 2;
        compl = 1;
        compr = 0;
    } else if (side[0] == 'B' || side[0] == 'b') {
        iside = 3;
        compl = 1;
        compr = 1;
    } else {
        iside = -1;
        compl = 0;
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
        xerbla("STGEVC", -(*info));
        return;
    }

    /* Count the number of eigenvectors to be computed */
    if (!ilall) {
        im = 0;
        ilcplx = 0;
        for (j = 0; j < n; j++) {
            if (ilcplx) {
                ilcplx = 0;
                continue;
            }
            if (j < n - 1) {
                if (S[(j + 1) + j * lds] != ZERO)
                    ilcplx = 1;
            }
            if (ilcplx) {
                if (select[j] || select[j + 1])
                    im += 2;
            } else {
                if (select[j])
                    im += 1;
            }
        }
    } else {
        im = n;
    }

    /* Check 2-by-2 diagonal blocks of A, B */
    ilabad = 0;
    ilbbad = 0;
    for (j = 0; j < n - 1; j++) {
        if (S[(j + 1) + j * lds] != ZERO) {
            if (P[j + j * ldp] == ZERO || P[(j + 1) + (j + 1) * ldp] == ZERO ||
                P[j + (j + 1) * ldp] != ZERO)
                ilbbad = 1;
            if (j < n - 2) {
                if (S[(j + 2) + (j + 1) * lds] != ZERO)
                    ilabad = 1;
            }
        }
    }

    if (ilabad) {
        *info = -5;
    } else if (ilbbad) {
        *info = -7;
    } else if ((compl && ldvl < n) || ldvl < 1) {
        *info = -10;
    } else if ((compr && ldvr < n) || ldvr < 1) {
        *info = -12;
    } else if (mm < im) {
        *info = -13;
    }
    if (*info != 0) {
        xerbla("STGEVC", -(*info));
        return;
    }

    /* Quick return if possible */
    *m = im;
    if (n == 0)
        return;

    /* Machine constants */
    safmin = slamch("Safe minimum");
    big = ONE / safmin;
    ulp = slamch("Epsilon") * slamch("Base");
    small = safmin * n / ulp;
    big = ONE / small;
    bignum = ONE / (safmin * n);

    /* Compute the 1-norm of each column of the strictly upper triangular
       part (i.e., excluding all elements belonging to the diagonal
       blocks) of A and B to check for possible overflow in the
       triangular solver. */
    anorm = fabsf(S[0 + 0 * lds]);
    if (n > 1)
        anorm = anorm + fabsf(S[1 + 0 * lds]);
    bnorm = fabsf(P[0 + 0 * ldp]);
    work[0] = ZERO;
    work[n] = ZERO;

    for (j = 1; j < n; j++) {
        temp = ZERO;
        temp2 = ZERO;
        if (S[j + (j - 1) * lds] == ZERO) {
            iend = j - 1;
        } else {
            iend = j - 2;
        }
        for (i = 0; i <= iend; i++) {
            temp = temp + fabsf(S[i + j * lds]);
            temp2 = temp2 + fabsf(P[i + j * ldp]);
        }
        work[j] = temp;
        work[n + j] = temp2;
        for (i = iend + 1; i <= (j + 1 < n ? j + 1 : n - 1); i++) {
            temp = temp + fabsf(S[i + j * lds]);
            temp2 = temp2 + fabsf(P[i + j * ldp]);
        }
        anorm = (anorm > temp ? anorm : temp);
        bnorm = (bnorm > temp2 ? bnorm : temp2);
    }

    ascale = ONE / (anorm > safmin ? anorm : safmin);
    bscale = ONE / (bnorm > safmin ? bnorm : safmin);

    /* Left eigenvectors */
    if (compl) {
        ieig = 0;

        ilcplx = 0;
        for (je = 0; je < n; je++) {
            if (ilcplx) {
                ilcplx = 0;
                continue;
            }
            nw = 1;
            if (je < n - 1) {
                if (S[(je + 1) + je * lds] != ZERO) {
                    ilcplx = 1;
                    nw = 2;
                }
            }
            if (ilall) {
                ilcomp = 1;
            } else if (ilcplx) {
                ilcomp = select[je] || select[je + 1];
            } else {
                ilcomp = select[je];
            }
            if (!ilcomp)
                continue;

            if (!ilcplx) {
                if (fabsf(S[je + je * lds]) <= safmin &&
                    fabsf(P[je + je * ldp]) <= safmin) {
                    for (jr = 0; jr < n; jr++)
                        VL[jr + ieig * ldvl] = ZERO;
                    VL[ieig + ieig * ldvl] = ONE;
                    ieig++;
                    continue;
                }
            }

            for (jr = 0; jr < nw * n; jr++)
                work[2 * n + jr] = ZERO;

            if (!ilcplx) {
                temp = ONE / fmaxf(fmaxf(fabsf(S[je + je * lds]) * ascale,
                                       fabsf(P[je + je * ldp]) * bscale), safmin);
                salfar = (temp * S[je + je * lds]) * ascale;
                sbeta = (temp * P[je + je * ldp]) * bscale;
                acoef = sbeta * ascale;
                bcoefr = salfar * bscale;
                bcoefi = ZERO;

                scale = ONE;
                int lsa = fabsf(sbeta) >= safmin && fabsf(acoef) < small;
                int lsb = fabsf(salfar) >= safmin && fabsf(bcoefr) < small;
                if (lsa)
                    scale = (small / fabsf(sbeta)) * fminf(anorm, big);
                if (lsb)
                    scale = fmaxf(scale, (small / fabsf(salfar)) * fminf(bnorm, big));
                if (lsa || lsb) {
                    scale = fminf(scale, ONE / (safmin * fmaxf(fmaxf(ONE, fabsf(acoef)), fabsf(bcoefr))));
                    if (lsa) {
                        acoef = ascale * (scale * sbeta);
                    } else {
                        acoef = scale * acoef;
                    }
                    if (lsb) {
                        bcoefr = bscale * (scale * salfar);
                    } else {
                        bcoefr = scale * bcoefr;
                    }
                }
                acoefa = fabsf(acoef);
                bcoefa = fabsf(bcoefr);

                work[2 * n + je] = ONE;
                xmax = ONE;
            } else {
                slag2(&S[je + je * lds], lds, &P[je + je * ldp], ldp,
                      safmin * SAFETY, &acoef, &temp, &bcoefr, &temp2, &bcoefi);
                bcoefi = -bcoefi;
                if (bcoefi == ZERO) {
                    *info = je + 1;
                    return;
                }

                acoefa = fabsf(acoef);
                bcoefa = fabsf(bcoefr) + fabsf(bcoefi);
                scale = ONE;
                if (acoefa * ulp < safmin && acoefa >= safmin)
                    scale = (safmin / ulp) / acoefa;
                if (bcoefa * ulp < safmin && bcoefa >= safmin)
                    scale = fmaxf(scale, (safmin / ulp) / bcoefa);
                if (safmin * acoefa > ascale)
                    scale = ascale / (safmin * acoefa);
                if (safmin * bcoefa > bscale)
                    scale = fminf(scale, bscale / (safmin * bcoefa));
                if (scale != ONE) {
                    acoef = scale * acoef;
                    acoefa = fabsf(acoef);
                    bcoefr = scale * bcoefr;
                    bcoefi = scale * bcoefi;
                    bcoefa = fabsf(bcoefr) + fabsf(bcoefi);
                }

                temp = acoef * S[(je + 1) + je * lds];
                temp2r = acoef * S[je + je * lds] - bcoefr * P[je + je * ldp];
                temp2i = -bcoefi * P[je + je * ldp];
                if (fabsf(temp) > fabsf(temp2r) + fabsf(temp2i)) {
                    work[2 * n + je] = ONE;
                    work[3 * n + je] = ZERO;
                    work[2 * n + je + 1] = -temp2r / temp;
                    work[3 * n + je + 1] = -temp2i / temp;
                } else {
                    work[2 * n + je + 1] = ONE;
                    work[3 * n + je + 1] = ZERO;
                    temp = acoef * S[je + (je + 1) * lds];
                    work[2 * n + je] = (bcoefr * P[(je + 1) + (je + 1) * ldp] -
                                        acoef * S[(je + 1) + (je + 1) * lds]) / temp;
                    work[3 * n + je] = bcoefi * P[(je + 1) + (je + 1) * ldp] / temp;
                }
                xmax = fmaxf(fabsf(work[2 * n + je]) + fabsf(work[3 * n + je]),
                            fabsf(work[2 * n + je + 1]) + fabsf(work[3 * n + je + 1]));
            }

            dmin = fmaxf(fmaxf(ulp * acoefa * anorm, ulp * bcoefa * bnorm), safmin);

            il2by2 = 0;

            for (j = je + nw; j < n; j++) {
                if (il2by2) {
                    il2by2 = 0;
                    continue;
                }

                na = 1;
                bdiag[0] = P[j + j * ldp];
                if (j < n - 1) {
                    if (S[(j + 1) + j * lds] != ZERO) {
                        il2by2 = 1;
                        bdiag[1] = P[(j + 1) + (j + 1) * ldp];
                        na = 2;
                    }
                }

                xscale = ONE / fmaxf(ONE, xmax);
                temp = fmaxf(fmaxf(work[j], work[n + j]),
                            acoefa * work[j] + bcoefa * work[n + j]);
                if (il2by2)
                    temp = fmaxf(fmaxf(temp, work[j + 1]),
                                fmaxf(work[n + j + 1],
                                     acoefa * work[j + 1] + bcoefa * work[n + j + 1]));
                if (temp > bignum * xscale) {
                    for (jw = 0; jw < nw; jw++) {
                        for (jr = je; jr < j; jr++)
                            work[(jw + 2) * n + jr] = xscale * work[(jw + 2) * n + jr];
                    }
                    xmax = xmax * xscale;
                }

                for (jw = 0; jw < nw; jw++) {
                    for (ja = 0; ja < na; ja++) {
                        sums[ja][jw] = ZERO;
                        sump[ja][jw] = ZERO;

                        for (jr = je; jr < j; jr++) {
                            sums[ja][jw] = sums[ja][jw] +
                                           S[jr + (j + ja) * lds] * work[(jw + 2) * n + jr];
                            sump[ja][jw] = sump[ja][jw] +
                                           P[jr + (j + ja) * ldp] * work[(jw + 2) * n + jr];
                        }
                    }
                }

                for (ja = 0; ja < na; ja++) {
                    if (ilcplx) {
                        sum[ja][0] = -acoef * sums[ja][0] +
                                     bcoefr * sump[ja][0] -
                                     bcoefi * sump[ja][1];
                        sum[ja][1] = -acoef * sums[ja][1] +
                                     bcoefr * sump[ja][1] +
                                     bcoefi * sump[ja][0];
                    } else {
                        sum[ja][0] = -acoef * sums[ja][0] +
                                     bcoefr * sump[ja][0];
                    }
                }

                slaln2(1, na, nw, dmin, acoef, &S[j + j * lds], lds,
                       bdiag[0], bdiag[1], &sum[0][0], 2, bcoefr,
                       bcoefi, &work[2 * n + j], n, &scale, &temp, &iinfo);
                if (scale < ONE) {
                    for (jw = 0; jw < nw; jw++) {
                        for (jr = je; jr < j; jr++)
                            work[(jw + 2) * n + jr] = scale * work[(jw + 2) * n + jr];
                    }
                    xmax = scale * xmax;
                }
                xmax = fmaxf(xmax, temp);
            }

            ieig++;
            if (ilback) {
                for (jw = 0; jw < nw; jw++) {
                    cblas_sgemv(CblasColMajor, CblasNoTrans, n, n - je, ONE,
                                &VL[0 + je * ldvl], ldvl,
                                &work[(jw + 2) * n + je], 1, ZERO,
                                &work[(jw + 4) * n + 0], 1);
                }
                slacpy(" ", n, nw, &work[4 * n + 0], n, &VL[0 + (je) * ldvl], ldvl);
                ibeg = 0;
            } else {
                slacpy(" ", n, nw, &work[2 * n + 0], n, &VL[0 + (ieig - 1) * ldvl], ldvl);
                ibeg = je;
            }

            xmax = ZERO;
            if (ilcplx) {
                for (j = ibeg; j < n; j++)
                    xmax = fmaxf(xmax, fabsf(VL[j + (ieig - 1) * ldvl]) +
                                      fabsf(VL[j + ieig * ldvl]));
            } else {
                for (j = ibeg; j < n; j++)
                    xmax = fmaxf(xmax, fabsf(VL[j + (ieig - 1) * ldvl]));
            }

            if (xmax > safmin) {
                xscale = ONE / xmax;
                for (jw = 0; jw < nw; jw++) {
                    for (jr = ibeg; jr < n; jr++)
                        VL[jr + (ieig - 1 + jw) * ldvl] = xscale * VL[jr + (ieig - 1 + jw) * ldvl];
                }
            }
            ieig = ieig + nw - 1;
        }
    }

    /* Right eigenvectors */
    if (compr) {
        ieig = im;

        ilcplx = 0;
        for (je = n - 1; je >= 0; je--) {
            if (ilcplx) {
                ilcplx = 0;
                continue;
            }
            nw = 1;
            if (je > 0) {
                if (S[je + (je - 1) * lds] != ZERO) {
                    ilcplx = 1;
                    nw = 2;
                }
            }
            if (ilall) {
                ilcomp = 1;
            } else if (ilcplx) {
                ilcomp = select[je] || select[je - 1];
            } else {
                ilcomp = select[je];
            }
            if (!ilcomp)
                continue;

            if (!ilcplx) {
                if (fabsf(S[je + je * lds]) <= safmin &&
                    fabsf(P[je + je * ldp]) <= safmin) {
                    ieig--;
                    for (jr = 0; jr < n; jr++)
                        VR[jr + ieig * ldvr] = ZERO;
                    VR[ieig + ieig * ldvr] = ONE;
                    continue;
                }
            }

            for (jw = 0; jw < nw; jw++) {
                for (jr = 0; jr < n; jr++)
                    work[(jw + 2) * n + jr] = ZERO;
            }

            if (!ilcplx) {
                temp = ONE / fmaxf(fmaxf(fabsf(S[je + je * lds]) * ascale,
                                       fabsf(P[je + je * ldp]) * bscale), safmin);
                salfar = (temp * S[je + je * lds]) * ascale;
                sbeta = (temp * P[je + je * ldp]) * bscale;
                acoef = sbeta * ascale;
                bcoefr = salfar * bscale;
                bcoefi = ZERO;

                scale = ONE;
                int lsa = fabsf(sbeta) >= safmin && fabsf(acoef) < small;
                int lsb = fabsf(salfar) >= safmin && fabsf(bcoefr) < small;
                if (lsa)
                    scale = (small / fabsf(sbeta)) * fminf(anorm, big);
                if (lsb)
                    scale = fmaxf(scale, (small / fabsf(salfar)) * fminf(bnorm, big));
                if (lsa || lsb) {
                    scale = fminf(scale, ONE / (safmin * fmaxf(fmaxf(ONE, fabsf(acoef)), fabsf(bcoefr))));
                    if (lsa) {
                        acoef = ascale * (scale * sbeta);
                    } else {
                        acoef = scale * acoef;
                    }
                    if (lsb) {
                        bcoefr = bscale * (scale * salfar);
                    } else {
                        bcoefr = scale * bcoefr;
                    }
                }
                acoefa = fabsf(acoef);
                bcoefa = fabsf(bcoefr);

                work[2 * n + je] = ONE;
                xmax = ONE;

                for (jr = 0; jr < je; jr++)
                    work[2 * n + jr] = bcoefr * P[jr + je * ldp] - acoef * S[jr + je * lds];
            } else {
                slag2(&S[(je - 1) + (je - 1) * lds], lds, &P[(je - 1) + (je - 1) * ldp], ldp,
                      safmin * SAFETY, &acoef, &temp, &bcoefr, &temp2, &bcoefi);
                if (bcoefi == ZERO) {
                    *info = je;
                    return;
                }

                acoefa = fabsf(acoef);
                bcoefa = fabsf(bcoefr) + fabsf(bcoefi);
                scale = ONE;
                if (acoefa * ulp < safmin && acoefa >= safmin)
                    scale = (safmin / ulp) / acoefa;
                if (bcoefa * ulp < safmin && bcoefa >= safmin)
                    scale = fmaxf(scale, (safmin / ulp) / bcoefa);
                if (safmin * acoefa > ascale)
                    scale = ascale / (safmin * acoefa);
                if (safmin * bcoefa > bscale)
                    scale = fminf(scale, bscale / (safmin * bcoefa));
                if (scale != ONE) {
                    acoef = scale * acoef;
                    acoefa = fabsf(acoef);
                    bcoefr = scale * bcoefr;
                    bcoefi = scale * bcoefi;
                    bcoefa = fabsf(bcoefr) + fabsf(bcoefi);
                }

                temp = acoef * S[je + (je - 1) * lds];
                temp2r = acoef * S[je + je * lds] - bcoefr * P[je + je * ldp];
                temp2i = -bcoefi * P[je + je * ldp];
                if (fabsf(temp) >= fabsf(temp2r) + fabsf(temp2i)) {
                    work[2 * n + je] = ONE;
                    work[3 * n + je] = ZERO;
                    work[2 * n + je - 1] = -temp2r / temp;
                    work[3 * n + je - 1] = -temp2i / temp;
                } else {
                    work[2 * n + je - 1] = ONE;
                    work[3 * n + je - 1] = ZERO;
                    temp = acoef * S[(je - 1) + je * lds];
                    work[2 * n + je] = (bcoefr * P[(je - 1) + (je - 1) * ldp] -
                                        acoef * S[(je - 1) + (je - 1) * lds]) / temp;
                    work[3 * n + je] = bcoefi * P[(je - 1) + (je - 1) * ldp] / temp;
                }

                xmax = fmaxf(fabsf(work[2 * n + je]) + fabsf(work[3 * n + je]),
                            fabsf(work[2 * n + je - 1]) + fabsf(work[3 * n + je - 1]));

                creala = acoef * work[2 * n + je - 1];
                cimaga = acoef * work[3 * n + je - 1];
                crealb = bcoefr * work[2 * n + je - 1] - bcoefi * work[3 * n + je - 1];
                cimagb = bcoefi * work[2 * n + je - 1] + bcoefr * work[3 * n + je - 1];
                cre2a = acoef * work[2 * n + je];
                cim2a = acoef * work[3 * n + je];
                cre2b = bcoefr * work[2 * n + je] - bcoefi * work[3 * n + je];
                cim2b = bcoefi * work[2 * n + je] + bcoefr * work[3 * n + je];
                for (jr = 0; jr < je - 1; jr++) {
                    work[2 * n + jr] = -creala * S[jr + (je - 1) * lds] +
                                       crealb * P[jr + (je - 1) * ldp] -
                                       cre2a * S[jr + je * lds] + cre2b * P[jr + je * ldp];
                    work[3 * n + jr] = -cimaga * S[jr + (je - 1) * lds] +
                                       cimagb * P[jr + (je - 1) * ldp] -
                                       cim2a * S[jr + je * lds] + cim2b * P[jr + je * ldp];
                }
            }

            dmin = fmaxf(fmaxf(ulp * acoefa * anorm, ulp * bcoefa * bnorm), safmin);

            il2by2 = 0;
            for (j = je - nw; j >= 0; j--) {
                if (!il2by2 && j > 0) {
                    if (S[j + (j - 1) * lds] != ZERO) {
                        il2by2 = 1;
                        continue;
                    }
                }
                bdiag[0] = P[j + j * ldp];
                if (il2by2) {
                    na = 2;
                    bdiag[1] = P[(j + 1) + (j + 1) * ldp];
                } else {
                    na = 1;
                }

                slaln2(0, na, nw, dmin, acoef, &S[j + j * lds], lds,
                       bdiag[0], bdiag[1], &work[2 * n + j], n, bcoefr,
                       bcoefi, &sum[0][0], 2, &scale, &temp, &iinfo);
                if (scale < ONE) {
                    for (jw = 0; jw < nw; jw++) {
                        for (jr = 0; jr <= je; jr++)
                            work[(jw + 2) * n + jr] = scale * work[(jw + 2) * n + jr];
                    }
                }
                xmax = fmaxf(scale * xmax, temp);

                for (jw = 0; jw < nw; jw++) {
                    for (ja = 0; ja < na; ja++)
                        work[(jw + 2) * n + j + ja] = sum[ja][jw];
                }

                if (j > 0) {
                    xscale = ONE / fmaxf(ONE, xmax);
                    temp = acoefa * work[j] + bcoefa * work[n + j];
                    if (il2by2)
                        temp = fmaxf(temp, acoefa * work[j + 1] + bcoefa * work[n + j + 1]);
                    temp = fmaxf(fmaxf(temp, acoefa), bcoefa);
                    if (temp > bignum * xscale) {
                        for (jw = 0; jw < nw; jw++) {
                            for (jr = 0; jr <= je; jr++)
                                work[(jw + 2) * n + jr] = xscale * work[(jw + 2) * n + jr];
                        }
                        xmax = xmax * xscale;
                    }

                    for (ja = 0; ja < na; ja++) {
                        if (ilcplx) {
                            creala = acoef * work[2 * n + j + ja];
                            cimaga = acoef * work[3 * n + j + ja];
                            crealb = bcoefr * work[2 * n + j + ja] - bcoefi * work[3 * n + j + ja];
                            cimagb = bcoefi * work[2 * n + j + ja] + bcoefr * work[3 * n + j + ja];
                            for (jr = 0; jr < j; jr++) {
                                work[2 * n + jr] = work[2 * n + jr] -
                                                   creala * S[jr + (j + ja) * lds] +
                                                   crealb * P[jr + (j + ja) * ldp];
                                work[3 * n + jr] = work[3 * n + jr] -
                                                   cimaga * S[jr + (j + ja) * lds] +
                                                   cimagb * P[jr + (j + ja) * ldp];
                            }
                        } else {
                            creala = acoef * work[2 * n + j + ja];
                            crealb = bcoefr * work[2 * n + j + ja];
                            for (jr = 0; jr < j; jr++) {
                                work[2 * n + jr] = work[2 * n + jr] -
                                                   creala * S[jr + (j + ja) * lds] +
                                                   crealb * P[jr + (j + ja) * ldp];
                            }
                        }
                    }
                }

                il2by2 = 0;
            }

            ieig = ieig - nw;
            if (ilback) {
                for (jw = 0; jw < nw; jw++) {
                    for (jr = 0; jr < n; jr++)
                        work[(jw + 4) * n + jr] = work[(jw + 2) * n + 0] * VR[jr + 0 * ldvr];

                    for (int jc = 1; jc <= je; jc++) {
                        for (jr = 0; jr < n; jr++)
                            work[(jw + 4) * n + jr] = work[(jw + 4) * n + jr] +
                                                      work[(jw + 2) * n + jc] * VR[jr + jc * ldvr];
                    }
                }

                for (jw = 0; jw < nw; jw++) {
                    for (jr = 0; jr < n; jr++)
                        VR[jr + (ieig + jw) * ldvr] = work[(jw + 4) * n + jr];
                }

                iend = n;
            } else {
                for (jw = 0; jw < nw; jw++) {
                    for (jr = 0; jr < n; jr++)
                        VR[jr + (ieig + jw) * ldvr] = work[(jw + 2) * n + jr];
                }

                iend = je + 1;
            }

            xmax = ZERO;
            if (ilcplx) {
                for (j = 0; j < iend; j++)
                    xmax = fmaxf(xmax, fabsf(VR[j + ieig * ldvr]) + fabsf(VR[j + (ieig + 1) * ldvr]));
            } else {
                for (j = 0; j < iend; j++)
                    xmax = fmaxf(xmax, fabsf(VR[j + ieig * ldvr]));
            }

            if (xmax > safmin) {
                xscale = ONE / xmax;
                for (jw = 0; jw < nw; jw++) {
                    for (jr = 0; jr < iend; jr++)
                        VR[jr + (ieig + jw) * ldvr] = xscale * VR[jr + (ieig + jw) * ldvr];
                }
            }
        }
    }
}
