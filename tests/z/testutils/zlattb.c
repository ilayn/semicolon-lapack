/**
 * @file zlattb.c
 * @brief ZLATTB generates a triangular test matrix in 2-dimensional banded
 *        storage.
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"
#include "test_rng.h"

void zlattb(const INT imat, const char* uplo, const char* trans, char* diag,
            const INT n, const INT kd, c128* AB, const INT ldab,
            c128* B, c128* work, f64* rwork, INT* info,
            uint64_t state[static 4])
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const f64 TWO = 2.0;

    INT upper;
    char type, dist, packit;
    INT kl, ku, mode;
    f64 anorm, cndnum;
    INT i, j, jcount, iy, lenj, ioff;
    f64 unfl, ulp, smlnum, bignum;
    f64 bnorm, bscal, tscal, texp, tleft, tnorm;
    c128 plus1, plus2, star1;
    f64 sfac, rexp;

    *info = 0;

    if ((imat >= 6 && imat <= 9) || imat == 17) {
        *diag = 'U';
    } else {
        *diag = 'N';
    }

    if (n <= 0)
        return;

    unfl = dlamch("S");
    ulp = dlamch("E") * dlamch("B");
    smlnum = unfl;
    bignum = (ONE - ulp) / smlnum;

    upper = (uplo[0] == 'U' || uplo[0] == 'u');

    if (upper) {
        zlatb4("ZTB", imat, n, n, &type, &kl, &ku, &anorm, &mode, &cndnum, &dist);
        ku = kd;
        ioff = (kd > n - 1) ? kd - n + 1 : 0;
        kl = 0;
        packit = 'Q';
    } else {
        zlatb4("ZTB", -imat, n, n, &type, &kl, &ku, &anorm, &mode, &cndnum, &dist);
        kl = kd;
        ioff = 0;
        ku = 0;
        packit = 'B';
    }

    if (imat <= 5) {
        char symm[2] = {type, '\0'};
        char dstr[2] = {dist, '\0'};
        char pack[2] = {packit, '\0'};
        zlatms(n, n, dstr, symm, rwork, mode, cndnum, anorm,
               kl, ku, pack, &AB[ioff], ldab, work, info, state);

    } else if (imat == 6) {
        if (upper) {
            for (j = 0; j < n; j++) {
                for (i = (kd - j > 0 ? kd - j : 0); i < kd; i++) {
                    AB[i + j * ldab] = ZERO;
                }
                AB[kd + j * ldab] = (f64)(j + 1);
            }
        } else {
            for (j = 0; j < n; j++) {
                AB[0 + j * ldab] = (f64)(j + 1);
                for (i = 1; i < (kd + 1 < n - j ? kd + 1 : n - j); i++) {
                    AB[i + j * ldab] = ZERO;
                }
            }
        }

    } else if (imat <= 9) {
        tnorm = sqrt(cndnum);

        if (upper) {
            for (j = 0; j < n; j++) {
                for (i = (kd - j > 0 ? kd - j : 0); i < kd; i++) {
                    AB[i + j * ldab] = ZERO;
                }
                AB[kd + j * ldab] = (f64)(j + 1);
            }
        } else {
            for (j = 0; j < n; j++) {
                for (i = 1; i < (kd + 1 < n - j ? kd + 1 : n - j); i++) {
                    AB[i + j * ldab] = ZERO;
                }
                AB[0 + j * ldab] = (f64)(j + 1);
            }
        }

        if (kd == 1) {
            if (upper) {
                AB[0 + 1 * ldab] = tnorm * zlarnd_rng(5, state);
                lenj = (n - 3) / 2;
                zlarnv_rng(2, lenj, work, state);
                for (j = 0; j < lenj; j++) {
                    AB[0 + 2 * (j + 2) * ldab] = tnorm * work[j];
                }
            } else {
                AB[1 + 0 * ldab] = tnorm * zlarnd_rng(5, state);
                lenj = (n - 3) / 2;
                zlarnv_rng(2, lenj, work, state);
                for (j = 0; j < lenj; j++) {
                    AB[1 + (2 * j + 2) * ldab] = tnorm * work[j];
                }
            }
        } else if (kd > 1) {
            star1 = tnorm * zlarnd_rng(5, state);
            sfac = sqrt(tnorm);
            plus1 = sfac * zlarnd_rng(5, state);
            for (j = 0; j < n; j += 2) {
                plus2 = star1 / plus1;
                work[j] = plus1;
                work[n + j] = star1;
                if (j + 1 < n) {
                    work[j + 1] = plus2;
                    work[n + j + 1] = ZERO;
                    plus1 = star1 / plus2;

                    rng_fill(state, 2, 1, &rexp);
                    if (rexp < ZERO) {
                        star1 = -pow(sfac, ONE - rexp) * zlarnd_rng(5, state);
                    } else {
                        star1 = pow(sfac, ONE + rexp) * zlarnd_rng(5, state);
                    }
                }
            }

            if (upper) {
                cblas_zcopy(n - 1, work, 1, &AB[(kd - 1) + 1 * ldab], ldab);
                cblas_zcopy(n - 2, &work[n], 1, &AB[(kd - 2) + 2 * ldab], ldab);
            } else {
                cblas_zcopy(n - 1, work, 1, &AB[1 + 0 * ldab], ldab);
                cblas_zcopy(n - 2, &work[n], 1, &AB[2 + 0 * ldab], ldab);
            }
        }

    } else if (imat == 10) {
        if (upper) {
            for (j = 0; j < n; j++) {
                lenj = (j < kd ? j : kd);
                zlarnv_rng(4, lenj, &AB[(kd - lenj) + j * ldab], state);
                AB[kd + j * ldab] = zlarnd_rng(5, state) * TWO;
            }
        } else {
            for (j = 0; j < n; j++) {
                lenj = (n - j - 1 < kd ? n - j - 1 : kd);
                if (lenj > 0)
                    zlarnv_rng(4, lenj, &AB[1 + j * ldab], state);
                AB[0 + j * ldab] = zlarnd_rng(5, state) * TWO;
            }
        }

        zlarnv_rng(2, n, B, state);
        iy = cblas_izamax(n, B, 1);
        bnorm = cabs(B[iy]);
        bscal = bignum / (ONE > bnorm ? ONE : bnorm);
        cblas_zdscal(n, bscal, B, 1);

    } else if (imat == 11) {
        zlarnv_rng(2, n, B, state);
        tscal = ONE / (f64)(kd + 1);
        if (upper) {
            for (j = 0; j < n; j++) {
                lenj = (j < kd ? j : kd);
                if (lenj > 0) {
                    zlarnv_rng(4, lenj, &AB[(kd + 1 - lenj) + j * ldab], state);
                    cblas_zdscal(lenj, tscal, &AB[(kd + 1 - lenj) + j * ldab], 1);
                }
                AB[kd + j * ldab] = zlarnd_rng(5, state);
            }
            AB[kd + (n - 1) * ldab] = smlnum * AB[kd + (n - 1) * ldab];
        } else {
            for (j = 0; j < n; j++) {
                lenj = (n - j - 1 < kd ? n - j - 1 : kd);
                if (lenj > 0) {
                    zlarnv_rng(4, lenj, &AB[1 + j * ldab], state);
                    cblas_zdscal(lenj, tscal, &AB[1 + j * ldab], 1);
                }
                AB[0 + j * ldab] = zlarnd_rng(5, state);
            }
            AB[0 + 0 * ldab] = smlnum * AB[0 + 0 * ldab];
        }

    } else if (imat == 12) {
        zlarnv_rng(2, n, B, state);
        if (upper) {
            for (j = 0; j < n; j++) {
                lenj = (j < kd ? j : kd);
                if (lenj > 0)
                    zlarnv_rng(4, lenj, &AB[(kd + 1 - lenj) + j * ldab], state);
                AB[kd + j * ldab] = zlarnd_rng(5, state);
            }
            AB[kd + (n - 1) * ldab] = smlnum * AB[kd + (n - 1) * ldab];
        } else {
            for (j = 0; j < n; j++) {
                lenj = (n - j - 1 < kd ? n - j - 1 : kd);
                if (lenj > 0)
                    zlarnv_rng(4, lenj, &AB[1 + j * ldab], state);
                AB[0 + j * ldab] = zlarnd_rng(5, state);
            }
            AB[0 + 0 * ldab] = smlnum * AB[0 + 0 * ldab];
        }

    } else if (imat == 13) {
        if (upper) {
            jcount = 1;
            for (j = n - 1; j >= 0; j--) {
                for (i = (kd - j > 0 ? kd - j : 0); i < kd; i++) {
                    AB[i + j * ldab] = ZERO;
                }
                if (jcount <= 2) {
                    AB[kd + j * ldab] = smlnum * zlarnd_rng(5, state);
                } else {
                    AB[kd + j * ldab] = zlarnd_rng(5, state);
                }
                jcount++;
                if (jcount > 4)
                    jcount = 1;
            }
        } else {
            jcount = 1;
            for (j = 0; j < n; j++) {
                for (i = 1; i < (n - j < kd + 1 ? n - j : kd + 1); i++) {
                    AB[i + j * ldab] = ZERO;
                }
                if (jcount <= 2) {
                    AB[0 + j * ldab] = smlnum * zlarnd_rng(5, state);
                } else {
                    AB[0 + j * ldab] = zlarnd_rng(5, state);
                }
                jcount++;
                if (jcount > 4)
                    jcount = 1;
            }
        }

        if (upper) {
            B[0] = ZERO;
            for (i = n - 1; i >= 1; i -= 2) {
                B[i] = ZERO;
                B[i - 1] = smlnum * zlarnd_rng(5, state);
            }
        } else {
            B[n - 1] = ZERO;
            for (i = 0; i < n - 1; i += 2) {
                B[i] = ZERO;
                B[i + 1] = smlnum * zlarnd_rng(5, state);
            }
        }

    } else if (imat == 14) {
        texp = ONE / (f64)(kd + 1);
        tscal = pow(smlnum, texp);
        zlarnv_rng(4, n, B, state);
        if (upper) {
            for (j = 0; j < n; j++) {
                for (i = (kd - j > 0 ? kd - j : 0); i < kd; i++) {
                    AB[i + j * ldab] = ZERO;
                }
                if (j > 0 && kd > 0)
                    AB[(kd - 1) + j * ldab] = CMPLX(-ONE, -ONE);
                AB[kd + j * ldab] = tscal * zlarnd_rng(5, state);
            }
            B[n - 1] = CMPLX(ONE, ONE);
        } else {
            for (j = 0; j < n; j++) {
                for (i = 2; i < (n - j < kd + 1 ? n - j : kd + 1); i++) {
                    AB[i + j * ldab] = ZERO;
                }
                if (j < n - 1 && kd > 0)
                    AB[1 + j * ldab] = CMPLX(-ONE, -ONE);
                AB[0 + j * ldab] = tscal * zlarnd_rng(5, state);
            }
            B[0] = CMPLX(ONE, ONE);
        }

    } else if (imat == 15) {
        iy = n / 2;
        if (upper) {
            for (j = 0; j < n; j++) {
                lenj = (j + 1 < kd + 1 ? j + 1 : kd + 1);
                zlarnv_rng(4, lenj, &AB[(kd + 1 - lenj) + j * ldab], state);
                if (j != iy) {
                    AB[kd + j * ldab] = zlarnd_rng(5, state) * TWO;
                } else {
                    AB[kd + j * ldab] = ZERO;
                }
            }
        } else {
            for (j = 0; j < n; j++) {
                lenj = (n - j < kd + 1 ? n - j : kd + 1);
                zlarnv_rng(4, lenj, &AB[0 + j * ldab], state);
                if (j != iy) {
                    AB[0 + j * ldab] = zlarnd_rng(5, state) * TWO;
                } else {
                    AB[0 + j * ldab] = ZERO;
                }
            }
        }
        zlarnv_rng(2, n, B, state);
        cblas_zdscal(n, TWO, B, 1);

    } else if (imat == 16) {
        tscal = unfl / ulp;
        tscal = (ONE - ulp) / tscal;
        for (j = 0; j < n; j++) {
            for (i = 0; i <= kd; i++) {
                AB[i + j * ldab] = ZERO;
            }
        }
        texp = ONE;
        if (kd > 0) {
            if (upper) {
                for (j = n - 1; j >= 0; j -= kd) {
                    for (i = j; i >= (j - kd + 1 > 0 ? j - kd + 1 : 0); i -= 2) {
                        AB[(j - i) + i * ldab] = -tscal / (f64)(kd + 2);
                        AB[kd + i * ldab] = ONE;
                        B[i] = texp * (ONE - ulp);
                        if (i > (j - kd + 1 > 0 ? j - kd + 1 : 0)) {
                            AB[(j - i + 1) + (i - 1) * ldab] = -(tscal / (f64)(kd + 2)) / (f64)(kd + 3);
                            AB[kd + (i - 1) * ldab] = ONE;
                            B[i - 1] = texp * (f64)((kd + 1) * (kd + 1) + kd);
                        }
                        texp = texp * TWO;
                    }
                    B[(j - kd + 1 > 0 ? j - kd + 1 : 0)] = ((f64)(kd + 2) / (f64)(kd + 3)) * tscal;
                }
            } else {
                for (j = 0; j < n; j += kd) {
                    texp = ONE;
                    lenj = (kd + 1 < n - j ? kd + 1 : n - j);
                    for (i = j; i < (n - 1 < j + kd - 1 ? n - 1 : j + kd - 1); i += 2) {
                        AB[(lenj - 1 - (i - j)) + j * ldab] = -tscal / (f64)(kd + 2);
                        AB[0 + j * ldab] = ONE;
                        B[j] = texp * (ONE - ulp);
                        if (i < (n - 1 < j + kd - 1 ? n - 1 : j + kd - 1)) {
                            AB[(lenj - 1 - (i - j + 1)) + (i + 1) * ldab] = -(tscal / (f64)(kd + 2)) / (f64)(kd + 3);
                            AB[0 + (i + 1) * ldab] = ONE;
                            B[i + 1] = texp * (f64)((kd + 1) * (kd + 1) + kd);
                        }
                        texp = texp * TWO;
                    }
                    B[(n - 1 < j + kd - 1 ? n - 1 : j + kd - 1)] = ((f64)(kd + 2) / (f64)(kd + 3)) * tscal;
                }
            }
        }

    } else if (imat == 17) {
        if (upper) {
            for (j = 0; j < n; j++) {
                lenj = (j < kd ? j : kd);
                zlarnv_rng(4, lenj, &AB[(kd - lenj) + j * ldab], state);
                AB[kd + j * ldab] = (f64)(j + 1);
            }
        } else {
            for (j = 0; j < n; j++) {
                lenj = (n - j - 1 < kd ? n - j - 1 : kd);
                if (lenj > 0)
                    zlarnv_rng(4, lenj, &AB[1 + j * ldab], state);
                AB[0 + j * ldab] = (f64)(j + 1);
            }
        }

        zlarnv_rng(2, n, B, state);
        iy = cblas_izamax(n, B, 1);
        bnorm = cabs(B[iy]);
        bscal = bignum / (ONE > bnorm ? ONE : bnorm);
        cblas_zdscal(n, bscal, B, 1);

    } else if (imat == 18) {
        tleft = bignum / (f64)(kd + 1);
        tscal = bignum * ((f64)(kd + 1) / (f64)(kd + 2));
        if (upper) {
            for (j = 0; j < n; j++) {
                lenj = (j + 1 < kd + 1 ? j + 1 : kd + 1);
                zlarnv_rng(5, lenj, &AB[(kd + 1 - lenj) + j * ldab], state);
                rng_fill(state, 1, lenj, &rwork[kd + 1 - lenj]);
                for (i = kd + 1 - lenj; i <= kd; i++) {
                    AB[i + j * ldab] = AB[i + j * ldab] * (tleft + rwork[i] * tscal);
                }
            }
        } else {
            for (j = 0; j < n; j++) {
                lenj = (n - j < kd + 1 ? n - j : kd + 1);
                zlarnv_rng(5, lenj, &AB[0 + j * ldab], state);
                rng_fill(state, 1, lenj, rwork);
                for (i = 0; i < lenj; i++) {
                    AB[i + j * ldab] = AB[i + j * ldab] * (tleft + rwork[i] * tscal);
                }
            }
        }
        zlarnv_rng(2, n, B, state);
        cblas_zdscal(n, TWO, B, 1);
    }

    if (!(trans[0] == 'N' || trans[0] == 'n')) {
        if (upper) {
            for (j = 0; j < n / 2; j++) {
                lenj = (n - 2 * j - 1 < kd + 1 ? n - 2 * j - 1 : kd + 1);
                cblas_zswap(lenj, &AB[kd + j * ldab], ldab - 1,
                            &AB[(kd + 1 - lenj) + (n - 1 - j) * ldab], -1);
            }
        } else {
            for (j = 0; j < n / 2; j++) {
                lenj = (n - 2 * j - 1 < kd + 1 ? n - 2 * j - 1 : kd + 1);
                cblas_zswap(lenj, &AB[0 + j * ldab], 1,
                            &AB[(lenj - 1) + (n - j - lenj) * ldab], -ldab + 1);
            }
        }
    }
}
