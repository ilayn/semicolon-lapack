/**
 * @file zlattp.c
 * @brief ZLATTP generates a triangular test matrix in packed storage.
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"
#include "test_rng.h"

void zlattp(const INT imat, const char* uplo, const char* trans, char* diag,
            const INT n, c128* AP, c128* B, c128* work, f64* rwork,
            INT* info, uint64_t state[static 4])
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const f64 TWO = 2.0;

    f64 unfl, ulp, smlnum, bignum;
    INT upper;
    char type, dist, packit;
    INT kl, ku, mode;
    f64 anorm, cndnum;
    INT i, j, jc, jcnext, jcount, jj, jl, jr, jx, iy;
    f64 bnorm, bscal, tscal, texp, tleft;
    c128 plus1, plus2, star1, ctemp, ra, rb, s;
    f64 sfac, rexp, c;
    f64 x, y, z, t;

    *info = 0;

    if ((imat >= 7 && imat <= 10) || imat == 18) {
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
        zlatb4("ZTP", imat, n, n, &type, &kl, &ku, &anorm, &mode, &cndnum, &dist);
        packit = 'C';
    } else {
        zlatb4("ZTP", -imat, n, n, &type, &kl, &ku, &anorm, &mode, &cndnum, &dist);
        packit = 'R';
    }

    if (imat <= 6) {
        char symm[2] = {type, '\0'};
        char dstr[2] = {dist, '\0'};
        char pack[2] = {packit, '\0'};
        zlatms(n, n, dstr, symm, rwork, mode, cndnum, anorm,
               kl, ku, pack, AP, n, work, info, state);

    } else if (imat == 7) {
        if (upper) {
            jc = 0;
            for (j = 0; j < n; j++) {
                for (i = 0; i < j; i++) {
                    AP[jc + i] = ZERO;
                }
                AP[jc + j] = (f64)(j + 1);
                jc += j + 1;
            }
        } else {
            jc = 0;
            for (j = 0; j < n; j++) {
                AP[jc] = (f64)(j + 1);
                for (i = j + 1; i < n; i++) {
                    AP[jc + i - j] = ZERO;
                }
                jc += n - j;
            }
        }

    } else if (imat <= 10) {
        if (upper) {
            jc = 0;
            for (j = 0; j < n; j++) {
                for (i = 0; i < j; i++) {
                    AP[jc + i] = ZERO;
                }
                AP[jc + j] = (f64)(j + 1);
                jc += j + 1;
            }
        } else {
            jc = 0;
            for (j = 0; j < n; j++) {
                AP[jc] = (f64)(j + 1);
                for (i = j + 1; i < n; i++) {
                    AP[jc + i - j] = ZERO;
                }
                jc += n - j;
            }
        }

        star1 = 0.25 * zlarnd_rng(5, state);
        sfac = 0.5;
        plus1 = sfac * zlarnd_rng(5, state);
        for (j = 0; j < n; j += 2) {
            plus2 = star1 / plus1;
            work[j] = plus1;
            work[n + j] = star1;
            if (j + 1 < n) {
                work[j + 1] = plus2;
                work[n + j + 1] = ZERO;
                plus1 = star1 / plus2;
                rexp = creal(zlarnd_rng(2, state));
                if (rexp < ZERO) {
                    star1 = -pow(sfac, ONE - rexp) * zlarnd_rng(5, state);
                } else {
                    star1 = pow(sfac, ONE + rexp) * zlarnd_rng(5, state);
                }
            }
        }

        x = sqrt(cndnum) - ONE / sqrt(cndnum);
        if (n > 2) {
            y = sqrt(TWO / (f64)(n - 2)) * x;
        } else {
            y = ZERO;
        }
        z = x * x;

        if (upper) {
            jc = 0;
            for (j = 1; j < n; j++) {
                AP[jc + 1] = y;
                if (j > 1)
                    AP[jc + j] = work[j - 2];
                if (j > 2)
                    AP[jc + j - 1] = work[n + j - 3];
                jc += j + 1;
            }
            jc -= n;
            AP[jc + 1] = z;
            for (j = 1; j < n - 1; j++) {
                AP[jc + j + 1] = y;
            }
        } else {
            for (i = 1; i < n - 1; i++) {
                AP[i] = y;
            }
            AP[n - 1] = z;
            jc = n;
            for (j = 1; j < n - 1; j++) {
                AP[jc + 1] = work[j - 1];
                if (j < n - 2)
                    AP[jc + 2] = work[n + j - 1];
                AP[jc + n - j - 1] = y;
                jc += n - j;
            }
        }

        if (upper) {
            jc = 0;
            for (j = 0; j < n - 1; j++) {
                jcnext = jc + j + 1;
                ra = AP[jcnext + j];
                rb = TWO;
                cblas_zrotg(&ra, &rb, &c, &s);

                if (n > j + 2) {
                    jx = jcnext + j + 2;
                    for (i = j + 2; i < n; i++) {
                        ctemp = c * AP[jx + j] + s * AP[jx + j + 1];
                        AP[jx + j + 1] = -conj(s) * AP[jx + j] + c * AP[jx + j + 1];
                        AP[jx + j] = ctemp;
                        jx += i + 1;
                    }
                }

                if (j > 0)
                    zrot(j, &AP[jcnext], 1, &AP[jc], 1, -c, -s);

                AP[jcnext + j] = -AP[jcnext + j];
                jc = jcnext;
            }
        } else {
            jc = 0;
            for (j = 0; j < n - 1; j++) {
                jcnext = jc + n - j;
                ra = AP[jc + 1];
                rb = TWO;
                cblas_zrotg(&ra, &rb, &c, &s);
                s = conj(s);

                if (n > j + 2)
                    zrot(n - j - 2, &AP[jcnext + 1], 1, &AP[jc + 2], 1, c, -s);

                if (j > 0) {
                    jx = 0;
                    for (i = 0; i < j; i++) {
                        ctemp = -c * AP[jx + j - i] + s * AP[jx + j - i + 1];
                        AP[jx + j - i + 1] = -conj(s) * AP[jx + j - i] - c * AP[jx + j - i + 1];
                        AP[jx + j - i] = ctemp;
                        jx += n - i;
                    }
                }

                AP[jc + 1] = -AP[jc + 1];
                jc = jcnext;
            }
        }

    } else if (imat == 11) {
        if (upper) {
            jc = 0;
            for (j = 0; j < n; j++) {
                zlarnv_rng(4, j, &AP[jc], state);
                AP[jc + j] = zlarnd_rng(5, state) * TWO;
                jc += j + 1;
            }
        } else {
            jc = 0;
            for (j = 0; j < n; j++) {
                if (j < n - 1)
                    zlarnv_rng(4, n - j - 1, &AP[jc + 1], state);
                AP[jc] = zlarnd_rng(5, state) * TWO;
                jc += n - j;
            }
        }

        zlarnv_rng(2, n, B, state);
        iy = cblas_izamax(n, B, 1);
        bnorm = cabs(B[iy]);
        bscal = bignum / (ONE > bnorm ? ONE : bnorm);
        cblas_zdscal(n, bscal, B, 1);

    } else if (imat == 12) {
        zlarnv_rng(2, n, B, state);
        tscal = ONE / (ONE > (f64)(n - 1) ? ONE : (f64)(n - 1));
        if (upper) {
            jc = 0;
            for (j = 0; j < n; j++) {
                zlarnv_rng(4, j, &AP[jc], state);
                cblas_zdscal(j, tscal, &AP[jc], 1);
                AP[jc + j] = zlarnd_rng(5, state);
                jc += j + 1;
            }
            AP[n * (n + 1) / 2 - 1] = smlnum * AP[n * (n + 1) / 2 - 1];
        } else {
            jc = 0;
            for (j = 0; j < n; j++) {
                zlarnv_rng(2, n - j - 1, &AP[jc + 1], state);
                cblas_zdscal(n - j - 1, tscal, &AP[jc + 1], 1);
                AP[jc] = zlarnd_rng(5, state);
                jc += n - j;
            }
            AP[0] = smlnum * AP[0];
        }

    } else if (imat == 13) {
        zlarnv_rng(2, n, B, state);
        if (upper) {
            jc = 0;
            for (j = 0; j < n; j++) {
                zlarnv_rng(4, j, &AP[jc], state);
                AP[jc + j] = zlarnd_rng(5, state);
                jc += j + 1;
            }
            AP[n * (n + 1) / 2 - 1] = smlnum * AP[n * (n + 1) / 2 - 1];
        } else {
            jc = 0;
            for (j = 0; j < n; j++) {
                zlarnv_rng(4, n - j - 1, &AP[jc + 1], state);
                AP[jc] = zlarnd_rng(5, state);
                jc += n - j;
            }
            AP[0] = smlnum * AP[0];
        }

    } else if (imat == 14) {
        if (upper) {
            jcount = 1;
            jc = (n - 1) * n / 2;
            for (j = n - 1; j >= 0; j--) {
                for (i = 0; i < j; i++) {
                    AP[jc + i] = ZERO;
                }
                if (jcount <= 2) {
                    AP[jc + j] = smlnum * zlarnd_rng(5, state);
                } else {
                    AP[jc + j] = zlarnd_rng(5, state);
                }
                jcount++;
                if (jcount > 4)
                    jcount = 1;
                jc -= j;
            }
        } else {
            jcount = 1;
            jc = 0;
            for (j = 0; j < n; j++) {
                for (i = j + 1; i < n; i++) {
                    AP[jc + i - j] = ZERO;
                }
                if (jcount <= 2) {
                    AP[jc] = smlnum * zlarnd_rng(5, state);
                } else {
                    AP[jc] = zlarnd_rng(5, state);
                }
                jcount++;
                if (jcount > 4)
                    jcount = 1;
                jc += n - j;
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

    } else if (imat == 15) {
        texp = ONE / (ONE > (f64)(n - 1) ? ONE : (f64)(n - 1));
        tscal = pow(smlnum, texp);
        zlarnv_rng(4, n, B, state);
        if (upper) {
            jc = 0;
            for (j = 0; j < n; j++) {
                for (i = 0; i < j - 1; i++) {
                    AP[jc + i] = ZERO;
                }
                if (j > 0)
                    AP[jc + j - 1] = CMPLX(-ONE, -ONE);
                AP[jc + j] = tscal * zlarnd_rng(5, state);
                jc += j + 1;
            }
            B[n - 1] = CMPLX(ONE, ONE);
        } else {
            jc = 0;
            for (j = 0; j < n; j++) {
                for (i = j + 2; i < n; i++) {
                    AP[jc + i - j] = ZERO;
                }
                if (j < n - 1)
                    AP[jc + 1] = CMPLX(-ONE, -ONE);
                AP[jc] = tscal * zlarnd_rng(5, state);
                jc += n - j;
            }
            B[0] = CMPLX(ONE, ONE);
        }

    } else if (imat == 16) {
        iy = n / 2;
        if (upper) {
            jc = 0;
            for (j = 0; j < n; j++) {
                zlarnv_rng(4, j + 1, &AP[jc], state);
                if (j != iy) {
                    AP[jc + j] = zlarnd_rng(5, state) * TWO;
                } else {
                    AP[jc + j] = ZERO;
                }
                jc += j + 1;
            }
        } else {
            jc = 0;
            for (j = 0; j < n; j++) {
                zlarnv_rng(4, n - j, &AP[jc], state);
                if (j != iy) {
                    AP[jc] = zlarnd_rng(5, state) * TWO;
                } else {
                    AP[jc] = ZERO;
                }
                jc += n - j;
            }
        }
        zlarnv_rng(2, n, B, state);
        cblas_zdscal(n, TWO, B, 1);

    } else if (imat == 17) {
        tscal = unfl / ulp;
        tscal = (ONE - ulp) / tscal;
        for (j = 0; j < n * (n + 1) / 2; j++) {
            AP[j] = ZERO;
        }
        texp = ONE;
        if (upper) {
            jc = (n - 1) * n / 2;
            for (j = n - 1; j >= 1; j -= 2) {
                AP[jc] = -tscal / (f64)(n + 1);
                AP[jc + j] = ONE;
                B[j] = texp * (ONE - ulp);
                jc -= j;
                AP[jc] = -(tscal / (f64)(n + 1)) / (f64)(n + 2);
                AP[jc + j - 1] = ONE;
                B[j - 1] = texp * (f64)(n * n + n - 1);
                texp *= TWO;
                jc -= j - 1;
            }
            B[0] = ((f64)(n + 1) / (f64)(n + 2)) * tscal;
        } else {
            jc = 0;
            for (j = 0; j < n - 1; j += 2) {
                AP[jc + n - j - 1] = -tscal / (f64)(n + 1);
                AP[jc] = ONE;
                B[j] = texp * (ONE - ulp);
                jc += n - j;
                AP[jc + n - j - 2] = -(tscal / (f64)(n + 1)) / (f64)(n + 2);
                AP[jc] = ONE;
                B[j + 1] = texp * (f64)(n * n + n - 1);
                texp *= TWO;
                jc += n - j - 1;
            }
            B[n - 1] = ((f64)(n + 1) / (f64)(n + 2)) * tscal;
        }

    } else if (imat == 18) {
        if (upper) {
            jc = 0;
            for (j = 0; j < n; j++) {
                zlarnv_rng(4, j, &AP[jc], state);
                AP[jc + j] = ZERO;
                jc += j + 1;
            }
        } else {
            jc = 0;
            for (j = 0; j < n; j++) {
                if (j < n - 1)
                    zlarnv_rng(4, n - j - 1, &AP[jc + 1], state);
                AP[jc] = ZERO;
                jc += n - j;
            }
        }

        zlarnv_rng(2, n, B, state);
        iy = cblas_izamax(n, B, 1);
        bnorm = cabs(B[iy]);
        bscal = bignum / (ONE > bnorm ? ONE : bnorm);
        cblas_zdscal(n, bscal, B, 1);

    } else if (imat == 19) {
        tleft = bignum / (ONE > (f64)(n - 1) ? ONE : (f64)(n - 1));
        tscal = bignum * ((f64)(n - 1) / (ONE > (f64)n ? ONE : (f64)n));
        if (upper) {
            jc = 0;
            for (j = 0; j < n; j++) {
                zlarnv_rng(5, j + 1, &AP[jc], state);
                rng_fill(state, 1, j + 1, rwork);
                for (i = 0; i <= j; i++) {
                    AP[jc + i] = AP[jc + i] * (tleft + rwork[i] * tscal);
                }
                jc += j + 1;
            }
        } else {
            jc = 0;
            for (j = 0; j < n; j++) {
                zlarnv_rng(5, n - j, &AP[jc], state);
                rng_fill(state, 1, n - j, rwork);
                for (i = 0; i < n - j; i++) {
                    AP[jc + i] = AP[jc + i] * (tleft + rwork[i] * tscal);
                }
                jc += n - j;
            }
        }
        zlarnv_rng(2, n, B, state);
        cblas_zdscal(n, TWO, B, 1);
    }

    if (!(trans[0] == 'N' || trans[0] == 'n')) {
        if (upper) {
            jj = 0;
            jr = n * (n + 1) / 2 - 1;
            for (j = 0; j < n / 2; j++) {
                jl = jj;
                for (i = j; i < n - j - 1; i++) {
                    t = creal(AP[jr - i + j]);
                    AP[jr - i + j] = AP[jl];
                    AP[jl] = t;
                    jl += i + 1;
                }
                jj += j + 2;
                jr -= n - j;
            }
        } else {
            jl = 0;
            jj = n * (n + 1) / 2 - 1;
            for (j = 0; j < n / 2; j++) {
                jr = jj;
                for (i = j; i < n - j - 1; i++) {
                    t = creal(AP[jl + i - j]);
                    AP[jl + i - j] = AP[jr];
                    AP[jr] = t;
                    jr -= i + 1;
                }
                jl += n - j;
                jj -= j + 2;
            }
        }
    }
}
