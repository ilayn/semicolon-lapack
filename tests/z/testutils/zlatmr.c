/**
 * @file zlatmr.c
 * @brief ZLATMR generates random matrices of various types for testing.
 *
 * Faithful port of LAPACK TESTING/MATGEN/zlatmr.f
 */

#include <complex.h>
#include "semicolon_cblas.h"
#include "verify.h"

/* Local helper: case-insensitive character comparison */
static INT lsame(char c1, char c2) {
    if (c1 == c2) return 1;
    if (c1 >= 'a' && c1 <= 'z') c1 = c1 - 'a' + 'A';
    if (c2 >= 'a' && c2 <= 'z') c2 = c2 - 'a' + 'A';
    return (c1 == c2);
}

static INT min_int(INT a, INT b) { return (a < b) ? a : b; }
static INT max_int(INT a, INT b) { return (a > b) ? a : b; }
static f64 max_dbl(f64 a, f64 b) { return (a > b) ? a : b; }

/**
 * ZLATMR generates random matrices of various types for testing
 * LAPACK programs.
 */
void zlatmr(
    const INT m,
    const INT n,
    const char* dist,
    const char* sym,
    c128* d,
    const INT mode,
    const f64 cond,
    const c128 dmax,
    const char* rsign,
    const char* grade,
    c128* dl,
    const INT model,
    const f64 condl,
    c128* dr,
    const INT moder,
    const f64 condr,
    const char* pivtng,
    const INT* ipivot,
    const INT kl,
    const INT ku,
    const f64 sparse,
    const f64 anorm,
    const char* pack,
    c128* A,
    const INT lda,
    INT* iwork,
    INT* info,
    uint64_t state[static 4])
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const c128 CONE = CMPLX(1.0, 0.0);
    const c128 CZERO = CMPLX(0.0, 0.0);

    INT badpvt, dzero, fulbnd;
    INT i, idist, igrade, iisub, ipack, ipvtng, irsign;
    INT isub, isym, j, jjsub, jsub, k, kll, kuu, mnmin;
    INT mnsub, mxsub, npvts;
    f64 onorm, temp;
    c128 calpha, ctemp;
    f64 tempa[1];

    *info = 0;

    if (m == 0 || n == 0) {
        return;
    }

    if (lsame(dist[0], 'U')) {
        idist = 1;
    } else if (lsame(dist[0], 'S')) {
        idist = 2;
    } else if (lsame(dist[0], 'N')) {
        idist = 3;
    } else if (lsame(dist[0], 'D')) {
        idist = 4;
    } else {
        idist = -1;
    }

    if (lsame(sym[0], 'H')) {
        isym = 0;
    } else if (lsame(sym[0], 'N')) {
        isym = 1;
    } else if (lsame(sym[0], 'S')) {
        isym = 2;
    } else {
        isym = -1;
    }

    if (lsame(rsign[0], 'F')) {
        irsign = 0;
    } else if (lsame(rsign[0], 'T')) {
        irsign = 1;
    } else {
        irsign = -1;
    }

    npvts = 0;
    if (lsame(pivtng[0], 'N')) {
        ipvtng = 0;
    } else if (lsame(pivtng[0], ' ')) {
        ipvtng = 0;
    } else if (lsame(pivtng[0], 'L')) {
        ipvtng = 1;
        npvts = m;
    } else if (lsame(pivtng[0], 'R')) {
        ipvtng = 2;
        npvts = n;
    } else if (lsame(pivtng[0], 'B')) {
        ipvtng = 3;
        npvts = min_int(n, m);
    } else if (lsame(pivtng[0], 'F')) {
        ipvtng = 3;
        npvts = min_int(n, m);
    } else {
        ipvtng = -1;
    }

    if (lsame(grade[0], 'N')) {
        igrade = 0;
    } else if (lsame(grade[0], 'L')) {
        igrade = 1;
    } else if (lsame(grade[0], 'R')) {
        igrade = 2;
    } else if (lsame(grade[0], 'B')) {
        igrade = 3;
    } else if (lsame(grade[0], 'E')) {
        igrade = 4;
    } else if (lsame(grade[0], 'H')) {
        igrade = 5;
    } else if (lsame(grade[0], 'S')) {
        igrade = 6;
    } else {
        igrade = -1;
    }

    if (lsame(pack[0], 'N')) {
        ipack = 0;
    } else if (lsame(pack[0], 'U')) {
        ipack = 1;
    } else if (lsame(pack[0], 'L')) {
        ipack = 2;
    } else if (lsame(pack[0], 'C')) {
        ipack = 3;
    } else if (lsame(pack[0], 'R')) {
        ipack = 4;
    } else if (lsame(pack[0], 'B')) {
        ipack = 5;
    } else if (lsame(pack[0], 'Q')) {
        ipack = 6;
    } else if (lsame(pack[0], 'Z')) {
        ipack = 7;
    } else {
        ipack = -1;
    }

    mnmin = min_int(m, n);
    kll = min_int(kl, m - 1);
    kuu = min_int(ku, n - 1);

    dzero = 0;
    if (igrade == 4 && model == 0) {
        for (i = 0; i < m; i++) {
            if (dl[i] == CZERO) {
                dzero = 1;
            }
        }
    }

    badpvt = 0;
    if (ipvtng > 0) {
        for (j = 0; j < npvts; j++) {
            if (ipivot[j] < 0 || ipivot[j] >= npvts) {
                badpvt = 1;
            }
        }
    }

    if (m < 0) {
        *info = -1;
    } else if (m != n && (isym == 0 || isym == 2)) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (idist == -1) {
        *info = -3;
    } else if (isym == -1) {
        *info = -5;
    } else if (mode < -6 || mode > 6) {
        *info = -7;
    } else if ((mode != -6 && mode != 0 && mode != 6) && cond < ONE) {
        *info = -8;
    } else if ((mode != -6 && mode != 0 && mode != 6) && irsign == -1) {
        *info = -10;
    } else if (igrade == -1 || (igrade == 4 && m != n) ||
               ((igrade == 1 || igrade == 2 || igrade == 3 ||
                 igrade == 4 || igrade == 6) && isym == 0) ||
               ((igrade == 1 || igrade == 2 || igrade == 3 ||
                 igrade == 4 || igrade == 5) && isym == 2)) {
        *info = -11;
    } else if (igrade == 4 && dzero) {
        *info = -12;
    } else if ((igrade == 1 || igrade == 3 || igrade == 4 ||
                igrade == 5 || igrade == 6) &&
               (model < -6 || model > 6)) {
        *info = -13;
    } else if ((igrade == 1 || igrade == 3 || igrade == 4 ||
                igrade == 5 || igrade == 6) &&
               (model != -6 && model != 0 && model != 6) && condl < ONE) {
        *info = -14;
    } else if ((igrade == 2 || igrade == 3) && (moder < -6 || moder > 6)) {
        *info = -16;
    } else if ((igrade == 2 || igrade == 3) &&
               (moder != -6 && moder != 0 && moder != 6) && condr < ONE) {
        *info = -17;
    } else if (ipvtng == -1 || (ipvtng == 3 && m != n) ||
               ((ipvtng == 1 || ipvtng == 2) && (isym == 0 || isym == 2))) {
        *info = -18;
    } else if (ipvtng != 0 && badpvt) {
        *info = -19;
    } else if (kl < 0) {
        *info = -20;
    } else if (ku < 0 || ((isym == 0 || isym == 2) && kl != ku)) {
        *info = -21;
    } else if (sparse < ZERO || sparse > ONE) {
        *info = -22;
    } else if (ipack == -1 || ((ipack == 1 || ipack == 2 ||
               ipack == 5 || ipack == 6) && isym == 1) ||
               (ipack == 3 && isym == 1 && (kl != 0 || m != n)) ||
               (ipack == 4 && isym == 1 && (ku != 0 || m != n))) {
        *info = -24;
    } else if (((ipack == 0 || ipack == 1 || ipack == 2) &&
                lda < max_int(1, m)) ||
               ((ipack == 3 || ipack == 4) && lda < 1) ||
               ((ipack == 5 || ipack == 6) && lda < kuu + 1) ||
               (ipack == 7 && lda < kll + kuu + 1)) {
        *info = -26;
    }

    if (*info != 0) {
        xerbla("ZLATMR", -(*info));
        return;
    }

    fulbnd = 0;
    if (kuu == n - 1 && kll == m - 1) {
        fulbnd = 1;
    }

    zlatm1(mode, cond, irsign, idist, d, mnmin, info, state);
    if (*info != 0) {
        *info = 1;
        return;
    }

    if (mode != 0 && mode != -6 && mode != 6) {
        temp = cabs(d[0]);
        for (i = 1; i < mnmin; i++) {
            temp = max_dbl(temp, cabs(d[i]));
        }
        if (temp == ZERO && dmax != CZERO) {
            *info = 2;
            return;
        }
        if (temp != ZERO) {
            calpha = dmax / temp;
        } else {
            calpha = CONE;
        }
        for (i = 0; i < mnmin; i++) {
            d[i] = calpha * d[i];
        }
    }

    if (isym == 0) {
        for (i = 0; i < mnmin; i++) {
            d[i] = CMPLX(creal(d[i]), 0.0);
        }
    }

    if (igrade == 1 || igrade == 3 || igrade == 4 ||
        igrade == 5 || igrade == 6) {
        zlatm1(model, condl, 0, idist, dl, m, info, state);
        if (*info != 0) {
            *info = 3;
            return;
        }
    }

    if (igrade == 2 || igrade == 3) {
        zlatm1(moder, condr, 0, idist, dr, n, info, state);
        if (*info != 0) {
            *info = 4;
            return;
        }
    }

    if (ipvtng > 0) {
        for (i = 0; i < npvts; i++) {
            iwork[i] = i;
        }
        if (fulbnd) {
            for (i = 0; i < npvts; i++) {
                k = ipivot[i];
                j = iwork[i];
                iwork[i] = iwork[k];
                iwork[k] = j;
            }
        } else {
            for (i = npvts - 1; i >= 0; i--) {
                k = ipivot[i];
                j = iwork[i];
                iwork[i] = iwork[k];
                iwork[k] = j;
            }
        }
    }

    if (fulbnd) {
        if (ipack == 0) {
            if (isym == 0) {
                for (j = 0; j < n; j++) {
                    for (i = 0; i <= j; i++) {
                        ctemp = zlatm3(m, n, i, j, &isub, &jsub, kl, ku,
                                       idist, d, igrade, dl, dr, ipvtng,
                                       iwork, sparse, state);
                        A[isub + jsub * lda] = ctemp;
                        A[jsub + isub * lda] = conj(ctemp);
                    }
                }
            } else if (isym == 1) {
                for (j = 0; j < n; j++) {
                    for (i = 0; i < m; i++) {
                        ctemp = zlatm3(m, n, i, j, &isub, &jsub, kl, ku,
                                       idist, d, igrade, dl, dr, ipvtng,
                                       iwork, sparse, state);
                        A[isub + jsub * lda] = ctemp;
                    }
                }
            } else if (isym == 2) {
                for (j = 0; j < n; j++) {
                    for (i = 0; i <= j; i++) {
                        ctemp = zlatm3(m, n, i, j, &isub, &jsub, kl, ku,
                                       idist, d, igrade, dl, dr, ipvtng,
                                       iwork, sparse, state);
                        A[isub + jsub * lda] = ctemp;
                        A[jsub + isub * lda] = ctemp;
                    }
                }
            }
        } else if (ipack == 1) {
            for (j = 0; j < n; j++) {
                for (i = 0; i <= j; i++) {
                    ctemp = zlatm3(m, n, i, j, &isub, &jsub, kl, ku,
                                   idist, d, igrade, dl, dr, ipvtng,
                                   iwork, sparse, state);
                    mnsub = min_int(isub, jsub);
                    mxsub = max_int(isub, jsub);
                    if (mxsub == isub && isym == 0) {
                        A[mnsub + mxsub * lda] = conj(ctemp);
                    } else {
                        A[mnsub + mxsub * lda] = ctemp;
                    }
                    if (mnsub != mxsub) {
                        A[mxsub + mnsub * lda] = CZERO;
                    }
                }
            }
        } else if (ipack == 2) {
            for (j = 0; j < n; j++) {
                for (i = 0; i <= j; i++) {
                    ctemp = zlatm3(m, n, i, j, &isub, &jsub, kl, ku,
                                   idist, d, igrade, dl, dr, ipvtng,
                                   iwork, sparse, state);
                    mnsub = min_int(isub, jsub);
                    mxsub = max_int(isub, jsub);
                    if (mxsub == jsub && isym == 0) {
                        A[mxsub + mnsub * lda] = conj(ctemp);
                    } else {
                        A[mxsub + mnsub * lda] = ctemp;
                    }
                    if (mnsub != mxsub) {
                        A[mnsub + mxsub * lda] = CZERO;
                    }
                }
            }
        } else if (ipack == 3) {
            for (j = 0; j < n; j++) {
                for (i = 0; i <= j; i++) {
                    ctemp = zlatm3(m, n, i, j, &isub, &jsub, kl, ku,
                                   idist, d, igrade, dl, dr, ipvtng,
                                   iwork, sparse, state);
                    mnsub = min_int(isub, jsub);
                    mxsub = max_int(isub, jsub);
                    k = mxsub * (mxsub + 1) / 2 + mnsub;
                    jjsub = k / lda;
                    iisub = k - lda * jjsub;
                    if (mxsub == isub && isym == 0) {
                        A[iisub + jjsub * lda] = conj(ctemp);
                    } else {
                        A[iisub + jjsub * lda] = ctemp;
                    }
                }
            }
        } else if (ipack == 4) {
            for (j = 0; j < n; j++) {
                for (i = 0; i <= j; i++) {
                    ctemp = zlatm3(m, n, i, j, &isub, &jsub, kl, ku,
                                   idist, d, igrade, dl, dr, ipvtng,
                                   iwork, sparse, state);
                    mnsub = min_int(isub, jsub);
                    mxsub = max_int(isub, jsub);
                    if (mnsub == 0) {
                        k = mxsub;
                    } else {
                        k = n * (n + 1) / 2 - (n - mnsub) * (n - mnsub + 1) / 2 +
                            mxsub - mnsub;
                    }
                    jjsub = k / lda;
                    iisub = k - lda * jjsub;
                    if (mxsub == jsub && isym == 0) {
                        A[iisub + jjsub * lda] = conj(ctemp);
                    } else {
                        A[iisub + jjsub * lda] = ctemp;
                    }
                }
            }
        } else if (ipack == 5) {
            for (j = 0; j < n; j++) {
                for (i = j - kuu; i <= j; i++) {
                    if (i < 0) {
                        A[(j - i) + (i + n) * lda] = CZERO;
                    } else {
                        ctemp = zlatm3(m, n, i, j, &isub, &jsub, kl, ku,
                                       idist, d, igrade, dl, dr, ipvtng,
                                       iwork, sparse, state);
                        mnsub = min_int(isub, jsub);
                        mxsub = max_int(isub, jsub);
                        if (mxsub == jsub && isym == 0) {
                            A[(mxsub - mnsub) + mnsub * lda] = conj(ctemp);
                        } else {
                            A[(mxsub - mnsub) + mnsub * lda] = ctemp;
                        }
                    }
                }
            }
        } else if (ipack == 6) {
            for (j = 0; j < n; j++) {
                for (i = j - kuu; i <= j; i++) {
                    ctemp = zlatm3(m, n, i, j, &isub, &jsub, kl, ku,
                                   idist, d, igrade, dl, dr, ipvtng,
                                   iwork, sparse, state);
                    mnsub = min_int(isub, jsub);
                    mxsub = max_int(isub, jsub);
                    if (mxsub == isub && isym == 0) {
                        A[(mnsub - mxsub + kuu) + mxsub * lda] = conj(ctemp);
                    } else {
                        A[(mnsub - mxsub + kuu) + mxsub * lda] = ctemp;
                    }
                }
            }
        } else if (ipack == 7) {
            if (isym != 1) {
                for (j = 0; j < n; j++) {
                    for (i = j - kuu; i <= j; i++) {
                        ctemp = zlatm3(m, n, i, j, &isub, &jsub, kl, ku,
                                       idist, d, igrade, dl, dr, ipvtng,
                                       iwork, sparse, state);
                        mnsub = min_int(isub, jsub);
                        mxsub = max_int(isub, jsub);
                        if (i < 0) {
                            A[(j - i + kuu) + (i + n) * lda] = CZERO;
                        }
                        if (mxsub == isub && isym == 0) {
                            A[(mnsub - mxsub + kuu) + mxsub * lda] = conj(ctemp);
                        } else {
                            A[(mnsub - mxsub + kuu) + mxsub * lda] = ctemp;
                        }
                        if (i >= 0 && mnsub != mxsub) {
                            if (mnsub == isub && isym == 0) {
                                A[(mxsub - mnsub + kuu) + mnsub * lda] = conj(ctemp);
                            } else {
                                A[(mxsub - mnsub + kuu) + mnsub * lda] = ctemp;
                            }
                        }
                    }
                }
            } else if (isym == 1) {
                for (j = 0; j < n; j++) {
                    for (i = j - kuu; i <= j + kll; i++) {
                        ctemp = zlatm3(m, n, i, j, &isub, &jsub, kl, ku,
                                       idist, d, igrade, dl, dr, ipvtng,
                                       iwork, sparse, state);
                        A[(isub - jsub + kuu) + jsub * lda] = ctemp;
                    }
                }
            }
        }
    } else {
        if (ipack == 0) {
            if (isym == 0) {
                for (j = 0; j < n; j++) {
                    for (i = 0; i <= j; i++) {
                        A[i + j * lda] = zlatm2(m, n, i, j, kl, ku, idist,
                                                 d, igrade, dl, dr, ipvtng,
                                                 iwork, sparse, state);
                        A[j + i * lda] = conj(A[i + j * lda]);
                    }
                }
            } else if (isym == 1) {
                for (j = 0; j < n; j++) {
                    for (i = 0; i < m; i++) {
                        A[i + j * lda] = zlatm2(m, n, i, j, kl, ku, idist,
                                                 d, igrade, dl, dr, ipvtng,
                                                 iwork, sparse, state);
                    }
                }
            } else if (isym == 2) {
                for (j = 0; j < n; j++) {
                    for (i = 0; i <= j; i++) {
                        A[i + j * lda] = zlatm2(m, n, i, j, kl, ku, idist,
                                                 d, igrade, dl, dr, ipvtng,
                                                 iwork, sparse, state);
                        A[j + i * lda] = A[i + j * lda];
                    }
                }
            }
        } else if (ipack == 1) {
            for (j = 0; j < n; j++) {
                for (i = 0; i <= j; i++) {
                    A[i + j * lda] = zlatm2(m, n, i, j, kl, ku, idist,
                                             d, igrade, dl, dr, ipvtng,
                                             iwork, sparse, state);
                    if (i != j) {
                        A[j + i * lda] = CZERO;
                    }
                }
            }
        } else if (ipack == 2) {
            for (j = 0; j < n; j++) {
                for (i = 0; i <= j; i++) {
                    if (isym == 0) {
                        A[j + i * lda] = conj(zlatm2(m, n, i, j, kl, ku,
                                                       idist, d, igrade, dl, dr,
                                                       ipvtng, iwork, sparse, state));
                    } else {
                        A[j + i * lda] = zlatm2(m, n, i, j, kl, ku, idist,
                                                 d, igrade, dl, dr, ipvtng,
                                                 iwork, sparse, state);
                    }
                    if (i != j) {
                        A[i + j * lda] = CZERO;
                    }
                }
            }
        } else if (ipack == 3) {
            isub = -1;
            jsub = 0;
            for (j = 0; j < n; j++) {
                for (i = 0; i <= j; i++) {
                    isub = isub + 1;
                    if (isub >= lda) {
                        isub = 0;
                        jsub = jsub + 1;
                    }
                    A[isub + jsub * lda] = zlatm2(m, n, i, j, kl, ku,
                                                   idist, d, igrade, dl, dr,
                                                   ipvtng, iwork, sparse, state);
                }
            }
        } else if (ipack == 4) {
            if (isym == 0 || isym == 2) {
                for (j = 0; j < n; j++) {
                    for (i = 0; i <= j; i++) {
                        if (i == 0) {
                            k = j;
                        } else {
                            k = n * (n + 1) / 2 - (n - i) * (n - i + 1) / 2 +
                                j - i;
                        }
                        jsub = k / lda;
                        isub = k - lda * jsub;
                        A[isub + jsub * lda] = zlatm2(m, n, i, j, kl, ku,
                                                       idist, d, igrade, dl, dr,
                                                       ipvtng, iwork, sparse, state);
                        if (isym == 0) {
                            A[isub + jsub * lda] = conj(A[isub + jsub * lda]);
                        }
                    }
                }
            } else {
                isub = -1;
                jsub = 0;
                for (j = 0; j < n; j++) {
                    for (i = j; i < m; i++) {
                        isub = isub + 1;
                        if (isub >= lda) {
                            isub = 0;
                            jsub = jsub + 1;
                        }
                        A[isub + jsub * lda] = zlatm2(m, n, i, j, kl, ku,
                                                       idist, d, igrade, dl, dr,
                                                       ipvtng, iwork, sparse, state);
                    }
                }
            }
        } else if (ipack == 5) {
            for (j = 0; j < n; j++) {
                for (i = j - kuu; i <= j; i++) {
                    if (i < 0) {
                        A[(j - i) + (i + n) * lda] = CZERO;
                    } else {
                        if (isym == 0) {
                            A[(j - i) + i * lda] = conj(zlatm2(m, n, i, j, kl,
                                                                 ku, idist, d, igrade, dl,
                                                                 dr, ipvtng, iwork, sparse, state));
                        } else {
                            A[(j - i) + i * lda] = zlatm2(m, n, i, j, kl, ku,
                                                           idist, d, igrade, dl, dr,
                                                           ipvtng, iwork, sparse, state);
                        }
                    }
                }
            }
        } else if (ipack == 6) {
            for (j = 0; j < n; j++) {
                for (i = j - kuu; i <= j; i++) {
                    A[(i - j + kuu) + j * lda] = zlatm2(m, n, i, j, kl, ku,
                                                         idist, d, igrade, dl, dr,
                                                         ipvtng, iwork, sparse, state);
                }
            }
        } else if (ipack == 7) {
            if (isym != 1) {
                for (j = 0; j < n; j++) {
                    for (i = j - kuu; i <= j; i++) {
                        A[(i - j + kuu) + j * lda] = zlatm2(m, n, i, j, kl, ku,
                                                             idist, d, igrade, dl,
                                                             dr, ipvtng, iwork, sparse, state);
                        if (i < 0) {
                            A[(j - i + kuu) + (i + n) * lda] = CZERO;
                        }
                        if (i >= 0 && i != j) {
                            if (isym == 0) {
                                A[(j - i + kuu) + i * lda] = conj(A[(i - j + kuu) + j * lda]);
                            } else {
                                A[(j - i + kuu) + i * lda] = A[(i - j + kuu) + j * lda];
                            }
                        }
                    }
                }
            } else if (isym == 1) {
                for (j = 0; j < n; j++) {
                    for (i = j - kuu; i <= j + kll; i++) {
                        A[(i - j + kuu) + j * lda] = zlatm2(m, n, i, j, kl, ku,
                                                             idist, d, igrade, dl,
                                                             dr, ipvtng, iwork, sparse, state);
                    }
                }
            }
        }
    }

    if (ipack == 0) {
        onorm = zlange("M", m, n, A, lda, tempa);
    } else if (ipack == 1) {
        onorm = zlansy("M", "U", n, A, lda, tempa);
    } else if (ipack == 2) {
        onorm = zlansy("M", "L", n, A, lda, tempa);
    } else if (ipack == 3) {
        onorm = zlansp("M", "U", n, A, tempa);
    } else if (ipack == 4) {
        onorm = zlansp("M", "L", n, A, tempa);
    } else if (ipack == 5) {
        onorm = zlansb("M", "L", n, kll, A, lda, tempa);
    } else if (ipack == 6) {
        onorm = zlansb("M", "U", n, kuu, A, lda, tempa);
    } else if (ipack == 7) {
        onorm = zlangb("M", n, kll, kuu, A, lda, tempa);
    }

    if (anorm >= ZERO) {
        if (anorm > ZERO && onorm == ZERO) {
            *info = 5;
            return;
        } else if ((anorm > ONE && onorm < ONE) ||
                   (anorm < ONE && onorm > ONE)) {
            if (ipack <= 2) {
                for (j = 0; j < n; j++) {
                    cblas_zdscal(m, ONE / onorm, &A[j * lda], 1);
                    cblas_zdscal(m, anorm, &A[j * lda], 1);
                }
            } else if (ipack == 3 || ipack == 4) {
                cblas_zdscal(n * (n + 1) / 2, ONE / onorm, A, 1);
                cblas_zdscal(n * (n + 1) / 2, anorm, A, 1);
            } else if (ipack >= 5) {
                for (j = 0; j < n; j++) {
                    cblas_zdscal(kll + kuu + 1, ONE / onorm, &A[j * lda], 1);
                    cblas_zdscal(kll + kuu + 1, anorm, &A[j * lda], 1);
                }
            }
        } else {
            if (ipack <= 2) {
                for (j = 0; j < n; j++) {
                    cblas_zdscal(m, anorm / onorm, &A[j * lda], 1);
                }
            } else if (ipack == 3 || ipack == 4) {
                cblas_zdscal(n * (n + 1) / 2, anorm / onorm, A, 1);
            } else if (ipack >= 5) {
                for (j = 0; j < n; j++) {
                    cblas_zdscal(kll + kuu + 1, anorm / onorm, &A[j * lda], 1);
                }
            }
        }
    }
}
