/**
 * @file dlattb.c
 * @brief DLATTB generates a triangular test matrix in 2-dimensional banded
 *        storage.
 */

#include <math.h>
#include <stdlib.h>
#include <cblas.h>
#include "verify.h"
#include "test_rng.h"

extern double dlamch(const char* cmach);

/**
 * DLATTB generates a triangular test matrix in 2-dimensional banded storage.
 * IMAT and UPLO uniquely specify the properties of the test matrix,
 * which is returned in the array AB.
 *
 * @param[in]     imat    An integer key describing which matrix to generate
 *                        for this path.
 * @param[in]     uplo    'U': Upper triangular, 'L': Lower triangular.
 * @param[in]     trans   'N': No transpose, 'T'/'C': Transpose.
 *                        Specifies whether the matrix or its transpose will
 *                        be used.
 * @param[out]    diag    'N': Non-unit triangular, 'U': Unit triangular.
 * @param[in]     n       The order of the matrix to be generated.
 * @param[in]     kd      The number of superdiagonals or subdiagonals of the
 *                        banded triangular matrix A. kd >= 0.
 * @param[out]    AB      The upper or lower triangular banded matrix A, stored
 *                        in the first kd+1 rows of AB. Double precision array,
 *                        dimension (ldab, n).
 *                        If UPLO = 'U', AB(kd+i-j,j) = A(i,j) for max(0,j-kd)<=i<=j.
 *                        If UPLO = 'L', AB(i-j,j) = A(i,j) for j<=i<=min(n-1,j+kd).
 * @param[in]     ldab    The leading dimension of the array AB. ldab >= kd+1.
 * @param[out]    B       Right hand side vector. Double precision array,
 *                        dimension (n).
 * @param[out]    work    Workspace. Double precision array, dimension (2*n).
 * @param[out]    info    = 0: successful exit
 *                        < 0: if info = -k, the k-th argument had an illegal value
 * @param[in,out] state   RNG state array of 4 uint64_t values.
 */
void dlattb(const int imat, const char* uplo, const char* trans, char* diag,
            const int n, const int kd, double* AB, const int ldab,
            double* B, double* work, int* info, uint64_t state[static 4])
{
    const double ZERO = 0.0;
    const double ONE = 1.0;
    const double TWO = 2.0;

    int upper;
    char type, dist, packit;
    int kl, ku, mode;
    double anorm, cndnum;
    int i, j, jcount, iy, lenj, ioff;
    double unfl, ulp, smlnum, bignum;
    double bnorm, bscal, tscal, texp, tleft, tnorm;
    double plus1, plus2, star1, sfac, rexp;

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
        dlatb4("DTB", imat, n, n, &type, &kl, &ku, &anorm, &mode, &cndnum, &dist);
        ku = kd;
        ioff = (kd > n - 1) ? kd - n + 1 : 0;
        kl = 0;
        packit = 'Q';
    } else {
        dlatb4("DTB", -imat, n, n, &type, &kl, &ku, &anorm, &mode, &cndnum, &dist);
        kl = kd;
        ioff = 0;
        ku = 0;
        packit = 'B';
    }

    if (imat <= 5) {
        char symm[2] = {type, '\0'};
        char dstr[2] = {dist, '\0'};
        char pack[2] = {packit, '\0'};
        dlatms(n, n, dstr, symm, B, mode, cndnum, anorm,
               kl, ku, pack, &AB[ioff], ldab, work, info, state);

    } else if (imat == 6) {
        if (upper) {
            for (j = 0; j < n; j++) {
                for (i = (kd + 1 - j > 0 ? kd + 1 - j : 0); i < kd; i++) {
                    AB[i + j * ldab] = ZERO;
                }
                AB[kd + j * ldab] = (double)(j + 1);
            }
        } else {
            for (j = 0; j < n; j++) {
                AB[0 + j * ldab] = (double)(j + 1);
                for (i = 1; i < (kd + 1 < n - j ? kd + 1 : n - j); i++) {
                    AB[i + j * ldab] = ZERO;
                }
            }
        }

    } else if (imat <= 9) {
        tnorm = sqrt(cndnum);

        if (upper) {
            for (j = 0; j < n; j++) {
                for (i = (kd + 1 - j > 0 ? kd + 1 - j : 0); i < kd; i++) {
                    AB[i + j * ldab] = ZERO;
                }
                AB[kd + j * ldab] = (double)(j + 1);
            }
        } else {
            for (j = 0; j < n; j++) {
                for (i = 1; i < (kd + 1 < n - j ? kd + 1 : n - j); i++) {
                    AB[i + j * ldab] = ZERO;
                }
                AB[0 + j * ldab] = (double)(j + 1);
            }
        }

        if (kd == 1) {
            if (upper) {
                AB[0 + 1 * ldab] = (rng_uniform(state) < 0.5 ? -tnorm : tnorm);
                lenj = (n - 3) / 2;
                dlarnv_rng(2, lenj, work, state);
                for (j = 0; j < lenj; j++) {
                    AB[0 + 2 * (j + 2) * ldab] = tnorm * work[j];
                }
            } else {
                AB[1 + 0 * ldab] = (rng_uniform(state) < 0.5 ? -tnorm : tnorm);
                lenj = (n - 3) / 2;
                dlarnv_rng(2, lenj, work, state);
                for (j = 0; j < lenj; j++) {
                    AB[1 + (2 * j + 2) * ldab] = tnorm * work[j];
                }
            }
        } else if (kd > 1) {
            star1 = (rng_uniform(state) < 0.5 ? -tnorm : tnorm);
            sfac = sqrt(tnorm);
            plus1 = (rng_uniform(state) < 0.5 ? -sfac : sfac);
            for (j = 0; j < n; j += 2) {
                plus2 = star1 / plus1;
                work[j] = plus1;
                work[n + j] = star1;
                if (j + 1 < n) {
                    work[j + 1] = plus2;
                    work[n + j + 1] = ZERO;
                    plus1 = star1 / plus2;

                    rexp = rng_uniform(state) * 2.0 - 1.0;
                    if (rexp < ZERO) {
                        star1 = -pow(sfac, ONE - rexp);
                    } else {
                        star1 = pow(sfac, ONE + rexp);
                    }
                }
            }

            if (upper) {
                cblas_dcopy(n - 1, work, 1, &AB[(kd - 1) + 1 * ldab], ldab);
                cblas_dcopy(n - 2, &work[n], 1, &AB[(kd - 2) + 2 * ldab], ldab);
            } else {
                cblas_dcopy(n - 1, work, 1, &AB[1 + 0 * ldab], ldab);
                cblas_dcopy(n - 2, &work[n], 1, &AB[2 + 0 * ldab], ldab);
            }
        }

    } else if (imat == 10) {
        if (upper) {
            for (j = 0; j < n; j++) {
                lenj = (j + 1 < kd + 1 ? j + 1 : kd + 1);
                dlarnv_rng(2, lenj, &AB[(kd + 1 - lenj) + j * ldab], state);
                AB[kd + j * ldab] = (AB[kd + j * ldab] >= ZERO ? TWO : -TWO);
            }
        } else {
            for (j = 0; j < n; j++) {
                lenj = (n - j < kd + 1 ? n - j : kd + 1);
                if (lenj > 0)
                    dlarnv_rng(2, lenj, &AB[0 + j * ldab], state);
                AB[0 + j * ldab] = (AB[0 + j * ldab] >= ZERO ? TWO : -TWO);
            }
        }

        dlarnv_rng(2, n, B, state);
        iy = cblas_idamax(n, B, 1);
        bnorm = fabs(B[iy]);
        bscal = bignum / (ONE > bnorm ? ONE : bnorm);
        cblas_dscal(n, bscal, B, 1);

    } else if (imat == 11) {
        dlarnv_rng(2, n, B, state);
        tscal = ONE / (double)(kd + 1);
        if (upper) {
            for (j = 0; j < n; j++) {
                lenj = (j + 1 < kd + 1 ? j + 1 : kd + 1);
                dlarnv_rng(2, lenj, &AB[(kd + 1 - lenj) + j * ldab], state);
                cblas_dscal(lenj - 1, tscal, &AB[(kd + 1 - lenj) + j * ldab], 1);
                AB[kd + j * ldab] = (AB[kd + j * ldab] >= ZERO ? ONE : -ONE);
            }
            AB[kd + (n - 1) * ldab] = smlnum * AB[kd + (n - 1) * ldab];
        } else {
            for (j = 0; j < n; j++) {
                lenj = (n - j < kd + 1 ? n - j : kd + 1);
                dlarnv_rng(2, lenj, &AB[0 + j * ldab], state);
                if (lenj > 1)
                    cblas_dscal(lenj - 1, tscal, &AB[1 + j * ldab], 1);
                AB[0 + j * ldab] = (AB[0 + j * ldab] >= ZERO ? ONE : -ONE);
            }
            AB[0 + 0 * ldab] = smlnum * AB[0 + 0 * ldab];
        }

    } else if (imat == 12) {
        dlarnv_rng(2, n, B, state);
        if (upper) {
            for (j = 0; j < n; j++) {
                lenj = (j + 1 < kd + 1 ? j + 1 : kd + 1);
                dlarnv_rng(2, lenj, &AB[(kd + 1 - lenj) + j * ldab], state);
                AB[kd + j * ldab] = (AB[kd + j * ldab] >= ZERO ? ONE : -ONE);
            }
            AB[kd + (n - 1) * ldab] = smlnum * AB[kd + (n - 1) * ldab];
        } else {
            for (j = 0; j < n; j++) {
                lenj = (n - j < kd + 1 ? n - j : kd + 1);
                dlarnv_rng(2, lenj, &AB[0 + j * ldab], state);
                AB[0 + j * ldab] = (AB[0 + j * ldab] >= ZERO ? ONE : -ONE);
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
                    AB[kd + j * ldab] = smlnum;
                } else {
                    AB[kd + j * ldab] = ONE;
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
                    AB[0 + j * ldab] = smlnum;
                } else {
                    AB[0 + j * ldab] = ONE;
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
                B[i - 1] = smlnum;
            }
        } else {
            B[n - 1] = ZERO;
            for (i = 0; i < n - 1; i += 2) {
                B[i] = ZERO;
                B[i + 1] = smlnum;
            }
        }

    } else if (imat == 14) {
        texp = ONE / (double)(kd + 1);
        tscal = pow(smlnum, texp);
        dlarnv_rng(2, n, B, state);
        if (upper) {
            for (j = 0; j < n; j++) {
                for (i = (kd + 1 - j > 0 ? kd + 1 - j : 0); i < kd; i++) {
                    AB[i + j * ldab] = ZERO;
                }
                if (j > 0 && kd > 0)
                    AB[(kd - 1) + j * ldab] = -ONE;
                AB[kd + j * ldab] = tscal;
            }
            B[n - 1] = ONE;
        } else {
            for (j = 0; j < n; j++) {
                for (i = 2; i < (n - j < kd + 1 ? n - j : kd + 1); i++) {
                    AB[i + j * ldab] = ZERO;
                }
                if (j < n - 1 && kd > 0)
                    AB[1 + j * ldab] = -ONE;
                AB[0 + j * ldab] = tscal;
            }
            B[0] = ONE;
        }

    } else if (imat == 15) {
        iy = n / 2;
        if (upper) {
            for (j = 0; j < n; j++) {
                lenj = (j + 1 < kd + 1 ? j + 1 : kd + 1);
                dlarnv_rng(2, lenj, &AB[(kd + 1 - lenj) + j * ldab], state);
                if (j != iy) {
                    AB[kd + j * ldab] = (AB[kd + j * ldab] >= ZERO ? TWO : -TWO);
                } else {
                    AB[kd + j * ldab] = ZERO;
                }
            }
        } else {
            for (j = 0; j < n; j++) {
                lenj = (n - j < kd + 1 ? n - j : kd + 1);
                dlarnv_rng(2, lenj, &AB[0 + j * ldab], state);
                if (j != iy) {
                    AB[0 + j * ldab] = (AB[0 + j * ldab] >= ZERO ? TWO : -TWO);
                } else {
                    AB[0 + j * ldab] = ZERO;
                }
            }
        }
        dlarnv_rng(2, n, B, state);
        cblas_dscal(n, TWO, B, 1);

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
                        AB[(j - i) + i * ldab] = -tscal / (double)(kd + 2);
                        AB[kd + i * ldab] = ONE;
                        B[i] = texp * (ONE - ulp);
                        if (i > (j - kd + 1 > 0 ? j - kd + 1 : 0)) {
                            AB[(j - i + 1) + (i - 1) * ldab] = -(tscal / (double)(kd + 2)) / (double)(kd + 3);
                            AB[kd + (i - 1) * ldab] = ONE;
                            B[i - 1] = texp * (double)((kd + 1) * (kd + 1) + kd);
                        }
                        texp = texp * TWO;
                    }
                    B[(j - kd + 1 > 0 ? j - kd + 1 : 0)] = ((double)(kd + 2) / (double)(kd + 3)) * tscal;
                }
            } else {
                for (j = 0; j < n; j += kd) {
                    texp = ONE;
                    lenj = (kd + 1 < n - j ? kd + 1 : n - j);
                    for (i = j; i < (n - 1 < j + kd - 1 ? n - 1 : j + kd - 1); i += 2) {
                        AB[(lenj - 1 - (i - j)) + j * ldab] = -tscal / (double)(kd + 2);
                        AB[0 + j * ldab] = ONE;
                        B[j] = texp * (ONE - ulp);
                        if (i < (n - 1 < j + kd - 1 ? n - 1 : j + kd - 1)) {
                            AB[(lenj - 1 - (i - j + 1)) + (i + 1) * ldab] = -(tscal / (double)(kd + 2)) / (double)(kd + 3);
                            AB[0 + (i + 1) * ldab] = ONE;
                            B[i + 1] = texp * (double)((kd + 1) * (kd + 1) + kd);
                        }
                        texp = texp * TWO;
                    }
                    B[(n - 1 < j + kd - 1 ? n - 1 : j + kd - 1)] = ((double)(kd + 2) / (double)(kd + 3)) * tscal;
                }
            }
        } else {
            for (j = 0; j < n; j++) {
                AB[0 + j * ldab] = ONE;
                B[j] = (double)(j + 1);
            }
        }

    } else if (imat == 17) {
        if (upper) {
            for (j = 0; j < n; j++) {
                lenj = (j < kd ? j : kd);
                dlarnv_rng(2, lenj, &AB[(kd - lenj) + j * ldab], state);
                AB[kd + j * ldab] = (double)(j + 1);
            }
        } else {
            for (j = 0; j < n; j++) {
                lenj = (n - j - 1 < kd ? n - j - 1 : kd);
                if (lenj > 0)
                    dlarnv_rng(2, lenj, &AB[1 + j * ldab], state);
                AB[0 + j * ldab] = (double)(j + 1);
            }
        }

        dlarnv_rng(2, n, B, state);
        iy = cblas_idamax(n, B, 1);
        bnorm = fabs(B[iy]);
        bscal = bignum / (ONE > bnorm ? ONE : bnorm);
        cblas_dscal(n, bscal, B, 1);

    } else if (imat == 18) {
        tleft = bignum / (ONE > (double)kd ? ONE : (double)kd);
        tscal = bignum * ((double)kd / (double)(kd + 1));
        if (upper) {
            for (j = 0; j < n; j++) {
                lenj = (j + 1 < kd + 1 ? j + 1 : kd + 1);
                dlarnv_rng(2, lenj, &AB[(kd + 1 - lenj) + j * ldab], state);
                for (i = kd + 1 - lenj; i <= kd; i++) {
                    AB[i + j * ldab] = (AB[i + j * ldab] >= ZERO ? tleft : -tleft) + tscal * AB[i + j * ldab];
                }
            }
        } else {
            for (j = 0; j < n; j++) {
                lenj = (n - j < kd + 1 ? n - j : kd + 1);
                dlarnv_rng(2, lenj, &AB[0 + j * ldab], state);
                for (i = 0; i < lenj; i++) {
                    AB[i + j * ldab] = (AB[i + j * ldab] >= ZERO ? tleft : -tleft) + tscal * AB[i + j * ldab];
                }
            }
        }
        dlarnv_rng(2, n, B, state);
        cblas_dscal(n, TWO, B, 1);
    }

    if (!(trans[0] == 'N' || trans[0] == 'n')) {
        if (upper) {
            for (j = 0; j < n / 2; j++) {
                lenj = (n - 2 * j - 1 < kd + 1 ? n - 2 * j - 1 : kd + 1);
                cblas_dswap(lenj, &AB[kd + j * ldab], ldab - 1,
                            &AB[(kd + 1 - lenj) + (n - 1 - j) * ldab], -1);
            }
        } else {
            for (j = 0; j < n / 2; j++) {
                lenj = (n - 2 * j - 1 < kd + 1 ? n - 2 * j - 1 : kd + 1);
                cblas_dswap(lenj, &AB[0 + j * ldab], 1,
                            &AB[(lenj - 1) + (n - j - lenj) * ldab], -ldab + 1);
            }
        }
    }
}
