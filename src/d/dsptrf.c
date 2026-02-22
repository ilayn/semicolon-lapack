/**
 * @file dsptrf.c
 * @brief DSPTRF computes the factorization of a real symmetric matrix stored
 *        in packed format using the Bunch-Kaufman diagonal pivoting method.
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "semicolon_lapack_double.h"

/**
 * DSPTRF computes the factorization of a real symmetric matrix A stored
 * in packed format using the Bunch-Kaufman diagonal pivoting method:
 *
 *    A = U*D*U**T  or  A = L*D*L**T
 *
 * where U (or L) is a product of permutation and unit upper (lower)
 * triangular matrices, and D is symmetric and block diagonal with
 * 1-by-1 and 2-by-2 diagonal blocks.
 *
 * @param[in]     uplo   Specifies whether the upper or lower triangular part
 *                       of the symmetric matrix A is stored:
 *                       - = 'U': Upper triangle of A is stored
 *                       - = 'L': Lower triangle of A is stored
 * @param[in]     n      The order of the matrix A. n >= 0.
 * @param[in,out] AP     On entry, the upper or lower triangle of the symmetric
 *                       matrix A, packed columnwise in a linear array of
 *                       dimension n*(n+1)/2.
 *                       On exit, the block diagonal matrix D and the multipliers
 *                       used to obtain the factor U or L, stored as a packed
 *                       triangular matrix.
 * @param[out]    ipiv   Details of the interchanges and the block structure of D.
 *                       Array of dimension n, 0-based indexing.
 *                       - If ipiv[k] >= 0, then rows and columns k and ipiv[k]
 *                         were interchanged and D(k,k) is a 1-by-1 diagonal block.
 *                       - If uplo = 'U' and ipiv[k] = ipiv[k-1] < 0, then rows and
 *                         columns k-1 and -ipiv[k]-1 were interchanged and
 *                         D(k-1:k,k-1:k) is a 2-by-2 diagonal block.
 *                       - If uplo = 'L' and ipiv[k] = ipiv[k+1] < 0, then rows and
 *                         columns k+1 and -ipiv[k]-1 were interchanged and
 *                         D(k:k+1,k:k+1) is a 2-by-2 diagonal block.
 * @param[out]    info
 *                           Exit status:
 *                           - = 0: successful exit
 *                           - < 0: if info = -i, the i-th argument had an illegal value
 *                           - > 0: if info = i, D(i,i) is exactly zero. The factorization
 *                           has been completed, but the block diagonal matrix D is
 *                           exactly singular, and division by zero will occur if
 *                           it is used to solve a system of equations.
 */
void dsptrf(
    const char* uplo,
    const INT n,
    f64* restrict AP,
    INT* restrict ipiv,
    INT* info)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const f64 EIGHT = 8.0;
    const f64 SEVTEN = 17.0;

    INT upper;
    INT i, imax = 0, j, jmax, k, kc, kk, knc, kp, kpc, kstep, kx, npp;
    f64 absakk, alpha, colmax, d11, d12, d21, d22, r1, rowmax, t, wk, wkm1, wkp1;

    *info = 0;
    upper = (uplo[0] == 'U' || uplo[0] == 'u');
    if (!upper && !(uplo[0] == 'L' || uplo[0] == 'l')) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    }
    if (*info != 0) {
        xerbla("DSPTRF", -(*info));
        return;
    }

    alpha = (ONE + sqrt(SEVTEN)) / EIGHT;

    if (upper) {
        /*
         * Factorize A as U*D*U**T using the upper triangle of A
         *
         * K is the main loop index, decreasing from N to 1 in steps of
         * 1 or 2
         */
        k = n - 1;
        kc = (n - 1) * n / 2;

        while (k >= 0) {
            knc = kc;
            kstep = 1;

            absakk = fabs(AP[kc + k]);

            if (k > 0) {
                imax = cblas_idamax(k, &AP[kc], 1);
                colmax = fabs(AP[kc + imax]);
            } else {
                colmax = ZERO;
            }

            if (fmax(absakk, colmax) == ZERO) {
                if (*info == 0)
                    *info = k + 1;
                kp = k;
            } else {
                if (absakk >= alpha * colmax) {
                    kp = k;
                } else {
                    rowmax = ZERO;
                    kx = imax + (imax + 1) * (imax + 2) / 2;
                    for (j = imax + 1; j <= k; j++) {
                        if (fabs(AP[kx]) > rowmax) {
                            rowmax = fabs(AP[kx]);
                        }
                        kx = kx + j + 1;
                    }
                    kpc = imax * (imax + 1) / 2;
                    if (imax > 0) {
                        jmax = cblas_idamax(imax, &AP[kpc], 1);
                        rowmax = fmax(rowmax, fabs(AP[kpc + jmax]));
                    }

                    if (absakk >= alpha * colmax * (colmax / rowmax)) {
                        kp = k;
                    } else if (fabs(AP[kpc + imax]) >= alpha * rowmax) {
                        kp = imax;
                    } else {
                        kp = imax;
                        kstep = 2;
                    }
                }

                kk = k - kstep + 1;
                if (kstep == 2)
                    knc = knc - k;
                if (kp != kk) {
                    if (kp > 0)
                        cblas_dswap(kp, &AP[knc], 1, &AP[kpc], 1);

                    kx = kpc + kp;
                    for (j = kp + 1; j < kk; j++) {
                        kx = kx + j;
                        t = AP[knc + j];
                        AP[knc + j] = AP[kx];
                        AP[kx] = t;
                    }

                    t = AP[knc + kk];
                    AP[knc + kk] = AP[kpc + kp];
                    AP[kpc + kp] = t;

                    if (kstep == 2) {
                        t = AP[kc + k - 1];
                        AP[kc + k - 1] = AP[kc + kp];
                        AP[kc + kp] = t;
                    }
                }

                if (kstep == 1) {
                    if (k > 0) {
                        r1 = ONE / AP[kc + k];
                        cblas_dspr(CblasColMajor, CblasUpper, k, -r1, &AP[kc], 1, AP);
                        cblas_dscal(k, r1, &AP[kc], 1);
                    }
                } else {
                    if (k > 1) {
                        d12 = AP[kc + k - 1];
                        d22 = AP[knc + k - 1] / d12;
                        d11 = AP[kc + k] / d12;
                        t = ONE / (d11 * d22 - ONE);
                        d12 = t / d12;

                        for (j = k - 2; j >= 0; j--) {
                            wkm1 = d12 * (d11 * AP[knc + j] - AP[kc + j]);
                            wk = d12 * (d22 * AP[kc + j] - AP[knc + j]);
                            for (i = j; i >= 0; i--) {
                                INT jc = i + j * (j + 1) / 2;
                                AP[jc] = AP[jc] - AP[kc + i] * wk - AP[knc + i] * wkm1;
                            }
                            AP[kc + j] = wk;
                            AP[knc + j] = wkm1;
                        }
                    }
                }
            }

            if (kstep == 1) {
                ipiv[k] = kp;
            } else {
                ipiv[k] = -(kp + 1);
                ipiv[k - 1] = -(kp + 1);
            }

            k = k - kstep;
            kc = knc - k - 1;
        }

    } else {
        /*
         * Factorize A as L*D*L**T using the lower triangle of A
         *
         * K is the main loop index, increasing from 1 to N in steps of
         * 1 or 2
         */
        k = 0;
        kc = 0;
        npp = n * (n + 1) / 2;

        while (k < n) {
            knc = kc;
            kstep = 1;

            absakk = fabs(AP[kc]);

            if (k < n - 1) {
                imax = k + 1 + cblas_idamax(n - k - 1, &AP[kc + 1], 1);
                colmax = fabs(AP[kc + imax - k]);
            } else {
                colmax = ZERO;
            }

            if (fmax(absakk, colmax) == ZERO) {
                if (*info == 0)
                    *info = k + 1;
                kp = k;
            } else {
                if (absakk >= alpha * colmax) {
                    kp = k;
                } else {
                    rowmax = ZERO;
                    kx = kc + imax - k;
                    for (j = k; j < imax; j++) {
                        if (fabs(AP[kx]) > rowmax) {
                            rowmax = fabs(AP[kx]);
                        }
                        kx = kx + n - j - 1;
                    }
                    kpc = npp - (n - imax) * (n - imax + 1) / 2;
                    if (imax < n - 1) {
                        jmax = imax + 1 + cblas_idamax(n - imax - 1, &AP[kpc + 1], 1);
                        rowmax = fmax(rowmax, fabs(AP[kpc + jmax - imax]));
                    }

                    if (absakk >= alpha * colmax * (colmax / rowmax)) {
                        kp = k;
                    } else if (fabs(AP[kpc]) >= alpha * rowmax) {
                        kp = imax;
                    } else {
                        kp = imax;
                        kstep = 2;
                    }
                }

                kk = k + kstep - 1;
                if (kstep == 2)
                    knc = knc + n - k;
                if (kp != kk) {
                    if (kp < n - 1)
                        cblas_dswap(n - kp - 1, &AP[knc + kp - kk + 1], 1, &AP[kpc + 1], 1);

                    kx = knc + kp - kk;
                    for (j = kk + 1; j < kp; j++) {
                        kx = kx + n - j;
                        t = AP[knc + j - kk];
                        AP[knc + j - kk] = AP[kx];
                        AP[kx] = t;
                    }

                    t = AP[knc];
                    AP[knc] = AP[kpc];
                    AP[kpc] = t;

                    if (kstep == 2) {
                        t = AP[kc + 1];
                        AP[kc + 1] = AP[kc + kp - k];
                        AP[kc + kp - k] = t;
                    }
                }

                if (kstep == 1) {
                    if (k < n - 1) {
                        r1 = ONE / AP[kc];
                        cblas_dspr(CblasColMajor, CblasLower, n - k - 1, -r1,
                                   &AP[kc + 1], 1, &AP[kc + n - k]);
                        cblas_dscal(n - k - 1, r1, &AP[kc + 1], 1);
                    }
                } else {
                    if (k < n - 2) {
                        d21 = AP[kc + 1];
                        d11 = AP[knc] / d21;
                        d22 = AP[kc] / d21;
                        t = ONE / (d11 * d22 - ONE);
                        d21 = t / d21;

                        for (j = k + 2; j < n; j++) {
                            wk = d21 * (d11 * AP[kc + j - k] - AP[knc + j - k - 1]);
                            wkp1 = d21 * (d22 * AP[knc + j - k - 1] - AP[kc + j - k]);
                            for (i = j; i < n; i++) {
                                INT jc = npp - (n - j) * (n - j + 1) / 2 + i - j;
                                AP[jc] = AP[jc] - AP[kc + i - k] * wk - AP[knc + i - k - 1] * wkp1;
                            }
                            AP[kc + j - k] = wk;
                            AP[knc + j - k - 1] = wkp1;
                        }
                    }
                }
            }

            if (kstep == 1) {
                ipiv[k] = kp;
            } else {
                ipiv[k] = -(kp + 1);
                ipiv[k + 1] = -(kp + 1);
            }

            k = k + kstep;
            kc = knc + n - k + 1;
        }
    }
}
