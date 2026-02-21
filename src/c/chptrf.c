/**
 * @file chptrf.c
 * @brief CHPTRF computes the factorization of a complex Hermitian matrix stored
 *        in packed format using the Bunch-Kaufman diagonal pivoting method.
 */

#include <complex.h>
#include <math.h>
#include <cblas.h>
#include "semicolon_lapack_complex_single.h"

/**
 * CHPTRF computes the factorization of a complex Hermitian packed
 * matrix A using the Bunch-Kaufman diagonal pivoting method:
 *
 *    A = U*D*U**H  or  A = L*D*L**H
 *
 * where U (or L) is a product of permutation and unit upper (lower)
 * triangular matrices, and D is Hermitian and block diagonal with
 * 1-by-1 and 2-by-2 diagonal blocks.
 *
 * @param[in]     uplo   Specifies whether the upper or lower triangular part
 *                       of the Hermitian matrix A is stored:
 *                       - = 'U': Upper triangle of A is stored
 *                       - = 'L': Lower triangle of A is stored
 * @param[in]     n      The order of the matrix A. n >= 0.
 * @param[in,out] AP     On entry, the upper or lower triangle of the Hermitian
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
void chptrf(
    const char* uplo,
    const int n,
    c64* restrict AP,
    int* restrict ipiv,
    int* info)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;
    const f32 EIGHT = 8.0f;
    const f32 SEVTEN = 17.0f;

    int upper;
    int i, imax = 0, j, jmax, k, kc, kk, knc, kp, kpc, kstep, kx, npp;
    f32 absakk, alpha, colmax, d, d11, d22, r1, rowmax, tt;
    c64 d12, d21, t, wk, wkm1, wkp1;

    *info = 0;
    upper = (uplo[0] == 'U' || uplo[0] == 'u');
    if (!upper && !(uplo[0] == 'L' || uplo[0] == 'l')) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    }
    if (*info != 0) {
        xerbla("CHPTRF", -(*info));
        return;
    }

    /* Initialize ALPHA for use in choosing pivot block size. */
    alpha = (ONE + sqrtf(SEVTEN)) / EIGHT;

    if (upper) {
        /*
         * Factorize A as U*D*U**H using the upper triangle of A
         *
         * K is the main loop index, decreasing from N to 1 in steps of
         * 1 or 2
         */
        k = n - 1;
        kc = (n - 1) * n / 2;

        while (k >= 0) {
            knc = kc;
            kstep = 1;

            absakk = fabsf(crealf(AP[kc + k]));

            /*
             * IMAX is the row-index of the largest off-diagonal element in
             * column K, and COLMAX is its absolute value
             */
            if (k > 0) {
                imax = cblas_icamax(k, &AP[kc], 1);
                colmax = cabs1f(AP[kc + imax]);
            } else {
                colmax = ZERO;
            }

            if (fmaxf(absakk, colmax) == ZERO) {
                /* Column K is zero: set INFO and continue */
                if (*info == 0)
                    *info = k + 1;
                kp = k;
                AP[kc + k] = crealf(AP[kc + k]);
            } else {
                if (absakk >= alpha * colmax) {
                    /* no interchange, use 1-by-1 pivot block */
                    kp = k;
                } else {
                    /*
                     * JMAX is the column-index of the largest off-diagonal
                     * element in row IMAX, and ROWMAX is its absolute value
                     */
                    rowmax = ZERO;
                    kx = imax + (imax + 1) * (imax + 2) / 2;
                    for (j = imax + 1; j <= k; j++) {
                        if (cabs1f(AP[kx]) > rowmax) {
                            rowmax = cabs1f(AP[kx]);
                        }
                        kx = kx + j + 1;
                    }
                    kpc = imax * (imax + 1) / 2;
                    if (imax > 0) {
                        jmax = cblas_icamax(imax, &AP[kpc], 1);
                        rowmax = fmaxf(rowmax, cabs1f(AP[kpc + jmax]));
                    }

                    if (absakk >= alpha * colmax * (colmax / rowmax)) {
                        /* no interchange, use 1-by-1 pivot block */
                        kp = k;
                    } else if (fabsf(crealf(AP[kpc + imax])) >= alpha * rowmax) {
                        /*
                         * interchange rows and columns K and IMAX, use 1-by-1
                         * pivot block
                         */
                        kp = imax;
                    } else {
                        /*
                         * interchange rows and columns K-1 and IMAX, use 2-by-2
                         * pivot block
                         */
                        kp = imax;
                        kstep = 2;
                    }
                }

                kk = k - kstep + 1;
                if (kstep == 2)
                    knc = knc - k;
                if (kp != kk) {
                    /*
                     * Interchange rows and columns KK and KP in the leading
                     * submatrix A(1:k,1:k)
                     */
                    if (kp > 0)
                        cblas_cswap(kp, &AP[knc], 1, &AP[kpc], 1);

                    kx = kpc + kp;
                    for (j = kp + 1; j < kk; j++) {
                        kx = kx + j;
                        t = conjf(AP[knc + j]);
                        AP[knc + j] = conjf(AP[kx]);
                        AP[kx] = t;
                    }
                    AP[kx + kk] = conjf(AP[kx + kk]);

                    r1 = crealf(AP[knc + kk]);
                    AP[knc + kk] = crealf(AP[kpc + kp]);
                    AP[kpc + kp] = r1;

                    if (kstep == 2) {
                        AP[kc + k] = crealf(AP[kc + k]);
                        t = AP[kc + k - 1];
                        AP[kc + k - 1] = AP[kc + kp];
                        AP[kc + kp] = t;
                    }
                } else {
                    AP[kc + k] = crealf(AP[kc + k]);
                    if (kstep == 2)
                        AP[kc - 1] = crealf(AP[kc - 1]);
                }

                /* Update the leading submatrix */

                if (kstep == 1) {
                    /*
                     * 1-by-1 pivot block D(k): column k now holds
                     *
                     * W(k) = U(k)*D(k)
                     *
                     * where U(k) is the k-th column of U
                     *
                     * Perform a rank-1 update of A(1:k-1,1:k-1) as
                     *
                     * A := A - U(k)*D(k)*U(k)**H = A - W(k)*1/D(k)*W(k)**H
                     */
                    r1 = ONE / crealf(AP[kc + k]);
                    cblas_chpr(CblasColMajor, CblasUpper, k, -r1, &AP[kc], 1, AP);

                    /* Store U(k) in column k */
                    cblas_csscal(k, r1, &AP[kc], 1);
                } else {
                    /*
                     * 2-by-2 pivot block D(k): columns k and k-1 now hold
                     *
                     * ( W(k-1) W(k) ) = ( U(k-1) U(k) )*D(k)
                     *
                     * where U(k) and U(k-1) are the k-th and (k-1)-th columns
                     * of U
                     *
                     * Perform a rank-2 update of A(1:k-2,1:k-2) as
                     *
                     * A := A - ( U(k-1) U(k) )*D(k)*( U(k-1) U(k) )**H
                     *    = A - ( W(k-1) W(k) )*inv(D(k))*( W(k-1) W(k) )**H
                     */
                    if (k > 1) {

                        d = slapy2(crealf(AP[kc + k - 1]), cimagf(AP[kc + k - 1]));
                        d22 = crealf(AP[knc + k - 1]) / d;
                        d11 = crealf(AP[kc + k]) / d;
                        tt = ONE / (d11 * d22 - ONE);
                        d12 = AP[kc + k - 1] / d;
                        d = tt / d;

                        for (j = k - 2; j >= 0; j--) {
                            wkm1 = d * (d11 * AP[knc + j] - conjf(d12) * AP[kc + j]);
                            wk = d * (d22 * AP[kc + j] - d12 * AP[knc + j]);
                            for (i = j; i >= 0; i--) {
                                int jc_idx = i + j * (j + 1) / 2;
                                AP[jc_idx] = AP[jc_idx] - AP[kc + i] * conjf(wk) - AP[knc + i] * conjf(wkm1);
                            }
                            AP[kc + j] = wk;
                            AP[knc + j] = wkm1;
                            AP[j + j * (j + 1) / 2] = crealf(AP[j + j * (j + 1) / 2]);
                        }

                    }
                }
            }

            /* Store details of the interchanges in IPIV */
            if (kstep == 1) {
                ipiv[k] = kp;
            } else {
                ipiv[k] = -(kp + 1);
                ipiv[k - 1] = -(kp + 1);
            }

            /* Decrease K and return to the start of the main loop */
            k = k - kstep;
            kc = knc - k - 1;
        }

    } else {
        /*
         * Factorize A as L*D*L**H using the lower triangle of A
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

            absakk = fabsf(crealf(AP[kc]));

            /*
             * IMAX is the row-index of the largest off-diagonal element in
             * column K, and COLMAX is its absolute value
             */
            if (k < n - 1) {
                imax = k + 1 + cblas_icamax(n - k - 1, &AP[kc + 1], 1);
                colmax = cabs1f(AP[kc + imax - k]);
            } else {
                colmax = ZERO;
            }

            if (fmaxf(absakk, colmax) == ZERO) {
                /* Column K is zero: set INFO and continue */
                if (*info == 0)
                    *info = k + 1;
                kp = k;
                AP[kc] = crealf(AP[kc]);
            } else {
                if (absakk >= alpha * colmax) {
                    /* no interchange, use 1-by-1 pivot block */
                    kp = k;
                } else {
                    /*
                     * JMAX is the column-index of the largest off-diagonal
                     * element in row IMAX, and ROWMAX is its absolute value
                     */
                    rowmax = ZERO;
                    kx = kc + imax - k;
                    for (j = k; j < imax; j++) {
                        if (cabs1f(AP[kx]) > rowmax) {
                            rowmax = cabs1f(AP[kx]);
                        }
                        kx = kx + n - j - 1;
                    }
                    kpc = npp - (n - imax) * (n - imax + 1) / 2;
                    if (imax < n - 1) {
                        jmax = imax + 1 + cblas_icamax(n - imax - 1, &AP[kpc + 1], 1);
                        rowmax = fmaxf(rowmax, cabs1f(AP[kpc + jmax - imax]));
                    }

                    if (absakk >= alpha * colmax * (colmax / rowmax)) {
                        /* no interchange, use 1-by-1 pivot block */
                        kp = k;
                    } else if (fabsf(crealf(AP[kpc])) >= alpha * rowmax) {
                        /*
                         * interchange rows and columns K and IMAX, use 1-by-1
                         * pivot block
                         */
                        kp = imax;
                    } else {
                        /*
                         * interchange rows and columns K+1 and IMAX, use 2-by-2
                         * pivot block
                         */
                        kp = imax;
                        kstep = 2;
                    }
                }

                kk = k + kstep - 1;
                if (kstep == 2)
                    knc = knc + n - k;
                if (kp != kk) {
                    /*
                     * Interchange rows and columns KK and KP in the trailing
                     * submatrix A(k:n,k:n)
                     */
                    if (kp < n - 1)
                        cblas_cswap(n - kp - 1, &AP[knc + kp - kk + 1], 1, &AP[kpc + 1], 1);

                    kx = knc + kp - kk;
                    for (j = kk + 1; j < kp; j++) {
                        kx = kx + n - j;
                        t = conjf(AP[knc + j - kk]);
                        AP[knc + j - kk] = conjf(AP[kx]);
                        AP[kx] = t;
                    }
                    AP[knc + kp - kk] = conjf(AP[knc + kp - kk]);

                    r1 = crealf(AP[knc]);
                    AP[knc] = crealf(AP[kpc]);
                    AP[kpc] = r1;

                    if (kstep == 2) {
                        AP[kc] = crealf(AP[kc]);
                        t = AP[kc + 1];
                        AP[kc + 1] = AP[kc + kp - k];
                        AP[kc + kp - k] = t;
                    }
                } else {
                    AP[kc] = crealf(AP[kc]);
                    if (kstep == 2)
                        AP[knc] = crealf(AP[knc]);
                }

                /* Update the trailing submatrix */

                if (kstep == 1) {
                    /*
                     * 1-by-1 pivot block D(k): column k now holds
                     *
                     * W(k) = L(k)*D(k)
                     *
                     * where L(k) is the k-th column of L
                     */
                    if (k < n - 1) {
                        /*
                         * Perform a rank-1 update of A(k+1:n,k+1:n) as
                         *
                         * A := A - L(k)*D(k)*L(k)**H = A - W(k)*(1/D(k))*W(k)**H
                         */
                        r1 = ONE / crealf(AP[kc]);
                        cblas_chpr(CblasColMajor, CblasLower, n - k - 1, -r1,
                                   &AP[kc + 1], 1, &AP[kc + n - k]);

                        /* Store L(k) in column K */
                        cblas_csscal(n - k - 1, r1, &AP[kc + 1], 1);
                    }
                } else {
                    /*
                     * 2-by-2 pivot block D(k): columns K and K+1 now hold
                     *
                     * ( W(k) W(k+1) ) = ( L(k) L(k+1) )*D(k)
                     *
                     * where L(k) and L(k+1) are the k-th and (k+1)-th columns
                     * of L
                     */
                    if (k < n - 2) {
                        /*
                         * Perform a rank-2 update of A(k+2:n,k+2:n) as
                         *
                         * A := A - ( L(k) L(k+1) )*D(k)*( L(k) L(k+1) )**H
                         *    = A - ( W(k) W(k+1) )*inv(D(k))*( W(k) W(k+1) )**H
                         */
                        d = slapy2(crealf(AP[kc + 1]), cimagf(AP[kc + 1]));
                        d11 = crealf(AP[knc]) / d;
                        d22 = crealf(AP[kc]) / d;
                        tt = ONE / (d11 * d22 - ONE);
                        d21 = AP[kc + 1] / d;
                        d = tt / d;

                        for (j = k + 2; j < n; j++) {
                            wk = d * (d11 * AP[kc + j - k] - d21 * AP[knc + j - k - 1]);
                            wkp1 = d * (d22 * AP[knc + j - k - 1] - conjf(d21) * AP[kc + j - k]);
                            for (i = j; i < n; i++) {
                                int jc_idx = npp - (n - j) * (n - j + 1) / 2 + i - j;
                                AP[jc_idx] = AP[jc_idx] - AP[kc + i - k] * conjf(wk) - AP[knc + i - k - 1] * conjf(wkp1);
                            }
                            AP[kc + j - k] = wk;
                            AP[knc + j - k - 1] = wkp1;
                            AP[npp - (n - j) * (n - j + 1) / 2] = crealf(AP[npp - (n - j) * (n - j + 1) / 2]);
                        }
                    }
                }
            }

            /* Store details of the interchanges in IPIV */
            if (kstep == 1) {
                ipiv[k] = kp;
            } else {
                ipiv[k] = -(kp + 1);
                ipiv[k + 1] = -(kp + 1);
            }

            /* Increase K and return to the start of the main loop */
            k = k + kstep;
            kc = knc + n - k + 1;
        }
    }
}
