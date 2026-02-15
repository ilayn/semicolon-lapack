/**
 * @file chptri.c
 * @brief CHPTRI computes the inverse of a complex Hermitian indefinite matrix in packed storage.
 */

#include <complex.h>
#include <math.h>
#include <cblas.h>
#include "semicolon_lapack_complex_single.h"

/**
 * CHPTRI computes the inverse of a complex Hermitian indefinite matrix A in
 * packed storage using the factorization A = U*D*U**H or A = L*D*L**H
 * computed by CHPTRF.
 *
 * @param[in]     uplo   Specifies whether the details of the factorization are
 *                       stored as an upper or lower triangular matrix:
 *                       - = 'U': Upper triangular, form is A = U*D*U**H
 *                       - = 'L': Lower triangular, form is A = L*D*L**H
 * @param[in]     n      The order of the matrix A. n >= 0.
 * @param[in,out] AP     On entry, the block diagonal matrix D and the multipliers
 *                       used to obtain the factor U or L as computed by CHPTRF,
 *                       stored as a packed triangular matrix of dimension n*(n+1)/2.
 *                       On exit, if info = 0, the (Hermitian) inverse of the
 *                       original matrix, stored as a packed triangular matrix.
 * @param[in]     ipiv   Details of the interchanges and the block structure of D
 *                       as determined by CHPTRF. Array of dimension n.
 * @param[out]    work   Workspace array of dimension n.
 * @param[out]    info
 *                           Exit status:
 *                           - = 0: successful exit
 *                           - < 0: if info = -i, the i-th argument had an illegal value
 *                           - > 0: if info = i, D(i,i) = 0; the matrix is singular and
 *                           its inverse could not be computed.
 */
void chptri(
    const char* uplo,
    const int n,
    c64* restrict AP,
    const int* restrict ipiv,
    c64* restrict work,
    int* info)
{
    const f32 ONE = 1.0f;
    const c64 ZERO = CMPLXF(0.0f, 0.0f);

    int upper;
    int j, k, kc, kcnext, kp, kpc, kstep, kx, npp;
    f32 ak, akp1, d, t;
    c64 akkp1, temp;

    *info = 0;
    upper = (uplo[0] == 'U' || uplo[0] == 'u');
    if (!upper && !(uplo[0] == 'L' || uplo[0] == 'l')) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    }
    if (*info != 0) {
        xerbla("CHPTRI", -(*info));
        return;
    }

    if (n == 0) return;

    if (upper) {
        /*
         * Upper triangular storage: examine D from bottom to top
         */
        kp = n * (n + 1) / 2 - 1;
        for (k = n - 1; k >= 0; k--) {
            if (ipiv[k] >= 0 && AP[kp] == ZERO) {
                *info = k + 1;
                return;
            }
            kp = kp - (k + 1);
        }
    } else {
        /*
         * Lower triangular storage: examine D from top to bottom.
         */
        kp = 0;
        for (k = 0; k < n; k++) {
            if (ipiv[k] >= 0 && AP[kp] == ZERO) {
                *info = k + 1;
                return;
            }
            kp = kp + n - k;
        }
    }
    *info = 0;

    if (upper) {
        /*
         * Compute inv(A) from the factorization A = U*D*U**H.
         *
         * K is the main loop index, increasing from 0 to N-1 in steps of
         * 1 or 2, depending on the size of the diagonal blocks.
         */
        k = 0;
        kc = 0;

        while (k < n) {
            kcnext = kc + k + 1;
            if (ipiv[k] >= 0) {
                /*
                 * 1 x 1 diagonal block
                 *
                 * Invert the diagonal block.
                 */
                AP[kc + k] = CMPLXF(ONE / crealf(AP[kc + k]), 0.0f);

                if (k > 0) {
                    c64 dotresult;
                    cblas_ccopy(k, &AP[kc], 1, work, 1);
                    const c64 NEG_CONE = CMPLXF(-1.0f, 0.0f);
                    cblas_chpmv(CblasColMajor, CblasUpper, k, &NEG_CONE, AP, work, 1, &ZERO, &AP[kc], 1);
                    cblas_cdotc_sub(k, work, 1, &AP[kc], 1, &dotresult);
                    AP[kc + k] = AP[kc + k] - CMPLXF(crealf(dotresult), 0.0f);
                }
                kstep = 1;
            } else {
                /*
                 * 2 x 2 diagonal block
                 *
                 * Invert the diagonal block.
                 */
                t = cabsf(AP[kcnext + k]);
                ak = crealf(AP[kc + k]) / t;
                akp1 = crealf(AP[kcnext + k + 1]) / t;
                akkp1 = AP[kcnext + k] / t;
                d = t * (ak * akp1 - ONE);
                AP[kc + k] = CMPLXF(akp1 / d, 0.0f);
                AP[kcnext + k + 1] = CMPLXF(ak / d, 0.0f);
                AP[kcnext + k] = -akkp1 / d;

                if (k > 0) {
                    c64 dotresult;
                    const c64 NEG_CONE = CMPLXF(-1.0f, 0.0f);

                    cblas_ccopy(k, &AP[kc], 1, work, 1);
                    cblas_chpmv(CblasColMajor, CblasUpper, k, &NEG_CONE, AP, work, 1, &ZERO, &AP[kc], 1);
                    cblas_cdotc_sub(k, work, 1, &AP[kc], 1, &dotresult);
                    AP[kc + k] = AP[kc + k] - CMPLXF(crealf(dotresult), 0.0f);
                    cblas_cdotc_sub(k, &AP[kc], 1, &AP[kcnext], 1, &dotresult);
                    AP[kcnext + k] = AP[kcnext + k] - dotresult;
                    cblas_ccopy(k, &AP[kcnext], 1, work, 1);
                    cblas_chpmv(CblasColMajor, CblasUpper, k, &NEG_CONE, AP, work, 1, &ZERO, &AP[kcnext], 1);
                    cblas_cdotc_sub(k, work, 1, &AP[kcnext], 1, &dotresult);
                    AP[kcnext + k + 1] = AP[kcnext + k + 1] - CMPLXF(crealf(dotresult), 0.0f);
                }
                kstep = 2;
                kcnext = kcnext + k + 2;
            }

            kp = (ipiv[k] >= 0) ? ipiv[k] : -ipiv[k] - 1;
            if (kp != k) {
                /*
                 * Interchange rows and columns K and KP in the leading
                 * submatrix A(0:k,0:k)
                 */
                kpc = kp * (kp + 1) / 2;
                cblas_cswap(kp, &AP[kc], 1, &AP[kpc], 1);
                kx = kpc + kp;
                for (j = kp + 1; j < k; j++) {
                    kx = kx + j;
                    temp = conjf(AP[kc + j]);
                    AP[kc + j] = conjf(AP[kx]);
                    AP[kx] = temp;
                }
                AP[kc + kp] = conjf(AP[kc + kp]);
                temp = AP[kc + k];
                AP[kc + k] = AP[kpc + kp];
                AP[kpc + kp] = temp;
                if (kstep == 2) {
                    temp = AP[kc + 2*k + 1];
                    AP[kc + 2*k + 1] = AP[kc + k + kp + 1];
                    AP[kc + k + kp + 1] = temp;
                }
            }

            k = k + kstep;
            kc = kcnext;
        }
    } else {
        /*
         * Compute inv(A) from the factorization A = L*D*L**H.
         *
         * K is the main loop index, decreasing from N-1 to 0 in steps of
         * 1 or 2, depending on the size of the diagonal blocks.
         */
        npp = n * (n + 1) / 2;
        k = n - 1;
        kc = npp - 1;

        while (k >= 0) {
            kcnext = kc - (n - k + 1);
            if (ipiv[k] >= 0) {
                /*
                 * 1 x 1 diagonal block
                 *
                 * Invert the diagonal block.
                 */
                AP[kc] = CMPLXF(ONE / crealf(AP[kc]), 0.0f);

                if (k < n - 1) {
                    c64 dotresult;
                    const c64 NEG_CONE = CMPLXF(-1.0f, 0.0f);
                    cblas_ccopy(n - k - 1, &AP[kc + 1], 1, work, 1);
                    cblas_chpmv(CblasColMajor, CblasLower, n - k - 1, &NEG_CONE, &AP[kc + n - k], work, 1, &ZERO, &AP[kc + 1], 1);
                    cblas_cdotc_sub(n - k - 1, work, 1, &AP[kc + 1], 1, &dotresult);
                    AP[kc] = AP[kc] - CMPLXF(crealf(dotresult), 0.0f);
                }
                kstep = 1;
            } else {
                /*
                 * 2 x 2 diagonal block
                 *
                 * Invert the diagonal block.
                 */
                t = cabsf(AP[kcnext + 1]);
                ak = crealf(AP[kcnext]) / t;
                akp1 = crealf(AP[kc]) / t;
                akkp1 = AP[kcnext + 1] / t;
                d = t * (ak * akp1 - ONE);
                AP[kcnext] = CMPLXF(akp1 / d, 0.0f);
                AP[kc] = CMPLXF(ak / d, 0.0f);
                AP[kcnext + 1] = -akkp1 / d;

                if (k < n - 1) {
                    c64 dotresult;
                    const c64 NEG_CONE = CMPLXF(-1.0f, 0.0f);

                    cblas_ccopy(n - k - 1, &AP[kc + 1], 1, work, 1);
                    cblas_chpmv(CblasColMajor, CblasLower, n - k - 1, &NEG_CONE, &AP[kc + n - k], work, 1, &ZERO, &AP[kc + 1], 1);
                    cblas_cdotc_sub(n - k - 1, work, 1, &AP[kc + 1], 1, &dotresult);
                    AP[kc] = AP[kc] - CMPLXF(crealf(dotresult), 0.0f);
                    cblas_cdotc_sub(n - k - 1, &AP[kc + 1], 1, &AP[kcnext + 2], 1, &dotresult);
                    AP[kcnext + 1] = AP[kcnext + 1] - dotresult;
                    cblas_ccopy(n - k - 1, &AP[kcnext + 2], 1, work, 1);
                    cblas_chpmv(CblasColMajor, CblasLower, n - k - 1, &NEG_CONE, &AP[kc + n - k], work, 1, &ZERO, &AP[kcnext + 2], 1);
                    cblas_cdotc_sub(n - k - 1, work, 1, &AP[kcnext + 2], 1, &dotresult);
                    AP[kcnext] = AP[kcnext] - CMPLXF(crealf(dotresult), 0.0f);
                }
                kstep = 2;
                kcnext = kcnext - (n - k + 2);
            }

            kp = (ipiv[k] >= 0) ? ipiv[k] : -ipiv[k] - 1;
            if (kp != k) {
                /*
                 * Interchange rows and columns K and KP in the trailing
                 * submatrix A(k:n-1,k:n-1)
                 */
                kpc = npp - (n - kp) * (n - kp + 1) / 2;
                if (kp < n - 1)
                    cblas_cswap(n - kp - 1, &AP[kc + kp - k + 1], 1, &AP[kpc + 1], 1);
                kx = kc + kp - k;
                for (j = k + 1; j < kp; j++) {
                    kx = kx + n - j;
                    temp = conjf(AP[kc + j - k]);
                    AP[kc + j - k] = conjf(AP[kx]);
                    AP[kx] = temp;
                }
                AP[kc + kp - k] = conjf(AP[kc + kp - k]);
                temp = AP[kc];
                AP[kc] = AP[kpc];
                AP[kpc] = temp;
                if (kstep == 2) {
                    temp = AP[kc - n + k];
                    AP[kc - n + k] = AP[kc - n + kp];
                    AP[kc - n + kp] = temp;
                }
            }

            k = k - kstep;
            kc = kcnext;
        }
    }
}
