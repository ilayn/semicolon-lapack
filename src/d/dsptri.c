/**
 * @file dsptri.c
 * @brief DSPTRI computes the inverse of a real symmetric indefinite matrix in packed storage.
 */

#include <math.h>
#include <stdlib.h>
#include <cblas.h>
#include "semicolon_lapack_double.h"

/**
 * DSPTRI computes the inverse of a real symmetric indefinite matrix A in
 * packed storage using the factorization A = U*D*U**T or A = L*D*L**T
 * computed by DSPTRF.
 *
 * @param[in]     uplo   Specifies whether the details of the factorization are
 *                       stored as an upper or lower triangular matrix:
 *                       - = 'U': Upper triangular, form is A = U*D*U**T
 *                       - = 'L': Lower triangular, form is A = L*D*L**T
 * @param[in]     n      The order of the matrix A. n >= 0.
 * @param[in,out] AP     On entry, the block diagonal matrix D and the multipliers
 *                       used to obtain the factor U or L as computed by DSPTRF,
 *                       stored as a packed triangular matrix of dimension n*(n+1)/2.
 *                       On exit, if info = 0, the (symmetric) inverse of the
 *                       original matrix, stored as a packed triangular matrix.
 * @param[in]     ipiv   Details of the interchanges and the block structure of D
 *                       as determined by DSPTRF. Array of dimension n.
 * @param[out]    work   Workspace array of dimension n.
 * @param[out]    info   Exit status:
 *                       - = 0: successful exit
 *                       - < 0: if info = -i, the i-th argument had an illegal value
 *                       - > 0: if info = i, D(i,i) = 0; the matrix is singular and
 *                              its inverse could not be computed.
 */
void dsptri(
    const char* uplo,
    const int n,
    double* const restrict AP,
    const int* const restrict ipiv,
    double* const restrict work,
    int* info)
{
    const double ONE = 1.0;
    const double ZERO = 0.0;

    int upper;
    int j, k, kc, kcnext, kp, kpc, kstep, kx, npp;
    double ak, akkp1, akp1, d, t, temp;

    *info = 0;
    upper = (uplo[0] == 'U' || uplo[0] == 'u');
    if (!upper && !(uplo[0] == 'L' || uplo[0] == 'l')) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    }
    if (*info != 0) {
        xerbla("DSPTRI", -(*info));
        return;
    }

    if (n == 0) return;

    if (upper) {
        /*
         * Upper triangular storage: examine D from bottom to top
         * Check that the diagonal matrix D is nonsingular
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
         * Lower triangular storage: examine D from top to bottom
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
         * Compute inv(A) from the factorization A = U*D*U**T.
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
                AP[kc + k] = ONE / AP[kc + k];

                if (k > 0) {
                    cblas_dcopy(k, &AP[kc], 1, work, 1);
                    cblas_dspmv(CblasColMajor, CblasUpper, k, -ONE, AP, work, 1, ZERO, &AP[kc], 1);
                    AP[kc + k] = AP[kc + k] - cblas_ddot(k, work, 1, &AP[kc], 1);
                }
                kstep = 1;
            } else {
                /*
                 * 2 x 2 diagonal block
                 *
                 * Invert the diagonal block.
                 */
                t = fabs(AP[kcnext + k]);
                ak = AP[kc + k] / t;
                akp1 = AP[kcnext + k + 1] / t;
                akkp1 = AP[kcnext + k] / t;
                d = t * (ak * akp1 - ONE);
                AP[kc + k] = akp1 / d;
                AP[kcnext + k + 1] = ak / d;
                AP[kcnext + k] = -akkp1 / d;

                if (k > 0) {
                    cblas_dcopy(k, &AP[kc], 1, work, 1);
                    cblas_dspmv(CblasColMajor, CblasUpper, k, -ONE, AP, work, 1, ZERO, &AP[kc], 1);
                    AP[kc + k] = AP[kc + k] - cblas_ddot(k, work, 1, &AP[kc], 1);
                    AP[kcnext + k] = AP[kcnext + k] - cblas_ddot(k, &AP[kc], 1, &AP[kcnext], 1);
                    cblas_dcopy(k, &AP[kcnext], 1, work, 1);
                    cblas_dspmv(CblasColMajor, CblasUpper, k, -ONE, AP, work, 1, ZERO, &AP[kcnext], 1);
                    AP[kcnext + k + 1] = AP[kcnext + k + 1] - cblas_ddot(k, work, 1, &AP[kcnext], 1);
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
                cblas_dswap(kp, &AP[kc], 1, &AP[kpc], 1);
                kx = kpc + kp;
                for (j = kp + 1; j < k; j++) {
                    kx = kx + j;
                    temp = AP[kc + j];
                    AP[kc + j] = AP[kx];
                    AP[kx] = temp;
                }
                temp = AP[kc + k];
                AP[kc + k] = AP[kpc + kp];
                AP[kpc + kp] = temp;
                if (kstep == 2) {
                    temp = AP[kcnext - 1];
                    AP[kcnext - 1] = AP[kc + kp + 1];
                    AP[kc + kp + 1] = temp;
                }
            }

            k = k + kstep;
            kc = kcnext;
        }
    } else {
        /*
         * Compute inv(A) from the factorization A = L*D*L**T.
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
                AP[kc] = ONE / AP[kc];

                if (k < n - 1) {
                    cblas_dcopy(n - k - 1, &AP[kc + 1], 1, work, 1);
                    cblas_dspmv(CblasColMajor, CblasLower, n - k - 1, -ONE, &AP[kc + n - k], work, 1, ZERO, &AP[kc + 1], 1);
                    AP[kc] = AP[kc] - cblas_ddot(n - k - 1, work, 1, &AP[kc + 1], 1);
                }
                kstep = 1;
            } else {
                /*
                 * 2 x 2 diagonal block
                 *
                 * Invert the diagonal block.
                 */
                t = fabs(AP[kcnext + 1]);
                ak = AP[kcnext] / t;
                akp1 = AP[kc] / t;
                akkp1 = AP[kcnext + 1] / t;
                d = t * (ak * akp1 - ONE);
                AP[kcnext] = akp1 / d;
                AP[kc] = ak / d;
                AP[kcnext + 1] = -akkp1 / d;

                if (k < n - 1) {
                    cblas_dcopy(n - k - 1, &AP[kc + 1], 1, work, 1);
                    cblas_dspmv(CblasColMajor, CblasLower, n - k - 1, -ONE, &AP[kc + n - k], work, 1, ZERO, &AP[kc + 1], 1);
                    AP[kc] = AP[kc] - cblas_ddot(n - k - 1, work, 1, &AP[kc + 1], 1);
                    AP[kcnext + 1] = AP[kcnext + 1] - cblas_ddot(n - k - 1, &AP[kc + 1], 1, &AP[kcnext + 2], 1);
                    cblas_dcopy(n - k - 1, &AP[kcnext + 2], 1, work, 1);
                    cblas_dspmv(CblasColMajor, CblasLower, n - k - 1, -ONE, &AP[kc + n - k], work, 1, ZERO, &AP[kcnext + 2], 1);
                    AP[kcnext] = AP[kcnext] - cblas_ddot(n - k - 1, work, 1, &AP[kcnext + 2], 1);
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
                    cblas_dswap(n - kp - 1, &AP[kc + kp - k + 1], 1, &AP[kpc + 1], 1);
                kx = kc + kp - k;
                for (j = k + 1; j < kp; j++) {
                    kx = kx + n - j;
                    temp = AP[kc + j - k];
                    AP[kc + j - k] = AP[kx];
                    AP[kx] = temp;
                }
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
