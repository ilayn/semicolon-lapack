/**
 * @file dspgst.c
 * @brief DSPGST reduces a real symmetric-definite generalized eigenproblem
 *        to standard form, using packed storage.
 */

#include "semicolon_cblas.h"
#include "semicolon_lapack_double.h"

/**
 * DSPGST reduces a real symmetric-definite generalized eigenproblem
 * to standard form, using packed storage.
 *
 * If ITYPE = 1, the problem is A*x = lambda*B*x,
 * and A is overwritten by inv(U**T)*A*inv(U) or inv(L)*A*inv(L**T)
 *
 * If ITYPE = 2 or 3, the problem is A*B*x = lambda*x or
 * B*A*x = lambda*x, and A is overwritten by U*A*U**T or L**T*A*L.
 *
 * B must have been previously factorized as U**T*U or L*L**T by DPPTRF.
 *
 * @param[in]     itype  = 1: compute inv(U**T)*A*inv(U) or inv(L)*A*inv(L**T);
 *                       = 2 or 3: compute U*A*U**T or L**T*A*L.
 * @param[in]     uplo   = 'U': Upper triangle of A is stored and B is factored
 *                              as U**T*U;
 *                       = 'L': Lower triangle of A is stored and B is factored
 *                              as L*L**T.
 * @param[in]     n      The order of the matrices A and B. n >= 0.
 * @param[in,out] AP     Double precision array, dimension (n*(n+1)/2).
 *                       On entry, the upper or lower triangle of the symmetric
 *                       matrix A, packed columnwise in a linear array.
 *                       On exit, if info = 0, the transformed matrix.
 * @param[in]     BP     Double precision array, dimension (n*(n+1)/2).
 *                       The triangular factor from the Cholesky factorization
 *                       of B, stored in the same format as A.
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value.
 */
void dspgst(const INT itype, const char* uplo, const INT n,
            f64* restrict AP, const f64* restrict BP,
            INT* info)
{
    const f64 ONE = 1.0;
    const f64 HALF = 0.5;

    INT upper;
    INT j, j1, j1j1, jj, k, k1, k1k1, kk;
    f64 ajj, akk, bjj, bkk, ct;

    *info = 0;
    upper = (uplo[0] == 'U' || uplo[0] == 'u');
    if (itype < 1 || itype > 3) {
        *info = -1;
    } else if (!upper && !(uplo[0] == 'L' || uplo[0] == 'l')) {
        *info = -2;
    } else if (n < 0) {
        *info = -3;
    }
    if (*info != 0) {
        xerbla("DSPGST", -(*info));
        return;
    }

    if (itype == 1) {
        if (upper) {

            /* Compute inv(U**T)*A*inv(U)

               j1 and jj are the indices of A(0,j) and A(j,j) */

            jj = -1;
            for (j = 0; j < n; j++) {
                j1 = jj + 1;
                jj = jj + j + 1;

                /* Compute the j-th column of the upper triangle of A */

                bjj = BP[jj];
                cblas_dtpsv(CblasColMajor, CblasUpper, CblasTrans, CblasNonUnit,
                            j + 1, BP, &AP[j1], 1);
                cblas_dspmv(CblasColMajor, CblasUpper, j, -ONE, AP, &BP[j1], 1,
                            ONE, &AP[j1], 1);
                cblas_dscal(j, ONE / bjj, &AP[j1], 1);
                AP[jj] = (AP[jj] - cblas_ddot(j, &AP[j1], 1, &BP[j1], 1)) / bjj;
            }
        } else {

            /* Compute inv(L)*A*inv(L**T)

               kk and k1k1 are the indices of A(k,k) and A(k+1,k+1) */

            kk = 0;
            for (k = 0; k < n; k++) {
                k1k1 = kk + n - k;

                /* Update the lower triangle of A(k:n-1,k:n-1) */

                akk = AP[kk];
                bkk = BP[kk];
                akk = akk / (bkk * bkk);
                AP[kk] = akk;
                if (k < n - 1) {
                    cblas_dscal(n - k - 1, ONE / bkk, &AP[kk + 1], 1);
                    ct = -HALF * akk;
                    cblas_daxpy(n - k - 1, ct, &BP[kk + 1], 1, &AP[kk + 1], 1);
                    cblas_dspr2(CblasColMajor, CblasLower, n - k - 1, -ONE,
                                &AP[kk + 1], 1, &BP[kk + 1], 1, &AP[k1k1]);
                    cblas_daxpy(n - k - 1, ct, &BP[kk + 1], 1, &AP[kk + 1], 1);
                    cblas_dtpsv(CblasColMajor, CblasLower, CblasNoTrans,
                                CblasNonUnit, n - k - 1, &BP[k1k1],
                                &AP[kk + 1], 1);
                }
                kk = k1k1;
            }
        }
    } else {
        if (upper) {

            /* Compute U*A*U**T

               k1 and kk are the indices of A(0,k) and A(k,k) */

            kk = -1;
            for (k = 0; k < n; k++) {
                k1 = kk + 1;
                kk = kk + k + 1;

                /* Update the upper triangle of A(0:k,0:k) */

                akk = AP[kk];
                bkk = BP[kk];
                cblas_dtpmv(CblasColMajor, CblasUpper, CblasNoTrans, CblasNonUnit,
                            k, BP, &AP[k1], 1);
                ct = HALF * akk;
                cblas_daxpy(k, ct, &BP[k1], 1, &AP[k1], 1);
                cblas_dspr2(CblasColMajor, CblasUpper, k, ONE, &AP[k1], 1,
                            &BP[k1], 1, AP);
                cblas_daxpy(k, ct, &BP[k1], 1, &AP[k1], 1);
                cblas_dscal(k, bkk, &AP[k1], 1);
                AP[kk] = akk * bkk * bkk;
            }
        } else {

            /* Compute L**T *A*L

               jj and j1j1 are the indices of A(j,j) and A(j+1,j+1) */

            jj = 0;
            for (j = 0; j < n; j++) {
                j1j1 = jj + n - j;

                /* Compute the j-th column of the lower triangle of A */

                ajj = AP[jj];
                bjj = BP[jj];
                AP[jj] = ajj * bjj + cblas_ddot(n - j - 1, &AP[jj + 1], 1,
                                                 &BP[jj + 1], 1);
                cblas_dscal(n - j - 1, bjj, &AP[jj + 1], 1);
                cblas_dspmv(CblasColMajor, CblasLower, n - j - 1, ONE,
                            &AP[j1j1], &BP[jj + 1], 1, ONE, &AP[jj + 1], 1);
                cblas_dtpmv(CblasColMajor, CblasLower, CblasTrans, CblasNonUnit,
                            n - j, &BP[jj], &AP[jj], 1);
                jj = j1j1;
            }
        }
    }
}
