/**
 * @file zhpgst.c
 * @brief ZHPGST reduces a complex Hermitian-definite generalized eigenproblem to standard form, using packed storage.
 */

#include "semicolon_lapack_complex_double.h"
#include <complex.h>
#include "semicolon_cblas.h"

/**
 * ZHPGST reduces a complex Hermitian-definite generalized
 * eigenproblem to standard form, using packed storage.
 *
 * If ITYPE = 1, the problem is A*x = lambda*B*x,
 * and A is overwritten by inv(U**H)*A*inv(U) or inv(L)*A*inv(L**H)
 *
 * If ITYPE = 2 or 3, the problem is A*B*x = lambda*x or
 * B*A*x = lambda*x, and A is overwritten by U*A*U**H or L**H*A*L.
 *
 * B must have been previously factorized as U**H*U or L*L**H by ZPPTRF.
 *
 * @param[in]     itype  = 1: compute inv(U**H)*A*inv(U) or inv(L)*A*inv(L**H);
 *                        = 2 or 3: compute U*A*U**H or L**H*A*L.
 * @param[in]     uplo   = 'U': Upper triangle of A is stored and B is factored as
 *                         U**H*U;
 *                        = 'L': Lower triangle of A is stored and B is factored as
 *                         L*L**H.
 * @param[in]     n      The order of the matrices A and B. n >= 0.
 * @param[in,out] AP     Complex*16 array, dimension (n*(n+1)/2).
 *                        On entry, the upper or lower triangle of the Hermitian matrix
 *                        A, packed columnwise in a linear array.
 *                        On exit, if info = 0, the transformed matrix, stored in the
 *                        same format as A.
 * @param[in]     BP     Complex*16 array, dimension (n*(n+1)/2).
 *                        The triangular factor from the Cholesky factorization of B,
 *                        stored in the same format as A, as returned by ZPPTRF.
 * @param[out]    info   = 0: successful exit
 *                        < 0: if info = -i, the i-th argument had an illegal value
 */
void zhpgst(
    const INT itype,
    const char* uplo,
    const INT n,
    c128* restrict AP,
    const c128* restrict BP,
    INT* info)
{
    const f64 ONE = 1.0;
    const f64 HALF = 0.5;
    const c128 CONE = CMPLX(1.0, 0.0);
    const c128 NEG_CONE = CMPLX(-1.0, 0.0);

    INT upper;
    INT j, j1, j1j1, jj, k, k1, k1k1, kk;
    f64 ajj, akk, bjj, bkk;
    c128 ct;

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
        xerbla("ZHPGST", -(*info));
        return;
    }

    if (itype == 1) {
        if (upper) {

            /* Compute inv(U**H)*A*inv(U) */

            /* j1 and jj are the indices of A(1,j) and A(j,j) */

            jj = -1;
            for (j = 0; j < n; j++) {
                j1 = jj + 1;
                jj = jj + j + 1;

                /* Compute the j-th column of the upper triangle of A */

                AP[jj] = CMPLX(creal(AP[jj]), 0.0);
                bjj = creal(BP[jj]);
                cblas_ztpsv(CblasColMajor, CblasUpper, CblasConjTrans,
                            CblasNonUnit, j + 1, BP, &AP[j1], 1);
                if (j > 0) {
                    cblas_zhpmv(CblasColMajor, CblasUpper, j,
                                &NEG_CONE, AP, &BP[j1], 1, &CONE,
                                &AP[j1], 1);
                    cblas_zdscal(j, ONE / bjj, &AP[j1], 1);
                }
                c128 dotc;
                if (j > 0) {
                    cblas_zdotc_sub(j, &AP[j1], 1, &BP[j1], 1, &dotc);
                } else {
                    dotc = CMPLX(0.0, 0.0);
                }
                AP[jj] = (AP[jj] - dotc) / bjj;
            }
        } else {

            /* Compute inv(L)*A*inv(L**H) */

            /* kk and k1k1 are the indices of A(k,k) and A(k+1,k+1) */

            kk = 0;
            for (k = 0; k < n; k++) {
                k1k1 = kk + n - k;

                /* Update the lower triangle of A(k:n,k:n) */

                akk = creal(AP[kk]);
                bkk = creal(BP[kk]);
                akk = akk / (bkk * bkk);
                AP[kk] = CMPLX(akk, 0.0);
                if (k < n - 1) {
                    cblas_zdscal(n - k - 1, ONE / bkk, &AP[kk + 1], 1);
                    ct = CMPLX(-HALF * akk, 0.0);
                    cblas_zaxpy(n - k - 1, &ct, &BP[kk + 1], 1,
                                &AP[kk + 1], 1);
                    cblas_zhpr2(CblasColMajor, CblasLower, n - k - 1,
                                &NEG_CONE, &AP[kk + 1], 1,
                                &BP[kk + 1], 1, &AP[k1k1]);
                    cblas_zaxpy(n - k - 1, &ct, &BP[kk + 1], 1,
                                &AP[kk + 1], 1);
                    cblas_ztpsv(CblasColMajor, CblasLower, CblasNoTrans,
                                CblasNonUnit, n - k - 1,
                                &BP[k1k1], &AP[kk + 1], 1);
                }
                kk = k1k1;
            }
        }
    } else {
        if (upper) {

            /* Compute U*A*U**H */

            /* k1 and kk are the indices of A(1,k) and A(k,k) */

            kk = -1;
            for (k = 0; k < n; k++) {
                k1 = kk + 1;
                kk = kk + k + 1;

                /* Update the upper triangle of A(1:k,1:k) */

                akk = creal(AP[kk]);
                bkk = creal(BP[kk]);
                if (k > 0) {
                    cblas_ztpmv(CblasColMajor, CblasUpper, CblasNoTrans,
                                CblasNonUnit, k, BP, &AP[k1], 1);
                }
                ct = CMPLX(HALF * akk, 0.0);
                if (k > 0) {
                    cblas_zaxpy(k, &ct, &BP[k1], 1, &AP[k1], 1);
                    cblas_zhpr2(CblasColMajor, CblasUpper, k,
                                &CONE, &AP[k1], 1, &BP[k1], 1, AP);
                    cblas_zaxpy(k, &ct, &BP[k1], 1, &AP[k1], 1);
                    cblas_zdscal(k, bkk, &AP[k1], 1);
                }
                AP[kk] = CMPLX(akk * bkk * bkk, 0.0);
            }
        } else {

            /* Compute L**H *A*L */

            /* jj and j1j1 are the indices of A(j,j) and A(j+1,j+1) */

            jj = 0;
            for (j = 0; j < n; j++) {
                j1j1 = jj + n - j;

                /* Compute the j-th column of the lower triangle of A */

                ajj = creal(AP[jj]);
                bjj = creal(BP[jj]);
                c128 dotc;
                if (j < n - 1) {
                    cblas_zdotc_sub(n - j - 1, &AP[jj + 1], 1,
                                    &BP[jj + 1], 1, &dotc);
                } else {
                    dotc = CMPLX(0.0, 0.0);
                }
                AP[jj] = CMPLX(ajj * bjj, 0.0) + dotc;
                if (j < n - 1) {
                    cblas_zdscal(n - j - 1, bjj, &AP[jj + 1], 1);
                    cblas_zhpmv(CblasColMajor, CblasLower, n - j - 1,
                                &CONE, &AP[j1j1], &BP[jj + 1], 1,
                                &CONE, &AP[jj + 1], 1);
                }
                cblas_ztpmv(CblasColMajor, CblasLower, CblasConjTrans,
                            CblasNonUnit, n - j, &BP[jj], &AP[jj], 1);
                jj = j1j1;
            }
        }
    }
}
