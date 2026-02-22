/**
 * @file zhetri_rook.c
 * @brief ZHETRI_ROOK computes the inverse of a complex Hermitian indefinite
 *        matrix using the factorization computed by ZHETRF_ROOK.
 */

#include "internal_build_defs.h"
#include <math.h>
#include <complex.h>
#include <cblas.h>
#include "semicolon_lapack_complex_double.h"

/**
 * ZHETRI_ROOK computes the inverse of a complex Hermitian indefinite matrix
 * A using the factorization A = U*D*U**H or A = L*D*L**H computed by
 * ZHETRF_ROOK.
 *
 * @param[in]     uplo  = 'U': Upper triangular, A = U*D*U**H
 *                        = 'L': Lower triangular, A = L*D*L**H
 * @param[in]     n     The order of the matrix A. n >= 0.
 * @param[in,out] A     Complex*16 array, dimension (lda, n).
 *                      On entry, the factored matrix from ZHETRF_ROOK.
 *                      On exit, the (Hermitian) inverse of the original matrix.
 * @param[in]     lda   The leading dimension of A. lda >= max(1, n).
 * @param[in]     ipiv  Integer array, dimension (n). The pivot indices
 *                      from ZHETRF_ROOK.
 * @param[out]    work  Complex*16 array, dimension (n).
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 *                         - > 0: if info = i, D(i,i) = 0; the matrix is singular.
 */
void zhetri_rook(
    const char* uplo,
    const INT n,
    c128* restrict A,
    const INT lda,
    const INT* restrict ipiv,
    c128* restrict work,
    INT* info)
{
    const f64 ONE = 1.0;
    const c128 NEG_CONE = CMPLX(-1.0, 0.0);
    const c128 ZERO = CMPLX(0.0, 0.0);

    *info = 0;
    INT upper = (uplo[0] == 'U' || uplo[0] == 'u');
    if (!upper && !(uplo[0] == 'L' || uplo[0] == 'l')) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -4;
    }
    if (*info != 0) {
        xerbla("ZHETRI_ROOK", -(*info));
        return;
    }

    if (n == 0) return;

    /* Check that the diagonal matrix D is nonsingular. */
    if (upper) {
        for (INT k = n - 1; k >= 0; k--) {
            if (ipiv[k] >= 0 && A[k + k * lda] == ZERO) {
                *info = k + 1;
                return;
            }
        }
    } else {
        for (INT k = 0; k < n; k++) {
            if (ipiv[k] >= 0 && A[k + k * lda] == ZERO) {
                *info = k + 1;
                return;
            }
        }
    }

    if (upper) {

        INT k = 0;
        while (k < n) {
            INT kstep;

            if (ipiv[k] >= 0) {

                A[k + k * lda] = CMPLX(ONE / creal(A[k + k * lda]), 0.0);

                if (k > 0) {
                    cblas_zcopy(k, &A[0 + k * lda], 1, work, 1);
                    cblas_zhemv(CblasColMajor, CblasUpper,
                                k, &NEG_CONE, A, lda, work, 1,
                                &ZERO, &A[0 + k * lda], 1);
                    c128 dotc;
                    cblas_zdotc_sub(k, work, 1, &A[0 + k * lda], 1, &dotc);
                    A[k + k * lda] = A[k + k * lda] - CMPLX(creal(dotc), 0.0);
                }
                kstep = 1;

            } else {

                f64 t = cabs(A[k + (k + 1) * lda]);
                f64 ak = creal(A[k + k * lda]) / t;
                f64 akp1 = creal(A[(k + 1) + (k + 1) * lda]) / t;
                c128 akkp1 = A[k + (k + 1) * lda] / t;
                f64 d = t * (ak * akp1 - ONE);
                A[k + k * lda] = CMPLX(akp1 / d, 0.0);
                A[(k + 1) + (k + 1) * lda] = CMPLX(ak / d, 0.0);
                A[k + (k + 1) * lda] = -akkp1 / d;

                if (k > 0) {
                    cblas_zcopy(k, &A[0 + k * lda], 1, work, 1);
                    cblas_zhemv(CblasColMajor, CblasUpper,
                                k, &NEG_CONE, A, lda, work, 1,
                                &ZERO, &A[0 + k * lda], 1);
                    c128 dotc;
                    cblas_zdotc_sub(k, work, 1, &A[0 + k * lda], 1, &dotc);
                    A[k + k * lda] = A[k + k * lda] - CMPLX(creal(dotc), 0.0);
                    cblas_zdotc_sub(k, &A[0 + k * lda], 1, &A[0 + (k + 1) * lda], 1, &dotc);
                    A[k + (k + 1) * lda] = A[k + (k + 1) * lda] - dotc;
                    cblas_zcopy(k, &A[0 + (k + 1) * lda], 1, work, 1);
                    cblas_zhemv(CblasColMajor, CblasUpper,
                                k, &NEG_CONE, A, lda, work, 1,
                                &ZERO, &A[0 + (k + 1) * lda], 1);
                    cblas_zdotc_sub(k, work, 1, &A[0 + (k + 1) * lda], 1, &dotc);
                    A[(k + 1) + (k + 1) * lda] = A[(k + 1) + (k + 1) * lda] -
                                                   CMPLX(creal(dotc), 0.0);
                }
                kstep = 2;
            }

            if (kstep == 1) {

                INT kp = ipiv[k];
                if (kp != k) {

                    if (kp > 0)
                        cblas_zswap(kp, &A[0 + k * lda], 1, &A[0 + kp * lda], 1);

                    for (INT j = kp + 1; j < k; j++) {
                        c128 temp = conj(A[j + k * lda]);
                        A[j + k * lda] = conj(A[kp + j * lda]);
                        A[kp + j * lda] = temp;
                    }

                    A[kp + k * lda] = conj(A[kp + k * lda]);

                    c128 temp = A[k + k * lda];
                    A[k + k * lda] = A[kp + kp * lda];
                    A[kp + kp * lda] = temp;
                }

            } else {

                /* (1) Interchange rows and columns K and -IPIV(K) */
                INT kp = -(ipiv[k] + 1);
                if (kp != k) {

                    if (kp > 0)
                        cblas_zswap(kp, &A[0 + k * lda], 1, &A[0 + kp * lda], 1);

                    for (INT j = kp + 1; j < k; j++) {
                        c128 temp = conj(A[j + k * lda]);
                        A[j + k * lda] = conj(A[kp + j * lda]);
                        A[kp + j * lda] = temp;
                    }

                    A[kp + k * lda] = conj(A[kp + k * lda]);

                    c128 temp = A[k + k * lda];
                    A[k + k * lda] = A[kp + kp * lda];
                    A[kp + kp * lda] = temp;

                    temp = A[k + (k + 1) * lda];
                    A[k + (k + 1) * lda] = A[kp + (k + 1) * lda];
                    A[kp + (k + 1) * lda] = temp;
                }

                /* (2) Interchange rows and columns K+1 and -IPIV(K+1) */
                k = k + 1;
                kp = -(ipiv[k] + 1);
                if (kp != k) {

                    if (kp > 0)
                        cblas_zswap(kp, &A[0 + k * lda], 1, &A[0 + kp * lda], 1);

                    for (INT j = kp + 1; j < k; j++) {
                        c128 temp = conj(A[j + k * lda]);
                        A[j + k * lda] = conj(A[kp + j * lda]);
                        A[kp + j * lda] = temp;
                    }

                    A[kp + k * lda] = conj(A[kp + k * lda]);

                    c128 temp = A[k + k * lda];
                    A[k + k * lda] = A[kp + kp * lda];
                    A[kp + kp * lda] = temp;
                }
            }

            k = k + 1;
        }

    } else {

        INT k = n - 1;
        while (k >= 0) {
            INT kstep;

            if (ipiv[k] >= 0) {

                A[k + k * lda] = CMPLX(ONE / creal(A[k + k * lda]), 0.0);

                if (k < n - 1) {
                    cblas_zcopy(n - k - 1, &A[(k + 1) + k * lda], 1, work, 1);
                    cblas_zhemv(CblasColMajor, CblasLower,
                                n - k - 1, &NEG_CONE, &A[(k + 1) + (k + 1) * lda], lda,
                                work, 1, &ZERO, &A[(k + 1) + k * lda], 1);
                    c128 dotc;
                    cblas_zdotc_sub(n - k - 1, work, 1, &A[(k + 1) + k * lda], 1, &dotc);
                    A[k + k * lda] = A[k + k * lda] - CMPLX(creal(dotc), 0.0);
                }
                kstep = 1;

            } else {

                f64 t = cabs(A[k + (k - 1) * lda]);
                f64 ak = creal(A[(k - 1) + (k - 1) * lda]) / t;
                f64 akp1 = creal(A[k + k * lda]) / t;
                c128 akkp1 = A[k + (k - 1) * lda] / t;
                f64 d = t * (ak * akp1 - ONE);
                A[(k - 1) + (k - 1) * lda] = CMPLX(akp1 / d, 0.0);
                A[k + k * lda] = CMPLX(ak / d, 0.0);
                A[k + (k - 1) * lda] = -akkp1 / d;

                if (k < n - 1) {
                    cblas_zcopy(n - k - 1, &A[(k + 1) + k * lda], 1, work, 1);
                    cblas_zhemv(CblasColMajor, CblasLower,
                                n - k - 1, &NEG_CONE, &A[(k + 1) + (k + 1) * lda], lda,
                                work, 1, &ZERO, &A[(k + 1) + k * lda], 1);
                    c128 dotc;
                    cblas_zdotc_sub(n - k - 1, work, 1, &A[(k + 1) + k * lda], 1, &dotc);
                    A[k + k * lda] = A[k + k * lda] - CMPLX(creal(dotc), 0.0);
                    cblas_zdotc_sub(n - k - 1, &A[(k + 1) + k * lda], 1,
                                    &A[(k + 1) + (k - 1) * lda], 1, &dotc);
                    A[k + (k - 1) * lda] = A[k + (k - 1) * lda] - dotc;
                    cblas_zcopy(n - k - 1, &A[(k + 1) + (k - 1) * lda], 1, work, 1);
                    cblas_zhemv(CblasColMajor, CblasLower,
                                n - k - 1, &NEG_CONE, &A[(k + 1) + (k + 1) * lda], lda,
                                work, 1, &ZERO, &A[(k + 1) + (k - 1) * lda], 1);
                    cblas_zdotc_sub(n - k - 1, work, 1,
                                    &A[(k + 1) + (k - 1) * lda], 1, &dotc);
                    A[(k - 1) + (k - 1) * lda] = A[(k - 1) + (k - 1) * lda] -
                                                   CMPLX(creal(dotc), 0.0);
                }
                kstep = 2;
            }

            if (kstep == 1) {

                INT kp = ipiv[k];
                if (kp != k) {

                    if (kp < n - 1)
                        cblas_zswap(n - kp - 1, &A[(kp + 1) + k * lda], 1,
                                    &A[(kp + 1) + kp * lda], 1);

                    for (INT j = k + 1; j < kp; j++) {
                        c128 temp = conj(A[j + k * lda]);
                        A[j + k * lda] = conj(A[kp + j * lda]);
                        A[kp + j * lda] = temp;
                    }

                    A[kp + k * lda] = conj(A[kp + k * lda]);

                    c128 temp = A[k + k * lda];
                    A[k + k * lda] = A[kp + kp * lda];
                    A[kp + kp * lda] = temp;
                }

            } else {

                /* (1) Interchange rows and columns K and -IPIV(K) */
                INT kp = -(ipiv[k] + 1);
                if (kp != k) {

                    if (kp < n - 1)
                        cblas_zswap(n - kp - 1, &A[(kp + 1) + k * lda], 1,
                                    &A[(kp + 1) + kp * lda], 1);

                    for (INT j = k + 1; j < kp; j++) {
                        c128 temp = conj(A[j + k * lda]);
                        A[j + k * lda] = conj(A[kp + j * lda]);
                        A[kp + j * lda] = temp;
                    }

                    A[kp + k * lda] = conj(A[kp + k * lda]);

                    c128 temp = A[k + k * lda];
                    A[k + k * lda] = A[kp + kp * lda];
                    A[kp + kp * lda] = temp;

                    temp = A[k + (k - 1) * lda];
                    A[k + (k - 1) * lda] = A[kp + (k - 1) * lda];
                    A[kp + (k - 1) * lda] = temp;
                }

                /* (2) Interchange rows and columns K-1 and -IPIV(K-1) */
                k = k - 1;
                kp = -(ipiv[k] + 1);
                if (kp != k) {

                    if (kp < n - 1)
                        cblas_zswap(n - kp - 1, &A[(kp + 1) + k * lda], 1,
                                    &A[(kp + 1) + kp * lda], 1);

                    for (INT j = k + 1; j < kp; j++) {
                        c128 temp = conj(A[j + k * lda]);
                        A[j + k * lda] = conj(A[kp + j * lda]);
                        A[kp + j * lda] = temp;
                    }

                    A[kp + k * lda] = conj(A[kp + k * lda]);

                    c128 temp = A[k + k * lda];
                    A[k + k * lda] = A[kp + kp * lda];
                    A[kp + kp * lda] = temp;
                }
            }

            k = k - 1;
        }
    }
}
