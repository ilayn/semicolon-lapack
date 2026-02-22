/**
 * @file zhetri2x.c
 * @brief ZHETRI2X computes the inverse of a complex hermitian indefinite matrix
 *        using the factorization computed by ZHETRF.
 */

#include <complex.h>
#include "semicolon_cblas.h"
#include "semicolon_lapack_complex_double.h"

/**
 * ZHETRI2X computes the inverse of a complex hermitian indefinite matrix
 * A using the factorization A = U*D*U**H or A = L*D*L**H computed by
 * ZHETRF.
 *
 * @param[in]     uplo  Specifies whether the details of the factorization
 *                      are stored as an upper or lower triangular matrix.
 *                      = 'U': Upper triangular, form is A = U*D*U**H;
 *                      = 'L': Lower triangular, form is A = L*D*L**H.
 * @param[in]     n     The order of the matrix A. n >= 0.
 * @param[in,out] A     Double complex array, dimension (lda, n).
 *                      On entry, the block diagonal matrix D and the multipliers
 *                      used to obtain the factor U or L as computed by ZHETRF.
 *                      On exit, if info = 0, the (symmetric) inverse of the
 *                      original matrix.
 * @param[in]     lda   The leading dimension of the array A. lda >= max(1,n).
 * @param[in]     ipiv  Integer array, dimension (n).
 *                      Details of the interchanges and the block structure of D
 *                      as determined by ZHETRF.
 * @param[out]    work  Double complex array, dimension (n+nb+1, nb+3).
 * @param[in]     nb    Block size.
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 *                         - > 0: if info = i, D(i,i) = 0; the matrix is singular.
 */
void zhetri2x(
    const char* uplo,
    const INT n,
    c128* restrict A,
    const INT lda,
    const INT* restrict ipiv,
    c128* restrict work,
    const INT nb,
    INT* info)
{
    const c128 ONE = CMPLX(1.0, 0.0);
    const c128 ZERO = CMPLX(0.0, 0.0);

    INT upper;
    INT i, iinfo, ip, k, cut, nnb;
    INT count;
    INT j, u11, invd;
    c128 ak, akkp1, akp1, d, t;
    c128 u01_i_j, u01_ip1_j;
    c128 u11_i_j, u11_ip1_j;
    const INT ldw = n + nb + 1;

    *info = 0;
    upper = (uplo[0] == 'U' || uplo[0] == 'u');
    if (!upper && !(uplo[0] == 'L' || uplo[0] == 'l')) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (lda < (1 > n ? 1 : n)) {
        *info = -4;
    }

    if (*info != 0) {
        xerbla("ZHETRI2X", -(*info));
        return;
    }
    if (n == 0) {
        return;
    }

    zsyconv(uplo, "C", n, A, lda, ipiv, work, &iinfo);

    if (upper) {

        for (*info = n - 1; *info >= 0; (*info)--) {
            if (ipiv[*info] >= 0 && A[*info + (*info) * lda] == ZERO) {
                *info = *info + 1;
                return;
            }
        }
    } else {

        for (*info = 0; *info < n; (*info)++) {
            if (ipiv[*info] >= 0 && A[*info + (*info) * lda] == ZERO) {
                *info = *info + 1;
                return;
            }
        }
    }
    *info = 0;

    u11 = n;
    invd = nb + 1;

    if (upper) {

        ztrtri(uplo, "U", n, A, lda, info);
        if (*info != 0) {
            return;
        }

        k = 0;
        while (k < n) {
            if (ipiv[k] >= 0) {
                work[k + invd * ldw] = CMPLX(1.0 / creal(A[k + k * lda]), 0.0);
                work[k + (invd + 1) * ldw] = ZERO;
                k = k + 1;
            } else {
                t = CMPLX(cabs(work[(k + 1) + 0 * ldw]), 0.0);
                ak = CMPLX(creal(A[k + k * lda]), 0.0) / t;
                akp1 = CMPLX(creal(A[(k + 1) + (k + 1) * lda]), 0.0) / t;
                akkp1 = work[(k + 1) + 0 * ldw] / t;
                d = t * (ak * akp1 - ONE);
                work[k + invd * ldw] = akp1 / d;
                work[(k + 1) + (invd + 1) * ldw] = ak / d;
                work[k + (invd + 1) * ldw] = -akkp1 / d;
                work[(k + 1) + invd * ldw] = conj(work[k + (invd + 1) * ldw]);
                k = k + 2;
            }
        }

        cut = n;
        while (cut > 0) {
            nnb = nb;
            if (cut <= nnb) {
                nnb = cut;
            } else {
                count = 0;
                for (i = cut - nnb; i < cut; i++) {
                    if (ipiv[i] < 0) count = count + 1;
                }
                if (count % 2 == 1) nnb = nnb + 1;
            }

            cut = cut - nnb;

            for (i = 0; i < cut; i++) {
                for (j = 0; j < nnb; j++) {
                    work[i + j * ldw] = A[i + (cut + j) * lda];
                }
            }

            for (i = 0; i < nnb; i++) {
                work[(u11 + i) + i * ldw] = ONE;
                for (j = 0; j < i; j++) {
                    work[(u11 + i) + j * ldw] = ZERO;
                }
                for (j = i + 1; j < nnb; j++) {
                    work[(u11 + i) + j * ldw] = A[(cut + i) + (cut + j) * lda];
                }
            }

            i = 0;
            while (i < cut) {
                if (ipiv[i] >= 0) {
                    for (j = 0; j < nnb; j++) {
                        work[i + j * ldw] = work[i + invd * ldw] * work[i + j * ldw];
                    }
                    i = i + 1;
                } else {
                    for (j = 0; j < nnb; j++) {
                        u01_i_j = work[i + j * ldw];
                        u01_ip1_j = work[(i + 1) + j * ldw];
                        work[i + j * ldw] = work[i + invd * ldw] * u01_i_j +
                                            work[i + (invd + 1) * ldw] * u01_ip1_j;
                        work[(i + 1) + j * ldw] = work[(i + 1) + invd * ldw] * u01_i_j +
                                                  work[(i + 1) + (invd + 1) * ldw] * u01_ip1_j;
                    }
                    i = i + 2;
                }
            }

            i = 0;
            while (i < nnb) {
                if (ipiv[cut + i] >= 0) {
                    for (j = i; j < nnb; j++) {
                        work[(u11 + i) + j * ldw] = work[(cut + i) + invd * ldw] *
                                                    work[(u11 + i) + j * ldw];
                    }
                    i = i + 1;
                } else {
                    for (j = i; j < nnb; j++) {
                        u11_i_j = work[(u11 + i) + j * ldw];
                        u11_ip1_j = work[(u11 + i + 1) + j * ldw];
                        work[(u11 + i) + j * ldw] = work[(cut + i) + invd * ldw] *
                                                     work[(u11 + i) + j * ldw] +
                                                     work[(cut + i) + (invd + 1) * ldw] *
                                                     work[(u11 + i + 1) + j * ldw];
                        work[(u11 + i + 1) + j * ldw] = work[(cut + i + 1) + invd * ldw] * u11_i_j +
                                                         work[(cut + i + 1) + (invd + 1) * ldw] * u11_ip1_j;
                    }
                    i = i + 2;
                }
            }

            cblas_ztrmm(CblasColMajor, CblasLeft, CblasUpper, CblasConjTrans,
                        CblasUnit, nnb, nnb, &ONE, &A[cut + cut * lda], lda,
                        &work[u11 + 0 * ldw], ldw);

            for (i = 0; i < nnb; i++) {
                for (j = i; j < nnb; j++) {
                    A[(cut + i) + (cut + j) * lda] = work[(u11 + i) + j * ldw];
                }
            }

            if (cut > 0) {
                cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans,
                            nnb, nnb, cut, &ONE, &A[0 + cut * lda], lda,
                            &work[0 + 0 * ldw], ldw, &ZERO, &work[u11 + 0 * ldw], ldw);

                for (i = 0; i < nnb; i++) {
                    for (j = i; j < nnb; j++) {
                        A[(cut + i) + (cut + j) * lda] += work[(u11 + i) + j * ldw];
                    }
                }

                cblas_ztrmm(CblasColMajor, CblasLeft, CblasUpper, CblasConjTrans,
                            CblasUnit, cut, nnb, &ONE, A, lda, &work[0 + 0 * ldw], ldw);

                for (i = 0; i < cut; i++) {
                    for (j = 0; j < nnb; j++) {
                        A[i + (cut + j) * lda] = work[i + j * ldw];
                    }
                }
            }

        }

        i = 0;
        while (i < n) {
            if (ipiv[i] >= 0) {
                ip = ipiv[i];
                if (i < ip) zheswapr(uplo, n, A, lda, i, ip);
                if (i > ip) zheswapr(uplo, n, A, lda, ip, i);
            } else {
                ip = -ipiv[i] - 1;
                i = i + 1;
                if ((i - 1) < ip) zheswapr(uplo, n, A, lda, i - 1, ip);
                if ((i - 1) > ip) zheswapr(uplo, n, A, lda, ip, i - 1);
            }
            i = i + 1;
        }

    } else {

        ztrtri(uplo, "U", n, A, lda, info);
        if (*info != 0) {
            return;
        }

        k = n - 1;
        while (k >= 0) {
            if (ipiv[k] >= 0) {
                work[k + invd * ldw] = CMPLX(1.0 / creal(A[k + k * lda]), 0.0);
                work[k + (invd + 1) * ldw] = ZERO;
                k = k - 1;
            } else {
                t = CMPLX(cabs(work[(k - 1) + 0 * ldw]), 0.0);
                ak = CMPLX(creal(A[(k - 1) + (k - 1) * lda]), 0.0) / t;
                akp1 = CMPLX(creal(A[k + k * lda]), 0.0) / t;
                akkp1 = work[(k - 1) + 0 * ldw] / t;
                d = t * (ak * akp1 - ONE);
                work[(k - 1) + invd * ldw] = akp1 / d;
                work[k + invd * ldw] = ak / d;
                work[k + (invd + 1) * ldw] = -akkp1 / d;
                work[(k - 1) + (invd + 1) * ldw] = conj(work[k + (invd + 1) * ldw]);
                k = k - 2;
            }
        }

        cut = 0;
        while (cut < n) {
            nnb = nb;
            if (cut + nnb > n) {
                nnb = n - cut;
            } else {
                count = 0;
                for (i = cut; i < cut + nnb; i++) {
                    if (ipiv[i] < 0) count = count + 1;
                }
                if (count % 2 == 1) nnb = nnb + 1;
            }

            for (i = 0; i < n - cut - nnb; i++) {
                for (j = 0; j < nnb; j++) {
                    work[i + j * ldw] = A[(cut + nnb + i) + (cut + j) * lda];
                }
            }

            for (i = 0; i < nnb; i++) {
                work[(u11 + i) + i * ldw] = ONE;
                for (j = i + 1; j < nnb; j++) {
                    work[(u11 + i) + j * ldw] = ZERO;
                }
                for (j = 0; j < i; j++) {
                    work[(u11 + i) + j * ldw] = A[(cut + i) + (cut + j) * lda];
                }
            }

            i = n - cut - nnb - 1;
            while (i >= 0) {
                if (ipiv[cut + nnb + i] >= 0) {
                    for (j = 0; j < nnb; j++) {
                        work[i + j * ldw] = work[(cut + nnb + i) + invd * ldw] *
                                            work[i + j * ldw];
                    }
                    i = i - 1;
                } else {
                    for (j = 0; j < nnb; j++) {
                        u01_i_j = work[i + j * ldw];
                        u01_ip1_j = work[(i - 1) + j * ldw];
                        work[i + j * ldw] = work[(cut + nnb + i) + invd * ldw] * u01_i_j +
                                            work[(cut + nnb + i) + (invd + 1) * ldw] * u01_ip1_j;
                        work[(i - 1) + j * ldw] = work[(cut + nnb + i - 1) + (invd + 1) * ldw] * u01_i_j +
                                                  work[(cut + nnb + i - 1) + invd * ldw] * u01_ip1_j;
                    }
                    i = i - 2;
                }
            }

            i = nnb - 1;
            while (i >= 0) {
                if (ipiv[cut + i] >= 0) {
                    for (j = 0; j < nnb; j++) {
                        work[(u11 + i) + j * ldw] = work[(cut + i) + invd * ldw] *
                                                    work[(u11 + i) + j * ldw];
                    }
                    i = i - 1;
                } else {
                    for (j = 0; j < nnb; j++) {
                        u11_i_j = work[(u11 + i) + j * ldw];
                        u11_ip1_j = work[(u11 + i - 1) + j * ldw];
                        work[(u11 + i) + j * ldw] = work[(cut + i) + invd * ldw] *
                                                     work[(u11 + i) + j * ldw] +
                                                     work[(cut + i) + (invd + 1) * ldw] * u11_ip1_j;
                        work[(u11 + i - 1) + j * ldw] = work[(cut + i - 1) + (invd + 1) * ldw] * u11_i_j +
                                                         work[(cut + i - 1) + invd * ldw] * u11_ip1_j;
                    }
                    i = i - 2;
                }
            }

            cblas_ztrmm(CblasColMajor, CblasLeft, CblasLower, CblasConjTrans,
                        CblasUnit, nnb, nnb, &ONE, &A[cut + cut * lda], lda,
                        &work[u11 + 0 * ldw], ldw);

            for (i = 0; i < nnb; i++) {
                for (j = 0; j <= i; j++) {
                    A[(cut + i) + (cut + j) * lda] = work[(u11 + i) + j * ldw];
                }
            }

            if ((cut + nnb) < n) {

                cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans,
                            nnb, nnb, n - nnb - cut, &ONE, &A[(cut + nnb) + cut * lda], lda,
                            &work[0 + 0 * ldw], ldw, &ZERO, &work[u11 + 0 * ldw], ldw);

                for (i = 0; i < nnb; i++) {
                    for (j = 0; j <= i; j++) {
                        A[(cut + i) + (cut + j) * lda] += work[(u11 + i) + j * ldw];
                    }
                }

                cblas_ztrmm(CblasColMajor, CblasLeft, CblasLower, CblasConjTrans,
                            CblasUnit, n - nnb - cut, nnb, &ONE,
                            &A[(cut + nnb) + (cut + nnb) * lda], lda, &work[0 + 0 * ldw], ldw);

                for (i = 0; i < n - cut - nnb; i++) {
                    for (j = 0; j < nnb; j++) {
                        A[(cut + nnb + i) + (cut + j) * lda] = work[i + j * ldw];
                    }
                }

            }

            cut = cut + nnb;
        }

        i = n - 1;
        while (i >= 0) {
            if (ipiv[i] >= 0) {
                ip = ipiv[i];
                if (i < ip) zheswapr(uplo, n, A, lda, i, ip);
                if (i > ip) zheswapr(uplo, n, A, lda, ip, i);
            } else {
                ip = -ipiv[i] - 1;
                if (i < ip) zheswapr(uplo, n, A, lda, i, ip);
                if (i > ip) zheswapr(uplo, n, A, lda, ip, i);
                i = i - 1;
            }
            i = i - 1;
        }

    }
}
