/**
 * @file zlavhp.c
 * @brief ZLAVHP performs one of the matrix-vector operations
 *        x := A*x or x := A^H*x, where A is a factor from the
 *        Hermitian factorization computed by ZHPTRF stored in packed format.
 */

#include "semicolon_cblas.h"
#include "verify.h"

void zlavhp(
    const char* uplo,
    const char* trans,
    const char* diag,
    const INT n,
    const INT nrhs,
    const c128* A,
    const INT* ipiv,
    c128* B,
    const INT ldb,
    INT* info)
{
    const c128 CONE = CMPLX(1.0, 0.0);

    INT nounit;
    INT j, k, kc, kcnext, kp;
    c128 d11, d12, d21, d22, t1, t2;

    *info = 0;
    if (!(uplo[0] == 'U' || uplo[0] == 'u') && !(uplo[0] == 'L' || uplo[0] == 'l')) {
        *info = -1;
    } else if (!(trans[0] == 'N' || trans[0] == 'n') &&
               !(trans[0] == 'C' || trans[0] == 'c')) {
        *info = -2;
    } else if (!(diag[0] == 'U' || diag[0] == 'u') && !(diag[0] == 'N' || diag[0] == 'n')) {
        *info = -3;
    } else if (n < 0) {
        *info = -4;
    } else if (ldb < (n > 1 ? n : 1)) {
        *info = -9;
    }
    if (*info != 0) {
        xerbla("ZLAVHP", -(*info));
        return;
    }

    if (n == 0)
        return;

    nounit = (diag[0] == 'N' || diag[0] == 'n');

    /*------------------------------------------
     * Compute  B := A * B  (No transpose)
     *------------------------------------------*/
    if (trans[0] == 'N' || trans[0] == 'n') {

        if (uplo[0] == 'U' || uplo[0] == 'u') {
            /* Compute B := U*B
             * where U = P(m)*inv(U(m))* ... *P(1)*inv(U(1))
             * Loop forward applying the transformations. */
            k = 0;
            kc = 0;
            while (k < n) {
                if (ipiv[k] >= 0) {
                    /* 1 x 1 pivot block.
                     * Multiply by the diagonal element if forming U * D. */
                    if (nounit)
                        cblas_zscal(nrhs, &A[kc + k], &B[k], ldb);

                    /* Multiply by P(K) * inv(U(K)) if K > 0. */
                    if (k > 0) {
                        /* Apply the transformation. */
                        cblas_zgeru(CblasColMajor, k, nrhs, &CONE,
                                    &A[kc], 1, &B[k], ldb, &B[0], ldb);

                        /* Interchange if P(K) != I. */
                        kp = ipiv[k];
                        if (kp != k)
                            cblas_zswap(nrhs, &B[k], ldb, &B[kp], ldb);
                    }
                    kc = kc + k + 1;
                    k++;
                } else {
                    /* 2 x 2 pivot block. */
                    kcnext = kc + k + 1;

                    /* Multiply by the diagonal block if forming U * D. */
                    if (nounit) {
                        d11 = A[kcnext - 1];
                        d22 = A[kcnext + k + 1];
                        d12 = A[kcnext + k];
                        d21 = conj(d12);
                        for (j = 0; j < nrhs; j++) {
                            t1 = B[k + j * ldb];
                            t2 = B[(k + 1) + j * ldb];
                            B[k + j * ldb] = d11 * t1 + d12 * t2;
                            B[(k + 1) + j * ldb] = d21 * t1 + d22 * t2;
                        }
                    }

                    /* Multiply by P(K) * inv(U(K)) if K > 0. */
                    if (k > 0) {
                        /* Apply the transformations. */
                        cblas_zgeru(CblasColMajor, k, nrhs, &CONE,
                                    &A[kc], 1, &B[k], ldb, &B[0], ldb);
                        cblas_zgeru(CblasColMajor, k, nrhs, &CONE,
                                    &A[kcnext], 1, &B[k + 1], ldb, &B[0], ldb);

                        /* Interchange if P(K) != I. */
                        kp = (ipiv[k] < 0) ? -(ipiv[k] + 1) : ipiv[k];
                        if (kp != k)
                            cblas_zswap(nrhs, &B[k], ldb, &B[kp], ldb);
                    }
                    kc = kcnext + k + 2;
                    k += 2;
                }
            }

        } else {
            /* Compute B := L*B
             * where L = P(1)*inv(L(1))* ... *P(m)*inv(L(m))
             * Loop backward applying the transformations to B. */
            k = n - 1;
            kc = n * (n + 1) / 2;
            while (k >= 0) {
                kc = kc - (n - k);

                if (ipiv[k] >= 0) {
                    /* 1 x 1 pivot block.
                     * Multiply by the diagonal element if forming L * D. */
                    if (nounit)
                        cblas_zscal(nrhs, &A[kc], &B[k], ldb);

                    /* Multiply by P(K) * inv(L(K)) if K < N-1. */
                    if (k < n - 1) {
                        kp = ipiv[k];

                        /* Apply the transformation. */
                        cblas_zgeru(CblasColMajor, n - k - 1, nrhs, &CONE,
                                    &A[kc + 1], 1, &B[k], ldb, &B[k + 1], ldb);

                        /* Interchange if a permutation was applied. */
                        if (kp != k)
                            cblas_zswap(nrhs, &B[k], ldb, &B[kp], ldb);
                    }
                    k--;
                } else {
                    /* 2 x 2 pivot block. */
                    kcnext = kc - (n - k + 1);

                    /* Multiply by the diagonal block if forming L * D. */
                    if (nounit) {
                        d11 = A[kcnext];
                        d22 = A[kc];
                        d21 = A[kcnext + 1];
                        d12 = conj(d21);
                        for (j = 0; j < nrhs; j++) {
                            t1 = B[(k - 1) + j * ldb];
                            t2 = B[k + j * ldb];
                            B[(k - 1) + j * ldb] = d11 * t1 + d12 * t2;
                            B[k + j * ldb] = d21 * t1 + d22 * t2;
                        }
                    }

                    /* Multiply by P(K) * inv(L(K)) if K < N-1. */
                    if (k < n - 1) {
                        /* Apply the transformation. */
                        cblas_zgeru(CblasColMajor, n - k - 1, nrhs, &CONE,
                                    &A[kc + 1], 1, &B[k], ldb, &B[k + 1], ldb);
                        cblas_zgeru(CblasColMajor, n - k - 1, nrhs, &CONE,
                                    &A[kcnext + 2], 1, &B[k - 1], ldb,
                                    &B[k + 1], ldb);

                        /* Interchange if a permutation was applied. */
                        kp = (ipiv[k] < 0) ? -(ipiv[k] + 1) : ipiv[k];
                        if (kp != k)
                            cblas_zswap(nrhs, &B[k], ldb, &B[kp], ldb);
                    }
                    kc = kcnext;
                    k -= 2;
                }
            }
        }

    /*-------------------------------------------------
     * Compute  B := A^H * B  (conjugate transpose)
     *-------------------------------------------------*/
    } else {

        if (uplo[0] == 'U' || uplo[0] == 'u') {
            /* Form B := U^H*B
             * where U  = P(m)*inv(U(m))* ... *P(1)*inv(U(1))
             * and   U^H = inv(U^H(1))*P(1)* ... *inv(U^H(m))*P(m)
             * Loop backward applying the transformations. */
            k = n - 1;
            kc = n * (n + 1) / 2;
            while (k >= 0) {
                kc = kc - (k + 1);

                if (ipiv[k] >= 0) {
                    /* 1 x 1 pivot block. */
                    if (k > 0) {
                        /* Interchange if P(K) != I. */
                        kp = ipiv[k];
                        if (kp != k)
                            cblas_zswap(nrhs, &B[k], ldb, &B[kp], ldb);

                        /* Apply the transformation. */
                        zlacgv(nrhs, &B[k], ldb);
                        cblas_zgemv(CblasColMajor, CblasConjTrans,
                                    k, nrhs, &CONE, &B[0], ldb,
                                    &A[kc], 1, &CONE, &B[k], ldb);
                        zlacgv(nrhs, &B[k], ldb);
                    }
                    if (nounit)
                        cblas_zscal(nrhs, &A[kc + k], &B[k], ldb);
                    k--;
                } else {
                    /* 2 x 2 pivot block. */
                    kcnext = kc - k;
                    if (k > 1) {
                        /* Interchange if P(K) != I. */
                        kp = (ipiv[k] < 0) ? -(ipiv[k] + 1) : ipiv[k];
                        if (kp != k - 1)
                            cblas_zswap(nrhs, &B[k - 1], ldb, &B[kp], ldb);

                        /* Apply the transformations. */
                        zlacgv(nrhs, &B[k], ldb);
                        cblas_zgemv(CblasColMajor, CblasConjTrans,
                                    k - 1, nrhs, &CONE, &B[0], ldb,
                                    &A[kc], 1, &CONE, &B[k], ldb);
                        zlacgv(nrhs, &B[k], ldb);

                        zlacgv(nrhs, &B[k - 1], ldb);
                        cblas_zgemv(CblasColMajor, CblasConjTrans,
                                    k - 1, nrhs, &CONE, &B[0], ldb,
                                    &A[kcnext], 1, &CONE, &B[k - 1], ldb);
                        zlacgv(nrhs, &B[k - 1], ldb);
                    }

                    /* Multiply by the diagonal block if non-unit. */
                    if (nounit) {
                        d11 = A[kc - 1];
                        d22 = A[kc + k];
                        d12 = A[kc + k - 1];
                        d21 = conj(d12);
                        for (j = 0; j < nrhs; j++) {
                            t1 = B[(k - 1) + j * ldb];
                            t2 = B[k + j * ldb];
                            B[(k - 1) + j * ldb] = d11 * t1 + d12 * t2;
                            B[k + j * ldb] = d21 * t1 + d22 * t2;
                        }
                    }
                    kc = kcnext;
                    k -= 2;
                }
            }

        } else {
            /* Form B := L^H*B
             * where L  = P(1)*inv(L(1))* ... *P(m)*inv(L(m))
             * and   L^H = inv(L(m))*P(m)* ... *inv(L(1))*P(1)
             * Loop forward applying the L-transformations. */
            k = 0;
            kc = 0;
            while (k < n) {
                if (ipiv[k] >= 0) {
                    /* 1 x 1 pivot block. */
                    if (k < n - 1) {
                        /* Interchange if P(K) != I. */
                        kp = ipiv[k];
                        if (kp != k)
                            cblas_zswap(nrhs, &B[k], ldb, &B[kp], ldb);

                        /* Apply the transformation. */
                        zlacgv(nrhs, &B[k], ldb);
                        cblas_zgemv(CblasColMajor, CblasConjTrans,
                                    n - k - 1, nrhs, &CONE, &B[k + 1], ldb,
                                    &A[kc + 1], 1, &CONE, &B[k], ldb);
                        zlacgv(nrhs, &B[k], ldb);
                    }
                    if (nounit)
                        cblas_zscal(nrhs, &A[kc], &B[k], ldb);
                    kc = kc + n - k;
                    k++;
                } else {
                    /* 2 x 2 pivot block. */
                    kcnext = kc + n - k;
                    if (k < n - 2) {
                        /* Interchange if P(K) != I. */
                        kp = (ipiv[k] < 0) ? -(ipiv[k] + 1) : ipiv[k];
                        if (kp != k + 1)
                            cblas_zswap(nrhs, &B[k + 1], ldb, &B[kp], ldb);

                        /* Apply the transformation. */
                        zlacgv(nrhs, &B[k + 1], ldb);
                        cblas_zgemv(CblasColMajor, CblasConjTrans,
                                    n - k - 2, nrhs, &CONE, &B[k + 2], ldb,
                                    &A[kcnext + 1], 1, &CONE, &B[k + 1], ldb);
                        zlacgv(nrhs, &B[k + 1], ldb);

                        zlacgv(nrhs, &B[k], ldb);
                        cblas_zgemv(CblasColMajor, CblasConjTrans,
                                    n - k - 2, nrhs, &CONE, &B[k + 2], ldb,
                                    &A[kc + 2], 1, &CONE, &B[k], ldb);
                        zlacgv(nrhs, &B[k], ldb);
                    }

                    /* Multiply by the diagonal block if non-unit. */
                    if (nounit) {
                        d11 = A[kc];
                        d22 = A[kcnext];
                        d21 = A[kc + 1];
                        d12 = conj(d21);
                        for (j = 0; j < nrhs; j++) {
                            t1 = B[k + j * ldb];
                            t2 = B[(k + 1) + j * ldb];
                            B[k + j * ldb] = d11 * t1 + d12 * t2;
                            B[(k + 1) + j * ldb] = d21 * t1 + d22 * t2;
                        }
                    }
                    kc = kcnext + (n - k - 1);
                    k += 2;
                }
            }
        }
    }
}
