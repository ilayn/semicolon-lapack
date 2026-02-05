/**
 * @file dlavsp.c
 * @brief DLAVSP performs one of the matrix-vector operations x := A*x or
 *        x := A'*x, where A is a factor from the block U*D*U' or L*D*L'
 *        factorization computed by DSPTRF.
 */

#include <math.h>
#include <cblas.h>
#include "verify.h"

void xerbla(const char* srname, const int info);

/**
 * DLAVSP performs one of the matrix-vector operations
 *    x := A*x  or  x := A'*x,
 * where x is an N element vector and A is one of the factors
 * from the block U*D*U' or L*D*L' factorization computed by DSPTRF.
 *
 * If TRANS = 'N', multiplies by U  or U * D  (or L  or L * D)
 * If TRANS = 'T', multiplies by U' or D * U' (or L' or D * L')
 * If TRANS = 'C', multiplies by U' or D * U' (or L' or D * L')
 *
 * @param[in]     uplo   'U': Upper triangular, 'L': Lower triangular
 * @param[in]     trans  'N': No transpose, 'T'/'C': Transpose
 * @param[in]     diag   'U': Unit diagonal (A = U or L only),
 *                       'N': Non-unit (A = U*D or A = L*D)
 * @param[in]     n      The number of rows and columns of the matrix A. n >= 0.
 * @param[in]     nrhs   The number of right hand sides. nrhs >= 0.
 * @param[in]     A      The block diagonal matrix D and the multipliers used to
 *                       obtain the factor U or L, stored as a packed triangular
 *                       matrix as computed by DSPTRF.
 *                       Double precision array, dimension (n*(n+1)/2).
 * @param[in]     ipiv   Pivot indices from DSPTRF. Integer array, dimension (n).
 *                       0-based indexing.
 * @param[in,out] B      On entry, B contains NRHS vectors of length N.
 *                       On exit, B is overwritten with the product A * B.
 *                       Double precision array, dimension (ldb, nrhs).
 * @param[in]     ldb    The leading dimension of the array B. ldb >= max(1,n).
 * @param[out]    info   = 0: successful exit
 *                       < 0: if info = -k, the k-th argument had an illegal value
 */
void dlavsp(
    const char* uplo,
    const char* trans,
    const char* diag,
    const int n,
    const int nrhs,
    const double* const restrict A,
    const int* const restrict ipiv,
    double* const restrict B,
    const int ldb,
    int* info)
{
    const double ONE = 1.0;

    int nounit;
    int j, k, kc, kcnext, kp;
    double d11, d12, d21, d22, t1, t2;

    *info = 0;
    if (!(uplo[0] == 'U' || uplo[0] == 'u') && !(uplo[0] == 'L' || uplo[0] == 'l')) {
        *info = -1;
    } else if (!(trans[0] == 'N' || trans[0] == 'n') &&
               !(trans[0] == 'T' || trans[0] == 't') &&
               !(trans[0] == 'C' || trans[0] == 'c')) {
        *info = -2;
    } else if (!(diag[0] == 'U' || diag[0] == 'u') && !(diag[0] == 'N' || diag[0] == 'n')) {
        *info = -3;
    } else if (n < 0) {
        *info = -4;
    } else if (ldb < (n > 1 ? n : 1)) {
        *info = -8;
    }
    if (*info != 0) {
        xerbla("DLAVSP", -(*info));
        return;
    }

    if (n == 0)
        return;

    nounit = (diag[0] == 'N' || diag[0] == 'n');

    if (trans[0] == 'N' || trans[0] == 'n') {

        if (uplo[0] == 'U' || uplo[0] == 'u') {

            k = 0;
            kc = 0;
            while (k < n) {

                if (ipiv[k] >= 0) {

                    if (nounit)
                        cblas_dscal(nrhs, A[kc + k], &B[k], ldb);

                    if (k > 0) {

                        cblas_dger(CblasColMajor, k, nrhs, ONE, &A[kc], 1,
                                   &B[k], ldb, &B[0], ldb);

                        kp = ipiv[k];
                        if (kp != k)
                            cblas_dswap(nrhs, &B[k], ldb, &B[kp], ldb);
                    }
                    kc = kc + k + 1;
                    k = k + 1;
                } else {

                    kcnext = kc + k + 1;

                    if (nounit) {
                        d11 = A[kcnext - 1];
                        d22 = A[kcnext + k + 1];
                        d12 = A[kcnext + k];
                        d21 = d12;
                        for (j = 0; j < nrhs; j++) {
                            t1 = B[k + j * ldb];
                            t2 = B[(k + 1) + j * ldb];
                            B[k + j * ldb] = d11 * t1 + d12 * t2;
                            B[(k + 1) + j * ldb] = d21 * t1 + d22 * t2;
                        }
                    }

                    if (k > 0) {

                        cblas_dger(CblasColMajor, k, nrhs, ONE, &A[kc], 1,
                                   &B[k], ldb, &B[0], ldb);
                        cblas_dger(CblasColMajor, k, nrhs, ONE, &A[kcnext], 1,
                                   &B[k + 1], ldb, &B[0], ldb);

                        kp = (ipiv[k] < 0) ? -(ipiv[k] + 1) : ipiv[k];
                        if (kp != k)
                            cblas_dswap(nrhs, &B[k], ldb, &B[kp], ldb);
                    }
                    kc = kcnext + k + 2;
                    k = k + 2;
                }
            }

        } else {

            k = n - 1;
            kc = n * (n + 1) / 2;
            while (k >= 0) {
                kc = kc - (n - k);

                if (ipiv[k] >= 0) {

                    if (nounit)
                        cblas_dscal(nrhs, A[kc], &B[k], ldb);

                    if (k != n - 1) {
                        kp = ipiv[k];

                        cblas_dger(CblasColMajor, n - k - 1, nrhs, ONE, &A[kc + 1], 1,
                                   &B[k], ldb, &B[k + 1], ldb);

                        if (kp != k)
                            cblas_dswap(nrhs, &B[k], ldb, &B[kp], ldb);
                    }
                    k = k - 1;

                } else {

                    kcnext = kc - (n - k + 1);

                    if (nounit) {
                        d11 = A[kcnext];
                        d22 = A[kc];
                        d21 = A[kcnext + 1];
                        d12 = d21;
                        for (j = 0; j < nrhs; j++) {
                            t1 = B[(k - 1) + j * ldb];
                            t2 = B[k + j * ldb];
                            B[(k - 1) + j * ldb] = d11 * t1 + d12 * t2;
                            B[k + j * ldb] = d21 * t1 + d22 * t2;
                        }
                    }

                    if (k != n - 1) {

                        cblas_dger(CblasColMajor, n - k - 1, nrhs, ONE, &A[kc + 1], 1,
                                   &B[k], ldb, &B[k + 1], ldb);
                        cblas_dger(CblasColMajor, n - k - 1, nrhs, ONE, &A[kcnext + 2], 1,
                                   &B[k - 1], ldb, &B[k + 1], ldb);

                        kp = (ipiv[k] < 0) ? -(ipiv[k] + 1) : ipiv[k];
                        if (kp != k)
                            cblas_dswap(nrhs, &B[k], ldb, &B[kp], ldb);
                    }
                    kc = kcnext;
                    k = k - 2;
                }
            }
        }

    } else {

        if (uplo[0] == 'U' || uplo[0] == 'u') {

            k = n - 1;
            kc = n * (n + 1) / 2;
            while (k >= 0) {
                kc = kc - (k + 1);

                if (ipiv[k] >= 0) {
                    if (k > 0) {

                        kp = ipiv[k];
                        if (kp != k)
                            cblas_dswap(nrhs, &B[k], ldb, &B[kp], ldb);

                        cblas_dgemv(CblasColMajor, CblasTrans, k, nrhs, ONE,
                                    &B[0], ldb, &A[kc], 1, ONE, &B[k], ldb);
                    }
                    if (nounit)
                        cblas_dscal(nrhs, A[kc + k], &B[k], ldb);
                    k = k - 1;

                } else {
                    kcnext = kc - k;
                    if (k > 1) {

                        kp = (ipiv[k] < 0) ? -(ipiv[k] + 1) : ipiv[k];
                        if (kp != k - 1)
                            cblas_dswap(nrhs, &B[k - 1], ldb, &B[kp], ldb);

                        cblas_dgemv(CblasColMajor, CblasTrans, k - 1, nrhs, ONE,
                                    &B[0], ldb, &A[kc], 1, ONE, &B[k], ldb);
                        cblas_dgemv(CblasColMajor, CblasTrans, k - 1, nrhs, ONE,
                                    &B[0], ldb, &A[kcnext], 1, ONE, &B[k - 1], ldb);
                    }

                    if (nounit) {
                        d11 = A[kc - 1];
                        d22 = A[kc + k];
                        d12 = A[kc + k - 1];
                        d21 = d12;
                        for (j = 0; j < nrhs; j++) {
                            t1 = B[(k - 1) + j * ldb];
                            t2 = B[k + j * ldb];
                            B[(k - 1) + j * ldb] = d11 * t1 + d12 * t2;
                            B[k + j * ldb] = d21 * t1 + d22 * t2;
                        }
                    }
                    kc = kcnext;
                    k = k - 2;
                }
            }

        } else {

            k = 0;
            kc = 0;
            while (k < n) {

                if (ipiv[k] >= 0) {
                    if (k < n - 1) {

                        kp = ipiv[k];
                        if (kp != k)
                            cblas_dswap(nrhs, &B[k], ldb, &B[kp], ldb);

                        cblas_dgemv(CblasColMajor, CblasTrans, n - k - 1, nrhs, ONE,
                                    &B[k + 1], ldb, &A[kc + 1], 1, ONE, &B[k], ldb);
                    }
                    if (nounit)
                        cblas_dscal(nrhs, A[kc], &B[k], ldb);
                    kc = kc + n - k;
                    k = k + 1;

                } else {
                    kcnext = kc + n - k;
                    if (k < n - 2) {

                        kp = (ipiv[k] < 0) ? -(ipiv[k] + 1) : ipiv[k];
                        if (kp != k + 1)
                            cblas_dswap(nrhs, &B[k + 1], ldb, &B[kp], ldb);

                        cblas_dgemv(CblasColMajor, CblasTrans, n - k - 2, nrhs, ONE,
                                    &B[k + 2], ldb, &A[kcnext + 1], 1, ONE,
                                    &B[k + 1], ldb);
                        cblas_dgemv(CblasColMajor, CblasTrans, n - k - 2, nrhs, ONE,
                                    &B[k + 2], ldb, &A[kc + 2], 1, ONE,
                                    &B[k], ldb);
                    }

                    if (nounit) {
                        d11 = A[kc];
                        d22 = A[kcnext];
                        d21 = A[kc + 1];
                        d12 = d21;
                        for (j = 0; j < nrhs; j++) {
                            t1 = B[k + j * ldb];
                            t2 = B[(k + 1) + j * ldb];
                            B[k + j * ldb] = d11 * t1 + d12 * t2;
                            B[(k + 1) + j * ldb] = d21 * t1 + d22 * t2;
                        }
                    }
                    kc = kcnext + (n - k - 1);
                    k = k + 2;
                }
            }
        }
    }
}
