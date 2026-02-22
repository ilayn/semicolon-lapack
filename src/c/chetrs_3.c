/**
 * @file chetrs_3.c
 * @brief CHETRS_3 solves a system of linear equations A*X = B with a complex Hermitian matrix using the factorization computed by CHETRF_RK or ZHETRF_BK.
 */

#include "internal_build_defs.h"
#include <complex.h>
#include <math.h>
#include <cblas.h>
#include "semicolon_lapack_complex_single.h"

/**
 * CHETRS_3 solves a system of linear equations A * X = B with a complex
 * Hermitian matrix A using the factorization computed
 * by CHETRF_RK or ZHETRF_BK:
 *
 *    A = P*U*D*(U**H)*(P**T) or A = P*L*D*(L**H)*(P**T),
 *
 * where U (or L) is unit upper (or lower) triangular matrix,
 * U**H (or L**H) is the conjugate of U (or L), P is a permutation
 * matrix, P**T is the transpose of P, and D is Hermitian and block
 * diagonal with 1-by-1 and 2-by-2 diagonal blocks.
 *
 * This algorithm is using Level 3 BLAS.
 *
 * @param[in] uplo
 *          Specifies whether the details of the factorization are
 *          stored as an upper or lower triangular matrix:
 *          = 'U':  Upper triangular, form is A = P*U*D*(U**H)*(P**T);
 *          = 'L':  Lower triangular, form is A = P*L*D*(L**H)*(P**T).
 *
 * @param[in] n
 *          The order of the matrix A. n >= 0.
 *
 * @param[in] nrhs
 *          The number of right hand sides. nrhs >= 0.
 *
 * @param[in] A
 *          Complex*16 array, dimension (lda, n).
 *          Diagonal of the block diagonal matrix D and factors U or L
 *          as computed by CHETRF_RK and ZHETRF_BK:
 *            a) ONLY diagonal elements of the Hermitian block diagonal
 *               matrix D on the diagonal of A, i.e. D(k,k) = A(k,k);
 *               (superdiagonal (or subdiagonal) elements of D
 *                should be provided on entry in array E), and
 *            b) If UPLO = 'U': factor U in the superdiagonal part of A.
 *               If UPLO = 'L': factor L in the subdiagonal part of A.
 *
 * @param[in] lda
 *          The leading dimension of the array A. lda >= max(1, n).
 *
 * @param[in] E
 *          Complex*16 array, dimension (n).
 *          On entry, contains the superdiagonal (or subdiagonal)
 *          elements of the Hermitian block diagonal matrix D
 *          with 1-by-1 or 2-by-2 diagonal blocks, where
 *          If UPLO = 'U': E(i) = D(i-1,i), i=2:N, E(1) not referenced;
 *          If UPLO = 'L': E(i) = D(i+1,i), i=1:N-1, E(N) not referenced.
 *
 *          NOTE: For 1-by-1 diagonal block D(k), where
 *          1 <= k <= N, the element E(k) is not referenced in both
 *          UPLO = 'U' or UPLO = 'L' cases.
 *
 * @param[in] ipiv
 *          Integer array, dimension (n).
 *          Details of the interchanges and the block structure of D
 *          as determined by CHETRF_RK or ZHETRF_BK.
 *
 * @param[in,out] B
 *          Complex*16 array, dimension (ldb, nrhs).
 *          On entry, the right hand side matrix B.
 *          On exit, the solution matrix X.
 *
 * @param[in] ldb
 *          The leading dimension of the array B. ldb >= max(1, n).
 *
 * @param[out] info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 */
void chetrs_3(
    const char* uplo,
    const INT n,
    const INT nrhs,
    const c64* restrict A,
    const INT lda,
    const c64* restrict E,
    const INT* restrict ipiv,
    c64* restrict B,
    const INT ldb,
    INT* info)
{
    const c64 ONE = CMPLXF(1.0f, 0.0f);

    INT upper;
    INT i, j, k, kp;
    f32 s;
    c64 ak, akm1, akm1k, bk, bkm1, denom;

    *info = 0;
    upper = (uplo[0] == 'U' || uplo[0] == 'u');

    if (!upper && !(uplo[0] == 'L' || uplo[0] == 'l')) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (nrhs < 0) {
        *info = -3;
    } else if (lda < (1 > n ? 1 : n)) {
        *info = -5;
    } else if (ldb < (1 > n ? 1 : n)) {
        *info = -9;
    }
    if (*info != 0) {
        xerbla("CHETRS_3", -(*info));
        return;
    }

    if (n == 0 || nrhs == 0) {
        return;
    }

    if (upper) {

        for (k = n - 1; k >= 0; k--) {
            kp = (ipiv[k] >= 0) ? ipiv[k] : -(ipiv[k] + 1);
            if (kp != k) {
                cblas_cswap(nrhs, &B[k + 0 * ldb], ldb, &B[kp + 0 * ldb], ldb);
            }
        }

        cblas_ctrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasUnit,
                    n, nrhs, &ONE, A, lda, B, ldb);

        i = n - 1;
        while (i >= 0) {
            if (ipiv[i] >= 0) {
                s = 1.0f / crealf(A[i + i * lda]);
                cblas_csscal(nrhs, s, &B[i + 0 * ldb], ldb);
            } else if (i > 0) {
                akm1k = E[i];
                akm1 = A[i - 1 + (i - 1) * lda] / akm1k;
                ak = A[i + i * lda] / conjf(akm1k);
                denom = akm1 * ak - ONE;
                for (j = 0; j < nrhs; j++) {
                    bkm1 = B[i - 1 + j * ldb] / akm1k;
                    bk = B[i + j * ldb] / conjf(akm1k);
                    B[i - 1 + j * ldb] = (ak * bkm1 - bk) / denom;
                    B[i + j * ldb] = (akm1 * bk - bkm1) / denom;
                }
                i = i - 1;
            }
            i = i - 1;
        }

        cblas_ctrsm(CblasColMajor, CblasLeft, CblasUpper, CblasConjTrans, CblasUnit,
                    n, nrhs, &ONE, A, lda, B, ldb);

        for (k = 0; k < n; k++) {
            kp = (ipiv[k] >= 0) ? ipiv[k] : -(ipiv[k] + 1);
            if (kp != k) {
                cblas_cswap(nrhs, &B[k + 0 * ldb], ldb, &B[kp + 0 * ldb], ldb);
            }
        }

    } else {

        for (k = 0; k < n; k++) {
            kp = (ipiv[k] >= 0) ? ipiv[k] : -(ipiv[k] + 1);
            if (kp != k) {
                cblas_cswap(nrhs, &B[k + 0 * ldb], ldb, &B[kp + 0 * ldb], ldb);
            }
        }

        cblas_ctrsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit,
                    n, nrhs, &ONE, A, lda, B, ldb);

        i = 0;
        while (i < n) {
            if (ipiv[i] >= 0) {
                s = 1.0f / crealf(A[i + i * lda]);
                cblas_csscal(nrhs, s, &B[i + 0 * ldb], ldb);
            } else if (i < n - 1) {
                akm1k = E[i];
                akm1 = A[i + i * lda] / conjf(akm1k);
                ak = A[i + 1 + (i + 1) * lda] / akm1k;
                denom = akm1 * ak - ONE;
                for (j = 0; j < nrhs; j++) {
                    bkm1 = B[i + j * ldb] / conjf(akm1k);
                    bk = B[i + 1 + j * ldb] / akm1k;
                    B[i + j * ldb] = (ak * bkm1 - bk) / denom;
                    B[i + 1 + j * ldb] = (akm1 * bk - bkm1) / denom;
                }
                i = i + 1;
            }
            i = i + 1;
        }

        cblas_ctrsm(CblasColMajor, CblasLeft, CblasLower, CblasConjTrans, CblasUnit,
                    n, nrhs, &ONE, A, lda, B, ldb);

        for (k = n - 1; k >= 0; k--) {
            kp = (ipiv[k] >= 0) ? ipiv[k] : -(ipiv[k] + 1);
            if (kp != k) {
                cblas_cswap(nrhs, &B[k + 0 * ldb], ldb, &B[kp + 0 * ldb], ldb);
            }
        }
    }
}
