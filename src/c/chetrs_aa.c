/**
 * @file chetrs_aa.c
 * @brief CHETRS_AA solves a system of linear equations A*X = B with a complex hermitian matrix using the factorization computed by CHETRF_AA.
 */

#include <complex.h>
#include <cblas.h>
#include "semicolon_lapack_complex_single.h"

/**
 * CHETRS_AA solves a system of linear equations A*X = B with a complex
 * hermitian matrix A using the factorization A = U**H*T*U or
 * A = L*T*L**H computed by CHETRF_AA.
 *
 * @param[in] uplo
 *          Specifies whether the details of the factorization are stored
 *          as an upper or lower triangular matrix.
 *          = 'U':  Upper triangular, form is A = U**H*T*U;
 *          = 'L':  Lower triangular, form is A = L*T*L**H.
 *
 * @param[in] n
 *          The order of the matrix A. n >= 0.
 *
 * @param[in] nrhs
 *          The number of right hand sides, i.e., the number of columns
 *          of the matrix B. nrhs >= 0.
 *
 * @param[in] A
 *          Complex*16 array, dimension (lda, n).
 *          Details of factors computed by CHETRF_AA.
 *
 * @param[in] lda
 *          The leading dimension of the array A. lda >= max(1, n).
 *
 * @param[in] ipiv
 *          Integer array, dimension (n).
 *          Details of the interchanges as computed by CHETRF_AA.
 *
 * @param[in,out] B
 *          Complex*16 array, dimension (ldb, nrhs).
 *          On entry, the right hand side matrix B.
 *          On exit, the solution matrix X.
 *
 * @param[in] ldb
 *          The leading dimension of the array B. ldb >= max(1, n).
 *
 * @param[out] work
 *          Complex*16 array, dimension (max(1, lwork)).
 *
 * @param[in] lwork
 *          The dimension of the array work.
 *          If min(n, nrhs) = 0, lwork >= 1, else lwork >= 3*n-2.
 *          If lwork = -1, then a workspace query is assumed.
 *
 * @param[out] info
 *                         - = 0:  successful exit
 *                         - < 0:  if info = -i, the i-th argument had an illegal value
 */
void chetrs_aa(
    const char* uplo,
    const int n,
    const int nrhs,
    const c64* restrict A,
    const int lda,
    const int* restrict ipiv,
    c64* restrict B,
    const int ldb,
    c64* restrict work,
    const int lwork,
    int* info)
{
    const c64 ONE = CMPLXF(1.0f, 0.0f);
    int upper, lquery;
    int k, kp, lwkmin;
    int minval;

    *info = 0;
    upper = (uplo[0] == 'U' || uplo[0] == 'u');
    lquery = (lwork == -1);

    minval = (n < nrhs) ? n : nrhs;
    if (minval == 0) {
        lwkmin = 1;
    } else {
        lwkmin = 3 * n - 2;
    }

    if (!upper && !(uplo[0] == 'L' || uplo[0] == 'l')) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (nrhs < 0) {
        *info = -3;
    } else if (lda < (1 > n ? 1 : n)) {
        *info = -5;
    } else if (ldb < (1 > n ? 1 : n)) {
        *info = -8;
    } else if (lwork < lwkmin && !lquery) {
        *info = -10;
    }

    if (*info != 0) {
        xerbla("CHETRS_AA", -(*info));
        return;
    } else if (lquery) {
        work[0] = CMPLXF((f32)lwkmin, 0.0f);
        return;
    }

    if (minval == 0) {
        return;
    }

    if (upper) {

        /*  Solve A*X = B, where A = U**H*T*U.  */

        /*  1) Forward substitution with U**H  */

        if (n > 1) {

            /*  Pivot, P**T * B -> B  */

            for (k = 0; k < n; k++) {
                kp = ipiv[k];
                if (kp != k) {
                    cblas_cswap(nrhs, &B[k + 0 * ldb], ldb, &B[kp + 0 * ldb], ldb);
                }
            }

            /*  Compute U**H \ B -> B    [ (U**H \P**T * B) ]  */

            cblas_ctrsm(CblasColMajor, CblasLeft, CblasUpper, CblasConjTrans, CblasUnit,
                        n - 1, nrhs, &ONE, &A[0 + 1 * lda], lda, &B[1 + 0 * ldb], ldb);
        }

        /*  2) Solve with triangular matrix T  */

        /*  Compute T \ B -> B   [ T \ (U**H \P**T * B) ]  */

        /* CLACPY('F', 1, N, A(1,1), LDA+1, WORK(N), 1) -- extract diagonal */
        for (k = 0; k < n; k++) {
            work[n - 1 + k] = A[k + k * lda];
        }
        if (n > 1) {
            /* CLACPY('F', 1, N-1, A(1,2), LDA+1, WORK(2*N), 1) -- superdiagonal */
            for (k = 0; k < n - 1; k++) {
                work[2 * n - 1 + k] = A[k + (k + 1) * lda];
            }
            /* CLACPY('F', 1, N-1, A(1,2), LDA+1, WORK(1), 1) -- copy to sub */
            for (k = 0; k < n - 1; k++) {
                work[k] = A[k + (k + 1) * lda];
            }
            /* CLACGV(N-1, WORK(1), 1) -- conjugate the sub-diagonal */
            for (k = 0; k < n - 1; k++) {
                work[k] = conjf(work[k]);
            }
        }
        cgtsv(n, nrhs, &work[0], &work[n - 1], &work[2 * n - 1], B, ldb, info);

        /*  3) Backward substitution with U  */

        if (n > 1) {

            /*  Compute U \ B -> B   [ U \ (T \ (U**H \P**T * B) ) ]  */

            cblas_ctrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasUnit,
                        n - 1, nrhs, &ONE, &A[0 + 1 * lda], lda, &B[1 + 0 * ldb], ldb);

            /*  Pivot, P * B  [ P * (U**H \ (T \ (U \P**T * B) )) ]  */

            for (k = n - 1; k >= 0; k--) {
                kp = ipiv[k];
                if (kp != k) {
                    cblas_cswap(nrhs, &B[k + 0 * ldb], ldb, &B[kp + 0 * ldb], ldb);
                }
            }
        }

    } else {

        /*  Solve A*X = B, where A = L*T*L**H.  */

        /*  1) Forward substitution with L  */

        if (n > 1) {

            /*  Pivot, P**T * B -> B  */

            for (k = 0; k < n; k++) {
                kp = ipiv[k];
                if (kp != k) {
                    cblas_cswap(nrhs, &B[k + 0 * ldb], ldb, &B[kp + 0 * ldb], ldb);
                }
            }

            /*  Compute L \ B -> B    [ (L \P**T * B) ]  */

            cblas_ctrsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit,
                        n - 1, nrhs, &ONE, &A[1 + 0 * lda], lda, &B[1 + 0 * ldb], ldb);
        }

        /*  2) Solve with triangular matrix T  */

        /*  Compute T \ B -> B   [ T \ (L \P**T * B) ]  */

        /* CLACPY('F', 1, N, A(1,1), LDA+1, WORK(N), 1) -- extract diagonal */
        for (k = 0; k < n; k++) {
            work[n - 1 + k] = A[k + k * lda];
        }
        if (n > 1) {
            /* CLACPY('F', 1, N-1, A(2,1), LDA+1, WORK(1), 1) -- subdiagonal */
            for (k = 0; k < n - 1; k++) {
                work[k] = A[(k + 1) + k * lda];
            }
            /* CLACPY('F', 1, N-1, A(2,1), LDA+1, WORK(2*N), 1) -- copy to super */
            for (k = 0; k < n - 1; k++) {
                work[2 * n - 1 + k] = A[(k + 1) + k * lda];
            }
            /* CLACGV(N-1, WORK(2*N), 1) -- conjugate the super-diagonal */
            for (k = 0; k < n - 1; k++) {
                work[2 * n - 1 + k] = conjf(work[2 * n - 1 + k]);
            }
        }
        cgtsv(n, nrhs, &work[0], &work[n - 1], &work[2 * n - 1], B, ldb, info);

        /*  3) Backward substitution with L**H  */

        if (n > 1) {

            /*  Compute L**H \ B -> B   [ L**H \ (T \ (L \P**T * B) ) ]  */

            cblas_ctrsm(CblasColMajor, CblasLeft, CblasLower, CblasConjTrans, CblasUnit,
                        n - 1, nrhs, &ONE, &A[1 + 0 * lda], lda, &B[1 + 0 * ldb], ldb);

            /*  Pivot, P * B  [ P * (L**H \ (T \ (L \P**T * B) )) ]  */

            for (k = n - 1; k >= 0; k--) {
                kp = ipiv[k];
                if (kp != k) {
                    cblas_cswap(nrhs, &B[k + 0 * ldb], ldb, &B[kp + 0 * ldb], ldb);
                }
            }
        }
    }
}
