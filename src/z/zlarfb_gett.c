/**
 * @file zlarfb_gett.c
 * @brief ZLARFB_GETT applies a complex Householder block reflector H from the left to a complex (K+M)-by-N triangular-pentagonal matrix.
 */

#include <complex.h>
#include <cblas.h>
#include "semicolon_lapack_complex_double.h"

/**
 * ZLARFB_GETT applies a complex Householder block reflector H from the
 * left to a complex (K+M)-by-N "triangular-pentagonal" matrix
 * composed of two block matrices: an upper trapezoidal K-by-N matrix A
 * stored in the array A, and a rectangular M-by-(N-K) matrix B, stored
 * in the array B. The block reflector H is stored in a compact
 * WY-representation, where the elementary reflectors are in the
 * arrays A, B and T.
 *
 * @param[in] ident
 *          If ident = 'I' or 'i', then V1 is an identity matrix and
 *             not stored.
 *          Otherwise, V1 is unit lower-triangular and stored in the
 *             left K-by-K block of the input matrix A.
 *
 * @param[in] m
 *          The number of rows of the matrix B. m >= 0.
 *
 * @param[in] n
 *          The number of columns of the matrices A and B. n >= 0.
 *
 * @param[in] k
 *          The number of rows of the matrix A.
 *          K is also order of the matrix T. 0 <= k <= n.
 *
 * @param[in] T
 *          Complex array, dimension (ldt, k).
 *          The upper-triangular K-by-K matrix T.
 *
 * @param[in] ldt
 *          The leading dimension of the array T. ldt >= k.
 *
 * @param[in,out] A
 *          Complex array, dimension (lda, n).
 *          On entry: upper-trapezoidal part contains A, columns below
 *          the diagonal contain columns of V1 (ones not stored).
 *          On exit: A is overwritten by H*A.
 *
 * @param[in] lda
 *          The leading dimension of the array A. lda >= max(1, k).
 *
 * @param[in,out] B
 *          Complex array, dimension (ldb, n).
 *          On entry: right M-by-(N-K) block contains B,
 *          left M-by-K block contains V2.
 *          On exit: B is overwritten by H*B.
 *
 * @param[in] ldb
 *          The leading dimension of the array B. ldb >= max(1, m).
 *
 * @param[out] work
 *          Complex array, dimension (ldwork, max(k, n-k)).
 *
 * @param[in] ldwork
 *          The leading dimension of the array work. ldwork >= max(1, k).
 */
void zlarfb_gett(
    const char* ident,
    const int m,
    const int n,
    const int k,
    const c128* const restrict T,
    const int ldt,
    c128* const restrict A,
    const int lda,
    c128* const restrict B,
    const int ldb,
    c128* restrict work,
    const int ldwork)
{
    const c128 CONE = 1.0;
    const c128 CNEG_ONE = -1.0;

    int lnotident;
    int i, j;

    if (m < 0 || n <= 0 || k == 0 || k > n)
        return;

    lnotident = !(ident[0] == 'I' || ident[0] == 'i');

    if (n > k) {

        for (j = 0; j < n - k; j++) {
            cblas_zcopy(k, &A[0 + (k + j) * lda], 1, &work[0 + j * ldwork], 1);
        }

        if (lnotident) {

            cblas_ztrmm(CblasColMajor, CblasLeft, CblasLower, CblasConjTrans, CblasUnit,
                        k, n - k, &CONE, A, lda, work, ldwork);
        }

        if (m > 0) {
            cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans,
                        k, n - k, m, &CONE, B, ldb, &B[0 + k * ldb], ldb, &CONE, work, ldwork);
        }

        cblas_ztrmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
                    k, n - k, &CONE, T, ldt, work, ldwork);

        if (m > 0) {
            cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                        m, n - k, k, &CNEG_ONE, B, ldb, work, ldwork, &CONE, &B[0 + k * ldb], ldb);
        }

        if (lnotident) {

            cblas_ztrmm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit,
                        k, n - k, &CONE, A, lda, work, ldwork);
        }

        for (j = 0; j < n - k; j++) {
            for (i = 0; i < k; i++) {
                A[i + (k + j) * lda] = A[i + (k + j) * lda] - work[i + j * ldwork];
            }
        }

    }

    for (j = 0; j < k; j++) {
        cblas_zcopy(j + 1, &A[0 + j * lda], 1, &work[0 + j * ldwork], 1);
    }

    for (j = 0; j < k - 1; j++) {
        for (i = j + 1; i < k; i++) {
            work[i + j * ldwork] = 0.0;
        }
    }

    if (lnotident) {

        cblas_ztrmm(CblasColMajor, CblasLeft, CblasLower, CblasConjTrans, CblasUnit,
                    k, k, &CONE, A, lda, work, ldwork);
    }

    cblas_ztrmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
                k, k, &CONE, T, ldt, work, ldwork);

    if (m > 0) {
        cblas_ztrmm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit,
                    m, k, &CNEG_ONE, work, ldwork, B, ldb);
    }

    if (lnotident) {

        cblas_ztrmm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit,
                    k, k, &CONE, A, lda, work, ldwork);

        for (j = 0; j < k - 1; j++) {
            for (i = j + 1; i < k; i++) {
                A[i + j * lda] = -work[i + j * ldwork];
            }
        }

    }

    for (j = 0; j < k; j++) {
        for (i = 0; i <= j; i++) {
            A[i + j * lda] = A[i + j * lda] - work[i + j * ldwork];
        }
    }
}
