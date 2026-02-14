/**
 * @file dqlt02.c
 * @brief DQLT02 tests DORGQL for generating a partial Q matrix from QL.
 *
 * Given the QL factorization of an m-by-n matrix A, DQLT02 generates
 * the orthogonal matrix Q defined by the factorization of the last k
 * columns of A; it compares L(m-n+1:m, n-k+1:n) with
 * Q(1:m, m-n+1:m)' * A(1:m, n-k+1:n), and checks that the columns of Q
 * are orthonormal.
 *
 * RESULT(0) = norm( L - Q'*A ) / ( M * norm(A) * EPS )
 * RESULT(1) = norm( I - Q'*Q ) / ( M * EPS )
 */

#include <stdlib.h>
#include <math.h>
#include "verify.h"
#include <cblas.h>

// Forward declarations
extern f64 dlamch(const char* cmach);
extern f64 dlange(const char* norm, const int m, const int n,
                     const f64* const restrict A, const int lda,
                     f64* const restrict work);
extern f64 dlansy(const char* norm, const char* uplo, const int n,
                     const f64* const restrict A, const int lda,
                     f64* const restrict work);
extern void dlacpy(const char* uplo, const int m, const int n,
                   const f64* const restrict A, const int lda,
                   f64* const restrict B, const int ldb);
extern void dlaset(const char* uplo, const int m, const int n,
                   const f64 alpha, const f64 beta,
                   f64* const restrict A, const int lda);
extern void dorgql(const int m, const int n, const int k,
                   f64* const restrict A, const int lda,
                   const f64* const restrict tau,
                   f64* const restrict work, const int lwork, int* info);

/**
 * @param[in]     m       Number of rows of Q to generate. m >= 0.
 * @param[in]     n       Number of columns of Q. m >= n >= 0.
 * @param[in]     k       Number of reflectors. n >= k >= 0.
 * @param[in]     A       The m-by-n original matrix (last k columns used).
 * @param[in]     AF      QL factorization from DGEQLF.
 * @param[out]    Q       The m-by-n matrix Q.
 * @param[out]    L       Workspace, dimension (lda, n).
 * @param[in]     lda     Leading dimension >= m.
 * @param[in]     tau     Scalar factors from DGEQLF, dimension min(m,n).
 * @param[out]    work    Workspace.
 * @param[in]     lwork   Dimension of work.
 * @param[out]    rwork   Workspace, dimension m.
 * @param[out]    result  Array of dimension 2.
 */
void dqlt02(const int m, const int n, const int k,
            const f64* const restrict A,
            const f64* const restrict AF,
            f64* const restrict Q,
            f64* const restrict L,
            const int lda,
            const f64* const restrict tau,
            f64* const restrict work, const int lwork,
            f64* const restrict rwork,
            f64* restrict result)
{
    /* Quick return if possible */
    if (m == 0 || n == 0 || k == 0) {
        result[0] = 0.0;
        result[1] = 0.0;
        return;
    }

    f64 eps = dlamch("E");
    int info;
    int minmn = m < n ? m : n;

    /* Copy the last k columns of the factorization to the array Q.
     * For QL, reflectors for the last k columns are stored in AF(1:m, n-k+1:n).
     * We need to copy:
     *   - The non-triangular part: AF(1:m-k, n-k+1:n) -> Q(1:m-k, n-k+1:n)
     *   - The upper triangular part: AF(m-k+1:m-1, n-k+2:n) -> Q(m-k+1:m-1, n-k+2:n)
     * In 0-indexed C:
     *   - AF(0:m-k-1, n-k:n-1) -> Q(0:m-k-1, n-k:n-1)
     *   - Upper of AF(m-k:m-2, n-k+1:n-1) -> Q(m-k:m-2, n-k+1:n-1)
     */
    dlaset("F", m, n, -1.0e+10, -1.0e+10, Q, lda);
    if (k < m) {
        /* Copy AF(0:m-k-1, n-k:n-1) -> Q(0:m-k-1, n-k:n-1) */
        dlacpy("F", m - k, k, &AF[0 + (n - k) * lda], lda,
               &Q[0 + (n - k) * lda], lda);
    }
    if (k > 1) {
        /* Copy upper triangle of AF(m-k:m-2, n-k+1:n-1) -> Q(m-k:m-2, n-k+1:n-1) */
        dlacpy("U", k - 1, k - 1, &AF[(m - k) + (n - k + 1) * lda], lda,
               &Q[(m - k) + (n - k + 1) * lda], lda);
    }

    /* Generate the last n columns of the matrix Q.
     * tau for the last k reflectors starts at tau[minmn-k] (0-indexed). */
    dorgql(m, n, k, Q, lda, &tau[minmn - k], work, lwork, &info);

    /* Copy L(m-n:m-1, n-k:n-1) - this is an n-by-k block.
     * The lower triangular part is in AF(m-k:m-1, n-k:n-1). */
    dlaset("F", n, k, 0.0, 0.0, &L[(m - n) + (n - k) * lda], lda);
    dlacpy("L", k, k, &AF[(m - k) + (n - k) * lda], lda,
           &L[(m - k) + (n - k) * lda], lda);

    /* Compute L(m-n:m-1, n-k:n-1) - Q(0:m-1, m-n:m-1)' * A(0:m-1, n-k:n-1)
     * This is: L_block -= Q' * A_block
     * where L_block is n x k, Q is m x n, A_block is m x k */
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                n, k, m, -1.0, Q, lda, &A[0 + (n - k) * lda], lda,
                1.0, &L[(m - n) + (n - k) * lda], lda);

    /* Compute norm( L - Q'*A ) / ( M * norm(A) * EPS ) */
    f64 anorm = dlange("1", m, k, &A[0 + (n - k) * lda], lda, rwork);
    f64 resid = dlange("1", n, k, &L[(m - n) + (n - k) * lda], lda, rwork);
    if (anorm > 0.0) {
        result[0] = ((resid / (f64)(m > 1 ? m : 1)) / anorm) / eps;
    } else {
        result[0] = 0.0;
    }

    /* Compute I - Q'*Q */
    dlaset("F", n, n, 0.0, 1.0, L, lda);
    cblas_dsyrk(CblasColMajor, CblasUpper, CblasTrans,
                n, m, -1.0, Q, lda, 1.0, L, lda);

    /* Compute norm( I - Q'*Q ) / ( M * EPS ) */
    resid = dlansy("1", "U", n, L, lda, rwork);
    result[1] = (resid / (f64)(m > 1 ? m : 1)) / eps;
}
