/**
 * @file dqlt01.c
 * @brief DQLT01 tests DGEQLF and partially tests DORGQL.
 *
 * Compares L with Q'*A, and checks that Q is orthogonal.
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
extern void dgeqlf(const int m, const int n,
                   f64* const restrict A, const int lda,
                   f64* const restrict tau,
                   f64* const restrict work, const int lwork, int* info);
extern void dorgql(const int m, const int n, const int k,
                   f64* const restrict A, const int lda,
                   const f64* const restrict tau,
                   f64* const restrict work, const int lwork, int* info);

/**
 * DQLT01 tests DGEQLF, which computes the QL factorization of an m-by-n
 * matrix A, and partially tests DORGQL which forms the m-by-m
 * orthogonal matrix Q.
 *
 * @param[in]     m       Number of rows of A. m >= 0.
 * @param[in]     n       Number of columns of A. n >= 0.
 * @param[in]     A       The m-by-n matrix A (original).
 * @param[in,out] AF      On entry, copy of A. On exit, QL factorization.
 * @param[out]    Q       The m-by-m orthogonal matrix Q.
 * @param[out]    L       Workspace, dimension (lda, max(m,n)).
 * @param[in]     lda     Leading dimension. lda >= max(m,n).
 * @param[out]    tau     Array of dimension min(m,n).
 * @param[out]    work    Workspace.
 * @param[in]     lwork   Dimension of work.
 * @param[out]    rwork   Workspace, dimension m.
 * @param[out]    result  Array of dimension 2.
 */
void dqlt01(const int m, const int n,
            const f64 * const restrict A,
            f64 * const restrict AF,
            f64 * const restrict Q,
            f64 * const restrict L,
            const int lda,
            f64 * const restrict tau,
            f64 * const restrict work, const int lwork,
            f64 * const restrict rwork,
            f64 * restrict result)
{
    int minmn = m < n ? m : n;
    f64 eps = dlamch("E");
    int info;

    /* Copy A to AF */
    dlacpy("F", m, n, A, lda, AF, lda);

    /* Factorize A in AF */
    dgeqlf(m, n, AF, lda, tau, work, lwork, &info);

    /* Copy details of Q.
     * For QL, reflectors are stored in the last minmn columns.
     * For m >= n: reflectors are in all n columns, above the lower triangle
     *             of AF(m-n:m-1, 0:n-1).
     * For m < n:  reflectors are in the last m columns, above the lower
     *             triangle of AF(0:m-1, n-m:n-1). */
    dlaset("F", m, m, -1.0e+10, -1.0e+10, Q, lda);
    if (m >= n) {
        if (n < m && n > 0) {
            /* Copy the non-triangular part: AF(0:m-n-1, 0:n-1) → Q(0:m-n-1, m-n:m-1) */
            dlacpy("F", m - n, n, AF, lda, &Q[0 + (m - n) * lda], lda);
        }
        if (n > 1) {
            /* Copy upper triangle of AF(m-n:m-2, 1:n-1) → Q(m-n:m-2, m-n+1:m-1) */
            dlacpy("U", n - 1, n - 1, &AF[(m - n) + 1 * lda], lda,
                   &Q[(m - n) + (m - n + 1) * lda], lda);
        }
    } else {
        /* m < n */
        if (m > 1) {
            /* Copy upper triangle of AF(0:m-2, n-m+1:n-1) → Q(0:m-2, 1:m-1) */
            dlacpy("U", m - 1, m - 1, &AF[0 + (n - m + 1) * lda], lda,
                   &Q[0 + 1 * lda], lda);
        }
    }

    /* Generate the m-by-m matrix Q */
    dorgql(m, m, minmn, Q, lda, tau, work, lwork, &info);

    /* Copy L: lower triangle of AF.
     * For m >= n: L is n-by-n in AF(m-n:m-1, 0:n-1).
     * For m < n:  L is m-by-n, with the lower-left m-by-m part in AF(0:m-1, n-m:n-1). */
    dlaset("F", m, n, 0.0, 0.0, L, lda);
    if (m >= n) {
        if (n > 0) {
            dlacpy("L", n, n, &AF[(m - n) + 0 * lda], lda,
                   &L[(m - n) + 0 * lda], lda);
        }
    } else {
        if (n > m && m > 0) {
            dlacpy("F", m, n - m, AF, lda, L, lda);
        }
        if (m > 0) {
            dlacpy("L", m, m, &AF[0 + (n - m) * lda], lda,
                   &L[0 + (n - m) * lda], lda);
        }
    }

    /* Compute L - Q'*A */
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                m, n, m, -1.0, Q, lda, A, lda, 1.0, L, lda);

    /* Compute norm( L - Q'*A ) / ( M * norm(A) * EPS ) */
    f64 anorm = dlange("1", m, n, A, lda, rwork);
    f64 resid = dlange("1", m, n, L, lda, rwork);
    if (anorm > 0.0) {
        result[0] = ((resid / (f64)(m > 1 ? m : 1)) / anorm) / eps;
    } else {
        result[0] = 0.0;
    }

    /* Compute I - Q'*Q */
    dlaset("F", m, m, 0.0, 1.0, L, lda);
    cblas_dsyrk(CblasColMajor, CblasUpper, CblasTrans,
                m, m, -1.0, Q, lda, 1.0, L, lda);

    /* Compute norm( I - Q'*Q ) / ( M * EPS ) */
    resid = dlansy("1", "U", m, L, lda, rwork);
    result[1] = (resid / (f64)(m > 1 ? m : 1)) / eps;
}
