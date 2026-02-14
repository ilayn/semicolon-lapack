/**
 * @file dqrt01.c
 * @brief DQRT01 tests DGEQRF and partially tests DORGQR.
 *
 * Compares R with Q'*A, and checks that Q is orthogonal.
 *
 * RESULT(0) = norm( R - Q'*A ) / ( M * norm(A) * EPS )
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
extern void dgeqrf(const int m, const int n,
                   f64* const restrict A, const int lda,
                   f64* const restrict tau,
                   f64* const restrict work, const int lwork, int* info);
extern void dorgqr(const int m, const int n, const int k,
                   f64* const restrict A, const int lda,
                   const f64* const restrict tau,
                   f64* const restrict work, const int lwork, int* info);

/**
 * DQRT01 tests DGEQRF, which computes the QR factorization of an m-by-n
 * matrix A, and partially tests DORGQR which forms the m-by-m
 * orthogonal matrix Q.
 *
 * @param[in]     m       Number of rows of A. m >= 0.
 * @param[in]     n       Number of columns of A. n >= 0.
 * @param[in]     A       The m-by-n matrix A (original).
 * @param[in,out] AF      On entry, copy of A. On exit, QR factorization of A.
 * @param[out]    Q       The m-by-m orthogonal matrix Q.
 * @param[out]    R       Workspace for R factor, dimension (lda, max(m,n)).
 * @param[in]     lda     Leading dimension of arrays. lda >= max(m,n).
 * @param[out]    tau     Array of dimension min(m,n).
 * @param[out]    work    Workspace, dimension lwork.
 * @param[in]     lwork   Dimension of work.
 * @param[out]    rwork   Workspace, dimension m.
 * @param[out]    result  Array of dimension 2 with test ratios.
 */
void dqrt01(const int m, const int n,
            const f64 * const restrict A,
            f64 * const restrict AF,
            f64 * const restrict Q,
            f64 * const restrict R,
            const int lda,
            f64 * const restrict tau,
            f64 * const restrict work, const int lwork,
            f64 * const restrict rwork,
            f64 * restrict result)
{
    int minmn = m < n ? m : n;
    f64 eps = dlamch("E");
    int info;

    /* Copy the matrix A to AF */
    dlacpy("F", m, n, A, lda, AF, lda);

    /* Factorize A in AF */
    dgeqrf(m, n, AF, lda, tau, work, lwork, &info);

    /* Copy details of Q: the lower triangle of AF (below diagonal) contains
     * the reflectors. Copy to Q. */
    dlaset("F", m, m, -1.0e+10, -1.0e+10, Q, lda);
    if (m > 1) {
        /* Copy lower trapezoid of AF (rows 1:m-1, cols 0:min(m-1,n)-1) to Q */
        dlacpy("L", m - 1, n, &AF[1 + 0 * lda], lda, &Q[1 + 0 * lda], lda);
    }

    /* Generate the m-by-m matrix Q */
    dorgqr(m, m, minmn, Q, lda, tau, work, lwork, &info);

    /* Copy R: upper triangle of AF */
    dlaset("F", m, n, 0.0, 0.0, R, lda);
    dlacpy("U", m, n, AF, lda, R, lda);

    /* Compute R - Q'*A */
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                m, n, m, -1.0, Q, lda, A, lda, 1.0, R, lda);

    /* Compute norm( R - Q'*A ) / ( M * norm(A) * EPS ) */
    f64 anorm = dlange("1", m, n, A, lda, rwork);
    f64 resid = dlange("1", m, n, R, lda, rwork);
    if (anorm > 0.0) {
        result[0] = ((resid / (f64)(m > 1 ? m : 1)) / anorm) / eps;
    } else {
        result[0] = 0.0;
    }

    /* Compute I - Q'*Q */
    dlaset("F", m, m, 0.0, 1.0, R, lda);
    cblas_dsyrk(CblasColMajor, CblasUpper, CblasTrans,
                m, m, -1.0, Q, lda, 1.0, R, lda);

    /* Compute norm( I - Q'*Q ) / ( M * EPS ) */
    resid = dlansy("1", "U", m, R, lda, rwork);
    result[1] = (resid / (f64)(m > 1 ? m : 1)) / eps;
}
