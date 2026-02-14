/**
 * @file dlqt01.c
 * @brief DLQT01 tests DGELQF and partially tests DORGLQ.
 *
 * Compares L with A*Q', and checks that Q is orthogonal.
 *
 * RESULT(0) = norm( L - A*Q' ) / ( N * norm(A) * EPS )
 * RESULT(1) = norm( I - Q*Q' ) / ( N * EPS )
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
extern void dgelqf(const int m, const int n,
                   f64* const restrict A, const int lda,
                   f64* const restrict tau,
                   f64* const restrict work, const int lwork, int* info);
extern void dorglq(const int m, const int n, const int k,
                   f64* const restrict A, const int lda,
                   const f64* const restrict tau,
                   f64* const restrict work, const int lwork, int* info);

/**
 * DLQT01 tests DGELQF, which computes the LQ factorization of an m-by-n
 * matrix A, and partially tests DORGLQ which forms the n-by-n
 * orthogonal matrix Q.
 *
 * @param[in]     m       Number of rows of A. m >= 0.
 * @param[in]     n       Number of columns of A. n >= 0.
 * @param[in]     A       The m-by-n matrix A (original).
 * @param[in,out] AF      On entry, copy of A. On exit, LQ factorization.
 * @param[out]    Q       The n-by-n orthogonal matrix Q.
 * @param[out]    L       Workspace, dimension (lda, max(m,n)).
 * @param[in]     lda     Leading dimension. lda >= max(m,n).
 * @param[out]    tau     Array of dimension min(m,n).
 * @param[out]    work    Workspace.
 * @param[in]     lwork   Dimension of work.
 * @param[out]    rwork   Workspace, dimension max(m,n).
 * @param[out]    result  Array of dimension 2.
 */
void dlqt01(const int m, const int n,
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
    dgelqf(m, n, AF, lda, tau, work, lwork, &info);

    /* Copy details of Q: the upper triangle of AF (above diagonal) contains
     * the reflectors for LQ. Copy to Q. */
    dlaset("F", n, n, -1.0e+10, -1.0e+10, Q, lda);
    if (n > 1) {
        /* Copy upper trapezoid of AF (rows 0:m-1, cols 1:n-1) to Q */
        dlacpy("U", m, n - 1, &AF[0 + 1 * lda], lda, &Q[0 + 1 * lda], lda);
    }

    /* Generate the n-by-n matrix Q */
    dorglq(n, n, minmn, Q, lda, tau, work, lwork, &info);

    /* Copy L: lower triangle of AF */
    dlaset("F", m, n, 0.0, 0.0, L, lda);
    dlacpy("L", m, n, AF, lda, L, lda);

    /* Compute L - A*Q' */
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                m, n, n, -1.0, A, lda, Q, lda, 1.0, L, lda);

    /* Compute norm( L - A*Q' ) / ( N * norm(A) * EPS ) */
    f64 anorm = dlange("1", m, n, A, lda, rwork);
    f64 resid = dlange("1", m, n, L, lda, rwork);
    if (anorm > 0.0) {
        result[0] = ((resid / (f64)(n > 1 ? n : 1)) / anorm) / eps;
    } else {
        result[0] = 0.0;
    }

    /* Compute I - Q*Q' */
    dlaset("F", n, n, 0.0, 1.0, L, lda);
    cblas_dsyrk(CblasColMajor, CblasUpper, CblasNoTrans,
                n, n, -1.0, Q, lda, 1.0, L, lda);

    /* Compute norm( I - Q*Q' ) / ( N * EPS ) */
    resid = dlansy("1", "U", n, L, lda, rwork);
    result[1] = (resid / (f64)(n > 1 ? n : 1)) / eps;
}
