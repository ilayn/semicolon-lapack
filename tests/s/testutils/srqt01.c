/**
 * @file srqt01.c
 * @brief SRQT01 tests SGERQF and partially tests SORGRQ.
 *
 * Compares R with A*Q', and checks that Q is orthogonal.
 *
 * RESULT(0) = norm( R - A*Q' ) / ( N * norm(A) * EPS )
 * RESULT(1) = norm( I - Q*Q' ) / ( N * EPS )
 */

#include <stdlib.h>
#include <math.h>
#include "verify.h"
#include <cblas.h>

// Forward declarations
extern f32 slamch(const char* cmach);
extern f32 slange(const char* norm, const int m, const int n,
                     const f32* const restrict A, const int lda,
                     f32* const restrict work);
extern f32 slansy(const char* norm, const char* uplo, const int n,
                     const f32* const restrict A, const int lda,
                     f32* const restrict work);
extern void slacpy(const char* uplo, const int m, const int n,
                   const f32* const restrict A, const int lda,
                   f32* const restrict B, const int ldb);
extern void slaset(const char* uplo, const int m, const int n,
                   const f32 alpha, const f32 beta,
                   f32* const restrict A, const int lda);
extern void sgerqf(const int m, const int n,
                   f32* const restrict A, const int lda,
                   f32* const restrict tau,
                   f32* const restrict work, const int lwork, int* info);
extern void sorgrq(const int m, const int n, const int k,
                   f32* const restrict A, const int lda,
                   const f32* const restrict tau,
                   f32* const restrict work, const int lwork, int* info);

/**
 * SRQT01 tests SGERQF, which computes the RQ factorization of an m-by-n
 * matrix A, and partially tests SORGRQ which forms the n-by-n
 * orthogonal matrix Q.
 *
 * @param[in]     m       Number of rows of A. m >= 0.
 * @param[in]     n       Number of columns of A. n >= 0.
 * @param[in]     A       The m-by-n matrix A (original).
 * @param[in,out] AF      On entry, copy of A. On exit, RQ factorization.
 * @param[out]    Q       The n-by-n orthogonal matrix Q.
 * @param[out]    R       Workspace, dimension (lda, max(m,n)).
 * @param[in]     lda     Leading dimension. lda >= max(m,n).
 * @param[out]    tau     Array of dimension min(m,n).
 * @param[out]    work    Workspace.
 * @param[in]     lwork   Dimension of work.
 * @param[out]    rwork   Workspace, dimension max(m,n).
 * @param[out]    result  Array of dimension 2.
 */
void srqt01(const int m, const int n,
            const f32 * const restrict A,
            f32 * const restrict AF,
            f32 * const restrict Q,
            f32 * const restrict R,
            const int lda,
            f32 * const restrict tau,
            f32 * const restrict work, const int lwork,
            f32 * const restrict rwork,
            f32 * restrict result)
{
    int minmn = m < n ? m : n;
    f32 eps = slamch("E");
    int info;

    /* Copy A to AF */
    slacpy("F", m, n, A, lda, AF, lda);

    /* Factorize A in AF */
    sgerqf(m, n, AF, lda, tau, work, lwork, &info);

    /* Copy details of Q.
     * For RQ, reflectors are stored row-wise in the last minmn rows.
     * For m <= n: reflectors are in all m rows, to the left of the upper
     *             triangle of AF(0:m-1, n-m:n-1).
     * For m > n:  reflectors are in the last n rows, to the left of the
     *             upper triangle of AF(m-n:m-1, 0:n-1). */
    slaset("F", n, n, -1.0e+10f, -1.0e+10f, Q, lda);
    if (m <= n) {
        if (m > 0 && m < n) {
            /* Copy non-triangular rows: AF(0:m-1, 0:n-m-1) → Q(n-m:n-1, 0:n-m-1) */
            slacpy("F", m, n - m, AF, lda, &Q[(n - m) + 0 * lda], lda);
        }
        if (m > 1) {
            /* Copy lower triangle of AF(1:m-1, n-m:n-2) → Q(n-m+1:n-1, n-m:n-2) */
            slacpy("L", m - 1, m - 1, &AF[1 + (n - m) * lda], lda,
                   &Q[(n - m + 1) + (n - m) * lda], lda);
        }
    } else {
        /* m > n */
        if (n > 1) {
            /* Copy lower triangle of AF(m-n+1:m-1, 0:n-2) → Q(1:n-1, 0:n-2) */
            slacpy("L", n - 1, n - 1, &AF[(m - n + 1) + 0 * lda], lda,
                   &Q[1 + 0 * lda], lda);
        }
    }

    /* Generate the n-by-n matrix Q */
    sorgrq(n, n, minmn, Q, lda, tau, work, lwork, &info);

    /* Copy R: upper triangle of AF.
     * For m <= n: R is m-by-m in AF(0:m-1, n-m:n-1).
     * For m > n:  R is m-by-n, with the upper-right n-by-n part in AF(m-n:m-1, 0:n-1). */
    slaset("F", m, n, 0.0f, 0.0f, R, lda);
    if (m <= n) {
        if (m > 0) {
            slacpy("U", m, m, &AF[0 + (n - m) * lda], lda,
                   &R[0 + (n - m) * lda], lda);
        }
    } else {
        if (m > n && n > 0) {
            slacpy("F", m - n, n, AF, lda, R, lda);
        }
        if (n > 0) {
            slacpy("U", n, n, &AF[(m - n) + 0 * lda], lda,
                   &R[(m - n) + 0 * lda], lda);
        }
    }

    /* Compute R - A*Q' */
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                m, n, n, -1.0f, A, lda, Q, lda, 1.0f, R, lda);

    /* Compute norm( R - A*Q' ) / ( N * norm(A) * EPS ) */
    f32 anorm = slange("1", m, n, A, lda, rwork);
    f32 resid = slange("1", m, n, R, lda, rwork);
    if (anorm > 0.0f) {
        result[0] = ((resid / (f32)(n > 1 ? n : 1)) / anorm) / eps;
    } else {
        result[0] = 0.0f;
    }

    /* Compute I - Q*Q' */
    slaset("F", n, n, 0.0f, 1.0f, R, lda);
    cblas_ssyrk(CblasColMajor, CblasUpper, CblasNoTrans,
                n, n, -1.0f, Q, lda, 1.0f, R, lda);

    /* Compute norm( I - Q*Q' ) / ( N * EPS ) */
    resid = slansy("1", "U", n, R, lda, rwork);
    result[1] = (resid / (f32)(n > 1 ? n : 1)) / eps;
}
