/**
 * @file sqrt01p.c
 * @brief SQRT01P tests SGEQRFP and partially tests SORGQR.
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
extern void sgeqrfp(const int m, const int n,
                    f32* const restrict A, const int lda,
                    f32* const restrict tau,
                    f32* const restrict work, const int lwork, int* info);
extern void sorgqr(const int m, const int n, const int k,
                   f32* const restrict A, const int lda,
                   const f32* const restrict tau,
                   f32* const restrict work, const int lwork, int* info);

/**
 * SQRT01P tests SGEQRFP, which computes the QR factorization of an m-by-n
 * matrix A, and partially tests SORGQR which forms the m-by-m
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
void sqrt01p(const int m, const int n,
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

    /* Copy the matrix A to AF */
    slacpy("F", m, n, A, lda, AF, lda);

    /* Factorize A in AF using SGEQRFP (QR with non-negative diagonal R) */
    sgeqrfp(m, n, AF, lda, tau, work, lwork, &info);

    /* Copy details of Q: the lower triangle of AF (below diagonal) contains
     * the reflectors. Copy to Q. */
    slaset("F", m, m, -1.0e+10f, -1.0e+10f, Q, lda);
    if (m > 1) {
        /* Copy lower trapezoid of AF (rows 1:m-1, cols 0:min(m-1,n)-1) to Q */
        slacpy("L", m - 1, n, &AF[1 + 0 * lda], lda, &Q[1 + 0 * lda], lda);
    }

    /* Generate the m-by-m matrix Q */
    sorgqr(m, m, minmn, Q, lda, tau, work, lwork, &info);

    /* Copy R: upper triangle of AF */
    slaset("F", m, n, 0.0f, 0.0f, R, lda);
    slacpy("U", m, n, AF, lda, R, lda);

    /* Compute R - Q'*A */
    cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                m, n, m, -1.0f, Q, lda, A, lda, 1.0f, R, lda);

    /* Compute norm( R - Q'*A ) / ( M * norm(A) * EPS ) */
    f32 anorm = slange("1", m, n, A, lda, rwork);
    f32 resid = slange("1", m, n, R, lda, rwork);
    if (anorm > 0.0f) {
        result[0] = ((resid / (f32)(m > 1 ? m : 1)) / anorm) / eps;
    } else {
        result[0] = 0.0f;
    }

    /* Compute I - Q'*Q */
    slaset("F", m, m, 0.0f, 1.0f, R, lda);
    cblas_ssyrk(CblasColMajor, CblasUpper, CblasTrans,
                m, m, -1.0f, Q, lda, 1.0f, R, lda);

    /* Compute norm( I - Q'*Q ) / ( M * EPS ) */
    resid = slansy("1", "U", m, R, lda, rwork);
    result[1] = (resid / (f32)(m > 1 ? m : 1)) / eps;
}
