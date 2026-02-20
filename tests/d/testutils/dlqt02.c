/**
 * @file dlqt02.c
 * @brief DLQT02 tests DORGLQ for generating a partial Q matrix from LQ.
 *
 * Given the LQ factorization of an m-by-n matrix A, DLQT02 generates
 * the orthogonal matrix Q defined by the factorization of the first k
 * rows of A; it compares L(0:k-1,0:m-1) with A(0:k-1,0:n-1)*Q(0:m-1,0:n-1)',
 * and checks that the rows of Q are orthonormal.
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
extern void dorglq(const int m, const int n, const int k,
                   f64* const restrict A, const int lda,
                   const f64* const restrict tau,
                   f64* const restrict work, const int lwork, int* info);

/**
 * @param[in]     m       Number of rows of Q to generate. m >= 0.
 * @param[in]     n       Number of columns of Q. n >= m >= 0.
 * @param[in]     k       Number of reflectors. m >= k >= 0.
 * @param[in]     A       The k-by-n original matrix (first k rows).
 * @param[in]     AF      LQ factorization from DGELQF.
 * @param[out]    Q       The m-by-n matrix Q.
 * @param[out]    L       Workspace, dimension (lda, m).
 * @param[in]     lda     Leading dimension >= n.
 * @param[in]     tau     Scalar factors from DGELQF, dimension m.
 * @param[out]    work    Workspace.
 * @param[in]     lwork   Dimension of work.
 * @param[out]    rwork   Workspace, dimension m.
 * @param[out]    result  Array of dimension 2.
 */
void dlqt02(const int m, const int n, const int k,
            const f64 * const restrict A,
            const f64 * const restrict AF,
            f64 * const restrict Q,
            f64 * const restrict L,
            const int lda,
            const f64 * const restrict tau,
            f64 * const restrict work, const int lwork,
            f64 * const restrict rwork,
            f64 * restrict result)
{
    f64 eps = dlamch("E");
    int info;

    /* Copy the first k rows of the factorization to Q */
    dlaset("F", m, n, -1.0e+10, -1.0e+10, Q, lda);
    if (n > 1) {
        dlacpy("U", k, n - 1, &AF[0 + 1 * lda], lda, &Q[0 + 1 * lda], lda);
    }

    /* Generate the first m rows of Q */
    dorglq(m, n, k, Q, lda, tau, work, lwork, &info);

    /* Copy L(0:k-1, 0:m-1) */
    dlaset("F", k, m, 0.0, 0.0, L, lda);
    dlacpy("L", k, m, AF, lda, L, lda);

    /* Compute L - A*Q' */
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                k, m, n, -1.0, A, lda, Q, lda, 1.0, L, lda);

    /* Compute norm( L - A*Q' ) / ( N * norm(A) * EPS ) */
    f64 anorm = dlange("1", k, n, A, lda, rwork);
    f64 resid = dlange("1", k, m, L, lda, rwork);
    if (anorm > 0.0) {
        result[0] = ((resid / (f64)(n > 1 ? n : 1)) / anorm) / eps;
    } else {
        result[0] = 0.0;
    }

    /* Compute I - Q*Q' */
    dlaset("F", m, m, 0.0, 1.0, L, lda);
    cblas_dsyrk(CblasColMajor, CblasUpper, CblasNoTrans,
                m, n, -1.0, Q, lda, 1.0, L, lda);

    /* Compute norm( I - Q*Q' ) / ( N * EPS ) */
    resid = dlansy("1", "U", m, L, lda, rwork);
    result[1] = (resid / (f64)(n > 1 ? n : 1)) / eps;
}
