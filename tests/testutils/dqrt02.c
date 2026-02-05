/**
 * @file dqrt02.c
 * @brief DQRT02 tests DORGQR for generating a partial Q matrix.
 *
 * Given the QR factorization of an m-by-n matrix A, DQRT02 generates
 * the orthogonal matrix Q defined by the factorization of the first k
 * columns of A; it compares R(0:n-1,0:k-1) with Q(0:m-1,0:n-1)'*A(0:m-1,0:k-1),
 * and checks that the columns of Q are orthonormal.
 *
 * RESULT(0) = norm( R - Q'*A ) / ( M * norm(A) * EPS )
 * RESULT(1) = norm( I - Q'*Q ) / ( M * EPS )
 */

#include <stdlib.h>
#include <math.h>
#include "verify.h"
#include <cblas.h>

// Forward declarations
extern double dlamch(const char* cmach);
extern double dlange(const char* norm, const int m, const int n,
                     const double* const restrict A, const int lda,
                     double* const restrict work);
extern double dlansy(const char* norm, const char* uplo, const int n,
                     const double* const restrict A, const int lda,
                     double* const restrict work);
extern void dlacpy(const char* uplo, const int m, const int n,
                   const double* const restrict A, const int lda,
                   double* const restrict B, const int ldb);
extern void dlaset(const char* uplo, const int m, const int n,
                   const double alpha, const double beta,
                   double* const restrict A, const int lda);
extern void dorgqr(const int m, const int n, const int k,
                   double* const restrict A, const int lda,
                   const double* const restrict tau,
                   double* const restrict work, const int lwork, int* info);

/**
 * @param[in]     m       Number of rows of Q to be generated. m >= 0.
 * @param[in]     n       Number of columns of Q. m >= n >= 0.
 * @param[in]     k       Number of reflectors. n >= k >= 0.
 * @param[in]     A       The m-by-n original matrix.
 * @param[in]     AF      The QR factorization from DGEQRF.
 * @param[out]    Q       The m-by-n matrix Q.
 * @param[out]    R       Workspace, dimension (lda, n).
 * @param[in]     lda     Leading dimension >= m.
 * @param[in]     tau     Array of dimension n. Scalar factors from DGEQRF.
 * @param[out]    work    Workspace.
 * @param[in]     lwork   Dimension of work.
 * @param[out]    rwork   Workspace, dimension m.
 * @param[out]    result  Array of dimension 2.
 */
void dqrt02(const int m, const int n, const int k,
            const double * const restrict A,
            const double * const restrict AF,
            double * const restrict Q,
            double * const restrict R,
            const int lda,
            const double * const restrict tau,
            double * const restrict work, const int lwork,
            double * const restrict rwork,
            double * restrict result)
{
    double eps = dlamch("E");
    int info;

    /* Copy the first k columns of the factorization to Q */
    dlaset("F", m, n, -1.0e+10, -1.0e+10, Q, lda);
    if (m > 1) {
        dlacpy("L", m - 1, k, &AF[1 + 0 * lda], lda, &Q[1 + 0 * lda], lda);
    }

    /* Generate the first n columns of Q */
    dorgqr(m, n, k, Q, lda, tau, work, lwork, &info);

    /* Copy R(0:n-1, 0:k-1) */
    dlaset("F", n, k, 0.0, 0.0, R, lda);
    dlacpy("U", n, k, AF, lda, R, lda);

    /* Compute R - Q'*A(0:m-1, 0:k-1) */
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                n, k, m, -1.0, Q, lda, A, lda, 1.0, R, lda);

    /* Compute norm( R - Q'*A ) / ( M * norm(A) * EPS ) */
    double anorm = dlange("1", m, k, A, lda, rwork);
    double resid = dlange("1", n, k, R, lda, rwork);
    if (anorm > 0.0) {
        result[0] = ((resid / (double)(m > 1 ? m : 1)) / anorm) / eps;
    } else {
        result[0] = 0.0;
    }

    /* Compute I - Q'*Q */
    dlaset("F", n, n, 0.0, 1.0, R, lda);
    cblas_dsyrk(CblasColMajor, CblasUpper, CblasTrans,
                n, m, -1.0, Q, lda, 1.0, R, lda);

    /* Compute norm( I - Q'*Q ) / ( M * EPS ) */
    resid = dlansy("1", "U", n, R, lda, rwork);
    result[1] = (resid / (double)(m > 1 ? m : 1)) / eps;
}
