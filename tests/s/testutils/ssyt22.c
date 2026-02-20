/**
 * @file ssyt22.c
 * @brief SSYT22 checks a partial decomposition of the form A U = U S
 *        where A is symmetric, the columns of U are orthonormal, and S is
 *        diagonal or symmetric tridiagonal.
 *
 * Port of LAPACK's TESTING/EIG/ssyt22.f to C.
 */

#include <math.h>
#include "verify.h"
#include <cblas.h>

/* Forward declarations */
extern f32 slamch(const char* cmach);
extern f32 slansy(const char* norm, const char* uplo, const int n,
                     const f32* const restrict A, const int lda,
                     f32* const restrict work);

/**
 * SSYT22 generally checks a decomposition of the form
 *
 *    A U = U S
 *
 * where A is symmetric, the columns of U are orthonormal, and S
 * is diagonal (if KBAND=0) or symmetric tridiagonal (if KBAND=1).
 *
 * If ITYPE=1, then U is represented as a dense matrix.
 *
 * Specifically, if ITYPE=1, then:
 *    RESULT[0] = | U' A U - S | / ( |A| m ulp )
 *    RESULT[1] = | I - U' U | / ( m ulp )
 *
 * @param[in]     itype  Type of test. Currently only itype=1 is implemented.
 * @param[in]     uplo   'U' for upper triangle, 'L' for lower triangle.
 * @param[in]     n      The size of the matrix A.
 * @param[in]     m      The number of columns of U to check.
 * @param[in]     kband  The bandwidth of S. 0 = diagonal, 1 = tridiagonal.
 * @param[in]     A      The original symmetric matrix A, dimension (lda, n).
 * @param[in]     lda    Leading dimension of A. lda >= max(1, n).
 * @param[in]     D      Diagonal of S, dimension (m).
 * @param[in]     E      Off-diagonal of S, dimension (m). E[0] is ignored,
 *                       E[j] is the (j-1,j) and (j,j-1) element for j >= 1.
 *                       Not referenced if kband=0.
 * @param[in]     U      Orthonormal matrix U, dimension (ldu, n).
 * @param[in]     ldu    Leading dimension of U. ldu >= max(1, n).
 * @param[in]     V      Not referenced in current implementation.
 * @param[in]     ldv    Leading dimension of V.
 * @param[in]     tau    Not referenced in current implementation.
 * @param[out]    work   Workspace array, dimension (2*n*n).
 * @param[out]    result Test ratios, dimension (2).
 */
void ssyt22(const int itype, const char* uplo, const int n, const int m,
            const int kband, const f32* const restrict A, const int lda,
            const f32* const restrict D, const f32* const restrict E,
            const f32* const restrict U, const int ldu,
            const f32* const restrict V, const int ldv,
            const f32* const restrict tau,
            f32* const restrict work, f32* restrict result)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;

    int j, jj, jj1, jj2, nn, nnp1;
    f32 anorm, ulp, unfl, wnorm;

    (void)V;    /* unused */
    (void)ldv;  /* unused */
    (void)tau;  /* unused */

    result[0] = ZERO;
    result[1] = ZERO;
    if (n <= 0 || m <= 0)
        return;

    unfl = slamch("S");
    ulp = slamch("P");

    /* Norm of A */
    anorm = slansy("1", uplo, n, A, lda, work);
    if (anorm < unfl)
        anorm = unfl;

    /*
     * Compute error matrix:
     * ITYPE=1: error = U' A U - S
     *
     * First compute WORK = A * U using DSYMM
     */
    cblas_ssymm(CblasColMajor, CblasLeft,
                (uplo[0] == 'U' || uplo[0] == 'u') ? CblasUpper : CblasLower,
                n, m, ONE, A, lda, U, ldu, ZERO, work, n);

    nn = n * n;
    nnp1 = nn;

    /* Compute WORK(nnp1:) = U' * WORK = U' * A * U */
    cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                m, m, n, ONE, U, ldu, work, n, ZERO, &work[nnp1], n);

    /* Subtract diagonal D from the result */
    for (j = 0; j < m; j++) {
        jj = nnp1 + j * n + j;
        work[jj] = work[jj] - D[j];
    }

    /* Subtract off-diagonal E if kband=1 */
    if (kband == 1 && n > 1) {
        for (j = 1; j < m; j++) {
            jj1 = nnp1 + j * n + j - 1;      /* (j-1, j) element */
            jj2 = nnp1 + (j - 1) * n + j;    /* (j, j-1) element */
            work[jj1] = work[jj1] - E[j - 1];
            work[jj2] = work[jj2] - E[j - 1];
        }
    }

    /* Compute norm of U' A U - S */
    wnorm = slansy("1", uplo, m, &work[nnp1], n, work);

    if (anorm > wnorm) {
        result[0] = (wnorm / anorm) / (m * ulp);
    } else {
        if (anorm < ONE) {
            f32 tmp = fminf(wnorm, (f32)m * anorm);
            result[0] = (tmp / anorm) / (m * ulp);
        } else {
            f32 tmp = fminf(wnorm / anorm, (f32)m);
            result[0] = tmp / (m * ulp);
        }
    }

    /*
     * Test 2: Compute U' U - I (only for itype=1)
     */
    if (itype == 1) {
        sort01("C", n, m, U, ldu, work, 2 * n * n, &result[1]);
    }
}
