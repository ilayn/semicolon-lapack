/**
 * @file sget54.c
 * @brief SGET54 checks a generalized decomposition of the form A = U*S*V' and B = U*T*V'.
 */

#include "verify.h"
#include <cblas.h>
#include <math.h>

extern f32 slamch(const char* cmach);
extern f32 slange(const char* norm, const int m, const int n,
                  const f32* A, const int lda, f32* work);
extern void slacpy(const char* uplo, const int m, const int n,
                   const f32* A, const int lda, f32* B, const int ldb);

/**
 * SGET54 checks a generalized decomposition of the form
 *
 *          A = U*S*V'  and B = U*T* V'
 *
 * where ' means transpose and U and V are orthogonal.
 *
 * Specifically,
 *
 *  RESULT = ||( A - U*S*V', B - U*T*V' )|| / (||( A, B )||*n*ulp )
 *
 * @param[in]     n       The size of the matrix. If it is zero, SGET54 does nothing.
 * @param[in]     A       Double precision array, dimension (lda, n).
 *                        The original (unfactored) matrix A.
 * @param[in]     lda     The leading dimension of A.
 * @param[in]     B       Double precision array, dimension (ldb, n).
 *                        The original (unfactored) matrix B.
 * @param[in]     ldb     The leading dimension of B.
 * @param[in]     S       Double precision array, dimension (lds, n).
 *                        The factored matrix S.
 * @param[in]     lds     The leading dimension of S.
 * @param[in]     T       Double precision array, dimension (ldt, n).
 *                        The factored matrix T.
 * @param[in]     ldt     The leading dimension of T.
 * @param[in]     U       Double precision array, dimension (ldu, n).
 *                        The orthogonal matrix on the left-hand side.
 * @param[in]     ldu     The leading dimension of U.
 * @param[in]     V       Double precision array, dimension (ldv, n).
 *                        The orthogonal matrix on the right-hand side.
 * @param[in]     ldv     The leading dimension of V.
 * @param[out]    work    Double precision array, dimension (3*n*n).
 * @param[out]    result  The value RESULT. Limited to 1/ulp to avoid overflow.
 *                        Errors are flagged by RESULT=10/ulp.
 */
void sget54(const int n, const f32* A, const int lda,
            const f32* B, const int ldb,
            const f32* S, const int lds,
            const f32* T, const int ldt,
            const f32* U, const int ldu,
            const f32* V, const int ldv,
            f32* work, f32* result)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;

    *result = ZERO;
    if (n <= 0)
        return;

    /* Constants */

    f32 unfl = slamch("Safe minimum");
    f32 ulp = slamch("Epsilon") * slamch("Base");

    /* compute the norm of (A,B) */

    slacpy("Full", n, n, A, lda, work, n);
    slacpy("Full", n, n, B, ldb, work + (size_t)n * n, n);
    f32 abnorm = slange("1", n, 2 * n, work, n, NULL);
    if (abnorm < unfl) abnorm = unfl;

    /* Compute W1 = A - U*S*V', and put in the array WORK(0:N*N-1) */

    slacpy(" ", n, n, A, lda, work, n);
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                n, n, n, ONE, U, ldu, S, lds, ZERO,
                work + (size_t)n * n, n);

    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                n, n, n, -ONE, work + (size_t)n * n, n, V, ldv,
                ONE, work, n);

    /* Compute W2 = B - U*T*V', and put in WORK(N*N:2*N*N-1) */

    slacpy(" ", n, n, B, ldb, work + (size_t)n * n, n);
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                n, n, n, ONE, U, ldu, T, ldt, ZERO,
                work + 2 * (size_t)n * n, n);

    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                n, n, n, -ONE, work + 2 * (size_t)n * n, n, V, ldv,
                ONE, work + (size_t)n * n, n);

    /* Compute norm(W)/ ( ulp*norm((A,B)) ) */

    f32 wnorm = slange("1", n, 2 * n, work, n, NULL);

    if (abnorm > wnorm) {
        *result = (wnorm / abnorm) / (2 * n * ulp);
    } else {
        if (abnorm < ONE) {
            f32 tmp = wnorm;
            if (tmp > 2 * n * abnorm) tmp = 2 * n * abnorm;
            *result = (tmp / abnorm) / (2 * n * ulp);
        } else {
            f32 tmp = wnorm / abnorm;
            if (tmp > (f32)(2 * n)) tmp = (f32)(2 * n);
            *result = tmp / (2 * n * ulp);
        }
    }
}
