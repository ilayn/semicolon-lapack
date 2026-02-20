/**
 * @file ssbt21.c
 * @brief SSBT21 checks a decomposition of the form A = U S U'
 *        where A is symmetric banded, U is orthogonal, and S is diagonal
 *        or symmetric tridiagonal.
 */

#include <math.h>
#include <cblas.h>
#include "verify.h"

extern f32 slamch(const char* cmach);
extern f32 slansb(const char* norm, const char* uplo, const int n,
                     const int k, const f32* AB, const int ldab, f32* work);
extern f32 slansp(const char* norm, const char* uplo, const int n,
                     const f32* AP, f32* work);
extern f32 slange(const char* norm, const int m, const int n,
                     const f32* A, const int lda, f32* work);

/**
 * SSBT21 generally checks a decomposition of the form
 *
 *    A = U S U'
 *
 * where ' means transpose, A is symmetric banded, U is orthogonal,
 * and S is diagonal (if KS=0) or symmetric tridiagonal (if KS=1).
 *
 * Specifically:
 *    RESULT[0] = | A - U S U' | / ( |A| n ulp )
 *    RESULT[1] = | I - U U' | / ( n ulp )
 *
 * @param[in]     uplo   'U' or 'L' for upper/lower triangle storage.
 * @param[in]     n      The size of the matrix.
 * @param[in]     ka     The bandwidth of A. Clamped to max(0, min(n-1, ka)).
 * @param[in]     ks     The bandwidth of S. 0 = diagonal, 1 = tridiagonal.
 * @param[in]     A      Symmetric banded matrix in SB format, dimension (lda, n).
 * @param[in]     lda    Leading dimension of A.
 * @param[in]     D      Diagonal of S, dimension (n).
 * @param[in]     E      Off-diagonal of S, dimension (n-1). Not referenced if ks=0.
 * @param[in]     U      Orthogonal matrix, dimension (ldu, n).
 * @param[in]     ldu    Leading dimension of U.
 * @param[out]    work   Workspace, dimension (n*n + n).
 * @param[out]    result Test ratios, dimension (2).
 */
void ssbt21(const char* uplo, const int n, const int ka, const int ks,
            const f32* A, const int lda,
            const f32* D, const f32* E,
            const f32* U, const int ldu,
            f32* work, f32* result)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;

    int lower;
    char cuplo;
    int ika, j, jc, jr, lw;
    f32 anorm, ulp, unfl, wnorm;
    CBLAS_UPLO cblas_uplo;

    /* Constants */

    result[0] = ZERO;
    result[1] = ZERO;
    if (n <= 0)
        return;

    ika = ka;
    if (ika > n - 1)
        ika = n - 1;
    if (ika < 0)
        ika = 0;
    lw = (n * (n + 1)) / 2;

    if (uplo[0] == 'U' || uplo[0] == 'u') {
        lower = 0;
        cuplo = 'U';
        cblas_uplo = CblasUpper;
    } else {
        lower = 1;
        cuplo = 'L';
        cblas_uplo = CblasLower;
    }

    unfl = slamch("S");
    ulp = slamch("E") * slamch("B");

    /* Do Test 1 */

    /* Norm of A: */

    anorm = slansb("1", &cuplo, n, ika, A, lda, work);
    if (anorm < unfl)
        anorm = unfl;

    /* Compute error matrix:    Error = A - U S U'

       Copy A from SB to SP storage format. */

    j = 0;
    for (jc = 0; jc < n; jc++) {
        if (lower) {
            int nelem = ika + 1;
            if (nelem > n - jc)
                nelem = n - jc;
            for (jr = 0; jr < nelem; jr++) {
                work[j] = A[jr + jc * lda];
                j++;
            }
            for (jr = nelem; jr < n - jc; jr++) {
                work[j] = ZERO;
                j++;
            }
        } else {
            int nzeros = (jc > ika) ? jc - ika : 0;
            for (jr = 0; jr < nzeros; jr++) {
                work[j] = ZERO;
                j++;
            }
            int top = (ika < jc) ? ika : jc;
            for (jr = top; jr >= 0; jr--) {
                work[j] = A[ika - jr + jc * lda];
                j++;
            }
        }
    }

    for (j = 0; j < n; j++) {
        cblas_sspr(CblasColMajor, cblas_uplo, n, -D[j], &U[j * ldu], 1, work);
    }

    if (n > 1 && ks == 1) {
        for (j = 0; j < n - 1; j++) {
            cblas_sspr2(CblasColMajor, cblas_uplo, n, -E[j],
                        &U[j * ldu], 1, &U[(j + 1) * ldu], 1, work);
        }
    }
    wnorm = slansp("1", &cuplo, n, work, &work[lw]);

    if (anorm > wnorm) {
        result[0] = (wnorm / anorm) / (n * ulp);
    } else {
        if (anorm < ONE) {
            f32 tmp = wnorm;
            if (tmp > (f32)n * anorm)
                tmp = (f32)n * anorm;
            result[0] = (tmp / anorm) / (n * ulp);
        } else {
            f32 tmp = wnorm / anorm;
            if (tmp > (f32)n)
                tmp = (f32)n;
            result[0] = tmp / (n * ulp);
        }
    }

    /* Do Test 2 */

    /* Compute  U U' - I */

    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                n, n, n, ONE, U, ldu, U, ldu, ZERO, work, n);

    for (j = 0; j < n; j++) {
        work[(n + 1) * j] -= ONE;
    }

    {
        f32 tmp = slange("1", n, n, work, n, &work[n * n]);
        if (tmp > (f32)n)
            tmp = (f32)n;
        result[1] = tmp / (n * ulp);
    }
}
