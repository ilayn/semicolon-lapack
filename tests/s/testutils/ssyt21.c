/**
 * @file ssyt21.c
 * @brief SSYT21 checks a decomposition of the form A = U S U'
 *        where A is symmetric, U is orthogonal, and S is diagonal or tridiagonal.
 *
 * Port of LAPACK's TESTING/EIG/ssyt21.f to C.
 */

#include <math.h>
#include "verify.h"
#include <cblas.h>

/* Forward declarations */
extern f32 slamch(const char* cmach);
extern f32 slansy(const char* norm, const char* uplo, const int n,
                     const f32* const restrict A, const int lda,
                     f32* const restrict work);
extern f32 slange(const char* norm, const int m, const int n,
                     const f32* const restrict A, const int lda,
                     f32* const restrict work);
extern void   slaset(const char* uplo, const int m, const int n,
                     const f32 alpha, const f32 beta,
                     f32* const restrict A, const int lda);
extern void   slacpy(const char* uplo, const int m, const int n,
                     const f32* const restrict A, const int lda,
                     f32* const restrict B, const int ldb);
extern void   slarfy(const char* uplo, const int n, const f32* const restrict V,
                     const int incv, const f32 tau,
                     f32* const restrict C, const int ldc,
                     f32* const restrict work);
extern void   sorm2r(const char* side, const char* trans, const int m, const int n,
                     const int k, const f32* const restrict A, const int lda,
                     const f32* const restrict tau,
                     f32* const restrict C, const int ldc,
                     f32* const restrict work, int* info);
extern void   sorm2l(const char* side, const char* trans, const int m, const int n,
                     const int k, const f32* const restrict A, const int lda,
                     const f32* const restrict tau,
                     f32* const restrict C, const int ldc,
                     f32* const restrict work, int* info);

/**
 * SSYT21 generally checks a decomposition of the form
 *
 *    A = U S U'
 *
 * where ' means transpose, A is symmetric, U is orthogonal, and S is
 * diagonal (if KBAND=0) or symmetric tridiagonal (if KBAND=1).
 *
 * If ITYPE=1, then U is represented as a dense matrix; otherwise U is
 * expressed as a product of Householder transformations, whose vectors
 * are stored in the array "V" and whose scaling constants are in "TAU".
 *
 * Specifically, if ITYPE=1, then:
 *    RESULT[0] = | A - U S U' | / ( |A| n ulp )
 *    RESULT[1] = | I - U U' | / ( n ulp )
 *
 * If ITYPE=2, then:
 *    RESULT[0] = | A - V S V' | / ( |A| n ulp )
 *
 * If ITYPE=3, then:
 *    RESULT[0] = | I - V U' | / ( n ulp )
 *
 * @param[in]     itype  Type of test.
 *                       1: U as dense matrix, test A - U S U' and I - U U'
 *                       2: U as Householder product V, test A - V S V'
 *                       3: Both U and V, test I - V U'
 * @param[in]     uplo   'U' for upper triangle, 'L' for lower triangle.
 * @param[in]     n      The size of the matrix. If zero, does nothing.
 * @param[in]     kband  The bandwidth of S. 0 = diagonal, 1 = tridiagonal.
 * @param[in]     A      The original symmetric matrix A, dimension (lda, n).
 * @param[in]     lda    Leading dimension of A. lda >= max(1, n).
 * @param[in]     D      Diagonal of S, dimension (n).
 * @param[in]     E      Off-diagonal of S, dimension (n-1). Not referenced if kband=0.
 * @param[in]     U      Orthogonal matrix U, dimension (ldu, n). Not referenced if itype=2.
 * @param[in]     ldu    Leading dimension of U. ldu >= max(1, n).
 * @param[in,out] V      Householder vectors, dimension (ldv, n). Modified temporarily if itype >= 2.
 * @param[in]     ldv    Leading dimension of V. ldv >= max(1, n).
 * @param[in]     tau    Householder scaling factors, dimension (n). Not referenced if itype < 2.
 * @param[out]    work   Workspace array, dimension (2*n*n).
 * @param[out]    result Test ratios, dimension (2). result[1] only set if itype=1.
 */
void ssyt21(const int itype, const char* uplo, const int n, const int kband,
            const f32* const restrict A, const int lda,
            const f32* const restrict D, const f32* const restrict E,
            const f32* const restrict U, const int ldu,
            f32* restrict V, const int ldv,
            const f32* const restrict tau,
            f32* const restrict work, f32* restrict result)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;
    const f32 TEN = 10.0f;

    int lower;
    char cuplo;
    int j, jcol, jrow, iinfo;
    f32 anorm, ulp, unfl, wnorm, vsave;

    result[0] = ZERO;
    if (itype == 1)
        result[1] = ZERO;
    if (n <= 0)
        return;

    if (uplo[0] == 'U' || uplo[0] == 'u') {
        lower = 0;
        cuplo = 'U';
    } else {
        lower = 1;
        cuplo = 'L';
    }

    unfl = slamch("S");
    ulp = slamch("E") * slamch("B");

    /* Error check */
    if (itype < 1 || itype > 3) {
        result[0] = TEN / ulp;
        return;
    }

    /* Norm of A */
    if (itype == 3) {
        anorm = ONE;
    } else {
        anorm = slansy("1", &cuplo, n, A, lda, work);
        if (anorm < unfl)
            anorm = unfl;
    }

    /* Compute error matrix */
    if (itype == 1) {
        /*
         * ITYPE=1: error = A - U S U'
         */
        slaset("F", n, n, ZERO, ZERO, work, n);
        slacpy(&cuplo, n, n, A, lda, work, n);

        /* Subtract U * diag(D) * U' */
        for (j = 0; j < n; j++) {
            cblas_ssyr(CblasColMajor,
                       lower ? CblasLower : CblasUpper,
                       n, -D[j], &U[j * ldu], 1, work, n);
        }

        /* If KBAND=1, subtract off-diagonal contribution */
        if (n > 1 && kband == 1) {
            for (j = 0; j < n - 1; j++) {
                cblas_ssyr2(CblasColMajor,
                            lower ? CblasLower : CblasUpper,
                            n, -E[j], &U[j * ldu], 1, &U[(j + 1) * ldu], 1,
                            work, n);
            }
        }
        wnorm = slansy("1", &cuplo, n, work, n, &work[n * n]);

    } else if (itype == 2) {
        /*
         * ITYPE=2: error = V S V' - A
         *
         * Build V S V' using the Householder vectors stored in V.
         * This is the inverse of the reduction: we reconstruct A from
         * the tridiagonal form and the Householder transformations.
         */
        slaset("F", n, n, ZERO, ZERO, work, n);

        if (lower) {
            work[n * n - 1] = D[n - 1];
            for (j = n - 2; j >= 0; j--) {
                if (kband == 1) {
                    /* Set off-diagonal elements */
                    work[(n + 1) * j + 1] = (ONE - tau[j]) * E[j];
                    for (int jr = j + 2; jr < n; jr++) {
                        work[j * n + jr] = -tau[j] * E[j] * V[jr + j * ldv];
                    }
                }

                vsave = V[(j + 1) + j * ldv];
                V[(j + 1) + j * ldv] = ONE;
                slarfy("L", n - j - 1, &V[(j + 1) + j * ldv], 1, tau[j],
                       &work[(n + 1) * (j + 1)], n, &work[n * n]);
                V[(j + 1) + j * ldv] = vsave;
                work[(n + 1) * j] = D[j];
            }
        } else {
            work[0] = D[0];
            for (j = 0; j < n - 1; j++) {
                if (kband == 1) {
                    /* Set off-diagonal elements */
                    work[(n + 1) * (j + 1) - 1] = (ONE - tau[j]) * E[j];
                    for (int jr = 0; jr < j; jr++) {
                        work[(j + 1) * n + jr] = -tau[j] * E[j] * V[jr + (j + 1) * ldv];
                    }
                }

                vsave = V[j + (j + 1) * ldv];
                V[j + (j + 1) * ldv] = ONE;
                slarfy("U", j + 1, &V[(j + 1) * ldv], 1, tau[j],
                       work, n, &work[n * n]);
                V[j + (j + 1) * ldv] = vsave;
                work[(n + 1) * (j + 1)] = D[j + 1];
            }
        }

        /* Subtract A */
        for (jcol = 0; jcol < n; jcol++) {
            if (lower) {
                for (jrow = jcol; jrow < n; jrow++) {
                    work[jrow + n * jcol] -= A[jrow + lda * jcol];
                }
            } else {
                for (jrow = 0; jrow <= jcol; jrow++) {
                    work[jrow + n * jcol] -= A[jrow + lda * jcol];
                }
            }
        }
        wnorm = slansy("1", &cuplo, n, work, n, &work[n * n]);

    } else if (itype == 3) {
        /*
         * ITYPE=3: error = U V' - I
         */
        if (n < 2)
            return;

        slacpy(" ", n, n, U, ldu, work, n);
        if (lower) {
            sorm2r("R", "T", n, n - 1, n - 1, &V[1], ldv, tau,
                   &work[n], n, &work[n * n], &iinfo);
        } else {
            sorm2l("R", "T", n, n - 1, n - 1, &V[ldv], ldv, tau,
                   work, n, &work[n * n], &iinfo);
        }
        if (iinfo != 0) {
            result[0] = TEN / ulp;
            return;
        }

        /* Subtract I from diagonal */
        for (j = 0; j < n; j++) {
            work[(n + 1) * j] -= ONE;
        }

        wnorm = slange("1", n, n, work, n, &work[n * n]);
    } else {
        wnorm = ZERO;
    }

    /* Compute result[0] */
    if (anorm > wnorm) {
        result[0] = (wnorm / anorm) / (n * ulp);
    } else {
        if (anorm < ONE) {
            f32 tmp = fminf(wnorm, (f32)n * anorm);
            result[0] = (tmp / anorm) / (n * ulp);
        } else {
            f32 tmp = fminf(wnorm / anorm, (f32)n);
            result[0] = tmp / (n * ulp);
        }
    }

    /* Test 2: U U' - I (only for itype=1) */
    if (itype == 1) {
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                    n, n, n, ONE, U, ldu, U, ldu, ZERO, work, n);

        for (j = 0; j < n; j++) {
            work[(n + 1) * j] -= ONE;
        }

        f32 tmp = slange("1", n, n, work, n, &work[n * n]);
        result[1] = fminf((f32)n, tmp) / (n * ulp);
    }
}
