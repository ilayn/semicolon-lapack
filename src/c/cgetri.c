/**
 * @file cgetri.c
 * @brief Computes the inverse of a matrix using LU factorization.
 */

#include "internal_build_defs.h"
#include <complex.h>
#include <cblas.h>
#include "../include/lapack_tuning.h"
#include "semicolon_lapack_complex_single.h"

/**
 * CGETRI computes the inverse of a matrix using the LU factorization
 * computed by CGETRF.
 *
 * This method inverts U and then computes inv(A) by solving the system
 * inv(A)*L = inv(U) for inv(A).
 *
 * @param[in]     n     The order of the matrix A (n >= 0).
 * @param[in,out] A     On entry, the factors L and U from the factorization
 *                      A = P*L*U as computed by cgetrf.
 *                      On exit, if info = 0, the inverse of the original matrix A.
 *                      Array of dimension (lda, n).
 * @param[in]     lda   The leading dimension of the array A (lda >= max(1,n)).
 * @param[in]     ipiv  The pivot indices from cgetrf; row i was interchanged
 *                      with row ipiv[i]. Array of dimension n, 0-based.
 * @param[out]    work  Workspace array of dimension (max(1,lwork)).
 *                      On exit, if info=0, then work[0] returns the optimal lwork.
 * @param[in]     lwork The dimension of the array work (lwork >= max(1,n)).
 *                      For optimal performance lwork >= n*nb.
 *                      If lwork = -1, a workspace query is assumed.
 * @param[out]    info
 *                           Exit status:
 *                           - = 0: successful exit
 *                           - < 0: if info = -i, the i-th argument had an illegal value
 *                           - > 0: if info = i, U(i-1,i-1) is exactly zero; the matrix is
 *                           singular and its inverse could not be computed.
 */
void cgetri(
    const INT n,
    c64* restrict A,
    const INT lda,
    const INT* restrict ipiv,
    c64* restrict work,
    const INT lwork,
    INT* info)
{
    const c64 ZERO = CMPLXF(0.0f, 0.0f);
    const c64 ONE = CMPLXF(1.0f, 0.0f);
    const c64 NEG_ONE = CMPLXF(-1.0f, 0.0f);

    INT lquery;
    INT i, iws, j, jb, jj, jp, ldwork, lwkopt, nb, nbmin, nn;

    // Test the input parameters
    *info = 0;
    nb = lapack_get_nb("GETRI");
    lwkopt = (n > 1) ? n * nb : 1;
    work[0] = (c64)lwkopt;

    lquery = (lwork == -1);
    if (n < 0) {
        *info = -1;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -3;
    } else if (lwork < (n > 1 ? n : 1) && !lquery) {
        *info = -6;
    }

    if (*info != 0) {
        xerbla("CGETRI", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    // Quick return if possible
    if (n == 0) {
        return;
    }

    // Form inv(U). If info > 0 from ctrtri, then U is singular,
    // and the inverse is not computed.
    ctrtri("U", "N", n, A, lda, info);
    if (*info > 0) {
        return;
    }

    nbmin = 2;
    ldwork = n;
    if (nb > 1 && nb < n) {
        iws = (ldwork * nb > 1) ? ldwork * nb : 1;
        if (lwork < iws) {
            nb = lwork / ldwork;
            nbmin = 2;  // Could use ILAENV for NBMIN, but we use 2 as default
        }
    } else {
        iws = n;
    }

    // Solve the equation inv(A)*L = inv(U) for inv(A)
    if (nb < nbmin || nb >= n) {
        // Use unblocked code
        for (j = n - 1; j >= 0; j--) {
            // Copy current column of L to WORK and replace with zeros
            for (i = j + 1; i < n; i++) {
                work[i] = A[i + j * lda];
                A[i + j * lda] = ZERO;
            }

            // Compute current column of inv(A)
            if (j < n - 1) {
                cblas_cgemv(CblasColMajor, CblasNoTrans, n, n - j - 1, &NEG_ONE,
                            &A[(j + 1) * lda], lda, &work[j + 1], 1, &ONE,
                            &A[j * lda], 1);
            }
        }
    } else {
        // Use blocked code
        nn = ((n - 1) / nb) * nb;
        for (j = nn; j >= 0; j -= nb) {
            jb = (nb < n - j) ? nb : n - j;

            // Copy current block column of L to WORK and replace with zeros
            for (jj = j; jj < j + jb; jj++) {
                for (i = jj + 1; i < n; i++) {
                    work[i + (jj - j) * ldwork] = A[i + jj * lda];
                    A[i + jj * lda] = ZERO;
                }
            }

            // Compute current block column of inv(A)
            if (j + jb < n) {
                cblas_cgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                            n, jb, n - j - jb, &NEG_ONE,
                            &A[(j + jb) * lda], lda,
                            &work[j + jb], ldwork,
                            &ONE, &A[j * lda], lda);
            }
            cblas_ctrsm(CblasColMajor, CblasRight, CblasLower, CblasNoTrans, CblasUnit,
                        n, jb, &ONE, &work[j], ldwork, &A[j * lda], lda);
        }
    }

    // Apply column interchanges
    for (j = n - 2; j >= 0; j--) {
        jp = ipiv[j];
        if (jp != j) {
            cblas_cswap(n, &A[j * lda], 1, &A[jp * lda], 1);
        }
    }

    work[0] = (c64)iws;
}
