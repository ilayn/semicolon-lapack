/**
 * @file dlaqps.c
 * @brief DLAQPS computes a step of QR factorization with column pivoting
 *        of a real M-by-N matrix A by using BLAS level 3.
 */

#include <math.h>
#include <float.h>
#include "semicolon_cblas.h"
#include "semicolon_lapack_double.h"

/**
 * DLAQPS computes a step of QR factorization with column pivoting
 * of a real M-by-N matrix A by using Blas-3. It tries to factorize
 * NB columns from A starting from the row OFFSET+1, and updates all
 * of the matrix with Blas-3 xGEMM.
 *
 * In some cases, due to catastrophic cancellations, it cannot
 * factorize NB columns. Hence, the actual number of factorized
 * columns is returned in KB.
 *
 * Block A(0:OFFSET-1, 0:N-1) is accordingly pivoted, but not factorized.
 *
 * @param[in]     m       The number of rows of the matrix A. m >= 0.
 * @param[in]     n       The number of columns of the matrix A. n >= 0.
 * @param[in]     offset  The number of rows of A that have been factorized
 *                        in previous steps.
 * @param[in]     nb      The number of columns to factorize.
 * @param[out]    kb      The number of columns actually factorized.
 * @param[in,out] A       Double precision array, dimension (lda, n).
 *                        On entry, the M-by-N matrix A.
 *                        On exit, block A(offset:m-1, 0:kb-1) is the triangular
 *                        factor obtained and block A(0:offset-1, 0:n-1) has been
 *                        accordingly pivoted, but not factorized.
 *                        The rest of the matrix, block A(offset:m-1, kb:n-1) has
 *                        been updated.
 * @param[in]     lda     The leading dimension of the array A. lda >= max(1, m).
 * @param[in,out] jpvt    Integer array, dimension (n).
 *                        jpvt[i] = k <==> Column k of the full matrix A has been
 *                        permuted into position i in AP.
 * @param[out]    tau     Double precision array, dimension (kb).
 *                        The scalar factors of the elementary reflectors.
 * @param[in,out] vn1     Double precision array, dimension (n).
 *                        The vector with the partial column norms.
 * @param[in,out] vn2     Double precision array, dimension (n).
 *                        The vector with the exact column norms.
 * @param[in,out] auxv    Double precision array, dimension (nb).
 *                        Auxiliary vector.
 * @param[in,out] F       Double precision array, dimension (ldf, nb).
 *                        Matrix F**T = L*Y**T*A.
 * @param[in]     ldf     The leading dimension of the array F. ldf >= max(1, n).
 */
void dlaqps(const INT m, const INT n, const INT offset, const INT nb,
            INT* kb,
            f64* restrict A, const INT lda,
            INT* restrict jpvt,
            f64* restrict tau,
            f64* restrict vn1,
            f64* restrict vn2,
            f64* restrict auxv,
            f64* restrict F, const INT ldf)
{
    INT itemp, j, k, lastrk, lsticc, pvt, rk;
    f64 akk, temp, temp2, tol3z;

    lastrk = m < n + offset ? m : n + offset;
    lsticc = -1;
    k = 0;
    tol3z = sqrt(dlamch("E"));

    /*
     * Beginning of while loop.
     */
    while (k < nb && lsticc == -1) {
        rk = offset + k;

        /*
         * Determine ith pivot column and swap if necessary
         */
        pvt = k + (INT)cblas_idamax(n - k, &vn1[k], 1);
        if (pvt != k) {
            cblas_dswap(m, &A[pvt * lda], 1, &A[k * lda], 1);
            cblas_dswap(k, &F[pvt], ldf, &F[k], ldf);
            itemp = jpvt[pvt];
            jpvt[pvt] = jpvt[k];
            jpvt[k] = itemp;
            vn1[pvt] = vn1[k];
            vn2[pvt] = vn2[k];
        }

        /*
         * Apply previous Householder reflectors to column K:
         * A(RK:M, K) := A(RK:M, K) - A(RK:M, 1:K-1)*F(K, 1:K-1)**T.
         */
        if (k > 0) {
            cblas_dgemv(CblasColMajor, CblasNoTrans,
                        m - rk, k, -1.0,
                        &A[rk], lda,
                        &F[k], ldf,
                        1.0, &A[rk + k * lda], 1);
        }

        /*
         * Generate elementary reflector H(k).
         */
        if (rk < m - 1) {
            dlarfg(m - rk, &A[rk + k * lda],
                   &A[rk + 1 + k * lda], 1, &tau[k]);
        } else {
            dlarfg(1, &A[rk + k * lda],
                   &A[rk + k * lda], 1, &tau[k]);
        }

        akk = A[rk + k * lda];
        A[rk + k * lda] = 1.0;

        /*
         * Compute Kth column of F:
         * F(K+1:N, K) := tau(K)*A(RK:M, K+1:N)**T * A(RK:M, K).
         */
        if (k < n - 1) {
            cblas_dgemv(CblasColMajor, CblasTrans,
                        m - rk, n - k - 1, tau[k],
                        &A[rk + (k + 1) * lda], lda,
                        &A[rk + k * lda], 1,
                        0.0, &F[k + 1 + k * ldf], 1);
        }

        /*
         * Padding F(1:K, K) with zeros.
         */
        for (j = 0; j <= k; j++) {
            F[j + k * ldf] = 0.0;
        }

        /*
         * Incremental updating of F:
         * F(1:N, K) := F(1:N, K) - tau(K)*F(1:N, 1:K-1)*A(RK:M, 1:K-1)**T
         *              * A(RK:M, K).
         */
        if (k > 0) {
            cblas_dgemv(CblasColMajor, CblasTrans,
                        m - rk, k, -tau[k],
                        &A[rk], lda,
                        &A[rk + k * lda], 1,
                        0.0, auxv, 1);

            cblas_dgemv(CblasColMajor, CblasNoTrans,
                        n, k, 1.0,
                        F, ldf,
                        auxv, 1,
                        1.0, &F[k * ldf], 1);
        }

        /*
         * Update the current row of A:
         * A(RK, K+1:N) := A(RK, K+1:N) - A(RK, 1:K)*F(K+1:N, 1:K)**T.
         */
        if (k < n - 1) {
            cblas_dgemv(CblasColMajor, CblasNoTrans,
                        n - k - 1, k + 1, -1.0,
                        &F[k + 1], ldf,
                        &A[rk], lda,
                        1.0, &A[rk + (k + 1) * lda], lda);
        }

        /*
         * Update partial column norms.
         */
        if (rk < lastrk - 1) {
            for (j = k + 1; j < n; j++) {
                if (vn1[j] != 0.0) {
                    /*
                     * NOTE: The following 4 lines follow from the analysis in
                     * Lapack Working Note 176.
                     */
                    temp = fabs(A[rk + j * lda]) / vn1[j];
                    temp = (1.0 + temp) * (1.0 - temp);
                    if (temp < 0.0) temp = 0.0;
                    temp2 = temp * (vn1[j] / vn2[j]) * (vn1[j] / vn2[j]);
                    if (temp2 <= tol3z) {
                        vn2[j] = (f64)lsticc;
                        lsticc = j;
                    } else {
                        vn1[j] = vn1[j] * sqrt(temp);
                    }
                }
            }
        }

        A[rk + k * lda] = akk;
        k++;
    }
    *kb = k;
    rk = offset + *kb;

    /*
     * Apply the block reflector to the rest of the matrix:
     * A(OFFSET+KB+1:M, KB+1:N) := A(OFFSET+KB+1:M, KB+1:N) -
     *                     A(OFFSET+KB+1:M, 1:KB)*F(KB+1:N, 1:KB)**T.
     */
    if (*kb < (n < m - offset ? n : m - offset)) {
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                    m - rk, n - *kb, *kb, -1.0,
                    &A[rk], lda,
                    &F[*kb], ldf,
                    1.0, &A[rk + *kb * lda], lda);
    }

    /*
     * Recomputation of difficult columns.
     */
    while (lsticc >= 0) {
        itemp = (INT)round(vn2[lsticc]);
        vn1[lsticc] = cblas_dnrm2(m - rk, &A[rk + lsticc * lda], 1);

        /*
         * NOTE: The computation of VN1(LSTICC) relies on the fact that
         * SNRM2 does not fail on vectors with norm below the value of
         * SQRT(DLAMCH("S"))
         */
        vn2[lsticc] = vn1[lsticc];
        lsticc = itemp;
    }
}
