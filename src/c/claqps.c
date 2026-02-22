/**
 * @file claqps.c
 * @brief CLAQPS computes a step of QR factorization with column pivoting
 *        of a complex M-by-N matrix A by using BLAS level 3.
 */

#include "internal_build_defs.h"
#include <math.h>
#include <float.h>
#include <complex.h>
#include <cblas.h>
#include "semicolon_lapack_complex_single.h"

/**
 * CLAQPS computes a step of QR factorization with column pivoting
 * of a complex M-by-N matrix A by using Blas-3. It tries to factorize
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
 * @param[in,out] A       Complex*16 array, dimension (lda, n).
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
 * @param[out]    tau     Complex*16 array, dimension (kb).
 *                        The scalar factors of the elementary reflectors.
 * @param[in,out] vn1     Single precision array, dimension (n).
 *                        The vector with the partial column norms.
 * @param[in,out] vn2     Single precision array, dimension (n).
 *                        The vector with the exact column norms.
 * @param[in,out] auxv    Complex*16 array, dimension (nb).
 *                        Auxiliary vector.
 * @param[in,out] F       Complex*16 array, dimension (ldf, nb).
 *                        Matrix F**H = L*Y**H*A.
 * @param[in]     ldf     The leading dimension of the array F. ldf >= max(1, n).
 */
void claqps(const INT m, const INT n, const INT offset, const INT nb,
            INT* kb,
            c64* restrict A, const INT lda,
            INT* restrict jpvt,
            c64* restrict tau,
            f32* restrict vn1,
            f32* restrict vn2,
            c64* restrict auxv,
            c64* restrict F, const INT ldf)
{
    const c64 CZERO = CMPLXF(0.0f, 0.0f);
    const c64 CONE = CMPLXF(1.0f, 0.0f);
    const c64 NEG_CONE = CMPLXF(-1.0f, 0.0f);

    INT itemp, j, k, lastrk, lsticc, pvt, rk;
    f32 temp, temp2, tol3z;
    c64 akk;
    c64 neg_tau;

    lastrk = m < n + offset ? m : n + offset;
    lsticc = 0;
    k = 0;
    tol3z = sqrtf(FLT_EPSILON);

    /*
     * Beginning of while loop.
     */
    while (k < nb && lsticc == 0) {
        k++;
        rk = offset + k;

        /*
         * Determine ith pivot column and swap if necessary.
         * pvt is 0-based column index.
         */
        pvt = (k - 1) + (INT)cblas_isamax(n - k + 1, &vn1[k - 1], 1);
        if (pvt != k - 1) {
            cblas_cswap(m, &A[pvt * lda], 1, &A[(k - 1) * lda], 1);
            cblas_cswap(k - 1, &F[pvt], ldf, &F[k - 1], ldf);
            itemp = jpvt[pvt];
            jpvt[pvt] = jpvt[k - 1];
            jpvt[k - 1] = itemp;
            vn1[pvt] = vn1[k - 1];
            vn2[pvt] = vn2[k - 1];
        }

        /*
         * Apply previous Householder reflectors to column K:
         * A(RK:M, K) := A(RK:M, K) - A(RK:M, 1:K-1)*F(K, 1:K-1)**H.
         */
        if (k > 1) {
            for (j = 0; j < k - 1; j++) {
                F[(k - 1) + j * ldf] = conjf(F[(k - 1) + j * ldf]);
            }
            cblas_cgemv(CblasColMajor, CblasNoTrans,
                        m - rk + 1, k - 1, &NEG_CONE,
                        &A[(rk - 1)], lda,
                        &F[(k - 1)], ldf,
                        &CONE, &A[(rk - 1) + (k - 1) * lda], 1);
            for (j = 0; j < k - 1; j++) {
                F[(k - 1) + j * ldf] = conjf(F[(k - 1) + j * ldf]);
            }
        }

        /*
         * Generate elementary reflector H(k).
         */
        if (rk < m) {
            clarfg(m - rk + 1, &A[(rk - 1) + (k - 1) * lda],
                   &A[rk + (k - 1) * lda], 1, &tau[k - 1]);
        } else {
            clarfg(1, &A[(rk - 1) + (k - 1) * lda],
                   &A[(rk - 1) + (k - 1) * lda], 1, &tau[k - 1]);
        }

        akk = A[(rk - 1) + (k - 1) * lda];
        A[(rk - 1) + (k - 1) * lda] = CONE;

        /*
         * Compute Kth column of F:
         * F(K+1:N, K) := tau(K)*A(RK:M, K+1:N)**H * A(RK:M, K).
         */
        if (k < n) {
            cblas_cgemv(CblasColMajor, CblasConjTrans,
                        m - rk + 1, n - k, &tau[k - 1],
                        &A[(rk - 1) + k * lda], lda,
                        &A[(rk - 1) + (k - 1) * lda], 1,
                        &CZERO, &F[k + (k - 1) * ldf], 1);
        }

        /*
         * Padding F(1:K, K) with zeros.
         */
        for (j = 0; j < k; j++) {
            F[j + (k - 1) * ldf] = CZERO;
        }

        /*
         * Incremental updating of F:
         * F(1:N, K) := F(1:N, K) - tau(K)*F(1:N, 1:K-1)*A(RK:M, 1:K-1)**H
         *              * A(RK:M, K).
         */
        if (k > 1) {
            neg_tau = -tau[k - 1];
            cblas_cgemv(CblasColMajor, CblasConjTrans,
                        m - rk + 1, k - 1, &neg_tau,
                        &A[(rk - 1)], lda,
                        &A[(rk - 1) + (k - 1) * lda], 1,
                        &CZERO, auxv, 1);

            cblas_cgemv(CblasColMajor, CblasNoTrans,
                        n, k - 1, &CONE,
                        F, ldf,
                        auxv, 1,
                        &CONE, &F[(k - 1) * ldf], 1);
        }

        /*
         * Update the current row of A:
         * A(RK, K+1:N) := A(RK, K+1:N) - A(RK, 1:K)*F(K+1:N, 1:K)**H.
         */
        if (k < n) {
            cblas_cgemm(CblasColMajor, CblasNoTrans, CblasConjTrans,
                        1, n - k, k, &NEG_CONE,
                        &A[(rk - 1)], lda,
                        &F[k], ldf,
                        &CONE, &A[(rk - 1) + k * lda], lda);
        }

        /*
         * Update partial column norms.
         */
        if (rk < lastrk) {
            for (j = k + 1; j <= n; j++) {
                if (vn1[j - 1] != 0.0f) {
                    /*
                     * NOTE: The following 4 lines follow from the analysis in
                     * Lapack Working Note 176.
                     */
                    temp = cabsf(A[(rk - 1) + (j - 1) * lda]) / vn1[j - 1];
                    temp = (1.0f + temp) * (1.0f - temp);
                    if (temp < 0.0f) temp = 0.0f;
                    temp2 = temp * (vn1[j - 1] / vn2[j - 1]) * (vn1[j - 1] / vn2[j - 1]);
                    if (temp2 <= tol3z) {
                        vn2[j - 1] = (f32)lsticc;
                        lsticc = j;
                    } else {
                        vn1[j - 1] = vn1[j - 1] * sqrtf(temp);
                    }
                }
            }
        }

        A[(rk - 1) + (k - 1) * lda] = akk;

    /* End of while loop. */
    }
    *kb = k;
    rk = offset + *kb;

    /*
     * Apply the block reflector to the rest of the matrix:
     * A(OFFSET+KB+1:M, KB+1:N) := A(OFFSET+KB+1:M, KB+1:N) -
     *                     A(OFFSET+KB+1:M, 1:KB)*F(KB+1:N, 1:KB)**H.
     */
    if (*kb < (n < m - offset ? n : m - offset)) {
        cblas_cgemm(CblasColMajor, CblasNoTrans, CblasConjTrans,
                    m - rk, n - *kb, *kb, &NEG_CONE,
                    &A[rk], lda,
                    &F[*kb], ldf,
                    &CONE, &A[rk + *kb * lda], lda);
    }

    /*
     * Recomputation of difficult columns.
     */
    while (lsticc > 0) {
        itemp = (INT)roundf(vn2[lsticc - 1]);
        vn1[lsticc - 1] = cblas_scnrm2(m - rk, &A[rk + (lsticc - 1) * lda], 1);

        /*
         * NOTE: The computation of VN1(LSTICC) relies on the fact that
         * SNRM2 does not fail on vectors with norm below the value of
         * SQRT(SLAMCH("S"))
         */
        vn2[lsticc - 1] = vn1[lsticc - 1];
        lsticc = itemp;
    }
}
