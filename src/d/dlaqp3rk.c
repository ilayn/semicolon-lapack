/**
 * @file dlaqp3rk.c
 * @brief DLAQP3RK computes a step of truncated QR factorization with column pivoting of a real matrix block using Level 3 BLAS.
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "semicolon_lapack_double.h"

/**
 * DLAQP3RK computes a step of truncated QR factorization with column
 * pivoting of a real M-by-N matrix block A(IOFFSET+1:M,1:N)
 * by using Level 3 BLAS as
 *
 *   A * P(KB) = Q(KB) * R(KB).
 *
 * The routine tries to factorize NB columns from A starting from
 * the row IOFFSET+1 and updates the residual matrix with BLAS 3
 * xGEMM. The number of actually factorized columns is returned
 * is smaller than NB.
 *
 * Block A(1:IOFFSET,1:N) is accordingly pivoted, but not factorized.
 *
 * The routine also overwrites the right-hand-sides B matrix stored
 * in A(IOFFSET+1:M,N+1:N+NRHS) with Q(KB)**T * B.
 *
 * @param[in] m
 *          The number of rows of the matrix A. m >= 0.
 *
 * @param[in] n
 *          The number of columns of the matrix A. n >= 0.
 *
 * @param[in] nrhs
 *          The number of right hand sides. nrhs >= 0.
 *
 * @param[in] ioffset
 *          The number of rows that must be pivoted but not factorized.
 *
 * @param[in,out] nb
 *          On entry, the factorization block size. On exit, possibly reduced.
 *
 * @param[in] abstol
 *          The absolute tolerance for maximum column 2-norm.
 *
 * @param[in] reltol
 *          The relative tolerance for maximum column 2-norm.
 *
 * @param[in] kp1
 *          The index of the column with the maximum 2-norm (0-based).
 *
 * @param[in] maxc2nrm
 *          The maximum column 2-norm of the original matrix.
 *
 * @param[in,out] A
 *          Double precision array, dimension (lda, n+nrhs).
 *
 * @param[in] lda
 *          The leading dimension of the array A. lda >= max(1, m).
 *
 * @param[out] done
 *          1 if factorization is complete, 0 otherwise.
 *
 * @param[out] KB
 *          The number of factorized columns.
 *
 * @param[out] maxc2nrmk
 *          The maximum column 2-norm of the residual matrix.
 *
 * @param[out] relmaxc2nrmk
 *          The ratio maxc2nrmk / maxc2nrm.
 *
 * @param[out] jpiv
 *          Integer array, dimension (n). Column pivot indices.
 *
 * @param[out] tau
 *          Double precision array, dimension (min(m-ioffset, n)).
 *
 * @param[in,out] vn1
 *          Double precision array, dimension (n). Partial column norms.
 *
 * @param[in,out] vn2
 *          Double precision array, dimension (n). Exact column norms.
 *
 * @param[out] auxv
 *          Double precision array, dimension (nb). Auxiliary vector.
 *
 * @param[out] F
 *          Double precision array, dimension (ldf, nb). Matrix F.
 *
 * @param[in] ldf
 *          The leading dimension of the array F. ldf >= max(1, n+nrhs).
 *
 * @param[out] iwork
 *          Integer array, dimension (n-1).
 *
 * @param[out] info
 *                         - = 0: successful exit
 *                         - = j (1 <= j <= n): NaN detected in column j
 *                         - = j (n+1 <= j <= 2*n): Inf detected in column j-n
 */
void dlaqp3rk(
    const INT m,
    const INT n,
    const INT nrhs,
    const INT ioffset,
    INT* nb,
    const f64 abstol,
    const f64 reltol,
    const INT kp1,
    const f64 maxc2nrm,
    f64* restrict A,
    const INT lda,
    INT* done,
    INT* KB,
    f64* maxc2nrmk,
    f64* relmaxc2nrmk,
    INT* restrict jpiv,
    f64* restrict tau,
    f64* restrict vn1,
    f64* restrict vn2,
    f64* restrict auxv,
    f64* restrict F,
    const INT ldf,
    INT* restrict iwork,
    INT* info)
{
    INT itemp, j, k, minmnfact, minmnupdt, lsticc, kp, i, iF;
    f64 aik, hugeval, temp, temp2, tol3z;
    INT nb_val;

    *info = 0;

    minmnfact = (m - ioffset < n) ? (m - ioffset) : n;
    minmnupdt = (m - ioffset < n + nrhs) ? (m - ioffset) : (n + nrhs);
    nb_val = (*nb < minmnfact) ? *nb : minmnfact;
    *nb = nb_val;
    tol3z = sqrt(dlamch("E"));
    hugeval = dlamch("O");

    k = 0;
    lsticc = -1;
    *done = 0;

    while (k < nb_val && lsticc == -1) {
        i = ioffset + k;

        if (i == 0) {

            kp = kp1;

        } else {

            kp = k + cblas_idamax(n - k, &vn1[k], 1);

            *maxc2nrmk = vn1[kp];

            if (disnan(*maxc2nrmk)) {

                *done = 1;

                *KB = k;
                iF = i;
                *info = *KB + kp + 1;

                *relmaxc2nrmk = *maxc2nrmk;

                if (nrhs > 0 && *KB < (m - ioffset)) {
                    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                                m - iF, nrhs, *KB, -1.0, &A[iF + 0 * lda], lda,
                                &F[n + 0 * ldf], ldf, 1.0, &A[iF + n * lda], lda);
                }

                return;
            }

            if (*maxc2nrmk == 0.0) {

                *done = 1;

                *KB = k;
                iF = i;
                *relmaxc2nrmk = 0.0;

                if (nrhs > 0 && *KB < (m - ioffset)) {
                    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                                m - iF, nrhs, *KB, -1.0, &A[iF + 0 * lda], lda,
                                &F[n + 0 * ldf], ldf, 1.0, &A[iF + n * lda], lda);
                }

                for (j = k; j < minmnfact; j++) {
                    tau[j] = 0.0;
                }

                return;

            }

            if (*info == 0 && *maxc2nrmk > hugeval) {
                *info = n + k + kp + 1;
            }

            *relmaxc2nrmk = *maxc2nrmk / maxc2nrm;

            if (*maxc2nrmk <= abstol || *relmaxc2nrmk <= reltol) {

                *done = 1;

                *KB = k;
                iF = i;

                if (*KB < minmnupdt) {
                    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                                m - iF, n + nrhs - *KB, *KB, -1.0, &A[iF + 0 * lda], lda,
                                &F[*KB + 0 * ldf], ldf, 1.0, &A[iF + (*KB) * lda], lda);
                }

                for (j = k; j < minmnfact; j++) {
                    tau[j] = 0.0;
                }

                return;

            }

        }

        if (kp != k) {
            cblas_dswap(m, &A[0 + kp * lda], 1, &A[0 + k * lda], 1);
            cblas_dswap(k, &F[kp + 0 * ldf], ldf, &F[k + 0 * ldf], ldf);
            vn1[kp] = vn1[k];
            vn2[kp] = vn2[k];
            itemp = jpiv[kp];
            jpiv[kp] = jpiv[k];
            jpiv[k] = itemp;
        }

        if (k > 0) {
            cblas_dgemv(CblasColMajor, CblasNoTrans, m - i, k, -1.0,
                        &A[i + 0 * lda], lda, &F[k + 0 * ldf], ldf,
                        1.0, &A[i + k * lda], 1);
        }

        if (i < m - 1) {
            dlarfg(m - i, &A[i + k * lda], &A[i + 1 + k * lda], 1, &tau[k]);
        } else {
            tau[k] = 0.0;
        }

        if (disnan(tau[k])) {

            *done = 1;

            *KB = k;
            iF = i;
            *info = k + 1;

            *maxc2nrmk = tau[k];
            *relmaxc2nrmk = tau[k];

            if (nrhs > 0 && *KB < (m - ioffset)) {
                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                            m - iF, nrhs, *KB, -1.0, &A[iF + 0 * lda], lda,
                            &F[n + 0 * ldf], ldf, 1.0, &A[iF + n * lda], lda);
            }

            return;
        }

        aik = A[i + k * lda];
        A[i + k * lda] = 1.0;

        if (k < n + nrhs - 1) {
            cblas_dgemv(CblasColMajor, CblasTrans, m - i, n + nrhs - k - 1,
                        tau[k], &A[i + (k + 1) * lda], lda,
                        &A[i + k * lda], 1, 0.0, &F[k + 1 + k * ldf], 1);
        }

        for (j = 0; j <= k; j++) {
            F[j + k * ldf] = 0.0;
        }

        if (k > 0) {
            cblas_dgemv(CblasColMajor, CblasTrans, m - i, k, -tau[k],
                        &A[i + 0 * lda], lda, &A[i + k * lda], 1,
                        0.0, &auxv[0], 1);

            cblas_dgemv(CblasColMajor, CblasNoTrans, n + nrhs, k, 1.0,
                        &F[0 + 0 * ldf], ldf, &auxv[0], 1, 1.0, &F[0 + k * ldf], 1);
        }

        if (k < n + nrhs - 1) {
            cblas_dgemv(CblasColMajor, CblasNoTrans, n + nrhs - k - 1, k + 1, -1.0,
                        &F[k + 1 + 0 * ldf], ldf, &A[i + 0 * lda], lda,
                        1.0, &A[i + (k + 1) * lda], lda);
        }

        A[i + k * lda] = aik;

        if (k < minmnfact - 1) {

            for (j = k + 1; j < n; j++) {
                if (vn1[j] != 0.0) {

                    temp = fabs(A[i + j * lda]) / vn1[j];
                    temp = (1.0 + temp) * (1.0 - temp);
                    temp = (0.0 > temp) ? 0.0 : temp;
                    temp2 = temp * pow(vn1[j] / vn2[j], 2.0);
                    if (temp2 <= tol3z) {

                        iwork[j - 1] = lsticc;

                        lsticc = j;

                    } else {
                        vn1[j] = vn1[j] * sqrt(temp);
                    }
                }
            }

        }

        k++;
    }

    *KB = k;
    iF = ioffset + *KB;

    if (*KB < minmnupdt) {
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                    m - iF, n + nrhs - *KB, *KB, -1.0, &A[iF + 0 * lda], lda,
                    &F[*KB + 0 * ldf], ldf, 1.0, &A[iF + (*KB) * lda], lda);
    }

    while (lsticc >= 0) {

        itemp = iwork[lsticc - 1];

        vn1[lsticc] = cblas_dnrm2(m - iF, &A[iF + lsticc * lda], 1);
        vn2[lsticc] = vn1[lsticc];

        lsticc = itemp;

    }
}
