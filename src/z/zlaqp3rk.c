/**
 * @file zlaqp3rk.c
 * @brief ZLAQP3RK computes a step of truncated QR factorization with column pivoting of a complex matrix block using Level 3 BLAS.
 */

#include <complex.h>
#include <math.h>
#include "semicolon_cblas.h"
#include "semicolon_lapack_complex_double.h"

/**
 * ZLAQP3RK computes a step of truncated QR factorization with column
 * pivoting of a complex M-by-N matrix block A(IOFFSET+1:M,1:N)
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
 * in A(IOFFSET+1:M,N+1:N+NRHS) with Q(KB)**H * B.
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
 *          Complex*16 array, dimension (lda, n+nrhs).
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
 *          Complex*16 array, dimension (min(m-ioffset, n)).
 *
 * @param[in,out] vn1
 *          Double precision array, dimension (n). Partial column norms.
 *
 * @param[in,out] vn2
 *          Double precision array, dimension (n). Exact column norms.
 *
 * @param[out] auxv
 *          Complex*16 array, dimension (nb). Auxiliary vector.
 *
 * @param[out] F
 *          Complex*16 array, dimension (ldf, nb). Matrix F.
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
void zlaqp3rk(
    const INT m,
    const INT n,
    const INT nrhs,
    const INT ioffset,
    INT* nb,
    const f64 abstol,
    const f64 reltol,
    const INT kp1,
    const f64 maxc2nrm,
    c128* restrict A,
    const INT lda,
    INT* done,
    INT* KB,
    f64* maxc2nrmk,
    f64* relmaxc2nrmk,
    INT* restrict jpiv,
    c128* restrict tau,
    f64* restrict vn1,
    f64* restrict vn2,
    c128* restrict auxv,
    c128* restrict F,
    const INT ldf,
    INT* restrict iwork,
    INT* info)
{
    const c128 CZERO = CMPLX(0.0, 0.0);
    const c128 CONE = CMPLX(1.0, 0.0);
    const c128 NEG_CONE = CMPLX(-1.0, 0.0);

    INT itemp, j, k, minmnfact, minmnupdt, lsticc, kp, i = 0, iF;
    f64 hugeval, taunan, temp, temp2, tol3z;
    c128 aik;
    c128 neg_tau;
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
        k++;
        i = ioffset + k;

        if (i == 1) {

            kp = kp1;

        } else {

            kp = (k - 1) + cblas_idamax(n - k + 1, &vn1[k - 1], 1);

            *maxc2nrmk = vn1[kp];

            if (disnan(*maxc2nrmk)) {

                *done = 1;

                *KB = k - 1;
                iF = i - 1;
                *info = *KB + kp + 1;

                *relmaxc2nrmk = *maxc2nrmk;

                if (nrhs > 0 && *KB < (m - ioffset)) {
                    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasConjTrans,
                                m - iF, nrhs, *KB, &NEG_CONE, &A[iF + 0 * lda], lda,
                                &F[n + 0 * ldf], ldf, &CONE, &A[iF + n * lda], lda);
                }

                return;
            }

            if (*maxc2nrmk == 0.0) {

                *done = 1;

                *KB = k - 1;
                iF = i - 1;
                *relmaxc2nrmk = 0.0;

                if (nrhs > 0 && *KB < (m - ioffset)) {
                    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasConjTrans,
                                m - iF, nrhs, *KB, &NEG_CONE, &A[iF + 0 * lda], lda,
                                &F[n + 0 * ldf], ldf, &CONE, &A[iF + n * lda], lda);
                }

                for (j = k - 1; j < minmnfact; j++) {
                    tau[j] = CZERO;
                }

                return;

            }

            if (*info == 0 && *maxc2nrmk > hugeval) {
                *info = n + k - 1 + kp + 1;
            }

            *relmaxc2nrmk = *maxc2nrmk / maxc2nrm;

            if (*maxc2nrmk <= abstol || *relmaxc2nrmk <= reltol) {

                *done = 1;

                *KB = k - 1;
                iF = i - 1;

                if (*KB < minmnupdt) {
                    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasConjTrans,
                                m - iF, n + nrhs - *KB, *KB, &NEG_CONE, &A[iF + 0 * lda], lda,
                                &F[*KB + 0 * ldf], ldf, &CONE, &A[iF + (*KB) * lda], lda);
                }

                for (j = k - 1; j < minmnfact; j++) {
                    tau[j] = CZERO;
                }

                return;

            }

        }

        if (kp != k - 1) {
            cblas_zswap(m, &A[0 + kp * lda], 1, &A[0 + (k - 1) * lda], 1);
            cblas_zswap(k - 1, &F[kp + 0 * ldf], ldf, &F[(k - 1) + 0 * ldf], ldf);
            vn1[kp] = vn1[k - 1];
            vn2[kp] = vn2[k - 1];
            itemp = jpiv[kp];
            jpiv[kp] = jpiv[k - 1];
            jpiv[k - 1] = itemp;
        }

        if (k > 1) {
            for (j = 0; j < k - 1; j++) {
                F[(k - 1) + j * ldf] = conj(F[(k - 1) + j * ldf]);
            }
            cblas_zgemv(CblasColMajor, CblasNoTrans, m - i + 1, k - 1, &NEG_CONE,
                        &A[(i - 1) + 0 * lda], lda, &F[(k - 1) + 0 * ldf], ldf,
                        &CONE, &A[(i - 1) + (k - 1) * lda], 1);
            for (j = 0; j < k - 1; j++) {
                F[(k - 1) + j * ldf] = conj(F[(k - 1) + j * ldf]);
            }
        }

        if (i < m) {
            zlarfg(m - i + 1, &A[(i - 1) + (k - 1) * lda], &A[i + (k - 1) * lda], 1, &tau[k - 1]);
        } else {
            tau[k - 1] = CZERO;
        }

        if (disnan(creal(tau[k - 1]))) {
            taunan = creal(tau[k - 1]);
        } else if (disnan(cimag(tau[k - 1]))) {
            taunan = cimag(tau[k - 1]);
        } else {
            taunan = 0.0;
        }

        if (disnan(taunan)) {

            *done = 1;

            *KB = k - 1;
            iF = i - 1;
            *info = k;

            *maxc2nrmk = taunan;
            *relmaxc2nrmk = taunan;

            if (nrhs > 0 && *KB < (m - ioffset)) {
                cblas_zgemm(CblasColMajor, CblasNoTrans, CblasConjTrans,
                            m - iF, nrhs, *KB, &NEG_CONE, &A[iF + 0 * lda], lda,
                            &F[n + 0 * ldf], ldf, &CONE, &A[iF + n * lda], lda);
            }

            return;
        }

        aik = A[(i - 1) + (k - 1) * lda];
        A[(i - 1) + (k - 1) * lda] = CONE;

        if (k < n + nrhs) {
            cblas_zgemv(CblasColMajor, CblasConjTrans, m - i + 1, n + nrhs - k,
                        &tau[k - 1], &A[(i - 1) + k * lda], lda,
                        &A[(i - 1) + (k - 1) * lda], 1, &CZERO, &F[k + (k - 1) * ldf], 1);
        }

        for (j = 0; j < k; j++) {
            F[j + (k - 1) * ldf] = CZERO;
        }

        if (k > 1) {
            neg_tau = -tau[k - 1];
            cblas_zgemv(CblasColMajor, CblasConjTrans, m - i + 1, k - 1, &neg_tau,
                        &A[(i - 1) + 0 * lda], lda, &A[(i - 1) + (k - 1) * lda], 1,
                        &CZERO, &auxv[0], 1);

            cblas_zgemv(CblasColMajor, CblasNoTrans, n + nrhs, k - 1, &CONE,
                        &F[0 + 0 * ldf], ldf, &auxv[0], 1, &CONE, &F[0 + (k - 1) * ldf], 1);
        }

        if (k < n + nrhs) {
            cblas_zgemm(CblasColMajor, CblasNoTrans, CblasConjTrans,
                        1, n + nrhs - k, k, &NEG_CONE,
                        &A[(i - 1) + 0 * lda], lda,
                        &F[k + 0 * ldf], ldf,
                        &CONE, &A[(i - 1) + k * lda], lda);
        }

        A[(i - 1) + (k - 1) * lda] = aik;

        if (k < minmnfact) {

            for (j = k; j < n; j++) {
                if (vn1[j] != 0.0) {

                    temp = cabs(A[(i - 1) + j * lda]) / vn1[j];
                    temp = (1.0 + temp) * (1.0 - temp);
                    temp = (temp > 0.0) ? temp : 0.0;
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

    }

    *KB = k;
    iF = i;

    if (*KB < minmnupdt) {
        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasConjTrans,
                    m - iF, n + nrhs - *KB, *KB, &NEG_CONE, &A[iF + 0 * lda], lda,
                    &F[*KB + 0 * ldf], ldf, &CONE, &A[iF + (*KB) * lda], lda);
    }

    while (lsticc >= 0) {

        itemp = iwork[lsticc - 1];

        vn1[lsticc] = cblas_dznrm2(m - iF, &A[iF + lsticc * lda], 1);
        vn2[lsticc] = vn1[lsticc];

        lsticc = itemp;

    }
}
