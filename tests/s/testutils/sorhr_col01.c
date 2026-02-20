/**
 * @file sorhr_col01.c
 * @brief SORHR_COL01 tests SORGTSQR and SORHR_COL using SLATSQR, SGEMQRT.
 */

#include <stdlib.h>
#include <math.h>
#include "verify.h"
#include "test_rng.h"
#include <cblas.h>

extern f32 slamch(const char* cmach);
extern f32 slange(const char* norm, const int m, const int n,
                     const f32* const restrict A, const int lda,
                     f32* const restrict work);
extern f32 slansy(const char* norm, const char* uplo, const int n,
                     const f32* const restrict A, const int lda,
                     f32* const restrict work);
extern void slacpy(const char* uplo, const int m, const int n,
                   const f32* const restrict A, const int lda,
                   f32* const restrict B, const int ldb);
extern void slaset(const char* uplo, const int m, const int n,
                   const f32 alpha, const f32 beta,
                   f32* const restrict A, const int lda);
extern void slatsqr(const int m, const int n, const int mb, const int nb,
                    f32* const restrict A, const int lda,
                    f32* const restrict T, const int ldt,
                    f32* const restrict work, const int lwork, int* info);
extern void sorgtsqr(const int m, const int n, const int mb, const int nb,
                     f32* const restrict A, const int lda,
                     const f32* const restrict T, const int ldt,
                     f32* restrict work, const int lwork, int* info);
extern void sorhr_col(const int m, const int n, const int nb,
                      f32* const restrict A, const int lda,
                      f32* restrict T, const int ldt,
                      f32* restrict D, int* info);
extern void sgemqrt(const char* side, const char* trans,
                    const int m, const int n, const int k, const int nb,
                    const f32* const restrict V, const int ldv,
                    const f32* const restrict T, const int ldt,
                    f32* const restrict C, const int ldc,
                    f32* const restrict work, int* info);
/**
 * SORHR_COL01 tests SORGTSQR and SORHR_COL using SLATSQR, SGEMQRT.
 *
 * @param[in]  m       Number of rows in test matrix.
 * @param[in]  n       Number of columns in test matrix.
 * @param[in]  mb1     Number of rows in row block.
 * @param[in]  nb1     Number of columns in column block (input).
 * @param[in]  nb2     Number of columns in column block (output).
 * @param[out] result  Results of each of the six tests.
 */
void sorhr_col01(const int m, const int n, const int mb1, const int nb1,
                 const int nb2, f32* restrict result)
{
    f32 eps = slamch("E");
    int k = m < n ? m : n;
    int l = m > n ? m : n;
    if (l < 1) l = 1;
    int info;
    int j, i;
    f32 anorm, resid, cnorm, dnorm;
    uint64_t rng_state[4];
    rng_seed(rng_state, 1988198919901991ULL);

    f32* A = malloc(m * n * sizeof(f32));
    f32* AF = malloc(m * n * sizeof(f32));
    f32* Q = malloc(l * l * sizeof(f32));
    f32* R = malloc(m * l * sizeof(f32));
    f32* rwork = malloc(l * sizeof(f32));
    f32* C = malloc(m * n * sizeof(f32));
    f32* CF = malloc(m * n * sizeof(f32));
    f32* D = malloc(n * m * sizeof(f32));
    f32* DF = malloc(n * m * sizeof(f32));

    for (j = 0; j < n; j++) {
        slarnv_rng(2, m, &A[j * m], rng_state);
    }
    slacpy("F", m, n, A, m, AF, m);

    int nrb;
    if (mb1 - n > 0) {
        nrb = (m - n + mb1 - n - 1) / (mb1 - n);
        if (nrb < 1) nrb = 1;
    } else {
        nrb = 1;
    }

    f32* T1 = malloc(nb1 * n * nrb * sizeof(f32));
    f32* T2 = malloc(nb2 * n * sizeof(f32));
    f32* DIAG = malloc(n * sizeof(f32));

    int nb1_ub = nb1 < n ? nb1 : n;
    int nb2_ub = nb2 < n ? nb2 : n;

    f32 workquery;
    int lwork;

    slatsqr(m, n, mb1, nb1_ub, AF, m, T1, nb1, &workquery, -1, &info);
    lwork = (int)workquery;
    sorgtsqr(m, n, mb1, nb1, AF, m, T1, nb1, &workquery, -1, &info);
    int lwork2 = (int)workquery;
    if (lwork2 > lwork) lwork = lwork2;
    if (nb2_ub * n > lwork) lwork = nb2_ub * n;
    if (nb2_ub * m > lwork) lwork = nb2_ub * m;

    f32* work = malloc(lwork * sizeof(f32));

    slatsqr(m, n, mb1, nb1_ub, AF, m, T1, nb1, work, lwork, &info);

    slacpy("U", n, n, AF, m, R, m);

    sorgtsqr(m, n, mb1, nb1, AF, m, T1, nb1, work, lwork, &info);

    sorhr_col(m, n, nb2, AF, m, T2, nb2, DIAG, &info);

    slacpy("U", n, n, R, m, AF, m);

    for (i = 0; i < n; i++) {
        if (DIAG[i] == -1.0f) {
            cblas_sscal(n - i, -1.0f, &AF[i + i * m], m);
        }
    }

    slaset("F", m, m, 0.0f, 1.0f, Q, m);

    sgemqrt("L", "N", m, m, k, nb2_ub, AF, m, T2, nb2, Q, m, work, &info);

    slaset("F", m, n, 0.0f, 0.0f, R, m);

    slacpy("U", m, n, AF, m, R, m);

    cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                m, n, m, -1.0f, Q, m, A, m, 1.0f, R, m);

    anorm = slange("1", m, n, A, m, rwork);
    resid = slange("1", m, n, R, m, rwork);
    if (anorm > 0.0f) {
        result[0] = resid / (eps * (m > 1 ? m : 1) * anorm);
    } else {
        result[0] = 0.0f;
    }

    slaset("F", m, m, 0.0f, 1.0f, R, m);
    cblas_ssyrk(CblasColMajor, CblasUpper, CblasTrans,
                m, m, -1.0f, Q, m, 1.0f, R, m);
    resid = slansy("1", "U", m, R, m, rwork);
    result[1] = resid / (eps * (m > 1 ? m : 1));

    for (j = 0; j < n; j++) {
        slarnv_rng(2, m, &C[j * m], rng_state);
    }
    cnorm = slange("1", m, n, C, m, rwork);
    slacpy("F", m, n, C, m, CF, m);

    sgemqrt("L", "N", m, n, k, nb2_ub, AF, m, T2, nb2, CF, m, work, &info);

    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                m, n, m, -1.0f, Q, m, C, m, 1.0f, CF, m);
    resid = slange("1", m, n, CF, m, rwork);
    if (cnorm > 0.0f) {
        result[2] = resid / (eps * (m > 1 ? m : 1) * cnorm);
    } else {
        result[2] = 0.0f;
    }

    slacpy("F", m, n, C, m, CF, m);

    sgemqrt("L", "T", m, n, k, nb2_ub, AF, m, T2, nb2, CF, m, work, &info);

    cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                m, n, m, -1.0f, Q, m, C, m, 1.0f, CF, m);
    resid = slange("1", m, n, CF, m, rwork);
    if (cnorm > 0.0f) {
        result[3] = resid / (eps * (m > 1 ? m : 1) * cnorm);
    } else {
        result[3] = 0.0f;
    }

    for (j = 0; j < m; j++) {
        slarnv_rng(2, n, &D[j * n], rng_state);
    }
    dnorm = slange("1", n, m, D, n, rwork);
    slacpy("F", n, m, D, n, DF, n);

    sgemqrt("R", "N", n, m, k, nb2_ub, AF, m, T2, nb2, DF, n, work, &info);

    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                n, m, m, -1.0f, D, n, Q, m, 1.0f, DF, n);
    resid = slange("1", n, m, DF, n, rwork);
    if (dnorm > 0.0f) {
        result[4] = resid / (eps * (m > 1 ? m : 1) * dnorm);
    } else {
        result[4] = 0.0f;
    }

    slacpy("F", n, m, D, n, DF, n);

    sgemqrt("R", "T", n, m, k, nb2_ub, AF, m, T2, nb2, DF, n, work, &info);

    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                n, m, m, -1.0f, D, n, Q, m, 1.0f, DF, n);
    resid = slange("1", n, m, DF, n, rwork);
    if (dnorm > 0.0f) {
        result[5] = resid / (eps * (m > 1 ? m : 1) * dnorm);
    } else {
        result[5] = 0.0f;
    }

    free(A);
    free(AF);
    free(Q);
    free(R);
    free(rwork);
    free(work);
    free(T1);
    free(T2);
    free(DIAG);
    free(C);
    free(CF);
    free(D);
    free(DF);
}
