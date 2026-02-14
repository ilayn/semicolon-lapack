/**
 * @file dorhr_col02.c
 * @brief DORHR_COL02 tests DORGTSQR_ROW and DORHR_COL inside DGETSQRHRT.
 */

#include <stdlib.h>
#include <math.h>
#include "verify.h"
#include "test_rng.h"
#include <cblas.h>

extern f64 dlamch(const char* cmach);
extern f64 dlange(const char* norm, const int m, const int n,
                     const f64* const restrict A, const int lda,
                     f64* const restrict work);
extern f64 dlansy(const char* norm, const char* uplo, const int n,
                     const f64* const restrict A, const int lda,
                     f64* const restrict work);
extern void dlacpy(const char* uplo, const int m, const int n,
                   const f64* const restrict A, const int lda,
                   f64* const restrict B, const int ldb);
extern void dlaset(const char* uplo, const int m, const int n,
                   const f64 alpha, const f64 beta,
                   f64* const restrict A, const int lda);
extern void dgetsqrhrt(const int m, const int n, const int mb1, const int nb1,
                       const int nb2, f64* const restrict A, const int lda,
                       f64* restrict T, const int ldt,
                       f64* restrict work, const int lwork, int* info);
extern void dgemqrt(const char* side, const char* trans,
                    const int m, const int n, const int k, const int nb,
                    const f64* const restrict V, const int ldv,
                    const f64* const restrict T, const int ldt,
                    f64* const restrict C, const int ldc,
                    f64* const restrict work, int* info);
/**
 * DORHR_COL02 tests DORGTSQR_ROW and DORHR_COL inside DGETSQRHRT.
 *
 * @param[in]  m       Number of rows in test matrix.
 * @param[in]  n       Number of columns in test matrix.
 * @param[in]  mb1     Number of rows in row block.
 * @param[in]  nb1     Number of columns in column block (input).
 * @param[in]  nb2     Number of columns in column block (output).
 * @param[out] result  Results of each of the six tests.
 */
void dorhr_col02(const int m, const int n, const int mb1, const int nb1,
                 const int nb2, f64* restrict result)
{
    f64 eps = dlamch("E");
    int k = m < n ? m : n;
    int l = m > n ? m : n;
    if (l < 1) l = 1;
    int info;
    int j;
    f64 anorm, resid, cnorm, dnorm;
    uint64_t rng_state[4];
    rng_seed(rng_state, 1988198919901991ULL);

    f64* A = malloc(m * n * sizeof(f64));
    f64* AF = malloc(m * n * sizeof(f64));
    f64* Q = malloc(l * l * sizeof(f64));
    f64* R = malloc(m * l * sizeof(f64));
    f64* rwork = malloc(l * sizeof(f64));
    f64* C = malloc(m * n * sizeof(f64));
    f64* CF = malloc(m * n * sizeof(f64));
    f64* D = malloc(n * m * sizeof(f64));
    f64* DF = malloc(n * m * sizeof(f64));

    for (j = 0; j < n; j++) {
        dlarnv_rng(2, m, &A[j * m], rng_state);
    }
    dlacpy("F", m, n, A, m, AF, m);

    int nrb;
    if (mb1 - n > 0) {
        nrb = (m - n + mb1 - n - 1) / (mb1 - n);
        if (nrb < 1) nrb = 1;
    } else {
        nrb = 1;
    }

    f64* T1 = malloc(nb1 * n * nrb * sizeof(f64));
    f64* T2 = malloc(nb2 * n * sizeof(f64));
    f64* DIAG = malloc(n * sizeof(f64));

    int nb2_ub = nb2 < n ? nb2 : n;

    f64 workquery;
    int lwork;

    dgetsqrhrt(m, n, mb1, nb1, nb2, AF, m, T2, nb2, &workquery, -1, &info);
    lwork = (int)workquery;
    if (nb2_ub * n > lwork) lwork = nb2_ub * n;
    if (nb2_ub * m > lwork) lwork = nb2_ub * m;

    f64* work = malloc(lwork * sizeof(f64));

    dlacpy("F", m, n, A, m, AF, m);

    dgetsqrhrt(m, n, mb1, nb1, nb2, AF, m, T2, nb2, work, lwork, &info);

    dlaset("F", m, m, 0.0, 1.0, Q, m);

    dgemqrt("L", "N", m, m, k, nb2_ub, AF, m, T2, nb2, Q, m, work, &info);

    dlaset("F", m, n, 0.0, 0.0, R, m);

    dlacpy("U", m, n, AF, m, R, m);

    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                m, n, m, -1.0, Q, m, A, m, 1.0, R, m);

    anorm = dlange("1", m, n, A, m, rwork);
    resid = dlange("1", m, n, R, m, rwork);
    if (anorm > 0.0) {
        result[0] = resid / (eps * (m > 1 ? m : 1) * anorm);
    } else {
        result[0] = 0.0;
    }

    dlaset("F", m, m, 0.0, 1.0, R, m);
    cblas_dsyrk(CblasColMajor, CblasUpper, CblasTrans,
                m, m, -1.0, Q, m, 1.0, R, m);
    resid = dlansy("1", "U", m, R, m, rwork);
    result[1] = resid / (eps * (m > 1 ? m : 1));

    for (j = 0; j < n; j++) {
        dlarnv_rng(2, m, &C[j * m], rng_state);
    }
    cnorm = dlange("1", m, n, C, m, rwork);
    dlacpy("F", m, n, C, m, CF, m);

    dgemqrt("L", "N", m, n, k, nb2_ub, AF, m, T2, nb2, CF, m, work, &info);

    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                m, n, m, -1.0, Q, m, C, m, 1.0, CF, m);
    resid = dlange("1", m, n, CF, m, rwork);
    if (cnorm > 0.0) {
        result[2] = resid / (eps * (m > 1 ? m : 1) * cnorm);
    } else {
        result[2] = 0.0;
    }

    dlacpy("F", m, n, C, m, CF, m);

    dgemqrt("L", "T", m, n, k, nb2_ub, AF, m, T2, nb2, CF, m, work, &info);

    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                m, n, m, -1.0, Q, m, C, m, 1.0, CF, m);
    resid = dlange("1", m, n, CF, m, rwork);
    if (cnorm > 0.0) {
        result[3] = resid / (eps * (m > 1 ? m : 1) * cnorm);
    } else {
        result[3] = 0.0;
    }

    for (j = 0; j < m; j++) {
        dlarnv_rng(2, n, &D[j * n], rng_state);
    }
    dnorm = dlange("1", n, m, D, n, rwork);
    dlacpy("F", n, m, D, n, DF, n);

    dgemqrt("R", "N", n, m, k, nb2_ub, AF, m, T2, nb2, DF, n, work, &info);

    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                n, m, m, -1.0, D, n, Q, m, 1.0, DF, n);
    resid = dlange("1", n, m, DF, n, rwork);
    if (dnorm > 0.0) {
        result[4] = resid / (eps * (m > 1 ? m : 1) * dnorm);
    } else {
        result[4] = 0.0;
    }

    dlacpy("F", n, m, D, n, DF, n);

    dgemqrt("R", "T", n, m, k, nb2_ub, AF, m, T2, nb2, DF, n, work, &info);

    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                n, m, m, -1.0, D, n, Q, m, 1.0, DF, n);
    resid = dlange("1", n, m, DF, n, rwork);
    if (dnorm > 0.0) {
        result[5] = resid / (eps * (m > 1 ? m : 1) * dnorm);
    } else {
        result[5] = 0.0;
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
