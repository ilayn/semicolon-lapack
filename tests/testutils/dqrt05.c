/**
 * @file dqrt05.c
 * @brief DQRT05 tests DTPQRT and DTPMQRT.
 */

#include <stdlib.h>
#include <math.h>
#include "verify.h"
#include "test_rng.h"
#include <cblas.h>

extern double dlamch(const char* cmach);
extern double dlange(const char* norm, const int m, const int n,
                     const double* const restrict A, const int lda,
                     double* const restrict work);
extern double dlansy(const char* norm, const char* uplo, const int n,
                     const double* const restrict A, const int lda,
                     double* const restrict work);
extern void dlacpy(const char* uplo, const int m, const int n,
                   const double* const restrict A, const int lda,
                   double* const restrict B, const int ldb);
extern void dlaset(const char* uplo, const int m, const int n,
                   const double alpha, const double beta,
                   double* const restrict A, const int lda);
extern void dtpqrt(const int m, const int n, const int l, const int nb,
                   double* const restrict A, const int lda,
                   double* const restrict B, const int ldb,
                   double* const restrict T, const int ldt,
                   double* const restrict work, int* info);
extern void dgemqrt(const char* side, const char* trans,
                    const int m, const int n, const int k, const int nb,
                    const double* const restrict V, const int ldv,
                    const double* const restrict T, const int ldt,
                    double* const restrict C, const int ldc,
                    double* const restrict work, int* info);
extern void dtpmqrt(const char* side, const char* trans,
                    const int m, const int n, const int k, const int l, const int nb,
                    const double* const restrict V, const int ldv,
                    const double* const restrict T, const int ldt,
                    double* const restrict A, const int lda,
                    double* const restrict B, const int ldb,
                    double* const restrict work, int* info);
/**
 * DQRT05 tests DTPQRT and DTPMQRT.
 *
 * @param[in]  m       Number of rows in lower part of the test matrix.
 * @param[in]  n       Number of columns in test matrix.
 * @param[in]  l       The number of rows of the upper trapezoidal part
 *                     the lower test matrix. 0 <= L <= M.
 * @param[in]  nb      Block size of the test matrix. NB <= N.
 * @param[out] result  Results of each of the six tests:
 *                     result[0] = | A - Q R |
 *                     result[1] = | I - Q^H Q |
 *                     result[2] = | Q C - Q C |
 *                     result[3] = | Q^H C - Q^H C |
 *                     result[4] = | C Q - C Q |
 *                     result[5] = | C Q^H - C Q^H |
 */
void dqrt05(const int m, const int n, const int l, const int nb, double* restrict result)
{
    double eps = dlamch("E");
    int k = n;
    int m2 = m + n;
    int np1 = (m > 0) ? n : 0;
    int lwork = m2 * m2 * nb;
    int ldt = nb;
    int info;
    int j;
    double anorm, resid, cnorm, dnorm;
    uint64_t rng_state[4];
    rng_seed(rng_state, 1988198919901991ULL);

    double* A = malloc(m2 * n * sizeof(double));
    double* AF = malloc(m2 * n * sizeof(double));
    double* Q = malloc(m2 * m2 * sizeof(double));
    double* R = malloc(m2 * m2 * sizeof(double));
    double* rwork = malloc(m2 * sizeof(double));
    double* work = malloc(lwork * sizeof(double));
    double* T = malloc(nb * n * sizeof(double));
    double* C = malloc(m2 * n * sizeof(double));
    double* CF = malloc(m2 * n * sizeof(double));
    double* D = malloc(n * m2 * sizeof(double));
    double* DF = malloc(n * m2 * sizeof(double));

    dlaset("F", m2, n, 0.0, 0.0, A, m2);
    dlaset("F", nb, n, 0.0, 0.0, T, nb);

    for (j = 0; j < n; j++) {
        dlarnv_rng(2, j + 1, &A[j * m2], rng_state);
    }
    if (m > 0) {
        int ml = m - l;
        for (j = 0; j < n; j++) {
            if (ml > 0) {
                dlarnv_rng(2, ml, &A[n + j * m2], rng_state);
            }
        }
    }
    if (l > 0) {
        int start_row = n + m - l;
        for (j = 0; j < n; j++) {
            int len = (j + 1 < l) ? (j + 1) : l;
            dlarnv_rng(2, len, &A[start_row + j * m2], rng_state);
        }
    }

    dlacpy("F", m2, n, A, m2, AF, m2);

    dtpqrt(m, n, l, nb, AF, m2, &AF[np1], m2, T, ldt, work, &info);

    dlaset("F", m2, m2, 0.0, 1.0, Q, m2);
    dgemqrt("R", "N", m2, m2, k, nb, AF, m2, T, ldt, Q, m2, work, &info);

    dlaset("F", m2, n, 0.0, 0.0, R, m2);
    dlacpy("U", m2, n, AF, m2, R, m2);

    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                m2, n, m2, -1.0, Q, m2, A, m2, 1.0, R, m2);
    anorm = dlange("1", m2, n, A, m2, rwork);
    resid = dlange("1", m2, n, R, m2, rwork);
    if (anorm > 0.0) {
        result[0] = resid / (eps * anorm * (m2 > 1 ? m2 : 1));
    } else {
        result[0] = 0.0;
    }

    dlaset("F", m2, m2, 0.0, 1.0, R, m2);
    cblas_dsyrk(CblasColMajor, CblasUpper, CblasTrans,
                m2, m2, -1.0, Q, m2, 1.0, R, m2);
    resid = dlansy("1", "U", m2, R, m2, rwork);
    result[1] = resid / (eps * (m2 > 1 ? m2 : 1));

    for (j = 0; j < n; j++) {
        dlarnv_rng(2, m2, &C[j * m2], rng_state);
    }
    cnorm = dlange("1", m2, n, C, m2, rwork);
    dlacpy("F", m2, n, C, m2, CF, m2);

    dtpmqrt("L", "N", m, n, k, l, nb, &AF[np1], m2, T, ldt,
            CF, m2, &CF[np1], m2, work, &info);

    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                m2, n, m2, -1.0, Q, m2, C, m2, 1.0, CF, m2);
    resid = dlange("1", m2, n, CF, m2, rwork);
    if (cnorm > 0.0) {
        result[2] = resid / (eps * (m2 > 1 ? m2 : 1) * cnorm);
    } else {
        result[2] = 0.0;
    }

    dlacpy("F", m2, n, C, m2, CF, m2);

    dtpmqrt("L", "T", m, n, k, l, nb, &AF[np1], m2, T, ldt,
            CF, m2, &CF[np1], m2, work, &info);

    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                m2, n, m2, -1.0, Q, m2, C, m2, 1.0, CF, m2);
    resid = dlange("1", m2, n, CF, m2, rwork);
    if (cnorm > 0.0) {
        result[3] = resid / (eps * (m2 > 1 ? m2 : 1) * cnorm);
    } else {
        result[3] = 0.0;
    }

    for (j = 0; j < m2; j++) {
        dlarnv_rng(2, n, &D[j * n], rng_state);
    }
    dnorm = dlange("1", n, m2, D, n, rwork);
    dlacpy("F", n, m2, D, n, DF, n);

    dtpmqrt("R", "N", n, m, n, l, nb, &AF[np1], m2, T, ldt,
            DF, n, &DF[np1 * n], n, work, &info);

    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                n, m2, m2, -1.0, D, n, Q, m2, 1.0, DF, n);
    resid = dlange("1", n, m2, DF, n, rwork);
    if (dnorm > 0.0) {
        result[4] = resid / (eps * (m2 > 1 ? m2 : 1) * dnorm);
    } else {
        result[4] = 0.0;
    }

    dlacpy("F", n, m2, D, n, DF, n);

    dtpmqrt("R", "T", n, m, n, l, nb, &AF[np1], m2, T, ldt,
            DF, n, &DF[np1 * n], n, work, &info);

    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                n, m2, m2, -1.0, D, n, Q, m2, 1.0, DF, n);
    resid = dlange("1", n, m2, DF, n, rwork);
    if (dnorm > 0.0) {
        result[5] = resid / (eps * (m2 > 1 ? m2 : 1) * dnorm);
    } else {
        result[5] = 0.0;
    }

    free(A);
    free(AF);
    free(Q);
    free(R);
    free(rwork);
    free(work);
    free(T);
    free(C);
    free(CF);
    free(D);
    free(DF);
}
