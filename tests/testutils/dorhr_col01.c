/**
 * @file dorhr_col01.c
 * @brief DORHR_COL01 tests DORGTSQR and DORHR_COL using DLATSQR, DGEMQRT.
 */

#include <stdlib.h>
#include <math.h>
#include "verify.h"
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
extern void dlatsqr(const int m, const int n, const int mb, const int nb,
                    double* const restrict A, const int lda,
                    double* const restrict T, const int ldt,
                    double* const restrict work, const int lwork, int* info);
extern void dorgtsqr(const int m, const int n, const int mb, const int nb,
                     double* const restrict A, const int lda,
                     const double* const restrict T, const int ldt,
                     double* restrict work, const int lwork, int* info);
extern void dorhr_col(const int m, const int n, const int nb,
                      double* const restrict A, const int lda,
                      double* restrict T, const int ldt,
                      double* restrict D, int* info);
extern void dgemqrt(const char* side, const char* trans,
                    const int m, const int n, const int k, const int nb,
                    const double* const restrict V, const int ldv,
                    const double* const restrict T, const int ldt,
                    double* const restrict C, const int ldc,
                    double* const restrict work, int* info);
/**
 * DORHR_COL01 tests DORGTSQR and DORHR_COL using DLATSQR, DGEMQRT.
 *
 * @param[in]  m       Number of rows in test matrix.
 * @param[in]  n       Number of columns in test matrix.
 * @param[in]  mb1     Number of rows in row block.
 * @param[in]  nb1     Number of columns in column block (input).
 * @param[in]  nb2     Number of columns in column block (output).
 * @param[out] result  Results of each of the six tests.
 */
void dorhr_col01(const int m, const int n, const int mb1, const int nb1,
                 const int nb2, double* restrict result)
{
    double eps = dlamch("E");
    int k = m < n ? m : n;
    int l = m > n ? m : n;
    if (l < 1) l = 1;
    int info;
    int j, i;
    double anorm, resid, cnorm, dnorm;
    uint64_t seed = 1988198919901991ULL;

    double* A = malloc(m * n * sizeof(double));
    double* AF = malloc(m * n * sizeof(double));
    double* Q = malloc(l * l * sizeof(double));
    double* R = malloc(m * l * sizeof(double));
    double* rwork = malloc(l * sizeof(double));
    double* C = malloc(m * n * sizeof(double));
    double* CF = malloc(m * n * sizeof(double));
    double* D = malloc(n * m * sizeof(double));
    double* DF = malloc(n * m * sizeof(double));

    for (j = 0; j < n; j++) {
        dlarnv_rng(2, &seed, m, &A[j * m]);
    }
    dlacpy("F", m, n, A, m, AF, m);

    int nrb;
    if (mb1 - n > 0) {
        nrb = (m - n + mb1 - n - 1) / (mb1 - n);
        if (nrb < 1) nrb = 1;
    } else {
        nrb = 1;
    }

    double* T1 = malloc(nb1 * n * nrb * sizeof(double));
    double* T2 = malloc(nb2 * n * sizeof(double));
    double* DIAG = malloc(n * sizeof(double));

    int nb1_ub = nb1 < n ? nb1 : n;
    int nb2_ub = nb2 < n ? nb2 : n;

    double workquery;
    int lwork;

    dlatsqr(m, n, mb1, nb1_ub, AF, m, T1, nb1, &workquery, -1, &info);
    lwork = (int)workquery;
    dorgtsqr(m, n, mb1, nb1, AF, m, T1, nb1, &workquery, -1, &info);
    int lwork2 = (int)workquery;
    if (lwork2 > lwork) lwork = lwork2;
    if (nb2_ub * n > lwork) lwork = nb2_ub * n;
    if (nb2_ub * m > lwork) lwork = nb2_ub * m;

    double* work = malloc(lwork * sizeof(double));

    dlatsqr(m, n, mb1, nb1_ub, AF, m, T1, nb1, work, lwork, &info);

    dlacpy("U", n, n, AF, m, R, m);

    dorgtsqr(m, n, mb1, nb1, AF, m, T1, nb1, work, lwork, &info);

    dorhr_col(m, n, nb2, AF, m, T2, nb2, DIAG, &info);

    dlacpy("U", n, n, R, m, AF, m);

    for (i = 0; i < n; i++) {
        if (DIAG[i] == -1.0) {
            cblas_dscal(n - i, -1.0, &AF[i + i * m], m);
        }
    }

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
        dlarnv_rng(2, &seed, m, &C[j * m]);
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
        dlarnv_rng(2, &seed, n, &D[j * n]);
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
