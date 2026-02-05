/**
 * @file dlqt05.c
 * @brief DLQT05 tests DTPLQT and DTPMLQT.
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
extern void dtplqt(const int m, const int n, const int l, const int mb,
                   double* const restrict A, const int lda,
                   double* const restrict B, const int ldb,
                   double* const restrict T, const int ldt,
                   double* const restrict work, int* info);
extern void dgemlqt(const char* side, const char* trans,
                    const int m, const int n, const int k, const int mb,
                    const double* const restrict V, const int ldv,
                    const double* const restrict T, const int ldt,
                    double* const restrict C, const int ldc,
                    double* const restrict work, int* info);
extern void dtpmlqt(const char* side, const char* trans,
                    const int m, const int n, const int k, const int l, const int mb,
                    const double* const restrict V, const int ldv,
                    const double* const restrict T, const int ldt,
                    double* const restrict A, const int lda,
                    double* const restrict B, const int ldb,
                    double* const restrict work, int* info);
/**
 * DLQT05 tests DTPLQT and DTPMLQT.
 *
 * @param[in]  m       Number of rows in lower part of the test matrix.
 * @param[in]  n       Number of columns in test matrix.
 * @param[in]  l       Number of rows of upper trapezoidal part. 0 <= L <= M.
 * @param[in]  nb      Block size of the test matrix. NB <= N.
 * @param[out] result  Results of each of the six tests:
 *                     result[0] = | A - Q*R |
 *                     result[1] = | I - Q^H*Q |
 *                     result[2] = | Q*C - Q*C |
 *                     result[3] = | Q^H*C - Q^H*C |
 *                     result[4] = | C*Q - C*Q |
 *                     result[5] = | C*Q^H - C*Q^H |
 */
void dlqt05(const int m, const int n, const int l, const int nb,
            double* restrict result)
{
    double eps = dlamch("E");
    int k = m;
    int n2 = m + n;
    int np1 = (n > 0) ? m + 1 : 1;
    int lwork = n2 * n2 * nb;
    int ldt = nb;
    int info;
    int j;
    double anorm, resid, cnorm, dnorm;
    uint64_t seed = 1988198919901991ULL;

    double* A = malloc(m * n2 * sizeof(double));
    double* AF = malloc(m * n2 * sizeof(double));
    double* Q = malloc(n2 * n2 * sizeof(double));
    double* R = malloc(n2 * n2 * sizeof(double));
    double* rwork = malloc(n2 * sizeof(double));
    double* work = malloc(lwork * sizeof(double));
    double* T = malloc(nb * m * sizeof(double));
    double* C = malloc(n2 * m * sizeof(double));
    double* CF = malloc(n2 * m * sizeof(double));
    double* D = malloc(m * n2 * sizeof(double));
    double* DF = malloc(m * n2 * sizeof(double));

    dlaset("F", m, n2, 0.0, 0.0, A, m);
    dlaset("F", nb, m, 0.0, 0.0, T, nb);
    for (j = 1; j <= m; j++) {
        dlarnv_rng(2, &seed, m - j + 1, &A[(j - 1) + (j - 1) * m]);
    }
    if (n > 0) {
        for (j = 1; j <= n - l; j++) {
            int col = m + j - 1;
            dlarnv_rng(2, &seed, m, &A[col * m]);
        }
    }
    if (l > 0) {
        for (j = 1; j <= l; j++) {
            int col = n + m - l + j - 1;
            dlarnv_rng(2, &seed, m - j + 1, &A[(j - 1) + col * m]);
        }
    }

    dlacpy("F", m, n2, A, m, AF, m);

    dtplqt(m, n, l, nb, AF, m, &AF[(np1 - 1) * m], m, T, ldt, work, &info);

    dlaset("F", n2, n2, 0.0, 1.0, Q, n2);
    dgemlqt("L", "N", n2, n2, k, nb, AF, m, T, ldt, Q, n2, work, &info);

    dlaset("F", n2, n2, 0.0, 0.0, R, n2);
    dlacpy("L", m, n2, AF, m, R, n2);

    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                m, n2, n2, -1.0, A, m, Q, n2, 1.0, R, n2);
    anorm = dlange("1", m, n2, A, m, rwork);
    resid = dlange("1", m, n2, R, n2, rwork);
    if (anorm > 0.0) {
        result[0] = resid / (eps * anorm * (n2 > 1 ? n2 : 1));
    } else {
        result[0] = 0.0;
    }

    dlaset("F", n2, n2, 0.0, 1.0, R, n2);
    cblas_dsyrk(CblasColMajor, CblasUpper, CblasNoTrans,
                n2, n2, -1.0, Q, n2, 1.0, R, n2);
    resid = dlansy("1", "U", n2, R, n2, rwork);
    result[1] = resid / (eps * (n2 > 1 ? n2 : 1));

    dlaset("F", n2, m, 0.0, 1.0, C, n2);
    for (j = 0; j < m; j++) {
        dlarnv_rng(2, &seed, n2, &C[j * n2]);
    }
    cnorm = dlange("1", n2, m, C, n2, rwork);
    dlacpy("F", n2, m, C, n2, CF, n2);

    dtpmlqt("L", "N", n, m, k, l, nb, &AF[(np1 - 1) * m], m, T, ldt,
            CF, n2, &CF[np1 - 1], n2, work, &info);

    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                n2, m, n2, -1.0, Q, n2, C, n2, 1.0, CF, n2);
    resid = dlange("1", n2, m, CF, n2, rwork);
    if (cnorm > 0.0) {
        result[2] = resid / (eps * (n2 > 1 ? n2 : 1) * cnorm);
    } else {
        result[2] = 0.0;
    }

    dlacpy("F", n2, m, C, n2, CF, n2);

    dtpmlqt("L", "T", n, m, k, l, nb, &AF[(np1 - 1) * m], m, T, ldt,
            CF, n2, &CF[np1 - 1], n2, work, &info);

    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                n2, m, n2, -1.0, Q, n2, C, n2, 1.0, CF, n2);
    resid = dlange("1", n2, m, CF, n2, rwork);
    if (cnorm > 0.0) {
        result[3] = resid / (eps * (n2 > 1 ? n2 : 1) * cnorm);
    } else {
        result[3] = 0.0;
    }

    for (j = 0; j < n2; j++) {
        dlarnv_rng(2, &seed, m, &D[j * m]);
    }
    dnorm = dlange("1", m, n2, D, m, rwork);
    dlacpy("F", m, n2, D, m, DF, m);

    dtpmlqt("R", "N", m, n, k, l, nb, &AF[(np1 - 1) * m], m, T, ldt,
            DF, m, &DF[(np1 - 1) * m], m, work, &info);

    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                m, n2, n2, -1.0, D, m, Q, n2, 1.0, DF, m);
    resid = dlange("1", m, n2, DF, m, rwork);
    if (cnorm > 0.0) {
        result[4] = resid / (eps * (n2 > 1 ? n2 : 1) * dnorm);
    } else {
        result[4] = 0.0;
    }

    dlacpy("F", m, n2, D, m, DF, m);

    dtpmlqt("R", "T", m, n, k, l, nb, &AF[(np1 - 1) * m], m, T, ldt,
            DF, m, &DF[(np1 - 1) * m], m, work, &info);

    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                m, n2, n2, -1.0, D, m, Q, n2, 1.0, DF, m);
    resid = dlange("1", m, n2, DF, m, rwork);
    if (cnorm > 0.0) {
        result[5] = resid / (eps * (n2 > 1 ? n2 : 1) * dnorm);
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
