/**
 * @file dlqt05.c
 * @brief DLQT05 tests DTPLQT and DTPMLQT.
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
extern void dtplqt(const int m, const int n, const int l, const int mb,
                   f64* const restrict A, const int lda,
                   f64* const restrict B, const int ldb,
                   f64* const restrict T, const int ldt,
                   f64* const restrict work, int* info);
extern void dgemlqt(const char* side, const char* trans,
                    const int m, const int n, const int k, const int mb,
                    const f64* const restrict V, const int ldv,
                    const f64* const restrict T, const int ldt,
                    f64* const restrict C, const int ldc,
                    f64* const restrict work, int* info);
extern void dtpmlqt(const char* side, const char* trans,
                    const int m, const int n, const int k, const int l, const int mb,
                    const f64* const restrict V, const int ldv,
                    const f64* const restrict T, const int ldt,
                    f64* const restrict A, const int lda,
                    f64* const restrict B, const int ldb,
                    f64* const restrict work, int* info);
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
            f64* restrict result)
{
    f64 eps = dlamch("E");
    int k = m;
    int n2 = m + n;
    int np1 = (n > 0) ? m + 1 : 1;
    int lwork = n2 * n2 * nb;
    int ldt = nb;
    int info;
    int j;
    f64 anorm, resid, cnorm, dnorm;
    uint64_t rng_state[4];
    rng_seed(rng_state, 1988198919901991ULL);

    f64* A = malloc(m * n2 * sizeof(f64));
    f64* AF = malloc(m * n2 * sizeof(f64));
    f64* Q = malloc(n2 * n2 * sizeof(f64));
    f64* R = malloc(n2 * n2 * sizeof(f64));
    f64* rwork = malloc(n2 * sizeof(f64));
    f64* work = malloc(lwork * sizeof(f64));
    f64* T = malloc(nb * m * sizeof(f64));
    f64* C = malloc(n2 * m * sizeof(f64));
    f64* CF = malloc(n2 * m * sizeof(f64));
    f64* D = malloc(m * n2 * sizeof(f64));
    f64* DF = malloc(m * n2 * sizeof(f64));

    dlaset("F", m, n2, 0.0, 0.0, A, m);
    dlaset("F", nb, m, 0.0, 0.0, T, nb);
    for (j = 1; j <= m; j++) {
        dlarnv_rng(2, m - j + 1, &A[(j - 1) + (j - 1) * m], rng_state);
    }
    if (n > 0) {
        for (j = 1; j <= n - l; j++) {
            int col = m + j - 1;
            dlarnv_rng(2, m, &A[col * m], rng_state);
        }
    }
    if (l > 0) {
        for (j = 1; j <= l; j++) {
            int col = n + m - l + j - 1;
            dlarnv_rng(2, m - j + 1, &A[(j - 1) + col * m], rng_state);
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
        dlarnv_rng(2, n2, &C[j * n2], rng_state);
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
        dlarnv_rng(2, m, &D[j * m], rng_state);
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
