/**
 * @file dqrt04.c
 * @brief DQRT04 tests DGEQRT and DGEMQRT.
 */

#include <stdlib.h>
#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"
#include "test_rng.h"
/**
 * DQRT04 tests DGEQRT and DGEMQRT.
 *
 * @param[in]  m       Number of rows of the test matrix.
 * @param[in]  n       Number of columns of the test matrix.
 * @param[in]  nb      Block size of the test matrix. NB <= min(M,N).
 * @param[out] result  Results of each of the six tests:
 *                     result[0] = | A - Q R |
 *                     result[1] = | I - Q^H Q |
 *                     result[2] = | Q C - Q C |
 *                     result[3] = | Q^H C - Q^H C |
 *                     result[4] = | C Q - C Q |
 *                     result[5] = | C Q^H - C Q^H |
 */
void dqrt04(const INT m, const INT n, const INT nb, f64* restrict result)
{
    f64 eps = dlamch("E");
    INT k = m < n ? m : n;
    INT ll = m > n ? m : n;
    INT lwork = (ll > 2 ? ll : 2) * (ll > 2 ? ll : 2) * nb;
    INT ldt = nb;
    INT info;
    INT j;
    f64 anorm, resid, cnorm, dnorm;
    uint64_t rng_state[4];
    rng_seed(rng_state, 1988198919901991ULL);

    f64* A = malloc(m * n * sizeof(f64));
    f64* AF = malloc(m * n * sizeof(f64));
    f64* Q = malloc(m * m * sizeof(f64));
    f64* R = malloc(m * ll * sizeof(f64));
    f64* rwork = malloc(ll * sizeof(f64));
    f64* work = malloc(lwork * sizeof(f64));
    f64* T = malloc(nb * n * sizeof(f64));
    f64* C = malloc(m * n * sizeof(f64));
    f64* CF = malloc(m * n * sizeof(f64));
    f64* D = malloc(n * m * sizeof(f64));
    f64* DF = malloc(n * m * sizeof(f64));

    for (j = 0; j < n; j++) {
        dlarnv_rng(2, m, &A[j * m], rng_state);
    }
    dlacpy("F", m, n, A, m, AF, m);

    dgeqrt(m, n, nb, AF, m, T, ldt, work, &info);

    dlaset("F", m, m, 0.0, 1.0, Q, m);
    dgemqrt("R", "N", m, m, k, nb, AF, m, T, ldt, Q, m, work, &info);

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

    dgemqrt("L", "N", m, n, k, nb, AF, m, T, nb, CF, m, work, &info);

    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                m, n, m, -1.0, Q, m, C, m, 1.0, CF, m);
    resid = dlange("1", m, n, CF, m, rwork);
    if (cnorm > 0.0) {
        result[2] = resid / (eps * (m > 1 ? m : 1) * cnorm);
    } else {
        result[2] = 0.0;
    }

    dlacpy("F", m, n, C, m, CF, m);

    dgemqrt("L", "T", m, n, k, nb, AF, m, T, nb, CF, m, work, &info);

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

    dgemqrt("R", "N", n, m, k, nb, AF, m, T, nb, DF, n, work, &info);

    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                n, m, m, -1.0, D, n, Q, m, 1.0, DF, n);
    resid = dlange("1", n, m, DF, n, rwork);
    if (dnorm > 0.0) {
        result[4] = resid / (eps * (m > 1 ? m : 1) * dnorm);
    } else {
        result[4] = 0.0;
    }

    dlacpy("F", n, m, D, n, DF, n);

    dgemqrt("R", "T", n, m, k, nb, AF, m, T, nb, DF, n, work, &info);

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
    free(T);
    free(C);
    free(CF);
    free(D);
    free(DF);
}
