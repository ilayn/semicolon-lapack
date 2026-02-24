/**
 * @file sqrt04.c
 * @brief SQRT04 tests SGEQRT and SGEMQRT.
 */

#include <stdlib.h>
#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"
#include "test_rng.h"
/**
 * SQRT04 tests SGEQRT and SGEMQRT.
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
void sqrt04(const INT m, const INT n, const INT nb, f32* restrict result)
{
    f32 eps = slamch("E");
    INT k = m < n ? m : n;
    INT ll = m > n ? m : n;
    INT lwork = (ll > 2 ? ll : 2) * (ll > 2 ? ll : 2) * nb;
    INT ldt = nb;
    INT info;
    INT j;
    f32 anorm, resid, cnorm, dnorm;
    uint64_t rng_state[4];
    rng_seed(rng_state, 1988198919901991ULL);

    f32* A = malloc(m * n * sizeof(f32));
    f32* AF = malloc(m * n * sizeof(f32));
    f32* Q = malloc(m * m * sizeof(f32));
    f32* R = malloc(m * ll * sizeof(f32));
    f32* rwork = malloc(ll * sizeof(f32));
    f32* work = malloc(lwork * sizeof(f32));
    f32* T = malloc(nb * n * sizeof(f32));
    f32* C = malloc(m * n * sizeof(f32));
    f32* CF = malloc(m * n * sizeof(f32));
    f32* D = malloc(n * m * sizeof(f32));
    f32* DF = malloc(n * m * sizeof(f32));

    for (j = 0; j < n; j++) {
        slarnv_rng(2, m, &A[j * m], rng_state);
    }
    slacpy("F", m, n, A, m, AF, m);

    sgeqrt(m, n, nb, AF, m, T, ldt, work, &info);

    slaset("F", m, m, 0.0f, 1.0f, Q, m);
    sgemqrt("R", "N", m, m, k, nb, AF, m, T, ldt, Q, m, work, &info);

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

    sgemqrt("L", "N", m, n, k, nb, AF, m, T, nb, CF, m, work, &info);

    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                m, n, m, -1.0f, Q, m, C, m, 1.0f, CF, m);
    resid = slange("1", m, n, CF, m, rwork);
    if (cnorm > 0.0f) {
        result[2] = resid / (eps * (m > 1 ? m : 1) * cnorm);
    } else {
        result[2] = 0.0f;
    }

    slacpy("F", m, n, C, m, CF, m);

    sgemqrt("L", "T", m, n, k, nb, AF, m, T, nb, CF, m, work, &info);

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

    sgemqrt("R", "N", n, m, k, nb, AF, m, T, nb, DF, n, work, &info);

    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                n, m, m, -1.0f, D, n, Q, m, 1.0f, DF, n);
    resid = slange("1", n, m, DF, n, rwork);
    if (dnorm > 0.0f) {
        result[4] = resid / (eps * (m > 1 ? m : 1) * dnorm);
    } else {
        result[4] = 0.0f;
    }

    slacpy("F", n, m, D, n, DF, n);

    sgemqrt("R", "T", n, m, k, nb, AF, m, T, nb, DF, n, work, &info);

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
    free(T);
    free(C);
    free(CF);
    free(D);
    free(DF);
}
