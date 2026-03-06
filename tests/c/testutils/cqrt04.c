/**
 * @file cqrt04.c
 * @brief CQRT04 tests CGEQRT and CGEMQRT.
 */

#include <stdlib.h>
#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"
#include "test_rng.h"
/**
 * CQRT04 tests CGEQRT and CGEMQRT.
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
void cqrt04(const INT m, const INT n, const INT nb, f32* restrict result)
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

    const c64 CONE = CMPLXF(1.0f, 0.0f);
    const c64 CZERO = CMPLXF(0.0f, 0.0f);
    const c64 CNEGONE = CMPLXF(-1.0f, 0.0f);

    c64* A = malloc(m * n * sizeof(c64));
    c64* AF = malloc(m * n * sizeof(c64));
    c64* Q = malloc(m * m * sizeof(c64));
    c64* R = malloc(m * ll * sizeof(c64));
    f32* rwork = malloc(ll * sizeof(f32));
    c64* work = malloc(lwork * sizeof(c64));
    c64* T = malloc(nb * n * sizeof(c64));
    c64* C = malloc(m * n * sizeof(c64));
    c64* CF = malloc(m * n * sizeof(c64));
    c64* D = malloc(n * m * sizeof(c64));
    c64* DF = malloc(n * m * sizeof(c64));

    for (j = 0; j < n; j++) {
        clarnv_rng(2, m, &A[j * m], rng_state);
    }
    clacpy("F", m, n, A, m, AF, m);

    cgeqrt(m, n, nb, AF, m, T, ldt, work, &info);

    claset("F", m, m, CZERO, CONE, Q, m);
    cgemqrt("R", "N", m, m, k, nb, AF, m, T, ldt, Q, m, work, &info);

    claset("F", m, n, CZERO, CZERO, R, m);
    clacpy("U", m, n, AF, m, R, m);

    cblas_cgemm(CblasColMajor, CblasConjTrans, CblasNoTrans,
                m, n, m, &CNEGONE, Q, m, A, m, &CONE, R, m);
    anorm = clange("1", m, n, A, m, rwork);
    resid = clange("1", m, n, R, m, rwork);
    if (anorm > 0.0f) {
        result[0] = resid / (eps * (m > 1 ? m : 1) * anorm);
    } else {
        result[0] = 0.0f;
    }

    claset("F", m, m, CZERO, CONE, R, m);
    cblas_cherk(CblasColMajor, CblasUpper, CblasConjTrans,
                m, m, -1.0f, Q, m, 1.0f, R, m);
    resid = clansy("1", "U", m, R, m, rwork);
    result[1] = resid / (eps * (m > 1 ? m : 1));

    for (j = 0; j < n; j++) {
        clarnv_rng(2, m, &C[j * m], rng_state);
    }
    cnorm = clange("1", m, n, C, m, rwork);
    clacpy("F", m, n, C, m, CF, m);

    cgemqrt("L", "N", m, n, k, nb, AF, m, T, nb, CF, m, work, &info);

    cblas_cgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                m, n, m, &CNEGONE, Q, m, C, m, &CONE, CF, m);
    resid = clange("1", m, n, CF, m, rwork);
    if (cnorm > 0.0f) {
        result[2] = resid / (eps * (m > 1 ? m : 1) * cnorm);
    } else {
        result[2] = 0.0f;
    }

    clacpy("F", m, n, C, m, CF, m);

    cgemqrt("L", "C", m, n, k, nb, AF, m, T, nb, CF, m, work, &info);

    cblas_cgemm(CblasColMajor, CblasConjTrans, CblasNoTrans,
                m, n, m, &CNEGONE, Q, m, C, m, &CONE, CF, m);
    resid = clange("1", m, n, CF, m, rwork);
    if (cnorm > 0.0f) {
        result[3] = resid / (eps * (m > 1 ? m : 1) * cnorm);
    } else {
        result[3] = 0.0f;
    }

    for (j = 0; j < m; j++) {
        clarnv_rng(2, n, &D[j * n], rng_state);
    }
    dnorm = clange("1", n, m, D, n, rwork);
    clacpy("F", n, m, D, n, DF, n);

    cgemqrt("R", "N", n, m, k, nb, AF, m, T, nb, DF, n, work, &info);

    cblas_cgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                n, m, m, &CNEGONE, D, n, Q, m, &CONE, DF, n);
    resid = clange("1", n, m, DF, n, rwork);
    if (dnorm > 0.0f) {
        result[4] = resid / (eps * (m > 1 ? m : 1) * dnorm);
    } else {
        result[4] = 0.0f;
    }

    clacpy("F", n, m, D, n, DF, n);

    cgemqrt("R", "C", n, m, k, nb, AF, m, T, nb, DF, n, work, &info);

    cblas_cgemm(CblasColMajor, CblasNoTrans, CblasConjTrans,
                n, m, m, &CNEGONE, D, n, Q, m, &CONE, DF, n);
    resid = clange("1", n, m, DF, n, rwork);
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
