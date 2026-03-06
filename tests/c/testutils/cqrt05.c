/**
 * @file cqrt05.c
 * @brief CQRT05 tests CTPQRT and CTPMQRT.
 */

#include <stdlib.h>
#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"
#include "test_rng.h"
/**
 * CQRT05 tests CTPQRT and CTPMQRT.
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
void cqrt05(const INT m, const INT n, const INT l, const INT nb, f32* restrict result)
{
    f32 eps = slamch("E");
    INT k = n;
    INT m2 = m + n;
    INT np1 = (m > 0) ? n : 0;
    INT lwork = m2 * m2 * nb;
    INT ldt = nb;
    INT info;
    INT j;
    f32 anorm, resid, cnorm, dnorm;
    uint64_t rng_state[4];
    rng_seed(rng_state, 1988198919901991ULL);

    const c64 CONE = CMPLXF(1.0f, 0.0f);
    const c64 CZERO = CMPLXF(0.0f, 0.0f);
    const c64 CNEGONE = CMPLXF(-1.0f, 0.0f);

    c64* A = malloc(m2 * n * sizeof(c64));
    c64* AF = malloc(m2 * n * sizeof(c64));
    c64* Q = malloc(m2 * m2 * sizeof(c64));
    c64* R = malloc(m2 * m2 * sizeof(c64));
    f32* rwork = malloc(m2 * sizeof(f32));
    c64* work = malloc(lwork * sizeof(c64));
    c64* T = malloc(nb * n * sizeof(c64));
    c64* C = malloc(m2 * n * sizeof(c64));
    c64* CF = malloc(m2 * n * sizeof(c64));
    c64* D = malloc(n * m2 * sizeof(c64));
    c64* DF = malloc(n * m2 * sizeof(c64));

    claset("F", m2, n, CZERO, CZERO, A, m2);
    claset("F", nb, n, CZERO, CZERO, T, nb);

    for (j = 0; j < n; j++) {
        clarnv_rng(2, j + 1, &A[j * m2], rng_state);
    }
    if (m > 0) {
        INT ml = m - l;
        for (j = 0; j < n; j++) {
            if (ml > 0) {
                clarnv_rng(2, ml, &A[n + j * m2], rng_state);
            }
        }
    }
    if (l > 0) {
        INT start_row = n + m - l;
        for (j = 0; j < n; j++) {
            INT len = (j + 1 < l) ? (j + 1) : l;
            clarnv_rng(2, len, &A[start_row + j * m2], rng_state);
        }
    }

    clacpy("F", m2, n, A, m2, AF, m2);

    ctpqrt(m, n, l, nb, AF, m2, &AF[np1], m2, T, ldt, work, &info);

    claset("F", m2, m2, CZERO, CONE, Q, m2);
    cgemqrt("R", "N", m2, m2, k, nb, AF, m2, T, ldt, Q, m2, work, &info);

    claset("F", m2, n, CZERO, CZERO, R, m2);
    clacpy("U", m2, n, AF, m2, R, m2);

    cblas_cgemm(CblasColMajor, CblasConjTrans, CblasNoTrans,
                m2, n, m2, &CNEGONE, Q, m2, A, m2, &CONE, R, m2);
    anorm = clange("1", m2, n, A, m2, rwork);
    resid = clange("1", m2, n, R, m2, rwork);
    if (anorm > 0.0f) {
        result[0] = resid / (eps * anorm * (m2 > 1 ? m2 : 1));
    } else {
        result[0] = 0.0f;
    }

    claset("F", m2, m2, CZERO, CONE, R, m2);
    cblas_cherk(CblasColMajor, CblasUpper, CblasConjTrans,
                m2, m2, -1.0f, Q, m2, 1.0f, R, m2);
    resid = clansy("1", "U", m2, R, m2, rwork);
    result[1] = resid / (eps * (m2 > 1 ? m2 : 1));

    for (j = 0; j < n; j++) {
        clarnv_rng(2, m2, &C[j * m2], rng_state);
    }
    cnorm = clange("1", m2, n, C, m2, rwork);
    clacpy("F", m2, n, C, m2, CF, m2);

    ctpmqrt("L", "N", m, n, k, l, nb, &AF[np1], m2, T, ldt,
            CF, m2, &CF[np1], m2, work, &info);

    cblas_cgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                m2, n, m2, &CNEGONE, Q, m2, C, m2, &CONE, CF, m2);
    resid = clange("1", m2, n, CF, m2, rwork);
    if (cnorm > 0.0f) {
        result[2] = resid / (eps * (m2 > 1 ? m2 : 1) * cnorm);
    } else {
        result[2] = 0.0f;
    }

    clacpy("F", m2, n, C, m2, CF, m2);

    ctpmqrt("L", "C", m, n, k, l, nb, &AF[np1], m2, T, ldt,
            CF, m2, &CF[np1], m2, work, &info);

    cblas_cgemm(CblasColMajor, CblasConjTrans, CblasNoTrans,
                m2, n, m2, &CNEGONE, Q, m2, C, m2, &CONE, CF, m2);
    resid = clange("1", m2, n, CF, m2, rwork);
    if (cnorm > 0.0f) {
        result[3] = resid / (eps * (m2 > 1 ? m2 : 1) * cnorm);
    } else {
        result[3] = 0.0f;
    }

    for (j = 0; j < m2; j++) {
        clarnv_rng(2, n, &D[j * n], rng_state);
    }
    dnorm = clange("1", n, m2, D, n, rwork);
    clacpy("F", n, m2, D, n, DF, n);

    ctpmqrt("R", "N", n, m, n, l, nb, &AF[np1], m2, T, ldt,
            DF, n, &DF[np1 * n], n, work, &info);

    cblas_cgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                n, m2, m2, &CNEGONE, D, n, Q, m2, &CONE, DF, n);
    resid = clange("1", n, m2, DF, n, rwork);
    if (dnorm > 0.0f) {
        result[4] = resid / (eps * (m2 > 1 ? m2 : 1) * dnorm);
    } else {
        result[4] = 0.0f;
    }

    clacpy("F", n, m2, D, n, DF, n);

    ctpmqrt("R", "C", n, m, n, l, nb, &AF[np1], m2, T, ldt,
            DF, n, &DF[np1 * n], n, work, &info);

    cblas_cgemm(CblasColMajor, CblasNoTrans, CblasConjTrans,
                n, m2, m2, &CNEGONE, D, n, Q, m2, &CONE, DF, n);
    resid = clange("1", n, m2, DF, n, rwork);
    if (dnorm > 0.0f) {
        result[5] = resid / (eps * (m2 > 1 ? m2 : 1) * dnorm);
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
