/**
 * @file cunhr_col02.c
 * @brief CUNHR_COL02 tests CUNGTSQR_ROW and CUNHR_COL inside CGETSQRHRT.
 */

#include <stdlib.h>
#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"
#include "test_rng.h"
/**
 * CUNHR_COL02 tests CUNGTSQR_ROW and CUNHR_COL inside CGETSQRHRT
 * (which calls CLATSQR, CUNGTSQR_ROW and CUNHR_COL) using CGEMQRT.
 * Therefore, CLATSQR (part of CGEQR), CGEMQRT (part of CGEMQR)
 * have to be tested before this test.
 *
 * @param[in]  m       Number of rows in test matrix.
 * @param[in]  n       Number of columns in test matrix.
 * @param[in]  mb1     Number of rows in row block.
 * @param[in]  nb1     Number of columns in column block (input).
 * @param[in]  nb2     Number of columns in column block (output).
 * @param[out] result  Results of each of the six tests.
 */
void cunhr_col02(const INT m, const INT n, const INT mb1, const INT nb1,
                 const INT nb2, f32* restrict result)
{
    const c64 CZERO = CMPLXF(0.0f, 0.0f);
    const c64 CONE = CMPLXF(1.0f, 0.0f);
    const c64 CNEGONE = CMPLXF(-1.0f, 0.0f);

    f32 eps = slamch("E");
    INT k = m < n ? m : n;
    INT l = m > n ? m : n;
    if (l < 1) l = 1;
    INT info;
    INT j;
    f32 anorm, resid, cnorm, dnorm;
    uint64_t rng_state[4];
    rng_seed(rng_state, 1988198919901991ULL);

    c64* A = malloc(m * n * sizeof(c64));
    c64* AF = malloc(m * n * sizeof(c64));
    c64* Q = malloc(l * l * sizeof(c64));
    c64* R = malloc(m * l * sizeof(c64));
    f32* rwork = malloc(l * sizeof(f32));
    c64* C = malloc(m * n * sizeof(c64));
    c64* CF = malloc(m * n * sizeof(c64));
    c64* D = malloc(n * m * sizeof(c64));
    c64* DF = malloc(n * m * sizeof(c64));

    for (j = 0; j < n; j++) {
        clarnv_rng(2, m, &A[j * m], rng_state);
    }
    clacpy("F", m, n, A, m, AF, m);

    INT nrb;
    if (mb1 - n > 0) {
        nrb = (m - n + mb1 - n - 1) / (mb1 - n);
        if (nrb < 1) nrb = 1;
    } else {
        nrb = 1;
    }

    c64* T1 = malloc(nb1 * n * nrb * sizeof(c64));
    c64* T2 = malloc(nb2 * n * sizeof(c64));
    c64* DIAG = malloc(n * sizeof(c64));

    INT nb2_ub = nb2 < n ? nb2 : n;

    c64 workquery;
    INT lwork;

    cgetsqrhrt(m, n, mb1, nb1, nb2, AF, m, T2, nb2, &workquery, -1, &info);
    lwork = (INT)crealf(workquery);
    if (nb2_ub * n > lwork) lwork = nb2_ub * n;
    if (nb2_ub * m > lwork) lwork = nb2_ub * m;

    c64* work = malloc(lwork * sizeof(c64));

    clacpy("F", m, n, A, m, AF, m);

    cgetsqrhrt(m, n, mb1, nb1, nb2, AF, m, T2, nb2, work, lwork, &info);

    claset("F", m, m, CZERO, CONE, Q, m);

    cgemqrt("L", "N", m, m, k, nb2_ub, AF, m, T2, nb2, Q, m, work, &info);

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

    cgemqrt("L", "N", m, n, k, nb2_ub, AF, m, T2, nb2, CF, m, work, &info);

    cblas_cgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                m, n, m, &CNEGONE, Q, m, C, m, &CONE, CF, m);
    resid = clange("1", m, n, CF, m, rwork);
    if (cnorm > 0.0f) {
        result[2] = resid / (eps * (m > 1 ? m : 1) * cnorm);
    } else {
        result[2] = 0.0f;
    }

    clacpy("F", m, n, C, m, CF, m);

    cgemqrt("L", "C", m, n, k, nb2_ub, AF, m, T2, nb2, CF, m, work, &info);

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

    cgemqrt("R", "N", n, m, k, nb2_ub, AF, m, T2, nb2, DF, n, work, &info);

    cblas_cgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                n, m, m, &CNEGONE, D, n, Q, m, &CONE, DF, n);
    resid = clange("1", n, m, DF, n, rwork);
    if (dnorm > 0.0f) {
        result[4] = resid / (eps * (m > 1 ? m : 1) * dnorm);
    } else {
        result[4] = 0.0f;
    }

    clacpy("F", n, m, D, n, DF, n);

    cgemqrt("R", "C", n, m, k, nb2_ub, AF, m, T2, nb2, DF, n, work, &info);

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
    free(T1);
    free(T2);
    free(DIAG);
    free(C);
    free(CF);
    free(D);
    free(DF);
}
