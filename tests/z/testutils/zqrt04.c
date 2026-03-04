/**
 * @file zqrt04.c
 * @brief ZQRT04 tests ZGEQRT and ZGEMQRT.
 */

#include <stdlib.h>
#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"
#include "test_rng.h"
/**
 * ZQRT04 tests ZGEQRT and ZGEMQRT.
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
void zqrt04(const INT m, const INT n, const INT nb, f64* restrict result)
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

    const c128 CONE = CMPLX(1.0, 0.0);
    const c128 CZERO = CMPLX(0.0, 0.0);
    const c128 CNEGONE = CMPLX(-1.0, 0.0);

    c128* A = malloc(m * n * sizeof(c128));
    c128* AF = malloc(m * n * sizeof(c128));
    c128* Q = malloc(m * m * sizeof(c128));
    c128* R = malloc(m * ll * sizeof(c128));
    f64* rwork = malloc(ll * sizeof(f64));
    c128* work = malloc(lwork * sizeof(c128));
    c128* T = malloc(nb * n * sizeof(c128));
    c128* C = malloc(m * n * sizeof(c128));
    c128* CF = malloc(m * n * sizeof(c128));
    c128* D = malloc(n * m * sizeof(c128));
    c128* DF = malloc(n * m * sizeof(c128));

    for (j = 0; j < n; j++) {
        zlarnv_rng(2, m, &A[j * m], rng_state);
    }
    zlacpy("F", m, n, A, m, AF, m);

    zgeqrt(m, n, nb, AF, m, T, ldt, work, &info);

    zlaset("F", m, m, CZERO, CONE, Q, m);
    zgemqrt("R", "N", m, m, k, nb, AF, m, T, ldt, Q, m, work, &info);

    zlaset("F", m, n, CZERO, CZERO, R, m);
    zlacpy("U", m, n, AF, m, R, m);

    cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans,
                m, n, m, &CNEGONE, Q, m, A, m, &CONE, R, m);
    anorm = zlange("1", m, n, A, m, rwork);
    resid = zlange("1", m, n, R, m, rwork);
    if (anorm > 0.0) {
        result[0] = resid / (eps * (m > 1 ? m : 1) * anorm);
    } else {
        result[0] = 0.0;
    }

    zlaset("F", m, m, CZERO, CONE, R, m);
    cblas_zherk(CblasColMajor, CblasUpper, CblasConjTrans,
                m, m, -1.0, Q, m, 1.0, R, m);
    resid = zlansy("1", "U", m, R, m, rwork);
    result[1] = resid / (eps * (m > 1 ? m : 1));

    for (j = 0; j < n; j++) {
        zlarnv_rng(2, m, &C[j * m], rng_state);
    }
    cnorm = zlange("1", m, n, C, m, rwork);
    zlacpy("F", m, n, C, m, CF, m);

    zgemqrt("L", "N", m, n, k, nb, AF, m, T, nb, CF, m, work, &info);

    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                m, n, m, &CNEGONE, Q, m, C, m, &CONE, CF, m);
    resid = zlange("1", m, n, CF, m, rwork);
    if (cnorm > 0.0) {
        result[2] = resid / (eps * (m > 1 ? m : 1) * cnorm);
    } else {
        result[2] = 0.0;
    }

    zlacpy("F", m, n, C, m, CF, m);

    zgemqrt("L", "C", m, n, k, nb, AF, m, T, nb, CF, m, work, &info);

    cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans,
                m, n, m, &CNEGONE, Q, m, C, m, &CONE, CF, m);
    resid = zlange("1", m, n, CF, m, rwork);
    if (cnorm > 0.0) {
        result[3] = resid / (eps * (m > 1 ? m : 1) * cnorm);
    } else {
        result[3] = 0.0;
    }

    for (j = 0; j < m; j++) {
        zlarnv_rng(2, n, &D[j * n], rng_state);
    }
    dnorm = zlange("1", n, m, D, n, rwork);
    zlacpy("F", n, m, D, n, DF, n);

    zgemqrt("R", "N", n, m, k, nb, AF, m, T, nb, DF, n, work, &info);

    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                n, m, m, &CNEGONE, D, n, Q, m, &CONE, DF, n);
    resid = zlange("1", n, m, DF, n, rwork);
    if (dnorm > 0.0) {
        result[4] = resid / (eps * (m > 1 ? m : 1) * dnorm);
    } else {
        result[4] = 0.0;
    }

    zlacpy("F", n, m, D, n, DF, n);

    zgemqrt("R", "C", n, m, k, nb, AF, m, T, nb, DF, n, work, &info);

    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasConjTrans,
                n, m, m, &CNEGONE, D, n, Q, m, &CONE, DF, n);
    resid = zlange("1", n, m, DF, n, rwork);
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
