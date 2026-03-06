/**
 * @file ctsqr01.c
 * @brief CTSQR01 tests CGEQR, CGELQ, CGEMLQ and CGEMQR.
 *
 * Port of LAPACK TESTING/LIN/ctsqr01.f to C.
 */

#include <stdlib.h>
#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"

/**
 * CTSQR01 tests CGEQR, CGELQ, CGEMLQ and CGEMQR.
 *
 * @param[in]     tssw    'TS' for testing tall skinny QR, anything else for
 *                        testing short wide LQ.
 * @param[in]     m       Number of rows in test matrix.
 * @param[in]     n       Number of columns in test matrix.
 * @param[in]     mb      Number of rows in row block in test matrix.
 * @param[in]     nb      Number of columns in column block test matrix.
 * @param[out]    result  Array of 6 results:
 *                        RESULT[0] = | A - Q R | or | A - L Q |
 *                        RESULT[1] = | I - Q^H Q | or | I - Q Q^H |
 *                        RESULT[2] = | Q C - Q C |
 *                        RESULT[3] = | Q^H C - Q^H C |
 *                        RESULT[4] = | C Q - C Q |
 *                        RESULT[5] = | C Q^H - C Q^H |
 */
void ctsqr01(const char* tssw, const INT m, const INT n, const INT mb,
             const INT nb, f32* result)
{
    const f32 ZERO = 0.0f;
    const c64 ONE = CMPLXF(1.0f, 0.0f);
    const c64 CZERO = CMPLXF(0.0f, 0.0f);
    const c64 CNEGONE = CMPLXF(-1.0f, 0.0f);

    INT ts, testzeros;
    INT info, j, k, l, lwork, tsize, mnb;
    f32 anorm, eps, resid, cnorm, dnorm;

    INT iseed[4] = {1988, 1989, 1990, 1991};
    c64 tquery[5], workquery[1];

    c64* A = NULL;
    c64* AF = NULL;
    c64* Q = NULL;
    c64* R = NULL;
    f32* rwork = NULL;
    c64* work = NULL;
    c64* T = NULL;
    c64* C = NULL;
    c64* CF = NULL;
    c64* D = NULL;
    c64* DF = NULL;
    c64* LQ = NULL;

    /* TEST TALL SKINNY OR SHORT WIDE */
    ts = (tssw[0] == 'T' || tssw[0] == 't') && (tssw[1] == 'S' || tssw[1] == 's');

    /* TEST MATRICES WITH HALF OF MATRIX BEING ZEROS */
    testzeros = 0;

    eps = slamch("E");
    k = (m < n) ? m : n;
    l = m;
    if (n > l) l = n;
    if (1 > l) l = 1;
    mnb = (mb > nb) ? mb : nb;
    lwork = (3 > l ? 3 : l) * mnb;

    /* Dynamically allocate local arrays */
    A = (c64*)malloc(m * n * sizeof(c64));
    AF = (c64*)malloc(m * n * sizeof(c64));
    Q = (c64*)malloc(l * l * sizeof(c64));
    R = (c64*)malloc(m * l * sizeof(c64));
    rwork = (f32*)malloc(l * sizeof(f32));
    C = (c64*)malloc(m * n * sizeof(c64));
    CF = (c64*)malloc(m * n * sizeof(c64));
    D = (c64*)malloc(n * m * sizeof(c64));
    DF = (c64*)malloc(n * m * sizeof(c64));
    LQ = (c64*)malloc(l * n * sizeof(c64));

    /* Put random numbers into A and copy to AF */
    for (j = 0; j < n; j++) {
        clarnv(2, iseed, m, &A[j * m]);
    }
    if (testzeros) {
        if (m >= 4) {
            for (j = 0; j < n; j++) {
                clarnv(2, iseed, m / 2, &A[m / 4 + j * m]);
            }
        }
    }
    clacpy("F", m, n, A, m, AF, m);

    if (ts) {
        /* Factor the matrix A in the array AF. */
        cgeqr(m, n, AF, m, tquery, -1, workquery, -1, &info);
        tsize = (INT)crealf(tquery[0]);
        lwork = (INT)crealf(workquery[0]);
        cgemqr("L", "N", m, m, k, AF, m, tquery, tsize, CF, m, workquery, -1, &info);
        if ((INT)crealf(workquery[0]) > lwork) lwork = (INT)crealf(workquery[0]);
        cgemqr("L", "N", m, n, k, AF, m, tquery, tsize, CF, m, workquery, -1, &info);
        if ((INT)crealf(workquery[0]) > lwork) lwork = (INT)crealf(workquery[0]);
        cgemqr("L", "C", m, n, k, AF, m, tquery, tsize, CF, m, workquery, -1, &info);
        if ((INT)crealf(workquery[0]) > lwork) lwork = (INT)crealf(workquery[0]);
        cgemqr("R", "N", n, m, k, AF, m, tquery, tsize, DF, n, workquery, -1, &info);
        if ((INT)crealf(workquery[0]) > lwork) lwork = (INT)crealf(workquery[0]);
        cgemqr("R", "C", n, m, k, AF, m, tquery, tsize, DF, n, workquery, -1, &info);
        if ((INT)crealf(workquery[0]) > lwork) lwork = (INT)crealf(workquery[0]);

        T = (c64*)malloc(tsize * sizeof(c64));
        work = (c64*)malloc(lwork * sizeof(c64));

        cgeqr(m, n, AF, m, T, tsize, work, lwork, &info);

        /* Generate the m-by-m matrix Q */
        claset("F", m, m, CZERO, ONE, Q, m);
        cgemqr("L", "N", m, m, k, AF, m, T, tsize, Q, m, work, lwork, &info);

        /* Copy R */
        claset("F", m, n, CZERO, CZERO, R, m);
        clacpy("U", m, n, AF, m, R, m);

        /* Compute |R - Q'*A| / |A| and store in RESULT(1) */
        cblas_cgemm(CblasColMajor, CblasConjTrans, CblasNoTrans, m, n, m,
                    &CNEGONE, Q, m, A, m, &ONE, R, m);
        anorm = clange("1", m, n, A, m, rwork);
        resid = clange("1", m, n, R, m, rwork);
        if (anorm > ZERO) {
            result[0] = resid / (eps * (1 > m ? 1 : m) * anorm);
        } else {
            result[0] = ZERO;
        }

        /* Compute |I - Q'*Q| and store in RESULT(2) */
        claset("F", m, m, CZERO, ONE, R, m);
        cblas_cherk(CblasColMajor, CblasUpper, CblasConjTrans, m, m, -1.0f, Q, m, 1.0f, R, m);
        resid = clansy("1", "U", m, R, m, rwork);
        result[1] = resid / (eps * (1 > m ? 1 : m));

        /* Generate random m-by-n matrix C and a copy CF */
        for (j = 0; j < n; j++) {
            clarnv(2, iseed, m, &C[j * m]);
        }
        cnorm = clange("1", m, n, C, m, rwork);
        clacpy("F", m, n, C, m, CF, m);

        /* Apply Q to C as Q*C */
        cgemqr("L", "N", m, n, k, AF, m, T, tsize, CF, m, work, lwork, &info);

        /* Compute |Q*C - Q*C| / |C| */
        cblas_cgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, m,
                    &CNEGONE, Q, m, C, m, &ONE, CF, m);
        resid = clange("1", m, n, CF, m, rwork);
        if (cnorm > ZERO) {
            result[2] = resid / (eps * (1 > m ? 1 : m) * cnorm);
        } else {
            result[2] = ZERO;
        }

        /* Copy C into CF again */
        clacpy("F", m, n, C, m, CF, m);

        /* Apply Q to C as QT*C */
        cgemqr("L", "C", m, n, k, AF, m, T, tsize, CF, m, work, lwork, &info);

        /* Compute |QT*C - QT*C| / |C| */
        cblas_cgemm(CblasColMajor, CblasConjTrans, CblasNoTrans, m, n, m,
                    &CNEGONE, Q, m, C, m, &ONE, CF, m);
        resid = clange("1", m, n, CF, m, rwork);
        if (cnorm > ZERO) {
            result[3] = resid / (eps * (1 > m ? 1 : m) * cnorm);
        } else {
            result[3] = ZERO;
        }

        /* Generate random n-by-m matrix D and a copy DF */
        for (j = 0; j < m; j++) {
            clarnv(2, iseed, n, &D[j * n]);
        }
        dnorm = clange("1", n, m, D, n, rwork);
        clacpy("F", n, m, D, n, DF, n);

        /* Apply Q to D as D*Q */
        cgemqr("R", "N", n, m, k, AF, m, T, tsize, DF, n, work, lwork, &info);

        /* Compute |D*Q - D*Q| / |D| */
        cblas_cgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, m, m,
                    &CNEGONE, D, n, Q, m, &ONE, DF, n);
        resid = clange("1", n, m, DF, n, rwork);
        if (dnorm > ZERO) {
            result[4] = resid / (eps * (1 > m ? 1 : m) * dnorm);
        } else {
            result[4] = ZERO;
        }

        /* Copy D into DF again */
        clacpy("F", n, m, D, n, DF, n);

        /* Apply Q to D as D*QT */
        cgemqr("R", "C", n, m, k, AF, m, T, tsize, DF, n, work, lwork, &info);

        /* Compute |D*QT - D*QT| / |D| */
        cblas_cgemm(CblasColMajor, CblasNoTrans, CblasConjTrans, n, m, m,
                    &CNEGONE, D, n, Q, m, &ONE, DF, n);
        resid = clange("1", n, m, DF, n, rwork);
        if (cnorm > ZERO) {
            result[5] = resid / (eps * (1 > m ? 1 : m) * dnorm);
        } else {
            result[5] = ZERO;
        }

    } else {
        /* Short and wide */
        cgelq(m, n, AF, m, tquery, -1, workquery, -1, &info);
        tsize = (INT)crealf(tquery[0]);
        lwork = (INT)crealf(workquery[0]);
        cgemlq("R", "N", n, n, k, AF, m, tquery, tsize, Q, n, workquery, -1, &info);
        if ((INT)crealf(workquery[0]) > lwork) lwork = (INT)crealf(workquery[0]);
        cgemlq("L", "N", n, m, k, AF, m, tquery, tsize, DF, n, workquery, -1, &info);
        if ((INT)crealf(workquery[0]) > lwork) lwork = (INT)crealf(workquery[0]);
        cgemlq("L", "C", n, m, k, AF, m, tquery, tsize, DF, n, workquery, -1, &info);
        if ((INT)crealf(workquery[0]) > lwork) lwork = (INT)crealf(workquery[0]);
        cgemlq("R", "N", m, n, k, AF, m, tquery, tsize, CF, m, workquery, -1, &info);
        if ((INT)crealf(workquery[0]) > lwork) lwork = (INT)crealf(workquery[0]);
        cgemlq("R", "C", m, n, k, AF, m, tquery, tsize, CF, m, workquery, -1, &info);
        if ((INT)crealf(workquery[0]) > lwork) lwork = (INT)crealf(workquery[0]);

        T = (c64*)malloc(tsize * sizeof(c64));
        work = (c64*)malloc(lwork * sizeof(c64));

        cgelq(m, n, AF, m, T, tsize, work, lwork, &info);

        /* Generate the n-by-n matrix Q */
        claset("F", n, n, CZERO, ONE, Q, n);
        cgemlq("R", "N", n, n, k, AF, m, T, tsize, Q, n, work, lwork, &info);

        /* Copy R */
        claset("F", m, n, CZERO, CZERO, LQ, l);
        clacpy("L", m, n, AF, m, LQ, l);

        /* Compute |L - A*Q'| / |A| and store in RESULT(1) */
        cblas_cgemm(CblasColMajor, CblasNoTrans, CblasConjTrans, m, n, n,
                    &CNEGONE, A, m, Q, n, &ONE, LQ, l);
        anorm = clange("1", m, n, A, m, rwork);
        resid = clange("1", m, n, LQ, l, rwork);
        if (anorm > ZERO) {
            result[0] = resid / (eps * (1 > n ? 1 : n) * anorm);
        } else {
            result[0] = ZERO;
        }

        /* Compute |I - Q'*Q| and store in RESULT(2) */
        claset("F", n, n, CZERO, ONE, LQ, l);
        cblas_cherk(CblasColMajor, CblasUpper, CblasConjTrans, n, n, -1.0f, Q, n, 1.0f, LQ, l);
        resid = clansy("1", "U", n, LQ, l, rwork);
        result[1] = resid / (eps * (1 > n ? 1 : n));

        /* Generate random m-by-n matrix C and a copy CF (stored in D, DF) */
        for (j = 0; j < m; j++) {
            clarnv(2, iseed, n, &D[j * n]);
        }
        dnorm = clange("1", n, m, D, n, rwork);
        clacpy("F", n, m, D, n, DF, n);

        /* Apply Q to C as Q*C */
        cgemlq("L", "N", n, m, k, AF, m, T, tsize, DF, n, work, lwork, &info);

        /* Compute |Q*D - Q*D| / |D| */
        cblas_cgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, m, n,
                    &CNEGONE, Q, n, D, n, &ONE, DF, n);
        resid = clange("1", n, m, DF, n, rwork);
        if (dnorm > ZERO) {
            result[2] = resid / (eps * (1 > n ? 1 : n) * dnorm);
        } else {
            result[2] = ZERO;
        }

        /* Copy D into DF again */
        clacpy("F", n, m, D, n, DF, n);

        /* Apply Q to D as QT*D */
        cgemlq("L", "C", n, m, k, AF, m, T, tsize, DF, n, work, lwork, &info);

        /* Compute |QT*D - QT*D| / |D| */
        cblas_cgemm(CblasColMajor, CblasConjTrans, CblasNoTrans, n, m, n,
                    &CNEGONE, Q, n, D, n, &ONE, DF, n);
        resid = clange("1", n, m, DF, n, rwork);
        if (dnorm > ZERO) {
            result[3] = resid / (eps * (1 > n ? 1 : n) * dnorm);
        } else {
            result[3] = ZERO;
        }

        /* Generate random n-by-m matrix D and a copy DF (stored in C, CF) */
        for (j = 0; j < n; j++) {
            clarnv(2, iseed, m, &C[j * m]);
        }
        cnorm = clange("1", m, n, C, m, rwork);
        clacpy("F", m, n, C, m, CF, m);

        /* Apply Q to C as C*Q */
        cgemlq("R", "N", m, n, k, AF, m, T, tsize, CF, m, work, lwork, &info);

        /* Compute |C*Q - C*Q| / |C| */
        cblas_cgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, n,
                    &CNEGONE, C, m, Q, n, &ONE, CF, m);
        resid = clange("1", n, m, DF, n, rwork);
        if (cnorm > ZERO) {
            result[4] = resid / (eps * (1 > n ? 1 : n) * cnorm);
        } else {
            result[4] = ZERO;
        }

        /* Copy C into CF again */
        clacpy("F", m, n, C, m, CF, m);

        /* Apply Q to D as D*QT */
        cgemlq("R", "C", m, n, k, AF, m, T, tsize, CF, m, work, lwork, &info);

        /* Compute |C*QT - C*QT| / |C| */
        cblas_cgemm(CblasColMajor, CblasNoTrans, CblasConjTrans, m, n, n,
                    &CNEGONE, C, m, Q, n, &ONE, CF, m);
        resid = clange("1", m, n, CF, m, rwork);
        if (cnorm > ZERO) {
            result[5] = resid / (eps * (1 > n ? 1 : n) * cnorm);
        } else {
            result[5] = ZERO;
        }
    }

    /* Deallocate all arrays */
    free(A);
    free(AF);
    free(Q);
    free(R);
    free(rwork);
    free(C);
    free(CF);
    free(D);
    free(DF);
    free(LQ);
    free(T);
    free(work);
}
