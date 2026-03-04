/**
 * @file ztsqr01.c
 * @brief ZTSQR01 tests ZGEQR, ZGELQ, ZGEMLQ and ZGEMQR.
 *
 * Port of LAPACK TESTING/LIN/ztsqr01.f to C.
 */

#include <stdlib.h>
#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"

/**
 * ZTSQR01 tests ZGEQR, ZGELQ, ZGEMLQ and ZGEMQR.
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
void ztsqr01(const char* tssw, const INT m, const INT n, const INT mb,
             const INT nb, f64* result)
{
    const f64 ZERO = 0.0;
    const c128 ONE = CMPLX(1.0, 0.0);
    const c128 CZERO = CMPLX(0.0, 0.0);
    const c128 CNEGONE = CMPLX(-1.0, 0.0);

    INT ts, testzeros;
    INT info, j, k, l, lwork, tsize, mnb;
    f64 anorm, eps, resid, cnorm, dnorm;

    INT iseed[4] = {1988, 1989, 1990, 1991};
    c128 tquery[5], workquery[1];

    c128* A = NULL;
    c128* AF = NULL;
    c128* Q = NULL;
    c128* R = NULL;
    f64* rwork = NULL;
    c128* work = NULL;
    c128* T = NULL;
    c128* C = NULL;
    c128* CF = NULL;
    c128* D = NULL;
    c128* DF = NULL;
    c128* LQ = NULL;

    /* TEST TALL SKINNY OR SHORT WIDE */
    ts = (tssw[0] == 'T' || tssw[0] == 't') && (tssw[1] == 'S' || tssw[1] == 's');

    /* TEST MATRICES WITH HALF OF MATRIX BEING ZEROS */
    testzeros = 0;

    eps = dlamch("E");
    k = (m < n) ? m : n;
    l = m;
    if (n > l) l = n;
    if (1 > l) l = 1;
    mnb = (mb > nb) ? mb : nb;
    lwork = (3 > l ? 3 : l) * mnb;

    /* Dynamically allocate local arrays */
    A = (c128*)malloc(m * n * sizeof(c128));
    AF = (c128*)malloc(m * n * sizeof(c128));
    Q = (c128*)malloc(l * l * sizeof(c128));
    R = (c128*)malloc(m * l * sizeof(c128));
    rwork = (f64*)malloc(l * sizeof(f64));
    C = (c128*)malloc(m * n * sizeof(c128));
    CF = (c128*)malloc(m * n * sizeof(c128));
    D = (c128*)malloc(n * m * sizeof(c128));
    DF = (c128*)malloc(n * m * sizeof(c128));
    LQ = (c128*)malloc(l * n * sizeof(c128));

    /* Put random numbers into A and copy to AF */
    for (j = 0; j < n; j++) {
        zlarnv(2, iseed, m, &A[j * m]);
    }
    if (testzeros) {
        if (m >= 4) {
            for (j = 0; j < n; j++) {
                zlarnv(2, iseed, m / 2, &A[m / 4 + j * m]);
            }
        }
    }
    zlacpy("F", m, n, A, m, AF, m);

    if (ts) {
        /* Factor the matrix A in the array AF. */
        zgeqr(m, n, AF, m, tquery, -1, workquery, -1, &info);
        tsize = (INT)creal(tquery[0]);
        lwork = (INT)creal(workquery[0]);
        zgemqr("L", "N", m, m, k, AF, m, tquery, tsize, CF, m, workquery, -1, &info);
        if ((INT)creal(workquery[0]) > lwork) lwork = (INT)creal(workquery[0]);
        zgemqr("L", "N", m, n, k, AF, m, tquery, tsize, CF, m, workquery, -1, &info);
        if ((INT)creal(workquery[0]) > lwork) lwork = (INT)creal(workquery[0]);
        zgemqr("L", "C", m, n, k, AF, m, tquery, tsize, CF, m, workquery, -1, &info);
        if ((INT)creal(workquery[0]) > lwork) lwork = (INT)creal(workquery[0]);
        zgemqr("R", "N", n, m, k, AF, m, tquery, tsize, DF, n, workquery, -1, &info);
        if ((INT)creal(workquery[0]) > lwork) lwork = (INT)creal(workquery[0]);
        zgemqr("R", "C", n, m, k, AF, m, tquery, tsize, DF, n, workquery, -1, &info);
        if ((INT)creal(workquery[0]) > lwork) lwork = (INT)creal(workquery[0]);

        T = (c128*)malloc(tsize * sizeof(c128));
        work = (c128*)malloc(lwork * sizeof(c128));

        zgeqr(m, n, AF, m, T, tsize, work, lwork, &info);

        /* Generate the m-by-m matrix Q */
        zlaset("F", m, m, CZERO, ONE, Q, m);
        zgemqr("L", "N", m, m, k, AF, m, T, tsize, Q, m, work, lwork, &info);

        /* Copy R */
        zlaset("F", m, n, CZERO, CZERO, R, m);
        zlacpy("U", m, n, AF, m, R, m);

        /* Compute |R - Q'*A| / |A| and store in RESULT(1) */
        cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans, m, n, m,
                    &CNEGONE, Q, m, A, m, &ONE, R, m);
        anorm = zlange("1", m, n, A, m, rwork);
        resid = zlange("1", m, n, R, m, rwork);
        if (anorm > ZERO) {
            result[0] = resid / (eps * (1 > m ? 1 : m) * anorm);
        } else {
            result[0] = ZERO;
        }

        /* Compute |I - Q'*Q| and store in RESULT(2) */
        zlaset("F", m, m, CZERO, ONE, R, m);
        cblas_zherk(CblasColMajor, CblasUpper, CblasConjTrans, m, m, -1.0, Q, m, 1.0, R, m);
        resid = zlansy("1", "U", m, R, m, rwork);
        result[1] = resid / (eps * (1 > m ? 1 : m));

        /* Generate random m-by-n matrix C and a copy CF */
        for (j = 0; j < n; j++) {
            zlarnv(2, iseed, m, &C[j * m]);
        }
        cnorm = zlange("1", m, n, C, m, rwork);
        zlacpy("F", m, n, C, m, CF, m);

        /* Apply Q to C as Q*C */
        zgemqr("L", "N", m, n, k, AF, m, T, tsize, CF, m, work, lwork, &info);

        /* Compute |Q*C - Q*C| / |C| */
        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, m,
                    &CNEGONE, Q, m, C, m, &ONE, CF, m);
        resid = zlange("1", m, n, CF, m, rwork);
        if (cnorm > ZERO) {
            result[2] = resid / (eps * (1 > m ? 1 : m) * cnorm);
        } else {
            result[2] = ZERO;
        }

        /* Copy C into CF again */
        zlacpy("F", m, n, C, m, CF, m);

        /* Apply Q to C as QT*C */
        zgemqr("L", "C", m, n, k, AF, m, T, tsize, CF, m, work, lwork, &info);

        /* Compute |QT*C - QT*C| / |C| */
        cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans, m, n, m,
                    &CNEGONE, Q, m, C, m, &ONE, CF, m);
        resid = zlange("1", m, n, CF, m, rwork);
        if (cnorm > ZERO) {
            result[3] = resid / (eps * (1 > m ? 1 : m) * cnorm);
        } else {
            result[3] = ZERO;
        }

        /* Generate random n-by-m matrix D and a copy DF */
        for (j = 0; j < m; j++) {
            zlarnv(2, iseed, n, &D[j * n]);
        }
        dnorm = zlange("1", n, m, D, n, rwork);
        zlacpy("F", n, m, D, n, DF, n);

        /* Apply Q to D as D*Q */
        zgemqr("R", "N", n, m, k, AF, m, T, tsize, DF, n, work, lwork, &info);

        /* Compute |D*Q - D*Q| / |D| */
        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, m, m,
                    &CNEGONE, D, n, Q, m, &ONE, DF, n);
        resid = zlange("1", n, m, DF, n, rwork);
        if (dnorm > ZERO) {
            result[4] = resid / (eps * (1 > m ? 1 : m) * dnorm);
        } else {
            result[4] = ZERO;
        }

        /* Copy D into DF again */
        zlacpy("F", n, m, D, n, DF, n);

        /* Apply Q to D as D*QT */
        zgemqr("R", "C", n, m, k, AF, m, T, tsize, DF, n, work, lwork, &info);

        /* Compute |D*QT - D*QT| / |D| */
        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasConjTrans, n, m, m,
                    &CNEGONE, D, n, Q, m, &ONE, DF, n);
        resid = zlange("1", n, m, DF, n, rwork);
        if (cnorm > ZERO) {
            result[5] = resid / (eps * (1 > m ? 1 : m) * dnorm);
        } else {
            result[5] = ZERO;
        }

    } else {
        /* Short and wide */
        zgelq(m, n, AF, m, tquery, -1, workquery, -1, &info);
        tsize = (INT)creal(tquery[0]);
        lwork = (INT)creal(workquery[0]);
        zgemlq("R", "N", n, n, k, AF, m, tquery, tsize, Q, n, workquery, -1, &info);
        if ((INT)creal(workquery[0]) > lwork) lwork = (INT)creal(workquery[0]);
        zgemlq("L", "N", n, m, k, AF, m, tquery, tsize, DF, n, workquery, -1, &info);
        if ((INT)creal(workquery[0]) > lwork) lwork = (INT)creal(workquery[0]);
        zgemlq("L", "C", n, m, k, AF, m, tquery, tsize, DF, n, workquery, -1, &info);
        if ((INT)creal(workquery[0]) > lwork) lwork = (INT)creal(workquery[0]);
        zgemlq("R", "N", m, n, k, AF, m, tquery, tsize, CF, m, workquery, -1, &info);
        if ((INT)creal(workquery[0]) > lwork) lwork = (INT)creal(workquery[0]);
        zgemlq("R", "C", m, n, k, AF, m, tquery, tsize, CF, m, workquery, -1, &info);
        if ((INT)creal(workquery[0]) > lwork) lwork = (INT)creal(workquery[0]);

        T = (c128*)malloc(tsize * sizeof(c128));
        work = (c128*)malloc(lwork * sizeof(c128));

        zgelq(m, n, AF, m, T, tsize, work, lwork, &info);

        /* Generate the n-by-n matrix Q */
        zlaset("F", n, n, CZERO, ONE, Q, n);
        zgemlq("R", "N", n, n, k, AF, m, T, tsize, Q, n, work, lwork, &info);

        /* Copy R */
        zlaset("F", m, n, CZERO, CZERO, LQ, l);
        zlacpy("L", m, n, AF, m, LQ, l);

        /* Compute |L - A*Q'| / |A| and store in RESULT(1) */
        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasConjTrans, m, n, n,
                    &CNEGONE, A, m, Q, n, &ONE, LQ, l);
        anorm = zlange("1", m, n, A, m, rwork);
        resid = zlange("1", m, n, LQ, l, rwork);
        if (anorm > ZERO) {
            result[0] = resid / (eps * (1 > n ? 1 : n) * anorm);
        } else {
            result[0] = ZERO;
        }

        /* Compute |I - Q'*Q| and store in RESULT(2) */
        zlaset("F", n, n, CZERO, ONE, LQ, l);
        cblas_zherk(CblasColMajor, CblasUpper, CblasConjTrans, n, n, -1.0, Q, n, 1.0, LQ, l);
        resid = zlansy("1", "U", n, LQ, l, rwork);
        result[1] = resid / (eps * (1 > n ? 1 : n));

        /* Generate random m-by-n matrix C and a copy CF (stored in D, DF) */
        for (j = 0; j < m; j++) {
            zlarnv(2, iseed, n, &D[j * n]);
        }
        dnorm = zlange("1", n, m, D, n, rwork);
        zlacpy("F", n, m, D, n, DF, n);

        /* Apply Q to C as Q*C */
        zgemlq("L", "N", n, m, k, AF, m, T, tsize, DF, n, work, lwork, &info);

        /* Compute |Q*D - Q*D| / |D| */
        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, m, n,
                    &CNEGONE, Q, n, D, n, &ONE, DF, n);
        resid = zlange("1", n, m, DF, n, rwork);
        if (dnorm > ZERO) {
            result[2] = resid / (eps * (1 > n ? 1 : n) * dnorm);
        } else {
            result[2] = ZERO;
        }

        /* Copy D into DF again */
        zlacpy("F", n, m, D, n, DF, n);

        /* Apply Q to D as QT*D */
        zgemlq("L", "C", n, m, k, AF, m, T, tsize, DF, n, work, lwork, &info);

        /* Compute |QT*D - QT*D| / |D| */
        cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans, n, m, n,
                    &CNEGONE, Q, n, D, n, &ONE, DF, n);
        resid = zlange("1", n, m, DF, n, rwork);
        if (dnorm > ZERO) {
            result[3] = resid / (eps * (1 > n ? 1 : n) * dnorm);
        } else {
            result[3] = ZERO;
        }

        /* Generate random n-by-m matrix D and a copy DF (stored in C, CF) */
        for (j = 0; j < n; j++) {
            zlarnv(2, iseed, m, &C[j * m]);
        }
        cnorm = zlange("1", m, n, C, m, rwork);
        zlacpy("F", m, n, C, m, CF, m);

        /* Apply Q to C as C*Q */
        zgemlq("R", "N", m, n, k, AF, m, T, tsize, CF, m, work, lwork, &info);

        /* Compute |C*Q - C*Q| / |C| */
        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, n,
                    &CNEGONE, C, m, Q, n, &ONE, CF, m);
        resid = zlange("1", n, m, DF, n, rwork);
        if (cnorm > ZERO) {
            result[4] = resid / (eps * (1 > n ? 1 : n) * cnorm);
        } else {
            result[4] = ZERO;
        }

        /* Copy C into CF again */
        zlacpy("F", m, n, C, m, CF, m);

        /* Apply Q to D as D*QT */
        zgemlq("R", "C", m, n, k, AF, m, T, tsize, CF, m, work, lwork, &info);

        /* Compute |C*QT - C*QT| / |C| */
        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasConjTrans, m, n, n,
                    &CNEGONE, C, m, Q, n, &ONE, CF, m);
        resid = zlange("1", m, n, CF, m, rwork);
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
