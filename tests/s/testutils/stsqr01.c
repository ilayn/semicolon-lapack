/**
 * @file stsqr01.c
 * @brief STSQR01 tests SGEQR, SGELQ, SGEMLQ and SGEMQR.
 *
 * Port of LAPACK TESTING/LIN/stsqr01.f to C.
 */

#include <stdlib.h>
#include <math.h>
#include <cblas.h>
#include "verify.h"

/* External declarations for LAPACK routines */
extern f32 slamch(const char* cmach);
extern f32 slange(const char* norm, const int m, const int n,
                     const f32* A, const int lda, f32* work);
extern f32 slansy(const char* norm, const char* uplo, const int n,
                     const f32* A, const int lda, f32* work);
extern void slaset(const char* uplo, const int m, const int n,
                   const f32 alpha, const f32 beta,
                   f32* A, const int lda);
extern void slacpy(const char* uplo, const int m, const int n,
                   const f32* A, const int lda,
                   f32* B, const int ldb);
extern void sgeqr(const int m, const int n, f32* A, const int lda,
                  f32* T, const int tsize, f32* work, const int lwork,
                  int* info);
extern void sgelq(const int m, const int n, f32* A, const int lda,
                  f32* T, const int tsize, f32* work, const int lwork,
                  int* info);
extern void sgemqr(const char* side, const char* trans,
                   const int m, const int n, const int k,
                   const f32* A, const int lda,
                   const f32* T, const int tsize,
                   f32* C, const int ldc,
                   f32* work, const int lwork, int* info);
extern void sgemlq(const char* side, const char* trans,
                   const int m, const int n, const int k,
                   const f32* A, const int lda,
                   const f32* T, const int tsize,
                   f32* C, const int ldc,
                   f32* work, const int lwork, int* info);
extern void slarnv(const int idist, int* iseed, const int n, f32* x);

/**
 * STSQR01 tests SGEQR, SGELQ, SGEMLQ and SGEMQR.
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
void stsqr01(const char* tssw, const int m, const int n, const int mb,
             const int nb, f32* result)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;

    int ts, testzeros;
    int info, j, k, l, lwork, tsize, mnb;
    f32 anorm, eps, resid, cnorm, dnorm;

    int iseed[4] = {1988, 1989, 1990, 1991};
    f32 tquery[5], workquery[1];

    f32* A = NULL;
    f32* AF = NULL;
    f32* Q = NULL;
    f32* R = NULL;
    f32* rwork = NULL;
    f32* work = NULL;
    f32* T = NULL;
    f32* C = NULL;
    f32* CF = NULL;
    f32* D = NULL;
    f32* DF = NULL;
    f32* LQ = NULL;

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
    A = (f32*)malloc(m * n * sizeof(f32));
    AF = (f32*)malloc(m * n * sizeof(f32));
    Q = (f32*)malloc(l * l * sizeof(f32));
    R = (f32*)malloc(m * l * sizeof(f32));
    rwork = (f32*)malloc(l * sizeof(f32));
    C = (f32*)malloc(m * n * sizeof(f32));
    CF = (f32*)malloc(m * n * sizeof(f32));
    D = (f32*)malloc(n * m * sizeof(f32));
    DF = (f32*)malloc(n * m * sizeof(f32));
    LQ = (f32*)malloc(l * n * sizeof(f32));

    /* Put random numbers into A and copy to AF */
    for (j = 0; j < n; j++) {
        slarnv(2, iseed, m, &A[j * m]);
    }
    if (testzeros) {
        if (m >= 4) {
            for (j = 0; j < n; j++) {
                slarnv(2, iseed, m / 2, &A[m / 4 + j * m]);
            }
        }
    }
    slacpy("F", m, n, A, m, AF, m);

    if (ts) {
        /* Factor the matrix A in the array AF. */
        sgeqr(m, n, AF, m, tquery, -1, workquery, -1, &info);
        tsize = (int)tquery[0];
        lwork = (int)workquery[0];
        sgemqr("L", "N", m, m, k, AF, m, tquery, tsize, CF, m, workquery, -1, &info);
        if ((int)workquery[0] > lwork) lwork = (int)workquery[0];
        sgemqr("L", "N", m, n, k, AF, m, tquery, tsize, CF, m, workquery, -1, &info);
        if ((int)workquery[0] > lwork) lwork = (int)workquery[0];
        sgemqr("L", "T", m, n, k, AF, m, tquery, tsize, CF, m, workquery, -1, &info);
        if ((int)workquery[0] > lwork) lwork = (int)workquery[0];
        sgemqr("R", "N", n, m, k, AF, m, tquery, tsize, DF, n, workquery, -1, &info);
        if ((int)workquery[0] > lwork) lwork = (int)workquery[0];
        sgemqr("R", "T", n, m, k, AF, m, tquery, tsize, DF, n, workquery, -1, &info);
        if ((int)workquery[0] > lwork) lwork = (int)workquery[0];

        T = (f32*)malloc(tsize * sizeof(f32));
        work = (f32*)malloc(lwork * sizeof(f32));

        sgeqr(m, n, AF, m, T, tsize, work, lwork, &info);

        /* Generate the m-by-m matrix Q */
        slaset("F", m, m, ZERO, ONE, Q, m);
        sgemqr("L", "N", m, m, k, AF, m, T, tsize, Q, m, work, lwork, &info);

        /* Copy R */
        slaset("F", m, n, ZERO, ZERO, R, m);
        slacpy("U", m, n, AF, m, R, m);

        /* Compute |R - Q'*A| / |A| and store in RESULT(1) */
        cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, m, n, m,
                    -ONE, Q, m, A, m, ONE, R, m);
        anorm = slange("1", m, n, A, m, rwork);
        resid = slange("1", m, n, R, m, rwork);
        if (anorm > ZERO) {
            result[0] = resid / (eps * (1 > m ? 1 : m) * anorm);
        } else {
            result[0] = ZERO;
        }

        /* Compute |I - Q'*Q| and store in RESULT(2) */
        slaset("F", m, m, ZERO, ONE, R, m);
        cblas_ssyrk(CblasColMajor, CblasUpper, CblasTrans, m, m, -ONE, Q, m, ONE, R, m);
        resid = slansy("1", "U", m, R, m, rwork);
        result[1] = resid / (eps * (1 > m ? 1 : m));

        /* Generate random m-by-n matrix C and a copy CF */
        for (j = 0; j < n; j++) {
            slarnv(2, iseed, m, &C[j * m]);
        }
        cnorm = slange("1", m, n, C, m, rwork);
        slacpy("F", m, n, C, m, CF, m);

        /* Apply Q to C as Q*C */
        sgemqr("L", "N", m, n, k, AF, m, T, tsize, CF, m, work, lwork, &info);

        /* Compute |Q*C - Q*C| / |C| */
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, m,
                    -ONE, Q, m, C, m, ONE, CF, m);
        resid = slange("1", m, n, CF, m, rwork);
        if (cnorm > ZERO) {
            result[2] = resid / (eps * (1 > m ? 1 : m) * cnorm);
        } else {
            result[2] = ZERO;
        }

        /* Copy C into CF again */
        slacpy("F", m, n, C, m, CF, m);

        /* Apply Q to C as QT*C */
        sgemqr("L", "T", m, n, k, AF, m, T, tsize, CF, m, work, lwork, &info);

        /* Compute |QT*C - QT*C| / |C| */
        cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, m, n, m,
                    -ONE, Q, m, C, m, ONE, CF, m);
        resid = slange("1", m, n, CF, m, rwork);
        if (cnorm > ZERO) {
            result[3] = resid / (eps * (1 > m ? 1 : m) * cnorm);
        } else {
            result[3] = ZERO;
        }

        /* Generate random n-by-m matrix D and a copy DF */
        for (j = 0; j < m; j++) {
            slarnv(2, iseed, n, &D[j * n]);
        }
        dnorm = slange("1", n, m, D, n, rwork);
        slacpy("F", n, m, D, n, DF, n);

        /* Apply Q to D as D*Q */
        sgemqr("R", "N", n, m, k, AF, m, T, tsize, DF, n, work, lwork, &info);

        /* Compute |D*Q - D*Q| / |D| */
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, m, m,
                    -ONE, D, n, Q, m, ONE, DF, n);
        resid = slange("1", n, m, DF, n, rwork);
        if (dnorm > ZERO) {
            result[4] = resid / (eps * (1 > m ? 1 : m) * dnorm);
        } else {
            result[4] = ZERO;
        }

        /* Copy D into DF again */
        slacpy("F", n, m, D, n, DF, n);

        /* Apply Q to D as D*QT */
        sgemqr("R", "T", n, m, k, AF, m, T, tsize, DF, n, work, lwork, &info);

        /* Compute |D*QT - D*QT| / |D| */
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans, n, m, m,
                    -ONE, D, n, Q, m, ONE, DF, n);
        resid = slange("1", n, m, DF, n, rwork);
        if (cnorm > ZERO) {
            result[5] = resid / (eps * (1 > m ? 1 : m) * dnorm);
        } else {
            result[5] = ZERO;
        }

    } else {
        /* Short and wide */
        sgelq(m, n, AF, m, tquery, -1, workquery, -1, &info);
        tsize = (int)tquery[0];
        lwork = (int)workquery[0];
        sgemlq("R", "N", n, n, k, AF, m, tquery, tsize, Q, n, workquery, -1, &info);
        if ((int)workquery[0] > lwork) lwork = (int)workquery[0];
        sgemlq("L", "N", n, m, k, AF, m, tquery, tsize, DF, n, workquery, -1, &info);
        if ((int)workquery[0] > lwork) lwork = (int)workquery[0];
        sgemlq("L", "T", n, m, k, AF, m, tquery, tsize, DF, n, workquery, -1, &info);
        if ((int)workquery[0] > lwork) lwork = (int)workquery[0];
        sgemlq("R", "N", m, n, k, AF, m, tquery, tsize, CF, m, workquery, -1, &info);
        if ((int)workquery[0] > lwork) lwork = (int)workquery[0];
        sgemlq("R", "T", m, n, k, AF, m, tquery, tsize, CF, m, workquery, -1, &info);
        if ((int)workquery[0] > lwork) lwork = (int)workquery[0];

        T = (f32*)malloc(tsize * sizeof(f32));
        work = (f32*)malloc(lwork * sizeof(f32));

        sgelq(m, n, AF, m, T, tsize, work, lwork, &info);

        /* Generate the n-by-n matrix Q */
        slaset("F", n, n, ZERO, ONE, Q, n);
        sgemlq("R", "N", n, n, k, AF, m, T, tsize, Q, n, work, lwork, &info);

        /* Copy R */
        slaset("F", m, n, ZERO, ZERO, LQ, l);
        slacpy("L", m, n, AF, m, LQ, l);

        /* Compute |L - A*Q'| / |A| and store in RESULT(1) */
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans, m, n, n,
                    -ONE, A, m, Q, n, ONE, LQ, l);
        anorm = slange("1", m, n, A, m, rwork);
        resid = slange("1", m, n, LQ, l, rwork);
        if (anorm > ZERO) {
            result[0] = resid / (eps * (1 > n ? 1 : n) * anorm);
        } else {
            result[0] = ZERO;
        }

        /* Compute |I - Q'*Q| and store in RESULT(2) */
        slaset("F", n, n, ZERO, ONE, LQ, l);
        cblas_ssyrk(CblasColMajor, CblasUpper, CblasTrans, n, n, -ONE, Q, n, ONE, LQ, l);
        resid = slansy("1", "U", n, LQ, l, rwork);
        result[1] = resid / (eps * (1 > n ? 1 : n));

        /* Generate random m-by-n matrix C and a copy CF (stored in D, DF) */
        for (j = 0; j < m; j++) {
            slarnv(2, iseed, n, &D[j * n]);
        }
        dnorm = slange("1", n, m, D, n, rwork);
        slacpy("F", n, m, D, n, DF, n);

        /* Apply Q to C as Q*C */
        sgemlq("L", "N", n, m, k, AF, m, T, tsize, DF, n, work, lwork, &info);

        /* Compute |Q*D - Q*D| / |D| */
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, m, n,
                    -ONE, Q, n, D, n, ONE, DF, n);
        resid = slange("1", n, m, DF, n, rwork);
        if (dnorm > ZERO) {
            result[2] = resid / (eps * (1 > n ? 1 : n) * dnorm);
        } else {
            result[2] = ZERO;
        }

        /* Copy D into DF again */
        slacpy("F", n, m, D, n, DF, n);

        /* Apply Q to D as QT*D */
        sgemlq("L", "T", n, m, k, AF, m, T, tsize, DF, n, work, lwork, &info);

        /* Compute |QT*D - QT*D| / |D| */
        cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, n, m, n,
                    -ONE, Q, n, D, n, ONE, DF, n);
        resid = slange("1", n, m, DF, n, rwork);
        if (dnorm > ZERO) {
            result[3] = resid / (eps * (1 > n ? 1 : n) * dnorm);
        } else {
            result[3] = ZERO;
        }

        /* Generate random n-by-m matrix D and a copy DF (stored in C, CF) */
        for (j = 0; j < n; j++) {
            slarnv(2, iseed, m, &C[j * m]);
        }
        cnorm = slange("1", m, n, C, m, rwork);
        slacpy("F", m, n, C, m, CF, m);

        /* Apply Q to C as C*Q */
        sgemlq("R", "N", m, n, k, AF, m, T, tsize, CF, m, work, lwork, &info);

        /* Compute |C*Q - C*Q| / |C| */
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, n,
                    -ONE, C, m, Q, n, ONE, CF, m);
        resid = slange("1", n, m, DF, n, rwork);
        if (cnorm > ZERO) {
            result[4] = resid / (eps * (1 > n ? 1 : n) * cnorm);
        } else {
            result[4] = ZERO;
        }

        /* Copy C into CF again */
        slacpy("F", m, n, C, m, CF, m);

        /* Apply Q to D as D*QT */
        sgemlq("R", "T", m, n, k, AF, m, T, tsize, CF, m, work, lwork, &info);

        /* Compute |C*QT - C*QT| / |C| */
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans, m, n, n,
                    -ONE, C, m, Q, n, ONE, CF, m);
        resid = slange("1", m, n, CF, m, rwork);
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
