/**
 * @file dtsqr01.c
 * @brief DTSQR01 tests DGEQR, DGELQ, DGEMLQ and DGEMQR.
 *
 * Port of LAPACK TESTING/LIN/dtsqr01.f to C.
 */

#include <stdlib.h>
#include <math.h>
#include <cblas.h>
#include "verify.h"

/* External declarations for LAPACK routines */
extern double dlamch(const char* cmach);
extern double dlange(const char* norm, const int m, const int n,
                     const double* A, const int lda, double* work);
extern double dlansy(const char* norm, const char* uplo, const int n,
                     const double* A, const int lda, double* work);
extern void dlaset(const char* uplo, const int m, const int n,
                   const double alpha, const double beta,
                   double* A, const int lda);
extern void dlacpy(const char* uplo, const int m, const int n,
                   const double* A, const int lda,
                   double* B, const int ldb);
extern void dgeqr(const int m, const int n, double* A, const int lda,
                  double* T, const int tsize, double* work, const int lwork,
                  int* info);
extern void dgelq(const int m, const int n, double* A, const int lda,
                  double* T, const int tsize, double* work, const int lwork,
                  int* info);
extern void dgemqr(const char* side, const char* trans,
                   const int m, const int n, const int k,
                   const double* A, const int lda,
                   const double* T, const int tsize,
                   double* C, const int ldc,
                   double* work, const int lwork, int* info);
extern void dgemlq(const char* side, const char* trans,
                   const int m, const int n, const int k,
                   const double* A, const int lda,
                   const double* T, const int tsize,
                   double* C, const int ldc,
                   double* work, const int lwork, int* info);
extern void dlarnv(const int idist, int* iseed, const int n, double* x);

/**
 * DTSQR01 tests DGEQR, DGELQ, DGEMLQ and DGEMQR.
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
void dtsqr01(const char* tssw, const int m, const int n, const int mb,
             const int nb, double* result)
{
    const double ZERO = 0.0;
    const double ONE = 1.0;

    int ts, testzeros;
    int info, j, k, l, lwork, tsize, mnb;
    double anorm, eps, resid, cnorm, dnorm;

    int iseed[4] = {1988, 1989, 1990, 1991};
    double tquery[5], workquery[1];

    double* A = NULL;
    double* AF = NULL;
    double* Q = NULL;
    double* R = NULL;
    double* rwork = NULL;
    double* work = NULL;
    double* T = NULL;
    double* C = NULL;
    double* CF = NULL;
    double* D = NULL;
    double* DF = NULL;
    double* LQ = NULL;

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
    A = (double*)malloc(m * n * sizeof(double));
    AF = (double*)malloc(m * n * sizeof(double));
    Q = (double*)malloc(l * l * sizeof(double));
    R = (double*)malloc(m * l * sizeof(double));
    rwork = (double*)malloc(l * sizeof(double));
    C = (double*)malloc(m * n * sizeof(double));
    CF = (double*)malloc(m * n * sizeof(double));
    D = (double*)malloc(n * m * sizeof(double));
    DF = (double*)malloc(n * m * sizeof(double));
    LQ = (double*)malloc(l * n * sizeof(double));

    /* Put random numbers into A and copy to AF */
    for (j = 0; j < n; j++) {
        dlarnv(2, iseed, m, &A[j * m]);
    }
    if (testzeros) {
        if (m >= 4) {
            for (j = 0; j < n; j++) {
                dlarnv(2, iseed, m / 2, &A[m / 4 + j * m]);
            }
        }
    }
    dlacpy("F", m, n, A, m, AF, m);

    if (ts) {
        /* Factor the matrix A in the array AF. */
        dgeqr(m, n, AF, m, tquery, -1, workquery, -1, &info);
        tsize = (int)tquery[0];
        lwork = (int)workquery[0];
        dgemqr("L", "N", m, m, k, AF, m, tquery, tsize, CF, m, workquery, -1, &info);
        if ((int)workquery[0] > lwork) lwork = (int)workquery[0];
        dgemqr("L", "N", m, n, k, AF, m, tquery, tsize, CF, m, workquery, -1, &info);
        if ((int)workquery[0] > lwork) lwork = (int)workquery[0];
        dgemqr("L", "T", m, n, k, AF, m, tquery, tsize, CF, m, workquery, -1, &info);
        if ((int)workquery[0] > lwork) lwork = (int)workquery[0];
        dgemqr("R", "N", n, m, k, AF, m, tquery, tsize, DF, n, workquery, -1, &info);
        if ((int)workquery[0] > lwork) lwork = (int)workquery[0];
        dgemqr("R", "T", n, m, k, AF, m, tquery, tsize, DF, n, workquery, -1, &info);
        if ((int)workquery[0] > lwork) lwork = (int)workquery[0];

        T = (double*)malloc(tsize * sizeof(double));
        work = (double*)malloc(lwork * sizeof(double));

        dgeqr(m, n, AF, m, T, tsize, work, lwork, &info);

        /* Generate the m-by-m matrix Q */
        dlaset("F", m, m, ZERO, ONE, Q, m);
        dgemqr("L", "N", m, m, k, AF, m, T, tsize, Q, m, work, lwork, &info);

        /* Copy R */
        dlaset("F", m, n, ZERO, ZERO, R, m);
        dlacpy("U", m, n, AF, m, R, m);

        /* Compute |R - Q'*A| / |A| and store in RESULT(1) */
        cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, m, n, m,
                    -ONE, Q, m, A, m, ONE, R, m);
        anorm = dlange("1", m, n, A, m, rwork);
        resid = dlange("1", m, n, R, m, rwork);
        if (anorm > ZERO) {
            result[0] = resid / (eps * (1 > m ? 1 : m) * anorm);
        } else {
            result[0] = ZERO;
        }

        /* Compute |I - Q'*Q| and store in RESULT(2) */
        dlaset("F", m, m, ZERO, ONE, R, m);
        cblas_dsyrk(CblasColMajor, CblasUpper, CblasTrans, m, m, -ONE, Q, m, ONE, R, m);
        resid = dlansy("1", "U", m, R, m, rwork);
        result[1] = resid / (eps * (1 > m ? 1 : m));

        /* Generate random m-by-n matrix C and a copy CF */
        for (j = 0; j < n; j++) {
            dlarnv(2, iseed, m, &C[j * m]);
        }
        cnorm = dlange("1", m, n, C, m, rwork);
        dlacpy("F", m, n, C, m, CF, m);

        /* Apply Q to C as Q*C */
        dgemqr("L", "N", m, n, k, AF, m, T, tsize, CF, m, work, lwork, &info);

        /* Compute |Q*C - Q*C| / |C| */
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, m,
                    -ONE, Q, m, C, m, ONE, CF, m);
        resid = dlange("1", m, n, CF, m, rwork);
        if (cnorm > ZERO) {
            result[2] = resid / (eps * (1 > m ? 1 : m) * cnorm);
        } else {
            result[2] = ZERO;
        }

        /* Copy C into CF again */
        dlacpy("F", m, n, C, m, CF, m);

        /* Apply Q to C as QT*C */
        dgemqr("L", "T", m, n, k, AF, m, T, tsize, CF, m, work, lwork, &info);

        /* Compute |QT*C - QT*C| / |C| */
        cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, m, n, m,
                    -ONE, Q, m, C, m, ONE, CF, m);
        resid = dlange("1", m, n, CF, m, rwork);
        if (cnorm > ZERO) {
            result[3] = resid / (eps * (1 > m ? 1 : m) * cnorm);
        } else {
            result[3] = ZERO;
        }

        /* Generate random n-by-m matrix D and a copy DF */
        for (j = 0; j < m; j++) {
            dlarnv(2, iseed, n, &D[j * n]);
        }
        dnorm = dlange("1", n, m, D, n, rwork);
        dlacpy("F", n, m, D, n, DF, n);

        /* Apply Q to D as D*Q */
        dgemqr("R", "N", n, m, k, AF, m, T, tsize, DF, n, work, lwork, &info);

        /* Compute |D*Q - D*Q| / |D| */
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, m, m,
                    -ONE, D, n, Q, m, ONE, DF, n);
        resid = dlange("1", n, m, DF, n, rwork);
        if (dnorm > ZERO) {
            result[4] = resid / (eps * (1 > m ? 1 : m) * dnorm);
        } else {
            result[4] = ZERO;
        }

        /* Copy D into DF again */
        dlacpy("F", n, m, D, n, DF, n);

        /* Apply Q to D as D*QT */
        dgemqr("R", "T", n, m, k, AF, m, T, tsize, DF, n, work, lwork, &info);

        /* Compute |D*QT - D*QT| / |D| */
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, n, m, m,
                    -ONE, D, n, Q, m, ONE, DF, n);
        resid = dlange("1", n, m, DF, n, rwork);
        if (cnorm > ZERO) {
            result[5] = resid / (eps * (1 > m ? 1 : m) * dnorm);
        } else {
            result[5] = ZERO;
        }

    } else {
        /* Short and wide */
        dgelq(m, n, AF, m, tquery, -1, workquery, -1, &info);
        tsize = (int)tquery[0];
        lwork = (int)workquery[0];
        dgemlq("R", "N", n, n, k, AF, m, tquery, tsize, Q, n, workquery, -1, &info);
        if ((int)workquery[0] > lwork) lwork = (int)workquery[0];
        dgemlq("L", "N", n, m, k, AF, m, tquery, tsize, DF, n, workquery, -1, &info);
        if ((int)workquery[0] > lwork) lwork = (int)workquery[0];
        dgemlq("L", "T", n, m, k, AF, m, tquery, tsize, DF, n, workquery, -1, &info);
        if ((int)workquery[0] > lwork) lwork = (int)workquery[0];
        dgemlq("R", "N", m, n, k, AF, m, tquery, tsize, CF, m, workquery, -1, &info);
        if ((int)workquery[0] > lwork) lwork = (int)workquery[0];
        dgemlq("R", "T", m, n, k, AF, m, tquery, tsize, CF, m, workquery, -1, &info);
        if ((int)workquery[0] > lwork) lwork = (int)workquery[0];

        T = (double*)malloc(tsize * sizeof(double));
        work = (double*)malloc(lwork * sizeof(double));

        dgelq(m, n, AF, m, T, tsize, work, lwork, &info);

        /* Generate the n-by-n matrix Q */
        dlaset("F", n, n, ZERO, ONE, Q, n);
        dgemlq("R", "N", n, n, k, AF, m, T, tsize, Q, n, work, lwork, &info);

        /* Copy R */
        dlaset("F", m, n, ZERO, ZERO, LQ, l);
        dlacpy("L", m, n, AF, m, LQ, l);

        /* Compute |L - A*Q'| / |A| and store in RESULT(1) */
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, m, n, n,
                    -ONE, A, m, Q, n, ONE, LQ, l);
        anorm = dlange("1", m, n, A, m, rwork);
        resid = dlange("1", m, n, LQ, l, rwork);
        if (anorm > ZERO) {
            result[0] = resid / (eps * (1 > n ? 1 : n) * anorm);
        } else {
            result[0] = ZERO;
        }

        /* Compute |I - Q'*Q| and store in RESULT(2) */
        dlaset("F", n, n, ZERO, ONE, LQ, l);
        cblas_dsyrk(CblasColMajor, CblasUpper, CblasTrans, n, n, -ONE, Q, n, ONE, LQ, l);
        resid = dlansy("1", "U", n, LQ, l, rwork);
        result[1] = resid / (eps * (1 > n ? 1 : n));

        /* Generate random m-by-n matrix C and a copy CF (stored in D, DF) */
        for (j = 0; j < m; j++) {
            dlarnv(2, iseed, n, &D[j * n]);
        }
        dnorm = dlange("1", n, m, D, n, rwork);
        dlacpy("F", n, m, D, n, DF, n);

        /* Apply Q to C as Q*C */
        dgemlq("L", "N", n, m, k, AF, m, T, tsize, DF, n, work, lwork, &info);

        /* Compute |Q*D - Q*D| / |D| */
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, m, n,
                    -ONE, Q, n, D, n, ONE, DF, n);
        resid = dlange("1", n, m, DF, n, rwork);
        if (dnorm > ZERO) {
            result[2] = resid / (eps * (1 > n ? 1 : n) * dnorm);
        } else {
            result[2] = ZERO;
        }

        /* Copy D into DF again */
        dlacpy("F", n, m, D, n, DF, n);

        /* Apply Q to D as QT*D */
        dgemlq("L", "T", n, m, k, AF, m, T, tsize, DF, n, work, lwork, &info);

        /* Compute |QT*D - QT*D| / |D| */
        cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, n, m, n,
                    -ONE, Q, n, D, n, ONE, DF, n);
        resid = dlange("1", n, m, DF, n, rwork);
        if (dnorm > ZERO) {
            result[3] = resid / (eps * (1 > n ? 1 : n) * dnorm);
        } else {
            result[3] = ZERO;
        }

        /* Generate random n-by-m matrix D and a copy DF (stored in C, CF) */
        for (j = 0; j < n; j++) {
            dlarnv(2, iseed, m, &C[j * m]);
        }
        cnorm = dlange("1", m, n, C, m, rwork);
        dlacpy("F", m, n, C, m, CF, m);

        /* Apply Q to C as C*Q */
        dgemlq("R", "N", m, n, k, AF, m, T, tsize, CF, m, work, lwork, &info);

        /* Compute |C*Q - C*Q| / |C| */
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, n,
                    -ONE, C, m, Q, n, ONE, CF, m);
        resid = dlange("1", n, m, DF, n, rwork);
        if (cnorm > ZERO) {
            result[4] = resid / (eps * (1 > n ? 1 : n) * cnorm);
        } else {
            result[4] = ZERO;
        }

        /* Copy C into CF again */
        dlacpy("F", m, n, C, m, CF, m);

        /* Apply Q to D as D*QT */
        dgemlq("R", "T", m, n, k, AF, m, T, tsize, CF, m, work, lwork, &info);

        /* Compute |C*QT - C*QT| / |C| */
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, m, n, n,
                    -ONE, C, m, Q, n, ONE, CF, m);
        resid = dlange("1", m, n, CF, m, rwork);
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
