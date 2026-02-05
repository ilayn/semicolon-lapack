/**
 * @file dget22.c
 * @brief DGET22 performs an eigenvector check.
 *
 * Faithful port from LAPACK TESTING/EIG/dget22.f
 */

#include <math.h>
#include <float.h>
#include <cblas.h>
#include <string.h>
#include "verify.h"

/* Forward declarations */
extern double dlamch(const char* cmach);
extern double dlange(const char* norm, const int m, const int n,
                     const double* A, const int lda, double* work);
extern void dlaset(const char* uplo, const int m, const int n,
                   const double alpha, const double beta,
                   double* A, const int lda);

/**
 * DGET22 does an eigenvector check.
 *
 * The basic test is:
 *    RESULT(0) = | A E  -  E W | / ( |A| |E| ulp )
 *
 * using the 1-norm.  It also tests the normalization of E:
 *    RESULT(1) = max | m-norm(E(j)) - 1 | / ( n ulp )
 *                 j
 *
 * where E(j) is the j-th eigenvector, and m-norm is the max-norm of a
 * vector.  If an eigenvector is complex, as determined from WI(j)
 * nonzero, then the max-norm of the vector ( er + i*ei ) is the maximum
 * of  |er(1)| + |ei(1)|, ... , |er(n)| + |ei(n)|
 *
 * W is a block diagonal matrix, with a 1 by 1 block for each real
 * eigenvalue and a 2 by 2 block for each complex conjugate pair.
 * If eigenvalues j and j+1 are a complex conjugate pair, so that
 * WR(j) = WR(j+1) = wr and WI(j) = - WI(j+1) = wi, then the 2 by 2
 * block corresponding to the pair will be:
 *
 *    (  wr  wi  )
 *    ( -wi  wr  )
 *
 * Such a block multiplying an n by 2 matrix ( ur ui ) on the right
 * will be the same as multiplying  ur + i*ui  by  wr + i*wi.
 *
 * @param[in] transa  'N': no transpose of A; 'T' or 'C': transpose A
 * @param[in] transe  'N': eigenvectors in columns; 'T' or 'C': in rows
 * @param[in] transw  'N': use WI as is; 'T' or 'C': use -WI (via transpose)
 * @param[in] n       The order of the matrix A. n >= 0.
 * @param[in] A       Double array, dimension (lda, n). The matrix.
 * @param[in] lda     Leading dimension of A. lda >= max(1, n).
 * @param[in] E       Double array, dimension (lde, n). Eigenvector matrix.
 * @param[in] lde     Leading dimension of E. lde >= max(1, n).
 * @param[in] wr      Double array, dimension (n). Real parts of eigenvalues.
 * @param[in] wi      Double array, dimension (n). Imaginary parts of eigenvalues.
 * @param[out] work   Double array, dimension (n*(n+1)).
 * @param[out] result Double array, dimension (2).
 *                    result[0] = | A E - E W | / ( |A| |E| ulp )
 *                    result[1] = max | m-norm(E(j)) - 1 | / ( n ulp )
 */
void dget22(const char* transa, const char* transe, const char* transw,
            const int n, const double* A, const int lda,
            const double* E, const int lde,
            const double* wr, const double* wi,
            double* work, double* result)
{
    const double zero = 0.0;
    const double one = 1.0;

    /* Initialize RESULT (in case n=0) */
    result[0] = zero;
    result[1] = zero;
    if (n <= 0)
        return;

    double unfl = dlamch("S");
    double ulp = dlamch("P");

    int itrnse = 0;
    int ince = 1;
    char norma = 'O';
    char norme = 'O';

    int transa_t = (transa[0] == 'T' || transa[0] == 't' ||
                    transa[0] == 'C' || transa[0] == 'c');
    int transe_t = (transe[0] == 'T' || transe[0] == 't' ||
                    transe[0] == 'C' || transe[0] == 'c');
    int transw_t = (transw[0] == 'T' || transw[0] == 't' ||
                    transw[0] == 'C' || transw[0] == 'c');

    if (transa_t) {
        norma = 'I';
    }
    if (transe_t) {
        norme = 'I';
        itrnse = 1;
        ince = lde;
    }

    /* Check normalization of E */
    double enrmin = one / ulp;
    double enrmax = zero;

    if (itrnse == 0) {
        /* Eigenvectors are column vectors */
        int ipair = 0;
        for (int jvec = 0; jvec < n; jvec++) {
            double temp1 = zero;
            if (ipair == 0 && jvec < n - 1 && wi[jvec] != zero)
                ipair = 1;

            if (ipair == 1) {
                /* Complex eigenvector */
                for (int j = 0; j < n; j++) {
                    double val = fabs(E[j + jvec * lde]) + fabs(E[j + (jvec + 1) * lde]);
                    if (val > temp1) temp1 = val;
                }
                if (temp1 < enrmin) enrmin = temp1;
                if (temp1 > enrmax) enrmax = temp1;
                ipair = 2;
            } else if (ipair == 2) {
                ipair = 0;
            } else {
                /* Real eigenvector */
                for (int j = 0; j < n; j++) {
                    double val = fabs(E[j + jvec * lde]);
                    if (val > temp1) temp1 = val;
                }
                if (temp1 < enrmin) enrmin = temp1;
                if (temp1 > enrmax) enrmax = temp1;
                ipair = 0;
            }
        }
    } else {
        /* Eigenvectors are row vectors */
        for (int jvec = 0; jvec < n; jvec++) {
            work[jvec] = zero;
        }

        for (int j = 0; j < n; j++) {
            int ipair = 0;
            for (int jvec = 0; jvec < n; jvec++) {
                if (ipair == 0 && jvec < n - 1 && wi[jvec] != zero)
                    ipair = 1;

                if (ipair == 1) {
                    double val = fabs(E[j + jvec * lde]) + fabs(E[j + (jvec + 1) * lde]);
                    if (val > work[jvec]) work[jvec] = val;
                    work[jvec + 1] = work[jvec];
                } else if (ipair == 2) {
                    ipair = 0;
                } else {
                    double val = fabs(E[j + jvec * lde]);
                    if (val > work[jvec]) work[jvec] = val;
                    ipair = 0;
                }
            }
        }

        for (int jvec = 0; jvec < n; jvec++) {
            if (work[jvec] < enrmin) enrmin = work[jvec];
            if (work[jvec] > enrmax) enrmax = work[jvec];
        }
    }

    /* Norm of A */
    char norma_str[2] = {norma, '\0'};
    double anorm = dlange(norma_str, n, n, A, lda, work);
    if (anorm < unfl) anorm = unfl;

    /* Norm of E */
    char norme_str[2] = {norme, '\0'};
    double enorm = dlange(norme_str, n, n, E, lde, work);
    if (enorm < ulp) enorm = ulp;

    /* Compute Error = AE - EW
     *
     * Work array layout: first n*n for the result matrix
     */
    dlaset("F", n, n, zero, zero, work, n);

    int ipair = 0;
    int ierow = 0;  /* 0-based: Fortran IEROW=1 -> C ierow=0 */
    int iecol = 0;  /* 0-based: Fortran IECOL=1 -> C iecol=0 */

    for (int jcol = 0; jcol < n; jcol++) {
        if (itrnse == 1) {
            ierow = jcol;
        } else {
            iecol = jcol;
        }

        if (ipair == 0 && wi[jcol] != zero)
            ipair = 1;

        if (ipair == 1) {
            /* Complex eigenvalue pair - form E * W where W is 2x2 block
             *
             * WMAT (column-major 2x2):
             *   ( wr   wi  )
             *   ( -wi  wr  )
             *
             * The TRANSW is handled by DGEMM, not by negating WI.
             */
            double wmat[4];  /* Column-major 2x2 */
            wmat[0] = wr[jcol];     /* W(0,0) = wr */
            wmat[1] = -wi[jcol];    /* W(1,0) = -wi */
            wmat[2] = wi[jcol];     /* W(0,1) = wi */
            wmat[3] = wr[jcol];     /* W(1,1) = wr */

            /* E(:,jcol:jcol+1) * W -> work(:,jcol:jcol+1)
             * In Fortran: DGEMM(TRANSE, TRANSW, N, 2, 2, ONE,
             *                   E(IEROW,IECOL), LDE, WMAT, 2, ZERO,
             *                   WORK(N*(JCOL-1)+1), N)
             */
            CBLAS_TRANSPOSE transe_cblas = transe_t ? CblasTrans : CblasNoTrans;
            CBLAS_TRANSPOSE transw_cblas = transw_t ? CblasTrans : CblasNoTrans;

            cblas_dgemm(CblasColMajor, transe_cblas, transw_cblas,
                        n, 2, 2, one, &E[ierow + iecol * lde], lde,
                        wmat, 2, zero, &work[n * jcol], n);
            ipair = 2;
        } else if (ipair == 2) {
            ipair = 0;
        } else {
            /* Real eigenvalue - scale E column by wr[jcol]
             *
             * Fortran: DAXPY(N, WR(JCOL), E(IEROW,IECOL), INCE,
             *                WORK(N*(JCOL-1)+1), 1)
             *
             * This adds WR(JCOL) * E(:,jcol) to work(:,jcol)
             * Since work was zeroed, this is: work(:,jcol) = WR * E(:,jcol)
             */
            cblas_daxpy(n, wr[jcol], &E[ierow + iecol * lde], ince,
                        &work[n * jcol], 1);
            ipair = 0;
        }
    }

    /* Compute A*E - work (which contains E*W)
     * Result goes into work: work = A*E - E*W
     *
     * Fortran: DGEMM(TRANSA, TRANSE, N, N, N, ONE, A, LDA, E, LDE,
     *                -ONE, WORK, N)
     */
    CBLAS_TRANSPOSE transa_cblas = transa_t ? CblasTrans : CblasNoTrans;
    CBLAS_TRANSPOSE transe_cblas = transe_t ? CblasTrans : CblasNoTrans;

    cblas_dgemm(CblasColMajor, transa_cblas, transe_cblas,
                n, n, n, one, A, lda, E, lde, -one, work, n);

    /* Norm of error */
    double errnrm = dlange("O", n, n, work, n, &work[n * n]) / enorm;

    /* Compute RESULT[0] (avoiding under/overflow) */
    if (anorm > errnrm) {
        result[0] = (errnrm / anorm) / ulp;
    } else {
        if (anorm < one) {
            result[0] = one / ulp;
        } else {
            double ratio = errnrm / anorm;
            if (ratio > one) ratio = one;
            result[0] = ratio / ulp;
        }
    }

    /* Compute RESULT[1]: the normalization error in E */
    double err1 = fabs(enrmax - one);
    double err2 = fabs(enrmin - one);
    result[1] = (err1 > err2 ? err1 : err2) / ((double)n * ulp);
}
