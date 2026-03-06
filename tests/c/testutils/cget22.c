/**
 * @file cget22.c
 * @brief CGET22 does an eigenvector check.
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"

/**
 * CGET22 does an eigenvector check.
 *
 * The basic test is:
 *
 *    RESULT(1) = | A E  -  E W | / ( |A| |E| ulp )
 *
 * using the 1-norm.  It also tests the normalization of E:
 *
 *    RESULT(2) = max | m-norm(E(j)) - 1 | / ( n ulp )
 *                 j
 *
 * where E(j) is the j-th eigenvector, and m-norm is the max-norm of a
 * vector.  The max-norm of a complex n-vector x in this case is the
 * maximum of |re(x(i)| + |im(x(i)| over i = 1, ..., n.
 *
 * @param[in]     transa  Specifies whether or not A is transposed.
 *                        'N': No transpose; 'T': Transpose; 'C': Conjugate transpose
 * @param[in]     transe  Specifies whether or not E is transposed.
 *                        'N': No transpose, eigenvectors in columns of E
 *                        'T': Transpose, eigenvectors in rows of E
 *                        'C': Conjugate transpose, eigenvectors in rows of E
 * @param[in]     transw  Specifies whether or not W is conjugated.
 *                        'N': No conjugate; 'C': Conjugate
 * @param[in]     n       The order of the matrix A. n >= 0.
 * @param[in]     A       Complex array, dimension (lda, n).
 * @param[in]     lda     Leading dimension of A. lda >= max(1, n).
 * @param[in]     E       Complex array, dimension (lde, n). Eigenvector matrix.
 * @param[in]     lde     Leading dimension of E. lde >= max(1, n).
 * @param[in]     W       Complex array, dimension (n). The eigenvalues.
 * @param[out]    work    Complex array, dimension (n*n).
 * @param[out]    rwork   Double array, dimension (n).
 * @param[out]    result  Double array, dimension (2).
 *                        result[0] = | A E - E W | / ( |A| |E| ulp )
 *                        result[1] = max | m-norm(E(j)) - 1 | / ( n ulp )
 */
void cget22(const char* transa, const char* transe, const char* transw,
            const INT n, const c64* A, const INT lda,
            const c64* E, const INT lde, const c64* W,
            c64* work, f32* rwork, f32* result)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;
    const c64 CONE = CMPLXF(1.0f, 0.0f);
    const c64 CNEGONE = CMPLXF(-1.0f, 0.0f);

    INT itrnse, itrnsw, jcol, jrow, jvec;
    f32 anorm, enorm, enrmax, enrmin, errnrm, temp1, ulp, unfl;
    c64 wtemp;
    char norma, norme;

    result[0] = ZERO;
    result[1] = ZERO;
    if (n <= 0)
        return;

    unfl = slamch("S");
    ulp = slamch("P");

    itrnse = 0;
    itrnsw = 0;
    norma = 'O';
    norme = 'O';

    if (transa[0] == 'T' || transa[0] == 't' ||
        transa[0] == 'C' || transa[0] == 'c') {
        norma = 'I';
    }

    if (transe[0] == 'T' || transe[0] == 't') {
        itrnse = 1;
        norme = 'I';
    } else if (transe[0] == 'C' || transe[0] == 'c') {
        itrnse = 2;
        norme = 'I';
    }

    if (transw[0] == 'C' || transw[0] == 'c') {
        itrnsw = 1;
    }

    /* Normalization of E */

    enrmin = ONE / ulp;
    enrmax = ZERO;
    if (itrnse == 0) {
        for (jvec = 0; jvec < n; jvec++) {
            temp1 = ZERO;
            for (INT j = 0; j < n; j++) {
                temp1 = fmaxf(temp1, cabs1f(E[j + jvec * lde]));
            }
            enrmin = fminf(enrmin, temp1);
            enrmax = fmaxf(enrmax, temp1);
        }
    } else {
        for (jvec = 0; jvec < n; jvec++) {
            rwork[jvec] = ZERO;
        }

        for (INT j = 0; j < n; j++) {
            for (jvec = 0; jvec < n; jvec++) {
                rwork[jvec] = fmaxf(rwork[jvec], cabs1f(E[jvec + j * lde]));
            }
        }

        for (jvec = 0; jvec < n; jvec++) {
            enrmin = fminf(enrmin, rwork[jvec]);
            enrmax = fmaxf(enrmax, rwork[jvec]);
        }
    }

    /* Norm of A */

    char norma_str[2] = {norma, '\0'};
    anorm = fmaxf(clange(norma_str, n, n, A, lda, rwork), unfl);

    /* Norm of E */

    char norme_str[2] = {norme, '\0'};
    enorm = fmaxf(clange(norme_str, n, n, E, lde, rwork), ulp);

    /* Norm of error:
     *
     * Error = AE - EW
     */

    claset("F", n, n, CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), work, n);

    INT joff = 0;
    for (jcol = 0; jcol < n; jcol++) {
        if (itrnsw == 0) {
            wtemp = W[jcol];
        } else {
            wtemp = conjf(W[jcol]);
        }

        if (itrnse == 0) {
            for (jrow = 0; jrow < n; jrow++) {
                work[joff + jrow] = E[jrow + jcol * lde] * wtemp;
            }
        } else if (itrnse == 1) {
            for (jrow = 0; jrow < n; jrow++) {
                work[joff + jrow] = E[jcol + jrow * lde] * wtemp;
            }
        } else {
            for (jrow = 0; jrow < n; jrow++) {
                work[joff + jrow] = conjf(E[jcol + jrow * lde]) * wtemp;
            }
        }
        joff += n;
    }

    /* Convert transa/transe to CBLAS enum */
    CBLAS_TRANSPOSE transa_cblas;
    if (transa[0] == 'C' || transa[0] == 'c')
        transa_cblas = CblasConjTrans;
    else if (transa[0] == 'T' || transa[0] == 't')
        transa_cblas = CblasTrans;
    else
        transa_cblas = CblasNoTrans;

    CBLAS_TRANSPOSE transe_cblas;
    if (transe[0] == 'C' || transe[0] == 'c')
        transe_cblas = CblasConjTrans;
    else if (transe[0] == 'T' || transe[0] == 't')
        transe_cblas = CblasTrans;
    else
        transe_cblas = CblasNoTrans;

    cblas_cgemm(CblasColMajor, transa_cblas, transe_cblas,
                n, n, n, &CONE, A, lda, E, lde, &CNEGONE, work, n);

    errnrm = clange("O", n, n, work, n, rwork) / enorm;

    /* Compute RESULT(1) (avoiding under/overflow) */

    if (anorm > errnrm) {
        result[0] = (errnrm / anorm) / ulp;
    } else {
        if (anorm < ONE) {
            result[0] = ONE / ulp;
        } else {
            result[0] = fminf(errnrm / anorm, ONE) / ulp;
        }
    }

    /* Compute RESULT(2) : the normalization error in E. */

    result[1] = fmaxf(fabsf(enrmax - ONE), fabsf(enrmin - ONE)) /
                ((f32)n * ulp);
}
