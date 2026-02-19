/**
 * @file dcsdts.c
 * @brief DCSDTS tests DORCSD and DORCSD2BY1 (CS decomposition).
 */

#include <math.h>
#include <cblas.h>
#include "semicolon_lapack_double.h"
#include "verify.h"

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MAX3(a, b, c) MAX(a, MAX(b, c))

static const f64 PIOVER2 = 1.57079632679489661923132169163975144210e0;

void dcsdts(const int m, const int p, const int q,
            const f64* X, f64* XF, const int ldx,
            f64* U1, const int ldu1,
            f64* U2, const int ldu2,
            f64* V1T, const int ldv1t,
            f64* V2T, const int ldv2t,
            f64* theta, int* iwork,
            f64* work, const int lwork,
            f64* rwork, f64* result)
{
    const f64 zero = 0.0;
    const f64 one = 1.0;

    f64 ulp = dlamch("Precision");
    f64 ulpinv = one / ulp;

    /*
     * The first half of the routine checks the 2-by-2 CSD
     */

    dlaset("Full", m, m, zero, one, work, ldx);
    cblas_dsyrk(CblasColMajor, CblasUpper, CblasTrans,
                m, m, -one, X, ldx, one, work, ldx);
    f64 eps2;
    if (m > 0) {
        eps2 = MAX(ulp, dlange("1", m, m, work, ldx, rwork) / (f64)m);
    } else {
        eps2 = ulp;
    }
    int r = MIN(MIN(p, m - p), MIN(q, m - q));

    dlacpy("Full", m, m, X, ldx, XF, ldx);

    int info;
    dorcsd("Y", "Y", "Y", "Y", "N", "D", m, p, q,
           &XF[0], ldx, &XF[q * ldx], ldx,
           &XF[p], ldx, &XF[p + q * ldx], ldx,
           theta, U1, ldu1, U2, ldu2, V1T, ldv1t, V2T, ldv2t,
           work, lwork, iwork, &info);

    /*
     * Compute XF := diag(U1,U2)'*X*diag(V1,V2) - [D11 D12; D21 D22]
     */
    dlacpy("Full", m, m, X, ldx, XF, ldx);

    /* X11 block: U1'*X11*V1 - D11 */
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                p, q, q, one, &XF[0], ldx, V1T, ldv1t, zero, work, ldx);
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                p, q, p, one, U1, ldu1, work, ldx, zero, &XF[0], ldx);

    for (int i = 0; i < MIN(p, q) - r; i++)
        XF[i + i * ldx] -= one;
    for (int i = 0; i < r; i++)
        XF[(MIN(p, q) - r + i) + (MIN(p, q) - r + i) * ldx] -= cos(theta[i]);

    /* X12 block: U1'*X12*V2 - D12 */
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                p, m - q, m - q, one, &XF[q * ldx], ldx, V2T, ldv2t,
                zero, work, ldx);
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                p, m - q, p, one, U1, ldu1, work, ldx,
                zero, &XF[q * ldx], ldx);

    for (int i = 0; i < MIN(p, m - q) - r; i++)
        XF[(p - 1 - i) + (m - 1 - i) * ldx] += one;
    for (int i = 0; i < r; i++)
        XF[(p - (MIN(p, m - q) - r) - 1 - i) + (m - (MIN(p, m - q) - r) - 1 - i) * ldx] +=
            sin(theta[r - 1 - i]);

    /* X21 block: U2'*X21*V1 - D21 */
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                m - p, q, q, one, &XF[p], ldx, V1T, ldv1t,
                zero, work, ldx);
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                m - p, q, m - p, one, U2, ldu2, work, ldx,
                zero, &XF[p], ldx);

    for (int i = 0; i < MIN(m - p, q) - r; i++)
        XF[(m - 1 - i) + (q - 1 - i) * ldx] -= one;
    for (int i = 0; i < r; i++)
        XF[(m - (MIN(m - p, q) - r) - 1 - i) + (q - (MIN(m - p, q) - r) - 1 - i) * ldx] -=
            sin(theta[r - 1 - i]);

    /* X22 block: U2'*X22*V2 - D22 */
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                m - p, m - q, m - q, one, &XF[p + q * ldx], ldx,
                V2T, ldv2t, zero, work, ldx);
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                m - p, m - q, m - p, one, U2, ldu2, work, ldx,
                zero, &XF[p + q * ldx], ldx);

    for (int i = 0; i < MIN(m - p, m - q) - r; i++)
        XF[(p + i) + (q + i) * ldx] -= one;
    for (int i = 0; i < r; i++)
        XF[(p + (MIN(m - p, m - q) - r) + i) + (q + (MIN(m - p, m - q) - r) + i) * ldx] -=
            cos(theta[i]);

    f64 resid;

    resid = dlange("1", p, q, &XF[0], ldx, rwork);
    result[0] = (resid / (f64)MAX3(1, p, q)) / eps2;

    resid = dlange("1", p, m - q, &XF[q * ldx], ldx, rwork);
    result[1] = (resid / (f64)MAX3(1, p, m - q)) / eps2;

    resid = dlange("1", m - p, q, &XF[p], ldx, rwork);
    result[2] = (resid / (f64)MAX3(1, m - p, q)) / eps2;

    resid = dlange("1", m - p, m - q, &XF[p + q * ldx], ldx, rwork);
    result[3] = (resid / (f64)MAX3(1, m - p, m - q)) / eps2;

    /* I - U1'*U1 */
    dlaset("Full", p, p, zero, one, work, ldu1);
    cblas_dsyrk(CblasColMajor, CblasUpper, CblasTrans,
                p, p, -one, U1, ldu1, one, work, ldu1);
    resid = dlansy("1", "Upper", p, work, ldu1, rwork);
    result[4] = (resid / (f64)MAX(1, p)) / ulp;

    /* I - U2'*U2 */
    dlaset("Full", m - p, m - p, zero, one, work, ldu2);
    cblas_dsyrk(CblasColMajor, CblasUpper, CblasTrans,
                m - p, m - p, -one, U2, ldu2, one, work, ldu2);
    resid = dlansy("1", "Upper", m - p, work, ldu2, rwork);
    result[5] = (resid / (f64)MAX(1, m - p)) / ulp;

    /* I - V1T*V1T' */
    dlaset("Full", q, q, zero, one, work, ldv1t);
    cblas_dsyrk(CblasColMajor, CblasUpper, CblasNoTrans,
                q, q, -one, V1T, ldv1t, one, work, ldv1t);
    resid = dlansy("1", "Upper", q, work, ldv1t, rwork);
    result[6] = (resid / (f64)MAX(1, q)) / ulp;

    /* I - V2T*V2T' */
    dlaset("Full", m - q, m - q, zero, one, work, ldv2t);
    cblas_dsyrk(CblasColMajor, CblasUpper, CblasNoTrans,
                m - q, m - q, -one, V2T, ldv2t, one, work, ldv2t);
    resid = dlansy("1", "Upper", m - q, work, ldv2t, rwork);
    result[7] = (resid / (f64)MAX(1, m - q)) / ulp;

    /* Check sorting */
    result[8] = zero;
    for (int i = 0; i < r; i++) {
        if (theta[i] < zero || theta[i] > PIOVER2)
            result[8] = ulpinv;
        if (i > 0) {
            if (theta[i] < theta[i - 1])
                result[8] = ulpinv;
        }
    }

    /*
     * The second half of the routine checks the 2-by-1 CSD
     */
    dlaset("Full", q, q, zero, one, work, ldx);
    cblas_dsyrk(CblasColMajor, CblasUpper, CblasTrans,
                q, m, -one, X, ldx, one, work, ldx);
    if (m > 0) {
        eps2 = MAX(ulp, dlange("1", q, q, work, ldx, rwork) / (f64)m);
    } else {
        eps2 = ulp;
    }
    r = MIN(MIN(p, m - p), MIN(q, m - q));

    dlacpy("Full", m, q, X, ldx, XF, ldx);

    dorcsd2by1("Y", "Y", "Y", m, p, q,
               &XF[0], ldx, &XF[p], ldx,
               theta, U1, ldu1, U2, ldu2, V1T, ldv1t,
               work, lwork, iwork, &info);

    /* Compute [X11;X21] := diag(U1,U2)'*[X11;X21]*V1 - [D11;D21] */

    /* X11 block */
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                p, q, q, one, X, ldx, V1T, ldv1t, zero, work, ldx);
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                p, q, p, one, U1, ldu1, work, ldx, zero, XF, ldx);

    for (int i = 0; i < MIN(p, q) - r; i++)
        XF[i + i * ldx] -= one;
    for (int i = 0; i < r; i++)
        XF[(MIN(p, q) - r + i) + (MIN(p, q) - r + i) * ldx] -= cos(theta[i]);

    /* X21 block */
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                m - p, q, q, one, &X[p], ldx, V1T, ldv1t,
                zero, work, ldx);
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                m - p, q, m - p, one, U2, ldu2, work, ldx,
                zero, &XF[p], ldx);

    for (int i = 0; i < MIN(m - p, q) - r; i++)
        XF[(m - 1 - i) + (q - 1 - i) * ldx] -= one;
    for (int i = 0; i < r; i++)
        XF[(m - (MIN(m - p, q) - r) - 1 - i) + (q - (MIN(m - p, q) - r) - 1 - i) * ldx] -=
            sin(theta[r - 1 - i]);

    resid = dlange("1", p, q, XF, ldx, rwork);
    result[9] = (resid / (f64)MAX3(1, p, q)) / eps2;

    resid = dlange("1", m - p, q, &XF[p], ldx, rwork);
    result[10] = (resid / (f64)MAX3(1, m - p, q)) / eps2;

    /* I - U1'*U1 */
    dlaset("Full", p, p, zero, one, work, ldu1);
    cblas_dsyrk(CblasColMajor, CblasUpper, CblasTrans,
                p, p, -one, U1, ldu1, one, work, ldu1);
    resid = dlansy("1", "Upper", p, work, ldu1, rwork);
    result[11] = (resid / (f64)MAX(1, p)) / ulp;

    /* I - U2'*U2 */
    dlaset("Full", m - p, m - p, zero, one, work, ldu2);
    cblas_dsyrk(CblasColMajor, CblasUpper, CblasTrans,
                m - p, m - p, -one, U2, ldu2, one, work, ldu2);
    resid = dlansy("1", "Upper", m - p, work, ldu2, rwork);
    result[12] = (resid / (f64)MAX(1, m - p)) / ulp;

    /* I - V1T*V1T' */
    dlaset("Full", q, q, zero, one, work, ldv1t);
    cblas_dsyrk(CblasColMajor, CblasUpper, CblasNoTrans,
                q, q, -one, V1T, ldv1t, one, work, ldv1t);
    resid = dlansy("1", "Upper", q, work, ldv1t, rwork);
    result[13] = (resid / (f64)MAX(1, q)) / ulp;

    /* Check sorting */
    result[14] = zero;
    for (int i = 0; i < r; i++) {
        if (theta[i] < zero || theta[i] > PIOVER2)
            result[14] = ulpinv;
        if (i > 0) {
            if (theta[i] < theta[i - 1])
                result[14] = ulpinv;
        }
    }
}
