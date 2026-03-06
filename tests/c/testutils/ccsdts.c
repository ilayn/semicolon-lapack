/**
 * @file ccsdts.c
 * @brief CCSDTS tests CUNCSD and CUNCSD2BY1 (CS decomposition).
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MAX3(a, b, c) MAX(a, MAX(b, c))

static const f32 PIOVER2 = 1.57079632679489661923132169163975144210e0f;

void ccsdts(const INT m, const INT p, const INT q,
            const c64* X, c64* XF, const INT ldx,
            c64* U1, const INT ldu1,
            c64* U2, const INT ldu2,
            c64* V1T, const INT ldv1t,
            c64* V2T, const INT ldv2t,
            f32* theta, INT* iwork,
            c64* work, const INT lwork,
            f32* rwork, f32* result)
{
    const f32 REALONE = 1.0f;
    const f32 REALZERO = 0.0f;
    const c64 CZERO = CMPLXF(0.0f, 0.0f);
    const c64 CONE = CMPLXF(1.0f, 0.0f);

    f32 ulp = slamch("Precision");
    f32 ulpinv = REALONE / ulp;

    /*
     * The first half of the routine checks the 2-by-2 CSD
     */

    claset("Full", m, m, CZERO, CONE, work, ldx);
    cblas_cherk(CblasColMajor, CblasUpper, CblasConjTrans,
                m, m, -REALONE, X, ldx, REALONE, work, ldx);
    f32 eps2;
    if (m > 0) {
        eps2 = MAX(ulp, clange("1", m, m, work, ldx, rwork) / (f32)m);
    } else {
        eps2 = ulp;
    }
    INT r = MIN(MIN(p, m - p), MIN(q, m - q));

    clacpy("Full", m, m, X, ldx, XF, ldx);

    INT info;
    cuncsd("Y", "Y", "Y", "Y", "N", "D", m, p, q,
           &XF[0], ldx, &XF[q * ldx], ldx,
           &XF[p], ldx, &XF[p + q * ldx], ldx,
           theta, U1, ldu1, U2, ldu2, V1T, ldv1t, V2T, ldv2t,
           work, lwork, rwork, 17 * (r + 2), iwork, &info);

    /*
     * Compute XF := diag(U1,U2)'*X*diag(V1,V2) - [D11 D12; D21 D22]
     */
    clacpy("Full", m, m, X, ldx, XF, ldx);

    /* X11 block: U1'*X11*V1 - D11 */
    cblas_cgemm(CblasColMajor, CblasNoTrans, CblasConjTrans,
                p, q, q, &CONE, &XF[0], ldx, V1T, ldv1t, &CZERO, work, ldx);
    cblas_cgemm(CblasColMajor, CblasConjTrans, CblasNoTrans,
                p, q, p, &CONE, U1, ldu1, work, ldx, &CZERO, &XF[0], ldx);

    for (INT i = 0; i < MIN(p, q) - r; i++)
        XF[i + i * ldx] -= CONE;
    for (INT i = 0; i < r; i++)
        XF[(MIN(p, q) - r + i) + (MIN(p, q) - r + i) * ldx] -=
            CMPLXF(cosf(theta[i]), 0.0f);

    /* X12 block: U1'*X12*V2 - D12 */
    cblas_cgemm(CblasColMajor, CblasNoTrans, CblasConjTrans,
                p, m - q, m - q, &CONE, &XF[q * ldx], ldx, V2T, ldv2t,
                &CZERO, work, ldx);
    cblas_cgemm(CblasColMajor, CblasConjTrans, CblasNoTrans,
                p, m - q, p, &CONE, U1, ldu1, work, ldx,
                &CZERO, &XF[q * ldx], ldx);

    for (INT i = 0; i < MIN(p, m - q) - r; i++)
        XF[(p - 1 - i) + (m - 1 - i) * ldx] += CONE;
    for (INT i = 0; i < r; i++)
        XF[(p - (MIN(p, m - q) - r) - 1 - i) + (m - (MIN(p, m - q) - r) - 1 - i) * ldx] +=
            CMPLXF(sinf(theta[r - 1 - i]), 0.0f);

    /* X21 block: U2'*X21*V1 - D21 */
    cblas_cgemm(CblasColMajor, CblasNoTrans, CblasConjTrans,
                m - p, q, q, &CONE, &XF[p], ldx, V1T, ldv1t,
                &CZERO, work, ldx);
    cblas_cgemm(CblasColMajor, CblasConjTrans, CblasNoTrans,
                m - p, q, m - p, &CONE, U2, ldu2, work, ldx,
                &CZERO, &XF[p], ldx);

    for (INT i = 0; i < MIN(m - p, q) - r; i++)
        XF[(m - 1 - i) + (q - 1 - i) * ldx] -= CONE;
    for (INT i = 0; i < r; i++)
        XF[(m - (MIN(m - p, q) - r) - 1 - i) + (q - (MIN(m - p, q) - r) - 1 - i) * ldx] -=
            CMPLXF(sinf(theta[r - 1 - i]), 0.0f);

    /* X22 block: U2'*X22*V2 - D22 */
    cblas_cgemm(CblasColMajor, CblasNoTrans, CblasConjTrans,
                m - p, m - q, m - q, &CONE, &XF[p + q * ldx], ldx,
                V2T, ldv2t, &CZERO, work, ldx);
    cblas_cgemm(CblasColMajor, CblasConjTrans, CblasNoTrans,
                m - p, m - q, m - p, &CONE, U2, ldu2, work, ldx,
                &CZERO, &XF[p + q * ldx], ldx);

    for (INT i = 0; i < MIN(m - p, m - q) - r; i++)
        XF[(p + i) + (q + i) * ldx] -= CONE;
    for (INT i = 0; i < r; i++)
        XF[(p + (MIN(m - p, m - q) - r) + i) + (q + (MIN(m - p, m - q) - r) + i) * ldx] -=
            CMPLXF(cosf(theta[i]), 0.0f);

    f32 resid;

    resid = clange("1", p, q, &XF[0], ldx, rwork);
    result[0] = (resid / (f32)MAX3(1, p, q)) / eps2;

    resid = clange("1", p, m - q, &XF[q * ldx], ldx, rwork);
    result[1] = (resid / (f32)MAX3(1, p, m - q)) / eps2;

    resid = clange("1", m - p, q, &XF[p], ldx, rwork);
    result[2] = (resid / (f32)MAX3(1, m - p, q)) / eps2;

    resid = clange("1", m - p, m - q, &XF[p + q * ldx], ldx, rwork);
    result[3] = (resid / (f32)MAX3(1, m - p, m - q)) / eps2;

    /* I - U1'*U1 */
    claset("Full", p, p, CZERO, CONE, work, ldu1);
    cblas_cherk(CblasColMajor, CblasUpper, CblasConjTrans,
                p, p, -REALONE, U1, ldu1, REALONE, work, ldu1);
    resid = clanhe("1", "Upper", p, work, ldu1, rwork);
    result[4] = (resid / (f32)MAX(1, p)) / ulp;

    /* I - U2'*U2 */
    claset("Full", m - p, m - p, CZERO, CONE, work, ldu2);
    cblas_cherk(CblasColMajor, CblasUpper, CblasConjTrans,
                m - p, m - p, -REALONE, U2, ldu2, REALONE, work, ldu2);
    resid = clanhe("1", "Upper", m - p, work, ldu2, rwork);
    result[5] = (resid / (f32)MAX(1, m - p)) / ulp;

    /* I - V1T*V1T' */
    claset("Full", q, q, CZERO, CONE, work, ldv1t);
    cblas_cherk(CblasColMajor, CblasUpper, CblasNoTrans,
                q, q, -REALONE, V1T, ldv1t, REALONE, work, ldv1t);
    resid = clanhe("1", "Upper", q, work, ldv1t, rwork);
    result[6] = (resid / (f32)MAX(1, q)) / ulp;

    /* I - V2T*V2T' */
    claset("Full", m - q, m - q, CZERO, CONE, work, ldv2t);
    cblas_cherk(CblasColMajor, CblasUpper, CblasNoTrans,
                m - q, m - q, -REALONE, V2T, ldv2t, REALONE, work, ldv2t);
    resid = clanhe("1", "Upper", m - q, work, ldv2t, rwork);
    result[7] = (resid / (f32)MAX(1, m - q)) / ulp;

    /* Check sorting */
    result[8] = REALZERO;
    for (INT i = 0; i < r; i++) {
        if (theta[i] < REALZERO || theta[i] > PIOVER2)
            result[8] = ulpinv;
        if (i > 0) {
            if (theta[i] < theta[i - 1])
                result[8] = ulpinv;
        }
    }

    /*
     * The second half of the routine checks the 2-by-1 CSD
     */
    claset("Full", q, q, CZERO, CONE, work, ldx);
    cblas_cherk(CblasColMajor, CblasUpper, CblasConjTrans,
                q, m, -REALONE, X, ldx, REALONE, work, ldx);
    if (m > 0) {
        eps2 = MAX(ulp, clange("1", q, q, work, ldx, rwork) / (f32)m);
    } else {
        eps2 = ulp;
    }
    r = MIN(MIN(p, m - p), MIN(q, m - q));

    clacpy("Full", m, m, X, ldx, XF, ldx);

    cuncsd2by1("Y", "Y", "Y", m, p, q,
               &XF[0], ldx, &XF[p], ldx,
               theta, U1, ldu1, U2, ldu2, V1T, ldv1t,
               work, lwork, rwork, 17 * (r + 2), iwork, &info);

    /* Compute [X11;X21] := diag(U1,U2)'*[X11;X21]*V1 - [D11;D21] */

    /* X11 block */
    cblas_cgemm(CblasColMajor, CblasNoTrans, CblasConjTrans,
                p, q, q, &CONE, X, ldx, V1T, ldv1t, &CZERO, work, ldx);
    cblas_cgemm(CblasColMajor, CblasConjTrans, CblasNoTrans,
                p, q, p, &CONE, U1, ldu1, work, ldx, &CZERO, XF, ldx);

    for (INT i = 0; i < MIN(p, q) - r; i++)
        XF[i + i * ldx] -= CONE;
    for (INT i = 0; i < r; i++)
        XF[(MIN(p, q) - r + i) + (MIN(p, q) - r + i) * ldx] -=
            CMPLXF(cosf(theta[i]), 0.0f);

    /* X21 block */
    cblas_cgemm(CblasColMajor, CblasNoTrans, CblasConjTrans,
                m - p, q, q, &CONE, &X[p], ldx, V1T, ldv1t,
                &CZERO, work, ldx);
    cblas_cgemm(CblasColMajor, CblasConjTrans, CblasNoTrans,
                m - p, q, m - p, &CONE, U2, ldu2, work, ldx,
                &CZERO, &XF[p], ldx);

    for (INT i = 0; i < MIN(m - p, q) - r; i++)
        XF[(m - 1 - i) + (q - 1 - i) * ldx] -= CONE;
    for (INT i = 0; i < r; i++)
        XF[(m - (MIN(m - p, q) - r) - 1 - i) + (q - (MIN(m - p, q) - r) - 1 - i) * ldx] -=
            CMPLXF(sinf(theta[r - 1 - i]), 0.0f);

    resid = clange("1", p, q, XF, ldx, rwork);
    result[9] = (resid / (f32)MAX3(1, p, q)) / eps2;

    resid = clange("1", m - p, q, &XF[p], ldx, rwork);
    result[10] = (resid / (f32)MAX3(1, m - p, q)) / eps2;

    /* I - U1'*U1 */
    claset("Full", p, p, CZERO, CONE, work, ldu1);
    cblas_cherk(CblasColMajor, CblasUpper, CblasConjTrans,
                p, p, -REALONE, U1, ldu1, REALONE, work, ldu1);
    resid = clanhe("1", "Upper", p, work, ldu1, rwork);
    result[11] = (resid / (f32)MAX(1, p)) / ulp;

    /* I - U2'*U2 */
    claset("Full", m - p, m - p, CZERO, CONE, work, ldu2);
    cblas_cherk(CblasColMajor, CblasUpper, CblasConjTrans,
                m - p, m - p, -REALONE, U2, ldu2, REALONE, work, ldu2);
    resid = clanhe("1", "Upper", m - p, work, ldu2, rwork);
    result[12] = (resid / (f32)MAX(1, m - p)) / ulp;

    /* I - V1T*V1T' */
    claset("Full", q, q, CZERO, CONE, work, ldv1t);
    cblas_cherk(CblasColMajor, CblasUpper, CblasNoTrans,
                q, q, -REALONE, V1T, ldv1t, REALONE, work, ldv1t);
    resid = clanhe("1", "Upper", q, work, ldv1t, rwork);
    result[13] = (resid / (f32)MAX(1, q)) / ulp;

    /* Check sorting */
    result[14] = REALZERO;
    for (INT i = 0; i < r; i++) {
        if (theta[i] < REALZERO || theta[i] > PIOVER2)
            result[14] = ulpinv;
        if (i > 0) {
            if (theta[i] < theta[i - 1])
                result[14] = ulpinv;
        }
    }
}
