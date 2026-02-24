/**
 * @file zget36.c
 * @brief ZGET36 tests ZTREXC, a routine for reordering diagonal entries of a
 *        matrix in complex Schur form.
 */

#include "semicolon_cblas.h"
#include "verify.h"
#include <string.h>

#define LDT 10
#define LWORK (2 * LDT * LDT)
#define NCASES 8

typedef struct {
    INT n;
    INT ifst;  /* 0-based */
    INT ilst;  /* 0-based */
    c128 mat[LDT * LDT]; /* column-major */
} ztrexc_case_t;

static void rowmajor_to_colmajor(const c128* rows, c128* cm, INT n, INT ldcm)
{
    memset(cm, 0, (size_t)ldcm * n * sizeof(c128));
    for (INT i = 0; i < n; i++)
        for (INT j = 0; j < n; j++)
            cm[i + j * ldcm] = rows[i * n + j];
}

static void build_test_cases(ztrexc_case_t* tc)
{
    /* Case 0: N=1, IFST=0, ILST=0 (Fortran 1,1) */
    static const c128 m0[1] = {
        CMPLX(0.0, 0.0)
    };
    tc[0].n = 1; tc[0].ifst = 0; tc[0].ilst = 0;
    rowmajor_to_colmajor(m0, tc[0].mat, 1, LDT);

    /* Case 1: N=3, IFST=0, ILST=2 (Fortran 1,3) */
    static const c128 m1[9] = {
        CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
        CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
        CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0)
    };
    tc[1].n = 3; tc[1].ifst = 0; tc[1].ilst = 2;
    rowmajor_to_colmajor(m1, tc[1].mat, 3, LDT);

    /* Case 2: N=4, IFST=3, ILST=0 (Fortran 4,1) */
    static const c128 m2[16] = {
        CMPLX(1.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
        CMPLX(0.0, 0.0), CMPLX(1.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
        CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(1.0, 0.0), CMPLX(0.0, 0.0),
        CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(1.0, 0.0)
    };
    tc[2].n = 4; tc[2].ifst = 3; tc[2].ilst = 0;
    rowmajor_to_colmajor(m2, tc[2].mat, 4, LDT);

    /* Case 3: N=4, IFST=3, ILST=0 (Fortran 4,1) */
    static const c128 m3[16] = {
        CMPLX(1.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
        CMPLX(0.0, 0.0), CMPLX(2.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0),
        CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(3.0, 0.0), CMPLX(0.0, 0.0),
        CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(0.0, 0.0), CMPLX(4.0, 0.0)
    };
    tc[3].n = 4; tc[3].ifst = 3; tc[3].ilst = 0;
    rowmajor_to_colmajor(m3, tc[3].mat, 4, LDT);

    /* Case 4: N=4, IFST=0, ILST=3 (Fortran 1,4) */
    static const c128 m4[16] = {
        CMPLX(12.0, 0.0), CMPLX(0.0, 20.0), CMPLX(-2.0, 0.0), CMPLX(10.0, 0.0),
        CMPLX(0.0, 0.0),  CMPLX(20.0, 0.0), CMPLX(2.0, -1.0),  CMPLX(0.0, 0.9),
        CMPLX(0.0, 0.0),  CMPLX(0.0, 0.0),  CMPLX(3.0, 0.0),   CMPLX(0.8, 0.0),
        CMPLX(0.0, 0.0),  CMPLX(0.0, 0.0),  CMPLX(0.0, 0.0),   CMPLX(8.0, 0.0)
    };
    tc[4].n = 4; tc[4].ifst = 0; tc[4].ilst = 3;
    rowmajor_to_colmajor(m4, tc[4].mat, 4, LDT);

    /* Case 5: N=5, IFST=4, ILST=0 (Fortran 5,1) */
    static const c128 m5[25] = {
        CMPLX(1.0, 1.0),  CMPLX(2.0, -1.0), CMPLX(2.0, -3.0), CMPLX(12.0, 3.0), CMPLX(2.0, 39.0),
        CMPLX(0.0, 0.0),  CMPLX(2.0, 3.0),  CMPLX(2.0, 3.0),  CMPLX(2.0, 13.0), CMPLX(2.0, 31.0),
        CMPLX(0.0, 0.0),  CMPLX(0.0, 0.0),  CMPLX(-2.0, 3.0), CMPLX(2.0, 3.0),  CMPLX(12.0, 3.0),
        CMPLX(0.0, 0.0),  CMPLX(0.0, 0.0),  CMPLX(0.0, 0.0),  CMPLX(2.0, -3.0), CMPLX(-2.0, 3.0),
        CMPLX(0.0, 0.0),  CMPLX(0.0, 0.0),  CMPLX(0.0, 0.0),  CMPLX(0.0, 0.0),  CMPLX(2.0, 3.0)
    };
    tc[5].n = 5; tc[5].ifst = 4; tc[5].ilst = 0;
    rowmajor_to_colmajor(m5, tc[5].mat, 5, LDT);

    /* Case 6: N=4, IFST=3, ILST=0 (Fortran 4,1) */
    static const c128 m6[16] = {
        CMPLX(0.0621, 0.7054), CMPLX(0.1062, 0.0503), CMPLX(0.6553, 0.5876), CMPLX(0.2560, 0.8642),
        CMPLX(0.0, 0.0),       CMPLX(0.2640, 0.5782), CMPLX(0.9700, 0.7256), CMPLX(0.5598, 0.1943),
        CMPLX(0.0, 0.0),       CMPLX(0.0, 0.0),       CMPLX(0.0380, 0.2849), CMPLX(0.9166, 0.0580),
        CMPLX(0.0, 0.0),       CMPLX(0.0, 0.0),       CMPLX(0.0, 0.0),       CMPLX(0.1402, 0.6908)
    };
    tc[6].n = 4; tc[6].ifst = 3; tc[6].ilst = 0;
    rowmajor_to_colmajor(m6, tc[6].mat, 4, LDT);

    /* Case 7: N=6, IFST=4, ILST=2 (Fortran 5,3) */
    static const c128 m7[36] = {
        CMPLX(10.0, 1.0),  CMPLX(10.0, 0.0),  CMPLX(30.0, 0.0),  CMPLX(0.0, 1.0),    CMPLX(10.0, 1.0),  CMPLX(10.0, 0.0),
        CMPLX(0.0, 0.0),   CMPLX(20.0, 1.0),  CMPLX(30.0, 0.0),  CMPLX(20.0, 1.0),   CMPLX(0.0, -1.0),  CMPLX(0.0, -10.0),
        CMPLX(0.0, 0.0),   CMPLX(0.0, 0.0),   CMPLX(30.0, 1.0),  CMPLX(0.0, 0.0),    CMPLX(2.0, 0.0),   CMPLX(0.0, 20.0),
        CMPLX(0.0, 0.0),   CMPLX(0.0, 0.0),   CMPLX(0.0, 0.0),   CMPLX(40.0, 1.0),   CMPLX(0.0, -10.0), CMPLX(-30.0, 0.0),
        CMPLX(0.0, 0.0),   CMPLX(0.0, 0.0),   CMPLX(0.0, 0.0),   CMPLX(0.0, 0.0),    CMPLX(50.0, 1.0),  CMPLX(0.0, 0.0),
        CMPLX(0.0, 0.0),   CMPLX(0.0, 0.0),   CMPLX(0.0, 0.0),   CMPLX(0.0, 0.0),    CMPLX(0.0, 0.0),   CMPLX(60.0, 1.0)
    };
    tc[7].n = 6; tc[7].ifst = 4; tc[7].ilst = 2;
    rowmajor_to_colmajor(m7, tc[7].mat, 6, LDT);
}

void zget36(f64* rmax, INT* lmax, INT* ninfo, INT* knt)
{
    const f64 ZERO = 0.0;
    const f64 ONE  = 1.0;
    const c128 CZERO = CMPLX(0.0, 0.0);
    const c128 CONE  = CMPLX(1.0, 0.0);

    f64 eps = dlamch("P");
    *rmax = ZERO;
    *lmax = 0;
    *knt = 0;
    *ninfo = 0;

    ztrexc_case_t cases[NCASES];
    build_test_cases(cases);

    c128 t1[LDT * LDT], t2[LDT * LDT], tmp[LDT * LDT];
    c128 q[LDT * LDT], work[LWORK], diag[LDT];
    f64 result[2], rwork[LDT];
    c128 ctemp;

    for (INT ic = 0; ic < NCASES; ic++) {
        INT n = cases[ic].n;
        INT ifst = cases[ic].ifst;
        INT ilst = cases[ic].ilst;

        (*knt)++;
        zlacpy("F", n, n, cases[ic].mat, LDT, tmp, LDT);
        zlacpy("F", n, n, tmp, LDT, t1, LDT);
        zlacpy("F", n, n, tmp, LDT, t2, LDT);
        f64 res = ZERO;
        INT info1, info2;

        zlaset("Full", n, n, CZERO, CONE, q, LDT);
        ztrexc("N", n, t1, LDT, q, LDT, ifst, ilst, &info1);
        for (INT i = 0; i < n; i++) {
            for (INT j = 0; j < n; j++) {
                if (i == j && q[i + j * LDT] != CONE)
                    res = res + ONE / eps;
                if (i != j && q[i + j * LDT] != CZERO)
                    res = res + ONE / eps;
            }
        }

        zlaset("Full", n, n, CZERO, CONE, q, LDT);
        ztrexc("V", n, t2, LDT, q, LDT, ifst, ilst, &info2);

        for (INT i = 0; i < n; i++)
            for (INT j = 0; j < n; j++)
                if (t1[i + j * LDT] != t2[i + j * LDT])
                    res = res + ONE / eps;
        if (info1 != 0 || info2 != 0)
            *ninfo = *ninfo + 1;
        if (info1 != info2)
            res = res + ONE / eps;

        cblas_zcopy(n, tmp, LDT + 1, diag, 1);
        if (ifst < ilst) {
            for (INT i = ifst + 1; i <= ilst; i++) {
                ctemp = diag[i];
                diag[i] = diag[i - 1];
                diag[i - 1] = ctemp;
            }
        } else if (ifst > ilst) {
            for (INT i = ifst - 1; i >= ilst; i--) {
                ctemp = diag[i + 1];
                diag[i + 1] = diag[i];
                diag[i] = ctemp;
            }
        }
        for (INT i = 0; i < n; i++)
            if (t2[i + i * LDT] != diag[i])
                res = res + ONE / eps;

        zhst01(n, 0, n - 1, tmp, LDT, t2, LDT, q, LDT, work, LWORK,
               rwork, result);
        res = res + result[0] + result[1];

        for (INT j = 0; j < n - 1; j++)
            for (INT i = j + 1; i < n; i++)
                if (t2[i + j * LDT] != CZERO)
                    res = res + ONE / eps;

        if (res > *rmax) {
            *rmax = res;
            *lmax = *knt;
        }
    }
}
