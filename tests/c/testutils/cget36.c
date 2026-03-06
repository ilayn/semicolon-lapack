/**
 * @file cget36.c
 * @brief CGET36 tests CTREXC, a routine for reordering diagonal entries of a
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
    c64 mat[LDT * LDT]; /* column-major */
} ztrexc_case_t;

static void rowmajor_to_colmajor(const c64* rows, c64* cm, INT n, INT ldcm)
{
    memset(cm, 0, (size_t)ldcm * n * sizeof(c64));
    for (INT i = 0; i < n; i++)
        for (INT j = 0; j < n; j++)
            cm[i + j * ldcm] = rows[i * n + j];
}

static void build_test_cases(ztrexc_case_t* tc)
{
    /* Case 0: N=1, IFST=0, ILST=0 (Fortran 1,1) */
    static const c64 m0[1] = {
        CMPLXF(0.0f, 0.0f)
    };
    tc[0].n = 1; tc[0].ifst = 0; tc[0].ilst = 0;
    rowmajor_to_colmajor(m0, tc[0].mat, 1, LDT);

    /* Case 1: N=3, IFST=0, ILST=2 (Fortran 1,3) */
    static const c64 m1[9] = {
        CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
        CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
        CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f)
    };
    tc[1].n = 3; tc[1].ifst = 0; tc[1].ilst = 2;
    rowmajor_to_colmajor(m1, tc[1].mat, 3, LDT);

    /* Case 2: N=4, IFST=3, ILST=0 (Fortran 4,1) */
    static const c64 m2[16] = {
        CMPLXF(1.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
        CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
        CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 0.0f), CMPLXF(0.0f, 0.0f),
        CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(1.0f, 0.0f)
    };
    tc[2].n = 4; tc[2].ifst = 3; tc[2].ilst = 0;
    rowmajor_to_colmajor(m2, tc[2].mat, 4, LDT);

    /* Case 3: N=4, IFST=3, ILST=0 (Fortran 4,1) */
    static const c64 m3[16] = {
        CMPLXF(1.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
        CMPLXF(0.0f, 0.0f), CMPLXF(2.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f),
        CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(3.0f, 0.0f), CMPLXF(0.0f, 0.0f),
        CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(0.0f, 0.0f), CMPLXF(4.0f, 0.0f)
    };
    tc[3].n = 4; tc[3].ifst = 3; tc[3].ilst = 0;
    rowmajor_to_colmajor(m3, tc[3].mat, 4, LDT);

    /* Case 4: N=4, IFST=0, ILST=3 (Fortran 1,4) */
    static const c64 m4[16] = {
        CMPLXF(12.0f, 0.0f), CMPLXF(0.0f, 20.0f), CMPLXF(-2.0f, 0.0f), CMPLXF(10.0f, 0.0f),
        CMPLXF(0.0f, 0.0f),  CMPLXF(20.0f, 0.0f), CMPLXF(2.0f, -1.0f),  CMPLXF(0.0f, 0.9f),
        CMPLXF(0.0f, 0.0f),  CMPLXF(0.0f, 0.0f),  CMPLXF(3.0f, 0.0f),   CMPLXF(0.8f, 0.0f),
        CMPLXF(0.0f, 0.0f),  CMPLXF(0.0f, 0.0f),  CMPLXF(0.0f, 0.0f),   CMPLXF(8.0f, 0.0f)
    };
    tc[4].n = 4; tc[4].ifst = 0; tc[4].ilst = 3;
    rowmajor_to_colmajor(m4, tc[4].mat, 4, LDT);

    /* Case 5: N=5, IFST=4, ILST=0 (Fortran 5,1) */
    static const c64 m5[25] = {
        CMPLXF(1.0f, 1.0f),  CMPLXF(2.0f, -1.0f), CMPLXF(2.0f, -3.0f), CMPLXF(12.0f, 3.0f), CMPLXF(2.0f, 39.0f),
        CMPLXF(0.0f, 0.0f),  CMPLXF(2.0f, 3.0f),  CMPLXF(2.0f, 3.0f),  CMPLXF(2.0f, 13.0f), CMPLXF(2.0f, 31.0f),
        CMPLXF(0.0f, 0.0f),  CMPLXF(0.0f, 0.0f),  CMPLXF(-2.0f, 3.0f), CMPLXF(2.0f, 3.0f),  CMPLXF(12.0f, 3.0f),
        CMPLXF(0.0f, 0.0f),  CMPLXF(0.0f, 0.0f),  CMPLXF(0.0f, 0.0f),  CMPLXF(2.0f, -3.0f), CMPLXF(-2.0f, 3.0f),
        CMPLXF(0.0f, 0.0f),  CMPLXF(0.0f, 0.0f),  CMPLXF(0.0f, 0.0f),  CMPLXF(0.0f, 0.0f),  CMPLXF(2.0f, 3.0f)
    };
    tc[5].n = 5; tc[5].ifst = 4; tc[5].ilst = 0;
    rowmajor_to_colmajor(m5, tc[5].mat, 5, LDT);

    /* Case 6: N=4, IFST=3, ILST=0 (Fortran 4,1) */
    static const c64 m6[16] = {
        CMPLXF(0.0621f, 0.7054f), CMPLXF(0.1062f, 0.0503f), CMPLXF(0.6553f, 0.5876f), CMPLXF(0.2560f, 0.8642f),
        CMPLXF(0.0f, 0.0f),       CMPLXF(0.2640f, 0.5782f), CMPLXF(0.9700f, 0.7256f), CMPLXF(0.5598f, 0.1943f),
        CMPLXF(0.0f, 0.0f),       CMPLXF(0.0f, 0.0f),       CMPLXF(0.0380f, 0.2849f), CMPLXF(0.9166f, 0.0580f),
        CMPLXF(0.0f, 0.0f),       CMPLXF(0.0f, 0.0f),       CMPLXF(0.0f, 0.0f),       CMPLXF(0.1402f, 0.6908f)
    };
    tc[6].n = 4; tc[6].ifst = 3; tc[6].ilst = 0;
    rowmajor_to_colmajor(m6, tc[6].mat, 4, LDT);

    /* Case 7: N=6, IFST=4, ILST=2 (Fortran 5,3) */
    static const c64 m7[36] = {
        CMPLXF(10.0f, 1.0f),  CMPLXF(10.0f, 0.0f),  CMPLXF(30.0f, 0.0f),  CMPLXF(0.0f, 1.0f),    CMPLXF(10.0f, 1.0f),  CMPLXF(10.0f, 0.0f),
        CMPLXF(0.0f, 0.0f),   CMPLXF(20.0f, 1.0f),  CMPLXF(30.0f, 0.0f),  CMPLXF(20.0f, 1.0f),   CMPLXF(0.0f, -1.0f),  CMPLXF(0.0f, -10.0f),
        CMPLXF(0.0f, 0.0f),   CMPLXF(0.0f, 0.0f),   CMPLXF(30.0f, 1.0f),  CMPLXF(0.0f, 0.0f),    CMPLXF(2.0f, 0.0f),   CMPLXF(0.0f, 20.0f),
        CMPLXF(0.0f, 0.0f),   CMPLXF(0.0f, 0.0f),   CMPLXF(0.0f, 0.0f),   CMPLXF(40.0f, 1.0f),   CMPLXF(0.0f, -10.0f), CMPLXF(-30.0f, 0.0f),
        CMPLXF(0.0f, 0.0f),   CMPLXF(0.0f, 0.0f),   CMPLXF(0.0f, 0.0f),   CMPLXF(0.0f, 0.0f),    CMPLXF(50.0f, 1.0f),  CMPLXF(0.0f, 0.0f),
        CMPLXF(0.0f, 0.0f),   CMPLXF(0.0f, 0.0f),   CMPLXF(0.0f, 0.0f),   CMPLXF(0.0f, 0.0f),    CMPLXF(0.0f, 0.0f),   CMPLXF(60.0f, 1.0f)
    };
    tc[7].n = 6; tc[7].ifst = 4; tc[7].ilst = 2;
    rowmajor_to_colmajor(m7, tc[7].mat, 6, LDT);
}

void cget36(f32* rmax, INT* lmax, INT* ninfo, INT* knt)
{
    const f32 ZERO = 0.0f;
    const f32 ONE  = 1.0f;
    const c64 CZERO = CMPLXF(0.0f, 0.0f);
    const c64 CONE  = CMPLXF(1.0f, 0.0f);

    f32 eps = slamch("P");
    *rmax = ZERO;
    *lmax = 0;
    *knt = 0;
    *ninfo = 0;

    ztrexc_case_t cases[NCASES];
    build_test_cases(cases);

    c64 t1[LDT * LDT], t2[LDT * LDT], tmp[LDT * LDT];
    c64 q[LDT * LDT], work[LWORK], diag[LDT];
    f32 result[2], rwork[LDT];
    c64 ctemp;

    for (INT ic = 0; ic < NCASES; ic++) {
        INT n = cases[ic].n;
        INT ifst = cases[ic].ifst;
        INT ilst = cases[ic].ilst;

        (*knt)++;
        clacpy("F", n, n, cases[ic].mat, LDT, tmp, LDT);
        clacpy("F", n, n, tmp, LDT, t1, LDT);
        clacpy("F", n, n, tmp, LDT, t2, LDT);
        f32 res = ZERO;
        INT info1, info2;

        claset("Full", n, n, CZERO, CONE, q, LDT);
        ctrexc("N", n, t1, LDT, q, LDT, ifst, ilst, &info1);
        for (INT i = 0; i < n; i++) {
            for (INT j = 0; j < n; j++) {
                if (i == j && q[i + j * LDT] != CONE)
                    res = res + ONE / eps;
                if (i != j && q[i + j * LDT] != CZERO)
                    res = res + ONE / eps;
            }
        }

        claset("Full", n, n, CZERO, CONE, q, LDT);
        ctrexc("V", n, t2, LDT, q, LDT, ifst, ilst, &info2);

        for (INT i = 0; i < n; i++)
            for (INT j = 0; j < n; j++)
                if (t1[i + j * LDT] != t2[i + j * LDT])
                    res = res + ONE / eps;
        if (info1 != 0 || info2 != 0)
            *ninfo = *ninfo + 1;
        if (info1 != info2)
            res = res + ONE / eps;

        cblas_ccopy(n, tmp, LDT + 1, diag, 1);
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

        chst01(n, 0, n - 1, tmp, LDT, t2, LDT, q, LDT, work, LWORK,
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
