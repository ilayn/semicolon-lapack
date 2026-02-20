/**
 * @file sget36.c
 * @brief SGET36 tests STREXC, a routine for moving blocks (either 1 by 1
 *        or 2 by 2) on the diagonal of a matrix in real Schur form.
 */

#include "verify.h"
#include <math.h>
#include <string.h>

extern f32 slamch(const char* cmach);
extern void slacpy(const char* uplo, const int m, const int n,
                   const f32* A, const int lda, f32* B, const int ldb);
extern void slaset(const char* uplo, const int m, const int n,
                   const f32 alpha, const f32 beta,
                   f32* A, const int lda);
extern void strexc(const char* compq, const int n, f32* T, const int ldt,
                   f32* Q, const int ldq, int* ifst, int* ilst,
                   f32* work, int* info);

#define LDT 10
#define LWORK (2 * LDT * LDT)
#define NCASES 14

typedef struct {
    int n;
    int ifst;  /* 0-based */
    int ilst;  /* 0-based */
    f32 mat[LDT * LDT]; /* column-major */
} trexc_case_t;

static void rowmajor_to_colmajor(const f32* rows, f32* cm, int n, int ldcm)
{
    memset(cm, 0, (size_t)ldcm * n * sizeof(f32));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            cm[i + j * ldcm] = rows[i * n + j];
}

static void build_test_cases(trexc_case_t* tc)
{
    /* Case 0: N=8, IFST=1, ILST=6 (0-based) */
    static const f32 m0[8][8] = {
        { 1.0f,  1.0f,  1.1f,  1.3f,  2.0f,  3.0f, -4.7f,  3.3f},
        {-1.0f,  1.0f,  3.7f,  7.9f,  4.0f,  5.3f,  3.3f, -0.9f},
        { 0.0f,  0.0f,  2.0f, -3.0f,  3.4f,  6.5f,  5.2f,  1.8f},
        { 0.0f,  0.0f,  4.0f,  2.0f, -5.3f, -8.9f, -0.2f, -0.5f},
        { 0.0f,  0.0f,  0.0f,  0.0f,  4.2f,  2.0f,  3.3f,  2.3f},
        { 0.0f,  0.0f,  0.0f,  0.0f, -3.7f,  4.2f,  9.9f,  8.8f},
        { 0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  9.9f,  8.8f},
        { 0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f, -9.9f,  9.9f}
    };
    tc[0].n = 8; tc[0].ifst = 1; tc[0].ilst = 6;
    rowmajor_to_colmajor(&m0[0][0], tc[0].mat, 8, LDT);

    /* Case 1: N=8, IFST=6, ILST=1 (Fortran 7→2) */
    tc[1].n = 8; tc[1].ifst = 6; tc[1].ilst = 1;
    rowmajor_to_colmajor(&m0[0][0], tc[1].mat, 8, LDT);

    /* Case 2: N=8, IFST=0, ILST=6 (Fortran 1→7) */
    static const f32 m2[8][8] = {
        { 1.0f,  1.0f,  1.1f,  1.3f,  2.0f,  3.0f, -4.7f,  3.3f},
        { 0.0f,  1.0f,  3.7f,  7.9f,  4.0f,  5.3f,  3.3f, -0.9f},
        { 0.0f,  0.0f,  2.0f, -3.0f,  3.4f,  6.5f,  5.2f,  1.8f},
        { 0.0f,  0.0f,  4.0f,  2.0f, -5.3f, -8.9f, -0.2f, -0.5f},
        { 0.0f,  0.0f,  0.0f,  0.0f,  4.2f,  2.0f,  3.3f,  2.3f},
        { 0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  4.2f,  9.9f,  8.8f},
        { 0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  9.9f,  8.8f},
        { 0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f, -9.9f,  9.9f}
    };
    tc[2].n = 8; tc[2].ifst = 0; tc[2].ilst = 6;
    rowmajor_to_colmajor(&m2[0][0], tc[2].mat, 8, LDT);

    /* Case 3: N=8, IFST=7, ILST=1 (Fortran 8→2) */
    static const f32 m3[8][8] = {
        { 1.0f,  1.0f,  1.1f,  1.3f,  2.0f,  3.0f, -4.7f,  3.3f},
        {-1.1f,  1.0f,  3.7f,  7.9f,  4.0f,  5.3f,  3.3f, -0.9f},
        { 0.0f,  0.0f,  2.0f, -3.0f,  3.4f,  6.5f,  5.2f,  1.8f},
        { 0.0f,  0.0f,  0.0f,  2.0f, -5.3f, -8.9f, -0.2f, -0.5f},
        { 0.0f,  0.0f,  0.0f,  0.0f,  4.2f,  2.0f,  3.3f,  2.3f},
        { 0.0f,  0.0f,  0.0f,  0.0f, -3.7f,  4.2f,  9.9f,  8.8f},
        { 0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  9.9f,  8.8f},
        { 0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  9.9f}
    };
    tc[3].n = 8; tc[3].ifst = 7; tc[3].ilst = 1;
    rowmajor_to_colmajor(&m3[0][0], tc[3].mat, 8, LDT);

    /* Case 4: N=7, IFST=1, ILST=6 (Fortran 2→7) */
    static const f32 m4[7][7] = {
        { 1.1f,  1.0e-16f,  2.7f,  3.3f,  2.3f,  3.4f,  5.6f},
        {-1.0e-16f,  1.1f,  4.2f,  5.1f, -0.1f, -0.2f, -0.3f},
        { 0.0f,  0.0f,  2.3f,  1.0f,  1.0e2f,  1.0e3f,  1.0e2f},
        { 0.0f,  0.0f,  0.0f,  3.9f,  3.2f,  6.5f,  3.2f},
        { 0.0f,  0.0f,  0.0f, -0.9f,  3.9f,  6.3f,  3.0f},
        { 0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  6.3f,  3.0f},
        { 0.0f,  0.0f,  0.0f,  0.0f,  0.0f, -0.9f,  6.3f}
    };
    tc[4].n = 7; tc[4].ifst = 1; tc[4].ilst = 6;
    rowmajor_to_colmajor(&m4[0][0], tc[4].mat, 7, LDT);

    /* Case 5: N=7, IFST=1, ILST=6 (Fortran 2→7) */
    static const f32 m5[7][7] = {
        { 1.1f,  1.0e-16f,  2.7f,  3.3f,  2.3f,  3.4f,  5.6f},
        {-1.0e-16f,  1.1f,  4.2f,  5.1f, -0.1f, -0.2f, -0.3f},
        { 0.0f,  0.0f,  2.3f,  1.0f,  1.0e2f,  1.0e3f,  1.0e2f},
        { 0.0f,  0.0f,  0.0f,  3.9f,  3.2e-15f,  6.5f,  3.2f},
        { 0.0f,  0.0f,  0.0f, -9.0e-16f,  3.9f,  6.3f,  3.0f},
        { 0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  6.3f,  3.0f},
        { 0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  6.4f}
    };
    tc[5].n = 7; tc[5].ifst = 1; tc[5].ilst = 6;
    rowmajor_to_colmajor(&m5[0][0], tc[5].mat, 7, LDT);

    /* Case 6: N=7, IFST=1, ILST=6 (Fortran 2→7) */
    static const f32 m6[7][7] = {
        { 1.1f,  1.0e-16f,  2.7f,  3.3f,  2.3f,  3.4f,  5.6f},
        {-1.0e-16f,  1.1f,  4.2f,  5.1f, -0.1f, -0.2f, -0.3f},
        { 0.0f,  0.0f,  2.3f,  1.0f,  1.0e2f,  1.0e3f,  1.0e2f},
        { 0.0f,  0.0f,  0.0f,  3.9f,  3.2e-15f,  6.5f,  3.2f},
        { 0.0f,  0.0f,  0.0f, -9.0e-16f,  3.9f,  6.3f,  3.0f},
        { 0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  6.3f,  3.0f},
        { 0.0f,  0.0f,  0.0f,  0.0f,  0.0f, -9.0e-21f,  6.3f}
    };
    tc[6].n = 7; tc[6].ifst = 1; tc[6].ilst = 6;
    rowmajor_to_colmajor(&m6[0][0], tc[6].mat, 7, LDT);

    /* Case 7: N=7, IFST=0, ILST=6 (Fortran 1→7) */
    static const f32 m7[7][7] = {
        { 1.1f,  1.0e-16f,  2.7f,  3.3f,  2.3f,  3.4f,  5.6f},
        { 0.0f,  1.1f,  4.2f,  5.1f, -0.1f, -0.2f, -0.3f},
        { 0.0f,  0.0f,  2.3f,  1.0f,  1.0e2f,  1.0e3f,  1.0e2f},
        { 0.0f,  0.0f,  0.0f,  3.9f,  3.2e-15f,  6.5f,  3.2f},
        { 0.0f,  0.0f,  0.0f, -9.0e-16f,  3.9f,  6.3f,  3.0f},
        { 0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  6.3f,  3.0f},
        { 0.0f,  0.0f,  0.0f,  0.0f,  0.0f, -9.0e-21f,  6.3f}
    };
    tc[7].n = 7; tc[7].ifst = 0; tc[7].ilst = 6;
    rowmajor_to_colmajor(&m7[0][0], tc[7].mat, 7, LDT);

    /* Case 8: N=7, IFST=0, ILST=6 (Fortran 1→7) */
    static const f32 m8[7][7] = {
        { 1.1f, -1.1f,  2.7f,  3.3f,  2.3f,  3.4f,  5.6f},
        { 2.3f,  1.1f,  4.2f,  5.1f, -0.1f, -0.2f, -0.3f},
        { 0.0f,  0.0f,  2.3f,  1.0f,  1.0e2f,  1.0e3f,  1.0e2f},
        { 0.0f,  0.0f,  0.0f,  3.9f,  3.2f,  6.5f,  3.2f},
        { 0.0f,  0.0f,  0.0f, -9.0e-21f,  3.9f,  6.3f,  3.0f},
        { 0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  6.3f,  3.0e-20f},
        { 0.0f,  0.0f,  0.0f,  0.0f,  0.0f, -9.0e-21f,  6.3f}
    };
    tc[8].n = 7; tc[8].ifst = 0; tc[8].ilst = 6;
    rowmajor_to_colmajor(&m8[0][0], tc[8].mat, 7, LDT);

    /* Case 9: N=7, IFST=6, ILST=1 (Fortran 7→2) */
    static const f32 m9[7][7] = {
        { 6.3f,  3.0f,  2.7f,  3.3f,  2.3f,  3.4f,  5.6f},
        {-0.9f,  6.3f,  4.2f,  5.1f, -0.1f, -0.2f, -0.3f},
        { 0.0f,  0.0f,  2.3f,  1.0f,  1.0e2f,  1.0e3f,  1.0e2f},
        { 0.0f,  0.0f,  0.0f,  3.9f,  3.2f,  6.5f,  3.2f},
        { 0.0f,  0.0f,  0.0f,  0.0f,  3.8f,  6.3f,  3.0f},
        { 0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  1.1f,  1.4e-20f},
        { 0.0f,  0.0f,  0.0f,  0.0f,  0.0f, -1.6e-20f,  1.1f}
    };
    tc[9].n = 7; tc[9].ifst = 6; tc[9].ilst = 1;
    rowmajor_to_colmajor(&m9[0][0], tc[9].mat, 7, LDT);

    /* Case 10: N=7, IFST=6, ILST=1 (Fortran 7→2) */
    static const f32 m10[7][7] = {
        { 6.3f,  3.0f,  2.7f,  3.3f,  2.3f,  3.4f,  5.6f},
        {-0.9f,  6.3f,  4.2f,  5.1f, -0.1f, -0.2f, -0.3f},
        { 0.0f,  0.0f,  2.3f,  1.0f,  1.0e2f,  1.0e3f,  1.0e2f},
        { 0.0f,  0.0f,  0.0f,  3.9f,  3.2f,  6.5f,  3.2f},
        { 0.0f,  0.0f,  0.0f, -0.9f,  3.9f,  6.3f,  3.0f},
        { 0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  1.1f,  1.4e-20f},
        { 0.0f,  0.0f,  0.0f,  0.0f,  0.0f, -1.6e-20f,  1.1f}
    };
    tc[10].n = 7; tc[10].ifst = 6; tc[10].ilst = 1;
    rowmajor_to_colmajor(&m10[0][0], tc[10].mat, 7, LDT);

    /* Case 11: N=7, IFST=6, ILST=1 (Fortran 7→2) */
    static const f32 m11[7][7] = {
        { 1.1f,  1.0e-16f,  2.7f,  3.3f,  2.3f,  3.4f,  5.6f},
        {-1.0e-16f,  1.1f,  4.2f,  5.1f, -0.1f, -0.2f, -0.3f},
        { 0.0f,  0.0f,  2.3f,  1.0f,  1.0e2f,  1.0e3f,  1.0e2f},
        { 0.0f,  0.0f,  0.0f,  3.9f,  3.2e-15f,  6.5f,  3.2f},
        { 0.0f,  0.0f,  0.0f, -9.0e-16f,  3.9f,  6.3f,  3.0f},
        { 0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  6.3f,  3.0f},
        { 0.0f,  0.0f,  0.0f,  0.0f,  0.0f, -9.0e-21f,  6.3f}
    };
    tc[11].n = 7; tc[11].ifst = 6; tc[11].ilst = 1;
    rowmajor_to_colmajor(&m11[0][0], tc[11].mat, 7, LDT);

    /* Case 12: N=7, IFST=6, ILST=0 (Fortran 7→1) */
    static const f32 m12[7][7] = {
        { 1.1f,  1.0e-16f,  2.7e6f,  3.3f,  2.3f,  3.4f,  5.6f},
        { 0.0f,  1.1f,  4.2e6f,  5.1f, -0.1f, -0.2f, -0.3f},
        { 0.0f,  0.0f,  2.3f,  1.0e7f,  1.0e8f,  1.0e3f,  1.0e2f},
        { 0.0f,  0.0f,  0.0f,  3.9f,  3.2e-15f,  6.5e4f,  3.2f},
        { 0.0f,  0.0f,  0.0f, -9.0e-16f,  3.9f,  6.3e3f,  3.0e5f},
        { 0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  6.3f,  3.0f},
        { 0.0f,  0.0f,  0.0f,  0.0f,  0.0f, -9.0e-21f,  6.3f}
    };
    tc[12].n = 7; tc[12].ifst = 6; tc[12].ilst = 0;
    rowmajor_to_colmajor(&m12[0][0], tc[12].mat, 7, LDT);

    /* Case 13: N=8, IFST=7, ILST=0 (Fortran 8→1) */
    static const f32 m13[8][8] = {
        { 1.1f, -1.0e-16f,  2.7e6f,  2.3e4f,  3.3f,  2.3f,  3.4f,  5.6f},
        { 1.0e-16f,  1.1f,  4.2e6f, -0.1f,  5.1f, -0.1f, -0.2f, -0.3f},
        { 0.0f,  0.0f,  2.3f,  1.1e-16f,  1.0e7f,  1.0e8f,  1.0e3f,  1.0e2f},
        { 0.0f,  0.0f, -1.1e-13f,  2.3f,  1.0e7f,  1.0e8f,  1.0e3f,  1.0e2f},
        { 0.0f,  0.0f,  0.0f,  0.0f,  3.9f,  3.2e-15f,  6.5e4f,  3.2f},
        { 0.0f,  0.0f,  0.0f,  0.0f, -9.0e-16f,  3.9f,  6.3e3f,  3.0e5f},
        { 0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  6.3f,  3.0e-20f},
        { 0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f, -9.0e-21f,  6.3f}
    };
    tc[13].n = 8; tc[13].ifst = 7; tc[13].ilst = 0;
    rowmajor_to_colmajor(&m13[0][0], tc[13].mat, 8, LDT);
}

void sget36(f32* rmax, int* lmax, int ninfo[3], int* knt)
{
    const f32 ZERO = 0.0f;
    const f32 ONE  = 1.0f;

    f32 eps = slamch("P");
    *rmax = ZERO;
    *lmax = 0;
    *knt = 0;
    ninfo[0] = 0;
    ninfo[1] = 0;
    ninfo[2] = 0;

    trexc_case_t cases[NCASES];
    build_test_cases(cases);

    f32 t1[LDT * LDT], t2[LDT * LDT], tmp[LDT * LDT];
    f32 q[LDT * LDT], work[LWORK], result[2];

    for (int ic = 0; ic < NCASES; ic++) {
        int n = cases[ic].n;
        int ifst = cases[ic].ifst;
        int ilst = cases[ic].ilst;

        (*knt)++;
        slacpy("F", n, n, cases[ic].mat, LDT, tmp, LDT);
        slacpy("F", n, n, tmp, LDT, t1, LDT);
        slacpy("F", n, n, tmp, LDT, t2, LDT);
        int ifstsv = ifst;
        int ilstsv = ilst;
        int ifst1 = ifst;
        int ilst1 = ilst;
        int ifst2 = ifst;
        int ilst2 = ilst;
        f32 res = ZERO;
        int info1, info2;

        slaset("Full", n, n, ZERO, ONE, q, LDT);
        strexc("N", n, t1, LDT, q, LDT, &ifst1, &ilst1, work, &info1);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (i == j && q[i + j * LDT] != ONE)
                    res = res + ONE / eps;
                if (i != j && q[i + j * LDT] != ZERO)
                    res = res + ONE / eps;
            }
        }

        slaset("Full", n, n, ZERO, ONE, q, LDT);
        strexc("V", n, t2, LDT, q, LDT, &ifst2, &ilst2, work, &info2);

        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                if (t1[i + j * LDT] != t2[i + j * LDT])
                    res = res + ONE / eps;
        if (ifst1 != ifst2)
            res = res + ONE / eps;
        if (ilst1 != ilst2)
            res = res + ONE / eps;
        if (info1 != info2)
            res = res + ONE / eps;

        if (info2 != 0) {
            ninfo[info2 - 1]++;
        } else {
            int d1 = ifst2 - ifstsv;
            if (d1 < 0) d1 = -d1;
            if (d1 > 1)
                res = res + ONE / eps;
            int d2 = ilst2 - ilstsv;
            if (d2 < 0) d2 = -d2;
            if (d2 > 1)
                res = res + ONE / eps;
        }

        shst01(n, 0, n - 1, tmp, LDT, t2, LDT, q, LDT, work, LWORK,
               result);
        res = res + result[0] + result[1];

        int loc = 0;
        while (loc < n - 1) {
            if (t2[(loc + 1) + loc * LDT] != ZERO) {
                if (t2[loc + (loc + 1) * LDT] == ZERO ||
                    t2[loc + loc * LDT] != t2[(loc + 1) + (loc + 1) * LDT] ||
                    copysignf(ONE, t2[loc + (loc + 1) * LDT]) ==
                    copysignf(ONE, t2[(loc + 1) + loc * LDT]))
                    res = res + ONE / eps;
                for (int i = loc + 2; i < n; i++) {
                    if (t2[i + loc * LDT] != ZERO)
                        res = res + ONE / res;
                    if (t2[i + (loc + 1) * LDT] != ZERO)
                        res = res + ONE / res;
                }
                loc = loc + 2;
            } else {
                for (int i = loc + 1; i < n; i++)
                    if (t2[i + loc * LDT] != ZERO)
                        res = res + ONE / res;
                loc = loc + 1;
            }
        }
        if (res > *rmax) {
            *rmax = res;
            *lmax = *knt;
        }
    }
}
