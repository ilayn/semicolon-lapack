/**
 * @file dget40.c
 * @brief DGET40 tests DTGEXC, a routine for swapping adjacent blocks
 *        on the diagonal of a pencil in real generalized Schur form.
 */

#include "verify.h"
#include <cblas.h>
#include <math.h>
#include <string.h>

extern f64 dlamch(const char* cmach);
extern void dlacpy(const char* uplo, const int m, const int n,
                   const f64* A, const int lda, f64* B, const int ldb);
extern void dlaset(const char* uplo, const int m, const int n,
                   const f64 alpha, const f64 beta,
                   f64* A, const int lda);
extern void dtgexc(const int wantq, const int wantz, const int n,
                   f64* restrict A, const int lda,
                   f64* restrict B, const int ldb,
                   f64* restrict Q, const int ldq,
                   f64* restrict Z, const int ldz,
                   int* ifst, int* ilst,
                   f64* restrict work, const int lwork, int* info);

#define LDT 10
#define LWORK (100 + 4*LDT + 16)

/* Test case data structure */
typedef struct {
    int n;
    int ifst;  /* 0-based */
    int ilst;  /* 0-based */
    f64 t[LDT * LDT]; /* column-major */
    f64 s[LDT * LDT]; /* column-major */
} tgexc_case_t;

/* Helper: set n×n identity matrix in column-major ldt-stride storage */
static void set_identity(f64* A, int n, int ldt)
{
    memset(A, 0, (size_t)ldt * n * sizeof(f64));
    for (int i = 0; i < n; i++)
        A[i + i * ldt] = 1.0;
}

/* Helper: copy row-major data to column-major storage */
static void rowmajor_to_colmajor(const f64* rows, f64* cm, int n, int ldcm)
{
    memset(cm, 0, (size_t)ldcm * n * sizeof(f64));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            cm[i + j * ldcm] = rows[i * n + j];
}

/*
 * Embedded test data from LAPACK dec.in.
 * 10 test cases extracted from lines 951-1104.
 * IFST/ILST converted from Fortran 1-based to C 0-based.
 */
static void build_test_cases(tgexc_case_t* tc)
{
    /* Base T matrix shared by cases 0-7 (8×8).
       Variations in rows 1-3 and diagonal blocks. */

    static const f64 t_base[8][8] = {
        { 1.0,  1.0,  1.1,  1.3,  2.0,  3.0, -4.7,  3.3},
        { 1.0,  1.0,  3.7,  7.9,  4.0,  5.3,  3.3, -0.9},
        { 0.0,  0.0,  2.0, -3.0,  3.4,  6.5,  5.2,  1.8},
        { 0.0,  0.0,  4.0,  2.0, -5.3, -8.9, -0.2, -0.5},
        { 0.0,  0.0,  0.0,  0.0,  4.2,  2.0,  3.3,  2.3},
        { 0.0,  0.0,  0.0,  0.0,  3.7,  4.2,  9.9,  8.8},
        { 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  9.9,  8.8},
        { 0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -9.9,  9.9}
    };

    /* Cases 0-3: N=8, IFST=0, ILST=4 (Fortran 1→5) */
    for (int k = 0; k < 4; k++) {
        tc[k].n = 8;
        tc[k].ifst = 0;
        tc[k].ilst = 4;
        rowmajor_to_colmajor(&t_base[0][0], tc[k].t, 8, LDT);
        set_identity(tc[k].s, 8, LDT);
    }

    /* Case 0: T(2,3)=-3.0, T(3,2)=4.0 — already in base */

    /* Case 1: T(2,3)=3.0 (was -3.0) */
    tc[1].t[2 + 3 * LDT] = 3.0;

    /* Case 2: T(2,3)=3.0, T(3,2)=0.0 (was 4.0) */
    tc[2].t[2 + 3 * LDT] = 3.0;
    tc[2].t[3 + 2 * LDT] = 0.0;

    /* Case 3: T(1,0)=0.0 (was 1.0), T(2,2)=0.0 (was 2.0), T(2,3)=3.0, T(3,2)=0.0 */
    tc[3].t[1 + 0 * LDT] = 0.0;
    tc[3].t[2 + 2 * LDT] = 0.0;
    tc[3].t[2 + 3 * LDT] = 3.0;
    tc[3].t[3 + 2 * LDT] = 0.0;

    /* Cases 4-7: N=8, IFST=4, ILST=0 (Fortran 5→1). Same T as 0-3. */
    for (int k = 0; k < 4; k++) {
        tc[k + 4].n = 8;
        tc[k + 4].ifst = 4;
        tc[k + 4].ilst = 0;
        memcpy(tc[k + 4].t, tc[k].t, sizeof(tc[k].t));
        set_identity(tc[k + 4].s, 8, LDT);
    }

    /* Case 8: N=4, IFST=0, ILST=2 (Fortran 1→3) */
    {
        static const f64 t8_rows[4][4] = {
            {1.0, 1.0, 1.1, 1.3},
            {1.0, 1.0, 3.7, 7.9},
            {0.0, 0.0, 1.0, 1.0},
            {0.0, 0.0, 1.0, 1.0}
        };
        tc[8].n = 4;
        tc[8].ifst = 0;
        tc[8].ilst = 2;
        rowmajor_to_colmajor(&t8_rows[0][0], tc[8].t, 4, LDT);
        set_identity(tc[8].s, 4, LDT);
    }

    /* Case 9: N=4, IFST=0, ILST=2 (Fortran 1→3), with non-identity S */
    {
        static const f64 t9_rows[4][4] = {
            { 7.214055213169724e-01,  9.376135742769982e-01,  5.318280700344581e-01,  9.787531445044610e-01},
            {-9.376135742769982e-01,  7.214055213169724e-01,  7.801161815573352e-01,  5.734592675974027e-01},
            { 0.000000000000000e+00,  0.000000000000000e+00,  7.214055213169726e-01,  9.376135742769983e-01},
            { 0.000000000000000e+00,  0.000000000000000e+00, -9.376135742769983e-01,  7.214055213169726e-01}
        };
        static const f64 s9_rows[4][4] = {
            {1.000000000000000e+00, 0.000000000000000e+00, 5.589642506777136e-01, 6.410964218208657e-01},
            {0.000000000000000e+00, 1.000000000000000e+00, 4.839061798610604e-01, 6.207808731846947e-01},
            {0.000000000000000e+00, 0.000000000000000e+00, 1.000000000000000e+00, 0.000000000000000e+00},
            {0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 1.000000000000000e+00}
        };
        tc[9].n = 4;
        tc[9].ifst = 0;
        tc[9].ilst = 2;
        rowmajor_to_colmajor(&t9_rows[0][0], tc[9].t, 4, LDT);
        rowmajor_to_colmajor(&s9_rows[0][0], tc[9].s, 4, LDT);
    }
}

#define NCASES 10

/**
 * DGET40 tests DTGEXC, a routine for swapping adjacent blocks (either
 * 1 by 1 or 2 by 2) on the diagonal of a pencil in real generalized Schur form.
 *
 * @param[out]    rmax    Value of the largest test ratio.
 * @param[out]    lmax    Example number where largest test ratio achieved.
 * @param[out]    ninfo   Integer array, dimension (2).
 *                        ninfo[0] = DTGEXC without accumulation returned INFO nonzero
 *                        ninfo[1] = DTGEXC with accumulation returned INFO nonzero
 * @param[out]    knt     Total number of examples tested.
 */
void dget40(f64* rmax, int* lmax, int* ninfo, int* knt)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    f64 eps = dlamch("P");
    *rmax = ZERO;
    *lmax = 0;
    *knt = 0;
    ninfo[0] = 0;
    ninfo[1] = 0;

    /* Build test cases */

    tgexc_case_t tc[NCASES];
    build_test_cases(tc);

    /* Local arrays */

    f64 t[LDT * LDT], t1[LDT * LDT], t2[LDT * LDT];
    f64 s[LDT * LDT], s1[LDT * LDT], s2[LDT * LDT];
    f64 q[LDT * LDT], z[LDT * LDT];
    f64 work[LWORK];
    f64 result[4];

    for (int ic = 0; ic < NCASES; ic++) {
        int n = tc[ic].n;
        (*knt)++;

        dlacpy("F", n, n, tc[ic].t, LDT, t, LDT);
        dlacpy("F", n, n, tc[ic].t, LDT, t1, LDT);
        dlacpy("F", n, n, tc[ic].t, LDT, t2, LDT);
        dlacpy("F", n, n, tc[ic].s, LDT, s, LDT);
        dlacpy("F", n, n, tc[ic].s, LDT, s1, LDT);
        dlacpy("F", n, n, tc[ic].s, LDT, s2, LDT);

        int ifst1 = tc[ic].ifst, ilst1 = tc[ic].ilst;
        int ifst2 = tc[ic].ifst, ilst2 = tc[ic].ilst;
        f64 res = ZERO;

        /* Test without accumulating Q and Z */

        dlaset("Full", n, n, ZERO, ONE, q, LDT);
        dlaset("Full", n, n, ZERO, ONE, z, LDT);
        int info1 = 0;
        dtgexc(0, 0, n, t1, LDT, s1, LDT, q, LDT,
               z, LDT, &ifst1, &ilst1, work, LWORK, &info1);
        if (info1 != 0)
            ninfo[0]++;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (i == j && q[i + j * LDT] != ONE)
                    res += ONE / eps;
                if (i != j && q[i + j * LDT] != ZERO)
                    res += ONE / eps;
                if (i == j && z[i + j * LDT] != ONE)
                    res += ONE / eps;
                if (i != j && z[i + j * LDT] != ZERO)
                    res += ONE / eps;
            }
        }

        /* Test with accumulating Q and Z */

        dlaset("Full", n, n, ZERO, ONE, q, LDT);
        dlaset("Full", n, n, ZERO, ONE, z, LDT);
        int info2 = 0;
        dtgexc(1, 1, n, t2, LDT, s2, LDT, q, LDT,
               z, LDT, &ifst2, &ilst2, work, LWORK, &info2);
        if (info2 != 0)
            ninfo[1]++;

        /* Compare T1 with T2 and S1 with S2 */

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (t1[i + j * LDT] != t2[i + j * LDT])
                    res += ONE / eps;
                if (s1[i + j * LDT] != s2[i + j * LDT])
                    res += ONE / eps;
            }
        }
        if (ifst1 != ifst2)
            res += ONE / eps;
        if (ilst1 != ilst2)
            res += ONE / eps;
        if (info1 != info2)
            res += ONE / eps;

        /* Test orthogonality of Q and Z and backward error on T2 and S2 */

        dget51(1, n, t, LDT, t2, LDT, q, LDT, z, LDT, work, &result[0]);
        dget51(1, n, s, LDT, s2, LDT, q, LDT, z, LDT, work, &result[1]);
        dget51(3, n, t, LDT, t2, LDT, q, LDT, q, LDT, work, &result[2]);
        dget51(3, n, t, LDT, t2, LDT, z, LDT, z, LDT, work, &result[3]);
        res += result[0] + result[1] + result[2] + result[3];

        if (res > *rmax) {
            *rmax = res;
            *lmax = *knt;
        }
    }
}
