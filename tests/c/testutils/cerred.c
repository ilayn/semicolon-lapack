/**
 * @file cerred.c
 * @brief CERRED tests the error exits for the eigenvalue driver routines.
 *
 * Port of LAPACK TESTING/EIG/cerred.f
 * Tests error exits for: CGEEV, CGEES, CGEEVX, CGEESX, CGESVD, CGESDD,
 * CGEJSV, CGESVDX, CGESVDQ
 */

#include <complex.h>
#include <math.h>
#include <string.h>
#include "semicolon_lapack_complex_single.h"
#include "verify.h"

static INT selopt;
static INT seldim;
static INT selval[20];
static f32 selwr[20];
static f32 selwi[20];

/**
 * CSLECT returns .TRUE. if the eigenvalue Z is to be selected,
 * otherwise it returns .FALSE.
 * It is used by CCHK41 to test if CGEES successfully sorts eigenvalues,
 * and by CCHK43 to test if CGEESX successfully sorts eigenvalues.
 *
 * The common block /SSLCT/ controls how eigenvalues are selected.
 * If SELOPT = 0, then CSLECT return .TRUE. when real(Z) is less than
 * zero, and .FALSE. otherwise.
 * If SELOPT is at least 1, CSLECT returns SELVAL(SELOPT) and adds 1
 * to SELOPT, cycling back to 1 at SELMAX.
 */
static INT cslect(const c64* z)
{
    const f32 ZERO = 0.0f;
    INT i, val;
    f32 rmin, x;

    if (selopt == 0) {
        val = (crealf(*z) < ZERO);
    } else {
        rmin = cabsf(*z - CMPLXF(selwr[0], selwi[0]));
        val = selval[0];
        for (i = 1; i < seldim; i++) {
            x = cabsf(*z - CMPLXF(selwr[i], selwi[i]));
            if (x <= rmin) {
                rmin = x;
                val = selval[i];
            }
        }
    }
    return val;
}

void cerred(const char* path, INT* ok, INT* nt)
{
    enum { NMAX = 4, LW = 5 * NMAX };
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;

    INT i, ihi, ilo, info, j, ns, sdim;
    f32 abnrm;

    INT b[NMAX];
    INT iw[4 * NMAX];
    f32 r1[NMAX], r2[NMAX], rw[LW], s[NMAX];
    c64 a[NMAX * NMAX], u[NMAX * NMAX], vl[NMAX * NMAX], vr[NMAX * NMAX];
    c64 vt[NMAX * NMAX], w[10 * NMAX], x[NMAX];

    char c2[2];
    c2[0] = path[1];
    c2[1] = path[2];

    /* Initialize A */
    for (j = 0; j < NMAX; j++) {
        for (i = 0; i < NMAX; i++) {
            a[i + j * NMAX] = ZERO;
        }
    }
    for (i = 0; i < NMAX; i++) {
        a[i + i * NMAX] = ONE;
    }
    xerbla_ok = 1;
    *nt = 0;

    if ((c2[0] == 'E' || c2[0] == 'e') && (c2[1] == 'V' || c2[1] == 'v')) {

        /* Test CGEEV */

        strcpy(xerbla_srnamt, "CGEEV ");
        xerbla_infot = 1;
        cgeev("X", "N", 0, a, 1, x, vl, 1, vr, 1, w, 1, rw, &info);
        chkxer("CGEEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        cgeev("N", "X", 0, a, 1, x, vl, 1, vr, 1, w, 1, rw, &info);
        chkxer("CGEEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        cgeev("N", "N", -1, a, 1, x, vl, 1, vr, 1, w, 1, rw, &info);
        chkxer("CGEEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        cgeev("N", "N", 2, a, 1, x, vl, 1, vr, 1, w, 4, rw, &info);
        chkxer("CGEEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        cgeev("V", "N", 2, a, 2, x, vl, 1, vr, 1, w, 4, rw, &info);
        chkxer("CGEEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        cgeev("N", "V", 2, a, 2, x, vl, 1, vr, 1, w, 4, rw, &info);
        chkxer("CGEEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 12;
        cgeev("V", "V", 1, a, 1, x, vl, 1, vr, 1, w, 1, rw, &info);
        chkxer("CGEEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 7;

    } else if ((c2[0] == 'E' || c2[0] == 'e') &&
               (c2[1] == 'S' || c2[1] == 's')) {

        /* Test CGEES */

        strcpy(xerbla_srnamt, "CGEES ");
        xerbla_infot = 1;
        cgees("X", "N", cslect, 0, a, 1, &sdim, x, vl, 1, w, 1,
              rw, b, &info);
        chkxer("CGEES ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        cgees("N", "X", cslect, 0, a, 1, &sdim, x, vl, 1, w, 1,
              rw, b, &info);
        chkxer("CGEES ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        cgees("N", "S", cslect, -1, a, 1, &sdim, x, vl, 1, w, 1,
              rw, b, &info);
        chkxer("CGEES ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        cgees("N", "S", cslect, 2, a, 1, &sdim, x, vl, 1, w, 4,
              rw, b, &info);
        chkxer("CGEES ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        cgees("V", "S", cslect, 2, a, 2, &sdim, x, vl, 1, w, 4,
              rw, b, &info);
        chkxer("CGEES ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 12;
        cgees("N", "S", cslect, 1, a, 1, &sdim, x, vl, 1, w, 1,
              rw, b, &info);
        chkxer("CGEES ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 6;

    } else if ((c2[0] == 'V' || c2[0] == 'v') &&
               (c2[1] == 'X' || c2[1] == 'x')) {

        /* Test CGEEVX */

        strcpy(xerbla_srnamt, "CGEEVX");
        xerbla_infot = 1;
        cgeevx("X", "N", "N", "N", 0, a, 1, x, vl, 1, vr, 1, &ilo,
               &ihi, s, &abnrm, r1, r2, w, 1, rw, &info);
        chkxer("CGEEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        cgeevx("N", "X", "N", "N", 0, a, 1, x, vl, 1, vr, 1, &ilo,
               &ihi, s, &abnrm, r1, r2, w, 1, rw, &info);
        chkxer("CGEEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        cgeevx("N", "N", "X", "N", 0, a, 1, x, vl, 1, vr, 1, &ilo,
               &ihi, s, &abnrm, r1, r2, w, 1, rw, &info);
        chkxer("CGEEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        cgeevx("N", "N", "N", "X", 0, a, 1, x, vl, 1, vr, 1, &ilo,
               &ihi, s, &abnrm, r1, r2, w, 1, rw, &info);
        chkxer("CGEEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        cgeevx("N", "N", "N", "N", -1, a, 1, x, vl, 1, vr, 1,
               &ilo, &ihi, s, &abnrm, r1, r2, w, 1, rw, &info);
        chkxer("CGEEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        cgeevx("N", "N", "N", "N", 2, a, 1, x, vl, 1, vr, 1, &ilo,
               &ihi, s, &abnrm, r1, r2, w, 4, rw, &info);
        chkxer("CGEEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        cgeevx("N", "V", "N", "N", 2, a, 2, x, vl, 1, vr, 1, &ilo,
               &ihi, s, &abnrm, r1, r2, w, 4, rw, &info);
        chkxer("CGEEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 12;
        cgeevx("N", "N", "V", "N", 2, a, 2, x, vl, 1, vr, 1, &ilo,
               &ihi, s, &abnrm, r1, r2, w, 4, rw, &info);
        chkxer("CGEEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 20;
        cgeevx("N", "N", "N", "N", 1, a, 1, x, vl, 1, vr, 1, &ilo,
               &ihi, s, &abnrm, r1, r2, w, 1, rw, &info);
        chkxer("CGEEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 20;
        cgeevx("N", "N", "V", "V", 1, a, 1, x, vl, 1, vr, 1, &ilo,
               &ihi, s, &abnrm, r1, r2, w, 2, rw, &info);
        chkxer("CGEEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 10;

    } else if ((c2[0] == 'S' || c2[0] == 's') &&
               (c2[1] == 'X' || c2[1] == 'x')) {

        /* Test CGEESX */

        strcpy(xerbla_srnamt, "CGEESX");
        xerbla_infot = 1;
        cgeesx("X", "N", cslect, "N", 0, a, 1, &sdim, x, vl, 1,
               &r1[0], &r2[0], w, 1, rw, b, &info);
        chkxer("CGEESX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        cgeesx("N", "X", cslect, "N", 0, a, 1, &sdim, x, vl, 1,
               &r1[0], &r2[0], w, 1, rw, b, &info);
        chkxer("CGEESX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        cgeesx("N", "N", cslect, "X", 0, a, 1, &sdim, x, vl, 1,
               &r1[0], &r2[0], w, 1, rw, b, &info);
        chkxer("CGEESX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        cgeesx("N", "N", cslect, "N", -1, a, 1, &sdim, x, vl, 1,
               &r1[0], &r2[0], w, 1, rw, b, &info);
        chkxer("CGEESX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        cgeesx("N", "N", cslect, "N", 2, a, 1, &sdim, x, vl, 1,
               &r1[0], &r2[0], w, 4, rw, b, &info);
        chkxer("CGEESX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 11;
        cgeesx("V", "N", cslect, "N", 2, a, 2, &sdim, x, vl, 1,
               &r1[0], &r2[0], w, 4, rw, b, &info);
        chkxer("CGEESX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 15;
        cgeesx("N", "N", cslect, "N", 1, a, 1, &sdim, x, vl, 1,
               &r1[0], &r2[0], w, 1, rw, b, &info);
        chkxer("CGEESX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 7;

    } else if ((c2[0] == 'B' || c2[0] == 'b') &&
               (c2[1] == 'D' || c2[1] == 'd')) {

        /* Test CGESVD */

        strcpy(xerbla_srnamt, "CGESVD");
        xerbla_infot = 1;
        cgesvd("X", "N", 0, 0, a, 1, s, u, 1, vt, 1, w, 1, rw, &info);
        chkxer("CGESVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        cgesvd("N", "X", 0, 0, a, 1, s, u, 1, vt, 1, w, 1, rw, &info);
        chkxer("CGESVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        cgesvd("O", "O", 0, 0, a, 1, s, u, 1, vt, 1, w, 1, rw, &info);
        chkxer("CGESVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        cgesvd("N", "N", -1, 0, a, 1, s, u, 1, vt, 1, w, 1, rw, &info);
        chkxer("CGESVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        cgesvd("N", "N", 0, -1, a, 1, s, u, 1, vt, 1, w, 1, rw, &info);
        chkxer("CGESVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        cgesvd("N", "N", 2, 1, a, 1, s, u, 1, vt, 1, w, 5, rw, &info);
        chkxer("CGESVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        cgesvd("A", "N", 2, 1, a, 2, s, u, 1, vt, 1, w, 5, rw, &info);
        chkxer("CGESVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 11;
        cgesvd("N", "A", 1, 2, a, 1, s, u, 1, vt, 1, w, 5, rw, &info);
        chkxer("CGESVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 8;

        /* Test CGESDD */

        strcpy(xerbla_srnamt, "CGESDD");
        xerbla_infot = 1;
        cgesdd("X", 0, 0, a, 1, s, u, 1, vt, 1, w, 1, rw, iw, &info);
        chkxer("CGESDD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        cgesdd("N", -1, 0, a, 1, s, u, 1, vt, 1, w, 1, rw, iw, &info);
        chkxer("CGESDD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        cgesdd("N", 0, -1, a, 1, s, u, 1, vt, 1, w, 1, rw, iw, &info);
        chkxer("CGESDD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        cgesdd("N", 2, 1, a, 1, s, u, 1, vt, 1, w, 5, rw, iw, &info);
        chkxer("CGESDD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        cgesdd("A", 2, 1, a, 2, s, u, 1, vt, 1, w, 5, rw, iw, &info);
        chkxer("CGESDD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        cgesdd("A", 1, 2, a, 1, s, u, 1, vt, 1, w, 5, rw, iw, &info);
        chkxer("CGESDD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt - 2;

        /* Test CGEJSV */

        strcpy(xerbla_srnamt, "CGEJSV");
        xerbla_infot = 1;
        cgejsv("X", "U", "V", "R", "N", "N",
               0, 0, a, 1, s, u, 1, vt, 1,
               w, 1, rw, 1, iw, &info);
        chkxer("CGEJSV", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        cgejsv("G", "X", "V", "R", "N", "N",
               0, 0, a, 1, s, u, 1, vt, 1,
               w, 1, rw, 1, iw, &info);
        chkxer("CGEJSV", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        cgejsv("G", "U", "X", "R", "N", "N",
               0, 0, a, 1, s, u, 1, vt, 1,
               w, 1, rw, 1, iw, &info);
        chkxer("CGEJSV", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        cgejsv("G", "U", "V", "X", "N", "N",
               0, 0, a, 1, s, u, 1, vt, 1,
               w, 1, rw, 1, iw, &info);
        chkxer("CGEJSV", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        cgejsv("G", "U", "V", "R", "X", "N",
               0, 0, a, 1, s, u, 1, vt, 1,
               w, 1, rw, 1, iw, &info);
        chkxer("CGEJSV", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        cgejsv("G", "U", "V", "R", "N", "X",
               0, 0, a, 1, s, u, 1, vt, 1,
               w, 1, rw, 1, iw, &info);
        chkxer("CGEJSV", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        cgejsv("G", "U", "V", "R", "N", "N",
               -1, 0, a, 1, s, u, 1, vt, 1,
               w, 1, rw, 1, iw, &info);
        chkxer("CGEJSV", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        cgejsv("G", "U", "V", "R", "N", "N",
               0, -1, a, 1, s, u, 1, vt, 1,
               w, 1, rw, 1, iw, &info);
        chkxer("CGEJSV", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        cgejsv("G", "U", "V", "R", "N", "N",
               2, 1, a, 1, s, u, 1, vt, 1,
               w, 1, rw, 1, iw, &info);
        chkxer("CGEJSV", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 13;
        cgejsv("G", "U", "V", "R", "N", "N",
               2, 2, a, 2, s, u, 1, vt, 2,
               w, 1, rw, 1, iw, &info);
        chkxer("CGEJSV", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 15;
        cgejsv("G", "U", "V", "R", "N", "N",
               2, 2, a, 2, s, u, 2, vt, 1,
               w, 1, rw, 1, iw, &info);
        chkxer("CGEJSV", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = 11;

        /* Test CGESVDX */

        strcpy(xerbla_srnamt, "CGESVDX");
        xerbla_infot = 1;
        cgesvdx("X", "N", "A", 0, 0, a, 1, ZERO, ZERO,
                0, 0, &ns, s, u, 1, vt, 1, w, 1, rw, iw, &info);
        chkxer("CGESVDX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        cgesvdx("N", "X", "A", 0, 0, a, 1, ZERO, ZERO,
                0, 0, &ns, s, u, 1, vt, 1, w, 1, rw, iw, &info);
        chkxer("CGESVDX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        cgesvdx("N", "N", "X", 0, 0, a, 1, ZERO, ZERO,
                0, 0, &ns, s, u, 1, vt, 1, w, 1, rw, iw, &info);
        chkxer("CGESVDX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        cgesvdx("N", "N", "A", -1, 0, a, 1, ZERO, ZERO,
                0, 0, &ns, s, u, 1, vt, 1, w, 1, rw, iw, &info);
        chkxer("CGESVDX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        cgesvdx("N", "N", "A", 0, -1, a, 1, ZERO, ZERO,
                0, 0, &ns, s, u, 1, vt, 1, w, 1, rw, iw, &info);
        chkxer("CGESVDX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        cgesvdx("N", "N", "A", 2, 1, a, 1, ZERO, ZERO,
                0, 0, &ns, s, u, 1, vt, 1, w, 1, rw, iw, &info);
        chkxer("CGESVDX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        cgesvdx("N", "N", "V", 2, 1, a, 2, -ONE, ZERO,
                0, 0, &ns, s, u, 1, vt, 1, w, 1, rw, iw, &info);
        chkxer("CGESVDX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        cgesvdx("N", "N", "V", 2, 1, a, 2, ONE, ZERO,
                0, 0, &ns, s, u, 1, vt, 1, w, 1, rw, iw, &info);
        chkxer("CGESVDX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        cgesvdx("N", "N", "I", 2, 2, a, 2, ZERO, ZERO,
                -1, 0, &ns, s, u, 1, vt, 1, w, 1, rw, iw, &info);
        chkxer("CGESVDX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 11;
        cgesvdx("V", "N", "I", 2, 2, a, 2, ZERO, ZERO,
                0, -1, &ns, s, u, 1, vt, 1, w, 1, rw, iw, &info);
        chkxer("CGESVDX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 15;
        cgesvdx("V", "N", "A", 2, 2, a, 2, ZERO, ZERO,
                0, 0, &ns, s, u, 1, vt, 1, w, 1, rw, iw, &info);
        chkxer("CGESVDX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 17;
        cgesvdx("N", "V", "A", 2, 2, a, 2, ZERO, ZERO,
                0, 0, &ns, s, u, 1, vt, 1, w, 1, rw, iw, &info);
        chkxer("CGESVDX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = 12;

        /* Test CGESVDQ */

        strcpy(xerbla_srnamt, "CGESVDQ");
        xerbla_infot = 1;
        cgesvdq("X", "P", "T", "A", "A", 0, 0, a, 1, s, u,
                0, vt, 0, &ns, iw, 1, w, 1, rw, 1, &info);
        chkxer("CGESVDQ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        cgesvdq("A", "X", "T", "A", "A", 0, 0, a, 1, s, u,
                0, vt, 0, &ns, iw, 1, w, 1, rw, 1, &info);
        chkxer("CGESVDQ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        cgesvdq("A", "P", "X", "A", "A", 0, 0, a, 1, s, u,
                0, vt, 0, &ns, iw, 1, w, 1, rw, 1, &info);
        chkxer("CGESVDQ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        cgesvdq("A", "P", "T", "X", "A", 0, 0, a, 1, s, u,
                0, vt, 0, &ns, iw, 1, w, 1, rw, 1, &info);
        chkxer("CGESVDQ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        cgesvdq("A", "P", "T", "A", "X", 0, 0, a, 1, s, u,
                0, vt, 0, &ns, iw, 1, w, 1, rw, 1, &info);
        chkxer("CGESVDQ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        cgesvdq("A", "P", "T", "A", "A", -1, 0, a, 1, s, u,
                0, vt, 0, &ns, iw, 1, w, 1, rw, 1, &info);
        chkxer("CGESVDQ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        cgesvdq("A", "P", "T", "A", "A", 0, 1, a, 1, s, u,
                0, vt, 0, &ns, iw, 1, w, 1, rw, 1, &info);
        chkxer("CGESVDQ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        cgesvdq("A", "P", "T", "A", "A", 1, 1, a, 0, s, u,
                0, vt, 0, &ns, iw, 1, w, 1, rw, 1, &info);
        chkxer("CGESVDQ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 12;
        cgesvdq("A", "P", "T", "A", "A", 1, 1, a, 1, s, u,
                -1, vt, 0, &ns, iw, 1, w, 1, rw, 1, &info);
        chkxer("CGESVDQ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 14;
        cgesvdq("A", "P", "T", "A", "A", 1, 1, a, 1, s, u,
                1, vt, -1, &ns, iw, 1, w, 1, rw, 1, &info);
        chkxer("CGESVDQ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 17;
        cgesvdq("A", "P", "T", "A", "A", 1, 1, a, 1, s, u,
                1, vt, 1, &ns, iw, -5, w, 1, rw, 1, &info);
        chkxer("CGESVDQ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = 11;
    }

    *ok = xerbla_ok;

    xerbla_srnamt[0] = '\0';
    xerbla_infot = 0;
}
