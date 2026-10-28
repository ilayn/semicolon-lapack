/**
 * @file zerred.c
 * @brief ZERRED tests the error exits for the eigenvalue driver routines.
 *
 * Port of LAPACK TESTING/EIG/zerred.f
 * Tests error exits for: ZGEEV, ZGEES, ZGEEVX, ZGEESX, ZGESVD, ZGESDD,
 * ZGEJSV, ZGESVDX, ZGESVDQ
 */

#include <complex.h>
#include <math.h>
#include <string.h>
#include "semicolon_lapack_complex_double.h"
#include "verify.h"

static INT selopt;
static INT seldim;
static INT selval[20];
static f64 selwr[20];
static f64 selwi[20];

/**
 * ZSLECT returns .TRUE. if the eigenvalue Z is to be selected,
 * otherwise it returns .FALSE.
 * It is used by ZCHK41 to test if ZGEES successfully sorts eigenvalues,
 * and by ZCHK43 to test if ZGEESX successfully sorts eigenvalues.
 *
 * The common block /SSLCT/ controls how eigenvalues are selected.
 * If SELOPT = 0, then ZSLECT return .TRUE. when real(Z) is less than
 * zero, and .FALSE. otherwise.
 * If SELOPT is at least 1, ZSLECT returns SELVAL(SELOPT) and adds 1
 * to SELOPT, cycling back to 1 at SELMAX.
 */
static INT zslect(const c128* z)
{
    const f64 ZERO = 0.0;
    INT i, val;
    f64 rmin, x;

    if (selopt == 0) {
        val = (creal(*z) < ZERO);
    } else {
        rmin = cabs(*z - CMPLX(selwr[0], selwi[0]));
        val = selval[0];
        for (i = 1; i < seldim; i++) {
            x = cabs(*z - CMPLX(selwr[i], selwi[i]));
            if (x <= rmin) {
                rmin = x;
                val = selval[i];
            }
        }
    }
    return val;
}

void zerred(const char* path, INT* ok, INT* nt)
{
    enum { NMAX = 4, LW = 5 * NMAX };
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    INT i, ihi, ilo, info, j, ns, sdim;
    f64 abnrm;

    INT b[NMAX];
    INT iw[4 * NMAX];
    f64 r1[NMAX], r2[NMAX], rw[LW], s[NMAX];
    c128 a[NMAX * NMAX], u[NMAX * NMAX], vl[NMAX * NMAX], vr[NMAX * NMAX];
    c128 vt[NMAX * NMAX], w[10 * NMAX], x[NMAX];

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

        /* Test ZGEEV */

        strcpy(xerbla_srnamt, "ZGEEV ");
        xerbla_infot = 1;
        zgeev("X", "N", 0, a, 1, x, vl, 1, vr, 1, w, 1, rw, &info);
        chkxer("ZGEEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        zgeev("N", "X", 0, a, 1, x, vl, 1, vr, 1, w, 1, rw, &info);
        chkxer("ZGEEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        zgeev("N", "N", -1, a, 1, x, vl, 1, vr, 1, w, 1, rw, &info);
        chkxer("ZGEEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        zgeev("N", "N", 2, a, 1, x, vl, 1, vr, 1, w, 4, rw, &info);
        chkxer("ZGEEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        zgeev("V", "N", 2, a, 2, x, vl, 1, vr, 1, w, 4, rw, &info);
        chkxer("ZGEEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        zgeev("N", "V", 2, a, 2, x, vl, 1, vr, 1, w, 4, rw, &info);
        chkxer("ZGEEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 12;
        zgeev("V", "V", 1, a, 1, x, vl, 1, vr, 1, w, 1, rw, &info);
        chkxer("ZGEEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 7;

    } else if ((c2[0] == 'E' || c2[0] == 'e') &&
               (c2[1] == 'S' || c2[1] == 's')) {

        /* Test ZGEES */

        strcpy(xerbla_srnamt, "ZGEES ");
        xerbla_infot = 1;
        zgees("X", "N", zslect, 0, a, 1, &sdim, x, vl, 1, w, 1,
              rw, b, &info);
        chkxer("ZGEES ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        zgees("N", "X", zslect, 0, a, 1, &sdim, x, vl, 1, w, 1,
              rw, b, &info);
        chkxer("ZGEES ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        zgees("N", "S", zslect, -1, a, 1, &sdim, x, vl, 1, w, 1,
              rw, b, &info);
        chkxer("ZGEES ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        zgees("N", "S", zslect, 2, a, 1, &sdim, x, vl, 1, w, 4,
              rw, b, &info);
        chkxer("ZGEES ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        zgees("V", "S", zslect, 2, a, 2, &sdim, x, vl, 1, w, 4,
              rw, b, &info);
        chkxer("ZGEES ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 12;
        zgees("N", "S", zslect, 1, a, 1, &sdim, x, vl, 1, w, 1,
              rw, b, &info);
        chkxer("ZGEES ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 6;

    } else if ((c2[0] == 'V' || c2[0] == 'v') &&
               (c2[1] == 'X' || c2[1] == 'x')) {

        /* Test ZGEEVX */

        strcpy(xerbla_srnamt, "ZGEEVX");
        xerbla_infot = 1;
        zgeevx("X", "N", "N", "N", 0, a, 1, x, vl, 1, vr, 1, &ilo,
               &ihi, s, &abnrm, r1, r2, w, 1, rw, &info);
        chkxer("ZGEEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        zgeevx("N", "X", "N", "N", 0, a, 1, x, vl, 1, vr, 1, &ilo,
               &ihi, s, &abnrm, r1, r2, w, 1, rw, &info);
        chkxer("ZGEEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        zgeevx("N", "N", "X", "N", 0, a, 1, x, vl, 1, vr, 1, &ilo,
               &ihi, s, &abnrm, r1, r2, w, 1, rw, &info);
        chkxer("ZGEEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        zgeevx("N", "N", "N", "X", 0, a, 1, x, vl, 1, vr, 1, &ilo,
               &ihi, s, &abnrm, r1, r2, w, 1, rw, &info);
        chkxer("ZGEEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        zgeevx("N", "N", "N", "N", -1, a, 1, x, vl, 1, vr, 1,
               &ilo, &ihi, s, &abnrm, r1, r2, w, 1, rw, &info);
        chkxer("ZGEEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        zgeevx("N", "N", "N", "N", 2, a, 1, x, vl, 1, vr, 1, &ilo,
               &ihi, s, &abnrm, r1, r2, w, 4, rw, &info);
        chkxer("ZGEEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        zgeevx("N", "V", "N", "N", 2, a, 2, x, vl, 1, vr, 1, &ilo,
               &ihi, s, &abnrm, r1, r2, w, 4, rw, &info);
        chkxer("ZGEEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 12;
        zgeevx("N", "N", "V", "N", 2, a, 2, x, vl, 1, vr, 1, &ilo,
               &ihi, s, &abnrm, r1, r2, w, 4, rw, &info);
        chkxer("ZGEEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 20;
        zgeevx("N", "N", "N", "N", 1, a, 1, x, vl, 1, vr, 1, &ilo,
               &ihi, s, &abnrm, r1, r2, w, 1, rw, &info);
        chkxer("ZGEEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 20;
        zgeevx("N", "N", "V", "V", 1, a, 1, x, vl, 1, vr, 1, &ilo,
               &ihi, s, &abnrm, r1, r2, w, 2, rw, &info);
        chkxer("ZGEEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 10;

    } else if ((c2[0] == 'S' || c2[0] == 's') &&
               (c2[1] == 'X' || c2[1] == 'x')) {

        /* Test ZGEESX */

        strcpy(xerbla_srnamt, "ZGEESX");
        xerbla_infot = 1;
        zgeesx("X", "N", zslect, "N", 0, a, 1, &sdim, x, vl, 1,
               &r1[0], &r2[0], w, 1, rw, b, &info);
        chkxer("ZGEESX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        zgeesx("N", "X", zslect, "N", 0, a, 1, &sdim, x, vl, 1,
               &r1[0], &r2[0], w, 1, rw, b, &info);
        chkxer("ZGEESX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        zgeesx("N", "N", zslect, "X", 0, a, 1, &sdim, x, vl, 1,
               &r1[0], &r2[0], w, 1, rw, b, &info);
        chkxer("ZGEESX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        zgeesx("N", "N", zslect, "N", -1, a, 1, &sdim, x, vl, 1,
               &r1[0], &r2[0], w, 1, rw, b, &info);
        chkxer("ZGEESX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        zgeesx("N", "N", zslect, "N", 2, a, 1, &sdim, x, vl, 1,
               &r1[0], &r2[0], w, 4, rw, b, &info);
        chkxer("ZGEESX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 11;
        zgeesx("V", "N", zslect, "N", 2, a, 2, &sdim, x, vl, 1,
               &r1[0], &r2[0], w, 4, rw, b, &info);
        chkxer("ZGEESX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 15;
        zgeesx("N", "N", zslect, "N", 1, a, 1, &sdim, x, vl, 1,
               &r1[0], &r2[0], w, 1, rw, b, &info);
        chkxer("ZGEESX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 7;

    } else if ((c2[0] == 'B' || c2[0] == 'b') &&
               (c2[1] == 'D' || c2[1] == 'd')) {

        /* Test ZGESVD */

        strcpy(xerbla_srnamt, "ZGESVD");
        xerbla_infot = 1;
        zgesvd("X", "N", 0, 0, a, 1, s, u, 1, vt, 1, w, 1, rw, &info);
        chkxer("ZGESVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        zgesvd("N", "X", 0, 0, a, 1, s, u, 1, vt, 1, w, 1, rw, &info);
        chkxer("ZGESVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        zgesvd("O", "O", 0, 0, a, 1, s, u, 1, vt, 1, w, 1, rw, &info);
        chkxer("ZGESVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        zgesvd("N", "N", -1, 0, a, 1, s, u, 1, vt, 1, w, 1, rw, &info);
        chkxer("ZGESVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        zgesvd("N", "N", 0, -1, a, 1, s, u, 1, vt, 1, w, 1, rw, &info);
        chkxer("ZGESVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        zgesvd("N", "N", 2, 1, a, 1, s, u, 1, vt, 1, w, 5, rw, &info);
        chkxer("ZGESVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        zgesvd("A", "N", 2, 1, a, 2, s, u, 1, vt, 1, w, 5, rw, &info);
        chkxer("ZGESVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 11;
        zgesvd("N", "A", 1, 2, a, 1, s, u, 1, vt, 1, w, 5, rw, &info);
        chkxer("ZGESVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 8;

        /* Test ZGESDD */

        strcpy(xerbla_srnamt, "ZGESDD");
        xerbla_infot = 1;
        zgesdd("X", 0, 0, a, 1, s, u, 1, vt, 1, w, 1, rw, iw, &info);
        chkxer("ZGESDD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        zgesdd("N", -1, 0, a, 1, s, u, 1, vt, 1, w, 1, rw, iw, &info);
        chkxer("ZGESDD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        zgesdd("N", 0, -1, a, 1, s, u, 1, vt, 1, w, 1, rw, iw, &info);
        chkxer("ZGESDD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        zgesdd("N", 2, 1, a, 1, s, u, 1, vt, 1, w, 5, rw, iw, &info);
        chkxer("ZGESDD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        zgesdd("A", 2, 1, a, 2, s, u, 1, vt, 1, w, 5, rw, iw, &info);
        chkxer("ZGESDD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        zgesdd("A", 1, 2, a, 1, s, u, 1, vt, 1, w, 5, rw, iw, &info);
        chkxer("ZGESDD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt - 2;

        /* Test ZGEJSV */

        strcpy(xerbla_srnamt, "ZGEJSV");
        xerbla_infot = 1;
        zgejsv("X", "U", "V", "R", "N", "N",
               0, 0, a, 1, s, u, 1, vt, 1,
               w, 1, rw, 1, iw, &info);
        chkxer("ZGEJSV", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        zgejsv("G", "X", "V", "R", "N", "N",
               0, 0, a, 1, s, u, 1, vt, 1,
               w, 1, rw, 1, iw, &info);
        chkxer("ZGEJSV", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        zgejsv("G", "U", "X", "R", "N", "N",
               0, 0, a, 1, s, u, 1, vt, 1,
               w, 1, rw, 1, iw, &info);
        chkxer("ZGEJSV", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        zgejsv("G", "U", "V", "X", "N", "N",
               0, 0, a, 1, s, u, 1, vt, 1,
               w, 1, rw, 1, iw, &info);
        chkxer("ZGEJSV", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        zgejsv("G", "U", "V", "R", "X", "N",
               0, 0, a, 1, s, u, 1, vt, 1,
               w, 1, rw, 1, iw, &info);
        chkxer("ZGEJSV", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        zgejsv("G", "U", "V", "R", "N", "X",
               0, 0, a, 1, s, u, 1, vt, 1,
               w, 1, rw, 1, iw, &info);
        chkxer("ZGEJSV", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        zgejsv("G", "U", "V", "R", "N", "N",
               -1, 0, a, 1, s, u, 1, vt, 1,
               w, 1, rw, 1, iw, &info);
        chkxer("ZGEJSV", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        zgejsv("G", "U", "V", "R", "N", "N",
               0, -1, a, 1, s, u, 1, vt, 1,
               w, 1, rw, 1, iw, &info);
        chkxer("ZGEJSV", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        zgejsv("G", "U", "V", "R", "N", "N",
               2, 1, a, 1, s, u, 1, vt, 1,
               w, 1, rw, 1, iw, &info);
        chkxer("ZGEJSV", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 13;
        zgejsv("G", "U", "V", "R", "N", "N",
               2, 2, a, 2, s, u, 1, vt, 2,
               w, 1, rw, 1, iw, &info);
        chkxer("ZGEJSV", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 15;
        zgejsv("G", "U", "V", "R", "N", "N",
               2, 2, a, 2, s, u, 2, vt, 1,
               w, 1, rw, 1, iw, &info);
        chkxer("ZGEJSV", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = 11;

        /* Test ZGESVDX */

        strcpy(xerbla_srnamt, "ZGESVDX");
        xerbla_infot = 1;
        zgesvdx("X", "N", "A", 0, 0, a, 1, ZERO, ZERO,
                0, 0, &ns, s, u, 1, vt, 1, w, 1, rw, iw, &info);
        chkxer("ZGESVDX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        zgesvdx("N", "X", "A", 0, 0, a, 1, ZERO, ZERO,
                0, 0, &ns, s, u, 1, vt, 1, w, 1, rw, iw, &info);
        chkxer("ZGESVDX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        zgesvdx("N", "N", "X", 0, 0, a, 1, ZERO, ZERO,
                0, 0, &ns, s, u, 1, vt, 1, w, 1, rw, iw, &info);
        chkxer("ZGESVDX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        zgesvdx("N", "N", "A", -1, 0, a, 1, ZERO, ZERO,
                0, 0, &ns, s, u, 1, vt, 1, w, 1, rw, iw, &info);
        chkxer("ZGESVDX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        zgesvdx("N", "N", "A", 0, -1, a, 1, ZERO, ZERO,
                0, 0, &ns, s, u, 1, vt, 1, w, 1, rw, iw, &info);
        chkxer("ZGESVDX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        zgesvdx("N", "N", "A", 2, 1, a, 1, ZERO, ZERO,
                0, 0, &ns, s, u, 1, vt, 1, w, 1, rw, iw, &info);
        chkxer("ZGESVDX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        zgesvdx("N", "N", "V", 2, 1, a, 2, -ONE, ZERO,
                0, 0, &ns, s, u, 1, vt, 1, w, 1, rw, iw, &info);
        chkxer("ZGESVDX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        zgesvdx("N", "N", "V", 2, 1, a, 2, ONE, ZERO,
                0, 0, &ns, s, u, 1, vt, 1, w, 1, rw, iw, &info);
        chkxer("ZGESVDX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        zgesvdx("N", "N", "I", 2, 2, a, 2, ZERO, ZERO,
                -1, 0, &ns, s, u, 1, vt, 1, w, 1, rw, iw, &info);
        chkxer("ZGESVDX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 11;
        zgesvdx("V", "N", "I", 2, 2, a, 2, ZERO, ZERO,
                0, -1, &ns, s, u, 1, vt, 1, w, 1, rw, iw, &info);
        chkxer("ZGESVDX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 15;
        zgesvdx("V", "N", "A", 2, 2, a, 2, ZERO, ZERO,
                0, 0, &ns, s, u, 1, vt, 1, w, 1, rw, iw, &info);
        chkxer("ZGESVDX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 17;
        zgesvdx("N", "V", "A", 2, 2, a, 2, ZERO, ZERO,
                0, 0, &ns, s, u, 1, vt, 1, w, 1, rw, iw, &info);
        chkxer("ZGESVDX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = 12;

        /* Test ZGESVDQ */

        strcpy(xerbla_srnamt, "ZGESVDQ");
        xerbla_infot = 1;
        zgesvdq("X", "P", "T", "A", "A", 0, 0, a, 1, s, u,
                0, vt, 0, &ns, iw, 1, w, 1, rw, 1, &info);
        chkxer("ZGESVDQ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        zgesvdq("A", "X", "T", "A", "A", 0, 0, a, 1, s, u,
                0, vt, 0, &ns, iw, 1, w, 1, rw, 1, &info);
        chkxer("ZGESVDQ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        zgesvdq("A", "P", "X", "A", "A", 0, 0, a, 1, s, u,
                0, vt, 0, &ns, iw, 1, w, 1, rw, 1, &info);
        chkxer("ZGESVDQ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        zgesvdq("A", "P", "T", "X", "A", 0, 0, a, 1, s, u,
                0, vt, 0, &ns, iw, 1, w, 1, rw, 1, &info);
        chkxer("ZGESVDQ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        zgesvdq("A", "P", "T", "A", "X", 0, 0, a, 1, s, u,
                0, vt, 0, &ns, iw, 1, w, 1, rw, 1, &info);
        chkxer("ZGESVDQ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        zgesvdq("A", "P", "T", "A", "A", -1, 0, a, 1, s, u,
                0, vt, 0, &ns, iw, 1, w, 1, rw, 1, &info);
        chkxer("ZGESVDQ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        zgesvdq("A", "P", "T", "A", "A", 0, 1, a, 1, s, u,
                0, vt, 0, &ns, iw, 1, w, 1, rw, 1, &info);
        chkxer("ZGESVDQ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        zgesvdq("A", "P", "T", "A", "A", 1, 1, a, 0, s, u,
                0, vt, 0, &ns, iw, 1, w, 1, rw, 1, &info);
        chkxer("ZGESVDQ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 12;
        zgesvdq("A", "P", "T", "A", "A", 1, 1, a, 1, s, u,
                -1, vt, 0, &ns, iw, 1, w, 1, rw, 1, &info);
        chkxer("ZGESVDQ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 14;
        zgesvdq("A", "P", "T", "A", "A", 1, 1, a, 1, s, u,
                1, vt, -1, &ns, iw, 1, w, 1, rw, 1, &info);
        chkxer("ZGESVDQ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 17;
        zgesvdq("A", "P", "T", "A", "A", 1, 1, a, 1, s, u,
                1, vt, 1, &ns, iw, -5, w, 1, rw, 1, &info);
        chkxer("ZGESVDQ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = 11;
    }

    *ok = xerbla_ok;

    xerbla_srnamt[0] = '\0';
    xerbla_infot = 0;
}
