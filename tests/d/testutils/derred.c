/**
 * @file derred.c
 * @brief DERRED tests the error exits for the eigenvalue driver routines.
 *
 * Port of LAPACK TESTING/EIG/derred.f
 * Tests error exits for: DGEEV, DGEES, DGEEVX, DGEESX, DGESVD, DGESDD,
 * DGEJSV, DGESVDX, DGESVDQ
 */

#include <string.h>
#include "semicolon_lapack_double.h"
#include "verify.h"

static INT dslect(const f64* zr, const f64* zi)
{
    (void)zi;
    return (*zr < 0.0);
}

void derred(const char* path, INT* ok, INT* nt)
{
    enum { NMAX = 4 };
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    char c2[2];
    INT i, ihi, ilo, info, j, ns, sdim;
    f64 abnrm;

    INT b[NMAX];
    INT iw[2 * NMAX];
    f64 a[NMAX * NMAX], r1[NMAX], r2[NMAX], s[NMAX], u[NMAX * NMAX];
    f64 vl[NMAX * NMAX], vr[NMAX * NMAX], vt[NMAX * NMAX];
    f64 w[10 * NMAX], wi[NMAX], wr[NMAX], rw[10 * NMAX];

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

        /* Test DGEEV */

        strcpy(xerbla_srnamt, "DGEEV ");
        xerbla_infot = 1;
        dgeev("X", "N", 0, a, 1, wr, wi, vl, 1, vr, 1, w, 1, &info);
        chkxer("DGEEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        dgeev("N", "X", 0, a, 1, wr, wi, vl, 1, vr, 1, w, 1, &info);
        chkxer("DGEEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        dgeev("N", "N", -1, a, 1, wr, wi, vl, 1, vr, 1, w, 1, &info);
        chkxer("DGEEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        dgeev("N", "N", 2, a, 1, wr, wi, vl, 1, vr, 1, w, 6, &info);
        chkxer("DGEEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        dgeev("V", "N", 2, a, 2, wr, wi, vl, 1, vr, 1, w, 8, &info);
        chkxer("DGEEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 11;
        dgeev("N", "V", 2, a, 2, wr, wi, vl, 1, vr, 1, w, 8, &info);
        chkxer("DGEEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 13;
        dgeev("V", "V", 1, a, 1, wr, wi, vl, 1, vr, 1, w, 3, &info);
        chkxer("DGEEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 7;

    } else if ((c2[0] == 'E' || c2[0] == 'e') &&
               (c2[1] == 'S' || c2[1] == 's')) {

        /* Test DGEES */

        strcpy(xerbla_srnamt, "DGEES ");
        xerbla_infot = 1;
        dgees("X", "N", dslect, 0, a, 1, &sdim, wr, wi, vl, 1, w,
              1, b, &info);
        chkxer("DGEES ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        dgees("N", "X", dslect, 0, a, 1, &sdim, wr, wi, vl, 1, w,
              1, b, &info);
        chkxer("DGEES ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        dgees("N", "S", dslect, -1, a, 1, &sdim, wr, wi, vl, 1, w,
              1, b, &info);
        chkxer("DGEES ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        dgees("N", "S", dslect, 2, a, 1, &sdim, wr, wi, vl, 1, w,
              6, b, &info);
        chkxer("DGEES ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 11;
        dgees("V", "S", dslect, 2, a, 2, &sdim, wr, wi, vl, 1, w,
              6, b, &info);
        chkxer("DGEES ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 13;
        dgees("N", "S", dslect, 1, a, 1, &sdim, wr, wi, vl, 1, w,
              2, b, &info);
        chkxer("DGEES ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 6;

    } else if ((c2[0] == 'V' || c2[0] == 'v') &&
               (c2[1] == 'X' || c2[1] == 'x')) {

        /* Test DGEEVX */

        strcpy(xerbla_srnamt, "DGEEVX");
        xerbla_infot = 1;
        dgeevx("X", "N", "N", "N", 0, a, 1, wr, wi, vl, 1, vr, 1,
               &ilo, &ihi, s, &abnrm, r1, r2, w, 1, iw, &info);
        chkxer("DGEEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        dgeevx("N", "X", "N", "N", 0, a, 1, wr, wi, vl, 1, vr, 1,
               &ilo, &ihi, s, &abnrm, r1, r2, w, 1, iw, &info);
        chkxer("DGEEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        dgeevx("N", "N", "X", "N", 0, a, 1, wr, wi, vl, 1, vr, 1,
               &ilo, &ihi, s, &abnrm, r1, r2, w, 1, iw, &info);
        chkxer("DGEEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        dgeevx("N", "N", "N", "X", 0, a, 1, wr, wi, vl, 1, vr, 1,
               &ilo, &ihi, s, &abnrm, r1, r2, w, 1, iw, &info);
        chkxer("DGEEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        dgeevx("N", "N", "N", "N", -1, a, 1, wr, wi, vl, 1, vr,
               1, &ilo, &ihi, s, &abnrm, r1, r2, w, 1, iw, &info);
        chkxer("DGEEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        dgeevx("N", "N", "N", "N", 2, a, 1, wr, wi, vl, 1, vr, 1,
               &ilo, &ihi, s, &abnrm, r1, r2, w, 1, iw, &info);
        chkxer("DGEEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 11;
        dgeevx("N", "V", "N", "N", 2, a, 2, wr, wi, vl, 1, vr, 1,
               &ilo, &ihi, s, &abnrm, r1, r2, w, 6, iw, &info);
        chkxer("DGEEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 13;
        dgeevx("N", "N", "V", "N", 2, a, 2, wr, wi, vl, 1, vr, 1,
               &ilo, &ihi, s, &abnrm, r1, r2, w, 6, iw, &info);
        chkxer("DGEEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 21;
        dgeevx("N", "N", "N", "N", 1, a, 1, wr, wi, vl, 1, vr, 1,
               &ilo, &ihi, s, &abnrm, r1, r2, w, 1, iw, &info);
        chkxer("DGEEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 21;
        dgeevx("N", "V", "N", "N", 1, a, 1, wr, wi, vl, 1, vr, 1,
               &ilo, &ihi, s, &abnrm, r1, r2, w, 2, iw, &info);
        chkxer("DGEEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 21;
        dgeevx("N", "N", "V", "V", 1, a, 1, wr, wi, vl, 1, vr, 1,
               &ilo, &ihi, s, &abnrm, r1, r2, w, 3, iw, &info);
        chkxer("DGEEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 11;

    } else if ((c2[0] == 'S' || c2[0] == 's') &&
               (c2[1] == 'X' || c2[1] == 'x')) {

        /* Test DGEESX */

        strcpy(xerbla_srnamt, "DGEESX");
        xerbla_infot = 1;
        dgeesx("X", "N", dslect, "N", 0, a, 1, &sdim, wr, wi, vl,
               1, &r1[0], &r2[0], w, 1, iw, 1, b, &info);
        chkxer("DGEESX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        dgeesx("N", "X", dslect, "N", 0, a, 1, &sdim, wr, wi, vl,
               1, &r1[0], &r2[0], w, 1, iw, 1, b, &info);
        chkxer("DGEESX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        dgeesx("N", "N", dslect, "X", 0, a, 1, &sdim, wr, wi, vl,
               1, &r1[0], &r2[0], w, 1, iw, 1, b, &info);
        chkxer("DGEESX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        dgeesx("N", "N", dslect, "N", -1, a, 1, &sdim, wr, wi, vl,
               1, &r1[0], &r2[0], w, 1, iw, 1, b, &info);
        chkxer("DGEESX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        dgeesx("N", "N", dslect, "N", 2, a, 1, &sdim, wr, wi, vl,
               1, &r1[0], &r2[0], w, 6, iw, 1, b, &info);
        chkxer("DGEESX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 12;
        dgeesx("V", "N", dslect, "N", 2, a, 2, &sdim, wr, wi, vl,
               1, &r1[0], &r2[0], w, 6, iw, 1, b, &info);
        chkxer("DGEESX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 16;
        dgeesx("N", "N", dslect, "N", 1, a, 1, &sdim, wr, wi, vl,
               1, &r1[0], &r2[0], w, 2, iw, 1, b, &info);
        chkxer("DGEESX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 7;

    } else if ((c2[0] == 'B' || c2[0] == 'b') &&
               (c2[1] == 'D' || c2[1] == 'd')) {

        /* Test DGESVD */

        strcpy(xerbla_srnamt, "DGESVD");
        xerbla_infot = 1;
        dgesvd("X", "N", 0, 0, a, 1, s, u, 1, vt, 1, w, 1, &info);
        chkxer("DGESVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        dgesvd("N", "X", 0, 0, a, 1, s, u, 1, vt, 1, w, 1, &info);
        chkxer("DGESVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        dgesvd("O", "O", 0, 0, a, 1, s, u, 1, vt, 1, w, 1, &info);
        chkxer("DGESVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        dgesvd("N", "N", -1, 0, a, 1, s, u, 1, vt, 1, w, 1,
               &info);
        chkxer("DGESVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        dgesvd("N", "N", 0, -1, a, 1, s, u, 1, vt, 1, w, 1,
               &info);
        chkxer("DGESVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        dgesvd("N", "N", 2, 1, a, 1, s, u, 1, vt, 1, w, 5, &info);
        chkxer("DGESVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        dgesvd("A", "N", 2, 1, a, 2, s, u, 1, vt, 1, w, 5, &info);
        chkxer("DGESVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 11;
        dgesvd("N", "A", 1, 2, a, 1, s, u, 1, vt, 1, w, 5, &info);
        chkxer("DGESVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = 8;

        /* Test DGESDD */

        strcpy(xerbla_srnamt, "DGESDD");
        xerbla_infot = 1;
        dgesdd("X", 0, 0, a, 1, s, u, 1, vt, 1, w, 1, iw, &info);
        chkxer("DGESDD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        dgesdd("N", -1, 0, a, 1, s, u, 1, vt, 1, w, 1, iw, &info);
        chkxer("DGESDD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        dgesdd("N", 0, -1, a, 1, s, u, 1, vt, 1, w, 1, iw, &info);
        chkxer("DGESDD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        dgesdd("N", 2, 1, a, 1, s, u, 1, vt, 1, w, 5, iw, &info);
        chkxer("DGESDD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        dgesdd("A", 2, 1, a, 2, s, u, 1, vt, 1, w, 5, iw, &info);
        chkxer("DGESDD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        dgesdd("A", 1, 2, a, 1, s, u, 1, vt, 1, w, 5, iw, &info);
        chkxer("DGESDD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = 6;

        /* Test DGEJSV */

        strcpy(xerbla_srnamt, "DGEJSV");
        xerbla_infot = 1;
        dgejsv("X", "U", "V", "R", "N", "N",
               0, 0, a, 1, s, u, 1, vt, 1,
               w, 1, iw, &info);
        chkxer("DGEJSV", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        dgejsv("G", "X", "V", "R", "N", "N",
               0, 0, a, 1, s, u, 1, vt, 1,
               w, 1, iw, &info);
        chkxer("DGEJSV", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        dgejsv("G", "U", "X", "R", "N", "N",
               0, 0, a, 1, s, u, 1, vt, 1,
               w, 1, iw, &info);
        chkxer("DGEJSV", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        dgejsv("G", "U", "V", "X", "N", "N",
               0, 0, a, 1, s, u, 1, vt, 1,
               w, 1, iw, &info);
        chkxer("DGEJSV", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        dgejsv("G", "U", "V", "R", "X", "N",
               0, 0, a, 1, s, u, 1, vt, 1,
               w, 1, iw, &info);
        chkxer("DGEJSV", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        dgejsv("G", "U", "V", "R", "N", "X",
               0, 0, a, 1, s, u, 1, vt, 1,
               w, 1, iw, &info);
        chkxer("DGEJSV", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        dgejsv("G", "U", "V", "R", "N", "N",
               -1, 0, a, 1, s, u, 1, vt, 1,
               w, 1, iw, &info);
        chkxer("DGEJSV", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        dgejsv("G", "U", "V", "R", "N", "N",
               0, -1, a, 1, s, u, 1, vt, 1,
               w, 1, iw, &info);
        chkxer("DGEJSV", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        dgejsv("G", "U", "V", "R", "N", "N",
               2, 1, a, 1, s, u, 1, vt, 1,
               w, 1, iw, &info);
        chkxer("DGEJSV", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 13;
        dgejsv("G", "U", "V", "R", "N", "N",
               2, 2, a, 2, s, u, 1, vt, 2,
               w, 1, iw, &info);
        chkxer("DGEJSV", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 15;
        dgejsv("G", "U", "V", "R", "N", "N",
               2, 2, a, 2, s, u, 2, vt, 1,
               w, 1, iw, &info);
        chkxer("DGEJSV", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = 11;

        /* Test DGESVDX */

        strcpy(xerbla_srnamt, "DGESVDX");
        xerbla_infot = 1;
        dgesvdx("X", "N", "A", 0, 0, a, 1, ZERO, ZERO,
                0, 0, &ns, s, u, 1, vt, 1, w, 1, iw, &info);
        chkxer("DGESVDX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        dgesvdx("N", "X", "A", 0, 0, a, 1, ZERO, ZERO,
                0, 0, &ns, s, u, 1, vt, 1, w, 1, iw, &info);
        chkxer("DGESVDX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        dgesvdx("N", "N", "X", 0, 0, a, 1, ZERO, ZERO,
                0, 0, &ns, s, u, 1, vt, 1, w, 1, iw, &info);
        chkxer("DGESVDX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        dgesvdx("N", "N", "A", -1, 0, a, 1, ZERO, ZERO,
                0, 0, &ns, s, u, 1, vt, 1, w, 1, iw, &info);
        chkxer("DGESVDX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        dgesvdx("N", "N", "A", 0, -1, a, 1, ZERO, ZERO,
                0, 0, &ns, s, u, 1, vt, 1, w, 1, iw, &info);
        chkxer("DGESVDX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        dgesvdx("N", "N", "A", 2, 1, a, 1, ZERO, ZERO,
                0, 0, &ns, s, u, 1, vt, 1, w, 1, iw, &info);
        chkxer("DGESVDX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        dgesvdx("N", "N", "V", 2, 1, a, 2, -ONE, ZERO,
                0, 0, &ns, s, u, 1, vt, 1, w, 1, iw, &info);
        chkxer("DGESVDX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        dgesvdx("N", "N", "V", 2, 1, a, 2, ONE, ZERO,
                0, 0, &ns, s, u, 1, vt, 1, w, 1, iw, &info);
        chkxer("DGESVDX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        dgesvdx("N", "N", "I", 2, 2, a, 2, ZERO, ZERO,
                -1, 0, &ns, s, u, 1, vt, 1, w, 1, iw, &info);
        chkxer("DGESVDX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 11;
        dgesvdx("V", "N", "I", 2, 2, a, 2, ZERO, ZERO,
                0, -1, &ns, s, u, 1, vt, 1, w, 1, iw, &info);
        chkxer("DGESVDX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 15;
        dgesvdx("V", "N", "A", 2, 2, a, 2, ZERO, ZERO,
                0, 0, &ns, s, u, 1, vt, 1, w, 1, iw, &info);
        chkxer("DGESVDX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 17;
        dgesvdx("N", "V", "A", 2, 2, a, 2, ZERO, ZERO,
                0, 0, &ns, s, u, 1, vt, 1, w, 1, iw, &info);
        chkxer("DGESVDX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = 12;

        /* Test DGESVDQ */

        strcpy(xerbla_srnamt, "DGESVDQ");
        xerbla_infot = 1;
        dgesvdq("X", "P", "T", "A", "A", 0, 0, a, 1, s, u,
                0, vt, 0, &ns, iw, 1, w, 1, rw, 1, &info);
        chkxer("DGESVDQ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        dgesvdq("A", "X", "T", "A", "A", 0, 0, a, 1, s, u,
                0, vt, 0, &ns, iw, 1, w, 1, rw, 1, &info);
        chkxer("DGESVDQ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        dgesvdq("A", "P", "X", "A", "A", 0, 0, a, 1, s, u,
                0, vt, 0, &ns, iw, 1, w, 1, rw, 1, &info);
        chkxer("DGESVDQ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        dgesvdq("A", "P", "T", "X", "A", 0, 0, a, 1, s, u,
                0, vt, 0, &ns, iw, 1, w, 1, rw, 1, &info);
        chkxer("DGESVDQ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        dgesvdq("A", "P", "T", "A", "X", 0, 0, a, 1, s, u,
                0, vt, 0, &ns, iw, 1, w, 1, rw, 1, &info);
        chkxer("DGESVDQ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        dgesvdq("A", "P", "T", "A", "A", -1, 0, a, 1, s, u,
                0, vt, 0, &ns, iw, 1, w, 1, rw, 1, &info);
        chkxer("DGESVDQ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        dgesvdq("A", "P", "T", "A", "A", 0, 1, a, 1, s, u,
                0, vt, 0, &ns, iw, 1, w, 1, rw, 1, &info);
        chkxer("DGESVDQ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        dgesvdq("A", "P", "T", "A", "A", 1, 1, a, 0, s, u,
                0, vt, 0, &ns, iw, 1, w, 1, rw, 1, &info);
        chkxer("DGESVDQ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 12;
        dgesvdq("A", "P", "T", "A", "A", 1, 1, a, 1, s, u,
                -1, vt, 0, &ns, iw, 1, w, 1, rw, 1, &info);
        chkxer("DGESVDQ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 14;
        dgesvdq("A", "P", "T", "A", "A", 1, 1, a, 1, s, u,
                1, vt, -1, &ns, iw, 1, w, 1, rw, 1, &info);
        chkxer("DGESVDQ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 17;
        dgesvdq("A", "P", "T", "A", "A", 1, 1, a, 1, s, u,
                1, vt, 1, &ns, iw, -5, w, 1, rw, 1, &info);
        chkxer("DGESVDQ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = 11;
    }

    *ok = xerbla_ok;

    xerbla_srnamt[0] = '\0';
    xerbla_infot = 0;
}
