/**
 * @file serred.c
 * @brief SERRED tests the error exits for the eigenvalue driver routines.
 *
 * Port of LAPACK TESTING/EIG/serred.f
 * Tests error exits for: SGEEV, SGEES, SGEEVX, SGEESX, SGESVD, SGESDD,
 * SGEJSV, SGESVDX, SGESVDQ
 */

#include <string.h>
#include "semicolon_lapack_single.h"
#include "verify.h"

static INT sslect(const f32* zr, const f32* zi)
{
    (void)zi;
    return (*zr < 0.0f);
}

void serred(const char* path, INT* ok, INT* nt)
{
    enum { NMAX = 4 };
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;

    char c2[2];
    INT i, ihi, ilo, info, j, ns, sdim;
    f32 abnrm;

    INT b[NMAX];
    INT iw[2 * NMAX];
    f32 a[NMAX * NMAX], r1[NMAX], r2[NMAX], s[NMAX], u[NMAX * NMAX];
    f32 vl[NMAX * NMAX], vr[NMAX * NMAX], vt[NMAX * NMAX];
    f32 w[10 * NMAX], wi[NMAX], wr[NMAX], rw[10 * NMAX];

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

        /* Test SGEEV */

        strcpy(xerbla_srnamt, "SGEEV ");
        xerbla_infot = 1;
        sgeev("X", "N", 0, a, 1, wr, wi, vl, 1, vr, 1, w, 1, &info);
        chkxer("SGEEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        sgeev("N", "X", 0, a, 1, wr, wi, vl, 1, vr, 1, w, 1, &info);
        chkxer("SGEEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        sgeev("N", "N", -1, a, 1, wr, wi, vl, 1, vr, 1, w, 1, &info);
        chkxer("SGEEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        sgeev("N", "N", 2, a, 1, wr, wi, vl, 1, vr, 1, w, 6, &info);
        chkxer("SGEEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        sgeev("V", "N", 2, a, 2, wr, wi, vl, 1, vr, 1, w, 8, &info);
        chkxer("SGEEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 11;
        sgeev("N", "V", 2, a, 2, wr, wi, vl, 1, vr, 1, w, 8, &info);
        chkxer("SGEEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 13;
        sgeev("V", "V", 1, a, 1, wr, wi, vl, 1, vr, 1, w, 3, &info);
        chkxer("SGEEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 7;

    } else if ((c2[0] == 'E' || c2[0] == 'e') &&
               (c2[1] == 'S' || c2[1] == 's')) {

        /* Test SGEES */

        strcpy(xerbla_srnamt, "SGEES ");
        xerbla_infot = 1;
        sgees("X", "N", sslect, 0, a, 1, &sdim, wr, wi, vl, 1, w,
              1, b, &info);
        chkxer("SGEES ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        sgees("N", "X", sslect, 0, a, 1, &sdim, wr, wi, vl, 1, w,
              1, b, &info);
        chkxer("SGEES ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        sgees("N", "S", sslect, -1, a, 1, &sdim, wr, wi, vl, 1, w,
              1, b, &info);
        chkxer("SGEES ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        sgees("N", "S", sslect, 2, a, 1, &sdim, wr, wi, vl, 1, w,
              6, b, &info);
        chkxer("SGEES ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 11;
        sgees("V", "S", sslect, 2, a, 2, &sdim, wr, wi, vl, 1, w,
              6, b, &info);
        chkxer("SGEES ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 13;
        sgees("N", "S", sslect, 1, a, 1, &sdim, wr, wi, vl, 1, w,
              2, b, &info);
        chkxer("SGEES ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 6;

    } else if ((c2[0] == 'V' || c2[0] == 'v') &&
               (c2[1] == 'X' || c2[1] == 'x')) {

        /* Test SGEEVX */

        strcpy(xerbla_srnamt, "SGEEVX");
        xerbla_infot = 1;
        sgeevx("X", "N", "N", "N", 0, a, 1, wr, wi, vl, 1, vr, 1,
               &ilo, &ihi, s, &abnrm, r1, r2, w, 1, iw, &info);
        chkxer("SGEEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        sgeevx("N", "X", "N", "N", 0, a, 1, wr, wi, vl, 1, vr, 1,
               &ilo, &ihi, s, &abnrm, r1, r2, w, 1, iw, &info);
        chkxer("SGEEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        sgeevx("N", "N", "X", "N", 0, a, 1, wr, wi, vl, 1, vr, 1,
               &ilo, &ihi, s, &abnrm, r1, r2, w, 1, iw, &info);
        chkxer("SGEEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        sgeevx("N", "N", "N", "X", 0, a, 1, wr, wi, vl, 1, vr, 1,
               &ilo, &ihi, s, &abnrm, r1, r2, w, 1, iw, &info);
        chkxer("SGEEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        sgeevx("N", "N", "N", "N", -1, a, 1, wr, wi, vl, 1, vr,
               1, &ilo, &ihi, s, &abnrm, r1, r2, w, 1, iw, &info);
        chkxer("SGEEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        sgeevx("N", "N", "N", "N", 2, a, 1, wr, wi, vl, 1, vr, 1,
               &ilo, &ihi, s, &abnrm, r1, r2, w, 1, iw, &info);
        chkxer("SGEEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 11;
        sgeevx("N", "V", "N", "N", 2, a, 2, wr, wi, vl, 1, vr, 1,
               &ilo, &ihi, s, &abnrm, r1, r2, w, 6, iw, &info);
        chkxer("SGEEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 13;
        sgeevx("N", "N", "V", "N", 2, a, 2, wr, wi, vl, 1, vr, 1,
               &ilo, &ihi, s, &abnrm, r1, r2, w, 6, iw, &info);
        chkxer("SGEEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 21;
        sgeevx("N", "N", "N", "N", 1, a, 1, wr, wi, vl, 1, vr, 1,
               &ilo, &ihi, s, &abnrm, r1, r2, w, 1, iw, &info);
        chkxer("SGEEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 21;
        sgeevx("N", "V", "N", "N", 1, a, 1, wr, wi, vl, 1, vr, 1,
               &ilo, &ihi, s, &abnrm, r1, r2, w, 2, iw, &info);
        chkxer("SGEEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 21;
        sgeevx("N", "N", "V", "V", 1, a, 1, wr, wi, vl, 1, vr, 1,
               &ilo, &ihi, s, &abnrm, r1, r2, w, 3, iw, &info);
        chkxer("SGEEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 11;

    } else if ((c2[0] == 'S' || c2[0] == 's') &&
               (c2[1] == 'X' || c2[1] == 'x')) {

        /* Test SGEESX */

        strcpy(xerbla_srnamt, "SGEESX");
        xerbla_infot = 1;
        sgeesx("X", "N", sslect, "N", 0, a, 1, &sdim, wr, wi, vl,
               1, &r1[0], &r2[0], w, 1, iw, 1, b, &info);
        chkxer("SGEESX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        sgeesx("N", "X", sslect, "N", 0, a, 1, &sdim, wr, wi, vl,
               1, &r1[0], &r2[0], w, 1, iw, 1, b, &info);
        chkxer("SGEESX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        sgeesx("N", "N", sslect, "X", 0, a, 1, &sdim, wr, wi, vl,
               1, &r1[0], &r2[0], w, 1, iw, 1, b, &info);
        chkxer("SGEESX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        sgeesx("N", "N", sslect, "N", -1, a, 1, &sdim, wr, wi, vl,
               1, &r1[0], &r2[0], w, 1, iw, 1, b, &info);
        chkxer("SGEESX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        sgeesx("N", "N", sslect, "N", 2, a, 1, &sdim, wr, wi, vl,
               1, &r1[0], &r2[0], w, 6, iw, 1, b, &info);
        chkxer("SGEESX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 12;
        sgeesx("V", "N", sslect, "N", 2, a, 2, &sdim, wr, wi, vl,
               1, &r1[0], &r2[0], w, 6, iw, 1, b, &info);
        chkxer("SGEESX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 16;
        sgeesx("N", "N", sslect, "N", 1, a, 1, &sdim, wr, wi, vl,
               1, &r1[0], &r2[0], w, 2, iw, 1, b, &info);
        chkxer("SGEESX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 7;

    } else if ((c2[0] == 'B' || c2[0] == 'b') &&
               (c2[1] == 'D' || c2[1] == 'd')) {

        /* Test SGESVD */

        strcpy(xerbla_srnamt, "SGESVD");
        xerbla_infot = 1;
        sgesvd("X", "N", 0, 0, a, 1, s, u, 1, vt, 1, w, 1, &info);
        chkxer("SGESVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        sgesvd("N", "X", 0, 0, a, 1, s, u, 1, vt, 1, w, 1, &info);
        chkxer("SGESVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        sgesvd("O", "O", 0, 0, a, 1, s, u, 1, vt, 1, w, 1, &info);
        chkxer("SGESVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        sgesvd("N", "N", -1, 0, a, 1, s, u, 1, vt, 1, w, 1,
               &info);
        chkxer("SGESVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        sgesvd("N", "N", 0, -1, a, 1, s, u, 1, vt, 1, w, 1,
               &info);
        chkxer("SGESVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        sgesvd("N", "N", 2, 1, a, 1, s, u, 1, vt, 1, w, 5, &info);
        chkxer("SGESVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        sgesvd("A", "N", 2, 1, a, 2, s, u, 1, vt, 1, w, 5, &info);
        chkxer("SGESVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 11;
        sgesvd("N", "A", 1, 2, a, 1, s, u, 1, vt, 1, w, 5, &info);
        chkxer("SGESVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = 8;

        /* Test SGESDD */

        strcpy(xerbla_srnamt, "SGESDD");
        xerbla_infot = 1;
        sgesdd("X", 0, 0, a, 1, s, u, 1, vt, 1, w, 1, iw, &info);
        chkxer("SGESDD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        sgesdd("N", -1, 0, a, 1, s, u, 1, vt, 1, w, 1, iw, &info);
        chkxer("SGESDD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        sgesdd("N", 0, -1, a, 1, s, u, 1, vt, 1, w, 1, iw, &info);
        chkxer("SGESDD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        sgesdd("N", 2, 1, a, 1, s, u, 1, vt, 1, w, 5, iw, &info);
        chkxer("SGESDD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        sgesdd("A", 2, 1, a, 2, s, u, 1, vt, 1, w, 5, iw, &info);
        chkxer("SGESDD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        sgesdd("A", 1, 2, a, 1, s, u, 1, vt, 1, w, 5, iw, &info);
        chkxer("SGESDD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = 6;

        /* Test SGEJSV */

        strcpy(xerbla_srnamt, "SGEJSV");
        xerbla_infot = 1;
        sgejsv("X", "U", "V", "R", "N", "N",
               0, 0, a, 1, s, u, 1, vt, 1,
               w, 1, iw, &info);
        chkxer("SGEJSV", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        sgejsv("G", "X", "V", "R", "N", "N",
               0, 0, a, 1, s, u, 1, vt, 1,
               w, 1, iw, &info);
        chkxer("SGEJSV", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        sgejsv("G", "U", "X", "R", "N", "N",
               0, 0, a, 1, s, u, 1, vt, 1,
               w, 1, iw, &info);
        chkxer("SGEJSV", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        sgejsv("G", "U", "V", "X", "N", "N",
               0, 0, a, 1, s, u, 1, vt, 1,
               w, 1, iw, &info);
        chkxer("SGEJSV", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        sgejsv("G", "U", "V", "R", "X", "N",
               0, 0, a, 1, s, u, 1, vt, 1,
               w, 1, iw, &info);
        chkxer("SGEJSV", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        sgejsv("G", "U", "V", "R", "N", "X",
               0, 0, a, 1, s, u, 1, vt, 1,
               w, 1, iw, &info);
        chkxer("SGEJSV", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        sgejsv("G", "U", "V", "R", "N", "N",
               -1, 0, a, 1, s, u, 1, vt, 1,
               w, 1, iw, &info);
        chkxer("SGEJSV", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        sgejsv("G", "U", "V", "R", "N", "N",
               0, -1, a, 1, s, u, 1, vt, 1,
               w, 1, iw, &info);
        chkxer("SGEJSV", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        sgejsv("G", "U", "V", "R", "N", "N",
               2, 1, a, 1, s, u, 1, vt, 1,
               w, 1, iw, &info);
        chkxer("SGEJSV", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 13;
        sgejsv("G", "U", "V", "R", "N", "N",
               2, 2, a, 2, s, u, 1, vt, 2,
               w, 1, iw, &info);
        chkxer("SGEJSV", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 15;
        sgejsv("G", "U", "V", "R", "N", "N",
               2, 2, a, 2, s, u, 2, vt, 1,
               w, 1, iw, &info);
        chkxer("SGEJSV", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = 11;

        /* Test SGESVDX */

        strcpy(xerbla_srnamt, "SGESVDX");
        xerbla_infot = 1;
        sgesvdx("X", "N", "A", 0, 0, a, 1, ZERO, ZERO,
                0, 0, &ns, s, u, 1, vt, 1, w, 1, iw, &info);
        chkxer("SGESVDX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        sgesvdx("N", "X", "A", 0, 0, a, 1, ZERO, ZERO,
                0, 0, &ns, s, u, 1, vt, 1, w, 1, iw, &info);
        chkxer("SGESVDX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        sgesvdx("N", "N", "X", 0, 0, a, 1, ZERO, ZERO,
                0, 0, &ns, s, u, 1, vt, 1, w, 1, iw, &info);
        chkxer("SGESVDX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        sgesvdx("N", "N", "A", -1, 0, a, 1, ZERO, ZERO,
                0, 0, &ns, s, u, 1, vt, 1, w, 1, iw, &info);
        chkxer("SGESVDX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        sgesvdx("N", "N", "A", 0, -1, a, 1, ZERO, ZERO,
                0, 0, &ns, s, u, 1, vt, 1, w, 1, iw, &info);
        chkxer("SGESVDX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        sgesvdx("N", "N", "A", 2, 1, a, 1, ZERO, ZERO,
                0, 0, &ns, s, u, 1, vt, 1, w, 1, iw, &info);
        chkxer("SGESVDX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        sgesvdx("N", "N", "V", 2, 1, a, 2, -ONE, ZERO,
                0, 0, &ns, s, u, 1, vt, 1, w, 1, iw, &info);
        chkxer("SGESVDX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        sgesvdx("N", "N", "V", 2, 1, a, 2, ONE, ZERO,
                0, 0, &ns, s, u, 1, vt, 1, w, 1, iw, &info);
        chkxer("SGESVDX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        sgesvdx("N", "N", "I", 2, 2, a, 2, ZERO, ZERO,
                -1, 0, &ns, s, u, 1, vt, 1, w, 1, iw, &info);
        chkxer("SGESVDX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 11;
        sgesvdx("V", "N", "I", 2, 2, a, 2, ZERO, ZERO,
                0, -1, &ns, s, u, 1, vt, 1, w, 1, iw, &info);
        chkxer("SGESVDX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 15;
        sgesvdx("V", "N", "A", 2, 2, a, 2, ZERO, ZERO,
                0, 0, &ns, s, u, 1, vt, 1, w, 1, iw, &info);
        chkxer("SGESVDX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 17;
        sgesvdx("N", "V", "A", 2, 2, a, 2, ZERO, ZERO,
                0, 0, &ns, s, u, 1, vt, 1, w, 1, iw, &info);
        chkxer("SGESVDX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = 12;

        /* Test SGESVDQ */

        strcpy(xerbla_srnamt, "SGESVDQ");
        xerbla_infot = 1;
        sgesvdq("X", "P", "T", "A", "A", 0, 0, a, 1, s, u,
                0, vt, 0, &ns, iw, 1, w, 1, rw, 1, &info);
        chkxer("SGESVDQ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        sgesvdq("A", "X", "T", "A", "A", 0, 0, a, 1, s, u,
                0, vt, 0, &ns, iw, 1, w, 1, rw, 1, &info);
        chkxer("SGESVDQ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        sgesvdq("A", "P", "X", "A", "A", 0, 0, a, 1, s, u,
                0, vt, 0, &ns, iw, 1, w, 1, rw, 1, &info);
        chkxer("SGESVDQ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        sgesvdq("A", "P", "T", "X", "A", 0, 0, a, 1, s, u,
                0, vt, 0, &ns, iw, 1, w, 1, rw, 1, &info);
        chkxer("SGESVDQ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        sgesvdq("A", "P", "T", "A", "X", 0, 0, a, 1, s, u,
                0, vt, 0, &ns, iw, 1, w, 1, rw, 1, &info);
        chkxer("SGESVDQ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        sgesvdq("A", "P", "T", "A", "A", -1, 0, a, 1, s, u,
                0, vt, 0, &ns, iw, 1, w, 1, rw, 1, &info);
        chkxer("SGESVDQ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        sgesvdq("A", "P", "T", "A", "A", 0, 1, a, 1, s, u,
                0, vt, 0, &ns, iw, 1, w, 1, rw, 1, &info);
        chkxer("SGESVDQ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        sgesvdq("A", "P", "T", "A", "A", 1, 1, a, 0, s, u,
                0, vt, 0, &ns, iw, 1, w, 1, rw, 1, &info);
        chkxer("SGESVDQ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 12;
        sgesvdq("A", "P", "T", "A", "A", 1, 1, a, 1, s, u,
                -1, vt, 0, &ns, iw, 1, w, 1, rw, 1, &info);
        chkxer("SGESVDQ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 14;
        sgesvdq("A", "P", "T", "A", "A", 1, 1, a, 1, s, u,
                1, vt, -1, &ns, iw, 1, w, 1, rw, 1, &info);
        chkxer("SGESVDQ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 17;
        sgesvdq("A", "P", "T", "A", "A", 1, 1, a, 1, s, u,
                1, vt, 1, &ns, iw, -5, w, 1, rw, 1, &info);
        chkxer("SGESVDQ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = 11;
    }

    *ok = xerbla_ok;

    xerbla_srnamt[0] = '\0';
    xerbla_infot = 0;
}
