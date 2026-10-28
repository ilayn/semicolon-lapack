/**
 * @file zerrst.c
 * @brief ZERRST tests the error exits for the Hermitian eigenvalue routines.
 *
 * Port of LAPACK TESTING/EIG/zerrst.f
 * Tests error exits for: ZHETRD, ZHETD2, ZUNGTR, ZUNMTR, ZHPTRD,
 * ZUPGTR, ZUPMTR, ZSTEQR, ZSTEIN, ZPTEQR, ZHBTRD,
 * ZHEEV, ZHEEVX, ZHEEVD, ZHBEV, ZHBEVX, ZHBEVD,
 * ZHPEV, ZHPEVX, ZHPEVD, and ZSTEDC.
 * ZHEEVD_2STAGE, ZHEEVR_2STAGE, ZHEEVX_2STAGE,
 * ZHEEV_2STAGE, ZHBEV_2STAGE, ZHBEVD_2STAGE,
 * ZHBEVX_2STAGE, ZHETRD_2STAGE
 */

#include <complex.h>
#include <math.h>
#include <string.h>
#include "semicolon_lapack_complex_double.h"
#include "verify.h"

void zerrst(const char* path, INT* ok, INT* nt)
{
    enum { NMAX = 3, LIW = 12 * NMAX, LW = 20 * NMAX };

    char c2[2];
    INT i, info, j, m, n;

    INT i1[NMAX], i2[NMAX], i3[NMAX], iw[LIW];
    f64 d[NMAX], e[NMAX], r[LW], rw[LW], x[NMAX];
    c128 a[NMAX * NMAX], c[NMAX * NMAX], q[NMAX * NMAX], tau[NMAX],
         w[LW], z[NMAX * NMAX];

    c2[0] = path[1];
    c2[1] = path[2];

    /* Set the variables to innocuous values. */
    for (j = 0; j < NMAX; j++) {
        for (i = 0; i < NMAX; i++) {
            a[i + j * NMAX] = 1.0 / (f64)(i + j + 2);
        }
    }
    for (j = 0; j < NMAX; j++) {
        d[j] = (f64)(j + 1);
        e[j] = 0.0;
        i1[j] = j;
        i2[j] = j;
        tau[j] = 1.0;
    }
    xerbla_ok = 1;
    *nt = 0;

    /* Test error exits for the ST path. */

    if ((c2[0] == 'S' || c2[0] == 's') && (c2[1] == 'T' || c2[1] == 't')) {

        /* ZHETRD */

        strcpy(xerbla_srnamt, "ZHETRD");
        xerbla_infot = 1;
        zhetrd("/", 0, a, 1, d, e, tau, w, 1, &info);
        chkxer("ZHETRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        zhetrd("U", -1, a, 1, d, e, tau, w, 1, &info);
        chkxer("ZHETRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        zhetrd("U", 2, a, 1, d, e, tau, w, 1, &info);
        chkxer("ZHETRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        zhetrd("U", 0, a, 1, d, e, tau, w, 0, &info);
        chkxer("ZHETRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 4;

        /* ZHETD2 */

        strcpy(xerbla_srnamt, "ZHETD2");
        xerbla_infot = 1;
        zhetd2("/", 0, a, 1, d, e, tau, &info);
        chkxer("ZHETD2", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        zhetd2("U", -1, a, 1, d, e, tau, &info);
        chkxer("ZHETD2", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        zhetd2("U", 2, a, 1, d, e, tau, &info);
        chkxer("ZHETD2", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 3;

        /* ZHETRD_2STAGE */

        strcpy(xerbla_srnamt, "ZHETRD_2STAGE");
        xerbla_infot = 1;
        zhetrd_2stage("/", "U", 0, a, 1, d, e, tau,
                      c, 1, w, 1, &info);
        chkxer("ZHETRD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 1;
        zhetrd_2stage("H", "U", 0, a, 1, d, e, tau,
                      c, 1, w, 1, &info);
        chkxer("ZHETRD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        zhetrd_2stage("N", "/", 0, a, 1, d, e, tau,
                      c, 1, w, 1, &info);
        chkxer("ZHETRD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        zhetrd_2stage("N", "U", -1, a, 1, d, e, tau,
                      c, 1, w, 1, &info);
        chkxer("ZHETRD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        zhetrd_2stage("N", "U", 2, a, 1, d, e, tau,
                      c, 1, w, 1, &info);
        chkxer("ZHETRD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        zhetrd_2stage("N", "U", 0, a, 1, d, e, tau,
                      c, 0, w, 1, &info);
        chkxer("ZHETRD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 12;
        zhetrd_2stage("N", "U", 0, a, 1, d, e, tau,
                      c, 1, w, 0, &info);
        chkxer("ZHETRD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 7;

        /* ZHETRD_HE2HB */

        strcpy(xerbla_srnamt, "ZHETRD_HE2HB");
        xerbla_infot = 1;
        zhetrd_he2hb("/", 0, 0, a, 1, c, 1, tau, w, 1, &info);
        chkxer("ZHETRD_HE2HB", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        zhetrd_he2hb("U", -1, 0, a, 1, c, 1, tau, w, 1, &info);
        chkxer("ZHETRD_HE2HB", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        zhetrd_he2hb("U", 0, -1, a, 1, c, 1, tau, w, 1, &info);
        chkxer("ZHETRD_HE2HB", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        zhetrd_he2hb("U", 2, 0, a, 1, c, 1, tau, w, 1, &info);
        chkxer("ZHETRD_HE2HB", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        zhetrd_he2hb("U", 0, 2, a, 1, c, 1, tau, w, 1, &info);
        chkxer("ZHETRD_HE2HB", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        zhetrd_he2hb("U", 0, 0, a, 1, c, 1, tau, w, 0, &info);
        chkxer("ZHETRD_HE2HB", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 6;

        /* ZHETRD_HB2ST */

        strcpy(xerbla_srnamt, "ZHETRD_HB2ST");
        xerbla_infot = 1;
        zhetrd_hb2st("/", "N", "U", 0, 0, a, 1, d, e,
                     c, 1, w, 1, &info);
        chkxer("ZHETRD_HB2ST", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        zhetrd_hb2st("Y", "/", "U", 0, 0, a, 1, d, e,
                     c, 1, w, 1, &info);
        chkxer("ZHETRD_HB2ST", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        zhetrd_hb2st("Y", "H", "U", 0, 0, a, 1, d, e,
                     c, 1, w, 1, &info);
        chkxer("ZHETRD_HB2ST", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        zhetrd_hb2st("Y", "N", "/", 0, 0, a, 1, d, e,
                     c, 1, w, 1, &info);
        chkxer("ZHETRD_HB2ST", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        zhetrd_hb2st("Y", "N", "U", -1, 0, a, 1, d, e,
                     c, 1, w, 1, &info);
        chkxer("ZHETRD_HB2ST", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        zhetrd_hb2st("Y", "N", "U", 0, -1, a, 1, d, e,
                     c, 1, w, 1, &info);
        chkxer("ZHETRD_HB2ST", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        zhetrd_hb2st("Y", "N", "U", 0, 1, a, 1, d, e,
                     c, 1, w, 1, &info);
        chkxer("ZHETRD_HB2ST", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 11;
        zhetrd_hb2st("Y", "N", "U", 0, 0, a, 1, d, e,
                     c, 0, w, 1, &info);
        chkxer("ZHETRD_HB2ST", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 13;
        zhetrd_hb2st("Y", "N", "U", 0, 0, a, 1, d, e,
                     c, 1, w, 0, &info);
        chkxer("ZHETRD_HB2ST", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 9;

        /* ZUNGTR */

        strcpy(xerbla_srnamt, "ZUNGTR");
        xerbla_infot = 1;
        zungtr("/", 0, a, 1, tau, w, 1, &info);
        chkxer("ZUNGTR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        zungtr("U", -1, a, 1, tau, w, 1, &info);
        chkxer("ZUNGTR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        zungtr("U", 2, a, 1, tau, w, 1, &info);
        chkxer("ZUNGTR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        zungtr("U", 3, a, 3, tau, w, 1, &info);
        chkxer("ZUNGTR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 4;

        /* ZUNMTR */

        strcpy(xerbla_srnamt, "ZUNMTR");
        xerbla_infot = 1;
        zunmtr("/", "U", "N", 0, 0, a, 1, tau, c, 1, w, 1, &info);
        chkxer("ZUNMTR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        zunmtr("L", "/", "N", 0, 0, a, 1, tau, c, 1, w, 1, &info);
        chkxer("ZUNMTR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        zunmtr("L", "U", "/", 0, 0, a, 1, tau, c, 1, w, 1, &info);
        chkxer("ZUNMTR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        zunmtr("L", "U", "N", -1, 0, a, 1, tau, c, 1, w, 1,
               &info);
        chkxer("ZUNMTR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        zunmtr("L", "U", "N", 0, -1, a, 1, tau, c, 1, w, 1,
               &info);
        chkxer("ZUNMTR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        zunmtr("L", "U", "N", 2, 0, a, 1, tau, c, 2, w, 1, &info);
        chkxer("ZUNMTR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        zunmtr("R", "U", "N", 0, 2, a, 1, tau, c, 1, w, 1, &info);
        chkxer("ZUNMTR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        zunmtr("L", "U", "N", 2, 0, a, 2, tau, c, 1, w, 1, &info);
        chkxer("ZUNMTR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 12;
        zunmtr("L", "U", "N", 0, 2, a, 1, tau, c, 1, w, 1, &info);
        chkxer("ZUNMTR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 12;
        zunmtr("R", "U", "N", 2, 0, a, 1, tau, c, 2, w, 1, &info);
        chkxer("ZUNMTR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 10;

        /* ZHPTRD */

        strcpy(xerbla_srnamt, "ZHPTRD");
        xerbla_infot = 1;
        zhptrd("/", 0, a, d, e, tau, &info);
        chkxer("ZHPTRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        zhptrd("U", -1, a, d, e, tau, &info);
        chkxer("ZHPTRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 2;

        /* ZUPGTR */

        strcpy(xerbla_srnamt, "ZUPGTR");
        xerbla_infot = 1;
        zupgtr("/", 0, a, tau, z, 1, w, &info);
        chkxer("ZUPGTR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        zupgtr("U", -1, a, tau, z, 1, w, &info);
        chkxer("ZUPGTR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        zupgtr("U", 2, a, tau, z, 1, w, &info);
        chkxer("ZUPGTR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 3;

        /* ZUPMTR */

        strcpy(xerbla_srnamt, "ZUPMTR");
        xerbla_infot = 1;
        zupmtr("/", "U", "N", 0, 0, a, tau, c, 1, w, &info);
        chkxer("ZUPMTR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        zupmtr("L", "/", "N", 0, 0, a, tau, c, 1, w, &info);
        chkxer("ZUPMTR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        zupmtr("L", "U", "/", 0, 0, a, tau, c, 1, w, &info);
        chkxer("ZUPMTR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        zupmtr("L", "U", "N", -1, 0, a, tau, c, 1, w, &info);
        chkxer("ZUPMTR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        zupmtr("L", "U", "N", 0, -1, a, tau, c, 1, w, &info);
        chkxer("ZUPMTR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        zupmtr("L", "U", "N", 2, 0, a, tau, c, 1, w, &info);
        chkxer("ZUPMTR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 6;

        /* ZPTEQR */

        strcpy(xerbla_srnamt, "ZPTEQR");
        xerbla_infot = 1;
        zpteqr("/", 0, d, e, z, 1, rw, &info);
        chkxer("ZPTEQR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        zpteqr("N", -1, d, e, z, 1, rw, &info);
        chkxer("ZPTEQR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        zpteqr("V", 2, d, e, z, 1, rw, &info);
        chkxer("ZPTEQR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 3;

        /* ZSTEIN */

        strcpy(xerbla_srnamt, "ZSTEIN");
        xerbla_infot = 1;
        zstein(-1, d, e, 0, x, i1, i2, z, 1, rw, iw, i3, &info);
        chkxer("ZSTEIN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        zstein(0, d, e, -1, x, i1, i2, z, 1, rw, iw, i3, &info);
        chkxer("ZSTEIN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        zstein(0, d, e, 1, x, i1, i2, z, 1, rw, iw, i3, &info);
        chkxer("ZSTEIN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        zstein(2, d, e, 0, x, i1, i2, z, 1, rw, iw, i3, &info);
        chkxer("ZSTEIN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 4;

        /* ZSTEQR */

        strcpy(xerbla_srnamt, "ZSTEQR");
        xerbla_infot = 1;
        zsteqr("/", 0, d, e, z, 1, rw, &info);
        chkxer("ZSTEQR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        zsteqr("N", -1, d, e, z, 1, rw, &info);
        chkxer("ZSTEQR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        zsteqr("V", 2, d, e, z, 1, rw, &info);
        chkxer("ZSTEQR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 3;

        /* ZSTEDC */

        strcpy(xerbla_srnamt, "ZSTEDC");
        xerbla_infot = 1;
        zstedc("/", 0, d, e, z, 1, w, 1, rw, 1, iw, 1, &info);
        chkxer("ZSTEDC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        zstedc("N", -1, d, e, z, 1, w, 1, rw, 1, iw, 1, &info);
        chkxer("ZSTEDC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        zstedc("V", 2, d, e, z, 1, w, 4, rw, 23, iw, 28, &info);
        chkxer("ZSTEDC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        zstedc("N", 2, d, e, z, 1, w, 0, rw, 1, iw, 1, &info);
        chkxer("ZSTEDC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        zstedc("V", 2, d, e, z, 2, w, 0, rw, 23, iw, 28, &info);
        chkxer("ZSTEDC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        zstedc("N", 2, d, e, z, 1, w, 1, rw, 0, iw, 1, &info);
        chkxer("ZSTEDC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        zstedc("I", 2, d, e, z, 2, w, 1, rw, 1, iw, 12, &info);
        chkxer("ZSTEDC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        zstedc("V", 2, d, e, z, 2, w, 4, rw, 1, iw, 28, &info);
        chkxer("ZSTEDC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 12;
        zstedc("N", 2, d, e, z, 1, w, 1, rw, 1, iw, 0, &info);
        chkxer("ZSTEDC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 12;
        zstedc("I", 2, d, e, z, 2, w, 1, rw, 23, iw, 0, &info);
        chkxer("ZSTEDC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 12;
        zstedc("V", 2, d, e, z, 2, w, 4, rw, 23, iw, 0, &info);
        chkxer("ZSTEDC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 11;

        /* ZHEEVD */

        strcpy(xerbla_srnamt, "ZHEEVD");
        xerbla_infot = 1;
        zheevd("/", "U", 0, a, 1, x, w, 1, rw, 1, iw, 1, &info);
        chkxer("ZHEEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        zheevd("N", "/", 0, a, 1, x, w, 1, rw, 1, iw, 1, &info);
        chkxer("ZHEEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        zheevd("N", "U", -1, a, 1, x, w, 1, rw, 1, iw, 1, &info);
        chkxer("ZHEEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        zheevd("N", "U", 2, a, 1, x, w, 3, rw, 2, iw, 1, &info);
        chkxer("ZHEEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        zheevd("N", "U", 1, a, 1, x, w, 0, rw, 1, iw, 1, &info);
        chkxer("ZHEEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        zheevd("N", "U", 2, a, 2, x, w, 2, rw, 2, iw, 1, &info);
        chkxer("ZHEEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        zheevd("V", "U", 2, a, 2, x, w, 3, rw, 25, iw, 12, &info);
        chkxer("ZHEEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        zheevd("N", "U", 1, a, 1, x, w, 1, rw, 0, iw, 1, &info);
        chkxer("ZHEEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        zheevd("N", "U", 2, a, 2, x, w, 3, rw, 1, iw, 1, &info);
        chkxer("ZHEEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        zheevd("V", "U", 2, a, 2, x, w, 8, rw, 18, iw, 12, &info);
        chkxer("ZHEEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 12;
        zheevd("N", "U", 1, a, 1, x, w, 1, rw, 1, iw, 0, &info);
        chkxer("ZHEEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 12;
        zheevd("V", "U", 2, a, 2, x, w, 8, rw, 25, iw, 11, &info);
        chkxer("ZHEEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 12;

        /* ZHEEVD_2STAGE */

        strcpy(xerbla_srnamt, "ZHEEVD_2STAGE");
        xerbla_infot = 1;
        zheevd_2stage("/", "U", 0, a, 1, x, w, 1,
                      rw, 1, iw, 1, &info);
        chkxer("ZHEEVD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 1;
        zheevd_2stage("V", "U", 0, a, 1, x, w, 1,
                      rw, 1, iw, 1, &info);
        chkxer("ZHEEVD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        zheevd_2stage("N", "/", 0, a, 1, x, w, 1,
                      rw, 1, iw, 1, &info);
        chkxer("ZHEEVD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        zheevd_2stage("N", "U", -1, a, 1, x, w, 1,
                      rw, 1, iw, 1, &info);
        chkxer("ZHEEVD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        zheevd_2stage("N", "U", 2, a, 1, x, w, 3,
                      rw, 2, iw, 1, &info);
        chkxer("ZHEEVD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        zheevd_2stage("N", "U", 1, a, 1, x, w, 0,
                      rw, 1, iw, 1, &info);
        chkxer("ZHEEVD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        zheevd_2stage("N", "U", 2, a, 2, x, w, 2,
                      rw, 2, iw, 1, &info);
        chkxer("ZHEEVD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        /* xerbla_infot = 8;
         * zheevd_2stage("V", "U", 2, a, 2, x, w, 3,
         *               rw, 25, iw, 12, &info);
         * chkxer("ZHEEVD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok); */
        xerbla_infot = 10;
        zheevd_2stage("N", "U", 1, a, 1, x, w, 1,
                      rw, 0, iw, 1, &info);
        chkxer("ZHEEVD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        zheevd_2stage("N", "U", 2, a, 2, x, w, 25,
                      rw, 1, iw, 1, &info);
        chkxer("ZHEEVD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        /* xerbla_infot = 10;
         * zheevd_2stage("V", "U", 2, a, 2, x, w, 8,
         *               rw, 18, iw, 12, &info);
         * chkxer("ZHEEVD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok); */
        xerbla_infot = 12;
        zheevd_2stage("N", "U", 1, a, 1, x, w, 1,
                      rw, 1, iw, 0, &info);
        chkxer("ZHEEVD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 12;
        /* zheevd_2stage("V", "U", 2, a, 2, x, w, 8,
         *               rw, 25, iw, 11, &info);
         * chkxer("ZHEEVD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok); */
        *nt = *nt + 10;

        /* ZHEEV */

        strcpy(xerbla_srnamt, "ZHEEV ");
        xerbla_infot = 1;
        zheev("/", "U", 0, a, 1, x, w, 1, rw, &info);
        chkxer("ZHEEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        zheev("N", "/", 0, a, 1, x, w, 1, rw, &info);
        chkxer("ZHEEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        zheev("N", "U", -1, a, 1, x, w, 1, rw, &info);
        chkxer("ZHEEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        zheev("N", "U", 2, a, 1, x, w, 3, rw, &info);
        chkxer("ZHEEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        zheev("N", "U", 2, a, 2, x, w, 2, rw, &info);
        chkxer("ZHEEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 5;

        /* ZHEEV_2STAGE */

        strcpy(xerbla_srnamt, "ZHEEV_2STAGE ");
        xerbla_infot = 1;
        zheev_2stage("/", "U", 0, a, 1, x, w, 1, rw, &info);
        chkxer("ZHEEV_2STAGE ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 1;
        zheev_2stage("V", "U", 0, a, 1, x, w, 1, rw, &info);
        chkxer("ZHEEV_2STAGE ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        zheev_2stage("N", "/", 0, a, 1, x, w, 1, rw, &info);
        chkxer("ZHEEV_2STAGE ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        zheev_2stage("N", "U", -1, a, 1, x, w, 1, rw, &info);
        chkxer("ZHEEV_2STAGE ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        zheev_2stage("N", "U", 2, a, 1, x, w, 3, rw, &info);
        chkxer("ZHEEV_2STAGE ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        zheev_2stage("N", "U", 2, a, 2, x, w, 2, rw, &info);
        chkxer("ZHEEV_2STAGE ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 6;

        /* ZHEEVX */

        strcpy(xerbla_srnamt, "ZHEEVX");
        xerbla_infot = 1;
        zheevx("/", "A", "U", 0, a, 1, 0.0, 0.0, -1, -1, 0.0,
               &m, x, z, 1, w, 1, rw, iw, i3, &info);
        chkxer("ZHEEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        zheevx("V", "/", "U", 0, a, 1, 0.0, 1.0, 0, -1, 0.0,
               &m, x, z, 1, w, 1, rw, iw, i3, &info);
        chkxer("ZHEEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        zheevx("V", "A", "/", 0, a, 1, 0.0, 0.0, -1, -1, 0.0,
               &m, x, z, 1, w, 1, rw, iw, i3, &info);
        xerbla_infot = 4;
        zheevx("V", "A", "U", -1, a, 1, 0.0, 0.0, -1, -1,
               0.0, &m, x, z, 1, w, 1, rw, iw, i3, &info);
        chkxer("ZHEEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        zheevx("V", "A", "U", 2, a, 1, 0.0, 0.0, -1, -1, 0.0,
               &m, x, z, 2, w, 3, rw, iw, i3, &info);
        chkxer("ZHEEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        zheevx("V", "V", "U", 1, a, 1, 0.0, 0.0, -1, -1, 0.0,
               &m, x, z, 1, w, 1, rw, iw, i3, &info);
        chkxer("ZHEEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        zheevx("V", "I", "U", 1, a, 1, 0.0, 0.0, -1, -1, 0.0,
               &m, x, z, 1, w, 1, rw, iw, i3, &info);
        chkxer("ZHEEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        zheevx("V", "I", "U", 2, a, 2, 0.0, 0.0, 1, 0, 0.0,
               &m, x, z, 2, w, 3, rw, iw, i3, &info);
        chkxer("ZHEEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 15;
        zheevx("V", "A", "U", 2, a, 2, 0.0, 0.0, -1, -1, 0.0,
               &m, x, z, 1, w, 3, rw, iw, i3, &info);
        chkxer("ZHEEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 17;
        zheevx("V", "A", "U", 2, a, 2, 0.0, 0.0, -1, -1, 0.0,
               &m, x, z, 2, w, 2, rw, iw, i1, &info);
        chkxer("ZHEEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 10;

        /* ZHEEVX_2STAGE */

        strcpy(xerbla_srnamt, "ZHEEVX_2STAGE");
        xerbla_infot = 1;
        zheevx_2stage("/", "A", "U", 0, a, 1,
                      0.0, 0.0, -1, -1, 0.0,
                      &m, x, z, 1, w, 1, rw, iw, i3, &info);
        chkxer("ZHEEVX_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 1;
        zheevx_2stage("V", "A", "U", 0, a, 1,
                      0.0, 0.0, -1, -1, 0.0,
                      &m, x, z, 1, w, 1, rw, iw, i3, &info);
        chkxer("ZHEEVX_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        zheevx_2stage("N", "/", "U", 0, a, 1,
                      0.0, 1.0, 0, -1, 0.0,
                      &m, x, z, 1, w, 1, rw, iw, i3, &info);
        chkxer("ZHEEVX_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        zheevx_2stage("N", "A", "/", 0, a, 1,
                      0.0, 0.0, -1, -1, 0.0,
                      &m, x, z, 1, w, 1, rw, iw, i3, &info);
        xerbla_infot = 4;
        zheevx_2stage("N", "A", "U", -1, a, 1,
                      0.0, 0.0, -1, -1, 0.0,
                      &m, x, z, 1, w, 1, rw, iw, i3, &info);
        chkxer("ZHEEVX_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        zheevx_2stage("N", "A", "U", 2, a, 1,
                      0.0, 0.0, -1, -1, 0.0,
                      &m, x, z, 2, w, 3, rw, iw, i3, &info);
        chkxer("ZHEEVX_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        zheevx_2stage("N", "V", "U", 1, a, 1,
                      0.0, 0.0, -1, -1, 0.0,
                      &m, x, z, 1, w, 1, rw, iw, i3, &info);
        chkxer("ZHEEVX_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        zheevx_2stage("N", "I", "U", 1, a, 1,
                      0.0, 0.0, -1, -1, 0.0,
                      &m, x, z, 1, w, 1, rw, iw, i3, &info);
        chkxer("ZHEEVX_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        zheevx_2stage("N", "I", "U", 2, a, 2,
                      0.0, 0.0, 1, 0, 0.0,
                      &m, x, z, 2, w, 3, rw, iw, i3, &info);
        chkxer("ZHEEVX_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 15;
        zheevx_2stage("N", "A", "U", 2, a, 2,
                      0.0, 0.0, -1, -1, 0.0,
                      &m, x, z, 0, w, 3, rw, iw, i3, &info);
        chkxer("ZHEEVX_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 17;
        zheevx_2stage("N", "A", "U", 2, a, 2,
                      0.0, 0.0, -1, -1, 0.0,
                      &m, x, z, 2, w, 0, rw, iw, i1, &info);
        chkxer("ZHEEVX_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 11;

        /* ZHEEVR */

        strcpy(xerbla_srnamt, "ZHEEVR");
        n = 1;
        xerbla_infot = 1;
        zheevr("/", "A", "U", 0, a, 1, 0.0, 0.0, 0, 0, 0.0,
               &m, r, z, 1, iw, q, 2 * n, rw, 24 * n, &iw[2 * n],
               10 * n, &info);
        chkxer("ZHEEVR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        zheevr("V", "/", "U", 0, a, 1, 0.0, 0.0, 0, 0, 0.0,
               &m, r, z, 1, iw, q, 2 * n, rw, 24 * n, &iw[2 * n],
               10 * n, &info);
        chkxer("ZHEEVR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        zheevr("V", "A", "/", -1, a, 1, 0.0, 0.0, 0, 0,
               0.0, &m, r, z, 1, iw, q, 2 * n, rw, 24 * n,
               &iw[2 * n], 10 * n, &info);
        chkxer("ZHEEVR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        zheevr("V", "A", "U", -1, a, 1, 0.0, 0.0, 0, 0,
               0.0, &m, r, z, 1, iw, q, 2 * n, rw, 24 * n,
               &iw[2 * n], 10 * n, &info);
        chkxer("ZHEEVR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        zheevr("V", "A", "U", 2, a, 1, 0.0, 0.0, 0, 0, 0.0,
               &m, r, z, 1, iw, q, 2 * n, rw, 24 * n, &iw[2 * n],
               10 * n, &info);
        chkxer("ZHEEVR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        zheevr("V", "V", "U", 1, a, 1, 0.0, 0.0, 0, 0, 0.0,
               &m, r, z, 1, iw, q, 2 * n, rw, 24 * n, &iw[2 * n],
               10 * n, &info);
        chkxer("ZHEEVR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        zheevr("V", "I", "U", 1, a, 1, 0.0, 0.0, -1, 0, 0.0,
               &m, r, z, 1, iw, q, 2 * n, rw, 24 * n, &iw[2 * n],
               10 * n, &info);
        chkxer("ZHEEVR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;

        zheevr("V", "I", "U", 2, a, 2, 0.0, 0.0, 1, 0, 0.0,
               &m, r, z, 1, iw, q, 2 * n, rw, 24 * n, &iw[2 * n],
               10 * n, &info);
        chkxer("ZHEEVR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 15;
        zheevr("V", "I", "U", 1, a, 1, 0.0, 0.0, 0, 0, 0.0,
               &m, r, z, 0, iw, q, 2 * n, rw, 24 * n, &iw[2 * n],
               10 * n, &info);
        chkxer("ZHEEVR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 18;
        zheevr("V", "I", "U", 1, a, 1, 0.0, 0.0, 0, 0, 0.0,
               &m, r, z, 1, iw, q, 0, rw, 24 * n, &iw[2 * n],
               10 * n, &info);
        chkxer("ZHEEVR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 20;
        zheevr("V", "I", "U", 1, a, 1, 0.0, 0.0, 0, 0, 0.0,
               &m, r, z, 1, iw, q, 2 * n, rw, 0, &iw[2 * n - 2],
               10 * n, &info);
        chkxer("ZHEEVR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 22;
        zheevr("V", "I", "U", 1, a, 1, 0.0, 0.0, 0, 0, 0.0,
               &m, r, z, 1, iw, q, 2 * n, rw, 24 * n, &iw[2 * n], 0,
               &info);
        chkxer("ZHEEVR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 12;

        /* ZHEEVR_2STAGE */

        strcpy(xerbla_srnamt, "ZHEEVR_2STAGE");
        n = 1;
        xerbla_infot = 1;
        zheevr_2stage("/", "A", "U", 0, a, 1,
                      0.0, 0.0, 0, 0, 0.0,
                      &m, r, z, 1, iw, q, 2 * n, rw, 24 * n, &iw[2 * n],
                      10 * n, &info);
        chkxer("ZHEEVR_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 1;
        zheevr_2stage("V", "A", "U", 0, a, 1,
                      0.0, 0.0, 0, 0, 0.0,
                      &m, r, z, 1, iw, q, 2 * n, rw, 24 * n, &iw[2 * n],
                      10 * n, &info);
        chkxer("ZHEEVR_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        zheevr_2stage("N", "/", "U", 0, a, 1,
                      0.0, 0.0, 0, 0, 0.0,
                      &m, r, z, 1, iw, q, 2 * n, rw, 24 * n, &iw[2 * n],
                      10 * n, &info);
        chkxer("ZHEEVR_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        zheevr_2stage("N", "A", "/", -1, a, 1,
                      0.0, 0.0, 0, 0, 0.0,
                      &m, r, z, 1, iw, q, 2 * n, rw, 24 * n,
                      &iw[2 * n], 10 * n, &info);
        chkxer("ZHEEVR_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        zheevr_2stage("N", "A", "U", -1, a, 1,
                      0.0, 0.0, 0, 0, 0.0,
                      &m, r, z, 1, iw, q, 2 * n, rw, 24 * n,
                      &iw[2 * n], 10 * n, &info);
        chkxer("ZHEEVR_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        zheevr_2stage("N", "A", "U", 2, a, 1,
                      0.0, 0.0, 0, 0, 0.0,
                      &m, r, z, 1, iw, q, 2 * n, rw, 24 * n, &iw[2 * n],
                      10 * n, &info);
        chkxer("ZHEEVR_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        zheevr_2stage("N", "V", "U", 1, a, 1,
                      0.0, 0.0, 0, 0, 0.0,
                      &m, r, z, 1, iw, q, 2 * n, rw, 24 * n, &iw[2 * n],
                      10 * n, &info);
        chkxer("ZHEEVR_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        zheevr_2stage("N", "I", "U", 1, a, 1,
                      0.0, 0.0, -1, 0, 0.0,
                      &m, r, z, 1, iw, q, 2 * n, rw, 24 * n, &iw[2 * n],
                      10 * n, &info);
        chkxer("ZHEEVR_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        zheevr_2stage("N", "I", "U", 2, a, 2,
                      0.0, 0.0, 1, 0, 0.0,
                      &m, r, z, 1, iw, q, 2 * n, rw, 24 * n, &iw[2 * n],
                      10 * n, &info);
        chkxer("ZHEEVR_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 15;
        zheevr_2stage("N", "I", "U", 1, a, 1,
                      0.0, 0.0, 0, 0, 0.0,
                      &m, r, z, 0, iw, q, 2 * n, rw, 24 * n, &iw[2 * n],
                      10 * n, &info);
        chkxer("ZHEEVR_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 18;
        zheevr_2stage("N", "I", "U", 1, a, 1,
                      0.0, 0.0, 0, 0, 0.0,
                      &m, r, z, 1, iw, q, 0, rw, 24 * n, &iw[2 * n],
                      10 * n, &info);
        chkxer("ZHEEVR_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 20;
        zheevr_2stage("N", "I", "U", 1, a, 1,
                      0.0, 0.0, 0, 0, 0.0,
                      &m, r, z, 1, iw, q, 26 * n, rw, 0, &iw[2 * n - 2],
                      10 * n, &info);
        chkxer("ZHEEVR_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 22;
        zheevr_2stage("N", "I", "U", 1, a, 1,
                      0.0, 0.0, 0, 0, 0.0,
                      &m, r, z, 1, iw, q, 26 * n, rw, 24 * n, &iw[2 * n], 0,
                      &info);
        chkxer("ZHEEVR_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 13;

        /* ZHPEVD */

        strcpy(xerbla_srnamt, "ZHPEVD");
        xerbla_infot = 1;
        zhpevd("/", "U", 0, a, x, z, 1, w, 1, rw, 1, iw, 1,
               &info);
        chkxer("ZHPEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        zhpevd("N", "/", 0, a, x, z, 1, w, 1, rw, 1, iw, 1,
               &info);
        chkxer("ZHPEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        zhpevd("N", "U", -1, a, x, z, 1, w, 1, rw, 1, iw, 1,
               &info);
        chkxer("ZHPEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        zhpevd("V", "U", 2, a, x, z, 1, w, 4, rw, 25, iw, 12,
               &info);
        chkxer("ZHPEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        zhpevd("N", "U", 1, a, x, z, 1, w, 0, rw, 1, iw, 1,
               &info);
        chkxer("ZHPEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        zhpevd("N", "U", 2, a, x, z, 2, w, 1, rw, 2, iw, 1,
               &info);
        chkxer("ZHPEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        zhpevd("V", "U", 2, a, x, z, 2, w, 2, rw, 25, iw, 12,
               &info);
        chkxer("ZHPEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 11;
        zhpevd("N", "U", 1, a, x, z, 1, w, 1, rw, 0, iw, 1,
               &info);
        chkxer("ZHPEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 11;
        zhpevd("N", "U", 2, a, x, z, 2, w, 2, rw, 1, iw, 1,
               &info);
        chkxer("ZHPEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 11;
        zhpevd("V", "U", 2, a, x, z, 2, w, 4, rw, 18, iw, 12,
               &info);
        chkxer("ZHPEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 13;
        zhpevd("N", "U", 1, a, x, z, 1, w, 1, rw, 1, iw, 0,
               &info);
        chkxer("ZHPEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 13;
        zhpevd("N", "U", 2, a, x, z, 2, w, 2, rw, 2, iw, 0,
               &info);
        chkxer("ZHPEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 13;
        zhpevd("V", "U", 2, a, x, z, 2, w, 4, rw, 25, iw, 2,
               &info);
        chkxer("ZHPEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 13;

        /* ZHPEV */

        strcpy(xerbla_srnamt, "ZHPEV ");
        xerbla_infot = 1;
        zhpev("/", "U", 0, a, x, z, 1, w, rw, &info);
        chkxer("ZHPEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        zhpev("N", "/", 0, a, x, z, 1, w, rw, &info);
        chkxer("ZHPEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        zhpev("N", "U", -1, a, x, z, 1, w, rw, &info);
        chkxer("ZHPEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        zhpev("V", "U", 2, a, x, z, 1, w, rw, &info);
        chkxer("ZHPEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 4;

        /* ZHPEVX */

        strcpy(xerbla_srnamt, "ZHPEVX");
        xerbla_infot = 1;
        zhpevx("/", "A", "U", 0, a, 0.0, 0.0, -1, -1, 0.0, &m,
               x, z, 1, w, rw, iw, i3, &info);
        chkxer("ZHPEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        zhpevx("V", "/", "U", 0, a, 0.0, 1.0, 0, -1, 0.0, &m,
               x, z, 1, w, rw, iw, i3, &info);
        chkxer("ZHPEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        zhpevx("V", "A", "/", 0, a, 0.0, 0.0, -1, -1, 0.0, &m,
               x, z, 1, w, rw, iw, i3, &info);
        chkxer("ZHPEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        zhpevx("V", "A", "U", -1, a, 0.0, 0.0, -1, -1, 0.0,
               &m, x, z, 1, w, rw, iw, i3, &info);
        chkxer("ZHPEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        zhpevx("V", "V", "U", 1, a, 0.0, 0.0, -1, -1, 0.0, &m,
               x, z, 1, w, rw, iw, i3, &info);
        chkxer("ZHPEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        zhpevx("V", "I", "U", 1, a, 0.0, 0.0, -1, -1, 0.0, &m,
               x, z, 1, w, rw, iw, i3, &info);
        chkxer("ZHPEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        zhpevx("V", "I", "U", 2, a, 0.0, 0.0, 1, 0, 0.0, &m,
               x, z, 2, w, rw, iw, i3, &info);
        chkxer("ZHPEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 14;
        zhpevx("V", "A", "U", 2, a, 0.0, 0.0, -1, -1, 0.0, &m,
               x, z, 1, w, rw, iw, i3, &info);
        chkxer("ZHPEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 8;

    /* Test error exits for the HB path. */

    } else if ((c2[0] == 'H' || c2[0] == 'h') && (c2[1] == 'B' || c2[1] == 'b')) {

        /* ZHBTRD */

        strcpy(xerbla_srnamt, "ZHBTRD");
        xerbla_infot = 1;
        zhbtrd("/", "U", 0, 0, a, 1, d, e, z, 1, w, &info);
        chkxer("ZHBTRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        zhbtrd("N", "/", 0, 0, a, 1, d, e, z, 1, w, &info);
        chkxer("ZHBTRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        zhbtrd("N", "U", -1, 0, a, 1, d, e, z, 1, w, &info);
        chkxer("ZHBTRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        zhbtrd("N", "U", 0, -1, a, 1, d, e, z, 1, w, &info);
        chkxer("ZHBTRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        zhbtrd("N", "U", 1, 1, a, 1, d, e, z, 1, w, &info);
        chkxer("ZHBTRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        zhbtrd("V", "U", 2, 0, a, 1, d, e, z, 1, w, &info);
        chkxer("ZHBTRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 6;

        /* ZHETRD_HB2ST */

        strcpy(xerbla_srnamt, "ZHETRD_HB2ST");
        xerbla_infot = 1;
        zhetrd_hb2st("/", "N", "U", 0, 0, a, 1, d, e,
                     c, 1, w, 1, &info);
        chkxer("ZHETRD_HB2ST", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        zhetrd_hb2st("N", "/", "U", 0, 0, a, 1, d, e,
                     c, 1, w, 1, &info);
        chkxer("ZHETRD_HB2ST", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        zhetrd_hb2st("N", "H", "U", 0, 0, a, 1, d, e,
                     c, 1, w, 1, &info);
        chkxer("ZHETRD_HB2ST", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        zhetrd_hb2st("N", "N", "/", 0, 0, a, 1, d, e,
                     c, 1, w, 1, &info);
        chkxer("ZHETRD_HB2ST", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        zhetrd_hb2st("N", "N", "U", -1, 0, a, 1, d, e,
                     c, 1, w, 1, &info);
        chkxer("ZHETRD_HB2ST", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        zhetrd_hb2st("N", "N", "U", 0, -1, a, 1, d, e,
                     c, 1, w, 1, &info);
        chkxer("ZHETRD_HB2ST", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        zhetrd_hb2st("N", "N", "U", 0, 1, a, 1, d, e,
                     c, 1, w, 1, &info);
        chkxer("ZHETRD_HB2ST", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 11;
        zhetrd_hb2st("N", "N", "U", 0, 0, a, 1, d, e,
                     c, 0, w, 1, &info);
        chkxer("ZHETRD_HB2ST", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 13;
        zhetrd_hb2st("N", "N", "U", 0, 0, a, 1, d, e,
                     c, 1, w, 0, &info);
        chkxer("ZHETRD_HB2ST", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 9;

        /* ZHBEVD */

        strcpy(xerbla_srnamt, "ZHBEVD");
        xerbla_infot = 1;
        zhbevd("/", "U", 0, 0, a, 1, x, z, 1, w, 1, rw, 1, iw, 1,
               &info);
        chkxer("ZHBEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        zhbevd("N", "/", 0, 0, a, 1, x, z, 1, w, 1, rw, 1, iw, 1,
               &info);
        chkxer("ZHBEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        zhbevd("N", "U", -1, 0, a, 1, x, z, 1, w, 1, rw, 1, iw,
               1, &info);
        chkxer("ZHBEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        zhbevd("N", "U", 0, -1, a, 1, x, z, 1, w, 1, rw, 1, iw,
               1, &info);
        chkxer("ZHBEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        zhbevd("N", "U", 2, 1, a, 1, x, z, 1, w, 2, rw, 2, iw, 1,
               &info);
        chkxer("ZHBEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        zhbevd("V", "U", 2, 1, a, 2, x, z, 1, w, 8, rw, 25, iw,
               12, &info);
        chkxer("ZHBEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 11;
        zhbevd("N", "U", 1, 0, a, 1, x, z, 1, w, 0, rw, 1, iw, 1,
               &info);
        chkxer("ZHBEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 11;
        zhbevd("N", "U", 2, 1, a, 2, x, z, 2, w, 1, rw, 2, iw, 1,
               &info);
        chkxer("ZHBEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 11;
        zhbevd("V", "U", 2, 1, a, 2, x, z, 2, w, 2, rw, 25, iw,
               12, &info);
        chkxer("ZHBEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 13;
        zhbevd("N", "U", 1, 0, a, 1, x, z, 1, w, 1, rw, 0, iw, 1,
               &info);
        chkxer("ZHBEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 13;
        zhbevd("N", "U", 2, 1, a, 2, x, z, 2, w, 2, rw, 1, iw, 1,
               &info);
        chkxer("ZHBEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 13;
        zhbevd("V", "U", 2, 1, a, 2, x, z, 2, w, 8, rw, 2, iw,
               12, &info);
        chkxer("ZHBEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 15;
        zhbevd("N", "U", 1, 0, a, 1, x, z, 1, w, 1, rw, 1, iw, 0,
               &info);
        chkxer("ZHBEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 15;
        zhbevd("N", "U", 2, 1, a, 2, x, z, 2, w, 2, rw, 2, iw, 0,
               &info);
        chkxer("ZHBEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 15;
        zhbevd("V", "U", 2, 1, a, 2, x, z, 2, w, 8, rw, 25, iw,
               2, &info);
        chkxer("ZHBEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 15;

        /* ZHBEVD_2STAGE */

        strcpy(xerbla_srnamt, "ZHBEVD_2STAGE");
        xerbla_infot = 1;
        zhbevd_2stage("/", "U", 0, 0, a, 1, x, z, 1,
                      w, 1, rw, 1, iw, 1, &info);
        chkxer("ZHBEVD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 1;
        zhbevd_2stage("V", "U", 0, 0, a, 1, x, z, 1,
                      w, 1, rw, 1, iw, 1, &info);
        chkxer("ZHBEVD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        zhbevd_2stage("N", "/", 0, 0, a, 1, x, z, 1,
                      w, 1, rw, 1, iw, 1, &info);
        chkxer("ZHBEVD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        zhbevd_2stage("N", "U", -1, 0, a, 1, x, z, 1,
                      w, 1, rw, 1, iw, 1, &info);
        chkxer("ZHBEVD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        zhbevd_2stage("N", "U", 0, -1, a, 1, x, z, 1,
                      w, 1, rw, 1, iw, 1, &info);
        chkxer("ZHBEVD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        zhbevd_2stage("N", "U", 2, 1, a, 1, x, z, 1,
                      w, 2, rw, 2, iw, 1, &info);
        chkxer("ZHBEVD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        zhbevd_2stage("N", "U", 2, 1, a, 2, x, z, 0,
                      w, 8, rw, 25, iw, 12, &info);
        chkxer("ZHBEVD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 11;
        zhbevd_2stage("N", "U", 1, 0, a, 1, x, z, 1,
                      w, 0, rw, 1, iw, 1, &info);
        chkxer("ZHBEVD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 11;
        zhbevd_2stage("N", "U", 2, 1, a, 2, x, z, 2,
                      w, 1, rw, 2, iw, 1, &info);
        chkxer("ZHBEVD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        /* xerbla_infot = 11;
         * zhbevd_2stage("V", "U", 2, 1, a, 2, x, z, 2,
         *               w, 2, rw, 25, iw, 12, &info);
         * chkxer("ZHBEVD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok); */
        xerbla_infot = 13;
        zhbevd_2stage("N", "U", 1, 0, a, 1, x, z, 1,
                      w, 1, rw, 0, iw, 1, &info);
        chkxer("ZHBEVD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 13;
        zhbevd_2stage("N", "U", 2, 1, a, 2, x, z, 2,
                      w, 25, rw, 1, iw, 1, &info);
        chkxer("ZHBEVD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        /* xerbla_infot = 13;
         * zhbevd_2stage("V", "U", 2, 1, a, 2, x, z, 2,
         *               w, 25, rw, 2, iw, 12, &info);
         * chkxer("ZHBEVD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok); */
        xerbla_infot = 15;
        zhbevd_2stage("N", "U", 1, 0, a, 1, x, z, 1,
                      w, 1, rw, 1, iw, 0, &info);
        chkxer("ZHBEVD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 15;
        zhbevd_2stage("N", "U", 2, 1, a, 2, x, z, 2,
                      w, 25, rw, 2, iw, 0, &info);
        chkxer("ZHBEVD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        /* xerbla_infot = 15;
         * zhbevd_2stage("V", "U", 2, 1, a, 2, x, z, 2,
         *               w, 25, rw, 25, iw, 2, &info);
         * chkxer("ZHBEVD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok); */
        *nt = *nt + 13;

        /* ZHBEV */

        strcpy(xerbla_srnamt, "ZHBEV ");
        xerbla_infot = 1;
        zhbev("/", "U", 0, 0, a, 1, x, z, 1, w, rw, &info);
        chkxer("ZHBEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        zhbev("N", "/", 0, 0, a, 1, x, z, 1, w, rw, &info);
        chkxer("ZHBEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        zhbev("N", "U", -1, 0, a, 1, x, z, 1, w, rw, &info);
        chkxer("ZHBEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        zhbev("N", "U", 0, -1, a, 1, x, z, 1, w, rw, &info);
        chkxer("ZHBEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        zhbev("N", "U", 2, 1, a, 1, x, z, 1, w, rw, &info);
        chkxer("ZHBEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        zhbev("V", "U", 2, 0, a, 1, x, z, 1, w, rw, &info);
        chkxer("ZHBEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 6;

        /* ZHBEV_2STAGE */

        strcpy(xerbla_srnamt, "ZHBEV_2STAGE ");
        xerbla_infot = 1;
        zhbev_2stage("/", "U", 0, 0, a, 1, x,
                     z, 1, w, 0, rw, &info);
        chkxer("ZHBEV_2STAGE ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 1;
        zhbev_2stage("V", "U", 0, 0, a, 1, x,
                     z, 1, w, 0, rw, &info);
        chkxer("ZHBEV_2STAGE ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        zhbev_2stage("N", "/", 0, 0, a, 1, x,
                     z, 1, w, 0, rw, &info);
        chkxer("ZHBEV_2STAGE ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        zhbev_2stage("N", "U", -1, 0, a, 1, x,
                     z, 1, w, 0, rw, &info);
        chkxer("ZHBEV_2STAGE ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        zhbev_2stage("N", "U", 0, -1, a, 1, x,
                     z, 1, w, 0, rw, &info);
        chkxer("ZHBEV_2STAGE ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        zhbev_2stage("N", "U", 2, 1, a, 1, x,
                     z, 1, w, 0, rw, &info);
        chkxer("ZHBEV_2STAGE ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        zhbev_2stage("N", "U", 2, 0, a, 1, x,
                     z, 0, w, 0, rw, &info);
        chkxer("ZHBEV_2STAGE ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 11;
        zhbev_2stage("N", "U", 2, 0, a, 1, x,
                     z, 1, w, 0, rw, &info);
        chkxer("ZHBEV_2STAGE ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 8;

        /* ZHBEVX */

        strcpy(xerbla_srnamt, "ZHBEVX");
        xerbla_infot = 1;
        zhbevx("/", "A", "U", 0, 0, a, 1, q, 1, 0.0, 0.0, -1,
               -1, 0.0, &m, x, z, 1, w, rw, iw, i3, &info);
        chkxer("ZHBEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        zhbevx("V", "/", "U", 0, 0, a, 1, q, 1, 0.0, 1.0, 0,
               -1, 0.0, &m, x, z, 1, w, rw, iw, i3, &info);
        chkxer("ZHBEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        zhbevx("V", "A", "/", 0, 0, a, 1, q, 1, 0.0, 0.0, -1,
               -1, 0.0, &m, x, z, 1, w, rw, iw, i3, &info);
        xerbla_infot = 4;
        zhbevx("V", "A", "U", -1, 0, a, 1, q, 1, 0.0, 0.0, -1,
               -1, 0.0, &m, x, z, 1, w, rw, iw, i3, &info);
        chkxer("ZHBEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        zhbevx("V", "A", "U", 0, -1, a, 1, q, 1, 0.0, 0.0, -1,
               -1, 0.0, &m, x, z, 1, w, rw, iw, i3, &info);
        chkxer("ZHBEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        zhbevx("V", "A", "U", 2, 1, a, 1, q, 2, 0.0, 0.0, -1,
               -1, 0.0, &m, x, z, 2, w, rw, iw, i3, &info);
        chkxer("ZHBEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        zhbevx("V", "A", "U", 2, 0, a, 1, q, 1, 0.0, 0.0, -1,
               -1, 0.0, &m, x, z, 2, w, rw, iw, i3, &info);
        chkxer("ZHBEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 11;
        zhbevx("V", "V", "U", 1, 0, a, 1, q, 1, 0.0, 0.0, -1,
               -1, 0.0, &m, x, z, 1, w, rw, iw, i3, &info);
        chkxer("ZHBEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 12;
        zhbevx("V", "I", "U", 1, 0, a, 1, q, 1, 0.0, 0.0, -1,
               -1, 0.0, &m, x, z, 1, w, rw, iw, i3, &info);
        chkxer("ZHBEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 13;
        zhbevx("V", "I", "U", 1, 0, a, 1, q, 1, 0.0, 0.0, 0,
               1, 0.0, &m, x, z, 1, w, rw, iw, i3, &info);
        chkxer("ZHBEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 18;
        zhbevx("V", "A", "U", 2, 0, a, 1, q, 2, 0.0, 0.0, -1,
               -1, 0.0, &m, x, z, 1, w, rw, iw, i3, &info);
        chkxer("ZHBEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 11;

        /* ZHBEVX_2STAGE */

        strcpy(xerbla_srnamt, "ZHBEVX_2STAGE");
        xerbla_infot = 1;
        zhbevx_2stage("/", "A", "U", 0, 0, a, 1, q, 1,
                      0.0, 0.0, -1, -1, 0.0,
                      &m, x, z, 1, w, 0, rw, iw, i3, &info);
        xerbla_infot = 1;
        zhbevx_2stage("V", "A", "U", 0, 0, a, 1, q, 1,
                      0.0, 0.0, -1, -1, 0.0,
                      &m, x, z, 1, w, 0, rw, iw, i3, &info);
        chkxer("ZHBEVX_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        zhbevx_2stage("N", "/", "U", 0, 0, a, 1, q, 1,
                      0.0, 1.0, 0, -1, 0.0,
                      &m, x, z, 1, w, 0, rw, iw, i3, &info);
        chkxer("ZHBEVX_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        zhbevx_2stage("N", "A", "/", 0, 0, a, 1, q, 1,
                      0.0, 0.0, -1, -1, 0.0,
                      &m, x, z, 1, w, 0, rw, iw, i3, &info);
        xerbla_infot = 4;
        zhbevx_2stage("N", "A", "U", -1, 0, a, 1, q, 1,
                      0.0, 0.0, -1, -1, 0.0,
                      &m, x, z, 1, w, 0, rw, iw, i3, &info);
        chkxer("ZHBEVX_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        zhbevx_2stage("N", "A", "U", 0, -1, a, 1, q, 1,
                      0.0, 0.0, -1, -1, 0.0,
                      &m, x, z, 1, w, 0, rw, iw, i3, &info);
        chkxer("ZHBEVX_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        zhbevx_2stage("N", "A", "U", 2, 1, a, 1, q, 2,
                      0.0, 0.0, -1, -1, 0.0,
                      &m, x, z, 2, w, 0, rw, iw, i3, &info);
        chkxer("ZHBEVX_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        /* xerbla_infot = 9;
         * zhbevx_2stage("V", "A", "U", 2, 0, a, 1, q, 1,
         *               0.0, 0.0, -1, -1, 0.0,
         *               &m, x, z, 2, w, 0, rw, iw, i3, &info);
         * chkxer("ZHBEVX_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok); */
        xerbla_infot = 11;
        zhbevx_2stage("N", "V", "U", 1, 0, a, 1, q, 1,
                      0.0, 0.0, -1, -1, 0.0,
                      &m, x, z, 1, w, 0, rw, iw, i3, &info);
        chkxer("ZHBEVX_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 12;
        zhbevx_2stage("N", "I", "U", 1, 0, a, 1, q, 1,
                      0.0, 0.0, -1, -1, 0.0,
                      &m, x, z, 1, w, 0, rw, iw, i3, &info);
        chkxer("ZHBEVX_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 13;
        zhbevx_2stage("N", "I", "U", 1, 0, a, 1, q, 1,
                      0.0, 0.0, 0, 1, 0.0,
                      &m, x, z, 1, w, 0, rw, iw, i3, &info);
        chkxer("ZHBEVX_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 18;
        zhbevx_2stage("N", "A", "U", 2, 0, a, 1, q, 2,
                      0.0, 0.0, -1, -1, 0.0,
                      &m, x, z, 0, w, 0, rw, iw, i3, &info);
        chkxer("ZHBEVX_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 20;
        zhbevx_2stage("N", "A", "U", 2, 0, a, 1, q, 2,
                      0.0, 0.0, -1, -1, 0.0,
                      &m, x, z, 1, w, 0, rw, iw, i3, &info);
        chkxer("ZHBEVX_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 12;
    }

    *ok = xerbla_ok;

    xerbla_srnamt[0] = '\0';
    xerbla_infot = 0;
}
