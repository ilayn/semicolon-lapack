/**
 * @file serrst.c
 * @brief SERRST tests the error exits for the symmetric eigenvalue routines.
 *
 * Port of LAPACK TESTING/EIG/serrst.f
 * Tests error exits for: SSYTRD, SSYTD2, SORGTR, SORMTR, SSPTRD,
 * SOPGTR, SOPMTR, SSTEQR, SSTERF, SSTEBZ, SSTEIN, SPTEQR, SSBTRD,
 * SSYEV, SSYEVX, SSYEVD, SSBEV, SSBEVX, SSBEVD,
 * SSPEV, SSPEVX, SSPEVD, SSTEV, SSTEVX, SSTEVD, and SSTEDC.
 * SSYEVD_2STAGE, SSYEVR_2STAGE, SSYEVX_2STAGE,
 * SSYEV_2STAGE, SSBEV_2STAGE, SSBEVD_2STAGE,
 * SSBEVX_2STAGE, SSYTRD_2STAGE, SSYTRD_SY2SB,
 * SSYTRD_SB2ST
 */

#include <math.h>
#include <string.h>
#include "semicolon_lapack_single.h"
#include "verify.h"

void serrst(const char* path, INT* ok, INT* nt)
{
    /* NMAX has to be at least 3 or LIW may be too small */
    enum { NMAX = 3, LIW = 12 * NMAX, LW = 20 * NMAX };

    INT i, info, j, m, n, nsplit;

    INT i1[NMAX], i2[NMAX], i3[NMAX], iw[LIW];
    f32 a[NMAX * NMAX], c[NMAX * NMAX], d[NMAX], e[NMAX], q[NMAX * NMAX];
    f32 r[NMAX], tau[NMAX], w[LW], x[NMAX], z[NMAX * NMAX];
    f32 rinf, rzero;

    /* Set the variables to innocuous values. */
    for (j = 0; j < NMAX; j++) {
        for (i = 0; i < NMAX; i++) {
            a[i + j * NMAX] = 1.0f / (f32)(i + j + 2);
        }
    }
    for (j = 0; j < NMAX; j++) {
        d[j] = (f32)(j + 1);
        e[j] = 0.0f;
        i1[j] = j;
        i2[j] = j;
        tau[j] = 1.0f;
    }
    xerbla_ok = 1;
    *nt = 0;

    /* Test error exits for the ST path. */
    if ((path[1] == 'S' || path[1] == 's') &&
        (path[2] == 'T' || path[2] == 't')) {

        /* SSYTRD */

        strcpy(xerbla_srnamt, "SSYTRD");
        xerbla_infot = 1;
        ssytrd("/", 0, a, 1, d, e, tau, w, 1, &info);
        chkxer("SSYTRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        ssytrd("U", -1, a, 1, d, e, tau, w, 1, &info);
        chkxer("SSYTRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        ssytrd("U", 2, a, 1, d, e, tau, w, 1, &info);
        chkxer("SSYTRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        ssytrd("U", 0, a, 1, d, e, tau, w, 0, &info);
        chkxer("SSYTRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 4;

        /* SSYTD2 */

        strcpy(xerbla_srnamt, "SSYTD2");
        xerbla_infot = 1;
        ssytd2("/", 0, a, 1, d, e, tau, &info);
        chkxer("SSYTD2", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        ssytd2("U", -1, a, 1, d, e, tau, &info);
        chkxer("SSYTD2", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        ssytd2("U", 2, a, 1, d, e, tau, &info);
        chkxer("SSYTD2", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 3;

        /* SSYTRD_2STAGE */

        strcpy(xerbla_srnamt, "SSYTRD_2STAGE");
        xerbla_infot = 1;
        ssytrd_2stage("/", "U", 0, a, 1, d, e, tau,
                      c, 1, w, 1, &info);
        chkxer("SSYTRD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 1;
        ssytrd_2stage("H", "U", 0, a, 1, d, e, tau,
                      c, 1, w, 1, &info);
        chkxer("SSYTRD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        ssytrd_2stage("N", "/", 0, a, 1, d, e, tau,
                      c, 1, w, 1, &info);
        chkxer("SSYTRD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        ssytrd_2stage("N", "U", -1, a, 1, d, e, tau,
                      c, 1, w, 1, &info);
        chkxer("SSYTRD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        ssytrd_2stage("N", "U", 2, a, 1, d, e, tau,
                      c, 1, w, 1, &info);
        chkxer("SSYTRD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        ssytrd_2stage("N", "U", 0, a, 1, d, e, tau,
                      c, 0, w, 1, &info);
        chkxer("SSYTRD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 12;
        ssytrd_2stage("N", "U", 0, a, 1, d, e, tau,
                      c, 1, w, 0, &info);
        chkxer("SSYTRD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 7;

        /* SSYTRD_SY2SB */

        strcpy(xerbla_srnamt, "SSYTRD_SY2SB");
        xerbla_infot = 1;
        ssytrd_sy2sb("/", 0, 0, a, 1, c, 1, tau, w, 1, &info);
        chkxer("SSYTRD_SY2SB", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        ssytrd_sy2sb("U", -1, 0, a, 1, c, 1, tau, w, 1, &info);
        chkxer("SSYTRD_SY2SB", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        ssytrd_sy2sb("U", 0, -1, a, 1, c, 1, tau, w, 1, &info);
        chkxer("SSYTRD_SY2SB", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        ssytrd_sy2sb("U", 2, 0, a, 1, c, 1, tau, w, 1, &info);
        chkxer("SSYTRD_SY2SB", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        ssytrd_sy2sb("U", 0, 2, a, 1, c, 1, tau, w, 1, &info);
        chkxer("SSYTRD_SY2SB", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        ssytrd_sy2sb("U", 0, 0, a, 1, c, 1, tau, w, 0, &info);
        chkxer("SSYTRD_SY2SB", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 6;

        /* SSYTRD_SB2ST */

        strcpy(xerbla_srnamt, "SSYTRD_SB2ST");
        xerbla_infot = 1;
        ssytrd_sb2st("/", "N", "U", 0, 0, a, 1, d, e,
                     c, 1, w, 1, &info);
        chkxer("SSYTRD_SB2ST", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        ssytrd_sb2st("Y", "/", "U", 0, 0, a, 1, d, e,
                     c, 1, w, 1, &info);
        chkxer("SSYTRD_SB2ST", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        ssytrd_sb2st("Y", "H", "U", 0, 0, a, 1, d, e,
                     c, 1, w, 1, &info);
        chkxer("SSYTRD_SB2ST", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        ssytrd_sb2st("Y", "N", "/", 0, 0, a, 1, d, e,
                     c, 1, w, 1, &info);
        chkxer("SSYTRD_SB2ST", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        ssytrd_sb2st("Y", "N", "U", -1, 0, a, 1, d, e,
                     c, 1, w, 1, &info);
        chkxer("SSYTRD_SB2ST", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        ssytrd_sb2st("Y", "N", "U", 0, -1, a, 1, d, e,
                     c, 1, w, 1, &info);
        chkxer("SSYTRD_SB2ST", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        ssytrd_sb2st("Y", "N", "U", 0, 1, a, 1, d, e,
                     c, 1, w, 1, &info);
        chkxer("SSYTRD_SB2ST", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 11;
        ssytrd_sb2st("Y", "N", "U", 0, 0, a, 1, d, e,
                     c, 0, w, 1, &info);
        chkxer("SSYTRD_SB2ST", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 13;
        ssytrd_sb2st("Y", "N", "U", 0, 0, a, 1, d, e,
                     c, 1, w, 0, &info);
        chkxer("SSYTRD_SB2ST", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 9;

        /* SORGTR */

        strcpy(xerbla_srnamt, "SORGTR");
        xerbla_infot = 1;
        sorgtr("/", 0, a, 1, tau, w, 1, &info);
        chkxer("SORGTR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        sorgtr("U", -1, a, 1, tau, w, 1, &info);
        chkxer("SORGTR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        sorgtr("U", 2, a, 1, tau, w, 1, &info);
        chkxer("SORGTR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        sorgtr("U", 3, a, 3, tau, w, 1, &info);
        chkxer("SORGTR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 4;

        /* SORMTR */

        strcpy(xerbla_srnamt, "SORMTR");
        xerbla_infot = 1;
        sormtr("/", "U", "N", 0, 0, a, 1, tau, c, 1, w, 1, &info);
        chkxer("SORMTR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        sormtr("L", "/", "N", 0, 0, a, 1, tau, c, 1, w, 1, &info);
        chkxer("SORMTR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        sormtr("L", "U", "/", 0, 0, a, 1, tau, c, 1, w, 1, &info);
        chkxer("SORMTR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        sormtr("L", "U", "N", -1, 0, a, 1, tau, c, 1, w, 1,
               &info);
        chkxer("SORMTR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        sormtr("L", "U", "N", 0, -1, a, 1, tau, c, 1, w, 1,
               &info);
        chkxer("SORMTR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        sormtr("L", "U", "N", 2, 0, a, 1, tau, c, 2, w, 1, &info);
        chkxer("SORMTR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        sormtr("R", "U", "N", 0, 2, a, 1, tau, c, 1, w, 1, &info);
        chkxer("SORMTR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        sormtr("L", "U", "N", 2, 0, a, 2, tau, c, 1, w, 1, &info);
        chkxer("SORMTR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 12;
        sormtr("L", "U", "N", 0, 2, a, 1, tau, c, 1, w, 1, &info);
        chkxer("SORMTR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 12;
        sormtr("R", "U", "N", 2, 0, a, 1, tau, c, 2, w, 1, &info);
        chkxer("SORMTR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 10;

        /* SSPTRD */

        strcpy(xerbla_srnamt, "SSPTRD");
        xerbla_infot = 1;
        ssptrd("/", 0, a, d, e, tau, &info);
        chkxer("SSPTRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        ssptrd("U", -1, a, d, e, tau, &info);
        chkxer("SSPTRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 2;

        /* SOPGTR */

        strcpy(xerbla_srnamt, "SOPGTR");
        xerbla_infot = 1;
        sopgtr("/", 0, a, tau, z, 1, w, &info);
        chkxer("SOPGTR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        sopgtr("U", -1, a, tau, z, 1, w, &info);
        chkxer("SOPGTR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        sopgtr("U", 2, a, tau, z, 1, w, &info);
        chkxer("SOPGTR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 3;

        /* SOPMTR */

        strcpy(xerbla_srnamt, "SOPMTR");
        xerbla_infot = 1;
        sopmtr("/", "U", "N", 0, 0, a, tau, c, 1, w, &info);
        chkxer("SOPMTR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        sopmtr("L", "/", "N", 0, 0, a, tau, c, 1, w, &info);
        chkxer("SOPMTR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        sopmtr("L", "U", "/", 0, 0, a, tau, c, 1, w, &info);
        chkxer("SOPMTR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        sopmtr("L", "U", "N", -1, 0, a, tau, c, 1, w, &info);
        chkxer("SOPMTR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        sopmtr("L", "U", "N", 0, -1, a, tau, c, 1, w, &info);
        chkxer("SOPMTR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        sopmtr("L", "U", "N", 2, 0, a, tau, c, 1, w, &info);
        chkxer("SOPMTR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 6;

        /* SPTEQR */

        strcpy(xerbla_srnamt, "SPTEQR");
        xerbla_infot = 1;
        spteqr("/", 0, d, e, z, 1, w, &info);
        chkxer("SPTEQR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        spteqr("N", -1, d, e, z, 1, w, &info);
        chkxer("SPTEQR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        spteqr("V", 2, d, e, z, 1, w, &info);
        chkxer("SPTEQR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 3;

        /* SSTEBZ */

        strcpy(xerbla_srnamt, "SSTEBZ");
        xerbla_infot = 1;
        sstebz("/", "E", 0, 0.0f, 1.0f, 0, -1, 0.0f, d, e, &m,
               &nsplit, x, i1, i2, w, iw, &info);
        chkxer("SSTEBZ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        sstebz("A", "/", 0, 0.0f, 0.0f, -1, -1, 0.0f, d, e, &m,
               &nsplit, x, i1, i2, w, iw, &info);
        chkxer("SSTEBZ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        sstebz("A", "E", -1, 0.0f, 0.0f, -1, -1, 0.0f, d, e, &m,
               &nsplit, x, i1, i2, w, iw, &info);
        chkxer("SSTEBZ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        sstebz("V", "E", 0, 0.0f, 0.0f, -1, -1, 0.0f, d, e, &m,
               &nsplit, x, i1, i2, w, iw, &info);
        chkxer("SSTEBZ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        sstebz("I", "E", 0, 0.0f, 0.0f, -1, -1, 0.0f, d, e, &m,
               &nsplit, x, i1, i2, w, iw, &info);
        chkxer("SSTEBZ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        sstebz("I", "E", 1, 0.0f, 0.0f, 1, 0, 0.0f, d, e, &m,
               &nsplit, x, i1, i2, w, iw, &info);
        chkxer("SSTEBZ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        sstebz("I", "E", 1, 0.0f, 0.0f, 0, -1, 0.0f, d, e, &m,
               &nsplit, x, i1, i2, w, iw, &info);
        chkxer("SSTEBZ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        sstebz("I", "E", 1, 0.0f, 0.0f, 0, 1, 0.0f, d, e, &m,
               &nsplit, x, i1, i2, w, iw, &info);
        chkxer("SSTEBZ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 8;

        /* SSTEIN */

        strcpy(xerbla_srnamt, "SSTEIN");
        xerbla_infot = 1;
        sstein(-1, d, e, 0, x, i1, i2, z, 1, w, iw, i3, &info);
        chkxer("SSTEIN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        sstein(0, d, e, -1, x, i1, i2, z, 1, w, iw, i3, &info);
        chkxer("SSTEIN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        sstein(0, d, e, 1, x, i1, i2, z, 1, w, iw, i3, &info);
        chkxer("SSTEIN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        sstein(2, d, e, 0, x, i1, i2, z, 1, w, iw, i3, &info);
        chkxer("SSTEIN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 4;

        /* A matrix that holds an infinity must come back from the
           drivers that scale it, instead of reaching XERBLA with a
           scale factor of zero. */

        rzero = 0.0f;
        rinf = 1.0f / rzero;
        for (j = 0; j < NMAX; j++) {
            d[j] = (f32)(j + 1);
            e[j] = 1.0f / (f32)(j + 2);
            for (i = 0; i < NMAX; i++) {
                a[i + j * NMAX] = rzero;
            }
            a[j + j * NMAX] = d[j];
        }
        d[0] = rinf;
        a[0 + 0 * NMAX] = rinf;
        e[NMAX - 1] = rzero;
        a[0 + 1 * NMAX] = e[0];
        a[1 + 2 * NMAX] = e[1];
        strcpy(xerbla_srnamt, "SSTEVX");
        xerbla_infot = 0;
        xerbla_lerr = 0;
        sstevx("V", "V", NMAX, d, e, rzero, 1.0f, 0, NMAX - 1,
               rzero, &m, x, z, NMAX, w, iw, i3, &info);
        if (xerbla_lerr) {
            xerbla_ok = 0;
        }
        d[0] = rinf;
        e[0] = 1.0f / (f32)2;
        e[1] = 1.0f / (f32)3;
        e[NMAX - 1] = rzero;
        strcpy(xerbla_srnamt, "SSYEVX");
        xerbla_infot = 0;
        xerbla_lerr = 0;
        ssyevx("V", "V", "U", NMAX, a, NMAX, rzero, 1.0f, 0,
               NMAX - 1, rzero, &m, x, z, NMAX, w, LW, iw, i3, &info);
        if (xerbla_lerr) {
            xerbla_ok = 0;
        }
        *nt = *nt + 2;

        /* SSTEQR */

        strcpy(xerbla_srnamt, "SSTEQR");
        xerbla_infot = 1;
        ssteqr("/", 0, d, e, z, 1, w, &info);
        chkxer("SSTEQR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        ssteqr("N", -1, d, e, z, 1, w, &info);
        chkxer("SSTEQR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        ssteqr("V", 2, d, e, z, 1, w, &info);
        chkxer("SSTEQR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 3;

        /* SSTERF */

        strcpy(xerbla_srnamt, "SSTERF");
        xerbla_infot = 1;
        ssterf(-1, d, e, &info);
        chkxer("SSTERF", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 1;

        /* SSTEDC */

        strcpy(xerbla_srnamt, "SSTEDC");
        xerbla_infot = 1;
        sstedc("/", 0, d, e, z, 1, w, 1, iw, 1, &info);
        chkxer("SSTEDC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        sstedc("N", -1, d, e, z, 1, w, 1, iw, 1, &info);
        chkxer("SSTEDC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        sstedc("V", 2, d, e, z, 1, w, 23, iw, 28, &info);
        chkxer("SSTEDC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        sstedc("N", 1, d, e, z, 1, w, 0, iw, 1, &info);
        chkxer("SSTEDC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        sstedc("I", 2, d, e, z, 2, w, 0, iw, 12, &info);
        chkxer("SSTEDC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        sstedc("V", 2, d, e, z, 2, w, 0, iw, 28, &info);
        chkxer("SSTEDC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        sstedc("N", 1, d, e, z, 1, w, 1, iw, 0, &info);
        chkxer("SSTEDC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        sstedc("I", 2, d, e, z, 2, w, 19, iw, 0, &info);
        chkxer("SSTEDC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        sstedc("V", 2, d, e, z, 2, w, 23, iw, 0, &info);
        chkxer("SSTEDC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 9;

        /* SSTEVD */

        strcpy(xerbla_srnamt, "SSTEVD");
        xerbla_infot = 1;
        sstevd("/", 0, d, e, z, 1, w, 1, iw, 1, &info);
        chkxer("SSTEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        sstevd("N", -1, d, e, z, 1, w, 1, iw, 1, &info);
        chkxer("SSTEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        sstevd("V", 2, d, e, z, 1, w, 19, iw, 12, &info);
        chkxer("SSTEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        sstevd("N", 1, d, e, z, 1, w, 0, iw, 1, &info);
        chkxer("SSTEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        sstevd("V", 2, d, e, z, 2, w, 12, iw, 12, &info);
        chkxer("SSTEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        sstevd("N", 0, d, e, z, 1, w, 1, iw, 0, &info);
        chkxer("SSTEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        sstevd("V", 2, d, e, z, 2, w, 19, iw, 11, &info);
        chkxer("SSTEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 7;

        /* SSTEV */

        strcpy(xerbla_srnamt, "SSTEV ");
        xerbla_infot = 1;
        sstev("/", 0, d, e, z, 1, w, &info);
        chkxer("SSTEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        sstev("N", -1, d, e, z, 1, w, &info);
        chkxer("SSTEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        sstev("V", 2, d, e, z, 1, w, &info);
        chkxer("SSTEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 3;

        /* SSTEVX */

        strcpy(xerbla_srnamt, "SSTEVX");
        xerbla_infot = 1;
        sstevx("/", "A", 0, d, e, 0.0f, 0.0f, -1, -1, 0.0f, &m,
               x, z, 1, w, iw, i3, &info);
        chkxer("SSTEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        sstevx("N", "/", 0, d, e, 0.0f, 1.0f, 0, -1, 0.0f, &m,
               x, z, 1, w, iw, i3, &info);
        chkxer("SSTEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        sstevx("N", "A", -1, d, e, 0.0f, 0.0f, -1, -1, 0.0f, &m,
               x, z, 1, w, iw, i3, &info);
        chkxer("SSTEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        sstevx("N", "V", 1, d, e, 0.0f, 0.0f, -1, -1, 0.0f, &m,
               x, z, 1, w, iw, i3, &info);
        chkxer("SSTEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        sstevx("N", "I", 1, d, e, 0.0f, 0.0f, -1, -1, 0.0f, &m,
               x, z, 1, w, iw, i3, &info);
        chkxer("SSTEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        sstevx("N", "I", 1, d, e, 0.0f, 0.0f, 1, 0, 0.0f, &m,
               x, z, 1, w, iw, i3, &info);
        chkxer("SSTEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        sstevx("N", "I", 2, d, e, 0.0f, 0.0f, 1, 0, 0.0f, &m,
               x, z, 1, w, iw, i3, &info);
        chkxer("SSTEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        sstevx("N", "I", 1, d, e, 0.0f, 0.0f, 0, 1, 0.0f, &m,
               x, z, 1, w, iw, i3, &info);
        chkxer("SSTEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 14;
        sstevx("V", "A", 2, d, e, 0.0f, 0.0f, -1, -1, 0.0f, &m,
               x, z, 1, w, iw, i3, &info);
        chkxer("SSTEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 9;

        /* SSTEVR */

        n = 1;
        strcpy(xerbla_srnamt, "SSTEVR");
        xerbla_infot = 1;
        sstevr("/", "A", 0, d, e, 0.0f, 0.0f, 0, 0, 0.0f, &m,
               r, z, 1, iw, x, 20 * n, &iw[2 * n], 10 * n, &info);
        chkxer("SSTEVR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        sstevr("V", "/", 0, d, e, 0.0f, 0.0f, 0, 0, 0.0f, &m,
               r, z, 1, iw, x, 20 * n, &iw[2 * n], 10 * n, &info);
        chkxer("SSTEVR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        sstevr("V", "A", -1, d, e, 0.0f, 0.0f, 0, 0, 0.0f, &m,
               r, z, 1, iw, x, 20 * n, &iw[2 * n], 10 * n, &info);
        chkxer("SSTEVR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        sstevr("V", "V", 1, d, e, 0.0f, 0.0f, 0, 0, 0.0f, &m,
               r, z, 1, iw, x, 20 * n, &iw[2 * n], 10 * n, &info);
        chkxer("SSTEVR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        sstevr("V", "I", 1, d, e, 0.0f, 0.0f, -1, 0, 0.0f, &m,
               w, z, 1, iw, x, 20 * n, &iw[2 * n], 10 * n, &info);
        chkxer("SSTEVR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        n = 2;
        sstevr("V", "I", 2, d, e, 0.0f, 0.0f, 1, 0, 0.0f, &m,
               w, z, 1, iw, x, 20 * n, &iw[2 * n], 10 * n, &info);
        chkxer("SSTEVR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 14;
        n = 1;
        sstevr("V", "I", 1, d, e, 0.0f, 0.0f, 0, 0, 0.0f, &m,
               w, z, 0, iw, x, 20 * n, &iw[2 * n], 10 * n, &info);
        chkxer("SSTEVR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 17;
        sstevr("V", "I", 1, d, e, 0.0f, 0.0f, 0, 0, 0.0f, &m,
               w, z, 1, iw, x, 20 * n - 1, &iw[2 * n], 10 * n, &info);
        chkxer("SSTEVR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 19;
        sstevr("V", "I", 1, d, e, 0.0f, 0.0f, 0, 0, 0.0f, &m,
               w, z, 1, iw, x, 20 * n, &iw[2 * n], 10 * n - 1, &info);
        chkxer("SSTEVR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 9;

        /* SSYEVD */

        strcpy(xerbla_srnamt, "SSYEVD");
        xerbla_infot = 1;
        ssyevd("/", "U", 0, a, 1, x, w, 1, iw, 1, &info);
        chkxer("SSYEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        ssyevd("N", "/", 0, a, 1, x, w, 1, iw, 1, &info);
        chkxer("SSYEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        ssyevd("N", "U", -1, a, 1, x, w, 1, iw, 1, &info);
        chkxer("SSYEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        ssyevd("N", "U", 2, a, 1, x, w, 3, iw, 1, &info);
        chkxer("SSYEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        ssyevd("N", "U", 1, a, 1, x, w, 0, iw, 1, &info);
        chkxer("SSYEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        ssyevd("N", "U", 2, a, 2, x, w, 4, iw, 1, &info);
        chkxer("SSYEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        ssyevd("V", "U", 2, a, 2, x, w, 20, iw, 12, &info);
        chkxer("SSYEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        ssyevd("N", "U", 1, a, 1, x, w, 1, iw, 0, &info);
        chkxer("SSYEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        ssyevd("N", "U", 2, a, 2, x, w, 5, iw, 0, &info);
        chkxer("SSYEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        ssyevd("V", "U", 2, a, 2, x, w, 27, iw, 11, &info);
        chkxer("SSYEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 10;

        /* SSYEVD_2STAGE */

        strcpy(xerbla_srnamt, "SSYEVD_2STAGE");
        xerbla_infot = 1;
        ssyevd_2stage("/", "U", 0, a, 1, x, w, 1, iw, 1, &info);
        chkxer("SSYEVD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 1;
        ssyevd_2stage("V", "U", 0, a, 1, x, w, 1, iw, 1, &info);
        chkxer("SSYEVD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        ssyevd_2stage("N", "/", 0, a, 1, x, w, 1, iw, 1, &info);
        chkxer("SSYEVD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        ssyevd_2stage("N", "U", -1, a, 1, x, w, 1, iw, 1, &info);
        chkxer("SSYEVD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        ssyevd_2stage("N", "U", 2, a, 1, x, w, 3, iw, 1, &info);
        chkxer("SSYEVD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        ssyevd_2stage("N", "U", 1, a, 1, x, w, 0, iw, 1, &info);
        chkxer("SSYEVD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        ssyevd_2stage("N", "U", 2, a, 2, x, w, 4, iw, 1, &info);
        chkxer("SSYEVD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        /* xerbla_infot = 8;
           ssyevd_2stage("V", "U", 2, a, 2, x, w, 20, iw, 12, &info);
           chkxer("SSYEVD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok); */
        xerbla_infot = 10;
        ssyevd_2stage("N", "U", 1, a, 1, x, w, 1, iw, 0, &info);
        chkxer("SSYEVD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        ssyevd_2stage("N", "U", 2, a, 2, x, w, 25, iw, 0, &info);
        chkxer("SSYEVD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        /* xerbla_infot = 10;
           ssyevd_2stage("V", "U", 2, a, 2, x, w, 27, iw, 11, &info);
           chkxer("SSYEVD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok); */
        *nt = *nt + 9;

        /* SSYEVR */

        strcpy(xerbla_srnamt, "SSYEVR");
        n = 1;
        xerbla_infot = 1;
        ssyevr("/", "A", "U", 0, a, 1, 0.0f, 0.0f, 0, 0, 0.0f,
               &m, r, z, 1, iw, q, 26 * n, &iw[2 * n], 10 * n, &info);
        chkxer("SSYEVR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        ssyevr("V", "/", "U", 0, a, 1, 0.0f, 0.0f, 0, 0, 0.0f,
               &m, r, z, 1, iw, q, 26 * n, &iw[2 * n], 10 * n, &info);
        chkxer("SSYEVR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        ssyevr("V", "A", "/", -1, a, 1, 0.0f, 0.0f, 0, 0,
               0.0f, &m, r, z, 1, iw, q, 26 * n, &iw[2 * n], 10 * n,
               &info);
        chkxer("SSYEVR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        ssyevr("V", "A", "U", -1, a, 1, 0.0f, 0.0f, 0, 0,
               0.0f, &m, r, z, 1, iw, q, 26 * n, &iw[2 * n], 10 * n,
               &info);
        chkxer("SSYEVR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        ssyevr("V", "A", "U", 2, a, 1, 0.0f, 0.0f, 0, 0, 0.0f,
               &m, r, z, 1, iw, q, 26 * n, &iw[2 * n], 10 * n, &info);
        chkxer("SSYEVR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        ssyevr("V", "V", "U", 1, a, 1, 0.0f, 0.0f, 0, 0, 0.0f,
               &m, r, z, 1, iw, q, 26 * n, &iw[2 * n], 10 * n, &info);
        chkxer("SSYEVR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        ssyevr("V", "I", "U", 1, a, 1, 0.0f, 0.0f, -1, 0, 0.0f,
               &m, r, z, 1, iw, q, 26 * n, &iw[2 * n], 10 * n, &info);
        chkxer("SSYEVR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;

        ssyevr("V", "I", "U", 2, a, 2, 0.0f, 0.0f, 1, 0, 0.0f,
               &m, r, z, 1, iw, q, 26 * n, &iw[2 * n], 10 * n, &info);
        chkxer("SSYEVR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 15;
        ssyevr("V", "I", "U", 1, a, 1, 0.0f, 0.0f, 0, 0, 0.0f,
               &m, r, z, 0, iw, q, 26 * n, &iw[2 * n], 10 * n, &info);
        chkxer("SSYEVR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 18;
        ssyevr("V", "I", "U", 1, a, 1, 0.0f, 0.0f, 0, 0, 0.0f,
               &m, r, z, 1, iw, q, 0, &iw[2 * n], 10 * n,
               &info);
        chkxer("SSYEVR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 20;
        ssyevr("V", "I", "U", 1, a, 1, 0.0f, 0.0f, 0, 0, 0.0f,
               &m, r, z, 1, iw, q, 26 * n, &iw[2 * n], 0,
               &info);
        chkxer("SSYEVR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 11;

        /* SSYEVR_2STAGE */

        strcpy(xerbla_srnamt, "SSYEVR_2STAGE");
        n = 1;
        xerbla_infot = 1;
        ssyevr_2stage("/", "A", "U", 0, a, 1,
                      0.0f, 0.0f, 0, 0, 0.0f,
                      &m, r, z, 1, iw, q, 26 * n, &iw[2 * n], 10 * n, &info);
        chkxer("SSYEVR_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 1;
        ssyevr_2stage("V", "A", "U", 0, a, 1,
                      0.0f, 0.0f, 0, 0, 0.0f,
                      &m, r, z, 1, iw, q, 26 * n, &iw[2 * n], 10 * n, &info);
        chkxer("SSYEVR_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        ssyevr_2stage("N", "/", "U", 0, a, 1,
                      0.0f, 0.0f, 0, 0, 0.0f,
                      &m, r, z, 1, iw, q, 26 * n, &iw[2 * n], 10 * n, &info);
        chkxer("SSYEVR_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        ssyevr_2stage("N", "A", "/", -1, a, 1,
                      0.0f, 0.0f, 0, 0, 0.0f,
                      &m, r, z, 1, iw, q, 26 * n, &iw[2 * n], 10 * n, &info);
        chkxer("SSYEVR_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        ssyevr_2stage("N", "A", "U", -1, a, 1,
                      0.0f, 0.0f, 0, 0, 0.0f,
                      &m, r, z, 1, iw, q, 26 * n, &iw[2 * n], 10 * n, &info);
        chkxer("SSYEVR_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        ssyevr_2stage("N", "A", "U", 2, a, 1,
                      0.0f, 0.0f, 0, 0, 0.0f,
                      &m, r, z, 1, iw, q, 26 * n, &iw[2 * n], 10 * n, &info);
        chkxer("SSYEVR_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        ssyevr_2stage("N", "V", "U", 1, a, 1,
                      0.0f, 0.0f, 0, 0, 0.0f,
                      &m, r, z, 1, iw, q, 26 * n, &iw[2 * n], 10 * n, &info);
        chkxer("SSYEVR_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        ssyevr_2stage("N", "I", "U", 1, a, 1,
                      0.0f, 0.0f, -1, 0, 0.0f,
                      &m, r, z, 1, iw, q, 26 * n, &iw[2 * n], 10 * n, &info);
        chkxer("SSYEVR_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        ssyevr_2stage("N", "I", "U", 2, a, 2,
                      0.0f, 0.0f, 1, 0, 0.0f,
                      &m, r, z, 1, iw, q, 26 * n, &iw[2 * n], 10 * n, &info);
        chkxer("SSYEVR_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 15;
        ssyevr_2stage("N", "I", "U", 1, a, 1,
                      0.0f, 0.0f, 0, 0, 0.0f,
                      &m, r, z, 0, iw, q, 26 * n, &iw[2 * n], 10 * n, &info);
        chkxer("SSYEVR_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 18;
        ssyevr_2stage("N", "I", "U", 1, a, 1,
                      0.0f, 0.0f, 0, 0, 0.0f,
                      &m, r, z, 1, iw, q, 0, &iw[2 * n], 10 * n,
                      &info);
        chkxer("SSYEVR_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 20;
        ssyevr_2stage("N", "I", "U", 1, a, 1,
                      0.0f, 0.0f, 0, 0, 0.0f,
                      &m, r, z, 1, iw, q, 26 * n, &iw[2 * n], 0,
                      &info);
        chkxer("SSYEVR_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 12;

        /* SSYEV */

        strcpy(xerbla_srnamt, "SSYEV ");
        xerbla_infot = 1;
        ssyev("/", "U", 0, a, 1, x, w, 1, &info);
        chkxer("SSYEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        ssyev("N", "/", 0, a, 1, x, w, 1, &info);
        chkxer("SSYEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        ssyev("N", "U", -1, a, 1, x, w, 1, &info);
        chkxer("SSYEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        ssyev("N", "U", 2, a, 1, x, w, 3, &info);
        chkxer("SSYEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        ssyev("N", "U", 1, a, 1, x, w, 1, &info);
        chkxer("SSYEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 5;

        /* SSYEV_2STAGE */

        strcpy(xerbla_srnamt, "SSYEV_2STAGE ");
        xerbla_infot = 1;
        ssyev_2stage("/", "U", 0, a, 1, x, w, 1, &info);
        chkxer("SSYEV_2STAGE ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 1;
        ssyev_2stage("V", "U", 0, a, 1, x, w, 1, &info);
        chkxer("SSYEV_2STAGE ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        ssyev_2stage("N", "/", 0, a, 1, x, w, 1, &info);
        chkxer("SSYEV_2STAGE ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        ssyev_2stage("N", "U", -1, a, 1, x, w, 1, &info);
        chkxer("SSYEV_2STAGE ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        ssyev_2stage("N", "U", 2, a, 1, x, w, 3, &info);
        chkxer("SSYEV_2STAGE ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        ssyev_2stage("N", "U", 1, a, 1, x, w, 1, &info);
        chkxer("SSYEV_2STAGE ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 6;

        /* SSYEVX */

        strcpy(xerbla_srnamt, "SSYEVX");
        xerbla_infot = 1;
        ssyevx("/", "A", "U", 0, a, 1, 0.0f, 0.0f, -1, -1, 0.0f,
               &m, x, z, 1, w, 1, iw, i3, &info);
        chkxer("SSYEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        ssyevx("N", "/", "U", 0, a, 1, 0.0f, 1.0f, 0, -1, 0.0f,
               &m, x, z, 1, w, 1, iw, i3, &info);
        chkxer("SSYEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        ssyevx("N", "A", "/", 0, a, 1, 0.0f, 0.0f, -1, -1, 0.0f,
               &m, x, z, 1, w, 1, iw, i3, &info);
        xerbla_infot = 4;
        ssyevx("N", "A", "U", -1, a, 1, 0.0f, 0.0f, -1, -1,
               0.0f, &m, x, z, 1, w, 1, iw, i3, &info);
        chkxer("SSYEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        ssyevx("N", "A", "U", 2, a, 1, 0.0f, 0.0f, -1, -1, 0.0f,
               &m, x, z, 1, w, 16, iw, i3, &info);
        chkxer("SSYEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        ssyevx("N", "V", "U", 1, a, 1, 0.0f, 0.0f, -1, -1, 0.0f,
               &m, x, z, 1, w, 8, iw, i3, &info);
        chkxer("SSYEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        ssyevx("N", "I", "U", 1, a, 1, 0.0f, 0.0f, -1, -1, 0.0f,
               &m, x, z, 1, w, 8, iw, i3, &info);
        chkxer("SSYEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        ssyevx("N", "I", "U", 1, a, 1, 0.0f, 0.0f, 1, 0, 0.0f,
               &m, x, z, 1, w, 8, iw, i3, &info);
        chkxer("SSYEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        ssyevx("N", "I", "U", 2, a, 2, 0.0f, 0.0f, 1, 0, 0.0f,
               &m, x, z, 1, w, 16, iw, i3, &info);
        chkxer("SSYEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        ssyevx("N", "I", "U", 1, a, 1, 0.0f, 0.0f, 0, 1, 0.0f,
               &m, x, z, 1, w, 8, iw, i3, &info);
        chkxer("SSYEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 15;
        ssyevx("V", "A", "U", 2, a, 2, 0.0f, 0.0f, -1, -1, 0.0f,
               &m, x, z, 1, w, 16, iw, i3, &info);
        chkxer("SSYEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 17;
        ssyevx("V", "A", "U", 1, a, 1, 0.0f, 0.0f, -1, -1, 0.0f,
               &m, x, z, 1, w, 0, iw, i3, &info);
        chkxer("SSYEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 12;

        /* SSYEVX_2STAGE */

        strcpy(xerbla_srnamt, "SSYEVX_2STAGE");
        xerbla_infot = 1;
        ssyevx_2stage("/", "A", "U", 0, a, 1,
                      0.0f, 0.0f, -1, -1, 0.0f,
                      &m, x, z, 1, w, 1, iw, i3, &info);
        chkxer("SSYEVX_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 1;
        ssyevx_2stage("V", "A", "U", 0, a, 1,
                      0.0f, 0.0f, -1, -1, 0.0f,
                      &m, x, z, 1, w, 1, iw, i3, &info);
        chkxer("SSYEVX_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        ssyevx_2stage("N", "/", "U", 0, a, 1,
                      0.0f, 1.0f, 0, -1, 0.0f,
                      &m, x, z, 1, w, 1, iw, i3, &info);
        chkxer("SSYEVX_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        ssyevx_2stage("N", "A", "/", 0, a, 1,
                      0.0f, 0.0f, -1, -1, 0.0f,
                      &m, x, z, 1, w, 1, iw, i3, &info);
        xerbla_infot = 4;
        ssyevx_2stage("N", "A", "U", -1, a, 1,
                      0.0f, 0.0f, -1, -1, 0.0f,
                      &m, x, z, 1, w, 1, iw, i3, &info);
        chkxer("SSYEVX_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        ssyevx_2stage("N", "A", "U", 2, a, 1,
                      0.0f, 0.0f, -1, -1, 0.0f,
                      &m, x, z, 1, w, 16, iw, i3, &info);
        chkxer("SSYEVX_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        ssyevx_2stage("N", "V", "U", 1, a, 1,
                      0.0f, 0.0f, -1, -1, 0.0f,
                      &m, x, z, 1, w, 8, iw, i3, &info);
        chkxer("SSYEVX_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        ssyevx_2stage("N", "I", "U", 1, a, 1,
                      0.0f, 0.0f, -1, -1, 0.0f,
                      &m, x, z, 1, w, 8, iw, i3, &info);
        chkxer("SSYEVX_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        ssyevx_2stage("N", "I", "U", 1, a, 1,
                      0.0f, 0.0f, 1, 0, 0.0f,
                      &m, x, z, 1, w, 8, iw, i3, &info);
        chkxer("SSYEVX_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        ssyevx_2stage("N", "I", "U", 2, a, 2,
                      0.0f, 0.0f, 1, 0, 0.0f,
                      &m, x, z, 1, w, 16, iw, i3, &info);
        chkxer("SSYEVX_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        ssyevx_2stage("N", "I", "U", 1, a, 1,
                      0.0f, 0.0f, 0, 1, 0.0f,
                      &m, x, z, 1, w, 8, iw, i3, &info);
        chkxer("SSYEVX_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 15;
        ssyevx_2stage("N", "A", "U", 2, a, 2,
                      0.0f, 0.0f, -1, -1, 0.0f,
                      &m, x, z, 0, w, 16, iw, i3, &info);
        chkxer("SSYEVX_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 17;
        ssyevx_2stage("N", "A", "U", 1, a, 1,
                      0.0f, 0.0f, -1, -1, 0.0f,
                      &m, x, z, 1, w, 0, iw, i3, &info);
        chkxer("SSYEVX_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 13;

        /* SSPEVD */

        strcpy(xerbla_srnamt, "SSPEVD");
        xerbla_infot = 1;
        sspevd("/", "U", 0, a, x, z, 1, w, 1, iw, 1, &info);
        chkxer("SSPEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        sspevd("N", "/", 0, a, x, z, 1, w, 1, iw, 1, &info);
        chkxer("SSPEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        sspevd("N", "U", -1, a, x, z, 1, w, 1, iw, 1, &info);
        chkxer("SSPEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        sspevd("V", "U", 2, a, x, z, 1, w, 23, iw, 12, &info);
        chkxer("SSPEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        sspevd("N", "U", 1, a, x, z, 1, w, 0, iw, 1, &info);
        chkxer("SSPEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        sspevd("N", "U", 2, a, x, z, 1, w, 3, iw, 1, &info);
        chkxer("SSPEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        sspevd("V", "U", 2, a, x, z, 2, w, 16, iw, 12, &info);
        chkxer("SSPEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 11;
        sspevd("N", "U", 1, a, x, z, 1, w, 1, iw, 0, &info);
        chkxer("SSPEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 11;
        sspevd("N", "U", 2, a, x, z, 1, w, 4, iw, 0, &info);
        chkxer("SSPEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 11;
        sspevd("V", "U", 2, a, x, z, 2, w, 23, iw, 11, &info);
        chkxer("SSPEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 10;

        /* SSPEV */

        strcpy(xerbla_srnamt, "SSPEV ");
        xerbla_infot = 1;
        sspev("/", "U", 0, a, w, z, 1, x, &info);
        chkxer("SSPEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        sspev("N", "/", 0, a, w, z, 1, x, &info);
        chkxer("SSPEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        sspev("N", "U", -1, a, w, z, 1, x, &info);
        chkxer("SSPEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        sspev("V", "U", 2, a, w, z, 1, x, &info);
        chkxer("SSPEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 4;

        /* SSPEVX */

        strcpy(xerbla_srnamt, "SSPEVX");
        xerbla_infot = 1;
        sspevx("/", "A", "U", 0, a, 0.0f, 0.0f, -1, -1, 0.0f, &m,
               x, z, 1, w, iw, i3, &info);
        chkxer("SSPEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        sspevx("N", "/", "U", 0, a, 0.0f, 0.0f, -1, -1, 0.0f, &m,
               x, z, 1, w, iw, i3, &info);
        chkxer("SSPEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        sspevx("N", "A", "/", 0, a, 0.0f, 0.0f, -1, -1, 0.0f, &m,
               x, z, 1, w, iw, i3, &info);
        xerbla_infot = 4;
        sspevx("N", "A", "U", -1, a, 0.0f, 0.0f, -1, -1, 0.0f,
               &m, x, z, 1, w, iw, i3, &info);
        chkxer("SSPEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        sspevx("N", "V", "U", 1, a, 0.0f, 0.0f, -1, -1, 0.0f, &m,
               x, z, 1, w, iw, i3, &info);
        chkxer("SSPEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        sspevx("N", "I", "U", 1, a, 0.0f, 0.0f, -1, -1, 0.0f, &m,
               x, z, 1, w, iw, i3, &info);
        chkxer("SSPEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        sspevx("N", "I", "U", 1, a, 0.0f, 0.0f, 1, 0, 0.0f, &m,
               x, z, 1, w, iw, i3, &info);
        chkxer("SSPEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        sspevx("N", "I", "U", 2, a, 0.0f, 0.0f, 1, 0, 0.0f, &m,
               x, z, 1, w, iw, i3, &info);
        chkxer("SSPEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        sspevx("N", "I", "U", 1, a, 0.0f, 0.0f, 0, 1, 0.0f, &m,
               x, z, 1, w, iw, i3, &info);
        chkxer("SSPEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 14;
        sspevx("V", "A", "U", 2, a, 0.0f, 0.0f, -1, -1, 0.0f, &m,
               x, z, 1, w, iw, i3, &info);
        chkxer("SSPEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 10;

    /* Test error exits for the SB path. */
    } else if ((path[1] == 'S' || path[1] == 's') &&
               (path[2] == 'B' || path[2] == 'b')) {

        /* SSBTRD */

        strcpy(xerbla_srnamt, "SSBTRD");
        xerbla_infot = 1;
        ssbtrd("/", "U", 0, 0, a, 1, d, e, z, 1, w, &info);
        chkxer("SSBTRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        ssbtrd("N", "/", 0, 0, a, 1, d, e, z, 1, w, &info);
        chkxer("SSBTRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        ssbtrd("N", "U", -1, 0, a, 1, d, e, z, 1, w, &info);
        chkxer("SSBTRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        ssbtrd("N", "U", 0, -1, a, 1, d, e, z, 1, w, &info);
        chkxer("SSBTRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        ssbtrd("N", "U", 1, 1, a, 1, d, e, z, 1, w, &info);
        chkxer("SSBTRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        ssbtrd("V", "U", 2, 0, a, 1, d, e, z, 1, w, &info);
        chkxer("SSBTRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 6;

        /* SSYTRD_SB2ST */

        strcpy(xerbla_srnamt, "SSYTRD_SB2ST");
        xerbla_infot = 1;
        ssytrd_sb2st("/", "N", "U", 0, 0, a, 1, d, e,
                     c, 1, w, 1, &info);
        chkxer("SSYTRD_SB2ST", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        ssytrd_sb2st("N", "/", "U", 0, 0, a, 1, d, e,
                     c, 1, w, 1, &info);
        chkxer("SSYTRD_SB2ST", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        ssytrd_sb2st("N", "H", "U", 0, 0, a, 1, d, e,
                     c, 1, w, 1, &info);
        chkxer("SSYTRD_SB2ST", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        ssytrd_sb2st("N", "N", "/", 0, 0, a, 1, d, e,
                     c, 1, w, 1, &info);
        chkxer("SSYTRD_SB2ST", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        ssytrd_sb2st("N", "N", "U", -1, 0, a, 1, d, e,
                     c, 1, w, 1, &info);
        chkxer("SSYTRD_SB2ST", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        ssytrd_sb2st("N", "N", "U", 0, -1, a, 1, d, e,
                     c, 1, w, 1, &info);
        chkxer("SSYTRD_SB2ST", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        ssytrd_sb2st("N", "N", "U", 0, 1, a, 1, d, e,
                     c, 1, w, 1, &info);
        chkxer("SSYTRD_SB2ST", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 11;
        ssytrd_sb2st("N", "N", "U", 0, 0, a, 1, d, e,
                     c, 0, w, 1, &info);
        chkxer("SSYTRD_SB2ST", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 13;
        ssytrd_sb2st("N", "N", "U", 0, 0, a, 1, d, e,
                     c, 1, w, 0, &info);
        chkxer("SSYTRD_SB2ST", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 9;

        /* SSBEVD */

        strcpy(xerbla_srnamt, "SSBEVD");
        xerbla_infot = 1;
        ssbevd("/", "U", 0, 0, a, 1, x, z, 1, w, 1, iw, 1, &info);
        chkxer("SSBEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        ssbevd("N", "/", 0, 0, a, 1, x, z, 1, w, 1, iw, 1, &info);
        chkxer("SSBEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        ssbevd("N", "U", -1, 0, a, 1, x, z, 1, w, 1, iw, 1,
               &info);
        chkxer("SSBEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        ssbevd("N", "U", 0, -1, a, 1, x, z, 1, w, 1, iw, 1,
               &info);
        chkxer("SSBEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        ssbevd("N", "U", 2, 1, a, 1, x, z, 1, w, 4, iw, 1, &info);
        chkxer("SSBEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        ssbevd("V", "U", 2, 1, a, 2, x, z, 1, w, 25, iw, 12,
               &info);
        chkxer("SSBEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 11;
        ssbevd("N", "U", 1, 0, a, 1, x, z, 1, w, 0, iw, 1, &info);
        chkxer("SSBEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 11;
        ssbevd("N", "U", 2, 0, a, 1, x, z, 1, w, 3, iw, 1, &info);
        chkxer("SSBEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 11;
        ssbevd("V", "U", 2, 0, a, 1, x, z, 2, w, 18, iw, 12,
               &info);
        chkxer("SSBEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 13;
        ssbevd("N", "U", 1, 0, a, 1, x, z, 1, w, 1, iw, 0, &info);
        chkxer("SSBEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 13;
        ssbevd("V", "U", 2, 0, a, 1, x, z, 2, w, 25, iw, 11,
               &info);
        chkxer("SSBEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 11;

        /* SSBEVD_2STAGE */

        strcpy(xerbla_srnamt, "SSBEVD_2STAGE");
        xerbla_infot = 1;
        ssbevd_2stage("/", "U", 0, 0, a, 1, x, z, 1, w,
                      1, iw, 1, &info);
        chkxer("SSBEVD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 1;
        ssbevd_2stage("V", "U", 0, 0, a, 1, x, z, 1, w,
                      1, iw, 1, &info);
        chkxer("SSBEVD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        ssbevd_2stage("N", "/", 0, 0, a, 1, x, z, 1, w,
                      1, iw, 1, &info);
        chkxer("SSBEVD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        ssbevd_2stage("N", "U", -1, 0, a, 1, x, z, 1, w,
                      1, iw, 1, &info);
        chkxer("SSBEVD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        ssbevd_2stage("N", "U", 0, -1, a, 1, x, z, 1, w,
                      1, iw, 1, &info);
        chkxer("SSBEVD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        ssbevd_2stage("N", "U", 2, 1, a, 1, x, z, 1, w,
                      4, iw, 1, &info);
        chkxer("SSBEVD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        /* xerbla_infot = 9;
           ssbevd_2stage("V", "U", 2, 1, a, 2, x, z, 1, w,
                         25, iw, 12, &info);
           chkxer("SSBEVD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok); */
        xerbla_infot = 11;
        ssbevd_2stage("N", "U", 1, 0, a, 1, x, z, 1, w,
                      0, iw, 1, &info);
        chkxer("SSBEVD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 11;
        ssbevd_2stage("N", "U", 2, 0, a, 1, x, z, 1, w,
                      3, iw, 1, &info);
        chkxer("SSBEVD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        /* xerbla_infot = 11;
           ssbevd_2stage("V", "U", 2, 0, a, 1, x, z, 2, w,
                         18, iw, 12, &info);
           chkxer("SSBEVD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok); */
        xerbla_infot = 13;
        ssbevd_2stage("N", "U", 1, 0, a, 1, x, z, 1, w,
                      1, iw, 0, &info);
        chkxer("SSBEVD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        /* xerbla_infot = 13;
           ssbevd_2stage("V", "U", 2, 0, a, 1, x, z, 2, w,
                         25, iw, 11, &info);
           chkxer("SSBEVD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
           *nt = *nt + 12; */
        *nt = *nt + 9;

        /* SSBEV */

        strcpy(xerbla_srnamt, "SSBEV ");
        xerbla_infot = 1;
        ssbev("/", "U", 0, 0, a, 1, x, z, 1, w, &info);
        chkxer("SSBEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        ssbev("N", "/", 0, 0, a, 1, x, z, 1, w, &info);
        chkxer("SSBEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        ssbev("N", "U", -1, 0, a, 1, x, z, 1, w, &info);
        chkxer("SSBEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        ssbev("N", "U", 0, -1, a, 1, x, z, 1, w, &info);
        chkxer("SSBEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        ssbev("N", "U", 2, 1, a, 1, x, z, 1, w, &info);
        chkxer("SSBEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        ssbev("V", "U", 2, 0, a, 1, x, z, 1, w, &info);
        chkxer("SSBEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 6;

        /* SSBEV_2STAGE */

        strcpy(xerbla_srnamt, "SSBEV_2STAGE ");
        xerbla_infot = 1;
        ssbev_2stage("/", "U", 0, 0, a, 1, x, z, 1, w, 0, &info);
        chkxer("SSBEV_2STAGE ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 1;
        ssbev_2stage("V", "U", 0, 0, a, 1, x, z, 1, w, 0, &info);
        chkxer("SSBEV_2STAGE ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        ssbev_2stage("N", "/", 0, 0, a, 1, x, z, 1, w, 0, &info);
        chkxer("SSBEV_2STAGE ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        ssbev_2stage("N", "U", -1, 0, a, 1, x, z, 1, w, 0, &info);
        chkxer("SSBEV_2STAGE ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        ssbev_2stage("N", "U", 0, -1, a, 1, x, z, 1, w, 0, &info);
        chkxer("SSBEV_2STAGE ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        ssbev_2stage("N", "U", 2, 1, a, 1, x, z, 1, w, 0, &info);
        chkxer("SSBEV_2STAGE ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        ssbev_2stage("N", "U", 2, 0, a, 1, x, z, 0, w, 0, &info);
        chkxer("SSBEV_2STAGE ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 11;
        ssbev_2stage("N", "U", 0, 0, a, 1, x, z, 1, w, 0, &info);
        chkxer("SSBEV_2STAGE ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 8;

        /* SSBEVX */

        strcpy(xerbla_srnamt, "SSBEVX");
        xerbla_infot = 1;
        ssbevx("/", "A", "U", 0, 0, a, 1, q, 1, 0.0f, 0.0f, -1,
               -1, 0.0f, &m, x, z, 1, w, iw, i3, &info);
        chkxer("SSBEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        ssbevx("N", "/", "U", 0, 0, a, 1, q, 1, 0.0f, 0.0f, -1,
               -1, 0.0f, &m, x, z, 1, w, iw, i3, &info);
        chkxer("SSBEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        ssbevx("N", "A", "/", 0, 0, a, 1, q, 1, 0.0f, 0.0f, -1,
               -1, 0.0f, &m, x, z, 1, w, iw, i3, &info);
        chkxer("SSBEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        ssbevx("N", "A", "U", -1, 0, a, 1, q, 1, 0.0f, 0.0f, -1,
               -1, 0.0f, &m, x, z, 1, w, iw, i3, &info);
        chkxer("SSBEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        ssbevx("N", "A", "U", 0, -1, a, 1, q, 1, 0.0f, 0.0f, -1,
               -1, 0.0f, &m, x, z, 1, w, iw, i3, &info);
        chkxer("SSBEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        ssbevx("N", "A", "U", 2, 1, a, 1, q, 1, 0.0f, 0.0f, -1,
               -1, 0.0f, &m, x, z, 1, w, iw, i3, &info);
        chkxer("SSBEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        ssbevx("V", "A", "U", 2, 0, a, 1, q, 1, 0.0f, 0.0f, -1,
               -1, 0.0f, &m, x, z, 2, w, iw, i3, &info);
        chkxer("SSBEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 11;
        ssbevx("N", "V", "U", 1, 0, a, 1, q, 1, 0.0f, 0.0f, -1,
               -1, 0.0f, &m, x, z, 1, w, iw, i3, &info);
        chkxer("SSBEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 12;
        ssbevx("N", "I", "U", 1, 0, a, 1, q, 1, 0.0f, 0.0f, -1,
               -1, 0.0f, &m, x, z, 1, w, iw, i3, &info);
        chkxer("SSBEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 12;
        ssbevx("N", "I", "U", 1, 0, a, 1, q, 1, 0.0f, 0.0f, 1,
               0, 0.0f, &m, x, z, 1, w, iw, i3, &info);
        chkxer("SSBEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 13;
        ssbevx("N", "I", "U", 2, 0, a, 1, q, 1, 0.0f, 0.0f, 1,
               0, 0.0f, &m, x, z, 1, w, iw, i3, &info);
        chkxer("SSBEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 13;
        ssbevx("N", "I", "U", 1, 0, a, 1, q, 1, 0.0f, 0.0f, 0,
               1, 0.0f, &m, x, z, 1, w, iw, i3, &info);
        chkxer("SSBEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 18;
        ssbevx("V", "A", "U", 2, 0, a, 1, q, 2, 0.0f, 0.0f, -1,
               -1, 0.0f, &m, x, z, 1, w, iw, i3, &info);
        chkxer("SSBEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 13;

        /* SSBEVX_2STAGE */

        strcpy(xerbla_srnamt, "SSBEVX_2STAGE");
        xerbla_infot = 1;
        ssbevx_2stage("/", "A", "U", 0, 0, a, 1, q, 1, 0.0f,
                      0.0f, -1, -1, 0.0f, &m, x, z, 1, w, 0, iw, i3, &info);
        chkxer("SSBEVX_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 1;
        ssbevx_2stage("V", "A", "U", 0, 0, a, 1, q, 1, 0.0f,
                      0.0f, -1, -1, 0.0f, &m, x, z, 1, w, 0, iw, i3, &info);
        chkxer("SSBEVX_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        ssbevx_2stage("N", "/", "U", 0, 0, a, 1, q, 1, 0.0f,
                      0.0f, -1, -1, 0.0f, &m, x, z, 1, w, 0, iw, i3, &info);
        chkxer("SSBEVX_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        ssbevx_2stage("N", "A", "/", 0, 0, a, 1, q, 1, 0.0f,
                      0.0f, -1, -1, 0.0f, &m, x, z, 1, w, 0, iw, i3, &info);
        chkxer("SSBEVX_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        ssbevx_2stage("N", "A", "U", -1, 0, a, 1, q, 1, 0.0f,
                      0.0f, -1, -1, 0.0f, &m, x, z, 1, w, 0, iw, i3, &info);
        chkxer("SSBEVX_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        ssbevx_2stage("N", "A", "U", 0, -1, a, 1, q, 1, 0.0f,
                      0.0f, -1, -1, 0.0f, &m, x, z, 1, w, 0, iw, i3, &info);
        chkxer("SSBEVX_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        ssbevx_2stage("N", "A", "U", 2, 1, a, 1, q, 1, 0.0f,
                      0.0f, -1, -1, 0.0f, &m, x, z, 1, w, 0, iw, i3, &info);
        chkxer("SSBEVX_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        /* xerbla_infot = 9;
           ssbevx_2stage("V", "A", "U", 2, 0, a, 1, q, 1, 0.0,
                         0.0, -1, -1, 0.0, &m, x, z, 2, w, 0, iw, i3, &info);
           chkxer("SSBEVX_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok); */
        xerbla_infot = 11;
        ssbevx_2stage("N", "V", "U", 1, 0, a, 1, q, 1, 0.0f,
                      0.0f, -1, -1, 0.0f, &m, x, z, 1, w, 0, iw, i3, &info);
        chkxer("SSBEVX_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 12;
        ssbevx_2stage("N", "I", "U", 1, 0, a, 1, q, 1, 0.0f,
                      0.0f, -1, -1, 0.0f, &m, x, z, 1, w, 0, iw, i3, &info);
        chkxer("SSBEVX_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 12;
        ssbevx_2stage("N", "I", "U", 1, 0, a, 1, q, 1, 0.0f,
                      0.0f, 1, 0, 0.0f, &m, x, z, 1, w, 0, iw, i3, &info);
        chkxer("SSBEVX_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 13;
        ssbevx_2stage("N", "I", "U", 2, 0, a, 1, q, 1, 0.0f,
                      0.0f, 1, 0, 0.0f, &m, x, z, 1, w, 0, iw, i3, &info);
        chkxer("SSBEVX_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 13;
        ssbevx_2stage("N", "I", "U", 1, 0, a, 1, q, 1, 0.0f,
                      0.0f, 0, 1, 0.0f, &m, x, z, 1, w, 0, iw, i3, &info);
        chkxer("SSBEVX_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        /* xerbla_infot = 18;
           ssbevx_2stage("V", "A", "U", 2, 0, a, 1, q, 2, 0.0,
                         0.0, -1, -1, 0.0, &m, x, z, 1, w, 0, iw, i3, &info);
           chkxer("SSBEVX_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok); */
        xerbla_infot = 20;
        ssbevx_2stage("N", "A", "U", 0, 0, a, 1, q, 1, 0.0f,
                      0.0f, -1, -1, 0.0f, &m, x, z, 1, w, 0, iw, i3, &info);
        chkxer("SSBEVX_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        /* *nt = *nt + 15; */
        *nt = *nt + 13;
    }

    *ok = xerbla_ok;

    xerbla_srnamt[0] = '\0';
    xerbla_infot = 0;
}
