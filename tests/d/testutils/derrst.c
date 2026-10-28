/**
 * @file derrst.c
 * @brief DERRST tests the error exits for the symmetric eigenvalue routines.
 *
 * Port of LAPACK TESTING/EIG/derrst.f
 * Tests error exits for: DSYTRD, DSYTD2, DORGTR, DORMTR, DSPTRD,
 * DOPGTR, DOPMTR, DSTEQR, DSTERF, DSTEBZ, DSTEIN, DPTEQR, DSBTRD,
 * DSYEV, DSYEVX, DSYEVD, DSBEV, DSBEVX, DSBEVD,
 * DSPEV, DSPEVX, DSPEVD, DSTEV, DSTEVX, DSTEVD, and DSTEDC.
 * DSYEVD_2STAGE, DSYEVR_2STAGE, DSYEVX_2STAGE,
 * DSYEV_2STAGE, DSBEV_2STAGE, DSBEVD_2STAGE,
 * DSBEVX_2STAGE, DSYTRD_2STAGE, DSYTRD_SY2SB,
 * DSYTRD_SB2ST
 */

#include <math.h>
#include <string.h>
#include "semicolon_lapack_double.h"
#include "verify.h"

void derrst(const char* path, INT* ok, INT* nt)
{
    /* NMAX has to be at least 3 or LIW may be too small */
    enum { NMAX = 3, LIW = 12 * NMAX, LW = 20 * NMAX };

    INT i, info, j, m, n, nsplit;

    INT i1[NMAX], i2[NMAX], i3[NMAX], iw[LIW];
    f64 a[NMAX * NMAX], c[NMAX * NMAX], d[NMAX], e[NMAX], q[NMAX * NMAX];
    f64 r[NMAX], tau[NMAX], w[LW], x[NMAX], z[NMAX * NMAX];
    f64 rinf, rzero;

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
    if ((path[1] == 'S' || path[1] == 's') &&
        (path[2] == 'T' || path[2] == 't')) {

        /* DSYTRD */

        strcpy(xerbla_srnamt, "DSYTRD");
        xerbla_infot = 1;
        dsytrd("/", 0, a, 1, d, e, tau, w, 1, &info);
        chkxer("DSYTRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        dsytrd("U", -1, a, 1, d, e, tau, w, 1, &info);
        chkxer("DSYTRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        dsytrd("U", 2, a, 1, d, e, tau, w, 1, &info);
        chkxer("DSYTRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        dsytrd("U", 0, a, 1, d, e, tau, w, 0, &info);
        chkxer("DSYTRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 4;

        /* DSYTD2 */

        strcpy(xerbla_srnamt, "DSYTD2");
        xerbla_infot = 1;
        dsytd2("/", 0, a, 1, d, e, tau, &info);
        chkxer("DSYTD2", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        dsytd2("U", -1, a, 1, d, e, tau, &info);
        chkxer("DSYTD2", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        dsytd2("U", 2, a, 1, d, e, tau, &info);
        chkxer("DSYTD2", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 3;

        /* DSYTRD_2STAGE */

        strcpy(xerbla_srnamt, "DSYTRD_2STAGE");
        xerbla_infot = 1;
        dsytrd_2stage("/", "U", 0, a, 1, d, e, tau,
                      c, 1, w, 1, &info);
        chkxer("DSYTRD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 1;
        dsytrd_2stage("H", "U", 0, a, 1, d, e, tau,
                      c, 1, w, 1, &info);
        chkxer("DSYTRD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        dsytrd_2stage("N", "/", 0, a, 1, d, e, tau,
                      c, 1, w, 1, &info);
        chkxer("DSYTRD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        dsytrd_2stage("N", "U", -1, a, 1, d, e, tau,
                      c, 1, w, 1, &info);
        chkxer("DSYTRD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        dsytrd_2stage("N", "U", 2, a, 1, d, e, tau,
                      c, 1, w, 1, &info);
        chkxer("DSYTRD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        dsytrd_2stage("N", "U", 0, a, 1, d, e, tau,
                      c, 0, w, 1, &info);
        chkxer("DSYTRD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 12;
        dsytrd_2stage("N", "U", 0, a, 1, d, e, tau,
                      c, 1, w, 0, &info);
        chkxer("DSYTRD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 7;

        /* DSYTRD_SY2SB */

        strcpy(xerbla_srnamt, "DSYTRD_SY2SB");
        xerbla_infot = 1;
        dsytrd_sy2sb("/", 0, 0, a, 1, c, 1, tau, w, 1, &info);
        chkxer("DSYTRD_SY2SB", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        dsytrd_sy2sb("U", -1, 0, a, 1, c, 1, tau, w, 1, &info);
        chkxer("DSYTRD_SY2SB", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        dsytrd_sy2sb("U", 0, -1, a, 1, c, 1, tau, w, 1, &info);
        chkxer("DSYTRD_SY2SB", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        dsytrd_sy2sb("U", 2, 0, a, 1, c, 1, tau, w, 1, &info);
        chkxer("DSYTRD_SY2SB", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        dsytrd_sy2sb("U", 0, 2, a, 1, c, 1, tau, w, 1, &info);
        chkxer("DSYTRD_SY2SB", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        dsytrd_sy2sb("U", 0, 0, a, 1, c, 1, tau, w, 0, &info);
        chkxer("DSYTRD_SY2SB", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 6;

        /* DSYTRD_SB2ST */

        strcpy(xerbla_srnamt, "DSYTRD_SB2ST");
        xerbla_infot = 1;
        dsytrd_sb2st("/", "N", "U", 0, 0, a, 1, d, e,
                     c, 1, w, 1, &info);
        chkxer("DSYTRD_SB2ST", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        dsytrd_sb2st("Y", "/", "U", 0, 0, a, 1, d, e,
                     c, 1, w, 1, &info);
        chkxer("DSYTRD_SB2ST", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        dsytrd_sb2st("Y", "H", "U", 0, 0, a, 1, d, e,
                     c, 1, w, 1, &info);
        chkxer("DSYTRD_SB2ST", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        dsytrd_sb2st("Y", "N", "/", 0, 0, a, 1, d, e,
                     c, 1, w, 1, &info);
        chkxer("DSYTRD_SB2ST", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        dsytrd_sb2st("Y", "N", "U", -1, 0, a, 1, d, e,
                     c, 1, w, 1, &info);
        chkxer("DSYTRD_SB2ST", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        dsytrd_sb2st("Y", "N", "U", 0, -1, a, 1, d, e,
                     c, 1, w, 1, &info);
        chkxer("DSYTRD_SB2ST", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        dsytrd_sb2st("Y", "N", "U", 0, 1, a, 1, d, e,
                     c, 1, w, 1, &info);
        chkxer("DSYTRD_SB2ST", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 11;
        dsytrd_sb2st("Y", "N", "U", 0, 0, a, 1, d, e,
                     c, 0, w, 1, &info);
        chkxer("DSYTRD_SB2ST", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 13;
        dsytrd_sb2st("Y", "N", "U", 0, 0, a, 1, d, e,
                     c, 1, w, 0, &info);
        chkxer("DSYTRD_SB2ST", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 9;

        /* DORGTR */

        strcpy(xerbla_srnamt, "DORGTR");
        xerbla_infot = 1;
        dorgtr("/", 0, a, 1, tau, w, 1, &info);
        chkxer("DORGTR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        dorgtr("U", -1, a, 1, tau, w, 1, &info);
        chkxer("DORGTR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        dorgtr("U", 2, a, 1, tau, w, 1, &info);
        chkxer("DORGTR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        dorgtr("U", 3, a, 3, tau, w, 1, &info);
        chkxer("DORGTR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 4;

        /* DORMTR */

        strcpy(xerbla_srnamt, "DORMTR");
        xerbla_infot = 1;
        dormtr("/", "U", "N", 0, 0, a, 1, tau, c, 1, w, 1, &info);
        chkxer("DORMTR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        dormtr("L", "/", "N", 0, 0, a, 1, tau, c, 1, w, 1, &info);
        chkxer("DORMTR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        dormtr("L", "U", "/", 0, 0, a, 1, tau, c, 1, w, 1, &info);
        chkxer("DORMTR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        dormtr("L", "U", "N", -1, 0, a, 1, tau, c, 1, w, 1,
               &info);
        chkxer("DORMTR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        dormtr("L", "U", "N", 0, -1, a, 1, tau, c, 1, w, 1,
               &info);
        chkxer("DORMTR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        dormtr("L", "U", "N", 2, 0, a, 1, tau, c, 2, w, 1, &info);
        chkxer("DORMTR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        dormtr("R", "U", "N", 0, 2, a, 1, tau, c, 1, w, 1, &info);
        chkxer("DORMTR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        dormtr("L", "U", "N", 2, 0, a, 2, tau, c, 1, w, 1, &info);
        chkxer("DORMTR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 12;
        dormtr("L", "U", "N", 0, 2, a, 1, tau, c, 1, w, 1, &info);
        chkxer("DORMTR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 12;
        dormtr("R", "U", "N", 2, 0, a, 1, tau, c, 2, w, 1, &info);
        chkxer("DORMTR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 10;

        /* DSPTRD */

        strcpy(xerbla_srnamt, "DSPTRD");
        xerbla_infot = 1;
        dsptrd("/", 0, a, d, e, tau, &info);
        chkxer("DSPTRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        dsptrd("U", -1, a, d, e, tau, &info);
        chkxer("DSPTRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 2;

        /* DOPGTR */

        strcpy(xerbla_srnamt, "DOPGTR");
        xerbla_infot = 1;
        dopgtr("/", 0, a, tau, z, 1, w, &info);
        chkxer("DOPGTR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        dopgtr("U", -1, a, tau, z, 1, w, &info);
        chkxer("DOPGTR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        dopgtr("U", 2, a, tau, z, 1, w, &info);
        chkxer("DOPGTR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 3;

        /* DOPMTR */

        strcpy(xerbla_srnamt, "DOPMTR");
        xerbla_infot = 1;
        dopmtr("/", "U", "N", 0, 0, a, tau, c, 1, w, &info);
        chkxer("DOPMTR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        dopmtr("L", "/", "N", 0, 0, a, tau, c, 1, w, &info);
        chkxer("DOPMTR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        dopmtr("L", "U", "/", 0, 0, a, tau, c, 1, w, &info);
        chkxer("DOPMTR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        dopmtr("L", "U", "N", -1, 0, a, tau, c, 1, w, &info);
        chkxer("DOPMTR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        dopmtr("L", "U", "N", 0, -1, a, tau, c, 1, w, &info);
        chkxer("DOPMTR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        dopmtr("L", "U", "N", 2, 0, a, tau, c, 1, w, &info);
        chkxer("DOPMTR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 6;

        /* DPTEQR */

        strcpy(xerbla_srnamt, "DPTEQR");
        xerbla_infot = 1;
        dpteqr("/", 0, d, e, z, 1, w, &info);
        chkxer("DPTEQR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        dpteqr("N", -1, d, e, z, 1, w, &info);
        chkxer("DPTEQR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        dpteqr("V", 2, d, e, z, 1, w, &info);
        chkxer("DPTEQR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 3;

        /* DSTEBZ */

        strcpy(xerbla_srnamt, "DSTEBZ");
        xerbla_infot = 1;
        dstebz("/", "E", 0, 0.0, 1.0, 0, -1, 0.0, d, e, &m,
               &nsplit, x, i1, i2, w, iw, &info);
        chkxer("DSTEBZ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        dstebz("A", "/", 0, 0.0, 0.0, -1, -1, 0.0, d, e, &m,
               &nsplit, x, i1, i2, w, iw, &info);
        chkxer("DSTEBZ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        dstebz("A", "E", -1, 0.0, 0.0, -1, -1, 0.0, d, e, &m,
               &nsplit, x, i1, i2, w, iw, &info);
        chkxer("DSTEBZ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        dstebz("V", "E", 0, 0.0, 0.0, -1, -1, 0.0, d, e, &m,
               &nsplit, x, i1, i2, w, iw, &info);
        chkxer("DSTEBZ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        dstebz("I", "E", 0, 0.0, 0.0, -1, -1, 0.0, d, e, &m,
               &nsplit, x, i1, i2, w, iw, &info);
        chkxer("DSTEBZ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        dstebz("I", "E", 1, 0.0, 0.0, 1, 0, 0.0, d, e, &m,
               &nsplit, x, i1, i2, w, iw, &info);
        chkxer("DSTEBZ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        dstebz("I", "E", 1, 0.0, 0.0, 0, -1, 0.0, d, e, &m,
               &nsplit, x, i1, i2, w, iw, &info);
        chkxer("DSTEBZ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        dstebz("I", "E", 1, 0.0, 0.0, 0, 1, 0.0, d, e, &m,
               &nsplit, x, i1, i2, w, iw, &info);
        chkxer("DSTEBZ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 8;

        /* DSTEIN */

        strcpy(xerbla_srnamt, "DSTEIN");
        xerbla_infot = 1;
        dstein(-1, d, e, 0, x, i1, i2, z, 1, w, iw, i3, &info);
        chkxer("DSTEIN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        dstein(0, d, e, -1, x, i1, i2, z, 1, w, iw, i3, &info);
        chkxer("DSTEIN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        dstein(0, d, e, 1, x, i1, i2, z, 1, w, iw, i3, &info);
        chkxer("DSTEIN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        dstein(2, d, e, 0, x, i1, i2, z, 1, w, iw, i3, &info);
        chkxer("DSTEIN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 4;

        /* A matrix that holds an infinity must come back from the
           drivers that scale it, instead of reaching XERBLA with a
           scale factor of zero. */

        rzero = 0.0;
        rinf = 1.0 / rzero;
        for (j = 0; j < NMAX; j++) {
            d[j] = (f64)(j + 1);
            e[j] = 1.0 / (f64)(j + 2);
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
        strcpy(xerbla_srnamt, "DSTEVX");
        xerbla_infot = 0;
        xerbla_lerr = 0;
        dstevx("V", "V", NMAX, d, e, rzero, 1.0, 0, NMAX - 1,
               rzero, &m, x, z, NMAX, w, iw, i3, &info);
        if (xerbla_lerr) {
            xerbla_ok = 0;
        }
        d[0] = rinf;
        e[0] = 1.0 / (f64)2;
        e[1] = 1.0 / (f64)3;
        e[NMAX - 1] = rzero;
        strcpy(xerbla_srnamt, "DSYEVX");
        xerbla_infot = 0;
        xerbla_lerr = 0;
        dsyevx("V", "V", "U", NMAX, a, NMAX, rzero, 1.0, 0,
               NMAX - 1, rzero, &m, x, z, NMAX, w, LW, iw, i3, &info);
        if (xerbla_lerr) {
            xerbla_ok = 0;
        }
        *nt = *nt + 2;

        /* DSTEQR */

        strcpy(xerbla_srnamt, "DSTEQR");
        xerbla_infot = 1;
        dsteqr("/", 0, d, e, z, 1, w, &info);
        chkxer("DSTEQR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        dsteqr("N", -1, d, e, z, 1, w, &info);
        chkxer("DSTEQR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        dsteqr("V", 2, d, e, z, 1, w, &info);
        chkxer("DSTEQR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 3;

        /* DSTERF */

        strcpy(xerbla_srnamt, "DSTERF");
        xerbla_infot = 1;
        dsterf(-1, d, e, &info);
        chkxer("DSTERF", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 1;

        /* DSTEDC */

        strcpy(xerbla_srnamt, "DSTEDC");
        xerbla_infot = 1;
        dstedc("/", 0, d, e, z, 1, w, 1, iw, 1, &info);
        chkxer("DSTEDC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        dstedc("N", -1, d, e, z, 1, w, 1, iw, 1, &info);
        chkxer("DSTEDC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        dstedc("V", 2, d, e, z, 1, w, 23, iw, 28, &info);
        chkxer("DSTEDC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        dstedc("N", 1, d, e, z, 1, w, 0, iw, 1, &info);
        chkxer("DSTEDC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        dstedc("I", 2, d, e, z, 2, w, 0, iw, 12, &info);
        chkxer("DSTEDC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        dstedc("V", 2, d, e, z, 2, w, 0, iw, 28, &info);
        chkxer("DSTEDC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        dstedc("N", 1, d, e, z, 1, w, 1, iw, 0, &info);
        chkxer("DSTEDC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        dstedc("I", 2, d, e, z, 2, w, 19, iw, 0, &info);
        chkxer("DSTEDC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        dstedc("V", 2, d, e, z, 2, w, 23, iw, 0, &info);
        chkxer("DSTEDC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 9;

        /* DSTEVD */

        strcpy(xerbla_srnamt, "DSTEVD");
        xerbla_infot = 1;
        dstevd("/", 0, d, e, z, 1, w, 1, iw, 1, &info);
        chkxer("DSTEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        dstevd("N", -1, d, e, z, 1, w, 1, iw, 1, &info);
        chkxer("DSTEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        dstevd("V", 2, d, e, z, 1, w, 19, iw, 12, &info);
        chkxer("DSTEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        dstevd("N", 1, d, e, z, 1, w, 0, iw, 1, &info);
        chkxer("DSTEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        dstevd("V", 2, d, e, z, 2, w, 12, iw, 12, &info);
        chkxer("DSTEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        dstevd("N", 0, d, e, z, 1, w, 1, iw, 0, &info);
        chkxer("DSTEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        dstevd("V", 2, d, e, z, 2, w, 19, iw, 11, &info);
        chkxer("DSTEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 7;

        /* DSTEV */

        strcpy(xerbla_srnamt, "DSTEV ");
        xerbla_infot = 1;
        dstev("/", 0, d, e, z, 1, w, &info);
        chkxer("DSTEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        dstev("N", -1, d, e, z, 1, w, &info);
        chkxer("DSTEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        dstev("V", 2, d, e, z, 1, w, &info);
        chkxer("DSTEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 3;

        /* DSTEVX */

        strcpy(xerbla_srnamt, "DSTEVX");
        xerbla_infot = 1;
        dstevx("/", "A", 0, d, e, 0.0, 0.0, -1, -1, 0.0, &m,
               x, z, 1, w, iw, i3, &info);
        chkxer("DSTEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        dstevx("N", "/", 0, d, e, 0.0, 1.0, 0, -1, 0.0, &m,
               x, z, 1, w, iw, i3, &info);
        chkxer("DSTEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        dstevx("N", "A", -1, d, e, 0.0, 0.0, -1, -1, 0.0, &m,
               x, z, 1, w, iw, i3, &info);
        chkxer("DSTEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        dstevx("N", "V", 1, d, e, 0.0, 0.0, -1, -1, 0.0, &m,
               x, z, 1, w, iw, i3, &info);
        chkxer("DSTEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        dstevx("N", "I", 1, d, e, 0.0, 0.0, -1, -1, 0.0, &m,
               x, z, 1, w, iw, i3, &info);
        chkxer("DSTEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        dstevx("N", "I", 1, d, e, 0.0, 0.0, 1, 0, 0.0, &m,
               x, z, 1, w, iw, i3, &info);
        chkxer("DSTEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        dstevx("N", "I", 2, d, e, 0.0, 0.0, 1, 0, 0.0, &m,
               x, z, 1, w, iw, i3, &info);
        chkxer("DSTEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        dstevx("N", "I", 1, d, e, 0.0, 0.0, 0, 1, 0.0, &m,
               x, z, 1, w, iw, i3, &info);
        chkxer("DSTEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 14;
        dstevx("V", "A", 2, d, e, 0.0, 0.0, -1, -1, 0.0, &m,
               x, z, 1, w, iw, i3, &info);
        chkxer("DSTEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 9;

        /* DSTEVR */

        n = 1;
        strcpy(xerbla_srnamt, "DSTEVR");
        xerbla_infot = 1;
        dstevr("/", "A", 0, d, e, 0.0, 0.0, 0, 0, 0.0, &m,
               r, z, 1, iw, x, 20 * n, &iw[2 * n], 10 * n, &info);
        chkxer("DSTEVR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        dstevr("V", "/", 0, d, e, 0.0, 0.0, 0, 0, 0.0, &m,
               r, z, 1, iw, x, 20 * n, &iw[2 * n], 10 * n, &info);
        chkxer("DSTEVR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        dstevr("V", "A", -1, d, e, 0.0, 0.0, 0, 0, 0.0, &m,
               r, z, 1, iw, x, 20 * n, &iw[2 * n], 10 * n, &info);
        chkxer("DSTEVR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        dstevr("V", "V", 1, d, e, 0.0, 0.0, 0, 0, 0.0, &m,
               r, z, 1, iw, x, 20 * n, &iw[2 * n], 10 * n, &info);
        chkxer("DSTEVR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        dstevr("V", "I", 1, d, e, 0.0, 0.0, -1, 0, 0.0, &m,
               w, z, 1, iw, x, 20 * n, &iw[2 * n], 10 * n, &info);
        chkxer("DSTEVR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        n = 2;
        dstevr("V", "I", 2, d, e, 0.0, 0.0, 1, 0, 0.0, &m,
               w, z, 1, iw, x, 20 * n, &iw[2 * n], 10 * n, &info);
        chkxer("DSTEVR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 14;
        n = 1;
        dstevr("V", "I", 1, d, e, 0.0, 0.0, 0, 0, 0.0, &m,
               w, z, 0, iw, x, 20 * n, &iw[2 * n], 10 * n, &info);
        chkxer("DSTEVR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 17;
        dstevr("V", "I", 1, d, e, 0.0, 0.0, 0, 0, 0.0, &m,
               w, z, 1, iw, x, 20 * n - 1, &iw[2 * n], 10 * n, &info);
        chkxer("DSTEVR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 19;
        dstevr("V", "I", 1, d, e, 0.0, 0.0, 0, 0, 0.0, &m,
               w, z, 1, iw, x, 20 * n, &iw[2 * n], 10 * n - 1, &info);
        chkxer("DSTEVR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 9;

        /* DSYEVD */

        strcpy(xerbla_srnamt, "DSYEVD");
        xerbla_infot = 1;
        dsyevd("/", "U", 0, a, 1, x, w, 1, iw, 1, &info);
        chkxer("DSYEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        dsyevd("N", "/", 0, a, 1, x, w, 1, iw, 1, &info);
        chkxer("DSYEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        dsyevd("N", "U", -1, a, 1, x, w, 1, iw, 1, &info);
        chkxer("DSYEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        dsyevd("N", "U", 2, a, 1, x, w, 3, iw, 1, &info);
        chkxer("DSYEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        dsyevd("N", "U", 1, a, 1, x, w, 0, iw, 1, &info);
        chkxer("DSYEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        dsyevd("N", "U", 2, a, 2, x, w, 4, iw, 1, &info);
        chkxer("DSYEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        dsyevd("V", "U", 2, a, 2, x, w, 20, iw, 12, &info);
        chkxer("DSYEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        dsyevd("N", "U", 1, a, 1, x, w, 1, iw, 0, &info);
        chkxer("DSYEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        dsyevd("N", "U", 2, a, 2, x, w, 5, iw, 0, &info);
        chkxer("DSYEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        dsyevd("V", "U", 2, a, 2, x, w, 27, iw, 11, &info);
        chkxer("DSYEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 10;

        /* DSYEVD_2STAGE */

        strcpy(xerbla_srnamt, "DSYEVD_2STAGE");
        xerbla_infot = 1;
        dsyevd_2stage("/", "U", 0, a, 1, x, w, 1, iw, 1, &info);
        chkxer("DSYEVD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 1;
        dsyevd_2stage("V", "U", 0, a, 1, x, w, 1, iw, 1, &info);
        chkxer("DSYEVD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        dsyevd_2stage("N", "/", 0, a, 1, x, w, 1, iw, 1, &info);
        chkxer("DSYEVD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        dsyevd_2stage("N", "U", -1, a, 1, x, w, 1, iw, 1, &info);
        chkxer("DSYEVD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        dsyevd_2stage("N", "U", 2, a, 1, x, w, 3, iw, 1, &info);
        chkxer("DSYEVD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        dsyevd_2stage("N", "U", 1, a, 1, x, w, 0, iw, 1, &info);
        chkxer("DSYEVD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        dsyevd_2stage("N", "U", 2, a, 2, x, w, 4, iw, 1, &info);
        chkxer("DSYEVD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        /* xerbla_infot = 8;
           dsyevd_2stage("V", "U", 2, a, 2, x, w, 20, iw, 12, &info);
           chkxer("DSYEVD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok); */
        xerbla_infot = 10;
        dsyevd_2stage("N", "U", 1, a, 1, x, w, 1, iw, 0, &info);
        chkxer("DSYEVD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        dsyevd_2stage("N", "U", 2, a, 2, x, w, 25, iw, 0, &info);
        chkxer("DSYEVD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        /* xerbla_infot = 10;
           dsyevd_2stage("V", "U", 2, a, 2, x, w, 27, iw, 11, &info);
           chkxer("DSYEVD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok); */
        *nt = *nt + 9;

        /* DSYEVR */

        strcpy(xerbla_srnamt, "DSYEVR");
        n = 1;
        xerbla_infot = 1;
        dsyevr("/", "A", "U", 0, a, 1, 0.0, 0.0, 0, 0, 0.0,
               &m, r, z, 1, iw, q, 26 * n, &iw[2 * n], 10 * n, &info);
        chkxer("DSYEVR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        dsyevr("V", "/", "U", 0, a, 1, 0.0, 0.0, 0, 0, 0.0,
               &m, r, z, 1, iw, q, 26 * n, &iw[2 * n], 10 * n, &info);
        chkxer("DSYEVR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        dsyevr("V", "A", "/", -1, a, 1, 0.0, 0.0, 0, 0,
               0.0, &m, r, z, 1, iw, q, 26 * n, &iw[2 * n], 10 * n,
               &info);
        chkxer("DSYEVR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        dsyevr("V", "A", "U", -1, a, 1, 0.0, 0.0, 0, 0,
               0.0, &m, r, z, 1, iw, q, 26 * n, &iw[2 * n], 10 * n,
               &info);
        chkxer("DSYEVR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        dsyevr("V", "A", "U", 2, a, 1, 0.0, 0.0, 0, 0, 0.0,
               &m, r, z, 1, iw, q, 26 * n, &iw[2 * n], 10 * n, &info);
        chkxer("DSYEVR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        dsyevr("V", "V", "U", 1, a, 1, 0.0, 0.0, 0, 0, 0.0,
               &m, r, z, 1, iw, q, 26 * n, &iw[2 * n], 10 * n, &info);
        chkxer("DSYEVR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        dsyevr("V", "I", "U", 1, a, 1, 0.0, 0.0, -1, 0, 0.0,
               &m, r, z, 1, iw, q, 26 * n, &iw[2 * n], 10 * n, &info);
        chkxer("DSYEVR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;

        dsyevr("V", "I", "U", 2, a, 2, 0.0, 0.0, 1, 0, 0.0,
               &m, r, z, 1, iw, q, 26 * n, &iw[2 * n], 10 * n, &info);
        chkxer("DSYEVR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 15;
        dsyevr("V", "I", "U", 1, a, 1, 0.0, 0.0, 0, 0, 0.0,
               &m, r, z, 0, iw, q, 26 * n, &iw[2 * n], 10 * n, &info);
        chkxer("DSYEVR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 18;
        dsyevr("V", "I", "U", 1, a, 1, 0.0, 0.0, 0, 0, 0.0,
               &m, r, z, 1, iw, q, 0, &iw[2 * n], 10 * n,
               &info);
        chkxer("DSYEVR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 20;
        dsyevr("V", "I", "U", 1, a, 1, 0.0, 0.0, 0, 0, 0.0,
               &m, r, z, 1, iw, q, 26 * n, &iw[2 * n], 0,
               &info);
        chkxer("DSYEVR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 11;

        /* DSYEVR_2STAGE */

        strcpy(xerbla_srnamt, "DSYEVR_2STAGE");
        n = 1;
        xerbla_infot = 1;
        dsyevr_2stage("/", "A", "U", 0, a, 1,
                      0.0, 0.0, 0, 0, 0.0,
                      &m, r, z, 1, iw, q, 26 * n, &iw[2 * n], 10 * n, &info);
        chkxer("DSYEVR_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 1;
        dsyevr_2stage("V", "A", "U", 0, a, 1,
                      0.0, 0.0, 0, 0, 0.0,
                      &m, r, z, 1, iw, q, 26 * n, &iw[2 * n], 10 * n, &info);
        chkxer("DSYEVR_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        dsyevr_2stage("N", "/", "U", 0, a, 1,
                      0.0, 0.0, 0, 0, 0.0,
                      &m, r, z, 1, iw, q, 26 * n, &iw[2 * n], 10 * n, &info);
        chkxer("DSYEVR_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        dsyevr_2stage("N", "A", "/", -1, a, 1,
                      0.0, 0.0, 0, 0, 0.0,
                      &m, r, z, 1, iw, q, 26 * n, &iw[2 * n], 10 * n, &info);
        chkxer("DSYEVR_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        dsyevr_2stage("N", "A", "U", -1, a, 1,
                      0.0, 0.0, 0, 0, 0.0,
                      &m, r, z, 1, iw, q, 26 * n, &iw[2 * n], 10 * n, &info);
        chkxer("DSYEVR_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        dsyevr_2stage("N", "A", "U", 2, a, 1,
                      0.0, 0.0, 0, 0, 0.0,
                      &m, r, z, 1, iw, q, 26 * n, &iw[2 * n], 10 * n, &info);
        chkxer("DSYEVR_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        dsyevr_2stage("N", "V", "U", 1, a, 1,
                      0.0, 0.0, 0, 0, 0.0,
                      &m, r, z, 1, iw, q, 26 * n, &iw[2 * n], 10 * n, &info);
        chkxer("DSYEVR_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        dsyevr_2stage("N", "I", "U", 1, a, 1,
                      0.0, 0.0, -1, 0, 0.0,
                      &m, r, z, 1, iw, q, 26 * n, &iw[2 * n], 10 * n, &info);
        chkxer("DSYEVR_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        dsyevr_2stage("N", "I", "U", 2, a, 2,
                      0.0, 0.0, 1, 0, 0.0,
                      &m, r, z, 1, iw, q, 26 * n, &iw[2 * n], 10 * n, &info);
        chkxer("DSYEVR_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 15;
        dsyevr_2stage("N", "I", "U", 1, a, 1,
                      0.0, 0.0, 0, 0, 0.0,
                      &m, r, z, 0, iw, q, 26 * n, &iw[2 * n], 10 * n, &info);
        chkxer("DSYEVR_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 18;
        dsyevr_2stage("N", "I", "U", 1, a, 1,
                      0.0, 0.0, 0, 0, 0.0,
                      &m, r, z, 1, iw, q, 0, &iw[2 * n], 10 * n,
                      &info);
        chkxer("DSYEVR_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 20;
        dsyevr_2stage("N", "I", "U", 1, a, 1,
                      0.0, 0.0, 0, 0, 0.0,
                      &m, r, z, 1, iw, q, 26 * n, &iw[2 * n], 0,
                      &info);
        chkxer("DSYEVR_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 12;

        /* DSYEV */

        strcpy(xerbla_srnamt, "DSYEV ");
        xerbla_infot = 1;
        dsyev("/", "U", 0, a, 1, x, w, 1, &info);
        chkxer("DSYEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        dsyev("N", "/", 0, a, 1, x, w, 1, &info);
        chkxer("DSYEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        dsyev("N", "U", -1, a, 1, x, w, 1, &info);
        chkxer("DSYEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        dsyev("N", "U", 2, a, 1, x, w, 3, &info);
        chkxer("DSYEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        dsyev("N", "U", 1, a, 1, x, w, 1, &info);
        chkxer("DSYEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 5;

        /* DSYEV_2STAGE */

        strcpy(xerbla_srnamt, "DSYEV_2STAGE ");
        xerbla_infot = 1;
        dsyev_2stage("/", "U", 0, a, 1, x, w, 1, &info);
        chkxer("DSYEV_2STAGE ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 1;
        dsyev_2stage("V", "U", 0, a, 1, x, w, 1, &info);
        chkxer("DSYEV_2STAGE ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        dsyev_2stage("N", "/", 0, a, 1, x, w, 1, &info);
        chkxer("DSYEV_2STAGE ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        dsyev_2stage("N", "U", -1, a, 1, x, w, 1, &info);
        chkxer("DSYEV_2STAGE ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        dsyev_2stage("N", "U", 2, a, 1, x, w, 3, &info);
        chkxer("DSYEV_2STAGE ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        dsyev_2stage("N", "U", 1, a, 1, x, w, 1, &info);
        chkxer("DSYEV_2STAGE ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 6;

        /* DSYEVX */

        strcpy(xerbla_srnamt, "DSYEVX");
        xerbla_infot = 1;
        dsyevx("/", "A", "U", 0, a, 1, 0.0, 0.0, -1, -1, 0.0,
               &m, x, z, 1, w, 1, iw, i3, &info);
        chkxer("DSYEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        dsyevx("N", "/", "U", 0, a, 1, 0.0, 1.0, 0, -1, 0.0,
               &m, x, z, 1, w, 1, iw, i3, &info);
        chkxer("DSYEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        dsyevx("N", "A", "/", 0, a, 1, 0.0, 0.0, -1, -1, 0.0,
               &m, x, z, 1, w, 1, iw, i3, &info);
        xerbla_infot = 4;
        dsyevx("N", "A", "U", -1, a, 1, 0.0, 0.0, -1, -1,
               0.0, &m, x, z, 1, w, 1, iw, i3, &info);
        chkxer("DSYEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        dsyevx("N", "A", "U", 2, a, 1, 0.0, 0.0, -1, -1, 0.0,
               &m, x, z, 1, w, 16, iw, i3, &info);
        chkxer("DSYEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        dsyevx("N", "V", "U", 1, a, 1, 0.0, 0.0, -1, -1, 0.0,
               &m, x, z, 1, w, 8, iw, i3, &info);
        chkxer("DSYEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        dsyevx("N", "I", "U", 1, a, 1, 0.0, 0.0, -1, -1, 0.0,
               &m, x, z, 1, w, 8, iw, i3, &info);
        chkxer("DSYEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        dsyevx("N", "I", "U", 1, a, 1, 0.0, 0.0, 1, 0, 0.0,
               &m, x, z, 1, w, 8, iw, i3, &info);
        chkxer("DSYEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        dsyevx("N", "I", "U", 2, a, 2, 0.0, 0.0, 1, 0, 0.0,
               &m, x, z, 1, w, 16, iw, i3, &info);
        chkxer("DSYEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        dsyevx("N", "I", "U", 1, a, 1, 0.0, 0.0, 0, 1, 0.0,
               &m, x, z, 1, w, 8, iw, i3, &info);
        chkxer("DSYEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 15;
        dsyevx("V", "A", "U", 2, a, 2, 0.0, 0.0, -1, -1, 0.0,
               &m, x, z, 1, w, 16, iw, i3, &info);
        chkxer("DSYEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 17;
        dsyevx("V", "A", "U", 1, a, 1, 0.0, 0.0, -1, -1, 0.0,
               &m, x, z, 1, w, 0, iw, i3, &info);
        chkxer("DSYEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 12;

        /* DSYEVX_2STAGE */

        strcpy(xerbla_srnamt, "DSYEVX_2STAGE");
        xerbla_infot = 1;
        dsyevx_2stage("/", "A", "U", 0, a, 1,
                      0.0, 0.0, -1, -1, 0.0,
                      &m, x, z, 1, w, 1, iw, i3, &info);
        chkxer("DSYEVX_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 1;
        dsyevx_2stage("V", "A", "U", 0, a, 1,
                      0.0, 0.0, -1, -1, 0.0,
                      &m, x, z, 1, w, 1, iw, i3, &info);
        chkxer("DSYEVX_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        dsyevx_2stage("N", "/", "U", 0, a, 1,
                      0.0, 1.0, 0, -1, 0.0,
                      &m, x, z, 1, w, 1, iw, i3, &info);
        chkxer("DSYEVX_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        dsyevx_2stage("N", "A", "/", 0, a, 1,
                      0.0, 0.0, -1, -1, 0.0,
                      &m, x, z, 1, w, 1, iw, i3, &info);
        xerbla_infot = 4;
        dsyevx_2stage("N", "A", "U", -1, a, 1,
                      0.0, 0.0, -1, -1, 0.0,
                      &m, x, z, 1, w, 1, iw, i3, &info);
        chkxer("DSYEVX_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        dsyevx_2stage("N", "A", "U", 2, a, 1,
                      0.0, 0.0, -1, -1, 0.0,
                      &m, x, z, 1, w, 16, iw, i3, &info);
        chkxer("DSYEVX_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        dsyevx_2stage("N", "V", "U", 1, a, 1,
                      0.0, 0.0, -1, -1, 0.0,
                      &m, x, z, 1, w, 8, iw, i3, &info);
        chkxer("DSYEVX_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        dsyevx_2stage("N", "I", "U", 1, a, 1,
                      0.0, 0.0, -1, -1, 0.0,
                      &m, x, z, 1, w, 8, iw, i3, &info);
        chkxer("DSYEVX_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        dsyevx_2stage("N", "I", "U", 1, a, 1,
                      0.0, 0.0, 1, 0, 0.0,
                      &m, x, z, 1, w, 8, iw, i3, &info);
        chkxer("DSYEVX_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        dsyevx_2stage("N", "I", "U", 2, a, 2,
                      0.0, 0.0, 1, 0, 0.0,
                      &m, x, z, 1, w, 16, iw, i3, &info);
        chkxer("DSYEVX_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        dsyevx_2stage("N", "I", "U", 1, a, 1,
                      0.0, 0.0, 0, 1, 0.0,
                      &m, x, z, 1, w, 8, iw, i3, &info);
        chkxer("DSYEVX_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 15;
        dsyevx_2stage("N", "A", "U", 2, a, 2,
                      0.0, 0.0, -1, -1, 0.0,
                      &m, x, z, 0, w, 16, iw, i3, &info);
        chkxer("DSYEVX_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 17;
        dsyevx_2stage("N", "A", "U", 1, a, 1,
                      0.0, 0.0, -1, -1, 0.0,
                      &m, x, z, 1, w, 0, iw, i3, &info);
        chkxer("DSYEVX_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 13;

        /* DSPEVD */

        strcpy(xerbla_srnamt, "DSPEVD");
        xerbla_infot = 1;
        dspevd("/", "U", 0, a, x, z, 1, w, 1, iw, 1, &info);
        chkxer("DSPEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        dspevd("N", "/", 0, a, x, z, 1, w, 1, iw, 1, &info);
        chkxer("DSPEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        dspevd("N", "U", -1, a, x, z, 1, w, 1, iw, 1, &info);
        chkxer("DSPEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        dspevd("V", "U", 2, a, x, z, 1, w, 23, iw, 12, &info);
        chkxer("DSPEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        dspevd("N", "U", 1, a, x, z, 1, w, 0, iw, 1, &info);
        chkxer("DSPEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        dspevd("N", "U", 2, a, x, z, 1, w, 3, iw, 1, &info);
        chkxer("DSPEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        dspevd("V", "U", 2, a, x, z, 2, w, 16, iw, 12, &info);
        chkxer("DSPEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 11;
        dspevd("N", "U", 1, a, x, z, 1, w, 1, iw, 0, &info);
        chkxer("DSPEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 11;
        dspevd("N", "U", 2, a, x, z, 1, w, 4, iw, 0, &info);
        chkxer("DSPEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 11;
        dspevd("V", "U", 2, a, x, z, 2, w, 23, iw, 11, &info);
        chkxer("DSPEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 10;

        /* DSPEV */

        strcpy(xerbla_srnamt, "DSPEV ");
        xerbla_infot = 1;
        dspev("/", "U", 0, a, w, z, 1, x, &info);
        chkxer("DSPEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        dspev("N", "/", 0, a, w, z, 1, x, &info);
        chkxer("DSPEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        dspev("N", "U", -1, a, w, z, 1, x, &info);
        chkxer("DSPEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        dspev("V", "U", 2, a, w, z, 1, x, &info);
        chkxer("DSPEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 4;

        /* DSPEVX */

        strcpy(xerbla_srnamt, "DSPEVX");
        xerbla_infot = 1;
        dspevx("/", "A", "U", 0, a, 0.0, 0.0, -1, -1, 0.0, &m,
               x, z, 1, w, iw, i3, &info);
        chkxer("DSPEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        dspevx("N", "/", "U", 0, a, 0.0, 0.0, -1, -1, 0.0, &m,
               x, z, 1, w, iw, i3, &info);
        chkxer("DSPEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        dspevx("N", "A", "/", 0, a, 0.0, 0.0, -1, -1, 0.0, &m,
               x, z, 1, w, iw, i3, &info);
        xerbla_infot = 4;
        dspevx("N", "A", "U", -1, a, 0.0, 0.0, -1, -1, 0.0,
               &m, x, z, 1, w, iw, i3, &info);
        chkxer("DSPEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        dspevx("N", "V", "U", 1, a, 0.0, 0.0, -1, -1, 0.0, &m,
               x, z, 1, w, iw, i3, &info);
        chkxer("DSPEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        dspevx("N", "I", "U", 1, a, 0.0, 0.0, -1, -1, 0.0, &m,
               x, z, 1, w, iw, i3, &info);
        chkxer("DSPEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        dspevx("N", "I", "U", 1, a, 0.0, 0.0, 1, 0, 0.0, &m,
               x, z, 1, w, iw, i3, &info);
        chkxer("DSPEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        dspevx("N", "I", "U", 2, a, 0.0, 0.0, 1, 0, 0.0, &m,
               x, z, 1, w, iw, i3, &info);
        chkxer("DSPEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        dspevx("N", "I", "U", 1, a, 0.0, 0.0, 0, 1, 0.0, &m,
               x, z, 1, w, iw, i3, &info);
        chkxer("DSPEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 14;
        dspevx("V", "A", "U", 2, a, 0.0, 0.0, -1, -1, 0.0, &m,
               x, z, 1, w, iw, i3, &info);
        chkxer("DSPEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 10;

    /* Test error exits for the SB path. */
    } else if ((path[1] == 'S' || path[1] == 's') &&
               (path[2] == 'B' || path[2] == 'b')) {

        /* DSBTRD */

        strcpy(xerbla_srnamt, "DSBTRD");
        xerbla_infot = 1;
        dsbtrd("/", "U", 0, 0, a, 1, d, e, z, 1, w, &info);
        chkxer("DSBTRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        dsbtrd("N", "/", 0, 0, a, 1, d, e, z, 1, w, &info);
        chkxer("DSBTRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        dsbtrd("N", "U", -1, 0, a, 1, d, e, z, 1, w, &info);
        chkxer("DSBTRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        dsbtrd("N", "U", 0, -1, a, 1, d, e, z, 1, w, &info);
        chkxer("DSBTRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        dsbtrd("N", "U", 1, 1, a, 1, d, e, z, 1, w, &info);
        chkxer("DSBTRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        dsbtrd("V", "U", 2, 0, a, 1, d, e, z, 1, w, &info);
        chkxer("DSBTRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 6;

        /* DSYTRD_SB2ST */

        strcpy(xerbla_srnamt, "DSYTRD_SB2ST");
        xerbla_infot = 1;
        dsytrd_sb2st("/", "N", "U", 0, 0, a, 1, d, e,
                     c, 1, w, 1, &info);
        chkxer("DSYTRD_SB2ST", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        dsytrd_sb2st("N", "/", "U", 0, 0, a, 1, d, e,
                     c, 1, w, 1, &info);
        chkxer("DSYTRD_SB2ST", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        dsytrd_sb2st("N", "H", "U", 0, 0, a, 1, d, e,
                     c, 1, w, 1, &info);
        chkxer("DSYTRD_SB2ST", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        dsytrd_sb2st("N", "N", "/", 0, 0, a, 1, d, e,
                     c, 1, w, 1, &info);
        chkxer("DSYTRD_SB2ST", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        dsytrd_sb2st("N", "N", "U", -1, 0, a, 1, d, e,
                     c, 1, w, 1, &info);
        chkxer("DSYTRD_SB2ST", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        dsytrd_sb2st("N", "N", "U", 0, -1, a, 1, d, e,
                     c, 1, w, 1, &info);
        chkxer("DSYTRD_SB2ST", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        dsytrd_sb2st("N", "N", "U", 0, 1, a, 1, d, e,
                     c, 1, w, 1, &info);
        chkxer("DSYTRD_SB2ST", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 11;
        dsytrd_sb2st("N", "N", "U", 0, 0, a, 1, d, e,
                     c, 0, w, 1, &info);
        chkxer("DSYTRD_SB2ST", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 13;
        dsytrd_sb2st("N", "N", "U", 0, 0, a, 1, d, e,
                     c, 1, w, 0, &info);
        chkxer("DSYTRD_SB2ST", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 9;

        /* DSBEVD */

        strcpy(xerbla_srnamt, "DSBEVD");
        xerbla_infot = 1;
        dsbevd("/", "U", 0, 0, a, 1, x, z, 1, w, 1, iw, 1, &info);
        chkxer("DSBEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        dsbevd("N", "/", 0, 0, a, 1, x, z, 1, w, 1, iw, 1, &info);
        chkxer("DSBEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        dsbevd("N", "U", -1, 0, a, 1, x, z, 1, w, 1, iw, 1,
               &info);
        chkxer("DSBEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        dsbevd("N", "U", 0, -1, a, 1, x, z, 1, w, 1, iw, 1,
               &info);
        chkxer("DSBEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        dsbevd("N", "U", 2, 1, a, 1, x, z, 1, w, 4, iw, 1, &info);
        chkxer("DSBEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        dsbevd("V", "U", 2, 1, a, 2, x, z, 1, w, 25, iw, 12,
               &info);
        chkxer("DSBEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 11;
        dsbevd("N", "U", 1, 0, a, 1, x, z, 1, w, 0, iw, 1, &info);
        chkxer("DSBEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 11;
        dsbevd("N", "U", 2, 0, a, 1, x, z, 1, w, 3, iw, 1, &info);
        chkxer("DSBEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 11;
        dsbevd("V", "U", 2, 0, a, 1, x, z, 2, w, 18, iw, 12,
               &info);
        chkxer("DSBEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 13;
        dsbevd("N", "U", 1, 0, a, 1, x, z, 1, w, 1, iw, 0, &info);
        chkxer("DSBEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 13;
        dsbevd("V", "U", 2, 0, a, 1, x, z, 2, w, 25, iw, 11,
               &info);
        chkxer("DSBEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 11;

        /* DSBEVD_2STAGE */

        strcpy(xerbla_srnamt, "DSBEVD_2STAGE");
        xerbla_infot = 1;
        dsbevd_2stage("/", "U", 0, 0, a, 1, x, z, 1, w,
                      1, iw, 1, &info);
        chkxer("DSBEVD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 1;
        dsbevd_2stage("V", "U", 0, 0, a, 1, x, z, 1, w,
                      1, iw, 1, &info);
        chkxer("DSBEVD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        dsbevd_2stage("N", "/", 0, 0, a, 1, x, z, 1, w,
                      1, iw, 1, &info);
        chkxer("DSBEVD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        dsbevd_2stage("N", "U", -1, 0, a, 1, x, z, 1, w,
                      1, iw, 1, &info);
        chkxer("DSBEVD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        dsbevd_2stage("N", "U", 0, -1, a, 1, x, z, 1, w,
                      1, iw, 1, &info);
        chkxer("DSBEVD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        dsbevd_2stage("N", "U", 2, 1, a, 1, x, z, 1, w,
                      4, iw, 1, &info);
        chkxer("DSBEVD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        /* xerbla_infot = 9;
           dsbevd_2stage("V", "U", 2, 1, a, 2, x, z, 1, w,
                         25, iw, 12, &info);
           chkxer("DSBEVD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok); */
        xerbla_infot = 11;
        dsbevd_2stage("N", "U", 1, 0, a, 1, x, z, 1, w,
                      0, iw, 1, &info);
        chkxer("DSBEVD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 11;
        dsbevd_2stage("N", "U", 2, 0, a, 1, x, z, 1, w,
                      3, iw, 1, &info);
        chkxer("DSBEVD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        /* xerbla_infot = 11;
           dsbevd_2stage("V", "U", 2, 0, a, 1, x, z, 2, w,
                         18, iw, 12, &info);
           chkxer("DSBEVD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok); */
        xerbla_infot = 13;
        dsbevd_2stage("N", "U", 1, 0, a, 1, x, z, 1, w,
                      1, iw, 0, &info);
        chkxer("DSBEVD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        /* xerbla_infot = 13;
           dsbevd_2stage("V", "U", 2, 0, a, 1, x, z, 2, w,
                         25, iw, 11, &info);
           chkxer("DSBEVD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
           *nt = *nt + 12; */
        *nt = *nt + 9;

        /* DSBEV */

        strcpy(xerbla_srnamt, "DSBEV ");
        xerbla_infot = 1;
        dsbev("/", "U", 0, 0, a, 1, x, z, 1, w, &info);
        chkxer("DSBEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        dsbev("N", "/", 0, 0, a, 1, x, z, 1, w, &info);
        chkxer("DSBEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        dsbev("N", "U", -1, 0, a, 1, x, z, 1, w, &info);
        chkxer("DSBEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        dsbev("N", "U", 0, -1, a, 1, x, z, 1, w, &info);
        chkxer("DSBEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        dsbev("N", "U", 2, 1, a, 1, x, z, 1, w, &info);
        chkxer("DSBEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        dsbev("V", "U", 2, 0, a, 1, x, z, 1, w, &info);
        chkxer("DSBEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 6;

        /* DSBEV_2STAGE */

        strcpy(xerbla_srnamt, "DSBEV_2STAGE ");
        xerbla_infot = 1;
        dsbev_2stage("/", "U", 0, 0, a, 1, x, z, 1, w, 0, &info);
        chkxer("DSBEV_2STAGE ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 1;
        dsbev_2stage("V", "U", 0, 0, a, 1, x, z, 1, w, 0, &info);
        chkxer("DSBEV_2STAGE ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        dsbev_2stage("N", "/", 0, 0, a, 1, x, z, 1, w, 0, &info);
        chkxer("DSBEV_2STAGE ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        dsbev_2stage("N", "U", -1, 0, a, 1, x, z, 1, w, 0, &info);
        chkxer("DSBEV_2STAGE ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        dsbev_2stage("N", "U", 0, -1, a, 1, x, z, 1, w, 0, &info);
        chkxer("DSBEV_2STAGE ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        dsbev_2stage("N", "U", 2, 1, a, 1, x, z, 1, w, 0, &info);
        chkxer("DSBEV_2STAGE ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        dsbev_2stage("N", "U", 2, 0, a, 1, x, z, 0, w, 0, &info);
        chkxer("DSBEV_2STAGE ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 11;
        dsbev_2stage("N", "U", 0, 0, a, 1, x, z, 1, w, 0, &info);
        chkxer("DSBEV_2STAGE ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 8;

        /* DSBEVX */

        strcpy(xerbla_srnamt, "DSBEVX");
        xerbla_infot = 1;
        dsbevx("/", "A", "U", 0, 0, a, 1, q, 1, 0.0, 0.0, -1,
               -1, 0.0, &m, x, z, 1, w, iw, i3, &info);
        chkxer("DSBEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        dsbevx("N", "/", "U", 0, 0, a, 1, q, 1, 0.0, 0.0, -1,
               -1, 0.0, &m, x, z, 1, w, iw, i3, &info);
        chkxer("DSBEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        dsbevx("N", "A", "/", 0, 0, a, 1, q, 1, 0.0, 0.0, -1,
               -1, 0.0, &m, x, z, 1, w, iw, i3, &info);
        chkxer("DSBEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        dsbevx("N", "A", "U", -1, 0, a, 1, q, 1, 0.0, 0.0, -1,
               -1, 0.0, &m, x, z, 1, w, iw, i3, &info);
        chkxer("DSBEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        dsbevx("N", "A", "U", 0, -1, a, 1, q, 1, 0.0, 0.0, -1,
               -1, 0.0, &m, x, z, 1, w, iw, i3, &info);
        chkxer("DSBEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        dsbevx("N", "A", "U", 2, 1, a, 1, q, 1, 0.0, 0.0, -1,
               -1, 0.0, &m, x, z, 1, w, iw, i3, &info);
        chkxer("DSBEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        dsbevx("V", "A", "U", 2, 0, a, 1, q, 1, 0.0, 0.0, -1,
               -1, 0.0, &m, x, z, 2, w, iw, i3, &info);
        chkxer("DSBEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 11;
        dsbevx("N", "V", "U", 1, 0, a, 1, q, 1, 0.0, 0.0, -1,
               -1, 0.0, &m, x, z, 1, w, iw, i3, &info);
        chkxer("DSBEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 12;
        dsbevx("N", "I", "U", 1, 0, a, 1, q, 1, 0.0, 0.0, -1,
               -1, 0.0, &m, x, z, 1, w, iw, i3, &info);
        chkxer("DSBEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 12;
        dsbevx("N", "I", "U", 1, 0, a, 1, q, 1, 0.0, 0.0, 1,
               0, 0.0, &m, x, z, 1, w, iw, i3, &info);
        chkxer("DSBEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 13;
        dsbevx("N", "I", "U", 2, 0, a, 1, q, 1, 0.0, 0.0, 1,
               0, 0.0, &m, x, z, 1, w, iw, i3, &info);
        chkxer("DSBEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 13;
        dsbevx("N", "I", "U", 1, 0, a, 1, q, 1, 0.0, 0.0, 0,
               1, 0.0, &m, x, z, 1, w, iw, i3, &info);
        chkxer("DSBEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 18;
        dsbevx("V", "A", "U", 2, 0, a, 1, q, 2, 0.0, 0.0, -1,
               -1, 0.0, &m, x, z, 1, w, iw, i3, &info);
        chkxer("DSBEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 13;

        /* DSBEVX_2STAGE */

        strcpy(xerbla_srnamt, "DSBEVX_2STAGE");
        xerbla_infot = 1;
        dsbevx_2stage("/", "A", "U", 0, 0, a, 1, q, 1, 0.0,
                      0.0, -1, -1, 0.0, &m, x, z, 1, w, 0, iw, i3, &info);
        chkxer("DSBEVX_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 1;
        dsbevx_2stage("V", "A", "U", 0, 0, a, 1, q, 1, 0.0,
                      0.0, -1, -1, 0.0, &m, x, z, 1, w, 0, iw, i3, &info);
        chkxer("DSBEVX_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        dsbevx_2stage("N", "/", "U", 0, 0, a, 1, q, 1, 0.0,
                      0.0, -1, -1, 0.0, &m, x, z, 1, w, 0, iw, i3, &info);
        chkxer("DSBEVX_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        dsbevx_2stage("N", "A", "/", 0, 0, a, 1, q, 1, 0.0,
                      0.0, -1, -1, 0.0, &m, x, z, 1, w, 0, iw, i3, &info);
        chkxer("DSBEVX_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        dsbevx_2stage("N", "A", "U", -1, 0, a, 1, q, 1, 0.0,
                      0.0, -1, -1, 0.0, &m, x, z, 1, w, 0, iw, i3, &info);
        chkxer("DSBEVX_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        dsbevx_2stage("N", "A", "U", 0, -1, a, 1, q, 1, 0.0,
                      0.0, -1, -1, 0.0, &m, x, z, 1, w, 0, iw, i3, &info);
        chkxer("DSBEVX_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        dsbevx_2stage("N", "A", "U", 2, 1, a, 1, q, 1, 0.0,
                      0.0, -1, -1, 0.0, &m, x, z, 1, w, 0, iw, i3, &info);
        chkxer("DSBEVX_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        /* xerbla_infot = 9;
           dsbevx_2stage("V", "A", "U", 2, 0, a, 1, q, 1, 0.0,
                         0.0, -1, -1, 0.0, &m, x, z, 2, w, 0, iw, i3, &info);
           chkxer("DSBEVX_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok); */
        xerbla_infot = 11;
        dsbevx_2stage("N", "V", "U", 1, 0, a, 1, q, 1, 0.0,
                      0.0, -1, -1, 0.0, &m, x, z, 1, w, 0, iw, i3, &info);
        chkxer("DSBEVX_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 12;
        dsbevx_2stage("N", "I", "U", 1, 0, a, 1, q, 1, 0.0,
                      0.0, -1, -1, 0.0, &m, x, z, 1, w, 0, iw, i3, &info);
        chkxer("DSBEVX_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 12;
        dsbevx_2stage("N", "I", "U", 1, 0, a, 1, q, 1, 0.0,
                      0.0, 1, 0, 0.0, &m, x, z, 1, w, 0, iw, i3, &info);
        chkxer("DSBEVX_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 13;
        dsbevx_2stage("N", "I", "U", 2, 0, a, 1, q, 1, 0.0,
                      0.0, 1, 0, 0.0, &m, x, z, 1, w, 0, iw, i3, &info);
        chkxer("DSBEVX_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 13;
        dsbevx_2stage("N", "I", "U", 1, 0, a, 1, q, 1, 0.0,
                      0.0, 0, 1, 0.0, &m, x, z, 1, w, 0, iw, i3, &info);
        chkxer("DSBEVX_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        /* xerbla_infot = 18;
           dsbevx_2stage("V", "A", "U", 2, 0, a, 1, q, 2, 0.0,
                         0.0, -1, -1, 0.0, &m, x, z, 1, w, 0, iw, i3, &info);
           chkxer("DSBEVX_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok); */
        xerbla_infot = 20;
        dsbevx_2stage("N", "A", "U", 0, 0, a, 1, q, 1, 0.0,
                      0.0, -1, -1, 0.0, &m, x, z, 1, w, 0, iw, i3, &info);
        chkxer("DSBEVX_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        /* *nt = *nt + 15; */
        *nt = *nt + 13;
    }

    *ok = xerbla_ok;

    xerbla_srnamt[0] = '\0';
    xerbla_infot = 0;
}
