/**
 * @file cerrst.c
 * @brief CERRST tests the error exits for the Hermitian eigenvalue routines.
 *
 * Port of LAPACK TESTING/EIG/cerrst.f
 * Tests error exits for: CHETRD, CHETD2, CUNGTR, CUNMTR, CHPTRD,
 * CUPGTR, CUPMTR, CSTEQR, CSTEIN, CPTEQR, CHBTRD,
 * CHEEV, CHEEVX, CHEEVD, CHBEV, CHBEVX, CHBEVD,
 * CHPEV, CHPEVX, CHPEVD, and CSTEDC.
 * CHEEVD_2STAGE, CHEEVR_2STAGE, CHEEVX_2STAGE,
 * CHEEV_2STAGE, CHBEV_2STAGE, CHBEVD_2STAGE,
 * CHBEVX_2STAGE, CHETRD_2STAGE, CHETRD_HE2HB,
 * CHETRD_HB2ST
 */

#include <complex.h>
#include <math.h>
#include <string.h>
#include "semicolon_lapack_complex_single.h"
#include "verify.h"

void cerrst(const char* path, INT* ok, INT* nt)
{
    enum { NMAX = 3, LIW = 12 * NMAX, LW = 20 * NMAX };

    char c2[2];
    INT i, info, j, m, n;

    INT i1[NMAX], i2[NMAX], i3[NMAX], iw[LIW];
    f32 d[NMAX], e[NMAX], r[LW], rw[LW], x[NMAX];
    c64 a[NMAX * NMAX], c[NMAX * NMAX], q[NMAX * NMAX], tau[NMAX],
         w[LW], z[NMAX * NMAX];

    c2[0] = path[1];
    c2[1] = path[2];

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

    if ((c2[0] == 'S' || c2[0] == 's') && (c2[1] == 'T' || c2[1] == 't')) {

        /* CHETRD */

        strcpy(xerbla_srnamt, "CHETRD");
        xerbla_infot = 1;
        chetrd("/", 0, a, 1, d, e, tau, w, 1, &info);
        chkxer("CHETRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        chetrd("U", -1, a, 1, d, e, tau, w, 1, &info);
        chkxer("CHETRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        chetrd("U", 2, a, 1, d, e, tau, w, 1, &info);
        chkxer("CHETRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        chetrd("U", 0, a, 1, d, e, tau, w, 0, &info);
        chkxer("CHETRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 4;

        /* CHETD2 */

        strcpy(xerbla_srnamt, "CHETD2");
        xerbla_infot = 1;
        chetd2("/", 0, a, 1, d, e, tau, &info);
        chkxer("CHETD2", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        chetd2("U", -1, a, 1, d, e, tau, &info);
        chkxer("CHETD2", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        chetd2("U", 2, a, 1, d, e, tau, &info);
        chkxer("CHETD2", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 3;

        /* CHETRD_2STAGE */

        strcpy(xerbla_srnamt, "CHETRD_2STAGE");
        xerbla_infot = 1;
        chetrd_2stage("/", "U", 0, a, 1, d, e, tau,
                      c, 1, w, 1, &info);
        chkxer("CHETRD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 1;
        chetrd_2stage("H", "U", 0, a, 1, d, e, tau,
                      c, 1, w, 1, &info);
        chkxer("CHETRD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        chetrd_2stage("N", "/", 0, a, 1, d, e, tau,
                      c, 1, w, 1, &info);
        chkxer("CHETRD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        chetrd_2stage("N", "U", -1, a, 1, d, e, tau,
                      c, 1, w, 1, &info);
        chkxer("CHETRD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        chetrd_2stage("N", "U", 2, a, 1, d, e, tau,
                      c, 1, w, 1, &info);
        chkxer("CHETRD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        chetrd_2stage("N", "U", 0, a, 1, d, e, tau,
                      c, 0, w, 1, &info);
        chkxer("CHETRD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 12;
        chetrd_2stage("N", "U", 0, a, 1, d, e, tau,
                      c, 1, w, 0, &info);
        chkxer("CHETRD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 7;

        /* CHETRD_HE2HB */

        strcpy(xerbla_srnamt, "CHETRD_HE2HB");
        xerbla_infot = 1;
        chetrd_he2hb("/", 0, 0, a, 1, c, 1, tau, w, 1, &info);
        chkxer("CHETRD_HE2HB", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        chetrd_he2hb("U", -1, 0, a, 1, c, 1, tau, w, 1, &info);
        chkxer("CHETRD_HE2HB", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        chetrd_he2hb("U", 0, -1, a, 1, c, 1, tau, w, 1, &info);
        chkxer("CHETRD_HE2HB", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        chetrd_he2hb("U", 2, 0, a, 1, c, 1, tau, w, 1, &info);
        chkxer("CHETRD_HE2HB", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        chetrd_he2hb("U", 0, 2, a, 1, c, 1, tau, w, 1, &info);
        chkxer("CHETRD_HE2HB", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        chetrd_he2hb("U", 0, 0, a, 1, c, 1, tau, w, 0, &info);
        chkxer("CHETRD_HE2HB", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 6;

        /* CHETRD_HB2ST */

        strcpy(xerbla_srnamt, "CHETRD_HB2ST");
        xerbla_infot = 1;
        chetrd_hb2st("/", "N", "U", 0, 0, a, 1, d, e,
                     c, 1, w, 1, &info);
        chkxer("CHETRD_HB2ST", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        chetrd_hb2st("Y", "/", "U", 0, 0, a, 1, d, e,
                     c, 1, w, 1, &info);
        chkxer("CHETRD_HB2ST", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        chetrd_hb2st("Y", "H", "U", 0, 0, a, 1, d, e,
                     c, 1, w, 1, &info);
        chkxer("CHETRD_HB2ST", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        chetrd_hb2st("Y", "N", "/", 0, 0, a, 1, d, e,
                     c, 1, w, 1, &info);
        chkxer("CHETRD_HB2ST", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        chetrd_hb2st("Y", "N", "U", -1, 0, a, 1, d, e,
                     c, 1, w, 1, &info);
        chkxer("CHETRD_HB2ST", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        chetrd_hb2st("Y", "N", "U", 0, -1, a, 1, d, e,
                     c, 1, w, 1, &info);
        chkxer("CHETRD_HB2ST", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        chetrd_hb2st("Y", "N", "U", 0, 1, a, 1, d, e,
                     c, 1, w, 1, &info);
        chkxer("CHETRD_HB2ST", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 11;
        chetrd_hb2st("Y", "N", "U", 0, 0, a, 1, d, e,
                     c, 0, w, 1, &info);
        chkxer("CHETRD_HB2ST", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 13;
        chetrd_hb2st("Y", "N", "U", 0, 0, a, 1, d, e,
                     c, 1, w, 0, &info);
        chkxer("CHETRD_HB2ST", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 9;

        /* CUNGTR */

        strcpy(xerbla_srnamt, "CUNGTR");
        xerbla_infot = 1;
        cungtr("/", 0, a, 1, tau, w, 1, &info);
        chkxer("CUNGTR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        cungtr("U", -1, a, 1, tau, w, 1, &info);
        chkxer("CUNGTR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        cungtr("U", 2, a, 1, tau, w, 1, &info);
        chkxer("CUNGTR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        cungtr("U", 3, a, 3, tau, w, 1, &info);
        chkxer("CUNGTR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 4;

        /* CUNMTR */

        strcpy(xerbla_srnamt, "CUNMTR");
        xerbla_infot = 1;
        cunmtr("/", "U", "N", 0, 0, a, 1, tau, c, 1, w, 1, &info);
        chkxer("CUNMTR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        cunmtr("L", "/", "N", 0, 0, a, 1, tau, c, 1, w, 1, &info);
        chkxer("CUNMTR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        cunmtr("L", "U", "/", 0, 0, a, 1, tau, c, 1, w, 1, &info);
        chkxer("CUNMTR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        cunmtr("L", "U", "N", -1, 0, a, 1, tau, c, 1, w, 1,
               &info);
        chkxer("CUNMTR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        cunmtr("L", "U", "N", 0, -1, a, 1, tau, c, 1, w, 1,
               &info);
        chkxer("CUNMTR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        cunmtr("L", "U", "N", 2, 0, a, 1, tau, c, 2, w, 1, &info);
        chkxer("CUNMTR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        cunmtr("R", "U", "N", 0, 2, a, 1, tau, c, 1, w, 1, &info);
        chkxer("CUNMTR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        cunmtr("L", "U", "N", 2, 0, a, 2, tau, c, 1, w, 1, &info);
        chkxer("CUNMTR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 12;
        cunmtr("L", "U", "N", 0, 2, a, 1, tau, c, 1, w, 1, &info);
        chkxer("CUNMTR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 12;
        cunmtr("R", "U", "N", 2, 0, a, 1, tau, c, 2, w, 1, &info);
        chkxer("CUNMTR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 10;

        /* CHPTRD */

        strcpy(xerbla_srnamt, "CHPTRD");
        xerbla_infot = 1;
        chptrd("/", 0, a, d, e, tau, &info);
        chkxer("CHPTRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        chptrd("U", -1, a, d, e, tau, &info);
        chkxer("CHPTRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 2;

        /* CUPGTR */

        strcpy(xerbla_srnamt, "CUPGTR");
        xerbla_infot = 1;
        cupgtr("/", 0, a, tau, z, 1, w, &info);
        chkxer("CUPGTR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        cupgtr("U", -1, a, tau, z, 1, w, &info);
        chkxer("CUPGTR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        cupgtr("U", 2, a, tau, z, 1, w, &info);
        chkxer("CUPGTR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 3;

        /* CUPMTR */

        strcpy(xerbla_srnamt, "CUPMTR");
        xerbla_infot = 1;
        cupmtr("/", "U", "N", 0, 0, a, tau, c, 1, w, &info);
        chkxer("CUPMTR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        cupmtr("L", "/", "N", 0, 0, a, tau, c, 1, w, &info);
        chkxer("CUPMTR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        cupmtr("L", "U", "/", 0, 0, a, tau, c, 1, w, &info);
        chkxer("CUPMTR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        cupmtr("L", "U", "N", -1, 0, a, tau, c, 1, w, &info);
        chkxer("CUPMTR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        cupmtr("L", "U", "N", 0, -1, a, tau, c, 1, w, &info);
        chkxer("CUPMTR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        cupmtr("L", "U", "N", 2, 0, a, tau, c, 1, w, &info);
        chkxer("CUPMTR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 6;

        /* CPTEQR */

        strcpy(xerbla_srnamt, "CPTEQR");
        xerbla_infot = 1;
        cpteqr("/", 0, d, e, z, 1, rw, &info);
        chkxer("CPTEQR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        cpteqr("N", -1, d, e, z, 1, rw, &info);
        chkxer("CPTEQR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        cpteqr("V", 2, d, e, z, 1, rw, &info);
        chkxer("CPTEQR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 3;

        /* CSTEIN */

        strcpy(xerbla_srnamt, "CSTEIN");
        xerbla_infot = 1;
        cstein(-1, d, e, 0, x, i1, i2, z, 1, rw, iw, i3, &info);
        chkxer("CSTEIN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        cstein(0, d, e, -1, x, i1, i2, z, 1, rw, iw, i3, &info);
        chkxer("CSTEIN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        cstein(0, d, e, 1, x, i1, i2, z, 1, rw, iw, i3, &info);
        chkxer("CSTEIN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        cstein(2, d, e, 0, x, i1, i2, z, 1, rw, iw, i3, &info);
        chkxer("CSTEIN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 4;

        /* CSTEQR */

        strcpy(xerbla_srnamt, "CSTEQR");
        xerbla_infot = 1;
        csteqr("/", 0, d, e, z, 1, rw, &info);
        chkxer("CSTEQR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        csteqr("N", -1, d, e, z, 1, rw, &info);
        chkxer("CSTEQR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        csteqr("V", 2, d, e, z, 1, rw, &info);
        chkxer("CSTEQR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 3;

        /* CSTEDC */

        strcpy(xerbla_srnamt, "CSTEDC");
        xerbla_infot = 1;
        cstedc("/", 0, d, e, z, 1, w, 1, rw, 1, iw, 1, &info);
        chkxer("CSTEDC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        cstedc("N", -1, d, e, z, 1, w, 1, rw, 1, iw, 1, &info);
        chkxer("CSTEDC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        cstedc("V", 2, d, e, z, 1, w, 4, rw, 23, iw, 28, &info);
        chkxer("CSTEDC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        cstedc("N", 2, d, e, z, 1, w, 0, rw, 1, iw, 1, &info);
        chkxer("CSTEDC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        cstedc("V", 2, d, e, z, 2, w, 0, rw, 23, iw, 28, &info);
        chkxer("CSTEDC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        cstedc("N", 2, d, e, z, 1, w, 1, rw, 0, iw, 1, &info);
        chkxer("CSTEDC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        cstedc("I", 2, d, e, z, 2, w, 1, rw, 1, iw, 12, &info);
        chkxer("CSTEDC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        cstedc("V", 2, d, e, z, 2, w, 4, rw, 1, iw, 28, &info);
        chkxer("CSTEDC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 12;
        cstedc("N", 2, d, e, z, 1, w, 1, rw, 1, iw, 0, &info);
        chkxer("CSTEDC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 12;
        cstedc("I", 2, d, e, z, 2, w, 1, rw, 23, iw, 0, &info);
        chkxer("CSTEDC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 12;
        cstedc("V", 2, d, e, z, 2, w, 4, rw, 23, iw, 0, &info);
        chkxer("CSTEDC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 11;

        /* CHEEVD */

        strcpy(xerbla_srnamt, "CHEEVD");
        xerbla_infot = 1;
        cheevd("/", "U", 0, a, 1, x, w, 1, rw, 1, iw, 1, &info);
        chkxer("CHEEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        cheevd("N", "/", 0, a, 1, x, w, 1, rw, 1, iw, 1, &info);
        chkxer("CHEEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        cheevd("N", "U", -1, a, 1, x, w, 1, rw, 1, iw, 1, &info);
        chkxer("CHEEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        cheevd("N", "U", 2, a, 1, x, w, 3, rw, 2, iw, 1, &info);
        chkxer("CHEEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        cheevd("N", "U", 1, a, 1, x, w, 0, rw, 1, iw, 1, &info);
        chkxer("CHEEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        cheevd("N", "U", 2, a, 2, x, w, 2, rw, 2, iw, 1, &info);
        chkxer("CHEEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        cheevd("V", "U", 2, a, 2, x, w, 3, rw, 25, iw, 12, &info);
        chkxer("CHEEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        cheevd("N", "U", 1, a, 1, x, w, 1, rw, 0, iw, 1, &info);
        chkxer("CHEEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        cheevd("N", "U", 2, a, 2, x, w, 3, rw, 1, iw, 1, &info);
        chkxer("CHEEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        cheevd("V", "U", 2, a, 2, x, w, 8, rw, 18, iw, 12, &info);
        chkxer("CHEEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 12;
        cheevd("N", "U", 1, a, 1, x, w, 1, rw, 1, iw, 0, &info);
        chkxer("CHEEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 12;
        cheevd("V", "U", 2, a, 2, x, w, 8, rw, 25, iw, 11, &info);
        chkxer("CHEEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 12;

        /* CHEEVD_2STAGE */

        strcpy(xerbla_srnamt, "CHEEVD_2STAGE");
        xerbla_infot = 1;
        cheevd_2stage("/", "U", 0, a, 1, x, w, 1,
                      rw, 1, iw, 1, &info);
        chkxer("CHEEVD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 1;
        cheevd_2stage("V", "U", 0, a, 1, x, w, 1,
                      rw, 1, iw, 1, &info);
        chkxer("CHEEVD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        cheevd_2stage("N", "/", 0, a, 1, x, w, 1,
                      rw, 1, iw, 1, &info);
        chkxer("CHEEVD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        cheevd_2stage("N", "U", -1, a, 1, x, w, 1,
                      rw, 1, iw, 1, &info);
        chkxer("CHEEVD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        cheevd_2stage("N", "U", 2, a, 1, x, w, 3,
                      rw, 2, iw, 1, &info);
        chkxer("CHEEVD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        cheevd_2stage("N", "U", 1, a, 1, x, w, 0,
                      rw, 1, iw, 1, &info);
        chkxer("CHEEVD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        cheevd_2stage("N", "U", 2, a, 2, x, w, 2,
                      rw, 2, iw, 1, &info);
        chkxer("CHEEVD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        /* xerbla_infot = 8;
         * cheevd_2stage("V", "U", 2, a, 2, x, w, 3,
         *               rw, 25, iw, 12, &info);
         * chkxer("CHEEVD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok); */
        xerbla_infot = 10;
        cheevd_2stage("N", "U", 1, a, 1, x, w, 1,
                      rw, 0, iw, 1, &info);
        chkxer("CHEEVD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        cheevd_2stage("N", "U", 2, a, 2, x, w, 25,
                      rw, 1, iw, 1, &info);
        chkxer("CHEEVD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        /* xerbla_infot = 10;
         * cheevd_2stage("V", "U", 2, a, 2, x, w, 8,
         *               rw, 18, iw, 12, &info);
         * chkxer("CHEEVD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok); */
        xerbla_infot = 12;
        cheevd_2stage("N", "U", 1, a, 1, x, w, 1,
                      rw, 1, iw, 0, &info);
        chkxer("CHEEVD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 12;
        /* cheevd_2stage("V", "U", 2, a, 2, x, w, 8,
         *               rw, 25, iw, 11, &info);
         * chkxer("CHEEVD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok); */
        *nt = *nt + 10;

        /* CHEEV */

        strcpy(xerbla_srnamt, "CHEEV ");
        xerbla_infot = 1;
        cheev("/", "U", 0, a, 1, x, w, 1, rw, &info);
        chkxer("CHEEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        cheev("N", "/", 0, a, 1, x, w, 1, rw, &info);
        chkxer("CHEEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        cheev("N", "U", -1, a, 1, x, w, 1, rw, &info);
        chkxer("CHEEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        cheev("N", "U", 2, a, 1, x, w, 3, rw, &info);
        chkxer("CHEEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        cheev("N", "U", 2, a, 2, x, w, 2, rw, &info);
        chkxer("CHEEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 5;

        /* CHEEV_2STAGE */

        strcpy(xerbla_srnamt, "CHEEV_2STAGE ");
        xerbla_infot = 1;
        cheev_2stage("/", "U", 0, a, 1, x, w, 1, rw, &info);
        chkxer("CHEEV_2STAGE ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 1;
        cheev_2stage("V", "U", 0, a, 1, x, w, 1, rw, &info);
        chkxer("CHEEV_2STAGE ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        cheev_2stage("N", "/", 0, a, 1, x, w, 1, rw, &info);
        chkxer("CHEEV_2STAGE ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        cheev_2stage("N", "U", -1, a, 1, x, w, 1, rw, &info);
        chkxer("CHEEV_2STAGE ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        cheev_2stage("N", "U", 2, a, 1, x, w, 3, rw, &info);
        chkxer("CHEEV_2STAGE ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        cheev_2stage("N", "U", 2, a, 2, x, w, 2, rw, &info);
        chkxer("CHEEV_2STAGE ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 6;

        /* CHEEVX */

        strcpy(xerbla_srnamt, "CHEEVX");
        xerbla_infot = 1;
        cheevx("/", "A", "U", 0, a, 1, 0.0f, 0.0f, -1, -1, 0.0f,
               &m, x, z, 1, w, 1, rw, iw, i3, &info);
        chkxer("CHEEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        cheevx("V", "/", "U", 0, a, 1, 0.0f, 1.0f, 0, -1, 0.0f,
               &m, x, z, 1, w, 1, rw, iw, i3, &info);
        chkxer("CHEEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        cheevx("V", "A", "/", 0, a, 1, 0.0f, 0.0f, -1, -1, 0.0f,
               &m, x, z, 1, w, 1, rw, iw, i3, &info);
        xerbla_infot = 4;
        cheevx("V", "A", "U", -1, a, 1, 0.0f, 0.0f, -1, -1,
               0.0f, &m, x, z, 1, w, 1, rw, iw, i3, &info);
        chkxer("CHEEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        cheevx("V", "A", "U", 2, a, 1, 0.0f, 0.0f, -1, -1, 0.0f,
               &m, x, z, 2, w, 3, rw, iw, i3, &info);
        chkxer("CHEEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        cheevx("V", "V", "U", 1, a, 1, 0.0f, 0.0f, -1, -1, 0.0f,
               &m, x, z, 1, w, 1, rw, iw, i3, &info);
        chkxer("CHEEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        cheevx("V", "I", "U", 1, a, 1, 0.0f, 0.0f, -1, -1, 0.0f,
               &m, x, z, 1, w, 1, rw, iw, i3, &info);
        chkxer("CHEEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        cheevx("V", "I", "U", 2, a, 2, 0.0f, 0.0f, 1, 0, 0.0f,
               &m, x, z, 2, w, 3, rw, iw, i3, &info);
        chkxer("CHEEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 15;
        cheevx("V", "A", "U", 2, a, 2, 0.0f, 0.0f, -1, -1, 0.0f,
               &m, x, z, 1, w, 3, rw, iw, i3, &info);
        chkxer("CHEEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 17;
        cheevx("V", "A", "U", 2, a, 2, 0.0f, 0.0f, -1, -1, 0.0f,
               &m, x, z, 2, w, 2, rw, iw, i1, &info);
        chkxer("CHEEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 10;

        /* CHEEVX_2STAGE */

        strcpy(xerbla_srnamt, "CHEEVX_2STAGE");
        xerbla_infot = 1;
        cheevx_2stage("/", "A", "U", 0, a, 1,
                      0.0f, 0.0f, -1, -1, 0.0f,
                      &m, x, z, 1, w, 1, rw, iw, i3, &info);
        chkxer("CHEEVX_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 1;
        cheevx_2stage("V", "A", "U", 0, a, 1,
                      0.0f, 0.0f, -1, -1, 0.0f,
                      &m, x, z, 1, w, 1, rw, iw, i3, &info);
        chkxer("CHEEVX_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        cheevx_2stage("N", "/", "U", 0, a, 1,
                      0.0f, 1.0f, 0, -1, 0.0f,
                      &m, x, z, 1, w, 1, rw, iw, i3, &info);
        chkxer("CHEEVX_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        cheevx_2stage("N", "A", "/", 0, a, 1,
                      0.0f, 0.0f, -1, -1, 0.0f,
                      &m, x, z, 1, w, 1, rw, iw, i3, &info);
        xerbla_infot = 4;
        cheevx_2stage("N", "A", "U", -1, a, 1,
                      0.0f, 0.0f, -1, -1, 0.0f,
                      &m, x, z, 1, w, 1, rw, iw, i3, &info);
        chkxer("CHEEVX_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        cheevx_2stage("N", "A", "U", 2, a, 1,
                      0.0f, 0.0f, -1, -1, 0.0f,
                      &m, x, z, 2, w, 3, rw, iw, i3, &info);
        chkxer("CHEEVX_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        cheevx_2stage("N", "V", "U", 1, a, 1,
                      0.0f, 0.0f, -1, -1, 0.0f,
                      &m, x, z, 1, w, 1, rw, iw, i3, &info);
        chkxer("CHEEVX_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        cheevx_2stage("N", "I", "U", 1, a, 1,
                      0.0f, 0.0f, -1, -1, 0.0f,
                      &m, x, z, 1, w, 1, rw, iw, i3, &info);
        chkxer("CHEEVX_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        cheevx_2stage("N", "I", "U", 2, a, 2,
                      0.0f, 0.0f, 1, 0, 0.0f,
                      &m, x, z, 2, w, 3, rw, iw, i3, &info);
        chkxer("CHEEVX_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 15;
        cheevx_2stage("N", "A", "U", 2, a, 2,
                      0.0f, 0.0f, -1, -1, 0.0f,
                      &m, x, z, 0, w, 3, rw, iw, i3, &info);
        chkxer("CHEEVX_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 17;
        cheevx_2stage("N", "A", "U", 2, a, 2,
                      0.0f, 0.0f, -1, -1, 0.0f,
                      &m, x, z, 2, w, 0, rw, iw, i1, &info);
        chkxer("CHEEVX_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 11;

        /* CHEEVR */

        strcpy(xerbla_srnamt, "CHEEVR");
        n = 1;
        xerbla_infot = 1;
        cheevr("/", "A", "U", 0, a, 1, 0.0f, 0.0f, 0, 0, 0.0f,
               &m, r, z, 1, iw, q, 2 * n, rw, 24 * n, &iw[2 * n],
               10 * n, &info);
        chkxer("CHEEVR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        cheevr("V", "/", "U", 0, a, 1, 0.0f, 0.0f, 0, 0, 0.0f,
               &m, r, z, 1, iw, q, 2 * n, rw, 24 * n, &iw[2 * n],
               10 * n, &info);
        chkxer("CHEEVR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        cheevr("V", "A", "/", -1, a, 1, 0.0f, 0.0f, 0, 0,
               0.0f, &m, r, z, 1, iw, q, 2 * n, rw, 24 * n,
               &iw[2 * n], 10 * n, &info);
        chkxer("CHEEVR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        cheevr("V", "A", "U", -1, a, 1, 0.0f, 0.0f, 0, 0,
               0.0f, &m, r, z, 1, iw, q, 2 * n, rw, 24 * n,
               &iw[2 * n], 10 * n, &info);
        chkxer("CHEEVR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        cheevr("V", "A", "U", 2, a, 1, 0.0f, 0.0f, 0, 0, 0.0f,
               &m, r, z, 1, iw, q, 2 * n, rw, 24 * n, &iw[2 * n],
               10 * n, &info);
        chkxer("CHEEVR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        cheevr("V", "V", "U", 1, a, 1, 0.0f, 0.0f, 0, 0, 0.0f,
               &m, r, z, 1, iw, q, 2 * n, rw, 24 * n, &iw[2 * n],
               10 * n, &info);
        chkxer("CHEEVR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        cheevr("V", "I", "U", 1, a, 1, 0.0f, 0.0f, -1, 0, 0.0f,
               &m, r, z, 1, iw, q, 2 * n, rw, 24 * n, &iw[2 * n],
               10 * n, &info);
        chkxer("CHEEVR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;

        cheevr("V", "I", "U", 2, a, 2, 0.0f, 0.0f, 1, 0, 0.0f,
               &m, r, z, 1, iw, q, 2 * n, rw, 24 * n, &iw[2 * n],
               10 * n, &info);
        chkxer("CHEEVR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 15;
        cheevr("V", "I", "U", 1, a, 1, 0.0f, 0.0f, 0, 0, 0.0f,
               &m, r, z, 0, iw, q, 2 * n, rw, 24 * n, &iw[2 * n],
               10 * n, &info);
        chkxer("CHEEVR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 18;
        cheevr("V", "I", "U", 1, a, 1, 0.0f, 0.0f, 0, 0, 0.0f,
               &m, r, z, 1, iw, q, 0, rw, 24 * n, &iw[2 * n],
               10 * n, &info);
        chkxer("CHEEVR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 20;
        cheevr("V", "I", "U", 1, a, 1, 0.0f, 0.0f, 0, 0, 0.0f,
               &m, r, z, 1, iw, q, 2 * n, rw, 0, &iw[2 * n - 2],
               10 * n, &info);
        chkxer("CHEEVR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 22;
        cheevr("V", "I", "U", 1, a, 1, 0.0f, 0.0f, 0, 0, 0.0f,
               &m, r, z, 1, iw, q, 2 * n, rw, 24 * n, &iw[2 * n], 0,
               &info);
        chkxer("CHEEVR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 12;

        /* CHEEVR_2STAGE */

        strcpy(xerbla_srnamt, "CHEEVR_2STAGE");
        n = 1;
        xerbla_infot = 1;
        cheevr_2stage("/", "A", "U", 0, a, 1,
                      0.0f, 0.0f, 0, 0, 0.0f,
                      &m, r, z, 1, iw, q, 2 * n, rw, 24 * n, &iw[2 * n],
                      10 * n, &info);
        chkxer("CHEEVR_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 1;
        cheevr_2stage("V", "A", "U", 0, a, 1,
                      0.0f, 0.0f, 0, 0, 0.0f,
                      &m, r, z, 1, iw, q, 2 * n, rw, 24 * n, &iw[2 * n],
                      10 * n, &info);
        chkxer("CHEEVR_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        cheevr_2stage("N", "/", "U", 0, a, 1,
                      0.0f, 0.0f, 0, 0, 0.0f,
                      &m, r, z, 1, iw, q, 2 * n, rw, 24 * n, &iw[2 * n],
                      10 * n, &info);
        chkxer("CHEEVR_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        cheevr_2stage("N", "A", "/", -1, a, 1,
                      0.0f, 0.0f, 0, 0, 0.0f,
                      &m, r, z, 1, iw, q, 2 * n, rw, 24 * n,
                      &iw[2 * n], 10 * n, &info);
        chkxer("CHEEVR_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        cheevr_2stage("N", "A", "U", -1, a, 1,
                      0.0f, 0.0f, 0, 0, 0.0f,
                      &m, r, z, 1, iw, q, 2 * n, rw, 24 * n,
                      &iw[2 * n], 10 * n, &info);
        chkxer("CHEEVR_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        cheevr_2stage("N", "A", "U", 2, a, 1,
                      0.0f, 0.0f, 0, 0, 0.0f,
                      &m, r, z, 1, iw, q, 2 * n, rw, 24 * n, &iw[2 * n],
                      10 * n, &info);
        chkxer("CHEEVR_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        cheevr_2stage("N", "V", "U", 1, a, 1,
                      0.0f, 0.0f, 0, 0, 0.0f,
                      &m, r, z, 1, iw, q, 2 * n, rw, 24 * n, &iw[2 * n],
                      10 * n, &info);
        chkxer("CHEEVR_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        cheevr_2stage("N", "I", "U", 1, a, 1,
                      0.0f, 0.0f, -1, 0, 0.0f,
                      &m, r, z, 1, iw, q, 2 * n, rw, 24 * n, &iw[2 * n],
                      10 * n, &info);
        chkxer("CHEEVR_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        cheevr_2stage("N", "I", "U", 2, a, 2,
                      0.0f, 0.0f, 1, 0, 0.0f,
                      &m, r, z, 1, iw, q, 2 * n, rw, 24 * n, &iw[2 * n],
                      10 * n, &info);
        chkxer("CHEEVR_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 15;
        cheevr_2stage("N", "I", "U", 1, a, 1,
                      0.0f, 0.0f, 0, 0, 0.0f,
                      &m, r, z, 0, iw, q, 2 * n, rw, 24 * n, &iw[2 * n],
                      10 * n, &info);
        chkxer("CHEEVR_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 18;
        cheevr_2stage("N", "I", "U", 1, a, 1,
                      0.0f, 0.0f, 0, 0, 0.0f,
                      &m, r, z, 1, iw, q, 0, rw, 24 * n, &iw[2 * n],
                      10 * n, &info);
        chkxer("CHEEVR_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 20;
        cheevr_2stage("N", "I", "U", 1, a, 1,
                      0.0f, 0.0f, 0, 0, 0.0f,
                      &m, r, z, 1, iw, q, 26 * n, rw, 0, &iw[2 * n - 2],
                      10 * n, &info);
        chkxer("CHEEVR_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 22;
        cheevr_2stage("N", "I", "U", 1, a, 1,
                      0.0f, 0.0f, 0, 0, 0.0f,
                      &m, r, z, 1, iw, q, 26 * n, rw, 24 * n, &iw[2 * n], 0,
                      &info);
        chkxer("CHEEVR_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 13;

        /* CHPEVD */

        strcpy(xerbla_srnamt, "CHPEVD");
        xerbla_infot = 1;
        chpevd("/", "U", 0, a, x, z, 1, w, 1, rw, 1, iw, 1,
               &info);
        chkxer("CHPEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        chpevd("N", "/", 0, a, x, z, 1, w, 1, rw, 1, iw, 1,
               &info);
        chkxer("CHPEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        chpevd("N", "U", -1, a, x, z, 1, w, 1, rw, 1, iw, 1,
               &info);
        chkxer("CHPEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        chpevd("V", "U", 2, a, x, z, 1, w, 4, rw, 25, iw, 12,
               &info);
        chkxer("CHPEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        chpevd("N", "U", 1, a, x, z, 1, w, 0, rw, 1, iw, 1,
               &info);
        chkxer("CHPEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        chpevd("N", "U", 2, a, x, z, 2, w, 1, rw, 2, iw, 1,
               &info);
        chkxer("CHPEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        chpevd("V", "U", 2, a, x, z, 2, w, 2, rw, 25, iw, 12,
               &info);
        chkxer("CHPEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 11;
        chpevd("N", "U", 1, a, x, z, 1, w, 1, rw, 0, iw, 1,
               &info);
        chkxer("CHPEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 11;
        chpevd("N", "U", 2, a, x, z, 2, w, 2, rw, 1, iw, 1,
               &info);
        chkxer("CHPEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 11;
        chpevd("V", "U", 2, a, x, z, 2, w, 4, rw, 18, iw, 12,
               &info);
        chkxer("CHPEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 13;
        chpevd("N", "U", 1, a, x, z, 1, w, 1, rw, 1, iw, 0,
               &info);
        chkxer("CHPEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 13;
        chpevd("N", "U", 2, a, x, z, 2, w, 2, rw, 2, iw, 0,
               &info);
        chkxer("CHPEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 13;
        chpevd("V", "U", 2, a, x, z, 2, w, 4, rw, 25, iw, 2,
               &info);
        chkxer("CHPEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 13;

        /* CHPEV */

        strcpy(xerbla_srnamt, "CHPEV ");
        xerbla_infot = 1;
        chpev("/", "U", 0, a, x, z, 1, w, rw, &info);
        chkxer("CHPEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        chpev("N", "/", 0, a, x, z, 1, w, rw, &info);
        chkxer("CHPEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        chpev("N", "U", -1, a, x, z, 1, w, rw, &info);
        chkxer("CHPEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        chpev("V", "U", 2, a, x, z, 1, w, rw, &info);
        chkxer("CHPEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 4;

        /* CHPEVX */

        strcpy(xerbla_srnamt, "CHPEVX");
        xerbla_infot = 1;
        chpevx("/", "A", "U", 0, a, 0.0f, 0.0f, -1, -1, 0.0f, &m,
               x, z, 1, w, rw, iw, i3, &info);
        chkxer("CHPEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        chpevx("V", "/", "U", 0, a, 0.0f, 1.0f, 0, -1, 0.0f, &m,
               x, z, 1, w, rw, iw, i3, &info);
        chkxer("CHPEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        chpevx("V", "A", "/", 0, a, 0.0f, 0.0f, -1, -1, 0.0f, &m,
               x, z, 1, w, rw, iw, i3, &info);
        chkxer("CHPEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        chpevx("V", "A", "U", -1, a, 0.0f, 0.0f, -1, -1, 0.0f,
               &m, x, z, 1, w, rw, iw, i3, &info);
        chkxer("CHPEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        chpevx("V", "V", "U", 1, a, 0.0f, 0.0f, -1, -1, 0.0f, &m,
               x, z, 1, w, rw, iw, i3, &info);
        chkxer("CHPEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        chpevx("V", "I", "U", 1, a, 0.0f, 0.0f, -1, -1, 0.0f, &m,
               x, z, 1, w, rw, iw, i3, &info);
        chkxer("CHPEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        chpevx("V", "I", "U", 2, a, 0.0f, 0.0f, 1, 0, 0.0f, &m,
               x, z, 2, w, rw, iw, i3, &info);
        chkxer("CHPEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 14;
        chpevx("V", "A", "U", 2, a, 0.0f, 0.0f, -1, -1, 0.0f, &m,
               x, z, 1, w, rw, iw, i3, &info);
        chkxer("CHPEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 8;

    /* Test error exits for the HB path. */

    } else if ((c2[0] == 'H' || c2[0] == 'h') && (c2[1] == 'B' || c2[1] == 'b')) {

        /* CHBTRD */

        strcpy(xerbla_srnamt, "CHBTRD");
        xerbla_infot = 1;
        chbtrd("/", "U", 0, 0, a, 1, d, e, z, 1, w, &info);
        chkxer("CHBTRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        chbtrd("N", "/", 0, 0, a, 1, d, e, z, 1, w, &info);
        chkxer("CHBTRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        chbtrd("N", "U", -1, 0, a, 1, d, e, z, 1, w, &info);
        chkxer("CHBTRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        chbtrd("N", "U", 0, -1, a, 1, d, e, z, 1, w, &info);
        chkxer("CHBTRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        chbtrd("N", "U", 1, 1, a, 1, d, e, z, 1, w, &info);
        chkxer("CHBTRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        chbtrd("V", "U", 2, 0, a, 1, d, e, z, 1, w, &info);
        chkxer("CHBTRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 6;

        /* CHETRD_HB2ST */

        strcpy(xerbla_srnamt, "CHETRD_HB2ST");
        xerbla_infot = 1;
        chetrd_hb2st("/", "N", "U", 0, 0, a, 1, d, e,
                     c, 1, w, 1, &info);
        chkxer("CHETRD_HB2ST", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        chetrd_hb2st("N", "/", "U", 0, 0, a, 1, d, e,
                     c, 1, w, 1, &info);
        chkxer("CHETRD_HB2ST", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        chetrd_hb2st("N", "H", "U", 0, 0, a, 1, d, e,
                     c, 1, w, 1, &info);
        chkxer("CHETRD_HB2ST", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        chetrd_hb2st("N", "N", "/", 0, 0, a, 1, d, e,
                     c, 1, w, 1, &info);
        chkxer("CHETRD_HB2ST", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        chetrd_hb2st("N", "N", "U", -1, 0, a, 1, d, e,
                     c, 1, w, 1, &info);
        chkxer("CHETRD_HB2ST", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        chetrd_hb2st("N", "N", "U", 0, -1, a, 1, d, e,
                     c, 1, w, 1, &info);
        chkxer("CHETRD_HB2ST", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        chetrd_hb2st("N", "N", "U", 0, 1, a, 1, d, e,
                     c, 1, w, 1, &info);
        chkxer("CHETRD_HB2ST", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 11;
        chetrd_hb2st("N", "N", "U", 0, 0, a, 1, d, e,
                     c, 0, w, 1, &info);
        chkxer("CHETRD_HB2ST", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 13;
        chetrd_hb2st("N", "N", "U", 0, 0, a, 1, d, e,
                     c, 1, w, 0, &info);
        chkxer("CHETRD_HB2ST", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 9;

        /* CHBEVD */

        strcpy(xerbla_srnamt, "CHBEVD");
        xerbla_infot = 1;
        chbevd("/", "U", 0, 0, a, 1, x, z, 1, w, 1, rw, 1, iw, 1,
               &info);
        chkxer("CHBEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        chbevd("N", "/", 0, 0, a, 1, x, z, 1, w, 1, rw, 1, iw, 1,
               &info);
        chkxer("CHBEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        chbevd("N", "U", -1, 0, a, 1, x, z, 1, w, 1, rw, 1, iw,
               1, &info);
        chkxer("CHBEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        chbevd("N", "U", 0, -1, a, 1, x, z, 1, w, 1, rw, 1, iw,
               1, &info);
        chkxer("CHBEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        chbevd("N", "U", 2, 1, a, 1, x, z, 1, w, 2, rw, 2, iw, 1,
               &info);
        chkxer("CHBEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        chbevd("V", "U", 2, 1, a, 2, x, z, 1, w, 8, rw, 25, iw,
               12, &info);
        chkxer("CHBEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 11;
        chbevd("N", "U", 1, 0, a, 1, x, z, 1, w, 0, rw, 1, iw, 1,
               &info);
        chkxer("CHBEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 11;
        chbevd("N", "U", 2, 1, a, 2, x, z, 2, w, 1, rw, 2, iw, 1,
               &info);
        chkxer("CHBEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 11;
        chbevd("V", "U", 2, 1, a, 2, x, z, 2, w, 2, rw, 25, iw,
               12, &info);
        chkxer("CHBEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 13;
        chbevd("N", "U", 1, 0, a, 1, x, z, 1, w, 1, rw, 0, iw, 1,
               &info);
        chkxer("CHBEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 13;
        chbevd("N", "U", 2, 1, a, 2, x, z, 2, w, 2, rw, 1, iw, 1,
               &info);
        chkxer("CHBEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 13;
        chbevd("V", "U", 2, 1, a, 2, x, z, 2, w, 8, rw, 2, iw,
               12, &info);
        chkxer("CHBEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 15;
        chbevd("N", "U", 1, 0, a, 1, x, z, 1, w, 1, rw, 1, iw, 0,
               &info);
        chkxer("CHBEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 15;
        chbevd("N", "U", 2, 1, a, 2, x, z, 2, w, 2, rw, 2, iw, 0,
               &info);
        chkxer("CHBEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 15;
        chbevd("V", "U", 2, 1, a, 2, x, z, 2, w, 8, rw, 25, iw,
               2, &info);
        chkxer("CHBEVD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 15;

        /* CHBEVD_2STAGE */

        strcpy(xerbla_srnamt, "CHBEVD_2STAGE");
        xerbla_infot = 1;
        chbevd_2stage("/", "U", 0, 0, a, 1, x, z, 1,
                      w, 1, rw, 1, iw, 1, &info);
        chkxer("CHBEVD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 1;
        chbevd_2stage("V", "U", 0, 0, a, 1, x, z, 1,
                      w, 1, rw, 1, iw, 1, &info);
        chkxer("CHBEVD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        chbevd_2stage("N", "/", 0, 0, a, 1, x, z, 1,
                      w, 1, rw, 1, iw, 1, &info);
        chkxer("CHBEVD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        chbevd_2stage("N", "U", -1, 0, a, 1, x, z, 1,
                      w, 1, rw, 1, iw, 1, &info);
        chkxer("CHBEVD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        chbevd_2stage("N", "U", 0, -1, a, 1, x, z, 1,
                      w, 1, rw, 1, iw, 1, &info);
        chkxer("CHBEVD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        chbevd_2stage("N", "U", 2, 1, a, 1, x, z, 1,
                      w, 2, rw, 2, iw, 1, &info);
        chkxer("CHBEVD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        chbevd_2stage("N", "U", 2, 1, a, 2, x, z, 0,
                      w, 8, rw, 25, iw, 12, &info);
        chkxer("CHBEVD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 11;
        chbevd_2stage("N", "U", 1, 0, a, 1, x, z, 1,
                      w, 0, rw, 1, iw, 1, &info);
        chkxer("CHBEVD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 11;
        chbevd_2stage("N", "U", 2, 1, a, 2, x, z, 2,
                      w, 1, rw, 2, iw, 1, &info);
        chkxer("CHBEVD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        /* xerbla_infot = 11;
         * chbevd_2stage("V", "U", 2, 1, a, 2, x, z, 2,
         *               w, 2, rw, 25, iw, 12, &info);
         * chkxer("CHBEVD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok); */
        xerbla_infot = 13;
        chbevd_2stage("N", "U", 1, 0, a, 1, x, z, 1,
                      w, 1, rw, 0, iw, 1, &info);
        chkxer("CHBEVD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 13;
        chbevd_2stage("N", "U", 2, 1, a, 2, x, z, 2,
                      w, 25, rw, 1, iw, 1, &info);
        chkxer("CHBEVD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        /* xerbla_infot = 13;
         * chbevd_2stage("V", "U", 2, 1, a, 2, x, z, 2,
         *               w, 25, rw, 2, iw, 12, &info);
         * chkxer("CHBEVD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok); */
        xerbla_infot = 15;
        chbevd_2stage("N", "U", 1, 0, a, 1, x, z, 1,
                      w, 1, rw, 1, iw, 0, &info);
        chkxer("CHBEVD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 15;
        chbevd_2stage("N", "U", 2, 1, a, 2, x, z, 2,
                      w, 25, rw, 2, iw, 0, &info);
        chkxer("CHBEVD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        /* xerbla_infot = 15;
         * chbevd_2stage("V", "U", 2, 1, a, 2, x, z, 2,
         *               w, 25, rw, 25, iw, 2, &info);
         * chkxer("CHBEVD_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok); */
        *nt = *nt + 13;

        /* CHBEV */

        strcpy(xerbla_srnamt, "CHBEV ");
        xerbla_infot = 1;
        chbev("/", "U", 0, 0, a, 1, x, z, 1, w, rw, &info);
        chkxer("CHBEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        chbev("N", "/", 0, 0, a, 1, x, z, 1, w, rw, &info);
        chkxer("CHBEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        chbev("N", "U", -1, 0, a, 1, x, z, 1, w, rw, &info);
        chkxer("CHBEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        chbev("N", "U", 0, -1, a, 1, x, z, 1, w, rw, &info);
        chkxer("CHBEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        chbev("N", "U", 2, 1, a, 1, x, z, 1, w, rw, &info);
        chkxer("CHBEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        chbev("V", "U", 2, 0, a, 1, x, z, 1, w, rw, &info);
        chkxer("CHBEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 6;

        /* CHBEV_2STAGE */

        strcpy(xerbla_srnamt, "CHBEV_2STAGE ");
        xerbla_infot = 1;
        chbev_2stage("/", "U", 0, 0, a, 1, x,
                     z, 1, w, 0, rw, &info);
        chkxer("CHBEV_2STAGE ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 1;
        chbev_2stage("V", "U", 0, 0, a, 1, x,
                     z, 1, w, 0, rw, &info);
        chkxer("CHBEV_2STAGE ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        chbev_2stage("N", "/", 0, 0, a, 1, x,
                     z, 1, w, 0, rw, &info);
        chkxer("CHBEV_2STAGE ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        chbev_2stage("N", "U", -1, 0, a, 1, x,
                     z, 1, w, 0, rw, &info);
        chkxer("CHBEV_2STAGE ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        chbev_2stage("N", "U", 0, -1, a, 1, x,
                     z, 1, w, 0, rw, &info);
        chkxer("CHBEV_2STAGE ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        chbev_2stage("N", "U", 2, 1, a, 1, x,
                     z, 1, w, 0, rw, &info);
        chkxer("CHBEV_2STAGE ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        chbev_2stage("N", "U", 2, 0, a, 1, x,
                     z, 0, w, 0, rw, &info);
        chkxer("CHBEV_2STAGE ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 11;
        chbev_2stage("N", "U", 2, 0, a, 1, x,
                     z, 1, w, 0, rw, &info);
        chkxer("CHBEV_2STAGE ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 8;

        /* CHBEVX */

        strcpy(xerbla_srnamt, "CHBEVX");
        xerbla_infot = 1;
        chbevx("/", "A", "U", 0, 0, a, 1, q, 1, 0.0f, 0.0f, -1,
               -1, 0.0f, &m, x, z, 1, w, rw, iw, i3, &info);
        chkxer("CHBEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        chbevx("V", "/", "U", 0, 0, a, 1, q, 1, 0.0f, 1.0f, 0,
               -1, 0.0f, &m, x, z, 1, w, rw, iw, i3, &info);
        chkxer("CHBEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        chbevx("V", "A", "/", 0, 0, a, 1, q, 1, 0.0f, 0.0f, -1,
               -1, 0.0f, &m, x, z, 1, w, rw, iw, i3, &info);
        xerbla_infot = 4;
        chbevx("V", "A", "U", -1, 0, a, 1, q, 1, 0.0f, 0.0f, -1,
               -1, 0.0f, &m, x, z, 1, w, rw, iw, i3, &info);
        chkxer("CHBEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        chbevx("V", "A", "U", 0, -1, a, 1, q, 1, 0.0f, 0.0f, -1,
               -1, 0.0f, &m, x, z, 1, w, rw, iw, i3, &info);
        chkxer("CHBEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        chbevx("V", "A", "U", 2, 1, a, 1, q, 2, 0.0f, 0.0f, -1,
               -1, 0.0f, &m, x, z, 2, w, rw, iw, i3, &info);
        chkxer("CHBEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        chbevx("V", "A", "U", 2, 0, a, 1, q, 1, 0.0f, 0.0f, -1,
               -1, 0.0f, &m, x, z, 2, w, rw, iw, i3, &info);
        chkxer("CHBEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 11;
        chbevx("V", "V", "U", 1, 0, a, 1, q, 1, 0.0f, 0.0f, -1,
               -1, 0.0f, &m, x, z, 1, w, rw, iw, i3, &info);
        chkxer("CHBEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 12;
        chbevx("V", "I", "U", 1, 0, a, 1, q, 1, 0.0f, 0.0f, -1,
               -1, 0.0f, &m, x, z, 1, w, rw, iw, i3, &info);
        chkxer("CHBEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 13;
        chbevx("V", "I", "U", 1, 0, a, 1, q, 1, 0.0f, 0.0f, 0,
               1, 0.0f, &m, x, z, 1, w, rw, iw, i3, &info);
        chkxer("CHBEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 18;
        chbevx("V", "A", "U", 2, 0, a, 1, q, 2, 0.0f, 0.0f, -1,
               -1, 0.0f, &m, x, z, 1, w, rw, iw, i3, &info);
        chkxer("CHBEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 11;

        /* CHBEVX_2STAGE */

        strcpy(xerbla_srnamt, "CHBEVX_2STAGE");
        xerbla_infot = 1;
        chbevx_2stage("/", "A", "U", 0, 0, a, 1, q, 1,
                      0.0f, 0.0f, -1, -1, 0.0f,
                      &m, x, z, 1, w, 0, rw, iw, i3, &info);
        xerbla_infot = 1;
        chbevx_2stage("V", "A", "U", 0, 0, a, 1, q, 1,
                      0.0f, 0.0f, -1, -1, 0.0f,
                      &m, x, z, 1, w, 0, rw, iw, i3, &info);
        chkxer("CHBEVX_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        chbevx_2stage("N", "/", "U", 0, 0, a, 1, q, 1,
                      0.0f, 1.0f, 0, -1, 0.0f,
                      &m, x, z, 1, w, 0, rw, iw, i3, &info);
        chkxer("CHBEVX_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        chbevx_2stage("N", "A", "/", 0, 0, a, 1, q, 1,
                      0.0f, 0.0f, -1, -1, 0.0f,
                      &m, x, z, 1, w, 0, rw, iw, i3, &info);
        xerbla_infot = 4;
        chbevx_2stage("N", "A", "U", -1, 0, a, 1, q, 1,
                      0.0f, 0.0f, -1, -1, 0.0f,
                      &m, x, z, 1, w, 0, rw, iw, i3, &info);
        chkxer("CHBEVX_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        chbevx_2stage("N", "A", "U", 0, -1, a, 1, q, 1,
                      0.0f, 0.0f, -1, -1, 0.0f,
                      &m, x, z, 1, w, 0, rw, iw, i3, &info);
        chkxer("CHBEVX_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        chbevx_2stage("N", "A", "U", 2, 1, a, 1, q, 2,
                      0.0f, 0.0f, -1, -1, 0.0f,
                      &m, x, z, 2, w, 0, rw, iw, i3, &info);
        chkxer("CHBEVX_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        /* xerbla_infot = 9;
         * chbevx_2stage("V", "A", "U", 2, 0, a, 1, q, 1,
         *               0.0, 0.0, -1, -1, 0.0,
         *               &m, x, z, 2, w, 0, rw, iw, i3, &info);
         * chkxer("CHBEVX_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok); */
        xerbla_infot = 11;
        chbevx_2stage("N", "V", "U", 1, 0, a, 1, q, 1,
                      0.0f, 0.0f, -1, -1, 0.0f,
                      &m, x, z, 1, w, 0, rw, iw, i3, &info);
        chkxer("CHBEVX_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 12;
        chbevx_2stage("N", "I", "U", 1, 0, a, 1, q, 1,
                      0.0f, 0.0f, -1, -1, 0.0f,
                      &m, x, z, 1, w, 0, rw, iw, i3, &info);
        chkxer("CHBEVX_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 13;
        chbevx_2stage("N", "I", "U", 1, 0, a, 1, q, 1,
                      0.0f, 0.0f, 0, 1, 0.0f,
                      &m, x, z, 1, w, 0, rw, iw, i3, &info);
        chkxer("CHBEVX_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 18;
        chbevx_2stage("N", "A", "U", 2, 0, a, 1, q, 2,
                      0.0f, 0.0f, -1, -1, 0.0f,
                      &m, x, z, 0, w, 0, rw, iw, i3, &info);
        chkxer("CHBEVX_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 20;
        chbevx_2stage("N", "A", "U", 2, 0, a, 1, q, 2,
                      0.0f, 0.0f, -1, -1, 0.0f,
                      &m, x, z, 1, w, 0, rw, iw, i3, &info);
        chkxer("CHBEVX_2STAGE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 12;
    }

    *ok = xerbla_ok;

    xerbla_srnamt[0] = '\0';
    xerbla_infot = 0;
}
