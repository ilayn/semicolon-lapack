/**
 * @file zerrhs.c
 * @brief ZERRHS tests the error exits for the nonsymmetric eigenvalue routines.
 *
 * Port of LAPACK TESTING/EIG/zerrhs.f
 * Tests error exits for: ZGEBAK, ZGEBAL, ZGEHRD, ZGEHD2, ZUNGHR, ZUNMHR,
 * ZHSEQR, ZHSEIN, ZTREVC, and ZTREVC3
 */

#include <complex.h>
#include <math.h>
#include <string.h>
#include "semicolon_lapack_complex_double.h"
#include "verify.h"

void zerrhs(const char* path, INT* ok, INT* nt)
{
    enum { NMAX = 3, LW = NMAX * NMAX };
    const f64 ONE = 1.0;

    char c2[2];
    INT i, ihi, ilo, info, j, m;

    INT sel[NMAX];
    INT ifaill[NMAX], ifailr[NMAX];
    f64 rw[NMAX], s[NMAX];
    c128 a[NMAX * NMAX], c[NMAX * NMAX], tau[NMAX],
         vl[NMAX * NMAX], vr[NMAX * NMAX], w[LW], x[NMAX];

    c2[0] = path[1];
    c2[1] = path[2];

    /* Set the variables to innocuous values. */
    for (j = 0; j < NMAX; j++) {
        for (i = 0; i < NMAX; i++) {
            a[i + j * NMAX] = ONE / (f64)(i + j + 2);
        }
        sel[j] = 1;
    }
    xerbla_ok = 1;
    *nt = 0;

    /* Test error exits of the nonsymmetric eigenvalue routines. */

    if ((c2[0] == 'H' || c2[0] == 'h') && (c2[1] == 'S' || c2[1] == 's')) {

        /* ZGEBAL */

        strcpy(xerbla_srnamt, "ZGEBAL");
        xerbla_infot = 1;
        zgebal("/", 0, a, 1, &ilo, &ihi, s, &info);
        chkxer("ZGEBAL", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        zgebal("N", -1, a, 1, &ilo, &ihi, s, &info);
        chkxer("ZGEBAL", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        zgebal("N", 2, a, 1, &ilo, &ihi, s, &info);
        chkxer("ZGEBAL", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 3;

        /* ZGEBAK */

        strcpy(xerbla_srnamt, "ZGEBAK");
        xerbla_infot = 1;
        zgebak("/", "R", 0, 0, -1, s, 0, a, 1, &info);
        chkxer("ZGEBAK", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        zgebak("N", "/", 0, 0, -1, s, 0, a, 1, &info);
        chkxer("ZGEBAK", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        zgebak("N", "R", -1, 0, -1, s, 0, a, 1, &info);
        chkxer("ZGEBAK", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        zgebak("N", "R", 0, -1, -1, s, 0, a, 1, &info);
        chkxer("ZGEBAK", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        zgebak("N", "R", 0, 1, -1, s, 0, a, 1, &info);
        chkxer("ZGEBAK", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        zgebak("N", "R", 2, 1, 0, s, 0, a, 2, &info);
        chkxer("ZGEBAK", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        zgebak("N", "R", 0, 0, 0, s, 0, a, 1, &info);
        chkxer("ZGEBAK", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        zgebak("N", "R", 0, 0, -1, s, -1, a, 1, &info);
        chkxer("ZGEBAK", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        zgebak("N", "R", 2, 0, 1, s, 0, a, 1, &info);
        chkxer("ZGEBAK", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 9;

        /* ZGEHRD */

        strcpy(xerbla_srnamt, "ZGEHRD");
        xerbla_infot = 1;
        zgehrd(-1, 0, 0, a, 1, tau, w, 1, &info);
        chkxer("ZGEHRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        zgehrd(0, -1, -1, a, 1, tau, w, 1, &info);
        chkxer("ZGEHRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        zgehrd(0, 1, -1, a, 1, tau, w, 1, &info);
        chkxer("ZGEHRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        zgehrd(1, 0, -1, a, 1, tau, w, 1, &info);
        chkxer("ZGEHRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        zgehrd(0, 0, 0, a, 1, tau, w, 1, &info);
        chkxer("ZGEHRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        zgehrd(2, 0, 0, a, 1, tau, w, 2, &info);
        chkxer("ZGEHRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        zgehrd(2, 0, 1, a, 2, tau, w, 1, &info);
        chkxer("ZGEHRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 7;

        /* ZGEHD2 */

        strcpy(xerbla_srnamt, "ZGEHD2");
        xerbla_infot = 1;
        zgehd2(-1, 0, 0, a, 1, tau, w, &info);
        chkxer("ZGEHD2", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        zgehd2(0, -1, -1, a, 1, tau, w, &info);
        chkxer("ZGEHD2", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        zgehd2(0, 1, -1, a, 1, tau, w, &info);
        chkxer("ZGEHD2", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        zgehd2(1, 0, -1, a, 1, tau, w, &info);
        chkxer("ZGEHD2", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        zgehd2(0, 0, 0, a, 1, tau, w, &info);
        chkxer("ZGEHD2", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        zgehd2(2, 0, 0, a, 1, tau, w, &info);
        chkxer("ZGEHD2", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 6;

        /* ZUNGHR */

        strcpy(xerbla_srnamt, "ZUNGHR");
        xerbla_infot = 1;
        zunghr(-1, 0, 0, a, 1, tau, w, 1, &info);
        chkxer("ZUNGHR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        zunghr(0, -1, -1, a, 1, tau, w, 1, &info);
        chkxer("ZUNGHR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        zunghr(0, 1, -1, a, 1, tau, w, 1, &info);
        chkxer("ZUNGHR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        zunghr(1, 0, -1, a, 1, tau, w, 1, &info);
        chkxer("ZUNGHR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        zunghr(0, 0, 0, a, 1, tau, w, 1, &info);
        chkxer("ZUNGHR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        zunghr(2, 0, 0, a, 1, tau, w, 1, &info);
        chkxer("ZUNGHR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        zunghr(3, 0, 2, a, 3, tau, w, 1, &info);
        chkxer("ZUNGHR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 7;

        /* ZUNMHR */

        strcpy(xerbla_srnamt, "ZUNMHR");
        xerbla_infot = 1;
        zunmhr("/", "N", 0, 0, 0, -1, a, 1, tau, c, 1, w, 1, &info);
        chkxer("ZUNMHR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        zunmhr("L", "/", 0, 0, 0, -1, a, 1, tau, c, 1, w, 1, &info);
        chkxer("ZUNMHR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        zunmhr("L", "N", -1, 0, 0, -1, a, 1, tau, c, 1, w, 1, &info);
        chkxer("ZUNMHR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        zunmhr("L", "N", 0, -1, 0, -1, a, 1, tau, c, 1, w, 1, &info);
        chkxer("ZUNMHR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        zunmhr("L", "N", 0, 0, -1, -1, a, 1, tau, c, 1, w, 1, &info);
        chkxer("ZUNMHR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        zunmhr("L", "N", 0, 0, 1, -1, a, 1, tau, c, 1, w, 1, &info);
        chkxer("ZUNMHR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        zunmhr("L", "N", 1, 2, 1, 0, a, 1, tau, c, 1, w, 2, &info);
        chkxer("ZUNMHR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        zunmhr("R", "N", 2, 1, 1, 0, a, 1, tau, c, 2, w, 2, &info);
        chkxer("ZUNMHR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        zunmhr("L", "N", 1, 1, 0, -1, a, 1, tau, c, 1, w, 1, &info);
        chkxer("ZUNMHR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        zunmhr("L", "N", 0, 1, 0, 0, a, 1, tau, c, 1, w, 1, &info);
        chkxer("ZUNMHR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        zunmhr("R", "N", 1, 0, 0, 0, a, 1, tau, c, 1, w, 1, &info);
        chkxer("ZUNMHR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        zunmhr("L", "N", 2, 1, 0, 0, a, 1, tau, c, 2, w, 1, &info);
        chkxer("ZUNMHR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        zunmhr("R", "N", 1, 2, 0, 0, a, 1, tau, c, 1, w, 1, &info);
        chkxer("ZUNMHR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 11;
        zunmhr("L", "N", 2, 1, 0, 0, a, 2, tau, c, 1, w, 1, &info);
        chkxer("ZUNMHR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 13;
        zunmhr("L", "N", 1, 2, 0, 0, a, 1, tau, c, 1, w, 1, &info);
        chkxer("ZUNMHR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 13;
        zunmhr("R", "N", 2, 1, 0, 0, a, 1, tau, c, 2, w, 1, &info);
        chkxer("ZUNMHR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 16;

        /* ZHSEQR */

        strcpy(xerbla_srnamt, "ZHSEQR");
        xerbla_infot = 1;
        zhseqr("/", "N", 0, 0, -1, a, 1, x, c, 1, w, 1, &info);
        chkxer("ZHSEQR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        zhseqr("E", "/", 0, 0, -1, a, 1, x, c, 1, w, 1, &info);
        chkxer("ZHSEQR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        zhseqr("E", "N", -1, 0, -1, a, 1, x, c, 1, w, 1, &info);
        chkxer("ZHSEQR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        zhseqr("E", "N", 0, -1, -1, a, 1, x, c, 1, w, 1, &info);
        chkxer("ZHSEQR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        zhseqr("E", "N", 0, 1, -1, a, 1, x, c, 1, w, 1, &info);
        chkxer("ZHSEQR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        zhseqr("E", "N", 1, 0, -1, a, 1, x, c, 1, w, 1, &info);
        chkxer("ZHSEQR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        zhseqr("E", "N", 1, 0, 1, a, 1, x, c, 1, w, 1, &info);
        chkxer("ZHSEQR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        zhseqr("E", "N", 2, 0, 1, a, 1, x, c, 2, w, 1, &info);
        chkxer("ZHSEQR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        zhseqr("E", "V", 2, 0, 1, a, 2, x, c, 1, w, 1, &info);
        chkxer("ZHSEQR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 9;

        /* ZHSEIN */

        strcpy(xerbla_srnamt, "ZHSEIN");
        xerbla_infot = 1;
        zhsein("/", "N", "N", sel, 0, a, 1, x, vl, 1, vr, 1, 0,
               &m, w, rw, ifaill, ifailr, &info);
        chkxer("ZHSEIN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        zhsein("R", "/", "N", sel, 0, a, 1, x, vl, 1, vr, 1, 0,
               &m, w, rw, ifaill, ifailr, &info);
        chkxer("ZHSEIN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        zhsein("R", "N", "/", sel, 0, a, 1, x, vl, 1, vr, 1, 0,
               &m, w, rw, ifaill, ifailr, &info);
        chkxer("ZHSEIN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        zhsein("R", "N", "N", sel, -1, a, 1, x, vl, 1, vr, 1, 0,
               &m, w, rw, ifaill, ifailr, &info);
        chkxer("ZHSEIN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        zhsein("R", "N", "N", sel, 2, a, 1, x, vl, 1, vr, 2, 4,
               &m, w, rw, ifaill, ifailr, &info);
        chkxer("ZHSEIN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        zhsein("L", "N", "N", sel, 2, a, 2, x, vl, 1, vr, 1, 4,
               &m, w, rw, ifaill, ifailr, &info);
        chkxer("ZHSEIN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 12;
        zhsein("R", "N", "N", sel, 2, a, 2, x, vl, 1, vr, 1, 4,
               &m, w, rw, ifaill, ifailr, &info);
        chkxer("ZHSEIN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 13;
        zhsein("R", "N", "N", sel, 2, a, 2, x, vl, 1, vr, 2, 1,
               &m, w, rw, ifaill, ifailr, &info);
        chkxer("ZHSEIN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 8;

        /* ZTREVC */

        strcpy(xerbla_srnamt, "ZTREVC");
        xerbla_infot = 1;
        ztrevc("/", "A", sel, 0, a, 1, vl, 1, vr, 1, 0, &m, w, rw,
               &info);
        chkxer("ZTREVC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        ztrevc("L", "/", sel, 0, a, 1, vl, 1, vr, 1, 0, &m, w, rw,
               &info);
        chkxer("ZTREVC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        ztrevc("L", "A", sel, -1, a, 1, vl, 1, vr, 1, 0, &m, w,
               rw, &info);
        chkxer("ZTREVC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        ztrevc("L", "A", sel, 2, a, 1, vl, 2, vr, 1, 4, &m, w, rw,
               &info);
        chkxer("ZTREVC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        ztrevc("L", "A", sel, 2, a, 2, vl, 1, vr, 1, 4, &m, w, rw,
               &info);
        chkxer("ZTREVC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        ztrevc("R", "A", sel, 2, a, 2, vl, 1, vr, 1, 4, &m, w, rw,
               &info);
        chkxer("ZTREVC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 11;
        ztrevc("L", "A", sel, 2, a, 2, vl, 2, vr, 1, 1, &m, w, rw,
               &info);
        chkxer("ZTREVC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 7;

        /* ZTREVC3 */

        strcpy(xerbla_srnamt, "ZTREVC3");
        xerbla_infot = 1;
        ztrevc3("/", "A", sel, 0, a, 1, vl, 1, vr, 1, 0, &m, w,
                LW, rw, 1, &info);
        chkxer("ZTREVC3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        ztrevc3("L", "/", sel, 0, a, 1, vl, 1, vr, 1, 0, &m, w,
                LW, rw, 1, &info);
        chkxer("ZTREVC3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        ztrevc3("L", "A", sel, -1, a, 1, vl, 1, vr, 1, 0, &m, w,
                LW, rw, 1, &info);
        chkxer("ZTREVC3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        ztrevc3("L", "A", sel, 2, a, 1, vl, 2, vr, 1, 4, &m, w,
                LW, rw, 2, &info);
        chkxer("ZTREVC3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        ztrevc3("L", "A", sel, 2, a, 2, vl, 1, vr, 1, 4, &m, w,
                LW, rw, 2, &info);
        chkxer("ZTREVC3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        ztrevc3("R", "A", sel, 2, a, 2, vl, 1, vr, 1, 4, &m, w,
                LW, rw, 2, &info);
        chkxer("ZTREVC3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 11;
        ztrevc3("L", "A", sel, 2, a, 2, vl, 2, vr, 1, 1, &m, w,
                LW, rw, 2, &info);
        chkxer("ZTREVC3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 14;
        ztrevc3("L", "A", sel, 2, a, 2, vl, 2, vr, 1, 2, &m, w,
                2, rw, 2, &info);
        chkxer("ZTREVC3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 16;
        ztrevc3("L", "A", sel, 2, a, 2, vl, 2, vr, 1, 2, &m, w,
                LW, rw, 1, &info);
        chkxer("ZTREVC3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 9;
    }

    *ok = xerbla_ok;

    xerbla_srnamt[0] = '\0';
    xerbla_infot = 0;
}
