/**
 * @file cerrhs.c
 * @brief CERRHS tests the error exits for the nonsymmetric eigenvalue routines.
 *
 * Port of LAPACK TESTING/EIG/cerrhs.f
 * Tests error exits for: CGEBAK, CGEBAL, CGEHRD, CGEHD2, CUNGHR, CUNMHR,
 * CHSEQR, CHSEIN, CTREVC, and CTREVC3
 */

#include <complex.h>
#include <math.h>
#include <string.h>
#include "semicolon_lapack_complex_single.h"
#include "verify.h"

void cerrhs(const char* path, INT* ok, INT* nt)
{
    enum { NMAX = 3, LW = NMAX * NMAX };
    const f32 ONE = 1.0f;

    char c2[2];
    INT i, ihi, ilo, info, j, m;

    INT sel[NMAX];
    INT ifaill[NMAX], ifailr[NMAX];
    f32 rw[NMAX], s[NMAX];
    c64 a[NMAX * NMAX], c[NMAX * NMAX], tau[NMAX],
         vl[NMAX * NMAX], vr[NMAX * NMAX], w[LW], x[NMAX];

    c2[0] = path[1];
    c2[1] = path[2];

    /* Set the variables to innocuous values. */
    for (j = 0; j < NMAX; j++) {
        for (i = 0; i < NMAX; i++) {
            a[i + j * NMAX] = ONE / (f32)(i + j + 2);
        }
        sel[j] = 1;
    }
    xerbla_ok = 1;
    *nt = 0;

    /* Test error exits of the nonsymmetric eigenvalue routines. */

    if ((c2[0] == 'H' || c2[0] == 'h') && (c2[1] == 'S' || c2[1] == 's')) {

        /* CGEBAL */

        strcpy(xerbla_srnamt, "CGEBAL");
        xerbla_infot = 1;
        cgebal("/", 0, a, 1, &ilo, &ihi, s, &info);
        chkxer("CGEBAL", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        cgebal("N", -1, a, 1, &ilo, &ihi, s, &info);
        chkxer("CGEBAL", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        cgebal("N", 2, a, 1, &ilo, &ihi, s, &info);
        chkxer("CGEBAL", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 3;

        /* CGEBAK */

        strcpy(xerbla_srnamt, "CGEBAK");
        xerbla_infot = 1;
        cgebak("/", "R", 0, 0, -1, s, 0, a, 1, &info);
        chkxer("CGEBAK", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        cgebak("N", "/", 0, 0, -1, s, 0, a, 1, &info);
        chkxer("CGEBAK", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        cgebak("N", "R", -1, 0, -1, s, 0, a, 1, &info);
        chkxer("CGEBAK", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        cgebak("N", "R", 0, -1, -1, s, 0, a, 1, &info);
        chkxer("CGEBAK", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        cgebak("N", "R", 0, 1, -1, s, 0, a, 1, &info);
        chkxer("CGEBAK", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        cgebak("N", "R", 2, 1, 0, s, 0, a, 2, &info);
        chkxer("CGEBAK", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        cgebak("N", "R", 0, 0, 0, s, 0, a, 1, &info);
        chkxer("CGEBAK", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        cgebak("N", "R", 0, 0, -1, s, -1, a, 1, &info);
        chkxer("CGEBAK", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        cgebak("N", "R", 2, 0, 1, s, 0, a, 1, &info);
        chkxer("CGEBAK", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 9;

        /* CGEHRD */

        strcpy(xerbla_srnamt, "CGEHRD");
        xerbla_infot = 1;
        cgehrd(-1, 0, 0, a, 1, tau, w, 1, &info);
        chkxer("CGEHRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        cgehrd(0, -1, -1, a, 1, tau, w, 1, &info);
        chkxer("CGEHRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        cgehrd(0, 1, -1, a, 1, tau, w, 1, &info);
        chkxer("CGEHRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        cgehrd(1, 0, -1, a, 1, tau, w, 1, &info);
        chkxer("CGEHRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        cgehrd(0, 0, 0, a, 1, tau, w, 1, &info);
        chkxer("CGEHRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        cgehrd(2, 0, 0, a, 1, tau, w, 2, &info);
        chkxer("CGEHRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        cgehrd(2, 0, 1, a, 2, tau, w, 1, &info);
        chkxer("CGEHRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 7;

        /* CGEHD2 */

        strcpy(xerbla_srnamt, "CGEHD2");
        xerbla_infot = 1;
        cgehd2(-1, 0, 0, a, 1, tau, w, &info);
        chkxer("CGEHD2", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        cgehd2(0, -1, -1, a, 1, tau, w, &info);
        chkxer("CGEHD2", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        cgehd2(0, 1, -1, a, 1, tau, w, &info);
        chkxer("CGEHD2", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        cgehd2(1, 0, -1, a, 1, tau, w, &info);
        chkxer("CGEHD2", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        cgehd2(0, 0, 0, a, 1, tau, w, &info);
        chkxer("CGEHD2", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        cgehd2(2, 0, 0, a, 1, tau, w, &info);
        chkxer("CGEHD2", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 6;

        /* CUNGHR */

        strcpy(xerbla_srnamt, "CUNGHR");
        xerbla_infot = 1;
        cunghr(-1, 0, 0, a, 1, tau, w, 1, &info);
        chkxer("CUNGHR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        cunghr(0, -1, -1, a, 1, tau, w, 1, &info);
        chkxer("CUNGHR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        cunghr(0, 1, -1, a, 1, tau, w, 1, &info);
        chkxer("CUNGHR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        cunghr(1, 0, -1, a, 1, tau, w, 1, &info);
        chkxer("CUNGHR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        cunghr(0, 0, 0, a, 1, tau, w, 1, &info);
        chkxer("CUNGHR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        cunghr(2, 0, 0, a, 1, tau, w, 1, &info);
        chkxer("CUNGHR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        cunghr(3, 0, 2, a, 3, tau, w, 1, &info);
        chkxer("CUNGHR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 7;

        /* CUNMHR */

        strcpy(xerbla_srnamt, "CUNMHR");
        xerbla_infot = 1;
        cunmhr("/", "N", 0, 0, 0, -1, a, 1, tau, c, 1, w, 1, &info);
        chkxer("CUNMHR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        cunmhr("L", "/", 0, 0, 0, -1, a, 1, tau, c, 1, w, 1, &info);
        chkxer("CUNMHR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        cunmhr("L", "N", -1, 0, 0, -1, a, 1, tau, c, 1, w, 1, &info);
        chkxer("CUNMHR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        cunmhr("L", "N", 0, -1, 0, -1, a, 1, tau, c, 1, w, 1, &info);
        chkxer("CUNMHR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        cunmhr("L", "N", 0, 0, -1, -1, a, 1, tau, c, 1, w, 1, &info);
        chkxer("CUNMHR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        cunmhr("L", "N", 0, 0, 1, -1, a, 1, tau, c, 1, w, 1, &info);
        chkxer("CUNMHR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        cunmhr("L", "N", 1, 2, 1, 0, a, 1, tau, c, 1, w, 2, &info);
        chkxer("CUNMHR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        cunmhr("R", "N", 2, 1, 1, 0, a, 1, tau, c, 2, w, 2, &info);
        chkxer("CUNMHR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        cunmhr("L", "N", 1, 1, 0, -1, a, 1, tau, c, 1, w, 1, &info);
        chkxer("CUNMHR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        cunmhr("L", "N", 0, 1, 0, 0, a, 1, tau, c, 1, w, 1, &info);
        chkxer("CUNMHR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        cunmhr("R", "N", 1, 0, 0, 0, a, 1, tau, c, 1, w, 1, &info);
        chkxer("CUNMHR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        cunmhr("L", "N", 2, 1, 0, 0, a, 1, tau, c, 2, w, 1, &info);
        chkxer("CUNMHR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        cunmhr("R", "N", 1, 2, 0, 0, a, 1, tau, c, 1, w, 1, &info);
        chkxer("CUNMHR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 11;
        cunmhr("L", "N", 2, 1, 0, 0, a, 2, tau, c, 1, w, 1, &info);
        chkxer("CUNMHR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 13;
        cunmhr("L", "N", 1, 2, 0, 0, a, 1, tau, c, 1, w, 1, &info);
        chkxer("CUNMHR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 13;
        cunmhr("R", "N", 2, 1, 0, 0, a, 1, tau, c, 2, w, 1, &info);
        chkxer("CUNMHR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 16;

        /* CHSEQR */

        strcpy(xerbla_srnamt, "CHSEQR");
        xerbla_infot = 1;
        chseqr("/", "N", 0, 0, -1, a, 1, x, c, 1, w, 1, &info);
        chkxer("CHSEQR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        chseqr("E", "/", 0, 0, -1, a, 1, x, c, 1, w, 1, &info);
        chkxer("CHSEQR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        chseqr("E", "N", -1, 0, -1, a, 1, x, c, 1, w, 1, &info);
        chkxer("CHSEQR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        chseqr("E", "N", 0, -1, -1, a, 1, x, c, 1, w, 1, &info);
        chkxer("CHSEQR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        chseqr("E", "N", 0, 1, -1, a, 1, x, c, 1, w, 1, &info);
        chkxer("CHSEQR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        chseqr("E", "N", 1, 0, -1, a, 1, x, c, 1, w, 1, &info);
        chkxer("CHSEQR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        chseqr("E", "N", 1, 0, 1, a, 1, x, c, 1, w, 1, &info);
        chkxer("CHSEQR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        chseqr("E", "N", 2, 0, 1, a, 1, x, c, 2, w, 1, &info);
        chkxer("CHSEQR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        chseqr("E", "V", 2, 0, 1, a, 2, x, c, 1, w, 1, &info);
        chkxer("CHSEQR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 9;

        /* CHSEIN */

        strcpy(xerbla_srnamt, "CHSEIN");
        xerbla_infot = 1;
        chsein("/", "N", "N", sel, 0, a, 1, x, vl, 1, vr, 1, 0,
               &m, w, rw, ifaill, ifailr, &info);
        chkxer("CHSEIN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        chsein("R", "/", "N", sel, 0, a, 1, x, vl, 1, vr, 1, 0,
               &m, w, rw, ifaill, ifailr, &info);
        chkxer("CHSEIN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        chsein("R", "N", "/", sel, 0, a, 1, x, vl, 1, vr, 1, 0,
               &m, w, rw, ifaill, ifailr, &info);
        chkxer("CHSEIN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        chsein("R", "N", "N", sel, -1, a, 1, x, vl, 1, vr, 1, 0,
               &m, w, rw, ifaill, ifailr, &info);
        chkxer("CHSEIN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        chsein("R", "N", "N", sel, 2, a, 1, x, vl, 1, vr, 2, 4,
               &m, w, rw, ifaill, ifailr, &info);
        chkxer("CHSEIN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        chsein("L", "N", "N", sel, 2, a, 2, x, vl, 1, vr, 1, 4,
               &m, w, rw, ifaill, ifailr, &info);
        chkxer("CHSEIN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 12;
        chsein("R", "N", "N", sel, 2, a, 2, x, vl, 1, vr, 1, 4,
               &m, w, rw, ifaill, ifailr, &info);
        chkxer("CHSEIN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 13;
        chsein("R", "N", "N", sel, 2, a, 2, x, vl, 1, vr, 2, 1,
               &m, w, rw, ifaill, ifailr, &info);
        chkxer("CHSEIN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 8;

        /* CTREVC */

        strcpy(xerbla_srnamt, "CTREVC");
        xerbla_infot = 1;
        ctrevc("/", "A", sel, 0, a, 1, vl, 1, vr, 1, 0, &m, w, rw,
               &info);
        chkxer("CTREVC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        ctrevc("L", "/", sel, 0, a, 1, vl, 1, vr, 1, 0, &m, w, rw,
               &info);
        chkxer("CTREVC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        ctrevc("L", "A", sel, -1, a, 1, vl, 1, vr, 1, 0, &m, w,
               rw, &info);
        chkxer("CTREVC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        ctrevc("L", "A", sel, 2, a, 1, vl, 2, vr, 1, 4, &m, w, rw,
               &info);
        chkxer("CTREVC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        ctrevc("L", "A", sel, 2, a, 2, vl, 1, vr, 1, 4, &m, w, rw,
               &info);
        chkxer("CTREVC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        ctrevc("R", "A", sel, 2, a, 2, vl, 1, vr, 1, 4, &m, w, rw,
               &info);
        chkxer("CTREVC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 11;
        ctrevc("L", "A", sel, 2, a, 2, vl, 2, vr, 1, 1, &m, w, rw,
               &info);
        chkxer("CTREVC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 7;

        /* CTREVC3 */

        strcpy(xerbla_srnamt, "CTREVC3");
        xerbla_infot = 1;
        ctrevc3("/", "A", sel, 0, a, 1, vl, 1, vr, 1, 0, &m, w,
                LW, rw, 1, &info);
        chkxer("CTREVC3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        ctrevc3("L", "/", sel, 0, a, 1, vl, 1, vr, 1, 0, &m, w,
                LW, rw, 1, &info);
        chkxer("CTREVC3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        ctrevc3("L", "A", sel, -1, a, 1, vl, 1, vr, 1, 0, &m, w,
                LW, rw, 1, &info);
        chkxer("CTREVC3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        ctrevc3("L", "A", sel, 2, a, 1, vl, 2, vr, 1, 4, &m, w,
                LW, rw, 2, &info);
        chkxer("CTREVC3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        ctrevc3("L", "A", sel, 2, a, 2, vl, 1, vr, 1, 4, &m, w,
                LW, rw, 2, &info);
        chkxer("CTREVC3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        ctrevc3("R", "A", sel, 2, a, 2, vl, 1, vr, 1, 4, &m, w,
                LW, rw, 2, &info);
        chkxer("CTREVC3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 11;
        ctrevc3("L", "A", sel, 2, a, 2, vl, 2, vr, 1, 1, &m, w,
                LW, rw, 2, &info);
        chkxer("CTREVC3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 14;
        ctrevc3("L", "A", sel, 2, a, 2, vl, 2, vr, 1, 2, &m, w,
                2, rw, 2, &info);
        chkxer("CTREVC3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 16;
        ctrevc3("L", "A", sel, 2, a, 2, vl, 2, vr, 1, 2, &m, w,
                LW, rw, 1, &info);
        chkxer("CTREVC3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 9;
    }

    *ok = xerbla_ok;

    xerbla_srnamt[0] = '\0';
    xerbla_infot = 0;
}
