/**
 * @file derrhs.c
 * @brief DERRHS tests the error exits for the nonsymmetric eigenvalue routines.
 *
 * Port of LAPACK TESTING/EIG/derrhs.f
 * Tests error exits for: DGEBAK, DGEBAL, DGEHRD, DGEHD2, DORGHR, DORMHR,
 * DHSEQR, DHSEIN, DTREVC, DTREVC3
 */

#include <string.h>
#include "semicolon_lapack_double.h"
#include "verify.h"

void derrhs(const char* path, INT* ok, INT* nt)
{
    enum { NMAX = 3, LW = (NMAX + 2) * (NMAX + 2) + NMAX };
    const f64 ONE = 1.0;

    INT i, ihi, ilo, info, j, m;

    INT sel[NMAX];
    INT ifaill[NMAX], ifailr[NMAX];
    f64 a[NMAX * NMAX], c[NMAX * NMAX], s[NMAX], tau[NMAX];
    f64 vl[NMAX * NMAX], vr[NMAX * NMAX], w[LW], wi[NMAX], wr[NMAX];

    /* Set the variables to innocuous values. */
    for (j = 0; j < NMAX; j++) {
        for (i = 0; i < NMAX; i++) {
            a[i + j * NMAX] = ONE / (f64)(i + j + 2);
        }
        wi[j] = (f64)(j + 1);
        sel[j] = 1;
    }
    xerbla_ok = 1;
    *nt = 0;

    /* Test error exits of the nonsymmetric eigenvalue routines. */

    if ((path[1] == 'H' || path[1] == 'h') &&
        (path[2] == 'S' || path[2] == 's')) {

        /* DGEBAL */

        strcpy(xerbla_srnamt, "DGEBAL");
        xerbla_infot = 1;
        dgebal("/", 0, a, 1, &ilo, &ihi, s, &info);
        chkxer("DGEBAL", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        dgebal("N", -1, a, 1, &ilo, &ihi, s, &info);
        chkxer("DGEBAL", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        dgebal("N", 2, a, 1, &ilo, &ihi, s, &info);
        chkxer("DGEBAL", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 3;

        /* DGEBAK */

        strcpy(xerbla_srnamt, "DGEBAK");
        xerbla_infot = 1;
        dgebak("/", "R", 0, 0, -1, s, 0, a, 1, &info);
        chkxer("DGEBAK", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        dgebak("N", "/", 0, 0, -1, s, 0, a, 1, &info);
        chkxer("DGEBAK", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        dgebak("N", "R", -1, 0, -1, s, 0, a, 1, &info);
        chkxer("DGEBAK", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        dgebak("N", "R", 0, -1, -1, s, 0, a, 1, &info);
        chkxer("DGEBAK", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        dgebak("N", "R", 0, 1, -1, s, 0, a, 1, &info);
        chkxer("DGEBAK", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        dgebak("N", "R", 2, 1, 0, s, 0, a, 2, &info);
        chkxer("DGEBAK", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        dgebak("N", "R", 0, 0, 0, s, 0, a, 1, &info);
        chkxer("DGEBAK", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        dgebak("N", "R", 0, 0, -1, s, -1, a, 1, &info);
        chkxer("DGEBAK", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        dgebak("N", "R", 2, 0, 1, s, 0, a, 1, &info);
        chkxer("DGEBAK", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 9;

        /* DGEHRD */

        strcpy(xerbla_srnamt, "DGEHRD");
        xerbla_infot = 1;
        dgehrd(-1, 0, 0, a, 1, tau, w, 1, &info);
        chkxer("DGEHRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        dgehrd(0, -1, -1, a, 1, tau, w, 1, &info);
        chkxer("DGEHRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        dgehrd(0, 1, -1, a, 1, tau, w, 1, &info);
        chkxer("DGEHRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        dgehrd(1, 0, -1, a, 1, tau, w, 1, &info);
        chkxer("DGEHRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        dgehrd(0, 0, 0, a, 1, tau, w, 1, &info);
        chkxer("DGEHRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        dgehrd(2, 0, 0, a, 1, tau, w, 2, &info);
        chkxer("DGEHRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        dgehrd(2, 0, 1, a, 2, tau, w, 1, &info);
        chkxer("DGEHRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 7;

        /* DGEHD2 */

        strcpy(xerbla_srnamt, "DGEHD2");
        xerbla_infot = 1;
        dgehd2(-1, 0, 0, a, 1, tau, w, &info);
        chkxer("DGEHD2", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        dgehd2(0, -1, -1, a, 1, tau, w, &info);
        chkxer("DGEHD2", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        dgehd2(0, 1, -1, a, 1, tau, w, &info);
        chkxer("DGEHD2", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        dgehd2(1, 0, -1, a, 1, tau, w, &info);
        chkxer("DGEHD2", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        dgehd2(0, 0, 0, a, 1, tau, w, &info);
        chkxer("DGEHD2", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        dgehd2(2, 0, 0, a, 1, tau, w, &info);
        chkxer("DGEHD2", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 6;

        /* DORGHR */

        strcpy(xerbla_srnamt, "DORGHR");
        xerbla_infot = 1;
        dorghr(-1, 0, 0, a, 1, tau, w, 1, &info);
        chkxer("DORGHR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        dorghr(0, -1, -1, a, 1, tau, w, 1, &info);
        chkxer("DORGHR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        dorghr(0, 1, -1, a, 1, tau, w, 1, &info);
        chkxer("DORGHR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        dorghr(1, 0, -1, a, 1, tau, w, 1, &info);
        chkxer("DORGHR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        dorghr(0, 0, 0, a, 1, tau, w, 1, &info);
        chkxer("DORGHR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        dorghr(2, 0, 0, a, 1, tau, w, 1, &info);
        chkxer("DORGHR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        dorghr(3, 0, 2, a, 3, tau, w, 1, &info);
        chkxer("DORGHR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 7;

        /* DORMHR */

        strcpy(xerbla_srnamt, "DORMHR");
        xerbla_infot = 1;
        dormhr("/", "N", 0, 0, 0, -1, a, 1, tau, c, 1, w, 1, &info);
        chkxer("DORMHR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        dormhr("L", "/", 0, 0, 0, -1, a, 1, tau, c, 1, w, 1, &info);
        chkxer("DORMHR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        dormhr("L", "N", -1, 0, 0, -1, a, 1, tau, c, 1, w, 1, &info);
        chkxer("DORMHR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        dormhr("L", "N", 0, -1, 0, -1, a, 1, tau, c, 1, w, 1, &info);
        chkxer("DORMHR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        dormhr("L", "N", 0, 0, -1, -1, a, 1, tau, c, 1, w, 1, &info);
        chkxer("DORMHR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        dormhr("L", "N", 0, 0, 1, -1, a, 1, tau, c, 1, w, 1, &info);
        chkxer("DORMHR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        dormhr("L", "N", 1, 2, 1, 0, a, 1, tau, c, 1, w, 2, &info);
        chkxer("DORMHR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        dormhr("R", "N", 2, 1, 1, 0, a, 1, tau, c, 2, w, 2, &info);
        chkxer("DORMHR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        dormhr("L", "N", 1, 1, 0, -1, a, 1, tau, c, 1, w, 1, &info);
        chkxer("DORMHR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        dormhr("L", "N", 0, 1, 0, 0, a, 1, tau, c, 1, w, 1, &info);
        chkxer("DORMHR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        dormhr("R", "N", 1, 0, 0, 0, a, 1, tau, c, 1, w, 1, &info);
        chkxer("DORMHR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        dormhr("L", "N", 2, 1, 0, 0, a, 1, tau, c, 2, w, 1, &info);
        chkxer("DORMHR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        dormhr("R", "N", 1, 2, 0, 0, a, 1, tau, c, 1, w, 1, &info);
        chkxer("DORMHR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 11;
        dormhr("L", "N", 2, 1, 0, 0, a, 2, tau, c, 1, w, 1, &info);
        chkxer("DORMHR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 13;
        dormhr("L", "N", 1, 2, 0, 0, a, 1, tau, c, 1, w, 1, &info);
        chkxer("DORMHR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 13;
        dormhr("R", "N", 2, 1, 0, 0, a, 1, tau, c, 2, w, 1, &info);
        chkxer("DORMHR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 16;

        /* DHSEQR */

        strcpy(xerbla_srnamt, "DHSEQR");
        xerbla_infot = 1;
        dhseqr("/", "N", 0, 0, -1, a, 1, wr, wi, c, 1, w, 1, &info);
        chkxer("DHSEQR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        dhseqr("E", "/", 0, 0, -1, a, 1, wr, wi, c, 1, w, 1, &info);
        chkxer("DHSEQR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        dhseqr("E", "N", -1, 0, -1, a, 1, wr, wi, c, 1, w, 1, &info);
        chkxer("DHSEQR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        dhseqr("E", "N", 0, -1, -1, a, 1, wr, wi, c, 1, w, 1, &info);
        chkxer("DHSEQR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        dhseqr("E", "N", 0, 1, -1, a, 1, wr, wi, c, 1, w, 1, &info);
        chkxer("DHSEQR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        dhseqr("E", "N", 1, 0, -1, a, 1, wr, wi, c, 1, w, 1, &info);
        chkxer("DHSEQR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        dhseqr("E", "N", 1, 0, 1, a, 1, wr, wi, c, 1, w, 1, &info);
        chkxer("DHSEQR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        dhseqr("E", "N", 2, 0, 1, a, 1, wr, wi, c, 2, w, 1, &info);
        chkxer("DHSEQR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 11;
        dhseqr("E", "V", 2, 0, 1, a, 2, wr, wi, c, 1, w, 1, &info);
        chkxer("DHSEQR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 13;
        dhseqr("E", "N", 2, 0, 1, a, 2, wr, wi, c, 1, w, 1, &info);
        chkxer("DHSEQR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 10;

        /* DHSEIN */

        strcpy(xerbla_srnamt, "DHSEIN");
        xerbla_infot = 1;
        dhsein("/", "N", "N", sel, 0, a, 1, wr, wi, vl, 1, vr, 1,
               0, &m, w, ifaill, ifailr, &info);
        chkxer("DHSEIN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        dhsein("R", "/", "N", sel, 0, a, 1, wr, wi, vl, 1, vr, 1,
               0, &m, w, ifaill, ifailr, &info);
        chkxer("DHSEIN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        dhsein("R", "N", "/", sel, 0, a, 1, wr, wi, vl, 1, vr, 1,
               0, &m, w, ifaill, ifailr, &info);
        chkxer("DHSEIN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        dhsein("R", "N", "N", sel, -1, a, 1, wr, wi, vl, 1, vr,
               1, 0, &m, w, ifaill, ifailr, &info);
        chkxer("DHSEIN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        dhsein("R", "N", "N", sel, 2, a, 1, wr, wi, vl, 1, vr, 2,
               4, &m, w, ifaill, ifailr, &info);
        chkxer("DHSEIN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 11;
        dhsein("L", "N", "N", sel, 2, a, 2, wr, wi, vl, 1, vr, 1,
               4, &m, w, ifaill, ifailr, &info);
        chkxer("DHSEIN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 13;
        dhsein("R", "N", "N", sel, 2, a, 2, wr, wi, vl, 1, vr, 1,
               4, &m, w, ifaill, ifailr, &info);
        chkxer("DHSEIN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 14;
        dhsein("R", "N", "N", sel, 2, a, 2, wr, wi, vl, 1, vr, 2,
               1, &m, w, ifaill, ifailr, &info);
        chkxer("DHSEIN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 8;

        /* DTREVC */

        strcpy(xerbla_srnamt, "DTREVC");
        xerbla_infot = 1;
        dtrevc("/", "A", sel, 0, a, 1, vl, 1, vr, 1, 0, &m, w, &info);
        chkxer("DTREVC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        dtrevc("L", "/", sel, 0, a, 1, vl, 1, vr, 1, 0, &m, w, &info);
        chkxer("DTREVC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        dtrevc("L", "A", sel, -1, a, 1, vl, 1, vr, 1, 0, &m, w, &info);
        chkxer("DTREVC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        dtrevc("L", "A", sel, 2, a, 1, vl, 2, vr, 1, 4, &m, w, &info);
        chkxer("DTREVC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        dtrevc("L", "A", sel, 2, a, 2, vl, 1, vr, 1, 4, &m, w, &info);
        chkxer("DTREVC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        dtrevc("R", "A", sel, 2, a, 2, vl, 1, vr, 1, 4, &m, w, &info);
        chkxer("DTREVC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 11;
        dtrevc("L", "A", sel, 2, a, 2, vl, 2, vr, 1, 1, &m, w, &info);
        chkxer("DTREVC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 7;

        /* DTREVC3 */

        strcpy(xerbla_srnamt, "DTREVC3");
        xerbla_infot = 1;
        dtrevc3("/", "A", sel, 0, a, 1, vl, 1, vr, 1, 0, &m, w,
                LW, &info);
        chkxer("DTREVC3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        dtrevc3("L", "/", sel, 0, a, 1, vl, 1, vr, 1, 0, &m, w,
                LW, &info);
        chkxer("DTREVC3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        dtrevc3("L", "A", sel, -1, a, 1, vl, 1, vr, 1, 0, &m, w,
                LW, &info);
        chkxer("DTREVC3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        dtrevc3("L", "A", sel, 2, a, 1, vl, 2, vr, 1, 4, &m, w,
                LW, &info);
        chkxer("DTREVC3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        dtrevc3("L", "A", sel, 2, a, 2, vl, 1, vr, 1, 4, &m, w,
                LW, &info);
        chkxer("DTREVC3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        dtrevc3("R", "A", sel, 2, a, 2, vl, 1, vr, 1, 4, &m, w,
                LW, &info);
        chkxer("DTREVC3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 11;
        dtrevc3("L", "A", sel, 2, a, 2, vl, 2, vr, 1, 1, &m, w,
                LW, &info);
        chkxer("DTREVC3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 14;
        dtrevc3("L", "A", sel, 2, a, 2, vl, 2, vr, 1, 2, &m, w,
                2, &info);
        chkxer("DTREVC3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 8;
    }

    *ok = xerbla_ok;

    xerbla_srnamt[0] = '\0';
    xerbla_infot = 0;
}
