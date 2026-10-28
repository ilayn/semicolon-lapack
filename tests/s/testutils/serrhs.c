/**
 * @file serrhs.c
 * @brief SERRHS tests the error exits for the nonsymmetric eigenvalue routines.
 *
 * Port of LAPACK TESTING/EIG/serrhs.f
 * Tests error exits for: SGEBAK, SGEBAL, SGEHRD, SGEHD2, SORGHR, SORMHR,
 * SHSEQR, SHSEIN, STREVC, STREVC3
 */

#include <string.h>
#include "semicolon_lapack_single.h"
#include "verify.h"

void serrhs(const char* path, INT* ok, INT* nt)
{
    enum { NMAX = 3, LW = (NMAX + 2) * (NMAX + 2) + NMAX };
    const f32 ONE = 1.0f;

    INT i, ihi, ilo, info, j, m;

    INT sel[NMAX];
    INT ifaill[NMAX], ifailr[NMAX];
    f32 a[NMAX * NMAX], c[NMAX * NMAX], s[NMAX], tau[NMAX];
    f32 vl[NMAX * NMAX], vr[NMAX * NMAX], w[LW], wi[NMAX], wr[NMAX];

    /* Set the variables to innocuous values. */
    for (j = 0; j < NMAX; j++) {
        for (i = 0; i < NMAX; i++) {
            a[i + j * NMAX] = ONE / (f32)(i + j + 2);
        }
        wi[j] = (f32)(j + 1);
        sel[j] = 1;
    }
    xerbla_ok = 1;
    *nt = 0;

    /* Test error exits of the nonsymmetric eigenvalue routines. */

    if ((path[1] == 'H' || path[1] == 'h') &&
        (path[2] == 'S' || path[2] == 's')) {

        /* SGEBAL */

        strcpy(xerbla_srnamt, "SGEBAL");
        xerbla_infot = 1;
        sgebal("/", 0, a, 1, &ilo, &ihi, s, &info);
        chkxer("SGEBAL", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        sgebal("N", -1, a, 1, &ilo, &ihi, s, &info);
        chkxer("SGEBAL", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        sgebal("N", 2, a, 1, &ilo, &ihi, s, &info);
        chkxer("SGEBAL", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 3;

        /* SGEBAK */

        strcpy(xerbla_srnamt, "SGEBAK");
        xerbla_infot = 1;
        sgebak("/", "R", 0, 0, -1, s, 0, a, 1, &info);
        chkxer("SGEBAK", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        sgebak("N", "/", 0, 0, -1, s, 0, a, 1, &info);
        chkxer("SGEBAK", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        sgebak("N", "R", -1, 0, -1, s, 0, a, 1, &info);
        chkxer("SGEBAK", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        sgebak("N", "R", 0, -1, -1, s, 0, a, 1, &info);
        chkxer("SGEBAK", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        sgebak("N", "R", 0, 1, -1, s, 0, a, 1, &info);
        chkxer("SGEBAK", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        sgebak("N", "R", 2, 1, 0, s, 0, a, 2, &info);
        chkxer("SGEBAK", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        sgebak("N", "R", 0, 0, 0, s, 0, a, 1, &info);
        chkxer("SGEBAK", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        sgebak("N", "R", 0, 0, -1, s, -1, a, 1, &info);
        chkxer("SGEBAK", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        sgebak("N", "R", 2, 0, 1, s, 0, a, 1, &info);
        chkxer("SGEBAK", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 9;

        /* SGEHRD */

        strcpy(xerbla_srnamt, "SGEHRD");
        xerbla_infot = 1;
        sgehrd(-1, 0, 0, a, 1, tau, w, 1, &info);
        chkxer("SGEHRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        sgehrd(0, -1, -1, a, 1, tau, w, 1, &info);
        chkxer("SGEHRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        sgehrd(0, 1, -1, a, 1, tau, w, 1, &info);
        chkxer("SGEHRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        sgehrd(1, 0, -1, a, 1, tau, w, 1, &info);
        chkxer("SGEHRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        sgehrd(0, 0, 0, a, 1, tau, w, 1, &info);
        chkxer("SGEHRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        sgehrd(2, 0, 0, a, 1, tau, w, 2, &info);
        chkxer("SGEHRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        sgehrd(2, 0, 1, a, 2, tau, w, 1, &info);
        chkxer("SGEHRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 7;

        /* SGEHD2 */

        strcpy(xerbla_srnamt, "SGEHD2");
        xerbla_infot = 1;
        sgehd2(-1, 0, 0, a, 1, tau, w, &info);
        chkxer("SGEHD2", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        sgehd2(0, -1, -1, a, 1, tau, w, &info);
        chkxer("SGEHD2", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        sgehd2(0, 1, -1, a, 1, tau, w, &info);
        chkxer("SGEHD2", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        sgehd2(1, 0, -1, a, 1, tau, w, &info);
        chkxer("SGEHD2", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        sgehd2(0, 0, 0, a, 1, tau, w, &info);
        chkxer("SGEHD2", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        sgehd2(2, 0, 0, a, 1, tau, w, &info);
        chkxer("SGEHD2", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 6;

        /* SORGHR */

        strcpy(xerbla_srnamt, "SORGHR");
        xerbla_infot = 1;
        sorghr(-1, 0, 0, a, 1, tau, w, 1, &info);
        chkxer("SORGHR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        sorghr(0, -1, -1, a, 1, tau, w, 1, &info);
        chkxer("SORGHR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        sorghr(0, 1, -1, a, 1, tau, w, 1, &info);
        chkxer("SORGHR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        sorghr(1, 0, -1, a, 1, tau, w, 1, &info);
        chkxer("SORGHR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        sorghr(0, 0, 0, a, 1, tau, w, 1, &info);
        chkxer("SORGHR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        sorghr(2, 0, 0, a, 1, tau, w, 1, &info);
        chkxer("SORGHR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        sorghr(3, 0, 2, a, 3, tau, w, 1, &info);
        chkxer("SORGHR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 7;

        /* SORMHR */

        strcpy(xerbla_srnamt, "SORMHR");
        xerbla_infot = 1;
        sormhr("/", "N", 0, 0, 0, -1, a, 1, tau, c, 1, w, 1, &info);
        chkxer("SORMHR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        sormhr("L", "/", 0, 0, 0, -1, a, 1, tau, c, 1, w, 1, &info);
        chkxer("SORMHR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        sormhr("L", "N", -1, 0, 0, -1, a, 1, tau, c, 1, w, 1, &info);
        chkxer("SORMHR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        sormhr("L", "N", 0, -1, 0, -1, a, 1, tau, c, 1, w, 1, &info);
        chkxer("SORMHR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        sormhr("L", "N", 0, 0, -1, -1, a, 1, tau, c, 1, w, 1, &info);
        chkxer("SORMHR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        sormhr("L", "N", 0, 0, 1, -1, a, 1, tau, c, 1, w, 1, &info);
        chkxer("SORMHR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        sormhr("L", "N", 1, 2, 1, 0, a, 1, tau, c, 1, w, 2, &info);
        chkxer("SORMHR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        sormhr("R", "N", 2, 1, 1, 0, a, 1, tau, c, 2, w, 2, &info);
        chkxer("SORMHR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        sormhr("L", "N", 1, 1, 0, -1, a, 1, tau, c, 1, w, 1, &info);
        chkxer("SORMHR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        sormhr("L", "N", 0, 1, 0, 0, a, 1, tau, c, 1, w, 1, &info);
        chkxer("SORMHR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        sormhr("R", "N", 1, 0, 0, 0, a, 1, tau, c, 1, w, 1, &info);
        chkxer("SORMHR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        sormhr("L", "N", 2, 1, 0, 0, a, 1, tau, c, 2, w, 1, &info);
        chkxer("SORMHR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        sormhr("R", "N", 1, 2, 0, 0, a, 1, tau, c, 1, w, 1, &info);
        chkxer("SORMHR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 11;
        sormhr("L", "N", 2, 1, 0, 0, a, 2, tau, c, 1, w, 1, &info);
        chkxer("SORMHR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 13;
        sormhr("L", "N", 1, 2, 0, 0, a, 1, tau, c, 1, w, 1, &info);
        chkxer("SORMHR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 13;
        sormhr("R", "N", 2, 1, 0, 0, a, 1, tau, c, 2, w, 1, &info);
        chkxer("SORMHR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 16;

        /* SHSEQR */

        strcpy(xerbla_srnamt, "SHSEQR");
        xerbla_infot = 1;
        shseqr("/", "N", 0, 0, -1, a, 1, wr, wi, c, 1, w, 1, &info);
        chkxer("SHSEQR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        shseqr("E", "/", 0, 0, -1, a, 1, wr, wi, c, 1, w, 1, &info);
        chkxer("SHSEQR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        shseqr("E", "N", -1, 0, -1, a, 1, wr, wi, c, 1, w, 1, &info);
        chkxer("SHSEQR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        shseqr("E", "N", 0, -1, -1, a, 1, wr, wi, c, 1, w, 1, &info);
        chkxer("SHSEQR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        shseqr("E", "N", 0, 1, -1, a, 1, wr, wi, c, 1, w, 1, &info);
        chkxer("SHSEQR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        shseqr("E", "N", 1, 0, -1, a, 1, wr, wi, c, 1, w, 1, &info);
        chkxer("SHSEQR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        shseqr("E", "N", 1, 0, 1, a, 1, wr, wi, c, 1, w, 1, &info);
        chkxer("SHSEQR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        shseqr("E", "N", 2, 0, 1, a, 1, wr, wi, c, 2, w, 1, &info);
        chkxer("SHSEQR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 11;
        shseqr("E", "V", 2, 0, 1, a, 2, wr, wi, c, 1, w, 1, &info);
        chkxer("SHSEQR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 13;
        shseqr("E", "N", 2, 0, 1, a, 2, wr, wi, c, 1, w, 1, &info);
        chkxer("SHSEQR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 10;

        /* SHSEIN */

        strcpy(xerbla_srnamt, "SHSEIN");
        xerbla_infot = 1;
        shsein("/", "N", "N", sel, 0, a, 1, wr, wi, vl, 1, vr, 1,
               0, &m, w, ifaill, ifailr, &info);
        chkxer("SHSEIN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        shsein("R", "/", "N", sel, 0, a, 1, wr, wi, vl, 1, vr, 1,
               0, &m, w, ifaill, ifailr, &info);
        chkxer("SHSEIN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        shsein("R", "N", "/", sel, 0, a, 1, wr, wi, vl, 1, vr, 1,
               0, &m, w, ifaill, ifailr, &info);
        chkxer("SHSEIN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        shsein("R", "N", "N", sel, -1, a, 1, wr, wi, vl, 1, vr,
               1, 0, &m, w, ifaill, ifailr, &info);
        chkxer("SHSEIN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        shsein("R", "N", "N", sel, 2, a, 1, wr, wi, vl, 1, vr, 2,
               4, &m, w, ifaill, ifailr, &info);
        chkxer("SHSEIN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 11;
        shsein("L", "N", "N", sel, 2, a, 2, wr, wi, vl, 1, vr, 1,
               4, &m, w, ifaill, ifailr, &info);
        chkxer("SHSEIN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 13;
        shsein("R", "N", "N", sel, 2, a, 2, wr, wi, vl, 1, vr, 1,
               4, &m, w, ifaill, ifailr, &info);
        chkxer("SHSEIN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 14;
        shsein("R", "N", "N", sel, 2, a, 2, wr, wi, vl, 1, vr, 2,
               1, &m, w, ifaill, ifailr, &info);
        chkxer("SHSEIN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 8;

        /* STREVC */

        strcpy(xerbla_srnamt, "STREVC");
        xerbla_infot = 1;
        strevc("/", "A", sel, 0, a, 1, vl, 1, vr, 1, 0, &m, w, &info);
        chkxer("STREVC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        strevc("L", "/", sel, 0, a, 1, vl, 1, vr, 1, 0, &m, w, &info);
        chkxer("STREVC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        strevc("L", "A", sel, -1, a, 1, vl, 1, vr, 1, 0, &m, w, &info);
        chkxer("STREVC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        strevc("L", "A", sel, 2, a, 1, vl, 2, vr, 1, 4, &m, w, &info);
        chkxer("STREVC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        strevc("L", "A", sel, 2, a, 2, vl, 1, vr, 1, 4, &m, w, &info);
        chkxer("STREVC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        strevc("R", "A", sel, 2, a, 2, vl, 1, vr, 1, 4, &m, w, &info);
        chkxer("STREVC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 11;
        strevc("L", "A", sel, 2, a, 2, vl, 2, vr, 1, 1, &m, w, &info);
        chkxer("STREVC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 7;

        /* STREVC3 */

        strcpy(xerbla_srnamt, "STREVC3");
        xerbla_infot = 1;
        strevc3("/", "A", sel, 0, a, 1, vl, 1, vr, 1, 0, &m, w,
                LW, &info);
        chkxer("STREVC3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        strevc3("L", "/", sel, 0, a, 1, vl, 1, vr, 1, 0, &m, w,
                LW, &info);
        chkxer("STREVC3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        strevc3("L", "A", sel, -1, a, 1, vl, 1, vr, 1, 0, &m, w,
                LW, &info);
        chkxer("STREVC3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        strevc3("L", "A", sel, 2, a, 1, vl, 2, vr, 1, 4, &m, w,
                LW, &info);
        chkxer("STREVC3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        strevc3("L", "A", sel, 2, a, 2, vl, 1, vr, 1, 4, &m, w,
                LW, &info);
        chkxer("STREVC3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        strevc3("R", "A", sel, 2, a, 2, vl, 1, vr, 1, 4, &m, w,
                LW, &info);
        chkxer("STREVC3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 11;
        strevc3("L", "A", sel, 2, a, 2, vl, 2, vr, 1, 1, &m, w,
                LW, &info);
        chkxer("STREVC3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 14;
        strevc3("L", "A", sel, 2, a, 2, vl, 2, vr, 1, 2, &m, w,
                2, &info);
        chkxer("STREVC3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 8;
    }

    *ok = xerbla_ok;

    xerbla_srnamt[0] = '\0';
    xerbla_infot = 0;
}
