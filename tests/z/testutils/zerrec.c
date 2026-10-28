/**
 * @file zerrec.c
 * @brief ZERREC tests the error exits for eigencondition estimation routines.
 *
 * Port of LAPACK TESTING/EIG/zerrec.f
 * Tests error exits for: ZTRSYL, ZTRSYL3, ZTREXC, ZTRSNA, ZTRSEN
 */

#include <string.h>
#include "semicolon_lapack_complex_double.h"
#include "verify.h"

void zerrec(const char* path, INT* ok, INT* nt)
{
    enum { NMAX = 4, LW = NMAX * (NMAX + 2) };
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    INT i, ifst, ilst, info, j, m;
    f64 scale;

    INT sel[NMAX];
    f64 rw[LW], s[NMAX], sep[NMAX], swork[NMAX];
    c128 a[NMAX * NMAX], b[NMAX * NMAX], c[NMAX * NMAX];
    c128 work[LW], x[NMAX];

    (void)path;
    xerbla_ok = 1;
    *nt = 0;

    /* Initialize A, B and SEL */
    for (j = 0; j < NMAX; j++) {
        for (i = 0; i < NMAX; i++) {
            a[i + j * NMAX] = ZERO;
            b[i + j * NMAX] = ZERO;
        }
    }
    for (i = 0; i < NMAX; i++) {
        a[i + i * NMAX] = ONE;
        sel[i] = 1;
    }

    /* Test ZTRSYL */

    strcpy(xerbla_srnamt, "ZTRSYL");
    xerbla_infot = 1;
    ztrsyl("X", "N", 1, 0, 0, a, 1, b, 1, c, 1, &scale, &info);
    chkxer("ZTRSYL", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 2;
    ztrsyl("N", "X", 1, 0, 0, a, 1, b, 1, c, 1, &scale, &info);
    chkxer("ZTRSYL", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 3;
    ztrsyl("N", "N", 0, 0, 0, a, 1, b, 1, c, 1, &scale, &info);
    chkxer("ZTRSYL", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 4;
    ztrsyl("N", "N", 1, -1, 0, a, 1, b, 1, c, 1, &scale, &info);
    chkxer("ZTRSYL", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 5;
    ztrsyl("N", "N", 1, 0, -1, a, 1, b, 1, c, 1, &scale, &info);
    chkxer("ZTRSYL", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 7;
    ztrsyl("N", "N", 1, 2, 0, a, 1, b, 1, c, 2, &scale, &info);
    chkxer("ZTRSYL", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 9;
    ztrsyl("N", "N", 1, 0, 2, a, 1, b, 1, c, 1, &scale, &info);
    chkxer("ZTRSYL", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 11;
    ztrsyl("N", "N", 1, 2, 0, a, 2, b, 1, c, 1, &scale, &info);
    chkxer("ZTRSYL", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    *nt = *nt + 8;

    /* Test ZTRSYL3 */

    strcpy(xerbla_srnamt, "ZTRSYL3");
    xerbla_infot = 1;
    ztrsyl3("X", "N", 1, 0, 0, a, 1, b, 1, c, 1, &scale,
            swork, NMAX, &info);
    chkxer("ZTRSYL3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 2;
    ztrsyl3("N", "X", 1, 0, 0, a, 1, b, 1, c, 1, &scale,
            swork, NMAX, &info);
    chkxer("ZTRSYL3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 3;
    ztrsyl3("N", "N", 0, 0, 0, a, 1, b, 1, c, 1, &scale,
            swork, NMAX, &info);
    chkxer("ZTRSYL3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 4;
    ztrsyl3("N", "N", 1, -1, 0, a, 1, b, 1, c, 1, &scale,
            swork, NMAX, &info);
    chkxer("ZTRSYL3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 5;
    ztrsyl3("N", "N", 1, 0, -1, a, 1, b, 1, c, 1, &scale,
            swork, NMAX, &info);
    chkxer("ZTRSYL3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 7;
    ztrsyl3("N", "N", 1, 2, 0, a, 1, b, 1, c, 2, &scale,
            swork, NMAX, &info);
    chkxer("ZTRSYL3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 9;
    ztrsyl3("N", "N", 1, 0, 2, a, 1, b, 1, c, 1, &scale,
            swork, NMAX, &info);
    chkxer("ZTRSYL3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 11;
    ztrsyl3("N", "N", 1, 2, 0, a, 2, b, 1, c, 1, &scale,
            swork, NMAX, &info);
    chkxer("ZTRSYL3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    *nt = *nt + 8;

    /* Test ZTREXC */

    strcpy(xerbla_srnamt, "ZTREXC");
    ifst = 0;
    ilst = 0;
    xerbla_infot = 1;
    ztrexc("X", 1, a, 1, b, 1, ifst, ilst, &info);
    chkxer("ZTREXC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 2;
    ztrexc("N", -1, a, 1, b, 1, ifst, ilst, &info);
    chkxer("ZTREXC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 4;
    ilst = 1;
    ztrexc("N", 2, a, 1, b, 1, ifst, ilst, &info);
    chkxer("ZTREXC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 6;
    ztrexc("V", 2, a, 2, b, 1, ifst, ilst, &info);
    chkxer("ZTREXC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 7;
    ifst = -1;
    ilst = 0;
    ztrexc("V", 1, a, 1, b, 1, ifst, ilst, &info);
    chkxer("ZTREXC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 7;
    ifst = 1;
    ztrexc("V", 1, a, 1, b, 1, ifst, ilst, &info);
    chkxer("ZTREXC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 8;
    ifst = 0;
    ilst = -1;
    ztrexc("V", 1, a, 1, b, 1, ifst, ilst, &info);
    chkxer("ZTREXC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 8;
    ilst = 1;
    ztrexc("V", 1, a, 1, b, 1, ifst, ilst, &info);
    chkxer("ZTREXC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    *nt = *nt + 8;

    /* Test ZTRSNA */

    strcpy(xerbla_srnamt, "ZTRSNA");
    xerbla_infot = 1;
    ztrsna("X", "A", sel, 0, a, 1, b, 1, c, 1, s, sep, 1, &m,
           work, 1, rw, &info);
    chkxer("ZTRSNA", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 2;
    ztrsna("B", "X", sel, 0, a, 1, b, 1, c, 1, s, sep, 1, &m,
           work, 1, rw, &info);
    chkxer("ZTRSNA", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 4;
    ztrsna("B", "A", sel, -1, a, 1, b, 1, c, 1, s, sep, 1, &m,
           work, 1, rw, &info);
    chkxer("ZTRSNA", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 6;
    ztrsna("V", "A", sel, 2, a, 1, b, 1, c, 1, s, sep, 2, &m,
           work, 2, rw, &info);
    chkxer("ZTRSNA", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 8;
    ztrsna("B", "A", sel, 2, a, 2, b, 1, c, 2, s, sep, 2, &m,
           work, 2, rw, &info);
    chkxer("ZTRSNA", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 10;
    ztrsna("B", "A", sel, 2, a, 2, b, 2, c, 1, s, sep, 2, &m,
           work, 2, rw, &info);
    chkxer("ZTRSNA", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 13;
    ztrsna("B", "A", sel, 1, a, 1, b, 1, c, 1, s, sep, 0, &m,
           work, 1, rw, &info);
    chkxer("ZTRSNA", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 13;
    ztrsna("B", "S", sel, 2, a, 2, b, 2, c, 2, s, sep, 1, &m,
           work, 1, rw, &info);
    chkxer("ZTRSNA", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 16;
    ztrsna("B", "A", sel, 2, a, 2, b, 2, c, 2, s, sep, 2, &m,
           work, 1, rw, &info);
    chkxer("ZTRSNA", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    *nt = *nt + 9;

    /* Test ZTRSEN */

    sel[0] = 0;
    strcpy(xerbla_srnamt, "ZTRSEN");
    xerbla_infot = 1;
    ztrsen("X", "N", sel, 0, a, 1, b, 1, x, &m, &s[0], &sep[0],
           work, 1, &info);
    chkxer("ZTRSEN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 2;
    ztrsen("N", "X", sel, 0, a, 1, b, 1, x, &m, &s[0], &sep[0],
           work, 1, &info);
    chkxer("ZTRSEN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 4;
    ztrsen("N", "N", sel, -1, a, 1, b, 1, x, &m, &s[0],
           &sep[0], work, 1, &info);
    chkxer("ZTRSEN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 6;
    ztrsen("N", "N", sel, 2, a, 1, b, 1, x, &m, &s[0], &sep[0],
           work, 2, &info);
    chkxer("ZTRSEN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 8;
    ztrsen("N", "V", sel, 2, a, 2, b, 1, x, &m, &s[0], &sep[0],
           work, 1, &info);
    chkxer("ZTRSEN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 14;
    ztrsen("N", "V", sel, 2, a, 2, b, 2, x, &m, &s[0], &sep[0],
           work, 0, &info);
    chkxer("ZTRSEN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 14;
    ztrsen("E", "V", sel, 3, a, 3, b, 3, x, &m, &s[0], &sep[0],
           work, 1, &info);
    chkxer("ZTRSEN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 14;
    ztrsen("V", "V", sel, 3, a, 3, b, 3, x, &m, &s[0], &sep[0],
           work, 3, &info);
    chkxer("ZTRSEN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    *nt = *nt + 8;

    *ok = xerbla_ok;

    xerbla_srnamt[0] = '\0';
    xerbla_infot = 0;
}
