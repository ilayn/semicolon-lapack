/**
 * @file cerrec.c
 * @brief CERREC tests the error exits for eigencondition estimation routines.
 *
 * Port of LAPACK TESTING/EIG/cerrec.f
 * Tests error exits for: CTRSYL, CTRSYL3, CTREXC, CTRSNA, CTRSEN
 */

#include <string.h>
#include "semicolon_lapack_complex_single.h"
#include "verify.h"

void cerrec(const char* path, INT* ok, INT* nt)
{
    enum { NMAX = 4, LW = NMAX * (NMAX + 2) };
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;

    INT i, ifst, ilst, info, j, m;
    f32 scale;

    INT sel[NMAX];
    f32 rw[LW], s[NMAX], sep[NMAX], swork[NMAX];
    c64 a[NMAX * NMAX], b[NMAX * NMAX], c[NMAX * NMAX];
    c64 work[LW], x[NMAX];

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

    /* Test CTRSYL */

    strcpy(xerbla_srnamt, "CTRSYL");
    xerbla_infot = 1;
    ctrsyl("X", "N", 1, 0, 0, a, 1, b, 1, c, 1, &scale, &info);
    chkxer("CTRSYL", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 2;
    ctrsyl("N", "X", 1, 0, 0, a, 1, b, 1, c, 1, &scale, &info);
    chkxer("CTRSYL", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 3;
    ctrsyl("N", "N", 0, 0, 0, a, 1, b, 1, c, 1, &scale, &info);
    chkxer("CTRSYL", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 4;
    ctrsyl("N", "N", 1, -1, 0, a, 1, b, 1, c, 1, &scale, &info);
    chkxer("CTRSYL", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 5;
    ctrsyl("N", "N", 1, 0, -1, a, 1, b, 1, c, 1, &scale, &info);
    chkxer("CTRSYL", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 7;
    ctrsyl("N", "N", 1, 2, 0, a, 1, b, 1, c, 2, &scale, &info);
    chkxer("CTRSYL", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 9;
    ctrsyl("N", "N", 1, 0, 2, a, 1, b, 1, c, 1, &scale, &info);
    chkxer("CTRSYL", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 11;
    ctrsyl("N", "N", 1, 2, 0, a, 2, b, 1, c, 1, &scale, &info);
    chkxer("CTRSYL", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    *nt = *nt + 8;

    /* Test CTRSYL3 */

    strcpy(xerbla_srnamt, "CTRSYL3");
    xerbla_infot = 1;
    ctrsyl3("X", "N", 1, 0, 0, a, 1, b, 1, c, 1, &scale,
            swork, NMAX, &info);
    chkxer("CTRSYL3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 2;
    ctrsyl3("N", "X", 1, 0, 0, a, 1, b, 1, c, 1, &scale,
            swork, NMAX, &info);
    chkxer("CTRSYL3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 3;
    ctrsyl3("N", "N", 0, 0, 0, a, 1, b, 1, c, 1, &scale,
            swork, NMAX, &info);
    chkxer("CTRSYL3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 4;
    ctrsyl3("N", "N", 1, -1, 0, a, 1, b, 1, c, 1, &scale,
            swork, NMAX, &info);
    chkxer("CTRSYL3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 5;
    ctrsyl3("N", "N", 1, 0, -1, a, 1, b, 1, c, 1, &scale,
            swork, NMAX, &info);
    chkxer("CTRSYL3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 7;
    ctrsyl3("N", "N", 1, 2, 0, a, 1, b, 1, c, 2, &scale,
            swork, NMAX, &info);
    chkxer("CTRSYL3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 9;
    ctrsyl3("N", "N", 1, 0, 2, a, 1, b, 1, c, 1, &scale,
            swork, NMAX, &info);
    chkxer("CTRSYL3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 11;
    ctrsyl3("N", "N", 1, 2, 0, a, 2, b, 1, c, 1, &scale,
            swork, NMAX, &info);
    chkxer("CTRSYL3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    *nt = *nt + 8;

    /* Test CTREXC */

    strcpy(xerbla_srnamt, "CTREXC");
    ifst = 0;
    ilst = 0;
    xerbla_infot = 1;
    ctrexc("X", 1, a, 1, b, 1, ifst, ilst, &info);
    chkxer("CTREXC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 2;
    ctrexc("N", -1, a, 1, b, 1, ifst, ilst, &info);
    chkxer("CTREXC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 4;
    ilst = 1;
    ctrexc("N", 2, a, 1, b, 1, ifst, ilst, &info);
    chkxer("CTREXC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 6;
    ctrexc("V", 2, a, 2, b, 1, ifst, ilst, &info);
    chkxer("CTREXC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 7;
    ifst = -1;
    ilst = 0;
    ctrexc("V", 1, a, 1, b, 1, ifst, ilst, &info);
    chkxer("CTREXC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 7;
    ifst = 1;
    ctrexc("V", 1, a, 1, b, 1, ifst, ilst, &info);
    chkxer("CTREXC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 8;
    ifst = 0;
    ilst = -1;
    ctrexc("V", 1, a, 1, b, 1, ifst, ilst, &info);
    chkxer("CTREXC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 8;
    ilst = 1;
    ctrexc("V", 1, a, 1, b, 1, ifst, ilst, &info);
    chkxer("CTREXC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    *nt = *nt + 8;

    /* Test CTRSNA */

    strcpy(xerbla_srnamt, "CTRSNA");
    xerbla_infot = 1;
    ctrsna("X", "A", sel, 0, a, 1, b, 1, c, 1, s, sep, 1, &m,
           work, 1, rw, &info);
    chkxer("CTRSNA", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 2;
    ctrsna("B", "X", sel, 0, a, 1, b, 1, c, 1, s, sep, 1, &m,
           work, 1, rw, &info);
    chkxer("CTRSNA", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 4;
    ctrsna("B", "A", sel, -1, a, 1, b, 1, c, 1, s, sep, 1, &m,
           work, 1, rw, &info);
    chkxer("CTRSNA", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 6;
    ctrsna("V", "A", sel, 2, a, 1, b, 1, c, 1, s, sep, 2, &m,
           work, 2, rw, &info);
    chkxer("CTRSNA", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 8;
    ctrsna("B", "A", sel, 2, a, 2, b, 1, c, 2, s, sep, 2, &m,
           work, 2, rw, &info);
    chkxer("CTRSNA", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 10;
    ctrsna("B", "A", sel, 2, a, 2, b, 2, c, 1, s, sep, 2, &m,
           work, 2, rw, &info);
    chkxer("CTRSNA", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 13;
    ctrsna("B", "A", sel, 1, a, 1, b, 1, c, 1, s, sep, 0, &m,
           work, 1, rw, &info);
    chkxer("CTRSNA", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 13;
    ctrsna("B", "S", sel, 2, a, 2, b, 2, c, 2, s, sep, 1, &m,
           work, 1, rw, &info);
    chkxer("CTRSNA", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 16;
    ctrsna("B", "A", sel, 2, a, 2, b, 2, c, 2, s, sep, 2, &m,
           work, 1, rw, &info);
    chkxer("CTRSNA", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    *nt = *nt + 9;

    /* Test CTRSEN */

    sel[0] = 0;
    strcpy(xerbla_srnamt, "CTRSEN");
    xerbla_infot = 1;
    ctrsen("X", "N", sel, 0, a, 1, b, 1, x, &m, &s[0], &sep[0],
           work, 1, &info);
    chkxer("CTRSEN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 2;
    ctrsen("N", "X", sel, 0, a, 1, b, 1, x, &m, &s[0], &sep[0],
           work, 1, &info);
    chkxer("CTRSEN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 4;
    ctrsen("N", "N", sel, -1, a, 1, b, 1, x, &m, &s[0],
           &sep[0], work, 1, &info);
    chkxer("CTRSEN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 6;
    ctrsen("N", "N", sel, 2, a, 1, b, 1, x, &m, &s[0], &sep[0],
           work, 2, &info);
    chkxer("CTRSEN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 8;
    ctrsen("N", "V", sel, 2, a, 2, b, 1, x, &m, &s[0], &sep[0],
           work, 1, &info);
    chkxer("CTRSEN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 14;
    ctrsen("N", "V", sel, 2, a, 2, b, 2, x, &m, &s[0], &sep[0],
           work, 0, &info);
    chkxer("CTRSEN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 14;
    ctrsen("E", "V", sel, 3, a, 3, b, 3, x, &m, &s[0], &sep[0],
           work, 1, &info);
    chkxer("CTRSEN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 14;
    ctrsen("V", "V", sel, 3, a, 3, b, 3, x, &m, &s[0], &sep[0],
           work, 3, &info);
    chkxer("CTRSEN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    *nt = *nt + 8;

    *ok = xerbla_ok;

    xerbla_srnamt[0] = '\0';
    xerbla_infot = 0;
}
