/**
 * @file derrec.c
 * @brief DERREC tests the error exits for eigencondition estimation routines.
 *
 * Port of LAPACK TESTING/EIG/derrec.f
 * Tests error exits for: DTRSYL, DTRSYL3, DTREXC, DTRSNA, DTRSEN
 */

#include <string.h>
#include "semicolon_lapack_double.h"
#include "verify.h"

void derrec(int* ok, int* nt)
{
    const int NMAX = 4;
    const f64 ONE = 1.0;
    const f64 ZERO = 0.0;

    int i, j, ifst, ilst, info, m;
    f64 scale;

    int sel[NMAX];
    int iwork[NMAX];
    f64 a[NMAX * NMAX], b[NMAX * NMAX], c[NMAX * NMAX];
    f64 s[NMAX], sep[NMAX], wi[NMAX], work[NMAX], wr[NMAX];

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

    /* Test DTRSYL */

    strcpy(xerbla_srnamt, "DTRSYL");
    xerbla_infot = 1;
    dtrsyl("X", "N", 1, 0, 0, a, 1, b, 1, c, 1, &scale, &info);
    chkxer("DTRSYL", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 2;
    dtrsyl("N", "X", 1, 0, 0, a, 1, b, 1, c, 1, &scale, &info);
    chkxer("DTRSYL", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 3;
    dtrsyl("N", "N", 0, 0, 0, a, 1, b, 1, c, 1, &scale, &info);
    chkxer("DTRSYL", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 4;
    dtrsyl("N", "N", 1, -1, 0, a, 1, b, 1, c, 1, &scale, &info);
    chkxer("DTRSYL", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 5;
    dtrsyl("N", "N", 1, 0, -1, a, 1, b, 1, c, 1, &scale, &info);
    chkxer("DTRSYL", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 7;
    dtrsyl("N", "N", 1, 2, 0, a, 1, b, 1, c, 2, &scale, &info);
    chkxer("DTRSYL", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 9;
    dtrsyl("N", "N", 1, 0, 2, a, 1, b, 1, c, 1, &scale, &info);
    chkxer("DTRSYL", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 11;
    dtrsyl("N", "N", 1, 2, 0, a, 2, b, 1, c, 1, &scale, &info);
    chkxer("DTRSYL", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    *nt = *nt + 8;

    /* Test DTRSYL3 */

    strcpy(xerbla_srnamt, "DTRSYL3");
    xerbla_infot = 1;
    dtrsyl3("X", "N", 1, 0, 0, a, 1, b, 1, c, 1, &scale,
            iwork, NMAX, (f64*)work, NMAX, &info);
    chkxer("DTRSYL3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 2;
    dtrsyl3("N", "X", 1, 0, 0, a, 1, b, 1, c, 1, &scale,
            iwork, NMAX, (f64*)work, NMAX, &info);
    chkxer("DTRSYL3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 3;
    dtrsyl3("N", "N", 0, 0, 0, a, 1, b, 1, c, 1, &scale,
            iwork, NMAX, (f64*)work, NMAX, &info);
    chkxer("DTRSYL3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 4;
    dtrsyl3("N", "N", 1, -1, 0, a, 1, b, 1, c, 1, &scale,
            iwork, NMAX, (f64*)work, NMAX, &info);
    chkxer("DTRSYL3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 5;
    dtrsyl3("N", "N", 1, 0, -1, a, 1, b, 1, c, 1, &scale,
            iwork, NMAX, (f64*)work, NMAX, &info);
    chkxer("DTRSYL3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 7;
    dtrsyl3("N", "N", 1, 2, 0, a, 1, b, 1, c, 2, &scale,
            iwork, NMAX, (f64*)work, NMAX, &info);
    chkxer("DTRSYL3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 9;
    dtrsyl3("N", "N", 1, 0, 2, a, 1, b, 1, c, 1, &scale,
            iwork, NMAX, (f64*)work, NMAX, &info);
    chkxer("DTRSYL3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 11;
    dtrsyl3("N", "N", 1, 2, 0, a, 2, b, 1, c, 1, &scale,
            iwork, NMAX, (f64*)work, NMAX, &info);
    chkxer("DTRSYL3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    *nt = *nt + 8;

    /* Test DTREXC */

    strcpy(xerbla_srnamt, "DTREXC");
    ifst = 0;
    ilst = 0;
    xerbla_infot = 1;
    dtrexc("X", 1, a, 1, b, 1, &ifst, &ilst, work, &info);
    chkxer("DTREXC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 2;
    dtrexc("N", -1, a, 1, b, 1, &ifst, &ilst, work, &info);
    chkxer("DTREXC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 4;
    ilst = 1;
    dtrexc("N", 2, a, 1, b, 1, &ifst, &ilst, work, &info);
    chkxer("DTREXC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 6;
    dtrexc("V", 2, a, 2, b, 1, &ifst, &ilst, work, &info);
    chkxer("DTREXC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 7;
    ifst = -1;
    ilst = 0;
    dtrexc("V", 1, a, 1, b, 1, &ifst, &ilst, work, &info);
    chkxer("DTREXC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 7;
    ifst = 1;
    dtrexc("V", 1, a, 1, b, 1, &ifst, &ilst, work, &info);
    chkxer("DTREXC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 8;
    ifst = 0;
    ilst = -1;
    dtrexc("V", 1, a, 1, b, 1, &ifst, &ilst, work, &info);
    chkxer("DTREXC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 8;
    ilst = 1;
    dtrexc("V", 1, a, 1, b, 1, &ifst, &ilst, work, &info);
    chkxer("DTREXC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    *nt = *nt + 8;

    /* Test DTRSNA */

    strcpy(xerbla_srnamt, "DTRSNA");
    xerbla_infot = 1;
    dtrsna("X", "A", sel, 0, a, 1, b, 1, c, 1, s, sep, 1, &m,
           work, 1, iwork, &info);
    chkxer("DTRSNA", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 2;
    dtrsna("B", "X", sel, 0, a, 1, b, 1, c, 1, s, sep, 1, &m,
           work, 1, iwork, &info);
    chkxer("DTRSNA", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 4;
    dtrsna("B", "A", sel, -1, a, 1, b, 1, c, 1, s, sep, 1, &m,
           work, 1, iwork, &info);
    chkxer("DTRSNA", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 6;
    dtrsna("V", "A", sel, 2, a, 1, b, 1, c, 1, s, sep, 2, &m,
           work, 2, iwork, &info);
    chkxer("DTRSNA", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 8;
    dtrsna("B", "A", sel, 2, a, 2, b, 1, c, 2, s, sep, 2, &m,
           work, 2, iwork, &info);
    chkxer("DTRSNA", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 10;
    dtrsna("B", "A", sel, 2, a, 2, b, 2, c, 1, s, sep, 2, &m,
           work, 2, iwork, &info);
    chkxer("DTRSNA", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 13;
    dtrsna("B", "A", sel, 1, a, 1, b, 1, c, 1, s, sep, 0, &m,
           work, 1, iwork, &info);
    chkxer("DTRSNA", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 13;
    dtrsna("B", "S", sel, 2, a, 2, b, 2, c, 2, s, sep, 1, &m,
           work, 2, iwork, &info);
    chkxer("DTRSNA", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 16;
    dtrsna("B", "A", sel, 2, a, 2, b, 2, c, 2, s, sep, 2, &m,
           work, 1, iwork, &info);
    chkxer("DTRSNA", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    *nt = *nt + 9;

    /* Test DTRSEN */

    sel[0] = 0;
    strcpy(xerbla_srnamt, "DTRSEN");
    xerbla_infot = 1;
    dtrsen("X", "N", sel, 0, a, 1, b, 1, wr, wi, &m, &s[0],
           &sep[0], work, 1, iwork, 1, &info);
    chkxer("DTRSEN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 2;
    dtrsen("N", "X", sel, 0, a, 1, b, 1, wr, wi, &m, &s[0],
           &sep[0], work, 1, iwork, 1, &info);
    chkxer("DTRSEN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 4;
    dtrsen("N", "N", sel, -1, a, 1, b, 1, wr, wi, &m, &s[0],
           &sep[0], work, 1, iwork, 1, &info);
    chkxer("DTRSEN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 6;
    dtrsen("N", "N", sel, 2, a, 1, b, 1, wr, wi, &m, &s[0],
           &sep[0], work, 2, iwork, 1, &info);
    chkxer("DTRSEN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 8;
    dtrsen("N", "V", sel, 2, a, 2, b, 1, wr, wi, &m, &s[0],
           &sep[0], work, 1, iwork, 1, &info);
    chkxer("DTRSEN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 15;
    dtrsen("N", "V", sel, 2, a, 2, b, 2, wr, wi, &m, &s[0],
           &sep[0], work, 0, iwork, 1, &info);
    chkxer("DTRSEN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 15;
    dtrsen("E", "V", sel, 3, a, 3, b, 3, wr, wi, &m, &s[0],
           &sep[0], work, 1, iwork, 1, &info);
    chkxer("DTRSEN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 15;
    dtrsen("V", "V", sel, 3, a, 3, b, 3, wr, wi, &m, &s[0],
           &sep[0], work, 3, iwork, 2, &info);
    chkxer("DTRSEN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 17;
    dtrsen("E", "V", sel, 2, a, 2, b, 2, wr, wi, &m, &s[0],
           &sep[0], work, 1, iwork, 0, &info);
    chkxer("DTRSEN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 17;
    dtrsen("V", "V", sel, 3, a, 3, b, 3, wr, wi, &m, &s[0],
           &sep[0], work, 4, iwork, 1, &info);
    chkxer("DTRSEN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    *nt = *nt + 10;

    *ok = xerbla_ok;

    /* Reset error-testing mode so subsequent tests use production xerbla behavior */
    xerbla_srnamt[0] = '\0';
    xerbla_infot = 0;
}
