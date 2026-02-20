/**
 * @file serrec.c
 * @brief SERREC tests the error exits for eigencondition estimation routines.
 *
 * Port of LAPACK TESTING/EIG/serrec.f
 * Tests error exits for: STRSYL, STRSYL3, STREXC, STRSNA, STRSEN
 */

#include <string.h>
#include "semicolon_lapack_single.h"
#include "verify.h"

void serrec(int* ok, int* nt)
{
    const int NMAX = 4;
    const f32 ONE = 1.0f;
    const f32 ZERO = 0.0f;

    int i, j, ifst, ilst, info, m;
    f32 scale;

    int sel[NMAX];
    int iwork[NMAX];
    f32 a[NMAX * NMAX], b[NMAX * NMAX], c[NMAX * NMAX];
    f32 s[NMAX], sep[NMAX], wi[NMAX], work[NMAX], wr[NMAX];

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

    /* Test STRSYL */

    strcpy(xerbla_srnamt, "STRSYL");
    xerbla_infot = 1;
    strsyl("X", "N", 1, 0, 0, a, 1, b, 1, c, 1, &scale, &info);
    chkxer("STRSYL", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 2;
    strsyl("N", "X", 1, 0, 0, a, 1, b, 1, c, 1, &scale, &info);
    chkxer("STRSYL", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 3;
    strsyl("N", "N", 0, 0, 0, a, 1, b, 1, c, 1, &scale, &info);
    chkxer("STRSYL", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 4;
    strsyl("N", "N", 1, -1, 0, a, 1, b, 1, c, 1, &scale, &info);
    chkxer("STRSYL", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 5;
    strsyl("N", "N", 1, 0, -1, a, 1, b, 1, c, 1, &scale, &info);
    chkxer("STRSYL", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 7;
    strsyl("N", "N", 1, 2, 0, a, 1, b, 1, c, 2, &scale, &info);
    chkxer("STRSYL", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 9;
    strsyl("N", "N", 1, 0, 2, a, 1, b, 1, c, 1, &scale, &info);
    chkxer("STRSYL", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 11;
    strsyl("N", "N", 1, 2, 0, a, 2, b, 1, c, 1, &scale, &info);
    chkxer("STRSYL", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    *nt = *nt + 8;

    /* Test STRSYL3 */

    strcpy(xerbla_srnamt, "STRSYL3");
    xerbla_infot = 1;
    strsyl3("X", "N", 1, 0, 0, a, 1, b, 1, c, 1, &scale,
            iwork, NMAX, (f32*)work, NMAX, &info);
    chkxer("STRSYL3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 2;
    strsyl3("N", "X", 1, 0, 0, a, 1, b, 1, c, 1, &scale,
            iwork, NMAX, (f32*)work, NMAX, &info);
    chkxer("STRSYL3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 3;
    strsyl3("N", "N", 0, 0, 0, a, 1, b, 1, c, 1, &scale,
            iwork, NMAX, (f32*)work, NMAX, &info);
    chkxer("STRSYL3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 4;
    strsyl3("N", "N", 1, -1, 0, a, 1, b, 1, c, 1, &scale,
            iwork, NMAX, (f32*)work, NMAX, &info);
    chkxer("STRSYL3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 5;
    strsyl3("N", "N", 1, 0, -1, a, 1, b, 1, c, 1, &scale,
            iwork, NMAX, (f32*)work, NMAX, &info);
    chkxer("STRSYL3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 7;
    strsyl3("N", "N", 1, 2, 0, a, 1, b, 1, c, 2, &scale,
            iwork, NMAX, (f32*)work, NMAX, &info);
    chkxer("STRSYL3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 9;
    strsyl3("N", "N", 1, 0, 2, a, 1, b, 1, c, 1, &scale,
            iwork, NMAX, (f32*)work, NMAX, &info);
    chkxer("STRSYL3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 11;
    strsyl3("N", "N", 1, 2, 0, a, 2, b, 1, c, 1, &scale,
            iwork, NMAX, (f32*)work, NMAX, &info);
    chkxer("STRSYL3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    *nt = *nt + 8;

    /* Test STREXC */

    strcpy(xerbla_srnamt, "STREXC");
    ifst = 0;
    ilst = 0;
    xerbla_infot = 1;
    strexc("X", 1, a, 1, b, 1, &ifst, &ilst, work, &info);
    chkxer("STREXC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 2;
    strexc("N", -1, a, 1, b, 1, &ifst, &ilst, work, &info);
    chkxer("STREXC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 4;
    ilst = 1;
    strexc("N", 2, a, 1, b, 1, &ifst, &ilst, work, &info);
    chkxer("STREXC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 6;
    strexc("V", 2, a, 2, b, 1, &ifst, &ilst, work, &info);
    chkxer("STREXC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 7;
    ifst = -1;
    ilst = 0;
    strexc("V", 1, a, 1, b, 1, &ifst, &ilst, work, &info);
    chkxer("STREXC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 7;
    ifst = 1;
    strexc("V", 1, a, 1, b, 1, &ifst, &ilst, work, &info);
    chkxer("STREXC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 8;
    ifst = 0;
    ilst = -1;
    strexc("V", 1, a, 1, b, 1, &ifst, &ilst, work, &info);
    chkxer("STREXC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 8;
    ilst = 1;
    strexc("V", 1, a, 1, b, 1, &ifst, &ilst, work, &info);
    chkxer("STREXC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    *nt = *nt + 8;

    /* Test STRSNA */

    strcpy(xerbla_srnamt, "STRSNA");
    xerbla_infot = 1;
    strsna("X", "A", sel, 0, a, 1, b, 1, c, 1, s, sep, 1, &m,
           work, 1, iwork, &info);
    chkxer("STRSNA", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 2;
    strsna("B", "X", sel, 0, a, 1, b, 1, c, 1, s, sep, 1, &m,
           work, 1, iwork, &info);
    chkxer("STRSNA", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 4;
    strsna("B", "A", sel, -1, a, 1, b, 1, c, 1, s, sep, 1, &m,
           work, 1, iwork, &info);
    chkxer("STRSNA", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 6;
    strsna("V", "A", sel, 2, a, 1, b, 1, c, 1, s, sep, 2, &m,
           work, 2, iwork, &info);
    chkxer("STRSNA", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 8;
    strsna("B", "A", sel, 2, a, 2, b, 1, c, 2, s, sep, 2, &m,
           work, 2, iwork, &info);
    chkxer("STRSNA", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 10;
    strsna("B", "A", sel, 2, a, 2, b, 2, c, 1, s, sep, 2, &m,
           work, 2, iwork, &info);
    chkxer("STRSNA", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 13;
    strsna("B", "A", sel, 1, a, 1, b, 1, c, 1, s, sep, 0, &m,
           work, 1, iwork, &info);
    chkxer("STRSNA", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 13;
    strsna("B", "S", sel, 2, a, 2, b, 2, c, 2, s, sep, 1, &m,
           work, 2, iwork, &info);
    chkxer("STRSNA", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 16;
    strsna("B", "A", sel, 2, a, 2, b, 2, c, 2, s, sep, 2, &m,
           work, 1, iwork, &info);
    chkxer("STRSNA", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    *nt = *nt + 9;

    /* Test STRSEN */

    sel[0] = 0;
    strcpy(xerbla_srnamt, "STRSEN");
    xerbla_infot = 1;
    strsen("X", "N", sel, 0, a, 1, b, 1, wr, wi, &m, &s[0],
           &sep[0], work, 1, iwork, 1, &info);
    chkxer("STRSEN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 2;
    strsen("N", "X", sel, 0, a, 1, b, 1, wr, wi, &m, &s[0],
           &sep[0], work, 1, iwork, 1, &info);
    chkxer("STRSEN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 4;
    strsen("N", "N", sel, -1, a, 1, b, 1, wr, wi, &m, &s[0],
           &sep[0], work, 1, iwork, 1, &info);
    chkxer("STRSEN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 6;
    strsen("N", "N", sel, 2, a, 1, b, 1, wr, wi, &m, &s[0],
           &sep[0], work, 2, iwork, 1, &info);
    chkxer("STRSEN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 8;
    strsen("N", "V", sel, 2, a, 2, b, 1, wr, wi, &m, &s[0],
           &sep[0], work, 1, iwork, 1, &info);
    chkxer("STRSEN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 15;
    strsen("N", "V", sel, 2, a, 2, b, 2, wr, wi, &m, &s[0],
           &sep[0], work, 0, iwork, 1, &info);
    chkxer("STRSEN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 15;
    strsen("E", "V", sel, 3, a, 3, b, 3, wr, wi, &m, &s[0],
           &sep[0], work, 1, iwork, 1, &info);
    chkxer("STRSEN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 15;
    strsen("V", "V", sel, 3, a, 3, b, 3, wr, wi, &m, &s[0],
           &sep[0], work, 3, iwork, 2, &info);
    chkxer("STRSEN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 17;
    strsen("E", "V", sel, 2, a, 2, b, 2, wr, wi, &m, &s[0],
           &sep[0], work, 1, iwork, 0, &info);
    chkxer("STRSEN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 17;
    strsen("V", "V", sel, 3, a, 3, b, 3, wr, wi, &m, &s[0],
           &sep[0], work, 4, iwork, 1, &info);
    chkxer("STRSEN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    *nt = *nt + 10;

    *ok = xerbla_ok;

    /* Reset error-testing mode so subsequent tests use production xerbla behavior */
    xerbla_srnamt[0] = '\0';
    xerbla_infot = 0;
}
