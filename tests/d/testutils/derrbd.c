/**
 * @file derrbd.c
 * @brief DERRBD tests the error exits for the SVD routines.
 *
 * Port of LAPACK TESTING/EIG/derrbd.f
 * Tests error exits for: DGEBRD, DGEBD2, DORGBR, DORMBR, DBDSQR, DBDSDC,
 * DBDSVDX
 */

#include <math.h>
#include <string.h>
#include "semicolon_lapack_double.h"
#include "verify.h"

void derrbd(INT* ok, INT* nt)
{
    enum { NMAX = 4, LW = 4 * NMAX };
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    INT i, j, info, ns;
    f64 rnan;

    INT iq[NMAX * NMAX], iw[NMAX];
    f64 a[NMAX * NMAX], d[NMAX], e[NMAX], q[NMAX * NMAX], s[NMAX], tp[NMAX];
    f64 tq[NMAX], u[NMAX * NMAX], v[NMAX * NMAX], w[LW];

    /* Set the variables to innocuous values. */
    for (j = 0; j < NMAX; j++) {
        for (i = 0; i < NMAX; i++) {
            a[i + j * NMAX] = ONE / (f64)(i + j + 2);
        }
    }
    xerbla_ok = 1;
    *nt = 0;

    /* Test error exits of the SVD routines. */

    /* DGEBRD */

    strcpy(xerbla_srnamt, "DGEBRD");
    xerbla_infot = 1;
    dgebrd(-1, 0, a, 1, d, e, tq, tp, w, 1, &info);
    chkxer("DGEBRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 2;
    dgebrd(0, -1, a, 1, d, e, tq, tp, w, 1, &info);
    chkxer("DGEBRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 4;
    dgebrd(2, 1, a, 1, d, e, tq, tp, w, 2, &info);
    chkxer("DGEBRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 10;
    dgebrd(2, 1, a, 2, d, e, tq, tp, w, 1, &info);
    chkxer("DGEBRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    *nt = *nt + 4;

    /* DGEBD2 */

    strcpy(xerbla_srnamt, "DGEBD2");
    xerbla_infot = 1;
    dgebd2(-1, 0, a, 1, d, e, tq, tp, w, &info);
    chkxer("DGEBD2", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 2;
    dgebd2(0, -1, a, 1, d, e, tq, tp, w, &info);
    chkxer("DGEBD2", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 4;
    dgebd2(2, 1, a, 1, d, e, tq, tp, w, &info);
    chkxer("DGEBD2", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    *nt = *nt + 3;

    /* DORGBR */

    strcpy(xerbla_srnamt, "DORGBR");
    xerbla_infot = 1;
    dorgbr("/", 0, 0, 0, a, 1, tq, w, 1, &info);
    chkxer("DORGBR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 2;
    dorgbr("Q", -1, 0, 0, a, 1, tq, w, 1, &info);
    chkxer("DORGBR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 3;
    dorgbr("Q", 0, -1, 0, a, 1, tq, w, 1, &info);
    chkxer("DORGBR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 3;
    dorgbr("Q", 0, 1, 0, a, 1, tq, w, 1, &info);
    chkxer("DORGBR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 3;
    dorgbr("Q", 1, 0, 1, a, 1, tq, w, 1, &info);
    chkxer("DORGBR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 3;
    dorgbr("P", 1, 0, 0, a, 1, tq, w, 1, &info);
    chkxer("DORGBR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 3;
    dorgbr("P", 0, 1, 1, a, 1, tq, w, 1, &info);
    chkxer("DORGBR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 4;
    dorgbr("Q", 0, 0, -1, a, 1, tq, w, 1, &info);
    chkxer("DORGBR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 6;
    dorgbr("Q", 2, 1, 1, a, 1, tq, w, 1, &info);
    chkxer("DORGBR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 9;
    dorgbr("Q", 2, 2, 1, a, 2, tq, w, 1, &info);
    chkxer("DORGBR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    *nt = *nt + 10;

    /* DORMBR */

    strcpy(xerbla_srnamt, "DORMBR");
    xerbla_infot = 1;
    dormbr("/", "L", "T", 0, 0, 0, a, 1, tq, u, 1, w, 1, &info);
    chkxer("DORMBR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 2;
    dormbr("Q", "/", "T", 0, 0, 0, a, 1, tq, u, 1, w, 1, &info);
    chkxer("DORMBR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 3;
    dormbr("Q", "L", "/", 0, 0, 0, a, 1, tq, u, 1, w, 1, &info);
    chkxer("DORMBR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 4;
    dormbr("Q", "L", "T", -1, 0, 0, a, 1, tq, u, 1, w, 1, &info);
    chkxer("DORMBR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 5;
    dormbr("Q", "L", "T", 0, -1, 0, a, 1, tq, u, 1, w, 1, &info);
    chkxer("DORMBR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 6;
    dormbr("Q", "L", "T", 0, 0, -1, a, 1, tq, u, 1, w, 1, &info);
    chkxer("DORMBR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 8;
    dormbr("Q", "L", "T", 2, 0, 0, a, 1, tq, u, 2, w, 1, &info);
    chkxer("DORMBR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 8;
    dormbr("Q", "R", "T", 0, 2, 0, a, 1, tq, u, 1, w, 1, &info);
    chkxer("DORMBR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 8;
    dormbr("P", "L", "T", 2, 0, 2, a, 1, tq, u, 2, w, 1, &info);
    chkxer("DORMBR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 8;
    dormbr("P", "R", "T", 0, 2, 2, a, 1, tq, u, 1, w, 1, &info);
    chkxer("DORMBR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 11;
    dormbr("Q", "R", "T", 2, 0, 0, a, 1, tq, u, 1, w, 1, &info);
    chkxer("DORMBR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 13;
    dormbr("Q", "L", "T", 0, 2, 0, a, 1, tq, u, 1, w, 1, &info);
    chkxer("DORMBR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 13;
    dormbr("Q", "R", "T", 2, 0, 0, a, 1, tq, u, 2, w, 1, &info);
    chkxer("DORMBR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    *nt = *nt + 13;

    /* DBDSQR */

    strcpy(xerbla_srnamt, "DBDSQR");
    xerbla_infot = 1;
    dbdsqr("/", 0, 0, 0, 0, d, e, v, 1, u, 1, a, 1, w, &info);
    chkxer("DBDSQR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 2;
    dbdsqr("U", -1, 0, 0, 0, d, e, v, 1, u, 1, a, 1, w, &info);
    chkxer("DBDSQR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 3;
    dbdsqr("U", 0, -1, 0, 0, d, e, v, 1, u, 1, a, 1, w, &info);
    chkxer("DBDSQR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 4;
    dbdsqr("U", 0, 0, -1, 0, d, e, v, 1, u, 1, a, 1, w, &info);
    chkxer("DBDSQR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 5;
    dbdsqr("U", 0, 0, 0, -1, d, e, v, 1, u, 1, a, 1, w, &info);
    chkxer("DBDSQR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 9;
    dbdsqr("U", 2, 1, 0, 0, d, e, v, 1, u, 1, a, 1, w, &info);
    chkxer("DBDSQR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 11;
    dbdsqr("U", 0, 0, 2, 0, d, e, v, 1, u, 1, a, 1, w, &info);
    chkxer("DBDSQR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 13;
    dbdsqr("U", 2, 0, 0, 1, d, e, v, 1, u, 1, a, 1, w, &info);
    chkxer("DBDSQR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    *nt = *nt + 8;

    /* DBDSQR without singular vectors must return when D contains a
     * NaN, which the dqds path would otherwise pass to DLASCL as a
     * scaling factor, instead of stopping in XERBLA. */

    rnan = (f64)NAN;
    for (j = 0; j < NMAX; j++) {
        d[j] = (f64)(j + 1);
        e[j] = ONE / (f64)(j + 2);
    }
    d[NMAX - 1] = rnan;
    e[NMAX - 1] = ZERO;
    strcpy(xerbla_srnamt, "DBDSQR");
    xerbla_infot = 0;
    xerbla_lerr = 0;
    dbdsqr("U", NMAX, 0, 0, 0, d, e, v, 1, u, 1, a, 1, w, &info);
    if (xerbla_lerr) {
        xerbla_ok = 0;
    }
    *nt = *nt + 1;

    /* DBDSDC */

    strcpy(xerbla_srnamt, "DBDSDC");
    xerbla_infot = 1;
    dbdsdc("/", "N", 0, d, e, u, 1, v, 1, q, iq, w, iw, &info);
    chkxer("DBDSDC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 2;
    dbdsdc("U", "/", 0, d, e, u, 1, v, 1, q, iq, w, iw, &info);
    chkxer("DBDSDC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 3;
    dbdsdc("U", "N", -1, d, e, u, 1, v, 1, q, iq, w, iw, &info);
    chkxer("DBDSDC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 7;
    dbdsdc("U", "I", 2, d, e, u, 1, v, 1, q, iq, w, iw, &info);
    chkxer("DBDSDC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 9;
    dbdsdc("U", "I", 2, d, e, u, 2, v, 1, q, iq, w, iw, &info);
    chkxer("DBDSDC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    *nt = *nt + 5;

    /* DBDSVDX */

    strcpy(xerbla_srnamt, "DBDSVDX");
    xerbla_infot = 1;
    dbdsvdx("X", "N", "A", 1, d, e, ZERO, ONE, 0, 0,
            &ns, s, q, 1, w, iw, &info);
    chkxer("DBDSVDX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 2;
    dbdsvdx("U", "X", "A", 1, d, e, ZERO, ONE, 0, 0,
            &ns, s, q, 1, w, iw, &info);
    chkxer("DBDSVDX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 3;
    dbdsvdx("U", "V", "X", 1, d, e, ZERO, ONE, 0, 0,
            &ns, s, q, 1, w, iw, &info);
    chkxer("DBDSVDX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 4;
    dbdsvdx("U", "V", "A", -1, d, e, ZERO, ONE, 0, 0,
            &ns, s, q, 1, w, iw, &info);
    chkxer("DBDSVDX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 7;
    dbdsvdx("U", "V", "V", 2, d, e, -ONE, ZERO, 0, 0,
            &ns, s, q, 1, w, iw, &info);
    chkxer("DBDSVDX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 8;
    dbdsvdx("U", "V", "V", 2, d, e, ONE, ZERO, 0, 0,
            &ns, s, q, 1, w, iw, &info);
    chkxer("DBDSVDX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 9;
    dbdsvdx("L", "V", "I", 2, d, e, ZERO, ZERO, -1, 1,
            &ns, s, q, 1, w, iw, &info);
    chkxer("DBDSVDX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 9;
    dbdsvdx("L", "V", "I", 4, d, e, ZERO, ZERO, 4, 1,
            &ns, s, q, 1, w, iw, &info);
    chkxer("DBDSVDX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 10;
    dbdsvdx("L", "V", "I", 4, d, e, ZERO, ZERO, 2, 1,
            &ns, s, q, 1, w, iw, &info);
    chkxer("DBDSVDX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 10;
    dbdsvdx("L", "V", "I", 4, d, e, ZERO, ZERO, 2, 4,
            &ns, s, q, 1, w, iw, &info);
    chkxer("DBDSVDX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 14;
    dbdsvdx("L", "V", "A", 4, d, e, ZERO, ZERO, 0, 0,
            &ns, s, q, 0, w, iw, &info);
    chkxer("DBDSVDX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 14;
    dbdsvdx("L", "V", "A", 4, d, e, ZERO, ZERO, 0, 0,
            &ns, s, q, 2, w, iw, &info);
    chkxer("DBDSVDX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    *nt = *nt + 12;

    *ok = xerbla_ok;

    xerbla_srnamt[0] = '\0';
    xerbla_infot = 0;
}
