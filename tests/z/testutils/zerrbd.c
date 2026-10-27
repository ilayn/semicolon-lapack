/**
 * @file zerrbd.c
 * @brief ZERRBD tests the error exits for the SVD routines.
 *
 * Port of LAPACK TESTING/EIG/zerrbd.f
 * Tests error exits for: ZGEBRD, ZGEBD2, ZUNGBR, ZUNMBR, ZBDSQR
 */

#include <complex.h>
#include <math.h>
#include <string.h>
#include "semicolon_lapack_complex_double.h"
#include "verify.h"

void zerrbd(INT* ok, INT* nt)
{
    enum { NMAX = 4, LW = NMAX };
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    INT i, j, info;
    f64 rnan;

    f64 d[NMAX], e[NMAX], rw[4 * NMAX];
    c128 a[NMAX * NMAX], tp[NMAX], tq[NMAX], u[NMAX * NMAX], v[NMAX * NMAX], w[LW];

    /* Set the variables to innocuous values. */
    for (j = 0; j < NMAX; j++) {
        for (i = 0; i < NMAX; i++) {
            a[i + j * NMAX] = CMPLX(ONE / (f64)(i + j + 2), ZERO);
        }
    }
    xerbla_ok = 1;
    *nt = 0;

    /* Test error exits of the SVD routines. */

    /* ZGEBRD */

    strcpy(xerbla_srnamt, "ZGEBRD");
    xerbla_infot = 1;
    zgebrd(-1, 0, a, 1, d, e, tq, tp, w, 1, &info);
    chkxer("ZGEBRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 2;
    zgebrd(0, -1, a, 1, d, e, tq, tp, w, 1, &info);
    chkxer("ZGEBRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 4;
    zgebrd(2, 1, a, 1, d, e, tq, tp, w, 2, &info);
    chkxer("ZGEBRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 10;
    zgebrd(2, 1, a, 2, d, e, tq, tp, w, 1, &info);
    chkxer("ZGEBRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    *nt = *nt + 4;

    /* ZGEBD2 */

    strcpy(xerbla_srnamt, "ZGEBD2");
    xerbla_infot = 1;
    zgebd2(-1, 0, a, 1, d, e, tq, tp, w, &info);
    chkxer("ZGEBD2", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 2;
    zgebd2(0, -1, a, 1, d, e, tq, tp, w, &info);
    chkxer("ZGEBD2", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 4;
    zgebd2(2, 1, a, 1, d, e, tq, tp, w, &info);
    chkxer("ZGEBD2", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    *nt = *nt + 3;

    /* ZUNGBR */

    strcpy(xerbla_srnamt, "ZUNGBR");
    xerbla_infot = 1;
    zungbr("/", 0, 0, 0, a, 1, tq, w, 1, &info);
    chkxer("ZUNGBR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 2;
    zungbr("Q", -1, 0, 0, a, 1, tq, w, 1, &info);
    chkxer("ZUNGBR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 3;
    zungbr("Q", 0, -1, 0, a, 1, tq, w, 1, &info);
    chkxer("ZUNGBR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 3;
    zungbr("Q", 0, 1, 0, a, 1, tq, w, 1, &info);
    chkxer("ZUNGBR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 3;
    zungbr("Q", 1, 0, 1, a, 1, tq, w, 1, &info);
    chkxer("ZUNGBR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 3;
    zungbr("P", 1, 0, 0, a, 1, tq, w, 1, &info);
    chkxer("ZUNGBR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 3;
    zungbr("P", 0, 1, 1, a, 1, tq, w, 1, &info);
    chkxer("ZUNGBR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 4;
    zungbr("Q", 0, 0, -1, a, 1, tq, w, 1, &info);
    chkxer("ZUNGBR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 6;
    zungbr("Q", 2, 1, 1, a, 1, tq, w, 1, &info);
    chkxer("ZUNGBR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 9;
    zungbr("Q", 2, 2, 1, a, 2, tq, w, 1, &info);
    chkxer("ZUNGBR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    *nt = *nt + 10;

    /* ZUNMBR */

    strcpy(xerbla_srnamt, "ZUNMBR");
    xerbla_infot = 1;
    zunmbr("/", "L", "T", 0, 0, 0, a, 1, tq, u, 1, w, 1, &info);
    chkxer("ZUNMBR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 2;
    zunmbr("Q", "/", "T", 0, 0, 0, a, 1, tq, u, 1, w, 1, &info);
    chkxer("ZUNMBR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 3;
    zunmbr("Q", "L", "/", 0, 0, 0, a, 1, tq, u, 1, w, 1, &info);
    chkxer("ZUNMBR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 4;
    zunmbr("Q", "L", "C", -1, 0, 0, a, 1, tq, u, 1, w, 1, &info);
    chkxer("ZUNMBR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 5;
    zunmbr("Q", "L", "C", 0, -1, 0, a, 1, tq, u, 1, w, 1, &info);
    chkxer("ZUNMBR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 6;
    zunmbr("Q", "L", "C", 0, 0, -1, a, 1, tq, u, 1, w, 1, &info);
    chkxer("ZUNMBR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 8;
    zunmbr("Q", "L", "C", 2, 0, 0, a, 1, tq, u, 2, w, 1, &info);
    chkxer("ZUNMBR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 8;
    zunmbr("Q", "R", "C", 0, 2, 0, a, 1, tq, u, 1, w, 1, &info);
    chkxer("ZUNMBR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 8;
    zunmbr("P", "L", "C", 2, 0, 2, a, 1, tq, u, 2, w, 1, &info);
    chkxer("ZUNMBR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 8;
    zunmbr("P", "R", "C", 0, 2, 2, a, 1, tq, u, 1, w, 1, &info);
    chkxer("ZUNMBR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 11;
    zunmbr("Q", "R", "C", 2, 0, 0, a, 1, tq, u, 1, w, 1, &info);
    chkxer("ZUNMBR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 13;
    zunmbr("Q", "L", "C", 0, 2, 0, a, 1, tq, u, 1, w, 0, &info);
    chkxer("ZUNMBR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 13;
    zunmbr("Q", "R", "C", 2, 0, 0, a, 1, tq, u, 2, w, 0, &info);
    chkxer("ZUNMBR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    *nt = *nt + 13;

    /* ZBDSQR */

    strcpy(xerbla_srnamt, "ZBDSQR");
    xerbla_infot = 1;
    zbdsqr("/", 0, 0, 0, 0, d, e, v, 1, u, 1, a, 1, rw, &info);
    chkxer("ZBDSQR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 2;
    zbdsqr("U", -1, 0, 0, 0, d, e, v, 1, u, 1, a, 1, rw, &info);
    chkxer("ZBDSQR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 3;
    zbdsqr("U", 0, -1, 0, 0, d, e, v, 1, u, 1, a, 1, rw, &info);
    chkxer("ZBDSQR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 4;
    zbdsqr("U", 0, 0, -1, 0, d, e, v, 1, u, 1, a, 1, rw, &info);
    chkxer("ZBDSQR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 5;
    zbdsqr("U", 0, 0, 0, -1, d, e, v, 1, u, 1, a, 1, rw, &info);
    chkxer("ZBDSQR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 9;
    zbdsqr("U", 2, 1, 0, 0, d, e, v, 1, u, 1, a, 1, rw, &info);
    chkxer("ZBDSQR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 11;
    zbdsqr("U", 0, 0, 2, 0, d, e, v, 1, u, 1, a, 1, rw, &info);
    chkxer("ZBDSQR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 13;
    zbdsqr("U", 2, 0, 0, 1, d, e, v, 1, u, 1, a, 1, rw, &info);
    chkxer("ZBDSQR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    *nt = *nt + 8;

    /* ZBDSQR without singular vectors must return when D contains a
     * NaN, which the dqds path would otherwise pass to DLASCL as a
     * scaling factor, instead of stopping in XERBLA. */

    rnan = (f64)NAN;
    for (j = 0; j < NMAX; j++) {
        d[j] = (f64)(j + 1);
        e[j] = ONE / (f64)(j + 2);
    }
    d[NMAX - 1] = rnan;
    e[NMAX - 1] = ZERO;
    strcpy(xerbla_srnamt, "ZBDSQR");
    xerbla_infot = 0;
    xerbla_lerr = 0;
    zbdsqr("U", NMAX, 0, 0, 0, d, e, v, 1, u, 1, a, 1, rw, &info);
    if (xerbla_lerr) {
        xerbla_ok = 0;
    }
    *nt = *nt + 1;

    *ok = xerbla_ok;

    xerbla_srnamt[0] = '\0';
    xerbla_infot = 0;
}
