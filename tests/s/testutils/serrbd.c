/**
 * @file serrbd.c
 * @brief SERRBD tests the error exits for the SVD routines.
 *
 * Port of LAPACK TESTING/EIG/serrbd.f
 * Tests error exits for: SGEBRD, SGEBD2, SORGBR, SORMBR, SBDSQR, SBDSDC,
 * SBDSVDX
 */

#include <math.h>
#include <string.h>
#include "semicolon_lapack_single.h"
#include "verify.h"

void serrbd(INT* ok, INT* nt)
{
    enum { NMAX = 4, LW = 4 * NMAX };
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;

    INT i, j, info, ns;
    f32 rnan;

    INT iq[NMAX * NMAX], iw[NMAX];
    f32 a[NMAX * NMAX], d[NMAX], e[NMAX], q[NMAX * NMAX], s[NMAX], tp[NMAX];
    f32 tq[NMAX], u[NMAX * NMAX], v[NMAX * NMAX], w[LW];

    /* Set the variables to innocuous values. */
    for (j = 0; j < NMAX; j++) {
        for (i = 0; i < NMAX; i++) {
            a[i + j * NMAX] = ONE / (f32)(i + j + 2);
        }
    }
    xerbla_ok = 1;
    *nt = 0;

    /* Test error exits of the SVD routines. */

    /* SGEBRD */

    strcpy(xerbla_srnamt, "SGEBRD");
    xerbla_infot = 1;
    sgebrd(-1, 0, a, 1, d, e, tq, tp, w, 1, &info);
    chkxer("SGEBRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 2;
    sgebrd(0, -1, a, 1, d, e, tq, tp, w, 1, &info);
    chkxer("SGEBRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 4;
    sgebrd(2, 1, a, 1, d, e, tq, tp, w, 2, &info);
    chkxer("SGEBRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 10;
    sgebrd(2, 1, a, 2, d, e, tq, tp, w, 1, &info);
    chkxer("SGEBRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    *nt = *nt + 4;

    /* SGEBD2 */

    strcpy(xerbla_srnamt, "SGEBD2");
    xerbla_infot = 1;
    sgebd2(-1, 0, a, 1, d, e, tq, tp, w, &info);
    chkxer("SGEBD2", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 2;
    sgebd2(0, -1, a, 1, d, e, tq, tp, w, &info);
    chkxer("SGEBD2", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 4;
    sgebd2(2, 1, a, 1, d, e, tq, tp, w, &info);
    chkxer("SGEBD2", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    *nt = *nt + 3;

    /* SORGBR */

    strcpy(xerbla_srnamt, "SORGBR");
    xerbla_infot = 1;
    sorgbr("/", 0, 0, 0, a, 1, tq, w, 1, &info);
    chkxer("SORGBR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 2;
    sorgbr("Q", -1, 0, 0, a, 1, tq, w, 1, &info);
    chkxer("SORGBR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 3;
    sorgbr("Q", 0, -1, 0, a, 1, tq, w, 1, &info);
    chkxer("SORGBR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 3;
    sorgbr("Q", 0, 1, 0, a, 1, tq, w, 1, &info);
    chkxer("SORGBR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 3;
    sorgbr("Q", 1, 0, 1, a, 1, tq, w, 1, &info);
    chkxer("SORGBR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 3;
    sorgbr("P", 1, 0, 0, a, 1, tq, w, 1, &info);
    chkxer("SORGBR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 3;
    sorgbr("P", 0, 1, 1, a, 1, tq, w, 1, &info);
    chkxer("SORGBR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 4;
    sorgbr("Q", 0, 0, -1, a, 1, tq, w, 1, &info);
    chkxer("SORGBR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 6;
    sorgbr("Q", 2, 1, 1, a, 1, tq, w, 1, &info);
    chkxer("SORGBR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 9;
    sorgbr("Q", 2, 2, 1, a, 2, tq, w, 1, &info);
    chkxer("SORGBR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    *nt = *nt + 10;

    /* SORMBR */

    strcpy(xerbla_srnamt, "SORMBR");
    xerbla_infot = 1;
    sormbr("/", "L", "T", 0, 0, 0, a, 1, tq, u, 1, w, 1, &info);
    chkxer("SORMBR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 2;
    sormbr("Q", "/", "T", 0, 0, 0, a, 1, tq, u, 1, w, 1, &info);
    chkxer("SORMBR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 3;
    sormbr("Q", "L", "/", 0, 0, 0, a, 1, tq, u, 1, w, 1, &info);
    chkxer("SORMBR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 4;
    sormbr("Q", "L", "T", -1, 0, 0, a, 1, tq, u, 1, w, 1, &info);
    chkxer("SORMBR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 5;
    sormbr("Q", "L", "T", 0, -1, 0, a, 1, tq, u, 1, w, 1, &info);
    chkxer("SORMBR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 6;
    sormbr("Q", "L", "T", 0, 0, -1, a, 1, tq, u, 1, w, 1, &info);
    chkxer("SORMBR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 8;
    sormbr("Q", "L", "T", 2, 0, 0, a, 1, tq, u, 2, w, 1, &info);
    chkxer("SORMBR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 8;
    sormbr("Q", "R", "T", 0, 2, 0, a, 1, tq, u, 1, w, 1, &info);
    chkxer("SORMBR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 8;
    sormbr("P", "L", "T", 2, 0, 2, a, 1, tq, u, 2, w, 1, &info);
    chkxer("SORMBR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 8;
    sormbr("P", "R", "T", 0, 2, 2, a, 1, tq, u, 1, w, 1, &info);
    chkxer("SORMBR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 11;
    sormbr("Q", "R", "T", 2, 0, 0, a, 1, tq, u, 1, w, 1, &info);
    chkxer("SORMBR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 13;
    sormbr("Q", "L", "T", 0, 2, 0, a, 1, tq, u, 1, w, 1, &info);
    chkxer("SORMBR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 13;
    sormbr("Q", "R", "T", 2, 0, 0, a, 1, tq, u, 2, w, 1, &info);
    chkxer("SORMBR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    *nt = *nt + 13;

    /* SBDSQR */

    strcpy(xerbla_srnamt, "SBDSQR");
    xerbla_infot = 1;
    sbdsqr("/", 0, 0, 0, 0, d, e, v, 1, u, 1, a, 1, w, &info);
    chkxer("SBDSQR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 2;
    sbdsqr("U", -1, 0, 0, 0, d, e, v, 1, u, 1, a, 1, w, &info);
    chkxer("SBDSQR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 3;
    sbdsqr("U", 0, -1, 0, 0, d, e, v, 1, u, 1, a, 1, w, &info);
    chkxer("SBDSQR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 4;
    sbdsqr("U", 0, 0, -1, 0, d, e, v, 1, u, 1, a, 1, w, &info);
    chkxer("SBDSQR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 5;
    sbdsqr("U", 0, 0, 0, -1, d, e, v, 1, u, 1, a, 1, w, &info);
    chkxer("SBDSQR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 9;
    sbdsqr("U", 2, 1, 0, 0, d, e, v, 1, u, 1, a, 1, w, &info);
    chkxer("SBDSQR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 11;
    sbdsqr("U", 0, 0, 2, 0, d, e, v, 1, u, 1, a, 1, w, &info);
    chkxer("SBDSQR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 13;
    sbdsqr("U", 2, 0, 0, 1, d, e, v, 1, u, 1, a, 1, w, &info);
    chkxer("SBDSQR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    *nt = *nt + 8;

    /* SBDSQR without singular vectors must return when D contains a
     * NaN, which the dqds path would otherwise pass to SLASCL as a
     * scaling factor, instead of stopping in XERBLA. */

    rnan = (f32)NAN;
    for (j = 0; j < NMAX; j++) {
        d[j] = (f32)(j + 1);
        e[j] = ONE / (f32)(j + 2);
    }
    d[NMAX - 1] = rnan;
    e[NMAX - 1] = ZERO;
    strcpy(xerbla_srnamt, "SBDSQR");
    xerbla_infot = 0;
    xerbla_lerr = 0;
    sbdsqr("U", NMAX, 0, 0, 0, d, e, v, 1, u, 1, a, 1, w, &info);
    if (xerbla_lerr) {
        xerbla_ok = 0;
    }
    *nt = *nt + 1;

    /* SBDSDC */

    strcpy(xerbla_srnamt, "SBDSDC");
    xerbla_infot = 1;
    sbdsdc("/", "N", 0, d, e, u, 1, v, 1, q, iq, w, iw, &info);
    chkxer("SBDSDC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 2;
    sbdsdc("U", "/", 0, d, e, u, 1, v, 1, q, iq, w, iw, &info);
    chkxer("SBDSDC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 3;
    sbdsdc("U", "N", -1, d, e, u, 1, v, 1, q, iq, w, iw, &info);
    chkxer("SBDSDC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 7;
    sbdsdc("U", "I", 2, d, e, u, 1, v, 1, q, iq, w, iw, &info);
    chkxer("SBDSDC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 9;
    sbdsdc("U", "I", 2, d, e, u, 2, v, 1, q, iq, w, iw, &info);
    chkxer("SBDSDC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    *nt = *nt + 5;

    /* SBDSVDX */

    strcpy(xerbla_srnamt, "SBDSVDX");
    xerbla_infot = 1;
    sbdsvdx("X", "N", "A", 1, d, e, ZERO, ONE, 0, 0,
            &ns, s, q, 1, w, iw, &info);
    chkxer("SBDSVDX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 2;
    sbdsvdx("U", "X", "A", 1, d, e, ZERO, ONE, 0, 0,
            &ns, s, q, 1, w, iw, &info);
    chkxer("SBDSVDX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 3;
    sbdsvdx("U", "V", "X", 1, d, e, ZERO, ONE, 0, 0,
            &ns, s, q, 1, w, iw, &info);
    chkxer("SBDSVDX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 4;
    sbdsvdx("U", "V", "A", -1, d, e, ZERO, ONE, 0, 0,
            &ns, s, q, 1, w, iw, &info);
    chkxer("SBDSVDX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 7;
    sbdsvdx("U", "V", "V", 2, d, e, -ONE, ZERO, 0, 0,
            &ns, s, q, 1, w, iw, &info);
    chkxer("SBDSVDX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 8;
    sbdsvdx("U", "V", "V", 2, d, e, ONE, ZERO, 0, 0,
            &ns, s, q, 1, w, iw, &info);
    chkxer("SBDSVDX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 9;
    sbdsvdx("L", "V", "I", 2, d, e, ZERO, ZERO, -1, 1,
            &ns, s, q, 1, w, iw, &info);
    chkxer("SBDSVDX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 9;
    sbdsvdx("L", "V", "I", 4, d, e, ZERO, ZERO, 4, 1,
            &ns, s, q, 1, w, iw, &info);
    chkxer("SBDSVDX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 10;
    sbdsvdx("L", "V", "I", 4, d, e, ZERO, ZERO, 2, 1,
            &ns, s, q, 1, w, iw, &info);
    chkxer("SBDSVDX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 10;
    sbdsvdx("L", "V", "I", 4, d, e, ZERO, ZERO, 2, 4,
            &ns, s, q, 1, w, iw, &info);
    chkxer("SBDSVDX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 14;
    sbdsvdx("L", "V", "A", 4, d, e, ZERO, ZERO, 0, 0,
            &ns, s, q, 0, w, iw, &info);
    chkxer("SBDSVDX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    xerbla_infot = 14;
    sbdsvdx("L", "V", "A", 4, d, e, ZERO, ZERO, 0, 0,
            &ns, s, q, 2, w, iw, &info);
    chkxer("SBDSVDX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
    *nt = *nt + 12;

    *ok = xerbla_ok;

    xerbla_srnamt[0] = '\0';
    xerbla_infot = 0;
}
