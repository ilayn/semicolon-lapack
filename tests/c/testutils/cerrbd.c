/**
 * @file cerrbd.c
 * @brief CERRBD tests the error exits for the SVD routines.
 *
 * Port of LAPACK TESTING/EIG/cerrbd.f
 * Tests error exits for: CGEBRD, CGEBD2, CUNGBR, CUNMBR, CBDSQR
 */

#include <complex.h>
#include <math.h>
#include <string.h>
#include "semicolon_lapack_complex_single.h"
#include "verify.h"

void cerrbd(const char* path, INT* ok, INT* nt)
{
    enum { NMAX = 4, LW = NMAX };
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;

    char c2[2];
    INT i, j, info;
    f32 rnan, rone;

    f32 d[NMAX], e[NMAX], rw[4 * NMAX];
    c64 a[NMAX * NMAX], tp[NMAX], tq[NMAX], u[NMAX * NMAX], v[NMAX * NMAX], w[LW];

    c2[0] = path[1];
    c2[1] = path[2];

    /* Set the variables to innocuous values. */
    for (j = 0; j < NMAX; j++) {
        for (i = 0; i < NMAX; i++) {
            a[i + j * NMAX] = CMPLXF(ONE / (f32)(i + j + 2), ZERO);
        }
    }
    xerbla_ok = 1;
    *nt = 0;

    /* Test error exits of the SVD routines. */

    if ((c2[0] == 'B' || c2[0] == 'b') && (c2[1] == 'D' || c2[1] == 'd')) {

        /* CGEBRD */

        strcpy(xerbla_srnamt, "CGEBRD");
        xerbla_infot = 1;
        cgebrd(-1, 0, a, 1, d, e, tq, tp, w, 1, &info);
        chkxer("CGEBRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        cgebrd(0, -1, a, 1, d, e, tq, tp, w, 1, &info);
        chkxer("CGEBRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        cgebrd(2, 1, a, 1, d, e, tq, tp, w, 2, &info);
        chkxer("CGEBRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        cgebrd(2, 1, a, 2, d, e, tq, tp, w, 1, &info);
        chkxer("CGEBRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 4;

        /* CGEBD2 */

        strcpy(xerbla_srnamt, "CGEBD2");
        xerbla_infot = 1;
        cgebd2(-1, 0, a, 1, d, e, tq, tp, w, &info);
        chkxer("CGEBD2", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        cgebd2(0, -1, a, 1, d, e, tq, tp, w, &info);
        chkxer("CGEBD2", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        cgebd2(2, 1, a, 1, d, e, tq, tp, w, &info);
        chkxer("CGEBD2", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 3;

        /* CUNGBR */

        strcpy(xerbla_srnamt, "CUNGBR");
        xerbla_infot = 1;
        cungbr("/", 0, 0, 0, a, 1, tq, w, 1, &info);
        chkxer("CUNGBR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        cungbr("Q", -1, 0, 0, a, 1, tq, w, 1, &info);
        chkxer("CUNGBR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        cungbr("Q", 0, -1, 0, a, 1, tq, w, 1, &info);
        chkxer("CUNGBR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        cungbr("Q", 0, 1, 0, a, 1, tq, w, 1, &info);
        chkxer("CUNGBR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        cungbr("Q", 1, 0, 1, a, 1, tq, w, 1, &info);
        chkxer("CUNGBR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        cungbr("P", 1, 0, 0, a, 1, tq, w, 1, &info);
        chkxer("CUNGBR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        cungbr("P", 0, 1, 1, a, 1, tq, w, 1, &info);
        chkxer("CUNGBR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        cungbr("Q", 0, 0, -1, a, 1, tq, w, 1, &info);
        chkxer("CUNGBR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        cungbr("Q", 2, 1, 1, a, 1, tq, w, 1, &info);
        chkxer("CUNGBR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        cungbr("Q", 2, 2, 1, a, 2, tq, w, 1, &info);
        chkxer("CUNGBR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 10;

        /* CUNMBR */

        strcpy(xerbla_srnamt, "CUNMBR");
        xerbla_infot = 1;
        cunmbr("/", "L", "T", 0, 0, 0, a, 1, tq, u, 1, w, 1, &info);
        chkxer("CUNMBR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        cunmbr("Q", "/", "T", 0, 0, 0, a, 1, tq, u, 1, w, 1, &info);
        chkxer("CUNMBR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        cunmbr("Q", "L", "/", 0, 0, 0, a, 1, tq, u, 1, w, 1, &info);
        chkxer("CUNMBR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        cunmbr("Q", "L", "C", -1, 0, 0, a, 1, tq, u, 1, w, 1, &info);
        chkxer("CUNMBR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        cunmbr("Q", "L", "C", 0, -1, 0, a, 1, tq, u, 1, w, 1, &info);
        chkxer("CUNMBR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        cunmbr("Q", "L", "C", 0, 0, -1, a, 1, tq, u, 1, w, 1, &info);
        chkxer("CUNMBR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        cunmbr("Q", "L", "C", 2, 0, 0, a, 1, tq, u, 2, w, 1, &info);
        chkxer("CUNMBR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        cunmbr("Q", "R", "C", 0, 2, 0, a, 1, tq, u, 1, w, 1, &info);
        chkxer("CUNMBR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        cunmbr("P", "L", "C", 2, 0, 2, a, 1, tq, u, 2, w, 1, &info);
        chkxer("CUNMBR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        cunmbr("P", "R", "C", 0, 2, 2, a, 1, tq, u, 1, w, 1, &info);
        chkxer("CUNMBR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 11;
        cunmbr("Q", "R", "C", 2, 0, 0, a, 1, tq, u, 1, w, 1, &info);
        chkxer("CUNMBR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 13;
        cunmbr("Q", "L", "C", 0, 2, 0, a, 1, tq, u, 1, w, 0, &info);
        chkxer("CUNMBR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 13;
        cunmbr("Q", "R", "C", 2, 0, 0, a, 1, tq, u, 2, w, 0, &info);
        chkxer("CUNMBR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 13;

        /* CBDSQR */

        strcpy(xerbla_srnamt, "CBDSQR");
        xerbla_infot = 1;
        cbdsqr("/", 0, 0, 0, 0, d, e, v, 1, u, 1, a, 1, rw, &info);
        chkxer("CBDSQR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        cbdsqr("U", -1, 0, 0, 0, d, e, v, 1, u, 1, a, 1, rw, &info);
        chkxer("CBDSQR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        cbdsqr("U", 0, -1, 0, 0, d, e, v, 1, u, 1, a, 1, rw, &info);
        chkxer("CBDSQR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        cbdsqr("U", 0, 0, -1, 0, d, e, v, 1, u, 1, a, 1, rw, &info);
        chkxer("CBDSQR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        cbdsqr("U", 0, 0, 0, -1, d, e, v, 1, u, 1, a, 1, rw, &info);
        chkxer("CBDSQR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        cbdsqr("U", 2, 1, 0, 0, d, e, v, 1, u, 1, a, 1, rw, &info);
        chkxer("CBDSQR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 11;
        cbdsqr("U", 0, 0, 2, 0, d, e, v, 1, u, 1, a, 1, rw, &info);
        chkxer("CBDSQR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 13;
        cbdsqr("U", 2, 0, 0, 1, d, e, v, 1, u, 1, a, 1, rw, &info);
        chkxer("CBDSQR", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 8;

        /* CBDSQR without singular vectors must return when D contains a
         * NaN, which the dqds path would otherwise pass to SLASCL as a
         * scaling factor, instead of stopping in XERBLA. */

        rone = ONE;
        rnan = sqrtf(-rone);
        for (j = 0; j < NMAX; j++) {
            d[j] = (f32)(j + 1);
            e[j] = ONE / (f32)(j + 2);
        }
        d[NMAX - 1] = rnan;
        e[NMAX - 1] = ZERO;
        strcpy(xerbla_srnamt, "CBDSQR");
        xerbla_infot = 0;
        xerbla_lerr = 0;
        cbdsqr("U", NMAX, 0, 0, 0, d, e, v, 1, u, 1, a, 1, rw, &info);
        if (xerbla_lerr) {
            xerbla_ok = 0;
        }
        *nt = *nt + 1;
    }

    *ok = xerbla_ok;

    xerbla_srnamt[0] = '\0';
    xerbla_infot = 0;
}
