/**
 * @file zerrgg.c
 * @brief ZERRGG tests the error exits for the generalized eigenvalue,
 *        generalized SVD, GLM, LSE, GQR and CSD routines.
 *
 * Port of LAPACK TESTING/EIG/zerrgg.f
 * Tests error exits for: ZGGES, ZGGESX, ZGGEV, ZGGEVX, ZGGES3, ZGGEV3,
 * ZGGGLM, ZGGHRD, ZGGLSE, ZGGQRF, ZGGRQF, ZGGSVD3, ZGGSVP3, ZHGEQZ,
 * ZTGEVC, ZTGEXC, ZTGSEN, ZTGSJA, ZTGSNA, ZTGSYL, ZUNBDB, and ZUNCSD.
 */

#include <complex.h>
#define LAPACK_TUNING_TEST
#include <math.h>
#include <string.h>
#include "semicolon_lapack_complex_double.h"
#include "verify.h"

void zerrgg(const char* path, INT* ok, INT* nt)
{
    enum { NMAX = 3, LW = 6 * NMAX };
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    char c2[2];
    INT dummyk = 0, dummyl = 0, i, ifst, ihi, ilo, ilst, info,
        j, m, ncycle, sdim, lwork;
    f64 anrm, bnrm, dif, scale, tola, tolb;

    INT bw[NMAX], sel[NMAX];
    INT iw[LW], idum[NMAX];
    f64 ls[NMAX], r1[NMAX], r2[NMAX], rce[NMAX], rcv[NMAX], rs[NMAX],
        rw[LW];
    c128 a[NMAX * NMAX], alpha[NMAX], b[NMAX * NMAX], beta[NMAX],
         q[NMAX * NMAX], tau[NMAX], u[NMAX * NMAX], v[NMAX * NMAX],
         w[LW], z[NMAX * NMAX];

    c2[0] = path[1];
    c2[1] = path[2];

    /* Set the variables to innocuous values. */
    for (j = 0; j < NMAX; j++) {
        sel[j] = 1;
        for (i = 0; i < NMAX; i++) {
            a[i + j * NMAX] = CMPLX(ZERO, ZERO);
            b[i + j * NMAX] = CMPLX(ZERO, ZERO);
        }
    }
    for (i = 0; i < NMAX; i++) {
        a[i + i * NMAX] = CMPLX(ONE, ZERO);
        b[i + i * NMAX] = CMPLX(ONE, ZERO);
    }
    xerbla_ok = 1;
    tola = ONE;
    tolb = ONE;
    ifst = 0;
    ilst = 0;
    *nt = 0;
    lwork = 1;

    /* Call XLAENV to set the parameters used in CLAQZ0 */
    xlaenv(12, 10);
    xlaenv(13, 12);
    xlaenv(14, 13);
    xlaenv(15, 2);
    xlaenv(17, 10);

    /* Test error exits for the GG path. */
    if ((c2[0] == 'G' || c2[0] == 'g') && (c2[1] == 'G' || c2[1] == 'g')) {

        /* ZGGHRD */

        strcpy(xerbla_srnamt, "ZGGHRD");
        xerbla_infot = 1;
        zgghrd("/", "N", 0, 0, -1, a, 1, b, 1, q, 1, z, 1, &info);
        chkxer("ZGGHRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        zgghrd("N", "/", 0, 0, -1, a, 1, b, 1, q, 1, z, 1, &info);
        chkxer("ZGGHRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        zgghrd("N", "N", -1, -1, -1, a, 1, b, 1, q, 1, z, 1, &info);
        chkxer("ZGGHRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        zgghrd("N", "N", 0, -1, -1, a, 1, b, 1, q, 1, z, 1, &info);
        chkxer("ZGGHRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        zgghrd("N", "N", 0, 0, 0, a, 1, b, 1, q, 1, z, 1, &info);
        chkxer("ZGGHRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        zgghrd("N", "N", 2, 0, 0, a, 1, b, 2, q, 1, z, 1, &info);
        chkxer("ZGGHRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        zgghrd("N", "N", 2, 0, 0, a, 2, b, 1, q, 1, z, 1, &info);
        chkxer("ZGGHRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 11;
        zgghrd("V", "N", 2, 0, 0, a, 2, b, 2, q, 1, z, 1, &info);
        chkxer("ZGGHRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 13;
        zgghrd("N", "V", 2, 0, 0, a, 2, b, 2, q, 1, z, 1, &info);
        chkxer("ZGGHRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 9;

        /* ZGGHD3 */

        strcpy(xerbla_srnamt, "ZGGHD3");
        xerbla_infot = 1;
        zgghd3("/", "N", 0, 0, -1, a, 1, b, 1, q, 1, z, 1, w, LW,
               &info);
        chkxer("ZGGHD3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        zgghd3("N", "/", 0, 0, -1, a, 1, b, 1, q, 1, z, 1, w, LW,
               &info);
        chkxer("ZGGHD3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        zgghd3("N", "N", -1, -1, -1, a, 1, b, 1, q, 1, z, 1, w, LW,
               &info);
        chkxer("ZGGHD3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        zgghd3("N", "N", 0, -1, -1, a, 1, b, 1, q, 1, z, 1, w, LW,
               &info);
        chkxer("ZGGHD3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        zgghd3("N", "N", 0, 0, 0, a, 1, b, 1, q, 1, z, 1, w, LW,
               &info);
        chkxer("ZGGHD3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        zgghd3("N", "N", 2, 0, 0, a, 1, b, 2, q, 1, z, 1, w, LW,
               &info);
        chkxer("ZGGHD3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        zgghd3("N", "N", 2, 0, 0, a, 2, b, 1, q, 1, z, 1, w, LW,
               &info);
        chkxer("ZGGHD3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 11;
        zgghd3("V", "N", 2, 0, 0, a, 2, b, 2, q, 1, z, 1, w, LW,
               &info);
        chkxer("ZGGHD3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 13;
        zgghd3("N", "V", 2, 0, 0, a, 2, b, 2, q, 1, z, 1, w, LW,
               &info);
        chkxer("ZGGHD3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 9;

        /* ZHGEQZ */

        strcpy(xerbla_srnamt, "ZHGEQZ");
        xerbla_infot = 1;
        zhgeqz("/", "N", "N", 0, 0, -1, a, 1, b, 1, alpha, beta,
               q, 1, z, 1, w, 1, rw, &info);
        chkxer("ZHGEQZ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        zhgeqz("E", "/", "N", 0, 0, -1, a, 1, b, 1, alpha, beta,
               q, 1, z, 1, w, 1, rw, &info);
        chkxer("ZHGEQZ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        zhgeqz("E", "N", "/", 0, 0, -1, a, 1, b, 1, alpha, beta,
               q, 1, z, 1, w, 1, rw, &info);
        chkxer("ZHGEQZ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        zhgeqz("E", "N", "N", -1, -1, -1, a, 1, b, 1, alpha, beta,
               q, 1, z, 1, w, 1, rw, &info);
        chkxer("ZHGEQZ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        zhgeqz("E", "N", "N", 0, -1, -1, a, 1, b, 1, alpha, beta,
               q, 1, z, 1, w, 1, rw, &info);
        chkxer("ZHGEQZ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        zhgeqz("E", "N", "N", 0, 0, 0, a, 1, b, 1, alpha, beta,
               q, 1, z, 1, w, 1, rw, &info);
        chkxer("ZHGEQZ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        zhgeqz("E", "N", "N", 2, 0, 0, a, 1, b, 2, alpha, beta,
               q, 1, z, 1, w, 1, rw, &info);
        chkxer("ZHGEQZ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        zhgeqz("E", "N", "N", 2, 0, 0, a, 2, b, 1, alpha, beta,
               q, 1, z, 1, w, 1, rw, &info);
        chkxer("ZHGEQZ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 14;
        zhgeqz("E", "V", "N", 2, 0, 0, a, 2, b, 2, alpha, beta,
               q, 1, z, 1, w, 1, rw, &info);
        chkxer("ZHGEQZ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 16;
        zhgeqz("E", "N", "V", 2, 0, 0, a, 2, b, 2, alpha, beta,
               q, 1, z, 1, w, 1, rw, &info);
        chkxer("ZHGEQZ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 10;

        /* ZTGEVC */

        strcpy(xerbla_srnamt, "ZTGEVC");
        xerbla_infot = 1;
        ztgevc("/", "A", sel, 0, a, 1, b, 1, q, 1, z, 1, 0, &m, w,
               rw, &info);
        chkxer("ZTGEVC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        ztgevc("R", "/", sel, 0, a, 1, b, 1, q, 1, z, 1, 0, &m, w,
               rw, &info);
        chkxer("ZTGEVC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        ztgevc("R", "A", sel, -1, a, 1, b, 1, q, 1, z, 1, 0, &m,
               w, rw, &info);
        chkxer("ZTGEVC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        ztgevc("R", "A", sel, 2, a, 1, b, 2, q, 1, z, 2, 0, &m, w,
               rw, &info);
        chkxer("ZTGEVC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        ztgevc("R", "A", sel, 2, a, 2, b, 1, q, 1, z, 2, 0, &m, w,
               rw, &info);
        chkxer("ZTGEVC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        ztgevc("L", "A", sel, 2, a, 2, b, 2, q, 1, z, 1, 0, &m, w,
               rw, &info);
        chkxer("ZTGEVC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 12;
        ztgevc("R", "A", sel, 2, a, 2, b, 2, q, 1, z, 1, 0, &m, w,
               rw, &info);
        chkxer("ZTGEVC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 13;
        ztgevc("R", "A", sel, 2, a, 2, b, 2, q, 1, z, 2, 1, &m, w,
               rw, &info);
        chkxer("ZTGEVC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 8;

    /* Test error exits for the GSV path. */
    } else if ((path[0] == 'G' || path[0] == 'g') &&
               (path[1] == 'S' || path[1] == 's') &&
               (path[2] == 'V' || path[2] == 'v')) {

        /* ZGGSVD3 */

        strcpy(xerbla_srnamt, "ZGGSVD3");
        xerbla_infot = 1;
        zggsvd3("/", "N", "N", 0, 0, 0, &dummyk, &dummyl, a, 1, b,
                1, r1, r2, u, 1, v, 1, q, 1, w, lwork, rw, idum,
                &info);
        chkxer("ZGGSVD3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        zggsvd3("N", "/", "N", 0, 0, 0, &dummyk, &dummyl, a, 1, b,
                1, r1, r2, u, 1, v, 1, q, 1, w, lwork, rw, idum,
                &info);
        chkxer("ZGGSVD3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        zggsvd3("N", "N", "/", 0, 0, 0, &dummyk, &dummyl, a, 1, b,
                1, r1, r2, u, 1, v, 1, q, 1, w, lwork, rw, idum,
                &info);
        chkxer("ZGGSVD3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        zggsvd3("N", "N", "N", -1, 0, 0, &dummyk, &dummyl, a, 1, b,
                1, r1, r2, u, 1, v, 1, q, 1, w, lwork, rw, idum,
                &info);
        chkxer("ZGGSVD3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        zggsvd3("N", "N", "N", 0, -1, 0, &dummyk, &dummyl, a, 1, b,
                1, r1, r2, u, 1, v, 1, q, 1, w, lwork, rw, idum,
                &info);
        chkxer("ZGGSVD3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        zggsvd3("N", "N", "N", 0, 0, -1, &dummyk, &dummyl, a, 1, b,
                1, r1, r2, u, 1, v, 1, q, 1, w, lwork, rw, idum,
                &info);
        chkxer("ZGGSVD3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        zggsvd3("N", "N", "N", 2, 1, 1, &dummyk, &dummyl, a, 1, b,
                1, r1, r2, u, 1, v, 1, q, 1, w, lwork, rw, idum,
                &info);
        chkxer("ZGGSVD3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 12;
        zggsvd3("N", "N", "N", 1, 1, 2, &dummyk, &dummyl, a, 1, b,
                1, r1, r2, u, 1, v, 1, q, 1, w, lwork, rw, idum,
                &info);
        chkxer("ZGGSVD3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 16;
        zggsvd3("U", "N", "N", 2, 2, 2, &dummyk, &dummyl, a, 2, b,
                2, r1, r2, u, 1, v, 1, q, 1, w, lwork, rw, idum,
                &info);
        chkxer("ZGGSVD3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 18;
        zggsvd3("N", "V", "N", 2, 2, 2, &dummyk, &dummyl, a, 2, b,
                2, r1, r2, u, 2, v, 1, q, 1, w, lwork, rw, idum,
                &info);
        chkxer("ZGGSVD3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 20;
        zggsvd3("N", "N", "Q", 2, 2, 2, &dummyk, &dummyl, a, 2, b,
                2, r1, r2, u, 2, v, 2, q, 1, w, lwork, rw, idum,
                &info);
        chkxer("ZGGSVD3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 11;

        /* ZGGSVP3 */

        strcpy(xerbla_srnamt, "ZGGSVP3");
        xerbla_infot = 1;
        zggsvp3("/", "N", "N", 0, 0, 0, a, 1, b, 1, tola, tolb,
                &dummyk, &dummyl, u, 1, v, 1, q, 1, iw, rw, tau, w,
                lwork, &info);
        chkxer("ZGGSVP3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        zggsvp3("N", "/", "N", 0, 0, 0, a, 1, b, 1, tola, tolb,
                &dummyk, &dummyl, u, 1, v, 1, q, 1, iw, rw, tau, w,
                lwork, &info);
        chkxer("ZGGSVP3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        zggsvp3("N", "N", "/", 0, 0, 0, a, 1, b, 1, tola, tolb,
                &dummyk, &dummyl, u, 1, v, 1, q, 1, iw, rw, tau, w,
                lwork, &info);
        chkxer("ZGGSVP3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        zggsvp3("N", "N", "N", -1, 0, 0, a, 1, b, 1, tola, tolb,
                &dummyk, &dummyl, u, 1, v, 1, q, 1, iw, rw, tau, w,
                lwork, &info);
        chkxer("ZGGSVP3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        zggsvp3("N", "N", "N", 0, -1, 0, a, 1, b, 1, tola, tolb,
                &dummyk, &dummyl, u, 1, v, 1, q, 1, iw, rw, tau, w,
                lwork, &info);
        chkxer("ZGGSVP3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        zggsvp3("N", "N", "N", 0, 0, -1, a, 1, b, 1, tola, tolb,
                &dummyk, &dummyl, u, 1, v, 1, q, 1, iw, rw, tau, w,
                lwork, &info);
        chkxer("ZGGSVP3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        zggsvp3("N", "N", "N", 2, 1, 1, a, 1, b, 1, tola, tolb,
                &dummyk, &dummyl, u, 1, v, 1, q, 1, iw, rw, tau, w,
                lwork, &info);
        chkxer("ZGGSVP3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        zggsvp3("N", "N", "N", 1, 2, 1, a, 1, b, 1, tola, tolb,
                &dummyk, &dummyl, u, 1, v, 1, q, 1, iw, rw, tau, w,
                lwork, &info);
        chkxer("ZGGSVP3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 16;
        zggsvp3("U", "N", "N", 2, 2, 2, a, 2, b, 2, tola, tolb,
                &dummyk, &dummyl, u, 1, v, 1, q, 1, iw, rw, tau, w,
                lwork, &info);
        chkxer("ZGGSVP3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 18;
        zggsvp3("N", "V", "N", 2, 2, 2, a, 2, b, 2, tola, tolb,
                &dummyk, &dummyl, u, 2, v, 1, q, 1, iw, rw, tau, w,
                lwork, &info);
        chkxer("ZGGSVP3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 20;
        zggsvp3("N", "N", "Q", 2, 2, 2, a, 2, b, 2, tola, tolb,
                &dummyk, &dummyl, u, 2, v, 2, q, 1, iw, rw, tau, w,
                lwork, &info);
        chkxer("ZGGSVP3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 11;

        /* ZTGSJA */

        strcpy(xerbla_srnamt, "ZTGSJA");
        xerbla_infot = 1;
        ztgsja("/", "N", "N", 0, 0, 0, dummyk, dummyl, a, 1, b,
               1, tola, tolb, r1, r2, u, 1, v, 1, q, 1, w,
               &ncycle, &info);
        chkxer("ZTGSJA", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        ztgsja("N", "/", "N", 0, 0, 0, dummyk, dummyl, a, 1, b,
               1, tola, tolb, r1, r2, u, 1, v, 1, q, 1, w,
               &ncycle, &info);
        chkxer("ZTGSJA", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        ztgsja("N", "N", "/", 0, 0, 0, dummyk, dummyl, a, 1, b,
               1, tola, tolb, r1, r2, u, 1, v, 1, q, 1, w,
               &ncycle, &info);
        chkxer("ZTGSJA", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        ztgsja("N", "N", "N", -1, 0, 0, dummyk, dummyl, a, 1, b,
               1, tola, tolb, r1, r2, u, 1, v, 1, q, 1, w,
               &ncycle, &info);
        chkxer("ZTGSJA", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        ztgsja("N", "N", "N", 0, -1, 0, dummyk, dummyl, a, 1, b,
               1, tola, tolb, r1, r2, u, 1, v, 1, q, 1, w,
               &ncycle, &info);
        chkxer("ZTGSJA", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        ztgsja("N", "N", "N", 0, 0, -1, dummyk, dummyl, a, 1, b,
               1, tola, tolb, r1, r2, u, 1, v, 1, q, 1, w,
               &ncycle, &info);
        chkxer("ZTGSJA", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        ztgsja("N", "N", "N", 0, 0, 0, dummyk, dummyl, a, 0, b,
               1, tola, tolb, r1, r2, u, 1, v, 1, q, 1, w,
               &ncycle, &info);
        chkxer("ZTGSJA", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 12;
        ztgsja("N", "N", "N", 0, 0, 0, dummyk, dummyl, a, 1, b,
               0, tola, tolb, r1, r2, u, 1, v, 1, q, 1, w,
               &ncycle, &info);
        chkxer("ZTGSJA", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 18;
        ztgsja("U", "N", "N", 0, 0, 0, dummyk, dummyl, a, 1, b,
               1, tola, tolb, r1, r2, u, 0, v, 1, q, 1, w,
               &ncycle, &info);
        chkxer("ZTGSJA", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 20;
        ztgsja("N", "V", "N", 0, 0, 0, dummyk, dummyl, a, 1, b,
               1, tola, tolb, r1, r2, u, 1, v, 0, q, 1, w,
               &ncycle, &info);
        chkxer("ZTGSJA", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 22;
        ztgsja("N", "N", "Q", 0, 0, 0, dummyk, dummyl, a, 1, b,
               1, tola, tolb, r1, r2, u, 1, v, 1, q, 0, w,
               &ncycle, &info);
        chkxer("ZTGSJA", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 11;

    /* Test error exits for the GLM path. */
    } else if ((path[0] == 'G' || path[0] == 'g') &&
               (path[1] == 'L' || path[1] == 'l') &&
               (path[2] == 'M' || path[2] == 'm')) {

        /* ZGGGLM */

        strcpy(xerbla_srnamt, "ZGGGLM");
        xerbla_infot = 1;
        zggglm(-1, 0, 0, a, 1, b, 1, tau, alpha, beta, w, LW,
               &info);
        chkxer("ZGGGLM", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        zggglm(0, -1, 0, a, 1, b, 1, tau, alpha, beta, w, LW,
               &info);
        chkxer("ZGGGLM", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        zggglm(0, 1, 0, a, 1, b, 1, tau, alpha, beta, w, LW,
               &info);
        chkxer("ZGGGLM", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        zggglm(0, 0, -1, a, 1, b, 1, tau, alpha, beta, w, LW,
               &info);
        chkxer("ZGGGLM", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        zggglm(1, 0, 0, a, 1, b, 1, tau, alpha, beta, w, LW,
               &info);
        chkxer("ZGGGLM", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        zggglm(0, 0, 0, a, 0, b, 1, tau, alpha, beta, w, LW,
               &info);
        chkxer("ZGGGLM", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        zggglm(0, 0, 0, a, 1, b, 0, tau, alpha, beta, w, LW,
               &info);
        chkxer("ZGGGLM", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 12;
        zggglm(1, 1, 1, a, 1, b, 1, tau, alpha, beta, w, 1,
               &info);
        chkxer("ZGGGLM", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 8;

    /* Test error exits for the LSE path. */
    } else if ((path[0] == 'L' || path[0] == 'l') &&
               (path[1] == 'S' || path[1] == 's') &&
               (path[2] == 'E' || path[2] == 'e')) {

        /* ZGGLSE */

        strcpy(xerbla_srnamt, "ZGGLSE");
        xerbla_infot = 1;
        zgglse(-1, 0, 0, a, 1, b, 1, tau, alpha, beta, w, LW,
               &info);
        chkxer("ZGGLSE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        zgglse(0, -1, 0, a, 1, b, 1, tau, alpha, beta, w, LW,
               &info);
        chkxer("ZGGLSE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        zgglse(0, 0, -1, a, 1, b, 1, tau, alpha, beta, w, LW,
               &info);
        chkxer("ZGGLSE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        zgglse(0, 0, 1, a, 1, b, 1, tau, alpha, beta, w, LW,
               &info);
        chkxer("ZGGLSE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        zgglse(0, 1, 0, a, 1, b, 1, tau, alpha, beta, w, LW,
               &info);
        chkxer("ZGGLSE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        zgglse(0, 0, 0, a, 0, b, 1, tau, alpha, beta, w, LW,
               &info);
        chkxer("ZGGLSE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        zgglse(0, 0, 0, a, 1, b, 0, tau, alpha, beta, w, LW,
               &info);
        chkxer("ZGGLSE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 12;
        zgglse(1, 1, 1, a, 1, b, 1, tau, alpha, beta, w, 1,
               &info);
        chkxer("ZGGLSE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 8;

    /* Test error exits for the CSD path. */
    } else if ((path[0] == 'C' || path[0] == 'c') &&
               (path[1] == 'S' || path[1] == 's') &&
               (path[2] == 'D' || path[2] == 'd')) {

        /* ZUNCSD */

        strcpy(xerbla_srnamt, "ZUNCSD");
        xerbla_infot = 7;
        zuncsd("Y", "Y", "Y", "Y", "N", "N",
               -1, 0, 0, a, 1, a,
               1, a, 1, a, 1, rs,
               a, 1, a, 1, a, 1, a,
               1, w, LW, rw, LW, iw, &info);
        chkxer("ZUNCSD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        zuncsd("Y", "Y", "Y", "Y", "N", "N",
               1, -1, 0, a, 1, a,
               1, a, 1, a, 1, rs,
               a, 1, a, 1, a, 1, a,
               1, w, LW, rw, LW, iw, &info);
        chkxer("ZUNCSD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        zuncsd("Y", "Y", "Y", "Y", "N", "N",
               1, 1, -1, a, 1, a,
               1, a, 1, a, 1, rs,
               a, 1, a, 1, a, 1, a,
               1, w, LW, rw, LW, iw, &info);
        chkxer("ZUNCSD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 11;
        zuncsd("Y", "Y", "Y", "Y", "N", "N",
               1, 1, 1, a, -1, a,
               1, a, 1, a, 1, rs,
               a, 1, a, 1, a, 1, a,
               1, w, LW, rw, LW, iw, &info);
        chkxer("ZUNCSD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 20;
        zuncsd("Y", "Y", "Y", "Y", "N", "N",
               1, 1, 1, a, 1, a,
               1, a, 1, a, 1, rs,
               a, -1, a, 1, a, 1, a,
               1, w, LW, rw, LW, iw, &info);
        chkxer("ZUNCSD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 22;
        zuncsd("Y", "Y", "Y", "Y", "N", "N",
               1, 1, 1, a, 1, a,
               1, a, 1, a, 1, rs,
               a, 1, a, -1, a, 1, a,
               1, w, LW, rw, LW, iw, &info);
        chkxer("ZUNCSD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 24;
        zuncsd("Y", "Y", "Y", "Y", "N", "N",
               1, 1, 1, a, 1, a,
               1, a, 1, a, 1, rs,
               a, 1, a, 1, a, -1, a,
               1, w, LW, rw, LW, iw, &info);
        chkxer("ZUNCSD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 26;
        zuncsd("Y", "Y", "Y", "Y", "N", "N",
               1, 1, 1, a, 1, a,
               1, a, 1, a, 1, rs,
               a, 1, a, 1, a, 1, a,
               -1, w, LW, rw, LW, iw, &info);
        chkxer("ZUNCSD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 8;

        /* ZUNBDB */

        strcpy(xerbla_srnamt, "ZUNBDB");
        xerbla_infot = 3;
        zunbdb("N", "N", -1, 0, 0, a, 1, b, 1, q, 1, z, 1,
               r1, r2, alpha, beta, tau, u, w, LW, &info);
        chkxer("ZUNBDB", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        zunbdb("N", "N", 1, -1, 0, a, 1, b, 1, q, 1, z, 1,
               r1, r2, alpha, beta, tau, u, w, LW, &info);
        chkxer("ZUNBDB", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        zunbdb("N", "N", 1, 1, -1, a, 1, b, 1, q, 1, z, 1,
               r1, r2, alpha, beta, tau, u, w, LW, &info);
        chkxer("ZUNBDB", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 3;

    /* Test error exits for the GQR path. */
    } else if ((path[0] == 'G' || path[0] == 'g') &&
               (path[1] == 'Q' || path[1] == 'q') &&
               (path[2] == 'R' || path[2] == 'r')) {

        /* ZGGQRF */

        strcpy(xerbla_srnamt, "ZGGQRF");
        xerbla_infot = 1;
        zggqrf(-1, 0, 0, a, 1, alpha, b, 1, beta, w, LW, &info);
        chkxer("ZGGQRF", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        zggqrf(0, -1, 0, a, 1, alpha, b, 1, beta, w, LW, &info);
        chkxer("ZGGQRF", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        zggqrf(0, 0, -1, a, 1, alpha, b, 1, beta, w, LW, &info);
        chkxer("ZGGQRF", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        zggqrf(0, 0, 0, a, 0, alpha, b, 1, beta, w, LW, &info);
        chkxer("ZGGQRF", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        zggqrf(0, 0, 0, a, 1, alpha, b, 0, beta, w, LW, &info);
        chkxer("ZGGQRF", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 11;
        zggqrf(1, 1, 2, a, 1, alpha, b, 1, beta, w, 1, &info);
        chkxer("ZGGQRF", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 6;

        /* ZGGRQF */

        strcpy(xerbla_srnamt, "ZGGRQF");
        xerbla_infot = 1;
        zggrqf(-1, 0, 0, a, 1, alpha, b, 1, beta, w, LW, &info);
        chkxer("ZGGRQF", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        zggrqf(0, -1, 0, a, 1, alpha, b, 1, beta, w, LW, &info);
        chkxer("ZGGRQF", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        zggrqf(0, 0, -1, a, 1, alpha, b, 1, beta, w, LW, &info);
        chkxer("ZGGRQF", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        zggrqf(0, 0, 0, a, 0, alpha, b, 1, beta, w, LW, &info);
        chkxer("ZGGRQF", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        zggrqf(0, 0, 0, a, 1, alpha, b, 0, beta, w, LW, &info);
        chkxer("ZGGRQF", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 11;
        zggrqf(1, 1, 2, a, 1, alpha, b, 1, beta, w, 1, &info);
        chkxer("ZGGRQF", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 6;

    /* Test error exits for the ZGS, ZGV, ZGX, and ZXV paths. */
    } else if (((path[0] == 'Z' || path[0] == 'z') &&
                (path[1] == 'G' || path[1] == 'g') &&
                (path[2] == 'S' || path[2] == 's')) ||
               ((path[0] == 'Z' || path[0] == 'z') &&
                (path[1] == 'G' || path[1] == 'g') &&
                (path[2] == 'V' || path[2] == 'v')) ||
               ((path[0] == 'Z' || path[0] == 'z') &&
                (path[1] == 'G' || path[1] == 'g') &&
                (path[2] == 'X' || path[2] == 'x')) ||
               ((path[0] == 'Z' || path[0] == 'z') &&
                (path[1] == 'X' || path[1] == 'x') &&
                (path[2] == 'V' || path[2] == 'v'))) {

        /* ZGGES */

        strcpy(xerbla_srnamt, "ZGGES ");
        xerbla_infot = 1;
        zgges("/", "N", "S", zlctes, 1, a, 1, b, 1, &sdim, alpha,
              beta, q, 1, u, 1, w, 1, rw, bw, &info);
        chkxer("ZGGES ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        zgges("N", "/", "S", zlctes, 1, a, 1, b, 1, &sdim, alpha,
              beta, q, 1, u, 1, w, 1, rw, bw, &info);
        chkxer("ZGGES ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        zgges("N", "V", "/", zlctes, 1, a, 1, b, 1, &sdim, alpha,
              beta, q, 1, u, 1, w, 1, rw, bw, &info);
        chkxer("ZGGES ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        zgges("N", "V", "S", zlctes, -1, a, 1, b, 1, &sdim, alpha,
              beta, q, 1, u, 1, w, 1, rw, bw, &info);
        chkxer("ZGGES ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        zgges("N", "V", "S", zlctes, 1, a, 0, b, 1, &sdim, alpha,
              beta, q, 1, u, 1, w, 1, rw, bw, &info);
        chkxer("ZGGES ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        zgges("N", "V", "S", zlctes, 1, a, 1, b, 0, &sdim, alpha,
              beta, q, 1, u, 1, w, 1, rw, bw, &info);
        chkxer("ZGGES ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 14;
        zgges("N", "V", "S", zlctes, 1, a, 1, b, 1, &sdim, alpha,
              beta, q, 0, u, 1, w, 1, rw, bw, &info);
        chkxer("ZGGES ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 14;
        zgges("V", "V", "S", zlctes, 2, a, 2, b, 2, &sdim, alpha,
              beta, q, 1, u, 2, w, 1, rw, bw, &info);
        chkxer("ZGGES ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 16;
        zgges("N", "V", "S", zlctes, 1, a, 1, b, 1, &sdim, alpha,
              beta, q, 1, u, 0, w, 1, rw, bw, &info);
        chkxer("ZGGES ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 16;
        zgges("V", "V", "S", zlctes, 2, a, 2, b, 2, &sdim, alpha,
              beta, q, 2, u, 1, w, 1, rw, bw, &info);
        chkxer("ZGGES ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 18;
        zgges("V", "V", "S", zlctes, 2, a, 2, b, 2, &sdim, alpha,
              beta, q, 2, u, 2, w, 1, rw, bw, &info);
        chkxer("ZGGES ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 11;

        /* ZGGES3 */

        strcpy(xerbla_srnamt, "ZGGES3");
        xerbla_infot = 1;
        zgges3("/", "N", "S", zlctes, 1, a, 1, b, 1, &sdim, alpha,
               beta, q, 1, u, 1, w, 1, rw, bw, &info);
        chkxer("ZGGES3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        zgges3("N", "/", "S", zlctes, 1, a, 1, b, 1, &sdim, alpha,
               beta, q, 1, u, 1, w, 1, rw, bw, &info);
        chkxer("ZGGES3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        zgges3("N", "V", "/", zlctes, 1, a, 1, b, 1, &sdim, alpha,
               beta, q, 1, u, 1, w, 1, rw, bw, &info);
        chkxer("ZGGES3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        zgges3("N", "V", "S", zlctes, -1, a, 1, b, 1, &sdim,
               alpha, beta, q, 1, u, 1, w, 1, rw, bw, &info);
        chkxer("ZGGES3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        zgges3("N", "V", "S", zlctes, 1, a, 0, b, 1, &sdim, alpha,
               beta, q, 1, u, 1, w, 1, rw, bw, &info);
        chkxer("ZGGES3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        zgges3("N", "V", "S", zlctes, 1, a, 1, b, 0, &sdim, alpha,
               beta, q, 1, u, 1, w, 1, rw, bw, &info);
        chkxer("ZGGES3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 14;
        zgges3("N", "V", "S", zlctes, 1, a, 1, b, 1, &sdim, alpha,
               beta, q, 0, u, 1, w, 1, rw, bw, &info);
        chkxer("ZGGES3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 14;
        zgges3("V", "V", "S", zlctes, 2, a, 2, b, 2, &sdim, alpha,
               beta, q, 1, u, 2, w, 1, rw, bw, &info);
        chkxer("ZGGES3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 16;
        zgges3("N", "V", "S", zlctes, 1, a, 1, b, 1, &sdim, alpha,
               beta, q, 1, u, 0, w, 1, rw, bw, &info);
        chkxer("ZGGES3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 16;
        zgges3("V", "V", "S", zlctes, 2, a, 2, b, 2, &sdim, alpha,
               beta, q, 2, u, 1, w, 1, rw, bw, &info);
        chkxer("ZGGES3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 18;
        zgges3("V", "V", "S", zlctes, 2, a, 2, b, 2, &sdim, alpha,
               beta, q, 2, u, 2, w, 1, rw, bw, &info);
        chkxer("ZGGES3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 11;

        /* ZGGESX */

        strcpy(xerbla_srnamt, "ZGGESX");
        xerbla_infot = 1;
        zggesx("/", "N", "S", zlctsx, "N", 1, a, 1, b, 1, &sdim,
               alpha, beta, q, 1, u, 1, rce, rcv, w, 1, rw, iw,
               1, bw, &info);
        chkxer("ZGGESX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        zggesx("N", "/", "S", zlctsx, "N", 1, a, 1, b, 1, &sdim,
               alpha, beta, q, 1, u, 1, rce, rcv, w, 1, rw, iw,
               1, bw, &info);
        chkxer("ZGGESX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        zggesx("V", "V", "/", zlctsx, "N", 1, a, 1, b, 1, &sdim,
               alpha, beta, q, 1, u, 1, rce, rcv, w, 1, rw, iw,
               1, bw, &info);
        chkxer("ZGGESX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        zggesx("V", "V", "S", zlctsx, "/", 1, a, 1, b, 1, &sdim,
               alpha, beta, q, 1, u, 1, rce, rcv, w, 1, rw, iw,
               1, bw, &info);
        chkxer("ZGGESX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        zggesx("V", "V", "S", zlctsx, "B", -1, a, 1, b, 1, &sdim,
               alpha, beta, q, 1, u, 1, rce, rcv, w, 1, rw, iw,
               1, bw, &info);
        chkxer("ZGGESX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        zggesx("V", "V", "S", zlctsx, "B", 1, a, 0, b, 1, &sdim,
               alpha, beta, q, 1, u, 1, rce, rcv, w, 1, rw, iw,
               1, bw, &info);
        chkxer("ZGGESX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        zggesx("V", "V", "S", zlctsx, "B", 1, a, 1, b, 0, &sdim,
               alpha, beta, q, 1, u, 1, rce, rcv, w, 1, rw, iw,
               1, bw, &info);
        chkxer("ZGGESX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 15;
        zggesx("V", "V", "S", zlctsx, "B", 1, a, 1, b, 1, &sdim,
               alpha, beta, q, 0, u, 1, rce, rcv, w, 1, rw, iw,
               1, bw, &info);
        chkxer("ZGGESX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 15;
        zggesx("V", "V", "S", zlctsx, "B", 2, a, 2, b, 2, &sdim,
               alpha, beta, q, 1, u, 1, rce, rcv, w, 1, rw, iw,
               1, bw, &info);
        chkxer("ZGGESX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 17;
        zggesx("V", "V", "S", zlctsx, "B", 1, a, 1, b, 1, &sdim,
               alpha, beta, q, 1, u, 0, rce, rcv, w, 1, rw, iw,
               1, bw, &info);
        chkxer("ZGGESX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 17;
        zggesx("V", "V", "S", zlctsx, "B", 2, a, 2, b, 2, &sdim,
               alpha, beta, q, 2, u, 1, rce, rcv, w, 1, rw, iw,
               1, bw, &info);
        chkxer("ZGGESX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 21;
        zggesx("V", "V", "S", zlctsx, "B", 2, a, 2, b, 2, &sdim,
               alpha, beta, q, 2, u, 2, rce, rcv, w, 1, rw, iw,
               1, bw, &info);
        chkxer("ZGGESX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 24;
        zggesx("V", "V", "S", zlctsx, "V", 1, a, 1, b, 1, &sdim,
               alpha, beta, q, 1, u, 1, rce, rcv, w, 32, rw, iw,
               0, bw, &info);
        chkxer("ZGGESX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 13;

        /* ZGGEV */

        strcpy(xerbla_srnamt, "ZGGEV ");
        xerbla_infot = 1;
        zggev("/", "N", 1, a, 1, b, 1, alpha, beta, q, 1, u, 1,
              w, 1, rw, &info);
        chkxer("ZGGEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        zggev("N", "/", 1, a, 1, b, 1, alpha, beta, q, 1, u, 1,
              w, 1, rw, &info);
        chkxer("ZGGEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        zggev("V", "V", -1, a, 1, b, 1, alpha, beta, q, 1, u, 1,
              w, 1, rw, &info);
        chkxer("ZGGEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        zggev("V", "V", 1, a, 0, b, 1, alpha, beta, q, 1, u, 1,
              w, 1, rw, &info);
        chkxer("ZGGEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        zggev("V", "V", 1, a, 1, b, 0, alpha, beta, q, 1, u, 1,
              w, 1, rw, &info);
        chkxer("ZGGEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 11;
        zggev("N", "V", 1, a, 1, b, 1, alpha, beta, q, 0, u, 1,
              w, 1, rw, &info);
        chkxer("ZGGEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 11;
        zggev("V", "V", 2, a, 2, b, 2, alpha, beta, q, 1, u, 2,
              w, 1, rw, &info);
        chkxer("ZGGEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 13;
        zggev("V", "N", 2, a, 2, b, 2, alpha, beta, q, 2, u, 0,
              w, 1, rw, &info);
        chkxer("ZGGEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 13;
        zggev("V", "V", 2, a, 2, b, 2, alpha, beta, q, 2, u, 1,
              w, 1, rw, &info);
        chkxer("ZGGEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 15;
        zggev("V", "V", 1, a, 1, b, 1, alpha, beta, q, 1, u, 1,
              w, 1, rw, &info);
        chkxer("ZGGEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 10;

        /* ZGGEV3 */

        strcpy(xerbla_srnamt, "ZGGEV3");
        xerbla_infot = 1;
        zggev3("/", "N", 1, a, 1, b, 1, alpha, beta, q, 1, u, 1,
               w, 1, rw, &info);
        chkxer("ZGGEV3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        zggev3("N", "/", 1, a, 1, b, 1, alpha, beta, q, 1, u, 1,
               w, 1, rw, &info);
        chkxer("ZGGEV3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        zggev3("V", "V", -1, a, 1, b, 1, alpha, beta, q, 1, u, 1,
               w, 1, rw, &info);
        chkxer("ZGGEV3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        zggev3("V", "V", 1, a, 0, b, 1, alpha, beta, q, 1, u, 1,
               w, 1, rw, &info);
        chkxer("ZGGEV3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        zggev3("V", "V", 1, a, 1, b, 0, alpha, beta, q, 1, u, 1,
               w, 1, rw, &info);
        chkxer("ZGGEV3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 11;
        zggev3("N", "V", 1, a, 1, b, 1, alpha, beta, q, 0, u, 1,
               w, 1, rw, &info);
        chkxer("ZGGEV3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 11;
        zggev3("V", "V", 2, a, 2, b, 2, alpha, beta, q, 1, u, 2,
               w, 1, rw, &info);
        chkxer("ZGGEV3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 13;
        zggev3("V", "N", 2, a, 2, b, 2, alpha, beta, q, 2, u, 0,
               w, 1, rw, &info);
        chkxer("ZGGEV3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 13;
        zggev3("V", "V", 2, a, 2, b, 2, alpha, beta, q, 2, u, 1,
               w, 1, rw, &info);
        chkxer("ZGGEV3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 15;
        zggev3("V", "V", 1, a, 1, b, 1, alpha, beta, q, 1, u, 1,
               w, 1, rw, &info);
        chkxer("ZGGEV3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 10;

        /* ZGGEVX */

        strcpy(xerbla_srnamt, "ZGGEVX");
        xerbla_infot = 1;
        zggevx("/", "N", "N", "N", 1, a, 1, b, 1, alpha, beta, q,
               1, u, 1, &ilo, &ihi, ls, rs, &anrm, &bnrm, rce, rcv,
               w, 1, rw, iw, bw, &info);
        chkxer("ZGGEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        zggevx("N", "/", "N", "N", 1, a, 1, b, 1, alpha, beta, q,
               1, u, 1, &ilo, &ihi, ls, rs, &anrm, &bnrm, rce, rcv,
               w, 1, rw, iw, bw, &info);
        chkxer("ZGGEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        zggevx("N", "N", "/", "N", 1, a, 1, b, 1, alpha, beta, q,
               1, u, 1, &ilo, &ihi, ls, rs, &anrm, &bnrm, rce, rcv,
               w, 1, rw, iw, bw, &info);
        chkxer("ZGGEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        zggevx("N", "N", "N", "/", 1, a, 1, b, 1, alpha, beta, q,
               1, u, 1, &ilo, &ihi, ls, rs, &anrm, &bnrm, rce, rcv,
               w, 1, rw, iw, bw, &info);
        chkxer("ZGGEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        zggevx("N", "N", "N", "N", -1, a, 1, b, 1, alpha, beta,
               q, 1, u, 1, &ilo, &ihi, ls, rs, &anrm, &bnrm, rce,
               rcv, w, 1, rw, iw, bw, &info);
        chkxer("ZGGEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        zggevx("N", "N", "N", "N", 1, a, 0, b, 1, alpha, beta, q,
               1, u, 1, &ilo, &ihi, ls, rs, &anrm, &bnrm, rce, rcv,
               w, 1, rw, iw, bw, &info);
        chkxer("ZGGEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        zggevx("N", "N", "N", "N", 1, a, 1, b, 0, alpha, beta, q,
               1, u, 1, &ilo, &ihi, ls, rs, &anrm, &bnrm, rce, rcv,
               w, 1, rw, iw, bw, &info);
        chkxer("ZGGEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 13;
        zggevx("N", "N", "N", "N", 1, a, 1, b, 1, alpha, beta, q,
               0, u, 1, &ilo, &ihi, ls, rs, &anrm, &bnrm, rce, rcv,
               w, 1, rw, iw, bw, &info);
        chkxer("ZGGEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 13;
        zggevx("N", "V", "N", "N", 2, a, 2, b, 2, alpha, beta, q,
               1, u, 2, &ilo, &ihi, ls, rs, &anrm, &bnrm, rce, rcv,
               w, 1, rw, iw, bw, &info);
        chkxer("ZGGEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 15;
        zggevx("N", "N", "N", "N", 1, a, 1, b, 1, alpha, beta, q,
               1, u, 0, &ilo, &ihi, ls, rs, &anrm, &bnrm, rce, rcv,
               w, 1, rw, iw, bw, &info);
        chkxer("ZGGEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 15;
        zggevx("N", "N", "V", "N", 2, a, 2, b, 2, alpha, beta, q,
               2, u, 1, &ilo, &ihi, ls, rs, &anrm, &bnrm, rce, rcv,
               w, 1, rw, iw, bw, &info);
        chkxer("ZGGEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 25;
        zggevx("N", "N", "V", "N", 2, a, 2, b, 2, alpha, beta, q,
               2, u, 2, &ilo, &ihi, ls, rs, &anrm, &bnrm, rce, rcv,
               w, 0, rw, iw, bw, &info);
        chkxer("ZGGEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 12;

        /* ZTGEXC */

        strcpy(xerbla_srnamt, "ZTGEXC");
        xerbla_infot = 3;
        ztgexc(1, 1, -1, a, 1, b, 1, q, 1, z, 1, ifst,
               &ilst, &info);
        chkxer("ZTGEXC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        ztgexc(1, 1, 1, a, 0, b, 1, q, 1, z, 1, ifst,
               &ilst, &info);
        chkxer("ZTGEXC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        ztgexc(1, 1, 1, a, 1, b, 0, q, 1, z, 1, ifst,
               &ilst, &info);
        chkxer("ZTGEXC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        ztgexc(0, 1, 1, a, 1, b, 1, q, 0, z, 1, ifst,
               &ilst, &info);
        chkxer("ZTGEXC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        ztgexc(1, 1, 1, a, 1, b, 1, q, 0, z, 1, ifst,
               &ilst, &info);
        chkxer("ZTGEXC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 11;
        ztgexc(1, 0, 1, a, 1, b, 1, q, 1, z, 0, ifst,
               &ilst, &info);
        chkxer("ZTGEXC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 11;
        ztgexc(1, 1, 1, a, 1, b, 1, q, 1, z, 0, ifst,
               &ilst, &info);
        chkxer("ZTGEXC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 7;

        /* ZTGSEN */

        strcpy(xerbla_srnamt, "ZTGSEN");
        xerbla_infot = 1;
        ztgsen(-1, 1, 1, sel, 1, a, 1, b, 1, alpha,
               beta, q, 1, z, 1, &m, &tola, &tolb, rcv, w, 1, iw, 1,
               &info);
        chkxer("ZTGSEN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        ztgsen(1, 1, 1, sel, -1, a, 1, b, 1, alpha,
               beta, q, 1, z, 1, &m, &tola, &tolb, rcv, w, 1, iw, 1,
               &info);
        chkxer("ZTGSEN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        ztgsen(1, 1, 1, sel, 1, a, 0, b, 1, alpha,
               beta, q, 1, z, 1, &m, &tola, &tolb, rcv, w, 1, iw, 1,
               &info);
        chkxer("ZTGSEN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        ztgsen(1, 1, 1, sel, 1, a, 1, b, 0, alpha,
               beta, q, 1, z, 1, &m, &tola, &tolb, rcv, w, 1, iw, 1,
               &info);
        chkxer("ZTGSEN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 13;
        ztgsen(1, 1, 1, sel, 1, a, 1, b, 1, alpha,
               beta, q, 0, z, 1, &m, &tola, &tolb, rcv, w, 1, iw, 1,
               &info);
        chkxer("ZTGSEN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 15;
        ztgsen(1, 1, 1, sel, 1, a, 1, b, 1, alpha,
               beta, q, 1, z, 0, &m, &tola, &tolb, rcv, w, 1, iw, 1,
               &info);
        chkxer("ZTGSEN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 21;
        ztgsen(3, 1, 1, sel, 1, a, 1, b, 1, alpha,
               beta, q, 1, z, 1, &m, &tola, &tolb, rcv, w, -5, iw,
               1, &info);
        chkxer("ZTGSEN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 23;
        ztgsen(0, 1, 1, sel, 1, a, 1, b, 1, alpha,
               beta, q, 1, z, 1, &m, &tola, &tolb, rcv, w, 20, iw,
               0, &info);
        chkxer("ZTGSEN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 23;
        ztgsen(1, 1, 1, sel, 1, a, 1, b, 1, alpha,
               beta, q, 1, z, 1, &m, &tola, &tolb, rcv, w, 20, iw,
               0, &info);
        chkxer("ZTGSEN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 23;
        ztgsen(5, 1, 1, sel, 1, a, 1, b, 1, alpha,
               beta, q, 1, z, 1, &m, &tola, &tolb, rcv, w, 20, iw,
               1, &info);
        chkxer("ZTGSEN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 11;

        /* ZTGSNA */

        strcpy(xerbla_srnamt, "ZTGSNA");
        xerbla_infot = 1;
        ztgsna("/", "A", sel, 1, a, 1, b, 1, q, 1, u, 1, r1, r2,
               1, &m, w, 1, iw, &info);
        chkxer("ZTGSNA", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        ztgsna("B", "/", sel, 1, a, 1, b, 1, q, 1, u, 1, r1, r2,
               1, &m, w, 1, iw, &info);
        chkxer("ZTGSNA", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        ztgsna("B", "A", sel, -1, a, 1, b, 1, q, 1, u, 1, r1, r2,
               1, &m, w, 1, iw, &info);
        chkxer("ZTGSNA", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        ztgsna("B", "A", sel, 1, a, 0, b, 1, q, 1, u, 1, r1, r2,
               1, &m, w, 1, iw, &info);
        chkxer("ZTGSNA", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        ztgsna("B", "A", sel, 1, a, 1, b, 0, q, 1, u, 1, r1, r2,
               1, &m, w, 1, iw, &info);
        chkxer("ZTGSNA", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        ztgsna("E", "A", sel, 1, a, 1, b, 1, q, 0, u, 1, r1, r2,
               1, &m, w, 1, iw, &info);
        chkxer("ZTGSNA", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 12;
        ztgsna("E", "A", sel, 1, a, 1, b, 1, q, 1, u, 0, r1, r2,
               1, &m, w, 1, iw, &info);
        chkxer("ZTGSNA", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 15;
        ztgsna("E", "A", sel, 1, a, 1, b, 1, q, 1, u, 1, r1, r2,
               0, &m, w, 1, iw, &info);
        chkxer("ZTGSNA", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 18;
        ztgsna("E", "A", sel, 1, a, 1, b, 1, q, 1, u, 1, r1, r2,
               1, &m, w, 0, iw, &info);
        chkxer("ZTGSNA", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 9;

        /* ZTGSYL */

        strcpy(xerbla_srnamt, "ZTGSYL");
        xerbla_infot = 1;
        ztgsyl("/", 0, 1, 1, a, 1, b, 1, q, 1, u, 1, v, 1, z, 1,
               &scale, &dif, w, 1, iw, &info);
        chkxer("ZTGSYL", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        ztgsyl("N", -1, 1, 1, a, 1, b, 1, q, 1, u, 1, v, 1, z, 1,
               &scale, &dif, w, 1, iw, &info);
        chkxer("ZTGSYL", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        ztgsyl("N", 0, 0, 1, a, 1, b, 1, q, 1, u, 1, v, 1, z, 1,
               &scale, &dif, w, 1, iw, &info);
        chkxer("ZTGSYL", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        ztgsyl("N", 0, 1, 0, a, 1, b, 1, q, 1, u, 1, v, 1, z, 1,
               &scale, &dif, w, 1, iw, &info);
        chkxer("ZTGSYL", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        ztgsyl("N", 0, 1, 1, a, 0, b, 1, q, 1, u, 1, v, 1, z, 1,
               &scale, &dif, w, 1, iw, &info);
        chkxer("ZTGSYL", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        ztgsyl("N", 0, 1, 1, a, 1, b, 0, q, 1, u, 1, v, 1, z, 1,
               &scale, &dif, w, 1, iw, &info);
        chkxer("ZTGSYL", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        ztgsyl("N", 0, 1, 1, a, 1, b, 1, q, 0, u, 1, v, 1, z, 1,
               &scale, &dif, w, 1, iw, &info);
        chkxer("ZTGSYL", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 12;
        ztgsyl("N", 0, 1, 1, a, 1, b, 1, q, 1, u, 0, v, 1, z, 1,
               &scale, &dif, w, 1, iw, &info);
        chkxer("ZTGSYL", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 14;
        ztgsyl("N", 0, 1, 1, a, 1, b, 1, q, 1, u, 1, v, 0, z, 1,
               &scale, &dif, w, 1, iw, &info);
        chkxer("ZTGSYL", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 16;
        ztgsyl("N", 0, 1, 1, a, 1, b, 1, q, 1, u, 1, v, 1, z, 0,
               &scale, &dif, w, 1, iw, &info);
        chkxer("ZTGSYL", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 20;
        ztgsyl("N", 1, 1, 1, a, 1, b, 1, q, 1, u, 1, v, 1, z, 1,
               &scale, &dif, w, 1, iw, &info);
        chkxer("ZTGSYL", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 20;
        ztgsyl("N", 2, 1, 1, a, 1, b, 1, q, 1, u, 1, v, 1, z, 1,
               &scale, &dif, w, 1, iw, &info);
        chkxer("ZTGSYL", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 12;
    }

    *ok = xerbla_ok;

    xerbla_srnamt[0] = '\0';
    xerbla_infot = 0;
}
