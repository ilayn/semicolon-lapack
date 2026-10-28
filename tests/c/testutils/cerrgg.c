/**
 * @file cerrgg.c
 * @brief CERRGG tests the error exits for the generalized eigenvalue,
 *        generalized SVD, GLM, LSE, GQR and CSD routines.
 *
 * Port of LAPACK TESTING/EIG/cerrgg.f
 * Tests error exits for: CGGES, CGGESX, CGGEV, CGGEVX, CGGES3, CGGEV3,
 * CGGGLM, CGGHRD, CGGLSE, CGGQRF, CGGRQF, CGGSVD3, CGGSVP3, CHGEQZ,
 * CTGEVC, CTGEXC, CTGSEN, CTGSJA, CTGSNA, CTGSYL, CUNBDB, and CUNCSD.
 */

#include <complex.h>
#define LAPACK_TUNING_TEST
#include <math.h>
#include <string.h>
#include "semicolon_lapack_complex_single.h"
#include "verify.h"

void cerrgg(const char* path, INT* ok, INT* nt)
{
    enum { NMAX = 3, LW = 6 * NMAX };
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;

    char c2[2];
    INT dummyk = 0, dummyl = 0, i, ifst, ihi, ilo, ilst, info,
        j, m, ncycle, sdim, lwork;
    f32 anrm, bnrm, dif, scale, tola, tolb;

    INT bw[NMAX], sel[NMAX];
    INT iw[LW], idum[NMAX];
    f32 ls[NMAX], r1[NMAX], r2[NMAX], rce[NMAX], rcv[NMAX], rs[NMAX],
        rw[LW];
    c64 a[NMAX * NMAX], alpha[NMAX], b[NMAX * NMAX], beta[NMAX],
         q[NMAX * NMAX], tau[NMAX], u[NMAX * NMAX], v[NMAX * NMAX],
         w[LW], z[NMAX * NMAX];

    c2[0] = path[1];
    c2[1] = path[2];

    /* Set the variables to innocuous values. */
    for (j = 0; j < NMAX; j++) {
        sel[j] = 1;
        for (i = 0; i < NMAX; i++) {
            a[i + j * NMAX] = CMPLXF(ZERO, ZERO);
            b[i + j * NMAX] = CMPLXF(ZERO, ZERO);
        }
    }
    for (i = 0; i < NMAX; i++) {
        a[i + i * NMAX] = CMPLXF(ONE, ZERO);
        b[i + i * NMAX] = CMPLXF(ONE, ZERO);
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

        /* CGGHRD */

        strcpy(xerbla_srnamt, "CGGHRD");
        xerbla_infot = 1;
        cgghrd("/", "N", 0, 0, -1, a, 1, b, 1, q, 1, z, 1, &info);
        chkxer("CGGHRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        cgghrd("N", "/", 0, 0, -1, a, 1, b, 1, q, 1, z, 1, &info);
        chkxer("CGGHRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        cgghrd("N", "N", -1, -1, -1, a, 1, b, 1, q, 1, z, 1, &info);
        chkxer("CGGHRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        cgghrd("N", "N", 0, -1, -1, a, 1, b, 1, q, 1, z, 1, &info);
        chkxer("CGGHRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        cgghrd("N", "N", 0, 0, 0, a, 1, b, 1, q, 1, z, 1, &info);
        chkxer("CGGHRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        cgghrd("N", "N", 2, 0, 0, a, 1, b, 2, q, 1, z, 1, &info);
        chkxer("CGGHRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        cgghrd("N", "N", 2, 0, 0, a, 2, b, 1, q, 1, z, 1, &info);
        chkxer("CGGHRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 11;
        cgghrd("V", "N", 2, 0, 0, a, 2, b, 2, q, 1, z, 1, &info);
        chkxer("CGGHRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 13;
        cgghrd("N", "V", 2, 0, 0, a, 2, b, 2, q, 1, z, 1, &info);
        chkxer("CGGHRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 9;

        /* CGGHD3 */

        strcpy(xerbla_srnamt, "CGGHD3");
        xerbla_infot = 1;
        cgghd3("/", "N", 0, 0, -1, a, 1, b, 1, q, 1, z, 1, w, LW,
               &info);
        chkxer("CGGHD3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        cgghd3("N", "/", 0, 0, -1, a, 1, b, 1, q, 1, z, 1, w, LW,
               &info);
        chkxer("CGGHD3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        cgghd3("N", "N", -1, -1, -1, a, 1, b, 1, q, 1, z, 1, w, LW,
               &info);
        chkxer("CGGHD3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        cgghd3("N", "N", 0, -1, -1, a, 1, b, 1, q, 1, z, 1, w, LW,
               &info);
        chkxer("CGGHD3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        cgghd3("N", "N", 0, 0, 0, a, 1, b, 1, q, 1, z, 1, w, LW,
               &info);
        chkxer("CGGHD3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        cgghd3("N", "N", 2, 0, 0, a, 1, b, 2, q, 1, z, 1, w, LW,
               &info);
        chkxer("CGGHD3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        cgghd3("N", "N", 2, 0, 0, a, 2, b, 1, q, 1, z, 1, w, LW,
               &info);
        chkxer("CGGHD3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 11;
        cgghd3("V", "N", 2, 0, 0, a, 2, b, 2, q, 1, z, 1, w, LW,
               &info);
        chkxer("CGGHD3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 13;
        cgghd3("N", "V", 2, 0, 0, a, 2, b, 2, q, 1, z, 1, w, LW,
               &info);
        chkxer("CGGHD3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 9;

        /* CHGEQZ */

        strcpy(xerbla_srnamt, "CHGEQZ");
        xerbla_infot = 1;
        chgeqz("/", "N", "N", 0, 0, -1, a, 1, b, 1, alpha, beta,
               q, 1, z, 1, w, 1, rw, &info);
        chkxer("CHGEQZ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        chgeqz("E", "/", "N", 0, 0, -1, a, 1, b, 1, alpha, beta,
               q, 1, z, 1, w, 1, rw, &info);
        chkxer("CHGEQZ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        chgeqz("E", "N", "/", 0, 0, -1, a, 1, b, 1, alpha, beta,
               q, 1, z, 1, w, 1, rw, &info);
        chkxer("CHGEQZ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        chgeqz("E", "N", "N", -1, -1, -1, a, 1, b, 1, alpha, beta,
               q, 1, z, 1, w, 1, rw, &info);
        chkxer("CHGEQZ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        chgeqz("E", "N", "N", 0, -1, -1, a, 1, b, 1, alpha, beta,
               q, 1, z, 1, w, 1, rw, &info);
        chkxer("CHGEQZ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        chgeqz("E", "N", "N", 0, 0, 0, a, 1, b, 1, alpha, beta,
               q, 1, z, 1, w, 1, rw, &info);
        chkxer("CHGEQZ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        chgeqz("E", "N", "N", 2, 0, 0, a, 1, b, 2, alpha, beta,
               q, 1, z, 1, w, 1, rw, &info);
        chkxer("CHGEQZ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        chgeqz("E", "N", "N", 2, 0, 0, a, 2, b, 1, alpha, beta,
               q, 1, z, 1, w, 1, rw, &info);
        chkxer("CHGEQZ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 14;
        chgeqz("E", "V", "N", 2, 0, 0, a, 2, b, 2, alpha, beta,
               q, 1, z, 1, w, 1, rw, &info);
        chkxer("CHGEQZ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 16;
        chgeqz("E", "N", "V", 2, 0, 0, a, 2, b, 2, alpha, beta,
               q, 1, z, 1, w, 1, rw, &info);
        chkxer("CHGEQZ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 10;

        /* CTGEVC */

        strcpy(xerbla_srnamt, "CTGEVC");
        xerbla_infot = 1;
        ctgevc("/", "A", sel, 0, a, 1, b, 1, q, 1, z, 1, 0, &m, w,
               rw, &info);
        chkxer("CTGEVC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        ctgevc("R", "/", sel, 0, a, 1, b, 1, q, 1, z, 1, 0, &m, w,
               rw, &info);
        chkxer("CTGEVC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        ctgevc("R", "A", sel, -1, a, 1, b, 1, q, 1, z, 1, 0, &m,
               w, rw, &info);
        chkxer("CTGEVC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        ctgevc("R", "A", sel, 2, a, 1, b, 2, q, 1, z, 2, 0, &m, w,
               rw, &info);
        chkxer("CTGEVC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        ctgevc("R", "A", sel, 2, a, 2, b, 1, q, 1, z, 2, 0, &m, w,
               rw, &info);
        chkxer("CTGEVC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        ctgevc("L", "A", sel, 2, a, 2, b, 2, q, 1, z, 1, 0, &m, w,
               rw, &info);
        chkxer("CTGEVC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 12;
        ctgevc("R", "A", sel, 2, a, 2, b, 2, q, 1, z, 1, 0, &m, w,
               rw, &info);
        chkxer("CTGEVC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 13;
        ctgevc("R", "A", sel, 2, a, 2, b, 2, q, 1, z, 2, 1, &m, w,
               rw, &info);
        chkxer("CTGEVC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 8;

    /* Test error exits for the GSV path. */
    } else if ((path[0] == 'G' || path[0] == 'g') &&
               (path[1] == 'S' || path[1] == 's') &&
               (path[2] == 'V' || path[2] == 'v')) {

        /* CGGSVD3 */

        strcpy(xerbla_srnamt, "CGGSVD3");
        xerbla_infot = 1;
        cggsvd3("/", "N", "N", 0, 0, 0, &dummyk, &dummyl, a, 1, b,
                1, r1, r2, u, 1, v, 1, q, 1, w, lwork, rw, idum,
                &info);
        chkxer("CGGSVD3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        cggsvd3("N", "/", "N", 0, 0, 0, &dummyk, &dummyl, a, 1, b,
                1, r1, r2, u, 1, v, 1, q, 1, w, lwork, rw, idum,
                &info);
        chkxer("CGGSVD3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        cggsvd3("N", "N", "/", 0, 0, 0, &dummyk, &dummyl, a, 1, b,
                1, r1, r2, u, 1, v, 1, q, 1, w, lwork, rw, idum,
                &info);
        chkxer("CGGSVD3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        cggsvd3("N", "N", "N", -1, 0, 0, &dummyk, &dummyl, a, 1, b,
                1, r1, r2, u, 1, v, 1, q, 1, w, lwork, rw, idum,
                &info);
        chkxer("CGGSVD3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        cggsvd3("N", "N", "N", 0, -1, 0, &dummyk, &dummyl, a, 1, b,
                1, r1, r2, u, 1, v, 1, q, 1, w, lwork, rw, idum,
                &info);
        chkxer("CGGSVD3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        cggsvd3("N", "N", "N", 0, 0, -1, &dummyk, &dummyl, a, 1, b,
                1, r1, r2, u, 1, v, 1, q, 1, w, lwork, rw, idum,
                &info);
        chkxer("CGGSVD3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        cggsvd3("N", "N", "N", 2, 1, 1, &dummyk, &dummyl, a, 1, b,
                1, r1, r2, u, 1, v, 1, q, 1, w, lwork, rw, idum,
                &info);
        chkxer("CGGSVD3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 12;
        cggsvd3("N", "N", "N", 1, 1, 2, &dummyk, &dummyl, a, 1, b,
                1, r1, r2, u, 1, v, 1, q, 1, w, lwork, rw, idum,
                &info);
        chkxer("CGGSVD3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 16;
        cggsvd3("U", "N", "N", 2, 2, 2, &dummyk, &dummyl, a, 2, b,
                2, r1, r2, u, 1, v, 1, q, 1, w, lwork, rw, idum,
                &info);
        chkxer("CGGSVD3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 18;
        cggsvd3("N", "V", "N", 2, 2, 2, &dummyk, &dummyl, a, 2, b,
                2, r1, r2, u, 2, v, 1, q, 1, w, lwork, rw, idum,
                &info);
        chkxer("CGGSVD3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 20;
        cggsvd3("N", "N", "Q", 2, 2, 2, &dummyk, &dummyl, a, 2, b,
                2, r1, r2, u, 2, v, 2, q, 1, w, lwork, rw, idum,
                &info);
        chkxer("CGGSVD3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 11;

        /* CGGSVP3 */

        strcpy(xerbla_srnamt, "CGGSVP3");
        xerbla_infot = 1;
        cggsvp3("/", "N", "N", 0, 0, 0, a, 1, b, 1, tola, tolb,
                &dummyk, &dummyl, u, 1, v, 1, q, 1, iw, rw, tau, w,
                lwork, &info);
        chkxer("CGGSVP3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        cggsvp3("N", "/", "N", 0, 0, 0, a, 1, b, 1, tola, tolb,
                &dummyk, &dummyl, u, 1, v, 1, q, 1, iw, rw, tau, w,
                lwork, &info);
        chkxer("CGGSVP3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        cggsvp3("N", "N", "/", 0, 0, 0, a, 1, b, 1, tola, tolb,
                &dummyk, &dummyl, u, 1, v, 1, q, 1, iw, rw, tau, w,
                lwork, &info);
        chkxer("CGGSVP3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        cggsvp3("N", "N", "N", -1, 0, 0, a, 1, b, 1, tola, tolb,
                &dummyk, &dummyl, u, 1, v, 1, q, 1, iw, rw, tau, w,
                lwork, &info);
        chkxer("CGGSVP3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        cggsvp3("N", "N", "N", 0, -1, 0, a, 1, b, 1, tola, tolb,
                &dummyk, &dummyl, u, 1, v, 1, q, 1, iw, rw, tau, w,
                lwork, &info);
        chkxer("CGGSVP3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        cggsvp3("N", "N", "N", 0, 0, -1, a, 1, b, 1, tola, tolb,
                &dummyk, &dummyl, u, 1, v, 1, q, 1, iw, rw, tau, w,
                lwork, &info);
        chkxer("CGGSVP3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        cggsvp3("N", "N", "N", 2, 1, 1, a, 1, b, 1, tola, tolb,
                &dummyk, &dummyl, u, 1, v, 1, q, 1, iw, rw, tau, w,
                lwork, &info);
        chkxer("CGGSVP3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        cggsvp3("N", "N", "N", 1, 2, 1, a, 1, b, 1, tola, tolb,
                &dummyk, &dummyl, u, 1, v, 1, q, 1, iw, rw, tau, w,
                lwork, &info);
        chkxer("CGGSVP3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 16;
        cggsvp3("U", "N", "N", 2, 2, 2, a, 2, b, 2, tola, tolb,
                &dummyk, &dummyl, u, 1, v, 1, q, 1, iw, rw, tau, w,
                lwork, &info);
        chkxer("CGGSVP3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 18;
        cggsvp3("N", "V", "N", 2, 2, 2, a, 2, b, 2, tola, tolb,
                &dummyk, &dummyl, u, 2, v, 1, q, 1, iw, rw, tau, w,
                lwork, &info);
        chkxer("CGGSVP3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 20;
        cggsvp3("N", "N", "Q", 2, 2, 2, a, 2, b, 2, tola, tolb,
                &dummyk, &dummyl, u, 2, v, 2, q, 1, iw, rw, tau, w,
                lwork, &info);
        chkxer("CGGSVP3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 11;

        /* CTGSJA */

        strcpy(xerbla_srnamt, "CTGSJA");
        xerbla_infot = 1;
        ctgsja("/", "N", "N", 0, 0, 0, dummyk, dummyl, a, 1, b,
               1, tola, tolb, r1, r2, u, 1, v, 1, q, 1, w,
               &ncycle, &info);
        chkxer("CTGSJA", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        ctgsja("N", "/", "N", 0, 0, 0, dummyk, dummyl, a, 1, b,
               1, tola, tolb, r1, r2, u, 1, v, 1, q, 1, w,
               &ncycle, &info);
        chkxer("CTGSJA", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        ctgsja("N", "N", "/", 0, 0, 0, dummyk, dummyl, a, 1, b,
               1, tola, tolb, r1, r2, u, 1, v, 1, q, 1, w,
               &ncycle, &info);
        chkxer("CTGSJA", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        ctgsja("N", "N", "N", -1, 0, 0, dummyk, dummyl, a, 1, b,
               1, tola, tolb, r1, r2, u, 1, v, 1, q, 1, w,
               &ncycle, &info);
        chkxer("CTGSJA", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        ctgsja("N", "N", "N", 0, -1, 0, dummyk, dummyl, a, 1, b,
               1, tola, tolb, r1, r2, u, 1, v, 1, q, 1, w,
               &ncycle, &info);
        chkxer("CTGSJA", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        ctgsja("N", "N", "N", 0, 0, -1, dummyk, dummyl, a, 1, b,
               1, tola, tolb, r1, r2, u, 1, v, 1, q, 1, w,
               &ncycle, &info);
        chkxer("CTGSJA", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        ctgsja("N", "N", "N", 0, 0, 0, dummyk, dummyl, a, 0, b,
               1, tola, tolb, r1, r2, u, 1, v, 1, q, 1, w,
               &ncycle, &info);
        chkxer("CTGSJA", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 12;
        ctgsja("N", "N", "N", 0, 0, 0, dummyk, dummyl, a, 1, b,
               0, tola, tolb, r1, r2, u, 1, v, 1, q, 1, w,
               &ncycle, &info);
        chkxer("CTGSJA", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 18;
        ctgsja("U", "N", "N", 0, 0, 0, dummyk, dummyl, a, 1, b,
               1, tola, tolb, r1, r2, u, 0, v, 1, q, 1, w,
               &ncycle, &info);
        chkxer("CTGSJA", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 20;
        ctgsja("N", "V", "N", 0, 0, 0, dummyk, dummyl, a, 1, b,
               1, tola, tolb, r1, r2, u, 1, v, 0, q, 1, w,
               &ncycle, &info);
        chkxer("CTGSJA", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 22;
        ctgsja("N", "N", "Q", 0, 0, 0, dummyk, dummyl, a, 1, b,
               1, tola, tolb, r1, r2, u, 1, v, 1, q, 0, w,
               &ncycle, &info);
        chkxer("CTGSJA", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 11;

    /* Test error exits for the GLM path. */
    } else if ((path[0] == 'G' || path[0] == 'g') &&
               (path[1] == 'L' || path[1] == 'l') &&
               (path[2] == 'M' || path[2] == 'm')) {

        /* CGGGLM */

        strcpy(xerbla_srnamt, "CGGGLM");
        xerbla_infot = 1;
        cggglm(-1, 0, 0, a, 1, b, 1, tau, alpha, beta, w, LW,
               &info);
        chkxer("CGGGLM", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        cggglm(0, -1, 0, a, 1, b, 1, tau, alpha, beta, w, LW,
               &info);
        chkxer("CGGGLM", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        cggglm(0, 1, 0, a, 1, b, 1, tau, alpha, beta, w, LW,
               &info);
        chkxer("CGGGLM", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        cggglm(0, 0, -1, a, 1, b, 1, tau, alpha, beta, w, LW,
               &info);
        chkxer("CGGGLM", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        cggglm(1, 0, 0, a, 1, b, 1, tau, alpha, beta, w, LW,
               &info);
        chkxer("CGGGLM", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        cggglm(0, 0, 0, a, 0, b, 1, tau, alpha, beta, w, LW,
               &info);
        chkxer("CGGGLM", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        cggglm(0, 0, 0, a, 1, b, 0, tau, alpha, beta, w, LW,
               &info);
        chkxer("CGGGLM", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 12;
        cggglm(1, 1, 1, a, 1, b, 1, tau, alpha, beta, w, 1,
               &info);
        chkxer("CGGGLM", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 8;

    /* Test error exits for the LSE path. */
    } else if ((path[0] == 'L' || path[0] == 'l') &&
               (path[1] == 'S' || path[1] == 's') &&
               (path[2] == 'E' || path[2] == 'e')) {

        /* CGGLSE */

        strcpy(xerbla_srnamt, "CGGLSE");
        xerbla_infot = 1;
        cgglse(-1, 0, 0, a, 1, b, 1, tau, alpha, beta, w, LW,
               &info);
        chkxer("CGGLSE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        cgglse(0, -1, 0, a, 1, b, 1, tau, alpha, beta, w, LW,
               &info);
        chkxer("CGGLSE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        cgglse(0, 0, -1, a, 1, b, 1, tau, alpha, beta, w, LW,
               &info);
        chkxer("CGGLSE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        cgglse(0, 0, 1, a, 1, b, 1, tau, alpha, beta, w, LW,
               &info);
        chkxer("CGGLSE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        cgglse(0, 1, 0, a, 1, b, 1, tau, alpha, beta, w, LW,
               &info);
        chkxer("CGGLSE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        cgglse(0, 0, 0, a, 0, b, 1, tau, alpha, beta, w, LW,
               &info);
        chkxer("CGGLSE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        cgglse(0, 0, 0, a, 1, b, 0, tau, alpha, beta, w, LW,
               &info);
        chkxer("CGGLSE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 12;
        cgglse(1, 1, 1, a, 1, b, 1, tau, alpha, beta, w, 1,
               &info);
        chkxer("CGGLSE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 8;

    /* Test error exits for the CSD path. */
    } else if ((path[0] == 'C' || path[0] == 'c') &&
               (path[1] == 'S' || path[1] == 's') &&
               (path[2] == 'D' || path[2] == 'd')) {

        /* CUNCSD */

        strcpy(xerbla_srnamt, "CUNCSD");
        xerbla_infot = 7;
        cuncsd("Y", "Y", "Y", "Y", "N", "N",
               -1, 0, 0, a, 1, a,
               1, a, 1, a, 1, rs,
               a, 1, a, 1, a, 1, a,
               1, w, LW, rw, LW, iw, &info);
        chkxer("CUNCSD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        cuncsd("Y", "Y", "Y", "Y", "N", "N",
               1, -1, 0, a, 1, a,
               1, a, 1, a, 1, rs,
               a, 1, a, 1, a, 1, a,
               1, w, LW, rw, LW, iw, &info);
        chkxer("CUNCSD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        cuncsd("Y", "Y", "Y", "Y", "N", "N",
               1, 1, -1, a, 1, a,
               1, a, 1, a, 1, rs,
               a, 1, a, 1, a, 1, a,
               1, w, LW, rw, LW, iw, &info);
        chkxer("CUNCSD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 11;
        cuncsd("Y", "Y", "Y", "Y", "N", "N",
               1, 1, 1, a, -1, a,
               1, a, 1, a, 1, rs,
               a, 1, a, 1, a, 1, a,
               1, w, LW, rw, LW, iw, &info);
        chkxer("CUNCSD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 20;
        cuncsd("Y", "Y", "Y", "Y", "N", "N",
               1, 1, 1, a, 1, a,
               1, a, 1, a, 1, rs,
               a, -1, a, 1, a, 1, a,
               1, w, LW, rw, LW, iw, &info);
        chkxer("CUNCSD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 22;
        cuncsd("Y", "Y", "Y", "Y", "N", "N",
               1, 1, 1, a, 1, a,
               1, a, 1, a, 1, rs,
               a, 1, a, -1, a, 1, a,
               1, w, LW, rw, LW, iw, &info);
        chkxer("CUNCSD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 24;
        cuncsd("Y", "Y", "Y", "Y", "N", "N",
               1, 1, 1, a, 1, a,
               1, a, 1, a, 1, rs,
               a, 1, a, 1, a, -1, a,
               1, w, LW, rw, LW, iw, &info);
        chkxer("CUNCSD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 26;
        cuncsd("Y", "Y", "Y", "Y", "N", "N",
               1, 1, 1, a, 1, a,
               1, a, 1, a, 1, rs,
               a, 1, a, 1, a, 1, a,
               -1, w, LW, rw, LW, iw, &info);
        chkxer("CUNCSD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 8;

        /* CUNBDB */

        strcpy(xerbla_srnamt, "CUNBDB");
        xerbla_infot = 3;
        cunbdb("N", "N", -1, 0, 0, a, 1, b, 1, q, 1, z, 1,
               r1, r2, alpha, beta, tau, u, w, LW, &info);
        chkxer("CUNBDB", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        cunbdb("N", "N", 1, -1, 0, a, 1, b, 1, q, 1, z, 1,
               r1, r2, alpha, beta, tau, u, w, LW, &info);
        chkxer("CUNBDB", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        cunbdb("N", "N", 1, 1, -1, a, 1, b, 1, q, 1, z, 1,
               r1, r2, alpha, beta, tau, u, w, LW, &info);
        chkxer("CUNBDB", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 3;

    /* Test error exits for the GQR path. */
    } else if ((path[0] == 'G' || path[0] == 'g') &&
               (path[1] == 'Q' || path[1] == 'q') &&
               (path[2] == 'R' || path[2] == 'r')) {

        /* CGGQRF */

        strcpy(xerbla_srnamt, "CGGQRF");
        xerbla_infot = 1;
        cggqrf(-1, 0, 0, a, 1, alpha, b, 1, beta, w, LW, &info);
        chkxer("CGGQRF", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        cggqrf(0, -1, 0, a, 1, alpha, b, 1, beta, w, LW, &info);
        chkxer("CGGQRF", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        cggqrf(0, 0, -1, a, 1, alpha, b, 1, beta, w, LW, &info);
        chkxer("CGGQRF", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        cggqrf(0, 0, 0, a, 0, alpha, b, 1, beta, w, LW, &info);
        chkxer("CGGQRF", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        cggqrf(0, 0, 0, a, 1, alpha, b, 0, beta, w, LW, &info);
        chkxer("CGGQRF", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 11;
        cggqrf(1, 1, 2, a, 1, alpha, b, 1, beta, w, 1, &info);
        chkxer("CGGQRF", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 6;

        /* CGGRQF */

        strcpy(xerbla_srnamt, "CGGRQF");
        xerbla_infot = 1;
        cggrqf(-1, 0, 0, a, 1, alpha, b, 1, beta, w, LW, &info);
        chkxer("CGGRQF", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        cggrqf(0, -1, 0, a, 1, alpha, b, 1, beta, w, LW, &info);
        chkxer("CGGRQF", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        cggrqf(0, 0, -1, a, 1, alpha, b, 1, beta, w, LW, &info);
        chkxer("CGGRQF", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        cggrqf(0, 0, 0, a, 0, alpha, b, 1, beta, w, LW, &info);
        chkxer("CGGRQF", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        cggrqf(0, 0, 0, a, 1, alpha, b, 0, beta, w, LW, &info);
        chkxer("CGGRQF", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 11;
        cggrqf(1, 1, 2, a, 1, alpha, b, 1, beta, w, 1, &info);
        chkxer("CGGRQF", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 6;

    /* Test error exits for the CGS, CGV, CGX, and CXV paths. */
    } else if (((path[0] == 'C' || path[0] == 'c') &&
                (path[1] == 'G' || path[1] == 'g') &&
                (path[2] == 'S' || path[2] == 's')) ||
               ((path[0] == 'C' || path[0] == 'c') &&
                (path[1] == 'G' || path[1] == 'g') &&
                (path[2] == 'V' || path[2] == 'v')) ||
               ((path[0] == 'C' || path[0] == 'c') &&
                (path[1] == 'G' || path[1] == 'g') &&
                (path[2] == 'X' || path[2] == 'x')) ||
               ((path[0] == 'C' || path[0] == 'c') &&
                (path[1] == 'X' || path[1] == 'x') &&
                (path[2] == 'V' || path[2] == 'v'))) {

        /* CGGES */

        strcpy(xerbla_srnamt, "CGGES ");
        xerbla_infot = 1;
        cgges("/", "N", "S", clctes, 1, a, 1, b, 1, &sdim, alpha,
              beta, q, 1, u, 1, w, 1, rw, bw, &info);
        chkxer("CGGES ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        cgges("N", "/", "S", clctes, 1, a, 1, b, 1, &sdim, alpha,
              beta, q, 1, u, 1, w, 1, rw, bw, &info);
        chkxer("CGGES ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        cgges("N", "V", "/", clctes, 1, a, 1, b, 1, &sdim, alpha,
              beta, q, 1, u, 1, w, 1, rw, bw, &info);
        chkxer("CGGES ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        cgges("N", "V", "S", clctes, -1, a, 1, b, 1, &sdim, alpha,
              beta, q, 1, u, 1, w, 1, rw, bw, &info);
        chkxer("CGGES ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        cgges("N", "V", "S", clctes, 1, a, 0, b, 1, &sdim, alpha,
              beta, q, 1, u, 1, w, 1, rw, bw, &info);
        chkxer("CGGES ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        cgges("N", "V", "S", clctes, 1, a, 1, b, 0, &sdim, alpha,
              beta, q, 1, u, 1, w, 1, rw, bw, &info);
        chkxer("CGGES ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 14;
        cgges("N", "V", "S", clctes, 1, a, 1, b, 1, &sdim, alpha,
              beta, q, 0, u, 1, w, 1, rw, bw, &info);
        chkxer("CGGES ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 14;
        cgges("V", "V", "S", clctes, 2, a, 2, b, 2, &sdim, alpha,
              beta, q, 1, u, 2, w, 1, rw, bw, &info);
        chkxer("CGGES ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 16;
        cgges("N", "V", "S", clctes, 1, a, 1, b, 1, &sdim, alpha,
              beta, q, 1, u, 0, w, 1, rw, bw, &info);
        chkxer("CGGES ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 16;
        cgges("V", "V", "S", clctes, 2, a, 2, b, 2, &sdim, alpha,
              beta, q, 2, u, 1, w, 1, rw, bw, &info);
        chkxer("CGGES ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 18;
        cgges("V", "V", "S", clctes, 2, a, 2, b, 2, &sdim, alpha,
              beta, q, 2, u, 2, w, 1, rw, bw, &info);
        chkxer("CGGES ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 11;

        /* CGGES3 */

        strcpy(xerbla_srnamt, "CGGES3");
        xerbla_infot = 1;
        cgges3("/", "N", "S", clctes, 1, a, 1, b, 1, &sdim, alpha,
               beta, q, 1, u, 1, w, 1, rw, bw, &info);
        chkxer("CGGES3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        cgges3("N", "/", "S", clctes, 1, a, 1, b, 1, &sdim, alpha,
               beta, q, 1, u, 1, w, 1, rw, bw, &info);
        chkxer("CGGES3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        cgges3("N", "V", "/", clctes, 1, a, 1, b, 1, &sdim, alpha,
               beta, q, 1, u, 1, w, 1, rw, bw, &info);
        chkxer("CGGES3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        cgges3("N", "V", "S", clctes, -1, a, 1, b, 1, &sdim,
               alpha, beta, q, 1, u, 1, w, 1, rw, bw, &info);
        chkxer("CGGES3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        cgges3("N", "V", "S", clctes, 1, a, 0, b, 1, &sdim, alpha,
               beta, q, 1, u, 1, w, 1, rw, bw, &info);
        chkxer("CGGES3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        cgges3("N", "V", "S", clctes, 1, a, 1, b, 0, &sdim, alpha,
               beta, q, 1, u, 1, w, 1, rw, bw, &info);
        chkxer("CGGES3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 14;
        cgges3("N", "V", "S", clctes, 1, a, 1, b, 1, &sdim, alpha,
               beta, q, 0, u, 1, w, 1, rw, bw, &info);
        chkxer("CGGES3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 14;
        cgges3("V", "V", "S", clctes, 2, a, 2, b, 2, &sdim, alpha,
               beta, q, 1, u, 2, w, 1, rw, bw, &info);
        chkxer("CGGES3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 16;
        cgges3("N", "V", "S", clctes, 1, a, 1, b, 1, &sdim, alpha,
               beta, q, 1, u, 0, w, 1, rw, bw, &info);
        chkxer("CGGES3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 16;
        cgges3("V", "V", "S", clctes, 2, a, 2, b, 2, &sdim, alpha,
               beta, q, 2, u, 1, w, 1, rw, bw, &info);
        chkxer("CGGES3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 18;
        cgges3("V", "V", "S", clctes, 2, a, 2, b, 2, &sdim, alpha,
               beta, q, 2, u, 2, w, 1, rw, bw, &info);
        chkxer("CGGES3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 11;

        /* CGGESX */

        strcpy(xerbla_srnamt, "CGGESX");
        xerbla_infot = 1;
        cggesx("/", "N", "S", clctsx, "N", 1, a, 1, b, 1, &sdim,
               alpha, beta, q, 1, u, 1, rce, rcv, w, 1, rw, iw,
               1, bw, &info);
        chkxer("CGGESX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        cggesx("N", "/", "S", clctsx, "N", 1, a, 1, b, 1, &sdim,
               alpha, beta, q, 1, u, 1, rce, rcv, w, 1, rw, iw,
               1, bw, &info);
        chkxer("CGGESX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        cggesx("V", "V", "/", clctsx, "N", 1, a, 1, b, 1, &sdim,
               alpha, beta, q, 1, u, 1, rce, rcv, w, 1, rw, iw,
               1, bw, &info);
        chkxer("CGGESX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        cggesx("V", "V", "S", clctsx, "/", 1, a, 1, b, 1, &sdim,
               alpha, beta, q, 1, u, 1, rce, rcv, w, 1, rw, iw,
               1, bw, &info);
        chkxer("CGGESX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        cggesx("V", "V", "S", clctsx, "B", -1, a, 1, b, 1, &sdim,
               alpha, beta, q, 1, u, 1, rce, rcv, w, 1, rw, iw,
               1, bw, &info);
        chkxer("CGGESX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        cggesx("V", "V", "S", clctsx, "B", 1, a, 0, b, 1, &sdim,
               alpha, beta, q, 1, u, 1, rce, rcv, w, 1, rw, iw,
               1, bw, &info);
        chkxer("CGGESX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        cggesx("V", "V", "S", clctsx, "B", 1, a, 1, b, 0, &sdim,
               alpha, beta, q, 1, u, 1, rce, rcv, w, 1, rw, iw,
               1, bw, &info);
        chkxer("CGGESX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 15;
        cggesx("V", "V", "S", clctsx, "B", 1, a, 1, b, 1, &sdim,
               alpha, beta, q, 0, u, 1, rce, rcv, w, 1, rw, iw,
               1, bw, &info);
        chkxer("CGGESX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 15;
        cggesx("V", "V", "S", clctsx, "B", 2, a, 2, b, 2, &sdim,
               alpha, beta, q, 1, u, 1, rce, rcv, w, 1, rw, iw,
               1, bw, &info);
        chkxer("CGGESX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 17;
        cggesx("V", "V", "S", clctsx, "B", 1, a, 1, b, 1, &sdim,
               alpha, beta, q, 1, u, 0, rce, rcv, w, 1, rw, iw,
               1, bw, &info);
        chkxer("CGGESX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 17;
        cggesx("V", "V", "S", clctsx, "B", 2, a, 2, b, 2, &sdim,
               alpha, beta, q, 2, u, 1, rce, rcv, w, 1, rw, iw,
               1, bw, &info);
        chkxer("CGGESX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 21;
        cggesx("V", "V", "S", clctsx, "B", 2, a, 2, b, 2, &sdim,
               alpha, beta, q, 2, u, 2, rce, rcv, w, 1, rw, iw,
               1, bw, &info);
        chkxer("CGGESX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 24;
        cggesx("V", "V", "S", clctsx, "V", 1, a, 1, b, 1, &sdim,
               alpha, beta, q, 1, u, 1, rce, rcv, w, 32, rw, iw,
               0, bw, &info);
        chkxer("CGGESX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 13;

        /* CGGEV */

        strcpy(xerbla_srnamt, "CGGEV ");
        xerbla_infot = 1;
        cggev("/", "N", 1, a, 1, b, 1, alpha, beta, q, 1, u, 1,
              w, 1, rw, &info);
        chkxer("CGGEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        cggev("N", "/", 1, a, 1, b, 1, alpha, beta, q, 1, u, 1,
              w, 1, rw, &info);
        chkxer("CGGEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        cggev("V", "V", -1, a, 1, b, 1, alpha, beta, q, 1, u, 1,
              w, 1, rw, &info);
        chkxer("CGGEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        cggev("V", "V", 1, a, 0, b, 1, alpha, beta, q, 1, u, 1,
              w, 1, rw, &info);
        chkxer("CGGEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        cggev("V", "V", 1, a, 1, b, 0, alpha, beta, q, 1, u, 1,
              w, 1, rw, &info);
        chkxer("CGGEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 11;
        cggev("N", "V", 1, a, 1, b, 1, alpha, beta, q, 0, u, 1,
              w, 1, rw, &info);
        chkxer("CGGEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 11;
        cggev("V", "V", 2, a, 2, b, 2, alpha, beta, q, 1, u, 2,
              w, 1, rw, &info);
        chkxer("CGGEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 13;
        cggev("V", "N", 2, a, 2, b, 2, alpha, beta, q, 2, u, 0,
              w, 1, rw, &info);
        chkxer("CGGEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 13;
        cggev("V", "V", 2, a, 2, b, 2, alpha, beta, q, 2, u, 1,
              w, 1, rw, &info);
        chkxer("CGGEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 15;
        cggev("V", "V", 1, a, 1, b, 1, alpha, beta, q, 1, u, 1,
              w, 1, rw, &info);
        chkxer("CGGEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 10;

        /* CGGEV3 */

        strcpy(xerbla_srnamt, "CGGEV3");
        xerbla_infot = 1;
        cggev3("/", "N", 1, a, 1, b, 1, alpha, beta, q, 1, u, 1,
               w, 1, rw, &info);
        chkxer("CGGEV3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        cggev3("N", "/", 1, a, 1, b, 1, alpha, beta, q, 1, u, 1,
               w, 1, rw, &info);
        chkxer("CGGEV3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        cggev3("V", "V", -1, a, 1, b, 1, alpha, beta, q, 1, u, 1,
               w, 1, rw, &info);
        chkxer("CGGEV3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        cggev3("V", "V", 1, a, 0, b, 1, alpha, beta, q, 1, u, 1,
               w, 1, rw, &info);
        chkxer("CGGEV3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        cggev3("V", "V", 1, a, 1, b, 0, alpha, beta, q, 1, u, 1,
               w, 1, rw, &info);
        chkxer("CGGEV3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 11;
        cggev3("N", "V", 1, a, 1, b, 1, alpha, beta, q, 0, u, 1,
               w, 1, rw, &info);
        chkxer("CGGEV3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 11;
        cggev3("V", "V", 2, a, 2, b, 2, alpha, beta, q, 1, u, 2,
               w, 1, rw, &info);
        chkxer("CGGEV3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 13;
        cggev3("V", "N", 2, a, 2, b, 2, alpha, beta, q, 2, u, 0,
               w, 1, rw, &info);
        chkxer("CGGEV3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 13;
        cggev3("V", "V", 2, a, 2, b, 2, alpha, beta, q, 2, u, 1,
               w, 1, rw, &info);
        chkxer("CGGEV3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 15;
        cggev3("V", "V", 1, a, 1, b, 1, alpha, beta, q, 1, u, 1,
               w, 1, rw, &info);
        chkxer("CGGEV3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 10;

        /* CGGEVX */

        strcpy(xerbla_srnamt, "CGGEVX");
        xerbla_infot = 1;
        cggevx("/", "N", "N", "N", 1, a, 1, b, 1, alpha, beta, q,
               1, u, 1, &ilo, &ihi, ls, rs, &anrm, &bnrm, rce, rcv,
               w, 1, rw, iw, bw, &info);
        chkxer("CGGEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        cggevx("N", "/", "N", "N", 1, a, 1, b, 1, alpha, beta, q,
               1, u, 1, &ilo, &ihi, ls, rs, &anrm, &bnrm, rce, rcv,
               w, 1, rw, iw, bw, &info);
        chkxer("CGGEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        cggevx("N", "N", "/", "N", 1, a, 1, b, 1, alpha, beta, q,
               1, u, 1, &ilo, &ihi, ls, rs, &anrm, &bnrm, rce, rcv,
               w, 1, rw, iw, bw, &info);
        chkxer("CGGEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        cggevx("N", "N", "N", "/", 1, a, 1, b, 1, alpha, beta, q,
               1, u, 1, &ilo, &ihi, ls, rs, &anrm, &bnrm, rce, rcv,
               w, 1, rw, iw, bw, &info);
        chkxer("CGGEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        cggevx("N", "N", "N", "N", -1, a, 1, b, 1, alpha, beta,
               q, 1, u, 1, &ilo, &ihi, ls, rs, &anrm, &bnrm, rce,
               rcv, w, 1, rw, iw, bw, &info);
        chkxer("CGGEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        cggevx("N", "N", "N", "N", 1, a, 0, b, 1, alpha, beta, q,
               1, u, 1, &ilo, &ihi, ls, rs, &anrm, &bnrm, rce, rcv,
               w, 1, rw, iw, bw, &info);
        chkxer("CGGEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        cggevx("N", "N", "N", "N", 1, a, 1, b, 0, alpha, beta, q,
               1, u, 1, &ilo, &ihi, ls, rs, &anrm, &bnrm, rce, rcv,
               w, 1, rw, iw, bw, &info);
        chkxer("CGGEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 13;
        cggevx("N", "N", "N", "N", 1, a, 1, b, 1, alpha, beta, q,
               0, u, 1, &ilo, &ihi, ls, rs, &anrm, &bnrm, rce, rcv,
               w, 1, rw, iw, bw, &info);
        chkxer("CGGEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 13;
        cggevx("N", "V", "N", "N", 2, a, 2, b, 2, alpha, beta, q,
               1, u, 2, &ilo, &ihi, ls, rs, &anrm, &bnrm, rce, rcv,
               w, 1, rw, iw, bw, &info);
        chkxer("CGGEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 15;
        cggevx("N", "N", "N", "N", 1, a, 1, b, 1, alpha, beta, q,
               1, u, 0, &ilo, &ihi, ls, rs, &anrm, &bnrm, rce, rcv,
               w, 1, rw, iw, bw, &info);
        chkxer("CGGEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 15;
        cggevx("N", "N", "V", "N", 2, a, 2, b, 2, alpha, beta, q,
               2, u, 1, &ilo, &ihi, ls, rs, &anrm, &bnrm, rce, rcv,
               w, 1, rw, iw, bw, &info);
        chkxer("CGGEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 25;
        cggevx("N", "N", "V", "N", 2, a, 2, b, 2, alpha, beta, q,
               2, u, 2, &ilo, &ihi, ls, rs, &anrm, &bnrm, rce, rcv,
               w, 0, rw, iw, bw, &info);
        chkxer("CGGEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 12;

        /* CTGEXC */

        strcpy(xerbla_srnamt, "CTGEXC");
        xerbla_infot = 3;
        ctgexc(1, 1, -1, a, 1, b, 1, q, 1, z, 1, ifst,
               &ilst, &info);
        chkxer("CTGEXC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        ctgexc(1, 1, 1, a, 0, b, 1, q, 1, z, 1, ifst,
               &ilst, &info);
        chkxer("CTGEXC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        ctgexc(1, 1, 1, a, 1, b, 0, q, 1, z, 1, ifst,
               &ilst, &info);
        chkxer("CTGEXC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        ctgexc(0, 1, 1, a, 1, b, 1, q, 0, z, 1, ifst,
               &ilst, &info);
        chkxer("CTGEXC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        ctgexc(1, 1, 1, a, 1, b, 1, q, 0, z, 1, ifst,
               &ilst, &info);
        chkxer("CTGEXC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 11;
        ctgexc(1, 0, 1, a, 1, b, 1, q, 1, z, 0, ifst,
               &ilst, &info);
        chkxer("CTGEXC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 11;
        ctgexc(1, 1, 1, a, 1, b, 1, q, 1, z, 0, ifst,
               &ilst, &info);
        chkxer("CTGEXC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 7;

        /* CTGSEN */

        strcpy(xerbla_srnamt, "CTGSEN");
        xerbla_infot = 1;
        ctgsen(-1, 1, 1, sel, 1, a, 1, b, 1, alpha,
               beta, q, 1, z, 1, &m, &tola, &tolb, rcv, w, 1, iw, 1,
               &info);
        chkxer("CTGSEN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        ctgsen(1, 1, 1, sel, -1, a, 1, b, 1, alpha,
               beta, q, 1, z, 1, &m, &tola, &tolb, rcv, w, 1, iw, 1,
               &info);
        chkxer("CTGSEN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        ctgsen(1, 1, 1, sel, 1, a, 0, b, 1, alpha,
               beta, q, 1, z, 1, &m, &tola, &tolb, rcv, w, 1, iw, 1,
               &info);
        chkxer("CTGSEN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        ctgsen(1, 1, 1, sel, 1, a, 1, b, 0, alpha,
               beta, q, 1, z, 1, &m, &tola, &tolb, rcv, w, 1, iw, 1,
               &info);
        chkxer("CTGSEN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 13;
        ctgsen(1, 1, 1, sel, 1, a, 1, b, 1, alpha,
               beta, q, 0, z, 1, &m, &tola, &tolb, rcv, w, 1, iw, 1,
               &info);
        chkxer("CTGSEN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 15;
        ctgsen(1, 1, 1, sel, 1, a, 1, b, 1, alpha,
               beta, q, 1, z, 0, &m, &tola, &tolb, rcv, w, 1, iw, 1,
               &info);
        chkxer("CTGSEN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 21;
        ctgsen(3, 1, 1, sel, 1, a, 1, b, 1, alpha,
               beta, q, 1, z, 1, &m, &tola, &tolb, rcv, w, -5, iw,
               1, &info);
        chkxer("CTGSEN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 23;
        ctgsen(0, 1, 1, sel, 1, a, 1, b, 1, alpha,
               beta, q, 1, z, 1, &m, &tola, &tolb, rcv, w, 20, iw,
               0, &info);
        chkxer("CTGSEN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 23;
        ctgsen(1, 1, 1, sel, 1, a, 1, b, 1, alpha,
               beta, q, 1, z, 1, &m, &tola, &tolb, rcv, w, 20, iw,
               0, &info);
        chkxer("CTGSEN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 23;
        ctgsen(5, 1, 1, sel, 1, a, 1, b, 1, alpha,
               beta, q, 1, z, 1, &m, &tola, &tolb, rcv, w, 20, iw,
               1, &info);
        chkxer("CTGSEN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 11;

        /* CTGSNA */

        strcpy(xerbla_srnamt, "CTGSNA");
        xerbla_infot = 1;
        ctgsna("/", "A", sel, 1, a, 1, b, 1, q, 1, u, 1, r1, r2,
               1, &m, w, 1, iw, &info);
        chkxer("CTGSNA", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        ctgsna("B", "/", sel, 1, a, 1, b, 1, q, 1, u, 1, r1, r2,
               1, &m, w, 1, iw, &info);
        chkxer("CTGSNA", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        ctgsna("B", "A", sel, -1, a, 1, b, 1, q, 1, u, 1, r1, r2,
               1, &m, w, 1, iw, &info);
        chkxer("CTGSNA", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        ctgsna("B", "A", sel, 1, a, 0, b, 1, q, 1, u, 1, r1, r2,
               1, &m, w, 1, iw, &info);
        chkxer("CTGSNA", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        ctgsna("B", "A", sel, 1, a, 1, b, 0, q, 1, u, 1, r1, r2,
               1, &m, w, 1, iw, &info);
        chkxer("CTGSNA", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        ctgsna("E", "A", sel, 1, a, 1, b, 1, q, 0, u, 1, r1, r2,
               1, &m, w, 1, iw, &info);
        chkxer("CTGSNA", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 12;
        ctgsna("E", "A", sel, 1, a, 1, b, 1, q, 1, u, 0, r1, r2,
               1, &m, w, 1, iw, &info);
        chkxer("CTGSNA", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 15;
        ctgsna("E", "A", sel, 1, a, 1, b, 1, q, 1, u, 1, r1, r2,
               0, &m, w, 1, iw, &info);
        chkxer("CTGSNA", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 18;
        ctgsna("E", "A", sel, 1, a, 1, b, 1, q, 1, u, 1, r1, r2,
               1, &m, w, 0, iw, &info);
        chkxer("CTGSNA", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 9;

        /* CTGSYL */

        strcpy(xerbla_srnamt, "CTGSYL");
        xerbla_infot = 1;
        ctgsyl("/", 0, 1, 1, a, 1, b, 1, q, 1, u, 1, v, 1, z, 1,
               &scale, &dif, w, 1, iw, &info);
        chkxer("CTGSYL", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        ctgsyl("N", -1, 1, 1, a, 1, b, 1, q, 1, u, 1, v, 1, z, 1,
               &scale, &dif, w, 1, iw, &info);
        chkxer("CTGSYL", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        ctgsyl("N", 0, 0, 1, a, 1, b, 1, q, 1, u, 1, v, 1, z, 1,
               &scale, &dif, w, 1, iw, &info);
        chkxer("CTGSYL", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        ctgsyl("N", 0, 1, 0, a, 1, b, 1, q, 1, u, 1, v, 1, z, 1,
               &scale, &dif, w, 1, iw, &info);
        chkxer("CTGSYL", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        ctgsyl("N", 0, 1, 1, a, 0, b, 1, q, 1, u, 1, v, 1, z, 1,
               &scale, &dif, w, 1, iw, &info);
        chkxer("CTGSYL", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        ctgsyl("N", 0, 1, 1, a, 1, b, 0, q, 1, u, 1, v, 1, z, 1,
               &scale, &dif, w, 1, iw, &info);
        chkxer("CTGSYL", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        ctgsyl("N", 0, 1, 1, a, 1, b, 1, q, 0, u, 1, v, 1, z, 1,
               &scale, &dif, w, 1, iw, &info);
        chkxer("CTGSYL", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 12;
        ctgsyl("N", 0, 1, 1, a, 1, b, 1, q, 1, u, 0, v, 1, z, 1,
               &scale, &dif, w, 1, iw, &info);
        chkxer("CTGSYL", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 14;
        ctgsyl("N", 0, 1, 1, a, 1, b, 1, q, 1, u, 1, v, 0, z, 1,
               &scale, &dif, w, 1, iw, &info);
        chkxer("CTGSYL", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 16;
        ctgsyl("N", 0, 1, 1, a, 1, b, 1, q, 1, u, 1, v, 1, z, 0,
               &scale, &dif, w, 1, iw, &info);
        chkxer("CTGSYL", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 20;
        ctgsyl("N", 1, 1, 1, a, 1, b, 1, q, 1, u, 1, v, 1, z, 1,
               &scale, &dif, w, 1, iw, &info);
        chkxer("CTGSYL", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 20;
        ctgsyl("N", 2, 1, 1, a, 1, b, 1, q, 1, u, 1, v, 1, z, 1,
               &scale, &dif, w, 1, iw, &info);
        chkxer("CTGSYL", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 12;
    }

    *ok = xerbla_ok;

    xerbla_srnamt[0] = '\0';
    xerbla_infot = 0;
}
