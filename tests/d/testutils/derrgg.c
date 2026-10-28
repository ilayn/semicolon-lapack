/**
 * @file derrgg.c
 * @brief DERRGG tests the error exits for the generalized nonsymmetric
 *        eigenvalue, GSVD, CSD, GLM, LSE and GQR routines.
 *
 * Port of LAPACK TESTING/EIG/derrgg.f
 * Tests error exits for: DGGES, DGGESX, DGGEV, DGGEVX, DGGGLM, DGGHRD,
 * DGGLSE, DGGQRF, DGGRQF, DGGSVD3, DGGSVP3, DHGEQZ, DORBDB, DORCSD,
 * DTGEVC, DTGEXC, DTGSEN, DTGSJA, DTGSNA, DGGES3, DGGEV3, and DTGSYL.
 */

#define LAPACK_TUNING_TEST
#include <math.h>
#include <string.h>
#include "semicolon_lapack_double.h"
#include "verify.h"

/*
 * Port of LAPACK TESTING/EIG/dlctes.f
 */
static INT dlctes(const f64* zr, const f64* zi, const f64* d)
{
    (void)zi;
    if (*d == 0.0) {
        return (*zr < 0.0);
    } else {
        return (copysign(1.0, *zr) != copysign(1.0, *d));
    }
}

/*
 * Port of LAPACK TESTING/EIG/dlctsx.f
 */
static INT dlctsx_m, dlctsx_n, dlctsx_mplusn, dlctsx_i, dlctsx_fs;

static INT dlctsx(const f64* ar, const f64* ai, const f64* beta)
{
    (void)ar;
    (void)ai;
    (void)beta;
    INT res;
    if (dlctsx_fs) {
        dlctsx_i = dlctsx_i + 1;
        if (dlctsx_i <= dlctsx_m) {
            res = 0;
        } else {
            res = 1;
        }
        if (dlctsx_i == dlctsx_mplusn) {
            dlctsx_fs = 0;
            dlctsx_i = 0;
        }
    } else {
        dlctsx_i = dlctsx_i + 1;
        if (dlctsx_i <= dlctsx_n) {
            res = 1;
        } else {
            res = 0;
        }
        if (dlctsx_i == dlctsx_mplusn) {
            dlctsx_fs = 1;
            dlctsx_i = 0;
        }
    }
    return res;
}

void derrgg(const char* path, INT* ok, INT* nt)
{
    enum { NMAX = 3, LW = 6 * NMAX };
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    char c2[2];
    INT dummyk = 0, dummyl = 0;
    INT i, ifst, ilo, ihi, ilst, info, j, m, ncycle, sdim, lwork;
    f64 anrm, bnrm, dif, scale, tola, tolb;

    INT bw[NMAX], sel[NMAX];
    INT iw[NMAX], idum[NMAX];
    f64 a[NMAX * NMAX], b[NMAX * NMAX], ls[NMAX];
    f64 q[NMAX * NMAX], r1[NMAX], r2[NMAX];
    f64 r3[NMAX], rce[2], rcv[2], rs[NMAX];
    f64 tau[NMAX], u[NMAX * NMAX], v[NMAX * NMAX];
    f64 w[LW], z[NMAX * NMAX];

    c2[0] = path[1];
    c2[1] = path[2];

    /* Set the variables to innocuous values. */
    for (j = 0; j < NMAX; j++) {
        sel[j] = 1;
        for (i = 0; i < NMAX; i++) {
            a[i + j * NMAX] = ZERO;
            b[i + j * NMAX] = ZERO;
        }
    }
    for (i = 0; i < NMAX; i++) {
        a[i + i * NMAX] = ONE;
        b[i + i * NMAX] = ONE;
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

        /* DGGHRD */

        strcpy(xerbla_srnamt, "DGGHRD");
        xerbla_infot = 1;
        dgghrd("/", "N", 0, 0, -1, a, 1, b, 1, q, 1, z, 1, &info);
        chkxer("DGGHRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        dgghrd("N", "/", 0, 0, -1, a, 1, b, 1, q, 1, z, 1, &info);
        chkxer("DGGHRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        dgghrd("N", "N", -1, -1, -1, a, 1, b, 1, q, 1, z, 1, &info);
        chkxer("DGGHRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        dgghrd("N", "N", 0, -1, -1, a, 1, b, 1, q, 1, z, 1, &info);
        chkxer("DGGHRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        dgghrd("N", "N", 0, 0, 0, a, 1, b, 1, q, 1, z, 1, &info);
        chkxer("DGGHRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        dgghrd("N", "N", 2, 0, 0, a, 1, b, 2, q, 1, z, 1, &info);
        chkxer("DGGHRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        dgghrd("N", "N", 2, 0, 0, a, 2, b, 1, q, 1, z, 1, &info);
        chkxer("DGGHRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 11;
        dgghrd("V", "N", 2, 0, 0, a, 2, b, 2, q, 1, z, 1, &info);
        chkxer("DGGHRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 13;
        dgghrd("N", "V", 2, 0, 0, a, 2, b, 2, q, 1, z, 1, &info);
        chkxer("DGGHRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 9;

        /* DGGHD3 */

        strcpy(xerbla_srnamt, "DGGHD3");
        xerbla_infot = 1;
        dgghd3("/", "N", 0, 0, -1, a, 1, b, 1, q, 1, z, 1, w, LW,
               &info);
        chkxer("DGGHD3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        dgghd3("N", "/", 0, 0, -1, a, 1, b, 1, q, 1, z, 1, w, LW,
               &info);
        chkxer("DGGHD3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        dgghd3("N", "N", -1, -1, -1, a, 1, b, 1, q, 1, z, 1, w, LW,
               &info);
        chkxer("DGGHD3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        dgghd3("N", "N", 0, -1, -1, a, 1, b, 1, q, 1, z, 1, w, LW,
               &info);
        chkxer("DGGHD3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        dgghd3("N", "N", 0, 0, 0, a, 1, b, 1, q, 1, z, 1, w, LW,
               &info);
        chkxer("DGGHD3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        dgghd3("N", "N", 2, 0, 0, a, 1, b, 2, q, 1, z, 1, w, LW,
               &info);
        chkxer("DGGHD3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        dgghd3("N", "N", 2, 0, 0, a, 2, b, 1, q, 1, z, 1, w, LW,
               &info);
        chkxer("DGGHD3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 11;
        dgghd3("V", "N", 2, 0, 0, a, 2, b, 2, q, 1, z, 1, w, LW,
               &info);
        chkxer("DGGHD3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 13;
        dgghd3("N", "V", 2, 0, 0, a, 2, b, 2, q, 1, z, 1, w, LW,
               &info);
        chkxer("DGGHD3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 9;

        /* DHGEQZ */

        strcpy(xerbla_srnamt, "DHGEQZ");
        xerbla_infot = 1;
        dhgeqz("/", "N", "N", 0, 0, -1, a, 1, b, 1, r1, r2, r3, q,
               1, z, 1, w, LW, &info);
        chkxer("DHGEQZ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        dhgeqz("E", "/", "N", 0, 0, -1, a, 1, b, 1, r1, r2, r3, q,
               1, z, 1, w, LW, &info);
        chkxer("DHGEQZ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        dhgeqz("E", "N", "/", 0, 0, -1, a, 1, b, 1, r1, r2, r3, q,
               1, z, 1, w, LW, &info);
        chkxer("DHGEQZ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        dhgeqz("E", "N", "N", -1, -1, -1, a, 1, b, 1, r1, r2, r3,
               q, 1, z, 1, w, LW, &info);
        chkxer("DHGEQZ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        dhgeqz("E", "N", "N", 0, -1, -1, a, 1, b, 1, r1, r2, r3, q,
               1, z, 1, w, LW, &info);
        chkxer("DHGEQZ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        dhgeqz("E", "N", "N", 0, 0, 0, a, 1, b, 1, r1, r2, r3, q,
               1, z, 1, w, LW, &info);
        chkxer("DHGEQZ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        dhgeqz("E", "N", "N", 2, 0, 0, a, 1, b, 2, r1, r2, r3, q,
               1, z, 1, w, LW, &info);
        chkxer("DHGEQZ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        dhgeqz("E", "N", "N", 2, 0, 0, a, 2, b, 1, r1, r2, r3, q,
               1, z, 1, w, LW, &info);
        chkxer("DHGEQZ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 15;
        dhgeqz("E", "V", "N", 2, 0, 0, a, 2, b, 2, r1, r2, r3, q,
               1, z, 1, w, LW, &info);
        chkxer("DHGEQZ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 17;
        dhgeqz("E", "N", "V", 2, 0, 0, a, 2, b, 2, r1, r2, r3, q,
               1, z, 1, w, LW, &info);
        chkxer("DHGEQZ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 10;

        /* DTGEVC */

        strcpy(xerbla_srnamt, "DTGEVC");
        xerbla_infot = 1;
        dtgevc("/", "A", sel, 0, a, 1, b, 1, q, 1, z, 1, 0, &m, w,
               &info);
        chkxer("DTGEVC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        dtgevc("R", "/", sel, 0, a, 1, b, 1, q, 1, z, 1, 0, &m, w,
               &info);
        chkxer("DTGEVC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        dtgevc("R", "A", sel, -1, a, 1, b, 1, q, 1, z, 1, 0, &m,
               w, &info);
        chkxer("DTGEVC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        dtgevc("R", "A", sel, 2, a, 1, b, 2, q, 1, z, 2, 0, &m, w,
               &info);
        chkxer("DTGEVC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        dtgevc("R", "A", sel, 2, a, 2, b, 1, q, 1, z, 2, 0, &m, w,
               &info);
        chkxer("DTGEVC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        dtgevc("L", "A", sel, 2, a, 2, b, 2, q, 1, z, 1, 0, &m, w,
               &info);
        chkxer("DTGEVC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 12;
        dtgevc("R", "A", sel, 2, a, 2, b, 2, q, 1, z, 1, 0, &m, w,
               &info);
        chkxer("DTGEVC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 13;
        dtgevc("R", "A", sel, 2, a, 2, b, 2, q, 1, z, 2, 1, &m, w,
               &info);
        chkxer("DTGEVC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 8;

    /* Test error exits for the GSV path. */
    } else if ((path[0] == 'G' || path[0] == 'g') &&
               (path[1] == 'S' || path[1] == 's') &&
               (path[2] == 'V' || path[2] == 'v')) {

        /* DGGSVD3 */

        strcpy(xerbla_srnamt, "DGGSVD3");
        xerbla_infot = 1;
        dggsvd3("/", "N", "N", 0, 0, 0, &dummyk, &dummyl, a, 1, b,
                1, r1, r2, u, 1, v, 1, q, 1, w, lwork, idum, &info);
        chkxer("DGGSVD3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        dggsvd3("N", "/", "N", 0, 0, 0, &dummyk, &dummyl, a, 1, b,
                1, r1, r2, u, 1, v, 1, q, 1, w, lwork, idum, &info);
        chkxer("DGGSVD3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        dggsvd3("N", "N", "/", 0, 0, 0, &dummyk, &dummyl, a, 1, b,
                1, r1, r2, u, 1, v, 1, q, 1, w, lwork, idum, &info);
        chkxer("DGGSVD3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        dggsvd3("N", "N", "N", -1, 0, 0, &dummyk, &dummyl, a, 1, b,
                1, r1, r2, u, 1, v, 1, q, 1, w, lwork, idum, &info);
        chkxer("DGGSVD3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        dggsvd3("N", "N", "N", 0, -1, 0, &dummyk, &dummyl, a, 1, b,
                1, r1, r2, u, 1, v, 1, q, 1, w, lwork, idum, &info);
        chkxer("DGGSVD3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        dggsvd3("N", "N", "N", 0, 0, -1, &dummyk, &dummyl, a, 1, b,
                1, r1, r2, u, 1, v, 1, q, 1, w, lwork, idum, &info);
        chkxer("DGGSVD3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        dggsvd3("N", "N", "N", 2, 1, 1, &dummyk, &dummyl, a, 1, b,
                1, r1, r2, u, 1, v, 1, q, 1, w, lwork, idum, &info);
        chkxer("DGGSVD3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 12;
        dggsvd3("N", "N", "N", 1, 1, 2, &dummyk, &dummyl, a, 1, b,
                1, r1, r2, u, 1, v, 1, q, 1, w, lwork, idum, &info);
        chkxer("DGGSVD3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 16;
        dggsvd3("U", "N", "N", 2, 2, 2, &dummyk, &dummyl, a, 2, b,
                2, r1, r2, u, 1, v, 1, q, 1, w, lwork, idum, &info);
        chkxer("DGGSVD3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 18;
        dggsvd3("N", "V", "N", 1, 1, 2, &dummyk, &dummyl, a, 1, b,
                2, r1, r2, u, 1, v, 1, q, 1, w, lwork, idum, &info);
        chkxer("DGGSVD3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 20;
        dggsvd3("N", "N", "Q", 1, 2, 1, &dummyk, &dummyl, a, 1, b,
                1, r1, r2, u, 1, v, 1, q, 1, w, lwork, idum, &info);
        chkxer("DGGSVD3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 11;

        /* DGGSVP3 */

        strcpy(xerbla_srnamt, "DGGSVP3");
        xerbla_infot = 1;
        dggsvp3("/", "N", "N", 0, 0, 0, a, 1, b, 1, tola, tolb,
                &dummyk, &dummyl, u, 1, v, 1, q, 1, iw, tau, w,
                lwork, &info);
        chkxer("DGGSVP3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        dggsvp3("N", "/", "N", 0, 0, 0, a, 1, b, 1, tola, tolb,
                &dummyk, &dummyl, u, 1, v, 1, q, 1, iw, tau, w,
                lwork, &info);
        chkxer("DGGSVP3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        dggsvp3("N", "N", "/", 0, 0, 0, a, 1, b, 1, tola, tolb,
                &dummyk, &dummyl, u, 1, v, 1, q, 1, iw, tau, w,
                lwork, &info);
        chkxer("DGGSVP3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        dggsvp3("N", "N", "N", -1, 0, 0, a, 1, b, 1, tola, tolb,
                &dummyk, &dummyl, u, 1, v, 1, q, 1, iw, tau, w,
                lwork, &info);
        chkxer("DGGSVP3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        dggsvp3("N", "N", "N", 0, -1, 0, a, 1, b, 1, tola, tolb,
                &dummyk, &dummyl, u, 1, v, 1, q, 1, iw, tau, w,
                lwork, &info);
        chkxer("DGGSVP3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        dggsvp3("N", "N", "N", 0, 0, -1, a, 1, b, 1, tola, tolb,
                &dummyk, &dummyl, u, 1, v, 1, q, 1, iw, tau, w,
                lwork, &info);
        chkxer("DGGSVP3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        dggsvp3("N", "N", "N", 2, 1, 1, a, 1, b, 1, tola, tolb,
                &dummyk, &dummyl, u, 1, v, 1, q, 1, iw, tau, w,
                lwork, &info);
        chkxer("DGGSVP3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        dggsvp3("N", "N", "N", 1, 2, 1, a, 1, b, 1, tola, tolb,
                &dummyk, &dummyl, u, 1, v, 1, q, 1, iw, tau, w,
                lwork, &info);
        chkxer("DGGSVP3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 16;
        dggsvp3("U", "N", "N", 2, 2, 2, a, 2, b, 2, tola, tolb,
                &dummyk, &dummyl, u, 1, v, 1, q, 1, iw, tau, w,
                lwork, &info);
        chkxer("DGGSVP3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 18;
        dggsvp3("N", "V", "N", 1, 2, 1, a, 1, b, 2, tola, tolb,
                &dummyk, &dummyl, u, 1, v, 1, q, 1, iw, tau, w,
                lwork, &info);
        chkxer("DGGSVP3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 20;
        dggsvp3("N", "N", "Q", 1, 1, 2, a, 1, b, 1, tola, tolb,
                &dummyk, &dummyl, u, 1, v, 1, q, 1, iw, tau, w,
                lwork, &info);
        chkxer("DGGSVP3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 11;

        /* DTGSJA */

        strcpy(xerbla_srnamt, "DTGSJA");
        xerbla_infot = 1;
        dtgsja("/", "N", "N", 0, 0, 0, dummyk, dummyl, a, 1, b,
               1, tola, tolb, r1, r2, u, 1, v, 1, q, 1, w,
               &ncycle, &info);
        chkxer("DTGSJA", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        dtgsja("N", "/", "N", 0, 0, 0, dummyk, dummyl, a, 1, b,
               1, tola, tolb, r1, r2, u, 1, v, 1, q, 1, w,
               &ncycle, &info);
        chkxer("DTGSJA", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        dtgsja("N", "N", "/", 0, 0, 0, dummyk, dummyl, a, 1, b,
               1, tola, tolb, r1, r2, u, 1, v, 1, q, 1, w,
               &ncycle, &info);
        chkxer("DTGSJA", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        dtgsja("N", "N", "N", -1, 0, 0, dummyk, dummyl, a, 1, b,
               1, tola, tolb, r1, r2, u, 1, v, 1, q, 1, w,
               &ncycle, &info);
        chkxer("DTGSJA", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        dtgsja("N", "N", "N", 0, -1, 0, dummyk, dummyl, a, 1, b,
               1, tola, tolb, r1, r2, u, 1, v, 1, q, 1, w,
               &ncycle, &info);
        chkxer("DTGSJA", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        dtgsja("N", "N", "N", 0, 0, -1, dummyk, dummyl, a, 1, b,
               1, tola, tolb, r1, r2, u, 1, v, 1, q, 1, w,
               &ncycle, &info);
        chkxer("DTGSJA", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        dtgsja("N", "N", "N", 0, 0, 0, dummyk, dummyl, a, 0, b,
               1, tola, tolb, r1, r2, u, 1, v, 1, q, 1, w,
               &ncycle, &info);
        chkxer("DTGSJA", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 12;
        dtgsja("N", "N", "N", 0, 0, 0, dummyk, dummyl, a, 1, b,
               0, tola, tolb, r1, r2, u, 1, v, 1, q, 1, w,
               &ncycle, &info);
        chkxer("DTGSJA", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 18;
        dtgsja("U", "N", "N", 0, 0, 0, dummyk, dummyl, a, 1, b,
               1, tola, tolb, r1, r2, u, 0, v, 1, q, 1, w,
               &ncycle, &info);
        chkxer("DTGSJA", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 20;
        dtgsja("N", "V", "N", 0, 0, 0, dummyk, dummyl, a, 1, b,
               1, tola, tolb, r1, r2, u, 1, v, 0, q, 1, w,
               &ncycle, &info);
        chkxer("DTGSJA", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 22;
        dtgsja("N", "N", "Q", 0, 0, 0, dummyk, dummyl, a, 1, b,
               1, tola, tolb, r1, r2, u, 1, v, 1, q, 0, w,
               &ncycle, &info);
        chkxer("DTGSJA", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 11;

    /* Test error exits for the GLM path. */
    } else if ((path[0] == 'G' || path[0] == 'g') &&
               (path[1] == 'L' || path[1] == 'l') &&
               (path[2] == 'M' || path[2] == 'm')) {

        /* DGGGLM */

        strcpy(xerbla_srnamt, "DGGGLM");
        xerbla_infot = 1;
        dggglm(-1, 0, 0, a, 1, b, 1, r1, r2, r3, w, LW, &info);
        chkxer("DGGGLM", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        dggglm(0, -1, 0, a, 1, b, 1, r1, r2, r3, w, LW, &info);
        chkxer("DGGGLM", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        dggglm(0, 1, 0, a, 1, b, 1, r1, r2, r3, w, LW, &info);
        chkxer("DGGGLM", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        dggglm(0, 0, -1, a, 1, b, 1, r1, r2, r3, w, LW, &info);
        chkxer("DGGGLM", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        dggglm(1, 0, 0, a, 1, b, 1, r1, r2, r3, w, LW, &info);
        chkxer("DGGGLM", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        dggglm(0, 0, 0, a, 0, b, 1, r1, r2, r3, w, LW, &info);
        chkxer("DGGGLM", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        dggglm(0, 0, 0, a, 1, b, 0, r1, r2, r3, w, LW, &info);
        chkxer("DGGGLM", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 12;
        dggglm(1, 1, 1, a, 1, b, 1, r1, r2, r3, w, 1, &info);
        chkxer("DGGGLM", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 8;

    /* Test error exits for the LSE path. */
    } else if ((path[0] == 'L' || path[0] == 'l') &&
               (path[1] == 'S' || path[1] == 's') &&
               (path[2] == 'E' || path[2] == 'e')) {

        /* DGGLSE */

        strcpy(xerbla_srnamt, "DGGLSE");
        xerbla_infot = 1;
        dgglse(-1, 0, 0, a, 1, b, 1, r1, r2, r3, w, LW, &info);
        chkxer("DGGLSE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        dgglse(0, -1, 0, a, 1, b, 1, r1, r2, r3, w, LW, &info);
        chkxer("DGGLSE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        dgglse(0, 0, -1, a, 1, b, 1, r1, r2, r3, w, LW, &info);
        chkxer("DGGLSE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        dgglse(0, 0, 1, a, 1, b, 1, r1, r2, r3, w, LW, &info);
        chkxer("DGGLSE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        dgglse(0, 1, 0, a, 1, b, 1, r1, r2, r3, w, LW, &info);
        chkxer("DGGLSE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        dgglse(0, 0, 0, a, 0, b, 1, r1, r2, r3, w, LW, &info);
        chkxer("DGGLSE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        dgglse(0, 0, 0, a, 1, b, 0, r1, r2, r3, w, LW, &info);
        chkxer("DGGLSE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 12;
        dgglse(1, 1, 1, a, 1, b, 1, r1, r2, r3, w, 1, &info);
        chkxer("DGGLSE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 8;

    /* Test error exits for the CSD path. */
    } else if ((path[0] == 'C' || path[0] == 'c') &&
               (path[1] == 'S' || path[1] == 's') &&
               (path[2] == 'D' || path[2] == 'd')) {

        /* DORCSD */

        strcpy(xerbla_srnamt, "DORCSD");
        xerbla_infot = 7;
        dorcsd("Y", "Y", "Y", "Y", "N", "N",
               -1, 0, 0, a, 1, b,
               1, q, 1, z, 1, r1,
               u, 1, v, 1, ls, 1, rs,
               1, w, LW, iw, &info);
        chkxer("DORCSD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        dorcsd("Y", "Y", "Y", "Y", "N", "N",
               1, -1, 0, a, 1, b,
               1, q, 1, z, 1, r1,
               u, 1, v, 1, ls, 1, rs,
               1, w, LW, iw, &info);
        chkxer("DORCSD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        dorcsd("Y", "Y", "Y", "Y", "N", "N",
               1, 1, -1, a, 1, b,
               1, q, 1, z, 1, r1,
               u, 1, v, 1, ls, 1, rs,
               1, w, LW, iw, &info);
        chkxer("DORCSD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 11;
        dorcsd("Y", "Y", "Y", "Y", "N", "N",
               1, 1, 1, a, -1, b,
               1, q, 1, z, 1, r1,
               u, 1, v, 1, ls, 1, rs,
               1, w, LW, iw, &info);
        chkxer("DORCSD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 20;
        dorcsd("Y", "Y", "Y", "Y", "N", "N",
               1, 1, 1, a, 1, b,
               1, q, 1, z, 1, r1,
               u, -1, v, 1, ls, 1, rs,
               1, w, LW, iw, &info);
        chkxer("DORCSD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 22;
        dorcsd("Y", "Y", "Y", "Y", "N", "N",
               1, 1, 1, a, 1, b,
               1, q, 1, z, 1, r1,
               u, 1, v, -1, ls, 1, rs,
               1, w, LW, iw, &info);
        chkxer("DORCSD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 24;
        dorcsd("Y", "Y", "Y", "Y", "N", "N",
               1, 1, 1, a, 1, b,
               1, q, 1, z, 1, r1,
               u, 1, v, 1, ls, -1, rs,
               1, w, LW, iw, &info);
        chkxer("DORCSD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 26;
        dorcsd("Y", "Y", "Y", "Y", "N", "N",
               1, 1, 1, a, 1, b,
               1, q, 1, z, 1, r1,
               u, 1, v, 1, ls, 1, rs,
               -1, w, LW, iw, &info);
        chkxer("DORCSD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 8;

        /* DORBDB */

        strcpy(xerbla_srnamt, "DORBDB");
        xerbla_infot = 3;
        dorbdb("N", "N", -1, 0, 0, a, 1, b, 1, q, 1, z, 1,
               r1, r2, r3, ls, rs, tau, w, LW, &info);
        chkxer("DORBDB", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        dorbdb("N", "N", 1, -1, 0, a, 1, b, 1, q, 1, z, 1,
               r1, r2, r3, ls, rs, tau, w, LW, &info);
        chkxer("DORBDB", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        dorbdb("N", "N", 1, 1, -1, a, 1, b, 1, q, 1, z, 1,
               r1, r2, r3, ls, rs, tau, w, LW, &info);
        chkxer("DORBDB", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 3;

    /* Test error exits for the GQR path. */
    } else if ((path[0] == 'G' || path[0] == 'g') &&
               (path[1] == 'Q' || path[1] == 'q') &&
               (path[2] == 'R' || path[2] == 'r')) {

        /* DGGQRF */

        strcpy(xerbla_srnamt, "DGGQRF");
        xerbla_infot = 1;
        dggqrf(-1, 0, 0, a, 1, r1, b, 1, r2, w, LW, &info);
        chkxer("DGGQRF", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        dggqrf(0, -1, 0, a, 1, r1, b, 1, r2, w, LW, &info);
        chkxer("DGGQRF", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        dggqrf(0, 0, -1, a, 1, r1, b, 1, r2, w, LW, &info);
        chkxer("DGGQRF", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        dggqrf(0, 0, 0, a, 0, r1, b, 1, r2, w, LW, &info);
        chkxer("DGGQRF", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        dggqrf(0, 0, 0, a, 1, r1, b, 0, r2, w, LW, &info);
        chkxer("DGGQRF", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 11;
        dggqrf(1, 1, 2, a, 1, r1, b, 1, r2, w, 1, &info);
        chkxer("DGGQRF", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 6;

        /* DGGRQF */

        strcpy(xerbla_srnamt, "DGGRQF");
        xerbla_infot = 1;
        dggrqf(-1, 0, 0, a, 1, r1, b, 1, r2, w, LW, &info);
        chkxer("DGGRQF", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        dggrqf(0, -1, 0, a, 1, r1, b, 1, r2, w, LW, &info);
        chkxer("DGGRQF", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        dggrqf(0, 0, -1, a, 1, r1, b, 1, r2, w, LW, &info);
        chkxer("DGGRQF", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        dggrqf(0, 0, 0, a, 0, r1, b, 1, r2, w, LW, &info);
        chkxer("DGGRQF", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        dggrqf(0, 0, 0, a, 1, r1, b, 0, r2, w, LW, &info);
        chkxer("DGGRQF", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 11;
        dggrqf(1, 1, 2, a, 1, r1, b, 1, r2, w, 1, &info);
        chkxer("DGGRQF", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 6;

    /* Test error exits for the DGS, DGV, DGX, and DXV paths. */
    } else if (((path[0] == 'D' || path[0] == 'd') &&
                (path[1] == 'G' || path[1] == 'g') &&
                (path[2] == 'S' || path[2] == 's')) ||
               ((path[0] == 'D' || path[0] == 'd') &&
                (path[1] == 'G' || path[1] == 'g') &&
                (path[2] == 'V' || path[2] == 'v')) ||
               ((path[0] == 'D' || path[0] == 'd') &&
                (path[1] == 'G' || path[1] == 'g') &&
                (path[2] == 'X' || path[2] == 'x')) ||
               ((path[0] == 'D' || path[0] == 'd') &&
                (path[1] == 'X' || path[1] == 'x') &&
                (path[2] == 'V' || path[2] == 'v'))) {

        /* DGGES */

        strcpy(xerbla_srnamt, "DGGES ");
        xerbla_infot = 1;
        dgges("/", "N", "S", dlctes, 1, a, 1, b, 1, &sdim, r1, r2,
              r3, q, 1, u, 1, w, 1, bw, &info);
        chkxer("DGGES ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        dgges("N", "/", "S", dlctes, 1, a, 1, b, 1, &sdim, r1, r2,
              r3, q, 1, u, 1, w, 1, bw, &info);
        chkxer("DGGES ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        dgges("N", "V", "/", dlctes, 1, a, 1, b, 1, &sdim, r1, r2,
              r3, q, 1, u, 1, w, 1, bw, &info);
        chkxer("DGGES ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        dgges("N", "V", "S", dlctes, -1, a, 1, b, 1, &sdim, r1,
              r2, r3, q, 1, u, 1, w, 1, bw, &info);
        chkxer("DGGES ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        dgges("N", "V", "S", dlctes, 1, a, 0, b, 1, &sdim, r1, r2,
              r3, q, 1, u, 1, w, 1, bw, &info);
        chkxer("DGGES ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        dgges("N", "V", "S", dlctes, 1, a, 1, b, 0, &sdim, r1, r2,
              r3, q, 1, u, 1, w, 1, bw, &info);
        chkxer("DGGES ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 15;
        dgges("N", "V", "S", dlctes, 1, a, 1, b, 1, &sdim, r1, r2,
              r3, q, 0, u, 1, w, 1, bw, &info);
        chkxer("DGGES ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 15;
        dgges("V", "V", "S", dlctes, 2, a, 2, b, 2, &sdim, r1, r2,
              r3, q, 1, u, 2, w, 1, bw, &info);
        chkxer("DGGES ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 17;
        dgges("N", "V", "S", dlctes, 1, a, 1, b, 1, &sdim, r1, r2,
              r3, q, 1, u, 0, w, 1, bw, &info);
        chkxer("DGGES ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 17;
        dgges("V", "V", "S", dlctes, 2, a, 2, b, 2, &sdim, r1, r2,
              r3, q, 2, u, 1, w, 1, bw, &info);
        chkxer("DGGES ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 19;
        dgges("V", "V", "S", dlctes, 2, a, 2, b, 2, &sdim, r1, r2,
              r3, q, 2, u, 2, w, 1, bw, &info);
        chkxer("DGGES ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 11;

        /* DGGES3 */

        strcpy(xerbla_srnamt, "DGGES3 ");
        xerbla_infot = 1;
        dgges3("/", "N", "S", dlctes, 1, a, 1, b, 1, &sdim, r1,
               r2, r3, q, 1, u, 1, w, 1, bw, &info);
        chkxer("DGGES3 ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        dgges3("N", "/", "S", dlctes, 1, a, 1, b, 1, &sdim, r1,
               r2, r3, q, 1, u, 1, w, 1, bw, &info);
        chkxer("DGGES3 ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        dgges3("N", "V", "/", dlctes, 1, a, 1, b, 1, &sdim, r1,
               r2, r3, q, 1, u, 1, w, 1, bw, &info);
        chkxer("DGGES3 ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        dgges3("N", "V", "S", dlctes, -1, a, 1, b, 1, &sdim, r1,
               r2, r3, q, 1, u, 1, w, 1, bw, &info);
        chkxer("DGGES3 ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        dgges3("N", "V", "S", dlctes, 1, a, 0, b, 1, &sdim, r1,
               r2, r3, q, 1, u, 1, w, 1, bw, &info);
        chkxer("DGGES3 ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        dgges3("N", "V", "S", dlctes, 1, a, 1, b, 0, &sdim, r1,
               r2, r3, q, 1, u, 1, w, 1, bw, &info);
        chkxer("DGGES3 ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 15;
        dgges3("N", "V", "S", dlctes, 1, a, 1, b, 1, &sdim, r1,
               r2, r3, q, 0, u, 1, w, 1, bw, &info);
        chkxer("DGGES3 ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 15;
        dgges3("V", "V", "S", dlctes, 2, a, 2, b, 2, &sdim, r1,
               r2, r3, q, 1, u, 2, w, 1, bw, &info);
        chkxer("DGGES3 ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 17;
        dgges3("N", "V", "S", dlctes, 1, a, 1, b, 1, &sdim, r1,
               r2, r3, q, 1, u, 0, w, 1, bw, &info);
        chkxer("DGGES3 ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 17;
        dgges3("V", "V", "S", dlctes, 2, a, 2, b, 2, &sdim, r1,
               r2, r3, q, 2, u, 1, w, 1, bw, &info);
        chkxer("DGGES3 ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 19;
        dgges3("V", "V", "S", dlctes, 2, a, 2, b, 2, &sdim, r1,
               r2, r3, q, 2, u, 2, w, 1, bw, &info);
        chkxer("DGGES3 ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 11;

        /* DGGESX */

        strcpy(xerbla_srnamt, "DGGESX");
        xerbla_infot = 1;
        dggesx("/", "N", "S", dlctsx, "N", 1, a, 1, b, 1, &sdim,
               r1, r2, r3, q, 1, u, 1, rce, rcv, w, 1, iw, 1, bw,
               &info);
        chkxer("DGGESX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        dggesx("N", "/", "S", dlctsx, "N", 1, a, 1, b, 1, &sdim,
               r1, r2, r3, q, 1, u, 1, rce, rcv, w, 1, iw, 1, bw,
               &info);
        chkxer("DGGESX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        dggesx("V", "V", "/", dlctsx, "N", 1, a, 1, b, 1, &sdim,
               r1, r2, r3, q, 1, u, 1, rce, rcv, w, 1, iw, 1, bw,
               &info);
        chkxer("DGGESX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        dggesx("V", "V", "S", dlctsx, "/", 1, a, 1, b, 1, &sdim,
               r1, r2, r3, q, 1, u, 1, rce, rcv, w, 1, iw, 1, bw,
               &info);
        chkxer("DGGESX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        dggesx("V", "V", "S", dlctsx, "B", -1, a, 1, b, 1, &sdim,
               r1, r2, r3, q, 1, u, 1, rce, rcv, w, 1, iw, 1, bw,
               &info);
        chkxer("DGGESX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        dggesx("V", "V", "S", dlctsx, "B", 1, a, 0, b, 1, &sdim,
               r1, r2, r3, q, 1, u, 1, rce, rcv, w, 1, iw, 1, bw,
               &info);
        chkxer("DGGESX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        dggesx("V", "V", "S", dlctsx, "B", 1, a, 1, b, 0, &sdim,
               r1, r2, r3, q, 1, u, 1, rce, rcv, w, 1, iw, 1, bw,
               &info);
        chkxer("DGGESX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 16;
        dggesx("V", "V", "S", dlctsx, "B", 1, a, 1, b, 1, &sdim,
               r1, r2, r3, q, 0, u, 1, rce, rcv, w, 1, iw, 1, bw,
               &info);
        chkxer("DGGESX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 16;
        dggesx("V", "V", "S", dlctsx, "B", 2, a, 2, b, 2, &sdim,
               r1, r2, r3, q, 1, u, 1, rce, rcv, w, 1, iw, 1, bw,
               &info);
        chkxer("DGGESX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 18;
        dggesx("V", "V", "S", dlctsx, "B", 1, a, 1, b, 1, &sdim,
               r1, r2, r3, q, 1, u, 0, rce, rcv, w, 1, iw, 1, bw,
               &info);
        chkxer("DGGESX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 18;
        dggesx("V", "V", "S", dlctsx, "B", 2, a, 2, b, 2, &sdim,
               r1, r2, r3, q, 2, u, 1, rce, rcv, w, 1, iw, 1, bw,
               &info);
        chkxer("DGGESX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 22;
        dggesx("V", "V", "S", dlctsx, "B", 2, a, 2, b, 2, &sdim,
               r1, r2, r3, q, 2, u, 2, rce, rcv, w, 1, iw, 1, bw,
               &info);
        chkxer("DGGESX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 24;
        dggesx("V", "V", "S", dlctsx, "V", 1, a, 1, b, 1, &sdim,
               r1, r2, r3, q, 1, u, 1, rce, rcv, w, 32, iw, 0,
               bw, &info);
        chkxer("DGGESX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 13;

        /* DGGEV */

        strcpy(xerbla_srnamt, "DGGEV ");
        xerbla_infot = 1;
        dggev("/", "N", 1, a, 1, b, 1, r1, r2, r3, q, 1, u, 1, w,
              1, &info);
        chkxer("DGGEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        dggev("N", "/", 1, a, 1, b, 1, r1, r2, r3, q, 1, u, 1, w,
              1, &info);
        chkxer("DGGEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        dggev("V", "V", -1, a, 1, b, 1, r1, r2, r3, q, 1, u, 1,
              w, 1, &info);
        chkxer("DGGEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        dggev("V", "V", 1, a, 0, b, 1, r1, r2, r3, q, 1, u, 1, w,
              1, &info);
        chkxer("DGGEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        dggev("V", "V", 1, a, 1, b, 0, r1, r2, r3, q, 1, u, 1, w,
              1, &info);
        chkxer("DGGEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 12;
        dggev("N", "V", 1, a, 1, b, 1, r1, r2, r3, q, 0, u, 1, w,
              1, &info);
        chkxer("DGGEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 12;
        dggev("V", "V", 2, a, 2, b, 2, r1, r2, r3, q, 1, u, 2, w,
              1, &info);
        chkxer("DGGEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 14;
        dggev("V", "N", 2, a, 2, b, 2, r1, r2, r3, q, 2, u, 0, w,
              1, &info);
        chkxer("DGGEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 14;
        dggev("V", "V", 2, a, 2, b, 2, r1, r2, r3, q, 2, u, 1, w,
              1, &info);
        chkxer("DGGEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 16;
        dggev("V", "V", 1, a, 1, b, 1, r1, r2, r3, q, 1, u, 1, w,
              1, &info);
        chkxer("DGGEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 10;

        /* DGGEV3 */

        xlaenv(12, 20);
        xlaenv(13, 4);
        xlaenv(14, 13);
        xlaenv(15, 2);
        xlaenv(17, 10);
        strcpy(xerbla_srnamt, "DGGEV3 ");
        xerbla_infot = 1;
        dggev3("/", "N", 1, a, 1, b, 1, r1, r2, r3, q, 1, u, 1,
               w, 1, &info);
        chkxer("DGGEV3 ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        dggev3("N", "/", 1, a, 1, b, 1, r1, r2, r3, q, 1, u, 1,
               w, 1, &info);
        chkxer("DGGEV3 ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        dggev3("V", "V", -1, a, 1, b, 1, r1, r2, r3, q, 1, u, 1,
               w, 1, &info);
        chkxer("DGGEV3 ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        dggev3("V", "V", 1, a, 0, b, 1, r1, r2, r3, q, 1, u, 1,
               w, 1, &info);
        chkxer("DGGEV3 ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        dggev3("V", "V", 1, a, 1, b, 0, r1, r2, r3, q, 1, u, 1,
               w, 1, &info);
        chkxer("DGGEV3 ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 12;
        dggev3("N", "V", 1, a, 1, b, 1, r1, r2, r3, q, 0, u, 1,
               w, 1, &info);
        chkxer("DGGEV3 ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 12;
        dggev3("V", "V", 2, a, 2, b, 2, r1, r2, r3, q, 1, u, 2,
               w, 1, &info);
        chkxer("DGGEV3 ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 14;
        dggev3("V", "N", 2, a, 2, b, 2, r1, r2, r3, q, 2, u, 0,
               w, 1, &info);
        chkxer("DGGEV3 ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 14;
        dggev3("V", "V", 2, a, 2, b, 2, r1, r2, r3, q, 2, u, 1,
               w, 1, &info);
        chkxer("DGGEV3 ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 16;
        dggev3("V", "V", 1, a, 1, b, 1, r1, r2, r3, q, 1, u, 1,
               w, 1, &info);
        chkxer("DGGEV3 ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 10;

        /* DGGEVX */

        strcpy(xerbla_srnamt, "DGGEVX");
        xerbla_infot = 1;
        dggevx("/", "N", "N", "N", 1, a, 1, b, 1, r1, r2, r3, q,
               1, u, 1, &ilo, &ihi, ls, rs, &anrm, &bnrm, rce, rcv,
               w, 1, iw, bw, &info);
        chkxer("DGGEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        dggevx("N", "/", "N", "N", 1, a, 1, b, 1, r1, r2, r3, q,
               1, u, 1, &ilo, &ihi, ls, rs, &anrm, &bnrm, rce, rcv,
               w, 1, iw, bw, &info);
        chkxer("DGGEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        dggevx("N", "N", "/", "N", 1, a, 1, b, 1, r1, r2, r3, q,
               1, u, 1, &ilo, &ihi, ls, rs, &anrm, &bnrm, rce, rcv,
               w, 1, iw, bw, &info);
        chkxer("DGGEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        dggevx("N", "N", "N", "/", 1, a, 1, b, 1, r1, r2, r3, q,
               1, u, 1, &ilo, &ihi, ls, rs, &anrm, &bnrm, rce, rcv,
               w, 1, iw, bw, &info);
        chkxer("DGGEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        dggevx("N", "N", "N", "N", -1, a, 1, b, 1, r1, r2, r3, q,
               1, u, 1, &ilo, &ihi, ls, rs, &anrm, &bnrm, rce, rcv,
               w, 1, iw, bw, &info);
        chkxer("DGGEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        dggevx("N", "N", "N", "N", 1, a, 0, b, 1, r1, r2, r3, q,
               1, u, 1, &ilo, &ihi, ls, rs, &anrm, &bnrm, rce, rcv,
               w, 1, iw, bw, &info);
        chkxer("DGGEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        dggevx("N", "N", "N", "N", 1, a, 1, b, 0, r1, r2, r3, q,
               1, u, 1, &ilo, &ihi, ls, rs, &anrm, &bnrm, rce, rcv,
               w, 1, iw, bw, &info);
        chkxer("DGGEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 14;
        dggevx("N", "N", "N", "N", 1, a, 1, b, 1, r1, r2, r3, q,
               0, u, 1, &ilo, &ihi, ls, rs, &anrm, &bnrm, rce, rcv,
               w, 1, iw, bw, &info);
        chkxer("DGGEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 14;
        dggevx("N", "V", "N", "N", 2, a, 2, b, 2, r1, r2, r3, q,
               1, u, 2, &ilo, &ihi, ls, rs, &anrm, &bnrm, rce, rcv,
               w, 1, iw, bw, &info);
        chkxer("DGGEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 16;
        dggevx("N", "N", "N", "N", 1, a, 1, b, 1, r1, r2, r3, q,
               1, u, 0, &ilo, &ihi, ls, rs, &anrm, &bnrm, rce, rcv,
               w, 1, iw, bw, &info);
        chkxer("DGGEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 16;
        dggevx("N", "N", "V", "N", 2, a, 2, b, 2, r1, r2, r3, q,
               2, u, 1, &ilo, &ihi, ls, rs, &anrm, &bnrm, rce, rcv,
               w, 1, iw, bw, &info);
        chkxer("DGGEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 26;
        dggevx("N", "N", "V", "N", 2, a, 2, b, 2, r1, r2, r3, q,
               2, u, 2, &ilo, &ihi, ls, rs, &anrm, &bnrm, rce, rcv,
               w, 1, iw, bw, &info);
        chkxer("DGGEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 12;

        /* DTGEXC */

        strcpy(xerbla_srnamt, "DTGEXC");
        xerbla_infot = 3;
        dtgexc(1, 1, -1, a, 1, b, 1, q, 1, z, 1, &ifst,
               &ilst, w, 1, &info);
        chkxer("DTGEXC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        dtgexc(1, 1, 1, a, 0, b, 1, q, 1, z, 1, &ifst,
               &ilst, w, 1, &info);
        chkxer("DTGEXC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        dtgexc(1, 1, 1, a, 1, b, 0, q, 1, z, 1, &ifst,
               &ilst, w, 1, &info);
        chkxer("DTGEXC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        dtgexc(0, 1, 1, a, 1, b, 1, q, 0, z, 1, &ifst,
               &ilst, w, 1, &info);
        chkxer("DTGEXC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        dtgexc(1, 1, 1, a, 1, b, 1, q, 0, z, 1, &ifst,
               &ilst, w, 1, &info);
        chkxer("DTGEXC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 11;
        dtgexc(1, 0, 1, a, 1, b, 1, q, 1, z, 0, &ifst,
               &ilst, w, 1, &info);
        chkxer("DTGEXC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 11;
        dtgexc(1, 1, 1, a, 1, b, 1, q, 1, z, 0, &ifst,
               &ilst, w, 1, &info);
        chkxer("DTGEXC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 15;
        dtgexc(1, 1, 1, a, 1, b, 1, q, 1, z, 1, &ifst,
               &ilst, w, 0, &info);
        chkxer("DTGEXC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 8;

        /* DTGSEN */

        strcpy(xerbla_srnamt, "DTGSEN");
        xerbla_infot = 1;
        dtgsen(-1, 1, 1, sel, 1, a, 1, b, 1, r1, r2,
               r3, q, 1, z, 1, &m, &tola, &tolb, rcv, w, 1, iw, 1,
               &info);
        chkxer("DTGSEN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        dtgsen(1, 1, 1, sel, -1, a, 1, b, 1, r1, r2,
               r3, q, 1, z, 1, &m, &tola, &tolb, rcv, w, 1, iw, 1,
               &info);
        chkxer("DTGSEN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        dtgsen(1, 1, 1, sel, 1, a, 0, b, 1, r1, r2, r3,
               q, 1, z, 1, &m, &tola, &tolb, rcv, w, 1, iw, 1,
               &info);
        chkxer("DTGSEN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        dtgsen(1, 1, 1, sel, 1, a, 1, b, 0, r1, r2, r3,
               q, 1, z, 1, &m, &tola, &tolb, rcv, w, 1, iw, 1,
               &info);
        chkxer("DTGSEN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 14;
        dtgsen(1, 1, 1, sel, 1, a, 1, b, 1, r1, r2, r3,
               q, 0, z, 1, &m, &tola, &tolb, rcv, w, 1, iw, 1,
               &info);
        chkxer("DTGSEN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 16;
        dtgsen(1, 1, 1, sel, 1, a, 1, b, 1, r1, r2, r3,
               q, 1, z, 0, &m, &tola, &tolb, rcv, w, 1, iw, 1,
               &info);
        chkxer("DTGSEN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 22;
        dtgsen(0, 1, 1, sel, 1, a, 1, b, 1, r1, r2, r3,
               q, 1, z, 1, &m, &tola, &tolb, rcv, w, 1, iw, 1,
               &info);
        chkxer("DTGSEN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 22;
        dtgsen(1, 1, 1, sel, 1, a, 1, b, 1, r1, r2, r3,
               q, 1, z, 1, &m, &tola, &tolb, rcv, w, 1, iw, 1,
               &info);
        chkxer("DTGSEN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 22;
        dtgsen(2, 1, 1, sel, 1, a, 1, b, 1, r1, r2, r3,
               q, 1, z, 1, &m, &tola, &tolb, rcv, w, 1, iw, 1,
               &info);
        chkxer("DTGSEN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 24;
        dtgsen(0, 1, 1, sel, 1, a, 1, b, 1, r1, r2, r3,
               q, 1, z, 1, &m, &tola, &tolb, rcv, w, 20, iw, 0,
               &info);
        chkxer("DTGSEN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 24;
        dtgsen(1, 1, 1, sel, 1, a, 1, b, 1, r1, r2, r3,
               q, 1, z, 1, &m, &tola, &tolb, rcv, w, 20, iw, 0,
               &info);
        chkxer("DTGSEN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 24;
        dtgsen(2, 1, 1, sel, 1, a, 1, b, 1, r1, r2, r3,
               q, 1, z, 1, &m, &tola, &tolb, rcv, w, 20, iw, 1,
               &info);
        chkxer("DTGSEN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 12;

        /* DTGSNA */

        strcpy(xerbla_srnamt, "DTGSNA");
        xerbla_infot = 1;
        dtgsna("/", "A", sel, 1, a, 1, b, 1, q, 1, u, 1, r1, r2,
               1, &m, w, 1, iw, &info);
        chkxer("DTGSNA", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        dtgsna("B", "/", sel, 1, a, 1, b, 1, q, 1, u, 1, r1, r2,
               1, &m, w, 1, iw, &info);
        chkxer("DTGSNA", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        dtgsna("B", "A", sel, -1, a, 1, b, 1, q, 1, u, 1, r1, r2,
               1, &m, w, 1, iw, &info);
        chkxer("DTGSNA", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        dtgsna("B", "A", sel, 1, a, 0, b, 1, q, 1, u, 1, r1, r2,
               1, &m, w, 1, iw, &info);
        chkxer("DTGSNA", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        dtgsna("B", "A", sel, 1, a, 1, b, 0, q, 1, u, 1, r1, r2,
               1, &m, w, 1, iw, &info);
        chkxer("DTGSNA", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        dtgsna("E", "A", sel, 1, a, 1, b, 1, q, 0, u, 1, r1, r2,
               1, &m, w, 1, iw, &info);
        chkxer("DTGSNA", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 12;
        dtgsna("E", "A", sel, 1, a, 1, b, 1, q, 1, u, 0, r1, r2,
               1, &m, w, 1, iw, &info);
        chkxer("DTGSNA", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 15;
        dtgsna("E", "A", sel, 1, a, 1, b, 1, q, 1, u, 1, r1, r2,
               0, &m, w, 1, iw, &info);
        chkxer("DTGSNA", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 18;
        dtgsna("E", "A", sel, 1, a, 1, b, 1, q, 1, u, 1, r1, r2,
               1, &m, w, 0, iw, &info);
        chkxer("DTGSNA", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 9;

        /* DTGSYL */

        strcpy(xerbla_srnamt, "DTGSYL");
        xerbla_infot = 1;
        dtgsyl("/", 0, 1, 1, a, 1, b, 1, q, 1, u, 1, v, 1, z, 1,
               &scale, &dif, w, 1, iw, &info);
        chkxer("DTGSYL", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        dtgsyl("N", -1, 1, 1, a, 1, b, 1, q, 1, u, 1, v, 1, z, 1,
               &scale, &dif, w, 1, iw, &info);
        chkxer("DTGSYL", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        dtgsyl("N", 0, 0, 1, a, 1, b, 1, q, 1, u, 1, v, 1, z, 1,
               &scale, &dif, w, 1, iw, &info);
        chkxer("DTGSYL", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        dtgsyl("N", 0, 1, 0, a, 1, b, 1, q, 1, u, 1, v, 1, z, 1,
               &scale, &dif, w, 1, iw, &info);
        chkxer("DTGSYL", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        dtgsyl("N", 0, 1, 1, a, 0, b, 1, q, 1, u, 1, v, 1, z, 1,
               &scale, &dif, w, 1, iw, &info);
        chkxer("DTGSYL", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        dtgsyl("N", 0, 1, 1, a, 1, b, 0, q, 1, u, 1, v, 1, z, 1,
               &scale, &dif, w, 1, iw, &info);
        chkxer("DTGSYL", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        dtgsyl("N", 0, 1, 1, a, 1, b, 1, q, 0, u, 1, v, 1, z, 1,
               &scale, &dif, w, 1, iw, &info);
        chkxer("DTGSYL", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 12;
        dtgsyl("N", 0, 1, 1, a, 1, b, 1, q, 1, u, 0, v, 1, z, 1,
               &scale, &dif, w, 1, iw, &info);
        chkxer("DTGSYL", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 14;
        dtgsyl("N", 0, 1, 1, a, 1, b, 1, q, 1, u, 1, v, 0, z, 1,
               &scale, &dif, w, 1, iw, &info);
        chkxer("DTGSYL", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 16;
        dtgsyl("N", 0, 1, 1, a, 1, b, 1, q, 1, u, 1, v, 1, z, 0,
               &scale, &dif, w, 1, iw, &info);
        chkxer("DTGSYL", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 20;
        dtgsyl("N", 1, 1, 1, a, 1, b, 1, q, 1, u, 1, v, 1, z, 1,
               &scale, &dif, w, 1, iw, &info);
        chkxer("DTGSYL", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 20;
        dtgsyl("N", 2, 1, 1, a, 1, b, 1, q, 1, u, 1, v, 1, z, 1,
               &scale, &dif, w, 1, iw, &info);
        chkxer("DTGSYL", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 12;
    }

    *ok = xerbla_ok;

    xerbla_srnamt[0] = '\0';
    xerbla_infot = 0;
}
