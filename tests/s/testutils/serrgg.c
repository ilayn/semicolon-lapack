/**
 * @file serrgg.c
 * @brief SERRGG tests the error exits for the generalized nonsymmetric
 *        eigenvalue, GSVD, CSD, GLM, LSE and GQR routines.
 *
 * Port of LAPACK TESTING/EIG/serrgg.f
 * Tests error exits for: SGGES, SGGESX, SGGEV, SGGEVX, SGGES3, SGGEV3,
 * SGGGLM, SGGHRD, SGGLSE, SGGQRF, SGGRQF, SGGSVD3, SGGSVP3, SHGEQZ,
 * SORBDB, SORCSD, STGEVC, STGEXC, STGSEN, STGSJA, STGSNA, and STGSYL.
 */

#define LAPACK_TUNING_TEST
#include <math.h>
#include <string.h>
#include "semicolon_lapack_single.h"
#include "verify.h"

/*
 * Port of LAPACK TESTING/EIG/slctes.f
 */
static INT slctes(const f32* zr, const f32* zi, const f32* d)
{
    (void)zi;
    if (*d == 0.0f) {
        return (*zr < 0.0f);
    } else {
        return (copysignf(1.0f, *zr) != copysignf(1.0f, *d));
    }
}

/*
 * Port of LAPACK TESTING/EIG/slctsx.f
 */
static INT slctsx_m, slctsx_n, slctsx_mplusn, slctsx_i, slctsx_fs;

static INT slctsx(const f32* ar, const f32* ai, const f32* beta)
{
    (void)ar;
    (void)ai;
    (void)beta;
    INT res;
    if (slctsx_fs) {
        slctsx_i = slctsx_i + 1;
        if (slctsx_i <= slctsx_m) {
            res = 0;
        } else {
            res = 1;
        }
        if (slctsx_i == slctsx_mplusn) {
            slctsx_fs = 0;
            slctsx_i = 0;
        }
    } else {
        slctsx_i = slctsx_i + 1;
        if (slctsx_i <= slctsx_n) {
            res = 1;
        } else {
            res = 0;
        }
        if (slctsx_i == slctsx_mplusn) {
            slctsx_fs = 1;
            slctsx_i = 0;
        }
    }
    return res;
}

void serrgg(const char* path, INT* ok, INT* nt)
{
    enum { NMAX = 3, LW = 6 * NMAX };
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;

    char c2[2];
    INT dummyk = 0, dummyl = 0;
    INT i, ifst, ilo, ihi, ilst, info, j, m, ncycle, sdim, lwork;
    f32 anrm, bnrm, dif, scale, tola, tolb;

    INT bw[NMAX], sel[NMAX];
    INT iw[NMAX], idum[NMAX];
    f32 a[NMAX * NMAX], b[NMAX * NMAX], ls[NMAX];
    f32 q[NMAX * NMAX], r1[NMAX], r2[NMAX];
    f32 r3[NMAX], rce[2], rcv[2], rs[NMAX];
    f32 tau[NMAX], u[NMAX * NMAX], v[NMAX * NMAX];
    f32 w[LW], z[NMAX * NMAX];

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

        /* SGGHRD */

        strcpy(xerbla_srnamt, "SGGHRD");
        xerbla_infot = 1;
        sgghrd("/", "N", 0, 0, -1, a, 1, b, 1, q, 1, z, 1, &info);
        chkxer("SGGHRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        sgghrd("N", "/", 0, 0, -1, a, 1, b, 1, q, 1, z, 1, &info);
        chkxer("SGGHRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        sgghrd("N", "N", -1, -1, -1, a, 1, b, 1, q, 1, z, 1, &info);
        chkxer("SGGHRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        sgghrd("N", "N", 0, -1, -1, a, 1, b, 1, q, 1, z, 1, &info);
        chkxer("SGGHRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        sgghrd("N", "N", 0, 0, 0, a, 1, b, 1, q, 1, z, 1, &info);
        chkxer("SGGHRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        sgghrd("N", "N", 2, 0, 0, a, 1, b, 2, q, 1, z, 1, &info);
        chkxer("SGGHRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        sgghrd("N", "N", 2, 0, 0, a, 2, b, 1, q, 1, z, 1, &info);
        chkxer("SGGHRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 11;
        sgghrd("V", "N", 2, 0, 0, a, 2, b, 2, q, 1, z, 1, &info);
        chkxer("SGGHRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 13;
        sgghrd("N", "V", 2, 0, 0, a, 2, b, 2, q, 1, z, 1, &info);
        chkxer("SGGHRD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 9;

        /* SGGHD3 */

        strcpy(xerbla_srnamt, "SGGHD3");
        xerbla_infot = 1;
        sgghd3("/", "N", 0, 0, -1, a, 1, b, 1, q, 1, z, 1, w, LW,
               &info);
        chkxer("SGGHD3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        sgghd3("N", "/", 0, 0, -1, a, 1, b, 1, q, 1, z, 1, w, LW,
               &info);
        chkxer("SGGHD3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        sgghd3("N", "N", -1, -1, -1, a, 1, b, 1, q, 1, z, 1, w, LW,
               &info);
        chkxer("SGGHD3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        sgghd3("N", "N", 0, -1, -1, a, 1, b, 1, q, 1, z, 1, w, LW,
               &info);
        chkxer("SGGHD3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        sgghd3("N", "N", 0, 0, 0, a, 1, b, 1, q, 1, z, 1, w, LW,
               &info);
        chkxer("SGGHD3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        sgghd3("N", "N", 2, 0, 0, a, 1, b, 2, q, 1, z, 1, w, LW,
               &info);
        chkxer("SGGHD3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        sgghd3("N", "N", 2, 0, 0, a, 2, b, 1, q, 1, z, 1, w, LW,
               &info);
        chkxer("SGGHD3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 11;
        sgghd3("V", "N", 2, 0, 0, a, 2, b, 2, q, 1, z, 1, w, LW,
               &info);
        chkxer("SGGHD3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 13;
        sgghd3("N", "V", 2, 0, 0, a, 2, b, 2, q, 1, z, 1, w, LW,
               &info);
        chkxer("SGGHD3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 9;

        /* SHGEQZ */

        strcpy(xerbla_srnamt, "SHGEQZ");
        xerbla_infot = 1;
        shgeqz("/", "N", "N", 0, 0, -1, a, 1, b, 1, r1, r2, r3, q,
               1, z, 1, w, LW, &info);
        chkxer("SHGEQZ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        shgeqz("E", "/", "N", 0, 0, -1, a, 1, b, 1, r1, r2, r3, q,
               1, z, 1, w, LW, &info);
        chkxer("SHGEQZ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        shgeqz("E", "N", "/", 0, 0, -1, a, 1, b, 1, r1, r2, r3, q,
               1, z, 1, w, LW, &info);
        chkxer("SHGEQZ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        shgeqz("E", "N", "N", -1, -1, -1, a, 1, b, 1, r1, r2, r3,
               q, 1, z, 1, w, LW, &info);
        chkxer("SHGEQZ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        shgeqz("E", "N", "N", 0, -1, -1, a, 1, b, 1, r1, r2, r3, q,
               1, z, 1, w, LW, &info);
        chkxer("SHGEQZ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        shgeqz("E", "N", "N", 0, 0, 0, a, 1, b, 1, r1, r2, r3, q,
               1, z, 1, w, LW, &info);
        chkxer("SHGEQZ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        shgeqz("E", "N", "N", 2, 0, 0, a, 1, b, 2, r1, r2, r3, q,
               1, z, 1, w, LW, &info);
        chkxer("SHGEQZ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        shgeqz("E", "N", "N", 2, 0, 0, a, 2, b, 1, r1, r2, r3, q,
               1, z, 1, w, LW, &info);
        chkxer("SHGEQZ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 15;
        shgeqz("E", "V", "N", 2, 0, 0, a, 2, b, 2, r1, r2, r3, q,
               1, z, 1, w, LW, &info);
        chkxer("SHGEQZ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 17;
        shgeqz("E", "N", "V", 2, 0, 0, a, 2, b, 2, r1, r2, r3, q,
               1, z, 1, w, LW, &info);
        chkxer("SHGEQZ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 10;

        /* STGEVC */

        strcpy(xerbla_srnamt, "STGEVC");
        xerbla_infot = 1;
        stgevc("/", "A", sel, 0, a, 1, b, 1, q, 1, z, 1, 0, &m, w,
               &info);
        chkxer("STGEVC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        stgevc("R", "/", sel, 0, a, 1, b, 1, q, 1, z, 1, 0, &m, w,
               &info);
        chkxer("STGEVC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        stgevc("R", "A", sel, -1, a, 1, b, 1, q, 1, z, 1, 0, &m,
               w, &info);
        chkxer("STGEVC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        stgevc("R", "A", sel, 2, a, 1, b, 2, q, 1, z, 2, 0, &m, w,
               &info);
        chkxer("STGEVC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        stgevc("R", "A", sel, 2, a, 2, b, 1, q, 1, z, 2, 0, &m, w,
               &info);
        chkxer("STGEVC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        stgevc("L", "A", sel, 2, a, 2, b, 2, q, 1, z, 1, 0, &m, w,
               &info);
        chkxer("STGEVC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 12;
        stgevc("R", "A", sel, 2, a, 2, b, 2, q, 1, z, 1, 0, &m, w,
               &info);
        chkxer("STGEVC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 13;
        stgevc("R", "A", sel, 2, a, 2, b, 2, q, 1, z, 2, 1, &m, w,
               &info);
        chkxer("STGEVC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 8;

    /* Test error exits for the GSV path. */
    } else if ((path[0] == 'G' || path[0] == 'g') &&
               (path[1] == 'S' || path[1] == 's') &&
               (path[2] == 'V' || path[2] == 'v')) {

        /* SGGSVD3 */

        strcpy(xerbla_srnamt, "SGGSVD3");
        xerbla_infot = 1;
        sggsvd3("/", "N", "N", 0, 0, 0, &dummyk, &dummyl, a, 1, b,
                1, r1, r2, u, 1, v, 1, q, 1, w, lwork, idum, &info);
        chkxer("SGGSVD3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        sggsvd3("N", "/", "N", 0, 0, 0, &dummyk, &dummyl, a, 1, b,
                1, r1, r2, u, 1, v, 1, q, 1, w, lwork, idum, &info);
        chkxer("SGGSVD3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        sggsvd3("N", "N", "/", 0, 0, 0, &dummyk, &dummyl, a, 1, b,
                1, r1, r2, u, 1, v, 1, q, 1, w, lwork, idum, &info);
        chkxer("SGGSVD3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        sggsvd3("N", "N", "N", -1, 0, 0, &dummyk, &dummyl, a, 1, b,
                1, r1, r2, u, 1, v, 1, q, 1, w, lwork, idum, &info);
        chkxer("SGGSVD3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        sggsvd3("N", "N", "N", 0, -1, 0, &dummyk, &dummyl, a, 1, b,
                1, r1, r2, u, 1, v, 1, q, 1, w, lwork, idum, &info);
        chkxer("SGGSVD3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        sggsvd3("N", "N", "N", 0, 0, -1, &dummyk, &dummyl, a, 1, b,
                1, r1, r2, u, 1, v, 1, q, 1, w, lwork, idum, &info);
        chkxer("SGGSVD3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        sggsvd3("N", "N", "N", 2, 1, 1, &dummyk, &dummyl, a, 1, b,
                1, r1, r2, u, 1, v, 1, q, 1, w, lwork, idum, &info);
        chkxer("SGGSVD3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 12;
        sggsvd3("N", "N", "N", 1, 1, 2, &dummyk, &dummyl, a, 1, b,
                1, r1, r2, u, 1, v, 1, q, 1, w, lwork, idum, &info);
        chkxer("SGGSVD3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 16;
        sggsvd3("U", "N", "N", 2, 2, 2, &dummyk, &dummyl, a, 2, b,
                2, r1, r2, u, 1, v, 1, q, 1, w, lwork, idum, &info);
        chkxer("SGGSVD3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 18;
        sggsvd3("N", "V", "N", 1, 1, 2, &dummyk, &dummyl, a, 1, b,
                2, r1, r2, u, 1, v, 1, q, 1, w, lwork, idum, &info);
        chkxer("SGGSVD3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 20;
        sggsvd3("N", "N", "Q", 1, 2, 1, &dummyk, &dummyl, a, 1, b,
                1, r1, r2, u, 1, v, 1, q, 1, w, lwork, idum, &info);
        chkxer("SGGSVD3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 11;

        /* SGGSVP3 */

        strcpy(xerbla_srnamt, "SGGSVP3");
        xerbla_infot = 1;
        sggsvp3("/", "N", "N", 0, 0, 0, a, 1, b, 1, tola, tolb,
                &dummyk, &dummyl, u, 1, v, 1, q, 1, iw, tau, w,
                lwork, &info);
        chkxer("SGGSVP3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        sggsvp3("N", "/", "N", 0, 0, 0, a, 1, b, 1, tola, tolb,
                &dummyk, &dummyl, u, 1, v, 1, q, 1, iw, tau, w,
                lwork, &info);
        chkxer("SGGSVP3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        sggsvp3("N", "N", "/", 0, 0, 0, a, 1, b, 1, tola, tolb,
                &dummyk, &dummyl, u, 1, v, 1, q, 1, iw, tau, w,
                lwork, &info);
        chkxer("SGGSVP3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        sggsvp3("N", "N", "N", -1, 0, 0, a, 1, b, 1, tola, tolb,
                &dummyk, &dummyl, u, 1, v, 1, q, 1, iw, tau, w,
                lwork, &info);
        chkxer("SGGSVP3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        sggsvp3("N", "N", "N", 0, -1, 0, a, 1, b, 1, tola, tolb,
                &dummyk, &dummyl, u, 1, v, 1, q, 1, iw, tau, w,
                lwork, &info);
        chkxer("SGGSVP3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        sggsvp3("N", "N", "N", 0, 0, -1, a, 1, b, 1, tola, tolb,
                &dummyk, &dummyl, u, 1, v, 1, q, 1, iw, tau, w,
                lwork, &info);
        chkxer("SGGSVP3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        sggsvp3("N", "N", "N", 2, 1, 1, a, 1, b, 1, tola, tolb,
                &dummyk, &dummyl, u, 1, v, 1, q, 1, iw, tau, w,
                lwork, &info);
        chkxer("SGGSVP3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        sggsvp3("N", "N", "N", 1, 2, 1, a, 1, b, 1, tola, tolb,
                &dummyk, &dummyl, u, 1, v, 1, q, 1, iw, tau, w,
                lwork, &info);
        chkxer("SGGSVP3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 16;
        sggsvp3("U", "N", "N", 2, 2, 2, a, 2, b, 2, tola, tolb,
                &dummyk, &dummyl, u, 1, v, 1, q, 1, iw, tau, w,
                lwork, &info);
        chkxer("SGGSVP3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 18;
        sggsvp3("N", "V", "N", 1, 2, 1, a, 1, b, 2, tola, tolb,
                &dummyk, &dummyl, u, 1, v, 1, q, 1, iw, tau, w,
                lwork, &info);
        chkxer("SGGSVP3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 20;
        sggsvp3("N", "N", "Q", 1, 1, 2, a, 1, b, 1, tola, tolb,
                &dummyk, &dummyl, u, 1, v, 1, q, 1, iw, tau, w,
                lwork, &info);
        chkxer("SGGSVP3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 11;

        /* STGSJA */

        strcpy(xerbla_srnamt, "STGSJA");
        xerbla_infot = 1;
        stgsja("/", "N", "N", 0, 0, 0, dummyk, dummyl, a, 1, b,
               1, tola, tolb, r1, r2, u, 1, v, 1, q, 1, w,
               &ncycle, &info);
        chkxer("STGSJA", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        stgsja("N", "/", "N", 0, 0, 0, dummyk, dummyl, a, 1, b,
               1, tola, tolb, r1, r2, u, 1, v, 1, q, 1, w,
               &ncycle, &info);
        chkxer("STGSJA", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        stgsja("N", "N", "/", 0, 0, 0, dummyk, dummyl, a, 1, b,
               1, tola, tolb, r1, r2, u, 1, v, 1, q, 1, w,
               &ncycle, &info);
        chkxer("STGSJA", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        stgsja("N", "N", "N", -1, 0, 0, dummyk, dummyl, a, 1, b,
               1, tola, tolb, r1, r2, u, 1, v, 1, q, 1, w,
               &ncycle, &info);
        chkxer("STGSJA", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        stgsja("N", "N", "N", 0, -1, 0, dummyk, dummyl, a, 1, b,
               1, tola, tolb, r1, r2, u, 1, v, 1, q, 1, w,
               &ncycle, &info);
        chkxer("STGSJA", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        stgsja("N", "N", "N", 0, 0, -1, dummyk, dummyl, a, 1, b,
               1, tola, tolb, r1, r2, u, 1, v, 1, q, 1, w,
               &ncycle, &info);
        chkxer("STGSJA", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        stgsja("N", "N", "N", 0, 0, 0, dummyk, dummyl, a, 0, b,
               1, tola, tolb, r1, r2, u, 1, v, 1, q, 1, w,
               &ncycle, &info);
        chkxer("STGSJA", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 12;
        stgsja("N", "N", "N", 0, 0, 0, dummyk, dummyl, a, 1, b,
               0, tola, tolb, r1, r2, u, 1, v, 1, q, 1, w,
               &ncycle, &info);
        chkxer("STGSJA", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 18;
        stgsja("U", "N", "N", 0, 0, 0, dummyk, dummyl, a, 1, b,
               1, tola, tolb, r1, r2, u, 0, v, 1, q, 1, w,
               &ncycle, &info);
        chkxer("STGSJA", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 20;
        stgsja("N", "V", "N", 0, 0, 0, dummyk, dummyl, a, 1, b,
               1, tola, tolb, r1, r2, u, 1, v, 0, q, 1, w,
               &ncycle, &info);
        chkxer("STGSJA", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 22;
        stgsja("N", "N", "Q", 0, 0, 0, dummyk, dummyl, a, 1, b,
               1, tola, tolb, r1, r2, u, 1, v, 1, q, 0, w,
               &ncycle, &info);
        chkxer("STGSJA", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 11;

    /* Test error exits for the GLM path. */
    } else if ((path[0] == 'G' || path[0] == 'g') &&
               (path[1] == 'L' || path[1] == 'l') &&
               (path[2] == 'M' || path[2] == 'm')) {

        /* SGGGLM */

        strcpy(xerbla_srnamt, "SGGGLM");
        xerbla_infot = 1;
        sggglm(-1, 0, 0, a, 1, b, 1, r1, r2, r3, w, LW, &info);
        chkxer("SGGGLM", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        sggglm(0, -1, 0, a, 1, b, 1, r1, r2, r3, w, LW, &info);
        chkxer("SGGGLM", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        sggglm(0, 1, 0, a, 1, b, 1, r1, r2, r3, w, LW, &info);
        chkxer("SGGGLM", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        sggglm(0, 0, -1, a, 1, b, 1, r1, r2, r3, w, LW, &info);
        chkxer("SGGGLM", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        sggglm(1, 0, 0, a, 1, b, 1, r1, r2, r3, w, LW, &info);
        chkxer("SGGGLM", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        sggglm(0, 0, 0, a, 0, b, 1, r1, r2, r3, w, LW, &info);
        chkxer("SGGGLM", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        sggglm(0, 0, 0, a, 1, b, 0, r1, r2, r3, w, LW, &info);
        chkxer("SGGGLM", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 12;
        sggglm(1, 1, 1, a, 1, b, 1, r1, r2, r3, w, 1, &info);
        chkxer("SGGGLM", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 8;

    /* Test error exits for the LSE path. */
    } else if ((path[0] == 'L' || path[0] == 'l') &&
               (path[1] == 'S' || path[1] == 's') &&
               (path[2] == 'E' || path[2] == 'e')) {

        /* SGGLSE */

        strcpy(xerbla_srnamt, "SGGLSE");
        xerbla_infot = 1;
        sgglse(-1, 0, 0, a, 1, b, 1, r1, r2, r3, w, LW, &info);
        chkxer("SGGLSE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        sgglse(0, -1, 0, a, 1, b, 1, r1, r2, r3, w, LW, &info);
        chkxer("SGGLSE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        sgglse(0, 0, -1, a, 1, b, 1, r1, r2, r3, w, LW, &info);
        chkxer("SGGLSE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        sgglse(0, 0, 1, a, 1, b, 1, r1, r2, r3, w, LW, &info);
        chkxer("SGGLSE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        sgglse(0, 1, 0, a, 1, b, 1, r1, r2, r3, w, LW, &info);
        chkxer("SGGLSE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        sgglse(0, 0, 0, a, 0, b, 1, r1, r2, r3, w, LW, &info);
        chkxer("SGGLSE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        sgglse(0, 0, 0, a, 1, b, 0, r1, r2, r3, w, LW, &info);
        chkxer("SGGLSE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 12;
        sgglse(1, 1, 1, a, 1, b, 1, r1, r2, r3, w, 1, &info);
        chkxer("SGGLSE", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 8;

    /* Test error exits for the CSD path. */
    } else if ((path[0] == 'C' || path[0] == 'c') &&
               (path[1] == 'S' || path[1] == 's') &&
               (path[2] == 'D' || path[2] == 'd')) {

        /* SORCSD */

        strcpy(xerbla_srnamt, "SORCSD");
        xerbla_infot = 7;
        sorcsd("Y", "Y", "Y", "Y", "N", "N",
               -1, 0, 0, a, 1, b,
               1, q, 1, z, 1, r1,
               u, 1, v, 1, ls, 1, rs,
               1, w, LW, iw, &info);
        chkxer("SORCSD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        sorcsd("Y", "Y", "Y", "Y", "N", "N",
               1, -1, 0, a, 1, b,
               1, q, 1, z, 1, r1,
               u, 1, v, 1, ls, 1, rs,
               1, w, LW, iw, &info);
        chkxer("SORCSD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        sorcsd("Y", "Y", "Y", "Y", "N", "N",
               1, 1, -1, a, 1, b,
               1, q, 1, z, 1, r1,
               u, 1, v, 1, ls, 1, rs,
               1, w, LW, iw, &info);
        chkxer("SORCSD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 11;
        sorcsd("Y", "Y", "Y", "Y", "N", "N",
               1, 1, 1, a, -1, b,
               1, q, 1, z, 1, r1,
               u, 1, v, 1, ls, 1, rs,
               1, w, LW, iw, &info);
        chkxer("SORCSD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 20;
        sorcsd("Y", "Y", "Y", "Y", "N", "N",
               1, 1, 1, a, 1, b,
               1, q, 1, z, 1, r1,
               u, -1, v, 1, ls, 1, rs,
               1, w, LW, iw, &info);
        chkxer("SORCSD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 22;
        sorcsd("Y", "Y", "Y", "Y", "N", "N",
               1, 1, 1, a, 1, b,
               1, q, 1, z, 1, r1,
               u, 1, v, -1, ls, 1, rs,
               1, w, LW, iw, &info);
        chkxer("SORCSD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 24;
        sorcsd("Y", "Y", "Y", "Y", "N", "N",
               1, 1, 1, a, 1, b,
               1, q, 1, z, 1, r1,
               u, 1, v, 1, ls, -1, rs,
               1, w, LW, iw, &info);
        chkxer("SORCSD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 26;
        sorcsd("Y", "Y", "Y", "Y", "N", "N",
               1, 1, 1, a, 1, b,
               1, q, 1, z, 1, r1,
               u, 1, v, 1, ls, 1, rs,
               -1, w, LW, iw, &info);
        chkxer("SORCSD", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 8;

        /* SORBDB */

        strcpy(xerbla_srnamt, "SORBDB");
        xerbla_infot = 3;
        sorbdb("N", "N", -1, 0, 0, a, 1, b, 1, q, 1, z, 1,
               r1, r2, r3, ls, rs, tau, w, LW, &info);
        chkxer("SORBDB", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        sorbdb("N", "N", 1, -1, 0, a, 1, b, 1, q, 1, z, 1,
               r1, r2, r3, ls, rs, tau, w, LW, &info);
        chkxer("SORBDB", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        sorbdb("N", "N", 1, 1, -1, a, 1, b, 1, q, 1, z, 1,
               r1, r2, r3, ls, rs, tau, w, LW, &info);
        chkxer("SORBDB", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 3;

    /* Test error exits for the GQR path. */
    } else if ((path[0] == 'G' || path[0] == 'g') &&
               (path[1] == 'Q' || path[1] == 'q') &&
               (path[2] == 'R' || path[2] == 'r')) {

        /* SGGQRF */

        strcpy(xerbla_srnamt, "SGGQRF");
        xerbla_infot = 1;
        sggqrf(-1, 0, 0, a, 1, r1, b, 1, r2, w, LW, &info);
        chkxer("SGGQRF", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        sggqrf(0, -1, 0, a, 1, r1, b, 1, r2, w, LW, &info);
        chkxer("SGGQRF", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        sggqrf(0, 0, -1, a, 1, r1, b, 1, r2, w, LW, &info);
        chkxer("SGGQRF", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        sggqrf(0, 0, 0, a, 0, r1, b, 1, r2, w, LW, &info);
        chkxer("SGGQRF", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        sggqrf(0, 0, 0, a, 1, r1, b, 0, r2, w, LW, &info);
        chkxer("SGGQRF", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 11;
        sggqrf(1, 1, 2, a, 1, r1, b, 1, r2, w, 1, &info);
        chkxer("SGGQRF", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 6;

        /* SGGRQF */

        strcpy(xerbla_srnamt, "SGGRQF");
        xerbla_infot = 1;
        sggrqf(-1, 0, 0, a, 1, r1, b, 1, r2, w, LW, &info);
        chkxer("SGGRQF", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        sggrqf(0, -1, 0, a, 1, r1, b, 1, r2, w, LW, &info);
        chkxer("SGGRQF", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        sggrqf(0, 0, -1, a, 1, r1, b, 1, r2, w, LW, &info);
        chkxer("SGGRQF", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        sggrqf(0, 0, 0, a, 0, r1, b, 1, r2, w, LW, &info);
        chkxer("SGGRQF", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        sggrqf(0, 0, 0, a, 1, r1, b, 0, r2, w, LW, &info);
        chkxer("SGGRQF", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 11;
        sggrqf(1, 1, 2, a, 1, r1, b, 1, r2, w, 1, &info);
        chkxer("SGGRQF", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 6;

    /* Test error exits for the SGS, SGV, SGX, and SXV paths. */
    } else if (((path[0] == 'S' || path[0] == 's') &&
                (path[1] == 'G' || path[1] == 'g') &&
                (path[2] == 'S' || path[2] == 's')) ||
               ((path[0] == 'S' || path[0] == 's') &&
                (path[1] == 'G' || path[1] == 'g') &&
                (path[2] == 'V' || path[2] == 'v')) ||
               ((path[0] == 'S' || path[0] == 's') &&
                (path[1] == 'G' || path[1] == 'g') &&
                (path[2] == 'X' || path[2] == 'x')) ||
               ((path[0] == 'S' || path[0] == 's') &&
                (path[1] == 'X' || path[1] == 'x') &&
                (path[2] == 'V' || path[2] == 'v'))) {

        /* SGGES */

        strcpy(xerbla_srnamt, "SGGES ");
        xerbla_infot = 1;
        sgges("/", "N", "S", slctes, 1, a, 1, b, 1, &sdim, r1, r2,
              r3, q, 1, u, 1, w, 1, bw, &info);
        chkxer("SGGES ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        sgges("N", "/", "S", slctes, 1, a, 1, b, 1, &sdim, r1, r2,
              r3, q, 1, u, 1, w, 1, bw, &info);
        chkxer("SGGES ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        sgges("N", "V", "/", slctes, 1, a, 1, b, 1, &sdim, r1, r2,
              r3, q, 1, u, 1, w, 1, bw, &info);
        chkxer("SGGES ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        sgges("N", "V", "S", slctes, -1, a, 1, b, 1, &sdim, r1,
              r2, r3, q, 1, u, 1, w, 1, bw, &info);
        chkxer("SGGES ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        sgges("N", "V", "S", slctes, 1, a, 0, b, 1, &sdim, r1, r2,
              r3, q, 1, u, 1, w, 1, bw, &info);
        chkxer("SGGES ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        sgges("N", "V", "S", slctes, 1, a, 1, b, 0, &sdim, r1, r2,
              r3, q, 1, u, 1, w, 1, bw, &info);
        chkxer("SGGES ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 15;
        sgges("N", "V", "S", slctes, 1, a, 1, b, 1, &sdim, r1, r2,
              r3, q, 0, u, 1, w, 1, bw, &info);
        chkxer("SGGES ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 15;
        sgges("V", "V", "S", slctes, 2, a, 2, b, 2, &sdim, r1, r2,
              r3, q, 1, u, 2, w, 1, bw, &info);
        chkxer("SGGES ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 17;
        sgges("N", "V", "S", slctes, 1, a, 1, b, 1, &sdim, r1, r2,
              r3, q, 1, u, 0, w, 1, bw, &info);
        chkxer("SGGES ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 17;
        sgges("V", "V", "S", slctes, 2, a, 2, b, 2, &sdim, r1, r2,
              r3, q, 2, u, 1, w, 1, bw, &info);
        chkxer("SGGES ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 19;
        sgges("V", "V", "S", slctes, 2, a, 2, b, 2, &sdim, r1, r2,
              r3, q, 2, u, 2, w, 1, bw, &info);
        chkxer("SGGES ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 11;

        /* SGGES3 */

        strcpy(xerbla_srnamt, "SGGES3");
        xerbla_infot = 1;
        sgges3("/", "N", "S", slctes, 1, a, 1, b, 1, &sdim, r1,
               r2, r3, q, 1, u, 1, w, 1, bw, &info);
        chkxer("SGGES3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        sgges3("N", "/", "S", slctes, 1, a, 1, b, 1, &sdim, r1,
               r2, r3, q, 1, u, 1, w, 1, bw, &info);
        chkxer("SGGES3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        sgges3("N", "V", "/", slctes, 1, a, 1, b, 1, &sdim, r1,
               r2, r3, q, 1, u, 1, w, 1, bw, &info);
        chkxer("SGGES3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        sgges3("N", "V", "S", slctes, -1, a, 1, b, 1, &sdim, r1,
               r2, r3, q, 1, u, 1, w, 1, bw, &info);
        chkxer("SGGES3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        sgges3("N", "V", "S", slctes, 1, a, 0, b, 1, &sdim, r1,
               r2, r3, q, 1, u, 1, w, 1, bw, &info);
        chkxer("SGGES3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        sgges3("N", "V", "S", slctes, 1, a, 1, b, 0, &sdim, r1,
               r2, r3, q, 1, u, 1, w, 1, bw, &info);
        chkxer("SGGES3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 15;
        sgges3("N", "V", "S", slctes, 1, a, 1, b, 1, &sdim, r1,
               r2, r3, q, 0, u, 1, w, 1, bw, &info);
        chkxer("SGGES3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 15;
        sgges3("V", "V", "S", slctes, 2, a, 2, b, 2, &sdim, r1,
               r2, r3, q, 1, u, 2, w, 1, bw, &info);
        chkxer("SGGES3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 17;
        sgges3("N", "V", "S", slctes, 1, a, 1, b, 1, &sdim, r1,
               r2, r3, q, 1, u, 0, w, 1, bw, &info);
        chkxer("SGGES3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 17;
        sgges3("V", "V", "S", slctes, 2, a, 2, b, 2, &sdim, r1,
               r2, r3, q, 2, u, 1, w, 1, bw, &info);
        chkxer("SGGES3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 19;
        sgges3("V", "V", "S", slctes, 2, a, 2, b, 2, &sdim, r1,
               r2, r3, q, 2, u, 2, w, 1, bw, &info);
        chkxer("SGGES3", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 11;

        /* SGGESX */

        strcpy(xerbla_srnamt, "SGGESX");
        xerbla_infot = 1;
        sggesx("/", "N", "S", slctsx, "N", 1, a, 1, b, 1, &sdim,
               r1, r2, r3, q, 1, u, 1, rce, rcv, w, 1, iw, 1, bw,
               &info);
        chkxer("SGGESX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        sggesx("N", "/", "S", slctsx, "N", 1, a, 1, b, 1, &sdim,
               r1, r2, r3, q, 1, u, 1, rce, rcv, w, 1, iw, 1, bw,
               &info);
        chkxer("SGGESX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        sggesx("V", "V", "/", slctsx, "N", 1, a, 1, b, 1, &sdim,
               r1, r2, r3, q, 1, u, 1, rce, rcv, w, 1, iw, 1, bw,
               &info);
        chkxer("SGGESX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        sggesx("V", "V", "S", slctsx, "/", 1, a, 1, b, 1, &sdim,
               r1, r2, r3, q, 1, u, 1, rce, rcv, w, 1, iw, 1, bw,
               &info);
        chkxer("SGGESX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        sggesx("V", "V", "S", slctsx, "B", -1, a, 1, b, 1, &sdim,
               r1, r2, r3, q, 1, u, 1, rce, rcv, w, 1, iw, 1, bw,
               &info);
        chkxer("SGGESX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        sggesx("V", "V", "S", slctsx, "B", 1, a, 0, b, 1, &sdim,
               r1, r2, r3, q, 1, u, 1, rce, rcv, w, 1, iw, 1, bw,
               &info);
        chkxer("SGGESX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        sggesx("V", "V", "S", slctsx, "B", 1, a, 1, b, 0, &sdim,
               r1, r2, r3, q, 1, u, 1, rce, rcv, w, 1, iw, 1, bw,
               &info);
        chkxer("SGGESX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 16;
        sggesx("V", "V", "S", slctsx, "B", 1, a, 1, b, 1, &sdim,
               r1, r2, r3, q, 0, u, 1, rce, rcv, w, 1, iw, 1, bw,
               &info);
        chkxer("SGGESX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 16;
        sggesx("V", "V", "S", slctsx, "B", 2, a, 2, b, 2, &sdim,
               r1, r2, r3, q, 1, u, 1, rce, rcv, w, 1, iw, 1, bw,
               &info);
        chkxer("SGGESX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 18;
        sggesx("V", "V", "S", slctsx, "B", 1, a, 1, b, 1, &sdim,
               r1, r2, r3, q, 1, u, 0, rce, rcv, w, 1, iw, 1, bw,
               &info);
        chkxer("SGGESX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 18;
        sggesx("V", "V", "S", slctsx, "B", 2, a, 2, b, 2, &sdim,
               r1, r2, r3, q, 2, u, 1, rce, rcv, w, 1, iw, 1, bw,
               &info);
        chkxer("SGGESX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 22;
        sggesx("V", "V", "S", slctsx, "B", 2, a, 2, b, 2, &sdim,
               r1, r2, r3, q, 2, u, 2, rce, rcv, w, 1, iw, 1, bw,
               &info);
        chkxer("SGGESX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 24;
        sggesx("V", "V", "S", slctsx, "V", 1, a, 1, b, 1, &sdim,
               r1, r2, r3, q, 1, u, 1, rce, rcv, w, 32, iw, 0,
               bw, &info);
        chkxer("SGGESX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 13;

        /* SGGEV */

        strcpy(xerbla_srnamt, "SGGEV ");
        xerbla_infot = 1;
        sggev("/", "N", 1, a, 1, b, 1, r1, r2, r3, q, 1, u, 1, w,
              1, &info);
        chkxer("SGGEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        sggev("N", "/", 1, a, 1, b, 1, r1, r2, r3, q, 1, u, 1, w,
              1, &info);
        chkxer("SGGEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        sggev("V", "V", -1, a, 1, b, 1, r1, r2, r3, q, 1, u, 1,
              w, 1, &info);
        chkxer("SGGEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        sggev("V", "V", 1, a, 0, b, 1, r1, r2, r3, q, 1, u, 1, w,
              1, &info);
        chkxer("SGGEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        sggev("V", "V", 1, a, 1, b, 0, r1, r2, r3, q, 1, u, 1, w,
              1, &info);
        chkxer("SGGEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 12;
        sggev("N", "V", 1, a, 1, b, 1, r1, r2, r3, q, 0, u, 1, w,
              1, &info);
        chkxer("SGGEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 12;
        sggev("V", "V", 2, a, 2, b, 2, r1, r2, r3, q, 1, u, 2, w,
              1, &info);
        chkxer("SGGEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 14;
        sggev("V", "N", 2, a, 2, b, 2, r1, r2, r3, q, 2, u, 0, w,
              1, &info);
        chkxer("SGGEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 14;
        sggev("V", "V", 2, a, 2, b, 2, r1, r2, r3, q, 2, u, 1, w,
              1, &info);
        chkxer("SGGEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 16;
        sggev("V", "V", 1, a, 1, b, 1, r1, r2, r3, q, 1, u, 1, w,
              1, &info);
        chkxer("SGGEV ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 10;

        /* SGGEV3 */

        strcpy(xerbla_srnamt, "SGGEV3 ");
        xerbla_infot = 1;
        sggev3("/", "N", 1, a, 1, b, 1, r1, r2, r3, q, 1, u, 1,
               w, 1, &info);
        chkxer("SGGEV3 ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        sggev3("N", "/", 1, a, 1, b, 1, r1, r2, r3, q, 1, u, 1,
               w, 1, &info);
        chkxer("SGGEV3 ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        sggev3("V", "V", -1, a, 1, b, 1, r1, r2, r3, q, 1, u, 1,
               w, 1, &info);
        chkxer("SGGEV3 ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        sggev3("V", "V", 1, a, 0, b, 1, r1, r2, r3, q, 1, u, 1,
               w, 1, &info);
        chkxer("SGGEV3 ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        sggev3("V", "V", 1, a, 1, b, 0, r1, r2, r3, q, 1, u, 1,
               w, 1, &info);
        chkxer("SGGEV3 ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 12;
        sggev3("N", "V", 1, a, 1, b, 1, r1, r2, r3, q, 0, u, 1,
               w, 1, &info);
        chkxer("SGGEV3 ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 12;
        sggev3("V", "V", 2, a, 2, b, 2, r1, r2, r3, q, 1, u, 2,
               w, 1, &info);
        chkxer("SGGEV3 ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 14;
        sggev3("V", "N", 2, a, 2, b, 2, r1, r2, r3, q, 2, u, 0,
               w, 1, &info);
        chkxer("SGGEV3 ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 14;
        sggev3("V", "V", 2, a, 2, b, 2, r1, r2, r3, q, 2, u, 1,
               w, 1, &info);
        chkxer("SGGEV3 ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 16;
        sggev3("V", "V", 1, a, 1, b, 1, r1, r2, r3, q, 1, u, 1,
               w, 1, &info);
        chkxer("SGGEV3 ", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 10;

        /* SGGEVX */

        strcpy(xerbla_srnamt, "SGGEVX");
        xerbla_infot = 1;
        sggevx("/", "N", "N", "N", 1, a, 1, b, 1, r1, r2, r3, q,
               1, u, 1, &ilo, &ihi, ls, rs, &anrm, &bnrm, rce, rcv,
               w, 1, iw, bw, &info);
        chkxer("SGGEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        sggevx("N", "/", "N", "N", 1, a, 1, b, 1, r1, r2, r3, q,
               1, u, 1, &ilo, &ihi, ls, rs, &anrm, &bnrm, rce, rcv,
               w, 1, iw, bw, &info);
        chkxer("SGGEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        sggevx("N", "N", "/", "N", 1, a, 1, b, 1, r1, r2, r3, q,
               1, u, 1, &ilo, &ihi, ls, rs, &anrm, &bnrm, rce, rcv,
               w, 1, iw, bw, &info);
        chkxer("SGGEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        sggevx("N", "N", "N", "/", 1, a, 1, b, 1, r1, r2, r3, q,
               1, u, 1, &ilo, &ihi, ls, rs, &anrm, &bnrm, rce, rcv,
               w, 1, iw, bw, &info);
        chkxer("SGGEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        sggevx("N", "N", "N", "N", -1, a, 1, b, 1, r1, r2, r3, q,
               1, u, 1, &ilo, &ihi, ls, rs, &anrm, &bnrm, rce, rcv,
               w, 1, iw, bw, &info);
        chkxer("SGGEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        sggevx("N", "N", "N", "N", 1, a, 0, b, 1, r1, r2, r3, q,
               1, u, 1, &ilo, &ihi, ls, rs, &anrm, &bnrm, rce, rcv,
               w, 1, iw, bw, &info);
        chkxer("SGGEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        sggevx("N", "N", "N", "N", 1, a, 1, b, 0, r1, r2, r3, q,
               1, u, 1, &ilo, &ihi, ls, rs, &anrm, &bnrm, rce, rcv,
               w, 1, iw, bw, &info);
        chkxer("SGGEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 14;
        sggevx("N", "N", "N", "N", 1, a, 1, b, 1, r1, r2, r3, q,
               0, u, 1, &ilo, &ihi, ls, rs, &anrm, &bnrm, rce, rcv,
               w, 1, iw, bw, &info);
        chkxer("SGGEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 14;
        sggevx("N", "V", "N", "N", 2, a, 2, b, 2, r1, r2, r3, q,
               1, u, 2, &ilo, &ihi, ls, rs, &anrm, &bnrm, rce, rcv,
               w, 1, iw, bw, &info);
        chkxer("SGGEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 16;
        sggevx("N", "N", "N", "N", 1, a, 1, b, 1, r1, r2, r3, q,
               1, u, 0, &ilo, &ihi, ls, rs, &anrm, &bnrm, rce, rcv,
               w, 1, iw, bw, &info);
        chkxer("SGGEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 16;
        sggevx("N", "N", "V", "N", 2, a, 2, b, 2, r1, r2, r3, q,
               2, u, 1, &ilo, &ihi, ls, rs, &anrm, &bnrm, rce, rcv,
               w, 1, iw, bw, &info);
        chkxer("SGGEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 26;
        sggevx("N", "N", "V", "N", 2, a, 2, b, 2, r1, r2, r3, q,
               2, u, 2, &ilo, &ihi, ls, rs, &anrm, &bnrm, rce, rcv,
               w, 1, iw, bw, &info);
        chkxer("SGGEVX", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 12;

        /* STGEXC */

        strcpy(xerbla_srnamt, "STGEXC");
        xerbla_infot = 3;
        stgexc(1, 1, -1, a, 1, b, 1, q, 1, z, 1, &ifst,
               &ilst, w, 1, &info);
        chkxer("STGEXC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        stgexc(1, 1, 1, a, 0, b, 1, q, 1, z, 1, &ifst,
               &ilst, w, 1, &info);
        chkxer("STGEXC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        stgexc(1, 1, 1, a, 1, b, 0, q, 1, z, 1, &ifst,
               &ilst, w, 1, &info);
        chkxer("STGEXC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        stgexc(0, 1, 1, a, 1, b, 1, q, 0, z, 1, &ifst,
               &ilst, w, 1, &info);
        chkxer("STGEXC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        stgexc(1, 1, 1, a, 1, b, 1, q, 0, z, 1, &ifst,
               &ilst, w, 1, &info);
        chkxer("STGEXC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 11;
        stgexc(1, 0, 1, a, 1, b, 1, q, 1, z, 0, &ifst,
               &ilst, w, 1, &info);
        chkxer("STGEXC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 11;
        stgexc(1, 1, 1, a, 1, b, 1, q, 1, z, 0, &ifst,
               &ilst, w, 1, &info);
        chkxer("STGEXC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 15;
        stgexc(1, 1, 1, a, 1, b, 1, q, 1, z, 1, &ifst,
               &ilst, w, 0, &info);
        chkxer("STGEXC", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 8;

        /* STGSEN */

        strcpy(xerbla_srnamt, "STGSEN");
        xerbla_infot = 1;
        stgsen(-1, 1, 1, sel, 1, a, 1, b, 1, r1, r2,
               r3, q, 1, z, 1, &m, &tola, &tolb, rcv, w, 1, iw, 1,
               &info);
        chkxer("STGSEN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 5;
        stgsen(1, 1, 1, sel, -1, a, 1, b, 1, r1, r2,
               r3, q, 1, z, 1, &m, &tola, &tolb, rcv, w, 1, iw, 1,
               &info);
        chkxer("STGSEN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 7;
        stgsen(1, 1, 1, sel, 1, a, 0, b, 1, r1, r2, r3,
               q, 1, z, 1, &m, &tola, &tolb, rcv, w, 1, iw, 1,
               &info);
        chkxer("STGSEN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 9;
        stgsen(1, 1, 1, sel, 1, a, 1, b, 0, r1, r2, r3,
               q, 1, z, 1, &m, &tola, &tolb, rcv, w, 1, iw, 1,
               &info);
        chkxer("STGSEN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 14;
        stgsen(1, 1, 1, sel, 1, a, 1, b, 1, r1, r2, r3,
               q, 0, z, 1, &m, &tola, &tolb, rcv, w, 1, iw, 1,
               &info);
        chkxer("STGSEN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 16;
        stgsen(1, 1, 1, sel, 1, a, 1, b, 1, r1, r2, r3,
               q, 1, z, 0, &m, &tola, &tolb, rcv, w, 1, iw, 1,
               &info);
        chkxer("STGSEN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 22;
        stgsen(0, 1, 1, sel, 1, a, 1, b, 1, r1, r2, r3,
               q, 1, z, 1, &m, &tola, &tolb, rcv, w, 1, iw, 1,
               &info);
        chkxer("STGSEN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 22;
        stgsen(1, 1, 1, sel, 1, a, 1, b, 1, r1, r2, r3,
               q, 1, z, 1, &m, &tola, &tolb, rcv, w, 1, iw, 1,
               &info);
        chkxer("STGSEN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 22;
        stgsen(2, 1, 1, sel, 1, a, 1, b, 1, r1, r2, r3,
               q, 1, z, 1, &m, &tola, &tolb, rcv, w, 1, iw, 1,
               &info);
        chkxer("STGSEN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 24;
        stgsen(0, 1, 1, sel, 1, a, 1, b, 1, r1, r2, r3,
               q, 1, z, 1, &m, &tola, &tolb, rcv, w, 20, iw, 0,
               &info);
        chkxer("STGSEN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 24;
        stgsen(1, 1, 1, sel, 1, a, 1, b, 1, r1, r2, r3,
               q, 1, z, 1, &m, &tola, &tolb, rcv, w, 20, iw, 0,
               &info);
        chkxer("STGSEN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 24;
        stgsen(2, 1, 1, sel, 1, a, 1, b, 1, r1, r2, r3,
               q, 1, z, 1, &m, &tola, &tolb, rcv, w, 20, iw, 1,
               &info);
        chkxer("STGSEN", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 12;

        /* STGSNA */

        strcpy(xerbla_srnamt, "STGSNA");
        xerbla_infot = 1;
        stgsna("/", "A", sel, 1, a, 1, b, 1, q, 1, u, 1, r1, r2,
               1, &m, w, 1, iw, &info);
        chkxer("STGSNA", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        stgsna("B", "/", sel, 1, a, 1, b, 1, q, 1, u, 1, r1, r2,
               1, &m, w, 1, iw, &info);
        chkxer("STGSNA", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        stgsna("B", "A", sel, -1, a, 1, b, 1, q, 1, u, 1, r1, r2,
               1, &m, w, 1, iw, &info);
        chkxer("STGSNA", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        stgsna("B", "A", sel, 1, a, 0, b, 1, q, 1, u, 1, r1, r2,
               1, &m, w, 1, iw, &info);
        chkxer("STGSNA", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        stgsna("B", "A", sel, 1, a, 1, b, 0, q, 1, u, 1, r1, r2,
               1, &m, w, 1, iw, &info);
        chkxer("STGSNA", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        stgsna("E", "A", sel, 1, a, 1, b, 1, q, 0, u, 1, r1, r2,
               1, &m, w, 1, iw, &info);
        chkxer("STGSNA", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 12;
        stgsna("E", "A", sel, 1, a, 1, b, 1, q, 1, u, 0, r1, r2,
               1, &m, w, 1, iw, &info);
        chkxer("STGSNA", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 15;
        stgsna("E", "A", sel, 1, a, 1, b, 1, q, 1, u, 1, r1, r2,
               0, &m, w, 1, iw, &info);
        chkxer("STGSNA", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 18;
        stgsna("E", "A", sel, 1, a, 1, b, 1, q, 1, u, 1, r1, r2,
               1, &m, w, 0, iw, &info);
        chkxer("STGSNA", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 9;

        /* STGSYL */

        strcpy(xerbla_srnamt, "STGSYL");
        xerbla_infot = 1;
        stgsyl("/", 0, 1, 1, a, 1, b, 1, q, 1, u, 1, v, 1, z, 1,
               &scale, &dif, w, 1, iw, &info);
        chkxer("STGSYL", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 2;
        stgsyl("N", -1, 1, 1, a, 1, b, 1, q, 1, u, 1, v, 1, z, 1,
               &scale, &dif, w, 1, iw, &info);
        chkxer("STGSYL", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 3;
        stgsyl("N", 0, 0, 1, a, 1, b, 1, q, 1, u, 1, v, 1, z, 1,
               &scale, &dif, w, 1, iw, &info);
        chkxer("STGSYL", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 4;
        stgsyl("N", 0, 1, 0, a, 1, b, 1, q, 1, u, 1, v, 1, z, 1,
               &scale, &dif, w, 1, iw, &info);
        chkxer("STGSYL", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 6;
        stgsyl("N", 0, 1, 1, a, 0, b, 1, q, 1, u, 1, v, 1, z, 1,
               &scale, &dif, w, 1, iw, &info);
        chkxer("STGSYL", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 8;
        stgsyl("N", 0, 1, 1, a, 1, b, 0, q, 1, u, 1, v, 1, z, 1,
               &scale, &dif, w, 1, iw, &info);
        chkxer("STGSYL", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 10;
        stgsyl("N", 0, 1, 1, a, 1, b, 1, q, 0, u, 1, v, 1, z, 1,
               &scale, &dif, w, 1, iw, &info);
        chkxer("STGSYL", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 12;
        stgsyl("N", 0, 1, 1, a, 1, b, 1, q, 1, u, 0, v, 1, z, 1,
               &scale, &dif, w, 1, iw, &info);
        chkxer("STGSYL", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 14;
        stgsyl("N", 0, 1, 1, a, 1, b, 1, q, 1, u, 1, v, 0, z, 1,
               &scale, &dif, w, 1, iw, &info);
        chkxer("STGSYL", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 16;
        stgsyl("N", 0, 1, 1, a, 1, b, 1, q, 1, u, 1, v, 1, z, 0,
               &scale, &dif, w, 1, iw, &info);
        chkxer("STGSYL", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 20;
        stgsyl("N", 1, 1, 1, a, 1, b, 1, q, 1, u, 1, v, 1, z, 1,
               &scale, &dif, w, 1, iw, &info);
        chkxer("STGSYL", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        xerbla_infot = 20;
        stgsyl("N", 2, 1, 1, a, 1, b, 1, q, 1, u, 1, v, 1, z, 1,
               &scale, &dif, w, 1, iw, &info);
        chkxer("STGSYL", xerbla_infot, &xerbla_lerr, &xerbla_ok);
        *nt = *nt + 12;
    }

    *ok = xerbla_ok;

    xerbla_srnamt[0] = '\0';
    xerbla_infot = 0;
}
