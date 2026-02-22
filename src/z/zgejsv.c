/**
 * @file zgejsv.c
 * @brief ZGEJSV computes the SVD of a complex M-by-N matrix using preconditioned
 *        Jacobi rotations with sophisticated preprocessing for high accuracy.
 */

#include "semicolon_lapack_complex_double.h"
#include <complex.h>
#include <math.h>
#include "semicolon_cblas.h"

static const f64 ZERO = 0.0;
static const f64 ONE = 1.0;
static const c128 CZERO = CMPLX(0.0, 0.0);
static const c128 CONE = CMPLX(1.0, 0.0);

/** @cond */
static inline INT max3i(INT a, INT b, INT c) {
    INT m = (a > b) ? a : b;
    return (m > c) ? m : c;
}

static inline INT max5i(INT a, INT b, INT c, INT d, INT e) {
    INT m = (a > b) ? a : b;
    m = (m > c) ? m : c;
    m = (m > d) ? m : d;
    return (m > e) ? m : e;
}
/** @endcond */

void zgejsv(const char* joba, const char* jobu, const char* jobv,
            const char* jobr, const char* jobt, const char* jobp,
            const INT m, const INT n,
            c128* restrict A, const INT lda,
            f64* restrict SVA,
            c128* restrict U, const INT ldu,
            c128* restrict V, const INT ldv,
            c128* restrict cwork, const INT lwork,
            f64* restrict rwork, const INT lrwork,
            INT* restrict iwork, INT* info)
{
    /* Local variables */
    c128 ctemp;
    f64 aapp, aaqq, aatmax, aatmin, big, big1, cond_ok;
    f64 condr1, condr2, entra, entrat, epsln, maxprj, scalem;
    f64 sconda, sfmin, small, temp1, uscal1, uscal2, xsc;
    INT ierr, n1, nr, numrank, p, q, warning;
    INT almort, defr, errest, goscal, jracc, kill, lquery, lsvec;
    INT l2aber, l2kill, l2pert, l2rank, l2tran;
    INT noscal, rowpiv, rsvec, transp;
    INT iwoff = 0;

    /* Workspace query variables */
    INT optwrk, minwrk, minrwrk, miniwrk;
    INT lwcon, lwlqf, lwqp3, lwqrf, lwunmlq, lwunmqr, lwunmqrm;
    INT lwsvdj, lwsvdjv, lrwqp3, lrwcon, lrwsvdj;
    INT lwrk_zgelqf = 0, lwrk_zgeqp3 = 0, lwrk_zgeqp3n, lwrk_zgeqrf = 0;
    INT lwrk_zgesvj, lwrk_zgesvjv, lwrk_zgesvju, lwrk_zunmlq;
    INT lwrk_zunmqr, lwrk_zunmqrm;
    c128 cdummy[1];
    f64 rdummy[1];

    /* Parse boolean flags */
    lsvec  = (jobu[0] == 'U' || jobu[0] == 'u' || jobu[0] == 'F' || jobu[0] == 'f');
    jracc  = (jobv[0] == 'J' || jobv[0] == 'j');
    rsvec  = (jobv[0] == 'V' || jobv[0] == 'v') || jracc;
    rowpiv = (joba[0] == 'F' || joba[0] == 'f' || joba[0] == 'G' || joba[0] == 'g');
    l2rank = (joba[0] == 'R' || joba[0] == 'r');
    l2aber = (joba[0] == 'A' || joba[0] == 'a');
    errest = (joba[0] == 'E' || joba[0] == 'e' || joba[0] == 'G' || joba[0] == 'g');
    l2tran = (jobt[0] == 'T' || jobt[0] == 't') && (m == n);
    l2kill = (jobr[0] == 'R' || jobr[0] == 'r');
    defr   = (jobr[0] == 'N' || jobr[0] == 'n');
    l2pert = (jobp[0] == 'P' || jobp[0] == 'p');

    lquery = (lwork == -1) || (lrwork == -1);

    /* Parameter validation */
    *info = 0;

    if (!(rowpiv || l2rank || l2aber || errest ||
          joba[0] == 'C' || joba[0] == 'c')) {
        *info = -1;
    }
    else if (!(lsvec || jobu[0] == 'N' || jobu[0] == 'n' ||
               ((jobu[0] == 'W' || jobu[0] == 'w') && rsvec && l2tran))) {
        *info = -2;
    }
    else if (!(rsvec || jobv[0] == 'N' || jobv[0] == 'n' ||
               ((jobv[0] == 'W' || jobv[0] == 'w') && lsvec && l2tran))) {
        *info = -3;
    }
    else if (!(l2kill || defr)) {
        *info = -4;
    }
    else if (!(jobt[0] == 'T' || jobt[0] == 't' ||
               jobt[0] == 'N' || jobt[0] == 'n')) {
        *info = -5;
    }
    else if (!(l2pert || jobp[0] == 'N' || jobp[0] == 'n')) {
        *info = -6;
    }
    else if (m < 0) {
        *info = -7;
    }
    else if (n < 0 || n > m) {
        *info = -8;
    }
    else if (lda < m) {
        *info = -10;
    }
    else if (lsvec && ldu < m) {
        *info = -13;
    }
    else if (rsvec && ldv < n) {
        *info = -15;
    }

    /* Workspace query */
    if (*info == 0) {
        lwqp3    = n + 1;
        lwqrf    = (1 > n) ? 1 : n;
        lwlqf    = (1 > n) ? 1 : n;
        lwunmlq  = (1 > n) ? 1 : n;
        lwunmqr  = (1 > n) ? 1 : n;
        lwunmqrm = (1 > m) ? 1 : m;
        lwcon    = 2 * n;
        lwsvdj   = (2 * n > 1) ? 2 * n : 1;
        lwsvdjv  = (2 * n > 1) ? 2 * n : 1;
        lrwqp3   = 2 * n;
        lrwcon   = n;
        lrwsvdj  = n;

        if (lquery) {
            zgeqp3(m, n, A, lda, iwork, NULL, cdummy, -1, rdummy, &ierr);
            lwrk_zgeqp3 = (INT)creal(cdummy[0]);
            zgeqrf(n, n, A, lda, NULL, cdummy, -1, &ierr);
            lwrk_zgeqrf = (INT)creal(cdummy[0]);
            zgelqf(n, n, A, lda, NULL, cdummy, -1, &ierr);
            lwrk_zgelqf = (INT)creal(cdummy[0]);
        }

        optwrk  = 2;
        miniwrk = n;

        if (!lsvec && !rsvec) {
            if (errest) {
                minwrk = n + lwqp3;
                if (n * n + lwcon > minwrk) minwrk = n * n + lwcon;
                if (n + lwqrf > minwrk)     minwrk = n + lwqrf;
                if (lwsvdj > minwrk)        minwrk = lwsvdj;
            } else {
                minwrk = n + lwqp3;
                if (n + lwqrf > minwrk)     minwrk = n + lwqrf;
                if (lwsvdj > minwrk)        minwrk = lwsvdj;
            }
            if (lquery) {
                zgesvj("L", "N", "N", n, n, A, lda, SVA, n,
                       V, ldv, cdummy, -1, rdummy, -1, &ierr);
                lwrk_zgesvj = (INT)creal(cdummy[0]);
                if (errest) {
                    optwrk = n + lwrk_zgeqp3;
                    if (n * n + lwcon > optwrk) optwrk = n * n + lwcon;
                    if (n + lwrk_zgeqrf > optwrk) optwrk = n + lwrk_zgeqrf;
                    if (lwrk_zgesvj > optwrk)  optwrk = lwrk_zgesvj;
                } else {
                    optwrk = n + lwrk_zgeqp3;
                    if (n + lwrk_zgeqrf > optwrk) optwrk = n + lwrk_zgeqrf;
                    if (lwrk_zgesvj > optwrk)  optwrk = lwrk_zgesvj;
                }
            }
            if (l2tran || rowpiv) {
                if (errest) {
                    minrwrk = max5i(7, 2 * m, lrwqp3, lrwcon, lrwsvdj);
                } else {
                    minrwrk = max3i(7, 2 * m, (lrwqp3 > lrwsvdj) ? lrwqp3 : lrwsvdj);
                }
            } else {
                if (errest) {
                    minrwrk = max3i(7, lrwqp3, (lrwcon > lrwsvdj) ? lrwcon : lrwsvdj);
                } else {
                    minrwrk = max3i(7, lrwqp3, lrwsvdj);
                }
            }
            if (rowpiv || l2tran) miniwrk = miniwrk + m;

        } else if (rsvec && !lsvec) {
            if (errest) {
                minwrk = n + lwqp3;
                if (lwcon > minwrk)         minwrk = lwcon;
                if (lwsvdj > minwrk)        minwrk = lwsvdj;
                if (n + lwlqf > minwrk)     minwrk = n + lwlqf;
                if (2 * n + lwqrf > minwrk) minwrk = 2 * n + lwqrf;
                if (n + lwsvdj > minwrk)    minwrk = n + lwsvdj;
                if (n + lwunmlq > minwrk)   minwrk = n + lwunmlq;
            } else {
                minwrk = n + lwqp3;
                if (lwsvdj > minwrk)        minwrk = lwsvdj;
                if (n + lwlqf > minwrk)     minwrk = n + lwlqf;
                if (2 * n + lwqrf > minwrk) minwrk = 2 * n + lwqrf;
                if (n + lwsvdj > minwrk)    minwrk = n + lwsvdj;
                if (n + lwunmlq > minwrk)   minwrk = n + lwunmlq;
            }
            if (lquery) {
                zgesvj("L", "U", "N", n, n, U, ldu, SVA, n, A,
                       lda, cdummy, -1, rdummy, -1, &ierr);
                lwrk_zgesvj = (INT)creal(cdummy[0]);
                zunmlq("L", "C", n, n, n, A, lda, NULL,
                       V, ldv, cdummy, -1, &ierr);
                lwrk_zunmlq = (INT)creal(cdummy[0]);
                if (errest) {
                    optwrk = n + lwrk_zgeqp3;
                    if (lwcon > optwrk)              optwrk = lwcon;
                    if (lwrk_zgesvj > optwrk)        optwrk = lwrk_zgesvj;
                    if (n + lwrk_zgelqf > optwrk)    optwrk = n + lwrk_zgelqf;
                    if (2 * n + lwrk_zgeqrf > optwrk) optwrk = 2 * n + lwrk_zgeqrf;
                    if (n + lwrk_zgesvj > optwrk)    optwrk = n + lwrk_zgesvj;
                    if (n + lwrk_zunmlq > optwrk)    optwrk = n + lwrk_zunmlq;
                } else {
                    optwrk = n + lwrk_zgeqp3;
                    if (lwrk_zgesvj > optwrk)        optwrk = lwrk_zgesvj;
                    if (n + lwrk_zgelqf > optwrk)    optwrk = n + lwrk_zgelqf;
                    if (2 * n + lwrk_zgeqrf > optwrk) optwrk = 2 * n + lwrk_zgeqrf;
                    if (n + lwrk_zgesvj > optwrk)    optwrk = n + lwrk_zgesvj;
                    if (n + lwrk_zunmlq > optwrk)    optwrk = n + lwrk_zunmlq;
                }
            }
            if (l2tran || rowpiv) {
                if (errest) {
                    minrwrk = max5i(7, 2 * m, lrwqp3, lrwsvdj, lrwcon);
                } else {
                    minrwrk = max3i(7, 2 * m, (lrwqp3 > lrwsvdj) ? lrwqp3 : lrwsvdj);
                }
            } else {
                if (errest) {
                    minrwrk = max3i(7, lrwqp3, (lrwsvdj > lrwcon) ? lrwsvdj : lrwcon);
                } else {
                    minrwrk = max3i(7, lrwqp3, lrwsvdj);
                }
            }
            if (rowpiv || l2tran) miniwrk = miniwrk + m;

        } else if (lsvec && !rsvec) {
            if (errest) {
                minwrk = n + max5i(lwqp3, lwcon, n + lwqrf, lwsvdj, lwunmqrm);
            } else {
                INT mx = lwqp3;
                if (n + lwqrf > mx) mx = n + lwqrf;
                if (lwsvdj > mx) mx = lwsvdj;
                if (lwunmqrm > mx) mx = lwunmqrm;
                minwrk = n + mx;
            }
            if (lquery) {
                zgesvj("L", "U", "N", n, n, U, ldu, SVA, n, A,
                       lda, cdummy, -1, rdummy, -1, &ierr);
                lwrk_zgesvj = (INT)creal(cdummy[0]);
                zunmqr("L", "N", m, n, n, A, lda, NULL, U,
                       ldu, cdummy, -1, &ierr);
                lwrk_zunmqrm = (INT)creal(cdummy[0]);
                if (errest) {
                    optwrk = n + max5i(lwrk_zgeqp3, lwcon, n + lwrk_zgeqrf,
                                       lwrk_zgesvj, lwrk_zunmqrm);
                } else {
                    INT mx = lwrk_zgeqp3;
                    if (n + lwrk_zgeqrf > mx) mx = n + lwrk_zgeqrf;
                    if (lwrk_zgesvj > mx)  mx = lwrk_zgesvj;
                    if (lwrk_zunmqrm > mx) mx = lwrk_zunmqrm;
                    optwrk = n + mx;
                }
            }
            if (l2tran || rowpiv) {
                if (errest) {
                    minrwrk = max5i(7, 2 * m, lrwqp3, lrwsvdj, lrwcon);
                } else {
                    minrwrk = max3i(7, 2 * m, (lrwqp3 > lrwsvdj) ? lrwqp3 : lrwsvdj);
                }
            } else {
                if (errest) {
                    minrwrk = max3i(7, lrwqp3, (lrwsvdj > lrwcon) ? lrwsvdj : lrwcon);
                } else {
                    minrwrk = max3i(7, lrwqp3, lrwsvdj);
                }
            }
            if (rowpiv || l2tran) miniwrk = miniwrk + m;

        } else {
            /* Full SVD requested */
            if (!jracc) {
                if (errest) {
                    INT t1 = n + lwqp3;
                    INT t2 = n + lwcon;
                    INT t3 = 2 * n + n * n + lwcon;
                    INT t4 = 2 * n + lwqrf;
                    INT t5 = 2 * n + lwqp3;
                    INT t6 = 2 * n + n * n + n + lwlqf;
                    INT t7 = 2 * n + n * n + n + n * n + lwcon;
                    INT t8 = 2 * n + n * n + n + lwsvdj;
                    INT t9 = 2 * n + n * n + n + lwsvdjv;
                    INT t10 = 2 * n + n * n + n + lwunmqr;
                    INT t11 = 2 * n + n * n + n + lwunmlq;
                    INT t12 = n + n * n + lwsvdj;
                    INT t13 = n + lwunmqrm;
                    minwrk = t1;
                    if (t2 > minwrk)  minwrk = t2;
                    if (t3 > minwrk)  minwrk = t3;
                    if (t4 > minwrk)  minwrk = t4;
                    if (t5 > minwrk)  minwrk = t5;
                    if (t6 > minwrk)  minwrk = t6;
                    if (t7 > minwrk)  minwrk = t7;
                    if (t8 > minwrk)  minwrk = t8;
                    if (t9 > minwrk)  minwrk = t9;
                    if (t10 > minwrk) minwrk = t10;
                    if (t11 > minwrk) minwrk = t11;
                    if (t12 > minwrk) minwrk = t12;
                    if (t13 > minwrk) minwrk = t13;
                } else {
                    INT t1 = n + lwqp3;
                    INT t2 = 2 * n + n * n + lwcon;
                    INT t3 = 2 * n + lwqrf;
                    INT t4 = 2 * n + lwqp3;
                    INT t5 = 2 * n + n * n + n + lwlqf;
                    INT t6 = 2 * n + n * n + n + n * n + lwcon;
                    INT t7 = 2 * n + n * n + n + lwsvdj;
                    INT t8 = 2 * n + n * n + n + lwsvdjv;
                    INT t9 = 2 * n + n * n + n + lwunmqr;
                    INT t10 = 2 * n + n * n + n + lwunmlq;
                    INT t11 = n + n * n + lwsvdj;
                    INT t12 = n + lwunmqrm;
                    minwrk = t1;
                    if (t2 > minwrk)  minwrk = t2;
                    if (t3 > minwrk)  minwrk = t3;
                    if (t4 > minwrk)  minwrk = t4;
                    if (t5 > minwrk)  minwrk = t5;
                    if (t6 > minwrk)  minwrk = t6;
                    if (t7 > minwrk)  minwrk = t7;
                    if (t8 > minwrk)  minwrk = t8;
                    if (t9 > minwrk)  minwrk = t9;
                    if (t10 > minwrk) minwrk = t10;
                    if (t11 > minwrk) minwrk = t11;
                    if (t12 > minwrk) minwrk = t12;
                }
                miniwrk = miniwrk + n;
                if (rowpiv || l2tran) miniwrk = miniwrk + m;
            } else {
                /* JRACC */
                if (errest) {
                    INT t1 = n + lwqp3;
                    INT t2 = n + lwcon;
                    INT t3 = 2 * n + lwqrf;
                    INT t4 = 2 * n + n * n + lwsvdjv;
                    INT t5 = 2 * n + n * n + n + lwunmqr;
                    INT t6 = n + lwunmqrm;
                    minwrk = t1;
                    if (t2 > minwrk) minwrk = t2;
                    if (t3 > minwrk) minwrk = t3;
                    if (t4 > minwrk) minwrk = t4;
                    if (t5 > minwrk) minwrk = t5;
                    if (t6 > minwrk) minwrk = t6;
                } else {
                    INT t1 = n + lwqp3;
                    INT t2 = 2 * n + lwqrf;
                    INT t3 = 2 * n + n * n + lwsvdjv;
                    INT t4 = 2 * n + n * n + n + lwunmqr;
                    INT t5 = n + lwunmqrm;
                    minwrk = t1;
                    if (t2 > minwrk) minwrk = t2;
                    if (t3 > minwrk) minwrk = t3;
                    if (t4 > minwrk) minwrk = t4;
                    if (t5 > minwrk) minwrk = t5;
                }
                if (rowpiv || l2tran) miniwrk = miniwrk + m;
            }
            if (lquery) {
                zunmqr("L", "N", m, n, n, A, lda, NULL, U,
                       ldu, cdummy, -1, &ierr);
                lwrk_zunmqrm = (INT)creal(cdummy[0]);
                zunmqr("L", "N", n, n, n, A, lda, NULL, U,
                       ldu, cdummy, -1, &ierr);
                lwrk_zunmqr = (INT)creal(cdummy[0]);
                if (!jracc) {
                    zgeqp3(n, n, A, lda, iwork, NULL, cdummy,
                           -1, rdummy, &ierr);
                    lwrk_zgeqp3n = (INT)creal(cdummy[0]);
                    zgesvj("L", "U", "N", n, n, U, ldu, SVA,
                           n, V, ldv, cdummy, -1, rdummy, -1, &ierr);
                    lwrk_zgesvj = (INT)creal(cdummy[0]);
                    zgesvj("U", "U", "N", n, n, U, ldu, SVA,
                           n, V, ldv, cdummy, -1, rdummy, -1, &ierr);
                    lwrk_zgesvju = (INT)creal(cdummy[0]);
                    zgesvj("L", "U", "V", n, n, U, ldu, SVA,
                           n, V, ldv, cdummy, -1, rdummy, -1, &ierr);
                    lwrk_zgesvjv = (INT)creal(cdummy[0]);
                    zunmlq("L", "C", n, n, n, A, lda, NULL,
                           V, ldv, cdummy, -1, &ierr);
                    lwrk_zunmlq = (INT)creal(cdummy[0]);
                    if (errest) {
                        INT t1 = n + lwrk_zgeqp3;
                        INT t2 = n + lwcon;
                        INT t3 = 2 * n + n * n + lwcon;
                        INT t4 = 2 * n + lwrk_zgeqrf;
                        INT t5 = 2 * n + lwrk_zgeqp3n;
                        INT t6 = 2 * n + n * n + n + lwrk_zgelqf;
                        INT t7 = 2 * n + n * n + n + n * n + lwcon;
                        INT t8 = 2 * n + n * n + n + lwrk_zgesvj;
                        INT t9 = 2 * n + n * n + n + lwrk_zgesvjv;
                        INT t10 = 2 * n + n * n + n + lwrk_zunmqr;
                        INT t11 = 2 * n + n * n + n + lwrk_zunmlq;
                        INT t12 = n + n * n + lwrk_zgesvju;
                        INT t13 = n + lwrk_zunmqrm;
                        optwrk = t1;
                        if (t2 > optwrk)  optwrk = t2;
                        if (t3 > optwrk)  optwrk = t3;
                        if (t4 > optwrk)  optwrk = t4;
                        if (t5 > optwrk)  optwrk = t5;
                        if (t6 > optwrk)  optwrk = t6;
                        if (t7 > optwrk)  optwrk = t7;
                        if (t8 > optwrk)  optwrk = t8;
                        if (t9 > optwrk)  optwrk = t9;
                        if (t10 > optwrk) optwrk = t10;
                        if (t11 > optwrk) optwrk = t11;
                        if (t12 > optwrk) optwrk = t12;
                        if (t13 > optwrk) optwrk = t13;
                    } else {
                        INT t1 = n + lwrk_zgeqp3;
                        INT t2 = 2 * n + n * n + lwcon;
                        INT t3 = 2 * n + lwrk_zgeqrf;
                        INT t4 = 2 * n + lwrk_zgeqp3n;
                        INT t5 = 2 * n + n * n + n + lwrk_zgelqf;
                        INT t6 = 2 * n + n * n + n + n * n + lwcon;
                        INT t7 = 2 * n + n * n + n + lwrk_zgesvj;
                        INT t8 = 2 * n + n * n + n + lwrk_zgesvjv;
                        INT t9 = 2 * n + n * n + n + lwrk_zunmqr;
                        INT t10 = 2 * n + n * n + n + lwrk_zunmlq;
                        INT t11 = n + n * n + lwrk_zgesvju;
                        INT t12 = n + lwrk_zunmqrm;
                        optwrk = t1;
                        if (t2 > optwrk)  optwrk = t2;
                        if (t3 > optwrk)  optwrk = t3;
                        if (t4 > optwrk)  optwrk = t4;
                        if (t5 > optwrk)  optwrk = t5;
                        if (t6 > optwrk)  optwrk = t6;
                        if (t7 > optwrk)  optwrk = t7;
                        if (t8 > optwrk)  optwrk = t8;
                        if (t9 > optwrk)  optwrk = t9;
                        if (t10 > optwrk) optwrk = t10;
                        if (t11 > optwrk) optwrk = t11;
                        if (t12 > optwrk) optwrk = t12;
                    }
                } else {
                    /* JRACC */
                    zgesvj("L", "U", "V", n, n, U, ldu, SVA,
                           n, V, ldv, cdummy, -1, rdummy, -1, &ierr);
                    lwrk_zgesvjv = (INT)creal(cdummy[0]);
                    zunmqr("L", "N", n, n, n, NULL, n, NULL,
                           V, ldv, cdummy, -1, &ierr);
                    lwrk_zunmqr = (INT)creal(cdummy[0]);
                    zunmqr("L", "N", m, n, n, A, lda, NULL, U,
                           ldu, cdummy, -1, &ierr);
                    lwrk_zunmqrm = (INT)creal(cdummy[0]);
                    if (errest) {
                        INT t1 = n + lwrk_zgeqp3;
                        INT t2 = n + lwcon;
                        INT t3 = 2 * n + lwrk_zgeqrf;
                        INT t4 = 2 * n + n * n;
                        INT t5 = 2 * n + n * n + lwrk_zgesvjv;
                        INT t6 = 2 * n + n * n + n + lwrk_zunmqr;
                        INT t7 = n + lwrk_zunmqrm;
                        optwrk = t1;
                        if (t2 > optwrk) optwrk = t2;
                        if (t3 > optwrk) optwrk = t3;
                        if (t4 > optwrk) optwrk = t4;
                        if (t5 > optwrk) optwrk = t5;
                        if (t6 > optwrk) optwrk = t6;
                        if (t7 > optwrk) optwrk = t7;
                    } else {
                        INT t1 = n + lwrk_zgeqp3;
                        INT t2 = 2 * n + lwrk_zgeqrf;
                        INT t3 = 2 * n + n * n;
                        INT t4 = 2 * n + n * n + lwrk_zgesvjv;
                        INT t5 = 2 * n + n * n + n + lwrk_zunmqr;
                        INT t6 = n + lwrk_zunmqrm;
                        optwrk = t1;
                        if (t2 > optwrk) optwrk = t2;
                        if (t3 > optwrk) optwrk = t3;
                        if (t4 > optwrk) optwrk = t4;
                        if (t5 > optwrk) optwrk = t5;
                        if (t6 > optwrk) optwrk = t6;
                    }
                }
            }
            if (l2tran || rowpiv) {
                minrwrk = max5i(7, 2 * m, lrwqp3, lrwsvdj, lrwcon);
            } else {
                minrwrk = max3i(7, lrwqp3, (lrwsvdj > lrwcon) ? lrwsvdj : lrwcon);
            }
        }

        minwrk  = (2 > minwrk)  ? 2 : minwrk;
        optwrk  = (minwrk > optwrk) ? minwrk : optwrk;
        if (lwork < minwrk && !lquery)  *info = -17;
        if (lrwork < minrwrk && !lquery) *info = -19;
    }

    if (*info != 0) {
        xerbla("ZGEJSV", -(*info));
        return;
    } else if (lquery) {
        cwork[0] = (c128)optwrk;
        cwork[1] = (c128)minwrk;
        rwork[0] = (f64)minrwrk;
        iwork[0] = (4 > miniwrk) ? 4 : miniwrk;
        return;
    }

    /* Quick return for void matrix */
    if (m == 0 || n == 0) {
        iwork[0] = 0; iwork[1] = 0; iwork[2] = 0; iwork[3] = 0;
        rwork[0] = ZERO; rwork[1] = ZERO; rwork[2] = ZERO; rwork[3] = ZERO;
        rwork[4] = ZERO; rwork[5] = ZERO; rwork[6] = ZERO;
        return;
    }

    n1 = n;
    if (lsvec) {
        if (jobu[0] == 'F' || jobu[0] == 'f') n1 = m;
    }

    epsln = dlamch("E");
    sfmin = dlamch("S");
    small = sfmin / epsln;
    big   = dlamch("O");

    /* Initialize SVA(1:N) = diag( ||A e_i||_2 )_1^N */
    scalem = ONE / sqrt((f64)m * (f64)n);
    noscal = 1;
    goscal = 1;

    for (p = 0; p < n; p++) {
        aapp = ZERO;
        aaqq = ONE;
        zlassq(m, &A[p * lda], 1, &aapp, &aaqq);
        if (aapp > big) {
            *info = -9;
            xerbla("ZGEJSV", -(*info));
            return;
        }
        aaqq = sqrt(aaqq);
        if ((aapp < (big / aaqq)) && noscal) {
            SVA[p] = aapp * aaqq;
        } else {
            noscal = 0;
            SVA[p] = aapp * (aaqq * scalem);
            if (goscal) {
                goscal = 0;
                cblas_dscal(p, scalem, SVA, 1);
            }
        }
    }

    if (noscal) scalem = ONE;

    aapp = ZERO;
    aaqq = big;
    for (p = 0; p < n; p++) {
        aapp = (aapp > SVA[p]) ? aapp : SVA[p];
        if (SVA[p] != ZERO) aaqq = (aaqq < SVA[p]) ? aaqq : SVA[p];
    }

    /* Quick return for zero matrix */
    if (aapp == ZERO) {
        if (lsvec) zlaset("G", m, n1, CZERO, CONE, U, ldu);
        if (rsvec) zlaset("G", n, n, CZERO, CONE, V, ldv);
        rwork[0] = ONE;
        rwork[1] = ONE;
        if (errest) rwork[2] = ONE;
        if (lsvec && rsvec) {
            rwork[3] = ONE;
            rwork[4] = ONE;
        }
        if (l2tran) {
            rwork[5] = ZERO;
            rwork[6] = ZERO;
        }
        iwork[0] = 0;
        iwork[1] = 0;
        iwork[2] = 0;
        iwork[3] = -1;
        return;
    }

    warning = 0;
    if (aaqq <= sfmin) {
        l2rank = 1;
        l2kill = 1;
        warning = 1;
    }

    /* Quick return for one-column matrix */
    if (n == 1) {
        if (lsvec) {
            zlascl("G", 0, 0, SVA[0], scalem, m, 1, A, lda, &ierr);
            zlacpy("A", m, 1, A, lda, U, ldu);
            if (n1 != n) {
                zgeqrf(m, n, U, ldu, cwork, &cwork[n], lwork - n, &ierr);
                zungqr(m, n1, 1, U, ldu, cwork, &cwork[n], lwork - n, &ierr);
                cblas_zcopy(m, A, 1, U, 1);
            }
        }
        if (rsvec) {
            V[0] = CONE;
        }
        if (SVA[0] < (big * scalem)) {
            SVA[0] = SVA[0] / scalem;
            scalem = ONE;
        }
        rwork[0] = ONE / scalem;
        rwork[1] = ONE;
        if (SVA[0] != ZERO) {
            iwork[0] = 1;
            if ((SVA[0] / scalem) >= sfmin) {
                iwork[1] = 1;
            } else {
                iwork[1] = 0;
            }
        } else {
            iwork[0] = 0;
            iwork[1] = 0;
        }
        iwork[2] = 0;
        iwork[3] = -1;
        if (errest) rwork[2] = ONE;
        if (lsvec && rsvec) {
            rwork[3] = ONE;
            rwork[4] = ONE;
        }
        if (l2tran) {
            rwork[5] = ZERO;
            rwork[6] = ZERO;
        }
        return;
    }

    transp = 0;

    aatmax = -ONE;
    aatmin = big;
    if (rowpiv || l2tran) {
        if (l2tran) {
            for (p = 0; p < m; p++) {
                xsc   = ZERO;
                temp1 = ONE;
                zlassq(n, &A[p], lda, &xsc, &temp1);
                rwork[m + p] = xsc * scalem;
                rwork[p]     = xsc * (scalem * sqrt(temp1));
                aatmax = (aatmax > rwork[p]) ? aatmax : rwork[p];
                if (rwork[p] != ZERO)
                    aatmin = (aatmin < rwork[p]) ? aatmin : rwork[p];
            }
        } else {
            for (p = 0; p < m; p++) {
                rwork[m + p] = scalem * cabs(A[p + cblas_izamax(n, &A[p], lda) * lda]);
                aatmax = (aatmax > rwork[m + p]) ? aatmax : rwork[m + p];
                aatmin = (aatmin < rwork[m + p]) ? aatmin : rwork[m + p];
            }
        }
    }

    entra  = ZERO;
    entrat = ZERO;
    if (l2tran) {
        xsc   = ZERO;
        temp1 = ONE;
        dlassq(n, SVA, 1, &xsc, &temp1);
        temp1 = ONE / temp1;

        entra = ZERO;
        for (p = 0; p < n; p++) {
            big1 = ((SVA[p] / xsc) * (SVA[p] / xsc)) * temp1;
            if (big1 != ZERO) entra = entra + big1 * log(big1);
        }
        entra = -entra / log((f64)n);

        entrat = ZERO;
        for (p = 0; p < m; p++) {
            big1 = ((rwork[p] / xsc) * (rwork[p] / xsc)) * temp1;
            if (big1 != ZERO) entrat = entrat + big1 * log(big1);
        }
        entrat = -entrat / log((f64)m);

        transp = (entrat < entra);

        if (transp) {
            for (p = 0; p < n - 1; p++) {
                A[p + p * lda] = conj(A[p + p * lda]);
                for (q = p + 1; q < n; q++) {
                    ctemp = conj(A[q + p * lda]);
                    A[q + p * lda] = conj(A[p + q * lda]);
                    A[p + q * lda] = ctemp;
                }
            }
            A[(n - 1) + (n - 1) * lda] = conj(A[(n - 1) + (n - 1) * lda]);

            for (p = 0; p < n; p++) {
                rwork[m + p] = SVA[p];
                SVA[p] = rwork[p];
            }
            temp1  = aapp;
            aapp   = aatmax;
            temp1  = aaqq;
            aaqq   = aatmin;
            kill   = lsvec;
            lsvec  = rsvec;
            rsvec  = kill;
            if (lsvec) n1 = n;

            rowpiv = 1;
        }
    }

    /* Scaling */
    big1  = sqrt(big);
    temp1 = sqrt(big / (f64)n);

    dlascl("G", 0, 0, aapp, temp1, n, 1, SVA, n, &ierr);
    if (aaqq > (aapp * sfmin)) {
        aaqq = (aaqq / aapp) * temp1;
    } else {
        aaqq = (aaqq * temp1) / aapp;
    }
    temp1 = temp1 * scalem;
    zlascl("G", 0, 0, aapp, temp1, m, n, A, lda, &ierr);

    uscal1 = temp1;
    uscal2 = aapp;

    if (l2kill) {
        xsc = sqrt(sfmin);
    } else {
        xsc = small;
        if ((aaqq < sqrt(sfmin)) && lsvec && rsvec) {
            jracc = 1;
        }
    }
    if (aaqq < xsc) {
        for (p = 0; p < n; p++) {
            if (SVA[p] < xsc) {
                zlaset("A", m, 1, CZERO, CZERO, &A[p * lda], lda);
                SVA[p] = ZERO;
            }
        }
    }

    /* Row pivoting */
    if (rowpiv) {
        if ((lsvec && rsvec) && !jracc) {
            iwoff = 2 * n;
        } else {
            iwoff = n;
        }
        for (p = 0; p < m - 1; p++) {
            q = cblas_idamax(m - p, &rwork[m + p], 1) + p;
            iwork[iwoff + p] = q;
            if (p != q) {
                temp1 = rwork[m + p];
                rwork[m + p] = rwork[m + q];
                rwork[m + q] = temp1;
            }
        }
        zlaswp(n, A, lda, 0, m - 2, &iwork[iwoff], 1);
    }

    /* First QR factorization with column pivoting */
    for (p = 0; p < n; p++) iwork[p] = 0;
    zgeqp3(m, n, A, lda, iwork, cwork, &cwork[n], lwork - n, rwork, &ierr);

    /* Rank detection */
    nr = 1;

    if (l2aber) {
        temp1 = sqrt((f64)n) * epsln;
        for (p = 1; p < n; p++) {
            if (cabs(A[p + p * lda]) >= temp1 * cabs(A[0])) {
                nr++;
            } else {
                break;
            }
        }
    } else if (l2rank) {
        temp1 = sqrt(sfmin);
        for (p = 1; p < n; p++) {
            if (cabs(A[p + p * lda]) < epsln * cabs(A[(p - 1) + (p - 1) * lda]) ||
                cabs(A[p + p * lda]) < small ||
                (l2kill && cabs(A[p + p * lda]) < temp1)) {
                break;
            }
            nr++;
        }
    } else {
        temp1 = sqrt(sfmin);
        for (p = 1; p < n; p++) {
            if (cabs(A[p + p * lda]) < small ||
                (l2kill && cabs(A[p + p * lda]) < temp1)) {
                break;
            }
            nr++;
        }
    }

    almort = 0;
    if (nr == n) {
        maxprj = ONE;
        for (p = 1; p < n; p++) {
            temp1 = cabs(A[p + p * lda]) / SVA[iwork[p]];
            maxprj = (maxprj < temp1) ? maxprj : temp1;
        }
        if (maxprj * maxprj >= ONE - (f64)n * epsln) almort = 1;
    }

    sconda = -ONE;
    condr1 = -ONE;
    condr2 = -ONE;

    if (errest) {
        if (n == nr) {
            if (rsvec) {
                zlacpy("U", n, n, A, lda, V, ldv);
                for (p = 0; p < n; p++) {
                    temp1 = SVA[iwork[p]];
                    cblas_zdscal(p + 1, ONE / temp1, &V[p * ldv], 1);
                }
                if (lsvec) {
                    zpocon("U", n, V, ldv, ONE, &temp1, &cwork[n], rwork, &ierr);
                } else {
                    zpocon("U", n, V, ldv, ONE, &temp1, cwork, rwork, &ierr);
                }
            } else if (lsvec) {
                zlacpy("U", n, n, A, lda, U, ldu);
                for (p = 0; p < n; p++) {
                    temp1 = SVA[iwork[p]];
                    cblas_zdscal(p + 1, ONE / temp1, &U[p * ldu], 1);
                }
                zpocon("U", n, U, ldu, ONE, &temp1, &cwork[n], rwork, &ierr);
            } else {
                zlacpy("U", n, n, A, lda, cwork, n);
                for (p = 0; p < n; p++) {
                    temp1 = SVA[iwork[p]];
                    cblas_zdscal(p + 1, ONE / temp1, &cwork[p * n], 1);
                }
                zpocon("U", n, cwork, n, ONE, &temp1, &cwork[n * n], rwork, &ierr);
            }
            if (temp1 != ZERO) {
                sconda = ONE / sqrt(temp1);
            } else {
                sconda = -ONE;
            }
        } else {
            sconda = -ONE;
        }
    }

    l2pert = l2pert && (cabs(A[0]) / cabs(A[(nr - 1) + (nr - 1) * lda]) > sqrt(big1));

    /* -------------------------------------------------------------------- */
    /* Phase 3:                                                             */
    /* -------------------------------------------------------------------- */


    if (!rsvec && !lsvec) {
        /* ============================================================== */
        /* Singular Values only                                           */
        /* ============================================================== */

        /* .. transpose A(1:NR,1:N) */
        for (p = 0; p < (n - 1 < nr ? n - 1 : nr); p++) {
            cblas_zcopy(n - p - 1, &A[p + (p + 1) * lda], lda, &A[(p + 1) + p * lda], 1);
            zlacgv(n - p, &A[p + p * lda], 1);
        }
        if (nr == n) A[(n - 1) + (n - 1) * lda] = conj(A[(n - 1) + (n - 1) * lda]);

        if (!almort) {

            if (l2pert) {
                xsc = epsln / (f64)n;
                for (q = 0; q < nr; q++) {
                    ctemp = CMPLX(xsc * cabs(A[q + q * lda]), 0.0);
                    for (p = 0; p < n; p++) {
                        if ((p > q && cabs(A[p + q * lda]) <= temp1) ||
                            (p < q))
                            A[p + q * lda] = ctemp;
                    }
                }
            } else {
                zlaset("U", nr - 1, nr - 1, CZERO, CZERO, &A[1 * lda], lda);
            }

            /* .. second preconditioning using the QR factorization */
            zgeqrf(n, nr, A, lda, cwork, &cwork[n], lwork - n, &ierr);

            /* .. and transpose upper to lower triangular */
            for (p = 0; p < nr - 1; p++) {
                cblas_zcopy(nr - p - 1, &A[p + (p + 1) * lda], lda, &A[(p + 1) + p * lda], 1);
                zlacgv(nr - p, &A[p + p * lda], 1);
            }

        }

        /* Row-cyclic Jacobi SVD algorithm with column pivoting */
        /* .. again some perturbation (a "background noise") is added */
        /*    to drown denormals */
        if (l2pert) {
            xsc = epsln / (f64)n;
            for (q = 0; q < nr; q++) {
                ctemp = CMPLX(xsc * cabs(A[q + q * lda]), 0.0);
                for (p = 0; p < nr; p++) {
                    if ((p > q && cabs(A[p + q * lda]) <= temp1) ||
                        (p < q))
                        A[p + q * lda] = ctemp;
                }
            }
        } else {
            zlaset("U", nr - 1, nr - 1, CZERO, CZERO, &A[1 * lda], lda);
        }

        zgesvj("L", "N", "N", nr, nr, A, lda, SVA,
               n, V, ldv, cwork, lwork, rwork, lrwork, info);

        scalem = rwork[0];
        numrank = (INT)(rwork[1] + 0.5);

    } else if ((rsvec && !lsvec && !jracc) ||
               (jracc && !lsvec && nr != n)) {
        /* ============================================================== */
        /* Singular Values and Right Singular Vectors                     */
        /* ============================================================== */

        if (almort) {

            /* .. in this case NR equals N */
            for (p = 0; p < nr; p++) {
                cblas_zcopy(n - p, &A[p + p * lda], lda, &V[p + p * ldv], 1);
                zlacgv(n - p, &V[p + p * ldv], 1);
            }
            zlaset("U", nr - 1, nr - 1, CZERO, CZERO, &V[1 * ldv], ldv);

            zgesvj("L", "U", "N", n, nr, V, ldv, SVA, nr, A, lda,
                   cwork, lwork, rwork, lrwork, info);
            scalem = rwork[0];
            numrank = (INT)(rwork[1] + 0.5);

        } else {

            /* .. two more QR factorizations ( one QRF is not enough, two require */
            /*    accumulated product of Jacobi rotations, three are perfect ) */
            zlaset("L", nr - 1, nr - 1, CZERO, CZERO, &A[1], lda);
            zgelqf(nr, n, A, lda, cwork, &cwork[n], lwork - n, &ierr);
            zlacpy("L", nr, nr, A, lda, V, ldv);
            zlaset("U", nr - 1, nr - 1, CZERO, CZERO, &V[1 * ldv], ldv);
            zgeqrf(nr, nr, V, ldv, &cwork[n], &cwork[2 * n], lwork - 2 * n, &ierr);
            for (p = 0; p < nr; p++) {
                cblas_zcopy(nr - p, &V[p + p * ldv], ldv, &V[p + p * ldv], 1);
                zlacgv(nr - p, &V[p + p * ldv], 1);
            }
            zlaset("U", nr - 1, nr - 1, CZERO, CZERO, &V[1 * ldv], ldv);

            zgesvj("L", "U", "N", nr, nr, V, ldv, SVA, nr, U, ldu,
                   &cwork[n], lwork - n, rwork, lrwork, info);
            scalem = rwork[0];
            numrank = (INT)(rwork[1] + 0.5);
            if (nr < n) {
                zlaset("A", n - nr, nr, CZERO, CZERO, &V[nr], ldv);
                zlaset("A", nr, n - nr, CZERO, CZERO, &V[nr * ldv], ldv);
                zlaset("A", n - nr, n - nr, CZERO, CONE, &V[nr + nr * ldv], ldv);
            }

            zunmlq("L", "C", n, n, nr, A, lda, cwork,
                   V, ldv, &cwork[n], lwork - n, &ierr);

        }

        /* .. permute the rows of V */
        zlapmr(0, n, n, V, ldv, iwork);

        if (transp) {
            zlacpy("A", n, n, V, ldv, U, ldu);
        }

    } else if (jracc && !lsvec && nr == n) {

        zlaset("L", n - 1, n - 1, CZERO, CZERO, &A[1], lda);

        zgesvj("U", "N", "V", n, n, A, lda, SVA, n, V, ldv,
               cwork, lwork, rwork, lrwork, info);
        scalem = rwork[0];
        numrank = (INT)(rwork[1] + 0.5);
        zlapmr(0, n, n, V, ldv, iwork);

    } else if (lsvec && !rsvec) {
        /* ============================================================== */
        /* Singular Values and Left Singular Vectors                      */
        /* ============================================================== */

        /* .. second preconditioning step to avoid need to accumulate */
        /*    Jacobi rotations in the Jacobi iterations. */
        for (p = 0; p < nr; p++) {
            cblas_zcopy(n - p, &A[p + p * lda], lda, &U[p + p * ldu], 1);
            zlacgv(n - p, &U[p + p * ldu], 1);
        }
        zlaset("U", nr - 1, nr - 1, CZERO, CZERO, &U[1 * ldu], ldu);

        zgeqrf(n, nr, U, ldu, &cwork[n], &cwork[2 * n], lwork - 2 * n, &ierr);

        for (p = 0; p < nr - 1; p++) {
            cblas_zcopy(nr - p - 1, &U[p + (p + 1) * ldu], ldu, &U[(p + 1) + p * ldu], 1);
            zlacgv(n - p, &U[p + p * ldu], 1);
        }
        zlaset("U", nr - 1, nr - 1, CZERO, CZERO, &U[1 * ldu], ldu);

        zgesvj("L", "U", "N", nr, nr, U, ldu, SVA, nr, A,
               lda, &cwork[n], lwork - n, rwork, lrwork, info);
        scalem = rwork[0];
        numrank = (INT)(rwork[1] + 0.5);

        if (nr < m) {
            zlaset("A", m - nr, nr, CZERO, CZERO, &U[nr], ldu);
            if (nr < n1) {
                zlaset("A", nr, n1 - nr, CZERO, CZERO, &U[nr * ldu], ldu);
                zlaset("A", m - nr, n1 - nr, CZERO, CONE, &U[nr + nr * ldu], ldu);
            }
        }

        zunmqr("L", "N", m, n1, n, A, lda, cwork, U,
               ldu, &cwork[n], lwork - n, &ierr);

        if (rowpiv)
            zlaswp(n1, U, ldu, 0, m - 2, &iwork[iwoff], -1);

        for (p = 0; p < n1; p++) {
            xsc = ONE / cblas_dznrm2(m, &U[p * ldu], 1);
            cblas_zdscal(m, xsc, &U[p * ldu], 1);
        }

        if (transp) {
            zlacpy("A", n, n, U, ldu, V, ldv);
        }

    } else {
        /* ============================================================== */
        /* Full SVD                                                       */
        /* ============================================================== */

        if (!jracc) {

        if (!almort) {

            /* Second Preconditioning Step (QRF [with pivoting]) */
            for (p = 0; p < nr; p++) {
                cblas_zcopy(n - p, &A[p + p * lda], lda, &V[p + p * ldv], 1);
                zlacgv(n - p, &V[p + p * ldv], 1);
            }

            if (l2pert) {
                xsc = sqrt(small);
                for (q = 0; q < nr; q++) {
                    ctemp = CMPLX(xsc * cabs(V[q + q * ldv]), 0.0);
                    for (p = 0; p < n; p++) {
                        if ((p > q && cabs(V[p + q * ldv]) <= temp1) ||
                            (p < q))
                            V[p + q * ldv] = ctemp;
                        if (p < q) V[p + q * ldv] = -V[p + q * ldv];
                    }
                }
            } else {
                zlaset("U", nr - 1, nr - 1, CZERO, CZERO, &V[1 * ldv], ldv);
            }

            /* Estimate the row scaled condition number of R1 */
            zlacpy("L", nr, nr, V, ldv, &cwork[2 * n], nr);
            for (p = 0; p < nr; p++) {
                temp1 = cblas_dznrm2(nr - p, &cwork[2 * n + p * nr + p], 1);
                cblas_zdscal(nr - p, ONE / temp1, &cwork[2 * n + p * nr + p], 1);
            }
            zpocon("L", nr, &cwork[2 * n], nr, ONE, &temp1,
                   &cwork[2 * n + nr * nr], rwork, &ierr);
            condr1 = ONE / sqrt(temp1);

            cond_ok = sqrt(sqrt((f64)nr));

            if (condr1 < cond_ok) {
                /* .. the second QRF without pivoting. */
                zgeqrf(n, nr, V, ldv, &cwork[n], &cwork[2 * n], lwork - 2 * n, &ierr);

                if (l2pert) {
                    xsc = sqrt(small) / epsln;
                    for (p = 1; p < nr; p++) {
                        for (q = 0; q < p; q++) {
                            ctemp = CMPLX(xsc * fmin(cabs(V[p + p * ldv]), cabs(V[q + q * ldv])), 0.0);
                            if (cabs(V[q + p * ldv]) <= temp1)
                                V[q + p * ldv] = ctemp;
                        }
                    }
                }

                if (nr != n)
                    zlacpy("A", n, nr, V, ldv, &cwork[2 * n], n);

                /* .. this transposed copy should be better than naive */
                for (p = 0; p < nr - 1; p++) {
                    cblas_zcopy(nr - p - 1, &V[p + (p + 1) * ldv], ldv, &V[(p + 1) + p * ldv], 1);
                    zlacgv(nr - p, &V[p + p * ldv], 1);
                }
                V[(nr - 1) + (nr - 1) * ldv] = conj(V[(nr - 1) + (nr - 1) * ldv]);

                condr2 = condr1;

            } else {

                /* .. ill-conditioned case: second QRF with pivoting */
                for (p = 0; p < nr; p++) {
                    iwork[n + p] = 0;
                }
                zgeqp3(n, nr, V, ldv, &iwork[n], &cwork[n],
                       &cwork[2 * n], lwork - 2 * n, rwork, &ierr);

                if (l2pert) {
                    xsc = sqrt(small);
                    for (p = 1; p < nr; p++) {
                        for (q = 0; q < p; q++) {
                            ctemp = CMPLX(xsc * fmin(cabs(V[p + p * ldv]), cabs(V[q + q * ldv])), 0.0);
                            if (cabs(V[q + p * ldv]) <= temp1)
                                V[q + p * ldv] = ctemp;
                        }
                    }
                }

                zlacpy("A", n, nr, V, ldv, &cwork[2 * n], n);

                if (l2pert) {
                    xsc = sqrt(small);
                    for (p = 1; p < nr; p++) {
                        for (q = 0; q < p; q++) {
                            ctemp = CMPLX(xsc * fmin(cabs(V[p + p * ldv]), cabs(V[q + q * ldv])), 0.0);
                            V[p + q * ldv] = -ctemp;
                        }
                    }
                } else {
                    zlaset("L", nr - 1, nr - 1, CZERO, CZERO, &V[1], ldv);
                }

                /* Now, compute R2 = L3 * Q3, the LQ factorization. */
                zgelqf(nr, nr, V, ldv, &cwork[2 * n + n * nr],
                       &cwork[2 * n + n * nr + nr], lwork - 2 * n - n * nr - nr, &ierr);

                /* .. and estimate the condition number */
                zlacpy("L", nr, nr, V, ldv, &cwork[2 * n + n * nr + nr], nr);
                for (p = 0; p < nr; p++) {
                    temp1 = cblas_dznrm2(p + 1, &cwork[2 * n + n * nr + nr + p], nr);
                    cblas_zdscal(p + 1, ONE / temp1, &cwork[2 * n + n * nr + nr + p], nr);
                }
                zpocon("L", nr, &cwork[2 * n + n * nr + nr], nr, ONE, &temp1,
                       &cwork[2 * n + n * nr + nr + nr * nr], rwork, &ierr);
                condr2 = ONE / sqrt(temp1);

                if (condr2 >= cond_ok) {
                    /* .. save the Householder vectors used for Q3 */
                    zlacpy("U", nr, nr, V, ldv, &cwork[2 * n], n);
                }

            }

            if (l2pert) {
                xsc = sqrt(small);
                for (q = 1; q < nr; q++) {
                    ctemp = xsc * V[q + q * ldv];
                    for (p = 0; p < q; p++) {
                        V[p + q * ldv] = -ctemp;
                    }
                }
            } else {
                zlaset("U", nr - 1, nr - 1, CZERO, CZERO, &V[1 * ldv], ldv);
            }

            /* Second preconditioning finished; continue with Jacobi SVD */
            if (condr1 < cond_ok) {

                zgesvj("L", "U", "N", nr, nr, V, ldv, SVA, nr, U, ldu,
                       &cwork[2 * n + n * nr + nr], lwork - 2 * n - n * nr - nr,
                       rwork, lrwork, info);
                scalem = rwork[0];
                numrank = (INT)(rwork[1] + 0.5);
                for (p = 0; p < nr; p++) {
                    cblas_zcopy(nr, &V[p * ldv], 1, &U[p * ldu], 1);
                    cblas_zdscal(nr, SVA[p], &V[p * ldv], 1);
                }

                if (nr == n) {
                    cblas_ztrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
                                nr, nr, &CONE, A, lda, V, ldv);
                } else {
                    cblas_ztrsm(CblasColMajor, CblasLeft, CblasUpper, CblasConjTrans, CblasNonUnit,
                                nr, nr, &CONE, &cwork[2 * n], n, V, ldv);
                    if (nr < n) {
                        zlaset("A", n - nr, nr, CZERO, CZERO, &V[nr], ldv);
                        zlaset("A", nr, n - nr, CZERO, CZERO, &V[nr * ldv], ldv);
                        zlaset("A", n - nr, n - nr, CZERO, CONE, &V[nr + nr * ldv], ldv);
                    }
                    zunmqr("L", "N", n, n, nr, &cwork[2 * n], n, &cwork[n],
                           V, ldv, &cwork[2 * n + n * nr + nr],
                           lwork - 2 * n - n * nr - nr, &ierr);
                }

            } else if (condr2 < cond_ok) {

                zgesvj("L", "U", "N", nr, nr, V, ldv, SVA, nr, U,
                       ldu, &cwork[2 * n + n * nr + nr], lwork - 2 * n - n * nr - nr,
                       rwork, lrwork, info);
                scalem = rwork[0];
                numrank = (INT)(rwork[1] + 0.5);
                for (p = 0; p < nr; p++) {
                    cblas_zcopy(nr, &V[p * ldv], 1, &U[p * ldu], 1);
                    cblas_zdscal(nr, SVA[p], &U[p * ldu], 1);
                }
                cblas_ztrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
                            nr, nr, &CONE, &cwork[2 * n], n, U, ldu);
                /* .. apply the permutation from the second QR factorization */
                for (q = 0; q < nr; q++) {
                    for (p = 0; p < nr; p++) {
                        cwork[2 * n + n * nr + nr + iwork[n + p]] = U[p + q * ldu];
                    }
                    for (p = 0; p < nr; p++) {
                        U[p + q * ldu] = cwork[2 * n + n * nr + nr + p];
                    }
                }
                if (nr < n) {
                    zlaset("A", n - nr, nr, CZERO, CZERO, &V[nr], ldv);
                    zlaset("A", nr, n - nr, CZERO, CZERO, &V[nr * ldv], ldv);
                    zlaset("A", n - nr, n - nr, CZERO, CONE, &V[nr + nr * ldv], ldv);
                }
                zunmqr("L", "N", n, n, nr, &cwork[2 * n], n, &cwork[n],
                       V, ldv, &cwork[2 * n + n * nr + nr],
                       lwork - 2 * n - n * nr - nr, &ierr);

            } else {
                /* Last line of defense. */
                zgesvj("L", "U", "V", nr, nr, V, ldv, SVA, nr, U,
                       ldu, &cwork[2 * n + n * nr + nr], lwork - 2 * n - n * nr - nr,
                       rwork, lrwork, info);
                scalem = rwork[0];
                numrank = (INT)(rwork[1] + 0.5);
                if (nr < n) {
                    zlaset("A", n - nr, nr, CZERO, CZERO, &V[nr], ldv);
                    zlaset("A", nr, n - nr, CZERO, CZERO, &V[nr * ldv], ldv);
                    zlaset("A", n - nr, n - nr, CZERO, CONE, &V[nr + nr * ldv], ldv);
                }
                zunmqr("L", "N", n, n, nr, &cwork[2 * n], n, &cwork[n],
                       V, ldv, &cwork[2 * n + n * nr + nr],
                       lwork - 2 * n - n * nr - nr, &ierr);

                zunmlq("L", "C", nr, nr, nr, &cwork[2 * n], n,
                       &cwork[2 * n + n * nr], U, ldu,
                       &cwork[2 * n + n * nr + nr],
                       lwork - 2 * n - n * nr - nr, &ierr);
                /* .. apply the permutation from the second QR factorization */
                for (q = 0; q < nr; q++) {
                    for (p = 0; p < nr; p++) {
                        cwork[2 * n + n * nr + nr + iwork[n + p]] = U[p + q * ldu];
                    }
                    for (p = 0; p < nr; p++) {
                        U[p + q * ldu] = cwork[2 * n + n * nr + nr + p];
                    }
                }

            }

            /* Permute the rows of V using the (column) permutation from the */
            /* first QRF. Also, scale the columns to make them unit in */
            /* Euclidean norm. This applies to all cases. */
            temp1 = sqrt((f64)n) * epsln;
            for (q = 0; q < n; q++) {
                for (p = 0; p < n; p++) {
                    cwork[2 * n + n * nr + nr + iwork[p]] = V[p + q * ldv];
                }
                for (p = 0; p < n; p++) {
                    V[p + q * ldv] = cwork[2 * n + n * nr + nr + p];
                }
                xsc = ONE / cblas_dznrm2(n, &V[q * ldv], 1);
                if (xsc < (ONE - temp1) || xsc > (ONE + temp1))
                    cblas_zdscal(n, xsc, &V[q * ldv], 1);
            }

            /* .. assemble the left singular vector matrix U (M x N). */
            if (nr < m) {
                zlaset("A", m - nr, nr, CZERO, CZERO, &U[nr], ldu);
                if (nr < n1) {
                    zlaset("A", nr, n1 - nr, CZERO, CZERO, &U[nr * ldu], ldu);
                    zlaset("A", m - nr, n1 - nr, CZERO, CONE, &U[nr + nr * ldu], ldu);
                }
            }

            zunmqr("L", "N", m, n1, n, A, lda, cwork, U,
                   ldu, &cwork[n], lwork - n, &ierr);

            /* The columns of U are normalized. */
            temp1 = sqrt((f64)m) * epsln;
            for (p = 0; p < nr; p++) {
                xsc = ONE / cblas_dznrm2(m, &U[p * ldu], 1);
                if (xsc < (ONE - temp1) || xsc > (ONE + temp1))
                    cblas_zdscal(m, xsc, &U[p * ldu], 1);
            }

            if (rowpiv)
                zlaswp(n1, U, ldu, 0, m - 2, &iwork[iwoff], -1);

        } else {
            /* ---------------------------------------------------------- */
            /* D.1 ALMORT: the initial matrix A has almost orthogonal     */
            /* columns and the second QRF is not needed                   */
            /* ---------------------------------------------------------- */

            zlacpy("U", n, n, A, lda, &cwork[n], n);
            if (l2pert) {
                xsc = sqrt(small);
                for (p = 1; p < n; p++) {
                    ctemp = xsc * cwork[n + p * n + p];
                    for (q = 0; q < p; q++) {
                        cwork[n + q * n + p] = -ctemp;
                    }
                }
            } else {
                zlaset("L", n - 1, n - 1, CZERO, CZERO, &cwork[n + 1], n);
            }

            zgesvj("U", "U", "N", n, n, &cwork[n], n, SVA,
                   n, U, ldu, &cwork[n + n * n], lwork - n - n * n,
                   rwork, lrwork, info);

            scalem = rwork[0];
            numrank = (INT)(rwork[1] + 0.5);
            for (p = 0; p < n; p++) {
                cblas_zcopy(n, &cwork[n + p * n], 1, &U[p * ldu], 1);
                cblas_zdscal(n, SVA[p], &cwork[n + p * n], 1);
            }

            cblas_ztrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
                        n, n, &CONE, A, lda, &cwork[n], n);
            for (p = 0; p < n; p++) {
                cblas_zcopy(n, &cwork[n + p], n, &V[iwork[p]], ldv);
            }
            temp1 = sqrt((f64)n) * epsln;
            for (p = 0; p < n; p++) {
                xsc = ONE / cblas_dznrm2(n, &V[p * ldv], 1);
                if (xsc < (ONE - temp1) || xsc > (ONE + temp1))
                    cblas_zdscal(n, xsc, &V[p * ldv], 1);
            }

            /* Assemble the left singular vector matrix U (M x N). */
            if (n < m) {
                zlaset("A", m - n, n, CZERO, CZERO, &U[n], ldu);
                if (n < n1) {
                    zlaset("A", n, n1 - n, CZERO, CZERO, &U[n * ldu], ldu);
                    zlaset("A", m - n, n1 - n, CZERO, CONE, &U[n + n * ldu], ldu);
                }
            }
            zunmqr("L", "N", m, n1, n, A, lda, cwork, U,
                   ldu, &cwork[n], lwork - n, &ierr);
            temp1 = sqrt((f64)m) * epsln;
            for (p = 0; p < n1; p++) {
                xsc = ONE / cblas_dznrm2(m, &U[p * ldu], 1);
                if (xsc < (ONE - temp1) || xsc > (ONE + temp1))
                    cblas_zdscal(m, xsc, &U[p * ldu], 1);
            }

            if (rowpiv)
                zlaswp(n1, U, ldu, 0, m - 2, &iwork[iwoff], -1);

        }

        } else {
            /* ============================================================ */
            /* D.2 JRACC: preconditioned Jacobi SVD with explicitly         */
            /* accumulated rotations                                        */
            /* ============================================================ */

            for (p = 0; p < nr; p++) {
                cblas_zcopy(n - p, &A[p + p * lda], lda, &V[p + p * ldv], 1);
                zlacgv(n - p, &V[p + p * ldv], 1);
            }

            if (l2pert) {
                xsc = sqrt(small / epsln);
                for (q = 0; q < nr; q++) {
                    ctemp = CMPLX(xsc * cabs(V[q + q * ldv]), 0.0);
                    for (p = 0; p < n; p++) {
                        if ((p > q && cabs(V[p + q * ldv]) <= temp1) ||
                            (p < q))
                            V[p + q * ldv] = ctemp;
                        if (p < q) V[p + q * ldv] = -V[p + q * ldv];
                    }
                }
            } else {
                zlaset("U", nr - 1, nr - 1, CZERO, CZERO, &V[1 * ldv], ldv);
            }

            zgeqrf(n, nr, V, ldv, &cwork[n], &cwork[2 * n], lwork - 2 * n, &ierr);
            zlacpy("L", n, nr, V, ldv, &cwork[2 * n], n);

            for (p = 0; p < nr; p++) {
                cblas_zcopy(nr - p, &V[p + p * ldv], ldv, &U[p + p * ldu], 1);
                zlacgv(nr - p, &U[p + p * ldu], 1);
            }

            if (l2pert) {
                xsc = sqrt(small / epsln);
                for (q = 1; q < nr; q++) {
                    for (p = 0; p < q; p++) {
                        ctemp = CMPLX(xsc * fmin(cabs(U[p + p * ldu]), cabs(U[q + q * ldu])), 0.0);
                        U[p + q * ldu] = -ctemp;
                    }
                }
            } else {
                zlaset("U", nr - 1, nr - 1, CZERO, CZERO, &U[1 * ldu], ldu);
            }

            zgesvj("L", "U", "V", nr, nr, U, ldu, SVA,
                   n, V, ldv, &cwork[2 * n + n * nr], lwork - 2 * n - n * nr,
                   rwork, lrwork, info);
            scalem = rwork[0];
            numrank = (INT)(rwork[1] + 0.5);

            if (nr < n) {
                zlaset("A", n - nr, nr, CZERO, CZERO, &V[nr], ldv);
                zlaset("A", nr, n - nr, CZERO, CZERO, &V[nr * ldv], ldv);
                zlaset("A", n - nr, n - nr, CZERO, CONE, &V[nr + nr * ldv], ldv);
            }

            zunmqr("L", "N", n, n, nr, &cwork[2 * n], n, &cwork[n],
                   V, ldv, &cwork[2 * n + n * nr + nr],
                   lwork - 2 * n - n * nr - nr, &ierr);

            temp1 = sqrt((f64)n) * epsln;
            for (q = 0; q < n; q++) {
                for (p = 0; p < n; p++) {
                    cwork[2 * n + n * nr + nr + iwork[p]] = V[p + q * ldv];
                }
                for (p = 0; p < n; p++) {
                    V[p + q * ldv] = cwork[2 * n + n * nr + nr + p];
                }
                xsc = ONE / cblas_dznrm2(n, &V[q * ldv], 1);
                if (xsc < (ONE - temp1) || xsc > (ONE + temp1))
                    cblas_zdscal(n, xsc, &V[q * ldv], 1);
            }

            if (nr < m) {
                zlaset("A", m - nr, nr, CZERO, CZERO, &U[nr], ldu);
                if (nr < n1) {
                    zlaset("A", nr, n1 - nr, CZERO, CZERO, &U[nr * ldu], ldu);
                    zlaset("A", m - nr, n1 - nr, CZERO, CONE, &U[nr + nr * ldu], ldu);
                }
            }

            zunmqr("L", "N", m, n1, n, A, lda, cwork, U,
                   ldu, &cwork[n], lwork - n, &ierr);

            if (rowpiv)
                zlaswp(n1, U, ldu, 0, m - 2, &iwork[iwoff], -1);

        }

        /* .. swap U and V because the procedure worked on A^* */
        if (transp) {
            for (p = 0; p < n; p++) {
                cblas_zswap(n, &U[p * ldu], 1, &V[p * ldv], 1);
            }
        }

    }
    /* end of the full SVD */

    /* -------------------------------------------------------------------- */
    /* Undo scaling, if necessary (and possible)                            */
    /* -------------------------------------------------------------------- */

    if (uscal2 <= (big / SVA[0]) * uscal1) {
        dlascl("G", 0, 0, uscal1, uscal2, nr, 1, SVA, n, &ierr);
        uscal1 = ONE;
        uscal2 = ONE;
    }

    if (nr < n) {
        for (p = nr; p < n; p++) {
            SVA[p] = ZERO;
        }
    }

    rwork[0] = uscal2 * scalem;
    rwork[1] = uscal1;
    if (errest) rwork[2] = sconda;
    if (lsvec && rsvec) {
        rwork[3] = condr1;
        rwork[4] = condr2;
    }
    if (l2tran) {
        rwork[5] = entra;
        rwork[6] = entrat;
    }

    iwork[0] = nr;
    iwork[1] = numrank;
    iwork[2] = warning;
    if (transp) {
        iwork[3] = 1;
    } else {
        iwork[3] = -1;
    }

    return;
}
