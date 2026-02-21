/**
 * @file zgesvdq.c
 * @brief ZGESVDQ computes SVD with a QR-Preconditioned QR SVD Method.
 */

#include "semicolon_lapack_complex_double.h"
#include "lapack_tuning.h"
#include <complex.h>
#include <math.h>
#include <cblas.h>

static const f64 ZERO = 0.0;
static const f64 ONE = 1.0;
static const c128 CZERO = CMPLX(0.0, 0.0);
static const c128 CONE = CMPLX(1.0, 0.0);

/**
 * ZGESVDQ computes the singular value decomposition (SVD) of a complex
 * M-by-N matrix A, where M >= N. The SVD of A is written as
 *
 *              A = U * SIGMA * V^*
 *
 * where SIGMA is an N-by-N diagonal matrix, U is an M-by-N orthonormal
 * matrix, and V is an N-by-N unitary matrix.
 */
void zgesvdq(const char* joba, const char* jobp, const char* jobr,
             const char* jobu, const char* jobv,
             const int m, const int n, c128* restrict A, const int lda,
             f64* restrict S, c128* restrict U, const int ldu,
             c128* restrict V, const int ldv, int* numrank,
             int* restrict iwork, const int liwork,
             c128* restrict cwork, const int lcwork,
             f64* restrict rwork, const int lrwork, int* info)
{
    int wntus, wntur, wntua, wntuf, lsvc0, lsvec, dntwu;
    int wntvr, wntva, rsvec, dntwv;
    int accla, acclm, acclh, conda;
    int rowprm, rtrans, lquery;
    int ierr, nr, n1 = 0, optratio, p, q;
    int ascaled;

    int lwcon, lwqp3, lwrk_zgelqf, lwrk_zgesvd, lwrk_zgesvd2,
        lwrk_zgeqp3 = 0, lwrk_zgeqrf, lwrk_zunmlq, lwrk_zunmqr = 0,
        lwrk_zunmqr2, lwlqf, lwqrf, lwsvd, lwsvd2, lwunq,
        lwunq2, lwunlq, minwrk, minwrk2, optwrk, optwrk2,
        iminwrk, rminwrk;

    f64 big, epsln, rtmp, sconda = -ONE, sfmin;
    c128 ctmp;
    c128 cdummy[1];
    f64 rdummy[1];

    wntus  = (jobu[0] == 'S' || jobu[0] == 's' || jobu[0] == 'U' || jobu[0] == 'u');
    wntur  = (jobu[0] == 'R' || jobu[0] == 'r');
    wntua  = (jobu[0] == 'A' || jobu[0] == 'a');
    wntuf  = (jobu[0] == 'F' || jobu[0] == 'f');
    lsvc0  = wntus || wntur || wntua;
    lsvec  = lsvc0 || wntuf;
    dntwu  = (jobu[0] == 'N' || jobu[0] == 'n');

    wntvr  = (jobv[0] == 'R' || jobv[0] == 'r');
    wntva  = (jobv[0] == 'A' || jobv[0] == 'a' || jobv[0] == 'V' || jobv[0] == 'v');
    rsvec  = wntvr || wntva;
    dntwv  = (jobv[0] == 'N' || jobv[0] == 'n');

    accla  = (joba[0] == 'A' || joba[0] == 'a');
    acclm  = (joba[0] == 'M' || joba[0] == 'm');
    conda  = (joba[0] == 'E' || joba[0] == 'e');
    acclh  = (joba[0] == 'H' || joba[0] == 'h') || conda;

    rowprm = (jobp[0] == 'P' || jobp[0] == 'p');
    rtrans = (jobr[0] == 'T' || jobr[0] == 't');

    if (rowprm) {
        iminwrk = (1 > n + m - 1) ? 1 : n + m - 1;
        rminwrk = 2;
        if (rminwrk < m) rminwrk = m;
        if (rminwrk < 5 * n) rminwrk = 5 * n;
    } else {
        iminwrk = (1 > n) ? 1 : n;
        rminwrk = (2 > 5 * n) ? 2 : 5 * n;
    }
    lquery = (liwork == -1 || lcwork == -1 || lrwork == -1);
    *info = 0;
    if (!(accla || acclm || acclh)) {
        *info = -1;
    } else if (!(rowprm || (jobp[0] == 'N' || jobp[0] == 'n'))) {
        *info = -2;
    } else if (!(rtrans || (jobr[0] == 'N' || jobr[0] == 'n'))) {
        *info = -3;
    } else if (!(lsvec || dntwu)) {
        *info = -4;
    } else if (wntur && wntva) {
        *info = -5;
    } else if (!(rsvec || dntwv)) {
        *info = -5;
    } else if (m < 0) {
        *info = -6;
    } else if (n < 0 || n > m) {
        *info = -7;
    } else if (lda < (1 > m ? 1 : m)) {
        *info = -9;
    } else if (ldu < 1 || (lsvc0 && ldu < m) || (wntuf && ldu < n)) {
        *info = -12;
    } else if (ldv < 1 || (rsvec && ldv < n) || (conda && ldv < n)) {
        *info = -14;
    } else if (liwork < iminwrk && !lquery) {
        *info = -17;
    }

    if (*info == 0) {

        lwqp3 = n + 1;
        if (wntus || wntur) {
            lwunq = (n > 1) ? n : 1;
        } else if (wntua) {
            lwunq = (m > 1) ? m : 1;
        } else {
            lwunq = 0;
        }
        lwcon = 2 * n;
        lwsvd = (3 * n > 1) ? 3 * n : 1;
        if (lquery) {
            zgeqp3(m, n, A, lda, iwork, NULL, cdummy, -1, rdummy, &ierr);
            lwrk_zgeqp3 = (int)creal(cdummy[0]);
            if (wntus || wntur) {
                zunmqr("L", "N", m, n, n, A, lda, NULL, U,
                        ldu, cdummy, -1, &ierr);
                lwrk_zunmqr = (int)creal(cdummy[0]);
            } else if (wntua) {
                zunmqr("L", "N", m, m, n, A, lda, NULL, U,
                        ldu, cdummy, -1, &ierr);
                lwrk_zunmqr = (int)creal(cdummy[0]);
            } else {
                lwrk_zunmqr = 0;
            }
        }
        optwrk = 2;
        if (!lsvec && !rsvec) {
            if (conda) {
                minwrk = n + lwqp3;
                if (minwrk < lwcon) minwrk = lwcon;
                if (minwrk < lwsvd) minwrk = lwsvd;
            } else {
                minwrk = n + lwqp3;
                if (minwrk < lwsvd) minwrk = lwsvd;
            }
            if (lquery) {
                zgesvd("N", "N", n, n, A, lda, S, U, ldu,
                        V, ldv, cdummy, -1, rdummy, &ierr);
                lwrk_zgesvd = (int)creal(cdummy[0]);
                if (conda) {
                    optwrk = n + lwrk_zgeqp3;
                    if (optwrk < n + lwcon) optwrk = n + lwcon;
                    if (optwrk < lwrk_zgesvd) optwrk = lwrk_zgesvd;
                } else {
                    optwrk = n + lwrk_zgeqp3;
                    if (optwrk < lwrk_zgesvd) optwrk = lwrk_zgesvd;
                }
            }
        } else if (lsvec && !rsvec) {
            if (conda) {
                minwrk = n + lwqp3;
                if (minwrk < n + lwcon) minwrk = n + lwcon;
                if (minwrk < n + lwsvd) minwrk = n + lwsvd;
                if (minwrk < n + lwunq) minwrk = n + lwunq;
            } else {
                minwrk = n + lwqp3;
                if (minwrk < n + lwsvd) minwrk = n + lwsvd;
                if (minwrk < n + lwunq) minwrk = n + lwunq;
            }
            if (lquery) {
                if (rtrans) {
                    zgesvd("N", "O", n, n, A, lda, S, U, ldu,
                            V, ldv, cdummy, -1, rdummy, &ierr);
                } else {
                    zgesvd("O", "N", n, n, A, lda, S, U, ldu,
                            V, ldv, cdummy, -1, rdummy, &ierr);
                }
                lwrk_zgesvd = (int)creal(cdummy[0]);
                if (conda) {
                    optwrk = n + lwrk_zgeqp3;
                    if (optwrk < n + lwcon) optwrk = n + lwcon;
                    if (optwrk < n + lwrk_zgesvd) optwrk = n + lwrk_zgesvd;
                    if (optwrk < n + lwrk_zunmqr) optwrk = n + lwrk_zunmqr;
                } else {
                    optwrk = n + lwrk_zgeqp3;
                    if (optwrk < n + lwrk_zgesvd) optwrk = n + lwrk_zgesvd;
                    if (optwrk < n + lwrk_zunmqr) optwrk = n + lwrk_zunmqr;
                }
            }
        } else if (rsvec && !lsvec) {
            if (conda) {
                minwrk = n + lwqp3;
                if (minwrk < n + lwcon) minwrk = n + lwcon;
                if (minwrk < n + lwsvd) minwrk = n + lwsvd;
            } else {
                minwrk = n + lwqp3;
                if (minwrk < n + lwsvd) minwrk = n + lwsvd;
            }
            if (lquery) {
                if (rtrans) {
                    zgesvd("O", "N", n, n, A, lda, S, U, ldu,
                            V, ldv, cdummy, -1, rdummy, &ierr);
                } else {
                    zgesvd("N", "O", n, n, A, lda, S, U, ldu,
                            V, ldv, cdummy, -1, rdummy, &ierr);
                }
                lwrk_zgesvd = (int)creal(cdummy[0]);
                if (conda) {
                    optwrk = n + lwrk_zgeqp3;
                    if (optwrk < n + lwcon) optwrk = n + lwcon;
                    if (optwrk < n + lwrk_zgesvd) optwrk = n + lwrk_zgesvd;
                } else {
                    optwrk = n + lwrk_zgeqp3;
                    if (optwrk < n + lwrk_zgesvd) optwrk = n + lwrk_zgesvd;
                }
            }
        } else {
            if (rtrans) {
                minwrk = lwqp3;
                if (minwrk < lwsvd) minwrk = lwsvd;
                if (minwrk < lwunq) minwrk = lwunq;
                if (conda && minwrk < lwcon) minwrk = lwcon;
                minwrk = minwrk + n;
                if (wntva) {
                    lwqrf  = (n / 2 > 1) ? n / 2 : 1;
                    lwsvd2 = (3 * (n / 2) > 1) ? 3 * (n / 2) : 1;
                    lwunq2 = (n > 1) ? n : 1;
                    minwrk2 = lwqp3;
                    if (minwrk2 < n / 2 + lwqrf) minwrk2 = n / 2 + lwqrf;
                    if (minwrk2 < n / 2 + lwsvd2) minwrk2 = n / 2 + lwsvd2;
                    if (minwrk2 < n / 2 + lwunq2) minwrk2 = n / 2 + lwunq2;
                    if (minwrk2 < lwunq) minwrk2 = lwunq;
                    if (conda && minwrk2 < lwcon) minwrk2 = lwcon;
                    minwrk2 = n + minwrk2;
                    if (minwrk < minwrk2) minwrk = minwrk2;
                }
            } else {
                minwrk = lwqp3;
                if (minwrk < lwsvd) minwrk = lwsvd;
                if (minwrk < lwunq) minwrk = lwunq;
                if (conda && minwrk < lwcon) minwrk = lwcon;
                minwrk = minwrk + n;
                if (wntva) {
                    lwlqf  = (n / 2 > 1) ? n / 2 : 1;
                    lwsvd2 = (3 * (n / 2) > 1) ? 3 * (n / 2) : 1;
                    lwunlq = (n > 1) ? n : 1;
                    minwrk2 = lwqp3;
                    if (minwrk2 < n / 2 + lwlqf) minwrk2 = n / 2 + lwlqf;
                    if (minwrk2 < n / 2 + lwsvd2) minwrk2 = n / 2 + lwsvd2;
                    if (minwrk2 < n / 2 + lwunlq) minwrk2 = n / 2 + lwunlq;
                    if (minwrk2 < lwunq) minwrk2 = lwunq;
                    if (conda && minwrk2 < lwcon) minwrk2 = lwcon;
                    minwrk2 = n + minwrk2;
                    if (minwrk < minwrk2) minwrk = minwrk2;
                }
            }
            if (lquery) {
                if (rtrans) {
                    zgesvd("O", "A", n, n, A, lda, S, U, ldu,
                            V, ldv, cdummy, -1, rdummy, &ierr);
                    lwrk_zgesvd = (int)creal(cdummy[0]);
                    optwrk = lwrk_zgeqp3;
                    if (optwrk < lwrk_zgesvd) optwrk = lwrk_zgesvd;
                    if (optwrk < lwrk_zunmqr) optwrk = lwrk_zunmqr;
                    if (conda && optwrk < lwcon) optwrk = lwcon;
                    optwrk = n + optwrk;
                    if (wntva) {
                        zgeqrf(n, n / 2, U, ldu, NULL, cdummy, -1, &ierr);
                        lwrk_zgeqrf = (int)creal(cdummy[0]);
                        zgesvd("S", "O", n / 2, n / 2, V, ldv, S, U,
                                ldu, NULL, 1, cdummy, -1, rdummy, &ierr);
                        lwrk_zgesvd2 = (int)creal(cdummy[0]);
                        zunmqr("R", "C", n, n, n / 2, U, ldu, NULL,
                                V, ldv, cdummy, -1, &ierr);
                        lwrk_zunmqr2 = (int)creal(cdummy[0]);
                        optwrk2 = lwrk_zgeqp3;
                        if (optwrk2 < n / 2 + lwrk_zgeqrf) optwrk2 = n / 2 + lwrk_zgeqrf;
                        if (optwrk2 < n / 2 + lwrk_zgesvd2) optwrk2 = n / 2 + lwrk_zgesvd2;
                        if (optwrk2 < n / 2 + lwrk_zunmqr2) optwrk2 = n / 2 + lwrk_zunmqr2;
                        if (conda && optwrk2 < lwcon) optwrk2 = lwcon;
                        optwrk2 = n + optwrk2;
                        if (optwrk < optwrk2) optwrk = optwrk2;
                    }
                } else {
                    zgesvd("S", "O", n, n, A, lda, S, U, ldu,
                            V, ldv, cdummy, -1, rdummy, &ierr);
                    lwrk_zgesvd = (int)creal(cdummy[0]);
                    optwrk = lwrk_zgeqp3;
                    if (optwrk < lwrk_zgesvd) optwrk = lwrk_zgesvd;
                    if (optwrk < lwrk_zunmqr) optwrk = lwrk_zunmqr;
                    if (conda && optwrk < lwcon) optwrk = lwcon;
                    optwrk = n + optwrk;
                    if (wntva) {
                        zgelqf(n / 2, n, U, ldu, NULL, cdummy, -1, &ierr);
                        lwrk_zgelqf = (int)creal(cdummy[0]);
                        zgesvd("S", "O", n / 2, n / 2, V, ldv, S, U,
                                ldu, NULL, 1, cdummy, -1, rdummy, &ierr);
                        lwrk_zgesvd2 = (int)creal(cdummy[0]);
                        zunmlq("R", "N", n, n, n / 2, U, ldu, NULL,
                                V, ldv, cdummy, -1, &ierr);
                        lwrk_zunmlq = (int)creal(cdummy[0]);
                        optwrk2 = lwrk_zgeqp3;
                        if (optwrk2 < n / 2 + lwrk_zgelqf) optwrk2 = n / 2 + lwrk_zgelqf;
                        if (optwrk2 < n / 2 + lwrk_zgesvd2) optwrk2 = n / 2 + lwrk_zgesvd2;
                        if (optwrk2 < n / 2 + lwrk_zunmlq) optwrk2 = n / 2 + lwrk_zunmlq;
                        if (conda && optwrk2 < lwcon) optwrk2 = lwcon;
                        optwrk2 = n + optwrk2;
                        if (optwrk < optwrk2) optwrk = optwrk2;
                    }
                }
            }
        }

        if (minwrk < 2) minwrk = 2;
        if (optwrk < 2) optwrk = 2;
        if (lcwork < minwrk && !lquery) *info = -19;

    }

    if (*info == 0 && lrwork < rminwrk && !lquery) {
        *info = -21;
    }
    if (*info != 0) {
        xerbla("ZGESVDQ", -(*info));
        return;
    } else if (lquery) {
        iwork[0] = iminwrk;
        cwork[0] = CMPLX((f64)optwrk, 0.0);
        cwork[1] = CMPLX((f64)minwrk, 0.0);
        rwork[0] = (f64)rminwrk;
        return;
    }

    if (m == 0 || n == 0) {
        return;
    }

    big = dlamch("O");
    ascaled = 0;
    if (rowprm) {
        for (p = 0; p < m; p++) {
            rwork[p] = zlange("M", 1, n, &A[p], lda, rdummy);
            if ((rwork[p] != rwork[p]) ||
                ((rwork[p] * ZERO) != ZERO)) {
                *info = -8;
                xerbla("ZGESVDQ", -(*info));
                return;
            }
        }
        for (p = 0; p < m - 1; p++) {
            q = cblas_idamax(m - p, &rwork[p], 1) + p;
            iwork[n + p] = q;
            if (p != q) {
                rtmp     = rwork[p];
                rwork[p] = rwork[q];
                rwork[q] = rtmp;
            }
        }

        if (rwork[0] == ZERO) {
            *numrank = 0;
            dlaset("G", n, 1, ZERO, ZERO, S, n);
            if (wntus) zlaset("G", m, n, CZERO, CONE, U, ldu);
            if (wntua) zlaset("G", m, m, CZERO, CONE, U, ldu);
            if (wntva) zlaset("G", n, n, CZERO, CONE, V, ldv);
            if (wntuf) {
                zlaset("G", n, 1, CZERO, CZERO, cwork, n);
                zlaset("G", m, n, CZERO, CONE, U, ldu);
            }
            for (p = 0; p < n; p++) {
                iwork[p] = p + 1;
            }
            if (rowprm) {
                for (p = n; p < n + m - 1; p++) {
                    iwork[p] = p - n + 1;
                }
            }
            if (conda) rwork[0] = -ONE;
            rwork[1] = -ONE;
            return;
        }

        if (rwork[0] > big / sqrt((f64)m)) {
            zlascl("G", 0, 0, sqrt((f64)m), ONE, m, n, A, lda, &ierr);
            ascaled = 1;
        }
        zlaswp(n, A, lda, 0, m - 2, &iwork[n], 1);
    }

    if (!rowprm) {
        rtmp = zlange("M", m, n, A, lda, rwork);
        if ((rtmp != rtmp) ||
            ((rtmp * ZERO) != ZERO)) {
            *info = -8;
            xerbla("ZGESVDQ", -(*info));
            return;
        }
        if (rtmp > big / sqrt((f64)m)) {
            zlascl("G", 0, 0, sqrt((f64)m), ONE, m, n, A, lda, &ierr);
            ascaled = 1;
        }
    }

    for (p = 0; p < n; p++) {
        iwork[p] = 0;
    }
    zgeqp3(m, n, A, lda, iwork, cwork, &cwork[n], lcwork - n,
            rwork, &ierr);

    epsln = dlamch("E");
    sfmin = dlamch("S");

    if (accla) {
        nr = 1;
        rtmp = sqrt((f64)n) * epsln;
        for (p = 1; p < n; p++) {
            if (cabs1(A[p + p * lda]) < (rtmp * cabs1(A[0]))) break;
            nr++;
        }
    } else if (acclm) {
        nr = 1;
        for (p = 1; p < n; p++) {
            if ((cabs1(A[p + p * lda]) < (epsln * cabs1(A[(p-1) + (p-1) * lda]))) ||
                (cabs1(A[p + p * lda]) < sfmin)) break;
            nr++;
        }
    } else {
        nr = 1;
        for (p = 1; p < n; p++) {
            if (cabs1(A[p + p * lda]) == ZERO) break;
            nr++;
        }

        if (conda) {
            zlacpy("U", n, n, A, lda, V, ldv);
            for (p = 0; p < nr; p++) {
                rtmp = cblas_dznrm2(p + 1, &V[p * ldv], 1);
                cblas_zdscal(p + 1, ONE / rtmp, &V[p * ldv], 1);
            }
            if (!lsvec && !rsvec) {
                zpocon("U", nr, V, ldv, ONE, &rtmp,
                        cwork, rwork, &ierr);
            } else {
                zpocon("U", nr, V, ldv, ONE, &rtmp,
                        &cwork[n], rwork, &ierr);
            }
            sconda = ONE / sqrt(rtmp);
        }
    }

    if (wntur) {
        n1 = nr;
    } else if (wntus || wntuf) {
        n1 = n;
    } else if (wntua) {
        n1 = m;
    }

    if (!rsvec && !lsvec) {
        /*
         * .. only the singular values are requested
         */
        if (rtrans) {

            int minmn = (n < nr) ? n : nr;
            for (p = 0; p < minmn; p++) {
                A[p + p * lda] = conj(A[p + p * lda]);
                for (q = p + 1; q < n; q++) {
                    A[q + p * lda] = conj(A[p + q * lda]);
                    if (q < nr) A[p + q * lda] = CZERO;
                }
            }

            zgesvd("N", "N", n, nr, A, lda, S, U, ldu,
                    V, ldv, cwork, lcwork, rwork, info);

        } else {

            if (nr > 1)
                zlaset("L", nr - 1, nr - 1, CZERO, CZERO, &A[1], lda);
            zgesvd("N", "N", nr, n, A, lda, S, U, ldu,
                    V, ldv, cwork, lcwork, rwork, info);

        }

    } else if (lsvec && !rsvec) {
        /*
         * .. the singular values and the left singular vectors requested
         */
        if (rtrans) {
            for (p = 0; p < nr; p++) {
                for (q = p; q < n; q++) {
                    U[q + p * ldu] = conj(A[p + q * lda]);
                }
            }
            if (nr > 1)
                zlaset("U", nr - 1, nr - 1, CZERO, CZERO, &U[ldu], ldu);
            zgesvd("N", "O", n, nr, U, ldu, S, NULL, 1,
                    NULL, 1, &cwork[n], lcwork - n, rwork, info);

            for (p = 0; p < nr; p++) {
                U[p + p * ldu] = conj(U[p + p * ldu]);
                for (q = p + 1; q < nr; q++) {
                    ctmp          = conj(U[q + p * ldu]);
                    U[q + p * ldu] = conj(U[p + q * ldu]);
                    U[p + q * ldu] = ctmp;
                }
            }

        } else {
            zlacpy("U", nr, n, A, lda, U, ldu);
            if (nr > 1)
                zlaset("L", nr - 1, nr - 1, CZERO, CZERO, &U[1], ldu);
            zgesvd("O", "N", nr, n, U, ldu, S, NULL, 1,
                    NULL, 1, &cwork[n], lcwork - n, rwork, info);
        }

        if (nr < m && !wntuf) {
            zlaset("A", m - nr, nr, CZERO, CZERO, &U[nr], ldu);
            if (nr < n1) {
                zlaset("A", nr, n1 - nr, CZERO, CZERO, &U[nr * ldu], ldu);
                zlaset("A", m - nr, n1 - nr, CZERO, CONE,
                        &U[nr + nr * ldu], ldu);
            }
        }

        if (!wntuf)
            zunmqr("L", "N", m, n1, n, A, lda, cwork, U,
                    ldu, &cwork[n], lcwork - n, &ierr);
        if (rowprm && !wntuf)
            zlaswp(n1, U, ldu, 0, m - 2, &iwork[n], -1);

    } else if (rsvec && !lsvec) {
        /*
         * .. the singular values and the right singular vectors requested
         */
        if (rtrans) {
            for (p = 0; p < nr; p++) {
                for (q = p; q < n; q++) {
                    V[q + p * ldv] = conj(A[p + q * lda]);
                }
            }
            if (nr > 1)
                zlaset("U", nr - 1, nr - 1, CZERO, CZERO, &V[ldv], ldv);

            if (wntvr || nr == n) {
                zgesvd("O", "N", n, nr, V, ldv, S, U, ldu,
                        NULL, 1, &cwork[n], lcwork - n, rwork, info);

                for (p = 0; p < nr; p++) {
                    V[p + p * ldv] = conj(V[p + p * ldv]);
                    for (q = p + 1; q < nr; q++) {
                        ctmp          = conj(V[q + p * ldv]);
                        V[q + p * ldv] = conj(V[p + q * ldv]);
                        V[p + q * ldv] = ctmp;
                    }
                }

                if (nr < n) {
                    for (p = 0; p < nr; p++) {
                        for (q = nr; q < n; q++) {
                            V[p + q * ldv] = conj(V[q + p * ldv]);
                        }
                    }
                }
                zlapmt(0, nr, n, V, ldv, iwork);
            } else {
                zlaset("G", n, n - nr, CZERO, CZERO, &V[nr * ldv], ldv);
                zgesvd("O", "N", n, n, V, ldv, S, U, ldu,
                        NULL, 1, &cwork[n], lcwork - n, rwork, info);

                for (p = 0; p < n; p++) {
                    V[p + p * ldv] = conj(V[p + p * ldv]);
                    for (q = p + 1; q < n; q++) {
                        ctmp          = conj(V[q + p * ldv]);
                        V[q + p * ldv] = conj(V[p + q * ldv]);
                        V[p + q * ldv] = ctmp;
                    }
                }
                zlapmt(0, n, n, V, ldv, iwork);
            }

        } else {
            zlacpy("U", nr, n, A, lda, V, ldv);
            if (nr > 1)
                zlaset("L", nr - 1, nr - 1, CZERO, CZERO, &V[1], ldv);

            if (wntvr || nr == n) {
                zgesvd("N", "O", nr, n, V, ldv, S, NULL, 1,
                        NULL, 1, &cwork[n], lcwork - n, rwork, info);
                zlapmt(0, nr, n, V, ldv, iwork);
            } else {
                zlaset("G", n - nr, n, CZERO, CZERO, &V[nr], ldv);
                zgesvd("N", "O", n, n, V, ldv, S, NULL, 1,
                        NULL, 1, &cwork[n], lcwork - n, rwork, info);
                zlapmt(0, n, n, V, ldv, iwork);
            }
        }

    } else {
        /*
         * .. FULL SVD requested
         */
        if (rtrans) {

            if (wntvr || nr == n) {
                for (p = 0; p < nr; p++) {
                    for (q = p; q < n; q++) {
                        V[q + p * ldv] = conj(A[p + q * lda]);
                    }
                }
                if (nr > 1)
                    zlaset("U", nr - 1, nr - 1, CZERO, CZERO, &V[ldv], ldv);

                zgesvd("O", "A", n, nr, V, ldv, S, NULL, 1,
                        U, ldu, &cwork[n], lcwork - n, rwork, info);

                for (p = 0; p < nr; p++) {
                    V[p + p * ldv] = conj(V[p + p * ldv]);
                    for (q = p + 1; q < nr; q++) {
                        ctmp          = conj(V[q + p * ldv]);
                        V[q + p * ldv] = conj(V[p + q * ldv]);
                        V[p + q * ldv] = ctmp;
                    }
                }
                if (nr < n) {
                    for (p = 0; p < nr; p++) {
                        for (q = nr; q < n; q++) {
                            V[p + q * ldv] = conj(V[q + p * ldv]);
                        }
                    }
                }
                zlapmt(0, nr, n, V, ldv, iwork);

                for (p = 0; p < nr; p++) {
                    U[p + p * ldu] = conj(U[p + p * ldu]);
                    for (q = p + 1; q < nr; q++) {
                        ctmp          = conj(U[q + p * ldu]);
                        U[q + p * ldu] = conj(U[p + q * ldu]);
                        U[p + q * ldu] = ctmp;
                    }
                }

                if (nr < m && !wntuf) {
                    zlaset("A", m - nr, nr, CZERO, CZERO, &U[nr], ldu);
                    if (nr < n1) {
                        zlaset("A", nr, n1 - nr, CZERO, CZERO, &U[nr * ldu], ldu);
                        zlaset("A", m - nr, n1 - nr, CZERO, CONE,
                                &U[nr + nr * ldu], ldu);
                    }
                }

            } else {
                optratio = 2;
                if (optratio * nr > n) {
                    for (p = 0; p < nr; p++) {
                        for (q = p; q < n; q++) {
                            V[q + p * ldv] = conj(A[p + q * lda]);
                        }
                    }
                    if (nr > 1)
                        zlaset("U", nr - 1, nr - 1, CZERO, CZERO, &V[ldv], ldv);

                    zlaset("A", n, n - nr, CZERO, CZERO, &V[nr * ldv], ldv);
                    zgesvd("O", "A", n, n, V, ldv, S, NULL, 1,
                            U, ldu, &cwork[n], lcwork - n, rwork, info);

                    for (p = 0; p < n; p++) {
                        V[p + p * ldv] = conj(V[p + p * ldv]);
                        for (q = p + 1; q < n; q++) {
                            ctmp          = conj(V[q + p * ldv]);
                            V[q + p * ldv] = conj(V[p + q * ldv]);
                            V[p + q * ldv] = ctmp;
                        }
                    }
                    zlapmt(0, n, n, V, ldv, iwork);

                    for (p = 0; p < n; p++) {
                        U[p + p * ldu] = conj(U[p + p * ldu]);
                        for (q = p + 1; q < n; q++) {
                            ctmp          = conj(U[q + p * ldu]);
                            U[q + p * ldu] = conj(U[p + q * ldu]);
                            U[p + q * ldu] = ctmp;
                        }
                    }

                    if (n < m && !wntuf) {
                        zlaset("A", m - n, n, CZERO, CZERO, &U[n], ldu);
                        if (n < n1) {
                            zlaset("A", n, n1 - n, CZERO, CZERO, &U[n * ldu], ldu);
                            zlaset("A", m - n, n1 - n, CZERO, CONE,
                                    &U[n + n * ldu], ldu);
                        }
                    }
                } else {
                    for (p = 0; p < nr; p++) {
                        for (q = p; q < n; q++) {
                            U[q + (nr + p) * ldu] = conj(A[p + q * lda]);
                        }
                    }
                    if (nr > 1)
                        zlaset("U", nr - 1, nr - 1, CZERO, CZERO,
                                &U[(nr + 1) * ldu], ldu);
                    zgeqrf(n, nr, &U[nr * ldu], ldu, &cwork[n],
                            &cwork[n + nr], lcwork - n - nr, &ierr);
                    for (p = 0; p < nr; p++) {
                        for (q = 0; q < n; q++) {
                            V[q + p * ldv] = conj(U[p + (nr + q) * ldu]);
                        }
                    }
                    zlaset("U", nr - 1, nr - 1, CZERO, CZERO, &V[ldv], ldv);
                    zgesvd("S", "O", nr, nr, V, ldv, S, U, ldu,
                            NULL, 1, &cwork[n + nr], lcwork - n - nr, rwork, info);
                    zlaset("A", n - nr, nr, CZERO, CZERO, &V[nr], ldv);
                    zlaset("A", nr, n - nr, CZERO, CZERO, &V[nr * ldv], ldv);
                    zlaset("A", n - nr, n - nr, CZERO, CONE,
                            &V[nr + nr * ldv], ldv);
                    zunmqr("R", "C", n, n, nr, &U[nr * ldu], ldu,
                            &cwork[n], V, ldv, &cwork[n + nr], lcwork - n - nr, &ierr);
                    zlapmt(0, n, n, V, ldv, iwork);

                    if (nr < m && !wntuf) {
                        zlaset("A", m - nr, nr, CZERO, CZERO, &U[nr], ldu);
                        if (nr < n1) {
                            zlaset("A", nr, n1 - nr, CZERO, CZERO,
                                    &U[nr * ldu], ldu);
                            zlaset("A", m - nr, n1 - nr, CZERO, CONE,
                                    &U[nr + nr * ldu], ldu);
                        }
                    }
                }
            }

        } else {

            if (wntvr || nr == n) {
                zlacpy("U", nr, n, A, lda, V, ldv);
                if (nr > 1)
                    zlaset("L", nr - 1, nr - 1, CZERO, CZERO, &V[1], ldv);

                zgesvd("S", "O", nr, n, V, ldv, S, U, ldu,
                        NULL, 1, &cwork[n], lcwork - n, rwork, info);
                zlapmt(0, nr, n, V, ldv, iwork);

                if (nr < m && !wntuf) {
                    zlaset("A", m - nr, nr, CZERO, CZERO, &U[nr], ldu);
                    if (nr < n1) {
                        zlaset("A", nr, n1 - nr, CZERO, CZERO,
                                &U[nr * ldu], ldu);
                        zlaset("A", m - nr, n1 - nr, CZERO, CONE,
                                &U[nr + nr * ldu], ldu);
                    }
                }

            } else {
                optratio = 2;
                if (optratio * nr > n) {
                    zlacpy("U", nr, n, A, lda, V, ldv);
                    if (nr > 1)
                        zlaset("L", nr - 1, nr - 1, CZERO, CZERO, &V[1], ldv);

                    zlaset("A", n - nr, n, CZERO, CZERO, &V[nr], ldv);
                    zgesvd("S", "O", n, n, V, ldv, S, U, ldu,
                            NULL, 1, &cwork[n], lcwork - n, rwork, info);
                    zlapmt(0, n, n, V, ldv, iwork);

                    if (n < m && !wntuf) {
                        zlaset("A", m - n, n, CZERO, CZERO, &U[n], ldu);
                        if (n < n1) {
                            zlaset("A", n, n1 - n, CZERO, CZERO,
                                    &U[n * ldu], ldu);
                            zlaset("A", m - n, n1 - n, CZERO, CONE,
                                    &U[n + n * ldu], ldu);
                        }
                    }
                } else {
                    zlacpy("U", nr, n, A, lda, &U[nr], ldu);
                    if (nr > 1)
                        zlaset("L", nr - 1, nr - 1, CZERO, CZERO,
                                &U[nr + 1], ldu);
                    zgelqf(nr, n, &U[nr], ldu, &cwork[n],
                            &cwork[n + nr], lcwork - n - nr, &ierr);
                    zlacpy("L", nr, nr, &U[nr], ldu, V, ldv);
                    if (nr > 1)
                        zlaset("U", nr - 1, nr - 1, CZERO, CZERO, &V[ldv], ldv);
                    zgesvd("S", "O", nr, nr, V, ldv, S, U, ldu,
                            NULL, 1, &cwork[n + nr], lcwork - n - nr, rwork, info);
                    zlaset("A", n - nr, nr, CZERO, CZERO, &V[nr], ldv);
                    zlaset("A", nr, n - nr, CZERO, CZERO, &V[nr * ldv], ldv);
                    zlaset("A", n - nr, n - nr, CZERO, CONE,
                            &V[nr + nr * ldv], ldv);
                    zunmlq("R", "N", n, n, nr, &U[nr], ldu,
                            &cwork[n], V, ldv, &cwork[n + nr], lcwork - n - nr, &ierr);
                    zlapmt(0, n, n, V, ldv, iwork);

                    if (nr < m && !wntuf) {
                        zlaset("A", m - nr, nr, CZERO, CZERO, &U[nr], ldu);
                        if (nr < n1) {
                            zlaset("A", nr, n1 - nr, CZERO, CZERO,
                                    &U[nr * ldu], ldu);
                            zlaset("A", m - nr, n1 - nr, CZERO, CONE,
                                    &U[nr + nr * ldu], ldu);
                        }
                    }
                }
            }
        }

        if (!wntuf)
            zunmqr("L", "N", m, n1, n, A, lda, cwork, U,
                    ldu, &cwork[n], lcwork - n, &ierr);
        if (rowprm && !wntuf)
            zlaswp(n1, U, ldu, 0, m - 2, &iwork[n], -1);

    }

    p = nr;
    for (q = nr - 1; q >= 0; q--) {
        if (S[q] > ZERO) break;
        nr--;
    }

    if (nr < n)
        dlaset("G", n - nr, 1, ZERO, ZERO, &S[nr], n);
    if (ascaled)
        dlascl("G", 0, 0, ONE, sqrt((f64)m), nr, 1, S, n, &ierr);
    if (conda) rwork[0] = sconda;
    rwork[1] = (f64)(p - nr);

    *numrank = nr;
}
