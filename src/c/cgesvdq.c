/**
 * @file cgesvdq.c
 * @brief CGESVDQ computes SVD with a QR-Preconditioned QR SVD Method.
 */

#include "semicolon_lapack_complex_single.h"
#include "lapack_tuning.h"
#include <complex.h>
#include <math.h>
#include <cblas.h>

static const f32 ZERO = 0.0f;
static const f32 ONE = 1.0f;
static const c64 CZERO = CMPLXF(0.0f, 0.0f);
static const c64 CONE = CMPLXF(1.0f, 0.0f);

/**
 * CGESVDQ computes the singular value decomposition (SVD) of a complex
 * M-by-N matrix A, where M >= N. The SVD of A is written as
 *
 *              A = U * SIGMA * V^*
 *
 * where SIGMA is an N-by-N diagonal matrix, U is an M-by-N orthonormal
 * matrix, and V is an N-by-N unitary matrix.
 */
void cgesvdq(const char* joba, const char* jobp, const char* jobr,
             const char* jobu, const char* jobv,
             const int m, const int n, c64* restrict A, const int lda,
             f32* restrict S, c64* restrict U, const int ldu,
             c64* restrict V, const int ldv, int* numrank,
             int* restrict iwork, const int liwork,
             c64* restrict cwork, const int lcwork,
             f32* restrict rwork, const int lrwork, int* info)
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

    f32 big, epsln, rtmp, sconda = -ONE, sfmin;
    c64 ctmp;
    c64 cdummy[1];
    f32 rdummy[1];

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
            cgeqp3(m, n, A, lda, iwork, NULL, cdummy, -1, rdummy, &ierr);
            lwrk_zgeqp3 = (int)crealf(cdummy[0]);
            if (wntus || wntur) {
                cunmqr("L", "N", m, n, n, A, lda, NULL, U,
                        ldu, cdummy, -1, &ierr);
                lwrk_zunmqr = (int)crealf(cdummy[0]);
            } else if (wntua) {
                cunmqr("L", "N", m, m, n, A, lda, NULL, U,
                        ldu, cdummy, -1, &ierr);
                lwrk_zunmqr = (int)crealf(cdummy[0]);
            } else {
                lwrk_zunmqr = 0;
            }
        }
        minwrk = 2;
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
                cgesvd("N", "N", n, n, A, lda, S, U, ldu,
                        V, ldv, cdummy, -1, rdummy, &ierr);
                lwrk_zgesvd = (int)crealf(cdummy[0]);
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
                    cgesvd("N", "O", n, n, A, lda, S, U, ldu,
                            V, ldv, cdummy, -1, rdummy, &ierr);
                } else {
                    cgesvd("O", "N", n, n, A, lda, S, U, ldu,
                            V, ldv, cdummy, -1, rdummy, &ierr);
                }
                lwrk_zgesvd = (int)crealf(cdummy[0]);
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
                    cgesvd("O", "N", n, n, A, lda, S, U, ldu,
                            V, ldv, cdummy, -1, rdummy, &ierr);
                } else {
                    cgesvd("N", "O", n, n, A, lda, S, U, ldu,
                            V, ldv, cdummy, -1, rdummy, &ierr);
                }
                lwrk_zgesvd = (int)crealf(cdummy[0]);
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
                    cgesvd("O", "A", n, n, A, lda, S, U, ldu,
                            V, ldv, cdummy, -1, rdummy, &ierr);
                    lwrk_zgesvd = (int)crealf(cdummy[0]);
                    optwrk = lwrk_zgeqp3;
                    if (optwrk < lwrk_zgesvd) optwrk = lwrk_zgesvd;
                    if (optwrk < lwrk_zunmqr) optwrk = lwrk_zunmqr;
                    if (conda && optwrk < lwcon) optwrk = lwcon;
                    optwrk = n + optwrk;
                    if (wntva) {
                        cgeqrf(n, n / 2, U, ldu, NULL, cdummy, -1, &ierr);
                        lwrk_zgeqrf = (int)crealf(cdummy[0]);
                        cgesvd("S", "O", n / 2, n / 2, V, ldv, S, U,
                                ldu, NULL, 1, cdummy, -1, rdummy, &ierr);
                        lwrk_zgesvd2 = (int)crealf(cdummy[0]);
                        cunmqr("R", "C", n, n, n / 2, U, ldu, NULL,
                                V, ldv, cdummy, -1, &ierr);
                        lwrk_zunmqr2 = (int)crealf(cdummy[0]);
                        optwrk2 = lwrk_zgeqp3;
                        if (optwrk2 < n / 2 + lwrk_zgeqrf) optwrk2 = n / 2 + lwrk_zgeqrf;
                        if (optwrk2 < n / 2 + lwrk_zgesvd2) optwrk2 = n / 2 + lwrk_zgesvd2;
                        if (optwrk2 < n / 2 + lwrk_zunmqr2) optwrk2 = n / 2 + lwrk_zunmqr2;
                        if (conda && optwrk2 < lwcon) optwrk2 = lwcon;
                        optwrk2 = n + optwrk2;
                        if (optwrk < optwrk2) optwrk = optwrk2;
                    }
                } else {
                    cgesvd("S", "O", n, n, A, lda, S, U, ldu,
                            V, ldv, cdummy, -1, rdummy, &ierr);
                    lwrk_zgesvd = (int)crealf(cdummy[0]);
                    optwrk = lwrk_zgeqp3;
                    if (optwrk < lwrk_zgesvd) optwrk = lwrk_zgesvd;
                    if (optwrk < lwrk_zunmqr) optwrk = lwrk_zunmqr;
                    if (conda && optwrk < lwcon) optwrk = lwcon;
                    optwrk = n + optwrk;
                    if (wntva) {
                        cgelqf(n / 2, n, U, ldu, NULL, cdummy, -1, &ierr);
                        lwrk_zgelqf = (int)crealf(cdummy[0]);
                        cgesvd("S", "O", n / 2, n / 2, V, ldv, S, U,
                                ldu, NULL, 1, cdummy, -1, rdummy, &ierr);
                        lwrk_zgesvd2 = (int)crealf(cdummy[0]);
                        cunmlq("R", "N", n, n, n / 2, U, ldu, NULL,
                                V, ldv, cdummy, -1, &ierr);
                        lwrk_zunmlq = (int)crealf(cdummy[0]);
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
        xerbla("CGESVDQ", -(*info));
        return;
    } else if (lquery) {
        iwork[0] = iminwrk;
        cwork[0] = CMPLXF((f32)optwrk, 0.0f);
        cwork[1] = CMPLXF((f32)minwrk, 0.0f);
        rwork[0] = (f32)rminwrk;
        return;
    }

    if (m == 0 || n == 0) {
        return;
    }

    big = slamch("O");
    ascaled = 0;
    if (rowprm) {
        for (p = 0; p < m; p++) {
            rwork[p] = clange("M", 1, n, &A[p], lda, rdummy);
            if ((rwork[p] != rwork[p]) ||
                ((rwork[p] * ZERO) != ZERO)) {
                *info = -8;
                xerbla("CGESVDQ", -(*info));
                return;
            }
        }
        for (p = 0; p < m - 1; p++) {
            q = cblas_isamax(m - p, &rwork[p], 1) + p;
            iwork[n + p] = q;
            if (p != q) {
                rtmp     = rwork[p];
                rwork[p] = rwork[q];
                rwork[q] = rtmp;
            }
        }

        if (rwork[0] == ZERO) {
            *numrank = 0;
            slaset("G", n, 1, ZERO, ZERO, S, n);
            if (wntus) claset("G", m, n, CZERO, CONE, U, ldu);
            if (wntua) claset("G", m, m, CZERO, CONE, U, ldu);
            if (wntva) claset("G", n, n, CZERO, CONE, V, ldv);
            if (wntuf) {
                claset("G", n, 1, CZERO, CZERO, cwork, n);
                claset("G", m, n, CZERO, CONE, U, ldu);
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

        if (rwork[0] > big / sqrtf((f32)m)) {
            clascl("G", 0, 0, sqrtf((f32)m), ONE, m, n, A, lda, &ierr);
            ascaled = 1;
        }
        claswp(n, A, lda, 0, m - 2, &iwork[n], 1);
    }

    if (!rowprm) {
        rtmp = clange("M", m, n, A, lda, rwork);
        if ((rtmp != rtmp) ||
            ((rtmp * ZERO) != ZERO)) {
            *info = -8;
            xerbla("CGESVDQ", -(*info));
            return;
        }
        if (rtmp > big / sqrtf((f32)m)) {
            clascl("G", 0, 0, sqrtf((f32)m), ONE, m, n, A, lda, &ierr);
            ascaled = 1;
        }
    }

    for (p = 0; p < n; p++) {
        iwork[p] = 0;
    }
    cgeqp3(m, n, A, lda, iwork, cwork, &cwork[n], lcwork - n,
            rwork, &ierr);

    epsln = slamch("E");
    sfmin = slamch("S");
    nr = n;

    if (accla) {
        nr = 1;
        rtmp = sqrtf((f32)n) * epsln;
        for (p = 1; p < n; p++) {
            if (cabs1f(A[p + p * lda]) < (rtmp * cabs1f(A[0]))) break;
            nr++;
        }
    } else if (acclm) {
        nr = 1;
        for (p = 1; p < n; p++) {
            if ((cabs1f(A[p + p * lda]) < (epsln * cabs1f(A[(p-1) + (p-1) * lda]))) ||
                (cabs1f(A[p + p * lda]) < sfmin)) break;
            nr++;
        }
    } else {
        nr = 1;
        for (p = 1; p < n; p++) {
            if (cabs1f(A[p + p * lda]) == ZERO) break;
            nr++;
        }

        if (conda) {
            clacpy("U", n, n, A, lda, V, ldv);
            for (p = 0; p < nr; p++) {
                rtmp = cblas_scnrm2(p + 1, &V[p * ldv], 1);
                cblas_csscal(p + 1, ONE / rtmp, &V[p * ldv], 1);
            }
            if (!lsvec && !rsvec) {
                cpocon("U", nr, V, ldv, ONE, &rtmp,
                        cwork, rwork, &ierr);
            } else {
                cpocon("U", nr, V, ldv, ONE, &rtmp,
                        &cwork[n], rwork, &ierr);
            }
            sconda = ONE / sqrtf(rtmp);
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
                A[p + p * lda] = conjf(A[p + p * lda]);
                for (q = p + 1; q < n; q++) {
                    A[q + p * lda] = conjf(A[p + q * lda]);
                    if (q < nr) A[p + q * lda] = CZERO;
                }
            }

            cgesvd("N", "N", n, nr, A, lda, S, U, ldu,
                    V, ldv, cwork, lcwork, rwork, info);

        } else {

            if (nr > 1)
                claset("L", nr - 1, nr - 1, CZERO, CZERO, &A[1], lda);
            cgesvd("N", "N", nr, n, A, lda, S, U, ldu,
                    V, ldv, cwork, lcwork, rwork, info);

        }

    } else if (lsvec && !rsvec) {
        /*
         * .. the singular values and the left singular vectors requested
         */
        if (rtrans) {
            for (p = 0; p < nr; p++) {
                for (q = p; q < n; q++) {
                    U[q + p * ldu] = conjf(A[p + q * lda]);
                }
            }
            if (nr > 1)
                claset("U", nr - 1, nr - 1, CZERO, CZERO, &U[ldu], ldu);
            cgesvd("N", "O", n, nr, U, ldu, S, NULL, 1,
                    NULL, 1, &cwork[n], lcwork - n, rwork, info);

            for (p = 0; p < nr; p++) {
                U[p + p * ldu] = conjf(U[p + p * ldu]);
                for (q = p + 1; q < nr; q++) {
                    ctmp          = conjf(U[q + p * ldu]);
                    U[q + p * ldu] = conjf(U[p + q * ldu]);
                    U[p + q * ldu] = ctmp;
                }
            }

        } else {
            clacpy("U", nr, n, A, lda, U, ldu);
            if (nr > 1)
                claset("L", nr - 1, nr - 1, CZERO, CZERO, &U[1], ldu);
            cgesvd("O", "N", nr, n, U, ldu, S, NULL, 1,
                    NULL, 1, &cwork[n], lcwork - n, rwork, info);
        }

        if (nr < m && !wntuf) {
            claset("A", m - nr, nr, CZERO, CZERO, &U[nr], ldu);
            if (nr < n1) {
                claset("A", nr, n1 - nr, CZERO, CZERO, &U[nr * ldu], ldu);
                claset("A", m - nr, n1 - nr, CZERO, CONE,
                        &U[nr + nr * ldu], ldu);
            }
        }

        if (!wntuf)
            cunmqr("L", "N", m, n1, n, A, lda, cwork, U,
                    ldu, &cwork[n], lcwork - n, &ierr);
        if (rowprm && !wntuf)
            claswp(n1, U, ldu, 0, m - 2, &iwork[n], -1);

    } else if (rsvec && !lsvec) {
        /*
         * .. the singular values and the right singular vectors requested
         */
        if (rtrans) {
            for (p = 0; p < nr; p++) {
                for (q = p; q < n; q++) {
                    V[q + p * ldv] = conjf(A[p + q * lda]);
                }
            }
            if (nr > 1)
                claset("U", nr - 1, nr - 1, CZERO, CZERO, &V[ldv], ldv);

            if (wntvr || nr == n) {
                cgesvd("O", "N", n, nr, V, ldv, S, U, ldu,
                        NULL, 1, &cwork[n], lcwork - n, rwork, info);

                for (p = 0; p < nr; p++) {
                    V[p + p * ldv] = conjf(V[p + p * ldv]);
                    for (q = p + 1; q < nr; q++) {
                        ctmp          = conjf(V[q + p * ldv]);
                        V[q + p * ldv] = conjf(V[p + q * ldv]);
                        V[p + q * ldv] = ctmp;
                    }
                }

                if (nr < n) {
                    for (p = 0; p < nr; p++) {
                        for (q = nr; q < n; q++) {
                            V[p + q * ldv] = conjf(V[q + p * ldv]);
                        }
                    }
                }
                clapmt(0, nr, n, V, ldv, iwork);
            } else {
                claset("G", n, n - nr, CZERO, CZERO, &V[nr * ldv], ldv);
                cgesvd("O", "N", n, n, V, ldv, S, U, ldu,
                        NULL, 1, &cwork[n], lcwork - n, rwork, info);

                for (p = 0; p < n; p++) {
                    V[p + p * ldv] = conjf(V[p + p * ldv]);
                    for (q = p + 1; q < n; q++) {
                        ctmp          = conjf(V[q + p * ldv]);
                        V[q + p * ldv] = conjf(V[p + q * ldv]);
                        V[p + q * ldv] = ctmp;
                    }
                }
                clapmt(0, n, n, V, ldv, iwork);
            }

        } else {
            clacpy("U", nr, n, A, lda, V, ldv);
            if (nr > 1)
                claset("L", nr - 1, nr - 1, CZERO, CZERO, &V[1], ldv);

            if (wntvr || nr == n) {
                cgesvd("N", "O", nr, n, V, ldv, S, NULL, 1,
                        NULL, 1, &cwork[n], lcwork - n, rwork, info);
                clapmt(0, nr, n, V, ldv, iwork);
            } else {
                claset("G", n - nr, n, CZERO, CZERO, &V[nr], ldv);
                cgesvd("N", "O", n, n, V, ldv, S, NULL, 1,
                        NULL, 1, &cwork[n], lcwork - n, rwork, info);
                clapmt(0, n, n, V, ldv, iwork);
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
                        V[q + p * ldv] = conjf(A[p + q * lda]);
                    }
                }
                if (nr > 1)
                    claset("U", nr - 1, nr - 1, CZERO, CZERO, &V[ldv], ldv);

                cgesvd("O", "A", n, nr, V, ldv, S, NULL, 1,
                        U, ldu, &cwork[n], lcwork - n, rwork, info);

                for (p = 0; p < nr; p++) {
                    V[p + p * ldv] = conjf(V[p + p * ldv]);
                    for (q = p + 1; q < nr; q++) {
                        ctmp          = conjf(V[q + p * ldv]);
                        V[q + p * ldv] = conjf(V[p + q * ldv]);
                        V[p + q * ldv] = ctmp;
                    }
                }
                if (nr < n) {
                    for (p = 0; p < nr; p++) {
                        for (q = nr; q < n; q++) {
                            V[p + q * ldv] = conjf(V[q + p * ldv]);
                        }
                    }
                }
                clapmt(0, nr, n, V, ldv, iwork);

                for (p = 0; p < nr; p++) {
                    U[p + p * ldu] = conjf(U[p + p * ldu]);
                    for (q = p + 1; q < nr; q++) {
                        ctmp          = conjf(U[q + p * ldu]);
                        U[q + p * ldu] = conjf(U[p + q * ldu]);
                        U[p + q * ldu] = ctmp;
                    }
                }

                if (nr < m && !wntuf) {
                    claset("A", m - nr, nr, CZERO, CZERO, &U[nr], ldu);
                    if (nr < n1) {
                        claset("A", nr, n1 - nr, CZERO, CZERO, &U[nr * ldu], ldu);
                        claset("A", m - nr, n1 - nr, CZERO, CONE,
                                &U[nr + nr * ldu], ldu);
                    }
                }

            } else {
                optratio = 2;
                if (optratio * nr > n) {
                    for (p = 0; p < nr; p++) {
                        for (q = p; q < n; q++) {
                            V[q + p * ldv] = conjf(A[p + q * lda]);
                        }
                    }
                    if (nr > 1)
                        claset("U", nr - 1, nr - 1, CZERO, CZERO, &V[ldv], ldv);

                    claset("A", n, n - nr, CZERO, CZERO, &V[nr * ldv], ldv);
                    cgesvd("O", "A", n, n, V, ldv, S, NULL, 1,
                            U, ldu, &cwork[n], lcwork - n, rwork, info);

                    for (p = 0; p < n; p++) {
                        V[p + p * ldv] = conjf(V[p + p * ldv]);
                        for (q = p + 1; q < n; q++) {
                            ctmp          = conjf(V[q + p * ldv]);
                            V[q + p * ldv] = conjf(V[p + q * ldv]);
                            V[p + q * ldv] = ctmp;
                        }
                    }
                    clapmt(0, n, n, V, ldv, iwork);

                    for (p = 0; p < n; p++) {
                        U[p + p * ldu] = conjf(U[p + p * ldu]);
                        for (q = p + 1; q < n; q++) {
                            ctmp          = conjf(U[q + p * ldu]);
                            U[q + p * ldu] = conjf(U[p + q * ldu]);
                            U[p + q * ldu] = ctmp;
                        }
                    }

                    if (n < m && !wntuf) {
                        claset("A", m - n, n, CZERO, CZERO, &U[n], ldu);
                        if (n < n1) {
                            claset("A", n, n1 - n, CZERO, CZERO, &U[n * ldu], ldu);
                            claset("A", m - n, n1 - n, CZERO, CONE,
                                    &U[n + n * ldu], ldu);
                        }
                    }
                } else {
                    for (p = 0; p < nr; p++) {
                        for (q = p; q < n; q++) {
                            U[q + (nr + p) * ldu] = conjf(A[p + q * lda]);
                        }
                    }
                    if (nr > 1)
                        claset("U", nr - 1, nr - 1, CZERO, CZERO,
                                &U[(nr + 1) * ldu], ldu);
                    cgeqrf(n, nr, &U[nr * ldu], ldu, &cwork[n],
                            &cwork[n + nr], lcwork - n - nr, &ierr);
                    for (p = 0; p < nr; p++) {
                        for (q = 0; q < n; q++) {
                            V[q + p * ldv] = conjf(U[p + (nr + q) * ldu]);
                        }
                    }
                    claset("U", nr - 1, nr - 1, CZERO, CZERO, &V[ldv], ldv);
                    cgesvd("S", "O", nr, nr, V, ldv, S, U, ldu,
                            NULL, 1, &cwork[n + nr], lcwork - n - nr, rwork, info);
                    claset("A", n - nr, nr, CZERO, CZERO, &V[nr], ldv);
                    claset("A", nr, n - nr, CZERO, CZERO, &V[nr * ldv], ldv);
                    claset("A", n - nr, n - nr, CZERO, CONE,
                            &V[nr + nr * ldv], ldv);
                    cunmqr("R", "C", n, n, nr, &U[nr * ldu], ldu,
                            &cwork[n], V, ldv, &cwork[n + nr], lcwork - n - nr, &ierr);
                    clapmt(0, n, n, V, ldv, iwork);

                    if (nr < m && !wntuf) {
                        claset("A", m - nr, nr, CZERO, CZERO, &U[nr], ldu);
                        if (nr < n1) {
                            claset("A", nr, n1 - nr, CZERO, CZERO,
                                    &U[nr * ldu], ldu);
                            claset("A", m - nr, n1 - nr, CZERO, CONE,
                                    &U[nr + nr * ldu], ldu);
                        }
                    }
                }
            }

        } else {

            if (wntvr || nr == n) {
                clacpy("U", nr, n, A, lda, V, ldv);
                if (nr > 1)
                    claset("L", nr - 1, nr - 1, CZERO, CZERO, &V[1], ldv);

                cgesvd("S", "O", nr, n, V, ldv, S, U, ldu,
                        NULL, 1, &cwork[n], lcwork - n, rwork, info);
                clapmt(0, nr, n, V, ldv, iwork);

                if (nr < m && !wntuf) {
                    claset("A", m - nr, nr, CZERO, CZERO, &U[nr], ldu);
                    if (nr < n1) {
                        claset("A", nr, n1 - nr, CZERO, CZERO,
                                &U[nr * ldu], ldu);
                        claset("A", m - nr, n1 - nr, CZERO, CONE,
                                &U[nr + nr * ldu], ldu);
                    }
                }

            } else {
                optratio = 2;
                if (optratio * nr > n) {
                    clacpy("U", nr, n, A, lda, V, ldv);
                    if (nr > 1)
                        claset("L", nr - 1, nr - 1, CZERO, CZERO, &V[1], ldv);

                    claset("A", n - nr, n, CZERO, CZERO, &V[nr], ldv);
                    cgesvd("S", "O", n, n, V, ldv, S, U, ldu,
                            NULL, 1, &cwork[n], lcwork - n, rwork, info);
                    clapmt(0, n, n, V, ldv, iwork);

                    if (n < m && !wntuf) {
                        claset("A", m - n, n, CZERO, CZERO, &U[n], ldu);
                        if (n < n1) {
                            claset("A", n, n1 - n, CZERO, CZERO,
                                    &U[n * ldu], ldu);
                            claset("A", m - n, n1 - n, CZERO, CONE,
                                    &U[n + n * ldu], ldu);
                        }
                    }
                } else {
                    clacpy("U", nr, n, A, lda, &U[nr], ldu);
                    if (nr > 1)
                        claset("L", nr - 1, nr - 1, CZERO, CZERO,
                                &U[nr + 1], ldu);
                    cgelqf(nr, n, &U[nr], ldu, &cwork[n],
                            &cwork[n + nr], lcwork - n - nr, &ierr);
                    clacpy("L", nr, nr, &U[nr], ldu, V, ldv);
                    if (nr > 1)
                        claset("U", nr - 1, nr - 1, CZERO, CZERO, &V[ldv], ldv);
                    cgesvd("S", "O", nr, nr, V, ldv, S, U, ldu,
                            NULL, 1, &cwork[n + nr], lcwork - n - nr, rwork, info);
                    claset("A", n - nr, nr, CZERO, CZERO, &V[nr], ldv);
                    claset("A", nr, n - nr, CZERO, CZERO, &V[nr * ldv], ldv);
                    claset("A", n - nr, n - nr, CZERO, CONE,
                            &V[nr + nr * ldv], ldv);
                    cunmlq("R", "N", n, n, nr, &U[nr], ldu,
                            &cwork[n], V, ldv, &cwork[n + nr], lcwork - n - nr, &ierr);
                    clapmt(0, n, n, V, ldv, iwork);

                    if (nr < m && !wntuf) {
                        claset("A", m - nr, nr, CZERO, CZERO, &U[nr], ldu);
                        if (nr < n1) {
                            claset("A", nr, n1 - nr, CZERO, CZERO,
                                    &U[nr * ldu], ldu);
                            claset("A", m - nr, n1 - nr, CZERO, CONE,
                                    &U[nr + nr * ldu], ldu);
                        }
                    }
                }
            }
        }

        if (!wntuf)
            cunmqr("L", "N", m, n1, n, A, lda, cwork, U,
                    ldu, &cwork[n], lcwork - n, &ierr);
        if (rowprm && !wntuf)
            claswp(n1, U, ldu, 0, m - 2, &iwork[n], -1);

    }

    p = nr;
    for (q = nr - 1; q >= 0; q--) {
        if (S[q] > ZERO) break;
        nr--;
    }

    if (nr < n)
        slaset("G", n - nr, 1, ZERO, ZERO, &S[nr], n);
    if (ascaled)
        slascl("G", 0, 0, ONE, sqrtf((f32)m), nr, 1, S, n, &ierr);
    if (conda) rwork[0] = sconda;
    rwork[1] = (f32)(p - nr);

    *numrank = nr;
}
