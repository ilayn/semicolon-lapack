/**
 * @file dgesvdq.c
 * @brief DGESVDQ computes SVD with a QR-Preconditioned QR SVD Method.
 */

#include "semicolon_lapack_double.h"
#include "lapack_tuning.h"
#include <math.h>
#include <cblas.h>

static const f64 ZERO = 0.0;
static const f64 ONE = 1.0;

/**
 * DGESVDQ computes the singular value decomposition (SVD) of a real
 * M-by-N matrix A, where M >= N. The SVD of A is written as
 *
 *              A = U * SIGMA * V**T
 *
 * where SIGMA is an N-by-N diagonal matrix, U is an M-by-N orthonormal
 * matrix, and V is an N-by-N orthogonal matrix.
 */
void dgesvdq(const char* joba, const char* jobp, const char* jobr,
             const char* jobu, const char* jobv,
             const int m, const int n, f64* restrict A, const int lda,
             f64* restrict S, f64* restrict U, const int ldu,
             f64* restrict V, const int ldv, int* numrank,
             int* restrict iwork, const int liwork,
             f64* restrict work, const int lwork,
             f64* restrict rwork, const int lrwork, int* info)
{
    int wntus, wntur, wntua, wntuf, lsvc0, lsvec, dntwu;
    int wntvr, wntva, rsvec, dntwv;
    int accla, acclm, acclh, conda;
    int rowprm, rtrans, lquery;
    int lwqp3, lwcon, lwsvd, lworq, minwrk, optwrk;
    int iminwrk, rminwrk;
    int ierr, iwoff, nr, n1, optratio, p, q;
    int ascaled;
    f64 big, epsln, sconda, sfmin, rtmp;

    /* Decode job parameters */
    wntus = (jobu[0] == 'S' || jobu[0] == 's' || jobu[0] == 'U' || jobu[0] == 'u');
    wntur = (jobu[0] == 'R' || jobu[0] == 'r');
    wntua = (jobu[0] == 'A' || jobu[0] == 'a');
    wntuf = (jobu[0] == 'F' || jobu[0] == 'f');
    lsvc0 = wntus || wntur || wntua;
    lsvec = lsvc0 || wntuf;
    dntwu = (jobu[0] == 'N' || jobu[0] == 'n');

    wntvr = (jobv[0] == 'R' || jobv[0] == 'r');
    wntva = (jobv[0] == 'A' || jobv[0] == 'a' || jobv[0] == 'V' || jobv[0] == 'v');
    rsvec = wntvr || wntva;
    dntwv = (jobv[0] == 'N' || jobv[0] == 'n');

    accla = (joba[0] == 'A' || joba[0] == 'a');
    acclm = (joba[0] == 'M' || joba[0] == 'm');
    conda = (joba[0] == 'E' || joba[0] == 'e');
    acclh = (joba[0] == 'H' || joba[0] == 'h') || conda;

    rowprm = (jobp[0] == 'P' || jobp[0] == 'p');
    rtrans = (jobr[0] == 'T' || jobr[0] == 't');

    /* Workspace requirements */
    if (rowprm) {
        iminwrk = conda ? (n + m - 1 + n) : (n + m - 1);
        iminwrk = (iminwrk > 1) ? iminwrk : 1;
        rminwrk = (m > 2) ? m : 2;
    } else {
        iminwrk = conda ? (n + n) : n;
        iminwrk = (iminwrk > 1) ? iminwrk : 1;
        rminwrk = 2;
    }

    lquery = (liwork == -1 || lwork == -1 || lrwork == -1);

    /* Workspace sizes */
    lwqp3 = 3 * n + 1;
    if (wntus || wntur) {
        lworq = (n > 1) ? n : 1;
    } else if (wntua) {
        lworq = (m > 1) ? m : 1;
    } else {
        lworq = 1;
    }
    lwcon = 3 * n;
    lwsvd = (5 * n > 1) ? 5 * n : 1;

    if (!lsvec && !rsvec) {
        if (conda) {
            minwrk = n + lwqp3;
            if (minwrk < lwcon) minwrk = lwcon;
            if (minwrk < lwsvd) minwrk = lwsvd;
        } else {
            minwrk = n + lwqp3;
            if (minwrk < lwsvd) minwrk = lwsvd;
        }
    } else if (lsvec && !rsvec) {
        minwrk = n + lwqp3;
        if (minwrk < lwsvd) minwrk = lwsvd;
        if (minwrk < lworq) minwrk = lworq;
        if (conda && minwrk < lwcon) minwrk = lwcon;
    } else if (rsvec && !lsvec) {
        minwrk = n + lwqp3;
        if (minwrk < lwsvd) minwrk = lwsvd;
        if (conda && minwrk < lwcon) minwrk = lwcon;
    } else {
        minwrk = n + lwqp3;
        if (minwrk < lwsvd) minwrk = lwsvd;
        if (minwrk < lworq) minwrk = lworq;
        if (conda && minwrk < lwcon) minwrk = lwcon;
    }
    if (minwrk < 2) minwrk = 2;
    optwrk = minwrk;

    /* Check input arguments */
    *info = 0;
    if (!accla && !acclm && !acclh) {
        *info = -1;
    } else if (!rowprm && !(jobp[0] == 'N' || jobp[0] == 'n')) {
        *info = -2;
    } else if (!rtrans && !(jobr[0] == 'N' || jobr[0] == 'n')) {
        *info = -3;
    } else if (!lsvec && !dntwu) {
        *info = -4;
    } else if (wntur && wntva) {
        *info = -5;
    } else if (!rsvec && !dntwv) {
        *info = -5;
    } else if (m < 0) {
        *info = -6;
    } else if (n < 0 || n > m) {
        *info = -7;
    } else if (lda < (1 > m ? 1 : m)) {
        *info = -9;
    } else if (ldu < 1 || (lsvc0 && ldu < m) || (wntuf && ldu < n)) {
        *info = -12;
    } else if (ldv < 1 || ((rsvec || conda) && ldv < n)) {
        *info = -14;
    } else if (liwork < iminwrk && !lquery) {
        *info = -17;
    }

    if (*info == 0 && !lquery) {
        if (lwork < minwrk) {
            *info = -19;
        }
        if (lrwork < rminwrk) {
            *info = -21;
        }
    }

    if (*info != 0) {
        xerbla("DGESVDQ", -(*info));
        return;
    }

    if (lquery) {
        iwork[0] = iminwrk;
        work[0] = (f64)optwrk;
        work[1] = (f64)minwrk;
        rwork[0] = (f64)rminwrk;
        return;
    }

    /* Quick return */
    if (m == 0 || n == 0) {
        *numrank = 0;
        return;
    }

    /* Get machine parameters */
    big = dlamch("O");
    epsln = dlamch("E");
    sfmin = dlamch("S");

    /* Initialize */
    ascaled = 0;
    iwoff = 1;
    sconda = -ONE;
    n1 = n;

    /*
     * Row pivoting
     */
    if (rowprm) {
        iwoff = m;
        /* Compute row infinity norms */
        for (p = 0; p < m; p++) {
            rwork[p] = dlange("M", 1, n, &A[p], lda, NULL);
            /* Check for NaN/Inf */
            if (rwork[p] != rwork[p] || (rwork[p] * ZERO) != ZERO) {
                *info = -8;
                xerbla("DGESVDQ", -(*info));
                return;
            }
        }
        /* Sort rows by decreasing infinity norm */
        for (p = 0; p < m - 1; p++) {
            q = cblas_idamax(m - p, &rwork[p], 1) + p;
            iwork[n + p] = q;
            if (p != q) {
                rtmp = rwork[p];
                rwork[p] = rwork[q];
                rwork[q] = rtmp;
            }
        }

        /* Quick return for zero matrix */
        if (rwork[0] == ZERO) {
            *numrank = 0;
            dlaset("G", n, 1, ZERO, ZERO, S, n);
            if (wntus) dlaset("G", m, n, ZERO, ONE, U, ldu);
            if (wntua) dlaset("G", m, m, ZERO, ONE, U, ldu);
            if (wntva) dlaset("G", n, n, ZERO, ONE, V, ldv);
            if (wntuf) {
                dlaset("G", n, 1, ZERO, ZERO, work, n);
                dlaset("G", m, n, ZERO, ONE, U, ldu);
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

        /* Scale if needed to prevent overflow */
        if (rwork[0] > big / sqrt((f64)m)) {
            dlascl("G", 0, 0, sqrt((f64)m), ONE, m, n, A, lda, &ierr);
            ascaled = 1;
        }
        /* Apply row permutation */
        dlaswp(n, A, lda, 0, m - 2, &iwork[n], 1);
    } else {
        /* No row pivoting */
        rtmp = dlange("M", m, n, A, lda, NULL);
        /* Check for NaN/Inf */
        if (rtmp != rtmp || (rtmp * ZERO) != ZERO) {
            *info = -8;
            xerbla("DGESVDQ", -(*info));
            return;
        }
        /* Scale if needed to prevent overflow */
        if (rtmp > big / sqrt((f64)m)) {
            dlascl("G", 0, 0, sqrt((f64)m), ONE, m, n, A, lda, &ierr);
            ascaled = 1;
        }
    }

    /*
     * QR factorization with column pivoting: A * P = Q * R
     */
    for (p = 0; p < n; p++) {
        iwork[p] = 0;
    }
    dgeqp3(m, n, A, lda, iwork, work, &work[n], lwork - n, &ierr);

    /*
     * Numerical rank determination
     */
    nr = 1;
    if (accla) {
        /* Aggressive truncation: |R(i,i)| < sqrt(n)*eps*|R(1,1)| */
        rtmp = sqrt((f64)n) * epsln;
        for (p = 1; p < n; p++) {
            if (fabs(A[p + p * lda]) < rtmp * fabs(A[0])) break;
            nr++;
        }
    } else if (acclm) {
        /* Medium truncation: sudden drop or underflow */
        nr = 1;
        for (p = 1; p < n; p++) {
            if (fabs(A[p + p * lda]) < epsln * fabs(A[(p-1) + (p-1) * lda]) ||
                fabs(A[p + p * lda]) < sfmin) break;
            nr++;
        }
    } else {
        /* High accuracy: only exact zeros */
        nr = 1;
        for (p = 1; p < n; p++) {
            if (fabs(A[p + p * lda]) == ZERO) break;
            nr++;
        }

        /* Condition number estimation for ACCLH with CONDA */
        if (conda) {
            dlacpy("U", n, n, A, lda, V, ldv);
            for (p = 0; p < nr; p++) {
                rtmp = cblas_dnrm2(p + 1, &V[p * ldv], 1);
                cblas_dscal(p + 1, ONE / rtmp, &V[p * ldv], 1);
            }
            if (!lsvec && !rsvec) {
                dpocon("U", nr, V, ldv, ONE, &rtmp, work, &iwork[n + iwoff], &ierr);
            } else {
                dpocon("U", nr, V, ldv, ONE, &rtmp, &work[n], &iwork[n + iwoff], &ierr);
            }
            sconda = ONE / sqrt(rtmp);
        }
    }

    /*
     * Compute N1: dimension for left singular vectors output
     */
    if (wntur) {
        n1 = nr;
    } else if (wntus || wntuf) {
        n1 = n;
    } else if (wntua) {
        n1 = m;
    }

    /*
     * Main computation branches
     */
    if (!rsvec && !lsvec) {
        /*
         * Singular values only
         */
        if (rtrans) {
            /* Compute SVD of R^T */
            int minmn = (n < nr) ? n : nr;
            for (p = 0; p < minmn; p++) {
                for (q = p + 1; q < n; q++) {
                    A[q + p * lda] = A[p + q * lda];
                    if (q < nr) A[p + q * lda] = ZERO;
                }
            }
            dgesvd("N", "N", n, nr, A, lda, S, U, ldu, V, ldv, work, lwork, info);
        } else {
            /* Compute SVD of R */
            if (nr > 1) {
                dlaset("L", nr - 1, nr - 1, ZERO, ZERO, &A[1], lda);
            }
            dgesvd("N", "N", nr, n, A, lda, S, U, ldu, V, ldv, work, lwork, info);
        }

    } else if (lsvec && !rsvec) {
        /*
         * Left singular vectors requested
         */
        if (rtrans) {
            /* Apply DGESVD to R^T */
            for (p = 0; p < nr; p++) {
                for (q = p; q < n; q++) {
                    U[q + p * ldu] = A[p + q * lda];
                }
            }
            if (nr > 1) {
                dlaset("U", nr - 1, nr - 1, ZERO, ZERO, &U[ldu], ldu);
            }
            /* JOBU='N': U not referenced; JOBVT='O': overwrites A */
            dgesvd("N", "O", n, nr, U, ldu, S, NULL, 1, NULL, 1, &work[n], lwork - n, info);
            /* Transpose result */
            for (p = 0; p < nr; p++) {
                for (q = p + 1; q < nr; q++) {
                    rtmp = U[q + p * ldu];
                    U[q + p * ldu] = U[p + q * ldu];
                    U[p + q * ldu] = rtmp;
                }
            }
        } else {
            /* Apply DGESVD to R */
            dlacpy("U", nr, n, A, lda, U, ldu);
            if (nr > 1) {
                dlaset("L", nr - 1, nr - 1, ZERO, ZERO, &U[1], ldu);
            }
            /* JOBU='O': overwrites A; JOBVT='N': VT not referenced */
            dgesvd("O", "N", nr, n, U, ldu, S, NULL, 1, NULL, 1, &work[n], lwork - n, info);
        }

        /* Assemble left singular vectors */
        if (nr < m && !wntuf) {
            dlaset("A", m - nr, nr, ZERO, ZERO, &U[nr], ldu);
            if (nr < n1) {
                dlaset("A", nr, n1 - nr, ZERO, ZERO, &U[nr * ldu], ldu);
                dlaset("A", m - nr, n1 - nr, ZERO, ONE, &U[nr + nr * ldu], ldu);
            }
        }

        /* Apply Q factor */
        if (!wntuf) {
            dormqr("L", "N", m, n1, n, A, lda, work, U, ldu, &work[n], lwork - n, &ierr);
        }
        /* Undo row pivoting */
        if (rowprm && !wntuf) {
            dlaswp(n1, U, ldu, 0, m - 2, &iwork[n], -1);
        }

    } else if (rsvec && !lsvec) {
        /*
         * Right singular vectors requested
         */
        if (rtrans) {
            /* Apply DGESVD to R^T */
            for (p = 0; p < nr; p++) {
                for (q = p; q < n; q++) {
                    V[q + p * ldv] = A[p + q * lda];
                }
            }
            if (nr > 1) {
                dlaset("U", nr - 1, nr - 1, ZERO, ZERO, &V[ldv], ldv);
            }

            if (wntvr || nr == n) {
                /* JOBU='O': overwrites A; JOBVT='N': VT not referenced */
                dgesvd("O", "N", n, nr, V, ldv, S, U, ldu, NULL, ldu, &work[n], lwork - n, info);
                /* Transpose */
                for (p = 0; p < nr; p++) {
                    for (q = p + 1; q < nr; q++) {
                        rtmp = V[q + p * ldv];
                        V[q + p * ldv] = V[p + q * ldv];
                        V[p + q * ldv] = rtmp;
                    }
                }
                if (nr < n) {
                    for (p = 0; p < nr; p++) {
                        for (q = nr; q < n; q++) {
                            V[p + q * ldv] = V[q + p * ldv];
                        }
                    }
                }
                dlapmt(0, nr, n, V, ldv, iwork);
            } else {
                /* Need all N right vectors, NR < N */
                dlaset("G", n, n - nr, ZERO, ZERO, &V[nr * ldv], ldv);
                /* JOBU='O': overwrites A; JOBVT='N': VT not referenced */
                dgesvd("O", "N", n, n, V, ldv, S, U, ldu, NULL, ldu, &work[n], lwork - n, info);
                /* Transpose */
                for (p = 0; p < n; p++) {
                    for (q = p + 1; q < n; q++) {
                        rtmp = V[q + p * ldv];
                        V[q + p * ldv] = V[p + q * ldv];
                        V[p + q * ldv] = rtmp;
                    }
                }
                dlapmt(0, n, n, V, ldv, iwork);
            }
        } else {
            /* Apply DGESVD to R */
            dlacpy("U", nr, n, A, lda, V, ldv);
            if (nr > 1) {
                dlaset("L", nr - 1, nr - 1, ZERO, ZERO, &V[1], ldv);
            }

            if (wntvr || nr == n) {
                /* JOBU='N': U not referenced; JOBVT='O': overwrites A */
                dgesvd("N", "O", nr, n, V, ldv, S, NULL, 1, NULL, 1, &work[n], lwork - n, info);
                dlapmt(0, nr, n, V, ldv, iwork);
            } else {
                /* Need all N right vectors, NR < N */
                dlaset("G", n - nr, n, ZERO, ZERO, &V[nr], ldv);
                /* JOBU='N': U not referenced; JOBVT='O': overwrites A */
                dgesvd("N", "O", n, n, V, ldv, S, NULL, 1, NULL, 1, &work[n], lwork - n, info);
                dlapmt(0, n, n, V, ldv, iwork);
            }
        }

    } else {
        /*
         * Full SVD: both left and right singular vectors
         */
        if (rtrans) {
            /*
             * Apply DGESVD to R^T
             */
            if (wntvr || nr == n) {
                /* Copy R^T to V */
                for (p = 0; p < nr; p++) {
                    for (q = p; q < n; q++) {
                        V[q + p * ldv] = A[p + q * lda];
                    }
                }
                if (nr > 1) {
                    dlaset("U", nr - 1, nr - 1, ZERO, ZERO, &V[ldv], ldv);
                }

                dgesvd("O", "A", n, nr, V, ldv, S, NULL, 1, U, ldu, &work[n], lwork - n, info);

                /* Transpose V */
                for (p = 0; p < nr; p++) {
                    for (q = p + 1; q < nr; q++) {
                        rtmp = V[q + p * ldv];
                        V[q + p * ldv] = V[p + q * ldv];
                        V[p + q * ldv] = rtmp;
                    }
                }
                if (nr < n) {
                    for (p = 0; p < nr; p++) {
                        for (q = nr; q < n; q++) {
                            V[p + q * ldv] = V[q + p * ldv];
                        }
                    }
                }
                dlapmt(0, nr, n, V, ldv, iwork);

                /* Transpose U */
                for (p = 0; p < nr; p++) {
                    for (q = p + 1; q < nr; q++) {
                        rtmp = U[q + p * ldu];
                        U[q + p * ldu] = U[p + q * ldu];
                        U[p + q * ldu] = rtmp;
                    }
                }

                /* Assemble U */
                if (nr < m && !wntuf) {
                    dlaset("A", m - nr, nr, ZERO, ZERO, &U[nr], ldu);
                    if (nr < n1) {
                        dlaset("A", nr, n1 - nr, ZERO, ZERO, &U[nr * ldu], ldu);
                        dlaset("A", m - nr, n1 - nr, ZERO, ONE, &U[nr + nr * ldu], ldu);
                    }
                }
            } else {
                /* Need all N right vectors, NR < N */
                optratio = 2;
                if (optratio * nr > n) {
                    /* Padding approach */
                    for (p = 0; p < nr; p++) {
                        for (q = p; q < n; q++) {
                            V[q + p * ldv] = A[p + q * lda];
                        }
                    }
                    if (nr > 1) {
                        dlaset("U", nr - 1, nr - 1, ZERO, ZERO, &V[ldv], ldv);
                    }
                    dlaset("A", n, n - nr, ZERO, ZERO, &V[nr * ldv], ldv);

                    dgesvd("O", "A", n, n, V, ldv, S, NULL, 1, U, ldu, &work[n], lwork - n, info);

                    /* Transpose V */
                    for (p = 0; p < n; p++) {
                        for (q = p + 1; q < n; q++) {
                            rtmp = V[q + p * ldv];
                            V[q + p * ldv] = V[p + q * ldv];
                            V[p + q * ldv] = rtmp;
                        }
                    }
                    dlapmt(0, n, n, V, ldv, iwork);

                    /* Transpose U */
                    for (p = 0; p < n; p++) {
                        for (q = p + 1; q < n; q++) {
                            rtmp = U[q + p * ldu];
                            U[q + p * ldu] = U[p + q * ldu];
                            U[p + q * ldu] = rtmp;
                        }
                    }

                    /* Assemble U */
                    if (n < m && !wntuf) {
                        dlaset("A", m - n, n, ZERO, ZERO, &U[n], ldu);
                        if (n < n1) {
                            dlaset("A", n, n1 - n, ZERO, ZERO, &U[n * ldu], ldu);
                            dlaset("A", m - n, n1 - n, ZERO, ONE, &U[n + n * ldu], ldu);
                        }
                    }
                } else {
                    /* QR optimization */
                    for (p = 0; p < nr; p++) {
                        for (q = p; q < n; q++) {
                            U[q + (nr + p) * ldu] = A[p + q * lda];
                        }
                    }
                    if (nr > 1) {
                        dlaset("U", nr - 1, nr - 1, ZERO, ZERO, &U[(nr + 1) * ldu], ldu);
                    }
                    dgeqrf(n, nr, &U[nr * ldu], ldu, &work[n], &work[n + nr], lwork - n - nr, &ierr);
                    for (p = 0; p < nr; p++) {
                        for (q = 0; q < n; q++) {
                            V[q + p * ldv] = U[p + (nr + q) * ldu];
                        }
                    }
                    dlaset("U", nr - 1, nr - 1, ZERO, ZERO, &V[ldv], ldv);
                    dgesvd("S", "O", nr, nr, V, ldv, S, U, ldu, NULL, 1, &work[n + nr], lwork - n - nr, info);
                    dlaset("A", n - nr, nr, ZERO, ZERO, &V[nr], ldv);
                    dlaset("A", nr, n - nr, ZERO, ZERO, &V[nr * ldv], ldv);
                    dlaset("A", n - nr, n - nr, ZERO, ONE, &V[nr + nr * ldv], ldv);
                    dormqr("R", "C", n, n, nr, &U[nr * ldu], ldu, &work[n], V, ldv, &work[n + nr], lwork - n - nr, &ierr);
                    dlapmt(0, n, n, V, ldv, iwork);

                    /* Assemble U */
                    if (nr < m && !wntuf) {
                        dlaset("A", m - nr, nr, ZERO, ZERO, &U[nr], ldu);
                        if (nr < n1) {
                            dlaset("A", nr, n1 - nr, ZERO, ZERO, &U[nr * ldu], ldu);
                            dlaset("A", m - nr, n1 - nr, ZERO, ONE, &U[nr + nr * ldu], ldu);
                        }
                    }
                }
            }
        } else {
            /*
             * Apply DGESVD to R (recommended option)
             */
            if (wntvr || nr == n) {
                /* Copy R to V */
                dlacpy("U", nr, n, A, lda, V, ldv);
                if (nr > 1) {
                    dlaset("L", nr - 1, nr - 1, ZERO, ZERO, &V[1], ldv);
                }

                dgesvd("S", "O", nr, n, V, ldv, S, U, ldu, NULL, 1, &work[n], lwork - n, info);
                dlapmt(0, nr, n, V, ldv, iwork);

                /* Assemble U */
                if (nr < m && !wntuf) {
                    dlaset("A", m - nr, nr, ZERO, ZERO, &U[nr], ldu);
                    if (nr < n1) {
                        dlaset("A", nr, n1 - nr, ZERO, ZERO, &U[nr * ldu], ldu);
                        dlaset("A", m - nr, n1 - nr, ZERO, ONE, &U[nr + nr * ldu], ldu);
                    }
                }
            } else {
                /* Need all N right vectors, NR < N */
                optratio = 2;
                if (optratio * nr > n) {
                    /* Padding approach */
                    dlacpy("U", nr, n, A, lda, V, ldv);
                    if (nr > 1) {
                        dlaset("L", nr - 1, nr - 1, ZERO, ZERO, &V[1], ldv);
                    }
                    dlaset("A", n - nr, n, ZERO, ZERO, &V[nr], ldv);

                    dgesvd("S", "O", n, n, V, ldv, S, U, ldu, NULL, 1, &work[n], lwork - n, info);
                    dlapmt(0, n, n, V, ldv, iwork);

                    /* Assemble U */
                    if (n < m && !wntuf) {
                        dlaset("A", m - n, n, ZERO, ZERO, &U[n], ldu);
                        if (n < n1) {
                            dlaset("A", n, n1 - n, ZERO, ZERO, &U[n * ldu], ldu);
                            dlaset("A", m - n, n1 - n, ZERO, ONE, &U[n + n * ldu], ldu);
                        }
                    }
                } else {
                    /* LQ optimization */
                    dlacpy("U", nr, n, A, lda, &U[nr], ldu);
                    if (nr > 1) {
                        dlaset("L", nr - 1, nr - 1, ZERO, ZERO, &U[nr + 1], ldu);
                    }
                    dgelqf(nr, n, &U[nr], ldu, &work[n], &work[n + nr], lwork - n - nr, &ierr);
                    dlacpy("L", nr, nr, &U[nr], ldu, V, ldv);
                    if (nr > 1) {
                        dlaset("U", nr - 1, nr - 1, ZERO, ZERO, &V[ldv], ldv);
                    }
                    dgesvd("S", "O", nr, nr, V, ldv, S, U, ldu, NULL, 1, &work[n + nr], lwork - n - nr, info);
                    dlaset("A", n - nr, nr, ZERO, ZERO, &V[nr], ldv);
                    dlaset("A", nr, n - nr, ZERO, ZERO, &V[nr * ldv], ldv);
                    dlaset("A", n - nr, n - nr, ZERO, ONE, &V[nr + nr * ldv], ldv);
                    dormlq("R", "N", n, n, nr, &U[nr], ldu, &work[n], V, ldv, &work[n + nr], lwork - n - nr, &ierr);
                    dlapmt(0, n, n, V, ldv, iwork);

                    /* Assemble U */
                    if (nr < m && !wntuf) {
                        dlaset("A", m - nr, nr, ZERO, ZERO, &U[nr], ldu);
                        if (nr < n1) {
                            dlaset("A", nr, n1 - nr, ZERO, ZERO, &U[nr * ldu], ldu);
                            dlaset("A", m - nr, n1 - nr, ZERO, ONE, &U[nr + nr * ldu], ldu);
                        }
                    }
                }
            }
        }

        /* Apply Q factor */
        if (!wntuf) {
            dormqr("L", "N", m, n1, n, A, lda, work, U, ldu, &work[n], lwork - n, &ierr);
        }
        /* Undo row pivoting */
        if (rowprm && !wntuf) {
            dlaswp(n1, U, ldu, 0, m - 2, &iwork[n], -1);
        }
    }

    /*
     * Check for zeros in S and update rank
     */
    p = nr;
    for (q = nr - 1; q >= 0; q--) {
        if (S[q] > ZERO) break;
        nr--;
    }

    /* Zero truncated singular values */
    if (nr < n) {
        dlaset("G", n - nr, 1, ZERO, ZERO, &S[nr], n);
    }

    /* Undo scaling */
    if (ascaled) {
        dlascl("G", 0, 0, ONE, sqrt((f64)m), nr, 1, S, n, &ierr);
    }

    if (conda) rwork[0] = sconda;
    rwork[1] = (f64)(p - nr);

    *numrank = nr;
    *info = ierr;
}
