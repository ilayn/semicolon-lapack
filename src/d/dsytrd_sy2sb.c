/**
 * @file dsytrd_sy2sb.c
 * @brief DSYTRD_SY2SB reduces a real symmetric matrix to band-diagonal form.
 */

#include "semicolon_lapack_double.h"
#include "semicolon_cblas.h"

void dsytrd_sy2sb(const char* uplo, const INT n, const INT kd,
                  f64* A, const INT lda,
                  f64* AB, const INT ldab,
                  f64* tau,
                  f64* work, const INT lwork, INT* info)
{
    const f64 rone = 1.0;
    const f64 zero = 0.0;
    const f64 one = 1.0;
    const f64 half = 0.5;

    INT lquery, upper;
    INT i, j, iinfo, lwmin, pn, pk, lk;
    INT ldt, ldw, lds2, lds1;
    INT ls2, ls1, lw, lt;
    INT tpos, wpos, s2pos, s1pos;

    *info = 0;
    upper = (uplo[0] == 'U' || uplo[0] == 'u');
    lquery = (lwork == -1);

    if (n <= kd + 1) {
        lwmin = 1;
    } else {
        lwmin = ilaenv2stage(4, "DSYTRD_SY2SB", " ", n, kd, -1, -1);
    }

    if (!upper && !(uplo[0] == 'L' || uplo[0] == 'l')) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (kd < 0) {
        *info = -3;
    } else if (lda < ((1 > n) ? 1 : n)) {
        *info = -5;
    } else if (ldab < ((1 > kd + 1) ? 1 : kd + 1)) {
        *info = -7;
    } else if (lwork < lwmin && !lquery) {
        *info = -10;
    }

    if (*info != 0) {
        xerbla("DSYTRD_SY2SB", -(*info));
        return;
    } else if (lquery) {
        work[0] = (f64)lwmin;
        return;
    }

    if (n <= kd + 1) {
        if (upper) {
            for (i = 0; i < n; i++) {
                lk = (kd + 1 < i + 1) ? (kd + 1) : (i + 1);
                cblas_dcopy(lk, &A[(i - lk + 1) + i * lda], 1,
                            &AB[(kd + 1 - lk) + i * ldab], 1);
            }
        } else {
            for (i = 0; i < n; i++) {
                lk = (kd + 1 < n - i) ? (kd + 1) : (n - i);
                cblas_dcopy(lk, &A[i + i * lda], 1, &AB[0 + i * ldab], 1);
            }
        }
        work[0] = 1.0;
        return;
    }

    ldt = kd;
    lds1 = kd;
    lt = ldt * kd;
    lw = n * kd;
    ls1 = lds1 * kd;
    ls2 = lwmin - lt - lw - ls1;
    tpos = 0;
    wpos = tpos + lt;
    s1pos = wpos + lw;
    s2pos = s1pos + ls1;
    if (upper) {
        ldw = kd;
        lds2 = kd;
    } else {
        ldw = n;
        lds2 = n;
    }

    dlaset("A", ldt, kd, zero, zero, &work[tpos], ldt);

    if (upper) {
        for (i = 0; i < n - kd; i += kd) {
            pn = n - i - kd;
            pk = (n - i - kd < kd) ? (n - i - kd) : kd;

            dgelqf(kd, pn, &A[i + (i + kd) * lda], lda,
                   &tau[i], &work[s2pos], ls2, &iinfo);

            for (j = i; j < i + pk; j++) {
                lk = ((kd < n - j - 1) ? kd : (n - j - 1)) + 1;
                cblas_dcopy(lk, &A[j + j * lda], lda,
                            &AB[kd + j * ldab], ldab - 1);
            }

            dlaset("Lower", pk, pk, zero, one,
                   &A[i + (i + kd) * lda], lda);

            dlarft("Forward", "Rowwise", pn, pk,
                   &A[i + (i + kd) * lda], lda, &tau[i],
                   &work[tpos], ldt);

            cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                        pk, pn, pk,
                        one, &work[tpos], ldt,
                        &A[i + (i + kd) * lda], lda,
                        zero, &work[s2pos], lds2);

            cblas_dsymm(CblasColMajor, CblasRight, CblasUpper,
                        pk, pn,
                        one, &A[(i + kd) + (i + kd) * lda], lda,
                        &work[s2pos], lds2,
                        zero, &work[wpos], ldw);

            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                        pk, pk, pn,
                        one, &work[wpos], ldw,
                        &work[s2pos], lds2,
                        zero, &work[s1pos], lds1);

            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                        pk, pn, pk,
                        -half, &work[s1pos], lds1,
                        &A[i + (i + kd) * lda], lda,
                        one, &work[wpos], ldw);

            cblas_dsyr2k(CblasColMajor, CblasUpper, CblasTrans,
                         pn, pk,
                         -one, &A[i + (i + kd) * lda], lda,
                         &work[wpos], ldw,
                         rone, &A[(i + kd) + (i + kd) * lda], lda);
        }

        for (j = n - kd; j < n; j++) {
            lk = ((kd < n - j - 1) ? kd : (n - j - 1)) + 1;
            cblas_dcopy(lk, &A[j + j * lda], lda,
                        &AB[kd + j * ldab], ldab - 1);
        }

    } else {

        for (i = 0; i < n - kd; i += kd) {
            pn = n - i - kd;
            pk = (n - i - kd < kd) ? (n - i - kd) : kd;

            dgeqrf(pn, kd, &A[(i + kd) + i * lda], lda,
                   &tau[i], &work[s2pos], ls2, &iinfo);

            for (j = i; j < i + pk; j++) {
                lk = ((kd < n - j - 1) ? kd : (n - j - 1)) + 1;
                cblas_dcopy(lk, &A[j + j * lda], 1, &AB[0 + j * ldab], 1);
            }

            dlaset("Upper", pk, pk, zero, one,
                   &A[(i + kd) + i * lda], lda);

            dlarft("Forward", "Columnwise", pn, pk,
                   &A[(i + kd) + i * lda], lda, &tau[i],
                   &work[tpos], ldt);

            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                        pn, pk, pk,
                        one, &A[(i + kd) + i * lda], lda,
                        &work[tpos], ldt,
                        zero, &work[s2pos], lds2);

            cblas_dsymm(CblasColMajor, CblasLeft, CblasLower,
                        pn, pk,
                        one, &A[(i + kd) + (i + kd) * lda], lda,
                        &work[s2pos], lds2,
                        zero, &work[wpos], ldw);

            cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                        pk, pk, pn,
                        one, &work[s2pos], lds2,
                        &work[wpos], ldw,
                        zero, &work[s1pos], lds1);

            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                        pn, pk, pk,
                        -half, &A[(i + kd) + i * lda], lda,
                        &work[s1pos], lds1,
                        one, &work[wpos], ldw);

            cblas_dsyr2k(CblasColMajor, CblasLower, CblasNoTrans,
                         pn, pk,
                         -one, &A[(i + kd) + i * lda], lda,
                         &work[wpos], ldw,
                         rone, &A[(i + kd) + (i + kd) * lda], lda);
        }

        for (j = n - kd; j < n; j++) {
            lk = ((kd < n - j - 1) ? kd : (n - j - 1)) + 1;
            cblas_dcopy(lk, &A[j + j * lda], 1, &AB[0 + j * ldab], 1);
        }
    }

    work[0] = (f64)lwmin;
}
