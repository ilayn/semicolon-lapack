/**
 * @file ssytrd_sy2sb.c
 * @brief SSYTRD_SY2SB reduces a real symmetric matrix to band-diagonal form.
 */

#include "semicolon_lapack_single.h"
#include <cblas.h>

void ssytrd_sy2sb(const char* uplo, const int n, const int kd,
                  f32* A, const int lda,
                  f32* AB, const int ldab,
                  f32* tau,
                  f32* work, const int lwork, int* info)
{
    const f32 rone = 1.0f;
    const f32 zero = 0.0f;
    const f32 one = 1.0f;
    const f32 half = 0.5f;

    int lquery, upper;
    int i, j, iinfo, lwmin, pn, pk, lk;
    int ldt, ldw, lds2, lds1;
    int ls2, ls1, lw, lt;
    int tpos, wpos, s2pos, s1pos;

    *info = 0;
    upper = (uplo[0] == 'U' || uplo[0] == 'u');
    lquery = (lwork == -1);

    if (n <= kd + 1) {
        lwmin = 1;
    } else {
        lwmin = ilaenv2stage(4, "SSYTRD_SY2SB", " ", n, kd, -1, -1);
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
        xerbla("SSYTRD_SY2SB", -(*info));
        return;
    } else if (lquery) {
        work[0] = (f32)lwmin;
        return;
    }

    if (n <= kd + 1) {
        if (upper) {
            for (i = 0; i < n; i++) {
                lk = (kd + 1 < i + 1) ? (kd + 1) : (i + 1);
                cblas_scopy(lk, &A[(i - lk + 1) + i * lda], 1,
                            &AB[(kd + 1 - lk) + i * ldab], 1);
            }
        } else {
            for (i = 0; i < n; i++) {
                lk = (kd + 1 < n - i) ? (kd + 1) : (n - i);
                cblas_scopy(lk, &A[i + i * lda], 1, &AB[0 + i * ldab], 1);
            }
        }
        work[0] = 1.0f;
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

    slaset("A", ldt, kd, zero, zero, &work[tpos], ldt);

    if (upper) {
        for (i = 0; i < n - kd; i += kd) {
            pn = n - i - kd;
            pk = (n - i - kd < kd) ? (n - i - kd) : kd;

            sgelqf(kd, pn, &A[i + (i + kd) * lda], lda,
                   &tau[i], &work[s2pos], ls2, &iinfo);

            for (j = i; j < i + pk; j++) {
                lk = ((kd < n - j - 1) ? kd : (n - j - 1)) + 1;
                cblas_scopy(lk, &A[j + j * lda], lda,
                            &AB[kd + j * ldab], ldab - 1);
            }

            slaset("Lower", pk, pk, zero, one,
                   &A[i + (i + kd) * lda], lda);

            slarft("Forward", "Rowwise", pn, pk,
                   &A[i + (i + kd) * lda], lda, &tau[i],
                   &work[tpos], ldt);

            cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                        pk, pn, pk,
                        one, &work[tpos], ldt,
                        &A[i + (i + kd) * lda], lda,
                        zero, &work[s2pos], lds2);

            cblas_ssymm(CblasColMajor, CblasRight, CblasUpper,
                        pk, pn,
                        one, &A[(i + kd) + (i + kd) * lda], lda,
                        &work[s2pos], lds2,
                        zero, &work[wpos], ldw);

            cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                        pk, pk, pn,
                        one, &work[wpos], ldw,
                        &work[s2pos], lds2,
                        zero, &work[s1pos], lds1);

            cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                        pk, pn, pk,
                        -half, &work[s1pos], lds1,
                        &A[i + (i + kd) * lda], lda,
                        one, &work[wpos], ldw);

            cblas_ssyr2k(CblasColMajor, CblasUpper, CblasTrans,
                         pn, pk,
                         -one, &A[i + (i + kd) * lda], lda,
                         &work[wpos], ldw,
                         rone, &A[(i + kd) + (i + kd) * lda], lda);
        }

        for (j = n - kd; j < n; j++) {
            lk = ((kd < n - j - 1) ? kd : (n - j - 1)) + 1;
            cblas_scopy(lk, &A[j + j * lda], lda,
                        &AB[kd + j * ldab], ldab - 1);
        }

    } else {

        for (i = 0; i < n - kd; i += kd) {
            pn = n - i - kd;
            pk = (n - i - kd < kd) ? (n - i - kd) : kd;

            sgeqrf(pn, kd, &A[(i + kd) + i * lda], lda,
                   &tau[i], &work[s2pos], ls2, &iinfo);

            for (j = i; j < i + pk; j++) {
                lk = ((kd < n - j - 1) ? kd : (n - j - 1)) + 1;
                cblas_scopy(lk, &A[j + j * lda], 1, &AB[0 + j * ldab], 1);
            }

            slaset("Upper", pk, pk, zero, one,
                   &A[(i + kd) + i * lda], lda);

            slarft("Forward", "Columnwise", pn, pk,
                   &A[(i + kd) + i * lda], lda, &tau[i],
                   &work[tpos], ldt);

            cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                        pn, pk, pk,
                        one, &A[(i + kd) + i * lda], lda,
                        &work[tpos], ldt,
                        zero, &work[s2pos], lds2);

            cblas_ssymm(CblasColMajor, CblasLeft, CblasLower,
                        pn, pk,
                        one, &A[(i + kd) + (i + kd) * lda], lda,
                        &work[s2pos], lds2,
                        zero, &work[wpos], ldw);

            cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                        pk, pk, pn,
                        one, &work[s2pos], lds2,
                        &work[wpos], ldw,
                        zero, &work[s1pos], lds1);

            cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                        pn, pk, pk,
                        -half, &A[(i + kd) + i * lda], lda,
                        &work[s1pos], lds1,
                        one, &work[wpos], ldw);

            cblas_ssyr2k(CblasColMajor, CblasLower, CblasNoTrans,
                         pn, pk,
                         -one, &A[(i + kd) + i * lda], lda,
                         &work[wpos], ldw,
                         rone, &A[(i + kd) + (i + kd) * lda], lda);
        }

        for (j = n - kd; j < n; j++) {
            lk = ((kd < n - j - 1) ? kd : (n - j - 1)) + 1;
            cblas_scopy(lk, &A[j + j * lda], 1, &AB[0 + j * ldab], 1);
        }
    }

    work[0] = (f32)lwmin;
}
