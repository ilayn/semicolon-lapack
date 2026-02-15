/**
 * @file chetrd_he2hb.c
 * @brief CHETRD_HE2HB reduces a complex Hermitian matrix to band-diagonal form.
 */

#include <complex.h>
#include <cblas.h>
#include "semicolon_lapack_complex_single.h"

/**
 * CHETRD_HE2HB reduces a complex Hermitian matrix A to complex Hermitian
 * band-diagonal form AB by a unitary similarity transformation:
 * Q**H * A * Q = AB.
 *
 * @param[in]     uplo    = 'U':  Upper triangle of A is stored;
 *                          = 'L':  Lower triangle of A is stored.
 * @param[in]     n       The order of the matrix A.  N >= 0.
 * @param[in]     kd      The number of superdiagonals of the reduced matrix
 *                         if UPLO = 'U', or the number of subdiagonals if
 *                         UPLO = 'L'.  KD >= 0.
 *                         The reduced matrix is stored in the array AB.
 * @param[in,out] A       Single complex array, dimension (LDA,N).
 *                         On entry, the Hermitian matrix A.  If UPLO = 'U',
 *                         the leading N-by-N upper triangular part of A
 *                         contains the upper triangular part of the matrix A,
 *                         and the strictly lower triangular part of A is not
 *                         referenced.  If UPLO = 'L', the leading N-by-N
 *                         lower triangular part of A contains the lower
 *                         triangular part of the matrix A, and the strictly
 *                         upper triangular part of A is not referenced.
 *                         On exit, if UPLO = 'U', the diagonal and first
 *                         superdiagonal of A are overwritten by the
 *                         corresponding elements of the tridiagonal matrix T,
 *                         and the elements above the first superdiagonal, with
 *                         the array TAU, represent the unitary matrix Q as a
 *                         product of elementary reflectors; if UPLO = 'L', the
 *                         diagonal and first subdiagonal of A are overwritten
 *                         by the corresponding elements of the tridiagonal
 *                         matrix T, and the elements below the first
 *                         subdiagonal, with the array TAU, represent the
 *                         unitary matrix Q as a product of elementary
 *                         reflectors. See Further Details.
 * @param[in]     lda     The leading dimension of the array A.  LDA >= max(1,N).
 * @param[out]    AB      Single complex array, dimension (LDAB,N).
 *                         On exit, the upper or lower triangle of the
 *                         Hermitian band matrix A, stored in the first KD+1
 *                         rows of the array.  The j-th column of A is stored
 *                         in the j-th column of the array AB as follows:
 *                         if UPLO = 'U', AB(kd+i-j,j) = A(i,j) for
 *                         max(0,j-kd)<=i<=j;
 *                         if UPLO = 'L', AB(i-j,j) = A(i,j) for
 *                         j<=i<=min(n-1,j+kd).
 * @param[in]     ldab    The leading dimension of the array AB.  LDAB >= KD+1.
 * @param[out]    tau     Single complex array, dimension (N-KD).
 *                         The scalar factors of the elementary reflectors
 *                         (see Further Details).
 * @param[out]    work    Single complex array, dimension (MAX(1,LWORK)).
 *                         On exit, if INFO = 0, or if LWORK = -1,
 *                         WORK(1) returns the size of LWORK.
 * @param[in]     lwork   The dimension of the array WORK.
 *                         If N <= KD+1, LWORK >= 1, else
 *                         LWORK = MAX(1, LWORK_QUERY).
 *                         LWORK_QUERY = N*KD + N*max(KD,FACTOPTNB) + 2*KD*KD
 *                         where FACTOPTNB is the blocking used by the QR or LQ
 *                         algorithm, usually FACTOPTNB=128 is a good choice
 *                         otherwise putting LWORK=-1 will provide the size of
 *                         WORK.
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value.
 */
void chetrd_he2hb(const char* uplo, const int n, const int kd,
                  c64* A, const int lda,
                  c64* AB, const int ldab,
                  c64* tau,
                  c64* work, const int lwork, int* info)
{
    const f32 rone = 1.0f;
    const c64 zero = CMPLXF(0.0f, 0.0f);
    const c64 one = CMPLXF(1.0f, 0.0f);
    const c64 mone = CMPLXF(-1.0f, 0.0f);
    const c64 mhalf = CMPLXF(-0.5f, 0.0f);

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
        lwmin = ilaenv2stage(4, "CHETRD_HE2HB", "", n, kd, -1, -1);
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
        xerbla("CHETRD_HE2HB", -(*info));
        return;
    } else if (lquery) {
        work[0] = CMPLXF((f32)lwmin, 0.0f);
        return;
    }

    if (n <= kd + 1) {
        if (upper) {
            for (i = 0; i < n; i++) {
                lk = (kd + 1 < i + 1) ? (kd + 1) : (i + 1);
                cblas_ccopy(lk, &A[(i - lk + 1) + i * lda], 1,
                            &AB[(kd + 1 - lk) + i * ldab], 1);
            }
        } else {
            for (i = 0; i < n; i++) {
                lk = (kd + 1 < n - i) ? (kd + 1) : (n - i);
                cblas_ccopy(lk, &A[i + i * lda], 1, &AB[0 + i * ldab], 1);
            }
        }
        work[0] = CMPLXF(1.0f, 0.0f);
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

    claset("A", ldt, kd, zero, zero, &work[tpos], ldt);

    if (upper) {
        for (i = 0; i < n - kd; i += kd) {
            pn = n - i - kd;
            pk = (n - i - kd < kd) ? (n - i - kd) : kd;

            cgelqf(kd, pn, &A[i + (i + kd) * lda], lda,
                   &tau[i], &work[s2pos], ls2, &iinfo);

            for (j = i; j < i + pk; j++) {
                lk = ((kd < n - j - 1) ? kd : (n - j - 1)) + 1;
                cblas_ccopy(lk, &A[j + j * lda], lda,
                            &AB[kd + j * ldab], ldab - 1);
            }

            claset("Lower", pk, pk, zero, one,
                   &A[i + (i + kd) * lda], lda);

            clarft("Forward", "Rowwise", pn, pk,
                   &A[i + (i + kd) * lda], lda, &tau[i],
                   &work[tpos], ldt);

            cblas_cgemm(CblasColMajor, CblasConjTrans, CblasNoTrans,
                        pk, pn, pk,
                        &one, &work[tpos], ldt,
                        &A[i + (i + kd) * lda], lda,
                        &zero, &work[s2pos], lds2);

            cblas_chemm(CblasColMajor, CblasRight, CblasUpper,
                        pk, pn,
                        &one, &A[(i + kd) + (i + kd) * lda], lda,
                        &work[s2pos], lds2,
                        &zero, &work[wpos], ldw);

            cblas_cgemm(CblasColMajor, CblasNoTrans, CblasConjTrans,
                        pk, pk, pn,
                        &one, &work[wpos], ldw,
                        &work[s2pos], lds2,
                        &zero, &work[s1pos], lds1);

            cblas_cgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                        pk, pn, pk,
                        &mhalf, &work[s1pos], lds1,
                        &A[i + (i + kd) * lda], lda,
                        &one, &work[wpos], ldw);

            cblas_cher2k(CblasColMajor, CblasUpper, CblasConjTrans,
                         pn, pk,
                         &mone, &A[i + (i + kd) * lda], lda,
                         &work[wpos], ldw,
                         rone, &A[(i + kd) + (i + kd) * lda], lda);
        }

        for (j = n - kd; j < n; j++) {
            lk = ((kd < n - j - 1) ? kd : (n - j - 1)) + 1;
            cblas_ccopy(lk, &A[j + j * lda], lda,
                        &AB[kd + j * ldab], ldab - 1);
        }

    } else {

        for (i = 0; i < n - kd; i += kd) {
            pn = n - i - kd;
            pk = (n - i - kd < kd) ? (n - i - kd) : kd;

            cgeqrf(pn, kd, &A[(i + kd) + i * lda], lda,
                   &tau[i], &work[s2pos], ls2, &iinfo);

            for (j = i; j < i + pk; j++) {
                lk = ((kd < n - j - 1) ? kd : (n - j - 1)) + 1;
                cblas_ccopy(lk, &A[j + j * lda], 1, &AB[0 + j * ldab], 1);
            }

            claset("Upper", pk, pk, zero, one,
                   &A[(i + kd) + i * lda], lda);

            clarft("Forward", "Columnwise", pn, pk,
                   &A[(i + kd) + i * lda], lda, &tau[i],
                   &work[tpos], ldt);

            cblas_cgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                        pn, pk, pk,
                        &one, &A[(i + kd) + i * lda], lda,
                        &work[tpos], ldt,
                        &zero, &work[s2pos], lds2);

            cblas_chemm(CblasColMajor, CblasLeft, CblasLower,
                        pn, pk,
                        &one, &A[(i + kd) + (i + kd) * lda], lda,
                        &work[s2pos], lds2,
                        &zero, &work[wpos], ldw);

            cblas_cgemm(CblasColMajor, CblasConjTrans, CblasNoTrans,
                        pk, pk, pn,
                        &one, &work[s2pos], lds2,
                        &work[wpos], ldw,
                        &zero, &work[s1pos], lds1);

            cblas_cgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                        pn, pk, pk,
                        &mhalf, &A[(i + kd) + i * lda], lda,
                        &work[s1pos], lds1,
                        &one, &work[wpos], ldw);

            cblas_cher2k(CblasColMajor, CblasLower, CblasNoTrans,
                         pn, pk,
                         &mone, &A[(i + kd) + i * lda], lda,
                         &work[wpos], ldw,
                         rone, &A[(i + kd) + (i + kd) * lda], lda);
        }

        for (j = n - kd; j < n; j++) {
            lk = ((kd < n - j - 1) ? kd : (n - j - 1)) + 1;
            cblas_ccopy(lk, &A[j + j * lda], 1, &AB[0 + j * ldab], 1);
        }
    }

    work[0] = CMPLXF((f32)lwmin, 0.0f);
}
