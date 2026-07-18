/**
 * @file ssytrd_sy2sb.c
 * @brief SSYTRD_SY2SB reduces a real symmetric matrix to band-diagonal form.
 */

#include "semicolon_lapack_single.h"
#include "semicolon_cblas.h"

/**
 * SSYTRD_SY2SB reduces a real symmetric matrix `A` to real symmetric
 * band-diagonal form `AB` by an orthogonal similarity transformation:
 * `Q**T * A * Q = AB`.
 *
 * @param[in]     uplo  `'U'`: Upper triangle of `A` is stored;
 *                      `'L'`: Lower triangle of `A` is stored.
 * @param[in]     n     The order of the matrix `A`. `n>=0`.
 * @param[in]     kd    The number of superdiagonals of the reduced matrix if
 *                      `uplo='U'`, or the number of subdiagonals if `uplo='L'`.
 *                      `kd>=0`. The reduced matrix is stored in the array `AB`.
 * @param[in,out] A     On entry, the symmetric matrix `A`. If `uplo='U'`, the leading
 *                      `n`-by-`n` upper triangular part of `A` contains the upper
 *                      triangular part of the matrix `A`, and the strictly lower
 *                      triangular part of `A` is not referenced. If `uplo='L'`, the
 *                      leading `n`-by-`n` lower triangular part of `A` contains the lower
 *                      triangular part of the matrix `A`, and the strictly upper
 *                      triangular part of `A` is not referenced.
 *                      On exit, if `uplo='U'`, the diagonal and first superdiagonal
 *                      of `A` are overwritten by the corresponding elements of the
 *                      tridiagonal matrix T, and the elements above the first
 *                      superdiagonal, with the array `tau`, represent the orthogonal
 *                      matrix Q as a product of elementary reflectors; if `uplo='L'`,
 *                      the diagonal and first subdiagonal of `A` are overwritten by the
 *                      corresponding elements of the tridiagonal matrix T, and the
 *                      elements below the first subdiagonal, with the array `tau`,
 *                      represent the orthogonal matrix Q as a product of elementary
 *                      reflectors. See Further Details.
 * @param[in]     lda   The leading dimension of the array `A`. `lda>=max(1,n)`.
 * @param[out]    AB    On exit, the upper or lower triangle of the symmetric band
 *                      matrix A, stored in the first `kd+1` rows of the array. The
 *                      j-th column of A is stored in the j-th column of the array `AB`
 *                      as follows: if `uplo='U'`, `AB[kd+i-j,j]=A[i,j]` for
 *                      `max(0,j-kd)<=i<=j`; if `uplo='L'`, `AB[i-j,j]=A[i,j]` for
 *                      `j<=i<=min(n-1,j+kd)`.
 * @param[in]     ldab  The leading dimension of the array `AB`. `ldab>=kd+1`.
 * @param[out]    tau   Array of dimension (`n-kd`). The scalar factors of the
 *                      elementary reflectors (see Further Details).
 * @param[out]    work  Array of dimension (`max(1,lwork)`). On exit, if `info=0`, or
 *                      if `lwork=-1`, `work[0]` returns the size of `lwork`.
 * @param[in]     lwork The dimension of the array `work` which should be calculated
 *                      by a workspace query.
 *                      If `n<=kd+1`, `lwork>=1`, else `lwork = max(1, LWORK_QUERY)`.
 *                      If `lwork=-1`, then a workspace query is assumed; the routine
 *                      only calculates the optimal size of the `work` array, returns
 *                      this value as the first entry of the `work` array.
 *                      `LWORK_QUERY = n*kd + n*max(kd,FACTOPTNB) + 2*kd*kd` where
 *                      FACTOPTNB is the blocking used by the QR or LQ algorithm,
 *                      usually FACTOPTNB=128 is a good choice otherwise putting
 *                      `lwork=-1` will provide the size of `work`.
 * @param[out]    info  `info=0`: successful exit
 *                      `info<0`: if `info=-i`, the i-th argument had an illegal value
 *
 * @par Further Details:
 * @rst
 * Implemented by Azzam Haidar.
 *
 * All details are available on technical report, SC11, SC13 papers.
 *
 * Azzam Haidar, Hatem Ltaief, and Jack Dongarra.
 * Parallel reduction to condensed forms for symmetric eigenvalue problems
 * using aggregated fine-grained and memory-aware kernels. In Proceedings
 * of 2011 International Conference for High Performance Computing,
 * Networking, Storage and Analysis (SC '11), New York, NY, USA,
 * Article 8 , 11 pages.
 * https://doi.org/10.1145/2063384.2063394
 *
 * A. Haidar, J. Kurzak, P. Luszczek, 2013.
 * An improved parallel singular value algorithm and its implementation
 * for multicore hardware, In Proceedings of 2013 International Conference
 * for High Performance Computing, Networking, Storage and Analysis (SC '13).
 * Denver, Colorado, USA, 2013.
 * Article 90, 12 pages.
 * https://doi.org/10.1145/2503210.2503292
 *
 * A. Haidar, R. Solca, S. Tomov, T. Schulthess and J. Dongarra.
 * A novel hybrid CPU-GPU generalized eigensolver for electronic structure
 * calculations based on fine-grained memory aware tasks.
 * International Journal of High Performance Computing Applications.
 * Volume 28 Issue 2, Pages 196-209, May 2014.
 * https://doi.org/10.1177/1094342013502097
 *
 * If ``uplo`` = 'U', the matrix Q is represented as a product of elementary
 * reflectors
 *
 * .. code-block:: text
 *
 *     Q = H(k-1)**T . . . H(1)**T H(0)**T, where k = n-kd.
 *
 *     Each H(i) has the form
 *
 *     H(i) = I - tau * v * v**T
 *
 *     where tau is a real scalar, and v is a real vector with
 *     v[0:i+kd-1] = 0 and v[i+kd] = 1; conjg(v[i+kd+1:n-1]) is stored on
 *     exit in A[i, i+kd+1:n-1], and tau in tau[i].
 *
 * If ``uplo`` = 'L', the matrix Q is represented as a product of elementary
 * reflectors
 *
 * .. code-block:: text
 *
 *     Q = H(0) H(1) . . . H(k-1), where k = n-kd.
 *
 *     Each H(i) has the form
 *
 *     H(i) = I - tau * v * v**T
 *
 *     where tau is a real scalar, and v is a real vector with
 *     v[kd:i] = 0 and v[i+kd+1] = 1; v[i+kd+2:n-1] is stored on exit in
 *     A[i+kd+2:n-1, i], and tau in tau[i].
 *
 * The contents of A on exit are illustrated by the following examples
 * with n = 5:
 *
 * .. code-block:: text
 *
 *     if UPLO = 'U':                       if UPLO = 'L':
 *
 *       (  ab  ab/v0  v0     v0     v0    )  (  ab                            )
 *       (      ab     ab/v1  v1     v1    )  (  ab/v0  ab                     )
 *       (             ab     ab/v2  v2    )  (  v0     ab/v1  ab              )
 *       (                    ab     ab/v3 )  (  v0     v1     ab/v2  ab       )
 *       (                           ab    )  (  v0     v1     v2     ab/v3 ab )
 *
 *     where d and e denote diagonal and off-diagonal elements of T, and vi
 *     denotes an element of the vector defining H(i).
 * @endrst
 */
void ssytrd_sy2sb(const char* uplo, const INT n, const INT kd,
                  f32* A, const INT lda,
                  f32* AB, const INT ldab,
                  f32* tau,
                  f32* work, const INT lwork, INT* info)
{
    const f32 rone = 1.0f;
    const f32 zero = 0.0f;
    const f32 one = 1.0f;
    const f32 half = 0.5f;

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
