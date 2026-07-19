/**
 * @file chetrd_2stage.c
 * @brief CHETRD_2STAGE reduces a complex Hermitian matrix to tridiagonal form using 2-stage algorithm.
 */

#include "semicolon_lapack_complex_single.h"
#include <complex.h>

/**
 * CHETRD_2STAGE reduces a complex Hermitian matrix `A` to real symmetric
 * tridiagonal form T by a unitary similarity transformation:
 * `Q1**H Q2**H * A * Q2 * Q1 = T`.
 *
 * @param[in]     vect   `'N'`: No need for the Householder representation, in
 *                              particular for the second stage (Band to tridiagonal)
 *                              and thus `lhous2` is of size `max(1,4*n)`;
 *                       `'V'`: the Householder representation is needed to either
 *                              generate Q1 Q2 or to apply Q1 Q2, then `lhous2` is to
 *                              be queried and computed. (NOT AVAILABLE IN THIS RELEASE).
 * @param[in]     uplo   `'U'`: Upper triangle of `A` is stored;
 *                       `'L'`: Lower triangle of `A` is stored.
 * @param[in]     n      The order of the matrix `A`. `n>=0`.
 * @param[in,out] A      On entry, the Hermitian matrix `A`. If `uplo='U'`, the leading
 *                       `n`-by-`n` upper triangular part of `A` contains the upper
 *                       triangular part of the matrix `A`, and the strictly lower
 *                       triangular part of `A` is not referenced. If `uplo='L'`, the
 *                       leading `n`-by-`n` lower triangular part of `A` contains the lower
 *                       triangular part of the matrix `A`, and the strictly upper
 *                       triangular part of `A` is not referenced.
 *                       On exit, if `uplo='U'`, the band superdiagonal of `A` are
 *                       overwritten by the corresponding elements of the internal
 *                       band-diagonal matrix AB, and the elements above the KD
 *                       superdiagonal, with the array `tau`, represent the unitary
 *                       matrix Q1 as a product of elementary reflectors; if `uplo='L'`,
 *                       the diagonal and band subdiagonal of `A` are overwritten by the
 *                       corresponding elements of the internal band-diagonal matrix AB,
 *                       and the elements below the KD subdiagonal, with the array `tau`,
 *                       represent the unitary matrix Q1 as a product of elementary
 *                       reflectors. See Further Details.
 * @param[in]     lda    The leading dimension of the array `A`. `lda>=max(1,n)`.
 * @param[out]    D      Array of dimension (`n`). The diagonal elements of the
 *                       tridiagonal matrix T.
 * @param[out]    E      Array of dimension (`n-1`). The off-diagonal elements of the
 *                       tridiagonal matrix T.
 * @param[out]    tau    Array of dimension (`n-kd`). The scalar factors of the
 *                       elementary reflectors of the first stage (see Further Details).
 * @param[out]    hous2  Array of dimension (`max(1,lhous2)`). Stores the Householder
 *                       representation of the stage2 band to tridiagonal.
 * @param[in]     lhous2 The dimension of the array `hous2`. `lhous2>=1`.
 *                       If `lwork=-1`, or `lhous2=-1`, then a query is assumed; the
 *                       routine only calculates the optimal size of the `hous2` array,
 *                       returns this value as the first entry of the `hous2` array.
 *                       If `vect='N'`, `lhous2 = max(1,4*n)`; if `vect='V'`, option
 *                       not yet available.
 * @param[out]    work   Array of dimension (`max(1,lwork)`). On exit, if `info=0`,
 *                       `work[0]` returns the optimal `lwork`.
 * @param[in]     lwork  The dimension of the array `work`.
 *                       If `n=0`, `lwork>=1`, else `lwork = max(1, dimension)`.
 *                       If `lwork=-1`, or `lhous2=-1`, then a workspace query is
 *                       assumed; the routine only calculates the optimal size of the
 *                       `work` array, returns this value as the first entry of the
 *                       `work` array.
 *                       @rst
 *                       .. code-block:: text
 *
 *                           lwork = MAX(1, dimension) where
 *                           dimension = max(stage1,stage2) + (kd+1)*n
 *                                     = n*kd + n*max(kd+1,FACTOPTNB)
 *                                       + max(2*kd*kd, kd*NTHREADS)
 *                                       + (kd+1)*n
 *                       @endrst
 *                       where `kd` is the blocking size of the reduction,
 *                       FACTOPTNB is the blocking used by the QR or LQ
 *                       algorithm, usually FACTOPTNB=128 is a good choice,
 *                       NTHREADS is the number of threads used when openMP
 *                       compilation is enabled, otherwise =1.
 * @param[out]    info
 *                       - `info=0`: successful exit
 *                       - `info<0`: if `info=-i`, the i-th argument had an illegal value
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
 * @endrst
 */
void chetrd_2stage(const char* vect, const char* uplo, const INT n,
                   c64* A, const INT lda,
                   f32* D, f32* E, c64* tau,
                   c64* hous2, const INT lhous2,
                   c64* work, const INT lwork, INT* info)
{
    INT lquery, upper;
    INT kd, ib, lwmin, lhmin, lwrk, ldab, wpos, abpos;

    *info = 0;
    (void)(vect[0] == 'V' || vect[0] == 'v');  /* wantq set but not used */
    upper = (uplo[0] == 'U' || uplo[0] == 'u');
    lquery = (lwork == -1) || (lhous2 == -1);

    kd = ilaenv2stage(1, "CHETRD_2STAGE", vect, n, -1, -1, -1);
    ib = ilaenv2stage(2, "CHETRD_2STAGE", vect, n, kd, -1, -1);
    if (n == 0) {
        lhmin = 1;
        lwmin = 1;
    } else {
        lhmin = ilaenv2stage(3, "CHETRD_2STAGE", vect, n, kd, ib, -1);
        lwmin = ilaenv2stage(4, "CHETRD_2STAGE", vect, n, kd, ib, -1);
    }

    if (!(vect[0] == 'N' || vect[0] == 'n')) {
        *info = -1;
    } else if (!upper && !(uplo[0] == 'L' || uplo[0] == 'l')) {
        *info = -2;
    } else if (n < 0) {
        *info = -3;
    } else if (lda < ((1 > n) ? 1 : n)) {
        *info = -5;
    } else if (lhous2 < lhmin && !lquery) {
        *info = -10;
    } else if (lwork < lwmin && !lquery) {
        *info = -12;
    }

    if (*info == 0) {
        hous2[0] = CMPLXF((f32)lhmin, 0.0f);
        work[0] = CMPLXF((f32)lwmin, 0.0f);
    }

    if (*info != 0) {
        xerbla("CHETRD_2STAGE", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    if (n == 0) {
        work[0] = CMPLXF(1.0f, 0.0f);
        return;
    }

    ldab = kd + 1;
    lwrk = lwork - ldab * n;
    abpos = 0;
    wpos = abpos + ldab * n;

    chetrd_he2hb(uplo, n, kd, A, lda, &work[abpos], ldab,
                 tau, &work[wpos], lwrk, info);
    if (*info != 0) {
        xerbla("CHETRD_HE2HB", -(*info));
        return;
    }

    chetrd_hb2st("Y", vect, uplo, n, kd,
                 &work[abpos], ldab, D, E,
                 hous2, lhous2, &work[wpos], lwrk, info);
    if (*info != 0) {
        xerbla("CHETRD_HB2ST", -(*info));
        return;
    }

    work[0] = CMPLXF((f32)lwmin, 0.0f);
}
