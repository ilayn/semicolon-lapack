/**
 * @file zgesvdq.c
 * @brief ZGESVDQ computes SVD with a QR-Preconditioned QR SVD Method.
 */

#include "semicolon_lapack_complex_double.h"
#include "lapack_tuning.h"
#include <complex.h>
#include <math.h>
#include "semicolon_cblas.h"

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
/**
 * ZGESVDQ computes the singular value decomposition (SVD) of a complex
 * `m`-by-`n` matrix `A`, where `m>=n`. The SVD of `A` is written as
 * @rst
 * .. code-block:: text
 *
 *                                    [++]   [xx]   [x0]   [xx]
 *              A = U * SIGMA * V^*,  [++] = [xx] * [ox] * [xx]
 *                                    [++]   [xx]
 * @endrst
 * where SIGMA is an `n`-by-`n` diagonal matrix, U is an `m`-by-`n` orthonormal
 * matrix, and V is an `n`-by-`n` unitary matrix. The diagonal elements
 * of SIGMA are the singular values of `A`. The columns of U and V are the
 * left and the right singular vectors of `A`, respectively.
 *
 * @param[in]     joba  Specifies the level of accuracy in the computed SVD.
 *                      `joba='A'`: The requested accuracy corresponds to having the
 *                      backward error bounded by `|| delta A ||_F <= f(m,n)*EPS*|| A ||_F`,
 *                      where `EPS = zlamch("Epsilon")`. This authorises ZGESVDQ to
 *                      truncate the computed triangular factor in a rank revealing
 *                      QR factorization whenever the truncated part is below the
 *                      threshold of the order of `EPS * ||A||_F`. This is aggressive
 *                      truncation level.
 *                      `joba='M'`: Similarly as with `'A'`, but the truncation is more
 *                      gentle: it is allowed only when there is a drop on the diagonal
 *                      of the triangular factor in the QR factorization. This is medium
 *                      truncation level.
 *                      `joba='H'`: High accuracy requested. No numerical rank
 *                      determination based on the rank revealing QR factorization is
 *                      attempted.
 *                      `joba='E'`: Same as `'H'`, and in addition the condition number
 *                      of column scaled `A` is estimated and returned in `rwork[0]`.
 *                      `N^(-1/4)*rwork[0] <= ||pinv(A_scaled)||_2 <= N^(1/4)*rwork[0]`.
 * @param[in]     jobp  `jobp='P'`: The rows of `A` are ordered in decreasing order with
 *                      respect to `||A(i,:)||_\infty`. This enhances numerical accuracy
 *                      at the cost of extra data movement. Recommended for numerical
 *                      robustness.
 *                      `jobp='N'`: No row pivoting.
 * @param[in]     jobr  `jobr='T'`: After the initial pivoted QR factorization, ZGESVD is
 *                      applied to the transposed `R**H` of the computed triangular factor
 *                      R. This involves some extra data movement (matrix transpositions).
 *                      Useful for experiments, research and development.
 *                      `jobr='N'`: The triangular factor R is given as input to ZGESVD.
 *                      This may be preferred as it involves less data movement.
 * @param[in]     jobu  `jobu='A'`: All `m` left singular vectors are computed and returned
 *                      in the matrix `U`. See the description of `U`.
 *                      `jobu='S'` or `jobu='U'`: `n = min(m,n)` left singular vectors are
 *                      computed and returned in the matrix `U`. See the description of `U`.
 *                      `jobu='R'`: Numerical rank `numrank` is determined and only
 *                      `numrank` left singular vectors are computed and returned in the
 *                      matrix `U`.
 *                      `jobu='F'`: The `n` left singular vectors are returned in factored
 *                      form as the product of the Q factor from the initial QR
 *                      factorization and the `n` left singular vectors of `(R**H, 0)**H`.
 *                      If row pivoting is used, then the necessary information on the row
 *                      pivoting is stored in `iwork[n:n+m-2]`.
 *                      `jobu='N'`: The left singular vectors are not computed.
 * @param[in]     jobv  `jobv='A'` or `jobv='V'`: All `n` right singular vectors are
 *                      computed and returned in the matrix `V`.
 *                      `jobv='R'`: Numerical rank `numrank` is determined and only
 *                      `numrank` right singular vectors are computed and returned in the
 *                      matrix `V`. This option is allowed only if `jobu='R'` or `jobu='N'`;
 *                      otherwise it is illegal.
 *                      `jobv='N'`: The right singular vectors are not computed.
 * @param[in]     m     The number of rows of the input matrix `A`. `m>=0`.
 * @param[in]     n     The number of columns of the input matrix `A`. `m>=n>=0`.
 * @param[in,out] A     Array of dimensions `lda` x `n`. On entry, the input matrix `A`.
 *                      On exit, if `jobu!='N'` or `jobv!='N'`, the lower triangle of `A`
 *                      contains the Householder vectors as stored by ZGEQP3. If
 *                      `jobu='F'`, these Householder vectors together with `cwork[0:n-1]`
 *                      can be used to restore the Q factors from the initial pivoted QR
 *                      factorization of `A`. See the description of `U`.
 * @param[in]     lda   The leading dimension of the array `A`. `lda>=max(1,m)`.
 * @param[out]    S     Array of dimension `n`. The singular values of `A`, ordered so
 *                      that `S[i]>=S[i+1]`.
 * @param[out]    U     Array of dimension `ldu` x `m` if `jobu='A'`; see the description
 *                      of `ldu`. In this case, on exit, `U` contains the `m` left singular
 *                      vectors.
 *                      `ldu` x `n` if `jobu='S'`, `'U'`, `'R'`; see the description of
 *                      `ldu`. In this case, `U` contains the leading `n` or the leading
 *                      `numrank` left singular vectors.
 *                      `ldu` x `n` if `jobu='F'`; see the description of `ldu`. In this
 *                      case `U` contains `n` x `n` unitary matrix that can be used to
 *                      form the left singular vectors.
 *                      If `jobu='N'`, `U` is not referenced.
 * @param[in]     ldu   The leading dimension of the array `U`.
 *                      If `jobu='A'`, `'S'`, `'U'`, `'R'`, `ldu>=max(1,m)`.
 *                      If `jobu='F'`, `ldu>=max(1,n)`.
 *                      Otherwise, `ldu>=1`.
 * @param[out]    V     Array of dimension `ldv` x `n` if `jobv='A'`, `'V'`, `'R'` or if
 *                      `joba='E'`.
 *                      If `jobv='A'` or `'V'`, `V` contains the `n`-by-`n` unitary
 *                      matrix `V**H`;
 *                      If `jobv='R'`, `V` contains the first `numrank` rows of `V**H`
 *                      (the right singular vectors, stored rowwise, of the `numrank`
 *                      largest singular values).
 *                      If `jobv='N'` and `joba='E'`, `V` is used as a workspace.
 *                      If `jobv='N'` and `joba!='E'`, `V` is not referenced.
 * @param[in]     ldv   The leading dimension of the array `V`.
 *                      If `jobv='A'`, `'V'`, `'R'`, or `joba='E'`, `ldv>=max(1,n)`.
 *                      Otherwise, `ldv>=1`.
 * @param[out]    numrank `numrank` is the numerical rank first determined after the rank
 *                      revealing QR factorization, following the strategy specified by the
 *                      value of `joba`. If `jobv='R'` and `jobu='R'`, only `numrank`
 *                      leading singular values and vectors are then requested in the call
 *                      of ZGESVD. The final value of `numrank` might be further reduced if
 *                      some singular values are computed as zeros.
 * @param[out]    iwork Integer array of dimension (`max(1,liwork)`).
 *                      On exit, `iwork[0:n-1]` contains column pivoting permutation of the
 *                      rank revealing QR factorization.
 *                      If `jobp='P'`, `iwork[n:n+m-2]` contains the indices of the sequence
 *                      of row swaps used in row pivoting. These can be used to restore the
 *                      left singular vectors in the case `jobu='F'`.
 *                      If `liwork`, `lcwork`, or `lrwork = -1`, then on exit, if `info=0`,
 *                      `iwork[0]` returns the minimal `liwork`.
 * @param[in]     liwork The dimension of the array `iwork`.
 *                      `liwork >= n+m-1`, if `jobp='P'` and `joba!='E'`;
 *                      `liwork >= n`, if `jobp='N'` and `joba!='E'`;
 *                      `liwork >= n+m-1+n`, if `jobp='P'` and `joba='E'`;
 *                      `liwork >= n+n`, if `jobp='N'` and `joba='E'`.
 *                      If `liwork=-1`, then a workspace query is assumed; the routine
 *                      only calculates and returns the optimal and minimal sizes
 *                      for the `cwork`, `iwork`, and `rwork` arrays, and no error
 *                      message related to `lcwork` is issued by XERBLA.
 * @param[out]    cwork Array of dimension (`max(2,lcwork)`), used as a workspace.
 *                      On exit, if, on entry, `lcwork!=-1`, `cwork[0:n-1]` contains
 *                      parameters needed to recover the Q factor from the QR factorization
 *                      computed by ZGEQP3.
 *                      If `liwork`, `lcwork`, or `lrwork = -1`, then on exit, if `info=0`,
 *                      `cwork[0]` returns the optimal `lcwork`, and
 *                      `cwork[1]` returns the minimal `lcwork`.
 * @param[in,out] lcwork The dimension of the array `cwork`. It is determined as follows:
 *                       @rst
 *                       .. code-block:: text
 *
 *                           Let  LWQP3 = N+1,  LWCON = 2*N, and let
 *                           LWUNQ = { MAX( N, 1 ),  if JOBU = 'R', 'S', or 'U'
 *                                   { MAX( M, 1 ),  if JOBU = 'A'
 *                           LWSVD = MAX( 3*N, 1 )
 *                           LWLQF = MAX( N/2, 1 ), LWSVD2 = MAX( 3*(N/2), 1 ), LWUNLQ = MAX( N, 1 ),
 *                           LWQRF = MAX( N/2, 1 ), LWUNQ2 = MAX( N, 1 )
 *                           Then the minimal value of LCWORK is:
 *                           = MAX( N + LWQP3, LWSVD )        if only the singular values are needed;
 *                           = MAX( N + LWQP3, LWCON, LWSVD ) if only the singular values are needed,
 *                                                    and a scaled condition estimate requested;
 *
 *                           = N + MAX( LWQP3, LWSVD, LWUNQ ) if the singular values and the left
 *                                                    singular vectors are requested;
 *                           = N + MAX( LWQP3, LWCON, LWSVD, LWUNQ ) if the singular values and the left
 *                                                    singular vectors are requested, and also
 *                                                    a scaled condition estimate requested;
 *
 *                           = N + MAX( LWQP3, LWSVD )        if the singular values and the right
 *                                                    singular vectors are requested;
 *                           = N + MAX( LWQP3, LWCON, LWSVD ) if the singular values and the right
 *                                                    singular vectors are requested, and also
 *                                                    a scaled condition estimate requested;
 *
 *                           = N + MAX( LWQP3, LWSVD, LWUNQ ) if the full SVD is requested with JOBV = 'R';
 *                                                    independent of JOBR;
 *                           = N + MAX( LWQP3, LWCON, LWSVD, LWUNQ ) if the full SVD is requested,
 *                                                    JOBV = 'R' and, also a scaled condition
 *                                                    estimate requested; independent of JOBR;
 *                           = MAX( N + MAX( LWQP3, LWSVD, LWUNQ ),
 *                          N + MAX( LWQP3, N/2+LWLQF, N/2+LWSVD2, N/2+LWUNLQ, LWUNQ) ) if the
 *                                          full SVD is requested with JOBV = 'A' or 'V', and
 *                                          JOBR ='N'
 *                           = MAX( N + MAX( LWQP3, LWCON, LWSVD, LWUNQ ),
 *                          N + MAX( LWQP3, LWCON, N/2+LWLQF, N/2+LWSVD2, N/2+LWUNLQ, LWUNQ ) )
 *                                          if the full SVD is requested with JOBV = 'A' or 'V', and
 *                                          JOBR ='N', and also a scaled condition number estimate
 *                                          requested.
 *                           = MAX( N + MAX( LWQP3, LWSVD, LWUNQ ),
 *                          N + MAX( LWQP3, N/2+LWQRF, N/2+LWSVD2, N/2+LWUNQ2, LWUNQ ) ) if the
 *                                          full SVD is requested with JOBV = 'A', 'V', and JOBR ='T'
 *                           = MAX( N + MAX( LWQP3, LWCON, LWSVD, LWUNQ ),
 *                          N + MAX( LWQP3, LWCON, N/2+LWQRF, N/2+LWSVD2, N/2+LWUNQ2, LWUNQ ) )
 *                                          if the full SVD is requested with JOBV = 'A', 'V' and
 *                                          JOBR ='T', and also a scaled condition number estimate
 *                                          requested.
 *                       @endrst
 *                       Finally, `lcwork` must be at least two: `lcwork = max(2,lcwork)`.
 *                       If `lcwork=-1`, then a workspace query is assumed; the routine
 *                       only calculates and returns the optimal and minimal sizes
 *                       for the `cwork`, `iwork`, and `rwork` arrays, and no error
 *                       message related to `lcwork` is issued by XERBLA.
 * @param[out]    rwork Array of dimension (`max(1,lrwork)`). On exit,
 *                      1. If `joba='E'`, `rwork[0]` contains an estimate of the condition
 *                      number of column scaled `A`. If `A = C * D` where D is diagonal and C
 *                      has unit columns in the Euclidean norm, then, assuming full column
 *                      rank, `N^(-1/4) * rwork[0] <= ||pinv(C)||_2 <= N^(1/4) * rwork[0]`.
 *                      Otherwise, `rwork[0] = -1`.
 *                      2. `rwork[1]` contains the number of singular values computed as
 *                      exact zeros in ZGESVD applied to the upper triangular or trapezoidal
 *                      R (from the initial QR factorization). In case of early exit (no call
 *                      to ZGESVD, such as in the case of zero matrix) `rwork[1] = -1`.
 *                      If `liwork`, `lcwork`, or `lrwork = -1`, then on exit, if `info=0`,
 *                      `rwork[0]` returns the minimal `lrwork`.
 * @param[in]     lrwork The dimension of the array `rwork`.
 *                      If `jobp='P'`, then `lrwork >= max(2,m)`.
 *                      Otherwise, `lrwork >= 2`.
 *                      If `lrwork=-1`, then a workspace query is assumed; the routine
 *                      only calculates and returns the optimal and minimal sizes
 *                      for the `cwork`, `iwork`, and `rwork` arrays, and no error
 *                      message related to `lcwork` is issued by XERBLA.
 * @param[out]    info  `info=0`: successful exit.
 *                      `info<0`: if `info=-i`, the i-th argument had an illegal value.
 *                      `info>0`: if ZBDSQR did not converge, `info` specifies how many
 *                      superdiagonals of an intermediate bidiagonal form B (computed in
 *                      ZGESVD) did not converge to zero.
 *
 * @par Further Details:
 * @rst
 * 1. The data movement (matrix transpose) is coded using simple nested
 * DO-loops because BLAS and LAPACK do not provide corresponding subroutines.
 * Those DO-loops are easily identified in this source code - by the CONTINUE
 * statements labeled with 11**. In an optimized version of this code, the
 * nested DO loops should be replaced with calls to an optimized subroutine.
 *
 * 2. This code scales A by 1/SQRT(M) if the largest ABS(A(i,j)) could cause
 * column norm overflow. This is the minimal precaution and it is left to the
 * SVD routine (ZGESVD) to do its own preemptive scaling if potential over-
 * or underflows are detected. To avoid repeated scanning of the array A,
 * an optimal implementation would do all necessary scaling before calling
 * ZGESVD and the scaling in ZGESVD can be switched off.
 *
 * 3. Other comments related to code optimization are given in comments in the
 * code, enclosed in [[double brackets]].
 * @endrst
 *
 * @par Bugs, examples and comments:
 * @rst
 * Please report all bugs and send interesting examples and/or comments to
 * drmac@math.hr. Thank you.
 * @endrst
 *
 * @par References:
 * @rst
 * [1] Zlatko Drmac, Algorithm 977: A QR-Preconditioned QR SVD Method for
 * Computing the SVD with High Accuracy. ACM Trans. Math. Softw.
 * 44(1): 11:1-11:30 (2017)
 *
 * SIGMA library, xGESVDQ section updated February 2016.
 * Developed and coded by Zlatko Drmac, Department of Mathematics
 * University of Zagreb, Croatia, drmac@math.hr
 * @endrst
 *
 * @par Contributors:
 * @rst
 * Developed and coded by Zlatko Drmac, Department of Mathematics
 * University of Zagreb, Croatia, drmac@math.hr
 * @endrst
 */
void zgesvdq(const char* joba, const char* jobp, const char* jobr,
             const char* jobu, const char* jobv,
             const INT m, const INT n, c128* restrict A, const INT lda,
             f64* restrict S, c128* restrict U, const INT ldu,
             c128* restrict V, const INT ldv, INT* numrank,
             INT* restrict iwork, const INT liwork,
             c128* restrict cwork, const INT lcwork,
             f64* restrict rwork, const INT lrwork, INT* info)
{
    INT wntus, wntur, wntua, wntuf, lsvc0, lsvec, dntwu;
    INT wntvr, wntva, rsvec, dntwv;
    INT accla, acclm, acclh, conda;
    INT rowprm, rtrans, lquery;
    INT ierr, nr, n1 = 0, optratio, p, q;
    INT ascaled;

    INT lwcon, lwqp3, lwrk_zgelqf, lwrk_zgesvd, lwrk_zgesvd2,
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
            lwrk_zgeqp3 = (INT)creal(cdummy[0]);
            if (wntus || wntur) {
                zunmqr("L", "N", m, n, n, A, lda, NULL, U,
                        ldu, cdummy, -1, &ierr);
                lwrk_zunmqr = (INT)creal(cdummy[0]);
            } else if (wntua) {
                zunmqr("L", "N", m, m, n, A, lda, NULL, U,
                        ldu, cdummy, -1, &ierr);
                lwrk_zunmqr = (INT)creal(cdummy[0]);
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
                lwrk_zgesvd = (INT)creal(cdummy[0]);
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
                lwrk_zgesvd = (INT)creal(cdummy[0]);
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
                lwrk_zgesvd = (INT)creal(cdummy[0]);
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
                    lwrk_zgesvd = (INT)creal(cdummy[0]);
                    optwrk = lwrk_zgeqp3;
                    if (optwrk < lwrk_zgesvd) optwrk = lwrk_zgesvd;
                    if (optwrk < lwrk_zunmqr) optwrk = lwrk_zunmqr;
                    if (conda && optwrk < lwcon) optwrk = lwcon;
                    optwrk = n + optwrk;
                    if (wntva) {
                        zgeqrf(n, n / 2, U, ldu, NULL, cdummy, -1, &ierr);
                        lwrk_zgeqrf = (INT)creal(cdummy[0]);
                        zgesvd("S", "O", n / 2, n / 2, V, ldv, S, U,
                                ldu, NULL, 1, cdummy, -1, rdummy, &ierr);
                        lwrk_zgesvd2 = (INT)creal(cdummy[0]);
                        zunmqr("R", "C", n, n, n / 2, U, ldu, NULL,
                                V, ldv, cdummy, -1, &ierr);
                        lwrk_zunmqr2 = (INT)creal(cdummy[0]);
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
                    lwrk_zgesvd = (INT)creal(cdummy[0]);
                    optwrk = lwrk_zgeqp3;
                    if (optwrk < lwrk_zgesvd) optwrk = lwrk_zgesvd;
                    if (optwrk < lwrk_zunmqr) optwrk = lwrk_zunmqr;
                    if (conda && optwrk < lwcon) optwrk = lwcon;
                    optwrk = n + optwrk;
                    if (wntva) {
                        zgelqf(n / 2, n, U, ldu, NULL, cdummy, -1, &ierr);
                        lwrk_zgelqf = (INT)creal(cdummy[0]);
                        zgesvd("S", "O", n / 2, n / 2, V, ldv, S, U,
                                ldu, NULL, 1, cdummy, -1, rdummy, &ierr);
                        lwrk_zgesvd2 = (INT)creal(cdummy[0]);
                        zunmlq("R", "N", n, n, n / 2, U, ldu, NULL,
                                V, ldv, cdummy, -1, &ierr);
                        lwrk_zunmlq = (INT)creal(cdummy[0]);
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

            INT minmn = (n < nr) ? n : nr;
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
