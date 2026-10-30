/**
 * @file sgesvdq.c
 * @brief SGESVDQ computes SVD with a QR-Preconditioned QR SVD Method.
 */

#include "semicolon_lapack_single.h"
#include "lapack_tuning.h"
#include <math.h>
#include "semicolon_cblas.h"

static const f32 ZERO = 0.0f;
static const f32 ONE = 1.0f;

/**
 * SGESVDQ computes the singular value decomposition (SVD) of a real
 * M-by-N matrix A, where M >= N. The SVD of A is written as
 *
 *              A = U * SIGMA * V**T
 *
 * where SIGMA is an N-by-N diagonal matrix, U is an M-by-N orthonormal
 * matrix, and V is an N-by-N orthogonal matrix.
 */
/**
 * SGESVDQ computes the singular value decomposition (SVD) of a real
 * `m`-by-`n` matrix `A`, where `m>=n`. The SVD of `A` is written as
 * @rst
 * .. code-block:: text
 *
 *                                    [++]   [xx]   [x0]   [xx]
 *              A = U * SIGMA * V^*,  [++] = [xx] * [ox] * [xx]
 *                                    [++]   [xx]
 * @endrst
 * where SIGMA is an `n`-by-`n` diagonal matrix, U is an `m`-by-`n` orthonormal
 * matrix, and V is an `n`-by-`n` orthogonal matrix. The diagonal elements
 * of SIGMA are the singular values of `A`. The columns of U and V are the
 * left and the right singular vectors of `A`, respectively.
 *
 * @param[in]     joba  Specifies the level of accuracy in the computed SVD.
 *                      `joba='A'`: The requested accuracy corresponds to having the
 *                      backward error bounded by `|| delta A ||_F <= f(m,n)*EPS*|| A ||_F`,
 *                      where `EPS = slamch("Epsilon")`. This authorises SGESVDQ to
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
 * @param[in]     jobr  `jobr='T'`: After the initial pivoted QR factorization, SGESVD is
 *                      applied to the transposed `R**T` of the computed triangular factor
 *                      R. This involves some extra data movement (matrix transpositions).
 *                      Useful for experiments, research and development.
 *                      `jobr='N'`: The triangular factor R is given as input to SGESVD.
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
 *                      factorization and the `n` left singular vectors of `(R**T, 0)**T`.
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
 *                      contains the Householder vectors as stored by SGEQP3. If
 *                      `jobu='F'`, these Householder vectors together with `work[0:n-1]`
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
 *                      case `U` contains `n` x `n` orthogonal matrix that can be used to
 *                      form the left singular vectors.
 *                      If `jobu='N'`, `U` is not referenced.
 * @param[in]     ldu   The leading dimension of the array `U`.
 *                      If `jobu='A'`, `'S'`, `'U'`, `'R'`, `ldu>=max(1,m)`.
 *                      If `jobu='F'`, `ldu>=max(1,n)`.
 *                      Otherwise, `ldu>=1`.
 * @param[out]    V     Array of dimension `ldv` x `n` if `jobv='A'`, `'V'`, `'R'` or if
 *                      `joba='E'`.
 *                      If `jobv='A'` or `'V'`, `V` contains the `n`-by-`n` orthogonal
 *                      matrix `V**T`;
 *                      If `jobv='R'`, `V` contains the first `numrank` rows of `V**T`
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
 *                      of SGESVD. The final value of `numrank` might be further reduced if
 *                      some singular values are computed as zeros.
 * @param[out]    iwork Integer array of dimension (`max(1,liwork)`).
 *                      On exit, `iwork[0:n-1]` contains column pivoting permutation of the
 *                      rank revealing QR factorization.
 *                      If `jobp='P'`, `iwork[n:n+m-2]` contains the indices of the sequence
 *                      of row swaps used in row pivoting. These can be used to restore the
 *                      left singular vectors in the case `jobu='F'`.
 *                      If `liwork`, `lwork`, or `lrwork = -1`, then on exit, if `info=0`,
 *                      `iwork[0]` returns the minimal `liwork`.
 * @param[in]     liwork The dimension of the array `iwork`.
 *                      `liwork >= n+m-1`, if `jobp='P'` and `joba!='E'`;
 *                      `liwork >= n`, if `jobp='N'` and `joba!='E'`;
 *                      `liwork >= n+m-1+n`, if `jobp='P'` and `joba='E'`;
 *                      `liwork >= n+n`, if `jobp='N'` and `joba='E'`.
 *                      If `liwork=-1`, then a workspace query is assumed; the routine
 *                      only calculates and returns the optimal and minimal sizes
 *                      for the `work`, `iwork`, and `rwork` arrays, and no error
 *                      message related to `lwork` is issued by XERBLA.
 * @param[out]    work  Array of dimension (`max(2,lwork)`), used as a workspace.
 *                      On exit, if, on entry, `lwork!=-1`, `work[0:n-1]` contains
 *                      parameters needed to recover the Q factor from the QR factorization
 *                      computed by SGEQP3.
 *                      If `liwork`, `lwork`, or `lrwork = -1`, then on exit, if `info=0`,
 *                      `work[0]` returns the optimal `lwork`, and
 *                      `work[1]` returns the minimal `lwork`.
 * @param[in,out] lwork The dimension of the array `work`. It is determined as follows:
 *                       @rst
 *                       .. code-block:: text
 *
 *                           Let  LWQP3 = 3*N+1,  LWCON = 3*N, and let
 *                           LWORQ = { MAX( N, 1 ),  if JOBU = 'R', 'S', or 'U'
 *                                   { MAX( M, 1 ),  if JOBU = 'A'
 *                           LWSVD = MAX( 5*N, 1 )
 *                           LWLQF = MAX( N/2, 1 ), LWSVD2 = MAX( 5*(N/2), 1 ), LWORLQ = MAX( N, 1 ),
 *                           LWQRF = MAX( N/2, 1 ), LWORQ2 = MAX( N, 1 )
 *                           Then the minimal value of LWORK is:
 *                           = MAX( N + LWQP3, LWSVD )        if only the singular values are needed;
 *                           = MAX( N + LWQP3, LWCON, LWSVD ) if only the singular values are needed,
 *                                                    and a scaled condition estimate requested;
 *
 *                           = N + MAX( LWQP3, LWSVD, LWORQ ) if the singular values and the left
 *                                                    singular vectors are requested;
 *                           = N + MAX( LWQP3, LWCON, LWSVD, LWORQ ) if the singular values and the left
 *                                                    singular vectors are requested, and also
 *                                                    a scaled condition estimate requested;
 *
 *                           = N + MAX( LWQP3, LWSVD )        if the singular values and the right
 *                                                    singular vectors are requested;
 *                           = N + MAX( LWQP3, LWCON, LWSVD ) if the singular values and the right
 *                                                    singular vectors are requested, and also
 *                                                    a scaled condition estimate requested;
 *
 *                           = N + MAX( LWQP3, LWSVD, LWORQ ) if the full SVD is requested with JOBV = 'R';
 *                                                    independent of JOBR;
 *                           = N + MAX( LWQP3, LWCON, LWSVD, LWORQ ) if the full SVD is requested,
 *                                                    JOBV = 'R' and, also a scaled condition
 *                                                    estimate requested; independent of JOBR;
 *                           = MAX( N + MAX( LWQP3, LWSVD, LWORQ ),
 *                          N + MAX( LWQP3, N/2+LWLQF, N/2+LWSVD2, N/2+LWORLQ, LWORQ) ) if the
 *                                          full SVD is requested with JOBV = 'A' or 'V', and
 *                                          JOBR ='N'
 *                           = MAX( N + MAX( LWQP3, LWCON, LWSVD, LWORQ ),
 *                          N + MAX( LWQP3, LWCON, N/2+LWLQF, N/2+LWSVD2, N/2+LWORLQ, LWORQ ) )
 *                                          if the full SVD is requested with JOBV = 'A' or 'V', and
 *                                          JOBR ='N', and also a scaled condition number estimate
 *                                          requested.
 *                           = MAX( N + MAX( LWQP3, LWSVD, LWORQ ),
 *                          N + MAX( LWQP3, N/2+LWQRF, N/2+LWSVD2, N/2+LWORQ2, LWORQ ) ) if the
 *                                          full SVD is requested with JOBV = 'A', 'V', and JOBR ='T'
 *                           = MAX( N + MAX( LWQP3, LWCON, LWSVD, LWORQ ),
 *                          N + MAX( LWQP3, LWCON, N/2+LWQRF, N/2+LWSVD2, N/2+LWORQ2, LWORQ ) )
 *                                          if the full SVD is requested with JOBV = 'A' or 'V', and
 *                                          JOBR ='T', and also a scaled condition number estimate
 *                                          requested.
 *                       @endrst
 *                       Finally, `lwork` must be at least two: `lwork = max(2,lwork)`.
 *                       If `lwork=-1`, then a workspace query is assumed; the routine
 *                       only calculates and returns the optimal and minimal sizes
 *                       for the `work`, `iwork`, and `rwork` arrays, and no error
 *                       message related to `lwork` is issued by XERBLA.
 * @param[out]    rwork Array of dimension (`max(1,lrwork)`). On exit,
 *                      1. If `joba='E'`, `rwork[0]` contains an estimate of the condition
 *                      number of column scaled `A`. If `A = C * D` where D is diagonal and C
 *                      has unit columns in the Euclidean norm, then, assuming full column
 *                      rank, `N^(-1/4) * rwork[0] <= ||pinv(C)||_2 <= N^(1/4) * rwork[0]`.
 *                      Otherwise, `rwork[0] = -1`.
 *                      2. `rwork[1]` contains the number of singular values computed as
 *                      exact zeros in SGESVD applied to the upper triangular or trapezoidal
 *                      R (from the initial QR factorization). In case of early exit (no call
 *                      to SGESVD, such as in the case of zero matrix) `rwork[1] = -1`.
 *                      If `liwork`, `lwork`, or `lrwork = -1`, then on exit, if `info=0`,
 *                      `rwork[0]` returns the minimal `lrwork`.
 * @param[in]     lrwork The dimension of the array `rwork`.
 *                      If `jobp='P'`, then `lrwork >= max(2,m)`.
 *                      Otherwise, `lrwork >= 2`.
 *                      If `lrwork=-1`, then a workspace query is assumed; the routine
 *                      only calculates and returns the optimal and minimal sizes
 *                      for the `work`, `iwork`, and `rwork` arrays, and no error
 *                      message related to `lwork` is issued by XERBLA.
 * @param[out]    info  `info=0`: successful exit.
 *                      `info<0`: if `info=-i`, the i-th argument had an illegal value.
 *                      `info>0`: if SBDSQR did not converge, `info` specifies how many
 *                      superdiagonals of an intermediate bidiagonal form B (computed in
 *                      SGESVD) did not converge to zero.
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
 * SVD routine (SGESVD) to do its own preemptive scaling if potential over-
 * or underflows are detected. To avoid repeated scanning of the array A,
 * an optimal implementation would do all necessary scaling before calling
 * SGESVD and the scaling in SGESVD can be switched off.
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
void sgesvdq(const char* joba, const char* jobp, const char* jobr,
             const char* jobu, const char* jobv,
             const INT m, const INT n, f32* restrict A, const INT lda,
             f32* restrict S, f32* restrict U, const INT ldu,
             f32* restrict V, const INT ldv, INT* numrank,
             INT* restrict iwork, const INT liwork,
             f32* restrict work, const INT lwork,
             f32* restrict rwork, const INT lrwork, INT* info)
{
    INT wntus, wntur, wntua, wntuf, lsvc0, lsvec, dntwu;
    INT wntvr, wntva, rsvec, dntwv;
    INT accla, acclm, acclh, conda;
    INT rowprm, rtrans, lquery;
    INT lwqp3, lwcon, lwsvd, lworq, minwrk, optwrk;
    INT iminwrk, rminwrk;
    INT ierr, iwoff, nr, n1, optratio, p, q;
    INT ascaled;
    f32 big, epsln, sconda, sfmin, rtmp;

    /* Test the input arguments */
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

    /*
     * .. compute the minimal and the optimal workspace lengths
     * [[The expressions for computing the minimal and the optimal
     * values of LWORK are written with a lot of redundancy and
     * can be simplified. However, this detailed form is easier for
     * maintenance and modifications of the code.]]
     */
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
        /*
         * .. minimal and optimal sizes of the workspace if
         * only the singular values are requested
         */
        minwrk = n + lwqp3;
        if (conda && minwrk < lwcon) minwrk = lwcon;
        if (minwrk < lwsvd) minwrk = lwsvd;
    } else if (lsvec && !rsvec) {
        /*
         * .. minimal and optimal sizes of the workspace if the
         * singular values and the left singular vectors are requested
         */
        INT t = lwqp3;
        if (t < lwsvd) t = lwsvd;
        if (t < lworq) t = lworq;
        if (conda && t < lwcon) t = lwcon;
        minwrk = n + t;
    } else if (rsvec && !lsvec) {
        /*
         * .. minimal and optimal sizes of the workspace if the
         * singular values and the right singular vectors are requested
         */
        INT t = lwqp3;
        if (t < lwsvd) t = lwsvd;
        if (conda && t < lwcon) t = lwcon;
        minwrk = n + t;
    } else {
        /*
         * .. minimal and optimal sizes of the workspace if the
         * full SVD is requested
         */
        INT t = lwqp3;
        if (t < lwsvd) t = lwsvd;
        if (t < lworq) t = lworq;
        if (conda && t < lwcon) t = lwcon;
        minwrk = n + t;
        if (wntva) {
            const INT half = n / 2;
            INT lwsvd2 = 5 * half;
            if (lwsvd2 < 1) lwsvd2 = 1;
            INT minwrk2;
            if (rtrans) {
                /* .. minimal workspace length for N x N/2 SGEQRF */
                INT lwqrf = (half > 1) ? half : 1;
                INT lworq2 = (n > 1) ? n : 1;
                minwrk2 = lwqp3;
                if (minwrk2 < half + lwqrf) minwrk2 = half + lwqrf;
                if (minwrk2 < half + lwsvd2) minwrk2 = half + lwsvd2;
                if (minwrk2 < half + lworq2) minwrk2 = half + lworq2;
                if (minwrk2 < lworq) minwrk2 = lworq;
            } else {
                /* .. minimal workspace length for N/2 x N SGELQF */
                INT lwlqf = (half > 1) ? half : 1;
                INT lworlq = (n > 1) ? n : 1;
                minwrk2 = lwqp3;
                if (minwrk2 < half + lwlqf) minwrk2 = half + lwlqf;
                if (minwrk2 < half + lwsvd2) minwrk2 = half + lwsvd2;
                if (minwrk2 < half + lworlq) minwrk2 = half + lworlq;
                if (minwrk2 < lworq) minwrk2 = lworq;
            }
            if (conda && minwrk2 < lwcon) minwrk2 = lwcon;
            minwrk2 = n + minwrk2;
            if (minwrk < minwrk2) minwrk = minwrk2;
        }
    }
    if (minwrk < 2) minwrk = 2;
    optwrk = minwrk;

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
        xerbla("SGESVDQ", -(*info));
        return;
    }

    if (lquery) {
        iwork[0] = iminwrk;
        work[0] = (f32)optwrk;
        work[1] = (f32)minwrk;
        rwork[0] = (f32)rminwrk;
        return;
    }

    /* Quick return if the matrix is void. */
    if (m == 0 || n == 0) {
        *numrank = 0;
        return;
    }

    big = slamch("O");
    epsln = slamch("E");
    sfmin = slamch("S");

    ascaled = 0;
    iwoff = 1;
    sconda = -ONE;
    n1 = n;

    if (rowprm) {
        iwoff = m;
        /*
         * .. reordering the rows in decreasing sequence in the
         * ell-infinity norm - this enhances numerical robustness in
         * the case of differently scaled rows.
         */
        for (p = 0; p < m; p++) {
            rwork[p] = slange("M", 1, n, &A[p], lda, NULL);
            /* .. check for NaN's and Inf's */
            if (rwork[p] != rwork[p] || (rwork[p] * ZERO) != ZERO) {
                *info = -8;
                xerbla("SGESVDQ", -(*info));
                return;
            }
        }
        for (p = 0; p < m - 1; p++) {
            q = cblas_isamax(m - p, &rwork[p], 1) + p;
            iwork[n + p] = q;
            if (p != q) {
                rtmp = rwork[p];
                rwork[p] = rwork[q];
                rwork[q] = rtmp;
            }
        }

        /* Quick return: A is the M x N zero matrix. */
        if (rwork[0] == ZERO) {
            *numrank = 0;
            slaset("G", n, 1, ZERO, ZERO, S, n);
            if (wntus) slaset("G", m, n, ZERO, ONE, U, ldu);
            if (wntua) slaset("G", m, m, ZERO, ONE, U, ldu);
            if (wntva) slaset("G", n, n, ZERO, ONE, V, ldv);
            if (wntuf) {
                slaset("G", n, 1, ZERO, ZERO, work, n);
                slaset("G", m, n, ZERO, ONE, U, ldu);
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

        /*
         * .. to prevent overflow in the QR factorization, scale the
         * matrix by 1/sqrt(M) if too large entry detected
         */
        if (rwork[0] > big / sqrtf((f32)m)) {
            slascl("G", 0, 0, sqrtf((f32)m), ONE, m, n, A, lda, &ierr);
            ascaled = 1;
        }
        slaswp(n, A, lda, 0, m - 2, &iwork[n], 1);
    } else {
        /*
         * .. At this stage, preemptive scaling is done only to avoid column
         * norms overflows during the QR factorization. The SVD procedure should
         * have its own scaling to save the singular values from overflows and
         * underflows. That depends on the SVD procedure.
         */
        rtmp = slange("M", m, n, A, lda, NULL);
        if (rtmp != rtmp || (rtmp * ZERO) != ZERO) {
            *info = -8;
            xerbla("SGESVDQ", -(*info));
            return;
        }
        /*
         * .. to prevent overflow in the QR factorization, scale the
         * matrix by 1/sqrt(M) if too large entry detected
         */
        if (rtmp > big / sqrtf((f32)m)) {
            slascl("G", 0, 0, sqrtf((f32)m), ONE, m, n, A, lda, &ierr);
            ascaled = 1;
        }
    }

    /*
     * .. QR factorization with column pivoting
     *
     * A * P = Q * [ R ]
     *             [ 0 ]
     */
    for (p = 0; p < n; p++) {
        iwork[p] = 0;
    }
    sgeqp3(m, n, A, lda, iwork, work, &work[n], lwork - n, &ierr);

    /*
     * If the user requested accuracy level allows truncation in the
     * computed upper triangular factor, the matrix R is examined and,
     * if possible, replaced with its leading upper trapezoidal part.
     */
    nr = 1;
    if (accla) {
        /*
         * Standard absolute error bound suffices. All sigma_i with
         * sigma_i < N*EPS*||A||_F are flushed to zero. This is an
         * aggressive enforcement of lower numerical rank by introducing a
         * backward error of the order of N*EPS*||A||_F.
         */
        rtmp = sqrtf((f32)n) * epsln;
        for (p = 1; p < n; p++) {
            if (fabsf(A[p + p * lda]) < rtmp * fabsf(A[0])) break;
            nr++;
        }
    } else if (acclm) {
        /*
         * .. similarly as above, only slightly more gentle (less aggressive).
         * Sudden drop on the diagonal of R is used as the criterion for being
         * close-to-rank-deficient. The threshold is set to EPSLN=SLAMCH('E').
         * [[This can be made more flexible by replacing this hard-coded value
         * with a user specified threshold.]] Also, the values that underflow
         * will be truncated.
         */
        nr = 1;
        for (p = 1; p < n; p++) {
            if (fabsf(A[p + p * lda]) < epsln * fabsf(A[(p-1) + (p-1) * lda]) ||
                fabsf(A[p + p * lda]) < sfmin) break;
            nr++;
        }
    } else {
        /*
         * .. RRQR not authorized to determine numerical rank except in the
         * obvious case of zero pivots.
         * .. inspect R for exact zeros on the diagonal;
         * R(i,i)=0 => R(i:n-1,i:n-1)=0.
         */
        nr = 1;
        for (p = 1; p < n; p++) {
            if (fabsf(A[p + p * lda]) == ZERO) break;
            nr++;
        }

        /*
         * Estimate the scaled condition number of A. Use the fact that it is
         * the same as the scaled condition number of R.
         */
        if (conda) {
            slacpy("U", n, n, A, lda, V, ldv);
            for (p = 0; p < nr; p++) {
                rtmp = cblas_snrm2(p + 1, &V[p * ldv], 1);
                cblas_sscal(p + 1, ONE / rtmp, &V[p * ldv], 1);
            }
            if (!lsvec && !rsvec) {
                spocon("U", nr, V, ldv, ONE, &rtmp, work, &iwork[n + iwoff], &ierr);
            } else {
                spocon("U", nr, V, ldv, ONE, &rtmp, &work[n], &iwork[n + iwoff], &ierr);
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
            /*
             * .. compute the singular values of R**T = [A](0:nr-1,0:n-1)**T
             * .. set the lower triangle of [A] to [A](0:nr-1,0:n-1)**T and
             * the upper triangle of [A] to zero.
             */
            INT minmn = (n < nr) ? n : nr;
            for (p = 0; p < minmn; p++) {
                for (q = p + 1; q < n; q++) {
                    A[q + p * lda] = A[p + q * lda];
                    if (q < nr) A[p + q * lda] = ZERO;
                }
            }
            sgesvd("N", "N", n, nr, A, lda, S, U, ldu, V, ldv, work, lwork, info);
        } else {
            /* .. compute the singular values of R = [A](0:nr-1,0:n-1) */
            if (nr > 1) {
                slaset("L", nr - 1, nr - 1, ZERO, ZERO, &A[1], lda);
            }
            sgesvd("N", "N", nr, n, A, lda, S, U, ldu, V, ldv, work, lwork, info);
        }

    } else if (lsvec && !rsvec) {
        /*
         * .. the singular values and the left singular vectors requested
         */
        if (rtrans) {
            /*
             * .. apply SGESVD to R**T
             * .. copy R**T into [U] and overwrite [U] with the right singular
             * vectors of R
             */
            for (p = 0; p < nr; p++) {
                for (q = p; q < n; q++) {
                    U[q + p * ldu] = A[p + q * lda];
                }
            }
            if (nr > 1) {
                slaset("U", nr - 1, nr - 1, ZERO, ZERO, &U[ldu], ldu);
            }
            /*
             * .. the left singular vectors not computed, the NR right singular
             * vectors overwrite [U](0:nr-1,0:nr-1) as transposed. These
             * will be pre-multiplied by Q to build the left singular vectors of A.
             */
            sgesvd("N", "O", n, nr, U, ldu, S, NULL, 1, NULL, 1, &work[n], lwork - n, info);
            for (p = 0; p < nr; p++) {
                for (q = p + 1; q < nr; q++) {
                    rtmp = U[q + p * ldu];
                    U[q + p * ldu] = U[p + q * ldu];
                    U[p + q * ldu] = rtmp;
                }
            }
        } else {
            /*
             * .. apply SGESVD to R
             * .. copy R into [U] and overwrite [U] with the left singular vectors
             */
            slacpy("U", nr, n, A, lda, U, ldu);
            if (nr > 1) {
                slaset("L", nr - 1, nr - 1, ZERO, ZERO, &U[1], ldu);
            }
            /*
             * .. the right singular vectors not computed, the NR left singular
             * vectors overwrite [U](0:nr-1,0:nr-1)
             */
            sgesvd("O", "N", nr, n, U, ldu, S, NULL, 1, NULL, 1, &work[n], lwork - n, info);
        }

        /*
         * .. assemble the left singular vector matrix U of dimensions
         * (M x NR) or (M x N) or (M x M).
         */
        if (nr < m && !wntuf) {
            slaset("A", m - nr, nr, ZERO, ZERO, &U[nr], ldu);
            if (nr < n1) {
                slaset("A", nr, n1 - nr, ZERO, ZERO, &U[nr * ldu], ldu);
                slaset("A", m - nr, n1 - nr, ZERO, ONE, &U[nr + nr * ldu], ldu);
            }
        }

        /*
         * The Q matrix from the first QRF is built into the left singular
         * vectors matrix U.
         */
        if (!wntuf) {
            sormqr("L", "N", m, n1, n, A, lda, work, U, ldu, &work[n], lwork - n, &ierr);
        }
        if (rowprm && !wntuf) {
            slaswp(n1, U, ldu, 0, m - 2, &iwork[n], -1);
        }

    } else if (rsvec && !lsvec) {
        /*
         * .. the singular values and the right singular vectors requested
         */
        if (rtrans) {
            /*
             * .. apply SGESVD to R**T
             * .. copy R**T into V and overwrite V with the left singular vectors
             */
            for (p = 0; p < nr; p++) {
                for (q = p; q < n; q++) {
                    V[q + p * ldv] = A[p + q * lda];
                }
            }
            if (nr > 1) {
                slaset("U", nr - 1, nr - 1, ZERO, ZERO, &V[ldv], ldv);
            }

            if (wntvr || nr == n) {
                /*
                 * .. the left singular vectors of R**T overwrite V, the right singular
                 * vectors not computed
                 */
                sgesvd("O", "N", n, nr, V, ldv, S, U, ldu, NULL, ldu, &work[n], lwork - n, info);
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
                slapmt(0, nr, n, V, ldv, iwork);
            } else {
                /*
                 * .. need all N right singular vectors and NR < N
                 * [!] This is simple implementation that augments [V](0:n-1,0:nr-1)
                 * by padding a zero block. In the case NR << N, a more efficient
                 * way is to first use the QR factorization. For more details
                 * how to implement this, see the " FULL SVD " branch.
                 */
                slaset("G", n, n - nr, ZERO, ZERO, &V[nr * ldv], ldv);
                sgesvd("O", "N", n, n, V, ldv, S, U, ldu, NULL, ldu, &work[n], lwork - n, info);
                for (p = 0; p < n; p++) {
                    for (q = p + 1; q < n; q++) {
                        rtmp = V[q + p * ldv];
                        V[q + p * ldv] = V[p + q * ldv];
                        V[p + q * ldv] = rtmp;
                    }
                }
                slapmt(0, n, n, V, ldv, iwork);
            }
        } else {
            /*
             * .. aply SGESVD to R
             * .. copy R into V and overwrite V with the right singular vectors
             */
            slacpy("U", nr, n, A, lda, V, ldv);
            if (nr > 1) {
                slaset("L", nr - 1, nr - 1, ZERO, ZERO, &V[1], ldv);
            }

            if (wntvr || nr == n) {
                /*
                 * .. the right singular vectors overwrite V, the NR left singular
                 * vectors stored in U(0:nr-1,0:nr-1)
                 */
                sgesvd("N", "O", nr, n, V, ldv, S, NULL, 1, NULL, 1, &work[n], lwork - n, info);
                slapmt(0, nr, n, V, ldv, iwork);
            } else {
                /*
                 * .. need all N right singular vectors and NR < N
                 * [!] This is simple implementation that augments [V](0:nr-1,0:n-1)
                 * by padding a zero block. In the case NR << N, a more efficient
                 * way is to first use the LQ factorization. For more details
                 * how to implement this, see the " FULL SVD " branch.
                 */
                slaset("G", n - nr, n, ZERO, ZERO, &V[nr], ldv);
                sgesvd("N", "O", n, n, V, ldv, S, NULL, 1, NULL, 1, &work[n], lwork - n, info);
                slapmt(0, n, n, V, ldv, iwork);
            }
        }

    } else {
        /*
         * .. FULL SVD requested
         */
        if (rtrans) {
            /*
             * .. apply SGESVD to R**T [[this option is left for R&D&T]]
             */
            if (wntvr || nr == n) {
                /*
                 * .. copy R**T into [V] and overwrite [V] with the left singular
                 * vectors of R**T
                 */
                for (p = 0; p < nr; p++) {
                    for (q = p; q < n; q++) {
                        V[q + p * ldv] = A[p + q * lda];
                    }
                }
                if (nr > 1) {
                    slaset("U", nr - 1, nr - 1, ZERO, ZERO, &V[ldv], ldv);
                }

                sgesvd("O", "A", n, nr, V, ldv, S, NULL, 1, U, ldu, &work[n], lwork - n, info);

                /* .. assemble V */
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
                slapmt(0, nr, n, V, ldv, iwork);

                for (p = 0; p < nr; p++) {
                    for (q = p + 1; q < nr; q++) {
                        rtmp = U[q + p * ldu];
                        U[q + p * ldu] = U[p + q * ldu];
                        U[p + q * ldu] = rtmp;
                    }
                }

                if (nr < m && !wntuf) {
                    slaset("A", m - nr, nr, ZERO, ZERO, &U[nr], ldu);
                    if (nr < n1) {
                        slaset("A", nr, n1 - nr, ZERO, ZERO, &U[nr * ldu], ldu);
                        slaset("A", m - nr, n1 - nr, ZERO, ONE, &U[nr + nr * ldu], ldu);
                    }
                }
            } else {
                /*
                 * .. need all N right singular vectors and NR < N
                 * .. copy R**T into [V] and overwrite [V] with the left singular
                 * vectors of R**T
                 * [[The optimal ratio N/NR for using QRF instead of padding
                 *   with zeros. Here hard coded to 2; it must be at least
                 *   two due to work space constraints.]]
                 */
                optratio = 2;
                if (optratio * nr > n) {
                    for (p = 0; p < nr; p++) {
                        for (q = p; q < n; q++) {
                            V[q + p * ldv] = A[p + q * lda];
                        }
                    }
                    if (nr > 1) {
                        slaset("U", nr - 1, nr - 1, ZERO, ZERO, &V[ldv], ldv);
                    }
                    slaset("A", n, n - nr, ZERO, ZERO, &V[nr * ldv], ldv);

                    sgesvd("O", "A", n, n, V, ldv, S, NULL, 1, U, ldu, &work[n], lwork - n, info);

                    for (p = 0; p < n; p++) {
                        for (q = p + 1; q < n; q++) {
                            rtmp = V[q + p * ldv];
                            V[q + p * ldv] = V[p + q * ldv];
                            V[p + q * ldv] = rtmp;
                        }
                    }
                    slapmt(0, n, n, V, ldv, iwork);

                    for (p = 0; p < n; p++) {
                        for (q = p + 1; q < n; q++) {
                            rtmp = U[q + p * ldu];
                            U[q + p * ldu] = U[p + q * ldu];
                            U[p + q * ldu] = rtmp;
                        }
                    }

                    /*
                     * .. assemble the left singular vector matrix U of dimensions
                     * (M x N1), i.e. (M x N) or (M x M).
                     */
                    if (n < m && !wntuf) {
                        slaset("A", m - n, n, ZERO, ZERO, &U[n], ldu);
                        if (n < n1) {
                            slaset("A", n, n1 - n, ZERO, ZERO, &U[n * ldu], ldu);
                            slaset("A", m - n, n1 - n, ZERO, ONE, &U[n + n * ldu], ldu);
                        }
                    }
                } else {
                    /*
                     * .. copy R**T into [U] and overwrite [U] with the right
                     * singular vectors of R
                     */
                    for (p = 0; p < nr; p++) {
                        for (q = p; q < n; q++) {
                            U[q + (nr + p) * ldu] = A[p + q * lda];
                        }
                    }
                    if (nr > 1) {
                        slaset("U", nr - 1, nr - 1, ZERO, ZERO, &U[(nr + 1) * ldu], ldu);
                    }
                    sgeqrf(n, nr, &U[nr * ldu], ldu, &work[n], &work[n + nr], lwork - n - nr, &ierr);
                    for (p = 0; p < nr; p++) {
                        for (q = 0; q < n; q++) {
                            V[q + p * ldv] = U[p + (nr + q) * ldu];
                        }
                    }
                    slaset("U", nr - 1, nr - 1, ZERO, ZERO, &V[ldv], ldv);
                    sgesvd("S", "O", nr, nr, V, ldv, S, U, ldu, NULL, 1, &work[n + nr], lwork - n - nr, info);
                    slaset("A", n - nr, nr, ZERO, ZERO, &V[nr], ldv);
                    slaset("A", nr, n - nr, ZERO, ZERO, &V[nr * ldv], ldv);
                    slaset("A", n - nr, n - nr, ZERO, ONE, &V[nr + nr * ldv], ldv);
                    sormqr("R", "C", n, n, nr, &U[nr * ldu], ldu, &work[n], V, ldv, &work[n + nr], lwork - n - nr, &ierr);
                    slapmt(0, n, n, V, ldv, iwork);

                    /*
                     * .. assemble the left singular vector matrix U of dimensions
                     * (M x NR) or (M x N) or (M x M).
                     */
                    if (nr < m && !wntuf) {
                        slaset("A", m - nr, nr, ZERO, ZERO, &U[nr], ldu);
                        if (nr < n1) {
                            slaset("A", nr, n1 - nr, ZERO, ZERO, &U[nr * ldu], ldu);
                            slaset("A", m - nr, n1 - nr, ZERO, ONE, &U[nr + nr * ldu], ldu);
                        }
                    }
                }
            }
        } else {
            /*
             * .. apply SGESVD to R [[this is the recommended option]]
             */
            if (wntvr || nr == n) {
                /* .. copy R into [V] and overwrite V with the right singular vectors */
                slacpy("U", nr, n, A, lda, V, ldv);
                if (nr > 1) {
                    slaset("L", nr - 1, nr - 1, ZERO, ZERO, &V[1], ldv);
                }

                sgesvd("S", "O", nr, n, V, ldv, S, U, ldu, NULL, 1, &work[n], lwork - n, info);
                slapmt(0, nr, n, V, ldv, iwork);

                /*
                 * .. now [V](0:nr-1,0:n-1) contains V(0:n-1,0:nr-1)**T
                 * .. assemble the left singular vector matrix U of dimensions
                 * (M x NR) or (M x N) or (M x M).
                 */
                if (nr < m && !wntuf) {
                    slaset("A", m - nr, nr, ZERO, ZERO, &U[nr], ldu);
                    if (nr < n1) {
                        slaset("A", nr, n1 - nr, ZERO, ZERO, &U[nr * ldu], ldu);
                        slaset("A", m - nr, n1 - nr, ZERO, ONE, &U[nr + nr * ldu], ldu);
                    }
                }
            } else {
                /*
                 * .. need all N right singular vectors and NR < N
                 * .. the requested number of the left singular vectors
                 *  is then N1 (N or M)
                 * [[The optimal ratio N/NR for using LQ instead of padding
                 *   with zeros. Here hard coded to 2; it must be at least
                 *   two due to work space constraints.]]
                 */
                optratio = 2;
                if (optratio * nr > n) {
                    slacpy("U", nr, n, A, lda, V, ldv);
                    if (nr > 1) {
                        slaset("L", nr - 1, nr - 1, ZERO, ZERO, &V[1], ldv);
                    }
                    slaset("A", n - nr, n, ZERO, ZERO, &V[nr], ldv);

                    sgesvd("S", "O", n, n, V, ldv, S, U, ldu, NULL, 1, &work[n], lwork - n, info);
                    slapmt(0, n, n, V, ldv, iwork);

                    /*
                     * .. now [V] contains the transposed matrix of the right
                     * singular vectors of A. The leading N left singular vectors
                     * are in [U](0:n-1,0:n-1)
                     * .. assemble the left singular vector matrix U of dimensions
                     * (M x N1), i.e. (M x N) or (M x M).
                     */
                    if (n < m && !wntuf) {
                        slaset("A", m - n, n, ZERO, ZERO, &U[n], ldu);
                        if (n < n1) {
                            slaset("A", n, n1 - n, ZERO, ZERO, &U[n * ldu], ldu);
                            slaset("A", m - n, n1 - n, ZERO, ONE, &U[n + n * ldu], ldu);
                        }
                    }
                } else {
                    slacpy("U", nr, n, A, lda, &U[nr], ldu);
                    if (nr > 1) {
                        slaset("L", nr - 1, nr - 1, ZERO, ZERO, &U[nr + 1], ldu);
                    }
                    sgelqf(nr, n, &U[nr], ldu, &work[n], &work[n + nr], lwork - n - nr, &ierr);
                    slacpy("L", nr, nr, &U[nr], ldu, V, ldv);
                    if (nr > 1) {
                        slaset("U", nr - 1, nr - 1, ZERO, ZERO, &V[ldv], ldv);
                    }
                    sgesvd("S", "O", nr, nr, V, ldv, S, U, ldu, NULL, 1, &work[n + nr], lwork - n - nr, info);
                    slaset("A", n - nr, nr, ZERO, ZERO, &V[nr], ldv);
                    slaset("A", nr, n - nr, ZERO, ZERO, &V[nr * ldv], ldv);
                    slaset("A", n - nr, n - nr, ZERO, ONE, &V[nr + nr * ldv], ldv);
                    sormlq("R", "N", n, n, nr, &U[nr], ldu, &work[n], V, ldv, &work[n + nr], lwork - n - nr, &ierr);
                    slapmt(0, n, n, V, ldv, iwork);

                    /*
                     * .. assemble the left singular vector matrix U of dimensions
                     * (M x NR) or (M x N) or (M x M).
                     */
                    if (nr < m && !wntuf) {
                        slaset("A", m - nr, nr, ZERO, ZERO, &U[nr], ldu);
                        if (nr < n1) {
                            slaset("A", nr, n1 - nr, ZERO, ZERO, &U[nr * ldu], ldu);
                            slaset("A", m - nr, n1 - nr, ZERO, ONE, &U[nr + nr * ldu], ldu);
                        }
                    }
                }
            }
        }

        /*
         * The Q matrix from the first QRF is built into the left singular
         * vectors matrix U.
         */
        if (!wntuf) {
            sormqr("L", "N", m, n1, n, A, lda, work, U, ldu, &work[n], lwork - n, &ierr);
        }
        if (rowprm && !wntuf) {
            slaswp(n1, U, ldu, 0, m - 2, &iwork[n], -1);
        }
    }

    /*
     * Check whether some singular values are returned as zeros, e.g.
     * due to underflow, and update the numerical rank.
     */
    p = nr;
    for (q = nr - 1; q >= 0; q--) {
        if (S[q] > ZERO) break;
        nr--;
    }

    /*
     * .. if numerical rank deficiency is detected, the truncated
     * singular values are set to zero.
     */
    if (nr < n) {
        slaset("G", n - nr, 1, ZERO, ZERO, &S[nr], n);
    }

    /*
     * .. undo scaling; this may cause overflow in the largest singular
     * values.
     */
    if (ascaled) {
        slascl("G", 0, 0, ONE, sqrtf((f32)m), nr, 1, S, n, &ierr);
    }

    if (conda) rwork[0] = sconda;
    rwork[1] = (f32)(p - nr);

    *numrank = nr;
    *info = ierr;
}
