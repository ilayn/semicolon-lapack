/**
 * @file sgejsv.c
 * @brief SGEJSV computes the SVD of a real M-by-N matrix using preconditioned
 *        Jacobi rotations with sophisticated preprocessing for high accuracy.
 */

#include "semicolon_lapack_single.h"
#include <math.h>
#include <cblas.h>

static const f32 ZERO = 0.0f;
static const f32 ONE = 1.0f;

/** @cond */
/* Helper: max of 3 integers */
static inline int max3i(int a, int b, int c) {
    int m = (a > b) ? a : b;
    return (m > c) ? m : c;
}
/** @endcond */

/**
 * SGEJSV computes the singular value decomposition (SVD) of a real M-by-N
 * matrix [A], where M >= N. The SVD of [A] is written as
 *
 *              [A] = [U] * [SIGMA] * [V]^t,
 *
 * where [SIGMA] is an N-by-N (M-by-N) matrix which is zero except for its N
 * diagonal elements, [U] is an M-by-N (or M-by-M) orthonormal matrix, and
 * [V] is an N-by-N orthogonal matrix. The diagonal elements of [SIGMA] are
 * the singular values of [A]. The columns of [U] and [V] are the left and
 * the right singular vectors of [A], respectively. The matrices [U] and [V]
 * are computed and stored in the arrays U and V, respectively. The diagonal
 * of [SIGMA] is computed and stored in the array SVA.
 *
 * SGEJSV can sometimes compute tiny singular values and their singular vectors
 * much more accurately than other SVD routines, see below under Further Details.
 *
 * @param[in] joba
 *        Specifies the level of accuracy:
 *        - 'C': This option works well (high relative accuracy) if A = B * D,
 *              with well-conditioned B and arbitrary diagonal matrix D.
 *              The accuracy cannot be spoiled by COLUMN scaling. The
 *              accuracy of the computed output depends on the condition of
 *              B, and the procedure aims at the best theoretical accuracy.
 *              The relative error max_{i=1:N}|d sigma_i| / sigma_i is
 *              bounded by f(M,N)*epsilon* cond(B), independent of D.
 *              The input matrix is preprocessed with the QRF with column
 *              pivoting. This initial preprocessing and preconditioning by
 *              a rank revealing QR factorization is common for all values of
 *              JOBA. Additional actions are specified as follows:
 *        - 'E': Computation as with 'C' with an additional estimate of the
 *              condition number of B. It provides a realistic error bound.
 *        - 'F': If A = D1 * C * D2 with ill-conditioned diagonal scalings
 *              D1, D2, and well-conditioned matrix C, this option gives
 *              higher accuracy than the 'C' option. If the structure of the
 *              input matrix is not known, and relative accuracy is
 *              desirable, then this option is advisable. The input matrix A
 *              is preprocessed with QR factorization with FULL (row and
 *              column) pivoting.
 *        - 'G': Computation as with 'F' with an additional estimate of the
 *              condition number of B, where A=D*B. If A has heavily weighted
 *              rows, then using this condition number gives too pessimistic
 *              error bound.
 *        - 'A': Small singular values are the noise and the matrix is treated
 *              as numerically rank deficient. The error in the computed
 *              singular values is bounded by f(m,n)*epsilon*||A||.
 *              The computed SVD A = U * S * V^t restores A up to
 *              f(m,n)*epsilon*||A||.
 *              This gives the procedure the licence to discard (set to zero)
 *              all singular values below N*epsilon*||A||.
 *        - 'R': Similar as in 'A'. Rank revealing property of the initial
 *              QR factorization is used do reveal (using triangular factor)
 *              a gap sigma_{r+1} < epsilon * sigma_r in which case the
 *              numerical RANK is declared to be r. The SVD is computed with
 *              absolute error bounds, but more accurately than with 'A'.
 *
 * @param[in] jobu
 *        Specifies whether to compute the columns of U:
 *        - 'U': N columns of U are returned in the array U.
 *        - 'F': full set of M left sing. vectors is returned in the array U.
 *        - 'W': U may be used as workspace of length M*N. See the description
 *              of U.
 *        - 'N': U is not computed.
 *
 * @param[in] jobv
 *        Specifies whether to compute the matrix V:
 *        - 'V': N columns of V are returned in the array V; Jacobi rotations
 *              are not explicitly accumulated.
 *        - 'J': N columns of V are returned in the array V, but they are
 *              computed as the product of Jacobi rotations. This option is
 *              allowed only if JOBU .NE. 'N', i.e. in computing the full SVD.
 *        - 'W': V may be used as workspace of length N*N. See the description
 *              of V.
 *        - 'N': V is not computed.
 *
 * @param[in] jobr
 *        Specifies the RANGE for the singular values. Issues the licence to
 *        set to zero small positive singular values if they are outside
 *        specified range. If A .NE. 0 is scaled so that the largest singular
 *        value of c*A is around sqrt(BIG), BIG=SLAMCH('O'), then JOBR issues
 *        the licence to kill columns of A whose norm in c*A is less than
 *        sqrt(SFMIN) (for JOBR = 'R'), or less than SMALL=SFMIN/EPSLN,
 *        where SFMIN=SLAMCH('S'), EPSLN=SLAMCH('E').
 *        - 'N': Do not kill small columns of c*A. This option assumes that
 *              BLAS and QR factorizations and triangular solvers are
 *              implemented to work in that range. If the condition of A
 *              is greater than BIG, use SGESVJ.
 *        - 'R': RESTRICTED range for sigma(c*A) is [sqrt(SFMIN), sqrt(BIG)]
 *              (roughly, as described above). This option is recommended.
 *        For computing the singular values in the FULL range [SFMIN,BIG]
 *        use SGESVJ.
 *
 * @param[in] jobt
 *        If the matrix is square then the procedure may determine to use
 *        transposed A if A^t seems to be better with respect to convergence.
 *        If the matrix is not square, JOBT is ignored. This is subject to
 *        changes in the future.
 *        The decision is based on two values of entropy over the adjoint
 *        orbit of A^t * A. See the descriptions of work[5] and work[6].
 *        - 'T': transpose if entropy test indicates possibly faster
 *              convergence of Jacobi process if A^t is taken as input. If A is
 *              replaced with A^t, then the row pivoting is included automatically.
 *        - 'N': do not speculate.
 *        This option can be used to compute only the singular values, or the
 *        full SVD (U, SIGMA and V). For only one set of singular vectors
 *        (U or V), the caller should provide both U and V, as one of the
 *        matrices is used as workspace if the matrix A is transposed.
 *        The implementer can easily remove this constraint and make the
 *        code more complicated. See the descriptions of U and V.
 *
 * @param[in] jobp
 *        Issues the licence to introduce structured perturbations to drown
 *        denormalized numbers. This licence should be active if the
 *        denormals are poorly implemented, causing slow computation,
 *        especially in cases of fast convergence (!). For details see [1,2].
 *        For the sake of simplicity, this perturbations are included only
 *        when the full SVD or only the singular values are requested. The
 *        implementer/user can easily add the perturbation for the cases of
 *        computing one set of singular vectors.
 *        - 'P': introduce perturbation
 *        - 'N': do not perturb
 *
 * @param[in] m
 *        The number of rows of the input matrix A. M >= 0.
 *
 * @param[in] n
 *        The number of columns of the input matrix A. M >= N >= 0.
 *
 * @param[in,out] A
 *        Double precision array, dimension (lda, n).
 *        On entry, the M-by-N matrix A.
 *
 * @param[in] lda
 *        The leading dimension of the array A. lda >= max(1, m).
 *
 * @param[out] SVA
 *        Double precision array, dimension (n).
 *        On exit:
 *        - For work[0]/work[1] = 1: The singular values of A. During the
 *          computation SVA contains Euclidean column norms of the
 *          iterated matrices in the array A.
 *        - For work[0] != work[1]: The singular values of A are
 *          (work[0]/work[1]) * SVA[0:n-1]. This factored form is used if
 *          sigma_max(A) overflows or if small singular values have been
 *          saved from underflow by scaling the input matrix A.
 *        - If JOBR='R' then some of the singular values may be returned
 *          as exact zeros obtained by "set to zero" because they are
 *          below the numerical rank threshold or are denormalized numbers.
 *
 * @param[out] U
 *        Double precision array, dimension (ldu, n) or (ldu, m).
 *        - If JOBU = 'U', then U contains on exit the M-by-N matrix of
 *          the left singular vectors.
 *        - If JOBU = 'F', then U contains on exit the M-by-M matrix of
 *          the left singular vectors, including an ONB of the orthogonal
 *          complement of the Range(A).
 *        - If JOBU = 'W' and (JOBV = 'V' and JOBT = 'T' and M = N),
 *          then U is used as workspace if the procedure replaces A with A^t.
 *          In that case, [V] is computed in U as left singular vectors of A^t
 *          and then copied back to the V array. This 'W' option is just
 *          a reminder to the caller that in this case U is reserved as
 *          workspace of length N*N.
 *        - If JOBU = 'N', U is not referenced, unless JOBT='T'.
 *
 * @param[in] ldu
 *        The leading dimension of the array U. ldu >= 1.
 *        If JOBU = 'U' or 'F' or 'W', then ldu >= m.
 *
 * @param[out] V
 *        Double precision array, dimension (ldv, n).
 *        - If JOBV = 'V' or 'J', then V contains on exit the N-by-N matrix of
 *          the right singular vectors.
 *        - If JOBV = 'W' and (JOBU = 'U' and JOBT = 'T' and M = N),
 *          then V is used as workspace if the procedure replaces A with A^t.
 *          In that case, [U] is computed in V as right singular vectors of A^t
 *          and then copied back to the U array. This 'W' option is just
 *          a reminder to the caller that in this case V is reserved as
 *          workspace of length N*N.
 *        - If JOBV = 'N', V is not referenced, unless JOBT='T'.
 *
 * @param[in] ldv
 *        The leading dimension of the array V. ldv >= 1.
 *        If JOBV = 'V' or 'J' or 'W', then ldv >= n.
 *
 * @param[out] work
 *        Double precision array, dimension (max(7, lwork)).
 *        On exit, if n > 0 and m > 0 (else not referenced):
 *        - work[0] = SCALE = work[1] / work[0] is the scaling factor such
 *                   that SCALE*SVA[0:n-1] are the computed singular values
 *                   of A. (See the description of SVA.)
 *        - work[1] = See the description of work[0].
 *        - work[2] = SCONDA is an estimate for the condition number of
 *                   column equilibrated A. (If JOBA = 'E' or 'G')
 *                   SCONDA is an estimate of sqrt(||(R^t * R)^(-1)||_1).
 *                   It is computed using SPOCON. It holds
 *                   N^(-1/4) * SCONDA <= ||R^(-1)||_2 <= N^(1/4) * SCONDA
 *                   where R is the triangular factor from the QRF of A.
 *                   However, if R is truncated and the numerical rank is
 *                   determined to be strictly smaller than N, SCONDA is
 *                   returned as -1, thus indicating that the smallest
 *                   singular values might be lost.
 *
 *        If full SVD is needed, the following two condition numbers are
 *        useful for the analysis of the algorithm. They are provided for
 *        a developer/implementer who is familiar with the details of
 *        the method.
 *
 *        - work[3] = an estimate of the scaled condition number of the
 *                   triangular factor in the first QR factorization.
 *        - work[4] = an estimate of the scaled condition number of the
 *                   triangular factor in the second QR factorization.
 *
 *        The following two parameters are computed if JOBT = 'T'.
 *        They are provided for a developer/implementer who is familiar
 *        with the details of the method.
 *
 *        - work[5] = the entropy of A^t*A :: this is the Shannon entropy
 *                   of diag(A^t*A) / Trace(A^t*A) taken as point in the
 *                   probability simplex.
 *        - work[6] = the entropy of A*A^t.
 *
 * @param[in] lwork
 *        Length of work to confirm proper allocation of work space.
 *        LWORK depends on the job:
 *
 *        If only SIGMA is needed (JOBU = 'N', JOBV = 'N') and
 *          - no scaled condition estimate required (JOBA != 'E','G'):
 *            lwork >= max(2*m+n, 4*n+1, 7). This is the minimal requirement.
 *            For optimal performance (blocked code) the optimal value
 *            is lwork >= max(2*m+n, 3*n+(n+1)*NB, 7). Here NB is the optimal
 *            block size for SGEQP3 and SGEQRF.
 *          - an estimate of the scaled condition number of A is
 *            required (JOBA='E', 'G'). In this case, lwork is the maximum
 *            of the above and n*n+4*n, i.e. lwork >= max(2*m+n, n*n+4*n, 7).
 *
 *        If SIGMA and the right singular vectors are needed (JOBV = 'V'):
 *            lwork >= max(2*m+n, 4*n+1, 7).
 *
 *        If SIGMA and the left singular vectors are needed:
 *            lwork >= max(2*m+n, 4*n+1, 7).
 *            if JOBU = 'F': lwork >= max(2*m+n, 3*n+(n+1)*NB, n+m*NB, 7).
 *
 *        If the full SVD is needed: (JOBU = 'U' or JOBU = 'F') and
 *          - if JOBV = 'V': lwork >= max(2*m+n, 6*n+2*n*n).
 *          - if JOBV = 'J': lwork >= max(2*m+n, 4*n+n*n, 2*n+n*n+6).
 *
 * @param[out] iwork
 *        Integer array, dimension (max(3, m+3*n)).
 *        On exit:
 *        - iwork[0] = the numerical rank determined after the initial
 *                    QR factorization with pivoting. See the descriptions
 *                    of JOBA and JOBR.
 *        - iwork[1] = the number of the computed nonzero singular values
 *        - iwork[2] = if nonzero, a warning message:
 *                    If iwork[2] = 1 then some of the column norms of A
 *                    were denormalized floats. The requested high accuracy
 *                    is not warranted by the data.
 *
 * @param[out] info
 *                           - < 0: if info = -i, then the i-th argument had an illegal value.
 *                           - = 0: successful exit.
 *                           - > 0: SGEJSV did not converge in the maximal allowed number
 *                           of sweeps. The computed values may be inaccurate.
 * @par Further Details:
 *
 * SGEJSV implements a preconditioned Jacobi SVD algorithm. It uses SGEQP3,
 * SGEQRF, and SGELQF as preprocessors and preconditioners. Optionally, an
 * additional row pivoting can be used as a preprocessor, which in some
 * cases results in much higher accuracy. An example is matrix A with the
 * structure A = D1 * C * D2, where D1, D2 are arbitrarily ill-conditioned
 * diagonal matrices and C is well-conditioned matrix. In that case, complete
 * pivoting in the first QR factorizations provides accuracy dependent on the
 * condition number of C, and independent of D1, D2. Such higher accuracy is
 * not completely understood theoretically, but it works well in practice.
 * Further, if A can be written as A = B*D, with well-conditioned B and some
 * diagonal D, then the high accuracy is guaranteed, both theoretically and
 * in software, independent of D. For more details see [1], [2].
 *
 * The computational range for the singular values can be the full range
 * (UNDERFLOW, OVERFLOW), provided that the machine arithmetic and the BLAS
 * & LAPACK routines called by SGEJSV are implemented to work in that range.
 * If that is not the case, then the restriction for safe computation with
 * the singular values in the range of normalized IEEE numbers is that the
 * spectral condition number kappa(A)=sigma_max(A)/sigma_min(A) does not
 * overflow. This code (SGEJSV) is best used in this restricted range,
 * meaning that singular values of magnitude below ||A||_2 / SLAMCH('O') are
 * returned as zeros. See JOBR for details on this.
 *
 * Further, this implementation is somewhat slower than the one described
 * in [1,2] due to replacement of some non-LAPACK components, and because
 * the choice of some tuning parameters in the iterative part (SGESVJ) is
 * left to the implementer on a particular machine.
 *
 * The rank revealing QR factorization (in this code: SGEQP3) should be
 * implemented as in [3]. We have a new version of SGEQP3 under development
 * that is more robust than the current one in LAPACK, with a cleaner cut in
 * rank deficient cases. It will be available in the SIGMA library [4].
 * If M is much larger than N, it is obvious that the initial QRF with
 * column pivoting can be preprocessed by the QRF without pivoting. That
 * well known trick is not used in SGEJSV because in some cases heavy row
 * weighting can be treated with complete pivoting. The overhead in cases
 * M much larger than N is then only due to pivoting, but the benefits in
 * terms of accuracy have prevailed. The implementer/user can incorporate
 * this extra QRF step easily. The implementer can also improve data movement
 * (matrix transpose, matrix copy, matrix transposed copy) - this
 * implementation of SGEJSV uses only the simplest, naive data movement.
 *
 * @par Contributors:
 * Zlatko Drmac (Zagreb, Croatia) and Kresimir Veselic (Hagen, Germany)
 *
 * @par References:
 * [1] Z. Drmac and K. Veselic: New fast and accurate Jacobi SVD algorithm I.
 *     SIAM J. Matrix Anal. Appl. Vol. 35, No. 2 (2008), pp. 1322-1342.
 *     LAPACK Working note 169.
 *
 * [2] Z. Drmac and K. Veselic: New fast and accurate Jacobi SVD algorithm II.
 *     SIAM J. Matrix Anal. Appl. Vol. 35, No. 2 (2008), pp. 1343-1362.
 *     LAPACK Working note 170.
 *
 * [3] Z. Drmac and Z. Bujanovic: On the failure of rank-revealing QR
 *     factorization software - a case study.
 *     ACM Trans. Math. Softw. Vol. 35, No 2 (2008), pp. 1-28.
 *     LAPACK Working note 176.
 *
 * [4] Z. Drmac: SIGMA - mathematical software library for accurate SVD, PSV,
 *     QSVD, (H,K)-SVD computations.
 *     Department of Mathematics, University of Zagreb, 2008.
 *
 * @par Bugs, examples and comments:
 * Please report all bugs and send interesting examples and/or comments to
 * drmac@math.hr. Thank you.
 */
void sgejsv(const char* joba, const char* jobu, const char* jobv,
            const char* jobr, const char* jobt, const char* jobp,
            const int m, const int n,
            f32* restrict A, const int lda,
            f32* restrict SVA,
            f32* restrict U, const int ldu,
            f32* restrict V, const int ldv,
            f32* restrict work, const int lwork,
            int* restrict iwork, int* info)
{
    /* Local variables */
    f32 aapp, aaqq, aatmax, aatmin, big, big1, cond_ok;
    f32 condr1, condr2, entra, entrat, epsln, maxprj, scalem;
    f32 sconda, sfmin, small, temp1, uscal1, uscal2, xsc;
    int ierr, n1, nr, numrank, p, q, warning;
    int almort, defr, errest, goscal, jracc, kill, lsvec;
    int l2aber, l2kill, l2pert, l2rank, l2tran;
    int noscal, rowpiv, rsvec, transp;

    /* -------------------------------------------------------------------- */
    /* Chunk 1: Parse boolean flags from CHARACTER*1 job parameters         */
    /* -------------------------------------------------------------------- */

    lsvec  = (jobu[0] == 'U' || jobu[0] == 'u' || jobu[0] == 'F' || jobu[0] == 'f');
    jracc  = (jobv[0] == 'J' || jobv[0] == 'j');
    rsvec  = (jobv[0] == 'V' || jobv[0] == 'v') || jracc;
    rowpiv = (joba[0] == 'F' || joba[0] == 'f' || joba[0] == 'G' || joba[0] == 'g');
    l2rank = (joba[0] == 'R' || joba[0] == 'r');
    l2aber = (joba[0] == 'A' || joba[0] == 'a');
    errest = (joba[0] == 'E' || joba[0] == 'e' || joba[0] == 'G' || joba[0] == 'g');
    l2tran = (jobt[0] == 'T' || jobt[0] == 't');
    l2kill = (jobr[0] == 'R' || jobr[0] == 'r');
    defr   = (jobr[0] == 'N' || jobr[0] == 'n');
    l2pert = (jobp[0] == 'P' || jobp[0] == 'p');

    /* -------------------------------------------------------------------- */
    /* Chunk 2: Parameter validation                                        */
    /* -------------------------------------------------------------------- */

    *info = 0;

    /* JOBA must be one of: C, E, F, G, A, R */
    if (!(rowpiv || l2rank || l2aber || errest ||
          joba[0] == 'C' || joba[0] == 'c')) {
        *info = -1;
    }
    /* JOBU must be one of: U, F, W, N */
    else if (!(lsvec || jobu[0] == 'N' || jobu[0] == 'n' ||
                        jobu[0] == 'W' || jobu[0] == 'w')) {
        *info = -2;
    }
    /* JOBV must be one of: V, J, W, N; and if J then JOBU must not be N */
    else if (!(rsvec || jobv[0] == 'N' || jobv[0] == 'n' ||
                        jobv[0] == 'W' || jobv[0] == 'w') ||
             (jracc && !lsvec)) {
        *info = -3;
    }
    /* JOBR must be one of: R, N */
    else if (!(l2kill || defr)) {
        *info = -4;
    }
    /* JOBT must be one of: T, N */
    else if (!(l2tran || jobt[0] == 'N' || jobt[0] == 'n')) {
        *info = -5;
    }
    /* JOBP must be one of: P, N */
    else if (!(l2pert || jobp[0] == 'N' || jobp[0] == 'n')) {
        *info = -6;
    }
    else if (m < 0) {
        *info = -7;
    }
    else if (n < 0 || n > m) {
        *info = -8;
    }
    else if (lda < m) {
        *info = -10;
    }
    else if (lsvec && ldu < m) {
        *info = -13;
    }
    else if (rsvec && ldv < n) {
        *info = -15;
    }
    /* Workspace validation: 6-case OR condition */
    else if (
        /* Case 1: Only SVals, no ERREST */
        (!lsvec && !rsvec && !errest && lwork < max3i(7, 4*n+1, 2*m+n)) ||
        /* Case 2: Only SVals + ERREST */
        (!lsvec && !rsvec && errest && lwork < max3i(7, 4*n+n*n, 2*m+n)) ||
        /* Case 3: U only */
        (lsvec && !rsvec && lwork < max3i(7, 2*m+n, 4*n+1)) ||
        /* Case 4: V only */
        (rsvec && !lsvec && lwork < max3i(7, 2*m+n, 4*n+1)) ||
        /* Case 5: Full SVD, no Jacobi accumulation */
        (lsvec && rsvec && !jracc && lwork < (2*m+n > 6*n+2*n*n ? 2*m+n : 6*n+2*n*n)) ||
        /* Case 6: Full SVD + Jacobi accumulation */
        (lsvec && rsvec && jracc && lwork < max3i(2*m+n, 4*n+n*n, 2*n+n*n+6))
    ) {
        *info = -17;
    }

    if (*info != 0) {
        xerbla("SGEJSV", -(*info));
        return;
    }

    /* -------------------------------------------------------------------- */
    /* Chunk 3: Quick returns                                               */
    /* -------------------------------------------------------------------- */

    /* Quick return for void matrix */
    if (m == 0 || n == 0) {
        iwork[0] = 0;
        iwork[1] = 0;
        iwork[2] = 0;
        for (p = 0; p < 7; p++) work[p] = ZERO;
        return;
    }

    /* Determine whether U should be M x N or M x M */
    n1 = n;
    if (lsvec && (jobu[0] == 'F' || jobu[0] == 'f')) {
        n1 = m;
    }

    /* Set numerical parameters */
    epsln = slamch("E");
    sfmin = slamch("S");
    small = sfmin / epsln;
    big   = slamch("O");

    /* -------------------------------------------------------------------- */
    /* Chunk 4: Initialize SVA = ||A e_i||_2, with scaling                  */
    /* -------------------------------------------------------------------- */

    scalem = ONE / sqrtf((f32)m * (f32)n);
    noscal = 1;
    goscal = 1;

    for (p = 0; p < n; p++) {
        aapp = ZERO;
        aaqq = ONE;
        slassq(m, &A[p * lda], 1, &aapp, &aaqq);
        if (aapp > big) {
            *info = -9;
            xerbla("SGEJSV", -(*info));
            return;
        }
        aaqq = sqrtf(aaqq);
        if (aapp < (big / aaqq) && noscal) {
            SVA[p] = aapp * aaqq;
        } else {
            noscal = 0;
            SVA[p] = aapp * (aaqq * scalem);
            if (goscal) {
                goscal = 0;
                cblas_sscal(p, scalem, SVA, 1);
            }
        }
    }

    if (noscal) scalem = ONE;

    /* Find max and min of SVA */
    aapp = ZERO;
    aaqq = big;
    for (p = 0; p < n; p++) {
        if (SVA[p] > aapp) aapp = SVA[p];
        if (SVA[p] != ZERO && SVA[p] < aaqq) aaqq = SVA[p];
    }

    /* Quick return for zero matrix */
    if (aapp == ZERO) {
        if (lsvec) slaset("G", m, n1, ZERO, ONE, U, ldu);
        if (rsvec) slaset("G", n, n, ZERO, ONE, V, ldv);
        work[0] = ONE;
        work[1] = ONE;
        if (errest) work[2] = ONE;
        if (lsvec && rsvec) {
            work[3] = ONE;
            work[4] = ONE;
        }
        if (l2tran) {
            work[5] = ZERO;
            work[6] = ZERO;
        }
        iwork[0] = 0;
        iwork[1] = 0;
        iwork[2] = 0;
        return;
    }

    /* Warning if denormalized column norms detected */
    warning = 0;
    if (aaqq <= sfmin) {
        l2rank = 1;
        l2kill = 1;
        warning = 1;
    }

    /* Quick return for one-column matrix */
    if (n == 1) {
        if (lsvec) {
            slascl("G", 0, 0, SVA[0], scalem, m, 1, A, lda, &ierr);
            slacpy("A", m, 1, A, lda, U, ldu);
            /* Computing all M left singular vectors of the M x 1 matrix */
            if (n1 != n) {
                sgeqrf(m, n, U, ldu, work, &work[n], lwork - n, &ierr);
                sorgqr(m, n1, 1, U, ldu, work, &work[n], lwork - n, &ierr);
                cblas_scopy(m, A, 1, U, 1);
            }
        }
        if (rsvec) {
            V[0] = ONE;
        }
        if (SVA[0] < big * scalem) {
            SVA[0] = SVA[0] / scalem;
            scalem = ONE;
        }
        work[0] = ONE / scalem;
        work[1] = ONE;
        if (SVA[0] != ZERO) {
            iwork[0] = 1;
            if ((SVA[0] / scalem) >= sfmin) {
                iwork[1] = 1;
            } else {
                iwork[1] = 0;
            }
        } else {
            iwork[0] = 0;
            iwork[1] = 0;
        }
        iwork[2] = 0;
        if (errest) work[2] = ONE;
        if (lsvec && rsvec) {
            work[3] = ONE;
            work[4] = ONE;
        }
        if (l2tran) {
            work[5] = ZERO;
            work[6] = ZERO;
        }
        return;
    }

    /* -------------------------------------------------------------------- */
    /* Chunk 5: Transpose decision (Shannon entropy) - L2TRAN               */
    /* -------------------------------------------------------------------- */

    transp = 0;
    l2tran = l2tran && (m == n);  /* Only for square matrices */

    aatmax = -ONE;
    aatmin = big;

    if (rowpiv || l2tran) {
        /* Compute row norms */
        if (l2tran) {
            for (p = 0; p < m; p++) {
                xsc = ZERO;
                temp1 = ONE;
                slassq(n, &A[p], lda, &xsc, &temp1);
                /* SLASSQ gets both the ell_2 and ell_infinity norm */
                work[m + n + p] = xsc * scalem;
                work[n + p] = xsc * (scalem * sqrtf(temp1));
                if (work[n + p] > aatmax) aatmax = work[n + p];
                if (work[n + p] != ZERO && work[n + p] < aatmin) aatmin = work[n + p];
            }
        } else {
            for (p = 0; p < m; p++) {
                work[m + n + p] = scalem * fabsf(A[p + cblas_isamax(n, &A[p], lda) * lda]);
                if (work[m + n + p] > aatmax) aatmax = work[m + n + p];
                if (work[m + n + p] < aatmin) aatmin = work[m + n + p];
            }
        }
    }

    /* Shannon entropy computation for transpose decision */
    entra = ZERO;
    entrat = ZERO;

    if (l2tran) {
        xsc = ZERO;
        temp1 = ONE;
        slassq(n, SVA, 1, &xsc, &temp1);
        temp1 = ONE / temp1;

        entra = ZERO;
        for (p = 0; p < n; p++) {
            big1 = ((SVA[p] / xsc) * (SVA[p] / xsc)) * temp1;
            if (big1 != ZERO) entra = entra + big1 * logf(big1);
        }
        entra = -entra / logf((f32)n);

        /* Same for row norms (A * A^t diagonal) */
        entrat = ZERO;
        for (p = 0; p < m; p++) {
            big1 = ((work[n + p] / xsc) * (work[n + p] / xsc)) * temp1;
            if (big1 != ZERO) entrat = entrat + big1 * logf(big1);
        }
        entrat = -entrat / logf((f32)m);

        /* Decide: if row entropy < column entropy, use transpose */
        transp = (entrat < entra);

        if (transp) {
            /* Transpose A in-place (only for square N x N) */
            for (p = 0; p < n - 1; p++) {
                for (q = p + 1; q < n; q++) {
                    temp1 = A[q + p * lda];
                    A[q + p * lda] = A[p + q * lda];
                    A[p + q * lda] = temp1;
                }
            }
            /* Swap SVA with row norms */
            for (p = 0; p < n; p++) {
                work[m + n + p] = SVA[p];
                SVA[p] = work[n + p];
            }
            /* Swap aapp/aatmax, aaqq/aatmin */
            temp1 = aapp;
            aapp = aatmax;
            temp1 = aaqq;
            aaqq = aatmin;
            /* Swap lsvec/rsvec */
            kill = lsvec;
            lsvec = rsvec;
            rsvec = kill;
            if (lsvec) n1 = n;
            rowpiv = 1;
        }
    }

    /* -------------------------------------------------------------------- */
    /* Chunk 6: Matrix scaling                                              */
    /* -------------------------------------------------------------------- */

    /* Scale the matrix so that its maximal singular value remains less
     * than sqrt(BIG) -- the matrix is scaled so that its maximal column
     * has Euclidean norm equal to sqrt(BIG/N). */

    big1 = sqrtf(big);
    temp1 = sqrtf(big / (f32)n);

    slascl("G", 0, 0, aapp, temp1, n, 1, SVA, n, &ierr);
    if (aaqq > aapp * sfmin) {
        aaqq = (aaqq / aapp) * temp1;
    } else {
        aaqq = (aaqq * temp1) / aapp;
    }

    temp1 = temp1 * scalem;
    slascl("G", 0, 0, aapp, temp1, m, n, A, lda, &ierr);

    /* Store for later unscaling */
    uscal1 = temp1;
    uscal2 = aapp;

    /* Determine kill threshold XSC */
    if (l2kill) {
        xsc = sqrtf(sfmin);
    } else {
        xsc = small;
        /* If matrix is very ill-conditioned, set JRACC flag */
        if (aaqq < sqrtf(sfmin) && lsvec && rsvec) {
            jracc = 1;
        }
    }

    /* Kill columns with SVA[p] < XSC */
    if (aaqq < xsc) {
        for (p = 0; p < n; p++) {
            if (SVA[p] < xsc) {
                slaset("A", m, 1, ZERO, ZERO, &A[p * lda], lda);
                SVA[p] = ZERO;
            }
        }
    }

    /* -------------------------------------------------------------------- */
    /* Chunk 7: Row pivoting + first QR factorization                       */
    /* -------------------------------------------------------------------- */

    if (rowpiv) {
        /* Bjoerck row pivoting: find max-norm rows */
        for (p = 0; p < m - 1; p++) {
            q = cblas_isamax(m - p, &work[m + n + p], 1) + p;
            iwork[2 * n + p] = q;  /* Store pivot index (0-based) */
            if (p != q) {
                temp1 = work[m + n + p];
                work[m + n + p] = work[m + n + q];
                work[m + n + q] = temp1;
            }
        }
        /* Apply row pivots to A */
        slaswp(n, A, lda, 0, m - 2, &iwork[2 * n], 1);
    }

    /* Initialize IWORK[0:n-1] = 0 (all columns free for SGEQP3) */
    for (p = 0; p < n; p++) {
        iwork[p] = 0;
    }

    /* First QR factorization with column pivoting */
    sgeqp3(m, n, A, lda, iwork, work, &work[n], lwork - n, &ierr);

    /* -------------------------------------------------------------------- */
    /* Chunk 8: Rank detection                                              */
    /* -------------------------------------------------------------------- */

    /* Detect numerical rank based on R factor diagonal */
    nr = 1;

    if (l2aber) {
        /* JOBA='A': Aggressive - treat small singular values as noise */
        temp1 = sqrtf((f32)n) * epsln;
        for (p = 1; p < n; p++) {
            if (fabsf(A[p + p * lda]) >= temp1 * fabsf(A[0])) {
                nr++;
            } else {
                break;
            }
        }
    } else if (l2rank) {
        /* JOBA='R': Rank-revealing - look for gap in diagonal */
        temp1 = sqrtf(sfmin);
        for (p = 1; p < n; p++) {
            if (fabsf(A[p + p * lda]) < epsln * fabsf(A[(p-1) + (p-1) * lda]) ||
                fabsf(A[p + p * lda]) < small ||
                (l2kill && fabsf(A[p + p * lda]) < temp1)) {
                break;
            }
            nr++;
        }
    } else {
        /* Default: High relative accuracy, gentle underflow cleanup */
        temp1 = sqrtf(sfmin);
        for (p = 1; p < n; p++) {
            if (fabsf(A[p + p * lda]) < small ||
                (l2kill && fabsf(A[p + p * lda]) < temp1)) {
                break;
            }
            nr++;
        }
    }

    /* -------------------------------------------------------------------- */
    /* Chunk 9: ALMORT check + SCONDA computation                           */
    /* -------------------------------------------------------------------- */

    almort = 0;
    if (nr == n) {
        maxprj = ONE;
        for (p = 1; p < n; p++) {
            temp1 = fabsf(A[p + p * lda]) / SVA[iwork[p]];
            if (temp1 < maxprj) maxprj = temp1;
        }
        if (maxprj * maxprj >= ONE - (f32)n * epsln) {
            almort = 1;
        }
    }

    /* Initialize condition estimates */
    sconda = -ONE;
    condr1 = -ONE;
    condr2 = -ONE;

    /* SCONDA computation (if ERREST and full rank) */
    if (errest && nr == n) {
        if (rsvec) {
            /* Use V as workspace */
            slacpy("U", n, n, A, lda, V, ldv);
            for (p = 0; p < n; p++) {
                temp1 = SVA[iwork[p]];
                cblas_sscal(p + 1, ONE / temp1, &V[p * ldv], 1);
            }
            spocon("U", n, V, ldv, ONE, &temp1, &work[n], &iwork[2*n + m], &ierr);
        } else if (lsvec) {
            /* Use U as workspace */
            slacpy("U", n, n, A, lda, U, ldu);
            for (p = 0; p < n; p++) {
                temp1 = SVA[iwork[p]];
                cblas_sscal(p + 1, ONE / temp1, &U[p * ldu], 1);
            }
            spocon("U", n, U, ldu, ONE, &temp1, &work[n], &iwork[2*n + m], &ierr);
        } else {
            /* Use WORK as workspace */
            slacpy("U", n, n, A, lda, &work[n], n);
            for (p = 0; p < n; p++) {
                temp1 = SVA[iwork[p]];
                cblas_sscal(p + 1, ONE / temp1, &work[n + p * n], 1);
            }
            spocon("U", n, &work[n], n, ONE, &temp1, &work[n + n*n], &iwork[2*n + m], &ierr);
        }
        sconda = ONE / sqrtf(temp1);
    }

    /* If there is no violent scaling, artificial perturbation is not needed */
    l2pert = l2pert && (fabsf(A[0] / A[(nr-1) + (nr-1) * lda]) > sqrtf(big1));

    /* -------------------------------------------------------------------- */
    /* Phase 4: SVD computation branches                                    */
    /* -------------------------------------------------------------------- */


    if (!rsvec && !lsvec) {
        /* ============================================================== */
        /* Branch A: Singular values only                                 */
        /* ============================================================== */

        /* Transpose R1 to lower triangular form */
        for (p = 0; p < (n - 1 < nr ? n - 1 : nr); p++) {
            cblas_scopy(n - p - 1, &A[p + (p + 1) * lda], lda, &A[(p + 1) + p * lda], 1);
        }

        /* Clear upper triangle */
        if (!almort) {
            /* Apply perturbation if L2PERT */
            if (l2pert) {
                /* NOTE: Fortran uses EPSLN/N, not sqrt(SMALL) */
                xsc = epsln / (f32)n;
                for (q = 0; q < nr; q++) {
                    temp1 = xsc * fabsf(A[q + q * lda]);
                    for (p = 0; p < nr; p++) {
                        if (p > q || A[p + q * lda] == ZERO) {
                            A[p + q * lda] = copysignf(temp1, A[p + q * lda]);
                        }
                    }
                }
            } else {
                slaset("U", nr - 1, nr - 1, ZERO, ZERO, &A[1 * lda], lda);
            }

            /* Second QR factorization */
            sgeqrf(n, nr, A, lda, work, &work[n], lwork - n, &ierr);

            /* Transpose to lower triangular */
            for (p = 0; p < nr - 1; p++) {
                cblas_scopy(nr - p - 1, &A[p + (p + 1) * lda], lda, &A[(p + 1) + p * lda], 1);
            }
        }

        /* Final perturbation */
        if (l2pert) {
            /* NOTE: Fortran uses EPSLN/N, not sqrt(SMALL) */
            xsc = epsln / (f32)n;
            for (q = 0; q < nr; q++) {
                temp1 = xsc * fabsf(A[q + q * lda]);
                for (p = 0; p < nr; p++) {
                    if (p > q || A[p + q * lda] == ZERO) {
                        A[p + q * lda] = copysignf(temp1, A[p + q * lda]);
                    }
                }
            }
        } else {
            slaset("U", nr - 1, nr - 1, ZERO, ZERO, &A[1 * lda], lda);
        }

        /* Call SGESVJ for singular values only */
        sgesvj("L", "N", "N", nr, nr, A, lda, SVA, n, V, ldv,
               work, lwork, info);
        scalem = work[0];
        numrank = (int)(work[1] + 0.5f);

    } else if (rsvec && !lsvec) {
        /* ============================================================== */
        /* Branch B: Right singular vectors only                          */
        /* ============================================================== */

        if (almort) {
            /* In this case NR equals N */
            for (p = 0; p < nr; p++) {
                cblas_scopy(n - p, &A[p + p * lda], lda, &V[p + p * ldv], 1);
            }
            slaset("U", nr - 1, nr - 1, ZERO, ZERO, &V[1 * ldv], ldv);

            sgesvj("L", "U", "N", n, nr, V, ldv, SVA, nr, A, lda,
                   work, lwork, info);
            scalem = work[0];
            numrank = (int)(work[1] + 0.5f);

        } else {
            /* Two more QR factorizations */
            slaset("L", nr - 1, nr - 1, ZERO, ZERO, &A[1], lda);
            sgelqf(nr, n, A, lda, work, &work[n], lwork - n, &ierr);
            slacpy("L", nr, nr, A, lda, V, ldv);
            slaset("U", nr - 1, nr - 1, ZERO, ZERO, &V[1 * ldv], ldv);
            sgeqrf(nr, nr, V, ldv, &work[n], &work[2 * n], lwork - 2 * n, &ierr);
            for (p = 0; p < nr; p++) {
                cblas_scopy(nr - p, &V[p + p * ldv], ldv, &V[p + p * ldv], 1);
            }
            slaset("U", nr - 1, nr - 1, ZERO, ZERO, &V[1 * ldv], ldv);

            sgesvj("L", "U", "N", nr, nr, V, ldv, SVA, nr, U, ldu,
                   &work[n], lwork - n, info);
            scalem = work[n];
            numrank = (int)(work[n + 1] + 0.5f);

            if (nr < n) {
                slaset("A", n - nr, nr, ZERO, ZERO, &V[nr], ldv);
                slaset("A", nr, n - nr, ZERO, ZERO, &V[nr * ldv], ldv);
                slaset("A", n - nr, n - nr, ZERO, ONE, &V[nr + nr * ldv], ldv);
            }

            sormlq("L", "T", n, n, nr, A, lda, work, V, ldv, &work[n], lwork - n, &ierr);
        }

        /* Permute rows of V by first QRP column pivots */
        for (p = 0; p < n; p++) {
            cblas_scopy(n, &V[p], ldv, &A[iwork[p]], lda);
        }
        slacpy("A", n, n, A, lda, V, ldv);

        if (transp) {
            slacpy("A", n, n, V, ldv, U, ldu);
        }

    } else if (lsvec && !rsvec) {
        /* ============================================================== */
        /* Branch C: Left singular vectors only                           */
        /* ============================================================== */

        /* Copy R1 to U */
        for (p = 0; p < nr; p++) {
            cblas_scopy(n - p, &A[p + p * lda], lda, &U[p + p * ldu], 1);
        }
        slaset("U", nr - 1, nr - 1, ZERO, ZERO, &U[1 * ldu], ldu);

        /* Second QRF preconditioning */
        sgeqrf(n, nr, U, ldu, &work[n], &work[2 * n], lwork - 2 * n, &ierr);

        /* Transpose to lower triangular */
        for (p = 0; p < nr - 1; p++) {
            cblas_scopy(nr - p - 1, &U[p + (p + 1) * ldu], ldu, &U[(p + 1) + p * ldu], 1);
        }
        slaset("U", nr - 1, nr - 1, ZERO, ZERO, &U[1 * ldu], ldu);

        /* Jacobi SVD on lower triangular */
        sgesvj("L", "U", "N", nr, nr, U, ldu, SVA, nr, A, lda,
               &work[n], lwork - n, info);
        scalem = work[n];
        numrank = (int)(work[n + 1] + 0.5f);

        /* Pad U if needed */
        if (nr < m) {
            slaset("A", m - nr, nr, ZERO, ZERO, &U[nr], ldu);
            if (nr < n1) {
                slaset("A", nr, n1 - nr, ZERO, ZERO, &U[nr * ldu], ldu);
                slaset("A", m - nr, n1 - nr, ZERO, ONE, &U[nr + nr * ldu], ldu);
            }
        }

        /* Apply Q from first QRF */
        sormqr("L", "N", m, n1, n, A, lda, work, U, ldu, &work[n], lwork - n, &ierr);

        /* Undo row pivoting */
        if (rowpiv) {
            slaswp(n1, U, ldu, 0, m - 2, &iwork[2 * n], -1);
        }

        /* Normalize columns of U */
        for (p = 0; p < n1; p++) {
            xsc = ONE / cblas_snrm2(m, &U[p * ldu], 1);
            cblas_sscal(m, xsc, &U[p * ldu], 1);
        }

        if (transp) {
            slacpy("A", n, n, U, ldu, V, ldv);
        }

    } else {
        /* ============================================================== */
        /* Branch D: Full SVD (both U and V)                              */
        /* ============================================================== */

        if (!jracc) {
            /* ---------------------------------------------------------- */
            /* Branch D.1: Non-JRACC path                                 */
            /* ---------------------------------------------------------- */

            if (!almort) {
                /* Copy R1 to V */
                for (p = 0; p < nr; p++) {
                    cblas_scopy(n - p, &A[p + p * lda], lda, &V[p + p * ldv], 1);
                }

                /* Perturbation */
                if (l2pert) {
                    xsc = sqrtf(small);
                    for (q = 0; q < nr; q++) {
                        temp1 = xsc * fabsf(V[q + q * ldv]);
                        for (p = 0; p < n; p++) {
                            if ((p > q && fabsf(V[p + q * ldv]) <= temp1) || p < q) {
                                V[p + q * ldv] = copysignf(temp1, V[p + q * ldv]);
                            }
                            if (p < q) V[p + q * ldv] = -V[p + q * ldv];
                        }
                    }
                } else {
                    slaset("U", nr - 1, nr - 1, ZERO, ZERO, &V[1 * ldv], ldv);
                }

                /* Estimate condition number of R1 */
                slacpy("L", nr, nr, V, ldv, &work[2 * n], nr);
                for (p = 0; p < nr; p++) {
                    temp1 = cblas_snrm2(nr - p, &work[2 * n + p * nr + p], 1);
                    cblas_sscal(nr - p, ONE / temp1, &work[2 * n + p * nr + p], 1);
                }
                spocon("L", nr, &work[2 * n], nr, ONE, &temp1,
                       &work[2 * n + nr * nr], &iwork[m + 2 * n], &ierr);
                condr1 = ONE / sqrtf(temp1);

                cond_ok = sqrtf((f32)nr);

                if (condr1 < cond_ok) {
                    /* R1 is well-conditioned: second QRF without pivoting */
                    sgeqrf(n, nr, V, ldv, &work[n], &work[2 * n], lwork - 2 * n, &ierr);

                    if (l2pert) {
                        xsc = sqrtf(small) / epsln;
                        for (p = 1; p < nr; p++) {
                            for (q = 0; q < p; q++) {
                                temp1 = xsc * fminf(fabsf(V[p + p * ldv]), fabsf(V[q + q * ldv]));
                                if (fabsf(V[q + p * ldv]) <= temp1) {
                                    V[q + p * ldv] = copysignf(temp1, V[q + p * ldv]);
                                }
                            }
                        }
                    }

                    if (nr != n) {
                        slacpy("A", n, nr, V, ldv, &work[2 * n], n);
                    }

                    /* Transpose upper to lower triangular */
                    for (p = 0; p < nr - 1; p++) {
                        cblas_scopy(nr - p - 1, &V[p + (p + 1) * ldv], ldv, &V[(p + 1) + p * ldv], 1);
                    }

                    condr2 = condr1;

                } else {
                    /* Ill-conditioned: second QRF with pivoting */
                    for (p = 0; p < nr; p++) {
                        iwork[n + p] = 0;
                    }
                    sgeqp3(n, nr, V, ldv, &iwork[n], &work[n], &work[2 * n], lwork - 2 * n, &ierr);

                    if (l2pert) {
                        xsc = sqrtf(small);
                        for (p = 1; p < nr; p++) {
                            for (q = 0; q < p; q++) {
                                temp1 = xsc * fminf(fabsf(V[p + p * ldv]), fabsf(V[q + q * ldv]));
                                if (fabsf(V[q + p * ldv]) <= temp1) {
                                    V[q + p * ldv] = copysignf(temp1, V[q + p * ldv]);
                                }
                            }
                        }
                    }

                    slacpy("A", n, nr, V, ldv, &work[2 * n], n);

                    if (l2pert) {
                        xsc = sqrtf(small);
                        for (p = 1; p < nr; p++) {
                            for (q = 0; q < p; q++) {
                                temp1 = xsc * fminf(fabsf(V[p + p * ldv]), fabsf(V[q + q * ldv]));
                                V[p + q * ldv] = -copysignf(temp1, V[q + p * ldv]);
                            }
                        }
                    } else {
                        slaset("L", nr - 1, nr - 1, ZERO, ZERO, &V[1], ldv);
                    }

                    /* LQ factorization for R2 */
                    sgelqf(nr, nr, V, ldv, &work[2 * n + n * nr],
                           &work[2 * n + n * nr + nr], lwork - 2 * n - n * nr - nr, &ierr);

                    /* Estimate condition number */
                    slacpy("L", nr, nr, V, ldv, &work[2 * n + n * nr + nr], nr);
                    for (p = 0; p < nr; p++) {
                        temp1 = cblas_snrm2(p + 1, &work[2 * n + n * nr + nr + p], nr);
                        cblas_sscal(p + 1, ONE / temp1, &work[2 * n + n * nr + nr + p], nr);
                    }
                    spocon("L", nr, &work[2 * n + n * nr + nr], nr, ONE, &temp1,
                           &work[2 * n + n * nr + nr + nr * nr], &iwork[m + 2 * n], &ierr);
                    condr2 = ONE / sqrtf(temp1);

                    if (condr2 >= cond_ok) {
                        /* Save Householder vectors for Q3 */
                        slacpy("U", nr, nr, V, ldv, &work[2 * n], n);
                    }
                }

                /* Clear upper triangle and add perturbation */
                if (l2pert) {
                    xsc = sqrtf(small);
                    for (q = 1; q < nr; q++) {
                        temp1 = xsc * V[q + q * ldv];
                        for (p = 0; p < q; p++) {
                            V[p + q * ldv] = -copysignf(temp1, V[p + q * ldv]);
                        }
                    }
                } else {
                    slaset("U", nr - 1, nr - 1, ZERO, ZERO, &V[1 * ldv], ldv);
                }

                /* SVD computation based on condition numbers */
                if (condr1 < cond_ok) {
                    /* Case D.1a: CONDR1 < COND_OK */
                    sgesvj("L", "U", "N", nr, nr, V, ldv, SVA, nr, U, ldu,
                           &work[2 * n + n * nr + nr], lwork - 2 * n - n * nr - nr, info);
                    scalem = work[2 * n + n * nr + nr];
                    numrank = (int)(work[2 * n + n * nr + nr + 1] + 0.5f);

                    for (p = 0; p < nr; p++) {
                        cblas_scopy(nr, &V[p * ldv], 1, &U[p * ldu], 1);
                        cblas_sscal(nr, SVA[p], &V[p * ldv], 1);
                    }

                    if (nr == n) {
                        cblas_strsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
                                    nr, nr, ONE, A, lda, V, ldv);
                    } else {
                        cblas_strsm(CblasColMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit,
                                    nr, nr, ONE, &work[2 * n], n, V, ldv);
                        if (nr < n) {
                            slaset("A", n - nr, nr, ZERO, ZERO, &V[nr], ldv);
                            slaset("A", nr, n - nr, ZERO, ZERO, &V[nr * ldv], ldv);
                            slaset("A", n - nr, n - nr, ZERO, ONE, &V[nr + nr * ldv], ldv);
                        }
                        sormqr("L", "N", n, n, nr, &work[2 * n], n, &work[n],
                               V, ldv, &work[2 * n + n * nr + nr], lwork - 2 * n - n * nr - nr, &ierr);
                    }

                } else if (condr2 < cond_ok) {
                    /* Case D.1b: CONDR2 < COND_OK */
                    sgesvj("L", "U", "N", nr, nr, V, ldv, SVA, nr, U, ldu,
                           &work[2 * n + n * nr + nr], lwork - 2 * n - n * nr - nr, info);
                    scalem = work[2 * n + n * nr + nr];
                    numrank = (int)(work[2 * n + n * nr + nr + 1] + 0.5f);

                    for (p = 0; p < nr; p++) {
                        cblas_scopy(nr, &V[p * ldv], 1, &U[p * ldu], 1);
                        cblas_sscal(nr, SVA[p], &U[p * ldu], 1);
                    }

                    cblas_strsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
                                nr, nr, ONE, &work[2 * n], n, U, ldu);

                    /* Apply permutation from second QRP */
                    for (q = 0; q < nr; q++) {
                        for (p = 0; p < nr; p++) {
                            work[2 * n + n * nr + nr + iwork[n + p]] = U[p + q * ldu];
                        }
                        for (p = 0; p < nr; p++) {
                            U[p + q * ldu] = work[2 * n + n * nr + nr + p];
                        }
                    }

                    if (nr < n) {
                        slaset("A", n - nr, nr, ZERO, ZERO, &V[nr], ldv);
                        slaset("A", nr, n - nr, ZERO, ZERO, &V[nr * ldv], ldv);
                        slaset("A", n - nr, n - nr, ZERO, ONE, &V[nr + nr * ldv], ldv);
                    }
                    sormqr("L", "N", n, n, nr, &work[2 * n], n, &work[n],
                           V, ldv, &work[2 * n + n * nr + nr], lwork - 2 * n - n * nr - nr, &ierr);

                } else {
                    /* Case D.1c: Last line of defense */
                    sgesvj("L", "U", "V", nr, nr, V, ldv, SVA, nr, U, ldu,
                           &work[2 * n + n * nr + nr], lwork - 2 * n - n * nr - nr, info);
                    scalem = work[2 * n + n * nr + nr];
                    numrank = (int)(work[2 * n + n * nr + nr + 1] + 0.5f);

                    if (nr < n) {
                        slaset("A", n - nr, nr, ZERO, ZERO, &V[nr], ldv);
                        slaset("A", nr, n - nr, ZERO, ZERO, &V[nr * ldv], ldv);
                        slaset("A", n - nr, n - nr, ZERO, ONE, &V[nr + nr * ldv], ldv);
                    }
                    sormqr("L", "N", n, n, nr, &work[2 * n], n, &work[n],
                           V, ldv, &work[2 * n + n * nr + nr], lwork - 2 * n - n * nr - nr, &ierr);

                    sormlq("L", "T", nr, nr, nr, &work[2 * n], n,
                           &work[2 * n + n * nr], U, ldu,
                           &work[2 * n + n * nr + nr], lwork - 2 * n - n * nr - nr, &ierr);

                    /* Apply permutation from second QRP */
                    for (q = 0; q < nr; q++) {
                        for (p = 0; p < nr; p++) {
                            work[2 * n + n * nr + nr + iwork[n + p]] = U[p + q * ldu];
                        }
                        for (p = 0; p < nr; p++) {
                            U[p + q * ldu] = work[2 * n + n * nr + nr + p];
                        }
                    }
                }

                /* Permute rows of V by first QRP column pivots and normalize */
                temp1 = sqrtf((f32)n) * epsln;
                for (q = 0; q < n; q++) {
                    for (p = 0; p < n; p++) {
                        work[2 * n + n * nr + nr + iwork[p]] = V[p + q * ldv];
                    }
                    for (p = 0; p < n; p++) {
                        V[p + q * ldv] = work[2 * n + n * nr + nr + p];
                    }
                    xsc = ONE / cblas_snrm2(n, &V[q * ldv], 1);
                    if (xsc < (ONE - temp1) || xsc > (ONE + temp1)) {
                        cblas_sscal(n, xsc, &V[q * ldv], 1);
                    }
                }

                /* Assemble U */
                if (nr < m) {
                    slaset("A", m - nr, nr, ZERO, ZERO, &U[nr], ldu);
                    if (nr < n1) {
                        slaset("A", nr, n1 - nr, ZERO, ZERO, &U[nr * ldu], ldu);
                        slaset("A", m - nr, n1 - nr, ZERO, ONE, &U[nr + nr * ldu], ldu);
                    }
                }

                sormqr("L", "N", m, n1, n, A, lda, work, U, ldu, &work[n], lwork - n, &ierr);

                /* Normalize U columns */
                temp1 = sqrtf((f32)m) * epsln;
                for (p = 0; p < nr; p++) {
                    xsc = ONE / cblas_snrm2(m, &U[p * ldu], 1);
                    if (xsc < (ONE - temp1) || xsc > (ONE + temp1)) {
                        cblas_sscal(m, xsc, &U[p * ldu], 1);
                    }
                }

                /* Undo row pivoting */
                if (rowpiv) {
                    slaswp(n1, U, ldu, 0, m - 2, &iwork[2 * n], -1);
                }

            } else {
                /* -------------------------------------------------------- */
                /* Branch D.1 ALMORT: columns almost orthogonal             */
                /* -------------------------------------------------------- */

                slacpy("U", n, n, A, lda, &work[n], n);

                if (l2pert) {
                    xsc = sqrtf(small);
                    for (p = 1; p < n; p++) {
                        /* NOTE: Fortran WORK(N+(p-1)*N+p) = element (p,p) = diagonal
                         * In C 0-based: work[n + p*n + p] for diagonal (p,p) */
                        temp1 = xsc * work[n + p * n + p];
                        for (q = 0; q < p; q++) {
                            /* LHS: element (p, q), RHS: element (q, p) */
                            work[n + q * n + p] = -copysignf(temp1, work[n + p * n + q]);
                        }
                    }
                } else {
                    slaset("L", n - 1, n - 1, ZERO, ZERO, &work[n + 1], n);
                }

                sgesvj("U", "U", "N", n, n, &work[n], n, SVA, n, U, ldu,
                       &work[n + n * n], lwork - n - n * n, info);
                scalem = work[n + n * n];
                numrank = (int)(work[n + n * n + 1] + 0.5f);

                for (p = 0; p < n; p++) {
                    cblas_scopy(n, &work[n + p * n], 1, &U[p * ldu], 1);
                    cblas_sscal(n, SVA[p], &work[n + p * n], 1);
                }

                cblas_strsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
                            n, n, ONE, A, lda, &work[n], n);

                for (p = 0; p < n; p++) {
                    cblas_scopy(n, &work[n + p], n, &V[iwork[p]], ldv);
                }

                temp1 = sqrtf((f32)n) * epsln;
                for (p = 0; p < n; p++) {
                    xsc = ONE / cblas_snrm2(n, &V[p * ldv], 1);
                    if (xsc < (ONE - temp1) || xsc > (ONE + temp1)) {
                        cblas_sscal(n, xsc, &V[p * ldv], 1);
                    }
                }

                /* Assemble U */
                if (n < m) {
                    slaset("A", m - n, n, ZERO, ZERO, &U[n], ldu);
                    if (n < n1) {
                        slaset("A", n, n1 - n, ZERO, ZERO, &U[n * ldu], ldu);
                        slaset("A", m - n, n1 - n, ZERO, ONE, &U[n + n * ldu], ldu);
                    }
                }

                sormqr("L", "N", m, n1, n, A, lda, work, U, ldu, &work[n], lwork - n, &ierr);

                temp1 = sqrtf((f32)m) * epsln;
                for (p = 0; p < n1; p++) {
                    xsc = ONE / cblas_snrm2(m, &U[p * ldu], 1);
                    if (xsc < (ONE - temp1) || xsc > (ONE + temp1)) {
                        cblas_sscal(m, xsc, &U[p * ldu], 1);
                    }
                }

                if (rowpiv) {
                    slaswp(n1, U, ldu, 0, m - 2, &iwork[2 * n], -1);
                }
            }

        } else {
            /* ---------------------------------------------------------- */
            /* Branch D.2: JRACC path - explicit Jacobi accumulation      */
            /* ---------------------------------------------------------- */

            for (p = 0; p < nr; p++) {
                cblas_scopy(n - p, &A[p + p * lda], lda, &V[p + p * ldv], 1);
            }

            if (l2pert) {
                xsc = sqrtf(small / epsln);
                for (q = 0; q < nr; q++) {
                    temp1 = xsc * fabsf(V[q + q * ldv]);
                    for (p = 0; p < n; p++) {
                        if ((p > q && fabsf(V[p + q * ldv]) <= temp1) || p < q) {
                            V[p + q * ldv] = copysignf(temp1, V[p + q * ldv]);
                        }
                        if (p < q) V[p + q * ldv] = -V[p + q * ldv];
                    }
                }
            } else {
                slaset("U", nr - 1, nr - 1, ZERO, ZERO, &V[1 * ldv], ldv);
            }

            sgeqrf(n, nr, V, ldv, &work[n], &work[2 * n], lwork - 2 * n, &ierr);
            slacpy("L", n, nr, V, ldv, &work[2 * n], n);

            for (p = 0; p < nr; p++) {
                cblas_scopy(nr - p, &V[p + p * ldv], ldv, &U[p + p * ldu], 1);
            }

            if (l2pert) {
                xsc = sqrtf(small / epsln);
                for (q = 1; q < nr; q++) {
                    for (p = 0; p < q; p++) {
                        temp1 = xsc * fminf(fabsf(U[p + p * ldu]), fabsf(U[q + q * ldu]));
                        U[p + q * ldu] = -copysignf(temp1, U[q + p * ldu]);
                    }
                }
            } else {
                slaset("U", nr - 1, nr - 1, ZERO, ZERO, &U[1 * ldu], ldu);
            }

            sgesvj("G", "U", "V", nr, nr, U, ldu, SVA, n, V, ldv,
                   &work[2 * n + n * nr], lwork - 2 * n - n * nr, info);
            scalem = work[2 * n + n * nr];
            numrank = (int)(work[2 * n + n * nr + 1] + 0.5f);

            if (nr < n) {
                slaset("A", n - nr, nr, ZERO, ZERO, &V[nr], ldv);
                slaset("A", nr, n - nr, ZERO, ZERO, &V[nr * ldv], ldv);
                slaset("A", n - nr, n - nr, ZERO, ONE, &V[nr + nr * ldv], ldv);
            }

            sormqr("L", "N", n, n, nr, &work[2 * n], n, &work[n],
                   V, ldv, &work[2 * n + n * nr + nr], lwork - 2 * n - n * nr - nr, &ierr);

            /* Permute rows and normalize */
            temp1 = sqrtf((f32)n) * epsln;
            for (q = 0; q < n; q++) {
                for (p = 0; p < n; p++) {
                    work[2 * n + n * nr + nr + iwork[p]] = V[p + q * ldv];
                }
                for (p = 0; p < n; p++) {
                    V[p + q * ldv] = work[2 * n + n * nr + nr + p];
                }
                xsc = ONE / cblas_snrm2(n, &V[q * ldv], 1);
                if (xsc < (ONE - temp1) || xsc > (ONE + temp1)) {
                    cblas_sscal(n, xsc, &V[q * ldv], 1);
                }
            }

            /* Assemble U */
            if (nr < m) {
                slaset("A", m - nr, nr, ZERO, ZERO, &U[nr], ldu);
                if (nr < n1) {
                    slaset("A", nr, n1 - nr, ZERO, ZERO, &U[nr * ldu], ldu);
                    slaset("A", m - nr, n1 - nr, ZERO, ONE, &U[nr + nr * ldu], ldu);
                }
            }

            sormqr("L", "N", m, n1, n, A, lda, work, U, ldu, &work[n], lwork - n, &ierr);

            if (rowpiv) {
                slaswp(n1, U, ldu, 0, m - 2, &iwork[2 * n], -1);
            }
        }

        /* Swap U and V if we worked on A^t */
        if (transp) {
            for (p = 0; p < n; p++) {
                cblas_sswap(n, &U[p * ldu], 1, &V[p * ldv], 1);
            }
        }
    }

    /* -------------------------------------------------------------------- */
    /* Chunk 15: Post-processing                                            */
    /* -------------------------------------------------------------------- */

    /* Undo scaling if possible */
    if (uscal2 <= (big / SVA[0]) * uscal1) {
        slascl("G", 0, 0, uscal1, uscal2, nr, 1, SVA, n, &ierr);
        uscal1 = ONE;
        uscal2 = ONE;
    }

    /* Zero SVA beyond NR */
    for (p = nr; p < n; p++) {
        SVA[p] = ZERO;
    }

    /* Fill output arrays */
    work[0] = uscal2 * scalem;
    work[1] = uscal1;
    if (errest) work[2] = sconda;
    if (lsvec && rsvec) {
        work[3] = condr1;
        work[4] = condr2;
    }
    if (l2tran) {
        work[5] = entra;
        work[6] = entrat;
    }

    iwork[0] = nr;
    iwork[1] = numrank;
    iwork[2] = warning;
}
