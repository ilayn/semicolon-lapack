/**
 * @file ztzrzf.c
 * @brief ZTZRZF reduces an upper trapezoidal matrix to upper triangular form.
 */

#include "internal_build_defs.h"
#include <complex.h>
#include <cblas.h>
#include "../include/lapack_tuning.h"
#include "semicolon_lapack_complex_double.h"

/**
 * ZTZRZF reduces the M-by-N ( M<=N ) complex upper trapezoidal matrix A
 * to upper triangular form by means of unitary transformations.
 *
 * The upper trapezoidal matrix A is factored as
 *
 *    A = ( R  0 ) * Z,
 *
 * where Z is an N-by-N unitary matrix and R is an M-by-M upper
 * triangular matrix.
 *
 * @param[in]     m     The number of rows of the matrix A. m >= 0.
 * @param[in]     n     The number of columns of the matrix A. n >= m.
 * @param[in,out] A     Double complex array, dimension (lda, n).
 *                      On entry, the leading M-by-N upper trapezoidal part of the
 *                      array A must contain the matrix to be factorized.
 *                      On exit, the leading M-by-M upper triangular part of A
 *                      contains the upper triangular matrix R, and elements M+1 to
 *                      N of the first M rows of A, with the array TAU, represent the
 *                      unitary matrix Z as a product of M elementary reflectors.
 * @param[in]     lda   The leading dimension of the array A. lda >= max(1, m).
 * @param[out]    tau   Double complex array, dimension (m).
 *                      The scalar factors of the elementary reflectors.
 * @param[out]    work  Double complex array, dimension (max(1, lwork)).
 *                      On exit, if info = 0, work[0] returns the optimal lwork.
 * @param[in]     lwork The dimension of the array work. lwork >= max(1, m).
 *                      For optimum performance lwork >= m*nb, where nb is
 *                      the optimal blocksize.
 *                      If lwork = -1, then a workspace query is assumed; the routine
 *                      only calculates the optimal size of the work array, returns
 *                      this value as the first entry of the work array, and no error
 *                      message related to lwork is issued by xerbla.
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value.
 */
void ztzrzf(const INT m, const INT n,
            c128* restrict A, const INT lda,
            c128* restrict tau,
            c128* restrict work, const INT lwork,
            INT* info)
{
    INT i, ib, iws, ki, kk, ldwork, lwkmin, lwkopt, mu, nb, nbmin, nx;
    INT lquery;

    /* Parameter validation */
    *info = 0;
    lquery = (lwork == -1);

    if (m < 0) {
        *info = -1;
    } else if (n < m) {
        *info = -2;
    } else if (lda < (m > 1 ? m : 1)) {
        *info = -4;
    }

    if (*info == 0) {
        if (m == 0 || m == n) {
            lwkopt = 1;
            lwkmin = 1;
        } else {
            /* Determine the block size.
             * Fortran: NB = ILAENV(1, 'ZGERQF', ' ', M, N, -1, -1) */
            nb = lapack_get_nb("GERQF");
            lwkopt = m * nb;
            lwkmin = m > 1 ? m : 1;
        }
        work[0] = (c128)lwkopt;

        if (lwork < lwkmin && !lquery) {
            *info = -7;
        }
    }

    if (*info != 0) {
        xerbla("ZTZRZF", -(*info));
        return;
    }

    if (lquery) {
        return;
    }

    /* Quick return if possible */
    if (m == 0) {
        return;
    }

    if (m == n) {
        for (i = 0; i < n; i++) {
            tau[i] = CMPLX(0.0, 0.0);
        }
        return;
    }

    nbmin = 2;
    nx = 1;

    if (nb > 1 && nb < m) {

        /* Determine when to cross over from blocked to unblocked code.
         * Fortran: NX = MAX(0, ILAENV(3, 'ZGERQF', ' ', M, N, -1, -1)) */
        nx = lapack_get_nx("GERQF");
        if (nx < 0) nx = 0;

        if (nx < m) {

            /* Determine if workspace is large enough for blocked code */
            ldwork = m;
            iws = ldwork * nb;
            if (lwork < iws) {

                /* Not enough workspace to use optimal NB: reduce NB and
                 * determine the minimum value of NB.
                 * Fortran: NB = LWORK / LDWORK
                 *          NBMIN = MAX(2, ILAENV(2, 'ZGERQF', ' ', M, N, -1, -1)) */
                nb = lwork / ldwork;
                nbmin = lapack_get_nbmin("GERQF");
                if (nbmin < 2) nbmin = 2;
            }
        }
    }

    if (nb >= nbmin && nb < m && nx < m) {

        /* Use blocked code initially.
         * The last kk rows are handled by the block method.
         *
         * Fortran: M1 = MIN(M+1, N)  -- 1-based column index M+1
         *          In 0-based, column index m (since M+1 in 1-based = m in 0-based)
         *          KI = ((M-NX-1)/NB)*NB
         *          KK = MIN(M, KI+NB)
         */
        ki = ((m - nx - 1) / nb) * nb;
        kk = m < (ki + nb) ? m : (ki + nb);

        /* Blocked loop, going backward.
         * Fortran: DO I = M-KK+KI+1, M-KK+1, -NB  (1-based)
         * In 0-based: for (i = m-kk+ki; i >= m-kk; i -= nb)
         */
        for (i = m - kk + ki; i >= m - kk; i -= nb) {
            ib = (m - i) < nb ? (m - i) : nb;

            /* Compute the TZ factorization of the current block
             * A(i:i+ib-1, i:n-1)
             *
             * Fortran: ZLATRZ(IB, N-I+1, N-M, A(I,I), LDA, TAU(I), WORK)
             * N-I+1 in Fortran with 1-based I corresponds to n-i in 0-based
             */
            zlatrz(ib, n - i, n - m, &A[i + i * lda], lda, &tau[i], work);

            if (i > 0) {

                /* Form the triangular factor of the block reflector
                 * H = H(i+ib-1) . . . H(i+1) H(i)
                 *
                 * Fortran: ZLARZT('Backward', 'Rowwise', N-M, IB, A(I,M1), LDA, TAU(I), WORK, LDWORK)
                 * M1 in Fortran is column M+1 (1-based) = column m (0-based)
                 */
                zlarzt("B", "R", n - m, ib, &A[i + m * lda], lda,
                       &tau[i], work, ldwork);

                /* Apply H to A(0:i-1, i:n-1) from the right
                 *
                 * Fortran: ZLARZB('Right', 'No transpose', 'Backward', 'Rowwise',
                 *                 I-1, N-I+1, IB, N-M, A(I,M1), LDA, WORK, LDWORK,
                 *                 A(1,I), LDA, WORK(IB+1), LDWORK)
                 * I-1 in Fortran with 1-based I = i rows in 0-based
                 * N-I+1 in Fortran with 1-based I = n-i in 0-based
                 * A(1,I) in Fortran = A[0 + i*lda] in 0-based
                 * WORK(IB+1) in Fortran = work[ib*ldwork] in 0-based
                 *   (WORK is used as a 2D array with leading dimension LDWORK;
                 *    IB+1 means row IB+1, which is the start of the (IB+1)-th row
                 *    in column-major with LDWORK stride = offset IB)
                 */
                zlarzb("R", "N", "B", "R",
                       i, n - i, ib, n - m,
                       &A[i + m * lda], lda,
                       work, ldwork,
                       &A[0 + i * lda], lda,
                       &work[ib], ldwork);
            }
        }

        /* After the DO loop, Fortran I overshoots by -NB:
         * MU = I + NB - 1 (Fortran 1-based).
         * In Fortran, after the loop, I still holds its last decremented value.
         * Fortran: MU = I + NB - 1, this gives the number of remaining rows.
         * In 0-based: after the loop, i = (m-kk) - nb. So mu = i + nb = m - kk.
         * But Fortran MU = I + NB - 1 where I is the 1-based value that failed the test.
         * The Fortran loop runs: I from (m-kk+ki+1) down to (m-kk+1) step -nb.
         * After the loop, I = (m-kk+1) - nb (the value that failed the >= test).
         * So MU = (m-kk+1) - nb + nb - 1 = m - kk.
         * In 0-based, mu = m - kk (number of rows to process with unblocked code).
         */
        mu = m - kk;
    } else {
        mu = m;
    }

    /* Use unblocked code to factor the last or only block.
     * Fortran: IF(MU.GT.0) ZLATRZ(MU, N, N-M, A, LDA, TAU, WORK)
     */
    if (mu > 0) {
        zlatrz(mu, n, n - m, A, lda, tau, work);
    }

    work[0] = (c128)lwkopt;
}
