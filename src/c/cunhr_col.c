/**
 * @file cunhr_col.c
 * @brief CUNHR_COL takes an M-by-N complex matrix Q_in with orthonormal
 *        columns as input, and performs Householder Reconstruction (HR).
 */
#include "semicolon_lapack_complex_single.h"
#include <complex.h>
#include "semicolon_cblas.h"

/**
 * CUNHR_COL takes an M-by-N complex matrix Q_in with orthonormal columns
 * as input, stored in A, and performs Householder Reconstruction (HR),
 * i.e. reconstructs Householder vectors V(i) implicitly representing
 * another M-by-N matrix Q_out, with the property that Q_in = Q_out*S,
 * where S is an N-by-N diagonal matrix with diagonal entries
 * equal to +1 or -1. The Householder vectors (columns V(i) of V) are
 * stored in A on output, and the diagonal entries of S are stored in D.
 * Block reflectors are also returned in T
 * (same output format as CGEQRT).
 *
 * @param[in]     m     The number of rows of the matrix A. m >= 0.
 * @param[in]     n     The number of columns of the matrix A. m >= n >= 0.
 * @param[in]     nb    The column block size to be used in the reconstruction
 *                      of Householder column vector blocks in the array A and
 *                      corresponding block reflectors in the array T. nb >= 1.
 *                      (Note that if nb > n, then n is used instead of nb
 *                      as the column block size.)
 * @param[in,out] A     Complex*16 array, dimension (lda, n).
 *                      On entry, the array A contains an M-by-N orthonormal
 *                      matrix Q_in, i.e the columns of A are orthogonal
 *                      unit vectors.
 *                      On exit, the elements below the diagonal of A represent
 *                      the unit lower-trapezoidal matrix V of Householder
 *                      column vectors V(i). The unit diagonal entries of V are
 *                      not stored (same format as the output below the diagonal
 *                      in A from CGEQRT). The matrix T and the matrix V stored
 *                      on output in A implicitly define Q_out.
 *                      The elements above the diagonal contain the factor U
 *                      of the "modified" LU-decomposition:
 *                         Q_in - ( S ) = V * U
 *                                ( 0 )
 *                      where 0 is a (M-N)-by-(M-N) zero matrix.
 * @param[in]     lda   The leading dimension of the array A. lda >= max(1, m).
 * @param[out]    T     Complex*16 array, dimension (ldt, n).
 *                      On exit, T(0:nb-1, 0:n-1) contains upper-triangular
 *                      block reflectors used to define Q_out stored in compact
 *                      form as a sequence of upper-triangular nb-by-nb column
 *                      blocks (same format as the output T in CGEQRT).
 * @param[in]     ldt   The leading dimension of the array T.
 *                      ldt >= max(1, min(nb, n)).
 * @param[out]    D     Complex*16 array, dimension min(m, n).
 *                      The elements can be only plus or minus one.
 *                      D(i) is constructed as D(i) = -SIGN(Q_in_i(i,i)), where
 *                      0 <= i < min(m,n), and Q_in_i is Q_in after performing
 *                      i steps of "modified" Gaussian elimination.
 * @param[out]    info  = 0: successful exit
 *                      < 0: if info = -i, the i-th argument had an illegal value
 */
void cunhr_col(const INT m, const INT n, const INT nb,
               c64* restrict A, const INT lda,
               c64* restrict T, const INT ldt,
               c64* restrict D,
               INT* info)
{
    const c64 CONE = CMPLXF(1.0f, 0.0f);
    const c64 CZERO = CMPLXF(0.0f, 0.0f);
    const c64 NEG_CONE = CMPLXF(-1.0f, 0.0f);

    INT i, iinfo, j, jb, jnb, nplusone;

    *info = 0;
    if (m < 0) {
        *info = -1;
    } else if (n < 0 || n > m) {
        *info = -2;
    } else if (nb < 1) {
        *info = -3;
    } else if (lda < (1 > m ? 1 : m)) {
        *info = -5;
    } else if (ldt < (1 > (nb < n ? nb : n) ? 1 : (nb < n ? nb : n))) {
        *info = -7;
    }

    if (*info != 0) {
        xerbla("CUNHR_COL", -(*info));
        return;
    }

    if ((m < n ? m : n) == 0) {
        return;
    }

    /*
     * (1) Compute the unit lower-trapezoidal V (ones on the diagonal
     * are not stored) by performing the "modified" LU-decomposition.
     *
     * Q_in - ( S ) = V * U = ( V1 ) * U,
     *         ( 0 )           ( V2 )
     *
     * where 0 is an (M-N)-by-N zero matrix.
     *
     * (1-1) Factor V1 and U.
     */
    claunhr_col_getrfnp(n, n, A, lda, D, &iinfo);

    /* (1-2) Solve for V2. */
    if (m > n) {
        cblas_ctrsm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans,
                    CblasNonUnit, m - n, n, &CONE, A, lda,
                    &A[n], lda);
    }

    /*
     * (2) Reconstruct the block reflector T stored in T(0:nb-1, 0:n-1)
     * as a sequence of upper-triangular blocks with nb-size column
     * blocking.
     *
     * Loop over the column blocks of size nb of the array A(0:m-1,0:n-1)
     * and the array T(0:nb-1,0:n-1), jb is the column index of a column
     * block, jnb is the column block size at each step jb.
     */
    nplusone = n + 1;

    /* Fortran: DO JB = 1, N, NB  =>  C: jb from 0, step nb */
    for (jb = 0; jb < n; jb += nb) {

        /* (2-0) Determine the column block size jnb. */
        jnb = (nplusone - (jb + 1)) < nb ? (nplusone - (jb + 1)) : nb;
        /* nplusone - (jb+1) = n - jb, same as Fortran NPLUSONE-JB */

        /*
         * (2-1) Copy the upper-triangular part of the current jnb-by-jnb
         * diagonal block U(jb) stored in A(jb:jb+jnb-1, jb:jb+jnb-1)
         * into the upper-triangular part of the current jnb-by-jnb block
         * T(0:jnb-1, jb:jb+jnb-1) column-by-column.
         */
        /* Fortran: JBTEMP1 = JB - 1 (1-based) => in 0-based the count
         * for column j (0-based) is: j - jb + 1 elements */
        for (j = jb; j < jb + jnb; j++) {
            cblas_ccopy(j - jb + 1, &A[jb + j * lda], 1, &T[0 + j * ldt], 1);
        }

        /*
         * (2-2) Perform on the upper-triangular part of the current
         * jnb-by-jnb diagonal block U(jb) stored in T(0:jnb-1, jb:jb+jnb-1):
         * (-1)*U(jb)*S(jb). Changing the sign of each j-th column
         * according to the sign of D(j).
         */
        for (j = jb; j < jb + jnb; j++) {
            if (D[j] == CONE) {
                cblas_cscal(j - jb + 1, &NEG_CONE, &T[0 + j * ldt], 1);
            }
        }

        /*
         * (2-3) Perform the triangular solve for the current block
         * matrix X(jb):
         *
         *     X(jb) * (A(jb)^H) = B(jb), where:
         *
         *     A(jb)^H  is a jnb-by-jnb unit upper-triangular
         *              coefficient block, and A(jb)=V1(jb), which
         *              is a jnb-by-jnb unit lower-triangular block
         *              stored in A(jb:jb+jnb-1, jb:jb+jnb-1).
         *
         *     B(jb)    is a jnb-by-jnb upper-triangular right-hand
         *              side block, B(jb) = (-1)*U(jb)*S(jb), stored
         *              in T(0:jnb-1, jb:jb+jnb-1).
         *
         *     X(jb)    is a jnb-by-jnb upper-triangular solution
         *              block, the block reflector T(jb), stored
         *              in T(0:jnb-1, jb:jb+jnb-1).
         *
         * (2-3a) Set the elements to zero below the diagonal.
         */

        /* Fortran: JBTEMP2 = JB - 2
         * DO J = JB, JB+JNB-2
         *   DO I = J-JBTEMP2, MIN(NB, N)
         *     T(I, J) = CZERO
         *
         * Fortran J is 1-based. In 0-based:
         * j goes from jb to jb+jnb-2 (i.e., all but last column of block)
         * Fortran I start: J - JBTEMP2 = J - (JB-2) = J - JB + 2 (1-based)
         *   => 0-based: (j+1) - (jb+1) + 2 - 1 = j - jb + 1
         * Fortran I end: MIN(NB, N) (1-based) => 0-based: min(nb, n) - 1
         */
        for (j = jb; j < jb + jnb - 1; j++) {
            for (i = j - jb + 1; i < (nb < n ? nb : n); i++) {
                T[i + j * ldt] = CZERO;
            }
        }

        /* (2-3b) Perform the triangular solve. */
        cblas_ctrsm(CblasColMajor, CblasRight, CblasLower,
                    CblasConjTrans, CblasUnit,
                    jnb, jnb, &CONE,
                    &A[jb + jb * lda], lda, &T[0 + jb * ldt], ldt);
    }
}
