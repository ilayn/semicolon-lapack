/**
 * @file zungtsqr.c
 * @brief ZUNGTSQR generates an M-by-N complex matrix Q_out with orthonormal columns.
 */

#include "semicolon_lapack_complex_double.h"
#include <complex.h>
#include <cblas.h>

/**
 * ZUNGTSQR generates an M-by-N complex matrix Q_out with orthonormal
 * columns, which are the first N columns of a product of complex unitary
 * matrices of order M which are returned by ZLATSQR
 *
 *      Q_out = first_N_columns_of( Q(1)_in * Q(2)_in * ... * Q(k)_in ).
 *
 * See the documentation for ZLATSQR.
 *
 * @param[in]     m      The number of rows of the matrix A. m >= 0.
 * @param[in]     n      The number of columns of the matrix A. m >= n >= 0.
 * @param[in]     mb     The row block size used by ZLATSQR to return
 *                       arrays A and T. mb > n.
 *                       (Note that if mb > m, then m is used instead of mb
 *                       as the row block size).
 * @param[in]     nb     The column block size used by ZLATSQR to return
 *                       arrays A and T. nb >= 1.
 *                       (Note that if nb > n, then n is used instead of nb
 *                       as the column block size).
 * @param[in,out] A      Double complex array, dimension (lda, n).
 *                       On entry, the elements on and above the diagonal are
 *                       not accessed. The elements below the diagonal represent
 *                       the unit lower-trapezoidal blocked matrix V computed by
 *                       ZLATSQR that defines the input matrices Q_in(k).
 *                       On exit, the array A contains an M-by-N orthonormal
 *                       matrix Q_out.
 * @param[in]     lda    The leading dimension of the array A. lda >= max(1, m).
 * @param[in]     T      Double complex array,
 *                       dimension (ldt, n * NIRB)
 *                       where NIRB = Number_of_input_row_blocks
 *                                  = MAX(1, CEIL((m-n)/(mb-n))).
 *                       The upper-triangular block reflectors used to define the
 *                       input matrices Q_in(k).
 * @param[in]     ldt    The leading dimension of the array T.
 *                       ldt >= max(1, min(nb, n)).
 * @param[out]    work   Workspace array, dimension (max(2, lwork)).
 *                       On exit, if info = 0, work[0] returns the optimal lwork.
 * @param[in]     lwork  The dimension of the array work. lwork >= (m+nb)*n.
 *                       If lwork = -1, then a workspace query is assumed.
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 */
void zungtsqr(const int m, const int n, const int mb, const int nb,
              c128* restrict A, const int lda,
              c128* restrict T, const int ldt,
              c128* restrict work, const int lwork,
              int* info)
{
    const c128 CONE = CMPLX(1.0, 0.0);
    const c128 CZERO = CMPLX(0.0, 0.0);

    int lquery;
    int iinfo, ldc, lworkopt, lc, lw, nblocal, j;

    /* Test the input parameters */

    lquery = (lwork == -1);
    *info = 0;
    if (m < 0) {
        *info = -1;
    } else if (n < 0 || m < n) {
        *info = -2;
    } else if (mb <= n) {
        *info = -3;
    } else if (nb < 1) {
        *info = -4;
    } else if (lda < (1 > m ? 1 : m)) {
        *info = -6;
    } else if (ldt < (1 > (nb < n ? nb : n) ? 1 : (nb < n ? nb : n))) {
        *info = -8;
    } else {

        /*
         * Test the input LWORK for the dimension of the array WORK.
         * This workspace is used to store array C(LDC, N) and WORK(LWORK)
         * in the call to ZLAMTSQR. See the documentation for ZLAMTSQR.
         */

        if (lwork < 2 && !lquery) {
            *info = -10;
        } else {

            /* Set block size for column blocks */

            nblocal = nb < n ? nb : n;

            /*
             * LWORK = -1, then set the size for the array C(LDC,N)
             * in ZLAMTSQR call and set the optimal size of the work array
             * WORK(LWORK) in ZLAMTSQR call.
             */

            ldc = m;
            lc = ldc * n;
            lw = n * nblocal;

            lworkopt = lc + lw;

            if ((lwork < (1 > lworkopt ? 1 : lworkopt)) && !lquery) {
                *info = -10;
            }
        }
    }

    /* Handle error in the input parameters and return workspace query. */

    if (*info != 0) {
        xerbla("ZUNGTSQR", -(*info));
        return;
    } else if (lquery) {
        work[0] = CMPLX((f64)lworkopt, 0.0);
        return;
    }

    /* Quick return if possible */

    if ((m < n ? m : n) == 0) {
        work[0] = CMPLX((f64)lworkopt, 0.0);
        return;
    }

    /*
     * (1) Form explicitly the tall-skinny M-by-N left submatrix Q1_in
     * of M-by-M orthogonal matrix Q_in, which is implicitly stored in
     * the subdiagonal part of input array A and in the input array T.
     * Perform by the following operation using the routine ZLAMTSQR.
     *
     *     Q1_in = Q_in * ( I ), where I is a N-by-N identity matrix,
     *                    ( 0 )        0 is a (M-N)-by-N zero matrix.
     *
     * (1a) Form M-by-N matrix in the array WORK(1:LDC*N) with ones
     * on the diagonal and zeros elsewhere.
     */

    zlaset("F", m, n, CZERO, CONE, work, ldc);

    /*
     * (1b)  On input, WORK(1:LDC*N) stores ( I );
     *                                      ( 0 )
     *
     *       On output, WORK(1:LDC*N) stores Q1_in.
     */

    zlamtsqr("L", "N", m, n, n, mb, nblocal, A, lda, T, ldt,
             work, ldc, &work[lc], lw, &iinfo);

    /*
     * (2) Copy the result from the part of the work array (1:M,1:N)
     * with the leading dimension LDC that starts at WORK(1) into
     * the output array A(1:M,1:N) column-by-column.
     */

    for (j = 0; j < n; j++) {
        cblas_zcopy(m, &work[j * ldc], 1, &A[j * lda], 1);
    }

    work[0] = CMPLX((f64)lworkopt, 0.0);
}
