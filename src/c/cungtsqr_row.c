/**
 * @file cungtsqr_row.c
 * @brief CUNGTSQR_ROW generates an M-by-N complex matrix Q_out with orthonormal columns from CLATSQR output using row-by-row algorithm.
 */

#include "semicolon_lapack_complex_single.h"
#include <complex.h>

/**
 * CUNGTSQR_ROW generates an M-by-N complex matrix Q_out with
 * orthonormal columns from the output of CLATSQR. These N orthonormal
 * columns are the first N columns of a product of complex unitary
 * matrices Q(k)_in of order M, which are returned by CLATSQR in
 * a special format.
 *
 *      Q_out = first_N_columns_of( Q(1)_in * Q(2)_in * ... * Q(k)_in ).
 *
 * The input matrices Q(k)_in are stored in row and column blocks in A.
 * See the documentation of CLATSQR for more details on the format of
 * Q(k)_in, where each Q(k)_in is represented by block Householder
 * transformations. This routine calls an auxiliary routine CLARFB_GETT,
 * where the computation is performed on each individual block. The
 * algorithm first sweeps NB-sized column blocks from the right to left
 * starting in the bottom row block and continues to the top row block
 * (hence _ROW in the routine name). This sweep is in reverse order of
 * the order in which CLATSQR generates the output blocks.
 *
 * @param[in] m
 *          The number of rows of the matrix A. m >= 0.
 *
 * @param[in] n
 *          The number of columns of the matrix A. m >= n >= 0.
 *
 * @param[in] mb
 *          The row block size used by CLATSQR. mb > n.
 *
 * @param[in] nb
 *          The column block size used by CLATSQR. nb >= 1.
 *
 * @param[in,out] A
 *          Single complex array, dimension (lda, n).
 *          On entry, the elements below the diagonal represent the unit
 *          lower-trapezoidal blocked matrix V computed by CLATSQR.
 *          On exit, the M-by-N orthonormal matrix Q_out.
 *
 * @param[in] lda
 *          The leading dimension of the array A. lda >= max(1, m).
 *
 * @param[in] T
 *          Single complex array, dimension (ldt, n * NIRB).
 *          The upper-triangular block reflectors from CLATSQR.
 *
 * @param[in] ldt
 *          The leading dimension of the array T. ldt >= max(1, min(nb, n)).
 *
 * @param[out] work
 *          Single complex workspace of size (max(1, lwork)).
 *          On exit, work[0] returns the optimal lwork.
 *
 * @param[in] lwork
 *          The dimension of the array work.
 *          lwork >= nblocal * max(nblocal, (n-nblocal)), where nblocal=min(nb,n).
 *          If lwork = -1, then a workspace query is assumed.
 *
 * @param[out] info
 *                         - = 0:  successful exit
 *                         - < 0:  if info = -i, the i-th argument had an illegal value
 */
void cungtsqr_row(
    const int m,
    const int n,
    const int mb,
    const int nb,
    c64* restrict A,
    const int lda,
    const c64* restrict T,
    const int ldt,
    c64* restrict work,
    const int lwork,
    int* info)
{
    const c64 CONE = CMPLXF(1.0f, 0.0f);
    const c64 CZERO = CMPLXF(0.0f, 0.0f);

    int lquery;
    int nblocal, mb2, itmp, ib_bottom;
    int lworkopt, num_all_row_blocks, jb_t, ib, imb;
    int kb, kb_last, knb, mb1;
    c64 dummy[1];
    int minval;

    *info = 0;
    lquery = (lwork == -1);
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
    } else {
        minval = (nb < n) ? nb : n;
        if (ldt < (1 > minval ? 1 : minval)) {
            *info = -8;
        } else if (lwork < 1 && !lquery) {
            *info = -10;
        }
    }

    nblocal = (nb < n) ? nb : n;

    if (*info == 0) {
        lworkopt = nblocal * ((nblocal > (n - nblocal)) ? nblocal : (n - nblocal));
    }

    if (*info != 0) {
        xerbla("CUNGTSQR_ROW", -(*info));
        return;
    } else if (lquery) {
        work[0] = CMPLXF((f32)lworkopt, 0.0f);
        return;
    }

    minval = (m < n) ? m : n;
    if (minval == 0) {
        work[0] = CMPLXF((f32)lworkopt, 0.0f);
        return;
    }

    claset("U", m, n, CZERO, CONE, A, lda);

    kb_last = ((n - 1) / nblocal) * nblocal;

    if (mb < m) {

        mb2 = mb - n;
        itmp = (m - mb - 1) / mb2;
        ib_bottom = itmp * mb2 + mb;
        num_all_row_blocks = itmp + 2;
        jb_t = num_all_row_blocks * n;

        for (ib = ib_bottom; ib >= mb; ib -= mb2) {

            imb = ((m - ib) < mb2) ? (m - ib) : mb2;

            jb_t = jb_t - n;

            for (kb = kb_last; kb >= 0; kb -= nblocal) {

                knb = (nblocal < (n - kb)) ? nblocal : (n - kb);

                clarfb_gett("I", imb, n - kb, knb,
                            &T[0 + (jb_t + kb) * ldt], ldt, &A[kb + kb * lda], lda,
                            &A[ib + kb * lda], lda, work, knb);

            }

        }

    }

    mb1 = (mb < m) ? mb : m;

    for (kb = kb_last; kb >= 0; kb -= nblocal) {

        knb = (nblocal < (n - kb)) ? nblocal : (n - kb);

        if (mb1 - kb - knb == 0) {

            clarfb_gett("N", 0, n - kb, knb,
                        &T[0 + kb * ldt], ldt, &A[kb + kb * lda], lda,
                        dummy, 1, work, knb);
        } else {
            clarfb_gett("N", mb1 - kb - knb, n - kb, knb,
                        &T[0 + kb * ldt], ldt, &A[kb + kb * lda], lda,
                        &A[(kb + knb) + kb * lda], lda, work, knb);

        }

    }

    work[0] = CMPLXF((f32)lworkopt, 0.0f);
}
