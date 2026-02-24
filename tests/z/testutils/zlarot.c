/**
 * @file zlarot.c
 * @brief ZLAROT applies a (Givens) rotation to two adjacent rows or columns.
 *
 * Faithful port of LAPACK TESTING/MATGEN/zlarot.f
 */

#include "verify.h"

/**
 * ZLAROT applies a (Givens) rotation to two adjacent rows or
 * columns, where one element of the first and/or last column/row
 * may be a separate variable. This is useful for matrices stored
 * in some format other than GE, so that elements of the matrix
 * may be used or modified for which no array element is provided.
 *
 * @param[in] lrows
 *     If true, then ZLAROT will rotate two rows. If false,
 *     then it will rotate two columns.
 *
 * @param[in] lleft
 *     If true, then xleft will be used instead of the
 *     corresponding element of A for the first element in the
 *     second row (if lrows=false) or column (if lrows=true).
 *     If false, then the corresponding element of A will be used.
 *
 * @param[in] lright
 *     If true, then xright will be used instead of the
 *     corresponding element of A for the last element in the
 *     first row (if lrows=false) or column (if lrows=true).
 *     If false, then the corresponding element of A will be used.
 *
 * @param[in] nl
 *     The length of the rows (if lrows=true) or columns (if
 *     lrows=false) to be rotated.
 *
 * @param[in] c
 *     Complex cosine of the Givens rotation. Note that in contrast
 *     to the output of ZROTG or to most versions of ZROT, both C
 *     and S are complex. For a Givens rotation, |C|^2 + |S|^2
 *     should be 1, but this is not checked.
 *
 * @param[in] s
 *     Complex sine of the Givens rotation.
 *     If lrows is true, then the matrix ( c  s )
 *                                       (-s* c*)  is applied from the left;
 *     if false, then the transpose (not conjugated) thereof is
 *     applied from the right.
 *
 * @param[in,out] A
 *     The array containing the rows/columns to be rotated.
 *
 * @param[in] lda
 *     The "effective" leading dimension of A.
 *
 * @param[in,out] xleft
 *     If lleft is true, used and modified instead of corresponding
 *     element of A.
 *
 * @param[in,out] xright
 *     If lright is true, used and modified instead of corresponding
 *     element of A.
 */
void zlarot(
    const INT lrows,
    const INT lleft,
    const INT lright,
    const INT nl,
    const c128 c,
    const c128 s,
    c128* A,
    const INT lda,
    c128* xleft,
    c128* xright)
{
    INT iinc, inext, ix, iy, iyt, j, nt;
    c128 tempx;
    c128 xt[2], yt[2];

    /* Set up indices, arrays for ends */
    if (lrows) {
        iinc = lda;
        inext = 1;
    } else {
        iinc = 1;
        inext = lda;
    }

    if (lleft) {
        nt = 1;
        ix = iinc;       /* 0-based: was 1 + IINC in Fortran 1-based */
        iy = 1 + lda;    /* 0-based: was 2 + LDA */
        xt[0] = A[0];
        yt[0] = *xleft;
    } else {
        nt = 0;
        ix = 0;
        iy = inext;
    }

    if (lright) {
        iyt = inext + (nl - 1) * iinc;
        xt[nt] = *xright;
        yt[nt] = A[iyt];
        nt = nt + 1;
    }

    /* Check for errors */
    if (nl < nt) {
        xerbla("ZLAROT", 4);
        return;
    }
    if (lda <= 0 || (!lrows && lda < nl - nt)) {
        xerbla("ZLAROT", 8);
        return;
    }

    /* Rotate: inline complex Givens rotation
     * ZROT with complex C, S:
     *   tempx    =  c * x + s * y
     *   y        = -conj(s) * x + conj(c) * y
     *   x        = tempx
     */
    for (j = 0; j < nl - nt; j++) {
        tempx = c * A[ix + j * iinc] + s * A[iy + j * iinc];
        A[iy + j * iinc] = -conj(s) * A[ix + j * iinc] +
                            conj(c) * A[iy + j * iinc];
        A[ix + j * iinc] = tempx;
    }

    for (j = 0; j < nt; j++) {
        tempx = c * xt[j] + s * yt[j];
        yt[j] = -conj(s) * xt[j] + conj(c) * yt[j];
        xt[j] = tempx;
    }

    /* Stuff values back into xleft, xright, etc. */
    if (lleft) {
        A[0] = xt[0];
        *xleft = yt[0];
    }

    if (lright) {
        *xright = xt[nt - 1];
        A[iyt] = yt[nt - 1];
    }
}
