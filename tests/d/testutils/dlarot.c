/**
 * @file dlarot.c
 * @brief DLAROT applies a (Givens) rotation to two adjacent rows or columns.
 *
 * Faithful port of LAPACK TESTING/MATGEN/dlarot.f
 */

#include "semicolon_cblas.h"
#include "verify.h"

/**
 * DLAROT applies a (Givens) rotation to two adjacent rows or
 * columns, where one element of the first and/or last column/row
 * may be a separate variable. This is useful for matrices stored
 * in some format other than GE, so that elements of the matrix
 * may be used or modified for which no array element is provided.
 *
 * @param[in] lrows
 *     If true, then DLAROT will rotate two rows. If false,
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
 *     lrows=false) to be rotated. If xleft and/or xright are
 *     used, the columns/rows they are in should be included in
 *     nl, e.g., if lleft = lright = true, then nl must be at
 *     least 2. The number of rows/columns to be rotated
 *     exclusive of those involving xleft and/or xright may
 *     not be negative.
 *
 * @param[in] c
 *     Cosine of the Givens rotation.
 *
 * @param[in] s
 *     Sine of the Givens rotation.
 *     If lrows is true, then the matrix ( c  s )
 *                                       (-s  c )  is applied from the left;
 *     if false, then the transpose thereof is applied from the right.
 *
 * @param[in,out] A
 *     The array containing the rows/columns to be rotated. The
 *     first element of A should be the upper left element to
 *     be rotated. Uses 0-based indexing internally.
 *
 * @param[in] lda
 *     The "effective" leading dimension of A. If A contains
 *     a matrix stored in GE or SY format, then this is just
 *     the leading dimension of A as dimensioned in the calling
 *     routine. If A contains a matrix stored in band (GB or SB)
 *     format, then this should be *one less* than the leading
 *     dimension used in the calling routine.
 *
 * @param[in,out] xleft
 *     If lleft is true, then xleft will be used and modified
 *     instead of A[1,0] (if lrows=true) or A[0,1] (if lrows=false).
 *
 * @param[in,out] xright
 *     If lright is true, then xright will be used and modified
 *     instead of A[0,nl-1] (if lrows=true) or A[nl-1,0]
 *     (if lrows=false).
 */
void dlarot(
    const INT lrows,
    const INT lleft,
    const INT lright,
    const INT nl,
    const f64 c,
    const f64 s,
    f64* A,
    const INT lda,
    f64* xleft,
    f64* xright)
{
    INT iinc, inext, ix, iy, iyt, nt;
    f64 xt[2], yt[2];

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
        xt[0] = A[0];    /* A(1) in Fortran */
        yt[0] = *xleft;
    } else {
        nt = 0;
        ix = 0;          /* 0-based: was 1 */
        iy = inext;      /* 0-based: was 1 + INEXT */
    }

    if (lright) {
        iyt = inext + (nl - 1) * iinc;  /* 0-based: was 1 + INEXT + (NL-1)*IINC */
        xt[nt] = *xright;
        yt[nt] = A[iyt];
        nt = nt + 1;
    }

    /* Check for errors */
    if (nl < nt) {
        xerbla("DLAROT", 4);
        return;
    }
    if (lda <= 0 || (!lrows && lda < nl - nt)) {
        xerbla("DLAROT", 8);
        return;
    }

    /* Rotate using BLAS drot */
    if (nl - nt > 0) {
        cblas_drot(nl - nt, &A[ix], iinc, &A[iy], iinc, c, s);
    }
    if (nt > 0) {
        cblas_drot(nt, xt, 1, yt, 1, c, s);
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
