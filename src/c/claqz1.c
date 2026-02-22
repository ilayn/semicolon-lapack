/**
 * @file claqz1.c
 * @brief CLAQZ1 chases a 1x1 shift bulge in a matrix pencil down a single position.
 */

#include "internal_build_defs.h"
#include "semicolon_lapack_complex_single.h"
#include <complex.h>

/**
 * CLAQZ1 chases a 1x1 shift bulge in a matrix pencil down a single position.
 *
 * @param[in]     ilq      Determines whether or not to update the matrix Q.
 * @param[in]     ilz      Determines whether or not to update the matrix Z.
 * @param[in]     k        Index indicating the position of the bulge.
 *                          On entry, the bulge is located in
 *                          (A(k+1,k),B(k+1,k)).
 *                          On exit, the bulge is located in
 *                          (A(k+2,k+1),B(k+2,k+1)).
 * @param[in]     istartm  Updates to (A,B) are restricted to
 *                          (istartm:k+2,k:istopm).
 * @param[in]     istopm   See istartm.
 * @param[in]     ihi      Index of the last row/column in the active block.
 * @param[in,out] A        Complex array, dimension (lda, n).
 * @param[in]     lda      The leading dimension of A.
 * @param[in,out] B        Complex array, dimension (ldb, n).
 * @param[in]     ldb      The leading dimension of B.
 * @param[in]     nq       The order of the matrix Q.
 * @param[in]     qstart   Start index of the matrix Q.
 * @param[in,out] Q        Complex array, dimension (ldq, nq).
 * @param[in]     ldq      The leading dimension of Q.
 * @param[in]     nz       The order of the matrix Z.
 * @param[in]     zstart   Start index of the matrix Z.
 * @param[in,out] Z        Complex array, dimension (ldz, nz).
 * @param[in]     ldz      The leading dimension of Z.
 */
void claqz1(const INT ilq, const INT ilz, const INT k, const INT istartm,
            const INT istopm, const INT ihi, c64* A, const INT lda,
            c64* B, const INT ldb, const INT nq, const INT qstart,
            c64* Q, const INT ldq, const INT nz, const INT zstart,
            c64* Z, const INT ldz)
{
    const c64 CZERO = CMPLXF(0.0f, 0.0f);

    f32 c;
    c64 s, temp;

    if (k + 1 == ihi) {

        /* Shift is located on the edge of the matrix, remove it */

        clartg(B[ihi + ihi * ldb], B[ihi + (ihi - 1) * ldb], &c, &s,
               &temp);
        B[ihi + ihi * ldb] = temp;
        B[ihi + (ihi - 1) * ldb] = CZERO;
        crot(ihi - istartm, &B[istartm + ihi * ldb], 1,
             &B[istartm + (ihi - 1) * ldb], 1, c, s);
        crot(ihi - istartm + 1, &A[istartm + ihi * lda], 1,
             &A[istartm + (ihi - 1) * lda], 1, c, s);
        if (ilz) {
            crot(nz, &Z[(ihi - zstart) * ldz], 1,
                 &Z[(ihi - zstart - 1) * ldz], 1, c, s);
        }

    } else {

        /* Normal operation, move bulge down */

        /* Apply transformation from the right */

        clartg(B[(k + 1) + (k + 1) * ldb], B[(k + 1) + k * ldb], &c, &s,
               &temp);
        B[(k + 1) + (k + 1) * ldb] = temp;
        B[(k + 1) + k * ldb] = CZERO;
        crot(k - istartm + 3, &A[istartm + (k + 1) * lda], 1,
             &A[istartm + k * lda], 1, c, s);
        crot(k - istartm + 1, &B[istartm + (k + 1) * ldb], 1,
             &B[istartm + k * ldb], 1, c, s);
        if (ilz) {
            crot(nz, &Z[(k - zstart + 1) * ldz], 1,
                 &Z[(k - zstart) * ldz], 1, c, s);
        }

        /* Apply transformation from the left */

        clartg(A[(k + 1) + k * lda], A[(k + 2) + k * lda], &c, &s, &temp);
        A[(k + 1) + k * lda] = temp;
        A[(k + 2) + k * lda] = CZERO;
        crot(istopm - k, &A[(k + 1) + (k + 1) * lda], lda,
             &A[(k + 2) + (k + 1) * lda], lda, c, s);
        crot(istopm - k, &B[(k + 1) + (k + 1) * ldb], ldb,
             &B[(k + 2) + (k + 1) * ldb], ldb, c, s);
        if (ilq) {
            crot(nq, &Q[(k - qstart + 1) * ldq], 1,
                 &Q[(k - qstart + 2) * ldq], 1, c, conjf(s));
        }

    }
}
