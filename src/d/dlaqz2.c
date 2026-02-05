/**
 * @file dlaqz2.c
 * @brief DLAQZ2 chases a 2x2 shift bulge in a matrix pencil down a single position.
 */

#include <cblas.h>
#include "semicolon_lapack_double.h"

/**
 * DLAQZ2 chases a 2x2 shift bulge in a matrix pencil down a single position.
 *
 * @param[in]     ilq      Determines whether or not to update Q.
 * @param[in]     ilz      Determines whether or not to update Z.
 * @param[in]     k        Index indicating the position of the bulge (0-based).
 *                         On entry, the bulge is located in
 *                         (A(k+1:k+2,k:k+1),B(k+1:k+2,k:k+1)).
 *                         On exit, the bulge is located in
 *                         (A(k+2:k+3,k+1:k+2),B(k+2:k+3,k+1:k+2)).
 * @param[in]     istartm  Start row for updates (0-based).
 * @param[in]     istopm   Stop column for updates (0-based).
 * @param[in]     ihi      Upper bound of active submatrix (0-based).
 * @param[in,out] A        Matrix A.
 * @param[in]     lda      Leading dimension of A.
 * @param[in,out] B        Matrix B.
 * @param[in]     ldb      Leading dimension of B.
 * @param[in]     nq       Order of matrix Q.
 * @param[in]     qstart   Start index of Q (0-based).
 * @param[in,out] Q        Matrix Q.
 * @param[in]     ldq      Leading dimension of Q.
 * @param[in]     nz       Order of matrix Z.
 * @param[in]     zstart   Start index of Z (0-based).
 * @param[in,out] Z        Matrix Z.
 * @param[in]     ldz      Leading dimension of Z.
 */
void dlaqz2(
    const int ilq,
    const int ilz,
    const int k,
    const int istartm,
    const int istopm,
    const int ihi,
    double* const restrict A,
    const int lda,
    double* const restrict B,
    const int ldb,
    const int nq,
    const int qstart,
    double* const restrict Q,
    const int ldq,
    const int nz,
    const int zstart,
    double* const restrict Z,
    const int ldz)
{
    const double ZERO = 0.0;

    double h[2 * 3];
    double c1, s1, c2, s2, temp;

    if (k + 2 == ihi) {
        /* Shift is located on the edge of the matrix, remove it */
        /* H = B(IHI-1:IHI, IHI-2:IHI) in Fortran 1-based */
        /* With 0-based ihi: rows (ihi-1, ihi), cols (ihi-2, ihi-1, ihi) */
        h[0 + 0 * 2] = B[(ihi - 1) + (ihi - 2) * ldb];
        h[1 + 0 * 2] = B[ihi + (ihi - 2) * ldb];
        h[0 + 1 * 2] = B[(ihi - 1) + (ihi - 1) * ldb];
        h[1 + 1 * 2] = B[ihi + (ihi - 1) * ldb];
        h[0 + 2 * 2] = B[(ihi - 1) + ihi * ldb];
        h[1 + 2 * 2] = B[ihi + ihi * ldb];

        /* Make H upper triangular */
        dlartg(h[0 + 0 * 2], h[1 + 0 * 2], &c1, &s1, &temp);
        h[1 + 0 * 2] = ZERO;
        h[0 + 0 * 2] = temp;
        cblas_drot(2, &h[0 + 1 * 2], 2, &h[1 + 1 * 2], 2, c1, s1);

        dlartg(h[1 + 2 * 2], h[1 + 1 * 2], &c1, &s1, &temp);
        cblas_drot(1, &h[0 + 2 * 2], 1, &h[0 + 1 * 2], 1, c1, s1);
        dlartg(h[0 + 1 * 2], h[0 + 0 * 2], &c2, &s2, &temp);

        /* DROT(IHI-ISTARTM+1, B(ISTARTM,IHI), 1, B(ISTARTM,IHI-1), 1, C1, S1) */
        /* Count: ihi - istartm + 1 (size, not adjusted) */
        cblas_drot(ihi - istartm + 1, &B[istartm + ihi * ldb], 1,
                   &B[istartm + (ihi - 1) * ldb], 1, c1, s1);
        cblas_drot(ihi - istartm + 1, &B[istartm + (ihi - 1) * ldb], 1,
                   &B[istartm + (ihi - 2) * ldb], 1, c2, s2);
        B[(ihi - 1) + (ihi - 2) * ldb] = ZERO;
        B[ihi + (ihi - 2) * ldb] = ZERO;
        cblas_drot(ihi - istartm + 1, &A[istartm + ihi * lda], 1,
                   &A[istartm + (ihi - 1) * lda], 1, c1, s1);
        cblas_drot(ihi - istartm + 1, &A[istartm + (ihi - 1) * lda], 1,
                   &A[istartm + (ihi - 2) * lda], 1, c2, s2);
        if (ilz) {
            /* Z(1, IHI-ZSTART+1) in Fortran → Z[0 + (ihi-zstart)*ldz] in C */
            cblas_drot(nz, &Z[0 + (ihi - zstart) * ldz], 1,
                       &Z[0 + (ihi - 1 - zstart) * ldz], 1, c1, s1);
            cblas_drot(nz, &Z[0 + (ihi - 1 - zstart) * ldz], 1,
                       &Z[0 + (ihi - 2 - zstart) * ldz], 1, c2, s2);
        }

        /* DLARTG(A(IHI-1,IHI-2), A(IHI,IHI-2), ...) */
        dlartg(A[(ihi - 1) + (ihi - 2) * lda], A[ihi + (ihi - 2) * lda],
               &c1, &s1, &temp);
        A[(ihi - 1) + (ihi - 2) * lda] = temp;
        A[ihi + (ihi - 2) * lda] = ZERO;
        /* DROT(ISTOPM-IHI+2, A(IHI-1,IHI-1), LDA, A(IHI,IHI-1), LDA, C1, S1) */
        cblas_drot(istopm - ihi + 2, &A[(ihi - 1) + (ihi - 1) * lda], lda,
                   &A[ihi + (ihi - 1) * lda], lda, c1, s1);
        cblas_drot(istopm - ihi + 2, &B[(ihi - 1) + (ihi - 1) * ldb], ldb,
                   &B[ihi + (ihi - 1) * ldb], ldb, c1, s1);
        if (ilq) {
            /* Q(1, IHI-1-QSTART+1) in Fortran → Q[0 + (ihi-1-qstart)*ldq] in C */
            cblas_drot(nq, &Q[0 + (ihi - 1 - qstart) * ldq], 1,
                       &Q[0 + (ihi - qstart) * ldq], 1, c1, s1);
        }

        /* DLARTG(B(IHI,IHI), B(IHI,IHI-1), ...) */
        dlartg(B[ihi + ihi * ldb], B[ihi + (ihi - 1) * ldb],
               &c1, &s1, &temp);
        B[ihi + ihi * ldb] = temp;
        B[ihi + (ihi - 1) * ldb] = ZERO;
        /* DROT(IHI-ISTARTM, ...) - note: no +1 here */
        cblas_drot(ihi - istartm, &B[istartm + ihi * ldb], 1,
                   &B[istartm + (ihi - 1) * ldb], 1, c1, s1);
        cblas_drot(ihi - istartm + 1, &A[istartm + ihi * lda], 1,
                   &A[istartm + (ihi - 1) * lda], 1, c1, s1);
        if (ilz) {
            cblas_drot(nz, &Z[0 + (ihi - zstart) * ldz], 1,
                       &Z[0 + (ihi - 1 - zstart) * ldz], 1, c1, s1);
        }

    } else {
        /* Normal operation, move bulge down */
        /* H = B(K+1:K+2, K:K+2) in Fortran 1-based */
        /* With 0-based k: rows (k+1, k+2), cols (k, k+1, k+2) */
        h[0 + 0 * 2] = B[(k + 1) + k * ldb];
        h[1 + 0 * 2] = B[(k + 2) + k * ldb];
        h[0 + 1 * 2] = B[(k + 1) + (k + 1) * ldb];
        h[1 + 1 * 2] = B[(k + 2) + (k + 1) * ldb];
        h[0 + 2 * 2] = B[(k + 1) + (k + 2) * ldb];
        h[1 + 2 * 2] = B[(k + 2) + (k + 2) * ldb];

        /* Make H upper triangular */
        dlartg(h[0 + 0 * 2], h[1 + 0 * 2], &c1, &s1, &temp);
        h[1 + 0 * 2] = ZERO;
        h[0 + 0 * 2] = temp;
        cblas_drot(2, &h[0 + 1 * 2], 2, &h[1 + 1 * 2], 2, c1, s1);

        /* Calculate Z1 and Z2 */
        dlartg(h[1 + 2 * 2], h[1 + 1 * 2], &c1, &s1, &temp);
        cblas_drot(1, &h[0 + 2 * 2], 1, &h[0 + 1 * 2], 1, c1, s1);
        dlartg(h[0 + 1 * 2], h[0 + 0 * 2], &c2, &s2, &temp);

        /* Apply transformations from the right */
        /* DROT(K+3-ISTARTM+1, A(ISTARTM,K+2), 1, A(ISTARTM,K+1), 1, C1, S1) */
        cblas_drot(k + 3 - istartm + 1, &A[istartm + (k + 2) * lda], 1,
                   &A[istartm + (k + 1) * lda], 1, c1, s1);
        cblas_drot(k + 3 - istartm + 1, &A[istartm + (k + 1) * lda], 1,
                   &A[istartm + k * lda], 1, c2, s2);
        cblas_drot(k + 2 - istartm + 1, &B[istartm + (k + 2) * ldb], 1,
                   &B[istartm + (k + 1) * ldb], 1, c1, s1);
        cblas_drot(k + 2 - istartm + 1, &B[istartm + (k + 1) * ldb], 1,
                   &B[istartm + k * ldb], 1, c2, s2);
        if (ilz) {
            /* Z(1, K+2-ZSTART+1) in Fortran → Z[0 + (k+2-zstart)*ldz] in C */
            cblas_drot(nz, &Z[0 + (k + 2 - zstart) * ldz], 1,
                       &Z[0 + (k + 1 - zstart) * ldz], 1, c1, s1);
            cblas_drot(nz, &Z[0 + (k + 1 - zstart) * ldz], 1,
                       &Z[0 + (k - zstart) * ldz], 1, c2, s2);
        }
        B[(k + 1) + k * ldb] = ZERO;
        B[(k + 2) + k * ldb] = ZERO;

        /* Calculate Q1 and Q2 */
        /* DLARTG(A(K+2,K), A(K+3,K), ...) */
        dlartg(A[(k + 2) + k * lda], A[(k + 3) + k * lda], &c1, &s1, &temp);
        A[(k + 2) + k * lda] = temp;
        A[(k + 3) + k * lda] = ZERO;
        dlartg(A[(k + 1) + k * lda], A[(k + 2) + k * lda], &c2, &s2, &temp);
        A[(k + 1) + k * lda] = temp;
        A[(k + 2) + k * lda] = ZERO;

        /* Apply transformations from the left */
        /* DROT(ISTOPM-K, A(K+2,K+1), LDA, A(K+3,K+1), LDA, C1, S1) */
        cblas_drot(istopm - k, &A[(k + 2) + (k + 1) * lda], lda,
                   &A[(k + 3) + (k + 1) * lda], lda, c1, s1);
        cblas_drot(istopm - k, &A[(k + 1) + (k + 1) * lda], lda,
                   &A[(k + 2) + (k + 1) * lda], lda, c2, s2);

        cblas_drot(istopm - k, &B[(k + 2) + (k + 1) * ldb], ldb,
                   &B[(k + 3) + (k + 1) * ldb], ldb, c1, s1);
        cblas_drot(istopm - k, &B[(k + 1) + (k + 1) * ldb], ldb,
                   &B[(k + 2) + (k + 1) * ldb], ldb, c2, s2);
        if (ilq) {
            /* Q(1, K+2-QSTART+1) in Fortran → Q[0 + (k+2-qstart)*ldq] in C */
            cblas_drot(nq, &Q[0 + (k + 2 - qstart) * ldq], 1,
                       &Q[0 + (k + 3 - qstart) * ldq], 1, c1, s1);
            cblas_drot(nq, &Q[0 + (k + 1 - qstart) * ldq], 1,
                       &Q[0 + (k + 2 - qstart) * ldq], 1, c2, s2);
        }
    }
}
