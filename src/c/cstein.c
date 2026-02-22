/**
 * @file cstein.c
 * @brief CSTEIN computes eigenvectors of a real symmetric tridiagonal matrix
 *        T corresponding to specified eigenvalues, using inverse iteration.
 */

#include <math.h>
#include <stdint.h>
#include <complex.h>
#include "semicolon_cblas.h"
#include "semicolon_lapack_complex_single.h"

/**
 * CSTEIN computes the eigenvectors of a real symmetric tridiagonal
 * matrix T corresponding to specified eigenvalues, using inverse
 * iteration.
 *
 * The maximum number of iterations allowed for each eigenvector is
 * specified by an internal parameter MAXITS (currently set to 5).
 *
 * Although the eigenvectors are real, they are stored in a complex
 * array, which may be passed to CUNMTR or CUPMTR for back
 * transformation to the eigenvectors of a complex Hermitian matrix
 * which was reduced to tridiagonal form.
 *
 * @param[in]     n       The order of the matrix. n >= 0.
 * @param[in]     D       Single precision array, dimension (n).
 *                        The n diagonal elements of the tridiagonal matrix T.
 * @param[in]     E       Single precision array, dimension (n-1).
 *                        The (n-1) subdiagonal elements of the tridiagonal
 *                        matrix T.
 * @param[in]     m       The number of eigenvectors to be found. 0 <= m <= n.
 * @param[in]     W       Single precision array, dimension (m).
 *                        The eigenvalues for which eigenvectors are to be
 *                        computed. The eigenvalues should be grouped by
 *                        split-off block and ordered from smallest to largest
 *                        within the block. (The output array W from SSTEBZ
 *                        with ORDER = 'B' is expected here.)
 * @param[in]     iblock  Integer array, dimension (m).
 *                        The submatrix indices associated with the
 *                        corresponding eigenvalues in W; iblock[i]=1 if
 *                        eigenvalue W[i] belongs to the first submatrix from
 *                        the top, =2 if W[i] belongs to the second submatrix,
 *                        etc. (1-based block numbers.)
 * @param[in]     isplit  Integer array, dimension (nsplit).
 *                        The splitting points, at which T breaks up into
 *                        submatrices. The first submatrix consists of
 *                        rows/columns 0 to isplit[0], the second of
 *                        rows/columns isplit[0]+1 through isplit[1], etc.
 *                        (0-based endpoint indices.)
 * @param[out]    Z       Complex*16 array, dimension (ldz, m).
 *                        The computed eigenvectors. The eigenvector associated
 *                        with the eigenvalue W[i] is stored in the i-th column
 *                        of Z. Any vector which fails to converge is set to its
 *                        current iterate after MAXITS iterations.
 *                        The imaginary parts of the eigenvectors are set to zero.
 * @param[in]     ldz     The leading dimension of the array Z. ldz >= max(1,n).
 * @param[out]    work    Single precision array, dimension (5*n).
 * @param[out]    iwork   Integer array, dimension (n).
 * @param[out]    ifail   Integer array, dimension (m).
 *                        On normal exit, all elements of ifail are zero.
 *                        If one or more eigenvectors fail to converge after
 *                        MAXITS iterations, then their indices are stored in
 *                        array ifail (0-based indices).
 * @param[out]    info
 *                         - = 0: successful exit.
 *                         - < 0: if info = -i, the i-th argument had an illegal
 *                           value.
 *                         - > 0: if info = i, then i eigenvectors failed to
 *                           converge in MAXITS iterations. Their indices are
 *                           stored in array ifail.
 */
void cstein(
    const INT n,
    const f32* restrict D,
    const f32* restrict E,
    const INT m,
    const f32* restrict W,
    const INT* restrict iblock,
    const INT* restrict isplit,
    c64* restrict Z,
    const INT ldz,
    f32* restrict work,
    INT* restrict iwork,
    INT* restrict ifail,
    INT* info)
{
    const INT MAXITS = 5;
    const INT EXTRA = 2;
    const f32 ODM3 = 1.0e-3f;
    const f32 ODM1 = 1.0e-1f;

    INT b1, blksiz, bn, gpind = 0, i, iinfo, its, j, j1, jblk, jmax, jr, nblk, nrmchk;
    f32 dtpcrt, eps, eps1, nrm, onenrm, ortol, pertol, scl, sep, tol, xj, xjm = 0.0f, ztr;
    uint64_t seed;

    /* Workspace offsets: each segment has n elements */
    const INT indrv1 = 0;
    const INT indrv2 = n;
    const INT indrv3 = 2 * n;
    const INT indrv4 = 3 * n;
    const INT indrv5 = 4 * n;

    /* Test the input parameters. */
    *info = 0;
    for (i = 0; i < m; i++) {
        ifail[i] = 0;
    }

    if (n < 0) {
        *info = -1;
    } else if (m < 0 || m > n) {
        *info = -4;
    } else if (ldz < (n > 1 ? n : 1)) {
        *info = -9;
    } else {
        for (j = 1; j < m; j++) {
            if (iblock[j] < iblock[j - 1]) {
                *info = -6;
                break;
            }
            if (iblock[j] == iblock[j - 1] && W[j] < W[j - 1]) {
                *info = -5;
                break;
            }
        }
    }

    if (*info != 0) {
        xerbla("CSTEIN", -(*info));
        return;
    }

    /* Quick return if possible */
    if (n == 0 || m == 0) {
        return;
    } else if (n == 1) {
        Z[0] = CMPLXF(1.0f, 0.0f);
        return;
    }

    /* Get machine constant: eps = eps*base (precision) */
    eps = slamch("P");

    /* Initialize seed for LCG random number generator. */
    seed = 1ULL;

    /*
     * Compute eigenvectors of matrix blocks.
     *
     * j1 is the 0-based index of the first eigenvalue in the current block.
     */
    j1 = 0;
    for (nblk = 1; nblk <= iblock[m - 1]; nblk++) {

        /*
         * Find starting and ending indices of block nblk (0-based).
         * isplit stores 0-based endpoint indices.
         */
        if (nblk == 1) {
            b1 = 0;
        } else {
            b1 = isplit[nblk - 2] + 1;
        }
        bn = isplit[nblk - 1];
        blksiz = bn - b1 + 1;

        if (blksiz != 1) {
            gpind = j1;

            /*
             * Compute reorthogonalization criterion and stopping criterion.
             */
            onenrm = fabsf(D[b1]) + fabsf(E[b1]);
            onenrm = fmaxf(onenrm, fabsf(D[bn]) + fabsf(E[bn - 1]));
            for (i = b1 + 1; i <= bn - 1; i++) {
                onenrm = fmaxf(onenrm, fabsf(D[i]) + fabsf(E[i - 1]) + fabsf(E[i]));
            }
            ortol = ODM3 * onenrm;

            dtpcrt = sqrtf(ODM1 / blksiz);
        }

        /* Loop through eigenvalues of block nblk. */
        jblk = 0;
        for (j = j1; j < m; j++) {
            if (iblock[j] != nblk) {
                j1 = j;
                break;
            }
            jblk++;
            xj = W[j];

            /* Skip all the work if the block size is one. */
            if (blksiz == 1) {
                work[indrv1] = 1.0f;
                goto store_eigenvector;
            }

            /*
             * If eigenvalues j and j-1 are too close, add a relatively
             * small perturbation.
             */
            if (jblk > 1) {
                eps1 = fabsf(eps * xj);
                pertol = 10.0f * eps1;
                sep = xj - xjm;
                if (sep < pertol) {
                    xj = xjm + pertol;
                }
            }

            its = 0;
            nrmchk = 0;

            /*
             * Get random starting vector using LCG.
             * Generates values in approximately [-1, 1].
             */
            for (i = 0; i < blksiz; i++) {
                seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
                work[indrv1 + i] = (f32)((int64_t)(seed >> 33)) / (f32)(1LL << 31);
            }

            /*
             * Copy the matrix T so it won't be destroyed in factorization.
             * indrv4: diagonal (D[b1..bn])
             * indrv2: upper subdiagonal (E[b1..bn-1]), stored at offset +1
             * indrv3: lower subdiagonal (E[b1..bn-1])
             */
            cblas_scopy(blksiz, &D[b1], 1, &work[indrv4], 1);
            cblas_scopy(blksiz - 1, &E[b1], 1, &work[indrv2 + 1], 1);
            cblas_scopy(blksiz - 1, &E[b1], 1, &work[indrv3], 1);

            /*
             * Compute LU factors with partial pivoting (PT = LU).
             * slagtf factors (T - xj*I).
             */
            tol = 0.0f;
            slagtf(blksiz, &work[indrv4], xj,
                   &work[indrv2 + 1],
                   &work[indrv3], tol, &work[indrv5], iwork,
                   &iinfo);

            /* Inverse iteration loop. */
            for (;;) {
                its++;
                if (its > MAXITS) {
                    goto convergence_failure;
                }

                /*
                 * Normalize and scale the righthand side vector Pb.
                 * jmax = index of element with largest absolute value.
                 * cblas_idamax returns 0-based index.
                 */
                jmax = (INT)cblas_isamax(blksiz, &work[indrv1], 1);
                scl = (f32)blksiz * onenrm *
                      fmaxf(eps, fabsf(work[indrv4 + blksiz - 1])) /
                      fabsf(work[indrv1 + jmax]);
                cblas_sscal(blksiz, scl, &work[indrv1], 1);

                /*
                 * Solve the system LU = Pb.
                 * slagts with job=-1 solves (T - lambda*I)x = y with perturbation.
                 */
                slagts(-1, blksiz, &work[indrv4],
                       &work[indrv2 + 1],
                       &work[indrv3], &work[indrv5], iwork,
                       &work[indrv1], &tol, &iinfo);

                /*
                 * Reorthogonalize by modified Gram-Schmidt if eigenvalues are
                 * close enough.
                 */
                if (jblk == 1) {
                    goto check_norm;
                }
                if (fabsf(xj - xjm) > ortol) {
                    gpind = j;
                }
                if (gpind != j) {
                    for (i = gpind; i < j; i++) {
                        ztr = 0.0f;
                        for (jr = 0; jr < blksiz; jr++) {
                            ztr += work[indrv1 + jr] *
                                   crealf(Z[b1 + jr + i * ldz]);
                        }
                        for (jr = 0; jr < blksiz; jr++) {
                            work[indrv1 + jr] -= ztr *
                                                 crealf(Z[b1 + jr + i * ldz]);
                        }
                    }
                }

check_norm:
                /* Check the infinity norm of the iterate. */
                jmax = (INT)cblas_isamax(blksiz, &work[indrv1], 1);
                nrm = fabsf(work[indrv1 + jmax]);

                /*
                 * Continue for additional iterations after norm reaches
                 * stopping criterion.
                 */
                if (nrm < dtpcrt) {
                    continue;
                }
                nrmchk++;
                if (nrmchk < EXTRA + 1) {
                    continue;
                }

                goto accept_eigenvector;
            }

convergence_failure:
            /*
             * If stopping criterion was not satisfied, update info and
             * store eigenvector index in array ifail.
             */
            (*info)++;
            ifail[(*info) - 1] = j;

accept_eigenvector:
            /* Accept iterate as j-th eigenvector. Normalize. */
            scl = 1.0f / cblas_snrm2(blksiz, &work[indrv1], 1);
            jmax = (INT)cblas_isamax(blksiz, &work[indrv1], 1);
            if (work[indrv1 + jmax] < 0.0f) {
                scl = -scl;
            }
            cblas_sscal(blksiz, scl, &work[indrv1], 1);

store_eigenvector:
            /* Zero out the full column, then copy block portion. */
            for (i = 0; i < n; i++) {
                Z[i + j * ldz] = CMPLXF(0.0f, 0.0f);
            }
            for (i = 0; i < blksiz; i++) {
                Z[b1 + i + j * ldz] = CMPLXF(work[indrv1 + i], 0.0f);
            }

            /* Save the shift to check eigenvalue spacing at next iteration. */
            xjm = xj;
        }

        /* If we exhausted all m eigenvalues without breaking, update j1. */
        if (j >= m) {
            j1 = m;
        }
    }
}
