/**
 * @file dlasd2.c
 * @brief DLASD2 merges the two sets of singular values together into a single
 *        sorted set. Used by dbdsdc.
 */

#include "semicolon_lapack_double.h"
#include <math.h>
#include "semicolon_cblas.h"

/**
 * DLASD2 merges the two sets of singular values together into a single
 * sorted set.  Then it tries to deflate the size of the problem.
 * There are two ways in which deflation can occur:  when two or more
 * singular values are close together or if there is a tiny entry in the
 * Z vector.  For each such occurrence the order of the related secular
 * equation problem is reduced by one.
 *
 * DLASD2 is called from DLASD1.
 *
 * @param[in]     nl      The row dimension of the upper block.  nl >= 1.
 * @param[in]     nr      The row dimension of the lower block.  nr >= 1.
 * @param[in]     sqre    = 0: the lower block is an nr-by-nr square matrix.
 *                        = 1: the lower block is an nr-by-(nr+1) rectangular matrix.
 *                        The bidiagonal matrix has n = nl + nr + 1 rows and
 *                        m = n + sqre >= n columns.
 * @param[out]    k       Contains the dimension of the non-deflated matrix,
 *                        This is the order of the related secular equation.
 *                        1 <= k <= n.
 * @param[in,out] D       Double precision array, dimension (n).
 *                        On entry D contains the singular values of the two
 *                        submatrices to be combined.  On exit D contains the
 *                        trailing (n-k) updated singular values (those which
 *                        were deflated) sorted into decreasing order.
 * @param[out]    Z       Double precision array, dimension (m).
 *                        On exit Z contains the updating row vector in the
 *                        secular equation.
 * @param[in]     alpha   Contains the diagonal element associated with the added row.
 * @param[in]     beta    Contains the off-diagonal element associated with the added
 *                        row.
 * @param[in,out] U       Double precision array, dimension (ldu, n).
 *                        On entry U contains the left singular vectors of two
 *                        submatrices in the two square blocks with corners at
 *                        (0,0), (nl-1, nl-1), and (nl+1, nl+1), (n-1,n-1).
 *                        On exit U contains the trailing (n-k) updated left
 *                        singular vectors (those which were deflated) in its
 *                        last n-k columns.
 * @param[in]     ldu     The leading dimension of the array U.  ldu >= n.
 * @param[in,out] VT      Double precision array, dimension (ldvt, m).
 *                        On entry VT**T contains the right singular vectors of
 *                        two submatrices in the two square blocks with corners
 *                        at (0,0), (nl, nl), and (nl+1, nl+1), (m-1,m-1).
 *                        On exit VT**T contains the trailing (n-k) updated right
 *                        singular vectors (those which were deflated) in its
 *                        last n-k columns.
 *                        In case sqre =1, the last row of VT spans the right null
 *                        space.
 * @param[in]     ldvt    The leading dimension of the array VT.  ldvt >= m.
 * @param[out]    DSIGMA  Double precision array, dimension (n).
 *                        Contains a copy of the diagonal elements (k-1 singular
 *                        values and one zero) in the secular equation.
 * @param[out]    U2      Double precision array, dimension (ldu2, n).
 *                        Contains a copy of the first k-1 left singular vectors
 *                        which will be used by DLASD3 in a matrix multiply (DGEMM)
 *                        to solve for the new left singular vectors. U2 is
 *                        arranged into four blocks. The first block contains a
 *                        column with 1 at nl and zero everywhere else; the second
 *                        block contains non-zero entries only at and above nl-1;
 *                        the third contains non-zero entries only below nl; and
 *                        the fourth is dense.
 * @param[in]     ldu2    The leading dimension of the array U2.  ldu2 >= n.
 * @param[out]    VT2     Double precision array, dimension (ldvt2, m).
 *                        VT2**T contains a copy of the first k right singular
 *                        vectors which will be used by DLASD3 in a matrix multiply
 *                        (DGEMM) to solve for the new right singular vectors. VT2
 *                        is arranged into three blocks. The first block contains
 *                        a row that corresponds to the special 0 diagonal element
 *                        in SIGMA; the second block contains non-zeros only at
 *                        and before nl; the third block contains non-zeros only
 *                        at and after nl+1.
 * @param[in]     ldvt2   The leading dimension of the array VT2.  ldvt2 >= m.
 * @param[out]    IDXP    Integer array, dimension (n).
 *                        This will contain the permutation used to place deflated
 *                        values of D at the end of the array. On output
 *                        IDXP[1:k-1] points to the nondeflated D-values and
 *                        IDXP[k:n-1] points to the deflated singular values.
 * @param[out]    IDX     Integer array, dimension (n).
 *                        This will contain the permutation used to sort the
 *                        contents of D into ascending order.
 * @param[out]    IDXC    Integer array, dimension (n).
 *                        This will contain the permutation used to arrange the
 *                        columns of the deflated U matrix into three groups:  the
 *                        first group contains non-zero entries only at and above
 *                        nl-1, the second contains non-zero entries only below
 *                        nl+1, and the third is dense.
 * @param[in,out] IDXQ    Integer array, dimension (n).
 *                        This contains the permutation which separately sorts the
 *                        two sub-problems in D into ascending order.  Note that
 *                        entries in the first half of this permutation must first
 *                        be moved one position backward; and entries in the
 *                        second half must first have nl+1 added to their values.
 * @param[out]    COLTYP  Integer array, dimension (n).
 *                        As workspace, this will contain a label which will
 *                        indicate which of the following types a column in the
 *                        U2 matrix or a row in the VT2 matrix is:
 *                        0 : non-zero in the upper half only
 *                        1 : non-zero in the lower half only
 *                        2 : dense
 *                        3 : deflated
 *                        On exit, it is an array of dimension 4, with COLTYP[i]
 *                        being the dimension of the i-th type columns.
 * @param[out]    info
 *                         - = 0: successful exit.
 *                         - < 0: if info = -i, the i-th argument had an illegal value.
 *
 * @par Contributors:
 * @rst
 * Ming Gu and Huan Ren, Computer Science Division, University of
 * California at Berkeley, USA
 * @endrst
 */
void dlasd2(const INT nl, const INT nr, const INT sqre, INT* k,
            f64* restrict D, f64* restrict Z,
            const f64 alpha, const f64 beta,
            f64* restrict U, const INT ldu,
            f64* restrict VT, const INT ldvt,
            f64* restrict DSIGMA,
            f64* restrict U2, const INT ldu2,
            f64* restrict VT2, const INT ldvt2,
            INT* restrict IDXP, INT* restrict IDX,
            INT* restrict IDXC, INT* restrict IDXQ,
            INT* restrict COLTYP, INT* info)
{
    INT ctot[4], psm[4];
    INT ct, i, idxi, idxj, idxjp, j, jp, jprev, k2, m, n;
    f64 c, eps, hlftol, s, tau, tol, z1;

    /* Test the input parameters. */
    *info = 0;

    if (nl < 1) {
        *info = -1;
    } else if (nr < 1) {
        *info = -2;
    } else if (sqre != 1 && sqre != 0) {
        *info = -3;
    }

    n = nl + nr + 1;
    m = n + sqre;

    if (ldu < n) {
        *info = -10;
    } else if (ldvt < m) {
        *info = -12;
    } else if (ldu2 < n) {
        *info = -15;
    } else if (ldvt2 < m) {
        *info = -17;
    }
    if (*info != 0) {
        xerbla("DLASD2", -(*info));
        return;
    }

    /* Generate the first part of the vector Z; and move the singular
     * values in the first part of D one position backward. */
    z1 = alpha * VT[nl + nl * ldvt];
    Z[0] = z1;
    for (i = nl - 1; i >= 0; i--) {
        Z[i + 1] = alpha * VT[i + nl * ldvt];
        D[i + 1] = D[i];
        IDXQ[i + 1] = IDXQ[i] + 1;
    }

    /* Generate the second part of the vector Z. */
    for (i = nl + 1; i < m; i++) {
        Z[i] = beta * VT[i + (nl + 1) * ldvt];
    }

    /* Initialize some reference arrays. */
    for (i = 1; i <= nl; i++) {
        COLTYP[i] = 0;
    }
    for (i = nl + 1; i < n; i++) {
        COLTYP[i] = 1;
    }

    /* Sort the singular values into increasing order */
    for (i = nl + 1; i < n; i++) {
        IDXQ[i] = IDXQ[i] + nl + 1;
    }

    /* DSIGMA, IDXC, IDXC, and the first column of U2
     * are used as storage space. */
    for (i = 1; i < n; i++) {
        DSIGMA[i] = D[IDXQ[i]];
        U2[i] = Z[IDXQ[i]];
        IDXC[i] = COLTYP[IDXQ[i]];
    }

    dlamrg(nl, nr, &DSIGMA[1], 1, 1, &IDX[1]);

    for (i = 1; i < n; i++) {
        idxi = 1 + IDX[i];
        D[i] = DSIGMA[idxi];
        Z[i] = U2[idxi];
        COLTYP[i] = IDXC[idxi];
    }

    /* Calculate the allowable deflation tolerance */
    eps = dlamch("Epsilon");
    tol = fabs(alpha) > fabs(beta) ? fabs(alpha) : fabs(beta);
    tol = 8.0 * 8.0 * eps * (fabs(D[n - 1]) > tol ? fabs(D[n - 1]) : tol);

    /* There are 2 kinds of deflation -- first a value in the z-vector
     * is small, second two (or more) singular values are very close
     * together (their difference is small).
     *
     * If the value in the z-vector is small, we simply permute the
     * array so that the corresponding singular value is moved to the
     * end.
     *
     * If two values in the D-vector are close, we perform a two-sided
     * rotation designed to make one of the corresponding z-vector
     * entries zero, and then permute the array so that the deflated
     * singular value is moved to the end.
     *
     * If there are multiple singular values then the problem deflates.
     * Here the number of equal singular values are found.  As each equal
     * singular value is found, an elementary reflector is computed to
     * rotate the corresponding singular subspace so that the
     * corresponding components of Z are zero in this new basis. */
    *k = 1;
    k2 = n;
    jprev = -1;

    for (j = 1; j < n; j++) {
        if (fabs(Z[j]) <= tol) {
            /* Deflate due to small z component. */
            k2--;
            IDXP[k2] = j;
            COLTYP[j] = 3;
            if (j == n - 1) {
                goto L120;
            }
        } else {
            jprev = j;
            break;
        }
    }

    j = jprev;
    while (1) {
        j++;
        if (j >= n) {
            break;
        }
        if (fabs(Z[j]) <= tol) {
            /* Deflate due to small z component. */
            k2--;
            IDXP[k2] = j;
            COLTYP[j] = 3;
        } else {
            /* Check if singular values are close enough to allow deflation. */
            if ((D[j] - D[jprev]) <= tol) {
                /* Deflation is possible. */
                s = Z[jprev];
                c = Z[j];

                /* Find sqrt(a**2+b**2) without overflow or
                 * destructive underflow. */
                tau = dlapy2(c, s);
                c = c / tau;
                s = -s / tau;
                Z[j] = tau;
                Z[jprev] = 0.0;

                /* Apply back the Givens rotation to the left and right
                 * singular vector matrices. */
                idxjp = IDXQ[IDX[jprev] + 1];
                idxj = IDXQ[IDX[j] + 1];
                if (idxjp <= nl) {
                    idxjp--;
                }
                if (idxj <= nl) {
                    idxj--;
                }
                cblas_drot(n, &U[idxjp * ldu], 1, &U[idxj * ldu], 1, c, s);
                cblas_drot(m, &VT[idxjp], ldvt, &VT[idxj], ldvt, c, s);

                if (COLTYP[j] != COLTYP[jprev]) {
                    COLTYP[j] = 2;
                }
                COLTYP[jprev] = 3;
                k2--;

                /* Insert the deflated index in the correct position in IDXP.
                 * If J - JPREV is greater than 1, the indices in between
                 * must be shifted to preserve the correct output order. */
                for (jp = jprev; jp < j; jp++) {
                    IDXP[k2 + j - 1 - jp] = jp;
                }
                jprev = j;
            } else {
                (*k)++;
                U2[*k - 1] = Z[jprev];
                DSIGMA[*k - 1] = D[jprev];
                IDXP[*k - 1] = jprev;
                jprev = j;
            }
        }
    }

    /* Record the last singular value. */
    (*k)++;
    U2[*k - 1] = Z[jprev];
    DSIGMA[*k - 1] = D[jprev];
    IDXP[*k - 1] = jprev;

L120:
    /* Count up the total number of the various types of columns, then
     * form a permutation which positions the four column types into
     * four groups of uniform structure (although one or more of these
     * groups may be empty). */
    for (j = 0; j < 4; j++) {
        ctot[j] = 0;
    }
    for (j = 1; j < n; j++) {
        ct = COLTYP[j];
        ctot[ct]++;
    }

    /* PSM(*) = Position in SubMatrix (of types 1 through 4) */
    psm[0] = 1;
    psm[1] = 1 + ctot[0];
    psm[2] = psm[1] + ctot[1];
    psm[3] = psm[2] + ctot[2];

    /* Fill out the IDXC array so that the permutation which it induces
     * will place all type-1 columns first, all type-2 columns next,
     * then all type-3's, and finally all type-4's, starting from the
     * second column. This applies similarly to the rows of VT. */
    for (j = 1; j < n; j++) {
        jp = IDXP[j];
        ct = COLTYP[jp];
        IDXC[psm[ct]] = j;
        psm[ct]++;
    }

    /* Sort the singular values and corresponding singular vectors into
     * DSIGMA, U2, and VT2 respectively.  The singular values/vectors
     * which were not deflated go into the first K slots of DSIGMA, U2,
     * and VT2 respectively, while those which were deflated go into the
     * last N - K slots, except that the first column/row will be treated
     * separately. */
    for (j = 1; j < n; j++) {
        jp = IDXP[j];
        DSIGMA[j] = D[jp];
        idxj = IDXQ[IDX[IDXP[IDXC[j]]] + 1];
        if (idxj <= nl) {
            idxj--;
        }
        cblas_dcopy(n, &U[idxj * ldu], 1, &U2[j * ldu2], 1);
        cblas_dcopy(m, &VT[idxj], ldvt, &VT2[j], ldvt2);
    }

    /* Determine DSIGMA[0], DSIGMA[1] and Z[0] */
    DSIGMA[0] = 0.0;
    hlftol = tol / 2.0;
    if (fabs(DSIGMA[1]) <= hlftol) {
        DSIGMA[1] = hlftol;
    }
    if (m > n) {
        Z[0] = dlapy2(z1, Z[m - 1]);
        if (Z[0] <= tol) {
            c = 1.0;
            s = 0.0;
            Z[0] = tol;
        } else {
            c = z1 / Z[0];
            s = Z[m - 1] / Z[0];
        }
    } else {
        if (fabs(z1) <= tol) {
            Z[0] = tol;
        } else {
            Z[0] = z1;
        }
    }

    /* Move the rest of the updating row to Z. */
    cblas_dcopy(*k - 1, &U2[1], 1, &Z[1], 1);

    /* Determine the first column of U2, the first row of VT2 and the
     * last row of VT. */
    dlaset("A", n, 1, 0.0, 0.0, U2, ldu2);
    U2[nl] = 1.0;

    if (m > n) {
        for (i = 0; i <= nl; i++) {
            VT[m - 1 + i * ldvt] = -s * VT[nl + i * ldvt];
            VT2[i * ldvt2] = c * VT[nl + i * ldvt];
        }
        for (i = nl + 1; i < m; i++) {
            VT2[i * ldvt2] = s * VT[m - 1 + i * ldvt];
            VT[m - 1 + i * ldvt] = c * VT[m - 1 + i * ldvt];
        }
    } else {
        cblas_dcopy(m, &VT[nl], ldvt, VT2, ldvt2);
    }
    if (m > n) {
        cblas_dcopy(m, &VT[m - 1], ldvt, &VT2[m - 1], ldvt2);
    }

    /* The deflated singular values and their corresponding vectors go
     * into the back of D, U, and V respectively. */
    if (n > *k) {
        cblas_dcopy(n - *k, &DSIGMA[*k], 1, &D[*k], 1);
        dlacpy("A", n, n - *k, &U2[*k * ldu2], ldu2, &U[*k * ldu], ldu);
        dlacpy("A", n - *k, m, &VT2[*k], ldvt2, &VT[*k], ldvt);
    }

    /* Copy CTOT into COLTYP for referencing in DLASD3. */
    for (j = 0; j < 4; j++) {
        COLTYP[j] = ctot[j];
    }
}
