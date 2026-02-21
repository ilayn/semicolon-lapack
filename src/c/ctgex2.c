/**
 * @file ctgex2.c
 * @brief CTGEX2 swaps adjacent diagonal blocks in an upper triangular matrix
 *        pair by a unitary equivalence transformation.
 */

#include <math.h>
#include <complex.h>
#include <cblas.h>
#include "semicolon_lapack_complex_single.h"

#define LDST 2
#define WANDS 1

/**
 * CTGEX2 swaps adjacent diagonal 1 by 1 blocks (A11,B11) and (A22,B22)
 * in an upper triangular matrix pair (A, B) by a unitary equivalence
 * transformation.
 *
 * (A, B) must be in generalized Schur canonical form, that is, A and
 * B are both upper triangular.
 *
 * Optionally, the matrices Q and Z of generalized Schur vectors are
 * updated.
 *
 *        Q(in) * A(in) * Z(in)**H = Q(out) * A(out) * Z(out)**H
 *        Q(in) * B(in) * Z(in)**H = Q(out) * B(out) * Z(out)**H
 *
 * @param[in]     wantq   If nonzero, update the left transformation matrix Q.
 * @param[in]     wantz   If nonzero, update the right transformation matrix Z.
 * @param[in]     n       The order of the matrices A and B. n >= 0.
 * @param[in,out] A       Complex array of dimension (lda, n). On entry, the
 *                        matrix A. On exit, the updated matrix A.
 * @param[in]     lda     The leading dimension of A. lda >= max(1, n).
 * @param[in,out] B       Complex array of dimension (ldb, n). On entry, the
 *                        matrix B. On exit, the updated matrix B.
 * @param[in]     ldb     The leading dimension of B. ldb >= max(1, n).
 * @param[in,out] Q       Complex array of dimension (ldq, n). If wantq, the
 *                        unitary matrix Q. On exit, the updated matrix Q.
 *                        Not referenced if !wantq.
 * @param[in]     ldq     The leading dimension of Q. ldq >= 1; if wantq, ldq >= n.
 * @param[in,out] Z       Complex array of dimension (ldz, n). If wantz, the
 *                        unitary matrix Z. On exit, the updated matrix Z.
 *                        Not referenced if !wantz.
 * @param[in]     ldz     The leading dimension of Z. ldz >= 1; if wantz, ldz >= n.
 * @param[in]     j1      The index to the first block (A11, B11). 0-based.
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - = 1: the transformed matrix pair (A, B) would be
 *                                too far from generalized Schur form; the
 *                                problem is ill-conditioned.
 */
void ctgex2(
    const int wantq,
    const int wantz,
    const int n,
    c64* restrict A,
    const int lda,
    c64* restrict B,
    const int ldb,
    c64* restrict Q,
    const int ldq,
    c64* restrict Z,
    const int ldz,
    const int j1,
    int* info)
{
    const c64 CZERO = CMPLXF(0.0f, 0.0f);
    const c64 CONE = CMPLXF(1.0f, 0.0f);
    const f32 TWENTY = 20.0f;

    int weak, strong;
    int i, m;
    f32 cq, cz, eps, sa, sb, scale, smlnum, sum, thresha, threshb;
    c64 cdum, f, g, sq, sz;

    c64 s[LDST * LDST], t[LDST * LDST], work[8];

    *info = 0;

    /* Quick return if possible */
    if (n <= 1) {
        return;
    }

    m = LDST;

    /* Make a local copy of selected block in (A, B) */
    clacpy("Full", m, m, &A[j1 + j1 * lda], lda, s, LDST);
    clacpy("Full", m, m, &B[j1 + j1 * ldb], ldb, t, LDST);

    /* Compute the threshold for testing the acceptance of swapping. */
    eps = slamch("P");
    smlnum = slamch("S") / eps;
    scale = crealf(CZERO);
    sum = crealf(CONE);
    clacpy("Full", m, m, s, LDST, work, m);
    clacpy("Full", m, m, t, LDST, &work[m * m], m);
    classq(m * m, work, 1, &scale, &sum);
    sa = scale * sqrtf(sum);
    scale = crealf(CZERO);
    sum = crealf(CONE);
    classq(m * m, &work[m * m], 1, &scale, &sum);
    sb = scale * sqrtf(sum);

    /* THRES has been changed from
          THRESH = MAX( TEN*EPS*SA, SMLNUM )
       to
          THRESH = MAX( TWENTY*EPS*SA, SMLNUM )
       on 04/01/10.
       "Bug" reported by Ondra Kamenik, confirmed by Julie Langou, fixed by
       Jim Demmel and Guillaume Revy. See forum post 1783. */

    thresha = TWENTY * eps * sa > smlnum ? TWENTY * eps * sa : smlnum;
    threshb = TWENTY * eps * sb > smlnum ? TWENTY * eps * sb : smlnum;

    /* Compute unitary QL and RQ that swap 1-by-1 and 1-by-1 blocks
       using Givens rotations and perform the swap tentatively. */

    f = s[1 + 1 * LDST] * t[0 + 0 * LDST] - t[1 + 1 * LDST] * s[0 + 0 * LDST];
    g = s[1 + 1 * LDST] * t[0 + 1 * LDST] - t[1 + 1 * LDST] * s[0 + 1 * LDST];
    sa = cabsf(s[1 + 1 * LDST]) * cabsf(t[0 + 0 * LDST]);
    sb = cabsf(s[0 + 0 * LDST]) * cabsf(t[1 + 1 * LDST]);
    clartg(g, f, &cz, &sz, &cdum);
    sz = -sz;
    crot(2, &s[0 + 0 * LDST], 1, &s[0 + 1 * LDST], 1, cz, conjf(sz));
    crot(2, &t[0 + 0 * LDST], 1, &t[0 + 1 * LDST], 1, cz, conjf(sz));
    if (sa >= sb) {
        clartg(s[0 + 0 * LDST], s[1 + 0 * LDST], &cq, &sq, &cdum);
    } else {
        clartg(t[0 + 0 * LDST], t[1 + 0 * LDST], &cq, &sq, &cdum);
    }
    crot(2, &s[0 + 0 * LDST], LDST, &s[1 + 0 * LDST], LDST, cq, sq);
    crot(2, &t[0 + 0 * LDST], LDST, &t[1 + 0 * LDST], LDST, cq, sq);

    /* Weak stability test: |S21| <= O(EPS F-norm((A)))
                           and  |T21| <= O(EPS F-norm((B))) */

    weak = (cabsf(s[1 + 0 * LDST]) <= thresha) &&
           (cabsf(t[1 + 0 * LDST]) <= threshb);
    if (!weak) {
        goto L20;
    }

    if (WANDS) {

        /* Strong stability test:
              F-norm((A-QL**H*S*QR)) <= O(EPS*F-norm((A)))
           and
              F-norm((B-QL**H*T*QR)) <= O(EPS*F-norm((B))) */

        clacpy("Full", m, m, s, LDST, work, m);
        clacpy("Full", m, m, t, LDST, &work[m * m], m);
        crot(2, &work[0], 1, &work[2], 1, cz, -conjf(sz));
        crot(2, &work[4], 1, &work[6], 1, cz, -conjf(sz));
        crot(2, &work[0], 2, &work[1], 2, cq, -sq);
        crot(2, &work[4], 2, &work[5], 2, cq, -sq);
        for (i = 0; i < 2; i++) {
            work[i] = work[i] - A[(j1 + i) + j1 * lda];
            work[i + 2] = work[i + 2] - A[(j1 + i) + (j1 + 1) * lda];
            work[i + 4] = work[i + 4] - B[(j1 + i) + j1 * ldb];
            work[i + 6] = work[i + 6] - B[(j1 + i) + (j1 + 1) * ldb];
        }
        scale = crealf(CZERO);
        sum = crealf(CONE);
        classq(m * m, work, 1, &scale, &sum);
        sa = scale * sqrtf(sum);
        scale = crealf(CZERO);
        sum = crealf(CONE);
        classq(m * m, &work[m * m], 1, &scale, &sum);
        sb = scale * sqrtf(sum);
        strong = (sa <= thresha) && (sb <= threshb);
        if (!strong) {
            goto L20;
        }
    }

    /* If the swap is accepted ("weakly" and "strongly"), apply the
       equivalence transformations to the original matrix pair (A,B) */

    crot(j1 + 2, &A[0 + j1 * lda], 1, &A[0 + (j1 + 1) * lda], 1, cz,
         conjf(sz));
    crot(j1 + 2, &B[0 + j1 * ldb], 1, &B[0 + (j1 + 1) * ldb], 1, cz,
         conjf(sz));
    crot(n - j1, &A[j1 + j1 * lda], lda, &A[(j1 + 1) + j1 * lda], lda, cq,
         sq);
    crot(n - j1, &B[j1 + j1 * ldb], ldb, &B[(j1 + 1) + j1 * ldb], ldb, cq,
         sq);

    /* Set N1 by N2 (2,1) blocks to 0 */
    A[(j1 + 1) + j1 * lda] = CZERO;
    B[(j1 + 1) + j1 * ldb] = CZERO;

    /* Accumulate transformations into Q and Z if requested. */
    if (wantz) {
        crot(n, &Z[0 + j1 * ldz], 1, &Z[0 + (j1 + 1) * ldz], 1, cz,
             conjf(sz));
    }
    if (wantq) {
        crot(n, &Q[0 + j1 * ldq], 1, &Q[0 + (j1 + 1) * ldq], 1, cq,
             conjf(sq));
    }

    /* Exit with INFO = 0 if swap was successfully performed. */
    return;

    /* Exit with INFO = 1 if swap was rejected. */
L20:
    *info = 1;
    return;
}

#undef LDST
#undef WANDS
