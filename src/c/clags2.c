/**
 * @file clags2.c
 * @brief CLAGS2 computes 2-by-2 unitary matrices U, V, and Q.
 */

#include "internal_build_defs.h"
#include <complex.h>
#include <math.h>
#include "semicolon_lapack_complex_single.h"

/**
 * CLAGS2 computes 2-by-2 unitary matrices U, V and Q, such
 * that if ( UPPER ) then
 *
 *           U**H *A*Q = U**H *( A1 A2 )*Q = ( x  0  )
 *                             ( 0  A3 )     ( x  x  )
 * and
 *           V**H*B*Q = V**H *( B1 B2 )*Q = ( x  0  )
 *                            ( 0  B3 )     ( x  x  )
 *
 * or if ( .NOT.UPPER ) then
 *
 *           U**H *A*Q = U**H *( A1 0  )*Q = ( x  x  )
 *                             ( A2 A3 )     ( 0  x  )
 * and
 *           V**H *B*Q = V**H *( B1 0  )*Q = ( x  x  )
 *                             ( B2 B3 )     ( 0  x  )
 * where
 *
 *   U = (   CSU    SNU ), V = (  CSV    SNV ),
 *       ( -SNU**H  CSU )      ( -SNV**H CSV )
 *
 *   Q = (   CSQ    SNQ )
 *       ( -SNQ**H  CSQ )
 *
 * The rows of the transformed A and B are parallel. Moreover, if the
 * input 2-by-2 matrix A is not zero, then the transformed (1,1) entry
 * of A is not zero. If the input matrices A and B are both not zero,
 * then the transformed (2,2) element of B is not zero, except when the
 * first rows of input A and B are parallel and the second rows are
 * zero.
 *
 * @param[in]  upper  = nonzero: the input matrices A and B are upper triangular.
 *                    = 0: the input matrices A and B are lower triangular.
 * @param[in]  a1     Element of the input 2-by-2 upper (lower) triangular matrix A.
 * @param[in]  a2     Element of the input 2-by-2 upper (lower) triangular matrix A.
 * @param[in]  a3     Element of the input 2-by-2 upper (lower) triangular matrix A.
 * @param[in]  b1     Element of the input 2-by-2 upper (lower) triangular matrix B.
 * @param[in]  b2     Element of the input 2-by-2 upper (lower) triangular matrix B.
 * @param[in]  b3     Element of the input 2-by-2 upper (lower) triangular matrix B.
 * @param[out] csu    The desired unitary matrix U.
 * @param[out] snu    The desired unitary matrix U.
 * @param[out] csv    The desired unitary matrix V.
 * @param[out] snv    The desired unitary matrix V.
 * @param[out] csq    The desired unitary matrix Q.
 * @param[out] snq    The desired unitary matrix Q.
 */
void clags2(const INT upper,
            const f32 a1, const c64 a2, const f32 a3,
            const f32 b1, const c64 b2, const f32 b3,
            f32* csu, c64* snu,
            f32* csv, c64* snv,
            f32* csq, c64* snq)
{
    f32 a, aua11, aua12, aua21, aua22, avb11, avb12, avb21, avb22;
    f32 csl, csr, d, fb, fc, s1, s2, snl, snr;
    f32 ua11r, ua22r, vb11r, vb22r;
    c64 b_var, c_var, d1, r;
    c64 ua11, ua12, ua21, ua22;
    c64 vb11, vb12, vb21, vb22;

    if (upper) {

        /* Input matrices A and B are upper triangular matrices
         *
         * Form matrix C = A*adj(B) = ( a b )
         *                             ( 0 d )
         */
        a = a1 * b3;
        d = a3 * b1;
        b_var = a2 * b1 - a1 * b2;
        fb = cabsf(b_var);

        /* Transform complex 2-by-2 matrix C to real matrix by unitary
         * diagonal matrix diag(1,D1).
         */
        d1 = CMPLXF(1.0f, 0.0f);
        if (fb != 0.0f) {
            d1 = b_var / fb;
        }

        /* The SVD of real 2 by 2 triangular C
         *
         *  ( CSL -SNL )*( A B )*(  CSR  SNR ) = ( R 0 )
         *  ( SNL  CSL ) ( 0 D ) ( -SNR  CSR )   ( 0 T )
         */
        slasv2(a, fb, d, &s1, &s2, &snr, &csr, &snl, &csl);

        if (fabsf(csl) >= fabsf(snl) || fabsf(csr) >= fabsf(snr)) {

            /* Compute the (1,1) and (1,2) elements of U**H *A and V**H *B,
             * and (1,2) element of |U|**H *|A| and |V|**H *|B|.
             */
            ua11r = csl * a1;
            ua12 = csl * a2 + d1 * snl * a3;

            vb11r = csr * b1;
            vb12 = csr * b2 + d1 * snr * b3;

            aua12 = fabsf(csl) * cabs1f(a2) + fabsf(snl) * fabsf(a3);
            avb12 = fabsf(csr) * cabs1f(b2) + fabsf(snr) * fabsf(b3);

            /* zero (1,2) elements of U**H *A and V**H *B */
            if ((fabsf(ua11r) + cabs1f(ua12)) == 0.0f) {
                clartg(-CMPLXF(vb11r, 0.0f), conjf(vb12), csq, snq, &r);
            } else if ((fabsf(vb11r) + cabs1f(vb12)) == 0.0f) {
                clartg(-CMPLXF(ua11r, 0.0f), conjf(ua12), csq, snq, &r);
            } else if (aua12 / (fabsf(ua11r) + cabs1f(ua12)) <=
                       avb12 / (fabsf(vb11r) + cabs1f(vb12))) {
                clartg(-CMPLXF(ua11r, 0.0f), conjf(ua12), csq, snq, &r);
            } else {
                clartg(-CMPLXF(vb11r, 0.0f), conjf(vb12), csq, snq, &r);
            }

            *csu = csl;
            *snu = -d1 * snl;
            *csv = csr;
            *snv = -d1 * snr;

        } else {

            /* Compute the (2,1) and (2,2) elements of U**H *A and V**H *B,
             * and (2,2) element of |U|**H *|A| and |V|**H *|B|.
             */
            ua21 = -conjf(d1) * snl * a1;
            ua22 = -conjf(d1) * snl * a2 + csl * a3;

            vb21 = -conjf(d1) * snr * b1;
            vb22 = -conjf(d1) * snr * b2 + csr * b3;

            aua22 = fabsf(snl) * cabs1f(a2) + fabsf(csl) * fabsf(a3);
            avb22 = fabsf(snr) * cabs1f(b2) + fabsf(csr) * fabsf(b3);

            /* zero (2,2) elements of U**H *A and V**H *B, and then swap. */
            if ((cabs1f(ua21) + cabs1f(ua22)) == 0.0f) {
                clartg(-conjf(vb21), conjf(vb22), csq, snq, &r);
            } else if ((cabs1f(vb21) + cabsf(vb22)) == 0.0f) {
                clartg(-conjf(ua21), conjf(ua22), csq, snq, &r);
            } else if (aua22 / (cabs1f(ua21) + cabs1f(ua22)) <=
                       avb22 / (cabs1f(vb21) + cabs1f(vb22))) {
                clartg(-conjf(ua21), conjf(ua22), csq, snq, &r);
            } else {
                clartg(-conjf(vb21), conjf(vb22), csq, snq, &r);
            }

            *csu = snl;
            *snu = d1 * csl;
            *csv = snr;
            *snv = d1 * csr;
        }

    } else {

        /* Input matrices A and B are lower triangular matrices
         *
         * Form matrix C = A*adj(B) = ( a 0 )
         *                             ( c d )
         */
        a = a1 * b3;
        d = a3 * b1;
        c_var = a2 * b3 - a3 * b2;
        fc = cabsf(c_var);

        /* Transform complex 2-by-2 matrix C to real matrix by unitary
         * diagonal matrix diag(d1,1).
         */
        d1 = CMPLXF(1.0f, 0.0f);
        if (fc != 0.0f) {
            d1 = c_var / fc;
        }

        /* The SVD of real 2 by 2 triangular C
         *
         *  ( CSL -SNL )*( A 0 )*(  CSR  SNR ) = ( R 0 )
         *  ( SNL  CSL ) ( C D ) ( -SNR  CSR )   ( 0 T )
         */
        slasv2(a, fc, d, &s1, &s2, &snr, &csr, &snl, &csl);

        if (fabsf(csr) >= fabsf(snr) || fabsf(csl) >= fabsf(snl)) {

            /* Compute the (2,1) and (2,2) elements of U**H *A and V**H *B,
             * and (2,1) element of |U|**H *|A| and |V|**H *|B|.
             */
            ua21 = -d1 * snr * a1 + csr * a2;
            ua22r = csr * a3;

            vb21 = -d1 * snl * b1 + csl * b2;
            vb22r = csl * b3;

            aua21 = fabsf(snr) * fabsf(a1) + fabsf(csr) * cabs1f(a2);
            avb21 = fabsf(snl) * fabsf(b1) + fabsf(csl) * cabs1f(b2);

            /* zero (2,1) elements of U**H *A and V**H *B. */
            if ((cabs1f(ua21) + fabsf(ua22r)) == 0.0f) {
                clartg(CMPLXF(vb22r, 0.0f), vb21, csq, snq, &r);
            } else if ((cabs1f(vb21) + fabsf(vb22r)) == 0.0f) {
                clartg(CMPLXF(ua22r, 0.0f), ua21, csq, snq, &r);
            } else if (aua21 / (cabs1f(ua21) + fabsf(ua22r)) <=
                       avb21 / (cabs1f(vb21) + fabsf(vb22r))) {
                clartg(CMPLXF(ua22r, 0.0f), ua21, csq, snq, &r);
            } else {
                clartg(CMPLXF(vb22r, 0.0f), vb21, csq, snq, &r);
            }

            *csu = csr;
            *snu = -conjf(d1) * snr;
            *csv = csl;
            *snv = -conjf(d1) * snl;

        } else {

            /* Compute the (1,1) and (1,2) elements of U**H *A and V**H *B,
             * and (1,1) element of |U|**H *|A| and |V|**H *|B|.
             */
            ua11 = csr * a1 + conjf(d1) * snr * a2;
            ua12 = conjf(d1) * snr * a3;

            vb11 = csl * b1 + conjf(d1) * snl * b2;
            vb12 = conjf(d1) * snl * b3;

            aua11 = fabsf(csr) * fabsf(a1) + fabsf(snr) * cabs1f(a2);
            avb11 = fabsf(csl) * fabsf(b1) + fabsf(snl) * cabs1f(b2);

            /* zero (1,1) elements of U**H *A and V**H *B, and then swap. */
            if ((cabs1f(ua11) + cabs1f(ua12)) == 0.0f) {
                clartg(vb12, vb11, csq, snq, &r);
            } else if ((cabs1f(vb11) + cabs1f(vb12)) == 0.0f) {
                clartg(ua12, ua11, csq, snq, &r);
            } else if (aua11 / (cabs1f(ua11) + cabs1f(ua12)) <=
                       avb11 / (cabs1f(vb11) + cabs1f(vb12))) {
                clartg(ua12, ua11, csq, snq, &r);
            } else {
                clartg(vb12, vb11, csq, snq, &r);
            }

            *csu = snr;
            *snu = conjf(d1) * csr;
            *csv = snl;
            *snv = conjf(d1) * csl;
        }
    }
}
