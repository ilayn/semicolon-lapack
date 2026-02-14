/**
 * @file zlags2.c
 * @brief ZLAGS2 computes 2-by-2 unitary matrices U, V, and Q.
 */

#include <complex.h>
#include <math.h>
#include "semicolon_lapack_complex_double.h"

/**
 * ZLAGS2 computes 2-by-2 unitary matrices U, V and Q, such
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
void zlags2(const int upper,
            const double a1, const double complex a2, const double a3,
            const double b1, const double complex b2, const double b3,
            double* csu, double complex* snu,
            double* csv, double complex* snv,
            double* csq, double complex* snq)
{
    double a, aua11, aua12, aua21, aua22, avb11, avb12, avb21, avb22;
    double csl, csr, d, fb, fc, s1, s2, snl, snr;
    double ua11r, ua22r, vb11r, vb22r;
    double complex b_var, c_var, d1, r, t;
    double complex ua11, ua12, ua21, ua22;
    double complex vb11, vb12, vb21, vb22;

    if (upper) {

        /* Input matrices A and B are upper triangular matrices
         *
         * Form matrix C = A*adj(B) = ( a b )
         *                             ( 0 d )
         */
        a = a1 * b3;
        d = a3 * b1;
        b_var = a2 * b1 - a1 * b2;
        fb = cabs(b_var);

        /* Transform complex 2-by-2 matrix C to real matrix by unitary
         * diagonal matrix diag(1,D1).
         */
        d1 = CMPLX(1.0, 0.0);
        if (fb != 0.0) {
            d1 = b_var / fb;
        }

        /* The SVD of real 2 by 2 triangular C
         *
         *  ( CSL -SNL )*( A B )*(  CSR  SNR ) = ( R 0 )
         *  ( SNL  CSL ) ( 0 D ) ( -SNR  CSR )   ( 0 T )
         */
        dlasv2(a, fb, d, &s1, &s2, &snr, &csr, &snl, &csl);

        if (fabs(csl) >= fabs(snl) || fabs(csr) >= fabs(snr)) {

            /* Compute the (1,1) and (1,2) elements of U**H *A and V**H *B,
             * and (1,2) element of |U|**H *|A| and |V|**H *|B|.
             */
            ua11r = csl * a1;
            ua12 = csl * a2 + d1 * snl * a3;

            vb11r = csr * b1;
            vb12 = csr * b2 + d1 * snr * b3;

            aua12 = fabs(csl) * cabs1(a2) + fabs(snl) * fabs(a3);
            avb12 = fabs(csr) * cabs1(b2) + fabs(snr) * fabs(b3);

            /* zero (1,2) elements of U**H *A and V**H *B */
            if ((fabs(ua11r) + cabs1(ua12)) == 0.0) {
                zlartg(-CMPLX(vb11r, 0.0), conj(vb12), csq, snq, &r);
            } else if ((fabs(vb11r) + cabs1(vb12)) == 0.0) {
                zlartg(-CMPLX(ua11r, 0.0), conj(ua12), csq, snq, &r);
            } else if (aua12 / (fabs(ua11r) + cabs1(ua12)) <=
                       avb12 / (fabs(vb11r) + cabs1(vb12))) {
                zlartg(-CMPLX(ua11r, 0.0), conj(ua12), csq, snq, &r);
            } else {
                zlartg(-CMPLX(vb11r, 0.0), conj(vb12), csq, snq, &r);
            }

            *csu = csl;
            *snu = -d1 * snl;
            *csv = csr;
            *snv = -d1 * snr;

        } else {

            /* Compute the (2,1) and (2,2) elements of U**H *A and V**H *B,
             * and (2,2) element of |U|**H *|A| and |V|**H *|B|.
             */
            ua21 = -conj(d1) * snl * a1;
            ua22 = -conj(d1) * snl * a2 + csl * a3;

            vb21 = -conj(d1) * snr * b1;
            vb22 = -conj(d1) * snr * b2 + csr * b3;

            aua22 = fabs(snl) * cabs1(a2) + fabs(csl) * fabs(a3);
            avb22 = fabs(snr) * cabs1(b2) + fabs(csr) * fabs(b3);

            /* zero (2,2) elements of U**H *A and V**H *B, and then swap. */
            if ((cabs1(ua21) + cabs1(ua22)) == 0.0) {
                zlartg(-conj(vb21), conj(vb22), csq, snq, &r);
            } else if ((cabs1(vb21) + cabs(vb22)) == 0.0) {
                zlartg(-conj(ua21), conj(ua22), csq, snq, &r);
            } else if (aua22 / (cabs1(ua21) + cabs1(ua22)) <=
                       avb22 / (cabs1(vb21) + cabs1(vb22))) {
                zlartg(-conj(ua21), conj(ua22), csq, snq, &r);
            } else {
                zlartg(-conj(vb21), conj(vb22), csq, snq, &r);
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
        fc = cabs(c_var);

        /* Transform complex 2-by-2 matrix C to real matrix by unitary
         * diagonal matrix diag(d1,1).
         */
        d1 = CMPLX(1.0, 0.0);
        if (fc != 0.0) {
            d1 = c_var / fc;
        }

        /* The SVD of real 2 by 2 triangular C
         *
         *  ( CSL -SNL )*( A 0 )*(  CSR  SNR ) = ( R 0 )
         *  ( SNL  CSL ) ( C D ) ( -SNR  CSR )   ( 0 T )
         */
        dlasv2(a, fc, d, &s1, &s2, &snr, &csr, &snl, &csl);

        if (fabs(csr) >= fabs(snr) || fabs(csl) >= fabs(snl)) {

            /* Compute the (2,1) and (2,2) elements of U**H *A and V**H *B,
             * and (2,1) element of |U|**H *|A| and |V|**H *|B|.
             */
            ua21 = -d1 * snr * a1 + csr * a2;
            ua22r = csr * a3;

            vb21 = -d1 * snl * b1 + csl * b2;
            vb22r = csl * b3;

            aua21 = fabs(snr) * fabs(a1) + fabs(csr) * cabs1(a2);
            avb21 = fabs(snl) * fabs(b1) + fabs(csl) * cabs1(b2);

            /* zero (2,1) elements of U**H *A and V**H *B. */
            if ((cabs1(ua21) + fabs(ua22r)) == 0.0) {
                zlartg(CMPLX(vb22r, 0.0), vb21, csq, snq, &r);
            } else if ((cabs1(vb21) + fabs(vb22r)) == 0.0) {
                zlartg(CMPLX(ua22r, 0.0), ua21, csq, snq, &r);
            } else if (aua21 / (cabs1(ua21) + fabs(ua22r)) <=
                       avb21 / (cabs1(vb21) + fabs(vb22r))) {
                zlartg(CMPLX(ua22r, 0.0), ua21, csq, snq, &r);
            } else {
                zlartg(CMPLX(vb22r, 0.0), vb21, csq, snq, &r);
            }

            *csu = csr;
            *snu = -conj(d1) * snr;
            *csv = csl;
            *snv = -conj(d1) * snl;

        } else {

            /* Compute the (1,1) and (1,2) elements of U**H *A and V**H *B,
             * and (1,1) element of |U|**H *|A| and |V|**H *|B|.
             */
            ua11 = csr * a1 + conj(d1) * snr * a2;
            ua12 = conj(d1) * snr * a3;

            vb11 = csl * b1 + conj(d1) * snl * b2;
            vb12 = conj(d1) * snl * b3;

            aua11 = fabs(csr) * fabs(a1) + fabs(snr) * cabs1(a2);
            avb11 = fabs(csl) * fabs(b1) + fabs(snl) * cabs1(b2);

            /* zero (1,1) elements of U**H *A and V**H *B, and then swap. */
            if ((cabs1(ua11) + cabs1(ua12)) == 0.0) {
                zlartg(vb12, vb11, csq, snq, &r);
            } else if ((cabs1(vb11) + cabs1(vb12)) == 0.0) {
                zlartg(ua12, ua11, csq, snq, &r);
            } else if (aua11 / (cabs1(ua11) + cabs1(ua12)) <=
                       avb11 / (cabs1(vb11) + cabs1(vb12))) {
                zlartg(ua12, ua11, csq, snq, &r);
            } else {
                zlartg(vb12, vb11, csq, snq, &r);
            }

            *csu = snr;
            *snu = conj(d1) * csr;
            *csv = snl;
            *snv = conj(d1) * csl;
        }
    }
}
