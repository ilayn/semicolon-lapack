/**
 * @file verify.h
 * @brief Prototypes for complex single (c-prefix) LAPACK test verification routines.
 *
 * These are ports of the verification routines from LAPACK/TESTING/LIN/
 * and LAPACK/TESTING/EIG/.
 */

#ifndef VERIFY_C_H
#define VERIFY_C_H

#include <stdint.h>
#include "semicolon_lapack_complex_single.h"

/* Real-valued shared verification routines (from d-prefix, precision-independent) */
f32 sget06(const f32 rcond, const f32 rcondc);
void ssvdct(const INT n, const f32* s, const f32* e, const f32 shift, INT* num);
void ssvdch(const INT n, const f32* s, const f32* e,
            const f32* svd, const f32 tol, INT* info);
void sstect(const INT n, const f32* a, const f32* b,
            const f32 shift, INT* num);
void sstech(const INT n, const f32* const restrict A, const f32* const restrict B,
            const f32* const restrict eig, const f32 tol,
            f32* const restrict work, INT* info);
f32 ssxt1(const INT ijob, const f32* const restrict D1, const INT n1,
          const f32* const restrict D2, const INT n2,
          const f32 abstol, const f32 ulp, const f32 unfl);

/* General (CGE) verification routines */
void cget01(const INT m, const INT n, const c64* const restrict A, const INT lda,
            c64* const restrict AFAC, const INT ldafac, const INT* const restrict ipiv,
            f32* const restrict rwork, f32* resid);

void cget02(const char* trans, const INT m, const INT n, const INT nrhs,
            const c64* const restrict A, const INT lda, const c64* const restrict X,
            const INT ldx, c64* const restrict B, const INT ldb,
            f32* const restrict rwork, f32* resid);

void cget03(const INT n, const c64* const restrict A, const INT lda,
            const c64* const restrict AINV, const INT ldainv, c64* const restrict work,
            const INT ldwork, f32* const restrict rwork, f32* rcond, f32* resid);

void cget04(const INT n, const INT nrhs, const c64* const restrict X, const INT ldx,
            const c64* const restrict XACT, const INT ldxact, const f32 rcond,
            f32* resid);

void cget07(const char* trans, const INT n, const INT nrhs,
            const c64* const restrict A, const INT lda,
            const c64* const restrict B, const INT ldb,
            const c64* const restrict X, const INT ldx,
            const c64* const restrict XACT, const INT ldxact,
            const f32* const restrict ferr, const INT chkferr,
            const f32* const restrict berr, f32* const restrict reslts);

void cget08(const char* trans, const INT m, const INT n, const INT nrhs,
            const c64* A, const INT lda, const c64* X, const INT ldx,
            c64* B, const INT ldb, f32* rwork, f32* resid);

void cget10(const INT m, const INT n,
            const c64* const restrict A, const INT lda,
            const c64* const restrict B, const INT ldb,
            c64* const restrict work, f32* const restrict rwork,
            f32* result);

void cget35(f32* rmax, INT* lmax, INT* ninfo, INT* knt);

void cget51(const INT itype, const INT n,
            const c64* A, const INT lda,
            const c64* B, const INT ldb,
            const c64* U, const INT ldu,
            const c64* V, const INT ldv,
            c64* work, f32* rwork, f32* result);

/* Eigenvalue selection function for generalized Schur tests */
INT clctes(const c64* z, const c64* d);

/* Stateful eigenvalue selection function for CGGESX reordering tests */
void clctsx_reset(INT m, INT n, INT mplusn);
INT clctsx(const c64* alpha, const c64* beta);

/* Generalized Schur decomposition verify */
void cget54(const INT n, const c64* A, const INT lda,
            const c64* B, const INT ldb,
            const c64* S, const INT lds,
            const c64* T, const INT ldt,
            const c64* U, const INT ldu,
            const c64* V, const INT ldv,
            c64* work, f32* result);

/* Banded (GB) verification routines */
void cgbt01(INT m, INT n, INT kl, INT ku,
            const c64* A, INT lda,
            const c64* AFAC, INT ldafac,
            const INT* ipiv,
            c64* work,
            f32* resid);

void cgbt02(const char* trans, INT m, INT n, INT kl, INT ku, INT nrhs,
            const c64* A, INT lda,
            const c64* X, INT ldx,
            c64* B, INT ldb,
            f32* rwork,
            f32* resid);

void cgbt05(const char* trans, INT n, INT kl, INT ku, INT nrhs,
            const c64* AB, INT ldab,
            const c64* B, INT ldb,
            const c64* X, INT ldx,
            const c64* XACT, INT ldxact,
            const f32* FERR,
            const f32* BERR,
            f32* reslts);

/* Tridiagonal (GT) verification routines */
void cgtt01(const INT n, const c64* DL, const c64* D, const c64* DU,
            const c64* DLF, const c64* DF, const c64* DUF, const c64* DU2,
            const INT* ipiv, c64* work, const INT ldwork, f32* resid);

void cgtt02(const char* trans, const INT n, const INT nrhs,
            const c64* DL, const c64* D, const c64* DU,
            const c64* X, const INT ldx,
            c64* B, const INT ldb,
            f32* resid);

void cgtt05(const char* trans, const INT n, const INT nrhs,
            const c64* DL, const c64* D, const c64* DU,
            const c64* B, const INT ldb,
            const c64* X, const INT ldx,
            const c64* XACT, const INT ldxact,
            const f32* FERR,
            const f32* BERR,
            f32* reslts);

/* Hermitian positive definite tridiagonal (PT) verification routines */
void cptt01(const INT n, const f32* D, const c64* E,
            const f32* DF, const c64* EF,
            c64* work, f32* resid);

void cptt02(const char* uplo, const INT n, const INT nrhs,
            const f32* D, const c64* E,
            const c64* X, const INT ldx,
            c64* B, const INT ldb,
            f32* resid);

void cptt05(const INT n, const INT nrhs,
            const f32* D, const c64* E,
            const c64* B, const INT ldb,
            const c64* X, const INT ldx,
            const c64* XACT, const INT ldxact,
            const f32* FERR, const f32* BERR,
            f32* reslts);

/* Hermitian positive semidefinite pivoted Cholesky (PS) verification routines */
void cpst01(const char* uplo, const INT n,
            const c64* const restrict A, const INT lda,
            c64* const restrict AFAC, const INT ldafac,
            c64* const restrict PERM, const INT ldperm,
            const INT* const restrict piv,
            f32* const restrict rwork, f32* resid, const INT rank);

/* Full triangular (TR) verification routines */
void ctrt01(const char* uplo, const char* diag, const INT n,
            const c64* A, const INT lda,
            c64* AINV, const INT ldainv,
            f32* rcond, f32* rwork, f32* resid);

void ctrt02(const char* uplo, const char* trans, const char* diag,
            const INT n, const INT nrhs,
            const c64* A, const INT lda,
            const c64* X, const INT ldx,
            const c64* B, const INT ldb,
            c64* work, f32* rwork, f32* resid);

void ctrt03(const char* uplo, const char* trans, const char* diag,
            const INT n, const INT nrhs, const c64* A, const INT lda,
            const f32 scale, const f32* cnorm, const f32 tscal,
            const c64* X, const INT ldx, const c64* B, const INT ldb,
            c64* work, f32* resid);

void ctrt05(const char* uplo, const char* trans, const char* diag,
            const INT n, const INT nrhs, const c64* A, const INT lda,
            const c64* B, const INT ldb, const c64* X, const INT ldx,
            const c64* XACT, const INT ldxact,
            const f32* ferr, const f32* berr, f32* reslts);

void ctrt06(const f32 rcond, const f32 rcondc,
            const char* uplo, const char* diag, const INT n,
            const c64* A, const INT lda, f32* rwork, f32* rat);

/* Banded triangular (TB) verification routines */
void ctbt02(const char* uplo, const char* trans, const char* diag,
            const INT n, const INT kd, const INT nrhs,
            const c64* AB, const INT ldab,
            const c64* X, const INT ldx,
            const c64* B, const INT ldb,
            c64* work, f32* rwork, f32* resid);

void ctbt03(const char* uplo, const char* trans, const char* diag,
            const INT n, const INT kd, const INT nrhs,
            const c64* AB, const INT ldab,
            const f32 scale, const f32* cnorm, const f32 tscal,
            const c64* X, const INT ldx,
            const c64* B, const INT ldb,
            c64* work, f32* resid);

void ctbt05(const char* uplo, const char* trans, const char* diag,
            const INT n, const INT kd, const INT nrhs,
            const c64* AB, const INT ldab,
            const c64* B, const INT ldb,
            const c64* X, const INT ldx,
            const c64* XACT, const INT ldxact,
            const f32* ferr, const f32* berr,
            f32* reslts);

void ctbt06(const f32 rcond, const f32 rcondc,
            const char* uplo, const char* diag, const INT n, const INT kd,
            const c64* AB, const INT ldab, f32* rwork, f32* rat);

/* Packed triangular (TP) verification routines */
void ctpt01(const char* uplo, const char* diag, const INT n,
            const c64* AP, c64* AINVP,
            f32* rcond, f32* rwork, f32* resid);

void ctpt02(const char* uplo, const char* trans, const char* diag,
            const INT n, const INT nrhs,
            const c64* AP, const c64* X, const INT ldx,
            const c64* B, const INT ldb,
            c64* work, f32* rwork, f32* resid);

void ctpt03(const char* uplo, const char* trans, const char* diag,
            const INT n, const INT nrhs,
            const c64* AP, const f32 scale, const f32* cnorm,
            const f32 tscal, const c64* X, const INT ldx,
            const c64* B, const INT ldb,
            c64* work, f32* resid);

void ctpt05(const char* uplo, const char* trans, const char* diag,
            const INT n, const INT nrhs,
            const c64* AP, const c64* B, const INT ldb,
            const c64* X, const INT ldx,
            const c64* XACT, const INT ldxact,
            const f32* ferr, const f32* berr,
            f32* reslts);

void ctpt06(const f32 rcond, const f32 rcondc,
            const char* uplo, const char* diag, const INT n,
            const c64* AP, f32* rwork, f32* rat);

/* Hermitian indefinite (HE) verification routines */
void chet01(const char* uplo, const INT n,
            const c64* const restrict A, const INT lda,
            const c64* const restrict AFAC, const INT ldafac,
            const INT* const restrict ipiv,
            c64* const restrict C, const INT ldc,
            f32* const restrict rwork, f32* resid);

void chet01_rook(const char* uplo, const INT n,
                 const c64* const restrict A, const INT lda,
                 const c64* const restrict AFAC, const INT ldafac,
                 const INT* const restrict ipiv,
                 c64* const restrict C, const INT ldc,
                 f32* const restrict rwork, f32* resid);

void chet01_aa(const char* uplo, const INT n,
               const c64* const restrict A, const INT lda,
               const c64* const restrict AFAC, const INT ldafac,
               const INT* const restrict ipiv,
               c64* const restrict C, const INT ldc,
               f32* const restrict rwork, f32* resid);

void chet01_3(const char* uplo, const INT n,
              const c64* const restrict A, const INT lda,
              c64* const restrict AFAC, const INT ldafac,
              c64* const restrict E,
              const INT* const restrict ipiv,
              c64* const restrict C, const INT ldc,
              f32* const restrict rwork, f32* resid);

/* Hermitian indefinite multiply helpers */
void clavhe(const char* uplo, const char* trans, const char* diag,
            const INT n, const INT nrhs,
            const c64* A, const INT lda,
            const INT* ipiv,
            c64* B, const INT ldb,
            INT* info);

void clavhe_rook(const char* uplo, const char* trans, const char* diag,
                 const INT n, const INT nrhs,
                 const c64* A, const INT lda,
                 const INT* ipiv,
                 c64* B, const INT ldb,
                 INT* info);

/* Symmetric indefinite multiply helpers */
void clavsy(const char* uplo, const char* trans, const char* diag,
            const INT n, const INT nrhs,
            const c64* A, const INT lda,
            const INT* ipiv,
            c64* B, const INT ldb,
            INT* info);

void clavsy_rook(const char* uplo, const char* trans, const char* diag,
                 const INT n, const INT nrhs,
                 const c64* A, const INT lda,
                 const INT* ipiv,
                 c64* B, const INT ldb,
                 INT* info);

/* Symmetric packed multiply helper */
void clavsp(const char* uplo, const char* trans, const char* diag,
            const INT n, const INT nrhs,
            const c64* A,
            const INT* ipiv,
            c64* B, const INT ldb,
            INT* info);

/* Symmetric indefinite (SY) verification routines */
void csyt01(const char* uplo, const INT n,
            const c64* const restrict A, const INT lda,
            const c64* const restrict AFAC, const INT ldafac,
            const INT* const restrict ipiv,
            c64* const restrict C, const INT ldc,
            f32* const restrict rwork, f32* resid);

void csyt01_rook(const char* uplo, const INT n,
                 const c64* const restrict A, const INT lda,
                 const c64* const restrict AFAC, const INT ldafac,
                 const INT* const restrict ipiv,
                 c64* const restrict C, const INT ldc,
                 f32* const restrict rwork, f32* resid);

void csyt01_3(const char* uplo, const INT n,
              const c64* const restrict A, const INT lda,
              c64* const restrict AFAC, const INT ldafac,
              c64* const restrict E,
              INT* const restrict ipiv,
              c64* const restrict C, const INT ldc,
              f32* const restrict rwork, f32* resid);

void csyt01_aa(const char* uplo, const INT n,
               const c64* const restrict A, const INT lda,
               const c64* const restrict AFAC, const INT ldafac,
               const INT* const restrict ipiv,
               c64* const restrict C, const INT ldc,
               f32* const restrict rwork, f32* resid);

/* Symmetric packed factorization verification */
void cspt01(const char* uplo, const INT n, const c64* A,
            const c64* AFAC, const INT* ipiv, c64* C, const INT ldc,
            f32* rwork, f32* resid);

/* Symmetric packed solve residual */
void cspt02(const char* uplo, const INT n, const INT nrhs,
            const c64* A, const c64* X, const INT ldx,
            c64* B, const INT ldb, f32* rwork, f32* resid);

/* Symmetric packed inverse verification */
void cspt03(const char* uplo, const INT n, const c64* A, const c64* AINV,
            c64* work, const INT ldw, f32* rwork, f32* rcond, f32* resid);

/* Symmetric full-storage solve residual */
void csyt02(const char* uplo, const INT n, const INT nrhs,
            const c64* A, const INT lda, const c64* X, const INT ldx,
            c64* B, const INT ldb, f32* rwork, f32* resid);

/* Symmetric full-storage inverse verification */
void csyt03(const char* uplo, const INT n, const c64* A, const INT lda,
            c64* AINV, const INT ldainv, c64* work, const INT ldwork,
            f32* rwork, f32* rcond, f32* resid);

/* Hermitian packed multiply helper */
void clavhp(const char* uplo, const char* trans, const char* diag,
            const INT n, const INT nrhs,
            const c64* A,
            const INT* ipiv,
            c64* B, const INT ldb,
            INT* info);

/* Hermitian packed factorization verification */
void chpt01(const char* uplo, const INT n, const c64* A,
            const c64* AFAC, const INT* ipiv, c64* C, const INT ldc,
            f32* rwork, f32* resid);

/* Hermitian positive definite (PO) verification routines */
void cpot01(const char* uplo, const INT n,
            const c64* const restrict A, const INT lda,
            c64* const restrict AFAC, const INT ldafac,
            f32* const restrict rwork, f32* resid);

void cpot02(const char* uplo, const INT n, const INT nrhs,
            const c64* const restrict A, const INT lda,
            const c64* const restrict X, const INT ldx,
            c64* const restrict B, const INT ldb,
            f32* const restrict rwork, f32* resid);

void cpot03(const char* uplo, const INT n,
            const c64* const restrict A, const INT lda,
            c64* const restrict AINV, const INT ldainv,
            c64* const restrict work, const INT ldwork,
            f32* const restrict rwork, f32* rcond, f32* resid);

void cpot05(const char* uplo, const INT n, const INT nrhs,
            const c64* const restrict A, const INT lda,
            const c64* const restrict B, const INT ldb,
            const c64* const restrict X, const INT ldx,
            const c64* const restrict XACT, const INT ldxact,
            const f32* const restrict ferr, const f32* const restrict berr,
            f32* const restrict reslts);

void cpot06(const char* uplo, const INT n, const INT nrhs,
            const c64* A, const INT lda, const c64* X, const INT ldx,
            c64* B, const INT ldb, f32* rwork, f32* resid);

/* Hermitian positive definite packed (PP) verification routines */
void cppt01(const char* uplo, const INT n,
            const c64* const restrict A,
            c64* const restrict AFAC,
            f32* const restrict rwork,
            f32* resid);

void cppt02(const char* uplo, const INT n, const INT nrhs,
            const c64* const restrict A,
            const c64* const restrict X, const INT ldx,
            c64* const restrict B, const INT ldb,
            f32* const restrict rwork,
            f32* resid);

void cppt03(const char* uplo, const INT n,
            const c64* const restrict A,
            const c64* const restrict AINV,
            c64* const restrict work, const INT ldwork,
            f32* const restrict rwork,
            f32* rcond, f32* resid);

/* Hermitian positive definite band (PB) verification routines */
void cpbt01(const char* uplo, const INT n, const INT kd,
            const c64* A, const INT lda,
            c64* AFAC, const INT ldafac,
            f32* rwork, f32* resid);

void cpbt02(const char* uplo, const INT n, const INT kd, const INT nrhs,
            const c64* A, const INT lda,
            const c64* X, const INT ldx,
            c64* B, const INT ldb,
            f32* rwork, f32* resid);

void cpbt05(const char* uplo, const INT n, const INT kd, const INT nrhs,
            const c64* AB, const INT ldab,
            const c64* B, const INT ldb,
            const c64* X, const INT ldx,
            const c64* XACT, const INT ldxact,
            const f32* ferr, const f32* berr,
            f32* reslts);

/* Hermitian positive definite packed (PP) verification routines */
void cppt05(const char* uplo, const INT n, const INT nrhs,
            const c64* const restrict AP,
            const c64* const restrict B, const INT ldb,
            const c64* const restrict X, const INT ldx,
            const c64* const restrict XACT, const INT ldxact,
            const f32* const restrict FERR,
            const f32* const restrict BERR,
            f32* const restrict reslts);

/* ================================================================
 * Matrix generation routines (TESTING/MATGEN c-prefix)
 * ================================================================ */

/* RNG: complex random number and vector generation */
c64 clarnd_rng(const INT idist, uint64_t state[static 4]);
void clarnv_rng(const INT idist, const INT n, c64* x, uint64_t state[static 4]);

/* Diagonal distribution */
void clatm1(const INT mode, const f32 cond, const INT irsign, const INT idist,
            c64* d, const INT n, INT* info, uint64_t state[static 4]);

/* Givens rotation application */
void clarot(const INT lrows, const INT lleft, const INT lright,
            const INT nl, const c64 c, const c64 s,
            c64* A, const INT lda, c64* xleft, c64* xright);

/* Hilbert matrix generation */
void clahilb(const INT n, const INT nrhs, c64* A, const INT lda,
             c64* X, const INT ldx, c64* B, const INT ldb,
             f32* work, INT* info, const char* path);

/* Kronecker product for generalized Sylvester */
void clakf2(const INT m, const INT n,
            const c64* A, const INT lda, const c64* B,
            const c64* D, const c64* E,
            c64* Z, const INT ldz);

/* Generalized Sylvester test matrices */
void clatm5(const INT prtype, const INT m, const INT n,
            c64* A, const INT lda,
            c64* B, const INT ldb,
            c64* C, const INT ldc,
            c64* D, const INT ldd,
            c64* E, const INT lde,
            c64* F, const INT ldf,
            c64* R, const INT ldr,
            c64* L, const INT ldl,
            const f32 alpha, INT qblcka, INT qblckb);

/* Generalized eigenvalue test matrices */
void clatm6(const INT type, const INT n,
            c64* A, const INT lda, c64* B,
            c64* X, const INT ldx, c64* Y, const INT ldy,
            const c64 alpha, const c64 beta,
            const c64 wx, const c64 wy,
            f32* S, f32* DIF);

/* Random matrix entry (column-wise RNG order) */
c64 clatm2(const INT m, const INT n, const INT i, const INT j,
            const INT kl, const INT ku, const INT idist,
            const c64* d, const INT igrade,
            const c64* dl, const c64* dr,
            const INT ipvtng, const INT* iwork, const f32 sparse,
            uint64_t state[static 4]);

/* Random matrix entry with pivot output (pivot-first RNG order) */
c64 clatm3(const INT m, const INT n, INT i, INT j,
            INT* isub, INT* jsub,
            const INT kl, const INT ku, const INT idist,
            const c64* d, const INT igrade,
            const c64* dl, const c64* dr,
            const INT ipvtng, const INT* iwork, const f32 sparse,
            uint64_t state[static 4]);

/* Complex general matrix with bandwidth reduction */
void clagge(const INT m, const INT n, const INT kl, const INT ku,
            const f32* d, c64* A, const INT lda,
            c64* work, INT* info, uint64_t state[static 4]);

/* Hermitian matrix generation */
void claghe(const INT n, const INT k, const f32* d,
            c64* A, const INT lda,
            c64* work, INT* info, uint64_t state[static 4]);

/* Complex symmetric matrix generation */
void clagsy(const INT n, const INT k, const f32* d,
            c64* A, const INT lda,
            c64* work, INT* info, uint64_t state[static 4]);

/* Random unitary similarity transformation */
void clarge(const INT n, c64* A, const INT lda,
            c64* work, INT* info, uint64_t state[static 4]);

/* Random unitary matrix application */
void claror(const char* side, const char* init,
            const INT m, const INT n,
            c64* A, const INT lda,
            c64* X, INT* info, uint64_t state[static 4]);

/* Matrix generation driver */
void clatmr(const INT m, const INT n, const char* dist, const char* sym,
            c64* d, const INT mode, const f32 cond, const c64 dmax,
            const char* rsign, const char* grade,
            c64* dl, const INT model, const f32 condl,
            c64* dr, const INT moder, const f32 condr,
            const char* pivtng, const INT* ipivot, const INT kl, const INT ku,
            const f32 sparse, const f32 anorm, const char* pack,
            c64* A, const INT lda, INT* iwork, INT* info,
            uint64_t state[static 4]);

/* Matrix generator with specified singular values / eigenvalues */
void clatms(const INT m, const INT n, const char* dist, const char* sym,
            f32* d, const INT mode, const f32 cond, const f32 dmax_,
            const INT kl, const INT ku, const char* pack,
            c64* A, const INT lda, c64* work, INT* info,
            uint64_t state[static 4]);

/* Test matrix generator with rank control */
void clatmt(const INT m, const INT n, const char* dist, const char* sym,
            f32* d, const INT mode, const f32 cond, const f32 dmax_,
            const INT rank,
            const INT kl, const INT ku, const char* pack,
            c64* A, const INT lda, c64* work, INT* info,
            uint64_t state[static 4]);

/* Non-symmetric matrix with specified eigenvalues */
void clatme(const INT n, const char* dist, c64* D,
            const INT mode, const f32 cond, const c64 dmax,
            const char* rsign, const char* upper,
            const char* sim, f32* DS, const INT modes, const f32 conds,
            const INT kl, const INT ku, const f32 anorm,
            c64* A, const INT lda, c64* work, INT* info,
            uint64_t state[static 4]);

/* Special symmetric test matrix generators */
void clatsp(const char* uplo, const INT n, c64* X,
            uint64_t state[static 4]);

void clatsy(const char* uplo, const INT n, c64* X, const INT ldx,
            uint64_t state[static 4]);

/* Non-negative real diagonal check */
INT cgennd(const INT m, const INT n, const c64* const restrict A, const INT lda);

/* Set diagonal imaginary parts to BIGNUM (Hermitian test helper) */
void claipd(const INT n, c64* A, const INT inda, const INT vinda);

/* Hermitian tridiagonal matrix-vector multiply */
void claptm(const char* uplo, const INT n, const INT nrhs,
            const f32 alpha, const f32* D, const c64* E,
            const c64* X, const INT ldx,
            const f32 beta, c64* B, const INT ldb);

/* Matrix parameter setup */
void clatb4(const char* path, const INT imat, const INT m, const INT n,
            char* type, INT* kl, INT* ku, f32* anorm,
            INT* mode, f32* cndnum, char* dist);

void clatb5(const char* path, const INT imat, const INT n,
            char* type, INT* kl, INT* ku, f32* anorm,
            INT* mode, f32* cndnum, char* dist);

/* Right-hand side generation */
void clarhs(const char* path, const char* xtype, const char* uplo,
            const char* trans, const INT m, const INT n, const INT kl,
            const INT ku, const INT nrhs, const c64* A, const INT lda,
            c64* X, const INT ldx, c64* B, const INT ldb,
            INT* info, uint64_t state[static 4]);

/* Triangular test matrix generators */
void clattr(const INT imat, const char* uplo, const char* trans, char* diag,
            const INT n, c64* A, const INT lda,
            c64* B, c64* work, f32* rwork, INT* info,
            uint64_t state[static 4]);

void clattb(const INT imat, const char* uplo, const char* trans, char* diag,
            const INT n, const INT kd, c64* AB, const INT ldab,
            c64* B, c64* work, f32* rwork, INT* info,
            uint64_t state[static 4]);

void clattp(const INT imat, const char* uplo, const char* trans, char* diag,
            const INT n, c64* AP, c64* B, c64* work, f32* rwork,
            INT* info, uint64_t state[static 4]);

/* QR verification routines */
void cqrt01(const INT m, const INT n,
            const c64* const restrict A,
            c64* const restrict AF,
            c64* const restrict Q,
            c64* const restrict R,
            const INT lda,
            c64* const restrict tau,
            c64* const restrict work, const INT lwork,
            f32* const restrict rwork,
            f32* restrict result);

void cqrt01p(const INT m, const INT n,
             const c64* const restrict A,
             c64* const restrict AF,
             c64* const restrict Q,
             c64* const restrict R,
             const INT lda,
             c64* const restrict tau,
             c64* const restrict work, const INT lwork,
             f32* const restrict rwork,
             f32* restrict result);

void cqrt02(const INT m, const INT n, const INT k,
            const c64* const restrict A,
            const c64* const restrict AF,
            c64* const restrict Q,
            c64* const restrict R,
            const INT lda,
            const c64* const restrict tau,
            c64* const restrict work, const INT lwork,
            f32* const restrict rwork,
            f32* restrict result);

void cqrt03(const INT m, const INT n, const INT k,
            const c64* const restrict AF,
            c64* const restrict C,
            c64* const restrict CC,
            c64* const restrict Q,
            const INT lda,
            const c64* const restrict tau,
            c64* const restrict work, const INT lwork,
            f32* const restrict rwork,
            f32* restrict result);

void cqrt04(const INT m, const INT n, const INT nb, f32* restrict result);

void cqrt05(const INT m, const INT n, const INT l, const INT nb, f32* restrict result);

f32 cqrt11(const INT m, const INT k, const c64* A, const INT lda,
           const c64* tau, c64* work, const INT lwork);

f32 cqrt12(const INT m, const INT n, const c64* A, const INT lda,
           const f32* S, c64* work, const INT lwork,
           f32* rwork);

void cqrt13(const INT scale, const INT m, const INT n,
            c64* A, const INT lda, f32* norma,
            uint64_t state[static 4]);

f32 cqrt14(const char* trans, const INT m, const INT n, const INT nrhs,
           const c64* A, const INT lda, const c64* X, const INT ldx,
           c64* work, const INT lwork);

void cqrt15(const INT scale, const INT rksel,
            const INT m, const INT n, const INT nrhs,
            c64* A, const INT lda, c64* B, const INT ldb,
            f32* S, INT* rank, f32* norma, f32* normb,
            c64* work, const INT lwork,
            uint64_t state[static 4]);

void cqrt16(const char* trans, const INT m, const INT n, const INT nrhs,
            const c64* A, const INT lda,
            const c64* X, const INT ldx,
            c64* B, const INT ldb,
            f32* rwork, f32* resid);

f32 cqrt17(const char* trans, const INT iresid,
           const INT m, const INT n, const INT nrhs,
           const c64* A, const INT lda,
           const c64* X, const INT ldx,
           const c64* B, const INT ldb,
           c64* C,
           c64* work, const INT lwork);

f32 cqpt01(const INT m, const INT n, const INT k,
           const c64* A, const c64* AF, const INT lda,
           const c64* tau, const INT* jpvt,
           c64* work, const INT lwork);

/* QL verification routines */
void cqlt01(const INT m, const INT n,
            const c64* const restrict A,
            c64* const restrict AF,
            c64* const restrict Q,
            c64* const restrict L,
            const INT lda,
            c64* const restrict tau,
            c64* const restrict work, const INT lwork,
            f32* const restrict rwork,
            f32* restrict result);

void cqlt02(const INT m, const INT n, const INT k,
            const c64* const restrict A,
            const c64* const restrict AF,
            c64* const restrict Q,
            c64* const restrict L,
            const INT lda,
            const c64* const restrict tau,
            c64* const restrict work, const INT lwork,
            f32* const restrict rwork,
            f32* restrict result);

void cqlt03(const INT m, const INT n, const INT k,
            const c64* const restrict AF,
            c64* const restrict C,
            c64* const restrict CC,
            c64* const restrict Q,
            const INT lda,
            const c64* const restrict tau,
            c64* const restrict work, const INT lwork,
            f32* const restrict rwork,
            f32* restrict result);

/* LQ verification routines */
void clqt01(const INT m, const INT n,
            const c64* const restrict A,
            c64* const restrict AF,
            c64* const restrict Q,
            c64* const restrict L,
            const INT lda,
            c64* const restrict tau,
            c64* const restrict work, const INT lwork,
            f32* const restrict rwork,
            f32* restrict result);

void clqt02(const INT m, const INT n, const INT k,
            const c64* const restrict A,
            const c64* const restrict AF,
            c64* const restrict Q,
            c64* const restrict L,
            const INT lda,
            const c64* const restrict tau,
            c64* const restrict work, const INT lwork,
            f32* const restrict rwork,
            f32* restrict result);

void clqt03(const INT m, const INT n, const INT k,
            c64* const restrict AF,
            c64* const restrict C,
            c64* const restrict CC,
            c64* const restrict Q,
            const INT lda,
            const c64* const restrict tau,
            c64* const restrict work, const INT lwork,
            f32* const restrict rwork,
            f32* restrict result);

void clqt04(const INT m, const INT n, const INT nb, f32* restrict result);

void clqt05(const INT m, const INT n, const INT l, const INT nb,
            f32* restrict result);

/* RQ verification routines */
void crqt01(const INT m, const INT n,
            const c64* const restrict A,
            c64* const restrict AF,
            c64* const restrict Q,
            c64* const restrict R,
            const INT lda,
            c64* const restrict tau,
            c64* const restrict work, const INT lwork,
            f32* const restrict rwork,
            f32* restrict result);

void crqt02(const INT m, const INT n, const INT k,
            const c64* const restrict A,
            const c64* const restrict AF,
            c64* const restrict Q,
            c64* const restrict R,
            const INT lda,
            const c64* const restrict tau,
            c64* const restrict work, const INT lwork,
            f32* const restrict rwork,
            f32* restrict result);

void crqt03(const INT m, const INT n, const INT k,
            c64* const restrict AF,
            c64* const restrict C,
            c64* const restrict CC,
            c64* const restrict Q,
            const INT lda,
            const c64* const restrict tau,
            c64* const restrict work, const INT lwork,
            f32* const restrict rwork,
            f32* restrict result);

/* RZ verification routines */
f32 crzt01(const INT m, const INT n, const c64* A, c64* AF,
           const INT lda, const c64* tau, c64* work, const INT lwork);

f32 crzt02(const INT m, const INT n, c64* AF, const INT lda,
           const c64* tau, c64* work, const INT lwork);

/* TSQR verification */
void ctsqr01(const char* tssw, const INT m, const INT n, const INT mb,
             const INT nb, f32* result);

/* Unitary Householder reconstruction verification */
void cunhr_col01(const INT m, const INT n, const INT mb1, const INT nb1,
                 const INT nb2, f32* restrict result);

void cunhr_col02(const INT m, const INT n, const INT mb1, const INT nb1,
                 const INT nb2, f32* restrict result);

/* Real-valued sorting (d-prefix, used by zchkq3/zchkqp3rk for singular values) */
void slaord(const char* job, const INT n, f32* X, const INT incx);

/* QL solve */
void cgeqls(const INT m, const INT n, const INT nrhs,
            c64* A, const INT lda, const c64* tau,
            c64* B, const INT ldb, c64* work, const INT lwork, INT* info);

/* RQ solve */
void cgerqs(const INT m, const INT n, const INT nrhs,
            c64* A, const INT lda, const c64* tau,
            c64* B, const INT ldb, c64* work, const INT lwork, INT* info);

/* GLM/GQR/GRQ/GSV/LSE test helpers (TESTING/EIG) */
void clsets(const INT m, const INT p, const INT n,
            const c64* A, c64* AF, const INT lda,
            const c64* B, c64* BF, const INT ldb,
            const c64* C, c64* CF,
            const c64* D, c64* DF,
            c64* X,
            c64* work, const INT lwork,
            f32* rwork, f32* result);

/* EIG matrix type parameters (real-only, shared from d-prefix) */
void slatb9(const char* path, const INT imat, const INT m, const INT p, const INT n,
            char* type, INT* kla, INT* kua, INT* klb, INT* kub,
            f32* anorm, f32* bnorm, INT* modea, INT* modeb,
            f32* cndnma, f32* cndnmb, char* dista, char* distb);

void cglmts(const INT n, const INT m, const INT p,
            const c64* A, c64* AF, const INT lda,
            const c64* B, c64* BF, const INT ldb,
            const c64* D, c64* DF,
            c64* X, c64* U,
            c64* work, const INT lwork,
            f32* rwork, f32* result);

void cgqrts(const INT n, const INT m, const INT p,
            const c64* A, c64* AF, c64* Q, c64* R, const INT lda,
            c64* taua,
            const c64* B, c64* BF, c64* Z, c64* T, c64* BWK, const INT ldb,
            c64* taub,
            c64* work, const INT lwork,
            f32* rwork, f32* result);

void cgrqts(const INT m, const INT p, const INT n,
            const c64* A, c64* AF, c64* Q, c64* R, const INT lda,
            c64* taua,
            const c64* B, c64* BF, c64* Z, c64* T, c64* BWK, const INT ldb,
            c64* taub,
            c64* work, const INT lwork,
            f32* rwork, f32* result);

void cgsvts3(const INT m, const INT p, const INT n,
             const c64* A, c64* AF, const INT lda,
             const c64* B, c64* BF, const INT ldb,
             c64* U, const INT ldu,
             c64* V, const INT ldv,
             c64* Q, const INT ldq,
             f32* alpha, f32* beta,
             c64* R, const INT ldr,
             INT* iwork,
             c64* work, const INT lwork,
             f32* rwork, f32* result);

/* Eigenvalue test matrix generator */
void clatm4(const INT itype, const INT n, const INT nz1, const INT nz2,
            const INT rsign, const f32 amagn, const f32 rcond,
            const f32 triang, const INT idist,
            c64* A, const INT lda,
            uint64_t state[static 4]);

/* Complex symmetric band matrix-vector product */
void csbmv(const char* uplo, const INT n, const INT k,
           const c64 alpha, const c64* A, const INT lda,
           const c64* X, const INT incx,
           const c64 beta, c64* Y, const INT incy);

/* Real diagonal generator (shared from d-prefix, clatms calls SLATM1 not CLATM1) */
void slatm1(const INT mode, const f32 cond, const INT irsign, const INT idist,
            f32* d, const INT n, INT* info, uint64_t state[static 4]);

/* Real diagonal generator with rank (shared from d-prefix, clatmt calls SLATM7) */
void slatm7(const INT mode, const f32 cond, const INT irsign, const INT idist,
            f32* d, const INT n, const INT rank, INT* info,
            uint64_t state[static 4]);

/* Hermitian band eigenvalue verification */
void chbt21(const char* uplo, const INT n, const INT ka, const INT ks,
            const c64* A, const INT lda,
            const f32* D, const f32* E,
            const c64* U, const INT ldu,
            c64* work, f32* rwork, f32* result);

/* Hermitian tridiagonal eigenvalue decomposition verification */
void cstt21(const INT n, const INT kband,
            const f32* const restrict AD, const f32* const restrict AE,
            const f32* const restrict SD, const f32* const restrict SE,
            const c64* const restrict U, const INT ldu,
            c64* const restrict work, f32* const restrict rwork,
            f32* restrict result);

/* Hermitian tridiagonal partial eigenvalue decomposition verification */
void cstt22(const INT n, const INT m, const INT kband,
            const f32* const restrict AD, const f32* const restrict AE,
            const f32* const restrict SD, const f32* const restrict SE,
            const c64* const restrict U, const INT ldu,
            c64* const restrict work, const INT ldwork,
            f32* const restrict rwork,
            f32* restrict result);

/* Hermitian partial eigendecomposition verification */
void chet22(const INT itype, const char* uplo, const INT n, const INT m,
            const INT kband, const c64* const restrict A, const INT lda,
            const f32* const restrict D, const f32* const restrict E,
            const c64* const restrict U, const INT ldu,
            const c64* const restrict V, const INT ldv,
            const c64* const restrict tau,
            c64* const restrict work, f32* const restrict rwork,
            f32* restrict result);

/* SVD verification routines */
void cbdt01(const INT m, const INT n, const INT kd,
            const c64* const restrict A, const INT lda,
            const c64* const restrict Q, const INT ldq,
            const f32* const restrict D, const f32* const restrict E,
            const c64* const restrict PT, const INT ldpt,
            c64* const restrict work, f32* const restrict rwork,
            f32* resid);

void cbdt02(const INT m, const INT n,
            const c64* const restrict B, const INT ldb,
            const c64* const restrict C, const INT ldc,
            const c64* const restrict U, const INT ldu,
            c64* const restrict work, f32* const restrict rwork,
            f32* resid);

void cbdt03(const char* uplo, const INT n, const INT kd,
            const f32* const restrict D, const f32* const restrict E,
            const c64* const restrict U, const INT ldu,
            const f32* const restrict S,
            const c64* const restrict VT, const INT ldvt,
            c64* const restrict work, f32* resid);

void cbdt05(const INT m, const INT n, const c64* const restrict A, const INT lda,
            const f32* const restrict S, const INT ns,
            const c64* const restrict U, const INT ldu,
            const c64* const restrict VT, const INT ldvt,
            c64* const restrict work, f32* resid);

/* Generalized eigenvalue verification routines */
void cget52(const INT left, const INT n,
            const c64* A, const INT lda,
            const c64* B, const INT ldb,
            const c64* E, const INT lde,
            const c64* alpha, const c64* beta,
            c64* work, f32* rwork, f32* result);

/* Unitary matrix check */
void cunt01(const char* rowcol, const INT m, const INT n,
            const c64* U, const INT ldu,
            c64* work, const INT lwork, f32* rwork, f32* resid);

/* Hessenberg reduction verification */
void chst01(const INT n, const INT ilo, const INT ihi,
            const c64* A, const INT lda,
            const c64* H, const INT ldh,
            const c64* Q, const INT ldq,
            c64* work, const INT lwork, f32* rwork, f32* result);

/* ================================================================
 * Eigenvalue test verification routines (TESTING/EIG)
 * ================================================================ */

/* Nonsymmetric eigenvalue verification (CGEEV/CGEEVX) */
void cget22(const char* transa, const char* transe, const char* transw,
            const INT n, const c64* A, const INT lda,
            const c64* E, const INT lde, const c64* W,
            c64* work, f32* rwork, f32* result);

void cget23(const INT comp, const INT isrt, const char* balanc,
            const INT jtype, const f32 thresh, const INT n,
            c64* A, const INT lda, c64* H,
            c64* W, c64* W1,
            c64* VL, const INT ldvl, c64* VR, const INT ldvr,
            c64* LRE, const INT ldlre,
            f32* rcondv, f32* rcndv1, const f32* rcdvin,
            f32* rconde, f32* rcnde1, const f32* rcdein,
            f32* scale, f32* scale1, f32* result,
            c64* work, const INT lwork, f32* rwork, INT* info);

void cget24(const INT comp, const INT jtype, const f32 thresh,
            const INT n, c64* A, const INT lda,
            c64* H, c64* HT,
            c64* W, c64* WT, c64* WTMP,
            c64* VS, const INT ldvs, c64* VS1,
            const f32 rcdein, const f32 rcdvin,
            const INT nslct, const INT* islct, const INT isrt,
            f32* result, c64* work, const INT lwork,
            f32* rwork, INT* bwork, INT* info);

/* Schur form reordering test (CTREXC) */
void cget36(f32* rmax, INT* lmax, INT* ninfo, INT* knt);

/* Condition number tests (CTRSNA/CTREVC, CTRSEN) */
void cget37(f32 rmax[3], INT lmax[3], INT ninfo[3], INT* knt);
void cget38(f32 rmax[3], INT lmax[3], INT ninfo[3], INT* knt);

/* Unitary matrix comparison */
void cunt03(const char* rc, const INT mu, const INT mv, const INT n,
            const INT k, const c64* const restrict U, const INT ldu,
            const c64* const restrict V, const INT ldv,
            c64* const restrict work, const INT lwork,
            f32* const restrict rwork, f32* result, INT* info);

/* Sylvester equation test (CTRSYL/CTRSYL3) */
void csyl01(const f32 thresh, INT* nfail, f32* rmax, INT* ninfo, INT* knt);

/* Hermitian eigenvalue decomposition verification */
void chet21(const INT itype, const char* uplo, const INT n, const INT kband,
            const c64* const restrict A, const INT lda,
            const f32* const restrict D, const f32* const restrict E,
            const c64* const restrict U, const INT ldu,
            c64* restrict V, const INT ldv,
            const c64* const restrict tau,
            c64* const restrict work, f32* const restrict rwork,
            f32* restrict result);

/* Hermitian packed eigenvalue decomposition verification */
void chpt21(const INT itype, const char* uplo, const INT n, const INT kband,
            const c64* AP, const f32* D, const f32* E,
            const c64* U, const INT ldu,
            c64* VP, const c64* tau,
            c64* work, f32* rwork, f32* result);

/* CSD (cosine-sine decomposition) verification */
void ccsdts(const INT m, const INT p, const INT q,
            const c64* X, c64* XF, const INT ldx,
            c64* U1, const INT ldu1,
            c64* U2, const INT ldu2,
            c64* V1T, const INT ldv1t,
            c64* V2T, const INT ldv2t,
            f32* theta, INT* iwork,
            c64* work, const INT lwork,
            f32* rwork, f32* result);

/* Generalized Hermitian-definite eigenvalue verification */
void csgt01(const INT itype, const char* uplo, const INT n, const INT m,
            const c64* A, const INT lda,
            const c64* B, const INT ldb,
            c64* Z, const INT ldz,
            const f32* D, c64* work, f32* rwork, f32* result);

/* Error checking utilities */
extern INT    xerbla_ok;
extern INT    xerbla_lerr;
extern char   xerbla_srnamt[33];
void chkxer(const char* srnamt, INT infot, INT* lerr, INT* ok);

#endif /* VERIFY_Z_H */
