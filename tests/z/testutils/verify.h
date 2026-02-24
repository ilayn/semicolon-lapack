/**
 * @file verify.h
 * @brief Prototypes for complex double (z-prefix) LAPACK test verification routines.
 *
 * These are ports of the verification routines from LAPACK/TESTING/LIN/
 * and LAPACK/TESTING/EIG/.
 */

#ifndef VERIFY_Z_H
#define VERIFY_Z_H

#include <stdint.h>
#include "semicolon_lapack_complex_double.h"

/* General (ZGE) verification routines */
void zget01(const INT m, const INT n, const c128* const restrict A, const INT lda,
            c128* const restrict AFAC, const INT ldafac, const INT* const restrict ipiv,
            f64* const restrict rwork, f64* resid);

void zget02(const char* trans, const INT m, const INT n, const INT nrhs,
            const c128* const restrict A, const INT lda, const c128* const restrict X,
            const INT ldx, c128* const restrict B, const INT ldb,
            f64* const restrict rwork, f64* resid);

void zget03(const INT n, const c128* const restrict A, const INT lda,
            const c128* const restrict AINV, const INT ldainv, c128* const restrict work,
            const INT ldwork, f64* const restrict rwork, f64* rcond, f64* resid);

void zget04(const INT n, const INT nrhs, const c128* const restrict X, const INT ldx,
            const c128* const restrict XACT, const INT ldxact, const f64 rcond,
            f64* resid);

void zget07(const char* trans, const INT n, const INT nrhs,
            const c128* const restrict A, const INT lda,
            const c128* const restrict B, const INT ldb,
            const c128* const restrict X, const INT ldx,
            const c128* const restrict XACT, const INT ldxact,
            const f64* const restrict ferr, const INT chkferr,
            const f64* const restrict berr, f64* const restrict reslts);

void zget08(const char* trans, const INT m, const INT n, const INT nrhs,
            const c128* A, const INT lda, const c128* X, const INT ldx,
            c128* B, const INT ldb, f64* rwork, f64* resid);

void zget10(const INT m, const INT n,
            const c128* const restrict A, const INT lda,
            const c128* const restrict B, const INT ldb,
            c128* const restrict work, f64* const restrict rwork,
            f64* result);

void zget35(f64* rmax, INT* lmax, INT* ninfo, INT* knt);

void zget51(const INT itype, const INT n,
            const c128* A, const INT lda,
            const c128* B, const INT ldb,
            const c128* U, const INT ldu,
            const c128* V, const INT ldv,
            c128* work, f64* rwork, f64* result);

/* Generalized Schur decomposition verify */
void zget54(const INT n, const c128* A, const INT lda,
            const c128* B, const INT ldb,
            const c128* S, const INT lds,
            const c128* T, const INT ldt,
            const c128* U, const INT ldu,
            const c128* V, const INT ldv,
            c128* work, f64* result);

/* Banded (GB) verification routines */
void zgbt01(INT m, INT n, INT kl, INT ku,
            const c128* A, INT lda,
            const c128* AFAC, INT ldafac,
            const INT* ipiv,
            c128* work,
            f64* resid);

void zgbt02(const char* trans, INT m, INT n, INT kl, INT ku, INT nrhs,
            const c128* A, INT lda,
            const c128* X, INT ldx,
            c128* B, INT ldb,
            f64* rwork,
            f64* resid);

void zgbt05(const char* trans, INT n, INT kl, INT ku, INT nrhs,
            const c128* AB, INT ldab,
            const c128* B, INT ldb,
            const c128* X, INT ldx,
            const c128* XACT, INT ldxact,
            const f64* FERR,
            const f64* BERR,
            f64* reslts);

/* Tridiagonal (GT) verification routines */
void zgtt01(const INT n, const c128* DL, const c128* D, const c128* DU,
            const c128* DLF, const c128* DF, const c128* DUF, const c128* DU2,
            const INT* ipiv, c128* work, const INT ldwork, f64* resid);

void zgtt02(const char* trans, const INT n, const INT nrhs,
            const c128* DL, const c128* D, const c128* DU,
            const c128* X, const INT ldx,
            c128* B, const INT ldb,
            f64* resid);

void zgtt05(const char* trans, const INT n, const INT nrhs,
            const c128* DL, const c128* D, const c128* DU,
            const c128* B, const INT ldb,
            const c128* X, const INT ldx,
            const c128* XACT, const INT ldxact,
            const f64* FERR,
            const f64* BERR,
            f64* reslts);

/* Hermitian indefinite (HE) verification routines */
void zhet01(const char* uplo, const INT n,
            const c128* const restrict A, const INT lda,
            const c128* const restrict AFAC, const INT ldafac,
            const INT* const restrict ipiv,
            c128* const restrict C, const INT ldc,
            f64* const restrict rwork, f64* resid);

void zhet01_rook(const char* uplo, const INT n,
                 const c128* const restrict A, const INT lda,
                 const c128* const restrict AFAC, const INT ldafac,
                 const INT* const restrict ipiv,
                 c128* const restrict C, const INT ldc,
                 f64* const restrict rwork, f64* resid);

void zhet01_aa(const char* uplo, const INT n,
               const c128* const restrict A, const INT lda,
               const c128* const restrict AFAC, const INT ldafac,
               const INT* const restrict ipiv,
               c128* const restrict C, const INT ldc,
               f64* const restrict rwork, f64* resid);

void zhet01_3(const char* uplo, const INT n,
              const c128* const restrict A, const INT lda,
              c128* const restrict AFAC, const INT ldafac,
              c128* const restrict E,
              const INT* const restrict ipiv,
              c128* const restrict C, const INT ldc,
              f64* const restrict rwork, f64* resid);

/* Hermitian indefinite multiply helpers */
void zlavhe(const char* uplo, const char* trans, const char* diag,
            const INT n, const INT nrhs,
            const c128* A, const INT lda,
            const INT* ipiv,
            c128* B, const INT ldb,
            INT* info);

void zlavhe_rook(const char* uplo, const char* trans, const char* diag,
                 const INT n, const INT nrhs,
                 const c128* A, const INT lda,
                 const INT* ipiv,
                 c128* B, const INT ldb,
                 INT* info);

/* Symmetric indefinite multiply helpers */
void zlavsy(const char* uplo, const char* trans, const char* diag,
            const INT n, const INT nrhs,
            const c128* A, const INT lda,
            const INT* ipiv,
            c128* B, const INT ldb,
            INT* info);

void zlavsy_rook(const char* uplo, const char* trans, const char* diag,
                 const INT n, const INT nrhs,
                 const c128* A, const INT lda,
                 const INT* ipiv,
                 c128* B, const INT ldb,
                 INT* info);

/* Symmetric packed multiply helper */
void zlavsp(const char* uplo, const char* trans, const char* diag,
            const INT n, const INT nrhs,
            const c128* A,
            const INT* ipiv,
            c128* B, const INT ldb,
            INT* info);

/* Hermitian packed multiply helper */
void zlavhp(const char* uplo, const char* trans, const char* diag,
            const INT n, const INT nrhs,
            const c128* A,
            const INT* ipiv,
            c128* B, const INT ldb,
            INT* info);

/* ================================================================
 * Matrix generation routines (TESTING/MATGEN z-prefix)
 * ================================================================ */

/* RNG: complex random number and vector generation */
c128 zlarnd_rng(const INT idist, uint64_t state[static 4]);
void zlarnv_rng(const INT idist, const INT n, c128* x, uint64_t state[static 4]);

/* Diagonal distribution */
void zlatm1(const INT mode, const f64 cond, const INT irsign, const INT idist,
            c128* d, const INT n, INT* info, uint64_t state[static 4]);

/* Givens rotation application */
void zlarot(const INT lrows, const INT lleft, const INT lright,
            const INT nl, const c128 c, const c128 s,
            c128* A, const INT lda, c128* xleft, c128* xright);

/* Hilbert matrix generation */
void zlahilb(const INT n, const INT nrhs, c128* A, const INT lda,
             c128* X, const INT ldx, c128* B, const INT ldb,
             f64* work, INT* info, const char* path);

/* Kronecker product for generalized Sylvester */
void zlakf2(const INT m, const INT n,
            const c128* A, const INT lda, const c128* B,
            const c128* D, const c128* E,
            c128* Z, const INT ldz);

/* Generalized Sylvester test matrices */
void zlatm5(const INT prtype, const INT m, const INT n,
            c128* A, const INT lda,
            c128* B, const INT ldb,
            c128* C, const INT ldc,
            c128* D, const INT ldd,
            c128* E, const INT lde,
            c128* F, const INT ldf,
            c128* R, const INT ldr,
            c128* L, const INT ldl,
            const f64 alpha, INT qblcka, INT qblckb);

/* Generalized eigenvalue test matrices */
void zlatm6(const INT type, const INT n,
            c128* A, const INT lda, c128* B,
            c128* X, const INT ldx, c128* Y, const INT ldy,
            const c128 alpha, const c128 beta,
            const c128 wx, const c128 wy,
            f64* S, f64* DIF);

/* Random matrix entry (column-wise RNG order) */
c128 zlatm2(const INT m, const INT n, const INT i, const INT j,
            const INT kl, const INT ku, const INT idist,
            const c128* d, const INT igrade,
            const c128* dl, const c128* dr,
            const INT ipvtng, const INT* iwork, const f64 sparse,
            uint64_t state[static 4]);

/* Random matrix entry with pivot output (pivot-first RNG order) */
c128 zlatm3(const INT m, const INT n, INT i, INT j,
            INT* isub, INT* jsub,
            const INT kl, const INT ku, const INT idist,
            const c128* d, const INT igrade,
            const c128* dl, const c128* dr,
            const INT ipvtng, const INT* iwork, const f64 sparse,
            uint64_t state[static 4]);

/* Complex general matrix with bandwidth reduction */
void zlagge(const INT m, const INT n, const INT kl, const INT ku,
            const f64* d, c128* A, const INT lda,
            c128* work, INT* info, uint64_t state[static 4]);

/* Hermitian matrix generation */
void zlaghe(const INT n, const INT k, const f64* d,
            c128* A, const INT lda,
            c128* work, INT* info, uint64_t state[static 4]);

/* Complex symmetric matrix generation */
void zlagsy(const INT n, const INT k, const f64* d,
            c128* A, const INT lda,
            c128* work, INT* info, uint64_t state[static 4]);

/* Random unitary similarity transformation */
void zlarge(const INT n, c128* A, const INT lda,
            c128* work, INT* info, uint64_t state[static 4]);

/* Random unitary matrix application */
void zlaror(const char* side, const char* init,
            const INT m, const INT n,
            c128* A, const INT lda,
            c128* X, INT* info, uint64_t state[static 4]);

/* Matrix generation driver */
void zlatmr(const INT m, const INT n, const char* dist, const char* sym,
            c128* d, const INT mode, const f64 cond, const c128 dmax,
            const char* rsign, const char* grade,
            c128* dl, const INT model, const f64 condl,
            c128* dr, const INT moder, const f64 condr,
            const char* pivtng, const INT* ipivot, const INT kl, const INT ku,
            const f64 sparse, const f64 anorm, const char* pack,
            c128* A, const INT lda, INT* iwork, INT* info,
            uint64_t state[static 4]);

/* Matrix generator with specified singular values / eigenvalues */
void zlatms(const INT m, const INT n, const char* dist, const char* sym,
            f64* d, const INT mode, const f64 cond, const f64 dmax_,
            const INT kl, const INT ku, const char* pack,
            c128* A, const INT lda, c128* work, INT* info,
            uint64_t state[static 4]);

/* Test matrix generator with rank control */
void zlatmt(const INT m, const INT n, const char* dist, const char* sym,
            f64* d, const INT mode, const f64 cond, const f64 dmax_,
            const INT rank,
            const INT kl, const INT ku, const char* pack,
            c128* A, const INT lda, c128* work, INT* info,
            uint64_t state[static 4]);

/* Non-symmetric matrix with specified eigenvalues */
void zlatme(const INT n, const char* dist, c128* D,
            const INT mode, const f64 cond, const c128 dmax,
            const char* rsign, const char* upper,
            const char* sim, f64* DS, const INT modes, const f64 conds,
            const INT kl, const INT ku, const f64 anorm,
            c128* A, const INT lda, c128* work, INT* info,
            uint64_t state[static 4]);

/* Special symmetric test matrix generators */
void zlatsp(const char* uplo, const INT n, c128* X,
            uint64_t state[static 4]);

void zlatsy(const char* uplo, const INT n, c128* X, const INT ldx,
            uint64_t state[static 4]);

/* Non-negative real diagonal check */
INT zgennd(const INT m, const INT n, const c128* const restrict A, const INT lda);

/* Set diagonal imaginary parts to BIGNUM (Hermitian test helper) */
void zlaipd(const INT n, c128* A, const INT inda, const INT vinda);

/* Hermitian tridiagonal matrix-vector multiply */
void zlaptm(const char* uplo, const INT n, const INT nrhs,
            const f64 alpha, const f64* D, const c128* E,
            const c128* X, const INT ldx,
            const f64 beta, c128* B, const INT ldb);

/* Matrix parameter setup */
void zlatb4(const char* path, const INT imat, const INT m, const INT n,
            char* type, INT* kl, INT* ku, f64* anorm,
            INT* mode, f64* cndnum, char* dist);

void zlatb5(const char* path, const INT imat, const INT n,
            char* type, INT* kl, INT* ku, f64* anorm,
            INT* mode, f64* cndnum, char* dist);

/* Right-hand side generation */
void zlarhs(const char* path, const char* xtype, const char* uplo,
            const char* trans, const INT m, const INT n, const INT kl,
            const INT ku, const INT nrhs, const c128* A, const INT lda,
            c128* X, const INT ldx, c128* B, const INT ldb,
            INT* info, uint64_t state[static 4]);

/* Triangular test matrix generators */
void zlattr(const INT imat, const char* uplo, const char* trans, char* diag,
            const INT n, c128* A, const INT lda,
            c128* B, c128* work, f64* rwork, INT* info,
            uint64_t state[static 4]);

void zlattb(const INT imat, const char* uplo, const char* trans, char* diag,
            const INT n, const INT kd, c128* AB, const INT ldab,
            c128* B, c128* work, f64* rwork, INT* info,
            uint64_t state[static 4]);

void zlattp(const INT imat, const char* uplo, const char* trans, char* diag,
            const INT n, c128* AP, c128* B, c128* work, f64* rwork,
            INT* info, uint64_t state[static 4]);

/* QL solve */
void zgeqls(const INT m, const INT n, const INT nrhs,
            c128* A, const INT lda, const c128* tau,
            c128* B, const INT ldb, c128* work, const INT lwork, INT* info);

/* RQ solve */
void zgerqs(const INT m, const INT n, const INT nrhs,
            c128* A, const INT lda, const c128* tau,
            c128* B, const INT ldb, c128* work, const INT lwork, INT* info);

/* GLM/GQR/GRQ/GSV/LSE test helpers (TESTING/EIG) */
void zlsets(const INT m, const INT p, const INT n,
            const c128* A, c128* AF, const INT lda,
            const c128* B, c128* BF, const INT ldb,
            const c128* C, c128* CF,
            const c128* D, c128* DF,
            c128* X,
            c128* work, const INT lwork,
            f64* rwork, f64* result);

void zglmts(const INT n, const INT m, const INT p,
            const c128* A, c128* AF, const INT lda,
            const c128* B, c128* BF, const INT ldb,
            const c128* D, c128* DF,
            c128* X, c128* U,
            c128* work, const INT lwork,
            f64* rwork, f64* result);

void zgqrts(const INT n, const INT m, const INT p,
            const c128* A, c128* AF, c128* Q, c128* R, const INT lda,
            c128* taua,
            const c128* B, c128* BF, c128* Z, c128* T, c128* BWK, const INT ldb,
            c128* taub,
            c128* work, const INT lwork,
            f64* rwork, f64* result);

void zgrqts(const INT m, const INT p, const INT n,
            const c128* A, c128* AF, c128* Q, c128* R, const INT lda,
            c128* taua,
            const c128* B, c128* BF, c128* Z, c128* T, c128* BWK, const INT ldb,
            c128* taub,
            c128* work, const INT lwork,
            f64* rwork, f64* result);

void zgsvts3(const INT m, const INT p, const INT n,
             const c128* A, c128* AF, const INT lda,
             const c128* B, c128* BF, const INT ldb,
             c128* U, const INT ldu,
             c128* V, const INT ldv,
             c128* Q, const INT ldq,
             f64* alpha, f64* beta,
             c128* R, const INT ldr,
             INT* iwork,
             c128* work, const INT lwork,
             f64* rwork, f64* result);

/* Eigenvalue test matrix generator */
void zlatm4(const INT itype, const INT n, const INT nz1, const INT nz2,
            const INT rsign, const f64 amagn, const f64 rcond,
            const f64 triang, const INT idist,
            c128* A, const INT lda,
            uint64_t state[static 4]);

/* Complex symmetric band matrix-vector product */
void zsbmv(const char* uplo, const INT n, const INT k,
           const c128 alpha, const c128* A, const INT lda,
           const c128* X, const INT incx,
           const c128 beta, c128* Y, const INT incy);

/* Real diagonal generator (shared from d-prefix, zlatms calls DLATM1 not ZLATM1) */
void dlatm1(const INT mode, const f64 cond, const INT irsign, const INT idist,
            f64* d, const INT n, INT* info, uint64_t state[static 4]);

/* Real diagonal generator with rank (shared from d-prefix, zlatmt calls DLATM7) */
void dlatm7(const INT mode, const f64 cond, const INT irsign, const INT idist,
            f64* d, const INT n, const INT rank, INT* info,
            uint64_t state[static 4]);

/* Hermitian band eigenvalue verification */
void zhbt21(const char* uplo, const INT n, const INT ka, const INT ks,
            const c128* A, const INT lda,
            const f64* D, const f64* E,
            const c128* U, const INT ldu,
            c128* work, f64* rwork, f64* result);

/* Hermitian tridiagonal eigenvalue decomposition verification */
void zstt21(const INT n, const INT kband,
            const f64* const restrict AD, const f64* const restrict AE,
            const f64* const restrict SD, const f64* const restrict SE,
            const c128* const restrict U, const INT ldu,
            c128* const restrict work, f64* const restrict rwork,
            f64* restrict result);

/* Hermitian tridiagonal partial eigenvalue decomposition verification */
void zstt22(const INT n, const INT m, const INT kband,
            const f64* const restrict AD, const f64* const restrict AE,
            const f64* const restrict SD, const f64* const restrict SE,
            const c128* const restrict U, const INT ldu,
            c128* const restrict work, const INT ldwork,
            f64* const restrict rwork,
            f64* restrict result);

/* Hermitian partial eigendecomposition verification */
void zhet22(const INT itype, const char* uplo, const INT n, const INT m,
            const INT kband, const c128* const restrict A, const INT lda,
            const f64* const restrict D, const f64* const restrict E,
            const c128* const restrict U, const INT ldu,
            const c128* const restrict V, const INT ldv,
            const c128* const restrict tau,
            c128* const restrict work, f64* const restrict rwork,
            f64* restrict result);

/* SVD verification routines */
void zbdt01(const INT m, const INT n, const INT kd,
            const c128* const restrict A, const INT lda,
            const c128* const restrict Q, const INT ldq,
            const f64* const restrict D, const f64* const restrict E,
            const c128* const restrict PT, const INT ldpt,
            c128* const restrict work, f64* const restrict rwork,
            f64* resid);

void zbdt02(const INT m, const INT n,
            const c128* const restrict B, const INT ldb,
            const c128* const restrict C, const INT ldc,
            const c128* const restrict U, const INT ldu,
            c128* const restrict work, f64* const restrict rwork,
            f64* resid);

void zbdt03(const char* uplo, const INT n, const INT kd,
            const f64* const restrict D, const f64* const restrict E,
            const c128* const restrict U, const INT ldu,
            const f64* const restrict S,
            const c128* const restrict VT, const INT ldvt,
            c128* const restrict work, f64* resid);

void zbdt05(const INT m, const INT n, const c128* const restrict A, const INT lda,
            const f64* const restrict S, const INT ns,
            const c128* const restrict U, const INT ldu,
            const c128* const restrict VT, const INT ldvt,
            c128* const restrict work, f64* resid);

/* Generalized eigenvalue verification routines */
void zget52(const INT left, const INT n,
            const c128* A, const INT lda,
            const c128* B, const INT ldb,
            const c128* E, const INT lde,
            const c128* alpha, const c128* beta,
            c128* work, f64* rwork, f64* result);

/* Unitary matrix check */
void zunt01(const char* rowcol, const INT m, const INT n,
            const c128* U, const INT ldu,
            c128* work, const INT lwork, f64* rwork, f64* resid);

/* Hessenberg reduction verification */
void zhst01(const INT n, const INT ilo, const INT ihi,
            const c128* A, const INT lda,
            const c128* H, const INT ldh,
            const c128* Q, const INT ldq,
            c128* work, const INT lwork, f64* rwork, f64* result);

/* ================================================================
 * Eigenvalue test verification routines (TESTING/EIG)
 * ================================================================ */

/* Nonsymmetric eigenvalue verification (ZGEEV/ZGEEVX) */
void zget22(const char* transa, const char* transe, const char* transw,
            const INT n, const c128* A, const INT lda,
            const c128* E, const INT lde, const c128* W,
            c128* work, f64* rwork, f64* result);

void zget23(const INT comp, const INT isrt, const char* balanc,
            const INT jtype, const f64 thresh, const INT n,
            c128* A, const INT lda, c128* H,
            c128* W, c128* W1,
            c128* VL, const INT ldvl, c128* VR, const INT ldvr,
            c128* LRE, const INT ldlre,
            f64* rcondv, f64* rcndv1, const f64* rcdvin,
            f64* rconde, f64* rcnde1, const f64* rcdein,
            f64* scale, f64* scale1, f64* result,
            c128* work, const INT lwork, f64* rwork, INT* info);

void zget24(const INT comp, const INT jtype, const f64 thresh,
            const INT n, c128* A, const INT lda,
            c128* H, c128* HT,
            c128* W, c128* WT, c128* WTMP,
            c128* VS, const INT ldvs, c128* VS1,
            const f64 rcdein, const f64 rcdvin,
            const INT nslct, const INT* islct, const INT isrt,
            f64* result, c128* work, const INT lwork,
            f64* rwork, INT* bwork, INT* info);

/* Schur form reordering test (ZTREXC) */
void zget36(f64* rmax, INT* lmax, INT* ninfo, INT* knt);

/* Condition number tests (ZTRSNA/ZTREVC, ZTRSEN) */
void zget37(f64 rmax[3], INT lmax[3], INT ninfo[3], INT* knt);
void zget38(f64 rmax[3], INT lmax[3], INT ninfo[3], INT* knt);

/* Unitary matrix comparison */
void zunt03(const char* rc, const INT mu, const INT mv, const INT n,
            const INT k, const c128* const restrict U, const INT ldu,
            const c128* const restrict V, const INT ldv,
            c128* const restrict work, const INT lwork,
            f64* const restrict rwork, f64* result, INT* info);

/* Sylvester equation test (ZTRSYL/ZTRSYL3) */
void zsyl01(const f64 thresh, INT* nfail, f64* rmax, INT* ninfo, INT* knt);

/* Hermitian eigenvalue decomposition verification */
void zhet21(const INT itype, const char* uplo, const INT n, const INT kband,
            const c128* const restrict A, const INT lda,
            const f64* const restrict D, const f64* const restrict E,
            const c128* const restrict U, const INT ldu,
            c128* restrict V, const INT ldv,
            const c128* const restrict tau,
            c128* const restrict work, f64* const restrict rwork,
            f64* restrict result);

/* Hermitian packed eigenvalue decomposition verification */
void zhpt21(const INT itype, const char* uplo, const INT n, const INT kband,
            const c128* AP, const f64* D, const f64* E,
            const c128* U, const INT ldu,
            c128* VP, const c128* tau,
            c128* work, f64* rwork, f64* result);

/* CSD (cosine-sine decomposition) verification */
void zcsdts(const INT m, const INT p, const INT q,
            const c128* X, c128* XF, const INT ldx,
            c128* U1, const INT ldu1,
            c128* U2, const INT ldu2,
            c128* V1T, const INT ldv1t,
            c128* V2T, const INT ldv2t,
            f64* theta, INT* iwork,
            c128* work, const INT lwork,
            f64* rwork, f64* result);

/* Generalized Hermitian-definite eigenvalue verification */
void zsgt01(const INT itype, const char* uplo, const INT n, const INT m,
            const c128* A, const INT lda,
            const c128* B, const INT ldb,
            c128* Z, const INT ldz,
            const f64* D, c128* work, f64* rwork, f64* result);

#endif /* VERIFY_Z_H */
