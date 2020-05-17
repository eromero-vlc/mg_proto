

#include <cstdio>

#include "MG_config.h"
#include "lattice/cmat_mult.h"
#include "utils/memory.h"
#include "utils/print_utils.h"
#include <complex>
#include <cmath>
#include <cfloat>
#include <iostream>
#include <cassert>

#define RESTRICT __restrict__

namespace MG {

namespace {
	template<const int N, const int M>
	struct CMatMultNaiveTT {
			static void fun(std::complex<float>* RESTRICT y,
					const std::complex<float>* RESTRICT  A, std::complex<float> alpha,
					const std::complex<float>* RESTRICT  x)
			{
				for(IndexType row=0; row < N; ++row) {
					for(IndexType col=0; col < M; ++col) {
						for(IndexType j=0; j < N; ++j) {
							y[row + N*col] += alpha * A[ row + N*j ] * x[ j + col*N ];
						}
					}
				}
			}
	};

	template<const int N, const int M>
	struct CTransMatMultNaiveTT {
		static void fun(std::complex<float>* RESTRICT y,
					const std::complex<float>*  RESTRICT A, std::complex<float> alpha,
					const std::complex<float>* RESTRICT  x)
			{
				for(IndexType row=0; row < N; ++row) {
					for(IndexType col=0; col < M; ++col) {
						for(IndexType j=0; j < N; ++j) {
							y[row + N*col] += alpha * std::conj(A[ j + N*row ]) * x[ j + col*N ];
						}
					}
				}
			}
	};

	template<int N, template<const int,const int> class MatMult>
	void CMatMultNaiveT(int ncols, std::complex<float>*y,
				const std::complex<float>* A, std::complex<float> alpha,
				const std::complex<float>* x) {

		while (ncols > 0) {
			if (ncols >= 256) {
				MatMult<N,256>::fun(y, A, alpha, x);
				ncols -= 256;
				x += 256*N;
				y += 256*N;
			} else if (ncols >= 128) {
				MatMult<N,128>::fun(y, A, alpha, x);
				ncols -= 128;
				x += 128*N;
				y += 128*N;
			} else if (ncols >= 64) {
				MatMult<N,64>::fun(y, A, alpha, x);
				ncols -= 64;
				x += 64*N;
				y += 64*N;
			} else if (ncols >= 32) {
				MatMult<N,32>::fun(y, A, alpha, x);
				ncols -= 32;
				x += 32*N;
				y += 32*N;
			} else if (ncols >= 16) {
				MatMult<N,16>::fun(y, A, alpha, x);
				ncols -= 16;
				x += 16*N;
				y += 16*N;
			} else if (ncols >= 8) {
				MatMult<N,8>::fun(y, A, alpha, x);
				ncols -= 8;
				x += 8*N;
				y += 8*N;
			} else if (ncols >= 4) {
				MatMult<N,4>::fun(y, A, alpha, x);
				ncols -= 4;
				x += 4*N;
			} else if (ncols >= 3) {
				MatMult<N,3>::fun(y, A, alpha, x);
				ncols -= 3;
				x += 3*N;
				y += 3*N;
			} else if (ncols >= 2) {
				MatMult<N,2>::fun(y, A, alpha, x);
				ncols -= 2;
				x += 2*N;
				y += 2*N;
			} else if (ncols >= 1) {
				MatMult<N,1>::fun(y, A, alpha, x);
				ncols -= 1;
				x += 1*N;
				y += 1*N;
				y += 1*N;
			}
		}
	}


#ifndef MGPROTO_USE_CBLAS
	extern "C" void cgemm_(const char *transa, const char *transb,
			LAPACK_BLASINT *m, LAPACK_BLASINT *n,
			LAPACK_BLASINT *k, std::complex<float> *alpha,
			const std::complex<float> *a, LAPACK_BLASINT *lda,
			const std::complex<float> *b, LAPACK_BLASINT *ldb,
			std::complex<float> *beta, std::complex<float> *c,
			LAPACK_BLASINT *ldc);
#else
	#include <cblas.h>
	CBLAS_TRANSPOSE toTrans(const char *trans) {
		const char t = *trans;
		if (t == 'n' || t == 'N') return CblasNoTrans;
		if (t == 't' || t == 'T') return CblasTrans;
		if (t == 'c' || t == 'C') return CblasConjTrans;
	}
#endif

}

void XGEMM(const char *transa, const char *transb, LAPACK_BLASINT m,
		LAPACK_BLASINT n, LAPACK_BLASINT k, std::complex<float> alpha,
		const std::complex<float> *a, LAPACK_BLASINT lda, const std::complex<float> *b,
		LAPACK_BLASINT ldb, std::complex<float> beta, std::complex<float> *c,
		LAPACK_BLASINT ldc) {
	assert(c != a && c != b);
	// if (m == 12 && lda == 12 && ldb == 12 && ldc == 12 && (transb[0] == 'n' || transb[0] == 'N')) {
	// 	if (transa[0] == 'N' || transa[0] == 'n') {
	// 		CMatMultNaiveT<12,CMatMultNaiveTT>(n, c, a, alpha, b);
	// 	} else {
	// 		CMatMultNaiveT<12,CTransMatMultNaiveTT>(n, c, a, alpha, b);
	// 	}
	// 	return;
	// }
#ifndef MGPROTO_USE_CBLAS
	cgemm_(transa, transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
#else
	cblas_cgemm(CblasColMajor, toTrans(transa), toTrans(transb), m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc);
#endif
}

void CMatMultNaive(float* y,
				   const float* A,
				   const float* x,
				   IndexType N,
				   IndexType ncol)
{
	CMatMultCoeffAddNaive(0.0, y, 1.0, A, x, N, ncol);
}

void CMatMultAddNaive(float* y,
		const float* A,
		const float* x,
		IndexType N,
		IndexType ncol)
{
	CMatMultCoeffAddNaive(1.0, y, 1.0, A, x, N, ncol);
}

void CMatMultCoeffAddNaive(float beta,
		std::complex<float>* y,
		IndexType ldy,
		float alpha,
		const std::complex<float>* A,
		IndexType ldA,
		const std::complex<float>* x,
		IndexType ldx,
		IndexType Arows,
		IndexType Acols,
		IndexType xcols)
{
	if (fabs(beta) == 0.0) {
		for (IndexType j=0; j < xcols; ++j) 
			for (IndexType i=0; i < Arows; ++i) y[i + ldy*j] = 0.0;
	}

	XGEMM("N", "N", Arows, xcols, Acols, alpha, A, ldA, x, ldx, beta, y, ldy);
}


void CMatMultCoeffAddNaive(float beta,
		float* y,
		float alpha,
		const float* A,
		const float* x,
		IndexType N,
		IndexType ncol)
{
	if (fabs(beta) == 0.0) for (IndexType i=0; i < N*ncol*2; ++i) y[i] = 0.0;

	// Pretend these are arrays of complex numbers
	std::complex<float>* yc = reinterpret_cast<std::complex<float>*>(y);
	const std::complex<float>* Ac = reinterpret_cast<const std::complex<float>*>(A);
	const std::complex<float>* xc = reinterpret_cast<const std::complex<float>*>(x);

	XGEMM("N", "N", N, ncol, N, alpha, Ac, N, xc, N, beta, yc, N);
}


void CMatAdjMultNaive(float* y,
				   const float* A,
				   const float* x,
				   IndexType N,
					IndexType ncol)
{
	for (IndexType i=0; i < N*ncol*2; ++i) y[i] = 0.0;

	std::complex<float>* yc = reinterpret_cast<std::complex<float>*>(y);
	const std::complex<float>* Ac = reinterpret_cast<const std::complex<float>*>(A);
	const std::complex<float>* xc = reinterpret_cast<const std::complex<float>*>(x);

	XGEMM("C", "N", N, ncol, N, 1.0, Ac, N, xc, N, 0.0, yc, N);
}

void CMatAdjMultCoeffAddNaive(float beta,
		std::complex<float>* y,
		IndexType ldy,
		float alpha,
		const std::complex<float>* A,
		IndexType ldA,
		const std::complex<float>* x,
		IndexType ldx,
		IndexType Arows,
		IndexType Acols,
		IndexType xcols)
{
	if (fabs(beta) == 0.0) {
		for (IndexType j=0; j < xcols; ++j) 
			for (IndexType i=0; i < Acols; ++i) y[i + ldy*j] = 0.0;
	}

	XGEMM("C", "N", Acols, xcols, Arows, alpha, A, ldA, x, ldx, beta, y, ldy);
}


void GcCMatMultGcNaive(float* y,
				   const float* A,
				   const float* x,
				   IndexType N,
					IndexType ncol)
{
	GcCMatMultGcCoeffAddNaive(0.0, y, 1.0, A, x, N, ncol);
}


void GcCMatMultGcCoeffAddNaive(float beta, float* y, float alpha,
				   const float* A,
				   const float* x,
				   IndexType N,
					IndexType ncol)
{
	if (fabs(beta) == 0.0) for (IndexType i=0; i < N*ncol*2; ++i) y[i] = 0.0;

	std::complex<float>* yc = reinterpret_cast<std::complex<float>*>(y);
	const std::complex<float>* Ac = reinterpret_cast<const std::complex<float>*>(A);
	const std::complex<float>* xc = reinterpret_cast<const std::complex<float>*>(x);

	XGEMM("N", "N", N/2, ncol, N/2,  alpha,  Ac,            N,  xc,      N, beta,  yc,      N);
	XGEMM("N", "N", N/2, ncol, N/2, -alpha, &Ac[N*N/2],     N, &xc[N/2], N, 1.0,   yc,      N);
	XGEMM("N", "N", N/2, ncol, N/2, -alpha, &Ac[N/2],       N, xc,       N, beta, &yc[N/2], N);
	XGEMM("N", "N", N/2, ncol, N/2,  alpha, &Ac[N*N/2+N/2], N, &xc[N/2], N, 1.0,  &yc[N/2], N);
}

}
