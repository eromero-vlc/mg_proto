

#include <cstdio>

#include "MG_config.h"
#include "lattice/cmat_mult.h"
#include "utils/memory.h"
#include "utils/print_utils.h"
#include <complex>
#include <cmath>
#include <vector>
#include <cfloat>
#include <iostream>
#include <cassert>

#ifdef MG_USE_XSIMD
#include "xsimd/xsimd.hpp"
#endif

#define RESTRICT __restrict__

namespace MG {

namespace {

	//
	// Trivial for-loop implementation of BLAS
	//

	template<const int N, const int M>
		struct CMatMultNaiveTT {
			static void fun(std::complex<float>* RESTRICT y,
					const std::complex<float>* RESTRICT  A, const int alpha,
					const std::complex<float>* RESTRICT  x[8], IndexType d)
			{
				if (alpha == 1) {
					for(IndexType row=0; row < N; ++row) {
						for(IndexType col=0; col < M; ++col) {
							std::complex<float> yj = 0.0;
							for(IndexType j=0; j < N; ++j)
								for(IndexType mu=0; mu < 8; ++mu)
									yj += A[ row + N*j + N*N*mu ] * x[mu][ j + col*N + d*N ];
							y[row + N*col + d*N] += yj;
						}
					}
				} else {
					for(IndexType row=0; row < N; ++row) {
						for(IndexType col=0; col < M; ++col) {
							std::complex<float> yj = 0.0;
							for(IndexType j=0; j < N; ++j)
								for(IndexType mu=0; mu < 8; ++mu)
									yj += A[ row + N*j + N*N*mu ] * x[mu][ j + col*N + d*N ];
							y[row + N*col + d*N] -= yj;
						}
					}
				}
			}
		};

	template<const int N, const int M>
	struct CTransMatMultNaiveTT {
		static void fun(std::complex<float>* RESTRICT y,
					const std::complex<float>*  RESTRICT A, int alpha,
					const std::complex<float>* RESTRICT  x[8], IndexType d)
			{
				if (alpha == 1) {
					for(IndexType row=0; row < N; ++row) {
						for(IndexType col=0; col < M; ++col) {
							std::complex<float> yj = 0.0;
							for(IndexType j=0; j < N; ++j)
								for(IndexType mu=0; mu < 8; ++mu)
									yj += std::conj(A[ j + N*row + N*N*mu ]) * x[mu][ j + col*N + d*N ];
							y[row + N*col + d*N] += yj;
						}
					}
				} else {
					for(IndexType row=0; row < N; ++row) {
						for(IndexType col=0; col < M; ++col) {
							std::complex<float> yj = 0.0;
							for(IndexType j=0; j < N; ++j)
								for(IndexType mu=0; mu < 8; ++mu)
									yj += std::conj(A[ j + N*row + N*N*mu]) * x[mu][ j + col*N + d*N ];
							y[row + N*col + d*N] -= yj;
						}
					}
				}
			}
	};

	template<int N, template<const int,const int> class MatMult>
	void CMatMultNaiveT(int ncols, std::complex<float>*y,
				const std::complex<float>* A, int alpha,
				const std::complex<const float>* const x_[8]) {

		int ncols0 = ncols;
		const std::complex<float>* x[8];
		for (int mu=0; mu<8; mu++) x[mu] = (const std::complex<float>*)x_[mu];
		while (ncols > 0) {
			if (ncols >= 256) {
				MatMult<N,256>::fun(y, A, alpha, x, ncols0-ncols);
				ncols -= 256;
			} else if (ncols >= 128) {
				MatMult<N,128>::fun(y, A, alpha, x, ncols0-ncols);
				ncols -= 128;
			} else if (ncols >= 64) {
				MatMult<N,64>::fun(y, A, alpha, x, ncols0-ncols);
				ncols -= 64;
			} else if (ncols >= 32) {
				MatMult<N,32>::fun(y, A, alpha, x, ncols0-ncols);
				ncols -= 32;
			} else if (ncols >= 16) {
				MatMult<N,16>::fun(y, A, alpha, x, ncols0-ncols);
				ncols -= 16;
			} else if (ncols >= 8) {
				MatMult<N,8>::fun(y, A, alpha, x, ncols0-ncols);
				ncols -= 8;
			} else if (ncols >= 4) {
				MatMult<N,4>::fun(y, A, alpha, x, ncols0-ncols);
				ncols -= 4;
			} else if (ncols >= 3) {
				MatMult<N,3>::fun(y, A, alpha, x, ncols0-ncols);
				ncols -= 3;
			} else if (ncols >= 2) {
				MatMult<N,2>::fun(y, A, alpha, x, ncols0-ncols);
				ncols -= 2;
			} else if (ncols >= 1) {
				MatMult<N,1>::fun(y, A, alpha, x, ncols0-ncols);
				ncols -= 1;
			}
		}
	}


	//
	// BLAS in row major
	//

#ifndef MGPROTO_USE_CBLAS
	extern "C" void cgemm_(const char *transa, const char *transb,
			LAPACK_BLASINT *m, LAPACK_BLASINT *n,
			LAPACK_BLASINT *k, std::complex<float> *alpha,
			const std::complex<float> *a, LAPACK_BLASINT *lda,
			const std::complex<float> *b, LAPACK_BLASINT *ldb,
			std::complex<float> *beta, std::complex<float> *c,
			LAPACK_BLASINT *ldc);
        extern "C" void cgemv_(const char *transa, LAPACK_BLASINT *m, LAPACK_BLASINT *n,
			const std::complex<float> *alpha, const std::complex<float> *a,
			LAPACK_BLASINT *lda, const std::complex<float> *x,
			LAPACK_BLASINT *incx, const std::complex<float> *beta,
			std::complex<float> *y, LAPACK_BLASINT *incy);
#else
	#include <cblas.h>
	CBLAS_TRANSPOSE toTrans(const char *trans) {
		const char t = *trans;
		if (t == 'n' || t == 'N') return CblasNoTrans;
		if (t == 't' || t == 'T') return CblasTrans;
		if (t == 'c' || t == 'C') return CblasConjTrans;
	}
#endif


	typedef float T;
	struct BlasRowMajor {

		static inline void fun(T* y,
				const T* A, float alpha,
				const T* const x[8], unsigned int M, unsigned int N, unsigned int d, unsigned int ld) {

			if (N <= 0 || M <= 0) return;
			assert(ld >= N);

			for (int mu=0; mu<8; mu++) {
#ifndef MGPROTO_USE_CBLAS
				LAPACK_BLASINT m = (LAPACK_BLASINT)M, n = (LAPACK_BLASINT)N, one = 1, ld_ = (LAPACK_BLASINT)ld;
				std::complex<float> alphac(alpha), onec(1.0);
				if (N == 1) {
					cgemv_("N", &m, &m, &alphac, (const std::complex<T>*)&A[M*M*2*mu], &m, (const std::complex<T>*)&x[mu][d*2], &ld_, &onec, (std::complex<T>*)&y[d*2], &ld_);
				} else {

					// BLAS's Fortran interface only deals with column-major matrices and x and y are row-major.
					// Transpose x and y implicitly to pass x and y as column-major matrices:
					//   y = A * x  --> y^T = x^T * A^T
					cgemm_("N", "T", &n, &m, &m, &alphac, (const std::complex<T>*)&x[mu][d*2], &ld_, (const std::complex<T>*)&A[M*M*2*mu], &m, &onec, (std::complex<T>*)&y[d*2], &ld_);
#else
					cblas_cgemm(CblasRowMajor, toTrans("T"), toTrans("N"), M, N, M, &alphac, &A[M*M*2*mu], M, &x[mu][d*2], ld, &onec, &y[d*2], ld);
#endif
				}
			}
		}


		static inline void test(const T* __restrict__ y, const T* __restrict__ y0,
				const T* __restrict__  A, float alpha,
				const T* __restrict__ const x[8], unsigned int M, unsigned int N, unsigned int d, unsigned int ld) {

			for(IndexType row=0; row < M; ++row) {
				for(IndexType col=0; col < N; ++col) {
					float yr=y0[ (row*ld + col + d)*2 ], yi=y0[ (row*ld + col + d)*2 + 1 ];
					for (unsigned int mu=0; mu < 8; ++mu) {
						for(IndexType j=0; j < M; ++j) {
							yr += alpha * (A[ (mu*M*M + row + j*M)*2 ] * x[mu][ (j*ld + col + d)*2     ] - A[ (mu*M*M + row + j*M)*2 + 1 ] * x[mu][ (j*ld + col + d)*2 + 1 ]);
							yi += alpha * (A[ (mu*M*M + row + j*M)*2 ] * x[mu][ (j*ld + col + d)*2 + 1 ] + A[ (mu*M*M + row + j*M)*2 + 1 ] * x[mu][ (j*ld + col + d)*2     ]);
						}
					}

					//float d=fabs(yr - y[ rowr + col*M*2 ]), v=fabs(yr);
					//if (d > v*1e-4) std::cout << "Diff " << row << " " << col << " diff: " << d << " abs: " << v << std::endl;
					assert(fabs(yr - y[ (row*ld + col + d)*2    ]) <= fabs(yr)*1e-4);
					assert(fabs(yi - y[ (row*ld + col + d)*2 + 1]) <= fabs(yi)*1e-4);
				}
			}
		}
	};


#ifdef MG_USE_XSIMD

	// template<unsigned int K, unsigned int BK, unsigned int V, unsigned int BM, unsigned int BN, unsigned int ldA, unsigned int ldx> struct AuxIJ;

	// template<unsigned int BM, unsigned int BN, unsigned int ldA, unsigned int ldx>
	// 	struct AuxIJ<12,12,8,BM,BN,ldA,ldx> {
	// 		static inline void fun(T* __restrict__ y, unsigned int ldy,
	// 				const T* __restrict__  A,
	// 				const T* __restrict__  x) {

	// 			static_assert(ldA % 12 == 0);
	// 			static_assert(ldx % 12 == 0);

	// 			using b_type8 = xsimd::batch<T,8>;
	// 			using b_type4 = xsimd::batch<T,4>;
	// 			constexpr unsigned int V = 8;
	// 			b_type8 xr8[BN], xi8[BN];
	// 			b_type4 xr4[BN], xi4[BN];
	// 			for(unsigned int col=0; col < BN; ++col) {
	// 				unsigned int j=0;
	// 				xr8[col].load_aligned(&x[ (j + col*ldx)*2 ]);
	// 				xi8[col].load_aligned(&x[ (j + col*ldx)*2+V ]);
	// 				j = V;
	// 				xr4[col].load_aligned(&x[ (j + col*ldx)*2 ]);
	// 				xi4[col].load_aligned(&x[ (j + col*ldx)*2+4 ]);
	// 			}

	// 			for(unsigned int row=0; row < BM; ++row) {
	// 				b_type8 Ar8, Ai8;
	// 				b_type4 Ar4, Ai4;
	// 				unsigned int j=0;
	// 				Ar8.load_aligned(&A[ (row*ldA + j)*2 ]);
	// 				Ai8.load_aligned(&A[ (row*ldA + j)*2+V ]);
	// 				j = V;
	// 				Ar4.load_aligned(&A[ (row*ldA + j)*2 ]);
	// 				Ai4.load_aligned(&A[ (row*ldA + j)*2+4 ]);
	// 				for(unsigned int col=0; col < BN; ++col) { 
	// 					y[(row + ldy*col)*2] += xsimd::hadd(Ar8 * xr8[col] - Ai8 * xi8[col])
	// 					                     +  xsimd::hadd(Ar4 * xr4[col] - Ai4 * xi4[col]);
	// 					y[(row + ldy*col)*2+1] += xsimd::hadd(Ai8 * xr8[col] + Ar8 * xi8[col])
	// 					                       +  xsimd::hadd(Ai4 * xr4[col] + Ar4 * xi4[col]);
	// 				} 
	// 			}
	// 		}
	// 	};

	// template<unsigned int K, unsigned int BK, unsigned int V, unsigned int BM, unsigned int BN, unsigned int ldA, unsigned int ldx> struct AuxJI;

	// template<unsigned int BM, unsigned int BN, unsigned int ldA, unsigned int ldx>
	// 	struct AuxJI<12,12,8,BM,BN,ldA,ldx> {
	// 		static inline void fun(T* __restrict__ y, unsigned int ldy,
	// 				const T* __restrict__  A,
	// 				const T* __restrict__  x) {

	// 			static_assert(ldA % 12 == 0);
	// 			static_assert(ldx % 12 == 0);

	// 			using b_type8 = xsimd::batch<T,8>;
	// 			using b_type4 = xsimd::batch<T,4>;
	// 			constexpr unsigned int V = 8;
	// 			b_type8 Ar8[BM], Ai8[BM];
	// 			b_type4 Ar4[BM], Ai4[BM];
	// 			for(unsigned int row=0; row < BM; ++row) {
	// 				unsigned int j=0;
	// 				Ar8[row].load_aligned(&A[ (row*ldA + j)*2 ]);
	// 				Ai8[row].load_aligned(&A[ (row*ldA + j)*2+V ]);
	// 				j = V;
	// 				Ar4[row].load_aligned(&A[ (row*ldA + j)*2 ]);
	// 				Ai4[row].load_aligned(&A[ (row*ldA + j)*2+4 ]);
	// 			}
	// 			for(unsigned int col=0; col < BN; ++col) { 
	// 				b_type8 xr8, xi8;
	// 				b_type4 xr4, xi4;
	// 				unsigned int j=0;
	// 				xr8.load_aligned(&x[ (j + col*ldx)*2 ]);
	// 				xi8.load_aligned(&x[ (j + col*ldx)*2+V ]);
	// 				j = V;
	// 				xr4.load_aligned(&x[ (j + col*ldx)*2 ]);
	// 				xi4.load_aligned(&x[ (j + col*ldx)*2+4 ]);

	// 				for(unsigned int row=0; row < BM; ++row) {
	// 					y[(row + ldy*col)*2] += xsimd::hadd(Ar8[row] * xr8 - Ai8[row] * xi8)
	// 					                     +  xsimd::hadd(Ar4[row] * xr4 - Ai4[row] * xi4);
	// 					y[(row + ldy*col)*2+1] += xsimd::hadd(Ai8[row] * xr8 + Ar8[row] * xi8)
	// 					                       +  xsimd::hadd(Ai4[row] * xr4 + Ar4[row] * xi4);
	// 				} 
	// 			}
	// 		}
	// 	};

	// template<unsigned int N, unsigned int M, unsigned int BN, unsigned int BM, unsigned int V>
	// 	static void fun12IJ(T* RESTRICT y,
	// 			const T* RESTRICT  A, std::complex<float> alpha,
	// 			const T* RESTRICT  x) {

	// 		for(unsigned int row=0; row < N; row+=BN) {
	// 			for(unsigned int col=0; col < M; col+=BM) {
	// 				for(unsigned int j=0; j < N; j+=12) {
	// 					//r += /* alpha * */ A[ row + BN*j ] * x[ j + col*BN ];
	// 					//Aux<BK,V>::fun0<BN,BM,N,N>(P<N,V>(y, 0, col), row, N, P<N,V>(A, j, row), P<N,V>(x, j, col));
	// 					AuxIJ<12,12,8,BN,BM,N,N>::fun(&y[(row + col*M)*2], N, &A[(row*N + j)*2], &x[(j + col*N)*2]);
	// 				}
	// 			}
	// 		}
	// 	}

	// template<unsigned int N, unsigned int M, unsigned int BN, unsigned int BM, unsigned int V>
	// 	static void fun12JI(T* RESTRICT y,
	// 			const T* RESTRICT  A, std::complex<float> alpha,
	// 			const T* RESTRICT  x) {

	// 		for(unsigned int col=0; col < M; col+=BM) {
	// 			for(unsigned int row=0; row < N; row+=BN) {
	// 				for(unsigned int j=0; j < N; j+=12) {
	// 					//r += /* alpha * */ A[ row + BN*j ] * x[ j + col*BN ];
	// 					//Aux<BK,V>::fun0<BN,BM,N,N>(P<N,V>(y, 0, col), row, N, P<N,V>(A, j, row), P<N,V>(x, j, col));
	// 					AuxJI<12,12,8,BN,BM,N,N>::fun(&y[(row + col*M)*2], N, &A[(row*N + j)*2], &x[(j + col*N)*2]);
	// 				}
	// 			}
	// 		}
	// 	}

	// template<unsigned int K, unsigned int BK, unsigned int N, unsigned int V, unsigned int BM, unsigned int BN, unsigned int ldA, unsigned int ldx, unsigned int ldy> struct AuxIJAxpy;

	// template<unsigned int BM, unsigned int N, unsigned int BN, unsigned int ldA, unsigned int ldx, unsigned int ldy>
	// 	struct AuxIJAxpy<12,12,N,8,BM,BN,ldA,ldx,ldy> {
	// 		static inline void funASplit(T* __restrict__ y,
	// 				const T* __restrict__  A,
	// 				const T* __restrict__  x) {

	// 			static_assert(ldA % 12 == 0);
	// 			static_assert(ldx % 12 == 0);
	// 			static_assert(ldy % 12 == 0);

	// 			using b_type8 = xsimd::batch<T,8>;
	// 			using b_type4 = xsimd::batch<T,4>;
	// 			constexpr unsigned int V = 8;
	// 			constexpr unsigned int BNA=3;

	// 			for (unsigned int colA0=0; colA0 < 12; colA0+=BNA) {
	// 				// Type 8
	// 				b_type8 Ar8[BNA], Ai8[BNA];
	// 				for(unsigned int col=0, colA1=colA0; col < BNA; ++col, ++colA1) {
	// 					unsigned int j=0;
	// 					Ar8[col].load_aligned(&A[ (j + colA1*ldA)*2 ]);
	// 					Ai8[col].load_aligned(&A[ (j + colA1*ldA)*2+V ]);
	// 				}

	// 				for(unsigned int colx=0; colx < N; ++colx) {
	// 					b_type8 yr8, yi8;
	// 					unsigned int j=0;
	// 					yr8.load_aligned(&y[ (j + colx*ldy)*2 ]);
	// 					yi8.load_aligned(&y[ (j + colx*ldy)*2+V ]);
	// 					for(unsigned int col=0,colA1=colA0; col < BNA; ++col,++colA1) { 
	// 						IndexType jr = (colA1 < V ? colA1     : V*2 + colA1 - V      );
	// 						IndexType ji = (colA1 < V ? colA1 + V : V*2 + colA1 - V + V/2);
	// 						b_type8 xr8(x[ colx*ldx*2 + jr]);
	// 						b_type8 xi8(x[ colx*ldx*2 + ji]);
	// 						//yr8 += Ar8[col] * xr8 - Ai8[col] * xi8;
	// 						yr8 = xsimd::fma(Ar8[col], xr8, xsimd::fnma(Ai8[col], xi8, yr8));
	// 						//yi8 += Ai8[col] * xr8 + Ar8[col] * xi8;
	// 						yi8 = xsimd::fma(Ai8[col], xr8, xsimd::fma(Ar8[col], xi8, yi8));
	// 					} 
	// 					yr8.store_aligned(&y[ (j + colx*ldy)*2 ]);
	// 					yi8.store_aligned(&y[ (j + colx*ldy)*2+V ]);
	// 				}

	// 				// Type 4
	// 				b_type4 Ar4[BNA], Ai4[BNA];
	// 				for(unsigned int col=0, colA1=colA0; col < BNA; ++col, ++colA1) {
	// 					unsigned int j = V;
	// 					Ar4[col].load_aligned(&A[ (j + colA1*ldA)*2 ]);
	// 					Ai4[col].load_aligned(&A[ (j + colA1*ldA)*2+4 ]);
	// 				}

	// 				for(unsigned int colx=0; colx < N; ++colx) {
	// 					b_type4 yr4, yi4;
	// 					unsigned int j=V;
	// 					yr4.load_aligned(&y[ (j + colx*ldy)*2 ]);
	// 					yi4.load_aligned(&y[ (j + colx*ldy)*2+4 ]);
	// 					for(unsigned int col=0,colA1=colA0; col < BNA; ++col,++colA1) { 
	// 						IndexType jr = (colA1 < V ? colA1     : V*2 + colA1 - V      );
	// 						IndexType ji = (colA1 < V ? colA1 + V : V*2 + colA1 - V + V/2);
	// 						b_type4 xr4(x[ colx*ldx*2 + jr]);
	// 						b_type4 xi4(x[ colx*ldx*2 + ji]);
	// 						//yr4 += Ar4[col] * xr4 - Ai4[col] * xi4;
	// 						yr4 = xsimd::fma(Ar4[col], xr4, xsimd::fnma(Ai4[col], xi4, yr4));
	// 						//yi4 += Ai4[col] * xr4 + Ar4[col] * xi4;
	// 						yi4 = xsimd::fma(Ai4[col], xr4, xsimd::fma(Ar4[col], xi4, yi4));
	// 					} 
	// 					yr4.store_aligned(&y[ (j + colx*ldy)*2 ]);
	// 					yi4.store_aligned(&y[ (j + colx*ldy)*2+4 ]);
	// 				}
	// 			}
	// 		}

	// 		static inline void funASplit4(T* __restrict__ y,
	// 				const T* __restrict__  A,
	// 				const T* __restrict__  x) {

	// 			static_assert(ldA % 12 == 0);
	// 			static_assert(ldx % 12 == 0);
	// 			static_assert(ldy % 12 == 0);

	// 			using b_type8 = xsimd::batch<T,8>;
	// 			using b_type4 = xsimd::batch<T,4>;
	// 			constexpr unsigned int V = 8;
	// 			constexpr unsigned int BNA=4;

	// 			for (unsigned int colA0=0; colA0 < 12; colA0+=BNA) {
	// 				// Type 8
	// 				b_type8 Ar8[BNA], Ai8[BNA];
	// 				for(unsigned int col=0, colA1=colA0; col < BNA; ++col, ++colA1) {
	// 					unsigned int j=0;
	// 					Ar8[col].load_aligned(&A[ (j + colA1*ldA)*2 ]);
	// 					Ai8[col].load_aligned(&A[ (j + colA1*ldA)*2+V ]);
	// 				}

	// 				for(unsigned int colx=0; colx < N; ++colx) {
	// 					b_type8 yr8, yi8;
	// 					unsigned int j=0;
	// 					yr8.load_aligned(&y[ (j + colx*ldy)*2 ]);
	// 					yi8.load_aligned(&y[ (j + colx*ldy)*2+V ]);
	// 					b_type4 xr4, xi4;
	// 					IndexType jr = (colA0 < V ? colA0     : V*2 + colA0 - V      );
	// 					IndexType ji = (colA0 < V ? colA0 + V : V*2 + colA0 - V + V/2);
	// 					xr4.load_aligned(&x[ colx*ldx*2 + jr]);
	// 					xi4.load_aligned(&x[ colx*ldx*2 + ji]);
	// 					for(unsigned int col=0,colA1=colA0; col < BNA; ++col,++colA1) { 
	// 						b_type8 xr8col(xr4[col]);
	// 						b_type8 xi8col(xi4[col]);
	// 						//yr8 += Ar8[col] * xr8 - Ai8[col] * xi8;
	// 						yr8 = xsimd::fma(Ar8[col], xr8col, xsimd::fnma(Ai8[col], xi8col, yr8));
	// 						//yi8 += Ai8[col] * xr8 + Ar8[col] * xi8;
	// 						yi8 = xsimd::fma(Ai8[col], xr8col, xsimd::fma(Ar8[col], xi8col, yi8));
	// 					} 
	// 					yr8.store_aligned(&y[ (j + colx*ldy)*2 ]);
	// 					yi8.store_aligned(&y[ (j + colx*ldy)*2+V ]);
	// 				}

	// 				// Type 4
	// 				b_type4 Ar4[BNA], Ai4[BNA];
	// 				for(unsigned int col=0, colA1=colA0; col < BNA; ++col, ++colA1) {
	// 					unsigned int j = V;
	// 					Ar4[col].load_aligned(&A[ (j + colA1*ldA)*2 ]);
	// 					Ai4[col].load_aligned(&A[ (j + colA1*ldA)*2+4 ]);
	// 				}

	// 				for(unsigned int colx=0; colx < N; ++colx) {
	// 					b_type4 yr4, yi4;
	// 					unsigned int j=V;
	// 					yr4.load_aligned(&y[ (j + colx*ldy)*2 ]);
	// 					yi4.load_aligned(&y[ (j + colx*ldy)*2+4 ]);
	// 					b_type4 xr4, xi4;
	// 					IndexType jr = (colA0 < V ? colA0     : V*2 + colA0 - V      );
	// 					IndexType ji = (colA0 < V ? colA0 + V : V*2 + colA0 - V + V/2);
	// 					xr4.load_aligned(&x[ colx*ldx*2 + jr]);
	// 					xi4.load_aligned(&x[ colx*ldx*2 + ji]);
	// 					for(unsigned int col=0,colA1=colA0; col < BNA; ++col,++colA1) { 
	// 						b_type4 xr4col(xr4[col]);
	// 						b_type4 xi4col(xi4[col]);
	// 						//yr4 += Ar4[col] * xr4 - Ai4[col] * xi4;
	// 						yr4 = xsimd::fma(Ar4[col], xr4col, xsimd::fnma(Ai4[col], xi4col, yr4));
	// 						//yi4 += Ai4[col] * xr4 + Ar4[col] * xi4;
	// 						yi4 = xsimd::fma(Ai4[col], xr4col, xsimd::fma(Ar4[col], xi4col, yi4));
	// 					} 
	// 					yr4.store_aligned(&y[ (j + colx*ldy)*2 ]);
	// 					yi4.store_aligned(&y[ (j + colx*ldy)*2+4 ]);
	// 				}
	// 			}
	// 		}

	// 		static inline void funASplit4Split(T* __restrict__ y,
	// 				const T* __restrict__  A,
	// 				const T* __restrict__ const x[8]) {

	// 			static_assert(ldA % 12 == 0);
	// 			static_assert(ldx % 12 == 0);
	// 			static_assert(ldy % 12 == 0);

	// 			using b_type8 = xsimd::batch<T,8>;
	// 			using b_type4 = xsimd::batch<T,4>;
	// 			constexpr unsigned int V = 8;
	// 			constexpr unsigned int BNA=3;
	// 			constexpr unsigned int Mu=1;

	// 			// Type 8
	// 			for (unsigned int mu0=0; mu0 < 8; mu0+=Mu) {
	// 				for (unsigned int colA0=0; colA0 < 12; colA0+=BNA) {
	// 					b_type8 Ar8[BNA*Mu], Ai8[BNA*Mu];
	// 					for(unsigned int mu=0, mu1=mu0; mu < Mu; ++mu, ++mu1) {
	// 						for(unsigned int col=0, colA1=colA0; col < BNA; ++col, ++colA1) {
	// 							unsigned int j=0;
	// 							Ar8[col + mu*BNA].load_aligned(&A[ mu1*ldA*ldA*2 + (j + colA1*ldA)*2 ]);
	// 							Ai8[col + mu*BNA].load_aligned(&A[ mu1*ldA*ldA*2 + (j + colA1*ldA)*2+V ]);
	// 						}
	// 					}

	// 					for(unsigned int colx=0; colx < N; ++colx) {
	// 						IndexType jr = (colA0 < V ? colA0     : V*2 + colA0 - V      );
	// 						IndexType ji = (colA0 < V ? colA0 + V : V*2 + colA0 - V + V/2);
	// 						unsigned int j=0;

	// 						// Real
	// 						b_type8 yr8;
	// 						yr8.load_aligned(&y[ (j + colx*ldy)*2 ]);
	// 						for(unsigned int mu=0, mu1=mu0; mu < Mu; ++mu, ++mu1) {
	// 							for(unsigned int col=0,colA1=colA0; col < BNA; ++col,++colA1) { 
	// 								b_type8 xr8col(x[mu1][colx*ldx*2 + jr + col]);
	// 								b_type8 xi8col(x[mu1][colx*ldx*2 + ji + col]);
	// 								//yr8 += Ar8[col] * xr8 - Ai8[col] * xi8;
	// 								yr8 = xsimd::fma(Ar8[col + mu*BNA], xr8col, xsimd::fnma(Ai8[col + mu*BNA], xi8col, yr8));
	// 							} 
	// 						}
	// 						yr8.store_aligned(&y[ (j + colx*ldy)*2 ]);

	// 						// Imaginary
	// 						b_type8 yi8;
	// 						yi8.load_aligned(&y[ (j + colx*ldy)*2+V ]);
	// 						for(unsigned int mu=0, mu1=mu0; mu < Mu; ++mu, ++mu1) {
	// 							for(unsigned int col=0,colA1=colA0; col < BNA; ++col,++colA1) { 
	// 								b_type8 xr8col(x[mu1][colx*ldx*2 + jr + col]);
	// 								b_type8 xi8col(x[mu1][colx*ldx*2 + ji + col]);
	// 								//yi8 += Ai8[col] * xr8 + Ar8[col] * xi8;
	// 								yi8 = xsimd::fma(Ai8[col + mu*BNA], xr8col, xsimd::fma(Ar8[col + mu*BNA], xi8col, yi8));
	// 							} 
	// 						}
	// 						yi8.store_aligned(&y[ (j + colx*ldy)*2+V ]);
	// 					}
	// 					// for(unsigned int mu=0, mu1=mu0+Mu; mu < Mu && mu1<8; ++mu, ++mu1)
	// 					// 	__builtin_prefetch(&x[mu1][0], 0 /*read*/, 3);
	// 				}
	// 			}

	// 			// Type 4
	// 			for (unsigned int mu0=0; mu0 < 8; mu0+=Mu) {
	// 				for (unsigned int colA0=0; colA0 < 12; colA0+=BNA) {
	// 					b_type4 Ar4[BNA*Mu], Ai4[BNA*Mu];
	// 					for(unsigned int mu=0, mu1=mu0; mu < Mu; ++mu, ++mu1) {
	// 						for(unsigned int col=0, colA1=colA0; col < BNA; ++col, ++colA1) {
	// 							unsigned int j = V;
	// 							Ar4[col + mu*BNA].load_aligned(&A[ mu1*ldA*ldA*2 + (j + colA1*ldA)*2 ]);
	// 							Ai4[col + mu*BNA].load_aligned(&A[ mu1*ldA*ldA*2 + (j + colA1*ldA)*2+4 ]);
	// 						}
	// 					}

	// 					for(unsigned int colx=0; colx < N; ++colx) {
	// 						IndexType jr = (colA0 < V ? colA0     : V*2 + colA0 - V      );
	// 						IndexType ji = (colA0 < V ? colA0 + V : V*2 + colA0 - V + V/2);
	// 						unsigned int j=V;

	// 						// Real
	// 						b_type4 yr4;
	// 						yr4.load_aligned(&y[ (j + colx*ldy)*2 ]);
	// 						for(unsigned int mu=0, mu1=mu0; mu < Mu; ++mu, ++mu1) {
	// 							for(unsigned int col=0,colA1=colA0; col < BNA; ++col,++colA1) { 
	// 								b_type4 xr4col(x[mu1][colx*ldx*2 + jr + col]);
	// 								b_type4 xi4col(x[mu1][colx*ldx*2 + ji + col]);
	// 								//yr4 += Ar4[col] * xr4 - Ai4[col] * xi4;
	// 								yr4 = xsimd::fma(Ar4[col + mu*BNA], xr4col, xsimd::fnma(Ai4[col + mu*BNA], xi4col, yr4));
	// 							} 
	// 						}
	// 						yr4.store_aligned(&y[ (j + colx*ldy)*2 ]);

	// 						// Imaginary
	// 						b_type4 yi4;
	// 						yi4.load_aligned(&y[ (j + colx*ldy)*2+4 ]);
	// 						for(unsigned int mu=0, mu1=mu0; mu < Mu; ++mu, ++mu1) {
	// 							for(unsigned int col=0,colA1=colA0; col < BNA; ++col,++colA1) { 
	// 								b_type4 xr4col(x[mu1][colx*ldx*2 + jr + col]);
	// 								b_type4 xi4col(x[mu1][colx*ldx*2 + ji + col]);
	// 								//yi4 += Ai4[col] * xr4 + Ar4[col] * xi4;
	// 								yi4 = xsimd::fma(Ai4[col + mu*BNA], xr4col, xsimd::fma(Ar4[col + mu*BNA], xi4col, yi4));
	// 							} 
	// 						}
	// 						yi4.store_aligned(&y[ (j + colx*ldy)*2+4 ]);
	// 					}
	// 				}
	// 			}
	// 		}

	// 		static inline void test(const T* __restrict__ y, const T* __restrict__ y0,
	// 				const T* __restrict__  A,
	// 				const T* __restrict__  x) {
	// 			constexpr unsigned int V = 8;
	// 			constexpr unsigned int M = 12;
	// 			for(IndexType row=0; row < M; ++row) {
	// 				IndexType rowr = (row < V)*(row    ) + (row >= V)*(V*2 + row - V      );
	// 				IndexType rowi = (row < V)*(row + V) + (row >= V)*(V*2 + row - V + V/2);
	// 				for(IndexType col=0; col < N; ++col) {
	// 					float yr=y0[ rowr + col*M*2 ], yi=y0[ rowi + col*M*2 ];
	// 					for(IndexType j=0; j < M; ++j) {
	// 						IndexType jr = (j < V ? j     : V*2 + j - V      );
	// 						IndexType ji = (j < V ? j + V : V*2 + j - V + V/2);
	// 						yr += A[ rowr + j*M*2 ] * x[ jr + col*M*2 ] - A[ rowi + j*M*2 ] * x[ ji + col*M*2 ];
	// 						yi += A[ rowr + j*M*2 ] * x[ ji + col*M*2 ] + A[ rowi + j*M*2 ] * x[ jr + col*M*2 ];
	// 					}
	// 					float d=fabs(yr - y[ rowr + col*M*2 ]), v=fabs(yr);
	// 					if (d > v*1e-4) std::cout << "Diff " << row << " " << col << " diff: " << d << " abs: " << v << std::endl;
	// 					assert(fabs(yr - y[ rowr + col*M*2 ]) <= fabs(yr)*1e-4);
	// 					assert(fabs(yi - y[ rowi + col*M*2 ]) <= fabs(yi)*1e-4);
	// 				}
	// 			}
	// 		}
	// 	};

	// template<unsigned int N, unsigned int M, unsigned int BN, unsigned int BM, unsigned int V>
	// 	static inline void fun12IJAxpy(T* RESTRICT y,
	// 			const T* RESTRICT  A, std::complex<float> alpha,
	// 			const T* RESTRICT  x) {

	// 		//for(unsigned int col=0; col < M; col+=BM) {
	// 		//	for(unsigned int row=0; row < N; row+=BN) {
	// 		//		for(unsigned int j=0; j < N; j+=12) {
	// 		//			//r += /* alpha * */ A[ row + BN*j ] * x[ j + col*BN ];
	// 		//			//Aux<BK,V>::fun0<BN,BM,N,N>(P<N,V>(y, 0, col), row, N, P<N,V>(A, j, row), P<N,V>(x, j, col));
	// 		//			AuxIJAxpy<12,12,8,BN,BM,N,N,N>::fun(&y[(row + col*M)*2], &A[(row*N + j)*2], &x[(j + col*N)*2]);
	// 		//		}
	// 		//	}
	// 		//}
	// 		//std::vector<float> y0(y, y+N*M*2);
	// 		AuxIJAxpy<12,12,M,8,BN,BM,N,N,N>::funASplit(y, A, x);
	// 		//AuxIJAxpy<12,12,M,8,BN,BM,N,N,N>::test(y, y0.data(), A, x);
	// 	}



	// template<const int N, const int M>
	// 	struct CMatMultNaiveTT {
	// 		static inline void fun(std::complex<float> *y,
	// 				const std::complex<float>* A, std::complex<float> alpha,
	// 				const std::complex<float>* x) {

	// 			//if (M <= 0) return;
	// 			//if      (M == 1) fun12<N,M,6,1,8>(y, A, alpha, x);
	// 			//else if (M == 2) fun12<N,M,6,2,8>(y, A, alpha, x);
	// 			//else if (M == 3) fun12<N,M,3,3,8>(y, A, alpha, x);
	// 			//else if (M >= 4) fun12<N,M,4,4,8>(y, A, alpha, x);
	// 			//if (M <= 4) {
	// 			//	fun12IJ<N,M,12,M,8>((float*)y, (const float*)A, alpha, (const float*)x);
	// 			//} else {
	// 			//	fun12JI<N,M,6,M,8>((float*)y, (const float*)A, alpha, (const float*)x);
	// 			//}
	// 			if (M <= 256) {
	// 				fun12IJAxpy<N,M,12,M<=2?M:2,8>((float*)y, (const float*)A, alpha, (const float*)x);
	// 			} else {
	// 				fun12JI<N,M,6,M,8>((float*)y, (const float*)A, alpha, (const float*)x);
	// 			}
	// 		}

	// 		static inline void fun(float *y, int ld,
	// 				const std::array<const float*,8> A, std::complex<float> alpha,
	// 				const std::array<const float*,8> x, int disp) {

	// 			disp = disp*ld;
	// 			const float * const xd[8] = {x[0] + disp, x[1] + disp, x[2] + disp, x[3] + disp, x[4] + disp, x[5] + disp, x[6] + disp, x[7] + disp};
	// 			AuxIJAxpy<N,N,M,8,0,0,N,N,N>::funASplit4Split(y, A[0], xd);
	// 		}
	// 	};

	 template<unsigned int M, unsigned int N, unsigned int V, unsigned int BM, unsigned int BN, int sign, unsigned int ldA> struct SimdBlas;

	 template<unsigned int N, unsigned int V, unsigned int BM, unsigned int BN, int sign, unsigned int ldA>
	 	struct SimdBlas<0,N,V,BM,BN,sign,ldA> {
			static inline void fun(T* __restrict__ y,
					const T* __restrict__  A,
					const T* __restrict__ const x[8], unsigned int d, unsigned int ld) {}
	 		static inline void test(const T* __restrict__ y, const T* __restrict__ y0,
	 				const T* __restrict__  A,
	 				const T* __restrict__ const x[8], unsigned int d, unsigned int ld) {}
		};

	 template<unsigned int M, unsigned int N, unsigned int V, unsigned int BM, unsigned int BN, int sign, unsigned int ldA>
	 	struct SimdBlas {

			static inline void fun(T* __restrict__ y,
					const T* __restrict__  A,
					const T* __restrict__ const x[8], unsigned int d, unsigned int ld) {

				static_assert(N % V == 0);
				static_assert(BN % V == 0);
				assert(ld % V == 0);

				using b_type = xsimd::batch<T,V>;

				for (unsigned int row0=0; row0+BM <= M; row0+=BM) {
					for (unsigned int col0=0; col0+BN <= N; col0+=BN) {
						b_type yr[BM*BN/V];
						b_type yi[BM*BN/V];
						for(unsigned int row=0,row1=row0; row < BM; ++row,++row1) { 
							for(unsigned int col=0,col1=col0+d; col < BN/V; ++col,col1+=V) { 
								yr[row + col*BM].load_aligned(&y[ (row1*ld + col1)*2   ]);
								yi[row + col*BM].load_aligned(&y[ (row1*ld + col1)*2+V ]);
							}
						}

						unsigned int Aidx = 0;
						for(unsigned int col=0,col1=col0+d; col < BN/V; ++col,col1+=V) { 
							for (unsigned int mu=0; mu < 8; ++mu) {
								for (unsigned int k=0; k < M; ++k) {
									b_type xr, xi;
									xr.load_aligned(&x[mu][ (k*ld + col1)*2   ]);
									xi.load_aligned(&x[mu][ (k*ld + col1)*2+V ]);
									for(unsigned int row=0,row1=row0; row < BM; ++row,++row1) { 
										//unsigned int Aidx = mu*ldA*ldA + row1 + k*ldA, Aidxr = Aidx*2, Aidxi = Aidxr+1;
										unsigned int Aidxr = Aidx*2, Aidxi = Aidxr+1; ++Aidx;
										b_type Ar(A[Aidxr]), Ai(A[Aidxi]);
										if (sign > 0) {
											//yr[row + col*BM] += A[Aidxr] * xr - A[Aidxi] * xi;
											yr[row + col*BM] = xsimd::fma(Ar, xr, xsimd::fnma(Ai, xi, yr[row + col*BM]));
											yi[row + col*BM] = xsimd::fma(Ai, xr, xsimd::fma (Ar, xi, yi[row + col*BM]));
										} else {
											//yr[row + col*BM] -= A[Aidxr] * xr - A[Aidxi] * xi;
											yr[row + col*BM] = xsimd::fnma(Ar, xr, xsimd::fma(Ai, xi, yr[row + col*BM]));
											yi[row + col*BM] = xsimd::fnma(Ai, xr, xsimd::fnma (Ar, xi, yi[row + col*BM]));
										}
									}
								}
							}
						}

						for(unsigned int row=0,row1=row0; row < BM; ++row,++row1) { 
							for(unsigned int col=0,col1=col0+d; col < BN/V; ++col,col1+=V) { 
								yr[row + col*BM].store_aligned(&y[ (row1*ld + col1)*2   ]);
								yi[row + col*BM].store_aligned(&y[ (row1*ld + col1)*2+V ]);
							}
						}
					}
				}

				SimdBlas<M%BM,N,V,M%BM,BN,sign,ldA>::fun(y, A + (M - M%BM)*2, x, d, ld);
			}


	 		static inline void test(const T* __restrict__ y, const T* __restrict__ y0,
	 				const T* __restrict__  A,
	 				const T* __restrict__ const x[8], unsigned int d, unsigned int ld) {
	 			for(IndexType row=0; row < M; ++row) {
	 				for(IndexType col=0; col < N; ++col) {
	 					float yr=y0[ (row*ld + col/V*V)*2 + col%V ], yi=y0[ (row*ld + col/V*V)*2 + col%V + V ];
						for (unsigned int mu=0; mu < 8; ++mu) {
							for(IndexType j=0; j < M; ++j) {
								const T xr = x[mu][ (j*ld + col/V*V)*2 + col%V     ];
								const T xi = x[mu][ (j*ld + col/V*V)*2 + col%V + V ];
								if (sign > 0) {
									yr += A[ (mu*M*M + row + j*M)*2 ] * xr - A[ (mu*M*M + row + j*M)*2 + 1 ] * xi;
									yi += A[ (mu*M*M + row + j*M)*2 ] * xi + A[ (mu*M*M + row + j*M)*2 + 1 ] * xr;
								} else {
									yr -= A[ (mu*M*M + row + j*M)*2 ] * xr - A[ (mu*M*M + row + j*M)*2 + 1 ] * xi;
									yi -= A[ (mu*M*M + row + j*M)*2 ] * xi + A[ (mu*M*M + row + j*M)*2 + 1 ] * xr;
								}
							}
						}
						
	 					//float d=fabs(yr - y[ rowr + col*M*2 ]), v=fabs(yr);
	 					//if (d > v*1e-4) std::cout << "Diff " << row << " " << col << " diff: " << d << " abs: " << v << std::endl;
	 					assert(fabs(yr - y[ (row*ld + col/V*V)*2 + col%V     ]) <= fabs(yr)*1e-4);
	 					assert(fabs(yi - y[ (row*ld + col/V*V)*2 + col%V  + V]) <= fabs(yi)*1e-4);
	 				}
	 			}
	 		}
	 	};



	template<unsigned int M, unsigned int V>
	int CMatMultRowMajorT(int ncols, T *y,
				const T* A, int alpha,
				const T* const x[8]) {
		int ncols0 = ncols;
		while (ncols >= V) {
			constexpr unsigned int BM = (V == 16 ? 12 : 5);
			if (alpha > 0) SimdBlas<M,V,V,BM,V, 1,M>::fun(y, A, x, ncols0-ncols, ncols0);
			else           SimdBlas<M,V,V,BM,V,-1,M>::fun(y, A, x, ncols0-ncols, ncols0);
			ncols -= V;
		}
		constexpr unsigned int V2 = std::max(V/2, 4u);
		if (ncols >= V2) {
			ncols = CMatMultRowMajorT<M,V2>(ncols, y, A, alpha, x);
		}
		return ncols;
	}

#endif // MG_USE_XSIMD

	template<int N>
	void CMatMultRowMajor(int ncols, std::complex<float>*y_,
				const std::complex<float>* A_, int alpha,
				const std::complex<const float>* const x_[8]) {

		int ncols0 = ncols;
		float *y = (float*)y_;
		const float *A = (const float*)A_;
		const float* x[8];
		for (int mu=0; mu<8; mu++) x[mu] = (const float*)x_[mu];
#ifdef MG_USE_XSIMD
		constexpr std::size_t simd_size = xsimd::simd_type<T>::size;
		ncols = CMatMultRowMajorT<N,simd_size>(ncols, y, A, alpha, x);
#endif
		if (ncols > 0) {
			//std::vector<float> y0(y, y+N*ncols*2);
			BlasRowMajor::fun(y, A, alpha, x, N, ncols, ncols0-ncols, ncols0);
			//BlasRowMajor::test(y, y0.data(), A, alpha, x, N, ncols, ncols0-ncols, ncols0);
			ncols = 0;
		}
	}
}

void XGEMM(const char *transa, const char *transb, LAPACK_BLASINT m,
		LAPACK_BLASINT n, LAPACK_BLASINT k, std::complex<float> alpha,
		const std::complex<float> *a, LAPACK_BLASINT lda, const std::complex<float> *b,
		LAPACK_BLASINT ldb, std::complex<float> beta, std::complex<float> *c,
		LAPACK_BLASINT ldc) {
	assert(c != a && c != b);
	//if (m == 12 && lda == 12 && ldb == 12 && ldc == 12 && (transb[0] == 'n' || transb[0] == 'N')) {
	//	//if (transa[0] == 'N' || transa[0] == 'n') {
	//		CMatMultNaiveT<12,CMatMultNaiveTT>(n, c, a, alpha, b);
	//	//} else {
	//	//	CMatMultNaiveT<12,CTransMatMultNaiveTT>(n, c, a, alpha, b);
	//	//}
	//	return;
	//}
	if (n == 1) {
		int mA; int nA;
		if (*transa == 'n' || *transa == 'N') mA = m, nA = k;
		else mA = k, nA = m;
		int incb = ((*transb == 'n' || *transb == 'N') ? 1 : ldb);
		int one = 1;
		cgemv_(transa, &mA, &nA, &alpha, a, &lda, b, &incb, &beta, c, &one);
		return;
	}
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

void CMatMultCoeffAddNaive(float beta,
		float* y,
		float alpha,
		const std::array<const float*,8> A,
		const std::array<const float*,8> x,
		IndexType N,
		IndexType ncol)
{
	if (fabs(beta) == 0.0) for (IndexType i=0; i < N*ncol*2; ++i) y[i] = 0.0;

	assert(alpha == -1.0 || alpha == 1.0);
	//CMatMultNaiveT<12,CMatMultNaiveTT>(ncol, (std::complex<float>*)y, (const std::complex<float>*)A[0], (int)alpha, (const std::complex<const float>* const*)x.data());
	CMatMultRowMajor<12>(ncol, (std::complex<float>*)y, (const std::complex<float>*)A[0], (int)alpha, (const std::complex<const float>* const*)x.data());
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
