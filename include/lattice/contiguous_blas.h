/*
 * block_blas.h
 *
 *  Created on: Nov 2, 2015
 *      Author: bjoo
 */

#ifndef INCLUDE_LATTICE_CONTIGUOUS_BLAS_H_
#define INCLUDE_LATTICE_CONTIGUOUS_BLAS_H_

#include "lattice/constants.h"
#include "utils/print_utils.h"

#include <complex>


using namespace MGUtils;

namespace MGGeometry {

	/** NormSquared of a list of real numbers.
	 *  I tried blocking this etc, but it seems to do best as it is
	 *  NB: This is not a tree reduction so, susceptible to rounding
	 */
	template<typename T>
	double NormSq(const T* data, IndexType num_real)
	{
		double result = 0;

#pragma omp simd aligned(data:MG_DEFAULT_ALIGNMENT)
		for(IndexType i=0; i < num_real; ++i) {
			result += data[i]*data[i];
		}

		return result;


	}

	/** InnerProduct of a list of complex numbers.
	 *  I tried blocking this etc, but it seems to do best as it is
	 *  NB: This is not a tree reduction so, susceptible to rounding
	 */


	template<typename T>
	std::complex<double> InnerProduct(const T* data_left, const T* data_right, IndexType num_complex)
	{
		double result_re=0;
		double result_im=0;

		/* FIXME: Is this reinterpretation always safe? */
		/* Best would be if I could vectorize with a stride of 2 but
		 * I worry it would break the vectorizer...
		 */
		const std::complex<T>* cl=reinterpret_cast<const std::complex<T>*>(data_left);
		const std::complex<T>* cr=reinterpret_cast<const std::complex<T>*>(data_right);

#pragma omp simd
		for(IndexType i=0; i < num_complex; ++i) {
			result_re += cl[i].real() * cr[i].real();
			result_re += cl[i].imag() * cr[i].imag();
			result_im += cl[i].real() * cr[i].imag();
			result_im -= cl[i].imag() * cr[i].real();
		}

		std::complex<double> result=std::complex<double>{result_re,result_im};
		return result;

	}

	/** Scale a function by a constant. Use this to E.g. Normalize a vector */
	template<typename T>
	void VScale(const T& scalar, T* data, IndexType num_real)
	{

#pragma omp simd
		for(IndexType i=0; i < num_real; ++i) {
			data[i] *= scalar;
		}
	}

	/** MCAXPY: y -= ax with a being Complex
	 * FIXME: This compiles nicely, and claims to vectorize but
	 * the vectorization gain appears small: ~1.0-1.5x
	 */
	template<typename T>
	void MCaxpy(T* y, const std::complex<T>& scalar, const T* x, IndexType num_complex)
	{
		const std::complex<T>* cx = reinterpret_cast<const std::complex<T>*>(x);
		std::complex<T>* cy = reinterpret_cast<std::complex<T>*>(y);
#pragma omp simd
		for(IndexType i=0; i < num_complex; i++) {
			cy[i]=cy[i] - scalar*cx[i];
		}

	}

}



#endif /* INCLUDE_LATTICE_CONTIGUOUS_BLAS_H_ */