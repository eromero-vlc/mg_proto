/*
 * * invmr_coarse.cpp
 *
 *  Created on: Mar 21, 2017
 *      Author: bjoo
 */

#include <lattice/coarse/invmr_coarse.h>
#include "lattice/constants.h"
#include "lattice/linear_operator.h"
#include "lattice/solver.h"
#include "lattice/mr_params.h"
#include "lattice/coarse/coarse_types.h"
#include "lattice/coarse/coarse_l1_blas.h"

#include "utils/print_utils.h"

#include <complex>
#include <algorithm>

namespace MG
{

namespace {
	/** all_false
	 *
	 * 	Returns true if all elements are false
	 */
	bool all_false(const std::vector<bool>& x) {
		return std::all_of(x.begin(), x.end(), [](bool b){return !b;});
	}

	/** negate
	 *
	 * 	Flip sign on all elements
	 */
	template <typename T>
	std::vector<std::complex<float>> negate(const std::vector<std::complex<T>>& x) {
		std::vector<std::complex<float>> r(x.size());
		std::transform(x.begin(), x.end(), r.begin(), [](const std::complex<T>& f) { return -f;});
		return r;
	}
}


//! Minimal-residual (MR) algorithm for a generic Linear Operator
/*! \ingroup invert
 * This subroutine uses the Minimal Residual (MR) algorithm to determine
 * the solution of the set of linear equations. Here we allow M to be nonhermitian.
 *
 *   	    Chi  =  M . Psi
 *
 * Algorithm:
 *
 *  Psi[0]                                      Argument
 *  r[0]    :=  Chi  -  M . Psi[0] ;            Initial residual
 *  IF |r[0]| <= RsdCG |Chi| THEN RETURN;       Converged?
 *  FOR k FROM 1 TO MaxCG DO                    MR iterations
 *      a[k-1]  := <M.r[k-1],r[k-1]> / <M.r[k-1],M.r[k-1]> ;
 *      ap[k-1] := MRovpar * a[k] ;             Overrelaxtion step
 *      Psi[k]  += ap[k-1] r[k-1] ;   	        New solution std::vector
 *      r[k]    -= ap[k-1] A . r[k-1] ;         New residual
 *      IF |r[k]| <= RsdCG |Chi| THEN RETURN;   Converged?

 * Arguments:

 *  \param M       Linear Operator             (Read)
 *  \param chi     Source                      (Read)
 *  \param psi     Solution                    (Modify)
 *  \param RsdCG   MR residual accuracy        (Read)
 *  \param MRovpar Overrelaxation parameter    (Read)
 *  \param MaxMR   Maximum MR iterations       (Read)

 * Local Variables:

 *  r   	Residual std::vector
 *  cp  	| r[k] |**2
 *  c   	| r[k-1] |**2
 *  k   	MR iteration counter
 *  a   	a[k]
 *  d   	< M.r[k], M.r[k] >
 *  R_Aux     Temporary for  M.Psi
 *  Mr        Temporary for  M.r

 * Global Variables:

 *  MaxMR       Maximum number of MR iterations allowed
 *  RsdCG       Maximum acceptable MR residual (relative to source)
 *
 * Subroutines:
 *
 *  M           Apply matrix to std::vector
 *
 * @{
 */


std::vector<LinearSolverResults>
InvMR_T(const LinearOperator<CoarseSpinor,CoarseGauge>& M,
		const CoarseSpinor& chi,
		CoarseSpinor& psi,
		const double& OmegaRelax,
		const double& RsdTarget,
		int MaxIter,
		IndexType OpType,
		ResiduumType resid_type,
		bool VerboseP,
		bool TerminateOnResidua)
{
	const int level = M.GetLevel();
	const CBSubset& subset = M.GetSubset();
	IndexType ncol = psi.GetNCol();


	const LatticeInfo& info = chi.GetInfo();
	{
		const LatticeInfo& M_info = M.GetInfo();
		AssertCompatible( M_info, info );
		const LatticeInfo& psi_info = psi.GetInfo();
		AssertCompatible( psi_info, info );
	}

	if( MaxIter < 0 ) {
		MasterLog(ERROR,"MR: level=%d Invalid Value: MaxIter < 0 ",level);
	}

	if ( MaxIter == 0 ) {
		LinearSolverResults res;
		// No work to do -- likely only happens in the case of a smoother
		res.resid_type=INVALID;
		res.n_count = 0;
		res.resid = -1;
		return std::vector<LinearSolverResults>(ncol, res);
	}

	std::vector<LinearSolverResults> res(ncol);

	for (int col=0; col < ncol; ++col) res[col].resid_type = resid_type;

	CoarseSpinor Mr(info, ncol);
	CoarseSpinor chi_internal(info, ncol);

	int k=0;


	// chi_internal[s] = chi;
	// ZeroVec(chi_internal);
	CopyVec(chi_internal, chi, subset);

	/*  r[0]  :=  Chi - M . Psi[0] */
	/*  r  :=  M . Psi  */
	M(Mr, psi, OpType);

	CoarseSpinor r(info, ncol);
	// r[s]= chi_internal - Mr;
	XmyzVec(chi_internal,Mr,r,subset);


	std::vector<double> norm_chi_internal;
	std::vector<double> rsd_sq(ncol, RsdTarget*RsdTarget);
	std::vector<double> cp;

	// TerminateOnResidua==true: if we met the residuum criterion we'd have terminated, safe to say no to terminate
	// TerminateOnResidua==false: We need to do at least 1 iteration (otherwise we'd have exited)
	std::vector<bool> continueP(ncol, true);

	if( TerminateOnResidua ) {
		norm_chi_internal = Norm2Vec(chi_internal, subset);

		if( resid_type == RELATIVE ) {
			for (int col=0; col < ncol; ++col) rsd_sq[col] *= norm_chi_internal[col];
		}

		/*  Cp = |r[0]|^2 */
		cp = Norm2Vec(r,subset);                 /* 2 Nc Ns  flops */

		if( VerboseP ) {
			for (int col=0; col < ncol; ++col)  {
				MasterLog(INFO, "MR: col=%d, level=%d iter=%d || r ||^2 = %16.8e  Target || r ||^2 = %16.8e",col,level,k,cp[col], rsd_sq[col]);
			}

		}

		/*  IF |r[0]| <= RsdMR |Chi| THEN RETURN; */
		for (int col=0; col < ncol; ++col)  {	
			if ( cp[col]  <=  rsd_sq[col] )
			{
				res[col].n_count = 0;
				res[col].resid   = sqrt(cp[col]);
				if( resid_type == ABSOLUTE ) {
					if( VerboseP ) {
						MasterLog(INFO, "MR Solver: col=%d level=%d Final iters=0 || r ||_accum=16.8e || r ||_actual = %16.8e",col,level,
								sqrt(cp[col]), res[col].resid);

					}
				}
				else {

					res[col].resid /= sqrt(norm_chi_internal[col]);
					if( VerboseP ) {
						MasterLog(INFO, "MR: col=%d level=%d Final iters=0 || r ||/|| b ||_accum=16.8e || r ||/|| b ||_actual = %16.8e",col,level,
								sqrt(cp[col]/norm_chi_internal[col]), res[col].resid);
					}
				}

				continueP[col] = false;
			}
		}
	}

	if (all_false(continueP)) return res;

	/* Main iteration loop */
	while( !all_false(continueP) && k < MaxIter)
	{
		++k;

		/*  a[k-1] := < M.r[k-1], r[k-1] >/ < M.r[k-1], M.r[k-1] > ; */
		/*  Mr = M * r  */
		M(Mr, r, OpType);
		/*  c = < M.r, r > */
		std::vector<std::complex<double>> c = InnerProductVec(Mr, r,subset);

		/*  d = | M.r | ** 2  */
		std::vector<double> d = Norm2Vec(Mr,subset);

		/*  a = c / d */
		std::vector<std::complex<float>> a(ncol);
		for (int col=0; col < ncol; ++col) a[col] = c[col] / d[col];

		/*  a[k-1] *= MRovpar ; */
		for (int col=0; col < ncol; ++col) a[col] = a[col] * std::complex<float>(OmegaRelax, 0);

		/*  Psi[k] += a[k-1] r[k-1] ; */
		//psi[s] += r * a;
		AxpyVec(a,r,psi,subset);

		/*  r[k] -= a[k-1] M . r[k-1] ; */
		// r[s] -= Mr * a;
		AxpyVec(negate(a),Mr,r,subset);

		if( TerminateOnResidua ) {

			/*  cp  =  | r[k] |**2 */
			cp = Norm2Vec(r,subset);
			if( VerboseP ) {
				for (int col=0; col < ncol; ++col)  {
					MasterLog(INFO, "MR: level=%d iter=%d col=%d || r ||^2 = %16.8e  Target || r^2 || = %16.8e", level,
							k, col, cp[col], rsd_sq[col] );
					if (continueP[col] && cp[col] <= rsd_sq[col]) {
						res[col].n_count = k;
						continueP[col] = false;
					}
				}
			}
		}
		else {
			if( VerboseP ) {
				MasterLog(INFO, "MR: level=%d iter=%d",level, k);
			}
		}

	}
	for (int col=0; col < ncol; ++col) {
		if (continueP[col]) res[col].n_count = k;
		res[col].resid = 0;
	}

	if( TerminateOnResidua) {
		// Compute the actual residual


		M(Mr, psi, OpType);
		//Double actual_res = norm2(chi_internal - Mr,s);
		std::vector<double> actual_res = XmyNorm2Vec(chi_internal,Mr,subset);
		for (int col=0; col < ncol; ++col) res[col].resid = sqrt(actual_res[col]);
		if( resid_type == ABSOLUTE ) {
			if( VerboseP ) {
				for (int col=0; col < ncol; ++col) 	 {
					MasterLog(INFO, "MR: level=%d col=%d Final iters=%d || r ||_accum=%16.8e || r ||_actual=%16.8e", level,
							col, res[col].n_count, sqrt(cp[col]), res[col].resid);
				}
			}
		}
		else {

			for (int col=0; col < ncol; ++col) res[col].resid /= sqrt(norm_chi_internal[col]);
			if( VerboseP ) {
				for (int col=0; col < ncol; ++col) {
					MasterLog(INFO, "MR: level=%d col=%d Final iters=%d || r ||_accum=%16.8e || r ||_actual=%16.8e", level,
							res[col].n_count, sqrt(cp[col]/norm_chi_internal[col]), res[col].resid);
				}
			}
		}
	}
	return res;
}




MRSolverCoarse::MRSolverCoarse(const LinearOperator<CoarseSpinor,CoarseGauge>& M,
		const MG::LinearSolverParamsBase& params) : _M(M),
				_params(static_cast<const MRSolverParams&>(params)){}

std::vector<LinearSolverResults>
MRSolverCoarse::operator()(CoarseSpinor& out,
		const CoarseSpinor& in,
		ResiduumType resid_type) const {
	return  InvMR_T(_M, in, out, _params.Omega, _params.RsdTarget,
			_params.MaxIter, LINOP_OP, resid_type, _params.VerboseP , true);

}


MRSmootherCoarse::MRSmootherCoarse(const LinearOperator<CoarseSpinor,CoarseGauge>& M,
		const MG::LinearSolverParamsBase& params) : _M(M), _params(static_cast<const MRSolverParams&>(params)) {}

MRSmootherCoarse::MRSmootherCoarse(const std::shared_ptr<const LinearOperator<CoarseSpinor,CoarseGauge>> M_ptr,
			  	  	   const MG::LinearSolverParamsBase& params) : _M(*M_ptr), _params(static_cast<const MRSolverParams&>(params)) {}

void
MRSmootherCoarse::operator()(CoarseSpinor& out, const CoarseSpinor& in) const {
	InvMR_T(_M, in, out, _params.Omega, _params.RsdTarget,
			_params.MaxIter, LINOP_OP,  RELATIVE, _params.VerboseP , _params.RsdTarget > 0.0 );
}

} // Namespace




