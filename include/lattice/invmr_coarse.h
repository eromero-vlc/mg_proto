/*! \file
 *  \brief Minimal-Residual (MR) for a generic fermion Linear Operator
 */
#ifndef INCLUDE_LATTICE_INVMR_COARSE_H_
#define INCLUDE_LATTICE_INVMR_COARSE_H_


#include "lattice/constants.h"
#include "lattice/linear_operator.h"
#include "lattice/solver.h"
#include "lattice/mr_params.h"
#include "lattice/coarse/coarse_types.h"



namespace MG  {


  class MRSolverCoarse : public LinearSolver<CoarseSpinor,CoarseGauge> {
  public:
	  MRSolverCoarse(const LinearOperator<CoarseSpinor,CoarseGauge>& M,
			  	     const MG::LinearSolverParamsBase& params);

	  LinearSolverResults operator()(CoarseSpinor& out,
			  	  	  	  	  	  	 const CoarseSpinor& in,
									 ResiduumType resid_type = RELATIVE) const;

  private:
	  const LinearOperator<CoarseSpinor,CoarseGauge>& _M;
	  const MRSolverParams& _params;

  };


  class MRSmootherCoarse : public Smoother<CoarseSpinor,CoarseGauge> {
  public:
	  MRSmootherCoarse(const LinearOperator<CoarseSpinor,CoarseGauge>& M,
			  	  	   const MG::LinearSolverParamsBase& params);


	  void operator()(CoarseSpinor& out, const CoarseSpinor& in) const;

  private:
	  const LinearOperator<CoarseSpinor,CoarseGauge>& _M;
	  const MRSolverParams& _params;

  };
}

#endif /* TEST_QDPXX_INVMR_COARSE_H_ */
