
#ifndef INCLUDE_LATTICE_COARSE_PRIMME_H_
#define INCLUDE_LATTICE_COARSE_PRIMME_H_

#include <complex>

#include "lattice/coarse/coarse_op.h"
#include "lattice/cmat_mult.h"
#include "utils/memory.h"
#include "utils/print_utils.h"
#include "MG_config.h"
#include "lattice/geometry_utils.h"

#ifdef MG_USE_PRIMME

#include "primme.h"

namespace MG {

	namespace {
		// Auxiliary function for global sum reduction
		void globalSumDouble(void *sendBuf, void *recvBuf, int *count, 
				primme_params *primme, int *ierr)
		{
			double *x = (double*)sendBuf, *y = (double*)recvBuf;
			if (x != y) {
				for (int i=0, count_=*count; i<count_; i++) y[i] = x[i];
			}
#ifdef MG_QMP_COMMS
			if (*count > 0) QMP_sum_double_array(y, *count);
#endif
			*ierr = 0;
		}

		// Auxiliary function for the matvec
		template<typename LinOpT> struct PrimmeMatrixMatvec {
			static void fun(void *x, PRIMME_INT *ldx, void *y, PRIMME_INT *ldy,
					int *blockSize, primme_params *primme, int *ierr)
			{
				// If this routine exits before reaching the end, notify it as an error by default
				*ierr = -1;

				// Quick return
				if (*blockSize <= 0) {*ierr = 0; return; }

				LinOpT *M = (LinOpT*)primme->matrix;

				std::shared_ptr<CoarseSpinor> xs = M->tmp(M->GetInfo(), *blockSize);
				std::shared_ptr<CoarseSpinor> ys = M->tmp(M->GetInfo(), *blockSize);

				PutColumns((const float*)x, *ldx*2, *xs, M->GetSubset());
				M->operatorForDeflation(*ys, *xs);
				Gamma5Vec(*ys);
				GetColumns(*ys, M->GetSubset(), (float*)y, *ldy*2);

				// We're good!
				*ierr = 0;
			}
		};


	}

  	template<typename LinOpT>
	void computeDeflation(CoarseSpinor &defl, const LinOpT& M)
	{
    		const LatticeInfo& info = defl.GetInfo();
		AssertCompatible(info, M.GetInfo());
		IndexType num_cbsites = info.GetNumCBSites();
		IndexType num_colorspin = defl.GetNumColorSpin();
		const CBSubset& cbsubset = M.GetSubset();
		size_t nLocal = (size_t)defl.GetNumColorSpin()*info.GetNumCBSites()*(cbsubset.end - cbsubset.start);
		IndexType nEv = defl.GetNCol();

		// Initialize PRIMME configuration
		primme_params primme;
		primme_initialize(&primme);

		// Set global sum reduction
		primme.numProcs = info.GetNodeInfo().NumNodes();
		primme.procID = info.GetNodeInfo().NodeID();
		primme.globalSumReal = globalSumDouble;
		primme.globalSumReal_type = primme_op_double;

		// Determine local and global matrix dimension
		double n = nLocal; QMP_sum_double_array(&n, 1);
		primme.n = n;             /* set global problem dimension */
		primme.nLocal = nLocal;   /* set local problem dimension */

		primme.numEvals = nEv;   /* Number of wanted eigenpairs */
		primme.eps = 1e-2;      /* ||r|| <= eps * ||matrix|| */
		//primme.maxBasisSize = eig_param->nKr - eig_param->nEv;

		// Create evals. For twisted operators, gamma * operator is not Hermitian, but normal.
		float *evals = new float[nEv];

		// Create residual norms
		float *rnorms = new float[nEv];

		// Create eigenvectors
		// NOTE: BLAS/LAPACK obsession with the Fortran's way of passing arrays is contagious, and eigenvectors passed to
		// PRIMME, library which relays on BLAS/LAPACK heavily, follow the same convention.
		std::complex<float> *evecs = new std::complex<float>[nLocal * nEv];

		// Seek for the largest eigenvalue in magnitude for using the inverse operator
		primme.target = primme_largest_abs;
		double zero = 0;
		primme.targetShifts = &zero;
		primme.numTargetShifts = 1;

		// Set operator
		primme.matrixMatvec = PrimmeMatrixMatvec<LinOpT>::fun;
		primme.matrix = (void*)&M;

		// primme.locking = 0;
		// primme.minRestartSize = primme.numEvals + 32;
		// primme.maxBasisSize = primme.numEvals + 64;

		// Set advanced options. If the operator is an inverter, configure PRIMME to minimize the
		// inverter applications. Otherwise use an strategy that minimizes orthogonalization time.
		primme_set_method(PRIMME_DEFAULT_MIN_MATVECS, &primme);

		// Display PRIMME configuration struct (optional)
		if (primme.procID == 0)
			primme_display_params(primme);

		primme.printLevel = 3;

		// Call primme
		int ret = cprimme(evals, evecs, rnorms, &primme);

		if (1) {
			MasterLog(INFO, "Time to solve problem using PRIMME  = %e\n", primme.stats.elapsedTime);
			MasterLog(INFO, "Time spent in matVec                = %e  %.1f%%\n", primme.stats.timeMatvec,
					100 * primme.stats.timeMatvec / primme.stats.elapsedTime);
			MasterLog(INFO, "Time spent in orthogonalization     = %e  %.1f%% (%.1f GFLOPS)\n", primme.stats.timeOrtho,
					100 * primme.stats.timeOrtho / primme.stats.elapsedTime,
					primme.stats.numOrthoInnerProds * primme.n / primme.stats.timeOrtho / 1e9);
			MasterLog(INFO, "Time spent in dense operations      = %e  %.1f%% (%.1f GFLOPS)\n", primme.stats.timeDense,
					100 * primme.stats.timeDense / primme.stats.elapsedTime, primme.stats.flopsDense / primme.stats.timeDense / 1e9);
			double timeComm = primme.stats.timeGlobalSum + primme.stats.timeBroadcast;
			MasterLog(INFO, "Time spent in communications        = %e  %.1f%%\n", timeComm,
					100 * timeComm / primme.stats.elapsedTime);
		}

		// Copy evecs to defl
		PutColumns((const float*)evecs, nLocal*2, defl, M.GetSubset());

		// Local clean-up
		delete [] rnorms;
		delete [] evals;
		delete evecs;
		primme_free(&primme);

	}
}

#else

namespace MG {
  	template<typename LinOpT>
	void computeDeflation(CoarseSpinor &defl, const LinOpT& M) {}
}

#endif // MG_USE_PRIMME


#endif // INCLUDE_LATTICE_COARSE_PRIMME_H_
