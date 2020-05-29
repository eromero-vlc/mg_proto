
#include <omp.h>
#include <cstdio>
#include <iostream>
#include <unistd.h>

#include "lattice/coarse/coarse_op.h"
#include "lattice/cmat_mult.h"
#include "utils/memory.h"
#include "utils/print_utils.h"
#include "MG_config.h"
#include <complex>
#include "lattice/geometry_utils.h"

#ifdef _OPENMP
#include <omp.h>
#endif


namespace MG {

	namespace {

		template<unsigned int M>
			static inline void copy_matrix_mu(const float* __restrict__ x, unsigned int n, float* __restrict__ y) {
				for (unsigned int i = 0; i < n; ++i) {
					const float* __restrict__ x0 = &x[i*M];
					float* __restrict__ y0 = &y[i*M*8];
#pragma omp simd aligned(x0:8*4) aligned(y0:8*4*8)
					for (unsigned int j = 0; j < M; ++j)
						y0[j] = x0[j];
				}
			}

		static inline void copy_matrix_mu(const float* x, unsigned int m, unsigned int n, float* y) {
			if      (m == 12) copy_matrix_mu<12>(x, n, y);
			else if (m == 24) copy_matrix_mu<24>(x, n, y);
			else if (m == 48) copy_matrix_mu<48>(x, n, y);
			else if (m == 32) copy_matrix_mu<32>(x, n, y);
			else if (m == 64) copy_matrix_mu<64>(x, n, y);
			else if (m == 96) copy_matrix_mu<96>(x, n, y);
			else if (m == 128) copy_matrix_mu<128>(x, n, y);
			else assert(false);
		}

		enum InitOp { zero, add };

		typedef std::array<const float*,8> Neigh_spinors;
		typedef std::array<const float*,8> Gauge_links;

		Neigh_spinors get_neigh_spinors(const HaloContainer<CoarseSpinor>& halo, const CoarseSpinor& in, int target_cb, int cbsite) {
			return GetNeighborDirs<CoarseSpinor,CoarseAccessor>(halo, in, target_cb, cbsite, true);
		}

		Gauge_links get_gauge_links(const CoarseGauge& in, int target_cb, int cbsite) {
			const float* gauge_base = in.GetSiteDirDataPtr(target_cb,cbsite,0, true);
			const IndexType gdir_offset = in.GetLinkOffset();
			return Gauge_links({{
					gauge_base,                    // X forward
					gauge_base+gdir_offset,        // X backward
					gauge_base+2*gdir_offset,      // Y forward
					gauge_base+3*gdir_offset,      // Y backward
					gauge_base+4*gdir_offset,      // Z forward
					gauge_base+5*gdir_offset,      // Z backward
					gauge_base+6*gdir_offset,      // T forward
					gauge_base+7*gdir_offset }});       // T backward
		}

		Gauge_links get_gauge_ad_links(const CoarseGauge& in, int target_cb, int cbsite, int dagger) {
			const float* gauge_base = ((dagger == LINOP_OP) ?
					in.GetSiteDirADDataPtr(target_cb,cbsite,0, true)
					: in.GetSiteDirDADataPtr(target_cb,cbsite,0, true));
			const IndexType gdir_offset = in.GetLinkOffset();
			return Gauge_links({{
					gauge_base,                    // X forward
					gauge_base+gdir_offset,        // X backward
					gauge_base+2*gdir_offset,      // Y forward
					gauge_base+3*gdir_offset,      // Y backward
					gauge_base+4*gdir_offset,      // Z forward
					gauge_base+5*gdir_offset,      // Z backward
					gauge_base+6*gdir_offset,      // T forward
					gauge_base+7*gdir_offset }});       // T backward
		}

		void genericSiteOffDiagXPayz(int N_colorspin,
				InitOp initop,
				float *output,
				const float alpha,
				const Gauge_links& gauge_links,
				IndexType dagger, 
				const float* spinor_cb,
				const Neigh_spinors& neigh_spinors,
				IndexType ncol, float *input=nullptr)
		{
			// This is the same as for the dagger because we have G_5 I G_5 = G_5 G_5 I = I
			// D is the diagonal
			if (initop == add) {
				for(int i=0; i < 2*N_colorspin*ncol; ++i) {
					output[i] = spinor_cb[i];
				}
			}

			if (dagger == LINOP_OP) {
				if (input == nullptr) {
					// Old fashion: output = \sum_mu gauge_links[mu] * neigh_spinors[mu]
					for(int mu=0; mu < 8; ++mu)
						CMatMultCoeffAddNaive(initop == zero && mu == 0 ? 0.0 : 1.0, output, alpha, gauge_links[mu], neigh_spinors[mu], N_colorspin, ncol);
				} else {
					// New fashion: output = [ gauge_links[0] gauge_links[1] ...] * [ neigh_spinors[0]; neigh_spinors[1]; ...]
					if (initop == zero) {
#pragma omp simd aligned(output:8*4)
						for(int i=0; i < 2*N_colorspin*ncol; ++i) {
							output[i] = 0.0;
						}
					}

					// Copy all neigh_spinors into the single matrix 'input'
					for(int mu=0; mu < 8; ++mu)
						copy_matrix_mu(neigh_spinors[mu], 2*N_colorspin, ncol, &input[2*N_colorspin*mu]);

					// Call a single GEMM
					const std::complex<float>* Ac = reinterpret_cast<const std::complex<float>*>(gauge_links[0]);
					const std::complex<float>* inputc = reinterpret_cast<const std::complex<float>*>(input);
					std::complex<float>* outputc = reinterpret_cast<std::complex<float>*>(output);
					XGEMM("N", "N", N_colorspin, ncol, N_colorspin*8, alpha, Ac, N_colorspin, inputc, N_colorspin*8, initop == zero ? 0.0 : 1.0, outputc, N_colorspin);
				}
			} else {
				for(int mu=0; mu < 8; ++mu)
					GcCMatMultGcCoeffAddNaive(initop == zero && mu == 0 ? 0.0 : 1.0, output, alpha, gauge_links[mu], neigh_spinors[mu], N_colorspin, ncol);
				assert(false);
			}
		}

		// Lost site apply clover...
		void siteApplyClover(int N_colorspin,
				float* output,
				const float* clover,
				const float* input,
				const IndexType dagger,
				IndexType ncol)
		{
			// CMatMult-s.
			if( dagger == LINOP_OP) {
				CMatMultNaive(output, clover, input, N_colorspin, ncol);
			}
			else {
				// Slow: CMatAdjMultNaive(output, clover, input, N_colorspin);

				// Use Cc Hermiticity for faster operation
				GcCMatMultGcNaive(output,clover,input, N_colorspin, ncol);
			}
		}

	}


	void CoarseDiracOp::unprecOp(CoarseSpinor& spinor_out,
			const CoarseGauge& gauge_clov_in,
			const CoarseSpinor& spinor_in,
			const IndexType target_cb,
			const IndexType dagger,
			const IndexType tid) const
	{
		// 	Synchronous for now -- maybe change to comms compute overlap later
		// We are in an OMP region.
		CommunicateHaloSyncInOMPParallel<CoarseSpinor,CoarseAccessor>(_halo,spinor_in,target_cb);

		IndexType ncol = spinor_in.GetNCol();
		int blocksize, ld;
		GetWorkload(ncol, blocksize, ld);

		float *input = new float[GetNumColorSpin()*ncol*2 * 8];

		// Site is output site
		for(IndexType site0 = tid*blocksize, max_site = _lattice_info.GetNumCBSites(); site0 < max_site; site0 += ld) {
			for(int blki=0, site=site0; blki < blocksize && site < max_site; ++site, ++blki) {

				float* output = spinor_out.GetSiteDataPtr(0, target_cb, site, true);

				const float* spinor_cb = spinor_in.GetSiteDataPtr(0, target_cb,site, true);
				const float* clov = gauge_clov_in.GetSiteDiagDataPtr(target_cb,site, true);
				siteApplyClover(GetNumColorSpin(), output,clov,spinor_cb,dagger, ncol);

				const Gauge_links gauge_links = get_gauge_links(gauge_clov_in, target_cb, site);
				const Neigh_spinors neigh_spinors = get_neigh_spinors(_halo,spinor_in,target_cb,site);
				genericSiteOffDiagXPayz(GetNumColorSpin(), InitOp::add, output, 1.0, gauge_links, dagger, output, neigh_spinors, ncol, input);
			}
		}

		delete[] input;
	}



	void CoarseDiracOp::M_diag(CoarseSpinor& spinor_out,
			const CoarseGauge& gauge_clov_in,
			const CoarseSpinor& spinor_in,
			const IndexType target_cb,
			const IndexType dagger,
			const IndexType tid) const
	{
		IndexType ncol = spinor_in.GetNCol();
		int min_site, max_site;
		GetWorkloadForDiag(tid, min_site, max_site);


		// Site is output site
		for(IndexType site=min_site; site < max_site;++site) {

			float* output = spinor_out.GetSiteDataPtr(0, target_cb, site, true);
			const float* clover = gauge_clov_in.GetSiteDiagDataPtr(target_cb,site, true);
			const float* input = spinor_in.GetSiteDataPtr(0, target_cb,site, true);

			siteApplyClover(GetNumColorSpin(), output, clover, input, dagger, ncol);
		}

	}

	void CoarseDiracOp::M_diagInv(CoarseSpinor& spinor_out,
			const CoarseGauge& gauge_clov_in,
			const CoarseSpinor& spinor_in,
			const IndexType target_cb,
			const IndexType dagger,
			const IndexType tid) const
	{
		IndexType ncol = spinor_in.GetNCol();
		int min_site, max_site;
		GetWorkloadForDiag(tid, min_site, max_site);

		// Site is output site
		for(IndexType site=min_site; site < max_site;++site) {

			float* output = spinor_out.GetSiteDataPtr(0, target_cb, site, true);
			const float* clover = gauge_clov_in.GetSiteInvDiagDataPtr(target_cb,site, true);
			const float* input = spinor_in.GetSiteDataPtr(0, target_cb,site, true);

			siteApplyClover(GetNumColorSpin(), output, clover, input, dagger, ncol);
		}

	}


	void CoarseDiracOp::M_D_xpay(CoarseSpinor& spinor_out,
			const float alpha,
			const CoarseGauge& gauge_clov_in,
			const CoarseSpinor& spinor_in,
			const IndexType target_cb,
			const IndexType dagger,
			const IndexType tid) const
	{
		// 	Synchronous for now -- maybe change to comms compute overlap later
		CommunicateHaloSyncInOMPParallel<CoarseSpinor,CoarseAccessor>(_halo,spinor_in,target_cb);

		IndexType ncol = spinor_in.GetNCol();
		int blocksize, ld;
		GetWorkload(ncol, blocksize, ld);

		float *input = new float[GetNumColorSpin()*ncol*2 * 8];

		// Site is output site
		for(IndexType site0 = tid*blocksize, max_site = _lattice_info.GetNumCBSites(); site0 < max_site; site0 += ld) {
			for(int blki=0, site=site0; blki < blocksize && site < max_site; ++site, ++blki) {

				float* output = spinor_out.GetSiteDataPtr(0, target_cb, site, true);

				const Gauge_links gauge_links = get_gauge_links(gauge_clov_in, target_cb, site);
				const Neigh_spinors neigh_spinors = get_neigh_spinors(_halo,spinor_in,target_cb,site);
				genericSiteOffDiagXPayz(GetNumColorSpin(), InitOp::add, output, alpha, gauge_links, dagger, output, neigh_spinors, ncol, input);
			}
		}

		delete[] input;
	}

	void CoarseDiracOp::M_AD_xpayz(CoarseSpinor& spinor_out,
			const float alpha,
			const CoarseGauge& gauge_in,
			const CoarseSpinor& spinor_in_cb,
			const CoarseSpinor& spinor_in_od,
			const IndexType target_cb,
			const IndexType dagger,
			const IndexType tid) const
	{
		// 	Synchronous for now -- maybe change to comms compute overlap later
		CommunicateHaloSyncInOMPParallel<CoarseSpinor,CoarseAccessor>(_halo,spinor_in_od,target_cb);

		IndexType ncol = spinor_in_cb.GetNCol();
		int blocksize, ld;
		GetWorkload(ncol, blocksize, ld);

		float *input = new float[GetNumColorSpin()*ncol*2 * 8];

		// Site is output site
		for(IndexType site0 = tid*blocksize, max_site = _lattice_info.GetNumCBSites(); site0 < max_site; site0 += ld) {
			for(int blki=0, site=site0; blki < blocksize && site < max_site; ++site, ++blki) {

				float* output = spinor_out.GetSiteDataPtr(0, target_cb, site, true);
				const float* spinor_cb = spinor_in_cb.GetSiteDataPtr(0, target_cb,site, true);
				const Gauge_links gauge_links = get_gauge_ad_links(gauge_in, target_cb, site, dagger);
				const Neigh_spinors neigh_spinors = get_neigh_spinors(_halo,spinor_in_od,target_cb,site);
				genericSiteOffDiagXPayz(GetNumColorSpin(), InitOp::add, output, alpha, gauge_links, dagger, spinor_cb, neigh_spinors, ncol, input);
			}
		}

		delete[] input;
	}

	void CoarseDiracOp::M_DA_xpayz(CoarseSpinor& spinor_out,
			const float alpha,
			const CoarseGauge& gauge_clov_in,
			const CoarseSpinor& spinor_cb,
			const CoarseSpinor& spinor_in,
			const IndexType target_cb,
			const IndexType dagger,
			const IndexType tid) const
	{
		// 	Synchronous for now -- maybe change to comms compute overlap later
		CommunicateHaloSyncInOMPParallel<CoarseSpinor,CoarseAccessor>(_halo,spinor_in,target_cb);

		IndexType ncol = spinor_in.GetNCol();
		int blocksize, ld;
		GetWorkload(ncol, blocksize, ld);

		float *input = new float[GetNumColorSpin()*ncol*2 * 8];

		// Site is output site
		for(IndexType site0 = tid*blocksize, max_site = _lattice_info.GetNumCBSites(); site0 < max_site; site0 += ld) {
			for(int blki=0, site=site0; blki < blocksize && site < max_site; ++site, ++blki) {

				float* output = spinor_out.GetSiteDataPtr(0, target_cb, site, true);
				const Gauge_links gauge_links = get_gauge_ad_links(gauge_clov_in, target_cb, site, dagger == LINOP_OP ? LINOP_DAGGER : LINOP_OP);
				const Neigh_spinors neigh_spinors = get_neigh_spinors(_halo,spinor_in,target_cb,site);
				const float* in_cb = spinor_cb.GetSiteDataPtr(0, target_cb,site, true);
				genericSiteOffDiagXPayz(GetNumColorSpin(), InitOp::add, output, alpha, gauge_links, dagger, in_cb, neigh_spinors, ncol, input);
			}
		}

		delete[] input;
	}


	void CoarseDiracOp::M_AD(CoarseSpinor& spinor_out,
			const CoarseGauge& gauge_clov_in,
			const CoarseSpinor& spinor_in,
			const IndexType target_cb,
			const IndexType dagger,
			const IndexType tid) const
	{
		// 	Synchronous for now -- maybe change to comms compute overlap later
		CommunicateHaloSyncInOMPParallel<CoarseSpinor,CoarseAccessor>(_halo,spinor_in,target_cb);

		IndexType ncol = spinor_in.GetNCol();
		int blocksize, ld;
		GetWorkload(ncol, blocksize, ld);

		float *input = new float[GetNumColorSpin()*ncol*2 * 8];

		// Site is output site
		for(IndexType site0 = tid*blocksize, max_site = _lattice_info.GetNumCBSites(); site0 < max_site; site0 += ld) {
			for(int blki=0, site=site0; blki < blocksize && site < max_site; ++site, ++blki) {

				const Gauge_links gauge_links = get_gauge_ad_links(gauge_clov_in, target_cb, site, dagger);
				const Neigh_spinors neigh_spinors = get_neigh_spinors(_halo,spinor_in,target_cb,site);
				float* output = spinor_out.GetSiteDataPtr(0, target_cb, site, true);
				genericSiteOffDiagXPayz(GetNumColorSpin(), InitOp::zero, output, 1.0, gauge_links, dagger, output, neigh_spinors, ncol, input);
			}
		}

		delete[] input;

	}


	void CoarseDiracOp::M_DA(CoarseSpinor& spinor_out,
			const CoarseGauge& gauge_clov_in,
			const CoarseSpinor& spinor_in,
			const IndexType target_cb,
			const IndexType dagger,
			const IndexType tid) const
	{
		// 	Synchronous for now -- maybe change to comms compute overlap later
		CommunicateHaloSyncInOMPParallel<CoarseSpinor,CoarseAccessor>(_halo,spinor_in,target_cb);

		IndexType ncol = spinor_in.GetNCol();
		int blocksize, ld;
		GetWorkload(ncol, blocksize, ld);

		float *input = new float[GetNumColorSpin()*ncol*2 * 8];

		// Site is output site
		for(IndexType site0 = tid*blocksize, max_site = _lattice_info.GetNumCBSites(); site0 < max_site; site0 += ld) {
			for(int blki=0, site=site0; blki < blocksize && site < max_site; ++site, ++blki) {

				const Gauge_links gauge_links = get_gauge_ad_links(gauge_clov_in, target_cb, site, dagger == LINOP_OP ? LINOP_DAGGER : LINOP_OP);
				const Neigh_spinors neigh_spinors = get_neigh_spinors(_halo,spinor_in,target_cb,site);
				float* output = spinor_out.GetSiteDataPtr(0, target_cb, site, true);
				genericSiteOffDiagXPayz(GetNumColorSpin(), InitOp::zero, output, 1.0, gauge_links, dagger, output, neigh_spinors, ncol, input);
			}
		}

		delete[] input;
	}



	// Apply a single direction of Dslash -- used for coarsening
	void CoarseDiracOp::DslashDir(CoarseSpinor& spinor_out,
			const CoarseGauge& gauge_in,
			const CoarseSpinor& spinor_in,
			const IndexType target_cb,
			const IndexType dir,
			const IndexType tid) const
	{

		// This needs to be figured out.

		int min_site, max_site;
		GetWorkloadForDiag(tid, min_site, max_site);
		const int N_colorspin = GetNumColorSpin();

		// The opposite direction
		int opp_dir = dir/2*2 + 1-dir%2;
		if( ! _halo.LocalDir(dir/2) ) {
			// Prepost receive
#pragma omp master
			{
				// Start receiving from this direction
				_halo.StartRecvFromDir(dir);
			}
			// No need for barrier here
			// Pack the opposite direction
			packFace<CoarseSpinor,CoarseAccessor>(_halo,spinor_in,1-target_cb,opp_dir);

			/// Need barrier to make sure all threads finished packing
#pragma omp barrier

			// Master calls MPI stuff
#pragma omp master
			{
				// Send the opposite direction
				_halo.StartSendToDir(opp_dir);
				_halo.FinishSendToDir(opp_dir);

				// Finish receiving from this direction
				_halo.FinishRecvFromDir(dir);
			}
			// Threads oughtn't start until finish is complete
#pragma omp barrier
		}



		// Site is output site
		for(IndexType site=min_site; site < max_site;++site) {


			float* output = spinor_out.GetSiteDataPtr(0, target_cb, site, true);
			const float* gauge_link_dir = gauge_in.GetSiteDirDataPtr(target_cb,site,dir, true);

			/* The following case statement selects neighbors.
			 *  It is culled from the full Dslash
			 *  It of course would get complicated if some of the neighbors were in a halo
			 */

			const float *neigh_spinor = GetNeighborDir<CoarseSpinor,CoarseAccessor>(_halo, spinor_in, dir, target_cb, site, true);

			// Multiply the link with the neighbor. EasyPeasy?
			CMatMultNaive(output, gauge_link_dir, neigh_spinor, N_colorspin, spinor_in.GetNCol());
		} // Loop over sites
	}




	CoarseDiracOp::CoarseDiracOp(const LatticeInfo& l_info)
		: _lattice_info(l_info),
		_n_colorspin(l_info.GetNumColorSpins()),
		_halo( l_info )
	{
#pragma omp parallel
#pragma omp master
		{
			// Print workloads
			for (int ncol=1; ncol < 256; ncol*=4) {
				int blocksize, ld;
				GetWorkload(ncol, blocksize, ld);
				MasterLog(INFO, "Lattice with %d spin-colors with %d columns: blocksize %d  and ld %d", _lattice_info.GetNumColorSpins(), ncol, blocksize, ld);
			}
		}
	}

	void CoarseDiracOp::GetWorkloadForDiag(int tid, int& min_site, int& max_site) const {
		// Get how many threads are going to work
#ifdef _OPENMP
		const std::size_t nthreads = omp_get_num_threads();
#else
		const std::size_t nthreads = 1;
#endif

		const std::size_t n_cbsites = _lattice_info.GetNumCBSites();
		min_site = n_cbsites / nthreads * tid + std::min(n_cbsites % nthreads, (std::size_t)tid);
		max_site = n_cbsites / nthreads * (tid+1) + std::min(n_cbsites % nthreads, (std::size_t)tid+1);
	}

	void CoarseDiracOp::GetWorkload(int ncols, int& blocksize, int& ld) const {
		// Get how many threads are going to work
#ifdef _OPENMP
		const std::size_t nthreads = omp_get_num_threads();
#else
		const std::size_t nthreads = 1;
#endif

#ifdef MG_HACK_HILBERT_CURVE
		// Get how much cache they can gather
		int cache_size = sysconf(_SC_LEVEL2_CACHE_SIZE)*nthreads;

		// Approximate how many nodes they can be hold on cache
		const int cs = _lattice_info.GetNumColorSpins(), n_neigbors = 2*n_dim, n_faces = 2*n_dim, excess = 2;
		const std::size_t vol = _lattice_info.GetNumSites();
		std::size_t max_lat_size = 1;
		while (true) {
			std::size_t l = 2*max_lat_size;
			if (l*l*l*l > vol || (1.0*l*l*l*l*(cs*cs*(n_neigbors+1)/2 + cs*ncols) + (n_faces - 1)*cs*ncols/2*l*l*l)*sizeof(float)*2 > cache_size * excess)
				break;
			max_lat_size = l;
		}
		std::size_t cache_vol = std::max((std::size_t)1, std::min(vol/2, max_lat_size*max_lat_size*max_lat_size*max_lat_size/2));
#else
		std::size_t cache_vol = _lattice_info.GetNumSites()/2;
#endif
		blocksize = (cache_vol + nthreads - 1) / nthreads;
		ld = blocksize * nthreads;
	}


#ifdef MG_WRITE_COARSE
#include <mpi.h>

	void CoarseDiracOp::write(const CoarseGauge& gauge, std::string& filename)
	{
		IndexType n_colorspin = gauge.GetNumColorSpin();
		IndexArray lattice_dims;
		gauge.GetInfo().LocalDimsToGlobalDims(lattice_dims, gauge.GetInfo().GetLatticeDimensions());
		IndexType nxh = gauge.GetNxh();
		IndexType nx = gauge.GetNx();
		IndexType ny = gauge.GetNy();
		IndexType nz = gauge.GetNz();
		IndexType nt = gauge.GetNt();
		unsigned long num_sites = gauge.GetInfo().GetNumSites(), offset;
		MPI_Scan(&num_sites, &offset, 1, MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);
		offset -= num_sites;
		MPI_File fh;
		MPI_File_delete(filename.c_str(), MPI_INFO_NULL);
		MPI_File_open(MPI_COMM_WORLD, filename.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
		if (offset == 0) {
			float header[6] = {4, (float)lattice_dims[0], (float)lattice_dims[1], (float)lattice_dims[2], (float)lattice_dims[3], (float)n_colorspin};
			MPI_Status status;
			MPI_File_write(fh, header, n_dim+2, MPI_FLOAT, &status);
		}
		MPI_File_set_view(fh, sizeof(float)*(n_dim+2 + (n_complex*n_colorspin*n_colorspin + n_dim*2)*(2*n_dim+1)*offset), MPI_FLOAT, MPI_FLOAT, "native", MPI_INFO_NULL);

		// Site is output site
		const int n_sites_cb = gauge.GetInfo().GetNumCBSites();
		for(IndexType site_cb=0; site_cb < n_sites_cb;++site_cb) {
			for(int target_cb=0; target_cb < 2;++target_cb) {

				const float* gauge_base = gauge.GetSiteDirDataPtr(target_cb,site_cb,0);
				const IndexType gdir_offset = gauge.GetLinkOffset();
				const float* clov = gauge.GetSiteDiagDataPtr(target_cb,site_cb);

				const float *gauge_links[9]={
					clov,                          // Diag
					gauge_base,                    // X forward
					gauge_base+gdir_offset,        // X backward
					gauge_base+2*gdir_offset,      // Y forward
					gauge_base+3*gdir_offset,      // Y backward
					gauge_base+4*gdir_offset,      // Z forward
					gauge_base+5*gdir_offset,      // Z backward
					gauge_base+6*gdir_offset,      // T forward
					gauge_base+7*gdir_offset };    // T backward

				// Turn site into x,y,z,t coords
				IndexArray local_site_coor;
				CBIndexToCoords(site_cb, target_cb, gauge.GetInfo().GetLatticeDimensions(), gauge.GetInfo().GetLatticeOrigin(), local_site_coor);

				// Compute global coordinate
				IndexArray global_site_coor;
				gauge.GetInfo().LocalCoordToGlobalCoord(global_site_coor, local_site_coor);

				// Compute neighbors	
				IndexArray coors[9];
				coors[0] = global_site_coor;
				for(int i=0,j=1; i<4; i++) {
					// Forward
					global_site_coor[i] = (global_site_coor[i] + 1) % lattice_dims[i];
					coors[j++] = global_site_coor;

					// Backward
					global_site_coor[i] = (global_site_coor[i] + lattice_dims[i] - 2) % lattice_dims[i];
					coors[j++] = global_site_coor;

					// Restore
					global_site_coor[i] = (global_site_coor[i] + 1) % lattice_dims[i];
				}

				for (int i=0; i<9; i++) {
					float coords[8] = {(float)coors[0][0], (float)coors[0][1], (float)coors[0][2], (float)coors[0][3], (float)coors[i][0], (float)coors[i][1], (float)coors[i][2], (float)coors[i][3]};
					MPI_Status status;
					MPI_File_write(fh, coords, 8, MPI_FLOAT, &status);
					MPI_File_write(fh, gauge_links[i], n_complex*n_colorspin*n_colorspin, MPI_FLOAT, &status);
				}
			}
		}

		MPI_File_close(&fh);
	}
#else
	void CoarseDiracOp::write(const CoarseGauge& gauge, std::string& filename)
	{
		(void)gauge;
		(void)filename;
	}
#endif // MG_WRITE

} // Namespace

