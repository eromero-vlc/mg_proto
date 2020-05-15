/*
 * coarse_types.h
 *
 *  Created on: Jan 21, 2016
 *      Author: bjoo
 */

#ifndef INCLUDE_LATTICE_COARSE_COARSE_TYPES_H_
#define INCLUDE_LATTICE_COARSE_COARSE_TYPES_H_

#include "lattice/constants.h"
#include "lattice/lattice_info.h"
#include "lattice/geometry_utils.h"
#include "utils/memory.h"
#include "utils/auxiliary.h"
#include "utils/print_utils.h"


using namespace MG;

namespace MG {

	/** Permutation over the checkerboard sites
	 *
	 *  p[cb][cbsite] returns the position of the site with CB index cbsite and
	 *  parity cb.
	 *
	 */

	using CBPermutation = std::shared_ptr< std::array<std::vector<IndexType>,4> >;

	/** Cache-efficient permutation over CB sites
	 *  \param LatticeInfo
	 *  \param ncol
	 *
	 *  Returns a permutation over the sites that increase the cache hits when
	 *  applying one of the CoarseDiracOp operations.
	 *
	 */
	CBPermutation cache_optimal_permutation(const LatticeInfo& info);

	/** Permute the rows of a matrix
	 *  \param v: matrix with p.size() rows and block_size columns made of floats
	 *  \param block_size: number of columns
	 *  \param p: permutation
	 *
	 *  Returns a new array that at row p[site] has the content of v[site,:].
	 *
	 */
	float* permute(const float* v, std::size_t block_size, const std::vector<IndexType>& p);

	/** Permute the rows of a matrix
	 *  \param v: matrix with p.size() rows and block_size columns made of floats
	 *  \param block_size: number of columns
	 *  \param p0: original permutation
	 *  \param p1: new permutation
	 *
	 *  Returns a new array that at row p1[site] has the content of v[p0[site],:].
	 *
	 */
	float* permute(const float* v, std::size_t block_size, const std::vector<IndexType>& p0, const std::vector<IndexType>& p1);


	/** Coarse Spinor
	 *  \param LatticeInfo
	 *
	 *  Basic Coarse Spinor. Holds memory for two checkerboards of sites
	 *  Regular site ordering: ie <cb><sites>< Nspin*Ncolor >< n_complex = fastest >
	 *
	 *
	 *  Destruction frees memory
	 *
	 */
	class CoarseSpinor : public AbstractSpinor<CoarseSpinor> {
	public:
		CoarseSpinor(const LatticeInfo& lattice_info, IndexType n_col=1) : _lattice_info(lattice_info), data{nullptr,nullptr},
				_n_color(lattice_info.GetNumColors()),
				_n_spin(lattice_info.GetNumSpins()),
				_n_colorspin(lattice_info.GetNumColors()*lattice_info.GetNumSpins()),
				_n_site_offset(n_complex*_n_colorspin*n_col),
				_n_xh( lattice_info.GetCBLatticeDimensions()[0] ),
				_n_x( lattice_info.GetLatticeDimensions()[0] ),
				_n_y( lattice_info.GetLatticeDimensions()[1] ),
				_n_z( lattice_info.GetLatticeDimensions()[2] ),
				_n_t( lattice_info.GetLatticeDimensions()[3] ),
				_n_col( n_col ),
				_n_col_offset(n_complex*_n_colorspin)//,
				//_perm(cache_optimal_permutation(lattice_info))
		{
#if 1
			// Check That we have 2 spins
			if( lattice_info.GetNumSpins() != 2 ) {
				MasterLog(ERROR, "Attempting to Create CoarseSpinor with num_spins != 2");
			}
#endif

			// Allocate Data
			IndexType num_floats_per_cb = _lattice_info.GetNumCBSites()*_n_site_offset;

			/* Non-Contiguout allocation */
			data[0] = (float *)MG::MemoryAllocate(num_floats_per_cb*sizeof(float), MG::REGULAR);
			data[1] = (float *)MG::MemoryAllocate(num_floats_per_cb*sizeof(float), MG::REGULAR);

			/* Offset the checkerboard */
			//data[1] = (data[0] + num_floats_per_cb);
		}

		/** GetSiteData
		 *
		 *  Returns a pointer to the data for a site in a cb
		 *  This is essentially a float array of size _n_site_offset
		 *  or it can be reinterpreted as _n_colorspin complexes
		 */
		inline
		float* GetSiteDataPtr(IndexType col, IndexType cb, IndexType site, bool raw=false)
		{
			if (!raw && _perm) site = (*_perm)[cb][site];
			return &data[cb][site*_n_site_offset+col*_n_col_offset];
		}

		inline
		const float* GetSiteDataPtr(IndexType col, IndexType cb, IndexType site, bool raw=false) const
		{
			if (!raw && _perm) site = (*_perm)[cb][site];
			return &data[cb][site*_n_site_offset+col*_n_col_offset];
		}


		IndexType PermIndexToCBIndex(IndexType site, int cb, bool raw=false) const
		{
			if (raw && _perm) site = (*_perm)[cb+2][site];
			return site;
		}

		~CoarseSpinor()
		{
			MemoryFree(data[0]);
			MemoryFree(data[1]);
			data[0] = nullptr;
			data[1] = nullptr;
		}

		inline
		IndexType GetNumColorSpin() const {
				return _n_colorspin;
		}

		inline
		IndexType GetNumColor() const {
				return _n_color;
		}

		inline
		IndexType GetNumSpin() const {
				return _n_spin;
		}

		inline
		const LatticeInfo& GetInfo() const {
			return _lattice_info;
		}

		inline
		const IndexType& GetNxh() const { return _n_xh; }

		inline
		const IndexType& GetNx() const { return _n_x; }

		inline
		const IndexType& GetNy() const { return _n_y; }

		inline
		const IndexType& GetNz() const { return _n_z; }

		inline
		const IndexType& GetNt() const { return _n_t; }

		inline
		const IndexType& GetNCol() const { return _n_col; }

		inline
		const IndexType& GetSiteDataLD() const { return _n_col_offset; }

		bool is_like(const CoarseSpinor& s) const {
			return _lattice_info.isCompatibleWith(s._lattice_info) && _n_col == s._n_col;
		}

		bool is_like(const LatticeInfo& info, int ncol) const {
			return GetInfo().isCompatibleWith(info) && GetNCol() == ncol;
		}

		CoarseSpinor* create_new() const {
			return new CoarseSpinor(GetInfo(), GetNCol());
		}

	private:
		const LatticeInfo& _lattice_info;
		float* data[2];  // Even and odd checkerboards

		const IndexType _n_color;
		const IndexType _n_spin;
		const IndexType _n_colorspin;
		const IndexType _n_site_offset;
		const IndexType _n_xh;
		const IndexType _n_x;
		const IndexType _n_y;
		const IndexType _n_z;
		const IndexType _n_t;
		const IndexType _n_col;
		const IndexType _n_col_offset;
		const CBPermutation _perm;

	};




	class CoarseGauge {
	public:
		CoarseGauge(const LatticeInfo& lattice_info) : _lattice_info(lattice_info), data{nullptr,nullptr}, diag_data{nullptr,nullptr},
		invdiag_data{nullptr,nullptr}, AD_data{nullptr,nullptr}, DA_data{nullptr,nullptr},
				_n_color(lattice_info.GetNumColors()),
				_n_spin(lattice_info.GetNumSpins()),
				_n_colorspin(lattice_info.GetNumColors()*lattice_info.GetNumSpins()),
				_n_link_offset(n_complex*_n_colorspin*_n_colorspin),
				_n_site_offset((2*n_dim)*_n_link_offset),
				_n_xh( lattice_info.GetCBLatticeDimensions()[0] ),
				_n_x( lattice_info.GetLatticeDimensions()[0] ),
				_n_y( lattice_info.GetLatticeDimensions()[1] ),
				_n_z( lattice_info.GetLatticeDimensions()[2] ),
				_n_t( lattice_info.GetLatticeDimensions()[3] )
		{
			// Check That we have 2 spins
			if( lattice_info.GetNumSpins() != 2 ) {
				MasterLog(ERROR, "Attempting to Create CoarseSpinor with num_spins != 2");
			}


			// Allocate Data - data, AD data and DA data are the off-diagonal links - 8 links per site (use n_site_iffset)
			IndexType offdiag_num_floats_per_cb = _lattice_info.GetNumCBSites()*_n_site_offset;

			// diag_data and invdiag data hold the clover terms. These are 1 link per site (use _n_link_offset)
			IndexType diag_num_floats_per_cb = _lattice_info.GetNumCBSites()*_n_link_offset;

			/* Contiguous allocation */
			data[0] = (float *)MG::MemoryAllocate(offdiag_num_floats_per_cb*sizeof(float), MG::REGULAR);
			data[1] = (float *)MG::MemoryAllocate(offdiag_num_floats_per_cb*sizeof(float), MG::REGULAR);

			diag_data[0] = (float *)MG::MemoryAllocate(diag_num_floats_per_cb*sizeof(float), MG::REGULAR);
			diag_data[1] = (float *)MG::MemoryAllocate(diag_num_floats_per_cb*sizeof(float), MG::REGULAR);

			invdiag_data[0] = (float *)MG::MemoryAllocate(diag_num_floats_per_cb*sizeof(float), MG::REGULAR);
			invdiag_data[1] = (float *)MG::MemoryAllocate(diag_num_floats_per_cb*sizeof(float), MG::REGULAR);

			AD_data[0] = (float *)MG::MemoryAllocate(offdiag_num_floats_per_cb*sizeof(float), MG::REGULAR);
			AD_data[1] = (float *)MG::MemoryAllocate(offdiag_num_floats_per_cb*sizeof(float), MG::REGULAR);

			DA_data[0] = (float *)MG::MemoryAllocate(offdiag_num_floats_per_cb*sizeof(float), MG::REGULAR);
			DA_data[1] = (float *)MG::MemoryAllocate(offdiag_num_floats_per_cb*sizeof(float), MG::REGULAR);

		}

		private:
		void _permute_array(float* v[2], IndexType block_size, CBPermutation new_p) {
			for (int i=0; i<2; i++) {
				float *old = v[i];
				if (_perm) {
					v[i] = MG::permute(v[i], block_size, (*_perm)[i], (*new_p)[i]);
				} else {
					v[i] = MG::permute(v[i], block_size, (*new_p)[i]);
				}
				MemoryFree(old);
			}
		}

		public:
		void permute(CBPermutation p) {
			// If both permutations are the same, skip it
			if (p == _perm) return;
		
			_permute_array(data, _n_site_offset, p);
			_permute_array(AD_data, _n_site_offset, p);
			_permute_array(DA_data, _n_site_offset, p);
			_permute_array(diag_data, _n_link_offset, p);
			_permute_array(invdiag_data, _n_link_offset, p);

			_perm = p;
		}

		IndexType PermIndexToCBIndex(IndexType site, int cb, bool raw=false) const
		{
			if (raw && _perm) site = (*_perm)[cb+2][site];
			return site;
		}


		/** GetSiteDirData
		 *
		 *  Returns a pointer to the link in direction mu
		 *  Conventions are:
		 *  	mu=0 - X forward
		 *  	mu=1 - X backward
		 *  	mu=2 - Y forwad
		 *  	mu=3 - Y backward
		 *  	mu=4 - Z forward
		 *  	mu=5 - Z backward
		 *      mu=6 - T forward
		 *      mu=7 - T backward
		 */
		inline
		float *GetSiteDirDataPtr(IndexType cb, IndexType site, IndexType mu, bool raw=false)
		{
			if (!raw && _perm) site = (*_perm)[cb][site];
			return &data[cb][site*_n_site_offset + mu*_n_link_offset];
		}

		inline
		const float *GetSiteDirDataPtr(IndexType cb, IndexType site, IndexType mu, bool raw=false) const
		{
			if (!raw && _perm) site = (*_perm)[cb][site];
			return &data[cb][site*_n_site_offset + mu*_n_link_offset];
		}

		/** GetSiteDirADData
		 *
		 *  Returns a pointer to the link in direction mu
		 *  Conventions are:
		 *  	mu=0 - X forward
		 *  	mu=1 - X backward
		 *  	mu=2 - Y forwad
		 *  	mu=3 - Y backward
		 *  	mu=4 - Z forward
		 *  	mu=5 - Z backward
		 *      mu=6 - T forward
		 *      mu=7 - T backward
		 */
		inline
		float *GetSiteDirADDataPtr(IndexType cb, IndexType site, IndexType mu, bool raw=false)
		{
			if (!raw && _perm) site = (*_perm)[cb][site];
			return &AD_data[cb][site*_n_site_offset + mu*_n_link_offset];
		}

		inline
		const float *GetSiteDirADDataPtr(IndexType cb, IndexType site, IndexType mu, bool raw=false) const
		{
			if (!raw && _perm) site = (*_perm)[cb][site];
			return &AD_data[cb][site*_n_site_offset + mu*_n_link_offset];
		}

		/** GetSiteDirDAData
			 *
			 *  Returns a pointer to the link in direction mu
			 *  Conventions are:
			 *  	mu=0 - X forward
			 *  	mu=1 - X backward
			 *  	mu=2 - Y forwad
			 *  	mu=3 - Y backward
			 *  	mu=4 - Z forward
			 *  	mu=5 - Z backward
			 *      mu=6 - T forward
			 *      mu=7 - T backward
			 */
		inline
		float *GetSiteDirDADataPtr(IndexType cb, IndexType site, IndexType mu, bool raw=false)
		{
			if (!raw && _perm) site = (*_perm)[cb][site];
			return &DA_data[cb][site*_n_site_offset + mu*_n_link_offset];
		}

		inline
		const float *GetSiteDirDADataPtr(IndexType cb, IndexType site, IndexType mu, bool raw=false) const
		{
			if (!raw && _perm) site = (*_perm)[cb][site];
			return &DA_data[cb][site*_n_site_offset + mu*_n_link_offset];
		}


		inline
		float *GetSiteDiagDataPtr(IndexType cb, IndexType site, bool raw=false)
		{
			if (!raw && _perm) site = (*_perm)[cb][site];
			return &diag_data[cb][site*_n_link_offset];
		}

		inline
		const float *GetSiteDiagDataPtr(IndexType cb, IndexType site, bool raw=false) const
		{
			if (!raw && _perm) site = (*_perm)[cb][site];
			return &diag_data[cb][site*_n_link_offset];
		}

		inline
		float *GetSiteInvDiagDataPtr(IndexType cb, IndexType site, bool raw=false)
		{
			if (!raw && _perm) site = (*_perm)[cb][site];
			return &invdiag_data[cb][site*_n_link_offset];
		}

		inline
		const float *GetSiteInvDiagDataPtr(IndexType cb, IndexType site, bool raw=false) const
		{
			if (!raw && _perm) site = (*_perm)[cb][site];
			return &invdiag_data[cb][site*_n_link_offset];
		}

		inline
		IndexType GetNCol() const { return 1; }

		~CoarseGauge()
		{
			MemoryFree(data[0]);
			MemoryFree(data[1]);
			MemoryFree(diag_data[0]);
			MemoryFree(diag_data[1]);
			MemoryFree(invdiag_data[0]);
			MemoryFree(invdiag_data[1]);
			MemoryFree(AD_data[0]);
			MemoryFree(AD_data[1]);
			MemoryFree(DA_data[0]);
			MemoryFree(DA_data[1]);

			data[0] = nullptr;
			data[1] = nullptr;
			diag_data[0] = nullptr;
			diag_data[1] = nullptr;
			invdiag_data[0] = nullptr;
			invdiag_data[1] = nullptr;
			AD_data[0] = nullptr;
			AD_data[1] = nullptr;
			DA_data[0] = nullptr;
			DA_data[1] = nullptr;
		}

		inline
		IndexType GetNumColorSpin() const {
				return _n_colorspin;
		}

		inline
		IndexType GetNumColor() const {
				return _n_color;
		}

		inline
		IndexType GetNumSpin() const {
				return _n_spin;
		}

		inline
		IndexType GetLinkOffset() const {
			return _n_link_offset;
		}

		inline
		IndexType GetSiteOffset() const {
			return _n_site_offset;
		}

		inline
		const LatticeInfo& GetInfo() const {
			 return _lattice_info;
		}

		inline
		const IndexType& GetNxh() const { return _n_xh; }

		inline
		const IndexType& GetNx() const { return _n_x; }

		inline
		const IndexType& GetNy() const { return _n_y; }

		inline
		const IndexType& GetNz() const { return _n_z; }

		inline
		const IndexType& GetNt() const { return _n_t; }

	private:
		const LatticeInfo& _lattice_info;
		float* data[2];        // Even and odd checkerboards off diagonal data (D)
		float* diag_data[2];   // Diagonal data (Clov, or A)
		float* invdiag_data[2]; // Inverse Clover (A^{-1})
		float* AD_data[2]; // holds A^{-1}_oo D_oe and A^{-1}_ee D_eo (AD)
		float* DA_data[2]; // holds D_oe A^{-1}_ee and D_eo A^{-1}_oo (DA)


		const IndexType _n_color;
		const IndexType _n_spin;
		const IndexType _n_colorspin;
		const IndexType _n_link_offset;
		const IndexType _n_site_offset;
		const IndexType _n_xh;
		const IndexType _n_x;
		const IndexType _n_y;
		const IndexType _n_z;
		const IndexType _n_t;

		CBPermutation _perm;
	};


	template<typename T>
	static size_t haloDatumSize(const LatticeInfo& info);

	template<>
	inline
	size_t haloDatumSize<CoarseSpinor>(const LatticeInfo& info)
	{
		return n_complex*info.GetNumColorSpins();
	}

	template<>
	inline
	size_t haloDatumSize<CoarseGauge>(const LatticeInfo& info)
	{
		return n_complex*info.GetNumColorSpins()*info.GetNumColorSpins();
	}

}



#endif /* INCLUDE_LATTICE_COARSE_COARSE_TYPES_H_ */
