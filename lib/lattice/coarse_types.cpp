#include <vector>
#include <cassert>
#include "MG_config.h"
#include "lattice/coarse/coarse_types.h"
#include "lattice/geometry_utils.h"
#include "utils/memory.h"

using namespace MG;

namespace MG {

#ifdef MG_HACK_HILBERT_CURVE

	/** Cache-efficient permutation over sites
	 *  \param begin: first element where to store the permutation
	 *  \param dim: lattice dimension
	 *  \param J: dimension in which the entry and exit point coordinates differ
	 *
	 *  Returns a permutation over the sites of lattice with dimensions 'dim' that increase the cache hits.
	 *  The routine writes prod(dim) values, being the first value at 'begin'.
	 *  The values of 'entry' indicates the face (1, backward; -1, forward) of the
	 *  first site. For instance, an entry of {1,1,1,1} correspond to the site at origin. The last
	 *  site on the permutation will differ only at coordinate 'J' from the first site, that is,
	 *  it is opposite to the first site with respect the coordinate J.
	 *
	 */
	void gen_hilbert_curve(std::vector<IndexType>::iterator begin, IndexArray dim, IndexArray entry, int J) {
		// Trivial cases: volumes zero and one
		std::size_t vol = Volume(dim);
		if (vol < 2) {
			for (IndexType i=0; i<vol; i++, begin++)
				*begin = i;
			return;
		}
		//assert(dim[J] > 1);

		// Extract the even factors for each dimension
		IndexArray original_dim(dim);
		IndexArray dime;
		std::fill_n(dime.begin(), dime.size(), 1); // init dime to all ones
		while (true) {
			bool all_dim_elemns_div_by_two = (dim[J] > 1);
			for (unsigned int i=0; i<dim.size(); i++)
				all_dim_elemns_div_by_two &= (dim[i] % 2 == 0);
			if (!all_dim_elemns_div_by_two) break;

			bool some_dim_elemn_greather_than_one = false;
			for (unsigned int i=0; i<dim.size(); i++)
				some_dim_elemn_greather_than_one |= (dim[i] > 1);
			if (!some_dim_elemn_greather_than_one) break;

			for (unsigned int i=0; i<dim.size(); i++) {
				if (dim[i] > 1) {
					dim[i] /= 2;
					dime[i] *= 2;
				}
			}
		}

		// If no remaining volume, take a factor of two
		vol = Volume(dim);
		if (vol == 1) {
			for (unsigned int i=0; i<dime.size(); i++) {
				if (dime[i] > 1) {
					dime[i] /= 2;
					dim[i] *= 2;
				}
			}
		}

		// Do snake walking! Rules:
		// - Reorder the coordinates such that J is the first one
		// - At each step try to move in the dimension with largest coordinate
		//   following the direction at each coordinate; in each attempt, reverse
		//   the direction of the coordinate if it fails.
		// - Move the 'entry' following the path of sites.
		std::vector<unsigned int> order;
		order.reserve(dim.size());
		order.push_back(J);
		for (unsigned int i=0; i<dim.size(); i++) {
			if (dim[i] > 1 && i != J) order.push_back(i);
		}

		// Exit point
		IndexArray exit(entry);
		exit[J] *= -1;

		IndexArray coor;
		for (unsigned int i=0; i<coor.size(); i++) {
			coor[i] = (entry[i] == 1 ? 0 : dim[i] - 1);
		}
		IndexArray dir(entry);
		vol = Volume(dim);
		const unsigned int vol_dime = Volume(dime);
		for (unsigned int i=0; i<vol; i++) {
			// Figure out the next move for the current site
			int next_direction;
			for (next_direction=(int)order.size()-1; next_direction>=0; next_direction--) {
				// If we move current site 'coor' in dimension 'order[next_direction]' towards the
				// direction 'dir[order[next_direction]', check if it is not hitting a lattice wall
				IndexType next_coor_order_mu = coor[order[next_direction]] + dir[order[next_direction]];
				if (0 <= next_coor_order_mu && next_coor_order_mu < dim[order[next_direction]]) break;

				// If hitting a lattice wall, the current direction in this dimension, and try the next dimension
				dir[order[next_direction]] *= -1;
			}
			if (next_direction < 0) {assert(i==vol-1); next_direction = 0;}

			// Figure out the next move the entry point
			unsigned int this_J;
			if (i < vol-1) {
				// If this is not the final move, move the entry point along the dimension
				// 'next_direction' if the destination still touches the exit face; otherwise
				// choose another dimension
				if (entry[order[next_direction]] == dir[order[next_direction]] || order.size() == 1) {
					this_J = order[next_direction];
				} else {
					this_J = order[(order.size() + next_direction - 1) % order.size()];
				}
			} else {
				// At the final site, move the entry point to align with the original entry point
				if (exit[J] != entry[J] || order.size() == 1) {
					this_J = J;
				} else {
					for (this_J=1; this_J<order.size() && exit[order[this_J]] == entry[order[this_J]]; this_J++);
					this_J = order[this_J < order.size() ? this_J : 0];
				}
			}

			// Generate permutation for the sublattice
			gen_hilbert_curve(begin, dime, entry, this_J);
			for (unsigned int j=0; j<vol_dime; j++, begin++) {
				IndexArray c;
				IndexToCoords(*begin, dime, c);
				for (unsigned int mu=0; mu<dim.size(); mu++) {
					c[mu] += coor[mu] * dime[mu];
				}
				*begin = CoordsToIndex(c, original_dim);
			}

			// Check boundaries
			//IndexArray prev;
			//if (i == 0) prev = coor; else IndexToCoords(*(begin-vol_dime-1), original_dim, prev);
			//IndexArray prev_next;
			//IndexToCoords(*(begin-vol_dime), original_dim, prev_next);
			//for(int j=0, distinct_coors=0; j<dim.size(); ++j) {
			//	IndexType distj = std::min((original_dim[j] + prev[j] - prev_next[j]) % original_dim[j], (prev_next[j] - prev[j]  + original_dim[j]) % original_dim[j]);
			//	assert(distj < 2);
			//	if (distj == 1) distinct_coors++;
			//	assert(distinct_coors < 3);
			//}

			// Move the entry point to the exit point
			entry[this_J] *= -1;

			// if (i == vol-1) {
			// 	for(int j=0; j<dim.size(); ++j) {
			// 		assert(entry[j] == exit[j]);
			// 	}
			// }

			// Move the site and the entry point along 'next_direction'
			coor[order[next_direction]] += dir[order[next_direction]];
			entry[order[next_direction]] *= -1;
		}
	} 

	CBPermutation cache_optimal_permutation(const LatticeInfo& info) {
		using reg = std::tuple<const LatticeInfo, CBPermutation>;
		static std::vector<reg> _perms;

		// Return the permutation if it was done before
		for (auto it=_perms.begin(); it != _perms.end(); ++it) {
			if (std::get<0>(*it).isCompatibleWith(info)) {
				return std::get<1>(*it);
			}
		}

		// Create a new one
		CBPermutation s(new std::array<std::vector<IndexType>,4>);
		for (int i=0; i<4; i++) (*s)[i].resize(info.GetNumCBSites());

		// Generate a path over all sites on the lattice
		IndexArray cb_lattice_size(info.GetLatticeDimensions());
		{
			int icb=0;
			for (icb=0; icb<n_dim && cb_lattice_size[icb] % 2 != 0; ++icb);
			assert(icb < n_dim);
			cb_lattice_size[icb] /= 2; // Size of checkberboarded lattice
		}
		int J = 0;
		for (int i=0; i<n_dim; i++) if (cb_lattice_size[i] > cb_lattice_size[J]) J=i;

		std::vector<IndexType> p(info.GetNumSites()/2);
		gen_hilbert_curve(p.begin(), cb_lattice_size, {1,1,1,1}, J);

		std::vector<bool> visited(info.GetNumSites()/2, false);
		IndexType icb[2] = {0,0};
		MasterLog(INFO, "Perm for lattice %d %d %d %d", info.GetLatticeDimensions()[0], info.GetLatticeDimensions()[1], info.GetLatticeDimensions()[2], info.GetLatticeDimensions()[3]);
		std::size_t fails[2] = {0, 0};
		IndexArray last_coor[2];
		for (std::size_t i=0; i<info.GetNumSites()/2; ++i) {
			// Check that p is a permutation over the sites
			assert(p[i] >= 0 && p[i] < info.GetNumSites()/2 && !visited[p[i]]);
			visited[p[i]] = true;

			for (int cb=0; cb<2; cb++) {
				// Get the coordinates of the site
				// Check
				IndexArray c;
				CBIndexToCoords(p[i], cb, info.GetLatticeDimensions(), info.GetLatticeOrigin(), c);
				if (i >= 1) {
					for(int j=0, num_dist_coor=0; j<c.size(); ++j) {
						IndexType dj = info.GetLatticeDimensions()[j];
						int dist_j = std::min((dj + c[j] - last_coor[cb][j]) % dj, (last_coor[cb][j] - c[j]  + dj) % dj);
						if (dist_j > 0) num_dist_coor++;
						if (dist_j > 1 || num_dist_coor > 2) {
							fails[cb]++;
							break;
						}
					}
				}
				last_coor[cb] = c;

				// Site with CB index and parity cb is going to be stored at position icb[cb]
				IndexType cbsite = p[i];
				(*s)[cb][cbsite] = icb[cb]++;

				// Reverse map
				(*s)[2+cb][ (*s)[cb][cbsite] ] = cbsite;
			}
		}
		MasterLog(INFO, "Perm fails: %d %d", (int)fails[0], (int)fails[1]);

		// Add it to the collection of permutations
		_perms.emplace_back(std::make_tuple(info, s));

		// Return the last acquisition
		return s;
	}
#else // MG_HACK_HILBERT_CURVE

	CBPermutation cache_optimal_permutation(const LatticeInfo& info) {
		(void)info;
		return CBPermutation();
	}

#endif // MG_HACK_HILBERT_CURVE


	float* permute(const float* v, std::size_t block_size, const std::vector<IndexType>& p) {

		float* r = (float *)MG::MemoryAllocate(block_size*sizeof(float), MG::REGULAR);
		for (IndexType i=0; i<p.size(); ++i) {
			for (IndexType j=0; j<block_size; ++j) {
				r[block_size*p[i] + j] = v[block_size*i + j];
			}
		}
		return r;
	}

	float* permute(const float* v, std::size_t block_size, const std::vector<IndexType>& p0, const std::vector<IndexType>& p1) {

		assert(p0.size() == p1.size());	
		float* r = (float *)MG::MemoryAllocate(block_size*sizeof(float), MG::REGULAR);
		for (IndexType i=0; i<p0.size(); ++i) {
			for (IndexType j=0; j<block_size; ++j) {
				r[block_size*p1[i] + j] = v[block_size*p0[i] + j];
			}
		}
		return r;
	}

}

