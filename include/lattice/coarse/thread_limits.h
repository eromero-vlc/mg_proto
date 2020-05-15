/*
 * thread_limits.h
 *
 *  Created on: Jan 6, 2017
 *      Author: bjoo
 */

#ifndef INCLUDE_LATTICE_COARSE_THREAD_LIMITS_H_
#define INCLUDE_LATTICE_COARSE_THREAD_LIMITS_H_

#include "lattice/constants.h"

namespace MG {
struct ThreadLimits {


/*
 *  Was:
 *    min_row,max_vrow
 */
	IndexType min_site;
	IndexType max_site;
};
}



#endif /* INCLUDE_LATTICE_COARSE_THREAD_LIMITS_H_ */
