/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef COMPUTE_CLASS

ComputeStyle(hessian, ComputeHessian)

#else

#ifndef LMP_COMPUTE_HESSIAN_H
#define LMP_COMPUTE_HESSIAN_H

#include "compute.h"

/* zero-based row- and column-major indexing macros for the hessian. */
#define idx2_r(i, j, ldj) ((i * ldj) + j)
#define idx2_c(i, j, ldi) ((j * ldi) + i)

namespace LAMMPS_NS {

  class ComputeHessian : public Compute {
  public:
    ComputeHessian(class LAMMPS *, int, char **);
    ~ComputeHessian();
    void init() {}
    void compute_vector();
  
  protected:
    int mylocalsize;
    int myglobalsize;
  
    double *fglobal_ref, *fglobal_new, *fglobal_copy;
    double *hessian;
  
    double epsilon, iepsilon;
  
    int pair_compute_flag;
    int kspace_compute_flag;
  
    void force_clear();
  };

}

#endif
#endif
