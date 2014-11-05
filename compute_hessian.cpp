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

#include "mpi.h"
#include "math.h"
#include "string.h"
#include "stdlib.h"
#include "neighbor.h"
#include "compute_hessian.h"
#include "atom.h"
#include "atom_vec.h"
#include "error.h"
#include "update.h"
#include "memory.h"
#include "domain.h"
#include "modify.h"
#include "force.h"
#include "pair.h"
#include "bond.h"
#include "angle.h"
#include "dihedral.h"
#include "improper.h"
#include "kspace.h"
#include "comm.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

ComputeHessian::ComputeHessian(LAMMPS *lmp, int narg, char **arg)
    : Compute(lmp, narg, arg) {
  if (narg != 4)
    error->all(FLERR, "Illegal compute hessian command");

  epsilon = atof(arg[3]);
  iepsilon = 1 / epsilon;

  /* even though this is a massive 2d array, return the a vector instead.
   * we will explicitly manage the addressing in each dimension with a
   * preprocessor index macro. */
  vector_flag = 1;
  extvector = 0;

  /* these values will change if the system size changes. */
  int ndofs = atom->natoms * 3;
  size_vector = ndofs * ndofs;

  mylocalsize = 0;
  myglobalsize = 0;

  fglobal_ref = fglobal_new = fglobal_copy = NULL;
  hessian = NULL;
}

/* ---------------------------------------------------------------------- */

ComputeHessian::~ComputeHessian() {
  free(fglobal_ref);
  free(fglobal_new);
  free(fglobal_copy);
  free(hessian);
}

/* ---------------------------------------------------------------------- */

void ComputeHessian::compute_vector(void) {
  invoked_vector = update->ntimestep;

  /* tags must be defined and consecutive. */
  if (atom->tag_enable == 0)
    error->all(FLERR,
               "Cannot use Hessian compute unless atoms have IDs");
  if (atom->tag_consecutive() == 0)
    error->all(FLERR,
               "Atom IDs must be consecutive for Hessian compute");

  /* get pointers to all the original data. */
  double **x = atom->x;
  double **f = atom->f;

  /* the global force and hessian arrays must be explicitly the correct size. */
  int needglobalsize = atom->natoms;
  int ndofs = atom->natoms * 3;
  bigint nhessianelements = ndofs * ndofs;
  if (needglobalsize != myglobalsize) {
    free (fglobal_ref); 
    free (fglobal_new); 
    free (fglobal_copy);
    free (hessian);

    fglobal_ref = (double *) malloc (ndofs * sizeof (double));   
    fglobal_new = (double *) malloc (ndofs * sizeof (double));   
    fglobal_copy = (double *) malloc (ndofs * sizeof (double));   
    hessian = (double *) malloc (nhessianelements * sizeof (double));

    /* always be sure to set the output vector since the address keeps changing. */
    vector = hessian;

    myglobalsize = needglobalsize;
  }

  /* a lot of the hessian will be zero, so start there. */
  memset (hessian, 0, nhessianelements * sizeof(double));

  /* set up a map if none exists so we can incrementally loop through all dofs
   * regardless of the location of the atom data. */
  int mapflag = 0;
  if (atom->map_style == 0) {
    mapflag = 1;
    atom->map_init();
    atom->map_set();
  }

  /* no energy or virial updates. */
  int eflag = 0;
  int vflag = 0;

  /* allow pair and kspace compute to be turned off via modify flags. */
  if (force->pair && force->pair->compute_flag)
    pair_compute_flag = 1;
  else
    pair_compute_flag = 0;
  if (force->kspace && force->kspace->compute_flag)
    kspace_compute_flag = 1;
  else
    kspace_compute_flag = 0;

  /* do a standard force call to get the reference forces. */
  comm->forward_comm();
  force_clear();
  if (modify->n_pre_force) modify->pre_force(vflag);

  if (pair_compute_flag) force->pair->compute(eflag, vflag);

  if (atom->molecular) {
    if (force->bond) force->bond->compute(eflag, vflag);
    if (force->angle) force->angle->compute(eflag, vflag);
    if (force->dihedral) force->dihedral->compute(eflag, vflag);
    if (force->improper) force->improper->compute(eflag, vflag);
  }

  if (kspace_compute_flag) force->kspace->compute(eflag, vflag);
  if (force->newton) comm->reverse_comm();
  if (modify->n_post_force) modify->post_force(vflag);

  /* construct fglobal_ref by explicit scatter and reduce to preserve atom-id
   * ordering. */
  int m, reduce_m;
  memset (&fglobal_copy[0], 0, myglobalsize * 3 * sizeof (double));
  for (int i = 1; i <= atom->natoms; i++) {
    m = atom->map(i);
    if (atom->mask[m]) {
      reduce_m = atom->tag[m] - 1;
      for (int j = 0; j < domain->dimension; j++)
        fglobal_copy[idx2_c(reduce_m, j, atom->natoms)] = f[m][j];
    }
  }
  MPI_Allreduce (fglobal_copy, fglobal_ref, ndofs, MPI_DOUBLE, MPI_SUM, world);

  /* do numerical hessian compute by forward differences. */
  int n, reduce_n, index_a, index_b, global_atom_a, global_atom_b;
  double mass, difference, mass_weight, xstore;
  for (int i = 1; i <= atom->natoms; i++) {

    m = atom->map(i);
    if (atom->mask[m]) {
      /* global ids in lammps are handled by 1-based indexing, while everything
       * local is 0-based. */
      global_atom_a = atom->tag[m] - 1;
      MPI_Bcast(&global_atom_a, 1, MPI_INT, comm->me, world);

      if (atom->rmass) {
        mass = atom->rmass[m];
        MPI_Bcast(&mass, 1, MPI_DOUBLE, comm->me, world);
      } else {
        mass = atom->mass[atom->type[m]];
        MPI_Bcast(&mass, 1, MPI_DOUBLE, comm->me, world);
      }
    }

    for (int j = 0; j < domain->dimension; j++) {
      /* increment the dof by epsilon on the right task. */
      if (atom->mask[m]) {
        xstore = x[m][j];
        x[m][j] += epsilon;
      }

      /* standard force call. */
      comm->forward_comm();
      force_clear();
      if (modify->n_pre_force) modify->pre_force(vflag);

      if (pair_compute_flag) force->pair->compute(eflag, vflag);

      if (atom->molecular) {
        if (force->bond) force->bond->compute(eflag, vflag);
        if (force->angle) force->angle->compute(eflag, vflag);
        if (force->dihedral) force->dihedral->compute(eflag, vflag);
        if (force->improper) force->improper->compute(eflag, vflag);
      }

      if (kspace_compute_flag) force->kspace->compute(eflag, vflag);

      /* put the original position back. */
      if (atom->mask[m]) x[m][j] = xstore;

      if (force->newton) comm->reverse_comm();
      if (modify->n_post_force) modify->post_force(vflag);

      /* construct fglobal_new by explicit scatter and reduce to preserve
       * atom-id ordering. */
      memset (&fglobal_copy[0], 0, myglobalsize * 3 * sizeof (double));
      for (int k = 1; k <= atom->natoms; k++) {
        n = atom->map(k);
        if (atom->mask[n]) {
          reduce_n = atom->tag[n] - 1;
          for (int l = 0; l < domain->dimension; l++)
            fglobal_copy[idx2_c(reduce_n, l, atom->natoms)] = f[n][l];
        }
      }
      MPI_Allreduce (fglobal_copy, fglobal_new, ndofs, MPI_DOUBLE, MPI_SUM, world);

      /* compute the difference (not using symmetry so we can do an in-place
       * reduciton). */
      index_a = j + 3 * global_atom_a;
      for (int k = 1; k <= atom->natoms; k++) {
        n = atom->map(k);
        if (atom->mask[n]) {
          global_atom_b = atom->tag[n] - 1;

          /* don't need to broadcast the second mass because it will only be used
           * on this rank. */
          if (atom->rmass)
            mass_weight = 1 / sqrt(mass * atom->rmass[n]);
          else
            mass_weight = 1 / sqrt(mass * atom->mass[atom->type[n]]);

          /* once again, global arrays use 1-based indexing, so have to rebase
           * them to 0. */
          for (int l = 0; l < domain->dimension; l++) {
            index_b = l + 3 * global_atom_b;
            difference =
                fglobal_ref[idx2_c(global_atom_b, l, atom->natoms)] - \
                fglobal_new[idx2_c(global_atom_b, l, atom->natoms)];

            hessian[idx2_c(index_a, index_b, ndofs)] =
                difference * iepsilon * mass_weight;
          }
        }
      }
    }
  }

  /* only reduce the hessian to the root task. */
  MPI_Reduce(MPI_IN_PLACE, hessian, nhessianelements, MPI_DOUBLE, MPI_SUM, 0, world);

  /* destroy the atom map. */
  if (mapflag) {
    atom->map_delete();
    atom->map_style = 0;
  }

  /* do a standard force call to get the original forces back. */
  comm->forward_comm();
  force_clear();
  if (modify->n_pre_force) modify->pre_force(vflag);

  if (pair_compute_flag) force->pair->compute(eflag, vflag);

  if (atom->molecular) {
    if (force->bond) force->bond->compute(eflag, vflag);
    if (force->angle) force->angle->compute(eflag, vflag);
    if (force->dihedral) force->dihedral->compute(eflag, vflag);
    if (force->improper) force->improper->compute(eflag, vflag);
  }

  if (kspace_compute_flag) force->kspace->compute(eflag, vflag);
  if (force->newton) comm->reverse_comm();
  if (modify->n_post_force) modify->post_force(vflag);
}

void ComputeHessian::force_clear() {
  size_t nbytes;
  int nlocal = atom->nlocal;

  nbytes = sizeof(double) * nlocal;
  if (force->newton) nbytes += sizeof(double) * atom->nghost;

  if (nbytes) memset (&atom->f[0][0], 0, 3 * nbytes);
}
