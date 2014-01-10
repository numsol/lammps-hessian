/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   LAMMPS-HESSIAN - Copyright (2014) Anthony B. Costa.
   anthony.costa@numericalsolutions.org, Numerical Solutions, Inc.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "mpi.h"
#include "math.h"
#include "string.h"
#include "stdlib.h"
#include "compute_hessian.h"
#include "atom.h"
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

  xlocal = flocal = NULL;
  torquelocal = NULL;
  erforcelocal = delocal = drholocal = NULL;
  fglobal_ref = fglobal_new = fglobal_copy = NULL;
  hessian = NULL;
}

/* ---------------------------------------------------------------------- */

ComputeHessian::~ComputeHessian() {
  memory->destroy(xlocal);
  memory->destroy(flocal);

  memory->destroy(torquelocal);
  memory->destroy(erforcelocal);
  memory->destroy(delocal);
  memory->destroy(drholocal);

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
               "Cannot use velocity create loop all unless atoms have IDs");
  if (atom->tag_consecutive() == 0)
    error->all(FLERR,
               "Atom IDs must be consecutive for velocity create loop all");

  /* set flags for what arrays to clear in force_clear(). */
  /* if they exist they are cleared and then replaced later. */
  torqueflag = erforceflag = e_flag = rho_flag = 0;
  if (atom->torque_flag)
    torqueflag = 1;
  if (atom->erforce_flag)
    erforceflag = 1;
  if (atom->e_flag)
    e_flag = 1;
  if (atom->rho_flag)
    rho_flag = 1;

  /* get pointers to all the original data. */
  double **x = atom->x;
  double **f = atom->f;
  double **torque = atom->torque;
  double *erforce = atom->erforce;
  double *de = atom->de;
  double *drho = atom->drho;

  /* only grow into larger size arrays for vectors which are not being managed
   * explicitly by mpi or returned. */
  int needlocalsize;
  if (force->newton)
    needlocalsize = atom->nlocal + atom->nghost;
  else
    needlocalsize = atom->nlocal;

  if (needlocalsize > mylocalsize) {
    memory->grow(xlocal, needlocalsize, 3, "hessian:xlocal");
    memory->grow(flocal, needlocalsize, 3, "hessian:flocal");

    if (torqueflag)
      memory->grow(torquelocal, needlocalsize, 3, "hessian:torque");
    if (erforceflag)
      memory->grow(erforcelocal, needlocalsize, "hessian:erforce");
    if (e_flag)
      memory->grow(delocal, needlocalsize, "hessian:de");
    if (rho_flag)
      memory->grow(drholocal, needlocalsize, "hessian:drho");

    mylocalsize = needlocalsize;
  }

  /* copy the data from lammps atom class to local array. */
  memcpy(&xlocal[0][0], &x[0][0], needlocalsize * 3 * sizeof(double));

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
  memset(hessian, 0, nhessianelements * sizeof(double));

  /* point all the class data to the local storage for now. */
  atom->x = xlocal;
  atom->f = flocal;
  atom->torque = torquelocal;
  atom->erforce = erforcelocal;
  atom->de = delocal;
  atom->drho = drholocal;

  /* construct fglobal_ref by explicit scatter and reduce to preserve atom-id
   * ordering. */
  int reduce_id;
  memset (&fglobal_copy[0], 0, myglobalsize * 3 * sizeof (double));
  for (int i = 0; i < atom->nlocal; i++) {
    reduce_id = atom->tag[i] - 1; 
    for (int j = 0; j < domain->dimension; j++) {
      fglobal_copy[idx2_r(reduce_id, j, 3)] = f[i][j];
    }
  }
  MPI_Allreduce (fglobal_copy, fglobal_ref, ndofs, MPI_DOUBLE, MPI_SUM, world);

  /* no energy or virial updates. */
  int eflag = 0;
  int vflag = 0;

  /* local map, per-processor flag to indicate the atom is on that task. */
  int m, myatom;

  /* parent mass will be broadcast from whatever task it lives on. */
  double mass;

  /* set up a map if none exists so we can incrementally loop through all dofs
   * regardless of the location of the atom data. */
  int mapflag = 0;
  if (atom->map_style == 0) {
    mapflag = 1;
    atom->map_style = 1;
    atom->map_init();
    atom->map_set();
  }

  /* embedded indices, and inner hessian quantities. */
  double difference, mass_weight;
  int index_a, index_b, global_atom_a, global_atom_b;

  /* allow pair and kspace compute to be turned off via modify flags. */
  if (force->pair && force->pair->compute_flag)
    pair_compute_flag = 1;
  else
    pair_compute_flag = 0;
  if (force->kspace && force->kspace->compute_flag)
    kspace_compute_flag = 1;
  else
    kspace_compute_flag = 0;

  /* do numerical hessian compute by forward differences. */
  for (int i = 1; i <= atom->natoms; i++) {

    myatom = 0;
    m = atom->map(i);
    if (m >= 0 && m < atom->nlocal) {
      myatom = 1;

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
      if (myatom)
        xlocal[m][j] += epsilon;

      /* standard force call. */
      comm->forward_comm();
      force_clear();

      if (modify->n_pre_force)
        modify->pre_force(vflag);

      if (pair_compute_flag)
        force->pair->compute(eflag, vflag);

      if (atom->molecular) {
        if (force->bond)
          force->bond->compute(eflag, vflag);
        if (force->angle)
          force->angle->compute(eflag, vflag);
        if (force->dihedral)
          force->dihedral->compute(eflag, vflag);
        if (force->improper)
          force->improper->compute(eflag, vflag);
      }

      if (kspace_compute_flag)
        force->kspace->compute(eflag, vflag);

      if (modify->n_post_force)
        modify->post_force(vflag);

      /* put the original position back. */
      if (myatom)
        xlocal[m][j] = x[m][j];

      /* construct fglobal_new by explicit scatter and reduce to preserve
       * atom-id ordering. */
      memset (&fglobal_copy[0], 0, myglobalsize * 3 * sizeof (double));
      for (int k = 0; k < atom->nlocal; k++) {
        reduce_id = atom->tag[k] - 1;
        for (int l = 0; l < 3; l++) {
          fglobal_copy[idx2_r(reduce_id, l, 3)] = flocal[k][l];
        }
      }
      MPI_Allreduce (fglobal_copy, fglobal_new, ndofs, MPI_DOUBLE, MPI_SUM, world);

      /* compute the difference (not using symmetry so we can do an in-place
       * reduciton). */
      index_a = j + 3 * global_atom_a;
      for (int k = 0; k < atom->nlocal; k++) {

        /* don't need to broadcast the second mass because it will only be used
         * on this rank. */
        if (atom->rmass)
          mass_weight = 1 / sqrt(mass * atom->rmass[k]);
        else
          mass_weight = 1 / sqrt(mass * atom->mass[atom->type[k]]);

        /* once again, global arrays use 1-based indexing, so have to rebase
         * them to 0. */
        global_atom_b = atom->tag[k] - 1;
        for (int l = 0; l < domain->dimension; l++) {
          index_b = l + 3 * global_atom_b;
          difference =
              fglobal_ref[idx2_r(global_atom_b, l, 3)] - fglobal_new[idx2_r(global_atom_b, l, 3)];
          hessian[idx2_c(index_a, index_b, ndofs)] =
              difference * iepsilon * mass_weight;
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

  /* point all the original class data back. */
  atom->x = x;
  atom->f = f;
  atom->torque = torque;
  atom->erforce = erforce;
  atom->de = de;
  atom->drho = drho;
}

void ComputeHessian::force_clear() {
  /* clear global force array. nall includes ghosts only if either newton flag
   * is set. */
  int nall;
  if (force->newton)
    nall = atom->nlocal + atom->nghost;
  else
    nall = atom->nlocal;

  size_t nbytes = sizeof(double) * nall;

  if (nbytes) {
    memset(&(atom->f[0][0]), 0, 3 * nbytes);
    if (torqueflag)
      memset(&(atom->torque[0][0]), 0, 3 * nbytes);
    if (erforceflag)
      memset(&(atom->erforce[0]), 0, nbytes);
    if (e_flag)
      memset(&(atom->de[0]), 0, nbytes);
    if (rho_flag)
      memset(&(atom->drho[0]), 0, nbytes);
  }
}
