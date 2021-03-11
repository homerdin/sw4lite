//-*-c++-*-
//  SW4 LICENSE
// # ----------------------------------------------------------------------
// # SW4 - Seismic Waves, 4th order
// # ----------------------------------------------------------------------
// # Copyright (c) 2013, Lawrence Livermore National Security, LLC. 
// # Produced at the Lawrence Livermore National Laboratory. 
// # 
// # Written by:
// # N. Anders Petersson (petersson1@llnl.gov)
// # Bjorn Sjogreen      (sjogreen2@llnl.gov)
// # 
// # LLNL-CODE-643337 
// # 
// # All rights reserved. 
// # 
// # This file is part of SW4, Version: 1.0
// # 
// # Please also read LICENCE.txt, which contains "Our Notice and GNU General Public License"
// # 
// # This program is free software; you can redistribute it and/or modify
// # it under the terms of the GNU General Public License (as published by
// # the Free Software Foundation) version 2, dated June 1991. 
// # 
// # This program is distributed in the hope that it will be useful, but
// # WITHOUT ANY WARRANTY; without even the IMPLIED WARRANTY OF
// # MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the terms and
// # conditions of the GNU General Public License for more details. 
// # 
// # You should have received a copy of the GNU General Public License
// # along with this program; if not, write to the Free Software
// # Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307, USA 
#ifndef GRID_POINT_SOURCE_H
#define GRID_POINT_SOURCE_H

#include <iostream>
#include <vector>
#include <string>
//#include "TimeDep.h"
#include "Source.h"
#include "Sarray.h"
//#include "Filter.h"
#include "sw4.h"
#include "sw4raja.h"
#include "time_functions.h"

class Managed {
public:
  //static size_t mem_total;
  static int host;
  static int device;
  Managed(){
  }
  ~Managed(){
    //mem_total=0;
  }
#if defined(CUDA_CODE)
  void *operator new(size_t len) {
    void *ptr;
    //mem_total+=len;
    //std::cout<<"Total mem is now "<<mem_total/1024/1024<<" MB \n";
    //std::cout<<"Call to Managed class "<<len<<"\n";
    ptr = cl::sycl::malloc_shared(len, *QU::qu);
//    SW4_CheckDeviceError(cudaMallocManaged(&ptr, len));

    //SW4_CheckDeviceError(cudaDeviceSynchronize());
    return ptr;
  }

  void *operator new[](size_t len) {
    void *ptr;
    //mem_total+=len;
    //std::cout<<"Total mem is now "<<mem_total/1204/1024<<" MB \n";
    //std::cout<<"Call to Managed class "<<len<<"\n";
//    SW4_CheckDeviceError(cudaMallocManaged(&ptr, len));
    ptr = cl::sycl::malloc_shared(len, *QU::qu);
    //SW4_CheckDeviceError(cudaDeviceSynchronize());
    return ptr;
  }
  

  void operator delete(void *ptr) {
    //SW4_CheckDeviceError(cudaDeviceSynchronize());
//    SW4_CheckDeviceError(cudaFree(ptr));
    cl::sycl::free(ptr, *QU::qu);
  }

  void operator delete[](void *ptr) {
    //SW4_CheckDeviceError(cudaDeviceSynchronize());
//    SW4_CheckDeviceError(cudaFree(ptr));
    cl::sycl::free(ptr, *QU::qu);
  }
#endif
};

class GridPointSource:public Managed
{
   friend std::ostream& operator<<(std::ostream& output, const GridPointSource& s);
public:

  GridPointSource(float_sw4 frequency, float_sw4 t0,
		  int i0, int j0, int k0, int g,
		  float_sw4 Fx, float_sw4 Fy, float_sw4 Fz,
		  timeDep tDep, int ncyc, float_sw4* pars, int npar, int* ipars, int nipar,
		  float_sw4* devpars, int* devipars,
		  float_sw4* jacobian=NULL, float_sw4* dddp=NULL, float_sw4* hess1=NULL,
		  float_sw4* hess2=NULL, float_sw4* hess3=NULL );

 ~GridPointSource();

  int m_i0,m_j0,m_k0; // grid point index
  int m_grid;

  size_t m_key; // Key for sorting sources.

#ifdef SW4_CUDA
  __host__ __device__ void getFxyz( float_sw4 t, float_sw4* fxyz ) const;
  __host__ __device__ void getFxyztt( float_sw4 t, float_sw4* fxyz ) const;
#else
  RAJA_HOST_DEVICE
  SYCL_EXTERNAL
  void getFxyz( float_sw4 t, float_sw4* fxyz )  const;
  RAJA_HOST_DEVICE
  SYCL_EXTERNAL
  void getFxyztt( float_sw4 t, float_sw4* fxyz ) const;
#endif
  void getFxyz_notime( float_sw4* fxyz ) const;

  // evaluate time fcn: RENAME to evalTimeFunc
  float_sw4 getTimeFunc(float_sw4 t) const;
  float_sw4 evalTimeFunc_t(float_sw4 t) const;
  float_sw4 evalTimeFunc_tt(float_sw4 t) const;
  float_sw4 evalTimeFunc_ttt(float_sw4 t) const;
  float_sw4 evalTimeFunc_tttt(float_sw4 t) const;

  void limitFrequency(float_sw4 max_freq);

  void add_to_gradient( std::vector<Sarray>& kappa, std::vector<Sarray> & eta,
			float_sw4 t, float_sw4 dt, float_sw4 gradient[11], std::vector<float_sw4> & h,
			Sarray& Jac, bool topography_exists );
  void add_to_hessian( std::vector<Sarray> & kappa, std::vector<Sarray> & eta,
		       float_sw4 t, float_sw4 dt, float_sw4 hessian[121], std::vector<float_sw4> & h );
  void set_derivative( int der, const float_sw4 dir[11] );
  void set_noderivative( );
  void print_info() const;
  void set_sort_key( size_t key );
#ifdef SW4_CUDA
  __device__ void init_dev();
#endif
   //// discretize a time function at each time step and change the time function to be "Discrete()"
   //  void discretizeTimeFuncAndFilter(float_sw4 tStart, float_sw4 dt, int nSteps, Filter *filter_ptr);

 

  GridPointSource();
#ifdef SW4_CUDA
  __device__ void initializeTimeFunction();
#else
RAJA_HOST_DEVICE
  void initializeTimeFunction();
#endif
private:
  float_sw4 mForces[3];
   //  float_sw4 mAmp;
  float_sw4 mFreq, mT0;

  timeDep mTimeDependence;
/*  float_sw4 (*mTimeFunc)(float_sw4 f, float_sw4 t,float_sw4* par, int npar, int* ipar, int nipar );
  float_sw4 (*mTimeFunc_t)(float_sw4 f, float_sw4 t,float_sw4* par, int npar, int* ipar, int nipar );
  float_sw4 (*mTimeFunc_tt)(float_sw4 f, float_sw4 t,float_sw4* par, int npar, int* ipar, int nipar );
  float_sw4 (*mTimeFunc_ttt)(float_sw4 f, float_sw4 t,float_sw4* par, int npar, int* ipar, int nipar );
  float_sw4 (*mTimeFunc_om)(float_sw4 f, float_sw4 t,float_sw4* par, int npar, int* ipar, int nipar );
  float_sw4 (*mTimeFunc_omtt)(float_sw4 f, float_sw4 t,float_sw4* par, int npar, int* ipar, int nipar );
  float_sw4 (*mTimeFunc_tttt)(float_sw4 f, float_sw4 t,float_sw4* par, int npar, int* ipar, int nipar );
  float_sw4 (*mTimeFunc_tttom)(float_sw4 f, float_sw4 t,float_sw4* par, int npar, int* ipar, int nipar );
  float_sw4 (*mTimeFunc_ttomom)(float_sw4 f, float_sw4 t,float_sw4* par, int npar, int* ipar, int nipar );
  float_sw4 (*mTimeFunc_tom)(float_sw4 f, float_sw4 t,float_sw4* par, int npar, int* ipar, int nipar );
  float_sw4 (*mTimeFunc_omom)(float_sw4 f, float_sw4 t,float_sw4* par, int npar, int* ipar, int nipar );
*/

  float_sw4 mTimeFunc(float_sw4 f, float_sw4 t,float_sw4* par, int npar, int* ipar, int nipar ) const {
    switch(mTimeDependence)
    {
    case iRicker :
      return RickerWavelet(f, t, par, npar, ipar, nipar);
      break;
    case iGaussian :
      return Gaussian(f, t, par, npar, ipar, nipar);
      break;
    case iRamp :
      return Ramp(f, t, par, npar, ipar, nipar);
      break;
    case iTriangle :
      return Triangle(f, t, par, npar, ipar, nipar);
      break;
    case iSawtooth :
      return Sawtooth(f, t, par, npar, ipar, nipar);
      break;
    case iSmoothWave :
      return SmoothWave(f, t, par, npar, ipar, nipar);
      break;
    case iErf :
      return Erf(f, t, par, npar, ipar, nipar);
      break;
    case iVerySmoothBump :
      return VerySmoothBump(f, t, par, npar, ipar, nipar);
      break;
    case iRickerInt :
      return RickerInt(f, t, par, npar, ipar, nipar);
      break;
    case iBrune :
      return Brune(f, t, par, npar, ipar, nipar);
      break;
    case iBruneSmoothed :
      return BruneSmoothed(f, t, par, npar, ipar, nipar);
      break;
    case iDBrune :
      return DBrune(f, t, par, npar, ipar, nipar);
      break;
    case iGaussianWindow :
      return GaussianWindow(f, t, par, npar, ipar, nipar);
      break;
    case iLiu :
      return Liu(f, t, par, npar, ipar, nipar);
      break;
    case iDirac :
      return Dirac(f, t, par, npar, ipar, nipar);
      break;
    case iDiscrete :
      return Discrete(f, t, par, npar, ipar, nipar);
      break;
    case iDiscrete6moments :
      return Discrete(f, t, par, npar, ipar, nipar);
      break;
    case iC6SmoothBump :
      return C6SmoothBump(f, t, par, npar, ipar, nipar);
      break;
    default :
      return RickerWavelet(f, t, par, npar, ipar, nipar);
    }

  }

  float_sw4 mTimeFunc_t(float_sw4 f, float_sw4 t,float_sw4* par, int npar, int* ipar, int nipar ) const {
    switch(mTimeDependence)
    {
    case iRicker :
      return RickerWavelet_t(f, t, par, npar, ipar, nipar);
      break;
    case iGaussian :
      return Gaussian_t(f, t, par, npar, ipar, nipar);
      break;
    case iRamp :
      return Ramp_t(f, t, par, npar, ipar, nipar);
      break;
    case iTriangle :
      return Triangle_t(f, t, par, npar, ipar, nipar);
      break;
    case iSawtooth :
      return Sawtooth_t(f, t, par, npar, ipar, nipar);
      break;
    case iSmoothWave :
      return SmoothWave_t(f, t, par, npar, ipar, nipar);
      break;
    case iErf :
      return Erf_t(f, t, par, npar, ipar, nipar);
      break;
    case iVerySmoothBump :
      return VerySmoothBump_t(f, t, par, npar, ipar, nipar);
      break;
    case iRickerInt :
      return RickerInt_t(f, t, par, npar, ipar, nipar);
      break;
    case iBrune :
      return Brune_t(f, t, par, npar, ipar, nipar);
      break;
    case iBruneSmoothed :
      return BruneSmoothed_t(f, t, par, npar, ipar, nipar);
      break;
    case iDBrune :
      return DBrune_t(f, t, par, npar, ipar, nipar);
      break;
    case iGaussianWindow :
      return GaussianWindow_t(f, t, par, npar, ipar, nipar);
      break;
    case iLiu :
      return Liu_t(f, t, par, npar, ipar, nipar);
       break;
    case iDirac :
      return  Dirac_t(f, t, par, npar, ipar, nipar);
       break;
    case iDiscrete :
      return  Discrete_t(f, t, par, npar, ipar, nipar);
       break;
    case iDiscrete6moments :
       return Discrete_t(f, t, par, npar, ipar, nipar);
       break;
    case iC6SmoothBump :
      return C6SmoothBump_t(f, t, par, npar, ipar, nipar);
      break;
    default :
      return RickerWavelet_t(f, t, par, npar, ipar, nipar);
    }
  }

  float_sw4 mTimeFunc_tt(float_sw4 f, float_sw4 t,float_sw4* par, int npar, int* ipar, int nipar ) const {
    switch(mTimeDependence)
    {
    case iRicker :
      return RickerWavelet_tt(f, t, par, npar, ipar, nipar);
      break;
    case iGaussian :
      return Gaussian_tt(f, t, par, npar, ipar, nipar);
      break;
    case iRamp :
      return Ramp_tt(f, t, par, npar, ipar, nipar);
      break;
    case iTriangle :
      return Triangle_tt(f, t, par, npar, ipar, nipar);
      break;
    case iSawtooth :
      return Sawtooth_tt(f, t, par, npar, ipar, nipar);
      break;
    case iSmoothWave :
      return SmoothWave_tt(f, t, par, npar, ipar, nipar);
      break;
    case iErf :
      return Erf_tt(f, t, par, npar, ipar, nipar);
      break;
    case iVerySmoothBump :
      return VerySmoothBump_tt(f, t, par, npar, ipar, nipar);
      break;
    case iRickerInt :
      return RickerInt_tt(f, t, par, npar, ipar, nipar);
      break;
    case iBrune :
      return Brune_tt(f, t, par, npar, ipar, nipar);
      break;
    case iBruneSmoothed :
      return BruneSmoothed_tt(f, t, par, npar, ipar, nipar);
      break;
    case iDBrune :
      return DBrune_tt(f, t, par, npar, ipar, nipar);
      break;
    case iGaussianWindow :
      return GaussianWindow_tt(f, t, par, npar, ipar, nipar);
      break;
    case iLiu :
       return Liu_tt(f, t, par, npar, ipar, nipar);
       break;
    case iDirac :
       return Dirac_tt(f, t, par, npar, ipar, nipar);
       break;
    case iDiscrete :
      return  Discrete_tt(f, t, par, npar, ipar, nipar);
       break;
    case iDiscrete6moments :
       return Discrete_tt(f, t, par, npar, ipar, nipar);
       break;
    case iC6SmoothBump :
      return C6SmoothBump_tt(f, t, par, npar, ipar, nipar);
      break;
    default :
     return  RickerWavelet_tt(f, t, par, npar, ipar, nipar);
    }

  }

  float_sw4 mTimeFunc_ttt(float_sw4 f, float_sw4 t,float_sw4* par, int npar, int* ipar, int nipar ) const {
    switch(mTimeDependence)
    {
    case iRicker :
      return RickerWavelet_ttt(f, t, par, npar, ipar, nipar);
      break;
    case iGaussian :
      return Gaussian_ttt(f, t, par, npar, ipar, nipar);
      break;
    case iRamp :
      return Ramp_ttt(f, t, par, npar, ipar, nipar);
      break;
    case iTriangle :
      return Triangle_ttt(f, t, par, npar, ipar, nipar);
      break;
    case iSawtooth :
      return Sawtooth_ttt(f, t, par, npar, ipar, nipar);
      break;
    case iSmoothWave :
      return SmoothWave_ttt(f, t, par, npar, ipar, nipar);
      break;
    case iErf :
      return Erf_ttt(f, t, par, npar, ipar, nipar);
      break;
    case iVerySmoothBump :
      return VerySmoothBump_ttt(f, t, par, npar, ipar, nipar);
      break;
    case iRickerInt :
      return RickerInt_ttt(f, t, par, npar, ipar, nipar);
      break;
    case iBrune :
      return Brune_ttt(f, t, par, npar, ipar, nipar);
      break;
    case iBruneSmoothed :
      return BruneSmoothed_ttt(f, t, par, npar, ipar, nipar);
      break;
    case iDBrune :
      return DBrune_ttt(f, t, par, npar, ipar, nipar);
      break;
    case iGaussianWindow :
      return GaussianWindow_ttt(f, t, par, npar, ipar, nipar);
      break;
    case iLiu :
       return Liu_ttt(f, t, par, npar, ipar, nipar);
       break;
    case iDirac :
       return Dirac_ttt(f, t, par, npar, ipar, nipar);
       break;
    case iDiscrete :
       return Discrete_ttt(f, t, par, npar, ipar, nipar);
       break;
    case iDiscrete6moments :
       return Discrete_ttt(f, t, par, npar, ipar, nipar);
       break;
    case iC6SmoothBump :
      return C6SmoothBump_ttt(f, t, par, npar, ipar, nipar);
      break;
    default :
      return RickerWavelet_ttt(f, t, par, npar, ipar, nipar);
    }

  }

  float_sw4 mTimeFunc_om(float_sw4 f, float_sw4 t,float_sw4* par, int npar, int* ipar, int nipar ) const {
    switch(mTimeDependence)
    {
    case iRicker :
      return RickerWavelet_om(f, t, par, npar, ipar, nipar);
      break;
    case iGaussian :
      return Gaussian_om(f, t, par, npar, ipar, nipar);
      break;
    case iRamp :
      return Ramp_om(f, t, par, npar, ipar, nipar);
      break;
    case iTriangle :
      return Triangle_om(f, t, par, npar, ipar, nipar);
      break;
    case iSawtooth :
      return Sawtooth_om(f, t, par, npar, ipar, nipar);
      break;
    case iSmoothWave :
      return SmoothWave_om(f, t, par, npar, ipar, nipar);
      break;
    case iErf :
      return Erf_om(f, t, par, npar, ipar, nipar);
      break;
    case iVerySmoothBump :
      return VerySmoothBump_om(f, t, par, npar, ipar, nipar);
      break;
    case iRickerInt :
      return RickerInt_om(f, t, par, npar, ipar, nipar);
      break;
    case iBrune :
      return Brune_om(f, t, par, npar, ipar, nipar);
      break;
    case iBruneSmoothed :
      return BruneSmoothed_om(f, t, par, npar, ipar, nipar);
      break;
    case iDBrune :
      return DBrune_om(f, t, par, npar, ipar, nipar);
      break;
    case iGaussianWindow :
      return GaussianWindow_om(f, t, par, npar, ipar, nipar);
      break;
    case iLiu :
       return Liu_om(f, t, par, npar, ipar, nipar);
       break;
    case iDirac :
      return  Dirac_om(f, t, par, npar, ipar, nipar);
       break;
    case iDiscrete :
      return  Discrete_om(f, t, par, npar, ipar, nipar);
       break;
    case iDiscrete6moments :
       return Discrete_om(f, t, par, npar, ipar, nipar);
       break;
    case iC6SmoothBump :
      return C6SmoothBump_om(f, t, par, npar, ipar, nipar);
      break;
    default :
      return RickerWavelet_om(f, t, par, npar, ipar, nipar);
    }

  }

  float_sw4 mTimeFunc_omtt(float_sw4 f, float_sw4 t,float_sw4* par, int npar, int* ipar, int nipar ) const {
    switch(mTimeDependence)
    {
    case iRicker :
      return RickerWavelet_omtt(f, t, par, npar, ipar, nipar);
      break;
    case iGaussian :
      return Gaussian_omtt(f, t, par, npar, ipar, nipar);
      break;
    case iRamp :
      return Ramp_omtt(f, t, par, npar, ipar, nipar);
      break;
    case iTriangle :
      return Triangle_omtt(f, t, par, npar, ipar, nipar);
      break;
    case iSawtooth :
      return Sawtooth_omtt(f, t, par, npar, ipar, nipar);
      break;
    case iSmoothWave :
      return SmoothWave_omtt(f, t, par, npar, ipar, nipar);
      break;
    case iErf :
      return Erf_omtt(f, t, par, npar, ipar, nipar);
      break;
    case iVerySmoothBump :
      return VerySmoothBump_omtt(f, t, par, npar, ipar, nipar);
      break;
    case iRickerInt :
      return RickerInt_omtt(f, t, par, npar, ipar, nipar);
      break;
    case iBrune :
      return Brune_omtt(f, t, par, npar, ipar, nipar);
      break;
    case iBruneSmoothed :
      return BruneSmoothed_omtt(f, t, par, npar, ipar, nipar);
      break;
    case iDBrune :
      return DBrune_omtt(f, t, par, npar, ipar, nipar);
      break;
    case iGaussianWindow :
      return GaussianWindow_omtt(f, t, par, npar, ipar, nipar);
      break;
    case iLiu :
       return Liu_omtt(f, t, par, npar, ipar, nipar);
       break;
    case iDirac :
       return Dirac_omtt(f, t, par, npar, ipar, nipar);
       break;
    case iDiscrete :
       return Discrete_omtt(f, t, par, npar, ipar, nipar);
       break;
    case iDiscrete6moments :
       return Discrete_omtt(f, t, par, npar, ipar, nipar);
       break;
    case iC6SmoothBump :
      return C6SmoothBump_omtt(f, t, par, npar, ipar, nipar);
      break;
    default :
      return RickerWavelet_omtt(f, t, par, npar, ipar, nipar);
    }

  }

float_sw4 mTimeFunc_tttt(float_sw4 f, float_sw4 t,float_sw4* par, int npar, int* ipar, int nipar ) const {

  switch( mTimeDependence )
  {
  case iVerySmoothBump :
    return  VerySmoothBump_tttt(f, t, par, npar, ipar, nipar);
     break;
  case iGaussian :
     return Gaussian_tttt(f, t, par, npar, ipar, nipar);
     break;
  case iDirac :
     return Dirac_tttt(f, t, par, npar, ipar, nipar);
     break;
  case iDiscrete :
     return Discrete_tttt(f, t, par, npar, ipar, nipar);
     break;
  case iDiscrete6moments :
     return Discrete_tttt(f, t, par, npar, ipar, nipar);
     break;
  default: 
     return Gaussian_tttt(f, t, par, npar, ipar, nipar);
  }
}


float_sw4 mTimeFunc_tttom(float_sw4 f, float_sw4 t,float_sw4* par, int npar, int* ipar, int nipar ) const {

  switch( mTimeDependence )
  {
  case iVerySmoothBump :
     return VerySmoothBump_tttt(f, t, par, npar, ipar, nipar);
     break;
  case iGaussian :
     return Gaussian_tttt(f, t, par, npar, ipar, nipar);
     break;
  case iDirac :
     return Dirac_tttt(f, t, par, npar, ipar, nipar);
     break;
  case iDiscrete :
     return Discrete_tttt(f, t, par, npar, ipar, nipar);
     break;
  case iDiscrete6moments :
     return Discrete_tttt(f, t, par, npar, ipar, nipar);
     break;
  default:
     return Gaussian_tttt(f, t, par, npar, ipar, nipar);
  }
}
float_sw4 mTimeFunc_ttomom(float_sw4 f, float_sw4 t,float_sw4* par, int npar, int* ipar, int nipar ) const {

  switch( mTimeDependence )
  {
  case iVerySmoothBump :
     return VerySmoothBump_tttt(f, t, par, npar, ipar, nipar);
     break;
  case iGaussian :
     return Gaussian_tttt(f, t, par, npar, ipar, nipar);
     break;
  case iDirac :
     return Dirac_tttt(f, t, par, npar, ipar, nipar);
     break;
  case iDiscrete :
     return Discrete_tttt(f, t, par, npar, ipar, nipar);
     break;
  case iDiscrete6moments :
     return Discrete_tttt(f, t, par, npar, ipar, nipar);
     break;
  default:
     return Gaussian_tttt(f, t, par, npar, ipar, nipar);
  }
}
float_sw4 mTimeFunc_tom(float_sw4 f, float_sw4 t,float_sw4* par, int npar, int* ipar, int nipar ) const {

  switch( mTimeDependence )
  {
  case iVerySmoothBump :
     return VerySmoothBump_tttt(f, t, par, npar, ipar, nipar);
     break;
  case iGaussian :
     return Gaussian_tttt(f, t, par, npar, ipar, nipar);
     break;
  case iDirac :
     return Dirac_tttt(f, t, par, npar, ipar, nipar);
     break;
  case iDiscrete :
     return Discrete_tttt(f, t, par, npar, ipar, nipar);
     break;
  case iDiscrete6moments :
     return Discrete_tttt(f, t, par, npar, ipar, nipar);
     break;
  default:
     return Gaussian_tttt(f, t, par, npar, ipar, nipar);
  }
}
float_sw4 mTimeFunc_omom(float_sw4 f, float_sw4 t,float_sw4* par, int npar, int* ipar, int nipar ) const {

  switch( mTimeDependence )
  {
  case iVerySmoothBump :
     return VerySmoothBump_tttt(f, t, par, npar, ipar, nipar);
     break;
  case iGaussian :
     return Gaussian_tttt(f, t, par, npar, ipar, nipar);
     break;
  case iDirac :
     return Dirac_tttt(f, t, par, npar, ipar, nipar);
     break;
  case iDiscrete :
     return Discrete_tttt(f, t, par, npar, ipar, nipar);
     break;
  case iDiscrete6moments :
     return Discrete_tttt(f, t, par, npar, ipar, nipar);
     break;
  default:
     return Gaussian_tttt(f, t, par, npar, ipar, nipar);
  }
}

  float_sw4* mPar;
  int* mIpar; 
  float_sw4* mdevPar; //GPU copy 
  int* mdevIpar; // GPU copy
  int  mNpar, mNipar;

  int mNcyc;
   //  float_sw4 m_min_exponent;
  int m_derivative;
  bool m_jacobian_known;
  float_sw4 m_jacobian[27];
  bool m_hessian_known;
  float_sw4 m_hesspos1[9], m_hesspos2[9], m_hesspos3[9], m_dddp[9]; 
  float_sw4 m_dir[11];
};

#endif
