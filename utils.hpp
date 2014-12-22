
#ifndef UTILS_HPP
#define	UTILS_HPP

extern "C" {

/**
 * Converts doubles: \c in to floats: \c out.
 * @param in input
 * @param out output
 * @param size size of arrays
 * @return \c true on success, \c false otherwise
 */
bool doublesToFloats(double* in, float** out, long size);

/**
 * Test of BLAS function cblas_sgemm.
 * 
 * I have trouble running Caffe (caffeLink) from Mathematica in CPU mode. There
 * is some problem with cblas_sggem() - Mathematica kernel just crashes after
 * calling this function. However in GPU mode everything is fine (no need for
 * cblas_sggem()).
 * 
 * I tried it with Atlas 3.8.4 (Debian version), Mathematica 10.0.1.0,
 * release candidate for Caffe 1.0 (Sep. 19 2014) and os: Debian 7.1 64x.
 * 
 * In case your kernel crushes in CPU mode, you can try calling this
 * function cblas_test and if it crushes again, then it is the mentioned
 * cblas_sggem.
 */
void cblasTest_();

}

#endif	/* UTILS_HPP */

