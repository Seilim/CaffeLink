
#ifndef BUILD_UTILS_HPP
#define	BUILD_UTILS_HPP

/* Mathematica library link necessary agruments. */
#define LIB_LINK_ARGS WolframLibraryData libData, mint Argc, MArgument *Args, MArgument Res

#ifndef INSTANTIATE_CLASS
// Instantiate a class with float and double specifications.
#define INSTANTIATE_CLASS(classname) \
  template class classname<float>; \
  template class classname<double>
#endif

#define TRAIN_NEW 0
#define TRAIN_SNAPSHOT 1
#define TRAIN_WEIGHTS 2

#define MSG_WRONG_ARGS "Wrong arguments, check stdout."

#endif	/* BUILD_UTILS_HPP */

