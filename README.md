CaffeLink
=========

Mathematica library link wrapper for [Caffe](https://github.com/BVLC/caffe)

This library allows using Caffe directly from [Mathematica](http://www.wolfram.com/mathematica/). CaffeLink can be also used as regular library from C++ applications. The interface and function calls are basicly the same as in Mathematica. 

### Installation
Assuming you have successfully build Caffe, you should have everything needed by CaffeLink.

1. Edit makefile as required
  * path to Mathematica C headers
  * path to Caffe and its headers
2. `make`
3. Copy or create link to `libcaffeLink.so` somewhere in `$LibraryPath`
  * eg: `/home/alfons/.Mathematica/SystemFiles/LibraryResources/Linux-x86-64/`
  * or: `/Users/alfons/Library/Mathematica/Applications/IPCU/LibraryResources/MacOSX-x86-64/`
4. Test installation with [liblink-test.nb](../master/module/demo/liblink-test.nb)

### Usage examples
* [LeNet](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf) MNIST example based on [Caffe example](https://github.com/BVLC/caffe/tree/master/examples/mnist)
  * Mathematica notebook: [mnist.nb](../master/module/demo/mnist.nb)
  * Pdf: [mnist.pdf](../master/module/demo/mnist.pdf)