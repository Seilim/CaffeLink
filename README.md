CaffeLink
=========

Mathematica library link wrapper for [Caffe](https://github.com/BVLC/caffe)

This library allows using Caffe directly from [Mathematica](http://www.wolfram.com/mathematica/). CaffeLink can be also used as regular library from C++ applications. The interface and function calls are basicly the same as in Mathematica.


##### RC2 Note
This branch `caffe_rc2` partially supports Caffe rc2 and newer. Everything
should work except generating rc2 definitions directly in Mathematica
(functions `newNet[]`, `newSolver[]`, and `getParamString[]`). Rc1 style still
can be used as Caffe automatically converts it to the rc2, but new layers can
not be defined this way. This works only with `prepareNetFile[]`, which is why
the examples are probably broken.

So, either use hand written rc2 style definitions in `prepareNetFile[]` or
`prepareNetString[]`. Or initialize CaffeLink with `caffe.proto_rc1`, generate
obsolete definitions in Mathematica, export them to a file and load with
`prepareNetFile[]`.


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

* [AlexNet](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks) ImageNet example based on [Caffe example](http://nbviewer.ipython.org/github/BVLC/caffe/blob/master/examples/filter_visualization.ipynb)
  * Mathematica notebook: [imageNet.nb](../master/module/demo/imageNet.nb)
  * Pdf: [imageNet.pdf](../master/module/demo/imageNet.pdf)
