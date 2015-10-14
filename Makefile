# Selects compiler.
CXX=g++
CXXFLAGS=-c

# If you do not have CUDA and/or compiled Caffe without GPU, uncomment this.
# CXXFLAGS+=-DCPU_ONLY

# Change this to your Mathematica installation path. Something like:
# <Mathematica_root>/SystemFiles/IncludeFiles/C
MATH_IDIR=/media/data/prog/mathematica-10/SystemFiles/IncludeFiles/C
# Change these 4 lines to point to your Caffe directory with builded library
# and headers.
CAFFE_DIR=../../fel-lin/caffe
CAFFE_IDIR=$(CAFFE_DIR)/include
CAFFE_IDIR2=$(CAFFE_DIR)/build/src
CAFFE_LDIR=$(CAFFE_DIR)/build/lib

CBLAS_LIB=
# To test whether cblas_sgemm() causes kernel to crash when called from
# Mathematica, uncomment next 2 lines to build with Cblas and complete
# cblas_test(). You neet to select proper cblas implementation.
# CBLAS_LIB=-lcblas -latlas
# CXXFLAGS+=-D CBLAS_TEST

DIST_DIR=dist
BUILD_DIR=build

IDIR=-I$(CAFFE_IDIR) -I$(CAFFE_IDIR2) -I$(MATH_IDIR)
LDIR=-L$(CAFFE_LDIR)
LIBS=$(CBLAS_LIB) -lcaffe -shared -fPIC

LDFLAGS=$(IDIR) $(LDIR) $(LIBS)
SOURCES=caffeLink.cpp CLnets.cpp utils.cpp\
  libLink_inputs.cpp libLink_outputs.cpp libLink_start.cpp
OBJECTS=$(SOURCES:%.cpp=$(BUILD_DIR)/%.o)
LIBNAME=libcaffeLink.so



all: dirs $(SOURCES) $(LIBNAME)

$(LIBNAME): $(OBJECTS)
	$(CXX) $(LDFLAGS) $(OBJECTS) -o $(DIST_DIR)/$@

$(BUILD_DIR)/%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $< -o $@

clean: clean.obj
	rm -rf $(DIST_DIR)/$(LIBNAME)

clean.obj:
	rm -rf $(OBJECTS)


dirs:
	mkdir -p $(DIST_DIR)
	mkdir -p $(BUILD_DIR)