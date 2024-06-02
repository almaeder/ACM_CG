CC = hipcc 
CCFLAGS = --offload-arch=gfx90a --std=c++17 -O3 -I"${MPICH_DIR}/include"  -I"/opt/rocm-5.2.3/include/"
CCFLAGS += -fopenmp  -w
CCFLAGS += -DUSE_CUDA
LDFLAGS = -L"${MPICH_DIR}/lib" -lmpi -L"/opt/rocm-5.2.3/lib/" -lhipblas -lhipsparse -lhipsolver  -L"/opt/rocm-5.2.3/rocprim/lib" -lrocsparse -lrocsolver -lrocblas

SRC_DIR = src
TEST_DIR = test
BUILD_DIR = build

SOURCES = $(wildcard $(SRC_DIR)/*.cpp) $(wildcard $(SRC_DIR)/*.cu)
SOURCES_TEST = $(wildcard $(TEST_DIR)/*.cpp) $(wildcard $(TEST_DIR)/*.cu)

CPP_SOURCES = $(filter %.cpp, $(SOURCES))
CU_SOURCES = $(filter %.cu, $(SOURCES))
CPP_SOURCES_TEST = $(filter %.cpp, $(SOURCES_TEST))
CU_SOURCES_TEST = $(filter %.cu, $(SOURCES_TEST))

CPP_OBJ_FILES = $(patsubst $(SRC_DIR)/%.cpp, $(BUILD_DIR)/%.o, $(CPP_SOURCES))
CU_OBJ_FILES = $(patsubst $(SRC_DIR)/%.cu, $(BUILD_DIR)/%.o, $(CU_SOURCES))
CPP_OBJ_FILES_TEST = $(patsubst $(TEST_DIR)/%.cpp, $(BUILD_DIR)/%.o, $(CPP_SOURCES_TEST))
CU_OBJ_FILES_TEST = $(patsubst $(TEST_DIR)/%.cu, $(BUILD_DIR)/%.o, $(CU_SOURCES_TEST))

CPP_OBJ_FILES_TEST := $(filter-out build/test.o, $(CPP_OBJ_FILES_TEST))
CPP_OBJ_FILES_TEST := $(filter-out build/test_split.o, $(CPP_OBJ_FILES_TEST))


.PHONY: all
all: $(CPP_OBJ_FILES) $(CU_OBJ_FILES)


.PHONY: test
test: build/test

build/test: $(CPP_OBJ_FILES) $(CU_OBJ_FILES) $(CPP_OBJ_FILES_TEST) $(CU_OBJ_FILES_TEST) build/test.o
	$(CC) $(CCFLAGS) $(CPP_OBJ_FILES) $(CU_OBJ_FILES) $(CPP_OBJ_FILES_TEST) $(CU_OBJ_FILES_TEST) build/test.o -o $@ $(LDFLAGS)

.PHONY: test_split
test_split: build/test_split

build/test_split: $(CPP_OBJ_FILES) $(CU_OBJ_FILES) $(CPP_OBJ_FILES_TEST) $(CU_OBJ_FILES_TEST) build/test_split.o
	$(CC) $(CCFLAGS) $(CPP_OBJ_FILES) $(CU_OBJ_FILES) $(CPP_OBJ_FILES_TEST) $(CU_OBJ_FILES_TEST) build/test_split.o -o $@ $(LDFLAGS)


# Rule for compiling C++ source files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(@D)
	$(CC) $(CCFLAGS) -c $< -o $@
$(BUILD_DIR)/%.o: $(TEST_DIR)/%.cpp
	@mkdir -p $(@D)
	$(CC) $(CCFLAGS) -c $< -o $@

# Rule for compiling CUDA source files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu
	@mkdir -p $(@D)
	$(CC) $(CCFLAGS) -c $< -o $@
$(BUILD_DIR)/%.o: $(TEST_DIR)/%.cu
	@mkdir -p $(@D)
	$(CC) $(CCFLAGS) -c $< -o $@

.PHONY: clean
clean:
	rm -r $(BUILD_DIR)
