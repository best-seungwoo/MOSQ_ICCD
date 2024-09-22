cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
file(STRINGS "qiskit_aer/VERSION.txt" VERSION_NUM)

if(AER_THRUST_BACKEND STREQUAL "ROCM")
  if(DEFINED ENV{ROCM_PATH})
    set(ROCM_PATH "$ENV{ROCM_PATH}")
  else()
    set(ROCM_PATH "/opt/rocm")
  endif()

  if(NOT DEFINED CMAKE_HIP_COMPILER)
    if(DEFINED ENV{CMAKE_HIP_COMPILER})
      set(CMAKE_HIP_COMPILER "$ENV{CMAKE_HIP_COMPILER}")
    else()
      set(CMAKE_HIP_COMPILER "${ROCM_PATH}/llvm/bin/clang++")
    endif()
  endif()

  if(NOT DEFINED CMAKE_CXX_COMPILER)
    if(DEFINED ENV{CMAKE_CXX_COMPILER})
      set(CMAKE_CXX_COMPILER "$ENV{CMAKE_CXX_COMPILER}")
    else()
      set(CMAKE_CXX_COMPILER "${CMAKE_HIP_COMPILER}")
    endif()
  endif()
endif()

set(EXTRA_LANGUAGES "")
if(AER_THRUST_BACKEND STREQUAL "CUDA")
  list(APPEND EXTRA_LANGUAGES CUDA)
endif()

include(CheckLanguage)
project(qasm_simulator VERSION ${VERSION_NUM} LANGUAGES CXX C ${EXTRA_LANGUAGES})

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
list(APPEND CMAKE_PREFIX_PATH ${CMAKE_SOURCE_DIR}/cmake)

if(NOT DEFINED AER_BLAS_LIB_PATH AND DEFINED ENV{AER_BLAS_LIB_PATH})
	set(AER_BLAS_LIB_PATH $ENV{AER_BLAS_LIB_PATH})
endif()

if(NOT DEFINED AER_THRUST_BACKEND AND DEFINED ENV{AER_THRUST_BACKEND})
	set(AER_THRUST_BACKEND $ENV{AER_THRUST_BACKEND})
endif()

if(AER_THRUST_BACKEND STREQUAL "CUDA")
  include(nvcc_add_compiler_options)
  set(CUDA_FOUND TRUE)
  include(FindCUDA/select_compute_arch)
else()
  set(CMAKE_CUDA_COMPILE_WHOLE_COMPILATION "")
endif()

option(STATIC_LINKING "Specify if we want statically link the executable (for redistribution mainly)" FALSE)
option(BUILD_TESTS "Specify whether we want to build tests or not" FALSE)
option(USE_BUNDLED_BLAS_WIN "Use the bundled openblas library on Windows when not using AER_BLAS_LIB_PATH" TRUE)

if(NOT DEFINED DISABLE_CONAN AND DEFINED ENV{DISABLE_CONAN})
	set(DISABLE_CONAN $ENV{DISABLE_CONAN})
endif()

include(CTest)
include(compiler_utils)
include(Linter)
include(findBLASInSpecificPath)
include(dependency_utils)

get_version(${VERSION_NUM})
configure_file("${PROJECT_SOURCE_DIR}/contrib/standalone/version.hpp.in"
               "${PROJECT_SOURCE_DIR}/contrib/standalone/version.hpp")

set(AER_SIMULATOR_CPP_SRC_DIR "${PROJECT_SOURCE_DIR}/src")
set(AER_SIMULATOR_CPP_EXTERNAL_LIBS
	"${USER_LIB_PATH}")

set(AER_COMPILER_DEFINITIONS "")

if(WIN32)
	set(CMAKE_PREFIX_PATH "${AER_SIMULATOR_CPP_EXTERNAL_LIBS} ${CMAKE_PREFIX_PATH}")
endif()

find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
	set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
	set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
endif(CCACHE_FOUND)

# Set default build type to Debug
if(NOT CMAKE_BUILD_TYPE)
	SET(CMAKE_BUILD_TYPE Debug CACHE STRING
		"Choose the type of build, options are: Debug Release"
		FORCE)
endif(NOT CMAKE_BUILD_TYPE)

if(APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
	enable_cxx_compiler_flag_if_supported("-stdlib=libc++")
endif()

if(STATIC_LINKING)
	if(APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
		message(WARNING "Clang on MacOS doesn't support some -static-* flags. Switching to dyn compilation...")
		unset(STATIC_LINKING)
	else()
	    if(NOT APPLE)
	        enable_cxx_compiler_flag_if_supported("-static")
	    endif()
	    enable_cxx_compiler_flag_if_supported("-static-libgcc")
	    enable_cxx_compiler_flag_if_supported("-static-libstdc++")
	endif()
endif()

if(NOT MSVC)
	if(CMAKE_SYSTEM_PROCESSOR MATCHES "ppc64le")
		enable_cxx_compiler_flag_if_supported("-mcpu=native")
	endif()
	enable_cxx_compiler_flag_if_supported("-pedantic")
	enable_cxx_compiler_flag_if_supported("-Wall")
	enable_cxx_compiler_flag_if_supported("-Wfloat-equal")
	enable_cxx_compiler_flag_if_supported("-Wundef")
	enable_cxx_compiler_flag_if_supported("-Wcast-align")
	enable_cxx_compiler_flag_if_supported("-Wwrite-strings")
	enable_cxx_compiler_flag_if_supported("-Wmissing-declarations")
	enable_cxx_compiler_flag_if_supported("-Wredundant-decls")
	enable_cxx_compiler_flag_if_supported("-Wshadow")
	enable_cxx_compiler_flag_if_supported("-Woverloaded-virtual")
else("Windows general compiler flags")
	enable_cxx_compiler_flag_if_supported("/Oi")
	enable_cxx_compiler_flag_if_supported("/bigobj")
endif()

if(STATIC_LINKING)
    SET(CMAKE_FIND_LIBRARY_SUFFIXES .a ${CMAKE_FIND_LIBRARY_SUFFIXES})
    if(WIN32)
        SET(CMAKE_FIND_LIBRARY_SUFFIXES .lib ${CMAKE_FIND_LIBRARY_SUFFIXES})
    endif()
endif()

set(BACKEND_REDIST_DEPS "")
setup_dependencies()

set(AER_LINKER_FLAGS " ")
set(AER_COMPILER_FLAGS " ")
if(MSVC)
  set(AER_COMPILER_FLAGS " /bigobj")
endif ()

if(NOT OPENMP_FOUND)
	message(STATUS "Looking for OpenMP support...")
	find_package(OpenMP QUIET)
	if(OPENMP_FOUND)
		set(AER_COMPILER_FLAGS "${AER_COMPILER_FLAGS} ${OpenMP_CXX_FLAGS}")
		set(AER_LINKER_FLAGS "${AER_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS} ${OpenMP_CXX_FLAGS}")
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
		set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
		if(APPLE)
			set(AER_SIMULATOR_CPP_EXTERNAL_LIBS ${AER_SIMULATOR_CPP_EXTERNAL_LIBS} ${OpenMP_CXX_INCLUDE_DIRS})
			if(NOT SKBUILD)
				set(AER_LIBRARIES "${AER_LIBRARIES}" "${OpenMP_${OpenMP_CXX_LIB_NAMES}_LIBRARY}")
				message(STATUS "Adding Clang: ${OpenMP_${OpenMP_CXX_LIB_NAMES}_LIBRARY}")
			else()
				get_filename_component(OPENMP_LIB_TO_COPY ${OpenMP_${OpenMP_CXX_LIB_NAMES}_LIBRARY} REALPATH)
				set(BACKEND_REDIST_DEPS ${BACKEND_REDIST_DEPS} ${OPENMP_LIB_TO_COPY})
			endif()
		endif()
		message(STATUS "OpenMP found!")
		message(STATUS "OpenMP_CXX_FLAGS = ${OpenMP_CXX_FLAGS}")
		message(STATUS "OpenMP_EXE_LINKER_FLAGS = ${OpenMP_EXE_LINKER_FLAGS}")
	else()
		message(STATUS "WARNING: No OpenMP support found!")
	endif()
endif()

if(STATIC_LINKING)
	message(STATUS "Using static linking with Threads...")
	set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
	set(THREADS_PREFER_PTHREAD_FLAG True)
endif()
find_package(Threads)

if(STATIC_LINKING)
	message(STATUS "Setting BLA_STATIC")
	set(BLA_STATIC TRUE)
endif()

if(AER_BLAS_LIB_PATH)
	find_BLAS_in_specific_path(${AER_BLAS_LIB_PATH})
else()
	if(APPLE)
		message(STATUS  "Looking for Apple BLAS & Lapack library...")
		set(BLA_VENDOR "Apple")
	else()
		message(STATUS "Looking for OpenBLAS library...")
		if(NOT BLA_VENDOR)
		  set(BLA_VENDOR "OpenBLAS")
		endif()
	endif()
	if(WIN32 AND USE_BUNDLED_BLAS_WIN)
		message(STATUS "Uncompressing OpenBLAS static library...")
		set(WIN_ARCH "win64" )
		if("${CMAKE_SIZEOF_VOID_P}" STREQUAL "4")
			set(WIN_ARCH "win32")
		endif()
		execute_process(COMMAND ${CMAKE_COMMAND} -E tar "xvfj" "${AER_SIMULATOR_CPP_SRC_DIR}/third-party/${WIN_ARCH}/lib/openblas.7z" WORKING_DIRECTORY  "${AER_SIMULATOR_CPP_SRC_DIR}/third-party/${WIN_ARCH}/lib/")
		set(BACKEND_REDIST_DEPS ${BACKEND_REDIST_DEPS} "${AER_SIMULATOR_CPP_SRC_DIR}/third-party/${WIN_ARCH}/lib/libopenblas.dll")
		set(BLAS_LIBRARIES "${AER_SIMULATOR_CPP_SRC_DIR}/third-party/${WIN_ARCH}/lib/libopenblas.dll.a")
		set(BLAS_FOUND True)
	else()
		find_package(BLAS QUIET)
	endif()
	if(NOT BLAS_FOUND)
		message(STATUS "OpenBLAS not found. Looking for any other BLAS & Lapack libraries...")
		unset(BLA_VENDOR)
		find_package(BLAS REQUIRED)
		find_package(LAPACK REQUIRED)
		set(BLAS_LIBRARIES "${BLAS_LIBRARIES};${LAPACK_LIBRARIES}")
	endif()
endif()

message(STATUS "BLAS library found: ${BLAS_LIBRARIES}")

if(CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL "x86_64" OR CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL "AMD64" OR CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL "amd64")
    if(APPLE OR UNIX)
        if (NOT CMAKE_OSX_ARCHITECTURES STREQUAL "arm64")
            set(SIMD_FLAGS_LIST "-mfma;-mavx2")
	        enable_cxx_compiler_flag_if_supported("-mpopcnt")
        endif()
	elseif(MSVC)
		set(SIMD_FLAGS_LIST "/arch:AVX2")
	endif()
endif()

set(AER_THRUST_SUPPORTED TRUE)
if(AER_THRUST_SUPPORTED)
	if(AER_THRUST_BACKEND STREQUAL "CUDA")
		message(STATUS "Thrust library: Looking for CUDA backend...")
		find_package(CUDA REQUIRED)
		message(STATUS "Thrust library: CUDA found!")
		if(NOT DEFINED AER_CUDA_ARCH)
			if(DEFINED ENV{AER_CUDA_ARCH})
				set(AER_CUDA_ARCH $ENV{AER_CUDA_ARCH})
			else()
				set(AER_CUDA_ARCH "Auto")
			endif()
		endif()
		cuda_select_nvcc_arch_flags(AER_CUDA_ARCH_FLAGS ${AER_CUDA_ARCH})

		string(REPLACE ";" " "  AER_CUDA_ARCH_FLAGS_EXPAND "${AER_CUDA_ARCH_FLAGS}")

		string(REGEX MATCHALL "sm_[0-9][0-9]" CUDA_SM "${AER_CUDA_ARCH_FLAGS}")
		string(REPLACE "sm_" ""  AER_CUDA_ARCHITECTURES "${CUDA_SM}")
		set(CMAKE_CUDA_ARCHITECTURES "${AER_CUDA_ARCHITECTURES}")
		message(STATUS "CMAKE_CUDA_ARCHITECTURES = ${CMAKE_CUDA_ARCHITECTURES}")

		if(NOT DEFINED AER_PYTHON_CUDA_ROOT AND DEFINED ENV{AER_PYTHON_CUDA_ROOT})
			set(AER_PYTHON_CUDA_ROOT $ENV{AER_PYTHON_CUDA_ROOT})
		endif()
		if(NOT DEFINED AER_CIBUILD AND DEFINED ENV{AER_CIBUILD})
			set(AER_CIBUILD $ENV{AER_CIBUILD})
		endif()
		if(AER_PYTHON_CUDA_ROOT)
			find_package(Python)
			if(AER_CIBUILD)
				set(PYTHON_SITE_PATH ${AER_PYTHON_CUDA_ROOT}/cpython-${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}.${Python_VERSION_PATCH}/lib/python${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}/site-packages)
				set(STR_ORIGIN "\$$ORIGIN")
			else()
				set(PYTHON_SITE_PATH ${AER_PYTHON_CUDA_ROOT}/lib/python${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}/site-packages)
				set(STR_ORIGIN "$ORIGIN")
			endif()

			message(STATUS "PYTHON_SITE_PATH = ${PYTHON_SITE_PATH}")

			set(AER_COMPILER_DEFINITIONS ${AER_COMPILER_DEFINITIONS} THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA)
			set(AER_COMPILER_DEFINITIONS ${AER_COMPILER_DEFINITIONS} AER_CUSTATEVEC AER_CUTENSORNET)

			set(CUDA_USE_STATIC_CUDA_RUNTIME OFF CACHE INTERNAL "")
			set(CMAKE_CUDA_RUNTIME_LIBRARY None)

			set(CUDA_NVCC_FLAGS "${AER_CUDA_ARCH_FLAGS_EXPAND} -DAER_THRUST_GPU -DAER_THRUST_CUDA -I${AER_SIMULATOR_CPP_SRC_DIR} -isystem ${AER_SIMULATOR_CPP_SRC_DIR}/third-party/headers -use_fast_math --expt-extended-lambda")
			set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -I${PYTHON_SITE_PATH}/cuquantum/include")
			set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -I${PYTHON_SITE_PATH}/cutensor/include")
			set(THRUST_DEPENDANT_LIBS "${THRUST_DEPENDANT_LIBS} -Wl,--disable-new-dtags")
			set(THRUST_DEPENDANT_LIBS "${THRUST_DEPENDANT_LIBS} -Wl,-rpath,'${STR_ORIGIN}/../../nvidia/cublas/lib'")
			set(THRUST_DEPENDANT_LIBS "${THRUST_DEPENDANT_LIBS} -Wl,-rpath,'${STR_ORIGIN}/../../nvidia/cusolver/lib'")
			set(THRUST_DEPENDANT_LIBS "${THRUST_DEPENDANT_LIBS} -Wl,-rpath,'${STR_ORIGIN}/../../nvidia/cusparse/lib'")
			set(THRUST_DEPENDANT_LIBS "${THRUST_DEPENDANT_LIBS} -Wl,-rpath,'${STR_ORIGIN}/../../nvidia/cuda_runtime/lib'")
			set(THRUST_DEPENDANT_LIBS "${THRUST_DEPENDANT_LIBS} -Wl,-rpath,'${STR_ORIGIN}/../../cutensor/lib'")
			set(THRUST_DEPENDANT_LIBS "${THRUST_DEPENDANT_LIBS} -Wl,-rpath,'${STR_ORIGIN}/../../cuquantum/lib'")
			set(THRUST_DEPENDANT_LIBS "${THRUST_DEPENDANT_LIBS} -Wl,--no-as-needed,-l:libcusolver.so.11")
			set(THRUST_DEPENDANT_LIBS "${THRUST_DEPENDANT_LIBS} -Wl,--no-as-needed,-l:libcusparse.so.${CUDA_VERSION_MAJOR}")
			set(THRUST_DEPENDANT_LIBS "${THRUST_DEPENDANT_LIBS} -L${PYTHON_SITE_PATH}/cuquantum/lib")
			set(THRUST_DEPENDANT_LIBS "${THRUST_DEPENDANT_LIBS} -Wl,--no-as-needed,-l:libcustatevec.so.1")
			set(THRUST_DEPENDANT_LIBS "${THRUST_DEPENDANT_LIBS} -Wl,--no-as-needed,-l:libcutensornet.so.2")

			if(CUDA_VERSION_MAJOR STREQUAL "11")
				set(THRUST_DEPENDANT_LIBS "${THRUST_DEPENDANT_LIBS} -Wl,--no-as-needed,-l:libcudart.so.${CUDA_VERSION_MAJOR}.0")
			else()
				set(THRUST_DEPENDANT_LIBS "${THRUST_DEPENDANT_LIBS} -Wl,--no-as-needed,-l:libcudart.so.${CUDA_VERSION_MAJOR},-as-needed")
			endif()
			set(THRUST_DEPENDANT_LIBS "${THRUST_DEPENDANT_LIBS} -Wl,--as-needed")

			string(STRIP ${CUDA_NVCC_FLAGS} CUDA_NVCC_FLAGS)
			string(STRIP ${THRUST_DEPENDANT_LIBS} THRUST_DEPENDANT_LIBS)
		else()
			set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} ${AER_CUDA_ARCH_FLAGS_EXPAND} -DAER_THRUST_GPU -DAER_THRUST_CUDA -I${AER_SIMULATOR_CPP_SRC_DIR} -isystem ${AER_SIMULATOR_CPP_SRC_DIR}/third-party/headers -use_fast_math --expt-extended-lambda")

			set(AER_COMPILER_DEFINITIONS ${AER_COMPILER_DEFINITIONS} THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA)
			set(THRUST_DEPENDANT_LIBS "-L${CUDA_TOOLKIT_ROOT_DIR}/lib64")
			if(NOT DEFINED CUQUANTUM_ROOT)
				if(DEFINED ENV{CUQUANTUM_ROOT})
					set(CUQUANTUM_ROOT $ENV{CUQUANTUM_ROOT})
				endif()
			endif()
			if(NOT DEFINED CUTENSOR_ROOT)
				if(DEFINED ENV{CUTENSOR_ROOT})
					set(CUTENSOR_ROOT $ENV{CUTENSOR_ROOT})
				endif()
			endif()
			if(NOT DEFINED AER_ENABLE_CUQUANTUM)
				if(DEFINED ENV{AER_ENABLE_CUQUANTUM})
					set(AER_ENABLE_CUQUANTUM $ENV{AER_ENABLE_CUQUANTUM})
				endif()
			endif()

			if(AER_ENABLE_CUQUANTUM)
				set(AER_COMPILER_DEFINITIONS ${AER_COMPILER_DEFINITIONS} AER_CUSTATEVEC AER_CUTENSORNET)
				if(DEFINED CUQUANTUM_ROOT)
					set(AER_COMPILER_FLAGS "${AER_COMPILER_FLAGS} -I${CUQUANTUM_ROOT}/include")
					set(THRUST_DEPENDANT_LIBS "${THRUST_DEPENDANT_LIBS} -L${CUQUANTUM_ROOT}/lib/${CUDA_VERSION_MAJOR}")
				endif()
				if(DEFINED CUTENSOR_ROOT)
					set(AER_COMPILER_FLAGS "${AER_COMPILER_FLAGS} -I${CUTENSOR_ROOT}/include")
					set(THRUST_DEPENDANT_LIBS "${THRUST_DEPENDANT_LIBS} -L${CUTENSOR_ROOT}/lib/${CUDA_VERSION_MAJOR}")
				endif()
       	     if(CUQUANTUM_STATIC)
					set(THRUST_DEPENDANT_LIBS "${THRUST_DEPENDANT_LIBS} -lcustatevec_static -lcutensornet_static  -lcutensor_static -lmetis_static -lcusolver_static -lcusparse_static -lcusolver_lapack_static -lcublas_static -lcublasLt_static -lculibos")
				else()
					set(THRUST_DEPENDANT_LIBS "${THRUST_DEPENDANT_LIBS} -lcustatevec -lcutensornet -lcutensor")
				endif()
			elseif(CUSTATEVEC_ROOT)
				set(AER_COMPILER_DEFINITIONS ${AER_COMPILER_DEFINITIONS} AER_CUSTATEVEC)
				set(AER_COMPILER_FLAGS "${AER_COMPILER_FLAGS} -I${CUSTATEVEC_ROOT}/include")
        	    if(CUSTATEVEC_STATIC)
					set(THRUST_DEPENDANT_LIBS "${THRUST_DEPENDANT_LIBS} -L${CUSTATEVEC_ROOT}/lib -L${CUSTATEVEC_ROOT}/lib/${CUDA_VERSION_MAJOR} -lcustatevec_static -lcusolver_static -lcusparse_static -lcusolver_lapack_static -lcublas_static -lcublasLt_static -lculibos")
				else()
					set(THRUST_DEPENDANT_LIBS "${THRUST_DEPENDANT_LIBS} -L${CUSTATEVEC_ROOT}/lib -L${CUSTATEVEC_ROOT}/lib/${CUDA_VERSION_MAJOR} -lcustatevec")
				endif()
			endif()
		endif()
	elseif(AER_THRUST_BACKEND STREQUAL "TBB")
		message(STATUS "TBB Support found!")
		set(THRUST_DEPENDENT_LIBS AER_DEPENDENCY_PKG::tbb)
		set(AER_COMPILER_DEFINITIONS ${AER_COMPILER_DEFINITIONS} AER_THRUST_CPU=TRUE)
	elseif(AER_THRUST_BACKEND STREQUAL "OMP")
		message(STATUS "Thrust library: Setting OMP backend")
		if(NOT OPENMP_FOUND)
			message(FATAL_ERROR "There's no OMP support. We cannot set Thrust backend to OMP!!")
		endif()
		set(AER_COMPILER_DEFINITIONS ${AER_COMPILER_DEFINITIONS} AER_THRUST_CPU=TRUE)
		set(THRUST_DEPENDENT_LIBS "")
	elseif(AER_THRUST_BACKEND STREQUAL "ROCM")
    if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
      message(FATAL_ERROR 
      "The compiler for ROCm builds must be Clang. Set CMAKE_CXX_COMPILER to <ROCm path>/llvm/bin/clang++")
    endif()
    
    set(CMAKE_CXX_FLAGS_DEBUG           "${CMAKE_CXX_FLAGS_DEBUG} -ggdb")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO  "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -ggdb")
    
    string(REPLACE " " ";" AER_ROCM_ARCH_LIST ${AER_ROCM_ARCH})
    set(GPU_TARGETS ${AER_ROCM_ARCH_LIST} CACHE INTERNAL "GPU targets to compile for")
    set(AMDGPU_TARGETS ${AER_ROCM_ARCH_LIST} CACHE INTERNAL "AMD GPU targets to compile for")
    set(CMAKE_HIP_ARCHITECTURES ${AER_ROCM_ARCH_LIST})
    
    mark_as_advanced(GPU_TARGETS)
    mark_as_advanced(AMDGPU_TARGETS)
    mark_as_advanced(CMAKE_HIP_ARCHITECTURES)
    message(STATUS "ROCm build targeting GPU Architectures: ${GPU_TARGETS}")
    
    message(STATUS "ROCm assumed path: ${ROCM_PATH}")
    list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH}/hip ${ROCM_PATH})
    list(APPEND CMAKE_MODULE_PATH ${ROCM_PATH}/hip/cmake ${ROCM_PATH})
    
    include(CheckLanguage)
    check_language(HIP)
    
    find_package(HIP CONFIG)
    if(HIP_FOUND)
      message(STATUS "Found HIP: " ${HIP_VERSION})
    else()
      message(FATAL_ERROR "Could not find HIP.")
    endif()

    list(APPEND AER_LIBRARIES hip::device)
    
    list(APPEND ROCM_EXTRA_DEFS AER_THRUST_GPU AER_THRUST_ROCM THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_HIP)
    
    list(TRANSFORM ROCM_EXTRA_DEFS PREPEND -D)
    add_definitions(${ROCM_EXTRA_DEFS})
    list(APPEND ROCM_EXTRA_FLAGS ${ROCM_EXTRA_DEFS})
    list(APPEND ROCM_EXTRA_FLAGS -isystem${ROCM_PATH}/include; -I${AER_SIMULATOR_CPP_SRC_DIR} ; -isystem${AER_SIMULATOR_CPP_SRC_DIR}/third-party/headers; -ffast-math; -fPIC)

    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
      list(APPEND ROCM_EXTRA_FLAGS -O0)
    else()
      list(APPEND ROCM_EXTRA_FLAGS -O3)
    endif()
    
    if(CMAKE_BUILD_TYPE STREQUAL "Debug" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
      list(APPEND ROCM_EXTRA_FLAGS -g; -ggdb)
    endif()
    
    list(APPEND ROCM_EXTRA_FLAGS -ferror-limit=3
                                 -Wno-unused-lambda-capture
                                 -Wno-bitwise-instead-of-logical
                                 -Wno-inconsistent-missing-override
                                 -Wno-cast-align
                                 -Wno-float-equal
                                 -Wno-unused-variable
                                 -Wno-unused-but-set-variable
                                 -Wno-switch
                                 -Wno-writable-strings
                                 -Wno-shadow
                                 -Wno-delete-non-abstract-non-virtual-dtor
                                 -Wno-pessimizing-move
                                 -Wno-return-type-c-linkage
                                 -Wno-overloaded-virtual
                                 -Wno-braced-scalar-init)
    
    if(AER_ENABLE_CUQUANTUM)
      message(WARNING "Implementation of cuQuantum is not available for ROCm builds.")
    endif()
	
	else()
		message(STATUS "No Thrust supported backend")
		set(AER_THRUST_SUPPORTED FALSE)
	endif()

	if(MSVC)
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /bigobj")
	endif()
endif()

if(AER_THRUST_SUPPORTED)
	set(AER_COMPILER_DEFINITIONS ${AER_COMPILER_DEFINITIONS} AER_THRUST_SUPPORTED=TRUE)
else()
	message(STATUS "No Thrust support enabled")
endif()

if(AER_DEBUG)
	set(AER_COMPILER_DEFINITIONS ${AER_COMPILER_DEFINITIONS} AER_DEBUG)
	set(AER_COMPILER_FLAGS "${AER_COMPILER_FLAGS} -g")
endif()

if(TEST_JSON)
	set(AER_COMPILER_DEFINITIONS ${AER_COMPILER_DEFINITIONS} TEST_JSON)
endif()

if(AER_MPI)
	find_package(MPI REQUIRED)
	set(AER_COMPILER_DEFINITIONS ${AER_COMPILER_DEFINITIONS} AER_MPI)
	set(AER_SIMULATOR_CPP_EXTERNAL_LIBS ${AER_SIMULATOR_CPP_EXTERNAL_LIBS} ${MPI_CXX_INCLUDE_PATH})
	set(MPI_DEPENDANT_LIBS ${MPI_CXX_LIBRARIES})
	if(AER_DISABLE_GDR)
	  set(AER_COMPILER_DEFINITIONS ${AER_COMPILER_DEFINITIONS} AER_DISABLE_GDR)
	endif()
else()
	set(MPI_DEPENDANT_LIBS "")
endif()

set(AER_LIBRARIES
	${AER_LIBRARIES}
	${BLAS_LIBRARIES}
	AER_DEPENDENCY_PKG::nlohmann_json
	AER_DEPENDENCY_PKG::spdlog
	Threads::Threads
	${CMAKE_DL_LIBS})

set(AER_COMPILER_DEFINITIONS ${AER_COMPILER_DEFINITIONS} ${CONAN_DEFINES})
if(SKBUILD)
	add_subdirectory(qiskit_aer/backends/wrappers)
else()
	set(AER_LIBRARIES
		${AER_LIBRARIES}
		${THRUST_DEPENDANT_LIBS}
		${MPI_DEPENDANT_LIBS})

	function(build_cuda target src_file is_exec)
		if(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64" OR CMAKE_SYSTEM_PROCESSOR STREQUAL "AMD64" OR CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL "amd64")
			if (NOT CMAKE_OSX_ARCHITECTURES STREQUAL "arm64")
				set(SIMD_SOURCE_FILE "${PROJECT_SOURCE_DIR}/src/simulators/statevector/qv_avx2.cpp")
			endif()
		endif()
		set_source_files_properties(${SIMD_SOURCE_FILE} PROPERTIES LANGUAGE CUDA)
		set_source_files_properties(${src_file} PROPERTIES LANGUAGE CUDA)
		set_source_files_properties(${src_file} PROPERTIES COMPILE_FLAGS "${CUDA_NVCC_FLAGS}")
		if(DEFINED SIMD_FLAGS_LIST)
			nvcc_add_compiler_options_list("${SIMD_FLAGS_LIST}" SIMD_FLAGS)
			set_source_files_properties(${SIMD_SOURCE_FILE} PROPERTIES COMPILE_FLAGS "${CUDA_NVCC_FLAGS} ${SIMD_FLAGS}")
		endif()
		if(${is_exec})
			add_executable(${target} ${src_file} ${SIMD_SOURCE_FILE})
		else()
			add_library(${target} ${src_file} ${SIMD_SOURCE_FILE})
		endif()
		target_link_libraries(${target} ${AER_LIBRARIES})
		string(STRIP ${AER_COMPILER_FLAGS} AER_COMPILER_FLAGS_STRIPPED)
		nvcc_add_compiler_options(${AER_COMPILER_FLAGS_STRIPPED} AER_COMPILER_FLAGS_OUT)

		set_target_properties(${target} PROPERTIES
			LINKER_LANGUAGE CXX
			CXX_STANDARD 14
			COMPILE_FLAGS ${AER_COMPILER_FLAGS_OUT}
			LINK_FLAGS ${AER_LINKER_FLAGS}
			RUNTIME_OUTPUT_DIRECTORY_DEBUG Debug
			RUNTIME_OUTPUT_DIRECTORY_RELEASE Release)
	endfunction()
	
	function(build_rocm target src_file is_exec)
  	set(SIMD_SOURCE_FILE "${PROJECT_SOURCE_DIR}/src/simulators/statevector/qv_avx2.cpp")
    
    set_source_files_properties(
      ${SIMD_SOURCE_FILE}
      ${src_file}
      PROPERTIES LANGUAGE CXX)
    
    if(${is_exec})
      add_executable(${target} ${src_file} ${SIMD_SOURCE_FILE})
    else()
      add_library(${target} ${src_file} ${SIMD_SOURCE_FILE})
    endif()
    
    target_compile_options(${target} PRIVATE ${ROCM_EXTRA_FLAGS} ${SIMD_FLAGS_LIST})
    target_compile_definitions(${target} PRIVATE ${ROCM_EXTRA_DEFS} ${AER_COMPILER_DEFINITIONS})
      
    target_link_libraries(${target} PRIVATE ${AER_LIBRARIES})

    set_target_properties(${target} PROPERTIES
      LINKER_LANGUAGE CXX
      CXX_STANDARD 14
      COMPILE_FLAGS ${AER_COMPILER_FLAGS}
      LINK_FLAGS ${AER_LINKER_FLAGS}
      RUNTIME_OUTPUT_DIRECTORY_DEBUG Debug
      RUNTIME_OUTPUT_DIRECTORY_RELEASE Release)
  endfunction()

	function(build_cpu target src_file is_exec)
		if(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64" OR CMAKE_SYSTEM_PROCESSOR STREQUAL "AMD64" OR CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL "amd64")
			if (NOT CMAKE_OSX_ARCHITECTURES STREQUAL "arm64")
				set(SIMD_SOURCE_FILE "${PROJECT_SOURCE_DIR}/src/simulators/statevector/qv_avx2.cpp")
			endif()
		endif()
		string(REPLACE ";" " " SIMD_FLAGS "${SIMD_FLAGS_LIST}")
		set_source_files_properties(${SIMD_SOURCE_FILE} PROPERTIES COMPILE_FLAGS "${SIMD_FLAGS}")
		if(${is_exec})
			add_executable(${target} ${src_file} ${SIMD_SOURCE_FILE})
		else()
			add_library(${target} SHARED ${src_file} ${SIMD_SOURCE_FILE})
		endif()
		target_link_libraries(${target} PRIVATE ${AER_LIBRARIES})
		set_target_properties(${target} PROPERTIES
			LINKER_LANGUAGE CXX
			CXX_STANDARD 14
			COMPILE_FLAGS ${AER_COMPILER_FLAGS}
			LINK_FLAGS ${AER_LINKER_FLAGS}
			RUNTIME_OUTPUT_DIRECTORY_DEBUG Debug
			RUNTIME_OUTPUT_DIRECTORY_RELEASE Release)

		target_include_directories(${target}
			PRIVATE ${AER_SIMULATOR_CPP_SRC_DIR}
			PRIVATE ${AER_SIMULATOR_CPP_EXTERNAL_LIBS})
		target_compile_definitions(${target}
			PRIVATE ${AER_COMPILER_DEFINITIONS})
		if(WIN32 AND NOT AER_BLAS_LIB_PATH)
			add_custom_command(TARGET ${target} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different
					${BACKEND_REDIST_DEPS}
					$<TARGET_FILE_DIR:${target}>)
			install(FILES ${BACKEND_REDIST_DEPS} DESTINATION bin)
		endif()
	endfunction()

	set(AER_SIMULATOR_SOURCE "${PROJECT_SOURCE_DIR}/contrib/standalone/qasm_simulator.cpp")
	set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
	if(CUDA_FOUND AND AER_THRUST_BACKEND STREQUAL "CUDA")
		build_cuda(qasm_simulator ${AER_SIMULATOR_SOURCE} TRUE)
	elseif(HIP_FOUND AND AER_THRUST_BACKEND STREQUAL "ROCM")
    build_rocm(qasm_simulator ${AER_SIMULATOR_SOURCE} TRUE)
	else()
		build_cpu(qasm_simulator ${AER_SIMULATOR_SOURCE} TRUE)
	endif()

	install(TARGETS qasm_simulator DESTINATION bin)

	if (CMAKE_SYSTEM_NAME STREQUAL "Linux" OR CMAKE_SYSTEM_NAME STREQUAL "Darwin")
		set(AER_RUNTIME_SOURCE "${PROJECT_SOURCE_DIR}/contrib/runtime/aer_runtime.cpp")
		if(CUDA_FOUND AND AER_THRUST_BACKEND STREQUAL "CUDA")
			build_cuda(aer ${AER_RUNTIME_SOURCE} FALSE)
		elseif(HIP_FOUND AND AER_THRUST_BACKEND STREQUAL "ROCM")
      build_rocm(aer ${AER_RUNTIME_SOURCE} FALSE)
		else()
			build_cpu(aer ${AER_RUNTIME_SOURCE} FALSE)
		endif()
		install(TARGETS aer)

		if(BUILD_TESTS AND NOT AER_MPI)
			add_executable(test_libaer "${PROJECT_SOURCE_DIR}/test/runtime/runtime_sample.c")
			target_include_directories(test_libaer PUBLIC "${PROJECT_SOURCE_DIR}/contrib/runtime/")
      set_target_properties(test_libaer PROPERTIES RUNTIME_OUTPUT_DIRECTORY_RELEASE bin
                                              		 LINK_FLAGS ${AER_LINKER_FLAGS})
			target_link_libraries(test_libaer PRIVATE ${AER_LIBRARIES})
			target_link_libraries(test_libaer PRIVATE aer)
			add_test(NAME aer_runtime_test COMMAND bin/test_libaer)
		endif()
	endif()

endif()

if(BUILD_TESTS)
	add_subdirectory(test)
endif()
