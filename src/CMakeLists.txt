set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

add_executable(compute_gt compute_gt.cpp)
target_link_libraries(compute_gt PRIVATE ${common_libraries})

# The 10M-base/1M-query MSMARCO release needs a batched exact ground-truth
# builder. Build it when Intel MKL and OpenMP are available; the rest of DRIP
# remains buildable without either dependency.
find_path(DRIP_MKL_INCLUDE_DIR mkl.h PATHS /usr/include/mkl /opt/intel/oneapi/mkl/latest/include)
find_library(DRIP_MKL_INTEL_LP64_LIBRARY mkl_intel_lp64)
find_library(DRIP_MKL_GNU_THREAD_LIBRARY mkl_gnu_thread)
find_library(DRIP_MKL_CORE_LIBRARY mkl_core)
find_package(OpenMP QUIET)

if(DRIP_MKL_INCLUDE_DIR AND DRIP_MKL_INTEL_LP64_LIBRARY AND DRIP_MKL_GNU_THREAD_LIBRARY AND DRIP_MKL_CORE_LIBRARY AND OpenMP_CXX_FOUND)
  add_executable(compute_gt_mkl compute_gt_mkl.cpp)
  target_include_directories(compute_gt_mkl PRIVATE ${DRIP_MKL_INCLUDE_DIR})
  target_link_libraries(compute_gt_mkl PRIVATE
    ${common_libraries}
    ${DRIP_MKL_INTEL_LP64_LIBRARY}
    ${DRIP_MKL_GNU_THREAD_LIBRARY}
    ${DRIP_MKL_CORE_LIBRARY}
    OpenMP::OpenMP_CXX
    pthread
    m
    dl)
else()
  message(WARNING "MKL/OpenMP not found; compute_gt_mkl will not be built.")
endif()

add_executable(query_generator_nearest query_generator_nearest.cpp)
target_link_libraries(query_generator_nearest PRIVATE ${common_libraries})

add_executable(test_drip test_drip.cpp)
target_link_libraries(test_drip PRIVATE ${common_libraries})

# add_executable(drip_qps_recall drip_qps_recall.cpp)
# target_link_libraries(drip_qps_recall PRIVATE ${common_libraries})
