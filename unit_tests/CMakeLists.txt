include(CTest)

include(FetchContent)
FetchContent_Declare(googletest
  GIT_REPOSITORY    https://github.com/google/googletest.git
  GIT_TAG           master
)
FetchContent_GetProperties(googletest)
if(NOT googletest_POPULATED)
  FetchContent_Populate(googletest)
  add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR}) 
endif()

SET(NAME KokkosRemote_TestAll)

if (Kokkos_ENABLE_MPISPACE)
  FILE(GLOB TEST_SRCS *.cpp) 
  list(FILTER TEST_SRCS EXCLUDE REGEX ".*Test_Subview\\.cpp$")
  message(${TEST_SRCS})
elseif(Kokkos_ENABLE_CUDARDMASPACE)
  file(GLOB TEST_SRCS experimental/*.cpp)
  list(APPEND TEST_SRCS Test_Main.cpp)
else()
  FILE(GLOB TEST_SRCS *.cpp) 
endif()

add_executable(${NAME} ${TEST_SRCS})
target_link_libraries(${NAME} PUBLIC kokkosremote)
target_link_libraries(${NAME} PUBLIC gtest_main)
