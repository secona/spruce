set(ADLET_SOURCE_DIR "${PROJECT_SOURCE_DIR}/third_party/adlet")
set(TACO_PROJECT_DIR "${PROJECT_SOURCE_DIR}/third_party/taco")

set(TACO_SRC_DIR "${TACO_PROJECT_DIR}/src")
set(TACO_INCLUDE_DIR "${TACO_PROJECT_DIR}/include")
add_subdirectory("${TACO_SRC_DIR}" taco EXCLUDE_FROM_ALL)
target_include_directories(taco SYSTEM PUBLIC "${TACO_INCLUDE_DIR}")
target_link_libraries(taco PRIVATE ${CMAKE_DL_LIBS})

add_library(adlet STATIC
  "${ADLET_SOURCE_DIR}/src/einsum.cpp"
  "${ADLET_SOURCE_DIR}/src/graph.cpp"
  "${ADLET_SOURCE_DIR}/src/node.cpp"
  "${ADLET_SOURCE_DIR}/src/tensor.cpp"
  "${ADLET_SOURCE_DIR}/src/utils.cpp"
  "${ADLET_SOURCE_DIR}/src/viz.cpp"
)
set(ADLET_INCLUDE_ROOT "${CMAKE_CURRENT_BINARY_DIR}/include")
file(MAKE_DIRECTORY "${ADLET_INCLUDE_ROOT}")
file(CREATE_LINK "${ADLET_SOURCE_DIR}/include" "${ADLET_INCLUDE_ROOT}/adlet"
  SYMBOLIC)
target_include_directories(adlet SYSTEM PUBLIC "${ADLET_INCLUDE_ROOT}")
target_compile_definitions(adlet PUBLIC SIZE_MACRO=2048)
target_link_libraries(adlet PUBLIC taco)
