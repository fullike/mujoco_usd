cmake_minimum_required(VERSION 3.12)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set( PLUGIN_NAME hdTiny )

project(MjUsdHydra)
set(USD_DIR "/home/zy/github/OpenUSD/install" CACHE PATH "Path to an existing USD build directory.")
set(MUJOCO_DIR "/home/zy/github/mujoco/build" CACHE PATH "Path to an existing USD build directory.")
set(mujoco_BINARY_DIR "/home/zy/github/mujoco/build" CACHE PATH "Path to an existing USD build directory.")

set(gdt_dir ${PROJECT_SOURCE_DIR}/common/gdt/)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${gdt_dir}/cmake/")
include(${gdt_dir}/cmake/configure_build_type.cmake)
include(${gdt_dir}/cmake/configure_optix.cmake)
include_directories(${gdt_dir})
include_directories(common)
add_subdirectory(${gdt_dir} EXCLUDE_FROM_ALL)


list(PREPEND CMAKE_PREFIX_PATH "${USD_DIR}")
list(PREPEND CMAKE_PREFIX_PATH "${MUJOCO_DIR}")
list(APPEND RPATH_LIST "${USD_DIR}/lib")
find_package(glfw3 REQUIRED)
find_package(GLEW REQUIRED)
find_package(OpenGL REQUIRED)
find_package(TBB REQUIRED tbb)
find_package(pxr REQUIRED)
find_package(mujoco REQUIRED)

cuda_compile_and_embed(embedded_ptx_code devicePrograms.cu)

add_executable(MjUsdHydra main.cpp)

add_library(${PLUGIN_NAME} SHARED
    ${embedded_ptx_code}
    optix7.h
    CUDABuffer.h
    LaunchParams.h
    SampleRenderer.h
    SampleRenderer.cpp
    Model.h
    Model.cpp
    rendererPlugin.cpp
    rendererPlugin.h
    renderDelegate.cpp
    renderDelegate.h
    renderPass.cpp
    renderPass.h
    mesh.cpp
    mesh.h
)

set_target_properties(${PLUGIN_NAME} PROPERTIES PREFIX "")

target_link_directories(
    MjUsdHydra
    PRIVATE
    "${USD_DIR}/lib"
)
target_link_directories(
    MjUsdHydra
    PRIVATE
    "${MUJOCO_DIR}/lib"
)
target_link_libraries(MjUsdHydra
    usd_usdImagingGL
    usd_usdImaging
    usd_usdHydra
    usd_hdx
    usd_hdSt
    usd_hd
    usd_glf
    usd_garch
    usd_pxOsd
    usd_usdRi
    usd_usdUI
    usd_usdShade
    usd_usdGeom
    usd_usd
    usd_usdUtils
    usd_pcp
    usd_sdf
    usd_plug
    usd_js
    usd_ar
    usd_work
    usd_tf
    usd_kind
    usd_arch
    usd_vt
    usd_gf
    usd_hf
    usd_cameraUtil
    usd_usdLux
)
target_link_libraries(MjUsdHydra
    glfw
    GLEW::GLEW
    OpenGL::GL
    TBB::tbb
    #${PXR_LIBRARIES}
    mujoco
)

target_include_directories(${PLUGIN_NAME} PRIVATE
    ${PXR_INCLUDE_DIRS}
    /home/zy/github/mujoco/include
    ${OPENGL_INCLUDE_DIR}
    ${GLFW3_INCLUDE_DIR}
    ${GLEW_INCLUDE_DIR}
)

target_include_directories(MjUsdHydra PRIVATE
    ${PXR_INCLUDE_DIRS}
    /home/zy/github/mujoco/include
    ${OPENGL_INCLUDE_DIR}
    ${GLFW3_INCLUDE_DIR}
    ${GLEW_INCLUDE_DIR}
)

install(TARGETS ${PLUGIN_NAME} LIBRARY DESTINATION ${USD_DIR}/plugin/usd)
install(FILES ${CMAKE_SOURCE_DIR}/plugInfo.json DESTINATION ${USD_DIR}/plugin/usd/${PLUGIN_NAME}/resources )
