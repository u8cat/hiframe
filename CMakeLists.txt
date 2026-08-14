# CMakeLists.txt
# Copyright (c) 2025, 张子辰

# This file is part of HDR Image Frame.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

cmake_minimum_required(VERSION 3.14)
project(hiframe)

set(CMAKE_CXX_STANDARD 20)

find_package(OpenCV REQUIRED)
find_package(Freetype REQUIRED)
find_package(PkgConfig REQUIRED)

pkg_check_modules(EXIV2 REQUIRED exiv2)
pkg_check_modules(RSVG REQUIRED librsvg-2.0)

find_package(Boost 1.74 REQUIRED COMPONENTS program_options)

find_library(ULTRAHDR_LIB NAMES uhdr PATHS /usr/local/lib /opt/libultrahdr/lib /usr/lib)
find_path(ULTRAHDR_INCLUDE NAMES ultrahdr_api.h PATHS /usr/local/include /opt/libultrahdr/include /usr/include)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${FREETYPE_INCLUDE_DIRS}
    ${EXIV2_INCLUDE_DIRS}
    ${RSVG_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
    ${ULTRAHDR_INCLUDE}
)

aux_source_directory(src HIFRAME_SRC)

add_executable(hiframe ${HIFRAME_SRC})

target_link_libraries(hiframe
    ${OpenCV_LIBS}
    ${FREETYPE_LIBRARIES}
    ${EXIV2_LIBRARIES}
    ${RSVG_LIBRARIES}
    ${Boost_LIBRARIES}
    ${ULTRAHDR_LIB}
    jpeg
)

install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/logo"
    DESTINATION "${CMAKE_INSTALL_PREFIX}/share/${PROJECT_NAME}"
    FILES_MATCHING PATTERN "*.svg")

install(TARGETS hiframe
    DESTINATION "${CMAKE_INSTALL_PREFIX}/bin")
