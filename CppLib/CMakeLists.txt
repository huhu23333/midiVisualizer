cmake_minimum_required(VERSION 3.14)
project(midi_render_backend LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(OpenCV REQUIRED)

set(SOURCES
    src/CvFuncs.cpp
    src/PianoRender.cpp
    src/NotesRender.cpp
    src/ParticleRender.cpp
    src/MidiPianoRender.cpp
    src/bindings.cpp
)

set(HEADERS
    include/render_backend.h
    src/CvFuncs.h
    src/PianoRender.h
    src/NotesRender.h
    src/ParticleRender.h
    src/MidiPianoRender.h
)

add_library(midi_render_backend SHARED ${SOURCES})

target_include_directories(midi_render_backend
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(midi_render_backend ${OpenCV_LIBS})