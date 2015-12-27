# ########################################################################
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ########################################################################

cmake_minimum_required (VERSION 3.3.2)
project (XANN CXX)

#get TARGET_PLATFORM, which will be available by configuring this script
if(TARGET_PLATFORM EQUAL 32 OR TARGET_PLATFORM EQUAL 64)
	message("TARGET_PLATFORM has been set to ${TARGET_PLATFORM}")
    set(TARGET_PLATFORM ${TARGET_PLATFORM} CACHE STRING "Target platform type (32-bit or 64-bit)" FORCE)
else()
	message("TARGET_PLATFORM has not been set, deriving TARGET_PLATFORM from CMAKE_SIZEOF_VOID_P")
    if(CMAKE_SIZEOF_VOID_P MATCHES 8)
        set(TARGET_PLATFORM "64" CACHE STRING "Target platform type (32-bit or 64-bit)" FORCE)
    else()
        set(TARGET_PLATFORM "32" CACHE STRING "Target platform type (32-bit or 64-bit)" FORCE)
    endif()
endif()
message(STATUS "Target platform: ${TARGET_PLATFORM}-bit")

#set FIND_LIBRARY_USE_LIB64_PATHS depending on ${TARGET_PLATFORM}
if(TARGET_PLATFORM EQUAL 32)
    set(_arch "x86" INTERNAL)
    set_property(GLOBAL PROPERTY FIND_LIBRARY_USE_LIB64_PATHS FALSE)
else()
    set(_arch "x86_64" INTERNAL)
    set_property(GLOBAL PROPERTY FIND_LIBRARY_USE_LIB64_PATHS TRUE)
    if( NOT APPLE )
        set( SUFFIX_LIB_DEFAULT "64" )
    endif( )
endif()

#AMD's find opencl, modified to only find the import library. Its output, ${OPENCL_LIBRARIES}, is reliant on ${FIND_LIBRARY_USE_LIB64_PATHS}
set( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR} ) #so that our FindOpenCL.cmake can be found
find_package(OpenCL REQUIRED)
message("Import library found: " ${OPENCL_LIBRARIES})

#adds google test and google mock
add_subdirectory( ./googlemock )

#adds ReferenceMatrixTest
add_executable(ReferenceMatrixTest 
	"${PROJECT_SOURCE_DIR}/tests/ReferenceMatrix/MatrixTest.cpp" 
	"${PROJECT_SOURCE_DIR}/include/ReferenceMatrix/ReferenceMatrix.h" 
	"${PROJECT_SOURCE_DIR}/include/ReferenceMatrix/ReferenceMatrixBuilder.h")
target_include_directories(ReferenceMatrixTest PUBLIC
	"${PROJECT_SOURCE_DIR}/include/ReferenceMatrix"
	"${PROJECT_SOURCE_DIR}/googletest/include"
	"${PROJECT_SOURCE_DIR}/googlemock/include"
	)
set_target_properties(ReferenceMatrixTest 
	PROPERTIES LINK_FLAGS_RELEASE /LTCG
)
target_link_libraries(ReferenceMatrixTest gtest gmock gtest_main)

#adds ANN
add_executable(ANNTest
	"${PROJECT_SOURCE_DIR}/tests/ANN/ANNTest.cpp"
	"${PROJECT_SOURCE_DIR}/include/ANN/ANN.h" 
	)
target_include_directories(ANNTest PUBLIC
	"${PROJECT_SOURCE_DIR}/include/ReferenceMatrix"
	"${PROJECT_SOURCE_DIR}/include/ANN"
	"${PROJECT_SOURCE_DIR}/googletest/include"
	"${PROJECT_SOURCE_DIR}/googlemock/include"
	)
target_link_libraries(ANNTest gtest gmock gtest_main gtest_main)

#adds Examples/Common/MINST
add_executable(MINST
	#"${PROJECT_SOURCE_DIR}/tests/MINST/main.cpp"
	"${PROJECT_SOURCE_DIR}/tests/MINST/MinstDataTest.cpp"
	"${PROJECT_SOURCE_DIR}/tests/MINST/MinstData.h"
	)
target_include_directories(MINST PUBLIC
	"${PROJECT_SOURCE_DIR}/googletest/include"
	"${PROJECT_SOURCE_DIR}/googlemock/include"
	)
add_definitions(-DMINSTDATA_ROOT="${PROJECT_SOURCE_DIR}/tests/MINST/MINSTDataset/")
target_link_libraries(MINST gtest gmock gtest_main)

#add Examples/Common/CharacterRecognition
add_executable(CharacterRecognition
	"${PROJECT_SOURCE_DIR}/tests/CharacterRecognition/ANNToMINSTConverter.h"
	"${PROJECT_SOURCE_DIR}/tests/CharacterRecognition/CharacterRecognitionTest.cpp"
	)
target_include_directories(CharacterRecognition PUBLIC
	"${PROJECT_SOURCE_DIR}/include/ANN"
	"${PROJECT_SOURCE_DIR}/include/ReferenceMatrix"
	"${PROJECT_SOURCE_DIR}/tests/MINST"
	"${PROJECT_SOURCE_DIR}/googletest/include"
	"${PROJECT_SOURCE_DIR}/googlemock/include"
	)
add_definitions(-DMINSTDATA_ROOT="${PROJECT_SOURCE_DIR}/tests/MINST/MINSTDataset/")
target_link_libraries(CharacterRecognition gtest gmock gtest_main)
	
if(OPENCL_FOUND) #defined by find_package(OpenCL REQUIRED)
	#adds OpenCLMatrix
	add_executable(OpenCLMatrixTest 
		"${PROJECT_SOURCE_DIR}/tests/OpenCLMatrix/OpenCLMatrixTest.cpp"
		"${PROJECT_SOURCE_DIR}/include/OpenCLMatrix/OpenCLMatrix.h" 
		"${PROJECT_SOURCE_DIR}/include/OpenCLMatrix/OpenCLMatrixBuilder.h"
		)
	target_include_directories(OpenCLMatrixTest PUBLIC
		"${PROJECT_SOURCE_DIR}/include/ReferenceMatrix"
		"${PROJECT_SOURCE_DIR}/include/OpenCLMatrix"
		"${PROJECT_SOURCE_DIR}/include/OpenCLMatrix/khronos"
		"${PROJECT_SOURCE_DIR}/googletest/include"
		"${PROJECT_SOURCE_DIR}/googlemock/include"
		)
	add_definitions(-DKERNEL_FULL_PATH="${PROJECT_SOURCE_DIR}/include/OpenCLMatrix/test2.cl")
	target_link_libraries(OpenCLMatrixTest gtest gmock gtest_main ${OPENCL_LIBRARIES})
	
	
	#add Examples/Common/CharacterRecognition/CharacterRecognitionOpenCL
	add_executable(CharacterRecognitionOpenCL
		"${PROJECT_SOURCE_DIR}/tests/CharacterRecognition/ANNToMINSTConverter.h"
		"${PROJECT_SOURCE_DIR}/tests/CharacterRecognition/CharacterRecognitionOpenCLTest.cpp"
		)
	target_include_directories(CharacterRecognitionOpenCL PUBLIC
		"${PROJECT_SOURCE_DIR}/include/ANN"
		"${PROJECT_SOURCE_DIR}/include/ReferenceMatrix"
		"${PROJECT_SOURCE_DIR}/include/OpenCLMatrix"
		"${PROJECT_SOURCE_DIR}/include/OpenCLMatrix/khronos"
		"${PROJECT_SOURCE_DIR}/tests/MINST"
		"${PROJECT_SOURCE_DIR}/googletest/include"
		"${PROJECT_SOURCE_DIR}/googlemock/include"
		)
	add_definitions(-DMINSTDATA_ROOT="${PROJECT_SOURCE_DIR}/tests/MINST/MINSTDataset/")
	#add_definitions(-DKERNEL_FULL_PATH="${PROJECT_SOURCE_DIR}/include/OpenCLMatrix/test2.cl")
	target_link_libraries(CharacterRecognitionOpenCL gtest gmock gtest_main ${OPENCL_LIBRARIES})
endif()

#Visual Studio Specific
if (MSVC)
	set(ALL_NON_OPENCL_TARGETS gtest gtest gmock_main gtest_main ReferenceMatrixTest MINST ANNTest CharacterRecognition)
	set(ALL_OPENCL_TARGETS OpenCLMatrixTest CharacterRecognitionOpenCL)

	#Compile-options for all projects dependent on gtest should be the same
	if(NOT (MSVC_VERSION LESS 1900)) #VS2015 or later
		set(MSVC_DEBUG_OPTIONS_2015 /MTd -ZI)
		target_compile_options(gtest PUBLIC "$<$<CONFIG:Debug>:${MSVC_DEBUG_OPTIONS_2015}>")
	else()
		set(MSVC_DEBUG_OPTIONS_b4_2015 /MTd -Zi)
		target_compile_options(gtest PUBLIC "$<$<CONFIG:Debug>:${MSVC_DEBUG_OPTIONS_b4_2015}>")
	endif()
	set(MSVC_RELEASE_OPTIONS /MT -Zi)
	target_compile_options(gtest PUBLIC "$<$<CONFIG:Release>:${MSVC_RELEASE_OPTIONS}>")

	#Makes breakpoints work
	set_target_properties(${ALL_NON_OPENCL_TARGETS}
		PROPERTIES LINK_FLAGS "/DEBUG"
	)
	if(OPENCL_FOUND)
		set_target_properties(${ALL_OPENCL_TARGETS}
			PROPERTIES LINK_FLAGS "/DEBUG"
		)
	endif()
	
	#LTCG for release builds
	set_target_properties(${ALL_NON_OPENCL_TARGETS}
		PROPERTIES LINK_FLAGS_RELEASE "/LTCG"
	)
	if(OPENCL_FOUND)
		set_target_properties(${ALL_OPENCL_TARGETS}
			PROPERTIES LINK_FLAGS_RELEASE "/LTCG"
		)
	endif()
	
	#Folders within Visual Studio
	set_property(GLOBAL PROPERTY USE_FOLDERS ON)
	set_target_properties(gmock gtest gmock_main gtest_main
		PROPERTIES FOLDER Google/Tools
	)
	set_target_properties(MINST CharacterRecognition CharacterRecognitionOpenCL
		PROPERTIES FOLDER Examples/Common
	)
endif()