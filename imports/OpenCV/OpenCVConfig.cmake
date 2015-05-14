# ===================================================================================
#  The OpenCV CMake configuration file
#
#             ** File generated automatically, do not modify **
#
#  Usage from an external project: 
#    In your CMakeLists.txt, add these lines:
#
#    FIND_PACKAGE(OpenCV REQUIRED )
#    TARGET_LINK_LIBRARIES(MY_TARGET_NAME ${OpenCV_LIBS})   
#
#    This file will define the following variables:
#      - OpenCV_LIBS          : The list of libraries to links against.
#      - OpenCV_LIB_DIR       : The directory where lib files are. Calling LINK_DIRECTORIES
#                                with this path is NOT needed.
#      - OpenCV_INCLUDE_DIRS  : The OpenCV include directories.
#      - OpenCV_VERSION       : The  version of this OpenCV build. Example: "1.2.0"
#      - OpenCV_VERSION_MAJOR : Major version part of OpenCV_VERSION. Example: "1"
#      - OpenCV_VERSION_MINOR : Minor version part of OpenCV_VERSION. Example: "2"
#      - OpenCV_VERSION_PATCH : Patch version part of OpenCV_VERSION. Example: "0"
#
# ===================================================================================


# Extract the directory where *this* file has been installed (determined at cmake run-time)
#  This variable may or may not be used below, depending on the parsing of OpenCVConfig.cmake
get_filename_component(THIS_OPENCV_CONFIG_PATH "${CMAKE_CURRENT_LIST_FILE}" PATH)

# ======================================================
# Include directories to add to the user project:
# ======================================================
INCLUDE_DIRECTORIES("${THIS_OPENCV_CONFIG_PATH}/include" "${THIS_OPENCV_CONFIG_PATH}/include/opencv")
# Provide the include directories to the caller
SET(OpenCV_INCLUDE_DIRS "${THIS_OPENCV_CONFIG_PATH}/include" "${THIS_OPENCV_CONFIG_PATH}/include/opencv")

# ======================================================
# Link directories to add to the user project:
# ======================================================
LINK_DIRECTORIES(""${THIS_OPENCV_CONFIG_PATH}/lib"")
# Provide the libs directory anyway, it may be needed in some cases.
SET(OpenCV_LIB_DIR ""${THIS_OPENCV_CONFIG_PATH}/lib"")

# ====================================================================
# Link libraries: e.g.   cxcore110.so, cv110.lib / cv110d.lib, etc...
# ====================================================================
set(OPENCV_LIB_COMPONENTS cxcore cv ml highgui cvaux)
SET(OpenCV_LIBS "")
foreach(__CVLIB ${OPENCV_LIB_COMPONENTS})
	# CMake>=2.6 supports the notation "debug XXd optimized XX"
	if (CMAKE_MAJOR_VERSION GREATER 2  OR  CMAKE_MINOR_VERSION GREATER 4)
		# Modern CMake:
		SET(OpenCV_LIBS ${OpenCV_LIBS} debug ${__CVLIB}210d optimized ${__CVLIB}210)
	else(CMAKE_MAJOR_VERSION GREATER 2  OR  CMAKE_MINOR_VERSION GREATER 4)
		# Old CMake:
		SET(OpenCV_LIBS ${OpenCV_LIBS} ${__CVLIB}210)
	endif(CMAKE_MAJOR_VERSION GREATER 2  OR  CMAKE_MINOR_VERSION GREATER 4)
endforeach(__CVLIB)

# For OpenCV built as static libs, we need the user to link against 
#  many more dependencies:
IF (NOT 1)
	# Under static libs, the user of OpenCV needs access to the 3rdparty libs as well:
	LINK_DIRECTORIES("C:/User/VP/ocv.build/vs2008ee/3rdparty/lib")

	SET(OpenCV_LIBS 
		 
		
		 
		 
		 
		 
		 
		${OpenCV_LIBS}) # The standard libs BEFORE (OpenCV_LIBS) since these can have "debug","optimized".
ENDIF(NOT 1)


# ======================================================
#  Version variables: 
# ======================================================
SET(OpenCV_VERSION 2.1.0)
SET(OpenCV_VERSION_MAJOR  2)
SET(OpenCV_VERSION_MINOR  1)
SET(OpenCV_VERSION_PATCH  0)
