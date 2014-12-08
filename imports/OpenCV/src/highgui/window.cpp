/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "_highgui.h"

// in later times, use this file as a dispatcher to implementations like cvcap.cpp

CV_IMPL void cvSetWindowProperty(const char* name, int prop_id, double prop_value)
{
	switch(prop_id)
	{
		case CV_WND_PROP_FULLSCREEN://accept CV_WINDOW_NORMAL or CV_WINDOW_FULLSCREEN 
		
			if (!name || (prop_value!=CV_WINDOW_NORMAL && prop_value!=CV_WINDOW_FULLSCREEN))//bag argument
				break;
		
			#if   defined WIN32 || defined _WIN32 
			cvChangeMode_W32(name,prop_value);
			#elif defined (HAVE_GTK)
			cvChangeMode_GTK(name,prop_value);
			#elif defined (HAVE_CARBON)
			cvChangeMode_QT(name,prop_value);
			#endif
		break;
		
		case CV_WND_PROP_AUTOSIZE:
		
		break;
		
	default:;
	}
}

/* return -1 if error */
CV_IMPL double cvGetWindowProperty(const char* name, int prop_id)
{
	switch(prop_id)
	{
		case CV_WND_PROP_FULLSCREEN:
		
			if (!name)//bad argument
				return -1;
				
			#if   defined WIN32 || defined _WIN32 
				return cvGetMode_W32(name);
			#elif defined (HAVE_GTK)
				return cvGetMode_GTK(name);
			#elif defined (HAVE_CARBON)
				return cvGetMode_QT(name);
			#endif
		return -1;
		
		case CV_WND_PROP_AUTOSIZE:
		
			if (!name)//bad argument
				return -1;
				
		return -1;
		
	default:
		return -1;
	}
}

namespace cv
{

void namedWindow( const string& winname, int flags )
{
    cvNamedWindow( winname.c_str(), flags );
}

//YV
void setWindowProperty(const string& winname, int prop_id, double prop_value)
{
	cvSetWindowProperty( winname.c_str(),prop_id,prop_value);
}

//YV
double getWindowProperty(const string& winname, int prop_id)
{
	return  cvGetWindowProperty(winname.c_str(),prop_id);
}

void imshow( const string& winname, const Mat& img )
{
    CvMat _img = img;
    cvShowImage( winname.c_str(), &_img );
}

int waitKey(int delay)
{
    return cvWaitKey(delay);
}

int createTrackbar(const string& trackbarName, const string& winName,
                   int* value, int count, TrackbarCallback callback,
                   void* userdata)
{
    return cvCreateTrackbar2(trackbarName.c_str(), winName.c_str(),
                             value, count, callback, userdata);
}

void setTrackbarPos( const string& trackbarName, const string& winName, int value )
{
    cvSetTrackbarPos(trackbarName.c_str(), winName.c_str(), value );
}

int getTrackbarPos( const string& trackbarName, const string& winName )
{
	return cvGetTrackbarPos(trackbarName.c_str(), winName.c_str());
}

}

#if   defined WIN32 || defined _WIN32         // see window_w32.cpp
#elif defined (HAVE_GTK)      // see window_gtk.cpp
#elif defined (HAVE_COCOA)   // see window_carbon.cpp
#elif defined (HAVE_CARBON)


#else

// No windowing system present at compile time ;-(
// 
// We will build place holders that don't break the API but give an error
// at runtime. This way people can choose to replace an installed HighGUI
// version with a more capable one without a need to recompile dependent
// applications or libraries.


#define CV_NO_GUI_ERROR(funcname) \
    cvError( CV_StsError, funcname, \
    "The function is not implemented. " \
    "Rebuild the library with Windows, GTK+ 2.x or Carbon support. "\
    "If you are on Ubuntu or Debian, install libgtk2.0-dev and pkg-config, then re-run cmake or configure script", \
    __FILE__, __LINE__ )


CV_IMPL int cvNamedWindow( const char*, int )
{
    CV_NO_GUI_ERROR("cvNamedWindow");
    return -1;
}    

CV_IMPL void cvDestroyWindow( const char* )
{
    CV_NO_GUI_ERROR( "cvDestroyWindow" );
}

CV_IMPL void
cvDestroyAllWindows( void )
{
    CV_NO_GUI_ERROR( "cvDestroyAllWindows" );
}

CV_IMPL void
cvShowImage( const char*, const CvArr* )
{
    CV_NO_GUI_ERROR( "cvShowImage" );
}

CV_IMPL void cvResizeWindow( const char*, int, int )
{
    CV_NO_GUI_ERROR( "cvResizeWindow" );
}

CV_IMPL void cvMoveWindow( const char*, int, int )
{
    CV_NO_GUI_ERROR( "cvMoveWindow" );
}

CV_IMPL int
cvCreateTrackbar( const char*, const char*,
                  int*, int, CvTrackbarCallback )
{
    CV_NO_GUI_ERROR( "cvCreateTrackbar" );
    return -1;
}

CV_IMPL int
cvCreateTrackbar2( const char* trackbar_name, const char* window_name,
                   int* val, int count, CvTrackbarCallback2 on_notify2,
                   void* userdata )
{
    CV_NO_GUI_ERROR( "cvCreateTrackbar2" );
    return -1;
}

CV_IMPL void
cvSetMouseCallback( const char*, CvMouseCallback, void* )
{
    CV_NO_GUI_ERROR( "cvSetMouseCallback" );
}

CV_IMPL int cvGetTrackbarPos( const char*, const char* )
{
    CV_NO_GUI_ERROR( "cvGetTrackbarPos" );
    return -1;
}

CV_IMPL void cvSetTrackbarPos( const char*, const char*, int )
{
    CV_NO_GUI_ERROR( "cvSetTrackbarPos" );
}

CV_IMPL void* cvGetWindowHandle( const char* )
{
    CV_NO_GUI_ERROR( "cvGetWindowHandle" );
    return 0;
}
    
CV_IMPL const char* cvGetWindowName( void* )
{
    CV_NO_GUI_ERROR( "cvGetWindowName" );
    return 0;
}

CV_IMPL int cvWaitKey( int )
{
    CV_NO_GUI_ERROR( "cvWaitKey" );
    return -1;
}

CV_IMPL int cvInitSystem( int argc, char** argv )
{

    CV_NO_GUI_ERROR( "cvInitSystem" );
    return -1;
}

CV_IMPL int cvStartWindowThread()
{

    CV_NO_GUI_ERROR( "cvStartWindowThread" );
    return -1;
}

#endif

/* End of file. */
