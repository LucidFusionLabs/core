LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

LOCAL_MODULE := lfjni

LOCAL_SRC_FILES := lfjni.cpp

LOCAL_C_INCLUDES := $(LOCAL_PATH)/../lfapp

LOCAL_LDLIBS := -lGLESv1_CM -llog -lz

LOCAL_STATIC_LIBRARIES := fv lfapp EASTL freetype avformat avcodec swscale avutil

include $(BUILD_SHARED_LIBRARY)
