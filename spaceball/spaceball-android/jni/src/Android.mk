LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

LOCAL_MODULE := lfjni

LOCAL_SRC_FILES := lfjni.cpp

LOCAL_C_INCLUDES := $(LOCAL_PATH)/../lfapp

LOCAL_LDLIBS := -lGLESv2 -lGLESv1_CM -llog -lz

LOCAL_STATIC_LIBRARIES := spaceball lfapp EASTL Box2D freetype protobuf libpng

include $(BUILD_SHARED_LIBRARY)
