LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)
LOCAL_MODULE := editor
LOCAL_SRC_FILES := $(LOCAL_PATH)/../../../android/editor/libeditor.a
include $(PREBUILT_STATIC_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := lfapp
LOCAL_SRC_FILES := $(LOCAL_PATH)/../../../android/editor/editor_lfapp_obj/libeditor_lfapp.a
include $(PREBUILT_STATIC_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := Box2D
LOCAL_SRC_FILES := $(LOCAL_PATH)/../../../android/imports/Box2D/Box2D/libBox2D.a
include $(PREBUILT_STATIC_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := libpng
LOCAL_SRC_FILES := $(LOCAL_PATH)/../../../android/imports/libpng/libpng.a
include $(PREBUILT_STATIC_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := libjpeg-turbo
LOCAL_SRC_FILES := $(LOCAL_PATH)/../../../android/imports/libjpeg-turbo/.libs/libturbojpeg.a
include $(PREBUILT_STATIC_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := lfjni
LOCAL_SRC_FILES := $(LOCAL_PATH)/../../../lfapp/lfjni/lfjni.cpp
LOCAL_LDLIBS := -lGLESv2 -lGLESv1_CM -llog -lz
LOCAL_STATIC_LIBRARIES := editor lfapp Box2D libpng libjpeg-turbo
include $(BUILD_SHARED_LIBRARY)

#include $(call all-subdir-makefiles)
