LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)
LOCAL_MODULE := chess
LOCAL_SRC_FILES := $(LOCAL_PATH)/../../../android/chess/libchess.a
include $(PREBUILT_STATIC_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := lfapp
LOCAL_SRC_FILES := $(LOCAL_PATH)/../../../android/chess/chess_lfapp_obj/libchess_lfapp.a
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
LOCAL_STATIC_LIBRARIES := chess lfapp Box2D libpng libjpeg-turbo
include $(BUILD_SHARED_LIBRARY)

#include $(call all-subdir-makefiles)
