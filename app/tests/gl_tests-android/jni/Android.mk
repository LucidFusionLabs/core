LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)
LOCAL_MODULE := app
LOCAL_SRC_FILES := $(LOCAL_PATH)/../../../../android/core/app_tests/libapp.so
include $(PREBUILT_SHARED_LIBRARY)

#include $(call all-subdir-makefiles)
