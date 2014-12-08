LOCAL_PATH:= $(call my-dir)
include $(CLEAR_VARS)
LOCAL_MODULE:= image
LOCAL_SRC_FILES:= libimage.a
include $(PREBUILT_STATIC_LIBRARY)
