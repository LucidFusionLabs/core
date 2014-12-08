LOCAL_PATH:= $(call my-dir)
include $(CLEAR_VARS)
LOCAL_MODULE:= market
LOCAL_SRC_FILES:= libmarket.a
include $(PREBUILT_STATIC_LIBRARY)
