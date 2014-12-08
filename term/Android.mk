LOCAL_PATH:= $(call my-dir)
include $(CLEAR_VARS)
LOCAL_MODULE:= term
LOCAL_SRC_FILES:= libterm.a
include $(PREBUILT_STATIC_LIBRARY)
