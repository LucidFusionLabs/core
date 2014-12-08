LOCAL_PATH:= $(call my-dir)
include $(CLEAR_VARS)
LOCAL_MODULE:= fv
LOCAL_SRC_FILES:= libfv.a
include $(PREBUILT_STATIC_LIBRARY)
