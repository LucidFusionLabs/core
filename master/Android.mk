LOCAL_PATH:= $(call my-dir)
include $(CLEAR_VARS)
LOCAL_MODULE:= master
LOCAL_SRC_FILES:= libmaster.a
include $(PREBUILT_STATIC_LIBRARY)
