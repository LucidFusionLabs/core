LOCAL_PATH:= $(call my-dir)
include $(CLEAR_VARS)
LOCAL_MODULE:= spaceball
LOCAL_SRC_FILES:= libspaceball.a
include $(PREBUILT_STATIC_LIBRARY)
