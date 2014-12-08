LOCAL_PATH:= $(call my-dir)
include $(CLEAR_VARS)
LOCAL_MODULE:= quake
LOCAL_SRC_FILES:= libquake.a
include $(PREBUILT_STATIC_LIBRARY)
