LOCAL_PATH:= $(call my-dir)
include $(CLEAR_VARS)
LOCAL_MODULE:= chess
LOCAL_SRC_FILES:= libchess.a
include $(PREBUILT_STATIC_LIBRARY)
