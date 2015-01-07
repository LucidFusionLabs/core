LOCAL_PATH:= $(call my-dir)
include $(CLEAR_VARS)
LOCAL_MODULE:= editor
LOCAL_SRC_FILES:= libeditor.a
include $(PREBUILT_STATIC_LIBRARY)
