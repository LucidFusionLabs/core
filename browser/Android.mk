LOCAL_PATH:= $(call my-dir)
include $(CLEAR_VARS)
LOCAL_MODULE:= browser
LOCAL_SRC_FILES:= libbrowser.a
include $(PREBUILT_STATIC_LIBRARY)
