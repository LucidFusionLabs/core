LOCAL_PATH:= $(call my-dir)
include $(CLEAR_VARS)
LOCAL_MODULE:= lfapp
LOCAL_SRC_FILES:= libspaceball_lfapp.a
include $(PREBUILT_STATIC_LIBRARY)
