LOCAL_PATH:= $(call my-dir)
include $(CLEAR_VARS)
LOCAL_MODULE:= freetype
LOCAL_SRC_FILES:= objs/.libs/libfreetype.a
include $(PREBUILT_STATIC_LIBRARY)
