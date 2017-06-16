package com.lucidfusionlabs.app;

public class NativeAPI {
    static { System.loadLibrary("native-lib"); }
    public static boolean created, init;
    public static native void create(MainActivity activity);
    public static native void main();
    public static native void newMainLoop(MainActivity activity, boolean reset);
    public static native void minimize();
    public static native void reshaped(int x, int y, int w, int h);
    public static native void keyPress(int keycode, int mod, int down);
    public static native void touch(int action, float x, float y, float p);
    public static native void fling(float x, float y, float vx, float vy);
    public static native void scroll(float x, float y, float sx, float sy);
    public static native void accel(float x, float y, float z);
    public static native void scale(float x, float y, float dx, float dy, boolean begin);
    public static native void shellRun(String text);
    public static native boolean getFrameEnabled();
}
