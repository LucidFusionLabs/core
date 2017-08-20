package com.lucidfusionlabs.app;

import android.util.Log;

public class NativeAPI {
    static { System.loadLibrary("native-lib"); }
    public static boolean created, init;
    public static native void create(MainActivity activity);
    public static native void main(int w, int h, int opengles_version);
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
    public static native boolean getSuspended();
    public static native boolean getFrameEnabled();
    public static native void log(int level, String msg);
    public static void FATAL(String msg) { if (init) log(-1, msg); else Log.e("lfl", msg); }
    public static void ERROR(String msg) { if (init) log( 0, msg); else Log.e("lfl", msg); }
    public static void INFO (String msg) { if (init) log( 3, msg); else Log.i("lfl", msg); }
    public static void DEBUG(String msg) { if (init) log( 7, msg); else Log.i("lfl", msg); }
}
