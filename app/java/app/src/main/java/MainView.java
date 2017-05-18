package com.lucidfusionlabs.app;

import java.net.*;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import android.os.*;
import android.view.*;
import android.media.*;
import android.content.*;
import android.content.res.Configuration;
import android.content.res.Resources;
import android.content.pm.ActivityInfo;
import android.hardware.*;
import android.util.Log;

import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.egl.*;

class MyGestureListener extends android.view.GestureDetector.SimpleOnGestureListener {
    public MainActivity main_activity;

    @Override
    public boolean onFling(MotionEvent e1, MotionEvent e2, float velocityX, float velocityY) {
        main_activity.nativeFling(e1.getX(), e1.getY(), velocityX, velocityY);
        return false;
    }
}

public class MainView extends android.view.SurfaceView
    implements SurfaceHolder.Callback, View.OnKeyListener, View.OnTouchListener, SensorEventListener {
    public MainActivity main_activity;
    public EGLContext egl_context;
    public EGLSurface egl_surface;
    public EGLDisplay egl_display;
    public GestureDetector gesture;
    public ScaleGestureDetector scale_gesture = null;
    public SensorManager sensor;
    public boolean have_surface;

    public MainView(MainActivity activity, Context context) {
        super(context);
        main_activity = activity;
        sensor = (SensorManager)context.getSystemService("sensor");  
        // gesture = new GestureDetector(new MyGestureListener(this));
        setFocusable(true);
        setFocusableInTouchMode(true);
        setOnKeyListener(this); 
        setOnTouchListener(this);   
        getHolder().addCallback(this); 
        requestFocus();
    }

    @Override
    protected void onSizeChanged(int xNew, int yNew, int xOld, int yOld) {
        Log.i("lfl", "MainView.onSizeChanged(" + xNew + ", " + yNew + ", " + xOld + ", " + yOld + ")");
    }

    @Override
    public void surfaceCreated(SurfaceHolder holder) {
        Log.i("lfl", "MainView.surfaceCreated()");
        sensor.registerListener(this, sensor.getDefaultSensor(Sensor.TYPE_ACCELEROMETER), SensorManager.SENSOR_DELAY_GAME, null);
        have_surface = true;
    }

    @Override
    public void surfaceChanged(SurfaceHolder holder, int format, int width, int height) {
        Log.i("lfl", "MainView.surfaceChanged(" + width + ", " + height + ")");
        main_activity.surfaceChanged(format, width, height);
    }

    @Override
    public void surfaceDestroyed(SurfaceHolder holder) {
        Log.i("lfl", "surfaceDestroyed()");
        sensor.unregisterListener(this, sensor.getDefaultSensor(Sensor.TYPE_ACCELEROMETER));
        have_surface = false;
    }

    @Override
    public void onSensorChanged(SensorEvent event) {
        if (event.sensor.getType() == Sensor.TYPE_ACCELEROMETER)
            main_activity.nativeAccel(event.values[0], event.values[1], event.values[2]);
    }

    @Override
    public boolean onKey(View v, int key_code, KeyEvent event) {
        int key_char = event.getUnicodeChar(), mod = 0;
        if (key_char == 0) {
            int meta_state = event.getMetaState();
            if ((meta_state & KeyEvent.META_SHIFT_ON) != 0) mod |= 1;
            if ((meta_state & KeyEvent.META_CTRL_ON)  != 0) mod |= 2;
            if ((meta_state & KeyEvent.META_ALT_ON)   != 0) mod |= 4;

            switch(event.getKeyCode()) {
                case KeyEvent.KEYCODE_DEL:        key_char = '\b';   break;
                case KeyEvent.KEYCODE_ESCAPE:     key_char = 0xE100; break;
                case KeyEvent.KEYCODE_DPAD_UP:    key_char = 0xE101; break;
                case KeyEvent.KEYCODE_DPAD_DOWN:  key_char = 0xE102; break;
                case KeyEvent.KEYCODE_DPAD_LEFT:  key_char = 0xE103; break;
                case KeyEvent.KEYCODE_DPAD_RIGHT: key_char = 0xE104; break;
                case KeyEvent.KEYCODE_CTRL_LEFT:  key_char = 0xE105; break;
                case KeyEvent.KEYCODE_CTRL_RIGHT: key_char = 0xE106; break;
                case KeyEvent.KEYCODE_META_LEFT:  key_char = 0xE107; break;
                case KeyEvent.KEYCODE_META_RIGHT: key_char = 0xE108; break;
                case KeyEvent.KEYCODE_TAB:        key_char = 0xE109; break;
                case KeyEvent.KEYCODE_PAGE_UP:    key_char = 0xE10A; break;
                case KeyEvent.KEYCODE_PAGE_DOWN:  key_char = 0xE10B; break;
                case KeyEvent.KEYCODE_F1:         key_char = 0xE10C; break;
                case KeyEvent.KEYCODE_F2:         key_char = 0xE10D; break;
                case KeyEvent.KEYCODE_F3:         key_char = 0xE10E; break;
                case KeyEvent.KEYCODE_F4:         key_char = 0xE10F; break;
                case KeyEvent.KEYCODE_F5:         key_char = 0xE110; break;
                case KeyEvent.KEYCODE_F6:         key_char = 0xE111; break;
                case KeyEvent.KEYCODE_F7:         key_char = 0xE112; break;
                case KeyEvent.KEYCODE_F8:         key_char = 0xE113; break;
                case KeyEvent.KEYCODE_F9:         key_char = 0xE114; break;
                case KeyEvent.KEYCODE_A:          key_char = 'a';    break;
                case KeyEvent.KEYCODE_B:          key_char = 'b';    break;
                case KeyEvent.KEYCODE_C:          key_char = 'c';    break;
                case KeyEvent.KEYCODE_D:          key_char = 'd';    break;
                case KeyEvent.KEYCODE_E:          key_char = 'e';    break;
                case KeyEvent.KEYCODE_F:          key_char = 'f';    break;
                case KeyEvent.KEYCODE_G:          key_char = 'g';    break;
                case KeyEvent.KEYCODE_H:          key_char = 'h';    break;
                case KeyEvent.KEYCODE_I:          key_char = 'i';    break;
                case KeyEvent.KEYCODE_J:          key_char = 'j';    break;
                case KeyEvent.KEYCODE_K:          key_char = 'k';    break;
                case KeyEvent.KEYCODE_L:          key_char = 'l';    break;
                case KeyEvent.KEYCODE_M:          key_char = 'm';    break;
                case KeyEvent.KEYCODE_N:          key_char = 'n';    break;
                case KeyEvent.KEYCODE_O:          key_char = 'o';    break;
                case KeyEvent.KEYCODE_P:          key_char = 'p';    break;
                case KeyEvent.KEYCODE_Q:          key_char = 'q';    break;
                case KeyEvent.KEYCODE_R:          key_char = 'r';    break;
                case KeyEvent.KEYCODE_S:          key_char = 's';    break;
                case KeyEvent.KEYCODE_T:          key_char = 't';    break;
                case KeyEvent.KEYCODE_U:          key_char = 'u';    break;
                case KeyEvent.KEYCODE_V:          key_char = 'v';    break;
                case KeyEvent.KEYCODE_W:          key_char = 'w';    break;
                case KeyEvent.KEYCODE_X:          key_char = 'x';    break;
                case KeyEvent.KEYCODE_Y:          key_char = 'y';    break;
                case KeyEvent.KEYCODE_Z:          key_char = 'z';    break;
            }
        }
        if (key_char == 0) return false;
        else if (event.getAction() == KeyEvent.ACTION_UP)   { main_activity.nativeKeyPress(key_char, mod, 0); return true; }
        else if (event.getAction() == KeyEvent.ACTION_DOWN) { main_activity.nativeKeyPress(key_char, mod, 1); return true; }
        else return false;
    }

    @Override
    public boolean onTouch(View v, MotionEvent event) {
        // gesture.onTouchEvent(event);
        if (scale_gesture != null) scale_gesture.onTouchEvent(event);
        final int action = event.getAction() & MotionEvent.ACTION_MASK;
        if (action == MotionEvent.ACTION_MOVE) {
            if (scale_gesture == null || !scale_gesture.isInProgress()) {
                for (int i = 0; i < event.getPointerCount(); i++) {
                    main_activity.nativeTouch(action, event.getX(i), event.getY(i), event.getPressure(i));
                }
            }
        } else {
            int action_index = (action == MotionEvent.ACTION_POINTER_DOWN || action == MotionEvent.ACTION_POINTER_UP) ? event.getActionIndex() : 0;
            main_activity.nativeTouch(action, event.getX(action_index), event.getY(action_index), event.getPressure(action_index));
        }
        return true; 
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int accuracy) {}

    public void initEGL() {
        if      (initEGL(2)) main_activity.egl_version = 2;
        else if (initEGL(1)) main_activity.egl_version = 1;
        else                 main_activity.egl_version = 0;
    }

    public boolean initEGL(int requestedVersion) {
        try {
            final int EGL_OPENGL_ES2_BIT = 4, EGL_CONTEXT_CLIENT_VERSION = 0x3098;
            final int[] attrib_list1 = { EGL10.EGL_DEPTH_SIZE, 16, EGL10.EGL_NONE };
            final int[] attrib_list2 = { EGL10.EGL_DEPTH_SIZE, 16, EGL10.EGL_RENDERABLE_TYPE, EGL_OPENGL_ES2_BIT, EGL10.EGL_NONE };
            final int[] context_attrib2 = { EGL_CONTEXT_CLIENT_VERSION, 2, EGL10.EGL_NONE };
            int[] attrib_list = requestedVersion >= 2 ? attrib_list2 : attrib_list1;
            int[] context_attrib = requestedVersion >= 2 ? context_attrib2 : null;
            int[] version = new int[2], num_config = new int[1];

            EGLConfig[] configs = new EGLConfig[1];
            EGL10 egl = (EGL10)EGLContext.getEGL();
            egl_display = egl.eglGetDisplay(EGL10.EGL_DEFAULT_DISPLAY);
            egl.eglInitialize(egl_display, version);

            if (!egl.eglChooseConfig(egl_display, attrib_list, configs, 1, num_config) || num_config[0] == 0) throw new Exception("eglChooseConfig");
            if ((egl_context = egl.eglCreateContext(egl_display, configs[0], EGL10.EGL_NO_CONTEXT, context_attrib)) == EGL10.EGL_NO_CONTEXT) throw new Exception("eglCreateContext");
            if ((egl_surface = egl.eglCreateWindowSurface(egl_display, configs[0], this, null)) == EGL10.EGL_NO_SURFACE) throw new Exception("eglCreateWindowSurface");
            if (!egl.eglMakeCurrent(egl_display, egl_surface, egl_surface, egl_context)) throw new Exception("eglMakeCurrent");
            makeCurrentEGL();
        } catch(Exception e) {
            Log.e("lfl", e.toString());
            for (StackTraceElement ste : e.getStackTrace()) Log.e("lfl", ste.toString());
            egl_context = null;
            egl_surface = null;
            egl_display = null;
            return false;
        }
        return true;
    }

    public void makeCurrentEGL() {
        EGL10 egl = (EGL10)EGLContext.getEGL();
        egl.eglMakeCurrent(egl_display, egl_surface, egl_surface, egl_context);
    }

    public void swapEGL() {
        try {
            EGL10 egl = (EGL10)EGLContext.getEGL();
            // egl.eglWaitNative(EGL10.EGL_NATIVE_RENDERABLE, null);
            egl.eglWaitNative(EGL10.EGL_CORE_NATIVE_ENGINE, null);
            egl.eglWaitGL();
            egl.eglSwapBuffers(egl_display, egl_surface);
        } catch(Exception e) {
            Log.e("lfl", e.toString());
            for (StackTraceElement ste : e.getStackTrace()) Log.e("lfl", ste.toString());
        }
    }
}

