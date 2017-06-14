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

public class OpenGLView extends android.view.SurfaceView implements SurfaceHolder.Callback {
    public MainActivity main_activity;
    public EGL10 egl;
    public EGLConfig[] egl_configs;
    public EGLContext egl_context;
    public EGLSurface egl_surface;
    public EGLDisplay egl_display;
    public Thread thread;
    public int surface_width, surface_height, egl_version;
    public boolean have_surface, reshape_pending;
    public Object sync = new Object();

    public OpenGLView(MainActivity activity, Context context) {
        super(context);
        main_activity = activity;
        setFocusable(true);
        setFocusableInTouchMode(true);
        getHolder().addCallback(this);
    }

    @Override protected void onSizeChanged(int xNew, int yNew, int xOld, int yOld) {
        Log.i("lfl", "OpenGLView.onSizeChanged(" + xNew + ", " + yNew + " <- " + xOld + ", " + yOld + ")");
    }

    @Override public void surfaceCreated(SurfaceHolder holder) {
        Log.i("lfl", "OpenGLView.surfaceCreated()");
        have_surface = true;
    }

    @Override public void surfaceChanged(SurfaceHolder holder, int format, int width, int height) {
        boolean init = surface_width == 0 && surface_height == 0, thread_exists = thread != null;
        boolean size_changed = height != surface_height || width != surface_width;
        Log.i("lfl", "OpenGLView.surfaceChanged(" + width + ", " + height + ") init=" + init + 
              ", size_changed=" + size_changed + ", thread_exists=" + thread_exists);
        surface_width = width;
        surface_height = height;
        if (!thread_exists) startRenderThread(true);
        if (size_changed && !init) synchronizedReshape();
    }

    @Override public void surfaceDestroyed(SurfaceHolder holder) {
        Log.i("lfl", "surfaceDestroyed()");
        have_surface = false;
    }

    public void synchronizedReshape() {
        if (Thread.currentThread() == thread) return;
        reshape_pending = true;
        main_activity.nativeReshaped(0, 0, surface_width, surface_height);
        synchronized(sync) {
            while (thread != null && reshape_pending) {
                try { sync.wait(); }
                catch (InterruptedException ex) { Thread.currentThread().interrupt(); }
            }
        }
        Log.i("lfl", "OpenGLView synchronizedReshape done");
    }

    public void onSynchronizedReshape() {
        synchronized(sync) { reshape_pending = false; }
        destroySurface();
        createSurface();
    }

    public void onResume() {
        boolean thread_exists = thread != null;
        Log.i("lfl", "OpenGLView.onResume() have_surface=" + have_surface + " thread_exists=" + thread_exists);
        if (have_surface && !thread_exists) startRenderThread(false);
    }

    public void onPause() {
        if (thread == null) return;
        Thread t = thread;
        thread = null;        
        main_activity.nativeMinimize();
        try { t.join(); }
        catch(Exception e) { Log.e("lfl", e.toString()); }
    }

    public void startRenderThread(boolean native_reset) {
        Log.i("lfl", "startRenderThread native_init=" + main_activity.native_init + " native_reset=" + native_reset);
        main_activity.screens.onStartRenderThread(main_activity, main_activity.gl_layout);
        if      (!main_activity.native_init) thread = new Thread(new Runnable() { public void run() { initEGL();        main_activity.nativeMain       ();      } }, "JNIMainThread");
        else if (native_reset)               thread = new Thread(new Runnable() { public void run() { initEGL();        main_activity.nativeNewMainLoop(true);  } }, "JNIMainThread");
        else                                 thread = new Thread(new Runnable() { public void run() { makeCurrentEGL(); main_activity.nativeNewMainLoop(false); } }, "JNIMainThread");
        thread.start();
        main_activity.native_init = true;
    }

    public boolean createSurface() {
        try {
            egl_surface = egl.eglCreateWindowSurface(egl_display, egl_configs[0], this, null);
            if (egl_surface == null || egl_surface == EGL10.EGL_NO_SURFACE) throw new Exception("eglCreateWindowSurface");
            if (!egl.eglMakeCurrent(egl_display, egl_surface, egl_surface, egl_context)) throw new Exception("eglMakeCurrent");
            return true;
        } catch(Exception e) {
            Log.e("lfl", e.toString());
            egl_surface = null;
            return false;
        }
    }

    public void destroySurface() {
        if (egl == null || egl_surface == null || egl_surface == EGL10.EGL_NO_SURFACE) return;
        try { 
            egl.eglMakeCurrent(egl_display, EGL10.EGL_NO_SURFACE, EGL10.EGL_NO_SURFACE, EGL10.EGL_NO_CONTEXT);
            egl.eglDestroySurface(egl_display, egl_surface);
        }
        catch(Exception e) { Log.e("lfl", e.toString()); }
        finally { egl_surface = null; }
    }

    public void initEGL() {
        if      (initEGL(2)) egl_version = 2;
        else if (initEGL(1)) egl_version = 1;
        else                 egl_version = 0;
    }

    public boolean initEGL(int requestedVersion) {
        final int EGL_OPENGL_ES2_BIT = 4, EGL_CONTEXT_CLIENT_VERSION = 0x3098;
        final int[] attrib_list1 = { EGL10.EGL_DEPTH_SIZE, 16, EGL10.EGL_NONE };
        final int[] attrib_list2 = { EGL10.EGL_DEPTH_SIZE, 16, EGL10.EGL_RENDERABLE_TYPE, EGL_OPENGL_ES2_BIT, EGL10.EGL_NONE };
        final int[] context_attrib2 = { EGL_CONTEXT_CLIENT_VERSION, 2, EGL10.EGL_NONE };
        final int[] attrib_list = requestedVersion >= 2 ? attrib_list2 : attrib_list1;
        final int[] context_attrib = requestedVersion >= 2 ? context_attrib2 : null;
        final int[] version = new int[2], num_config = new int[1];
        egl_configs = new EGLConfig[1];

        try {
            egl = (EGL10)EGLContext.getEGL();
            egl_display = egl.eglGetDisplay(EGL10.EGL_DEFAULT_DISPLAY);
            egl.eglInitialize(egl_display, version);
            if (!egl.eglChooseConfig(egl_display, attrib_list, egl_configs, 1, num_config) || num_config[0] == 0) throw new Exception("eglChooseConfig");
            if ((egl_context = egl.eglCreateContext(egl_display, egl_configs[0], EGL10.EGL_NO_CONTEXT, context_attrib)) == EGL10.EGL_NO_CONTEXT) throw new Exception("eglCreateContext");
        } catch(Exception e) {
            Log.e("lfl", e.toString());
            egl = null;
            egl_display = null;
            egl_context = null;
            return false;
        }

        return createSurface();
    }

    public void makeCurrentEGL() {
        egl.eglMakeCurrent(egl_display, egl_surface, egl_surface, egl_context);
    }

    public void swapEGL() {
        try {
            egl.eglSwapBuffers(egl_display, egl_surface);
            synchronized(sync) { sync.notifyAll(); }
        } catch(Exception e) { Log.e("lfl", "swapEGL " + e.toString()); }
    }
}
