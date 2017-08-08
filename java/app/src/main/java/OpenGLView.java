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
    public boolean paused, have_surface, reshape_synchronized, reshape_pending;
    public Object sync = new Object();

    public OpenGLView(MainActivity activity, Context context) {
        super(context);
        main_activity = activity;
        setFocusable(true);
        setFocusableInTouchMode(true);
        getHolder().addCallback(this);
    }

    @Override protected void onSizeChanged(int xNew, int yNew, int xOld, int yOld) {
        NativeAPI.INFO("OpenGLView.onSizeChanged(" + xNew + ", " + yNew + " <- " + xOld + ", " + yOld + ")");
    }

    @Override public void surfaceCreated(SurfaceHolder holder) {
        NativeAPI.INFO("OpenGLView.surfaceCreated()");
        have_surface = true;
    }

    @Override public void surfaceChanged(SurfaceHolder holder, int format, int width, int height) {
        boolean init = surface_width == 0 && surface_height == 0, thread_exists = thread != null;
        boolean size_changed = height != surface_height || width != surface_width;
        NativeAPI.INFO("OpenGLView.surfaceChanged(" + width + ", " + height + ") init=" + init + 
              ", size_changed=" + size_changed + ", thread_exists=" + thread_exists);
        surface_width = width;
        surface_height = height;
        if (!thread_exists) startRenderThread();
        if (size_changed && !init) synchronizedReshape();
    }

    @Override public void surfaceDestroyed(SurfaceHolder holder) {
        NativeAPI.INFO("surfaceDestroyed()");
        have_surface = false;
    }

    public void synchronizedReshape() {
        boolean gl_thread = Thread.currentThread() == thread;
        NativeAPI.INFO("NativeAPI.reshaped(0, 0, " + surface_width + ", " + surface_height + ") gl_thread=" + gl_thread + " frame_enabled=" + NativeAPI.getFrameEnabled());
        NativeAPI.reshaped(0, 0, surface_width, surface_height);
        if (reshape_synchronized) {
            if (gl_thread) return;
            reshape_pending = true;
            synchronized(sync) {
                while (thread != null && reshape_pending && NativeAPI.getFrameEnabled()) {
                    try { sync.wait(); }
                    catch (InterruptedException ex) { Thread.currentThread().interrupt(); }
                }
            }
            NativeAPI.INFO("OpenGLView synchronizedReshape done");
        }
    }

    public void onSynchronizedReshape() {
        destroySurface();
        createSurface();
        synchronized(sync) { reshape_pending = false; }
    }

    public void onResume() {
        boolean thread_exists = thread != null;
        NativeAPI.INFO("OpenGLView.onResume() have_surface=" + have_surface + " thread_exists=" + thread_exists);
        if (paused) {
            paused = false;
            if (!thread_exists) startRenderThread();
        }
    }

    public void onPause() {
        if (thread == null) return;
        Thread t = thread;
        thread = null;        
        NativeAPI.minimize();
        try { t.join(); }
        catch(Exception e) { NativeAPI.ERROR(e.toString()); }
        paused = true;
    }

    public void startRenderThread() {
        NativeAPI.INFO("startRenderThread NativeAPI.init=" + NativeAPI.init);
        main_activity.screens.onStartRenderThread(main_activity, main_activity.gl_layout);
        if   (!NativeAPI.init) thread = new Thread(new Runnable() { public void run() { initEGL(); NativeAPI.main(surface_width, surface_height, egl_version); shutdownEGL(); } }, "JNIMainThread");
        else                   thread = new Thread(new Runnable() { public void run() { initEGL(); NativeAPI.newMainLoop(main_activity, true);                 shutdownEGL(); } }, "JNIMainThread");
        thread.start();
        NativeAPI.init = true;
    }

    public boolean createSurface() {
        try {
            egl_surface = egl.eglCreateWindowSurface(egl_display, egl_configs[0], this, null);
            if (egl_surface == null || egl_surface == EGL10.EGL_NO_SURFACE) throw new Exception(getEGLError(egl, "eglCreateWindowSurface"));
            if (!egl.eglMakeCurrent(egl_display, egl_surface, egl_surface, egl_context)) throw new Exception(getEGLError(egl, "eglMakeCurrent"));
            return true;
        } catch(Exception e) {
            NativeAPI.ERROR(e.toString());
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
        catch(Exception e) { NativeAPI.ERROR(e.toString()); }
        finally { egl_surface = null; have_surface = false; }
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
            NativeAPI.ERROR(e.toString());
            egl = null;
            egl_display = null;
            egl_context = null;
            return false;
        }

        return createSurface();
    }

    public void shutdownEGL() {
        destroySurface();
        try {
            EGL10 egl = (EGL10)EGLContext.getEGL();
            egl.eglDestroyContext(egl_display, egl_context);
            egl.eglTerminate(egl_display);
        } catch(Exception e) { NativeAPI.ERROR(e.toString()); }
        egl_display = null;
        egl_context = null;
    }

    public void makeCurrentEGL() {
        egl.eglMakeCurrent(egl_display, egl_surface, egl_surface, egl_context);
    }

    public void swapEGL() {
        try {
            egl.eglSwapBuffers(egl_display, egl_surface);
            synchronized(sync) { sync.notifyAll(); }
        } catch(Exception e) { NativeAPI.ERROR("swapEGL " + e.toString()); }
    }

    public static String getEGLError(EGL10 egl, String prefix) {
        return prefix + ": " + getEGLString(egl.eglGetError());
    }

    public static String getEGLString(int error) {
        switch(error) {
            case EGL10.EGL_SUCCESS:             return "No error";
            case EGL10.EGL_NOT_INITIALIZED:     return "EGL not initialized or failed to initialize";
            case EGL10.EGL_BAD_ACCESS:          return "Resource inaccessible";
            case EGL10.EGL_BAD_ALLOC:           return "Cannot allocate resources";
            case EGL10.EGL_BAD_ATTRIBUTE:       return "Unrecognized attribute or attribute value";
            case EGL10.EGL_BAD_CONTEXT:         return "Invalid EGL context";
            case EGL10.EGL_BAD_CONFIG:          return "Invalid EGL frame buffer configuration";
            case EGL10.EGL_BAD_CURRENT_SURFACE: return "Current surface is no longer valid";
            case EGL10.EGL_BAD_DISPLAY:         return "Invalid EGL display";
            case EGL10.EGL_BAD_SURFACE:         return "Invalid surface";
            case EGL10.EGL_BAD_MATCH:           return "Inconsistent arguments";
            case EGL10.EGL_BAD_PARAMETER:       return "Invalid argument";
            case EGL10.EGL_BAD_NATIVE_PIXMAP:   return "Invalid native pixmap";
            case EGL10.EGL_BAD_NATIVE_WINDOW:   return "Invalid native window";
            default:                            return "Unknown";
        }
    }
}
