package com.lucidfusionlabs.app;

import java.net.*;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import android.os.*;
import android.view.*;
import android.widget.FrameLayout;
import android.widget.FrameLayout.LayoutParams;
import android.media.*;
import android.content.*;
import android.content.res.Configuration;
import android.content.pm.ActivityInfo;
import android.hardware.*;
import android.util.Log;
import android.net.Uri;

import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.egl.*;

public class Activity extends android.app.Activity {
    static { System.loadLibrary("app"); }
    
    public static native void main(Object activity);
    public static native void mainloop(Object activity);
    public static native void minimize();
    public static native void resize(int x, int y);
    public static native void key(int upordonw, int keycode);
    public static native void touch(int action, float x, float y, float p);
    public static native void fling(float x, float y, float vx, float vy);
    public static native void scroll(float x, float y, float sx, float sy);
    public static native void accel(float x, float y, float z);

    public static Activity instance;
    public static boolean init;
    public FrameLayout frameLayout;
    public GameView view;
    public Thread thread;
    public AudioManager audio;
    public GPlusClient gplus;
    public Advertising advertising;
    public boolean waiting_activity_result;
    public int egl_version;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        Log.i("lfl", "Activity.onCreate()");

        instance = this;
        requestWindowFeature(Window.FEATURE_NO_TITLE);
        super.onCreate(savedInstanceState);
        this.getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        
        frameLayout = new FrameLayout(this);
        view = new GameView(this, getApplication());
        frameLayout.addView(view, new LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.MATCH_PARENT));

        if (ActivityConfig.advertising) advertising = new Advertising(this, frameLayout);
        setContentView(frameLayout, new LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.MATCH_PARENT));

        if (ActivityConfig.play_services) gplus = new GPlusClient(this);
        audio = (AudioManager)getSystemService(Context.AUDIO_SERVICE);
    }
    
    @Override
    protected void onDestroy() {
        Log.i("lfl", "Activity.onDestroy()");
        if (advertising != null) advertising.onDestroy();
        super.onDestroy();
    }

    @Override
    protected void onStop() {
        Log.i("lfl", "Activity.onStop()");
        super.onStop();
        if (gplus != null) gplus.onStop();
    }
    
    @Override
    protected void onPause() {
        Log.i("lfl", "Activity.onPause() enter");
        super.onPause();
        if (waiting_activity_result || thread == null) return;

        Thread t = thread;
        thread = null;        
        minimize();
        try { t.join(); }
        catch(Exception e) { Log.e("lfl", e.toString()); }
        Log.i("lfl", "Activity.onPause() exit");
    }

    @Override
    protected void onStart() {
        Log.i("lfl", "Activity.onStart()");
        super.onStart();
        if (gplus != null) gplus.onStart(this);
    }

    @Override
    public void onResume() {
        Log.i("lfl", "Activity.onResume()");
        super.onResume();
    }

    @Override
    public void onConfigurationChanged(Configuration newConfig) {
        Log.i("lfl", "onConfigurationChanged");
        super.onConfigurationChanged(newConfig);
    }

    @Override
    protected void onActivityResult(int request, int response, Intent data) {
        Log.i("lfl", "Activity.onActivityResult(" + request + ", " + response + ")");
        waiting_activity_result = false;
        super.onActivityResult(request, response, data);
        if (gplus != null) gplus.onActivityResult(request, response, data);
    }

    void surfaceChanged(int format, int width, int height) {
        boolean thread_exists = thread != null;
        Log.i("lfl", "surfaceChanged init= " + init + ", thread_exists=" + thread_exists);
        resize(width, height);
        if (thread_exists) return;
        if (!init) thread = new Thread(new Runnable() { public void run() { view.initEGL(); main    (Activity.instance); } }, "JNIMainThread");
        else       thread = new Thread(new Runnable() { public void run() { view.initEGL(); mainloop(Activity.instance); } }, "JNIMainThread");
        thread.start();
        init = true;
    }

    void forceExit() {
        Log.i("lfl", "Activity.froceExit()");
        android.os.Process.killProcess(android.os.Process.myPid());
    }

    public String getFilesDirCanonicalPath() {
        try {
            return getFilesDir().getCanonicalPath();
        } catch (final Exception e) { Log.e("lfl", e.toString()); return ""; }
    }

    public void toggleKeyboard() {
        android.view.inputmethod.InputMethodManager imm = (android.view.inputmethod.InputMethodManager)getSystemService(Context.INPUT_METHOD_SERVICE);
        imm.toggleSoftInput(android.view.inputmethod.InputMethodManager.SHOW_FORCED, 0);
    }
    public void showKeyboard() {
        android.view.inputmethod.InputMethodManager imm = (android.view.inputmethod.InputMethodManager)getSystemService(Context.INPUT_METHOD_SERVICE);
        imm.showSoftInput(view, android.view.inputmethod.InputMethodManager.SHOW_FORCED);
    }
    public void hideKeyboard() {
        android.view.inputmethod.InputMethodManager imm = (android.view.inputmethod.InputMethodManager)getSystemService(Context.INPUT_METHOD_SERVICE);
        // imm.hideSoftInput(view, 0);
    }

    public MediaPlayer loadMusicResource(String filename) {
        android.content.res.Resources res = getResources();
        String package_name = getPackageName();
        int soundId = res.getIdentifier(filename, "raw", getPackageName());
        Log.i("lfl", "loadMusicAsset " + package_name + " " + filename + " " + soundId);
        return MediaPlayer.create(this, soundId);
    }
    public void playMusic(MediaPlayer mp) {
        mp.start();
    }
    public void playBackgroundMusic(MediaPlayer mp) {
        mp.setLooping(true);
        mp.start();
    }

    public int maxVolume() { return audio.getStreamMaxVolume(AudioManager.STREAM_MUSIC); }
    public int getVolume() { return audio.getStreamVolume(AudioManager.STREAM_MUSIC); }
    public void setVolume(int v) { audio.setStreamVolume(AudioManager.STREAM_MUSIC, v, 0); }
    
    public String getModelName() {
        return android.os.Build.MANUFACTURER + " " + android.os.Build.MODEL;
    }
    
    public int getAddress() {
        int ret = 0;
        try {
            for (java.util.Enumeration<NetworkInterface> en = NetworkInterface.getNetworkInterfaces(); en.hasMoreElements();) {
                NetworkInterface intf = en.nextElement();
                for (java.util.Enumeration<InetAddress> IpAddresses = intf.getInetAddresses(); IpAddresses.hasMoreElements();) {
                    InetAddress inetAddress = IpAddresses.nextElement();
                    if (inetAddress.isLoopbackAddress() || inetAddress instanceof Inet6Address) continue;
                    ret = ByteBuffer.wrap(inetAddress.getAddress()).getInt();
                }
            }  
        } catch(Exception e) {}
        return ret;
    }

    public int getBroadcastAddress() {
        int ret = 0;
        try {
            for (java.util.Enumeration<NetworkInterface> en = NetworkInterface.getNetworkInterfaces(); en.hasMoreElements();) {
                NetworkInterface intf = en.nextElement();
                for (java.util.Enumeration<InetAddress> IpAddresses = intf.getInetAddresses(); IpAddresses.hasMoreElements();) {
                    InetAddress inetAddress = IpAddresses.nextElement(), bcastAddress = null;
                    if (inetAddress.isLoopbackAddress() || inetAddress instanceof Inet6Address) continue;
                    // Log.i("lfl", "ip address: " + inetAddress);
                    NetworkInterface ipv4_intf = NetworkInterface.getByInetAddress(inetAddress);
                    for (InterfaceAddress ifaceAddress : ipv4_intf.getInterfaceAddresses()) bcastAddress = ifaceAddress.getBroadcast();
                    // Log.i("lfl", "broadcast aaddress: " + bcastAddress);
                    ret = ByteBuffer.wrap(bcastAddress.getAddress()).getInt();   
                }
            }  
        } catch(Exception e) {}
        return ret;
    }

    public void openBrowser(String url) {
        final Intent intent = new Intent(Intent.ACTION_VIEW).setData(Uri.parse(url));
        startActivity(intent);
    }

    public void hideAds() { /* advertising.hideAds(); */ }
    public void showAds() { /* advertising.showAds(); */ }
}

class MyGestureListener extends android.view.GestureDetector.SimpleOnGestureListener {
    @Override
    public boolean onFling(MotionEvent e1, MotionEvent e2, float velocityX, float velocityY) {
        Activity.fling(e1.getX(), e1.getY(), velocityX, velocityY);
        return false;
    }
}

class GameView extends android.view.SurfaceView implements SurfaceHolder.Callback, View.OnKeyListener, View.OnTouchListener, SensorEventListener {

    public EGLContext egl_context;
    public EGLSurface egl_surface;
    public EGLDisplay egl_display;
    public GestureDetector gesture;
    public SensorManager sensor;

    public GameView(Activity activity, Context context) {
        super(context);
        // gesture = new GestureDetector(new MyGestureListener());
        sensor = (SensorManager)context.getSystemService("sensor");  
        setFocusable(true);
        setFocusableInTouchMode(true);
        setOnKeyListener(this); 
        setOnTouchListener(this);   
        getHolder().addCallback(this); 
        requestFocus();
    }

    @Override
    protected void onSizeChanged(int xNew, int yNew, int xOld, int yOld) {
        Log.i("lfl", "GameView.onSizeChanged(" + xNew + ", " + yNew + ", " + xOld + ", " + yOld + ")");
    }

    public void surfaceCreated(SurfaceHolder holder) {
        Log.i("lfl", "GameView.surfaceCreated()");
        sensor.registerListener(this, sensor.getDefaultSensor(Sensor.TYPE_ACCELEROMETER), SensorManager.SENSOR_DELAY_GAME, null);
    }

    public void surfaceChanged(SurfaceHolder holder, int format, int width, int height) {
        Log.i("lfl", "GameView.surfaceChanged(" + width + ", " + height + ")");
        Activity.instance.surfaceChanged(format, width, height);
    }

    public void surfaceDestroyed(SurfaceHolder holder) {
        Log.i("lfl", "surfaceDestroyed()");
        sensor.unregisterListener(this, sensor.getDefaultSensor(Sensor.TYPE_ACCELEROMETER));
    }

    public void onSensorChanged(SensorEvent event) {
        if (event.sensor.getType() == Sensor.TYPE_ACCELEROMETER) Activity.accel(event.values[0], event.values[1], event.values[2]);
    }

    public boolean onKey(View v, int keyCode, KeyEvent event) {
        int keyChar = event.getUnicodeChar();
        if (keyChar == 0) {
            switch(event.getKeyCode()) {
                case KeyEvent.KEYCODE_DEL:        keyChar = '\b';   break;
                case KeyEvent.KEYCODE_ESCAPE:     keyChar = 0xE100; break;
                case KeyEvent.KEYCODE_DPAD_UP:    keyChar = 0xE101; break;
                case KeyEvent.KEYCODE_DPAD_DOWN:  keyChar = 0xE102; break;
                case KeyEvent.KEYCODE_DPAD_LEFT:  keyChar = 0xE103; break;
                case KeyEvent.KEYCODE_DPAD_RIGHT: keyChar = 0xE104; break;
                case KeyEvent.KEYCODE_CTRL_LEFT:  keyChar = 0xE105; break;
                case KeyEvent.KEYCODE_CTRL_RIGHT: keyChar = 0xE106; break;
                case KeyEvent.KEYCODE_META_LEFT:  keyChar = 0xE107; break;
                case KeyEvent.KEYCODE_META_RIGHT: keyChar = 0xE108; break;
                case KeyEvent.KEYCODE_TAB:        keyChar = 0xE109; break;
                case KeyEvent.KEYCODE_PAGE_UP:    keyChar = 0xE10A; break;
                case KeyEvent.KEYCODE_PAGE_DOWN:  keyChar = 0xE10B; break;
                case KeyEvent.KEYCODE_F1:         keyChar = 0xE10C; break;
                case KeyEvent.KEYCODE_F2:         keyChar = 0xE10D; break;
                case KeyEvent.KEYCODE_F3:         keyChar = 0xE10E; break;
                case KeyEvent.KEYCODE_F4:         keyChar = 0xE10F; break;
                case KeyEvent.KEYCODE_F5:         keyChar = 0xE110; break;
                case KeyEvent.KEYCODE_F6:         keyChar = 0xE111; break;
                case KeyEvent.KEYCODE_F7:         keyChar = 0xE112; break;
                case KeyEvent.KEYCODE_F8:         keyChar = 0xE113; break;
                case KeyEvent.KEYCODE_F9:         keyChar = 0xE114; break;
            }
        }
        if      (event.getAction() == KeyEvent.ACTION_UP)   { Activity.key(0, keyChar); return true; }
        else if (event.getAction() == KeyEvent.ACTION_DOWN) { Activity.key(1, keyChar); return true; }
        else return false;
    }

    public boolean onTouch(View v, MotionEvent event) {
        // gesture.onTouchEvent(event);
        final int action = event.getAction() & MotionEvent.ACTION_MASK;
        if (action == MotionEvent.ACTION_MOVE) {
            for (int i = 0; i < event.getPointerCount(); i++) {
                Activity.touch(action, event.getX(i), event.getY(i), event.getPressure(i));
            }
        } else {
            int action_index = (action == MotionEvent.ACTION_POINTER_DOWN || action == MotionEvent.ACTION_POINTER_UP) ? event.getActionIndex() : 0;
            Activity.touch(action, event.getX(action_index), event.getY(action_index), event.getPressure(action_index));
        }
        return true; 
    }

    public void onAccuracyChanged(Sensor sensor, int accuracy) {}

    public void initEGL() {
        if      (initEGL(2)) Activity.instance.egl_version = 2;
        else if (initEGL(1)) Activity.instance.egl_version = 1;
        else                 Activity.instance.egl_version = 0;
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
        } catch(Exception e) {
            Log.e("lfl", e.toString());
            for (StackTraceElement ste : e.getStackTrace()) Log.e("lfl", ste.toString());
            egl_context = null; egl_surface = null; egl_display = null;
            return false;
        }
        return true;
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
