package com.lucidfusionlabs.lfjava;

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
import android.content.pm.ActivityInfo;
import android.hardware.*;
import android.util.Log;
import android.net.Uri;

import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.egl.*;

public class Activity extends android.app.Activity {
    static { System.loadLibrary("lfjni"); }
    
    public static native void main(Object activity);
    public static native void shutdown();
    public static native void resize(int x, int y);
    public static native void key(int upordonw, int keycode);
    public static native void touch(int action, float x, float y, float p);
    public static native void fling(float x, float y, float vx, float vy);
    public static native void scroll(float x, float y, float sx, float sy);
    public static native void accel(float x, float y, float z);

    public static Activity instance;
    public FrameLayout frameLayout;
    public SurfaceView view;
    public Thread thread;
    public AudioManager audio;
    public GPlusClient gplus;
    public Advertising advertising;
    public boolean waiting_activity_result;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        Log.i("lfjava", "Activity.onCreate()");

        instance = this;
        requestWindowFeature(Window.FEATURE_NO_TITLE);
        super.onCreate(savedInstanceState);
        this.getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN,
                                  WindowManager.LayoutParams.FLAG_FULLSCREEN);
        this.getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE);        

        // final ActivityManager activityManager = (ActivityManager)getSystemService(Context.ACTIVITY_SERVICE);
        // final ConfigurationInfo configurationInfo = activityManager.getDeviceConfigurationInfo();
        // final boolean supportsEs2 = configurationInfo.reqGlEsVersion >= 0x20000;
        // if (supportsEs2) {}        
        
        frameLayout = new FrameLayout(this);
        view = new SurfaceView(this, getApplication());
        frameLayout.addView(view, new LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.MATCH_PARENT));

        if (ActivityConfig.advertising) advertising = new Advertising(this, frameLayout);
        setContentView(frameLayout, new LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.MATCH_PARENT));

        if (ActivityConfig.play_services) gplus = new GPlusClient(this);
        audio = (AudioManager)getSystemService(Context.AUDIO_SERVICE);
    }
    
    @Override
    protected void onDestroy() {
        Log.i("lfjava", "Activity.onDestroy()");
        if (advertising != null) advertising.onDestroy();
        super.onDestroy();
        close();
        exit();
    }

    @Override
    protected void onPause() {
        Log.i("lfjava", "Activity.onPause()");
        super.onPause();
        if (!waiting_activity_result) { close(); }
    }

    @Override
    protected void onStart() {
        Log.i("lfjava", "Activity.onStart()");
        super.onStart();
        if (gplus != null) gplus.onStart(this);
    }

    @Override
    protected void onStop() {
        Log.i("lfjava", "Activity.onStop()");
        super.onStop();
        if (gplus != null) gplus.onStop();
    }
    
    @Override
    protected void onActivityResult(int request, int response, Intent data) {
        Log.i("lfjava", "Activity.onActivityResult(" + request + ", " + response + ")");
        waiting_activity_result = false;
        super.onActivityResult(request, response, data);
        if (gplus != null) gplus.onActivityResult(request, response, data);
    }
    
    void exit() {
        Log.i("lfjava", "Activity.exit()");
        android.os.Process.killProcess(android.os.Process.myPid());
    }

    void open(int format, int width, int height) {
        Log.i("lfjava", "Activity.open()");
        resize(width, height);

        if (thread != null) return;
        thread = new Thread(new Runnable() { public void run() {
            Activity.main(Activity.instance);
            Activity.instance.finish();
        } }, "JNIMainThread");
        thread.start();
    }

    void close() {
        if (thread == null) return;
        Log.i("lfjava", "Activity.close() enter");
        Thread t = thread;
        thread = null;        
        shutdown();
        try { t.join(); }
        catch(Exception e) { Log.e("lfjava", e.toString()); }
        Log.i("lfjava", "Activity.close() exit");
    }

    public int readFile(String filename, byte[] buf, int size) {
        try {
            java.io.FileInputStream inputStream = openFileInput(filename);
            inputStream.read(buf, 0, size);
            inputStream.close();
            return size;
        } catch (final Exception e) { Log.e("lfjava", e.toString()); return 0; }
    }
    public int sizeFile(String filename) {
        try {
            java.io.File file = new java.io.File(getFilesDir(), filename);
            return file.exists() ? (int)file.length() : 0;
        } catch (final Exception e) { Log.e("lfjava", e.toString()); return 0; }
    }
    public java.io.FileOutputStream openFileWriter(String filename) {
        try { return openFileOutput(filename, Context.MODE_PRIVATE); } 
        catch (final Exception e) { Log.e("lfjava", e.toString()); return null; }
    }
    public void writeFile(java.io.FileOutputStream outputStream, byte[] buf, int size) {
        try { outputStream.write(buf, 0, size); }
        catch (final Exception e) { Log.e("lfjava", e.toString()); }
    }
    public void closeFileWriter(java.io.FileOutputStream outputStream) {
        try { outputStream.close(); }
        catch (final Exception e) { Log.e("lfjava", e.toString()); }
    }

    public void toggleKeyboard() {
        android.view.inputmethod.InputMethodManager imm = (android.view.inputmethod.InputMethodManager)getSystemService(Context.INPUT_METHOD_SERVICE);
        imm.toggleSoftInput(android.view.inputmethod.InputMethodManager.SHOW_FORCED, 0);
    }

    public MediaPlayer loadMusicResource(String filename) {
        android.content.res.Resources res = getResources();
        String package_name = getPackageName();
        int soundId = res.getIdentifier(filename, "raw", getPackageName());
        Log.i("lfjava", "loadMusicAsset " + package_name + " " + filename + " " + soundId);
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
                    // Log.i("lfjava", "ip address: " + inetAddress);
                    NetworkInterface ipv4_intf = NetworkInterface.getByInetAddress(inetAddress);
                    for (InterfaceAddress ifaceAddress : ipv4_intf.getInterfaceAddresses()) bcastAddress = ifaceAddress.getBroadcast();
                    // Log.i("lfjava", "broadcast aaddress: " + bcastAddress);
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

class SurfaceView extends android.view.SurfaceView implements SurfaceHolder.Callback, View.OnKeyListener, View.OnTouchListener, SensorEventListener {

    public EGLContext egl_context;
    public EGLSurface egl_surface;
    public EGLDisplay egl_display;
    public GestureDetector gesture;
    public SensorManager sensor;

    public SurfaceView(Activity activity, Context context) {
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

    public void surfaceCreated(SurfaceHolder holder) {
       sensor.registerListener(this, sensor.getDefaultSensor(Sensor.TYPE_ACCELEROMETER), SensorManager.SENSOR_DELAY_GAME, null);
    }

    public void surfaceChanged(SurfaceHolder holder, int format, int width, int height) {
        Log.i("lfjava", "SurfaceView.surfaceChanged(" + width + ", " + height + ")");
        Activity.instance.open(format, width, height);
    }

    public void surfaceDestroyed(SurfaceHolder holder) {
        Log.i("lfjava", "surfaceDestroyed()");
        Activity.instance.close(); 
        sensor.unregisterListener(this, sensor.getDefaultSensor(Sensor.TYPE_ACCELEROMETER));
    }

    public void onSensorChanged(SensorEvent event) {
        if (event.sensor.getType() == Sensor.TYPE_ACCELEROMETER) Activity.accel(event.values[0], event.values[1], event.values[2]);
    }

    public boolean onKey(View v, int keyCode, KeyEvent event) {
        int keyChar = event.getUnicodeChar();
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
        }
        catch(Exception e) {
            Log.e("lfjava", e.toString());
            for (StackTraceElement ste : e.getStackTrace()) Log.e("lfjava", ste.toString());
            egl_context = null; egl_surface = null; egl_display = null;
            return false;
        }
        return true;
    }
    
    public int initEGL1() {
        if (initEGL(1)) return 1;
        return -1;
    }
    public int initEGL2() {
        if (initEGL(2)) return 2;
        if (initEGL(1)) return 1;
        return -1;
    }

    public void swapEGL() {
        try {
            EGL10 egl = (EGL10)EGLContext.getEGL();
            // egl.eglWaitNative(EGL10.EGL_NATIVE_RENDERABLE, null);
            egl.eglWaitNative(EGL10.EGL_CORE_NATIVE_ENGINE, null);
            egl.eglWaitGL();
            egl.eglSwapBuffers(egl_display, egl_surface);
        }
        catch(Exception e) {
            Log.e("lfjava", e.toString());
            for (StackTraceElement ste : e.getStackTrace()) Log.e("lfjava", ste.toString());
        }
    }    
}
