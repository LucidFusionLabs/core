package com.lucidfusionlabs.app;

import java.net.*;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.HashMap;

import android.os.*;
import android.view.*;
import android.widget.FrameLayout;
import android.widget.FrameLayout.LayoutParams;
import android.widget.LinearLayout;
import android.widget.TableLayout;
import android.widget.RelativeLayout;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;
import android.media.*;
import android.content.*;
import android.content.res.Configuration;
import android.content.res.Resources;
import android.content.pm.ActivityInfo;
import android.hardware.*;
import android.util.Log;
import android.util.Pair;
import android.util.TypedValue;
import android.net.Uri;
import android.graphics.Rect;
import android.app.ActionBar;
import android.app.AlertDialog;
import android.app.Fragment;

import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.egl.*;

public class MainActivity extends android.app.Activity {
    static { System.loadLibrary("app"); }
    
    public native void AppCreate();
    public native void AppMain();
    public native void AppNewMainLoop(boolean reset);
    public native void AppMinimize();
    public native void AppReshaped(int x, int y, int w, int h);
    public native void AppKeyPress(int keycode, int mod, int down);
    public native void AppTouch(int action, float x, float y, float p);
    public native void AppFling(float x, float y, float vx, float vy);
    public native void AppScroll(float x, float y, float sx, float sy);
    public native void AppAccel(float x, float y, float z);
    public static native void AppFocusedShellRun(String text);

    public static boolean app_created, app_init, disable_title;
    public static JWidgets jwidgets = new JWidgets();

    public Resources resources;
    public FrameLayout frame_layout;
    public ActionBar action_bar;
    public Window root_window;
    public View root_view;
    public MainView view;
    public Thread thread;
    public AudioManager audio;
    public GPlusClient gplus;
    public Advertising advertising;
    public boolean waiting_activity_result;
    public int surface_width, surface_height, egl_version;
    public int attr_listPreferredItemHeight, attr_scrollbarSize;
    public float display_density;
    public PreferenceFragment preferences;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        Log.i("lfl", "MainActivity.onCreate() app_created=" + app_created);
        super.onCreate(savedInstanceState);
        Context context = getApplication();
        LayoutInflater inflater = (LayoutInflater)context.getSystemService(LAYOUT_INFLATER_SERVICE);

        frame_layout = (FrameLayout)inflater.inflate(R.layout.main, null);
        root_window = getWindow();
        resources = getResources();
        view = new MainView(this, context);
        audio = (AudioManager)getSystemService(Context.AUDIO_SERVICE);
        action_bar = getActionBar();

        if (!app_created) {
            Log.i("lfl", "MainActivity.AppCreate()");
            app_created = true;
            AppCreate();
        } else {
            if (disable_title) disableTitle();
        }

        setContentView(frame_layout);
        frame_layout.addView(view, new LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.MATCH_PARENT));
        display_density = resources.getDisplayMetrics().density;

        int[] attr = new int[] { android.R.attr.listPreferredItemHeight, android.R.attr.scrollbarSize };
        android.content.res.TypedArray attr_val = context.obtainStyledAttributes(attr);
        attr_listPreferredItemHeight = attr_val.getDimensionPixelSize(0, -1);
        attr_scrollbarSize           = attr_val.getDimensionPixelSize(1, -1);

        root_view = root_window.getDecorView().findViewById(android.R.id.content);
        root_view.getViewTreeObserver().addOnGlobalLayoutListener(
            new ViewTreeObserver.OnGlobalLayoutListener() {
            public void onGlobalLayout(){
                if (!app_init) return;
                Rect r = new Rect();
                View view = root_window.getDecorView();
                view.getWindowVisibleDisplayFrame(r);

                int actionbar_h = action_bar.isShowing() ? action_bar.getHeight() : 0;
                int h = r.bottom - r.top - actionbar_h;
                for (JToolbar toolbar : jwidgets.toolbar_bottom) {
                    View tb = toolbar.view;
                    if (tb == null) continue;
                    FrameLayout.LayoutParams params = (FrameLayout.LayoutParams)tb.getLayoutParams();
                    params.bottomMargin = surface_height - h;
                    tb.setLayoutParams(params);
                    h -= tb.getHeight();
                }

                // Log.i("lfl", "onGlobalLayout(" + r.left + ", " + r.top + ", " + r.right + ", " + r.bottom + ", " + actionbar_h + ") = " + h);
                AppReshaped(r.left, surface_height - h, r.right - r.left, h); 
            }});
    }

    @Override
    protected void onStart() {
        Log.i("lfl", "MainActivity.onStart()");
        super.onStart();
        if (gplus != null) gplus.onStart(this);
    }

    @Override
    public void onResume() {
        boolean have_surface = view.have_surface, thread_exists = thread != null;
        Log.i("lfl", "MainActivity.onResume() have_surface=" + have_surface + " thread_exists=" + thread_exists);
        super.onResume();
        jwidgets.onResume(this);
        if (have_surface && !thread_exists) startRenderThread(false);
    }
    
    @Override
    protected void onPause() {
        Log.i("lfl", "MainActivity.onPause() enter");
        super.onPause();
        if (waiting_activity_result || thread == null) return;

        Thread t = thread;
        thread = null;        
        AppMinimize();
        try { t.join(); }
        catch(Exception e) { Log.e("lfl", e.toString()); }
        Log.i("lfl", "MainActivity.onPause() exit");
    }
    
    @Override
    protected void onStop() {
        Log.i("lfl", "MainActivity.onStop()");
        super.onStop();
        if (gplus != null) gplus.onStop();
    }

    @Override
    protected void onDestroy() {
        Log.i("lfl", "MainActivity.onDestroy()");
        jwidgets.onDestroy();
        if (advertising != null) advertising.onDestroy();
        super.onDestroy();
    }

    @Override
    public void onConfigurationChanged(Configuration newConfig) {
        Log.i("lfl", "onConfigurationChanged");
        super.onConfigurationChanged(newConfig);
    }

    @Override
    protected void onActivityResult(int request, int response, Intent data) {
        Log.i("lfl", "MainActivity.onActivityResult(" + request + ", " + response + ")");
        waiting_activity_result = false;
        super.onActivityResult(request, response, data);
        if (gplus != null) gplus.onActivityResult(request, response, data);
    }

    @Override
    public void onBackPressed() {
        if (getFragmentManager().getBackStackEntryCount() > 1) {
            super.onBackPressed();
            showBackFragment(false, true);
        } else {
            Fragment frag = getFragmentManager().findFragmentByTag("0");
            if (frag == null || !(frag instanceof JListViewFragment)) return;
            JModelItem nav_left = ((JListViewFragment)frag).data.nav_left;
            if (nav_left != null && nav_left.cb != null) nav_left.cb.run();
        }
    } 

    public void surfaceChanged(int format, int width, int height) {
        surface_width = width;
        surface_height = height;
        boolean thread_exists = thread != null;
        Log.i("lfl", "surfaceChanged thread_exists=" + thread_exists);
        AppReshaped(0, 0, width, height);
        if (thread_exists) return;
        startRenderThread(true);
    }

    public void startRenderThread(boolean app_reset) {
        Log.i("lfl", "startRenderThread app_init= " + app_init);
        if      (!app_init) thread = new Thread(new Runnable() { public void run() { view.initEGL();        AppMain       ();      } }, "JNIMainThread");
        else if (app_reset) thread = new Thread(new Runnable() { public void run() { view.initEGL();        AppNewMainLoop(true);  } }, "JNIMainThread");
        else                thread = new Thread(new Runnable() { public void run() { view.makeCurrentEGL(); AppNewMainLoop(false); } }, "JNIMainThread");
        thread.start();
        app_init = true;
    }

    public void forceExit() {
        Log.i("lfl", "MainActivity.froceExit()");
        android.os.Process.killProcess(android.os.Process.myPid());
    }

    public void createAdvertising() { advertising = new Advertising(this, frame_layout); }
    public void createGPLus()       { gplus       = new GPlusClient(this); }

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
        imm.hideSoftInputFromWindow(view.getWindowToken(), 0);
    }

    public void hideKeyboardAfteEnter() {
        android.view.inputmethod.InputMethodManager imm = (android.view.inputmethod.InputMethodManager)getSystemService(Context.INPUT_METHOD_SERVICE);
        imm.hideSoftInputFromWindow(view.getWindowToken(), android.view.inputmethod.InputMethodManager.HIDE_NOT_ALWAYS); 
    }

    public void disableTitle() {
        disable_title = true;
        action_bar.hide();
        // requestWindowFeature(Window.FEATURE_NO_TITLE);
    }

    public void enableKeepScreenOn() {
        root_window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
    }

    public void setCaption(final String text) {
        runOnUiThread(new Runnable() { public void run() { setTitle(text); } });
    }

    public MediaPlayer loadMusicResource(String filename) {
        String package_name = getPackageName();
        int soundId = resources.getIdentifier(filename, "raw", getPackageName());
        Log.i("lfl", "loadMusicAsset " + package_name + " " + filename + " " + soundId);
        return MediaPlayer.create(this, soundId);
    }

    public void playMusic(MediaPlayer mp) { mp.start(); }
    public void playBackgroundMusic(MediaPlayer mp) { mp.setLooping(true); mp.start(); }

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

    public boolean openPreferences() {
        if (preferences == null) return false;
        getFragmentManager().beginTransaction()
            .replace(R.id.content_frame, preferences).addToBackStack("Preferences").commit();
        return true;
    }

    public void openBrowser(String url) {
        final Intent intent = new Intent(Intent.ACTION_VIEW).setData(Uri.parse(url));
        startActivity(intent);
    }

    public void openListViewActivity() {
        final Intent intent = new Intent(this, com.lucidfusionlabs.app.JListViewActivity.class);
        startActivity(intent);
    }

    public void hideAds() { if (advertising != null) advertising.hideAds(); }
    public void showAds() { if (advertising != null) advertising.showAds(); }

    public int getDrawableResId(String n) { 
      return getResources().getIdentifier(n, "drawable", getPackageName());
    }

    public void showBackFragment(boolean show_content, boolean show_title) {
        int stack_size = getFragmentManager().getBackStackEntryCount();
        if (stack_size == 0) return;
        String tag = Integer.toString(stack_size-1);
        Fragment frag = getFragmentManager().findFragmentByTag(tag);
        if (show_content) getFragmentManager().beginTransaction().replace(R.id.content_frame, frag, tag).commit();
        if (show_title && frag instanceof JListViewFragment) setTitle(((JListViewFragment)frag).parent_widget.title);
    }
}

class MyGestureListener extends android.view.GestureDetector.SimpleOnGestureListener {
    public MainActivity main_activity;

    @Override
    public boolean onFling(MotionEvent e1, MotionEvent e2, float velocityX, float velocityY) {
        main_activity.AppFling(e1.getX(), e1.getY(), velocityX, velocityY);
        return false;
    }
}

class MainView extends android.view.SurfaceView implements SurfaceHolder.Callback, View.OnKeyListener, View.OnTouchListener, SensorEventListener {
    public MainActivity main_activity;
    public EGLContext egl_context;
    public EGLSurface egl_surface;
    public EGLDisplay egl_display;
    public GestureDetector gesture;
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

    public void surfaceCreated(SurfaceHolder holder) {
        Log.i("lfl", "MainView.surfaceCreated()");
        sensor.registerListener(this, sensor.getDefaultSensor(Sensor.TYPE_ACCELEROMETER), SensorManager.SENSOR_DELAY_GAME, null);
        have_surface = true;
    }

    public void surfaceChanged(SurfaceHolder holder, int format, int width, int height) {
        Log.i("lfl", "MainView.surfaceChanged(" + width + ", " + height + ")");
        main_activity.surfaceChanged(format, width, height);
    }

    public void surfaceDestroyed(SurfaceHolder holder) {
        Log.i("lfl", "surfaceDestroyed()");
        sensor.unregisterListener(this, sensor.getDefaultSensor(Sensor.TYPE_ACCELEROMETER));
        have_surface = false;
    }

    public void onSensorChanged(SensorEvent event) {
        if (event.sensor.getType() == Sensor.TYPE_ACCELEROMETER)
            main_activity.AppAccel(event.values[0], event.values[1], event.values[2]);
    }

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
        else if (event.getAction() == KeyEvent.ACTION_UP)   { main_activity.AppKeyPress(key_char, mod, 0); return true; }
        else if (event.getAction() == KeyEvent.ACTION_DOWN) { main_activity.AppKeyPress(key_char, mod, 1); return true; }
        else return false;
    }

    public boolean onTouch(View v, MotionEvent event) {
        // gesture.onTouchEvent(event);
        final int action = event.getAction() & MotionEvent.ACTION_MASK;
        if (action == MotionEvent.ACTION_MOVE) {
            for (int i = 0; i < event.getPointerCount(); i++) {
                main_activity.AppTouch(action, event.getX(i), event.getY(i), event.getPressure(i));
            }
        } else {
            int action_index = (action == MotionEvent.ACTION_POINTER_DOWN || action == MotionEvent.ACTION_POINTER_UP) ? event.getActionIndex() : 0;
            main_activity.AppTouch(action, event.getX(action_index), event.getY(action_index), event.getPressure(action_index));
        }
        return true; 
    }

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
