package com.lucidfusionlabs.app;

import java.net.*;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.HashMap;
import java.util.concurrent.Callable;
import java.util.concurrent.FutureTask;

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
import android.graphics.Bitmap;
import android.app.AlertDialog;
import android.preference.PreferenceManager;
import android.support.v4.app.Fragment;
import android.support.v4.app.FragmentManager;
import android.support.v7.app.ActionBar;

public class MainActivity extends com.lucidfusionlabs.core.LifecycleActivity {
    static { System.loadLibrary("native-lib"); }
    
    public native void nativeCreate();
    public native void nativeMain();
    public native void nativeNewMainLoop(boolean reset);
    public native void nativeMinimize();
    public native void nativeReshaped(int x, int y, int w, int h);
    public native void nativeKeyPress(int keycode, int mod, int down);
    public native void nativeTouch(int action, float x, float y, float p);
    public native void nativeFling(float x, float y, float vx, float vy);
    public native void nativeScroll(float x, float y, float sx, float sy);
    public native void nativeAccel(float x, float y, float z);
    public native void nativeScale(float x, float y, float dx, float dy, boolean begin);
    public static native void nativeShellRun(String text);

    public static boolean native_created, native_init, disable_title;
    public static Screens screens = new Screens();
    public static ArrayList<Bitmap> bitmaps = new ArrayList<Bitmap>();

    public Resources resources;
    public FrameLayout main_layout, frame_layout;
    public ActionBar action_bar;
    public Window root_window;
    public View root_view;
    public MainView view;
    public Thread thread;
    public AudioManager audio;
    public boolean waiting_activity_result;
    public int surface_width, surface_height, egl_version;
    public int attr_listPreferredItemHeight, attr_scrollbarSize;
    public float display_density;
    public PreferenceFragment preference_fragment;
    public HashMap<String, String> preference_default = new HashMap<String, String>();
    public SharedPreferences preferences;
    public Handler handler = new Handler();

    protected void onCreated() {}

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        Log.i("lfl", "MainActivity.onCreate() native_created=" + native_created);
        preferences = PreferenceManager.getDefaultSharedPreferences(this);
        setTheme(preferences.getString("theme", "Dark").equals("Light") ?
                 android.support.v7.appcompat.R.style.Theme_AppCompat_Light :
                 android.support.v7.appcompat.R.style.Theme_AppCompat);
        super.onCreate(savedInstanceState);

        if (!isTaskRoot()) {
            Log.i("lfl", "MainActivity.onCreate() isTaskRoot() == false");
            finish();
            return;
        }

        onCreated();
        Context context = getApplication();
        LayoutInflater inflater = (LayoutInflater)context.getSystemService(LAYOUT_INFLATER_SERVICE);
        main_layout = (FrameLayout)inflater.inflate(R.layout.main, null);
        root_window = getWindow();
        resources = getResources();
        view = new MainView(this, context);
        audio = (AudioManager)getSystemService(Context.AUDIO_SERVICE);
        action_bar = getSupportActionBar();

        if (!native_created) {
            native_created = true;
            ScreenFragmentNavigator.clearFragmentBackstack(this);
            Log.i("lfl", "MainActivity.nativeCreate()");
            nativeCreate();
        } else {
            if (disable_title) disableTitle();
        }

        setContentView(main_layout);
        frame_layout = new FrameLayout(this);
        main_layout.addView(frame_layout, new LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.MATCH_PARENT));
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
                if (!native_init) return;
                Rect r = new Rect();
                View view = root_window.getDecorView();
                view.getWindowVisibleDisplayFrame(r);

                int actionbar_h = action_bar.isShowing() ? action_bar.getHeight() : 0;
                int h = r.bottom - r.top - actionbar_h;
                for (Toolbar toolbar : screens.toolbar_bottom) {
                    View tb = toolbar.view;
                    if (tb == null) continue;
                    FrameLayout.LayoutParams params = (FrameLayout.LayoutParams)tb.getLayoutParams();
                    params.bottomMargin = surface_height - h;
                    tb.setLayoutParams(params);
                    h -= tb.getHeight();
                }

                // Log.i("lfl", "onGlobalLayout(" + r.left + ", " + r.top + ", " + r.right + ", " + r.bottom + ", " + actionbar_h + ") = " + h);
                nativeReshaped(r.left, surface_height - h, r.right - r.left, h); 
            }});

        final MainActivity self = this;
        getSupportFragmentManager().addOnBackStackChangedListener(new FragmentManager.OnBackStackChangedListener() {
            @Override public void onBackStackChanged() {
                int stack_size = getSupportFragmentManager().getBackStackEntryCount() ;
                if (stack_size > 1 || (stack_size == 1 && ScreenFragmentNavigator.haveFragmentNavLeft(self, "0"))) {
                    action_bar.setHomeButtonEnabled(true);
                    action_bar.setDisplayHomeAsUpEnabled(true);
                } else {
                    action_bar.setDisplayHomeAsUpEnabled(false);
                    action_bar.setHomeButtonEnabled(false);
                }
            }});

        staticLifecycle.onActivityCreated(this, savedInstanceState);
    }

    @Override
    protected void onStart() {
        Log.i("lfl", "MainActivity.onStart()");
        super.onStart();
        staticLifecycle.onActivityStarted(this);
        activityLifecycle.onActivityStarted(this);
    }

    @Override
    public void onResume() {
        boolean have_surface = view.have_surface, thread_exists = thread != null;
        Log.i("lfl", "MainActivity.onResume() have_surface=" + have_surface + " thread_exists=" + thread_exists);
        super.onResume();
        screens.onResume(this);
        staticLifecycle.onActivityResumed(this);
        activityLifecycle.onActivityResumed(this);
        if (have_surface && !thread_exists) startRenderThread(false);
    }
    
    @Override
    protected void onPause() {
        Log.i("lfl", "MainActivity.onPause() enter");
        super.onPause();
        staticLifecycle.onActivityPaused(this);
        activityLifecycle.onActivityPaused(this);
        if (waiting_activity_result || thread == null) return;

        Thread t = thread;
        thread = null;        
        nativeMinimize();
        try { t.join(); }
        catch(Exception e) { Log.e("lfl", e.toString()); }
        Log.i("lfl", "MainActivity.onPause() exit");
    }
    
    @Override
    protected void onStop() {
        Log.i("lfl", "MainActivity.onStop()");
        super.onStop();
        staticLifecycle.onActivityStopped(this);
        activityLifecycle.onActivityStopped(this);
    }

    @Override
    protected void onDestroy() {
        Log.i("lfl", "MainActivity.onDestroy()");
        staticLifecycle.onActivityDestroyed(this);
        activityLifecycle.onActivityDestroyed(this);
        screens.onDestroy();
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
        staticLifecycle.onActivityResult(this, request, response, data);
        activityLifecycle.onActivityResult(this, request, response, data);
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        switch (item.getItemId()) {
            case android.R.id.home:
                onBackPressed();
                return true;
        }
        return super.onOptionsItemSelected(item);
    }

    @Override
    public void onBackPressed() { ScreenFragmentNavigator.onBackPressed(this); }

    public void superOnBackPressed() { super.onBackPressed(); } 

    public void surfaceChanged(int format, int width, int height) {
        surface_width = width;
        surface_height = height;
        boolean thread_exists = thread != null;
        Log.i("lfl", "surfaceChanged thread_exists=" + thread_exists);
        nativeReshaped(0, 0, width, height);
        if (thread_exists) return;
        startRenderThread(true);
    }

    public void startRenderThread(boolean native_reset) {
        Log.i("lfl", "startRenderThread native_init=" + native_init + " native_reset=" + native_reset);
        screens.onStartRenderThread(this, frame_layout);
        if      (!native_init) thread = new Thread(new Runnable() { public void run() { view.initEGL();        nativeMain       ();      } }, "JNIMainThread");
        else if (native_reset) thread = new Thread(new Runnable() { public void run() { view.initEGL();        nativeNewMainLoop(true);  } }, "JNIMainThread");
        else                   thread = new Thread(new Runnable() { public void run() { view.makeCurrentEGL(); nativeNewMainLoop(false); } }, "JNIMainThread");
        thread.start();
        native_init = true;
    }

    public void forceExit() {
        Log.i("lfl", "MainActivity.forceExit()");
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
        imm.hideSoftInputFromWindow(view.getWindowToken(), 0);
    }

    public void hideKeyboardAfterEnter() {
        android.view.inputmethod.InputMethodManager imm = (android.view.inputmethod.InputMethodManager)getSystemService(Context.INPUT_METHOD_SERVICE);
        imm.hideSoftInputFromWindow(view.getWindowToken(), android.view.inputmethod.InputMethodManager.HIDE_NOT_ALWAYS); 
    }

    public void disableTitle() {
        disable_title = true;
        action_bar.hide();
        // requestWindowFeature(Window.FEATURE_NO_TITLE);
    }

    public void enableKeepScreenOn(boolean enabled) {
        if (enabled) root_window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
    }

    public void enablePanRecognizer(boolean enabled) {
    }

    public void enablePinchRecognizer(final boolean enabled) {
        final MainActivity self = this;
        runOnUiThread(new Runnable() { public void run() {
            if (!enabled) view.scale_gesture = null;
            else view.scale_gesture = new ScaleGestureDetector
                (self, new ScaleGestureDetector.SimpleOnScaleGestureListener() {
                    @Override
                    public boolean onScale(ScaleGestureDetector g) {
                        nativeScale(g.getFocusX(), g.getFocusY(),
                                    g.getCurrentSpanX() - g.getPreviousSpanX(),
                                    g.getCurrentSpanY() - g.getPreviousSpanY(), !g.isInProgress());
                        return true;
                    }});
        }});
    }

    public void setCaption(final String text) {
        runOnUiThread(new Runnable() { public void run() { setTitle(text); } });
    }

    public void setTheme(final String text) {
        Log.i("lfl", "setTheme: " + text);
        runOnUiThread(new Runnable() { public void run() { recreate(); } });
    }

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
        if (preference_fragment == null) return false;
        getSupportFragmentManager().beginTransaction()
            .replace(R.id.content_frame, preference_fragment).addToBackStack("Preferences").commit();
        return true;
    }

    public void openBrowser(String url) {
        final Intent intent = new Intent(Intent.ACTION_VIEW).setData(Uri.parse(url));
        startActivity(intent);
    }

    public int getDrawableResId(final String n) { 
        if (n.length() == 0) {
            FutureTask<Integer> future = new FutureTask<Integer>(new Callable<Integer>(){
                public Integer call() throws Exception {
                    bitmaps.add(null);
                    return -bitmaps.size();
                }});
            try { runOnUiThread(future); return future.get(); }
            catch(Exception e) { return 0; }
        } else {
            return getResources().getIdentifier(n, "drawable", getPackageName());
        }
    }

    public void updateBitmap(final int encoded_id, final int width, final int height, final int fmt, final int[] pixels) {
        runOnUiThread(new Runnable() { public void run() {
            final int id = -encoded_id - 1;
            if (id < 0 || id >= bitmaps.size()) throw new java.lang.IllegalArgumentException();
            bitmaps.set(id, Bitmap.createBitmap(pixels, width, height, Bitmap.Config.ARGB_8888));
        }});
    }

    public String getVersionName() {
        try { return getPackageManager().getPackageInfo(getPackageName(), 0).versionName; }
        catch(Exception e) { return ""; }
    }

    public void setDefaultPreferences(final ArrayList<Pair<String, String>> prefs) {
        for (Pair<String, String> i : prefs) preference_default.put(i.first, i.second);
    }

    public void updatePreferences(final ArrayList<Pair<String, String>> prefs) {
        SharedPreferences.Editor editor = preferences.edit();
        for (Pair<String, String> i : prefs) editor.putString(i.first, i.second);
        editor.apply();
    }

    public String getPreference(final String key) {
        String dp = preference_default.get(key);
        try { return preferences.getString(key, dp != null ? dp : ""); }
        catch(Exception e) { return dp != null ? dp : ""; }
    }

    public int maxVolume() { return audio.getStreamMaxVolume(AudioManager.STREAM_MUSIC); }
    public int getVolume() { return audio.getStreamVolume(AudioManager.STREAM_MUSIC); }
    public void setVolume(int v) { audio.setStreamVolume(AudioManager.STREAM_MUSIC, v, 0); }

    public MediaPlayer loadMusicResource(String filename) {
        String package_name = getPackageName();
        int soundId = resources.getIdentifier(filename, "raw", getPackageName());
        Log.i("lfl", "loadMusicAsset " + package_name + " " + filename + " " + soundId);
        return MediaPlayer.create(this, soundId);
    }

    public void playMusic(MediaPlayer mp) { mp.start(); }
    public void playBackgroundMusic(MediaPlayer mp) { mp.setLooping(true); mp.start(); }
}
