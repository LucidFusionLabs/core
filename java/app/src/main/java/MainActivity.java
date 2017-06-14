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
import com.lucidfusionlabs.core.ModelItem;

public class MainActivity extends com.lucidfusionlabs.core.LifecycleActivity
    implements FragmentManager.OnBackStackChangedListener, View.OnKeyListener, View.OnTouchListener,
    SensorEventListener {

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

    public Handler handler = new Handler();
    public Resources resources;
    public FrameLayout main_layout;
    public LinearLayout gl_layout;
    public ActionBar action_bar;
    public Window root_window;
    public View root_view;
    public OpenGLView gl_view;
    public AudioManager audio;
    public SensorManager sensor;
    public ClipboardManager clipboard;
    public boolean waiting_activity_result;
    public int attr_listPreferredItemHeight, attr_scrollbarSize;
    public float display_density;
    public ArrayList<ModelItem> context_menu;
    public SharedPreferences preferences;
    public HashMap<String, String> preference_default = new HashMap<String, String>();
    public GestureDetector gesture_detector;
    public ScaleGestureDetector scale_detector;

    protected void onCreated() {}
    protected PreferenceFragment createPreferenceFragment() { return null; }

    @Override protected void onCreate(Bundle savedInstanceState) {
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
        root_window = getWindow();
        root_view = root_window.getDecorView().findViewById(android.R.id.content);
        main_layout = (FrameLayout)inflater.inflate(R.layout.main, null);
        resources = getResources();
        audio = (AudioManager)getSystemService(Context.AUDIO_SERVICE);
        sensor = (SensorManager)getSystemService("sensor");
        clipboard = (ClipboardManager)getSystemService(Context.CLIPBOARD_SERVICE);
        action_bar = getSupportActionBar();
        display_density = resources.getDisplayMetrics().density;

        int[] attr = new int[] { android.R.attr.listPreferredItemHeight, android.R.attr.scrollbarSize };
        android.content.res.TypedArray attr_val = context.obtainStyledAttributes(attr);
        attr_listPreferredItemHeight = attr_val.getDimensionPixelSize(0, -1);
        attr_scrollbarSize           = attr_val.getDimensionPixelSize(1, -1);

        gl_view = new OpenGLView(this, context);
        gl_view.setOnKeyListener(this); 
        gl_view.setOnTouchListener(this);
        gl_view.requestFocus();

        if (!native_created) {
            native_created = true;
            ScreenFragmentNavigator.clearFragmentBackstack(this);
            Log.i("lfl", "MainActivity.nativeCreate()");
            nativeCreate();
        } else {
            if (disable_title) disableTitle();
        }

        gl_layout = new LinearLayout(this);
        gl_layout.setOrientation(LinearLayout.VERTICAL);
        gl_layout.addView(gl_view, new LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, 0, 1.0f));
        main_layout.addView(gl_layout, new LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.MATCH_PARENT));
        setContentView(main_layout);

        getSupportFragmentManager().addOnBackStackChangedListener(this);
        staticLifecycle.onActivityCreated(this, savedInstanceState);
    }

    @Override protected void onStart() {
        Log.i("lfl", "MainActivity.onStart()");
        super.onStart();
        staticLifecycle.onActivityStarted(this);
        activityLifecycle.onActivityStarted(this);
    }

    @Override public void onResume() {
        super.onResume();
        screens.onResume(this);
        staticLifecycle.onActivityResumed(this);
        activityLifecycle.onActivityResumed(this);
        gl_view.onResume();
    }
    
    @Override protected void onPause() {
        Log.i("lfl", "MainActivity.onPause() enter");
        super.onPause();
        staticLifecycle.onActivityPaused(this);
        activityLifecycle.onActivityPaused(this);
        if (!waiting_activity_result) gl_view.onPause();
        Log.i("lfl", "MainActivity.onPause() exit");
    }
    
    @Override protected void onStop() {
        Log.i("lfl", "MainActivity.onStop()");
        super.onStop();
        staticLifecycle.onActivityStopped(this);
        activityLifecycle.onActivityStopped(this);
    }

    @Override protected void onDestroy() {
        Log.i("lfl", "MainActivity.onDestroy()");
        staticLifecycle.onActivityDestroyed(this);
        activityLifecycle.onActivityDestroyed(this);
        screens.onDestroy();
        super.onDestroy();
    }

    @Override public void onConfigurationChanged(Configuration newConfig) {
        Log.i("lfl", "onConfigurationChanged");
        super.onConfigurationChanged(newConfig);
    }

    @Override protected void onActivityResult(int request, int response, Intent data) {
        Log.i("lfl", "MainActivity.onActivityResult(" + request + ", " + response + ")");
        waiting_activity_result = false;
        super.onActivityResult(request, response, data);
        staticLifecycle.onActivityResult(this, request, response, data);
        activityLifecycle.onActivityResult(this, request, response, data);
    }

    @Override public void onBackStackChanged() {
        ScreenFragmentNavigator.updateHomeButton(this);
    }

    @Override public boolean onOptionsItemSelected(MenuItem item) {
        switch (item.getItemId()) {
            case android.R.id.home:
                onBackPressed();
                return true;
        }
        return super.onOptionsItemSelected(item);
    }

    @Override public void onCreateContextMenu(ContextMenu menu, View v, ContextMenu.ContextMenuInfo menuInfo) {
        if (context_menu == null) return;
        for (int i=0, l=context_menu.size(); i < l; i++) {
            ModelItem item = context_menu.get(i);
            menu.add(0, i, 0, item.val.equals("Keyboard") ? "Toggle Keyboard" : item.val);
        }
    }

    @Override public boolean onContextItemSelected(MenuItem i) {
        if (context_menu == null || i.getItemId() >= context_menu.size()) return false;
        ModelItem item = context_menu.get(i.getItemId());
        if (item.val.equals("Keyboard")) toggleKeyboard();
        else if (item.val.equals("Paste")) paste(getClipboardText());
        else if (item.cb != null) item.cb.run();
        return true;
    }

    @Override public void onBackPressed() { ScreenFragmentNavigator.onBackPressed(this); }

    public void superOnBackPressed() { super.onBackPressed(); } 

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
        imm.showSoftInput(gl_view, android.view.inputmethod.InputMethodManager.SHOW_FORCED);
    }

    public void hideKeyboard() {
        android.view.inputmethod.InputMethodManager imm = (android.view.inputmethod.InputMethodManager)getSystemService(Context.INPUT_METHOD_SERVICE);
        imm.hideSoftInputFromWindow(gl_view.getWindowToken(), 0);
    }

    public void hideKeyboardAfterEnter() {
        android.view.inputmethod.InputMethodManager imm = (android.view.inputmethod.InputMethodManager)getSystemService(Context.INPUT_METHOD_SERVICE);
        imm.hideSoftInputFromWindow(gl_view.getWindowToken(), android.view.inputmethod.InputMethodManager.HIDE_NOT_ALWAYS);
    }

    public void disableTitle() {
        disable_title = true;
        action_bar.hide();
    }

    public void enableKeepScreenOn(final boolean enabled) {
        if (enabled) root_window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
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
        PreferenceFragment preference_fragment = createPreferenceFragment();;
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

    public String getClipboardText() {
        if (!clipboard.hasPrimaryClip() ||
            !clipboard.getPrimaryClipDescription().hasMimeType(android.content.ClipDescription.MIMETYPE_TEXT_PLAIN)) return "";
        ClipData.Item item = clipboard.getPrimaryClip().getItemAt(0);
        String pasteData = item == null ? null : item.getText().toString();
        return pasteData == null ? "" : pasteData;
    }

    public void setClipboardText(String v) {
        clipboard.setPrimaryClip(ClipData.newPlainText("text", v));
    }

    public String getCurrentStackTrace() {
        return java.util.Arrays.toString(Thread.currentThread().getStackTrace());
    }

    public void showContextMenu(final ArrayList<ModelItem> model) {
        runOnUiThread(new Runnable() { public void run() {
            context_menu = model;
            registerForContextMenu(gl_layout);
            openContextMenu(gl_layout);
            unregisterForContextMenu(gl_layout);
        }});
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

    public void paste(String s) {
        for (char c : s.toCharArray()) {
            nativeKeyPress(c, 0, 1);
            nativeKeyPress(c, 0, 0);
        }
    }

    public void enableAccelerometer(final boolean enabled) {
        // sensor.registerListener(this, sensor.getDefaultSensor(Sensor.TYPE_ACCELEROMETER), SensorManager.SENSOR_DELAY_GAME, null);
        // sensor.unregisterListener(this, sensor.getDefaultSensor(Sensor.TYPE_ACCELEROMETER));
    }

    public void enablePanRecognizer(final boolean enabled) {
    }

    public void enablePinchRecognizer(final boolean enabled) {
        final MainActivity self = this;
        runOnUiThread(new Runnable() { public void run() {
            if (!enabled) scale_detector = null;
            else scale_detector = new ScaleGestureDetector
                (self, new ScaleGestureDetector.SimpleOnScaleGestureListener() {
                    @Override
                    public boolean onScale(ScaleGestureDetector g) {
                        float dx = g.getCurrentSpanX() - g.getPreviousSpanX();
                        float dy = g.getCurrentSpanY() - g.getPreviousSpanY();
                        Log.i("lfl", "scale " +  g.getFocusX() + " " + g.getFocusY() + " " + dx + " " + dy + " " + g.isInProgress());
                        nativeScale(g.getFocusX(), g.getFocusY(), dx, dy, !g.isInProgress());
                        return true;
                    }});
        }});
    }

    @Override public boolean onKey(View v, int key_code, KeyEvent event) {
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
        else if (event.getAction() == KeyEvent.ACTION_UP)   { nativeKeyPress(key_char, mod, 0); return true; }
        else if (event.getAction() == KeyEvent.ACTION_DOWN) { nativeKeyPress(key_char, mod, 1); return true; }
        else return false;
    }

    @Override public boolean onTouch(View v, MotionEvent event) {
        // gesture.onTouchEvent(event);
        if (scale_detector != null) scale_detector.onTouchEvent(event);
        final int action = event.getAction() & MotionEvent.ACTION_MASK;
        if (action == MotionEvent.ACTION_MOVE) {
            if (scale_detector == null || !scale_detector.isInProgress()) {
                for (int i = 0; i < event.getPointerCount(); i++) {
                    nativeTouch(action, event.getX(i), event.getY(i), event.getPressure(i));
                }
            }
        } else {
            int action_index = (action == MotionEvent.ACTION_POINTER_DOWN || action == MotionEvent.ACTION_POINTER_UP) ? event.getActionIndex() : 0;
            nativeTouch(action, event.getX(action_index), event.getY(action_index), event.getPressure(action_index));
        }
        return true; 
    }

    @Override public void onAccuracyChanged(Sensor sensor, int accuracy) {}

    @Override public void onSensorChanged(SensorEvent event) {
        if (event.sensor.getType() == Sensor.TYPE_ACCELEROMETER)
            nativeAccel(event.values[0], event.values[1], event.values[2]);
    }
}

class MyGestureListener extends android.view.GestureDetector.SimpleOnGestureListener {
    public MainActivity main_activity;

    @Override
    public boolean onFling(MotionEvent e1, MotionEvent e2, float velocityX, float velocityY) {
        main_activity.nativeFling(e1.getX(), e1.getY(), velocityX, velocityY);
        return false;
    }
}
