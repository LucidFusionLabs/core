package com.lucidfusionlabs.ads;

import java.net.*;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import android.app.Activity;
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

import com.google.android.gms.ads.*;
import com.lucidfusionlabs.core.LifecycleActivity;
import com.lucidfusionlabs.core.ActivityLifecycleListenerList;

class Advertising extends com.lucidfusionlabs.core.ActivityLifecycleListener {
    public List<String> testDevices;
    public String adUnitId;
    public AdView adView;
    public AdRequest adRequest;

    protected Advertising(Activity act, String unitId, List<String> td, ActivityLifecycleListenerList p) {
        testDevices = td;
        adUnitId = unitId;
        onActivityCreated(act, null);
        p.registerLifecycleListener(this);
    }

    public static Advertising createInstance(LifecycleActivity act, String unitId, List<String> td) {
        return new Advertising(act, unitId, td, act.activityLifecycle);
    }

    public static Advertising createStaticInstance(LifecycleActivity act, String unitId, List<String> td) {
        return new Advertising(act, unitId, td, act.staticLifecycle);
    }
    
    @Override public void onActivityCreated(Activity activity, Bundle savedInstanceState) {
        adView = new AdView(activity);
        adView.setAdSize(AdSize.BANNER);
        adView.setAdUnitId(adUnitId);
    }

    @Override public void onActivityDestroyed(Activity act) {
        if (adView == null) return;
        adView.destroy();
        adView = null;
    }

    public void hide(Activity activity) {
        activity.runOnUiThread(new Runnable() {
            public void run() {
                adView.setVisibility(AdView.INVISIBLE);
            }
        });
    }

    public void show(final Activity activity, final FrameLayout frameLayout, final LayoutParams layoutParams) {
        activity.runOnUiThread(new Runnable() {
            public void run() {
                if (LifecycleActivity.replaceViewParent(adView, frameLayout))
                    frameLayout.addView(adView, layoutParams);

                if (adRequest == null) { 
                    Log.i("lfl", "creating adRequest");
                    AdRequest.Builder builder = new AdRequest.Builder();
                    for (String td : testDevices) builder.addTestDevice(td);
                    adRequest = builder.build();
                    adView.loadAd(adRequest);
                } else {
                    adView.setVisibility(AdView.VISIBLE);
                    adView.bringToFront();
                }
            }
        });
    }

    public void showTop(final Activity activity, final FrameLayout frameLayout) {
        show(activity, frameLayout, new LayoutParams(LayoutParams.WRAP_CONTENT, LayoutParams.WRAP_CONTENT,
                                                     Gravity.TOP + Gravity.CENTER_HORIZONTAL));
    }
}
