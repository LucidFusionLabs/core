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
import android.content.pm.ActivityInfo;
import android.hardware.*;
import android.util.Log;
import android.net.Uri;

import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.egl.*;

import com.google.android.gms.ads.*;

class Advertising {
    public Activity activity;
    public AdView adView;
    public AdRequest adRequest;
    public Advertising(Activity act, FrameLayout frameLayout) {
        activity = act;
        adView = new AdView(activity);
        adView.setAdSize(AdSize.BANNER);
        adView.setAdUnitId("ca-app-pub-6299262366078490/8570530772");
        frameLayout.addView(adView, new LayoutParams(LayoutParams.WRAP_CONTENT, LayoutParams.WRAP_CONTENT, Gravity.TOP + Gravity.CENTER_HORIZONTAL));
    }
    public void onDestroy() {
        if (adView != null) { adView.destroy(); adView=null; } 
    }
    public void hideAds() {
        activity.runOnUiThread(new Runnable() {
            public void run() {
                adView.setVisibility(AdView.INVISIBLE);
            }
        });
    }
    public void showAds() {
        activity.runOnUiThread(new Runnable() {
            public void run() {
                if (adRequest == null) { 
                    Log.i("lfl", "creating adRequest");
                    adRequest = new AdRequest.Builder()
                        .addTestDevice("0DBA42FB5610516D099B960C0424B343")
                        .addTestDevice("52615418751B72981EF42A9681544EDB")
                        .build();
                    adView.loadAd(adRequest);
                } else {
                    adView.setVisibility(AdView.VISIBLE);
                    adView.bringToFront();
                }
            }
        });
    }
}
