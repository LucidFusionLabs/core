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

import com.google.android.gms.ads.*;
import com.lucidfusionlabs.core.ModelItem;
import com.lucidfusionlabs.core.LifecycleActivity;
import com.lucidfusionlabs.core.ActivityLifecycleListenerList;

class Advertising extends com.lucidfusionlabs.core.ActivityLifecycleListener
    implements com.lucidfusionlabs.core.ViewOwner {

    public int type, valign;
    public String adUnitId;
    public List<String> testDevices;
    public AdView adView;
    public boolean started, shown;

    public static Advertising createInstance(LifecycleActivity act, int t, int v, String unitId, List<String> td) {
        return new Advertising(act, t, v, unitId, td, act.activityLifecycle);
    }

    public static Advertising createStaticInstance(LifecycleActivity act, int t, int v, String unitId, List<String> td) {
        return new Advertising(act, t, v, unitId, td, act.staticLifecycle);
    }

    protected Advertising(Activity act, int t, int v, String unitId, List<String> td, ActivityLifecycleListenerList p) {
        type = t;
        valign = v;
        testDevices = td;
        adUnitId = unitId;
        adView = createView(act);
        p.registerLifecycleListener(this);
    }

    private AdView createView(final Context context) {
        AdView ret = new AdView(context);
        if (type == ModelItem.AD_TYPE_BANNER) ret.setAdSize(AdSize.BANNER);
        else Log.e("lfl", "unknown ad type=" + type);
        ret.setAdUnitId(adUnitId);
        return ret;
    }

    @Override public void onActivityCreated(Activity activity, Bundle savedInstanceState) {
        adView = createView(activity);
    }

    @Override public void onActivityResumed(Activity activity) {
        if (adView == null) return;
        adView.resume();
    }

    @Override public void onActivityPaused(Activity activity) {
        if (adView == null) return;
        adView.pause();
    }

    @Override public void onActivityDestroyed(Activity act) {
        if (adView != null) adView.destroy();
        adView = null;
        started = false;
    }

    @Override public void clearView() {}

    @Override public View getView(final Context context) {
        if (adView == null) adView = createView(context);
        return adView; 
    }

    @Override public void onViewAttached() {
        shown = true;
        if (!started && (started = true)) { 
            Log.i("lfl", "creating adRequest");
            AdRequest.Builder builder = new AdRequest.Builder();
            for (String td : testDevices) builder.addTestDevice(td);
            adView.loadAd(builder.build());
        } else {
            adView.setVisibility(AdView.VISIBLE);
            adView.bringToFront();
        }
    }

    public void show(final Activity activity, final FrameLayout frameLayout) {
        activity.runOnUiThread(new Runnable() {
            public void run() {
                if (LifecycleActivity.replaceViewParent(adView, frameLayout))
                    frameLayout.addView(adView, new LayoutParams
                                        (LayoutParams.WRAP_CONTENT, LayoutParams.WRAP_CONTENT,
                                         ModelItem.VAlignToGravity(valign) | Gravity.CENTER_HORIZONTAL));
                onViewAttached();
            }
        });
    }

    public void hide(Activity activity) {
        activity.runOnUiThread(new Runnable() {
            public void run() {
                adView.setVisibility(AdView.INVISIBLE);
            }
        });
    }
}
