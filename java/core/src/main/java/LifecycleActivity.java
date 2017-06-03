package com.lucidfusionlabs.core;

import android.view.View;
import android.view.ViewGroup;
import android.view.ViewParent;
import android.support.v7.app.AppCompatActivity;

public class LifecycleActivity extends AppCompatActivity {
    public static ActivityLifecycleListenerList   staticLifecycle = new ActivityLifecycleListenerList();
    public        ActivityLifecycleListenerList activityLifecycle = new ActivityLifecycleListenerList();

    public String getAndroidId() {
        return android.provider.Settings.Secure.getString
            (getContentResolver(), android.provider.Settings.Secure.ANDROID_ID);
    }

    public static boolean replaceViewParent(View v, ViewParent new_parent) {
        ViewParent p = v.getParent();
        if (p == new_parent) return false;
        if (p != null && (p instanceof ViewGroup)) ((ViewGroup)p).removeView(v);
        return true;
    }
}
