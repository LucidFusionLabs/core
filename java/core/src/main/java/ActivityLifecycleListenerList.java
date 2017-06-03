package com.lucidfusionlabs.core;

import java.util.HashMap;
import java.util.ArrayList;
import android.util.Log;
import android.os.Bundle;
import android.content.Intent;

public class ActivityLifecycleListenerList extends ActivityLifecycleListener {
    private ArrayList<ActivityLifecycleListener> registeredListeners = new ArrayList<>();

    public void registerLifecycleListener  (ActivityLifecycleListener listener) { registeredListeners.add   (listener); }
    public void unregisterLifecycleListener(ActivityLifecycleListener listener) { registeredListeners.remove(listener); }
    public ActivityLifecycleListener[] copyRegisteredListeners() {
        ActivityLifecycleListener[] ret = new ActivityLifecycleListener[registeredListeners.size()];
        registeredListeners.toArray(ret);
        return ret;
    }

    @Override
    public void onActivityCreated(LifecycleActivity activity, Bundle savedInstanceState) {
        ActivityLifecycleListener[] lifecycleListeners = copyRegisteredListeners();
        for (ActivityLifecycleListener c : lifecycleListeners) c.onActivityCreated(activity, savedInstanceState);
    }
    
    @Override
    public void onActivityStarted(LifecycleActivity activity) {
        ActivityLifecycleListener[] lifecycleListeners = copyRegisteredListeners();
        for (ActivityLifecycleListener c : lifecycleListeners) c.onActivityStarted(activity);
    }
    
    @Override
    public void onActivityResumed(LifecycleActivity activity) {
        ActivityLifecycleListener[] lifecycleListeners = copyRegisteredListeners();
        for (ActivityLifecycleListener c : lifecycleListeners) c.onActivityResumed(activity);
    }
    
    @Override
    public void onActivityPaused(LifecycleActivity activity) {
        ActivityLifecycleListener[] lifecycleListeners = copyRegisteredListeners();
        for (ActivityLifecycleListener c : lifecycleListeners) c.onActivityPaused(activity);
    }
    
    @Override
    public void onActivityStopped(LifecycleActivity activity) {
        ActivityLifecycleListener[] lifecycleListeners = copyRegisteredListeners();
        for (ActivityLifecycleListener c : lifecycleListeners) c.onActivityStopped(activity);
    }
    
    @Override
    public void onActivitySaveInstanceState(LifecycleActivity activity, Bundle outState) {
        ActivityLifecycleListener[] lifecycleListeners = copyRegisteredListeners();
        for (ActivityLifecycleListener c : lifecycleListeners) c.onActivitySaveInstanceState(activity, outState);
    }

    @Override
    public void onActivityResult(LifecycleActivity activity, int request, int response, Intent data) {
        ActivityLifecycleListener[] lifecycleListeners = copyRegisteredListeners();
        for (ActivityLifecycleListener c : lifecycleListeners) c.onActivityResult(activity, request, response, data);
    }

    @Override
    public void onActivityDestroyed(LifecycleActivity activity) {
        ActivityLifecycleListener[] lifecycleListeners = copyRegisteredListeners();
        for (ActivityLifecycleListener c : lifecycleListeners) c.onActivityDestroyed(activity);
    }
}
