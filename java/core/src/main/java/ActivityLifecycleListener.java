package com.lucidfusionlabs.core;

import android.os.Bundle;
import android.content.Intent;

public class ActivityLifecycleListener {
    public void onActivityCreated(LifecycleActivity activity, Bundle savedInstanceState) {}
    public void onActivityStarted(LifecycleActivity activity) {}
    public void onActivityResumed(LifecycleActivity activity) {}
    public void onActivityPaused(LifecycleActivity activity) {}
    public void onActivityStopped(LifecycleActivity activity) {}
    public void onActivitySaveInstanceState(LifecycleActivity activity, Bundle outState) {}
    public void onActivityResult(LifecycleActivity activity, int request, int response, Intent data) {}
    public void onActivityDestroyed(LifecycleActivity activity) {}
}
