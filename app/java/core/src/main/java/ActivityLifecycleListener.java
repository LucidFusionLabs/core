package com.lucidfusionlabs.core;

import android.os.Bundle;
import android.content.Intent;
import android.app.Activity;
import android.app.Application.ActivityLifecycleCallbacks;

public class ActivityLifecycleListener implements ActivityLifecycleCallbacks {
    @Override
    public void onActivityCreated(Activity activity, Bundle savedInstanceState) {}
    
    @Override
    public void onActivityStarted(Activity activity) {}
    
    @Override
    public void onActivityResumed(Activity activity) {}
    
    @Override
    public void onActivityPaused(Activity activity) {}
    
    @Override
    public void onActivityStopped(Activity activity) {}
    
    @Override
    public void onActivitySaveInstanceState(Activity activity, Bundle outState) {}

    public void onActivityResult(Activity activity, int request, int response, Intent data) {}

    @Override
    public void onActivityDestroyed(Activity activity) {}
}
