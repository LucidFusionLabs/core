package com.lucidfusionlabs.billing;

import android.app.Activity;
import android.content.Intent;
import android.content.Context;
import android.content.ComponentName;
import android.content.ServiceConnection;
import android.os.IBinder;
import android.os.Bundle;
import com.android.vending.billing.IInAppBillingService;
import com.lucidfusionlabs.core.LifecycleActivity;
import com.lucidfusionlabs.core.ActivityLifecycleListenerList;

public class PurchaseManager extends com.lucidfusionlabs.core.ActivityLifecycleListener {
    Context mContext;
    IInAppBillingService mService;
    ServiceConnection mServiceConn;

    public static PurchaseManager createInstance(LifecycleActivity act) {
        return new PurchaseManager(act, act.activityLifecycle);
    }

    public static PurchaseManager createStaticInstance(LifecycleActivity act) {
        return new PurchaseManager(act, act.staticLifecycle);
    }

    protected PurchaseManager(com.lucidfusionlabs.core.LifecycleActivity activity, ActivityLifecycleListenerList p) {
        final String packageName = activity.getPackageName();
        mContext = activity;
        mServiceConn = new ServiceConnection() {
             @Override
             public void onServiceDisconnected(ComponentName name) { mService = null; }
          
             @Override
             public void onServiceConnected(ComponentName name, IBinder service) {
                 mService = IInAppBillingService.Stub.asInterface(service);
                 try { Bundle ownedItems = mService.getPurchases(3, packageName, "inapp", null); }
                 catch(Exception e) {}
             }
        };

        Intent serviceIntent =
            new Intent("com.android.vending.billing.InAppBillingService.BIND");
        serviceIntent.setPackage("com.android.vending");
        mContext.bindService(serviceIntent, mServiceConn, Context.BIND_AUTO_CREATE);

        p.registerLifecycleListener(this);
    }

    @Override public void onActivityDestroyed(Activity act) {
        if (mServiceConn != null) mContext.unbindService(mServiceConn);
    }
}
