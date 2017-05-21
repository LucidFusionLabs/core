package com.lucidfusionlabs.billing;

import com.android.vending.billing.IInAppBillingService;
import android.content.Intent;
import android.content.Context;
import android.content.ComponentName;
import android.content.ServiceConnection;
import android.os.IBinder;
import android.os.Bundle;

public class PurchaseManager extends com.lucidfusionlabs.core.ActivityLifecycleListener {
    Context mContext;
    IInAppBillingService mService;
    ServiceConnection mServiceConn;

    public PurchaseManager(com.lucidfusionlabs.core.LifecycleActivity activity) {
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
    }

    public void onDestroy() {
        if (mServiceConn != null) mContext.unbindService(mServiceConn);
    }
}
