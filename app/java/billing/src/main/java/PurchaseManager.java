package com.lucidfusionlabs.billing;

import com.android.vending.billing.IInAppBillingService;
import android.content.Intent;
import android.content.Context;
import android.content.ComponentName;
import android.content.ServiceConnection;
import android.os.IBinder;
import android.os.Bundle;
import android.app.Activity;

public class PurchaseManager {
    IInAppBillingService mService;
    ServiceConnection mServiceConn;

    public PurchaseManager(Activity activity) {
        final String packageName = activity.getPackageName();
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
        activity.bindService(serviceIntent, mServiceConn, Context.BIND_AUTO_CREATE);
    }
}
