package com.lucidfusionlabs.billing;

import java.util.ArrayList;
import java.util.HashMap;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.security.PublicKey;
import java.security.Signature;
import java.security.KeyFactory;
import android.app.Activity;
import android.content.Intent;
import android.content.Context;
import android.content.ComponentName;
import android.content.ServiceConnection;
import android.os.IBinder;
import android.os.Bundle;
import android.util.Log;
import android.util.Base64;
import com.android.vending.billing.IInAppBillingService;
import com.lucidfusionlabs.core.LifecycleActivity;
import com.lucidfusionlabs.core.ActivityLifecycleListenerList;

public class PurchaseManager extends com.lucidfusionlabs.core.ActivityLifecycleListener {
    public static final String FILENAME = "purchase.ser";
    public static final String KEY_FACTORY_ALGORITHM = "RSA";
    public static final String SIGNATURE_ALGORITHM = "SHA1withRSA";

    final String packageName;
    final PublicKey mPubKey;
    LifecycleActivity mContext;
    IInAppBillingService mService;
    ServiceConnection mServiceConn;
    HashMap<String, String> mProducts = new HashMap<String, String>();

    public static PurchaseManager createInstance(LifecycleActivity act, String pubkey) {
        return new PurchaseManager(act, pubkey, act.activityLifecycle);
    }

    public static PurchaseManager createStaticInstance(LifecycleActivity act, String pubkey) {
        return new PurchaseManager(act, pubkey, act.staticLifecycle);
    }

    protected PurchaseManager(LifecycleActivity activity, String pubkey, ActivityLifecycleListenerList p) {
        packageName = activity.getPackageName();
        mPubKey = generatePublicKey(pubkey);
        mServiceConn = new ServiceConnection() {
             @Override
             public void onServiceDisconnected(ComponentName name) { mService = null; }
          
             @Override
             public void onServiceConnected(ComponentName name, IBinder service) {
                 mService = IInAppBillingService.Stub.asInterface(service);
             }
        };

        onActivityCreated(activity, null);
        p.registerLifecycleListener(this);
        loadPurchasesFromInternalStorage();
    }

    @Override public void onActivityCreated(LifecycleActivity activity, Bundle savedInstanceState) {
        mContext = activity;
        if (mServiceConn == null) return;
        Intent serviceIntent =
            new Intent("com.android.vending.billing.InAppBillingService.BIND");
        serviceIntent.setPackage("com.android.vending");
        mContext.bindService(serviceIntent, mServiceConn, Context.BIND_AUTO_CREATE);
    }

    @Override public void onActivityDestroyed(LifecycleActivity act) {
        if (mServiceConn == null) return;
        mContext.unbindService(mServiceConn);
    }

    public static PublicKey generatePublicKey(String encodedPublicKey) {
        try {
            byte[] decodedKey = Base64.decode(encodedPublicKey, Base64.DEFAULT);
            KeyFactory keyFactory = KeyFactory.getInstance(KEY_FACTORY_ALGORITHM);
            return keyFactory.generatePublic(new java.security.spec.X509EncodedKeySpec(decodedKey));
        } catch (java.security.NoSuchAlgorithmException e) {
            throw new RuntimeException(e);
        } catch (java.security.spec.InvalidKeySpecException e) {
            Log.e("lfl", "Invalid key specification.");
            throw new IllegalArgumentException(e);
        }
    }

    public static boolean verifyPurchase(PublicKey pubkey, String signedData, String signature) {
        if (signedData.length() == 0 || signature.length() == 0) return false;
        byte[] signatureBytes;
        try { signatureBytes = Base64.decode(signature, Base64.DEFAULT); }
        catch (IllegalArgumentException e) { Log.e("lfl", "Base64 decoding failed."); return false; }

        try {
            Signature sig = Signature.getInstance(SIGNATURE_ALGORITHM);
            sig.initVerify(pubkey);
            sig.update(signedData.getBytes());
            if (!sig.verify(signatureBytes)) { Log.e("lf", "Signature verification failed."); return false; }
            return true;
        }
        catch (java.security.NoSuchAlgorithmException e) { Log.e("lfl", "NoSuchAlgorithmException."); }
        catch (java.security.InvalidKeyException e) { Log.e("lfl", "Invalid key specification."); }
        catch (java.security.SignatureException e) { Log.e("lfl", "Signature exception."); }
        return false;
    }

    public boolean restorePurchases() {
        if (mService == null) return false;
        try {
            Bundle ownedItems = mService.getPurchases(3, packageName, "inapp", null);
            int response = ownedItems.getInt("RESPONSE_CODE");
            Log.i("lfl", "restorePurchases: response=" + response);
            if (response != 0) return false;

            ArrayList<String> products = ownedItems.getStringArrayList("INAPP_PURCHASE_ITEM_LIST");
            ArrayList<String> data     = ownedItems.getStringArrayList("INAPP_PURCHASE_DATA_LIST");
            ArrayList<String> sig      = ownedItems.getStringArrayList("INAPP_DATA_SIGNATURE");
            if (!savePurchasesToInternalStorage(products, data, sig)) return false;
            return loadPurchasesFromInternalStorage();
        }
        catch(Exception e) {
            return false;
        }
    }

    public boolean loadPurchasesFromInternalStorage() {
        try {
            FileInputStream fis = mContext.openFileInput(FILENAME);
            ObjectInputStream ois = new ObjectInputStream(fis);
            if (!((String)ois.readObject()).equals(mContext.getAndroidId())) return false;

            ArrayList<?> products = (ArrayList<?>)ois.readObject();
            ArrayList<?> data     = (ArrayList<?>)ois.readObject();
            ArrayList<?> sig      = (ArrayList<?>)ois.readObject();
            if (products.size() != data.size() || data.size() != sig.size()) return false;

            mProducts.clear();
            for (int i = 0, l = sig.size(); i < l; i++)
                if (verifyPurchase(mPubKey, (String)data.get(i), (String)sig.get(i)))
                    mProducts.put((String)products.get(i), "");
            return true;

         } catch(java.io.IOException ioe) {
            Log.e("lfl", "savePurchasesToInternalStorage: " + ioe.toString());
            return false;
         } catch(ClassNotFoundException e) {
            Log.e("lfl", "savePurchasesToInternalStorage: " + e.toString());
            return false;
        }
    }

    public boolean savePurchasesToInternalStorage(ArrayList<String> products, ArrayList<String> data, ArrayList<String> sig) {
        if (products.size() != data.size() || data.size() != sig.size()) return false;
        try {
            FileOutputStream fos = mContext.openFileOutput(FILENAME, Context.MODE_PRIVATE);
            ObjectOutputStream oos = new ObjectOutputStream(fos);
            // XXX We need Play Store to sign the getPurchases() data with a device identifier appended
            oos.writeObject(mContext.getAndroidId());
            oos.writeObject(products);
            oos.writeObject(data);
            oos.writeObject(sig);
            oos.close();
            fos.close();
            return true;
         } catch(java.io.IOException ioe) {
             Log.e("lfl", "savePurchasesToInternalStorage: " + ioe.toString());
             return false;
         }
    }
}
