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
import org.json.JSONObject;
import org.json.JSONException;
import com.android.vending.billing.IInAppBillingService;
import com.lucidfusionlabs.core.NativeCallback;
import com.lucidfusionlabs.core.NativeIntCB;
import com.lucidfusionlabs.core.LifecycleActivity;
import com.lucidfusionlabs.core.ActivityLifecycleListenerList;

public class PurchaseManager extends com.lucidfusionlabs.core.ActivityLifecycleListener {
    public static final int PURCHASE_REQUEST_CODE = 1078;
    public static final String FILENAME = "purchase.ser";
    public static final String KEY_FACTORY_ALGORITHM = "RSA";
    public static final String SIGNATURE_ALGORITHM = "SHA1withRSA";

    final String packageName;
    final PublicKey mPubKey;
    LifecycleActivity mContext;
    IInAppBillingService mService;
    ServiceConnection mServiceConn;
    HashMap<String, String> mProducts = new HashMap<String, String>();
    HashMap<String, NativeIntCB> mOutstandingPurchase = new HashMap<String, NativeIntCB>();

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
        if (mService == null || mServiceConn == null) return;
        mContext.unbindService(mServiceConn);
    }

    @Override public void onActivityResult(LifecycleActivity activity, int requestCode, int resultCode, Intent data) {
        if (requestCode != PURCHASE_REQUEST_CODE || data == null) return;
        int responseCode = data.getIntExtra("RESPONSE_CODE", 0);
        String purchaseData = data.getStringExtra("INAPP_PURCHASE_DATA");
        String dataSignature = data.getStringExtra("INAPP_DATA_SIGNATURE"), product;
        try {
            JSONObject jo = new JSONObject(purchaseData);
            product = jo.getString("productId");
        } catch (Exception e) { Log.e("lfl", "onPurchaseResult: " + e.toString()); return; }
        NativeIntCB result_cb = mOutstandingPurchase.get(product);
        mOutstandingPurchase.remove(product);
        if (result_cb == null) { Log.e("lfl", "onPurchaseResult missing product=" + product); return; }
        int success = resultCode == android.app.Activity.RESULT_OK ? 1 : 0;
        if (success != 0) restorePurchases();
        result_cb.run(success);
    }

    public boolean canPurchase() { return mService != null; }

    public boolean queryPurchase(final ArrayList<String> products,
                                 final NativeCallback done_cb, final NativeProductCB product_cb) {
        if (mService == null) return false;
        final Bundle querySkus = new Bundle();
        querySkus.putStringArrayList("ITEM_ID_LIST", products);
        android.os.AsyncTask.execute(new Runnable() { @Override public void run() {
            try {
                Bundle skuDetails = mService.getSkuDetails(3, packageName, "inapp", querySkus);
                int response = skuDetails.getInt("RESPONSE_CODE");
                if (response == 0) {
                     ArrayList<String> responseList = skuDetails.getStringArrayList("DETAILS_LIST");
                     for (String thisResponse : responseList) {
                         org.json.JSONObject object = new org.json.JSONObject(thisResponse);
                         product_cb.run(object.getString("productId"), "", "", object.getString("price"));
                     }
                 }
            } catch (Exception e) { Log.e("lfl", "queryPurchase: " + e.toString()); }
            done_cb.run();
        }});
        return true;
    }

    public boolean makePurchase(final String product, final NativeIntCB result_cb) {
        if (result_cb == null || mOutstandingPurchase.get(product) != null) return false;
        mOutstandingPurchase.put(product, result_cb);
        try {
            Bundle buyIntentBundle = mService.getBuyIntent(3, packageName, product, "inapp", "");
            android.app.PendingIntent pendingIntent = buyIntentBundle.getParcelable("BUY_INTENT");
            mContext.startIntentSenderForResult(pendingIntent.getIntentSender(),
                                                PURCHASE_REQUEST_CODE, new Intent(),
                                                Integer.valueOf(0), Integer.valueOf(0), Integer.valueOf(0));
            return true;
        }
        catch(Exception e) {
            Log.e("lfl", "makePurchase: " + e.toString());
            mOutstandingPurchase.remove(product);
            return false;
        }
    }

    public boolean havePurchase(final String product) {
        return mProducts.get(product) != null;
    }

    public boolean restorePurchases() {
        if (mService == null) { Log.e("lfl", "restorePurchases: no service"); return false; }
        try {
            Bundle ownedItems = mService.getPurchases(3, packageName, "inapp", null);
            int response = ownedItems.getInt("RESPONSE_CODE");
            Log.i("lfl", "restorePurchases: response=" + response);
            if (response != 0) return false;

            ArrayList<String> products = ownedItems.getStringArrayList("INAPP_PURCHASE_ITEM_LIST");
            ArrayList<String> data     = ownedItems.getStringArrayList("INAPP_PURCHASE_DATA_LIST");
            ArrayList<String> sig      = ownedItems.getStringArrayList("INAPP_DATA_SIGNATURE_LIST");
            return savePurchasesToInternalStorage(products, data, sig);
        }
        catch(Exception e) {
            Log.e("lfl", "restorePurchases: " + e.toString());
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

         } catch(Exception e) {
            Log.e("lfl", "loadPurchasesFromInternalStorage: " + e.toString());
            return false;
        }
    }

    public boolean savePurchasesToInternalStorage(ArrayList<String> products, ArrayList<String> data, ArrayList<String> sig) {
        if (products.size() != data.size() || data.size() != sig.size()) { Log.e("lfl", "invalid savePurchasesToInternalStorage"); return false; }
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
         } catch(Exception e) {
             Log.e("lfl", "savePurchasesToInternalStorage: " + e.toString());
             return false;
         }
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
            if (!sig.verify(signatureBytes)) { Log.e("lfl", "Signature verification failed."); return false; }
            return true;
        }
        catch (java.security.NoSuchAlgorithmException e) { Log.e("lfl", "NoSuchAlgorithmException."); }
        catch (java.security.InvalidKeyException e) { Log.e("lfl", "Invalid key specification."); }
        catch (java.security.SignatureException e) { Log.e("lfl", "Signature exception."); }
        return false;
    }
}
