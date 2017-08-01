package com.lucidfusionlabs.app;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.HashMap;
import java.util.concurrent.Callable;
import java.util.concurrent.FutureTask;

import android.os.*;
import android.view.*;
import android.widget.FrameLayout;
import android.widget.FrameLayout.LayoutParams;
import android.widget.LinearLayout;
import android.widget.TableLayout;
import android.widget.RelativeLayout;
import android.widget.Button;
import android.widget.ImageButton;
import android.widget.EditText;
import android.widget.TextView;
import android.media.*;
import android.content.*;
import android.content.res.Configuration;
import android.content.res.Resources;
import android.content.pm.ActivityInfo;
import android.hardware.*;
import android.util.Pair;
import android.util.TypedValue;
import android.net.Uri;
import android.graphics.Rect;
import android.support.v4.app.Fragment;
import android.support.v7.app.AppCompatActivity;
import com.lucidfusionlabs.core.ModelItem;
import com.lucidfusionlabs.core.ModelItemChange;
import com.lucidfusionlabs.core.NativeIntIntCB;

public class TableScreen extends Screen {
    public ModelItemRecyclerViewAdapter model;
    public native void RunHideCB();

    public TableScreen(final MainActivity activity, String t, ArrayList<ModelItem> m, long lsp) {
        super(Screen.TYPE_TABLE, t, lsp);
        model = new ModelItemRecyclerViewAdapter(this, m);
        model.display_density = activity.display_density;
    }

    public void clearToolbarView()     { if (model.toolbar     != null) model.toolbar    .clearView(); }
    public void clearAdvertisingView() { if (model.advertising != null) model.advertising.clearView(); }

    @Override public void clear() {
        clearToolbarView();
        clearAdvertisingView();
    }

    @Override public ScreenFragment createFragment() {
        return RecyclerViewScreenFragment.newInstance(this);
    }

    @Override public void show(final MainActivity activity, final boolean show_or_hide) {
        activity.runOnUiThread(new Runnable() { public void run() {
            if (show_or_hide) {
                activity.action_bar.show();
                activity.getSupportFragmentManager().beginTransaction().replace(R.id.content_frame, createFragment()).commit();
            } else {
                if (activity.disable_title) activity.action_bar.hide();
                Fragment frag = activity.getSupportFragmentManager().findFragmentById(R.id.content_frame);
                activity.getSupportFragmentManager().beginTransaction().remove(frag).commit();
            }
        }});
    }

    public void addNavButton(final AppCompatActivity activity, final int halign, final ModelItem row) {
        activity.runOnUiThread
            (new Runnable() { public void run() { model.addNavButton(halign, row); }});
    }
    
    public void delNavButton(final AppCompatActivity activity, final int halign) {
        activity.runOnUiThread
            (new Runnable() { public void run() { model.delNavButton(halign); }});
    }

    public String getKey(final AppCompatActivity activity, final int s, final int r) {
        FutureTask<String> future = new FutureTask<String>
            (new Callable<String>(){ public String call() throws Exception {
                return model.getKey(s, r);
            }});
        try {
            activity.runOnUiThread(future);
            return future.get();
        } catch(Exception e) {
            NativeAPI.ERROR("getKey " + e.toString());
            return new String();
        }
    }

    public String getVal(final AppCompatActivity activity, final int s, final int r) {
        FutureTask<String> future = new FutureTask<String>
            (new Callable<String>(){ public String call() throws Exception {
                return model.getVal(s, r);
            }});
        try {
            activity.runOnUiThread(future);
            return future.get();
        } catch(Exception e) {
            NativeAPI.ERROR("getVal " + e.toString());
            return new String();
        }
    }

    public Integer getTag(final AppCompatActivity activity, final int s, final int r) {
        FutureTask<Integer> future = new FutureTask<Integer>
            (new Callable<Integer>(){ public Integer call() throws Exception {
                return model.getTag(s, r);
            }});
        try {
            activity.runOnUiThread(future);
            return future.get();
        } catch(Exception e) {
            NativeAPI.ERROR("getTag " + e.toString());
            return 0;
        }
    }
    
    public Pair<Long, ArrayList<Integer>> getPicked(final AppCompatActivity activity, final int section, final int row) {
        FutureTask<Pair<Long, ArrayList<Integer>>> future = new FutureTask<Pair<Long, ArrayList<Integer>>>
            (new Callable<Pair<Long, ArrayList<Integer>>>(){
                public Pair<Long, ArrayList<Integer>> call() throws Exception {
                    return model.getPicked(section, row);
                }});
        try {
            activity.runOnUiThread(future);
            return future.get();
        } catch(Exception e) {
            NativeAPI.ERROR("getPicked " + e.toString());
            return null;
        }
    }

    public ArrayList<Pair<String, String>> getSectionText(final AppCompatActivity activity, final int section) {
        FutureTask<ArrayList<Pair<String, String>>> future = new FutureTask<ArrayList<Pair<String, String>>>
            (new Callable<ArrayList<Pair<String, String>>>(){
                public ArrayList<Pair<String, String>> call() throws Exception {
                    return model.getSectionText(model.recyclerview, section);
                }});
        try {
            activity.runOnUiThread(future);
            return future.get();
        } catch(Exception e) {
            NativeAPI.ERROR("getSectionText " + e.toString());
            return new ArrayList<Pair<String, String>>();
        }
    }

    public void beginUpdates(final AppCompatActivity activity) {
        activity.runOnUiThread
            (new Runnable() { public void run() { model.beginUpdates(); }});
    }

    public void endUpdates(final AppCompatActivity activity) {
        activity.runOnUiThread
            (new Runnable() { public void run() { model.endUpdates(); }});
    }

    public void setHeader(final AppCompatActivity activity, final int section, final ModelItem header) {
        activity.runOnUiThread
            (new Runnable() { public void run() { model.setHeader(section, header); }});
    }

    public void addRow(final AppCompatActivity activity, final int section, final ModelItem row) {
        activity.runOnUiThread
            (new Runnable() { public void run() { model.addRow(section, row); }});
    }

    public void selectRow(final AppCompatActivity activity, final int s, final int r) {
        activity.runOnUiThread
            (new Runnable() { public void run() { model.selectRow(s, r); }});
    }

    public void replaceRow(final AppCompatActivity activity, final int s, final int r, final ModelItem v) {
        activity.runOnUiThread
            (new Runnable() { public void run() { model.replaceRow(s, r, v); }});
    }

    public void replaceSection(final AppCompatActivity activity, final int section, final ModelItem h,
                               final int flag, final ArrayList<ModelItem> v) {
        activity.runOnUiThread
            (new Runnable() { public void run() { model.replaceSection(section, h, flag, v); }});
    }

    public void applyChangeList(final AppCompatActivity activity, final ArrayList<ModelItemChange> changes) {
        activity.runOnUiThread
            (new Runnable() { public void run() { model.applyChangeList(changes); }});
    }

    public void setSectionValues(final AppCompatActivity activity, final int section, final ArrayList<String> v) {
        activity.runOnUiThread
            (new Runnable() { public void run() { model.setSectionValues(section, v); }});
    }

    public void setSectionColors(final AppCompatActivity activity, final int section, final ArrayList<Integer> v) {
        activity.runOnUiThread
            (new Runnable() { public void run() { model.setSectionColors(section, v); }});
    }

    public void setTag(final AppCompatActivity activity, final int s, final int r, final int v) {
        activity.runOnUiThread
            (new Runnable() { public void run() { model.setTag(s, r, v); }});
    }

    public void setKey(final AppCompatActivity activity, final int s, final int r, final String v) {
        activity.runOnUiThread
            (new Runnable() { public void run() { model.setKey(s, r, v); }});
    }

    public void setValue(final AppCompatActivity activity, final int s, final int r, final String v) {
        activity.runOnUiThread
            (new Runnable() { public void run() { model.setValue(s, r, v); }});
    }

    public void setColor(final AppCompatActivity activity, final int s, final int r, final int v) {
        activity.runOnUiThread(new Runnable() { public void run() {
            model.setColor(s, r, v);
        }});
    }

    public void setHidden(final AppCompatActivity activity, final int s, final int r, final int v) {
        activity.runOnUiThread(new Runnable() { public void run() {
            if (model.setHidden(s, r, v)) ScreenFragmentNavigator.updateHomeButton(activity);
        }});
    }

    public void setSelected(final AppCompatActivity activity, final int s, final int r, final int v) {
        activity.runOnUiThread
            (new Runnable() { public void run() { model.setSelected(s, r, v); }});
    }

    public void setTitle(final AppCompatActivity activity, final String v) {
        activity.runOnUiThread
            (new Runnable() { public void run() { title = v; }});
    }

    public void setEditable(final AppCompatActivity activity, final int s, final int start_row, final NativeIntIntCB intint_cb) {
        activity.runOnUiThread
            (new Runnable() { public void run() { model.setEditable(s, start_row, intint_cb); }});
    }

    public void setToolbar(final AppCompatActivity activity, final com.lucidfusionlabs.core.ViewOwner x) {
        activity.runOnUiThread(new Runnable() { public void run() {
            clearToolbarView();
            model.toolbar = x;
        }});
    }

    public void setAdvertising(final AppCompatActivity activity, final com.lucidfusionlabs.core.ViewOwner x) {
        activity.runOnUiThread(new Runnable() { public void run() {
            clearAdvertisingView();
            model.advertising = x;
        }});
    }
}
