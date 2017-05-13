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
import android.util.Log;
import android.util.Pair;
import android.util.TypedValue;
import android.net.Uri;
import android.graphics.Rect;

public class TableScreen extends Screen {
    public ModelItemRecyclerViewAdapter model;
    public RecyclerViewScreenFragment view;
    public native void RunHideCB();

    public TableScreen(final MainActivity activity, String t, ArrayList<ModelItem> m, long lsp) {
        super(Screen.TYPE_TABLE, activity, t, lsp);
        model = new ModelItemRecyclerViewAdapter(activity, m, this);
    }

    public void clear() { view = null; }
    public ScreenFragment get(final MainActivity activity) { return getView(activity); }

    public RecyclerViewScreenFragment getView(final MainActivity activity) {
        if (view == null) view = createView(activity);
        return view;
    }

    public RecyclerViewScreenFragment createView(final MainActivity activity) {
        return new RecyclerViewScreenFragment(activity, this, model, null);
    }

    public void show(final MainActivity activity, final boolean show_or_hide) {
        activity.runOnUiThread(new Runnable() { public void run() {
            ScreenFragment table = get(activity);
            if (show_or_hide) {
                activity.action_bar.show();
                activity.getSupportFragmentManager().beginTransaction().replace(R.id.content_frame, table).commit();
            } else {
                if (activity.disable_title) activity.action_bar.hide();
                activity.getSupportFragmentManager().beginTransaction().remove(table).commit();
            }
        }});
    }

    public void addNavButton(final MainActivity activity, final int halign, final ModelItem row) {
        activity.runOnUiThread
            (new Runnable() { public void run() { getView(activity).data.addNavButton(halign, row); }});
    }
    
    public void delNavButton(final MainActivity activity, final int halign) {
        activity.runOnUiThread
            (new Runnable() { public void run() { getView(activity).data.delNavButton(halign); }});
    }

    public String getKey(final MainActivity activity, final int s, final int r) {
        FutureTask<String> future = new FutureTask<String>
            (new Callable<String>(){ public String call() throws Exception {
                return getView(activity).data.getKey(s, r);
            }});
        try { activity.runOnUiThread(future); return future.get(); }
        catch(Exception e) { return new String(); }
    }

    public Integer getTag(final MainActivity activity, final int s, final int r) {
        FutureTask<Integer> future = new FutureTask<Integer>
            (new Callable<Integer>(){ public Integer call() throws Exception {
                return getView(activity).data.getTag(s, r);
            }});
        try { activity.runOnUiThread(future); return future.get(); }
        catch(Exception e) { return 0; }
    }
    
    public Pair<Long, ArrayList<Integer>> getPicked(final MainActivity activity, final int section, final int row) {
        FutureTask<Pair<Long, ArrayList<Integer>>> future = new FutureTask<Pair<Long, ArrayList<Integer>>>
            (new Callable<Pair<Long, ArrayList<Integer>>>(){
                public Pair<Long, ArrayList<Integer>> call() throws Exception {
                    RecyclerViewScreenFragment table = getView(activity);
                    return table.data.getPicked(section, row);
                }});
        try { activity.runOnUiThread(future); return future.get(); }
        catch(Exception e) { return null; }
    }

    public ArrayList<Pair<String, String>> getSectionText(final MainActivity activity, final int section) {
        FutureTask<ArrayList<Pair<String, String>>> future = new FutureTask<ArrayList<Pair<String, String>>>
            (new Callable<ArrayList<Pair<String, String>>>(){
                public ArrayList<Pair<String, String>> call() throws Exception {
                    RecyclerViewScreenFragment table = getView(activity);
                    return table.data.getSectionText(table.recyclerview, section);
                }});
        try { activity.runOnUiThread(future); return future.get(); }
        catch(Exception e) { return new ArrayList<Pair<String, String>>(); }
    }

    public void beginUpdates(final MainActivity activity) {
        activity.runOnUiThread
            (new Runnable() { public void run() { getView(activity).data.beginUpdates(); }});
    }

    public void endUpdates(final MainActivity activity) {
        activity.runOnUiThread
            (new Runnable() { public void run() { getView(activity).data.endUpdates(); }});
    }

    public void addRow(final MainActivity activity, final int section, final ModelItem row) {
        activity.runOnUiThread
            (new Runnable() { public void run() { getView(activity).data.addRow(section, row); }});
    }

    public void selectRow(final MainActivity activity, final int s, final int r) {
        activity.runOnUiThread
            (new Runnable() { public void run() { getView(activity).data.selectRow(s, r); }});
    }

    public void replaceRow(final MainActivity activity, final int s, final int r, final ModelItem v) {
        activity.runOnUiThread
            (new Runnable() { public void run() { getView(activity).data.replaceRow(s, r, v); }});
    }

    public void replaceSection(final MainActivity activity, final int section, final ModelItem h,
                               final int flag, final ArrayList<ModelItem> v) {
        activity.runOnUiThread
            (new Runnable() { public void run() { getView(activity).data.replaceSection(section, h, flag, v); }});
    }

    public void applyChangeList(final MainActivity activity, final ArrayList<ModelItemChange> changes) {
        activity.runOnUiThread
            (new Runnable() { public void run() { ModelItemChange.applyChangeList(changes, getView(activity).data); }});
    }

    public void setSectionValues(final MainActivity activity, final int section, final ArrayList<String> v) {
        activity.runOnUiThread
            (new Runnable() { public void run() { getView(activity).data.setSectionValues(section, v); }});
    }

    public void setTag(final MainActivity activity, final int s, final int r, final int v) {
        activity.runOnUiThread
            (new Runnable() { public void run() { getView(activity).data.setTag(s, r, v); }});
    }

    public void setKey(final MainActivity activity, final int s, final int r, final String v) {
        activity.runOnUiThread
            (new Runnable() { public void run() { getView(activity).data.setKey(s, r, v); }});
    }

    public void setValue(final MainActivity activity, final int s, final int r, final String v) {
        activity.runOnUiThread
            (new Runnable() { public void run() { getView(activity).data.setValue(s, r, v); }});
    }

    public void setHidden(final MainActivity activity, final int s, final int r, final boolean v) {
        activity.runOnUiThread
            (new Runnable() { public void run() { getView(activity).data.setHidden(s, r, v); }});
    }

    public void setSelected(final MainActivity activity, final int s, final int r, final int v) {
        activity.runOnUiThread
            (new Runnable() { public void run() { getView(activity).data.setSelected(s, r, v); }});
    }

    public void setTitle(final MainActivity activity, final String v) {
        activity.runOnUiThread
            (new Runnable() { public void run() { title = v; }});
    }

    public void setEditable(final MainActivity activity, final int s, final int start_row, final NativeIntIntCB intint_cb) {
        activity.runOnUiThread
            (new Runnable() { public void run() { getView(activity).data.setEditable(s, start_row, intint_cb); }});
    }
}
