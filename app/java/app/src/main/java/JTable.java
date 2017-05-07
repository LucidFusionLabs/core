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
import android.app.ActionBar;
import android.app.AlertDialog;

public class JTable extends JWidget {
    public JRecyclerViewAdapter model;
    public JRecyclerViewFragment view;
    public native void RunHideCB();

    public JTable(final MainActivity activity, String t, ArrayList<JModelItem> m, long lsp) {
        super(JWidget.TYPE_TABLE, activity, t, lsp);
        model = new JRecyclerViewAdapter(activity, m, this);
    }

    public void clear() { view = null; }

    public JRecyclerViewFragment get(final MainActivity activity) {
        if (view == null) {
            view = new JRecyclerViewFragment(activity, this, model, null);
        }
        return view;
    }

    public void show(final MainActivity activity, final boolean show_or_hide) {
        activity.runOnUiThread(new Runnable() { public void run() {
            JRecyclerViewFragment table = get(activity);
            if (show_or_hide) {
                activity.action_bar.show();
                activity.getFragmentManager().beginTransaction().replace(R.id.content_frame, table).commit();
            } else {
                if (activity.disable_title) activity.action_bar.hide();
                activity.getFragmentManager().beginTransaction().remove(table).commit();
            }
        }});
    }

    public void addNavButton(final MainActivity activity, final int halign, final JModelItem row) {
        activity.runOnUiThread
            (new Runnable() { public void run() { get(activity).data.addNavButton(halign, row); }});
    }
    
    public void delNavButton(final MainActivity activity, final int halign) {
        activity.runOnUiThread
            (new Runnable() { public void run() { get(activity).data.delNavButton(halign); }});
    }

    public String getKey(final MainActivity activity, final int s, final int r) {
        FutureTask<String> future = new FutureTask<String>
            (new Callable<String>(){ public String call() throws Exception {
                return get(activity).data.getKey(s, r);
            }});
        try { activity.runOnUiThread(future); return future.get(); }
        catch(Exception e) { return new String(); }
    }

    public Integer getTag(final MainActivity activity, final int s, final int r) {
        FutureTask<Integer> future = new FutureTask<Integer>
            (new Callable<Integer>(){ public Integer call() throws Exception {
                return get(activity).data.getTag(s, r);
            }});
        try { activity.runOnUiThread(future); return future.get(); }
        catch(Exception e) { return 0; }
    }
    
    public Pair<Long, ArrayList<Integer>> getPicked(final MainActivity activity, final int section, final int row) {
        FutureTask<Pair<Long, ArrayList<Integer>>> future = new FutureTask<Pair<Long, ArrayList<Integer>>>
            (new Callable<Pair<Long, ArrayList<Integer>>>(){
                public Pair<Long, ArrayList<Integer>> call() throws Exception {
                    JRecyclerViewFragment table = get(activity);
                    return table.data.getPicked(section, row);
                }});
        try { activity.runOnUiThread(future); return future.get(); }
        catch(Exception e) { return null; }
    }

    public ArrayList<Pair<String, String>> getSectionText(final MainActivity activity, final int section) {
        FutureTask<ArrayList<Pair<String, String>>> future = new FutureTask<ArrayList<Pair<String, String>>>
            (new Callable<ArrayList<Pair<String, String>>>(){
                public ArrayList<Pair<String, String>> call() throws Exception {
                    JRecyclerViewFragment table = get(activity);
                    return table.data.getSectionText(table.recyclerview, section);
                }});
        try { activity.runOnUiThread(future); return future.get(); }
        catch(Exception e) { return new ArrayList<Pair<String, String>>(); }
    }

    public void beginUpdates(final MainActivity activity) {
        activity.runOnUiThread
            (new Runnable() { public void run() { get(activity).data.beginUpdates(); }});
    }

    public void endUpdates(final MainActivity activity) {
        activity.runOnUiThread
            (new Runnable() { public void run() { get(activity).data.endUpdates(); }});
    }

    public void addRow(final MainActivity activity, final int section, final JModelItem row) {
        activity.runOnUiThread
            (new Runnable() { public void run() { get(activity).data.addRow(section, row); }});
    }

    public void selectRow(final MainActivity activity, final int s, final int r) {
        activity.runOnUiThread
            (new Runnable() { public void run() { get(activity).data.selectRow(s, r); }});
    }

    public void replaceRow(final MainActivity activity, final int s, final int r, final JModelItem v) {
        activity.runOnUiThread
            (new Runnable() { public void run() { get(activity).data.replaceRow(s, r, v); }});
    }

    public void replaceSection(final MainActivity activity, final int section, final JModelItem h,
                               final int flag, final ArrayList<JModelItem> v) {
        activity.runOnUiThread
            (new Runnable() { public void run() { get(activity).data.replaceSection(section, h, flag, v); }});
    }

    public void applyChangeList(final MainActivity activity, final ArrayList<JModelItemChange> changes) {
        activity.runOnUiThread
            (new Runnable() { public void run() { JModelItemChange.applyChangeList(changes, get(activity).data); }});
    }

    public void setSectionValues(final MainActivity activity, final int section, final ArrayList<String> v) {
        activity.runOnUiThread
            (new Runnable() { public void run() { get(activity).data.setSectionValues(section, v); }});
    }

    public void setTag(final MainActivity activity, final int s, final int r, final int v) {
        activity.runOnUiThread
            (new Runnable() { public void run() { get(activity).data.setTag(s, r, v); }});
    }

    public void setKey(final MainActivity activity, final int s, final int r, final String v) {
        activity.runOnUiThread
            (new Runnable() { public void run() { get(activity).data.setKey(s, r, v); }});
    }

    public void setValue(final MainActivity activity, final int s, final int r, final String v) {
        activity.runOnUiThread
            (new Runnable() { public void run() { get(activity).data.setValue(s, r, v); }});
    }

    public void setHidden(final MainActivity activity, final int s, final int r, final boolean v) {
        activity.runOnUiThread
            (new Runnable() { public void run() { get(activity).data.setHidden(s, r, v); }});
    }

    public void setTitle(final MainActivity activity, final String v) {
        activity.runOnUiThread
            (new Runnable() { public void run() { title = v; }});
    }

    public void setEditable(final MainActivity activity, final int s, final int start_row, final LIntIntCB intint_cb) {
        activity.runOnUiThread
            (new Runnable() { public void run() { get(activity).data.setEditable(s, start_row, intint_cb); }});
    }
}
