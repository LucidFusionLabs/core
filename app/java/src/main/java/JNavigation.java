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
import android.app.Fragment;
import android.app.FragmentManager;

public class JNavigation extends JWidget {
    public JNavigation(final MainActivity activity) {
        super(JWidget.TYPE_NAVIGATION, activity, "", 0);
    }

    public void clear() {}

    public void show(final MainActivity activity, final boolean show_or_hide) {
        final JNavigation self = this;
        activity.runOnUiThread(new Runnable() { public void run() {
            if (show_or_hide) {
                activity.jwidgets.navigations.add(self);
                activity.action_bar.show();
                activity.showBackFragment(true, true);
            } else {
                int size = activity.jwidgets.navigations.size();
                assert size != 0 && activity.jwidgets.navigations.get(size-1) == self;
                activity.jwidgets.navigations.remove(size-1);

                if (activity.disable_title) activity.action_bar.hide();
                if (activity.getFragmentManager().findFragmentById(R.id.content_frame) != null) {
                    Fragment frag = activity.getFragmentManager().findFragmentById(R.id.content_frame);
                    activity.getFragmentManager().beginTransaction().remove(frag).commit();
                }
            }
        }});
    }

    public void pushTable(final MainActivity activity, final JTable x) {
        activity.runOnUiThread(new Runnable() { public void run() {
            x.changed = false;
            String tag = Integer.toString(activity.getFragmentManager().getBackStackEntryCount());
            JListViewFragment frag = x.get(activity);
            activity.getFragmentManager().beginTransaction()
                .replace(R.id.content_frame, frag, tag).addToBackStack(tag).commit();
            activity.setTitle(frag.parent_widget.title);
        }});
    }
    
    public void pushTextView(final MainActivity activity, final JTextView x) {
        activity.runOnUiThread(new Runnable() { public void run() {
            x.changed = false;
            String tag = Integer.toString(activity.getFragmentManager().getBackStackEntryCount());
            JTextViewFragment frag = x.get(activity);
            activity.getFragmentManager().beginTransaction()
                .replace(R.id.content_frame, frag, tag).addToBackStack(tag).commit();
            activity.setTitle(frag.parent_widget.title);
        }});
    }

    public void popView(final MainActivity activity, final int n) {
        activity.runOnUiThread(new Runnable() { public void run() {
            int stack_size = activity.getFragmentManager().getBackStackEntryCount();
            if (stack_size <= 0 || n <= 0) return;
            int target = Math.max(0, activity.getFragmentManager().getBackStackEntryCount() - n);
            activity.getFragmentManager().popBackStackImmediate
                ((target >= 0 ? Integer.toString(target) : null), FragmentManager.POP_BACK_STACK_INCLUSIVE);
            activity.showBackFragment(false, true);
        }});
    }

    public void popToRoot(final MainActivity activity) {
        activity.runOnUiThread(new Runnable() { public void run() {
            if (activity.getFragmentManager().getBackStackEntryCount() < 2) return;
            activity.getFragmentManager().popBackStackImmediate("1", FragmentManager.POP_BACK_STACK_INCLUSIVE);
            activity.showBackFragment(false, true);
        }});
    }

    public void popAll(final MainActivity activity) {
        activity.runOnUiThread(new Runnable() { public void run() {
            activity.getFragmentManager().popBackStackImmediate(null, FragmentManager.POP_BACK_STACK_INCLUSIVE);
        }});
    }

    public Fragment getBack(final MainActivity activity) {
        FutureTask<Fragment> future = new FutureTask<Fragment>
            (new Callable<Fragment>(){ public Fragment call() throws Exception {
                int stack_size = activity.getFragmentManager().getBackStackEntryCount();
                if (stack_size == 0) return null;
                String tag = Integer.toString(stack_size-1);
                return activity.getFragmentManager().findFragmentByTag(tag);
            }});
        try { activity.runOnUiThread(future); return future.get(); }
        catch(Exception e) { return null; }
    }

    public long getBackTableSelf(final MainActivity activity) {
        Fragment frag = getBack(activity);
        if (frag == null || !(frag instanceof JFragment)) return 0;
        JFragment jfrag = (JListViewFragment)frag;
        return (jfrag.parent_widget instanceof JTable) ? jfrag.parent_widget.lfl_self : 0;
    }
}
