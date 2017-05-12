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
import android.app.AlertDialog;
import android.support.v4.app.Fragment;
import android.support.v4.app.FragmentManager;

public class ScreenFragmentNavigator {
    public int shown_index = -1;
    public ArrayList<Screen> stack = new ArrayList<Screen>();
    public ScreenFragmentNavigator(final MainActivity activity) {}

    public void show(final MainActivity activity, final boolean show_or_hide) {
        final ScreenFragmentNavigator self = this;
        activity.runOnUiThread(new Runnable() { public void run() {
            if (show_or_hide) {
                if (shown_index >= 0) Log.i("lfl", "Show already shown navbar");
                else {
                    activity.screens.navigators.add(self);
                    activity.frame_layout.setVisibility(View.GONE);
                    activity.action_bar.show();
                    shown_index = activity.screens.navigators.size()-1;
                    int stack_size = activity.getSupportFragmentManager().getBackStackEntryCount();
                    if (stack_size != 0) Log.e("lfl", "nested show? stack_size=" + stack_size);
                    for (Screen x : stack) {
                        x.changed = false;
                        String tag = Integer.toString(stack_size++);
                        ScreenFragment frag = x.get(activity);
                        activity.getSupportFragmentManager().beginTransaction()
                            .replace(R.id.content_frame, frag, tag).addToBackStack(tag).commit();
                        activity.setTitle(frag.parent_screen.title);
                    }
                }
            } else {
                if (shown_index < 0) Log.i("lfl", "Hide unshown navbar");
                else {
                    activity.getSupportFragmentManager().popBackStackImmediate(null, FragmentManager.POP_BACK_STACK_INCLUSIVE);
                    int size = activity.screens.navigators.size();
                    if (shown_index != size-1) Log.e("lfl", "hide navigator  shown_index=" + shown_index + " size=" + size);
                    if (shown_index < size) activity.screens.navigators.remove(shown_index);
                    if (activity.disable_title) activity.action_bar.hide();
                    if (activity.getSupportFragmentManager().findFragmentById(R.id.content_frame) != null) {
                        Fragment frag = activity.getSupportFragmentManager().findFragmentById(R.id.content_frame);
                        activity.getSupportFragmentManager().beginTransaction().remove(frag).commit();
                    }
                    if (size == 1) activity.frame_layout.setVisibility(View.VISIBLE);
                    shown_index = -1;
                }
            }
        }});
    }

    public void pushTable   (final MainActivity activity, final TableScreen    x) { pushView(activity, x); }
    public void pushTextView(final MainActivity activity, final TextViewScreen x) { pushView(activity, x); }

    public void pushView(final MainActivity activity, final Screen x) {
        stack.add(x);
        activity.runOnUiThread(new Runnable() { public void run() {
            if (shown_index < 0) return;
            x.changed = false;
            String tag = Integer.toString(activity.getSupportFragmentManager().getBackStackEntryCount());
            ScreenFragment frag = x.get(activity);
            activity.getSupportFragmentManager().beginTransaction()
                .replace(R.id.content_frame, frag, tag).addToBackStack(tag).commit();
            activity.setTitle(frag.parent_screen.title);
        }});
    }

    public void popView(final MainActivity activity, final int n) {
        if (n <= 0) return;
        if (stack.size() < n) Log.e("lfl", "pop " + n + " views with stack size " + stack.size());
        if (stack.size() > 0) stack.subList(Math.max(0, stack.size()-n), stack.size()).clear();
        activity.runOnUiThread(new Runnable() { public void run() {
            if (shown_index < 0) return;
            int stack_size = runBackFragmentHideCB(activity, n);
            if (stack_size <= 0 || n <= 0) return;
            int target = Math.max(0, stack_size - n);
            activity.getSupportFragmentManager().popBackStackImmediate
                ((target >= 0 ? Integer.toString(target) : null), FragmentManager.POP_BACK_STACK_INCLUSIVE);
            showBackFragment(activity, false, true);
        }});
    }

    public void popToRoot(final MainActivity activity) {
        if (stack.size() > 1) stack.subList(1, stack.size()).clear();
        activity.runOnUiThread(new Runnable() { public void run() {
            if (shown_index < 0) return;
            int stack_size = activity.getSupportFragmentManager().getBackStackEntryCount();
            if (stack_size < 2) return;
            runBackFragmentHideCB(activity, stack_size - 1);
            activity.getSupportFragmentManager().popBackStackImmediate("1", FragmentManager.POP_BACK_STACK_INCLUSIVE);
            showBackFragment(activity, false, true);
        }});
    }

    public void popAll(final MainActivity activity) {
        stack.clear();
        activity.runOnUiThread(new Runnable() { public void run() {
            if (shown_index < 0) return;
            int stack_size = activity.getSupportFragmentManager().getBackStackEntryCount();
            runBackFragmentHideCB(activity, stack_size);
            activity.getSupportFragmentManager().popBackStackImmediate(null, FragmentManager.POP_BACK_STACK_INCLUSIVE);
        }});
    }

    public Fragment getBack(final MainActivity activity) {
        FutureTask<Fragment> future = new FutureTask<Fragment>
            (new Callable<Fragment>(){ public Fragment call() throws Exception {
                if (shown_index < 0) return null;
                int stack_size = activity.getSupportFragmentManager().getBackStackEntryCount();
                if (stack_size == 0) return null;
                String tag = Integer.toString(stack_size-1);
                return activity.getSupportFragmentManager().findFragmentByTag(tag);
            }});
        try { activity.runOnUiThread(future); return future.get(); }
        catch(Exception e) { return null; }
    }

    public long getBackTableSelf(final MainActivity activity) {
        Fragment frag = getBack(activity);
        if (frag == null || !(frag instanceof ScreenFragment)) return 0;
        ScreenFragment jfrag = (ScreenFragment)frag;
        return (jfrag.parent_screen instanceof TableScreen) ? jfrag.parent_screen.nativeParent : 0;
    }

    public static void onBackPressed(final MainActivity activity) {
        int back_count = activity.getSupportFragmentManager().getBackStackEntryCount();
        if (back_count > 1) {
            if (activity.screens.navigators.size() >= 0) {
                ScreenFragmentNavigator n = activity.screens.navigators.get(activity.screens.navigators.size()-1);
                if (back_count != n.stack.size()) Log.e("lfl", "back_count=" + back_count + " but stack_size=" + n.stack.size());
                if (n.stack.size() > 0) n.stack.remove(n.stack.size()-1);
            } else Log.e("lfl", "back_count=" + back_count + " but no navigators");

            runBackFragmentHideCB(activity, 1);
            activity.superOnBackPressed();
            showBackFragment(activity, false, true);

        } else if (back_count == 1) {
            runBackFragmentHideCB(activity, 1);
            Fragment frag = activity.getSupportFragmentManager().findFragmentByTag("0");
            if (frag != null && frag instanceof RecyclerViewScreenFragment) {
                ModelItem nav_left = ((RecyclerViewScreenFragment)frag).data.nav_left;
                if (nav_left != null && nav_left.cb != null) {
                    nav_left.cb.run();
                    return;
                }
            }
            activity.moveTaskToBack(true);

        } else {
            activity.superOnBackPressed();
        }
    } 

    public static int runBackFragmentHideCB(final MainActivity activity, int n) {
        int stack_size = activity.getSupportFragmentManager().getBackStackEntryCount();
        if (stack_size == 0) return stack_size;
        for (int i = 0, l = Math.min(n, stack_size); i < l; i++) {
            String tag = Integer.toString(stack_size-1-i);
            Fragment frag = activity.getSupportFragmentManager().findFragmentByTag(tag);
            if (frag == null || !(frag instanceof ScreenFragment)) continue;
            ScreenFragment jfrag = (ScreenFragment)frag;
            if (jfrag.parent_screen.nativeParent == 0) continue;
            if (jfrag.parent_screen instanceof TableScreen) ((TableScreen)jfrag.parent_screen).RunHideCB();
        }
        return stack_size;
    }

    public static void showBackFragment(final MainActivity activity, boolean show_content, boolean show_title) {
        int stack_size = activity.getSupportFragmentManager().getBackStackEntryCount();
        if (stack_size == 0) return;
        String tag = Integer.toString(stack_size-1);
        Fragment frag = activity.getSupportFragmentManager().findFragmentByTag(tag);
        if (show_content) activity.getSupportFragmentManager().beginTransaction().replace(R.id.content_frame, frag, tag).commit();
        if (show_title && frag instanceof ScreenFragment) activity.setTitle(((ScreenFragment)frag).parent_screen.title);
    }
}
