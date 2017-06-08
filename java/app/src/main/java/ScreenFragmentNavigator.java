package com.lucidfusionlabs.app;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.HashMap;
import java.util.concurrent.Callable;

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
import android.support.v7.app.AppCompatActivity;
import android.support.v7.app.ActionBar;
import com.lucidfusionlabs.core.ModelItem;

public class ScreenFragmentNavigator {
    public int shown_index = -1;
    public ArrayList<Screen> stack = new ArrayList<Screen>();

    public long getBackTableNativeParent() {
        synchronized(stack) {
            if (stack.size() == 0) return 0;
            Screen back = stack.get(stack.size()-1);
            return (back.screenType == Screen.TYPE_TABLE) ? back.nativeParent : 0;
        }
    }

    public void clear() { shown_index = -1; }

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
                    synchronized(stack) {
                        for (Screen x : stack) {
                            x.changed = false;
                            String tag = Integer.toString(stack_size++);
                            ScreenFragment frag = x.createFragment();
                            activity.getSupportFragmentManager().beginTransaction()
                                .replace(R.id.content_frame, frag, tag).addToBackStack(tag).commit();
                            activity.setTitle(frag.parent_screen.title);
                        }
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

    public void pushTable   (final AppCompatActivity activity, final TableScreen x) { pushView(activity, x); }
    public void pushTextView(final AppCompatActivity activity, final TextScreen  x) { pushView(activity, x); }

    public void pushView(final AppCompatActivity activity, final Screen x) {
        synchronized(stack) { stack.add(x); }
        activity.runOnUiThread(new Runnable() { public void run() {
            if (shown_index < 0) return;
            x.changed = false;
            String tag = Integer.toString(activity.getSupportFragmentManager().getBackStackEntryCount());
            ScreenFragment frag = x.createFragment();
            activity.getSupportFragmentManager().beginTransaction()
                .replace(R.id.content_frame, frag, tag).addToBackStack(tag).commit();
            activity.setTitle(frag.parent_screen.title);
        }});
    }

    public void popView(final AppCompatActivity activity, final int n) {
        if (n <= 0) return;
        synchronized(stack) {
            if (stack.size() < n) Log.e("lfl", "pop " + n + " views with stack size " + stack.size());
            if (stack.size() > 0) stack.subList(Math.max(0, stack.size()-n), stack.size()).clear();
        }
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

    public void popToRoot(final AppCompatActivity activity) {
        synchronized(stack) { if (stack.size() > 1) stack.subList(1, stack.size()).clear(); }
        activity.runOnUiThread(new Runnable() { public void run() {
            if (shown_index < 0) return;
            int stack_size = activity.getSupportFragmentManager().getBackStackEntryCount();
            if (stack_size < 2) return;
            runBackFragmentHideCB(activity, stack_size - 1);
            activity.getSupportFragmentManager().popBackStackImmediate("1", FragmentManager.POP_BACK_STACK_INCLUSIVE);
            showBackFragment(activity, false, true);
        }});
    }

    public void popAll(final AppCompatActivity activity) {
        synchronized(stack) { stack.clear(); }
        activity.runOnUiThread(new Runnable() { public void run() {
            if (shown_index < 0) return;
            int stack_size = activity.getSupportFragmentManager().getBackStackEntryCount();
            runBackFragmentHideCB(activity, stack_size);
            activity.getSupportFragmentManager().popBackStackImmediate(null, FragmentManager.POP_BACK_STACK_INCLUSIVE);
        }});
    }

    public static boolean haveFragmentNavLeft(final AppCompatActivity activity, String tag) {
        Fragment frag = activity.getSupportFragmentManager().findFragmentByTag(tag);
        if (frag != null && frag instanceof RecyclerViewScreenFragment) {
            ModelItem nav_left = ((RecyclerViewScreenFragment)frag).adapter.nav_left;
            return nav_left != null && nav_left.cb != null;
        } else return false;
    }

    public static void clearFragmentBackstack(final AppCompatActivity activity) {
        if (activity.getSupportFragmentManager().getBackStackEntryCount() > 0)
            activity.getSupportFragmentManager().popBackStackImmediate(null, FragmentManager.POP_BACK_STACK_INCLUSIVE);
    }

    public static void onBackPressed(final MainActivity activity) {
        int back_count = activity.getSupportFragmentManager().getBackStackEntryCount();
        if (back_count > 1) {
            if (activity.screens.navigators.size() >= 0) {
                ScreenFragmentNavigator n = activity.screens.navigators.get(activity.screens.navigators.size()-1);
                synchronized(n.stack) {
                    if (back_count != n.stack.size()) Log.e("lfl", "back_count=" + back_count + " but stack_size=" + n.stack.size());
                    if (n.stack.size() > 0) n.stack.remove(n.stack.size()-1);
                }
            } else Log.e("lfl", "back_count=" + back_count + " but no navigators");

            runBackFragmentHideCB(activity, 1);
            activity.superOnBackPressed();
            showBackFragment(activity, false, true);

        } else if (back_count == 1) {
            runBackFragmentHideCB(activity, 1);
            Fragment frag = activity.getSupportFragmentManager().findFragmentByTag("0");
            if (frag != null && frag instanceof RecyclerViewScreenFragment) {
                ModelItem nav_left = ((RecyclerViewScreenFragment)frag).adapter.nav_left;
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

    public static int runBackFragmentHideCB(final AppCompatActivity activity, int n) {
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

    public static void showBackFragment(final AppCompatActivity activity, boolean show_content, boolean show_title) {
        int stack_size = activity.getSupportFragmentManager().getBackStackEntryCount();
        if (stack_size == 0) return;
        String tag = Integer.toString(stack_size-1);
        Fragment frag = activity.getSupportFragmentManager().findFragmentByTag(tag);
        if (frag == null) { Log.e("lfl", "showBackFragment tag=" + tag + ": null"); return; }
        if (show_content) activity.getSupportFragmentManager().beginTransaction().replace(R.id.content_frame, frag, tag).commit();
        if (show_title && frag instanceof ScreenFragment) activity.setTitle(((ScreenFragment)frag).parent_screen.title);
    }

    public static void updateHomeButton(final AppCompatActivity activity) {
        ActionBar action_bar = activity.getSupportActionBar();
        int stack_size = activity.getSupportFragmentManager().getBackStackEntryCount() ;
        if (stack_size > 1 || (stack_size == 1 && ScreenFragmentNavigator.haveFragmentNavLeft(activity, "0"))) {
            action_bar.setHomeButtonEnabled(true);
            action_bar.setDisplayHomeAsUpEnabled(true);
        } else {
            action_bar.setDisplayHomeAsUpEnabled(false);
            action_bar.setHomeButtonEnabled(false);
        }
    }
}
