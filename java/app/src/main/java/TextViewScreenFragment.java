package com.lucidfusionlabs.app;

import java.util.ArrayList;
import java.util.List;

import android.annotation.SuppressLint;
import android.app.Fragment;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.AdapterView;
import android.widget.AdapterView.OnItemClickListener;
import android.widget.ArrayAdapter;
import android.widget.TextView;
import android.widget.LinearLayout;

public class TextViewScreenFragment extends ScreenFragment implements OnItemClickListener {
    public String data;
    public TextView textview;
    public static String PARENT_SCREEN_ID = "PARENT_SCREEN_ID";

    public static final TextViewScreenFragment newInstance(TextScreen parent) {
        TextViewScreenFragment ret = new TextViewScreenFragment();
        Bundle bundle = new Bundle(1);
        bundle.putInt(PARENT_SCREEN_ID, parent.screenId);
        ret.setArguments(bundle);
        ret.data = parent.text;
        ret.parent_screen = parent;
        return ret;
    }

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        if (MainActivity.screens.next_screen_id == 1) return;
        int parent_screen_id = getArguments().getInt(PARENT_SCREEN_ID);
        if (parent_screen_id < MainActivity.screens.screens.size()) {
            parent_screen = MainActivity.screens.screens.get(parent_screen_id);
            if (!(parent_screen instanceof TextScreen)) data = "";
            else data = ((TextScreen)parent_screen).text;
        }
    }

    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container, Bundle savedInstanceState) {
        LinearLayout layout = (LinearLayout)inflater.inflate(R.layout.textview_main, container, false);
        textview = (TextView)layout.findViewById(R.id.textview_main);
        textview.setText(data);
        textview.setMovementMethod(new android.text.method.ScrollingMovementMethod());
        return layout;
    }

    @Override
    public void onItemClick(AdapterView<?> parent, View view, int position, long id) {
    }
}
