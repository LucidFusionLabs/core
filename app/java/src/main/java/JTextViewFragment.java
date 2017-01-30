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
import android.util.Log;

public class JTextViewFragment extends Fragment implements OnItemClickListener {
    public MainActivity main_activity;
    public JWidget parent_widget;
    public String data;
    public TextView textview;
    public long lfl_self;

    JTextViewFragment(final MainActivity activity, final JWidget p, final String d, final long lsp) {
        main_activity = activity;
        parent_widget = p;
        data = d;
        lfl_self = lsp;
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
