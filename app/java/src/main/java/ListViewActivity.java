package com.lucidfusionlabs.app;

import java.util.ArrayList;
import java.util.List;

import android.app.Activity;
import android.content.Context;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ArrayAdapter;
import android.widget.LinearLayout;
import android.widget.ListView;
import android.widget.TextView;

public class ListViewActivity extends Activity {
    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.listview_main);

        ArrayList<String> list = new ArrayList<String>();
        list.add("Test1");
        list.add("Test2");
        list.add("Test3");
        ArrayAdapter<String> adapter = new ArrayAdapter<String>
            (this, R.layout.listview_cell, R.id.listview_cell_title, list);

        ListView listView = (ListView)this.findViewById(R.id.list);
        listView.setAdapter(adapter);
    }
}
