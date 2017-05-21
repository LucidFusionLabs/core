package com.lucidfusionlabs.core;

import java.util.ArrayList;

public interface ModelItemList {
    public ArrayList<ModelItem> getData();
    public int getSectionBeginRowId(int section);
    public int getSectionEndRowId(int section);
    public int getSectionSize(int section);
    public int getCollapsedRowId(int section, int row);
}
