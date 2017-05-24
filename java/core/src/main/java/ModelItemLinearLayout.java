package com.lucidfusionlabs.core;

import android.view.View;
import android.view.ViewGroup;
import android.view.ViewParent;
import android.content.Context;
import android.widget.LinearLayout;
import android.util.AttributeSet;

public class ModelItemLinearLayout extends LinearLayout {
    public ModelItem item, section;
    public ModelItemLinearLayout(Context context) { super(context); }
    public ModelItemLinearLayout(Context context, AttributeSet attrs) { super(context, attrs); }
    public ModelItemLinearLayout(Context context, AttributeSet attrs, int defStyleAttr) { super(context, attrs, defStyleAttr); }
    public ModelItemLinearLayout(Context context, AttributeSet attrs, int defStyleAttr, int defStyleRes) { super(context, attrs, defStyleAttr, defStyleRes); }

    @Override protected void onMeasure(int widthMeasureSpec, int heightMeasureSpec) {
        super.onMeasure(widthMeasureSpec, heightMeasureSpec);
        if (section != null && !item.equals(section) &&
            (section.flags & ModelItem.TABLE_SECTION_FLAG_DOUBLE_ROW_HEIGHT) != 0) {
            ViewGroup.LayoutParams lp = getLayoutParams();
            lp.height = getMeasuredHeight()*2;
            setLayoutParams(lp);
            super.onMeasure(widthMeasureSpec, heightMeasureSpec);
        }
    }
}
