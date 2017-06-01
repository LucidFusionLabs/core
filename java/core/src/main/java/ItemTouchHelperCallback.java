package com.lucidfusionlabs.core;

import android.support.v7.widget.RecyclerView;
import android.support.v7.widget.helper.ItemTouchHelper;

public class ItemTouchHelperCallback extends ItemTouchHelper.Callback {
    public interface Listener {
        public void onItemTouchHelperDismiss(int id);
        public boolean onItemTouchHelperMove(int from, int to);
    }

    public final Listener mListener;

    public ItemTouchHelperCallback(Listener listener) {
        mListener = listener;
    }

    @Override public boolean isLongPressDragEnabled() { return false; }
    @Override public boolean isItemViewSwipeEnabled() { return false; }

    @Override public int getMovementFlags(RecyclerView recyclerView, RecyclerView.ViewHolder viewHolder) {
        int dragFlags = ItemTouchHelper.UP | ItemTouchHelper.DOWN;
        int swipeFlags = ItemTouchHelper.START | ItemTouchHelper.END;
        return makeMovementFlags(dragFlags, swipeFlags);
    }

    @Override public boolean onMove(RecyclerView recyclerView, RecyclerView.ViewHolder viewHolder,  RecyclerView.ViewHolder target) {
        return mListener.onItemTouchHelperMove(viewHolder.getAdapterPosition(), target.getAdapterPosition());
    }

    @Override public void onSwiped(RecyclerView.ViewHolder viewHolder, int direction) {
        mListener.onItemTouchHelperDismiss(viewHolder.getAdapterPosition());
    }
}
