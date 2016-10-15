/*
 * $Id: assets.cpp 1334 2014-11-28 09:14:21Z justin $
 * Copyright (C) 2009 Lucid Fusion Labs

 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.

 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "core/app/flow.h"
#include "core/app/ipc.h"

namespace LFL {
void LayersInterface::Update() {
  for (auto &i : this->layer) i->Run(TilesInterface::RunFlag::ClearEmpty);
  if (app->main_process) app->main_process->SwapTree(0, this);
}

void LayersInterface::Draw(const Box &b, const point &p) {
  if (layer.size() > 0) layer[0]->Draw(b, p + node[0].scrolled);
  for (int c=node[0].child_offset; c; c=child[c-1].next_child_offset) {
    const Node *n = &node[child[c-1].node_id-1];
    Scissor s(app->focused->gd, n->box + b.TopLeft());
    if (layer.size() > n->layer_id) layer[n->layer_id]->Draw(b, p + n->scrolled);
  }
}

void TilesInterface::AddDrawableBoxArray(const DrawableBoxArray &box, point p) {
  for (DrawableBoxIterator iter(box.data); !iter.Done(); iter.Increment()) {
    const Drawable::Attr *attr = box.attr.GetAttr(iter.cur_attr1);
    bool attr_set = 0;
    if (attr->bg) {
      ContextOpen();
      if (!attr_set && (attr_set=1)) SetAttr(attr);
      InitDrawBackground(p);
      DrawableBoxRun(iter.Data(), iter.Length(), attr, VectorGet(box.line, iter.cur_attr2))
        .DrawBackground(app->focused->gd, p, bind(&TilesInterface::DrawBackground, this, _1, _2));
      ContextClose();
    }
    if (1) {
      ContextOpen();
      if (!attr_set && (attr_set=1)) SetAttr(attr);
      InitDrawBox(p);
      DrawableBoxRun(iter.Data(), iter.Length(), attr).Draw(app->focused->gd, p, bind(&TilesInterface::DrawBox, this, _1, _2, _3));
      ContextClose();
    }
  }
}

void Tiles::InitDrawBox(const point &p) {
  PreAdd(bind(&DrawableBoxRun::draw, DrawableBoxRun(0,0,attr), app->focused->gd, p));
  if (attr->scissor) AddScissor(*attr->scissor + p);
}

void Tiles::InitDrawBackground(const point &p) {
  PreAdd(bind(&DrawableBoxRun::DrawBackground,
              DrawableBoxRun(0,0,attr), app->focused->gd, p, &DrawableBoxRun::DefaultDrawBackgroundCB));
}

void Tiles::DrawBox(GraphicsContext *gc, const Drawable *d, const Box &b) {
  AddCallback(&b, bind(&Drawable::Draw, d, gc, b));
}

void Tiles::DrawBackground(GraphicsDevice *gd, const Box &b) {
  AddCallback(&b, bind(&GraphicsContext::DrawTexturedBox1, gd, b, NullPointer<float>(), 0));
}

void Tiles::AddScissor(const Box &b) {
  PreAdd(bind(&Tiles::PushScissor, this, b));
  PostAdd(bind(&GraphicsDevice::PopScissor, app->focused->gd));
}

#ifdef  LFL_TILES_IPC_DEBUG
#define TilesIPCDebug(...) DebugPrintf(__VA_ARGS__)
#else
#define TilesIPCDebug(...)
#endif

void TilesIPC::SetAttr(const Drawable::Attr *a) {
  attr = a;
  prepend[context_depth]->cb.Add(MultiProcessPaintResource::SetAttr(*attr));
  TilesIPCDebug("TilesIPC SetAttr %s", a->DebugString().c_str());
}

void TilesIPC::InitDrawBox(const point &p) {
  prepend[context_depth]->cb.Add(MultiProcessPaintResource::InitDrawBox(p));
  if (attr->scissor) AddScissor(*attr->scissor + p);
  TilesIPCDebug("TilesIPC InitDrawBox %s", p.DebugString().c_str());
}

void TilesIPC::InitDrawBackground(const point &p) {
  prepend[context_depth]->cb.Add(MultiProcessPaintResource::InitDrawBackground(p));
  TilesIPCDebug("TilesIPC InitDrawBackground %s", p.DebugString().c_str());
}

void TilesIPC::DrawBox(GraphicsContext *gc, const Drawable *d, const Box &b) {
  if (d) AddCallback(&b, MultiProcessPaintResource::DrawBox(b, d->TexId()));
  TilesIPCDebug("TilesIPC DrawBox %s", b.DebugString().c_str());
}

void TilesIPC::DrawBackground(GraphicsDevice *gd, const Box &b) {
  AddCallback(&b, MultiProcessPaintResource::DrawBackground(b));
  TilesIPCDebug("TilesIPC DrawBackground %s", b.DebugString().c_str());
}

void TilesIPC::AddScissor(const Box &b) {
  prepend[context_depth]->cb.Add(MultiProcessPaintResource::PushScissor(b));
  append [context_depth]->cb.Add(MultiProcessPaintResource::PopScissor());
  TilesIPCDebug("TilesIPC AddScissor %s", b.DebugString().c_str());
}

void TilesIPCClient::Run(int flag) {
  bool clear_empty = (flag & RunFlag::ClearEmpty);
  TilesMatrixIter(&mat) {
    if (!tile->cb.Count() && !clear_empty) continue;
    app->main_process->Paint(layer, point(j, i), flag, tile->cb);
  }
}

int MultiProcessPaintResource::Run(const Box &t) const {
  GraphicsContext gc(app->focused->gd, &attr);
  ProcessAPIClient *s = CheckPointer(app->render_process.get());
  Iterator i(data.buf);
  int si=0, sd=0, count=0; 
  TilesIPCDebug("MPPR Begin");
  for (; i.offset + sizeof(int) <= data.size(); count++) {
    int type = *i.Get<int>();
    switch (type) {
      default:                       FATAL("unknown type ", type);
      case SetAttr           ::Type: { auto c=i.Get<SetAttr>           (); c->Update(&attr, app->render_process.get());           i.offset += SetAttr           ::Size; TilesIPCDebug("MPPR SetAttr %s",     attr.DebugString().c_str()); } break;
      case InitDrawBox       ::Type: { auto c=i.Get<InitDrawBox>       (); DrawableBoxRun(0,0,&attr).draw(gc.gd, c->p);           i.offset += InitDrawBox       ::Size; TilesIPCDebug("MPPR InitDrawBox %s", c->p.DebugString().c_str()); } break;
      case InitDrawBackground::Type: { auto c=i.Get<InitDrawBackground>(); DrawableBoxRun(0,0,&attr).DrawBackground(gc.gd, c->p); i.offset += InitDrawBackground::Size; TilesIPCDebug("MPPR InitDrawBG %s",  c->p.DebugString().c_str()); } break;
      case DrawBackground    ::Type: { auto c=i.Get<DrawBackground>    (); gc.DrawTexturedBox(c->b);                                      i.offset += DrawBackground    ::Size; TilesIPCDebug("MPPR DrawBG %s",      c->b.DebugString().c_str()); } break;
      case PushScissor       ::Type: { auto c=i.Get<PushScissor>       (); gc.gd->PushScissorOffset(t, c->b);     si++;           i.offset += PushScissor       ::Size; TilesIPCDebug("MPPR PushScissor %s", c->b.DebugString().c_str()); } break;
      case PopScissor        ::Type: { auto c=i.Get<PopScissor>        (); gc.gd->PopScissor(); CHECK_LT(sd, si); sd++;           i.offset += PopScissor        ::Size; TilesIPCDebug("MPPR PopScissor");                                 } break;
      case DrawBox           ::Type: { auto c=i.Get<DrawBox>           ();
                                       auto d=(c->id > 0 && c->id <= s->drawable.size()) ? s->drawable[c->id-1] : Singleton<BoxFilled>::Get();
                                       d->Draw(&gc, c->b); i.offset += DrawBox::Size; TilesIPCDebug("DrawBox MPPR %s", c->b.DebugString().c_str());
                                     } break;
    }
  }
  if (si != sd) { ERROR("mismatching scissor ", si, " != ", sd); for (int i=sd; i<si; ++i) app->focused->gd->PopScissor(); }
  TilesIPCDebug("MPPR End");
  return count;
}

void MultiProcessPaintResource::SetAttr::Update(Drawable::Attr *o, ProcessAPIClient *s) const {
  *o = Drawable::Attr((font_id > 0 && font_id <= s->font_table.size()) ? s->font_table[font_id-1] : NULL,
                      hfg ? &fg : NULL, hbg ? &bg : NULL, underline, blend);
  if (hs) o->scissor = &scissor;
  if (tex_id > 0 && tex_id <= s->drawable.size()) o->tex = dynamic_cast<Texture*>(s->drawable[tex_id-1]);
}

}; // namespace LFL
