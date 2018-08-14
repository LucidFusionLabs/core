/*
 * $Id$
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

#include "core/game/particles.h"

namespace LFL {
DEFINE_bool(hull_geometry, false, "Draw entity bounding hull"); 

Entity *Scene::Add(const string &name, unique_ptr<Entity> ent) {
  if (name.empty()) return 0;
  auto e = (entity[name] = move(ent)).get();
  if (e->asset) {
    EntityVector &eav = asset[e->asset->name];
    eav.push_back(e);
    sort(eav.begin(), eav.end(), Entity::ZSort);
  }
  return e;
}

bool Scene::ChangeAsset(const string &entity_name, Asset *new_asset) {
  Entity *e = FindUniqueOrNull(entity, entity_name);
  return e ? ChangeAsset(e, new_asset) : false;
}

bool Scene::ChangeAsset(Entity *e, Asset *new_asset) {
  if (e->asset) {
    EntityVector &eav = asset[e->asset->name];
    for (auto j = eav.begin(); j != eav.end(); ++j) if (*j == e) { eav.erase(j); break; }
  }
  e->asset = new_asset;
  if (new_asset) {
    EntityVector &eav = asset[e->asset->name];
    eav.push_back(e);
    sort(eav.begin(), eav.end(), Entity::ZSort);
  }
  return true;
}

void Scene::Del(const string &name) { 
  auto i = entity.find(name);
  if (i == entity.end()) return;
  Entity *e = i->second.get();
  if (e->asset) {
    EntityVector &eav = asset[e->asset->name];
    for (auto j = eav.begin(); j != eav.end(); ++j) if (*j == e) { eav.erase(j); break; }
  }
  entity.erase(i);
}

void Scene::Del(const EntityVector &deleted) {
  for (auto e : deleted) Del(e->name);
}

void Scene::Select(GraphicsDevice *gd, Geometry *geom) {
  int width=0, vert_size=0;
  if (geom) {
    width = geom->width * sizeof(float);
    vert_size = geom->count * width;
    gd->VertexPointer(geom->vd, gd->c.Float, width, 0, &geom->vert[0], vert_size, &geom->vert_ind, false, geom->primtype);
  }

  if (geom && geom->tex_offset >= 0) {
    gd->EnableTexture();
    gd->TexPointer(geom->td, gd->c.Float, width, geom->tex_offset, &geom->vert[0], vert_size, &geom->vert_ind, false);
  }
  else gd->DisableTexture();

  if (geom && geom->norm_offset >= 0) {
    gd->EnableLighting();
    gd->EnableNormals();
    gd->NormalPointer(geom->vd, gd->c.Float, width, geom->norm_offset, &geom->vert[0], vert_size, &geom->vert_ind, false);
  }
  else {
    gd->DisableLighting();
    gd->DisableNormals();
  }

  if (geom && geom->color_offset >= 0) {
    gd->EnableVertexColor();
    gd->ColorPointer(4, gd->c.Float, width, geom->color_offset, &geom->vert[0], vert_size, &geom->vert_ind, false);
  }
  else {
    gd->DisableVertexColor();
  }

  if (geom && geom->color) gd->SetColor(geom->col);
}

void Scene::Select(GraphicsDevice *gd, const Asset *a) {
  if (FLAGS_hull_geometry && a && a->hull) Select(gd, a->hull);
  else if (a && a->geometry)               Select(gd, a->geometry);

  if (a && a->tex.ID && !(FLAGS_hull_geometry && a->hull)) {
    if (a->tex.cubemap) {
      gd->DisableTexture();
      gd->BindCubeMap(a->tex.ID);
      if      (a->texgen == TexGen::LINEAR)     gd->TextureGenLinear();
      else if (a->texgen == TexGen::REFLECTION) gd->TextureGenReflection();
    } else {
      gd->DisableCubeMap();
      gd->EnableTexture();
      gd->BindTexture(gd->c.Texture2D, a->tex.ID);
    }

    gd->EnableBlend();
    gd->BlendMode(a->blends, a->blendt);
  }
  else {
    gd->DisableBlend();
    gd->DisableTexture();
  }

  if (a && a->color) gd->SetColor(a->col);
  else gd->Color4f(1,1,1,1);

  if (!a || !a->tex.ID) gd->DisableTexture();
}

void Scene::Select(GraphicsDevice *gd) {
  Scene::Select(gd, NullPointer<Geometry>());
  Scene::Select(gd, NullPointer<const Asset>());
}

void Scene::Draw(GraphicsDevice *gd, const Geometry *geom, Entity*, int start_vert, int num_verts) {
  gd->DrawArrays(geom->primtype, start_vert, num_verts ? num_verts : geom->count);
}

void Scene::Draw(GraphicsDevice *gd, Asset *a, Entity *e) {
  gd->MatrixModelview();
  gd->PushMatrix();

  if (a->translate) {
    v3 ort   = e->ort;                   ort  .Norm();
    v3 up    = e->up;                    up   .Norm();
    v3 right = v3::Cross(e->ort, e->up); right.Norm();

    if (float s = a->scale) {
      float m[16] = { s*right.x, s*right.y, s*right.z, 0,
                      s*up.x,    s*up.y,    s*up.z,    0,
                      s*ort.x,   s*ort.y,   s*ort.z,   0,
                      e->pos.x,  e->pos.y,  e->pos.z,  1 };
      if (gd->track_model_matrix) gd->model_matrix = m44(m);
      gd->Mult(m);
    } else {
      float m[16] = { right.x,  right.y,  right.z,  0,
                      up.x,     up.y,     up.z,     0,
                      ort.x,    ort.y,    ort.z,    0,
                      e->pos.x, e->pos.y, e->pos.z, 1 };
      if (gd->track_model_matrix) gd->model_matrix = m44(m);
      gd->Mult(m);
    }
  } else {
    if (a->rotate) gd->Rotatef(a->rotate, 0, 1, 0);
    if (a->scale && !(FLAGS_hull_geometry && a->hull)) gd->Scalef(a->scale, a->scale, a->scale);
  }

  if (a->hull && FLAGS_hull_geometry) {
    Draw(gd, a->hull, e);
    return;
  }

  if (a->cb) {
    if (FLAGS_gd_debug) DebugPrintf("scene.DrawAssetCB %s", a->name.c_str());
    a->cb(gd, a, e);
  }

  if (e->cb) {
    if (FLAGS_gd_debug) DebugPrintf("scene.DrawEntityCB %s", a->name.c_str());
    e->cb(a, e);
  }

  if (a->geometry) {
    if (FLAGS_gd_debug) DebugPrintf("scene.DrawGeometry %s", a->name.c_str());
    Draw(gd, a->geometry, e);
  }

  gd->PopMatrix();
}

void Scene::Draw(GraphicsDevice *gd, vector<Asset> *assets) {
  for (auto &a : *assets) {
    if (a.zsort) continue;
    Draw(gd, &a);
  }
}

void Scene::Draw(GraphicsDevice *gd, Asset *a, EntityFilter *filter) {
  EntityVector &eav = asset[a->name];
  Draw(gd, a, filter, eav);
}

void Scene::Draw(GraphicsDevice *gd, Asset *a, EntityFilter *filter, const EntityVector &eav) {
  if (a->tex.cubemap && a->tex.cubemap != CubeMap::PX) return;

  Select(gd, a);

  for (auto e : eav) {
    if (filter && filter->Filter(e)) continue;
    Draw(gd, a, e);
  }
}

void Scene::DrawParticles(GraphicsDevice *gd, Entity *e, unsigned dt) {
  if (!e->particles) return;
  ParticleSystem *particles = reinterpret_cast<ParticleSystem*>(e->particles);
  particles->pos = e->pos;
  particles->ort = e->ort;
  particles->updir = e->up;
  particles->vel = e->ort;
  particles->vel.Scale(-0.01);
  particles->Update(&cam, dt, 0, 0, 0);
  particles->Draw(gd);
  gd->EnableDepthTest();
}

void Scene::ZSortDraw(GraphicsDevice *gd, EntityFilter *filter, unsigned dt) {
  Asset *last_asset = 0;
  for (auto e : zsort) {
    if (filter && filter->Filter(e)) continue;

    float zangle = v3::Dot(cam.ort, e->ort);
    if (zangle <= 0) { DrawParticles(gd, e, dt); last_asset=0; }

    if (e->asset != last_asset) { Select(gd, e->asset); last_asset = e->asset; }
    Draw(gd, e->asset, e);

    if (zangle > 0) { DrawParticles(gd, e, dt); last_asset=0; }
  }
}

void Scene::ZSort(const vector<Asset> &assets) {
  zsort.clear();

  for (auto const &a : assets) { 
    if (!a.zsort) continue;

    for (auto e : asset[a.name]) {
      e->zsort = v3::Dot(cam.ort, e->pos);
      zsort.push_back(e);
    }
  }

  sort(zsort.begin(), zsort.end(), Entity::ZSort);
}

}; // namespace LFL
