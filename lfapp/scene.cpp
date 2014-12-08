/*
 * $Id: scene.cpp 1314 2014-10-16 04:43:45Z justin $
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

#include "lfapp/lfapp.h"

namespace LFL {
DEFINE_bool(hull_geometry, false, "Draw entity bounding hull"); 

Entity *Scene::Add(const string &name, Entity *e) {
    if (!name.size()) return 0;
    entityMap[name] = e;

    if (e->asset) {
        EntityVector &eav = assetMap[e->asset->name];
        eav.push_back(e);
        sort(eav.begin(), eav.end(), Entity::cmp);
    }
    return e;
}

bool Scene::ChangeAsset(const string &entity_name, Asset *new_asset) {
    Entity *e = FindOrNull(entityMap, entity_name);
    return e ? ChangeAsset(e, new_asset) : false;
}

bool Scene::ChangeAsset(Entity *e, Asset *new_asset) {
    if (e->asset) {
        EntityVector &eav = assetMap[e->asset->name];
        for (EntityVector::iterator j = eav.begin(); j != eav.end(); j++) if (*j == e) { eav.erase(j); break; }
    }
    e->asset = new_asset;
    if (new_asset) {
        EntityVector &eav = assetMap[e->asset->name];
        eav.push_back(e);
        sort(eav.begin(), eav.end(), Entity::cmp);
    }
    return true;
}

void Scene::Del(const string &name) { 
    EntityMap::iterator i = entityMap.find(name);
    if (i == entityMap.end()) return;
    Entity *e = (*i).second;
    if (e->asset) {
        EntityVector &eav = assetMap[e->asset->name];
        for (EntityVector::iterator j = eav.begin(); j != eav.end(); j++) if (*j == e) { eav.erase(j); break; }
    }
    entityMap.erase(i);
}

void Scene::Del(const EntityVector &deleted) {
    for (Scene::EntityVector::const_iterator i = deleted.begin(); i != deleted.end(); i++) Del((*i)->name);
}

void Scene::Select(Geometry *geom) {
    int width=0, vert_size=0;
    if (geom) {
        width = geom->width * sizeof(float);
        vert_size = geom->count * width;
        screen->gd->VertexPointer(geom->vd, GraphicsDevice::Float, width, 0, &geom->vert[0], vert_size, &geom->vert_ind, false);
    }

    if (geom && geom->tex_offset >= 0) {
        screen->gd->EnableTexture();
        screen->gd->TexPointer(geom->td, GraphicsDevice::Float, width, geom->tex_offset, &geom->vert[0], vert_size, &geom->vert_ind, false);
    }
    else screen->gd->DisableTexture();

    if (geom && geom->norm_offset >= 0) {
        screen->gd->EnableLighting();
        screen->gd->EnableNormals();
        screen->gd->NormalPointer(geom->vd, GraphicsDevice::Float, width, geom->norm_offset, &geom->vert[0], vert_size, &geom->vert_ind, false);
    }
    else {
        screen->gd->DisableLighting();
        screen->gd->DisableNormals();
    }

    if (geom && geom->color_offset >= 0) {
        screen->gd->EnableVertexColor();
        screen->gd->ColorPointer(4, GraphicsDevice::Float, width, geom->color_offset, &geom->vert[0], vert_size, &geom->vert_ind, false);
    }
    else {
        screen->gd->DisableVertexColor();
    }

    if (geom && geom->color) screen->gd->SetColor(geom->col);
}

void Scene::Select(const Asset *a) {
    if (a && a->tex.ID && !(FLAGS_hull_geometry && a->hull)) {
        if (a->tex.cubemap) {
            screen->gd->DisableTexture();
            screen->gd->BindCubeMap(a->tex.ID);

            if      (a->texgen == TexGen::LINEAR)     screen->gd->TextureGenLinear();
            else if (a->texgen == TexGen::REFLECTION) screen->gd->TextureGenReflection();

        } else {
            screen->gd->DisableCubeMap();
            screen->gd->EnableTexture();
            screen->gd->BindTexture(GraphicsDevice::Texture2D, a->tex.ID);
        }

        screen->gd->EnableBlend();
        screen->gd->BlendMode(a->blends, a->blendt);
    }
    else {
        screen->gd->DisableBlend();
        screen->gd->DisableTexture();
    }

    screen->gd->Color4f(1,1,1,1);

    if (FLAGS_hull_geometry && a && a->hull) Select(a->hull);
    else if (a && a->geometry)               Select(a->geometry);

    if (a && a->color) screen->gd->SetColor(a->col);

    if (!a || !a->tex.ID) screen->gd->DisableTexture();
}

void Scene::Select() {
    Scene::Select((Geometry*)0);
    Scene::Select((const Asset*)0);
}

void Scene::Draw(const Geometry *geom, Entity*, int start_vert, int num_verts) {
    screen->gd->DrawArrays(geom->primtype, start_vert, num_verts ? num_verts : geom->count);
}

void Scene::Draw(Asset *a, Entity *e) {
    screen->gd->MatrixModelview();
    screen->gd->PushMatrix();

    if (a->translate) {
        v3 ort = e->ort; ort.norm();
        v3 up = e->up; up.norm();
        v3 right = v3::cross(e->ort, e->up); right.norm();

        float m[16] = { right.x, right.y, right.z, 0,
                        up.x,    up.y,    up.z,    0,
                        ort.x,   ort.y,   ort.z,   0,
                        0,       0,       0,       1 };

        screen->gd->Translate(e->pos.x, e->pos.y, e->pos.z);
        screen->gd->Mult(m);
    }
    if (a->rotate) screen->gd->Rotatef(a->rotate, 0, 1, 0);
    if (a->scale && !(FLAGS_hull_geometry && a->hull)) screen->gd->Scalef(a->scale, a->scale, a->scale);

    if (a->hull && FLAGS_hull_geometry) {
        Draw(a->hull, e);
        return;
    }

    if (a->cb) {
        if (FLAGS_gd_debug) INFO("scene.DrawCB ", a->name);
        a->cb(a, e);
    }

    if (a->geometry) {
        if (FLAGS_gd_debug) INFO("scene.DrawGeometry ", a->name);
        Draw(a->geometry, e);
    }

    screen->gd->PopMatrix();
}

void Scene::Draw(vector<Asset> *assets) {
    for (vector<Asset>::iterator a = assets->begin(); a != assets->end(); ++a) {
        if (a->zsort) continue;
        Draw(&(*a));
    }
}

void Scene::Draw(Asset *a, Filter *filter) {
    EntityVector &eav = assetMap[a->name];
    Draw(a, filter, eav);
}

void Scene::Draw(Asset *a, Filter *filter, const EntityVector &eav) {
    if (a->tex.cubemap && a->tex.cubemap != CubeMap::PX) return;

    Select(a);

    for (EntityVector::const_iterator j = eav.begin(); j != eav.end(); j++) {
        Entity *e = *j;
        if (filter && filter->filter(e)) continue;

        Draw(a, e);
    }
}

void Scene::DrawParticles(Entity *e, unsigned dt) {
    if (!e->particles) return;
    ParticleSystem *particles = (ParticleSystem*)e->particles;
    particles->pos = e->pos;
    particles->ort = e->ort;
    particles->updir = e->up;
    particles->vel = e->ort;
    particles->vel.scale(-0.01);
    particles->Update(dt,0,0,0);
    particles->Draw();
    screen->gd->EnableDepthTest();
}

void Scene::ZSortDraw(Filter *filter, unsigned dt) {
    Asset *last_asset = 0;
    for (EntityVector::const_iterator j = zsortVector.begin(); j != zsortVector.end(); j++) {
        Entity *e = *j;
        if (filter && filter->filter(e)) continue;

        float zangle = v3::dot(screen->camMain->ort, e->ort);
        if (zangle <= 0) { DrawParticles(e, dt); last_asset=0; }

        if (e->asset != last_asset) { Select(e->asset); last_asset = e->asset; }
        Draw(e->asset, e);

        if (zangle > 0) { DrawParticles(e, dt); last_asset=0; }
    }
}

void Scene::ZSort(const vector<Asset> &assets) {
    zsortVector.clear();

    for (vector<Asset>::const_iterator a = assets.begin(); a != assets.end(); ++a) {
        if (!a->zsort) continue;

        EntityVector &eav = assetMap[a->name];
        for (EntityVector::const_iterator j = eav.begin(); j != eav.end(); j++) {
             Entity *e = *j;
             e->zsort = v3::dot(screen->camMain->ort, e->pos);
             zsortVector.push_back(e);
        }
    }

    sort(zsortVector.begin(), zsortVector.end(), Entity::cmp);
}

}; // namespace LFL
