/*
 * $Id: scene.h 1335 2014-12-02 04:13:46Z justin $
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

#ifndef __LFL_LFAPP_SCENE_H__
#define __LFL_LFAPP_SCENE_H__
namespace LFL {

struct Animation {
    Time start=0;
    Shader *shader=0;
    unsigned short id=0, seq=0, len=0;
    void Reset() { shader = 0; id = seq = len = 0; }
    void Increment() { if (!id) return; seq=Now()-start; if (seq >= len) Reset(); }
    void Start(Shader *Shader=0) { start=Now()-seq; shader=Shader; }
    void Start(unsigned short Id, unsigned short Seq=0, unsigned short Len=MilliSeconds(500), Shader *Shader=0)
    { start=Now()-Seq; shader=Shader; id=Id; seq=Seq; len=Len; }
    float Percent() const { return id ? (float)(Now() - start) / len : 0; }
    bool ShaderActive() const { return id && shader; }
};

struct Entity {
    string name;
    Asset *asset=0; 
    v3 pos=v3(0,0,0), ort=v3(0,0,1), up=v3(0,1,0), vel;
    float zsort=0;
    unsigned buttons=0, namehash=0;
    Animation animation;
    Time updated;
    void *body=0, *particles=0, *userdata=0;
    Color color1, color2;
    int type=0;
    struct Type { enum { STATIC=1, PLAYER=2, BOT=3 }; };

    Entity() {}
    Entity(const char *N, Asset *A) : name(N), asset(A), namehash(fnv32(N)) {}
    Entity(const char *N, Asset *A, const v3 &p) : name(N), asset(A), pos(p), namehash(fnv32(N)) {}
    Entity(const char *N, Asset *A, const v3 &p, const v3 &v) : name(N), asset(A), pos(p), vel(v), namehash(fnv32(N)) {}
    Entity(const char *N, Asset *A, const v3 &p, const v3 &o, const v3 &u) : name(N), asset(A), pos(p), ort(o), up(u), namehash(fnv32(N)) {}

    Entity(const char *N, const v3 &p, const v3 &v) : name(N), pos(p), vel(v), namehash(fnv32(N)) {}
    Entity(const char *N, const v3 &p, const v3 &o, const v3 &u) : name(N), pos(p), ort(o), up(u), namehash(fnv32(N)) {}

    Entity(const v3 &p, const v3 &o, const v3 &u) : pos(p), ort(o), up(u) {}
    Entity(const v3 &p, const v3 &v) : pos(p), vel(v) {}
    static bool cmp(const Entity *l, const Entity *r) { return l->zsort > r->zsort; }

    void SetName(const string &n) { name=n; namehash=fnv32(n.c_str()); }
    void raise(float f) { v3 u = up;  u.scale( f); pos.add(u); }
    void lower(float f) { v3 d = up;  d.scale(-f); pos.add(d); }
    void fwd  (float f) { v3 s = ort; s.scale( f); pos.add(s); }
    void rev  (float f) { v3 r = ort; r.scale(-f); pos.add(r); }
    void right(float f) { v3 r = v3::cross(ort, up); r.scale( f); pos.add(r); }
    void left (float f) { v3 l = v3::cross(ort, up); l.scale(-f); pos.add(l); } 
    void rollleft (float f) { m33 m = m33::rotaxis( f, ort.x, ort.y, ort.z); up  = m.transform(up ); }
    void rollright(float f) { m33 m = m33::rotaxis(-f, ort.x, ort.y, ort.z); up  = m.transform(up ); }
    void yawleft  (float f) { m33 m = m33::rotaxis(-f, up .x, up .y, up .z); ort = m.transform(ort); }
    void yawright (float f) { m33 m = m33::rotaxis( f, up .x, up .y, up .z); ort = m.transform(ort); }
    void pitchup  (float f) { v3 r = v3::cross(ort, up); r.norm(); m33 m = m33::rotaxis( f, r.x, r.y, r.z); up = m.transform(up); ort = m.transform(ort); }
    void pitchdown(float f) { v3 r = v3::cross(ort, up); r.norm(); m33 m = m33::rotaxis(-f, r.x, r.y, r.z); up = m.transform(up); ort = m.transform(ort); }
    void look() { v3 targ = pos; targ.add(ort); screen->gd->LookAt(pos, targ, up); }

    void MoveUp   (unsigned t) { raise    (t/1000.0*FLAGS_ksens); }
    void MoveDown (unsigned t) { lower    (t/1000.0*FLAGS_ksens); }
    void MoveFwd  (unsigned t) { fwd      (t/1000.0*FLAGS_ksens); }
    void MoveRev  (unsigned t) { rev      (t/1000.0*FLAGS_ksens); }
    void MoveLeft (unsigned t) { left     (t/1000.0*FLAGS_ksens); }
    void MoveRight(unsigned t) { right    (t/1000.0*FLAGS_ksens); }
    void RollLeft (unsigned t) { rollleft (t/1000.0*FLAGS_msens); }
    void RollRight(unsigned t) { rollright(t/1000.0*FLAGS_msens); }
    void YawLeft  (unsigned t) { yawleft  (t/1000.0*FLAGS_msens); }
    void YawRight (unsigned t) { yawright (t/1000.0*FLAGS_msens); }
    void PitchUp  (unsigned t) { pitchup  (t/1000.0*FLAGS_msens*FLAGS_invert); }
    void PitchDown(unsigned t) { pitchdown(t/1000.0*FLAGS_msens*FLAGS_invert); }
};

struct Scene {
    typedef map<string, Entity*> EntityMap;
    EntityMap entityMap;

    typedef vector<Entity*> EntityVector;
    typedef map<string, EntityVector> EntityAssetMap;
    EntityAssetMap assetMap;

    EntityVector zsortVector;

    Entity *Get(const string &n) { return FindOrNull(entityMap, n); }
    Entity *Add(Entity *e) { return Add(e->name, e); }
    Entity *Add(const string &name, Entity *);
    bool ChangeAsset(const string &entity_name, Asset *new_asset);
    bool ChangeAsset(Entity *e, Asset *new_asset);
    void Del(const string &name);
    void Del(const EntityVector &vec);

    static void Select(const Asset *);
    static void Select(Geometry *);
    static void Select();

    static void Draw(Asset *a, Entity*);
    static void Draw(const Geometry *a, Entity*, int start_vert=0, int num_verts=0);
    static void DrawParticles(Entity *e, unsigned dt);

    struct Filter { virtual bool filter(Entity *e) = 0; };
    struct EntityFilter : public Filter {
        Entity *entity;
        EntityFilter(Entity *e) : entity(e) {}
        bool filter(Entity *e) { return entity == e; }
    };
    struct LastUpdatedFilter : public Filter {
        Filter *super;
        Time cutoff;
        EntityVector *filtered;

        LastUpdatedFilter(Filter *Super, Time val, EntityVector *out=0) : super(Super), cutoff(val), filtered(out) {}
        bool filter(Entity *e) {
            if (super && super->filter(e)) return true;
            if (e->updated >= cutoff) return false;
            if (filtered) filtered->push_back(e);
            return true;
        }
    };

    void Draw(vector<Asset> *assets);
    void Draw(Asset *a, Filter *filter=0);
    static void Draw(Asset *a, Filter *filter, const EntityVector&);

    void ZSortDraw(Filter *filter, unsigned dt);
    void ZSort(const vector<Asset> &assets);
};

}; // namespace LFL
#endif // __LFL_LFAPP_SCENE_H__
