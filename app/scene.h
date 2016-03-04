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

#ifndef LFL_CORE_APP_SCENE_H__
#define LFL_CORE_APP_SCENE_H__
namespace LFL {

struct Animation {
  Time start;
  Shader *shader=0;
  unsigned short id=0, seq=0, len=0;
  void Reset() { shader = 0; id = seq = len = 0; }
  void Increment() { if (!id) return; seq=(Now()-start).count(); if (seq >= len) Reset(); }
  void Start(Shader *Shader=0) { start=Now()-milliseconds(seq); shader=Shader; }
  void Start(unsigned short Id, unsigned short Seq=0, unsigned short Len=500, Shader *Shader=0)
  { start=Now()-Time(Seq); shader=Shader; id=Id; seq=Seq; len=Len; }
  float Percent() const { return id ? duration_cast<FTime>(Now() - start).count() / len : 0; }
  bool ShaderActive() const { return id && shader; }
};

struct Entity {
  struct Type { enum { STATIC=1, PLAYER=2, BOT=3 }; };
  typedef function<void(Asset*, Entity*)> DrawCB;

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
  DrawCB cb;

  Entity() {}
  Entity(const char *N, Asset *A, const DrawCB &C=DrawCB())               : name(N), asset(A),                        namehash(fnv32(N)), cb(C) {}
  Entity(const char *N, Asset *A, const v3 &p)                            : name(N), asset(A), pos(p),                namehash(fnv32(N)) {}
  Entity(const char *N, Asset *A, const v3 &p, const v3 &v)               : name(N), asset(A), pos(p), vel(v),        namehash(fnv32(N)) {}
  Entity(const char *N, Asset *A, const v3 &p, const v3 &o, const v3 &u)  : name(N), asset(A), pos(p), ort(o), up(u), namehash(fnv32(N)) {}

  Entity(const char *N, const v3 &p, const v3 &v)              : name(N), pos(p), vel(v),        namehash(fnv32(N)) {}
  Entity(const char *N, const v3 &p, const v3 &o, const v3 &u) : name(N), pos(p), ort(o), up(u), namehash(fnv32(N)) {}

  Entity(const v3 &p, const v3 &v)              : pos(p), vel(v) {}
  Entity(const v3 &p, const v3 &o, const v3 &u) : pos(p), ort(o), up(u) {}

  v3 Right() const { v3 r = v3::Cross(ort, up); r.Norm(); return r; }
  void Look(GraphicsDevice *gd) const { v3 targ = pos; targ.Add(ort); gd->LookAt(pos, targ, up); }
  void SetName(const string &n) { name=n; namehash=fnv32(n.c_str()); }

  void MoveUp   (unsigned t) { AddUp   (t/ 1000.0*FLAGS_ksens); }
  void MoveDown (unsigned t) { AddUp   (t/-1000.0*FLAGS_ksens); }
  void MoveFwd  (unsigned t) { AddOrt  (t/ 1000.0*FLAGS_ksens); }
  void MoveRev  (unsigned t) { AddOrt  (t/-1000.0*FLAGS_ksens); }
  void MoveRight(unsigned t) { AddRight(t/ 1000.0*FLAGS_ksens); }
  void MoveLeft (unsigned t) { AddRight(t/-1000.0*FLAGS_ksens); }
  void YawRight (unsigned t) { RotUp   (t/ 1000.0*FLAGS_msens); }
  void YawLeft  (unsigned t) { RotUp   (t/-1000.0*FLAGS_msens); }
  void RollLeft (unsigned t) { RotOrt  (t/ 1000.0*FLAGS_msens); }
  void RollRight(unsigned t) { RotOrt  (t/-1000.0*FLAGS_msens); }
  void PitchDown(unsigned t) { RotRight(t/ 1000.0*FLAGS_msens*FLAGS_invert); }
  void PitchUp  (unsigned t) { RotRight(t/-1000.0*FLAGS_msens*FLAGS_invert); }

  void AddUp   (float f) { v3 u = up;      u.Scale(f); pos.Add(u); }
  void AddOrt  (float f) { v3 s = ort;     s.Scale(f); pos.Add(s); }
  void AddRight(float f) { v3 r = Right(); r.Scale(f); pos.Add(r); }
  void RotUp   (float f) { m33 m = m33::RotAxis(f, up);      ort = m.Transform(ort); }
  void RotOrt  (float f) { m33 m = m33::RotAxis(f, ort);     up  = m.Transform(up); }
  void RotRight(float f) { m33 m = m33::RotAxis(f, Right()); up  = m.Transform(up); ort = m.Transform(ort); }

  static bool ZSort(const Entity *l, const Entity *r) { return l->zsort > r->zsort; }
};

struct Scene {
  typedef vector<Entity*> EntityVector;
  typedef map<string, Entity*> EntityMap;
  typedef map<string, EntityVector> EntityAssetMap;

  struct EntityFilter { virtual bool Filter(Entity *e) = 0; };
  struct SingleEntityFilter : public EntityFilter {
    Entity *entity;
    SingleEntityFilter(Entity *e) : entity(e) {}
    bool Filter(Entity *e) { return entity == e; }
  };
  struct LastUpdatedFilter : public EntityFilter {
    EntityFilter *super;
    Time cutoff;
    EntityVector *filtered;

    LastUpdatedFilter(EntityFilter *Super, Time val, EntityVector *out=0) : super(Super), cutoff(val), filtered(out) {}
    bool Filter(Entity *e) {
      if (super && super->Filter(e)) return true;
      if (e->updated >= cutoff) return false;
      if (filtered) filtered->push_back(e);
      return true;
    }
  };

  EntityMap entity;
  EntityAssetMap asset;
  EntityVector zsort;

  Entity *Get(const string &n) { return FindOrNull(entity, n); }
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

  void Draw(vector<Asset> *assets);
  void Draw(Asset *a, EntityFilter *filter=0);
  static void Draw(Asset *a, EntityFilter *filter, const EntityVector&);

  void ZSortDraw(EntityFilter *filter, unsigned dt);
  void ZSort(const vector<Asset> &assets);
};

}; // namespace LFL
#endif // LFL_CORE_APP_SCENE_H__
