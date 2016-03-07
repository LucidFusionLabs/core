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

#ifndef LFL_CORE_GAME_PROTO_H__
#define LFL_CORE_GAME_PROTO_H__
namespace LFL {

struct GameProtocol {
  struct Header : public Serializable::Header {};

  struct Position {
    static const int size = 12, scale = 1000;
    int32_t x, y, z;

    void From(const v3 &v) { x=int32_t(v.x*scale); y=int32_t(v.y*scale); z=int32_t(v.z*scale); }
    void To(v3 *v) { v->x=float(x)/scale; v->y=float(y)/scale; v->z=float(z)/scale; }
    void Out(Serializable::Stream *o) const { o->Htonl( x); o->Htonl( y); o->Htonl( z); }
    void In(const Serializable::Stream *i)  { i->Ntohl(&x); i->Ntohl(&y); i->Ntohl(&z); }
  };

  struct Orientation {
    static const int size = 12, scale=16384;
    int16_t ort_x, ort_y, ort_z, up_x, up_y, up_z;

    void From(const v3 &ort, const v3 &up) {
      ort_x = int16_t(ort.x*scale); ort_y = int16_t(ort.y*scale); ort_z = int16_t(ort.z*scale);
      up_x  = int16_t(up.x *scale); up_y  = int16_t(up.y *scale); up_z  = int16_t(up.z *scale);
    }
    void To(v3 *ort, v3 *up) {
      ort->x = float(ort_x)/scale; ort->y = float(ort_y)/scale; ort->z = float(ort_z)/scale;
      up ->x = float(up_x) /scale; up ->y = float(up_y) /scale; up ->z = float(up_z) /scale;
    }
    void Out(Serializable::Stream *o) const { o->Htons( ort_x); o->Htons( ort_y); o->Htons( ort_z); o->Htons( up_x); o->Htons( up_y); o->Htons( up_z); }
    void In(const Serializable::Stream *i)  { i->Ntohs(&ort_x); i->Ntohs(&ort_y); i->Ntohs(&ort_z); i->Ntohs(&up_x); i->Ntohs(&up_y); i->Ntohs(&up_z); }
  };

  struct Velocity {
    static const int size = 6, scale=1000;
    uint16_t x, y, z;

    void From(const v3 &v) { x=uint16_t(v.x*scale); y=uint16_t(v.y*scale); z=uint16_t(v.z*scale); }
    void To(v3 *v) { v->x=float(x)/scale; v->y=float(y)/scale; v->z=float(z)/scale; }
    void Out(Serializable::Stream *o) const { o->Htons( x); o->Htons( y); o->Htons( z); }
    void In(const Serializable::Stream *i)  { i->Ntohs(&x); i->Ntohs(&y); i->Ntohs(&z); }
  };

  struct Entity {
    static const int size = 8 + Position::size + Orientation::size + Velocity::size;
    uint16_t id, type, anim_id, anim_len;
    Position pos;
    Orientation ort;
    Velocity vel;

    void From(const LFL::Entity *e) { id=atoi(e->name.c_str()); type=e->asset?e->asset->typeID:0; anim_id=e->animation.id; anim_len=e->animation.len; pos.From(e->pos); ort.From(e->ort, e->up); vel.From(e->vel); }
    void Out(Serializable::Stream *o) const { o->Htons( id); o->Htons( type); o->Htons( anim_id); o->Htons( anim_len); pos.Out(o); ort.Out(o); vel.Out(o); }
    void In(const Serializable::Stream *i)  { i->Ntohs(&id); i->Ntohs(&type); i->Ntohs(&anim_id); i->Ntohs(&anim_len); pos.In(i);  ort.In(i);  vel.In(i);  }
  };

  struct Collision {
    static const int size = 8;
    uint16_t fmt, id1, id2, time;

    void Out(Serializable::Stream *o) const { o->Htons( fmt); o->Htons( id1); o->Htons( id2); o->Htons( time); }
    void In(const Serializable::Stream *i)  { i->Ntohs(&fmt); i->Ntohs(&id1); i->Ntohs(&id2); i->Ntohs(&time); }
  };

  struct ChallengeRequest : public Serializable {
    static const int ID = 1;
    ChallengeRequest() : Serializable(ID) {}

    int HeaderSize() const { return 0; }
    int Size() const { return HeaderSize(); }
    void Out(Serializable::Stream *o) const {}
    int   In(const Serializable::Stream *i) { return 0; }
  };

  struct ChallengeResponse : public Serializable {
    static const int ID = 2;
    int32_t token;
    ChallengeResponse() : Serializable(ID) {}

    int HeaderSize() const { return 4; }
    int Size() const { return HeaderSize(); }
    void Out(Serializable::Stream *o) const { o->Htonl( token); }
    int   In(const Serializable::Stream *i) { i->Ntohl(&token); return 0; }
  };

  struct JoinRequest : public Serializable {
    static const int ID = 3;
    int32_t token;
    string PlayerName;
    JoinRequest() : Serializable(ID) {}

    int HeaderSize() const { return 4; }
    int Size() const { return HeaderSize() + PlayerName.size(); }
    void Out(Serializable::Stream *o) const { o->Htonl( token); o->String(PlayerName); }
    int   In(const Serializable::Stream *i) { i->Ntohl(&token); PlayerName = i->Get(); return 0; }
  };

  struct JoinResponse : public Serializable {
    static const int ID = 4;
    string rcon;
    JoinResponse() : Serializable(ID) {}

    int HeaderSize() const { return 0; }
    int Size() const { return rcon.size(); }
    void Out(Serializable::Stream *o) const { o->String(rcon); }
    int   In(const Serializable::Stream *i) { rcon = i->Get(); return 0; }
  };

  struct WorldUpdate : public Serializable {
    static const int ID = 5;
    uint16_t id;
    vector<Entity> entity;
    vector<Collision> collision;
    WorldUpdate() : Serializable(ID) {}

    int HeaderSize() const { return 6; }
    int Size() const { return HeaderSize() + entity.size() * Entity::size + collision.size() * Collision::size; }

    void Out(Serializable::Stream *o) const {
      unsigned short entities=entity.size(), collisions=collision.size();
      o->Htons(id); o->Htons(entities); o->Htons(collisions);
      for (int i=0; i<entities;   i++) entity   [i].Out(o);
      for (int i=0; i<collisions; i++) collision[i].Out(o);
    }

    int In(const Serializable::Stream *in) {
      unsigned short entities, collisions;
      in->Ntohs(&id); in->Ntohs(&entities); in->Ntohs(&collisions);
      if (!Check(in)) return -1;

      entity.resize(entities); collision.resize(collisions);
      for (int i=0; i<entities;   i++) entity[i]   .In(in);
      for (int i=0; i<collisions; i++) collision[i].In(in);
      return 0;
    }
  };

  struct PlayerUpdate : public Serializable {
    static const int ID = 6;
    uint16_t id_WorldUpdate, time_since_WorldUpdate;
    uint32_t buttons;
    Orientation ort;
    PlayerUpdate() : Serializable(ID) {}

    int HeaderSize() const { return 8 + Orientation::size; }
    int Size() const { return HeaderSize(); }
    void Out(Serializable::Stream *o) const { o->Htons( id_WorldUpdate); o->Htons( time_since_WorldUpdate); o->Htonl( buttons); ort.Out(o); }
    int   In(const Serializable::Stream *i) { i->Ntohs(&id_WorldUpdate); i->Ntohs(&time_since_WorldUpdate); i->Ntohl(&buttons); ort.In(i); return 0; }
  };

  struct RconRequest : public Serializable {
    static const int ID = 7;
    string Text;
    RconRequest(const string &t=string()) : Serializable(ID), Text(t) {}

    int HeaderSize() const { return 0; }
    int Size() const { return HeaderSize() + Text.size(); }
    void Out(Serializable::Stream *o) const { o->String(Text); }
    int   In(const Serializable::Stream *i) { Text = i->Get(); return 0; }
  };

  struct RconResponse : public Serializable {
    static const int ID = 8;
    RconResponse() : Serializable(ID) {}

    int HeaderSize() const { return 0; }
    int Size() const { return HeaderSize(); }
    void Out(Serializable::Stream *o) const {}
    int   In(const Serializable::Stream *i) { return 0; }
  };

  struct PlayerList : public RconRequest {
    static const int ID = 9;
    PlayerList() { Id=ID; }
  };
};

}; // namespace LFL
#endif // LFL_CORE_GAME_PROTO_H__
