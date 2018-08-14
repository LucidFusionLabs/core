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

#ifndef LFL_CORE_GAME_PARTICLES_H__
#define LFL_CORE_GAME_PARTICLES_H__
namespace LFL {
  
struct ParticleSystem {
  string name;
  Color color;
  v3 pos, vel, ort, updir;
  vector<v3> *pos_transform;
  int pos_transform_index;
  ParticleSystem(const string &n) : name(n), ort(0,0,1), updir(0,1,0), pos_transform(0), pos_transform_index(0) {}
  virtual void Update(Entity *cam, unsigned dt, int mx, int my, int mdown) = 0;
  virtual void Draw(GraphicsDevice*) = 0;
};

template <int MP, int MH, bool PerParticleColor> struct Particles : public ParticleSystem {
  typedef Particles<MP, MH, PerParticleColor> ParticlesType;
  static const int MaxParticles=MP, MaxHistory=MH, VertFloats=(PerParticleColor ? 9 : 5), VertSize=VertFloats*sizeof(float);
  static const int ParticleVerts=6, ParticleSize=ParticleVerts*VertSize, NumFloats=MaxParticles*ParticleVerts*VertFloats;
  static const int Trails=MaxHistory>2, TrailVertFloats=(PerParticleColor ? 7 : 3), TrailVertSize=TrailVertFloats*sizeof(float);
  static const int MaxTrailVerts=6*(MaxHistory-2), NumTrailFloats=(Trails ? MaxParticles*MaxTrailVerts*TrailVertFloats : 1);
  struct Emitter { enum { None=0, Mouse=1, Sprinkler=2, RainbowFade=4, GlowFade=8, FadeFromWhite=16 }; };

  struct Particle {
    ParticlesType *config;
    v3 history[MH], vel;
    int history_len, bounceage;
    float radius, age, maxage, remaining;
    Color color, start_color;
    bool dead;

    void InitColor() {
      if (config->rand_color)
        color = Color(Rand(config->rand_color_min.r(), config->rand_color_max.r()),
                      Rand(config->rand_color_min.g(), config->rand_color_max.g()),
                      Rand(config->rand_color_min.b(), config->rand_color_max.b()),
                      Rand(config->rand_color_min.a(), config->rand_color_max.a()));
      else if (config->emitter_type & Emitter::RainbowFade) color = Color::fade(config->color_fade);
      else                                                  color = config->color;
      start_color = color;
    }

    void Init() {
      InitColor();
      radius = Rand(config->radius_min, config->radius_max);
      history_len = Trails ? int(Rand(max(3.0f, config->radius_min), float(MaxHistory))) : 1;

      v3 start;
      if (!config->move_with_pos) start = config->pos;
      if (config->pos_transform) {
        const v3 &tf = (*config->pos_transform)[config->pos_transform_index++];
        v3 right = v3::Cross(config->ort, config->updir);
        start.Add(right * tf.x + config->updir * tf.y + config->ort * tf.z);
        if (config->pos_transform_index >= config->pos_transform->size()) config->pos_transform_index = 0;
      }
      start.Add(v3::Rand() * Rand(0.0f, config->rand_initpos));

      for (int i=0; i<history_len; i++) history[i] = start;

      if (config->emitter_type & Emitter::Sprinkler) {
        if (1) vel  = v3(2.0*cos(config->emitter_angle), 2.0,                   2.0*sin(config->emitter_angle));
        if (0) vel += v3(0.5*Rand(1.0,2.0)-.25,          0.5*Rand(1.0,2.0)-.25, 0.5*Rand(1.0,2.0)-.25);
      } else { 
        vel = config->vel*25.0 + v3::Rand()*Rand(0.0f, config->rand_initvel);
      }

      remaining = 1;
      bounceage = 2;
      maxage = Rand(config->age_min, config->age_max);
      dead = false;
      age = 0; 
    }

    void Update(float secs) {
      float bounced = false;
      if (config->gravity) vel += v3(0, config->gravity * secs, 0);
      if (config->floor && history[0].y + vel.y < config->floorval) {
        bounced = true;
        vel.Scale(0.75);
        vel.y *= -0.5f;
      }

      if (config->trails) for (int i=history_len-1; i>0; i--) history[i] = history[i-1];
      history[0] += vel * secs;

      age += secs * (!config->floor ? 1 : (bounced ? bounceage++ : 0.25));
      if (age < maxage) remaining = 1 - age / maxage;
      else dead = true;
    }
  };

  int num_particles, nops=0, texture=0, verts_id=-1, trailverts_id=-1, num_trailverts, emitter_type=0, blend_mode_s=0, blend_mode_t=0, burst=0;
  float floorval=0, gravity=0, radius_min, radius_max, age_min=.05, age_max=1, rand_initpos, rand_initvel, emitter_angle=0, color_fade=0;
  long long ticks_seen=0, ticks_processed=0, ticks_step=0;
  float verts[NumFloats], trailverts[NumTrailFloats];
  bool trails, floor=0, always_on, per_particle_color, radius_decay=1, billboard=0, move_with_pos=0, blend=1, rand_color=0;
  Color rand_color_min, rand_color_max;
  Particle particles[MP], *free_list[MP];
  GraphicsDevice *gd=0;
  Entity *cam=0;

  virtual ~Particles() {}
  Particles(const string &n, bool AlwaysOn=false, float RadiusMin=10, float RadiusMax=40, float RandInitPos=5, float RandInitVel=500) :
    ParticleSystem(n), num_particles(AlwaysOn ? MaxParticles : 0), radius_min(RadiusMin), radius_max(RadiusMax), rand_initpos(RandInitPos),
    rand_initvel(RandInitVel), trails(Trails), always_on(AlwaysOn), per_particle_color(PerParticleColor) {
    for (int i=0; i<MP; i++) {
      Particle *particle = &particles[i];
      particle->dead = true;
      particle->config = this;
      free_list[i] = particle;
      if (always_on) particle->Init();
      float *v = particle_verts(i);
      AssignTex(v, 0, 0); v += VertFloats;
      AssignTex(v, 0, 1); v += VertFloats;
      AssignTex(v, 1, 0); v += VertFloats;
      AssignTex(v, 0, 1); v += VertFloats;
      AssignTex(v, 1, 0); v += VertFloats;
      AssignTex(v, 1, 1); v += VertFloats;
    }
  }

  float       *particle_verts(int n)       { return &verts[n * ParticleVerts * VertFloats]; }
  const float *particle_verts(int n) const { return &verts[n * ParticleVerts * VertFloats]; }

  Particle *AddParticle() {
    if (num_particles == MP) { nops++; return 0; }
    CHECK(num_particles < MP);
    Particle *particle = free_list[num_particles++];
    particle->Init();
    return particle;
  }

  void DelParticle(Particle *particle) {
    CHECK(num_particles > 0);
    free_list[--num_particles] = particle;
  }

  void Update(Entity *C, unsigned dt, int mx, int my, int mdown) {
    cam = C;
    if (!dt) return;
    ticks_seen += dt;
    float secs = dt / 1000.0;

    if (emitter_type & Emitter::Mouse) {
      if (mdown) for(int i=0; i<100; i++) AddParticle();
      v3 mouse_delta = v3(mx - pos.x, my - pos.y, 0);
      vel += (mouse_delta - vel) * 0.25;
    }
    if (emitter_type & Emitter::Sprinkler) {
      emitter_angle += 0.5 * secs;
      while (emitter_angle > M_TAU) emitter_angle -= M_TAU;
    }
    if (emitter_type & Emitter::RainbowFade) {
      color_fade += secs / 10;
      while (color_fade >= 1) color_fade -= 1;
    }
    if (burst) {
      for (int i=0; i<burst; i++) AddParticle();
    }

    pos += vel;
    if (floor && pos.y < floorval) { pos.y = floorval; vel.y = 0; }

    unsigned steps = 0, step = ticks_step ? ticks_step : (ticks_seen - ticks_processed);
    for (/**/; ticks_seen >= ticks_processed + step; ticks_processed += step) steps++;
    if (!steps) return;

    num_trailverts = 0;
    int out_particles = 0;
    float stepsecs = step / 1000.0;
    for (int i=0; i<MP; i++) {
      if (particles[i].dead) continue;
      UpdateParticle(&particles[i], stepsecs, steps, particle_verts(out_particles++), &trailverts[num_trailverts * TrailVertFloats]);
    }
  }

  void UpdateParticle(Particle *particle, float stepsecs, int steps, float *v, float *tv) {
    for (int i=0; i<steps; i++) {
      particle->Update(stepsecs);
      if (particle->dead) {
        if (always_on) particle->Init();
        else return DelParticle(particle);
      }
    }
    UpdateVertices(particle, v, tv);
  }

  void UpdateVertices(Particle *particle, float *v, float *tv) {
    float *vin = v, remaining = particle->remaining, size = particle->radius * (radius_decay ? remaining : 1);
    if (emitter_type & Emitter::GlowFade) particle->color = Color(remaining, remaining * 0.75, 1-remaining, 1.0);
    if (emitter_type & Emitter::FadeFromWhite) particle->color = Color::Interpolate(Color::white, particle->start_color, remaining);

    v3 p = particle->history[0], right, up;
    if (move_with_pos) p.Add(pos);
    if (billboard) { right = v3::Cross(cam->ort, cam->up) * size; up = cam->up * size; }
    else           { right = v3(size, 0, 0);                      up = v3(0, size, 0); }

    v3 o1=p, o2=p, o3=p, o4=p;
    o1.Add(-right + -up);
    o2.Add(-right +  up);
    o3.Add( right + -up);
    o4.Add( right +  up);

    AssignPosColor(v, o1, PerParticleColor ? &particle->color : 0, 2); v += VertFloats;
    AssignPosColor(v, o2, PerParticleColor ? &particle->color : 0, 2); v += VertFloats;
    AssignPosColor(v, o3, PerParticleColor ? &particle->color : 0, 2); v += VertFloats;
    AssignPosColor(v, o2, PerParticleColor ? &particle->color : 0, 2); v += VertFloats;
    AssignPosColor(v, o3, PerParticleColor ? &particle->color : 0, 2); v += VertFloats;
    AssignPosColor(v, o4, PerParticleColor ? &particle->color : 0, 2); v += VertFloats;

    if (trails) {
      v3 last_v1, last_v2, *history = particle->history;
      int history_len = particle->history_len;
      for (int i = 0; i < history_len - 1; i++) {
        float step = 1.0f - i / float(history_len-1);
        v3 dp = history[i] - history[i+1];
        v3 perp1 = v3::Cross(dp, updir);
        v3 perp2 = v3::Cross(dp, perp1);
        perp1 = v3::Cross(dp, perp2);
        perp1.Norm();

        Color trail_color(step, step * 0.25f, 1.0 - step, step * 0.5);
        v3 off = perp1 * (particle->radius * particle->remaining * step * 0.1);
        v3 v1 = history[i] - off, v2 = history[i] + off;
        if (i > 0) {
          AssignPosColor(tv, last_v1, PerParticleColor ? &trail_color : 0, 0); tv += TrailVertFloats;
          AssignPosColor(tv, last_v2, PerParticleColor ? &trail_color : 0, 0); tv += TrailVertFloats;
          AssignPosColor(tv,      v1, PerParticleColor ? &trail_color : 0, 0); tv += TrailVertFloats;
          num_trailverts += 3;
        }
        last_v1 = v1;
        last_v2 = v2;
      }
    }
  }

  void Draw(GraphicsDevice *GD) {
    if (!gd && (gd = GD)) {
      if (!blend_mode_s) blend_mode_s = gd->c.SrcAlpha;
      if (!blend_mode_t) blend_mode_t = gd->c.One;
    }
    gd->DisableDepthTest();
    gd->DisableLighting();
    gd->DisableNormals();
    if (blend) {
      gd->EnableBlend();
      gd->BlendMode(blend_mode_s, blend_mode_t);
    }
    if (PerParticleColor) gd->EnableVertexColor();
    if (texture) {
      gd->EnableTexture();
      gd->BindTexture(gd->c.Texture2D, texture);
    }

    int update_size = verts_id < 0 ? sizeof(verts) : num_particles * ParticleSize;
    DrawParticles(gd->c.Triangles, num_particles*ParticleVerts, verts, update_size);

    if (trails) {
      int trail_update_size = trailverts_id < 0 ? sizeof(trailverts) : num_trailverts * TrailVertSize;
      DrawTrails(trailverts, trail_update_size);
    }

    if (PerParticleColor) gd->DisableVertexColor();
  }

  void DrawParticles(int prim_type, int num_verts, float *v, int l) {
    if (1)                gd->VertexPointer(3, gd->c.Float, VertSize, 0,               v, l, &verts_id, true, prim_type);
    if (1)                gd->TexPointer   (2, gd->c.Float, VertSize, 3*sizeof(float), v, l, &verts_id, false);
    if (PerParticleColor) gd->ColorPointer (4, gd->c.Float, VertSize, 5*sizeof(float), v, l, &verts_id, true);
    gd->DrawArrays(prim_type, 0, num_verts);
  }

  void DrawTrails(float *v, int l) {
    gd->DisableTexture();
    if (1)                gd->VertexPointer(3, gd->c.Float, TrailVertSize, 0,               v, l, &trailverts_id, true, gd->c.Triangles);
    if (PerParticleColor) gd->ColorPointer (4, gd->c.Float, TrailVertSize, 3*sizeof(float), v, l, &trailverts_id, true);
    gd->DrawArrays(gd->c.Triangles, 0, num_trailverts);
  }

  void AssetDrawCB(GraphicsDevice *d, Asset *out, Entity *e) { pos = e->pos; Draw(d); }

  static void AssignTex(float *out, float tx, float ty) { out[3]=tx; out[4]=ty; }
  static void AssignPosColor(float *out, const v3 &v, const Color *c, int tex_size) {
    if (1) { out[0]=v.x; out[1]=v.y; out[2]=v.z; }
    if (c) { int oi=3+tex_size; out[oi++]=c->r(); out[oi++]=c->g(); out[oi++]=c->b(); out[oi++]=c->a(); }
  }
};

}; // namespace LFL
#endif // LFL_CORE_GAME_PARTICLES_H__
