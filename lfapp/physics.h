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

#ifndef LFL_LFAPP_PHYSICS_H__
#define LFL_LFAPP_PHYSICS_H__
namespace LFL {

struct Physics {
  struct CollidesWith {
    unsigned self, collides;
    CollidesWith(unsigned S=0, unsigned C=0) : self(S), collides(C) {}
  };
  struct Contact { v3 p1, p2, n2; };
  typedef function<void(const Entity*, const Entity*, int, Contact*)> CollidedCB;

  virtual void SetGravity(const v3 &gravity) = 0;
  virtual void *AddSphere(float radius, const v3 &pos, const v3 &ort, float mass, const CollidesWith &cw) = 0;
  virtual void *AddBox(const v3 &half_ext, const v3 &pos, const v3 &ort, float mass, const CollidesWith &cw) = 0;
  virtual void *AddPlane(const v3 &normal, const v3 &pos, const CollidesWith &cw) = 0;
  virtual void Free(void *body) = 0;

  virtual void Input(const Entity *e, Time timestep, bool angular) = 0;
  virtual void Output(Entity *e, Time timestep) = 0;
  virtual void SetPosition(Entity *e, const v3 &pos, const v3 &ort) = 0;
  virtual void SetContinuous(Entity *e, float threshhold, float sweepradius) {}

  virtual void Collided(bool contact_pts, CollidedCB cb) = 0;
  virtual void Update(Time timestep) = 0;
};

struct SimplePhysics : public Physics {
  Scene *scene;
  SimplePhysics(Scene *s) : scene(s) { INFO("SimplePhysics"); }

  virtual void SetGravity(const v3 &gravity) {}
  virtual void *AddSphere(float radius, const v3 &pos, const v3 &ort, float mass, const CollidesWith &cw) { return 0; }
  virtual void *AddBox(const v3 &half_ext, const v3 &pos, const v3 &ort, float mass, const CollidesWith &cw) { return 0; }
  virtual void *AddPlane(const v3 &normal, const v3 &pos, const CollidesWith &cw) { return 0; }
  virtual void Free(void *) {}

  virtual void Input(const Entity *e, Time timestep, bool angular) {}
  virtual void Output(Entity *e, Time timestep) {}
  virtual void SetPosition(Entity *e, const v3 &pos, const v3 &ort) {}
  virtual void Collided(bool contact_pts, CollidedCB cb) {}

  virtual void Update(Time timestep) {
    for (auto const &a : scene->asset)
      for (auto e : a.second)
        Update(e, timestep);
  }

  static void Update(Entity *e, Time timestep) { e->pos.Add(e->vel * (timestep.count() / 1000.0)); }
};
}; // namespace LFL

#ifdef LFL_BOX2D
#include "Box2D/Box2D.h"
namespace LFL {
struct Box2DScene : public Physics {
  Scene *scene;
  b2World world;
  double groundY=-INFINITY;
  Box2DScene(Scene *s, b2Vec2 gravity) : scene(s), world(gravity) { INFO("Box2DPhysics"); }

  virtual void Free(void *) {}
  virtual void SetGravity(const v3 &gravity) {}
  b2Body *Add(const v3 &pos, const v3 &ort) {
    b2BodyDef bodyDef;
    bodyDef.type = b2_dynamicBody;
    bodyDef.position.Set(pos.x, pos.z);
    bodyDef.angle = GetAngle(ort);
    return world.CreateBody(&bodyDef);
  }

  void AddFixture(b2Body *body, b2Shape *shape, float density, float friction) {
    b2FixtureDef fixtureDef;
    fixtureDef.shape = shape;
    fixtureDef.density = density;
    fixtureDef.friction = friction;
    body->CreateFixture(&fixtureDef);
  }

  virtual void *AddSphere(float radius, const v3 &pos, const v3 &ort, float mass, const CollidesWith &cw) {
    b2Body *body = Add(pos, ort);
    b2CircleShape circle;
    circle.m_radius = radius;
    AddFixture(body, &circle, 1.0, 0.3);
    body->SetBullet(true);
    return body;
  }

  virtual void *AddBox(const v3 &half_ext, const v3 &pos, const v3 &ort, float mass, const CollidesWith &cw) {
    b2Body *body = Add(pos, ort);
    b2PolygonShape dynamicBox;
    dynamicBox.SetAsBox(half_ext.x, half_ext.z);
    AddFixture(body, &dynamicBox, 1.0, 0.3);
    return body;
  }

  virtual void *AddPlane(const v3 &normal, const v3 &pos, const CollidesWith &cw) { groundY = pos.y; return 0; }
  virtual void Input(const Entity *e, Time timestep, bool angular) {
    if (!e || !e->body) return;
    b2Body *body = (b2Body*)e->body;
    body->SetUserData((void*)e);
    body->SetLinearVelocity(b2Vec2(e->vel.x, e->vel.z));

    if (!angular) return;
    float angle = GetAngle(e->ort);
    if (0) {
      body->SetTransform(body->GetPosition(), angle);
      body->SetAngularVelocity(0);
    } else {
      float delta = angle - body->GetAngle();
      while (delta < -M_PI) delta += M_TAU;
      while (delta >  M_PI) delta -= M_TAU;
      body->SetAngularVelocity(delta * 1000.0 / timestep.count());
    }
  }

  virtual void Output(Entity *e, Time timestep) {
    if (!e || !e->body) return;
    b2Body *body = (b2Body*)e->body;
    b2Vec2 position = body->GetPosition(), velocity = body->GetLinearVelocity(), orientation = body->GetWorldVector(b2Vec2(0,1));;
    e->vel = v3(velocity.x, e->vel.y, velocity.y);
    if (e->vel.y) {
      e->pos.y += e->vel.y / timestep.count();
      if (e->pos.y < groundY) {
        e->pos.y = groundY;
        e->vel.y = 0;
      }
    }
    e->pos = v3(position.x, e->pos.y, position.y);
    e->ort = v3(orientation.x, 0, orientation.y);        
    e->up = v3(0, 1, 0);
  }

  virtual void SetPosition(Entity *e, const v3 &pos, const v3 &ort) {
    b2Body *body = (b2Body*)e->body;
    body->SetTransform(b2Vec2(pos.x, pos.z), GetAngle(ort));
  }

  virtual void Collided(bool contact_pts, CollidedCB cb) {
    for (b2Contact* c = world.GetContactList(); c; c = c->GetNext()) {
      b2Fixture *fixtA = c->GetFixtureA(), *fixtB = c->GetFixtureB();
      b2Body *bodyA = fixtA->GetBody(), *bodyB = fixtB->GetBody();
      const Entity *eA = (Entity*)bodyA->GetUserData(), *eB = (Entity*)bodyB->GetUserData();
      if (!eA || !eB) continue;
      if (1) /*(!contact_pts)*/ { cb(eA, eB, 0, 0); continue; }
    }
  }

  virtual void Update(Time timestep) {
    static int velocityIterations = 6, positionIterations = 2;
    world.Step(timestep.count()/1000.0, velocityIterations, positionIterations);
  }

  static float GetAngle(const v3 &ort) { return atan2(ort.z, ort.x) - M_PI/2; }
};
}; // namespace LFL
#endif

#ifdef LFL_BULLET
#include "btBulletDynamicsCommon.h"
namespace LFL {
struct BulletScene : public Physics {
  btBroadphaseInterface *broadphase;
  btDefaultCollisionConfiguration *collisionConfiguration;
  btCollisionDispatcher *dispatcher;
  btSequentialImpulseConstraintSolver *solver;
  btDiscreteDynamicsWorld *dynamicsWorld;
#define btConstruct btRigidBody::btRigidBodyConstructionInfo

  ~BulletScene() { delete dynamicsWorld; delete solver; delete dispatcher; delete collisionConfiguration; delete broadphase; }
  BulletScene(v3 *gravity=0) { 
    INFO("BulletPhysics");
    broadphase = new btDbvtBroadphase();
    collisionConfiguration = new btDefaultCollisionConfiguration();
    dispatcher = new btCollisionDispatcher(collisionConfiguration);
    solver = new btSequentialImpulseConstraintSolver;

    dynamicsWorld = new btDiscreteDynamicsWorld(dispatcher, broadphase, solver, collisionConfiguration);
    if (gravity) SetGravity(*gravity);
    else SetGravity(v3(0,0,0));
  }

  void Free(void *) {}
  void SetGravity(const v3 &gravity) { dynamicsWorld->setGravity(btVector3(gravity.x, gravity.y, gravity.z)); }
  void *Add(btRigidBody *rigidBody, const CollidesWith &cw) {
    dynamicsWorld->addRigidBody(rigidBody, cw.self, cw.collides);
    return rigidBody;
  }

  void *Add(btCollisionShape *shape, v3 pos, float mass, const CollidesWith &cw) {
    btVector3 inertia(0,0,0);
    if (mass) shape->calculateLocalInertia(mass, inertia);
    btRigidBody *body = new btRigidBody(btConstruct(mass, MotionState(pos), shape, inertia));
    body->setActivationState(DISABLE_DEACTIVATION);
    return Add(body, cw);
  }

  void *AddSphere(float radius, const v3 &pos, const v3 &ort, float mass, const CollidesWith &cw) { return Add(new btSphereShape(radius), pos, mass, cw); }
  void *AddBox(const v3 &half_ext, const v3 &pos, const v3 &ort, float mass, const CollidesWith &cw) { return Add(new btBoxShape(Vector(half_ext)), pos, mass, cw); }
  void *AddPlane(const v3 &normal, const v3 &pos, const CollidesWith &cw) { return Add(new btStaticPlaneShape(Vector(normal), -Plane(pos, normal).d), v3(0,0,0), 0, cw); }

  void Input(const Entity *e, Time timestep, bool angular) {
    btRigidBody *body = (btRigidBody*)e->body;
    body->setUserPointer((void*)e);

    // linear
    body->setLinearVelocity(btVector3(e->vel.x, e->vel.y, e->vel.z));

    // angular
    if (!angular) return;

    v3 right = v3::cross(e->ort, e->up);
    right.norm();

    btTransform src, dst;
    body->getMotionState()->getWorldTransform(src);

    btMatrix3x3 basis;
    basis.setValue(right.x,e->ort.x,e->up.x,
                   right.y,e->ort.y,e->up.y,
                   right.z,e->ort.z,e->up.z);
    dst = src;
    dst.setBasis(basis);

    btVector3 vel, avel;
    btTransformUtil::calculateVelocity(src, dst, timestep.count()/1000.0, vel, avel);

    body->setAngularVelocity(avel);
  }

  void Output(Entity *e, Time timestep) {
    btRigidBody *body = (btRigidBody*)e->body;
    btVector3 vel = body->getLinearVelocity();
    e->vel = v3(vel.getX(), vel.getY(), vel.getZ());

    btTransform trans;
    body->getMotionState()->getWorldTransform(trans);
    e->pos = v3(trans.getOrigin().getX(), trans.getOrigin().getY(), trans.getOrigin().getZ());

    btMatrix3x3 basis = trans.getBasis();
    btVector3 ort = basis.getColumn(1), up = basis.getColumn(2);

    e->ort = v3(ort[0], ort[1], ort[2]);
    e->up = v3(up[0], up[1], up[2]);
  }

  void SetPosition(Entity *e, const v3 &pos, const v3 &ort) {
    btRigidBody *body = (btRigidBody*)e->body;
    btTransform trans = body->getCenterOfMassTransform();
    trans.setOrigin(btVector3(pos.x, pos.y, pos.z));
    body->setCenterOfMassTransform(trans);
  }

  void SetContinuous(Entity *e, float threshhold, float sweepradius) {
    btRigidBody *body = (btRigidBody*)e->body;
    body->setCcdMotionThreshold(threshhold);
    body->setCcdSweptSphereRadius(sweepradius);
  }

  void Update(Time timestep) { dynamicsWorld->stepSimulation(timestep.count()/1000.0, 1000, 1/180.0); }
  void Collided(bool contact_pts, CollidedCB cb) {
    int numManifolds = dynamicsWorld->getDispatcher()->getNumManifolds();
    for (int i=0;i<numManifolds;i++) {
      btPersistentManifold *contactManifold = dynamicsWorld->getDispatcher()->getManifoldByIndexInternal(i);
      btCollisionObject *obA = (btCollisionObject*)contactManifold->getBody0();
      btCollisionObject *obB = (btCollisionObject*)contactManifold->getBody1();
      const Entity *eA = (Entity*)obA->getUserPointer(), *eB = (Entity*)obB->getUserPointer();
      if (!eA || !eB) continue;
      if (!contact_pts) { cb(eA, eB, 0, 0); continue; }

      vector<Physics::Contact> contacts;
      int numContacts = contactManifold->getNumContacts();
      for (int j=0; j<numContacts; j++) {
        btManifoldPoint& pt = contactManifold->getContactPoint(j);
        if (pt.getDistance()< 0.f) {
          const btVector3& ptA = pt.getPositionWorldOnA();
          const btVector3& ptB = pt.getPositionWorldOnB();
          const btVector3& normalOnB = pt.m_normalWorldOnB;
          Physics::Contact contact = { v3(ptA[0], ptA[1], ptA[2]), v3(ptB[0], ptB[1], ptB[2]), v3(normalOnB[0], normalOnB[1], normalOnB[2]) };
          contacts.push_back(contact);
        }
      }
      if (contacts.size()) cb(eA, eB, contacts.size(), &contacts[0]);
    }
  }

  static btVector3 Vector(const v3 &x) { return btVector3(x.x, x.y, x.z); }
  static btDefaultMotionState *MotionState(const v3 &pos) {
    return new btDefaultMotionState(btTransform(btQuaternion(0,0,0,1), btVector3(pos.x, pos.y, pos.z)));
  }
};
}; // namespace LFL
#endif /* LFL_BULLET */

#ifdef LFL_ODE
#include "ode/ode.h"
namespace LFL {
struct ODEScene : public Physics {
  dWorldID world;
  dSpaceID space;
  dJointGroupID contactgroup;
  ODEScene() {
    ODEInit();
    INFO("ODEPhysics");
    world = dWorldCreate();
    space = dSimpleSpaceCreate(0);
    contactgroup = dJointGroupCreate(0);
    SetGravity(v3(0,0,0));
  }

  virtual ~ODEScene() {
    dJointGroupDestroy(contactgroup);
    dSpaceDestroy(space);
    dWorldDestroy(world);
  }

  virtual void Free(void *b) {
    dGeomID geom = (dGeomID)b;
    dGeomDestroy(geom);
  }

  virtual void SetGravity(const v3 &gravity) { dWorldSetGravity(world, gravity.x, gravity.y, gravity.z); }
  virtual void *AddPlane(const v3 &normal, const v3 &pos, const CollidesWith &cw) {
    Plane plane(pos, normal);
    dGeomID geom = dCreatePlane(space, plane.a, plane.b, plane.c, -plane.d);
    dGeomSetCategoryBits(geom, cw.self);
    dGeomSetCollideBits(geom, cw.collides);
    return geom;
  }

  virtual void *AddBox(const v3 &half_ext, const v3 &pos, const v3 &ort, float mass, const CollidesWith &cw) {
    return NewObject(dBodyCreate(world), pos, cw,
                     dCreateBox(space, half_ext.x*2, half_ext.y*2, half_ext.z*2),
                     MassBox(mass, 1, half_ext.x*2, half_ext.y*2, half_ext.z*2));
  }

  virtual void *AddSphere(float radius, const v3 &pos, const v3 &ort, float mass, const CollidesWith &cw) {
    return NewObject(dBodyCreate(world), pos, cw, dCreateSphere(space, radius), MassSphere(mass, 1, radius));
  }

  virtual void Input(const Entity *e, Time timestep, bool angular) {
    if (!e->body) return;
    dGeomID geom = (dGeomID)e->body;
    dBodyID body = dGeomGetBody(geom);

    // linear
    dBodySetLinearVel(body, e->vel.x, e->vel.y, e->vel.z);

    // angular
    if (!angular) return;

    v3 right = v3::cross(e->ort, e->up);
    right.norm();

    dMatrix3 r = {
      e->up.x,   e->up.y,   e->up.z,  0,
      right.x,   right.y,   right.z,  0,
      e->ort.x,  e->ort.y,  e->ort.z, 0
    };
    dGeomSetRotation(geom, r);
  }

  virtual void Output(Entity *e, Time timestep) {
    if (!e->body) return;
    dGeomID geom = (dGeomID)e->body;
    dBodyID body = dGeomGetBody(geom);

    const dReal *pos = dGeomGetPosition(geom);
    e->pos = v3(pos[0], pos[1], pos[2]);

    const dReal *rot = dGeomGetRotation(geom);
    e->up = v3(rot[0], rot[1], rot[2]);
    e->ort = v3(rot[8], rot[9], rot[10]);

    const dReal *vel = dBodyGetLinearVel(body);
    e->vel = v3(vel[0], vel[1], vel[2]);
  }

  virtual void SetPosition(Entity *e, const v3 &pos, const v3 &ort) {
    if (!e->body) return;
    dGeomID geom = (dGeomID)e->body;
    dGeomSetPosition(geom, pos.x, pos.y, pos.z);
  }

  virtual void Update(Time timestep) {
    dSpaceCollide(space, this, nearCallback);

    dWorldStep(world, timestep.count()/1000.0);
    // dWorldQuickStep(world, timestep.count()/1000.0);

    dJointGroupEmpty(contactgroup);
  }

  virtual void Collide(dGeomID o1, dGeomID o2, dContact *contact) {
    dBodyID b1 = dGeomGetBody(o1), b2 = dGeomGetBody(o2);
    dJointID j = dJointCreateContact(world, contactgroup, contact);
    dJointAttach(j, b1, b2);
  }

  static void ODEInit() { dInitODE2(0); dAllocateODEDataForThread(dAllocateMaskAll); }
  static void ODEFree() { dCloseODE(); }

  static dMass MassBox(float mass, float density, float lx, float ly, float lz) {
    dMass m;
    dMassSetBox(&m, density, lx, ly, lz);
    dMassAdjust(&m, mass);
    return m;
  }

  static dMass MassSphere(float mass, float density, float radius) {
    dMass m;
    dMassSetSphere(&m, density, radius);
    dMassAdjust(&m, mass);
    return m;
  }

  static dGeomID NewObject(dBodyID body, v3 pos, CollidesWith cw, dGeomID geom, dMass mass) {
    dBodySetMass(body, &mass);
    dGeomSetBody(geom, body);
    dGeomSetPosition(geom, pos.x, pos.y, pos.z);
    dGeomSetCategoryBits(geom, cw.self);
    dGeomSetCollideBits(geom, cw.collides);
    return geom;
  }

  static void nearCallback(void *opaque, dGeomID o1, dGeomID o2) {
    if (dGeomIsSpace(o1) || dGeomIsSpace(o2)) {
      // colliding a space with something
      dSpaceCollide2(o1, o2, opaque, nearCallback);

      // collide all geoms internal to the space(s)
      if (dGeomIsSpace(o1)) dSpaceCollide((dSpaceID)o1, opaque, nearCallback);
      if (dGeomIsSpace(o2)) dSpaceCollide((dSpaceID)o2, opaque, nearCallback);
    }
    else {
      // colliding two non-space geoms, so generate contact
      // points between o1 and o2
      static const int max_contacts = 1;
      dContact contact[max_contacts];
      for (int i = 0; i < max_contacts; i++) {
        contact[i].surface.mode = dContactBounce | dContactSoftCFM;
        contact[i].surface.mu = dInfinity;
        contact[i].surface.mu2 = 0;
        contact[i].surface.bounce = 0.01;
        contact[i].surface.bounce_vel = 0.1;
        contact[i].surface.soft_cfm = 0.01;
      }

      int num_contact = dCollide(o1, o2, max_contacts, &contact[0].geom, sizeof(dContact));
      for (int i = 0; i < num_contact; i++) // add these contact points to the simulation
        ((ODEScene*)opaque)->Collide(o1, o2, &contact[i]);
    }
  }
};
}; // namespace LFL
#endif /* LFL_ODE */

#endif /* LFL_LFAPP_PHYSICS_H__ */
