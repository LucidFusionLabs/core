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

#ifndef LFL_CORE_GAME_GAME_H__
#define LFL_CORE_GAME_GAME_H__

#include "core/game/proto.h"
#include "core/game/particles.h"

namespace LFL {
DECLARE_bool(rcon_debug);

struct Game {
  typedef unsigned short EntityID, TeamType;
  struct State { enum { GAME_ON=1, GAME_OVER=2 }; };
  struct ConnectionData;
  struct Team {
    enum { Spectator=1, Home=2, Red=2, Away=3, Blue=3 }; 
    static int Random() { return 2 + Rand<int>(0, 1); }
    static int FromString(const string &s);
    static bool IsTeamID(TeamType n) { return n == Spectator || n == Home || n == Away; }
  };

  struct ReliableNetwork {
    virtual void Heartbeat(Connection *c) {}
    virtual void WriteWithRetry(Connection *c, SerializableProto *req, unsigned short seq) = 0;
  };

  struct Network {
    struct Visitor {
      virtual void Visit(Connection *c, Game::ConnectionData *cd) = 0;
      static void VisitClients(SocketService *svc, Visitor *visitor);
    };

    struct BroadcastVisitor : public Visitor {
      Game::Network *net;
      SerializableProto *msg;
      int sent;
      BroadcastVisitor(Game::Network *n, SerializableProto *m) : net(n), msg(m), sent(0) {}
      virtual void Visit(Connection *c, Game::ConnectionData *cd);
    };

    struct BroadcastWithRetryVisitor : public Visitor {
      Game::Network *net;
      SerializableProto *msg;
      Connection *skip;
      int sent;
      BroadcastWithRetryVisitor(Game::Network *n, SerializableProto *m, Connection *Skip=0) : net(n), msg(m), skip(Skip), sent(0) {}
      virtual void Visit(Connection *c, Game::ConnectionData *cd);
    };

    Network() {}
    virtual int Write(Connection *c, int method, const char *data, int len) = 0;
    virtual void WriteWithRetry(ReliableNetwork*, Connection*, SerializableProto*, unsigned short seq) = 0;

    int Write(Connection *c, int method, unsigned short seq, SerializableProto *msg);
    int Broadcast(SocketService *svc, SerializableProto *msg);
    int BroadcastWithRetry(SocketService *svc, SerializableProto *msg, Connection *skip);
  };

#ifdef LFL_ANDROID
  struct GoogleMultiplayerNetwork : public Network {
    virtual int Write(Connection *c, int method, const char *data, int len);
    virtual void WriteWithRetry(ReliableNetwork *n, Connection *c, SerializableProto *req, unsigned short seq);
  };
#endif

  struct InProcessNetwork : public Network {
    virtual int Write(Connection *c, int method, const char *data, int len);
    virtual void WriteWithRetry(ReliableNetwork *n, Connection *c, SerializableProto *req, unsigned short seq);
  };

  struct TCPNetwork : public Network {
    virtual int Write(Connection *c, int method, const char *buf, int len) { return c->WriteFlush(buf, len); }
    virtual void WriteWithRetry(ReliableNetwork *reliable, Connection *c, SerializableProto *req, unsigned short seq);
  };

  struct UDPNetwork : public Network {
    virtual int Write(Connection *c, int method, const char *buf, int len);
    virtual void WriteWithRetry(ReliableNetwork *reliable, Connection *c, SerializableProto *req, unsigned short seq);
  };

  struct ReliableUDPNetwork : public ReliableNetwork {
    typedef unordered_map<unsigned short, pair<Time, string> > RetryMap;
    Game::Network *net;
    RetryMap retry;
    unsigned method;
    Time timeout;
    ReliableUDPNetwork(unsigned m, unsigned t=500) :
      net(Singleton<Game::UDPNetwork>::Set()), method(m), timeout(t) {}

    void Clear() { retry.clear(); }
    void Acknowledged(unsigned short id) { retry.erase(id); }
    void WriteWithRetry(Connection *c, SerializableProto *req, unsigned short seq);
    void Heartbeat(Connection *c);
  };

  struct ConnectionData : public Connection::Data {
    Game::EntityID entityID=0;
    string playerName;
    unsigned short ping=0, team=0, seq=0, score=0;
    bool rcon_auth=0;
    Game::ReliableUDPNetwork retry;
    ConnectionData() : retry(UDPClient::Sendto) {}
    static void Init(Connection *c) { CHECK_EQ(nullptr, c->data.get()); c->data = make_unique<ConnectionData>(); }
    static ConnectionData *Get(Connection *c) { return reinterpret_cast<ConnectionData*>(c->data.get()); }
  };

  struct Controller {
    unsigned long buttons;
    Controller(int b = 0) : buttons(b) {}

    void Reset()         { buttons = 0; }
    void Set(int index)  { buttons |= (1<<index); }
    void SetUp()         { Set(31); }
    void SetDown()       { Set(30); }
    void SetForward()    { Set(29); }
    void SetBack()       { Set(28); }
    void SetLeft()       { Set(27); }
    void SetRight()      { Set(26); }
    void SetPlayerList() { Set(24); }

    bool Get(int index)  const { return (buttons & (1<<index)) != 0; }
    bool GetUp()         const { return Get(31); }
    bool GetDown()       const { return Get(30); }
    bool GetForward()    const { return Get(29); }
    bool GetBack()       const { return Get(28); }
    bool GetLeft()       const { return Get(27); }
    bool GetRight()      const { return Get(26); }
    bool GetPlayerList() const { return Get(24); }
    v3 Acceleration(v3 ort, v3 u) const;

    static void PrintButtons(unsigned buttons);
  };

  struct Credit {
    string component, credit, link, license;
    Credit(const string &comp, const string &by, const string &url, const string &l) : component(comp), credit(by), link(url), license(l) {}
  };
  typedef vector<Credit> Credits;

  Scene *scene;
  EntityID next_id=0;
  Time started, ranlast;
  bool broadcast_enabled=1;
  int state=State::GAME_ON, teamcount[4], red_score=0, blue_score=0; 
  Game(Scene *s) : scene(s), started(Now()), ranlast(started) { memzeros(teamcount); }

  virtual void Update(GameServer *server, Time timestep) {};
  virtual Entity *JoinEntity(ConnectionData *cd, EntityID, TeamType *team) { int *tc = TeamCount(*team); if (tc) (*tc)++; return 0; }
  virtual void    PartEntity(ConnectionData *cd, Entity *e, TeamType team) { int *tc = TeamCount(team);  if (tc) (*tc)--; }
  virtual bool      JoinRcon(ConnectionData *cd, Entity *e, string *out) { StrAppend(out, "player_entity ", cd->entityID, "\nplayer_team ", cd->team, "\n"); return true; }
  virtual bool    JoinedRcon(ConnectionData *cd, Entity *e, string *out) { StrAppend(out, "print *** ", cd->playerName, " joins"); return true; }
  int *TeamCount(int team) { CHECK(team >= 0 && team < sizeofarray(teamcount)); return &teamcount[team]; }

  EntityID NewID() { do { if (!++next_id) continue; } while (scene->Get(StrCat(next_id))); return next_id; }
  Entity *Add(EntityID id, unique_ptr<Entity> e) { e->SetName(StringPrintf("%05d", id)); return scene->Add(move(e)); }
  Entity *Get(EntityID id) { return scene->Get(StringPrintf("%05d", id)); }
  virtual void Del(EntityID id) { scene->Del(StringPrintf("%05d", id)); }
  static EntityID GetID(const Entity *e) { return atoi(e->name); }
};

struct GameBots {
  struct Bot {
    Entity *entity;
    Game::ConnectionData *player_data;
    Time last_shot=Time(0);
    Bot(Entity *e=0, Game::ConnectionData *pd=0) : entity(e), player_data(pd) {}
    static bool ComparePlayerEntityID(const Bot &l, const Bot &r) { return l.player_data->entityID < r.player_data->entityID; }
  };

  typedef vector<Bot> BotVector;
  Game *world;
  BotVector bots;
  GameBots(Game *w) : world(w) {}

  virtual void Update(Time dt) {}
  virtual void Insert(int num);
  virtual bool RemoveFromTeam(int team);
  virtual void Delete(Bot *b);
  virtual void Clear();
};

struct GameTCPServer : public TCPServer {
  using TCPServer::TCPServer;
};

struct GameUDPServer : public UDPServer {
  int secret1, secret2;
  GameUDPServer(SocketServices *n, int port) : UDPServer(n, port), secret1(rand()), secret2(rand()) {}

  int Hash(Connection *c);
  int UDPFilter(SocketConnection *c, const char *content, int content_len) override;
};

struct GameGPlusServer : public GPlusServer {
  using GPlusServer::GPlusServer;
};

struct GameInProcessServer : public InProcessServer {
  using InProcessServer::InProcessServer;
};

struct GameServer : public Connection::Handler {
  struct History {
    GameProtocol::WorldUpdate WorldUpdate;
    struct WorldUpdateHistory { unsigned short id; Time time; } send_WorldUpdate[3];
    int send_WorldUpdate_index=0, num_send_WorldUpdate=0;
    Time time_post_MasterUpdate=Time(0);
    History() { WorldUpdate.id=0; memzeros(send_WorldUpdate); }
  };

  ApplicationShutdown *shutdown;
  SocketServices *net;
  Game *world;
  GameBots *bots=0;
  Time timestep;
  int proto=0;
  vector<SocketService*> svc;
  unique_ptr<GameTCPServer> tcp_transport;
  unique_ptr<GameUDPServer> udp_transport;
  unique_ptr<GPlusServer> gplus_transport;
  unique_ptr<InProcessServer> inprocess_transport;
  string rcon_auth_passwd, master_sink_url, local_game_name, local_game_url;
  const vector<Asset> *assets;
  History last;

  GameServer(ApplicationShutdown *s, SocketServices *n, Game *w, unsigned ts, const string &name, const vector<Asset> *a) :
    shutdown(s), net(n), world(w), timestep(ts), local_game_name(name), assets(a) {}

  void InitTransport(SocketServices *net, int p, int port);
  int Connected(Connection *c) { Game::ConnectionData::Init(c); return 0; }
  void Close(Connection *c);
  int Read(Connection *c);
  int Read(Connection *c, const char *content, int content_len);
  void Write(Connection *c, int method, unsigned short seq, SerializableProto *msg);
  void WriteWithRetry(Connection *c, Game::ConnectionData *cd, SerializableProto *msg);
  void WritePrintWithRetry(Connection *c, Game::ConnectionData *cd, const string &text);
  int BroadcastWithRetry(SerializableProto *msg, Connection *skip=0);
  int BroadcastPrintWithRetry(const string &text, Connection *skip = 0);
  int Frame();
  void RconResponseCB(Connection *c, GameProtocol::Header *hdr, GameProtocol::RconResponse *req);
  void JoinRequestCB(Connection *c, GameProtocol::Header *hdr, GameProtocol::JoinRequest *req);
  void PlayerUpdateCB(Connection *c, GameProtocol::Header *hdr, GameProtocol::PlayerUpdate *pup);
  void RconRequestCB(Connection *c, GameProtocol::Header *hdr, GameProtocol::RconRequest *rcon);
  virtual void RconRequestCB(Connection *c, Game::ConnectionData *cd, const string &cmd, const string &arg);
  static void AppendSerializedPlayerData(const Game::ConnectionData *icd, string *out);
};

struct GameClient {
  struct History {
    unsigned buttons=0;
    Time     time_frame=Time(0);
    Time     time_send_PlayerUpdate=Time(0);
    Time     time_recv_WorldUpdate[2];
    unsigned short id_WorldUpdate, seq_WorldUpdate;
    deque<GameProtocol::WorldUpdate> WorldUpdate;
    History() { time_recv_WorldUpdate[0]=time_recv_WorldUpdate[1]=Time(0); }
  };
  struct Replay {
    Time start=Time(0);
    unsigned short start_ind=0, while_seq=0;
    bool just_ended=0;
    bool enabled() { return start_ind; }
    void disable() { just_ended=start_ind; start_ind=0; }
  };

  AssetStore *assetstore;
  SocketServices *sockets;
  string playername;
  Connection *conn=0;
  Game *world;
  unsigned short seq=0, entity_id=0, team=0, cam_id=1;
  bool reorienting=1;
  Game::Network *net=0;
  InProcessServer *inprocess_server=0;
  Game::ReliableUDPNetwork retry;
  map<unsigned, string> assets;
  Game::Controller control;
  History last;
  Replay replay, gameover;
  View *playerlist;
  TextArea *chat;

  ~GameClient() { Reset(); }
  GameClient(AssetStore *a, SocketServices *n, Game *w, View *PlayerList, TextArea *Chat) :
    assetstore(a), sockets(n), world(w), retry(UDPClient::Write), playerlist(PlayerList), chat(Chat) {}

  virtual void NewEntityCB(Entity *) {}
  virtual void DelEntityCB(Entity *) {}
  virtual void SetEntityCB(Entity *, const string &k, const string &v) {}
  virtual void AnimationChange(Entity *, int NewID, int NewSeq) {}
  virtual void RconRequestCB(const string &cmd, const string &arg, int seq) {}

  Entity *WorldAddEntity(int id) { return world->Add(id, make_unique<Entity>()); }
  void WorldAddEntityFinish(Entity *e, int type);
  void WorldDeleteEntity(Scene::EntityVector &v);
  void Rcon(const string &text);

  void MoveUp    (unsigned t) { control.SetUp();      }
  void MoveDown  (unsigned t) { control.SetDown();    }
  void MoveFwd   (unsigned t) { control.SetForward(); }
  void MoveRev   (unsigned t) { control.SetBack();    }
  void MoveLeft  (unsigned t) { control.SetLeft();    }
  void MoveRight (unsigned t) { control.SetRight();   }
  void SetCamera(const vector<string> &a) { cam_id = atoi(a.size() ? a[0] : ""); }
  void RconCmd  (const vector<string> &a) { if (a.size()) Rcon(a[0]); }
  void SetTeam  (const vector<string> &a) { if (a.size()) Rcon(StrCat("team ", a[0])); }
  void SetName  (const vector<string> &a);

  void MyEntityName(vector<string>);
  void Reset();
  
  bool Connected() { return conn && conn->state == Connection::Connected; }
  int Connect(int prot, const string &url, int default_port);
  void DatagramRead(Connection *c, const char *content, int len) { Read(c, content, len); }
  void StreamRead(Connection *c) { c->ReadFlush(max(0, Read(c, c->rb.begin(), c->rb.size()))); }
  int Read(Connection *c, const char *content, int content_length);

  void Frame();
  void UpdateWorld();
  void ChallengeResponseCB(Connection *c, GameProtocol::Header *hdr, GameProtocol::ChallengeResponse *challenge);
  void JoinResponseCB(Connection *c, GameProtocol::Header *hdr, GameProtocol::JoinResponse *joined);
  void WorldUpdateCB(Connection *c, GameProtocol::Header *hdr, GameProtocol::WorldUpdate *wu);
  void PlayerListCB(Connection *c, GameProtocol::Header *hdr, GameProtocol::PlayerList *pl) { playerlist->HandleTextMessage(pl->Text); }
  void RconResponseCB(Connection *c, GameProtocol::Header *hdr, GameProtocol::RconResponse*) { retry.Acknowledged(hdr->seq); }
  void RconRequestCB(Connection *c, GameProtocol::Header *hdr, GameProtocol::RconRequest *rcon);
  void RconRequestCB(Connection *c, GameProtocol::Header *hdr, const string &rcon);
  void Me(Entity *e, int cam_id, bool assign_entity);
};

struct GameSettings {
  typedef CategoricalVariable<const char *> Value;
  struct Setting {
    string key; Value *value;
    Setting(const string &k, Value *v) : key(k), value(v) {}
  };
  typedef vector<Setting> Vector;

  Vector vec;
  GameSettings() {}

  void SetIndex(int n, int v)  { CHECK(n < vec.size()); vec[n].value->ind = v; };
  int GetIndex(int n)    const { CHECK(n < vec.size()); return vec[n].value->ind; }
  const char *Get(int n) const { CHECK(n < vec.size()); return vec[n].value->Cur(); }
};

struct GameMultiTouchControls {
  enum { LEFT=5, RIGHT=6, UP=7, DOWN=2, UP_LEFT=8, UP_RIGHT=9, DOWN_LEFT=3, DOWN_RIGHT=4 };
  Window *parent;
  GameClient *client;
  Font *dpad_font;
  Box lpad_win, rpad_win;
  int lpad_tbx, lpad_tby, rpad_tbx, rpad_tby, lpad_down=0, rpad_down=0;
  float dp0_x=0, dp0_y=0, dp1_x=0, dp1_y=0;
  bool swipe_controls=0;

  GameMultiTouchControls(Window *W, GameClient *C) : parent(W), client(C),
  dpad_font(W->parent->fonts->Get(W->gl_h, "dpad_atlas", "", 0, Color::white)),
  lpad_win(W->Box(.03, .05, .2, .2)),
  rpad_win(W->Box(.78, .05, .2, .2)),
  lpad_tbx(RoundF(lpad_win.w * .6)), lpad_tby(RoundF(lpad_win.h *.6)),
  rpad_tbx(RoundF(rpad_win.w * .6)), rpad_tby(RoundF(rpad_win.h *.6)) {}

  void Draw(GraphicsDevice *gd) {
    dpad_font->Select(gd);
    dpad_font->DrawGlyph(gd, lpad_down, lpad_win);
    dpad_font->DrawGlyph(gd, rpad_down, rpad_win);
  }

  void Update(unsigned clicks) {
    Entity *cam = &client->world->scene->cam;
#if 0
    if (swipe_controls) {
      if (parent->gesture_dpad_stop[0]) dp0_x = dp0_y = 0;
      else if (parent->gesture_dpad_dx[0] || parent->gesture_dpad_dy[0]) {
        dp0_x = dp0_y = 0;
        if (fabs(parent->gesture_dpad_dx[0]) > fabs(parent->gesture_dpad_dy[0])) dp0_x = parent->gesture_dpad_dx[0];
        else                                                                     dp0_y = parent->gesture_dpad_dy[0];
      }

      if (parent->gesture_dpad_stop[1]) dp1_x = dp1_y = 0;
      else if (parent->gesture_dpad_dx[1] || parent->gesture_dpad_dy[1]) {
        dp1_x = dp1_y = 0;
        if (fabs(parent->gesture_dpad_dx[1]) > fabs(parent->gesture_dpad_dy[1])) dp1_x = parent->gesture_dpad_dx[1];
        else                                                                     dp1_y = parent->gesture_dpad_dy[1];
      }

      lpad_down = rpad_down = 0;
      if (FLAGS_swap_axis) {
        if (dp0_y > 0) { lpad_down = LEFT;    client->MoveLeft     (clicks); }
        if (dp0_y < 0) { lpad_down = RIGHT;   client->MoveRight    (clicks); }
        if (dp0_x < 0) { lpad_down = UP;      client->MoveFwd      (clicks); }
        if (dp0_x > 0) { lpad_down = DOWN;    client->MoveRev      (clicks); }
        if (dp1_y > 0) { rpad_down = LEFT;    cam->YawLeft (clicks); }
        if (dp1_y < 0) { rpad_down = RIGHT;   cam->YawRight(clicks); }
        if (dp1_x < 0) { rpad_down = UP;   /* cam->MoveUp  (clicks); */ }
        if (dp1_x > 0) { rpad_down = DOWN; /* cam->MoveDown(clicks); */ }
      } else {
        if (dp1_x < 0) { lpad_down = LEFT;    client->MoveLeft     (clicks); }
        if (dp1_x > 0) { lpad_down = RIGHT;   client->MoveRight    (clicks); }
        if (dp1_y < 0) { lpad_down = UP;      client->MoveFwd      (clicks); }
        if (dp1_y > 0) { lpad_down = DOWN;    client->MoveRev      (clicks); }
        if (dp0_x < 0) { rpad_down = LEFT;    cam->YawLeft (clicks); } 
        if (dp0_x > 0) { rpad_down = RIGHT;   cam->YawRight(clicks); }
        if (dp0_y < 0) { rpad_down = UP;   /* cam->MoveUp  (clicks); */ }
        if (dp0_y > 0) { rpad_down = DOWN; /* cam->MoveDown(clicks); */ }
      }
    } else {
      point l, r;
      if (FLAGS_swap_axis) {
        l.x=int(parent->gesture_dpad_x[0]); l.y=int(parent->gesture_dpad_y[0]);
        r.x=int(parent->gesture_dpad_x[1]); r.y=int(parent->gesture_dpad_y[1]);
      } else {
        r.x=int(parent->gesture_dpad_x[0]); r.y=int(parent->gesture_dpad_y[0]);
        l.x=int(parent->gesture_dpad_x[1]); l.y=int(parent->gesture_dpad_y[1]);
      }
      l = Input::TransformMouseCoordinate(l);
      r = Input::TransformMouseCoordinate(r);
#if 0
      INFOf("l(%d, %d) lw(%f, %f, %f, %f) r(%d, %d) rw(%f, %f, %f, %f)", 
            l.x, l.y, lpad_win.x, lpad_win.y, lpad_win.w, lpad_win.h,
            r.x, r.y, rpad_win.x, rpad_win.y, rpad_win.w, rpad_win.h);
#endif
      lpad_down = rpad_down = 0;

      if (l.x >= lpad_win.x - lpad_tbx && l.x <= lpad_win.x + lpad_win.w + lpad_tbx &&
          l.y >= lpad_win.y - lpad_tby && l.y <= lpad_win.y + lpad_win.h + lpad_tby)
      {
        static const float dd = 13/360.0*M_TAU;
        int clx = l.x - lpad_win.centerX(), cly = l.y - lpad_win.centerY();
        float a = atan2f(cly, clx);
        if (a < 0) a += M_TAU;

        if      (a < M_TAU*1/8.0 - dd) { lpad_down = RIGHT;      client->MoveRight(clicks); }
        if      (a < M_TAU*1/8.0 + dd) { lpad_down = UP_RIGHT;   client->MoveRight(clicks); client->MoveFwd(clicks); }
        else if (a < M_TAU*3/8.0 - dd) { lpad_down = UP;         client->MoveFwd  (clicks); }
        else if (a < M_TAU*3/8.0 + dd) { lpad_down = UP_LEFT;    client->MoveFwd  (clicks); client->MoveLeft(clicks); }
        else if (a < M_TAU*5/8.0 - dd) { lpad_down = LEFT;       client->MoveLeft (clicks); }
        else if (a < M_TAU*5/8.0 + dd) { lpad_down = DOWN_LEFT;  client->MoveLeft (clicks); client->MoveRev(clicks); }
        else if (a < M_TAU*7/8.0 - dd) { lpad_down = DOWN;       client->MoveRev  (clicks); }
        else if (a < M_TAU*7/8.0 + dd) { lpad_down = DOWN_RIGHT; client->MoveRev  (clicks); client->MoveRight(clicks); }
        else                           { lpad_down = RIGHT;      client->MoveRight(clicks); }
      }

      if (r.x >= rpad_win.x - rpad_tbx && r.x <= rpad_win.x + rpad_win.w + rpad_tbx &&
          r.y >= rpad_win.y - rpad_tby && r.y <= rpad_win.y + rpad_win.h + rpad_tby) {
        bool g1 = (rpad_win.w * (r.y - (rpad_win.y))              - rpad_win.h * (r.x - rpad_win.x)) > 0; 
        bool g2 = (rpad_win.w * (r.y - (rpad_win.y + rpad_win.h)) + rpad_win.h * (r.x - rpad_win.x)) > 0; 

        if      ( g1 && !g2) { rpad_down = LEFT;  cam->YawLeft (clicks); }
        else if (!g1 &&  g2) { rpad_down = RIGHT; cam->YawRight(clicks); }
        else if ( g1 &&  g2) { rpad_down = UP;    }
        else if (!g1 && !g2) { rpad_down = DOWN;  }
      }
    }
#endif
  }
};

}; // namespace LFL
#include "core/game/game_menu.h"
#include "core/game/physics.h"

#endif /* LFL_CORE_GAME_GAME_H__ */
