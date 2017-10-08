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

  void InitTransport(SocketServices *net, int p, int port) {
    proto = p;
    if (proto == Protocol::TCP) {
      if (!tcp_transport) {
        local_game_url = StrCat(port);
        tcp_transport = make_unique<GameTCPServer>(net, port);
        tcp_transport->game_network = Singleton<Game::TCPNetwork>::Set();
        tcp_transport->handler = this;
        svc.push_back(tcp_transport.get());
      }
    } else if (proto == Protocol::UDP) {
      if (!udp_transport) {
        local_game_url = StrCat(port);
        udp_transport = make_unique<GameUDPServer>(net, port);
        udp_transport->game_network = Singleton<Game::UDPNetwork>::Set();
        udp_transport->handler = this;
        svc.push_back(udp_transport.get());
      }
#ifdef LFL_ANDROID
      if (!gplus_transport) {
        gplus_transport = make_unique<GameGPlusServer>(net);
        gplus_transport->game_network = Singleton<Game::GoogleMultiplayerNetwork>::Set();
        gplus_transport->handler = this;
        svc.push_back(gplus_transport.get());
      }
#endif
    } else if (proto == Protocol::InProcess) {
      if (!inprocess_transport) {
        inprocess_transport = make_unique<GameInProcessServer>(net);
        inprocess_transport->game_network = Singleton<Game::InProcessNetwork>::Set();
        inprocess_transport->handler = this;
        svc.push_back(inprocess_transport.get());
      }
    }
  }

  int Connected(Connection *c) { Game::ConnectionData::Init(c); return 0; }
  void Close(Connection *c) {
    unique_ptr<Game::ConnectionData> cd(Game::ConnectionData::Get(c));
    world->PartEntity(cd.get(), world->Get(cd->entityID), cd->team);
    c->handler.release();
    if (!cd->team) return;

    INFO(c->Name(), ": ", cd->playerName, " left");
    GameProtocol::RconRequest print(StrCat("print *** ", cd->playerName, " left"));
    BroadcastWithRetry(&print, c);
  }

  int Read(Connection *c) {
    if (proto == Protocol::TCP) c->ReadFlush(max(0, Read(c, c->rb.begin(), c->rb.size())));
    else {
      int ret;
      for (int i=0; i<c->packets.size(); i++)
        if ((ret = Read(c, c->rb.begin() + c->packets[i].offset, c->packets[i].len)) < 0) return ret;
    }
    return 0;
  }

  int Read(Connection *c, const char *content, int content_len) {
    if (content_len < GameProtocol::Header::size) return -1;
    Serializable::ConstStream in(content, content_len);
    GameProtocol::Header hdr;
    hdr.In(&in);

    // echo "ping" | nc -u 127.0.0.1 27640
    static string ping="ping\n";
    if (in.size == ping.size() && in.buf == ping) {
      INFO(c->Name(), ": ping");
      string pong = StrCat("map=default\nname=", local_game_name, "\nplayers=", last.num_send_WorldUpdate, "/24\n");
      c->SendTo(pong.data(), pong.size());
    }
#   define elif_parse(Request, in) \
    else if (hdr.id == GameProtocol::Request::ID) { \
      GameProtocol::Request req; \
      if (req.Read(&in)) return 0; \
      Request ## CB(c, &hdr, &req); \
    }
    elif_parse(JoinRequest, in)
    elif_parse(PlayerUpdate, in)
    elif_parse(RconRequest, in)
    elif_parse(RconResponse, in)
    else ERROR(c->Name(), ": parse failed: unknown type ", hdr.id, " len=", content_len);
    return in.offset;
  }

  void Write(Connection *c, int method, unsigned short seq, SerializableProto *msg) {
    static_cast<Game::Network*>(c->svc->game_network)->Write(c, method, seq, msg);
  }

  void WriteWithRetry(Connection *c, Game::ConnectionData *cd, SerializableProto *msg) {
    static_cast<Game::Network*>(c->svc->game_network)->WriteWithRetry(&cd->retry, c, msg, cd->seq++);
  }

  void WritePrintWithRetry(Connection *c, Game::ConnectionData *cd, const string &text) {
    GameProtocol::RconRequest print(text);
    WriteWithRetry(c, cd, &print);
  }

  int BroadcastWithRetry(SerializableProto *msg, Connection *skip=0) {
    int ret = 0;
    for (int i=0; i<svc.size(); ++i) ret += static_cast<Game::Network*>(svc[i]->game_network)->BroadcastWithRetry(svc[i], msg, skip);
    return ret;
  }

  int BroadcastPrintWithRetry(const string &text, Connection *skip = 0) {
    GameProtocol::RconRequest print(text);
    return BroadcastWithRetry(&print);
  }

  int Frame() {
    Time now = Now();
    static const Time MasterUpdateInterval = Minutes(5);
    if (now > last.time_post_MasterUpdate + MasterUpdateInterval || last.time_post_MasterUpdate == Time(0)) {
      last.time_post_MasterUpdate = now;
      if (!master_sink_url.empty())
        HTTPClient::WPost(net, master_sink_url, "application/octet-stream", local_game_url.c_str(), local_game_url.size());
    }

    int updated = 0;
    for (/**/; world->ranlast + timestep <= now; world->ranlast += timestep) {
      world->Update(this, timestep);
      updated++;
    }
    if (!updated || !world->broadcast_enabled) return 0;

    Scene *scene = world->scene;
    last.WorldUpdate.id++;
    last.WorldUpdate.entity.resize(scene->entity.size());

    int entity_type_index = 0, entity_index = 0;
    for (auto const &a : scene->asset)
      for (auto e : a.second) 
        last.WorldUpdate.entity[entity_index++].From(e);

    last.send_WorldUpdate[last.send_WorldUpdate_index].id = last.WorldUpdate.id;
    last.send_WorldUpdate[last.send_WorldUpdate_index].time = now;
    last.send_WorldUpdate_index = (last.send_WorldUpdate_index + 1) % sizeofarray(last.send_WorldUpdate);

    last.num_send_WorldUpdate = 0;
    for (int i = 0; i < svc.size(); ++i) {
      last.num_send_WorldUpdate += static_cast<Game::Network*>(svc[i]->game_network)->Broadcast(svc[i], &last.WorldUpdate);
    }
    if (bots) bots->Update(timestep); 
    return 0;
  }

  void RconResponseCB(Connection *c, GameProtocol::Header *hdr, GameProtocol::RconResponse *req) {
    Game::ConnectionData::Get(c)->retry.Acknowledged(hdr->seq);
  }

  void JoinRequestCB(Connection *c, GameProtocol::Header *hdr, GameProtocol::JoinRequest *req) {
    Game::ConnectionData *cd = Game::ConnectionData::Get(c);
    cd->playerName = req->PlayerName;

    bool rejoin = cd->entityID != 0;
    if (!rejoin) {
      cd->entityID = world->NewID();
      world->JoinEntity(cd, cd->entityID, &cd->team);
    }
    Entity *e = world->Get(cd->entityID);

    GameProtocol::JoinResponse response;
    response.rcon = "set_asset";
    for (vector<Asset>::const_iterator a = assets->begin(); a != assets->end(); ++a)
      StrAppend(&response.rcon, " ", a->typeID, "=", a->name);
    response.rcon += "\n";
    world->JoinRcon(cd, e, &response.rcon);
    Write(c, UDPClient::Sendto, hdr->seq, &response);

    GameProtocol::RconRequest rcon_broadcast;
    INFO(c->Name(), ": ", cd->playerName, rejoin?" re":" ", "joins, entity_id=", e->name);
    if (world->JoinedRcon(cd, e, &rcon_broadcast.Text)) BroadcastWithRetry(&rcon_broadcast);
  }

  void PlayerUpdateCB(Connection *c, GameProtocol::Header *hdr, GameProtocol::PlayerUpdate *pup) {
    Game::ConnectionData *cd = Game::ConnectionData::Get(c);
    for (int i=0; i<sizeofarray(last.send_WorldUpdate); i++) {
      if (last.send_WorldUpdate[i].id != pup->id_WorldUpdate) continue;
      cd->ping = (Now() - last.send_WorldUpdate[i].time - Time(pup->time_since_WorldUpdate)).count();
      break;
    }

    Entity *e = world->Get(cd->entityID);
    if (!e) { ERROR("missing entity ", cd->entityID); return; }
    pup->ort.To(&e->ort, &e->up);
    e->buttons = pup->buttons;

    if (!Game::Controller(e->buttons).GetPlayerList()) return;
    GameProtocol::PlayerList playerlist;
    playerlist.Text = StrCat(world->red_score, ",", world->blue_score, "\n");
    for (int i = 0; i < svc.size(); ++i)
      for (auto iter = svc[i]->endpoint.begin(), e = svc[i]->endpoint.end(); iter != e; ++iter)
        AppendSerializedPlayerData(Game::ConnectionData::Get(iter->second.get()), &playerlist.Text);
    if (bots) for (GameBots::BotVector::iterator i = bots->bots.begin(); i != bots->bots.end(); i++)
      AppendSerializedPlayerData(i->player_data, &playerlist.Text);
    Write(c, UDPClient::Sendto, cd->seq++, &playerlist);
  }

  void RconRequestCB(Connection *c, GameProtocol::Header *hdr, GameProtocol::RconRequest *rcon) {
    GameProtocol::RconResponse response;
    Write(c, UDPClient::Sendto, hdr->seq, &response);

    Game::ConnectionData *cd = Game::ConnectionData::Get(c);
    StringLineIter lines(rcon->Text);
    for (string line = lines.NextString(); !lines.Done(); line = lines.NextString()) {
      if (FLAGS_rcon_debug) INFO("rcon: ", line);
      string cmd, arg;
      Split(line, isspace, &cmd, &arg);

      if (cmd == "say") {
        BroadcastPrintWithRetry(StrCat("print <", cd->playerName, "> ", arg));
      } else if (cmd == "name") {
        if (arg.empty()) { WritePrintWithRetry(c, cd, "usage: rcon name <name_here>"); continue; }
        BroadcastPrintWithRetry(StrCat("print *** ", cd->playerName, " changed name to ", arg));
        cd->playerName = arg;
      } else if (cmd == "team") {
        int requested_team = Game::Team::FromString(arg);
        if (requested_team && requested_team != cd->team) {
          world->PartEntity(cd, world->Get(cd->entityID), cd->team);
          cd->team = requested_team;
          world->JoinEntity(cd, cd->entityID, &cd->team);
        }
      } else if (cmd == "anim") {
        string a1, a2;
        Split(arg, isspace, &a1, &a2);
        Entity *e = world->Get(atoi(a1));
        int anim_id = atoi(a2);
        if (!e || !anim_id) { WritePrintWithRetry(c, cd, "usage: anim <entity_id> <anim_id>"); continue; }
        e->animation.Start(anim_id);
      } else if (cmd == "auth") {
        if (arg == rcon_auth_passwd && rcon_auth_passwd.size()) cd->rcon_auth = true;
        WritePrintWithRetry(c, cd, StrCat("rcon auth: ", cd->rcon_auth ? "enabled" : "disabled"));
      } else if (cmd == "shutdown") {
        if (cd->rcon_auth) shutdown->Shutdown();
      } else {
        RconRequestCB(c, cd, cmd, arg);
      }
    }
  }

  virtual void RconRequestCB(Connection *c, Game::ConnectionData *cd, const string &cmd, const string &arg) {
    GameProtocol::RconRequest print(StrCat("print *** Unknown command: ", cmd));
    WriteWithRetry(c, cd, &print);
  }

  static void AppendSerializedPlayerData(const Game::ConnectionData *icd, string *out) {
    if (!icd || !icd->team || !out) return;
    StringAppendf(out, "%d,%s,%d,%d,%d\n", icd->entityID, icd->playerName.c_str(), icd->team, icd->score, icd->ping); /* XXX escape player name */
  }
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
  void WorldAddEntityFinish(Entity *e, int type) {
    CHECK(!e->asset);
    world->scene->ChangeAsset(e, assetstore->asset(assets[type]));
    NewEntityCB(e);
  }

  void WorldDeleteEntity(Scene::EntityVector &v) {
    for (Scene::EntityVector::iterator i = v.begin(); i != v.end(); i++) DelEntityCB(*i);
    world->scene->Del(v);
  }

  void Rcon(const string &text) {
    if (!conn) return;
    GameProtocol::RconRequest req(text);
    net->WriteWithRetry(&retry, conn, &req, seq++);
  }

  void MoveUp    (unsigned t) { control.SetUp();      }
  void MoveDown  (unsigned t) { control.SetDown();    }
  void MoveFwd   (unsigned t) { control.SetForward(); }
  void MoveRev   (unsigned t) { control.SetBack();    }
  void MoveLeft  (unsigned t) { control.SetLeft();    }
  void MoveRight (unsigned t) { control.SetRight();   }
  void SetCamera(const vector<string> &a) { cam_id = atoi(a.size() ? a[0] : ""); }
  void RconCmd  (const vector<string> &a) { if (a.size()) Rcon(a[0]); }
  void SetTeam  (const vector<string> &a) { if (a.size()) Rcon(StrCat("team ", a[0])); }
  void SetName  (const vector<string> &a) {
    string n = a.size() ? a[0] : "";
    if (playername == n) return;
    playername = n;
    Rcon(StrCat("name ", n));
    Singleton<FlagMap>::Set()->Set("player_name", n);
  }

  void MyEntityName(vector<string>) {
    Entity *e = world->Get(entity_id);
    INFO("me = ", e ? e->name : "");
  }

  void Reset() {
    if (conn) conn->SetError();
    conn=0; seq=0; entity_id=0; team=0; cam_id=1; reorienting=1; net=0;
    retry.Clear();
    last.id_WorldUpdate=last.seq_WorldUpdate=0;
    replay.disable();
    gameover.disable();
  }
  
  bool Connected() { return conn && conn->state == Connection::Connected; }
  int Connect(int prot, const string &url, int default_port) {
    Reset();
    if (prot == Protocol::TCP) {
      net = Singleton<Game::TCPNetwork>::Set();
      conn = sockets->tcp_client->Connect(url, default_port);
      auto h = make_unique<Connection::CallbackHandler>(bind(&GameClient::StreamRead, this, _1), Connection::CB());
      h->connected_cb = [=](Connection *c){ 
        GameProtocol::JoinRequest req;
        req.PlayerName = playername;
        net->WriteWithRetry(&retry, c, &req, seq++);
      };
      if (Connected()) h->connected_cb(conn);
      conn->handler = move(h);
      last.time_send_PlayerUpdate = Now();
      return 0;

    } else if (prot == Protocol::UDP) {
      net = Singleton<Game::UDPNetwork>::Set();
      conn = sockets->udp_client->PersistentConnection
        (url, bind(&GameClient::DatagramRead, this, _1, _2, _3), Connection::CB(), default_port);
      if (!Connected()) return 0;

      GameProtocol::ChallengeRequest req;
      net->WriteWithRetry(&retry, conn, &req, seq++);
      last.time_send_PlayerUpdate = Now();
      return 0;

#ifdef LFL_ANDROID
    } else if (prot == Protocol::GPLUS) {
      GPlusClient *gplus_client = sockets->gplus_client.get(sockets);
      sockets->Enable(gplus_client);
      net = Singleton<Game::GoogleMultiplayerNetwork>::Get();
      conn = gplus_client->PersistentConnection
        (url, bind(&GameClient::DatagramRead, this, _1, _2, _3), Connection::CB());

      GameProtocol::JoinRequest req;
      req.PlayerName = playername;
      net->WriteWithRetry(&retry, conn, &req, seq++);
      last.time_send_PlayerUpdate = Now();
      return 0;
#endif
    } else if (prot == Protocol::InProcess) {
      InProcessClient *inprocess_client = sockets->inprocess_client.get(sockets);
      sockets->Enable(inprocess_client);
      CHECK(inprocess_server);
      net = Singleton<Game::InProcessNetwork>::Set();
      conn = inprocess_client->PersistentConnection
        (inprocess_server, bind(&GameClient::DatagramRead, this, _1, _2, _3), Connection::CB());

      GameProtocol::JoinRequest req;
      req.PlayerName = playername;
      net->WriteWithRetry(&retry, conn, &req, seq++);
      last.time_send_PlayerUpdate = Now();
      return 0;
    }
    return -1;
  }

  void DatagramRead(Connection *c, const char *content, int len) { Read(c, content, len); }
  void StreamRead(Connection *c) { c->ReadFlush(max(0, Read(c, c->rb.begin(), c->rb.size()))); }

  int Read(Connection *c, const char *content, int content_length) {
    if (c != conn) return 0;
    if (!content || !content_length) { INFO(c->Name(), ": close"); Reset(); return 0 ; }

    Serializable::ConstStream in(content, content_length);
    GameProtocol::Header hdr;
    hdr.In(&in);
    if (0) {}
    elif_parse(ChallengeResponse, in)
    elif_parse(JoinResponse, in)
    elif_parse(WorldUpdate, in)
    elif_parse(RconRequest, in)
    elif_parse(RconResponse, in)
    elif_parse(PlayerList, in)        
    else ERROR("parse failed: unknown type ", hdr.id);
    return in.offset;
  }

  void Frame() {
    if (!Connected()) return;
    retry.Heartbeat(conn);
    Game::Controller CS = control;
    control.Reset();
    UpdateWorld();
    if (reorienting || (CS.buttons == last.buttons && last.time_send_PlayerUpdate + Time(100) > Now())) return;

    Entity *cam = &world->scene->cam;
    GameProtocol::PlayerUpdate pup;
    pup.id_WorldUpdate = last.id_WorldUpdate;
    pup.time_since_WorldUpdate = (Now() - last.time_recv_WorldUpdate[0]).count();
    pup.buttons = CS.buttons;
    pup.ort.From(cam->ort, cam->up);
    net->Write(conn, UDPClient::Write, seq++, &pup);

    last.buttons = CS.buttons;
    last.time_send_PlayerUpdate = Now();
  }

  void UpdateWorld() {
    int WorldUpdates = last.WorldUpdate.size();
    if (WorldUpdates < 2) return;

    last.time_frame = Now();
    unsigned updateInterval = (last.time_recv_WorldUpdate[0] - last.time_recv_WorldUpdate[1]).count();
    unsigned updateLast = (last.time_frame - last.time_recv_WorldUpdate[0]).count();
    Entity *cam = &world->scene->cam;

    if (1) { /* interpolate */
      GameProtocol::WorldUpdate *wu1=0, *wu2=0;

      /* replay */
      if (replay.enabled() && replay.while_seq != last.seq_WorldUpdate) replay.disable();
      if (replay.enabled()) {
        float frames = min(WorldUpdates - replay.start_ind - 2.0f, (Now() - replay.start).count() / float(updateInterval));
        int ind = int(replay.start_ind + frames);
        wu1 = &last.WorldUpdate[ind+1];
        wu2 = &last.WorldUpdate[ind];
        updateLast = unsigned((frames - floor(frames)) * updateInterval);
      }

      /* game over */
      if (gameover.enabled() && gameover.while_seq != last.seq_WorldUpdate) gameover.disable();

      if (!wu1) {
        wu1 = &last.WorldUpdate[WorldUpdates-1];
        wu2 = &last.WorldUpdate[WorldUpdates-2];
      }

      for (int i=0, j=0; i<wu1->entity.size() && j<wu2->entity.size(); /**/) {
        GameProtocol::Entity *e1 = &wu1->entity[i], *e2 = &wu2->entity[j];
        if      (e1->id < e2->id) { i++; continue; }
        else if (e2->id < e1->id) { j++; continue; }

        bool me = entity_id == e1->id;
        Entity *e = world->Get(e1->id);
        if (!e) e = WorldAddEntity(e1->id);
        if (!e->asset) WorldAddEntityFinish(e, e1->type);

        v3 next_pos, delta_pos;
        e1->ort.To(&e->ort, &e->up);
        e1->pos.To(&next_pos);
        e2->pos.To(&e->pos);

        if (updateLast) {
          delta_pos = next_pos;
          delta_pos.Sub(e->pos);
          delta_pos.Scale(min(1000.0f, float(updateLast)) / updateInterval);
          e->pos.Add(delta_pos);
        }

        if (e1->anim_id != e->animation.id) {
          e->animation.Reset();
          e->animation.id = e1->anim_id;
          e->animation.len = e1->anim_len;
          AnimationChange(e, e1->anim_id, e1->anim_len);
        }

        e->updated = last.time_frame;
        if (me && reorienting) { reorienting=0; e1->ort.To(&cam->ort, &cam->up); }
        if (me && !replay.enabled()) Me(e, cam_id, !reorienting);
        i++; j++;
      }
    }
  }

  void ChallengeResponseCB(Connection *c, GameProtocol::Header *hdr, GameProtocol::ChallengeResponse *challenge) {
    retry.Acknowledged(hdr->seq);
    GameProtocol::JoinRequest req;
    req.token = challenge->token;
    req.PlayerName = playername;
    net->WriteWithRetry(&retry, c, &req, seq++);
  }

  void JoinResponseCB(Connection *c, GameProtocol::Header *hdr, GameProtocol::JoinResponse *joined) {
    retry.Acknowledged(hdr->seq);
    RconRequestCB(c, hdr, joined->rcon);
  }

  void WorldUpdateCB(Connection *c, GameProtocol::Header *hdr, GameProtocol::WorldUpdate *wu) {       
    if (hdr->seq <= last.seq_WorldUpdate) {
      unsigned short cs = hdr->seq, ls = last.seq_WorldUpdate;
      cs += 16384; ls += 16384;
      if (cs <= ls) return;
    }

    last.id_WorldUpdate = wu->id;
    last.seq_WorldUpdate = hdr->seq;
    last.time_recv_WorldUpdate[1] = last.time_recv_WorldUpdate[0];
    last.time_recv_WorldUpdate[0] = Now();

    if (last.WorldUpdate.size() > 200) last.WorldUpdate.pop_front();
    last.WorldUpdate.push_back(*wu);
  }

  void PlayerListCB(Connection *c, GameProtocol::Header *hdr, GameProtocol::PlayerList *pl) { playerlist->HandleTextMessage(pl->Text); }
  void RconResponseCB(Connection *c, GameProtocol::Header *hdr, GameProtocol::RconResponse*) { retry.Acknowledged(hdr->seq); }
  void RconRequestCB(Connection *c, GameProtocol::Header *hdr, GameProtocol::RconRequest *rcon) {
    GameProtocol::RconResponse response;
    net->Write(c, UDPClient::Write, hdr->seq, &response);
    RconRequestCB(c, hdr, rcon->Text);
  }

  void RconRequestCB(Connection *c, GameProtocol::Header *hdr, const string &rcon) {
    StringLineIter lines(rcon);
    for (string line = lines.NextString(); !lines.Done(); line = lines.NextString()) {
      if (FLAGS_rcon_debug) INFO("rcon: ", line);
      string cmd, arg;
      Split(line, isspace, &cmd, &arg);

      if (cmd == "print") {
        INFO(arg);
        if (chat) chat->Write(arg);
      } else if (cmd == "player_team") {
        team = atoi(arg);
      } else if (cmd == "player_entity") {
        entity_id = atoi(arg);
        reorienting = true;
      } else if (cmd == "set_asset") {
        vector<string> items;
        Split(arg, isspace, &items);
        for (vector<string>::const_iterator i = items.begin(); i != items.end(); ++i) {
          string k, v;
          Split(*i, isint<'='>, &k, &v);
          assets[atoi(k)] = v;
          INFO("GameAsset[", k, "] = ", v);
        }
      } else if (cmd == "set_entity") {
        vector<string> items;
        Split(arg, isspace, &items);
        for (vector<string>::const_iterator i = items.begin(); i != items.end(); ++i) {
          vector<string> args;
          Split(*i, isint2<'.', '='>, &args);
          if (args.size() != 3) { ERROR("unknown arg ", *i); continue; }
          int id = atoi(args[0]);
          Entity *e = world->Get(id); 
          if (!e) e = WorldAddEntity(id);
          if      (args[1] == "color1") e->color1 = Color(args[2]);
          else if (args[1] == "color2") e->color2 = Color(args[2]);
          else SetEntityCB(e, args[1], args[2]);
        }
      } else {
        RconRequestCB(cmd, arg, hdr->seq);
      }
    }
  }

  void Me(Entity *e, int cam_id, bool assign_entity) {
    Entity *cam = &world->scene->cam;
    cam->pos = e->pos;
    if (cam_id == 1) {
      v3 v = cam->ort * 2;
      cam->pos.Sub(v);
    } else if (cam_id == 2) {
      v3 v = cam->ort * 2;
      cam->pos.Sub(v);
      v = cam->up * .1;
      cam->pos.Add(v);
    } else if (cam_id == 3) {
      assign_entity = false;
    } else if (cam_id == 4) {
      cam->pos = v3(0,3.8,0);
      cam->ort = v3(0,-1,0);
      cam->up  = v3(0,0,-1);
      assign_entity    = false;
    } if (assign_entity) {
      e->ort = cam->ort;
      e->up  = cam->up;
    }
  }
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
