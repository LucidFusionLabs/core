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

#include "core/app/gl/view.h"
#include "core/app/gl/toolkit.h"
#include "core/app/shell.h"
#include "core/web/browser/browser.h"
#include "core/game/game.h"

namespace LFL {
int Game::Team::FromString(const string &s) {
  if      (s == "home")      return Home;
  else if (s == "away")      return Away;
  else if (s == "spectator") return Spectator;
  else                       return 0;
}

void Game::Network::Visitor::VisitClients(SocketService *svc, Visitor *visitor) {
  if (svc->protocol == Protocol::TCP) {
    for (auto iter = svc->conn.begin(), e = svc->conn.end(); iter != e; ++iter) {
      Connection *c = iter->second.get();
      Game::ConnectionData *cd = Game::ConnectionData::Get(c);
      if (!cd->team) continue;
      visitor->Visit(c, cd);
    }
  } else {
    for (auto iter = svc->endpoint.begin(), e = svc->endpoint.end(); iter != e; ++iter) {
      Connection *c = iter->second.get();
      Game::ConnectionData *cd = Game::ConnectionData::Get(c);
      if (!cd->team) continue;
      visitor->Visit(c, cd);
    }
  }
}

void Game::Network::BroadcastVisitor::Visit(Connection *c, Game::ConnectionData *cd) {
  net->Write(c, UDPClient::Sendto, cd->seq++, msg);
  cd->retry.Heartbeat(c);
  sent++;
}

void Game::Network::BroadcastWithRetryVisitor::Visit(Connection *c, Game::ConnectionData *cd) {
  if (c == skip) return;
  net->WriteWithRetry(&cd->retry, c, msg, cd->seq++);
  sent++;
}

int Game::Network::Write(Connection *c, int method, unsigned short seq, SerializableProto *msg) {
  string buf;
  msg->ToString(&buf, seq);
  return Write(c, method, buf.data(), buf.size());
}

int Game::Network::Broadcast(SocketService *svc, SerializableProto *msg) {
  BroadcastVisitor visitor(this, msg);
  Visitor::VisitClients(svc, &visitor);
  return visitor.sent;
}

int Game::Network::BroadcastWithRetry(SocketService *svc, SerializableProto *msg, Connection *skip) {
  BroadcastWithRetryVisitor visitor(this, msg, skip);
  Visitor::VisitClients(svc, &visitor);
  return visitor.sent;
}

#ifdef LFL_ANDROID
int Game::GoogleMultiplayerNetwork::Write(Connection *c, int method, const char *data, int len) {
  if (c->endpoint_name.empty()) return ERRORv(-1, c->Name(), " blank send");

  static JNI *jni = Singleton<JNI>::Get();
  if (jni->gplus) {
    static jmethodID jni_gplus_method_write =
      CheckNotNull(jni->env->GetMethodID(jni->gplus_class, "write", "(Ljava/lang/String;Ljava/nio/ByteBuffer;)V"));
    jstring pn = jni->env->NewStringUTF(c->endpoint_name.c_str());
    jobject bytes = jni->env->NewDirectByteBuffer(Void(data), len);
    jni->env->CallVoidMethod(jni->gplus, jni_gplus_method_write, pn, bytes);
    jni->env->DeleteLocalRef(bytes);
    jni->env->DeleteLocalRef(pn);
  } else ERRORf("no gplus %p", jni->gplus);
  return 0;
}

void Game::GoogleMultiplayerNetwork::WriteWithRetry(ReliableNetwork *n, Connection *c, SerializableProto *req, unsigned short seq) {
  if (c->endpoint_name.empty()) return ERROR(c->Name(), " blank send");
  string v;
  req->ToString(&v, seq);

  static JNI *jni = Singleton<JNI>::Get();
  if (jni->gplus) {
    static jmethodID jni_gplus_method_write_with_retry =
      CheckNotNull(jni->env->GetMethodID(jni->gplus_class, "writeWithRetry",
                                         "(Ljava/lang/String;Ljava/nio/ByteBuffer;)V"));
    jstring pn = jni->env->NewStringUTF(c->endpoint_name.c_str());
    jobject bytes = jni->env->NewDirectByteBuffer(Void(v.c_str()), v.size());
    jni->env->CallVoidMethod(jni->gplus, jni_gplus_method_write_with_retry, pn, bytes);
    jni->env->DeleteLocalRef(bytes);
    jni->env->DeleteLocalRef(pn);
  } else ERRORf("no gplus %p", jni->gplus);
}
#endif

int Game::InProcessNetwork::Write(Connection *c, int method, const char *data, int len) {
  if (!c->next) return ERRORv(-1, c->Name(), " blank send");
  c->next->AddPacket(data, len);
  return 0;
}

void Game::InProcessNetwork::WriteWithRetry(ReliableNetwork *n, Connection *c, SerializableProto *req, unsigned short seq) {
  string v;
  req->ToString(&v, seq);
  Write(c, 0, v.data(), v.size());
}

void Game::TCPNetwork::WriteWithRetry(ReliableNetwork *reliable, Connection *c, SerializableProto *req, unsigned short seq) {
  string v;
  req->ToString(&v, seq);
  Write(c, 0, v.data(), v.size());
}

int Game::UDPNetwork::Write(Connection *c, int method, const char *buf, int len) {
  return method == UDPClient::Sendto ? c->SendTo(buf, len) : c->WriteFlush(buf, len);
}

void Game::UDPNetwork::WriteWithRetry(ReliableNetwork *reliable, Connection *c, SerializableProto *req, unsigned short seq) {
  return reliable->WriteWithRetry(c, req, seq);
}

void Game::ReliableUDPNetwork::WriteWithRetry(Connection *c, SerializableProto *req, unsigned short seq) {
  pair<Time, string> &msg = retry[seq];
  req->ToString(&msg.second, seq);

  msg.first = Now();
  net->Write(c, method, msg.second.data(), msg.second.size());
}

void Game::ReliableUDPNetwork::Heartbeat(Connection *c) {
  for (auto i = retry.begin(), e = retry.end(); i != e; ++i) {
    if (i->second.first + timeout > Now()) continue;
    i->second.first = Now();
    net->Write(c, method, i->second.second.data(), i->second.second.size());
  }
}

v3 Game::Controller::Acceleration(v3 ort, v3 u) const {
  v3 ret, r = v3::Cross(ort, u);
  ort.Norm(); u.Norm(); r.Norm();
  bool up=GetUp(), down=GetDown(), forward=GetForward(), back=GetBack(), left=GetLeft(), right=GetRight();

  if (forward && !back) {                ret.Add(ort); }
  if (back && !forward) { ort.Scale(-1); ret.Add(ort); }
  if (right && !left)   {                ret.Add(r);   }
  if (!right && left)   { r.Scale(-1);   ret.Add(r);   }
  if (up && !down)      {                ret.Add(u);   }
  if (down && !up)      { u.Scale(-1);   ret.Add(u);   }

  ret.Norm();
  return ret;
}

void Game::Controller::PrintButtons(unsigned buttons) {
  string button_text;
  for (int i=0, l=sizeof(buttons)*8; i<l; i++) if (buttons & (1<<i)) StrAppend(&button_text, i, ", ");
  INFO("buttons: ", button_text);
}

void GameBots::Insert(int num) {
  for (int i=0; i<num; i++) {
    Game::ConnectionData *cd = new Game::ConnectionData();
    cd->playerName = StrCat("bot", i+1);
    cd->entityID = world->NewID();
    Entity *e = world->JoinEntity(cd, cd->entityID, &cd->team);
    e->type = Entity::Type::BOT;
    if (cd->team == Game::Team::Red || cd->team == Game::Team::Blue) {
      bots.push_back(Bot(e, cd));
    } else {
      world->PartEntity(cd, e, cd->team);
      delete cd;
      return;
    }
  }
  sort(bots.begin(), bots.end(), Bot::ComparePlayerEntityID);
}

bool GameBots::RemoveFromTeam(int team) {
  for (int i=bots.size()-1; i>=1; i--) {
    if (bots[i].player_data->team == team) { Delete(&bots[i]); bots.erase(bots.begin()+i); return true; }
  } return false;
}

void GameBots::Delete(Bot *b) {
  world->PartEntity(b->player_data, b->entity, b->player_data->team);
  delete b->player_data;
}

void GameBots::Clear() {
  for (int i=0; i<bots.size(); i++) Delete(&bots[i]);
  bots.clear();
}

void GameServer::InitTransport(SocketServices *net, int p, int port) {
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

void GameServer::Close(Connection *c) {
  unique_ptr<Game::ConnectionData> cd(Game::ConnectionData::Get(c));
  world->PartEntity(cd.get(), world->Get(cd->entityID), cd->team);
  c->handler.release();
  if (!cd->team) return;

  INFO(c->Name(), ": ", cd->playerName, " left");
  GameProtocol::RconRequest print(StrCat("print *** ", cd->playerName, " left"));
  BroadcastWithRetry(&print, c);
}

int GameServer::Read(Connection *c) {
  if (proto == Protocol::TCP) c->ReadFlush(max(0, Read(c, c->rb.begin(), c->rb.size())));
  else {
    int ret;
    for (int i=0; i<c->packets.size(); i++)
      if ((ret = Read(c, c->rb.begin() + c->packets[i].offset, c->packets[i].len)) < 0) return ret;
  }
  return 0;
}

int GameServer::Read(Connection *c, const char *content, int content_len) {
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

void GameServer::Write(Connection *c, int method, unsigned short seq, SerializableProto *msg) {
  static_cast<Game::Network*>(c->svc->game_network)->Write(c, method, seq, msg);
}

void GameServer::WriteWithRetry(Connection *c, Game::ConnectionData *cd, SerializableProto *msg) {
  static_cast<Game::Network*>(c->svc->game_network)->WriteWithRetry(&cd->retry, c, msg, cd->seq++);
}

void GameServer::WritePrintWithRetry(Connection *c, Game::ConnectionData *cd, const string &text) {
  GameProtocol::RconRequest print(text);
  WriteWithRetry(c, cd, &print);
}

int GameServer::BroadcastWithRetry(SerializableProto *msg, Connection *skip) {
  int ret = 0;
  for (int i=0; i<svc.size(); ++i) ret += static_cast<Game::Network*>(svc[i]->game_network)->BroadcastWithRetry(svc[i], msg, skip);
  return ret;
}

int GameServer::BroadcastPrintWithRetry(const string &text, Connection *skip) {
  GameProtocol::RconRequest print(text);
  return BroadcastWithRetry(&print);
}

int GameServer::Frame() {
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

void GameServer::RconResponseCB(Connection *c, GameProtocol::Header *hdr, GameProtocol::RconResponse *req) {
  Game::ConnectionData::Get(c)->retry.Acknowledged(hdr->seq);
}

void GameServer::JoinRequestCB(Connection *c, GameProtocol::Header *hdr, GameProtocol::JoinRequest *req) {
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

void GameServer::PlayerUpdateCB(Connection *c, GameProtocol::Header *hdr, GameProtocol::PlayerUpdate *pup) {
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

void GameServer::RconRequestCB(Connection *c, GameProtocol::Header *hdr, GameProtocol::RconRequest *rcon) {
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

void GameServer::RconRequestCB(Connection *c, Game::ConnectionData *cd, const string &cmd, const string &arg) {
  GameProtocol::RconRequest print(StrCat("print *** Unknown command: ", cmd));
  WriteWithRetry(c, cd, &print);
}

void GameServer::AppendSerializedPlayerData(const Game::ConnectionData *icd, string *out) {
  if (!icd || !icd->team || !out) return;
  StringAppendf(out, "%d,%s,%d,%d,%d\n", icd->entityID, icd->playerName.c_str(), icd->team, icd->score, icd->ping); /* XXX escape player name */
}

int GameUDPServer::Hash(Connection *c) {
  IPV4Endpoint remote = c ? c->RemoteIPV4() : IPV4Endpoint();
  int conn_key[3] = { int(remote.addr), secret1, remote.port }; 
  return fnv32(conn_key, sizeof(conn_key), secret2);
}

int GameUDPServer::UDPFilter(SocketConnection *c, const char *content, int content_len) {
  if (content_len < GameProtocol::Header::size) return -1;

  Serializable::ConstStream in(content, content_len);
  GameProtocol::Header hdr;
  hdr.In(&in);

  GameProtocol::ChallengeRequest challenge;
  GameProtocol::JoinRequest join;

  // echo "ping" | nc -u 127.0.0.1 27640
  static string ping="ping\n";
  if (in.size == ping.size() && in.buf == ping) {
    dynamic_cast<GameServer*>(handler)->Read(c, content, content_len);
  } else if (hdr.id == GameProtocol::ChallengeRequest::ID && !challenge.Read(&in)) {
    GameProtocol::ChallengeResponse response;
    response.token = Hash(c);
    string buf;
    response.ToString(&buf, hdr.seq);
    c->SendTo(buf.data(), buf.size());
  } else if (hdr.id == GameProtocol::JoinRequest::ID && !join.Read(&in)) {
    if (join.token == Hash(c)) return 0;
  } else ERROR(c->Name(), ": parse failed: unknown type ", hdr.id, " bytes ", in.size);
  return 1;
}

void GameClient::WorldAddEntityFinish(Entity *e, int type) {
  CHECK(!e->asset);
  world->scene->ChangeAsset(e, assetstore->asset(assets[type]));
  NewEntityCB(e);
}

void GameClient::WorldDeleteEntity(Scene::EntityVector &v) {
  for (Scene::EntityVector::iterator i = v.begin(); i != v.end(); i++) DelEntityCB(*i);
  world->scene->Del(v);
}

void GameClient::Rcon(const string &text) {
  if (!conn) return;
  GameProtocol::RconRequest req(text);
  net->WriteWithRetry(&retry, conn, &req, seq++);
}

void GameClient::SetName(const vector<string> &a) {
  string n = a.size() ? a[0] : "";
  if (playername == n) return;
  playername = n;
  Rcon(StrCat("name ", n));
  Singleton<FlagMap>::Set()->Set("player_name", n);
}

void GameClient::MyEntityName(vector<string>) {
  Entity *e = world->Get(entity_id);
  INFO("me = ", e ? e->name : "");
}

void GameClient::Reset() {
  if (conn) conn->SetError();
  conn=0; seq=0; entity_id=0; team=0; cam_id=1; reorienting=1; net=0;
  retry.Clear();
  last.id_WorldUpdate=last.seq_WorldUpdate=0;
  replay.disable();
  gameover.disable();
}

int GameClient::Connect(int prot, const string &url, int default_port) {
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

int GameClient::Read(Connection *c, const char *content, int content_length) {
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

void GameClient::Frame() {
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

void GameClient::UpdateWorld() {
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

void GameClient::ChallengeResponseCB(Connection *c, GameProtocol::Header *hdr, GameProtocol::ChallengeResponse *challenge) {
  retry.Acknowledged(hdr->seq);
  GameProtocol::JoinRequest req;
  req.token = challenge->token;
  req.PlayerName = playername;
  net->WriteWithRetry(&retry, c, &req, seq++);
}

void GameClient::JoinResponseCB(Connection *c, GameProtocol::Header *hdr, GameProtocol::JoinResponse *joined) {
  retry.Acknowledged(hdr->seq);
  RconRequestCB(c, hdr, joined->rcon);
}

void GameClient::WorldUpdateCB(Connection *c, GameProtocol::Header *hdr, GameProtocol::WorldUpdate *wu) {       
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

void GameClient::RconRequestCB(Connection *c, GameProtocol::Header *hdr, GameProtocol::RconRequest *rcon) {
  GameProtocol::RconResponse response;
  net->Write(c, UDPClient::Write, hdr->seq, &response);
  RconRequestCB(c, hdr, rcon->Text);
}

void GameClient::RconRequestCB(Connection *c, GameProtocol::Header *hdr, const string &rcon) {
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

void GameClient::Me(Entity *e, int cam_id, bool assign_entity) {
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

}; // namespace LFL
