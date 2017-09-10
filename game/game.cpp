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
#include "core/web/browser.h"
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

}; // namespace LFL
