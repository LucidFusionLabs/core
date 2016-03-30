/*
 * $Id: assets.cpp 1334 2014-11-28 09:14:21Z justin $
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

#include "core/app/gui.h"
#include "core/web/browser.h"
#include "core/game/game.h"

namespace LFL {
int Game::Team::FromString(const string &s) {
  if      (s == "home")      return Home;
  else if (s == "away")      return Away;
  else if (s == "spectator") return Spectator;
  else                       return 0;
}

void Game::Network::Visitor::VisitClients(Service *svc, Visitor *visitor) {
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

int Game::Network::Write(Connection *c, int method, unsigned short seq, Serializable *msg) {
  string buf;
  msg->ToString(&buf, seq);
  return Write(c, method, buf.data(), buf.size());
}

int Game::Network::Broadcast(Service *svc, Serializable *msg) {
  BroadcastVisitor visitor(this, msg);
  Visitor::VisitClients(svc, &visitor);
  return visitor.sent;
}

int Game::Network::BroadcastWithRetry(Service *svc, Serializable *msg, Connection *skip) {
  BroadcastWithRetryVisitor visitor(this, msg, skip);
  Visitor::VisitClients(svc, &visitor);
  return visitor.sent;
}

#ifdef LFL_ANDROID
int Game::GoogleMultiplayerNetwork::Write(Connection *c, int method, const char *data, int len) {
  if (c->endpoint_name.empty()) return ERRORv(-1, c->Name(), " blank send");
  AndroidGPlusSendUnreliable(c->endpoint_name.c_str(), data, len);
  return 0;
}

void Game::GoogleMultiplayerNetwork::WriteWithRetry(ReliableNetwork *n, Connection *c, Serializable *req, unsigned short seq) {
  if (c->endpoint_name.empty()) return ERROR(c->Name(), " blank send");
  string v;
  req->ToString(&v, seq);
  int ret;
  if ((ret = AndroidGPlusSendReliable(c->endpoint_name.c_str(), buf.c_str(), buf.size())) < 0) ERROR("WriteWithRetry ", ret);
}
#endif

int Game::InProcessNetwork::Write(Connection *c, int method, const char *data, int len) {
  if (!c->next) return ERRORv(-1, c->Name(), " blank send");
  c->next->AddPacket(data, len);
  return 0;
}

void Game::InProcessNetwork::WriteWithRetry(ReliableNetwork *n, Connection *c, Serializable *req, unsigned short seq) {
  string v;
  req->ToString(&v, seq);
  Write(c, 0, v.data(), v.size());
}

void Game::TCPNetwork::WriteWithRetry(ReliableNetwork *reliable, Connection *c, Serializable *req, unsigned short seq) {
  string v;
  req->ToString(&v, seq);
  Write(c, 0, v.data(), v.size());
}

int Game::UDPNetwork::Write(Connection *c, int method, const char *buf, int len) {
  return method == UDPClient::Sendto ? c->SendTo(buf, len) : c->WriteFlush(buf, len);
}

void Game::UDPNetwork::WriteWithRetry(ReliableNetwork *reliable, Connection *c, Serializable *req, unsigned short seq) {
  return reliable->WriteWithRetry(c, req, seq);
}

void Game::ReliableUDPNetwork::WriteWithRetry(Connection *c, Serializable *req, unsigned short seq) {
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
  int conn_key[3] = { int(c->addr), secret1, c->port }; 
  return fnv32(conn_key, sizeof(conn_key), secret2);
}

int GameUDPServer::UDPFilter(Connection *c, const char *content, int content_len) {
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
