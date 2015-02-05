/*
 * $Id: game.h 1335 2014-12-02 04:13:46Z justin $
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

#ifndef __LFL_LFAPP_GAME_H__
#define __LFL_LFAPP_GAME_H__
namespace LFL {

DECLARE_bool(rcon_debug);

struct GameServer;
struct Game {
    typedef unsigned short EntityID, TeamType;
    struct State { enum { GAME_ON=1, GAME_OVER=2 }; };
    struct ConnectionData;
    struct Team {
        enum { Spectator=1, Home=2, Red=2, Away=3, Blue=3 }; 
        static int Random() { return 2 + ::rand() % 2; }
        static int FromString(const string &s) {
            if      (s == "home")      return Home;
            else if (s == "away")      return Away;
            else if (s == "spectator") return Spectator;
            else                       return 0;
        }
        static bool IsTeamID(TeamType n) { return n == Spectator || n == Home || n == Away; }
    };

    struct ReliableNetwork {
        virtual void Heartbeat(Connection *c) {}
        virtual void WriteWithRetry(Connection *c, Serializable *req, unsigned short seq) = 0;
    };
    struct Network {
        struct Visitor {
            virtual void Visit(Connection *c, Game::ConnectionData *cd) = 0;
            static void Accept(Service *svc, Visitor *visitor) {
                for (Service::EndpointMap::iterator iter = svc->endpoint.begin(); iter != svc->endpoint.end(); iter++) {
                    Connection *c = (*iter).second;
                    Game::ConnectionData *cd = Game::ConnectionData::Get(c);
                    if (!cd->team) continue;
                    visitor->Visit(c, cd);
                }
            }
        };
        struct BroadcastVisitor : public Visitor {
            Game::Network *net; Serializable *msg; int sent;
            BroadcastVisitor(Game::Network *n, Serializable *m) : net(n), msg(m), sent(0) {}
            virtual void Visit(Connection *c, Game::ConnectionData *cd) {
                net->Write(c, UDPClient::Sendto, cd->seq++, msg);
                cd->retry.Heartbeat(c);
                sent++;
            }
        };
        struct BroadcastWithRetryVisitor : public Visitor {
            Game::Network *net; Serializable *msg; Connection *skip; int sent;
            BroadcastWithRetryVisitor(Game::Network *n, Serializable *m, Connection *Skip=0) : net(n), msg(m), skip(Skip), sent(0) {}
            virtual void Visit(Connection *c, Game::ConnectionData *cd) {
                if (c == skip) return;
                net->WriteWithRetry(&cd->retry, c, msg, cd->seq++);
                sent++;
            }
        };

        virtual int Write(Connection *c, int method, const char *data, int len) = 0;
        virtual void WriteWithRetry(ReliableNetwork*, Connection*, Serializable*, unsigned short seq) = 0;

        int Write(Connection *c, int method, unsigned short seq, Serializable *msg) {
            string buf; msg->ToString(&buf, seq);
            return Write(c, method, buf.data(), buf.size());
        }
        int Broadcast(Service *svc, Serializable *msg) {
            BroadcastVisitor visitor(this, msg);
            Visitor::Accept(svc, &visitor);
            return visitor.sent;
        }
        int BroadcastWithRetry(Service *svc, Serializable *msg, Connection *skip=0) {
            BroadcastWithRetryVisitor visitor(this, msg, skip);
            Visitor::Accept(svc, &visitor);
            return visitor.sent;
        }
    };
#ifdef LFL_ANDROID
    struct GoogleMultiplayerNetwork : public Network {
        virtual int Write(Connection *c, int method, const char *data, int len) {
            if (c->endpoint_name.empty()) { ERROR(c->name(), " blank send"); return -1; }
            android_gplus_send_unreliable(c->endpoint_name.c_str(), data, len);
            return 0;
        }
        virtual void WriteWithRetry(ReliableNetwork *n, Connection *c, Serializable *req, unsigned short seq) {
            if (c->endpoint_name.empty()) { ERROR(c->name(), " blank send"); return; }
            string buf = req->ToString(); int ret;
            if ((ret = android_gplus_send_reliable(c->endpoint_name.c_str(), buf.c_str(), buf.size())) < 0) ERROR("WriteWithRetry ", ret);
        }
    };
#endif
    struct UDPNetwork : public Network {
        virtual int Write(Connection *c, int method, const char *buf, int len) {
            return method == UDPClient::Sendto ? c->sendto(buf, len) : c->writeflush(buf, len);
        }
        virtual void WriteWithRetry(ReliableNetwork *reliable, Connection *c, Serializable *req, unsigned short seq) {
            return reliable->WriteWithRetry(c, req, seq);
        }
    };
    struct ReliableUDPNetwork : public ReliableNetwork {
        typedef map<unsigned short, pair<Time, string> > RetryMap;
        Game::Network *net;
        RetryMap retry;
        unsigned method, timeout;
        ReliableUDPNetwork(unsigned m, unsigned t=500) : net(Singleton<UDPNetwork>::Get()), method(m), timeout(t) {}

        void Clear() { retry.clear(); }
        void Acknowledged(unsigned short id) { retry.erase(id); }
        void WriteWithRetry(Connection *c, Serializable *req, unsigned short seq) {
            pair<Time, string> &msg = retry[seq];
            req->ToString(&msg.second, seq);

            msg.first = Now();
            net->Write(c, method, msg.second.data(), msg.second.size());
        }
        void Heartbeat(Connection *c) {
            for (RetryMap::iterator i = retry.begin(); i != retry.end(); i++) {
                if ((*i).second.first + timeout > Now()) continue;

                (*i).second.first = Now();
                net->Write(c, method, (*i).second.second.data(), (*i).second.second.size());
            }
        }
    };

    struct Protocol {
        struct Header : public Serializable::Header {};
        struct Position {
            static const int size = 12, scale = 1000;
            int x, y, z;

            void From(const v3 &v) { x=(int)(v.x*scale); y=(int)(v.y*scale); z=(int)(v.z*scale); }
            void To(v3 *v) { v->x=(float)x/scale; v->y=(float)y/scale; v->z=(float)z/scale; }
            void Out(Serializable::Stream *o) const { o->Htonl( x); o->Htonl( y); o->Htonl( z); }
            void In(const Serializable::Stream *i)  { i->Ntohl(&x); i->Ntohl(&y); i->Ntohl(&z); }
        };
        struct Orientation {
            static const int size = 12, scale=16384;
            short ort_x, ort_y, ort_z, up_x, up_y, up_z;

            void From(const v3 &ort, const v3 &up) {
                ort_x = (short)(ort.x*scale); ort_y = (short)(ort.y*scale); ort_z = (short)(ort.z*scale);
                 up_x = (short)(up.x*scale);  up_y  = (short)(up.y*scale);  up_z =  (short)(up.z*scale);
            }
            void To(v3 *ort, v3 *up) {
                ort->x = (float)ort_x/scale; ort->y = (float)ort_y/scale; ort->z = (float)ort_z/scale;
                 up->x = (float) up_x/scale;  up->y = (float) up_y/scale;  up->z = (float) up_z/scale;
            }
            void Out(Serializable::Stream *o) const { o->Htons( ort_x); o->Htons( ort_y); o->Htons( ort_z); o->Htons( up_x); o->Htons( up_y); o->Htons( up_z); }
            void In(const Serializable::Stream *i)  { i->Ntohs(&ort_x); i->Ntohs(&ort_y); i->Ntohs(&ort_z); i->Ntohs(&up_x); i->Ntohs(&up_y); i->Ntohs(&up_z); }
        };
        struct Velocity {
            static const int size = 6, scale=1000;
            unsigned short x, y, z;

            void From(const v3 &v) { x=(unsigned short)(v.x*scale); y=(unsigned short)(v.y*scale); z=(unsigned short)(v.z*scale); }
            void To(v3 *v) { v->x=(float)x/scale; v->y=(float)y/scale; v->z=(float)z/scale; }
            void Out(Serializable::Stream *o) const { o->Htons( x); o->Htons( y); o->Htons( z); }
            void In(const Serializable::Stream *i)  { i->Ntohs(&x); i->Ntohs(&y); i->Ntohs(&z); }
        };
        struct Entity {
            static const int size = 8 + Position::size + Orientation::size + Velocity::size;
            unsigned short id, type, anim_id, anim_len;
            Position pos;
            Orientation ort;
            Velocity vel;

            void From(const LFL::Entity *e) { id=atoi(e->name.c_str()); type=e->asset?e->asset->typeID:0; anim_id=e->animation.id; anim_len=e->animation.len; pos.From(e->pos); ort.From(e->ort, e->up); vel.From(e->vel); }
            void Out(Serializable::Stream *o) const { o->Htons( id); o->Htons( type); o->Htons( anim_id); o->Htons( anim_len); pos.Out(o); ort.Out(o); vel.Out(o); }
            void In(const Serializable::Stream *i)  { i->Ntohs(&id); i->Ntohs(&type); i->Ntohs(&anim_id); i->Ntohs(&anim_len); pos.In(i);  ort.In(i);  vel.In(i);  }
        };
        struct Collision {
            static const int size = 8;
            unsigned short fmt, id1, id2, time;

            void Out(Serializable::Stream *o) const { o->Htons( fmt); o->Htons( id1); o->Htons( id2); o->Htons( time); }
            void In(const Serializable::Stream *i)  { i->Ntohs(&fmt); i->Ntohs(&id1); i->Ntohs(&id2); i->Ntohs(&time); }
        };
        struct ChallengeRequest : public Serializable {
            int Type() const { return 1; }
            int Size() const { return HeaderSize(); }
            int HeaderSize() const { return 0; }

            void Out(Serializable::Stream *o) const {}
            int   In(const Serializable::Stream *i) { return 0; }
        };
        struct ChallengeResponse : public Serializable {
            int token;

            int Type() const { return 2; }
            int Size() const { return HeaderSize(); }
            int HeaderSize() const { return 4; }

            void Out(Serializable::Stream *o) const { o->Htonl( token); }
            int   In(const Serializable::Stream *i) { i->Ntohl(&token); return 0; }
        };
        struct JoinRequest : public Serializable {
            int token;
            string PlayerName;

            int Type() const { return 3; }
            int Size() const { return HeaderSize() + PlayerName.size(); }
            int HeaderSize() const { return 4; }

            void Out(Serializable::Stream *o) const { o->Htonl( token); o->String(PlayerName); }
            int   In(const Serializable::Stream *i) { i->Ntohl(&token); PlayerName = i->Get(); return 0; }
        };
        struct JoinResponse : public Serializable {
            string rcon;

            int Type() const { return 4; }
            int Size() const { return rcon.size(); }
            int HeaderSize() const { return 0; }

            void Out(Serializable::Stream *o) const { o->String(rcon); }
            int   In(const Serializable::Stream *i) { rcon = i->Get(); return 0; }
        };
        struct WorldUpdate : public Serializable {
            unsigned short id;
            vector<Protocol::Entity> entity;
            vector<Protocol::Collision> collision;

            int Type() const { return 5; }
            int Size() const { return HeaderSize() + entity.size() * Entity::size + collision.size() * Collision::size; }
            int HeaderSize() const { return 6; }

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
            unsigned short id_WorldUpdate, time_since_WorldUpdate;
            unsigned buttons;
            Orientation ort;

            int Type() const { return 6; }
            int Size() const { return HeaderSize(); }
            int HeaderSize() const { return 8 + Orientation::size; }

            void Out(Serializable::Stream *o) const { o->Htons( id_WorldUpdate); o->Htons( time_since_WorldUpdate); o->Htonl( buttons); ort.Out(o); }
            int   In(const Serializable::Stream *i) { i->Ntohs(&id_WorldUpdate); i->Ntohs(&time_since_WorldUpdate); i->Ntohl(&buttons); ort.In(i); return 0; }
        };
        struct RconRequest : public Serializable {
            string Text;
            RconRequest(const string &t=string()) : Text(t) {}

            int Type() const { return 7; }
            int Size() const { return HeaderSize() + Text.size(); }
            int HeaderSize() const { return 0; }

            void Out(Serializable::Stream *o) const { o->String(Text); }
            int   In(const Serializable::Stream *i) { Text = i->Get(); return 0; }
        };
        struct RconResponse : public Serializable {
            int Type() const { return 8; }
            int Size() const { return HeaderSize(); }
            int HeaderSize() const { return 0; }

            void Out(Serializable::Stream *o) const {}
            int   In(const Serializable::Stream *i) { return 0; }
        };
        struct PlayerList : public RconRequest {
            int Type() const { return 9; }
        };
    };

    struct ConnectionData {
        Game::EntityID entityID=0;
        string playerName;
        unsigned short ping=0, team=0, seq=0, score=0;
        bool rcon_auth=0;
        Game::ReliableUDPNetwork retry;
        ConnectionData() : retry(UDPClient::Sendto) {}

        static void Init(Connection *out) { ConnectionData *cd = new(out->wb) ConnectionData(); }
        static ConnectionData *Get(Connection *c) { return (ConnectionData*)c->wb; }
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

        v3 Acceleration(v3 ort, v3 u) const {
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

        static void PrintButtons(unsigned buttons) {
            string button_text;
            for (int i=0, l=sizeof(buttons)*8; i<l; i++) if (buttons & (1<<i)) StrAppend(&button_text, i, ", ");
            INFO("buttons: ", button_text);
        }
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

    virtual void Update(GameServer *server, unsigned timestep) {};
    virtual Entity *JoinEntity(ConnectionData *cd, EntityID, TeamType *team) { int *tc = TeamCount(*team); if (tc) (*tc)++; return 0; }
    virtual void    PartEntity(ConnectionData *cd, Entity *e, TeamType team) { int *tc = TeamCount(team);  if (tc) (*tc)--; }
    virtual bool      JoinRcon(ConnectionData *cd, Entity *e, string *out) { StrAppend(out, "player_entity ", cd->entityID, "\nplayer_team ", cd->team, "\n"); return true; }
    virtual bool    JoinedRcon(ConnectionData *cd, Entity *e, string *out) { StrAppend(out, "print *** ", cd->playerName, " joins"); return true; }
    int *TeamCount(int team) { CHECK(team >= 0 && team < sizeofarray(teamcount)); return &teamcount[team]; }

    EntityID NewID() { do { if (!++next_id) continue; } while (scene->Get(StrCat(next_id))); return next_id; }
    Entity *Add(EntityID id, Entity *e) { e->SetName(StringPrintf("%05d", id)); return scene->Add(e); }
    Entity *Get(EntityID id) { return scene->Get(StringPrintf("%05d", id)); }
    virtual void Del(EntityID id) { scene->Del(StringPrintf("%05d", id)); }
    static EntityID GetID(const Entity *e) { return atoi(e->name.c_str()); }
};

struct GameBots {
    struct Bot {
        Entity *entity;
        Game::ConnectionData *player_data;
        Time last_shot=0;
        Bot(Entity *e=0, Game::ConnectionData *pd=0) : entity(e), player_data(pd) {}
        static bool ComparePlayerEntityID(const Bot &l, const Bot &r) { return l.player_data->entityID < r.player_data->entityID; }
    };
    typedef vector<Bot> BotVector;
    Game *world;
    BotVector bots;
    GameBots(Game *w) : world(w) {}

    virtual void Insert(int num) {
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
    virtual bool RemoveFromTeam(int team) {
        for (int i=bots.size()-1; i>=1; i--) {
            if (bots[i].player_data->team == team) { Delete(&bots[i]); bots.erase(bots.begin()+i); return true; }
        } return false;
    }
    virtual void Delete(Bot *b) {
        world->PartEntity(b->player_data, b->entity, b->player_data->team);
        delete b->player_data;
    }
    virtual void Clear() {
        for (int i=0; i<bots.size(); i++) Delete(&bots[i]);
        bots.clear();
    }
    virtual void Update(unsigned dt) {}
};

struct GameServer : public Query {
    struct History {
        Game::Protocol::WorldUpdate WorldUpdate;
        struct WorldUpdateHistory { unsigned short id; Time time; } send_WorldUpdate[3];
        int send_WorldUpdate_index=0, num_send_WorldUpdate=0;
        Time time_post_MasterUpdate=0;
        History() { WorldUpdate.id=0; memzeros(send_WorldUpdate); }
    };

    Game *world;
    GameBots *bots;
    unsigned timestep;
    vector<Service*> svc;
    string rcon_auth_passwd, master_sink_url, local_game_name, local_game_url;
    const vector<Asset> *assets;
    History last;
    GameServer(Game *w, unsigned ts, const string &name, const string &url, const vector<Asset> *a) :
        world(w), bots(0), timestep(ts), local_game_name(name), local_game_url(url), assets(a) {}

    int Connected(Connection *c) { Game::ConnectionData::Init(c); return 0; }
    void Close(Connection *c) {
        Game::ConnectionData *cd = Game::ConnectionData::Get(c);
        world->PartEntity(cd, world->Get(cd->entityID), cd->team);
        c->query = 0;
        if (!cd->team) return;

        INFO(c->name(), ": ", cd->playerName.c_str(), " left");
        Game::Protocol::RconRequest print(StrCat("print *** ", cd->playerName, " left"));
        BroadcastWithRetry(&print, c);
    }
    int Read(Connection *c) {
        for (int i=0; i<c->packets.size(); i++) {
            int ret = Read(c, c->packets[i].buf, c->packets[i].len);
            if (ret < 0) return ret;
        }
        return 0;
    }
    int Read(Connection *c, const char *content, int content_len) {
        if (content_len < Game::Protocol::Header::size) return -1;

        Serializable::ConstStream in(content, content_len);
        Game::Protocol::Header hdr;
        hdr.In(&in);

        // echo "ping" | nc -u 127.0.0.1 27640
        static string ping="ping\n";
        if (in.size == ping.size() && in.buf == ping) {
            INFO(c->name(), ": ping");
            string pong = StrCat("map=default\nname=", local_game_name, "\nplayers=", last.num_send_WorldUpdate, "/24\n");
            c->sendto(pong.data(), pong.size());
        }
#define game_protocol_request_type(Request) ((Game::Protocol::Request*)0)->Game::Protocol::Request::Type()
#       define elif_parse(Request, in) \
        else if (hdr.id == game_protocol_request_type(Request)) { \
            Game::Protocol::Request req; \
            if (!req.Read(&in)) Request ## CB(c, &hdr, &req); \
        }
        elif_parse(JoinRequest, in)
        elif_parse(PlayerUpdate, in)
        elif_parse(RconRequest, in)
        elif_parse(RconResponse, in)
        else ERROR(c->name(), ": parse failed: unknown type ", hdr.id);
        return 0;
    }

    void Write(Connection *c, int method, unsigned short seq, Serializable *msg) {
        ((Game::Network*)c->svc->game_network)->Write(c, method, seq, msg);
    }
    void WriteWithRetry(Connection *c, Game::ConnectionData *cd, Serializable *msg) {
        ((Game::Network*)c->svc->game_network)->WriteWithRetry(&cd->retry, c, msg, cd->seq++);
    }
    void WritePrintWithRetry(Connection *c, Game::ConnectionData *cd, const string &text) {
        Game::Protocol::RconRequest print(text);
        WriteWithRetry(c, cd, &print);
    }
    int BroadcastWithRetry(Serializable *msg, Connection *skip=0) {
        int ret = 0; for (int i = 0; i < svc.size(); ++i)
            ret += ((Game::Network*)svc[i]->game_network)->BroadcastWithRetry(svc[i], msg, skip);
        return ret;
    }
    int BroadcastPrintWithRetry(const string &text, Connection *skip = 0) {
        Game::Protocol::RconRequest print(text);
        return BroadcastWithRetry(&print);
    }

    int Frame() {
        Time now = Now();
        static const int MasterUpdateInterval = Minutes(5);
        if (now > last.time_post_MasterUpdate + MasterUpdateInterval || !last.time_post_MasterUpdate) {
            last.time_post_MasterUpdate = now;
            if (!master_sink_url.empty())
                Singleton<HTTPClient>::Get()->WPost(master_sink_url.c_str(), "application/octet-stream", (char*)local_game_url.c_str(), local_game_url.size());
        }

        int updated = 0;
        for (/**/; world->ranlast + timestep <= now; world->ranlast += timestep) {
            world->Update(this, timestep);
            updated++;
        }
        if (!updated || !world->broadcast_enabled) return 0;

        Scene *scene = world->scene;
        last.WorldUpdate.id++;
        last.WorldUpdate.entity.resize(scene->entityMap.size());

        int entity_type_index = 0, entity_index = 0;
        for (Scene::EntityAssetMap::iterator i = scene->assetMap.begin(); i != scene->assetMap.end(); i++)
            for (Scene::EntityVector::iterator j = (*i).second.begin(); j != (*i).second.end(); j++)
                last.WorldUpdate.entity[entity_index++].From(*j);

        last.send_WorldUpdate[last.send_WorldUpdate_index].id = last.WorldUpdate.id;
        last.send_WorldUpdate[last.send_WorldUpdate_index].time = now;
        last.send_WorldUpdate_index = (last.send_WorldUpdate_index + 1) % sizeofarray(last.send_WorldUpdate);

        last.num_send_WorldUpdate = 0;
        for (int i = 0; i < svc.size(); ++i) {
            last.num_send_WorldUpdate += ((Game::Network*)svc[i]->game_network)->Broadcast(svc[i], &last.WorldUpdate);
        }
        if (bots) bots->Update(timestep * updated); 
        return 0;
    }

    void RconResponseCB(Connection *c, Game::Protocol::Header *hdr, Game::Protocol::RconResponse *req) {
        Game::ConnectionData::Get(c)->retry.Acknowledged(hdr->seq);
    }
    void JoinRequestCB(Connection *c, Game::Protocol::Header *hdr, Game::Protocol::JoinRequest *req) {
        Game::ConnectionData *cd = Game::ConnectionData::Get(c);
        cd->playerName = req->PlayerName;

        bool rejoin = cd->entityID != 0;
        if (!rejoin) {
            cd->entityID = world->NewID();
            world->JoinEntity(cd, cd->entityID, &cd->team);
        }
        Entity *e = world->Get(cd->entityID);

        Game::Protocol::JoinResponse response;
        response.rcon = "set_asset";
        for (vector<Asset>::const_iterator a = assets->begin(); a != assets->end(); ++a)
            StrAppend(&response.rcon, " ", a->typeID, "=", a->name);
        response.rcon += "\n";
        world->JoinRcon(cd, e, &response.rcon);
        Write(c, UDPClient::Sendto, hdr->seq, &response);

        Game::Protocol::RconRequest rcon_broadcast;
        INFO(c->name(), ": ", cd->playerName, rejoin?" re":" ", "joins, entity_id=", e->name);
        if (world->JoinedRcon(cd, e, &rcon_broadcast.Text)) BroadcastWithRetry(&rcon_broadcast);
    }
    void PlayerUpdateCB(Connection *c, Game::Protocol::Header *hdr, Game::Protocol::PlayerUpdate *pup) {
        Game::ConnectionData *cd = Game::ConnectionData::Get(c);
        for (int i=0; i<sizeofarray(last.send_WorldUpdate); i++) {
            if (last.send_WorldUpdate[i].id != pup->id_WorldUpdate) continue;
            cd->ping = Now() - last.send_WorldUpdate[i].time - pup->time_since_WorldUpdate;
            break;
        }

        Entity *e = world->Get(cd->entityID);
        if (!e) { ERROR("missing entity ", cd->entityID); return; }
        pup->ort.To(&e->ort, &e->up);
        e->buttons = pup->buttons;

        if (!Game::Controller(e->buttons).GetPlayerList()) return;
        Game::Protocol::PlayerList playerlist;
        playerlist.Text = StrCat(world->red_score, ",", world->blue_score, "\n");
        for (int i = 0; i < svc.size(); ++i)
            for (Service::EndpointMap::iterator iter = svc[i]->endpoint.begin(); iter != svc[i]->endpoint.end(); iter++)
                AppendSerializedPlayerData(Game::ConnectionData::Get(iter->second), &playerlist.Text);
        if (bots) for (GameBots::BotVector::iterator i = bots->bots.begin(); i != bots->bots.end(); i++)
            AppendSerializedPlayerData(i->player_data, &playerlist.Text);
        Write(c, UDPClient::Sendto, cd->seq++, &playerlist);
    }
    void RconRequestCB(Connection *c, Game::Protocol::Header *hdr, Game::Protocol::RconRequest *rcon) {
        Game::Protocol::RconResponse response;
        Write(c, UDPClient::Sendto, hdr->seq, &response);

        Game::ConnectionData *cd = Game::ConnectionData::Get(c);
        StringLineIter lines(rcon->Text.c_str());
        for (const char *line = lines.Next(); line; line = lines.Next()) {
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
                Entity *e = world->Get(atoi(a1.c_str()));
                int anim_id = atoi(a2.c_str());
                if (!e || !anim_id) { WritePrintWithRetry(c, cd, "usage: anim <entity_id> <anim_id>"); continue; }
                e->animation.Start(anim_id);
            } else if (cmd == "auth") {
                if (arg == rcon_auth_passwd && rcon_auth_passwd.size()) cd->rcon_auth = true;
                WritePrintWithRetry(c, cd, StrCat("rcon auth: ", cd->rcon_auth ? "enabled" : "disabled"));
            } else if (cmd == "shutdown") {
                if (cd->rcon_auth) app->run = false;
            } else {
                RconRequestCB(c, cd, cmd, arg);
            }
        }
    }
    virtual void RconRequestCB(Connection *c, Game::ConnectionData *cd, const string &cmd, const string &arg) {
        Game::Protocol::RconRequest print(StrCat("print *** Unknown command: ", cmd));
        WriteWithRetry(c, cd, &print);
    }

    static void AppendSerializedPlayerData(const Game::ConnectionData *icd, string *out) {
        if (!icd || !icd->team || !out) return;
        StringAppendf(out, "%d,%s,%d,%d,%d\n", icd->entityID, icd->playerName.c_str(), icd->team, icd->score, icd->ping); /* XXX escape player name */
    }
};

struct GameUDPServer : public UDPServer {
    int secret1, secret2;
    GameUDPServer(int port) : UDPServer(port), secret1(::rand()), secret2(::rand()) {}

    int Hash(Connection *c) {
        int conn_key[3] = { (int)c->addr, secret1, c->port }; 
        return fnv32(conn_key, sizeof(conn_key), secret2);
    }
    int UDPFilter(Connection *c, const char *content, int content_len) {
        if (content_len < Game::Protocol::Header::size) return -1;

        Serializable::ConstStream in(content, content_len);
        Game::Protocol::Header hdr;
        hdr.In(&in);

        Game::Protocol::ChallengeRequest challenge;
        Game::Protocol::JoinRequest join;

        // echo "ping" | nc -u 127.0.0.1 27640
        static string ping="ping\n";
        if (in.size == ping.size() && in.buf == ping) {
            ((GameServer*)query)->Read(c, content, content_len);
        } else if (hdr.id == game_protocol_request_type(ChallengeRequest) && !challenge.Read(&in)) {
            Game::Protocol::ChallengeResponse response;
            response.token = Hash(c);
            string buf;
            response.ToString(&buf, hdr.seq);
            c->sendto(buf.data(), buf.size());
        } else if (hdr.id == game_protocol_request_type(JoinRequest) && !join.Read(&in)) {
            if (join.token == Hash(c)) return 0;
        } else ERROR(c->name(), ": parse failed: unknown type ", hdr.id, " bytes ", in.size);
        return 1;
    }
};

struct GameClient {
    struct History {
        unsigned buttons=0;
        Time     time_frame=0;
        Time     time_send_PlayerUpdate=0;
        Time     time_recv_WorldUpdate[2];
        unsigned short id_WorldUpdate, seq_WorldUpdate;
        deque<Game::Protocol::WorldUpdate> WorldUpdate;
        History() { time_recv_WorldUpdate[0]=time_recv_WorldUpdate[1]=0; }
    };
    struct Replay {
        Time start=0;
        unsigned short start_ind=0, while_seq=0;
        bool just_ended=0;
        bool enabled() { return start_ind; }
        void disable() { just_ended=start_ind; start_ind=0; }
    };

    string playername;
    Connection *conn=0;
    Game *world;
    GUI *playerlist;
    TextArea *chat;
    unsigned short seq=0, entity_id=0, team=0, cam=1;
    bool reorienting=1;
    Game::Network *net=0;
    Game::ReliableUDPNetwork retry;
    map<unsigned, string> assets;
    Game::Controller control;
    History last;
    Replay replay, gameover;

    ~GameClient() { Reset(); }
    GameClient(Game *w, GUI *PlayerList, TextArea *Chat) :
        world(w), playerlist(PlayerList), chat(Chat), retry(UDPClient::Write) {}

    virtual void NewEntityCB(Entity *) {}
    virtual void DelEntityCB(Entity *) {}
    virtual void SetEntityCB(Entity *, const string &k, const string &v) {}
    virtual void AnimationChange(Entity *, int NewID, int NewSeq) {}
    virtual void RconRequestCB(const string &cmd, const string &arg, int seq) {}

    Entity *WorldAddEntity(int id) { return world->Add(id, new Entity()); }
    void WorldAddEntityFinish(Entity *e, int type) {
        CHECK(!e->asset);
        world->scene->ChangeAsset(e, app->shell.asset(assets[type]));
        NewEntityCB(e);
    }
    void WorldDeleteEntity(Scene::EntityVector &v) {
        for (Scene::EntityVector::iterator i = v.begin(); i != v.end(); i++) DelEntityCB(*i);
        world->scene->Del(v);
    }
    void Rcon(const string &text) {
        if (!conn) return;
        Game::Protocol::RconRequest req(text);
        net->WriteWithRetry(&retry, conn, &req, seq++);
    }
    void MoveUp    (unsigned t) { control.SetUp();      }
    void MoveDown  (unsigned t) { control.SetDown();    }
    void MoveFwd   (unsigned t) { control.SetForward(); }
    void MoveRev   (unsigned t) { control.SetBack();    }
    void MoveLeft  (unsigned t) { control.SetLeft();    }
    void MoveRight (unsigned t) { control.SetRight();   }
    void SetCamera(const vector<string> &a) { cam = atoi(a.size() ? a[0].c_str() : ""); }
    void RconCmd  (const vector<string> &a) { if (a.size()) Rcon(a[0]); }
    void SetTeam  (const vector<string> &a) { if (a.size()) Rcon(StrCat("team ", a[0])); }
    void SetName  (const vector<string> &a) {
        string n = a.size() ? a[0] : "";
        if (playername == n) return;
        playername = n;
        Rcon(StrCat("name ", n));
        Singleton<FlagMap>::Get()->Set("player_name", n);
    }
    void MyEntityName(vector<string>) {
        Entity *e = world->Get(entity_id);
        INFO("me = ", e ? e->name : "");
    }

    void Reset() { if (conn) conn->_error(); conn=0; seq=0; entity_id=0; team=0; cam=1; reorienting=1; net=0; retry.Clear(); last.id_WorldUpdate=last.seq_WorldUpdate=0; replay.disable(); gameover.disable(); }
    bool Connected() { return conn && conn->state == Connection::Connected; }
    int Connect(const string &url, int default_port) {
        Reset();
        net = Singleton<Game::UDPNetwork>::Get();
        conn = Singleton<UDPClient>::Get()->PersistentConnection(url, bind(&GameClient::Read, this, _1, _2, _3),
                                                                      bind(&GameClient::Heartbeat, this, _1), default_port);
        if (!Connected()) return 0;

        Game::Protocol::ChallengeRequest req;
        net->WriteWithRetry(&retry, conn, &req, seq++);
        last.time_send_PlayerUpdate = Now();
        return 0;
    }
    int ConnectGPlus(const string &participant_name) {
#ifdef LFL_ANDROID
        GPlusClient *gplus_client = Singleton<GPlusClient>::Get();
        service_enable(gplus_client);
        reset();
        net = Singleton<Game::GoogleMultiplayerNetwork>::Get();
        conn = gplus_client->persistentConnection(participant_name, bind(&GameClient::UDPClientResponseCB, this, _1, _2, _3), bind(&GameClient::UDPClientHeartbeatCB, this, _1));

        Game::Protocol::JoinRequest req;
        req.PlayerName = playername;
        net->WriteWithRetry(&retry, conn, &req, seq++);
        last.time_send_PlayerUpdate = Now();
        return 0;
#else
        return -1;
#endif
    }
    void Read(Connection *c, const char *content, int content_length) {
        if (c != conn) return;
        if (!content) { INFO(c->name(), ": close"); Reset(); return; }

        Serializable::ConstStream in(content, content_length);
        Game::Protocol::Header hdr;
        hdr.In(&in);
        if (0) {}
        elif_parse(ChallengeResponse, in)
        elif_parse(JoinResponse, in)
        elif_parse(WorldUpdate, in)
        elif_parse(RconRequest, in)
        elif_parse(RconResponse, in)
        elif_parse(PlayerList, in)        
        else ERROR("parse failed: unknown type ", hdr.id);
    }
    void Heartbeat(Connection*) {
        if (!Connected()) return;
        retry.Heartbeat(conn);
        Game::Controller CS = control;
        control.Reset();
        Frame();
        if (reorienting || (CS.buttons == last.buttons && last.time_send_PlayerUpdate + 100 > Now())) return;

        Game::Protocol::PlayerUpdate pup;
        pup.id_WorldUpdate = last.id_WorldUpdate;
        pup.time_since_WorldUpdate = Now() - last.time_recv_WorldUpdate[0];
        pup.buttons = CS.buttons;
        pup.ort.From(screen->cam->ort, screen->cam->up);
        net->Write(conn, UDPClient::Write, seq++, &pup);

        last.buttons = CS.buttons;
        last.time_send_PlayerUpdate = Now();
    }
    void Frame() {
        int WorldUpdates = last.WorldUpdate.size();
        if (WorldUpdates < 2) return;

        last.time_frame = Now();
        unsigned updateInterval = last.time_recv_WorldUpdate[0] - last.time_recv_WorldUpdate[1];
        unsigned updateLast = last.time_frame - last.time_recv_WorldUpdate[0];

        if (1) { /* interpolate */
            Game::Protocol::WorldUpdate *wu1=0, *wu2=0;

            /* replay */
            if (replay.enabled() && replay.while_seq != last.seq_WorldUpdate) replay.disable();
            if (replay.enabled()) {
                float frames = min(WorldUpdates - replay.start_ind - 2.0f, (Now() - replay.start) / (float)updateInterval);
                int ind = (int)(replay.start_ind + frames);
                wu1 = &last.WorldUpdate[ind+1];
                wu2 = &last.WorldUpdate[ind];
                updateLast = (unsigned)((frames - floor(frames)) * updateInterval);
            }

            /* game over */
            if (gameover.enabled() && gameover.while_seq != last.seq_WorldUpdate) gameover.disable();

            if (!wu1) {
                wu1 = &last.WorldUpdate[WorldUpdates-1];
                wu2 = &last.WorldUpdate[WorldUpdates-2];
            }

            for (int i=0, j=0; i<wu1->entity.size() && j<wu2->entity.size(); /**/) {
                Game::Protocol::Entity *e1 = &wu1->entity[i], *e2 = &wu2->entity[j];
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
                    delta_pos.Scale(min(1000.0f, (float)updateLast) / updateInterval);
                    e->pos.Add(delta_pos);
                }

                if (e1->anim_id != e->animation.id) {
                    e->animation.Reset();
                    e->animation.id = e1->anim_id;
                    e->animation.len = e1->anim_len;
                    AnimationChange(e, e1->anim_id, e1->anim_len);
                }

                e->updated = last.time_frame;
                if (me && reorienting) { reorienting=0; e1->ort.To(&screen->cam->ort, &screen->cam->up); }
                if (me && !replay.enabled()) Me(e, cam, !reorienting);
                i++; j++;
            }
        }
    }

    void ChallengeResponseCB(Connection *c, Game::Protocol::Header *hdr, Game::Protocol::ChallengeResponse *challenge) {
        retry.Acknowledged(hdr->seq);
        Game::Protocol::JoinRequest req;
        req.token = challenge->token;
        req.PlayerName = playername;
        net->WriteWithRetry(&retry, c, &req, seq++);
    }
    void JoinResponseCB(Connection *c, Game::Protocol::Header *hdr, Game::Protocol::JoinResponse *joined) {
        retry.Acknowledged(hdr->seq);
        RconRequestCB(c, hdr, joined->rcon);
    }
    void WorldUpdateCB(Connection *c, Game::Protocol::Header *hdr, Game::Protocol::WorldUpdate *wu) {       
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
    void PlayerListCB(Connection *c, Game::Protocol::Header *hdr, Game::Protocol::PlayerList *pl) { playerlist->HandleTextMessage(pl->Text); }
    void RconResponseCB(Connection *c, Game::Protocol::Header *hdr, Game::Protocol::RconResponse*) { retry.Acknowledged(hdr->seq); }
    void RconRequestCB(Connection *c, Game::Protocol::Header *hdr, Game::Protocol::RconRequest *rcon) {
        Game::Protocol::RconResponse response;
        net->Write(c, UDPClient::Write, hdr->seq, &response);
        RconRequestCB(c, hdr, rcon->Text);
    }
    void RconRequestCB(Connection *c, Game::Protocol::Header *hdr, const string &rcon) {
        StringLineIter lines(rcon.c_str());
        for (const char *line = lines.Next(); line; line = lines.Next()) {
            if (FLAGS_rcon_debug) INFO("rcon: ", line);
            string cmd, arg;
            Split(line, isspace, &cmd, &arg);

            if (cmd == "print") {
                INFO(arg);
                if (chat) chat->Write(arg);
            } else if (cmd == "player_team") {
                team = atoi(arg.c_str());
            } else if (cmd == "player_entity") {
                entity_id = atoi(arg.c_str());
                reorienting = true;
            } else if (cmd == "set_asset") {
                vector<string> items;
                Split(arg, isspace, &items);
                for (vector<string>::const_iterator i = items.begin(); i != items.end(); ++i) {
                    string k, v;
                    Split(*i, isint<'='>, &k, &v);
                    assets[atoi(k.c_str())] = v;
                    INFO("GameAsset[", k, "] = ", v);
                }
            } else if (cmd == "set_entity") {
                vector<string> items;
                Split(arg, isspace, &items);
                for (vector<string>::const_iterator i = items.begin(); i != items.end(); ++i) {
                    vector<string> args;
                    Split(*i, isint2<'.', '='>, &args);
                    if (args.size() != 3) { ERROR("unknown arg ", *i); continue; }
                    int id = atoi(args[0].c_str());
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

    static void Me(Entity *e, int cam, bool assign_entity) {
        screen->cam->pos = e->pos;
        if (cam == 1) {
            v3 v = screen->cam->ort * 2;
            screen->cam->pos.Sub(v);
        } else if (cam == 2) {
            v3 v = screen->cam->ort * 2;
            screen->cam->pos.Sub(v);
            v = screen->cam->up * .1;
            screen->cam->pos.Add(v);
        } else if (cam == 3) {
            assign_entity = false;
        } else if (cam == 4) {
            screen->cam->pos = v3(0,3.8,0);
            screen->cam->ort = v3(0,-1,0);
            screen->cam->up  = v3(0,0,-1);
            assign_entity    = false;
        } if (assign_entity) {
            e->ort = screen->cam->ort;
            e->up  = screen->cam->up;
        }
    }
};

struct GameSettings {
    typedef ValueSet<const char *> Value;
    struct Setting {
        string key; Value *value;
        Setting(const string &k, Value *v) : key(k), value(v) {}
    };
    typedef vector<Setting> Vector;
    Vector vec;
    void SetIndex(int n, int v)  { CHECK(n < vec.size()); vec[n].value->i = v; };
    int GetIndex(int n)    const { CHECK(n < vec.size()); return vec[n].value->i; }
    const char *Get(int n) const { CHECK(n < vec.size()); return vec[n].value->Cur(); }
};

struct GameMenuGUI : public GUI, public Query {
    struct Server { string addr, name, players; };
    typedef Particles<1024, 1, true> MenuParticles;

    GUI topbar;
    GameSettings *settings=0;
    IPV4::Addr ip, broadcast_ip;
    UDPServer pinger;
    string master_get_url;
    vector<Server> master_server_list;

    Asset *title;
    Font *font, *mobile_font=0;
    Box titlewin, menuhdr, menuftr1, menuftr2;
    int default_port, selected=1, last_selected=0, sub_selected=0, last_sub_selected=0, master_server_selected=-1;
    int line_clicked=-1, decay_box_line=-1, decay_box_left=0;
    Widget::Button tab1, tab2, tab3, tab4, tab1_server_start, tab2_server_join, sub_tab1, sub_tab2, sub_tab3; 
    TextGUI tab2_server_address, tab3_player_name;
    Widget::Scrollbar tab1_options, tab2_servers, tab3_sensitivity, tab3_volume, *current_scrollbar;
#ifdef LFL_ANDROID
    Widget::Button gplus_signin_button, gplus_signout_button, gplus_quick, gplus_invite, gplus_accept;
#endif
    SimpleBrowser browser;
    MenuParticles particles;

    GameMenuGUI(LFL::Window *W, const string &master_url, int port, Asset *t=0, Asset *parts=0) :
    GUI(W), topbar(W), pinger(-1), master_get_url(master_url), title(t),
    font(Fonts::Get("Origicide.ttf", 12, Color::white)), default_port(port),
    tab1(&topbar, 0, font, "single player", MouseController::CB([&](){ if (!Typed::Changed(&selected, 1)) ToggleActive(); })),
    tab2(&topbar, 0, font, "multi player",  MouseController::CB([&](){ if (!Typed::Changed(&selected, 2)) ToggleActive(); })), 
    tab3(&topbar, 0, font, "options",       MouseController::CB([&](){ if (!Typed::Changed(&selected, 3)) ToggleActive(); })),
    tab4(&topbar, 0, font, "quit",          MouseController::CB(bind(&GameMenuGUI::MenuQuit, this))),
    tab1_server_start(this, 0, font, "start", MouseController::CB(bind(&GameMenuGUI::MenuServerStart, this))),
    tab2_server_join (this, 0, font, "join",  MouseController::CB(bind(&GameMenuGUI::MenuServerJoin,  this))),
    sub_tab1(this, 0, font, "g+",             MouseController::CB([&]() { sub_selected=1; })),
    sub_tab2(this, 0, font, "join",           MouseController::CB([&]() { sub_selected=2; })),
    sub_tab3(this, 0, font, "start",          MouseController::CB([&]() { sub_selected=3; })),
    tab2_server_address(W, font),
    tab3_player_name   (W, font),
    tab1_options    (this, menuftr1),
    tab2_servers    (this, menuftr2),
    tab3_sensitivity(this, Box(), Widget::Scrollbar::Flag::Horizontal),
    tab3_volume     (this, Box(), Widget::Scrollbar::Flag::Horizontal), current_scrollbar(0),
#ifdef LFL_ANDROID
    gplus_signin_button (this, 0, 0,    0,            MouseController::CB([&](){ android_gplus_signin(); gplus_signin_button.decay = 10; })),
    gplus_signout_button(this, 0, font, "g+ Signout", MouseController::CB([&](){ android_gplus_signout(); })),
    gplus_quick         (this, 0, font, "match" ,     MouseController::CB([&](){ android_gplus_quick_game(); })),
    gplus_invite        (this, 0, font, "invite",     MouseController::CB([&](){ android_gplus_invite(); })),
    gplus_accept        (this, 0, font, "accept",     MouseController::CB([&](){ android_gplus_accept(); })),
#endif
    browser(W, font, box), particles("GameMenuParticles") {
        tab1.outline = tab2.outline = tab3.outline = tab4.outline = tab1_server_start.outline = tab2_server_join.outline = sub_tab1.outline = sub_tab2.outline = sub_tab3.outline = &font->fg;
        Layout();
        tab2_server_address.cmd_prefix.clear();
        tab3_player_name.cmd_prefix.clear();
        tab3_player_name.deactivate_on_enter = tab2_server_address.deactivate_on_enter = true;
        tab3_player_name.runcb = bind(&TextGUI::AssignInput, (TextGUI*)&tab3_player_name, _1);
        tab3_player_name   .SetToggleKey(0, ToggleBool::OneShot);
        tab2_server_address.SetToggleKey(0, ToggleBool::OneShot);
        tab2_server_address.runcb = bind(&GameMenuGUI::MenuAddServer, this, _1);
        tab3_sensitivity.increment = .1;
        tab3_sensitivity.doc_height = 10;
        tab3_sensitivity.scrolled = FLAGS_msens / 10.0;
        tab3_volume.increment = .5;
        tab3_volume.doc_height = SystemAudio::GetMaxVolume();
        tab3_volume.scrolled = (float)SystemAudio::GetVolume() / tab3_volume.doc_height;

        sub_selected = 2;
        if (parts) {
            particles.emitter_type = MenuParticles::Emitter::Mouse | MenuParticles::Emitter::GlowFade;
            particles.texture = parts->tex.ID;
        }
#ifdef LFL_ANDROID
        mobile_font = Fonts::Get("MobileAtlas", 0, Color::black);
        gplus_signin_button.EnableHover();
#endif
        pinger.query = this;
        app->network.Enable(&pinger);
        Network::broadcast_enabled(pinger.listener()->socket, true);
        Sniffer::GetIPAddress(&ip);
        Sniffer::GetBroadcastAddress(&broadcast_ip);
    }

    void Activate  () { active=1; topbar.active=1; selected=last_selected=0; app->shell.mouseout(vector<string>()); Advertising::HideAds(); }
    void Deactivate() { active=0; topbar.active=0; UpdateSettings(); tab3_player_name.active=false; Advertising::ShowAds(); }
    bool DecayBoxIfMatch(int l1, int l2) { if (l1 != l2) return 0; decay_box_line = l1; decay_box_left = 10; return 1; }
    void UpdateSettings() {
        app->shell.Run(StrCat("name ", tab3_player_name.Text()));
        app->shell.Run(StrCat("msens ", StrCat(tab3_sensitivity.scrolled * tab3_sensitivity.doc_height)));
    }

    void MenuQuit() { selected=4; app->run=0; }
    void MenuLineClicked() { line_clicked = -MousePosition().y / font->height; }
    void MenuServerStart() {
        if (selected != 1 && !(selected == 2 && sub_selected == 3)) return;
        ToggleActive();
        app->shell.Run("local_server");
    }
    void MenuServerJoin() {
        if (selected != 2 || master_server_selected < 0 || master_server_selected >= master_server_list.size()) return;
        ToggleActive();
        app->shell.Run(StrCat("server ", master_server_list[master_server_selected].addr));
    }
    void MenuAddServer(const string &text) {
        int delim = text.find(':');
        if (delim != string::npos) Network::sendto(pinger.listener()->socket, Network::resolve(text.substr(0, delim)),
                                                   atoi(text.c_str()+delim+1), "ping\n", 5);
    }

    void Refresh() { 
        if (broadcast_ip) Network::sendto(pinger.listener()->socket, broadcast_ip, default_port, "ping\n", 5);
        if (!master_get_url.empty()) Singleton<HTTPClient>::Get()->WGet(master_get_url.c_str(), 0, bind(&GameMenuGUI::MasterGetResponseCB, this, _1, _2, _3, _4, _5));
        master_server_list.clear(); master_server_selected=-1;
    }
    void MasterGetResponseCB(Connection *c, const char *h, const string &ct, const char *cb, int cl) {
        if (!cb || !cl) return;
        string servers(cb, cl);
        StringLineIter lines(servers.c_str());
        for (const char *p, *l = lines.Next(); l; l = lines.Next()) {
            if (!(p = strchr(l, ':'))) continue;
            Network::sendto(pinger.listener()->socket, Network::addr(string(l, p-l)), atoi(p+1), "ping\n", 5);
        }
    }
    void Close(Connection *c) { c->query=0; }
    int Read(Connection *c) {
        for (int i=0; i<c->packets.size(); i++) {
            string reply(c->packets[i].buf, c->packets[i].len);
            PingResponseCB(c, reply);
        }
        return 0;
    }
    void PingResponseCB(Connection *c, const string &reply) {
        if (ip && ip == c->addr) return;
        string name, players;
        StringLineIter lines(reply.c_str());
        for (const char *p, *l = lines.Next(); l; l = lines.Next()) {
            if (!(p = strchr(l, '='))) continue;
            string k(l, p-l), v(p+1);
            if      (k == "name")    name    = v;
            else if (k == "players") players = v;
        }
        Server s = { c->name(), StrCat(name, (c->addr & broadcast_ip) == c->addr ? " (local)" : ""), players };
        master_server_list.push_back(s);
    }

    void Layout() {
        topbar.box = screen->Box(0, .95, 1, .05);
        titlewin   = screen->Box(.15, .9, .7, .05); 
        box        = screen->Box(.15, .4, .7, .5);
        menuhdr    = Box (box.x, box.y+box.h-font->height, box.w, font->height);
        menuftr1   = Box (box.x, box.y+font->height*4, box.w, box.h-font->height*5);
        menuftr2   = Box (box.x, box.y+font->height*4, box.w, box.h-font->height*6); 
        { 
            Flow topbarflow(&topbar.box, font, topbar.Reset());
            tab1.box = tab2.box = tab3.box = tab4.box = Box(topbar.box.w/4, topbar.box.h);
            tab1.Layout(&topbarflow); tab2.Layout(&topbarflow);
            tab3.Layout(&topbarflow); tab4.Layout(&topbarflow);
        }
    }
    void LayoutMenu() {
        Box b;
        Flow menuflow(&box, font, Reset());
        AddClickBox(Box(0, -box.h, box.w, box.h), MouseController::CB(bind(&GameMenuGUI::MenuLineClicked, this)));

        current_scrollbar = 0;
        if      (selected == 1) { current_scrollbar = &tab1_options;        menuflow.container = &menuftr1; }
        else if (selected == 2) { current_scrollbar = &tab2_servers;        menuflow.container = &menuftr2; }
        else if (selected == 3) { current_scrollbar = &browser.v_scrollbar; menuflow.container = &box; }
        menuflow.p.y -= current_scrollbar ? (int)(current_scrollbar->scrolled * current_scrollbar->doc_height) : 0;

        int my_selected = selected;
        if (my_selected == 2) {
            sub_tab1.box = sub_tab2.box = sub_tab3.box = Box(box.w/3, font->height);
            sub_tab1.outline = (sub_selected == 1) ? 0 : &Color::white;
            sub_tab2.outline = (sub_selected == 2) ? 0 : &Color::white;
            sub_tab3.outline = (sub_selected == 3) ? 0 : &Color::white;
            sub_tab1.Layout(&menuflow);
            sub_tab2.Layout(&menuflow);
            sub_tab3.Layout(&menuflow);
            menuflow.AppendNewline();

            if (sub_selected == 1) {
#ifdef LFL_ANDROID
                Scissor s(*menuflow.container);
                bool gplus_signedin = android_gplus_signedin();
                if (!gplus_signedin) LayoutGPlusSigninButton(&menuflow, gplus_signedin);
                else {
                    int fw = menuflow.container->w, bh = font->height;
                    if (menuflow.AppendBox(fw/3.0, bh, 0/3.0, &gplus_quick.win))  gplus_quick. Draw(true);
                    if (menuflow.AppendBox(fw/3.0, bh, 1/3.0, &gplus_invite.win)) gplus_invite.Draw(true);
                    if (menuflow.AppendBox(fw/3.0, bh, 2/3.0, &gplus_accept.win)) gplus_accept.Draw(true);
                    menuflow.AppendNewlines(1);
                }
#endif
            } else if (sub_selected == 2) {
                if (last_selected != 2 || last_sub_selected != 2) Refresh();
                {
                    Scissor s(*menuflow.container);

                    menuflow.AppendText(0,   "Server List:");
                    menuflow.AppendText(.75, "Players\n");
                    for (int i = 0; i < master_server_list.size(); ++i) {
                        if (line_clicked == menuflow.out->line.size()) master_server_selected = i;
                        if (master_server_selected == i) menuflow.out->PushBack(menuflow.CurrentLineBox(), menuflow.cur_attr, Singleton<BoxOutline>::Get());
                        menuflow.AppendText(0,   master_server_list[i].name);
                        menuflow.AppendText(.75, master_server_list[i].players);
                        menuflow.AppendNewlines(1);
                    }

                    menuflow.AppendText("\n[ add server ]");
                    if (DecayBoxIfMatch(line_clicked, menuflow.out->line.size())) {
                        TouchDevice::OpenKeyboard();
                        tab2_server_address.active = true;
                    }

                    if (tab2_server_address.active) menuflow.AppendText(.37, ":");
                    menuflow.AppendRow(.4, .6, &b);
                    menuflow.AppendNewlines(1);
                    { ScissorStack ss; tab2_server_address.Draw(b + box.TopLeft()); }
                }
                tab2_server_join.LayoutBox(&menuflow, Box(box.w*.2, -box.h*.8, box.w*.6, box.h*.1));
            } else if (sub_selected == 3) {
                my_selected = 1;
            }
        }
        if (my_selected == 1) {
            menuflow.AppendNewline();
            if (settings) {
                Scissor s(*menuflow.container);
                for (GameSettings::Vector::iterator i = settings->vec.begin(); i != settings->vec.end(); ++i) {
                    if (DecayBoxIfMatch(line_clicked, menuflow.out->line.size())) i->value->Next();
                    menuflow.AppendText(0,  i->key + ":");
                    menuflow.AppendText(.6, i->value->Cur());
                    menuflow.AppendNewlines(1);
                }
            }
            tab1_server_start.LayoutBox(&menuflow, Box(box.w*.2, -box.h*.8, box.w*.6, box.h*.1));
        }
        if (my_selected == 3) {
            Scissor s(*menuflow.container);
#ifdef LFL_ANDROID
            LayoutGPlusSigninButton(&menuflow, android_gplus_signedin());
#endif
            menuflow.AppendText("\nPlayer Name:");
            if (DecayBoxIfMatch(line_clicked, menuflow.out->line.size())) {
                TouchDevice::OpenKeyboard();
                tab3_player_name.active = true;
            }
            menuflow.AppendRow(.6, .4, &b);
            { ScissorStack ss; tab3_player_name.Draw(b + box.TopLeft()); }

            menuflow.AppendText("\nControl Sensitivity:");
            menuflow.AppendRow(.6, .35, &b);
            tab3_sensitivity.LayoutFixed(b);
            tab3_sensitivity.Update();

            menuflow.AppendText("\nVolume:");
            menuflow.AppendRow(.6, .35, &b);
            tab3_volume.LayoutFixed(b);
            tab3_volume.Update();

            if (tab3_volume.dirty) {
                tab3_volume.dirty = false;
                SystemAudio::SetVolume((int)(tab3_volume.scrolled * tab3_volume.doc_height));
            }

            menuflow.AppendNewlines(1);
            if (last_selected != 3) browser.Open("http://lucidfusionlabs.com/apps.html");
            browser.Draw(&menuflow, box.TopLeft() + menuflow.p);
        }
        if (current_scrollbar) current_scrollbar->SetDocHeight(menuflow.Height());
    }
    void LayoutGPlusSigninButton(Flow *menuflow, bool signedin) {
#ifdef LFL_ANDROID
        int bh = menuflow->font->height*2, bw = bh * 41/9.0;
        if (menuflow->AppendBox(bw, bh, (.95 - (float)bw/menuflow->frame.w)/2, &gplus_signin_button.win)) {
            if (!signedin) { 
                mobile_font->Select();
                gplus_signin_button.Draw(mobile_font, gplus_signin_button.decay ? 2 : (gplus_signin_button.hover ? 1 : 0));
            } else {
                gplus_signout_button.win = gplus_signin_button.win;
                gplus_signout_button.Draw(true);
            }
        }
        menuflow->AppendNewlines(1);
#endif
    }

    void Draw(unsigned clicks, Shader *MyShader) {
        screen->gd->EnableBlend();
        vector<Box*> bgwins;
        bgwins.push_back(&topbar.box);
        if (selected) bgwins.push_back(&box);
        glTimeResolutionShaderWindows(MyShader, Color(25, 60, 130, 120), bgwins);

        if (title && selected) {
            screen->gd->DisableBlend();
            title->tex.Draw(titlewin);
            screen->gd->EnableBlend();
            screen->gd->SetColor(font->fg);
            BoxOutline().Draw(titlewin);
        }

        LayoutMenu();
        GUI::Draw();
        topbar.Draw();
        tab3_volume.Update();
        tab3_sensitivity.Update();

        if (current_scrollbar) current_scrollbar->Update();

        if (particles.texture) {
            particles.Update(clicks, screen->mouse.x, screen->mouse.y, app->input.MouseButton1Down());
            particles.Draw();
        }

        if (selected) BoxOutline().Draw(box);
        if (decay_box_line >= 0 && decay_box_line < child_box.line.size() && decay_box_left > 0) {
            BoxOutline().Draw(child_box.line[decay_box_line] + box.TopLeft());
            decay_box_left--;
        }

        line_clicked = -1;
        last_selected = selected;
        last_sub_selected = sub_selected;
    }
};

struct GamePlayerListGUI : public GUI {
    typedef vector<string> Player;
    typedef vector<Player> PlayerList;

    Font *font;
    bool toggled=0;
    string titlename, titletext, team1, team2;
    PlayerList playerlist;
    int winning_team=0;
    GamePlayerListGUI(LFL::Window *W, const char *TitleName, const char *Team1, const char *Team2)
        : GUI(W), font(Fonts::Get("Origicide.ttf", 12, Color::black)),
        titlename(TitleName), team1(Team1), team2(Team2) {}

    void HandleTextMessage(const string &in) {
        playerlist.clear();
        StringLineIter lines(in.c_str());
        const char *hdr = lines.Next();
        string red_score_text, blue_score_text;
        Split(hdr, iscomma, &red_score_text, &blue_score_text);
        int red_score=atoi(red_score_text.c_str()), blue_score=atoi(blue_score_text.c_str());
        winning_team = blue_score > red_score ? Game::Team::Blue : Game::Team::Red;
        titletext = StrCat(titlename, " ", red_score, "-", blue_score);

        for (const char *line = lines.Next(); line; line = lines.Next()) {
            StringWordIter words(line, 0, iscomma);
            Player player;
            for (const char *word = words.Next(); word; word = words.Next()) player.push_back(word);
            if (player.size() < 5) continue;
            playerlist.push_back(player);
        }
        sort(playerlist.begin(), playerlist.end(), PlayerCompareScore);
    }
    void Draw(Shader *MyShader) {
        if (!toggled) Deactivate();
        screen->gd->EnableBlend();
        Box win = screen->Box(.1, .1, .8, .8, false);
        glTimeResolutionShaderWindows(MyShader, Color(255, 255, 255, 120), win);

        int fh = win.h/2-font->height*2;
        BoxArray outgeom1, outgeom2;
        Box out1(win.x, win.centerY(), win.w, fh), out2(win.x, win.y, win.w, fh);
        Flow menuflow1(&out1, font, &outgeom1), menuflow2(&out2, font, &outgeom2);
        for (PlayerList::iterator it = playerlist.begin(); it != playerlist.end(); it++) {
            const Player &p = (*it);
            int team = atoi(p[2].c_str());
            bool winner = team == winning_team;
            LayoutLine(winner ? &menuflow1 : &menuflow2, PlayerName(p), PlayerScore(p), PlayerPing(p));
        }
        outgeom1.Draw(out1.TopLeft());
        outgeom2.Draw(out2.TopLeft());
        font->Draw(titletext, Box(win.x, win.top()-font->height, win.w, font->height), 0, Font::Flag::AlignCenter);
    }
    void LayoutLine(Flow *flow, const string &name, const string &score, const string &ping) {
        flow->AppendText(name);
        flow->AppendText(.6, score);
        flow->AppendText(.85, ping);
        flow->AppendNewlines(1);
    }

    static bool PlayerCompareScore(const Player &l, const Player &r) { return atoi(PlayerScore(l)) > atoi(PlayerScore(r)); }
    static string PlayerName (const Player &p) { return p.size() < 5 ? "" : p[1]; }
    static string PlayerScore(const Player &p) { return p.size() < 5 ? "" : p[3]; }
    static string PlayerPing (const Player &p) { return p.size() < 5 ? "" : p[4]; }
};

struct GameChatGUI : public TextArea {
    GameClient **server;
    GameChatGUI(LFL::Window *W, int key, GameClient **s) : TextArea(W, Fonts::Get("Origicide.ttf", 10, Color::grey80)), server(s) { 
        SetToggleKey(key, ToggleBool::OneShot);
        write_timestamp=deactivate_on_enter=true;
    }

    void Run(string text) {
        if (server && *server) (*server)->Rcon(StrCat("say ", text));
        active = false;
    }
    void Draw() {
        if (!active && Now() - write_last >= Seconds(5)) return;
        screen->gd->EnableBlend(); 
        {
            int h = (int)(screen->height/1.6);
            Scissor scissor(Box(1, screen->height-h+1, screen->width, screen->height*.15, false));
            TextArea::Draw(Box(0, screen->height-h, screen->width, (int)(screen->height*.15)), active);
        }
    }
};

struct GameMultiTouchControls {
    enum { LEFT=5, RIGHT=6, UP=7, DOWN=2, UP_LEFT=8, UP_RIGHT=9, DOWN_LEFT=3, DOWN_RIGHT=4 };
    GameClient *client;
    Font *dpad_font;
    Box lpad_win, rpad_win;
    int lpad_tbx, lpad_tby, rpad_tbx, rpad_tby, lpad_down=0, rpad_down=0;
    float dp0_x=0, dp0_y=0, dp1_x=0, dp1_y=0;
    bool swipe_controls=0;

    GameMultiTouchControls(GameClient *C) : client(C),
        dpad_font(Fonts::Get("dpad_atlas", 0, Color::black)),
        lpad_win(screen->Box(.03, .05, .2, .2)),
        rpad_win(screen->Box(.78, .05, .2, .2)),
        lpad_tbx(RoundF(lpad_win.w * .6)), lpad_tby(RoundF(lpad_win.h *.6)),
        rpad_tbx(RoundF(rpad_win.w * .6)), rpad_tby(RoundF(rpad_win.h *.6)) {}

    void Draw() {
        dpad_font->Select();
        dpad_font->DrawGlyph(lpad_down, lpad_win);
        dpad_font->DrawGlyph(rpad_down, rpad_win);
    }

    void Update(unsigned clicks) {
        if (swipe_controls) {
            if (screen->gesture_dpad_stop[0]) dp0_x = dp0_y = 0;
            else if (screen->gesture_dpad_dx[0] || screen->gesture_dpad_dy[0]) {
                dp0_x = dp0_y = 0;
                if (fabs(screen->gesture_dpad_dx[0]) > fabs(screen->gesture_dpad_dy[0])) dp0_x = screen->gesture_dpad_dx[0];
                else                                                                     dp0_y = screen->gesture_dpad_dy[0];
            }

            if (screen->gesture_dpad_stop[1]) dp1_x = dp1_y = 0;
            else if (screen->gesture_dpad_dx[1] || screen->gesture_dpad_dy[1]) {
                dp1_x = dp1_y = 0;
                if (fabs(screen->gesture_dpad_dx[1]) > fabs(screen->gesture_dpad_dy[1])) dp1_x = screen->gesture_dpad_dx[1];
                else                                                                     dp1_y = screen->gesture_dpad_dy[1];
            }

            lpad_down = rpad_down = 0;
            if (FLAGS_swap_axis) {
                if (dp0_y > 0) { lpad_down = LEFT;    client->MoveLeft     (clicks); }
                if (dp0_y < 0) { lpad_down = RIGHT;   client->MoveRight    (clicks); }
                if (dp0_x < 0) { lpad_down = UP;      client->MoveFwd      (clicks); }
                if (dp0_x > 0) { lpad_down = DOWN;    client->MoveRev      (clicks); }
                if (dp1_y > 0) { rpad_down = LEFT;    screen->cam->YawLeft (clicks); }
                if (dp1_y < 0) { rpad_down = RIGHT;   screen->cam->YawRight(clicks); }
                if (dp1_x < 0) { rpad_down = UP;   /* screen->cam->MoveUp  (clicks); */ }
                if (dp1_x > 0) { rpad_down = DOWN; /* screen->cam->MoveDown(clicks); */ }
            } else {
                if (dp1_x < 0) { lpad_down = LEFT;    client->MoveLeft     (clicks); }
                if (dp1_x > 0) { lpad_down = RIGHT;   client->MoveRight    (clicks); }
                if (dp1_y < 0) { lpad_down = UP;      client->MoveFwd      (clicks); }
                if (dp1_y > 0) { lpad_down = DOWN;    client->MoveRev      (clicks); }
                if (dp0_x < 0) { rpad_down = LEFT;    screen->cam->YawLeft (clicks); } 
                if (dp0_x > 0) { rpad_down = RIGHT;   screen->cam->YawRight(clicks); }
                if (dp0_y < 0) { rpad_down = UP;   /* screen->cam->MoveUp  (clicks); */ }
                if (dp0_y > 0) { rpad_down = DOWN; /* screen->cam->MoveDown(clicks); */ }
            }
        } else {
            point l, r;
            if (FLAGS_swap_axis) {
                l.x=(int)screen->gesture_dpad_x[0]; l.y=(int)screen->gesture_dpad_y[0];
                r.x=(int)screen->gesture_dpad_x[1]; r.y=(int)screen->gesture_dpad_y[1];
            } else {
                r.x=(int)screen->gesture_dpad_x[0]; r.y=(int)screen->gesture_dpad_y[0];
                l.x=(int)screen->gesture_dpad_x[1]; l.y=(int)screen->gesture_dpad_y[1];
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

                if      ( g1 && !g2) { rpad_down = LEFT;  screen->cam->YawLeft (clicks); }
                else if (!g1 &&  g2) { rpad_down = RIGHT; screen->cam->YawRight(clicks); }
                else if ( g1 &&  g2) { rpad_down = UP;    }
                else if (!g1 && !g2) { rpad_down = DOWN;  }
            }
        }
    }
};
}; // namespace LFL

#include "lfapp/physics.h"

#endif /* __LFL_LFAPP_GAME_H__ */
