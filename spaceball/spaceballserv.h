/*
 * $Id: spaceballserv.h 1314 2014-10-16 04:43:45Z justin $
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

#ifndef __LFL_SPACEBALL_SPACEBALLSERV_H__
#define __LFL_SPACEBALL_SPACEBALLSERV_H__
namespace LFL {

struct SpaceballSettings : public GameSettings {
    static const int GAME_TYPE=0, GAME_LIMIT=1, GAME_CONTROL=2;
    static const int LIMIT_5M=0, LIMIT_10M=1, LIMIT_3G=2, LIMIT_10G=3;
    static const int TYPE_TOURNAMENT=0, TYPE_EXHIBITION=1, TYPE_EMPTYCOURT=2, TYPE_SIMULATION=3;
    static const int CONTROL_TEAM=0, CONTROL_PLAYER=1;
    static const int PLAYERS_SINGLE=0, PLAYERS_MULTIPLE=1;
    SpaceballSettings() {
        vec.push_back(Setting("Game Type",    new ValueSet<const char *>(4, "Tournament", "Exhibition", "Practice Court", "Simulation")));
        vec.push_back(Setting("Game Limit",   new ValueSet<const char *>(4, "5 Minutes", "10 Minutes", "3 Goals", "10 Goals")));
        vec.push_back(Setting("Game Control", new ValueSet<const char *>(2, "Team", "Player")));
    }
};

struct SpaceballTeam {
    string name, skybox_name; int font_index;
    Color field_color, goal_color, stripe_colors[6];
    Light light; Material ship_color;

    SpaceballTeam(const string &n, const string &sn, int fi, const Color &FC, const Color &GC, const Color &SC, const v4 &light_pos, const Color &light_color, bool stripe_color_fwd) 
        : name(n), skybox_name(sn), font_index(fi), field_color(FC), goal_color(GC) {
        light.pos = light_pos;
        ship_color.SetMaterialColor(SC);
        light.color.SetLightColor(light_color);
        float h, s, v;
        GC.ToHSV(&h, &s, &v);
        for (int i = 0; i < sizeofarray(stripe_colors); i++) {
            h += 30 * (stripe_color_fwd ? 1 : -1);
            stripe_colors[i] = Color::FromHSV(h, s, v);
        }
    }

    static SpaceballTeam *GetRandom(SpaceballTeam *filter = 0) {
        static vector<SpaceballTeam> *teams = GetList();
        for (int i=0; i<1000; i++) {
            SpaceballTeam *ret = &(*teams)[::rand() % teams->size()];
            if (ret != filter) return ret;
        }
        FATAL("no team found ", teams->size(), " ", filter?filter->name:"");
    }
    static SpaceballTeam *Get(const string &n) {
        static vector<SpaceballTeam> *teams = GetList();
        for (vector<SpaceballTeam>::iterator i = teams->begin(); i != teams->end(); ++i)
            if (i->name == n) return &(*i);
        return 0;
    }
    static vector<SpaceballTeam> *GetList() {
        static vector<SpaceballTeam> ret;
        if (ret.size()) return &ret;
        ret.push_back(SpaceballTeam("adama",      "sky19", 0,  Color("2E5B4B"), Color("4F4FD9"), Color("7373D9"), v4(  -0.20,  -1.00,   -0.40, 1), Color::white, true));
        ret.push_back(SpaceballTeam("sophia",     "sky36", 13, Color("66BCCD"), Color("3F92D2"), Color("66A3D2"), v4(  12.02,  83.63, -181.28, 1), Color::white, false));
        ret.push_back(SpaceballTeam("muria",      "sky33", 11, Color("6DAA66"), Color("FF7640"), Color("FF9B73"), v4( 166.48,  60.36,  -92.94, 1), Color::white, true));
        ret.push_back(SpaceballTeam("leonis",     "sky24", 3,  Color("66CDAA"), Color("DAFB3F"), Color("E3FB71"), v4( 138.29,  71.82,  125.36, 1), Color::white, true));
        ret.push_back(SpaceballTeam("aquaria",    "sky25", 4,  Color("3F7E68"), Color("36D792"), Color("61D7A4"), v4( 170.57,  90.23,  -52.54, 1), Color::white, false));
        ret.push_back(SpaceballTeam("tauron",     "sky32", 10, Color("2E5B4B"), Color("FFDE40"), Color("FFE773"), v4(  22.83, 194.83,  -38.93, 1), Color::white, false));
        ret.push_back(SpaceballTeam("libran",     "sky28", 7,  Color("3F7E68"), Color("9F3ED5"), Color("AD66D5"), v4(  44.18, 192.25,  -32.93, 1), Color::white, true));
        ret.push_back(SpaceballTeam("canceron",   "sky21", 1,  Color("4A947A"), Color("FF9640"), Color("FFB273"), v4(   2.22,  49.60, -193.73, 1), Color::white, false));
        ret.push_back(SpaceballTeam("newearth",   "sky23", 2,  Color("57AE64"), Color("39E639"), Color("67E667"), v4( -58.98,  54.04, -183.30, 1), Color::white, true));
        ret.push_back(SpaceballTeam("picon",      "sky37", 14, Color("3F7E48"), Color("8EF13C"), Color("A9F16C"), v4(-130.15, 127.39,  -82.63, 1), Color::white, false));
        ret.push_back(SpaceballTeam("virgon",     "sky29", 8,  Color("57AE90"), Color("33CCCC"), Color("5CCCCC"), v4(-138.87,  18.72,  142.70, 1), Color::white, true));
        ret.push_back(SpaceballTeam("gemenon",    "sky27", 6,  Color("57AE64"), Color("6A48D7"), Color("876ED7"), v4( -29.48, 112.59,  162.64, 1), Color::white, false));
        ret.push_back(SpaceballTeam("caprica",    "sky38", 15, Color("57AE90"), Color("D235D2"), Color("D25FD2"), v4(-172.27, 101.14,   -9.65, 1), Color::white, true));
        ret.push_back(SpaceballTeam("aerilon",    "sky31", 9,  Color("66CD76"), Color("FF4040"), Color("FF7373"), v4(   3.08,  51.37,  193.26, 1), Color::white, true));
        ret.push_back(SpaceballTeam("sagittaron", "sky26", 5,  Color("4A9455"), Color("F13C73"), Color("F16D95"), v4(  42.01,  94.75, -171.04, 1), Color::white, false));
        ret.push_back(SpaceballTeam("scorpia",    "sky34", 12, Color("66CD76"), Color("FFAD40"), Color("FFC373"), v4(-185.35,  59.88,  -45.35, 1), Color::white, true));
        return &ret;
    }
};

struct SpaceballMap : public MapAsset {
    SpaceballTeam *home, *away;
    SpaceballMap() : home(0), away(0) {}
    Skybox skybox;
    Asset *SkyboxAsset() { return skybox.asset(); }
    void Draw(const Entity &camera);
    void Load(const string &home_name, const string &away_name);
};

struct SpaceballGame : public Game {
    static const int WallBit=1, BallBit=2, ShipBit=4, ReplaySeconds=6, GameOverSeconds=12, PlayersPerTeam=5;
    static const int AnimShipBoost=1, AnimExplode=2;

    struct FieldDefinition {
        v3 A, B, C, D, E, F, G, H;
        static FieldDefinition *get() {
            // the "field" is a 3D box
            static FieldDefinition inst = { 
                v3(-17, -3,  27), // 0:A: back: upper left
                v3(-17, -5,  27), // 1:B: back: lower left
                v3( 17, -5,  27), // 2:C: back: lower right
                v3( 17, -3,  27), // 3:D: back: uper right
                v3(-17, -3, -27), // 4:E: front: upper left
                v3(-17, -5, -27), // 5:F: front: lower left
                v3( 17, -5, -27), // 6:G: front: lower right
                v3( 17, -3, -27)  // 7:H: front: upper right
            };
            return &inst;
        }
        float HorizontalThird(v3 x) { return 3 * (x.x + C.x) / fabs(B.x - C.x); }
        float VerticalThird(v3 x) { return 3 * (x.z + C.z) / fabs(G.z - C.z); }
        bool RedSide(v3 x) { return x.z < 0; }
    };

    struct Goals { static v3 get() { return v3(.15, 1, 1); } };

    struct Ball : public Entity {
        static float radius() { return .15; }
        static float mass() { return 12.0; }
        static v3 start_pos() { return v3(0, 0, 0); }
        EntityID last_collided_with, last_collided_with_red, last_collided_with_blue;
        Time last_collided, last_collided_red, last_collided_blue;
        Ball(Asset *a, void *b, const v3 &p, const v3 &v) : Entity("", a, p, v) { body=b; }
    };

    struct Ship : public Entity {
        Time boost; float Ysnapped;
        static float mass() { return 32.0; }
        static v3 radius() { return v3(.6, .6, Ball::radius()); }
        static bool get_boost(Game::Controller *c) { return c->Get(19); }
        static void set_boost(Game::Controller *c) { return c->Set(19); }
        float val_boost(Game::Controller *c) {
            if (!boost) { if (get_boost(c)) boost = Now(); return 0; }
            if (!get_boost(c)) { float ret = (Now() - boost); boost = 0; return ret; }
            return 0;
        }
        Ship(Asset *a, void *b, const v3 &p, const v3 &o, const v3 &u, float YSnapped=0) : Entity("", a, p, o, u), boost(0), Ysnapped(YSnapped)
        { body = b; color1 = Color::fade(Rand(0,1)); }
    };

    struct StartPositions {
        v3 red[PlayersPerTeam], blue[PlayersPerTeam];
        static StartPositions *get() {
            FieldDefinition *fd = FieldDefinition::get();
            float ypos = fd->B.y + Ship::radius().y * .7;
            static StartPositions inst = {
                v3(0,            ypos, fd->B.z * .9),
                v3(fd->B.x * .6, ypos, fd->B.z * .65),
                v3(fd->C.x * .6, ypos, fd->C.z * .65),
                v3(fd->B.x * .6, ypos, fd->B.z * .3),
                v3(fd->C.x * .6, ypos, fd->C.z * .3),

                v3(0,            ypos, fd->F.z * .9),
                v3(fd->F.x * .6, ypos, fd->F.z * .65),
                v3(fd->G.x * .6, ypos, fd->G.z * .65),
                v3(fd->F.x * .6, ypos, fd->F.z * .3),
                v3(fd->G.x * .6, ypos, fd->G.z * .3)
            };
            return &inst;
        }
    };

    static bool IsShipAssetName(const string &an) { return an.substr(0,4) == "ship"; }
    static bool IsRedShipAssetName(const string &an) { return an == "shipred"; }
    static bool IsSpectatorAssetName(const string &an) { return an == "ship"; }
    static bool IsShip(const Entity *e) { return e->asset && IsShipAssetName(e->asset->name); }
    static bool IsBall(const Entity *e) { return e->asset && e->asset->name == string("ball"); }
    static v3 StartOrientation(int team) { return v3(0,0, team == Team::Home ? -1 : 1); }
    static v3 StartPosition(int team, int *redstartindex, int *bluestartindex) {
        v3 ret;
        if (team == Team::Home) {
            ret = StartPositions::get()->red[*redstartindex];
            *redstartindex = (*redstartindex + 1) % PlayersPerTeam;
        }
        if (team == Team::Away) {
            ret = StartPositions::get()->blue[*bluestartindex];
            *bluestartindex = (*bluestartindex + 1) % PlayersPerTeam;
        }
        return ret;
    }

    Physics *physics;
    vector<Plane> planes;

    float goal_min_x, goal_max_x;
    int GoalX(int id, float x) { return (x >= goal_min_x && x <= goal_max_x) ? id : 0; }

    SpaceballTeam *home, *away;
    Time last_scored; int red_startindex, blue_startindex, game_players, game_type, game_limit, game_control;
    Callback game_finished_cb;

    SpaceballGame(Scene *s) : Game(s), last_scored(0), red_startindex(::rand() % PlayersPerTeam), blue_startindex(::rand() % PlayersPerTeam), game_players(0), game_type(0), game_limit(0), game_control(0) {
        FieldDefinition *fd = FieldDefinition::get();
        planes.push_back(Plane(fd->C, fd->B, fd->A)); // way back
        planes.push_back(Plane(fd->A, fd->B, fd->E)); // left
        planes.push_back(Plane(fd->E, fd->D, fd->A)); // top
        planes.push_back(Plane(fd->C, fd->D, fd->G)); // right
        planes.push_back(Plane(fd->B, fd->C, fd->F)); // bottom
        planes.push_back(Plane(fd->E, fd->F, fd->G)); // up front

#if defined(LFL_BULLET)
        physics = new BulletScene();
#elif defined(LFL_ODE)
        physics = new ODEScene();
#elif defined(LFL_BOX2D)
        physics = new Box2DScene(scene, b2Vec2(0, 0));
#else
        physics = new SimplePhysics(scene);
#endif
        Physics::CollidesWith cw(WallBit, BallBit);
        physics->AddPlane(Plane::Normal(fd->B, fd->C, fd->F), v3(0, fd->B.y+Ball::radius(), 0), cw); // bottom

        v3 goals = SpaceballGame::Goals::get();
        goal_min_x = goals.x * fd->B.x;
        goal_max_x = goals.x * fd->C.x;

        RandomTeams();
    }

    void Init() { NewBall(NewID()); }
    void Reset() { last_scored=0; red_score=blue_score=0; started=ranlast=Now(); state=State::GAME_ON; ResetWorld(); }
    void RandomTeams() { home = SpaceballTeam::GetRandom(); away = SpaceballTeam::GetRandom(home); }
    void AssignShipColor(Ship *ship, SpaceballTeam *team) {
        const Scene::EntityVector &other_ships = scene->assetMap[ship->asset->name];
        vector<Color> other_ship_colors, remaining_colors;
        for (Scene::EntityVector::const_iterator i = other_ships.begin(); i != other_ships.end(); ++i) other_ship_colors.push_back(((Ship*)*i)->color1);
        sort(other_ship_colors.begin(), other_ship_colors.end());
        set_difference(&team->stripe_colors[0], &team->stripe_colors[sizeofarray(team->stripe_colors)],
                       other_ship_colors.begin(), other_ship_colors.end(), inserter(remaining_colors, remaining_colors.begin()));
        if (!remaining_colors.size()) ERROR("no remaining colors");
        else ship->color1 = remaining_colors[::rand() % remaining_colors.size()];
    }

    string MapRcon() { return StrCat("map ", home->name, " ", away->name, " ", Now()-started, "\n"); }
    bool JoinRcon(ConnectionData *cd, Entity *e, string *out) { 
        StrAppend(out, MapRcon(), "set_entity");
        for (Scene::EntityAssetMap::iterator i = scene->assetMap.begin(); i != scene->assetMap.end(); i++) {
            if (!SpaceballGame::IsShipAssetName((*i).first)) continue;
            for (Scene::EntityVector::iterator j = (*i).second.begin(); j != (*i).second.end(); j++)
                StrAppend(out, " ", (*j)->name, ".color1=", (*j)->color1.HexString());
        }
        *out += "\n";
        return Game::JoinRcon(cd, e, out);
    }
    bool JoinedRcon(ConnectionData *cd, Entity *e, string *out) { 
        StrAppend(out, "set_entity ", e->name, ".color1=", e->color1.HexString(), "\n");
        return Game::JoinedRcon(cd, e, out); 
    }

    void PartEntity(ConnectionData *cd, Entity *e, TeamType team) { Game::PartEntity(cd, e, team); Del(cd->entityID); }
    Entity *JoinEntity(ConnectionData *cd, EntityID id, TeamType *team) {
        int requested_team = Team::IsTeamID(*team) ? *team : 0;
        int rp = *TeamCount(Team::Home), bp = *TeamCount(Team::Away);
        if (rp + bp >= PlayersPerTeam*2 || requested_team == Team::Spectator) {
            *team = Team::Spectator;
            return NewSpectator(id, *team);
        }

        if      (rp < bp) *team = Team::Home;
        else if (bp < rp) *team = Team::Away;
        else              *team = requested_team ? requested_team : Team::Random();

        Entity *ret = NewShip(id, *team);
        ret->userdata = cd;

        Game::JoinEntity(cd, id, team);
        return ret;
    }

    Entity *NewSpectator(EntityID id, int team) { return Add(id, new Ship(app->shell.asset("ship"), 0, v3(), v3(0,0,1), v3(0,1,0), 0)); }
    Entity *NewShip(EntityID id, int team) {
        bool red = team == Team::Home;
        v3 ort = StartOrientation(team);
        v3 pos = StartPosition(team, &red_startindex, &blue_startindex);
        void *body = physics->AddBox(Ship::radius(), pos, ort, Ship::mass(), Physics::CollidesWith(ShipBit, BallBit));
        Ship *ship = new Ship(app->shell.asset(red ? "shipred" : "shipblue"), body, pos, ort, v3(0,1,0), StartPositions::get()->blue[0].y);
        AssignShipColor(ship, red ? home : away);
        Entity *ret = Add(id, ship);
        physics->SetContinuous(ret, .001, Ship::radius().z - .001);
        return ret;
    }

    Entity *NewBall(EntityID id) {
        void *body = physics->AddSphere(Ball::radius(), Ball::start_pos(), v3(0,0,1), Ball::mass(),
                                        Physics::CollidesWith(BallBit, WallBit|ShipBit));
        Entity *ret = Add(id, new Ball(app->shell.asset("ball"), body, Ball::start_pos(), v3(0,0,1)));
        physics->SetContinuous(ret, .001, Ball::radius() - .001);
        return ret;
    }

    void ResetBall(Entity *e, int goal) {
        e->vel = v3(0, 0, 0);
        e->pos = Ball::start_pos();
        e->pos.z += goal == Team::Home ? -4 : 4;
        physics->SetPosition(e, e->pos, e->ort);
    }

    void ResetWorld(int goal=0, int points=0, bool reset_balls=true) {
        red_startindex  = ::rand() % PlayersPerTeam;
        blue_startindex = ::rand() % PlayersPerTeam;
        for (Scene::EntityAssetMap::iterator i = scene->assetMap.begin(); i != scene->assetMap.end(); i++) {
            bool ball = reset_balls && i->first == "ball";
            if (!ball && i->first != "shipred" && i->first != "shipblue") continue;
            for (Scene::EntityVector::iterator j = (*i).second.begin(); j != (*i).second.end(); j++) {
                if (ball) { ResetBall(*j, goal); continue; }
                Game::ConnectionData *cd = (Game::ConnectionData*)(*j)->userdata;
                if (cd->team == goal) cd->score += points;
                (*j)->ort = StartOrientation(cd->team);
                (*j)->pos = StartPosition(cd->team, &red_startindex, &blue_startindex);
                physics->SetPosition(*j, (*j)->pos, (*j)->ort);
            }
        }
    }

    void Update(GameServer *server, unsigned timestep) {
        if (!broadcast_enabled) {
            int wait_seconds = ReplaySeconds;
            if (state == State::GAME_OVER) wait_seconds = GameOverSeconds;
            if (Now() < last_scored + wait_seconds * 1000) return;

            if (state == State::GAME_ON && 
                ((game_limit == SpaceballSettings::LIMIT_3G  && (red_score >=  3 || blue_score >=  3)) ||
                 (game_limit == SpaceballSettings::LIMIT_10G && (red_score >= 10 || blue_score >= 10))))
                return GameOver(server);

            if (state == State::GAME_OVER) {
                if (game_finished_cb) game_finished_cb();
                else                  StartNextGame(server);
            }

            broadcast_enabled = true;
            return;
        }

        int goal = 0;
        Entity *e = 0;
        for (Scene::EntityAssetMap::iterator i = scene->assetMap.begin(); i != scene->assetMap.end() && !goal; i++)
            for (Scene::EntityVector::iterator j = (*i).second.begin(); j != (*i).second.end() && !goal; j++) {
                e = (*j);

                if (SpaceballGame::IsShipAssetName(i->first)) {
                    Ship *ship = (Ship*)e;
                    Game::Controller but(e->buttons);

                    float boost = min(1000.0f, ship->val_boost(&but));
                    if (!boost) {
                        e->vel = but.Acceleration(e->ort, e->up);
                        e->vel.Scale(5);
                    } else {
                        e->vel = e->ort;
                        e->vel.Scale(boost / 40);
                    }

                    if (boost > 200) e->animation.Start(AnimShipBoost);
                    else             e->animation.Increment();

                    if (ship->Ysnapped)
                        e->vel.y = ship->Ysnapped - e->pos.y;

                    if (IsSpectatorAssetName(i->first)) SimplePhysics::Update(e, timestep);
                    else physics->Input(e, timestep, true);
                }
                else if ((*i).first == "ball") {
                    for (int k=0; k<planes.size(); k++) {
                        float distance = planes[k].Distance(e->pos);
                        if (distance > 0) continue;
                        if      (k==5) { if (e->vel.z < 0) { if ((goal = GoalX(Team::Home, e->pos.x))) break; e->vel.z *= -1; } }
                        else if (k==0) { if (e->vel.z > 0) { if ((goal = GoalX(Team::Away, e->pos.x))) break; e->vel.z *= -1; } }
                        else if (k==1) { if (e->vel.x < 0) e->vel.x *= -1; }
                        else if (k==2) { if (e->vel.y > 0) e->vel.y *= -1; }
                        else if (k==3) { if (e->vel.x > 0) e->vel.x *= -1; }
                        // physics engine handles ground plane
                    }

                    if (goal) ResetBall(e, goal);
                    else {
                        v3 gravity(0, -0.01f, 0);
                        e->vel.Scale(0.98);
                        e->vel += gravity * timestep;
                    }

                    physics->Input(e, timestep, true);
                }
            }

        if (goal) {
            string scoredby;
            Ball *ball = (Ball*)e;
            bool red = goal == Team::Home;
            EntityID scorer_id = red ? ball->last_collided_with_red : ball->last_collided_with_blue;
            Entity *scorer = Get(scorer_id);
            if (scorer) {
                Game::ConnectionData *cd = (Game::ConnectionData*)scorer->userdata;
                scoredby = cd->playerName;
                cd->score += 10;
            }

            int *team_score = red ? &red_score : &blue_score;
            (*team_score)++;

            INFO(scoredby, " scores for ", red ? "red" : "blue");
            Game::Protocol::RconRequest cmd(StrCat("goal ", goal, " ", scoredby));
            server->BroadcastWithRetry(&cmd);

            ResetWorld(goal, 10, false);
            broadcast_enabled = false;
            last_scored = Now();
            return;
        }

        physics->Update(timestep);
        physics->Collided(false, bind(&SpaceballGame::CollidedCB, this, _1, _2, _3, _4));

        for (Scene::EntityAssetMap::iterator i = scene->assetMap.begin(); i != scene->assetMap.end(); i++) 
            for (Scene::EntityVector::iterator j = (*i).second.begin(); j != (*i).second.end(); j++) {
                Entity *e = (*j);
                physics->Output(e, timestep);
            }

        unsigned game_length = Now() - started;
        if (state == State::GAME_ON &&
            ((game_limit == SpaceballSettings::LIMIT_5M  && game_length >= Minutes(5)) ||
             (game_limit == SpaceballSettings::LIMIT_10M && game_length >= Minutes(10))))
            return GameOver(server);
    }

    void GameOver(GameServer *server) {
        INFO("game over ", red_score, " ", blue_score);
        Game::Protocol::RconRequest cmd(StrCat("win ", red_score > blue_score ? Team::Home : Team::Away));
        server->BroadcastWithRetry(&cmd);
        state = State::GAME_OVER;
        last_scored = Now();
        broadcast_enabled = false;
    }

    void StartNextGame(GameServer *server) {
        Reset();
        RandomTeams();
        INFO("change map to ", home->name, " v ", away->name);
        Game::Protocol::RconRequest cmd;
        cmd.Text = MapRcon();
        server->BroadcastWithRetry(&cmd);
    }

    void CollidedCB(const Entity *e1, const Entity *e2, int n, Physics::Contact *pt) {
        Ship *s=0; Ball *b=0;
        if (IsShip(e1)) s = (Ship*)e1;
        if (IsShip(e2)) s = (Ship*)e2;
        if (IsBall(e1)) b = (Ball*)e1;
        if (IsBall(e2)) b = (Ball*)e2;
        if (!s || !b) return;

        b->last_collided_with = GetID(s);
        b->last_collided = Now();

        if (IsRedShipAssetName(s->asset->name)) {
            b->last_collided_with_red = b->last_collided_with;
            b->last_collided_red = b->last_collided;
        } else {
            b->last_collided_with_blue = b->last_collided_with;
            b->last_collided_blue = b->last_collided; 
        }
    }
};

struct SpaceballBots : public GameBots {
    static float PossessionThreshold() { return 0.4; }
    static float PossessionThreatenedDistance() { return 0.7; }
    static float PossessionDesperationDistance() { return 0.5; }
    static float AlwaysShootDistance() { return INFINITY; }
    static int Formation() { return 1; }

    struct Role {
        enum { G=1, LD=2, CD=3, RD=4, LF=5, CF=6, RF=7 };
        static bool IsDefender(int n) { return n == LD || n == CD || n == RD; }
        static bool IsForward(int n) { return n == LF || n == CF || n == RF; }
        static v3 StartPosition(int n, bool red) {
            SpaceballGame::StartPositions *sp = SpaceballGame::StartPositions::get();
            v3 *spv = red ? sp->red : sp->blue;
            if      (n == G)  return spv[0];
            else if (n == LD) return spv[1];
            else if (n == RD) return spv[2];
            else if (n == LF) return spv[3];
            else if (n == RF) return spv[4];
            FATAL("unknown start position ", n);
        }
    };

    struct Player {
        Entity *entity, *cover;
        Game::ConnectionData *player_data;
        float dist_from_own_goal, dist_from_opp_goal;
        int role;

        Player() : entity(0), cover(0), player_data(0), dist_from_own_goal(0), dist_from_opp_goal(0), role(0) {}
        Player(Entity *e, Game::ConnectionData *cd, float d1, float d2) : entity(e), cover(0), player_data(cd), dist_from_own_goal(d1), dist_from_opp_goal(d2), role(0) {}

        static bool sort_entityid(const Player &l, const Player &r) { return l.player_data->entityID < r.player_data->entityID; }
        static bool sort_owngoaldist(const Player &l, const Player &r) { return l.dist_from_own_goal < r.dist_from_own_goal; }
        static bool sort_oppgoaldist(const Player &l, const Player &r) { return l.dist_from_opp_goal < r.dist_from_opp_goal; }
    };

    struct Team {
        Entity *closest_player_to_ball, *closest_bot_to_ball;
        float closest_player_distance_to_ball, closest_bot_distance_to_ball;
        bool possession;
        v3 goal_center;
        vector<Player> players;

        Team() { Reset(); }
        void Reset() { 
            closest_player_to_ball=closest_bot_to_ball=0;
            closest_player_distance_to_ball=closest_bot_distance_to_ball=INFINITY;
            possession=0;
            players.clear();
        }

        void AssignRoles() {
            int ind = 0;
            sort(players.begin(), players.end(), Player::sort_owngoaldist);
            if (Formation() == 1) {
                if (ind < players.size()) players[ind++].role = Role::G;
                AssignLeftRightRolePair(&ind, Role::LD, Role::RD);
                AssignLeftRightRolePair(&ind, Role::LF, Role::RF);  
            }
            sort(players.begin(), players.end(), Player::sort_oppgoaldist);
        }
        void AssignLeftRightRolePair(int *ind, int left, int right) {
            if (*ind < players.size()) {
                players[(*ind)++].role = left;
                if (*ind == players.size()) players[*ind-1].role = players[*ind-1].entity->pos.x < 0 ? left : right;
            }
            if (*ind < players.size()) {
                players[(*ind)++].role = right;
                if (players[*ind-1].entity->pos.x < players[*ind-2].entity->pos.x) swap(players[*ind-1].role, players[*ind-2].role);
            }
        }
        void AssignCoverage(Team *opponents) {
            Entity *carrier = 0;
            float not_behind_op = goal_center.z - opponents->goal_center.z;
            if (opponents->possession && opponents->closest_player_to_ball) {
                carrier = opponents->closest_player_to_ball;
                Player *nearest = 0, *nearest_not_behind = 0;
                float nearest_dist = INFINITY, nearest_not_behind_dist = INFINITY;
                for (vector<Player>::iterator i = players.begin(); i != players.end(); i++) {
                    i->cover = 0;
                    float dist = v3::Dist2(i->entity->pos, carrier->pos);
                    if (dist < nearest_dist) { nearest_dist = dist; nearest = &(*i); }
                    if (dist < nearest_not_behind_dist &&
                        ((not_behind_op <= 0 && i->entity->pos.z < carrier->pos.z) ||
                         (not_behind_op  > 0 && i->entity->pos.z > carrier->pos.z))) { nearest_not_behind_dist = dist; nearest_not_behind = &(*i); }
                }
                if (nearest_not_behind) nearest_not_behind->cover = carrier;
                else if (nearest) nearest->cover = carrier;
            }
            for (vector<Player>::iterator i = opponents->players.begin(); i != opponents->players.end(); i++) {
                if (i->entity == carrier) continue;

                Player *nearest = 0;
                float nearest_dist = INFINITY;
                for (vector<Player>::iterator j = players.begin(); j != players.end(); j++) {
                    if (j->cover) continue;
                    float dist = v3::Dist2(i->entity->pos, j->entity->pos);
                    if (dist < nearest_dist) { nearest_dist = dist; nearest = &(*j); }
                }
                if (nearest) nearest->cover = i->entity;
            }
        }
    } redteam, blueteam;

    SpaceballBots(Game *w) : GameBots(w) {
        // mark goals
        SpaceballGame::FieldDefinition *fd = SpaceballGame::FieldDefinition::get();
        redteam .goal_center = fd->B;
        blueteam.goal_center = fd->F;
        redteam .goal_center.x = fabs(fd->B.x - fd->C.x);
        blueteam.goal_center.x = fabs(fd->F.x - fd->G.x);
    }

    void Update(unsigned dt) {
        redteam.Reset();
        blueteam.Reset();
        Time now = Now();
        Scene *scene = world->scene;
        SpaceballGame::FieldDefinition *fd = SpaceballGame::FieldDefinition::get();

        // find the ships & ball
        Entity *ball=0;
        for (Scene::EntityAssetMap::iterator i = scene->assetMap.begin(); i != scene->assetMap.end(); i++) {
            for (Scene::EntityVector::iterator j = (*i).second.begin(); j != (*i).second.end(); j++) {
                if (i->first == "ball") { ball = *j; continue; }

                bool red = i->first == "shipred", blue = i->first == "shipblue";
                if (!red && !blue) continue;

                Team *team = red ? &redteam : &blueteam, *opponent = red ? &blueteam : &redteam;
                team->players.push_back(Player(*j, (Game::ConnectionData*)(*j)->userdata,
                    v3::Dist2((*j)->pos, team->goal_center),
                    v3::Dist2((*j)->pos, opponent->goal_center)));
            }
        }
        if (!ball) return;
        bool ball_on_red_side = fd->RedSide(ball->pos);

        // find who is closest
        for (Scene::EntityAssetMap::iterator i = scene->assetMap.begin(); i != scene->assetMap.end(); i++) {
            bool red = i->first == "shipred", blue = i->first == "shipblue";
            if (!red && !blue) continue;

            Team *team = red ? &redteam : &blueteam;
            for (Scene::EntityVector::iterator j = (*i).second.begin(); j != (*i).second.end(); j++) {                
                Entity *e = *j;
                float dist2 = v3::Dist2(ball->pos, e->pos);
                if (dist2 < team->closest_player_distance_to_ball) {
                    team->closest_player_distance_to_ball = dist2;
                    team->closest_player_to_ball = e;
                }
                if (e->type == Entity::Type::BOT && dist2 < team->closest_bot_distance_to_ball) {
                    team->closest_bot_distance_to_ball = dist2;
                    team->closest_bot_to_ball = e;
                }
            }
        }

        // determine possession
        bool rpp = redteam.closest_player_distance_to_ball < PossessionThreshold();
        bool bpp = blueteam.closest_player_distance_to_ball < PossessionThreshold();
        redteam.possession = rpp && redteam.closest_player_distance_to_ball < blueteam.closest_player_distance_to_ball;
        blueteam.possession = bpp && !redteam.possession;

        // Coach
        redteam.AssignRoles();
        blueteam.AssignRoles();
        redteam.AssignCoverage(&blueteam);
        blueteam.AssignCoverage(&redteam);
        vector<Player> assignments;
        for (vector<Player>::const_iterator i = redteam.players.begin(); i != redteam.players.end(); i++) assignments.push_back(*i);
        for (vector<Player>::const_iterator i = blueteam.players.begin(); i != blueteam.players.end(); i++) assignments.push_back(*i);
        sort(assignments.begin(), assignments.end(), Player::sort_entityid);
        int assignment_index = 0;

        // generalize & respond
        for (BotVector::iterator i = bots.begin(); i != bots.end(); i++) {
            Bot *b = &(*i);
            bool red = i->entity->asset->name == string("shipred");
            Team *team = red ? &redteam : &blueteam, *opponent = red ? &blueteam : &redteam;
            bool nearest = i->entity == team->closest_player_to_ball;
            bool my_ball = nearest && team->possession, fire = false;
            bool can_shoot = my_ball && now - i->last_shot > Seconds(1);
            Game::Controller buttons;

            // receive instructions
            Entity *cover = 0;
            int role = Role::LD;
            while (assignment_index < assignments.size() && assignments[assignment_index].player_data->entityID < i->player_data->entityID) assignment_index++;
            if (assignment_index < assignments.size() && assignments[assignment_index].player_data->entityID == i->player_data->entityID) {
                role = assignments[assignment_index].role;
                cover = assignments[assignment_index++].cover;
            }
            bool defender = Role::IsDefender(role);

            /* Swarm AI: A Solution to Soccer (Kutsenok) http://www.dreamspike.com/kutsenok/attachments/File/SwarmAISoccerThesis04-_Kutsenok.pdf */
            if (my_ball) {
                float goaldist = v3::Dist2(opponent->goal_center, i->entity->pos); 
                float space = opponent->closest_player_distance_to_ball;

                if (can_shoot && goaldist < AlwaysShootDistance() && (fire = Shoot(b, ball, opponent->goal_center, &buttons))) { /**/ }
                else if (space < PossessionThreatenedDistance()) {
                    if ((fire = Pass(b, ball, &buttons))) { /**/ }
                    else if (space < PossessionDesperationDistance() && (fire = Shoot(b, ball, opponent->goal_center, &buttons))) { /**/ }
                    else EscapePursuit(b, ball, opponent->closest_player_to_ball, &buttons);
                }
                else if ((fire = PassPainting(b, ball, &buttons))) { /**/ }
                else {
                    v3 dribble_target = opponent->goal_center;
                    float vertical_third = SpaceballGame::FieldDefinition::get()->VerticalThird(i->entity->pos);
                    if (vertical_third < 2 || vertical_third >= 3) dribble_target += v3(Rand(0.3, 0.6), 0, 0) * (vertical_third - 1.5);
                    Dribble(b, ball, dribble_target, &buttons);
                }
            } else {
               if (nearest) Intercept(b, ball, &buttons);
               else if (team->possession ||
                   (defender && (!cover || (cover && red != fd->RedSide(cover->pos)))))
               {
                   RubberBandMovement(b, Role::StartPosition(role, red), Role::IsForward(role), &buttons);
               }
               else {
                   fire = Cover(b, ball, cover, &buttons);
               }
            }

            if (fire) i->last_shot = Now();
            else SpaceballGame::Ship::set_boost(&buttons); // release = fire
            i->entity->buttons = buttons.buttons;
        }
    }
    void Intercept(Bot *b, Entity *ball, Game::Controller *buttons) {
        v3 dir = ball->pos - b->entity->pos;
        dir.y = 0;
        dir.Norm();
        b->entity->ort = dir;
        buttons->SetForward();
    }
    bool Pass(Bot *b, Entity *ball, Game::Controller *buttons) {
        return false;
    }
    bool PassPainting(Bot *b, Entity *ball, Game::Controller *buttons) {
        return false;
    }
    void Dribble(Bot *b, Entity *ball, v3 dribble_target, Game::Controller *buttons) {
    }
    bool Shoot(Bot *b, Entity *ball, v3 goal_center, Game::Controller *buttons) {
        SpaceballGame *spaceball = (SpaceballGame*)world;
        v3 targ = goal_center;
        targ.x += fabs(spaceball->goal_max_x - spaceball->goal_min_x) / 2 * Rand(-1.0, 1.0);
        b->entity->ort = targ - b->entity->pos;
        b->entity->ort.Norm();
        return true;
    }
    bool Cover(Bot *b, Entity *ball, Entity *target, Game::Controller *buttons) {
        return false;
    }
    void EscapePursuit(Bot *b, Entity *ball, Entity *pursuer, Game::Controller *buttons) {
    }
    void RubberBandMovement(Bot *b, v3 zone, bool forward, Game::Controller *buttons) {
    }
};

struct SpaceballServer : public GameServer {
    Scene scene;
    GameUDPServer *udp_transport;
    GPlusServer *gplus_transport;

    SpaceballGame *World() { return (SpaceballGame*)world; }
    SpaceballBots *Bots() { return (SpaceballBots*)bots; }
    SpaceballServer(const string &name, int port, int framerate, const vector<Asset> *assets)
        : GameServer(new SpaceballGame(&scene), 1000/framerate, name, StrCat(port), assets)
    {
        World()->Init();

        udp_transport = new GameUDPServer(port);
        udp_transport->game_network = Singleton<Game::UDPNetwork>::Get();
        udp_transport->query = this;
        svc.push_back(udp_transport);

#ifdef LFL_ANDROID
        gplus_transport = new GPlusServer();
        gplus_transport->game_network = Singleton<Game::GoogleMultiplayerNetwork>::Get();
        gplus_transport->query = this;
        svc.push_back(gplus_transport);
#endif
    }
    void RconRequestCB(Connection *c, Game::ConnectionData *cd, const string &cmd, const string &arg) {
        SpaceballGame *world = World();
        SpaceballBots *bots = Bots();
        if (cmd == "player_switch" && bots && (cd->team == Game::Team::Home || cd->team == Game::Team::Away)) {
            SpaceballBots::Team *team = (cd->team == Game::Team::Home ? &bots->redteam : &bots->blueteam);
            Entity *old_player_entity = world->Get(cd->entityID);
            if (world->game_control != SpaceballSettings::CONTROL_TEAM || 
                !team->closest_bot_to_ball /* || team->closest_player_to_ball == old_player_entity */) return;

            for (int i=0; i<bots->bots.size(); ++i) { 
                Entity *nearest = bots->bots[i].entity;
                if (nearest != team->closest_bot_to_ball) continue;

                swap(old_player_entity->type, nearest->type);
                swap(old_player_entity->userdata, nearest->userdata);
                ((Game::ConnectionData*)old_player_entity->userdata)->entityID = cd->entityID;
                bots->bots[i].entity = old_player_entity;
                cd->entityID = Game::GetID(nearest);

                Game::Protocol::RconRequest cmd;
                cmd.Text = StrCat("player_entity ", cd->entityID);
                WriteWithRetry(c, cd, &cmd);
                break;
            }
        } else {
            GameServer::RconRequestCB(c, cd, cmd, arg);
        }
    }
};

}; // namespace LFL
#endif // __LFL_SPACEBALL_SPACEBALLSERV_H__
