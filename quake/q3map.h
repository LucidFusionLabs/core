/*
 * $Id: q3map.h 1309 2014-10-10 19:20:55Z justin $
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

#ifndef __LFL_QUAKE_Q3MAP_H__
#define __LFL_QUAKE_Q3MAP_H__
namespace LFL {

struct Q3MapAsset : public MapAsset {
    enum eLumps {
        kEntities = 0,     // Stores player/object positions, etc...
        kTextures,         // Stores texture information
        kPlanes,           // Stores the splitting planes
        kNodes,            // Stores the BSP nodes
        kLeafs,            // Stores the leafs of the nodes
        kLeafFaces,        // Stores the leaf's indices into the faces
        kLeafBrushes,      // Stores the leaf's indices into the brushes
        kModels,           // Stores the info of world models
        kBrushes,          // Stores the brushes info (for collision)
        kBrushSides,       // Stores the brush surfaces info
        kVertices,         // Stores the level vertices
        kMeshVerts,        // Stores the model vertices offsets
        kShaders,          // Stores the shader files (blending, anims..)
        kFaces,            // Stores the faces for the level
        kLightmaps,        // Stores the lightmaps for the level
        kLightVolumes,     // Stores extra world lighting information
        kVisData,          // Stores PVS and cluster info (visibility)
        kMaxLumps          // A constant to store the number of lumps
    };

    struct tBSPHeader {
        char strID[4];     // This should always be 'IBSP'
        int version;       // This should be 0x2e for Quake 3 files
    } hdr;

    struct tBSPLump {
        int offset;
        int length;
    } lumps[kMaxLumps];

    struct tBSPVertex {
        float vPosition[3];      // (x, y, z) position. 
        float vTextureCoord[2];  // (u, v) texture coordinate
        float vLightmapCoord[2]; // (u, v) lightmap coordinate
        float vNormal[3];        // (x, y, z) normal vector
        unsigned char color[4];  // RGBA color for the vertex 
    } *vertex;

    struct tBSPFace {
        int textureID;        // The index into the texture array 
        int effect;           // The index for the effects (or -1 = n/a) 
        int type;             // 1=polygon, 2=patch, 3=mesh, 4=billboard 
        int vertexIndex;      // The index into this face's first vertex 
        int numOfVerts;       // The number of vertices for this face 
        int meshVertIndex;    // The index into the first meshvertex 
        int numMeshVerts;     // The number of mesh vertices 
        int lightmapID;       // The texture index for the lightmap 
        int lMapCorner[2];    // The face's lightmap corner in the image 
        int lMapSize[2];      // The size of the lightmap section 
        float lMapPos[3];     // The 3D origin of lightmap. 
        float lMapBitsets[2][3]; // The 3D space for s and t unit vectors. 
        float vNormal[3];     // The face normal. 
        int size[2];          // The bezier patch dimensions. 
    } *face;

    struct tBSPTexture {
        char strName[64];   // The name of the texture w/o the extension 
        int flags;          // The surface flags (unknown) 
        int contents;       // The content flags (unknown)
    } *texture;

    struct tBSPLightmap {
        unsigned char imageBits[128][128][3]; // The RGB data in a 128x128 image
    };

    struct tBSPNode {
        int plane;      // The index into the planes array 
        int front;      // The child index for the front node 
        int back;       // The child index for the back node 
        int mins[3];    // The bounding box min position. 
        int maxs[3];    // The bounding box max position. 
    } *node;

    struct tBSPLeaf {
        int cluster;           // The visibility cluster 
        int area;              // The area portal 
        int mins[3];           // The bounding box min position 
        int maxs[3];           // The bounding box max position 
        int leafface;          // The first index into the face array 
        int numOfLeafFaces;    // The number of faces for this leaf 
        int leafBrush;         // The first index for into the brushes 
        int numOfLeafBrushes;  // The number of brushes for this leaf 
    } *leaf;

    struct tBSPPlane {
        float vNormal[3];     // Plane normal. 
        float d;              // The plane distance from origin 
    } *plane;

    struct tBSPVisData {
        int numOfClusters;       // The number of clusters
        int bytesPerCluster;	 // Bytes (8 bits) in the cluster's bitset
        unsigned char *pBitsets; // Array of bytes holding the cluster vis.
    } *visdata;

    char *strEntities;

    struct tBSPBrush {
        int brushSide;           // The starting brush side for the brush 
        int numOfBrushSides;     // Number of brush sides for the brush
        int textureID;           // The texture index for the brush
    } *brush;

    struct tBSPBrushSide {
        int plane;              // The plane index
        int textureID;          // The texture index
    } *brushside;

    struct tBSPModel {
        float min[3];           // The min position for the bounding box
        float max[3];           // The max position for the bounding box. 
        int faceIndex;          // The first face index in the model 
        int numOfFaces;         // The number of faces in the model 
        int brushIndex;         // The first brush index in the model 
        int numOfBrushes;       // The number brushes for the model
    } *model;

    struct tBSPShader {
        char strName[64];     // The name of the shader file 
        int brushIndex;       // The brush index for this shader 
        int unknown;          // This is 99% of the time 5
    } *shader;

    struct tBSPLights {
        unsigned char ambient[3];       // This is the ambient color in RGB
        unsigned char directional[3];	// This is the directional color in RGB
        unsigned char direction[2];     // The direction of the light: [phi,theta] 
    } *lights;

    static const int LightMapSize = 128*128*3;

    BlockChainAlloc alloc;
    void *out[kMaxLumps];
    int num[kMaxLumps], *leaffaces, *leafbrushes, *meshverts, vert_id, ind_id, vert_size, ind_size, norm_offset, tex_offset, lightmap_offset;
    Asset *asset, *lightmap;

    static Q3MapAsset *Load(const string& fn) {
        Q3MapAsset *ret = new Q3MapAsset();

        { // first pass to find the .bsp
            ArchiveIter iter(fn.c_str());
            for (const char *afn = iter.Next(); Running() && afn; afn = iter.Next()) {
                if (!SuffixMatch(afn, ".bsp", false)) { iter.Skip(); continue; }
                BufferFile bf(string((const char *)iter.Data(), iter.Size()));
                if (ret->LoadBSP(&bf)) { delete ret; return 0; }
                break;
            }
        }

        { // second pass for rest of the files
            ArchiveIter iter(fn.c_str());
            for (const char *afnp = iter.Next(); Running() && afnp; afnp = iter.Next()) {
                string afn = afnp, afp = afn.substr(0, afn.rfind('.'));
                for (int i=0, l=ret->num[kTextures]; i<l; i++) if (afp == ret->texture[i].strName) {
                    ret->asset[i].Load(iter.Data(), afn.c_str(), iter.Size());
                    INFO("load[", i, "] ", afn, " tex=", ret->asset[i].tex.ID);
                }
            }
        }

        return ret;
    }

    int LoadBSP(File *f) {
        if (!File::ReadSuccess(f, &hdr, sizeof(hdr))) return -1;
        if (memcmp(hdr.strID, "IBSP", 4)) { ERROR("hdr ", hdr.strID[0], hdr.strID[1], hdr.strID[2], hdr.strID[3]); return -1; }
        if (hdr.version != 0x2e	&& hdr.version != 0x2f) { ERROR("ver ", hdr.version); return -1; }
        if (!File::ReadSuccess(f, lumps, sizeof(lumps))) return -1;

        num[kVertices]     = lumps[kVertices]    .length / sizeof(tBSPVertex);
        num[kFaces]        = lumps[kFaces]       .length / sizeof(tBSPFace);
        num[kTextures]     = lumps[kTextures]    .length / sizeof(tBSPTexture);
        num[kLightmaps]    = lumps[kLightmaps]   .length / sizeof(tBSPLightmap);
        num[kNodes]        = lumps[kNodes]       .length / sizeof(tBSPNode);
        num[kLeafs]        = lumps[kLeafs]       .length / sizeof(tBSPLeaf);
        num[kLeafFaces]    = lumps[kLeafFaces]   .length / sizeof(int);
        num[kPlanes]       = lumps[kPlanes]      .length / sizeof(tBSPPlane);
        num[kVisData]      = lumps[kVisData]     .length - sizeof(int)*2;
        num[kEntities]     = lumps[kEntities]    .length;
        num[kBrushes]      = lumps[kBrushes]     .length / sizeof(tBSPBrush);
        num[kLeafBrushes]  = lumps[kLeafBrushes] .length / sizeof(int);
        num[kBrushSides]   = lumps[kBrushSides]  .length / sizeof(tBSPBrushSide);
        num[kModels]       = lumps[kModels]      .length / sizeof(tBSPModel);
        num[kMeshVerts]    = lumps[kMeshVerts]   .length / sizeof(int);
        num[kShaders]      = lumps[kShaders  ]   .length / sizeof(tBSPShader);
        num[kLightmaps]    = lumps[kLightmaps]   .length / LightMapSize;
        num[kLightVolumes] = lumps[kLightVolumes].length / sizeof(tBSPLights);

        int map_assets = num[kTextures] + num[kLightmaps];
        asset = (Asset*)alloc.Malloc(sizeof(Asset) * map_assets);
        lightmap = asset + num[kTextures];
        for (int i = 0; i < map_assets; ++i) new((void*)&asset[i]) Asset();

        out[kVertices]     = vertex      =    (tBSPVertex*)alloc.Malloc(lumps[kVertices]    .length);
        out[kFaces]        = face        =      (tBSPFace*)alloc.Malloc(lumps[kFaces]       .length);
        out[kTextures]     = texture     =   (tBSPTexture*)alloc.Malloc(lumps[kTextures]    .length);
        out[kLightmaps]                  =  (tBSPLightmap*)alloc.Malloc(lumps[kLightmaps]   .length);
        out[kNodes]        = node        =      (tBSPNode*)alloc.Malloc(lumps[kNodes]       .length);
        out[kLeafs]        = leaf        =      (tBSPLeaf*)alloc.Malloc(lumps[kLeafs]       .length);
        out[kLeafFaces]    = leaffaces   =           (int*)alloc.Malloc(lumps[kLeafFaces]   .length);
        out[kPlanes]       = plane       =     (tBSPPlane*)alloc.Malloc(lumps[kPlanes]      .length);
        out[kVisData]      = visdata     =   (tBSPVisData*)alloc.Malloc(lumps[kVisData]     .length);
        out[kEntities]     = strEntities =          (char*)alloc.Malloc(lumps[kEntities]    .length);
        out[kBrushes]      = brush       =     (tBSPBrush*)alloc.Malloc(lumps[kBrushes]     .length);
        out[kLeafBrushes]  = leafbrushes =           (int*)alloc.Malloc(lumps[kLeafBrushes] .length);
        out[kBrushSides]   = brushside   = (tBSPBrushSide*)alloc.Malloc(lumps[kBrushSides]  .length);
        out[kModels]       = model       =     (tBSPModel*)alloc.Malloc(lumps[kModels]      .length);
        out[kMeshVerts]    = meshverts   =           (int*)alloc.Malloc(lumps[kMeshVerts]   .length);
        out[kShaders]      = shader      =    (tBSPShader*)alloc.Malloc(lumps[kShaders]     .length);
        out[kLightVolumes] = lights      =    (tBSPLights*)alloc.Malloc(lumps[kLightVolumes].length);

        visdata->pBitsets = ((unsigned char*)visdata) + sizeof(int)*2;
        vert_id = ind_id = -1;
        vert_size = num[kVertices] * sizeof(tBSPVertex);
        ind_size = num[kMeshVerts] * sizeof(int);
        norm_offset     = (char*)&vertex[0].vNormal[0]        - (char*)&vertex[0].vPosition[0];
        tex_offset      = (char*)&vertex[0].vTextureCoord[0]  - (char*)&vertex[0].vPosition[0];
        lightmap_offset = (char*)&vertex[0].vLightmapCoord[0] - (char*)&vertex[0].vPosition[0];

        for (int i = 0; i < kMaxLumps; i++)
            if (!File::SeekReadSuccess(f, lumps[i].offset, out[i], lumps[i].length)) return -1;

        for (int i = 0; i < num[kLightmaps]; i++) {
            const unsigned char *src = (const unsigned char*)out[kLightmaps] + LightMapSize * i;
            lightmap[i].tex.LoadGL(src, point(128, 128), Pixel::RGB24, 128*3);
            INFO("lm[", i, "] tex=", lightmap[i].tex.ID);
        }

        /* globalize vertex indices */
        for (int i=0, l=num[kFaces]; i<l; i++) {
            tBSPFace *face_i = &face[i];
            for (int j=0, l2=face_i->numMeshVerts; j<l2; j++) {
                meshverts[face_i->meshVertIndex + j] += face_i->vertexIndex;
            }
        }

        INFO("loaded q3 map ", f->Filename());
        return 0;
    }

    tBSPLeaf *FindLeaf(v3 pos) {
        int id = 0;
        while (id >= 0) {
            if (id >= num[kNodes]) return 0;
            tBSPNode *n = &node[id];
            Plane *p = (Plane*)&plane[n->plane];
            id = (p->Distance(pos, false) >= 0) ? n->front : n->back;
        }
        id = ~id;
        if (id < 0 || id >= num[kLeafs]) return 0;
        return &leaf[id];
    }

    void Draw(const Entity &camera) {
        tBSPLeaf *cam_leaf = FindLeaf(camera.pos);
        if (!cam_leaf) FATAL("no leaf: ", cam_leaf);

        screen->gd->EnableTexture();
        screen->gd->EnableLighting();
        screen->gd->EnableDepthTest();

        vector<int> visible_faces;
        set<int> seen_faces;
        for (int i=0, l=num[kLeafs]; i<l; i++) {
            tBSPLeaf *leaf_i = &leaf[i];
            if (!IsClusterVisible(visdata, cam_leaf->cluster, leaf_i->cluster)) continue;
            AddFaces(leaf_i, leaffaces, &seen_faces, &visible_faces);
        }
        for (vector<int>::iterator i = visible_faces.begin(); i != visible_faces.end(); i++) Draw(*i);
    }

    void Draw(int face_index) {
        tBSPFace *face_i = &face[face_index];
        if (face_i->textureID >= 0) {
            if (face_i->textureID >= num[kTextures]) FATAL("overflow ", face_i->textureID, " ", num[kTextures]);
            Scene::Select(&asset[face_i->textureID]);
        }

        if (face_i->numMeshVerts) {
            screen->gd->VertexPointer(3, GraphicsDevice::Float, sizeof(tBSPVertex), 0,           (float*)vertex, vert_size, &vert_id, false);
            screen->gd->NormalPointer(3, GraphicsDevice::Float, sizeof(tBSPVertex), norm_offset, (float*)vertex, vert_size, &vert_id, false);

#ifndef LFL_GLES2
            screen->gd->ActiveTexture(1);        
            if (face_i->lightmapID >= 0) {
                Scene::Select(&lightmap[face_i->lightmapID]);
                screen->gd->TexPointer(2, GraphicsDevice::Float, sizeof(tBSPVertex), lightmap_offset, (float*)vertex, vert_size, &vert_id, false);
            } else {
                screen->gd->DisableTexture();
            }
            screen->gd->ActiveTexture(0);
#endif

            screen->gd->TexPointer(2, GraphicsDevice::Float, sizeof(tBSPVertex), tex_offset, (float*)vertex, vert_size, &vert_id, false);
#if !defined(LFL_IPHONE) && !defined(LFL_ANDROID)
            screen->gd->Draw(GraphicsDevice::Triangles, face_i->numMeshVerts, GraphicsDevice::UnsignedInt, face_i->meshVertIndex * sizeof(int), meshverts, ind_size, &ind_id, false);
#endif
        }
    }

    static bool IsClusterVisible(tBSPVisData *pPVS, int current, int test) {
        if (current < 0) return true;
        if (!pPVS->pBitsets || test < 0 || current >= pPVS->numOfClusters || test >= pPVS->numOfClusters) FATAL("visible ", current, " ", test);
        return pPVS->pBitsets[(current*pPVS->bytesPerCluster) + (test/8)] & (1 << (test & 7));
    }

    static void AddFaces(tBSPLeaf *leaf_i, int *leaffaces, set<int> *seen_faces, vector<int> *visible_faces) {
        for (int i=0, l=leaf_i->numOfLeafFaces, faceind=leaf_i->leafface; i<l; i++) {
            int face_ind = leaffaces[faceind+i];
            if (!seen_faces->insert(face_ind).second) continue;
            visible_faces->push_back(face_ind);
        }
    }
};

}; // namespace LFL
#endif // __LFL_QUAKE_Q3MAP_H__
