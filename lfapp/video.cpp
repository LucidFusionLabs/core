/*
 * $Id: video.cpp 1336 2014-12-08 09:29:59Z justin $
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

#ifdef LFL_OPENCV
#include "opencv/cxcore.h"
#endif

#include "lfapp/lfapp.h"
#include "lfapp/dom.h"
#include "lfapp/css.h"
#include "lfapp/flow.h"
#include "lfapp/gui.h"

#if defined(LFL_GLEW) && !defined(LFL_HEADLESS)
#define glewGetContext() ((GLEWContext*)screen->glew_context)
#include <GL/glew.h>
#endif

#define LFL_GLSL_SHADERS
#if (defined(LFL_ANDROID) || defined(LFL_IPHONE)) && !defined(LFL_GLES2)
#undef LFL_GLSL_SHADERS
#endif

#if defined(LFL_HEADLESS)
#define GLint int
#define GLuint unsigned 
#define glFlush(x)
#define glGetError() 0
#define glGetString(a) ""
#define glReadPixels(a,b,c,d,e,f,g)
#define glTexImage2D(a,b,c,d,e,f,g,h,i)
#define glTexSubImage2D(a,b,c,d,e,f,g,h,i)
#define glGetTexImage(a,b,c,d,e)
#define glGetTexLevelParameteriv(a,b,c,d)
#define glTexParameteri(a,b,c)
#define glGenRenderbuffers(a,b)
#define glGenFramebuffers(a,b)
#define glBindRenderbuffer(a,b)
#define glBindFramebuffer(a,b)
#define glRenderbufferStorage(a,b,c,d)
#define glFramebufferRenderbuffer(a,b,c,d)
#define glFramebufferTexture2D(a,b,c,d,e)
#define glCheckFramebufferStatus(a) 0
#define GL_FRAMEBUFFER 0
#define GL_LUMINANCE 0
#define GL_LUMINANCE_ALPHA 0
#define GL_RGB 0
#define GL_RGBA 0
#define GL_TEXTURE_2D 0
#define GL_TEXTURE_WIDTH 0
#define GL_TEXTURE_CUBE_MAP 0
#define GL_TEXTURE_CUBE_MAP_NEGATIVE_X 0
#define GL_TEXTURE_CUBE_MAP_NEGATIVE_Y 0
#define GL_TEXTURE_CUBE_MAP_NEGATIVE_Z 0
#define GL_TEXTURE_CUBE_MAP_POSITIVE_X 0
#define GL_TEXTURE_CUBE_MAP_POSITIVE_Y 0
#define GL_TEXTURE_CUBE_MAP_POSITIVE_Z 0
#define GL_UNSIGNED_BYTE 0
#define GL_FRAGMENT_SHADER 0
#define GL_ACTIVE_UNIFORMS 0
#define GL_ACTIVE_ATTRIBUTES 0
#define GL_FRAMEBUFFER_COMPLETE 0
#define GL_VERTEX_SHADER 0
#define GL_MAX_VERTEX_UNIFORM_COMPONENTS 0
#define GL_MAX_VERTEX_ATTRIBS 0
#define GL_DEPTH_COMPONENT16 0

#elif defined(LFL_IPHONE)
#ifdef LFL_GLES2
#include <OpenGLES/ES2/gl.h>
#include <OpenGLES/ES2/glext.h>
#endif
#include <OpenGLES/ES1/gl.h>
#include <OpenGLES/ES1/glext.h>
#define glOrtho glOrthof 
#define glFrustum glFrustumf 

#elif defined(LFL_ANDROID)
#ifdef LFL_GLES2
#include <GLES2/gl2.h>
#include <GLES2/gl2ext.h>
#endif
#include <GLES/gl.h>
#include <GLES/glext.h>
#define glOrtho glOrthof 
#define glFrustum glFrustumf 

#elif defined(__APPLE__)
#include <OpenGL/glu.h>

#else
#include <GL/glu.h>
#endif

extern "C" {
#ifdef LFL_FFMPEG
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>
#endif
};

#ifdef LFL_QT
#include <QtOpenGL>
#include <QApplication>
#undef main
static QApplication *lfl_qapp;
static vector<string> lfl_argv;  
extern "C" int LFLQTMain(int argc, const char *argv[]);
extern "C" int      main(int argc, const char *argv[]) {
    for (int i=0; i<argc; i++) lfl_argv.push_back(argv[i]);
    QApplication app(argc, (char**)argv);
    lfl_qapp = &app;
    LFL::Window::Create(LFL::screen);
    return app.exec();
}
#endif // LFL_QT

#ifdef LFL_GLFWVIDEO
#include "GLFW/glfw3.h"
#endif

#ifdef LFL_SDLVIDEO
#include "SDL.h"
#if defined(LFL_ANDROID)
extern "C" {
#include "SDL_androidvideo.h"
};
#endif
#endif

namespace LFL {
const int Texture::CoordMinX = 0;
const int Texture::CoordMinY = 1;
const int Texture::CoordMaxX = 2;
const int Texture::CoordMaxY = 3;

Color Color::white (1.0, 1.0, 1.0);
Color Color::black (0.0, 0.0, 0.0);
Color Color::red   (1.0, 0.0, 0.0);
Color Color::green (0.0, 1.0, 0.0);
Color Color::blue  (0.0, 0.0, 1.0);
Color Color::cyan  (0.0, 1.0, 1.0);
Color Color::yellow(1.0, 1.0, 0.0);
Color Color::magenta(1.0, 0.0, 1.0);
Color Color::grey90(.9, .9, .9);
Color Color::grey80(.8, .8, .8);
Color Color::grey70(.7, .7, .7);
Color Color::grey60(.6, .6, .6);
Color Color::grey50(.5, .5, .5);
Color Color::grey40(.4, .4, .4);
Color Color::grey30(.3, .3, .3);
Color Color::grey20(.2, .2, .2);
Color Color::grey10(.1, .1, .1);
Color Color::clear(0.0, 0.0, 0.0, 0.0);

#ifdef LFL_GLES2
DEFINE_int(request_gles_version, 2, "OpenGLES version");
#else
DEFINE_int(request_gles_version, 1, "OpenGLES version");
#endif

DEFINE_bool(gd_debug, false, "Debug graphics device");
DEFINE_float(rotate_view, 0, "Rotate view by angle");
DEFINE_float(field_of_view, 45, "Field of view");
DEFINE_float(near_plane, 1, "Near clipping plane");
DEFINE_float(far_plane, 100, "Far clipping plane");
DEFINE_int(dots_per_inch, 75, "Screen DPI");
DEFINE_bool(swap_axis, false," Swap x,y axis");

#ifndef LFL_HEADLESS
#ifdef LFL_GDDEBUG
#define GDDebug(...) { screen->gd->CheckForError(__FILE__, __LINE__); if (FLAGS_gd_debug) INFO(__VA_ARGS__); }
#else 
#define GDDebug(...)
#endif
const int GraphicsDevice::Float            = GL_FLOAT;
const int GraphicsDevice::Points           = GL_POINTS;
const int GraphicsDevice::Lines            = GL_LINES;
const int GraphicsDevice::LineLoop         = GL_LINE_LOOP;
const int GraphicsDevice::Triangles        = GL_TRIANGLES;
const int GraphicsDevice::TriangleStrip    = GL_TRIANGLE_STRIP;
const int GraphicsDevice::Texture2D        = GL_TEXTURE_2D;
const int GraphicsDevice::UnsignedInt      = GL_UNSIGNED_INT;
const int GraphicsDevice::Ambient          = GL_AMBIENT;
const int GraphicsDevice::Diffuse          = GL_DIFFUSE;
const int GraphicsDevice::Specular         = GL_SPECULAR;
const int GraphicsDevice::Position         = GL_POSITION;
const int GraphicsDevice::Emission         = GL_EMISSION;
const int GraphicsDevice::One              = GL_ONE;
const int GraphicsDevice::SrcAlpha         = GL_SRC_ALPHA;
const int GraphicsDevice::OneMinusSrcAlpha = GL_ONE_MINUS_SRC_ALPHA;
const int GraphicsDevice::OneMinusDstColor = GL_ONE_MINUS_DST_COLOR;
const int GraphicsDevice::Fill             = GL_FILL;
const int GraphicsDevice::Line             = GL_LINE;
const int GraphicsDevice::Point            = GL_POINT;

struct OpenGLES1 : public GraphicsDevice {
    int target_matrix;
    OpenGLES1() : target_matrix(-1) { default_color.push_back(Color(1.0, 1.0, 1.0, 1.0)); }
    void Init() {
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glEnableClientState(GL_VERTEX_ARRAY);
#if !defined(LFL_IPHONE) && !defined(LFL_ANDROID)
        float black[]={0,0,0,1};
        glLightModelfv(GL_LIGHT_MODEL_AMBIENT, black);
        glLightModeli(GL_LIGHT_MODEL_LOCAL_VIEWER, 1);
        // glLightModelf(GL_LIGHT_MODEL_COLOR_CONTROL, GL_SEPARATE_SPECULAR_COLOR);
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
#endif
        GDDebug("Init");
    }
    void UpdateColor() { const Color &c = default_color.back(); glColor4f(c.r(), c.g(), c.b(), c.a()); }
    bool ShaderSupport() {
#if defined(LFL_ANDROID) || defined(LFL_IPHONE)
        return false;
#endif
        const char *ver = (const char*)glGetString(GL_VERSION);
        return ver && *ver == '2';
    }
    void  EnableTexture() {  glEnable(GL_TEXTURE_2D);  glEnableClientState(GL_TEXTURE_COORD_ARRAY); GDDebug("Texture=1"); }
    void DisableTexture() { glDisable(GL_TEXTURE_2D); glDisableClientState(GL_TEXTURE_COORD_ARRAY); GDDebug("Texture=0"); }
    void  EnableLighting() {  glEnable(GL_LIGHTING);  glEnable(GL_COLOR_MATERIAL); GDDebug("Lighting=1"); }
    void DisableLighting() { glDisable(GL_LIGHTING); glDisable(GL_COLOR_MATERIAL); GDDebug("Lighting=0"); }
    void  EnableVertexColor() {  glEnableClientState(GL_COLOR_ARRAY); GDDebug("VertexColor=1"); }
    void DisableVertexColor() { glDisableClientState(GL_COLOR_ARRAY); GDDebug("VertexColor=0"); }
    void  EnableNormals() {  glEnableClientState(GL_NORMAL_ARRAY); GDDebug("Normals=1"); }
    void DisableNormals() { glDisableClientState(GL_NORMAL_ARRAY); GDDebug("Normals=0"); }
    //void TextureEnvReplace()  { glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);  GDDebug("TextureEnv=R"); }
    //void TextureEnvModulate() { glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE); GDDebug("TextureEnv=M"); }
    void  EnableLight(int n) { if (n)  glEnable(GL_LIGHT1); else  glEnable(GL_LIGHT0); GDDebug("Light", n, "=1"); }
    void DisableLight(int n) { if (n) glDisable(GL_LIGHT1); else glDisable(GL_LIGHT0); GDDebug("Light", n, "=0"); }
    void Material(int t, float *color) { glMaterialfv(GL_FRONT_AND_BACK, t, color); }
    void Light(int n, int t, float *color) { glLightfv(((n) ? GL_LIGHT1 : GL_LIGHT0), t, color); }
#if defined(LFL_IPHONE) || defined(LFL_ANDROID)
    void TextureGenLinear() {}
    void TextureGenReflection() {}
    void DisableTextureGen() {}
#else
    void  EnableTextureGen() {  glEnable(GL_TEXTURE_GEN_S);  glEnable(GL_TEXTURE_GEN_T);  glEnable(GL_TEXTURE_GEN_R); GDDebug("TextureGen=1"); }
    void DisableTextureGen() { glDisable(GL_TEXTURE_GEN_S); glDisable(GL_TEXTURE_GEN_T); glDisable(GL_TEXTURE_GEN_R); GDDebug("TextureGen=0"); }
    void TextureGenLinear() {
        static float X[4] = { -1,0,0,0 }, Y[4] = { 0,-1,0,0 }, Z[4] = { 0,0,-1,0 };
        EnableTextureGen();
        glTexGeni(GL_S, GL_TEXTURE_GEN_MODE, GL_OBJECT_LINEAR);
        glTexGeni(GL_T, GL_TEXTURE_GEN_MODE, GL_OBJECT_LINEAR);
        glTexGeni(GL_R, GL_TEXTURE_GEN_MODE, GL_OBJECT_LINEAR);
        glTexGenfv(GL_S, GL_OBJECT_PLANE, X);
        glTexGenfv(GL_T, GL_OBJECT_PLANE, Y);
        glTexGenfv(GL_R, GL_OBJECT_PLANE, Z);
        GDDebug("TextureGen=L");
    }
    void TextureGenReflection() {
        EnableTextureGen();
        glTexGeni(GL_S, GL_TEXTURE_GEN_MODE, GL_OBJECT_LINEAR);
        glTexGeni(GL_T, GL_TEXTURE_GEN_MODE, GL_OBJECT_LINEAR);
        glTexGeni(GL_R, GL_TEXTURE_GEN_MODE, GL_OBJECT_LINEAR);
        GDDebug("TextureGen=R");
    }
#endif
    void DisableCubeMap()   { glDisable(GL_TEXTURE_CUBE_MAP); DisableTextureGen();                   GDDebug("CubeMap=", 0); }
    void BindCubeMap(int n) {  glEnable(GL_TEXTURE_CUBE_MAP); glBindTexture(GL_TEXTURE_CUBE_MAP, n); GDDebug("CubeMap=", n); }
    void ActiveTexture(int n) {
        glClientActiveTexture(GL_TEXTURE0 + n);
        glActiveTexture(GL_TEXTURE0 + n);
        // glTexEnvf (GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_COMBINE);
        // glTexEnvf (GL_TEXTURE_ENV, GL_COMBINE_RGB_EXT, GL_MODULATE);
        GDDebug("ActiveTexture=", n);
    }
    void BindTexture(int t, int n) { glBindTexture(t, n); GDDebug("BindTexture=", t, ",", n); }
    void VertexPointer(int m, int t, int w, int o, float *verts, int l, int *out, bool ud) { glVertexPointer  (m, t, w, verts + o/sizeof(float)); GDDebug("VertexPointer"); }
    void TexPointer   (int m, int t, int w, int o, float *tex,   int l, int *out, bool ud) { glTexCoordPointer(m, t, w, tex   + o/sizeof(float)); GDDebug("TexPointer"); }
    void ColorPointer (int m, int t, int w, int o, float *verts, int l, int *out, bool ud) { glColorPointer   (m, t, w, verts + o/sizeof(float)); GDDebug("ColorPointer"); }
    void NormalPointer(int m, int t, int w, int o, float *verts, int l, int *out, bool ud) { glNormalPointer  (   t, w, verts + o/sizeof(float)); GDDebug("NormalPointer"); }
    void Color4f(float r, float g, float b, float a) { default_color.back() = Color(r,g,b,a); UpdateColor(); }
    void MatrixProjection() { target_matrix=2; glMatrixMode(GL_PROJECTION); }
    void MatrixModelview() { target_matrix=1; glMatrixMode(GL_MODELVIEW); }
    void LoadIdentity() { glLoadIdentity(); }
    void PushMatrix() { glPushMatrix(); }
    void PopMatrix() { glPopMatrix(); }
    void GetMatrix(m44 *out) { glGetFloatv(target_matrix == 2 ? GL_PROJECTION_MATRIX : GL_MODELVIEW_MATRIX, &(*out)[0][0]); }
    void PrintMatrix() {}
    void Scalef(float x, float y, float z) { glScalef(x, y, z); }
    void Rotatef(float angle, float x, float y, float z) { glRotatef(angle, x, y, z); }
    void Ortho(float l, float r, float b, float t, float nv, float fv) { glOrtho(l,r, b,t, nv,fv); }
    void Frustum(float l, float r, float b, float t, float nv, float fv) { glFrustum(l,r, b,t, nv,fv); }
    void Mult(const float *m) { glMultMatrixf(m); }
    void Translate(float x, float y, float z) { glTranslatef(x, y, z); }
    void Draw(int pt, int np, int it, int o, void *index, int l, int *out, bool dirty) {
        glDrawElements(pt, np, it, (char*)index + o);
        GDDebug("Draw(", pt, ", ", np, ", ", it, ", ", o, ", ", index, ", ", l, ", ", dirty, ")");
    }
    void DrawArrays(int type, int o, int n) {
        glDrawArrays(type, o, n);
        GDDebug("DrawArrays(", type, ", ", o, ", ", n, ")");
    }
    void UseShader(Shader *S) { if (!S) S = &app->video.shader_default; glUseProgram(S->ID); GDDebug("Shader=", S->name); }
};

#ifdef LFL_GLES2
#ifdef LFL_QT
class OpenGLES2 : public QWindow, protected QOpenGLFunctions, public GraphicsDevice {
  public:
    bool QT_init, QT_grabbed;
    point QT_mp;

    void render_request() { QCoreApplication::postEvent(this, new QEvent(QEvent::UpdateRequest)); }
    bool event(QEvent *event) {
        if (event->type() != QEvent::UpdateRequest) return QWindow::event(event); 
        QCoreApplication::processEvents();

        if (!QT_init) {
            CHECK(!LFL::screen->gl);
            LFL::screen->gl = new QOpenGLContext(this);
            ((QOpenGLContext*)LFL::screen->gl)->setFormat(requestedFormat());
            ((QOpenGLContext*)LFL::screen->gl)->create();
            Window::MakeCurrent(LFL::screen);
            initializeOpenGLFunctions();

            vector<const char *> av;
            for (int i=0; i<lfl_argv.size(); i++) av.push_back(lfl_argv[i].c_str());
            av.push_back(0);
            LFLQTMain(lfl_argv.size(), &av[0]);
            QT_init = true;
            if (!app->run) { app->Free(); lfl_qapp->exit(); return true; }
        }

        app->Frame();

        if (!app->run) { app->Free(); lfl_qapp->exit(); return true; }
        if (!app->scheduler.wait_forever) render_request();
        return true;
    }
    void resizeEvent(QResizeEvent *ev) { QWindow::resizeEvent(ev); if (!QT_init) return; LFL::screen->Reshaped(ev->size().width(), ev->size().height()); }
    void keyPressEvent  (QKeyEvent *ev) { if (!QT_init) return; app->input.QueueKey(QT_key(ev->key()), 1); ev->accept(); }
    void keyReleaseEvent(QKeyEvent *ev) { if (!QT_init) return; app->input.QueueKey(QT_key(ev->key()), 0); ev->accept(); }
    void mousePressEvent  (QMouseEvent *ev) { if (!QT_init) return; app->input.QueueMouseClick(QT_mouse_button(ev->button()), 1, point(ev->x(), ev->y())); }
    void mouseReleaseEvent(QMouseEvent *ev) { if (!QT_init) return; app->input.QueueMouseClick(QT_mouse_button(ev->button()), 0, point(ev->x(), ev->y())); }
    void mouseMoveEvent   (QMouseEvent *ev) {
        point p(ev->x(), ev->y()), dx = p - QT_mp;
        if (!QT_init || (!p.x && !p.y)) return;
        app->input.QueueMouseMovement(p, dx);
        if (!QT_grabbed) {
            QT_mp = p;
        } else {
            QT_mp = point(width()/2, height()/2);
            QCursor::setPos(mapToGlobal(QPoint(QT_mp.x, QT_mp.y)));
        }
    }
    static unsigned QT_key(unsigned k) { return k < 256 && isalpha(k) ? ::tolower(k) : k; }
    static unsigned QT_mouse_button(int b) {
        if      (b == Qt::LeftButton)  return 1;
        else if (b == Qt::RightButton) return 2;
        return 0;
    }
#else /* LFL_QT */
struct OpenGLES2 : public GraphicsDevice {
    void *QT_init, *QT_grabbed, *QT_mx, *QT_my, *QT_GL_context;
#endif /* LFL_QT */

    Shader *shader;
    int enabled_array, enabled_indexarray, matrix_target;
    vector<m44> modelview_matrix, projection_matrix;
    bool dirty_matrix, dirty_color;
    int cubemap_on, normals_on, texture_on, colorverts_on, lighting_on;
    LFL::Material material;
    LFL::Light light[4];

    struct VertexAttribPointer {
        int m, t, w, o;
        VertexAttribPointer()                           : m(0), t(0), w(0), o(0) {}
        VertexAttribPointer(int M, int T, int W, int O) : m(M), t(T), w(W), o(O) {}
    } position_ptr, tex_ptr, color_ptr, normal_ptr;

    OpenGLES2() : QT_init(0), QT_grabbed(0), shader(0), enabled_array(-1), enabled_indexarray(-1), matrix_target(-1), dirty_matrix(1), dirty_color(1), cubemap_on(0), normals_on(0), texture_on(0), colorverts_on(0), lighting_on(0) {
        modelview_matrix.push_back(m44::Identity());
        projection_matrix.push_back(m44::Identity());
        default_color.push_back(Color(1.0, 1.0, 1.0, 1.0));
    }

    void Init() {
        // GetIntegerv(GL_FRAMEBUFFER_BINDING, &oldFBO);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        string vertex_shader = LocalFile::FileContents(StrCat(ASSETS_DIR, "lfapp_vertex.glsl"));
        string pixel_shader  = LocalFile::FileContents(StrCat(ASSETS_DIR, "lfapp_pixel.glsl"));
        Shader::Create("lfapp",          vertex_shader, pixel_shader, ShaderDefines(1,0,1,0), &app->video.shader_default);
        Shader::Create("lfapp_cubemap",  vertex_shader, pixel_shader, ShaderDefines(1,0,0,1), &app->video.shader_cubemap);
        Shader::Create("lfapp_normals",  vertex_shader, pixel_shader, ShaderDefines(0,1,1,0), &app->video.shader_normals);
        Shader::Create("lfapp_cubenorm", vertex_shader, pixel_shader, ShaderDefines(0,1,0,1), &app->video.shader_cubenorm);
        UseShader(0);
        GDDebug("Init");
    }

    vector<m44> *TargetMatrix() {
        if      (matrix_target == 1) return &modelview_matrix;
        else if (matrix_target == 2) return &projection_matrix;
        else FATAL("uknown matrix ", matrix_target);
    }
    void UpdateMatrix() { dirty_matrix = true; }
    void UpdateColor() { dirty_color = true; }
    void UpdateShader() {
        if (cubemap_on && normals_on) UseShader(&app->video.shader_cubenorm);
        else if          (cubemap_on) UseShader(&app->video.shader_cubemap);
        else if          (normals_on) UseShader(&app->video.shader_normals);
        else                          UseShader(&app->video.shader_default);
    }
    void UpdatePosition() {
        glEnableVertexAttribArray(shader->slot_position);
        SetVertexAttribPointer(shader->slot_position, position_ptr);
    }
    void UpdateTexture() {
        bool supports = shader->slot_tex >= 0;
        glUniform1i(shader->uniform_texon, texture_on && supports);
        if (supports) {
            if (texture_on) {  glEnableVertexAttribArray(shader->slot_tex); SetVertexAttribPointer(shader->slot_tex, tex_ptr); }
            else            { glDisableVertexAttribArray(shader->slot_tex); }
        } else if (texture_on && FLAGS_gd_debug) ERROR("shader doesnt support texture");
    }
    void UpdateColorVerts() {
        bool supports = shader->slot_color >= 0;
        glUniform1i(shader->uniform_coloron, colorverts_on && supports);
        if (supports) {
            if (colorverts_on) {  glEnableVertexAttribArray(shader->slot_color); SetVertexAttribPointer(shader->slot_color, color_ptr); }
            else               { glDisableVertexAttribArray(shader->slot_color); }
        } else if (colorverts_on) ERROR("shader doesnt support vertex color");
    }
    void UpdateNormals() {
        bool supports = shader->slot_normal >= 0;
        glUniform1i(shader->uniform_normalon, normals_on && supports);
        if (supports) {
            if (normals_on) {  glEnableVertexAttribArray(shader->slot_normal); SetVertexAttribPointer(shader->slot_normal, normal_ptr); }
            else            { glDisableVertexAttribArray(shader->slot_normal); }
        } else if (normals_on) ERROR("shader doesnt support normals");
    }

    bool ShaderSupport() { return true; }
    bool LightingSupport() { return false; }
    void EnableTexture()  { if (Changed(&texture_on, 1)) UpdateTexture(); GDDebug("Texture=1"); }
    void DisableTexture() { if (Changed(&texture_on, 0)) UpdateTexture(); GDDebug("Texture=0"); }
    void EnableLighting()  { lighting_on=1; GDDebug("Lighting=1"); }
    void DisableLighting() { lighting_on=0; GDDebug("Lighting=0"); }
    void EnableVertexColor()  { if (Changed(&colorverts_on, 1)) UpdateColorVerts(); GDDebug("VertexColor=1"); }
    void DisableVertexColor() { if (Changed(&colorverts_on, 0)) UpdateColorVerts(); GDDebug("VertexColor=0"); }
    void EnableNormals()  { if (Changed(&normals_on, 1)) { UpdateShader(); UpdateNormals(); } GDDebug("Normals=1"); }
    void DisableNormals() { if (Changed(&normals_on, 0)) { UpdateShader(); UpdateNormals(); } GDDebug("Normals=0"); }
    void EnableLight(int n) {}
    void DisableLight(int n) {}
    void Material(int t, float *v) {
        if      (t == GL_AMBIENT)             material.ambient  = Color(v);
        else if (t == GL_DIFFUSE)             material.diffuse  = Color(v);
        else if (t == GL_SPECULAR)            material.specular = Color(v);
        else if (t == GL_EMISSION)            material.emissive = Color(v);
        else if (t == GL_AMBIENT_AND_DIFFUSE) material.ambient = material.diffuse = Color(v);
        shader->dirty_material = app->video.shader_cubenorm.dirty_material = app->video.shader_normals.dirty_material = 1;
    }
    void Light(int n, int t, float *v) {
        bool light_pos = 0, light_color = 0;
        if (n != 0) { ERROR("ignoring Light(", n, ")"); return; }

        if      (t == GL_POSITION) { light_pos=1;   light[n].pos = modelview_matrix.back().Transform(v4(v)); }
        else if (t == GL_AMBIENT)  { light_color=1; light[n].color.ambient  = Color(v); }
        else if (t == GL_DIFFUSE)  { light_color=1; light[n].color.diffuse  = Color(v); }
        else if (t == GL_SPECULAR) { light_color=1; light[n].color.specular = Color(v); }

        if (light_pos)   { shader->dirty_light_pos  [n] = app->video.shader_cubenorm.dirty_light_pos  [n] = app->video.shader_normals.dirty_light_pos  [n] = 1; }
        if (light_color) { shader->dirty_light_color[n] = app->video.shader_cubenorm.dirty_light_color[n] = app->video.shader_normals.dirty_light_color[n] = 1; }
    }
    void DisableCubeMap()   { if (Changed(&cubemap_on, 0)) { UpdateShader(); glUniform1i(shader->uniform_cubeon, 0); }                                                                                 GDDebug("CubeMap=", 0); }
    void BindCubeMap(int n) { if (Changed(&cubemap_on, 1)) { UpdateShader(); glUniform1i(shader->uniform_cubeon, 1); } glUniform1i(shader->uniform_cubetex, 0); glBindTexture(GL_TEXTURE_CUBE_MAP, n); GDDebug("CubeMap=", n); }
    void TextureGenLinear() {}
    void TextureGenReflection() {}
    void ActiveTexture(int n) { glActiveTexture(n ? GL_TEXTURE1 : GL_TEXTURE0); GDDebug("ActivteTexture=", n); }
    void BindTexture(int t, int n) {
        glActiveTexture(GL_TEXTURE0); 
        glBindTexture(t, n);
        glUniform1i(shader->uniform_tex, 0);
        GDDebug("BindTexture=", t, ",", n);
    }
    void SetVertexAttribPointer(int slot, const VertexAttribPointer &ptr) { 
        glVertexAttribPointer(slot, ptr.m, ptr.t, GL_FALSE, ptr.w, (GLvoid*)(long)ptr.o);
    }
    void VertexPointer(int m, int t, int w, int o, float *verts, int l, int *out, bool dirty) {
        bool input_dirty = dirty, first = (*out == -1);
        if (first) { glGenBuffers(1, (GLuint*)out); dirty = true; }
        if (*out != enabled_array) {
            CHECK(shader);
            CHECK((!o && !w) || o < w);
            enabled_array = *out;
            glBindBuffer(GL_ARRAY_BUFFER, *out);
            position_ptr = VertexAttribPointer(m, t, w, o);
            UpdatePosition();
        }
        if (first || dirty) {
            if (first) glBufferData(GL_ARRAY_BUFFER, l, verts, input_dirty ? GL_DYNAMIC_DRAW : GL_STATIC_DRAW);
            else       glBufferSubData(GL_ARRAY_BUFFER, 0, l, verts);
        }
        GDDebug("VertexPointer");
    }
    void TexPointer(int m, int t, int w, int o, float *tex, int l, int *out, bool dirty) {
        CHECK_LT(o, w);
        CHECK(*out == enabled_array);
        tex_ptr = VertexAttribPointer(m, t, w, o);
        if (!texture_on) EnableTexture();
        else if (shader->slot_tex >= 0) SetVertexAttribPointer(shader->slot_tex, tex_ptr);
        GDDebug("TexPointer");
    }
    void ColorPointer(int m, int t, int w, int o, float *verts, int l, int *out, bool dirty) {
        CHECK_LT(o, w);
        CHECK(*out == enabled_array);
        color_ptr = VertexAttribPointer(m, t, w, o);
        if (!colorverts_on) EnableVertexColor();
        else if (shader->slot_color >= 0) SetVertexAttribPointer(shader->slot_color, color_ptr);
        GDDebug("ColorPointer");
    }
    void NormalPointer(int m, int t, int w, int o, float *verts, int l, int *out, bool dirty) {
        CHECK_LT(o, w);
        CHECK(*out == enabled_array);
        normal_ptr = VertexAttribPointer(m, t, w, o);
        if (!normals_on) EnableNormals();
        else if (shader->slot_normal >= 0) SetVertexAttribPointer(shader->slot_normal, normal_ptr);
        GDDebug("NormalPointer");
    }
    void MatrixModelview()  { matrix_target=1; }
    void MatrixProjection() { matrix_target=2; }
    void GetMatrix(m44 *out) { *out = TargetMatrix()->back(); }
    void PopMatrix() {
        vector<m44> *target = TargetMatrix();
        if      (target->size() >= 1) target->pop_back();
        else if (target->size() == 1) target->back().Assign(m44::Identity());
        UpdateMatrix();
    }
    void PrintMatrix()        { TargetMatrix()->back().Print(StrCat("mt", matrix_target)); }
    void PushMatrix()         { TargetMatrix()->push_back(TargetMatrix()->back()); UpdateMatrix(); }
    void LoadIdentity()       { TargetMatrix()->back().Assign(m44::Identity());    UpdateMatrix(); }
    void Mult(const float *m) { TargetMatrix()->back().Mult(m44(m));               UpdateMatrix(); }
    void Scalef(float x, float y, float z) {
#if 0
        TargetMatrix()->back().mult(m44::scale(x, y, z));
#else
        m44 &m = TargetMatrix()->back();
        m[0].x *= x; m[0].y *= x; m[0].z *= x;
        m[1].x *= y; m[1].y *= y; m[1].z *= y;
        m[2].x *= z; m[2].y *= z; m[2].z *= z;
#endif
        UpdateMatrix();
    }
    void Rotatef(float angle, float x, float y, float z) { TargetMatrix()->back().Mult(m44::Rotate(DegreeToRadian(angle), x, y, z)); UpdateMatrix(); }
    void Ortho  (float l, float r, float b, float t, float nv, float fv) { TargetMatrix()->back().Mult(m44::Ortho  (l, r, b, t, nv, fv)); UpdateMatrix(); }
    void Frustum(float l, float r, float b, float t, float nv, float fv) { TargetMatrix()->back().Mult(m44::Frustum(l, r, b, t, nv, fv)); UpdateMatrix(); }
    void Translate(float x, float y, float z) { 
#if 0
        TargetMatrix()->back().mult(m44::translate(x, y, z));
#else
        m44 &m = TargetMatrix()->back();
        m[3].x += m[0].x * x + m[1].x * y + m[2].x * z;
        m[3].y += m[0].y * x + m[1].y * y + m[2].y * z;
        m[3].z += m[0].z * x + m[1].z * y + m[2].z * z;
        m[3].w += m[0].w * x + m[1].w * y + m[2].w * z;
#endif
        UpdateMatrix();
    }
    void Color4f(float r, float g, float b, float a) {
        if (lighting_on) {
            float c[] = { r, g, b, a };
            Material(GL_AMBIENT_AND_DIFFUSE, c);
        } else {
            default_color.back() = Color(r,g,b,a);
            UpdateColor();
        }
    }
    void Draw(int pt, int np, int it, int o, void *index, int l, int *out, bool dirty) {
        bool input_dirty = dirty;
        if (*out == -1) { glGenBuffers(1, (GLuint*)out); dirty = true; }
        if (*out != enabled_indexarray) { 
            enabled_indexarray = *out;
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, *out);
        }
#if 0
        int bound_buf, bound_array_buf;
        glGetIntegerv(GL_ARRAY_BUFFER_BINDING, &bound_buf);
        glGetIntegerv(GL_ELEMENT_ARRAY_BUFFER_BINDING, &bound_array_buf);
        INFO("bb=", bound_buf, " bab=", bound_array_buf);
#endif
        if (dirty) glBufferData(GL_ELEMENT_ARRAY_BUFFER, l, index, input_dirty ? GL_DYNAMIC_DRAW : GL_STATIC_DRAW);

        PushDirtyState();
        glDrawElements(pt, np, it, (GLvoid*)(long)o);
        GDDebug("Draw(", pt, ", ", np, ", ", it, ", ", o, ", ", index, ", ", l, ", ", dirty, ")");
    }
    void DrawArrays(int type, int o, int n) {
        GDDebug("DrawArrays-Pre(", type, ", ", o, ", ", n, ")");
        //glBindBuffer(GL_ARRAY_BUFFER, 0);
        //glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
        GDDebug("DrawArrays-Push(", type, ", ", o, ", ", n, ")");
        PushDirtyState();
        glDrawArrays(type, o, n);
        GDDebug("DrawArrays-Post(", type, ", ", o, ", ", n, ")");
    }
    void PushDirtyState() {
        if (dirty_matrix) {
            dirty_matrix = false;
            m44 m = projection_matrix.back();
            m.Mult(modelview_matrix.back());
            glUniformMatrix4fv(shader->uniform_modelviewproj, 1, 0, m[0]);
            glUniformMatrix4fv(shader->uniform_modelview,     1, 0, modelview_matrix.back()[0]);
        }
        if (dirty_color && shader->uniform_colordefault >= 0) {
            dirty_color = false;
            glUniform4fv(shader->uniform_colordefault, 1, default_color.back().x);
        }
        if (shader->dirty_material) {
            glUniform4fv(shader->uniform_material_ambient,  1, material.ambient.x);
            glUniform4fv(shader->uniform_material_diffuse,  1, material.diffuse.x);
            glUniform4fv(shader->uniform_material_specular, 1, material.specular.x);
            glUniform4fv(shader->uniform_material_emission, 1, material.emissive.x);
        }
        for (int i=0; i<sizeofarray(light) && i<sizeofarray(shader->dirty_light_pos); i++) {
            if (shader->dirty_light_pos[i]) {
                shader->dirty_light_pos[i] = 0;
                glUniform4fv(shader->uniform_light0_pos, 1, light[i].pos);
            }
            if (shader->dirty_light_color[i]) {
                shader->dirty_light_color[i] = 0;
                glUniform4fv(shader->uniform_light0_ambient,  1, light[i].color.ambient.x);
                glUniform4fv(shader->uniform_light0_diffuse,  1, light[i].color.diffuse.x);
                glUniform4fv(shader->uniform_light0_specular, 1, light[i].color.specular.x);
            }
        }
    }
    void UseShader(Shader *S) {
        if (!S) return UpdateShader();
        if (shader == S || !S->ID) return;
        shader = S;
        glUseProgram(shader->ID);
        GDDebug("Shader=", S->name);
        dirty_matrix = dirty_color = true;
        for (int i=0, s; i<S->MaxVertexAttrib; i++) {
            if ((s = S->unused_attrib_slot[i]) < 0) break;
            glDisableVertexAttribArray(s);
        }
        UpdatePosition();
        UpdateNormals();
        UpdateColorVerts();
        UpdateTexture();
    }
};
#endif // LFL_GLES2

// Shader interaface
int GraphicsDevice::CreateProgram() { return glCreateProgram(); }
int GraphicsDevice::CreateShader(int t) { return glCreateShader(t); }
void GraphicsDevice::ShaderSource(int shader, int count, const char **source, int *len) { glShaderSource(shader, count, source, len); }
void GraphicsDevice::CompileShader(int shader) {
    char buf[1024] = {0}; int l=0;
    glCompileShader(shader);
    glGetShaderInfoLog(shader, sizeof(buf), &l, buf);
    if (l) INFO(buf);
}
void GraphicsDevice::AttachShader(int prog, int shader) { glAttachShader(prog, shader); }
void GraphicsDevice::BindAttribLocation(int prog, int loc, const string &name) { glBindAttribLocation(prog, loc, name.c_str()); }
void GraphicsDevice::LinkProgram(int prog) {
    char buf[1024] = {0}; int l=0;
    glLinkProgram(prog);
    glGetProgramInfoLog(prog, sizeof(buf), &l, buf);
    if (l) INFO(buf);
    GLint link_status;
    glGetProgramiv(prog, GL_LINK_STATUS, &link_status);
    if (link_status != GL_TRUE) FATAL("link failed");
}
void GraphicsDevice::GetProgramiv(int p, int t, int *out) { glGetProgramiv(p, t, out); }
void GraphicsDevice::GetIntegerv(int t, int *out) { glGetIntegerv(t, out); }
int GraphicsDevice::GetAttribLocation (int prog, const string &name) { return glGetAttribLocation (prog, name.c_str()); }
int GraphicsDevice::GetUniformLocation(int prog, const string &name) { return glGetUniformLocation(prog, name.c_str()); }
void GraphicsDevice::Uniform1i(int u, int v) { glUniform1i(u, v); }
void GraphicsDevice::Uniform1f(int u, float v) { glUniform1f(u, v); }
void GraphicsDevice::Uniform2f(int u, float v1, float v2) { glUniform2f(u, v1, v2); }
void GraphicsDevice::Uniform3fv(int u, int n, const float *v) { glUniform3fv(u, n, v); }

// Common layer
void GraphicsDevice::Flush() { glFlush(); }
void GraphicsDevice::Clear() { glClear(GL_COLOR_BUFFER_BIT | (draw_mode == DrawMode::_3D ? GL_DEPTH_BUFFER_BIT : 0)); }
void GraphicsDevice::ClearColor(const Color &c) { glClearColor(c.r(), c.g(), c.b(), c.a()); }
void GraphicsDevice::PushColor() { default_color.push_back(default_color.back()); UpdateColor();  }
void GraphicsDevice::PopColor() {
    if      (default_color.size() >= 1) default_color.pop_back();
    else if (default_color.size() == 1) default_color.back() = Color(1.0, 1.0, 1.0, 1.0);
    UpdateColor();
}
void GraphicsDevice::PointSize(float n) { glPointSize(n); }
void GraphicsDevice::LineWidth(float n) { glLineWidth(n); }
void GraphicsDevice::DelTextures(int n, const unsigned *id) { glDeleteTextures(n, id); }
void GraphicsDevice::GenTextures(int t, int n, unsigned *out) {
    for (int i=0; i<n; i++) CHECK_EQ(0, out[i]);
    if (t == GL_TEXTURE_CUBE_MAP) glEnable(GL_TEXTURE_CUBE_MAP);
    glGenTextures(n, out);
    for (int i=0; i<n; i++) {
        glBindTexture(t, out[i]);
        glTexParameteri(t, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(t, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    }
}

void GraphicsDevice::CheckForError(const char *file, int line) {
    GLint gl_error=0, gl_validate_status=0;
    if ((gl_error = glGetError())) {
        ERROR(file, ":", line, " gl error: ", gl_error);
        BreakHook();
#ifdef LFL_GLES2
        if (screen->opengles_version == 2) {
            Shader *shader = ((OpenGLES2*)screen->gd)->shader;
            glValidateProgram(shader->ID);
            glGetProgramiv(shader->ID, GL_VALIDATE_STATUS, &gl_validate_status);
            if (gl_validate_status != GL_TRUE) ERROR(shader->name, ": gl validate status ", gl_validate_status);

            char buf[1024]; int len;
            glGetProgramInfoLog(shader->ID, sizeof(buf), &len, buf);
            if (len) INFO(buf);
        }
#endif
    }
}

void GraphicsDevice::EnableDepthTest()  {  glEnable(GL_DEPTH_TEST); glDepthMask(GL_TRUE);  GDDebug("DepthTest=1"); }
void GraphicsDevice::DisableDepthTest() { glDisable(GL_DEPTH_TEST); glDepthMask(GL_FALSE); GDDebug("DepthTest=0"); }
void GraphicsDevice::DisableBlend() { glDisable(GL_BLEND);                                                    GDDebug("Blend=0"); }
void GraphicsDevice::EnableBlend()  {  glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); GDDebug("Blend=1"); }
void GraphicsDevice::BlendMode(int sm, int dm) { glBlendFunc(sm, dm); GDDebug("BlendMode=", sm, ",", dm); }
void GraphicsDevice::RestoreViewport(int dm) { ViewPort(screen->Box()); DrawMode(dm); }
void GraphicsDevice::DrawMode(int dm, bool flush) { return DrawMode(dm, screen->width, screen->height, flush); }
void GraphicsDevice::DrawMode(int dm, int W, int H, bool flush) {
    if (draw_mode == dm && !flush) return;
    draw_mode = dm;
    Color4f(1,1,1,1);
    MatrixProjection();
    LoadIdentity();
    if (FLAGS_rotate_view) Rotatef(FLAGS_rotate_view,0,0,1);

    bool _2D = draw_mode == DrawMode::_2D;
    if (_2D) Ortho(0, W, 0, H, 0, 100);
    else {
        float aspect=(float)W/H;
        double top = tan(FLAGS_field_of_view * M_PI/360.0) * FLAGS_near_plane;
        screen->gd->Frustum(aspect*-top, aspect*top, -top, top, FLAGS_near_plane, FLAGS_far_plane);
    }

    if (_2D) DisableDepthTest();
    else     EnableDepthTest();

    MatrixModelview();
    LoadIdentity();
    Scene::Select();
    if (_2D) EnableLayering();
}

void GraphicsDevice::LookAt(const v3 &pos, const v3 &targ, const v3 &up) {
    v3 Z = pos - targ;       Z.Norm();
    v3 X = v3::Cross(up, Z); X.Norm();
    v3 Y = v3::Cross(Z,  X); Y.Norm();
    float m[16] = {
        X.x, Y.x, Z.x, 0.0,
        X.y, Y.y, Z.y, 0.0,
        X.z, Y.z, Z.z, 0.0,
        0.0, 0.0, 0.0, 1.0
    };
    Mult(m);
    Translate(-pos.x, -pos.y, -pos.z);
}

void GraphicsDevice::ViewPort(Box w) {
    if (FLAGS_swap_axis) w.swapaxis(screen->width, screen->height);
    glViewport(w.x, w.y, w.w, w.h);
}

void GraphicsDevice::Scissor(Box w) {
    if (FLAGS_swap_axis) w.swapaxis(screen->width, screen->height);
    glEnable(GL_SCISSOR_TEST);
    glScissor(w.x, w.y, w.w, w.h);
}

void GraphicsDevice::PushScissor(Box w) {
    auto &ss = scissor_stack.back();
    if (ss.empty()) ss.push_back(w);
    else ss.push_back(w.Intersect(ss.back()));
    screen->gd->Scissor(ss.back());
}

void GraphicsDevice::PopScissor() {
    auto &ss = scissor_stack.back();
    if (ss.size()) ss.pop_back();
    if (ss.size()) screen->gd->Scissor(ss.back());
    else           glDisable(GL_SCISSOR_TEST);
}

void GraphicsDevice::PushScissorStack() {
    scissor_stack.push_back(vector<Box>());
    glDisable(GL_SCISSOR_TEST);
}

void GraphicsDevice::PopScissorStack() {
    CHECK_GT(scissor_stack.size(), 1);
    scissor_stack.pop_back();
    screen->gd->Scissor(scissor_stack.back().back());
}

void GraphicsDevice::DrawPixels(const Box &b, const Texture &tex) {
    Texture temp;
    temp.Resize(tex.width, tex.height, tex.pf, Texture::Flag::CreateGL);
    temp.UpdateGL(tex.buf, LFL::Box(tex.width, tex.height), Texture::Flag::FlipY); 
    b.Draw(temp.coord);
    temp.ClearGL();
}

int GraphicsDevice::VertsPerPrimitive(int primtype) {
    if (primtype == GL_TRIANGLES) return 3;
    return 0;
}

#else // LFL_HEADLESS
struct FakeGraphicsDevice : public GraphicsDevice {
    virtual void Init() {}
    virtual bool ShaderSupport() { return 0; }
    virtual void EnableTexture() {}
    virtual void DisableTexture() {}
    virtual void EnableLighting() {}
    virtual void DisableLighting() {}
    virtual void EnableNormals() {}
    virtual void DisableNormals() {}
    virtual void EnableVertexColor() {}
    virtual void DisableVertexColor() {}
    virtual void EnableLight(int n) {}
    virtual void DisableLight(int n) {}
    virtual void DisableCubeMap() {}
    virtual void BindCubeMap(int n) {}
    virtual void TextureGenLinear() {}
    virtual void TextureGenReflection() {}
    virtual void Material(int t, float *color) {}
    virtual void Light(int n, int t, float *color) {}
    virtual void BindTexture(int t, int n) {}
    virtual void ActiveTexture(int n) {}
    virtual void TexPointer(int m, int t, int w, int o, float *tex, int l, int *out, bool dirty) {}
    virtual void VertexPointer(int m, int t, int w, int o, float *verts, int l, int *out, bool dirty) {}
    virtual void ColorPointer(int m, int t, int w, int o, float *verts, int l, int *out, bool dirty) {}
    virtual void NormalPointer(int m, int t, int w, int o, float *verts, int l, int *out, bool dirty) {}
    virtual void Color4f(float r, float g, float b, float a) {}
    virtual void UpdateColor() {}
    virtual void MatrixProjection() {}
    virtual void MatrixModelview() {}
    virtual void LoadIdentity() {}
    virtual void PushMatrix() {}
    virtual void PopMatrix() {}
    virtual void GetMatrix(m44 *out) {}
    virtual void PrintMatrix() {}
    virtual void Scalef(float x, float y, float z) {}
    virtual void Rotatef(float angle, float x, float y, float z) {}
    virtual void Ortho(float l, float r, float b, float t, float nv, float fv) {}
    virtual void Frustum(float l, float r, float b, float t, float nv, float fv) {}
    virtual void Mult(const float *m) {}
    virtual void Translate(float x, float y, float z) {}
    virtual void UseShader(Shader *shader) {}
    virtual void Draw(int pt, int np, int it, int o, void *index, int l, int *out, bool dirty) {}
    virtual void DrawArrays(int t, int o, int n) {}
};

const int GraphicsDevice::Float = 0;
const int GraphicsDevice::Points = 0;
const int GraphicsDevice::Lines = 0;
const int GraphicsDevice::LineLoop = 0;
const int GraphicsDevice::Triangles = 0;
const int GraphicsDevice::TriangleStrip = 0;
const int GraphicsDevice::Texture2D = 0;
const int GraphicsDevice::UnsignedInt = 0;
const int GraphicsDevice::Ambient = 0;
const int GraphicsDevice::Diffuse = 0;
const int GraphicsDevice::Specular = 0;
const int GraphicsDevice::Position = 0;
const int GraphicsDevice::Emission = 0;
const int GraphicsDevice::One = 0;
const int GraphicsDevice::SrcAlpha = 0;
const int GraphicsDevice::OneMinusSrcAlpha = 0;
const int GraphicsDevice::OneMinusDstColor = 0;
const int GraphicsDevice::Fill = 0;
const int GraphicsDevice::Line = 0;
const int GraphicsDevice::Point = 0;

int GraphicsDevice::CreateProgram() { return 0; }
int GraphicsDevice::CreateShader(int t) { return 0; }
int GraphicsDevice::GetAttribLocation (int prog, const string &name) { return 0; }
int GraphicsDevice::GetUniformLocation(int prog, const string &name) { return 0; }
void GraphicsDevice::ShaderSource(int shader, int count, const char **source, int *len) {}
void GraphicsDevice::CompileShader(int shader) {}
void GraphicsDevice::AttachShader(int prog, int shader) {}
void GraphicsDevice::BindAttribLocation(int prog, int loc, const string &name) {}
void GraphicsDevice::LinkProgram(int prog) {}
void GraphicsDevice::GetProgramiv(int p, int t, int *out) {}
void GraphicsDevice::GetIntegerv(int t, int *out) {}
void GraphicsDevice::Uniform1i(int u, int v) {}
void GraphicsDevice::Uniform1f(int u, float v) {}
void GraphicsDevice::Uniform2f(int u, float v1, float v2) {}
void GraphicsDevice::Uniform3fv(int u, int n, const float *v) {}
void GraphicsDevice::Flush() {}
void GraphicsDevice::Clear() {}
void GraphicsDevice::ClearColor(const Color &c) {}
void GraphicsDevice::PushColor() {}
void GraphicsDevice::PopColor() {}
void GraphicsDevice::PointSize(float n) {}
void GraphicsDevice::LineWidth(float n) {}
void GraphicsDevice::DelTextures(int n, const unsigned *id) {}
void GraphicsDevice::GenTextures(int t, int n, unsigned *out) {}
void GraphicsDevice::CheckForError(const char *file, int line) {}
void GraphicsDevice::EnableDepthTest() {}
void GraphicsDevice::DisableDepthTest() {}
void GraphicsDevice::DisableBlend() {}
void GraphicsDevice::EnableBlend() {}
void GraphicsDevice::BlendMode(int sm, int dm) {}
void GraphicsDevice::RestoreViewport(int dm) {}
void GraphicsDevice::DrawMode(int dm, bool flush) {}
void GraphicsDevice::DrawMode(int dm, int W, int H, bool flush) {}
void GraphicsDevice::LookAt(const v3 &pos, const v3 &targ, const v3 &up) {}
void GraphicsDevice::ViewPort(Box w) {}
void GraphicsDevice::Scissor(Box w) {}
void GraphicsDevice::PushScissor(Box w) {}
void GraphicsDevice::PopScissor() {}
void GraphicsDevice::PushScissorStack() {}
void GraphicsDevice::PopScissorStack() {}
void GraphicsDevice::DrawPixels(const Box&, const Texture&) {}
int GraphicsDevice::VertsPerPrimitive(int primtype) { return 0; }

bool Window::Create(Window *W) { screen->gd = new FakeGraphicsDevice(); Window::active[W->id] = W; return true; }
void Window::Close(Window *W) {}
void Window::MakeCurrent(Window *W) {}
#endif // LFL_HEADLESS

#ifdef LFL_ANDROIDVIDEO
struct AndroidVideoModule : public Module {
    int Init() {
        INFO("AndroidVideoModule::Init()");
        if (AndroidVideoInit(FLAGS_request_gles_version)) return -1;
        return 0;
    }
};
#endif

#ifdef LFL_IPHONEVIDEO
extern "C" void iPhoneVideoSwap();
struct IPhoneVideoModule : public Module {
    int Init() {
        INFO("IPhoneVideoModule::Init()");
        NativeWindowInit();
        NativeWindowSize(&screen->width, &screen->height);
        return 0;
    }
};
#endif

#ifdef LFL_OSXVIDEO
extern "C" void OSXVideoSwap(void*);
extern "C" void *OSXCreateWindow(int W, int H, struct NativeWindow *nw);
extern "C" void OSXMakeWindowCurrent(void *O);
extern "C" void OSXSetWindowSize(void*, int W, int H);
extern "C" void *OSXCreateGLContext(void *O);
struct OSXVideoModule : public Module {
    int Init() {
        INFO("OSXVideoModule::Init()");
        NativeWindowInit();
        NativeWindowSize(&screen->width, &screen->height);
        CHECK(Window::Create(screen));
        return 0;
    }
};
bool Window::Create(Window *W) { 
    W->id = OSXCreateWindow(W->width, W->height, W);
    if (W->id) Window::active[W->id] = W;
    return true; 
}
void Window::MakeCurrent(Window *W) { 
    if (W) OSXMakeWindowCurrent((screen = W)->id);
}
void Window::Close(Window *W) {
    Window::active.erase(W->id);
    if (Window::active.empty()) app->run = false;
}
#endif

#ifdef LFL_QT
struct QTVideoModule : public Module {
    int Init() {
        INFO("QTVideoModule::Init()");
        screen->Reshape(screen->width, screen->height);
        return 0;
    }
};
bool Window::Create(Window *W) {
    CHECK(!W->id && !W->gd);
    OpenGLES2 *gd = new OpenGLES2();
    QWindow *qwin = (QWindow*)gd;
    W->id = qwin;
    W->gd = gd;
    W->opengles_version = 2;

    QSurfaceFormat format;
    format.setSamples(16);
    gd->setFormat(format);
    gd->setSurfaceType(QWindow::OpenGLSurface);
    gd->resize(screen->width, screen->height);
    gd->show();
    gd->render_request();

    Window::active[W->id] = W;
    return true;
}
void Window::Close(Window *W) {
    Window::active.erase(W->id);
    if (Window::active.empty()) app->run = false;
    screen = 0;
}
void Window::MakeCurrent(Window *W) {
    screen = W; 
    ((QOpenGLContext*)screen->gl)->makeCurrent((QWindow*)screen->id);
}
void Mouse::GrabFocus()    { ((OpenGLES2*)screen->gd)->QT_grabbed=1; ((QWindow*)screen->id)->setCursor(Qt::BlankCursor); app->grab_mode.On();  screen->cursor_grabbed=true;  }
void Mouse::ReleaseFocus() { ((OpenGLES2*)screen->gd)->QT_grabbed=0; ((QWindow*)screen->id)->unsetCursor();              app->grab_mode.Off(); screen->cursor_grabbed=false; }
#endif

#ifdef LFL_GLFWVIDEO
/* struct NativeWindow { GLFWwindow *id; }; */
struct GLFWVideoModule : public Module {
    int Init() {
        INFO("GLFWVideoModule::Init");
        CHECK(Window::Create(screen));
        Window::MakeCurrent(screen);
        glfwSwapInterval(1);
        return 0;
    }
    int Free() {
        glfwTerminate();
        return 0;
    }
};
bool Window::Create(Window *W) {
    GLFWwindow *share = Window::active.empty() ? 0 : (GLFWwindow*)Window::active.begin()->second->id;
    if (!(W->id = glfwCreateWindow(W->width, W->height, W->caption.c_str(), 0, share))) { ERROR("glfwCreateWindow"); return false; }
    Window::active[W->id] = W;
    return true;
}
void Window::MakeCurrent(Window *W) {
    glfwMakeContextCurrent((GLFWwindow*)W->id);
    screen = W;
}
void Window::Close(Window *W) {
    Window::active.erase(W->id);
    bool done = Window::active.empty();
    if (done) app->shell.quit(vector<string>());
    if (!done) glfwDestroyWindow((GLFWwindow*)W->id);
    if (app->window_closed_cb) app->window_closed_cb(W);
    screen = 0;
}
#endif

#ifdef LFL_SDLVIDEO
/* struct NativeWindow { SDL_Window* id; SDL_GLContext gl; SDL_Surface *surface; }; */
struct SDLVideoModule : public Module {
    int Init() {
        INFO("SFLVideoModule::Init");
        CHECK(Window::Create(screen));
        Window::MakeCurrent(screen);
        SDL_GL_SetSwapInterval(1);
        return 0;
    }
    int Free() {
        SDL_Quit();
        return 0;
    }
};
bool Window::Create(Window *W) {
    int createFlag = SDL_WINDOW_RESIZABLE | SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN;
#if defined(LFL_IPHONE) || defined(LFL_ANDROID)
    createFlag |= SDL_WINDOW_BORDERLESS;
    int bitdepth[] = { 5, 6, 5 };
#else
    int bitdepth[] = { 8, 8, 8 };
#endif
    SDL_GL_SetAttribute(SDL_GL_RED_SIZE, bitdepth[0]);
    SDL_GL_SetAttribute(SDL_GL_GREEN_SIZE, bitdepth[1]);
    SDL_GL_SetAttribute(SDL_GL_BLUE_SIZE, bitdepth[2]);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 16);
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);

    if (!(W->id = SDL_CreateWindow(W->caption.c_str(), SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, W->width, W->height, createFlag)))
    { ERROR("SDL_CreateWindow: ",     SDL_GetError()); return false; }

    if (!Window::active.empty()) W->gl = Window::active.begin()->second->gl;
    else if (!(W->gl = SDL_GL_CreateContext((SDL_Window*)W->id)))
    { ERROR("SDL_GL_CreateContext: ", SDL_GetError()); return false; } 

    SDL_Surface* icon = SDL_LoadBMP(StrCat(ASSETS_DIR, "icon.bmp").c_str());
    SDL_SetWindowIcon((SDL_Window*)W->id, icon);

    Window::active[(void*)(long)SDL_GetWindowID((SDL_Window*)W->id)] = W;
    return true;
}
void Window::MakeCurrent(Window *W) {
    if (SDL_GL_MakeCurrent((SDL_Window*)W->id, W->gl) < 0) ERROR("SDL_GL_MakeCurrent: ", SDL_GetError());
    screen = W; 
}
void Window::Close(Window *W) {
    SDL_GL_MakeCurrent(NULL, NULL);
    Window::active.erase((void*)(long)SDL_GetWindowID((SDL_Window*)W->id));
    if (Window::active.empty()) {
        app->run = false;
        SDL_GL_DeleteContext(W->gl);
    }
    SDL_DestroyWindow((SDL_Window*)W->id);
    if (app->window_closed_cb) app->window_closed_cb(W);
    screen = 0;
}
#endif /* LFL_SDLVIDEO */

int Video::Init() {
    INFO("Video::Init()");
#if defined(LFL_QT)
    impl = new QTVideoModule();
#elif defined(LFL_GLFWVIDEO)
    impl = new GLFWVideoModule();
#elif defined(LFL_SDLVIDEO)
    impl = new SDLVideoModule();
#elif defined(LFL_ANDROIDVIDEO)
    impl = new AndroidVideoModule();
#elif defined(LFL_IPHONEVIDEO)
    impl = new IPhoneVideoModule();
#elif defined(LFL_OSXVIDEO)
    impl = new OSXVideoModule();
#endif
    if (impl) if (impl->Init()) return -1;

#if defined(LFL_GLEW) && !defined(LFL_HEADLESS)
#ifdef GLEW_MX
    screen.glew_context = new GLEWContext();
#endif
    GLenum glew_err;
    if ((glew_err = glewInit()) != GLEW_OK) { ERROR("glewInit: ", glewGetErrorString(glew_err)); return -1; }
#endif

    if (!screen->gd) CreateGraphicsDevice(screen);
    InitGraphicsDevice(screen);

    INFO("OpenGL Version: ", SpellNull((const char *)glGetString(GL_VERSION)));
    INFO("OpenGL Vendor: ",  SpellNull((const char *)glGetString(GL_VENDOR)));
#ifdef LFL_GLSL_SHADERS
    INFO("GL_SHADING_LANGUAGE_VERSION: ", SpellNull((const char *)glGetString(GL_SHADING_LANGUAGE_VERSION)));
#endif
    const char *glexts = SpellNull((const char *)glGetString(GL_EXTENSIONS));
    INFO("GL_EXTENSIONS: ", glexts);
#if defined(LFL_ANDROID) || defined(LFL_IPHONE)
    screen->opengles_cubemap = strstr(glexts, "GL_EXT_texture_cube_map") != 0;
#else
    screen->opengles_cubemap = strstr(glexts, "GL_ARB_texture_cube_map") != 0;
#endif
    INFO("lfapp_opengles_cubemap = ", screen->opengles_cubemap ? "true" : "false");

    init_fonts_cb();
    if (!screen->console) screen->InitConsole();
    return 0;
}

void *Video::CreateGLContext(Window *W) {
#if defined(LFL_OSXVIDEO)
    return OSXCreateGLContext(screen->id);
#else
    return 0;
#endif
}

void Video::CreateGraphicsDevice(Window *W) {
    CHECK(!W->gd);
#ifndef LFL_HEADLESS
#ifdef LFL_GLES2
#if !defined(LFL_IPHONE) && !defined(LFL_ANDROID)
    W->opengles_version = FLAGS_request_gles_version;
#endif
    if (W->opengles_version == 2) W->gd = new OpenGLES2();
    else
#endif /* LFL_GLES2 */
    W->gd = new OpenGLES1();
#endif /* LFL_HEADLESS */
}

void Video::InitGraphicsDevice(Window *W) {
    W->gd->Init();
    W->gd->ViewPort(W->Box());
    W->gd->DrawMode(W->gd->default_draw_mode);

    float pos[]={-.5,1,-.3f,0}, grey20[]={.2f,.2f,.2f,1}, white[]={1,1,1,1}, black[]={0,0,0,1};
    W->gd->EnableLight(0);
    W->gd->Light(0, GraphicsDevice::Position, pos);
    W->gd->Light(0, GraphicsDevice::Ambient,  grey20);
    W->gd->Light(0, GraphicsDevice::Diffuse,  white);
    W->gd->Light(0, GraphicsDevice::Specular, white);
    W->gd->Material(GraphicsDevice::Emission, black);
    W->gd->Material(GraphicsDevice::Specular, grey20);
    INFO("opengl_init: width=", W->width, ", height=", W->height, ", opengles_version: ", W->opengles_version);
}

void Video::InitFonts() {
    vector<string> atlas_font_size;
    Split(FLAGS_atlas_font_sizes, iscomma, &atlas_font_size);
    FontEngine *font_engine = Fonts::DefaultFontEngine();
    for (int i=0; i<atlas_font_size.size(); i++) {
        int size = atoi(atlas_font_size[i].c_str());
        font_engine->Init(FontDesc(FLAGS_default_font, FLAGS_default_font_family, size, Color::white, Color::clear, FLAGS_default_font_flag));
    }

    FontEngine *atlas_engine = Singleton<AtlasFontEngine>::Get();
    atlas_engine->Init(FontDesc("MenuAtlas1", "", 0, Color::black));
    atlas_engine->Init(FontDesc("MenuAtlas2", "", 0, Color::black));
}

int Video::Swap() {
#ifndef LFL_QT
    screen->gd->Flush();
#endif

#if defined(LFL_QT)
    ((QOpenGLContext*)screen->gl)->swapBuffers((QWindow*)screen->id);
#elif defined(LFL_ANDROIDVIDEO)
    AndroidVideoSwap();
#elif defined(LFL_GLFWVIDEO)
    glfwSwapBuffers((GLFWwindow*)screen->id);
#elif defined(LFL_SDLVIDEO)
    SDL_GL_SwapWindow((SDL_Window*)screen->id);
#elif defined(LFL_IPHONEVIDEO)
    iPhoneVideoSwap();
#elif defined(LFL_OSXVIDEO)
    OSXVideoSwap(screen->id);
#endif

    screen->gd->CheckForError(__FILE__, __LINE__);
    return 0;
}

int Video::Free() {
    if (impl) impl->Free();
    return 0;
}

void Window::Reshape(int w, int h) {
#if defined(LFL_QT)
    ((QWindow*)id)->resize(w, h);
    Window::MakeCurrent(screen);
#elif defined(LFL_GLFWVIDEO)
    glfwSetWindowSize((GLFWwindow*)id, w, h);
#elif defined(LFL_SDLVIDEO)
    SDL_SetWindowSize((SDL_Window*)id, w, h);
#elif defined(LFL_OSXVIDEO)
    OSXSetWindowSize(id, w, h);
#endif
}

void Window::Reshaped(int w, int h) {
    width = w;
    height = h;
    if (!gd) return;
    gd->ViewPort(LFL::Box(width, height));
    gd->DrawMode(screen->gd->default_draw_mode);
    for (auto g = screen->mouse_gui.begin(); g != screen->mouse_gui.end(); ++g) (*g)->Layout();
    if (app->reshaped_cb) app->reshaped_cb();
}

void Window::SwapAxis() {
    FLAGS_rotate_view = FLAGS_rotate_view ? 0 : -90;
    FLAGS_swap_axis = FLAGS_rotate_view != 0;
    Reshaped(height, width);
}

void Window::Frame(unsigned clicks, unsigned mic_samples, bool cam_sample, int flag) {
    if (minimized) return;
    if (screen != this) Window::MakeCurrent(this);

    if (FLAGS_lfapp_video) {
        gd->Clear();
        gd->LoadIdentity();
    }

    /* frame */
    int ret = frame_cb ? frame_cb(screen, clicks, mic_samples, cam_sample, flag) : 0;
    ClearEvents();

    /* allow app to skip frame */
    if (ret < 0) return;
    fps.Add(clicks);

    if (FLAGS_lfapp_video) {
        app->video.Swap();
        gd->DrawMode(gd->default_draw_mode);
    }
}

int Depth::OpenGLID(int id) {
    switch(id) {
        case _16: return GL_DEPTH_COMPONENT16;
    } return 0;
}

int CubeMap::OpenGLID(int id) {
    switch (id) {
        case NX: return GL_TEXTURE_CUBE_MAP_NEGATIVE_X;    case PX: return GL_TEXTURE_CUBE_MAP_POSITIVE_X;
        case NY: return GL_TEXTURE_CUBE_MAP_NEGATIVE_Y;    case PY: return GL_TEXTURE_CUBE_MAP_POSITIVE_Y;
        case NZ: return GL_TEXTURE_CUBE_MAP_NEGATIVE_Z;    case PZ: return GL_TEXTURE_CUBE_MAP_POSITIVE_Z;
    } return GL_TEXTURE_2D;
}

int ColorChannel::PixelOffset(int c) {
    switch (c) {
        case Red:  return 0;    case Green: return 1;
        case Blue: return 2;    case Alpha: return 3;
    }
    return 0;
}

const char *Pixel::Name(int p) {
    switch (p) {
        case RGB32: return "RGB32";    case RGB555:  return "RGB555";
        case BGR32: return "BGR32";    case BGR555:  return "BGR555";
        case RGB24: return "RGB24";    case RGB565:  return "RGB565";
        case BGR24: return "BGR24";    case BGR565:  return "BGR565";
        case RGBA:  return "RGBA";     case YUYV422: return "YUYV422";
        case GRAY8: return "GRAY8";    case GRAYA8:  return "GRAYA8";
        case LCD:   return "LCD";
    }; return 0; 
}

int Pixel::size(int p) {
    switch (p) {
        case RGB32:   case BGR32:  case RGBA:                             return 4;
        case RGB24:   case BGR24:  case LCD:                              return 3;
        case RGB555:  case BGR555: case RGB565: case BGR565: case GRAYA8: return 2;
        case YUYV422: case GRAY8:                                         return 1;
        default:                                                          return 0;
    }
}

int Pixel::OpenGLID(int p) {
    switch (p) {
        case RGBA:   case RGB32: case BGR32: return GL_RGBA;
        case RGB24:  case BGR24:             return GL_RGB;
        case GRAYA8:                         return GL_LUMINANCE_ALPHA;
        case GRAY8:                          return GL_LUMINANCE;
        default:                             return -1;
    }
}

string FloatContainer::DebugString() const {
    string ret = StrCat(Box::DebugString(), " fl{");
    for (int i=0; i<float_left.size(); i++) StrAppend(&ret, i?",":"", i, "=", float_left[i].DebugString());
    StrAppend(&ret, "} fr{");
    for (int i=0; i<float_right.size(); i++) StrAppend(&ret, i?",":"", i, "=", float_right[i].DebugString());
    return ret + "}";
}

string Box::DebugString() const { return StringPrintf("Box = { %d, %d, %d, %d }", x, y, w, h); }

void Box::Draw(const float *texcoord) const {
    static int verts_ind=-1;
    static const float default_texcoord[4] = {0, 0, 1, 1};
    const float *tc = X_or_Y(texcoord, default_texcoord);
    float verts[] = { (float)x,   (float)y,   tc[Texture::CoordMinX], tc[Texture::CoordMinY],
                      (float)x,   (float)y+h, tc[Texture::CoordMinX], tc[Texture::CoordMaxY],
                      (float)x+w, (float)y,   tc[Texture::CoordMaxX], tc[Texture::CoordMinY],
                      (float)x+w, (float)y+h, tc[Texture::CoordMaxX], tc[Texture::CoordMaxY] };
    if (1)        screen->gd->VertexPointer(2, GraphicsDevice::Float, sizeof(float)*4, 0,               verts, sizeof(verts), &verts_ind, true);
    if (texcoord) screen->gd->TexPointer   (2, GraphicsDevice::Float, sizeof(float)*4, sizeof(float)*2, verts, sizeof(verts), &verts_ind, false);
    if (1)        screen->gd->DrawArrays(GraphicsDevice::TriangleStrip, 0, 4);
}

void Box::DrawCrimped(const float *texcoord, int orientation, float scrollX, float scrollY) const {
    float left=x, right=x+w, top=y, bottom=y+h;
    float texMinX, texMinY, texMaxX, texMaxY, texMidX1, texMidX2, texMidY1, texMidY2;

    scrollX *= (texcoord[2] - texcoord[0]);
    scrollY *= (texcoord[3] - texcoord[1]);
    scrollX = ScrollCrimped(texcoord[0], texcoord[2], scrollX, &texMinX, &texMidX1, &texMidX2, &texMaxX);
    scrollY = ScrollCrimped(texcoord[1], texcoord[3], scrollY, &texMinY, &texMidY1, &texMidY2, &texMaxY);

#   define DrawCrimpedBoxTriangleStrip() \
    screen->gd->VertexPointer(2, GraphicsDevice::Float, 4*sizeof(float), 0,               verts, sizeof(verts), &vind, true); \
    screen->gd->TexPointer   (2, GraphicsDevice::Float, 4*sizeof(float), 2*sizeof(float), verts, sizeof(verts), &vind, false); \
    screen->gd->DrawArrays(GraphicsDevice::TriangleStrip, 0, 4); \
    screen->gd->DrawArrays(GraphicsDevice::TriangleStrip, 4, 4); \
    screen->gd->DrawArrays(GraphicsDevice::TriangleStrip, 8, 4); \
    screen->gd->DrawArrays(GraphicsDevice::TriangleStrip, 12, 4);

    switch (orientation) {
        case 0: {
            static int vind = -1;
            float xmid = x + w * scrollX, ymid = y + h * scrollY, verts[] = {
                /*02*/ xmid,  top,  texMidX1, texMaxY,  /*01*/ left, top,  texMinX,  texMaxY,  /*03*/ xmid,  ymid,   texMidX1, texMidY1, /*04*/ left, ymid,   texMinX,  texMidY1,
                /*06*/ right, top,  texMaxX,  texMaxY,  /*05*/ xmid, top,  texMidX2, texMaxY,  /*07*/ right, ymid,   texMaxX,  texMidY1, /*08*/ xmid, ymid,   texMidX2, texMidY1,
                /*10*/ right, ymid, texMaxX,  texMidY2, /*09*/ xmid, ymid, texMidX2, texMidY2, /*11*/ right, bottom, texMaxX,  texMinY,  /*12*/ xmid, bottom, texMidX2, texMinY,
                /*14*/ xmid,  ymid, texMidX1, texMidY2, /*13*/ left, ymid, texMinX,  texMidY2, /*15*/ xmid,  bottom, texMidX1, texMinY,  /*16*/ left, bottom, texMinX,  texMinY 
            };
            DrawCrimpedBoxTriangleStrip();
        } break;
        case 1: {
            static int vind = -1;
            float xmid = x + w * scrollX, ymid = y + h * (1-scrollY), verts[] = {
                /*02*/ xmid,  top,  texMidX1, texMinY,  /*01*/ left,  top,  texMinX,  texMinY,  /*03*/ xmid, ymid,    texMidX1, texMidY2, /*04*/ left, ymid,   texMinX,  texMidY2,
                /*06*/ right, top,  texMaxX,  texMinY,  /*05*/ xmid,  top,  texMidX2, texMinY,  /*07*/ right, ymid,   texMaxX,  texMidY2, /*08*/ xmid, ymid,   texMidX2, texMidY2,
                /*10*/ right, ymid, texMaxX,  texMidY1, /*09*/ xmid,  ymid, texMidX2, texMidY1, /*11*/ right, bottom, texMaxX,  texMaxY,  /*12*/ xmid, bottom, texMidX2, texMaxY,
                /*14*/ xmid,  ymid, texMidX1, texMidY1, /*13*/ left,  ymid, texMinX,  texMidY1, /*15*/ xmid, bottom,  texMidX1, texMaxY,  /*16*/ left, bottom, texMinX,  texMaxY 
            };
            DrawCrimpedBoxTriangleStrip();
        } break;
        case 2: {
            static int vind = -1;
            float xmid = x + w * (1-scrollX), ymid = y + h * scrollY, verts[] = {
                /*02*/ xmid,  top,  texMidX2, texMaxY,  /*01*/ left,  top,  texMaxX,  texMaxY,  /*03*/ xmid, ymid,    texMidX2, texMidY1, /*04*/ left, ymid,   texMaxX,  texMidY1,
                /*06*/ right, top,  texMinX,  texMaxY,  /*05*/ xmid,  top,  texMidX1, texMaxY,  /*07*/ right, ymid,   texMinX,  texMidY1, /*08*/ xmid, ymid,   texMidX1, texMidY1,
                /*10*/ right, ymid, texMinX,  texMidY2, /*09*/ xmid,  ymid, texMidX1, texMidY2, /*11*/ right, bottom, texMinX,  texMinY,  /*12*/ xmid, bottom, texMidX1, texMinY,
                /*14*/ xmid,  ymid, texMidX2, texMidY2, /*13*/ left,  ymid, texMaxX,  texMidY2, /*15*/ xmid, bottom,  texMidX2, texMinY,  /*16*/ left, bottom, texMaxX,  texMinY 
            };
            DrawCrimpedBoxTriangleStrip();
        } break;
        case 3: {
            static int vind = -1;
            float xmid = x + w * (1-scrollX), ymid = y + h * (1-scrollY), verts[] = {
                /*02*/ xmid,  top,  texMidX2, texMinY,  /*01*/ left,  top,   texMaxX,  texMinY,  /*03*/ xmid, ymid,    texMidX2, texMidY2, /*04*/ left, ymid,   texMaxX,  texMidY2,
                /*06*/ right, top,  texMinX,  texMinY,  /*05*/ xmid,  top,   texMidX1, texMinY,  /*07*/ right, ymid,   texMinX,  texMidY2, /*08*/ xmid, ymid,   texMidX1, texMidY2,
                /*10*/ right, ymid, texMinX,  texMidY1, /*09*/ xmid,  ymid,  texMidX1, texMidY1, /*11*/ right, bottom, texMinX,  texMaxY,  /*12*/ xmid, bottom, texMidX1, texMaxY,
                /*14*/ xmid,  ymid, texMidX2, texMidY1, /*13*/ left,  ymid,  texMaxX,  texMidY1, /*15*/ xmid, bottom,  texMidX2, texMaxY,  /*16*/ left, bottom, texMaxX,  texMaxY 
            };
            DrawCrimpedBoxTriangleStrip();
        } break;
        case 4: {
            static int vind = -1;
            float xmid = x + w * (1-scrollY), ymid = y + h * scrollX, verts[] = {
                /*13*/ xmid,  top,  texMinX,  texMidY2, /*16*/ left,  top,  texMinX,  texMaxY,  /*14*/ xmid, ymid,    texMidX1, texMidY2, /*15*/ left, ymid,   texMidX1, texMaxY, 
                /*01*/ right, top,  texMinX,  texMinY,  /*04*/ xmid,  top,  texMinX,  texMidY1, /*02*/ right, ymid,   texMidX1, texMinY,  /*03*/ xmid, ymid,   texMidX1, texMidY1,
                /*05*/ right, ymid, texMidX2, texMinY,  /*08*/ xmid,  ymid, texMidX2, texMidY1, /*06*/ right, bottom, texMaxX,  texMinY,  /*07*/ xmid, bottom, texMaxX,  texMidY1,
                /*09*/ xmid,  ymid, texMidX2, texMidY2, /*12*/ left,  ymid, texMidX2, texMaxY,  /*10*/ xmid, bottom,  texMaxX,  texMidY2, /*11*/ left, bottom, texMaxX,  texMaxY 
            };
            DrawCrimpedBoxTriangleStrip();
        } break;
        case 5: {
            static int vind = -1;
            float xmid = x + w * scrollY, ymid = y + h * scrollX, verts[] = {
                /*13*/ xmid,  top,  texMinX,  texMidY1, /*16*/ left,  top,  texMinX,  texMinY,  /*14*/ xmid, ymid,    texMidX1, texMidY1, /*15*/ left, ymid,   texMidX1, texMinY, 
                /*01*/ right, top,  texMinX,  texMaxY,  /*04*/ xmid,  top,  texMinX,  texMidY2, /*02*/ right, ymid,   texMidX1, texMaxY,  /*03*/ xmid, ymid,   texMidX1, texMidY2,
                /*05*/ right, ymid, texMidX2, texMaxY,  /*08*/ xmid,  ymid, texMidX2, texMidY2, /*06*/ right, bottom, texMaxX,  texMaxY,  /*07*/ xmid, bottom, texMaxX,  texMidY2,
                /*09*/ xmid,  ymid, texMidX2, texMidY1, /*12*/ left,  ymid, texMidX2, texMinY,  /*10*/ xmid, bottom,  texMaxX,  texMidY1, /*11*/ left, bottom, texMaxX,  texMinY 
            };
            DrawCrimpedBoxTriangleStrip();
        } break;
        case 6: {
            static int vind = -1;
            float xmid = x + w * (1-scrollY), ymid = y + h * (1-scrollX), verts[] = {
                /*13*/ xmid,  top,  texMaxX,  texMidY2, /*16*/ left,  top,  texMaxX,  texMaxY,  /*14*/ xmid, ymid,    texMidX2, texMidY2, /*15*/ left, ymid,   texMidX2, texMaxY, 
                /*01*/ right, top,  texMaxX,  texMinY,  /*04*/ xmid,  top,  texMaxX,  texMidY1, /*02*/ right, ymid,   texMidX2, texMinY,  /*03*/ xmid, ymid,   texMidX2, texMidY1,
                /*05*/ right, ymid, texMidX1, texMinY,  /*08*/ xmid,  ymid, texMidX1, texMidY1, /*06*/ right, bottom, texMinX,  texMinY,  /*07*/ xmid, bottom, texMinX,  texMidY1,
                /*09*/ xmid,  ymid, texMidX1, texMidY2, /*12*/ left,  ymid, texMidX1, texMaxY,  /*10*/ xmid, bottom,  texMinX,  texMidY2, /*11*/ left, bottom, texMinX,  texMaxY 
            };
            DrawCrimpedBoxTriangleStrip();
        } break;
        case 7: {
            static int vind = -1;
            float xmid = x + w * scrollY, ymid = y + h * (1-scrollX), verts[] = {
                /*13*/ xmid,  top,  texMaxX,  texMidY1, /*16*/ left,  top,  texMaxX,  texMinY,  /*14*/ xmid, ymid,    texMidX2, texMidY1, /*15*/ left, ymid,   texMidX2, texMinY, 
                /*01*/ right, top,  texMaxX,  texMaxY,  /*04*/ xmid,  top,  texMaxX,  texMidY2, /*02*/ right, ymid,   texMidX2, texMaxY,  /*03*/ xmid, ymid,   texMidX2, texMidY2,
                /*05*/ right, ymid, texMidX1, texMaxY,  /*08*/ xmid,  ymid, texMidX1, texMidY2, /*06*/ right, bottom, texMinX,  texMaxY,  /*07*/ xmid, bottom, texMinX,  texMidY2,
                /*09*/ xmid,  ymid, texMidX1, texMidY1, /*12*/ left,  ymid, texMidX1, texMinY,  /*10*/ xmid, bottom,  texMinX,  texMidY1, /*11*/ left, bottom, texMinX,  texMinY 
            };
            DrawCrimpedBoxTriangleStrip();
        } break;
    }
}

float Box::ScrollCrimped(float tex0, float tex1, float scroll, float *min, float *mid1, float *mid2, float *max) {
    if (tex1 < 1.0 && tex0 == 0.0) {
        *mid1=tex1; *mid2=0;
        if (scroll > 0) *min = *max = tex1 - scroll;
        else            *min = *max = tex0 - scroll;
    } else if (tex0 > 0.0 && tex1 == 1.0) {
        *mid1=1; *mid2=tex0;
        if (scroll > 0) *min = *max = tex0 + scroll;
        else            *min = *max = tex1 + scroll;
    } else if (tex0 == 0 && tex1 == 1) {
        *min = *max = 1;
        *mid1 = tex1; *mid2 = tex0;
    } else {
        return 0;
    }
    return (*mid1 - *min) / (tex1 - tex0); 
}

void Box3::Draw(const point &p, const Color *c) const {
    if (c) screen->gd->SetColor(*c);
    for (int i=0; i<3; i++) if (v[i].h) (v[i] + p).Draw();
}

void Drawable::AttrVec::Insert(const Drawable::Attr &v) {
    if (v.font) font_refs.Insert(&v.font->ref);
    push_back(v);
}

void SimpleVideoResampler::RGB2BGRCopyPixels(unsigned char *dst, const unsigned char *src, int l, int bpp) {
    for (int k = 0; k < l; k++) for (int i = 0; i < bpp; i++) dst[k*bpp+(!i?2:(i==2?0:i))] = src[k*bpp+i];
}

bool SimpleVideoResampler::Supports(int f) { return f == Pixel::RGB24 || f == Pixel::BGR24 || f == Pixel::RGB32 || f == Pixel::BGR32 || f == Pixel::RGBA; }

bool SimpleVideoResampler::Opened() { return s_fmt && d_fmt && s_width && d_width && s_height && d_height; }

void SimpleVideoResampler::Open(int sw, int sh, int sf, int dw, int dh, int df) {
    s_fmt = sf; s_width = sw; s_height = sh;
    d_fmt = df; d_width = dw; d_height = dh;
    // INFO("resample ", BlankNull(Pixel::Name(s_fmt)), " -> ", BlankNull(Pixel::Name(d_fmt)), " : (", sw, ",", sh, ") -> (", dw, ",", dh, ")");
}

void SimpleVideoResampler::Resample(const unsigned char *sb, int sls, unsigned char *db, int dls, bool flip_x, bool flip_y) {
    if (!Opened()) { ERROR("resample not opened()"); return; }

    int sw = Pixel::size(s_fmt), dw = Pixel::size(d_fmt);
    if (sw * s_width > sls) { ERROR(sw * s_width, " > ", sls); return; }
    if (dw * d_width > dls) { ERROR(dw * d_width, " > ", dls); return; }
    
    if (s_width == d_width && s_height == d_height) {
        for (int y=0; y<d_height; y++) {
            for (int x=0; x<d_width; x++) {
                const unsigned char *sp = (sb + sls * y                           + x                          * sw);
                /**/  unsigned char *dp = (db + dls * (flip_y ? d_height-1-y : y) + (flip_x ? d_width-1-x : x) * dw);
                CopyPixel(s_fmt, d_fmt, sp, dp, x == 0, x == d_width-1);
            }
        }
    } else {
        for (int po=0; po<sw && po<dw; po++) {
            Matrix M(s_height, s_width);
            ColorChannelToMatrix(sb, s_width, s_height, sw, sls, 0, 0, &M, po);
            for (int y=0; y<d_height; y++) {
                for (int x=0; x<d_width; x++) {
                    unsigned char *dp = (db + dls * (flip_y ? d_height-1-y : y) + (flip_x ? d_width-1-x : x) * dw);
                    *(dp + po) = MatrixAsFunc(&M, x?(float)x/(d_width-1):0, y?(float)y/(d_height-1):0) * 255;
                }
            }
        }
    }
}

void SimpleVideoResampler::CopyPixel(int s_fmt, int d_fmt, const unsigned char *sp, unsigned char *dp, bool sxb, bool sxe, int f) {
    unsigned char r, g, b, a;
    switch (s_fmt) {
        case Pixel::RGB24: r = *sp++; g = *sp++; b = *sp++; a = ((f & Flag::TransparentBlack) && !r && !g && !b) ? 0 : 255; break;
        case Pixel::BGR24: r = *sp++; g = *sp++; b = *sp++; a = ((f & Flag::TransparentBlack) && !r && !g && !b) ? 0 : 255; break;
        case Pixel::RGB32: r = *sp++; g = *sp++; b = *sp++; a=*sp++; break;
        case Pixel::BGR32: r = *sp++; g = *sp++; b = *sp++; a=*sp++; break;
        case Pixel::RGBA:  r = *sp++; g = *sp++; b = *sp++; a=*sp++; break;
        case Pixel::GRAY8: r = 255;   g = 255;   b = 255;   a=*sp++; break;
        // case Pixel::GRAY8: r = g = b = a = *sp++; break;
        case Pixel::LCD: 
            r = (sxb ? 0 : *(sp-1)) / 3.0 + *sp / 3.0 + (          *(sp+1)) / 3.0; sp++; 
            g = (          *(sp-1)) / 3.0 + *sp / 3.0 + (          *(sp+1)) / 3.0; sp++; 
            b = (          *(sp-1)) / 3.0 + *sp / 3.0 + (sxe ? 0 : *(sp+1)) / 3.0; sp++;
            a = ((f & Flag::TransparentBlack) && !r && !g && !b) ? 0 : 255;
            break;
        default: ERROR("s_fmt ", s_fmt, " not supported"); return;
    }
    switch (d_fmt) {
        case Pixel::RGB24: *dp++ = r; *dp++ = g; *dp++ = b; break;
        case Pixel::BGR24: *dp++ = r; *dp++ = g; *dp++ = b; break;
        case Pixel::RGB32: *dp++ = r; *dp++ = g; *dp++ = b; *dp++ = a; break;
        case Pixel::BGR32: *dp++ = r; *dp++ = g; *dp++ = b; *dp++ = a; break;
        case Pixel::RGBA:  *dp++ = r; *dp++ = g; *dp++ = b; *dp++ = a; break;
        default: ERROR("d_fmt ", d_fmt, " not supported"); return;
    }
}

void SimpleVideoResampler::Blit(const unsigned char *src, unsigned char *dst, int w, int h,
                                int sf, int sls, int sx, int sy,
                                int df, int dls, int dx, int dy, int flag) {
    bool flip_y = flag & Flag::FlipY;
    int sw = Pixel::size(sf), dw = Pixel::size(df); 
    for (int yi = 0; yi < h; ++yi) {
        for (int xi = 0; xi < w; ++xi) {
            int sind = flip_y ? sy + h - yi - 1 : sy + yi;
            const unsigned char *sp = src + (sls*(sind)    + (sx + xi)*sw);
            unsigned       char *dp = dst + (dls*(dy + yi) + (dx + xi)*dw);
            CopyPixel(sf, df, sp, dp, xi == 0, xi == w-1, flag);
        }
    }
}

void SimpleVideoResampler::Filter(unsigned char *buf, int w, int h,
                                  int pf, int ls, int x, int y,
                                  Matrix *kernel, int channel, int flag) {
    Matrix M(h, w), out(h, w);
    int pw = Pixel::size(pf);
    ColorChannelToMatrix(buf, w, h, pw, ls, x, y, &M, ColorChannel::PixelOffset(channel));
    Matrix::Convolve(&M, kernel, &out, (flag & Flag::ZeroOnly) ? mZeroOnly : 0);
    MatrixToColorChannel(&out, w, h, pw, ls, x, y, buf, ColorChannel::PixelOffset(channel));
}

void SimpleVideoResampler::ColorChannelToMatrix(const unsigned char *buf, int w, int h,
                                                int pw, int ls, int x, int y,
                                                Matrix *out, int po) {
    MatrixIter(out) { 
        const unsigned char *p = buf + (ls*(y + i) + (x + j)*pw);
        out->row(i)[j] = *(p + po) / 255.0;
    }
}

void SimpleVideoResampler::MatrixToColorChannel(const Matrix *M, int w, int h,
                                                int pw, int ls, int x, int y,
                                                unsigned char *out, int po) {
    MatrixIter(M) { 
        unsigned char *p = out + (ls*(y + i) + (x + j)*pw);
        *(p + po) = M->row(i)[j] * 255.0;
    }
}

#ifdef LFL_FFMPEG
FFMPEGVideoResampler::~FFMPEGVideoResampler() { if (conv) sws_freeContext((SwsContext*)conv); }
bool FFMPEGVideoResampler::Opened() { return conv || simple_resampler_passthru; }

void FFMPEGVideoResampler::Open(int sw, int sh, int sf, int dw, int dh, int df) {
    s_fmt = sf; s_width = sw; s_height = sh;
    d_fmt = df; d_width = dw; d_height = dh;
    // INFO("resample ", BlankNull(Pixel::Name(s_fmt)), " -> ", BlankNull(Pixel::Name(d_fmt)), " : (", sw, ",", sh, ") -> (", dw, ",", dh, ")");

    if (SimpleVideoResampler::Supports(s_fmt) && SimpleVideoResampler::Supports(d_fmt) && sw == dw && sh == dh)
    { simple_resampler_passthru = 1; return; }

    conv = sws_getContext(sw, sh, (PixelFormat)Pixel::ToFFMpegId(sf),
                          dw, dh, (PixelFormat)Pixel::ToFFMpegId(df), SWS_BICUBIC, 0, 0, 0);
}

void FFMPEGVideoResampler::Resample(const unsigned char *s, int sls, unsigned char *d, int dls, bool flip_x, bool flip_y) {
    if (simple_resampler_passthru) return SimpleVideoResampler::Resample(s, sls, d, dls, flip_x, flip_y);
    uint8_t *source  [4] = { (uint8_t*)s, 0, 0, 0 }, *dest[  4] = { (uint8_t*)d, 0, 0, 0 };
    int      sourcels[4] = {         sls, 0, 0, 0 },  destls[4] = {         dls, 0, 0, 0 };
    if (flip_y) {
        source[0] += sls * (s_height - 1);
        sourcels[0] *= -1;
    }
    sws_scale((SwsContext*)conv,
        flip_y ? source   : source,
        flip_y ? sourcels : sourcels, 0, s_height, dest, destls);
}

int Pixel::FromFFMpegId(int fmt) {
    switch (fmt) {
        case AV_PIX_FMT_RGB32:    return Pixel::RGB32;
        case AV_PIX_FMT_BGR32:    return Pixel::BGR32;
        case AV_PIX_FMT_RGB24:    return Pixel::RGB24;
        case AV_PIX_FMT_BGR24:    return Pixel::BGR24;
        case AV_PIX_FMT_GRAY8:    return Pixel::GRAY8;
        case AV_PIX_FMT_YUV410P:  return Pixel::YUV410P;
        case AV_PIX_FMT_YUV420P:  return Pixel::YUV420P;
        case AV_PIX_FMT_YUYV422:  return Pixel::YUYV422;
        case AV_PIX_FMT_YUVJ420P: return Pixel::YUVJ420P;
        case AV_PIX_FMT_YUVJ422P: return Pixel::YUVJ422P;
        case AV_PIX_FMT_YUVJ444P: return Pixel::YUVJ444P;
        default: ERROR("unknown pixel fmt: ", fmt); return 0;
    }
}

int Pixel::ToFFMpegId(int fmt) {
    switch (fmt) {
        case Pixel::RGB32:    return AV_PIX_FMT_RGB32;
        case Pixel::BGR32:    return AV_PIX_FMT_BGR32;
        case Pixel::RGB24:    return AV_PIX_FMT_RGB24;
        case Pixel::BGR24:    return AV_PIX_FMT_BGR24;
        case Pixel::RGBA:     return AV_PIX_FMT_RGBA;
        case Pixel::GRAY8:    return AV_PIX_FMT_GRAY8;
        case Pixel::YUV410P:  return AV_PIX_FMT_YUV410P;
        case Pixel::YUV420P:  return AV_PIX_FMT_YUV420P;
        case Pixel::YUYV422:  return AV_PIX_FMT_YUYV422;
        case Pixel::YUVJ420P: return AV_PIX_FMT_YUVJ420P;
        case Pixel::YUVJ422P: return AV_PIX_FMT_YUVJ422P;
        case Pixel::YUVJ444P: return AV_PIX_FMT_YUVJ444P;
        default: ERROR("unknown pixel fmt: ", fmt); return 0;
    }
}
#endif /* LFL_FFMPEG */

/* Texture */

void Texture::Coordinates(float *texcoord, int w, int h, int wd, int hd) {
    texcoord[CoordMinX] = texcoord[CoordMinY] = 0;
    texcoord[CoordMaxX] = (float)w / wd;
    texcoord[CoordMaxY] = (float)h / hd;
}

void Texture::Resize(int W, int H, int PF, int flag) {
    if (PF) pf = PF;
    width=W; height=H;
    if (buf || (flag & Flag::CreateBuf)) RenewBuffer();
    if (!ID && (flag & Flag::CreateGL)) {
        if (!cubemap) {
            screen->gd->DisableCubeMap();
            screen->gd->GenTextures(GL_TEXTURE_2D, 1, &ID);
        } else if (cubemap == CubeMap::PX) {
            screen->gd->ActiveTexture(0);
            screen->gd->GenTextures(GL_TEXTURE_CUBE_MAP, 1, &ID);
            glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        }
    }
    if (ID || cubemap) {
        int opengl_width = NextPowerOfTwo(width), opengl_height = NextPowerOfTwo(height);
        int gl_tt = GLTexType(), gl_pt = GLPixelType();
        if (ID) screen->gd->BindTexture(gl_tt, ID);
        glTexImage2D(gl_tt, 0, gl_pt, opengl_width, opengl_height, 0, gl_pt, GL_UNSIGNED_BYTE, 0);
        Coordinates(coord, width, height, opengl_width, opengl_height);
    }
}

void Texture::LoadBuffer(const unsigned char *B, const point &dim, int PF, int linesize, int flag) {
    Resize(dim.x, dim.y, pf, Flag::CreateBuf);
    SimpleVideoResampler::Blit(B, buf, width, height,
                               PF, linesize,   0, 0,
                               pf, LineSize(), 0, 0, flag);
}

void Texture::UpdateBuffer(const unsigned char *B, const point &dim, int PF, int linesize, int flag) {
    bool resample = flag & Flag::Resample;
    VideoResampler conv;
    conv.Open(dim.x, dim.y, PF, resample ? width : dim.x, resample ? height : dim.y, pf);
    conv.Resample(B, linesize, buf, LineSize(), 0, flag & Flag::FlipY);
}

void Texture::UpdateBuffer(const unsigned char *B, const ::LFL::Box &box, int PF, int linesize, int blit_flag) {
    SimpleVideoResampler::Blit(B, buf, box.w, box.h, PF, linesize, 0, 0, pf, LineSize(), box.x, box.y, blit_flag);
}

void Texture::Bind() const { screen->gd->BindTexture(GLTexType(), ID); }
void Texture::ClearGL() { if (ID) screen->gd->DelTextures(1, &ID); ID=0; }

void Texture::LoadGL(const unsigned char *B, const point &dim, int PF, int linesize, int flag) {
    Texture temp;
    temp .Resize(dim.x, dim.y, Pixel::RGBA, Flag::CreateBuf);
    temp .UpdateBuffer(B, dim, PF, linesize, Flag::FlipY);
    this->Resize(dim.x, dim.y, Pixel::RGBA, Flag::CreateGL);
    this->UpdateGL(temp.buf, LFL::Box(dim), flag);
}

void Texture::UpdateGL(const unsigned char *B, const ::LFL::Box &box, int flag) {
    int gl_tt = GLTexType(), gl_y = (flag & Flag::FlipY) ? (height - box.y - box.h) : box.y;
    screen->gd->BindTexture(gl_tt, ID);
    glTexSubImage2D(gl_tt, 0, box.x, gl_y, box.w, box.h, GLPixelType(), GL_UNSIGNED_BYTE, B);
}

void Texture::DumpGL(unsigned tex_id) {
    if (tex_id) {
        GLint gl_tt = GLTexType(), tex_w = 0, tex_h = 0;
        screen->gd->BindTexture(gl_tt, tex_id);
        glGetTexLevelParameteriv(gl_tt, 0, GL_TEXTURE_WIDTH, &tex_w);
        glGetTexLevelParameteriv(gl_tt, 0, GL_TEXTURE_WIDTH, &tex_h);
        CHECK_GT((width  = tex_w), 0);
        CHECK_GT((height = tex_h), 0);
    }
    RenewBuffer();
    glGetTexImage(GLTexType(), 0, GLPixelType(), GL_UNSIGNED_BYTE, buf);
}

void Texture::ToIplImage(_IplImage *out) {
#ifdef LFL_OPENCV
    memset(out, 0, sizeof(IplImage));
    out->nSize = sizeof(IplImage);
    out->nChannels = Pixel::size(pf);
    out->depth = IPL_DEPTH_8U;
    out->origin = 1;
    out->width = width;
    out->height = height;
    out->widthStep = out->width * out->nChannels;
    out->imageSize = out->widthStep * out->height;
    out->imageData = (char*)buf;
    out->imageDataOrigin = out->imageData;
#else
    ERROR("ToIplImage not implemented");
#endif
}

#ifdef __APPLE__
#import <CoreGraphics/CGBitmapContext.h> 
CGContextRef Texture::CGBitMap() { return CGBitMap(0, 0, width, height); }
CGContextRef Texture::CGBitMap(int X, int Y, int W, int H) {
    int linesize = LineSize(); CGImageAlphaInfo alpha_info;
    if      (pf == Pixel::RGBA)                        alpha_info = kCGImageAlphaPremultipliedLast;
    else if (pf == Pixel::RGB32 || pf == Pixel::BGR32) alpha_info = kCGImageAlphaNoneSkipLast;
    else { ERROR("unsupported pixel format: ", Pixel::Name(pf)); return 0; }
    CGColorSpaceRef colors = CGColorSpaceCreateDeviceRGB();
    CGContextRef ret = CGBitmapContextCreate(buf + Y*linesize + X*PixelSize(), W, H, 8, linesize, colors, alpha_info);
    CGColorSpaceRelease(colors);
    return ret;
}
#endif

void Texture::Screenshot() {
    Resize(screen->width, screen->height, Pixel::RGBA, Flag::CreateBuf);
    unsigned char *pixels = NewBuffer();
    glReadPixels(0, 0, screen->width, screen->height, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    UpdateBuffer(pixels, point(screen->width, screen->height), Pixel::RGBA, screen->width*4, Flag::FlipY);
    delete [] pixels;
}

/* DepthTexture */

void DepthTexture::Resize(int W, int H, int DF, int flag) {
    if (DF) df = DF;
    width=W; height=H;
    if (!ID && (flag & Flag::CreateGL)) glGenRenderbuffers(1, &ID);
    int opengl_width = NextPowerOfTwo(width), opengl_height = NextPowerOfTwo(height);
    if (ID) {
        glBindRenderbuffer(GL_RENDERBUFFER, ID);
        glRenderbufferStorage(GL_RENDERBUFFER, Depth::OpenGLID(df), opengl_width, opengl_height);
    }
}

/* FrameBuffer */

void FrameBuffer::Resize(int W, int H, int flag) {
    width=W; height=H;
    if (!ID && (flag & Flag::CreateGL)) {
        glGenFramebuffers(1, &ID);
        if (flag & Flag::CreateTexture)      AllocTexture(&tex);
        if (flag & Flag::CreateDepthTexture) AllocDepthTexture(&depth);
    } else {
        tex.Resize(width, height);
        depth.Resize(width, height);
    }
    Attach(tex.ID, depth.ID);
    int status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if (status != GL_FRAMEBUFFER_COMPLETE) ERROR("FrameBuffer status ", status);
    if (flag & Flag::ReleaseFB) Release();
}

void FrameBuffer::AllocDepthTexture(DepthTexture *out) { CHECK_EQ(out->ID, 0); out->Create(width, height); }
void FrameBuffer::AllocTexture     (     Texture *out) { CHECK_EQ(out->ID, 0); out->Create(width, height); 
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
}

void FrameBuffer::Release() { glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0); }
void FrameBuffer::Attach(int ct, int dt) {
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, ID);
    if (ct) {
        tex.ID = ct;
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex.ID, 0);
    }
    if (dt) {
        depth.ID = dt;
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth.ID);
    }
}

void FrameBuffer::Render(FrameCB cb) {
    int dm = screen->gd->draw_mode;
    Attach();
    screen->gd->ViewPort(Box(0, 0, tex.width, tex.height));
    screen->gd->Clear();
    cb(0, 0, 0, 0, 0);
    Release();
    screen->gd->RestoreViewport(dm);
}

/* Shader */

void Shader::SetGlobalUniform1f(const string &name, float v) {
    screen->gd->UseShader(&app->video.shader_default);  app->video.shader_default .SetUniform1f(name, v);
    screen->gd->UseShader(&app->video.shader_normals);  app->video.shader_normals .SetUniform1f(name, v);
    screen->gd->UseShader(&app->video.shader_cubemap);  app->video.shader_cubemap .SetUniform1f(name, v);
    screen->gd->UseShader(&app->video.shader_cubenorm); app->video.shader_cubenorm.SetUniform1f(name, v);
}

void Shader::SetGlobalUniform2f(const string &name, float v1, float v2){ 
    screen->gd->UseShader(&app->video.shader_default);  app->video.shader_default .SetUniform2f(name, v1, v2);
    screen->gd->UseShader(&app->video.shader_normals);  app->video.shader_normals .SetUniform2f(name, v1, v2);
    screen->gd->UseShader(&app->video.shader_cubemap);  app->video.shader_cubemap .SetUniform2f(name, v1, v2);
    screen->gd->UseShader(&app->video.shader_cubenorm); app->video.shader_cubenorm.SetUniform2f(name, v1, v2);
}

#ifdef LFL_GLSL_SHADERS
int Shader::Create(const string &name, const string &vertex_shader, const string &fragment_shader, const ShaderDefines &defines, Shader *out) {
    GLuint p = screen->gd->CreateProgram();

    string hdr; 
#ifdef LFL_GLES2
    if (screen->opengles_version == 2) hdr += "#define LFL_GLES2\r\n";
#endif
    hdr += defines.text + string("\r\n");

    if (vertex_shader.size()) {
        GLuint vs = screen->gd->CreateShader(GL_VERTEX_SHADER);
        const char *vss[] = { hdr.c_str(), vertex_shader.c_str(), 0 };
        screen->gd->ShaderSource(vs, 2, vss, 0);
        screen->gd->CompileShader(vs);
        screen->gd->AttachShader(p, vs);
    }

    if (fragment_shader.size()) {
        GLuint fs = screen->gd->CreateShader(GL_FRAGMENT_SHADER);
        const char *fss[] = { hdr.c_str(), fragment_shader.c_str(), 0 };
        screen->gd->ShaderSource(fs, 2, fss, 0);
        screen->gd->CompileShader(fs);
        screen->gd->AttachShader(p, fs);
    }

    if (1)                    screen->gd->BindAttribLocation(p, 0, "Position"   );
    if (defines.normals)      screen->gd->BindAttribLocation(p, 1, "Normal"     );
    if (defines.vertex_color) screen->gd->BindAttribLocation(p, 2, "VertexColor");
    if (defines.tex_2d)       screen->gd->BindAttribLocation(p, 3, "TexCoordIn" );

    screen->gd->LinkProgram(p);

    int active_uniforms=0, max_uniform_components=0, active_attributes=0, max_attributes=0;
    screen->gd->GetProgramiv(p, GL_ACTIVE_UNIFORMS, &active_uniforms);
    screen->gd->GetProgramiv(p, GL_ACTIVE_ATTRIBUTES, &active_attributes);
#if !defined(LFL_ANDROID) && !defined(LFL_IPHONE)
    screen->gd->GetIntegerv(GL_MAX_VERTEX_UNIFORM_COMPONENTS, &max_uniform_components);
#endif
    screen->gd->GetIntegerv(GL_MAX_VERTEX_ATTRIBS, &max_attributes);
    INFO("shader mu=", active_uniforms, " avg_comps/", max_uniform_components, ", ma=", active_attributes, "/", max_attributes);

    bool log_missing_attrib = false;
    if (out) {
        *out = Shader();
        out->ID = p;
        out->name = name;
        if ((out->slot_position             = screen->gd->GetAttribLocation (p, "Position"))            < 0 && log_missing_attrib) INFO("shader ", name, " missing Position");
        if ((out->slot_normal               = screen->gd->GetAttribLocation (p, "Normal"))              < 0 && log_missing_attrib) INFO("shader ", name, " missing Normal");
        if ((out->slot_color                = screen->gd->GetAttribLocation (p, "VertexColor"))         < 0 && log_missing_attrib) INFO("shader ", name, " missing VertexColor");
        if ((out->slot_tex                  = screen->gd->GetAttribLocation (p, "TexCoordIn"))          < 0 && log_missing_attrib) INFO("shader ", name, " missing TexCoordIn");
        if ((out->uniform_modelview         = screen->gd->GetUniformLocation(p, "Modelview"))           < 0 && log_missing_attrib) INFO("shader ", name, " missing Modelview");
        if ((out->uniform_modelviewproj     = screen->gd->GetUniformLocation(p, "ModelviewProjection")) < 0 && log_missing_attrib) INFO("shader ", name, " missing ModelviewProjection");
        if ((out->uniform_tex               = screen->gd->GetUniformLocation(p, "Texture"))             < 0 && log_missing_attrib) INFO("shader ", name, " missing Texture");
        if ((out->uniform_cubetex           = screen->gd->GetUniformLocation(p, "CubeTexture"))         < 0 && log_missing_attrib) INFO("shader ", name, " missing CubeTexture");
        if ((out->uniform_normalon          = screen->gd->GetUniformLocation(p, "NormalEnabled"))       < 0 && log_missing_attrib) INFO("shader ", name, " missing NormalEnabled");
        if ((out->uniform_texon             = screen->gd->GetUniformLocation(p, "TexCoordEnabled"))     < 0 && log_missing_attrib) INFO("shader ", name, " missing TexCoordEnabled");
        if ((out->uniform_coloron           = screen->gd->GetUniformLocation(p, "VertexColorEnabled"))  < 0 && log_missing_attrib) INFO("shader ", name, " missing VertexColorEnabled");
        if ((out->uniform_cubeon            = screen->gd->GetUniformLocation(p, "CubeMapEnabled"))      < 0 && log_missing_attrib) INFO("shader ", name, " missing CubeMapEnabled");
        if ((out->uniform_colordefault      = screen->gd->GetUniformLocation(p, "DefaultColor"))        < 0 && log_missing_attrib) INFO("shader ", name, " missing DefaultColor");
        if ((out->uniform_material_ambient  = screen->gd->GetUniformLocation(p, "MaterialAmbient"))     < 0 && log_missing_attrib) INFO("shader ", name, " missing MaterialAmbient");
        if ((out->uniform_material_diffuse  = screen->gd->GetUniformLocation(p, "MaterialDiffuse"))     < 0 && log_missing_attrib) INFO("shader ", name, " missing MaterialDiffuse");
        if ((out->uniform_material_specular = screen->gd->GetUniformLocation(p, "MaterialSpecular"))    < 0 && log_missing_attrib) INFO("shader ", name, " missing MaterialSpecular");
        if ((out->uniform_material_emission = screen->gd->GetUniformLocation(p, "MaterialEmission"))    < 0 && log_missing_attrib) INFO("shader ", name, " missing MaterialEmission");
        if ((out->uniform_light0_pos        = screen->gd->GetUniformLocation(p, "LightZeroPosition"))   < 0 && log_missing_attrib) INFO("shader ", name, " missing LightZeroPosition");
        if ((out->uniform_light0_ambient    = screen->gd->GetUniformLocation(p, "LightZeroAmbient"))    < 0 && log_missing_attrib) INFO("shader ", name, " missing LightZeroAmbient");
        if ((out->uniform_light0_diffuse    = screen->gd->GetUniformLocation(p, "LightZeroDiffuse"))    < 0 && log_missing_attrib) INFO("shader ", name, " missing LightZeroDiffuse");
        if ((out->uniform_light0_specular   = screen->gd->GetUniformLocation(p, "LightZeroSpecular"))   < 0 && log_missing_attrib) INFO("shader ", name, " missing LightZeroSpecular");

        int unused_attrib = 0;
        memset(out->unused_attrib_slot, -1, sizeof(out->unused_attrib_slot));
        for (int i=0; i<MaxVertexAttrib; i++) {
            if (out->slot_position == i || out->slot_normal == i || out->slot_color == i || out->slot_tex == i) continue;
            out->unused_attrib_slot[unused_attrib++] = i;
        }
    }

    return p;
}

int Shader::GetUniformIndex(const string &name) { return screen->gd->GetUniformLocation(ID, name); }
void Shader::SetUniform1i(const string &name, float v) { screen->gd->Uniform1i(GetUniformIndex(name), v); }
void Shader::SetUniform1f(const string &name, float v) { screen->gd->Uniform1f(GetUniformIndex(name), v); }
void Shader::SetUniform2f(const string &name, float v1, float v2) { screen->gd->Uniform2f(GetUniformIndex(name), v1, v2); }
void Shader::SetUniform3fv(const string &name, const float *v) { screen->gd->Uniform3fv(GetUniformIndex(name), 1, v); }
void Shader::SetUniform3fv(const string &name, int n, const float *v) { screen->gd->Uniform3fv(GetUniformIndex(name), n, v); }

#else /* LFL_GLSL_SHADERS */

int Shader::Create(const string &vertex_shader, const string &fragment_shader, const string &defines, Shader *out) { return -1; }
int Shader::GetUniformIndex(const string &name) { return -1; }
void Shader::SetUniform1i(const string &name, float v) {}
void Shader::SetUniform1f(const string &name, float v) {}
void Shader::SetUniform2f(const string &name, float v1, float v2) {}
void Shader::SetUniform3fv(const string &name, const float *v) {}
void Shader::SetUniform3fv(const string &name, int n, const float *v) {}
void Shader::ActiveTexture(int n) {}
#endif /* LFL_GLSL_SHADERS */

/* BoxRun */

point DrawableBoxRun::Draw(point p, DrawCB cb) {
    Box w;
    DrawBackground(p);
    if (attr->tex) attr->tex->Bind();
    if (attr->tex || attr->font) screen->gd-> SetColor(attr->fg ? *attr->fg : Color::white);
    else                         screen->gd->FillColor(attr->fg ? *attr->fg : Color::white);
    if (attr->font) attr->font->Select();
    else if (attr->tex) screen->gd->EnableLayering();
    if (attr->scissor) screen->gd->PushScissor(*attr->scissor + p);
    for (auto i = data.buf, e = data.end(); i != e; ++i) if (i->drawable) cb(i->drawable, (w = i->box + p), attr);
    if (attr->scissor) screen->gd->PopScissor();
    return point(w.x + w.w, w.y);
}

void DrawableBoxRun::DrawBackground(point p, DrawBackgroundCB cb) {
    if (attr->bg) screen->gd->FillColor(*attr->bg);
    if (!attr->bg || !data.size()) return;
    int line_height = line ? line->h : (attr->font ? attr->font->Height() : 0);
    if (!line_height) return;
    int left = data[0].LeftBound(attr), right = data.back().RightBound(attr);
    cb(Box(p.x + left, p.y - line_height, right - left, line_height));
}

}; // namespace LFL
