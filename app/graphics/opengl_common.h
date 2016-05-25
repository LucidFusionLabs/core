/*
 * $Id: camera.cpp 1330 2014-11-06 03:04:15Z justin $
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

void Flush() { ClearDeferred(); glFlush(); }
void Clear() { glClear(GL_COLOR_BUFFER_BIT | (draw_mode == DrawMode::_3D ? GL_DEPTH_BUFFER_BIT : 0)); }
void ClearDepth() { glClear(GL_DEPTH_BUFFER_BIT); }
void ClearColor(const Color &c) { clear_color=c; glClearColor(c.r(), c.g(), c.b(), c.a()); GDDebug("ClearColor=", c.DebugString()); }
void PointSize(float n) { glPointSize(n); }
void LineWidth(float n) { glLineWidth(n); }
void DelTextures(int n, const unsigned *id) {
  ClearDeferred();
  if (FLAGS_gd_debug) for (int i=0; i<n; i++) GDDebug("DelTexture ", id[i]);
  glDeleteTextures(n, id);
}
void TexImage2D(int targ, int l, int fi, int w, int h, int b, int f, int t, const void *data) { glTexImage2D(targ, l, fi, w, h, b, f, t, data); } 
void TexSubImage2D(int targ, int l, int xo, int yo, int w, int h, int f, int t, const void *data) { glTexSubImage2D(targ, l, xo, yo, w, h, f, t, data); }
void GenTextures(int t, int n, unsigned *out) {
  ClearDeferred();
  for (int i=0; i<n; i++) CHECK_EQ(0, out[i]);
  if (t == GL_TEXTURE_CUBE_MAP) glEnable(GL_TEXTURE_CUBE_MAP);
  glGenTextures(n, out);
  for (int i=0; i<n; i++) {
    glBindTexture(t, out[i]);
    glTexParameteri(t, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(t, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    GDDebug("GenTexture ", t, ",", out[i]);
  }
}

void CheckForError(const char *file, int line) {
  GLint gl_error=0, gl_validate_status=0;
  if ((gl_error = glGetError())) {
    ERROR(file, ":", line, " gl error: ", gl_error);
    BreakHook();
    if (app->opengles_version == 2) {
      glValidateProgram(shader->ID);
      glGetProgramiv(shader->ID, GL_VALIDATE_STATUS, &gl_validate_status);
      if (gl_validate_status != GL_TRUE) ERROR(shader->name, ": gl validate status ", gl_validate_status);

      char buf[1024]; int len;
      glGetProgramInfoLog(shader->ID, sizeof(buf), &len, buf);
      if (len) INFO(buf);
    }
  }
}

void EnableScissor() { glEnable(GL_SCISSOR_TEST); }
void DisableScissor() { glDisable(GL_SCISSOR_TEST); }
void EnableDepthTest()  {  glEnable(GL_DEPTH_TEST); glDepthMask(GL_TRUE);  GDDebug("DepthTest=1"); }
void DisableDepthTest() { glDisable(GL_DEPTH_TEST); glDepthMask(GL_FALSE); GDDebug("DepthTest=0"); }
void DisableBlend() { if (Changed(&blend_enabled, false)) { ClearDeferred(); glDisable(GL_BLEND);                                                    GDDebug("Blend=0"); } }
void EnableBlend()  { if (Changed(&blend_enabled, true )) { ClearDeferred();  glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); GDDebug("Blend=1"); } }
void BlendMode(int sm, int dm) { ClearDeferred(); glBlendFunc(sm, dm); GDDebug("BlendMode=", sm, ",", dm); }
void ViewPort(Box w) {
  if (FLAGS_swap_axis) w.swapaxis(LFL::screen->width, LFL::screen->height);
  ClearDeferred();
  glViewport(w.x, w.y, w.w, w.h);
}

void Scissor(Box w) {
  if (FLAGS_swap_axis) w.swapaxis(LFL::screen->width, LFL::screen->height);
  ClearDeferred();
  EnableScissor();
  glScissor(w.x, w.y, w.w, w.h);
}

void TexParameter(int t, int p, int v) {
  glTexParameteri(t, p, v);
  GDDebug("TexParameter ", t, " ", p, " ", v);
}

void GenRenderBuffers(int n, unsigned *out) {
  glGenRenderbuffersEXT(n, out);
  if (FLAGS_gd_debug) for (int i=0; i<n; i++) GDDebug("GenRenderBuffer ", out[i]);
}

void DelRenderBuffers(int n, const unsigned *id) { 
  if (FLAGS_gd_debug) for (int i=0; i<n; i++) GDDebug("DelRenderBuffer ", id[i]);
  glDeleteRenderbuffersEXT(n, id);
}

void BindRenderBuffer(int id) { glBindRenderbufferEXT(GL_RENDERBUFFER, id); }
void RenderBufferStorage(int d, int w, int h) { glRenderbufferStorageEXT(GL_RENDERBUFFER, d, w, h); }
void GenFrameBuffers(int n, unsigned *out) {
  glGenFramebuffersEXT(n, out);
  if (FLAGS_gd_debug) for (int i=0; i<n; i++) GDDebug("GenFrameBuffer ", out[i]);
}

void DelFrameBuffers(int n, const unsigned *id) {
  if (FLAGS_gd_debug) for (int i=0; i<n; i++) GDDebug("DelFrameBuffer ", id[i]);
  glDeleteFramebuffersEXT(n, id);
}

void BindFrameBuffer(int id) { ClearDeferred(); glBindFramebufferEXT(GL_FRAMEBUFFER, id); }
void FrameBufferTexture(int id) { glFramebufferTexture2DEXT(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, id, 0); }
void FrameBufferDepthTexture(int id) { glFramebufferRenderbufferEXT(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, id); }
int CheckFrameBufferStatus() { return glCheckFramebufferStatusEXT(GL_FRAMEBUFFER); }

void Screenshot(Texture *out) { ScreenshotBox(out, Box(LFL::screen->width, LFL::screen->height), Texture::Flag::FlipY); }
void ScreenshotBox(Texture *out, const Box &b, int flag) {
  ClearDeferred();
  out->Resize(b.w, b.h, Texture::preferred_pf, Texture::Flag::CreateBuf);
  unsigned char *pixels = out->NewBuffer();
  glReadPixels(b.x, b.y, b.w, b.h, out->GLPixelType(), out->GLBufferType(), pixels);
  out->UpdateBuffer(pixels, point(b.w, b.h), out->pf, b.w*4, flag);
  delete [] pixels;
}

void DumpTexture(Texture *out, unsigned tex_id) {
#if !defined(LFL_MOBILE) && !defined(LFL_EMSCRIPTEN)
  if (tex_id) {
    GLint gl_tt = out->GLTexType(), tex_w = 0, tex_h = 0;
    BindTexture(gl_tt, tex_id);
    glGetTexLevelParameteriv(gl_tt, 0, GL_TEXTURE_WIDTH, &tex_w);
    glGetTexLevelParameteriv(gl_tt, 0, GL_TEXTURE_WIDTH, &tex_h);
    CHECK_GT((out->width  = tex_w), 0);
    CHECK_GT((out->height = tex_h), 0);
  }
  out->RenewBuffer();
  glGetTexImage(out->GLTexType(), 0, out->GLPixelType(), out->GLBufferType(), out->buf);
#endif
}

const char *GetString(int t) { return SpellNull(MakeSigned(glGetString(t))); }
const char *GetGLEWString(int t) {
#ifdef LFL_GLEW
  return SpellNull(MakeSigned(glewGetString(t)));
#else
  return "";
#endif
}

int CreateProgram() { int p=glCreateProgram(); GDDebug("CreateProgram ", p); return p; }
void DelProgram(int p) { glDeleteProgram(p); GDDebug("DelProgram ", p); }
int CreateShader(int t) { return glCreateShader(t); }
void ShaderSource(int shader, int count, const char **source, int *len) { glShaderSource(shader, count, source, len); }
void CompileShader(int shader) {
  char buf[1024] = {0}; int l=0;
  glCompileShader(shader);
  glGetShaderInfoLog(shader, sizeof(buf), &l, buf);
  if (l) INFO(buf);
}
void AttachShader(int prog, int shader) { glAttachShader(prog, shader); }
void DelShader(int shader) { glDeleteShader(shader); }
void BindAttribLocation(int prog, int loc, const string &name) { glBindAttribLocation(prog, loc, name.c_str()); }
void LinkProgram(int prog) {
  char buf[1024] = {0}; int l=0;
  glLinkProgram(prog);
  glGetProgramInfoLog(prog, sizeof(buf), &l, buf);
  if (l) INFO(buf);
  GLint link_status;
  glGetProgramiv(prog, GL_LINK_STATUS, &link_status);
  if (link_status != GL_TRUE) FATAL("link failed");
}
void GetProgramiv(int p, int t, int *out) { glGetProgramiv(p, t, out); }
void GetIntegerv(int t, int *out) { glGetIntegerv(t, out); }
int GetAttribLocation (int prog, const string &name) { return glGetAttribLocation (prog, name.c_str()); }
int GetUniformLocation(int prog, const string &name) { return glGetUniformLocation(prog, name.c_str()); }
void Uniform1i(int u, int v) { glUniform1i(u, v); }
void Uniform1f(int u, float v) { glUniform1f(u, v); }
void Uniform2f(int u, float v1, float v2) { glUniform2f(u, v1, v2); }
void Uniform3f(int u, float v1, float v2, float v3) { glUniform3f(u, v1, v2, v3); }
void Uniform4f(int u, float v1, float v2, float v3, float v4) { glUniform4f(u, v1, v2, v3, v4); }
void Uniform3fv(int u, int n, const float *v) { glUniform3fv(u, n, v); }

void InitDefaultLight() {
  float pos[]={-.5,1,-.3f,0}, grey20[]={.2f,.2f,.2f,1}, white[]={1,1,1,1}, black[]={0,0,0,1};
  EnableLight(0);
  Light(0, GraphicsDevice::Position, pos);
  Light(0, GraphicsDevice::Ambient,  grey20);
  Light(0, GraphicsDevice::Diffuse,  white);
  Light(0, GraphicsDevice::Specular, white);
  Material(GraphicsDevice::Emission, black);
  Material(GraphicsDevice::Specular, grey20);
}

void LogVersion() {
  const char *glslver = GetString(ShaderVersion);
  const char *glexts = GetString(Extensions);
  INFO("OpenGL Version: ", GetString(Version));
  INFO("OpenGL Vendor: ", GetString(Vendor));
  INFO("GLEW Version: ", GetGLEWString(GLEWVersion));
  INFO("GL_SHADING_LANGUAGE_VERSION: ", glslver);
  INFO("GL_EXTENSIONS: ", glexts);

  app->opengles_version = 1 + (glslver != NULL);
#ifdef LFL_MOBILE
  have_cubemap = strstr(glexts, "GL_EXT_texture_cube_map") != 0;
#else
  have_cubemap = strstr(glexts, "GL_ARB_texture_cube_map") != 0;
#endif
  int depth_bits=0;
  GetIntegerv(DepthBits, &depth_bits);
  INFO("opengles_version = ", app->opengles_version, ", depth_bits = ", depth_bits);
  INFO("have_cubemap = ", have_cubemap ? "true" : "false");

#if 0
  int dim[2] = { 0, 0 };
  screen->gd->GetIntegerv(GraphicsDevice::MaxViewportDims, dim);
  INFO("max_viewport_dims = ", dim[0], ", ", dim[1]);
#endif
}
