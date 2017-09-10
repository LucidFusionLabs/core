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

namespace LFL {
const int GraphicsDevice::Float = 0;
const int GraphicsDevice::Points = 0;
const int GraphicsDevice::Lines = 0;
const int GraphicsDevice::LineLoop = 0;
const int GraphicsDevice::Triangles = 0;
const int GraphicsDevice::TriangleStrip = 0;
const int GraphicsDevice::Polygon = 0;
const int GraphicsDevice::Texture2D = 0;
const int GraphicsDevice::TextureCubeMap = 0;
const int GraphicsDevice::UnsignedByte = 0;
const int GraphicsDevice::UnsignedInt = 0;
const int GraphicsDevice::FramebufferComplete = 0;
const int GraphicsDevice::FramebufferUndefined = 0;
const int GraphicsDevice::FramebufferBinding = 0;
const int GraphicsDevice::Ambient = 0;
const int GraphicsDevice::Diffuse = 0;
const int GraphicsDevice::Specular = 0;
const int GraphicsDevice::Emission = 0;
const int GraphicsDevice::Position = 0;
const int GraphicsDevice::One = 0;
const int GraphicsDevice::SrcAlpha = 0;
const int GraphicsDevice::OneMinusSrcAlpha = 0;
const int GraphicsDevice::OneMinusDstColor = 0;
const int GraphicsDevice::TextureWrapS = 0;
const int GraphicsDevice::TextureWrapT = 0;
const int GraphicsDevice::ClampToEdge = 0;
const int GraphicsDevice::VertexShader = 0;
const int GraphicsDevice::FragmentShader = 0;
const int GraphicsDevice::ShaderVersion = 0;
const int GraphicsDevice::Extensions = 0;
const int GraphicsDevice::GLEWVersion = 0;
const int GraphicsDevice::Version = 0;
const int GraphicsDevice::Vendor = 0;
const int GraphicsDevice::DepthBits = 0;
const int GraphicsDevice::ScissorTest = 0;
const int GraphicsDevice::ActiveUniforms = 0;
const int GraphicsDevice::ActiveAttributes = 0;
const int GraphicsDevice::MaxVertexAttributes = 0;
const int GraphicsDevice::MaxVertexUniformComp = 0;
const int GraphicsDevice::MaxViewportDims = 0;
const int GraphicsDevice::ViewportBox = 0;
const int GraphicsDevice::ScissorBox = 0;
const int GraphicsDevice::Fill = 0;
const int GraphicsDevice::Line = 0;
const int GraphicsDevice::Point = 0;
const int GraphicsDevice::GLPreferredBuffer = 0;
const int GraphicsDevice::GLInternalFormat = 0;

int Depth::OpenGLID(int id) { return 0; }
int CubeMap::OpenGLID(int id) { return 0; }
int Pixel::OpenGLID(int p) { return 0; }

struct NullGraphicsDevice : public GraphicsDevice {
  NullGraphicsDevice(Window *w) : GraphicsDevice(w, 2) {}
  void Init(AssetLoading*, const Box&) {}
  bool GetEnabled(int) { return 0; }
  bool ShaderSupport() const { return 0; }
  void MarkDirty() {}
  void EnableTexture() {}
  void DisableTexture() {}
  void EnableLighting() {}
  void DisableLighting() {}
  void EnableNormals() {}
  void DisableNormals() {}
  void EnableVertexColor() {}
  void DisableVertexColor() {}
  void EnableLight(int n) {}
  void DisableLight(int n) {}
  void DisableCubeMap() {}
  void BindCubeMap(int n) {}
  void TextureGenLinear() {}
  void TextureGenReflection() {}
  void Material(int t, float *color) {}
  void Light(int n, int t, float *color) {}
  void BindTexture(int t, int n) {}
  void ActiveTexture(int n) {}
  bool VertexPointer(int m, int t, int w, int o, float *verts, int l, int *out, bool dirty, int) { return true; }
  void TexPointer(int m, int t, int w, int o, float *tex, int l, int *out, bool dirty) {}
  void ColorPointer(int m, int t, int w, int o, float *verts, int l, int *out, bool dirty) {}
  void NormalPointer(int m, int t, int w, int o, float *verts, int l, int *out, bool dirty) {}
  void Color4f(float r, float g, float b, float a) {}
  void UpdateColor() {}
  void MatrixProjection() {}
  void MatrixModelview() {}
  void LoadIdentity() {}
  void PushMatrix() {}
  void PopMatrix() {}
  void GetMatrix(m44 *out) {}
  void PrintMatrix() {}
  void Scalef(float x, float y, float z) {}
  void Rotatef(float angle, float x, float y, float z) {}
  void Ortho(float l, float r, float b, float t, float nv, float fv) {}
  void Frustum(float l, float r, float b, float t, float nv, float fv) {}
  void Mult(const float *m) {}
  void Translate(float x, float y, float z) {}
  void UseShader(Shader *shader) {}
  void DrawElements(int pt, int np, int it, int o, void *index, int l, int *out, bool dirty) {}
  void DrawArrays(int t, int o, int n) {}
  void DeferDrawArrays(int t, int o, int n) {}

  void Finish() {}
  void Flush() {}
  void Clear() {}
  void ClearDepth() {}
  void ClearColor(const Color &c) {}
  void PointSize(float n) {}
  void LineWidth(float n) {}
  void GenTextures(int t, int n, unsigned *out) {}
  void DelTextures(int n, const unsigned *id) {}
  void TexImage2D(int targ, int l, int fi, int w, int h, int b, int f, int t, const void*) {}
  void TexSubImage2D(int targ, int l, int xo, int yo, int w, int h, int f, int t, const void*) {}
  void CopyTexImage2D(int targ, int l, int fi, int x, int y, int w, int h, int b) {}
  void CopyTexSubImage2D(int targ, int l, int xo, int yo, int x, int y, int w, int h) {}
  void CheckForError(const char *file, int line) {}
  void EnableScissor() {}
  void DisableScissor() {}
  void EnableDepthTest() {}
  void DisableDepthTest() {}
  void DisableBlend() {}
  void EnableBlend() {}
  void BlendMode(int sm, int dm) {}
  void ViewPort(Box w) {}
  void Scissor(Box w) {}
  void TexParameter(int, int, int) {}
  void GenRenderBuffers(int n, unsigned *out) {}
  void DelRenderBuffers(int n, const unsigned *id) {}
  void BindRenderBuffer(int id) {}
  void RenderBufferStorage(int d, int w, int h) {}
  void GenFrameBuffers(int n, unsigned *out) {}
  void DelFrameBuffers(int n, const unsigned *id) {}
  void BindFrameBuffer(int id) {}
  void FrameBufferTexture(int id) {}
  void FrameBufferDepthTexture(int id) {}
  int CheckFrameBufferStatus() { return 0; }
  void Screenshot(Texture *out) {}
  void ScreenshotBox(Texture *out, const Box &b, int flag) {}
  void DumpTexture(Texture *out, unsigned tex_id) {}
  const char *GetString(int t) { return ""; }
  const char *GetGLEWString(int t) { return ""; }

  int CreateProgram() { return 0; }
  void DelProgram(int p) {}
  int CreateShader(int t) { return 0; }
  void CompileShader(int shader, vector<const char*> source) {}
  void AttachShader(int prog, int shader) {}
  void DelShader(int shader) {}
  void BindAttribLocation(int prog, int loc, const string &name) {}
  bool LinkProgram(int prog) { return false; }
  void GetProgramiv(int p, int t, int *out) {}
  void GetIntegerv(int t, int *out) {}
  int GetAttribLocation (int prog, const string &name) { return 0; }
  int GetUniformLocation(int prog, const string &name) { return 0; }
  void Uniform1i(int u, int v) {}
  void Uniform1f(int u, float v) {}
  void Uniform2f(int u, float v1, float v2) {}
  void Uniform3f(int u, float v1, float v2, float v3) {}
  void Uniform4f(int u, float v1, float v2, float v3, float v4) {}
  void Uniform3fv(int u, int n, const float *v) {}
};

unique_ptr<GraphicsDevice> CreateGraphicsDevice(Window *w, Shaders*, int) { return make_unique<NullGraphicsDevice>(w); }

}; // namespace LFL
