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

#include "gd_common.h"

namespace LFL {
void ShaderBasedGraphicsDevice::EnableTexture()      { if (Changed(&texture_on,    true))  { ClearDeferred(); UpdateTexture();    } GDDebug("Texture=1"); }
void ShaderBasedGraphicsDevice::DisableTexture()     { if (Changed(&texture_on,    false)) { ClearDeferred(); UpdateTexture();    } GDDebug("Texture=0"); }
void ShaderBasedGraphicsDevice::EnableLighting()     { lighting_on=1; GDDebug("Lighting=1"); }
void ShaderBasedGraphicsDevice::DisableLighting()    { lighting_on=0; GDDebug("Lighting=0"); }
void ShaderBasedGraphicsDevice::EnableNormals()      { if (Changed(&normals_on,    true))  { UpdateShader();  UpdateNormals();    } GDDebug("Normals=1"); }
void ShaderBasedGraphicsDevice::DisableNormals()     { if (Changed(&normals_on,    false)) { UpdateShader();  UpdateNormals();    } GDDebug("Normals=0"); }
void ShaderBasedGraphicsDevice::EnableVertexColor()  { if (Changed(&colorverts_on, true))  { ClearDeferred(); UpdateColorVerts(); } GDDebug("VertexColor=1"); }
void ShaderBasedGraphicsDevice::DisableVertexColor() { if (Changed(&colorverts_on, false)) { ClearDeferred(); UpdateColorVerts(); } GDDebug("VertexColor=0"); }
void ShaderBasedGraphicsDevice::EnableLight(int n) {}
void ShaderBasedGraphicsDevice::DisableLight(int n) {}
void ShaderBasedGraphicsDevice::DisableCubeMap()     { if (Changed(&cubemap_on,    false)) { UpdateShader(); }                      GDDebug("CubeMap=", 0); }

void ShaderBasedGraphicsDevice::MatrixProjection() { matrix_target=2; }
void ShaderBasedGraphicsDevice::MatrixModelview()  { matrix_target=1; }
void ShaderBasedGraphicsDevice::LoadIdentity()     { TargetMatrix()->back().Assign(m44::Identity());    UpdateMatrix(); }
void ShaderBasedGraphicsDevice::PushMatrix()       { TargetMatrix()->push_back(TargetMatrix()->back()); UpdateMatrix(); }
void ShaderBasedGraphicsDevice::PopMatrix() {
  vector<m44> *target = TargetMatrix();
  if      (target->size() >= 1) target->pop_back();
  else if (target->size() == 1) target->back().Assign(m44::Identity());
  UpdateMatrix();
}

void ShaderBasedGraphicsDevice::GetMatrix(m44 *out)  { *out = TargetMatrix()->back(); }
void ShaderBasedGraphicsDevice::PrintMatrix()        { TargetMatrix()->back().Print(StrCat("mt", matrix_target)); }

void ShaderBasedGraphicsDevice::Scalef(float x, float y, float z) {
  m44 &m = TargetMatrix()->back();
  m[0].x *= x; m[0].y *= x; m[0].z *= x;
  m[1].x *= y; m[1].y *= y; m[1].z *= y;
  m[2].x *= z; m[2].y *= z; m[2].z *= z;
  UpdateMatrix();
}

void ShaderBasedGraphicsDevice::Rotatef(float angle, float x, float y, float z) { TargetMatrix()->back().Mult(m44::Rotate(DegreeToRadian(angle), x, y, z)); UpdateMatrix(); }
void ShaderBasedGraphicsDevice::Ortho  (float l, float r, float b, float t, float nv, float fv) { TargetMatrix()->back().Mult(m44::Ortho  (l, r, b, t, nv, fv)); UpdateMatrix(); }
void ShaderBasedGraphicsDevice::Frustum(float l, float r, float b, float t, float nv, float fv) { TargetMatrix()->back().Mult(m44::Frustum(l, r, b, t, nv, fv)); UpdateMatrix(); }
void ShaderBasedGraphicsDevice::Mult(const float *m) { TargetMatrix()->back().Mult(m44(m));               UpdateMatrix(); }

void ShaderBasedGraphicsDevice::Translate(float x, float y, float z) { 
  m44 &m = TargetMatrix()->back();
  m[3].x += m[0].x * x + m[1].x * y + m[2].x * z;
  m[3].y += m[0].y * x + m[1].y * y + m[2].y * z;
  m[3].z += m[0].z * x + m[1].z * y + m[2].z * z;
  m[3].w += m[0].w * x + m[1].w * y + m[2].w * z;
  UpdateMatrix();
}

void ShaderBasedGraphicsDevice::Color4f(float r, float g, float b, float a) {
  if (lighting_on) {
    float color[] = { r, g, b, a };
    Material(c.AmbientAndDiffuse, color);
  } else if (Changed(&default_color.back(), Color(r,g,b,a))) UpdateColor();
}

void ShaderBasedGraphicsDevice::Light(int n, int t, float *v) {
  bool light_pos = 0, light_color = 0;
  if (n != 0) return ERROR("ignoring Light(", n, ")");

  if      (t == c.Position) { light_pos=1;   light[n].pos = modelview_matrix.back().Transform(v4(v)); }
  else if (t == c.Ambient)  { light_color=1; light[n].color.ambient  = Color(v); }
  else if (t == c.Diffuse)  { light_color=1; light[n].color.diffuse  = Color(v); }
  else if (t == c.Specular) { light_color=1; light[n].color.specular = Color(v); }

  if (light_pos)   { shader->dirty_light_pos  [n] = shaders->shader_cubenorm.dirty_light_pos  [n] = shaders->shader_normals.dirty_light_pos  [n] = 1; }
  if (light_color) { shader->dirty_light_color[n] = shaders->shader_cubenorm.dirty_light_color[n] = shaders->shader_normals.dirty_light_color[n] = 1; }
}

void ShaderBasedGraphicsDevice::Material(int t, float *v) {
  if      (t == c.Ambient)           material.ambient  = Color(v);
  else if (t == c.Diffuse)           material.diffuse  = Color(v);
  else if (t == c.Specular)          material.specular = Color(v);
  else if (t == c.Emission)          material.emissive = Color(v);
  else if (t == c.AmbientAndDiffuse) material.ambient = material.diffuse = Color(v);
  UpdateMaterial();
}

void ShaderBasedGraphicsDevice::SetDontClearDeferred(bool v) { dont_clear_deferred = v; GDDebug("SetDontClearDeferred = ", v); }

int ShaderBasedGraphicsDevice::AddDeferredVertexSpace(int l) {
  if (l + deferred.vertexbuffer_len > deferred.vertexbuffer_size) ClearDeferred();
  int ret = deferred.vertexbuffer_len;
  deferred.vertexbuffer_len += l;
  CHECK_LE(deferred.vertexbuffer_len, deferred.vertexbuffer_size);
  return ret;
}

vector<m44> *ShaderBasedGraphicsDevice::TargetMatrix() {
  if      (matrix_target == 1) return &modelview_matrix;
  else if (matrix_target == 2) return &projection_matrix;
  else FATAL("uknown matrix ", matrix_target);
}

void ShaderBasedGraphicsDevice::UpdateShader() {
  if (cubemap_on && normals_on) UseShader(&shaders->shader_cubenorm);
  else if          (cubemap_on) UseShader(&shaders->shader_cubemap);
  else if          (normals_on) UseShader(&shaders->shader_normals);
  else                          UseShader(&shaders->shader_default);
}

void ShaderBasedGraphicsDevice::UpdateMatrix() { ClearDeferred(); dirty_matrix = true; GDDebug("UpdateMatrix"); }
void ShaderBasedGraphicsDevice::UpdateColor()  { ClearDeferred(); dirty_color = true;  GDDebug("UpdateColor"); }
void ShaderBasedGraphicsDevice::UpdateMaterial() {
  ClearDeferred();
  shader->dirty_material = shaders->shader_cubenorm.dirty_material = shaders->shader_normals.dirty_material = true;
  GDDebug("UpdateMaterial");
}

void ShaderBasedGraphicsDevice::PushDirtyState() {
  if (dirty_matrix) {
    dirty_matrix = false;
    m44 m = projection_matrix.back();
    m.Mult(modelview_matrix.back());
    if (1)                  UniformMatrix4fv(shader->uniform_modelviewproj, 1, m[0]);
    if (1)                  UniformMatrix4fv(shader->uniform_modelview,     1, modelview_matrix.back()[0]);
    if (1)                  Uniform3fv      (shader->uniform_campos,        1, camera_pos);
    if (invert_view_matrix) UniformMatrix4fv(shader->uniform_invview,       1, invview_matrix[0]);
    if (track_model_matrix) UniformMatrix4fv(shader->uniform_model,         1, model_matrix[0]);
  }
  if (dirty_color && shader->uniform_colordefault >= 0) {
    dirty_color = false;
    Uniform4fv(shader->uniform_colordefault, 1, default_color.back().x);
  }
  if (shader->dirty_material) {
    Uniform4fv(shader->uniform_material_ambient,  1, material.ambient.x);
    Uniform4fv(shader->uniform_material_diffuse,  1, material.diffuse.x);
    Uniform4fv(shader->uniform_material_specular, 1, material.specular.x);
    Uniform4fv(shader->uniform_material_emission, 1, material.emissive.x);
  }
  for (int i=0; i<sizeofarray(light) && i<sizeofarray(shader->dirty_light_pos); i++) {
    if (shader->dirty_light_pos[i]) {
      shader->dirty_light_pos[i] = 0;
      Uniform4fv(shader->uniform_light0_pos, 1, light[i].pos);
    }
    if (shader->dirty_light_color[i]) {
      shader->dirty_light_color[i] = 0;
      Uniform4fv(shader->uniform_light0_ambient,  1, light[i].color.ambient.x);
      Uniform4fv(shader->uniform_light0_diffuse,  1, light[i].color.diffuse.x);
      Uniform4fv(shader->uniform_light0_specular, 1, light[i].color.specular.x);
    }
  }
}

}; // namespace LFL
