attribute vec4 Position;
varying vec3 eye, dir;

uniform mat4 Modelview;
uniform mat4 ModelviewProjection;
uniform float fov_x, fov_y;

float fov2scale(float fov) { return tan(fov/2.0); }

// Draw an untransformed rectangle covering the whole screen.
// Get camera position and interpolated directions from the modelview matrix.
void main() {
  gl_Position = Position;
  eye = vec3(Modelview[3]);
  dir = vec3(Modelview * vec4(
    fov2scale(fov_x)*Position.x, fov2scale(fov_y)*Position.y, 1, 0) );
}
